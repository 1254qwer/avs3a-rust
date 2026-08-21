use crate::crc16;
pub use crate::error::StreamError;

use crate::error::HeaderError;
use crate::header::{FrameHeader, MAX_HEADER_BYTES, MAX_PAYLOAD_BYTES, parse_header_at};

const DEFAULT_BUFFER_LIMIT: usize = MAX_HEADER_BYTES + MAX_PAYLOAD_BYTES;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedFrame {
    header: FrameHeader,
    bytes: Vec<u8>,
    actual_crc: u16,
}

impl EncodedFrame {
    pub(crate) fn new(header: FrameHeader, bytes: Vec<u8>) -> Self {
        let actual_crc = crc16(&bytes[header.header_len..]);
        Self {
            header,
            bytes,
            actual_crc,
        }
    }

    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn header_bytes(&self) -> &[u8] {
        &self.bytes[..self.header.header_len]
    }

    pub fn payload(&self) -> &[u8] {
        &self.bytes[self.header.header_len..]
    }

    pub fn expected_crc(&self) -> u16 {
        self.header.crc
    }

    pub fn actual_crc(&self) -> u16 {
        self.actual_crc
    }

    pub fn crc_is_valid(&self) -> bool {
        self.header.crc == self.actual_crc
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// Bytes skipped while looking for a syntactically valid sync header.
    Skipped {
        bytes: usize,
    },
    Frame(EncodedFrame),
}

/// Incremental, allocation-bounded AV3A elementary-stream parser.
///
/// Arbitrary input chunking is supported. A frame is emitted only after its
/// complete payload is present; all indexing is therefore bounded by the
/// parsed frame size. CRC status is retained on [`EncodedFrame`] for callers
/// to inspect, while [`crate::decoder::Decoder`] enforces it before synthesis.
#[derive(Debug)]
pub struct FrameStream {
    buffer: Vec<u8>,
    max_buffer: usize,
}

impl Default for FrameStream {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameStream {
    pub fn new() -> Self {
        Self::with_buffer_limit(DEFAULT_BUFFER_LIMIT)
    }

    pub fn with_buffer_limit(max_buffer: usize) -> Self {
        Self {
            buffer: Vec::new(),
            max_buffer,
        }
    }

    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    pub fn push(&mut self, input: &[u8]) -> Result<Vec<StreamEvent>, StreamError> {
        let mut events = Vec::new();
        let mut input_offset = 0;

        while input_offset < input.len() {
            self.process(&mut events)?;
            let free = self.max_buffer.saturating_sub(self.buffer.len());
            if free == 0 {
                return Err(StreamError::BufferLimit {
                    limit: self.max_buffer,
                });
            }
            let take = free.min(input.len() - input_offset);
            self.buffer
                .extend_from_slice(&input[input_offset..input_offset + take]);
            input_offset += take;
        }
        self.process(&mut events)?;
        Ok(events)
    }

    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, StreamError> {
        let mut events = Vec::new();
        self.process(&mut events)?;
        if self.buffer.is_empty() {
            Ok(events)
        } else {
            Err(StreamError::TrailingData {
                bytes: self.buffer.len(),
            })
        }
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    fn process(&mut self, events: &mut Vec<StreamEvent>) -> Result<(), StreamError> {
        let mut skipped = 0_usize;
        loop {
            if self.buffer.len() < 2 {
                break;
            }

            let Some(offset) = find_sync(&self.buffer) else {
                let keep = usize::from(self.buffer.last() == Some(&0xff));
                let discard = self.buffer.len() - keep;
                self.buffer.drain(..discard);
                skipped = skipped.saturating_add(discard);
                break;
            };
            if offset != 0 {
                self.buffer.drain(..offset);
                skipped = skipped.saturating_add(offset);
            }

            let header = match parse_header_at(&self.buffer, 0) {
                Ok(header) => header,
                Err(HeaderError::NeedMoreData { .. }) => break,
                Err(_) => {
                    self.buffer.drain(..1);
                    skipped = skipped.saturating_add(1);
                    continue;
                }
            };
            if header.frame_len > self.max_buffer {
                return Err(StreamError::BufferLimit {
                    limit: self.max_buffer,
                });
            }
            if self.buffer.len() < header.frame_len {
                break;
            }

            flush_skipped(events, &mut skipped);
            let bytes: Vec<u8> = self.buffer.drain(..header.frame_len).collect();
            events.push(StreamEvent::Frame(EncodedFrame::new(header, bytes)));
        }
        flush_skipped(events, &mut skipped);
        Ok(())
    }
}

fn find_sync(input: &[u8]) -> Option<usize> {
    input
        .windows(2)
        .position(|window| window[0] == 0xff && window[1] & 0xf0 == 0xf0)
}

fn flush_skipped(events: &mut Vec<StreamEvent>, skipped: &mut usize) {
    if *skipped == 0 {
        return;
    }
    if let Some(StreamEvent::Skipped { bytes }) = events.last_mut() {
        *bytes = bytes.saturating_add(*skipped);
    } else {
        events.push(StreamEvent::Skipped { bytes: *skipped });
    }
    *skipped = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;
    use crate::header::ChannelConfig;

    fn make_frame(payload_byte: u8) -> Vec<u8> {
        // 64 kbps mono at 48 kHz: floor(64000 * 1024 / 48000) - 56
        // header bits, rounded up to bytes.
        let payload_len = ((64_000_usize * 1_024 / 48_000) - 56).div_ceil(8);
        let payload = vec![payload_byte; payload_len];
        let crc = crc16(&payload);
        let mut writer = BitWriter::new();
        writer.write_bits(0xfff, 12).unwrap();
        writer.write_bits(2, 4).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 3).unwrap();
        writer.write_bits(0, 3).unwrap();
        writer.write_bits(2, 4).unwrap();
        writer.write_bits(u64::from(crc >> 8), 8).unwrap();
        writer
            .write_bits(ChannelConfig::Mono.index().into(), 7)
            .unwrap();
        writer.write_bits(1, 2).unwrap();
        writer.write_bits(4, 4).unwrap();
        writer.write_bits(u64::from(crc & 0xff), 8).unwrap();
        let mut frame = writer.into_bytes();
        frame.extend_from_slice(&payload);
        frame
    }

    fn frames(events: &[StreamEvent]) -> impl Iterator<Item = &EncodedFrame> {
        events.iter().filter_map(|event| match event {
            StreamEvent::Frame(frame) => Some(frame),
            StreamEvent::Skipped { .. } => None,
        })
    }

    #[test]
    fn accepts_every_input_split() {
        let frame = make_frame(0x5a);
        for split in 0..=frame.len() {
            let mut stream = FrameStream::new();
            let mut events = stream.push(&frame[..split]).unwrap();
            events.extend(stream.push(&frame[split..]).unwrap());
            assert_eq!(frames(&events).count(), 1, "split at {split}");
            assert!(frames(&events).next().unwrap().crc_is_valid());
            assert_eq!(stream.buffered_len(), 0);
        }
    }

    #[test]
    fn emits_multiple_frames_and_reports_leading_garbage() {
        let first = make_frame(1);
        let second = make_frame(2);
        let mut input = vec![0x00, 0xff, 0xf1, 0x00, 0x12];
        input.extend_from_slice(&first);
        input.extend_from_slice(&second);
        let mut stream = FrameStream::new();
        let events = stream.push(&input).unwrap();
        assert_eq!(frames(&events).count(), 2);
        assert!(matches!(
            events.first(),
            Some(StreamEvent::Skipped { bytes: 5 })
        ));
    }

    #[test]
    fn retains_crc_failure_for_decoder() {
        let mut frame = make_frame(0x11);
        let last = frame.len() - 1;
        frame[last] ^= 1;
        let mut stream = FrameStream::new();
        let events = stream.push(&frame).unwrap();
        assert!(!frames(&events).next().unwrap().crc_is_valid());
    }

    #[test]
    fn finish_rejects_truncated_frame() {
        let frame = make_frame(0);
        let mut stream = FrameStream::new();
        stream.push(&frame[..frame.len() - 1]).unwrap();
        assert!(matches!(
            stream.finish(),
            Err(StreamError::TrailingData { .. })
        ));
    }
}
