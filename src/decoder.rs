use crate::error::DecodeError;
use crate::header::{BitDepth, ChannelConfig, CodecProfile, FrameHeader, NnType, SoundBedType};
use crate::stream::EncodedFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderConfig {
    pub sample_rate: u32,
    pub bitrate: u32,
    pub channels: u8,
    pub samples_per_channel: u32,
    pub bit_depth: BitDepth,
    pub profile: CodecProfile,
    pub nn_type: NnType,
    pub channel_config: Option<ChannelConfig>,
    pub sound_bed_type: Option<SoundBedType>,
    pub hoa_order: Option<u8>,
    pub objects: u8,
    pub bed_channels: u8,
    pub has_lfe: bool,
    pub bed_bitrate: Option<u32>,
    pub object_bitrate: Option<u32>,
}

impl DecoderConfig {
    pub fn from_header(header: &FrameHeader) -> Self {
        Self {
            sample_rate: header.sample_rate,
            bitrate: header.bitrate,
            channels: header.channels,
            samples_per_channel: header.samples_per_channel,
            bit_depth: header.bit_depth,
            profile: header.profile,
            nn_type: header.nn_type,
            channel_config: header.channel_config,
            sound_bed_type: header.sound_bed_type,
            hoa_order: header.hoa_order,
            objects: header.objects,
            bed_channels: header.bed_channels,
            has_lfe: header.has_lfe,
            bed_bitrate: header.bed_bitrate,
            object_bitrate: header.object_bitrate,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFrame {
    config: DecoderConfig,
    samples: Vec<i16>,
}

impl AudioFrame {
    pub fn config(&self) -> DecoderConfig {
        self.config
    }

    pub fn samples(&self) -> &[i16] {
        &self.samples
    }

    pub fn into_samples(self) -> Vec<i16> {
        self.samples
    }
}

/// Boundary between checked framing/state management and the DSP port.
///
/// Implementations receive an output slice with exactly
/// `channels * samples_per_channel` samples. They cannot change its length,
/// which prevents a backend from silently violating the public frame shape.
/// The payload still contains the frame-level metadata prefix; built-in
/// backends parse it before passing the remaining audio bits to their cores.
pub trait DecoderBackend {
    fn configure(&mut self, config: DecoderConfig) -> Result<(), DecodeError>;

    fn decode_frame(
        &mut self,
        header: &FrameHeader,
        payload: &[u8],
        output: &mut [i16],
    ) -> Result<(), DecodeError>;
}

#[derive(Debug)]
pub struct PendingDecoder<B> {
    backend: B,
}

impl<B: DecoderBackend> PendingDecoder<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn configure(mut self, header: &FrameHeader) -> Result<Decoder<B>, DecodeError> {
        let config = DecoderConfig::from_header(header);
        self.backend.configure(config)?;
        Ok(Decoder {
            backend: self.backend,
            config,
            frame_index: 0,
        })
    }
}

#[derive(Debug)]
pub struct Decoder<B> {
    backend: B,
    config: DecoderConfig,
    frame_index: u64,
}

impl<B: DecoderBackend> Decoder<B> {
    pub fn config(&self) -> DecoderConfig {
        self.config
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    pub fn into_backend(self) -> B {
        self.backend
    }

    pub fn decode(&mut self, frame: &EncodedFrame) -> Result<AudioFrame, DecodeError> {
        let mut samples = vec![0_i16; self.sample_count()?];
        self.decode_into(frame, &mut samples)?;
        Ok(AudioFrame {
            config: self.config,
            samples,
        })
    }

    /// Decode into caller-owned interleaved PCM storage.
    ///
    /// This is the allocation-free streaming path. The output length must be
    /// exactly `channels * samples_per_channel`, and decoder state advances
    /// only after CRC, configuration and backend decoding all succeed.
    pub fn decode_into(
        &mut self,
        frame: &EncodedFrame,
        samples: &mut [i16],
    ) -> Result<(), DecodeError> {
        if !frame.crc_is_valid() {
            return Err(DecodeError::CrcMismatch {
                expected: frame.expected_crc(),
                actual: frame.actual_crc(),
            });
        }
        let frame_config = DecoderConfig::from_header(frame.header());
        if frame_config != self.config {
            return Err(DecodeError::ConfigurationChanged);
        }
        let sample_count = self.sample_count()?;
        if samples.len() != sample_count {
            return Err(DecodeError::SampleCount {
                expected: sample_count,
                actual: samples.len(),
            });
        }
        self.backend
            .decode_frame(frame.header(), frame.payload(), samples)?;
        self.frame_index = self.frame_index.saturating_add(1);
        Ok(())
    }

    fn sample_count(&self) -> Result<usize, DecodeError> {
        usize::from(self.config.channels)
            .checked_mul(
                usize::try_from(self.config.samples_per_channel).map_err(|_| {
                    DecodeError::SampleCount {
                        expected: 0,
                        actual: 0,
                    }
                })?,
            )
            .ok_or(DecodeError::SampleCount {
                expected: usize::MAX,
                actual: 0,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BitWriter, ChannelConfig, FrameStream, StreamEvent, crc16};

    #[derive(Debug, Default)]
    struct TestBackend {
        configured: bool,
    }

    impl DecoderBackend for TestBackend {
        fn configure(&mut self, _config: DecoderConfig) -> Result<(), DecodeError> {
            self.configured = true;
            Ok(())
        }

        fn decode_frame(
            &mut self,
            _header: &FrameHeader,
            payload: &[u8],
            output: &mut [i16],
        ) -> Result<(), DecodeError> {
            let value = i16::from(payload[0]);
            output.fill(value);
            Ok(())
        }
    }

    fn frame() -> EncodedFrame {
        let payload_len = ((64_000_usize * 1_024 / 48_000) - 56).div_ceil(8);
        let payload = vec![7; payload_len];
        let crc = crc16(&payload);
        let mut writer = BitWriter::new();
        for (value, width) in [
            (0xfff, 12),
            (2, 4),
            (0, 1),
            (0, 3),
            (0, 3),
            (2, 4),
            (u64::from(crc >> 8), 8),
            (ChannelConfig::Mono.index().into(), 7),
            (1, 2),
            (4, 4),
            (u64::from(crc & 0xff), 8),
        ] {
            writer.write_bits(value, width).unwrap();
        }
        let mut bytes = writer.into_bytes();
        bytes.extend_from_slice(&payload);
        let mut stream = FrameStream::new();
        stream
            .push(&bytes)
            .unwrap()
            .into_iter()
            .find_map(|event| match event {
                StreamEvent::Frame(frame) => Some(frame),
                StreamEvent::Skipped { .. } => None,
            })
            .unwrap()
    }

    #[test]
    fn decoder_configures_before_decoding() {
        let encoded = frame();
        let mut decoder = PendingDecoder::new(TestBackend::default())
            .configure(encoded.header())
            .unwrap();
        assert!(decoder.backend().configured);
        let audio = decoder.decode(&encoded).unwrap();
        assert_eq!(audio.samples().len(), 1_024);
        assert!(audio.samples().iter().all(|sample| *sample == 7));
        assert_eq!(decoder.frame_index(), 1);
    }

    #[test]
    fn decoder_rejects_bad_crc_before_backend() {
        let encoded = frame();
        let header = *encoded.header();
        let mut bytes = encoded.bytes().to_vec();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        let encoded = EncodedFrame::new(header, bytes);
        let mut decoder = PendingDecoder::new(TestBackend::default())
            .configure(encoded.header())
            .unwrap();
        assert!(matches!(
            decoder.decode(&encoded),
            Err(DecodeError::CrcMismatch { .. })
        ));
        assert_eq!(decoder.frame_index(), 0);
    }

    #[test]
    fn decoder_treats_bitrate_as_temporal_configuration() {
        let encoded = frame();
        let mut decoder = PendingDecoder::new(TestBackend::default())
            .configure(encoded.header())
            .unwrap();
        let mut changed_header = *encoded.header();
        changed_header.bitrate = 72_000;
        let changed = EncodedFrame::new(changed_header, encoded.bytes().to_vec());
        assert!(changed.crc_is_valid());
        assert!(matches!(
            decoder.decode(&changed),
            Err(DecodeError::ConfigurationChanged)
        ));
        assert_eq!(decoder.frame_index(), 0);
    }

    #[test]
    fn decode_into_reuses_caller_storage() {
        let encoded = frame();
        let mut decoder = PendingDecoder::new(TestBackend::default())
            .configure(encoded.header())
            .unwrap();
        let mut samples = [0_i16; 1_024];
        decoder.decode_into(&encoded, &mut samples).unwrap();
        assert!(samples.iter().all(|&sample| sample == 7));
        assert_eq!(decoder.frame_index(), 1);

        let mut wrong = [9_i16; 17];
        assert!(matches!(
            decoder.decode_into(&encoded, &mut wrong),
            Err(DecodeError::SampleCount {
                expected: 1_024,
                actual: 17
            })
        ));
        assert_eq!(wrong, [9; 17]);
        assert_eq!(decoder.frame_index(), 1);
    }
}
