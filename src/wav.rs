use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::WavError;

const WAV_HEADER_LEN: u64 = 44;
const RIFF_DATA_OVERHEAD: u64 = 36;
const SAMPLES_PER_WRITE_CHUNK: usize = 4_096;
const BYTES_PER_WRITE_CHUNK: usize = SAMPLES_PER_WRITE_CHUNK * 2;

/// Seekable PCM16 WAV writer that owns and finalizes its output.
///
/// Call [`WavWriter::finalize`] to observe any final seek/write error.  The
/// `Drop` implementation also makes a best effort to repair the header, so an
/// early `?` does not normally leave a zero-length RIFF header behind.
#[derive(Debug)]
pub struct WavWriter<W: Write + Seek> {
    inner: Option<W>,
    channels: u16,
    sample_rate: u32,
    data_bytes: u64,
    finalized: bool,
    sample_bytes: [u8; BYTES_PER_WRITE_CHUNK],
}

impl WavWriter<File> {
    pub fn create(
        path: impl AsRef<Path>,
        channels: u16,
        sample_rate: u32,
    ) -> Result<Self, WavError> {
        let file = File::create(path)?;
        Self::new(file, channels, sample_rate)
    }
}

impl<W: Write + Seek> WavWriter<W> {
    pub fn new(inner: W, channels: u16, sample_rate: u32) -> Result<Self, WavError> {
        if channels == 0 {
            return Err(WavError::InvalidChannels(channels));
        }
        if sample_rate == 0 {
            return Err(WavError::InvalidSampleRate(sample_rate));
        }
        let block_align = channels.checked_mul(2).ok_or(WavError::SizeOverflow)?;
        sample_rate
            .checked_mul(u32::from(block_align))
            .ok_or(WavError::SizeOverflow)?;

        let mut writer = Self {
            inner: Some(inner),
            channels,
            sample_rate,
            data_bytes: 0,
            finalized: false,
            sample_bytes: [0; BYTES_PER_WRITE_CHUNK],
        };
        writer.write_header(0)?;
        Ok(writer)
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn data_bytes(&self) -> u64 {
        self.data_bytes
    }

    pub fn write_samples(&mut self, samples: &[i16]) -> Result<(), WavError> {
        if !samples.len().is_multiple_of(usize::from(self.channels)) {
            return Err(WavError::InvalidSampleCount {
                channels: usize::from(self.channels),
                samples: samples.len(),
            });
        }
        let added = u64::try_from(samples.len())
            .map_err(|_| WavError::SizeOverflow)?
            .checked_mul(2)
            .ok_or(WavError::SizeOverflow)?;
        let new_size = self
            .data_bytes
            .checked_add(added)
            .ok_or(WavError::SizeOverflow)?;
        checked_riff_size(new_size)?;

        let inner = self.inner.as_mut().ok_or(WavError::NotFinalized)?;
        for chunk in samples.chunks(SAMPLES_PER_WRITE_CHUNK) {
            for (sample, bytes) in chunk
                .iter()
                .zip(self.sample_bytes[..chunk.len() * 2].chunks_exact_mut(2))
            {
                bytes.copy_from_slice(&sample.to_le_bytes());
            }
            inner.write_all(&self.sample_bytes[..chunk.len() * 2])?;
        }
        self.data_bytes = new_size;
        self.finalized = false;
        Ok(())
    }

    /// Finalize sizes, flush, and return the owned writer.
    pub fn finalize(mut self) -> Result<W, WavError> {
        self.finalize_inner()?;
        self.inner.take().ok_or(WavError::NotFinalized)
    }

    fn write_header(&mut self, data_bytes: u64) -> Result<(), WavError> {
        let data_size = u32::try_from(data_bytes).map_err(|_| WavError::SizeOverflow)?;
        let riff_size = checked_riff_size(data_bytes)?;
        let block_align = self.channels.checked_mul(2).ok_or(WavError::SizeOverflow)?;
        let byte_rate = self
            .sample_rate
            .checked_mul(u32::from(block_align))
            .ok_or(WavError::SizeOverflow)?;

        let mut header = [0_u8; WAV_HEADER_LEN as usize];
        header[0..4].copy_from_slice(b"RIFF");
        header[4..8].copy_from_slice(&riff_size.to_le_bytes());
        header[8..12].copy_from_slice(b"WAVE");
        header[12..16].copy_from_slice(b"fmt ");
        header[16..20].copy_from_slice(&16_u32.to_le_bytes());
        header[20..22].copy_from_slice(&1_u16.to_le_bytes());
        header[22..24].copy_from_slice(&self.channels.to_le_bytes());
        header[24..28].copy_from_slice(&self.sample_rate.to_le_bytes());
        header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
        header[32..34].copy_from_slice(&block_align.to_le_bytes());
        header[34..36].copy_from_slice(&16_u16.to_le_bytes());
        header[36..40].copy_from_slice(b"data");
        header[40..44].copy_from_slice(&data_size.to_le_bytes());

        let inner = self.inner.as_mut().ok_or(WavError::NotFinalized)?;
        inner.seek(SeekFrom::Start(0))?;
        inner.write_all(&header)?;
        inner.seek(SeekFrom::Start(WAV_HEADER_LEN + data_bytes))?;
        Ok(())
    }

    fn finalize_inner(&mut self) -> Result<(), WavError> {
        if self.finalized || self.inner.is_none() {
            return Ok(());
        }
        self.write_header(self.data_bytes)?;
        self.inner.as_mut().ok_or(WavError::NotFinalized)?.flush()?;
        self.finalized = true;
        Ok(())
    }
}

impl<W: Write + Seek> Drop for WavWriter<W> {
    fn drop(&mut self) {
        let _ = self.finalize_inner();
    }
}

fn checked_riff_size(data_bytes: u64) -> Result<u32, WavError> {
    let size = RIFF_DATA_OVERHEAD
        .checked_add(data_bytes)
        .ok_or(WavError::SizeOverflow)?;
    u32::try_from(size).map_err(|_| WavError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn writes_portable_little_endian_pcm_header() {
        let cursor = Cursor::new(Vec::new());
        let mut wav = WavWriter::new(cursor, 2, 48_000).unwrap();
        wav.write_samples(&[-32_768, -1, 0, 32_767]).unwrap();
        let cursor = wav.finalize().unwrap();
        let bytes = cursor.into_inner();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 44);
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            48_000
        );
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 8);
        assert_eq!(
            &bytes[44..],
            &[0x00, 0x80, 0xff, 0xff, 0x00, 0x00, 0xff, 0x7f]
        );
    }

    #[test]
    fn rejects_partial_interleaved_frame() {
        let cursor = Cursor::new(Vec::new());
        let mut wav = WavWriter::new(cursor, 2, 48_000).unwrap();
        assert!(matches!(
            wav.write_samples(&[1, 2, 3]),
            Err(WavError::InvalidSampleCount {
                channels: 2,
                samples: 3
            })
        ));
    }

    #[test]
    fn drop_repairs_header() {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut wav = WavWriter::new(&mut cursor, 1, 44_100).unwrap();
            wav.write_samples(&[1, 2, 3]).unwrap();
        }
        let bytes = cursor.into_inner();
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 6);
    }
}
