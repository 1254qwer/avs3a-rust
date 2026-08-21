use crate::error::DecodeError;
use crate::header::{BitDepth, ChannelConfig, CodecProfile, FrameHeader, NnType, SoundBedType};
use crate::mono_backend::float_to_pcm16;
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

/// Full-scale magnitude of the decoder's float output.
///
/// Synthesis runs in PCM16 units rather than `-1.0..=1.0`, because the
/// reference decoder's rounding and saturation rule is defined against integer
/// full scale. Divide by this to reach the normalised range most audio APIs
/// expect.
pub const FLOAT_FULL_SCALE: f32 = 32_768.0;

/// Boundary between checked framing/state management and the DSP port.
///
/// Implementations receive an output slice with exactly
/// `channels * samples_per_channel` samples. They cannot change its length,
/// which prevents a backend from silently violating the public frame shape.
/// The payload still contains the frame-level metadata prefix; built-in
/// backends parse it before passing the remaining audio bits to their cores.
///
/// Output is floats in PCM16 units (see [`FLOAT_FULL_SCALE`]) because that is
/// what every synthesis pipeline natively produces. Quantisation belongs to
/// [`Decoder`], so a renderer that wants to downmix or apply gain never has to
/// undo a rounding step the decoder had no reason to take.
pub trait DecoderBackend {
    fn configure(&mut self, config: DecoderConfig) -> Result<(), DecodeError>;

    fn decode_frame(
        &mut self,
        header: &FrameHeader,
        payload: &[u8],
        output: &mut [f32],
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
            float_scratch: Vec::new(),
            last_clipped_samples: 0,
            total_clipped_samples: 0,
        })
    }
}

#[derive(Debug)]
pub struct Decoder<B> {
    backend: B,
    config: DecoderConfig,
    frame_index: u64,
    /// Float staging for the PCM16 path, allocated on first use so callers
    /// that consume floats never pay for it.
    float_scratch: Vec<f32>,
    last_clipped_samples: usize,
    total_clipped_samples: u64,
}

impl<B: DecoderBackend> Decoder<B> {
    pub fn config(&self) -> DecoderConfig {
        self.config
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Interleaved sample count of one decoded frame.
    pub fn sample_count(&self) -> Result<usize, DecodeError> {
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

    /// Samples clamped to full scale by the most recent PCM16 conversion.
    pub fn last_clipped_samples(&self) -> usize {
        self.last_clipped_samples
    }

    /// Samples clamped to full scale since the decoder was configured or reset.
    pub fn total_clipped_samples(&self) -> u64 {
        self.total_clipped_samples
    }

    /// Reset temporal decoder state while keeping the validated stream
    /// configuration. The next frame is treated as frame zero.
    pub fn reset(&mut self) -> Result<(), DecodeError> {
        self.backend.configure(self.config)?;
        self.frame_index = 0;
        self.last_clipped_samples = 0;
        self.total_clipped_samples = 0;
        Ok(())
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

    /// Decode into caller-owned interleaved PCM16 storage.
    ///
    /// The output length must be exactly `channels * samples_per_channel`, and
    /// decoder state advances only after CRC, configuration and backend
    /// decoding all succeed.
    pub fn decode_into(
        &mut self,
        frame: &EncodedFrame,
        samples: &mut [i16],
    ) -> Result<(), DecodeError> {
        let sample_count = self.prepare(frame, samples.len())?;
        if self.float_scratch.len() != sample_count {
            self.float_scratch.clear();
            self.float_scratch.resize(sample_count, 0.0);
        }
        self.backend
            .decode_frame(frame.header(), frame.payload(), &mut self.float_scratch)?;
        let clipped = float_to_pcm16(&self.float_scratch, samples);
        self.last_clipped_samples = clipped;
        self.total_clipped_samples = self
            .total_clipped_samples
            .saturating_add(u64::try_from(clipped).unwrap_or(u64::MAX));
        self.frame_index = self.frame_index.saturating_add(1);
        Ok(())
    }

    /// Decode into caller-owned interleaved float storage.
    ///
    /// This is the backend's native output, so nothing is copied and nothing is
    /// quantised. Samples are in PCM16 units, not `-1.0..=1.0`: divide by
    /// [`FLOAT_FULL_SCALE`] for the normalised range most audio APIs expect.
    /// Values may exceed full scale, because the bitstream can encode an
    /// overshoot that only clips once [`Self::decode_into`] quantises it.
    pub fn decode_into_f32(
        &mut self,
        frame: &EncodedFrame,
        samples: &mut [f32],
    ) -> Result<(), DecodeError> {
        self.prepare(frame, samples.len())?;
        self.backend
            .decode_frame(frame.header(), frame.payload(), samples)?;
        self.frame_index = self.frame_index.saturating_add(1);
        Ok(())
    }

    /// Validate a frame against the decoder configuration before decoding it.
    fn prepare(&self, frame: &EncodedFrame, output_len: usize) -> Result<usize, DecodeError> {
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
        if output_len != sample_count {
            return Err(DecodeError::SampleCount {
                expected: sample_count,
                actual: output_len,
            });
        }
        Ok(sample_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;
    use crate::crc16;
    use crate::header::ChannelConfig;
    use crate::stream::{FrameStream, StreamEvent};

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
            output: &mut [f32],
        ) -> Result<(), DecodeError> {
            output.fill(f32::from(payload[0]));
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
    fn decoder_reset_reuses_configuration_and_restarts_frame_index() {
        let encoded = frame();
        let mut decoder = PendingDecoder::new(TestBackend::default())
            .configure(encoded.header())
            .unwrap();
        let mut samples = vec![0_i16; 1_024];
        decoder.decode_into(&encoded, &mut samples).unwrap();
        assert_eq!(decoder.frame_index(), 1);
        decoder.reset().unwrap();
        assert_eq!(decoder.frame_index(), 0);
        decoder.decode_into(&encoded, &mut samples).unwrap();
        assert_eq!(decoder.frame_index(), 1);
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

    #[test]
    fn decode_into_f32_matches_the_pcm16_path_without_quantising() {
        let encoded = frame();
        let mut decoder = PendingDecoder::new(TestBackend::default())
            .configure(encoded.header())
            .unwrap();
        let mut floats = vec![0.0_f32; decoder.sample_count().unwrap()];
        decoder.decode_into_f32(&encoded, &mut floats).unwrap();
        assert!(floats.iter().all(|&sample| sample == 7.0));
        assert_eq!(decoder.frame_index(), 1);
        // Only the PCM16 path quantises, so it alone tracks clipping.
        assert_eq!(decoder.total_clipped_samples(), 0);

        let mut wrong = [0.0_f32; 17];
        assert!(matches!(
            decoder.decode_into_f32(&encoded, &mut wrong),
            Err(DecodeError::SampleCount {
                expected: 1_024,
                actual: 17
            })
        ));
        assert_eq!(decoder.frame_index(), 1);
    }

    #[test]
    fn pcm16_path_counts_clipping_and_reset_clears_it() {
        let encoded = frame();

        /// Emits twice full scale, so every sample saturates.
        #[derive(Debug, Default)]
        struct LoudBackend;
        impl DecoderBackend for LoudBackend {
            fn configure(&mut self, _config: DecoderConfig) -> Result<(), DecodeError> {
                Ok(())
            }
            fn decode_frame(
                &mut self,
                _header: &FrameHeader,
                _payload: &[u8],
                output: &mut [f32],
            ) -> Result<(), DecodeError> {
                output.fill(FLOAT_FULL_SCALE * 2.0);
                Ok(())
            }
        }

        let mut decoder = PendingDecoder::new(LoudBackend)
            .configure(encoded.header())
            .unwrap();
        let mut samples = vec![0_i16; 1_024];
        decoder.decode_into(&encoded, &mut samples).unwrap();
        assert!(samples.iter().all(|&sample| sample == i16::MAX));
        assert_eq!(decoder.last_clipped_samples(), 1_024);
        assert_eq!(decoder.total_clipped_samples(), 1_024);
        decoder.decode_into(&encoded, &mut samples).unwrap();
        assert_eq!(decoder.total_clipped_samples(), 2_048);
        decoder.reset().unwrap();
        assert_eq!(decoder.total_clipped_samples(), 0);
    }
}
