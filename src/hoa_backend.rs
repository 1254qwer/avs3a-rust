use crate::decoder::{DecoderBackend, DecoderConfig};
use crate::error::DecodeError;
use crate::header::{ChannelConfig, CodecProfile, FrameHeader, MAX_CHANNELS};
use crate::hoa_core::{HoaCoreDecodeError, HoaCoreDecoder, HoaCoreDiagnostics};
use crate::metadata::{MetadataPayloadParser, MetadataSummary};
use crate::metadata_values::FrameMetadata;
use crate::model::AVS3_FEATURE_DIMENSIONS;

/// Public PCM16 backend for first-, second- and third-order HOA frames.
///
/// The backend consumes frame metadata before the HOA audio syntax, owns all
/// core and post-filter temporal state, and returns interleaved HOA components
/// in ACN order as emitted by the reference synthesis matrix.
#[derive(Debug)]
pub struct HoaDecoderBackend {
    core: HoaCoreDecoder<'static>,
    metadata: MetadataPayloadParser,
    configured: Option<DecoderConfig>,
    last_diagnostics: Option<HoaCoreDiagnostics>,
    last_metadata: Option<MetadataSummary>,
}

impl HoaDecoderBackend {
    pub fn new_builtin() -> Result<Self, HoaCoreDecodeError> {
        Ok(Self {
            core: HoaCoreDecoder::new_builtin()?,
            metadata: MetadataPayloadParser::new(),
            configured: None,
            last_diagnostics: None,
            last_metadata: None,
        })
    }

    pub fn core(&self) -> &HoaCoreDecoder<'static> {
        &self.core
    }

    pub fn core_mut(&mut self) -> &mut HoaCoreDecoder<'static> {
        &mut self.core
    }

    pub fn last_diagnostics(&self) -> Option<HoaCoreDiagnostics> {
        self.last_diagnostics
    }

    pub fn last_metadata(&self) -> Option<MetadataSummary> {
        self.last_metadata
    }

    pub fn last_metadata_values(&self) -> Option<&FrameMetadata> {
        self.last_metadata?;
        self.metadata.last_metadata()
    }
}

impl DecoderBackend for HoaDecoderBackend {
    fn configure(&mut self, config: DecoderConfig) -> Result<(), DecodeError> {
        if config.profile != CodecProfile::Hoa || config.objects != 0 {
            return Err(DecodeError::UnsupportedBackend);
        }
        let (expected_config, expected_order, expected_channels) = match config.hoa_order {
            Some(1) => (ChannelConfig::Hoa1, 1, 4),
            Some(2) => (ChannelConfig::Hoa2, 2, 9),
            Some(3) => (ChannelConfig::Hoa3, 3, 16),
            _ => return Err(DecodeError::UnsupportedBackend),
        };
        if config.channel_config != Some(expected_config)
            || config.hoa_order != Some(expected_order)
        {
            return Err(DecodeError::UnsupportedBackend);
        }
        if usize::from(config.channels) != expected_channels
            || usize::from(config.bed_channels) != expected_channels
        {
            return Err(DecodeError::ChannelCount {
                expected: expected_channels,
                actual: usize::from(config.channels),
            });
        }
        if usize::from(config.channels) > usize::from(MAX_CHANNELS) {
            return Err(DecodeError::ChannelCount {
                expected: usize::from(MAX_CHANNELS),
                actual: usize::from(config.channels),
            });
        }
        if config.samples_per_channel != AVS3_FEATURE_DIMENSIONS as u32 {
            return Err(DecodeError::SampleCount {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: usize::try_from(config.samples_per_channel).unwrap_or(usize::MAX),
            });
        }

        self.core.reset();
        self.last_diagnostics = None;
        self.last_metadata = None;
        self.metadata.prepare_storage();
        self.configured = Some(config);
        Ok(())
    }

    fn decode_frame(
        &mut self,
        header: &FrameHeader,
        payload: &[u8],
        output: &mut [f32],
    ) -> Result<(), DecodeError> {
        let Some(configured) = self.configured else {
            return Err(DecodeError::UnsupportedBackend);
        };
        let sample_count = usize::from(configured.channels)
            .checked_mul(AVS3_FEATURE_DIMENSIONS)
            .ok_or(DecodeError::SampleCount {
                expected: usize::MAX,
                actual: output.len(),
            })?;
        if output.len() != sample_count {
            return Err(DecodeError::SampleCount {
                expected: sample_count,
                actual: output.len(),
            });
        }

        self.last_metadata = None;
        let parsed = self.metadata.parse(payload, header.payload_bits)?;
        let metadata = parsed.summary();
        let mut audio_header = *header;
        audio_header.payload_bits = parsed.audio_bits();
        audio_header.payload_len = parsed.audio_payload().len();
        audio_header.frame_len = audio_header.header_len + audio_header.payload_len;
        let diagnostics = self
            .core
            .decode(parsed.audio_payload(), &audio_header, output)?;
        self.last_diagnostics = Some(diagnostics);
        self.last_metadata = Some(metadata);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{BitDepth, NnType};

    fn config(order: u8) -> DecoderConfig {
        let (channel_config, channels) = match order {
            1 => (ChannelConfig::Hoa1, 4),
            2 => (ChannelConfig::Hoa2, 9),
            3 => (ChannelConfig::Hoa3, 16),
            _ => panic!("invalid test order"),
        };
        DecoderConfig {
            sample_rate: 48_000,
            bitrate: 192_000,
            channels,
            samples_per_channel: AVS3_FEATURE_DIMENSIONS as u32,
            bit_depth: BitDepth::Sixteen,
            profile: CodecProfile::Hoa,
            nn_type: NnType::Main,
            channel_config: Some(channel_config),
            sound_bed_type: None,
            hoa_order: Some(order),
            objects: 0,
            bed_channels: channels,
            has_lfe: false,
            bed_bitrate: None,
            object_bitrate: None,
        }
    }

    #[test]
    fn accepts_all_three_consistent_hoa_orders() {
        let mut backend = HoaDecoderBackend::new_builtin().unwrap();
        for order in 1..=3 {
            backend.configure(config(order)).unwrap();
        }
    }

    #[test]
    fn rejects_profile_and_order_mismatches() {
        let mut backend = HoaDecoderBackend::new_builtin().unwrap();
        let mut mismatched = config(1);
        mismatched.channel_config = Some(ChannelConfig::Hoa2);
        assert!(matches!(
            backend.configure(mismatched),
            Err(DecodeError::UnsupportedBackend)
        ));

        let mut wrong_profile = config(1);
        wrong_profile.profile = CodecProfile::ChannelBased;
        assert!(matches!(
            backend.configure(wrong_profile),
            Err(DecodeError::UnsupportedBackend)
        ));
    }
}
