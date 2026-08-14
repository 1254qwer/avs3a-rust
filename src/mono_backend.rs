use crate::core_side::CoreBitstreamConfig;
use crate::decoder::{DecoderBackend, DecoderConfig};
use crate::error::DecodeError;
use crate::header::{ChannelConfig, CodecProfile, FrameHeader, SoundBedType};
use crate::metadata::{MetadataPayloadParser, MetadataSummary};
use crate::metadata_values::FrameMetadata;
use crate::model::AVS3_FEATURE_DIMENSIONS;
use crate::mono::{MonoCoreDecodeError, MonoCoreDecoder, MonoCoreDiagnostics};

/// Public decoder backend for channel-based mono AVS3 frames.
///
/// The backend owns all entropy, DSP, random and overlap state. It converts
/// the core's floating output with the C decoder's `floor(x + 0.5)` rule and
/// saturates to PCM16 while reporting clipping counts.
#[derive(Debug)]
pub struct MonoDecoderBackend {
    core: MonoCoreDecoder<'static>,
    metadata: MetadataPayloadParser,
    configured: Option<DecoderConfig>,
    floating_output: [f32; AVS3_FEATURE_DIMENSIONS],
    last_diagnostics: Option<MonoCoreDiagnostics>,
    last_metadata: Option<MetadataSummary>,
    last_clipped_samples: usize,
    total_clipped_samples: u64,
}

impl MonoDecoderBackend {
    pub fn new_builtin() -> Result<Self, MonoCoreDecodeError> {
        Ok(Self {
            core: MonoCoreDecoder::new_builtin()?,
            metadata: MetadataPayloadParser::new(),
            configured: None,
            floating_output: [0.0; AVS3_FEATURE_DIMENSIONS],
            last_diagnostics: None,
            last_metadata: None,
            last_clipped_samples: 0,
            total_clipped_samples: 0,
        })
    }

    pub fn core(&self) -> &MonoCoreDecoder<'static> {
        &self.core
    }

    pub fn core_mut(&mut self) -> &mut MonoCoreDecoder<'static> {
        &mut self.core
    }

    pub fn last_diagnostics(&self) -> Option<MonoCoreDiagnostics> {
        self.last_diagnostics
    }

    pub fn last_metadata(&self) -> Option<MetadataSummary> {
        self.last_metadata
    }

    pub fn last_metadata_values(&self) -> Option<&FrameMetadata> {
        self.last_metadata?;
        self.metadata.last_metadata()
    }

    pub fn last_clipped_samples(&self) -> usize {
        self.last_clipped_samples
    }

    pub fn total_clipped_samples(&self) -> u64 {
        self.total_clipped_samples
    }
}

impl DecoderBackend for MonoDecoderBackend {
    fn configure(&mut self, config: DecoderConfig) -> Result<(), DecodeError> {
        if config.channels != 1 {
            return Err(DecodeError::ChannelCount {
                expected: 1,
                actual: usize::from(config.channels),
            });
        }
        if config.samples_per_channel != AVS3_FEATURE_DIMENSIONS as u32 {
            return Err(DecodeError::SampleCount {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: usize::try_from(config.samples_per_channel).unwrap_or(usize::MAX),
            });
        }
        let channel_mono = config.profile == CodecProfile::ChannelBased
            && config.channel_config == Some(ChannelConfig::Mono)
            && config.sound_bed_type.is_none()
            && config.objects == 0
            && config.bed_channels == 1
            && !config.has_lfe
            && config.bed_bitrate == Some(config.bitrate)
            && config.object_bitrate.is_none();
        let object_mono = config.profile == CodecProfile::Mixed
            && config.channel_config.is_none()
            && config.sound_bed_type == Some(SoundBedType::ObjectsOnly)
            && config.objects == 1
            && config.bed_channels == 0
            && !config.has_lfe
            && config.bed_bitrate.is_none()
            && config.object_bitrate == Some(config.bitrate);
        if !channel_mono && !object_mono {
            return Err(DecodeError::UnsupportedBackend);
        }
        self.core.reset();
        self.last_diagnostics = None;
        self.last_metadata = None;
        self.last_clipped_samples = 0;
        self.total_clipped_samples = 0;
        self.metadata.prepare_storage();
        self.configured = Some(config);
        Ok(())
    }

    fn decode_frame(
        &mut self,
        header: &FrameHeader,
        payload: &[u8],
        output: &mut [i16],
    ) -> Result<(), DecodeError> {
        if output.len() != AVS3_FEATURE_DIMENSIONS {
            return Err(DecodeError::SampleCount {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: output.len(),
            });
        }
        if self.configured.is_none() {
            return Err(DecodeError::UnsupportedBackend);
        }

        self.last_metadata = None;
        let parsed = self.metadata.parse_with_object_count(
            payload,
            header.payload_bits,
            usize::from(header.objects),
        )?;
        let metadata = parsed.summary();
        let mut audio_header = *header;
        audio_header.payload_bits = parsed.audio_bits();
        audio_header.payload_len = parsed.audio_payload().len();
        audio_header.frame_len = audio_header.header_len + audio_header.payload_len;
        let config =
            CoreBitstreamConfig::for_mono(&audio_header).map_err(MonoCoreDecodeError::from)?;
        let diagnostics =
            self.core
                .decode(parsed.audio_payload(), config, &mut self.floating_output)?;
        let clipped = float_to_pcm16(&self.floating_output, output);
        self.last_diagnostics = Some(diagnostics);
        self.last_metadata = Some(metadata);
        self.last_clipped_samples = clipped;
        self.total_clipped_samples = self
            .total_clipped_samples
            .saturating_add(u64::try_from(clipped).unwrap_or(u64::MAX));
        Ok(())
    }
}

/// Convert finite floating synthesis samples using the AVS3 reference's
/// rounding and saturation rule. Non-finite values are mapped to zero and
/// counted as clipped so platform-specific float-to-int behavior cannot leak
/// through the safe API.
pub fn float_to_pcm16(input: &[f32], output: &mut [i16]) -> usize {
    assert_eq!(input.len(), output.len(), "PCM conversion length mismatch");
    let mut clipped = 0;
    for (&sample, destination) in input.iter().zip(output) {
        if !sample.is_finite() {
            *destination = 0;
            clipped += 1;
            continue;
        }
        let rounded = (sample + 0.5_f32).floor();
        if rounded > f32::from(i16::MAX) {
            *destination = i16::MAX;
            clipped += 1;
        } else if rounded < f32::from(i16::MIN) {
            *destination = i16::MIN;
            clipped += 1;
        } else {
            *destination = rounded as i16;
        }
    }
    clipped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_conversion_matches_c_rounding_and_saturation() {
        let input = [
            -32_769.0,
            -32_768.6,
            -32_768.5,
            -1.5,
            -0.5,
            -0.499,
            0.0,
            0.499,
            0.5,
            1.5,
            32_766.5,
            32_767.4,
            32_767.5,
            f32::NAN,
        ];
        let mut output = [0_i16; 14];
        assert_eq!(float_to_pcm16(&input, &mut output), 4);
        assert_eq!(
            output,
            [-32_768, -32_768, -32_768, -1, 0, 0, 0, 0, 1, 2, 32_767, 32_767, 32_767, 0,]
        );
    }
}
