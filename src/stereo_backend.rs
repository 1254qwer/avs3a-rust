use crate::decoder::{DecoderBackend, DecoderConfig};
use crate::error::DecodeError;
use crate::header::{ChannelConfig, CodecProfile, FrameHeader, SoundBedType};
use crate::metadata::{MetadataPayloadParser, MetadataSummary};
use crate::metadata_values::FrameMetadata;
use crate::model::AVS3_FEATURE_DIMENSIONS;
use crate::mono_backend::float_to_pcm16;
use crate::stereo::{STEREO_CHANNELS, StereoCodingMode};
use crate::stereo_core::{
    McrCoreDiagnostics, STEREO_FRAME_SAMPLES, StereoCoreDecodeError, StereoCoreDecoder,
    StereoCoreDiagnostics,
};

/// Public PCM16 backend for channel-based stereo AVS3 frames.
///
/// Bitrates above 32 kbps use the ordinary MS/ILD path. The reference switches
/// 24/32 kbps streams to MCR, which is decoded through the dedicated upmix path.
#[derive(Debug)]
pub struct StereoDecoderBackend {
    core: StereoCoreDecoder<'static>,
    metadata: MetadataPayloadParser,
    configured: Option<DecoderConfig>,
    floating_output: [f32; STEREO_FRAME_SAMPLES],
    last_diagnostics: Option<StereoCoreDiagnostics>,
    last_mcr_diagnostics: Option<McrCoreDiagnostics>,
    last_metadata: Option<MetadataSummary>,
    last_clipped_samples: usize,
    total_clipped_samples: u64,
}

impl StereoDecoderBackend {
    pub fn new_builtin() -> Result<Self, StereoCoreDecodeError> {
        Ok(Self {
            core: StereoCoreDecoder::new_builtin()?,
            metadata: MetadataPayloadParser::new(),
            configured: None,
            floating_output: [0.0; STEREO_FRAME_SAMPLES],
            last_diagnostics: None,
            last_mcr_diagnostics: None,
            last_metadata: None,
            last_clipped_samples: 0,
            total_clipped_samples: 0,
        })
    }

    pub fn core(&self) -> &StereoCoreDecoder<'static> {
        &self.core
    }

    pub fn core_mut(&mut self) -> &mut StereoCoreDecoder<'static> {
        &mut self.core
    }

    pub fn last_diagnostics(&self) -> Option<StereoCoreDiagnostics> {
        self.last_diagnostics
    }

    pub fn last_mcr_diagnostics(&self) -> Option<McrCoreDiagnostics> {
        self.last_mcr_diagnostics
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

impl DecoderBackend for StereoDecoderBackend {
    fn configure(&mut self, config: DecoderConfig) -> Result<(), DecodeError> {
        if config.channels != STEREO_CHANNELS as u8 {
            return Err(DecodeError::ChannelCount {
                expected: STEREO_CHANNELS,
                actual: usize::from(config.channels),
            });
        }
        if config.samples_per_channel != AVS3_FEATURE_DIMENSIONS as u32 {
            return Err(DecodeError::SampleCount {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: usize::try_from(config.samples_per_channel).unwrap_or(usize::MAX),
            });
        }
        let channel_stereo = config.profile == CodecProfile::ChannelBased
            && config.channel_config == Some(ChannelConfig::Stereo)
            && config.sound_bed_type.is_none()
            && config.objects == 0
            && config.bed_channels == STEREO_CHANNELS as u8
            && !config.has_lfe
            && config.bed_bitrate == Some(config.bitrate)
            && config.object_bitrate.is_none();
        let object_stereo = config.profile == CodecProfile::Mixed
            && config.channel_config.is_none()
            && config.sound_bed_type == Some(SoundBedType::ObjectsOnly)
            && config.objects == STEREO_CHANNELS as u8
            && config.bed_channels == 0
            && !config.has_lfe
            && config.bed_bitrate.is_none()
            && config
                .object_bitrate
                .and_then(|bitrate| bitrate.checked_mul(STEREO_CHANNELS as u32))
                == Some(config.bitrate);
        if !channel_stereo && !object_stereo {
            return Err(DecodeError::UnsupportedBackend);
        }
        self.core.reset();
        self.last_diagnostics = None;
        self.last_mcr_diagnostics = None;
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
        if output.len() != STEREO_FRAME_SAMPLES {
            return Err(DecodeError::SampleCount {
                expected: STEREO_FRAME_SAMPLES,
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
        let mode = StereoCodingMode::for_bitrate(audio_header.bitrate);
        let (diagnostics, mcr_diagnostics) = match mode {
            StereoCodingMode::MidSide => (
                Some(self.core.decode(
                    parsed.audio_payload(),
                    &audio_header,
                    &mut self.floating_output,
                )?),
                None,
            ),
            StereoCodingMode::Mcr => (
                None,
                Some(self.core.decode_mcr(
                    parsed.audio_payload(),
                    &audio_header,
                    &mut self.floating_output,
                )?),
            ),
        };
        let clipped = float_to_pcm16(&self.floating_output, output);
        self.last_diagnostics = diagnostics;
        self.last_mcr_diagnostics = mcr_diagnostics;
        self.last_metadata = Some(metadata);
        self.last_clipped_samples = clipped;
        self.total_clipped_samples = self
            .total_clipped_samples
            .saturating_add(u64::try_from(clipped).unwrap_or(u64::MAX));
        Ok(())
    }
}
