use crate::decoder::{DecoderBackend, DecoderConfig};
use crate::error::DecodeError;
use crate::header::{ChannelConfig, CodecProfile, FrameHeader, MAX_CHANNELS, SoundBedType};
use crate::mc::is_multichannel_config;
use crate::mc_core::{McCoreDecodeError, McCoreDecoder, McCoreDiagnostics};
use crate::metadata::{MetadataPayloadParser, MetadataSummary};
use crate::metadata_values::FrameMetadata;
use crate::model::AVS3_FEATURE_DIMENSIONS;

/// Public backend for channel-based multichannel AVS3 frames.
///
/// All large buffers and channel-local DSP state are allocated once. Metadata
/// is consumed before the audio payload, and output samples are interleaved in
/// the channel order declared by the AVS3 channel configuration.
#[derive(Debug)]
pub struct McDecoderBackend {
    core: McCoreDecoder<'static>,
    metadata: MetadataPayloadParser,
    configured: Option<DecoderConfig>,
    last_diagnostics: Option<McCoreDiagnostics>,
    last_metadata: Option<MetadataSummary>,
}

impl McDecoderBackend {
    pub fn new_builtin() -> Result<Self, McCoreDecodeError> {
        Ok(Self {
            core: McCoreDecoder::new_builtin()?,
            metadata: MetadataPayloadParser::new(),
            configured: None,
            last_diagnostics: None,
            last_metadata: None,
        })
    }

    pub fn core(&self) -> &McCoreDecoder<'static> {
        &self.core
    }

    pub fn core_mut(&mut self) -> &mut McCoreDecoder<'static> {
        &mut self.core
    }

    pub fn last_diagnostics(&self) -> Option<McCoreDiagnostics> {
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

impl DecoderBackend for McDecoderBackend {
    fn configure(&mut self, config: DecoderConfig) -> Result<(), DecodeError> {
        let channels = usize::from(config.channels);
        let channel_mc = config.profile == CodecProfile::ChannelBased
            && config.channel_config.is_some_and(is_multichannel_config)
            && config.sound_bed_type.is_none()
            && config.objects == 0
            && config.bed_channels == config.channels
            && config
                .channel_config
                .is_some_and(|channel_config| channel_config.channels() == config.channels)
            && config
                .channel_config
                .is_some_and(|channel_config| channel_config.has_lfe() == config.has_lfe)
            && config.bed_bitrate == Some(config.bitrate)
            && config.object_bitrate.is_none();
        let object_mc = config.profile == CodecProfile::Mixed
            && config.channel_config.is_none()
            && config.sound_bed_type == Some(SoundBedType::ObjectsOnly)
            && channels >= 3
            && config.objects == config.channels
            && config.bed_channels == 0
            && !config.has_lfe
            && config.bed_bitrate.is_none()
            && mix_bitrate_is_consistent(config);
        let bed_object_mc = config.profile == CodecProfile::Mixed
            && config.sound_bed_type == Some(SoundBedType::ChannelBed)
            && config.objects != 0
            && config.channel_config.is_some_and(|channel_config| {
                channel_config == ChannelConfig::Stereo || is_multichannel_config(channel_config)
            })
            && config
                .channel_config
                .is_some_and(|channel_config| channel_config.channels() == config.bed_channels)
            && usize::from(config.bed_channels).checked_add(usize::from(config.objects))
                == Some(channels)
            && config
                .channel_config
                .is_some_and(|channel_config| channel_config.has_lfe() == config.has_lfe)
            && mix_bitrate_is_consistent(config);
        if !channel_mc && !object_mc && !bed_object_mc {
            return Err(DecodeError::UnsupportedBackend);
        }
        if channels > usize::from(MAX_CHANNELS) {
            return Err(DecodeError::ChannelCount {
                expected: usize::from(MAX_CHANNELS),
                actual: channels,
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
        let diagnostics = self
            .core
            .decode(parsed.audio_payload(), &audio_header, output)?;
        self.last_diagnostics = Some(diagnostics);
        self.last_metadata = Some(metadata);
        Ok(())
    }
}

fn mix_bitrate_is_consistent(config: DecoderConfig) -> bool {
    let object_total = config
        .object_bitrate
        .and_then(|bitrate| bitrate.checked_mul(u32::from(config.objects)));
    let expected = match config.sound_bed_type {
        Some(SoundBedType::ObjectsOnly) => object_total,
        Some(SoundBedType::ChannelBed) => object_total
            .zip(config.bed_bitrate)
            .and_then(|(objects, bed)| objects.checked_add(bed)),
        None => None,
    };
    expected == Some(config.bitrate)
}
