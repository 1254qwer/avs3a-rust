use crate::decoder::{DecoderBackend, DecoderConfig};
use crate::error::DecodeError;
use crate::header::{CodecProfile, FrameHeader, SoundBedType};
use crate::mc_backend::McDecoderBackend;
use crate::metadata::MetadataSummary;
use crate::metadata_values::FrameMetadata;
use crate::mono_backend::MonoDecoderBackend;
use crate::stereo_backend::StereoDecoderBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixCoreKind {
    Mono,
    Stereo,
    Multichannel,
}

#[derive(Debug)]
enum MixBackendState {
    Unconfigured,
    Mono(Box<MonoDecoderBackend>),
    Stereo(Box<StereoDecoderBackend>),
    Multichannel(Box<McDecoderBackend>),
}

/// Public PCM16 backend for AVS3 Mix frames.
///
/// The reference format reuses mono for one object, stereo for two objects,
/// and the MC pipeline for three or more objects or any channel-bed Mix. The
/// selected backend is allocated during configuration, so unused neural and
/// DSP workspaces are not retained.
#[derive(Debug)]
pub struct MixDecoderBackend {
    state: MixBackendState,
}

impl MixDecoderBackend {
    pub fn new_builtin() -> Result<Self, DecodeError> {
        Ok(Self {
            state: MixBackendState::Unconfigured,
        })
    }

    pub fn core_kind(&self) -> Option<MixCoreKind> {
        match self.state {
            MixBackendState::Unconfigured => None,
            MixBackendState::Mono(_) => Some(MixCoreKind::Mono),
            MixBackendState::Stereo(_) => Some(MixCoreKind::Stereo),
            MixBackendState::Multichannel(_) => Some(MixCoreKind::Multichannel),
        }
    }

    pub fn mono_backend(&self) -> Option<&MonoDecoderBackend> {
        match &self.state {
            MixBackendState::Mono(backend) => Some(backend),
            _ => None,
        }
    }

    pub fn stereo_backend(&self) -> Option<&StereoDecoderBackend> {
        match &self.state {
            MixBackendState::Stereo(backend) => Some(backend),
            _ => None,
        }
    }

    pub fn multichannel_backend(&self) -> Option<&McDecoderBackend> {
        match &self.state {
            MixBackendState::Multichannel(backend) => Some(backend),
            _ => None,
        }
    }

    pub fn last_metadata(&self) -> Option<MetadataSummary> {
        match &self.state {
            MixBackendState::Unconfigured => None,
            MixBackendState::Mono(backend) => backend.last_metadata(),
            MixBackendState::Stereo(backend) => backend.last_metadata(),
            MixBackendState::Multichannel(backend) => backend.last_metadata(),
        }
    }

    pub fn last_metadata_values(&self) -> Option<&FrameMetadata> {
        match &self.state {
            MixBackendState::Unconfigured => None,
            MixBackendState::Mono(backend) => backend.last_metadata_values(),
            MixBackendState::Stereo(backend) => backend.last_metadata_values(),
            MixBackendState::Multichannel(backend) => backend.last_metadata_values(),
        }
    }
}

impl DecoderBackend for MixDecoderBackend {
    fn configure(&mut self, config: DecoderConfig) -> Result<(), DecodeError> {
        if config.profile != CodecProfile::Mixed {
            return Err(DecodeError::UnsupportedBackend);
        }
        let state = match config.sound_bed_type {
            Some(SoundBedType::ObjectsOnly) if config.objects == 1 => {
                let mut backend = Box::new(MonoDecoderBackend::new_builtin()?);
                backend.configure(config)?;
                MixBackendState::Mono(backend)
            }
            Some(SoundBedType::ObjectsOnly) if config.objects == 2 => {
                let mut backend = Box::new(StereoDecoderBackend::new_builtin()?);
                backend.configure(config)?;
                MixBackendState::Stereo(backend)
            }
            Some(SoundBedType::ObjectsOnly | SoundBedType::ChannelBed) => {
                let mut backend = Box::new(McDecoderBackend::new_builtin()?);
                backend.configure(config)?;
                MixBackendState::Multichannel(backend)
            }
            None => return Err(DecodeError::UnsupportedBackend),
        };
        self.state = state;
        Ok(())
    }

    fn decode_frame(
        &mut self,
        header: &FrameHeader,
        payload: &[u8],
        output: &mut [f32],
    ) -> Result<(), DecodeError> {
        match &mut self.state {
            MixBackendState::Unconfigured => Err(DecodeError::UnsupportedBackend),
            MixBackendState::Mono(backend) => backend.decode_frame(header, payload, output),
            MixBackendState::Stereo(backend) => backend.decode_frame(header, payload, output),
            MixBackendState::Multichannel(backend) => backend.decode_frame(header, payload, output),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{BitDepth, ChannelConfig, NnType};

    fn config(objects: u8) -> DecoderConfig {
        DecoderConfig {
            sample_rate: 48_000,
            bitrate: 64_000 * u32::from(objects),
            channels: objects,
            samples_per_channel: 1_024,
            bit_depth: BitDepth::Sixteen,
            profile: CodecProfile::Mixed,
            nn_type: NnType::Main,
            channel_config: None,
            sound_bed_type: Some(SoundBedType::ObjectsOnly),
            hoa_order: None,
            objects,
            bed_channels: 0,
            has_lfe: false,
            bed_bitrate: None,
            object_bitrate: Some(64_000),
        }
    }

    #[test]
    fn lazily_selects_the_reference_core_family() {
        let mut backend = MixDecoderBackend::new_builtin().unwrap();
        assert_eq!(backend.core_kind(), None);

        backend.configure(config(1)).unwrap();
        assert_eq!(backend.core_kind(), Some(MixCoreKind::Mono));
        assert!(backend.mono_backend().is_some());

        backend.configure(config(2)).unwrap();
        assert_eq!(backend.core_kind(), Some(MixCoreKind::Stereo));
        assert!(backend.stereo_backend().is_some());

        backend.configure(config(3)).unwrap();
        assert_eq!(backend.core_kind(), Some(MixCoreKind::Multichannel));
        assert!(backend.multichannel_backend().is_some());

        let mut invalid = config(1);
        invalid.profile = CodecProfile::ChannelBased;
        assert!(matches!(
            backend.configure(invalid),
            Err(DecodeError::UnsupportedBackend)
        ));
        assert_eq!(backend.core_kind(), Some(MixCoreKind::Multichannel));

        let mut bed = config(1);
        bed.channels = 7;
        bed.bitrate = 448_000;
        bed.channel_config = Some(ChannelConfig::Mc5_1);
        bed.sound_bed_type = Some(SoundBedType::ChannelBed);
        bed.bed_channels = 6;
        bed.has_lfe = true;
        bed.bed_bitrate = Some(384_000);
        backend.configure(bed).unwrap();
        assert_eq!(backend.core_kind(), Some(MixCoreKind::Multichannel));
    }
}
