use crate::{
    ChannelConfig, CodecProfile, DecodeError, Decoder, FrameHeader, HoaDecoderBackend,
    McDecoderBackend, MixDecoderBackend, MonoDecoderBackend, PendingDecoder, StereoDecoderBackend,
    is_multichannel_config,
};

/// The built-in AV3A decoder selected from a frame header.
///
/// This is the common stateful entry point for applications that do not need
/// to choose a synthesis backend themselves. It owns one configured decoder
/// for the lifetime of a stream and exposes reset for seek recovery.
#[derive(Debug)]
pub enum BuiltinDecoder {
    Mono(Box<Decoder<MonoDecoderBackend>>),
    Stereo(Box<Decoder<StereoDecoderBackend>>),
    Mc(Box<Decoder<McDecoderBackend>>),
    Mix(Box<Decoder<MixDecoderBackend>>),
    Hoa(Box<Decoder<HoaDecoderBackend>>),
}

impl BuiltinDecoder {
    pub fn configure(header: &FrameHeader) -> Result<Self, DecodeError> {
        match (header.profile, header.channel_config) {
            (CodecProfile::ChannelBased, Some(ChannelConfig::Mono)) => Ok(Self::Mono(Box::new(
                PendingDecoder::new(MonoDecoderBackend::new_builtin()?).configure(header)?,
            ))),
            (CodecProfile::ChannelBased, Some(ChannelConfig::Stereo)) => {
                Ok(Self::Stereo(Box::new(
                    PendingDecoder::new(StereoDecoderBackend::new_builtin()?).configure(header)?,
                )))
            }
            (CodecProfile::ChannelBased, Some(config)) if is_multichannel_config(config) => {
                Ok(Self::Mc(Box::new(
                    PendingDecoder::new(McDecoderBackend::new_builtin()?).configure(header)?,
                )))
            }
            (CodecProfile::Mixed, _) => Ok(Self::Mix(Box::new(
                PendingDecoder::new(MixDecoderBackend::new_builtin()?).configure(header)?,
            ))),
            (
                CodecProfile::Hoa,
                Some(ChannelConfig::Hoa1 | ChannelConfig::Hoa2 | ChannelConfig::Hoa3),
            ) => Ok(Self::Hoa(Box::new(
                PendingDecoder::new(HoaDecoderBackend::new_builtin()?).configure(header)?,
            ))),
            _ => Err(DecodeError::UnsupportedBackend),
        }
    }

    pub fn decode_into(
        &mut self,
        frame: &crate::EncodedFrame,
        output: &mut [i16],
    ) -> Result<(), DecodeError> {
        match self {
            Self::Mono(decoder) => decoder.decode_into(frame, output),
            Self::Stereo(decoder) => decoder.decode_into(frame, output),
            Self::Mc(decoder) => decoder.decode_into(frame, output),
            Self::Mix(decoder) => decoder.decode_into(frame, output),
            Self::Hoa(decoder) => decoder.decode_into(frame, output),
        }
    }

    pub fn reset(&mut self) -> Result<(), DecodeError> {
        match self {
            Self::Mono(decoder) => decoder.reset(),
            Self::Stereo(decoder) => decoder.reset(),
            Self::Mc(decoder) => decoder.reset(),
            Self::Mix(decoder) => decoder.reset(),
            Self::Hoa(decoder) => decoder.reset(),
        }
    }

    pub fn config(&self) -> crate::DecoderConfig {
        match self {
            Self::Mono(decoder) => decoder.config(),
            Self::Stereo(decoder) => decoder.config(),
            Self::Mc(decoder) => decoder.config(),
            Self::Mix(decoder) => decoder.config(),
            Self::Hoa(decoder) => decoder.config(),
        }
    }

    pub fn frame_index(&self) -> u64 {
        match self {
            Self::Mono(decoder) => decoder.frame_index(),
            Self::Stereo(decoder) => decoder.frame_index(),
            Self::Mc(decoder) => decoder.frame_index(),
            Self::Mix(decoder) => decoder.frame_index(),
            Self::Hoa(decoder) => decoder.frame_index(),
        }
    }

    pub fn last_clipped_samples(&self) -> usize {
        match self {
            Self::Mono(decoder) => decoder.backend().last_clipped_samples(),
            Self::Stereo(decoder) => decoder.backend().last_clipped_samples(),
            Self::Mc(decoder) => decoder.backend().last_clipped_samples(),
            Self::Mix(decoder) => decoder.backend().last_clipped_samples(),
            Self::Hoa(decoder) => decoder.backend().last_clipped_samples(),
        }
    }
}
