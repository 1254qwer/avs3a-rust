use crate::decoder::{Decoder, PendingDecoder};
use crate::error::DecodeError;
use crate::header::{ChannelConfig, CodecProfile, FrameHeader};
use crate::hoa_backend::HoaDecoderBackend;
use crate::hoa_synthesis::HOA_BASIS_DELAY_FRAMES;
use crate::mc_backend::McDecoderBackend;
use crate::mc_side::is_multichannel_config;
use crate::metadata_values::FrameMetadata;
use crate::mix_backend::MixDecoderBackend;
use crate::mono_backend::MonoDecoderBackend;
use crate::stereo_backend::StereoDecoderBackend;

/// Frames to decode and discard after a reset before output is exact.
///
/// Every synthesis pipeline carries an MDCT overlap from the previous frame, so
/// the first frame after a reset is missing the first half of its window and
/// only the second frame onwards is correct.
pub const CHANNEL_WARMUP_FRAMES: u64 = 1;

/// HOA warm-up depth.
///
/// HOA post-synthesis additionally delays the spatial basis indices by
/// [`crate::hoa::HOA_BASIS_DELAY_FRAMES`] frames, and its analysis stage reads the previous
/// frame's transport channels. The oldest basis slot therefore has to be filled
/// by a frame whose transport channels were themselves already correct, which
/// is what the extra frame over [`CHANNEL_WARMUP_FRAMES`] buys.
pub const HOA_WARMUP_FRAMES: u64 = HOA_BASIS_DELAY_FRAMES as u64;

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

/// Forward a call to whichever `Decoder<B>` the variant holds.
///
/// The variants differ only in the backend type, so every accessor would
/// otherwise be the same five-arm match written out again.
macro_rules! dispatch {
    ($self:expr, |$decoder:ident| $body:expr) => {
        match $self {
            Self::Mono($decoder) => $body,
            Self::Stereo($decoder) => $body,
            Self::Mc($decoder) => $body,
            Self::Mix($decoder) => $body,
            Self::Hoa($decoder) => $body,
        }
    };
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

    /// Decode into caller-owned interleaved PCM16 storage.
    pub fn decode_into(
        &mut self,
        frame: &crate::stream::EncodedFrame,
        output: &mut [i16],
    ) -> Result<(), DecodeError> {
        dispatch!(self, |decoder| decoder.decode_into(frame, output))
    }

    /// Decode into caller-owned interleaved float storage.
    ///
    /// This is the synthesis pipeline's native output. A renderer that
    /// downmixes, applies gain or measures loudness should use it rather than
    /// undoing the PCM16 quantisation that [`Self::decode_into`] performs.
    pub fn decode_into_f32(
        &mut self,
        frame: &crate::stream::EncodedFrame,
        output: &mut [f32],
    ) -> Result<(), DecodeError> {
        dispatch!(self, |decoder| decoder.decode_into_f32(frame, output))
    }

    pub fn reset(&mut self) -> Result<(), DecodeError> {
        dispatch!(self, |decoder| decoder.reset())
    }

    pub fn config(&self) -> crate::decoder::DecoderConfig {
        dispatch!(self, |decoder| decoder.config())
    }

    pub fn frame_index(&self) -> u64 {
        dispatch!(self, |decoder| decoder.frame_index())
    }

    /// Interleaved sample count of one decoded frame.
    ///
    /// This is the exact length [`Self::decode_into`] and
    /// [`Self::decode_into_f32`] require, so callers do not have to rederive it
    /// from [`Self::config`].
    pub fn sample_count(&self) -> Result<usize, DecodeError> {
        dispatch!(self, |decoder| decoder.sample_count())
    }

    /// Frames to decode and discard after [`Self::reset`] before output is
    /// exact.
    ///
    /// Seeking to an arbitrary frame means resetting and replaying this many
    /// preceding frames. The count depends on the profile, so reading it here
    /// keeps callers from hard-coding a value that is right for channel-based
    /// audio and wrong for HOA.
    pub fn warmup_frames(&self) -> u64 {
        match self {
            Self::Mono(_) | Self::Stereo(_) | Self::Mc(_) | Self::Mix(_) => CHANNEL_WARMUP_FRAMES,
            Self::Hoa(_) => HOA_WARMUP_FRAMES,
        }
    }

    /// Samples clamped to full scale by the most recent PCM16 conversion.
    pub fn last_clipped_samples(&self) -> usize {
        dispatch!(self, |decoder| decoder.last_clipped_samples())
    }

    /// Samples clamped to full scale since the decoder was configured or reset.
    pub fn total_clipped_samples(&self) -> u64 {
        dispatch!(self, |decoder| decoder.total_clipped_samples())
    }

    /// Metadata carried by the most recently decoded frame.
    ///
    /// Mix and HOA streams put object positions, gains and loudness here. A
    /// renderer needs them frame by frame, which is why they are reachable
    /// without unwrapping the concrete backend.
    pub fn last_metadata_values(&self) -> Option<&FrameMetadata> {
        dispatch!(self, |decoder| decoder.backend().last_metadata_values())
    }
}
