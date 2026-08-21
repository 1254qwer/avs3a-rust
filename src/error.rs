use core::fmt;

use crate::hoa_core::HoaCoreDecodeError;
use crate::mc_core::McCoreDecodeError;
use crate::metadata::MetadataError;
use crate::mono::MonoCoreDecodeError;
use crate::stereo_core::StereoCoreDecodeError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitstreamError {
    UnexpectedEof {
        position: usize,
        requested: usize,
        available: usize,
    },
    InvalidWidth(usize),
    PositionOverflow,
}

impl fmt::Display for BitstreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof {
                position,
                requested,
                available,
            } => write!(
                f,
                "bitstream ended at bit {position}; need {requested} bits, {available} available"
            ),
            Self::InvalidWidth(width) => {
                write!(f, "cannot read {width} bits (valid range: 0..=64)")
            }
            Self::PositionOverflow => f.write_str("bit position overflow"),
        }
    }
}

impl std::error::Error for BitstreamError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderError {
    NeedMoreData { needed: usize, available: usize },
    NoSync,
    InvalidSync { offset: usize, value: u16 },
    UnsupportedCodec { offset: usize, value: u8 },
    AncillaryData { offset: usize },
    UnsupportedProfile { offset: usize, value: u8 },
    UnsupportedNnType(u8),
    InvalidSamplingRateIndex(u8),
    InvalidChannelConfig(u8),
    InvalidSoundBedType(u8),
    InvalidHoaOrder(u8),
    InvalidBitDepth(u8),
    InvalidBitrateIndex { config: u8, index: u8 },
    InvalidObjectCount(u8),
    PayloadTooLarge { size: usize, limit: usize },
    ArithmeticOverflow,
    Bitstream(BitstreamError),
}

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NeedMoreData { needed, available } => {
                write!(
                    f,
                    "need at least {needed} bytes, only {available} available"
                )
            }
            Self::NoSync => f.write_str("no AV3A sync word found"),
            Self::InvalidSync { offset, value } => {
                write!(f, "invalid sync word 0x{value:03x} at byte {offset}")
            }
            Self::UnsupportedCodec { offset, value } => {
                write!(f, "unsupported audio codec id {value} at byte {offset}")
            }
            Self::AncillaryData { offset } => {
                write!(f, "ancillary-data flag is set at byte {offset}")
            }
            Self::UnsupportedProfile { offset, value } => {
                write!(f, "unsupported coding profile {value} at byte {offset}")
            }
            Self::UnsupportedNnType(value) => write!(f, "unsupported neural-network type {value}"),
            Self::InvalidSamplingRateIndex(index) => {
                write!(f, "invalid sampling-rate index {index}")
            }
            Self::InvalidChannelConfig(index) => write!(f, "invalid channel configuration {index}"),
            Self::InvalidSoundBedType(value) => write!(f, "invalid sound-bed type {value}"),
            Self::InvalidHoaOrder(order) => write!(f, "unsupported HOA order {order}"),
            Self::InvalidBitDepth(value) => write!(f, "invalid sample resolution {value}"),
            Self::InvalidBitrateIndex { config, index } => {
                write!(
                    f,
                    "invalid bitrate index {index} for channel configuration {config}"
                )
            }
            Self::InvalidObjectCount(value) => write!(f, "invalid object count {value}"),
            Self::PayloadTooLarge { size, limit } => {
                write!(
                    f,
                    "frame payload is {size} bytes; maximum supported size is {limit}"
                )
            }
            Self::ArithmeticOverflow => f.write_str("frame size arithmetic overflow"),
            Self::Bitstream(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for HeaderError {}

impl From<BitstreamError> for HeaderError {
    fn from(value: BitstreamError) -> Self {
        Self::Bitstream(value)
    }
}

#[derive(Debug)]
pub enum StreamError {
    Header(HeaderError),
    BufferLimit { limit: usize },
    TrailingData { bytes: usize },
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Header(error) => error.fmt(f),
            Self::BufferLimit { limit } => write!(f, "stream buffer exceeds {limit} bytes"),
            Self::TrailingData { bytes } => {
                write!(f, "{bytes} trailing bytes do not form a complete frame")
            }
        }
    }
}

impl std::error::Error for StreamError {}

impl From<HeaderError> for StreamError {
    fn from(value: HeaderError) -> Self {
        Self::Header(value)
    }
}

#[derive(Debug)]
pub enum DecodeError {
    InvalidInput(HeaderError),
    IncompleteFrame { needed: usize, available: usize },
    UnsupportedBackend,
    CrcMismatch { expected: u16, actual: u16 },
    ConfigurationChanged,
    ChannelCount { expected: usize, actual: usize },
    SampleCount { expected: usize, actual: usize },
    Metadata(MetadataError),
    HoaCore(HoaCoreDecodeError),
    McCore(McCoreDecodeError),
    MonoCore(MonoCoreDecodeError),
    StereoCore(StereoCoreDecodeError),
    Wav(WavError),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(error) => error.fmt(f),
            Self::IncompleteFrame { needed, available } => {
                write!(f, "incomplete frame: need {needed} bytes, have {available}")
            }
            Self::UnsupportedBackend => {
                f.write_str("the AVS3 synthesis backend has not been ported yet")
            }
            Self::CrcMismatch { expected, actual } => {
                write!(
                    f,
                    "payload CRC mismatch: expected 0x{expected:04x}, got 0x{actual:04x}"
                )
            }
            Self::ConfigurationChanged => {
                f.write_str("stream configuration changed; create or reset the decoder")
            }
            Self::ChannelCount { expected, actual } => {
                write!(
                    f,
                    "channel count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::SampleCount { expected, actual } => {
                write!(
                    f,
                    "sample count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::Metadata(error) => error.fmt(f),
            Self::HoaCore(error) => error.fmt(f),
            Self::McCore(error) => error.fmt(f),
            Self::MonoCore(error) => error.fmt(f),
            Self::StereoCore(error) => error.fmt(f),
            Self::Wav(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidInput(error) => Some(error),
            Self::Metadata(error) => Some(error),
            Self::HoaCore(error) => Some(error),
            Self::McCore(error) => Some(error),
            Self::MonoCore(error) => Some(error),
            Self::StereoCore(error) => Some(error),
            Self::Wav(error) => Some(error),
            _ => None,
        }
    }
}

impl From<HeaderError> for DecodeError {
    fn from(value: HeaderError) -> Self {
        Self::InvalidInput(value)
    }
}

impl From<WavError> for DecodeError {
    fn from(value: WavError) -> Self {
        Self::Wav(value)
    }
}

impl From<MetadataError> for DecodeError {
    fn from(value: MetadataError) -> Self {
        Self::Metadata(value)
    }
}

impl From<HoaCoreDecodeError> for DecodeError {
    fn from(value: HoaCoreDecodeError) -> Self {
        Self::HoaCore(value)
    }
}

impl From<MonoCoreDecodeError> for DecodeError {
    fn from(value: MonoCoreDecodeError) -> Self {
        Self::MonoCore(value)
    }
}

impl From<McCoreDecodeError> for DecodeError {
    fn from(value: McCoreDecodeError) -> Self {
        Self::McCore(value)
    }
}

impl From<StereoCoreDecodeError> for DecodeError {
    fn from(value: StereoCoreDecodeError) -> Self {
        Self::StereoCore(value)
    }
}

#[derive(Debug)]
pub enum Mp4Error {
    Io(std::io::Error),
    NeedMoreData {
        needed: u64,
        available: u64,
    },
    Truncated {
        context: &'static str,
        needed: u64,
        available: u64,
    },
    InvalidBoxSize {
        kind: [u8; 4],
        size: u64,
    },
    BoxTooLarge {
        kind: [u8; 4],
        size: u64,
        limit: u64,
    },
    MissingBox {
        kind: [u8; 4],
    },
    UnsupportedVersion {
        kind: [u8; 4],
        version: u8,
    },
    NoAv3aTrack,
    InvalidSampleTable(&'static str),
    InconsistentIndex {
        declared: usize,
        indexed: usize,
    },
    TooManySamples {
        count: usize,
        limit: usize,
    },
    SampleTooLarge {
        index: usize,
        size: usize,
        limit: usize,
    },
    SampleOutOfRange {
        index: usize,
        count: usize,
    },
    FrameExceedsSample {
        index: usize,
        frame_len: usize,
        sample_size: usize,
    },
    ArithmeticOverflow,
    Header(HeaderError),
}

impl fmt::Display for Mp4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::NeedMoreData { needed, available } => write!(
                f,
                "need the first {needed} bytes of the file to read its metadata, {available} available"
            ),
            Self::Truncated {
                context,
                needed,
                available,
            } => write!(
                f,
                "`{context}` box ends early: need {needed} bytes, {available} available"
            ),
            Self::InvalidBoxSize { kind, size } => write!(
                f,
                "`{}` box declares an invalid size of {size} bytes",
                kind.escape_ascii()
            ),
            Self::BoxTooLarge { kind, size, limit } => write!(
                f,
                "`{}` box is {size} bytes; maximum supported size is {limit}",
                kind.escape_ascii()
            ),
            Self::MissingBox { kind } => {
                write!(f, "required `{}` box is missing", kind.escape_ascii())
            }
            Self::UnsupportedVersion { kind, version } => write!(
                f,
                "unsupported `{}` box version {version}",
                kind.escape_ascii()
            ),
            Self::NoAv3aTrack => f.write_str("the file has no AV3A track"),
            Self::InvalidSampleTable(reason) => write!(f, "invalid sample table: {reason}"),
            Self::InconsistentIndex { declared, indexed } => write!(
                f,
                "sample table places {indexed} of the {declared} declared samples"
            ),
            Self::TooManySamples { count, limit } => write!(
                f,
                "track has {count} samples; maximum supported count is {limit}"
            ),
            Self::SampleTooLarge { index, size, limit } => write!(
                f,
                "sample {index} is {size} bytes; maximum supported size is {limit}"
            ),
            Self::SampleOutOfRange { index, count } => {
                write!(f, "sample {index} is out of range for {count} samples")
            }
            Self::FrameExceedsSample {
                index,
                frame_len,
                sample_size,
            } => write!(
                f,
                "sample {index} holds {sample_size} bytes but its header declares a {frame_len}-byte frame"
            ),
            Self::ArithmeticOverflow => f.write_str("container offset arithmetic overflow"),
            Self::Header(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for Mp4Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Header(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Mp4Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<HeaderError> for Mp4Error {
    fn from(value: HeaderError) -> Self {
        Self::Header(value)
    }
}

#[derive(Debug)]
pub enum WavError {
    Io(std::io::Error),
    InvalidChannels(u16),
    InvalidSampleRate(u32),
    InvalidSampleCount { channels: usize, samples: usize },
    SizeOverflow,
    NotFinalized,
}

impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::InvalidChannels(value) => write!(f, "invalid channel count {value}"),
            Self::InvalidSampleRate(value) => write!(f, "invalid sample rate {value}"),
            Self::InvalidSampleCount { channels, samples } => {
                write!(
                    f,
                    "{samples} samples are not divisible by {channels} channels"
                )
            }
            Self::SizeOverflow => f.write_str("WAV data exceeds the RIFF 32-bit size limit"),
            Self::NotFinalized => {
                f.write_str("WAV writer must be finalized before use is complete")
            }
        }
    }
}

impl std::error::Error for WavError {}

impl From<std::io::Error> for WavError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
