use core::fmt;

use crate::bitstream::BitReader;
use crate::core_side::{
    BweConfig, CoreBitstreamConfig, CoreBitstreamError, CoreSideInfo, LsfCodebookMode,
    ParsedNeuralQc, WindowGrouping, parse_core_side_prefix, parse_grouping, parse_neural_qc,
    qc_side_bits,
};
use crate::error::BitstreamError;
use crate::header::{ChannelConfig, CodecProfile, FrameHeader, SoundBedType};
use crate::mcr::{McrError, McrSideInfo};
use crate::neural_qc::MAX_QC_BITSTREAM_BYTES;

pub const STEREO_CHANNELS: usize = 2;
pub const STEREO_MCR_BITRATE_THRESHOLD: u32 = 32_000;
const ENERGY_BALANCE_RANGE: f32 = 16.0;
const BITS_SPLIT_RANGE: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoCodingMode {
    MidSide,
    Mcr,
}

impl StereoCodingMode {
    /// Select the mode fixed by the reference decoder at configuration time.
    pub const fn for_bitrate(total_bitrate: u32) -> Self {
        if total_bitrate <= STEREO_MCR_BITRATE_THRESHOLD {
            Self::Mcr
        } else {
            Self::MidSide
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StereoSideInfo {
    mid_side: bool,
    ild: Option<u8>,
    bits_ratio: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StereoFrameSideInfo<'decoder> {
    cores: [CoreSideInfo; STEREO_CHANNELS],
    stereo: StereoSideInfo,
    neural_qc: [ParsedNeuralQc<'decoder>; STEREO_CHANNELS],
    entropy_bytes: [usize; STEREO_CHANNELS],
    consumed_bits: usize,
    padding_bits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McrFrameSideInfo<'decoder> {
    cores: [CoreSideInfo; STEREO_CHANNELS],
    mcr: McrSideInfo,
    neural_qc: ParsedNeuralQc<'decoder>,
    entropy_bytes: usize,
    consumed_bits: usize,
    padding_bits: usize,
}

impl<'decoder> McrFrameSideInfo<'decoder> {
    pub fn cores(self) -> [CoreSideInfo; STEREO_CHANNELS] {
        self.cores
    }

    pub fn mcr(self) -> McrSideInfo {
        self.mcr
    }

    pub fn neural_qc(self) -> ParsedNeuralQc<'decoder> {
        self.neural_qc
    }

    pub fn entropy_bytes(self) -> usize {
        self.entropy_bytes
    }

    pub fn consumed_bits(self) -> usize {
        self.consumed_bits
    }

    pub fn padding_bits(self) -> usize {
        self.padding_bits
    }
}

impl<'decoder> StereoFrameSideInfo<'decoder> {
    pub fn cores(self) -> [CoreSideInfo; STEREO_CHANNELS] {
        self.cores
    }

    pub fn stereo(self) -> StereoSideInfo {
        self.stereo
    }

    pub fn neural_qc(self) -> [ParsedNeuralQc<'decoder>; STEREO_CHANNELS] {
        self.neural_qc
    }

    pub fn entropy_bytes(self) -> [usize; STEREO_CHANNELS] {
        self.entropy_bytes
    }

    pub fn consumed_bits(self) -> usize {
        self.consumed_bits
    }

    pub fn padding_bits(self) -> usize {
        self.padding_bits
    }
}

/// Allocation-stable parser for the ordinary stereo payload syntax.
///
/// AVS3 places both core prefixes first, then both grouping records, stereo
/// mode bits, and finally the two neural QC records. Separate fixed buffers
/// preserve that wire order while allowing the returned QC views to borrow
/// each channel's entropy bytes without per-frame allocation.
#[derive(Debug, Clone)]
pub struct StereoSideInfoDecoder {
    context: [[u8; MAX_QC_BITSTREAM_BYTES]; STEREO_CHANNELS],
    base: [[u8; MAX_QC_BITSTREAM_BYTES]; STEREO_CHANNELS],
}

impl StereoSideInfoDecoder {
    pub fn new() -> Self {
        Self {
            context: [[0; MAX_QC_BITSTREAM_BYTES]; STEREO_CHANNELS],
            base: [[0; MAX_QC_BITSTREAM_BYTES]; STEREO_CHANNELS],
        }
    }

    pub fn parse<'decoder>(
        &'decoder mut self,
        payload: &[u8],
        header: &FrameHeader,
    ) -> Result<StereoFrameSideInfo<'decoder>, StereoError> {
        if !is_stereo_coded_header(header) {
            return Err(StereoError::NotStereo {
                profile: header.profile,
                channel_config: header.channel_config,
                channels: header.channels,
            });
        }
        let actual_mode = StereoCodingMode::for_bitrate(header.bitrate);
        if actual_mode != StereoCodingMode::MidSide {
            return Err(StereoError::UnexpectedCodingMode {
                expected: StereoCodingMode::MidSide,
                actual: actual_mode,
                bitrate: header.bitrate,
            });
        }
        let available_bits = payload.len().saturating_mul(8);
        if header.payload_bits > available_bits {
            return Err(CoreBitstreamError::PayloadTooShort {
                declared_bits: header.payload_bits,
                available_bits,
            }
            .into());
        }

        let lsf_mode = if header.bitrate / STEREO_CHANNELS as u32 > 32_000 {
            LsfCodebookMode::HighBitrate
        } else {
            LsfCodebookMode::LowBitrate
        };
        let config = CoreBitstreamConfig::new(
            header.nn_type,
            header.payload_bits,
            lsf_mode,
            BweConfig::for_stereo_bitrate(header.bitrate)?,
        )?;
        let mut reader = BitReader::with_bit_len(payload, header.payload_bits)?;

        let left_prefix = parse_core_side_prefix(&mut reader, config)?;
        let right_prefix = parse_core_side_prefix(&mut reader, config)?;
        let left_grouping = parse_grouping(&mut reader, left_prefix.transform_type())?;
        let right_grouping = parse_grouping(&mut reader, right_prefix.transform_type())?;
        let stereo = StereoSideInfo::parse(&mut reader)?;

        let left_reserved = qc_side_bits(header.nn_type, left_grouping.count())?;
        let right_reserved = qc_side_bits(header.nn_type, right_grouping.count())?;
        let reserved_bits = left_reserved
            .checked_add(right_reserved)
            .ok_or(CoreBitstreamError::IntegerOverflow)?;
        let used_and_reserved = reader
            .position()
            .checked_add(reserved_bits)
            .ok_or(CoreBitstreamError::IntegerOverflow)?;
        let available_qc_bits = header.payload_bits.checked_sub(used_and_reserved).ok_or(
            CoreBitstreamError::QcBudgetUnderflow {
                payload_bits: header.payload_bits,
                used_bits: reader.position(),
                reserved_bits,
            },
        )?;
        let entropy_bytes = stereo_bytes_allocation(available_qc_bits, stereo.bits_ratio())?;

        let [left_context, right_context] = &mut self.context;
        let [left_base, right_base] = &mut self.base;
        let left_qc = parse_neural_qc(
            &mut reader,
            config,
            left_grouping,
            entropy_bytes[0],
            left_context,
            left_base,
        )?;
        let right_qc = parse_neural_qc(
            &mut reader,
            config,
            right_grouping,
            entropy_bytes[1],
            right_context,
            right_base,
        )?;

        let consumed_bits = reader.position();
        let padding_bits = reader.remaining();
        debug_assert!(padding_bits < 8);
        Ok(StereoFrameSideInfo {
            cores: [
                left_prefix.finish(left_grouping),
                right_prefix.finish(right_grouping),
            ],
            stereo,
            neural_qc: [left_qc, right_qc],
            entropy_bytes,
            consumed_bits,
            padding_bits,
        })
    }

    pub fn parse_mcr<'decoder>(
        &'decoder mut self,
        payload: &[u8],
        header: &FrameHeader,
    ) -> Result<McrFrameSideInfo<'decoder>, StereoError> {
        if !is_stereo_coded_header(header) {
            return Err(StereoError::NotStereo {
                profile: header.profile,
                channel_config: header.channel_config,
                channels: header.channels,
            });
        }
        let actual_mode = StereoCodingMode::for_bitrate(header.bitrate);
        if actual_mode != StereoCodingMode::Mcr {
            return Err(StereoError::UnexpectedCodingMode {
                expected: StereoCodingMode::Mcr,
                actual: actual_mode,
                bitrate: header.bitrate,
            });
        }

        let available_bits = payload.len().saturating_mul(8);
        if header.payload_bits > available_bits {
            return Err(CoreBitstreamError::PayloadTooShort {
                declared_bits: header.payload_bits,
                available_bits,
            }
            .into());
        }

        let lsf_mode = if header.bitrate / STEREO_CHANNELS as u32 > 32_000 {
            LsfCodebookMode::HighBitrate
        } else {
            LsfCodebookMode::LowBitrate
        };
        let config = CoreBitstreamConfig::new(
            header.nn_type,
            header.payload_bits,
            lsf_mode,
            BweConfig::for_stereo_bitrate(header.bitrate)?,
        )?;
        let mut reader = BitReader::with_bit_len(payload, header.payload_bits)?;

        let left_prefix = parse_core_side_prefix(&mut reader, config)?;
        let right_prefix = parse_core_side_prefix(&mut reader, config)?;
        let left_grouping = parse_grouping(&mut reader, left_prefix.transform_type())?;
        let mcr = McrSideInfo::parse(&mut reader, left_prefix.transform_type())?;
        let reserved_bits = qc_side_bits(header.nn_type, left_grouping.count())?;
        let used_and_reserved = reader
            .position()
            .checked_add(reserved_bits)
            .ok_or(CoreBitstreamError::IntegerOverflow)?;
        let available_qc_bits = header.payload_bits.checked_sub(used_and_reserved).ok_or(
            CoreBitstreamError::QcBudgetUnderflow {
                payload_bits: header.payload_bits,
                used_bits: reader.position(),
                reserved_bits,
            },
        )?;
        let entropy_bytes = available_qc_bits / 8;
        if entropy_bytes > MAX_QC_BITSTREAM_BYTES {
            return Err(CoreBitstreamError::EntropyPayloadTooLarge {
                bytes: entropy_bytes,
                limit: MAX_QC_BITSTREAM_BYTES,
            }
            .into());
        }

        let [left_context, _right_context] = &mut self.context;
        let [left_base, _right_base] = &mut self.base;
        let left_qc = parse_neural_qc(
            &mut reader,
            config,
            left_grouping,
            entropy_bytes,
            left_context,
            left_base,
        )?;

        let consumed_bits = reader.position();
        let padding_bits = reader.remaining();
        debug_assert!(padding_bits < 8);
        Ok(McrFrameSideInfo {
            cores: [
                left_prefix.finish(left_grouping),
                right_prefix.finish(WindowGrouping::single()),
            ],
            mcr,
            neural_qc: left_qc,
            entropy_bytes,
            consumed_bits,
            padding_bits,
        })
    }
}

fn is_stereo_coded_header(header: &FrameHeader) -> bool {
    if header.channels != STEREO_CHANNELS as u8 || header.has_lfe {
        return false;
    }
    let channel_stereo = header.profile == CodecProfile::ChannelBased
        && header.channel_config == Some(ChannelConfig::Stereo)
        && header.sound_bed_type.is_none()
        && header.objects == 0
        && header.bed_channels == STEREO_CHANNELS as u8
        && header.bed_bitrate == Some(header.bitrate)
        && header.object_bitrate.is_none();
    let object_stereo = header.profile == CodecProfile::Mixed
        && header.channel_config.is_none()
        && header.sound_bed_type == Some(SoundBedType::ObjectsOnly)
        && header.objects == STEREO_CHANNELS as u8
        && header.bed_channels == 0
        && header.bed_bitrate.is_none()
        && header
            .object_bitrate
            .and_then(|bitrate| bitrate.checked_mul(STEREO_CHANNELS as u32))
            == Some(header.bitrate);
    channel_stereo || object_stereo
}

impl Default for StereoSideInfoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StereoSideInfo {
    /// Parse the ordinary (non-MCR) stereo mode bits at the reader's current
    /// position: one MS flag, an optional four-bit ILD, and a three-bit split.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, StereoError> {
        let mid_side = reader.read_u8(1)? != 0;
        let ild = mid_side.then(|| reader.read_u8(4)).transpose()?;
        let bits_ratio = reader.read_u8(3)?;
        Ok(Self {
            mid_side,
            ild,
            bits_ratio,
        })
    }

    pub fn mid_side(self) -> bool {
        self.mid_side
    }

    pub fn ild(self) -> Option<u8> {
        self.ild
    }

    pub fn bits_ratio(self) -> u8 {
        self.bits_ratio
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StereoError {
    NotStereo {
        profile: CodecProfile,
        channel_config: Option<ChannelConfig>,
        channels: u8,
    },
    UnexpectedCodingMode {
        expected: StereoCodingMode,
        actual: StereoCodingMode,
        bitrate: u32,
    },
    InvalidBitsRatio(u8),
    InvalidIld(u8),
    ChannelLengthMismatch {
        left: usize,
        right: usize,
    },
    AllocationOverflow,
    Mcr(McrError),
    Bitstream(BitstreamError),
    CoreBitstream(CoreBitstreamError),
}

impl fmt::Display for StereoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotStereo {
                profile,
                channel_config,
                channels,
            } => write!(
                f,
                "stereo parser requires channel-based stereo; got profile {profile:?}, configuration {channel_config:?}, {channels} channels"
            ),
            Self::UnexpectedCodingMode {
                expected,
                actual,
                bitrate,
            } => write!(
                f,
                "stereo bitrate {bitrate} selects {actual:?} coding, but this parser expects {expected:?}"
            ),
            Self::InvalidBitsRatio(value) => {
                write!(
                    f,
                    "stereo bit-split ratio {value} does not fit its three-bit field"
                )
            }
            Self::InvalidIld(value) => {
                write!(f, "stereo ILD {value} is invalid; expected 1..=15")
            }
            Self::ChannelLengthMismatch { left, right } => write!(
                f,
                "stereo spectra have different lengths: left {left}, right {right}"
            ),
            Self::AllocationOverflow => f.write_str("stereo byte allocation overflow"),
            Self::Mcr(error) => error.fmt(f),
            Self::Bitstream(error) => error.fmt(f),
            Self::CoreBitstream(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for StereoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(error) => Some(error),
            Self::Mcr(error) => Some(error),
            Self::CoreBitstream(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BitstreamError> for StereoError {
    fn from(value: BitstreamError) -> Self {
        Self::Bitstream(value)
    }
}

impl From<CoreBitstreamError> for StereoError {
    fn from(value: CoreBitstreamError) -> Self {
        Self::CoreBitstream(value)
    }
}

impl From<McrError> for StereoError {
    fn from(value: McrError) -> Self {
        Self::Mcr(value)
    }
}

/// Reproduce `StereoBitsAllocation` without its narrowing `short` arithmetic.
///
/// AVS3 first discards incomplete bytes, then quantizes the left share in
/// eighths of the remaining whole-byte budget. Any remainder goes right.
pub fn stereo_bytes_allocation(
    available_bits: usize,
    bits_ratio: u8,
) -> Result<[usize; STEREO_CHANNELS], StereoError> {
    if usize::from(bits_ratio) >= BITS_SPLIT_RANGE {
        return Err(StereoError::InvalidBitsRatio(bits_ratio));
    }
    let available_bytes = available_bits / 8;
    let left = usize::from(bits_ratio)
        .checked_mul(available_bytes / BITS_SPLIT_RANGE)
        .ok_or(StereoError::AllocationOverflow)?;
    let right = available_bytes
        .checked_sub(left)
        .ok_or(StereoError::AllocationOverflow)?;
    Ok([left, right])
}

/// Apply the decoder's inverse MS matrix followed by inverse ILD balancing.
///
/// The loop order and `f32` arithmetic match `StereoInvMsProcess`. The standard
/// library's correctly rounded `FRAC_1_SQRT_2` avoids a per-frame `sqrt` call.
pub fn inverse_mid_side(left: &mut [f32], right: &mut [f32], ild: u8) -> Result<(), StereoError> {
    if left.len() != right.len() {
        return Err(StereoError::ChannelLengthMismatch {
            left: left.len(),
            right: right.len(),
        });
    }
    if !(1..=15).contains(&ild) {
        return Err(StereoError::InvalidIld(ild));
    }

    for (left_value, right_value) in left.iter_mut().zip(right.iter_mut()) {
        let original_left = *left_value;
        let original_right = *right_value;
        *left_value = core::f32::consts::FRAC_1_SQRT_2 * (original_left + original_right);
        *right_value = core::f32::consts::FRAC_1_SQRT_2 * (original_left - original_right);
    }

    let energy_relation = ENERGY_BALANCE_RANGE / f32::from(ild) - 1.0;
    if energy_relation > 1.0 {
        for value in right {
            *value *= energy_relation;
        }
    } else if energy_relation < 1.0 {
        let inverse_relation = 1.0 / energy_relation;
        for value in left {
            *value *= inverse_relation;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;
    use crate::core_side::{BweWhiteningLevel, TransformType};
    use crate::header::{AudioCodecId, BitDepth, NnType};

    const FRAME_LEN: usize = 1_024;

    fn deterministic_spectra() -> ([f32; FRAME_LEN], [f32; FRAME_LEN]) {
        let mut left = [0.0_f32; FRAME_LEN];
        let mut right = [0.0_f32; FRAME_LEN];
        let mut state = 0xa511_e9b3_u32;
        for (left_value, right_value) in left.iter_mut().zip(&mut right) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let bits = ((state & 1) << 31) | 0x4300_0000 | ((state >> 1) & 0x007f_ffff);
            *left_value = f32::from_bits(bits);

            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let bits = ((state & 1) << 31) | 0x4280_0000 | ((state >> 1) & 0x007f_ffff);
            *right_value = f32::from_bits(bits);
        }
        (left, right)
    }

    fn fingerprint(values: &[f32]) -> u64 {
        values
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
                (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3)
            })
    }

    fn stereo_header(bitrate: u32, payload_bits: usize) -> FrameHeader {
        FrameHeader {
            codec_id: AudioCodecId::Avs3P3,
            nn_type: NnType::Main,
            profile: CodecProfile::ChannelBased,
            sample_rate: 48_000,
            bit_depth: BitDepth::Sixteen,
            channel_config: Some(ChannelConfig::Stereo),
            sound_bed_type: None,
            hoa_order: None,
            objects: 0,
            bed_channels: 2,
            channels: 2,
            has_lfe: false,
            bed_bitrate: Some(bitrate),
            object_bitrate: None,
            bitrate,
            crc: 0,
            header_len: 7,
            payload_bits,
            payload_len: payload_bits.div_ceil(8),
            frame_len: 7 + payload_bits.div_ceil(8),
            samples_per_channel: FRAME_LEN as u32,
        }
    }

    fn write_lbr_core_prefix(
        writer: &mut BitWriter,
        transform: u64,
        lsf: [u64; 5],
        envelopes: [u64; 6],
        whitening: [BweWhiteningLevel; 3],
    ) {
        writer.write_bits(transform, 2).unwrap();
        for (value, width) in lsf.into_iter().zip([8, 8, 7, 7, 6]) {
            writer.write_bits(value, width).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        for value in envelopes {
            writer.write_bits(value, 7).unwrap();
        }
        for level in whitening {
            match level {
                BweWhiteningLevel::Off => writer.write_bits(0, 1).unwrap(),
                BweWhiteningLevel::Mid => writer.write_bits(0b10, 2).unwrap(),
                BweWhiteningLevel::High => writer.write_bits(0b11, 2).unwrap(),
            }
        }
    }

    fn ordinary_stereo_payload() -> (FrameHeader, Vec<u8>, [usize; 2], usize) {
        let payload_bits = 1_309;
        let header = stereo_header(64_000, payload_bits);
        let mut writer = BitWriter::new();
        write_lbr_core_prefix(
            &mut writer,
            1,
            [3, 5, 7, 9, 11],
            [1, 2, 3, 4, 5, 6],
            [
                BweWhiteningLevel::Off,
                BweWhiteningLevel::Mid,
                BweWhiteningLevel::High,
            ],
        );
        write_lbr_core_prefix(
            &mut writer,
            2,
            [17, 19, 21, 23, 25],
            [127, 100, 80, 60, 40, 20],
            [
                BweWhiteningLevel::High,
                BweWhiteningLevel::Off,
                BweWhiteningLevel::Mid,
            ],
        );

        writer.write_bits(1, 1).unwrap();
        for indicator in [0, 0, 0, 1, 1, 1, 1, 1] {
            writer.write_bits(indicator, 1).unwrap();
        }
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(5, 4).unwrap();
        writer.write_bits(4, 3).unwrap();

        let mode_end = writer.bit_len();
        let reserved_qc_bits = 22 + 19;
        let entropy_bytes =
            stereo_bytes_allocation(payload_bits - mode_end - reserved_qc_bits, 4).unwrap();
        assert_eq!(mode_end, 191);
        assert_eq!(entropy_bytes, [64, 70]);

        writer.write_bits(1, 1).unwrap();
        writer.write_bits(37, 7).unwrap();
        writer.write_bits(3, 3).unwrap();
        writer.write_bits(7, 3).unwrap();
        let left_context: [u8; 6] = [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7];
        writer.write_bits(left_context.len() as u64, 8).unwrap();
        for byte in left_context {
            writer.write_bits(u64::from(byte), 8).unwrap();
        }
        for index in left_context.len()..entropy_bytes[0] {
            writer
                .write_bits(u64::try_from(index & 0xff).unwrap(), 8)
                .unwrap();
        }

        writer.write_bits(0, 1).unwrap();
        writer.write_bits(91, 7).unwrap();
        writer.write_bits(5, 3).unwrap();
        let right_context: [u8; 2] = [0x12, 0x34];
        writer.write_bits(right_context.len() as u64, 8).unwrap();
        for byte in right_context {
            writer.write_bits(u64::from(byte), 8).unwrap();
        }
        for index in right_context.len()..entropy_bytes[1] {
            writer
                .write_bits(u64::try_from((index * 3) & 0xff).unwrap(), 8)
                .unwrap();
        }

        let consumed_bits = writer.bit_len();
        assert_eq!(consumed_bits, 1_304);
        assert!(payload_bits - consumed_bits < 8);
        let mut payload = writer.into_bytes();
        payload.resize(payload_bits.div_ceil(8), 0);
        (header, payload, entropy_bytes, consumed_bits)
    }

    fn mcr_stereo_payload() -> (FrameHeader, Vec<u8>, [[u16; 6]; 2], usize) {
        let payload_bits = 626;
        let header = stereo_header(32_000, payload_bits);
        let mut writer = BitWriter::new();
        write_lbr_core_prefix(
            &mut writer,
            1,
            [3, 5, 7, 9, 11],
            [1, 2, 3, 4, 5, 6],
            [
                BweWhiteningLevel::Off,
                BweWhiteningLevel::Mid,
                BweWhiteningLevel::High,
            ],
        );
        write_lbr_core_prefix(
            &mut writer,
            1,
            [17, 19, 21, 23, 25],
            [7, 8, 9, 10, 11, 12],
            [
                BweWhiteningLevel::High,
                BweWhiteningLevel::Off,
                BweWhiteningLevel::Mid,
            ],
        );

        writer.write_bits(1, 1).unwrap();
        for indicator in [0, 0, 0, 1, 1, 1, 1, 1] {
            writer.write_bits(indicator, 1).unwrap();
        }
        let indexes = [[1_u16, 2, 3, 4, 5, 255], [255_u16, 5, 4, 3, 2, 1]];
        for subvector in 0..6 {
            for subspectrum in &indexes {
                writer
                    .write_bits(u64::from(subspectrum[subvector]), 8)
                    .unwrap();
            }
        }
        assert_eq!(writer.bit_len(), 279);

        let reserved_qc_bits = 22;
        let entropy_bytes = (payload_bits - writer.bit_len() - reserved_qc_bits) / 8;
        assert_eq!(entropy_bytes, 40);
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(37, 7).unwrap();
        writer.write_bits(3, 3).unwrap();
        writer.write_bits(7, 3).unwrap();
        let context: [u8; 6] = [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7];
        writer.write_bits(context.len() as u64, 8).unwrap();
        for byte in context {
            writer.write_bits(u64::from(byte), 8).unwrap();
        }
        for index in context.len()..entropy_bytes {
            writer
                .write_bits(u64::try_from(index & 0xff).unwrap(), 8)
                .unwrap();
        }

        let consumed_bits = writer.bit_len();
        assert_eq!(consumed_bits, 621);
        let mut payload = writer.into_bytes();
        payload.resize(payload_bits.div_ceil(8), 0);
        (header, payload, indexes, consumed_bits)
    }

    #[test]
    fn selects_mcr_at_the_reference_threshold() {
        assert_eq!(StereoCodingMode::for_bitrate(24_000), StereoCodingMode::Mcr);
        assert_eq!(StereoCodingMode::for_bitrate(32_000), StereoCodingMode::Mcr);
        assert_eq!(
            StereoCodingMode::for_bitrate(32_001),
            StereoCodingMode::MidSide
        );
    }

    #[test]
    fn stereo_bwe_configs_match_reference_tables() {
        let low = BweConfig::for_stereo_bitrate(64_000).unwrap().unwrap();
        assert_eq!(low.target_tiles(), [352, 480, 608, 768]);
        assert_eq!(low.source_tiles(), [64, 96, 144]);
        assert_eq!(
            low.scale_factor_bands(),
            [352, 416, 480, 544, 608, 672, 768]
        );

        let middle = BweConfig::for_stereo_bitrate(80_000).unwrap().unwrap();
        assert_eq!(middle.target_tiles(), [544, 672, 832]);
        assert_eq!(middle.source_tiles(), [144, 192]);

        let high = BweConfig::for_stereo_bitrate(128_000).unwrap().unwrap();
        assert_eq!(high.target_tiles(), [672, 832]);
        assert_eq!(high.source_tiles(), [192]);
        assert_eq!(BweConfig::for_stereo_bitrate(144_000).unwrap(), None);
    }

    #[test]
    fn parses_complete_ordinary_stereo_side_information_in_c_wire_order() {
        let (header, payload, expected_entropy_bytes, consumed_bits) = ordinary_stereo_payload();
        let mut decoder = StereoSideInfoDecoder::new();
        let parsed = decoder.parse(&payload, &header).unwrap();
        assert_eq!(parsed.entropy_bytes(), expected_entropy_bytes);
        assert_eq!(parsed.consumed_bits(), consumed_bits);
        assert_eq!(parsed.padding_bits(), header.payload_bits - consumed_bits);

        let [left, right] = parsed.cores();
        assert_eq!(left.transform_type(), TransformType::Short);
        assert_eq!(left.lsf().mode(), LsfCodebookMode::LowBitrate);
        assert_eq!(left.lsf().indexes(), [3, 5, 7, 9, 11]);
        assert_eq!(left.grouping().count(), 2);
        assert_eq!(left.bwe().unwrap().envelope_indexes(), [1, 2, 3, 4, 5, 6]);
        assert_eq!(
            left.bwe().unwrap().whitening_levels(),
            [
                BweWhiteningLevel::Off,
                BweWhiteningLevel::Mid,
                BweWhiteningLevel::High,
            ]
        );
        assert_eq!(right.transform_type(), TransformType::LongToShort);
        assert_eq!(right.lsf().indexes(), [17, 19, 21, 23, 25]);
        assert_eq!(right.grouping().count(), 1);

        let stereo = parsed.stereo();
        assert!(stereo.mid_side());
        assert_eq!(stereo.ild(), Some(5));
        assert_eq!(stereo.bits_ratio(), 4);

        let [left_qc, right_qc] = parsed.neural_qc();
        assert_eq!(
            left_qc.bitstreams().context(),
            [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7]
        );
        assert_eq!(
            left_qc.bitstreams().base().len(),
            expected_entropy_bytes[0] - 6
        );
        assert_eq!(right_qc.bitstreams().context(), [0x12, 0x34]);
        assert_eq!(
            right_qc.bitstreams().base().len(),
            expected_entropy_bytes[1] - 2
        );
    }

    #[test]
    fn complete_stereo_parser_rejects_truncation_and_wrong_coding_mode() {
        let (header, payload, _, _) = ordinary_stereo_payload();
        let mut decoder = StereoSideInfoDecoder::new();
        for end in 0..payload.len() {
            assert!(
                decoder.parse(&payload[..end], &header).is_err(),
                "prefix {end}"
            );
        }

        let mcr_header = stereo_header(32_000, header.payload_bits);
        assert_eq!(
            decoder.parse(&payload, &mcr_header),
            Err(StereoError::UnexpectedCodingMode {
                expected: StereoCodingMode::MidSide,
                actual: StereoCodingMode::Mcr,
                bitrate: 32_000,
            })
        );
        assert!(matches!(
            decoder.parse_mcr(&payload, &header),
            Err(StereoError::UnexpectedCodingMode {
                expected: StereoCodingMode::Mcr,
                actual: StereoCodingMode::MidSide,
                bitrate: 64_000,
            })
        ));
    }

    #[test]
    fn parses_complete_short_mcr_side_information_in_c_wire_order() {
        let (header, payload, expected_indexes, consumed_bits) = mcr_stereo_payload();
        let mut decoder = StereoSideInfoDecoder::new();
        let parsed = decoder.parse_mcr(&payload, &header).unwrap();

        assert_eq!(parsed.entropy_bytes(), 40);
        assert_eq!(parsed.consumed_bits(), consumed_bits);
        assert_eq!(parsed.padding_bits(), 5);
        assert_eq!(parsed.mcr().vq_indexes(), &expected_indexes);
        assert!(parsed.mcr().short_window());

        let [left, right] = parsed.cores();
        assert_eq!(left.transform_type(), TransformType::Short);
        assert_eq!(left.grouping().count(), 2);
        assert_eq!(right.transform_type(), TransformType::Short);
        assert_eq!(right.grouping().count(), 1);
        assert_eq!(right.lsf().indexes(), [17, 19, 21, 23, 25]);
        assert_eq!(
            parsed.neural_qc().bitstreams().context(),
            [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7]
        );
        assert_eq!(parsed.neural_qc().bitstreams().base().len(), 34);
    }

    #[test]
    fn parses_both_ordinary_stereo_side_bit_shapes() {
        let mut writer = BitWriter::new();
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(5, 4).unwrap();
        writer.write_bits(7, 3).unwrap();
        writer.write_bits(0b101, 3).unwrap();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        let side = StereoSideInfo::parse(&mut reader).unwrap();
        assert!(side.mid_side());
        assert_eq!(side.ild(), Some(5));
        assert_eq!(side.bits_ratio(), 7);
        assert_eq!(reader.position(), 8);
        assert_eq!(reader.read_bits(3).unwrap(), 0b101);

        let mut reader = BitReader::with_bit_len(&[0b0011_0000], 4).unwrap();
        let side = StereoSideInfo::parse(&mut reader).unwrap();
        assert!(!side.mid_side());
        assert_eq!(side.ild(), None);
        assert_eq!(side.bits_ratio(), 3);
        assert_eq!(reader.remaining(), 0);
    }

    #[test]
    fn allocation_matches_c_reference_vectors() {
        assert_eq!(stereo_bytes_allocation(7, 7).unwrap(), [0, 0]);
        assert_eq!(stereo_bytes_allocation(8, 7).unwrap(), [0, 1]);
        assert_eq!(stereo_bytes_allocation(64, 0).unwrap(), [0, 8]);
        assert_eq!(stereo_bytes_allocation(64, 1).unwrap(), [1, 7]);
        assert_eq!(stereo_bytes_allocation(64, 4).unwrap(), [4, 4]);
        assert_eq!(stereo_bytes_allocation(626, 7).unwrap(), [63, 15]);
        assert_eq!(stereo_bytes_allocation(1_309, 4).unwrap(), [80, 83]);
        assert_eq!(stereo_bytes_allocation(4_095, 7).unwrap(), [441, 70]);
        assert_eq!(
            stereo_bytes_allocation(64, 8),
            Err(StereoError::InvalidBitsRatio(8))
        );
    }

    #[test]
    fn inverse_ms_and_ild_are_bit_exact_with_c() {
        let expected = [
            (1, 0xf35a_5f27_b7f1_156f, 0x91c7_7e42_c16c_f886),
            (5, 0xf35a_5f27_b7f1_156f, 0xf010_ee41_26e1_6024),
            (8, 0xf35a_5f27_b7f1_156f, 0x01f8_8138_e738_f11f),
            (15, 0x2dd9_cc79_5064_54c0, 0x01f8_8138_e738_f11f),
        ];
        for (ild, expected_left, expected_right) in expected {
            let (mut left, mut right) = deterministic_spectra();
            inverse_mid_side(&mut left, &mut right, ild).unwrap();
            assert_eq!(fingerprint(&left), expected_left, "left ILD {ild}");
            assert_eq!(fingerprint(&right), expected_right, "right ILD {ild}");
        }
    }

    #[test]
    fn inverse_ms_rejects_invalid_input_without_mutating_it() {
        let mut left = [1.0_f32, 2.0];
        let mut short_right = [3.0_f32];
        assert_eq!(
            inverse_mid_side(&mut left, &mut short_right, 5),
            Err(StereoError::ChannelLengthMismatch { left: 2, right: 1 })
        );
        assert_eq!(left, [1.0, 2.0]);
        assert_eq!(short_right, [3.0]);

        let mut right = [3.0_f32, 4.0];
        assert_eq!(
            inverse_mid_side(&mut left, &mut right, 0),
            Err(StereoError::InvalidIld(0))
        );
        assert_eq!(left, [1.0, 2.0]);
        assert_eq!(right, [3.0, 4.0]);
    }
}
