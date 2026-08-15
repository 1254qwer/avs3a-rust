use core::fmt;

use crate::bitstream::BitReader;
use crate::core_side::{
    BweConfig, CoreBitstreamConfig, CoreBitstreamError, CoreSideInfo, CoreSideInfoPrefix,
    LsfCodebookMode, ParsedNeuralQc, WindowGrouping, parse_core_side_prefix, parse_grouping,
    parse_neural_qc, qc_side_bits,
};
use crate::error::BitstreamError;
use crate::header::{ChannelConfig, CodecProfile, FrameHeader, MAX_CHANNELS};
use crate::mc::{MC_NO_ILD_INDEX, McError, apply_mc_ild};
use crate::model::AVS3_FEATURE_DIMENSIONS;
use crate::neural_qc::MAX_QC_BITSTREAM_BYTES;

pub const MAX_HOA_GROUPS: usize = 3;
pub const MAX_HOA_BASIS: usize = 4;
pub const MAX_HOA_GROUP_PAIRS: usize = MAX_CHANNELS as usize / 2;
pub const HOA_BASIS_INDEX_BITS: usize = 12;
pub const HOA_BASIS_TABLE_LEN: usize = 1_343;
pub const HOA_SFB_COUNT: usize = 21;

pub const HOA_SFB_BOUNDARIES: [usize; HOA_SFB_COUNT + 1] = [
    0, 8, 24, 40, 56, 72, 88, 104, 128, 160, 192, 224, 256, 288, 336, 384, 432, 480, 544, 608, 672,
    768,
];

const HOA_SCENE_TYPE_BITS: usize = 4;
const HOA_SPATIAL_ANALYSIS_BITS: usize = 1;
const HOA_VECTOR_CHANNEL_BITS: usize = 4;
const HOA_PAIR_COUNT_BITS: usize = 4;
const HOA_DMX_MODE_BITS: usize = 1;
const HOA_ILD_BITS: usize = 5;
const HOA_RATIO_BITS: usize = 4;
const HOA_RATIO_RANGE: usize = 1 << HOA_RATIO_BITS;
const LSF_LOW_BITRATE_THRESHOLD: u64 = 32_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HoaError {
    NotHoa {
        profile: CodecProfile,
        channel_config: Option<ChannelConfig>,
        hoa_order: Option<u8>,
        channels: u8,
    },
    UnsupportedBitrate {
        order: u8,
        bitrate: u32,
    },
    InvalidGroupChannelCount {
        channels: usize,
    },
    InvalidChannelIndex {
        index: usize,
        channels: usize,
    },
    InvalidVectorChannelCount {
        channels: usize,
        limit: usize,
    },
    InvalidBasisIndex {
        index: usize,
        limit: usize,
    },
    InvalidPairCount {
        group: usize,
        pairs: usize,
        limit: usize,
    },
    InvalidPairIndex {
        group: usize,
        index: usize,
        combinations: usize,
    },
    InvalidIldIndex(u8),
    SideInfoConfigurationMismatch,
    ZeroChannelRatio {
        group: usize,
    },
    GroupRatioExceedsBudget {
        group: usize,
        requested_bytes: usize,
        remaining_bytes: usize,
    },
    SpectrumChannelMismatch {
        expected: usize,
        actual: usize,
    },
    AllocationOverflow,
    Bitstream(BitstreamError),
    CoreBitstream(CoreBitstreamError),
    Mc(McError),
}

impl fmt::Display for HoaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotHoa {
                profile,
                channel_config,
                hoa_order,
                channels,
            } => write!(
                f,
                "HOA parser requires an HOA profile; got profile {profile:?}, configuration {channel_config:?}, order {hoa_order:?}, {channels} channels"
            ),
            Self::UnsupportedBitrate { order, bitrate } => {
                write!(f, "HOA order {order} does not support bitrate {bitrate}")
            }
            Self::InvalidGroupChannelCount { channels } => write!(
                f,
                "HOA group channel count {channels} is outside 2..={MAX_CHANNELS}"
            ),
            Self::InvalidChannelIndex { index, channels } => write!(
                f,
                "HOA channel index {index} is outside a {channels}-channel transport configuration"
            ),
            Self::InvalidVectorChannelCount { channels, limit } => write!(
                f,
                "HOA frame declares {channels} vector channels; valid range is 1..={limit}"
            ),
            Self::InvalidBasisIndex { index, limit } => write!(
                f,
                "HOA basis index {index} is outside the table of {limit} entries"
            ),
            Self::InvalidPairCount {
                group,
                pairs,
                limit,
            } => write!(
                f,
                "HOA group {group} declares {pairs} pairs; decoder limit is {limit}"
            ),
            Self::InvalidPairIndex {
                group,
                index,
                combinations,
            } => write!(
                f,
                "HOA group {group} pair index {index} is outside {combinations} channel combinations"
            ),
            Self::InvalidIldIndex(index) => {
                write!(f, "HOA ILD index {index} is outside 0..={MC_NO_ILD_INDEX}")
            }
            Self::SideInfoConfigurationMismatch => {
                f.write_str("HOA side information does not match the bitrate configuration")
            }
            Self::ZeroChannelRatio { group } => {
                write!(f, "HOA group {group} has a zero channel-ratio sum")
            }
            Self::GroupRatioExceedsBudget {
                group,
                requested_bytes,
                remaining_bytes,
            } => write!(
                f,
                "HOA group {group} requests {requested_bytes} bytes; only {remaining_bytes} remain"
            ),
            Self::SpectrumChannelMismatch { expected, actual } => write!(
                f,
                "HOA inverse DMX received {actual} spectra; expected {expected}"
            ),
            Self::AllocationOverflow => f.write_str("HOA byte allocation arithmetic overflow"),
            Self::Bitstream(error) => error.fmt(f),
            Self::CoreBitstream(error) => error.fmt(f),
            Self::Mc(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for HoaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(error) => Some(error),
            Self::CoreBitstream(error) => Some(error),
            Self::Mc(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BitstreamError> for HoaError {
    fn from(value: BitstreamError) -> Self {
        Self::Bitstream(value)
    }
}

impl From<CoreBitstreamError> for HoaError {
    fn from(value: CoreBitstreamError) -> Self {
        Self::CoreBitstream(value)
    }
}

impl From<McError> for HoaError {
    fn from(value: McError) -> Self {
        Self::Mc(value)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HoaGroupConfig {
    channels: usize,
    channel_offset: usize,
    pair_index_bits: usize,
    core_lines: usize,
    bwe_enabled: bool,
}

impl HoaGroupConfig {
    pub fn channels(self) -> usize {
        self.channels
    }

    pub fn channel_offset(self) -> usize {
        self.channel_offset
    }

    pub fn pair_index_bits(self) -> usize {
        self.pair_index_bits
    }

    pub fn core_lines(self) -> usize {
        self.core_lines
    }

    pub fn bwe_enabled(self) -> bool {
        self.bwe_enabled
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoaBitstreamConfig {
    order: u8,
    output_channels: usize,
    transport_channels: usize,
    foreground_channels: usize,
    residual_channels: usize,
    default_spatial_analysis: bool,
    groups: [HoaGroupConfig; MAX_HOA_GROUPS],
    group_count: usize,
}

impl HoaBitstreamConfig {
    pub fn for_header(header: &FrameHeader) -> Result<Self, HoaError> {
        let order = match header.channel_config {
            Some(ChannelConfig::Hoa1) => 1,
            Some(ChannelConfig::Hoa2) => 2,
            Some(ChannelConfig::Hoa3) => 3,
            _ => {
                return Err(HoaError::NotHoa {
                    profile: header.profile,
                    channel_config: header.channel_config,
                    hoa_order: header.hoa_order,
                    channels: header.channels,
                });
            }
        };
        let output_channels = usize::from((order + 1) * (order + 1));
        if header.profile != CodecProfile::Hoa
            || header.hoa_order != Some(order)
            || usize::from(header.channels) != output_channels
            || usize::from(header.bed_channels) != output_channels
            || header.objects != 0
            || header.has_lfe
        {
            return Err(HoaError::NotHoa {
                profile: header.profile,
                channel_config: header.channel_config,
                hoa_order: header.hoa_order,
                channels: header.channels,
            });
        }

        let (default_spatial_analysis, group_count, group_channels, core_lines, group_bwe) =
            match (order, header.bitrate) {
                (1, 48_000 | 96_000 | 128_000 | 192_000 | 256_000) => {
                    (false, 1, [4, 0, 0], [1_024, 0, 0], [true, false, false])
                }
                (2, 192_000) => (false, 1, [9, 0, 0], [352, 0, 0], [true, false, false]),
                (2, 256_000) => (false, 1, [9, 0, 0], [384, 0, 0], [true, false, false]),
                (2, 320_000) => (false, 1, [9, 0, 0], [544, 0, 0], [true, false, false]),
                (2, 384_000) => (false, 1, [9, 0, 0], [672, 0, 0], [true, false, false]),
                (2, 480_000) => (false, 1, [9, 0, 0], [672, 0, 0], [true, false, false]),
                (2, 512_000) => (false, 1, [9, 0, 0], [768, 0, 0], [false, false, false]),
                (2, 640_000) => (false, 1, [9, 0, 0], [800, 0, 0], [false, false, false]),
                (3, 256_000) => (true, 2, [2, 6, 0], [732, 384, 0], [false, true, false]),
                (3, 320_000) => (true, 2, [2, 7, 0], [732, 384, 0], [false, true, false]),
                (3, 384_000) => (true, 2, [2, 9, 0], [768, 384, 0], [false, true, false]),
                (3, 512_000) => (true, 2, [2, 10, 0], [768, 544, 0], [false, true, false]),
                (3, 640_000) => (true, 2, [2, 12, 0], [768, 672, 0], [false, true, false]),
                (3, 896_000) => (false, 1, [16, 0, 0], [672, 0, 0], [true, false, false]),
                _ => {
                    return Err(HoaError::UnsupportedBitrate {
                        order,
                        bitrate: header.bitrate,
                    });
                }
            };

        let mut groups = [HoaGroupConfig::default(); MAX_HOA_GROUPS];
        let mut channel_offset = 0_usize;
        for group in 0..group_count {
            let channels = group_channels[group];
            groups[group] = HoaGroupConfig {
                channels,
                channel_offset,
                pair_index_bits: hoa_pair_index_bits(channels)?,
                core_lines: core_lines[group],
                bwe_enabled: group_bwe[group],
            };
            channel_offset = channel_offset
                .checked_add(channels)
                .ok_or(HoaError::AllocationOverflow)?;
        }
        if channel_offset > usize::from(MAX_CHANNELS) {
            return Err(HoaError::AllocationOverflow);
        }

        let foreground_channels = if order == 1 { 0 } else { group_channels[0] };
        let residual_channels = if order == 1 {
            0
        } else {
            channel_offset
                .checked_sub(foreground_channels)
                .ok_or(HoaError::AllocationOverflow)?
        };
        Ok(Self {
            order,
            output_channels,
            transport_channels: channel_offset,
            foreground_channels,
            residual_channels,
            default_spatial_analysis,
            groups,
            group_count,
        })
    }

    pub fn order(self) -> u8 {
        self.order
    }

    pub fn output_channels(self) -> usize {
        self.output_channels
    }

    pub fn transport_channels(self) -> usize {
        self.transport_channels
    }

    pub fn foreground_channels(self) -> usize {
        self.foreground_channels
    }

    pub fn residual_channels(self) -> usize {
        self.residual_channels
    }

    pub fn default_spatial_analysis(self) -> bool {
        self.default_spatial_analysis
    }

    pub fn groups(&self) -> &[HoaGroupConfig] {
        &self.groups[..self.group_count]
    }

    pub fn group_for_channel(self, channel: usize) -> Result<HoaGroupConfig, HoaError> {
        if channel >= self.transport_channels {
            return Err(HoaError::InvalidChannelIndex {
                index: channel,
                channels: self.transport_channels,
            });
        }
        self.groups()
            .iter()
            .copied()
            .find(|group| {
                channel >= group.channel_offset && channel < group.channel_offset + group.channels
            })
            .ok_or(HoaError::InvalidChannelIndex {
                index: channel,
                channels: self.transport_channels,
            })
    }

    pub fn core_for_channel(
        self,
        header: &FrameHeader,
        channel: usize,
    ) -> Result<CoreBitstreamConfig, HoaError> {
        let group = self.group_for_channel(channel)?;
        let threshold = LSF_LOW_BITRATE_THRESHOLD
            .checked_mul(
                u64::try_from(self.transport_channels).map_err(|_| HoaError::AllocationOverflow)?,
            )
            .ok_or(HoaError::AllocationOverflow)?;
        let lsf_mode = if u64::from(header.bitrate) > threshold {
            LsfCodebookMode::HighBitrate
        } else {
            LsfCodebookMode::LowBitrate
        };
        let bwe = group
            .bwe_enabled
            .then(|| BweConfig::for_hoa_bitrate(self.order, header.bitrate))
            .transpose()?;
        Ok(CoreBitstreamConfig::new(
            header.nn_type,
            header.payload_bits,
            lsf_mode,
            bwe,
        )?)
    }
}

pub fn hoa_pair_index_bits(channels: usize) -> Result<usize, HoaError> {
    if !(2..=usize::from(MAX_CHANNELS)).contains(&channels) {
        return Err(HoaError::InvalidGroupChannelCount { channels });
    }
    let combinations = channels
        .checked_mul(channels - 1)
        .and_then(|value| value.checked_div(2))
        .ok_or(HoaError::AllocationOverflow)?;
    Ok((usize::BITS as usize - (combinations - 1).leading_zeros() as usize).max(1))
}

pub fn hoa_pair_from_index(index: usize, channels: usize) -> Result<(usize, usize), HoaError> {
    if !(2..=usize::from(MAX_CHANNELS)).contains(&channels) {
        return Err(HoaError::InvalidGroupChannelCount { channels });
    }
    let combinations = channels
        .checked_mul(channels - 1)
        .and_then(|value| value.checked_div(2))
        .ok_or(HoaError::AllocationOverflow)?;
    if index >= combinations {
        return Err(HoaError::InvalidPairIndex {
            group: 0,
            index,
            combinations,
        });
    }

    let mut current = 0;
    for second in 1..channels {
        for first in 0..second {
            if current == index {
                return Ok((first, second));
            }
            current += 1;
        }
    }
    unreachable!("validated HOA pair index must map to one combination")
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HoaDmxMode {
    #[default]
    FullMs,
    SfbMs,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HoaPairSideInfo {
    index: u8,
    first: u8,
    second: u8,
    mode: HoaDmxMode,
    sfb_mask: [bool; HOA_SFB_COUNT],
}

impl HoaPairSideInfo {
    pub fn index(self) -> usize {
        usize::from(self.index)
    }

    pub fn first(self) -> usize {
        usize::from(self.first)
    }

    pub fn second(self) -> usize {
        usize::from(self.second)
    }

    pub fn mode(self) -> HoaDmxMode {
        self.mode
    }

    pub fn sfb_mask(&self) -> &[bool; HOA_SFB_COUNT] {
        &self.sfb_mask
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HoaGroupSideInfo {
    channels: usize,
    channel_offset: usize,
    pairs: [HoaPairSideInfo; MAX_HOA_GROUP_PAIRS],
    pair_count: usize,
    ild_indexes: [u8; MAX_CHANNELS as usize],
    group_bit_ratio: u8,
    channel_bit_ratios: [u8; MAX_CHANNELS as usize],
}

impl HoaGroupSideInfo {
    pub fn channels(self) -> usize {
        self.channels
    }

    pub fn channel_offset(self) -> usize {
        self.channel_offset
    }

    pub fn pairs(&self) -> &[HoaPairSideInfo] {
        &self.pairs[..self.pair_count]
    }

    pub fn ild_indexes(&self) -> &[u8] {
        &self.ild_indexes[..self.channels]
    }

    pub fn group_bit_ratio(self) -> u8 {
        self.group_bit_ratio
    }

    pub fn channel_bit_ratios(&self) -> &[u8] {
        &self.channel_bit_ratios[..self.channels]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoaSideInfo {
    transport_channels: usize,
    group_count: usize,
    scene_type: u8,
    spatial_analysis: bool,
    vector_channels: usize,
    basis_indices: [u16; MAX_HOA_BASIS],
    groups: [HoaGroupSideInfo; MAX_HOA_GROUPS],
}

impl HoaSideInfo {
    pub fn parse(reader: &mut BitReader<'_>, config: HoaBitstreamConfig) -> Result<Self, HoaError> {
        let scene_type = reader.read_u8(HOA_SCENE_TYPE_BITS)?;
        let spatial_analysis = reader.read_u8(HOA_SPATIAL_ANALYSIS_BITS)? != 0;
        let mut vector_channels = if config.default_spatial_analysis {
            config.foreground_channels
        } else {
            0
        };
        if spatial_analysis {
            vector_channels = usize::from(reader.read_u8(HOA_VECTOR_CHANNEL_BITS)?);
            let limit = MAX_HOA_BASIS.min(
                config
                    .transport_channels
                    .checked_sub(config.residual_channels)
                    .ok_or(HoaError::AllocationOverflow)?,
            );
            if vector_channels == 0 || vector_channels > limit {
                return Err(HoaError::InvalidVectorChannelCount {
                    channels: vector_channels,
                    limit,
                });
            }
        }
        if vector_channels > MAX_HOA_BASIS {
            return Err(HoaError::InvalidVectorChannelCount {
                channels: vector_channels,
                limit: MAX_HOA_BASIS,
            });
        }

        let mut basis_indices = [0_u16; MAX_HOA_BASIS];
        for basis_index in &mut basis_indices[..vector_channels] {
            *basis_index = reader.read_bits(HOA_BASIS_INDEX_BITS)? as u16;
            if usize::from(*basis_index) >= HOA_BASIS_TABLE_LEN {
                return Err(HoaError::InvalidBasisIndex {
                    index: usize::from(*basis_index),
                    limit: HOA_BASIS_TABLE_LEN,
                });
            }
        }

        let mut groups = [HoaGroupSideInfo::default(); MAX_HOA_GROUPS];
        for (group_index, (group_side, group_config)) in groups[..config.group_count]
            .iter_mut()
            .zip(config.groups())
            .enumerate()
        {
            let pair_count = usize::from(reader.read_u8(HOA_PAIR_COUNT_BITS)?);
            if pair_count > MAX_HOA_GROUP_PAIRS {
                return Err(HoaError::InvalidPairCount {
                    group: group_index,
                    pairs: pair_count,
                    limit: MAX_HOA_GROUP_PAIRS,
                });
            }
            let combinations = group_config
                .channels
                .checked_mul(group_config.channels - 1)
                .and_then(|value| value.checked_div(2))
                .ok_or(HoaError::AllocationOverflow)?;
            let mut pairs = [HoaPairSideInfo::default(); MAX_HOA_GROUP_PAIRS];
            for pair in &mut pairs[..pair_count] {
                let pair_index = reader.read_bits(group_config.pair_index_bits)? as usize;
                let (first, second) = hoa_pair_from_index(pair_index, group_config.channels)
                    .map_err(|error| match error {
                        HoaError::InvalidPairIndex { index, .. } => HoaError::InvalidPairIndex {
                            group: group_index,
                            index,
                            combinations,
                        },
                        other => other,
                    })?;
                let mode = if reader.read_u8(HOA_DMX_MODE_BITS)? == 0 {
                    HoaDmxMode::FullMs
                } else {
                    HoaDmxMode::SfbMs
                };
                let mut sfb_mask = [true; HOA_SFB_COUNT];
                if mode == HoaDmxMode::SfbMs {
                    for flag in &mut sfb_mask {
                        *flag = reader.read_u8(1)? != 0;
                    }
                }
                *pair = HoaPairSideInfo {
                    index: pair_index as u8,
                    first: first as u8,
                    second: second as u8,
                    mode,
                    sfb_mask,
                };
            }

            let mut ild_indexes = [MC_NO_ILD_INDEX; MAX_CHANNELS as usize];
            if pair_count != 0 {
                for ild in &mut ild_indexes[..group_config.channels] {
                    *ild = reader.read_u8(HOA_ILD_BITS)?;
                    if *ild > MC_NO_ILD_INDEX {
                        return Err(HoaError::InvalidIldIndex(*ild));
                    }
                }
            }
            let group_bit_ratio = reader.read_u8(HOA_RATIO_BITS)?;
            let mut channel_bit_ratios = [0_u8; MAX_CHANNELS as usize];
            for ratio in &mut channel_bit_ratios[..group_config.channels] {
                *ratio = reader.read_u8(HOA_RATIO_BITS)?;
            }
            *group_side = HoaGroupSideInfo {
                channels: group_config.channels,
                channel_offset: group_config.channel_offset,
                pairs,
                pair_count,
                ild_indexes,
                group_bit_ratio,
                channel_bit_ratios,
            };
        }

        Ok(Self {
            transport_channels: config.transport_channels,
            group_count: config.group_count,
            scene_type,
            spatial_analysis,
            vector_channels,
            basis_indices,
            groups,
        })
    }

    pub fn transport_channels(self) -> usize {
        self.transport_channels
    }

    pub fn scene_type(self) -> u8 {
        self.scene_type
    }

    pub fn spatial_analysis(self) -> bool {
        self.spatial_analysis
    }

    pub fn vector_channels(self) -> usize {
        self.vector_channels
    }

    pub fn basis_indices(&self) -> &[u16] {
        &self.basis_indices[..self.vector_channels]
    }

    pub fn groups(&self) -> &[HoaGroupSideInfo] {
        &self.groups[..self.group_count]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoaByteAllocation {
    channels: usize,
    bytes: [usize; MAX_CHANNELS as usize],
    trailing_bits: usize,
}

impl HoaByteAllocation {
    pub fn channel_bytes(&self) -> &[usize] {
        &self.bytes[..self.channels]
    }

    pub fn trailing_bits(self) -> usize {
        self.trailing_bits
    }
}

pub fn hoa_bytes_allocation(
    available_bits: usize,
    side_info: HoaSideInfo,
    config: HoaBitstreamConfig,
) -> Result<HoaByteAllocation, HoaError> {
    if side_info.transport_channels != config.transport_channels
        || side_info.group_count != config.group_count
    {
        return Err(HoaError::SideInfoConfigurationMismatch);
    }

    let total_bytes = available_bits / 8;
    let trailing_bits = available_bits % 8;
    let mut group_bytes = [0_usize; MAX_HOA_GROUPS];
    let mut remaining_bytes = total_bytes;
    for (group, destination) in group_bytes[..config.group_count.saturating_sub(1)]
        .iter_mut()
        .enumerate()
    {
        let requested_bytes = total_bytes
            .checked_mul(usize::from(side_info.groups[group].group_bit_ratio))
            .ok_or(HoaError::AllocationOverflow)?
            / HOA_RATIO_RANGE;
        if requested_bytes > remaining_bytes {
            return Err(HoaError::GroupRatioExceedsBudget {
                group,
                requested_bytes,
                remaining_bytes,
            });
        }
        *destination = requested_bytes;
        remaining_bytes -= requested_bytes;
    }
    group_bytes[config.group_count - 1] = remaining_bytes;

    let mut bytes = [0_usize; MAX_CHANNELS as usize];
    for (group_index, (group_side, group_config)) in
        side_info.groups().iter().zip(config.groups()).enumerate()
    {
        if group_side.channels != group_config.channels
            || group_side.channel_offset != group_config.channel_offset
        {
            return Err(HoaError::SideInfoConfigurationMismatch);
        }
        let sum_ratio = group_side.channel_bit_ratios()[..group_config.channels]
            .iter()
            .try_fold(0_usize, |sum, &ratio| {
                sum.checked_add(usize::from(ratio))
                    .ok_or(HoaError::AllocationOverflow)
            })?;
        if sum_ratio == 0 {
            return Err(HoaError::ZeroChannelRatio { group: group_index });
        }
        let bytes_per_range = group_bytes[group_index] / sum_ratio;
        let residual_bytes = group_bytes[group_index] % sum_ratio;
        let output = &mut bytes
            [group_config.channel_offset..group_config.channel_offset + group_config.channels];
        for (destination, &ratio) in output.iter_mut().zip(group_side.channel_bit_ratios()) {
            *destination = bytes_per_range
                .checked_mul(usize::from(ratio))
                .ok_or(HoaError::AllocationOverflow)?;
        }
        if residual_bytes >= group_config.channels {
            let increment = residual_bytes / group_config.channels;
            for destination in output.iter_mut() {
                *destination = destination
                    .checked_add(increment)
                    .ok_or(HoaError::AllocationOverflow)?;
            }
            output[0] = output[0]
                .checked_add(residual_bytes % group_config.channels)
                .ok_or(HoaError::AllocationOverflow)?;
        } else {
            output[0] = output[0]
                .checked_add(residual_bytes)
                .ok_or(HoaError::AllocationOverflow)?;
        }
    }

    debug_assert_eq!(
        bytes[..config.transport_channels].iter().sum::<usize>(),
        total_bytes
    );
    Ok(HoaByteAllocation {
        channels: config.transport_channels,
        bytes,
        trailing_bits,
    })
}

pub fn inverse_hoa_dmx(
    spectra: &mut [[f32; AVS3_FEATURE_DIMENSIONS]],
    side_info: HoaSideInfo,
    config: HoaBitstreamConfig,
) -> Result<(), HoaError> {
    if side_info.transport_channels != config.transport_channels
        || side_info.group_count != config.group_count
    {
        return Err(HoaError::SideInfoConfigurationMismatch);
    }
    if spectra.len() != config.transport_channels {
        return Err(HoaError::SpectrumChannelMismatch {
            expected: config.transport_channels,
            actual: spectra.len(),
        });
    }

    for (group_side, group_config) in side_info.groups().iter().zip(config.groups()) {
        if group_side.channels != group_config.channels
            || group_side.channel_offset != group_config.channel_offset
        {
            return Err(HoaError::SideInfoConfigurationMismatch);
        }
        for pair in group_side.pairs() {
            let first = group_config.channel_offset + pair.first();
            let second = group_config.channel_offset + pair.second();
            let (first_spectrum, second_spectrum) = two_spectra_mut(spectra, first, second);
            for (band, &enabled) in pair.sfb_mask().iter().enumerate() {
                if enabled {
                    inverse_hoa_subband(
                        first_spectrum,
                        second_spectrum,
                        HOA_SFB_BOUNDARIES[band],
                        HOA_SFB_BOUNDARIES[band + 1],
                    );
                }
            }
        }
    }

    for (group_side, group_config) in side_info.groups().iter().zip(config.groups()) {
        for (local_channel, &ild_index) in group_side.ild_indexes().iter().enumerate() {
            apply_mc_ild(
                &mut spectra[group_config.channel_offset + local_channel],
                ild_index,
            )?;
        }
    }
    Ok(())
}

fn inverse_hoa_subband(
    first: &mut [f32; AVS3_FEATURE_DIMENSIONS],
    second: &mut [f32; AVS3_FEATURE_DIMENSIONS],
    start: usize,
    stop: usize,
) {
    for line in start..stop {
        let original_first = first[line];
        first[line] = (original_first + second[line]) * core::f32::consts::FRAC_1_SQRT_2;
        second[line] = (original_first - second[line]) * core::f32::consts::FRAC_1_SQRT_2;
    }
}

fn two_spectra_mut<const N: usize>(
    spectra: &mut [[f32; N]],
    first: usize,
    second: usize,
) -> (&mut [f32; N], &mut [f32; N]) {
    debug_assert_ne!(first, second);
    if first < second {
        let (before_second, from_second) = spectra.split_at_mut(second);
        (&mut before_second[first], &mut from_second[0])
    } else {
        let (before_first, from_first) = spectra.split_at_mut(first);
        (&mut from_first[0], &mut before_first[second])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoaFrameSideInfo<'decoder> {
    channels: usize,
    cores: [Option<CoreSideInfo>; MAX_CHANNELS as usize],
    hoa: HoaSideInfo,
    neural_qc: [Option<ParsedNeuralQc<'decoder>>; MAX_CHANNELS as usize],
    allocation: HoaByteAllocation,
    consumed_bits: usize,
    padding_bits: usize,
}

impl<'decoder> HoaFrameSideInfo<'decoder> {
    pub fn channels(self) -> usize {
        self.channels
    }

    pub fn core(self, channel: usize) -> Option<CoreSideInfo> {
        self.cores.get(channel).copied().flatten()
    }

    pub fn hoa(self) -> HoaSideInfo {
        self.hoa
    }

    pub fn neural_qc(self, channel: usize) -> Option<ParsedNeuralQc<'decoder>> {
        self.neural_qc.get(channel).copied().flatten()
    }

    pub fn allocation(self) -> HoaByteAllocation {
        self.allocation
    }

    pub fn consumed_bits(self) -> usize {
        self.consumed_bits
    }

    pub fn padding_bits(self) -> usize {
        self.padding_bits
    }
}

/// Allocation-stable parser for the complete HOA audio payload syntax.
///
/// The wire order matches the reference decoder: all core prefixes, all
/// grouping records, HOA mode side information, then one neural QC record per
/// bitrate-derived transport channel.
#[derive(Debug, Clone)]
pub struct HoaSideInfoDecoder {
    context: [[u8; MAX_QC_BITSTREAM_BYTES]; MAX_CHANNELS as usize],
    base: [[u8; MAX_QC_BITSTREAM_BYTES]; MAX_CHANNELS as usize],
}

impl HoaSideInfoDecoder {
    pub fn new() -> Self {
        Self {
            context: [[0; MAX_QC_BITSTREAM_BYTES]; MAX_CHANNELS as usize],
            base: [[0; MAX_QC_BITSTREAM_BYTES]; MAX_CHANNELS as usize],
        }
    }

    pub fn parse<'decoder>(
        &'decoder mut self,
        payload: &[u8],
        header: &FrameHeader,
    ) -> Result<HoaFrameSideInfo<'decoder>, HoaError> {
        let config = HoaBitstreamConfig::for_header(header)?;
        let available_bits = payload.len().saturating_mul(8);
        if header.payload_bits > available_bits {
            return Err(CoreBitstreamError::PayloadTooShort {
                declared_bits: header.payload_bits,
                available_bits,
            }
            .into());
        }

        let channels = config.transport_channels;
        let mut reader = BitReader::with_bit_len(payload, header.payload_bits)?;
        let mut core_configs = [None; MAX_CHANNELS as usize];
        let mut prefixes: [Option<CoreSideInfoPrefix>; MAX_CHANNELS as usize] =
            [None; MAX_CHANNELS as usize];
        for channel in 0..channels {
            let core_config = config.core_for_channel(header, channel)?;
            core_configs[channel] = Some(core_config);
            prefixes[channel] = Some(parse_core_side_prefix(&mut reader, core_config)?);
        }

        let mut groupings = [WindowGrouping::single(); MAX_CHANNELS as usize];
        let mut cores = [None; MAX_CHANNELS as usize];
        for channel in 0..channels {
            let prefix = prefixes[channel].expect("all configured HOA prefixes parsed");
            let grouping = parse_grouping(&mut reader, prefix.transform_type())?;
            groupings[channel] = grouping;
            cores[channel] = Some(prefix.finish(grouping));
        }

        let hoa = HoaSideInfo::parse(&mut reader, config)?;
        let mut reserved_bits = 0_usize;
        for grouping in &groupings[..channels] {
            reserved_bits = reserved_bits
                .checked_add(qc_side_bits(header.nn_type, grouping.count())?)
                .ok_or(HoaError::AllocationOverflow)?;
        }
        let used_and_reserved = reader
            .position()
            .checked_add(reserved_bits)
            .ok_or(HoaError::AllocationOverflow)?;
        let available_qc_bits = header.payload_bits.checked_sub(used_and_reserved).ok_or(
            CoreBitstreamError::QcBudgetUnderflow {
                payload_bits: header.payload_bits,
                used_bits: reader.position(),
                reserved_bits,
            },
        )?;
        let allocation = hoa_bytes_allocation(available_qc_bits, hoa, config)?;

        let mut neural_qc = [None; MAX_CHANNELS as usize];
        for (((((context, base), slot), grouping), core_config), &entropy_bytes) in self.context
            [..channels]
            .iter_mut()
            .zip(&mut self.base[..channels])
            .zip(&mut neural_qc[..channels])
            .zip(&groupings[..channels])
            .zip(&core_configs[..channels])
            .zip(&allocation.bytes[..channels])
        {
            *slot = Some(parse_neural_qc(
                &mut reader,
                core_config.expect("all configured HOA cores initialized"),
                *grouping,
                entropy_bytes,
                context,
                base,
            )?);
        }

        let consumed_bits = reader.position();
        let padding_bits = reader.remaining();
        debug_assert_eq!(padding_bits, allocation.trailing_bits);
        debug_assert!(padding_bits < 8);
        Ok(HoaFrameSideInfo {
            channels,
            cores,
            hoa,
            neural_qc,
            allocation,
            consumed_bits,
            padding_bits,
        })
    }
}

impl Default for HoaSideInfoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioCodecId, BitDepth, BitWriter, NnType};

    fn header(order: u8, bitrate: u32, payload_bits: usize) -> FrameHeader {
        let channel_config = match order {
            1 => ChannelConfig::Hoa1,
            2 => ChannelConfig::Hoa2,
            3 => ChannelConfig::Hoa3,
            _ => panic!("unsupported test HOA order"),
        };
        let channels = (order + 1) * (order + 1);
        FrameHeader {
            codec_id: AudioCodecId::Avs3P3,
            nn_type: NnType::Main,
            profile: CodecProfile::Hoa,
            sample_rate: 48_000,
            bit_depth: BitDepth::Sixteen,
            channel_config: Some(channel_config),
            sound_bed_type: None,
            hoa_order: Some(order),
            objects: 0,
            bed_channels: channels,
            channels,
            has_lfe: false,
            bed_bitrate: None,
            object_bitrate: None,
            bitrate,
            crc: 0,
            header_len: 7,
            payload_bits,
            payload_len: payload_bits.div_ceil(8),
            frame_len: 7 + payload_bits.div_ceil(8),
            samples_per_channel: AVS3_FEATURE_DIMENSIONS as u32,
        }
    }

    fn write(writer: &mut BitWriter, value: u64, width: usize) {
        writer.write_bits(value, width).unwrap();
    }

    fn write_hoa3_side(writer: &mut BitWriter) {
        write(writer, 9, 4);
        write(writer, 0, 1);
        write(writer, 12, 12);
        write(writer, 1_342, 12);

        write(writer, 1, 4);
        write(writer, 0, 1);
        write(writer, 0, 1);
        for ild in [30, 0] {
            write(writer, ild, 5);
        }
        write(writer, 6, 4);
        for ratio in [8, 8] {
            write(writer, ratio, 4);
        }

        write(writer, 1, 4);
        write(writer, 2, 5);
        write(writer, 1, 1);
        for band in 0..HOA_SFB_COUNT {
            write(writer, u64::from(band % 3 == 0), 1);
        }
        for ild in [30, 1, 2, 3, 4, 5, 6] {
            write(writer, ild, 5);
        }
        write(writer, 10, 4);
        for ratio in 1..=7 {
            write(writer, ratio, 4);
        }
    }

    fn parsed_hoa3_side() -> (HoaBitstreamConfig, HoaSideInfo) {
        let config = HoaBitstreamConfig::for_header(&header(3, 320_000, 155)).unwrap();
        let mut writer = BitWriter::new();
        write_hoa3_side(&mut writer);
        assert_eq!(writer.bit_len(), 155);
        let payload = writer.into_bytes();
        let mut reader = BitReader::with_bit_len(&payload, 155).unwrap();
        let side = HoaSideInfo::parse(&mut reader, config).unwrap();
        assert_eq!(reader.remaining(), 0);
        (config, side)
    }

    #[test]
    fn bitrate_configuration_matches_reference_tables() {
        let foa = HoaBitstreamConfig::for_header(&header(1, 48_000, 0)).unwrap();
        assert_eq!(foa.output_channels(), 4);
        assert_eq!(foa.transport_channels(), 4);
        assert_eq!(foa.foreground_channels(), 0);
        assert_eq!(foa.residual_channels(), 0);
        assert!(!foa.default_spatial_analysis());
        assert_eq!(foa.groups().len(), 1);
        assert_eq!(foa.groups()[0].channels(), 4);
        assert_eq!(foa.groups()[0].pair_index_bits(), 3);
        assert_eq!(foa.groups()[0].core_lines(), 1_024);
        assert!(foa.groups()[0].bwe_enabled());

        let hoa2_rows = [
            (192_000, 352, true),
            (256_000, 384, true),
            (320_000, 544, true),
            (384_000, 672, true),
            (480_000, 672, true),
            (512_000, 768, false),
            (640_000, 800, false),
        ];
        for (bitrate, core_lines, bwe_enabled) in hoa2_rows {
            let config = HoaBitstreamConfig::for_header(&header(2, bitrate, 0)).unwrap();
            assert_eq!(config.output_channels(), 9);
            assert_eq!(config.transport_channels(), 9);
            assert_eq!(config.groups().len(), 1);
            assert_eq!(config.groups()[0].channels(), 9);
            assert_eq!(config.groups()[0].pair_index_bits(), 6);
            assert_eq!(config.groups()[0].core_lines(), core_lines);
            assert_eq!(config.groups()[0].bwe_enabled(), bwe_enabled);
        }

        let hoa3_rows = [
            (256_000, [2, 6], [732, 384], [false, true], true),
            (320_000, [2, 7], [732, 384], [false, true], true),
            (384_000, [2, 9], [768, 384], [false, true], true),
            (512_000, [2, 10], [768, 544], [false, true], true),
            (640_000, [2, 12], [768, 672], [false, true], true),
        ];
        for (bitrate, channels, core_lines, bwe, spatial) in hoa3_rows {
            let config = HoaBitstreamConfig::for_header(&header(3, bitrate, 0)).unwrap();
            assert_eq!(config.output_channels(), 16);
            assert_eq!(config.transport_channels(), channels.iter().sum());
            assert_eq!(config.groups().len(), 2);
            assert_eq!(
                config
                    .groups()
                    .iter()
                    .map(|group| group.channels())
                    .collect::<Vec<_>>(),
                channels
            );
            assert_eq!(
                config
                    .groups()
                    .iter()
                    .map(|group| group.core_lines())
                    .collect::<Vec<_>>(),
                core_lines
            );
            assert_eq!(
                config
                    .groups()
                    .iter()
                    .map(|group| group.bwe_enabled())
                    .collect::<Vec<_>>(),
                bwe
            );
            assert_eq!(config.default_spatial_analysis(), spatial);
        }

        let hoa3_full = HoaBitstreamConfig::for_header(&header(3, 896_000, 0)).unwrap();
        assert_eq!(hoa3_full.transport_channels(), 16);
        assert_eq!(hoa3_full.groups().len(), 1);
        assert_eq!(hoa3_full.groups()[0].channels(), 16);
        assert_eq!(hoa3_full.groups()[0].core_lines(), 672);
        assert!(hoa3_full.groups()[0].bwe_enabled());
        assert!(!hoa3_full.default_spatial_analysis());

        assert!(
            foa.core_for_channel(&header(1, 128_000, 0), 0)
                .unwrap()
                .bwe()
                .is_some()
        );
        let hoa3_low = HoaBitstreamConfig::for_header(&header(3, 256_000, 0)).unwrap();
        assert!(
            hoa3_low
                .core_for_channel(&header(3, 256_000, 0), 0)
                .unwrap()
                .bwe()
                .is_none()
        );
        assert!(
            hoa3_low
                .core_for_channel(&header(3, 256_000, 0), 2)
                .unwrap()
                .bwe()
                .is_some()
        );
    }

    #[test]
    fn hoa_bwe_configs_match_reference_tables() {
        let elow = BweConfig::for_hoa_bitrate(2, 192_000).unwrap();
        assert_eq!((elow.num_tiles(), elow.num_scale_factor_bands()), (2, 4));
        assert_eq!((elow.start_line(), elow.stop_line()), (352, 736));
        assert_eq!(elow.target_tiles(), &[352, 480, 736]);
        assert_eq!(elow.source_tiles(), &[64, 96]);
        assert_eq!(elow.scale_factor_bands(), &[352, 416, 480, 544, 736]);
        assert_eq!(elow.scale_factor_tile_wrap(), &[0, 2, 4]);

        let low = BweConfig::for_hoa_bitrate(1, 48_000).unwrap();
        assert_eq!((low.num_tiles(), low.num_scale_factor_bands()), (3, 6));
        assert_eq!((low.start_line(), low.stop_line()), (384, 832));
        assert_eq!(low.target_tiles(), &[384, 512, 672, 832]);
        assert_eq!(low.source_tiles(), &[96, 144, 192]);
        assert_eq!(
            low.scale_factor_bands(),
            &[384, 448, 512, 576, 672, 736, 832]
        );
        assert_eq!(low.scale_factor_tile_wrap(), &[0, 2, 4, 6]);

        let middle = BweConfig::for_hoa_bitrate(2, 320_000).unwrap();
        assert_eq!(
            (middle.num_tiles(), middle.num_scale_factor_bands()),
            (2, 4)
        );
        assert_eq!((middle.start_line(), middle.stop_line()), (544, 832));
        assert_eq!(middle.target_tiles(), &[544, 672, 832]);
        assert_eq!(middle.source_tiles(), &[144, 192]);
        assert_eq!(middle.scale_factor_bands(), &[544, 608, 672, 736, 832]);
        assert_eq!(middle.scale_factor_tile_wrap(), &[0, 2, 4]);

        let high = BweConfig::for_hoa_bitrate(3, 896_000).unwrap();
        assert_eq!((high.num_tiles(), high.num_scale_factor_bands()), (1, 2));
        assert_eq!((high.start_line(), high.stop_line()), (672, 832));
        assert_eq!(high.target_tiles(), &[672, 832]);
        assert_eq!(high.source_tiles(), &[192]);
        assert_eq!(high.scale_factor_bands(), &[672, 736, 832]);
        assert_eq!(high.scale_factor_tile_wrap(), &[0, 2]);
    }

    #[test]
    fn pair_indexes_use_hoa_second_channel_major_order() {
        assert_eq!(hoa_pair_index_bits(2).unwrap(), 1);
        assert_eq!(hoa_pair_index_bits(4).unwrap(), 3);
        assert_eq!(hoa_pair_index_bits(9).unwrap(), 6);
        assert_eq!(hoa_pair_index_bits(16).unwrap(), 7);
        let pairs = (0..6)
            .map(|index| hoa_pair_from_index(index, 4).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(pairs, [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]);
        assert!(matches!(
            hoa_pair_from_index(6, 4),
            Err(HoaError::InvalidPairIndex {
                index: 6,
                combinations: 6,
                ..
            })
        ));
    }

    #[test]
    fn parses_grouped_hoa3_side_information() {
        let (_, side) = parsed_hoa3_side();
        assert_eq!(side.scene_type(), 9);
        assert!(!side.spatial_analysis());
        assert_eq!(side.vector_channels(), 2);
        assert_eq!(side.basis_indices(), &[12, 1_342]);
        assert_eq!(side.groups().len(), 2);

        let first = side.groups()[0];
        assert_eq!(first.channels(), 2);
        assert_eq!(first.channel_offset(), 0);
        assert_eq!(first.group_bit_ratio(), 6);
        assert_eq!(first.channel_bit_ratios(), &[8, 8]);
        assert_eq!(first.ild_indexes(), &[30, 0]);
        assert_eq!(first.pairs().len(), 1);
        assert_eq!(
            (first.pairs()[0].first(), first.pairs()[0].second()),
            (0, 1)
        );
        assert_eq!(first.pairs()[0].mode(), HoaDmxMode::FullMs);
        assert!(first.pairs()[0].sfb_mask().iter().all(|&flag| flag));

        let second = side.groups()[1];
        assert_eq!(second.channels(), 7);
        assert_eq!(second.channel_offset(), 2);
        assert_eq!(second.group_bit_ratio(), 10);
        assert_eq!(second.channel_bit_ratios(), &[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(second.ild_indexes(), &[30, 1, 2, 3, 4, 5, 6]);
        assert_eq!(
            (second.pairs()[0].first(), second.pairs()[0].second()),
            (1, 2)
        );
        assert_eq!(second.pairs()[0].mode(), HoaDmxMode::SfbMs);
        for (band, &flag) in second.pairs()[0].sfb_mask().iter().enumerate() {
            assert_eq!(flag, band % 3 == 0);
        }
    }

    #[test]
    fn allocation_matches_reference_integer_order() {
        let (config, side) = parsed_hoa3_side();
        let allocation = hoa_bytes_allocation(1_003, side, config).unwrap();
        assert_eq!(
            allocation.channel_bytes(),
            &[23, 23, 7, 7, 9, 11, 13, 15, 17]
        );
        assert_eq!(allocation.trailing_bits(), 3);

        let mut zero_ratio = side;
        zero_ratio.groups[1].channel_bit_ratios.fill(0);
        assert!(matches!(
            hoa_bytes_allocation(1_003, zero_ratio, config),
            Err(HoaError::ZeroChannelRatio { group: 1 })
        ));
    }

    #[test]
    fn rejects_unsafe_side_information_values() {
        let config = HoaBitstreamConfig::for_header(&header(3, 320_000, 32)).unwrap();
        let mut too_many_vectors = BitWriter::new();
        write(&mut too_many_vectors, 0, 4);
        write(&mut too_many_vectors, 1, 1);
        write(&mut too_many_vectors, 3, 4);
        let bits = too_many_vectors.bit_len();
        let payload = too_many_vectors.into_bytes();
        let mut reader = BitReader::with_bit_len(&payload, bits).unwrap();
        assert!(matches!(
            HoaSideInfo::parse(&mut reader, config),
            Err(HoaError::InvalidVectorChannelCount {
                channels: 3,
                limit: 2
            })
        ));

        let mut invalid_basis = BitWriter::new();
        write(&mut invalid_basis, 0, 4);
        write(&mut invalid_basis, 0, 1);
        write(&mut invalid_basis, HOA_BASIS_TABLE_LEN as u64, 12);
        let bits = invalid_basis.bit_len();
        let payload = invalid_basis.into_bytes();
        let mut reader = BitReader::with_bit_len(&payload, bits).unwrap();
        assert!(matches!(
            HoaSideInfo::parse(&mut reader, config),
            Err(HoaError::InvalidBasisIndex {
                index: HOA_BASIS_TABLE_LEN,
                limit: HOA_BASIS_TABLE_LEN
            })
        ));

        let foa = HoaBitstreamConfig::for_header(&header(1, 192_000, 32)).unwrap();
        let mut invalid_ild = BitWriter::new();
        write(&mut invalid_ild, 0, 4);
        write(&mut invalid_ild, 0, 1);
        write(&mut invalid_ild, 1, 4);
        write(&mut invalid_ild, 0, 3);
        write(&mut invalid_ild, 0, 1);
        write(&mut invalid_ild, 31, 5);
        let bits = invalid_ild.bit_len();
        let payload = invalid_ild.into_bytes();
        let mut reader = BitReader::with_bit_len(&payload, bits).unwrap();
        assert!(matches!(
            HoaSideInfo::parse(&mut reader, foa),
            Err(HoaError::InvalidIldIndex(31))
        ));
    }

    #[test]
    fn inverse_dmx_applies_masks_then_group_ild() {
        let (config, side) = parsed_hoa3_side();
        let mut spectra = vec![[0.0_f32; AVS3_FEATURE_DIMENSIONS]; 9];
        spectra[0].fill(2.0);
        spectra[1].fill(1.0);
        spectra[3].fill(4.0);
        spectra[4].fill(2.0);
        inverse_hoa_dmx(&mut spectra, side, config).unwrap();

        let c = core::f32::consts::FRAC_1_SQRT_2;
        assert_eq!(spectra[0][0], 3.0 * c);
        assert_eq!(spectra[1][0], c * 1.777_777_8);
        assert_eq!(spectra[0][800], 2.0);
        assert_eq!(spectra[1][800], 1.777_777_8);
        assert_eq!(spectra[3][0], 6.0 * c * 0.75);
        assert_eq!(spectra[4][0], 2.0 * c * 0.5625);
        assert_eq!(spectra[3][8], 4.0 * 0.75);
        assert_eq!(spectra[4][8], 2.0 * 0.5625);
    }

    #[test]
    fn parses_complete_foa_wire_order() {
        let payload_bits = 460;
        let frame_header = header(1, 192_000, payload_bits);
        let mut writer = BitWriter::new();

        for _ in 0..4 {
            write(&mut writer, 0, 2);
            for width in [8, 8, 7, 7, 6, 5, 5] {
                write(&mut writer, 0, width);
            }
            write(&mut writer, 0, 1);
            write(&mut writer, 0, 1);
            for _ in 0..4 {
                write(&mut writer, 0, 7);
            }
            write(&mut writer, 0, 1);
            write(&mut writer, 0, 1);
        }

        write(&mut writer, 3, 4);
        write(&mut writer, 0, 1);
        write(&mut writer, 0, 4);
        write(&mut writer, 15, 4);
        for _ in 0..4 {
            write(&mut writer, 4, 4);
        }
        assert_eq!(writer.bit_len(), 349);

        for channel in 0..4 {
            write(&mut writer, 0, 1);
            write(&mut writer, 37, 7);
            write(&mut writer, 3, 3);
            write(&mut writer, 0, 8);
            write(&mut writer, channel, 8);
        }
        assert_eq!(writer.bit_len(), 457);
        write(&mut writer, 0, 3);
        let payload = writer.into_bytes();

        let mut decoder = HoaSideInfoDecoder::new();
        let parsed = decoder.parse(&payload, &frame_header).unwrap();
        assert_eq!(parsed.channels(), 4);
        assert_eq!(parsed.hoa().scene_type(), 3);
        assert_eq!(parsed.hoa().vector_channels(), 0);
        assert_eq!(parsed.allocation().channel_bytes(), &[1, 1, 1, 1]);
        assert_eq!(parsed.consumed_bits(), 457);
        assert_eq!(parsed.padding_bits(), 3);
        for channel in 0..4 {
            assert!(parsed.core(channel).unwrap().bwe().is_some());
            assert!(parsed.neural_qc(channel).is_some());
        }
    }
}
