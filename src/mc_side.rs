use core::fmt;

use crate::bitstream::BitReader;
use crate::core_side::{
    BweConfig, CoreBitstreamConfig, CoreBitstreamError, CoreSideInfo, CoreSideInfoPrefix,
    LsfCodebookMode, ParsedNeuralQc, WindowGrouping, parse_core_side_prefix, parse_grouping,
    parse_neural_qc, qc_side_bits,
};
use crate::error::BitstreamError;
use crate::header::{ChannelConfig, CodecProfile, FrameHeader, MAX_CHANNELS, SoundBedType};
use crate::model::AVS3_FEATURE_DIMENSIONS;
use crate::neural_qc::MAX_QC_BITSTREAM_BYTES;

pub const MC_LFE_CHANNEL_INDEX: usize = 3;
pub const MC_LFE_RESERVED_LINES: usize = 32;
pub const MC_SILENCE_BYTES: usize = 8;
pub const MC_ILD_CODEBOOK_LEN: usize = 30;
pub const MC_NO_ILD_INDEX: u8 = MC_ILD_CODEBOOK_LEN as u8;
pub const MAX_MC_PAIRS: usize = MAX_CHANNELS as usize / 2;

/// Normative multichannel inverse-ILD scalar codebook.
///
/// Index [`MC_NO_ILD_INDEX`] is the bitstream sentinel for unity gain and is
/// intentionally not stored in this table.
pub const MC_ILD_CODEBOOK: [f32; MC_ILD_CODEBOOK_LEN] = [
    1.777_777_8,
    0.75,
    0.5625,
    3.2,
    5.333_333_5,
    0.8125,
    1.066_666_7,
    4.0,
    0.1875,
    1.142_857_2,
    0.4375,
    1.454_545_5,
    0.125,
    0.625,
    2.285_714_4,
    0.5,
    16.0,
    2.0,
    0.875,
    0.25,
    1.333_333_4,
    0.375,
    1.6,
    8.0,
    0.6875,
    0.0625,
    1.230_769_3,
    0.3125,
    0.9375,
    2.666_666_7,
];

const MC_PAIR_COUNT_BITS: usize = 4;
const MC_ILD_BITS: usize = 5;
const MC_RATIO_BITS: usize = 6;
const MC_RATIO_SCOPE: i64 = 1 << MC_RATIO_BITS;
const MC_MAX_CHANNEL_BYTES: i64 = (256_000_i64 * 1_024 / 48_000) / 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McError {
    NotChannelBasedMultichannel {
        profile: CodecProfile,
        channel_config: Option<ChannelConfig>,
        channels: u8,
    },
    InvalidChannelCount {
        channels: usize,
        limit: usize,
    },
    InvalidLfeConfiguration {
        channel_config: ChannelConfig,
        header_has_lfe: bool,
    },
    InvalidBitrateConfiguration {
        total_bitrate: u32,
        bed_bitrate: Option<u32>,
        object_bitrate: Option<u32>,
        objects: u8,
    },
    InvalidPairCount {
        pairs: usize,
        limit: usize,
    },
    InvalidPairIndex {
        index: usize,
        combinations: usize,
    },
    InvalidIldIndex(u8),
    InvalidChannelIndex {
        index: usize,
        channels: usize,
    },
    SpectrumChannelMismatch {
        expected: usize,
        actual: usize,
    },
    ChannelLengthMismatch {
        first: usize,
        second: usize,
    },
    InvalidSpectrumLength {
        expected: usize,
        actual: usize,
    },
    SideInfoChannelMismatch {
        expected: usize,
        actual: usize,
    },
    InsufficientAllocation {
        available_bytes: usize,
        minimum_bytes: usize,
    },
    InvalidRatioSum,
    AllocationOverflow,
    Bitstream(BitstreamError),
    CoreBitstream(CoreBitstreamError),
}

impl fmt::Display for McError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotChannelBasedMultichannel {
                profile,
                channel_config,
                channels,
            } => write!(
                f,
                "MC parser requires channel-based multichannel audio or a Mix layout with at least three total channels; got profile {profile:?}, configuration {channel_config:?}, {channels} channels"
            ),
            Self::InvalidChannelCount { channels, limit } => {
                write!(f, "MC channel count {channels} is outside 2..={limit}")
            }
            Self::InvalidLfeConfiguration {
                channel_config,
                header_has_lfe,
            } => write!(
                f,
                "MC configuration {channel_config} and header LFE flag {header_has_lfe} disagree"
            ),
            Self::InvalidBitrateConfiguration {
                total_bitrate,
                bed_bitrate,
                object_bitrate,
                objects,
            } => write!(
                f,
                "MC Mix bitrate {total_bitrate} does not match bed {bed_bitrate:?} plus {objects} objects at {object_bitrate:?} each"
            ),
            Self::InvalidPairCount { pairs, limit } => {
                write!(
                    f,
                    "MC frame declares {pairs} pairs; decoder limit is {limit}"
                )
            }
            Self::InvalidPairIndex {
                index,
                combinations,
            } => write!(
                f,
                "MC pair index {index} is outside {combinations} channel combinations"
            ),
            Self::InvalidIldIndex(index) => {
                write!(f, "MC ILD index {index} is outside 0..={MC_NO_ILD_INDEX}")
            }
            Self::InvalidChannelIndex { index, channels } => write!(
                f,
                "MC channel index {index} is outside a {channels}-channel configuration"
            ),
            Self::SpectrumChannelMismatch { expected, actual } => write!(
                f,
                "MC coupling received {actual} spectra; expected {expected}"
            ),
            Self::ChannelLengthMismatch { first, second } => write!(
                f,
                "MC pair spectra have different lengths: {first} and {second}"
            ),
            Self::InvalidSpectrumLength { expected, actual } => {
                write!(f, "MC spectrum has {actual} lines; expected {expected}")
            }
            Self::SideInfoChannelMismatch { expected, actual } => write!(
                f,
                "MC side information has {actual} channels; configuration expects {expected}"
            ),
            Self::InsufficientAllocation {
                available_bytes,
                minimum_bytes,
            } => write!(
                f,
                "MC entropy budget has {available_bytes} bytes; at least {minimum_bytes} are required"
            ),
            Self::InvalidRatioSum => {
                f.write_str("MC bit allocation cannot redistribute with a zero ratio sum")
            }
            Self::AllocationOverflow => f.write_str("MC byte allocation arithmetic overflow"),
            Self::Bitstream(error) => error.fmt(f),
            Self::CoreBitstream(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for McError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(error) => Some(error),
            Self::CoreBitstream(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BitstreamError> for McError {
    fn from(value: BitstreamError) -> Self {
        Self::Bitstream(value)
    }
}

impl From<CoreBitstreamError> for McError {
    fn from(value: CoreBitstreamError) -> Self {
        Self::CoreBitstream(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McBitstreamConfig {
    core: CoreBitstreamConfig,
    channels: usize,
    bed_channels: usize,
    ild_channels: usize,
    has_lfe: bool,
    lfe_bytes: usize,
    pair_index_bits: usize,
}

impl McBitstreamConfig {
    pub fn for_header(header: &FrameHeader) -> Result<Self, McError> {
        let channels = usize::from(header.channels);
        if !(2..=usize::from(MAX_CHANNELS)).contains(&channels) {
            return Err(McError::InvalidChannelCount {
                channels,
                limit: usize::from(MAX_CHANNELS),
            });
        }

        let (bed_channels, ild_channels, lfe_bitrate) = match header.profile {
            CodecProfile::ChannelBased => {
                let Some(channel_config) = header.channel_config else {
                    return Err(not_mc_coded(header));
                };
                if !is_multichannel_config(channel_config)
                    || channels != usize::from(channel_config.channels())
                    || usize::from(header.bed_channels) != channels
                    || header.objects != 0
                    || header.sound_bed_type.is_some()
                    || header.bed_bitrate != Some(header.bitrate)
                    || header.object_bitrate.is_some()
                {
                    return Err(not_mc_coded(header));
                }
                validate_lfe(header, channel_config)?;
                (
                    channels,
                    channels - usize::from(header.has_lfe),
                    header.bitrate,
                )
            }
            CodecProfile::Mixed => match header.sound_bed_type {
                Some(SoundBedType::ObjectsOnly)
                    if header.channel_config.is_none()
                        && header.bed_channels == 0
                        && header.objects == header.channels
                        && channels >= 3
                        && !header.has_lfe
                        && header.bed_bitrate.is_none() =>
                {
                    validate_mix_bitrate(header)?;
                    (0, 0, header.bitrate)
                }
                Some(SoundBedType::ChannelBed) => {
                    let Some(channel_config) = header.channel_config else {
                        return Err(not_mc_coded(header));
                    };
                    let bed_channels = usize::from(header.bed_channels);
                    if !(channel_config == ChannelConfig::Stereo
                        || is_multichannel_config(channel_config))
                        || bed_channels != usize::from(channel_config.channels())
                        || header.objects == 0
                        || bed_channels.checked_add(usize::from(header.objects)) != Some(channels)
                    {
                        return Err(not_mc_coded(header));
                    }
                    validate_lfe(header, channel_config)?;
                    validate_mix_bitrate(header)?;
                    let ild_channels = bed_channels - usize::from(header.has_lfe);
                    let lfe_bitrate = header
                        .bed_bitrate
                        .ok_or_else(|| invalid_mix_bitrate(header))?;
                    (bed_channels, ild_channels, lfe_bitrate)
                }
                _ => return Err(not_mc_coded(header)),
            },
            CodecProfile::Hoa => return Err(not_mc_coded(header)),
        };

        let core_channels = channels - usize::from(header.has_lfe);
        let high_bitrate_threshold = 32_000_u64
            .checked_mul(u64::try_from(core_channels).map_err(|_| McError::AllocationOverflow)?)
            .ok_or(McError::AllocationOverflow)?;
        let lsf_mode = if u64::from(header.bitrate) > high_bitrate_threshold {
            LsfCodebookMode::HighBitrate
        } else {
            LsfCodebookMode::LowBitrate
        };
        let core = CoreBitstreamConfig::new(
            header.nn_type,
            header.payload_bits,
            lsf_mode,
            BweConfig::for_multichannel_bitrate(header.bitrate, core_channels)?,
        )?;
        let lfe_bytes = if header.has_lfe {
            mc_lfe_bytes(lfe_bitrate, ild_channels)?
        } else {
            0
        };

        Ok(Self {
            core,
            channels,
            bed_channels,
            ild_channels,
            has_lfe: header.has_lfe,
            lfe_bytes,
            pair_index_bits: mc_pair_index_bits(channels)?,
        })
    }

    pub fn core(self) -> CoreBitstreamConfig {
        self.core
    }

    pub fn channels(self) -> usize {
        self.channels
    }

    pub fn bed_channels(self) -> usize {
        self.bed_channels
    }

    pub fn has_lfe(self) -> bool {
        self.has_lfe
    }

    pub fn coupled_channels(self) -> usize {
        self.ild_channels
    }

    pub fn ild_channels(self) -> usize {
        self.ild_channels
    }

    pub fn lfe_channel(self) -> Option<usize> {
        self.has_lfe.then_some(MC_LFE_CHANNEL_INDEX)
    }

    pub fn lfe_bytes(self) -> usize {
        self.lfe_bytes
    }

    pub fn pair_index_bits(self) -> usize {
        self.pair_index_bits
    }
}

fn not_mc_coded(header: &FrameHeader) -> McError {
    McError::NotChannelBasedMultichannel {
        profile: header.profile,
        channel_config: header.channel_config,
        channels: header.channels,
    }
}

fn validate_lfe(header: &FrameHeader, channel_config: ChannelConfig) -> Result<(), McError> {
    if header.has_lfe != channel_config.has_lfe() {
        return Err(McError::InvalidLfeConfiguration {
            channel_config,
            header_has_lfe: header.has_lfe,
        });
    }
    Ok(())
}

fn invalid_mix_bitrate(header: &FrameHeader) -> McError {
    McError::InvalidBitrateConfiguration {
        total_bitrate: header.bitrate,
        bed_bitrate: header.bed_bitrate,
        object_bitrate: header.object_bitrate,
        objects: header.objects,
    }
}

fn validate_mix_bitrate(header: &FrameHeader) -> Result<(), McError> {
    let object_total = header
        .object_bitrate
        .and_then(|bitrate| bitrate.checked_mul(u32::from(header.objects)));
    let expected = match header.sound_bed_type {
        Some(SoundBedType::ObjectsOnly) => object_total,
        Some(SoundBedType::ChannelBed) => object_total
            .zip(header.bed_bitrate)
            .and_then(|(objects, bed)| objects.checked_add(bed)),
        None => None,
    };
    if expected != Some(header.bitrate) {
        return Err(invalid_mix_bitrate(header));
    }
    Ok(())
}

pub const fn is_multichannel_config(config: ChannelConfig) -> bool {
    matches!(
        config,
        ChannelConfig::Mc5_1
            | ChannelConfig::Mc7_1
            | ChannelConfig::Mc10_2
            | ChannelConfig::Mc22_2
            | ChannelConfig::Mc4_0
            | ChannelConfig::Mc5_1_2
            | ChannelConfig::Mc5_1_4
            | ChannelConfig::Mc7_1_2
            | ChannelConfig::Mc7_1_4
    )
}

fn mc_lfe_bytes(total_bitrate: u32, coupled_channels: usize) -> Result<usize, McError> {
    if coupled_channels == 0 {
        return Err(McError::InvalidChannelCount {
            channels: coupled_channels,
            limit: usize::from(MAX_CHANNELS),
        });
    }
    let cpe_rate = u64::from(total_bitrate)
        .checked_mul(2)
        .ok_or(McError::AllocationOverflow)?
        / u64::try_from(coupled_channels).map_err(|_| McError::AllocationOverflow)?;
    Ok(if cpe_rate < 64_000 {
        10
    } else if cpe_rate < 96_000 {
        15
    } else {
        20
    })
}

pub fn mc_pair_index_bits(channels: usize) -> Result<usize, McError> {
    if !(2..=usize::from(MAX_CHANNELS)).contains(&channels) {
        return Err(McError::InvalidChannelCount {
            channels,
            limit: usize::from(MAX_CHANNELS),
        });
    }
    let combinations = channels
        .checked_mul(channels - 1)
        .and_then(|value| value.checked_div(2))
        .ok_or(McError::AllocationOverflow)?;
    let bits = usize::BITS as usize - (combinations - 1).leading_zeros() as usize;
    Ok(bits.max(1))
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct McPair {
    first: u8,
    second: u8,
}

impl McPair {
    pub fn first(self) -> usize {
        usize::from(self.first)
    }

    pub fn second(self) -> usize {
        usize::from(self.second)
    }
}

pub fn mc_pair_from_index(index: usize, channels: usize) -> Result<McPair, McError> {
    let combinations = channels
        .checked_mul(channels.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or(McError::AllocationOverflow)?;
    if !(2..=usize::from(MAX_CHANNELS)).contains(&channels) {
        return Err(McError::InvalidChannelCount {
            channels,
            limit: usize::from(MAX_CHANNELS),
        });
    }
    if index >= combinations {
        return Err(McError::InvalidPairIndex {
            index,
            combinations,
        });
    }

    let mut current = 0;
    for first in 0..channels - 1 {
        for second in first + 1..channels {
            if current == index {
                return Ok(McPair {
                    first: first as u8,
                    second: second as u8,
                });
            }
            current += 1;
        }
    }
    unreachable!("validated pair index must map to one combination")
}

/// Map the MC coupling order used by pair indexes and ILD values to the
/// decoder's output/core channel order.
///
/// The reference decoder removes output LFE channel 3 from the coupled bed,
/// appends it after all other bed channels, and leaves layouts without LFE in
/// their original order.
pub fn mc_coupling_channel_to_output(
    coupling_channel: usize,
    config: McBitstreamConfig,
) -> Result<usize, McError> {
    if coupling_channel >= config.channels {
        return Err(McError::InvalidChannelIndex {
            index: coupling_channel,
            channels: config.channels,
        });
    }
    if !config.has_lfe || coupling_channel < MC_LFE_CHANNEL_INDEX {
        return Ok(coupling_channel);
    }
    let lfe_coupling_channel = config.bed_channels - 1;
    if coupling_channel == lfe_coupling_channel {
        return Ok(MC_LFE_CHANNEL_INDEX);
    }
    if coupling_channel < lfe_coupling_channel {
        return Ok(coupling_channel + 1);
    }
    Ok(coupling_channel)
}

/// Map an output/core channel index to the MC coupling order.
pub fn mc_output_channel_to_coupling(
    output_channel: usize,
    config: McBitstreamConfig,
) -> Result<usize, McError> {
    if output_channel >= config.channels {
        return Err(McError::InvalidChannelIndex {
            index: output_channel,
            channels: config.channels,
        });
    }
    if !config.has_lfe || output_channel < MC_LFE_CHANNEL_INDEX {
        return Ok(output_channel);
    }
    if output_channel == MC_LFE_CHANNEL_INDEX {
        return Ok(config.bed_channels - 1);
    }
    if output_channel < config.bed_channels {
        return Ok(output_channel - 1);
    }
    Ok(output_channel)
}

/// Apply one inverse MC mid/side pair in the C decoder's arithmetic order.
pub fn inverse_mc_pair(first: &mut [f32], second: &mut [f32]) -> Result<(), McError> {
    if first.len() != second.len() {
        return Err(McError::ChannelLengthMismatch {
            first: first.len(),
            second: second.len(),
        });
    }
    for (first_value, second_value) in first.iter_mut().zip(second) {
        let original_first = *first_value;
        let original_second = *second_value;
        *first_value = (original_first + original_second) * core::f32::consts::FRAC_1_SQRT_2;
        *second_value = (original_first - original_second) * core::f32::consts::FRAC_1_SQRT_2;
    }
    Ok(())
}

/// Apply one inverse MC ILD codebook scalar. The value 30 is unity gain.
pub fn apply_mc_ild(spectrum: &mut [f32], index: u8) -> Result<(), McError> {
    if index == MC_NO_ILD_INDEX {
        return Ok(());
    }
    let factor = MC_ILD_CODEBOOK
        .get(usize::from(index))
        .copied()
        .ok_or(McError::InvalidIldIndex(index))?;
    for value in spectrum {
        *value *= factor;
    }
    Ok(())
}

/// Apply reverse pair upmix and inverse ILD to output-ordered spectra.
///
/// Pair indexes and ILD entries are coupling-ordered. Passing output-ordered
/// storage here keeps the rest of the decoder channel-local while preserving
/// the C decoder's LFE relocation semantics.
pub fn inverse_mc_coupling(
    spectra: &mut [[f32; AVS3_FEATURE_DIMENSIONS]],
    side_info: McSideInfo,
    config: McBitstreamConfig,
) -> Result<(), McError> {
    if side_info.channels != config.channels {
        return Err(McError::SideInfoChannelMismatch {
            expected: config.channels,
            actual: side_info.channels,
        });
    }
    if spectra.len() != config.channels {
        return Err(McError::SpectrumChannelMismatch {
            expected: config.channels,
            actual: spectra.len(),
        });
    }

    for pair in side_info.pairs().iter().rev() {
        let first = mc_coupling_channel_to_output(pair.first(), config)?;
        let second = mc_coupling_channel_to_output(pair.second(), config)?;
        let (first_spectrum, second_spectrum) = two_spectra_mut(spectra, first, second);
        inverse_mc_pair(first_spectrum, second_spectrum)?;
    }

    for coupling_channel in 0..config.ild_channels() {
        let output_channel = mc_coupling_channel_to_output(coupling_channel, config)?;
        apply_mc_ild(
            &mut spectra[output_channel],
            side_info.ild_indexes[coupling_channel],
        )?;
    }
    Ok(())
}

/// Clear all LFE coefficients above the 32 normative reserved lines.
pub fn clear_mc_lfe_spectrum(spectrum: &mut [f32]) -> Result<(), McError> {
    if spectrum.len() != AVS3_FEATURE_DIMENSIONS {
        return Err(McError::InvalidSpectrumLength {
            expected: AVS3_FEATURE_DIMENSIONS,
            actual: spectrum.len(),
        });
    }
    spectrum[MC_LFE_RESERVED_LINES..].fill(0.0);
    Ok(())
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
pub struct McSideInfo {
    channels: usize,
    has_silence_flags: bool,
    silence: [bool; MAX_CHANNELS as usize],
    pairs: [McPair; MAX_MC_PAIRS],
    pair_count: usize,
    ild_indexes: [u8; MAX_CHANNELS as usize],
    bit_ratios: [Option<u8>; MAX_CHANNELS as usize],
}

impl McSideInfo {
    pub fn parse(reader: &mut BitReader<'_>, config: McBitstreamConfig) -> Result<Self, McError> {
        let channels = config.channels;
        let has_silence_flags = reader.read_u8(1)? != 0;
        let mut silence = [false; MAX_CHANNELS as usize];
        if has_silence_flags {
            for (channel, flag) in silence.iter_mut().enumerate().take(channels) {
                if config.lfe_channel() == Some(channel) {
                    continue;
                }
                *flag = reader.read_u8(1)? != 0;
            }
        }

        let pair_count = usize::from(reader.read_u8(MC_PAIR_COUNT_BITS)?);
        if pair_count > MAX_MC_PAIRS {
            return Err(McError::InvalidPairCount {
                pairs: pair_count,
                limit: MAX_MC_PAIRS,
            });
        }
        let mut pairs = [McPair::default(); MAX_MC_PAIRS];
        let mut ild_indexes = [MC_NO_ILD_INDEX; MAX_CHANNELS as usize];
        for pair in &mut pairs[..pair_count] {
            let pair_index = reader.read_bits(config.pair_index_bits)? as usize;
            *pair = mc_pair_from_index(pair_index, channels)?;
            for channel in [pair.first(), pair.second()] {
                let ild = reader.read_u8(MC_ILD_BITS)?;
                if ild > MC_NO_ILD_INDEX {
                    return Err(McError::InvalidIldIndex(ild));
                }
                ild_indexes[channel] = ild;
            }
        }

        let mut bit_ratios = [None; MAX_CHANNELS as usize];
        for channel in 0..channels {
            if config.lfe_channel() == Some(channel) || silence[channel] {
                continue;
            }
            bit_ratios[channel] = Some(reader.read_u8(MC_RATIO_BITS)?);
        }

        Ok(Self {
            channels,
            has_silence_flags,
            silence,
            pairs,
            pair_count,
            ild_indexes,
            bit_ratios,
        })
    }

    pub fn channels(self) -> usize {
        self.channels
    }

    pub fn has_silence_flags(self) -> bool {
        self.has_silence_flags
    }

    pub fn silence_flags(&self) -> &[bool] {
        &self.silence[..self.channels]
    }

    pub fn pairs(&self) -> &[McPair] {
        &self.pairs[..self.pair_count]
    }

    pub fn ild_indexes(&self) -> &[u8] {
        &self.ild_indexes[..self.channels]
    }

    pub fn bit_ratios(&self) -> &[Option<u8>] {
        &self.bit_ratios[..self.channels]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McByteAllocation {
    channels: usize,
    bytes: [usize; MAX_CHANNELS as usize],
}

impl McByteAllocation {
    pub fn channel_bytes(&self) -> &[usize] {
        &self.bytes[..self.channels]
    }
}

pub fn mc_bytes_allocation(
    available_bits: usize,
    side_info: McSideInfo,
    config: McBitstreamConfig,
) -> Result<McByteAllocation, McError> {
    if side_info.channels != config.channels {
        return Err(McError::SideInfoChannelMismatch {
            expected: config.channels,
            actual: side_info.channels,
        });
    }

    let channels = config.channels;
    let silence_count = side_info.silence[..channels]
        .iter()
        .filter(|&&flag| flag)
        .count();
    let silence_bits = silence_count
        .checked_mul(MC_SILENCE_BYTES * 8)
        .ok_or(McError::AllocationOverflow)?;
    let adjusted_bits =
        available_bits
            .checked_sub(silence_bits)
            .ok_or(McError::InsufficientAllocation {
                available_bytes: available_bits / 8,
                minimum_bytes: silence_count * MC_SILENCE_BYTES,
            })?;
    let active_channels = channels - silence_count;

    let mut compact_ratios = [0_i64; MAX_CHANNELS as usize];
    let mut ratio_count = 0;
    for channel in 0..channels {
        if config.lfe_channel() == Some(channel) || side_info.silence[channel] {
            continue;
        }
        compact_ratios[ratio_count] = i64::from(side_info.bit_ratios[channel].ok_or(
            McError::SideInfoChannelMismatch {
                expected: channels,
                actual: channel,
            },
        )?);
        ratio_count += 1;
    }

    let mut active_bytes = allocate_active_bytes(
        adjusted_bits,
        &compact_ratios[..ratio_count],
        active_channels,
        config.has_lfe,
        config.lfe_bytes,
    )?;
    let mut bytes = [0_usize; MAX_CHANNELS as usize];
    let mut active_index = 0;
    for (channel, destination) in bytes[..channels].iter_mut().enumerate() {
        if config.lfe_channel() == Some(channel) {
            *destination = config.lfe_bytes;
            continue;
        }
        if side_info.silence[channel] {
            *destination = MC_SILENCE_BYTES;
            continue;
        }
        if config.has_lfe && active_index == MC_LFE_CHANNEL_INDEX {
            active_index += 1;
        }
        let value = active_bytes
            .get_mut(active_index)
            .ok_or(McError::AllocationOverflow)?;
        *destination = usize::try_from(*value).map_err(|_| McError::AllocationOverflow)?;
        active_index += 1;
    }

    Ok(McByteAllocation { channels, bytes })
}

fn allocate_active_bytes(
    available_bits: usize,
    ratios: &[i64],
    active_channels: usize,
    has_lfe: bool,
    lfe_bytes: usize,
) -> Result<[i64; MAX_CHANNELS as usize], McError> {
    let non_lfe_channels = active_channels
        .checked_sub(usize::from(has_lfe))
        .ok_or(McError::AllocationOverflow)?;
    if ratios.len() != non_lfe_channels {
        return Err(McError::SideInfoChannelMismatch {
            expected: non_lfe_channels,
            actual: ratios.len(),
        });
    }
    if non_lfe_channels == 0 || (has_lfe && active_channels <= MC_LFE_CHANNEL_INDEX) {
        return Err(McError::InvalidChannelCount {
            channels: active_channels,
            limit: usize::from(MAX_CHANNELS),
        });
    }

    let total_bytes = available_bits
        .checked_div(8)
        .and_then(|value| value.checked_sub(lfe_bytes))
        .ok_or(McError::InsufficientAllocation {
            available_bytes: available_bits / 8,
            minimum_bytes: lfe_bytes,
        })?;
    let minimum_bytes = non_lfe_channels
        .checked_mul(MC_SILENCE_BYTES)
        .ok_or(McError::AllocationOverflow)?;
    if total_bytes < minimum_bytes {
        return Err(McError::InsufficientAllocation {
            available_bytes: total_bytes,
            minimum_bytes,
        });
    }

    let total_bytes = i64::try_from(total_bytes).map_err(|_| McError::AllocationOverflow)?;
    let mut left_bytes =
        total_bytes - i64::try_from(minimum_bytes).map_err(|_| McError::AllocationOverflow)?;
    let mut bytes = [0_i64; MAX_CHANNELS as usize];
    let mut allocated = 0_i64;
    let mut most_bytes = 0_i64;
    let mut most_bytes_channel = 0;
    let mut ratio_sum = 0_i64;

    for channel in 0..non_lfe_channels {
        bytes[channel] = ratios[channel] * left_bytes / MC_RATIO_SCOPE + MC_SILENCE_BYTES as i64;
        allocated += bytes[channel];
        if bytes[channel] > most_bytes {
            most_bytes = bytes[channel];
            most_bytes_channel = channel;
        }
        ratio_sum += ratios[channel];
    }

    if allocated != total_bytes {
        let difference = allocated - total_bytes;
        allocated = 0;
        for channel in 0..non_lfe_channels {
            bytes[channel] -= difference * ratios[channel] / MC_RATIO_SCOPE;
            bytes[channel] = bytes[channel].max(MC_SILENCE_BYTES as i64);
            allocated += bytes[channel];
        }
    }
    if allocated != total_bytes {
        bytes[most_bytes_channel] += total_bytes - allocated;
    }

    let mut capped = [false; MAX_CHANNELS as usize];
    for _ in 0..non_lfe_channels {
        let mut reallocate = None;
        for channel in 0..non_lfe_channels {
            if bytes[channel] > MC_MAX_CHANNEL_BYTES && !capped[channel] {
                left_bytes = bytes[channel] - MC_MAX_CHANNEL_BYTES;
                bytes[channel] = MC_MAX_CHANNEL_BYTES;
                capped[channel] = true;
                ratio_sum -= ratios[channel];
                reallocate = Some(left_bytes);
                break;
            }
        }

        let Some(left_bytes) = reallocate else {
            break;
        };
        if ratio_sum == 0 {
            return Err(McError::InvalidRatioSum);
        }
        allocated = 0;
        for channel in 0..non_lfe_channels {
            if capped[channel] {
                continue;
            }
            let increment = ratios[channel] * left_bytes / ratio_sum;
            allocated += increment;
            bytes[channel] += increment;
        }
        if allocated != left_bytes {
            bytes[most_bytes_channel] += left_bytes - allocated;
        }
    }

    if has_lfe {
        for channel in (MC_LFE_CHANNEL_INDEX + 1..active_channels).rev() {
            bytes[channel] = bytes[channel - 1];
        }
        bytes[MC_LFE_CHANNEL_INDEX] =
            i64::try_from(lfe_bytes).map_err(|_| McError::AllocationOverflow)?;
    }
    Ok(bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McFrameSideInfo<'decoder> {
    channels: usize,
    cores: [Option<CoreSideInfo>; MAX_CHANNELS as usize],
    mc: McSideInfo,
    neural_qc: [Option<ParsedNeuralQc<'decoder>>; MAX_CHANNELS as usize],
    allocation: McByteAllocation,
    consumed_bits: usize,
    padding_bits: usize,
}

impl<'decoder> McFrameSideInfo<'decoder> {
    pub fn channels(self) -> usize {
        self.channels
    }

    pub fn core(self, channel: usize) -> Option<CoreSideInfo> {
        self.cores.get(channel).copied().flatten()
    }

    pub fn mc(self) -> McSideInfo {
        self.mc
    }

    pub fn neural_qc(self, channel: usize) -> Option<ParsedNeuralQc<'decoder>> {
        self.neural_qc.get(channel).copied().flatten()
    }

    pub fn allocation(self) -> McByteAllocation {
        self.allocation
    }

    pub fn consumed_bits(self) -> usize {
        self.consumed_bits
    }

    pub fn padding_bits(self) -> usize {
        self.padding_bits
    }
}

/// Allocation-stable parser for channel-based multichannel payloads.
///
/// It follows the C wire order exactly: every core prefix, every grouping
/// record, MC mode side information, then one neural QC record per channel.
#[derive(Debug, Clone)]
pub struct McSideInfoDecoder {
    context: [[u8; MAX_QC_BITSTREAM_BYTES]; MAX_CHANNELS as usize],
    base: [[u8; MAX_QC_BITSTREAM_BYTES]; MAX_CHANNELS as usize],
}

impl McSideInfoDecoder {
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
    ) -> Result<McFrameSideInfo<'decoder>, McError> {
        let config = McBitstreamConfig::for_header(header)?;
        let available_bits = payload.len().saturating_mul(8);
        if header.payload_bits > available_bits {
            return Err(CoreBitstreamError::PayloadTooShort {
                declared_bits: header.payload_bits,
                available_bits,
            }
            .into());
        }

        let channels = config.channels;
        let mut reader = BitReader::with_bit_len(payload, header.payload_bits)?;
        let mut prefixes: [Option<CoreSideInfoPrefix>; MAX_CHANNELS as usize] =
            [None; MAX_CHANNELS as usize];
        for prefix in &mut prefixes[..channels] {
            *prefix = Some(parse_core_side_prefix(&mut reader, config.core)?);
        }

        let mut groupings = [WindowGrouping::single(); MAX_CHANNELS as usize];
        let mut cores = [None; MAX_CHANNELS as usize];
        for channel in 0..channels {
            let prefix = prefixes[channel].expect("all configured MC prefixes parsed");
            let grouping = parse_grouping(&mut reader, prefix.transform_type())?;
            groupings[channel] = grouping;
            cores[channel] = Some(prefix.finish(grouping));
        }

        let mc = McSideInfo::parse(&mut reader, config)?;
        let mut reserved_bits = 0_usize;
        for grouping in &groupings[..channels] {
            reserved_bits = reserved_bits
                .checked_add(qc_side_bits(header.nn_type, grouping.count())?)
                .ok_or(McError::AllocationOverflow)?;
        }
        let used_and_reserved = reader
            .position()
            .checked_add(reserved_bits)
            .ok_or(McError::AllocationOverflow)?;
        let available_qc_bits = header.payload_bits.checked_sub(used_and_reserved).ok_or(
            CoreBitstreamError::QcBudgetUnderflow {
                payload_bits: header.payload_bits,
                used_bits: reader.position(),
                reserved_bits,
            },
        )?;
        let allocation = mc_bytes_allocation(available_qc_bits, mc, config)?;

        let mut neural_qc = [None; MAX_CHANNELS as usize];
        for (channel, (((context, base), slot), grouping)) in self.context[..channels]
            .iter_mut()
            .zip(&mut self.base[..channels])
            .zip(&mut neural_qc[..channels])
            .zip(&groupings[..channels])
            .enumerate()
        {
            *slot = Some(parse_neural_qc(
                &mut reader,
                config.core,
                *grouping,
                allocation.bytes[channel],
                context,
                base,
            )?);
        }

        let consumed_bits = reader.position();
        let padding_bits = reader.remaining();
        debug_assert!(padding_bits < 8);
        Ok(McFrameSideInfo {
            channels,
            cores,
            mc,
            neural_qc,
            allocation,
            consumed_bits,
            padding_bits,
        })
    }
}

impl Default for McSideInfoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;
    use crate::header::{AudioCodecId, BitDepth, NnType};

    fn header(payload_bits: usize) -> FrameHeader {
        FrameHeader {
            codec_id: AudioCodecId::Avs3P3,
            nn_type: NnType::Main,
            profile: CodecProfile::ChannelBased,
            sample_rate: 48_000,
            bit_depth: BitDepth::Sixteen,
            channel_config: Some(ChannelConfig::Mc5_1),
            sound_bed_type: None,
            hoa_order: None,
            objects: 0,
            bed_channels: 6,
            channels: 6,
            has_lfe: true,
            bed_bitrate: Some(384_000),
            object_bitrate: None,
            bitrate: 384_000,
            crc: 0,
            header_len: 7,
            payload_bits,
            payload_len: payload_bits.div_ceil(8),
            frame_len: 7 + payload_bits.div_ceil(8),
            samples_per_channel: 1_024,
        }
    }

    fn write(writer: &mut BitWriter, value: u64, width: usize) {
        writer.write_bits(value, width).unwrap();
    }

    #[test]
    fn pair_indexes_cover_lexicographic_channel_combinations() {
        assert_eq!(mc_pair_index_bits(2).unwrap(), 1);
        assert_eq!(mc_pair_index_bits(6).unwrap(), 4);
        assert_eq!(mc_pair_index_bits(12).unwrap(), 7);

        let pairs = (0..6)
            .map(|index| mc_pair_from_index(index, 4).unwrap())
            .map(|pair| (pair.first(), pair.second()))
            .collect::<Vec<_>>();
        assert_eq!(pairs, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
        assert!(matches!(
            mc_pair_from_index(6, 4),
            Err(McError::InvalidPairIndex {
                index: 6,
                combinations: 6
            })
        ));
    }

    #[test]
    fn coupling_order_moves_lfe_to_the_end_and_round_trips() {
        let config = McBitstreamConfig::for_header(&header(512)).unwrap();
        let expected_output_order = [0, 1, 2, 4, 5, 3];
        for (coupling, output) in expected_output_order.into_iter().enumerate() {
            assert_eq!(
                mc_coupling_channel_to_output(coupling, config).unwrap(),
                output
            );
            assert_eq!(
                mc_output_channel_to_coupling(output, config).unwrap(),
                coupling
            );
        }

        let mut no_lfe = header(512);
        no_lfe.channel_config = Some(ChannelConfig::Mc4_0);
        no_lfe.channels = 4;
        no_lfe.bed_channels = 4;
        no_lfe.has_lfe = false;
        no_lfe.bed_bitrate = Some(192_000);
        no_lfe.bitrate = 192_000;
        let config = McBitstreamConfig::for_header(&no_lfe).unwrap();
        for channel in 0..4 {
            assert_eq!(
                mc_coupling_channel_to_output(channel, config).unwrap(),
                channel
            );
            assert_eq!(
                mc_output_channel_to_coupling(channel, config).unwrap(),
                channel
            );
        }
    }

    #[test]
    fn mix_layout_keeps_lfe_between_bed_and_object_channels() {
        let mut mixed = header(512);
        mixed.profile = CodecProfile::Mixed;
        mixed.sound_bed_type = Some(SoundBedType::ChannelBed);
        mixed.objects = 2;
        mixed.channels = 8;
        mixed.bed_bitrate = Some(96_000);
        mixed.object_bitrate = Some(192_000);
        mixed.bitrate = 480_000;
        mixed.header_len = 9;
        mixed.frame_len = 9 + mixed.payload_len;

        let config = McBitstreamConfig::for_header(&mixed).unwrap();
        assert_eq!(config.channels(), 8);
        assert_eq!(config.bed_channels(), 6);
        assert_eq!(config.ild_channels(), 5);
        assert_eq!(config.lfe_bytes(), 10);
        let expected_output_order = [0, 1, 2, 4, 5, 3, 6, 7];
        for (coupling, output) in expected_output_order.into_iter().enumerate() {
            assert_eq!(
                mc_coupling_channel_to_output(coupling, config).unwrap(),
                output
            );
            assert_eq!(
                mc_output_channel_to_coupling(output, config).unwrap(),
                coupling
            );
        }

        mixed.bitrate += 1;
        assert!(matches!(
            McBitstreamConfig::for_header(&mixed),
            Err(McError::InvalidBitrateConfiguration { .. })
        ));
    }

    #[test]
    fn object_only_mix_does_not_apply_bed_ild_scalars() {
        let mut objects = header(512);
        objects.profile = CodecProfile::Mixed;
        objects.channel_config = None;
        objects.sound_bed_type = Some(SoundBedType::ObjectsOnly);
        objects.objects = 3;
        objects.bed_channels = 0;
        objects.channels = 3;
        objects.has_lfe = false;
        objects.bed_bitrate = None;
        objects.object_bitrate = Some(64_000);
        objects.bitrate = 192_000;
        objects.header_len = 8;
        objects.frame_len = 8 + objects.payload_len;

        let config = McBitstreamConfig::for_header(&objects).unwrap();
        assert_eq!(config.bed_channels(), 0);
        assert_eq!(config.ild_channels(), 0);
        let mut spectra = [[1.0_f32; AVS3_FEATURE_DIMENSIONS]; 3];
        let mut side = McSideInfo {
            channels: 3,
            has_silence_flags: false,
            silence: [false; MAX_CHANNELS as usize],
            pairs: [McPair::default(); MAX_MC_PAIRS],
            pair_count: 0,
            ild_indexes: [0; MAX_CHANNELS as usize],
            bit_ratios: [None; MAX_CHANNELS as usize],
        };
        side.ild_indexes[..3].copy_from_slice(&[0, 1, 2]);
        inverse_mc_coupling(&mut spectra, side, config).unwrap();
        assert!(spectra.iter().flatten().all(|&value| value == 1.0));
    }

    #[test]
    fn inverse_coupling_matches_c_reference_order_and_arithmetic() {
        let config = McBitstreamConfig::for_header(&header(512)).unwrap();
        let mut spectra = [[0.0_f32; AVS3_FEATURE_DIMENSIONS]; 6];
        for coupling_channel in 0..6 {
            let output_channel = mc_coupling_channel_to_output(coupling_channel, config).unwrap();
            for (line, value) in spectra[output_channel].iter_mut().enumerate() {
                let raw = (((coupling_channel as i32 + 3) * 97 + line as i32 * 13) % 257) - 128;
                *value = raw as f32 * 0.03125;
            }
        }

        let mut side = McSideInfo {
            channels: 6,
            has_silence_flags: false,
            silence: [false; MAX_CHANNELS as usize],
            pairs: [McPair::default(); MAX_MC_PAIRS],
            pair_count: 3,
            ild_indexes: [MC_NO_ILD_INDEX; MAX_CHANNELS as usize],
            bit_ratios: [None; MAX_CHANNELS as usize],
        };
        side.pairs[..3].copy_from_slice(&[
            McPair {
                first: 0,
                second: 4,
            },
            McPair {
                first: 1,
                second: 5,
            },
            McPair {
                first: 0,
                second: 2,
            },
        ]);
        side.ild_indexes[..6].copy_from_slice(&[0, 1, 29, 30, 16, 30]);

        inverse_mc_coupling(&mut spectra, side, config).unwrap();
        assert_eq!(
            spectra.each_ref().map(|spectrum| fingerprint(spectrum)),
            [
                0xc50d_d29c_736e_ed85,
                0x3f14_5375_6000_cb47,
                0x2221_5ff7_1797_94c5,
                0x9685_b4e8_90de_fe95,
                0x520e_c4fe_74e8_7325,
                0x6bd4_df5a_e466_a003,
            ]
        );
    }

    #[test]
    fn lfe_processing_preserves_only_normative_low_frequency_lines() {
        let mut spectrum = [1.0_f32; AVS3_FEATURE_DIMENSIONS];
        clear_mc_lfe_spectrum(&mut spectrum).unwrap();
        assert!(
            spectrum[..MC_LFE_RESERVED_LINES]
                .iter()
                .all(|&value| value == 1.0)
        );
        assert!(
            spectrum[MC_LFE_RESERVED_LINES..]
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(matches!(
            clear_mc_lfe_spectrum(&mut [0.0; 31]),
            Err(McError::InvalidSpectrumLength {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: 31
            })
        ));
    }

    #[test]
    fn parses_silence_pairs_ild_and_compact_ratios() {
        let config = McBitstreamConfig::for_header(&header(512)).unwrap();
        let mut writer = BitWriter::new();
        write(&mut writer, 1, 1);
        for flag in [0, 1, 0, 0, 0] {
            write(&mut writer, flag, 1);
        }
        write(&mut writer, 2, 4);
        write(&mut writer, 0, 4);
        write(&mut writer, 0, 5);
        write(&mut writer, 1, 5);
        write(&mut writer, 14, 4);
        write(&mut writer, 29, 5);
        write(&mut writer, 30, 5);
        for ratio in [16, 17, 18, 19] {
            write(&mut writer, ratio, 6);
        }
        assert_eq!(writer.bit_len(), 62);
        let payload = writer.into_bytes();
        let mut reader = BitReader::with_bit_len(&payload, 62).unwrap();
        let side = McSideInfo::parse(&mut reader, config).unwrap();

        assert_eq!(
            side.silence_flags(),
            &[false, true, false, false, false, false]
        );
        assert_eq!(
            side.pairs()
                .iter()
                .map(|pair| (pair.first(), pair.second()))
                .collect::<Vec<_>>(),
            [(0, 1), (4, 5)]
        );
        assert_eq!(side.ild_indexes(), &[0, 1, 30, 30, 29, 30]);
        assert_eq!(
            side.bit_ratios(),
            &[Some(16), None, Some(17), None, Some(18), Some(19)]
        );
        assert_eq!(reader.remaining(), 0);
    }

    #[test]
    fn allocation_matches_reference_integer_order() {
        let config = McBitstreamConfig::for_header(&header(2_000)).unwrap();
        let side = McSideInfo {
            channels: 6,
            has_silence_flags: true,
            silence: [
                false, true, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false,
            ],
            pairs: [McPair::default(); MAX_MC_PAIRS],
            pair_count: 0,
            ild_indexes: [MC_NO_ILD_INDEX; MAX_CHANNELS as usize],
            bit_ratios: [
                Some(16),
                None,
                Some(16),
                None,
                Some(16),
                Some(16),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        };
        let allocation = mc_bytes_allocation(1_524, side, config).unwrap();
        assert_eq!(allocation.channel_bytes(), &[42, 8, 40, 20, 40, 40]);
    }

    #[test]
    fn parses_complete_six_channel_wire_order() {
        let payload_bits = 2_000;
        let header = header(payload_bits);
        let mut writer = BitWriter::new();

        for _ in 0..6 {
            write(&mut writer, 0, 2); // long transform
            for width in [8, 8, 7, 7, 6, 5, 5] {
                write(&mut writer, 0, width);
            }
            write(&mut writer, 0, 1);
            write(&mut writer, 0, 1);
        }

        write(&mut writer, 1, 1);
        for flag in [0, 1, 0, 0, 0] {
            write(&mut writer, flag, 1);
        }
        write(&mut writer, 2, 4);
        write(&mut writer, 0, 4);
        write(&mut writer, 0, 5);
        write(&mut writer, 1, 5);
        write(&mut writer, 14, 4);
        write(&mut writer, 29, 5);
        write(&mut writer, 30, 5);
        for _ in 0..4 {
            write(&mut writer, 16, 6);
        }
        assert_eq!(writer.bit_len(), 362);

        for entropy_bytes in [42, 8, 40, 20, 40, 40] {
            write(&mut writer, 0, 1);
            write(&mut writer, 37, 7);
            write(&mut writer, 3, 3);
            write(&mut writer, 0, 8);
            for _ in 0..entropy_bytes {
                write(&mut writer, 0, 8);
            }
        }
        assert_eq!(writer.bit_len(), 1_996);
        write(&mut writer, 0, 4);
        let payload = writer.into_bytes();

        let mut decoder = McSideInfoDecoder::new();
        let parsed = decoder.parse(&payload, &header).unwrap();
        assert_eq!(parsed.channels(), 6);
        assert_eq!(
            parsed.allocation().channel_bytes(),
            &[42, 8, 40, 20, 40, 40]
        );
        assert_eq!(parsed.consumed_bits(), 1_996);
        assert_eq!(parsed.padding_bits(), 4);
        for channel in 0..6 {
            assert!(parsed.core(channel).is_some());
            assert!(parsed.neural_qc(channel).is_some());
        }
    }

    fn fingerprint(values: &[f32]) -> u64 {
        values
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
                (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3)
            })
    }
}
