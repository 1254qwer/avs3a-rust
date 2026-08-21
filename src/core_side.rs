use core::fmt;

use crate::bitstream::BitReader;
use crate::error::BitstreamError;
use crate::header::{FrameHeader, NnType};
use crate::neural_qc::{
    AVS3_SHORT_BLOCKS, LowComplexityNeuralQc, MAX_QC_BITSTREAM_BYTES, MainNeuralQc,
    NeuralBitstreams, NeuralQcError, NoiseFilling, NoiseGroup,
};

pub const MAX_LSF_CODEBOOKS: usize = 7;
pub const MAX_TNS_FILTERS: usize = 2;
pub const MAX_TNS_ORDER: usize = 8;
pub const MAX_BWE_TILES: usize = 4;
pub const MAX_BWE_SCALE_FACTOR_BANDS: usize = 8;

const HBR_LSF_WIDTHS: [usize; 7] = [8, 8, 7, 7, 6, 5, 5];
const LBR_LSF_WIDTHS: [usize; 5] = [8, 8, 7, 7, 6];
const LSF_LOW_BITRATE_THRESHOLD: u32 = 32_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreBitstreamError {
    NotMono {
        channels: u8,
    },
    UnsupportedMonoBweBitrate(u32),
    UnsupportedHoaBweBitrate {
        order: u8,
        bitrate: u32,
    },
    InvalidCoreChannelCount(usize),
    InvalidBweConfig {
        tiles: usize,
        scale_factor_bands: usize,
    },
    PayloadTooShort {
        declared_bits: usize,
        available_bits: usize,
    },
    QcBudgetUnderflow {
        payload_bits: usize,
        used_bits: usize,
        reserved_bits: usize,
    },
    EntropyPayloadTooLarge {
        bytes: usize,
        limit: usize,
    },
    ContextLengthExceedsEntropyPayload {
        context_bytes: usize,
        entropy_bytes: usize,
    },
    InvalidTnsCode {
        coefficient: usize,
        code: u16,
        bits: u8,
    },
    IntegerOverflow,
    Bitstream(BitstreamError),
    NeuralQc(NeuralQcError),
}

impl fmt::Display for CoreBitstreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotMono { channels } => {
                write!(
                    f,
                    "mono core parser requires one channel, header has {channels}"
                )
            }
            Self::UnsupportedMonoBweBitrate(bitrate) => {
                write!(f, "mono bitrate {bitrate} has no AVS3 BWE configuration")
            }
            Self::UnsupportedHoaBweBitrate { order, bitrate } => write!(
                f,
                "HOA order {order} bitrate {bitrate} has no AVS3 BWE configuration"
            ),
            Self::InvalidCoreChannelCount(channels) => {
                write!(f, "core channel count {channels} is invalid")
            }
            Self::InvalidBweConfig {
                tiles,
                scale_factor_bands,
            } => write!(
                f,
                "BWE configuration has {tiles} tiles and {scale_factor_bands} scale-factor bands"
            ),
            Self::PayloadTooShort {
                declared_bits,
                available_bits,
            } => write!(
                f,
                "core payload declares {declared_bits} bits, only {available_bits} are present"
            ),
            Self::QcBudgetUnderflow {
                payload_bits,
                used_bits,
                reserved_bits,
            } => write!(
                f,
                "QC budget underflow: payload has {payload_bits} bits, core used {used_bits}, QC side information needs {reserved_bits}"
            ),
            Self::EntropyPayloadTooLarge { bytes, limit } => write!(
                f,
                "neural entropy payload has {bytes} bytes; fixed decoder limit is {limit}"
            ),
            Self::ContextLengthExceedsEntropyPayload {
                context_bytes,
                entropy_bytes,
            } => write!(
                f,
                "context stream declares {context_bytes} bytes, but the channel has only {entropy_bytes} entropy bytes"
            ),
            Self::InvalidTnsCode {
                coefficient,
                code,
                bits,
            } => write!(
                f,
                "invalid TNS Huffman prefix 0x{code:x}/{bits} bits for coefficient {coefficient}"
            ),
            Self::IntegerOverflow => f.write_str("core bitstream size arithmetic overflow"),
            Self::Bitstream(error) => error.fmt(f),
            Self::NeuralQc(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for CoreBitstreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(error) => Some(error),
            Self::NeuralQc(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BitstreamError> for CoreBitstreamError {
    fn from(value: BitstreamError) -> Self {
        Self::Bitstream(value)
    }
}

impl From<NeuralQcError> for CoreBitstreamError {
    fn from(value: NeuralQcError) -> Self {
        Self::NeuralQc(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformType {
    Long,
    Short,
    LongToShort,
    ShortToLong,
}

impl TransformType {
    fn from_wire(value: u8) -> Self {
        match value {
            0 => Self::Long,
            1 => Self::Short,
            2 => Self::LongToShort,
            3 => Self::ShortToLong,
            _ => unreachable!("two-bit transform type"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsfCodebookMode {
    HighBitrate,
    LowBitrate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LsfSideInfo {
    mode: LsfCodebookMode,
    indexes: [u16; MAX_LSF_CODEBOOKS],
    count: usize,
}

impl LsfSideInfo {
    pub fn mode(self) -> LsfCodebookMode {
        self.mode
    }

    pub fn indexes(&self) -> &[u16] {
        &self.indexes[..self.count]
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TnsCoefficient {
    index: i8,
    code: u16,
    bits: u8,
}

impl TnsCoefficient {
    /// Quantized PARCOR index in the C decoder's `-8..=7` range.
    pub fn index(self) -> i8 {
        self.index
    }

    pub fn code(self) -> u16 {
        self.code
    }

    pub fn bits(self) -> u8 {
        self.bits
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TnsFilterSideInfo {
    enabled: bool,
    order: usize,
    coefficients: [TnsCoefficient; MAX_TNS_ORDER],
}

impl TnsFilterSideInfo {
    pub fn enabled(self) -> bool {
        self.enabled
    }

    pub fn order(self) -> usize {
        self.order
    }

    pub fn coefficients(&self) -> &[TnsCoefficient] {
        &self.coefficients[..self.order]
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TnsSideInfo {
    filters: [TnsFilterSideInfo; MAX_TNS_FILTERS],
}

impl TnsSideInfo {
    pub fn filters(&self) -> &[TnsFilterSideInfo; MAX_TNS_FILTERS] {
        &self.filters
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BweWhiteningLevel {
    Off,
    Mid,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BweConfig {
    num_tiles: usize,
    num_scale_factor_bands: usize,
    start_line: usize,
    stop_line: usize,
    target_tiles: [u16; MAX_BWE_TILES + 1],
    source_tiles: [u16; MAX_BWE_TILES],
    scale_factor_bands: [u16; MAX_BWE_SCALE_FACTOR_BANDS + 1],
    scale_factor_tile_wrap: [u8; MAX_BWE_TILES + 1],
}

impl BweConfig {
    pub fn for_mono_bitrate(bitrate: u32) -> Result<Option<Self>, CoreBitstreamError> {
        let config = if bitrate <= 32_000 {
            MONO_BWE_32K
        } else if matches!(bitrate, 44_000 | 56_000) {
            MONO_BWE_48K
        } else if matches!(bitrate, 64_000 | 72_000) {
            MONO_BWE_64K
        } else if matches!(bitrate, 80_000 | 96_000) {
            MONO_BWE_96K
        } else if bitrate > 96_000 {
            return Ok(None);
        } else {
            return Err(CoreBitstreamError::UnsupportedMonoBweBitrate(bitrate));
        };
        config.validate()?;
        Ok(Some(config))
    }

    pub fn for_stereo_bitrate(bitrate: u32) -> Result<Option<Self>, CoreBitstreamError> {
        let config = if bitrate <= 64_000 {
            STEREO_BWE_64K
        } else if bitrate <= 96_000 {
            STEREO_BWE_96K
        } else if bitrate <= 128_000 {
            STEREO_BWE_128K
        } else {
            return Ok(None);
        };
        config.validate()?;
        Ok(Some(config))
    }

    pub fn for_multichannel_bitrate(
        total_bitrate: u32,
        core_channels: usize,
    ) -> Result<Option<Self>, CoreBitstreamError> {
        if core_channels == 0 {
            return Err(CoreBitstreamError::InvalidCoreChannelCount(core_channels));
        }
        let bitrate_per_cpe = u64::from(total_bitrate)
            .checked_mul(2)
            .ok_or(CoreBitstreamError::IntegerOverflow)?
            / u64::try_from(core_channels).map_err(|_| CoreBitstreamError::IntegerOverflow)?;
        let config = if bitrate_per_cpe <= 56_000 {
            MC_BWE_48K
        } else if bitrate_per_cpe <= 75_000 {
            MC_BWE_64K
        } else if bitrate_per_cpe <= 108_000 {
            MC_BWE_96K
        } else if bitrate_per_cpe <= 128_000 {
            MC_BWE_128K
        } else {
            return Ok(None);
        };
        config.validate()?;
        Ok(Some(config))
    }

    pub fn for_hoa_bitrate(order: u8, bitrate: u32) -> Result<Self, CoreBitstreamError> {
        let config = match (order, bitrate) {
            (1, 0..=128_000) => HOA_BWE_LOW,
            (1, 192_000) => HOA_BWE_MIDDLE,
            (1, 256_000) => HOA_BWE_HIGH,
            (2, 192_000) => HOA_BWE_ELOW,
            (2, 256_000) => HOA_BWE_LOW,
            (2, 320_000) => HOA_BWE_MIDDLE,
            (2, 384_000 | 480_000) => HOA_BWE_HIGH,
            (3, 256_000 | 320_000 | 384_000) => HOA_BWE_LOW,
            (3, 512_000) => HOA_BWE_MIDDLE,
            (3, 640_000 | 896_000) => HOA_BWE_HIGH,
            _ => return Err(CoreBitstreamError::UnsupportedHoaBweBitrate { order, bitrate }),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn num_tiles(self) -> usize {
        self.num_tiles
    }

    pub fn num_scale_factor_bands(self) -> usize {
        self.num_scale_factor_bands
    }

    pub fn start_line(self) -> usize {
        self.start_line
    }

    pub fn stop_line(self) -> usize {
        self.stop_line
    }

    pub fn target_tiles(&self) -> &[u16] {
        &self.target_tiles[..=self.num_tiles]
    }

    pub fn source_tiles(&self) -> &[u16] {
        &self.source_tiles[..self.num_tiles]
    }

    pub fn scale_factor_bands(&self) -> &[u16] {
        &self.scale_factor_bands[..=self.num_scale_factor_bands]
    }

    pub fn scale_factor_tile_wrap(&self) -> &[u8] {
        &self.scale_factor_tile_wrap[..=self.num_tiles]
    }

    fn validate(self) -> Result<(), CoreBitstreamError> {
        if self.num_tiles == 0
            || self.num_tiles > MAX_BWE_TILES
            || self.num_scale_factor_bands == 0
            || self.num_scale_factor_bands > MAX_BWE_SCALE_FACTOR_BANDS
        {
            return Err(CoreBitstreamError::InvalidBweConfig {
                tiles: self.num_tiles,
                scale_factor_bands: self.num_scale_factor_bands,
            });
        }
        Ok(())
    }
}

const MONO_BWE_32K: BweConfig = BweConfig {
    num_tiles: 3,
    num_scale_factor_bands: 6,
    start_line: 352,
    stop_line: 768,
    target_tiles: [352, 480, 608, 768, 0],
    source_tiles: [64, 96, 144, 0],
    scale_factor_bands: [352, 416, 480, 544, 608, 672, 768, 0, 0],
    scale_factor_tile_wrap: [0, 2, 4, 6, 0],
};

const MONO_BWE_48K: BweConfig = BweConfig {
    num_tiles: 3,
    num_scale_factor_bands: 6,
    start_line: 448,
    stop_line: 832,
    target_tiles: [448, 544, 672, 832, 0],
    source_tiles: [96, 144, 192, 0],
    scale_factor_bands: [448, 496, 544, 608, 672, 736, 832, 0, 0],
    scale_factor_tile_wrap: [0, 2, 4, 6, 0],
};

const MONO_BWE_64K: BweConfig = BweConfig {
    num_tiles: 2,
    num_scale_factor_bands: 4,
    start_line: 544,
    stop_line: 832,
    target_tiles: [544, 672, 832, 0, 0],
    source_tiles: [144, 192, 0, 0],
    scale_factor_bands: [544, 608, 672, 736, 832, 0, 0, 0, 0],
    scale_factor_tile_wrap: [0, 2, 4, 0, 0],
};

const MONO_BWE_96K: BweConfig = BweConfig {
    num_tiles: 1,
    num_scale_factor_bands: 2,
    start_line: 672,
    stop_line: 832,
    target_tiles: [672, 832, 0, 0, 0],
    source_tiles: [192, 0, 0, 0],
    scale_factor_bands: [672, 736, 832, 0, 0, 0, 0, 0, 0],
    scale_factor_tile_wrap: [0, 2, 0, 0, 0],
};

// The reference's 48 and 64 kbps stereo rows are identical.
const STEREO_BWE_64K: BweConfig = BweConfig {
    num_tiles: 3,
    num_scale_factor_bands: 6,
    start_line: 352,
    stop_line: 768,
    target_tiles: [352, 480, 608, 768, 0],
    source_tiles: [64, 96, 144, 0],
    scale_factor_bands: [352, 416, 480, 544, 608, 672, 768, 0, 0],
    scale_factor_tile_wrap: [0, 2, 4, 6, 0],
};

const STEREO_BWE_96K: BweConfig = BweConfig {
    num_tiles: 2,
    num_scale_factor_bands: 4,
    start_line: 544,
    stop_line: 832,
    target_tiles: [544, 672, 832, 0, 0],
    source_tiles: [144, 192, 0, 0],
    scale_factor_bands: [544, 608, 672, 736, 832, 0, 0, 0, 0],
    scale_factor_tile_wrap: [0, 2, 4, 0, 0],
};

const STEREO_BWE_128K: BweConfig = BweConfig {
    num_tiles: 1,
    num_scale_factor_bands: 2,
    start_line: 672,
    stop_line: 832,
    target_tiles: [672, 832, 0, 0, 0],
    source_tiles: [192, 0, 0, 0],
    scale_factor_bands: [672, 736, 832, 0, 0, 0, 0, 0, 0],
    scale_factor_tile_wrap: [0, 2, 0, 0, 0],
};

const MC_BWE_48K: BweConfig = BweConfig {
    num_tiles: 3,
    num_scale_factor_bands: 6,
    start_line: 352,
    stop_line: 768,
    target_tiles: [352, 448, 576, 768, 0],
    source_tiles: [64, 96, 144, 0],
    scale_factor_bands: [352, 400, 448, 512, 576, 672, 768, 0, 0],
    scale_factor_tile_wrap: [0, 2, 4, 6, 0],
};

const MC_BWE_64K: BweConfig = BweConfig {
    num_tiles: 3,
    num_scale_factor_bands: 5,
    start_line: 400,
    stop_line: 768,
    target_tiles: [400, 512, 672, 768, 0],
    source_tiles: [64, 96, 144, 0],
    scale_factor_bands: [400, 448, 512, 576, 672, 768, 0, 0, 0],
    scale_factor_tile_wrap: [0, 2, 4, 5, 0],
};

const MC_BWE_96K: BweConfig = BweConfig {
    num_tiles: 2,
    num_scale_factor_bands: 4,
    start_line: 544,
    stop_line: 832,
    target_tiles: [544, 672, 832, 0, 0],
    source_tiles: [144, 192, 0, 0],
    scale_factor_bands: [544, 608, 672, 736, 832, 0, 0, 0, 0],
    scale_factor_tile_wrap: [0, 2, 4, 0, 0],
};

const MC_BWE_128K: BweConfig = BweConfig {
    num_tiles: 1,
    num_scale_factor_bands: 2,
    start_line: 672,
    stop_line: 832,
    target_tiles: [672, 832, 0, 0, 0],
    source_tiles: [192, 0, 0, 0],
    scale_factor_bands: [672, 736, 832, 0, 0, 0, 0, 0, 0],
    scale_factor_tile_wrap: [0, 2, 0, 0, 0],
};

const HOA_BWE_ELOW: BweConfig = BweConfig {
    num_tiles: 2,
    num_scale_factor_bands: 4,
    start_line: 352,
    stop_line: 736,
    target_tiles: [352, 480, 736, 0, 0],
    source_tiles: [64, 96, 0, 0],
    scale_factor_bands: [352, 416, 480, 544, 736, 0, 0, 0, 0],
    scale_factor_tile_wrap: [0, 2, 4, 0, 0],
};

const HOA_BWE_LOW: BweConfig = BweConfig {
    num_tiles: 3,
    num_scale_factor_bands: 6,
    start_line: 384,
    stop_line: 832,
    target_tiles: [384, 512, 672, 832, 0],
    source_tiles: [96, 144, 192, 0],
    scale_factor_bands: [384, 448, 512, 576, 672, 736, 832, 0, 0],
    scale_factor_tile_wrap: [0, 2, 4, 6, 0],
};

const HOA_BWE_MIDDLE: BweConfig = BweConfig {
    num_tiles: 2,
    num_scale_factor_bands: 4,
    start_line: 544,
    stop_line: 832,
    target_tiles: [544, 672, 832, 0, 0],
    source_tiles: [144, 192, 0, 0],
    scale_factor_bands: [544, 608, 672, 736, 832, 0, 0, 0, 0],
    scale_factor_tile_wrap: [0, 2, 4, 0, 0],
};

const HOA_BWE_HIGH: BweConfig = BweConfig {
    num_tiles: 1,
    num_scale_factor_bands: 2,
    start_line: 672,
    stop_line: 832,
    target_tiles: [672, 832, 0, 0, 0],
    source_tiles: [192, 0, 0, 0],
    scale_factor_bands: [672, 736, 832, 0, 0, 0, 0, 0, 0],
    scale_factor_tile_wrap: [0, 2, 0, 0, 0],
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BweSideInfo {
    envelope_indexes: [u8; MAX_BWE_SCALE_FACTOR_BANDS],
    num_scale_factor_bands: usize,
    whitening_levels: [BweWhiteningLevel; MAX_BWE_TILES],
    num_tiles: usize,
}

impl BweSideInfo {
    pub fn envelope_indexes(&self) -> &[u8] {
        &self.envelope_indexes[..self.num_scale_factor_bands]
    }

    pub fn whitening_levels(&self) -> &[BweWhiteningLevel] {
        &self.whitening_levels[..self.num_tiles]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowGrouping {
    count: usize,
    indicator: [NoiseGroup; AVS3_SHORT_BLOCKS],
}

impl WindowGrouping {
    pub(crate) const fn single() -> Self {
        Self {
            count: 1,
            indicator: [NoiseGroup::Transient; AVS3_SHORT_BLOCKS],
        }
    }

    pub fn count(self) -> usize {
        self.count
    }

    pub fn indicator(self) -> [NoiseGroup; AVS3_SHORT_BLOCKS] {
        self.indicator
    }

    fn noise_filling(
        self,
        num_lines: usize,
        indexes: [u8; 2],
    ) -> Result<NoiseFilling, NeuralQcError> {
        if self.count == 1 {
            NoiseFilling::single(num_lines, indexes[0])
        } else {
            NoiseFilling::two_groups(num_lines, self.indicator, indexes)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreSideInfo {
    transform_type: TransformType,
    lsf: LsfSideInfo,
    tns: TnsSideInfo,
    bwe: Option<BweSideInfo>,
    grouping: WindowGrouping,
}

impl CoreSideInfo {
    pub fn transform_type(self) -> TransformType {
        self.transform_type
    }

    pub fn lsf(self) -> LsfSideInfo {
        self.lsf
    }

    pub fn tns(self) -> TnsSideInfo {
        self.tns
    }

    pub fn bwe(self) -> Option<BweSideInfo> {
        self.bwe
    }

    pub fn grouping(self) -> WindowGrouping {
        self.grouping
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CoreSideInfoPrefix {
    transform_type: TransformType,
    lsf: LsfSideInfo,
    tns: TnsSideInfo,
    bwe: Option<BweSideInfo>,
}

impl CoreSideInfoPrefix {
    pub(crate) fn transform_type(self) -> TransformType {
        self.transform_type
    }

    pub(crate) fn finish(self, grouping: WindowGrouping) -> CoreSideInfo {
        CoreSideInfo {
            transform_type: self.transform_type,
            lsf: self.lsf,
            tns: self.tns,
            bwe: self.bwe,
            grouping,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreBitstreamConfig {
    nn_type: NnType,
    payload_bits: usize,
    lsf_mode: LsfCodebookMode,
    bwe: Option<BweConfig>,
}

impl CoreBitstreamConfig {
    pub fn for_mono(header: &FrameHeader) -> Result<Self, CoreBitstreamError> {
        if header.channels != 1 {
            return Err(CoreBitstreamError::NotMono {
                channels: header.channels,
            });
        }
        let lsf_mode = if header.bitrate > LSF_LOW_BITRATE_THRESHOLD {
            LsfCodebookMode::HighBitrate
        } else {
            LsfCodebookMode::LowBitrate
        };
        Ok(Self {
            nn_type: header.nn_type,
            payload_bits: header.payload_bits,
            lsf_mode,
            bwe: BweConfig::for_mono_bitrate(header.bitrate)?,
        })
    }

    pub fn new(
        nn_type: NnType,
        payload_bits: usize,
        lsf_mode: LsfCodebookMode,
        bwe: Option<BweConfig>,
    ) -> Result<Self, CoreBitstreamError> {
        if let Some(config) = bwe {
            config.validate()?;
        }
        Ok(Self {
            nn_type,
            payload_bits,
            lsf_mode,
            bwe,
        })
    }

    pub fn nn_type(self) -> NnType {
        self.nn_type
    }

    pub fn payload_bits(self) -> usize {
        self.payload_bits
    }

    pub fn lsf_mode(self) -> LsfCodebookMode {
        self.lsf_mode
    }

    pub fn bwe(self) -> Option<BweConfig> {
        self.bwe
    }

    pub fn noise_fill_lines(self) -> usize {
        self.bwe.map_or(1_024, BweConfig::start_line)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParsedNeuralQc<'decoder> {
    Main(MainNeuralQc<'decoder>),
    LowComplexity(LowComplexityNeuralQc<'decoder>),
}

impl<'decoder> ParsedNeuralQc<'decoder> {
    pub fn nn_type(self) -> NnType {
        match self {
            Self::Main(_) => NnType::Main,
            Self::LowComplexity(_) => NnType::LowComplexity,
        }
    }

    pub fn bitstreams(self) -> NeuralBitstreams<'decoder> {
        match self {
            Self::Main(value) => value.bitstreams(),
            Self::LowComplexity(value) => value.bitstreams(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonoFrameSideInfo<'decoder> {
    core: CoreSideInfo,
    neural_qc: ParsedNeuralQc<'decoder>,
    entropy_bytes: usize,
    consumed_bits: usize,
    padding_bits: usize,
}

impl<'decoder> MonoFrameSideInfo<'decoder> {
    pub fn core(self) -> CoreSideInfo {
        self.core
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

    pub fn side_information_bits(self) -> usize {
        self.consumed_bits - self.entropy_bytes * 8
    }
}

/// Reusable mono/core side-information parser.
///
/// Context and base entropy bytes are copied into fixed buffers because AVS3
/// syntax bytes commonly start at a non-byte-aligned payload position. The
/// parser performs no heap allocation after construction.
#[derive(Debug, Clone)]
pub struct MonoSideInfoDecoder {
    context: [u8; MAX_QC_BITSTREAM_BYTES],
    base: [u8; MAX_QC_BITSTREAM_BYTES],
}

impl MonoSideInfoDecoder {
    pub fn new() -> Self {
        Self {
            context: [0; MAX_QC_BITSTREAM_BYTES],
            base: [0; MAX_QC_BITSTREAM_BYTES],
        }
    }

    pub fn parse<'decoder>(
        &'decoder mut self,
        payload: &[u8],
        config: CoreBitstreamConfig,
    ) -> Result<MonoFrameSideInfo<'decoder>, CoreBitstreamError> {
        let available_bits = payload.len().saturating_mul(8);
        if config.payload_bits > available_bits {
            return Err(CoreBitstreamError::PayloadTooShort {
                declared_bits: config.payload_bits,
                available_bits,
            });
        }
        let mut reader = BitReader::with_bit_len(payload, config.payload_bits)?;
        let prefix = parse_core_side_prefix(&mut reader, config)?;
        let grouping = parse_grouping(&mut reader, prefix.transform_type())?;

        let qc_reserved_bits = qc_side_bits(config.nn_type, grouping.count)?;
        let used_and_reserved = reader
            .position()
            .checked_add(qc_reserved_bits)
            .ok_or(CoreBitstreamError::IntegerOverflow)?;
        let entropy_bits = config.payload_bits.checked_sub(used_and_reserved).ok_or(
            CoreBitstreamError::QcBudgetUnderflow {
                payload_bits: config.payload_bits,
                used_bits: reader.position(),
                reserved_bits: qc_reserved_bits,
            },
        )?;
        let entropy_bytes = entropy_bits / 8;
        if entropy_bytes > MAX_QC_BITSTREAM_BYTES {
            return Err(CoreBitstreamError::EntropyPayloadTooLarge {
                bytes: entropy_bytes,
                limit: MAX_QC_BITSTREAM_BYTES,
            });
        }

        let neural_qc = parse_neural_qc(
            &mut reader,
            config,
            grouping,
            entropy_bytes,
            &mut self.context,
            &mut self.base,
        )?;

        let consumed_bits = reader.position();
        let padding_bits = reader.remaining();
        debug_assert!(padding_bits < 8);
        Ok(MonoFrameSideInfo {
            core: prefix.finish(grouping),
            neural_qc,
            entropy_bytes,
            consumed_bits,
            padding_bits,
        })
    }
}

impl Default for MonoSideInfoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn parse_core_side_prefix(
    reader: &mut BitReader<'_>,
    config: CoreBitstreamConfig,
) -> Result<CoreSideInfoPrefix, CoreBitstreamError> {
    let transform_type = TransformType::from_wire(reader.read_u8(2)?);
    let lsf = parse_lsf(reader, config.lsf_mode)?;
    let tns = parse_tns(reader)?;
    let bwe = config
        .bwe
        .map(|bwe_config| parse_bwe(reader, bwe_config))
        .transpose()?;
    Ok(CoreSideInfoPrefix {
        transform_type,
        lsf,
        tns,
        bwe,
    })
}

pub(crate) fn parse_neural_qc<'decoder>(
    reader: &mut BitReader<'_>,
    config: CoreBitstreamConfig,
    grouping: WindowGrouping,
    entropy_bytes: usize,
    context: &'decoder mut [u8; MAX_QC_BITSTREAM_BYTES],
    base: &'decoder mut [u8; MAX_QC_BITSTREAM_BYTES],
) -> Result<ParsedNeuralQc<'decoder>, CoreBitstreamError> {
    if entropy_bytes > MAX_QC_BITSTREAM_BYTES {
        return Err(CoreBitstreamError::EntropyPayloadTooLarge {
            bytes: entropy_bytes,
            limit: MAX_QC_BITSTREAM_BYTES,
        });
    }

    let (feature_amplified, scale_index) = match config.nn_type {
        NnType::Main => (reader.read_u8(1)? != 0, reader.read_u8(7)?),
        NnType::LowComplexity => (false, reader.read_u8(8)?),
    };
    let mut noise_indexes = [0_u8; 2];
    noise_indexes[0] = reader.read_u8(3)?;
    if grouping.count == 2 {
        noise_indexes[1] = reader.read_u8(3)?;
    }
    let context_bytes = usize::from(reader.read_u8(8)?);
    if context_bytes > entropy_bytes {
        return Err(CoreBitstreamError::ContextLengthExceedsEntropyPayload {
            context_bytes,
            entropy_bytes,
        });
    }
    let base_bytes = entropy_bytes - context_bytes;
    reader.read_bytes_into(&mut context[..context_bytes])?;
    reader.read_bytes_into(&mut base[..base_bytes])?;

    let bitstreams = NeuralBitstreams::new(&context[..context_bytes], &base[..base_bytes])?;
    let noise_filling = grouping.noise_filling(config.noise_fill_lines(), noise_indexes)?;
    match config.nn_type {
        NnType::Main => Ok(ParsedNeuralQc::Main(MainNeuralQc::new(
            bitstreams,
            noise_filling,
            feature_amplified,
            scale_index,
        )?)),
        NnType::LowComplexity => Ok(ParsedNeuralQc::LowComplexity(LowComplexityNeuralQc::new(
            bitstreams,
            noise_filling,
            scale_index,
        ))),
    }
}

fn parse_lsf(
    reader: &mut BitReader<'_>,
    mode: LsfCodebookMode,
) -> Result<LsfSideInfo, CoreBitstreamError> {
    let widths: &[usize] = match mode {
        LsfCodebookMode::HighBitrate => &HBR_LSF_WIDTHS,
        LsfCodebookMode::LowBitrate => &LBR_LSF_WIDTHS,
    };
    let mut indexes = [0_u16; MAX_LSF_CODEBOOKS];
    for (index, &width) in widths.iter().enumerate() {
        indexes[index] = reader.read_bits(width)? as u16;
    }
    Ok(LsfSideInfo {
        mode,
        indexes,
        count: widths.len(),
    })
}

fn parse_tns(reader: &mut BitReader<'_>) -> Result<TnsSideInfo, CoreBitstreamError> {
    let mut output = TnsSideInfo::default();
    for filter in &mut output.filters {
        filter.enabled = reader.read_u8(1)? != 0;
        if !filter.enabled {
            continue;
        }
        filter.order = usize::from(reader.read_u8(3)?) + 1;
        for coefficient in 0..filter.order {
            filter.coefficients[coefficient] = decode_tns_coefficient(reader, coefficient)?;
        }
    }
    Ok(output)
}

fn decode_tns_coefficient(
    reader: &mut BitReader<'_>,
    coefficient: usize,
) -> Result<TnsCoefficient, CoreBitstreamError> {
    let table = &TNS_CODES[coefficient];
    let max_bits = table.iter().map(|item| item.bits).max().unwrap_or(0);
    let mut code = 0_u16;
    for bits in 1..=max_bits {
        code = (code << 1) | u16::from(reader.read_u8(1)?);
        if let Some(index) = table
            .iter()
            .position(|item| item.bits == bits && item.code == code)
        {
            return Ok(TnsCoefficient {
                index: index as i8 - 8,
                code,
                bits,
            });
        }
    }
    Err(CoreBitstreamError::InvalidTnsCode {
        coefficient,
        code,
        bits: max_bits,
    })
}

fn parse_bwe(
    reader: &mut BitReader<'_>,
    config: BweConfig,
) -> Result<BweSideInfo, CoreBitstreamError> {
    let mut envelope_indexes = [0_u8; MAX_BWE_SCALE_FACTOR_BANDS];
    for value in &mut envelope_indexes[..config.num_scale_factor_bands] {
        *value = reader.read_u8(7)?;
    }
    let mut whitening_levels = [BweWhiteningLevel::Off; MAX_BWE_TILES];
    for value in &mut whitening_levels[..config.num_tiles] {
        *value = if reader.read_u8(1)? == 0 {
            BweWhiteningLevel::Off
        } else if reader.read_u8(1)? == 0 {
            BweWhiteningLevel::Mid
        } else {
            BweWhiteningLevel::High
        };
    }
    Ok(BweSideInfo {
        envelope_indexes,
        num_scale_factor_bands: config.num_scale_factor_bands,
        whitening_levels,
        num_tiles: config.num_tiles,
    })
}

pub(crate) fn parse_grouping(
    reader: &mut BitReader<'_>,
    transform_type: TransformType,
) -> Result<WindowGrouping, CoreBitstreamError> {
    let mut grouping = WindowGrouping {
        count: 1,
        indicator: [NoiseGroup::Transient; AVS3_SHORT_BLOCKS],
    };
    if transform_type == TransformType::Short {
        grouping.count = usize::from(reader.read_u8(1)?) + 1;
        if grouping.count == 2 {
            for group in &mut grouping.indicator {
                *group = if reader.read_u8(1)? == 0 {
                    NoiseGroup::Transient
                } else {
                    NoiseGroup::Other
                };
            }
        }
    }
    Ok(grouping)
}

pub(crate) fn qc_side_bits(nn_type: NnType, groups: usize) -> Result<usize, CoreBitstreamError> {
    let scale_bits = match nn_type {
        NnType::Main => 1_usize + 7,
        NnType::LowComplexity => 8,
    };
    let noise_bits = groups
        .checked_mul(3)
        .ok_or(CoreBitstreamError::IntegerOverflow)?;
    scale_bits
        .checked_add(noise_bits)
        .and_then(|value| value.checked_add(8))
        .ok_or(CoreBitstreamError::IntegerOverflow)
}

#[derive(Debug, Clone, Copy)]
struct TnsCode {
    code: u16,
    bits: u8,
}

const TNS_CODES: [[TnsCode; 16]; 8] = [
    [
        TnsCode {
            code: 4053,
            bits: 12,
        },
        TnsCode {
            code: 1012,
            bits: 10,
        },
        TnsCode { code: 507, bits: 9 },
        TnsCode { code: 127, bits: 7 },
        TnsCode { code: 30, bits: 5 },
        TnsCode { code: 0, bits: 3 },
        TnsCode { code: 1, bits: 3 },
        TnsCode { code: 2, bits: 3 },
        TnsCode { code: 2, bits: 2 },
        TnsCode { code: 3, bits: 3 },
        TnsCode { code: 6, bits: 3 },
        TnsCode { code: 14, bits: 4 },
        TnsCode { code: 62, bits: 6 },
        TnsCode { code: 252, bits: 8 },
        TnsCode {
            code: 2027,
            bits: 11,
        },
        TnsCode {
            code: 8105,
            bits: 13,
        },
    ],
    [
        TnsCode {
            code: 15360,
            bits: 15,
        },
        TnsCode {
            code: 7681,
            bits: 14,
        },
        TnsCode {
            code: 3841,
            bits: 13,
        },
        TnsCode {
            code: 961,
            bits: 11,
        },
        TnsCode { code: 241, bits: 9 },
        TnsCode { code: 61, bits: 7 },
        TnsCode { code: 14, bits: 5 },
        TnsCode { code: 2, bits: 3 },
        TnsCode { code: 2, bits: 2 },
        TnsCode { code: 3, bits: 2 },
        TnsCode { code: 0, bits: 2 },
        TnsCode { code: 6, bits: 4 },
        TnsCode { code: 31, bits: 6 },
        TnsCode { code: 121, bits: 8 },
        TnsCode {
            code: 481,
            bits: 10,
        },
        TnsCode {
            code: 1921,
            bits: 12,
        },
    ],
    [
        TnsCode {
            code: 27136,
            bits: 15,
        },
        TnsCode {
            code: 27137,
            bits: 15,
        },
        TnsCode {
            code: 3393,
            bits: 12,
        },
        TnsCode { code: 425, bits: 9 },
        TnsCode { code: 107, bits: 7 },
        TnsCode { code: 52, bits: 6 },
        TnsCode { code: 12, bits: 4 },
        TnsCode { code: 7, bits: 3 },
        TnsCode { code: 0, bits: 1 },
        TnsCode { code: 2, bits: 2 },
        TnsCode { code: 27, bits: 5 },
        TnsCode { code: 213, bits: 8 },
        TnsCode {
            code: 849,
            bits: 10,
        },
        TnsCode {
            code: 1697,
            bits: 11,
        },
        TnsCode {
            code: 6785,
            bits: 13,
        },
        TnsCode {
            code: 27138,
            bits: 15,
        },
    ],
    [
        TnsCode {
            code: 8708,
            bits: 14,
        },
        TnsCode {
            code: 8709,
            bits: 14,
        },
        TnsCode {
            code: 8710,
            bits: 14,
        },
        TnsCode {
            code: 1089,
            bits: 11,
        },
        TnsCode { code: 273, bits: 9 },
        TnsCode { code: 137, bits: 8 },
        TnsCode { code: 35, bits: 6 },
        TnsCode { code: 5, bits: 3 },
        TnsCode { code: 0, bits: 1 },
        TnsCode { code: 3, bits: 2 },
        TnsCode { code: 9, bits: 4 },
        TnsCode { code: 16, bits: 5 },
        TnsCode { code: 69, bits: 7 },
        TnsCode {
            code: 545,
            bits: 10,
        },
        TnsCode {
            code: 8711,
            bits: 14,
        },
        TnsCode {
            code: 4352,
            bits: 13,
        },
    ],
    [
        TnsCode {
            code: 4100,
            bits: 14,
        },
        TnsCode {
            code: 4101,
            bits: 14,
        },
        TnsCode {
            code: 4102,
            bits: 14,
        },
        TnsCode {
            code: 257,
            bits: 10,
        },
        TnsCode { code: 65, bits: 8 },
        TnsCode { code: 17, bits: 6 },
        TnsCode { code: 5, bits: 4 },
        TnsCode { code: 0, bits: 2 },
        TnsCode { code: 1, bits: 1 },
        TnsCode { code: 3, bits: 3 },
        TnsCode { code: 9, bits: 5 },
        TnsCode { code: 33, bits: 7 },
        TnsCode { code: 129, bits: 9 },
        TnsCode {
            code: 513,
            bits: 11,
        },
        TnsCode {
            code: 4103,
            bits: 14,
        },
        TnsCode {
            code: 2048,
            bits: 13,
        },
    ],
    [
        TnsCode {
            code: 8272,
            bits: 14,
        },
        TnsCode {
            code: 8273,
            bits: 14,
        },
        TnsCode {
            code: 2069,
            bits: 12,
        },
        TnsCode {
            code: 516,
            bits: 10,
        },
        TnsCode { code: 128, bits: 8 },
        TnsCode { code: 65, bits: 7 },
        TnsCode { code: 17, bits: 5 },
        TnsCode { code: 5, bits: 3 },
        TnsCode { code: 0, bits: 1 },
        TnsCode { code: 3, bits: 2 },
        TnsCode { code: 9, bits: 4 },
        TnsCode { code: 33, bits: 6 },
        TnsCode { code: 259, bits: 9 },
        TnsCode {
            code: 1035,
            bits: 11,
        },
        TnsCode {
            code: 8274,
            bits: 14,
        },
        TnsCode {
            code: 8275,
            bits: 14,
        },
    ],
    [
        TnsCode {
            code: 13312,
            bits: 14,
        },
        TnsCode {
            code: 13313,
            bits: 14,
        },
        TnsCode {
            code: 3329,
            bits: 12,
        },
        TnsCode {
            code: 833,
            bits: 10,
        },
        TnsCode { code: 209, bits: 8 },
        TnsCode { code: 53, bits: 6 },
        TnsCode { code: 12, bits: 4 },
        TnsCode { code: 2, bits: 2 },
        TnsCode { code: 0, bits: 1 },
        TnsCode { code: 7, bits: 3 },
        TnsCode { code: 27, bits: 5 },
        TnsCode { code: 105, bits: 7 },
        TnsCode { code: 417, bits: 9 },
        TnsCode {
            code: 1665,
            bits: 11,
        },
        TnsCode {
            code: 13314,
            bits: 14,
        },
        TnsCode {
            code: 13315,
            bits: 14,
        },
    ],
    [
        TnsCode {
            code: 10490,
            bits: 14,
        },
        TnsCode {
            code: 2625,
            bits: 12,
        },
        TnsCode {
            code: 657,
            bits: 10,
        },
        TnsCode { code: 165, bits: 8 },
        TnsCode { code: 83, bits: 7 },
        TnsCode { code: 21, bits: 5 },
        TnsCode { code: 4, bits: 3 },
        TnsCode { code: 3, bits: 2 },
        TnsCode {
            code: 10497,
            bits: 14,
        },
        TnsCode { code: 0, bits: 1 },
        TnsCode { code: 11, bits: 4 },
        TnsCode { code: 40, bits: 6 },
        TnsCode { code: 329, bits: 9 },
        TnsCode {
            code: 1313,
            bits: 11,
        },
        TnsCode {
            code: 10498,
            bits: 14,
        },
        TnsCode {
            code: 10499,
            bits: 14,
        },
    ],
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;

    const MAIN_REFERENCE_PAYLOAD_BITS: usize = 277;
    const MAIN_REFERENCE_PAYLOAD: [u8; 35] = [
        0x44, 0x72, 0x61, 0x63, 0xb6, 0x23, 0xa0, 0xf0, 0xea, 0x00, 0xfb, 0xdc, 0x10, 0x30, 0x2f,
        0x40, 0x2a, 0x40, 0xc0, 0xff, 0x6f, 0x01, 0xc7, 0xe9, 0x5f, 0x03, 0x84, 0xa0, 0xd8, 0x7f,
        0xfd, 0x51, 0xf6, 0xf2, 0x00,
    ];

    const LC_REFERENCE_PAYLOAD_BITS: usize = 126;
    const LC_REFERENCE_PAYLOAD: [u8; 16] = [
        0x80, 0xfe, 0xa0, 0x8c, 0xcd, 0x1f, 0xa9, 0x5b, 0xa0, 0x42, 0x46, 0x8a, 0xcf, 0x13, 0x57,
        0x80,
    ];

    fn main_reference_config() -> CoreBitstreamConfig {
        CoreBitstreamConfig::new(
            NnType::Main,
            MAIN_REFERENCE_PAYLOAD_BITS,
            LsfCodebookMode::HighBitrate,
            BweConfig::for_mono_bitrate(64_000).unwrap(),
        )
        .unwrap()
    }

    fn lc_reference_config() -> CoreBitstreamConfig {
        CoreBitstreamConfig::new(
            NnType::LowComplexity,
            LC_REFERENCE_PAYLOAD_BITS,
            LsfCodebookMode::LowBitrate,
            None,
        )
        .unwrap()
    }

    #[test]
    fn mono_bwe_configs_match_reference_tables() {
        let cases = [
            (16_000, 3, 6, 352),
            (44_000, 3, 6, 448),
            (64_000, 2, 4, 544),
            (96_000, 1, 2, 672),
        ];
        for (bitrate, tiles, bands, start) in cases {
            let config = BweConfig::for_mono_bitrate(bitrate).unwrap().unwrap();
            assert_eq!(config.num_tiles(), tiles);
            assert_eq!(config.num_scale_factor_bands(), bands);
            assert_eq!(config.start_line(), start);
        }
        assert!(BweConfig::for_mono_bitrate(128_000).unwrap().is_none());
        assert_eq!(
            BweConfig::for_mono_bitrate(48_000).unwrap_err(),
            CoreBitstreamError::UnsupportedMonoBweBitrate(48_000)
        );
    }

    #[test]
    fn parses_minimal_lc_long_frame() {
        let mut writer = BitWriter::new();
        writer.write_bits(0, 2).unwrap();
        for (value, width) in [3_u64, 5, 7, 9, 11].into_iter().zip(LBR_LSF_WIDTHS) {
            writer.write_bits(value, width).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(91, 8).unwrap();
        writer.write_bits(5, 3).unwrap();
        writer.write_bits(2, 8).unwrap();
        writer.write_bits(0x12, 8).unwrap();
        writer.write_bits(0x34, 8).unwrap();
        writer.write_bits(0x56, 8).unwrap();
        writer.write_bits(0x78, 8).unwrap();
        let payload_bits = writer.bit_len();
        let mut payload = writer.into_bytes();
        payload.push(0xff);

        let config = CoreBitstreamConfig::new(
            NnType::LowComplexity,
            payload_bits,
            LsfCodebookMode::LowBitrate,
            None,
        )
        .unwrap();
        let mut decoder = MonoSideInfoDecoder::new();
        let parsed = decoder.parse(&payload, config).unwrap();
        assert_eq!(parsed.core().transform_type(), TransformType::Long);
        assert_eq!(parsed.core().lsf().indexes(), [3, 5, 7, 9, 11]);
        assert_eq!(parsed.entropy_bytes(), 4);
        assert_eq!(parsed.padding_bits(), 0);
        let streams = parsed.neural_qc().bitstreams();
        assert_eq!(streams.context(), [0x12, 0x34]);
        assert_eq!(streams.base(), [0x56, 0x78]);
    }

    #[test]
    fn main_side_information_matches_c_reference_vector() {
        let mut decoder = MonoSideInfoDecoder::new();
        let parsed = decoder
            .parse(&MAIN_REFERENCE_PAYLOAD, main_reference_config())
            .unwrap();
        let core = parsed.core();

        assert_eq!(core.transform_type(), TransformType::Short);
        assert_eq!(core.lsf().mode(), LsfCodebookMode::HighBitrate);
        assert_eq!(core.lsf().indexes(), [17, 201, 66, 99, 45, 17, 3]);

        let tns = core.tns();
        let filters = tns.filters();
        assert!(filters[0].enabled());
        assert_eq!(filters[0].order(), 3);
        assert_eq!(
            filters[0]
                .coefficients()
                .iter()
                .map(|value| (value.index(), value.code(), value.bits()))
                .collect::<Vec<_>>(),
            [(-3, 0, 3), (6, 481, 10), (-8, 27_136, 15)]
        );
        assert!(filters[1].enabled());
        assert_eq!(filters[1].order(), 8);
        assert_eq!(
            filters[1]
                .coefficients()
                .iter()
                .map(|value| (value.index(), value.code(), value.bits()))
                .collect::<Vec<_>>(),
            [
                (0, 2, 2),
                (1, 3, 2),
                (2, 27, 5),
                (3, 16, 5),
                (4, 129, 9),
                (5, 1_035, 11),
                (6, 13_314, 14),
                (7, 10_499, 14),
            ]
        );

        let bwe = core.bwe().unwrap();
        assert_eq!(bwe.envelope_indexes(), [1, 127, 55, 64]);
        assert_eq!(
            bwe.whitening_levels(),
            [BweWhiteningLevel::Off, BweWhiteningLevel::High]
        );
        assert_eq!(core.grouping().count(), 2);
        assert_eq!(
            core.grouping().indicator(),
            [
                NoiseGroup::Transient,
                NoiseGroup::Transient,
                NoiseGroup::Transient,
                NoiseGroup::Other,
                NoiseGroup::Other,
                NoiseGroup::Other,
                NoiseGroup::Other,
                NoiseGroup::Other,
            ]
        );

        let ParsedNeuralQc::Main(qc) = parsed.neural_qc() else {
            panic!("C reference vector is a main-profile payload");
        };
        assert!(qc.feature_amplified());
        assert_eq!(qc.scale_index(), 37);
        assert_eq!(qc.noise_filling().quantized_indexes(), [3, 7]);
        assert_eq!(qc.bitstreams().context(), [0x84, 0xa0, 0xd8]);
        assert_eq!(qc.bitstreams().base(), [0x7f, 0xfd, 0x51, 0xf6, 0xf2]);
        assert_eq!(parsed.entropy_bytes(), 8);
        assert_eq!(parsed.consumed_bits(), 272);
        assert_eq!(parsed.side_information_bits(), 208);
        assert_eq!(parsed.padding_bits(), 5);
    }

    #[test]
    fn lc_side_information_matches_c_reference_vector() {
        let mut decoder = MonoSideInfoDecoder::new();
        let parsed = decoder
            .parse(&LC_REFERENCE_PAYLOAD, lc_reference_config())
            .unwrap();
        let core = parsed.core();

        assert_eq!(core.transform_type(), TransformType::LongToShort);
        assert_eq!(core.lsf().mode(), LsfCodebookMode::LowBitrate);
        assert_eq!(core.lsf().indexes(), [3, 250, 65, 12, 51]);
        assert_eq!(core.bwe(), None);
        assert_eq!(core.grouping().count(), 1);
        assert_eq!(
            core.grouping().indicator(),
            [NoiseGroup::Transient; AVS3_SHORT_BLOCKS]
        );

        let tns = core.tns();
        let filters = tns.filters();
        assert!(!filters[0].enabled());
        assert_eq!(filters[0].order(), 0);
        assert!(filters[1].enabled());
        assert_eq!(filters[1].order(), 1);
        let coefficient = filters[1].coefficients()[0];
        assert_eq!(
            (coefficient.index(), coefficient.code(), coefficient.bits()),
            (7, 8_105, 13)
        );

        let ParsedNeuralQc::LowComplexity(qc) = parsed.neural_qc() else {
            panic!("C reference vector is a low-complexity payload");
        };
        assert_eq!(qc.scale_index(), 91);
        assert_eq!(qc.noise_filling().quantized_indexes(), [5, 0]);
        assert_eq!(qc.bitstreams().context(), [0x12, 0x34]);
        assert_eq!(qc.bitstreams().base(), [0x56, 0x78, 0x9a, 0xbc]);
        assert_eq!(parsed.entropy_bytes(), 6);
        assert_eq!(parsed.consumed_bits(), 123);
        assert_eq!(parsed.side_information_bits(), 75);
        assert_eq!(parsed.padding_bits(), 3);
    }

    #[test]
    fn rejects_every_truncated_reference_byte_prefix() {
        for byte_len in 0..MAIN_REFERENCE_PAYLOAD.len() {
            let error = MonoSideInfoDecoder::new()
                .parse(&MAIN_REFERENCE_PAYLOAD[..byte_len], main_reference_config())
                .unwrap_err();
            assert_eq!(
                error,
                CoreBitstreamError::PayloadTooShort {
                    declared_bits: MAIN_REFERENCE_PAYLOAD_BITS,
                    available_bits: byte_len * 8,
                },
                "byte prefix {byte_len}"
            );
        }
    }

    #[test]
    fn valid_bit_limit_cannot_read_storage_padding_as_side_information() {
        // The C vector needs 208 syntax bits before any entropy byte can be
        // accepted. Every shorter declared bit length must fail even though
        // all 35 storage bytes remain accessible in the backing slice.
        for payload_bits in 0..208 {
            let config = CoreBitstreamConfig::new(
                NnType::Main,
                payload_bits,
                LsfCodebookMode::HighBitrate,
                BweConfig::for_mono_bitrate(64_000).unwrap(),
            )
            .unwrap();
            assert!(
                MonoSideInfoDecoder::new()
                    .parse(&MAIN_REFERENCE_PAYLOAD, config)
                    .is_err(),
                "unexpectedly accepted {payload_bits} valid bits"
            );
        }
    }

    #[test]
    fn truncated_long_tns_code_returns_eof() {
        let mut writer = BitWriter::new();
        writer.write_bits(0, 2).unwrap();
        for width in HBR_LSF_WIDTHS {
            writer.write_bits(0, width).unwrap();
        }
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(0, 3).unwrap();
        let longest = TNS_CODES[0][15];
        writer
            .write_bits(u64::from(longest.code >> 1), usize::from(longest.bits - 1))
            .unwrap();
        let payload_bits = writer.bit_len();
        let payload = writer.into_bytes();
        let config = CoreBitstreamConfig::new(
            NnType::Main,
            payload_bits,
            LsfCodebookMode::HighBitrate,
            None,
        )
        .unwrap();

        let error = MonoSideInfoDecoder::new()
            .parse(&payload, config)
            .unwrap_err();
        assert!(matches!(
            error,
            CoreBitstreamError::Bitstream(BitstreamError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn rejects_qc_budget_underflow() {
        let mut writer = BitWriter::new();
        writer.write_bits(0, 2).unwrap();
        for width in LBR_LSF_WIDTHS {
            writer.write_bits(0, width).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        let payload_bits = writer.bit_len();
        let payload = writer.into_bytes();
        let config = CoreBitstreamConfig::new(
            NnType::LowComplexity,
            payload_bits,
            LsfCodebookMode::LowBitrate,
            None,
        )
        .unwrap();

        assert_eq!(
            MonoSideInfoDecoder::new()
                .parse(&payload, config)
                .unwrap_err(),
            CoreBitstreamError::QcBudgetUnderflow {
                payload_bits,
                used_bits: payload_bits,
                reserved_bits: 19,
            }
        );
    }

    #[test]
    fn rejects_entropy_payload_over_fixed_workspace_limit() {
        let mut writer = BitWriter::new();
        writer.write_bits(0, 2).unwrap();
        for width in LBR_LSF_WIDTHS {
            writer.write_bits(0, width).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        let core_bits = writer.bit_len();
        let payload_bits = core_bits + 19 + (MAX_QC_BITSTREAM_BYTES + 1) * 8;
        let mut payload = writer.into_bytes();
        payload.resize(payload_bits.div_ceil(8), 0);
        let config = CoreBitstreamConfig::new(
            NnType::LowComplexity,
            payload_bits,
            LsfCodebookMode::LowBitrate,
            None,
        )
        .unwrap();

        assert_eq!(
            MonoSideInfoDecoder::new()
                .parse(&payload, config)
                .unwrap_err(),
            CoreBitstreamError::EntropyPayloadTooLarge {
                bytes: MAX_QC_BITSTREAM_BYTES + 1,
                limit: MAX_QC_BITSTREAM_BYTES,
            }
        );
    }

    #[test]
    fn rejects_declared_context_larger_than_entropy_budget() {
        let mut writer = BitWriter::new();
        writer.write_bits(0, 2).unwrap();
        for width in LBR_LSF_WIDTHS {
            writer.write_bits(0, width).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 8).unwrap();
        writer.write_bits(0, 3).unwrap();
        writer.write_bits(2, 8).unwrap();
        writer.write_bits(0, 8).unwrap();
        let payload = writer.into_bytes();
        let config = CoreBitstreamConfig::new(
            NnType::LowComplexity,
            payload.len() * 8,
            LsfCodebookMode::LowBitrate,
            None,
        )
        .unwrap();
        let error = MonoSideInfoDecoder::new()
            .parse(&payload, config)
            .unwrap_err();
        assert!(matches!(
            error,
            CoreBitstreamError::ContextLengthExceedsEntropyPayload { .. }
        ));
    }
}
