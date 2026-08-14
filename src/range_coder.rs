use core::fmt;

const DEFAULT_PRECISION: u8 = 16;
const DEFAULT_OVERFLOW_WIDTH: u8 = 4;
const MAX_MODEL_CDFS: usize = u16::MAX as usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RangeCoderError {
    ConfigLengthMismatch {
        cdfs: usize,
        offsets: usize,
    },
    InvalidPrecision(u8),
    InvalidOverflowWidth(u8),
    InvalidCdfLength {
        index: usize,
        length: usize,
    },
    InvalidCdfStart {
        index: usize,
        value: u32,
    },
    InvalidCdfEnd {
        index: usize,
        expected: u32,
        actual: u32,
    },
    NonIncreasingCdf {
        index: usize,
        position: usize,
    },
    CdfIndexOutOfRange {
        position: usize,
        index: usize,
        available: usize,
    },
    OutputLengthMismatch {
        indexes: usize,
        output: usize,
    },
    InvalidInterval,
    OverflowCodeTooLong(u32),
    TooManyCdfs(usize),
    ModelDataTooShort {
        needed: usize,
        available: usize,
    },
    IntegerOverflow,
}

impl fmt::Display for RangeCoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfigLengthMismatch { cdfs, offsets } => {
                write!(
                    f,
                    "range-coder config has {cdfs} CDFs but {offsets} offsets"
                )
            }
            Self::InvalidPrecision(value) => write!(f, "invalid CDF precision {value}"),
            Self::InvalidOverflowWidth(value) => write!(f, "invalid overflow width {value}"),
            Self::InvalidCdfLength { index, length } => {
                write!(f, "CDF {index} has invalid length {length}")
            }
            Self::InvalidCdfStart { index, value } => {
                write!(f, "CDF {index} starts at {value}, expected 0")
            }
            Self::InvalidCdfEnd {
                index,
                expected,
                actual,
            } => {
                write!(f, "CDF {index} ends at {actual}, expected {expected}")
            }
            Self::NonIncreasingCdf { index, position } => {
                write!(
                    f,
                    "CDF {index} is not strictly increasing at element {position}"
                )
            }
            Self::CdfIndexOutOfRange {
                position,
                index,
                available,
            } => write!(
                f,
                "CDF index {index} at output position {position} exceeds {available} distributions"
            ),
            Self::OutputLengthMismatch { indexes, output } => write!(
                f,
                "range decoder has {indexes} CDF indexes but {output} output slots"
            ),
            Self::InvalidInterval => f.write_str("range decoder reached an invalid interval"),
            Self::OverflowCodeTooLong(widths) => {
                write!(f, "range overflow code requests {widths} sections")
            }
            Self::TooManyCdfs(value) => write!(f, "model requests {value} CDF tables"),
            Self::ModelDataTooShort { needed, available } => {
                write!(
                    f,
                    "range-coder model needs {needed} bytes, only {available} available"
                )
            }
            Self::IntegerOverflow => f.write_str("range decoder integer overflow"),
        }
    }
}

impl std::error::Error for RangeCoderError {}

/// Owned and validated cumulative distribution tables for the AVS3 neural
/// entropy coder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeCoderConfig {
    cdfs: Vec<Vec<u32>>,
    offsets: Vec<i16>,
    precision: u8,
    overflow_width: u8,
}

impl RangeCoderConfig {
    pub fn new(cdfs: Vec<Vec<u32>>, offsets: Vec<i16>) -> Result<Self, RangeCoderError> {
        Self::with_parameters(cdfs, offsets, DEFAULT_PRECISION, DEFAULT_OVERFLOW_WIDTH)
    }

    pub fn with_parameters(
        cdfs: Vec<Vec<u32>>,
        offsets: Vec<i16>,
        precision: u8,
        overflow_width: u8,
    ) -> Result<Self, RangeCoderError> {
        if cdfs.len() != offsets.len() {
            return Err(RangeCoderError::ConfigLengthMismatch {
                cdfs: cdfs.len(),
                offsets: offsets.len(),
            });
        }
        if !(1..=16).contains(&precision) {
            return Err(RangeCoderError::InvalidPrecision(precision));
        }
        if !(1..=16).contains(&overflow_width) {
            return Err(RangeCoderError::InvalidOverflowWidth(overflow_width));
        }
        let expected_end = 1_u32 << precision;
        for (index, cdf) in cdfs.iter().enumerate() {
            if cdf.len() < 3 {
                return Err(RangeCoderError::InvalidCdfLength {
                    index,
                    length: cdf.len(),
                });
            }
            if cdf[0] != 0 {
                return Err(RangeCoderError::InvalidCdfStart {
                    index,
                    value: cdf[0],
                });
            }
            if cdf.last().copied() != Some(expected_end) {
                return Err(RangeCoderError::InvalidCdfEnd {
                    index,
                    expected: expected_end,
                    actual: cdf.last().copied().unwrap_or_default(),
                });
            }
            if let Some(position) = cdf.windows(2).position(|pair| pair[0] >= pair[1]) {
                return Err(RangeCoderError::NonIncreasingCdf {
                    index,
                    position: position + 1,
                });
            }
        }
        Ok(Self {
            cdfs,
            offsets,
            precision,
            overflow_width,
        })
    }

    /// Parse the little-endian range-coder block consumed by the C
    /// `InitRangeCoderConfig` routine.
    ///
    /// The caller supplies the number of CDFs, which is stored elsewhere in
    /// the neural model. The returned byte count allows parsing to continue at
    /// the next model block. Input must already have the model-wide XOR `0x55`
    /// obfuscation removed.
    pub fn from_model_bytes(
        input: &[u8],
        num_cdfs: usize,
    ) -> Result<(Self, usize), RangeCoderError> {
        if num_cdfs > MAX_MODEL_CDFS {
            return Err(RangeCoderError::TooManyCdfs(num_cdfs));
        }
        let header_bytes = num_cdfs
            .checked_mul(4)
            .ok_or(RangeCoderError::IntegerOverflow)?;
        require_model_bytes(input, header_bytes)?;

        let mut cursor = 0_usize;
        let mut lengths = Vec::with_capacity(num_cdfs);
        for _ in 0..num_cdfs {
            lengths.push(usize::from(read_u16(input, &mut cursor)?));
        }
        let mut offsets = Vec::with_capacity(num_cdfs);
        for _ in 0..num_cdfs {
            offsets.push(read_i16(input, &mut cursor)?);
        }
        let table_values = lengths.iter().try_fold(0_usize, |total, &length| {
            total
                .checked_add(length)
                .ok_or(RangeCoderError::IntegerOverflow)
        })?;
        let table_bytes = table_values
            .checked_mul(4)
            .ok_or(RangeCoderError::IntegerOverflow)?;
        let total_bytes = header_bytes
            .checked_add(table_bytes)
            .ok_or(RangeCoderError::IntegerOverflow)?;
        require_model_bytes(input, total_bytes)?;

        let mut cdfs = Vec::with_capacity(num_cdfs);
        for length in lengths {
            let mut cdf = Vec::with_capacity(length);
            for _ in 0..length {
                cdf.push(read_u32(input, &mut cursor)?);
            }
            cdfs.push(cdf);
        }
        debug_assert_eq!(cursor, total_bytes);
        Ok((Self::new(cdfs, offsets)?, cursor))
    }

    pub fn cdfs(&self) -> &[Vec<u32>] {
        &self.cdfs
    }

    pub fn offsets(&self) -> &[i16] {
        &self.offsets
    }

    pub fn precision(&self) -> u8 {
        self.precision
    }

    pub fn overflow_width(&self) -> u8 {
        self.overflow_width
    }
}

#[derive(Debug)]
pub struct RangeDecoder<'a> {
    input: &'a [u8],
    input_position: usize,
    base: u32,
    size_minus_one: u32,
    value: u32,
    initialized: bool,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            input_position: 0,
            base: 0,
            size_minus_one: u32::MAX,
            value: 0,
            initialized: false,
        }
    }

    /// Number of physically present input bytes consumed. Decoder tail zeros
    /// are implicit, matching the C encoder's omission of trailing zero bytes.
    pub fn bytes_consumed(&self) -> usize {
        self.input_position
    }

    pub fn decode(
        &mut self,
        config: &RangeCoderConfig,
        cdf_indexes: &[u16],
    ) -> Result<Vec<i32>, RangeCoderError> {
        let mut output = vec![0_i32; cdf_indexes.len()];
        self.decode_into(config, cdf_indexes, &mut output)?;
        Ok(output)
    }

    pub fn decode_into(
        &mut self,
        config: &RangeCoderConfig,
        cdf_indexes: &[u16],
        output: &mut [i32],
    ) -> Result<(), RangeCoderError> {
        if cdf_indexes.len() != output.len() {
            return Err(RangeCoderError::OutputLengthMismatch {
                indexes: cdf_indexes.len(),
                output: output.len(),
            });
        }
        let max_overflow = (1_u32 << config.overflow_width) - 1;
        let max_sections = 32_u32.div_ceil(u32::from(config.overflow_width));

        for (position, (&raw_index, output_value)) in
            cdf_indexes.iter().zip(output.iter_mut()).enumerate()
        {
            let index = usize::from(raw_index);
            let cdf = config
                .cdfs
                .get(index)
                .ok_or(RangeCoderError::CdfIndexOutOfRange {
                    position,
                    index,
                    available: config.cdfs.len(),
                })?;
            let max_value =
                i32::try_from(cdf.len() - 2).map_err(|_| RangeCoderError::IntegerOverflow)?;
            let mut value = i32::try_from(self.decode_symbol(cdf, config.precision)?)
                .map_err(|_| RangeCoderError::IntegerOverflow)?;

            if value == max_value {
                let mut widths = 0_u32;
                loop {
                    let section = self.decode_uniform_symbol(config.overflow_width)?;
                    widths = widths
                        .checked_add(section)
                        .ok_or(RangeCoderError::IntegerOverflow)?;
                    if widths > max_sections {
                        return Err(RangeCoderError::OverflowCodeTooLong(widths));
                    }
                    if section != max_overflow {
                        break;
                    }
                }

                let total_bits = widths
                    .checked_mul(u32::from(config.overflow_width))
                    .ok_or(RangeCoderError::IntegerOverflow)?;
                if total_bits > 32 {
                    return Err(RangeCoderError::OverflowCodeTooLong(widths));
                }
                let mut overflow = 0_u32;
                for section_index in 0..widths {
                    let section = self.decode_uniform_symbol(config.overflow_width)?;
                    let shift = section_index
                        .checked_mul(u32::from(config.overflow_width))
                        .ok_or(RangeCoderError::IntegerOverflow)?;
                    overflow |= section
                        .checked_shl(shift)
                        .ok_or(RangeCoderError::IntegerOverflow)?;
                }

                let magnitude =
                    i32::try_from(overflow >> 1).map_err(|_| RangeCoderError::IntegerOverflow)?;
                value = if overflow & 1 != 0 {
                    magnitude.checked_neg().and_then(|item| item.checked_sub(1))
                } else {
                    magnitude.checked_add(max_value)
                }
                .ok_or(RangeCoderError::IntegerOverflow)?;
            }

            value = value
                .checked_add(i32::from(config.offsets[index]))
                .ok_or(RangeCoderError::IntegerOverflow)?;
            *output_value = value;
        }
        Ok(())
    }

    fn decode_symbol(&mut self, cdf: &[u32], precision: u8) -> Result<u32, RangeCoderError> {
        self.initialize();

        let size = u64::from(self.size_minus_one) + 1;
        let relative = self.value.wrapping_sub(self.base);
        let offset = ((u64::from(relative) + 1) << precision) - 1;
        let pv = cdf.partition_point(|&value| size * u64::from(value) <= offset);
        if pv == 0 || pv >= cdf.len() {
            return Err(RangeCoderError::InvalidInterval);
        }

        let a = (size * u64::from(cdf[pv - 1])) >> precision;
        let upper_scaled = (size * u64::from(cdf[pv])) >> precision;
        let b = upper_scaled
            .checked_sub(1)
            .ok_or(RangeCoderError::InvalidInterval)?;
        if a > b || (offset >> precision) < a || (offset >> precision) > b {
            return Err(RangeCoderError::InvalidInterval);
        }
        let a = u32::try_from(a).map_err(|_| RangeCoderError::IntegerOverflow)?;
        let b = u32::try_from(b).map_err(|_| RangeCoderError::IntegerOverflow)?;
        self.base = self.base.wrapping_add(a);
        self.size_minus_one = b - a;

        if self.size_minus_one >> 16 == 0 {
            self.base <<= 16;
            self.size_minus_one = (self.size_minus_one << 16) | 0xffff;
            self.read_u16();
        }
        u32::try_from(pv - 1).map_err(|_| RangeCoderError::IntegerOverflow)
    }

    fn decode_uniform_symbol(&mut self, precision: u8) -> Result<u32, RangeCoderError> {
        self.initialize();
        let size = u64::from(self.size_minus_one) + 1;
        let relative = self.value.wrapping_sub(self.base);
        let offset = ((u64::from(relative) + 1) << precision) - 1;
        let symbol = offset / size;
        let symbol_limit = 1_u64 << precision;
        if symbol >= symbol_limit {
            return Err(RangeCoderError::InvalidInterval);
        }
        let lower = (size * symbol) >> precision;
        let upper = ((size * (symbol + 1)) >> precision)
            .checked_sub(1)
            .ok_or(RangeCoderError::InvalidInterval)?;
        self.update_interval(lower, upper)?;
        u32::try_from(symbol).map_err(|_| RangeCoderError::IntegerOverflow)
    }

    fn initialize(&mut self) {
        if !self.initialized {
            self.read_u16();
            self.read_u16();
            self.initialized = true;
        }
    }

    fn update_interval(&mut self, lower: u64, upper: u64) -> Result<(), RangeCoderError> {
        if lower > upper {
            return Err(RangeCoderError::InvalidInterval);
        }
        let lower = u32::try_from(lower).map_err(|_| RangeCoderError::IntegerOverflow)?;
        let upper = u32::try_from(upper).map_err(|_| RangeCoderError::IntegerOverflow)?;
        self.base = self.base.wrapping_add(lower);
        self.size_minus_one = upper - lower;

        if self.size_minus_one >> 16 == 0 {
            self.base <<= 16;
            self.size_minus_one = (self.size_minus_one << 16) | 0xffff;
            self.read_u16();
        }
        Ok(())
    }

    fn read_u16(&mut self) {
        for _ in 0..2 {
            self.value <<= 8;
            if let Some(&byte) = self.input.get(self.input_position) {
                self.value |= u32::from(byte);
                self.input_position += 1;
            }
        }
    }
}

fn require_model_bytes(input: &[u8], needed: usize) -> Result<(), RangeCoderError> {
    if input.len() < needed {
        Err(RangeCoderError::ModelDataTooShort {
            needed,
            available: input.len(),
        })
    } else {
        Ok(())
    }
}

fn read_u16(input: &[u8], cursor: &mut usize) -> Result<u16, RangeCoderError> {
    let end = cursor
        .checked_add(2)
        .ok_or(RangeCoderError::IntegerOverflow)?;
    require_model_bytes(input, end)?;
    let value = u16::from_le_bytes(input[*cursor..end].try_into().map_err(|_| {
        RangeCoderError::ModelDataTooShort {
            needed: end,
            available: input.len(),
        }
    })?);
    *cursor = end;
    Ok(value)
}

fn read_i16(input: &[u8], cursor: &mut usize) -> Result<i16, RangeCoderError> {
    read_u16(input, cursor).map(|value| i16::from_le_bytes(value.to_le_bytes()))
}

fn read_u32(input: &[u8], cursor: &mut usize) -> Result<u32, RangeCoderError> {
    let end = cursor
        .checked_add(4)
        .ok_or(RangeCoderError::IntegerOverflow)?;
    require_model_bytes(input, end)?;
    let value = u32::from_le_bytes(input[*cursor..end].try_into().map_err(|_| {
        RangeCoderError::ModelDataTooShort {
            needed: end,
            available: input.len(),
        }
    })?);
    *cursor = end;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_config() -> RangeCoderConfig {
        RangeCoderConfig::new(
            vec![
                vec![0, 8_192, 24_576, 49_152, 65_536],
                vec![0, 4_096, 16_384, 32_768, 57_344, 65_536],
            ],
            vec![-1, -2],
        )
        .unwrap()
    }

    #[test]
    fn decodes_c_reference_vector_including_overflow() {
        // Produced by RangeEncodeProcess in the adjacent C reference at -O1.
        let encoded = [
            0x09, 0x49, 0x35, 0x51, 0x15, 0xf6, 0x28, 0xc1, 0xc7, 0x9f, 0x7e, 0xf3, 0x3e, 0xa4,
        ];
        let indexes = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        let expected = [-1, 0, 1, 2, -2, 5, -9, 17, -2, -1, 0, 1, 2, 3, 4, 5];
        let mut decoder = RangeDecoder::new(&encoded);
        assert_eq!(
            decoder.decode(&reference_config(), &indexes).unwrap(),
            expected
        );
        assert_eq!(decoder.bytes_consumed(), encoded.len());
    }

    #[test]
    fn supports_implicit_tail_zeros() {
        let config = RangeCoderConfig::new(vec![vec![0, 32_768, 65_536]], vec![0]).unwrap();
        let mut decoder = RangeDecoder::new(&[]);
        assert_eq!(decoder.decode(&config, &[0, 0]).unwrap(), [0, 0]);
        assert_eq!(decoder.bytes_consumed(), 0);
    }

    #[test]
    fn rejects_malformed_cdf_before_decoding() {
        assert!(matches!(
            RangeCoderConfig::new(vec![vec![0, 100, 99, 65_536]], vec![0]),
            Err(RangeCoderError::NonIncreasingCdf { .. })
        ));
        assert!(matches!(
            RangeCoderConfig::new(vec![vec![1, 32_768, 65_536]], vec![0]),
            Err(RangeCoderError::InvalidCdfStart { .. })
        ));
    }

    #[test]
    fn rejects_invalid_cdf_index() {
        let mut decoder = RangeDecoder::new(&[]);
        assert!(matches!(
            decoder.decode(&reference_config(), &[2]),
            Err(RangeCoderError::CdfIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn parses_owned_little_endian_model_block() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3_u16.to_le_bytes());
        bytes.extend_from_slice(&4_u16.to_le_bytes());
        bytes.extend_from_slice(&(-1_i16).to_le_bytes());
        bytes.extend_from_slice(&2_i16.to_le_bytes());
        for value in [0_u32, 32_768, 65_536, 0, 8_192, 32_768, 65_536] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(b"next block");

        let (config, consumed) = RangeCoderConfig::from_model_bytes(&bytes, 2).unwrap();
        assert_eq!(consumed, 36);
        assert_eq!(config.offsets(), &[-1, 2]);
        assert_eq!(config.cdfs()[1], [0, 8_192, 32_768, 65_536]);
    }

    #[test]
    fn rejects_truncated_model_before_table_allocation() {
        let bytes = [3, 0, 0, 0];
        assert!(matches!(
            RangeCoderConfig::from_model_bytes(&bytes, 1),
            Err(RangeCoderError::ModelDataTooShort { .. })
        ));
    }
}
