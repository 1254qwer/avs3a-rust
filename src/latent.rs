use core::fmt;

/// Generous hard limits around AVS3 neural latent tensors.  The reference
/// model uses 64 dimensions and 16 channels for the base latent, but limits
/// remain larger so compatible model revisions do not require an API change.
pub const MAX_LATENT_DIMENSIONS: usize = 1 << 20;
pub const MAX_LATENT_CHANNELS: usize = 4_096;
pub const MAX_LATENT_VALUES: usize = 1 << 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatentError {
    ZeroDimensions,
    ZeroChannels,
    DimensionsTooLarge {
        dimensions: usize,
        limit: usize,
    },
    ChannelsTooLarge {
        channels: usize,
        limit: usize,
    },
    ValueCountOverflow,
    TooManyValues {
        values: usize,
        limit: usize,
    },
    LengthMismatch {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    QuantizerChannelMismatch {
        shape: usize,
        quantizer: usize,
    },
    NonFiniteMedian {
        channel: usize,
        bits: u32,
    },
    EmptyContextScales,
    TooManyContextScales {
        scales: usize,
        limit: usize,
    },
    InvalidContextScale {
        index: usize,
        bits: u32,
    },
    UnsortedContextScales {
        index: usize,
    },
    NonFiniteContextValue {
        index: usize,
        bits: u32,
    },
}

impl fmt::Display for LatentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions => f.write_str("latent tensor has zero dimensions"),
            Self::ZeroChannels => f.write_str("latent tensor has zero channels"),
            Self::DimensionsTooLarge { dimensions, limit } => {
                write!(f, "latent dimension {dimensions} exceeds limit {limit}")
            }
            Self::ChannelsTooLarge { channels, limit } => {
                write!(f, "latent channel count {channels} exceeds limit {limit}")
            }
            Self::ValueCountOverflow => f.write_str("latent tensor size overflows usize"),
            Self::TooManyValues { values, limit } => {
                write!(f, "latent tensor has {values} values; limit is {limit}")
            }
            Self::LengthMismatch {
                buffer,
                expected,
                actual,
            } => write!(
                f,
                "{buffer} has {actual} values; latent shape requires {expected}"
            ),
            Self::QuantizerChannelMismatch { shape, quantizer } => write!(
                f,
                "latent shape has {shape} channels but quantizer has {quantizer}"
            ),
            Self::NonFiniteMedian { channel, bits } => write!(
                f,
                "quantizer median for channel {channel} is not finite (bits 0x{bits:08x})"
            ),
            Self::EmptyContextScales => f.write_str("context-scale table is empty"),
            Self::TooManyContextScales { scales, limit } => {
                write!(
                    f,
                    "context-scale table has {scales} entries; limit is {limit}"
                )
            }
            Self::InvalidContextScale { index, bits } => write!(
                f,
                "context scale {index} is not finite and positive (bits 0x{bits:08x})"
            ),
            Self::UnsortedContextScales { index } => write!(
                f,
                "context scales are decreasing between entries {} and {index}",
                index - 1
            ),
            Self::NonFiniteContextValue { index, bits } => write!(
                f,
                "context output {index} is not finite (bits 0x{bits:08x})"
            ),
        }
    }
}

impl std::error::Error for LatentError {}

/// Checked two-dimensional latent shape.
///
/// Neural feature buffers use the C reference implementation's channel-major
/// layout: value `(dimension, channel)` is stored at
/// `dimension + channel * dimensions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentShape {
    dimensions: usize,
    channels: usize,
    values: usize,
}

impl LatentShape {
    pub fn new(dimensions: usize, channels: usize) -> Result<Self, LatentError> {
        if dimensions == 0 {
            return Err(LatentError::ZeroDimensions);
        }
        if channels == 0 {
            return Err(LatentError::ZeroChannels);
        }
        if dimensions > MAX_LATENT_DIMENSIONS {
            return Err(LatentError::DimensionsTooLarge {
                dimensions,
                limit: MAX_LATENT_DIMENSIONS,
            });
        }
        if channels > MAX_LATENT_CHANNELS {
            return Err(LatentError::ChannelsTooLarge {
                channels,
                limit: MAX_LATENT_CHANNELS,
            });
        }
        let values = dimensions
            .checked_mul(channels)
            .ok_or(LatentError::ValueCountOverflow)?;
        if values > MAX_LATENT_VALUES {
            return Err(LatentError::TooManyValues {
                values,
                limit: MAX_LATENT_VALUES,
            });
        }
        Ok(Self {
            dimensions,
            channels,
            values,
        })
    }

    pub fn dimensions(self) -> usize {
        self.dimensions
    }

    pub fn channels(self) -> usize {
        self.channels
    }

    pub fn len(self) -> usize {
        self.values
    }

    pub fn is_empty(self) -> bool {
        false
    }

    fn check_len<T>(self, buffer: &'static str, values: &[T]) -> Result<(), LatentError> {
        if values.len() == self.values {
            Ok(())
        } else {
            Err(LatentError::LengthMismatch {
                buffer,
                expected: self.values,
                actual: values.len(),
            })
        }
    }
}

/// Convert a channel-major latent tensor into the dimension-interleaved order
/// consumed by `RangeDecodeProcess`/`RangeEncodeProcess` in the C codec.
pub fn flatten_for_entropy_coder<T: Copy>(
    shape: LatentShape,
    channel_major: &[T],
) -> Result<Vec<T>, LatentError> {
    shape.check_len("channel-major latent", channel_major)?;
    let mut flattened = channel_major.to_vec();
    flatten_for_entropy_coder_into(shape, channel_major, &mut flattened)?;
    Ok(flattened)
}

pub fn flatten_for_entropy_coder_into<T: Copy>(
    shape: LatentShape,
    channel_major: &[T],
    flattened: &mut [T],
) -> Result<(), LatentError> {
    shape.check_len("channel-major latent", channel_major)?;
    shape.check_len("flattened latent", flattened)?;
    for dimension in 0..shape.dimensions {
        for channel in 0..shape.channels {
            flattened[dimension * shape.channels + channel] =
                channel_major[dimension + channel * shape.dimensions];
        }
    }
    Ok(())
}

/// Convert entropy-coder order back to the channel-major layout used by the
/// neural layers and quantizer.
pub fn unflatten_from_entropy_coder<T: Copy>(
    shape: LatentShape,
    flattened: &[T],
) -> Result<Vec<T>, LatentError> {
    shape.check_len("flattened latent", flattened)?;
    let mut channel_major = flattened.to_vec();
    unflatten_from_entropy_coder_into(shape, flattened, &mut channel_major)?;
    Ok(channel_major)
}

pub fn unflatten_from_entropy_coder_into<T: Copy>(
    shape: LatentShape,
    flattened: &[T],
    channel_major: &mut [T],
) -> Result<(), LatentError> {
    shape.check_len("flattened latent", flattened)?;
    shape.check_len("channel-major latent", channel_major)?;
    for dimension in 0..shape.dimensions {
        for channel in 0..shape.channels {
            channel_major[dimension + channel * shape.dimensions] =
                flattened[dimension * shape.channels + channel];
        }
    }
    Ok(())
}

/// CDF table indexes for a VAE/context latent: channel 0, channel 1, ... for
/// every dimension.
pub fn channel_cdf_indexes(shape: LatentShape) -> Vec<u16> {
    let mut indexes = vec![0_u16; shape.values];
    // The freshly sized destination cannot fail validation.
    channel_cdf_indexes_into(shape, &mut indexes).expect("shape-sized CDF index buffer");
    indexes
}

pub fn channel_cdf_indexes_into(
    shape: LatentShape,
    indexes: &mut [u16],
) -> Result<(), LatentError> {
    shape.check_len("channel CDF indexes", indexes)?;
    let mut position = 0;
    for _ in 0..shape.dimensions {
        for channel in 0..shape.channels {
            // LatentShape caps channels well below u16::MAX.
            indexes[position] = channel as u16;
            position += 1;
        }
    }
    Ok(())
}

/// Per-channel scalar quantizer used by the neural entropy decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantizer {
    quantile_medians: Vec<f32>,
}

impl Quantizer {
    pub fn new(quantile_medians: Vec<f32>) -> Result<Self, LatentError> {
        if quantile_medians.is_empty() {
            return Err(LatentError::ZeroChannels);
        }
        if quantile_medians.len() > MAX_LATENT_CHANNELS {
            return Err(LatentError::ChannelsTooLarge {
                channels: quantile_medians.len(),
                limit: MAX_LATENT_CHANNELS,
            });
        }
        for (channel, &median) in quantile_medians.iter().enumerate() {
            if !median.is_finite() {
                return Err(LatentError::NonFiniteMedian {
                    channel,
                    bits: median.to_bits(),
                });
            }
        }
        Ok(Self { quantile_medians })
    }

    pub fn channels(&self) -> usize {
        self.quantile_medians.len()
    }

    pub fn quantile_medians(&self) -> &[f32] {
        &self.quantile_medians
    }

    pub fn dequantize(
        &self,
        shape: LatentShape,
        quantized: &[i32],
    ) -> Result<Vec<f32>, LatentError> {
        let mut output = vec![0.0_f32; shape.len()];
        self.dequantize_into(shape, quantized, &mut output)?;
        Ok(output)
    }

    pub fn dequantize_into(
        &self,
        shape: LatentShape,
        quantized: &[i32],
        output: &mut [f32],
    ) -> Result<(), LatentError> {
        if shape.channels != self.channels() {
            return Err(LatentError::QuantizerChannelMismatch {
                shape: shape.channels,
                quantizer: self.channels(),
            });
        }
        shape.check_len("quantized latent", quantized)?;
        shape.check_len("dequantized latent", output)?;

        // Preserve the C loop order as well as its explicit i32 -> f32 cast.
        for dimension in 0..shape.dimensions {
            for channel in 0..shape.channels {
                let index = dimension + channel * shape.dimensions;
                output[index] = quantized[index] as f32 + self.quantile_medians[channel];
            }
        }
        Ok(())
    }
}

/// Sorted positive scale table used to select a base-model CDF from the
/// hyper-prior decoder output.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextScaleTable {
    scales: Vec<f32>,
}

impl ContextScaleTable {
    pub fn new(scales: Vec<f32>) -> Result<Self, LatentError> {
        if scales.is_empty() {
            return Err(LatentError::EmptyContextScales);
        }
        if scales.len() > u16::MAX as usize {
            return Err(LatentError::TooManyContextScales {
                scales: scales.len(),
                limit: u16::MAX as usize,
            });
        }
        for (index, &scale) in scales.iter().enumerate() {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(LatentError::InvalidContextScale {
                    index,
                    bits: scale.to_bits(),
                });
            }
            if index != 0 && scales[index - 1] > scale {
                return Err(LatentError::UnsortedContextScales { index });
            }
        }
        Ok(Self { scales })
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Select the first scale greater than or equal to each context value,
    /// falling back to the last scale. The output is already in entropy-coder
    /// order, matching `MdctDequantDecodeHyper`.
    pub fn cdf_indexes(
        &self,
        shape: LatentShape,
        context_output: &[f32],
    ) -> Result<Vec<u16>, LatentError> {
        shape.check_len("context output", context_output)?;
        let mut indexes = vec![0_u16; shape.values];
        self.cdf_indexes_into(shape, context_output, &mut indexes)?;
        Ok(indexes)
    }

    pub fn cdf_indexes_into(
        &self,
        shape: LatentShape,
        context_output: &[f32],
        indexes: &mut [u16],
    ) -> Result<(), LatentError> {
        shape.check_len("context output", context_output)?;
        shape.check_len("context CDF indexes", indexes)?;
        let mut output_position = 0;
        for dimension in 0..shape.dimensions {
            for channel in 0..shape.channels {
                let input_index = dimension + channel * shape.dimensions;
                let value = context_output[input_index];
                if !value.is_finite() {
                    return Err(LatentError::NonFiniteContextValue {
                        index: input_index,
                        bits: value.to_bits(),
                    });
                }
                let selected = self.scales.partition_point(|&scale| scale < value);
                let selected = selected.min(self.scales.len() - 1);
                indexes[output_position] = selected as u16;
                output_position += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_conversion_matches_mdct_decoder_loops() {
        let shape = LatentShape::new(3, 4).unwrap();
        let channel_major = [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32];
        let flattened = flatten_for_entropy_coder(shape, &channel_major).unwrap();
        assert_eq!(flattened, [0, 10, 20, 30, 1, 11, 21, 31, 2, 12, 22, 32]);
        assert_eq!(
            unflatten_from_entropy_coder(shape, &flattened).unwrap(),
            channel_major
        );
        assert_eq!(
            channel_cdf_indexes(shape),
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        );
    }

    #[test]
    fn dequantization_matches_c_reference_bits() {
        // Generated by latent_quant.c at both -O0 and -O3. The flattened
        // values are first converted with the exact loops in
        // mdct_dequant_decode.c.
        let shape = LatentShape::new(5, 4).unwrap();
        let flattened = [
            -1,
            0,
            1,
            2,
            i32::MIN,
            i32::MAX,
            -16_777_217,
            16_777_217,
            -9,
            17,
            -2,
            5,
            123_456_789,
            -123_456_789,
            3,
            -3,
            0,
            -1,
            42,
            -42,
        ];
        let channel_major = unflatten_from_entropy_coder(shape, &flattened).unwrap();
        assert_eq!(
            channel_major,
            [
                -1,
                i32::MIN,
                -9,
                123_456_789,
                0,
                0,
                i32::MAX,
                17,
                -123_456_789,
                -1,
                1,
                -16_777_217,
                -2,
                3,
                42,
                2,
                16_777_217,
                5,
                -3,
                -42,
            ]
        );
        let quantizer = Quantizer::new(vec![
            f32::from_bits(0xbeff_3fe2),
            f32::from_bits(0x3e3c_9387),
            f32::from_bits(0x3d04_0240),
            f32::from_bits(0xbb6c_6969),
        ])
        .unwrap();
        let actual_bits: Vec<u32> = quantizer
            .dequantize(shape, &channel_major)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(
            actual_bits,
            [
                0xbfbf_cff8,
                0xcf00_0000,
                0xc117_f9ff,
                0x4ceb_79a3,
                0xbeff_3fe2,
                0x3e3c_9387,
                0x4f00_0000,
                0x4189_7927,
                0xcceb_79a3,
                0xbf50_db1e,
                0x3f84_2012,
                0xcb80_0000,
                0xbffb_dfee,
                0x4042_1009,
                0x4228_2101,
                0x3fff_89cb,
                0x4b80_0000,
                0x409f_e273,
                0xc040_3b1a,
                0xc228_03b2,
            ]
        );
    }

    #[test]
    fn context_scale_selection_matches_first_greater_or_equal_rule() {
        let shape = LatentShape::new(2, 3).unwrap();
        let table = ContextScaleTable::new(vec![0.25, 0.5, 1.0, 2.0]).unwrap();
        // Channel-major: channel 0 [negative, exact], channel 1 [between,
        // above], channel 2 [first, just above first].
        let output = [-1.0, 0.5, 0.75, 99.0, 0.25, 0.250_001];
        assert_eq!(
            table.cdf_indexes(shape, &output).unwrap(),
            [0, 2, 0, 1, 3, 1]
        );
    }

    #[test]
    fn rejects_bad_shapes_lengths_and_non_finite_data() {
        assert_eq!(LatentShape::new(0, 1), Err(LatentError::ZeroDimensions));
        assert_eq!(LatentShape::new(1, 0), Err(LatentError::ZeroChannels));

        let shape = LatentShape::new(2, 2).unwrap();
        assert!(matches!(
            flatten_for_entropy_coder(shape, &[1, 2, 3]),
            Err(LatentError::LengthMismatch { .. })
        ));
        assert!(matches!(
            Quantizer::new(vec![f32::NAN]),
            Err(LatentError::NonFiniteMedian { .. })
        ));
        assert!(matches!(
            ContextScaleTable::new(vec![1.0, 0.5]),
            Err(LatentError::UnsortedContextScales { .. })
        ));
        let table = ContextScaleTable::new(vec![0.5, 1.0]).unwrap();
        assert!(matches!(
            table.cdf_indexes(shape, &[0.0, f32::INFINITY, 0.0, 0.0]),
            Err(LatentError::NonFiniteContextValue { .. })
        ));
    }
}
