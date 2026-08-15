use core::fmt;

use wide::f32x8;

use crate::LatentShape;
use crate::model::{Activation, CnnLayer, CnnNetwork, Padding};

pub const MAX_CNN_WORKSPACE_VALUES: usize = 1 << 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CnnError {
    ForwardNetwork,
    InputLength {
        expected: usize,
        actual: usize,
    },
    OutputLength {
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        index: usize,
        bits: u32,
    },
    UnsupportedPadding {
        layer: usize,
        padding: Padding,
    },
    UnsupportedStride {
        layer: usize,
        stride: usize,
    },
    UnsupportedKernel {
        layer: usize,
        stride: usize,
        kernel_size: usize,
    },
    UnsupportedActivation {
        layer: usize,
        activation: Activation,
    },
    MissingGdnParameters {
        layer: usize,
    },
    InvalidGdnParameters {
        layer: usize,
        expected_beta: usize,
        actual_beta: usize,
        expected_gamma: usize,
        actual_gamma: usize,
    },
    InvalidNormalization {
        layer: usize,
        index: usize,
        bits: u32,
    },
    NonFiniteOutput {
        layer: usize,
        index: usize,
        bits: u32,
    },
    WorkspaceTooLarge {
        values: usize,
        limit: usize,
    },
}

impl fmt::Display for CnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForwardNetwork => {
                f.write_str("scalar CNN decoder requires a transpose-convolution network")
            }
            Self::InputLength { expected, actual } => {
                write!(f, "CNN input has {actual} values; expected {expected}")
            }
            Self::OutputLength { expected, actual } => {
                write!(f, "CNN output has {actual} values; expected {expected}")
            }
            Self::NonFiniteInput { index, bits } => {
                write!(f, "CNN input {index} is not finite (bits 0x{bits:08x})")
            }
            Self::UnsupportedPadding { layer, padding } => {
                write!(f, "CNN layer {layer} uses unsupported {padding:?} padding")
            }
            Self::UnsupportedStride { layer, stride } => {
                write!(f, "CNN layer {layer} uses unsupported stride {stride}")
            }
            Self::UnsupportedKernel {
                layer,
                stride,
                kernel_size,
            } => write!(
                f,
                "CNN layer {layer} uses kernel {kernel_size} with stride {stride}; the C two-part path only defines kernels 3 and 5"
            ),
            Self::UnsupportedActivation { layer, activation } => {
                write!(
                    f,
                    "CNN layer {layer} uses unsupported activation {activation:?}"
                )
            }
            Self::MissingGdnParameters { layer } => {
                write!(f, "CNN layer {layer} has no GDN/IGDN parameters")
            }
            Self::InvalidGdnParameters {
                layer,
                expected_beta,
                actual_beta,
                expected_gamma,
                actual_gamma,
            } => write!(
                f,
                "CNN layer {layer} GDN shape is beta={actual_beta}, gamma={actual_gamma}; expected beta={expected_beta}, gamma={expected_gamma}"
            ),
            Self::InvalidNormalization { layer, index, bits } => write!(
                f,
                "CNN layer {layer} normalization term {index} is not finite and positive (bits 0x{bits:08x})"
            ),
            Self::NonFiniteOutput { layer, index, bits } => write!(
                f,
                "CNN layer {layer} output {index} is not finite (bits 0x{bits:08x})"
            ),
            Self::WorkspaceTooLarge { values, limit } => write!(
                f,
                "CNN needs {values} values per workspace buffer; limit is {limit}"
            ),
        }
    }
}

impl std::error::Error for CnnError {}

/// Allocation-stable scalar runner for a parsed AVS3 decoder network.
///
/// Construction allocates three bounded work buffers. `decode` then performs
/// no allocation and preserves the grouping/order of `GEMM_REFORM_ENC` from
/// the C reference. The returned slice is valid until the next mutable call.
#[derive(Debug)]
pub struct ScalarCnnDecoder<'model> {
    network: &'model CnnNetwork,
    current: Vec<f32>,
    next: Vec<f32>,
    scratch: Vec<f32>,
    output_len: usize,
}

impl<'model> ScalarCnnDecoder<'model> {
    pub fn new(network: &'model CnnNetwork) -> Result<Self, CnnError> {
        if !network.is_transpose() {
            return Err(CnnError::ForwardNetwork);
        }

        let mut workspace_values = network.input_shape().len();
        for (index, layer) in network.layers().iter().enumerate() {
            validate_layer(layer, index)?;
            workspace_values = workspace_values.max(layer.output_shape().len());
        }
        if workspace_values > MAX_CNN_WORKSPACE_VALUES {
            return Err(CnnError::WorkspaceTooLarge {
                values: workspace_values,
                limit: MAX_CNN_WORKSPACE_VALUES,
            });
        }

        Ok(Self {
            network,
            current: vec![0.0; workspace_values],
            next: vec![0.0; workspace_values],
            scratch: vec![0.0; workspace_values],
            output_len: 0,
        })
    }

    pub fn network(&self) -> &CnnNetwork {
        self.network
    }

    pub fn output_shape(&self) -> LatentShape {
        self.network.output_shape()
    }

    pub fn decode(&mut self, input: &[f32]) -> Result<&[f32], CnnError> {
        let expected = self.network.input_shape().len();
        if input.len() != expected {
            return Err(CnnError::InputLength {
                expected,
                actual: input.len(),
            });
        }
        for (index, &value) in input.iter().enumerate() {
            if !value.is_finite() {
                return Err(CnnError::NonFiniteInput {
                    index,
                    bits: value.to_bits(),
                });
            }
        }

        self.current[..expected].copy_from_slice(input);
        let mut current_len = expected;
        for (layer_index, layer) in self.network.layers().iter().enumerate() {
            debug_assert_eq!(current_len, layer.input_shape().len());
            let next_len = layer.output_shape().len();
            decode_layer(
                layer,
                layer_index,
                &self.current[..current_len],
                &mut self.next[..next_len],
                &mut self.scratch[..next_len],
            )?;
            core::mem::swap(&mut self.current, &mut self.next);
            current_len = next_len;
        }
        self.output_len = current_len;
        Ok(&self.current[..self.output_len])
    }

    pub fn decode_into(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), CnnError> {
        let expected = self.output_shape().len();
        if output.len() != expected {
            return Err(CnnError::OutputLength {
                expected,
                actual: output.len(),
            });
        }
        output.copy_from_slice(self.decode(input)?);
        Ok(())
    }
}

fn validate_layer(layer: &CnnLayer, index: usize) -> Result<(), CnnError> {
    if !layer.is_transpose() {
        return Err(CnnError::ForwardNetwork);
    }
    if layer.padding() != Padding::Same {
        return Err(CnnError::UnsupportedPadding {
            layer: index,
            padding: layer.padding(),
        });
    }
    match layer.stride() {
        1 => {}
        2 if matches!(layer.kernel_size(), 3 | 5) => {}
        2 => {
            return Err(CnnError::UnsupportedKernel {
                layer: index,
                stride: 2,
                kernel_size: layer.kernel_size(),
            });
        }
        stride => {
            return Err(CnnError::UnsupportedStride {
                layer: index,
                stride,
            });
        }
    }
    match layer.activation() {
        Activation::Linear | Activation::Relu => {}
        Activation::Gdn | Activation::Igdn => {
            let parameters = layer
                .gdn_parameters()
                .ok_or(CnnError::MissingGdnParameters { layer: index })?;
            let channels = layer.output_shape().channels();
            let expected_gamma = channels * channels;
            if parameters.beta().len() != channels || parameters.gamma().len() != expected_gamma {
                return Err(CnnError::InvalidGdnParameters {
                    layer: index,
                    expected_beta: channels,
                    actual_beta: parameters.beta().len(),
                    expected_gamma,
                    actual_gamma: parameters.gamma().len(),
                });
            }
        }
        activation => {
            return Err(CnnError::UnsupportedActivation {
                layer: index,
                activation,
            });
        }
    }
    Ok(())
}

fn decode_layer(
    layer: &CnnLayer,
    layer_index: usize,
    input: &[f32],
    output: &mut [f32],
    scratch: &mut [f32],
) -> Result<(), CnnError> {
    match layer.stride() {
        1 => convolve_transpose_stride_one(layer, input, output),
        2 => convolve_transpose_stride_two(layer, input, output),
        stride => {
            return Err(CnnError::UnsupportedStride {
                layer: layer_index,
                stride,
            });
        }
    }

    if let Some(bias) = layer.bias() {
        let dimensions = layer.output_shape().dimensions();
        for channel in 0..layer.output_shape().channels() {
            for dimension in 0..dimensions {
                output[dimension + channel * dimensions] += bias[channel];
            }
        }
    }

    match layer.activation() {
        Activation::Linear => {}
        Activation::Relu => {
            for value in output.iter_mut() {
                if *value <= 0.0 {
                    *value = 0.0;
                }
            }
        }
        Activation::Gdn | Activation::Igdn => {
            apply_normalization(layer, layer_index, output, scratch)?;
        }
        activation => {
            return Err(CnnError::UnsupportedActivation {
                layer: layer_index,
                activation,
            });
        }
    }

    for (index, &value) in output.iter().enumerate() {
        if !value.is_finite() {
            return Err(CnnError::NonFiniteOutput {
                layer: layer_index,
                index,
                bits: value.to_bits(),
            });
        }
    }
    Ok(())
}

fn convolve_transpose_stride_one(layer: &CnnLayer, input: &[f32], output: &mut [f32]) {
    let dimensions = layer.input_shape().dimensions();
    let input_channels = layer.input_shape().channels();
    let output_channels = layer.output_shape().channels();
    let kernel_size = layer.kernel_size();
    let padding_begin = (kernel_size - 1) / 2;
    let dot_len = kernel_size * input_channels;
    let kernel = layer.kernel_coefficients();

    for output_channel in 0..output_channels {
        let padding_end = kernel_size - 1 - padding_begin;
        let interior_end = dimensions.saturating_sub(padding_end);
        let mut dimension = 0;
        while dimension < dimensions {
            if dimension >= padding_begin && dimension + 8 <= interior_end {
                let values = reformed_sum_wide(dot_len, |index| {
                    let input_channel = index / kernel_size;
                    let column_tap = index % kernel_size;
                    let source_position = dimension + column_tap - padding_begin;
                    let model_tap = kernel_size - column_tap - 1;
                    let kernel_index = (model_tap * output_channels + output_channel)
                        * input_channels
                        + input_channel;
                    let input_index = input_channel * dimensions + source_position;
                    let features = f32x8::new(
                        input[input_index..input_index + 8]
                            .try_into()
                            .expect("wide CNN input is an eight-value interior block"),
                    );
                    features * f32x8::splat(kernel[kernel_index])
                });
                let output_index = dimension + output_channel * dimensions;
                output[output_index..output_index + 8].copy_from_slice(&values.to_array());
                dimension += 8;
                continue;
            }

            let mut input_channel = 0;
            let mut column_tap = 0;
            output[dimension + output_channel * dimensions] = reformed_sum(dot_len, |_| {
                let source_position =
                    dimension as isize + column_tap as isize - padding_begin as isize;
                let feature = padded_value(input, dimensions, input_channel, source_position);
                let model_tap = kernel_size - column_tap - 1;
                let kernel_index =
                    (model_tap * output_channels + output_channel) * input_channels + input_channel;
                let product = kernel[kernel_index] * feature;
                column_tap += 1;
                if column_tap == kernel_size {
                    column_tap = 0;
                    input_channel += 1;
                }
                product
            });
            dimension += 1;
        }
    }
}

fn convolve_transpose_stride_two(layer: &CnnLayer, input: &[f32], output: &mut [f32]) {
    let input_dimensions = layer.input_shape().dimensions();
    let output_dimensions = layer.output_shape().dimensions();
    let input_channels = layer.input_shape().channels();
    let output_channels = layer.output_shape().channels();
    let kernel_size = layer.kernel_size();
    let odd_kernel_size = kernel_size.div_ceil(2);
    let even_kernel_size = (kernel_size - 1) / 2;
    let kernel = layer.kernel_coefficients();

    for output_channel in 0..output_channels {
        let interior_end = if kernel_size == 5 {
            input_dimensions - 1
        } else {
            input_dimensions
        };
        let mut dimension = 0;
        while dimension < input_dimensions {
            if dimension > 0 && dimension + 8 <= interior_end {
                convolve_transpose_stride_two_wide(
                    input,
                    output,
                    kernel,
                    input_dimensions,
                    output_dimensions,
                    input_channels,
                    output_channels,
                    kernel_size,
                    output_channel,
                    dimension,
                );
                dimension += 8;
                continue;
            }
            let mut odd_input_channel = 0;
            let mut odd_part_tap = 0;
            let odd = reformed_sum(odd_kernel_size * input_channels, |_| {
                let feature = padded_value(
                    input,
                    input_dimensions,
                    odd_input_channel,
                    dimension as isize + odd_part_tap as isize - 1,
                );
                let model_tap = kernel_size - 2 * odd_part_tap - 1;
                let kernel_index = (model_tap * output_channels + output_channel) * input_channels
                    + odd_input_channel;
                let product = kernel[kernel_index] * feature;
                odd_part_tap += 1;
                if odd_part_tap == odd_kernel_size {
                    odd_part_tap = 0;
                    odd_input_channel += 1;
                }
                product
            });
            let mut even_input_channel = 0;
            let mut even_part_tap = 0;
            let even = reformed_sum(even_kernel_size * input_channels, |_| {
                let source_position = if kernel_size == 3 {
                    dimension as isize
                } else {
                    dimension as isize + even_part_tap as isize - 1
                };
                let feature =
                    padded_value(input, input_dimensions, even_input_channel, source_position);
                let model_tap = kernel_size - (2 * even_part_tap + 1) - 1;
                let kernel_index = (model_tap * output_channels + output_channel) * input_channels
                    + even_input_channel;
                let product = kernel[kernel_index] * feature;
                even_part_tap += 1;
                if even_part_tap == even_kernel_size {
                    even_part_tap = 0;
                    even_input_channel += 1;
                }
                product
            });

            let output_index = output_channel * output_dimensions + 2 * dimension;
            if kernel_size == 5 {
                output[output_index] = even;
                output[output_index + 1] = odd;
            } else {
                output[output_index] = odd;
                output[output_index + 1] = even;
            }
            dimension += 1;
        }
    }
}

#[inline(always)]
fn convolve_transpose_stride_two_wide(
    input: &[f32],
    output: &mut [f32],
    kernel: &[f32],
    input_dimensions: usize,
    output_dimensions: usize,
    input_channels: usize,
    output_channels: usize,
    kernel_size: usize,
    output_channel: usize,
    dimension: usize,
) {
    let odd_kernel_size = kernel_size.div_ceil(2);
    let even_kernel_size = (kernel_size - 1) / 2;

    let odd = reformed_sum_wide(odd_kernel_size * input_channels, |index| {
        let input_channel = index / odd_kernel_size;
        let tap = index % odd_kernel_size;
        let source_position = dimension + tap - 1;
        let model_tap = kernel_size - 2 * tap - 1;
        let kernel_index =
            (model_tap * output_channels + output_channel) * input_channels + input_channel;
        let input_index = input_channel * input_dimensions + source_position;
        let features = f32x8::new(
            input[input_index..input_index + 8]
                .try_into()
                .expect("wide CNN input is an eight-value interior block"),
        );
        features * f32x8::splat(kernel[kernel_index])
    });

    let even = reformed_sum_wide(even_kernel_size * input_channels, |index| {
        let input_channel = index / even_kernel_size;
        let tap = index % even_kernel_size;
        let source_position = if kernel_size == 3 {
            dimension
        } else {
            dimension + tap - 1
        };
        let model_tap = kernel_size - (2 * tap + 1) - 1;
        let kernel_index =
            (model_tap * output_channels + output_channel) * input_channels + input_channel;
        let input_index = input_channel * input_dimensions + source_position;
        let features = f32x8::new(
            input[input_index..input_index + 8]
                .try_into()
                .expect("wide CNN input is an eight-value interior block"),
        );
        features * f32x8::splat(kernel[kernel_index])
    });

    let odd = odd.to_array();
    let even = even.to_array();
    let output_base = output_channel * output_dimensions + 2 * dimension;
    for lane in 0..8 {
        let output_index = output_base + 2 * lane;
        if kernel_size == 5 {
            output[output_index] = even[lane];
            output[output_index + 1] = odd[lane];
        } else {
            output[output_index] = odd[lane];
            output[output_index + 1] = even[lane];
        }
    }
}

// Each SIMD lane follows the scalar GEMM_REFORM accumulation independently;
// this vectorizes positions without changing their floating-point order.
#[inline(always)]
fn reformed_sum_wide(length: usize, mut product: impl FnMut(usize) -> f32x8) -> f32x8 {
    let grouped_end = length / 8 * 8;
    let mut first = f32x8::ZERO;
    let mut second = f32x8::ZERO;
    let mut third = f32x8::ZERO;
    let mut fourth = f32x8::ZERO;
    let mut index = 0;
    while index < grouped_end {
        first += product(index);
        first += product(index + 1);
        second += product(index + 2);
        second += product(index + 3);
        third += product(index + 4);
        third += product(index + 5);
        fourth += product(index + 6);
        fourth += product(index + 7);
        index += 8;
    }
    let grouped = (first + second) + (third + fourth);

    let mut tail = f32x8::ZERO;
    while index < length {
        tail += product(index);
        index += 1;
    }
    grouped + tail
}

fn padded_value(input: &[f32], dimensions: usize, channel: usize, position: isize) -> f32 {
    if position < 0 || position as usize >= dimensions {
        0.0
    } else {
        input[channel * dimensions + position as usize]
    }
}

fn apply_normalization(
    layer: &CnnLayer,
    layer_index: usize,
    output: &mut [f32],
    squared: &mut [f32],
) -> Result<(), CnnError> {
    let parameters = layer
        .gdn_parameters()
        .ok_or(CnnError::MissingGdnParameters { layer: layer_index })?;
    let dimensions = layer.output_shape().dimensions();
    let channels = layer.output_shape().channels();
    for (square, &value) in squared.iter_mut().zip(output.iter()) {
        *square = value * value;
    }

    let gamma = parameters.gamma();
    let activation = layer.activation();

    for (output_channel, output_row) in output.chunks_mut(dimensions).enumerate() {
        let gamma_row = &gamma[output_channel * channels..(output_channel + 1) * channels];
        let beta = parameters.beta()[output_channel];
        let mut dimension = 0;
        while dimension + 8 <= dimensions {
            let weighted = reformed_sum_wide(channels, |input_channel| {
                let input_index = input_channel * dimensions + dimension;
                let values = f32x8::new(
                    squared[input_index..input_index + 8]
                        .try_into()
                        .expect("wide GDN input is an eight-value block"),
                );
                values * f32x8::splat(gamma_row[input_channel])
            });
            let radicands = (weighted + f32x8::splat(beta)).to_array();
            for lane in 0..8 {
                let radicand = radicands[lane];
                if !radicand.is_finite() || radicand <= 0.0 {
                    let index = dimension + lane + output_channel * dimensions;
                    return Err(CnnError::InvalidNormalization {
                        layer: layer_index,
                        index,
                        bits: radicand.to_bits(),
                    });
                }
                let normalization = f64::from(radicand).sqrt() as f32;
                match activation {
                    Activation::Gdn => output_row[dimension + lane] /= normalization,
                    Activation::Igdn => output_row[dimension + lane] *= normalization,
                    _ => unreachable!("normalization is only called for GDN/IGDN"),
                }
            }
            dimension += 8;
        }

        while dimension < dimensions {
            let weighted = reformed_sum(channels, |input_channel| {
                squared[dimension + input_channel * dimensions] * gamma_row[input_channel]
            });
            let radicand = weighted + beta;
            if !radicand.is_finite() || radicand <= 0.0 {
                let index = dimension + output_channel * dimensions;
                return Err(CnnError::InvalidNormalization {
                    layer: layer_index,
                    index,
                    bits: radicand.to_bits(),
                });
            }
            // The reference calls the double-precision C sqrt and then casts
            // its result to float before multiplying/dividing.
            let normalization = f64::from(radicand).sqrt() as f32;
            match activation {
                Activation::Gdn => output_row[dimension] /= normalization,
                Activation::Igdn => output_row[dimension] *= normalization,
                _ => unreachable!("normalization is only called for GDN/IGDN"),
            }
            dimension += 1;
        }
    }
    Ok(())
}

/// Scalar dot product with the exact four-accumulator grouping selected by
/// `GEMM_REFORM_ENC`, including its separately accumulated tail.
fn reformed_sum(length: usize, mut product: impl FnMut(usize) -> f32) -> f32 {
    let grouped_end = length / 8 * 8;
    let mut first = 0.0_f32;
    let mut second = 0.0_f32;
    let mut third = 0.0_f32;
    let mut fourth = 0.0_f32;
    let mut index = 0;
    while index < grouped_end {
        first += product(index);
        first += product(index + 1);
        second += product(index + 2);
        second += product(index + 3);
        third += product(index + 4);
        third += product(index + 5);
        fourth += product(index + 6);
        fourth += product(index + 7);
        index += 8;
    }
    let first_half = first + second;
    let second_half = third + fourth;
    let grouped = first_half + second_half;

    let mut tail = 0.0_f32;
    while index < length {
        tail += product(index);
        index += 1;
    }
    grouped + tail
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelEncoding, NeuralModel, NeuralModelType};

    fn push_i16(output: &mut Vec<u8>, value: i16) {
        output.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u16(output: &mut Vec<u8>, value: u16) {
        output.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(output: &mut Vec<u8>, value: u32) {
        output.extend_from_slice(&value.to_le_bytes());
    }

    fn push_f32(output: &mut Vec<u8>, value: f32) {
        output.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    fn append_encoder(output: &mut Vec<u8>) {
        push_i16(output, 1); // layers
        push_i16(output, 0); // SAME
        push_i16(output, 2); // stride
        push_i16(output, 0); // no bias
        push_i16(output, 1); // LINEAR
        push_i16(output, 1); // kernel
        push_i16(output, 1); // input channels
        push_i16(output, 3); // output channels
        push_f32(output, 0.125);
        push_f32(output, -0.25);
        push_f32(output, 0.5);
    }

    fn append_decoder_layer_zero(output: &mut Vec<u8>) {
        push_i16(output, 0); // SAME
        push_i16(output, 1); // stride
        push_i16(output, 1); // bias
        push_i16(output, 5); // IGDN
        push_i16(output, 3); // kernel
        push_i16(output, 3); // input channels
        push_i16(output, 2); // output channels
        for tap in 0..3 {
            for output_channel in 0..2 {
                for input_channel in 0..3 {
                    let ordinal = tap * 2 * 3 + output_channel * 3 + input_channel;
                    push_f32(output, (ordinal as f32 - 8.0) / 32.0);
                }
            }
        }
        push_f32(output, 0.125);
        push_f32(output, -0.0625);
        push_f32(output, 1.0);
        push_f32(output, 0.75);
        // Serialized (input, output); the model loader transposes it exactly
        // like InitGdnParam.
        push_f32(output, 0.25);
        push_f32(output, 0.03125);
        push_f32(output, 0.0625);
        push_f32(output, 0.375);
    }

    fn append_decoder_layer_one(output: &mut Vec<u8>) {
        push_i16(output, 0); // SAME
        push_i16(output, 2); // stride
        push_i16(output, 1); // bias
        push_i16(output, 1); // LINEAR
        push_i16(output, 5); // kernel
        push_i16(output, 2); // input channels
        push_i16(output, 1); // output channels
        for tap in 0..5 {
            for output_channel in 0..1 {
                for input_channel in 0..2 {
                    let ordinal = tap * 2 + output_channel * 2 + input_channel;
                    push_f32(output, (ordinal as f32 - 4.0) / 16.0);
                }
            }
        }
        push_f32(output, 0.03125);
    }

    fn append_decoder_layer_one_kernel_three(output: &mut Vec<u8>) {
        push_i16(output, 0); // SAME
        push_i16(output, 2); // stride
        push_i16(output, 1); // bias
        push_i16(output, 1); // LINEAR
        push_i16(output, 3); // kernel
        push_i16(output, 2); // input channels
        push_i16(output, 1); // output channels
        for tap in 0..3 {
            for output_channel in 0..1 {
                for input_channel in 0..2 {
                    let ordinal = tap * 2 + output_channel * 2 + input_channel;
                    push_f32(output, (ordinal as f32 - 2.0) / 16.0);
                }
            }
        }
        push_f32(output, -0.015625);
    }

    fn reference_model() -> NeuralModel {
        reference_model_with_kernel(false)
    }

    fn reference_model_with_kernel(kernel_three: bool) -> NeuralModel {
        let mut bytes = Vec::new();
        append_encoder(&mut bytes);
        push_i16(&mut bytes, 2); // decoder layers
        append_decoder_layer_zero(&mut bytes);
        if kernel_three {
            append_decoder_layer_one_kernel_three(&mut bytes);
        } else {
            append_decoder_layer_one(&mut bytes);
        }
        for _ in 0..3 {
            push_f32(&mut bytes, 0.0); // medians
        }
        for _ in 0..3 {
            push_u16(&mut bytes, 3); // CDF lengths
        }
        for _ in 0..3 {
            push_i16(&mut bytes, 0); // CDF offsets
        }
        for _ in 0..3 {
            for value in [0_u32, 32_768, 65_536] {
                push_u32(&mut bytes, value);
            }
        }
        NeuralModel::from_bytes(&bytes, NeuralModelType::Vae, ModelEncoding::Plain).unwrap()
    }

    fn reference_input() -> Vec<f32> {
        let mut input = vec![0.0_f32; 512 * 3];
        for channel in 0..3 {
            for dimension in 0..512 {
                let value = (dimension * 17 + channel * 13) % 31;
                input[dimension + channel * 512] = (value as f32 - 15.0) / 16.0;
            }
        }
        input
    }

    fn hash_f32(values: &[f32]) -> u64 {
        values
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
                (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3)
            })
    }

    fn sample_bits(values: &[f32]) -> Vec<u32> {
        [0, 1, 2, 7, 127, 255, 511, 512, 777, 1023]
            .into_iter()
            .filter_map(|index| values.get(index))
            .map(|value| value.to_bits())
            .collect()
    }

    #[test]
    fn reformed_sum_keeps_four_accumulators_and_a_tail() {
        let values = [
            16_777_216.0_f32,
            1.0,
            -16_777_216.0,
            1.0,
            3.0,
            -3.0,
            5.0,
            -5.0,
            0.25,
        ];
        // This is intentionally not equivalent to a simple left fold.
        assert_eq!(reformed_sum(values.len(), |index| values[index]), 1.25);
    }

    #[test]
    fn transpose_convolution_and_igdn_match_c_reference_bits() {
        // Generated by the C reference's InitCnnLayer,
        // Conv1DTranspose and Conv1DTranspose2Part at -O0/-O1/-O3.
        let model = reference_model();
        let layers = model.base().decoder().layers();
        let input = reference_input();
        let mut first = vec![0.0_f32; layers[0].output_shape().len()];
        let mut first_scratch = vec![0.0_f32; first.len()];
        decode_layer(&layers[0], 0, &input, &mut first, &mut first_scratch).unwrap();
        assert_eq!(hash_f32(&first), 0xc65c_d318_05f6_dfe1);
        assert_eq!(
            sample_bits(&first),
            [
                0xbc60_2bfd,
                0x3db8_311f,
                0x3ecf_6056,
                0xbc00_29aa,
                0x3ef2_5035,
                0xbc00_29aa,
                0x3db4_73be,
                0xbe09_a501,
                0xbe0d_2c79,
                0xbe47_eadc,
            ]
        );

        let mut second = vec![0.0_f32; layers[1].output_shape().len()];
        let mut second_scratch = vec![0.0_f32; second.len()];
        decode_layer(&layers[1], 1, &first, &mut second, &mut second_scratch).unwrap();
        assert_eq!(hash_f32(&second), 0xc6ff_103e_7c01_ec53);
        assert_eq!(
            sample_bits(&second),
            [
                0x3d29_6aa0,
                0x3bbc_b0e0,
                0xbba7_60d0,
                0x3e80_ae9b,
                0xbe07_48a9,
                0x3e80_ae9b,
                0xbba6_3300,
                0x3c25_9652,
                0xbc8d_4d96,
                0x3e36_ec52,
            ]
        );

        let mut decoder = ScalarCnnDecoder::new(model.base().decoder()).unwrap();
        assert_eq!(decoder.decode(&input).unwrap(), second);
        // A second call reuses the same buffers and remains deterministic.
        assert_eq!(decoder.decode(&input).unwrap(), second);
    }

    #[test]
    fn runner_rejects_wrong_lengths_and_non_finite_input() {
        let model = reference_model();
        let mut decoder = ScalarCnnDecoder::new(model.base().decoder()).unwrap();
        assert!(matches!(
            decoder.decode(&[]),
            Err(CnnError::InputLength { .. })
        ));
        let mut input = reference_input();
        input[17] = f32::NAN;
        assert!(matches!(
            decoder.decode(&input),
            Err(CnnError::NonFiniteInput { index: 17, .. })
        ));
    }

    #[test]
    fn stride_two_kernel_three_matches_c_reference_bits() {
        let model = reference_model_with_kernel(true);
        let layers = model.base().decoder().layers();
        let input = reference_input();
        let mut first = vec![0.0_f32; layers[0].output_shape().len()];
        let mut scratch = vec![0.0_f32; first.len()];
        decode_layer(&layers[0], 0, &input, &mut first, &mut scratch).unwrap();

        let mut output = vec![0.0_f32; layers[1].output_shape().len()];
        scratch.resize(output.len(), 0.0);
        decode_layer(&layers[1], 1, &first, &mut output, &mut scratch).unwrap();
        assert_eq!(hash_f32(&output), 0xf998_56b7_4567_0ab6);
        assert_eq!(
            sample_bits(&output),
            [
                0xbbb4_aafe,
                0xbcc4_d280,
                0xbd54_ec1a,
                0xbc09_2f34,
                0xbc8e_ba9d,
                0xbc09_2f34,
                0xbcda_1944,
                0xbd16_9a6c,
                0xbcfa_93cf,
                0xbce3_f56e,
            ]
        );
    }
}
