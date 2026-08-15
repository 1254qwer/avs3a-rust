use core::fmt;

use crate::latent::{ContextScaleTable, LatentError, LatentShape, MAX_LATENT_CHANNELS, Quantizer};
use crate::range_coder::{RangeCoderConfig, RangeCoderError};

pub const AVS3_MODEL_XOR_MASK: u8 = 0x55;
pub const AVS3_FEATURE_DIMENSIONS: usize = 1_024;
pub const DEFAULT_MAX_MODEL_BYTES: usize = 4 * 1024 * 1024;
pub const DEFAULT_MAX_MODEL_VALUES: usize = 4 * 1024 * 1024;
pub const DEFAULT_MAX_MODEL_LAYERS: usize = 10;
pub const DEFAULT_MAX_MODEL_CHANNELS: usize = 4_096;
pub const DEFAULT_MAX_KERNEL_SIZE: usize = 255;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelEncoding {
    Plain,
    Xor55,
}

impl ModelEncoding {
    fn mask(self) -> u8 {
        match self {
            Self::Plain => 0,
            Self::Xor55 => AVS3_MODEL_XOR_MASK,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelLimits {
    pub max_model_bytes: usize,
    pub max_collection_values: usize,
    pub max_layers: usize,
    pub max_channels: usize,
    pub max_kernel_size: usize,
}

impl Default for ModelLimits {
    fn default() -> Self {
        Self {
            max_model_bytes: DEFAULT_MAX_MODEL_BYTES,
            max_collection_values: DEFAULT_MAX_MODEL_VALUES,
            max_layers: DEFAULT_MAX_MODEL_LAYERS,
            max_channels: DEFAULT_MAX_MODEL_CHANNELS,
            max_kernel_size: DEFAULT_MAX_KERNEL_SIZE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    InputTooLarge {
        bytes: usize,
        limit: usize,
    },
    UnexpectedEof {
        position: usize,
        requested: usize,
        available: usize,
    },
    IntegerOverflow,
    CollectionTooLarge {
        collection: &'static str,
        values: usize,
        limit: usize,
    },
    NonFiniteFloat {
        position: usize,
        bits: u32,
    },
    InvalidLayerCount {
        network: &'static str,
        value: i16,
        limit: usize,
    },
    InvalidField {
        field: &'static str,
        value: i16,
    },
    ChannelMismatch {
        network: &'static str,
        layer: usize,
        expected: usize,
        actual: usize,
    },
    DimensionNotDivisible {
        network: &'static str,
        layer: usize,
        dimensions: usize,
        stride: usize,
    },
    NetworkOutputMismatch {
        network: &'static str,
        expected_dimensions: usize,
        expected_channels: usize,
        actual_dimensions: usize,
        actual_channels: usize,
    },
    TrailingData {
        bytes: usize,
    },
    Latent(LatentError),
    RangeCoder(RangeCoderError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooLarge { bytes, limit } => {
                write!(f, "model has {bytes} bytes; limit is {limit}")
            }
            Self::UnexpectedEof {
                position,
                requested,
                available,
            } => write!(
                f,
                "model ended at byte {position}; need {requested} bytes, {available} remain"
            ),
            Self::IntegerOverflow => f.write_str("model size arithmetic overflow"),
            Self::CollectionTooLarge {
                collection,
                values,
                limit,
            } => write!(f, "{collection} has {values} values; limit is {limit}"),
            Self::NonFiniteFloat { position, bits } => write!(
                f,
                "model float at byte {position} is not finite (bits 0x{bits:08x})"
            ),
            Self::InvalidLayerCount {
                network,
                value,
                limit,
            } => write!(
                f,
                "{network} declares {value} layers; valid range is 1..={limit}"
            ),
            Self::InvalidField { field, value } => {
                write!(f, "model field {field} has invalid value {value}")
            }
            Self::ChannelMismatch {
                network,
                layer,
                expected,
                actual,
            } => write!(
                f,
                "{network} layer {layer} expects {actual} input channels; previous shape has {expected}"
            ),
            Self::DimensionNotDivisible {
                network,
                layer,
                dimensions,
                stride,
            } => write!(
                f,
                "{network} layer {layer} cannot downsample {dimensions} dimensions by stride {stride}"
            ),
            Self::NetworkOutputMismatch {
                network,
                expected_dimensions,
                expected_channels,
                actual_dimensions,
                actual_channels,
            } => write!(
                f,
                "{network} output is {actual_dimensions}x{actual_channels}; expected {expected_dimensions}x{expected_channels}"
            ),
            Self::TrailingData { bytes } => {
                write!(f, "model has {bytes} unconsumed trailing bytes")
            }
            Self::Latent(error) => error.fmt(f),
            Self::RangeCoder(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Latent(error) => Some(error),
            Self::RangeCoder(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LatentError> for ModelError {
    fn from(value: LatentError) -> Self {
        Self::Latent(value)
    }
}

impl From<RangeCoderError> for ModelError {
    fn from(value: RangeCoderError) -> Self {
        Self::RangeCoder(value)
    }
}

/// Bounds-checked, explicitly little-endian replacement for the C model's
/// `modul_structure + memcpy + nIndex` cursor.
///
/// The XOR-obfuscated embedded model can be consumed without first creating a
/// decrypted copy. Failed scalar or collection reads leave `position()`
/// unchanged, which makes block parsers transactional.
#[derive(Debug, Clone)]
pub struct ModelReader<'a> {
    input: &'a [u8],
    position: usize,
    xor_mask: u8,
    limits: ModelLimits,
}

impl<'a> ModelReader<'a> {
    pub fn new(input: &'a [u8]) -> Result<Self, ModelError> {
        Self::with_limits(input, ModelEncoding::Plain, ModelLimits::default())
    }

    pub fn new_xor55(input: &'a [u8]) -> Result<Self, ModelError> {
        Self::with_limits(input, ModelEncoding::Xor55, ModelLimits::default())
    }

    pub fn with_limits(
        input: &'a [u8],
        encoding: ModelEncoding,
        limits: ModelLimits,
    ) -> Result<Self, ModelError> {
        if input.len() > limits.max_model_bytes {
            return Err(ModelError::InputTooLarge {
                bytes: input.len(),
                limit: limits.max_model_bytes,
            });
        }
        Ok(Self {
            input,
            position: 0,
            xor_mask: encoding.mask(),
            limits,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn remaining(&self) -> usize {
        self.input.len() - self.position
    }

    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    pub fn limits(&self) -> ModelLimits {
        self.limits
    }

    pub fn read_u16(&mut self) -> Result<u16, ModelError> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    pub fn read_i16(&mut self) -> Result<i16, ModelError> {
        Ok(i16::from_le_bytes(self.read_array()?))
    }

    pub fn read_u32(&mut self) -> Result<u32, ModelError> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    pub fn read_f32(&mut self) -> Result<f32, ModelError> {
        self.transaction(|reader| {
            let position = reader.position;
            let value = f32::from_bits(reader.read_u32()?);
            if !value.is_finite() {
                return Err(ModelError::NonFiniteFloat {
                    position,
                    bits: value.to_bits(),
                });
            }
            Ok(value)
        })
    }

    pub fn read_u16s(
        &mut self,
        count: usize,
        collection: &'static str,
    ) -> Result<Vec<u16>, ModelError> {
        self.read_collection(count, 2, collection, |reader| reader.read_u16())
    }

    pub fn read_i16s(
        &mut self,
        count: usize,
        collection: &'static str,
    ) -> Result<Vec<i16>, ModelError> {
        self.read_collection(count, 2, collection, |reader| reader.read_i16())
    }

    pub fn read_u32s(
        &mut self,
        count: usize,
        collection: &'static str,
    ) -> Result<Vec<u32>, ModelError> {
        self.read_collection(count, 4, collection, |reader| reader.read_u32())
    }

    pub fn read_f32s(
        &mut self,
        count: usize,
        collection: &'static str,
    ) -> Result<Vec<f32>, ModelError> {
        self.read_collection(count, 4, collection, |reader| reader.read_f32())
    }

    pub fn read_range_coder_config(
        &mut self,
        num_cdfs: usize,
    ) -> Result<RangeCoderConfig, ModelError> {
        self.transaction(|reader| {
            if num_cdfs > u16::MAX as usize {
                return Err(RangeCoderError::TooManyCdfs(num_cdfs).into());
            }
            reader.check_collection("range-coder CDF tables", num_cdfs)?;
            let lengths = reader.read_u16s(num_cdfs, "range-coder CDF lengths")?;
            let offsets = reader.read_i16s(num_cdfs, "range-coder CDF offsets")?;
            let table_values = lengths.iter().try_fold(0_usize, |total, &length| {
                total
                    .checked_add(usize::from(length))
                    .ok_or(ModelError::IntegerOverflow)
            })?;
            reader.check_collection("range-coder CDF values", table_values)?;

            // Check the complete table block before allocating any individual
            // CDF, so a truncated count vector cannot cause partial work.
            let table_bytes = table_values
                .checked_mul(4)
                .ok_or(ModelError::IntegerOverflow)?;
            reader.require(table_bytes)?;

            let mut cdfs = Vec::with_capacity(num_cdfs);
            for length in lengths {
                cdfs.push(reader.read_u32s(usize::from(length), "range-coder CDF values")?);
            }
            Ok(RangeCoderConfig::new(cdfs, offsets)?)
        })
    }

    fn read_collection<T, F>(
        &mut self,
        count: usize,
        element_bytes: usize,
        collection: &'static str,
        mut read: F,
    ) -> Result<Vec<T>, ModelError>
    where
        F: FnMut(&mut Self) -> Result<T, ModelError>,
    {
        self.check_collection(collection, count)?;
        let bytes = count
            .checked_mul(element_bytes)
            .ok_or(ModelError::IntegerOverflow)?;
        self.require(bytes)?;
        self.transaction(|reader| {
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read(reader)?);
            }
            Ok(values)
        })
    }

    fn check_collection(&self, collection: &'static str, values: usize) -> Result<(), ModelError> {
        if values > self.limits.max_collection_values {
            Err(ModelError::CollectionTooLarge {
                collection,
                values,
                limit: self.limits.max_collection_values,
            })
        } else {
            Ok(())
        }
    }

    fn require(&self, requested: usize) -> Result<usize, ModelError> {
        let end = self
            .position
            .checked_add(requested)
            .ok_or(ModelError::IntegerOverflow)?;
        if end > self.input.len() {
            Err(ModelError::UnexpectedEof {
                position: self.position,
                requested,
                available: self.remaining(),
            })
        } else {
            Ok(end)
        }
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], ModelError> {
        let end = self.require(N)?;
        let mut decoded = [0_u8; N];
        for (output, &input) in decoded.iter_mut().zip(&self.input[self.position..end]) {
            *output = input ^ self.xor_mask;
        }
        self.position = end;
        Ok(decoded)
    }

    fn transaction<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> Result<T, ModelError>,
    ) -> Result<T, ModelError> {
        let checkpoint = self.position;
        match operation(self) {
            Ok(value) => Ok(value),
            Err(error) => {
                self.position = checkpoint;
                Err(error)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    Same,
    Valid,
}

impl Padding {
    fn parse(value: i16) -> Result<Self, ModelError> {
        match value {
            0 => Ok(Self::Same),
            1 => Ok(Self::Valid),
            _ => Err(ModelError::InvalidField {
                field: "CNN padding",
                value,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Linear,
    Sigmoid,
    Tanh,
    Gdn,
    Igdn,
    Dn,
    Idn,
}

impl Activation {
    fn parse(value: i16) -> Result<Self, ModelError> {
        match value {
            0 => Ok(Self::Relu),
            1 => Ok(Self::Linear),
            2 => Ok(Self::Sigmoid),
            3 => Ok(Self::Tanh),
            4 => Ok(Self::Gdn),
            5 => Ok(Self::Igdn),
            6 => Ok(Self::Dn),
            7 => Ok(Self::Idn),
            _ => Err(ModelError::InvalidField {
                field: "CNN activation",
                value,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GdnParameters {
    beta: Vec<f32>,
    gamma: Vec<f32>,
}

impl GdnParameters {
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }

    /// Matrix in the exact post-load layout used by C's GEMM:
    /// `gamma[output_channel * channels + input_channel]`.
    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CnnLayer {
    transpose: bool,
    padding: Padding,
    stride: usize,
    activation: Activation,
    kernel_size: usize,
    input_shape: LatentShape,
    output_shape: LatentShape,
    kernel: Vec<f32>,
    bias: Option<Vec<f32>>,
    gdn: Option<GdnParameters>,
}

impl CnnLayer {
    pub fn is_transpose(&self) -> bool {
        self.transpose
    }

    pub fn padding(&self) -> Padding {
        self.padding
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn activation(&self) -> Activation {
        self.activation
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub fn input_shape(&self) -> LatentShape {
        self.input_shape
    }

    pub fn output_shape(&self) -> LatentShape {
        self.output_shape
    }

    pub fn kernel_coefficients(&self) -> &[f32] {
        &self.kernel
    }

    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    pub fn gdn_parameters(&self) -> Option<&GdnParameters> {
        self.gdn.as_ref()
    }

    /// Read a coefficient using normalized `(tap, input, output)` indexes.
    /// The serialized transpose-convolution tensor swaps its last two axes;
    /// this method hides that model-format detail from the scalar backend.
    pub fn kernel_coefficient(
        &self,
        tap: usize,
        input_channel: usize,
        output_channel: usize,
    ) -> Option<f32> {
        if tap >= self.kernel_size
            || input_channel >= self.input_shape.channels()
            || output_channel >= self.output_shape.channels()
        {
            return None;
        }
        let index = if self.transpose {
            (tap * self.output_shape.channels() + output_channel) * self.input_shape.channels()
                + input_channel
        } else {
            (tap * self.input_shape.channels() + input_channel) * self.output_shape.channels()
                + output_channel
        };
        self.kernel.get(index).copied()
    }

    fn read(
        reader: &mut ModelReader<'_>,
        transpose: bool,
        input_shape: LatentShape,
        network: &'static str,
        layer: usize,
    ) -> Result<Self, ModelError> {
        let padding = Padding::parse(reader.read_i16()?)?;
        let stride_raw = reader.read_i16()?;
        let stride = positive_bounded(stride_raw, "CNN stride", 2)?;
        let use_bias = match reader.read_i16()? {
            0 => false,
            1 => true,
            value => {
                return Err(ModelError::InvalidField {
                    field: "CNN use-bias flag",
                    value,
                });
            }
        };
        let activation = Activation::parse(reader.read_i16()?)?;
        let kernel_size = positive_bounded(
            reader.read_i16()?,
            "CNN kernel size",
            reader.limits.max_kernel_size,
        )?;
        let input_channels = positive_bounded(
            reader.read_i16()?,
            "CNN input channels",
            reader.limits.max_channels.min(MAX_LATENT_CHANNELS),
        )?;
        let output_channels = positive_bounded(
            reader.read_i16()?,
            "CNN output channels",
            reader.limits.max_channels.min(MAX_LATENT_CHANNELS),
        )?;
        if input_channels != input_shape.channels() {
            return Err(ModelError::ChannelMismatch {
                network,
                layer,
                expected: input_shape.channels(),
                actual: input_channels,
            });
        }

        let output_dimensions = if transpose {
            input_shape
                .dimensions()
                .checked_mul(stride)
                .ok_or(ModelError::IntegerOverflow)?
        } else {
            if !input_shape.dimensions().is_multiple_of(stride) {
                return Err(ModelError::DimensionNotDivisible {
                    network,
                    layer,
                    dimensions: input_shape.dimensions(),
                    stride,
                });
            }
            input_shape.dimensions() / stride
        };
        let output_shape = LatentShape::new(output_dimensions, output_channels)?;

        let kernel_values = kernel_size
            .checked_mul(input_channels)
            .and_then(|value| value.checked_mul(output_channels))
            .ok_or(ModelError::IntegerOverflow)?;
        let kernel = reader.read_f32s(kernel_values, "CNN kernel coefficients")?;
        let bias = if use_bias {
            Some(reader.read_f32s(output_channels, "CNN bias coefficients")?)
        } else {
            None
        };
        let gdn = if matches!(activation, Activation::Gdn | Activation::Igdn) {
            let beta = reader.read_f32s(output_channels, "GDN beta coefficients")?;
            let gamma_values = output_channels
                .checked_mul(output_channels)
                .ok_or(ModelError::IntegerOverflow)?;
            let serialized_gamma = reader.read_f32s(gamma_values, "GDN gamma coefficients")?;
            // InitGdnParam reads in (input, output) order, then writes to
            // gamma[input + output * channels]. Preserve the resulting C
            // memory layout so GEMM consumes coefficients in the same order.
            let mut gamma = vec![0.0_f32; gamma_values];
            for input_channel in 0..output_channels {
                for output_channel in 0..output_channels {
                    gamma[input_channel + output_channel * output_channels] =
                        serialized_gamma[input_channel * output_channels + output_channel];
                }
            }
            Some(GdnParameters { beta, gamma })
        } else {
            None
        };

        Ok(Self {
            transpose,
            padding,
            stride,
            activation,
            kernel_size,
            input_shape,
            output_shape,
            kernel,
            bias,
            gdn,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CnnNetwork {
    transpose: bool,
    input_shape: LatentShape,
    output_shape: LatentShape,
    layers: Vec<CnnLayer>,
}

impl CnnNetwork {
    pub fn is_transpose(&self) -> bool {
        self.transpose
    }

    pub fn input_shape(&self) -> LatentShape {
        self.input_shape
    }

    pub fn output_shape(&self) -> LatentShape {
        self.output_shape
    }

    pub fn layers(&self) -> &[CnnLayer] {
        &self.layers
    }

    fn read(
        reader: &mut ModelReader<'_>,
        transpose: bool,
        input_shape: LatentShape,
        network: &'static str,
    ) -> Result<Self, ModelError> {
        let count_raw = reader.read_i16()?;
        if count_raw <= 0 || count_raw as usize > reader.limits.max_layers {
            return Err(ModelError::InvalidLayerCount {
                network,
                value: count_raw,
                limit: reader.limits.max_layers,
            });
        }
        let count = count_raw as usize;
        let mut layers = Vec::with_capacity(count);
        let mut shape = input_shape;
        for layer in 0..count {
            let parsed = CnnLayer::read(reader, transpose, shape, network, layer)?;
            shape = parsed.output_shape;
            layers.push(parsed);
        }
        Ok(Self {
            transpose,
            input_shape,
            output_shape: shape,
            layers,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuralModelType {
    Vae,
    Hyper,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeuralCodecModel {
    input_shape: LatentShape,
    latent_shape: LatentShape,
    encoder: CnnNetwork,
    decoder: CnnNetwork,
    quantizer: Quantizer,
    context_scales: Option<ContextScaleTable>,
    range_coder: RangeCoderConfig,
}

impl NeuralCodecModel {
    pub fn input_shape(&self) -> LatentShape {
        self.input_shape
    }

    pub fn latent_shape(&self) -> LatentShape {
        self.latent_shape
    }

    pub fn encoder(&self) -> &CnnNetwork {
        &self.encoder
    }

    pub fn decoder(&self) -> &CnnNetwork {
        &self.decoder
    }

    pub fn quantizer(&self) -> &Quantizer {
        &self.quantizer
    }

    pub fn context_scales(&self) -> Option<&ContextScaleTable> {
        self.context_scales.as_ref()
    }

    pub fn range_coder(&self) -> &RangeCoderConfig {
        &self.range_coder
    }

    fn read(
        reader: &mut ModelReader<'_>,
        input_shape: LatentShape,
        has_context: bool,
        encoder_name: &'static str,
        decoder_name: &'static str,
    ) -> Result<Self, ModelError> {
        let encoder = CnnNetwork::read(reader, false, input_shape, encoder_name)?;
        let latent_shape = encoder.output_shape;
        let decoder = CnnNetwork::read(reader, true, latent_shape, decoder_name)?;
        if decoder.output_shape != input_shape {
            return Err(ModelError::NetworkOutputMismatch {
                network: decoder_name,
                expected_dimensions: input_shape.dimensions(),
                expected_channels: input_shape.channels(),
                actual_dimensions: decoder.output_shape.dimensions(),
                actual_channels: decoder.output_shape.channels(),
            });
        }

        let medians = reader.read_f32s(latent_shape.channels(), "quantizer median coefficients")?;
        let quantizer = Quantizer::new(medians)?;

        let context_scales = if has_context {
            let count_raw = reader.read_i16()?;
            let count = positive_bounded(count_raw, "context-scale count", u16::MAX as usize)?;
            Some(ContextScaleTable::new(
                reader.read_f32s(count, "context scales")?,
            )?)
        } else {
            None
        };
        let num_cdfs = context_scales
            .as_ref()
            .map_or(latent_shape.channels(), |table| table.scales().len());
        let range_coder = reader.read_range_coder_config(num_cdfs)?;

        Ok(Self {
            input_shape,
            latent_shape,
            encoder,
            decoder,
            quantizer,
            context_scales,
            range_coder,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeuralModel {
    model_type: NeuralModelType,
    base: NeuralCodecModel,
    context: Option<NeuralCodecModel>,
}

impl NeuralModel {
    pub fn from_bytes(
        input: &[u8],
        model_type: NeuralModelType,
        encoding: ModelEncoding,
    ) -> Result<Self, ModelError> {
        Self::from_bytes_with_limits(input, model_type, encoding, ModelLimits::default())
    }

    pub fn from_obfuscated_bytes(
        input: &[u8],
        model_type: NeuralModelType,
    ) -> Result<Self, ModelError> {
        Self::from_bytes(input, model_type, ModelEncoding::Xor55)
    }

    pub fn from_bytes_with_limits(
        input: &[u8],
        model_type: NeuralModelType,
        encoding: ModelEncoding,
        limits: ModelLimits,
    ) -> Result<Self, ModelError> {
        let mut reader = ModelReader::with_limits(input, encoding, limits)?;
        let base_input = LatentShape::new(AVS3_FEATURE_DIMENSIONS, 1)?;
        let base = NeuralCodecModel::read(
            &mut reader,
            base_input,
            model_type == NeuralModelType::Hyper,
            "base encoder",
            "base decoder",
        )?;
        let context = if model_type == NeuralModelType::Hyper {
            Some(NeuralCodecModel::read(
                &mut reader,
                base.latent_shape,
                false,
                "context encoder",
                "context decoder",
            )?)
        } else {
            None
        };
        if !reader.is_empty() {
            return Err(ModelError::TrailingData {
                bytes: reader.remaining(),
            });
        }
        Ok(Self {
            model_type,
            base,
            context,
        })
    }

    pub fn model_type(&self) -> NeuralModelType {
        self.model_type
    }

    pub fn base(&self) -> &NeuralCodecModel {
        &self.base
    }

    pub fn context(&self) -> Option<&NeuralCodecModel> {
        self.context.as_ref()
    }
}

fn positive_bounded(value: i16, field: &'static str, maximum: usize) -> Result<usize, ModelError> {
    if value <= 0 || value as usize > maximum {
        Err(ModelError::InvalidField { field, value })
    } else {
        Ok(value as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn push_layer(
        output: &mut Vec<u8>,
        stride: i16,
        input_channels: i16,
        output_channels: i16,
        transpose: bool,
    ) {
        push_i16(output, 0); // SAME
        push_i16(output, stride);
        push_i16(output, 1); // bias
        push_i16(output, 1); // LINEAR
        push_i16(output, 1); // kernel size
        push_i16(output, input_channels);
        push_i16(output, output_channels);
        // The serialized axis order differs, but the scalar count does not.
        for index in 0..usize::from(input_channels as u16) * usize::from(output_channels as u16) {
            let sign = if transpose { -1.0 } else { 1.0 };
            push_f32(output, sign * (index + 1) as f32 / 8.0);
        }
        for index in 0..output_channels {
            push_f32(output, f32::from(index) / 16.0);
        }
    }

    fn push_range_config(output: &mut Vec<u8>, num_cdfs: usize) {
        for _ in 0..num_cdfs {
            push_u16(output, 3);
        }
        for index in 0..num_cdfs {
            push_i16(output, -(index as i16));
        }
        for _ in 0..num_cdfs {
            for value in [0_u32, 32_768, 65_536] {
                push_u32(output, value);
            }
        }
    }

    fn push_codec(
        output: &mut Vec<u8>,
        input_channels: i16,
        latent_channels: i16,
        has_context: bool,
    ) {
        push_i16(output, 1);
        push_layer(output, 2, input_channels, latent_channels, false);
        push_i16(output, 1);
        push_layer(output, 2, latent_channels, input_channels, true);
        for channel in 0..latent_channels {
            push_f32(output, f32::from(channel) / 4.0);
        }
        let num_cdfs = if has_context {
            push_i16(output, 2);
            push_f32(output, 0.5);
            push_f32(output, 2.0);
            2
        } else {
            latent_channels as usize
        };
        push_range_config(output, num_cdfs);
    }

    fn vae_model() -> Vec<u8> {
        let mut output = Vec::new();
        push_codec(&mut output, 1, 2, false);
        output
    }

    fn hyper_model() -> Vec<u8> {
        let mut output = Vec::new();
        push_codec(&mut output, 1, 2, true);
        // Base latent dimensions are 512, with two channels.
        push_codec(&mut output, 2, 2, false);
        output
    }

    #[test]
    fn reader_is_little_endian_transactional_and_xor_aware() {
        let bytes = [0x34, 0x12, 0x00, 0x00, 0xc0, 0x7f];
        let mut reader = ModelReader::new(&bytes).unwrap();
        assert_eq!(reader.read_u16().unwrap(), 0x1234);
        let checkpoint = reader.position();
        assert!(matches!(
            reader.read_f32(),
            Err(ModelError::NonFiniteFloat { .. })
        ));
        assert_eq!(reader.position(), checkpoint);

        let encoded: Vec<u8> = 0x1234_u16
            .to_le_bytes()
            .into_iter()
            .map(|byte| byte ^ AVS3_MODEL_XOR_MASK)
            .collect();
        assert_eq!(
            ModelReader::new_xor55(&encoded).unwrap().read_u16(),
            Ok(0x1234)
        );

        let mut short = ModelReader::new(&[1]).unwrap();
        assert!(matches!(
            short.read_u16(),
            Err(ModelError::UnexpectedEof { .. })
        ));
        assert_eq!(short.position(), 0);
    }

    #[test]
    fn xor_prefix_matches_the_embedded_c_model_header() {
        let prefix = [
            0x51, 0x55, 0x55, 0x55, 0x57, 0x55, 0x54, 0x55, 0x51, 0x55, 0x50, 0x55, 0x54, 0x55,
            0x57, 0x55,
        ];
        let mut reader = ModelReader::new_xor55(&prefix).unwrap();
        assert_eq!(reader.read_i16().unwrap(), 4); // encoder layers
        assert_eq!(reader.read_i16().unwrap(), 0); // SAME
        assert_eq!(reader.read_i16().unwrap(), 2); // stride
        assert_eq!(reader.read_i16().unwrap(), 1); // bias
        assert_eq!(reader.read_i16().unwrap(), 4); // GDN
        assert_eq!(reader.read_i16().unwrap(), 5); // kernel
        assert_eq!(reader.read_i16().unwrap(), 1); // input channels
        assert_eq!(reader.read_i16().unwrap(), 2); // output channels
    }

    #[test]
    fn parses_owned_vae_model_in_plain_and_obfuscated_forms() {
        let plain = vae_model();
        let model =
            NeuralModel::from_bytes(&plain, NeuralModelType::Vae, ModelEncoding::Plain).unwrap();
        assert_eq!(
            model.base().input_shape(),
            LatentShape::new(1024, 1).unwrap()
        );
        assert_eq!(
            model.base().latent_shape(),
            LatentShape::new(512, 2).unwrap()
        );
        assert_eq!(model.base().encoder().layers().len(), 1);
        assert_eq!(model.base().decoder().layers().len(), 1);
        assert_eq!(model.base().quantizer().quantile_medians(), [0.0, 0.25]);
        assert!(model.base().context_scales().is_none());
        assert!(model.context().is_none());

        let obfuscated: Vec<u8> = plain
            .iter()
            .map(|&byte| byte ^ AVS3_MODEL_XOR_MASK)
            .collect();
        assert_eq!(
            NeuralModel::from_obfuscated_bytes(&obfuscated, NeuralModelType::Vae).unwrap(),
            model
        );
    }

    #[test]
    fn parses_hyper_model_and_normalizes_kernel_indexes() {
        let bytes = hyper_model();
        let model =
            NeuralModel::from_bytes(&bytes, NeuralModelType::Hyper, ModelEncoding::Plain).unwrap();
        assert_eq!(model.base().context_scales().unwrap().scales(), [0.5, 2.0]);
        let context = model.context().unwrap();
        assert_eq!(context.input_shape(), LatentShape::new(512, 2).unwrap());
        assert_eq!(context.latent_shape(), LatentShape::new(256, 2).unwrap());

        let decoder = &context.decoder().layers()[0];
        assert!(decoder.is_transpose());
        assert_eq!(decoder.kernel_coefficient(0, 0, 0), Some(-0.125));
        assert_eq!(decoder.kernel_coefficient(0, 1, 0), Some(-0.25));
        assert_eq!(decoder.kernel_coefficient(0, 0, 1), Some(-0.375));
        assert_eq!(decoder.kernel_coefficient(0, 1, 1), Some(-0.5));
    }

    #[test]
    fn every_truncated_model_prefix_is_rejected_without_panicking() {
        let bytes = hyper_model();
        for end in 0..bytes.len() {
            assert!(
                NeuralModel::from_bytes(
                    &bytes[..end],
                    NeuralModelType::Hyper,
                    ModelEncoding::Plain,
                )
                .is_err(),
                "prefix of {end} bytes was accepted"
            );
        }
        assert!(
            NeuralModel::from_bytes(&bytes, NeuralModelType::Hyper, ModelEncoding::Plain,).is_ok()
        );
    }

    #[test]
    fn rejects_trailing_data_and_collection_limit_before_allocation() {
        let mut bytes = vae_model();
        bytes.push(0);
        assert_eq!(
            NeuralModel::from_bytes(&bytes, NeuralModelType::Vae, ModelEncoding::Plain,),
            Err(ModelError::TrailingData { bytes: 1 })
        );

        let mut reader = ModelReader::with_limits(
            &[0; 16],
            ModelEncoding::Plain,
            ModelLimits {
                max_collection_values: 2,
                ..ModelLimits::default()
            },
        )
        .unwrap();
        assert!(matches!(
            reader.read_u32s(3, "test values"),
            Err(ModelError::CollectionTooLarge { .. })
        ));
        assert_eq!(reader.position(), 0);
    }
}
