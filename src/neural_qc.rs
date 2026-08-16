use core::fmt;

use crate::builtin_neural_model;
use crate::cnn::{CnnError, ScalarCnnDecoder};
use crate::feature_scale_tables::{LOW_COMPLEXITY_SCALE_BITS, MAIN_AMPLIFIED_SCALE_BITS};
use crate::latent::{
    LatentError, LatentShape, channel_cdf_indexes_into, unflatten_from_entropy_coder_into,
};
use crate::model::{AVS3_FEATURE_DIMENSIONS, ModelError, NeuralCodecModel, NeuralModel};
use crate::random::{AVS3_RAND_MAX, Avs3Random};
use crate::range_coder::{RangeCoderError, RangeDecoder};

pub const MAX_QC_BITSTREAM_BYTES: usize = 1_024;
pub const AVS3_SHORT_BLOCKS: usize = 8;
pub const AVS3_NOISE_GROUPS: usize = 2;
pub const MAX_MAIN_SCALE_INDEX: u8 = (1 << 7) - 1;
pub const MAX_NOISE_FILLING_INDEX: u8 = (1 << 3) - 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeuralQcError {
    MissingContextModel,
    MissingContextScales,
    BitstreamTooLong {
        stream: &'static str,
        bytes: usize,
        limit: usize,
    },
    MainScaleIndexOutOfRange(u8),
    NoiseFillingIndexOutOfRange {
        group: usize,
        index: u8,
    },
    NoiseFillingLinesOutOfRange {
        lines: usize,
        limit: usize,
    },
    NoiseFillingDimensionsOutOfRange {
        dimensions: usize,
        available: usize,
    },
    ContextOutputShape {
        expected: LatentShape,
        actual: LatentShape,
    },
    LowComplexityOutputLength {
        expected: usize,
        actual: usize,
    },
    Model(ModelError),
    Latent(LatentError),
    RangeCoder(RangeCoderError),
    Cnn(CnnError),
}

impl fmt::Display for NeuralQcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingContextModel => {
                f.write_str("neural spectrum decoding requires a hyper-prior context model")
            }
            Self::MissingContextScales => {
                f.write_str("base neural model has no hyper-prior context-scale table")
            }
            Self::BitstreamTooLong {
                stream,
                bytes,
                limit,
            } => write!(
                f,
                "{stream} range bitstream has {bytes} bytes; limit is {limit}"
            ),
            Self::MainScaleIndexOutOfRange(index) => write!(
                f,
                "main-profile feature-scale index {index} exceeds {MAX_MAIN_SCALE_INDEX}"
            ),
            Self::NoiseFillingIndexOutOfRange { group, index } => write!(
                f,
                "noise-filling index {index} for group {group} exceeds {MAX_NOISE_FILLING_INDEX}"
            ),
            Self::NoiseFillingLinesOutOfRange { lines, limit } => write!(
                f,
                "noise filling requests {lines} spectrum lines; limit is {limit}"
            ),
            Self::NoiseFillingDimensionsOutOfRange {
                dimensions,
                available,
            } => write!(
                f,
                "noise filling reaches {dimensions} latent dimensions; only {available} are available"
            ),
            Self::ContextOutputShape { expected, actual } => write!(
                f,
                "context decoder output is {}x{}; base latent requires {}x{}",
                actual.dimensions(),
                actual.channels(),
                expected.dimensions(),
                expected.channels()
            ),
            Self::LowComplexityOutputLength { expected, actual } => write!(
                f,
                "low-complexity latent has {actual} values; expected {expected} spectrum lines"
            ),
            Self::Model(error) => error.fmt(f),
            Self::Latent(error) => error.fmt(f),
            Self::RangeCoder(error) => error.fmt(f),
            Self::Cnn(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for NeuralQcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Latent(error) => Some(error),
            Self::RangeCoder(error) => Some(error),
            Self::Cnn(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelError> for NeuralQcError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

impl From<LatentError> for NeuralQcError {
    fn from(value: LatentError) -> Self {
        Self::Latent(value)
    }
}

impl From<RangeCoderError> for NeuralQcError {
    fn from(value: RangeCoderError) -> Self {
        Self::RangeCoder(value)
    }
}

impl From<CnnError> for NeuralQcError {
    fn from(value: CnnError) -> Self {
        Self::Cnn(value)
    }
}

/// Borrowed entropy-coded payloads for one channel of one AVS3 frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeuralBitstreams<'a> {
    context: &'a [u8],
    base: &'a [u8],
}

impl<'a> NeuralBitstreams<'a> {
    pub fn new(context: &'a [u8], base: &'a [u8]) -> Result<Self, NeuralQcError> {
        check_bitstream_len("context", context)?;
        check_bitstream_len("base", base)?;
        Ok(Self { context, base })
    }

    pub fn context(self) -> &'a [u8] {
        self.context
    }

    pub fn base(self) -> &'a [u8] {
        self.base
    }
}

fn check_bitstream_len(stream: &'static str, bytes: &[u8]) -> Result<(), NeuralQcError> {
    if bytes.len() > MAX_QC_BITSTREAM_BYTES {
        Err(NeuralQcError::BitstreamTooLong {
            stream,
            bytes: bytes.len(),
            limit: MAX_QC_BITSTREAM_BYTES,
        })
    } else {
        Ok(())
    }
}

/// Meaning of an entry in the C decoder's eight-element `groupIndicator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseGroup {
    Transient,
    Other,
}

/// Validated noise-filling side information for a long/single-group frame or
/// a two-group short-window frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoiseFilling {
    num_lines: usize,
    group_count: usize,
    group_indicator: [NoiseGroup; AVS3_SHORT_BLOCKS],
    quantized_indexes: [u8; AVS3_NOISE_GROUPS],
}

impl NoiseFilling {
    pub fn single(num_lines: usize, quantized_index: u8) -> Result<Self, NeuralQcError> {
        Self::validate_lines(num_lines)?;
        Self::validate_index(0, quantized_index)?;
        Ok(Self {
            num_lines,
            group_count: 1,
            group_indicator: [NoiseGroup::Transient; AVS3_SHORT_BLOCKS],
            quantized_indexes: [quantized_index, 0],
        })
    }

    pub fn two_groups(
        num_lines: usize,
        group_indicator: [NoiseGroup; AVS3_SHORT_BLOCKS],
        quantized_indexes: [u8; AVS3_NOISE_GROUPS],
    ) -> Result<Self, NeuralQcError> {
        Self::validate_lines(num_lines)?;
        for (group, &index) in quantized_indexes.iter().enumerate() {
            Self::validate_index(group, index)?;
        }
        Ok(Self {
            num_lines,
            group_count: AVS3_NOISE_GROUPS,
            group_indicator,
            quantized_indexes,
        })
    }

    pub fn num_lines(self) -> usize {
        self.num_lines
    }

    pub fn group_count(self) -> usize {
        self.group_count
    }

    pub fn group_indicator(self) -> [NoiseGroup; AVS3_SHORT_BLOCKS] {
        self.group_indicator
    }

    pub fn quantized_indexes(self) -> [u8; AVS3_NOISE_GROUPS] {
        self.quantized_indexes
    }

    fn validate_lines(num_lines: usize) -> Result<(), NeuralQcError> {
        if num_lines > AVS3_FEATURE_DIMENSIONS {
            Err(NeuralQcError::NoiseFillingLinesOutOfRange {
                lines: num_lines,
                limit: AVS3_FEATURE_DIMENSIONS,
            })
        } else {
            Ok(())
        }
    }

    fn validate_index(group: usize, index: u8) -> Result<(), NeuralQcError> {
        if index > MAX_NOISE_FILLING_INDEX {
            Err(NeuralQcError::NoiseFillingIndexOutOfRange { group, index })
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MainNeuralQc<'a> {
    bitstreams: NeuralBitstreams<'a>,
    noise_filling: NoiseFilling,
    feature_amplified: bool,
    scale_index: u8,
}

impl<'a> MainNeuralQc<'a> {
    pub fn new(
        bitstreams: NeuralBitstreams<'a>,
        noise_filling: NoiseFilling,
        feature_amplified: bool,
        scale_index: u8,
    ) -> Result<Self, NeuralQcError> {
        if scale_index > MAX_MAIN_SCALE_INDEX {
            return Err(NeuralQcError::MainScaleIndexOutOfRange(scale_index));
        }
        Ok(Self {
            bitstreams,
            noise_filling,
            feature_amplified,
            scale_index,
        })
    }

    pub fn bitstreams(self) -> NeuralBitstreams<'a> {
        self.bitstreams
    }

    pub fn noise_filling(self) -> NoiseFilling {
        self.noise_filling
    }

    pub fn feature_amplified(self) -> bool {
        self.feature_amplified
    }

    pub fn scale_index(self) -> u8 {
        self.scale_index
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LowComplexityNeuralQc<'a> {
    bitstreams: NeuralBitstreams<'a>,
    noise_filling: NoiseFilling,
    scale_index: u8,
}

impl<'a> LowComplexityNeuralQc<'a> {
    pub fn new(
        bitstreams: NeuralBitstreams<'a>,
        noise_filling: NoiseFilling,
        scale_index: u8,
    ) -> Self {
        Self {
            bitstreams,
            noise_filling,
            scale_index,
        }
    }

    pub fn bitstreams(self) -> NeuralBitstreams<'a> {
        self.bitstreams
    }

    pub fn noise_filling(self) -> NoiseFilling {
        self.noise_filling
    }

    pub fn scale_index(self) -> u8 {
        self.scale_index
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuralSpectrumDiagnostics {
    feature_scale: f32,
    noise_parameters: [f32; AVS3_NOISE_GROUPS],
    noise_group_count: usize,
    context_bytes_consumed: usize,
    base_bytes_consumed: usize,
}

impl NeuralSpectrumDiagnostics {
    pub fn feature_scale(self) -> f32 {
        self.feature_scale
    }

    pub fn noise_parameters(self) -> [f32; AVS3_NOISE_GROUPS] {
        self.noise_parameters
    }

    pub fn noise_group_count(self) -> usize {
        self.noise_group_count
    }

    pub fn context_bytes_consumed(self) -> usize {
        self.context_bytes_consumed
    }

    pub fn base_bytes_consumed(self) -> usize {
        self.base_bytes_consumed
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodedNeuralSpectrum<'decoder> {
    spectrum: &'decoder [f32],
    diagnostics: NeuralSpectrumDiagnostics,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum PreparedNeuralSpectrum<'a> {
    Main {
        input: MainNeuralQc<'a>,
        context_bytes_consumed: usize,
        base_bytes_consumed: usize,
    },
    LowComplexity {
        input: LowComplexityNeuralQc<'a>,
        context_bytes_consumed: usize,
        base_bytes_consumed: usize,
    },
}

impl<'decoder> DecodedNeuralSpectrum<'decoder> {
    pub fn spectrum(&self) -> &'decoder [f32] {
        self.spectrum
    }

    pub fn diagnostics(&self) -> NeuralSpectrumDiagnostics {
        self.diagnostics
    }

    pub fn into_parts(self) -> (&'decoder [f32], NeuralSpectrumDiagnostics) {
        (self.spectrum, self.diagnostics)
    }
}

/// Allocation-stable AVS3 hyper-prior entropy and neural spectrum decoder.
///
/// All latent/index/output buffers are allocated by [`Self::new`]. Calls to
/// `decode_main` and `decode_low_complexity` perform no heap allocation. One
/// [`Avs3Random`] should be shared by the top-level decoder across channels so
/// its ordering continues to match the C reference's former global `rand()`.
#[derive(Debug)]
pub struct NeuralSpectrumDecoder<'model> {
    base: &'model NeuralCodecModel,
    context: &'model NeuralCodecModel,
    context_decoder: ScalarCnnDecoder<'model>,
    base_decoder: ScalarCnnDecoder<'model>,
    context_cdf_indexes: Vec<u16>,
    context_flattened: Vec<i32>,
    context_quantized: Vec<i32>,
    context_dequantized: Vec<f32>,
    base_cdf_indexes: Vec<u16>,
    base_flattened: Vec<i32>,
    base_quantized: Vec<i32>,
    base_dequantized: Vec<f32>,
    low_complexity_output: Vec<f32>,
}

impl<'model> NeuralSpectrumDecoder<'model> {
    pub fn new(model: &'model NeuralModel) -> Result<Self, NeuralQcError> {
        let base = model.base();
        let context = model.context().ok_or(NeuralQcError::MissingContextModel)?;
        if base.context_scales().is_none() {
            return Err(NeuralQcError::MissingContextScales);
        }
        if context.decoder().output_shape() != base.latent_shape() {
            return Err(NeuralQcError::ContextOutputShape {
                expected: base.latent_shape(),
                actual: context.decoder().output_shape(),
            });
        }
        let context_shape = context.latent_shape();
        let base_shape = base.latent_shape();
        let mut context_cdf_indexes = vec![0_u16; context_shape.len()];
        channel_cdf_indexes_into(context_shape, &mut context_cdf_indexes)?;

        Ok(Self {
            base,
            context,
            context_decoder: ScalarCnnDecoder::new(context.decoder())?,
            base_decoder: ScalarCnnDecoder::new(base.decoder())?,
            context_cdf_indexes,
            context_flattened: vec![0_i32; context_shape.len()],
            context_quantized: vec![0_i32; context_shape.len()],
            context_dequantized: vec![0.0_f32; context_shape.len()],
            base_cdf_indexes: vec![0_u16; base_shape.len()],
            base_flattened: vec![0_i32; base_shape.len()],
            base_quantized: vec![0_i32; base_shape.len()],
            base_dequantized: vec![0.0_f32; base_shape.len()],
            low_complexity_output: vec![0.0_f32; AVS3_FEATURE_DIMENSIONS],
        })
    }

    pub fn base_model(&self) -> &'model NeuralCodecModel {
        self.base
    }

    pub fn context_model(&self) -> &'model NeuralCodecModel {
        self.context
    }

    pub fn decode_main<'decoder>(
        &'decoder mut self,
        input: MainNeuralQc<'_>,
        random: &mut Avs3Random,
    ) -> Result<DecodedNeuralSpectrum<'decoder>, NeuralQcError> {
        let prepared = self.prepare_main(input)?;
        self.finish_prepared(prepared, random)
    }

    pub(crate) fn prepare_main<'a>(
        &mut self,
        input: MainNeuralQc<'a>,
    ) -> Result<PreparedNeuralSpectrum<'a>, NeuralQcError> {
        let (context_bytes_consumed, base_bytes_consumed) =
            self.decode_latents(input.bitstreams)?;
        Ok(PreparedNeuralSpectrum::Main {
            input,
            context_bytes_consumed,
            base_bytes_consumed,
        })
    }

    pub(crate) fn prepare_low_complexity<'a>(
        &mut self,
        input: LowComplexityNeuralQc<'a>,
    ) -> Result<PreparedNeuralSpectrum<'a>, NeuralQcError> {
        if self.base.latent_shape().len() != AVS3_FEATURE_DIMENSIONS {
            return Err(NeuralQcError::LowComplexityOutputLength {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: self.base.latent_shape().len(),
            });
        }
        let (context_bytes_consumed, base_bytes_consumed) =
            self.decode_latents(input.bitstreams)?;
        Ok(PreparedNeuralSpectrum::LowComplexity {
            input,
            context_bytes_consumed,
            base_bytes_consumed,
        })
    }

    pub(crate) fn finish_prepared<'decoder>(
        &'decoder mut self,
        prepared: PreparedNeuralSpectrum<'_>,
        random: &mut Avs3Random,
    ) -> Result<DecodedNeuralSpectrum<'decoder>, NeuralQcError> {
        match prepared {
            PreparedNeuralSpectrum::Main {
                input,
                context_bytes_consumed,
                base_bytes_consumed,
            } => self.finish_main(input, context_bytes_consumed, base_bytes_consumed, random),
            PreparedNeuralSpectrum::LowComplexity {
                input,
                context_bytes_consumed,
                base_bytes_consumed,
            } => self.finish_low_complexity(
                input,
                context_bytes_consumed,
                base_bytes_consumed,
                random,
            ),
        }
    }

    /// Count the PRNG values consumed by noise filling without mutating the
    /// latent buffer. This lets the multichannel core assign each channel an
    /// independent random cursor before finishing channels in parallel.
    pub(crate) fn noise_random_draws(
        &self,
        prepared: PreparedNeuralSpectrum<'_>,
    ) -> Result<usize, NeuralQcError> {
        let (noise_filling, noise_dimensions) = match prepared {
            PreparedNeuralSpectrum::Main { input, .. } => {
                let mut dimensions = input.noise_filling.num_lines;
                for layer in self.base.decoder().layers() {
                    dimensions /= layer.stride();
                }
                (input.noise_filling, dimensions)
            }
            PreparedNeuralSpectrum::LowComplexity { input, .. } => (
                input.noise_filling,
                input.noise_filling.num_lines / self.base.latent_shape().channels(),
            ),
        };
        let shape = self.base.latent_shape();
        if noise_dimensions > shape.dimensions() {
            return Err(NeuralQcError::NoiseFillingDimensionsOutOfRange {
                dimensions: noise_dimensions,
                available: shape.dimensions(),
            });
        }
        let ranges = noise_ranges(shape, noise_dimensions, noise_filling);
        let medians = self.base.quantizer().quantile_medians();
        let mut draws = 0;
        for range in ranges[..noise_filling.group_count].iter() {
            for dimension in range.0..range.1 {
                for (channel, &median) in medians.iter().enumerate() {
                    let index = dimension + channel * shape.dimensions();
                    if self.base_dequantized[index] == median {
                        draws += 1;
                    }
                }
            }
        }
        Ok(draws)
    }

    fn finish_main<'decoder>(
        &'decoder mut self,
        input: MainNeuralQc<'_>,
        context_bytes_consumed: usize,
        base_bytes_consumed: usize,
        random: &mut Avs3Random,
    ) -> Result<DecodedNeuralSpectrum<'decoder>, NeuralQcError> {
        let noise_filling = input.noise_filling;
        let mut noise_dimensions = noise_filling.num_lines;
        for layer in self.base.decoder().layers() {
            noise_dimensions /= layer.stride();
        }
        let noise_parameters = self.apply_noise_filling(noise_filling, noise_dimensions, random)?;
        let feature_scale = main_feature_scale(input.feature_amplified, input.scale_index);
        for value in &mut self.base_dequantized {
            *value /= feature_scale;
        }

        let diagnostics = NeuralSpectrumDiagnostics {
            feature_scale,
            noise_parameters,
            noise_group_count: noise_filling.group_count,
            context_bytes_consumed,
            base_bytes_consumed,
        };
        let spectrum = self.base_decoder.decode(&self.base_dequantized)?;
        Ok(DecodedNeuralSpectrum {
            spectrum,
            diagnostics,
        })
    }

    pub fn decode_low_complexity<'decoder>(
        &'decoder mut self,
        input: LowComplexityNeuralQc<'_>,
        random: &mut Avs3Random,
    ) -> Result<DecodedNeuralSpectrum<'decoder>, NeuralQcError> {
        let prepared = self.prepare_low_complexity(input)?;
        self.finish_prepared(prepared, random)
    }

    fn finish_low_complexity<'decoder>(
        &'decoder mut self,
        input: LowComplexityNeuralQc<'_>,
        context_bytes_consumed: usize,
        base_bytes_consumed: usize,
        random: &mut Avs3Random,
    ) -> Result<DecodedNeuralSpectrum<'decoder>, NeuralQcError> {
        let noise_filling = input.noise_filling;
        let noise_dimensions = noise_filling.num_lines / self.base.latent_shape().channels();
        let noise_parameters = self.apply_noise_filling(noise_filling, noise_dimensions, random)?;
        let feature_scale = low_complexity_feature_scale(input.scale_index);

        let shape = self.base.latent_shape();
        for dimension in 0..shape.dimensions() {
            for channel in 0..shape.channels() {
                self.low_complexity_output[dimension * shape.channels() + channel] =
                    self.base_dequantized[dimension + channel * shape.dimensions()] / feature_scale;
            }
        }

        Ok(DecodedNeuralSpectrum {
            spectrum: &self.low_complexity_output,
            diagnostics: NeuralSpectrumDiagnostics {
                feature_scale,
                noise_parameters,
                noise_group_count: noise_filling.group_count,
                context_bytes_consumed,
                base_bytes_consumed,
            },
        })
    }

    fn decode_latents(
        &mut self,
        bitstreams: NeuralBitstreams<'_>,
    ) -> Result<(usize, usize), NeuralQcError> {
        let context_shape = self.context.latent_shape();
        let mut context_range = RangeDecoder::new(bitstreams.context);
        context_range.decode_into(
            self.context.range_coder(),
            &self.context_cdf_indexes,
            &mut self.context_flattened,
        )?;
        let context_bytes_consumed = context_range.bytes_consumed();
        unflatten_from_entropy_coder_into(
            context_shape,
            &self.context_flattened,
            &mut self.context_quantized,
        )?;
        self.context.quantizer().dequantize_into(
            context_shape,
            &self.context_quantized,
            &mut self.context_dequantized,
        )?;

        let context_output = self.context_decoder.decode(&self.context_dequantized)?;
        self.base
            .context_scales()
            .ok_or(NeuralQcError::MissingContextScales)?
            .cdf_indexes_into(
                self.base.latent_shape(),
                context_output,
                &mut self.base_cdf_indexes,
            )?;

        let mut base_range = RangeDecoder::new(bitstreams.base);
        base_range.decode_into(
            self.base.range_coder(),
            &self.base_cdf_indexes,
            &mut self.base_flattened,
        )?;
        let base_bytes_consumed = base_range.bytes_consumed();
        let base_shape = self.base.latent_shape();
        unflatten_from_entropy_coder_into(
            base_shape,
            &self.base_flattened,
            &mut self.base_quantized,
        )?;
        self.base.quantizer().dequantize_into(
            base_shape,
            &self.base_quantized,
            &mut self.base_dequantized,
        )?;
        Ok((context_bytes_consumed, base_bytes_consumed))
    }

    fn apply_noise_filling(
        &mut self,
        noise_filling: NoiseFilling,
        noise_dimensions: usize,
        random: &mut Avs3Random,
    ) -> Result<[f32; AVS3_NOISE_GROUPS], NeuralQcError> {
        let shape = self.base.latent_shape();
        if noise_dimensions > shape.dimensions() {
            return Err(NeuralQcError::NoiseFillingDimensionsOutOfRange {
                dimensions: noise_dimensions,
                available: shape.dimensions(),
            });
        }

        let ranges = noise_ranges(shape, noise_dimensions, noise_filling);
        let mut noise_parameters = [0.0_f32; AVS3_NOISE_GROUPS];
        let medians = self.base.quantizer().quantile_medians();
        for group in 0..noise_filling.group_count {
            let noise_parameter = noise_filling.quantized_indexes[group] as f32 / 23.34_f32;
            noise_parameters[group] = noise_parameter;
            for dimension in ranges[group].0..ranges[group].1 {
                for (channel, &median) in medians.iter().enumerate() {
                    let index = dimension + channel * shape.dimensions();
                    if self.base_dequantized[index] == median {
                        let mut noise = random.next_u31() as f32 / AVS3_RAND_MAX as f32;
                        noise = noise * 2.0_f32 - 1.0_f32;
                        noise *= noise_parameter;
                        self.base_dequantized[index] += noise;
                    }
                }
            }
        }
        Ok(noise_parameters)
    }
}

impl NeuralSpectrumDecoder<'static> {
    pub fn new_builtin() -> Result<Self, NeuralQcError> {
        Self::new(builtin_neural_model()?)
    }
}

fn noise_ranges(
    shape: LatentShape,
    noise_dimensions: usize,
    noise_filling: NoiseFilling,
) -> [(usize, usize); AVS3_NOISE_GROUPS] {
    if noise_filling.group_count == 1 {
        return [(0, noise_dimensions), (0, noise_dimensions)];
    }

    let transient_blocks = noise_filling
        .group_indicator
        .iter()
        .filter(|&&group| group == NoiseGroup::Transient)
        .count();
    let other_blocks = AVS3_SHORT_BLOCKS - transient_blocks;
    // Preserve C's f32 divisions and its truncating cast to `short`.
    let first_end =
        (noise_dimensions as f32 / AVS3_SHORT_BLOCKS as f32 * transient_blocks as f32) as usize;
    let second_start = shape.dimensions() / AVS3_SHORT_BLOCKS * transient_blocks;
    let second_len =
        (noise_dimensions as f32 / AVS3_SHORT_BLOCKS as f32 * other_blocks as f32) as usize;
    [(0, first_end), (second_start, second_start + second_len)]
}

fn main_feature_scale(amplified: bool, scale_index: u8) -> f32 {
    let mut scale = if amplified {
        f32::from_bits(MAIN_AMPLIFIED_SCALE_BITS[usize::from(scale_index)])
    } else {
        scale_index as f32 / 127.0_f32
    };
    if scale == 0.0_f32 {
        scale = 1.0_f32;
    }
    scale
}

fn low_complexity_feature_scale(scale_index: u8) -> f32 {
    let mut scale = f32::from_bits(LOW_COMPLEXITY_SCALE_BITS[usize::from(scale_index)]);
    if scale == 0.0_f32 {
        scale = 1.0_f32;
    }
    scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature_fingerprint(values: &[f32]) -> u64 {
        values
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
                (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3)
            })
    }

    #[test]
    fn side_information_constructors_enforce_wire_widths() {
        let oversized = [0_u8; MAX_QC_BITSTREAM_BYTES + 1];
        assert_eq!(
            NeuralBitstreams::new(&oversized, &[]).unwrap_err(),
            NeuralQcError::BitstreamTooLong {
                stream: "context",
                bytes: MAX_QC_BITSTREAM_BYTES + 1,
                limit: MAX_QC_BITSTREAM_BYTES,
            }
        );

        let streams = NeuralBitstreams::new(&[], &[]).unwrap();
        let noise = NoiseFilling::single(1_024, 7).unwrap();
        assert_eq!(
            MainNeuralQc::new(streams, noise, false, 128).unwrap_err(),
            NeuralQcError::MainScaleIndexOutOfRange(128)
        );
        assert_eq!(
            NoiseFilling::single(1_025, 0).unwrap_err(),
            NeuralQcError::NoiseFillingLinesOutOfRange {
                lines: 1_025,
                limit: 1_024,
            }
        );
        assert_eq!(
            NoiseFilling::single(0, 8).unwrap_err(),
            NeuralQcError::NoiseFillingIndexOutOfRange { group: 0, index: 8 }
        );
    }

    #[test]
    fn feature_scale_preserves_c_operation_order() {
        assert_eq!(main_feature_scale(false, 0).to_bits(), 1.0_f32.to_bits());
        assert_eq!(main_feature_scale(false, 127).to_bits(), 1.0_f32.to_bits());
        assert_eq!(main_feature_scale(false, 37).to_bits(), 0x3e95_2a55);
        assert_eq!(main_feature_scale(true, 37).to_bits(), 0x402c_59ba);
        assert_eq!(low_complexity_feature_scale(0).to_bits(), 0x322b_cc77);
        assert_eq!(
            low_complexity_feature_scale(255).to_bits(),
            1.0_f32.to_bits()
        );

        let fingerprint = |values: &[u32]| {
            values
                .iter()
                .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
                    (hash ^ u64::from(*value)).wrapping_mul(0x100_0000_01b3)
                })
        };
        assert_eq!(
            fingerprint(&MAIN_AMPLIFIED_SCALE_BITS),
            0xc9ca_1b97_65cf_a043
        );
        assert_eq!(
            fingerprint(&LOW_COMPLEXITY_SCALE_BITS),
            0x784f_7606_18f0_1fd4
        );
    }

    #[test]
    fn short_window_ranges_match_c_grouping_math() {
        let shape = LatentShape::new(64, 16).unwrap();
        let indicator = [
            NoiseGroup::Transient,
            NoiseGroup::Transient,
            NoiseGroup::Transient,
            NoiseGroup::Other,
            NoiseGroup::Other,
            NoiseGroup::Other,
            NoiseGroup::Other,
            NoiseGroup::Other,
        ];
        let filling = NoiseFilling::two_groups(45, indicator, [3, 7]).unwrap();
        assert_eq!(noise_ranges(shape, 45, filling), [(0, 16), (24, 52)]);
    }

    #[test]
    fn builtin_decoder_runs_main_and_lc_without_reallocation_contract_changes() {
        let streams = NeuralBitstreams::new(&[], &[]).unwrap();
        let noise = NoiseFilling::single(0, 0).unwrap();
        let mut decoder = NeuralSpectrumDecoder::new_builtin().unwrap();
        let mut random = Avs3Random::new();

        let main = decoder
            .decode_main(
                MainNeuralQc::new(streams, noise, false, 0).unwrap(),
                &mut random,
            )
            .unwrap();
        assert_eq!(main.spectrum().len(), AVS3_FEATURE_DIMENSIONS);
        assert!(main.spectrum().iter().all(|value| value.is_finite()));

        let lc = decoder
            .decode_low_complexity(LowComplexityNeuralQc::new(streams, noise, 255), &mut random)
            .unwrap();
        assert_eq!(lc.spectrum().len(), AVS3_FEATURE_DIMENSIONS);
        assert!(lc.spectrum().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn prepared_and_one_shot_paths_are_bit_exact() {
        let context = [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7];
        let base = [
            0x7f, 0xfd, 0x51, 0xf6, 0xf2, 0x24, 0x34, 0xad, 0x04, 0xde, 0x75, 0xcd, 0x9d, 0x0c,
            0x76, 0xeb, 0xb3, 0x76, 0xaf, 0x47, 0xda, 0x43, 0x33, 0xf0, 0xd4, 0xeb,
        ];
        let streams = NeuralBitstreams::new(&context, &base).unwrap();
        let noise = NoiseFilling::single(720, 5).unwrap();

        let main_input = MainNeuralQc::new(streams, noise, true, 37).unwrap();
        let mut one_shot = NeuralSpectrumDecoder::new_builtin().unwrap();
        let mut one_shot_random = Avs3Random::new();
        let decoded = one_shot
            .decode_main(main_input, &mut one_shot_random)
            .unwrap();
        let expected_spectrum = decoded.spectrum().to_vec();
        let expected_diagnostics = decoded.diagnostics();
        let expected_next_random = one_shot_random.next_u31();

        let mut staged = NeuralSpectrumDecoder::new_builtin().unwrap();
        let mut staged_random = Avs3Random::new();
        let prepared = staged.prepare_main(main_input).unwrap();
        let decoded = staged
            .finish_prepared(prepared, &mut staged_random)
            .unwrap();
        assert_eq!(decoded.spectrum(), expected_spectrum);
        assert_eq!(decoded.diagnostics(), expected_diagnostics);
        assert_eq!(staged_random.next_u31(), expected_next_random);

        let lc_input = LowComplexityNeuralQc::new(streams, noise, 91);
        let mut one_shot = NeuralSpectrumDecoder::new_builtin().unwrap();
        let mut one_shot_random = Avs3Random::new();
        let decoded = one_shot
            .decode_low_complexity(lc_input, &mut one_shot_random)
            .unwrap();
        let expected_spectrum = decoded.spectrum().to_vec();
        let expected_diagnostics = decoded.diagnostics();
        let expected_next_random = one_shot_random.next_u31();

        let mut staged = NeuralSpectrumDecoder::new_builtin().unwrap();
        let mut staged_random = Avs3Random::new();
        let prepared = staged.prepare_low_complexity(lc_input).unwrap();
        let decoded = staged
            .finish_prepared(prepared, &mut staged_random)
            .unwrap();
        assert_eq!(decoded.spectrum(), expected_spectrum);
        assert_eq!(decoded.diagnostics(), expected_diagnostics);
        assert_eq!(staged_random.next_u31(), expected_next_random);
    }

    #[test]
    fn complete_main_and_lc_paths_match_c_reference_vectors() {
        // Encoded by RangeEncodeProcess with every context/base quantized
        // latent set to zero, using the built-in model. The spectrum values
        // come from MdctDequantDecodeHyper{,Lc} on glibc at -O1.
        let context = [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7];
        let base = [
            0x7f, 0xfd, 0x51, 0xf6, 0xf2, 0x24, 0x34, 0xad, 0x04, 0xde, 0x75, 0xcd, 0x9d, 0x0c,
            0x76, 0xeb, 0xb3, 0x76, 0xaf, 0x47, 0xda, 0x43, 0x33, 0xf0, 0xd4, 0xeb,
        ];
        let streams = NeuralBitstreams::new(&context, &base).unwrap();
        let indicator = [
            NoiseGroup::Transient,
            NoiseGroup::Transient,
            NoiseGroup::Transient,
            NoiseGroup::Other,
            NoiseGroup::Other,
            NoiseGroup::Other,
            NoiseGroup::Other,
            NoiseGroup::Other,
        ];
        let noise = NoiseFilling::two_groups(720, indicator, [3, 7]).unwrap();
        let positions = [0, 1, 15, 16, 255, 511, 719, 720, 1023];

        let mut decoder = NeuralSpectrumDecoder::new_builtin().unwrap();
        let mut random = Avs3Random::new();
        let main = decoder
            .decode_main(
                MainNeuralQc::new(streams, noise, true, 37).unwrap(),
                &mut random,
            )
            .unwrap();
        assert_eq!(feature_fingerprint(main.spectrum()), 0x015b_ee23_8ed6_728b);
        assert_eq!(main.diagnostics().feature_scale().to_bits(), 0x402c_59ba);
        assert_eq!(
            main.diagnostics().noise_parameters().map(f32::to_bits),
            [0x3e03_9e9a, 0x3e99_8e5e]
        );
        assert_eq!(
            positions.map(|index| main.spectrum()[index].to_bits()),
            [
                0xc0fa_fd0b,
                0xc146_9233,
                0x3fd8_1188,
                0xc018_c098,
                0xc15b_80d0,
                0xc093_07db,
                0xbf0d_0454,
                0xc1e2_f7af,
                0x3eb6_f4bc,
            ]
        );
        assert_eq!(random.next_u31(), 1_738_110_294);

        let mut random = Avs3Random::new();
        let lc = decoder
            .decode_low_complexity(LowComplexityNeuralQc::new(streams, noise, 91), &mut random)
            .unwrap();
        assert_eq!(feature_fingerprint(lc.spectrum()), 0x9e96_9a84_2e47_42b1);
        assert_eq!(lc.diagnostics().feature_scale().to_bits(), 0x36f0_3e57);
        assert_eq!(
            positions.map(|index| lc.spectrum()[index].to_bits()),
            [
                0x463e_d90d,
                0xc56d_0204,
                0x4669_7d13,
                0x4598_4545,
                0x458e_ac4c,
                0x4688_140d,
                0xc531_24c6,
                0x46fb_8a13,
                0x0000_0000,
            ]
        );
        assert_eq!(random.next_u31(), 1_738_110_294);
    }
}
