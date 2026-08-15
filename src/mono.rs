use core::fmt;

use crate::bwe::{BweSynthesis, BweSynthesisError};
use crate::core_side::{
    CoreBitstreamConfig, CoreBitstreamError, CoreSideInfo, MonoSideInfoDecoder, ParsedNeuralQc,
};
use crate::fd_shaping::{FdShapingError, FdSpectrumShaping};
use crate::mdct_synthesis::{MdctSynthesis, MdctSynthesisError};
use crate::model::{AVS3_FEATURE_DIMENSIONS, NeuralModel};
use crate::neural_qc::{NeuralQcError, NeuralSpectrumDecoder, NeuralSpectrumDiagnostics};
use crate::random::Avs3Random;
use crate::spectrum::{SpectrumReorder, SpectrumReorderError};
use crate::tns::{TnsSynthesis, TnsSynthesisError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MonoCoreDecodeError {
    InvalidOutputLength { expected: usize, actual: usize },
    InconsistentBweSideInformation,
    UnexpectedNeuralProfile,
    SideInformation(CoreBitstreamError),
    NeuralQc(NeuralQcError),
    SpectrumReorder(SpectrumReorderError),
    Bwe(BweSynthesisError),
    Tns(TnsSynthesisError),
    FdShaping(FdShapingError),
    MdctSynthesis(MdctSynthesisError),
}

impl fmt::Display for MonoCoreDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOutputLength { expected, actual } => write!(
                f,
                "mono synthesis output has {actual} samples; expected {expected}"
            ),
            Self::InconsistentBweSideInformation => {
                f.write_str("mono BWE configuration and side information are inconsistent")
            }
            Self::UnexpectedNeuralProfile => {
                f.write_str("parsed neural QC profile does not match the frame configuration")
            }
            Self::SideInformation(error) => error.fmt(f),
            Self::NeuralQc(error) => error.fmt(f),
            Self::SpectrumReorder(error) => error.fmt(f),
            Self::Bwe(error) => error.fmt(f),
            Self::Tns(error) => error.fmt(f),
            Self::FdShaping(error) => error.fmt(f),
            Self::MdctSynthesis(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for MonoCoreDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SideInformation(error) => Some(error),
            Self::NeuralQc(error) => Some(error),
            Self::SpectrumReorder(error) => Some(error),
            Self::Bwe(error) => Some(error),
            Self::Tns(error) => Some(error),
            Self::FdShaping(error) => Some(error),
            Self::MdctSynthesis(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CoreBitstreamError> for MonoCoreDecodeError {
    fn from(value: CoreBitstreamError) -> Self {
        Self::SideInformation(value)
    }
}

impl From<NeuralQcError> for MonoCoreDecodeError {
    fn from(value: NeuralQcError) -> Self {
        Self::NeuralQc(value)
    }
}

impl From<SpectrumReorderError> for MonoCoreDecodeError {
    fn from(value: SpectrumReorderError) -> Self {
        Self::SpectrumReorder(value)
    }
}

impl From<BweSynthesisError> for MonoCoreDecodeError {
    fn from(value: BweSynthesisError) -> Self {
        Self::Bwe(value)
    }
}

impl From<TnsSynthesisError> for MonoCoreDecodeError {
    fn from(value: TnsSynthesisError) -> Self {
        Self::Tns(value)
    }
}

impl From<FdShapingError> for MonoCoreDecodeError {
    fn from(value: FdShapingError) -> Self {
        Self::FdShaping(value)
    }
}

impl From<MdctSynthesisError> for MonoCoreDecodeError {
    fn from(value: MdctSynthesisError) -> Self {
        Self::MdctSynthesis(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonoCoreDiagnostics {
    core: CoreSideInfo,
    neural: NeuralSpectrumDiagnostics,
    entropy_bytes: usize,
    consumed_bits: usize,
    padding_bits: usize,
}

impl MonoCoreDiagnostics {
    pub fn core(self) -> CoreSideInfo {
        self.core
    }

    pub fn neural(self) -> NeuralSpectrumDiagnostics {
        self.neural
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

/// Complete mono/core payload-to-floating-PCM pipeline.
///
/// This composes side-bit parsing, Main/LC neural inverse QC, short-window
/// degrouping, BWE, TNS, inverse FD shaping and stateful IMDCT/overlap-add in
/// the same order as `Avs3MonoDec`. Decoder-local random and overlap state are
/// committed only after a frame succeeds; malformed frames therefore do not
/// advance temporal state. Construction performs all heap allocation.
pub struct MonoCoreDecoder<'model> {
    side_information: MonoSideInfoDecoder,
    neural: NeuralSpectrumDecoder<'model>,
    reorder: SpectrumReorder,
    bwe: BweSynthesis,
    tns: TnsSynthesis,
    fd_shaping: FdSpectrumShaping,
    mdct_synthesis: MdctSynthesis,
    random: Avs3Random,
    spectrum: [f32; AVS3_FEATURE_DIMENSIONS],
}

impl<'model> MonoCoreDecoder<'model> {
    pub fn new(model: &'model NeuralModel) -> Result<Self, MonoCoreDecodeError> {
        Ok(Self {
            side_information: MonoSideInfoDecoder::new(),
            neural: NeuralSpectrumDecoder::new(model)?,
            reorder: SpectrumReorder::new(),
            bwe: BweSynthesis::new(),
            tns: TnsSynthesis::new(),
            fd_shaping: FdSpectrumShaping::new(),
            mdct_synthesis: MdctSynthesis::new(),
            random: Avs3Random::new(),
            spectrum: [0.0; AVS3_FEATURE_DIMENSIONS],
        })
    }

    pub fn reset(&mut self) {
        self.random.reset();
        self.mdct_synthesis.reset();
        self.spectrum.fill(0.0);
    }

    pub fn random(&self) -> &Avs3Random {
        &self.random
    }

    pub fn random_mut(&mut self) -> &mut Avs3Random {
        &mut self.random
    }

    pub fn last_shaped_spectrum(&self) -> &[f32; AVS3_FEATURE_DIMENSIONS] {
        &self.spectrum
    }

    pub fn decode(
        &mut self,
        payload: &[u8],
        config: CoreBitstreamConfig,
        output: &mut [f32],
    ) -> Result<MonoCoreDiagnostics, MonoCoreDecodeError> {
        if output.len() != AVS3_FEATURE_DIMENSIONS {
            return Err(MonoCoreDecodeError::InvalidOutputLength {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: output.len(),
            });
        }

        let parsed = self.side_information.parse(payload, config)?;
        let core = parsed.core();
        let entropy_bytes = parsed.entropy_bytes();
        let consumed_bits = parsed.consumed_bits();
        let padding_bits = parsed.padding_bits();
        let mut frame_random = self.random.clone();

        let decoded = match parsed.neural_qc() {
            ParsedNeuralQc::Main(input) if config.nn_type() == crate::NnType::Main => {
                self.neural.decode_main(input, &mut frame_random)?
            }
            ParsedNeuralQc::LowComplexity(input)
                if config.nn_type() == crate::NnType::LowComplexity =>
            {
                self.neural
                    .decode_low_complexity(input, &mut frame_random)?
            }
            _ => return Err(MonoCoreDecodeError::UnexpectedNeuralProfile),
        };
        let neural_diagnostics = decoded.diagnostics();
        self.spectrum.copy_from_slice(decoded.spectrum());

        self.reorder
            .degroup(core.grouping(), core.transform_type(), &mut self.spectrum)?;
        match (config.bwe(), core.bwe()) {
            (Some(bwe_config), Some(bwe_side_info)) => self.bwe.apply(
                bwe_config,
                bwe_side_info,
                &mut self.spectrum,
                &mut frame_random,
            )?,
            (None, None) => {}
            _ => return Err(MonoCoreDecodeError::InconsistentBweSideInformation),
        }
        self.tns
            .apply(core.tns(), core.transform_type(), &mut self.spectrum)?;
        self.fd_shaping.apply(core.lsf(), &mut self.spectrum)?;
        self.mdct_synthesis
            .synthesize(&self.spectrum, core.transform_type(), output)?;

        self.random = frame_random;
        Ok(MonoCoreDiagnostics {
            core,
            neural: neural_diagnostics,
            entropy_bytes,
            consumed_bits,
            padding_bits,
        })
    }
}

impl MonoCoreDecoder<'static> {
    pub fn new_builtin() -> Result<Self, MonoCoreDecodeError> {
        let model = crate::builtin_neural_model().map_err(NeuralQcError::from)?;
        Self::new(model)
    }
}

impl fmt::Debug for MonoCoreDecoder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MonoCoreDecoder")
            .field("neural", &self.neural)
            .field("fd_shaping", &self.fd_shaping)
            .field("mdct_synthesis", &self.mdct_synthesis)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BitWriter, BweConfig, LsfCodebookMode, NnType, TransformType};

    fn reference_payload() -> (Vec<u8>, usize) {
        let mut writer = BitWriter::new();
        writer.write_bits(1, 2).unwrap();
        for (value, width) in [17_u64, 201, 66, 99, 45, 17, 3]
            .into_iter()
            .zip([8, 8, 7, 7, 6, 5, 5])
        {
            writer.write_bits(value, width).unwrap();
        }

        writer.write_bits(1, 1).unwrap();
        writer.write_bits(2, 3).unwrap();
        for (code, bits) in [(0, 3), (481, 10), (27_136, 15)] {
            writer.write_bits(code, bits).unwrap();
        }
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(7, 3).unwrap();
        for (code, bits) in [
            (2, 2),
            (3, 2),
            (27, 5),
            (16, 5),
            (129, 9),
            (1_035, 11),
            (13_314, 14),
            (10_499, 14),
        ] {
            writer.write_bits(code, bits).unwrap();
        }

        for envelope in [1, 127, 55, 64] {
            writer.write_bits(envelope, 7).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(1, 1).unwrap();

        writer.write_bits(1, 1).unwrap();
        for group in [0, 0, 0, 1, 1, 1, 1, 1] {
            writer.write_bits(group, 1).unwrap();
        }

        writer.write_bits(1, 1).unwrap();
        writer.write_bits(37, 7).unwrap();
        writer.write_bits(3, 3).unwrap();
        writer.write_bits(7, 3).unwrap();
        let context: [u8; 6] = [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7];
        let base: [u8; 26] = [
            0x7f, 0xfd, 0x51, 0xf6, 0xf2, 0x24, 0x34, 0xad, 0x04, 0xde, 0x75, 0xcd, 0x9d, 0x0c,
            0x76, 0xeb, 0xb3, 0x76, 0xaf, 0x47, 0xda, 0x43, 0x33, 0xf0, 0xd4, 0xeb,
        ];
        writer.write_bits(context.len() as u64, 8).unwrap();
        for byte in context.into_iter().chain(base) {
            writer.write_bits(u64::from(byte), 8).unwrap();
        }
        assert_eq!(writer.bit_len(), 464);
        let payload_bits = writer.bit_len() + 5;
        let mut payload = writer.into_bytes();
        payload.resize(payload_bits.div_ceil(8), 0);
        (payload, payload_bits)
    }

    #[test]
    fn complete_main_mono_pipeline_stays_close_to_c() {
        let (payload, payload_bits) = reference_payload();
        let config = CoreBitstreamConfig::new(
            NnType::Main,
            payload_bits,
            LsfCodebookMode::HighBitrate,
            BweConfig::for_mono_bitrate(64_000).unwrap(),
        )
        .unwrap();
        let mut decoder = MonoCoreDecoder::new_builtin().unwrap();
        let mut output = [0.0_f32; AVS3_FEATURE_DIMENSIONS];
        let diagnostics = decoder.decode(&payload, config, &mut output).unwrap();

        assert_eq!(diagnostics.core().transform_type(), TransformType::Short);
        assert_eq!(diagnostics.entropy_bytes(), 32);
        assert_eq!(diagnostics.consumed_bits(), 464);
        assert_eq!(diagnostics.padding_bits(), 5);
        assert_eq!(diagnostics.neural().feature_scale().to_bits(), 0x402c_59ba);
        assert!(output[..448].iter().all(|value| *value == 0.0));

        let positions = [0, 447, 448, 449, 575, 576, 700, 900, 1023];
        let expected_bits = [
            0x0000_0000,
            0x0000_0000,
            0x4246_8943,
            0xc341_b5ac,
            0xc5fc_c7eb,
            0x456d_9e5b,
            0x47e5_049c,
            0xc7a1_3978,
            0xc814_5ff6,
        ];
        for (position, bits) in positions.into_iter().zip(expected_bits) {
            let expected = f32::from_bits(bits);
            let error = (output[position] - expected).abs();
            let tolerance = 3.0e-5_f32 * expected.abs().max(1.0);
            assert!(
                error <= tolerance,
                "position {position}: Rust={} C={expected} error={error} tolerance={tolerance}",
                output[position]
            );
        }
        assert_eq!(decoder.random_mut().next_u31(), 431_530_126);
    }

    #[test]
    fn bad_output_length_does_not_advance_temporal_state() {
        let (payload, payload_bits) = reference_payload();
        let config = CoreBitstreamConfig::new(
            NnType::Main,
            payload_bits,
            LsfCodebookMode::HighBitrate,
            BweConfig::for_mono_bitrate(64_000).unwrap(),
        )
        .unwrap();
        let mut decoder = MonoCoreDecoder::new_builtin().unwrap();
        let original_random = decoder.random().clone();
        let mut output = [7.0_f32; 17];
        let error = decoder.decode(&payload, config, &mut output).unwrap_err();
        assert_eq!(
            error,
            MonoCoreDecodeError::InvalidOutputLength {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: 17,
            }
        );
        assert_eq!(decoder.random(), &original_random);
        assert_eq!(output, [7.0; 17]);
    }
}
