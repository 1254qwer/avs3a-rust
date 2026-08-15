use core::fmt;

use crate::bwe::{BweSynthesis, BweSynthesisError};
use crate::core_side::{BweConfig, CoreSideInfo, ParsedNeuralQc};
use crate::fd_shaping::{FdShapingError, FdSpectrumShaping};
use crate::header::{FrameHeader, NnType};
use crate::mcr::{McrError, McrSideInfo, McrSynthesis};
use crate::mdct_synthesis::{MdctSynthesis, MdctSynthesisError};
use crate::model::{AVS3_FEATURE_DIMENSIONS, NeuralModel};
use crate::neural_qc::{NeuralQcError, NeuralSpectrumDecoder, NeuralSpectrumDiagnostics};
use crate::random::Avs3Random;
use crate::spectrum::{SpectrumReorder, SpectrumReorderError};
use crate::stereo::{
    STEREO_CHANNELS, StereoError, StereoSideInfo, StereoSideInfoDecoder, inverse_mid_side,
};
use crate::tns::{TnsSynthesis, TnsSynthesisError};

pub const STEREO_FRAME_SAMPLES: usize = AVS3_FEATURE_DIMENSIONS * STEREO_CHANNELS;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StereoCoreDecodeError {
    InvalidOutputLength { expected: usize, actual: usize },
    UnexpectedNeuralProfile { channel: usize },
    InconsistentBweSideInformation { channel: usize },
    MissingIld,
    Mcr(McrError),
    SideInformation(StereoError),
    NeuralQc(NeuralQcError),
    SpectrumReorder(SpectrumReorderError),
    Bwe(BweSynthesisError),
    Tns(TnsSynthesisError),
    FdShaping(FdShapingError),
    MdctSynthesis(MdctSynthesisError),
}

impl fmt::Display for StereoCoreDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOutputLength { expected, actual } => write!(
                f,
                "stereo synthesis output has {actual} samples; expected {expected}"
            ),
            Self::UnexpectedNeuralProfile { channel } => write!(
                f,
                "parsed neural QC profile does not match stereo channel {channel} configuration"
            ),
            Self::InconsistentBweSideInformation { channel } => write!(
                f,
                "stereo channel {channel} BWE configuration and side information are inconsistent"
            ),
            Self::MissingIld => f.write_str("MS stereo frame is missing its ILD parameter"),
            Self::Mcr(error) => error.fmt(f),
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

impl std::error::Error for StereoCoreDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SideInformation(error) => Some(error),
            Self::NeuralQc(error) => Some(error),
            Self::SpectrumReorder(error) => Some(error),
            Self::Bwe(error) => Some(error),
            Self::Tns(error) => Some(error),
            Self::FdShaping(error) => Some(error),
            Self::MdctSynthesis(error) => Some(error),
            Self::Mcr(error) => Some(error),
            _ => None,
        }
    }
}

impl From<StereoError> for StereoCoreDecodeError {
    fn from(value: StereoError) -> Self {
        Self::SideInformation(value)
    }
}

impl From<NeuralQcError> for StereoCoreDecodeError {
    fn from(value: NeuralQcError) -> Self {
        Self::NeuralQc(value)
    }
}

impl From<SpectrumReorderError> for StereoCoreDecodeError {
    fn from(value: SpectrumReorderError) -> Self {
        Self::SpectrumReorder(value)
    }
}

impl From<BweSynthesisError> for StereoCoreDecodeError {
    fn from(value: BweSynthesisError) -> Self {
        Self::Bwe(value)
    }
}

impl From<TnsSynthesisError> for StereoCoreDecodeError {
    fn from(value: TnsSynthesisError) -> Self {
        Self::Tns(value)
    }
}

impl From<FdShapingError> for StereoCoreDecodeError {
    fn from(value: FdShapingError) -> Self {
        Self::FdShaping(value)
    }
}

impl From<MdctSynthesisError> for StereoCoreDecodeError {
    fn from(value: MdctSynthesisError) -> Self {
        Self::MdctSynthesis(value)
    }
}

impl From<McrError> for StereoCoreDecodeError {
    fn from(value: McrError) -> Self {
        Self::Mcr(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoCoreDiagnostics {
    cores: [CoreSideInfo; STEREO_CHANNELS],
    stereo: StereoSideInfo,
    neural: [NeuralSpectrumDiagnostics; STEREO_CHANNELS],
    entropy_bytes: [usize; STEREO_CHANNELS],
    consumed_bits: usize,
    padding_bits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McrCoreDiagnostics {
    cores: [CoreSideInfo; STEREO_CHANNELS],
    mcr: McrSideInfo,
    neural: NeuralSpectrumDiagnostics,
    entropy_bytes: usize,
    consumed_bits: usize,
    padding_bits: usize,
}

impl McrCoreDiagnostics {
    pub fn cores(self) -> [CoreSideInfo; STEREO_CHANNELS] {
        self.cores
    }

    pub fn mcr(self) -> McrSideInfo {
        self.mcr
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

impl StereoCoreDiagnostics {
    pub fn cores(self) -> [CoreSideInfo; STEREO_CHANNELS] {
        self.cores
    }

    pub fn stereo(self) -> StereoSideInfo {
        self.stereo
    }

    pub fn neural(self) -> [NeuralSpectrumDiagnostics; STEREO_CHANNELS] {
        self.neural
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

/// Ordinary stereo payload-to-interleaved-floating-PCM decoder.
///
/// Neural decoding and BWE share one explicit PRNG in the same left-to-right
/// order as the C decoder. Each channel owns its overlap state, while the
/// allocation-heavy neural and FFT plans are constructed once and reused.
/// Low-bitrate MCR frames decode one neural spectrum and use the normative
/// rotation codebooks to synthesize both channels before channel-local DSP.
pub struct StereoCoreDecoder<'model> {
    side_information: StereoSideInfoDecoder,
    neural: NeuralSpectrumDecoder<'model>,
    reorder: [SpectrumReorder; STEREO_CHANNELS],
    bwe: [BweSynthesis; STEREO_CHANNELS],
    tns: [TnsSynthesis; STEREO_CHANNELS],
    fd_shaping: [FdSpectrumShaping; STEREO_CHANNELS],
    mdct_synthesis: [MdctSynthesis; STEREO_CHANNELS],
    mcr: McrSynthesis,
    random: Avs3Random,
    spectra: [[f32; AVS3_FEATURE_DIMENSIONS]; STEREO_CHANNELS],
    synthesis: [[f32; AVS3_FEATURE_DIMENSIONS]; STEREO_CHANNELS],
}

impl<'model> StereoCoreDecoder<'model> {
    pub fn new(model: &'model NeuralModel) -> Result<Self, StereoCoreDecodeError> {
        Ok(Self {
            side_information: StereoSideInfoDecoder::new(),
            neural: NeuralSpectrumDecoder::new(model)?,
            reorder: [SpectrumReorder::new(), SpectrumReorder::new()],
            bwe: [BweSynthesis::new(), BweSynthesis::new()],
            tns: [TnsSynthesis::new(), TnsSynthesis::new()],
            fd_shaping: [FdSpectrumShaping::new(), FdSpectrumShaping::new()],
            mdct_synthesis: [MdctSynthesis::new(), MdctSynthesis::new()],
            mcr: McrSynthesis::new(),
            random: Avs3Random::new(),
            spectra: [[0.0; AVS3_FEATURE_DIMENSIONS]; STEREO_CHANNELS],
            synthesis: [[0.0; AVS3_FEATURE_DIMENSIONS]; STEREO_CHANNELS],
        })
    }

    pub fn reset(&mut self) {
        self.random.reset();
        for channel in 0..STEREO_CHANNELS {
            self.mdct_synthesis[channel].reset();
            self.spectra[channel].fill(0.0);
            self.synthesis[channel].fill(0.0);
        }
    }

    pub fn random(&self) -> &Avs3Random {
        &self.random
    }

    pub fn random_mut(&mut self) -> &mut Avs3Random {
        &mut self.random
    }

    pub fn last_shaped_spectra(&self) -> &[[f32; AVS3_FEATURE_DIMENSIONS]; STEREO_CHANNELS] {
        &self.spectra
    }

    pub fn decode(
        &mut self,
        payload: &[u8],
        header: &FrameHeader,
        output: &mut [f32],
    ) -> Result<StereoCoreDiagnostics, StereoCoreDecodeError> {
        if output.len() != STEREO_FRAME_SAMPLES {
            return Err(StereoCoreDecodeError::InvalidOutputLength {
                expected: STEREO_FRAME_SAMPLES,
                actual: output.len(),
            });
        }

        let parsed = self.side_information.parse(payload, header)?;
        let cores = parsed.cores();
        let stereo = parsed.stereo();
        let entropy_bytes = parsed.entropy_bytes();
        let consumed_bits = parsed.consumed_bits();
        let padding_bits = parsed.padding_bits();
        let neural_qc = parsed.neural_qc();
        let mut frame_random = self.random.clone();
        let mut neural_diagnostics = [None; STEREO_CHANNELS];

        for channel in 0..STEREO_CHANNELS {
            let decoded = match neural_qc[channel] {
                ParsedNeuralQc::Main(input) if header.nn_type == NnType::Main => {
                    self.neural.decode_main(input, &mut frame_random)?
                }
                ParsedNeuralQc::LowComplexity(input) if header.nn_type == NnType::LowComplexity => {
                    self.neural
                        .decode_low_complexity(input, &mut frame_random)?
                }
                _ => {
                    return Err(StereoCoreDecodeError::UnexpectedNeuralProfile { channel });
                }
            };
            neural_diagnostics[channel] = Some(decoded.diagnostics());
            self.spectra[channel].copy_from_slice(decoded.spectrum());
            self.reorder[channel].degroup(
                cores[channel].grouping(),
                cores[channel].transform_type(),
                &mut self.spectra[channel],
            )?;
        }

        if stereo.mid_side() {
            let ild = stereo.ild().ok_or(StereoCoreDecodeError::MissingIld)?;
            let [left, right] = &mut self.spectra;
            inverse_mid_side(left, right, ild)?;
        }

        self.apply_post_synthesis(cores, header.bitrate, &mut frame_random)?;
        for sample in 0..AVS3_FEATURE_DIMENSIONS {
            for channel in 0..STEREO_CHANNELS {
                output[sample * STEREO_CHANNELS + channel] = self.synthesis[channel][sample];
            }
        }

        self.random = frame_random;
        Ok(StereoCoreDiagnostics {
            cores,
            stereo,
            neural: neural_diagnostics.map(|value| value.expect("both channels decoded")),
            entropy_bytes,
            consumed_bits,
            padding_bits,
        })
    }

    pub fn decode_mcr(
        &mut self,
        payload: &[u8],
        header: &FrameHeader,
        output: &mut [f32],
    ) -> Result<McrCoreDiagnostics, StereoCoreDecodeError> {
        if output.len() != STEREO_FRAME_SAMPLES {
            return Err(StereoCoreDecodeError::InvalidOutputLength {
                expected: STEREO_FRAME_SAMPLES,
                actual: output.len(),
            });
        }

        let parsed = self.side_information.parse_mcr(payload, header)?;
        let cores = parsed.cores();
        let mcr = parsed.mcr();
        let entropy_bytes = parsed.entropy_bytes();
        let consumed_bits = parsed.consumed_bits();
        let padding_bits = parsed.padding_bits();
        let neural_qc = parsed.neural_qc();
        let mut frame_random = self.random.clone();

        let decoded = match neural_qc {
            ParsedNeuralQc::Main(input) if header.nn_type == NnType::Main => {
                self.neural.decode_main(input, &mut frame_random)?
            }
            ParsedNeuralQc::LowComplexity(input) if header.nn_type == NnType::LowComplexity => self
                .neural
                .decode_low_complexity(input, &mut frame_random)?,
            _ => {
                return Err(StereoCoreDecodeError::UnexpectedNeuralProfile { channel: 0 });
            }
        };
        let neural_diagnostics = decoded.diagnostics();
        self.spectra[0].copy_from_slice(decoded.spectrum());
        self.reorder[0].degroup(
            cores[0].grouping(),
            cores[0].transform_type(),
            &mut self.spectra[0],
        )?;
        let [left, right] = &mut self.spectra;
        self.mcr.apply(mcr, left, right)?;

        self.apply_post_synthesis(cores, header.bitrate, &mut frame_random)?;
        for sample in 0..AVS3_FEATURE_DIMENSIONS {
            for channel in 0..STEREO_CHANNELS {
                output[sample * STEREO_CHANNELS + channel] = self.synthesis[channel][sample];
            }
        }

        self.random = frame_random;
        Ok(McrCoreDiagnostics {
            cores,
            mcr,
            neural: neural_diagnostics,
            entropy_bytes,
            consumed_bits,
            padding_bits,
        })
    }

    fn apply_post_synthesis(
        &mut self,
        cores: [CoreSideInfo; STEREO_CHANNELS],
        bitrate: u32,
        frame_random: &mut Avs3Random,
    ) -> Result<(), StereoCoreDecodeError> {
        let bwe_config = BweConfig::for_stereo_bitrate(bitrate).map_err(StereoError::from)?;
        for (channel, core) in cores.iter().copied().enumerate() {
            match (bwe_config, core.bwe()) {
                (Some(config), Some(side_info)) => self.bwe[channel].apply(
                    config,
                    side_info,
                    &mut self.spectra[channel],
                    frame_random,
                )?,
                (None, None) => {}
                _ => {
                    return Err(StereoCoreDecodeError::InconsistentBweSideInformation { channel });
                }
            }
            self.tns[channel].apply(
                core.tns(),
                core.transform_type(),
                &mut self.spectra[channel],
            )?;
            self.fd_shaping[channel].apply(core.lsf(), &mut self.spectra[channel])?;
        }

        for (channel, core) in cores.iter().copied().enumerate() {
            self.mdct_synthesis[channel].synthesize(
                &self.spectra[channel],
                core.transform_type(),
                &mut self.synthesis[channel],
            )?;
        }
        Ok(())
    }
}

impl StereoCoreDecoder<'static> {
    pub fn new_builtin() -> Result<Self, StereoCoreDecodeError> {
        let model = crate::builtin_neural_model().map_err(NeuralQcError::from)?;
        Self::new(model)
    }
}

impl fmt::Debug for StereoCoreDecoder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StereoCoreDecoder")
            .field("neural", &self.neural)
            .field("fd_shaping", &self.fd_shaping)
            .field("mdct_synthesis", &"two channel-local FFT/overlap states")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AudioCodecId, BitDepth, BitWriter, ChannelConfig, CodecProfile, TransformType,
        float_to_pcm16,
    };

    fn header() -> FrameHeader {
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
            bed_bitrate: Some(64_000),
            object_bitrate: None,
            bitrate: 64_000,
            crc: 0,
            header_len: 7,
            payload_bits: 1_309,
            payload_len: 164,
            frame_len: 171,
            samples_per_channel: AVS3_FEATURE_DIMENSIONS as u32,
        }
    }

    fn mcr_header() -> FrameHeader {
        FrameHeader {
            bed_bitrate: Some(32_000),
            bitrate: 32_000,
            payload_bits: 626,
            payload_len: 79,
            frame_len: 86,
            ..header()
        }
    }

    fn write_core_prefix(writer: &mut BitWriter, lsf: [u64; 5], envelopes: [u64; 6]) {
        writer.write_bits(1, 2).unwrap();
        for (value, width) in lsf.into_iter().zip([8, 8, 7, 7, 6]) {
            writer.write_bits(value, width).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        for envelope in envelopes {
            writer.write_bits(envelope, 7).unwrap();
        }
        for _ in 0..3 {
            writer.write_bits(0, 1).unwrap();
        }
    }

    fn write_qc(writer: &mut BitWriter, entropy_bytes: usize) {
        let context: [u8; 6] = [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7];
        let base: [u8; 26] = [
            0x7f, 0xfd, 0x51, 0xf6, 0xf2, 0x24, 0x34, 0xad, 0x04, 0xde, 0x75, 0xcd, 0x9d, 0x0c,
            0x76, 0xeb, 0xb3, 0x76, 0xaf, 0x47, 0xda, 0x43, 0x33, 0xf0, 0xd4, 0xeb,
        ];
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(37, 7).unwrap();
        writer.write_bits(3, 3).unwrap();
        writer.write_bits(7, 3).unwrap();
        writer.write_bits(context.len() as u64, 8).unwrap();
        for byte in context.into_iter().chain(base) {
            writer.write_bits(u64::from(byte), 8).unwrap();
        }
        for _ in context.len() + base.len()..entropy_bytes {
            writer.write_bits(0, 8).unwrap();
        }
    }

    fn reference_payload() -> Vec<u8> {
        let mut writer = BitWriter::new();
        write_core_prefix(&mut writer, [3, 5, 7, 9, 11], [1, 2, 3, 4, 5, 6]);
        write_core_prefix(&mut writer, [17, 19, 21, 23, 25], [7, 8, 9, 10, 11, 12]);
        for _ in 0..2 {
            writer.write_bits(1, 1).unwrap();
            for indicator in [0, 0, 0, 1, 1, 1, 1, 1] {
                writer.write_bits(indicator, 1).unwrap();
            }
        }
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(5, 4).unwrap();
        writer.write_bits(4, 3).unwrap();
        assert_eq!(writer.bit_len(), 196);

        write_qc(&mut writer, 64);
        write_qc(&mut writer, 69);
        assert_eq!(writer.bit_len(), 1_304);
        let mut payload = writer.into_bytes();
        payload.resize(164, 0);
        payload
    }

    fn mcr_reference_payload() -> Vec<u8> {
        let mut writer = BitWriter::new();
        write_core_prefix(&mut writer, [3, 5, 7, 9, 11], [1, 2, 3, 4, 5, 6]);
        write_core_prefix(&mut writer, [17, 19, 21, 23, 25], [7, 8, 9, 10, 11, 12]);
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
        assert_eq!(writer.bit_len(), 275);
        write_qc(&mut writer, 41);
        assert_eq!(writer.bit_len(), 625);
        let mut payload = writer.into_bytes();
        payload.resize(79, 0);
        payload
    }

    fn fingerprint(values: &[f32]) -> u64 {
        values
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
                (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3)
            })
    }

    fn pcm_fingerprint(values: &[i16]) -> u64 {
        values
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
                (hash ^ u64::from(*value as u16)).wrapping_mul(0x100_0000_01b3)
            })
    }

    /// RustFFT's automatic planner is deliberately architecture-specific.
    /// Keep an exact baseline for every architecture used by CI instead of
    /// treating the SIMD result as if it were the scalar C FFT.
    #[cfg(target_arch = "x86_64")]
    fn short_mcr_fingerprint_baseline() -> [u64; 4] {
        [
            0xc5fd_6984_de2a_265b,
            0xd1f0_c0eb_b745_3453,
            0x7515_3048_77e6_198f,
            0x291b_df9d_9077_9ad0,
        ]
    }

    #[cfg(target_arch = "aarch64")]
    fn short_mcr_fingerprint_baseline() -> [u64; 4] {
        // Stable RustFFT 6.4.1 NEON output reproduced by Android and Linux,
        // macOS and Windows AArch64 CI.
        [
            0xd81e_4999_9fcd_2ad4,
            0xa4ca_e75c_01e5_1ffb,
            0x8e50_4505_bdb3_9357,
            0xdf7d_c254_a177_6513,
        ]
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn short_mcr_fingerprint_baseline() -> [u64; 4] {
        panic!(
            "no strict RustFFT fingerprint baseline for target architecture {}",
            std::env::consts::ARCH
        );
    }

    #[test]
    fn stereo_neural_degroup_ms_and_bwe_order_is_bit_exact_with_c() {
        let header = header();
        let payload = reference_payload();
        let mut side_decoder = StereoSideInfoDecoder::new();
        let parsed = side_decoder.parse(&payload, &header).unwrap();
        let cores = parsed.cores();
        let neural_qc = parsed.neural_qc();
        let mut neural = NeuralSpectrumDecoder::new_builtin().unwrap();
        let mut random = Avs3Random::new();
        let mut spectra = [[0.0_f32; AVS3_FEATURE_DIMENSIONS]; STEREO_CHANNELS];
        let mut reorder = [SpectrumReorder::new(), SpectrumReorder::new()];
        let expected_neural = [0xa8c7_da41_5b7c_0077, 0xd085_01af_39c6_f24a];
        let expected_degroup = [0xab1c_f5b2_51a8_83ed, 0xc692_e63d_2748_078e];
        for channel in 0..STEREO_CHANNELS {
            let ParsedNeuralQc::Main(input) = neural_qc[channel] else {
                panic!("reference uses Main QC")
            };
            let decoded = neural.decode_main(input, &mut random).unwrap();
            spectra[channel].copy_from_slice(decoded.spectrum());
            assert_eq!(fingerprint(&spectra[channel]), expected_neural[channel]);
            reorder[channel]
                .degroup(
                    cores[channel].grouping(),
                    cores[channel].transform_type(),
                    &mut spectra[channel],
                )
                .unwrap();
            assert_eq!(fingerprint(&spectra[channel]), expected_degroup[channel]);
        }

        let [left, right] = &mut spectra;
        inverse_mid_side(left, right, 5).unwrap();
        assert_eq!(
            spectra.map(|spectrum| fingerprint(&spectrum)),
            [0xbede_6807_c5ff_18bc, 0xf4b0_b67e_b088_a681]
        );

        let config = BweConfig::for_stereo_bitrate(header.bitrate)
            .unwrap()
            .unwrap();
        let mut bwe = [BweSynthesis::new(), BweSynthesis::new()];
        for channel in 0..STEREO_CHANNELS {
            bwe[channel]
                .apply(
                    config,
                    cores[channel].bwe().unwrap(),
                    &mut spectra[channel],
                    &mut random,
                )
                .unwrap();
        }
        assert_eq!(
            spectra.map(|spectrum| fingerprint(&spectrum)),
            [0xc0c7_b467_f043_8bcc, 0x5408_b4f8_499e_2b6b]
        );
        assert_eq!(random.next_u31(), 845_563_291);
    }

    #[test]
    fn complete_ordinary_stereo_pipeline_stays_close_to_c() {
        let header = header();
        let payload = reference_payload();
        let mut decoder = StereoCoreDecoder::new_builtin().unwrap();
        let mut output = [0.0_f32; STEREO_FRAME_SAMPLES];
        let diagnostics = decoder.decode(&payload, &header, &mut output).unwrap();

        assert_eq!(diagnostics.entropy_bytes(), [64, 69]);
        assert_eq!(diagnostics.consumed_bits(), 1_304);
        assert_eq!(diagnostics.padding_bits(), 5);
        assert_eq!(
            diagnostics.cores().map(CoreSideInfo::transform_type),
            [TransformType::Short, TransformType::Short]
        );
        assert!(diagnostics.stereo().mid_side());
        assert_eq!(diagnostics.stereo().ild(), Some(5));
        for sample in 0..448 {
            assert_eq!(output[sample * STEREO_CHANNELS], 0.0);
            assert_eq!(output[sample * STEREO_CHANNELS + 1], 0.0);
        }

        let positions = [0, 447, 448, 449, 575, 576, 700, 900, 1023];
        let expected = [
            [
                0x0000_0000,
                0x0000_0000,
                0xbdda_2442,
                0xbeac_46be,
                0x418a_df1d,
                0x405d_2365,
                0xc0ba_a027,
                0x415f_9cd4,
                0x4258_78c0,
            ],
            [
                0x0000_0000,
                0x0000_0000,
                0xbeae_a800,
                0xbf67_19ae,
                0x425e_605e,
                0x4284_283c,
                0x429e_873f,
                0x431e_1ccc,
                0xc297_ad28,
            ],
        ];
        for channel in 0..STEREO_CHANNELS {
            for (position, bits) in positions.into_iter().zip(expected[channel]) {
                let actual = output[position * STEREO_CHANNELS + channel];
                let expected = f32::from_bits(bits);
                let error = (actual - expected).abs();
                let tolerance = 2.0e-4_f32 * expected.abs().max(1.0);
                assert!(
                    error <= tolerance,
                    "channel {channel} position {position}: Rust={actual} C={expected} error={error} tolerance={tolerance}"
                );
            }
        }
        let mut pcm = [0_i16; STEREO_FRAME_SAMPLES];
        assert_eq!(float_to_pcm16(&output, &mut pcm), 0);
        assert_eq!(pcm_fingerprint(&pcm), 0xa036_2bb2_f0ab_465a);
        assert_eq!(decoder.random_mut().next_u31(), 845_563_291);
    }

    #[test]
    fn complete_short_mcr_pipeline_stays_close_to_c() {
        let header = mcr_header();
        let payload = mcr_reference_payload();
        let mut decoder = StereoCoreDecoder::new_builtin().unwrap();
        let mut output = [0.0_f32; STEREO_FRAME_SAMPLES];
        let diagnostics = decoder.decode_mcr(&payload, &header, &mut output).unwrap();

        assert_eq!(diagnostics.entropy_bytes(), 41);
        assert_eq!(diagnostics.consumed_bits(), 625);
        assert_eq!(diagnostics.padding_bits(), 1);
        assert_eq!(
            diagnostics.cores().map(CoreSideInfo::transform_type),
            [TransformType::Short, TransformType::Short]
        );
        assert_eq!(
            diagnostics.mcr().vq_indexes(),
            &[[1_u16, 2, 3, 4, 5, 255], [255_u16, 5, 4, 3, 2, 1],]
        );
        for sample in 0..448 {
            assert_eq!(output[sample * STEREO_CHANNELS], 0.0);
            assert_eq!(output[sample * STEREO_CHANNELS + 1], 0.0);
        }

        let spectra = decoder.last_shaped_spectra();
        let mut pcm = [0_i16; STEREO_FRAME_SAMPLES];
        assert_eq!(float_to_pcm16(&output, &mut pcm), 0);
        let positions = [448, 449, 575, 576, 694, 900, 965, 994, 1022, 1023];
        let expected = [
            [
                0xbd0a_be6b,
                0xbee3_30ff,
                0x40b0_a6e3,
                0xc183_3dc7,
                0xc1a7_7568,
                0x4303_cceb,
                0xc2a0_435d,
                0x42c6_fc51,
                0x414a_e514,
                0x4188_96ef,
            ],
            [
                0xbd9a_b671,
                0x3e8a_f4c6,
                0x4144_fbdb,
                0x421c_f900,
                0x4206_0032,
                0xc346_7f35,
                0x426a_8d86,
                0xc205_fe83,
                0xc214_b324,
                0xc266_d717,
            ],
        ];
        for channel in 0..STEREO_CHANNELS {
            for (position, bits) in positions.into_iter().zip(expected[channel]) {
                let actual = output[position * STEREO_CHANNELS + channel];
                let expected = f32::from_bits(bits);
                let error = (actual - expected).abs();
                let tolerance = 2.0e-4_f32 * expected.abs().max(1.0);
                assert!(
                    error <= tolerance,
                    "channel {channel} position {position}: Rust={actual} C={expected} error={error} tolerance={tolerance}"
                );
            }
        }
        let actual = [
            fingerprint(&spectra[0]),
            fingerprint(&spectra[1]),
            fingerprint(&output),
            pcm_fingerprint(&pcm),
        ];
        assert_eq!(actual, short_mcr_fingerprint_baseline());
    }

    #[test]
    fn invalid_output_length_does_not_advance_stereo_state() {
        let mut decoder = StereoCoreDecoder::new_builtin().unwrap();
        let original_random = decoder.random().clone();
        let mut output = [7.0_f32; 17];
        assert_eq!(
            decoder
                .decode(&reference_payload(), &header(), &mut output)
                .unwrap_err(),
            StereoCoreDecodeError::InvalidOutputLength {
                expected: STEREO_FRAME_SAMPLES,
                actual: 17,
            }
        );
        assert_eq!(decoder.random(), &original_random);
        assert_eq!(output, [7.0; 17]);
    }
}
