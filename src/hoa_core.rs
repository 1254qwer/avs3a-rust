use core::fmt;

use crate::bwe::{BweSynthesis, BweSynthesisError};
use crate::core_side::{CoreSideInfo, ParsedNeuralQc};
use crate::fd_shaping::{FdShapingError, FdSpectrumShaping};
use crate::header::{FrameHeader, MAX_CHANNELS, NnType};
use crate::hoa::{HoaBitstreamConfig, HoaError, HoaSideInfo, HoaSideInfoDecoder, inverse_hoa_dmx};
use crate::hoa_synthesis::{HoaPostSynthesis, HoaPostSynthesisError};
use crate::mdct_synthesis::{MdctSynthesis, MdctSynthesisError};
use crate::model::{AVS3_FEATURE_DIMENSIONS, NeuralModel};
use crate::neural_qc::{NeuralQcError, NeuralSpectrumDecoder, NeuralSpectrumDiagnostics};
use crate::random::Avs3Random;
use crate::spectrum::{SpectrumReorder, SpectrumReorderError};
use crate::tns::{TnsSynthesis, TnsSynthesisError};

pub const HOA_MAX_FRAME_SAMPLES: usize = AVS3_FEATURE_DIMENSIONS * MAX_CHANNELS as usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HoaCoreDecodeError {
    InvalidOutputLength { expected: usize, actual: usize },
    MissingCoreSideInformation { channel: usize },
    MissingNeuralSideInformation { channel: usize },
    UnexpectedNeuralProfile { channel: usize },
    InconsistentBweSideInformation { channel: usize },
    AllocationOverflow,
    Hoa(HoaError),
    NeuralQc(NeuralQcError),
    SpectrumReorder(SpectrumReorderError),
    Bwe(BweSynthesisError),
    Tns(TnsSynthesisError),
    FdShaping(FdShapingError),
    MdctSynthesis(MdctSynthesisError),
    HoaPostSynthesis(HoaPostSynthesisError),
}

impl fmt::Display for HoaCoreDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOutputLength { expected, actual } => write!(
                f,
                "HOA synthesis output has {actual} samples; expected {expected}"
            ),
            Self::MissingCoreSideInformation { channel } => {
                write!(
                    f,
                    "HOA transport channel {channel} is missing core side information"
                )
            }
            Self::MissingNeuralSideInformation { channel } => write!(
                f,
                "HOA transport channel {channel} is missing neural side information"
            ),
            Self::UnexpectedNeuralProfile { channel } => write!(
                f,
                "parsed neural QC profile does not match HOA transport channel {channel} configuration"
            ),
            Self::InconsistentBweSideInformation { channel } => write!(
                f,
                "HOA transport channel {channel} BWE configuration and side information are inconsistent"
            ),
            Self::AllocationOverflow => f.write_str("HOA decoder buffer-size arithmetic overflow"),
            Self::Hoa(error) => error.fmt(f),
            Self::NeuralQc(error) => error.fmt(f),
            Self::SpectrumReorder(error) => error.fmt(f),
            Self::Bwe(error) => error.fmt(f),
            Self::Tns(error) => error.fmt(f),
            Self::FdShaping(error) => error.fmt(f),
            Self::MdctSynthesis(error) => error.fmt(f),
            Self::HoaPostSynthesis(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for HoaCoreDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Hoa(error) => Some(error),
            Self::NeuralQc(error) => Some(error),
            Self::SpectrumReorder(error) => Some(error),
            Self::Bwe(error) => Some(error),
            Self::Tns(error) => Some(error),
            Self::FdShaping(error) => Some(error),
            Self::MdctSynthesis(error) => Some(error),
            Self::HoaPostSynthesis(error) => Some(error),
            _ => None,
        }
    }
}

impl From<HoaError> for HoaCoreDecodeError {
    fn from(value: HoaError) -> Self {
        Self::Hoa(value)
    }
}

impl From<NeuralQcError> for HoaCoreDecodeError {
    fn from(value: NeuralQcError) -> Self {
        Self::NeuralQc(value)
    }
}

impl From<SpectrumReorderError> for HoaCoreDecodeError {
    fn from(value: SpectrumReorderError) -> Self {
        Self::SpectrumReorder(value)
    }
}

impl From<BweSynthesisError> for HoaCoreDecodeError {
    fn from(value: BweSynthesisError) -> Self {
        Self::Bwe(value)
    }
}

impl From<TnsSynthesisError> for HoaCoreDecodeError {
    fn from(value: TnsSynthesisError) -> Self {
        Self::Tns(value)
    }
}

impl From<FdShapingError> for HoaCoreDecodeError {
    fn from(value: FdShapingError) -> Self {
        Self::FdShaping(value)
    }
}

impl From<MdctSynthesisError> for HoaCoreDecodeError {
    fn from(value: MdctSynthesisError) -> Self {
        Self::MdctSynthesis(value)
    }
}

impl From<HoaPostSynthesisError> for HoaCoreDecodeError {
    fn from(value: HoaPostSynthesisError) -> Self {
        Self::HoaPostSynthesis(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoaCoreDiagnostics {
    transport_channels: usize,
    output_channels: usize,
    cores: [Option<CoreSideInfo>; MAX_CHANNELS as usize],
    hoa: HoaSideInfo,
    neural: [Option<NeuralSpectrumDiagnostics>; MAX_CHANNELS as usize],
    entropy_bytes: [usize; MAX_CHANNELS as usize],
    consumed_bits: usize,
    padding_bits: usize,
}

impl HoaCoreDiagnostics {
    pub fn transport_channels(self) -> usize {
        self.transport_channels
    }

    pub fn output_channels(self) -> usize {
        self.output_channels
    }

    pub fn core(self, channel: usize) -> Option<CoreSideInfo> {
        self.cores.get(channel).copied().flatten()
    }

    pub fn hoa(self) -> HoaSideInfo {
        self.hoa
    }

    pub fn neural(self, channel: usize) -> Option<NeuralSpectrumDiagnostics> {
        self.neural.get(channel).copied().flatten()
    }

    pub fn entropy_bytes(&self) -> &[usize] {
        &self.entropy_bytes[..self.transport_channels]
    }

    pub fn consumed_bits(self) -> usize {
        self.consumed_bits
    }

    pub fn padding_bits(self) -> usize {
        self.padding_bits
    }
}

#[derive(Debug)]
struct HoaChannelDsp {
    reorder: SpectrumReorder,
    bwe: BweSynthesis,
    tns: TnsSynthesis,
    fd_shaping: FdSpectrumShaping,
    mdct_synthesis: MdctSynthesis,
}

impl HoaChannelDsp {
    fn new() -> Self {
        Self {
            reorder: SpectrumReorder::new(),
            bwe: BweSynthesis::new(),
            tns: TnsSynthesis::new(),
            fd_shaping: FdSpectrumShaping::new(),
            mdct_synthesis: MdctSynthesis::new(),
        }
    }

    fn reset(&mut self) {
        self.mdct_synthesis.reset();
    }
}

/// HOA payload-to-interleaved-floating-PCM decoder.
///
/// Neural decoding shares one decoder-local PRNG in transport-channel order.
/// Core overlap, HOA analysis delay, delayed basis state and final synthesis
/// overlap are all owned by this instance. Construction performs every heap
/// allocation used by the frame path.
pub struct HoaCoreDecoder<'model> {
    side_information: HoaSideInfoDecoder,
    neural: NeuralSpectrumDecoder<'model>,
    channel_dsp: Vec<HoaChannelDsp>,
    post_synthesis: HoaPostSynthesis,
    random: Avs3Random,
    spectra: Vec<[f32; AVS3_FEATURE_DIMENSIONS]>,
    transport_time: Vec<[f32; AVS3_FEATURE_DIMENSIONS]>,
    hoa_output: Vec<[f32; AVS3_FEATURE_DIMENSIONS]>,
    last_transport_channels: usize,
    last_output_channels: usize,
}

impl<'model> HoaCoreDecoder<'model> {
    pub fn new(model: &'model NeuralModel) -> Result<Self, HoaCoreDecodeError> {
        let capacity = usize::from(MAX_CHANNELS);
        let mut channel_dsp = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            channel_dsp.push(HoaChannelDsp::new());
        }
        Ok(Self {
            side_information: HoaSideInfoDecoder::new(),
            neural: NeuralSpectrumDecoder::new(model)?,
            channel_dsp,
            post_synthesis: HoaPostSynthesis::new(),
            random: Avs3Random::new(),
            spectra: vec![[0.0; AVS3_FEATURE_DIMENSIONS]; capacity],
            transport_time: vec![[0.0; AVS3_FEATURE_DIMENSIONS]; capacity],
            hoa_output: vec![[0.0; AVS3_FEATURE_DIMENSIONS]; capacity],
            last_transport_channels: 0,
            last_output_channels: 0,
        })
    }

    pub fn reset(&mut self) {
        self.random.reset();
        for channel in &mut self.channel_dsp {
            channel.reset();
        }
        self.post_synthesis.reset();
        for spectrum in &mut self.spectra {
            spectrum.fill(0.0);
        }
        for channel in &mut self.transport_time {
            channel.fill(0.0);
        }
        for channel in &mut self.hoa_output {
            channel.fill(0.0);
        }
        self.last_transport_channels = 0;
        self.last_output_channels = 0;
    }

    pub fn random(&self) -> &Avs3Random {
        &self.random
    }

    pub fn random_mut(&mut self) -> &mut Avs3Random {
        &mut self.random
    }

    pub fn post_synthesis(&self) -> &HoaPostSynthesis {
        &self.post_synthesis
    }

    pub fn last_shaped_spectra(&self) -> &[[f32; AVS3_FEATURE_DIMENSIONS]] {
        &self.spectra[..self.last_transport_channels]
    }

    pub fn last_transport_time(&self) -> &[[f32; AVS3_FEATURE_DIMENSIONS]] {
        &self.transport_time[..self.last_transport_channels]
    }

    pub fn decode(
        &mut self,
        payload: &[u8],
        header: &FrameHeader,
        output: &mut [f32],
    ) -> Result<HoaCoreDiagnostics, HoaCoreDecodeError> {
        let config = HoaBitstreamConfig::for_header(header)?;
        let transport_channels = config.transport_channels();
        let output_channels = config.output_channels();
        let expected_output = output_channels
            .checked_mul(AVS3_FEATURE_DIMENSIONS)
            .ok_or(HoaCoreDecodeError::AllocationOverflow)?;
        if output.len() != expected_output {
            return Err(HoaCoreDecodeError::InvalidOutputLength {
                expected: expected_output,
                actual: output.len(),
            });
        }

        let parsed = self.side_information.parse(payload, header)?;
        let hoa = parsed.hoa();
        let allocation = parsed.allocation();
        let consumed_bits = parsed.consumed_bits();
        let padding_bits = parsed.padding_bits();
        let mut frame_random = self.random.clone();
        let mut cores = [None; MAX_CHANNELS as usize];
        let mut neural_diagnostics = [None; MAX_CHANNELS as usize];
        let mut entropy_bytes = [0_usize; MAX_CHANNELS as usize];
        entropy_bytes[..transport_channels].copy_from_slice(allocation.channel_bytes());

        for channel in 0..transport_channels {
            let core = parsed
                .core(channel)
                .ok_or(HoaCoreDecodeError::MissingCoreSideInformation { channel })?;
            let neural_qc = parsed
                .neural_qc(channel)
                .ok_or(HoaCoreDecodeError::MissingNeuralSideInformation { channel })?;
            let decoded = match neural_qc {
                ParsedNeuralQc::Main(input) if header.nn_type == NnType::Main => {
                    self.neural.decode_main(input, &mut frame_random)?
                }
                ParsedNeuralQc::LowComplexity(input) if header.nn_type == NnType::LowComplexity => {
                    self.neural
                        .decode_low_complexity(input, &mut frame_random)?
                }
                _ => return Err(HoaCoreDecodeError::UnexpectedNeuralProfile { channel }),
            };
            neural_diagnostics[channel] = Some(decoded.diagnostics());
            self.spectra[channel].copy_from_slice(decoded.spectrum());
            self.channel_dsp[channel].reorder.degroup(
                core.grouping(),
                core.transform_type(),
                &mut self.spectra[channel],
            )?;
            cores[channel] = Some(core);
        }

        inverse_hoa_dmx(&mut self.spectra[..transport_channels], hoa, config)?;

        for (channel, core) in cores[..transport_channels].iter().copied().enumerate() {
            let core = core.ok_or(HoaCoreDecodeError::MissingCoreSideInformation { channel })?;
            let core_config = config.core_for_channel(header, channel)?;
            match (core_config.bwe(), core.bwe()) {
                (Some(bwe), Some(side_info)) => self.channel_dsp[channel].bwe.apply(
                    bwe,
                    side_info,
                    &mut self.spectra[channel],
                    &mut frame_random,
                )?,
                (None, None) => {}
                _ => {
                    return Err(HoaCoreDecodeError::InconsistentBweSideInformation { channel });
                }
            }
            self.channel_dsp[channel].tns.apply(
                core.tns(),
                core.transform_type(),
                &mut self.spectra[channel],
            )?;
            self.channel_dsp[channel]
                .fd_shaping
                .apply(core.lsf(), &mut self.spectra[channel])?;
            self.channel_dsp[channel].mdct_synthesis.synthesize(
                &self.spectra[channel],
                core.transform_type(),
                &mut self.transport_time[channel],
            )?;
        }

        self.post_synthesis.process(
            &self.transport_time[..transport_channels],
            config,
            hoa,
            &mut self.hoa_output[..output_channels],
        )?;
        for sample in 0..AVS3_FEATURE_DIMENSIONS {
            for channel in 0..output_channels {
                output[sample * output_channels + channel] = self.hoa_output[channel][sample];
            }
        }

        self.random = frame_random;
        self.last_transport_channels = transport_channels;
        self.last_output_channels = output_channels;
        Ok(HoaCoreDiagnostics {
            transport_channels,
            output_channels,
            cores,
            hoa,
            neural: neural_diagnostics,
            entropy_bytes,
            consumed_bits,
            padding_bits,
        })
    }
}

impl HoaCoreDecoder<'static> {
    pub fn new_builtin() -> Result<Self, HoaCoreDecodeError> {
        let model = crate::builtin_neural_model().map_err(NeuralQcError::from)?;
        Self::new(model)
    }
}

impl fmt::Debug for HoaCoreDecoder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HoaCoreDecoder")
            .field("neural", &self.neural)
            .field("channel_states", &self.channel_dsp.len())
            .field("post_synthesis", &self.post_synthesis)
            .field("last_transport_channels", &self.last_transport_channels)
            .field("last_output_channels", &self.last_output_channels)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BitWriter;
    use crate::header::{AudioCodecId, BitDepth, ChannelConfig, CodecProfile};

    const AUDIO_BITS: usize = 4_038;

    fn header() -> FrameHeader {
        FrameHeader {
            codec_id: AudioCodecId::Avs3P3,
            nn_type: NnType::Main,
            profile: CodecProfile::Hoa,
            sample_rate: 48_000,
            bit_depth: BitDepth::Sixteen,
            channel_config: Some(ChannelConfig::Hoa1),
            sound_bed_type: None,
            hoa_order: Some(1),
            objects: 0,
            bed_channels: 4,
            channels: 4,
            has_lfe: false,
            bed_bitrate: None,
            object_bitrate: None,
            bitrate: 192_000,
            crc: 0,
            header_len: 7,
            payload_bits: AUDIO_BITS,
            payload_len: AUDIO_BITS.div_ceil(8),
            frame_len: 7 + AUDIO_BITS.div_ceil(8),
            samples_per_channel: AVS3_FEATURE_DIMENSIONS as u32,
        }
    }

    fn write_qc(writer: &mut BitWriter, entropy_bytes: usize) {
        let context: [u8; 6] = [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7];
        let base: [u8; 26] = [
            0x7f, 0xfd, 0x51, 0xf6, 0xf2, 0x24, 0x34, 0xad, 0x04, 0xde, 0x75, 0xcd, 0x9d, 0x0c,
            0x76, 0xeb, 0xb3, 0x76, 0xaf, 0x47, 0xda, 0x43, 0x33, 0xf0, 0xd4, 0xeb,
        ];
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(37, 7).unwrap();
        writer.write_bits(3, 3).unwrap();
        writer.write_bits(context.len() as u64, 8).unwrap();
        for byte in context.into_iter().chain(base).take(entropy_bytes) {
            writer.write_bits(u64::from(byte), 8).unwrap();
        }
        for _ in (context.len() + base.len()).min(entropy_bytes)..entropy_bytes {
            writer.write_bits(0, 8).unwrap();
        }
    }

    fn payload() -> Vec<u8> {
        let mut writer = BitWriter::new();
        for _ in 0..4 {
            writer.write_bits(0, 2).unwrap();
            for width in [8, 8, 7, 7, 6, 5, 5] {
                writer.write_bits(0, width).unwrap();
            }
            writer.write_bits(0, 1).unwrap();
            writer.write_bits(0, 1).unwrap();
            for _ in 0..4 {
                writer.write_bits(0, 7).unwrap();
            }
            writer.write_bits(0, 1).unwrap();
            writer.write_bits(0, 1).unwrap();
        }

        writer.write_bits(3, 4).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 4).unwrap();
        writer.write_bits(15, 4).unwrap();
        for _ in 0..4 {
            writer.write_bits(4, 4).unwrap();
        }
        assert_eq!(writer.bit_len(), 349);

        for entropy_bytes in [115, 112, 112, 112] {
            write_qc(&mut writer, entropy_bytes);
        }
        assert_eq!(writer.bit_len(), 4_033);
        writer.write_bits(0, 5).unwrap();
        writer.into_bytes()
    }

    #[test]
    fn complete_foa_pipeline_reaches_interleaved_pcm() {
        let mut decoder = HoaCoreDecoder::new_builtin().unwrap();
        let mut output = [0.0_f32; AVS3_FEATURE_DIMENSIONS * 4];
        let diagnostics = decoder.decode(&payload(), &header(), &mut output).unwrap();

        assert_eq!(diagnostics.transport_channels(), 4);
        assert_eq!(diagnostics.output_channels(), 4);
        assert_eq!(diagnostics.entropy_bytes(), &[115, 112, 112, 112]);
        assert_eq!(diagnostics.consumed_bits(), 4_033);
        assert_eq!(diagnostics.padding_bits(), 5);
        assert_eq!(diagnostics.hoa().scene_type(), 3);
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(output.iter().any(|&sample| sample != 0.0));
    }

    #[test]
    fn invalid_output_length_does_not_advance_random_or_hoa_delay() {
        let mut decoder = HoaCoreDecoder::new_builtin().unwrap();
        let original_random = decoder.random().clone();
        let mut output = [7.0_f32; 17];
        assert_eq!(
            decoder
                .decode(&payload(), &header(), &mut output)
                .unwrap_err(),
            HoaCoreDecodeError::InvalidOutputLength {
                expected: AVS3_FEATURE_DIMENSIONS * 4,
                actual: 17,
            }
        );
        assert_eq!(decoder.random(), &original_random);
        assert_eq!(
            decoder.post_synthesis().delayed_basis_indices(),
            &[[0; 4]; 2]
        );
        assert_eq!(output, [7.0; 17]);
    }
}
