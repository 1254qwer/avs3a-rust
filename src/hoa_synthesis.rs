use core::fmt;

use crate::header::MAX_CHANNELS;
use crate::hoa_side::{
    HOA_BASIS_TABLE_LEN, HoaBitstreamConfig, HoaError, HoaSideInfo, MAX_HOA_BASIS,
};
use crate::imdct::{FastImdct, ImdctError};
use crate::mdct::{FastMdct, MdctError};
use crate::model::AVS3_FEATURE_DIMENSIONS;

pub const HOA_FRAME_SAMPLES: usize = AVS3_FEATURE_DIMENSIONS;
pub const HOA_OVERLAP_SIZE: usize = HOA_FRAME_SAMPLES / 2;
pub const HOA_POST_TRANSFORM_LEN: usize = HOA_OVERLAP_SIZE * 2;
pub const HOA_SPATIAL_TABLE_BYTES_LEN: usize = 6_400;
pub const HOA_SPATIAL_TABLE_FNV1A: u64 = 0x91a0_296f_d4de_f1af;

const HOA_ANGLE_VALUES: usize = HOA_BASIS_TABLE_LEN * 2;
const HOA_ANGLE_BYTES: usize = HOA_ANGLE_VALUES * core::mem::size_of::<i16>();
const HOA_SIN_TABLE_LEN: usize = 257;
const HOA_SIN_QUARTER: usize = 256;
const HOA_SIN_HALF: usize = 512;
const HOA_SIN_THREE_QUARTERS: usize = 768;
const HOA_SIN_FULL: usize = 1_024;
/// Frames the HOA spatial basis indices are delayed by before use.
pub const HOA_BASIS_DELAY_FRAMES: usize = 2;
const HOA_OUTPUT_CHANNELS: usize = MAX_CHANNELS as usize;
const AVS3_PI: f32 = core::f32::consts::PI;

const HOA_SPATIAL_TABLE_BYTES: &[u8; HOA_SPATIAL_TABLE_BYTES_LEN] =
    include_bytes!("../assets/avs3a_hoa_spatial_tables.bin");

pub fn hoa_spatial_table_bytes() -> &'static [u8; HOA_SPATIAL_TABLE_BYTES_LEN] {
    HOA_SPATIAL_TABLE_BYTES
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HoaPostSynthesisError {
    InvalidTransportChannelCount {
        expected: usize,
        actual: usize,
    },
    InvalidOutputChannelCount {
        expected: usize,
        actual: usize,
    },
    SideInfoConfigurationMismatch,
    InvalidRecoveryLayout {
        vector_channels: usize,
        residual_channels: usize,
        transport_channels: usize,
    },
    Hoa(HoaError),
    Mdct(MdctError),
    Imdct(ImdctError),
}

impl fmt::Display for HoaPostSynthesisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransportChannelCount { expected, actual } => write!(
                f,
                "HOA post-filter received {actual} transport channels; expected {expected}"
            ),
            Self::InvalidOutputChannelCount { expected, actual } => write!(
                f,
                "HOA post-filter received {actual} output channels; expected {expected}"
            ),
            Self::SideInfoConfigurationMismatch => {
                f.write_str("HOA post-filter side information does not match its configuration")
            }
            Self::InvalidRecoveryLayout {
                vector_channels,
                residual_channels,
                transport_channels,
            } => write!(
                f,
                "HOA recovery needs {vector_channels} vector and {residual_channels} residual channels, but only {transport_channels} transports exist"
            ),
            Self::Hoa(error) => error.fmt(f),
            Self::Mdct(error) => error.fmt(f),
            Self::Imdct(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for HoaPostSynthesisError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Hoa(error) => Some(error),
            Self::Mdct(error) => Some(error),
            Self::Imdct(error) => Some(error),
            _ => None,
        }
    }
}

impl From<HoaError> for HoaPostSynthesisError {
    fn from(value: HoaError) -> Self {
        Self::Hoa(value)
    }
}

impl From<MdctError> for HoaPostSynthesisError {
    fn from(value: MdctError) -> Self {
        Self::Mdct(value)
    }
}

impl From<ImdctError> for HoaPostSynthesisError {
    fn from(value: ImdctError) -> Self {
        Self::Imdct(value)
    }
}

/// Derive the 16 real third-order HOA basis coefficients for one table index.
pub fn hoa_basis_coefficients(index: usize) -> Result<[f32; HOA_OUTPUT_CHANNELS], HoaError> {
    if index >= HOA_BASIS_TABLE_LEN {
        return Err(HoaError::InvalidBasisIndex {
            index,
            limit: HOA_BASIS_TABLE_LEN,
        });
    }
    let [azimuth, elevation] = hoa_angle_pair(index);
    let sin_azimuth = quantized_sine(azimuth);
    let cos_azimuth = quantized_cosine(azimuth);
    let sin_elevation = quantized_sine(elevation);
    let cos_elevation = quantized_cosine(elevation);
    Ok(third_order_basis(
        sin_azimuth,
        cos_azimuth,
        sin_elevation,
        cos_elevation,
    ))
}

/// Stateful 512-hop HOA analysis, spatial recovery and synthesis filter.
///
/// Input transport channels are the 1024 time samples produced by each core's
/// ordinary AVS3 overlap-add stage. The filter analyzes two 1024-sample
/// windows beginning half a frame before the current input, optionally
/// recovers up to 16 HOA components with two-frame-delayed basis indices, then
/// performs two windowed 1024-point inverse transforms. All storage and FFT
/// plans are decoder-owned and reused across frames.
pub struct HoaPostSynthesis {
    mdct: FastMdct,
    imdct: FastImdct,
    window: [f32; HOA_OVERLAP_SIZE],
    analysis_delay: Vec<[f32; HOA_FRAME_SAMPLES]>,
    spectra: Vec<[f32; HOA_FRAME_SAMPLES]>,
    recovery: Vec<[f32; HOA_FRAME_SAMPLES]>,
    synthesis_overlap: Vec<[f32; HOA_OVERLAP_SIZE]>,
    delayed_basis_indices: [[u16; MAX_HOA_BASIS]; HOA_BASIS_DELAY_FRAMES],
    basis_matrix: [[f32; MAX_HOA_BASIS]; HOA_OUTPUT_CHANNELS],
    transform_signal: [f32; HOA_POST_TRANSFORM_LEN],
    last_transport_channels: usize,
    last_output_channels: usize,
}

impl HoaPostSynthesis {
    pub fn new() -> Self {
        let mut window = [0.0_f32; HOA_OVERLAP_SIZE];
        fill_sine_window(&mut window);
        Self {
            mdct: FastMdct::new(),
            imdct: FastImdct::new(),
            window,
            analysis_delay: vec![[0.0; HOA_FRAME_SAMPLES]; HOA_OUTPUT_CHANNELS],
            spectra: vec![[0.0; HOA_FRAME_SAMPLES]; HOA_OUTPUT_CHANNELS],
            recovery: vec![[0.0; HOA_FRAME_SAMPLES]; HOA_OUTPUT_CHANNELS],
            synthesis_overlap: vec![[0.0; HOA_OVERLAP_SIZE]; HOA_OUTPUT_CHANNELS],
            delayed_basis_indices: [[0; MAX_HOA_BASIS]; HOA_BASIS_DELAY_FRAMES],
            basis_matrix: [[0.0; MAX_HOA_BASIS]; HOA_OUTPUT_CHANNELS],
            transform_signal: [0.0; HOA_POST_TRANSFORM_LEN],
            last_transport_channels: 0,
            last_output_channels: 0,
        }
    }

    pub fn reset(&mut self) {
        for channel in &mut self.analysis_delay {
            channel.fill(0.0);
        }
        for channel in &mut self.spectra {
            channel.fill(0.0);
        }
        for channel in &mut self.recovery {
            channel.fill(0.0);
        }
        for channel in &mut self.synthesis_overlap {
            channel.fill(0.0);
        }
        self.delayed_basis_indices = [[0; MAX_HOA_BASIS]; HOA_BASIS_DELAY_FRAMES];
        self.basis_matrix = [[0.0; MAX_HOA_BASIS]; HOA_OUTPUT_CHANNELS];
        self.transform_signal.fill(0.0);
        self.last_transport_channels = 0;
        self.last_output_channels = 0;
    }

    pub fn delayed_basis_indices(&self) -> &[[u16; MAX_HOA_BASIS]; HOA_BASIS_DELAY_FRAMES] {
        &self.delayed_basis_indices
    }

    pub fn last_spectra(&self) -> &[[f32; HOA_FRAME_SAMPLES]] {
        &self.spectra[..self.last_output_channels]
    }

    pub fn process(
        &mut self,
        transport_time: &[[f32; HOA_FRAME_SAMPLES]],
        config: HoaBitstreamConfig,
        side_info: HoaSideInfo,
        output: &mut [[f32; HOA_FRAME_SAMPLES]],
    ) -> Result<(), HoaPostSynthesisError> {
        let transport_channels = config.transport_channels();
        let output_channels = config.output_channels();
        if transport_time.len() != transport_channels {
            return Err(HoaPostSynthesisError::InvalidTransportChannelCount {
                expected: transport_channels,
                actual: transport_time.len(),
            });
        }
        if output.len() != output_channels {
            return Err(HoaPostSynthesisError::InvalidOutputChannelCount {
                expected: output_channels,
                actual: output.len(),
            });
        }
        if side_info.transport_channels() != transport_channels {
            return Err(HoaPostSynthesisError::SideInfoConfigurationMismatch);
        }
        let vector_channels = side_info.vector_channels();
        let residual_channels = config.residual_channels();
        if vector_channels + residual_channels > transport_channels {
            return Err(HoaPostSynthesisError::InvalidRecoveryLayout {
                vector_channels,
                residual_channels,
                transport_channels,
            });
        }

        self.analyze(transport_time)?;
        for channel in transport_channels..output_channels {
            self.spectra[channel].fill(0.0);
        }
        if side_info.spatial_analysis() {
            self.recover_spatial(
                vector_channels,
                residual_channels,
                output_channels,
                transport_channels,
            )?;
        }
        self.synthesize(output)?;

        for (delay, current) in self.analysis_delay[..transport_channels]
            .iter_mut()
            .zip(transport_time)
        {
            delay.copy_from_slice(current);
        }
        for basis in 0..vector_channels {
            self.delayed_basis_indices[0][basis] = self.delayed_basis_indices[1][basis];
            self.delayed_basis_indices[1][basis] = side_info.basis_indices()[basis];
        }
        self.last_transport_channels = transport_channels;
        self.last_output_channels = output_channels;
        Ok(())
    }

    fn analyze(&mut self, transport_time: &[[f32; HOA_FRAME_SAMPLES]]) -> Result<(), MdctError> {
        for (channel, current) in transport_time.iter().enumerate() {
            for subframe in 0..2 {
                for sample in 0..HOA_OVERLAP_SIZE {
                    let (left, right) = if subframe == 0 {
                        (
                            self.analysis_delay[channel][HOA_OVERLAP_SIZE + sample],
                            current[sample],
                        )
                    } else {
                        (current[sample], current[HOA_OVERLAP_SIZE + sample])
                    };
                    self.transform_signal[sample] = left * self.window[sample];
                    self.transform_signal[HOA_OVERLAP_SIZE + sample] =
                        right * self.window[HOA_OVERLAP_SIZE - 1 - sample];
                }
                let spectrum_start = subframe * HOA_OVERLAP_SIZE;
                self.mdct.process(
                    &self.transform_signal,
                    &mut self.spectra[channel][spectrum_start..spectrum_start + HOA_OVERLAP_SIZE],
                )?;
            }
        }
        Ok(())
    }

    fn recover_spatial(
        &mut self,
        vector_channels: usize,
        residual_channels: usize,
        output_channels: usize,
        transport_channels: usize,
    ) -> Result<(), HoaPostSynthesisError> {
        debug_assert!(vector_channels <= MAX_HOA_BASIS);
        debug_assert!(vector_channels + residual_channels <= transport_channels);
        for vector in 0..vector_channels {
            let coefficients =
                hoa_basis_coefficients(usize::from(self.delayed_basis_indices[0][vector]))?;
            for (output, &coefficient) in coefficients.iter().take(output_channels).enumerate() {
                self.basis_matrix[output][vector] = coefficient;
            }
        }

        for channel in &mut self.recovery[..output_channels] {
            channel.fill(0.0);
        }
        for sample in 0..HOA_FRAME_SAMPLES {
            for output in 0..output_channels {
                let mut value = 0.0_f32;
                for vector in 0..vector_channels {
                    value += self.spectra[vector][sample] * self.basis_matrix[output][vector];
                }
                self.recovery[output][sample] = value;
            }
        }
        for residual in 0..residual_channels {
            let source = vector_channels + residual;
            for sample in 0..HOA_FRAME_SAMPLES {
                self.recovery[residual][sample] += self.spectra[source][sample];
            }
        }
        for channel in 0..output_channels {
            self.spectra[channel].copy_from_slice(&self.recovery[channel]);
        }
        Ok(())
    }

    fn synthesize(&mut self, output: &mut [[f32; HOA_FRAME_SAMPLES]]) -> Result<(), ImdctError> {
        for (channel, channel_output) in output.iter_mut().enumerate() {
            channel_output.fill(0.0);
            for subframe in 0..2 {
                self.transform_signal.fill(0.0);
                let spectrum_start = subframe * HOA_OVERLAP_SIZE;
                self.transform_signal[..HOA_OVERLAP_SIZE].copy_from_slice(
                    &self.spectra[channel][spectrum_start..spectrum_start + HOA_OVERLAP_SIZE],
                );
                self.imdct.process(&mut self.transform_signal)?;

                let output_start = subframe * HOA_OVERLAP_SIZE;
                for sample in 0..HOA_OVERLAP_SIZE {
                    let left = self.transform_signal[sample] * self.window[sample];
                    let right = self.transform_signal[HOA_OVERLAP_SIZE + sample]
                        * self.window[HOA_OVERLAP_SIZE - 1 - sample];
                    channel_output[output_start + sample] =
                        left + self.synthesis_overlap[channel][sample];
                    self.synthesis_overlap[channel][sample] = right;
                }
            }
        }
        Ok(())
    }
}

impl Default for HoaPostSynthesis {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for HoaPostSynthesis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HoaPostSynthesis")
            .field("mdct", &self.mdct)
            .field("imdct", &self.imdct)
            .field("state_channels", &self.analysis_delay.len())
            .field("last_transport_channels", &self.last_transport_channels)
            .field("last_output_channels", &self.last_output_channels)
            .finish_non_exhaustive()
    }
}

fn hoa_angle_pair(index: usize) -> [usize; 2] {
    debug_assert!(index < HOA_BASIS_TABLE_LEN);
    let offset = index * 2 * core::mem::size_of::<i16>();
    [
        usize::try_from(i16::from_le_bytes([
            HOA_SPATIAL_TABLE_BYTES[offset],
            HOA_SPATIAL_TABLE_BYTES[offset + 1],
        ]))
        .expect("normative HOA azimuth index is nonnegative"),
        usize::try_from(i16::from_le_bytes([
            HOA_SPATIAL_TABLE_BYTES[offset + 2],
            HOA_SPATIAL_TABLE_BYTES[offset + 3],
        ]))
        .expect("normative HOA elevation index is nonnegative"),
    ]
}

fn sine_table(index: usize) -> f32 {
    debug_assert!(index < HOA_SIN_TABLE_LEN);
    let offset = HOA_ANGLE_BYTES + index * core::mem::size_of::<f32>();
    f32::from_le_bytes([
        HOA_SPATIAL_TABLE_BYTES[offset],
        HOA_SPATIAL_TABLE_BYTES[offset + 1],
        HOA_SPATIAL_TABLE_BYTES[offset + 2],
        HOA_SPATIAL_TABLE_BYTES[offset + 3],
    ])
}

fn quantized_sine(index: usize) -> f32 {
    debug_assert!(index <= HOA_SIN_FULL);
    if index <= HOA_SIN_QUARTER {
        sine_table(index)
    } else if index <= HOA_SIN_HALF {
        sine_table(HOA_SIN_HALF - index)
    } else if index <= HOA_SIN_THREE_QUARTERS {
        -sine_table(index - HOA_SIN_HALF)
    } else {
        -sine_table(HOA_SIN_FULL - index)
    }
}

fn quantized_cosine(index: usize) -> f32 {
    debug_assert!(index <= HOA_SIN_FULL);
    if index <= HOA_SIN_QUARTER {
        sine_table(HOA_SIN_QUARTER - index)
    } else if index <= HOA_SIN_HALF {
        -sine_table(index - HOA_SIN_QUARTER)
    } else if index <= HOA_SIN_THREE_QUARTERS {
        -sine_table(HOA_SIN_THREE_QUARTERS - index)
    } else {
        sine_table(index - HOA_SIN_THREE_QUARTERS)
    }
}

fn c_sqrt(value: f32) -> f32 {
    f64::from(value).sqrt() as f32
}

fn third_order_basis(
    sin_azimuth: f32,
    cos_azimuth: f32,
    sin_elevation: f32,
    cos_elevation: f32,
) -> [f32; HOA_OUTPUT_CHANNELS] {
    let r00 = c_sqrt(1.0_f32 / 4.0_f32 / AVS3_PI);
    let r01 = c_sqrt(3.0_f32 / 4.0_f32 / AVS3_PI);
    let r04 = c_sqrt(5.0_f32 / 16.0_f32 / AVS3_PI);
    let r05 = c_sqrt(15.0_f32 / 4.0_f32 / AVS3_PI);
    let r07 = c_sqrt(15.0_f32 / 16.0_f32 / AVS3_PI);
    let r09 = c_sqrt(7.0_f32 / 16.0_f32 / AVS3_PI);
    let r10 = c_sqrt(21.0_f32 / 32.0_f32 / AVS3_PI);
    let r12 = c_sqrt(105.0_f32 / 16.0_f32 / AVS3_PI);
    let r14 = c_sqrt(35.0_f32 / 32.0_f32 / AVS3_PI);

    let sin_azimuth_squared = sin_azimuth * sin_azimuth;
    let sin_elevation_squared = sin_elevation * sin_elevation;
    let cos_azimuth_squared = cos_azimuth * cos_azimuth;
    let cos_elevation_squared = cos_elevation * cos_elevation;
    let sin_cos_azimuth = sin_azimuth * cos_azimuth;
    let mut result = [0.0_f32; HOA_OUTPUT_CHANNELS];

    result[0] = r00;
    result[2] = r01 * sin_elevation;
    let mut temporary = r01 * cos_elevation;
    result[1] = temporary * sin_azimuth;
    result[3] = temporary * cos_azimuth;
    result[6] = r04 * (3.0_f32 * sin_elevation_squared - 1.0_f32);
    temporary = r05 * cos_elevation * sin_elevation;
    result[5] = temporary * sin_azimuth;
    result[7] = temporary * cos_azimuth;
    temporary = r07 * cos_elevation_squared;
    result[4] = temporary * 2.0_f32 * sin_cos_azimuth;
    result[8] = temporary * (2.0_f32 * cos_azimuth_squared - 1.0_f32);
    result[12] = r09 * (5.0_f32 * sin_elevation_squared * sin_elevation - 3.0_f32 * sin_elevation);
    temporary = r10 * cos_elevation * (5.0_f32 * sin_elevation_squared - 1.0_f32);
    result[11] = temporary * sin_azimuth;
    result[13] = temporary * cos_azimuth;
    temporary = r12 * cos_elevation_squared * sin_elevation;
    result[10] = temporary * 2.0_f32 * sin_cos_azimuth;
    result[14] = temporary * (2.0_f32 * cos_azimuth_squared - 1.0_f32);
    temporary = r14 * cos_elevation_squared * cos_elevation;
    result[9] = temporary * (3.0_f32 * sin_azimuth - 4.0_f32 * sin_azimuth_squared * sin_azimuth);
    result[15] = temporary * (4.0_f32 * cos_azimuth_squared * cos_azimuth - 3.0_f32 * cos_azimuth);
    result
}

fn fill_sine_window(window: &mut [f32]) {
    let scale = AVS3_PI / (2.0_f32 * window.len() as f32);
    for (index, value) in window.iter_mut().enumerate() {
        let angle = scale * (index as f32 + 0.5_f32);
        *value = f64::from(angle).sin() as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::{BitReader, BitWriter};
    use crate::header::{AudioCodecId, BitDepth, ChannelConfig, CodecProfile, FrameHeader, NnType};

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        })
    }

    fn header(order: u8, bitrate: u32, payload_bits: usize) -> FrameHeader {
        let channel_config = match order {
            1 => ChannelConfig::Hoa1,
            2 => ChannelConfig::Hoa2,
            3 => ChannelConfig::Hoa3,
            _ => panic!("unsupported HOA test order"),
        };
        let channels = (order + 1) * (order + 1);
        FrameHeader {
            codec_id: AudioCodecId::Avs3P3,
            nn_type: NnType::Main,
            profile: CodecProfile::Hoa,
            sample_rate: 48_000,
            bit_depth: BitDepth::Sixteen,
            channel_config: Some(channel_config),
            sound_bed_type: None,
            hoa_order: Some(order),
            objects: 0,
            bed_channels: channels,
            channels,
            has_lfe: false,
            bed_bitrate: None,
            object_bitrate: None,
            bitrate,
            crc: 0,
            header_len: 7,
            payload_bits,
            payload_len: payload_bits.div_ceil(8),
            frame_len: 7 + payload_bits.div_ceil(8),
            samples_per_channel: HOA_FRAME_SAMPLES as u32,
        }
    }

    fn side_info(
        config: HoaBitstreamConfig,
        spatial_analysis: bool,
        basis_indices: &[u16],
    ) -> HoaSideInfo {
        let mut writer = BitWriter::new();
        writer.write_bits(0, 4).unwrap();
        writer.write_bits(u64::from(spatial_analysis), 1).unwrap();
        if spatial_analysis {
            writer.write_bits(basis_indices.len() as u64, 4).unwrap();
        }
        let vector_channels = if spatial_analysis {
            basis_indices.len()
        } else if config.default_spatial_analysis() {
            config.foreground_channels()
        } else {
            0
        };
        for index in basis_indices.iter().copied().take(vector_channels) {
            writer.write_bits(u64::from(index), 12).unwrap();
        }
        for group in config.groups() {
            writer.write_bits(0, 4).unwrap();
            writer.write_bits(0, 4).unwrap();
            for _ in 0..group.channels() {
                writer.write_bits(1, 4).unwrap();
            }
        }
        let bit_len = writer.bit_len();
        let payload = writer.into_bytes();
        let mut reader = BitReader::with_bit_len(&payload, bit_len).unwrap();
        let side_info = HoaSideInfo::parse(&mut reader, config).unwrap();
        assert_eq!(reader.remaining(), 0);
        side_info
    }

    #[test]
    fn spatial_asset_has_normative_layout_and_fingerprint() {
        assert_eq!(fnv1a(hoa_spatial_table_bytes()), HOA_SPATIAL_TABLE_FNV1A);
        assert_eq!(hoa_angle_pair(0), [2, 768]);
        assert_eq!(hoa_angle_pair(1_339), [960, 128]);
        assert_eq!(hoa_angle_pair(1_340), [0, 0]);
        assert_eq!(hoa_angle_pair(1_342), [0, 0]);
        assert_eq!(sine_table(0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(sine_table(64).to_bits(), 0.382_683_f32.to_bits());
        assert_eq!(sine_table(256).to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn basis_coefficients_match_c_reference() {
        let references = [
            (
                0,
                [
                    0x3e90_6eba,
                    0x8000_0000,
                    0xbefa_2a1c,
                    0x8000_0000,
                    0x0000_0000,
                    0x0000_0000,
                    0x3f21_7b01,
                    0x0000_0000,
                    0x0000_0000,
                    0x8000_0000,
                    0x8000_0000,
                    0x8000_0000,
                    0xbf3f_10f8,
                    0x8000_0000,
                    0x8000_0000,
                    0x8000_0000,
                ],
            ),
            (
                512,
                [
                    0x3e90_6eba,
                    0xbe6b_36e4,
                    0x3d13_3a10,
                    0xbedc_0705,
                    0x3ee7_4c0a,
                    0xbd1a_c484,
                    0xbe9e_dbd9,
                    0xbd90_c65a,
                    0x3e9a_8c24,
                    0xbf15_1ae0,
                    0x3db4_12e0,
                    0x3e56_11c0,
                    0xbda7_25ec,
                    0x3ec8_3f63,
                    0x3d70_a472,
                    0xbd6a_f712,
                ],
            ),
            (
                1_340,
                [
                    0x3e90_6eba,
                    0x0000_0000,
                    0x0000_0000,
                    0x3efa_2a1c,
                    0x0000_0000,
                    0x0000_0000,
                    0xbea1_7b01,
                    0x0000_0000,
                    0x3f0b_d8a0,
                    0x0000_0000,
                    0x0000_0000,
                    0x8000_0000,
                    0x0000_0000,
                    0xbeea_01e8,
                    0x0000_0000,
                    0x3f17_0d18,
                ],
            ),
        ];
        for (index, expected) in references {
            assert_eq!(
                hoa_basis_coefficients(index).unwrap().map(f32::to_bits),
                expected,
                "basis index {index}"
            );
        }
    }

    #[test]
    fn zero_transport_stays_zero_and_basis_delay_advances_after_success() {
        let config = HoaBitstreamConfig::for_header(&header(3, 320_000, 0)).unwrap();
        let first_side = side_info(config, true, &[1, 512]);
        let second_side = side_info(config, true, &[127, 1_000]);
        let transport = vec![[0.0; HOA_FRAME_SAMPLES]; config.transport_channels()];
        let mut output = vec![[7.0; HOA_FRAME_SAMPLES]; config.output_channels()];
        let mut synthesis = HoaPostSynthesis::new();

        synthesis
            .process(&transport, config, first_side, &mut output)
            .unwrap();
        assert!(output.iter().flatten().all(|&sample| sample == 0.0));
        assert_eq!(synthesis.delayed_basis_indices()[0][..2], [0, 0]);
        assert_eq!(synthesis.delayed_basis_indices()[1][..2], [1, 512]);

        synthesis
            .process(&transport, config, second_side, &mut output)
            .unwrap();
        assert_eq!(synthesis.delayed_basis_indices()[0][..2], [1, 512]);
        assert_eq!(synthesis.delayed_basis_indices()[1][..2], [127, 1_000]);
    }

    #[test]
    fn channel_mismatch_does_not_advance_state_or_touch_output() {
        let config = HoaBitstreamConfig::for_header(&header(1, 96_000, 0)).unwrap();
        let side = side_info(config, false, &[]);
        let transport = vec![[0.0; HOA_FRAME_SAMPLES]; 3];
        let mut output = vec![[7.0; HOA_FRAME_SAMPLES]; config.output_channels()];
        let mut synthesis = HoaPostSynthesis::new();
        assert_eq!(
            synthesis
                .process(&transport, config, side, &mut output)
                .unwrap_err(),
            HoaPostSynthesisError::InvalidTransportChannelCount {
                expected: 4,
                actual: 3,
            }
        );
        assert!(output.iter().flatten().all(|&sample| sample == 7.0));
        assert_eq!(synthesis.delayed_basis_indices(), &[[0; 4]; 2]);
    }

    #[test]
    fn three_frame_post_filter_timing_matches_c_reference() {
        let config = HoaBitstreamConfig::for_header(&header(1, 96_000, 0)).unwrap();
        let side = side_info(config, false, &[]);
        let mut transport = vec![[0.0; HOA_FRAME_SAMPLES]; config.transport_channels()];
        let mut output = vec![[0.0; HOA_FRAME_SAMPLES]; config.output_channels()];
        let mut synthesis = HoaPostSynthesis::new();
        let positions = [
            0, 1, 2, 127, 128, 255, 256, 511, 512, 768, 1_021, 1_022, 1_023,
        ];
        let references = [
            [
                0x3139_5a99,
                0xb162_31a2,
                0x32a6_e53f,
                0xb42a_ce9c,
                0xb598_ae09,
                0xb57c_1749,
                0x357c_dd96,
                0xb5eb_ffee,
                0x3f6f_b86d,
                0x3f1e_0d9d,
                0x3f08_973b,
                0x3f24_c42f,
                0xbf08_3eca,
            ],
            [
                0x3f7e_11d0,
                0x3f74_41ab,
                0xbf2c_f55e,
                0x3f64_5373,
                0x3f46_65f8,
                0xbf62_fcaa,
                0x3f45_032b,
                0x3f01_e5bc,
                0x3f56_6366,
                0xbf0e_e1da,
                0x3f66_95f4,
                0x3f65_ff9e,
                0x3f67_ca87,
            ],
            [
                0xbf4c_6913,
                0x3f51_a74a,
                0x3f63_52e7,
                0xbf09_a90c,
                0x3f14_9908,
                0xbf3d_cc97,
                0xbf7d_8c1d,
                0x3f5f_87f9,
                0x3f1f_1639,
                0xbf65_9361,
                0xbf13_9bc6,
                0x3f2e_fb2a,
                0xbf71_9e86,
            ],
        ];
        let mut random_state = 0xa511_e9b3_u32;
        for (frame, expected) in references.into_iter().enumerate() {
            for sample in &mut transport[0] {
                random_state ^= random_state << 13;
                random_state ^= random_state >> 17;
                random_state ^= random_state << 5;
                let bits =
                    ((random_state & 1) << 31) | 0x3f00_0000 | ((random_state >> 1) & 0x007f_ffff);
                *sample = f32::from_bits(bits);
            }
            synthesis
                .process(&transport, config, side, &mut output)
                .unwrap();
            for (position, expected_bits) in positions.into_iter().zip(expected) {
                let reference = f32::from_bits(expected_bits);
                let error = (output[0][position] - reference).abs();
                assert!(
                    error <= 2.0e-5,
                    "frame {frame}, sample {position}: Rust={} C={reference}, error={error}",
                    output[0][position]
                );
            }
            assert!(output[1..].iter().flatten().all(|&sample| sample == 0.0));
        }
    }
}
