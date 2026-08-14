use core::fmt;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::core_side::{LsfCodebookMode, LsfSideInfo};
use crate::model::AVS3_FEATURE_DIMENSIONS;

const LPC_ORDER: usize = 16;
const FFT_LEN: usize = 512;
const INTERPOLATION: usize = AVS3_FEATURE_DIMENSIONS / (FFT_LEN / 2);
const LSF_MIN_GAP: f32 = 50.0;
const NYQUIST_FREQUENCY: f32 = 24_000.0;
const GAMMA_LPC: f32 = 0.94;
const AVS3_PI: f32 = core::f32::consts::PI;

pub const FD_TABLE_VALUES: usize = 10_992;
pub const FD_TABLE_BYTES_LEN: usize = FD_TABLE_VALUES * 4;
pub const FD_TABLE_FNV1A: u64 = 0x9ce2_64f0_19b7_5cc4;

const FD_TABLE_BYTES: &[u8; FD_TABLE_BYTES_LEN] = include_bytes!("../assets/avs3a_fd_tables.bin");

const MEAN_LSF: usize = 0;
const HBR_STAGE1_CB1: usize = 16;
const HBR_STAGE1_CB2: usize = 2_320;
const HBR_STAGE2_CB1: usize = 4_112;
const HBR_STAGE2_CB2: usize = 4_496;
const HBR_STAGE2_CB3: usize = 4_880;
const HBR_STAGE2_CB4: usize = 5_072;
const HBR_STAGE2_CB5: usize = 5_168;
const LBR_STAGE1_CB1: usize = 5_296;
const LBR_STAGE1_CB2: usize = 7_600;
const LBR_STAGE2_CB1: usize = 9_392;
const LBR_STAGE2_CB2: usize = 10_032;
const LBR_STAGE2_CB3: usize = 10_544;

const SFB_BORDERS: [usize; 50] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 1_024,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FdShapingError {
    InvalidSpectrumLength {
        expected: usize,
        actual: usize,
    },
    InvalidLsfCodebookCount {
        mode: LsfCodebookMode,
        expected: usize,
        actual: usize,
    },
    InvalidLsfCodebookIndex {
        codebook: usize,
        index: u16,
        entries: usize,
    },
}

impl fmt::Display for FdShapingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpectrumLength { expected, actual } => write!(
                f,
                "FD-shaping spectrum has {actual} lines; expected {expected}"
            ),
            Self::InvalidLsfCodebookCount {
                mode,
                expected,
                actual,
            } => write!(
                f,
                "{mode:?} LSF side information has {actual} codebooks; expected {expected}"
            ),
            Self::InvalidLsfCodebookIndex {
                codebook,
                index,
                entries,
            } => write!(
                f,
                "LSF codebook {codebook} index {index} is outside {entries} entries"
            ),
        }
    }
}

impl std::error::Error for FdShapingError {}

/// Fast inverse frequency-domain LPC shaping for one AVS3 core spectrum.
///
/// Normative LSF codebooks are read directly from the bundled little-endian
/// table asset. RustFFT replaces the C radix-2 FFT used to evaluate the LPC
/// response; all FFT plans and scratch buffers are cached at construction and
/// the per-frame path does not allocate.
pub struct FdSpectrumShaping {
    fft: Arc<dyn Fft<f32>>,
    fft_buffer: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    raw_gain: [f32; FFT_LEN],
    interpolated_gain: [f32; AVS3_FEATURE_DIMENSIONS],
}

impl FdSpectrumShaping {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_LEN);
        let scratch_len = fft.get_inplace_scratch_len();
        Self {
            fft,
            fft_buffer: vec![Complex::new(0.0, 0.0); FFT_LEN],
            fft_scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            raw_gain: [0.0; FFT_LEN],
            interpolated_gain: [0.0; AVS3_FEATURE_DIMENSIONS],
        }
    }

    pub fn apply(
        &mut self,
        lsf_side_info: LsfSideInfo,
        spectrum: &mut [f32],
    ) -> Result<(), FdShapingError> {
        if spectrum.len() != AVS3_FEATURE_DIMENSIONS {
            return Err(FdShapingError::InvalidSpectrumLength {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: spectrum.len(),
            });
        }
        validate_lsf_indexes(lsf_side_info)?;

        let quantized_lsf = decode_lsf(lsf_side_info);
        let lsp = lsf_to_lsp(&quantized_lsf);
        let lpc = lsp_to_lpc(&lsp);
        self.calculate_gain(&lpc);
        for (value, &gain) in spectrum.iter_mut().zip(&self.interpolated_gain) {
            *value *= gain;
        }
        Ok(())
    }

    pub fn gain(&self) -> &[f32; AVS3_FEATURE_DIMENSIONS] {
        &self.interpolated_gain
    }

    fn calculate_gain(&mut self, lpc: &[f32; LPC_ORDER + 1]) {
        let mut weighting = 1.0_f32;
        for (index, value) in self.fft_buffer.iter_mut().enumerate() {
            if index <= LPC_ORDER {
                let weighted_lpc = lpc[index] * weighting;
                let angle = index as f32 * AVS3_PI / FFT_LEN as f32;
                value.re = weighted_lpc * (f64::from(angle).cos() as f32);
                value.im = -weighted_lpc * (f64::from(angle).sin() as f32);
                weighting *= GAMMA_LPC;
            } else {
                *value = Complex::new(0.0, 0.0);
            }
        }

        let scratch_len = self.fft.get_inplace_scratch_len();
        self.fft
            .process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch[..scratch_len]);
        for (destination, value) in self.raw_gain.iter_mut().zip(&self.fft_buffer) {
            let magnitude_squared = value.re * value.re + value.im * value.im;
            *destination = (1.0_f64 / f64::from(magnitude_squared).sqrt()) as f32;
        }

        for index in 0..FFT_LEN / 2 {
            let base = self.raw_gain[index];
            let step = (self.raw_gain[index + 1] - base) / INTERPOLATION as f32;
            let output = index * INTERPOLATION;
            self.interpolated_gain[output] = base;
            self.interpolated_gain[output + 1] = base + step;
            self.interpolated_gain[output + 2] = base + 2.0_f32 * step;
            self.interpolated_gain[output + 3] = base + 3.0_f32 * step;
        }

        for band in 0..SFB_BORDERS.len() - 1 {
            let start = SFB_BORDERS[band];
            let stop = SFB_BORDERS[band + 1];
            let mut average = 0.0_f32;
            for &value in &self.interpolated_gain[start..stop] {
                average += value;
            }
            average /= (stop - start) as f32;
            self.interpolated_gain[start..stop].fill(average);
        }
    }
}

impl Default for FdSpectrumShaping {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for FdSpectrumShaping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FdSpectrumShaping")
            .field("fft_length", &FFT_LEN)
            .field("scratch_values", &self.fft_scratch.len())
            .finish_non_exhaustive()
    }
}

pub fn fd_table_bytes() -> &'static [u8; FD_TABLE_BYTES_LEN] {
    FD_TABLE_BYTES
}

fn validate_lsf_indexes(side_info: LsfSideInfo) -> Result<(), FdShapingError> {
    let limits: &[usize] = match side_info.mode() {
        LsfCodebookMode::HighBitrate => &[256, 256, 128, 128, 64, 32, 32],
        LsfCodebookMode::LowBitrate => &[256, 256, 128, 128, 64],
    };
    if side_info.indexes().len() != limits.len() {
        return Err(FdShapingError::InvalidLsfCodebookCount {
            mode: side_info.mode(),
            expected: limits.len(),
            actual: side_info.indexes().len(),
        });
    }
    for (codebook, (&index, &entries)) in side_info.indexes().iter().zip(limits).enumerate() {
        if usize::from(index) >= entries {
            return Err(FdShapingError::InvalidLsfCodebookIndex {
                codebook,
                index,
                entries,
            });
        }
    }
    Ok(())
}

fn decode_lsf(side_info: LsfSideInfo) -> [f32; LPC_ORDER] {
    let indexes = side_info.indexes();
    let mut decoded = [0.0_f32; LPC_ORDER];
    match side_info.mode() {
        LsfCodebookMode::HighBitrate => {
            for (index, value) in decoded[..9].iter_mut().enumerate() {
                *value = table_value(HBR_STAGE1_CB1, usize::from(indexes[0]) * 9 + index);
            }
            for (index, value) in decoded[9..].iter_mut().enumerate() {
                *value = table_value(HBR_STAGE1_CB2, usize::from(indexes[1]) * 7 + index);
            }
            for index in 0..3 {
                decoded[index] += table_value(HBR_STAGE2_CB1, usize::from(indexes[2]) * 3 + index);
                decoded[3 + index] +=
                    table_value(HBR_STAGE2_CB2, usize::from(indexes[3]) * 3 + index);
                decoded[6 + index] +=
                    table_value(HBR_STAGE2_CB3, usize::from(indexes[4]) * 3 + index);
                decoded[9 + index] +=
                    table_value(HBR_STAGE2_CB4, usize::from(indexes[5]) * 3 + index);
            }
            for index in 0..4 {
                decoded[12 + index] +=
                    table_value(HBR_STAGE2_CB5, usize::from(indexes[6]) * 4 + index);
            }
        }
        LsfCodebookMode::LowBitrate => {
            for (index, value) in decoded[..9].iter_mut().enumerate() {
                *value = table_value(LBR_STAGE1_CB1, usize::from(indexes[0]) * 9 + index);
            }
            for (index, value) in decoded[9..].iter_mut().enumerate() {
                *value = table_value(LBR_STAGE1_CB2, usize::from(indexes[1]) * 7 + index);
            }
            for (index, value) in decoded[..5].iter_mut().enumerate() {
                *value += table_value(LBR_STAGE2_CB1, usize::from(indexes[2]) * 5 + index);
            }
            for index in 0..4 {
                decoded[5 + index] +=
                    table_value(LBR_STAGE2_CB2, usize::from(indexes[3]) * 4 + index);
            }
            for index in 0..7 {
                decoded[9 + index] +=
                    table_value(LBR_STAGE2_CB3, usize::from(indexes[4]) * 7 + index);
            }
        }
    }

    let mut output = [0.0_f32; LPC_ORDER];
    for index in 0..LPC_ORDER {
        output[index] = decoded[index] + table_value(MEAN_LSF, index);
    }
    let mut minimum = LSF_MIN_GAP;
    for value in &mut output {
        if *value < minimum {
            *value = minimum;
        }
        minimum = *value + LSF_MIN_GAP;
    }
    let mut maximum = NYQUIST_FREQUENCY - LSF_MIN_GAP;
    for value in output.iter_mut().rev() {
        if *value > maximum {
            *value = maximum;
        }
        maximum = *value - LSF_MIN_GAP;
    }
    output
}

fn lsf_to_lsp(lsf: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER] {
    let mut lsp = [0.0_f32; LPC_ORDER];
    for (destination, &frequency) in lsp.iter_mut().zip(lsf) {
        let angle = frequency * AVS3_PI / NYQUIST_FREQUENCY;
        *destination = f64::from(angle).cos() as f32;
    }
    lsp
}

fn lsp_to_lpc(lsp: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER + 1] {
    let mut polynomial_1 = lsp_polynomial(lsp, 0);
    let mut polynomial_2 = lsp_polynomial(lsp, 1);
    for index in (1..=LPC_ORDER / 2).rev() {
        polynomial_1[index] += polynomial_1[index - 1];
        polynomial_2[index] -= polynomial_2[index - 1];
    }

    let mut lpc = [0.0_f32; LPC_ORDER + 1];
    lpc[0] = 1.0;
    for index in 0..LPC_ORDER / 2 {
        lpc[1 + index] = 0.5_f32 * (polynomial_1[1 + index] + polynomial_2[1 + index]);
        lpc[LPC_ORDER - index] = 0.5_f32 * (polynomial_1[1 + index] - polynomial_2[1 + index]);
    }
    lpc
}

fn lsp_polynomial(lsp: &[f32; LPC_ORDER], first_root: usize) -> [f32; LPC_ORDER / 2 + 1] {
    let mut polynomial = [0.0_f32; LPC_ORDER / 2 + 1];
    polynomial[0] = 1.0;
    let mut root = first_root;
    let mut factor = -2.0_f32 * lsp[root];
    polynomial[1] = factor;

    for order in 2..=LPC_ORDER / 2 {
        root += 2;
        factor = -2.0_f32 * lsp[root];
        polynomial[order] = factor * polynomial[order - 1] + 2.0_f32 * polynomial[order - 2];
        for index in (2..order).rev() {
            polynomial[index] += factor * polynomial[index - 1] + polynomial[index - 2];
        }
        polynomial[1] += factor;
    }
    polynomial
}

#[inline]
fn table_value(table_offset: usize, index: usize) -> f32 {
    let byte = (table_offset + index) * 4;
    f32::from_le_bytes([
        FD_TABLE_BYTES[byte],
        FD_TABLE_BYTES[byte + 1],
        FD_TABLE_BYTES[byte + 2],
        FD_TABLE_BYTES[byte + 3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BweConfig, CoreBitstreamConfig, MonoSideInfoDecoder, NnType};

    const HBR_PAYLOAD: [u8; 35] = [
        0x44, 0x72, 0x61, 0x63, 0xb6, 0x23, 0xa0, 0xf0, 0xea, 0x00, 0xfb, 0xdc, 0x10, 0x30, 0x2f,
        0x40, 0x2a, 0x40, 0xc0, 0xff, 0x6f, 0x01, 0xc7, 0xe9, 0x5f, 0x03, 0x84, 0xa0, 0xd8, 0x7f,
        0xfd, 0x51, 0xf6, 0xf2, 0x00,
    ];
    const LBR_PAYLOAD: [u8; 16] = [
        0x80, 0xfe, 0xa0, 0x8c, 0xcd, 0x1f, 0xa9, 0x5b, 0xa0, 0x42, 0x46, 0x8a, 0xcf, 0x13, 0x57,
        0x80,
    ];

    fn lsf_side_info(mode: LsfCodebookMode) -> LsfSideInfo {
        let (payload, payload_bits, nn_type, bwe) = match mode {
            LsfCodebookMode::HighBitrate => (
                HBR_PAYLOAD.as_slice(),
                277,
                NnType::Main,
                BweConfig::for_mono_bitrate(64_000).unwrap(),
            ),
            LsfCodebookMode::LowBitrate => {
                (LBR_PAYLOAD.as_slice(), 126, NnType::LowComplexity, None)
            }
        };
        let config = CoreBitstreamConfig::new(nn_type, payload_bits, mode, bwe).unwrap();
        let mut decoder = MonoSideInfoDecoder::new();
        decoder.parse(payload, config).unwrap().core().lsf()
    }

    fn deterministic_spectrum(seed: u32) -> [f32; AVS3_FEATURE_DIMENSIONS] {
        let mut spectrum = [0.0_f32; AVS3_FEATURE_DIMENSIONS];
        let mut state = seed;
        for value in &mut spectrum {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let bits = ((state & 1) << 31) | 0x3f00_0000 | ((state >> 1) & 0x007f_ffff);
            *value = f32::from_bits(bits);
        }
        spectrum
    }

    fn assert_close_to_c(mode: LsfCodebookMode, seed: u32, expected_bits: [u32; 19]) {
        let mut spectrum = deterministic_spectrum(seed);
        FdSpectrumShaping::new()
            .apply(lsf_side_info(mode), &mut spectrum)
            .unwrap();
        let positions = [
            0, 3, 4, 39, 40, 95, 96, 107, 108, 239, 240, 511, 512, 831, 832, 927, 928, 1000, 1023,
        ];
        for (position, bits) in positions.into_iter().zip(expected_bits) {
            let expected = f32::from_bits(bits);
            let error = (spectrum[position] - expected).abs();
            let tolerance = 2.0e-5_f32 * expected.abs().max(1.0);
            assert!(
                error <= tolerance,
                "{mode:?} position {position}: Rust={} C={expected} error={error} tolerance={tolerance}",
                spectrum[position]
            );
        }
    }

    #[test]
    fn hbr_and_lbr_shaping_stay_close_to_c_reference_fft() {
        assert_close_to_c(
            LsfCodebookMode::HighBitrate,
            0x51f1_5e01,
            [
                0xc1cc_76f8,
                0x4211_2a15,
                0x4205_ac81,
                0xc1bf_ad8f,
                0x4150_2eeb,
                0xc0d4_345e,
                0x40e4_13b5,
                0xc0bc_d789,
                0xc059_166d,
                0x3f19_00e7,
                0x3f08_40fd,
                0x3ec9_7e54,
                0xbe6d_3064,
                0xbef7_0156,
                0xbf12_7dc8,
                0xbeef_2d03,
                0x3e82_971b,
                0x3e88_82f0,
                0x3e81_25a3,
            ],
        );
        assert_close_to_c(
            LsfCodebookMode::LowBitrate,
            0x51f1_5e02,
            [
                0x4027_6f79,
                0xc042_7e92,
                0xc03f_1814,
                0xc0b4_b3e8,
                0xc086_8f67,
                0xc083_6458,
                0x4080_312c,
                0x404e_307d,
                0x4047_cabd,
                0x403a_eb2b,
                0x3fcf_62b4,
                0xbf7c_b658,
                0xbf4f_0498,
                0x3e3e_c048,
                0x3e00_1124,
                0xbdf0_8135,
                0x3da8_310a,
                0xbdc0_507a,
                0xbdd2_5ff5,
            ],
        );
    }

    #[test]
    fn bundled_table_has_expected_fingerprint() {
        let fingerprint = fd_table_bytes()
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
            });
        assert_eq!(fingerprint, FD_TABLE_FNV1A);
        assert_eq!(fd_table_bytes().len(), FD_TABLE_BYTES_LEN);
    }

    #[test]
    fn rejects_wrong_spectrum_length_before_mutating() {
        let mut spectrum = [3.0_f32; 17];
        let error = FdSpectrumShaping::new()
            .apply(lsf_side_info(LsfCodebookMode::HighBitrate), &mut spectrum)
            .unwrap_err();
        assert_eq!(
            error,
            FdShapingError::InvalidSpectrumLength {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: 17,
            }
        );
        assert_eq!(spectrum, [3.0; 17]);
    }
}
