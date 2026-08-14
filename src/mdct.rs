use core::fmt;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

const SHORT_MDCT_LEN: usize = 256;
const HALF_LONG_MDCT_LEN: usize = 1_024;
const LONG_MDCT_LEN: usize = 2_048;
const MAX_COMPLEX_FFT_LEN: usize = LONG_MDCT_LEN / 4;
const AVS3_PI: f32 = core::f32::consts::PI;
const TWIDDLE_OFFSET: f32 = 0.125;
const INVERSE_MDCT_NORMALIZATION: f32 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MdctError {
    UnsupportedLength(usize),
    InvalidOutputLength { expected: usize, actual: usize },
}

impl fmt::Display for MdctError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedLength(length) => write!(
                f,
                "AVS3 MDCT length {length} is unsupported; expected 256, 1024, or 2048"
            ),
            Self::InvalidOutputLength { expected, actual } => write!(
                f,
                "AVS3 MDCT output has {actual} coefficients; expected {expected}"
            ),
        }
    }
}

impl std::error::Error for MdctError {}

/// SIMD-capable AVS3 forward MDCT backed by `rustfft`.
///
/// The input folding, twiddles, coefficient permutation and normalization are
/// the codec's reference operations. Only the internal complex FFT is
/// delegated to RustFFT. Plans and work buffers are allocated at construction;
/// [`Self::process`] does not allocate or mutate its input.
pub struct FastMdct {
    fft_64: Arc<dyn Fft<f32>>,
    fft_256: Arc<dyn Fft<f32>>,
    fft_512: Arc<dyn Fft<f32>>,
    work: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl FastMdct {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft_64 = planner.plan_fft_forward(SHORT_MDCT_LEN / 4);
        let fft_256 = planner.plan_fft_forward(HALF_LONG_MDCT_LEN / 4);
        let fft_512 = planner.plan_fft_forward(LONG_MDCT_LEN / 4);
        let scratch_len = [
            fft_64.get_inplace_scratch_len(),
            fft_256.get_inplace_scratch_len(),
            fft_512.get_inplace_scratch_len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        Self {
            fft_64,
            fft_256,
            fft_512,
            work: vec![Complex::new(0.0, 0.0); MAX_COMPLEX_FFT_LEN],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
        }
    }

    /// Transform `N` time-domain samples into `N/2` MDCT coefficients.
    pub fn process(&mut self, signal: &[f32], output: &mut [f32]) -> Result<(), MdctError> {
        let length = signal.len();
        let fft = match length {
            SHORT_MDCT_LEN => &self.fft_64,
            HALF_LONG_MDCT_LEN => &self.fft_256,
            LONG_MDCT_LEN => &self.fft_512,
            _ => return Err(MdctError::UnsupportedLength(length)),
        };
        let expected_output = length / 2;
        if output.len() != expected_output {
            return Err(MdctError::InvalidOutputLength {
                expected: expected_output,
                actual: output.len(),
            });
        }

        let fft_len = length / 4;
        let work = &mut self.work[..fft_len];
        let frequency = 2.0_f32 * AVS3_PI / length as f32;
        let cosine_step = f64::from(frequency).cos() as f32;
        let sine_step = f64::from(frequency).sin() as f32;
        let first_angle = frequency * TWIDDLE_OFFSET;
        let first_cosine = f64::from(first_angle).cos() as f32;
        let first_sine = f64::from(first_angle).sin() as f32;

        let mut cosine = first_cosine;
        let mut sine = first_sine;
        for (index, value) in work.iter_mut().enumerate() {
            let folded_index = length / 2 - 1 - 2 * index;
            let temporary_real = if index < length / 8 {
                signal[length / 4 + folded_index] + signal[length + length / 4 - 1 - folded_index]
            } else {
                signal[length / 4 + folded_index] - signal[length / 4 - 1 - folded_index]
            };

            let unfolded_index = 2 * index;
            let temporary_imaginary = if index < length / 8 {
                signal[length / 4 + unfolded_index] - signal[length / 4 - 1 - unfolded_index]
            } else {
                signal[length / 4 + unfolded_index]
                    + signal[length + length / 4 - 1 - unfolded_index]
            };

            value.re = temporary_real * cosine + temporary_imaginary * sine;
            value.im = temporary_imaginary * cosine - temporary_real * sine;

            let old_cosine = cosine;
            cosine = cosine * cosine_step - sine * sine_step;
            sine = sine * cosine_step + old_cosine * sine_step;
        }

        let required_scratch = fft.get_inplace_scratch_len();
        fft.process_with_scratch(work, &mut self.scratch[..required_scratch]);

        cosine = first_cosine;
        sine = first_sine;
        let output_scale = (length as f64).sqrt();
        for (index, value) in work.iter().enumerate() {
            let temporary_real = INVERSE_MDCT_NORMALIZATION * (value.re * cosine + value.im * sine);
            let temporary_imaginary =
                INVERSE_MDCT_NORMALIZATION * (value.im * cosine - value.re * sine);

            output[2 * index] = (-f64::from(temporary_real) / output_scale) as f32;
            output[length / 2 - 1 - 2 * index] =
                (f64::from(temporary_imaginary) / output_scale) as f32;

            let old_cosine = cosine;
            cosine = cosine * cosine_step - sine * sine_step;
            sine = sine * cosine_step + old_cosine * sine_step;
        }
        Ok(())
    }
}

impl Default for FastMdct {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for FastMdct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FastMdct")
            .field("cached_fft_lengths", &[64, 256, 512])
            .field("work_values", &self.work.len())
            .field("scratch_values", &self.scratch.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imdct::FastImdct;

    fn deterministic_coefficients(length: usize) -> Vec<f32> {
        let mut coefficients = vec![0.0_f32; length / 2];
        let mut state = 0x1bad_f00d_u32;
        for value in &mut coefficients {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let bits = ((state & 1) << 31) | 0x3f00_0000 | ((state >> 1) & 0x007f_ffff);
            *value = f32::from_bits(bits);
        }
        coefficients
    }

    fn deterministic_signal(length: usize) -> Vec<f32> {
        let mut signal = vec![0.0_f32; length];
        let mut state = 0x0ddc_0ffe_u32;
        for value in &mut signal {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let bits = ((state & 1) << 31) | 0x3f00_0000 | ((state >> 1) & 0x007f_ffff);
            *value = f32::from_bits(bits);
        }
        signal
    }

    fn reference_positions(length: usize) -> [usize; 13] {
        let coefficients = length / 2;
        [
            0,
            1,
            2,
            coefficients / 8 - 1,
            coefficients / 8,
            coefficients / 4 - 1,
            coefficients / 4,
            coefficients / 2 - 1,
            coefficients / 2,
            3 * coefficients / 4,
            coefficients - 3,
            coefficients - 2,
            coefficients - 1,
        ]
    }

    fn assert_close_to_c(length: usize, expected_bits: [u32; 13], tolerance: f32) {
        let signal = deterministic_signal(length);
        let mut output = vec![0.0_f32; length / 2];
        FastMdct::new().process(&signal, &mut output).unwrap();
        for (position, expected_bits) in reference_positions(length).into_iter().zip(expected_bits)
        {
            let expected = f32::from_bits(expected_bits);
            let error = (output[position] - expected).abs();
            assert!(
                error <= tolerance,
                "N={length} position {position}: Rust={} C={expected} error={error}",
                output[position]
            );
        }
    }

    #[test]
    fn mdct_recovers_coefficients_from_the_matching_imdct() {
        let mut mdct = FastMdct::new();
        let mut imdct = FastImdct::new();
        for length in [SHORT_MDCT_LEN, HALF_LONG_MDCT_LEN, LONG_MDCT_LEN] {
            let expected = deterministic_coefficients(length);
            let mut signal = vec![0.0_f32; length];
            signal[..length / 2].copy_from_slice(&expected);
            imdct.process(&mut signal).unwrap();

            let mut actual = vec![0.0_f32; length / 2];
            mdct.process(&signal, &mut actual).unwrap();
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let expected = 2.0 * expected;
                let error = (actual - expected).abs();
                assert!(
                    error <= 4.0e-5,
                    "N={length} coefficient {index}: actual={actual} expected={expected} error={error}"
                );
            }
        }
    }

    #[test]
    fn supported_sizes_stay_close_to_c_reference_fft() {
        assert_close_to_c(
            SHORT_MDCT_LEN,
            [
                0x4008_5c41,
                0xbf43_54a5,
                0xbcad_84a1,
                0xbef6_ef20,
                0x3ed4_a39c,
                0x3e6b_8b78,
                0xbdf5_59c0,
                0xbf03_a43c,
                0xbfde_31a8,
                0x3fcf_46b1,
                0x3f3e_b484,
                0xbf1e_885d,
                0x3ef7_f0cb,
            ],
            2.0e-5,
        );
        assert_close_to_c(
            HALF_LONG_MDCT_LEN,
            [
                0xbf62_a907,
                0xbf93_8000,
                0xbf61_0563,
                0xbf95_7ada,
                0x3fcb_8f94,
                0x3e1b_84f8,
                0x3eec_c04c,
                0x3e78_cdee,
                0x3f3e_71fe,
                0x400d_5dd2,
                0xbfe6_68c3,
                0x3e16_a246,
                0xbfad_2eae,
            ],
            5.0e-5,
        );
        assert_close_to_c(
            LONG_MDCT_LEN,
            [
                0xbf87_5e04,
                0x4009_d784,
                0xbf29_0db5,
                0xbf05_51c8,
                0xbefc_b434,
                0x3ee1_f528,
                0xbd67_6997,
                0x3ec1_45a7,
                0xbf54_eeb8,
                0x3fd7_7876,
                0x3e62_ce94,
                0x3f85_28d1,
                0xbebf_2d0f,
            ],
            8.0e-5,
        );
    }

    #[test]
    fn rejects_bad_lengths_before_writing_output() {
        let mut mdct = FastMdct::new();
        let mut output = [7.0_f32; 17];
        assert_eq!(
            mdct.process(&[0.0; 512], &mut output).unwrap_err(),
            MdctError::UnsupportedLength(512)
        );
        assert_eq!(output, [7.0; 17]);

        assert_eq!(
            mdct.process(&[0.0; HALF_LONG_MDCT_LEN], &mut output)
                .unwrap_err(),
            MdctError::InvalidOutputLength {
                expected: HALF_LONG_MDCT_LEN / 2,
                actual: output.len(),
            }
        );
        assert_eq!(output, [7.0; 17]);
    }
}
