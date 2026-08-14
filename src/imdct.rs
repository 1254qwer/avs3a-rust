use core::fmt;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

const SHORT_IMDCT_LEN: usize = 256;
const HALF_LONG_IMDCT_LEN: usize = 1_024;
const LONG_IMDCT_LEN: usize = 2_048;
const MAX_COMPLEX_FFT_LEN: usize = LONG_IMDCT_LEN / 4;
const AVS3_PI: f32 = core::f32::consts::PI;
const TWIDDLE_OFFSET: f32 = 0.125;
const MDCT_NORMALIZATION: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImdctError {
    UnsupportedLength(usize),
}

impl fmt::Display for ImdctError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedLength(length) => write!(
                f,
                "AVS3 IMDCT length {length} is unsupported; expected 256, 1024, or 2048"
            ),
        }
    }
}

impl std::error::Error for ImdctError {}

/// SIMD-capable AVS3 IMDCT backed by `rustfft`.
///
/// The codec-specific pre/post twiddles, output permutation and normalization
/// retain the reference algorithm. Only the internal complex IFFT is replaced
/// with RustFFT, so small final-bit differences from the scalar C radix-2 FFT
/// are expected. Plans and all work buffers are allocated once in [`Self::new`];
/// [`Self::process`] performs no allocation.
pub struct FastImdct {
    fft_64: Arc<dyn Fft<f32>>,
    fft_256: Arc<dyn Fft<f32>>,
    fft_512: Arc<dyn Fft<f32>>,
    work: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl FastImdct {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft_64 = planner.plan_fft_inverse(SHORT_IMDCT_LEN / 4);
        let fft_256 = planner.plan_fft_inverse(HALF_LONG_IMDCT_LEN / 4);
        let fft_512 = planner.plan_fft_inverse(LONG_IMDCT_LEN / 4);
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

    /// Transform `N/2` coefficients in the first half of `signal` into `N`
    /// time-domain samples in place, matching the C decoder's buffer contract.
    pub fn process(&mut self, signal: &mut [f32]) -> Result<(), ImdctError> {
        let length = signal.len();
        let fft = match length {
            SHORT_IMDCT_LEN => &self.fft_64,
            HALF_LONG_IMDCT_LEN => &self.fft_256,
            LONG_IMDCT_LEN => &self.fft_512,
            _ => return Err(ImdctError::UnsupportedLength(length)),
        };
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
            let temporary_real = -signal[2 * index];
            let temporary_imaginary = signal[length / 2 - 1 - 2 * index];
            value.re = temporary_real * cosine - temporary_imaginary * sine;
            value.im = temporary_imaginary * cosine + temporary_real * sine;

            let old_cosine = cosine;
            cosine = cosine * cosine_step - sine * sine_step;
            sine = sine * cosine_step + old_cosine * sine_step;
        }

        let required_scratch = fft.get_inplace_scratch_len();
        fft.process_with_scratch(work, &mut self.scratch[..required_scratch]);
        let inverse_scale = 1.0_f32 / fft_len as f32;
        for value in work.iter_mut() {
            value.re *= inverse_scale;
            value.im *= inverse_scale;
        }

        cosine = first_cosine;
        sine = first_sine;
        for (index, value) in work.iter().enumerate() {
            let temporary_real = MDCT_NORMALIZATION * (value.re * cosine - value.im * sine);
            let temporary_imaginary = MDCT_NORMALIZATION * (value.im * cosine + value.re * sine);

            signal[length / 2 + length / 4 - 1 - 2 * index] = temporary_real;
            if index < length / 8 {
                signal[length / 2 + length / 4 + 2 * index] = temporary_real;
            } else {
                signal[2 * index - length / 4] = -temporary_real;
            }

            signal[length / 4 + 2 * index] = temporary_imaginary;
            if index < length / 8 {
                signal[length / 4 - 1 - 2 * index] = -temporary_imaginary;
            } else {
                signal[length / 4 + length - 1 - 2 * index] = temporary_imaginary;
            }

            let old_cosine = cosine;
            cosine = cosine * cosine_step - sine * sine_step;
            sine = sine * cosine_step + old_cosine * sine_step;
        }

        // `sqrt` returns double in the C source, so the reference multiply is
        // carried out in f64 before narrowing back to the signal buffer.
        let output_scale = (length as f64).sqrt();
        for value in signal {
            *value = (f64::from(*value) * output_scale) as f32;
        }
        Ok(())
    }
}

impl Default for FastImdct {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for FastImdct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FastImdct")
            .field("cached_fft_lengths", &[64, 256, 512])
            .field("work_values", &self.work.len())
            .field("scratch_values", &self.scratch.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_signal(length: usize) -> Vec<f32> {
        let mut signal = vec![0.0_f32; length];
        let mut state = 0xc001_d00d_u32;
        for value in &mut signal[..length / 2] {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let bits = ((state & 1) << 31) | 0x3f00_0000 | ((state >> 1) & 0x007f_ffff);
            *value = f32::from_bits(bits);
        }
        signal
    }

    fn reference_positions(length: usize) -> [usize; 13] {
        [
            0,
            1,
            2,
            length / 8 - 1,
            length / 8,
            length / 4 - 1,
            length / 4,
            length / 2 - 1,
            length / 2,
            3 * length / 4,
            length - 3,
            length - 2,
            length - 1,
        ]
    }

    fn assert_close_to_c(length: usize, expected_bits: [u32; 13], tolerance: f32) {
        let mut signal = deterministic_signal(length);
        FastImdct::new().process(&mut signal).unwrap();
        for (position, expected_bits) in reference_positions(length).into_iter().zip(expected_bits)
        {
            let expected = f32::from_bits(expected_bits);
            let error = (signal[position] - expected).abs();
            assert!(
                error <= tolerance,
                "N={length} position {position}: Rust={} C={expected} error={error}",
                signal[position]
            );
        }
    }

    #[test]
    fn supported_sizes_stay_close_to_c_reference_fft() {
        assert_close_to_c(
            SHORT_IMDCT_LEN,
            [
                0x3ede_6088,
                0x3eea_062e,
                0x3e36_df2f,
                0xbf3b_a807,
                0xbfa6_26c6,
                0xbd65_30c7,
                0x3d65_30c7,
                0xbede_6088,
                0x3eb2_dcc1,
                0xbd82_1bc9,
                0xbe01_8dd2,
                0xbf1b_b0d7,
                0x3eb2_dcc1,
            ],
            2.0e-5,
        );
        assert_close_to_c(
            HALF_LONG_IMDCT_LEN,
            [
                0x3f30_e9d4,
                0x3d8e_cd0a,
                0x3ebd_646e,
                0xbf96_e2e3,
                0xbe09_18d4,
                0x3f12_afb2,
                0xbf12_afb2,
                0xbf30_e9d4,
                0x3fa8_0451,
                0x3ef3_1f34,
                0xbf31_93a7,
                0xbe08_0862,
                0x3fa8_0451,
            ],
            5.0e-5,
        );
        assert_close_to_c(
            LONG_IMDCT_LEN,
            [
                0x3f7b_d401,
                0xbe53_0c4d,
                0x3e7e_efab,
                0x3f04_e95d,
                0xbed3_7c02,
                0x3fbe_e7cc,
                0xbfbe_e7cc,
                0xbf7b_d401,
                0x3f93_0f62,
                0x3dc2_d49c,
                0x3f2c_2d90,
                0x3f25_3e22,
                0x3f93_0f62,
            ],
            8.0e-5,
        );
    }

    #[test]
    fn rejects_unknown_length_without_touching_signal() {
        let mut signal = vec![3.0_f32; 512];
        assert_eq!(
            FastImdct::new().process(&mut signal).unwrap_err(),
            ImdctError::UnsupportedLength(512)
        );
        assert_eq!(signal, vec![3.0; 512]);
    }
}
