use core::fmt;

use crate::core_side::TransformType;
use crate::imdct::{FastImdct, ImdctError};
use crate::model::AVS3_FEATURE_DIMENSIONS;
use crate::neural_qc::AVS3_SHORT_BLOCKS;

const FRAME_LEN: usize = AVS3_FEATURE_DIMENSIONS;
const SHORT_BLOCK_LEN: usize = FRAME_LEN / AVS3_SHORT_BLOCKS;
const LONG_IMDCT_LEN: usize = FRAME_LEN * 2;
const SHORT_IMDCT_LEN: usize = SHORT_BLOCK_LEN * 2;
const TRANSITION_PADDING: usize = 448;
const AVS3_PI: f32 = core::f32::consts::PI;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MdctSynthesisError {
    InvalidSpectrumLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
    Imdct(ImdctError),
}

impl fmt::Display for MdctSynthesisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpectrumLength { expected, actual } => {
                write!(f, "MDCT spectrum has {actual} lines; expected {expected}")
            }
            Self::InvalidOutputLength { expected, actual } => {
                write!(
                    f,
                    "synthesis output has {actual} samples; expected {expected}"
                )
            }
            Self::Imdct(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for MdctSynthesisError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Imdct(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ImdctError> for MdctSynthesisError {
    fn from(value: ImdctError) -> Self {
        Self::Imdct(value)
    }
}

/// Stateful AVS3 IMDCT, windowing and overlap-add synthesis.
///
/// Input short-window spectra use the frequency-major interleave produced by
/// TNS/FD processing; this stage performs the final block deinterleave itself.
/// The overlap buffer is owned per decoder instance, eliminating the timing
/// and cross-instance coupling risks of global synthesis state. Construction
/// initializes all windows, FFT plans and work buffers; frames allocate
/// nothing.
pub struct MdctSynthesis {
    imdct: FastImdct,
    long_window: [f32; FRAME_LEN],
    short_window: [f32; SHORT_BLOCK_LEN],
    overlap: [f32; FRAME_LEN],
    time_signal: [f32; LONG_IMDCT_LEN],
    short_block: [f32; SHORT_IMDCT_LEN],
    short_output: [f32; FRAME_LEN],
    short_overlap: [f32; SHORT_BLOCK_LEN],
}

impl MdctSynthesis {
    pub fn new() -> Self {
        let mut long_window = [0.0_f32; FRAME_LEN];
        let mut short_window = [0.0_f32; SHORT_BLOCK_LEN];
        fill_sine_window(&mut long_window);
        fill_sine_window(&mut short_window);
        Self {
            imdct: FastImdct::new(),
            long_window,
            short_window,
            overlap: [0.0; FRAME_LEN],
            time_signal: [0.0; LONG_IMDCT_LEN],
            short_block: [0.0; SHORT_IMDCT_LEN],
            short_output: [0.0; FRAME_LEN],
            short_overlap: [0.0; SHORT_BLOCK_LEN],
        }
    }

    pub fn reset(&mut self) {
        self.overlap.fill(0.0);
    }

    pub fn overlap_buffer(&self) -> &[f32; FRAME_LEN] {
        &self.overlap
    }

    pub fn synthesize(
        &mut self,
        spectrum: &[f32],
        transform_type: TransformType,
        output: &mut [f32],
    ) -> Result<(), MdctSynthesisError> {
        if spectrum.len() != FRAME_LEN {
            return Err(MdctSynthesisError::InvalidSpectrumLength {
                expected: FRAME_LEN,
                actual: spectrum.len(),
            });
        }
        if output.len() != FRAME_LEN {
            return Err(MdctSynthesisError::InvalidOutputLength {
                expected: FRAME_LEN,
                actual: output.len(),
            });
        }

        if transform_type == TransformType::Short {
            self.synthesize_short(spectrum, output)
        } else {
            self.synthesize_long(spectrum, transform_type, output)
        }
    }

    fn synthesize_long(
        &mut self,
        spectrum: &[f32],
        transform_type: TransformType,
        output: &mut [f32],
    ) -> Result<(), MdctSynthesisError> {
        self.time_signal.fill(0.0);
        self.time_signal[..FRAME_LEN].copy_from_slice(spectrum);
        self.imdct.process(&mut self.time_signal)?;

        match transform_type {
            TransformType::Long => {
                for index in 0..FRAME_LEN {
                    self.time_signal[index] *= self.long_window[index];
                    self.time_signal[FRAME_LEN + index] *= self.long_window[FRAME_LEN - 1 - index];
                }
            }
            TransformType::LongToShort => {
                for index in 0..FRAME_LEN {
                    self.time_signal[index] *= self.long_window[index];
                }
                let short_start = FRAME_LEN + TRANSITION_PADDING;
                for index in 0..SHORT_BLOCK_LEN {
                    self.time_signal[short_start + index] *=
                        self.short_window[SHORT_BLOCK_LEN - 1 - index];
                }
                self.time_signal[short_start + SHORT_BLOCK_LEN..].fill(0.0);
            }
            TransformType::ShortToLong => {
                self.time_signal[..TRANSITION_PADDING].fill(0.0);
                for index in 0..SHORT_BLOCK_LEN {
                    self.time_signal[TRANSITION_PADDING + index] *= self.short_window[index];
                }
                // The following 448 samples are the flat portion. The right
                // half starts exactly at FRAME_LEN and uses the long window.
                for index in 0..FRAME_LEN {
                    self.time_signal[FRAME_LEN + index] *= self.long_window[FRAME_LEN - 1 - index];
                }
            }
            TransformType::Short => unreachable!("short transform dispatched separately"),
        }

        for (value, &previous) in self.time_signal[..FRAME_LEN].iter_mut().zip(&self.overlap) {
            *value += previous;
        }
        self.overlap.copy_from_slice(&self.time_signal[FRAME_LEN..]);
        output.copy_from_slice(&self.time_signal[..FRAME_LEN]);
        Ok(())
    }

    fn synthesize_short(
        &mut self,
        spectrum: &[f32],
        output: &mut [f32],
    ) -> Result<(), MdctSynthesisError> {
        self.time_signal.fill(0.0);
        for block in 0..AVS3_SHORT_BLOCKS {
            for line in 0..SHORT_BLOCK_LEN {
                self.time_signal[block * SHORT_BLOCK_LEN + line] =
                    spectrum[block + AVS3_SHORT_BLOCKS * line];
            }
        }

        self.short_overlap.copy_from_slice(
            &self.overlap[TRANSITION_PADDING..TRANSITION_PADDING + SHORT_BLOCK_LEN],
        );
        self.short_output.fill(0.0);
        for block in 0..AVS3_SHORT_BLOCKS {
            self.short_block.fill(0.0);
            let spectrum_start = block * SHORT_BLOCK_LEN;
            self.short_block[..SHORT_BLOCK_LEN].copy_from_slice(
                &self.time_signal[spectrum_start..spectrum_start + SHORT_BLOCK_LEN],
            );
            self.imdct.process(&mut self.short_block)?;

            for index in 0..SHORT_BLOCK_LEN {
                self.short_block[index] *= self.short_window[index];
                self.short_block[SHORT_BLOCK_LEN + index] *=
                    self.short_window[SHORT_BLOCK_LEN - 1 - index];
                self.short_block[index] += self.short_overlap[index];
            }
            self.short_overlap
                .copy_from_slice(&self.short_block[SHORT_BLOCK_LEN..]);
            self.short_output[spectrum_start..spectrum_start + SHORT_BLOCK_LEN]
                .copy_from_slice(&self.short_block[..SHORT_BLOCK_LEN]);
        }

        output[..TRANSITION_PADDING].copy_from_slice(&self.overlap[..TRANSITION_PADDING]);
        output[TRANSITION_PADDING..]
            .copy_from_slice(&self.short_output[..FRAME_LEN - TRANSITION_PADDING]);

        self.overlap[..TRANSITION_PADDING]
            .copy_from_slice(&self.short_output[FRAME_LEN - TRANSITION_PADDING..]);
        self.overlap[TRANSITION_PADDING..TRANSITION_PADDING + SHORT_BLOCK_LEN]
            .copy_from_slice(&self.short_overlap);
        self.overlap[TRANSITION_PADDING + SHORT_BLOCK_LEN..].fill(0.0);
        Ok(())
    }
}

impl Default for MdctSynthesis {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MdctSynthesis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MdctSynthesis")
            .field("imdct", &self.imdct)
            .field("overlap_samples", &self.overlap.len())
            .finish_non_exhaustive()
    }
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

    fn spectrum(seed: u32) -> [f32; FRAME_LEN] {
        let mut output = [0.0_f32; FRAME_LEN];
        let mut state = seed;
        for value in &mut output {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let bits = ((state & 1) << 31) | 0x3f00_0000 | ((state >> 1) & 0x007f_ffff);
            *value = f32::from_bits(bits);
        }
        output
    }

    fn positions() -> [usize; 14] {
        [
            0, 1, 2, 127, 128, 447, 448, 575, 576, 767, 768, 1021, 1022, 1023,
        ]
    }

    fn assert_samples_close(output: &[f32], expected_bits: [u32; 14]) {
        for (position, bits) in positions().into_iter().zip(expected_bits) {
            let expected = f32::from_bits(bits);
            let error = (output[position] - expected).abs();
            assert!(
                error <= 2.0e-5,
                "position {position}: Rust={} C={expected} error={error}",
                output[position]
            );
        }
    }

    #[test]
    fn transition_sequence_matches_c_overlap_timing() {
        let vectors = [
            (
                TransformType::Long,
                0x1020_3041,
                [
                    0xba31_01ee,
                    0xbb1f_6fa4,
                    0xbb63_276f,
                    0x3d49_423b,
                    0xbe46_62f7,
                    0x3f47_0e70,
                    0x3e28_0692,
                    0xbe4c_6b80,
                    0xbf72_ee3e,
                    0x3e28_8b94,
                    0xbf9e_abce,
                    0x3f67_6046,
                    0x3f87_5555,
                    0x3f61_5f79,
                ],
            ),
            (
                TransformType::LongToShort,
                0x1020_3042,
                [
                    0xbf62_5434,
                    0x3d0b_e2a7,
                    0x3d12_1c1d,
                    0xbe97_f7b3,
                    0x3ea8_c028,
                    0xbf80_7767,
                    0xbf3d_9224,
                    0xbe3f_bc1e,
                    0x3f4d_3fec,
                    0x3f32_6175,
                    0xbf55_c4b6,
                    0x3f00_ad71,
                    0xbe50_8678,
                    0xbf4d_4b2a,
                ],
            ),
            (
                TransformType::Short,
                0x1020_3043,
                [
                    0xbfb0_af62,
                    0xbf1c_b298,
                    0x3f11_f5a1,
                    0x3d9b_1bfb,
                    0xbd66_b7a3,
                    0x3e64_8a54,
                    0xbe88_19d4,
                    0x3eae_92c2,
                    0x3fb9_3d9c,
                    0xbf6e_e77f,
                    0x3f5f_b5ab,
                    0xbec8_b882,
                    0x3f19_11f0,
                    0x3f88_f081,
                ],
            ),
            (
                TransformType::Short,
                0x1020_3044,
                [
                    0xbf77_905e,
                    0xbefc_1e44,
                    0x3df8_68ac,
                    0x3c64_9cec,
                    0x3d63_0063,
                    0xbf64_1d0f,
                    0x3e61_fbc1,
                    0xbda8_5b46,
                    0x3fb1_945f,
                    0x3e9a_fb07,
                    0x3fa0_d3f2,
                    0xbf37_c156,
                    0x3fc2_422c,
                    0x3f9e_5412,
                ],
            ),
            (
                TransformType::ShortToLong,
                0x1020_3045,
                [
                    0xbdb8_dd78,
                    0xbf9e_f0bc,
                    0x3e54_ac6e,
                    0xbdf7_1138,
                    0xbe85_2735,
                    0xbe89_34fa,
                    0xbf56_33e5,
                    0x3ef4_705a,
                    0x3e90_b008,
                    0x3f5c_693e,
                    0x3d2e_4009,
                    0xbf21_f0d6,
                    0x3eac_1460,
                    0x3e0a_f1d2,
                ],
            ),
            (
                TransformType::Long,
                0x1020_3046,
                [
                    0x3c1c_9140,
                    0x3edc_f9d1,
                    0x3f29_9414,
                    0x3f2a_401b,
                    0xbf82_bc57,
                    0x3f9b_e880,
                    0xbdc4_75c4,
                    0xbe56_fb3a,
                    0x3f5f_8ae3,
                    0x3f5f_3be4,
                    0x3ef5_3738,
                    0x3fd1_db63,
                    0xbea4_5e50,
                    0xbfa8_2f4e,
                ],
            ),
        ];

        let mut synthesis = MdctSynthesis::new();
        let mut output = [0.0_f32; FRAME_LEN];
        for (transform, seed, expected) in vectors {
            synthesis
                .synthesize(&spectrum(seed), transform, &mut output)
                .unwrap();
            assert_samples_close(&output, expected);
            assert!(output.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn invalid_lengths_do_not_change_overlap_or_output() {
        let mut synthesis = MdctSynthesis::new();
        let original_overlap = *synthesis.overlap_buffer();
        let mut output = [7.0_f32; FRAME_LEN];
        let error = synthesis
            .synthesize(&[0.0; 17], TransformType::Long, &mut output)
            .unwrap_err();
        assert_eq!(
            error,
            MdctSynthesisError::InvalidSpectrumLength {
                expected: FRAME_LEN,
                actual: 17,
            }
        );
        assert_eq!(output, [7.0; FRAME_LEN]);
        assert_eq!(synthesis.overlap_buffer(), &original_overlap);
    }
}
