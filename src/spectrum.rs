use core::fmt;

use crate::core_side::{TransformType, WindowGrouping};
use crate::model::AVS3_FEATURE_DIMENSIONS;
use crate::neural_qc::{AVS3_SHORT_BLOCKS, NoiseGroup};

const SHORT_BLOCK_LINES: usize = AVS3_FEATURE_DIMENSIONS / AVS3_SHORT_BLOCKS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumReorderError {
    InvalidSpectrumLength { expected: usize, actual: usize },
}

impl fmt::Display for SpectrumReorderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpectrumLength { expected, actual } => write!(
                f,
                "short-window spectrum has {actual} lines; expected {expected}"
            ),
        }
    }
}

impl std::error::Error for SpectrumReorderError {}

/// Reusable workspace for AVS3's short-window spectrum permutations.
///
/// Neural QC returns a group-interleaved spectrum. [`Self::degroup`] restores
/// original short-block order while retaining the codec's frequency-major
/// interleave. TNS temporarily converts that layout to block-major form, and
/// IMDCT needs the same conversion permanently. All operations use one fixed
/// scratch array and allocate nothing per frame.
#[derive(Debug, Clone)]
pub struct SpectrumReorder {
    scratch: [f32; AVS3_FEATURE_DIMENSIONS],
}

impl SpectrumReorder {
    pub fn new() -> Self {
        Self {
            scratch: [0.0; AVS3_FEATURE_DIMENSIONS],
        }
    }

    pub fn degroup(
        &mut self,
        grouping: WindowGrouping,
        transform_type: TransformType,
        spectrum: &mut [f32],
    ) -> Result<(), SpectrumReorderError> {
        check_spectrum_len(spectrum)?;
        if transform_type != TransformType::Short || grouping.count() == 1 {
            return Ok(());
        }

        let indicator = grouping.indicator();
        let transient_blocks = indicator
            .iter()
            .filter(|&&group| group == NoiseGroup::Transient)
            .count();
        let other_blocks = AVS3_SHORT_BLOCKS - transient_blocks;

        // Deinterleave each coded group into contiguous, block-major data.
        for block in 0..transient_blocks {
            for line in 0..SHORT_BLOCK_LINES {
                self.scratch[block * SHORT_BLOCK_LINES + line] =
                    spectrum[block + transient_blocks * line];
            }
        }
        let group_offset = transient_blocks * SHORT_BLOCK_LINES;
        for block in 0..other_blocks {
            for line in 0..SHORT_BLOCK_LINES {
                self.scratch[group_offset + block * SHORT_BLOCK_LINES + line] =
                    spectrum[group_offset + block + other_blocks * line];
            }
        }

        // Put transient and non-transient blocks back at their original time
        // positions. Source and destination are separate, so block copies are
        // safe even for corrupted but bounded indicator patterns.
        let mut transient_index = 0;
        let mut other_index = transient_blocks;
        for (block, &group) in indicator.iter().enumerate() {
            let source_block = match group {
                NoiseGroup::Transient => {
                    let value = transient_index;
                    transient_index += 1;
                    value
                }
                NoiseGroup::Other => {
                    let value = other_index;
                    other_index += 1;
                    value
                }
            };
            spectrum[block * SHORT_BLOCK_LINES..(block + 1) * SHORT_BLOCK_LINES].copy_from_slice(
                &self.scratch
                    [source_block * SHORT_BLOCK_LINES..(source_block + 1) * SHORT_BLOCK_LINES],
            );
        }

        self.interleave_exact(spectrum);
        Ok(())
    }

    pub fn deinterleave(&mut self, spectrum: &mut [f32]) -> Result<(), SpectrumReorderError> {
        check_spectrum_len(spectrum)?;
        self.deinterleave_exact(spectrum);
        Ok(())
    }

    pub fn interleave(&mut self, spectrum: &mut [f32]) -> Result<(), SpectrumReorderError> {
        check_spectrum_len(spectrum)?;
        self.interleave_exact(spectrum);
        Ok(())
    }

    pub(crate) fn deinterleave_exact(&mut self, spectrum: &mut [f32]) {
        debug_assert_eq!(spectrum.len(), AVS3_FEATURE_DIMENSIONS);
        for block in 0..AVS3_SHORT_BLOCKS {
            for line in 0..SHORT_BLOCK_LINES {
                self.scratch[block * SHORT_BLOCK_LINES + line] =
                    spectrum[block + AVS3_SHORT_BLOCKS * line];
            }
        }
        spectrum.copy_from_slice(&self.scratch);
    }

    pub(crate) fn interleave_exact(&mut self, spectrum: &mut [f32]) {
        debug_assert_eq!(spectrum.len(), AVS3_FEATURE_DIMENSIONS);
        for block in 0..AVS3_SHORT_BLOCKS {
            for line in 0..SHORT_BLOCK_LINES {
                self.scratch[block + AVS3_SHORT_BLOCKS * line] =
                    spectrum[block * SHORT_BLOCK_LINES + line];
            }
        }
        spectrum.copy_from_slice(&self.scratch);
    }
}

impl Default for SpectrumReorder {
    fn default() -> Self {
        Self::new()
    }
}

fn check_spectrum_len(spectrum: &[f32]) -> Result<(), SpectrumReorderError> {
    if spectrum.len() == AVS3_FEATURE_DIMENSIONS {
        Ok(())
    } else {
        Err(SpectrumReorderError::InvalidSpectrumLength {
            expected: AVS3_FEATURE_DIMENSIONS,
            actual: spectrum.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;
    use crate::core_side::{BweConfig, CoreBitstreamConfig, LsfCodebookMode, MonoSideInfoDecoder};
    use crate::header::NnType;

    const TWO_GROUP_PAYLOAD: [u8; 35] = [
        0x44, 0x72, 0x61, 0x63, 0xb6, 0x23, 0xa0, 0xf0, 0xea, 0x00, 0xfb, 0xdc, 0x10, 0x30, 0x2f,
        0x40, 0x2a, 0x40, 0xc0, 0xff, 0x6f, 0x01, 0xc7, 0xe9, 0x5f, 0x03, 0x84, 0xa0, 0xd8, 0x7f,
        0xfd, 0x51, 0xf6, 0xf2, 0x00,
    ];

    fn two_group_info() -> WindowGrouping {
        let config = CoreBitstreamConfig::new(
            NnType::Main,
            277,
            LsfCodebookMode::HighBitrate,
            BweConfig::for_mono_bitrate(64_000).unwrap(),
        )
        .unwrap();
        let mut decoder = MonoSideInfoDecoder::new();
        decoder
            .parse(&TWO_GROUP_PAYLOAD, config)
            .unwrap()
            .core()
            .grouping()
    }

    fn one_group_info() -> WindowGrouping {
        let mut writer = BitWriter::new();
        writer.write_bits(1, 2).unwrap();
        for width in [8, 8, 7, 7, 6, 5, 5] {
            writer.write_bits(0, width).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 7).unwrap();
        writer.write_bits(0, 3).unwrap();
        writer.write_bits(0, 8).unwrap();
        let payload_bits = writer.bit_len();
        let payload = writer.into_bytes();
        let config = CoreBitstreamConfig::new(
            NnType::Main,
            payload_bits,
            LsfCodebookMode::HighBitrate,
            None,
        )
        .unwrap();
        let mut decoder = MonoSideInfoDecoder::new();
        decoder.parse(&payload, config).unwrap().core().grouping()
    }

    fn deterministic_spectrum() -> [f32; AVS3_FEATURE_DIMENSIONS] {
        let mut spectrum = [0.0_f32; AVS3_FEATURE_DIMENSIONS];
        let mut state = 0xa511_e9b3_u32;
        for value in &mut spectrum {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let bits = ((state & 1) << 31) | 0x3f00_0000 | ((state >> 1) & 0x007f_ffff);
            *value = f32::from_bits(bits);
        }
        spectrum
    }

    fn fingerprint(values: &[f32]) -> u64 {
        values
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
                (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3)
            })
    }

    #[test]
    fn two_group_degrouping_matches_c() {
        let mut spectrum = deterministic_spectrum();
        SpectrumReorder::new()
            .degroup(two_group_info(), TransformType::Short, &mut spectrum)
            .unwrap();
        assert_eq!(fingerprint(&spectrum), 0x178e_f528_bbae_0a4e);
        let positions = [
            0, 1, 2, 3, 7, 8, 127, 128, 255, 256, 511, 512, 767, 768, 1023,
        ];
        assert_eq!(
            positions.map(|index| spectrum[index].to_bits()),
            [
                0x3f6f_b8d0,
                0xbf41_293a,
                0xbf2a_6979,
                0x3f09_1c62,
                0xbf43_1041,
                0x3f2c_579b,
                0x3f10_d5a5,
                0xbf37_3870,
                0x3f04_1399,
                0x3f1f_b8b8,
                0xbf2d_4fe4,
                0xbf03_e3c8,
                0xbf53_ef09,
                0xbf11_6ce1,
                0x3f01_e60b,
            ]
        );
    }

    #[test]
    fn one_group_short_frame_is_unchanged_like_c() {
        let mut spectrum = deterministic_spectrum();
        let original = spectrum;
        SpectrumReorder::new()
            .degroup(one_group_info(), TransformType::Short, &mut spectrum)
            .unwrap();
        assert_eq!(spectrum, original);
        assert_eq!(fingerprint(&spectrum), 0x52fa_b9ee_28e6_3574);
    }

    #[test]
    fn interleave_and_deinterleave_are_exact_inverses() {
        let original = deterministic_spectrum();
        let mut spectrum = original;
        let mut reorder = SpectrumReorder::new();
        reorder.deinterleave(&mut spectrum).unwrap();
        assert_ne!(spectrum, original);
        reorder.interleave(&mut spectrum).unwrap();
        assert_eq!(spectrum, original);
    }
}
