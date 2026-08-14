use core::fmt;

use crate::bitstream::BitReader;
use crate::core_side::TransformType;
use crate::error::BitstreamError;
use crate::model::AVS3_FEATURE_DIMENSIONS;

pub const MCR_SCALE_FACTOR_BANDS: usize = 18;
pub const MCR_SUBVECTOR_DIMENSIONS: usize = 3;
pub const MCR_SUBVECTORS: usize = MCR_SCALE_FACTOR_BANDS / MCR_SUBVECTOR_DIMENSIONS;
pub const MCR_SUBSPECTRA: usize = 2;
pub const MCR_LONG_INDEX_BITS: usize = 9;
pub const MCR_SHORT_INDEX_BITS: usize = 8;
pub const MCR_LONG_CODEBOOK_ENTRIES: usize = 512;
pub const MCR_SHORT_CODEBOOK_ENTRIES: usize = 256;

const MCR_SFB_BORDERS: [usize; MCR_SCALE_FACTOR_BANDS + 1] = [
    0, 4, 8, 12, 16, 22, 28, 34, 40, 48, 56, 64, 76, 88, 100, 116, 132, 154, 176,
];
const ROTATION_VALUES_PER_ANGLE: usize = 2;
const ROTATION_BYTES_PER_ANGLE: usize = ROTATION_VALUES_PER_ANGLE * std::mem::size_of::<f32>();
const LONG_ANGLES: usize = MCR_LONG_CODEBOOK_ENTRIES * MCR_SUBVECTOR_DIMENSIONS;
const SHORT_ANGLES: usize = MCR_SHORT_CODEBOOK_ENTRIES * MCR_SUBVECTOR_DIMENSIONS;
const TOTAL_ANGLES: usize = LONG_ANGLES + SHORT_ANGLES;

pub const MCR_ROTATION_VALUES: usize = TOTAL_ANGLES * ROTATION_VALUES_PER_ANGLE;
pub const MCR_ROTATION_BYTES_LEN: usize = MCR_ROTATION_VALUES * std::mem::size_of::<f32>();
pub const MCR_ROTATION_FNV1A: u64 = 0x5b62_aa9a_6b23_145a;

const MCR_ROTATION_BYTES: &[u8; MCR_ROTATION_BYTES_LEN] =
    include_bytes!("../assets/avs3a_mcr_rotations.bin");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McrSideInfo {
    vq_indexes: [[u16; MCR_SUBVECTORS]; MCR_SUBSPECTRA],
    short_window: bool,
}

impl McrSideInfo {
    pub fn new(
        vq_indexes: [[u16; MCR_SUBVECTORS]; MCR_SUBSPECTRA],
        short_window: bool,
    ) -> Result<Self, McrError> {
        let entries = if short_window {
            MCR_SHORT_CODEBOOK_ENTRIES
        } else {
            MCR_LONG_CODEBOOK_ENTRIES
        };
        for (subspectrum, indexes) in vq_indexes.iter().enumerate() {
            for (subvector, &index) in indexes.iter().enumerate() {
                if usize::from(index) >= entries {
                    return Err(McrError::InvalidCodebookIndex {
                        subspectrum,
                        subvector,
                        index,
                        entries,
                    });
                }
            }
        }
        Ok(Self {
            vq_indexes,
            short_window,
        })
    }

    pub(crate) fn parse(
        reader: &mut BitReader<'_>,
        transform_type: TransformType,
    ) -> Result<Self, McrError> {
        let short_window = transform_type == TransformType::Short;
        let width = if short_window {
            MCR_SHORT_INDEX_BITS
        } else {
            MCR_LONG_INDEX_BITS
        };
        let mut indexes = [[0_u16; MCR_SUBVECTORS]; MCR_SUBSPECTRA];
        for subvector in 0..MCR_SUBVECTORS {
            for subspectrum_indexes in &mut indexes {
                subspectrum_indexes[subvector] = reader.read_bits(width)? as u16;
            }
        }
        Self::new(indexes, short_window)
    }

    pub fn vq_indexes(&self) -> &[[u16; MCR_SUBVECTORS]; MCR_SUBSPECTRA] {
        &self.vq_indexes
    }

    pub fn short_window(self) -> bool {
        self.short_window
    }

    pub fn index_bits(self) -> usize {
        if self.short_window {
            MCR_SHORT_INDEX_BITS
        } else {
            MCR_LONG_INDEX_BITS
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McrError {
    InvalidSpectrumLength {
        channel: usize,
        expected: usize,
        actual: usize,
    },
    InvalidCodebookIndex {
        subspectrum: usize,
        subvector: usize,
        index: u16,
        entries: usize,
    },
    Bitstream(BitstreamError),
}

impl fmt::Display for McrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpectrumLength {
                channel,
                expected,
                actual,
            } => write!(
                f,
                "MCR channel {channel} spectrum has {actual} lines; expected {expected}"
            ),
            Self::InvalidCodebookIndex {
                subspectrum,
                subvector,
                index,
                entries,
            } => write!(
                f,
                "MCR subspectrum {subspectrum} subvector {subvector} index {index} is outside {entries} entries"
            ),
            Self::Bitstream(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for McrError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BitstreamError> for McrError {
    fn from(value: BitstreamError) -> Self {
        Self::Bitstream(value)
    }
}

/// Allocation-free MCR upmix using precomputed normative rotations.
///
/// The input left spectrum has already been neural-decoded and degrouped. The
/// operation first copies it to the right spectrum, then applies the inverse
/// rotations independently to even and odd lines in the first 18 MCR bands.
#[derive(Debug, Default, Clone, Copy)]
pub struct McrSynthesis;

impl McrSynthesis {
    pub const fn new() -> Self {
        Self
    }

    pub fn apply(
        &self,
        side_info: McrSideInfo,
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(), McrError> {
        if left.len() != AVS3_FEATURE_DIMENSIONS {
            return Err(McrError::InvalidSpectrumLength {
                channel: 0,
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: left.len(),
            });
        }
        if right.len() != AVS3_FEATURE_DIMENSIONS {
            return Err(McrError::InvalidSpectrumLength {
                channel: 1,
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: right.len(),
            });
        }

        right.copy_from_slice(left);
        for subspectrum in 0..MCR_SUBSPECTRA {
            for band in 0..MCR_SCALE_FACTOR_BANDS {
                let subvector = band / MCR_SUBVECTOR_DIMENSIONS;
                let dimension = band % MCR_SUBVECTOR_DIMENSIONS;
                let codebook_index = usize::from(side_info.vq_indexes[subspectrum][subvector]);
                let (cosine, sine) = rotation(side_info.short_window, codebook_index, dimension);
                for half_line in MCR_SFB_BORDERS[band]..MCR_SFB_BORDERS[band + 1] {
                    let line = half_line * MCR_SUBSPECTRA + subspectrum;
                    let value = left[line];
                    let cosine_product = cosine * value;
                    let sine_product = sine * value;
                    left[line] = cosine_product - sine_product;
                    right[line] = sine_product + cosine_product;
                }
            }
        }
        Ok(())
    }
}

pub fn mcr_rotation_bytes() -> &'static [u8; MCR_ROTATION_BYTES_LEN] {
    MCR_ROTATION_BYTES
}

fn rotation(short_window: bool, codebook_index: usize, dimension: usize) -> (f32, f32) {
    let angle = if short_window { LONG_ANGLES } else { 0 }
        + codebook_index * MCR_SUBVECTOR_DIMENSIONS
        + dimension;
    debug_assert!(angle < TOTAL_ANGLES);
    let offset = angle * ROTATION_BYTES_PER_ANGLE;
    let cosine = f32::from_le_bytes(
        MCR_ROTATION_BYTES[offset..offset + 4]
            .try_into()
            .expect("validated MCR rotation asset offset"),
    );
    let sine = f32::from_le_bytes(
        MCR_ROTATION_BYTES[offset + 4..offset + 8]
            .try_into()
            .expect("validated MCR rotation asset offset"),
    );
    (cosine, sine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BitWriter;

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        })
    }

    fn fingerprint(values: &[f32]) -> u64 {
        values
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
                (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3)
            })
    }

    fn deterministic_spectrum() -> [f32; AVS3_FEATURE_DIMENSIONS] {
        let mut spectrum = [0.0_f32; AVS3_FEATURE_DIMENSIONS];
        let mut state = 0x83d2_e19b_u32;
        for value in &mut spectrum {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let sign = (state & 1) << 31;
            *value = f32::from_bits(sign | 0x4200_0000 | ((state >> 1) & 0x007f_ffff));
        }
        spectrum
    }

    #[test]
    fn embedded_rotations_have_expected_fingerprint_and_values() {
        assert_eq!(mcr_rotation_bytes().len(), 18_432);
        assert_eq!(fnv1a(mcr_rotation_bytes()), MCR_ROTATION_FNV1A);
        assert_eq!(rotation(false, 0, 0).0.to_bits(), 0x3e05_76da);
        assert_eq!(rotation(false, 511, 2).1.to_bits(), 0xbebb_b7db);
        assert_eq!(rotation(true, 0, 0).0.to_bits(), 0x3e24_a553);
        assert_eq!(rotation(true, 255, 2).1.to_bits(), 0xbf77_a16c);
    }

    #[test]
    fn parses_even_then_odd_indexes_for_each_subvector() {
        let mut writer = BitWriter::new();
        let even = [1_u16, 2, 3, 4, 5, 255];
        let odd = [255_u16, 5, 4, 3, 2, 1];
        for subvector in 0..MCR_SUBVECTORS {
            writer.write_bits(u64::from(even[subvector]), 8).unwrap();
            writer.write_bits(u64::from(odd[subvector]), 8).unwrap();
        }
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        let side = McrSideInfo::parse(&mut reader, TransformType::Short).unwrap();
        assert_eq!(side.vq_indexes(), &[even, odd]);
        assert!(side.short_window());
        assert_eq!(reader.position(), 96);
    }

    #[test]
    fn inverse_rotations_match_c_reference_fingerprints() {
        let indexes = [[0, 31, 127, 255, 384, 511], [511, 384, 255, 127, 31, 0]];
        let side = McrSideInfo::new(indexes, false).unwrap();
        let mut left = deterministic_spectrum();
        let original = left;
        let mut right = [0.0_f32; AVS3_FEATURE_DIMENSIONS];
        McrSynthesis::new()
            .apply(side, &mut left, &mut right)
            .unwrap();
        assert_eq!(fingerprint(&left), 0x5d7f_1673_ba1b_a175);
        assert_eq!(fingerprint(&right), 0xb2d9_d1d9_de4b_a34d);
        assert_eq!(&left[352..], &original[352..]);
        assert_eq!(&right[352..], &original[352..]);
    }

    #[test]
    fn validates_indexes_and_lengths_before_mutation() {
        assert_eq!(
            McrSideInfo::new([[256; MCR_SUBVECTORS]; MCR_SUBSPECTRA], true),
            Err(McrError::InvalidCodebookIndex {
                subspectrum: 0,
                subvector: 0,
                index: 256,
                entries: 256,
            })
        );

        let side = McrSideInfo::new([[0; MCR_SUBVECTORS]; MCR_SUBSPECTRA], false).unwrap();
        let mut left = [1.0_f32; 17];
        let mut right = [2.0_f32; AVS3_FEATURE_DIMENSIONS];
        assert!(McrSynthesis::new()
            .apply(side, &mut left, &mut right)
            .is_err());
        assert_eq!(left, [1.0; 17]);
        assert_eq!(right, [2.0; AVS3_FEATURE_DIMENSIONS]);
    }
}
