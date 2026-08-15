use core::fmt;

use crate::core_side::{MAX_TNS_FILTERS, MAX_TNS_ORDER, TnsSideInfo, TransformType};
use crate::model::AVS3_FEATURE_DIMENSIONS;
use crate::spectrum::SpectrumReorder;

// The C decoder expresses these borders in Hz and converts them with integer
// arithmetic: 2 * 1024 * frequency / 48_000.
const TNS_FILTER_RANGES: [(usize, usize); MAX_TNS_FILTERS] = [(28, 230), (230, 853)];

// AVS3's four-bit PARCOR reconstruction codebook. Keeping the normative f32
// literals avoids per-frame trigonometry and preserves the reference bits.
const TNS_PARCOR: [f32; 16] = [
    -0.995_734_16,
    -0.961_825_67,
    -0.895_163_3,
    -0.798_017_2,
    -0.673_695_6,
    -0.526_432_16,
    -0.361_241_67,
    -0.183_749_51,
    0.0,
    0.207_911_69,
    0.406_736_64,
    0.587_785_24,
    0.743_144_8,
    0.866_025_4,
    0.951_056_54,
    0.994_521_9,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TnsSynthesisError {
    InvalidSpectrumLength { expected: usize, actual: usize },
}

impl fmt::Display for TnsSynthesisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpectrumLength { expected, actual } => {
                write!(f, "TNS spectrum has {actual} lines; expected {expected}")
            }
        }
    }
}

impl std::error::Error for TnsSynthesisError {}

/// Allocation-stable temporal-noise-shaping synthesis stage.
///
/// AVS3 filters the two frequency ranges from high to low and deliberately
/// carries one lattice state across the range boundary. Short-window spectra
/// are deinterleaved before filtering and restored afterwards. The scratch
/// array is owned by this object, so applying TNS never allocates.
#[derive(Debug, Clone)]
pub struct TnsSynthesis {
    reorder: SpectrumReorder,
}

impl TnsSynthesis {
    pub fn new() -> Self {
        Self {
            reorder: SpectrumReorder::new(),
        }
    }

    pub fn apply(
        &mut self,
        side_info: TnsSideInfo,
        transform_type: TransformType,
        spectrum: &mut [f32],
    ) -> Result<(), TnsSynthesisError> {
        if spectrum.len() != AVS3_FEATURE_DIMENSIONS {
            return Err(TnsSynthesisError::InvalidSpectrumLength {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: spectrum.len(),
            });
        }

        let short_window = transform_type == TransformType::Short;
        if short_window {
            self.reorder.deinterleave_exact(spectrum);
        }

        let mut state = [0.0_f32; MAX_TNS_ORDER];
        let mut parcor = [0.0_f32; MAX_TNS_ORDER];
        let filters = side_info.filters();

        // This order and the shared state match RunTnsFilter in the reference
        // decoder. Resetting state between the filters changes decoded audio.
        for filter_index in (0..MAX_TNS_FILTERS).rev() {
            let filter = filters[filter_index];
            if !filter.enabled() {
                continue;
            }

            let order = filter.order();
            debug_assert!((1..=MAX_TNS_ORDER).contains(&order));
            for (destination, coefficient) in parcor[..order].iter_mut().zip(filter.coefficients())
            {
                let codebook_index = usize::try_from(i16::from(coefficient.index()) + 8)
                    .expect("validated TNS coefficient index");
                *destination = TNS_PARCOR[codebook_index];
            }

            let (start, stop) = TNS_FILTER_RANGES[filter_index];
            for value in &mut spectrum[start..stop] {
                *value = synthesis_lattice_sample(*value, &parcor[..order], &mut state[..order]);
            }
        }

        if short_window {
            self.reorder.interleave_exact(spectrum);
        }
        Ok(())
    }
}

impl Default for TnsSynthesis {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn synthesis_lattice_sample(mut value: f32, parcor: &[f32], state: &mut [f32]) -> f32 {
    let last = parcor.len() - 1;
    value -= parcor[last] * state[last];
    for stage in (0..last).rev() {
        value -= parcor[stage] * state[stage];
        state[stage + 1] = parcor[stage] * value + state[stage];
    }
    state[0] = value;
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BweConfig, CoreBitstreamConfig, LsfCodebookMode, MonoSideInfoDecoder, NnType};

    const REFERENCE_PAYLOAD: [u8; 35] = [
        0x44, 0x72, 0x61, 0x63, 0xb6, 0x23, 0xa0, 0xf0, 0xea, 0x00, 0xfb, 0xdc, 0x10, 0x30, 0x2f,
        0x40, 0x2a, 0x40, 0xc0, 0xff, 0x6f, 0x01, 0xc7, 0xe9, 0x5f, 0x03, 0x84, 0xa0, 0xd8, 0x7f,
        0xfd, 0x51, 0xf6, 0xf2, 0x00,
    ];

    fn reference_side_info() -> TnsSideInfo {
        let config = CoreBitstreamConfig::new(
            NnType::Main,
            277,
            LsfCodebookMode::HighBitrate,
            BweConfig::for_mono_bitrate(64_000).unwrap(),
        )
        .unwrap();
        MonoSideInfoDecoder::new()
            .parse(&REFERENCE_PAYLOAD, config)
            .unwrap()
            .core()
            .tns()
    }

    fn deterministic_spectrum() -> [f32; AVS3_FEATURE_DIMENSIONS] {
        let mut spectrum = [0.0_f32; AVS3_FEATURE_DIMENSIONS];
        let mut state = 0x6d2b_79f5_u32;
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
    fn long_window_synthesis_matches_c_at_all_optimization_levels() {
        let mut spectrum = deterministic_spectrum();
        TnsSynthesis::new()
            .apply(reference_side_info(), TransformType::Long, &mut spectrum)
            .unwrap();

        assert_eq!(fingerprint(&spectrum), 0xbb0b_f52d_4072_384b);
        let positions = [0, 27, 28, 29, 100, 229, 230, 231, 400, 852, 853, 900, 1023];
        assert_eq!(
            positions.map(|index| spectrum[index].to_bits()),
            [
                0xbf57_638f,
                0xbf1a_3bc1,
                0x430e_11b4,
                0x431e_333e,
                0x430c_2469,
                0x4264_e6b0,
                0x3f75_e799,
                0xc00b_7cf5,
                0x41b4_1637,
                0x4232_552a,
                0x3f78_b4c3,
                0x3f3e_2765,
                0x3f51_15b0,
            ]
        );
    }

    #[test]
    fn short_window_deinterleave_filter_and_restore_matches_c() {
        let mut spectrum = deterministic_spectrum();
        TnsSynthesis::new()
            .apply(reference_side_info(), TransformType::Short, &mut spectrum)
            .unwrap();

        assert_eq!(fingerprint(&spectrum), 0xc665_7f8a_5bb9_7f68);
        let positions = [0, 27, 28, 29, 100, 229, 230, 231, 400, 852, 853, 900, 1023];
        assert_eq!(
            positions.map(|index| spectrum[index].to_bits()),
            [
                0xbf57_638f,
                0xc1fe_c28c,
                0xc122_86ce,
                0x41df_3d45,
                0xc17b_3e2e,
                0xc04c_5564,
                0xc246_72ca,
                0x3f5e_6e1a,
                0x40a0_fd48,
                0x423d_3089,
                0xc1c4_c33c,
                0xc218_8627,
                0x3f51_15b0,
            ]
        );
    }

    #[test]
    fn rejects_wrong_spectrum_length_before_mutating() {
        let mut spectrum = [3.0_f32; 17];
        let error = TnsSynthesis::new()
            .apply(reference_side_info(), TransformType::Long, &mut spectrum)
            .unwrap_err();
        assert_eq!(
            error,
            TnsSynthesisError::InvalidSpectrumLength {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: spectrum.len(),
            }
        );
        assert_eq!(spectrum, [3.0; 17]);
    }
}
