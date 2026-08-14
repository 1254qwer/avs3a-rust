use core::fmt;

use crate::core_side::{BweConfig, BweSideInfo, BweWhiteningLevel};
use crate::model::AVS3_FEATURE_DIMENSIONS;
use crate::random::{Avs3Random, AVS3_RAND_MAX};

const WHITENING_RADIUS: usize = 7;
const WHITENING_WIDTH: f32 = (WHITENING_RADIUS * 2 + 1) as f32;

// C reference values of:
//   (float)pow(2.0f, envelope_index / 4.24966f - 4.0f)
//
// The table removes libm from the per-frame path and fixes the exact target
// energy on every platform supported by the crate.
const TARGET_ENERGY_BITS: [u32; 128] = [
    0x3d80_0000,
    0x3d96_ad3f,
    0x3db1_5ef6,
    0x3dd0_cb5a,
    0x3df5_c8e7,
    0x3e10_aa0d,
    0x3e2a_4b14,
    0x3e48_7679,
    0x3e6b_fa28,
    0x3e8a_e446,
    0x3ea3_7f7e,
    0x3ec0_76b3,
    0x3ee2_8f99,
    0x3f05_5976,
    0x3f1c_f953,
    0x3f38_c8a4,
    0x3f59_853a,
    0x3f80_0745,
    0x3f96_b5cd,
    0x3fb1_6908,
    0x3fd0_d735,
    0x3ff5_d6dc,
    0x4010_b244,
    0x402a_54bf,
    0x4048_81db,
    0x406c_078e,
    0x408a_ec29,
    0x40a3_88c7,
    0x40c0_81a1,
    0x40e2_9c76,
    0x4105_6109,
    0x411d_023d,
    0x4138_d322,
    0x4159_9194,
    0x4180_0e8a,
    0x4196_be5f,
    0x41b1_731b,
    0x41d0_e315,
    0x41f5_e4d1,
    0x4210_ba7e,
    0x422a_5e6b,
    0x4248_8d41,
    0x426c_14f5,
    0x428a_f40f,
    0x42a3_9210,
    0x42c0_8c93,
    0x42e2_a954,
    0x4305_689e,
    0x431d_0b27,
    0x4338_dda4,
    0x4359_9def,
    0x4380_15d2,
    0x4396_c6eb,
    0x43b1_7d33,
    0x43d0_eeed,
    0x43f5_f2cd,
    0x4410_c2b3,
    0x442a_681b,
    0x4448_98a0,
    0x446c_2262,
    0x448a_fbf0,
    0x44a3_9b5d,
    0x44c0_977e,
    0x44e2_b638,
    0x4505_702f,
    0x451d_1416,
    0x4538_e820,
    0x4559_aa4f,
    0x4580_1d15,
    0x4596_cf7e,
    0x45b1_874a,
    0x45d0_fac6,
    0x45f6_00bf,
    0x4610_caee,
    0x462a_71cc,
    0x4648_a400,
    0x466c_2fc5,
    0x468b_03d7,
    0x46a3_a4ab,
    0x46c0_a269,
    0x46e2_c313,
    0x4705_77c5,
    0x471d_1d04,
    0x4738_f29c,
    0x4759_b6a6,
    0x4780_245e,
    0x4796_d812,
    0x47b1_9153,
    0x47d1_06a8,
    0x47f6_0ebc,
    0x4810_d32a,
    0x482a_7b6f,
    0x4848_af69,
    0x486c_3d34,
    0x488b_0bbf,
    0x48a3_adeb,
    0x48c0_ad5e,
    0x48e2_cff8,
    0x4905_7f5c,
    0x491d_25e6,
    0x4938_fd20,
    0x4959_c307,
    0x4980_2ba7,
    0x4996_e098,
    0x49b1_9b6c,
    0x49d1_128b,
    0x49f6_1cba,
    0x4a10_db5a,
    0x4a2a_8520,
    0x4a48_bad2,
    0x4a6c_4aa3,
    0x4a8b_139b,
    0x4aa3_b73a,
    0x4ac0_b852,
    0x4ae2_dcde,
    0x4b05_86e8,
    0x4b1d_2ed6,
    0x4b39_07a5,
    0x4b59_cf69,
    0x4b80_32e6,
    0x4b96_e92d,
    0x4bb1_a586,
    0x4bd1_1e6f,
    0x4bf6_2aa4,
    0x4c10_e396,
    0x4c2a_8ed3,
    0x4c48_c63c,
    0x4c6c_57fe,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BweSynthesisError {
    InvalidSpectrumLength {
        expected: usize,
        actual: usize,
    },
    SideInfoDoesNotMatchConfig {
        config_tiles: usize,
        side_info_tiles: usize,
        config_bands: usize,
        side_info_bands: usize,
    },
}

impl fmt::Display for BweSynthesisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpectrumLength { expected, actual } => {
                write!(f, "BWE spectrum has {actual} lines; expected {expected}")
            }
            Self::SideInfoDoesNotMatchConfig {
                config_tiles,
                side_info_tiles,
                config_bands,
                side_info_bands,
            } => write!(
                f,
                "BWE side information has {side_info_tiles} tiles/{side_info_bands} bands; configuration requires {config_tiles}/{config_bands}"
            ),
        }
    }
}

impl std::error::Error for BweSynthesisError {}

/// Allocation-stable AVS3 bandwidth-extension synthesis.
///
/// Source-tile copying, all three whitening modes and SFB envelope recovery
/// follow the C reference operation order. Random state is supplied by the
/// caller because neural noise filling and BWE share one decoder-local stream.
#[derive(Debug, Clone)]
pub struct BweSynthesis {
    copied_spectrum: [f32; AVS3_FEATURE_DIMENSIONS],
}

impl BweSynthesis {
    pub fn new() -> Self {
        Self {
            copied_spectrum: [0.0; AVS3_FEATURE_DIMENSIONS],
        }
    }

    pub fn apply(
        &mut self,
        config: BweConfig,
        side_info: BweSideInfo,
        spectrum: &mut [f32],
        random: &mut Avs3Random,
    ) -> Result<(), BweSynthesisError> {
        if spectrum.len() != AVS3_FEATURE_DIMENSIONS {
            return Err(BweSynthesisError::InvalidSpectrumLength {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: spectrum.len(),
            });
        }
        let side_info_tiles = side_info.whitening_levels().len();
        let side_info_bands = side_info.envelope_indexes().len();
        if side_info_tiles != config.num_tiles()
            || side_info_bands != config.num_scale_factor_bands()
        {
            return Err(BweSynthesisError::SideInfoDoesNotMatchConfig {
                config_tiles: config.num_tiles(),
                side_info_tiles,
                config_bands: config.num_scale_factor_bands(),
                side_info_bands,
            });
        }

        self.copy_tiles(config, spectrum);
        self.apply_whitening(config, side_info, spectrum, random);
        apply_envelopes(config, side_info, spectrum);
        spectrum[config.stop_line()..].fill(0.0);
        Ok(())
    }

    fn copy_tiles(&mut self, config: BweConfig, spectrum: &[f32]) {
        self.copied_spectrum.fill(0.0);
        self.copied_spectrum[..config.start_line()]
            .copy_from_slice(&spectrum[..config.start_line()]);

        for tile in 0..config.num_tiles() {
            let start = usize::from(config.target_tiles()[tile]);
            let stop = usize::from(config.target_tiles()[tile + 1]);
            let source = usize::from(config.source_tiles()[tile]);
            let width = stop - start;
            self.copied_spectrum[start..stop].copy_from_slice(&spectrum[source..source + width]);
        }
    }

    fn apply_whitening(
        &self,
        config: BweConfig,
        side_info: BweSideInfo,
        spectrum: &mut [f32],
        random: &mut Avs3Random,
    ) {
        for (tile, &level) in side_info.whitening_levels().iter().enumerate() {
            let start = usize::from(config.target_tiles()[tile]);
            let stop = usize::from(config.target_tiles()[tile + 1]);
            match level {
                BweWhiteningLevel::Off => {
                    spectrum[start..stop].copy_from_slice(&self.copied_spectrum[start..stop])
                }
                BweWhiteningLevel::Mid => {
                    for (offset, output) in spectrum[start..stop].iter_mut().enumerate() {
                        let line = start + offset;
                        let mut square_sum = 0.0_f32;
                        for &value in
                            &self.copied_spectrum[line - WHITENING_RADIUS..=line + WHITENING_RADIUS]
                        {
                            square_sum += value * value;
                        }
                        let average = (square_sum / WHITENING_WIDTH).sqrt();
                        *output = if average == 0.0 {
                            self.copied_spectrum[line]
                        } else {
                            self.copied_spectrum[line] / average
                        };
                    }
                }
                BweWhiteningLevel::High => {
                    let mut absolute_sum = 0.0_f32;
                    for &value in &self.copied_spectrum[start..stop] {
                        absolute_sum += value.abs();
                    }
                    if absolute_sum > 0.0 {
                        for output in &mut spectrum[start..stop] {
                            *output = (random.next_u31() as f32 / AVS3_RAND_MAX as f32) * 2.0 - 1.0;
                        }
                    } else {
                        spectrum[start..stop].fill(0.0);
                    }
                }
            }
        }
    }
}

impl Default for BweSynthesis {
    fn default() -> Self {
        Self::new()
    }
}

fn apply_envelopes(config: BweConfig, side_info: BweSideInfo, spectrum: &mut [f32]) {
    for (band, &envelope_index) in side_info.envelope_indexes().iter().enumerate() {
        let start = usize::from(config.scale_factor_bands()[band]);
        let stop = usize::from(config.scale_factor_bands()[band + 1]);
        let mut current_energy = 0.0_f32;
        for &value in &spectrum[start..stop] {
            current_energy += value * value;
        }
        current_energy /= (stop - start) as f32;

        let gain = if current_energy != 0.0 {
            let target_energy = f32::from_bits(TARGET_ENERGY_BITS[usize::from(envelope_index)]);
            (target_energy / current_energy).sqrt()
        } else {
            1.0
        };
        for value in &mut spectrum[start..stop] {
            *value *= gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BitWriter, CoreBitstreamConfig, LsfCodebookMode, MonoSideInfoDecoder, NnType,
        ParsedNeuralQc,
    };

    const MAIN_REFERENCE_PAYLOAD: [u8; 35] = [
        0x44, 0x72, 0x61, 0x63, 0xb6, 0x23, 0xa0, 0xf0, 0xea, 0x00, 0xfb, 0xdc, 0x10, 0x30, 0x2f,
        0x40, 0x2a, 0x40, 0xc0, 0xff, 0x6f, 0x01, 0xc7, 0xe9, 0x5f, 0x03, 0x84, 0xa0, 0xd8, 0x7f,
        0xfd, 0x51, 0xf6, 0xf2, 0x00,
    ];

    fn side_info_64k() -> (BweConfig, BweSideInfo) {
        let config = BweConfig::for_mono_bitrate(64_000).unwrap().unwrap();
        let bitstream_config = CoreBitstreamConfig::new(
            NnType::Main,
            277,
            LsfCodebookMode::HighBitrate,
            Some(config),
        )
        .unwrap();
        let mut decoder = MonoSideInfoDecoder::new();
        let parsed = decoder
            .parse(&MAIN_REFERENCE_PAYLOAD, bitstream_config)
            .unwrap();
        assert!(matches!(parsed.neural_qc(), ParsedNeuralQc::Main(_)));
        (config, parsed.core().bwe().unwrap())
    }

    fn side_info_32k() -> (BweConfig, BweSideInfo) {
        let config = BweConfig::for_mono_bitrate(32_000).unwrap().unwrap();
        let mut writer = BitWriter::new();
        writer.write_bits(0, 2).unwrap();
        for width in [8, 8, 7, 7, 6, 5, 5] {
            writer.write_bits(0, width).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        for envelope in [0, 17, 37, 63, 91, 127] {
            writer.write_bits(envelope, 7).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 7).unwrap();
        writer.write_bits(0, 3).unwrap();
        writer.write_bits(0, 8).unwrap();
        let payload_bits = writer.bit_len();
        let payload = writer.into_bytes();
        let bitstream_config = CoreBitstreamConfig::new(
            NnType::Main,
            payload_bits,
            LsfCodebookMode::HighBitrate,
            Some(config),
        )
        .unwrap();
        let mut decoder = MonoSideInfoDecoder::new();
        let parsed = decoder.parse(&payload, bitstream_config).unwrap();
        (config, parsed.core().bwe().unwrap())
    }

    fn deterministic_spectrum() -> [f32; AVS3_FEATURE_DIMENSIONS] {
        let mut spectrum = [0.0_f32; AVS3_FEATURE_DIMENSIONS];
        let mut state = 0x91e1_0da5_u32;
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
    fn mono_32k_all_whitening_modes_match_c() {
        let (config, side_info) = side_info_32k();
        assert_eq!(
            side_info.whitening_levels(),
            [
                BweWhiteningLevel::Off,
                BweWhiteningLevel::Mid,
                BweWhiteningLevel::High,
            ]
        );
        let mut spectrum = deterministic_spectrum();
        let mut random = Avs3Random::new();
        BweSynthesis::new()
            .apply(config, side_info, &mut spectrum, &mut random)
            .unwrap();

        assert_eq!(fingerprint(&spectrum), 0x8be4_1361_14fb_34a5);
        let positions = [
            0, 351, 352, 353, 415, 479, 480, 543, 544, 607, 608, 671, 672, 767, 768, 831, 832, 900,
            1023,
        ];
        assert_eq!(
            positions.map(|index| spectrum[index].to_bits()),
            [
                0xbf75_a237,
                0xbf40_d4c6,
                0xbe9b_a419,
                0x3e8d_4ad8,
                0x3e87_3bf8,
                0xbf99_1960,
                0x4065_23a9,
                0xc0ac_a9d1,
                0xc200_a823,
                0x4215_3bb6,
                0x43ed_2f34,
                0x43f4_aabd,
                0xc5ca_2973,
                0x460a_7c27,
                0,
                0,
                0,
                0,
                0,
            ]
        );
        assert_eq!(random.next_u31(), 1_231_192_379);
    }

    #[test]
    fn mono_64k_reference_side_info_matches_c() {
        let (config, side_info) = side_info_64k();
        let mut spectrum = deterministic_spectrum();
        let mut random = Avs3Random::new();
        BweSynthesis::new()
            .apply(config, side_info, &mut spectrum, &mut random)
            .unwrap();

        assert_eq!(fingerprint(&spectrum), 0xe8de_94e6_7b24_f1df);
        let positions = [
            0, 351, 352, 353, 415, 479, 480, 543, 544, 607, 608, 671, 672, 767, 768, 831, 832, 900,
            1023,
        ];
        assert_eq!(
            positions.map(|index| spectrum[index].to_bits()),
            [
                0xbf75_a237,
                0xbf40_d4c6,
                0xbf58_4f3a,
                0x3f5e_1445,
                0xbf06_74bd,
                0xbf11_39de,
                0xbf71_743b,
                0x3f78_4358,
                0x3e8e_8a1f,
                0x3eb4_3f9e,
                0x45ad_1ae1,
                0x45b0_7100,
                0x41c9_7195,
                0x428a_0134,
                0xc184_3ce6,
                0x4250_1d1d,
                0,
                0,
                0,
            ]
        );
        assert_eq!(random.next_u31(), 1_231_192_379);
    }

    #[test]
    fn rejects_wrong_length_before_consuming_random_state() {
        let (config, side_info) = side_info_64k();
        let mut short_spectrum = [1.0_f32; 17];
        let mut random = Avs3Random::new();
        let error = BweSynthesis::new()
            .apply(config, side_info, &mut short_spectrum, &mut random)
            .unwrap_err();
        assert_eq!(
            error,
            BweSynthesisError::InvalidSpectrumLength {
                expected: AVS3_FEATURE_DIMENSIONS,
                actual: 17,
            }
        );
        assert_eq!(random.next_u31(), 1_804_289_383);
        assert_eq!(short_spectrum, [1.0; 17]);
    }
}
