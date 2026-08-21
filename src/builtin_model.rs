use std::sync::OnceLock;

use crate::model::{ModelEncoding, ModelError, NeuralModel, NeuralModelType};

pub const BUILTIN_MODEL_LEN: usize = 79_930;
pub const BUILTIN_MODEL_FNV1A: u64 = 0xc16c_f4fc_1e16_52b0;

static BUILTIN_MODEL_BYTES: &[u8; BUILTIN_MODEL_LEN] =
    include_bytes!("../assets/avs3a_hyper_model.bin");
static BUILTIN_MODEL: OnceLock<Result<NeuralModel, ModelError>> = OnceLock::new();

/// Plaintext bytes of the AVS3-P3 hyper-prior model shipped with the C
/// reference implementation.
pub fn builtin_model_bytes() -> &'static [u8; BUILTIN_MODEL_LEN] {
    BUILTIN_MODEL_BYTES
}

/// Parse the built-in model once per process and share its immutable weights
/// between decoder instances.
pub fn builtin_neural_model() -> Result<&'static NeuralModel, ModelError> {
    match BUILTIN_MODEL.get_or_init(|| {
        NeuralModel::from_bytes(
            BUILTIN_MODEL_BYTES,
            NeuralModelType::Hyper,
            ModelEncoding::Plain,
        )
    }) {
        Ok(model) => Ok(model),
        Err(error) => Err(error.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::ScalarCnnDecoder;
    use crate::latent::LatentShape;

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        })
    }

    fn feature_fingerprint(values: &[f32]) -> u64 {
        values
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
                (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3)
            })
    }

    fn deterministic_input(
        shape: LatentShape,
        dimension_factor: usize,
        channel_factor: usize,
        modulus: usize,
        offset: usize,
        divisor: usize,
    ) -> Vec<f32> {
        let mut input = vec![0.0_f32; shape.len()];
        for channel in 0..shape.channels() {
            for dimension in 0..shape.dimensions() {
                let value = (dimension * dimension_factor + channel * channel_factor) % modulus;
                input[dimension + channel * shape.dimensions()] =
                    (value as f32 - offset as f32) / divisor as f32;
            }
        }
        input
    }

    #[test]
    fn embedded_plaintext_asset_has_expected_fingerprint_and_topology() {
        assert_eq!(fnv1a(builtin_model_bytes()), BUILTIN_MODEL_FNV1A);
        let model = builtin_neural_model().unwrap();
        assert_eq!(
            model.base().latent_shape(),
            LatentShape::new(64, 16).unwrap()
        );
        assert_eq!(model.base().decoder().layers().len(), 4);
        let context = model.context().unwrap();
        assert_eq!(context.latent_shape(), LatentShape::new(16, 16).unwrap());
        assert_eq!(context.decoder().layers().len(), 3);
    }

    #[test]
    fn complete_builtin_decoders_match_c_reference_fingerprints() {
        let model = builtin_neural_model().unwrap();
        let context = model.context().unwrap();
        let context_input = deterministic_input(context.latent_shape(), 17, 13, 31, 15, 16);
        let mut context_decoder = ScalarCnnDecoder::new(context.decoder()).unwrap();
        let context_output = context_decoder.decode(&context_input).unwrap();
        assert_eq!(feature_fingerprint(context_output), 0x68fb_ea61_c73b_efc9);
        assert_eq!(context_output[0].to_bits(), 0x3e66_f73f);
        assert_eq!(context_output.last().unwrap().to_bits(), 0xbde0_387b);

        let base = model.base();
        let base_input = deterministic_input(base.latent_shape(), 19, 11, 47, 23, 32);
        let mut base_decoder = ScalarCnnDecoder::new(base.decoder()).unwrap();
        let base_output = base_decoder.decode(&base_input).unwrap();
        assert_eq!(feature_fingerprint(base_output), 0x011e_720d_9e64_3a65);
        assert_eq!(base_output[0].to_bits(), 0x4382_8b6b);
        assert_eq!(base_output.last().unwrap().to_bits(), 0xc096_8429);
    }
}
