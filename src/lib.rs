//! Safe building blocks for an AVS3-P3 (AV3A) decoder.
//!
//! The original implementation mixes frame framing, decoder state, DSP
//! buffers and file I/O in one C object graph. This crate keeps those concerns
//! separated: all sizes are checked before allocation or indexing, temporal
//! state owns its buffers, and complete channel-based, Mix and HOA synthesis
//! pipelines are isolated behind [`DecoderBackend`].

#![forbid(unsafe_code)]

mod bitstream;
mod builtin_decoder;
mod builtin_model;
mod bwe;
mod cnn;
mod core_side;
mod decoder;
mod error;
mod fd_shaping;
mod feature_scale_tables;
mod header;
mod hoa;
mod hoa_backend;
mod hoa_core;
mod hoa_synthesis;
mod imdct;
mod latent;
mod mc;
mod mc_backend;
mod mc_core;
mod mcr;
mod mdct;
mod mdct_synthesis;
mod metadata;
mod metadata_values;
mod mix_backend;
mod model;
mod mono;
mod mono_backend;
mod neural_qc;
mod random;
mod range_coder;
mod spectrum;
mod stereo;
mod stereo_backend;
mod stereo_core;
mod stream;
mod tns;
mod wav;

pub use bitstream::{BitReader, BitWriter};
pub use builtin_decoder::BuiltinDecoder;
pub use builtin_model::{
    BUILTIN_MODEL_FNV1A, BUILTIN_MODEL_LEN, builtin_model_bytes, builtin_neural_model,
};
pub use bwe::{BweSynthesis, BweSynthesisError};
pub use cnn::{CnnError, MAX_CNN_WORKSPACE_VALUES, ScalarCnnDecoder};
pub use core_side::{
    BweConfig, BweSideInfo, BweWhiteningLevel, CoreBitstreamConfig, CoreBitstreamError,
    CoreSideInfo, LsfCodebookMode, LsfSideInfo, MAX_BWE_SCALE_FACTOR_BANDS, MAX_BWE_TILES,
    MAX_LSF_CODEBOOKS, MAX_TNS_FILTERS, MAX_TNS_ORDER, MonoFrameSideInfo, MonoSideInfoDecoder,
    ParsedNeuralQc, TnsCoefficient, TnsFilterSideInfo, TnsSideInfo, TransformType, WindowGrouping,
};
pub use decoder::{AudioFrame, Decoder, DecoderBackend, DecoderConfig, PendingDecoder};
pub use error::{BitstreamError, DecodeError, HeaderError, StreamError, WavError};
pub use fd_shaping::{
    FD_TABLE_BYTES_LEN, FD_TABLE_FNV1A, FD_TABLE_VALUES, FdShapingError, FdSpectrumShaping,
    fd_table_bytes,
};
pub use header::{
    AudioCodecId, BitDepth, ChannelConfig, CodecProfile, FrameHeader, HeaderInfo, MAX_CHANNELS,
    MAX_HEADER_BYTES, MAX_PAYLOAD_BYTES, NnType, SoundBedType,
};
pub use hoa::{
    HOA_BASIS_INDEX_BITS, HOA_BASIS_TABLE_LEN, HOA_SFB_BOUNDARIES, HOA_SFB_COUNT,
    HoaBitstreamConfig, HoaByteAllocation, HoaDmxMode, HoaError, HoaFrameSideInfo, HoaGroupConfig,
    HoaGroupSideInfo, HoaPairSideInfo, HoaSideInfo, HoaSideInfoDecoder, MAX_HOA_BASIS,
    MAX_HOA_GROUP_PAIRS, MAX_HOA_GROUPS, hoa_bytes_allocation, hoa_pair_from_index,
    hoa_pair_index_bits, inverse_hoa_dmx,
};
pub use hoa_backend::HoaDecoderBackend;
pub use hoa_core::{HOA_MAX_FRAME_SAMPLES, HoaCoreDecodeError, HoaCoreDecoder, HoaCoreDiagnostics};
pub use hoa_synthesis::{
    HOA_FRAME_SAMPLES, HOA_OVERLAP_SIZE, HOA_POST_TRANSFORM_LEN, HOA_SPATIAL_TABLE_BYTES_LEN,
    HOA_SPATIAL_TABLE_FNV1A, HoaPostSynthesis, HoaPostSynthesisError, hoa_basis_coefficients,
    hoa_spatial_table_bytes,
};
pub use imdct::{FastImdct, ImdctError};
pub use latent::{
    ContextScaleTable, LatentError, LatentShape, MAX_LATENT_CHANNELS, MAX_LATENT_DIMENSIONS,
    MAX_LATENT_VALUES, Quantizer, channel_cdf_indexes, channel_cdf_indexes_into,
    flatten_for_entropy_coder, flatten_for_entropy_coder_into, unflatten_from_entropy_coder,
    unflatten_from_entropy_coder_into,
};
pub use mc::{
    MAX_MC_PAIRS, MC_ILD_CODEBOOK, MC_ILD_CODEBOOK_LEN, MC_LFE_CHANNEL_INDEX,
    MC_LFE_RESERVED_LINES, MC_NO_ILD_INDEX, MC_SILENCE_BYTES, McBitstreamConfig, McByteAllocation,
    McError, McFrameSideInfo, McPair, McSideInfo, McSideInfoDecoder, apply_mc_ild,
    clear_mc_lfe_spectrum, inverse_mc_coupling, inverse_mc_pair, is_multichannel_config,
    mc_bytes_allocation, mc_coupling_channel_to_output, mc_output_channel_to_coupling,
    mc_pair_from_index, mc_pair_index_bits,
};
pub use mc_backend::McDecoderBackend;
pub use mc_core::{MC_MAX_FRAME_SAMPLES, McCoreDecodeError, McCoreDecoder, McCoreDiagnostics};
pub use mcr::{
    MCR_LONG_CODEBOOK_ENTRIES, MCR_LONG_INDEX_BITS, MCR_ROTATION_BYTES_LEN, MCR_ROTATION_FNV1A,
    MCR_ROTATION_VALUES, MCR_SCALE_FACTOR_BANDS, MCR_SHORT_CODEBOOK_ENTRIES, MCR_SHORT_INDEX_BITS,
    MCR_SUBSPECTRA, MCR_SUBVECTOR_DIMENSIONS, MCR_SUBVECTORS, McrError, McrSideInfo, McrSynthesis,
    mcr_rotation_bytes,
};
pub use mdct::{FastMdct, MdctError};
pub use mdct_synthesis::{MdctSynthesis, MdctSynthesisError};
pub use metadata::{
    DynamicMetadataSummary, MAX_DYNAMIC_METADATA_OBJECTS, METADATA_PRESENCE_BITS, MetadataError,
    MetadataPayloadParser, MetadataSummary, ParsedMetadataPayload, StaticMetadataSummary,
};
pub use metadata_values::{
    AudioChannelMetadata, AudioContentMetadata, AudioObjectInteractionMetadata,
    AudioObjectMetadata, AudioPackMetadata, AudioProgrammeMetadata, BasicMetadata,
    ChannelLockMetadata, ComplementaryObjectGroup, DialogueMetadata, DirectSpeakerPosition,
    DynamicLevel1Metadata, DynamicLevel2Metadata, DynamicMetadata, DynamicObjectExtent,
    DynamicObjectMetadata, DynamicObjectPosition, FrameMetadata, GainInteractionMetadata,
    HoaPackMetadata, LoudnessMetadata, MetadataGain, MetadataGainUnit, MetadataList,
    ObjectDivergenceMetadata, PackChannelReference, PositionInteractionMetadata,
    ProgrammeReferenceScreen, ProgrammeScreenPosition, StaticMetadata,
    VrAcousticEnvironmentMetadata, VrAudioEffectMetadata, VrDrcMetadata, VrEqBandMetadata,
    VrExtensionMetadata, VrRenderInfoMetadata, VrSurfaceMetadata, VrVertex,
};
pub use mix_backend::{MixCoreKind, MixDecoderBackend};
pub use model::{
    AVS3_FEATURE_DIMENSIONS, AVS3_MODEL_XOR_MASK, Activation, CnnLayer, CnnNetwork,
    DEFAULT_MAX_KERNEL_SIZE, DEFAULT_MAX_MODEL_BYTES, DEFAULT_MAX_MODEL_CHANNELS,
    DEFAULT_MAX_MODEL_LAYERS, DEFAULT_MAX_MODEL_VALUES, GdnParameters, ModelEncoding, ModelError,
    ModelLimits, ModelReader, NeuralCodecModel, NeuralModel, NeuralModelType, Padding,
};
pub use mono::{MonoCoreDecodeError, MonoCoreDecoder, MonoCoreDiagnostics};
pub use mono_backend::{MonoDecoderBackend, float_to_pcm16};
pub use neural_qc::{
    AVS3_NOISE_GROUPS, AVS3_SHORT_BLOCKS, DecodedNeuralSpectrum, LowComplexityNeuralQc,
    MAX_MAIN_SCALE_INDEX, MAX_NOISE_FILLING_INDEX, MAX_QC_BITSTREAM_BYTES, MainNeuralQc,
    NeuralBitstreams, NeuralQcError, NeuralSpectrumDecoder, NeuralSpectrumDiagnostics,
    NoiseFilling, NoiseGroup,
};
pub use random::{AVS3_RAND_MAX, Avs3Random};
pub use range_coder::{RangeCoderConfig, RangeCoderError, RangeDecoder};
pub use spectrum::{SpectrumReorder, SpectrumReorderError};
pub use stereo::{
    McrFrameSideInfo, STEREO_CHANNELS, STEREO_MCR_BITRATE_THRESHOLD, StereoCodingMode, StereoError,
    StereoFrameSideInfo, StereoSideInfo, StereoSideInfoDecoder, inverse_mid_side,
    stereo_bytes_allocation,
};
pub use stereo_backend::StereoDecoderBackend;
pub use stereo_core::{
    McrCoreDiagnostics, STEREO_FRAME_SAMPLES, StereoCoreDecodeError, StereoCoreDecoder,
    StereoCoreDiagnostics,
};
pub use stream::{EncodedFrame, FrameStream, StreamEvent};
pub use tns::{TnsSynthesis, TnsSynthesisError};
pub use wav::WavWriter;

/// Parse the first complete AV3A header in `input`.
///
/// This is a convenience wrapper for applications that do not need an
/// incremental stream parser.  `HeaderInfo::offset` reports the number of
/// bytes of leading data before the sync word.
pub fn parse_header(input: &[u8]) -> Result<HeaderInfo, HeaderError> {
    header::parse_header(input)
}

/// Calculate the CRC used by the AVS3 reference implementation.
///
/// It uses the CCITT `0x1021` polynomial and an initial value of `0xffff`,
/// but its byte recurrence is not the standard CRC-16/CCITT recurrence.
/// Keeping this distinction explicit prevents a superficially reasonable but
/// bitstream-incompatible replacement.
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0xffff_u16;
    for &byte in data {
        let table_index = (crc >> 8) as u8;
        let mut table_value = u16::from(table_index) << 8;
        for _ in 0..8 {
            table_value = if table_value & 0x8000 != 0 {
                (table_value << 1) ^ 0x1021
            } else {
                table_value << 1
            };
        }
        crc = (crc << 8) | u16::from(byte);
        crc ^= table_value;
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    // The first nine bytes of the repository's MC 7.1.4/48 kHz sample.
    const SAMPLE_HEADER: [u8; 9] = [0xff, 0xf2, 0x00, 0x71, 0xa2, 0x94, 0x1b, 0x0e, 0x51];

    #[test]
    fn parses_reference_header() {
        let info = parse_header(&SAMPLE_HEADER).expect("reference header is valid");
        assert_eq!(info.offset, 0);
        assert_eq!(info.header.header_len, 7);
        assert_eq!(info.header.sample_rate, 44_100);
        assert_eq!(info.header.channels, 12);
        assert_eq!(info.header.bitrate, 832_000);
        assert_eq!(info.header.payload_len, 2_408);
        assert_eq!(info.header.frame_len, 2_415);
        assert_eq!(info.header.crc, 0x8d1b);
    }

    #[test]
    fn crc_matches_reference_recurrence() {
        assert_eq!(crc16(b"123456789"), 0xa69d);
    }
}
