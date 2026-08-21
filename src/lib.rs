//! Safe building blocks for an AVS3-P3 (AV3A) decoder.
//!
//! The original implementation mixes frame framing, decoder state, DSP
//! buffers and file I/O in one C object graph. This crate keeps those concerns
//! separated: all sizes are checked before allocation or indexing, temporal
//! state owns its buffers, and complete channel-based, Mix and HOA synthesis
//! pipelines are isolated behind [`backend::DecoderBackend`].
//!
//! # Where to start
//!
//! To play or transcode a file, the root re-exports below are enough: sniff the
//! input with [`is_iso_bmff`], feed it through [`Mp4FrameReader`] or
//! [`FrameStream`], and hand each [`EncodedFrame`] to a [`BuiltinDecoder`].
//!
//! # Module map
//!
//! Everything else is grouped by the stage of the pipeline it belongs to, so a
//! reader can find one stage without scrolling past all the others.
//!
//! | Module | Stage |
//! |---|---|
//! | [`bitstream`] | MSB-first bit reader and writer |
//! | [`header`] | frame header parsing and stream configuration |
//! | [`stream`] | elementary-stream framing and resynchronisation |
//! | [`mp4`] | MP4/M4A container demux and sample indexing |
//! | [`decode`] | decoder driver, warm-up depth and PCM16 quantisation |
//! | [`backend`] | the synthesis backends and the trait they implement |
//! | [`metadata`] | frame metadata payloads and their ADM value types |
//! | [`side_info`] | core bitstream side information shared by all profiles |
//! | [`mono`] | mono core decoder |
//! | [`stereo`] | stereo side info, core decoder and MCR |
//! | [`multichannel`] | multichannel side info, pairing, ILD and core decoder |
//! | [`hoa`] | HOA side info, core decoder and spatial post-synthesis |
//! | [`dsp`] | transforms and spectral stages shared across profiles |
//! | [`entropy`] | range decoding and latent (de)quantisation |
//! | [`neural`] | neural model loading, CNN evaluation and neural QC |
//! | [`wav`] | PCM16 WAV output |
//!
//! Items reachable through these modules are public so that individual stages
//! can be exercised against the reference implementation. Their grouping is
//! the stable contract; which file a stage happens to live in is not.

#![forbid(unsafe_code)]

mod builtin_decoder;
mod builtin_model;
mod bwe;
mod cnn;
mod core_side;
mod decoder;
mod error;
mod fd_shaping;
mod feature_scale_tables;
mod hoa_backend;
mod hoa_core;
mod hoa_side;
mod hoa_synthesis;
mod imdct;
mod latent;
mod mc_backend;
mod mc_core;
mod mc_side;
mod mcr;
mod mdct;
mod mdct_synthesis;
mod metadata_values;
mod mix_backend;
mod model;
mod mono_backend;
mod mono_core;
mod neural_qc;
mod random;
mod range_coder;
mod spectrum;
mod stereo_backend;
mod stereo_core;
mod stereo_side;
mod tns;

pub mod bitstream;
pub mod header;
pub mod metadata;
pub mod mp4;
pub mod stream;
pub mod wav;

/// Decoder driver: frame validation, warm-up depth and PCM16 quantisation.
///
/// [`decode::Decoder`] owns the checked framing and state management around a
/// [`backend::DecoderBackend`]; [`decode::BuiltinDecoder`] picks the right
/// backend from a frame header so callers do not have to.
pub mod decode {
    pub use crate::builtin_decoder::{BuiltinDecoder, CHANNEL_WARMUP_FRAMES, HOA_WARMUP_FRAMES};
    pub use crate::decoder::{
        AudioFrame, Decoder, DecoderConfig, FLOAT_FULL_SCALE, PendingDecoder,
    };
    pub use crate::error::DecodeError;
    pub use crate::mono_backend::float_to_pcm16;
}

/// The synthesis backends and the trait they implement.
///
/// A backend turns one validated frame into interleaved floats. Selecting one
/// by hand is only necessary when a stream's profile is already known;
/// otherwise use [`crate::decode::BuiltinDecoder`].
pub mod backend {
    pub use crate::decoder::DecoderBackend;
    pub use crate::hoa_backend::HoaDecoderBackend;
    pub use crate::mc_backend::McDecoderBackend;
    pub use crate::mix_backend::{MixCoreKind, MixDecoderBackend};
    pub use crate::mono_backend::MonoDecoderBackend;
    pub use crate::stereo_backend::StereoDecoderBackend;
}

/// Core bitstream side information shared by every coding profile.
///
/// These are the parsed per-frame coding decisions — window grouping, LSF,
/// TNS and bandwidth-extension parameters — that sit between the frame header
/// and the profile-specific cores.
pub mod side_info {
    pub use crate::core_side::{
        BweConfig, BweSideInfo, BweWhiteningLevel, CoreBitstreamConfig, CoreBitstreamError,
        CoreSideInfo, LsfCodebookMode, LsfSideInfo, MAX_BWE_SCALE_FACTOR_BANDS, MAX_BWE_TILES,
        MAX_LSF_CODEBOOKS, MAX_TNS_FILTERS, MAX_TNS_ORDER, MonoFrameSideInfo, MonoSideInfoDecoder,
        ParsedNeuralQc, TnsCoefficient, TnsFilterSideInfo, TnsSideInfo, TransformType,
        WindowGrouping,
    };
}

/// Mono core decoder.
pub mod mono {
    pub use crate::mono_core::{MonoCoreDecodeError, MonoCoreDecoder, MonoCoreDiagnostics};
}

/// Stereo side information, core decoder and MCR.
///
/// The reference switches 24/32 kbps streams from mid/side to MCR, so both
/// paths live here.
pub mod stereo {
    pub use crate::mcr::{
        MCR_LONG_CODEBOOK_ENTRIES, MCR_LONG_INDEX_BITS, MCR_ROTATION_BYTES_LEN, MCR_ROTATION_FNV1A,
        MCR_ROTATION_VALUES, MCR_SCALE_FACTOR_BANDS, MCR_SHORT_CODEBOOK_ENTRIES,
        MCR_SHORT_INDEX_BITS, MCR_SUBSPECTRA, MCR_SUBVECTOR_DIMENSIONS, MCR_SUBVECTORS, McrError,
        McrSideInfo, McrSynthesis, mcr_rotation_bytes,
    };
    pub use crate::stereo_core::{
        McrCoreDiagnostics, STEREO_FRAME_SAMPLES, StereoCoreDecodeError, StereoCoreDecoder,
        StereoCoreDiagnostics,
    };
    pub use crate::stereo_side::{
        McrFrameSideInfo, STEREO_CHANNELS, STEREO_MCR_BITRATE_THRESHOLD, StereoCodingMode,
        StereoError, StereoFrameSideInfo, StereoSideInfo, StereoSideInfoDecoder, inverse_mid_side,
        stereo_bytes_allocation,
    };
}

/// Multichannel side information, channel pairing, ILD and core decoder.
pub mod multichannel {
    pub use crate::mc_core::{
        MC_MAX_FRAME_SAMPLES, McCoreDecodeError, McCoreDecoder, McCoreDiagnostics,
    };
    pub use crate::mc_side::{
        MAX_MC_PAIRS, MC_ILD_CODEBOOK, MC_ILD_CODEBOOK_LEN, MC_LFE_CHANNEL_INDEX,
        MC_LFE_RESERVED_LINES, MC_NO_ILD_INDEX, MC_SILENCE_BYTES, McBitstreamConfig,
        McByteAllocation, McError, McFrameSideInfo, McPair, McSideInfo, McSideInfoDecoder,
        apply_mc_ild, clear_mc_lfe_spectrum, inverse_mc_coupling, inverse_mc_pair,
        is_multichannel_config, mc_bytes_allocation, mc_coupling_channel_to_output,
        mc_output_channel_to_coupling, mc_pair_from_index, mc_pair_index_bits,
    };
}

/// HOA side information, core decoder and spatial post-synthesis.
pub mod hoa {
    pub use crate::hoa_core::{
        HOA_MAX_FRAME_SAMPLES, HoaCoreDecodeError, HoaCoreDecoder, HoaCoreDiagnostics,
    };
    pub use crate::hoa_side::{
        HOA_BASIS_INDEX_BITS, HOA_BASIS_TABLE_LEN, HOA_SFB_BOUNDARIES, HOA_SFB_COUNT,
        HoaBitstreamConfig, HoaByteAllocation, HoaDmxMode, HoaError, HoaFrameSideInfo,
        HoaGroupConfig, HoaGroupSideInfo, HoaPairSideInfo, HoaSideInfo, HoaSideInfoDecoder,
        MAX_HOA_BASIS, MAX_HOA_GROUP_PAIRS, MAX_HOA_GROUPS, hoa_bytes_allocation,
        hoa_pair_from_index, hoa_pair_index_bits, inverse_hoa_dmx,
    };
    pub use crate::hoa_synthesis::{
        HOA_BASIS_DELAY_FRAMES, HOA_FRAME_SAMPLES, HOA_OVERLAP_SIZE, HOA_POST_TRANSFORM_LEN,
        HOA_SPATIAL_TABLE_BYTES_LEN, HOA_SPATIAL_TABLE_FNV1A, HoaPostSynthesis,
        HoaPostSynthesisError, hoa_basis_coefficients, hoa_spatial_table_bytes,
    };
}

/// Transforms and spectral stages shared across profiles.
///
/// Each stage is separately constructible so it can be compared against the
/// reference implementation in isolation.
pub mod dsp {
    pub use crate::bwe::{BweSynthesis, BweSynthesisError};
    pub use crate::fd_shaping::{
        FD_TABLE_BYTES_LEN, FD_TABLE_FNV1A, FD_TABLE_VALUES, FdShapingError, FdSpectrumShaping,
        fd_table_bytes,
    };
    pub use crate::imdct::{FastImdct, ImdctError};
    pub use crate::mdct::{FastMdct, MdctError};
    pub use crate::mdct_synthesis::{MdctSynthesis, MdctSynthesisError};
    pub use crate::random::{AVS3_RAND_MAX, Avs3Random};
    pub use crate::spectrum::{SpectrumReorder, SpectrumReorderError};
    pub use crate::tns::{TnsSynthesis, TnsSynthesisError};
}

/// Range decoding and latent (de)quantisation.
pub mod entropy {
    pub use crate::latent::{
        ContextScaleTable, LatentError, LatentShape, MAX_LATENT_CHANNELS, MAX_LATENT_DIMENSIONS,
        MAX_LATENT_VALUES, Quantizer, channel_cdf_indexes, channel_cdf_indexes_into,
        flatten_for_entropy_coder, flatten_for_entropy_coder_into, unflatten_from_entropy_coder,
        unflatten_from_entropy_coder_into,
    };
    pub use crate::range_coder::{RangeCoderConfig, RangeCoderError, RangeDecoder};
}

/// Neural model loading, CNN evaluation and neural quantisation/coding.
pub mod neural {
    pub use crate::builtin_model::{
        BUILTIN_MODEL_FNV1A, BUILTIN_MODEL_LEN, builtin_model_bytes, builtin_neural_model,
    };
    pub use crate::cnn::{CnnError, MAX_CNN_WORKSPACE_VALUES, ScalarCnnDecoder};
    pub use crate::model::{
        AVS3_FEATURE_DIMENSIONS, AVS3_MODEL_XOR_MASK, Activation, CnnLayer, CnnNetwork,
        DEFAULT_MAX_KERNEL_SIZE, DEFAULT_MAX_MODEL_BYTES, DEFAULT_MAX_MODEL_CHANNELS,
        DEFAULT_MAX_MODEL_LAYERS, DEFAULT_MAX_MODEL_VALUES, GdnParameters, ModelEncoding,
        ModelError, ModelLimits, ModelReader, NeuralCodecModel, NeuralModel, NeuralModelType,
        Padding,
    };
    pub use crate::neural_qc::{
        AVS3_NOISE_GROUPS, AVS3_SHORT_BLOCKS, DecodedNeuralSpectrum, LowComplexityNeuralQc,
        MAX_MAIN_SCALE_INDEX, MAX_NOISE_FILLING_INDEX, MAX_QC_BITSTREAM_BYTES, MainNeuralQc,
        NeuralBitstreams, NeuralQcError, NeuralSpectrumDecoder, NeuralSpectrumDiagnostics,
        NoiseFilling, NoiseGroup,
    };
}

// The decode path, re-exported so the common case needs one import.
pub use crate::decode::{BuiltinDecoder, DecodeError};
pub use crate::header::{FrameHeader, parse_header};
pub use crate::mp4::{Av3aTrack, Mp4FrameReader, is_iso_bmff};
pub use crate::stream::{EncodedFrame, FrameStream, StreamEvent};
pub use crate::wav::WavWriter;

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
