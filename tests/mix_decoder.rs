use avs3a::backend::{MixCoreKind, MixDecoderBackend};
use avs3a::bitstream::{BitReader, BitWriter};
use avs3a::decode::PendingDecoder;
use avs3a::header::{AudioCodecId, BitDepth, ChannelConfig, CodecProfile, NnType, SoundBedType};
use avs3a::multichannel::{
    MC_LFE_RESERVED_LINES, McBitstreamConfig, McSideInfo, mc_bytes_allocation,
};
use avs3a::neural::AVS3_FEATURE_DIMENSIONS;
use avs3a::{FrameHeader, FrameStream, StreamEvent, crc16};

mod support;

use support::expected_rustfft_fingerprint;

const CONTEXT: [u8; 6] = [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7];
const BASE: [u8; 26] = [
    0x7f, 0xfd, 0x51, 0xf6, 0xf2, 0x24, 0x34, 0xad, 0x04, 0xde, 0x75, 0xcd, 0x9d, 0x0c, 0x76, 0xeb,
    0xb3, 0x76, 0xaf, 0x47, 0xda, 0x43, 0x33, 0xf0, 0xd4, 0xeb,
];

fn write_qc(writer: &mut BitWriter, entropy_bytes: usize, groups: usize) {
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(37, 7).unwrap();
    writer.write_bits(3, 3).unwrap();
    if groups == 2 {
        writer.write_bits(7, 3).unwrap();
    }
    writer.write_bits(CONTEXT.len() as u64, 8).unwrap();
    for byte in CONTEXT.into_iter().chain(BASE).take(entropy_bytes) {
        writer.write_bits(u64::from(byte), 8).unwrap();
    }
    for _ in (CONTEXT.len() + BASE.len()).min(entropy_bytes)..entropy_bytes {
        writer.write_bits(0, 8).unwrap();
    }
}

fn pad_to(writer: &mut BitWriter, target_bits: usize) {
    assert!(writer.bit_len() <= target_bits);
    while writer.bit_len() < target_bits {
        let width = (target_bits - writer.bit_len()).min(64);
        writer.write_bits(0, width).unwrap();
    }
}

fn append_bits(writer: &mut BitWriter, payload: &[u8], payload_bits: usize) {
    let mut reader = BitReader::with_bit_len(payload, payload_bits).unwrap();
    while reader.remaining() != 0 {
        let width = reader.remaining().min(64);
        writer
            .write_bits(reader.read_bits(width).unwrap(), width)
            .unwrap();
    }
}

fn with_empty_metadata(audio: &[u8], audio_bits: usize) -> Vec<u8> {
    let mut writer = BitWriter::new();
    writer.write_bits(0, 2).unwrap();
    append_bits(&mut writer, audio, audio_bits);
    writer.into_bytes()
}

fn with_muted_dynamic_metadata(audio: &[u8], audio_bits: usize) -> Vec<u8> {
    let mut writer = BitWriter::new();
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(0, 3).unwrap();
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(0, 5).unwrap();
    assert_eq!(writer.bit_len(), 11);
    append_bits(&mut writer, audio, audio_bits);
    writer.into_bytes()
}

fn mix_header_prefix(sample_rate_index: u64, crc: u16) -> BitWriter {
    let mut writer = BitWriter::new();
    for (value, width) in [
        (0xfff, 12),
        (2, 4),
        (0, 1),
        (0, 3),
        (1, 3),
        (sample_rate_index, 4),
        (u64::from(crc >> 8), 8),
    ] {
        writer.write_bits(value, width).unwrap();
    }
    writer
}

fn frame_objects_only(
    payload: &[u8],
    sample_rate_index: u64,
    objects: u8,
    object_bitrate_index: u64,
) -> Vec<u8> {
    let crc = crc16(payload);
    let mut writer = mix_header_prefix(sample_rate_index, crc);
    for (value, width) in [
        (0, 2),
        (u64::from(objects - 1), 7),
        (object_bitrate_index, 4),
        (1, 2),
        (u64::from(crc & 0xff), 8),
    ] {
        writer.write_bits(value, width).unwrap();
    }
    let mut frame = writer.into_bytes();
    frame.extend_from_slice(payload);
    frame
}

fn frame_bed_and_object(payload: &[u8]) -> Vec<u8> {
    let crc = crc16(payload);
    let mut writer = mix_header_prefix(2, crc);
    for (value, width) in [
        (1, 2),
        (u64::from(ChannelConfig::Mc5_1.index()), 7),
        (3, 4),
        (0, 7),
        (4, 4),
        (1, 2),
        (u64::from(crc & 0xff), 8),
    ] {
        writer.write_bits(value, width).unwrap();
    }
    let mut frame = writer.into_bytes();
    frame.extend_from_slice(payload);
    frame
}

fn parse_one_frame(bytes: &[u8]) -> avs3a::EncodedFrame {
    let mut stream = FrameStream::new();
    let mut events = Vec::new();
    for chunk in bytes.chunks(29) {
        events.extend(stream.push(chunk).unwrap());
    }
    events.extend(stream.finish().unwrap());
    let mut frames = events.into_iter().map(|event| match event {
        StreamEvent::Frame(frame) => frame,
        StreamEvent::Skipped { bytes } => panic!("unexpected resynchronization over {bytes} bytes"),
    });
    let frame = frames.next().expect("one complete frame");
    assert!(frames.next().is_none());
    frame
}

fn mono_audio(target_bits: usize) -> Vec<u8> {
    let mut writer = BitWriter::new();
    writer.write_bits(1, 2).unwrap();
    for (value, width) in [17_u64, 201, 66, 99, 45, 17, 3]
        .into_iter()
        .zip([8, 8, 7, 7, 6, 5, 5])
    {
        writer.write_bits(value, width).unwrap();
    }
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(2, 3).unwrap();
    for (code, bits) in [(0, 3), (481, 10), (27_136, 15)] {
        writer.write_bits(code, bits).unwrap();
    }
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(7, 3).unwrap();
    for (code, bits) in [
        (2, 2),
        (3, 2),
        (27, 5),
        (16, 5),
        (129, 9),
        (1_035, 11),
        (13_314, 14),
        (10_499, 14),
    ] {
        writer.write_bits(code, bits).unwrap();
    }
    for envelope in [1, 127, 55, 64] {
        writer.write_bits(envelope, 7).unwrap();
    }
    for value in [0, 1, 1] {
        writer.write_bits(value, 1).unwrap();
    }
    writer.write_bits(1, 1).unwrap();
    for group in [0, 0, 0, 1, 1, 1, 1, 1] {
        writer.write_bits(group, 1).unwrap();
    }
    let entropy_bytes = (target_bits - writer.bit_len() - 22) / 8;
    write_qc(&mut writer, entropy_bytes, 2);
    pad_to(&mut writer, target_bits);
    writer.into_bytes()
}

fn write_stereo_core_prefix(writer: &mut BitWriter, lsf: [u64; 5], envelopes: [u64; 6]) {
    writer.write_bits(1, 2).unwrap();
    for (value, width) in lsf.into_iter().zip([8, 8, 7, 7, 6]) {
        writer.write_bits(value, width).unwrap();
    }
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(0, 1).unwrap();
    for envelope in envelopes {
        writer.write_bits(envelope, 7).unwrap();
    }
    for _ in 0..3 {
        writer.write_bits(0, 1).unwrap();
    }
}

fn mcr_audio(target_bits: usize) -> Vec<u8> {
    let mut writer = BitWriter::new();
    write_stereo_core_prefix(&mut writer, [3, 5, 7, 9, 11], [1, 2, 3, 4, 5, 6]);
    write_stereo_core_prefix(&mut writer, [17, 19, 21, 23, 25], [7, 8, 9, 10, 11, 12]);
    writer.write_bits(1, 1).unwrap();
    for indicator in [0, 0, 0, 1, 1, 1, 1, 1] {
        writer.write_bits(indicator, 1).unwrap();
    }
    let indexes = [[1_u16, 2, 3, 4, 5, 255], [255_u16, 5, 4, 3, 2, 1]];
    for subvector in 0..6 {
        for subspectrum in &indexes {
            writer
                .write_bits(u64::from(subspectrum[subvector]), 8)
                .unwrap();
        }
    }
    let entropy_bytes = (target_bits - writer.bit_len() - 22) / 8;
    write_qc(&mut writer, entropy_bytes, 2);
    pad_to(&mut writer, target_bits);
    writer.into_bytes()
}

fn write_mc_core_prefix(writer: &mut BitWriter) {
    writer.write_bits(0, 2).unwrap();
    for width in [8, 8, 7, 7, 6, 5, 5] {
        writer.write_bits(0, width).unwrap();
    }
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(0, 1).unwrap();
}

fn write_mc_mode(writer: &mut BitWriter) {
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(2, 4).unwrap();
    for (pair_index, first_ild, second_ild) in [(0, 17, 30), (19, 8, 4)] {
        writer.write_bits(pair_index, 5).unwrap();
        writer.write_bits(first_ild, 5).unwrap();
        writer.write_bits(second_ild, 5).unwrap();
    }
    for ratio in [11, 11, 11, 10, 10, 11] {
        writer.write_bits(ratio, 6).unwrap();
    }
}

fn bed_object_audio(target_bits: usize) -> (Vec<u8>, Vec<usize>) {
    let header = FrameHeader {
        codec_id: AudioCodecId::Avs3P3,
        nn_type: NnType::Main,
        profile: CodecProfile::Mixed,
        sample_rate: 48_000,
        bit_depth: BitDepth::Sixteen,
        channel_config: Some(ChannelConfig::Mc5_1),
        sound_bed_type: Some(SoundBedType::ChannelBed),
        hoa_order: None,
        objects: 1,
        bed_channels: 6,
        channels: 7,
        has_lfe: true,
        bed_bitrate: Some(384_000),
        object_bitrate: Some(64_000),
        bitrate: 448_000,
        crc: 0,
        header_len: 9,
        payload_bits: target_bits,
        payload_len: target_bits.div_ceil(8),
        frame_len: 9 + target_bits.div_ceil(8),
        samples_per_channel: AVS3_FEATURE_DIMENSIONS as u32,
    };
    let config = McBitstreamConfig::for_header(&header).unwrap();

    let mut side_writer = BitWriter::new();
    write_mc_mode(&mut side_writer);
    let side_bits = side_writer.bit_len();
    let side_bytes = side_writer.into_bytes();
    let mut side_reader = BitReader::with_bit_len(&side_bytes, side_bits).unwrap();
    let side = McSideInfo::parse(&mut side_reader, config).unwrap();

    let mut writer = BitWriter::new();
    for _ in 0..7 {
        write_mc_core_prefix(&mut writer);
    }
    write_mc_mode(&mut writer);
    let reserved_bits = 7 * 19;
    let available_bits = target_bits - writer.bit_len() - reserved_bits;
    let allocation = mc_bytes_allocation(available_bits, side, config).unwrap();
    for &entropy_bytes in allocation.channel_bytes() {
        write_qc(&mut writer, entropy_bytes, 1);
    }
    pad_to(&mut writer, target_bits);
    (writer.into_bytes(), allocation.channel_bytes().to_vec())
}

fn pcm_fingerprint(values: &[i16]) -> u64 {
    values
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            (hash ^ u64::from(*value as u16)).wrapping_mul(0x100_0000_01b3)
        })
}

#[test]
fn object_only_mix_dispatches_to_mono_core() {
    const AUDIO_BITS: usize = 616;
    let payload = with_empty_metadata(&mono_audio(AUDIO_BITS), AUDIO_BITS);
    let frame = frame_objects_only(&payload, 1, 1, 4);
    let encoded = parse_one_frame(&frame);
    assert_eq!(encoded.header().profile, CodecProfile::Mixed);
    assert_eq!(
        encoded.header().sound_bed_type,
        Some(SoundBedType::ObjectsOnly)
    );
    assert_eq!(encoded.header().object_bitrate, Some(64_000));
    assert_eq!(encoded.header().payload_bits, AUDIO_BITS + 2);

    let mut decoder = PendingDecoder::new(MixDecoderBackend::new_builtin().unwrap())
        .configure(encoded.header())
        .unwrap();
    let audio = decoder.decode(&encoded).unwrap();
    assert_eq!(decoder.backend().core_kind(), Some(MixCoreKind::Mono));
    assert_eq!(audio.samples().len(), AVS3_FEATURE_DIMENSIONS);
    assert_eq!(
        decoder.backend().last_metadata().unwrap().audio_bits(),
        AUDIO_BITS
    );
    assert_eq!(
        decoder
            .backend()
            .mono_backend()
            .unwrap()
            .last_diagnostics()
            .unwrap()
            .entropy_bytes(),
        51
    );
    assert_eq!(
        pcm_fingerprint(audio.samples()),
        expected_rustfft_fingerprint(0xba61_f453_9d48_5862, 0x7f23_d010_f66f_1695)
    );
}

#[test]
fn two_object_mix_uses_low_bitrate_mcr() {
    const AUDIO_BITS: usize = 677;
    let payload = with_empty_metadata(&mcr_audio(AUDIO_BITS), AUDIO_BITS);
    let frame = frame_objects_only(&payload, 3, 2, 0);
    let encoded = parse_one_frame(&frame);
    assert_eq!(encoded.header().channels, 2);
    assert_eq!(encoded.header().object_bitrate, Some(16_000));
    assert_eq!(encoded.header().bitrate, 32_000);
    assert_eq!(encoded.header().payload_bits, AUDIO_BITS + 2);

    let mut decoder = PendingDecoder::new(MixDecoderBackend::new_builtin().unwrap())
        .configure(encoded.header())
        .unwrap();
    let audio = decoder.decode(&encoded).unwrap();
    assert_eq!(decoder.backend().core_kind(), Some(MixCoreKind::Stereo));
    let stereo = decoder.backend().stereo_backend().unwrap();
    assert!(stereo.last_diagnostics().is_none());
    assert_eq!(stereo.last_mcr_diagnostics().unwrap().entropy_bytes(), 47);
    assert_eq!(
        pcm_fingerprint(audio.samples()),
        expected_rustfft_fingerprint(0x291b_df9d_9077_9ad0, 0xdf7d_c254_a177_6513)
    );
}

#[test]
fn channel_bed_mix_keeps_lfe_before_objects_in_coupling_order() {
    const AUDIO_BITS: usize = 9_474;
    let (audio, expected_allocation) = bed_object_audio(AUDIO_BITS);
    let payload = with_muted_dynamic_metadata(&audio, AUDIO_BITS);
    let frame = frame_bed_and_object(&payload);
    let encoded = parse_one_frame(&frame);
    assert_eq!(encoded.header().channels, 7);
    assert_eq!(encoded.header().bed_channels, 6);
    assert_eq!(encoded.header().objects, 1);
    assert_eq!(encoded.header().bed_bitrate, Some(384_000));
    assert_eq!(encoded.header().object_bitrate, Some(64_000));
    assert_eq!(encoded.header().payload_bits, AUDIO_BITS + 11);

    let config = McBitstreamConfig::for_header(encoded.header()).unwrap();
    assert_eq!(config.bed_channels(), 6);
    assert_eq!(config.ild_channels(), 5);
    assert_eq!(config.lfe_bytes(), 20);

    let mut decoder = PendingDecoder::new(MixDecoderBackend::new_builtin().unwrap())
        .configure(encoded.header())
        .unwrap();
    let audio = decoder.decode(&encoded).unwrap();
    assert_eq!(
        decoder.backend().core_kind(),
        Some(MixCoreKind::Multichannel)
    );
    let metadata = decoder.backend().last_metadata().unwrap();
    assert_eq!(metadata.consumed_bits(), 11);
    assert_eq!(metadata.dynamic_metadata().unwrap().objects(), 1);
    let metadata_values = decoder.backend().last_metadata_values().unwrap();
    assert!(!metadata_values.has_static_metadata());
    let dynamic = metadata_values.dynamic_metadata().unwrap();
    assert_eq!(dynamic.level, 0);
    assert_eq!(dynamic.objects.len(), 1);
    assert!(dynamic.objects[0].muted);
    assert_eq!(dynamic.objects[0].transport_channel_reference, 0);
    assert_eq!(dynamic.objects[0].level1, None);
    let multichannel = decoder.backend().multichannel_backend().unwrap();
    let diagnostics = multichannel.last_diagnostics().unwrap();
    assert_eq!(diagnostics.entropy_bytes(), expected_allocation);
    assert!(
        multichannel.core().last_shaped_spectra()[3][MC_LFE_RESERVED_LINES..]
            .iter()
            .all(|&value| value == 0.0)
    );
    assert_eq!(pcm_fingerprint(audio.samples()), 0xf9ff_0e5e_e744_7013);
}
