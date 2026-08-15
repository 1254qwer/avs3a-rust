use avs3a::{
    BitReader, BitWriter, ChannelConfig, FrameStream, MetadataPayloadParser, PendingDecoder,
    StereoDecoderBackend, StreamEvent, crc16,
};

mod support;

use support::expected_rustfft_fingerprint;

fn write_core_prefix(writer: &mut BitWriter, lsf: [u64; 5], envelopes: [u64; 6]) {
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

fn write_qc(writer: &mut BitWriter, entropy_bytes: usize) {
    let context: [u8; 6] = [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7];
    let base: [u8; 26] = [
        0x7f, 0xfd, 0x51, 0xf6, 0xf2, 0x24, 0x34, 0xad, 0x04, 0xde, 0x75, 0xcd, 0x9d, 0x0c, 0x76,
        0xeb, 0xb3, 0x76, 0xaf, 0x47, 0xda, 0x43, 0x33, 0xf0, 0xd4, 0xeb,
    ];
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(37, 7).unwrap();
    writer.write_bits(3, 3).unwrap();
    writer.write_bits(7, 3).unwrap();
    writer.write_bits(context.len() as u64, 8).unwrap();
    for byte in context.into_iter().chain(base) {
        writer.write_bits(u64::from(byte), 8).unwrap();
    }
    for _ in context.len() + base.len()..entropy_bytes {
        writer.write_bits(0, 8).unwrap();
    }
}

fn reference_payload() -> Vec<u8> {
    let mut writer = BitWriter::new();
    write_core_prefix(&mut writer, [3, 5, 7, 9, 11], [1, 2, 3, 4, 5, 6]);
    write_core_prefix(&mut writer, [17, 19, 21, 23, 25], [7, 8, 9, 10, 11, 12]);
    for _ in 0..2 {
        writer.write_bits(1, 1).unwrap();
        for indicator in [0, 0, 0, 1, 1, 1, 1, 1] {
            writer.write_bits(indicator, 1).unwrap();
        }
    }
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(5, 4).unwrap();
    writer.write_bits(4, 3).unwrap();
    write_qc(&mut writer, 64);
    write_qc(&mut writer, 69);
    assert_eq!(writer.bit_len(), 1_304);
    let mut payload = writer.into_bytes();
    payload.resize(164, 0);
    payload
}

fn mcr_reference_payload() -> Vec<u8> {
    let mut writer = BitWriter::new();
    write_core_prefix(&mut writer, [3, 5, 7, 9, 11], [1, 2, 3, 4, 5, 6]);
    write_core_prefix(&mut writer, [17, 19, 21, 23, 25], [7, 8, 9, 10, 11, 12]);
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
    assert_eq!(writer.bit_len(), 275);
    write_qc(&mut writer, 40);
    assert_eq!(writer.bit_len(), 617);
    let mut payload = writer.into_bytes();
    payload.resize(78, 0);
    payload
}

fn with_empty_metadata(audio_payload: &[u8], audio_bits: usize) -> Vec<u8> {
    let mut reader = BitReader::with_bit_len(audio_payload, audio_bits).unwrap();
    let mut writer = BitWriter::new();
    writer.write_bits(0, 2).unwrap();
    while reader.remaining() != 0 {
        let width = reader.remaining().min(64);
        writer
            .write_bits(reader.read_bits(width).unwrap(), width)
            .unwrap();
    }
    assert_eq!(writer.bit_len(), audio_bits + 2);
    writer.into_bytes()
}

fn with_minimal_static_metadata() -> Vec<u8> {
    let mut audio = BitWriter::new();
    write_core_prefix(&mut audio, [3, 5, 7, 9, 11], [1, 2, 3, 4, 5, 6]);
    write_core_prefix(&mut audio, [17, 19, 21, 23, 25], [7, 8, 9, 10, 11, 12]);
    for _ in 0..2 {
        audio.write_bits(1, 1).unwrap();
        for indicator in [0, 0, 0, 1, 1, 1, 1, 1] {
            audio.write_bits(indicator, 1).unwrap();
        }
    }
    audio.write_bits(1, 1).unwrap();
    audio.write_bits(5, 4).unwrap();
    audio.write_bits(4, 3).unwrap();
    write_qc(&mut audio, 60);
    write_qc(&mut audio, 62);
    assert_eq!(audio.bit_len(), 1_216);
    audio.write_bits(0, 3).unwrap();
    let audio_bits = audio.bit_len();
    let audio = audio.into_bytes();

    let mut writer = BitWriter::new();
    writer.write_bits(1, 1).unwrap();
    write_minimal_static_metadata(&mut writer);
    writer.write_bits(0, 1).unwrap();
    let mut reader = BitReader::with_bit_len(&audio, audio_bits).unwrap();
    while reader.remaining() != 0 {
        let width = reader.remaining().min(64);
        writer
            .write_bits(reader.read_bits(width).unwrap(), width)
            .unwrap();
    }
    assert_eq!(writer.bit_len(), 1_309);
    writer.into_bytes()
}

fn write_minimal_static_metadata(writer: &mut BitWriter) {
    for (value, width) in [
        (0, 1), // no VR extension
        (0, 3), // Basic L1
        (0, 4), // programme flags
        (0, 2), // one programme content reference
        (0, 2),
        (0, 2), // one content
        (0, 2), // content index
        (0, 4), // content flags
        (0, 3), // one object reference
        (0, 3),
        (0, 3), // one object
        (0, 3), // object index
        (0, 8), // object flags
        (0, 3), // one pack reference
        (0, 3),
        (0, 3), // one pack
        (0, 3), // pack index
        (0, 1), // no importance
        (0, 1), // no channel reuse
        (3, 3), // objects type label
        (0, 5), // absolute distance
        (0, 5), // pack start index
        (0, 5), // one channel reference
        (7, 5), // channel 7
        (0, 5), // one channel
        (7, 5), // channel format 7
        (0, 1), // no channel gain
    ] {
        writer.write_bits(value, width).unwrap();
    }
}

fn framed_stereo(payload: &[u8], bitrate_index: u64) -> Vec<u8> {
    let crc = crc16(payload);
    let mut writer = BitWriter::new();
    for (value, width) in [
        (0xfff, 12),
        (2, 4),
        (0, 1),
        (0, 3),
        (0, 3),
        (2, 4),
        (u64::from(crc >> 8), 8),
        (u64::from(ChannelConfig::Stereo.index()), 7),
        (1, 2),
        (bitrate_index, 4),
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
    let events = stream.push(bytes).unwrap();
    stream.finish().unwrap();
    let mut frames = events.into_iter().map(|event| match event {
        StreamEvent::Frame(frame) => frame,
        StreamEvent::Skipped { bytes } => panic!("unexpected resynchronization over {bytes} bytes"),
    });
    let frame = frames.next().expect("one complete frame");
    assert!(frames.next().is_none());
    frame
}

fn pcm_fingerprint(values: &[i16]) -> u64 {
    values
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            (hash ^ u64::from(*value as u16)).wrapping_mul(0x100_0000_01b3)
        })
}

#[test]
fn public_framing_crc_and_ordinary_stereo_backend_match_c_pcm16() {
    // Stereo bitrate table index 3 is 64 kbps. At 48 kHz its payload is 1309
    // valid bits stored in 164 bytes, exactly matching the constructed vector.
    let payload = with_empty_metadata(&reference_payload(), 1_307);
    let encoded = parse_one_frame(&framed_stereo(&payload, 3));
    assert!(encoded.crc_is_valid());
    assert_eq!(encoded.header().channel_config, Some(ChannelConfig::Stereo));
    assert_eq!(encoded.header().bitrate, 64_000);
    assert_eq!(encoded.header().payload_bits, 1_309);

    let backend = StereoDecoderBackend::new_builtin().unwrap();
    let mut decoder = PendingDecoder::new(backend)
        .configure(encoded.header())
        .unwrap();
    let audio = decoder.decode(&encoded).unwrap();
    assert_eq!(audio.samples().len(), 2_048);
    assert_eq!(decoder.frame_index(), 1);
    assert_eq!(decoder.backend().last_clipped_samples(), 0);
    assert_eq!(pcm_fingerprint(audio.samples()), 0xa036_2bb2_f0ab_465a);
    let metadata = decoder.backend().last_metadata().unwrap();
    assert_eq!(metadata.consumed_bits(), 2);
    assert_eq!(metadata.audio_bits(), 1_307);
    let diagnostics = decoder.backend().last_diagnostics().unwrap();
    assert_eq!(diagnostics.entropy_bytes(), [64, 69]);
    assert_eq!(diagnostics.padding_bits(), 3);
}

#[test]
fn public_framing_crc_and_mcr_stereo_backend_decode_payload() {
    // Stereo bitrate table index 1 is 32 kbps. At 48 kHz it carries 626 valid
    // payload bits in 79 bytes and selects MCR instead of ordinary MS/ILD.
    let payload = with_empty_metadata(&mcr_reference_payload(), 624);
    let encoded = parse_one_frame(&framed_stereo(&payload, 1));
    assert!(encoded.crc_is_valid());
    assert_eq!(encoded.header().channel_config, Some(ChannelConfig::Stereo));
    assert_eq!(encoded.header().bitrate, 32_000);
    assert_eq!(encoded.header().payload_bits, 626);

    let backend = StereoDecoderBackend::new_builtin().unwrap();
    let mut decoder = PendingDecoder::new(backend)
        .configure(encoded.header())
        .unwrap();
    let audio = decoder.decode(&encoded).unwrap();
    assert_eq!(audio.samples().len(), 2_048);
    assert_eq!(decoder.frame_index(), 1);
    assert_eq!(decoder.backend().last_clipped_samples(), 0);
    assert!(decoder.backend().last_diagnostics().is_none());
    let metadata = decoder.backend().last_metadata().unwrap();
    assert_eq!(metadata.consumed_bits(), 2);
    assert_eq!(metadata.audio_bits(), 624);
    let diagnostics = decoder.backend().last_mcr_diagnostics().unwrap();
    assert_eq!(diagnostics.entropy_bytes(), 40);
    assert_eq!(diagnostics.padding_bits(), 7);
    assert_eq!(
        pcm_fingerprint(audio.samples()),
        expected_rustfft_fingerprint(0x291b_df9d_9077_9ad0, 0xdf7d_c254_a177_6513)
    );
}

#[test]
fn public_backend_retains_complete_static_metadata_values() {
    let encoded = parse_one_frame(&framed_stereo(&with_minimal_static_metadata(), 3));
    let backend = StereoDecoderBackend::new_builtin().unwrap();
    let mut decoder = PendingDecoder::new(backend)
        .configure(encoded.header())
        .unwrap();
    let audio = decoder.decode(&encoded).unwrap();

    let summary = decoder.backend().last_metadata().unwrap();
    assert_eq!(summary.consumed_bits(), 90);
    assert_eq!(summary.audio_bits(), 1_219);
    let metadata = decoder.backend().last_metadata_values().unwrap();
    let static_metadata = metadata.static_metadata().unwrap();
    assert_eq!(static_metadata.basic_level, 0);
    assert_eq!(
        static_metadata
            .basic
            .programme
            .content_references
            .as_slice(),
        &[0]
    );
    assert_eq!(static_metadata.basic.packs[0].channels[0].channel_index, 7);
    assert_eq!(static_metadata.basic.channels[0].index, 7);
    assert_eq!(
        decoder
            .backend()
            .last_diagnostics()
            .unwrap()
            .entropy_bytes(),
        [60, 62]
    );
    assert_eq!(audio.samples().len(), 2_048);
    assert_eq!(pcm_fingerprint(audio.samples()), 0xa036_2bb2_f0ab_465a);
}

#[test]
fn public_24_kbps_stereo_backend_configures_mcr_mode() {
    // At 48 kHz, 24 kbps leaves 456 payload bits, stored in 57 bytes. The
    // backend can be configured from this framing even before a full vector
    // for the smaller MCR entropy budget is added.
    let encoded = parse_one_frame(&framed_stereo(&[0; 57], 0));
    assert!(encoded.crc_is_valid());
    assert_eq!(encoded.header().bitrate, 24_000);
    assert_eq!(encoded.header().payload_bits, 456);

    let backend = StereoDecoderBackend::new_builtin().unwrap();
    let decoder = PendingDecoder::new(backend)
        .configure(encoded.header())
        .unwrap();
    assert_eq!(decoder.frame_index(), 0);
}

#[test]
fn public_parser_consumes_populated_static_metadata_before_audio() {
    let mut writer = BitWriter::new();
    writer.write_bits(1, 1).unwrap();
    write_minimal_static_metadata(&mut writer);
    writer.write_bits(0, 1).unwrap();
    let metadata_bits = writer.bit_len();
    writer.write_bits(0b101_0011_0111, 11).unwrap();
    let payload_bits = writer.bit_len();
    let payload = writer.into_bytes();

    let mut parser = MetadataPayloadParser::new();
    let parsed = parser.parse(&payload, payload_bits).unwrap();
    let summary = parsed.summary();
    let static_metadata = summary.static_metadata().unwrap();
    assert_eq!(summary.consumed_bits(), metadata_bits);
    assert_eq!(summary.audio_bits(), 11);
    assert_eq!(static_metadata.basic_level(), 0);
    assert_eq!(static_metadata.contents(), 1);
    assert_eq!(static_metadata.objects(), 1);
    assert_eq!(static_metadata.packs(), 1);
    assert_eq!(static_metadata.channels(), 1);
    assert_eq!(parsed.audio_payload(), &[0b1010_0110, 0b1110_0000]);
}
