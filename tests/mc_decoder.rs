use avs3a::{
    AVS3_FEATURE_DIMENSIONS, BitReader, BitWriter, ChannelConfig, FrameStream,
    MC_LFE_RESERVED_LINES, McDecoderBackend, PendingDecoder, StreamEvent, crc16,
};

const AUDIO_BITS: usize = 8_134;
const FRAME_PAYLOAD_BITS: usize = AUDIO_BITS + 2;
const CHANNELS: usize = 6;

fn write_qc(writer: &mut BitWriter, entropy_bytes: usize) {
    let context: [u8; 6] = [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7];
    let base: [u8; 26] = [
        0x7f, 0xfd, 0x51, 0xf6, 0xf2, 0x24, 0x34, 0xad, 0x04, 0xde, 0x75, 0xcd, 0x9d, 0x0c, 0x76,
        0xeb, 0xb3, 0x76, 0xaf, 0x47, 0xda, 0x43, 0x33, 0xf0, 0xd4, 0xeb,
    ];
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(37, 7).unwrap();
    writer.write_bits(3, 3).unwrap();
    writer.write_bits(context.len() as u64, 8).unwrap();
    for byte in context.into_iter().chain(base).take(entropy_bytes) {
        writer.write_bits(u64::from(byte), 8).unwrap();
    }
    for _ in (context.len() + base.len()).min(entropy_bytes)..entropy_bytes {
        writer.write_bits(0, 8).unwrap();
    }
}

fn audio_payload() -> Vec<u8> {
    let mut writer = BitWriter::new();
    for _ in 0..CHANNELS {
        writer.write_bits(0, 2).unwrap();
        for width in [8, 8, 7, 7, 6, 5, 5] {
            writer.write_bits(0, width).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
    }

    writer.write_bits(0, 1).unwrap();
    writer.write_bits(0, 4).unwrap();
    for ratio in [13, 13, 13, 13, 12] {
        writer.write_bits(ratio, 6).unwrap();
    }
    assert_eq!(writer.bit_len(), 335);

    for entropy_bytes in [194, 190, 190, 20, 190, 176] {
        write_qc(&mut writer, entropy_bytes);
    }
    assert_eq!(writer.bit_len(), AUDIO_BITS - 5);
    writer.write_bits(0, 5).unwrap();
    assert_eq!(writer.bit_len(), AUDIO_BITS);
    writer.into_bytes()
}

fn framed_reference() -> Vec<u8> {
    let audio = audio_payload();
    let mut audio_reader = BitReader::with_bit_len(&audio, AUDIO_BITS).unwrap();
    let mut payload_writer = BitWriter::new();
    payload_writer.write_bits(0, 2).unwrap();
    while audio_reader.remaining() != 0 {
        let width = audio_reader.remaining().min(64);
        payload_writer
            .write_bits(audio_reader.read_bits(width).unwrap(), width)
            .unwrap();
    }
    assert_eq!(payload_writer.bit_len(), FRAME_PAYLOAD_BITS);
    let payload = payload_writer.into_bytes();
    let crc = crc16(&payload);

    let mut header = BitWriter::new();
    for (value, width) in [
        (0xfff, 12),
        (2, 4),
        (0, 1),
        (0, 3),
        (0, 3),
        (2, 4),
        (u64::from(crc >> 8), 8),
        (u64::from(ChannelConfig::Mc5_1.index()), 7),
        (1, 2),
        (3, 4),
        (u64::from(crc & 0xff), 8),
    ] {
        header.write_bits(value, width).unwrap();
    }
    let mut frame = header.into_bytes();
    frame.extend_from_slice(&payload);
    frame
}

fn parse_one_frame(bytes: &[u8]) -> avs3a::EncodedFrame {
    let mut stream = FrameStream::new();
    let mut events = Vec::new();
    for chunk in bytes.chunks(37) {
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

fn pcm_fingerprint(values: &[i16]) -> u64 {
    values
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            (hash ^ u64::from(*value as u16)).wrapping_mul(0x100_0000_01b3)
        })
}

#[test]
fn public_framing_crc_and_mc_backend_decode_complete_5_1_frame() {
    let encoded = parse_one_frame(&framed_reference());
    assert!(encoded.crc_is_valid());
    assert_eq!(encoded.header().channel_config, Some(ChannelConfig::Mc5_1));
    assert_eq!(encoded.header().channels, CHANNELS as u8);
    assert_eq!(encoded.header().bitrate, 384_000);
    assert_eq!(encoded.header().payload_bits, FRAME_PAYLOAD_BITS);

    let backend = McDecoderBackend::new_builtin().unwrap();
    let mut decoder = PendingDecoder::new(backend)
        .configure(encoded.header())
        .unwrap();
    let mut samples = vec![0_i16; CHANNELS * AVS3_FEATURE_DIMENSIONS];
    decoder.decode_into(&encoded, &mut samples).unwrap();

    assert_eq!(decoder.frame_index(), 1);
    let metadata = decoder.backend().last_metadata().unwrap();
    assert_eq!(metadata.consumed_bits(), 2);
    assert_eq!(metadata.audio_bits(), AUDIO_BITS);
    let diagnostics = decoder.backend().last_diagnostics().unwrap();
    assert_eq!(diagnostics.channels(), CHANNELS);
    assert_eq!(diagnostics.entropy_bytes(), &[194, 190, 190, 20, 190, 176]);
    assert_eq!(diagnostics.consumed_bits(), AUDIO_BITS - 5);
    assert_eq!(diagnostics.padding_bits(), 5);
    assert_eq!(pcm_fingerprint(&samples), 0x0e61_2554_b394_cba8);
    assert!(
        decoder.backend().core().last_shaped_spectra()[3][MC_LFE_RESERVED_LINES..]
            .iter()
            .all(|&value| value == 0.0)
    );
}
