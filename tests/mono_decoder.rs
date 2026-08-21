use avs3a::backend::MonoDecoderBackend;
use avs3a::bitstream::{BitReader, BitWriter};
use avs3a::decode::PendingDecoder;
use avs3a::header::ChannelConfig;
use avs3a::side_info::TransformType;
use avs3a::{DecodeError, FrameStream, StreamEvent, crc16};

const SAMPLE_RATE_INDEX_96_KHZ: u64 = 1;
const MONO_BITRATE_INDEX_64_KBPS: u64 = 4;

fn main_reference_payload() -> Vec<u8> {
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
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(1, 1).unwrap();

    writer.write_bits(1, 1).unwrap();
    for group in [0, 0, 0, 1, 1, 1, 1, 1] {
        writer.write_bits(group, 1).unwrap();
    }

    writer.write_bits(1, 1).unwrap();
    writer.write_bits(37, 7).unwrap();
    writer.write_bits(3, 3).unwrap();
    writer.write_bits(7, 3).unwrap();
    let context: [u8; 6] = [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7];
    let base: [u8; 26] = [
        0x7f, 0xfd, 0x51, 0xf6, 0xf2, 0x24, 0x34, 0xad, 0x04, 0xde, 0x75, 0xcd, 0x9d, 0x0c, 0x76,
        0xeb, 0xb3, 0x76, 0xaf, 0x47, 0xda, 0x43, 0x33, 0xf0, 0xd4, 0xeb,
    ];
    writer.write_bits(context.len() as u64, 8).unwrap();
    for byte in context.into_iter().chain(base) {
        writer.write_bits(u64::from(byte), 8).unwrap();
    }

    // A real header determines the payload budget. At 64 kbps/96 kHz it is
    // 626 bits, so retain the C vector and add entropy tail bytes that the
    // range decoder does not need after reconstructing all symbols.
    for _ in 0..20 {
        writer.write_bits(0, 8).unwrap();
    }
    assert_eq!(writer.bit_len(), 624);
    let mut payload = writer.into_bytes();
    payload.resize(79, 0);
    payload
}

fn framed_main_reference() -> Vec<u8> {
    let payload = with_empty_metadata(&main_reference_payload(), 624);
    let crc = crc16(&payload);
    let mut writer = BitWriter::new();
    for (value, width) in [
        (0xfff, 12),
        (2, 4),
        (0, 1),
        (0, 3),
        (0, 3),
        (SAMPLE_RATE_INDEX_96_KHZ, 4),
        (u64::from(crc >> 8), 8),
        (u64::from(ChannelConfig::Mono.index()), 7),
        (1, 2),
        (MONO_BITRATE_INDEX_64_KBPS, 4),
        (u64::from(crc & 0xff), 8),
    ] {
        writer.write_bits(value, width).unwrap();
    }
    let mut frame = writer.into_bytes();
    frame.extend_from_slice(&payload);
    frame
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

fn parse_one_frame(bytes: &[u8]) -> avs3a::EncodedFrame {
    let mut stream = FrameStream::new();
    let mut events = Vec::new();
    for chunk in bytes.chunks(13) {
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

#[test]
fn public_framing_crc_and_mono_decoder_match_the_c_pipeline_vector() {
    let encoded = parse_one_frame(&framed_main_reference());
    assert!(encoded.crc_is_valid());
    assert_eq!(encoded.header().sample_rate, 96_000);
    assert_eq!(encoded.header().bitrate, 64_000);
    assert_eq!(encoded.header().payload_bits, 626);

    let backend = MonoDecoderBackend::new_builtin().unwrap();
    let mut decoder = PendingDecoder::new(backend)
        .configure(encoded.header())
        .unwrap();
    let audio = decoder.decode(&encoded).unwrap();

    assert_eq!(decoder.frame_index(), 1);
    assert_eq!(audio.samples().len(), 1_024);
    let metadata = decoder.backend().last_metadata().unwrap();
    assert_eq!(metadata.consumed_bits(), 2);
    assert_eq!(metadata.audio_bits(), 624);
    let diagnostics = decoder.backend().last_diagnostics().unwrap();
    assert_eq!(diagnostics.core().transform_type(), TransformType::Short);
    assert_eq!(diagnostics.entropy_bytes(), 52);
    assert_eq!(diagnostics.consumed_bits(), 624);
    assert_eq!(diagnostics.padding_bits(), 0);

    let positions = [0, 447, 448, 449, 575, 576, 700, 900, 1023];
    assert_eq!(
        positions.map(|position| audio.samples()[position]),
        [0, 0, 50, -194, -8_089, 3_802, i16::MAX, i16::MIN, i16::MIN]
    );

    let clipped_before = decoder.total_clipped_samples();
    let mut damaged_bytes = framed_main_reference();
    *damaged_bytes.last_mut().unwrap() ^= 1;
    let damaged = parse_one_frame(&damaged_bytes);
    assert!(!damaged.crc_is_valid());
    assert!(matches!(
        decoder.decode(&damaged),
        Err(DecodeError::CrcMismatch { .. })
    ));
    assert_eq!(decoder.frame_index(), 1);
    assert_eq!(decoder.total_clipped_samples(), clipped_before);
}
