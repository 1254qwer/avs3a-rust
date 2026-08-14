use avs3a::{
    crc16, BitReader, BitWriter, ChannelConfig, CodecProfile, FrameStream, HoaDecoderBackend,
    PendingDecoder, StreamEvent, AVS3_FEATURE_DIMENSIONS,
};

const FOA_AUDIO_BITS: usize = 4_038;
const FOA_FRAME_PAYLOAD_BITS: usize = FOA_AUDIO_BITS + 2;
const FOA_CHANNELS: usize = 4;
const HOA3_AUDIO_BITS: usize = 6_768;
const HOA3_FRAME_PAYLOAD_BITS: usize = HOA3_AUDIO_BITS + 2;
const HOA3_TRANSPORT_CHANNELS: usize = 9;
const HOA3_OUTPUT_CHANNELS: usize = 16;

fn write_qc(writer: &mut BitWriter, entropy_bytes: usize) {
    let context: [u8; 6] = [0x84, 0xa0, 0xd8, 0x95, 0xb9, 0xa7];
    let base: [u8; 26] = [
        0x7f, 0xfd, 0x51, 0xf6, 0xf2, 0x24, 0x34, 0xad, 0x04, 0xde, 0x75, 0xcd, 0x9d, 0x0c, 0x76,
        0xeb, 0xb3, 0x76, 0xaf, 0x47, 0xda, 0x43, 0x33, 0xf0, 0xd4, 0xeb,
    ];
    writer.write_bits(0, 1).unwrap();
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

fn foa_audio_payload() -> Vec<u8> {
    let mut writer = BitWriter::new();
    for _ in 0..FOA_CHANNELS {
        writer.write_bits(0, 2).unwrap();
        for width in [8, 8, 7, 7, 6, 5, 5] {
            writer.write_bits(0, width).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
        for _ in 0..4 {
            writer.write_bits(0, 7).unwrap();
        }
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 1).unwrap();
    }

    writer.write_bits(3, 4).unwrap();
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(0, 4).unwrap();
    writer.write_bits(15, 4).unwrap();
    for _ in 0..FOA_CHANNELS {
        writer.write_bits(4, 4).unwrap();
    }
    assert_eq!(writer.bit_len(), 349);

    for entropy_bytes in [115, 112, 112, 112] {
        write_qc(&mut writer, entropy_bytes);
    }
    assert_eq!(writer.bit_len(), 4_033);
    writer.write_bits(0, 5).unwrap();
    assert_eq!(writer.bit_len(), FOA_AUDIO_BITS);
    writer.into_bytes()
}

fn framed_foa_reference() -> Vec<u8> {
    let audio = foa_audio_payload();
    let mut audio_reader = BitReader::with_bit_len(&audio, FOA_AUDIO_BITS).unwrap();
    let mut payload_writer = BitWriter::new();
    payload_writer.write_bits(0, 2).unwrap();
    while audio_reader.remaining() != 0 {
        let width = audio_reader.remaining().min(64);
        payload_writer
            .write_bits(audio_reader.read_bits(width).unwrap(), width)
            .unwrap();
    }
    assert_eq!(payload_writer.bit_len(), FOA_FRAME_PAYLOAD_BITS);
    let payload = payload_writer.into_bytes();
    let crc = crc16(&payload);

    let mut header = BitWriter::new();
    for (value, width) in [
        (0xfff, 12),
        (2, 4),
        (0, 1),
        (0, 3),
        (2, 3),
        (2, 4),
        (u64::from(crc >> 8), 8),
        (0, 4),
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

fn write_hbr_core_prefix(writer: &mut BitWriter, bwe_enabled: bool) {
    writer.write_bits(0, 2).unwrap();
    for width in [8, 8, 7, 7, 6, 5, 5] {
        writer.write_bits(0, width).unwrap();
    }
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(0, 1).unwrap();
    if bwe_enabled {
        for _ in 0..6 {
            writer.write_bits(0, 7).unwrap();
        }
        for _ in 0..3 {
            writer.write_bits(0, 1).unwrap();
        }
    }
}

fn hoa3_audio_payload() -> Vec<u8> {
    let mut writer = BitWriter::new();
    for channel in 0..HOA3_TRANSPORT_CHANNELS {
        write_hbr_core_prefix(&mut writer, channel >= 2);
    }
    assert_eq!(writer.bit_len(), 765);

    writer.write_bits(5, 4).unwrap();
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(2, 4).unwrap();
    writer.write_bits(1, 12).unwrap();
    writer.write_bits(512, 12).unwrap();

    writer.write_bits(0, 4).unwrap();
    writer.write_bits(6, 4).unwrap();
    writer.write_bits(8, 4).unwrap();
    writer.write_bits(8, 4).unwrap();

    writer.write_bits(0, 4).unwrap();
    writer.write_bits(10, 4).unwrap();
    for _ in 0..7 {
        writer.write_bits(1, 4).unwrap();
    }
    assert_eq!(writer.bit_len(), 850);

    for entropy_bytes in [135, 134, 65, 64, 64, 64, 64, 64, 64] {
        write_qc(&mut writer, entropy_bytes);
    }
    assert_eq!(writer.bit_len(), 6_765);
    writer.write_bits(0, 3).unwrap();
    assert_eq!(writer.bit_len(), HOA3_AUDIO_BITS);
    writer.into_bytes()
}

fn framed_hoa3_reference() -> Vec<u8> {
    let audio = hoa3_audio_payload();
    let mut audio_reader = BitReader::with_bit_len(&audio, HOA3_AUDIO_BITS).unwrap();
    let mut payload_writer = BitWriter::new();
    payload_writer.write_bits(0, 2).unwrap();
    while audio_reader.remaining() != 0 {
        let width = audio_reader.remaining().min(64);
        payload_writer
            .write_bits(audio_reader.read_bits(width).unwrap(), width)
            .unwrap();
    }
    assert_eq!(payload_writer.bit_len(), HOA3_FRAME_PAYLOAD_BITS);
    let payload = payload_writer.into_bytes();
    let crc = crc16(&payload);

    let mut header = BitWriter::new();
    for (value, width) in [
        (0xfff, 12),
        (2, 4),
        (0, 1),
        (0, 3),
        (2, 3),
        (2, 4),
        (u64::from(crc >> 8), 8),
        (2, 4),
        (1, 2),
        (1, 4),
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
fn public_framing_crc_and_hoa_backend_decode_complete_foa_frame() {
    let encoded = parse_one_frame(&framed_foa_reference());
    assert!(encoded.crc_is_valid());
    assert_eq!(encoded.header().profile, CodecProfile::Hoa);
    assert_eq!(encoded.header().channel_config, Some(ChannelConfig::Hoa1));
    assert_eq!(encoded.header().hoa_order, Some(1));
    assert_eq!(encoded.header().channels, FOA_CHANNELS as u8);
    assert_eq!(encoded.header().bitrate, 192_000);
    assert_eq!(encoded.header().payload_bits, FOA_FRAME_PAYLOAD_BITS);

    let backend = HoaDecoderBackend::new_builtin().unwrap();
    let mut decoder = PendingDecoder::new(backend)
        .configure(encoded.header())
        .unwrap();
    let mut samples = vec![0_i16; FOA_CHANNELS * AVS3_FEATURE_DIMENSIONS];
    decoder.decode_into(&encoded, &mut samples).unwrap();

    assert_eq!(decoder.frame_index(), 1);
    let metadata = decoder.backend().last_metadata().unwrap();
    assert_eq!(metadata.consumed_bits(), 2);
    assert_eq!(metadata.audio_bits(), FOA_AUDIO_BITS);
    let diagnostics = decoder.backend().last_diagnostics().unwrap();
    assert_eq!(diagnostics.transport_channels(), FOA_CHANNELS);
    assert_eq!(diagnostics.output_channels(), FOA_CHANNELS);
    assert_eq!(diagnostics.entropy_bytes(), &[115, 112, 112, 112]);
    assert_eq!(diagnostics.consumed_bits(), 4_033);
    assert_eq!(diagnostics.padding_bits(), 5);
    assert_eq!(pcm_fingerprint(&samples), 0x4244_691c_01a8_dc66);
    assert!(samples.iter().any(|&sample| sample != 0));
}

#[test]
fn hoa3_spatial_pipeline_uses_nine_transports_and_two_frame_basis_delay() {
    let frame_bytes = framed_hoa3_reference();
    let mut stream_bytes = Vec::with_capacity(frame_bytes.len() * 3);
    for _ in 0..3 {
        stream_bytes.extend_from_slice(&frame_bytes);
    }

    let mut stream = FrameStream::new();
    let mut events = Vec::new();
    for chunk in stream_bytes.chunks(97) {
        events.extend(stream.push(chunk).unwrap());
    }
    events.extend(stream.finish().unwrap());
    let frames = events
        .into_iter()
        .map(|event| match event {
            StreamEvent::Frame(frame) => frame,
            StreamEvent::Skipped { bytes } => {
                panic!("unexpected resynchronization over {bytes} bytes")
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(frames.len(), 3);
    let header = frames[0].header();
    assert_eq!(header.profile, CodecProfile::Hoa);
    assert_eq!(header.channel_config, Some(ChannelConfig::Hoa3));
    assert_eq!(header.hoa_order, Some(3));
    assert_eq!(header.channels, HOA3_OUTPUT_CHANNELS as u8);
    assert_eq!(header.bitrate, 320_000);
    assert_eq!(header.payload_bits, HOA3_FRAME_PAYLOAD_BITS);

    let backend = HoaDecoderBackend::new_builtin().unwrap();
    let mut decoder = PendingDecoder::new(backend).configure(header).unwrap();
    let mut samples = vec![0_i16; HOA3_OUTPUT_CHANNELS * AVS3_FEATURE_DIMENSIONS];
    let mut fingerprints = [0_u64; 3];
    for (frame_index, frame) in frames.iter().enumerate() {
        decoder.decode_into(frame, &mut samples).unwrap();
        fingerprints[frame_index] = pcm_fingerprint(&samples);
        let diagnostics = decoder.backend().last_diagnostics().unwrap();
        assert_eq!(diagnostics.transport_channels(), HOA3_TRANSPORT_CHANNELS);
        assert_eq!(diagnostics.output_channels(), HOA3_OUTPUT_CHANNELS);
        assert_eq!(
            diagnostics.entropy_bytes(),
            &[135, 134, 65, 64, 64, 64, 64, 64, 64]
        );
        assert_eq!(diagnostics.consumed_bits(), 6_765);
        assert_eq!(diagnostics.padding_bits(), 3);
        assert!(diagnostics.hoa().spatial_analysis());
        assert_eq!(diagnostics.hoa().basis_indices(), &[1, 512]);

        let delayed = decoder
            .backend()
            .core()
            .post_synthesis()
            .delayed_basis_indices();
        if frame_index == 0 {
            assert_eq!(delayed[0][..2], [0, 0]);
        } else {
            assert_eq!(delayed[0][..2], [1, 512]);
        }
        assert_eq!(delayed[1][..2], [1, 512]);
    }
    assert_eq!(decoder.frame_index(), 3);
    assert_eq!(
        fingerprints,
        [
            0xe3ee_95db_af37_11f4,
            0x703d_2b13_3c8a_88f4,
            0x22ae_e220_dd46_d44e,
        ]
    );
}
