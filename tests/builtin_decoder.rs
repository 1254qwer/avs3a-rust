use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use avs3a::decode::warmup_frames_for;
use avs3a::{BuiltinDecoder, EncodedFrame, FrameStream, StreamEvent};

fn first_frames(count: usize) -> Vec<EncodedFrame> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test.av3a");
    let mut input = BufReader::new(File::open(path).unwrap());
    let mut parser = FrameStream::new();
    let mut buffer = [0_u8; 8 * 1024];
    let mut frames = Vec::with_capacity(count);

    while frames.len() < count {
        let read = input.read(&mut buffer).unwrap();
        assert_ne!(read, 0, "fixture ended before {count} frames");
        for event in parser.push(&buffer[..read]).unwrap() {
            match event {
                StreamEvent::Frame(frame) => frames.push(frame),
                StreamEvent::Skipped { bytes } => {
                    panic!("fixture parser skipped {bytes} bytes")
                }
            }
            if frames.len() == count {
                break;
            }
        }
    }
    frames
}

#[test]
fn builtin_decoder_preserves_state_and_resets_to_a_clean_stream() {
    std::thread::Builder::new()
        .name("builtin-decoder-reset".to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(run_reset_test)
        .unwrap()
        .join()
        .unwrap();
}

/// `warmup_frames_for` exists so a caller can size a seek without allocating a
/// decoder. It reads the answer off the header while `BuiltinDecoder` reads it
/// off the constructed variant, so pin them together — two matches on the same
/// header fields are exactly the kind of thing that drifts apart.
#[test]
fn warmup_frames_for_agrees_with_a_constructed_decoder() {
    let frames = first_frames(1);
    let header = *frames[0].header();
    // Reading the answer off the header needs no stack to speak of; building a
    // decoder to check it against does, hence the thread — which is the cost
    // `warmup_frames_for` exists to let callers avoid.
    assert_eq!(warmup_frames_for(&header).unwrap(), 1);
    let constructed = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || BuiltinDecoder::configure(&header).unwrap().warmup_frames())
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(warmup_frames_for(frames[0].header()).unwrap(), constructed);
}

fn run_reset_test() {
    let frames = first_frames(2);
    let sample_count = usize::from(frames[0].header().channels)
        * usize::try_from(frames[0].header().samples_per_channel).unwrap();
    let mut first_output = vec![0_i16; sample_count];
    let mut output = vec![0_i16; sample_count];
    let mut decoder = BuiltinDecoder::configure(frames[0].header()).unwrap();

    decoder.decode_into(&frames[0], &mut first_output).unwrap();
    decoder.decode_into(&frames[1], &mut output).unwrap();
    assert_eq!(decoder.frame_index(), 2);

    decoder.reset().unwrap();
    assert_eq!(decoder.frame_index(), 0);
    decoder.decode_into(&frames[0], &mut output).unwrap();

    assert_eq!(decoder.frame_index(), 1);
    assert_eq!(output, first_output);

    decoder.reset().unwrap();
    decoder.decode_into(&frames[1], &mut output).unwrap();
    assert_eq!(decoder.frame_index(), 1);
}
