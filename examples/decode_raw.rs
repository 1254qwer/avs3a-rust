//! Incrementally decode a raw AV3A stream to PCM16 WAV.
//! Run: cargo run --release --example decode_raw -- input.av3a output.wav
//! Use distinct input and output paths; the output is overwritten.

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;

use avs3a::{BuiltinDecoder, FrameStream, StreamEvent, WavWriter};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let input = args
        .next()
        .ok_or("usage: decode_raw <input.av3a> <output.wav>")?;
    let output = args.next().ok_or("missing output path")?;
    if args.next().is_some() {
        return Err("too many arguments".into());
    }

    let mut file = File::open(input)?;
    let mut parser = FrameStream::new();
    let mut chunk = [0_u8; 64 * 1024];
    let mut state = None;

    let mut accept = |event: StreamEvent| -> Result<(), Box<dyn Error>> {
        let frame = match event {
            StreamEvent::Frame(frame) => frame,
            StreamEvent::Skipped { bytes } => {
                // Resynchronisation is observable, not seamless audio recovery.
                return Err(format!("parser skipped {bytes} bytes").into());
            }
        };
        if state.is_none() {
            let decoder = BuiltinDecoder::configure(frame.header())?;
            let pcm = vec![0_i16; decoder.sample_count()?];
            let config = decoder.config();
            let channels = u16::from(config.channels);
            let wav = WavWriter::create(&output, channels, config.sample_rate)?;
            state = Some((decoder, pcm, wav));
        }
        let (decoder, pcm, wav) = state.as_mut().ok_or("decoder not initialized")?;
        decoder.decode_into(&frame, pcm)?;
        wav.write_samples(pcm)?;
        Ok(())
    };

    loop {
        let count = file.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        for event in parser.push(&chunk[..count])? {
            accept(event)?;
        }
    }
    // Only call finish at actual EOF, not at the end of each input chunk.
    for event in parser.finish()? {
        accept(event)?;
    }
    let (_, _, wav) = state.ok_or("no complete AV3A frame found")?;
    wav.finalize()?;
    Ok(())
}
