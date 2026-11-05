//! Decode AV3A in MP4/M4A, optionally starting at a zero-based track sample.
//! Run: cargo run --release --example decode_mp4 -- input.m4a output.wav [start_sample]
//! Use a complete local file and distinct input/output paths. No edit-list trimming
//! or object/HOA rendering is performed. See [the MP4 guide](../docs/mp4.md).
//! Preroll restores synthesis history, not necessarily the continuous PRNG state.

use std::env;
use std::error::Error;
use std::fs::File;

use avs3a::{BuiltinDecoder, Mp4FrameReader, WavWriter};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let input = args
        .next()
        .ok_or("usage: decode_mp4 <input.m4a> <output.wav> [start_sample]")?;
    let output = args.next().ok_or("missing output path")?;
    let target = match args.next() {
        Some(value) => value
            .to_str()
            .ok_or("invalid sample index")?
            .parse::<usize>()?,
        None => 0,
    };
    if args.next().is_some() {
        return Err("too many arguments".into());
    }

    let mut reader = Mp4FrameReader::open(File::open(input)?)?;
    let track = reader.track();
    let count = track.samples().len();
    eprintln!(
        "{} samples, {:.3} seconds; container hints: {} Hz / {} channels",
        count,
        track.duration_seconds(),
        track.declared_sample_rate(),
        track.declared_channels(),
    );
    // Valid positions are 0..=count. count itself is EOF, not the last frame.
    reader.seek_to_sample(target)?;
    if target == count {
        eprintln!("already at EOF; no output file created");
        return Ok(());
    }

    // Configure from an in-band header, never from the container hints.
    reader.seek_to_sample(0)?;
    let first = reader.next_frame()?.ok_or("empty track")?;
    let mut decoder = BuiltinDecoder::configure(first.header())?;
    let config = decoder.config();
    eprintln!(
        "decoded format: {} Hz / {} channels",
        config.sample_rate, config.channels
    );
    let mut pcm = vec![0_i16; decoder.sample_count()?];

    let warmup = usize::try_from(decoder.warmup_frames())?;
    let start = target.saturating_sub(warmup);
    reader.seek_to_sample(start)?;
    // Also use reset when reusing a decoder after an earlier playback position.
    // It retains this stream's configuration; it does not move the reader.
    decoder.reset()?;
    let mut wav = WavWriter::create(output, u16::from(config.channels), config.sample_rate)?;

    loop {
        let index = reader.position();
        let Some(frame) = reader.next_frame()? else {
            break;
        };
        decoder.decode_into(&frame, &mut pcm)?;
        if index < target {
            continue; // Decode the preroll but do not output it.
        }
        wav.write_samples(&pcm)?;
    }
    wav.finalize()?;
    Ok(())
}
