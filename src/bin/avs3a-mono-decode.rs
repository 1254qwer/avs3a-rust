use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use avs3a::backend::MonoDecoderBackend;
use avs3a::decode::{Decoder, PendingDecoder};
use avs3a::{EncodedFrame, FrameStream, StreamEvent, WavWriter};

const READ_BUFFER_SIZE: usize = 64 * 1024;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(input) = arguments.next() else {
        print_usage(&program);
        return Err("missing AV3A input path".into());
    };
    if input == "-h" || input == "--help" {
        print_usage(&program);
        return Ok(());
    }
    let Some(output) = arguments.next() else {
        print_usage(&program);
        return Err("missing WAV output path".into());
    };
    if arguments.next().is_some() {
        return Err("only input and output paths may be specified".into());
    }

    let input = PathBuf::from(input);
    let output = PathBuf::from(output);
    let mut reader = BufReader::new(File::open(&input)?);
    let mut parser = FrameStream::new();
    let mut buffer = [0_u8; READ_BUFFER_SIZE];
    let mut state = DecodeState::new(&output);

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        for event in parser.push(&buffer[..read])? {
            state.accept(event)?;
        }
    }
    for event in parser.finish()? {
        state.accept(event)?;
    }
    state.finish(&input)?;
    Ok(())
}

fn print_usage(program: &std::ffi::OsStr) {
    eprintln!(
        "Usage: {} <input.av3a> <output.wav>\n\nDecodes channel-based mono AVS3 to PCM16 WAV. Stereo, MC, HOA and Mix streams are rejected.",
        PathBuf::from(program).display()
    );
}

struct DecodeState<'path> {
    output_path: &'path Path,
    decoder: Option<Decoder<MonoDecoderBackend>>,
    wav: Option<WavWriter<File>>,
    frames: u64,
    clipped_samples: u64,
}

impl<'path> DecodeState<'path> {
    fn new(output_path: &'path Path) -> Self {
        Self {
            output_path,
            decoder: None,
            wav: None,
            frames: 0,
            clipped_samples: 0,
        }
    }

    fn accept(&mut self, event: StreamEvent) -> Result<(), Box<dyn std::error::Error>> {
        match event {
            StreamEvent::Skipped { bytes } => {
                return Err(format!(
                    "refusing to synthesize after parser resynchronization skipped {bytes} bytes"
                )
                .into());
            }
            StreamEvent::Frame(frame) => self.decode_frame(&frame)?,
        }
        Ok(())
    }

    fn decode_frame(&mut self, frame: &EncodedFrame) -> Result<(), Box<dyn std::error::Error>> {
        if self.decoder.is_none() {
            let decoder = PendingDecoder::new(MonoDecoderBackend::new_builtin()?)
                .configure(frame.header())?;
            let wav = WavWriter::create(self.output_path, 1, frame.header().sample_rate)?;
            self.decoder = Some(decoder);
            self.wav = Some(wav);
        }

        let decoder = self.decoder.as_mut().expect("decoder initialized above");
        let audio = decoder.decode(frame)?;
        self.clipped_samples = self
            .clipped_samples
            .saturating_add(u64::try_from(decoder.last_clipped_samples()).unwrap_or(u64::MAX));
        self.wav
            .as_mut()
            .expect("WAV writer initialized with decoder")
            .write_samples(audio.samples())?;
        self.frames = self.frames.saturating_add(1);
        Ok(())
    }

    fn finish(mut self, input: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let Some(wav) = self.wav.take() else {
            return Err("no complete mono AVS3 frame found".into());
        };
        wav.finalize()?;
        println!("input: {}", input.display());
        println!("output: {}", self.output_path.display());
        println!("frames: {}", self.frames);
        println!("clipped samples: {}", self.clipped_samples);
        Ok(())
    }
}
