use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufReader, ErrorKind, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use avs3a::mp4::ISO_BMFF_SNIFF_BYTES;
use avs3a::neural::AVS3_FEATURE_DIMENSIONS;
use avs3a::{
    BuiltinDecoder, EncodedFrame, FrameStream, Mp4FrameReader, StreamEvent, WavWriter, is_iso_bmff,
};

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
    let max_frames = parse_optional_frame_limit(&mut arguments)?;

    let input = PathBuf::from(input);
    let output = PathBuf::from(output);
    let mut file = File::open(&input)?;
    let mut state = DecodeState::new(&output, max_frames);
    let container = if sniff_iso_bmff(&mut file)? {
        decode_container(BufReader::new(file), &mut state)?
    } else {
        decode_elementary_stream(BufReader::new(file), &mut state)?
    };
    state.finish(&input, container)?;
    Ok(())
}

/// What the input file turned out to be.
enum Container {
    ElementaryStream,
    IsoBmff { samples: usize, seconds: f64 },
}

/// Peek at the file's first bytes and rewind.
fn sniff_iso_bmff(file: &mut File) -> Result<bool, Box<dyn std::error::Error>> {
    let mut prefix = [0_u8; ISO_BMFF_SNIFF_BYTES];
    let detected = match file.read_exact(&mut prefix) {
        Ok(()) => is_iso_bmff(&prefix),
        // A file too short to hold a box header cannot be a container.
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => false,
        Err(error) => return Err(error.into()),
    };
    file.seek(SeekFrom::Start(0))?;
    Ok(detected)
}

fn decode_container<R: Read + Seek>(
    reader: R,
    state: &mut DecodeState<'_>,
) -> Result<Container, Box<dyn std::error::Error>> {
    let mut frames = Mp4FrameReader::open(reader)?;
    let container = Container::IsoBmff {
        samples: frames.track().samples().len(),
        seconds: frames.track().duration_seconds(),
    };
    while let Some(frame) = frames.next_frame()? {
        state.decode_frame(&frame)?;
        if state.reached_limit() {
            break;
        }
    }
    Ok(container)
}

fn decode_elementary_stream<R: Read>(
    mut reader: R,
    state: &mut DecodeState<'_>,
) -> Result<Container, Box<dyn std::error::Error>> {
    let mut parser = FrameStream::new();
    let mut buffer = [0_u8; READ_BUFFER_SIZE];

    'input: loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        for event in parser.push(&buffer[..read])? {
            state.accept(event)?;
            if state.reached_limit() {
                break 'input;
            }
        }
    }
    if !state.reached_limit() {
        for event in parser.finish()? {
            state.accept(event)?;
        }
    }
    Ok(Container::ElementaryStream)
}

fn parse_optional_frame_limit(
    arguments: &mut impl Iterator<Item = OsString>,
) -> Result<Option<u64>, Box<dyn std::error::Error>> {
    let Some(option) = arguments.next() else {
        return Ok(None);
    };
    if option != "--frames" {
        return Err(format!("unknown option: {}", option.to_string_lossy()).into());
    }
    let value = arguments
        .next()
        .ok_or("--frames requires a positive integer")?;
    if arguments.next().is_some() {
        return Err("too many command-line arguments".into());
    }
    let value = value
        .to_str()
        .ok_or("--frames value is not valid UTF-8")?
        .parse::<u64>()?;
    if value == 0 {
        return Err("--frames must be greater than zero".into());
    }
    Ok(Some(value))
}

fn print_usage(program: &std::ffi::OsStr) {
    eprintln!(
        "Usage: {} <input.av3a|input.mp4|input.m4a> <output.wav> [--frames <count>]\n\nDecodes channel-based, Mix or HOA AVS3 to PCM16 WAV.\nAccepts a raw elementary stream or an MP4/M4A container; the format is detected.",
        PathBuf::from(program).display()
    );
}

struct DecodeState<'path> {
    output_path: &'path Path,
    decoder: Option<BuiltinDecoder>,
    wav: Option<WavWriter<File>>,
    samples: Vec<i16>,
    frames: u64,
    max_frames: Option<u64>,
    channels: u16,
    sample_rate: u32,
}

impl<'path> DecodeState<'path> {
    fn new(output_path: &'path Path, max_frames: Option<u64>) -> Self {
        Self {
            output_path,
            decoder: None,
            wav: None,
            samples: Vec::new(),
            frames: 0,
            max_frames,
            channels: 0,
            sample_rate: 0,
        }
    }

    fn reached_limit(&self) -> bool {
        self.max_frames.is_some_and(|limit| self.frames >= limit)
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
        if self.reached_limit() {
            return Ok(());
        }
        if self.decoder.is_none() {
            let channels = u16::from(frame.header().channels);
            let sample_count = usize::from(channels)
                .checked_mul(AVS3_FEATURE_DIMENSIONS)
                .ok_or("PCM frame sample count overflow")?;
            self.decoder = Some(BuiltinDecoder::configure(frame.header())?);
            self.wav = Some(WavWriter::create(
                self.output_path,
                channels,
                frame.header().sample_rate,
            )?);
            self.samples.resize(sample_count, 0);
            self.channels = channels;
            self.sample_rate = frame.header().sample_rate;
        }

        let decoder = self.decoder.as_mut().expect("decoder initialized above");
        decoder.decode_into(frame, &mut self.samples)?;
        self.wav
            .as_mut()
            .expect("WAV writer initialized with decoder")
            .write_samples(&self.samples)?;
        self.frames = self.frames.saturating_add(1);
        Ok(())
    }

    fn finish(
        mut self,
        input: &Path,
        container: Container,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(wav) = self.wav.take() else {
            return Err("no complete supported AVS3 frame found".into());
        };
        wav.finalize()?;
        println!("input: {}", input.display());
        match container {
            Container::ElementaryStream => println!("container: raw elementary stream"),
            Container::IsoBmff { samples, seconds } => {
                println!("container: MP4 ({samples} samples, {seconds:.3} s)");
            }
        }
        println!("output: {}", self.output_path.display());
        println!(
            "format: {} channels at {} Hz",
            self.channels, self.sample_rate
        );
        println!("frames: {}", self.frames);
        println!(
            "clipped samples: {}",
            self.decoder
                .as_ref()
                .map_or(0, BuiltinDecoder::total_clipped_samples)
        );
        Ok(())
    }
}
