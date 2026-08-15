use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use avs3a::{
    AVS3_FEATURE_DIMENSIONS, ChannelConfig, CodecProfile, Decoder, EncodedFrame, FrameHeader,
    FrameStream, HoaDecoderBackend, McDecoderBackend, MixDecoderBackend, MonoDecoderBackend,
    PendingDecoder, StereoDecoderBackend, StreamEvent, WavWriter, is_multichannel_config,
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
    let mut reader = BufReader::new(File::open(&input)?);
    let mut parser = FrameStream::new();
    let mut buffer = [0_u8; READ_BUFFER_SIZE];
    let mut state = DecodeState::new(&output, max_frames);

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
    state.finish(&input)?;
    Ok(())
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
        "Usage: {} <input.av3a> <output.wav> [--frames <count>]\n\nDecodes channel-based, Mix or HOA AVS3 to PCM16 WAV.",
        PathBuf::from(program).display()
    );
}

#[derive(Debug)]
enum ActiveDecoder {
    Mono(Box<Decoder<MonoDecoderBackend>>),
    Stereo(Box<Decoder<StereoDecoderBackend>>),
    Mc(Box<Decoder<McDecoderBackend>>),
    Mix(Box<Decoder<MixDecoderBackend>>),
    Hoa(Box<Decoder<HoaDecoderBackend>>),
}

impl ActiveDecoder {
    fn configure(header: &FrameHeader) -> Result<Self, Box<dyn std::error::Error>> {
        match (header.profile, header.channel_config) {
            (CodecProfile::ChannelBased, Some(ChannelConfig::Mono)) => Ok(Self::Mono(Box::new(
                PendingDecoder::new(MonoDecoderBackend::new_builtin()?).configure(header)?,
            ))),
            (CodecProfile::ChannelBased, Some(ChannelConfig::Stereo)) => {
                Ok(Self::Stereo(Box::new(
                    PendingDecoder::new(StereoDecoderBackend::new_builtin()?).configure(header)?,
                )))
            }
            (CodecProfile::ChannelBased, Some(config)) if is_multichannel_config(config) => {
                Ok(Self::Mc(Box::new(
                    PendingDecoder::new(McDecoderBackend::new_builtin()?).configure(header)?,
                )))
            }
            (CodecProfile::Mixed, _) => Ok(Self::Mix(Box::new(
                PendingDecoder::new(MixDecoderBackend::new_builtin()?).configure(header)?,
            ))),
            (
                CodecProfile::Hoa,
                Some(ChannelConfig::Hoa1 | ChannelConfig::Hoa2 | ChannelConfig::Hoa3),
            ) => Ok(Self::Hoa(Box::new(
                PendingDecoder::new(HoaDecoderBackend::new_builtin()?).configure(header)?,
            ))),
            (profile, config) => Err(format!(
                "unsupported profile/channel configuration: {profile:?}/{config:?}"
            )
            .into()),
        }
    }

    fn decode_into(
        &mut self,
        frame: &EncodedFrame,
        output: &mut [i16],
    ) -> Result<(), avs3a::DecodeError> {
        match self {
            Self::Mono(decoder) => decoder.decode_into(frame, output),
            Self::Stereo(decoder) => decoder.decode_into(frame, output),
            Self::Mc(decoder) => decoder.decode_into(frame, output),
            Self::Mix(decoder) => decoder.decode_into(frame, output),
            Self::Hoa(decoder) => decoder.decode_into(frame, output),
        }
    }

    fn last_clipped_samples(&self) -> usize {
        match self {
            Self::Mono(decoder) => decoder.backend().last_clipped_samples(),
            Self::Stereo(decoder) => decoder.backend().last_clipped_samples(),
            Self::Mc(decoder) => decoder.backend().last_clipped_samples(),
            Self::Mix(decoder) => decoder.backend().last_clipped_samples(),
            Self::Hoa(decoder) => decoder.backend().last_clipped_samples(),
        }
    }
}

struct DecodeState<'path> {
    output_path: &'path Path,
    decoder: Option<ActiveDecoder>,
    wav: Option<WavWriter<File>>,
    samples: Vec<i16>,
    frames: u64,
    max_frames: Option<u64>,
    clipped_samples: u64,
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
            clipped_samples: 0,
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
            self.decoder = Some(ActiveDecoder::configure(frame.header())?);
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
        self.clipped_samples = self
            .clipped_samples
            .saturating_add(u64::try_from(decoder.last_clipped_samples()).unwrap_or(u64::MAX));
        self.wav
            .as_mut()
            .expect("WAV writer initialized with decoder")
            .write_samples(&self.samples)?;
        self.frames = self.frames.saturating_add(1);
        Ok(())
    }

    fn finish(mut self, input: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let Some(wav) = self.wav.take() else {
            return Err("no complete supported AVS3 frame found".into());
        };
        wav.finalize()?;
        println!("input: {}", input.display());
        println!("output: {}", self.output_path.display());
        println!(
            "format: {} channels at {} Hz",
            self.channels, self.sample_rate
        );
        println!("frames: {}", self.frames);
        println!("clipped samples: {}", self.clipped_samples);
        Ok(())
    }
}
