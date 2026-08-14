use std::env;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use avs3a::{
    CodecProfile, DecoderConfig, EncodedFrame, FrameHeader, FrameStream, McSideInfoDecoder,
    MetadataPayloadParser, SoundBedType, StreamEvent,
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
    let mut args = env::args_os();
    let program = args.next().unwrap_or_default();
    let mut verify_crc = false;
    let mut mc_side_info = false;
    let mut input = None;

    for argument in args {
        if argument == "--verify-crc" {
            verify_crc = true;
        } else if argument == "--mc-side-info" {
            mc_side_info = true;
        } else if argument == "-h" || argument == "--help" {
            print_usage(&program);
            return Ok(());
        } else if input.replace(PathBuf::from(argument)).is_some() {
            return Err("only one input file may be specified".into());
        }
    }
    let Some(input) = input else {
        print_usage(&program);
        return Err("missing input file".into());
    };

    let file = File::open(&input)?;
    let mut reader = BufReader::new(file);
    let mut parser = FrameStream::new();
    let mut buffer = [0_u8; READ_BUFFER_SIZE];
    let mut summary = Summary::default();

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        for event in parser.push(&buffer[..read])? {
            summary.accept(event, verify_crc)?;
        }
    }
    for event in parser.finish()? {
        summary.accept(event, verify_crc)?;
    }
    summary.print(&input)?;
    if mc_side_info {
        summary.print_mc_side_info()?;
    }
    Ok(())
}

fn print_usage(program: &std::ffi::OsStr) {
    eprintln!(
        "Usage: {} [--verify-crc] [--mc-side-info] <input.av3a>\n\nParses an AV3A elementary stream without invoking a synthesis backend.",
        PathBuf::from(program).display()
    );
}

#[derive(Debug, Default)]
struct Summary {
    first: Option<EncodedFrame>,
    last_config: Option<DecoderConfig>,
    frames: u64,
    bytes: u64,
    skipped: u64,
    crc_failures: u64,
    config_changes: u64,
}

impl Summary {
    fn accept(
        &mut self,
        event: StreamEvent,
        verify_crc: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match event {
            StreamEvent::Skipped { bytes } => {
                self.skipped = self.skipped.saturating_add(u64::try_from(bytes)?);
            }
            StreamEvent::Frame(frame) => {
                record_configuration_change(
                    &mut self.last_config,
                    &mut self.config_changes,
                    frame.header(),
                );
                if !frame.crc_is_valid() {
                    self.crc_failures = self.crc_failures.saturating_add(1);
                    if verify_crc {
                        return Err(format!(
                            "CRC mismatch in frame {}: expected 0x{:04x}, got 0x{:04x}",
                            self.frames,
                            frame.expected_crc(),
                            frame.actual_crc()
                        )
                        .into());
                    }
                }
                self.bytes = self
                    .bytes
                    .saturating_add(u64::try_from(frame.bytes().len())?);
                self.frames = self.frames.saturating_add(1);
                if self.first.is_none() {
                    self.first = Some(frame);
                }
            }
        }
        Ok(())
    }

    fn print(&self, input: &std::path::Path) -> io::Result<()> {
        let Some(first) = &self.first else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "no complete AV3A frames found",
            ));
        };
        let header = first.header();
        println!("file: {}", input.display());
        println!("profile: {:?}", header.profile);
        println!(
            "channel layout: {}",
            header
                .channel_config
                .map_or_else(|| "objects".to_owned(), |value| value.to_string())
        );
        println!(
            "channels: {} ({} bed + {} objects)",
            header.channels, header.bed_channels, header.objects
        );
        println!("sample rate: {} Hz", header.sample_rate);
        println!("bit depth: {} bit", header.bit_depth.bits());
        println!("bitrate: {} bps", header.bitrate);
        if header.profile == CodecProfile::Mixed {
            let bed = match header.sound_bed_type {
                Some(SoundBedType::ObjectsOnly) => "objects only",
                Some(SoundBedType::ChannelBed) => "channel bed",
                None => "unknown",
            };
            println!("sound bed: {bed}");
            if let Some(bitrate) = header.bed_bitrate {
                println!("bed bitrate: {bitrate} bps");
            }
            if let Some(bitrate) = header.object_bitrate {
                println!("object bitrate: {bitrate} bps per object");
            }
        }
        println!(
            "frame size: {} header + {} payload bytes",
            header.header_len, header.payload_len
        );
        println!("frames: {}", self.frames);
        println!("parsed bytes: {}", self.bytes);
        println!("skipped bytes: {}", self.skipped);
        println!("CRC failures: {}", self.crc_failures);
        println!("configuration changes: {}", self.config_changes);
        Ok(())
    }

    fn print_mc_side_info(&self) -> Result<(), Box<dyn std::error::Error>> {
        let first = self.first.as_ref().ok_or("no complete AV3A frames found")?;
        let header = first.header();
        let mut metadata_parser = MetadataPayloadParser::new();
        let metadata = metadata_parser.parse_with_object_count(
            first.payload(),
            header.payload_bits,
            usize::from(header.objects),
        )?;
        let metadata_summary = metadata.summary();
        let mut audio_header = *header;
        audio_header.payload_bits = metadata.audio_bits();
        audio_header.payload_len = metadata.audio_payload().len();
        audio_header.frame_len = audio_header.header_len + audio_header.payload_len;

        let mut mc_parser = McSideInfoDecoder::new();
        let side = mc_parser.parse(metadata.audio_payload(), &audio_header)?;
        let mc = side.mc();
        println!(
            "first-frame metadata bits: {}",
            metadata_summary.consumed_bits()
        );
        println!("first-frame MC side/QC bits: {}", side.consumed_bits());
        println!("first-frame padding bits: {}", side.padding_bits());
        println!(
            "first-frame entropy bytes: {:?}",
            side.allocation().channel_bytes()
        );
        println!("first-frame silence flags: {:?}", mc.silence_flags());
        println!(
            "first-frame pairs: {:?}",
            mc.pairs()
                .iter()
                .map(|pair| (pair.first(), pair.second()))
                .collect::<Vec<_>>()
        );
        println!("first-frame ILD indexes: {:?}", mc.ild_indexes());
        println!("first-frame bit ratios: {:?}", mc.bit_ratios());
        Ok(())
    }
}

fn record_configuration_change(
    last: &mut Option<DecoderConfig>,
    changes: &mut u64,
    header: &FrameHeader,
) {
    let current = DecoderConfig::from_header(header);
    if last.is_some_and(|previous| previous != current) {
        *changes = changes.saturating_add(1);
    }
    *last = Some(current);
}

#[cfg(test)]
mod tests {
    use avs3a::{AudioCodecId, BitDepth, ChannelConfig, NnType};

    use super::*;

    fn header() -> FrameHeader {
        FrameHeader {
            codec_id: AudioCodecId::Avs3P3,
            nn_type: NnType::Main,
            profile: CodecProfile::Mixed,
            sample_rate: 48_000,
            bit_depth: BitDepth::Sixteen,
            channel_config: Some(ChannelConfig::Stereo),
            sound_bed_type: Some(SoundBedType::ChannelBed),
            hoa_order: None,
            objects: 1,
            bed_channels: 2,
            channels: 3,
            has_lfe: false,
            bed_bitrate: Some(448_000),
            object_bitrate: Some(64_000),
            bitrate: 512_000,
            crc: 0,
            header_len: 9,
            payload_bits: 10_850,
            payload_len: 1_357,
            frame_len: 1_366,
            samples_per_channel: 1_024,
        }
    }

    #[test]
    fn counts_adjacent_full_configuration_transitions() {
        let first = header();
        let mut changed = first;
        changed.object_bitrate = Some(72_000);
        changed.bitrate = 520_000;
        let mut last = None;
        let mut changes = 0;

        record_configuration_change(&mut last, &mut changes, &first);
        record_configuration_change(&mut last, &mut changes, &first);
        record_configuration_change(&mut last, &mut changes, &changed);
        record_configuration_change(&mut last, &mut changes, &changed);
        record_configuration_change(&mut last, &mut changes, &first);

        assert_eq!(changes, 2);
    }
}
