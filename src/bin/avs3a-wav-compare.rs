use std::env;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const COMPARE_BUFFER_BYTES: usize = 64 * 1024;

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
    let Some(left_path) = arguments.next() else {
        print_usage(&program);
        return Err("missing first WAV path".into());
    };
    if left_path == "-h" || left_path == "--help" {
        print_usage(&program);
        return Ok(());
    }
    let Some(right_path) = arguments.next() else {
        print_usage(&program);
        return Err("missing second WAV path".into());
    };
    if arguments.next().is_some() {
        return Err("too many command-line arguments".into());
    }

    let left_path = PathBuf::from(left_path);
    let right_path = PathBuf::from(right_path);
    let mut left = Pcm16Wav::new(BufReader::new(File::open(&left_path)?))?;
    let mut right = Pcm16Wav::new(BufReader::new(File::open(&right_path)?))?;
    let comparison = compare(&mut left, &mut right)?;
    comparison.print(&left_path, &right_path);
    Ok(())
}

fn print_usage(program: &std::ffi::OsStr) {
    eprintln!(
        "Usage: {} <left.wav> <right.wav>\n\nCompares PCM16 WAV data without loading either file into memory.",
        PathBuf::from(program).display()
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pcm16Format {
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    data_offset: u64,
    data_bytes: u64,
}

impl Pcm16Format {
    fn frames(self) -> u64 {
        self.data_bytes / u64::from(self.block_align)
    }
}

#[derive(Debug)]
struct Pcm16Wav<R> {
    reader: R,
    format: Pcm16Format,
}

impl<R: Read + Seek> Pcm16Wav<R> {
    fn new(mut reader: R) -> io::Result<Self> {
        let format = parse_pcm16_format(&mut reader)?;
        reader.seek(SeekFrom::Start(format.data_offset))?;
        Ok(Self { reader, format })
    }
}

fn parse_pcm16_format(reader: &mut (impl Read + Seek)) -> io::Result<Pcm16Format> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;

    let mut riff_header = [0_u8; 12];
    reader.read_exact(&mut riff_header)?;
    if &riff_header[0..4] != b"RIFF" || &riff_header[8..12] != b"WAVE" {
        return Err(invalid_data("input is not a little-endian RIFF/WAVE file"));
    }
    let riff_size = u64::from(u32::from_le_bytes(
        riff_header[4..8].try_into().expect("four-byte RIFF size"),
    ));
    let riff_end = 8_u64
        .checked_add(riff_size)
        .ok_or_else(|| invalid_data("RIFF size overflow"))?;
    if riff_end > file_len {
        return Err(invalid_data("RIFF size exceeds the file length"));
    }

    let mut format = None;
    let mut data = None;
    let mut offset = 12_u64;
    while offset < riff_end {
        if riff_end - offset < 8 {
            return Err(invalid_data("truncated RIFF chunk header"));
        }
        reader.seek(SeekFrom::Start(offset))?;
        let mut chunk_header = [0_u8; 8];
        reader.read_exact(&mut chunk_header)?;
        let chunk_size = u64::from(u32::from_le_bytes(
            chunk_header[4..8]
                .try_into()
                .expect("four-byte RIFF chunk size"),
        ));
        let chunk_data = offset
            .checked_add(8)
            .ok_or_else(|| invalid_data("RIFF chunk offset overflow"))?;
        let chunk_end = chunk_data
            .checked_add(chunk_size)
            .and_then(|value| value.checked_add(chunk_size & 1))
            .ok_or_else(|| invalid_data("RIFF chunk size overflow"))?;
        if chunk_end > riff_end {
            return Err(invalid_data("RIFF chunk exceeds the declared RIFF size"));
        }

        match &chunk_header[0..4] {
            b"fmt " => {
                if format.is_some() {
                    return Err(invalid_data("duplicate WAV fmt chunk"));
                }
                if chunk_size < 16 {
                    return Err(invalid_data("WAV fmt chunk is shorter than 16 bytes"));
                }
                let mut bytes = [0_u8; 16];
                reader.read_exact(&mut bytes)?;
                let audio_format = u16::from_le_bytes(bytes[0..2].try_into().unwrap());
                let channels = u16::from_le_bytes(bytes[2..4].try_into().unwrap());
                let sample_rate = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
                let byte_rate = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
                let block_align = u16::from_le_bytes(bytes[12..14].try_into().unwrap());
                let bits_per_sample = u16::from_le_bytes(bytes[14..16].try_into().unwrap());
                if audio_format != 1 || bits_per_sample != 16 {
                    return Err(invalid_data(format!(
                        "unsupported WAV format {audio_format} with {bits_per_sample} bits per sample"
                    )));
                }
                if channels == 0 || sample_rate == 0 {
                    return Err(invalid_data(
                        "WAV channels and sample rate must be non-zero",
                    ));
                }
                let expected_align = channels
                    .checked_mul(2)
                    .ok_or_else(|| invalid_data("WAV block alignment overflow"))?;
                let expected_rate = sample_rate
                    .checked_mul(u32::from(expected_align))
                    .ok_or_else(|| invalid_data("WAV byte rate overflow"))?;
                if block_align != expected_align || byte_rate != expected_rate {
                    return Err(invalid_data("inconsistent PCM16 WAV rate or alignment"));
                }
                format = Some((channels, sample_rate, block_align));
            }
            b"data" if data.is_some() => {
                return Err(invalid_data("duplicate WAV data chunk"));
            }
            b"data" => data = Some((chunk_data, chunk_size)),
            _ => {}
        }
        offset = chunk_end;
    }

    let (channels, sample_rate, block_align) =
        format.ok_or_else(|| invalid_data("missing WAV fmt chunk"))?;
    let (data_offset, data_bytes) = data.ok_or_else(|| invalid_data("missing WAV data chunk"))?;
    if data_bytes % u64::from(block_align) != 0 {
        return Err(invalid_data(
            "WAV data length is not a whole number of interleaved frames",
        ));
    }
    Ok(Pcm16Format {
        channels,
        sample_rate,
        block_align,
        data_offset,
        data_bytes,
    })
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FirstMismatch {
    sample_index: u64,
    left: i16,
    right: i16,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Metrics {
    samples: u64,
    different: u64,
    max_absolute_error: u32,
    squared_error: u128,
    first_mismatch: Option<FirstMismatch>,
}

impl Metrics {
    fn observe(&mut self, sample_index: u64, left: i16, right: i16) {
        self.samples = self.samples.saturating_add(1);
        let difference = i32::from(left) - i32::from(right);
        if difference == 0 {
            return;
        }
        let absolute = difference.unsigned_abs();
        self.different = self.different.saturating_add(1);
        self.max_absolute_error = self.max_absolute_error.max(absolute);
        self.squared_error = self
            .squared_error
            .saturating_add(u128::from(absolute) * u128::from(absolute));
        self.first_mismatch.get_or_insert(FirstMismatch {
            sample_index,
            left,
            right,
        });
    }

    fn difference_percent(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            100.0 * self.different as f64 / self.samples as f64
        }
    }

    fn rms_error(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            (self.squared_error as f64 / self.samples as f64).sqrt()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Comparison {
    format: Pcm16Format,
    left_frames: u64,
    right_frames: u64,
    total: Metrics,
    channels: Vec<Metrics>,
}

impl Comparison {
    fn print(&self, left: &Path, right: &Path) {
        println!("left: {}", left.display());
        println!("right: {}", right.display());
        println!(
            "format: {} channels at {} Hz, PCM16",
            self.format.channels, self.format.sample_rate
        );
        println!(
            "frames: {} left, {} right, {} compared",
            self.left_frames,
            self.right_frames,
            self.total.samples / u64::from(self.format.channels)
        );
        println!("samples compared: {}", self.total.samples);
        println!(
            "different samples: {} ({:.6}%)",
            self.total.different,
            self.total.difference_percent()
        );
        println!("maximum absolute error: {}", self.total.max_absolute_error);
        println!("RMS error: {:.9} LSB", self.total.rms_error());
        if let Some(first) = self.total.first_mismatch {
            let channels = u64::from(self.format.channels);
            println!(
                "first mismatch: frame {}, channel {}, left {}, right {}, delta {}",
                first.sample_index / channels,
                first.sample_index % channels,
                first.left,
                first.right,
                i32::from(first.left) - i32::from(first.right)
            );
        } else {
            println!("first mismatch: none");
        }
        for (channel, metrics) in self.channels.iter().enumerate() {
            println!(
                "channel {channel}: different {} ({:.6}%), max {}, RMS {:.9} LSB",
                metrics.different,
                metrics.difference_percent(),
                metrics.max_absolute_error,
                metrics.rms_error()
            );
        }
        if self.left_frames == self.right_frames && self.total.different == 0 {
            println!("result: identical PCM16 data");
        } else {
            println!("result: PCM16 data differs");
        }
    }
}

fn compare<L: Read + Seek, R: Read + Seek>(
    left: &mut Pcm16Wav<L>,
    right: &mut Pcm16Wav<R>,
) -> io::Result<Comparison> {
    if (left.format.channels, left.format.sample_rate)
        != (right.format.channels, right.format.sample_rate)
    {
        return Err(invalid_data(format!(
            "WAV formats differ: {} channels at {} Hz versus {} channels at {} Hz",
            left.format.channels,
            left.format.sample_rate,
            right.format.channels,
            right.format.sample_rate
        )));
    }

    left.reader.seek(SeekFrom::Start(left.format.data_offset))?;
    right
        .reader
        .seek(SeekFrom::Start(right.format.data_offset))?;
    let compared_bytes = left.format.data_bytes.min(right.format.data_bytes);
    let mut remaining = compared_bytes;
    let mut sample_index = 0_u64;
    let mut total = Metrics::default();
    let mut channels = vec![Metrics::default(); usize::from(left.format.channels)];
    let mut left_bytes = vec![0_u8; COMPARE_BUFFER_BYTES];
    let mut right_bytes = vec![0_u8; COMPARE_BUFFER_BYTES];

    while remaining != 0 {
        let take = usize::try_from(remaining.min(COMPARE_BUFFER_BYTES as u64))
            .expect("comparison chunk always fits usize");
        left.reader.read_exact(&mut left_bytes[..take])?;
        right.reader.read_exact(&mut right_bytes[..take])?;
        // Fixed-size chunks let the samples be decoded without a fallible
        // slice-to-array conversion at every step.
        let (left_pairs, _) = left_bytes[..take].as_chunks::<2>();
        let (right_pairs, _) = right_bytes[..take].as_chunks::<2>();
        for (left_sample, right_sample) in left_pairs.iter().zip(right_pairs) {
            let left_sample = i16::from_le_bytes(*left_sample);
            let right_sample = i16::from_le_bytes(*right_sample);
            total.observe(sample_index, left_sample, right_sample);
            let channel = usize::try_from(sample_index % u64::from(left.format.channels))
                .expect("channel index fits usize");
            channels[channel].observe(sample_index, left_sample, right_sample);
            sample_index = sample_index.saturating_add(1);
        }
        remaining -= u64::try_from(take).expect("comparison chunk fits u64");
    }

    Ok(Comparison {
        format: left.format,
        left_frames: left.format.frames(),
        right_frames: right.format.frames(),
        total,
        channels,
    })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use avs3a::WavWriter;

    use super::*;

    fn wav(samples: &[i16], channels: u16) -> Cursor<Vec<u8>> {
        let mut writer = WavWriter::new(Cursor::new(Vec::new()), channels, 48_000).unwrap();
        writer.write_samples(samples).unwrap();
        writer.finalize().unwrap()
    }

    #[test]
    fn compares_pcm16_samples_and_channels() {
        let mut left = Pcm16Wav::new(wav(&[0, 100, -200, 32_767], 2)).unwrap();
        let mut right = Pcm16Wav::new(wav(&[0, 101, -198, 32_760], 2)).unwrap();
        let result = compare(&mut left, &mut right).unwrap();

        assert_eq!(result.left_frames, 2);
        assert_eq!(result.right_frames, 2);
        assert_eq!(result.total.samples, 4);
        assert_eq!(result.total.different, 3);
        assert_eq!(result.total.max_absolute_error, 7);
        assert_eq!(result.total.squared_error, 54);
        assert_eq!(
            result.total.first_mismatch,
            Some(FirstMismatch {
                sample_index: 1,
                left: 100,
                right: 101,
            })
        );
        assert_eq!(result.channels[0].different, 1);
        assert_eq!(result.channels[0].max_absolute_error, 2);
        assert_eq!(result.channels[1].different, 2);
        assert_eq!(result.channels[1].max_absolute_error, 7);
    }

    #[test]
    fn compares_the_common_prefix_when_lengths_differ() {
        let mut left = Pcm16Wav::new(wav(&[1, 2], 1)).unwrap();
        let mut right = Pcm16Wav::new(wav(&[1, 2, 3], 1)).unwrap();
        let result = compare(&mut left, &mut right).unwrap();

        assert_eq!(result.left_frames, 2);
        assert_eq!(result.right_frames, 3);
        assert_eq!(result.total.samples, 2);
        assert_eq!(result.total.different, 0);
    }

    #[test]
    fn rejects_inconsistent_formats() {
        let mut left = Pcm16Wav::new(wav(&[1, 2], 1)).unwrap();
        let mut right = Pcm16Wav::new(wav(&[1, 2], 2)).unwrap();
        let error = compare(&mut left, &mut right).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
}
