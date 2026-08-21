//! Container-layer tests against the repository's real AV3A fixture.
//!
//! The synthetic files in `src/mp4.rs` cover box-level edge cases. These tests
//! cover the part unit tests cannot: that a container built around genuine
//! frames yields exactly the frames the elementary-stream parser yields, and
//! that the numbers callers depend on for seeking and byte-range fetching are
//! right.

use std::fs;
use std::io::Cursor;
use std::path::PathBuf;

use avs3a::mp4::Mp4Error;
use avs3a::{Av3aTrack, BuiltinDecoder, EncodedFrame, FrameStream, Mp4FrameReader, StreamEvent};

/// Samples per chunk in the muxed files, chosen so the last chunk is short and
/// the sample-to-chunk table needs two runs.
const SAMPLES_PER_CHUNK: usize = 87;

fn fixture_bytes() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test.av3a");
    fs::read(path).expect("fixture is present")
}

fn elementary_frames() -> Vec<EncodedFrame> {
    let mut parser = FrameStream::new();
    let mut frames = Vec::new();
    let mut accept = |events: Vec<StreamEvent>| {
        for event in events {
            match event {
                StreamEvent::Frame(frame) => frames.push(frame),
                StreamEvent::Skipped { bytes } => panic!("fixture parser skipped {bytes} bytes"),
            }
        }
    };
    accept(parser.push(&fixture_bytes()).unwrap());
    accept(parser.finish().unwrap());
    frames
}

fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + body.len());
    out.extend_from_slice(&u32::try_from(8 + body.len()).unwrap().to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

fn full_box(version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut out = vec![version];
    out.extend_from_slice(&flags.to_be_bytes()[1..]);
    out.extend_from_slice(body);
    out
}

/// Builds an MP4 around real frames, including boxes the reader must skip.
struct Mux<'a> {
    frames: &'a [EncodedFrame],
}

impl Mux<'_> {
    fn chunk_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![SAMPLES_PER_CHUNK; self.frames.len() / SAMPLES_PER_CHUNK];
        let remainder = self.frames.len() % SAMPLES_PER_CHUNK;
        if remainder != 0 {
            sizes.push(remainder);
        }
        sizes
    }

    fn moov(&self, chunk_offsets: &[u32]) -> Vec<u8> {
        let header = self.frames[0].header();
        let count = u32::try_from(self.frames.len()).unwrap();
        let delta = header.samples_per_channel;
        let duration = u64::from(count) * u64::from(delta);

        // A bare AudioSampleEntry plus `btrt`: AV3A has no configuration box.
        let mut entry = vec![0_u8; 6];
        entry.extend_from_slice(&1_u16.to_be_bytes());
        entry.extend_from_slice(&[0; 8]);
        entry.extend_from_slice(&u16::from(header.channels).to_be_bytes());
        entry.extend_from_slice(&16_u16.to_be_bytes());
        entry.extend_from_slice(&[0; 4]);
        entry.extend_from_slice(&(header.sample_rate << 16).to_be_bytes());
        entry.extend_from_slice(&boxed(b"btrt", &[0; 12]));
        let stsd = boxed(
            b"stsd",
            &full_box(
                0,
                0,
                &[1_u32.to_be_bytes().as_slice(), &boxed(b"av3a", &entry)].concat(),
            ),
        );

        let stts = boxed(
            b"stts",
            &full_box(
                0,
                0,
                &[
                    1_u32.to_be_bytes(),
                    count.to_be_bytes(),
                    delta.to_be_bytes(),
                ]
                .concat(),
            ),
        );

        let chunk_sizes = self.chunk_sizes();
        let mut runs: Vec<(u32, u32)> = vec![(1, u32::try_from(SAMPLES_PER_CHUNK).unwrap())];
        if let Some(&last) = chunk_sizes.last().filter(|&&n| n != SAMPLES_PER_CHUNK) {
            runs.push((
                u32::try_from(chunk_sizes.len()).unwrap(),
                u32::try_from(last).unwrap(),
            ));
        }
        let mut stsc_body = u32::try_from(runs.len()).unwrap().to_be_bytes().to_vec();
        for (first_chunk, samples) in &runs {
            stsc_body.extend_from_slice(&first_chunk.to_be_bytes());
            stsc_body.extend_from_slice(&samples.to_be_bytes());
            stsc_body.extend_from_slice(&1_u32.to_be_bytes());
        }
        let stsc = boxed(b"stsc", &full_box(0, 0, &stsc_body));

        // Per-sample sizes rather than the constant form, so the real frame
        // lengths are what places every sample.
        let mut stsz_body = [0_u32.to_be_bytes(), count.to_be_bytes()].concat();
        for frame in self.frames {
            stsz_body.extend_from_slice(&u32::try_from(frame.bytes().len()).unwrap().to_be_bytes());
        }
        let stsz = boxed(b"stsz", &full_box(0, 0, &stsz_body));

        let mut stco_body = u32::try_from(chunk_offsets.len())
            .unwrap()
            .to_be_bytes()
            .to_vec();
        for offset in chunk_offsets {
            stco_body.extend_from_slice(&offset.to_be_bytes());
        }
        let stco = boxed(b"stco", &full_box(0, 0, &stco_body));

        let stbl = boxed(b"stbl", &[stsd, stts, stsc, stsz, stco].concat());
        let smhd = boxed(b"smhd", &full_box(0, 0, &[0; 4]));
        let dinf = boxed(
            b"dinf",
            &boxed(
                b"dref",
                &full_box(
                    0,
                    0,
                    &[
                        1_u32.to_be_bytes().as_slice(),
                        &boxed(b"url ", &full_box(0, 1, &[])),
                    ]
                    .concat(),
                ),
            ),
        );
        let minf = boxed(b"minf", &[smhd, dinf, stbl].concat());

        let mut mdhd_body = vec![0_u8; 8];
        mdhd_body.extend_from_slice(&header.sample_rate.to_be_bytes());
        mdhd_body.extend_from_slice(&u32::try_from(duration).unwrap().to_be_bytes());
        mdhd_body.extend_from_slice(&[0; 4]);
        let mdhd = boxed(b"mdhd", &full_box(0, 0, &mdhd_body));
        let mut hdlr_body = vec![0_u8; 4];
        hdlr_body.extend_from_slice(b"soun");
        hdlr_body.extend_from_slice(&[0; 12]);
        hdlr_body.extend_from_slice(b"AV3A\0");
        let hdlr = boxed(b"hdlr", &full_box(0, 0, &hdlr_body));
        let mdia = boxed(b"mdia", &[mdhd, hdlr, minf].concat());

        let mut tkhd_body = vec![0_u8; 8];
        tkhd_body.extend_from_slice(&7_u32.to_be_bytes()); // track_id
        tkhd_body.extend_from_slice(&[0; 72]);
        let tkhd = boxed(b"tkhd", &full_box(0, 1, &tkhd_body));
        let mut elst_body = 1_u32.to_be_bytes().to_vec();
        elst_body.extend_from_slice(&u32::try_from(duration).unwrap().to_be_bytes());
        elst_body.extend_from_slice(&0_i32.to_be_bytes());
        elst_body.extend_from_slice(&1_i16.to_be_bytes());
        elst_body.extend_from_slice(&0_i16.to_be_bytes());
        let edts = boxed(b"edts", &boxed(b"elst", &full_box(0, 0, &elst_body)));
        let trak = boxed(b"trak", &[tkhd, edts, mdia].concat());

        let mvhd = boxed(b"mvhd", &full_box(0, 0, &[0; 96]));
        boxed(b"moov", &[mvhd, trak].concat())
    }

    fn build(&self, mdat_first: bool) -> Vec<u8> {
        let ftyp = boxed(b"ftyp", b"isom\0\0\x02\0isomiso2av3a");
        let chunk_sizes = self.chunk_sizes();

        // Chunk offsets live inside `moov`, so size it with placeholders first.
        // Entry widths are fixed, so one rebuild lands on the real offsets.
        let probe = self.moov(&vec![0; chunk_sizes.len()]);
        let mdat_data_start = if mdat_first {
            ftyp.len() + 8
        } else {
            ftyp.len() + probe.len() + 8
        };

        let mut offsets = Vec::with_capacity(chunk_sizes.len());
        let mut position = mdat_data_start;
        let mut sample = 0_usize;
        for &size in &chunk_sizes {
            offsets.push(u32::try_from(position).unwrap());
            for frame in &self.frames[sample..sample + size] {
                position += frame.bytes().len();
            }
            sample += size;
        }
        let moov = self.moov(&offsets);
        assert_eq!(moov.len(), probe.len(), "placeholder moov changed size");

        let media: Vec<u8> = self
            .frames
            .iter()
            .flat_map(|frame| frame.bytes().iter().copied())
            .collect();
        let mdat = boxed(b"mdat", &media);
        if mdat_first {
            [ftyp, mdat, moov].concat()
        } else {
            [ftyp, moov, mdat].concat()
        }
    }
}

#[test]
fn container_yields_the_same_frames_as_the_elementary_stream() {
    let frames = elementary_frames();
    assert!(frames.len() > 1_000, "fixture is unexpectedly short");
    let file = Mux { frames: &frames }.build(false);

    let mut reader = Mp4FrameReader::open(Cursor::new(file)).expect("indexes the container");
    let track = reader.track();
    assert_eq!(track.track_id(), 7);
    assert_eq!(track.samples().len(), frames.len());
    assert_eq!(track.timescale(), frames[0].header().sample_rate);
    assert_eq!(
        track.declared_channels(),
        u16::from(frames[0].header().channels)
    );
    assert_eq!(track.edits().len(), 1);
    assert!(track.edits()[0].is_identity());

    for (index, expected) in frames.iter().enumerate() {
        let frame = reader
            .next_frame()
            .expect("read succeeds")
            .unwrap_or_else(|| panic!("container ended at sample {index}"));
        assert_eq!(frame.bytes(), expected.bytes(), "sample {index} differs");
        assert!(frame.crc_is_valid());
    }
    assert!(reader.next_frame().expect("read succeeds").is_none());
}

#[test]
fn timestamps_and_seek_lookup_span_the_whole_track() {
    let frames = elementary_frames();
    let file = Mux { frames: &frames }.build(false);
    let track = Av3aTrack::read_from(&mut Cursor::new(file)).expect("indexes the container");

    let delta = u64::from(frames[0].header().samples_per_channel);
    assert_eq!(track.duration(), delta * frames.len() as u64);
    for index in [0_usize, 1, 86, 87, 88, frames.len() - 1] {
        let sample = track.samples()[index];
        assert_eq!(sample.timestamp, delta * index as u64);
        assert_eq!(track.sample_at_time(sample.timestamp), Some(index));
        // Mid-frame timestamps must resolve to the frame that contains them.
        assert_eq!(
            track.sample_at_time(sample.timestamp + delta - 1),
            Some(index)
        );
    }

    // Chunk boundaries are where offset arithmetic would drift.
    let (start, end) = track.data_range().unwrap().expect("track has samples");
    let total: u64 = frames.iter().map(|frame| frame.bytes().len() as u64).sum();
    assert_eq!(end - start, total);
}

#[test]
fn partial_prefix_reports_the_exact_length_needed_to_index() {
    let frames = elementary_frames();
    // `mdat` first is the layout that forces a fetcher to read past the media.
    let file = Mux { frames: &frames }.build(true);
    let expected = Av3aTrack::read_from(&mut Cursor::new(file.clone()))
        .expect("indexes the container")
        .samples()
        .len();

    // Start from a prefix that stops inside `ftyp` and follow the reported
    // lengths. Each step must be strictly larger, or a fetcher would spin.
    let mut needed = 4_u64;
    let mut steps = 0;
    let track = loop {
        steps += 1;
        assert!(steps < 8, "did not converge after {steps} requests");
        let prefix = &file[..usize::try_from(needed).unwrap()];
        match Av3aTrack::from_prefix(prefix) {
            Ok(track) => break track,
            Err(Mp4Error::NeedMoreData {
                needed: next,
                available,
            }) => {
                assert_eq!(available, needed);
                assert!(next > needed, "{next} does not advance past {needed}");
                assert!(next <= file.len() as u64);
                needed = next;
            }
            Err(error) => panic!("unexpected error: {error}"),
        }
    };
    assert_eq!(track.samples().len(), expected);
    // `ftyp` header, `ftyp` body, `mdat` header, past `mdat` to the `moov`
    // header, then the `moov` body. Only the last request is large.
    assert_eq!(steps, 5);
}

#[test]
fn decoding_through_the_container_matches_the_elementary_stream() {
    std::thread::Builder::new()
        .name("mp4-container-decode".to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(run_decode_comparison)
        .unwrap()
        .join()
        .unwrap();
}

fn run_decode_comparison() {
    const FRAMES: usize = 6;
    let frames = elementary_frames();
    let file = Mux { frames: &frames }.build(false);
    let mut reader = Mp4FrameReader::open(Cursor::new(file)).expect("indexes the container");

    let mut direct = BuiltinDecoder::configure(frames[0].header()).unwrap();
    let mut through = BuiltinDecoder::configure(frames[0].header()).unwrap();
    let mut expected = vec![0_i16; direct.sample_count().unwrap()];
    let mut actual = vec![0_i16; through.sample_count().unwrap()];

    for (index, expected_frame) in frames.iter().take(FRAMES).enumerate() {
        let frame = reader.next_frame().unwrap().expect("a frame");
        direct.decode_into(expected_frame, &mut expected).unwrap();
        through.decode_into(&frame, &mut actual).unwrap();
        assert_eq!(actual, expected, "frame {index} decoded differently");
    }
}

#[test]
fn warmup_frames_is_exactly_what_seeking_needs() {
    std::thread::Builder::new()
        .name("mp4-container-seek".to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(run_seek_warmup)
        .unwrap()
        .join()
        .unwrap();
}

/// Pins `warmup_frames()` to the smallest count that reproduces a linear decode.
///
/// A too-small value leaves an audible transient on every seek and a too-large
/// one wastes work, so the test asserts both that the advertised count is enough
/// and that one fewer is not.
fn run_seek_warmup() {
    const TARGET: usize = 5;
    let frames = elementary_frames();
    let file = Mux { frames: &frames }.build(false);
    let track = Av3aTrack::read_from(&mut Cursor::new(file.clone())).unwrap();

    let mut decoder = BuiltinDecoder::configure(frames[0].header()).unwrap();
    let warmup = usize::try_from(decoder.warmup_frames()).unwrap();
    assert!(
        TARGET > warmup,
        "target frame must sit past the warm-up depth"
    );

    // Linear decode through the target frame is the reference.
    let mut linear = vec![0_i16; decoder.sample_count().unwrap()];
    for frame in &frames[..=TARGET] {
        decoder.decode_into(frame, &mut linear).unwrap();
    }

    let decode_from = |start: usize| {
        let mut reader = Mp4FrameReader::open(Cursor::new(file.clone())).unwrap();
        reader.seek_to_sample(start).unwrap();
        let mut decoder = BuiltinDecoder::configure(frames[0].header()).unwrap();
        let mut output = vec![0_i16; decoder.sample_count().unwrap()];
        for _ in start..=TARGET {
            let frame = reader.next_frame().unwrap().expect("a frame");
            decoder.decode_into(&frame, &mut output).unwrap();
        }
        output
    };

    assert_eq!(
        decode_from(TARGET - warmup),
        linear,
        "{warmup} warm-up frames should reproduce a linear decode"
    );
    assert_ne!(
        decode_from(TARGET),
        linear,
        "decoding the target frame with no warm-up should not match"
    );

    // The sample index is what a seek actually resolves against.
    let delta = u64::from(frames[0].header().samples_per_channel);
    assert_eq!(track.sample_at_time(delta * TARGET as u64), Some(TARGET));
}
