//! MP4 / M4A container support for AV3A elementary streams.
//!
//! AV3A in the ISO base media file format is unusually simple to demux: the
//! `stsd` entry is a bare `AudioSampleEntry` with no codec configuration box,
//! because every decoder parameter travels in band in each frame header. The
//! container only has to answer three questions — which track is AV3A, where
//! does each sample live, and when is it presented — so this module reads just
//! the `stbl` sub-tree rather than modelling the whole format.
//!
//! Two entry points cover the two ways callers reach a file:
//!
//! * [`Av3aTrack::read_from`] seeks over the top-level boxes and reads only
//!   `moov`, so indexing costs are proportional to the metadata rather than to
//!   the media.
//! * [`Av3aTrack::from_prefix`] indexes a partially downloaded file, and when
//!   the prefix is too short it reports exactly how many bytes are still
//!   required ([`Mp4Error::NeedMoreData`]). A byte-range fetcher can then ask
//!   for them in one more request instead of repeatedly widening a guess.

use std::io::{Read, Seek, SeekFrom};

pub use crate::error::Mp4Error;
use crate::header::{MAX_HEADER_BYTES, MAX_PAYLOAD_BYTES, parse_header_at};
use crate::stream::EncodedFrame;

/// The `stsd` sample entry format that identifies an AV3A track.
pub const AV3A_SAMPLE_ENTRY: [u8; 4] = *b"av3a";

/// Upper bound on a `moov` box that will be read into memory.
///
/// Real `moov` boxes are a few tens of kilobytes; the limit only exists so a
/// corrupt or hostile size field cannot request an arbitrary allocation.
const MAX_MOOV_BYTES: u64 = 64 << 20;

/// Upper bound on the sample index, about 24 hours of 1024-sample frames.
const MAX_SAMPLES: usize = 1 << 22;

/// Largest byte count a single sample may declare.
///
/// One sample is one access unit, so it cannot exceed one maximally sized
/// frame plus whatever padding a muxer added; the padding allowance is
/// generous because the value only bounds a read buffer.
const MAX_SAMPLE_BYTES: usize = 2 * (MAX_HEADER_BYTES + MAX_PAYLOAD_BYTES);

/// Bytes of a file prefix [`is_iso_bmff`] needs to decide.
pub const ISO_BMFF_SNIFF_BYTES: usize = 8;

/// Whether a file prefix looks like an ISO base media file rather than a raw
/// AV3A elementary stream.
///
/// The format requires a `ftyp` box first, so its type tag sits at byte 4. An
/// elementary stream starts with a 12-bit sync word of all ones, which cannot
/// appear there. A prefix shorter than [`ISO_BMFF_SNIFF_BYTES`] is reported as
/// not a container, because it is too short to hold even a box header.
pub fn is_iso_bmff(prefix: &[u8]) -> bool {
    prefix
        .get(4..ISO_BMFF_SNIFF_BYTES)
        .is_some_and(|kind| kind == b"ftyp")
}

/// One access unit: a single AV3A frame stored in `mdat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mp4Sample {
    /// Absolute byte offset of the sample within the file.
    pub offset: u64,
    /// Sample size in bytes.
    pub size: u32,
    /// Decode timestamp in media timescale units.
    pub timestamp: u64,
    /// Duration in media timescale units.
    pub duration: u32,
}

impl Mp4Sample {
    /// Byte offset one past the end of the sample.
    pub fn end(&self) -> Result<u64, Mp4Error> {
        self.offset
            .checked_add(u64::from(self.size))
            .ok_or(Mp4Error::ArithmeticOverflow)
    }
}

/// One `elst` entry, in media timescale units.
///
/// AV3A files written by the reference muxer carry an identity edit list, but
/// encoder delay compensation is expressed here when it is present, so the
/// entries are surfaced rather than silently ignored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mp4Edit {
    /// Duration of the edit in movie timescale units.
    pub segment_duration: u64,
    /// Start time within the media, or `-1` for an empty edit.
    pub media_time: i64,
    /// Playback rate; `1.0` for normal playback.
    pub media_rate: f64,
}

impl Mp4Edit {
    /// Whether the edit plays the media unmodified from its start.
    pub fn is_identity(&self) -> bool {
        self.media_time == 0 && (self.media_rate - 1.0).abs() < f64::EPSILON
    }
}

/// The AV3A track of an MP4/M4A file, with a fully resolved sample index.
#[derive(Debug, Clone)]
pub struct Av3aTrack {
    track_id: u32,
    timescale: u32,
    duration: u64,
    channels: u16,
    sample_rate: u32,
    samples: Vec<Mp4Sample>,
    edits: Vec<Mp4Edit>,
}

impl Av3aTrack {
    /// Index the AV3A track of a seekable MP4/M4A file.
    ///
    /// Only the `moov` box is read; the media data is seeked over, so the cost
    /// does not grow with the length of the audio.
    pub fn read_from<R: Read + Seek>(reader: &mut R) -> Result<Self, Mp4Error> {
        let moov = read_moov(reader)?;
        Self::from_moov_body(&moov)
    }

    /// Index the AV3A track from the leading bytes of an MP4/M4A file.
    ///
    /// When `bytes` stops short of the metadata the error is
    /// [`Mp4Error::NeedMoreData`], whose `needed` field is the exact prefix
    /// length that would let the next attempt succeed. Callers fetching over
    /// byte ranges should request that length rather than widening a guess.
    pub fn from_prefix(bytes: &[u8]) -> Result<Self, Mp4Error> {
        let available = bytes.len() as u64;
        let mut pos = 0_usize;
        while pos < bytes.len() {
            let rest = &bytes[pos..];
            let decoded = match decode_box_header(rest) {
                BoxHeaderResult::NeedMore(needed) => {
                    return Err(Mp4Error::NeedMoreData {
                        needed: offset_plus(pos, needed)?,
                        available,
                    });
                }
                BoxHeaderResult::Decoded(decoded) => decoded,
            };
            // A `size == 0` box runs to the end of the file, which a prefix
            // cannot bound. Treating it as "the rest of what we have" is only
            // useful for `moov`; any other box swallows the remainder and the
            // loop ends with `moov` missing.
            let total = match decoded.size {
                Some(size) => usize::try_from(size).map_err(|_| Mp4Error::ArithmeticOverflow)?,
                None => rest.len(),
            };
            if total < decoded.header_len {
                return Err(Mp4Error::InvalidBoxSize {
                    kind: decoded.kind,
                    size: total as u64,
                });
            }
            if decoded.kind == *b"moov" {
                if total > rest.len() {
                    return Err(Mp4Error::NeedMoreData {
                        needed: offset_plus(pos, total)?,
                        available,
                    });
                }
                return Self::from_moov_body(&rest[decoded.header_len..total]);
            }
            if total > rest.len() {
                // The box is only partly downloaded. Report how far the caller
                // has to read to see the *next* box header, so a file with
                // `mdat` before `moov` converges in one more request.
                return Err(Mp4Error::NeedMoreData {
                    needed: offset_plus(
                        pos,
                        total.checked_add(8).ok_or(Mp4Error::ArithmeticOverflow)?,
                    )?,
                    available,
                });
            }
            pos = pos.checked_add(total).ok_or(Mp4Error::ArithmeticOverflow)?;
        }
        Err(Mp4Error::MissingBox { kind: *b"moov" })
    }

    /// `tkhd` track identifier.
    pub fn track_id(&self) -> u32 {
        self.track_id
    }

    /// Media timescale, in units per second.
    pub fn timescale(&self) -> u32 {
        self.timescale
    }

    /// Track duration in media timescale units.
    pub fn duration(&self) -> u64 {
        self.duration
    }

    /// Track duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        if self.timescale == 0 {
            return 0.0;
        }
        self.duration as f64 / f64::from(self.timescale)
    }

    /// Channel count declared by the `stsd` entry.
    ///
    /// The in-band frame header is authoritative: `AudioSampleEntry` predates
    /// immersive layouts and commonly reports 2 for a 7.1.4 track. Use this
    /// only as a hint before the first frame is decoded.
    pub fn declared_channels(&self) -> u16 {
        self.channels
    }

    /// Sample rate declared by the `stsd` entry.
    ///
    /// Stored as 16.16 fixed point in the container, so rates of 65536 Hz and
    /// above cannot be represented there. As with the channel count, the frame
    /// header is authoritative.
    pub fn declared_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The sample index, in decode order.
    pub fn samples(&self) -> &[Mp4Sample] {
        &self.samples
    }

    /// The `elst` entries, empty when the file has no edit list.
    pub fn edits(&self) -> &[Mp4Edit] {
        &self.edits
    }

    /// The byte range spanned by the media data, or `None` for an empty track.
    ///
    /// Useful for a byte-range fetcher that wants to know which part of the
    /// file is audio before it starts downloading.
    pub fn data_range(&self) -> Result<Option<(u64, u64)>, Mp4Error> {
        let Some(first) = self.samples.first() else {
            return Ok(None);
        };
        let mut start = first.offset;
        let mut end = first.end()?;
        for sample in &self.samples[1..] {
            start = start.min(sample.offset);
            end = end.max(sample.end()?);
        }
        Ok(Some((start, end)))
    }

    /// Index of the last sample whose timestamp is at or before `timestamp`.
    ///
    /// Every AV3A sample is a sync sample, so the result is directly seekable;
    /// the caller still has to feed [`crate::BuiltinDecoder::warmup_frames`]
    /// preceding frames to refill the synthesis overlap.
    pub fn sample_at_time(&self, timestamp: u64) -> Option<usize> {
        if self.samples.first()?.timestamp > timestamp {
            return None;
        }
        let index = self
            .samples
            .partition_point(|sample| sample.timestamp <= timestamp);
        Some(index.saturating_sub(1))
    }

    fn from_moov_body(moov: &[u8]) -> Result<Self, Mp4Error> {
        let mut iter = BoxIter::new(moov, "moov");
        while let Some((kind, body)) = iter.next_box()? {
            if kind != *b"trak" {
                continue;
            }
            if let Some(track) = Self::from_trak_body(body)? {
                return Ok(track);
            }
        }
        Err(Mp4Error::NoAv3aTrack)
    }

    /// Parse one `trak`, returning `None` when it is not an AV3A track.
    ///
    /// Non-audio tracks are skipped rather than rejected, so a file that
    /// muxes AV3A alongside video still indexes. Once the `av3a` sample entry
    /// is found, anything missing is an error.
    fn from_trak_body(trak: &[u8]) -> Result<Option<Self>, Mp4Error> {
        let Some(mdia) = find_child(trak, b"mdia", "trak")? else {
            return Ok(None);
        };
        let Some(minf) = find_child(mdia, b"minf", "mdia")? else {
            return Ok(None);
        };
        let Some(stbl) = find_child(minf, b"stbl", "minf")? else {
            return Ok(None);
        };
        let Some(stsd) = find_child(stbl, b"stsd", "stbl")? else {
            return Ok(None);
        };
        let Some(entry) = parse_stsd(stsd)? else {
            return Ok(None);
        };

        let mdhd = require_child(mdia, b"mdhd", "mdia")?;
        let (timescale, duration) = parse_mdhd(mdhd)?;
        let track_id = match find_child(trak, b"tkhd", "trak")? {
            Some(tkhd) => parse_tkhd(tkhd)?,
            None => 0,
        };
        let edits = match find_child(trak, b"edts", "trak")? {
            Some(edts) => match find_child(edts, b"elst", "edts")? {
                Some(elst) => parse_elst(elst)?,
                None => Vec::new(),
            },
            None => Vec::new(),
        };

        let tables = SampleTables::from_stbl(stbl)?;
        let samples = tables.build()?;

        Ok(Some(Self {
            track_id,
            timescale,
            duration,
            channels: entry.channels,
            sample_rate: entry.sample_rate,
            samples,
            edits,
        }))
    }
}

/// Reads AV3A frames out of an MP4/M4A container in decode order.
///
/// The reader keeps one scratch buffer and reuses it for every sample, so
/// iterating a track allocates once per emitted frame rather than once per
/// read.
#[derive(Debug)]
pub struct Mp4FrameReader<R> {
    reader: R,
    track: Av3aTrack,
    position: usize,
    /// Where `reader` currently is, so sequential reads skip the seek.
    cursor: Option<u64>,
    buffer: Vec<u8>,
}

impl<R: Read + Seek> Mp4FrameReader<R> {
    /// Index the container and prepare to read from its first sample.
    pub fn open(mut reader: R) -> Result<Self, Mp4Error> {
        let track = Av3aTrack::read_from(&mut reader)?;
        Ok(Self {
            reader,
            track,
            position: 0,
            cursor: None,
            buffer: Vec::new(),
        })
    }

    /// Pair an already indexed track with a reader over the same file.
    pub fn with_track(reader: R, track: Av3aTrack) -> Self {
        Self {
            reader,
            track,
            position: 0,
            cursor: None,
            buffer: Vec::new(),
        }
    }

    /// The indexed track.
    pub fn track(&self) -> &Av3aTrack {
        &self.track
    }

    /// Index of the sample [`Self::next_frame`] will return.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Move to `index`, which may be one past the last sample.
    ///
    /// Warm-up is left to the caller: start this at
    /// [`crate::BuiltinDecoder::warmup_frames`] samples before the target and
    /// discard that many decoded frames.
    pub fn seek_to_sample(&mut self, index: usize) -> Result<(), Mp4Error> {
        if index > self.track.samples.len() {
            return Err(Mp4Error::SampleOutOfRange {
                index,
                count: self.track.samples.len(),
            });
        }
        self.position = index;
        Ok(())
    }

    /// Read the next frame, or `None` at the end of the track.
    pub fn next_frame(&mut self) -> Result<Option<EncodedFrame>, Mp4Error> {
        let Some(sample) = self.track.samples.get(self.position).copied() else {
            return Ok(None);
        };
        let index = self.position;
        let size = usize::try_from(sample.size).map_err(|_| Mp4Error::ArithmeticOverflow)?;
        if size > MAX_SAMPLE_BYTES {
            return Err(Mp4Error::SampleTooLarge {
                index,
                size,
                limit: MAX_SAMPLE_BYTES,
            });
        }

        if self.cursor != Some(sample.offset) {
            self.reader.seek(SeekFrom::Start(sample.offset))?;
        }
        // Clear first so `resize` cannot copy stale bytes when it grows.
        self.buffer.clear();
        self.buffer.resize(size, 0);
        // A short read leaves the cursor unknown, so drop it before failing.
        self.cursor = None;
        self.reader.read_exact(&mut self.buffer)?;
        self.cursor = Some(sample.end()?);

        let header = parse_header_at(&self.buffer, 0)?;
        if header.frame_len > size {
            return Err(Mp4Error::FrameExceedsSample {
                index,
                frame_len: header.frame_len,
                sample_size: size,
            });
        }
        // Trailing bytes are muxer padding: one sample is one access unit.
        let bytes = self.buffer[..header.frame_len].to_vec();
        self.position += 1;
        Ok(Some(EncodedFrame::new(header, bytes)))
    }

    /// Recover the underlying reader.
    pub fn into_inner(self) -> R {
        self.reader
    }
}

/// Read the `moov` box of a seekable file without reading the media data.
fn read_moov<R: Read + Seek>(reader: &mut R) -> Result<Vec<u8>, Mp4Error> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;

    let mut pos = 0_u64;
    let mut header = [0_u8; 16];
    while pos < file_len {
        let available = file_len - pos;
        let want = usize::try_from(available.min(16)).expect("capped at 16");
        reader.read_exact(&mut header[..want])?;
        let decoded = match decode_box_header(&header[..want]) {
            // The trailing bytes are too short to be a box; treat them the
            // same way a truncated file would be treated.
            BoxHeaderResult::NeedMore(_) => break,
            BoxHeaderResult::Decoded(decoded) => decoded,
        };
        let total = decoded.size.unwrap_or(available);
        if total < decoded.header_len as u64 {
            return Err(Mp4Error::InvalidBoxSize {
                kind: decoded.kind,
                size: total,
            });
        }
        if total > available {
            return Err(Mp4Error::Truncated {
                context: "file",
                needed: total,
                available,
            });
        }

        let body_start = pos + decoded.header_len as u64;
        if decoded.kind == *b"moov" {
            let body_len = total - decoded.header_len as u64;
            if body_len > MAX_MOOV_BYTES {
                return Err(Mp4Error::BoxTooLarge {
                    kind: decoded.kind,
                    size: body_len,
                    limit: MAX_MOOV_BYTES,
                });
            }
            let mut body = vec![0_u8; body_len as usize];
            reader.seek(SeekFrom::Start(body_start))?;
            reader.read_exact(&mut body)?;
            return Ok(body);
        }

        pos += total;
        reader.seek(SeekFrom::Start(pos))?;
    }
    Err(Mp4Error::MissingBox { kind: *b"moov" })
}

/// The `stbl` tables needed to place every sample, before they are expanded.
struct SampleTables {
    chunk_offsets: Vec<u64>,
    runs: Vec<StscRun>,
    sizes: SampleSizes,
    timings: Vec<SttsEntry>,
}

#[derive(Debug, Clone, Copy)]
struct StscRun {
    /// One-based index of the first chunk the run covers.
    first_chunk: u32,
    samples_per_chunk: u32,
}

#[derive(Debug, Clone, Copy)]
struct SttsEntry {
    count: u32,
    delta: u32,
}

/// `stsz` stores either one size for every sample or a size per sample.
enum SampleSizes {
    Constant { size: u32, count: u32 },
    PerSample(Vec<u32>),
}

impl SampleSizes {
    fn count(&self) -> usize {
        match self {
            Self::Constant { count, .. } => *count as usize,
            Self::PerSample(sizes) => sizes.len(),
        }
    }

    fn get(&self, index: usize) -> Option<u32> {
        match self {
            Self::Constant { size, count } => (index < *count as usize).then_some(*size),
            Self::PerSample(sizes) => sizes.get(index).copied(),
        }
    }
}

impl SampleTables {
    fn from_stbl(stbl: &[u8]) -> Result<Self, Mp4Error> {
        let chunk_offsets = match find_child(stbl, b"stco", "stbl")? {
            Some(stco) => parse_stco(stco)?,
            None => parse_co64(require_child(stbl, b"co64", "stbl")?)?,
        };
        Ok(Self {
            chunk_offsets,
            runs: parse_stsc(require_child(stbl, b"stsc", "stbl")?)?,
            sizes: parse_stsz(require_child(stbl, b"stsz", "stbl")?)?,
            timings: parse_stts(require_child(stbl, b"stts", "stbl")?)?,
        })
    }

    /// Expand the tables into one entry per sample.
    fn build(&self) -> Result<Vec<Mp4Sample>, Mp4Error> {
        let total = self.sizes.count();
        if total > MAX_SAMPLES {
            return Err(Mp4Error::TooManySamples {
                count: total,
                limit: MAX_SAMPLES,
            });
        }
        if total == 0 {
            return Ok(Vec::new());
        }
        if self.runs.is_empty() {
            return Err(Mp4Error::InvalidSampleTable(
                "stsc has no sample-to-chunk runs",
            ));
        }

        let mut samples = Vec::with_capacity(total);
        let mut timeline = SttsTimeline::new(&self.timings);

        for (position, run) in self.runs.iter().enumerate() {
            // `first_chunk` is one-based and must not run backwards, or two
            // runs would claim the same chunk.
            let first = run
                .first_chunk
                .checked_sub(1)
                .ok_or(Mp4Error::InvalidSampleTable("stsc first_chunk is zero"))?
                as usize;
            let end = match self.runs.get(position + 1) {
                Some(next) => {
                    let next_first = next
                        .first_chunk
                        .checked_sub(1)
                        .ok_or(Mp4Error::InvalidSampleTable("stsc first_chunk is zero"))?
                        as usize;
                    if next_first < first {
                        return Err(Mp4Error::InvalidSampleTable(
                            "stsc first_chunk decreases between runs",
                        ));
                    }
                    next_first.min(self.chunk_offsets.len())
                }
                None => self.chunk_offsets.len(),
            };
            if first > self.chunk_offsets.len() {
                return Err(Mp4Error::InvalidSampleTable(
                    "stsc references a chunk beyond stco/co64",
                ));
            }

            for chunk in first..end {
                let mut offset = self.chunk_offsets[chunk];
                for _ in 0..run.samples_per_chunk {
                    // Chunks may describe more samples than `stsz` declares;
                    // `stsz` is authoritative, so stop rather than invent one.
                    if samples.len() == total {
                        return Ok(samples);
                    }
                    let size =
                        self.sizes
                            .get(samples.len())
                            .ok_or(Mp4Error::InvalidSampleTable(
                                "stsz is shorter than declared",
                            ))?;
                    let (timestamp, duration) = timeline.next()?;
                    samples.push(Mp4Sample {
                        offset,
                        size,
                        timestamp,
                        duration,
                    });
                    offset = offset
                        .checked_add(u64::from(size))
                        .ok_or(Mp4Error::ArithmeticOverflow)?;
                }
            }
        }

        if samples.len() != total {
            // Silently returning a short index would truncate playback, so
            // report the inconsistency instead.
            return Err(Mp4Error::InconsistentIndex {
                declared: total,
                indexed: samples.len(),
            });
        }
        Ok(samples)
    }
}

/// Walks `stts` run-length entries to produce per-sample timestamps.
struct SttsTimeline<'a> {
    entries: &'a [SttsEntry],
    entry: usize,
    used: u32,
    time: u64,
}

impl<'a> SttsTimeline<'a> {
    fn new(entries: &'a [SttsEntry]) -> Self {
        Self {
            entries,
            entry: 0,
            used: 0,
            time: 0,
        }
    }

    fn next(&mut self) -> Result<(u64, u32), Mp4Error> {
        while let Some(entry) = self.entries.get(self.entry) {
            if self.used < entry.count {
                self.used += 1;
                let timestamp = self.time;
                self.time = self
                    .time
                    .checked_add(u64::from(entry.delta))
                    .ok_or(Mp4Error::ArithmeticOverflow)?;
                return Ok((timestamp, entry.delta));
            }
            self.entry += 1;
            self.used = 0;
        }
        Err(Mp4Error::InvalidSampleTable(
            "stts covers fewer samples than stsz declares",
        ))
    }
}

/// The parts of an `AudioSampleEntry` worth surfacing.
struct AudioSampleEntry {
    channels: u16,
    sample_rate: u32,
}

/// Find an AV3A sample entry in `stsd`.
fn parse_stsd(stsd: &[u8]) -> Result<Option<AudioSampleEntry>, Mp4Error> {
    let mut bytes = Bytes::new(stsd, "stsd");
    bytes.full_box_header()?;
    // The entry count is redundant here: the box body already bounds the
    // entries, and a wrong count must not stop a valid entry from being found.
    bytes.skip(4)?;

    let mut iter = BoxIter::new(bytes.rest(), "stsd");
    while let Some((kind, body)) = iter.next_box()? {
        if kind == AV3A_SAMPLE_ENTRY {
            return Ok(Some(parse_audio_sample_entry(body)?));
        }
    }
    Ok(None)
}

fn parse_audio_sample_entry(body: &[u8]) -> Result<AudioSampleEntry, Mp4Error> {
    let mut bytes = Bytes::new(body, "av3a");
    bytes.skip(6)?; // reserved
    bytes.skip(2)?; // data_reference_index
    bytes.skip(2)?; // SoundDescription version
    bytes.skip(2)?; // revision level
    bytes.skip(4)?; // vendor
    let channels = bytes.u16()?;
    bytes.skip(2)?; // sample size
    bytes.skip(2)?; // compression id
    bytes.skip(2)?; // packet size
    // 16.16 fixed point, so the integer part is the rate.
    let sample_rate = bytes.u32()? >> 16;
    Ok(AudioSampleEntry {
        channels,
        sample_rate,
    })
}

fn parse_tkhd(tkhd: &[u8]) -> Result<u32, Mp4Error> {
    let mut bytes = Bytes::new(tkhd, "tkhd");
    let (version, _flags) = bytes.full_box_header()?;
    match version {
        0 => bytes.skip(8)?,  // creation_time, modification_time
        1 => bytes.skip(16)?, // 64-bit creation_time, modification_time
        _ => {
            return Err(Mp4Error::UnsupportedVersion {
                kind: *b"tkhd",
                version,
            });
        }
    }
    bytes.u32()
}

fn parse_mdhd(mdhd: &[u8]) -> Result<(u32, u64), Mp4Error> {
    let mut bytes = Bytes::new(mdhd, "mdhd");
    let (version, _flags) = bytes.full_box_header()?;
    match version {
        0 => {
            bytes.skip(8)?; // creation_time, modification_time
            let timescale = bytes.u32()?;
            let duration = u64::from(bytes.u32()?);
            Ok((timescale, duration))
        }
        1 => {
            bytes.skip(16)?; // 64-bit creation_time, modification_time
            let timescale = bytes.u32()?;
            let duration = bytes.u64()?;
            Ok((timescale, duration))
        }
        _ => Err(Mp4Error::UnsupportedVersion {
            kind: *b"mdhd",
            version,
        }),
    }
}

fn parse_elst(elst: &[u8]) -> Result<Vec<Mp4Edit>, Mp4Error> {
    let mut bytes = Bytes::new(elst, "elst");
    let (version, _flags) = bytes.full_box_header()?;
    let entry_size = match version {
        0 => 12,
        1 => 20,
        _ => {
            return Err(Mp4Error::UnsupportedVersion {
                kind: *b"elst",
                version,
            });
        }
    };
    let count = bytes.bounded_entry_count("elst", entry_size)?;

    let mut edits = Vec::with_capacity(count);
    for _ in 0..count {
        let (segment_duration, media_time) = if version == 1 {
            (bytes.u64()?, bytes.i64()?)
        } else {
            (u64::from(bytes.u32()?), i64::from(bytes.i32()?))
        };
        let integer = bytes.i16()?;
        let fraction = bytes.i16()?;
        edits.push(Mp4Edit {
            segment_duration,
            media_time,
            media_rate: f64::from(integer) + f64::from(fraction) / 65_536.0,
        });
    }
    Ok(edits)
}

fn parse_stco(stco: &[u8]) -> Result<Vec<u64>, Mp4Error> {
    let mut bytes = Bytes::new(stco, "stco");
    bytes.full_box_header()?;
    let count = bytes.bounded_entry_count("stco", 4)?;
    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        offsets.push(u64::from(bytes.u32()?));
    }
    Ok(offsets)
}

fn parse_co64(co64: &[u8]) -> Result<Vec<u64>, Mp4Error> {
    let mut bytes = Bytes::new(co64, "co64");
    bytes.full_box_header()?;
    let count = bytes.bounded_entry_count("co64", 8)?;
    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        offsets.push(bytes.u64()?);
    }
    Ok(offsets)
}

fn parse_stsc(stsc: &[u8]) -> Result<Vec<StscRun>, Mp4Error> {
    let mut bytes = Bytes::new(stsc, "stsc");
    bytes.full_box_header()?;
    let count = bytes.bounded_entry_count("stsc", 12)?;
    let mut runs = Vec::with_capacity(count);
    for _ in 0..count {
        let first_chunk = bytes.u32()?;
        let samples_per_chunk = bytes.u32()?;
        bytes.skip(4)?; // sample_description_index
        runs.push(StscRun {
            first_chunk,
            samples_per_chunk,
        });
    }
    Ok(runs)
}

fn parse_stsz(stsz: &[u8]) -> Result<SampleSizes, Mp4Error> {
    let mut bytes = Bytes::new(stsz, "stsz");
    bytes.full_box_header()?;
    let sample_size = bytes.u32()?;
    let sample_count = bytes.u32()?;
    if sample_size != 0 {
        return Ok(SampleSizes::Constant {
            size: sample_size,
            count: sample_count,
        });
    }
    let count = bytes.checked_entry_count("stsz", sample_count, 4)?;
    let mut sizes = Vec::with_capacity(count);
    for _ in 0..count {
        sizes.push(bytes.u32()?);
    }
    Ok(SampleSizes::PerSample(sizes))
}

fn parse_stts(stts: &[u8]) -> Result<Vec<SttsEntry>, Mp4Error> {
    let mut bytes = Bytes::new(stts, "stts");
    bytes.full_box_header()?;
    let count = bytes.bounded_entry_count("stts", 8)?;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let sample_count = bytes.u32()?;
        let sample_delta = bytes.u32()?;
        entries.push(SttsEntry {
            count: sample_count,
            delta: sample_delta,
        });
    }
    Ok(entries)
}

/// A decoded box header.
struct BoxHeader {
    kind: [u8; 4],
    header_len: usize,
    /// Total box size including the header, or `None` for "to end of file".
    size: Option<u64>,
}

enum BoxHeaderResult {
    /// This many bytes from the start of the box are needed to decode it.
    NeedMore(usize),
    Decoded(BoxHeader),
}

fn decode_box_header(bytes: &[u8]) -> BoxHeaderResult {
    if bytes.len() < 8 {
        return BoxHeaderResult::NeedMore(8);
    }
    let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let kind = [bytes[4], bytes[5], bytes[6], bytes[7]];
    match size {
        0 => BoxHeaderResult::Decoded(BoxHeader {
            kind,
            header_len: 8,
            size: None,
        }),
        1 => {
            if bytes.len() < 16 {
                return BoxHeaderResult::NeedMore(16);
            }
            let large = u64::from_be_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ]);
            BoxHeaderResult::Decoded(BoxHeader {
                kind,
                header_len: 16,
                size: Some(large),
            })
        }
        _ => BoxHeaderResult::Decoded(BoxHeader {
            kind,
            header_len: 8,
            size: Some(u64::from(size)),
        }),
    }
}

/// Iterates the child boxes of a fully available box body.
struct BoxIter<'a> {
    buf: &'a [u8],
    pos: usize,
    ctx: &'static str,
}

/// A child box: its four-character type and its body.
type ChildBox<'a> = ([u8; 4], &'a [u8]);

impl<'a> BoxIter<'a> {
    fn new(buf: &'a [u8], ctx: &'static str) -> Self {
        Self { buf, pos: 0, ctx }
    }

    fn next_box(&mut self) -> Result<Option<ChildBox<'a>>, Mp4Error> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let rest = &self.buf[self.pos..];
        let decoded = match decode_box_header(rest) {
            BoxHeaderResult::NeedMore(needed) => {
                return Err(Mp4Error::Truncated {
                    context: self.ctx,
                    needed: needed as u64,
                    available: rest.len() as u64,
                });
            }
            BoxHeaderResult::Decoded(decoded) => decoded,
        };
        let total = match decoded.size {
            Some(size) => usize::try_from(size).map_err(|_| Mp4Error::ArithmeticOverflow)?,
            None => rest.len(),
        };
        if total < decoded.header_len {
            return Err(Mp4Error::InvalidBoxSize {
                kind: decoded.kind,
                size: total as u64,
            });
        }
        if total > rest.len() {
            return Err(Mp4Error::Truncated {
                context: self.ctx,
                needed: total as u64,
                available: rest.len() as u64,
            });
        }
        self.pos += total;
        Ok(Some((decoded.kind, &rest[decoded.header_len..total])))
    }
}

fn find_child<'a>(
    body: &'a [u8],
    kind: &[u8; 4],
    ctx: &'static str,
) -> Result<Option<&'a [u8]>, Mp4Error> {
    let mut iter = BoxIter::new(body, ctx);
    while let Some((found, child)) = iter.next_box()? {
        if found == *kind {
            return Ok(Some(child));
        }
    }
    Ok(None)
}

fn require_child<'a>(
    body: &'a [u8],
    kind: &[u8; 4],
    ctx: &'static str,
) -> Result<&'a [u8], Mp4Error> {
    find_child(body, kind, ctx)?.ok_or(Mp4Error::MissingBox { kind: *kind })
}

fn offset_plus(offset: usize, extra: usize) -> Result<u64, Mp4Error> {
    offset
        .checked_add(extra)
        .map(|total| total as u64)
        .ok_or(Mp4Error::ArithmeticOverflow)
}

/// A bounds-checked big-endian cursor over one box body.
struct Bytes<'a> {
    buf: &'a [u8],
    pos: usize,
    ctx: &'static str,
}

impl<'a> Bytes<'a> {
    fn new(buf: &'a [u8], ctx: &'static str) -> Self {
        Self { buf, pos: 0, ctx }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn rest(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], Mp4Error> {
        if self.remaining() < count {
            return Err(Mp4Error::Truncated {
                context: self.ctx,
                needed: (self.pos + count) as u64,
                available: self.buf.len() as u64,
            });
        }
        let slice = &self.buf[self.pos..self.pos + count];
        self.pos += count;
        Ok(slice)
    }

    fn skip(&mut self, count: usize) -> Result<(), Mp4Error> {
        self.take(count).map(|_| ())
    }

    fn u16(&mut self) -> Result<u16, Mp4Error> {
        let bytes = self.take(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn i16(&mut self) -> Result<i16, Mp4Error> {
        let bytes = self.take(2)?;
        Ok(i16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, Mp4Error> {
        let bytes = self.take(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn i32(&mut self) -> Result<i32, Mp4Error> {
        let bytes = self.take(4)?;
        Ok(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, Mp4Error> {
        let bytes = self.take(8)?;
        Ok(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn i64(&mut self) -> Result<i64, Mp4Error> {
        let bytes = self.take(8)?;
        Ok(i64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Read a full-box version byte and 24-bit flags.
    fn full_box_header(&mut self) -> Result<(u8, u32), Mp4Error> {
        let bytes = self.take(4)?;
        let flags = u32::from_be_bytes([0, bytes[1], bytes[2], bytes[3]]);
        Ok((bytes[0], flags))
    }

    /// Read an entry count and check the body can actually hold that many.
    ///
    /// Validating before reserving keeps a corrupt count from turning into a
    /// multi-gigabyte allocation.
    fn bounded_entry_count(
        &mut self,
        ctx: &'static str,
        entry_size: usize,
    ) -> Result<usize, Mp4Error> {
        let declared = self.u32()?;
        self.checked_entry_count(ctx, declared, entry_size)
    }

    fn checked_entry_count(
        &mut self,
        ctx: &'static str,
        declared: u32,
        entry_size: usize,
    ) -> Result<usize, Mp4Error> {
        let count = declared as usize;
        let needed = count
            .checked_mul(entry_size)
            .ok_or(Mp4Error::ArithmeticOverflow)?;
        if needed > self.remaining() {
            return Err(Mp4Error::Truncated {
                context: ctx,
                needed: (self.pos + needed) as u64,
                available: self.buf.len() as u64,
            });
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;
    use crate::crc16;
    use crate::header::ChannelConfig;
    use std::io::Cursor;

    fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + body.len());
        let size = u32::try_from(8 + body.len()).expect("test box fits in 32 bits");
        out.extend_from_slice(&size.to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    fn concat(parts: &[&[u8]]) -> Vec<u8> {
        parts.concat()
    }

    fn full_box(version: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![version, 0, 0, 0];
        out.extend_from_slice(payload);
        out
    }

    fn av3a_entry(channels: u16, sample_rate: u32) -> Vec<u8> {
        let mut body = vec![0_u8; 6]; // reserved
        body.extend_from_slice(&1_u16.to_be_bytes()); // data_reference_index
        body.extend_from_slice(&[0; 8]); // version, revision, vendor
        body.extend_from_slice(&channels.to_be_bytes());
        body.extend_from_slice(&16_u16.to_be_bytes()); // sample size
        body.extend_from_slice(&[0; 4]); // compression id, packet size
        body.extend_from_slice(&(sample_rate << 16).to_be_bytes());
        boxed(b"av3a", &body)
    }

    fn stsd(entry: &[u8]) -> Vec<u8> {
        let mut body = full_box(0, &1_u32.to_be_bytes());
        body.extend_from_slice(entry);
        boxed(b"stsd", &body)
    }

    fn stts(entries: &[(u32, u32)]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&u32::try_from(entries.len()).unwrap().to_be_bytes());
        for (count, delta) in entries {
            payload.extend_from_slice(&count.to_be_bytes());
            payload.extend_from_slice(&delta.to_be_bytes());
        }
        boxed(b"stts", &full_box(0, &payload))
    }

    fn stsc(runs: &[(u32, u32)]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&u32::try_from(runs.len()).unwrap().to_be_bytes());
        for (first_chunk, samples_per_chunk) in runs {
            payload.extend_from_slice(&first_chunk.to_be_bytes());
            payload.extend_from_slice(&samples_per_chunk.to_be_bytes());
            payload.extend_from_slice(&1_u32.to_be_bytes());
        }
        boxed(b"stsc", &full_box(0, &payload))
    }

    fn stsz_constant(size: u32, count: u32) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&size.to_be_bytes());
        payload.extend_from_slice(&count.to_be_bytes());
        boxed(b"stsz", &full_box(0, &payload))
    }

    fn stsz_per_sample(sizes: &[u32]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&0_u32.to_be_bytes());
        payload.extend_from_slice(&u32::try_from(sizes.len()).unwrap().to_be_bytes());
        for size in sizes {
            payload.extend_from_slice(&size.to_be_bytes());
        }
        boxed(b"stsz", &full_box(0, &payload))
    }

    fn stco(offsets: &[u32]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&u32::try_from(offsets.len()).unwrap().to_be_bytes());
        for offset in offsets {
            payload.extend_from_slice(&offset.to_be_bytes());
        }
        boxed(b"stco", &full_box(0, &payload))
    }

    fn co64(offsets: &[u64]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&u32::try_from(offsets.len()).unwrap().to_be_bytes());
        for offset in offsets {
            payload.extend_from_slice(&offset.to_be_bytes());
        }
        boxed(b"co64", &full_box(0, &payload))
    }

    fn mdhd(timescale: u32, duration: u32) -> Vec<u8> {
        let mut payload = vec![0_u8; 8]; // creation, modification
        payload.extend_from_slice(&timescale.to_be_bytes());
        payload.extend_from_slice(&duration.to_be_bytes());
        payload.extend_from_slice(&[0; 4]); // language, pre_defined
        boxed(b"mdhd", &full_box(0, &payload))
    }

    fn tkhd(track_id: u32) -> Vec<u8> {
        let mut payload = vec![0_u8; 8]; // creation, modification
        payload.extend_from_slice(&track_id.to_be_bytes());
        payload.extend_from_slice(&[0; 68]);
        boxed(b"tkhd", &full_box(0, &payload))
    }

    fn elst(entries: &[(u32, i32)]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&u32::try_from(entries.len()).unwrap().to_be_bytes());
        for (segment_duration, media_time) in entries {
            payload.extend_from_slice(&segment_duration.to_be_bytes());
            payload.extend_from_slice(&media_time.to_be_bytes());
            payload.extend_from_slice(&1_i16.to_be_bytes());
            payload.extend_from_slice(&0_i16.to_be_bytes());
        }
        boxed(b"edts", &boxed(b"elst", &full_box(0, &payload)))
    }

    /// Assemble `ftyp + moov{trak{...}} + mdat` around the given tables.
    fn build_file(stbl_children: &[&[u8]], extra_trak: &[&[u8]], mdat: &[u8]) -> Vec<u8> {
        let stbl = boxed(b"stbl", &concat(stbl_children));
        let minf = boxed(b"minf", &stbl);
        let mdia = boxed(b"mdia", &concat(&[&mdhd(44_100, 14_336), &minf]));
        let mut trak_children = vec![tkhd(1)];
        for extra in extra_trak {
            trak_children.push(extra.to_vec());
        }
        trak_children.push(mdia);
        let trak = boxed(
            b"trak",
            &concat(&trak_children.iter().map(Vec::as_slice).collect::<Vec<_>>()),
        );
        let moov = boxed(b"moov", &trak);
        concat(&[
            &boxed(b"ftyp", b"isom\0\0\x02\0isomiso2"),
            &moov,
            &boxed(b"mdat", mdat),
        ])
    }

    fn standard_file() -> Vec<u8> {
        // 5 chunks: three of 3 samples, two of 2 samples => 13 samples.
        build_file(
            &[
                &stsd(&av3a_entry(2, 44_100)),
                &stts(&[(13, 1_024)]),
                &stsc(&[(1, 3), (4, 2)]),
                &stsz_constant(100, 13),
                &stco(&[1_000, 1_300, 1_600, 1_900, 2_100]),
            ],
            &[],
            &[],
        )
    }

    #[test]
    fn expands_constant_size_samples_across_chunk_runs() {
        let track = Av3aTrack::from_prefix(&standard_file()).expect("valid file");
        assert_eq!(track.samples().len(), 13);
        assert_eq!(track.track_id(), 1);
        assert_eq!(track.timescale(), 44_100);
        assert_eq!(track.declared_channels(), 2);
        assert_eq!(track.declared_sample_rate(), 44_100);

        // First chunk run: chunks 1..=3 hold three 100-byte samples each.
        assert_eq!(track.samples()[0].offset, 1_000);
        assert_eq!(track.samples()[1].offset, 1_100);
        assert_eq!(track.samples()[2].offset, 1_200);
        assert_eq!(track.samples()[3].offset, 1_300);
        // Second run: chunks 4..=5 hold two samples each.
        assert_eq!(track.samples()[9].offset, 1_900);
        assert_eq!(track.samples()[11].offset, 2_100);
        assert_eq!(track.samples()[12].offset, 2_200);
        assert!(track.samples().iter().all(|sample| sample.size == 100));
    }

    #[test]
    fn derives_timestamps_from_stts() {
        let track = Av3aTrack::from_prefix(&standard_file()).expect("valid file");
        for (index, sample) in track.samples().iter().enumerate() {
            assert_eq!(sample.timestamp, index as u64 * 1_024);
            assert_eq!(sample.duration, 1_024);
        }
        assert_eq!(track.sample_at_time(0), Some(0));
        assert_eq!(track.sample_at_time(1_023), Some(0));
        assert_eq!(track.sample_at_time(1_024), Some(1));
        assert_eq!(track.sample_at_time(u64::MAX), Some(12));
    }

    #[test]
    fn accepts_per_sample_sizes_and_64_bit_offsets() {
        let file = build_file(
            &[
                &stsd(&av3a_entry(12, 44_100)),
                &stts(&[(2, 1_024), (1, 512)]),
                &stsc(&[(1, 3)]),
                &stsz_per_sample(&[10, 20, 30]),
                &co64(&[1 << 33]),
            ],
            &[],
            &[],
        );
        let track = Av3aTrack::from_prefix(&file).expect("valid file");
        let samples = track.samples();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].offset, 1 << 33);
        assert_eq!(samples[1].offset, (1 << 33) + 10);
        assert_eq!(samples[2].offset, (1 << 33) + 30);
        assert_eq!(samples[2].size, 30);
        assert_eq!(samples[2].timestamp, 2_048);
        assert_eq!(samples[2].duration, 512);
        assert_eq!(track.data_range().unwrap(), Some((1 << 33, (1 << 33) + 60)));
    }

    #[test]
    fn reads_the_edit_list() {
        let file = build_file(
            &[
                &stsd(&av3a_entry(2, 48_000)),
                &stts(&[(1, 1_024)]),
                &stsc(&[(1, 1)]),
                &stsz_constant(64, 1),
                &stco(&[900]),
            ],
            &[&elst(&[(14_336, 0)])],
            &[],
        );
        let track = Av3aTrack::from_prefix(&file).expect("valid file");
        assert_eq!(track.edits().len(), 1);
        assert!(track.edits()[0].is_identity());
        assert_eq!(track.edits()[0].segment_duration, 14_336);
    }

    #[test]
    fn reports_the_prefix_length_needed_when_moov_follows_mdat() {
        // Rebuild the standard file with `mdat` first, then truncate inside it.
        let full = standard_file();
        let ftyp_len = 8 + 16;
        let moov = {
            let mut iter = BoxIter::new(&full[ftyp_len..], "file");
            let (kind, body) = iter.next_box().unwrap().unwrap();
            assert_eq!(kind, *b"moov");
            boxed(b"moov", body)
        };
        let mdat = boxed(b"mdat", &vec![0_u8; 4_096]);
        let file = concat(&[&full[..ftyp_len], &mdat, &moov]);

        let prefix = &file[..ftyp_len + 100];
        let error = Av3aTrack::from_prefix(prefix).expect_err("moov is not in the prefix");
        let Mp4Error::NeedMoreData { needed, available } = error else {
            panic!("expected NeedMoreData, got {error:?}");
        };
        assert_eq!(available, prefix.len() as u64);
        // Past `mdat` plus the next box header.
        assert_eq!(needed, (ftyp_len + mdat.len() + 8) as u64);

        // The reported length is enough to find `moov` and learn its size.
        let error = Av3aTrack::from_prefix(&file[..needed as usize])
            .expect_err("moov header is known but its body is not");
        let Mp4Error::NeedMoreData { needed, .. } = error else {
            panic!("expected NeedMoreData, got {error:?}");
        };
        assert_eq!(needed, file.len() as u64);
        assert_eq!(
            Av3aTrack::from_prefix(&file[..needed as usize])
                .expect("complete moov")
                .samples()
                .len(),
            13
        );
    }

    #[test]
    fn skips_tracks_without_an_av3a_sample_entry() {
        let file = build_file(
            &[
                &stsd(&boxed(b"mp4a", &av3a_entry(2, 44_100)[8..])),
                &stts(&[(1, 1_024)]),
                &stsc(&[(1, 1)]),
                &stsz_constant(64, 1),
                &stco(&[900]),
            ],
            &[],
            &[],
        );
        assert!(matches!(
            Av3aTrack::from_prefix(&file),
            Err(Mp4Error::NoAv3aTrack)
        ));
    }

    #[test]
    fn rejects_a_sample_table_shorter_than_stsz_declares() {
        let file = build_file(
            &[
                &stsd(&av3a_entry(2, 44_100)),
                &stts(&[(4, 1_024)]),
                &stsc(&[(1, 1)]),
                &stsz_constant(64, 4),
                &stco(&[900, 964]),
            ],
            &[],
            &[],
        );
        assert!(matches!(
            Av3aTrack::from_prefix(&file),
            Err(Mp4Error::InconsistentIndex {
                declared: 4,
                indexed: 2
            })
        ));
    }

    #[test]
    fn rejects_an_entry_count_larger_than_its_box() {
        // Claim 1024 chunk offsets in a box that holds one.
        let mut payload = Vec::new();
        payload.extend_from_slice(&1_024_u32.to_be_bytes());
        payload.extend_from_slice(&900_u32.to_be_bytes());
        let file = build_file(
            &[
                &stsd(&av3a_entry(2, 44_100)),
                &stts(&[(1, 1_024)]),
                &stsc(&[(1, 1)]),
                &stsz_constant(64, 1),
                &boxed(b"stco", &full_box(0, &payload)),
            ],
            &[],
            &[],
        );
        assert!(matches!(
            Av3aTrack::from_prefix(&file),
            Err(Mp4Error::Truncated {
                context: "stco",
                ..
            })
        ));
    }

    /// A real AV3A frame: mono, 64 kbps at 48 kHz, mirroring the `stream`
    /// tests so the reader is exercised against a header the parser accepts.
    fn make_frame(payload_byte: u8) -> Vec<u8> {
        let payload_len = ((64_000_usize * 1_024 / 48_000) - 56).div_ceil(8);
        let payload = vec![payload_byte; payload_len];
        let crc = crc16(&payload);
        let mut writer = BitWriter::new();
        writer.write_bits(0xfff, 12).unwrap();
        writer.write_bits(2, 4).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 3).unwrap();
        writer.write_bits(0, 3).unwrap();
        writer.write_bits(2, 4).unwrap();
        writer.write_bits(u64::from(crc >> 8), 8).unwrap();
        writer
            .write_bits(ChannelConfig::Mono.index().into(), 7)
            .unwrap();
        writer.write_bits(1, 2).unwrap();
        writer.write_bits(4, 4).unwrap();
        writer.write_bits(u64::from(crc & 0xff), 8).unwrap();
        let mut frame = writer.into_bytes();
        frame.extend_from_slice(&payload);
        frame
    }

    #[test]
    fn reads_frames_out_of_a_container() {
        let frames: Vec<Vec<u8>> = (0..3).map(|index| make_frame(index as u8 + 1)).collect();
        let frame_len = frames[0].len();
        let mdat_payload: Vec<u8> = frames.concat();

        // Lay the file out, then patch `stco` with the real `mdat` offset.
        let placeholder = build_file(
            &[
                &stsd(&av3a_entry(1, 48_000)),
                &stts(&[(3, 1_024)]),
                &stsc(&[(1, 3)]),
                &stsz_constant(frame_len as u32, 3),
                &stco(&[0]),
            ],
            &[],
            &mdat_payload,
        );
        let mdat_start = placeholder.len() - mdat_payload.len();
        let file = build_file(
            &[
                &stsd(&av3a_entry(1, 48_000)),
                &stts(&[(3, 1_024)]),
                &stsc(&[(1, 3)]),
                &stsz_constant(frame_len as u32, 3),
                &stco(&[mdat_start as u32]),
            ],
            &[],
            &mdat_payload,
        );

        let mut reader = Mp4FrameReader::open(Cursor::new(file)).expect("valid container");
        assert_eq!(reader.track().samples().len(), 3);
        for expected in &frames {
            let frame = reader
                .next_frame()
                .expect("read succeeds")
                .expect("a frame");
            assert!(frame.crc_is_valid());
            assert_eq!(frame.bytes(), expected.as_slice());
        }
        assert!(reader.next_frame().expect("read succeeds").is_none());

        reader.seek_to_sample(1).expect("in range");
        let frame = reader
            .next_frame()
            .expect("read succeeds")
            .expect("a frame");
        assert_eq!(frame.bytes(), frames[1].as_slice());
        assert!(matches!(
            reader.seek_to_sample(4),
            Err(Mp4Error::SampleOutOfRange { index: 4, count: 3 })
        ));
    }
}
