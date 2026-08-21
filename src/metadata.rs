use core::fmt;

use crate::bitstream::BitReader;
use crate::error::BitstreamError;
use crate::header::MAX_PAYLOAD_BYTES;
pub use crate::metadata_values::*;

pub const METADATA_PRESENCE_BITS: usize = 2;
pub const MAX_DYNAMIC_METADATA_OBJECTS: usize = 32;

const MAX_METADATA_CHANNELS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticMetadataSummary {
    consumed_bits: usize,
    basic_level: u8,
    contents: u8,
    objects: u8,
    packs: u8,
    channels: u8,
    vr_extension_level: Option<u8>,
}

impl StaticMetadataSummary {
    pub fn consumed_bits(self) -> usize {
        self.consumed_bits
    }

    pub fn basic_level(self) -> u8 {
        self.basic_level
    }

    pub fn contents(self) -> u8 {
        self.contents
    }

    pub fn objects(self) -> u8 {
        self.objects
    }

    pub fn packs(self) -> u8 {
        self.packs
    }

    pub fn channels(self) -> u8 {
        self.channels
    }

    pub fn vr_extension_level(self) -> Option<u8> {
        self.vr_extension_level
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicMetadataSummary {
    consumed_bits: usize,
    level: u8,
    objects: u8,
}

impl DynamicMetadataSummary {
    pub fn consumed_bits(self) -> usize {
        self.consumed_bits
    }

    pub fn level(self) -> u8 {
        self.level
    }

    pub fn objects(self) -> u8 {
        self.objects
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataSummary {
    static_metadata: Option<StaticMetadataSummary>,
    dynamic_metadata: Option<DynamicMetadataSummary>,
    consumed_bits: usize,
    audio_bits: usize,
}

impl MetadataSummary {
    pub fn has_static_metadata(self) -> bool {
        self.static_metadata.is_some()
    }

    pub fn has_dynamic_metadata(self) -> bool {
        self.dynamic_metadata.is_some()
    }

    pub fn static_metadata(self) -> Option<StaticMetadataSummary> {
        self.static_metadata
    }

    pub fn dynamic_metadata(self) -> Option<DynamicMetadataSummary> {
        self.dynamic_metadata
    }

    pub fn consumed_bits(self) -> usize {
        self.consumed_bits
    }

    pub fn audio_bits(self) -> usize {
        self.audio_bits
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataError {
    TooManyDynamicObjects { objects: usize, limit: usize },
    UnmappedChannelFormat { index: u8 },
    PayloadTooLarge { bytes: usize, limit: usize },
    Bitstream(BitstreamError),
}

impl fmt::Display for MetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyDynamicObjects { objects, limit } => write!(
                f,
                "dynamic metadata has {objects} objects; parser limit is {limit}"
            ),
            Self::UnmappedChannelFormat { index } => {
                write!(
                    f,
                    "metadata channel format {index} is not referenced by a pack"
                )
            }
            Self::PayloadTooLarge { bytes, limit } => write!(
                f,
                "metadata-stripped payload needs {bytes} bytes; workspace limit is {limit}"
            ),
            Self::Bitstream(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for MetadataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BitstreamError> for MetadataError {
    fn from(value: BitstreamError) -> Self {
        Self::Bitstream(value)
    }
}

#[derive(Debug)]
pub struct ParsedMetadataPayload<'parser> {
    summary: MetadataSummary,
    metadata: &'parser FrameMetadata,
    audio_payload: &'parser [u8],
}

impl ParsedMetadataPayload<'_> {
    pub fn summary(&self) -> MetadataSummary {
        self.summary
    }

    pub fn metadata(&self) -> &FrameMetadata {
        self.metadata
    }

    pub fn audio_payload(&self) -> &[u8] {
        self.audio_payload
    }

    pub fn audio_bits(&self) -> usize {
        self.summary.audio_bits
    }
}

/// Parses frame metadata and restores the following audio payload to bit zero
/// without allocating per frame.
///
/// [`Self::parse`] uses zero dynamic objects, which is correct for
/// channel-based and HOA frames. Mixed frames must call
/// [`Self::parse_with_object_count`] with the object count from their header.
#[derive(Debug, Clone)]
pub struct MetadataPayloadParser {
    audio_payload: [u8; MAX_PAYLOAD_BYTES],
    metadata: Option<Box<FrameMetadata>>,
    last_parse_succeeded: bool,
}

impl MetadataPayloadParser {
    pub fn new() -> Self {
        Self {
            audio_payload: [0; MAX_PAYLOAD_BYTES],
            metadata: None,
            last_parse_succeeded: false,
        }
    }

    pub fn last_metadata(&self) -> Option<&FrameMetadata> {
        if self.last_parse_succeeded {
            self.metadata.as_deref()
        } else {
            None
        }
    }

    pub(crate) fn prepare_storage(&mut self) {
        if self.metadata.is_none() {
            self.metadata = Some(Box::new(FrameMetadata::default()));
        }
    }

    pub fn parse<'parser>(
        &'parser mut self,
        payload: &[u8],
        payload_bits: usize,
    ) -> Result<ParsedMetadataPayload<'parser>, MetadataError> {
        self.parse_with_object_count(payload, payload_bits, 0)
    }

    pub fn parse_with_object_count<'parser>(
        &'parser mut self,
        payload: &[u8],
        payload_bits: usize,
        dynamic_objects: usize,
    ) -> Result<ParsedMetadataPayload<'parser>, MetadataError> {
        self.last_parse_succeeded = false;
        self.prepare_storage();
        let metadata = self.metadata.as_mut().expect("metadata storage prepared");
        metadata.clear_presence();
        let mut reader = BitReader::with_bit_len(payload, payload_bits)?;

        let has_static_metadata = read_flag(&mut reader)?;
        metadata.has_static_metadata = has_static_metadata;
        if has_static_metadata {
            parse_static_metadata(&mut reader, &mut metadata.static_metadata)?;
        }

        let has_dynamic_metadata = read_flag(&mut reader)?;
        metadata.has_dynamic_metadata = has_dynamic_metadata;
        if has_dynamic_metadata {
            parse_dynamic_metadata(&mut reader, dynamic_objects, &mut metadata.dynamic_metadata)?;
        }

        let consumed_bits = reader.position();
        let audio_bits = reader.remaining();
        let audio_bytes = audio_bits.div_ceil(8);
        if audio_bytes > self.audio_payload.len() {
            return Err(MetadataError::PayloadTooLarge {
                bytes: audio_bytes,
                limit: self.audio_payload.len(),
            });
        }

        let full_bytes = audio_bits / 8;
        reader.read_bytes_into(&mut self.audio_payload[..full_bytes])?;
        let trailing_bits = audio_bits % 8;
        if trailing_bits != 0 {
            self.audio_payload[full_bytes] = reader.read_u8(trailing_bits)? << (8 - trailing_bits);
        }
        debug_assert_eq!(reader.remaining(), 0);

        let static_metadata = has_static_metadata.then(|| {
            let metadata = &metadata.static_metadata;
            StaticMetadataSummary {
                consumed_bits: metadata.consumed_bits,
                basic_level: metadata.basic_level,
                contents: metadata.basic.contents.len() as u8,
                objects: metadata.basic.objects.len() as u8,
                packs: metadata.basic.packs.len() as u8,
                channels: metadata.basic.channels.len() as u8,
                vr_extension_level: metadata.vr_extension_level,
            }
        });
        let dynamic_metadata = has_dynamic_metadata.then(|| {
            let metadata = &metadata.dynamic_metadata;
            DynamicMetadataSummary {
                consumed_bits: metadata.consumed_bits,
                level: metadata.level,
                objects: metadata.objects.len() as u8,
            }
        });
        let summary = MetadataSummary {
            static_metadata,
            dynamic_metadata,
            consumed_bits,
            audio_bits,
        };
        self.last_parse_succeeded = true;
        Ok(ParsedMetadataPayload {
            summary,
            metadata,
            audio_payload: &self.audio_payload[..audio_bytes],
        })
    }
}

impl Default for MetadataPayloadParser {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
struct ChannelSyntaxReference {
    type_label: u8,
    pack_format_id: u8,
    matrix_output_channels: u8,
}

const RES_VR_VERTEX: f32 = 200.0_f32 / 127.0_f32;
const RES_VR_UNIT: f32 = 1.0_f32 / 127.0_f32;
const RES_VR_EQ_FC: f32 = 58.061_8_f32 / 127.0_f32;
const RES_VR_EQ_Q_LOW: f32 = 0.9_f32 / 63.0_f32;
const RES_VR_EQ_Q_HIGH: f32 = 11.0_f32 / 63.0_f32;
const RES_VR_EQ_GAIN: f32 = 40.0_f32 / 127.0_f32;
const RES_VR_ATTACK: f32 = 99.0_f32 / 15.0_f32;
const RES_VR_RELEASE: f32 = 250.0_f32 / 15.0_f32;
const RES_VR_THRESHOLD: f32 = 90.0_f32 / 127.0_f32;
const RES_VR_PRE_GAIN: f32 = 20.0_f32 / 127.0_f32;
const RES_VR_POST_GAIN: f32 = 20.0_f32 / 127.0_f32;
const RES_VR_RATIO: f32 = 99.0_f32 / 127.0_f32;
const RES_VR_EFFECT_GAIN: f32 = 40.0_f32 / 127.0_f32;

const RES_STATIC_RANGE_MIN: f32 = 1.0_f32 / 127.0_f32;
const RES_STATIC_RANGE_MAX: f32 = 15.0_f32 / 127.0_f32;
const RES_STATIC_RANGE_DB_MIN: f32 = -80.0_f32 / 127.0_f32;
const RES_STATIC_RANGE_DB_MAX: f32 = 24.0_f32 / 127.0_f32;
const RES_MATRIX_COEFFICIENT: f32 = 9.9_f32 / 255.0_f32;
const RES_DUCKING_DEPTH: f32 = -62.0_f32 / 31.0_f32;
const RES_STATIC_ABSOLUTE_DISTANCE: f32 = 1.230_448_f32 / 31.0_f32;
const RES_STATIC_NFC_REFERENCE_DISTANCE: f32 = 1.0_f32 / 15.0_f32;
const RES_STATIC_GAIN_LOW: f32 = 1.0_f32 / 63.0_f32;
const RES_STATIC_GAIN_HIGH: f32 = 15.0_f32 / 63.0_f32;
const RES_STATIC_GAIN_DB_LOW: f32 = -80.0_f32 / 63.0_f32;
const RES_STATIC_GAIN_DB_HIGH: f32 = 24.0_f32 / 63.0_f32;
const RES_STATIC_X: f32 = 2.0_f32 / 255.0_f32;
const RES_STATIC_Y: f32 = 2.0_f32 / 63.0_f32;
const RES_STATIC_Z: f32 = 2.0_f32 / 15.0_f32;
const RES_STATIC_CARTESIAN_WIDTH: f32 = 1.0_f32 / 127.0_f32;
const RES_STATIC_AZIMUTH_MIN: f32 = -180.0_f32 / 255.0_f32;
const RES_STATIC_AZIMUTH_MAX: f32 = 180.0_f32 / 255.0_f32;
const RES_STATIC_ELEVATION_MIN: f32 = -90.0_f32 / 63.0_f32;
const RES_STATIC_ELEVATION_MAX: f32 = 90.0_f32 / 63.0_f32;
const RES_STATIC_AZIMUTH: f32 = 360.0_f32 / 255.0_f32;
const RES_STATIC_ELEVATION: f32 = 180.0_f32 / 63.0_f32;
const RES_STATIC_SCREEN_ELEVATION: f32 = 90.0_f32 / 63.0_f32;
const RES_STATIC_DISTANCE: f32 = 1.0_f32 / 15.0_f32;
const RES_STATIC_POLAR_WIDTH: f32 = 180.0_f32 / 127.0_f32;
const RES_STATIC_LOUDNESS: f32 = -70.0_f32 / 31.0_f32;
const RES_STATIC_LOUDNESS_RANGE: f32 = 60.0_f32 / 31.0_f32;

const RES_OBJECT_AZIMUTH: f32 = 360.0_f32 / 255.0_f32;
const RES_OBJECT_ELEVATION: f32 = 180.0_f32 / 63.0_f32;
const RES_OBJECT_DISTANCE: f32 = 1.0_f32 / 15.0_f32;
const RES_OBJECT_GAIN: f32 = 6.0_f32 / 127.0_f32;
const RES_OBJECT_WIDTH: f32 = 360.0_f32 / 127.0_f32;
const RES_OBJECT_HEIGHT: f32 = 360.0_f32 / 31.0_f32;
const RES_OBJECT_DEPTH: f32 = 1.0_f32 / 15.0_f32;
const RES_OBJECT_DIFFUSE: f32 = 1.0_f32 / 127.0_f32;
const RES_OBJECT_X: f32 = 2.0_f32 / 255.0_f32;
const RES_OBJECT_Y: f32 = 2.0_f32 / 63.0_f32;
const RES_OBJECT_Z: f32 = 2.0_f32 / 15.0_f32;
const RES_OBJECT_WIDTH_X: f32 = 1.0_f32 / 127.0_f32;
const RES_OBJECT_HEIGHT_Y: f32 = 1.0_f32 / 31.0_f32;
const RES_OBJECT_DEPTH_Z: f32 = 1.0_f32 / 15.0_f32;
const RES_CHANNEL_LOCK_DISTANCE: f32 = 2.0_f32 / 15.0_f32;
const RES_OBJECT_DIVERGENCE: f32 = 1.0_f32 / 15.0_f32;
const RES_OBJECT_DIVERGENCE_AZIMUTH: f32 = 180.0_f32 / 63.0_f32;

fn read_flag(reader: &mut BitReader<'_>) -> Result<bool, MetadataError> {
    Ok(reader.read_u8(1)? != 0)
}

fn read_count(reader: &mut BitReader<'_>, width: usize) -> Result<usize, MetadataError> {
    Ok(usize::from(reader.read_u8(width)?) + 1)
}

fn scaled(raw: u8, resolution: f32) -> f32 {
    f32::from(raw) * resolution
}

fn centered(raw: u8, midpoint: i16, resolution: f32) -> f32 {
    f32::from(i16::from(raw) - midpoint) * resolution
}

fn parse_static_metadata(
    reader: &mut BitReader<'_>,
    output: &mut StaticMetadata,
) -> Result<(), MetadataError> {
    output.consumed_bits = 0;
    output.vr_extension_level = None;
    let start = reader.position();
    let has_vr_extension = read_flag(reader)?;
    output.basic_level = reader.read_u8(3)?.min(1);
    parse_basic_l1(reader, &mut output.basic)?;

    if has_vr_extension {
        let level = reader.read_u8(3)?;
        output.vr_extension_level = Some(level);
        if level == 0 {
            let vr_extension = output
                .vr_extension_l1
                .get_or_insert_with(VrExtensionMetadata::default);
            parse_vr_extension_l1(reader, vr_extension)?;
        } else {
            output.vr_extension_l1 = None;
        }
    } else {
        output.vr_extension_l1 = None;
    }

    output.consumed_bits = reader.position() - start;
    Ok(())
}

fn parse_basic_l1(
    reader: &mut BitReader<'_>,
    output: &mut BasicMetadata,
) -> Result<(), MetadataError> {
    output.programme = parse_audio_programme(reader)?;

    let contents = read_count(reader, 2)?;
    for content in output.contents.prepare(contents) {
        *content = parse_audio_content(reader)?;
    }

    let objects = read_count(reader, 3)?;
    for object in output.objects.prepare(objects) {
        *object = parse_audio_object(reader)?;
    }

    let packs = read_count(reader, 3)?;
    let mut channel_references = [None; MAX_METADATA_CHANNELS];
    for pack in output.packs.prepare(packs) {
        parse_audio_pack(reader, pack, &mut channel_references)?;
    }

    let channels = read_count(reader, 5)?;
    for channel in output.channels.prepare(channels) {
        parse_audio_channel(reader, channel, &channel_references)?;
    }
    Ok(())
}

fn parse_audio_programme(
    reader: &mut BitReader<'_>,
) -> Result<AudioProgrammeMetadata, MetadataError> {
    let has_language = read_flag(reader)?;
    let has_max_ducking_depth = read_flag(reader)?;
    let has_loudness = read_flag(reader)?;
    let has_reference_screen = read_flag(reader)?;

    let language = has_language.then(|| reader.read_u8(4)).transpose()?;
    let max_ducking_depth = has_max_ducking_depth
        .then(|| reader.read_u8(5))
        .transpose()?
        .map(|raw| scaled(raw, RES_DUCKING_DEPTH));
    let loudness = has_loudness.then(|| parse_loudness(reader)).transpose()?;
    let reference_screen = has_reference_screen
        .then(|| parse_programme_reference_screen(reader))
        .transpose()?;

    let mut output = AudioProgrammeMetadata {
        language,
        max_ducking_depth,
        loudness,
        reference_screen,
        ..AudioProgrammeMetadata::default()
    };
    let contents = read_count(reader, 2)?;
    for reference in output.content_references.prepare(contents) {
        *reference = reader.read_u8(2)?;
    }
    Ok(output)
}

fn parse_loudness(reader: &mut BitReader<'_>) -> Result<LoudnessMetadata, MetadataError> {
    let has_integrated_loudness = read_flag(reader)?;
    let has_loudness_range = read_flag(reader)?;
    let has_max_true_peak = read_flag(reader)?;
    let has_max_momentary = read_flag(reader)?;
    let has_max_short_term = read_flag(reader)?;
    let has_dialogue_loudness = read_flag(reader)?;

    Ok(LoudnessMetadata {
        integrated_loudness: has_integrated_loudness
            .then(|| reader.read_u8(5))
            .transpose()?
            .map(|raw| scaled(raw, RES_STATIC_LOUDNESS)),
        loudness_range: has_loudness_range
            .then(|| reader.read_u8(5))
            .transpose()?
            .map(|raw| scaled(raw, RES_STATIC_LOUDNESS_RANGE) + 10.0_f32),
        max_true_peak: has_max_true_peak
            .then(|| reader.read_u8(5))
            .transpose()?
            .map(|raw| scaled(raw, RES_STATIC_LOUDNESS)),
        max_momentary: has_max_momentary
            .then(|| reader.read_u8(5))
            .transpose()?
            .map(|raw| scaled(raw, RES_STATIC_LOUDNESS)),
        max_short_term: has_max_short_term
            .then(|| reader.read_u8(5))
            .transpose()?
            .map(|raw| scaled(raw, RES_STATIC_LOUDNESS)),
        dialogue_loudness: has_dialogue_loudness
            .then(|| reader.read_u8(5))
            .transpose()?
            .map(|raw| scaled(raw, RES_STATIC_LOUDNESS)),
    })
}

fn parse_programme_reference_screen(
    reader: &mut BitReader<'_>,
) -> Result<ProgrammeReferenceScreen, MetadataError> {
    let cartesian = read_flag(reader)?;
    let aspect_ratio = reader.read_u8(3)?;
    let position = if cartesian {
        ProgrammeScreenPosition::Cartesian {
            x: centered(reader.read_u8(8)?, 128, RES_STATIC_X),
            y: centered(reader.read_u8(6)?, 32, RES_STATIC_Y),
            z: centered(reader.read_u8(4)?, 8, RES_STATIC_Z),
            width: scaled(reader.read_u8(7)?, RES_STATIC_CARTESIAN_WIDTH),
        }
    } else {
        ProgrammeScreenPosition::Polar {
            azimuth: centered(reader.read_u8(8)?, 128, RES_STATIC_AZIMUTH),
            elevation: scaled(reader.read_u8(6)?, RES_STATIC_SCREEN_ELEVATION),
            distance: scaled(reader.read_u8(4)?, RES_STATIC_DISTANCE),
            width: scaled(reader.read_u8(7)?, RES_STATIC_POLAR_WIDTH),
        }
    };
    Ok(ProgrammeReferenceScreen {
        aspect_ratio,
        position,
    })
}

fn parse_dialogue(reader: &mut BitReader<'_>) -> Result<DialogueMetadata, MetadataError> {
    Ok(DialogueMetadata {
        attribute: reader.read_u8(2)?,
        dialogue_type: reader.read_u8(3)?,
    })
}

fn parse_audio_content(reader: &mut BitReader<'_>) -> Result<AudioContentMetadata, MetadataError> {
    let index = reader.read_u8(2)?;
    let has_language = read_flag(reader)?;
    let has_loudness = read_flag(reader)?;
    let has_dialogue = read_flag(reader)?;
    let has_complementary_groups = read_flag(reader)?;

    let mut output = AudioContentMetadata {
        index,
        language: has_language.then(|| reader.read_u8(4)).transpose()?,
        loudness: has_loudness.then(|| parse_loudness(reader)).transpose()?,
        dialogue: has_dialogue.then(|| parse_dialogue(reader)).transpose()?,
        ..AudioContentMetadata::default()
    };

    if has_complementary_groups {
        let groups = read_count(reader, 2)?;
        for group in output.complementary_object_groups.prepare(groups) {
            let objects = read_count(reader, 3)?;
            for reference in group.object_references.prepare(objects) {
                *reference = reader.read_u8(3)?;
            }
        }
    }

    let objects = read_count(reader, 3)?;
    for reference in output.object_references.prepare(objects) {
        *reference = reader.read_u8(3)?;
    }
    Ok(output)
}

fn parse_audio_object(reader: &mut BitReader<'_>) -> Result<AudioObjectMetadata, MetadataError> {
    let index = reader.read_u8(3)?;
    let has_language = read_flag(reader)?;
    let has_dialogue = read_flag(reader)?;
    let has_importance = read_flag(reader)?;
    let disable_ducking = read_flag(reader)?;
    let has_interaction = read_flag(reader)?;
    let has_gain = read_flag(reader)?;
    let head_locked = read_flag(reader)?;
    let muted = read_flag(reader)?;

    let language = has_language.then(|| reader.read_u8(4)).transpose()?;
    let dialogue = has_dialogue.then(|| parse_dialogue(reader)).transpose()?;
    let importance = has_importance
        .then(|| reader.read_u8(4))
        .transpose()?
        .map(|value| value.min(10));

    let (name, interaction) = if has_interaction {
        let mut name = [0; 24];
        for character in &mut name {
            *character = reader.read_u8(8)?;
        }
        (Some(name), Some(parse_audio_object_interaction(reader)?))
    } else {
        (None, None)
    };

    let gain = has_gain.then(|| parse_static_gain(reader)).transpose()?;

    let mut output = AudioObjectMetadata {
        index,
        language,
        dialogue,
        importance,
        disable_ducking,
        head_locked,
        muted,
        name,
        interaction,
        gain,
        ..AudioObjectMetadata::default()
    };
    let packs = read_count(reader, 3)?;
    for reference in output.pack_references.prepare(packs) {
        *reference = reader.read_u8(3)?;
    }
    Ok(output)
}

fn parse_audio_object_interaction(
    reader: &mut BitReader<'_>,
) -> Result<AudioObjectInteractionMetadata, MetadataError> {
    let on_off_interact = read_flag(reader)?;
    let has_gain = read_flag(reader)?;
    let has_position = read_flag(reader)?;

    let gain = if has_gain {
        let unit = read_gain_unit(reader)?;
        let minimum_raw = reader.read_u8(7)?;
        let maximum_raw = reader.read_u8(7)?;
        let (minimum, maximum) = match unit {
            MetadataGainUnit::Linear => (
                scaled(minimum_raw, RES_STATIC_RANGE_MIN),
                scaled(maximum_raw, RES_STATIC_RANGE_MAX) + 1.0_f32,
            ),
            MetadataGainUnit::Decibels => (
                scaled(minimum_raw, RES_STATIC_RANGE_DB_MIN),
                scaled(maximum_raw, RES_STATIC_RANGE_DB_MAX),
            ),
        };
        Some(GainInteractionMetadata {
            unit,
            minimum,
            maximum,
        })
    } else {
        None
    };

    let position = if has_position {
        let cartesian = read_flag(reader)?;
        if cartesian {
            Some(PositionInteractionMetadata::Cartesian {
                x_min: centered(reader.read_u8(8)?, 128, RES_STATIC_X),
                x_max: centered(reader.read_u8(8)?, 128, RES_STATIC_X),
                y_min: centered(reader.read_u8(6)?, 32, RES_STATIC_Y),
                y_max: centered(reader.read_u8(6)?, 32, RES_STATIC_Y),
                z_min: centered(reader.read_u8(4)?, 8, RES_STATIC_Z),
                z_max: centered(reader.read_u8(4)?, 8, RES_STATIC_Z),
            })
        } else {
            Some(PositionInteractionMetadata::Polar {
                azimuth_min: scaled(reader.read_u8(8)?, RES_STATIC_AZIMUTH_MIN),
                azimuth_max: scaled(reader.read_u8(8)?, RES_STATIC_AZIMUTH_MAX),
                elevation_min: scaled(reader.read_u8(6)?, RES_STATIC_ELEVATION_MIN),
                elevation_max: scaled(reader.read_u8(6)?, RES_STATIC_ELEVATION_MAX),
                distance_min: scaled(reader.read_u8(4)?, RES_STATIC_DISTANCE),
                distance_max: scaled(reader.read_u8(4)?, RES_STATIC_DISTANCE),
            })
        }
    } else {
        None
    };

    Ok(AudioObjectInteractionMetadata {
        on_off_interact,
        gain,
        position,
    })
}

fn read_gain_unit(reader: &mut BitReader<'_>) -> Result<MetadataGainUnit, MetadataError> {
    Ok(if read_flag(reader)? {
        MetadataGainUnit::Decibels
    } else {
        MetadataGainUnit::Linear
    })
}

fn parse_static_gain(reader: &mut BitReader<'_>) -> Result<MetadataGain, MetadataError> {
    let unit = read_gain_unit(reader)?;
    let upper_half = read_flag(reader)?;
    let raw = reader.read_u8(6)?;
    let value = match (unit, upper_half) {
        (MetadataGainUnit::Linear, false) => scaled(raw, RES_STATIC_GAIN_LOW),
        (MetadataGainUnit::Linear, true) => scaled(raw, RES_STATIC_GAIN_HIGH) + 1.0_f32,
        (MetadataGainUnit::Decibels, false) => scaled(raw, RES_STATIC_GAIN_DB_LOW),
        (MetadataGainUnit::Decibels, true) => scaled(raw, RES_STATIC_GAIN_DB_HIGH),
    };
    Ok(MetadataGain { unit, value })
}

fn parse_direct_speaker_position(
    reader: &mut BitReader<'_>,
) -> Result<DirectSpeakerPosition, MetadataError> {
    Ok(DirectSpeakerPosition {
        azimuth: centered(reader.read_u8(8)?, 128, RES_STATIC_AZIMUTH),
        elevation: centered(reader.read_u8(6)?, 32, RES_STATIC_ELEVATION),
        distance: scaled(reader.read_u8(4)?, RES_STATIC_DISTANCE),
        screen_edge_lock: reader.read_u8(2)?,
    })
}

fn parse_audio_pack(
    reader: &mut BitReader<'_>,
    output: &mut AudioPackMetadata,
    channel_references: &mut [Option<ChannelSyntaxReference>; MAX_METADATA_CHANNELS],
) -> Result<(), MetadataError> {
    *output = AudioPackMetadata::default();
    output.index = reader.read_u8(3)?;
    let has_importance = read_flag(reader)?;
    output.channel_reuse = read_flag(reader)?;
    output.importance = has_importance
        .then(|| reader.read_u8(4))
        .transpose()?
        .map(|value| value.min(10));
    output.type_label = reader.read_u8(3)?.clamp(1, 5);

    let absolute_distance_code = reader.read_u8(5)?;
    let logarithmic_distance = scaled(absolute_distance_code, RES_STATIC_ABSOLUTE_DISTANCE);
    output.absolute_distance = 10.0_f32.powf(logarithmic_distance) - 1.0_f32;

    if output.type_label == 4 {
        output.hoa = Some(HoaPackMetadata {
            normalization: reader.read_u8(2)?,
            nfc_reference_distance: scaled(reader.read_u8(4)?, RES_STATIC_NFC_REFERENCE_DISTANCE),
            screen_reference: read_flag(reader)?,
            order: reader.read_u8(3)?,
        });
    }

    if output.type_label == 1 || output.type_label == 2 {
        output.pack_format_id = Some(reader.read_u8(6)?);
        if output.type_label == 2 {
            let matrix_channels = read_count(reader, 5)?;
            for position in output.matrix_output_positions.prepare(matrix_channels) {
                *position = parse_direct_speaker_position(reader)?;
            }
        }
    }

    if !output.channel_reuse {
        output.pack_format_start_index = Some(reader.read_u8(5)?);
    }

    let channels = read_count(reader, 5)?;
    let type_label = output.type_label;
    let pack_format_id = output.pack_format_id.unwrap_or(0);
    let matrix_output_channels = output.matrix_output_positions.len() as u8;
    for channel in output.channels.prepare(channels) {
        let channel_index = reader.read_u8(5)?;
        channel_references[usize::from(channel_index)] = Some(ChannelSyntaxReference {
            type_label,
            pack_format_id,
            matrix_output_channels,
        });
        *channel = PackChannelReference {
            channel_index,
            transformed_channel_reference: output
                .channel_reuse
                .then(|| reader.read_u8(5))
                .transpose()?,
        };
    }
    Ok(())
}

fn parse_audio_channel(
    reader: &mut BitReader<'_>,
    output: &mut AudioChannelMetadata,
    channel_references: &[Option<ChannelSyntaxReference>; MAX_METADATA_CHANNELS],
) -> Result<(), MetadataError> {
    *output = AudioChannelMetadata::default();
    output.index = reader.read_u8(5)?;
    if read_flag(reader)? {
        output.gain = Some(parse_static_gain(reader)?);
    }

    let reference = channel_references[usize::from(output.index)].ok_or(
        MetadataError::UnmappedChannelFormat {
            index: output.index,
        },
    )?;
    if reference.type_label == 1 && reference.pack_format_id == 63 {
        output.direct_speaker_position = Some(parse_direct_speaker_position(reader)?);
    } else if reference.type_label == 2 {
        for coefficient in output
            .matrix_coefficients
            .prepare(usize::from(reference.matrix_output_channels))
        {
            *coefficient = scaled(reader.read_u8(8)?, RES_MATRIX_COEFFICIENT) + 0.1_f32;
        }
    }
    Ok(())
}

fn parse_vr_extension_l1(
    reader: &mut BitReader<'_>,
    output: &mut VrExtensionMetadata,
) -> Result<(), MetadataError> {
    let has_acoustic_environment = read_flag(reader)?;
    let has_render_info = read_flag(reader)?;
    output.ambisonic_order = reader.read_u8(3)?;
    if has_acoustic_environment {
        let environment = output
            .acoustic_environment
            .get_or_insert_with(VrAcousticEnvironmentMetadata::default);
        parse_vr_acoustic_environment(reader, environment)?;
    } else {
        output.acoustic_environment = None;
    }
    output.render_info = has_render_info
        .then(|| parse_vr_render_info(reader))
        .transpose()?;
    Ok(())
}

fn parse_vr_acoustic_environment(
    reader: &mut BitReader<'_>,
    output: &mut VrAcousticEnvironmentMetadata,
) -> Result<(), MetadataError> {
    let has_early_reflection_gain = read_flag(reader)?;
    let has_late_reverb_gain = read_flag(reader)?;
    let reverb_type = reader.read_u8(2)?;
    let early_reflection_gain = has_early_reflection_gain
        .then(|| reader.read_u8(7))
        .transpose()?
        .map(|raw| scaled(raw, RES_VR_UNIT));
    let late_reverb_gain = has_late_reverb_gain
        .then(|| reader.read_u8(7))
        .transpose()?
        .map(|raw| scaled(raw, RES_VR_UNIT));
    let low_frequency_processing = read_flag(reader)?;
    let convolution_reverb_type = (reverb_type == 2).then(|| reader.read_u8(5)).transpose()?;

    output.early_reflection_gain = early_reflection_gain;
    output.late_reverb_gain = late_reverb_gain;
    output.reverb_type = reverb_type;
    output.low_frequency_processing = low_frequency_processing;
    output.convolution_reverb_type = convolution_reverb_type;
    let surfaces = read_count(reader, 3)?;
    for surface in output.surfaces.prepare(surfaces) {
        parse_vr_surface(reader, surfaces, surface)?;
    }
    Ok(())
}

fn parse_vr_surface(
    reader: &mut BitReader<'_>,
    surfaces: usize,
    output: &mut VrSurfaceMetadata,
) -> Result<(), MetadataError> {
    output.material = reader.read_u8(5)?;
    let (absorption, scattering) = if output.material == 31 {
        let mut absorption = [0.0; 8];
        let mut scattering = [0.0; 8];
        for index in 0..8 {
            absorption[index] = scaled(reader.read_u8(7)?, RES_VR_UNIT);
            scattering[index] = scaled(reader.read_u8(7)?, RES_VR_UNIT);
        }
        (Some(absorption), Some(scattering))
    } else {
        (None, None)
    };

    let encoded_vertices = read_count(reader, 5)?;
    let minimum_vertices = 8_usize.div_ceil(surfaces);
    let maximum_vertices = 36 / surfaces;
    let vertices = encoded_vertices.clamp(minimum_vertices, maximum_vertices);
    output.absorption = absorption;
    output.scattering = scattering;
    for vertex in output.vertices.prepare(vertices) {
        *vertex = VrVertex {
            x: centered(reader.read_u8(7)?, 64, RES_VR_VERTEX),
            y: centered(reader.read_u8(7)?, 64, RES_VR_VERTEX),
            z: centered(reader.read_u8(7)?, 64, RES_VR_VERTEX),
        };
    }
    Ok(())
}

fn parse_vr_render_info(reader: &mut BitReader<'_>) -> Result<VrRenderInfoMetadata, MetadataError> {
    let target_device = read_flag(reader)?;
    let hrtf_type = reader.read_u8(4)?;
    let mut headphone_types = [0; 16];
    for headphone_type in &mut headphone_types {
        *headphone_type = reader.read_u8(7)?;
    }
    Ok(VrRenderInfoMetadata {
        target_device,
        hrtf_type,
        headphone_types,
        audio_effect: parse_vr_audio_effect(reader)?,
    })
}

fn parse_vr_audio_effect(
    reader: &mut BitReader<'_>,
) -> Result<VrAudioEffectMetadata, MetadataError> {
    let has_eq = read_flag(reader)?;
    let has_drc = read_flag(reader)?;
    let has_gain = read_flag(reader)?;
    let effect_chain = (has_eq || has_drc || has_gain)
        .then(|| reader.read_u8(3))
        .transpose()?
        .map(|value| value.min(5));

    let mut output = VrAudioEffectMetadata {
        effect_chain,
        ..VrAudioEffectMetadata::default()
    };
    if has_eq {
        let bands = usize::from(reader.read_u8(4)?.min(10)) + 1;
        for band in output.eq_bands.prepare(bands) {
            *band = parse_vr_eq_band(reader)?;
        }
    }
    if has_drc {
        output.drc = Some(VrDrcMetadata {
            attack_time: scaled(reader.read_u8(4)?, RES_VR_ATTACK) + 1.0_f32,
            release_time: scaled(reader.read_u8(4)?, RES_VR_RELEASE) + 50.0_f32,
            threshold: scaled(reader.read_u8(7)?, RES_VR_THRESHOLD) - 80.0_f32,
            pre_gain: centered(reader.read_u8(7)?, 64, RES_VR_PRE_GAIN),
            post_gain: scaled(reader.read_u8(7)?, RES_VR_POST_GAIN),
            ratio: scaled(reader.read_u8(7)?, RES_VR_RATIO) + 1.0_f32,
        });
    }
    if has_gain {
        output.gain = Some(centered(reader.read_u8(7)?, 64, RES_VR_EFFECT_GAIN));
    }
    Ok(output)
}

fn parse_vr_eq_band(reader: &mut BitReader<'_>) -> Result<VrEqBandMetadata, MetadataError> {
    let eq_type = reader.read_u8(3)?;
    let frequency_code = reader.read_u8(7)?;
    let logarithmic_frequency = scaled(frequency_code, RES_VR_EQ_FC);
    let center_frequency =
        10.0_f32.powf((logarithmic_frequency + 20.0_f32 * 20.0_f32.log10()) / 20.0_f32);
    let upper_q_range = read_flag(reader)?;
    let q_code = reader.read_u8(6)?;
    let q = if upper_q_range {
        scaled(q_code, RES_VR_EQ_Q_HIGH) + 1.0_f32
    } else {
        scaled(q_code, RES_VR_EQ_Q_LOW) + 0.1_f32
    };
    Ok(VrEqBandMetadata {
        eq_type,
        center_frequency,
        q,
        gain: centered(reader.read_u8(7)?, 64, RES_VR_EQ_GAIN),
    })
}

fn parse_dynamic_metadata(
    reader: &mut BitReader<'_>,
    objects: usize,
    output: &mut DynamicMetadata,
) -> Result<(), MetadataError> {
    if objects > MAX_DYNAMIC_METADATA_OBJECTS {
        return Err(MetadataError::TooManyDynamicObjects {
            objects,
            limit: MAX_DYNAMIC_METADATA_OBJECTS,
        });
    }

    output.consumed_bits = 0;
    let start = reader.position();
    output.level = reader.read_u8(3)?;
    let level = output.level;
    for object in output.objects.prepare(objects) {
        object.muted = read_flag(reader)?;
        object.transport_channel_reference = reader.read_u8(5)?;
        if !object.muted && (level == 0 || level == 1) {
            object.level1 = Some(parse_dynamic_l1(reader)?);
            if level == 1 {
                object.level2 = Some(parse_dynamic_l2(reader)?);
            }
        }
    }
    output.consumed_bits = reader.position() - start;
    Ok(())
}

fn parse_dynamic_l1(reader: &mut BitReader<'_>) -> Result<DynamicLevel1Metadata, MetadataError> {
    let cartesian = read_flag(reader)?;
    let has_extent = read_flag(reader)?;
    let has_gain = read_flag(reader)?;
    let has_diffuse = read_flag(reader)?;
    let has_importance = read_flag(reader)?;

    let (position, extent) = if cartesian {
        let position = DynamicObjectPosition::Cartesian {
            x: centered(reader.read_u8(8)?, 128, RES_OBJECT_X),
            y: centered(reader.read_u8(6)?, 32, RES_OBJECT_Y),
            z: centered(reader.read_u8(4)?, 8, RES_OBJECT_Z),
        };
        let extent = has_extent
            .then(|| -> Result<_, MetadataError> {
                Ok(DynamicObjectExtent::Cartesian {
                    width_x: scaled(reader.read_u8(7)?, RES_OBJECT_WIDTH_X),
                    height_y: scaled(reader.read_u8(5)?, RES_OBJECT_HEIGHT_Y),
                    depth_z: scaled(reader.read_u8(4)?, RES_OBJECT_DEPTH_Z),
                })
            })
            .transpose()?;
        (position, extent)
    } else {
        let position = DynamicObjectPosition::Polar {
            azimuth: centered(reader.read_u8(8)?, 128, RES_OBJECT_AZIMUTH),
            elevation: centered(reader.read_u8(6)?, 32, RES_OBJECT_ELEVATION),
            distance: scaled(reader.read_u8(4)?, RES_OBJECT_DISTANCE),
        };
        let extent = has_extent
            .then(|| -> Result<_, MetadataError> {
                Ok(DynamicObjectExtent::Polar {
                    width: scaled(reader.read_u8(7)?, RES_OBJECT_WIDTH),
                    height: scaled(reader.read_u8(5)?, RES_OBJECT_HEIGHT),
                    depth: scaled(reader.read_u8(4)?, RES_OBJECT_DEPTH),
                })
            })
            .transpose()?;
        (position, extent)
    };

    Ok(DynamicLevel1Metadata {
        position,
        extent,
        gain: has_gain
            .then(|| reader.read_u8(7))
            .transpose()?
            .map(|raw| scaled(raw, RES_OBJECT_GAIN)),
        diffuse: has_diffuse
            .then(|| reader.read_u8(7))
            .transpose()?
            .map(|raw| scaled(raw, RES_OBJECT_DIFFUSE)),
        jump_position: read_flag(reader)?,
        importance: has_importance
            .then(|| reader.read_u8(4))
            .transpose()?
            .map(|value| value.min(10)),
    })
}

fn parse_dynamic_l2(reader: &mut BitReader<'_>) -> Result<DynamicLevel2Metadata, MetadataError> {
    let channel_lock = if read_flag(reader)? {
        let locked = read_flag(reader)?;
        Some(ChannelLockMetadata {
            locked,
            maximum_distance: locked
                .then(|| reader.read_u8(4))
                .transpose()?
                .map(|raw| scaled(raw, RES_CHANNEL_LOCK_DISTANCE)),
        })
    } else {
        None
    };

    let object_divergence = if read_flag(reader)? {
        let divergence_code = reader.read_u8(4)?;
        Some(ObjectDivergenceMetadata {
            divergence: scaled(divergence_code, RES_OBJECT_DIVERGENCE),
            azimuth_range: (divergence_code != 0)
                .then(|| reader.read_u8(6))
                .transpose()?
                .map(|raw| scaled(raw, RES_OBJECT_DIVERGENCE_AZIMUTH)),
        })
    } else {
        None
    };

    Ok(DynamicLevel2Metadata {
        channel_lock,
        object_divergence,
        object_screen_reference: read_flag(reader)?.then(|| read_flag(reader)).transpose()?,
        screen_edge_lock: read_flag(reader)?.then(|| reader.read_u8(2)).transpose()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;
    use crate::header::MAX_PAYLOAD_BYTES;

    fn write(writer: &mut BitWriter, value: u64, width: usize) {
        writer.write_bits(value, width).unwrap();
    }

    fn assert_close(actual: f32, expected: f32) {
        let tolerance = 8.0_f32 * f32::EPSILON * expected.abs().max(1.0_f32);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }

    #[test]
    fn metadata_storage_sizes_are_bounded() {
        assert!(core::mem::size_of::<FrameMetadata>() < 20 * 1024);
        assert!(
            core::mem::size_of::<MetadataPayloadParser>() <= MAX_PAYLOAD_BYTES + 32,
            "complete metadata must remain out of the backend's inline state"
        );
    }

    fn write_minimal_static_metadata(writer: &mut BitWriter) {
        write(writer, 0, 1); // no VR extension
        write(writer, 0, 3); // Basic L1

        write(writer, 0, 4); // programme flags
        write(writer, 0, 2); // one programme content reference
        write(writer, 0, 2);

        write(writer, 0, 2); // one content
        write(writer, 0, 2); // content index
        write(writer, 0, 4); // content flags
        write(writer, 0, 3); // one object reference
        write(writer, 0, 3);

        write(writer, 0, 3); // one object
        write(writer, 0, 3); // object index
        write(writer, 0, 8); // object flags
        write(writer, 0, 3); // one pack reference
        write(writer, 0, 3);

        write(writer, 0, 3); // one pack
        write(writer, 0, 3); // pack index
        write(writer, 0, 1); // no importance
        write(writer, 0, 1); // no channel reuse
        write(writer, 3, 3); // objects type label
        write(writer, 0, 5); // absolute distance
        write(writer, 0, 5); // pack start index
        write(writer, 0, 5); // one channel reference
        write(writer, 7, 5); // channel 7

        write(writer, 0, 5); // one channel
        write(writer, 7, 5); // channel format 7
        write(writer, 0, 1); // no channel gain
    }

    fn write_loudness(writer: &mut BitWriter, flags: u8) {
        write(writer, u64::from(flags), 6);
        for bit in (0..6).rev() {
            if flags & (1 << bit) != 0 {
                write(writer, bit as u64, 5);
            }
        }
    }

    fn write_direct_speaker_position(writer: &mut BitWriter) {
        write(writer, 0x81, 8);
        write(writer, 0x21, 6);
        write(writer, 9, 4);
        write(writer, 2, 2);
    }

    fn write_feature_rich_static_metadata(writer: &mut BitWriter) {
        write(writer, 1, 1); // VR extension present
        write(writer, 7, 3); // clamped to Basic L1

        write(writer, 0b1111, 4); // programme flags
        write(writer, 9, 4); // language
        write(writer, 17, 5); // max ducking depth
        write_loudness(writer, 0b11_1111);
        write(writer, 1, 1); // cartesian reference screen
        write(writer, 5, 3);
        write(writer, 0x80, 8);
        write(writer, 0x20, 6);
        write(writer, 8, 4);
        write(writer, 64, 7);
        write(writer, 3, 2); // four programme content references
        for reference in 0..4 {
            write(writer, reference, 2);
        }

        write(writer, 1, 2); // two content entries
        write(writer, 1, 2); // content 0 index
        write(writer, 0b1111, 4);
        write(writer, 7, 4); // content language
        write_loudness(writer, 0b10_1011);
        write(writer, 2, 2);
        write(writer, 5, 3);
        write(writer, 1, 2); // two complementary groups
        write(writer, 1, 3); // two complementary objects
        write(writer, 1, 3);
        write(writer, 2, 3);
        write(writer, 2, 3); // three complementary objects
        write(writer, 3, 3);
        write(writer, 4, 3);
        write(writer, 5, 3);
        write(writer, 1, 3); // two object references
        write(writer, 0, 3);
        write(writer, 1, 3);

        write(writer, 2, 2); // content 1 index
        write(writer, 0, 4); // no optional content fields
        write(writer, 0, 3); // one object reference
        write(writer, 1, 3);

        write(writer, 1, 3); // two object entries
        write(writer, 0, 3); // object 0 index
        write(writer, 0xff, 8); // every object flag
        write(writer, 6, 4); // object language
        write(writer, 1, 2);
        write(writer, 4, 3); // dialogue
        write(writer, 14, 4); // importance, clamped to 10
        for character in 0..24 {
            write(writer, 0x41 + character, 8);
        }
        write(writer, 0b111, 3); // all interaction flags
        write(writer, 1, 1); // logarithmic gain interaction
        write(writer, 7, 7);
        write(writer, 120, 7);
        write(writer, 1, 1); // cartesian position interaction
        for (value, width) in [(0x80, 8), (0x7f, 8), (0x20, 6), (0x1f, 6), (8, 4), (7, 4)] {
            write(writer, value, width);
        }
        write(writer, 1, 1); // object gain unit
        write(writer, 1, 1); // upper half
        write(writer, 45, 6);
        write(writer, 1, 3); // two pack references
        write(writer, 0, 3);
        write(writer, 1, 3);

        write(writer, 1, 3); // object 1 index
        write(writer, 0, 8); // no optional object fields
        write(writer, 0, 3); // one pack reference
        write(writer, 1, 3);

        write(writer, 1, 3); // two packs
        write(writer, 0, 3); // direct-speaker pack
        write(writer, 1, 1); // importance
        write(writer, 0, 1); // no reuse
        write(writer, 12, 4);
        write(writer, 1, 3); // direct-speaker type
        write(writer, 13, 5);
        write(writer, 63, 6); // explicit speaker positions
        write(writer, 2, 5); // start index
        write(writer, 0, 5); // one channel
        write(writer, 3, 5);

        write(writer, 1, 3); // matrix pack
        write(writer, 0, 1); // no importance
        write(writer, 1, 1); // channel reuse
        write(writer, 2, 3); // matrix type
        write(writer, 21, 5);
        write(writer, 5, 6); // pack format id
        write(writer, 2, 5); // three matrix output channels
        for _ in 0..3 {
            write_direct_speaker_position(writer);
        }
        write(writer, 0, 5); // one channel
        write(writer, 4, 5);
        write(writer, 7, 5); // transformed channel reference

        write(writer, 1, 5); // two channel formats
        write(writer, 3, 5);
        write(writer, 1, 1); // channel gain
        write(writer, 1, 1); // dB unit
        write(writer, 1, 1); // upper half
        write(writer, 39, 6);
        write_direct_speaker_position(writer);
        write(writer, 4, 5);
        write(writer, 0, 1); // no channel gain
        for coefficient in [3, 127, 255] {
            write(writer, coefficient, 8);
        }

        write(writer, 0, 3); // VR extension L1
        write(writer, 1, 1); // acoustic environment
        write(writer, 1, 1); // render info
        write(writer, 3, 3); // ambisonic order

        write(writer, 1, 1); // early reflection gain
        write(writer, 1, 1); // late reverb gain
        write(writer, 2, 2); // convolution reverb
        write(writer, 63, 7);
        write(writer, 95, 7);
        write(writer, 1, 1); // low-frequency processing
        write(writer, 17, 5); // convolution type
        write(writer, 1, 3); // two surfaces

        write(writer, 31, 5); // custom material
        for value in 0..16 {
            write(writer, value * 7, 7);
        }
        write(writer, 0, 5); // clamped up to four vertices
        for value in 0..4 * 3 {
            write(writer, value * 3, 7);
        }

        write(writer, 4, 5); // predefined material
        write(writer, 31, 5); // clamped down to eighteen vertices
        for value in 0..18 * 3 {
            write(writer, value * 2, 7);
        }

        write(writer, 1, 1); // target device
        write(writer, 9, 4); // HRTF type
        for headphone in 0..16 {
            write(writer, headphone * 5, 7);
        }
        write(writer, 0b111, 3); // EQ, DRC, and gain
        write(writer, 7, 3); // effect chain, clamped to 5
        write(writer, 15, 4); // clamped to eleven EQ bands
        for band in 0..11 {
            write(writer, band % 8, 3);
            write(writer, band * 7, 7);
            write(writer, band % 2, 1);
            write(writer, band * 3, 6);
            write(writer, band * 9, 7);
        }
        write(writer, 15, 4);
        write(writer, 14, 4);
        for value in [100, 90, 80, 70] {
            write(writer, value, 7);
        }
        write(writer, 64, 7); // effect gain
    }

    #[test]
    fn empty_metadata_prefix_restores_unaligned_audio_bits() {
        let mut writer = BitWriter::new();
        write(&mut writer, 0, 1);
        write(&mut writer, 0, 1);
        write(&mut writer, 0b1_0110_0100_1101, 13);
        let bit_len = writer.bit_len();
        let payload = writer.into_bytes();

        let mut parser = MetadataPayloadParser::new();
        let parsed = parser.parse(&payload, bit_len).unwrap();
        assert_eq!(parsed.summary().consumed_bits(), 2);
        assert_eq!(parsed.summary().audio_bits(), 13);
        assert!(!parsed.summary().has_static_metadata());
        assert!(!parsed.summary().has_dynamic_metadata());
        assert_eq!(parsed.audio_payload(), &[0b1011_0010, 0b0110_1000]);
    }

    #[test]
    fn populated_basic_l1_is_consumed_before_audio() {
        let mut writer = BitWriter::new();
        write(&mut writer, 1, 1);
        write_minimal_static_metadata(&mut writer);
        write(&mut writer, 0, 1);
        let metadata_bits = writer.bit_len();
        assert_eq!(metadata_bits, 90); // Avs3MetadataDec nextBitPos
        write(&mut writer, 0b101_0110_0011, 11);
        let payload_bits = writer.bit_len();
        let payload = writer.into_bytes();

        let mut parser = MetadataPayloadParser::new();
        let parsed = parser.parse(&payload, payload_bits).unwrap();
        let summary = parsed.summary();
        let static_metadata = summary.static_metadata().unwrap();
        assert_eq!(summary.consumed_bits(), metadata_bits);
        assert_eq!(summary.audio_bits(), 11);
        assert_eq!(static_metadata.basic_level(), 0);
        assert_eq!(static_metadata.contents(), 1);
        assert_eq!(static_metadata.objects(), 1);
        assert_eq!(static_metadata.packs(), 1);
        assert_eq!(static_metadata.channels(), 1);
        assert_eq!(static_metadata.vr_extension_level(), None);
        let values = parsed.metadata();
        assert!(values.has_static_metadata());
        assert!(!values.has_dynamic_metadata());
        let values = values.static_metadata().unwrap();
        assert_eq!(values.consumed_bits, 88);
        assert_eq!(values.basic.programme.content_references.as_slice(), &[0]);
        assert_eq!(values.basic.contents[0].index, 0);
        assert_eq!(values.basic.contents[0].object_references.as_slice(), &[0]);
        assert_eq!(values.basic.objects[0].index, 0);
        assert_eq!(values.basic.objects[0].pack_references.as_slice(), &[0]);
        assert_eq!(values.basic.packs[0].type_label, 3);
        assert_eq!(values.basic.packs[0].pack_format_start_index, Some(0));
        assert_eq!(values.basic.packs[0].channels[0].channel_index, 7);
        assert_eq!(values.basic.channels[0].index, 7);
        assert_eq!(parsed.audio_payload(), &[0b1010_1100, 0b0110_0000]);
    }

    #[test]
    fn dynamic_l1_l2_uses_header_object_count() {
        let mut writer = BitWriter::new();
        write(&mut writer, 0, 1);
        write(&mut writer, 1, 1);
        write(&mut writer, 1, 3); // dynamic L1 + L2

        write(&mut writer, 0, 1); // object 0 is active
        write(&mut writer, 17, 5);
        write(&mut writer, 0b0_1111, 5); // polar, all optional L1 fields
        write(&mut writer, 0, 8 + 6 + 4 + 7 + 5 + 4 + 7 + 7 + 1 + 4);
        write(&mut writer, 1, 1); // channel lock present
        write(&mut writer, 1, 1);
        write(&mut writer, 0, 4);
        write(&mut writer, 1, 1); // divergence present and non-zero
        write(&mut writer, 3, 4);
        write(&mut writer, 0, 6);
        write(&mut writer, 1, 1); // object screen ref
        write(&mut writer, 0, 1);
        write(&mut writer, 1, 1); // screen edge lock
        write(&mut writer, 2, 2);

        write(&mut writer, 1, 1); // object 1 is muted
        write(&mut writer, 2, 5);
        let metadata_bits = writer.bit_len();
        assert_eq!(metadata_bits, 97); // Avs3MetadataDec nextBitPos
        write(&mut writer, 0b110_1010, 7);
        let payload_bits = writer.bit_len();
        let payload = writer.into_bytes();

        let mut parser = MetadataPayloadParser::new();
        let parsed = parser
            .parse_with_object_count(&payload, payload_bits, 2)
            .unwrap();
        let dynamic = parsed.summary().dynamic_metadata().unwrap();
        assert_eq!(parsed.summary().consumed_bits(), metadata_bits);
        assert_eq!(dynamic.level(), 1);
        assert_eq!(dynamic.objects(), 2);
        let dynamic = parsed.metadata().dynamic_metadata().unwrap();
        assert_eq!(dynamic.consumed_bits, 95);
        assert_eq!(dynamic.level, 1);
        let active = &dynamic.objects[0];
        assert!(!active.muted);
        assert_eq!(active.transport_channel_reference, 17);
        let level1 = active.level1.unwrap();
        assert_eq!(level1.importance, Some(0));
        assert!(!level1.jump_position);
        assert_close(level1.gain.unwrap(), 0.0);
        assert_close(level1.diffuse.unwrap(), 0.0);
        match level1.position {
            DynamicObjectPosition::Polar {
                azimuth,
                elevation,
                distance,
            } => {
                assert_close(azimuth, -128.0 * RES_OBJECT_AZIMUTH);
                assert_close(elevation, -32.0 * RES_OBJECT_ELEVATION);
                assert_close(distance, 0.0);
            }
            DynamicObjectPosition::Cartesian { .. } => panic!("expected polar position"),
        }
        assert_eq!(
            level1.extent,
            Some(DynamicObjectExtent::Polar {
                width: 0.0,
                height: 0.0,
                depth: 0.0,
            })
        );
        let level2 = active.level2.unwrap();
        assert_eq!(
            level2.channel_lock,
            Some(ChannelLockMetadata {
                locked: true,
                maximum_distance: Some(0.0),
            })
        );
        assert_close(level2.object_divergence.unwrap().divergence, 0.2);
        assert_eq!(level2.object_divergence.unwrap().azimuth_range, Some(0.0));
        assert_eq!(level2.object_screen_reference, Some(false));
        assert_eq!(level2.screen_edge_lock, Some(2));
        let muted = &dynamic.objects[1];
        assert!(muted.muted);
        assert_eq!(muted.transport_channel_reference, 2);
        assert_eq!(muted.level1, None);
        assert_eq!(muted.level2, None);
        assert_eq!(parsed.audio_payload(), &[0b1101_0100]);
        assert_eq!(parsed.audio_bits(), 7);
    }

    #[test]
    fn alternate_metadata_value_branches_match_c_dequantization() {
        let mut writer = BitWriter::new();
        for (value, width) in [(0, 1), (3, 3), (200, 8), (45, 6), (12, 4), (100, 7)] {
            write(&mut writer, value, width);
        }
        let bits = writer.bit_len();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let screen = parse_programme_reference_screen(&mut reader).unwrap();
        assert_eq!(screen.aspect_ratio, 3);
        assert_eq!(
            screen.position,
            ProgrammeScreenPosition::Polar {
                azimuth: 72.0 * RES_STATIC_AZIMUTH,
                elevation: 45.0 * RES_STATIC_SCREEN_ELEVATION,
                distance: 12.0 * RES_STATIC_DISTANCE,
                width: 100.0 * RES_STATIC_POLAR_WIDTH,
            }
        );
        assert_eq!(reader.remaining(), 0);

        let mut writer = BitWriter::new();
        for (value, width) in [
            (0b011, 3),
            (0, 1),
            (64, 7),
            (32, 7),
            (0, 1),
            (255, 8),
            (255, 8),
            (63, 6),
            (63, 6),
            (15, 4),
            (8, 4),
        ] {
            write(&mut writer, value, width);
        }
        let bits = writer.bit_len();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let interaction = parse_audio_object_interaction(&mut reader).unwrap();
        assert!(!interaction.on_off_interact);
        let gain = interaction.gain.unwrap();
        assert_eq!(gain.unit, MetadataGainUnit::Linear);
        assert_close(gain.minimum, 64.0 * RES_STATIC_RANGE_MIN);
        assert_close(gain.maximum, 32.0 * RES_STATIC_RANGE_MAX + 1.0);
        assert_eq!(
            interaction.position,
            Some(PositionInteractionMetadata::Polar {
                azimuth_min: -180.0,
                azimuth_max: 180.0,
                elevation_min: -90.0,
                elevation_max: 90.0,
                distance_min: 1.0,
                distance_max: 8.0 * RES_STATIC_DISTANCE,
            })
        );
        assert_eq!(reader.remaining(), 0);

        let mut writer = BitWriter::new();
        write(&mut writer, 0, 1);
        write(&mut writer, 0, 1);
        write(&mut writer, 63, 6);
        let bytes = writer.into_bytes();
        let mut reader = BitReader::with_bit_len(&bytes, 8).unwrap();
        assert_eq!(
            parse_static_gain(&mut reader).unwrap(),
            MetadataGain {
                unit: MetadataGainUnit::Linear,
                value: 1.0,
            }
        );

        let mut writer = BitWriter::new();
        for (value, width) in [
            (2, 3),
            (0, 1),
            (0, 1),
            (4, 3),
            (0, 5),
            (2, 2),
            (15, 4),
            (1, 1),
            (3, 3),
            (5, 5),
            (0, 5),
            (9, 5),
        ] {
            write(&mut writer, value, width);
        }
        let bits = writer.bit_len();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let mut pack = AudioPackMetadata::default();
        let mut channel_references = [None; MAX_METADATA_CHANNELS];
        parse_audio_pack(&mut reader, &mut pack, &mut channel_references).unwrap();
        assert_eq!(pack.index, 2);
        assert_eq!(pack.type_label, 4);
        assert_eq!(
            pack.hoa,
            Some(HoaPackMetadata {
                normalization: 2,
                nfc_reference_distance: 1.0,
                screen_reference: true,
                order: 3,
            })
        );
        assert_eq!(pack.pack_format_start_index, Some(5));
        assert_eq!(pack.channels[0].channel_index, 9);
        assert_eq!(reader.remaining(), 0);

        let mut writer = BitWriter::new();
        write(&mut writer, 0b1_1111, 5);
        for (value, width) in [
            (128, 8),
            (32, 6),
            (8, 4),
            (127, 7),
            (31, 5),
            (15, 4),
            (127, 7),
            (127, 7),
            (1, 1),
            (15, 4),
        ] {
            write(&mut writer, value, width);
        }
        let bits = writer.bit_len();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let dynamic = parse_dynamic_l1(&mut reader).unwrap();
        assert_eq!(
            dynamic.position,
            DynamicObjectPosition::Cartesian {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }
        );
        assert_eq!(
            dynamic.extent,
            Some(DynamicObjectExtent::Cartesian {
                width_x: 1.0,
                height_y: 1.0,
                depth_z: 1.0,
            })
        );
        assert_eq!(dynamic.gain, Some(6.0));
        assert_eq!(dynamic.diffuse, Some(1.0));
        assert!(dynamic.jump_position);
        assert_eq!(dynamic.importance, Some(10));
        assert_eq!(reader.remaining(), 0);

        let mut writer = BitWriter::new();
        for (value, width) in [(1, 1), (0, 1), (1, 1), (0, 4), (0, 1), (0, 1)] {
            write(&mut writer, value, width);
        }
        let bits = writer.bit_len();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let level2 = parse_dynamic_l2(&mut reader).unwrap();
        assert_eq!(
            level2.channel_lock,
            Some(ChannelLockMetadata {
                locked: false,
                maximum_distance: None,
            })
        );
        assert_eq!(
            level2.object_divergence,
            Some(ObjectDivergenceMetadata {
                divergence: 0.0,
                azimuth_range: None,
            })
        );
        assert_eq!(level2.object_screen_reference, None);
        assert_eq!(level2.screen_edge_lock, None);
        assert_eq!(reader.remaining(), 0);
    }

    #[test]
    fn feature_rich_static_metadata_matches_c_bit_consumption() {
        let mut writer = BitWriter::new();
        write(&mut writer, 1, 1);
        write_feature_rich_static_metadata(&mut writer);
        write(&mut writer, 0, 1);
        let metadata_bits = writer.bit_len();
        assert_eq!(metadata_bits, 1_761); // Avs3MetadataDec nextBitPos
        write(&mut writer, 0b1_0010_1101, 9);
        let payload_bits = writer.bit_len();
        let payload = writer.into_bytes();

        let mut parser = MetadataPayloadParser::new();
        let parsed = parser.parse(&payload, payload_bits).unwrap();
        let static_metadata = parsed.summary().static_metadata().unwrap();
        assert_eq!(parsed.summary().consumed_bits(), metadata_bits);
        assert_eq!(static_metadata.basic_level(), 1);
        assert_eq!(static_metadata.contents(), 2);
        assert_eq!(static_metadata.objects(), 2);
        assert_eq!(static_metadata.packs(), 2);
        assert_eq!(static_metadata.channels(), 2);
        assert_eq!(static_metadata.vr_extension_level(), Some(0));

        let values = parsed.metadata().static_metadata().unwrap();
        assert_eq!(values.consumed_bits, 1_759);
        assert_eq!(values.basic_level, 1);
        let programme = &values.basic.programme;
        assert_eq!(programme.language, Some(9));
        assert_close(programme.max_ducking_depth.unwrap(), -34.0);
        let loudness = programme.loudness.unwrap();
        assert_close(
            loudness.integrated_loudness.unwrap(),
            5.0 * RES_STATIC_LOUDNESS,
        );
        assert_close(
            loudness.loudness_range.unwrap(),
            4.0 * RES_STATIC_LOUDNESS_RANGE + 10.0,
        );
        assert_close(loudness.max_true_peak.unwrap(), 3.0 * RES_STATIC_LOUDNESS);
        assert_close(loudness.max_momentary.unwrap(), 2.0 * RES_STATIC_LOUDNESS);
        assert_close(loudness.max_short_term.unwrap(), RES_STATIC_LOUDNESS);
        assert_close(loudness.dialogue_loudness.unwrap(), 0.0);
        assert_eq!(programme.content_references.as_slice(), &[0, 1, 2, 3]);
        let screen = programme.reference_screen.unwrap();
        assert_eq!(screen.aspect_ratio, 5);
        assert_eq!(
            screen.position,
            ProgrammeScreenPosition::Cartesian {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                width: 64.0 * RES_STATIC_CARTESIAN_WIDTH,
            }
        );

        let first_content = &values.basic.contents[0];
        assert_eq!(first_content.index, 1);
        assert_eq!(first_content.language, Some(7));
        assert_eq!(
            first_content.dialogue,
            Some(DialogueMetadata {
                attribute: 2,
                dialogue_type: 5,
            })
        );
        assert_eq!(first_content.complementary_object_groups.len(), 2);
        assert_eq!(
            first_content.complementary_object_groups[0]
                .object_references
                .as_slice(),
            &[1, 2]
        );
        assert_eq!(
            first_content.complementary_object_groups[1]
                .object_references
                .as_slice(),
            &[3, 4, 5]
        );
        assert_eq!(first_content.object_references.as_slice(), &[0, 1]);

        let first_object = &values.basic.objects[0];
        assert_eq!(first_object.index, 0);
        assert_eq!(first_object.language, Some(6));
        assert_eq!(first_object.importance, Some(10));
        assert!(first_object.disable_ducking);
        assert!(first_object.head_locked);
        assert!(first_object.muted);
        assert_eq!(first_object.name.as_ref().unwrap()[0], 0x41);
        assert_eq!(first_object.name.as_ref().unwrap()[23], 0x58);
        let interaction = first_object.interaction.unwrap();
        assert!(interaction.on_off_interact);
        let interaction_gain = interaction.gain.unwrap();
        assert_eq!(interaction_gain.unit, MetadataGainUnit::Decibels);
        assert_close(interaction_gain.minimum, 7.0 * RES_STATIC_RANGE_DB_MIN);
        assert_close(interaction_gain.maximum, 120.0 * RES_STATIC_RANGE_DB_MAX);
        assert_eq!(
            interaction.position,
            Some(PositionInteractionMetadata::Cartesian {
                x_min: 0.0,
                x_max: -RES_STATIC_X,
                y_min: 0.0,
                y_max: -RES_STATIC_Y,
                z_min: 0.0,
                z_max: -RES_STATIC_Z,
            })
        );
        let object_gain = first_object.gain.unwrap();
        assert_eq!(object_gain.unit, MetadataGainUnit::Decibels);
        assert_close(object_gain.value, 45.0 * RES_STATIC_GAIN_DB_HIGH);
        assert_eq!(first_object.pack_references.as_slice(), &[0, 1]);

        let direct_pack = &values.basic.packs[0];
        assert_eq!(direct_pack.importance, Some(10));
        assert_eq!(direct_pack.type_label, 1);
        assert_eq!(direct_pack.pack_format_id, Some(63));
        assert_eq!(direct_pack.pack_format_start_index, Some(2));
        assert_eq!(direct_pack.channels[0].channel_index, 3);
        let matrix_pack = &values.basic.packs[1];
        assert!(matrix_pack.channel_reuse);
        assert_eq!(matrix_pack.type_label, 2);
        assert_eq!(matrix_pack.pack_format_id, Some(5));
        assert_eq!(matrix_pack.matrix_output_positions.len(), 3);
        assert_eq!(
            matrix_pack.channels[0].transformed_channel_reference,
            Some(7)
        );
        let direct_channel = &values.basic.channels[0];
        assert_eq!(direct_channel.index, 3);
        assert_close(
            direct_channel.gain.unwrap().value,
            39.0 * RES_STATIC_GAIN_DB_HIGH,
        );
        assert_eq!(
            direct_channel
                .direct_speaker_position
                .unwrap()
                .screen_edge_lock,
            2
        );
        let matrix_channel = &values.basic.channels[1];
        assert_eq!(matrix_channel.index, 4);
        assert_eq!(matrix_channel.matrix_coefficients.len(), 3);
        assert_close(
            matrix_channel.matrix_coefficients[2],
            255.0 * RES_MATRIX_COEFFICIENT + 0.1,
        );

        let vr = values.vr_extension_l1.as_ref().unwrap();
        assert_eq!(vr.ambisonic_order, 3);
        let environment = vr.acoustic_environment.as_ref().unwrap();
        assert_close(
            environment.early_reflection_gain.unwrap(),
            63.0 * RES_VR_UNIT,
        );
        assert_close(environment.late_reverb_gain.unwrap(), 95.0 * RES_VR_UNIT);
        assert_eq!(environment.reverb_type, 2);
        assert!(environment.low_frequency_processing);
        assert_eq!(environment.convolution_reverb_type, Some(17));
        assert_eq!(environment.surfaces.len(), 2);
        assert_eq!(environment.surfaces[0].material, 31);
        assert_eq!(environment.surfaces[0].vertices.len(), 4);
        assert_close(
            environment.surfaces[0].absorption.unwrap()[7],
            98.0 * RES_VR_UNIT,
        );
        assert_close(
            environment.surfaces[0].scattering.unwrap()[7],
            105.0 * RES_VR_UNIT,
        );
        assert_eq!(environment.surfaces[1].material, 4);
        assert_eq!(environment.surfaces[1].vertices.len(), 18);
        assert_eq!(environment.surfaces[1].absorption, None);

        let render = vr.render_info.as_ref().unwrap();
        assert!(render.target_device);
        assert_eq!(render.hrtf_type, 9);
        assert_eq!(render.headphone_types[15], 75);
        assert_eq!(render.audio_effect.effect_chain, Some(5));
        assert_eq!(render.audio_effect.eq_bands.len(), 11);
        assert_eq!(render.audio_effect.eq_bands[10].eq_type, 2);
        assert_close(render.audio_effect.eq_bands[0].center_frequency, 20.0);
        assert_close(render.audio_effect.eq_bands[0].q, 0.1);
        assert_close(render.audio_effect.eq_bands[0].gain, -64.0 * RES_VR_EQ_GAIN);
        let drc = render.audio_effect.drc.unwrap();
        assert_close(drc.attack_time, 100.0);
        assert_close(drc.release_time, 14.0 * RES_VR_RELEASE + 50.0);
        assert_close(drc.threshold, 100.0 * RES_VR_THRESHOLD - 80.0);
        assert_close(drc.pre_gain, 26.0 * RES_VR_PRE_GAIN);
        assert_close(drc.post_gain, 80.0 * RES_VR_POST_GAIN);
        assert_close(drc.ratio, 70.0 * RES_VR_RATIO + 1.0);
        assert_eq!(render.audio_effect.gain, Some(0.0));
        assert_eq!(parsed.audio_payload(), &[0b1001_0110, 0b1000_0000]);
        assert_eq!(parsed.audio_bits(), 9);
    }

    #[test]
    fn metadata_reads_are_bounded() {
        let mut writer = BitWriter::new();
        write(&mut writer, 1, 1);
        write_minimal_static_metadata(&mut writer);
        write(&mut writer, 0, 1);
        let metadata_bits = writer.bit_len();
        let payload = writer.into_bytes();

        let mut parser = MetadataPayloadParser::new();
        for bit_len in 0..metadata_bits {
            assert!(
                parser.parse(&payload, bit_len).is_err(),
                "cut at bit {bit_len}"
            );
            assert!(parser.last_metadata().is_none());
        }
        assert_eq!(
            parser.parse(&payload, metadata_bits).unwrap().audio_bits(),
            0
        );
        assert!(parser.last_metadata().unwrap().has_static_metadata());
    }

    #[test]
    fn rejects_unmapped_channel_and_excessive_dynamic_object_count() {
        let mut writer = BitWriter::new();
        write(&mut writer, 1, 1);
        write_minimal_static_metadata(&mut writer);
        write(&mut writer, 0, 1);
        let bit_len = writer.bit_len();
        let mut payload = writer.into_bytes();
        // The channel-format index begins six bits before the dynamic flag.
        let channel_index_position = bit_len - 7;
        for offset in 0..5 {
            let absolute = channel_index_position + offset;
            payload[absolute / 8] &= !(1 << (7 - absolute % 8));
        }

        let mut parser = MetadataPayloadParser::new();
        assert_eq!(
            parser.parse(&payload, bit_len).unwrap_err(),
            MetadataError::UnmappedChannelFormat { index: 0 }
        );

        assert_eq!(
            parser
                .parse_with_object_count(&[0b0100_0000], 2, 33)
                .unwrap_err(),
            MetadataError::TooManyDynamicObjects {
                objects: 33,
                limit: 32,
            }
        );
    }

    #[test]
    fn rejects_audio_larger_than_the_fixed_workspace() {
        let payload = vec![0; MAX_PAYLOAD_BYTES + 1];
        let mut parser = MetadataPayloadParser::new();
        assert_eq!(
            parser.parse(&payload, payload.len() * 8).unwrap_err(),
            MetadataError::PayloadTooLarge {
                bytes: MAX_PAYLOAD_BYTES + 1,
                limit: MAX_PAYLOAD_BYTES,
            }
        );
    }
}
