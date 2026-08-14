use core::array;

#[derive(Debug, Clone, PartialEq)]
pub struct MetadataList<T, const CAPACITY: usize> {
    len: usize,
    values: [T; CAPACITY],
}

impl<T, const CAPACITY: usize> MetadataList<T, CAPACITY> {
    pub fn as_slice(&self) -> &[T] {
        &self.values[..self.len]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }

    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    pub const fn capacity(&self) -> usize {
        CAPACITY
    }
}

impl<T: Default, const CAPACITY: usize> MetadataList<T, CAPACITY> {
    pub(crate) fn prepare(&mut self, len: usize) -> &mut [T] {
        assert!(len <= CAPACITY);
        self.len = len;
        for value in &mut self.values[..len] {
            *value = T::default();
        }
        &mut self.values[..len]
    }
}

impl<T: Default, const CAPACITY: usize> Default for MetadataList<T, CAPACITY> {
    fn default() -> Self {
        Self {
            len: 0,
            values: array::from_fn(|_| T::default()),
        }
    }
}

impl<'a, T, const CAPACITY: usize> IntoIterator for &'a MetadataList<T, CAPACITY> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, const CAPACITY: usize> core::ops::Index<usize> for MetadataList<T, CAPACITY> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataGainUnit {
    Linear,
    Decibels,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetadataGain {
    pub unit: MetadataGainUnit,
    pub value: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LoudnessMetadata {
    pub integrated_loudness: Option<f32>,
    pub loudness_range: Option<f32>,
    pub max_true_peak: Option<f32>,
    pub max_momentary: Option<f32>,
    pub max_short_term: Option<f32>,
    pub dialogue_loudness: Option<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DirectSpeakerPosition {
    pub azimuth: f32,
    pub elevation: f32,
    pub distance: f32,
    pub screen_edge_lock: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProgrammeScreenPosition {
    Polar {
        azimuth: f32,
        elevation: f32,
        distance: f32,
        width: f32,
    },
    Cartesian {
        x: f32,
        y: f32,
        z: f32,
        width: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgrammeReferenceScreen {
    pub aspect_ratio: u8,
    pub position: ProgrammeScreenPosition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DialogueMetadata {
    pub attribute: u8,
    pub dialogue_type: u8,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioProgrammeMetadata {
    pub language: Option<u8>,
    pub max_ducking_depth: Option<f32>,
    pub loudness: Option<LoudnessMetadata>,
    pub reference_screen: Option<ProgrammeReferenceScreen>,
    pub content_references: MetadataList<u8, 4>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ComplementaryObjectGroup {
    pub object_references: MetadataList<u8, 8>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioContentMetadata {
    pub index: u8,
    pub language: Option<u8>,
    pub loudness: Option<LoudnessMetadata>,
    pub dialogue: Option<DialogueMetadata>,
    pub complementary_object_groups: MetadataList<ComplementaryObjectGroup, 4>,
    pub object_references: MetadataList<u8, 8>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainInteractionMetadata {
    pub unit: MetadataGainUnit,
    pub minimum: f32,
    pub maximum: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionInteractionMetadata {
    Polar {
        azimuth_min: f32,
        azimuth_max: f32,
        elevation_min: f32,
        elevation_max: f32,
        distance_min: f32,
        distance_max: f32,
    },
    Cartesian {
        x_min: f32,
        x_max: f32,
        y_min: f32,
        y_max: f32,
        z_min: f32,
        z_max: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioObjectInteractionMetadata {
    pub on_off_interact: bool,
    pub gain: Option<GainInteractionMetadata>,
    pub position: Option<PositionInteractionMetadata>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioObjectMetadata {
    pub index: u8,
    pub language: Option<u8>,
    pub dialogue: Option<DialogueMetadata>,
    pub importance: Option<u8>,
    pub disable_ducking: bool,
    pub head_locked: bool,
    pub muted: bool,
    pub name: Option<[u8; 24]>,
    pub interaction: Option<AudioObjectInteractionMetadata>,
    pub gain: Option<MetadataGain>,
    pub pack_references: MetadataList<u8, 8>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoaPackMetadata {
    pub normalization: u8,
    pub nfc_reference_distance: f32,
    pub screen_reference: bool,
    pub order: u8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PackChannelReference {
    pub channel_index: u8,
    pub transformed_channel_reference: Option<u8>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioPackMetadata {
    pub index: u8,
    pub importance: Option<u8>,
    pub channel_reuse: bool,
    pub type_label: u8,
    pub absolute_distance: f32,
    pub hoa: Option<HoaPackMetadata>,
    pub pack_format_id: Option<u8>,
    pub matrix_output_positions: MetadataList<DirectSpeakerPosition, 32>,
    pub pack_format_start_index: Option<u8>,
    pub channels: MetadataList<PackChannelReference, 32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioChannelMetadata {
    pub index: u8,
    pub gain: Option<MetadataGain>,
    pub direct_speaker_position: Option<DirectSpeakerPosition>,
    pub matrix_coefficients: MetadataList<f32, 32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BasicMetadata {
    pub programme: AudioProgrammeMetadata,
    pub contents: MetadataList<AudioContentMetadata, 4>,
    pub objects: MetadataList<AudioObjectMetadata, 8>,
    pub packs: MetadataList<AudioPackMetadata, 8>,
    pub channels: MetadataList<AudioChannelMetadata, 32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VrEqBandMetadata {
    pub eq_type: u8,
    pub center_frequency: f32,
    pub q: f32,
    pub gain: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VrDrcMetadata {
    pub attack_time: f32,
    pub release_time: f32,
    pub threshold: f32,
    pub pre_gain: f32,
    pub post_gain: f32,
    pub ratio: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VrAudioEffectMetadata {
    pub effect_chain: Option<u8>,
    pub eq_bands: MetadataList<VrEqBandMetadata, 11>,
    pub drc: Option<VrDrcMetadata>,
    pub gain: Option<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VrVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VrSurfaceMetadata {
    pub material: u8,
    pub absorption: Option<[f32; 8]>,
    pub scattering: Option<[f32; 8]>,
    pub vertices: MetadataList<VrVertex, 32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VrAcousticEnvironmentMetadata {
    pub early_reflection_gain: Option<f32>,
    pub late_reverb_gain: Option<f32>,
    pub reverb_type: u8,
    pub low_frequency_processing: bool,
    pub convolution_reverb_type: Option<u8>,
    pub surfaces: MetadataList<VrSurfaceMetadata, 8>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VrRenderInfoMetadata {
    pub target_device: bool,
    pub hrtf_type: u8,
    pub headphone_types: [u8; 16],
    pub audio_effect: VrAudioEffectMetadata,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VrExtensionMetadata {
    pub ambisonic_order: u8,
    pub acoustic_environment: Option<VrAcousticEnvironmentMetadata>,
    pub render_info: Option<VrRenderInfoMetadata>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StaticMetadata {
    pub consumed_bits: usize,
    pub basic_level: u8,
    pub basic: BasicMetadata,
    pub vr_extension_level: Option<u8>,
    pub vr_extension_l1: Option<VrExtensionMetadata>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynamicObjectPosition {
    Polar {
        azimuth: f32,
        elevation: f32,
        distance: f32,
    },
    Cartesian {
        x: f32,
        y: f32,
        z: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynamicObjectExtent {
    Polar {
        width: f32,
        height: f32,
        depth: f32,
    },
    Cartesian {
        width_x: f32,
        height_y: f32,
        depth_z: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicLevel1Metadata {
    pub position: DynamicObjectPosition,
    pub extent: Option<DynamicObjectExtent>,
    pub gain: Option<f32>,
    pub diffuse: Option<f32>,
    pub jump_position: bool,
    pub importance: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelLockMetadata {
    pub locked: bool,
    pub maximum_distance: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectDivergenceMetadata {
    pub divergence: f32,
    pub azimuth_range: Option<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DynamicLevel2Metadata {
    pub channel_lock: Option<ChannelLockMetadata>,
    pub object_divergence: Option<ObjectDivergenceMetadata>,
    pub object_screen_reference: Option<bool>,
    pub screen_edge_lock: Option<u8>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DynamicObjectMetadata {
    pub muted: bool,
    pub transport_channel_reference: u8,
    pub level1: Option<DynamicLevel1Metadata>,
    pub level2: Option<DynamicLevel2Metadata>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DynamicMetadata {
    pub consumed_bits: usize,
    pub level: u8,
    pub objects: MetadataList<DynamicObjectMetadata, 32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameMetadata {
    pub(crate) has_static_metadata: bool,
    pub(crate) static_metadata: StaticMetadata,
    pub(crate) has_dynamic_metadata: bool,
    pub(crate) dynamic_metadata: DynamicMetadata,
}

impl FrameMetadata {
    pub fn has_static_metadata(&self) -> bool {
        self.has_static_metadata
    }

    pub fn has_dynamic_metadata(&self) -> bool {
        self.has_dynamic_metadata
    }

    pub fn static_metadata(&self) -> Option<&StaticMetadata> {
        self.has_static_metadata.then_some(&self.static_metadata)
    }

    pub fn dynamic_metadata(&self) -> Option<&DynamicMetadata> {
        self.has_dynamic_metadata.then_some(&self.dynamic_metadata)
    }

    pub(crate) fn clear_presence(&mut self) {
        self.has_static_metadata = false;
        self.has_dynamic_metadata = false;
    }
}
