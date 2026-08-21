use core::fmt;

use crate::bitstream::BitReader;
pub use crate::error::HeaderError;

use crate::error::BitstreamError;

pub const MAX_HEADER_BYTES: usize = 9;
pub const MAX_PAYLOAD_BYTES: usize = 12_300;
pub const MAX_CHANNELS: u8 = 16;

const SYNC_WORD: u16 = 0x0fff;
const FRAME_SAMPLES: u32 = 1_024;

const SAMPLE_RATES: [u32; 9] = [
    192_000, 96_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 8_000,
];

const BITRATE_MONO: &[u32] = &[
    16_000, 32_000, 44_000, 56_000, 64_000, 72_000, 80_000, 96_000, 128_000, 144_000, 164_000,
    192_000,
];
const BITRATE_STEREO: &[u32] = &[
    24_000, 32_000, 48_000, 64_000, 80_000, 96_000, 128_000, 144_000, 192_000, 256_000, 320_000,
];
const BITRATE_MC_5_1: &[u32] = &[
    192_000, 256_000, 320_000, 384_000, 448_000, 512_000, 640_000, 720_000, 144_000, 96_000,
    128_000, 160_000,
];
const BITRATE_MC_7_1: &[u32] = &[
    192_000, 480_000, 256_000, 384_000, 576_000, 640_000, 128_000, 160_000,
];
const BITRATE_MC_4_0: &[u32] = &[48_000, 96_000, 128_000, 192_000, 256_000];
const BITRATE_MC_5_1_2: &[u32] = &[152_000, 320_000, 480_000, 576_000];
const BITRATE_MC_5_1_4: &[u32] = &[176_000, 384_000, 576_000, 704_000, 256_000, 448_000];
const BITRATE_MC_7_1_2: &[u32] = &[216_000, 480_000, 576_000, 384_000, 768_000];
const BITRATE_MC_7_1_4: &[u32] = &[240_000, 608_000, 384_000, 512_000, 832_000];
const BITRATE_FOA: &[u32] = &[48_000, 96_000, 128_000, 192_000, 256_000];
const BITRATE_HOA2: &[u32] = &[
    192_000, 256_000, 320_000, 384_000, 480_000, 512_000, 640_000,
];
const BITRATE_HOA3: &[u32] = &[256_000, 320_000, 384_000, 512_000, 640_000, 896_000];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodecId {
    Avs3P3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NnType {
    Main,
    LowComplexity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecProfile {
    ChannelBased,
    Mixed,
    Hoa,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepth {
    Eight,
    Sixteen,
    TwentyFour,
}

impl BitDepth {
    pub const fn bits(self) -> u8 {
        match self {
            Self::Eight => 8,
            Self::Sixteen => 16,
            Self::TwentyFour => 24,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChannelConfig {
    Mono = 0,
    Stereo = 1,
    Mc5_1 = 2,
    Mc7_1 = 3,
    Mc10_2 = 4,
    Mc22_2 = 5,
    Mc4_0 = 6,
    Mc5_1_2 = 7,
    Mc5_1_4 = 8,
    Mc7_1_2 = 9,
    Mc7_1_4 = 10,
    Hoa1 = 11,
    Hoa2 = 12,
    Hoa3 = 13,
}

impl ChannelConfig {
    pub fn from_index(value: u8) -> Result<Self, HeaderError> {
        match value {
            0 => Ok(Self::Mono),
            1 => Ok(Self::Stereo),
            2 => Ok(Self::Mc5_1),
            3 => Ok(Self::Mc7_1),
            4 => Ok(Self::Mc10_2),
            5 => Ok(Self::Mc22_2),
            6 => Ok(Self::Mc4_0),
            7 => Ok(Self::Mc5_1_2),
            8 => Ok(Self::Mc5_1_4),
            9 => Ok(Self::Mc7_1_2),
            10 => Ok(Self::Mc7_1_4),
            11 => Ok(Self::Hoa1),
            12 => Ok(Self::Hoa2),
            13 => Ok(Self::Hoa3),
            _ => Err(HeaderError::InvalidChannelConfig(value)),
        }
    }

    pub const fn index(self) -> u8 {
        self as u8
    }

    pub const fn channels(self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Mc5_1 => 6,
            Self::Mc7_1 => 8,
            Self::Mc10_2 => 12,
            Self::Mc22_2 => 24,
            Self::Mc4_0 => 4,
            Self::Mc5_1_2 => 8,
            Self::Mc5_1_4 | Self::Mc7_1_2 => 10,
            Self::Mc7_1_4 => 12,
            Self::Hoa1 => 4,
            Self::Hoa2 => 9,
            Self::Hoa3 => 16,
        }
    }

    pub const fn has_lfe(self) -> bool {
        !matches!(
            self,
            Self::Mono | Self::Stereo | Self::Mc4_0 | Self::Hoa1 | Self::Hoa2 | Self::Hoa3
        )
    }

    fn bitrate_table(self) -> Option<&'static [u32]> {
        match self {
            Self::Mono => Some(BITRATE_MONO),
            Self::Stereo => Some(BITRATE_STEREO),
            Self::Mc5_1 => Some(BITRATE_MC_5_1),
            Self::Mc7_1 => Some(BITRATE_MC_7_1),
            // The C reference lists these configurations but gives them NULL
            // bitrate tables. Rejecting them is preferable to dereferencing it.
            Self::Mc10_2 | Self::Mc22_2 => None,
            Self::Mc4_0 => Some(BITRATE_MC_4_0),
            Self::Mc5_1_2 => Some(BITRATE_MC_5_1_2),
            Self::Mc5_1_4 => Some(BITRATE_MC_5_1_4),
            Self::Mc7_1_2 => Some(BITRATE_MC_7_1_2),
            Self::Mc7_1_4 => Some(BITRATE_MC_7_1_4),
            Self::Hoa1 => Some(BITRATE_FOA),
            Self::Hoa2 => Some(BITRATE_HOA2),
            Self::Hoa3 => Some(BITRATE_HOA3),
        }
    }

    fn bitrate(self, index: u8) -> Result<u32, HeaderError> {
        self.bitrate_table()
            .and_then(|table| table.get(usize::from(index)))
            .copied()
            .ok_or(HeaderError::InvalidBitrateIndex {
                config: self.index(),
                index,
            })
    }
}

impl fmt::Display for ChannelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Mono => "mono",
            Self::Stereo => "stereo",
            Self::Mc5_1 => "5.1",
            Self::Mc7_1 => "7.1",
            Self::Mc10_2 => "10.2",
            Self::Mc22_2 => "22.2",
            Self::Mc4_0 => "4.0",
            Self::Mc5_1_2 => "5.1.2",
            Self::Mc5_1_4 => "5.1.4",
            Self::Mc7_1_2 => "7.1.2",
            Self::Mc7_1_4 => "7.1.4",
            Self::Hoa1 => "HOA1",
            Self::Hoa2 => "HOA2",
            Self::Hoa3 => "HOA3",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundBedType {
    ObjectsOnly,
    ChannelBed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub codec_id: AudioCodecId,
    pub nn_type: NnType,
    pub profile: CodecProfile,
    pub sample_rate: u32,
    pub bit_depth: BitDepth,
    pub channel_config: Option<ChannelConfig>,
    pub sound_bed_type: Option<SoundBedType>,
    pub hoa_order: Option<u8>,
    pub objects: u8,
    pub bed_channels: u8,
    pub channels: u8,
    pub has_lfe: bool,
    pub bed_bitrate: Option<u32>,
    pub object_bitrate: Option<u32>,
    pub bitrate: u32,
    pub crc: u16,
    pub header_len: usize,
    pub payload_bits: usize,
    pub payload_len: usize,
    pub frame_len: usize,
    pub samples_per_channel: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderInfo {
    pub offset: usize,
    pub header: FrameHeader,
}

pub fn parse_header(input: &[u8]) -> Result<HeaderInfo, HeaderError> {
    if input.len() < 2 {
        return Err(HeaderError::NeedMoreData {
            needed: 2,
            available: input.len(),
        });
    }

    let mut first_error = None;
    for offset in 0..input.len().saturating_sub(1) {
        if input[offset] != 0xff || input[offset + 1] & 0xf0 != 0xf0 {
            continue;
        }
        match parse_header_at(&input[offset..], offset) {
            Ok(header) => return Ok(HeaderInfo { offset, header }),
            Err(error @ HeaderError::NeedMoreData { .. }) => return Err(error),
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }

    if input.last() == Some(&0xff) {
        return Err(HeaderError::NeedMoreData {
            needed: input.len() + 1,
            available: input.len(),
        });
    }
    Err(first_error.unwrap_or(HeaderError::NoSync))
}

pub(crate) fn parse_header_at(
    input: &[u8],
    stream_offset: usize,
) -> Result<FrameHeader, HeaderError> {
    let mut bits = BitReader::new(input);
    let sync = read(&mut bits, 12, input.len())? as u16;
    if sync != SYNC_WORD {
        return Err(HeaderError::InvalidSync {
            offset: stream_offset,
            value: sync,
        });
    }

    let codec = read(&mut bits, 4, input.len())? as u8;
    if codec != 2 {
        return Err(HeaderError::UnsupportedCodec {
            offset: stream_offset,
            value: codec,
        });
    }
    if read(&mut bits, 1, input.len())? != 0 {
        return Err(HeaderError::AncillaryData {
            offset: stream_offset,
        });
    }

    let nn_type = match read(&mut bits, 3, input.len())? as u8 {
        0 => NnType::Main,
        1 => NnType::LowComplexity,
        value => return Err(HeaderError::UnsupportedNnType(value)),
    };
    let profile = match read(&mut bits, 3, input.len())? as u8 {
        0 => CodecProfile::ChannelBased,
        1 => CodecProfile::Mixed,
        2 => CodecProfile::Hoa,
        value => {
            return Err(HeaderError::UnsupportedProfile {
                offset: stream_offset,
                value,
            });
        }
    };
    let sample_rate_index = read(&mut bits, 4, input.len())? as u8;
    let sample_rate = SAMPLE_RATES
        .get(usize::from(sample_rate_index))
        .copied()
        .ok_or(HeaderError::InvalidSamplingRateIndex(sample_rate_index))?;
    let crc_high = read(&mut bits, 8, input.len())? as u16;

    let mut channel_config = None;
    let mut sound_bed_type = None;
    let mut hoa_order = None;
    let mut objects = 0_u8;
    let mut bed_channels = 0_u8;
    let mut bed_bitrate = None;
    let mut object_bitrate = None;
    let channels;
    let mut has_lfe = false;
    let bitrate;
    let header_budget;

    match profile {
        CodecProfile::ChannelBased => {
            let raw_config = read(&mut bits, 7, input.len())? as u8;
            let config = ChannelConfig::from_index(raw_config)?;
            if matches!(
                config,
                ChannelConfig::Hoa1 | ChannelConfig::Hoa2 | ChannelConfig::Hoa3
            ) {
                return Err(HeaderError::InvalidChannelConfig(raw_config));
            }
            channel_config = Some(config);
            bed_channels = config.channels();
            channels = bed_channels;
            has_lfe = config.has_lfe();
            header_budget = 56_u32;

            let resolution = read(&mut bits, 2, input.len())? as u8;
            let bit_depth = bit_depth(resolution)?;
            let bitrate_index = read(&mut bits, 4, input.len())? as u8;
            bitrate = config.bitrate(bitrate_index)?;
            bed_bitrate = Some(bitrate);
            let crc_low = read(&mut bits, 8, input.len())? as u16;
            finish_header(HeaderParts {
                nn_type,
                profile,
                sample_rate,
                bit_depth,
                channel_config,
                sound_bed_type,
                hoa_order,
                objects,
                bed_channels,
                channels,
                has_lfe,
                bed_bitrate,
                object_bitrate,
                bitrate,
                crc: (crc_high << 8) | crc_low,
                syntax_bits: bits.position(),
                header_budget,
            })
        }
        CodecProfile::Mixed => {
            let raw_bed_type = read(&mut bits, 2, input.len())? as u8;
            let bed_type = match raw_bed_type {
                0 => SoundBedType::ObjectsOnly,
                1 => SoundBedType::ChannelBed,
                value => return Err(HeaderError::InvalidSoundBedType(value)),
            };
            sound_bed_type = Some(bed_type);

            match bed_type {
                SoundBedType::ObjectsOnly => {
                    objects = object_count(read(&mut bits, 7, input.len())? as u8)?;
                    let object_bitrate_index = read(&mut bits, 4, input.len())? as u8;
                    let per_object_bitrate = ChannelConfig::Mono.bitrate(object_bitrate_index)?;
                    object_bitrate = Some(per_object_bitrate);
                    channels = objects;
                    bitrate = per_object_bitrate
                        .checked_mul(u32::from(objects))
                        .ok_or(HeaderError::ArithmeticOverflow)?;
                    header_budget = 64;
                }
                SoundBedType::ChannelBed => {
                    let raw_config = read(&mut bits, 7, input.len())? as u8;
                    let config = ChannelConfig::from_index(raw_config)?;
                    if matches!(
                        config,
                        ChannelConfig::Mono
                            | ChannelConfig::Hoa1
                            | ChannelConfig::Hoa2
                            | ChannelConfig::Hoa3
                    ) {
                        return Err(HeaderError::InvalidChannelConfig(raw_config));
                    }
                    channel_config = Some(config);
                    bed_channels = config.channels();
                    let bed_bitrate_index = read(&mut bits, 4, input.len())? as u8;
                    let channel_bed_bitrate = config.bitrate(bed_bitrate_index)?;
                    bed_bitrate = Some(channel_bed_bitrate);
                    objects = object_count(read(&mut bits, 7, input.len())? as u8)?;
                    let object_bitrate_index = read(&mut bits, 4, input.len())? as u8;
                    let per_object_bitrate = ChannelConfig::Mono.bitrate(object_bitrate_index)?;
                    object_bitrate = Some(per_object_bitrate);
                    channels = bed_channels
                        .checked_add(objects)
                        .ok_or(HeaderError::ArithmeticOverflow)?;
                    if channels > MAX_CHANNELS {
                        return Err(HeaderError::InvalidObjectCount(objects));
                    }
                    bitrate = per_object_bitrate
                        .checked_mul(u32::from(objects))
                        .and_then(|value| value.checked_add(channel_bed_bitrate))
                        .ok_or(HeaderError::ArithmeticOverflow)?;
                    has_lfe = config.has_lfe();
                    header_budget = 72;
                }
            }

            let resolution = read(&mut bits, 2, input.len())? as u8;
            let bit_depth = bit_depth(resolution)?;
            let crc_low = read(&mut bits, 8, input.len())? as u16;
            finish_header(HeaderParts {
                nn_type,
                profile,
                sample_rate,
                bit_depth,
                channel_config,
                sound_bed_type,
                hoa_order,
                objects,
                bed_channels,
                channels,
                has_lfe,
                bed_bitrate,
                object_bitrate,
                bitrate,
                crc: (crc_high << 8) | crc_low,
                syntax_bits: bits.position(),
                header_budget,
            })
        }
        CodecProfile::Hoa => {
            let order = (read(&mut bits, 4, input.len())? as u8)
                .checked_add(1)
                .ok_or(HeaderError::ArithmeticOverflow)?;
            let config = match order {
                1 => ChannelConfig::Hoa1,
                2 => ChannelConfig::Hoa2,
                3 => ChannelConfig::Hoa3,
                value => return Err(HeaderError::InvalidHoaOrder(value)),
            };
            channel_config = Some(config);
            hoa_order = Some(order);
            bed_channels = config.channels();
            channels = bed_channels;
            header_budget = 56;

            let resolution = read(&mut bits, 2, input.len())? as u8;
            let bit_depth = bit_depth(resolution)?;
            let bitrate_index = read(&mut bits, 4, input.len())? as u8;
            bitrate = config.bitrate(bitrate_index)?;
            let crc_low = read(&mut bits, 8, input.len())? as u16;
            finish_header(HeaderParts {
                nn_type,
                profile,
                sample_rate,
                bit_depth,
                channel_config,
                sound_bed_type,
                hoa_order,
                objects,
                bed_channels,
                channels,
                has_lfe,
                bed_bitrate,
                object_bitrate,
                bitrate,
                crc: (crc_high << 8) | crc_low,
                syntax_bits: bits.position(),
                header_budget,
            })
        }
    }
}

#[derive(Debug)]
struct HeaderParts {
    nn_type: NnType,
    profile: CodecProfile,
    sample_rate: u32,
    bit_depth: BitDepth,
    channel_config: Option<ChannelConfig>,
    sound_bed_type: Option<SoundBedType>,
    hoa_order: Option<u8>,
    objects: u8,
    bed_channels: u8,
    channels: u8,
    has_lfe: bool,
    bed_bitrate: Option<u32>,
    object_bitrate: Option<u32>,
    bitrate: u32,
    crc: u16,
    syntax_bits: usize,
    header_budget: u32,
}

fn finish_header(parts: HeaderParts) -> Result<FrameHeader, HeaderError> {
    if parts.channels == 0 || parts.channels > MAX_CHANNELS {
        return Err(HeaderError::InvalidChannelConfig(
            parts.channel_config.map_or(u8::MAX, ChannelConfig::index),
        ));
    }
    let total_bits = u64::from(parts.bitrate)
        .checked_mul(u64::from(FRAME_SAMPLES))
        .ok_or(HeaderError::ArithmeticOverflow)?
        / u64::from(parts.sample_rate);
    let payload_bits = total_bits
        .checked_sub(u64::from(parts.header_budget))
        .ok_or(HeaderError::ArithmeticOverflow)?;
    if payload_bits == 0 {
        return Err(HeaderError::ArithmeticOverflow);
    }
    let payload_bits =
        usize::try_from(payload_bits).map_err(|_| HeaderError::ArithmeticOverflow)?;
    let payload_len = payload_bits
        .checked_add(7)
        .ok_or(HeaderError::ArithmeticOverflow)?
        / 8;
    if payload_len > MAX_PAYLOAD_BYTES {
        return Err(HeaderError::PayloadTooLarge {
            size: payload_len,
            limit: MAX_PAYLOAD_BYTES,
        });
    }
    let header_len = parts
        .syntax_bits
        .checked_add(7)
        .ok_or(HeaderError::ArithmeticOverflow)?
        / 8;
    let frame_len = header_len
        .checked_add(payload_len)
        .ok_or(HeaderError::ArithmeticOverflow)?;

    Ok(FrameHeader {
        codec_id: AudioCodecId::Avs3P3,
        nn_type: parts.nn_type,
        profile: parts.profile,
        sample_rate: parts.sample_rate,
        bit_depth: parts.bit_depth,
        channel_config: parts.channel_config,
        sound_bed_type: parts.sound_bed_type,
        hoa_order: parts.hoa_order,
        objects: parts.objects,
        bed_channels: parts.bed_channels,
        channels: parts.channels,
        has_lfe: parts.has_lfe,
        bed_bitrate: parts.bed_bitrate,
        object_bitrate: parts.object_bitrate,
        bitrate: parts.bitrate,
        crc: parts.crc,
        header_len,
        payload_bits,
        payload_len,
        frame_len,
        samples_per_channel: FRAME_SAMPLES,
    })
}

fn read(
    bits: &mut BitReader<'_>,
    width: usize,
    available_bytes: usize,
) -> Result<u64, HeaderError> {
    bits.read_bits(width).map_err(|error| match error {
        BitstreamError::UnexpectedEof {
            position,
            requested,
            ..
        } => {
            let needed_bits = position.saturating_add(requested);
            HeaderError::NeedMoreData {
                needed: needed_bits.saturating_add(7) / 8,
                available: available_bytes,
            }
        }
        other => HeaderError::Bitstream(other),
    })
}

fn bit_depth(value: u8) -> Result<BitDepth, HeaderError> {
    match value {
        0 => Ok(BitDepth::Eight),
        1 => Ok(BitDepth::Sixteen),
        2 => Ok(BitDepth::TwentyFour),
        _ => Err(HeaderError::InvalidBitDepth(value)),
    }
}

fn object_count(encoded: u8) -> Result<u8, HeaderError> {
    let count = encoded
        .checked_add(1)
        .ok_or(HeaderError::InvalidObjectCount(encoded))?;
    if count > MAX_CHANNELS {
        return Err(HeaderError::InvalidObjectCount(count));
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;

    fn base_header(profile: u64) -> BitWriter {
        let mut writer = BitWriter::new();
        writer.write_bits(SYNC_WORD.into(), 12).unwrap();
        writer.write_bits(2, 4).unwrap();
        writer.write_bits(0, 1).unwrap();
        writer.write_bits(0, 3).unwrap();
        writer.write_bits(profile, 3).unwrap();
        writer.write_bits(2, 4).unwrap();
        writer.write_bits(0xab, 8).unwrap();
        writer
    }

    #[test]
    fn parses_mono_header() {
        let mut writer = base_header(0);
        writer.write_bits(0, 7).unwrap();
        writer.write_bits(1, 2).unwrap();
        writer.write_bits(4, 4).unwrap();
        writer.write_bits(0xcd, 8).unwrap();
        let header = parse_header_at(&writer.into_bytes(), 0).unwrap();
        assert_eq!(header.channel_config, Some(ChannelConfig::Mono));
        assert_eq!(header.sample_rate, 48_000);
        assert_eq!(header.channels, 1);
        assert_eq!(header.bed_bitrate, Some(64_000));
        assert_eq!(header.object_bitrate, None);
        assert_eq!(header.crc, 0xabcd);
        assert_eq!(header.header_len, 7);
    }

    #[test]
    fn parses_object_only_mix_header() {
        let mut writer = base_header(1);
        writer.write_bits(0, 2).unwrap();
        writer.write_bits(2, 7).unwrap();
        writer.write_bits(4, 4).unwrap();
        writer.write_bits(1, 2).unwrap();
        writer.write_bits(0xcd, 8).unwrap();
        let header = parse_header_at(&writer.into_bytes(), 0).unwrap();
        assert_eq!(header.sound_bed_type, Some(SoundBedType::ObjectsOnly));
        assert_eq!(header.objects, 3);
        assert_eq!(header.channels, 3);
        assert_eq!(header.header_len, 8);
        assert_eq!(header.bitrate, 192_000);
        assert_eq!(header.bed_bitrate, None);
        assert_eq!(header.object_bitrate, Some(64_000));
    }

    #[test]
    fn parses_channel_bed_mix_header() {
        let mut writer = base_header(1);
        writer.write_bits(1, 2).unwrap();
        writer
            .write_bits(ChannelConfig::Mc5_1.index().into(), 7)
            .unwrap();
        writer.write_bits(0, 4).unwrap();
        writer.write_bits(1, 7).unwrap();
        writer.write_bits(4, 4).unwrap();
        writer.write_bits(1, 2).unwrap();
        writer.write_bits(0xcd, 8).unwrap();
        let header = parse_header_at(&writer.into_bytes(), 0).unwrap();
        assert_eq!(header.sound_bed_type, Some(SoundBedType::ChannelBed));
        assert_eq!(header.channel_config, Some(ChannelConfig::Mc5_1));
        assert_eq!(header.objects, 2);
        assert_eq!(header.bed_channels, 6);
        assert_eq!(header.channels, 8);
        assert_eq!(header.header_len, 9);
        assert_eq!(header.bitrate, 320_000);
        assert_eq!(header.bed_bitrate, Some(192_000));
        assert_eq!(header.object_bitrate, Some(64_000));
        assert!(header.has_lfe);
    }

    #[test]
    fn parses_hoa_header() {
        let mut writer = base_header(2);
        writer.write_bits(1, 4).unwrap();
        writer.write_bits(2, 2).unwrap();
        writer.write_bits(3, 4).unwrap();
        writer.write_bits(0xcd, 8).unwrap();
        let header = parse_header_at(&writer.into_bytes(), 0).unwrap();
        assert_eq!(header.channel_config, Some(ChannelConfig::Hoa2));
        assert_eq!(header.hoa_order, Some(2));
        assert_eq!(header.channels, 9);
        assert_eq!(header.header_len, 7);
    }

    #[test]
    fn scans_past_false_sync() {
        let mut input = vec![0xff, 0xf1, 0, 0, 0, 0, 0, 0, 0, 0x12, 0x34];
        input.extend_from_slice(&[0xff, 0xf2, 0x00, 0x71, 0xa2, 0x94, 0x1b]);
        let info = parse_header(&input).unwrap();
        assert_eq!(info.offset, 11);
    }

    #[test]
    fn rejects_null_c_bitrate_tables() {
        let mut writer = base_header(0);
        writer
            .write_bits(ChannelConfig::Mc10_2.index().into(), 7)
            .unwrap();
        writer.write_bits(1, 2).unwrap();
        writer.write_bits(0, 4).unwrap();
        writer.write_bits(0, 8).unwrap();
        assert!(matches!(
            parse_header_at(&writer.into_bytes(), 0),
            Err(HeaderError::InvalidBitrateIndex {
                config: 4,
                index: 0
            })
        ));
    }

    #[test]
    fn every_truncated_reference_prefix_is_an_error() {
        let header = [0xff, 0xf2, 0x00, 0x71, 0xa2, 0x94, 0x1b];
        for length in 0..header.len() {
            assert!(
                parse_header(&header[..length]).is_err(),
                "prefix length {length}"
            );
        }
        assert!(parse_header(&header).is_ok());
    }
}
