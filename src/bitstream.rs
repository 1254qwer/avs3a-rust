pub use crate::error::BitstreamError;

/// MSB-first reader used by the AV3A header and side-information syntax.
#[derive(Debug, Clone, Copy)]
pub struct BitReader<'a> {
    data: &'a [u8],
    position: usize,
    bit_len: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
            bit_len: data.len().saturating_mul(8),
        }
    }

    /// Construct a reader restricted to the first `bit_len` valid bits.
    ///
    /// This is useful for payloads whose last storage byte contains padding:
    /// reads cannot silently consume those padding bits merely because the
    /// backing slice has room for them.
    pub fn with_bit_len(data: &'a [u8], bit_len: usize) -> Result<Self, BitstreamError> {
        let available = data.len().saturating_mul(8);
        if bit_len > available {
            return Err(BitstreamError::UnexpectedEof {
                position: 0,
                requested: bit_len,
                available,
            });
        }
        Ok(Self {
            data,
            position: 0,
            bit_len,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn remaining(&self) -> usize {
        self.bit_len.saturating_sub(self.position)
    }

    pub fn bit_len(&self) -> usize {
        self.bit_len
    }

    pub fn read_bits(&mut self, width: usize) -> Result<u64, BitstreamError> {
        if width > 64 {
            return Err(BitstreamError::InvalidWidth(width));
        }
        if width == 0 {
            return Ok(0);
        }
        if width > self.remaining() {
            return Err(BitstreamError::UnexpectedEof {
                position: self.position,
                requested: width,
                available: self.remaining(),
            });
        }

        let end = self
            .position
            .checked_add(width)
            .ok_or(BitstreamError::PositionOverflow)?;
        let mut value = 0_u64;
        while self.position < end {
            let byte = self.data[self.position / 8];
            let bit = (byte >> (7 - (self.position % 8))) & 1;
            value = (value << 1) | u64::from(bit);
            self.position += 1;
        }
        Ok(value)
    }

    pub fn read_u8(&mut self, width: usize) -> Result<u8, BitstreamError> {
        let value = self.read_bits(width)?;
        u8::try_from(value).map_err(|_| BitstreamError::InvalidWidth(width))
    }

    /// Read whole syntax bytes at the current bit position.
    ///
    /// The destination is filled transactionally: the reader position and
    /// destination remain unchanged if the complete byte sequence is not
    /// available. Aligned reads use `copy_from_slice`; unaligned reads combine
    /// adjacent source bytes without an eight-iteration bit loop.
    pub fn read_bytes_into(&mut self, output: &mut [u8]) -> Result<(), BitstreamError> {
        let width = output
            .len()
            .checked_mul(8)
            .ok_or(BitstreamError::PositionOverflow)?;
        if width > self.remaining() {
            return Err(BitstreamError::UnexpectedEof {
                position: self.position,
                requested: width,
                available: self.remaining(),
            });
        }
        if output.is_empty() {
            return Ok(());
        }

        let byte_index = self.position / 8;
        let bit_offset = self.position % 8;
        if bit_offset == 0 {
            let end = byte_index
                .checked_add(output.len())
                .ok_or(BitstreamError::PositionOverflow)?;
            output.copy_from_slice(&self.data[byte_index..end]);
        } else {
            for (offset, destination) in output.iter_mut().enumerate() {
                let index = byte_index + offset;
                *destination =
                    (self.data[index] << bit_offset) | (self.data[index + 1] >> (8 - bit_offset));
            }
        }
        self.position = self
            .position
            .checked_add(width)
            .ok_or(BitstreamError::PositionOverflow)?;
        Ok(())
    }

    pub fn skip(&mut self, width: usize) -> Result<(), BitstreamError> {
        if width > self.remaining() {
            return Err(BitstreamError::UnexpectedEof {
                position: self.position,
                requested: width,
                available: self.remaining(),
            });
        }
        self.position += width;
        Ok(())
    }
}

/// Small MSB-first writer useful for constructing synthetic headers in tests.
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    data: Vec<u8>,
    position: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn write_bits(&mut self, value: u64, width: usize) -> Result<(), BitstreamError> {
        if width > 64 {
            return Err(BitstreamError::InvalidWidth(width));
        }
        if width < 64 && value >> width != 0 {
            return Err(BitstreamError::InvalidWidth(width));
        }
        let end = self
            .position
            .checked_add(width)
            .ok_or(BitstreamError::PositionOverflow)?;
        let bytes = end.checked_add(7).ok_or(BitstreamError::PositionOverflow)? / 8;
        if self.data.len() < bytes {
            self.data.resize(bytes, 0);
        }
        for bit_offset in 0..width {
            let source_shift = width - bit_offset - 1;
            let bit = ((value >> source_shift) & 1) as u8;
            let index = self.position + bit_offset;
            let byte = &mut self.data[index / 8];
            let mask = 1_u8 << (7 - (index % 8));
            if bit != 0 {
                *byte |= mask;
            } else {
                *byte &= !mask;
            }
        }
        self.position = end;
        Ok(())
    }

    pub fn bit_len(&self) -> usize {
        self.position
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_across_bytes_msb_first() {
        let mut reader = BitReader::new(&[0b1011_0010, 0b0110_0001]);
        assert_eq!(reader.read_bits(3).unwrap(), 0b101);
        assert_eq!(reader.read_bits(9).unwrap(), 0b100100110);
        assert_eq!(reader.position(), 12);
    }

    #[test]
    fn rejects_truncated_reads_without_advancing() {
        let mut reader = BitReader::new(&[0xff]);
        assert_eq!(
            reader.read_bits(9),
            Err(BitstreamError::UnexpectedEof {
                position: 0,
                requested: 9,
                available: 8
            })
        );
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn bit_length_hides_storage_padding() {
        let mut reader = BitReader::with_bit_len(&[0xab, 0xff], 12).unwrap();
        assert_eq!(reader.bit_len(), 12);
        assert_eq!(reader.read_bits(8).unwrap(), 0xab);
        assert_eq!(reader.read_bits(4).unwrap(), 0xf);
        assert_eq!(
            reader.read_bits(1),
            Err(BitstreamError::UnexpectedEof {
                position: 12,
                requested: 1,
                available: 0,
            })
        );
    }

    #[test]
    fn reads_aligned_and_unaligned_byte_slices_transactionally() {
        let data = [0b1011_0010, 0b0110_0001, 0b1110_0101, 0b0101_1010];
        let mut aligned = BitReader::new(&data);
        let mut aligned_output = [0_u8; 3];
        aligned.read_bytes_into(&mut aligned_output).unwrap();
        assert_eq!(aligned_output, data[..3]);
        assert_eq!(aligned.position(), 24);

        let mut unaligned = BitReader::new(&data);
        assert_eq!(unaligned.read_bits(3).unwrap(), 0b101);
        let mut output = [0_u8; 3];
        unaligned.read_bytes_into(&mut output).unwrap();
        assert_eq!(output, [0b1001_0011, 0b0000_1111, 0b0010_1010]);
        assert_eq!(unaligned.position(), 27);

        let mut short = BitReader::with_bit_len(&data, 26).unwrap();
        short.skip(3).unwrap();
        let mut untouched = [7_u8; 3];
        assert!(short.read_bytes_into(&mut untouched).is_err());
        assert_eq!(untouched, [7; 3]);
        assert_eq!(short.position(), 3);
    }

    #[test]
    fn writer_round_trips() {
        let mut writer = BitWriter::new();
        writer.write_bits(0x5, 3).unwrap();
        writer.write_bits(0x1a3, 9).unwrap();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_bits(3).unwrap(), 0x5);
        assert_eq!(reader.read_bits(9).unwrap(), 0x1a3);
    }
}
