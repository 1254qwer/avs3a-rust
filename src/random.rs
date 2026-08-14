/// Largest value returned by the AVS3-compatible pseudo-random generator.
///
/// The C reference divides `rand()` by `RAND_MAX`.  Fixing that value here is
/// important: it is not guaranteed to be the same on every C runtime.
pub const AVS3_RAND_MAX: u32 = 2_147_483_647;

const STATE_DEGREE: usize = 31;
const STATE_SEPARATION: usize = 3;
const PARK_MILLER_MULTIPLIER: u64 = 16_807;

/// Decoder-local replacement for the C implementation's process-global
/// `rand()` state.
///
/// The default state reproduces glibc's unseeded `rand()` sequence (seed 1),
/// which is what the AVS3 reference decoder uses on Linux.  Keeping the state
/// explicit makes frame/channel ordering deterministic and avoids coupling
/// independent decoder instances through a libc global.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Avs3Random {
    state: [u32; STATE_DEGREE],
    front: usize,
    rear: usize,
}

impl Avs3Random {
    /// Construct the glibc-compatible default stream (seed 1).
    pub fn new() -> Self {
        Self::from_seed(1)
    }

    /// Construct a deterministic stream from a positive 31-bit seed.
    ///
    /// Seed zero follows glibc and is treated as seed one. Values outside the
    /// positive signed 31-bit range are reduced modulo `RAND_MAX`; the AVS3
    /// compatibility path itself always uses the default seed.
    pub fn from_seed(seed: u32) -> Self {
        let mut normalized = seed % AVS3_RAND_MAX;
        if normalized == 0 {
            normalized = 1;
        }

        let mut state = [0_u32; STATE_DEGREE];
        state[0] = normalized;
        for index in 1..STATE_DEGREE {
            state[index] = ((u64::from(state[index - 1]) * PARK_MILLER_MULTIPLIER)
                % u64::from(AVS3_RAND_MAX)) as u32;
        }

        let mut random = Self {
            state,
            front: STATE_SEPARATION,
            rear: 0,
        };
        // glibc discards 10 * degree values after initializing TYPE_3 state.
        for _ in 0..STATE_DEGREE * 10 {
            random.next_u31();
        }
        random
    }

    /// Reset this state to the unseeded C/glibc sequence.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Return the next integer in `0..=AVS3_RAND_MAX`.
    pub fn next_u31(&mut self) -> u32 {
        let value = self.state[self.front].wrapping_add(self.state[self.rear]);
        self.state[self.front] = value;

        self.front += 1;
        if self.front == STATE_DEGREE {
            self.front = 0;
        }
        self.rear += 1;
        if self.rear == STATE_DEGREE {
            self.rear = 0;
        }

        (value >> 1) & AVS3_RAND_MAX
    }
}

impl Default for Avs3Random {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_stream_matches_glibc_rand() {
        let expected = [
            1_804_289_383,
            846_930_886,
            1_681_692_777,
            1_714_636_915,
            1_957_747_793,
            424_238_335,
            719_885_386,
            1_649_760_492,
            596_516_649,
            1_189_641_421,
        ];
        let mut random = Avs3Random::new();
        for value in expected {
            assert_eq!(random.next_u31(), value);
        }
    }

    #[test]
    fn zero_seed_and_reset_use_the_default_stream() {
        let mut zero = Avs3Random::from_seed(0);
        let mut default = Avs3Random::default();
        for _ in 0..100 {
            assert_eq!(zero.next_u31(), default.next_u31());
        }

        zero.reset();
        assert_eq!(zero.next_u31(), 1_804_289_383);
    }

    #[test]
    fn clones_continue_independently_from_the_same_state() {
        let mut first = Avs3Random::from_seed(17);
        for _ in 0..13 {
            first.next_u31();
        }
        let mut second = first.clone();
        for _ in 0..100 {
            assert_eq!(first.next_u31(), second.next_u31());
        }
    }
}
