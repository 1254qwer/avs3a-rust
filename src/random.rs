//! The pseudo-random stream AVS3 decoders draw noise from.
//!
//! Two decode stages consume it — BWE whitening and latent noise filling — so
//! both the sequence and the number of draws per frame have to match the
//! reference, or the output diverges for the rest of the frame.
//!
//! The reference defines its own `rand()` in `bwe_dec.c`, which at link time
//! overrides the C library's for every translation unit:
//!
//! ```c
//! #define RAND_MAX 0x7FFF
//! static unsigned int seed = 432078;
//! int rand() { seed = (1103515245 * seed + 12345) & RAND_MAX; return (int)seed; }
//! ```
//!
//! That `#define` is local to `bwe_dec.c`, so `noise_filling.c` — which only
//! includes `<stdlib.h>` — divides by the C library's `RAND_MAX` instead. Under
//! glibc that is `2147483647`, and since the generator never returns more than
//! `32767` the noise term collapses to a near-constant `-1.0` rather than
//! spanning the `[-1, 1]` its own comment describes. That is a build defect
//! rather than intended behaviour, so both call sites here divide by
//! [`AVS3_RAND_MAX`].

/// Value the reference divides `rand()` by, and the largest it can return.
pub const AVS3_RAND_MAX: u32 = 0x7FFF;

const INITIAL_SEED: u32 = 432_078;
const MULTIPLIER: u32 = 1_103_515_245;
const INCREMENT: u32 = 12_345;

/// Decoder-local replacement for the C implementation's global seed.
///
/// Keeping the state explicit stops independent decoder instances from coupling
/// through one global. The stream is still free-running within an instance: it is
/// seeded once and advances with every draw, so output depends on how many frames
/// have been decoded and not on the bitstream alone. That is the C behaviour, and
/// it is why AV3A cannot be seeked sample-exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Avs3Random {
    seed: u32,
}

impl Avs3Random {
    /// Construct the stream from the reference's initial seed.
    pub fn new() -> Self {
        Self::from_seed(INITIAL_SEED)
    }

    /// Construct a stream from an explicit seed, for tests.
    pub fn from_seed(seed: u32) -> Self {
        Self { seed }
    }

    /// Reset to the start of the sequence.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Return the next value in `0..=AVS3_RAND_MAX`, as C's `rand()` would.
    pub fn rand(&mut self) -> u32 {
        // Wrapping matches the C, which multiplies into an `unsigned int` and lets
        // it overflow. The mask is `mod 32768`, so only the seed's low 15 bits ever
        // matter and the period is exactly 32768.
        self.seed = self.seed.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT) & AVS3_RAND_MAX;
        self.seed
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
    fn stream_matches_bwe_dec_c() {
        let expected = [
            5871, 17916, 17541, 25306, 13067, 17896, 9217, 21158, 29927, 22676, 32573, 29490,
        ];
        let mut random = Avs3Random::new();
        for value in expected {
            assert_eq!(random.rand(), value);
        }
    }

    #[test]
    fn every_draw_fits_the_reference_range() {
        let mut random = Avs3Random::new();
        for _ in 0..100_000 {
            assert!(random.rand() <= AVS3_RAND_MAX);
        }
    }

    /// Scaled the way both call sites scale it, the noise spans `[-1, 1]` rather
    /// than collapsing the way the C library's `RAND_MAX` would.
    #[test]
    fn scaled_noise_is_uniform_over_the_full_range() {
        let mut random = Avs3Random::new();
        let mut above = 0;
        let (mut low, mut high) = (f32::MAX, f32::MIN);
        for _ in 0..10_000 {
            let noise = (random.rand() as f32 / AVS3_RAND_MAX as f32) * 2.0 - 1.0;
            assert!((-1.0..=1.0).contains(&noise), "out of range: {noise}");
            low = low.min(noise);
            high = high.max(noise);
            if noise > 0.0 {
                above += 1;
            }
        }
        assert!(
            (4000..6000).contains(&above),
            "not uniform: {above} above zero"
        );
        assert!(
            low < -0.99 && high > 0.99,
            "range not covered: {low}..{high}"
        );
    }

    #[test]
    fn period_is_the_state_size() {
        let mut random = Avs3Random::new();
        let first = random.rand();
        let mut steps = 1usize;
        while random.rand() != first {
            steps += 1;
            assert!(steps <= 1 << 16, "period did not close");
        }
        assert_eq!(steps, 1 << 15);
    }

    #[test]
    fn reset_returns_to_the_start_of_the_stream() {
        let mut random = Avs3Random::new();
        let first = random.rand();
        for _ in 0..1000 {
            random.rand();
        }
        random.reset();
        assert_eq!(random.rand(), first);
    }

    #[test]
    fn clones_continue_independently_from_the_same_state() {
        let mut first = Avs3Random::new();
        for _ in 0..13 {
            first.rand();
        }
        let mut second = first.clone();
        for _ in 0..100 {
            assert_eq!(first.rand(), second.rand());
        }
    }
}
