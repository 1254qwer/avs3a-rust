//! The pseudo-random stream AVS3 decoders draw noise from.
//!
//! Two decode stages consume it — BWE whitening and latent noise filling — so
//! both the sequence and the number of draws per frame have to match, or the
//! output diverges for the rest of the frame.
//!
//! Shipping decoders link the Microsoft UCRT `rand()`, whose per-thread seed
//! starts at 1:
//!
//! ```c
//! seed = 214013 * seed + 2531011;
//! return (seed >> 16) & 0x7fff;
//! ```
//!
//! The AVS3-P3 reference instead defines its own `rand()` in `bwe_dec.c`
//! (seed 432078, `1103515245 * seed + 12345`), which under glibc overrides the
//! C library's for every translation unit. The two are not interchangeable:
//! noise filling supplies the high band of every latent dimension the quantizer
//! collapsed onto its zero bin, so the choice decides the high-frequency content
//! of every channel that has any. Measured against a shipping decoder, the
//! reference generator leaves −34 dB of residual where this one leaves −92 dB,
//! so only this one is implemented.
//!
//! `bwe_dec.c` also carries `#define RAND_MAX 0x7FFF`, which is local to that
//! file, so `noise_filling.c` — which only includes `<stdlib.h>` — divides by
//! the C library's `RAND_MAX` instead. Under glibc that is `2147483647`, and
//! since the generator never returns more than `32767` the noise term collapses
//! to a near-constant `-1.0` rather than spanning the `[-1, 1]` its own comment
//! describes. That is a build defect rather than intended behaviour, and the
//! UCRT's `RAND_MAX` is `32767` anyway, so both call sites here divide by
//! [`AVS3_RAND_MAX`].

/// Value both call sites divide `rand()` by, and the largest it can return.
pub const AVS3_RAND_MAX: u32 = 0x7FFF;

const INITIAL_SEED: u32 = 1;
const MULTIPLIER: u32 = 214_013;
const INCREMENT: u32 = 2_531_011;

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
    /// Construct the stream at the UCRT's initial per-thread seed.
    pub fn new() -> Self {
        Self { seed: INITIAL_SEED }
    }

    /// Reset to the start of the sequence.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Return the next value in `0..=AVS3_RAND_MAX`, as C's `rand()` would.
    pub fn rand(&mut self) -> u32 {
        // Wrapping matches the C, which multiplies into an `unsigned int` and lets
        // it overflow. Only the middle bits are handed out, so the whole 32-bit
        // state carries forward and the period is 2^32 rather than 2^15.
        self.seed = self.seed.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
        (self.seed >> 16) & AVS3_RAND_MAX
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
    fn stream_matches_ucrt_rand() {
        // The textbook Microsoft sequence for a program that never called srand.
        let expected = [
            41, 18467, 6334, 26500, 19169, 15724, 11478, 29358, 26962, 24464, 5705, 28145,
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

    /// The state is wider than a draw, so the sequence does not close after
    /// the 32768 values a 15-bit generator would repeat over.
    #[test]
    fn period_is_longer_than_the_draw_range() {
        let mut random = Avs3Random::new();
        let first = random.seed;
        for _ in 0..1 << 15 {
            random.rand();
            assert_ne!(random.seed, first, "state repeated within 32768 draws");
        }
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
