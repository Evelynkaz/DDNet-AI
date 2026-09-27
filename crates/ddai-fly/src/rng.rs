//! A tiny, dependency-free, deterministic PRNG (splitmix64 — same generator the phase-0 benchmark
//! used, see `docs/research/rust-stack.md` §4) for reproducible parameter initialisation and for
//! the stability test's/CLI bench's synthetic input traffic. Not cryptographic, not `rand`: we
//! only need "same seed -> same numbers, forever" determinism, and pulling in `rand` would be a
//! dependency this crate's core doesn't otherwise need (see acceptance criterion 1's dependency
//! list).

/// splitmix64 (Vigna, 2015): a 64-bit state, 64-bit output PRNG that passes standard statistical
/// test suites and is often used to seed other generators. Reference implementation:
/// <https://prng.di.unimi.it/splitmix64.c>.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0, 1)`, using the top 24 bits (matches `f32`'s mantissa precision, so
    /// every representable output is equally likely).
    #[inline]
    pub fn next_f32_unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // top 24 bits of a fresh 64-bit draw
        (bits as f32) / (1u32 << 24) as f32
    }

    /// Standard normal `f32` (mean 0, std 1) via Box-Muller, using two fresh uniform draws. Not
    /// the fastest possible generator, but this is only ever used for one-time parameter
    /// initialisation and test input synthesis, never per-substep.
    #[inline]
    pub fn next_gaussian(&mut self) -> f32 {
        // Avoid u1 == 0.0 (ln(0) = -inf) by drawing from (0, 1] instead of [0, 1).
        let u1 = 1.0 - self.next_f32_unit();
        let u2 = self.next_f32_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Derives an independent-looking 64-bit seed from a base seed and an index (e.g. a type or
/// neuron index), so per-item deterministic randomness doesn't need one giant shared generator
/// threaded through the whole init pass — each item's value depends only on `(seed, index)`,
/// which also makes property tests ("does item 5's value change if I add a type at the end?")
/// straightforward. Uses `SplitMix64` itself as the mixer (splitmix64 is designed to be used this
/// way: seed one instance per stream from a counter).
pub fn seeded_for(seed: u64, index: u64) -> SplitMix64 {
    let mut mixer = SplitMix64::new(seed ^ index.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    // Discard the first draw: with some (seed, index) pairs the very first `next_u64` output can
    // be close to the initial state for a low-quality mix; one extra round is cheap and matches
    // the reference generator's own "always advance state before reading" discipline.
    let _ = mixer.next_u64();
    mixer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_sequence() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_give_different_sequences() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        let seq_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn unit_draws_stay_in_range() {
        let mut r = SplitMix64::new(7);
        for _ in 0..10_000 {
            let x = r.next_f32_unit();
            assert!((0.0..1.0).contains(&x), "x={x} out of [0,1)");
        }
    }

    #[test]
    fn gaussian_draws_are_finite_and_roughly_standard() {
        let mut r = SplitMix64::new(123);
        let n = 20_000;
        let samples: Vec<f32> = (0..n).map(|_| r.next_gaussian()).collect();
        assert!(samples.iter().all(|x| x.is_finite()));
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let var: f32 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05, "mean={mean}");
        assert!((var - 1.0).abs() < 0.1, "var={var}");
    }

    #[test]
    fn seeded_for_gives_different_streams_per_index() {
        let mut a = seeded_for(99, 0);
        let mut b = seeded_for(99, 1);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn seeded_for_is_deterministic() {
        let mut a = seeded_for(99, 5);
        let mut b = seeded_for(99, 5);
        assert_eq!(a.next_u64(), b.next_u64());
    }
}
