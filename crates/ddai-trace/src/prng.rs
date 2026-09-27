//! A small, explicitly-specified PRNG for the `random-v1` scenario generator.
//!
//! `rand`'s default algorithms are not used here on purpose (see the task spec / `docs/PLAN.md`
//! §1.4): their exact output stream is not part of that crate's semver guarantees and may change
//! between versions, which would silently change every previously-generated scenario's bytes.
//! `SplitMix64` (Sebastiano Vigna, public domain) is simple enough to pin down completely in
//! `docs/formats.md`, so a from-scratch reimplementation (in this crate, or later in another
//! language) reproduces the exact same stream forever.

/// SplitMix64: `state += 0x9E3779B97F4A7C15`, then a fixed xor/multiply/xor/multiply/xor
/// finalizer. See <https://prng.di.unimi.it/splitmix64.c> for the reference C implementation
/// this mirrors bit-for-bit.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seeds the generator. Any `u64` is a valid seed (including `0`).
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, bound)`. `bound == 0` always returns `0`.
    ///
    /// Uses the "multiply-high" debiasing trick (take the high 64 bits of a 64x64->128 bit
    /// multiply): the bias this leaves is at most `bound / 2^64`, unmeasurably small for the
    /// `bound` values this crate ever uses.
    pub fn below(&mut self, bound: u32) -> u32 {
        if bound == 0 {
            return 0;
        }
        ((self.next_u64() as u128 * bound as u128) >> 64) as u32
    }

    /// Uniform integer in `[lo, hi]` (inclusive on both ends). Requires `lo <= hi`.
    pub fn range_inclusive(&mut self, lo: i32, hi: i32) -> i32 {
        assert!(lo <= hi, "range_inclusive: lo {lo} > hi {hi}");
        let span = (hi - lo) as u32 + 1;
        lo + self.below(span) as i32
    }

    /// `true` with probability `numerator / denominator` (both `below(denominator) < numerator`,
    /// so `numerator >= denominator` is always `true` and `numerator == 0` is always `false`).
    pub fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        self.below(denominator) < numerator
    }

    /// Picks a uniformly random element of a non-empty slice.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        assert!(!items.is_empty(), "pick: empty slice");
        &items[self.below(items.len() as u32) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_splitmix64_stream() {
        // First outputs of the reference splitmix64.c seeded with 0, computed independently
        // from the public-domain reference algorithm (https://prng.di.unimi.it/splitmix64.c).
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xe220a8397b1dcdaf);
        assert_eq!(rng.next_u64(), 0x6e789e6aa1b965f4);
        assert_eq!(rng.next_u64(), 0x06c45d188009454f);
        assert_eq!(rng.next_u64(), 0xf88bb8a8724c81ec);
    }

    #[test]
    fn same_seed_gives_identical_stream() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        let seq_a: Vec<u64> = (0..50).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..50).map(|_| b.next_u64()).collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        let seq_a: Vec<u64> = (0..20).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..20).map(|_| b.next_u64()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn below_never_reaches_bound() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..10_000 {
            assert!(rng.below(5) < 5);
        }
    }

    #[test]
    fn below_zero_is_always_zero() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..100 {
            assert_eq!(rng.below(0), 0);
        }
    }

    #[test]
    fn range_inclusive_stays_in_bounds() {
        let mut rng = SplitMix64::new(123);
        for _ in 0..10_000 {
            let v = rng.range_inclusive(-3, 3);
            assert!((-3..=3).contains(&v));
        }
    }

    #[test]
    fn range_inclusive_single_value() {
        let mut rng = SplitMix64::new(1);
        for _ in 0..10 {
            assert_eq!(rng.range_inclusive(5, 5), 5);
        }
    }

    #[test]
    fn chance_zero_and_full() {
        let mut rng = SplitMix64::new(9);
        for _ in 0..1000 {
            assert!(!rng.chance(0, 10));
            assert!(rng.chance(10, 10));
        }
    }
}
