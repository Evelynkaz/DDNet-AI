//! Port of the old bot's deterministic RNG, `src/nn/rng.ts` (cited by line number below), plus the
//! planner's separate opponent-seed LCG step (`src/plan/planner.ts:1044`).
//!
//! `Rng` is `splitmix32`-seeded `xoshiro128**`, with `next_float` derived from `next_u32` and a
//! Box-Muller `next_gaussian` that carries one spare Gaussian sample between calls (reset only by
//! constructing a fresh `Rng`, never by any other method). All state (`s0..s3`, `have_spare`,
//! `spare`) is public so callers (the planner's `SimState` save/restore, the parity harness in
//! `tools/jsmath-oracle`) can read and write it directly, matching that the TS class's fields are
//! only `private` at the type-checker level — the parity harness there reaches into them anyway
//! (`docs/research/orig-plan.md` §2.4).

use crate::{PI, cos, log, sin, sqrt};

/// `splitmix32`, `src/nn/rng.ts:1-11`: `Rng::new`'s seeding step. Not part of the public API (the
/// old bot never constructs a bare `splitmix32` generator on its own — only `Rng`'s constructor
/// does), but ported as its own function to mirror the TS source's shape.
fn splitmix32_next(state: &mut u32) -> u32 {
    *state = state.wrapping_add(0x9e37_79b9);
    let mut z = *state;
    z = (z ^ (z >> 16)).wrapping_mul(0x21f0_aaad);
    z = (z ^ (z >> 15)).wrapping_mul(0x735a_2d97);
    z ^= z >> 15;
    z
}

/// `rotl(x, k)`, `src/nn/rng.ts:13-15`.
#[inline]
fn rotl(x: u32, k: u32) -> u32 {
    x.rotate_left(k)
}

/// Port of the TS `Rng` class (`src/nn/rng.ts:17-66`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rng {
    pub s0: u32,
    pub s1: u32,
    pub s2: u32,
    pub s3: u32,
    pub have_spare: bool,
    pub spare: f64,
}

impl Rng {
    /// `new Rng(seed)`, `src/nn/rng.ts:25-31`: seeds `s0..s3` by calling the `splitmix32` closure
    /// four times in order.
    pub fn new(seed: u32) -> Self {
        let mut state = seed;
        let s0 = splitmix32_next(&mut state);
        let s1 = splitmix32_next(&mut state);
        let s2 = splitmix32_next(&mut state);
        let s3 = splitmix32_next(&mut state);
        Rng {
            s0,
            s1,
            s2,
            s3,
            have_spare: false,
            spare: 0.0,
        }
    }

    /// Restores an `Rng` from previously-saved state (`s0..s3`, `have_spare`, `spare`) — the
    /// counterpart to reading the public fields directly, kept as a named constructor for
    /// call-site clarity in `SimState` restore code.
    pub fn from_state(s0: u32, s1: u32, s2: u32, s3: u32, have_spare: bool, spare: f64) -> Self {
        Rng {
            s0,
            s1,
            s2,
            s3,
            have_spare,
            spare,
        }
    }

    /// `nextU32()`, xoshiro128**, `src/nn/rng.ts:33-46`.
    pub fn next_u32(&mut self) -> u32 {
        let result = rotl(self.s1.wrapping_mul(5), 7).wrapping_mul(9);
        let t = self.s1 << 9;

        self.s2 ^= self.s0;
        self.s3 ^= self.s1;
        self.s1 ^= self.s2;
        self.s0 ^= self.s3;

        self.s2 ^= t;
        self.s3 = rotl(self.s3, 11);

        result
    }

    /// `nextFloat()`, `src/nn/rng.ts:48-50`: `nextU32() / 4294967296`. `next_u32()` is always
    /// exactly representable as `f64` (it is a `u32`), and dividing by the exact power of two
    /// `2^32` is itself exact rounding-wise (only the exponent changes) — so `as f64` here is not
    /// an approximation, it is the bit-exact JS `Number(u32) / 4294967296` operation.
    pub fn next_float(&mut self) -> f64 {
        f64::from(self.next_u32()) / 4_294_967_296.0
    }

    /// `nextGaussian()`, Box-Muller with a carried spare, `src/nn/rng.ts:52-65`. The RNG-draw
    /// order matters for parity: `nextFloat()` for `u` (looping while exactly `0.0`, `while (u ===
    /// 0)`), then `nextFloat()` for `v` the same way, *then* `sqrt`/`log`/`sin`/`cos` — ported via
    /// this crate's own [`sqrt`]/[`log`]/[`sin`]/[`cos`]/[`PI`], not `std`, so this is V8-bit-exact
    /// too, not just algorithmically the same shape.
    pub fn next_gaussian(&mut self) -> f64 {
        if self.have_spare {
            self.have_spare = false;
            return self.spare;
        }
        let mut u = 0.0;
        while u == 0.0 {
            u = self.next_float();
        }
        let mut v = 0.0;
        while v == 0.0 {
            v = self.next_float();
        }
        let mag = sqrt(-2.0 * log(u));
        self.spare = mag * sin(2.0 * PI * v);
        self.have_spare = true;
        mag * cos(2.0 * PI * v)
    }
}

/// The planner's opponent-seed LCG step (`src/plan/planner.ts:1044`):
/// `this.oppSeed = (this.oppSeed * 1664525 + 1013904223) >>> 0`. In JS, `oppSeed * 1664525` is at
/// most `(2^32 - 1) * 1664525 < 2^53`, so the multiplication is exact double arithmetic before the
/// `>>> 0` truncation; `u32::wrapping_mul`/`wrapping_add` compute the same modulo-2^32 result
/// directly, without going through `f64` at all.
pub fn opp_seed_next(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_seed() {
        let mut a = Rng::new(12345);
        let mut b = Rng::new(12345);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.next_u32(), b.next_u32());
    }

    #[test]
    fn next_float_is_in_unit_range() {
        let mut r = Rng::new(42);
        for _ in 0..10_000 {
            let f = r.next_float();
            assert!((0.0..1.0).contains(&f));
        }
    }

    #[test]
    fn gaussian_spare_is_carried_and_consumed_in_pairs() {
        let mut r = Rng::new(7);
        assert!(!r.have_spare);
        let _first = r.next_gaussian();
        assert!(r.have_spare);
        let spare = r.spare;
        let second = r.next_gaussian();
        assert_eq!(second, spare);
        assert!(!r.have_spare);
    }

    #[test]
    fn state_roundtrips() {
        let mut r = Rng::new(99);
        let _ = r.next_gaussian(); // populate the spare so have_spare/spare are non-default
        let saved = r;
        let restored = Rng::from_state(r.s0, r.s1, r.s2, r.s3, r.have_spare, r.spare);
        assert_eq!(saved, restored);
    }

    #[test]
    fn opp_seed_matches_planner_lcg_by_hand() {
        let seed = 7u32;
        let next = opp_seed_next(seed);
        assert_eq!(next, (7u64 * 1664525 + 1013904223) as u32);
    }

    // Review finding F10: `next_gaussian`'s `while (u === 0) u = this.nextFloat()` /
    // `while (v === 0) ...` retry loops (`src/nn/rng.ts:59-60`) guard against `nextU32()`
    // returning exactly 0 (probability 2^-32 per draw) — an event so rare that no amount of
    // realistic fuzzing (the full oracle's 10^8 draws included) will ever hit it by chance, so a
    // bug here (e.g. a single draw instead of a retry loop) would ship silently. These three
    // tests force it directly via `Rng::from_state`, crafting `s1` so a specific draw is
    // *provably* exactly 0 (`next_u32`'s `result = rotl(s1.wrapping_mul(5), 7).wrapping_mul(9)`
    // below, so `s1 == 0` forces `result == 0` on that call), and check both
    // the returned Gaussian *and* the resulting `s0..s3`/`have_spare`/`spare` state against bits
    // recorded once from the real `src/nn/rng.ts`, run by Node (24.21.0 / V8
    // 13.6.233.17-node.53): TypeScript's `private` is a compile-time-only annotation, so
    // `r.s1 = 0` on a constructed `Rng` works at runtime under Node's type-stripping the same way
    // `Rng::from_state` does here — this is not a Rust-only trick.
    //
    // All three fail (proven by temporarily replacing each `while` with a single unconditional
    // draw and re-running): case 1 gives a completely different `g1`/`g2` and final state: with
    // the retry removed, `nextFloat()` returns exactly `0.0` for `u`, so `mag = sqrt(-2*log(0)) =
    // sqrt(Infinity) = Infinity`, propagating `Infinity`/`NaN` throughout — nowhere close to the
    // finite recorded values below. Case 2 (mutated the same way) similarly ends up with `v = 0`
    // feeding `sin(0)`/`cos(0)` instead of the real second draw, giving `spare = 0.0` and a
    // different `g1`, and consuming one fewer `next_u32()` call, so the final state differs too.

    #[test]
    fn next_gaussian_retries_when_u_is_exactly_zero() {
        // `s1 = 0` forces the *first* `next_u32()` call (u's first draw) to return exactly 0.
        let mut r = Rng::from_state(1, 0, 3, 4, false, 0.0);
        let g1 = r.next_gaussian();
        assert_eq!(g1.to_bits(), 0x4014_42ea_4f09_6017);
        assert_eq!(
            (r.s0, r.s1, r.s2, r.s3, r.have_spare),
            (16_789_506, 9221, 11776, 8_398_856, true)
        );
        assert_eq!(r.spare.to_bits(), 0x3f2b_f8f7_2d76_eddc);

        // The carried spare from `g1` above, returned without drawing anything further.
        let g2 = r.next_gaussian();
        assert_eq!(g2.to_bits(), 0x3f2b_f8f7_2d76_eddc);
        assert!(!r.have_spare);
    }

    #[test]
    fn next_gaussian_retries_when_v_is_exactly_zero() {
        // `s0 ^ s1 ^ s2 == 0` (1^2^3 == 0) makes `next_u32`'s state update set the *next* call's
        // `s1` to `s1 ^ (s2 ^ s0)` = 0 (see `next_u32`'s `s2 ^= s0; ...; s1 ^= s2;`), so u's first
        // draw (this call) is a normal nonzero value, but v's first draw (the next `next_u32`
        // call) is forced to exactly 0, exercising the *second* `while` loop instead of the first.
        let mut r = Rng::from_state(1, 2, 3, 4, false, 0.0);
        let g1 = r.next_gaussian();
        assert_eq!(g1.to_bits(), 0x4014_42b8_6511_3450);
        assert_eq!(
            (r.s0, r.s1, r.s2, r.s3, r.have_spare),
            (25_179_138, 12295, 540_162, 2_107_404, true)
        );
        assert_eq!(r.spare.to_bits(), 0x3fa6_7cac_3de3_94e4);

        let g2 = r.next_gaussian();
        assert_eq!(g2.to_bits(), 0x3fa6_7cac_3de3_94e4);
        assert!(!r.have_spare);
    }

    #[test]
    fn next_gaussian_spare_passthrough_matches_real_v8() {
        // Sanity check for the two tests above: the `have_spare` fast path (no draws, no retry
        // loop involved at all) still matches real V8 bit-for-bit for an arbitrary crafted state.
        let mut r = Rng::from_state(11, 22, 33, 44, true, f64::from_bits(0x3fbf_9add_3739_635f));
        let g1 = r.next_gaussian();
        assert_eq!(g1.to_bits(), 0x3fbf_9add_3739_635f);
        assert_eq!((r.s0, r.s1, r.s2, r.s3, r.have_spare), (11, 22, 33, 44, false));
    }
}
