// Ported from DDNet 20.1 `src/game/prng.h`/`prng.cpp` (`CPrng`, PCG-XSH-RR with 64-bit state and
// 32-bit output — from
// <https://en.wikipedia.org/w/index.php?title=Permuted_congruential_generator&oldid=901497400#Example_code>,
// as the original C++ file's own comment cites). `prng.h`/`prng.cpp` carry no license header of
// their own in the 20.1 tree (unlike `gamecore`/`collision`, whose files start with DDNet's zlib
// notice) — see `NOTICE` for this repository's attribution of the wider DDNet codebase this file
// is still a derived port of.
//
// Altered for DDNet-AI: rewritten in Rust; `Description()` (a debug/logging helper, never called
// by `gamecore.cpp`/`collision.cpp`) is not ported — nothing in this crate's physics needs it.
//
// This is `CWorldCore::m_pPrng`'s target type: the *world* PRNG a real DDNet server seeds from
// its own secure-random source and uses (via `CWorldCore::RandomOr0`) exactly once in core-level
// physics — picking which `TELEOUT` a hook that flew through a `TILE_TELEINHOOK` comes out of
// when more than one exists for that number. Oracle A (task 1.2) never seeds one (`m_pPrng ==
// nullptr`, see `docs/formats.md` §5.4), so `core_world` always runs with `prng: None`; this
// module exists so a real DDRace `World` (task 1.6) has a faithful, seedable PRNG to plug in.

/// `CPrng`: seeded, then [`Prng::random_bits`] repeatedly to draw 32-bit values. Matches
/// `RandomBits()`'s `dbg_assert(m_Seeded, ...)` by panicking if called before [`Prng::seed`].
#[derive(Debug, Clone, Default)]
pub struct Prng {
    seeded: bool,
    state: u64,
    increment: u64,
}

const MULTIPLIER: u64 = 6364136223846793005;

impl Prng {
    /// `CPrng()`: an unseeded instance.
    pub fn new() -> Self {
        Prng::default()
    }

    /// `CPrng::Seed(uint64_t aSeed[2])`: `m_Increment = (aSeed[1] << 1) | 1`, `m_State =
    /// aSeed[0] + m_Increment`, then discards one `RandomBits()` draw (matching the C++ source,
    /// which calls `RandomBits()` once at the end of `Seed` for reasons it doesn't comment on —
    /// ported as-is since it's part of the observable sequence).
    pub fn seed(&mut self, seed: [u64; 2]) {
        self.seeded = true;
        self.increment = (seed[1] << 1) | 1;
        self.state = seed[0].wrapping_add(self.increment);
        self.random_bits();
    }

    /// `CPrng::RandomBits()`.
    ///
    /// # Panics
    ///
    /// If [`Prng::seed`] has never been called (matches the C++ `dbg_assert`).
    pub fn random_bits(&mut self) -> u32 {
        assert!(
            self.seeded,
            "prng needs to be seeded before it can generate random numbers"
        );
        let x = self.state;
        let count = (x >> 59) as u32;
        self.state = x.wrapping_mul(MULTIPLIER).wrapping_add(self.increment);
        let x = x ^ (x >> 18);
        let truncated = (x >> 27) as u32;
        // `RotateRight32(x, Shift)` (`prng.cpp`) — `Count` (`x >> 59` of a `u64`) is always in
        // `0..=31`, so this is exactly a 32-bit right-rotate by `Count` bits.
        truncated.rotate_right(count)
    }

    /// Whether [`Prng::seed`] has been called.
    pub fn is_seeded(&self) -> bool {
        self.seeded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "prng needs to be seeded")]
    fn random_bits_panics_before_seeding() {
        let mut p = Prng::new();
        p.random_bits();
    }

    #[test]
    fn new_prng_is_not_seeded() {
        assert!(!Prng::new().is_seeded());
    }

    #[test]
    fn seed_marks_seeded() {
        let mut p = Prng::new();
        p.seed([1, 2]);
        assert!(p.is_seeded());
    }

    #[test]
    fn same_seed_gives_the_same_sequence() {
        let mut a = Prng::new();
        a.seed([42, 7]);
        let mut b = Prng::new();
        b.seed([42, 7]);
        for _ in 0..20 {
            assert_eq!(a.random_bits(), b.random_bits());
        }
    }

    #[test]
    fn different_seeds_give_different_sequences() {
        let mut a = Prng::new();
        a.seed([1, 1]);
        let mut b = Prng::new();
        b.seed([2, 1]);
        let seq_a: Vec<u32> = (0..10).map(|_| a.random_bits()).collect();
        let seq_b: Vec<u32> = (0..10).map(|_| b.random_bits()).collect();
        assert_ne!(seq_a, seq_b);
    }

    // Reference vectors computed from an independent Python re-implementation of the exact same
    // algorithm (PCG-XSH-RR, 64-bit state / 32-bit output — see the module doc comment) — pins
    // this port against a second, from-scratch reading of the spec rather than only against
    // itself.
    #[test]
    fn matches_independent_reference_implementation_seed_42_7() {
        let mut p = Prng::new();
        p.seed([42, 7]);
        let got: Vec<u32> = (0..5).map(|_| p.random_bits()).collect();
        assert_eq!(got, vec![1956239935, 1010964048, 2769188248, 3076816759, 888960798]);
    }

    #[test]
    fn matches_independent_reference_implementation_seed_0_0() {
        let mut p = Prng::new();
        p.seed([0, 0]);
        let got: Vec<u32> = (0..5).map(|_| p.random_bits()).collect();
        assert_eq!(got, vec![3837872008, 932996374, 1548399547, 1612522464, 473443212]);
    }

    #[test]
    fn matches_independent_reference_implementation_large_seed() {
        let mut p = Prng::new();
        p.seed([0xDEADBEEF12345678, 0x1]);
        let got: Vec<u32> = (0..3).map(|_| p.random_bits()).collect();
        assert_eq!(got, vec![3948654854, 2352441614, 103549412]);
    }

    /// Known-answer test against the *real* DDNet C++ `CPrng`, not just an independent
    /// re-implementation (residual finding from review round 2: "Oracle A never seeds one" so
    /// the bulk/fuzz parity tests never exercise this type). Reproduces
    /// `src/test/prng_test.cpp`'s `Prng.EqualsPcg32GlobalDemo` from the pinned DDNet 20.1 tree
    /// verbatim: `CPrng Prng; Prng.Seed({42, 54}); for (auto Expected : PCG32_GLOBAL_DEMO)
    /// EXPECT_EQ(Prng.RandomBits(), Expected);`. `PCG32_GLOBAL_DEMO` itself is the first 6
    /// `uint32_t`s of `pcg32-global-demo.c`'s "Round 1: 32bit:" line (seed 42/54) from
    /// <https://www.pcg-random.org/using-pcg-c-basic.html> — a reference independent of both
    /// DDNet and this crate, so this single test cross-checks three independent
    /// implementations (upstream pcg-c-basic, DDNet's `CPrng`, and this port) at once.
    #[test]
    fn matches_real_ddnet_cprng_prng_test_cpp_equals_pcg32_global_demo() {
        let mut p = Prng::new();
        p.seed([42, 54]);
        let got: Vec<u32> = (0..6).map(|_| p.random_bits()).collect();
        assert_eq!(
            got,
            vec![0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e]
        );
    }
}
