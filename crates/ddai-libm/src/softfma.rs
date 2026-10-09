//! A correctly rounded fused multiply-add in safe integer arithmetic, and the choice between it and the
//! platform's `fma` (task 5.5a review, F2).
//!
//! The ports in this crate reproduce glibc's FMA variants, so they need `fma(a, b, c) = round(a * b + c)` with
//! a single, correct rounding. `f64::mul_add` is that on a target that has the instruction; on a target
//! without it (the default x86-64 builds, see D-001) it is a call to the C library's `fma`, and C libraries
//! differ: glibc's is correct (and is what the reference results come from); mingw-w64's is known to be wrong
//! in corner cases (Rust issue 140515, mingw-w64 bug 848); the MSVC UCRT's was found wrong on ordinary inputs
//! by the first Windows CI run of this crate (`log` gave other bits than glibc on `windows-latest`, which has
//! FMA hardware: the UCRT's `fma` is not reliable even there). A wrong `fma` changes the last bit of
//! `sinf`/`powf`/`log`/`atan2`/`pow` for a few inputs and the physics silently drifts from the server. So
//! [`mode`] decides once per process which `fma` the ports use, and **the C library's is used only where it is
//! known to be glibc's**:
//!
//! * the target has the `fma` feature at compile time (`-C target-feature=+fma`): the instruction, no check;
//! * `linux-gnu` (glibc), on a CPU with FMA: glibc's `fma`, after it agreed with the software one on a fixed
//!   set of corner-case operands (overflow, underflow, subnormals, cancellation, ties, signed zeros);
//! * everywhere else (Windows, musl, macOS, a CPU without FMA, the `force-soft-fma` feature): the software
//!   `fma` of this module.
//!
//! The software `fma` costs about 54 ns per call instead of 5 ns (D-127 has the numbers, about +7% on
//! `World::step`). A faster path on Windows needs `unsafe` (a `#[target_feature(enable = "fma")]` copy of every port,
//! entered after `is_x86_feature_detected!`), which this crate does not have; a build for CPUs known to have FMA can
//! use `-C target-feature=+fma` instead.
//!
//! The software `fma` is written from scratch (exact 128-bit integer arithmetic, one rounding at the end); it
//! is not derived from musl's or any other implementation. Its unit tests probe it against the hardware
//! instruction on Linux (10^9 operand triples, edge cases included, `--ignored`).

use std::sync::atomic::{AtomicU8, Ordering};

/// Which `fma` the ports use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FmaMode {
    /// Compiled with the `fma` target feature: `mul_add` is the instruction.
    Instruction,
    /// glibc's `fma` on `linux-gnu` (checked against the software one at first use).
    CLibrary,
    /// The software `fma` of this module: any platform but `linux-gnu`, a CPU without FMA, a glibc `fma` that failed the
    /// check, or the `force-soft-fma` feature.
    Software,
}

const UNDECIDED: u8 = 0;
const LIBRARY: u8 = 1;
const SOFTWARE: u8 = 2;

static MODE: AtomicU8 = AtomicU8::new(UNDECIDED);

/// What the ports use on this machine (decides on first call; the decision is final for the process).
#[must_use]
pub fn mode() -> FmaMode {
    if cfg!(all(target_feature = "fma", not(feature = "force-soft-fma"))) {
        return FmaMode::Instruction;
    }
    match MODE.load(Ordering::Relaxed) {
        LIBRARY => FmaMode::CLibrary,
        SOFTWARE => FmaMode::Software,
        _ => {
            decide();
            mode()
        }
    }
}

/// What the ports use and why, for the start-up log.
#[must_use]
pub fn mode_description() -> &'static str {
    match mode() {
        FmaMode::Instruction => "fma instruction (compile-time target feature)",
        FmaMode::CLibrary => "glibc fma (agrees with the software fma on the self-check)",
        FmaMode::Software => {
            if cfg!(feature = "force-soft-fma") {
                "software fma (forced by the force-soft-fma feature)"
            } else if !C_LIBRARY_IS_GLIBC {
                "software fma (the C library of this platform is not glibc; its fma is not trusted)"
            } else if cpu_lacks_fma() {
                "software fma (this CPU has no FMA)"
            } else {
                "software fma (the C library's fma failed the self-check)"
            }
        }
    }
}

/// Whether the C library behind `f64::mul_add` is glibc's (the only one whose `fma` is known to be correct).
const C_LIBRARY_IS_GLIBC: bool = cfg!(all(target_os = "linux", target_env = "gnu"));

fn cpu_lacks_fma() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        !std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        // Every other architecture this runs on (aarch64, riscv64gc with the F/D extensions' fused forms) has a
        // fused multiply-add; the self-check below still guards the C library's.
        false
    }
}

#[cold]
fn decide() {
    let mode = if cfg!(feature = "force-soft-fma")
        || !C_LIBRARY_IS_GLIBC
        || cpu_lacks_fma()
        || !library_fma_agrees(f64::mul_add)
    {
        SOFTWARE
    } else {
        LIBRARY
    };
    MODE.store(mode, Ordering::Relaxed);
}

/// Whether the ports run with [`soft_fma`] (decides on first call, like [`mode`]); one relaxed atomic load.
#[inline(always)]
pub(crate) fn use_soft() -> bool {
    if cfg!(all(target_feature = "fma", not(feature = "force-soft-fma"))) {
        return false;
    }
    match MODE.load(Ordering::Relaxed) {
        LIBRARY => false,
        SOFTWARE => true,
        _ => {
            decide();
            use_soft()
        }
    }
}

/// `a * b + c` with one rounding, as the ports use it (see [`mode`] for which implementation runs).
#[inline(always)]
#[must_use]
pub fn fma(a: f64, b: f64, c: f64) -> f64 {
    if cfg!(all(target_feature = "fma", not(feature = "force-soft-fma"))) {
        return a.mul_add(b, c);
    }
    match MODE.load(Ordering::Relaxed) {
        LIBRARY => a.mul_add(b, c),
        SOFTWARE => soft_fma(a, b, c),
        _ => {
            decide();
            fma(a, b, c)
        }
    }
}

/// Does `candidate` (the C library's `fma`, or a stand-in in the tests) return the software `fma`'s bits on
/// the self-check operands? NaN results only have to be NaN (their payload depends on the instruction form).
pub(crate) fn library_fma_agrees(candidate: impl Fn(f64, f64, f64) -> f64) -> bool {
    check_operands().all(|(a, b, c)| {
        let want = soft_fma(a, b, c);
        let got = candidate(a, b, c);
        want.to_bits() == got.to_bits() || (want.is_nan() && got.is_nan())
    })
}

/// The fixed operand triples of the self-check: every combination of the special values, then a few
/// thousand pseudo-random triples of each corner-case family.
pub(crate) fn check_operands() -> impl Iterator<Item = (f64, f64, f64)> {
    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        3.0,
        f64::MAX,
        -f64::MAX,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::EPSILON,
        1.0 + f64::EPSILON,
        1.0 - f64::EPSILON / 2.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::from_bits(0x1a70_0000_0000_0000), // 2^-600
        f64::from_bits(0x7fe0_0000_0000_0000),
    ];
    let cube = specials.into_iter().flat_map(move |a| {
        specials
            .into_iter()
            .flat_map(move |b| specials.into_iter().map(move |c| (a, b, c)))
    });
    cube.chain(Families::new(0x5eed_f00d_0123_4567, 4000))
}

/// Pseudo-random triples of the corner-case families, `per_family` of each.
pub(crate) struct Families {
    state: u64,
    left: usize,
    per_family: usize,
    family: u8,
}

impl Families {
    pub(crate) fn new(seed: u64, per_family: usize) -> Self {
        Self {
            state: seed,
            left: per_family,
            per_family,
            family: 0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A finite double with the given biased exponent and a random mantissa.
    fn with_exp(&mut self, biased: u64) -> f64 {
        let r = self.next_u64();
        f64::from_bits(((r & 1) << 63) | (biased << 52) | ((r >> 12) & 0x000f_ffff_ffff_ffff))
    }
}

pub(crate) const FAMILIES: u8 = 9;

impl Iterator for Families {
    type Item = (f64, f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.left == 0 {
            self.family += 1;
            self.left = self.per_family;
        }
        if self.family >= FAMILIES {
            return None;
        }
        self.left -= 1;
        let r = self.next_u64();
        Some(match self.family {
            // All bits random (mostly huge or tiny, sometimes inf/NaN).
            0 => (
                f64::from_bits(self.next_u64()),
                f64::from_bits(self.next_u64()),
                f64::from_bits(r),
            ),
            // Moderate exponents: ordinary fma.
            1 => {
                let (ea, eb, ec) = (
                    1023 - 30 + r % 60,
                    1023 - 30 + (r >> 8) % 60,
                    1023 - 60 + (r >> 16) % 120,
                );
                (self.with_exp(ea), self.with_exp(eb), self.with_exp(ec))
            }
            // Cancellation: c is minus the rounded product, nudged by a few ulps.
            2 => {
                let (ea, eb) = (1023 - 200 + r % 400, 1023 - 200 + (r >> 10) % 400);
                let (a, b) = (self.with_exp(ea), self.with_exp(eb));
                let nudge = (self.next_u64() % 9) as i64 - 4;
                let c = f64::from_bits((-(a * b)).to_bits().wrapping_add(nudge as u64));
                (a, b, c)
            }
            // Results in the subnormal range.
            3 => {
                let ea = 1023 - 600 + r % 400;
                let eb = 1023 - 520 - (r >> 12) % 100 - (ea.saturating_sub(1023 - 600)) / 4;
                let (a, b) = (self.with_exp(ea), self.with_exp(eb.max(1)));
                let c = if (r >> 40) & 1 == 0 {
                    f64::from_bits(((r >> 3) & 1) << 63 | (self.next_u64() & 0x000f_ffff_ffff_ffff))
                } else {
                    self.with_exp((r >> 20) % 3)
                };
                (a, b, c)
            }
            // Overflow and near-overflow.
            4 => {
                let (ea, eb) = (1023 + 400 + r % 100, 1023 + 400 - (r >> 8) % 100);
                let (a, b) = (self.with_exp(ea.min(2046)), self.with_exp(eb.clamp(1, 2046)));
                let c = self.with_exp(2046 - (r >> 20) % 4);
                (a, b, if (r >> 30) & 1 == 0 { c } else { -c })
            }
            // Few significant bits: exact products and exact ties.
            5 => {
                let small = |s: &mut Self, bits: u32| {
                    let m = s.next_u64() & ((1u64 << bits) - 1);
                    let e = (s.next_u64() % 40) as i32 - 20;
                    (m as f64) * 2f64.powi(e) * if s.next_u64() & 1 == 0 { 1.0 } else { -1.0 }
                };
                let (a, b) = (small(self, 27), small(self, 26));
                let c = if (r >> 33) & 3 == 0 { -(a * b) } else { small(self, 53) };
                (a, b, c)
            }
            // Large exponent gap between product and addend (sticky handling).
            6 => {
                let (ea, eb) = (1023 - 100 + r % 200, 1023 - 100 + (r >> 10) % 200);
                let gap = 50 + (r >> 20) % 1000;
                let ec = (ea + eb)
                    .saturating_sub(1023)
                    .saturating_add(if (r >> 40) & 1 == 0 { gap } else { 0 });
                let ec = if (r >> 41) & 1 == 0 {
                    ec.clamp(1, 2046)
                } else {
                    (ea + eb).saturating_sub(1023 + gap).max(1)
                };
                (self.with_exp(ea), self.with_exp(eb), self.with_exp(ec))
            }
            // A product that is an exact tie at 53 bits (54 bits, odd) and a tiny addend of either sign that breaks the tie: the
            // sticky bit decides the rounding, and a borrow decides it in the other direction.
            8 => {
                let odd27 = |s: &mut Self| ((s.next_u64() & ((1 << 26) - 1)) | (1 << 26) | 1) as f64;
                let (a, b) = (odd27(self), odd27(self));
                let sign = |s: &mut Self| if s.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                let scale = 2f64.powi(((r >> 7) % 61) as i32 - 30);
                let k = 1 + (r >> 20) % 200;
                let c = sign(self) * (1 + (r >> 40) % 7) as f64 * 2f64.powi(-(k as i32)) * scale;
                (a * sign(self) * scale, b, c)
            }
            // Exponents summing to the subnormal boundary exactly, c zero or tiny.
            _ => {
                let ea = 1023 - 300 + r % 600;
                let eb = (2046 - ea).saturating_sub(40 + (r >> 12) % 4).clamp(1, 2046);
                let (a, b) = (self.with_exp(ea), self.with_exp(eb.min(1023 - 300 + (r >> 20) % 600)));
                let c = [0.0, -0.0, f64::from_bits(1), -f64::from_bits(1), f64::MIN_POSITIVE][((r >> 50) % 5) as usize];
                (a, b, c)
            }
        })
    }
}

/// `(sign, mantissa, exponent)` of a finite non-zero double: `x = (-1)^sign * mantissa * 2^exponent`, with the
/// mantissa an integer below 2^53.
fn decompose(x: f64) -> (bool, u64, i32) {
    let bits = x.to_bits();
    let neg = bits >> 63 != 0;
    let field = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    if field == 0 {
        (neg, frac, -1074)
    } else {
        (neg, frac | (1 << 52), field - 1075)
    }
}

/// A finite non-zero value held as `m * 2^e` with the top bit of `m` at bit 125 (room for a carry, and 72
/// guard bits below a double's mantissa).
#[derive(Clone, Copy)]
struct Wide {
    neg: bool,
    m: u128,
    e: i32,
}

impl Wide {
    fn new(neg: bool, m: u128, e: i32) -> Self {
        let shift = m.leading_zeros() as i32 - 2;
        Self {
            neg,
            m: m << shift,
            e: e - shift,
        }
    }
}

/// `round(a * b + c)` with one rounding to nearest, ties to even: the C `fma` in round-to-nearest mode, in
/// safe integer arithmetic. Special values follow IEEE 754 (a NaN result is a quiet NaN; its payload is not
/// promised to match any particular instruction).
#[must_use]
pub fn soft_fma(a: f64, b: f64, c: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        if a.is_finite() && b.is_finite() {
            // The addend is infinite or NaN and the product is finite (possibly too large for a double, which
            // does not matter): the result is the addend.
            return c + c;
        }
        // An infinite or NaN factor: the product is infinite or NaN whatever the addend (or the addend
        // cancels it to NaN), which the plain operations give.
        return a * b + c;
    }
    if a == 0.0 || b == 0.0 {
        // The product is an exact signed zero: the plain sum is exact too, signs of zero included.
        return a * b + c;
    }

    let (sa, ma, ea) = decompose(a);
    let (sb, mb, eb) = decompose(b);
    let prod = Wide::new(sa != sb, u128::from(ma) * u128::from(mb), ea + eb);
    if c == 0.0 {
        return round(prod);
    }
    let (sc, mc, ec) = decompose(c);
    let addend = Wide::new(sc, u128::from(mc), ec);

    // Align the smaller on the larger (both have their top bit at 125, so the larger exponent is the larger
    // magnitude); what falls off the bottom is folded into the lowest bit ("jamming"), which is enough for
    // the final rounding because at least 70 bits stay below the double's mantissa.
    let (hi, lo) = if prod.e >= addend.e {
        (prod, addend)
    } else {
        (addend, prod)
    };
    let d = hi.e - lo.e;
    let lo_m = if d == 0 {
        lo.m
    } else if d >= 127 {
        1
    } else {
        let lost = lo.m & ((1u128 << d) - 1) != 0;
        (lo.m >> d) | u128::from(lost)
    };
    let sum = if hi.neg == lo.neg {
        Wide {
            neg: hi.neg,
            m: hi.m + lo_m,
            e: hi.e,
        }
    } else if hi.m > lo_m {
        Wide {
            neg: hi.neg,
            m: hi.m - lo_m,
            e: hi.e,
        }
    } else if hi.m < lo_m {
        Wide {
            neg: lo.neg,
            m: lo_m - hi.m,
            e: hi.e,
        }
    } else {
        return 0.0; // exact cancellation is +0 in round-to-nearest
    };
    round(sum)
}

/// Round `w` (non-zero) to the nearest double, ties to even, handling subnormals and overflow.
fn round(w: Wide) -> f64 {
    let top = 127 - w.m.leading_zeros() as i32; // index of the top bit
    let exp = w.e + top; // the value is in [2^exp, 2^(exp+1))
    let sign = u64::from(w.neg) << 63;
    if exp >= 1024 {
        return f64::from_bits(sign | 0x7ff0_0000_0000_0000);
    }
    // Exponent of the lowest bit that survives: 52 below the top for a normal result, -1074 for a subnormal.
    let q = (exp - 52).max(-1074);
    let shift = q - w.e;
    let mantissa: u128 = if shift > 0 {
        if shift >= 128 {
            0
        } else {
            let kept = w.m >> shift;
            let rest = w.m & ((1u128 << shift) - 1);
            let half = 1u128 << (shift - 1);
            if rest > half || (rest == half && kept & 1 == 1) {
                kept + 1
            } else {
                kept
            }
        }
    } else {
        w.m << (-shift)
    };
    // `(q + 1074) << 52` is the exponent field of the value `2^q * 2^52` minus one unit; the mantissa, which
    // carries the implicit bit for a normal result, supplies that unit (and a rounding carry to 2^53 bumps the
    // exponent by one, with a zero fraction: exactly right). For a subnormal result `q + 1074 == 0` and the
    // mantissa is the bit pattern (2^52 is the smallest normal).
    let bits = ((((q + 1074) as u64) << 52) + mantissa as u64) | sign;
    if bits & 0x7fff_ffff_ffff_ffff >= 0x7ff0_0000_0000_0000 {
        return f64::from_bits(sign | 0x7ff0_0000_0000_0000);
    }
    f64::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// FNV-1a over the result bits of `soft_fma` on [`check_operands`] (NaNs folded to one pattern).
    fn check_hash(f: impl Fn(f64, f64, f64) -> f64) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for (a, b, c) in check_operands() {
            let r = f(a, b, c);
            let bits = if r.is_nan() { 0x7ff8_0000_0000_0000 } else { r.to_bits() };
            for byte in bits.to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// Recorded from the hardware `vfmadd` of an x86-64 Linux machine (glibc 2.39's `fma`): the software
    /// `fma` must reproduce it on every platform, which is the test that does not need a trustworthy C
    /// library `fma` on the machine it runs on.
    const CHECK_HASH: u64 = 0x0237_66c5_217f_e914;

    #[test]
    fn the_software_fma_reproduces_the_recorded_hardware_results_everywhere() {
        let got = check_hash(soft_fma);
        assert_eq!(got, CHECK_HASH, "software fma hash {got:#018x}");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn the_recorded_hash_is_the_hardware_one() {
        assert_eq!(check_hash(f64::mul_add), CHECK_HASH);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn the_software_fma_equals_mul_add_on_the_check_operands() {
        let n = check_operands().count();
        assert!(n > 40_000, "{n}");
        for (a, b, c) in check_operands() {
            let (s, h) = (soft_fma(a, b, c), a.mul_add(b, c));
            assert!(
                same(s, h),
                "fma({a:e}, {b:e}, {c:e}): software {s:e} ({:#x}), hardware {h:e} ({:#x})",
                s.to_bits(),
                h.to_bits()
            );
        }
    }

    #[test]
    fn a_plain_multiply_add_fails_the_self_check() {
        // The mutation: an "fma" that rounds twice must be rejected (that is what would make the C library
        // `fma` of a broken platform fall back to the software one).
        assert!(!library_fma_agrees(|a, b, c| a * b + c));
        // And one that is only wrong in a corner case (overflow of the intermediate product) too.
        assert!(!library_fma_agrees(|a, b, c| {
            let p = a * b;
            if p.is_infinite() && a.is_finite() && b.is_finite() && c.is_finite() {
                p
            } else {
                soft_fma(a, b, c)
            }
        }));
        assert!(library_fma_agrees(soft_fma));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn glibc_fma_passes_the_self_check() {
        assert!(library_fma_agrees(f64::mul_add));
    }

    #[test]
    fn hand_picked_cases() {
        // 1 + eps/2 squared is 1 + eps + eps^2/4: the fused result keeps the tail the plain one drops.
        let a = 1.0 + f64::EPSILON;
        assert_eq!(
            soft_fma(a, 1.0 - f64::EPSILON / 2.0, -1.0),
            f64::EPSILON / 2.0 - f64::EPSILON * f64::EPSILON / 2.0
        );
        // Exact cancellation is +0; the sign of an exact zero sum follows IEEE.
        assert_eq!(soft_fma(2.0, 3.0, -6.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(soft_fma(-0.0, 1.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(soft_fma(0.0, 1.0, -0.0).to_bits(), 0.0f64.to_bits());
        // The product overflows a double but the sum does not.
        assert_eq!(soft_fma(f64::MAX, 2.0, -f64::MAX), f64::MAX);
        // Overflow to infinity, both signs.
        assert_eq!(soft_fma(f64::MAX, 2.0, 0.0), f64::INFINITY);
        assert_eq!(soft_fma(-f64::MAX, 2.0, 0.0), f64::NEG_INFINITY);
        // Underflow: half the smallest subnormal rounds to even (zero), a bit more rounds up.
        let tiny = f64::from_bits(1);
        assert_eq!(soft_fma(tiny, 0.5, 0.0).to_bits(), 0);
        assert_eq!(soft_fma(tiny, 0.5000000000000001, 0.0).to_bits(), 1);
        assert_eq!(soft_fma(-tiny, 0.5, 0.0).to_bits(), (-0.0f64).to_bits());
        // Gradual underflow into the normal range and a rounding carry across the boundary.
        assert_eq!(soft_fma(f64::MIN_POSITIVE, 0.5, 0.0).to_bits(), 1 << 51);
        assert_eq!(
            soft_fma(f64::from_bits(0x000f_ffff_ffff_ffff), 1.0, f64::from_bits(1)).to_bits(),
            f64::MIN_POSITIVE.to_bits()
        );
        // Specials.
        assert!(soft_fma(f64::INFINITY, 0.0, 1.0).is_nan());
        assert_eq!(soft_fma(2.0, 3.0, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(soft_fma(f64::INFINITY, 1.0, f64::NEG_INFINITY).is_nan());
        assert_eq!(soft_fma(f64::MAX, f64::MAX, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(soft_fma(1.0, 2.0, f64::NAN).is_nan());
    }

    #[test]
    fn the_probe_families_reach_the_corner_cases() {
        let (mut sub, mut inf, mut zero, mut nan, mut cancel, mut total) = (0, 0, 0, 0, 0, 0);
        for (a, b, c) in Families::new(7, 20_000) {
            total += 1;
            let r = soft_fma(a, b, c);
            if r.is_nan() {
                nan += 1;
            } else if r.is_infinite() {
                inf += 1;
            } else if r == 0.0 {
                zero += 1;
                if a.is_finite() && b.is_finite() && c.is_finite() && a != 0.0 && b != 0.0 && c != 0.0 {
                    cancel += 1;
                }
            } else if r.abs() < f64::MIN_POSITIVE {
                sub += 1;
            }
        }
        assert_eq!(total, 20_000 * usize::from(FAMILIES));
        // Each corner is hit thousands of times, not by luck.
        assert!(
            sub > 3_000 && inf > 2_000 && zero > 1_000 && nan > 10 && cancel > 1_000,
            "sub {sub} inf {inf} zero {zero} nan {nan} cancel {cancel}"
        );
    }

    /// 10^8 operand triples against the hardware instruction (`LIBM_FMA_PROBES=n` changes the number;
    /// release build recommended). Linux only: the reference is the C library's `fma`.
    #[test]
    #[ignore = "10^8 probes against the hardware fma (Linux)"]
    #[cfg(target_os = "linux")]
    fn soft_fma_vs_hardware_probe() {
        let total: u64 = std::env::var("LIBM_FMA_PROBES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100_000_000);
        let per = total / u64::from(FAMILIES) / 2;
        let mut n = 0u64;
        let mut bad = 0u64;
        for seed in 0..2u64 {
            let fam = Families::new(0x1234_5678_9abc_def0 ^ (seed << 60), per as usize);
            for (a, b, c) in fam {
                n += 1;
                let (s, h) = (soft_fma(a, b, c), a.mul_add(b, c));
                if !same(s, h) {
                    bad += 1;
                    if bad <= 10 {
                        eprintln!(
                            "fma({a:e}, {b:e}, {c:e}) [{:#x} {:#x} {:#x}]: software {:#x}, hardware {:#x}",
                            a.to_bits(),
                            b.to_bits(),
                            c.to_bits(),
                            s.to_bits(),
                            h.to_bits()
                        );
                    }
                }
            }
        }
        println!("soft_fma: {n} probes, {bad} mismatches");
        assert_eq!(bad, 0);
    }
}
