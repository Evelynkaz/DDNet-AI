// [`scalbn`] below is adapted from fdlibm's `s_scalbn.c` (http://www.netlib.org/fdlibm), not from
// V8 (V8 itself never vendors `scalbn` — it calls the platform's, see the doc comment below), so
// it carries fdlibm's own notice, not V8's:
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
// Altered: ported from C (fdlibm's `s_scalbn.c`) to Rust for DDNet-AI, using safe
// `f64::to_bits`/`from_bits` (see this file's other helpers) instead of the original's
// pointer-aliasing macros. Same operations, same order, no `unsafe`.

//! Bit-level helpers shared by the fdlibm ports, replacing the C macros in V8's
//! `src/base/ieee754.cc` (`EXTRACT_WORDS`, `GET_HIGH_WORD`, `GET_LOW_WORD`, `INSERT_WORDS`,
//! `SET_HIGH_WORD` — `SET_LOW_WORD`'s one call site, in `__ieee754_rem_pio2`, is ported as a
//! direct [`insert_words`] instead, see that function's comment) with equivalent operations on
//! [`f64::to_bits`] / [`f64::from_bits`]. The original macros dig the high/low 32-bit halves out
//! of a `double`
//! through pointer aliasing (`base::bit_cast`, safe in C++ but requiring `unsafe` transmutes if
//! ported literally in Rust); `to_bits`/`from_bits` give the exact same 64-bit pattern without
//! any `unsafe` at all, so the port below is behaviorally identical, just without pointer tricks.
//!
//! Also carries [`scalbn`]: V8's `__kernel_rem_pio2` calls the *platform* `scalbn`/`floor`/`fabs`
//! (via `#include <cmath>`, unqualified — resolving to glibc's on this target) rather than a
//! ported fdlibm routine, since `scalbn`/`floor`/`fabs` are simple, exactly-defined bit/exponent
//! operations with a single correct IEEE-754 result (no transcendental approximation, hence no
//! room for a glibc/fdlibm divergence the way `sin`/`cos`/`log` have). Rust's [`f64::floor`] and
//! [`f64::abs`] already match glibc exactly for that reason; `scalbn` has no Rust `std`
//! equivalent, so [`scalbn`] below is a direct, literal port of fdlibm's own `s_scalbn.c`
//! (same constants — `two54`/`twom54`/`huge`/`tiny` — and the same `+-50000` overflow/underflow
//! guards), not an independent reimplementation from general knowledge of the algorithm's shape
//! — see the license header above this doc comment.

/// High 32 bits of `d`'s IEEE-754 bit pattern (sign + exponent + top 20 mantissa bits), as an
/// `i32` — replaces `GET_HIGH_WORD(i, d)`.
#[inline]
pub(super) fn get_high_word(d: f64) -> i32 {
    (d.to_bits() >> 32) as i32
}

/// Low 32 bits of `d`'s IEEE-754 bit pattern (bottom 32 mantissa bits), as a `u32` — replaces
/// `GET_LOW_WORD(i, d)`.
#[inline]
pub(super) fn get_low_word(d: f64) -> u32 {
    (d.to_bits() & 0xFFFF_FFFF) as u32
}

/// `(high, low)` 32-bit halves of `d`'s IEEE-754 bit pattern — replaces `EXTRACT_WORDS(ix0, ix1,
/// d)`.
#[inline]
pub(super) fn extract_words(d: f64) -> (i32, u32) {
    (get_high_word(d), get_low_word(d))
}

/// Builds an `f64` from its high/low 32-bit halves — replaces `INSERT_WORDS(d, ix0, ix1)`.
#[inline]
pub(super) fn insert_words(hi: i32, lo: u32) -> f64 {
    f64::from_bits(((hi as u32 as u64) << 32) | (lo as u64))
}

/// Replaces the high 32 bits of `d` with `v`, keeping the low 32 bits — replaces
/// `SET_HIGH_WORD(d, v)`.
#[inline]
pub(super) fn set_high_word(d: f64, v: i32) -> f64 {
    let bits = (d.to_bits() & 0x0000_0000_FFFF_FFFF) | ((v as u32 as u64) << 32);
    f64::from_bits(bits)
}

/// `scalbn(x, n)` = `x * 2^n`, computed exactly via the exponent field (so it correctly
/// overflows/underflows exactly where real `scalbn` would, unlike `x * 2f64.powi(n)`, which can
/// prematurely overflow when `n` alone is out of `f64`'s exponent range even though the true
/// product would still be finite).
///
/// This is the standard fdlibm/musl `scalbn` algorithm (the same one glibc's `s_scalbn.c`
/// implements — it is a simple, exactly-defined exponent-bit operation with a single correct
/// IEEE-754 result, so there is no glibc-vs-fdlibm divergence to worry about here, unlike the
/// transcendental functions). `n` is always small in `__kernel_rem_pio2`'s call sites (well within
/// the `+-50000` guard below), so the "integer overflow in n" edge case is unreachable there, but
/// is kept for a faithful, general-purpose port.
pub(super) fn scalbn(mut x: f64, n: i32) -> f64 {
    const TWO54: f64 = 1.801_439_850_948_198_4e16; // 0x4350000000000000
    const TWOM54: f64 = 5.551_115_123_125_782_7e-17; // 0x3C90000000000000
    const HUGE: f64 = 1.0e300;
    const TINY: f64 = 1.0e-300;

    let mut hx = get_high_word(x);
    let mut k = (hx & 0x7FF0_0000) >> 20; // extract exponent
    if k == 0 {
        // 0 or subnormal x
        if (get_low_word(x) | ((hx & 0x7FFF_FFFF) as u32)) == 0 {
            return x; // +-0
        }
        x *= TWO54;
        hx = get_high_word(x);
        k = ((hx & 0x7FF0_0000) >> 20) - 54;
        if n < -50_000 {
            return TINY * x; // underflow
        }
    }
    if k == 0x7FF {
        return x + x; // NaN or Inf
    }
    k += n;
    if k > 0x7FE {
        return HUGE * HUGE.copysign(x); // overflow
    }
    const SIGN_AND_LOW_MANTISSA: i32 = 0x800F_FFFFu32 as i32;
    if k > 0 {
        // normal result
        return set_high_word(x, (hx & SIGN_AND_LOW_MANTISSA) | (k << 20));
    }
    if k <= -54 {
        return if n > 50_000 {
            HUGE * HUGE.copysign(x) // overflow (n+k integer overflow guard)
        } else {
            TINY * TINY.copysign(x) // underflow
        };
    }
    k += 54; // subnormal result
    let x = set_high_word(x, (hx & SIGN_AND_LOW_MANTISSA) | (k << 20));
    x * TWOM54
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_roundtrip() {
        for x in [0.0, -0.0, 1.0, -1.0, f64::NAN, f64::INFINITY, 1.5e300, 5e-300] {
            let (hi, lo) = extract_words(x);
            assert_eq!(insert_words(hi, lo).to_bits(), x.to_bits());
        }
    }

    #[test]
    fn scalbn_matches_ldexp_in_normal_range() {
        assert_eq!(scalbn(1.0, 10), 1024.0);
        assert_eq!(scalbn(1.0, -1), 0.5);
        assert_eq!(scalbn(0.0, 5), 0.0);
        assert_eq!(scalbn(-0.0, 5).to_bits(), (-0.0f64).to_bits());
        assert_eq!(scalbn(3.0, 0), 3.0);
    }

    #[test]
    fn scalbn_overflows_and_underflows_like_real_scalbn() {
        assert!(scalbn(1.0, 2000).is_infinite());
        assert_eq!(scalbn(1.0, -2000), 0.0);
        // A tiny value scaled up by a huge exponent that alone would overflow `2^n`, but the true
        // product is still finite: this is exactly the case a naive `x * 2f64.powi(n)` gets wrong.
        assert!(scalbn(5e-300, 1050).is_finite());
    }
}
