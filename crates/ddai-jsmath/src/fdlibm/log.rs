// The following is adapted from fdlibm (http://www.netlib.org/fdlibm).
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
// The original source code covered by the above license above has been
// modified significantly by Google Inc.
// Copyright 2016 the V8 project authors. All rights reserved.
//
// Altered: ported from C++ (V8's `src/base/ieee754.cc`, Node 24.21.0 / V8 13.6 tag) to Rust for
// DDNet-AI. Same operations, same order, no `unsafe`, no FMA contraction.

//! `log`, ported line-for-line from `deps/v8/src/base/ieee754.cc:1638-1717`. This is the one
//! transcendental the old bot's `Rng::next_gaussian` needs (`src/nn/rng.ts:61`,
//! `Math.sqrt(-2 * Math.log(u))`).

use super::bits::{extract_words, get_high_word, set_high_word};

const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-1;
const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
const TWO54: f64 = 1.801_439_850_948_198_400_00e16;
const LG1: f64 = 6.666_666_666_666_735_130e-1;
const LG2: f64 = 3.999_999_999_940_941_908e-1;
const LG3: f64 = 2.857_142_874_366_239_149e-1;
const LG4: f64 = 2.222_219_843_214_978_396e-1;
const LG5: f64 = 1.818_357_216_161_805_012e-1;
const LG6: f64 = 1.531_383_769_920_937_332e-1;
const LG7: f64 = 1.479_819_860_511_658_591e-1;

/// `log(x)`. `deps/v8/src/base/ieee754.cc:1638-1717`.
///
/// V8 returns a *signaling* NaN for `log` of a negative number
/// (`std::numeric_limits<double>::signaling_NaN()`); this port returns the ordinary quiet
/// `f64::NAN` instead — the task's own proof methodology explicitly treats "both NaN" as a match
/// regardless of payload/signaling bit (see the crate README and `tools/jsmath-oracle`), and a
/// signaling NaN surviving a JS `Number` round-trip unquieted is not something the old bot's
/// scoring can observe or depend on either way.
pub fn log(mut x: f64) -> f64 {
    let (mut hx, lx) = extract_words(x);

    let mut k: i32 = 0;
    if hx < 0x0010_0000 {
        // x < 2**-1022
        if ((hx & 0x7FFF_FFFF) | lx as i32) == 0 {
            return f64::NEG_INFINITY; // log(+-0) = -inf
        }
        if hx < 0 {
            return f64::NAN; // log(-#) = NaN (a quiet NaN here, see doc comment above)
        }
        k -= 54;
        x *= TWO54; // subnormal number, scale up x
        hx = get_high_word(x);
    }
    if hx >= 0x7FF0_0000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000F_FFFF;
    let i = (hx + 0x9_5F64) & 0x10_0000;
    x = set_high_word(x, hx | (i ^ 0x3FF0_0000)); // normalize x or x/2
    k += i >> 20;
    let f = x - 1.0;
    if (0x000F_FFFF & (2 + hx)) < 3 {
        // -2**-20 <= f < 2**-20
        if f == 0.0 {
            return if k == 0 {
                0.0
            } else {
                let dk = f64::from(k);
                dk * LN2_HI + dk * LN2_LO
            };
        }
        let r = f * f * (0.5 - 0.333_333_333_333_333_33 * f);
        return if k == 0 {
            f - r
        } else {
            let dk = f64::from(k);
            dk * LN2_HI - ((r - dk * LN2_LO) - f)
        };
    }
    let s = f / (2.0 + f);
    let dk = f64::from(k);
    let z = s * s;
    let i = hx - 0x6_147A;
    let w = z * z;
    let j = 0x6_B851 - hx;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let i = i | j;
    let r = t2 + t1;
    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - (hfsq - s * (hfsq + r))
        } else {
            dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
        }
    } else if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_points() {
        assert_eq!(log(1.0), 0.0);
        assert_eq!(log(0.0), f64::NEG_INFINITY);
        assert_eq!(log(-0.0), f64::NEG_INFINITY);
        assert!(log(-1.0).is_nan());
        assert!(log(f64::NAN).is_nan());
        assert!((log(std::f64::consts::E) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn log_of_infinity() {
        assert_eq!(log(f64::INFINITY), f64::INFINITY);
    }
}
