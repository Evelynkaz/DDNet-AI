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

//! `expm1` (`exp(x) - 1`, accurate for small `x`), ported line-for-line from
//! `deps/v8/src/base/ieee754.cc:2213-2331`. Not part of this crate's public API (the old bot never
//! calls `Math.expm1` directly), but needed as-is: V8's `tanh` calls `ieee754::expm1` internally
//! (see `tanh.rs`), and porting `tanh` faithfully means porting what it actually calls, not
//! reimplementing an equivalent formula from scratch.

use super::bits::{get_high_word, get_low_word, insert_words};

const TINY: f64 = 1.0e-300;
const O_THRESHOLD: f64 = 7.097_827_128_933_839_730_96e2;
const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-1;
const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
const INVLN2: f64 = 1.442_695_040_888_963_387_00;
// Scaled Q's: Qn_here = 2**n * Qn_above, for R(2*z) where z = hxs = x*x/2.
const Q1: f64 = -3.333_333_333_333_313_164_28e-2;
const Q2: f64 = 1.587_301_587_254_814_601_65e-3;
const Q3: f64 = -7.936_507_578_674_879_424_73e-5;
const Q4: f64 = 4.008_217_827_329_362_395_52e-6;
const Q5: f64 = -2.010_992_181_836_243_713_26e-7;
const HUGE: f64 = 1.0e300;

/// `expm1(x)`. `deps/v8/src/base/ieee754.cc:2213-2331`.
pub fn expm1(mut x: f64) -> f64 {
    let hx0 = get_high_word(x);
    let xsb = (hx0 as u32) & 0x8000_0000; // sign bit of x
    let hx = (hx0 & 0x7FFF_FFFF) as u32; // high word of |x|

    // Filter out huge and non-finite arguments.
    if hx >= 0x4043_687A {
        // |x| >= 56*ln2
        if hx >= 0x4086_2E42 {
            // |x| >= 709.78...
            if hx >= 0x7FF0_0000 {
                let low = get_low_word(x);
                if ((hx & 0xF_FFFF) | low) != 0 {
                    return x + x; // NaN
                }
                return if xsb == 0 { x } else { -1.0 }; // exp(+-inf) = {inf, -1}
            }
            if x > O_THRESHOLD {
                return HUGE * HUGE; // overflow
            }
        }
        if xsb != 0 {
            // x < -56*ln2, return -1.0 with inexact.
            if x + TINY < 0.0 {
                return TINY - 1.0; // raise inexact, return -1
            }
        }
    }

    // Argument reduction. `k`/`c` stay `0` (their spec-given defaults, "c is 0" per the source's
    // own comment further down) unless the `hx > 0x3FD62E42` branch below runs; that branch is
    // also the only one that reads back `hi`/`lo`, so — unlike the original C, which leaves all
    // four uninitialized and relies on every reachable path assigning them before use — this port
    // computes them as one `if`/`else` expression instead of pre-seeding dead placeholder values.
    let (k, c): (i32, f64);
    if hx > 0x3FD6_2E42 {
        // |x| > 0.5 ln2
        let (hi, lo);
        if hx < 0x3FF0_A2B2 {
            // and |x| < 1.5 ln2
            if xsb == 0 {
                hi = x - LN2_HI;
                lo = LN2_LO;
                k = 1;
            } else {
                hi = x + LN2_HI;
                lo = -LN2_LO;
                k = -1;
            }
        } else {
            k = (INVLN2 * x + if xsb == 0 { 0.5 } else { -0.5 }) as i32;
            let t = f64::from(k);
            hi = x - t * LN2_HI; // t*ln2_hi is exact here
            lo = t * LN2_LO;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if hx < 0x3C90_0000 {
        // |x| < 2**-54, return x.
        let t = HUGE + x; // return x with inexact flags when x != 0
        return x - (t - (HUGE + x));
    } else {
        k = 0;
        c = 0.0;
    }

    // x is now in the primary range.
    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let r1 = 1.0 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
    let t = 3.0 - r1 * hfx;
    let mut e = hxs * ((r1 - t) / (6.0 - x * t));
    if k == 0 {
        return x - (x * e - hxs); // c is 0
    }
    let twopk = insert_words(0x3FF0_0000i32.wrapping_add(k << 20), 0); // 2^k
    e = x * (e - c) - c;
    e -= hxs;
    if k == -1 {
        return 0.5 * (x - e) - 0.5;
    }
    if k == 1 {
        return if x < -0.25 {
            -2.0 * (e - (x + 0.5))
        } else {
            1.0 + 2.0 * (x - e)
        };
    }
    if k <= -2 || k > 56 {
        // Suffices to return exp(x) - 1.
        let mut y = 1.0 - (e - x);
        y = if k == 1024 {
            y * 2.0 * 8.988_465_674_311_58e307
        } else {
            y * twopk
        };
        return y - 1.0;
    }
    if k < 20 {
        let t = insert_words(0x3FF0_0000 - (0x0020_0000 >> k), 0); // t = 1 - 2^-k
        (t - (e - x)) * twopk
    } else {
        let t = insert_words((0x3FF - k) << 20, 0); // t = 2^-k
        (x - (e + t) + 1.0) * twopk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_points() {
        assert_eq!(expm1(0.0), 0.0);
        assert!((expm1(1.0) - (std::f64::consts::E - 1.0)).abs() < 1e-14);
        assert_eq!(expm1(f64::NEG_INFINITY), -1.0);
        assert!(expm1(f64::INFINITY).is_infinite());
        assert!(expm1(f64::NAN).is_nan());
    }
}
