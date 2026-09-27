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

//! `exp`, ported line-for-line from `deps/v8/src/base/ieee754.cc:1447-1538`.

use super::bits::{get_high_word, get_low_word, insert_words};

const HALF: [f64; 2] = [0.5, -0.5];
const O_THRESHOLD: f64 = 7.097_827_128_933_839_730_96e2;
const U_THRESHOLD: f64 = -7.451_332_191_019_411_084_20e2;
const LN2_HI: [f64; 2] = [6.931_471_803_691_238_164_90e-1, -6.931_471_803_691_238_164_90e-1];
const LN2_LO: [f64; 2] = [1.908_214_929_270_587_700_02e-10, -1.908_214_929_270_587_700_02e-10];
const INVLN2: f64 = 1.442_695_040_888_963_387_00;
const P1: f64 = 1.666_666_666_666_660_190_37e-1;
const P2: f64 = -2.777_777_777_701_559_338_42e-3;
const P3: f64 = 6.613_756_321_437_934_361_17e-5;
const P4: f64 = -1.653_390_220_546_525_153_90e-6;
const P5: f64 = 4.138_136_797_057_238_460_39e-8;
const E: f64 = std::f64::consts::E;
const HUGE: f64 = 1.0e300;
const TWOM1000: f64 = 9.332_636_185_032_188_789_90e-302;
const TWO1023: f64 = 8.988_465_674_311_579_539e307;

/// `exp(x)`. `deps/v8/src/base/ieee754.cc:1447-1538`.
pub fn exp(x: f64) -> f64 {
    let hx0 = get_high_word(x);
    let xsb = ((hx0 >> 31) & 1) as usize; // sign bit of x
    let hx = (hx0 & 0x7FFF_FFFF) as u32; // high word of |x|

    // Filter out non-finite arguments.
    if hx >= 0x4086_2E42 {
        if hx >= 0x7FF0_0000 {
            let lx = get_low_word(x);
            if ((hx & 0xF_FFFF) | lx) != 0 {
                return x + x; // NaN
            } else {
                return if xsb == 0 { x } else { 0.0 }; // exp(+-inf) = {inf, 0}
            }
        }
        if x > O_THRESHOLD {
            return HUGE * HUGE; // overflow
        }
        if x < U_THRESHOLD {
            return TWOM1000 * TWOM1000; // underflow
        }
    }

    // Argument reduction.
    let mut k: i32 = 0;
    let mut hi = 0.0f64;
    let mut lo = 0.0f64;
    let mut x = x;
    if hx > 0x3FD6_2E42 {
        // |x| > 0.5 ln2
        if hx < 0x3FF0_A2B2 {
            // and |x| < 1.5 ln2
            if x == 1.0 {
                return E;
            }
            hi = x - LN2_HI[xsb];
            lo = LN2_LO[xsb];
            k = 1 - xsb as i32 - xsb as i32;
        } else {
            k = (INVLN2 * x + HALF[xsb]) as i32;
            let t = f64::from(k);
            hi = x - t * LN2_HI[0]; // t*ln2HI is exact here
            lo = t * LN2_LO[0];
        }
        x = hi - lo;
    } else if hx < 0x3E30_0000 {
        // |x| < 2**-28
        if HUGE + x > 1.0 {
            return 1.0 + x; // trigger inexact
        }
    } else {
        k = 0;
    }

    // x is now in the primary range.
    let t = x * x;
    // `INSERT_WORDS(twopk, 0x3FF00000 + static_cast<int32_t>(static_cast<uint32_t>(k) << 20), 0)`:
    // the original casts `k` (or `k+1000`) through `uint32_t` before shifting only because
    // left-shifting a *negative* signed integer is undefined behavior in C++ (pre-C++20) — the
    // resulting bit pattern is the same either way. Rust's `<<` on `i32` has no such UB (it is
    // always a well-defined two's-complement shift), so this ports directly, with
    // `wrapping_add` standing in for C++'s de-facto (if technically UB) wraparound `+`.
    let twopk = if k >= -1021 {
        insert_words(0x3FF0_0000i32.wrapping_add(k << 20), 0)
    } else {
        insert_words(0x3FF0_0000i32.wrapping_add((k + 1000) << 20), 0)
    };
    let c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if k == 0 {
        return 1.0 - ((x * c) / (c - 2.0) - x);
    }
    let y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
    if k >= -1021 {
        if k == 1024 {
            return y * 2.0 * TWO1023;
        }
        y * twopk
    } else {
        y * twopk * TWOM1000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_points() {
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(1.0), E);
        assert!(exp(f64::INFINITY).is_infinite());
        assert_eq!(exp(f64::NEG_INFINITY), 0.0);
        assert!(exp(f64::NAN).is_nan());
    }

    #[test]
    fn overflow_and_underflow() {
        assert!(exp(1000.0).is_infinite());
        assert_eq!(exp(-1000.0), 0.0);
    }
}
