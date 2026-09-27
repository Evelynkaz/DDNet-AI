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

//! `tanh`, ported line-for-line from `deps/v8/src/base/ieee754.cc:2986-3020`. Calls [`super::expm1::expm1`]
//! internally, exactly as V8 does — see `expm1.rs`.

use super::bits::get_high_word;
use super::expm1::expm1;

const TINY: f64 = 1.0e-300;
const HUGE: f64 = 1.0e300;

/// `tanh(x)`. `deps/v8/src/base/ieee754.cc:2986-3020`.
pub fn tanh(x: f64) -> f64 {
    let jx = get_high_word(x);
    let ix = jx & 0x7FFF_FFFF;

    // x is INF or NaN.
    if ix >= 0x7FF0_0000 {
        return if jx >= 0 {
            1.0 / x + 1.0 // tanh(+-inf) = +-1
        } else {
            1.0 / x - 1.0 // tanh(NaN) = NaN
        };
    }

    let z = if ix < 0x4036_0000 {
        // |x| < 22
        if ix < 0x3E30_0000 {
            // |x| < 2**-28
            if HUGE + x > 1.0 {
                return x; // tanh(tiny) = tiny, with inexact
            }
        }
        if ix >= 0x3FF0_0000 {
            // |x| >= 1
            let t = expm1(2.0 * x.abs());
            1.0 - 2.0 / (t + 2.0)
        } else {
            let t = expm1(-2.0 * x.abs());
            -t / (t + 2.0)
        }
    } else {
        // |x| >= 22, return +-1.
        1.0 - TINY // raise inexact flag
    };
    if jx >= 0 { z } else { -z }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_points() {
        assert_eq!(tanh(0.0), 0.0);
        assert!(tanh(-0.0).is_sign_negative());
        assert_eq!(tanh(f64::INFINITY), 1.0);
        assert_eq!(tanh(f64::NEG_INFINITY), -1.0);
        assert!(tanh(f64::NAN).is_nan());
        assert!((tanh(1.0) - 0.7615941559557649).abs() < 1e-15);
    }
}
