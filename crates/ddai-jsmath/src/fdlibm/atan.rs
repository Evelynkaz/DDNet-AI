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

//! `atan`/`atan2`, ported line-for-line from `deps/v8/src/base/ieee754.cc`: `atan` (lines
//! 1116-1200) and `atan2` (1228-1319). Both share the `atanhi`/`atanlo`/`aT` tables (V8 declares
//! them as `static const` locals inside `atan`; `atan2` calls `atan` for its actual work and does
//! not use the tables directly, but is kept in this file to mirror the source's grouping and
//! because it shares `atan`'s special-case constants in spirit).

use super::bits::{extract_words, get_high_word};

const ATAN_HI: [f64; 4] = [
    4.636_476_090_008_060_935_15e-1,
    7.853_981_633_974_482_789_99e-1,
    9.827_937_232_473_290_540_82e-1,
    1.570_796_326_794_896_558_00,
];

const ATAN_LO: [f64; 4] = [
    2.269_877_745_296_168_709_24e-17,
    3.061_616_997_868_383_017_93e-17,
    1.390_331_103_123_099_845_16e-17,
    6.123_233_995_736_766_035_87e-17,
];

#[rustfmt::skip]
const AT: [f64; 11] = [
    3.333_333_333_333_293_180_27e-1,
    -1.999_999_999_987_648_324_76e-1,
    1.428_571_427_250_346_637_11e-1,
    -1.111_111_040_546_235_578_80e-1,
    9.090_887_133_436_506_561_96e-2,
    -7.691_876_205_044_829_994_95e-2,
    6.661_073_137_387_531_206_69e-2,
    -5.833_570_133_790_573_486_45e-2,
    4.976_877_994_615_932_360_17e-2,
    -3.653_157_274_421_691_552_70e-2,
    1.628_582_011_536_578_236_23e-2,
];

const HUGE: f64 = 1.0e300;

/// `atan(x)`. `deps/v8/src/base/ieee754.cc:1116-1200`.
pub fn atan(mut x: f64) -> f64 {
    let hx = get_high_word(x);
    let ix = hx & 0x7FFF_FFFF;
    if ix >= 0x4410_0000 {
        // |x| >= 2^66
        let (_, low) = extract_words(x);
        if ix > 0x7FF0_0000 || (ix == 0x7FF0_0000 && low != 0) {
            return x + x; // NaN
        }
        return if hx > 0 {
            ATAN_HI[3] + ATAN_LO[3]
        } else {
            -ATAN_HI[3] - ATAN_LO[3]
        };
    }
    let id: i32;
    if ix < 0x3FDC_0000 {
        // |x| < 0.4375
        if ix < 0x3E40_0000 {
            // |x| < 2^-27
            if HUGE + x > 1.0 {
                return x; // raise inexact
            }
        }
        id = -1;
    } else {
        x = x.abs();
        if ix < 0x3FF3_0000 {
            // |x| < 1.1875
            if ix < 0x3FE6_0000 {
                // 7/16 <= |x| < 11/16
                id = 0;
                x = (2.0 * x - 1.0) / (2.0 + x);
            } else {
                // 11/16 <= |x| < 19/16
                id = 1;
                x = (x - 1.0) / (x + 1.0);
            }
        } else if ix < 0x4003_8000 {
            // |x| < 2.4375
            id = 2;
            x = (x - 1.5) / (1.0 + 1.5 * x);
        } else {
            // 2.4375 <= |x| < 2^66
            id = 3;
            x = -1.0 / x;
        }
    }
    // End of argument reduction.
    let z = x * x;
    let w = z * z;
    // Break the sum from i=0 to 10 of aT[i]*z**(i+1) into odd and even polynomials.
    let s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))));
    let s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))));
    if id < 0 {
        x - x * (s1 + s2)
    } else {
        let idx = id as usize;
        let z = ATAN_HI[idx] - ((x * (s1 + s2) - ATAN_LO[idx]) - x);
        if hx < 0 { -z } else { z }
    }
}

const PI_O_4: f64 = 7.853_981_633_974_482_790_0e-1;
const PI_O_2: f64 = 1.570_796_326_794_896_558_0;
const PI: f64 = 3.141_592_653_589_793_116_0;
const PI_LO: f64 = 1.224_646_799_147_353_177_2e-16;
const TINY: f64 = 1.0e-300;

/// `atan2(y, x)`. `deps/v8/src/base/ieee754.cc:1228-1319`.
///
/// The original C computes "is NaN"/"is zero" via integer bit tricks on the high/low words
/// (`(ix | ((lx | NegateWithWraparound(lx)) >> 31)) > 0x7FF00000`, etc. — `v | (-v)` has its sign
/// bit set iff `v != 0`, which is how it folds the low-word check into one comparison without
/// branching). Those tricks are pure boolean predicates with a single unambiguous answer (unlike
/// the *arithmetic* below, where operation order affects rounding), so this port uses the
/// equivalent, provably-identical `f64` predicates (`is_nan`/`== 0.0`/`is_infinite`) instead of
/// re-deriving the bit tricks — see the crate README "atan2: почему без битовых трюков для
/// NaN/0/Inf" for the full equivalence proof. Everything from `m` onward (the actual formula) is
/// ported as-is.
pub fn atan2(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return x + y; // x or y is NaN
    }
    if x == 1.0 {
        return atan(y);
    }
    let hx = get_high_word(x);
    let hy = get_high_word(y);
    let ix = hx & 0x7FFF_FFFF;
    let iy = hy & 0x7FFF_FFFF;
    let m = ((hy >> 31) & 1) | ((hx >> 30) & 2); // 2*sign(x) + sign(y)

    if y == 0.0 {
        return match m {
            0 | 1 => y, // atan(+-0, +anything) = +-0
            2 => PI + TINY,
            _ => -PI - TINY,
        };
    }
    if x == 0.0 {
        return if hy < 0 { -PI_O_2 - TINY } else { PI_O_2 + TINY };
    }
    if x.is_infinite() {
        if y.is_infinite() {
            return match m {
                0 => PI_O_4 + TINY,
                1 => -PI_O_4 - TINY,
                2 => 3.0 * PI_O_4 + TINY,
                _ => -3.0 * PI_O_4 - TINY,
            };
        } else {
            return match m {
                0 => 0.0,
                1 => -0.0,
                2 => PI + TINY,
                _ => -PI - TINY,
            };
        }
    }
    if y.is_infinite() {
        return if hy < 0 { -PI_O_2 - TINY } else { PI_O_2 + TINY };
    }

    // Compute y/x.
    let k = (iy - ix) >> 20;
    let z = if k > 60 {
        // |y/x| > 2^60
        PI_O_2 + 0.5 * PI_LO
    } else if hx < 0 && k < -60 {
        // 0 > |y|/x > -2^-60
        0.0
    } else {
        atan((y / x).abs()) // safe to do y/x
    };
    let m = if k > 60 { m & 1 } else { m };
    match m {
        0 => z,                // atan(+,+)
        1 => -z,               // atan(-,+)
        2 => PI - (z - PI_LO), // atan(+,-)
        _ => (z - PI_LO) - PI, // atan(-,-)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atan_matches_known_points() {
        assert_eq!(atan(0.0), 0.0);
        assert!(atan(-0.0).is_sign_negative());
        assert!((atan(1.0) - std::f64::consts::FRAC_PI_4).abs() < 1e-15);
    }

    #[test]
    fn atan2_special_cases() {
        assert_eq!(atan2(0.0, 1.0), 0.0);
        assert!(atan2(-0.0, 1.0).is_sign_negative());
        assert!(atan2(f64::NAN, 1.0).is_nan());
        assert!(atan2(1.0, f64::NAN).is_nan());
        assert!((atan2(1.0, 1.0) - std::f64::consts::FRAC_PI_4).abs() < 1e-15);
    }
}
