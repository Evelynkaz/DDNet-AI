// Single-precision two-argument arctangent, ported from fdlibm's `e_atan2f.c` as shipped in glibc 2.39
// (`sysdeps/ieee754/flt-32/e_atan2f.c`).
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
// Altered on 2026-10-09: rewritten from C to Rust for DDNet-AI, otherwise line-for-line with the same operation order.
// glibc builds this file without FMA (no `ifunc` variant), so there is no fused operation here.

//! `atan2f`, bit-identical to glibc 2.39's `__atan2f`.

use crate::atanf::atanf;
use crate::util::nan_sum_f32;

const TINY: f32 = 1.0e-30;
const ZERO: f32 = f32::from_bits(0x00000000); // 0.0
const PI_O_4: f32 = f32::from_bits(0x3f490fdb); // 7.853_981_852_5e-1
const PI_O_2: f32 = f32::from_bits(0x3fc90fdb); // 1.570_796_370_5
const PI: f32 = f32::from_bits(0x40490fdb); // 3.141_592_741_0
const PI_LO: f32 = f32::from_bits(0xb3bbbd2e); // -8.742_277_657_3e-8

/// `atan2f(y, x)`, bit-identical to glibc 2.39.
#[must_use]
pub fn atan2f(y: f32, x: f32) -> f32 {
    let hx = x.to_bits() as i32;
    let ix = hx & 0x7fff_ffff;
    let hy = y.to_bits() as i32;
    let iy = hy & 0x7fff_ffff;
    if ix > 0x7f80_0000 || iy > 0x7f80_0000 {
        // x or y is NaN
        return nan_sum_f32(x, y);
    }
    if hx == 0x3f80_0000 {
        return atanf(y); // x=1.0
    }
    let m = ((hy >> 31) & 1) | ((hx >> 30) & 2); // 2*sign(x)+sign(y)

    // when y = 0
    if iy == 0 {
        return match m {
            0 | 1 => y,      // atan(+-0,+anything)=+-0
            2 => PI + TINY,  // atan(+0,-anything) = pi
            _ => -PI - TINY, // atan(-0,-anything) =-pi
        };
    }
    // when x = 0
    if ix == 0 {
        return if hy < 0 { -PI_O_2 - TINY } else { PI_O_2 + TINY };
    }
    // when x is INF
    if ix == 0x7f80_0000 {
        if iy == 0x7f80_0000 {
            return match m {
                0 => PI_O_4 + TINY,        // atan(+INF,+INF)
                1 => -PI_O_4 - TINY,       // atan(-INF,+INF)
                2 => 3.0 * PI_O_4 + TINY,  // atan(+INF,-INF)
                _ => -3.0 * PI_O_4 - TINY, // atan(-INF,-INF)
            };
        }
        return match m {
            0 => ZERO,       // atan(+...,+INF)
            1 => -ZERO,      // atan(-...,+INF)
            2 => PI + TINY,  // atan(+...,-INF)
            _ => -PI - TINY, // atan(-...,-INF)
        };
    }
    // when y is INF
    if iy == 0x7f80_0000 {
        return if hy < 0 { -PI_O_2 - TINY } else { PI_O_2 + TINY };
    }

    // compute y/x
    let k = (iy - ix) >> 23;
    let z = if k > 60 {
        PI_O_2 + 0.5 * PI_LO // |y/x| >  2**60
    } else if hx < 0 && k < -60 {
        0.0 // |y|/x < -2**60
    } else {
        atanf((y / x).abs()) // safe to do y/x
    };
    match m {
        0 => z,                                         // atan(+,+)
        1 => f32::from_bits(z.to_bits() ^ 0x8000_0000), // atan(-,+)
        2 => PI - (z - PI_LO),                          // atan(+,-)
        _ => (z - PI_LO) - PI,                          // atan(-,-)
    }
}
