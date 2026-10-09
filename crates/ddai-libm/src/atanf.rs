// Single-precision arctangent, ported from fdlibm's `s_atanf.c` as shipped in glibc 2.39
// (`sysdeps/ieee754/flt-32/s_atanf.c`).
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
// Altered: rewritten from C to Rust for DDNet-AI (safe bit operations instead of macros on unions),
// otherwise line-for-line with the same operation order. glibc builds this file without FMA (it has no
// `ifunc` variant), so there is no fused operation here, and the Rust `f32` operations round exactly as
// the C `float` operations do.

//! `atanf`, bit-identical to glibc 2.39's `__atanf`.

// All constants are given as bit patterns of the `float` values the C compiler makes of the decimal
// literals in the source (shown in the trailing comments).
const ATANHI: [f32; 4] = [
    f32::from_bits(0x3eed6338), // 4.6364760399e-1
    f32::from_bits(0x3f490fda), // 7.8539812565e-1
    f32::from_bits(0x3f7b985e), // 9.8279368877e-1
    f32::from_bits(0x3fc90fda), // 1.5707962513
];

const ATANLO: [f32; 4] = [
    f32::from_bits(0x31ac3769), // 5.0121582440e-9
    f32::from_bits(0x33222168), // 3.7748947079e-8
    f32::from_bits(0x33140fb4), // 3.4473217170e-8
    f32::from_bits(0x33a22168), // 7.5497894159e-8
];

// Note: the C source annotates aT[0] with the bit pattern 0x3eaaaaaa, but the decimal literal next to it
// (3.3333334327e-01) is what the compiler uses, and it rounds to 0x3eaaaaab.
const A_T: [f32; 11] = [
    f32::from_bits(0x3eaaaaab), // 3.3333334327e-1
    f32::from_bits(0xbe4ccccd), // -2.0000000298e-1
    f32::from_bits(0x3e124925), // 1.4285714924e-1
    f32::from_bits(0xbde38e38), // -1.1111110449e-1
    f32::from_bits(0x3dba2e6e), // 9.0908870101e-2
    f32::from_bits(0xbd9d8795), // -7.6918758452e-2
    f32::from_bits(0x3d886b35), // 6.6610731184e-2
    f32::from_bits(0xbd6ef16b), // -5.8335702866e-2
    f32::from_bits(0x3d4bda59), // 4.9768779427e-2
    f32::from_bits(0xbd15a221), // -3.6531571299e-2
    f32::from_bits(0x3c8569d7), // 1.6285819933e-2
];

/// `atanf(x)`, bit-identical to glibc 2.39.
#[must_use]
pub fn atanf(x: f32) -> f32 {
    let hx = x.to_bits() as i32;
    let ix = hx & 0x7fff_ffff;
    let id: i32;
    let mut x = x;
    if ix >= 0x4c00_0000 {
        // |x| >= 2^25
        if ix > 0x7f80_0000 {
            return x + x; // NaN
        }
        if hx > 0 {
            return ATANHI[3] + ATANLO[3];
        }
        return -ATANHI[3] - ATANLO[3];
    }
    if ix < 0x3ee0_0000 {
        // |x| < 0.4375
        if ix < 0x3100_0000 {
            // |x| < 2^-29: atan(x) == x (glibc also raises inexact/underflow here)
            return x;
        }
        id = -1;
    } else {
        x = x.abs();
        if ix < 0x3f98_0000 {
            // |x| < 1.1875
            if ix < 0x3f30_0000 {
                // 7/16 <= |x| < 11/16
                id = 0;
                x = (2.0 * x - 1.0) / (2.0 + x);
            } else {
                // 11/16 <= |x| < 19/16
                id = 1;
                x = (x - 1.0) / (x + 1.0);
            }
        } else if ix < 0x401c_0000 {
            // |x| < 2.4375
            id = 2;
            x = (x - 1.5) / (1.0 + 1.5 * x);
        } else {
            // 2.4375 <= |x| < 2^66
            id = 3;
            x = -1.0 / x;
        }
    }
    // end of argument reduction
    let z = x * x;
    let w = z * z;
    // break sum from i=0 to 10 aT[i]z**(i+1) into odd and even poly
    let s1 = z * (A_T[0] + w * (A_T[2] + w * (A_T[4] + w * (A_T[6] + w * (A_T[8] + w * A_T[10])))));
    let s2 = w * (A_T[1] + w * (A_T[3] + w * (A_T[5] + w * (A_T[7] + w * A_T[9]))));
    if id < 0 {
        x - x * (s1 + s2)
    } else {
        let id = id as usize;
        let z = ATANHI[id] - ((x * (s1 + s2) - ATANLO[id]) - x);
        if hx < 0 { -z } else { z }
    }
}
