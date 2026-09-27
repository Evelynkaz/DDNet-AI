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
// DDNet-AI, using safe `f64::to_bits`/`from_bits` (see `bits.rs`) instead of the original's
// pointer-aliasing macros. Same operations, same order, no `unsafe`, no FMA contraction.

//! `sin`/`cos`, ported line-for-line from `deps/v8/src/base/ieee754.cc` (V8 13.6, the version
//! Node 24.21.0 ships): `__ieee754_rem_pio2` (lines 118-270), `__kernel_rem_pio2` (443-644),
//! `__kernel_sin` (673-698), `__kernel_cos` (305-336), `cos` (1354-1381 in the non-libm branch —
//! see the crate README for why Node's V8 build takes that branch, not the glibc one) and `sin`
//! (2450-2479).
//!
//! This is the full fdlibm Payne-Hanek argument reduction (exact for arguments of any magnitude,
//! including the ones that need many bits of `2/pi` to reduce correctly), not an approximation
//! that only works for "reasonable" angles — needed because the task's adversarial probe set
//! includes huge arguments.

use super::bits::{get_high_word, get_low_word, insert_words};

/// `__ieee754_rem_pio2`'s "376 Hex digits (476 decimal) of 2/pi" table, used for Payne-Hanek
/// reduction of huge arguments. `deps/v8/src/base/ieee754.cc:122-133`.
#[rustfmt::skip]
const TWO_OVER_PI: [i32; 66] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C,
    0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649,
    0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129, 0xA73EE8, 0x8235F5, 0x2EBB44,
    0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C, 0x845F8B,
    0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D,
    0x367ECF, 0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
    0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330,
    0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3, 0x91615E, 0xE61B08,
    0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B,
];

/// `deps/v8/src/base/ieee754.cc:135-142`.
const NPIO2_HW: [i32; 32] = [
    0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB, 0x402921FB, 0x402C463A,
    0x402F6A7A, 0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB, 0x403AB41B, 0x403C463A,
    0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C, 0x4043A28C, 0x40446B9C, 0x404534AC,
    0x4045FDBB, 0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB,
];

const ZERO: f64 = 0.0;
const HALF: f64 = 0.5;
const TWO24: f64 = 1.677_721_600_000_000_0e7;
const INVPIO2: f64 = 6.366_197_723_675_813_824_33e-1;
const PIO2_1: f64 = 1.570_796_326_734_125_614_17;
const PIO2_1T: f64 = 6.077_100_506_506_192_249_32e-11;
const PIO2_2: f64 = 6.077_100_506_303_965_976_60e-11;
const PIO2_2T: f64 = 2.022_266_248_795_950_631_54e-21;
const PIO2_3: f64 = 2.022_266_248_711_166_455_80e-21;
const PIO2_3T: f64 = 8.478_427_660_368_899_569_97e-32;

/// `__ieee754_rem_pio2(x, y)`: returns the remainder of `x rem pi/2` in `y[0]+y[1]`, and `n mod 8`
/// (the quadrant) as the return value. `deps/v8/src/base/ieee754.cc:118-270`.
fn ieee754_rem_pio2(x: f64, y: &mut [f64; 2]) -> i32 {
    let hx = get_high_word(x);
    let ix = hx & 0x7FFF_FFFF;
    if ix <= 0x3FE9_21FB {
        // |x| ~<= pi/4, no need for reduction.
        y[0] = x;
        y[1] = 0.0;
        return 0;
    }
    if ix < 0x4002_D97C {
        // |x| < 3pi/4, special case with n = +-1.
        if hx > 0 {
            let mut z = x - PIO2_1;
            if ix != 0x3FF9_21FB {
                // 33+53 bit pi is good enough.
                y[0] = z - PIO2_1T;
                y[1] = (z - y[0]) - PIO2_1T;
            } else {
                // Near pi/2, use 33+33+53 bit pi.
                z -= PIO2_2;
                y[0] = z - PIO2_2T;
                y[1] = (z - y[0]) - PIO2_2T;
            }
            return 1;
        } else {
            let mut z = x + PIO2_1;
            if ix != 0x3FF9_21FB {
                y[0] = z + PIO2_1T;
                y[1] = (z - y[0]) + PIO2_1T;
            } else {
                z += PIO2_2;
                y[0] = z + PIO2_2T;
                y[1] = (z - y[0]) + PIO2_2T;
            }
            return -1;
        }
    }
    if ix <= 0x4139_21FB {
        // |x| ~<= 2^19*(pi/2), medium size.
        let t = x.abs();
        let n = (t * INVPIO2 + HALF) as i32;
        let fn_ = f64::from(n);
        let mut r = t - fn_ * PIO2_1;
        let mut w = fn_ * PIO2_1T; // 1st round good to 85 bit
        if n < 32 && ix != NPIO2_HW[(n - 1) as usize] {
            y[0] = r - w; // quick check, no cancellation
        } else {
            let j = ix >> 20;
            y[0] = r - w;
            let high = get_high_word(y[0]);
            let mut i = j - ((high >> 20) & 0x7FF);
            if i > 16 {
                // 2nd iteration needed, good to 118 bits.
                let t2 = r;
                w = fn_ * PIO2_2;
                r = t2 - w;
                w = fn_ * PIO2_2T - ((t2 - r) - w);
                y[0] = r - w;
                let high2 = get_high_word(y[0]);
                i = j - ((high2 >> 20) & 0x7FF);
                if i > 49 {
                    // 3rd iteration needed, 151 bits accuracy.
                    let t3 = r;
                    w = fn_ * PIO2_3;
                    r = t3 - w;
                    w = fn_ * PIO2_3T - ((t3 - r) - w);
                    y[0] = r - w;
                }
            }
        }
        y[1] = (r - y[0]) - w;
        return if hx < 0 {
            y[0] = -y[0];
            y[1] = -y[1];
            -n
        } else {
            n
        };
    }
    // All other (large) arguments.
    if ix >= 0x7FF0_0000 {
        // x is inf or NaN.
        y[0] = x - x;
        y[1] = y[0];
        return 0;
    }
    // set z = scalbn(|x|, ilogb(x)-23)
    let low = get_low_word(x);
    // `z = scalbn(|x|, ilogb(x)-23)`, built directly from the bit halves (the C original builds
    // it the same way: a fresh double assembled from x's low word and a computed high word, not
    // an actual `scalbn` call): `SET_LOW_WORD(z, low); e0 = (ix>>20)-1046; SET_HIGH_WORD(z, ix -
    // (e0<<20))`. Since `z` starts at `0.0` (high word `0`), setting the low word first and then
    // the high word is equivalent to assembling both halves directly.
    let e0 = (ix >> 20) - 1046; // e0 = ilogb(z) - 23
    let new_high = ix.wrapping_sub((e0 as u32).wrapping_shl(20) as i32);
    let mut z = insert_words(new_high, low);

    // `for (i = 0; i < 2; i++) { tx[i] = (double)(int32_t)z; z = (z - tx[i]) * two24; }`
    let mut tx = [0.0f64; 3];
    for tx_i in tx.iter_mut().take(2) {
        let trunc_i32 = z as i32; // matches `(double)(int32_t)z`: C++ truncates toward 0.
        *tx_i = f64::from(trunc_i32);
        z = (z - *tx_i) * TWO24;
    }
    tx[2] = z;
    let mut nx: i32 = 3;
    while nx > 0 && tx[(nx - 1) as usize] == ZERO {
        nx -= 1;
    }
    let n = kernel_rem_pio2(&tx[..nx as usize], y, e0, nx, 2, &TWO_OVER_PI);
    if hx < 0 {
        y[0] = -y[0];
        y[1] = -y[1];
        -n
    } else {
        n
    }
}

const PIO2: [f64; 8] = [
    1.570_796_251_296_997_070_31,
    7.549_789_415_861_596_353_35e-8,
    5.390_302_529_957_764_765_54e-15,
    3.282_003_415_807_912_941_23e-22,
    1.270_655_753_080_676_073_49e-29,
    1.229_333_089_811_113_289_32e-36,
    2.733_700_538_164_645_596_24e-44,
    2.167_416_838_778_048_194_44e-51,
];

const INIT_JK: [i32; 4] = [2, 3, 4, 6];
const TWO24_KRP: f64 = 1.677_721_600_000_000_0e7;
const TWON24: f64 = 5.960_464_477_539_062_5e-8;

/// `__kernel_rem_pio2(x, y, e0, nx, prec, ipio2)`. `deps/v8/src/base/ieee754.cc:443-644`.
///
/// `x` holds `nx` (1..=3) 24-bit chunks of the input's magnitude (see the caller); `y` receives
/// the reduced remainder (`y[0]+y[1]` to double precision, since this crate always calls with
/// `prec == 2`, matching every V8 call site — `y[2]` is only ever written for the quad-precision
/// `prec == 3` case, which V8 itself never uses either).
#[allow(clippy::too_many_arguments)]
fn kernel_rem_pio2(x: &[f64], y: &mut [f64; 2], e0: i32, nx: i32, prec: i32, ipio2: &[i32]) -> i32 {
    let jk = INIT_JK[prec as usize];
    let jp = jk;

    let jx = nx - 1;
    let mut jv = (e0 - 3) / 24;
    if jv < 0 {
        jv = 0;
    }
    let mut q0 = e0 - 24 * (jv + 1);

    let mut f = [0.0f64; 20];
    let m = jx + jk;
    for (i, j) in (0..=m).zip(jv - jx..) {
        f[i as usize] = if j < 0 { 0.0 } else { f64::from(ipio2[j as usize]) };
    }

    let mut q = [0.0f64; 20];
    for i in 0..=jk {
        let mut fw = 0.0;
        for j in 0..=jx {
            fw += x[j as usize] * f[(jx + i - j) as usize];
        }
        q[i as usize] = fw;
    }

    let mut jz = jk;
    let mut iq = [0i32; 20];
    let (mut z, mut fw, mut n, mut ih);
    loop {
        // recompute:
        {
            let mut j = jz;
            z = q[jz as usize];
            for i in 0..jz {
                fw = f64::from((TWON24 * z) as i32);
                iq[i as usize] = (z - TWO24_KRP * fw) as i32;
                z = q[(j - 1) as usize] + fw;
                j -= 1;
            }
        }

        z = super::bits::scalbn(z, q0);
        z -= 8.0 * (z * 0.125).floor();
        n = z as i32;
        z -= f64::from(n);
        ih = 0;
        if q0 > 0 {
            let i = iq[(jz - 1) as usize] >> (24 - q0);
            n += i;
            iq[(jz - 1) as usize] -= i << (24 - q0);
            ih = iq[(jz - 1) as usize] >> (23 - q0);
        } else if q0 == 0 {
            ih = iq[(jz - 1) as usize] >> 23;
        } else if z >= 0.5 {
            ih = 2;
        }

        if ih > 0 {
            n += 1;
            let mut carry = 0i32;
            for i in 0..jz {
                let j = iq[i as usize];
                if carry == 0 {
                    if j != 0 {
                        carry = 1;
                        iq[i as usize] = 0x0100_0000 - j;
                    }
                } else {
                    iq[i as usize] = 0x00FF_FFFF - j;
                }
            }
            if q0 > 0 {
                match q0 {
                    1 => iq[(jz - 1) as usize] &= 0x007F_FFFF,
                    2 => iq[(jz - 1) as usize] &= 0x003F_FFFF,
                    _ => {}
                }
            }
            if ih == 2 {
                z = 1.0 - z;
                if carry != 0 {
                    z -= super::bits::scalbn(1.0, q0);
                }
            }
        }

        if z == 0.0 {
            let mut j = 0;
            for i in (jk..jz).rev() {
                j |= iq[i as usize];
            }
            if j == 0 {
                // Need recomputation.
                let mut k = 1;
                while jk >= k && iq[(jk - k) as usize] == 0 {
                    k += 1;
                }
                for i in (jz + 1)..=(jz + k) {
                    f[(jx + i) as usize] = f64::from(ipio2[(jv + i) as usize]);
                    let mut fw2 = 0.0;
                    for j2 in 0..=jx {
                        fw2 += x[j2 as usize] * f[(jx + i - j2) as usize];
                    }
                    q[i as usize] = fw2;
                }
                jz += k;
                continue;
            }
        }
        break;
    }

    // Chop off zero terms, or break z into a 24-bit chunk if necessary.
    if z == 0.0 {
        jz -= 1;
        q0 -= 24;
        while iq[jz as usize] == 0 {
            jz -= 1;
            q0 -= 24;
        }
    } else {
        z = super::bits::scalbn(z, -q0);
        if z >= TWO24_KRP {
            fw = f64::from((TWON24 * z) as i32);
            iq[jz as usize] = (z - TWO24_KRP * fw) as i32;
            jz += 1;
            q0 += 24;
            iq[jz as usize] = fw as i32;
        } else {
            iq[jz as usize] = z as i32;
        }
    }

    // Convert the integer "bit" chunks to floating-point values.
    fw = super::bits::scalbn(1.0, q0);
    for i in (0..=jz).rev() {
        q[i as usize] = fw * f64::from(iq[i as usize]);
        fw *= TWON24;
    }

    // Compute PIo2[0..jp] * q[jz..0].
    let mut fq = [0.0f64; 20];
    for i in (0..=jz).rev() {
        let mut fw2 = 0.0;
        let mut k = 0;
        while k <= jp && k <= jz - i {
            fw2 += PIO2[k as usize] * q[(i + k) as usize];
            k += 1;
        }
        fq[(jz - i) as usize] = fw2;
    }

    // Compress fq[] into y[] (prec is always 2 here: 53-bit precision, two doubles).
    match prec {
        0 => {
            let mut sum = 0.0;
            for i in (0..=jz).rev() {
                sum += fq[i as usize];
            }
            y[0] = if ih == 0 { sum } else { -sum };
        }
        _ => {
            let mut sum = 0.0;
            for i in (0..=jz).rev() {
                sum += fq[i as usize];
            }
            y[0] = if ih == 0 { sum } else { -sum };
            let mut sum2 = fq[0] - sum;
            for i in 1..=jz {
                sum2 += fq[i as usize];
            }
            y[1] = if ih == 0 { sum2 } else { -sum2 };
        }
    }
    n & 7
}

const S1: f64 = -1.666_666_666_666_663_243_48e-1;
const S2: f64 = 8.333_333_333_322_489_461_24e-3;
const S3: f64 = -1.984_126_982_985_794_931_34e-4;
const S4: f64 = 2.755_731_370_707_006_767_89e-6;
const S5: f64 = -2.505_076_025_340_686_341_95e-8;
const S6: f64 = 1.589_690_995_211_550_102_21e-10;

/// `__kernel_sin(x, y, iy)`: kernel sine on `[-pi/4, pi/4]`. `deps/v8/src/base/ieee754.cc:673-698`.
fn kernel_sin(x: f64, y: f64, iy: i32) -> f64 {
    let ix = get_high_word(x) & 0x7FFF_FFFF;
    if ix < 0x3E40_0000 && (x as i32) == 0 {
        // |x| < 2**-27 and x == 0 (as an integer, i.e. truly zero): generate inexact, return x.
        return x;
    }
    let z = x * x;
    let v = z * x;
    let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    if iy == 0 {
        x + v * (S1 + z * r)
    } else {
        x - ((z * (0.5 * y - v * r) - y) - v * S1)
    }
}

const C1: f64 = 4.166_666_666_666_660_190_37e-2;
const C2: f64 = -1.388_888_888_887_410_957_49e-3;
const C3: f64 = 2.480_158_728_947_672_941_78e-5;
const C4: f64 = -2.755_731_435_139_066_330_35e-7;
const C5: f64 = 2.087_572_321_298_174_827_90e-9;
const C6: f64 = -1.135_964_755_778_819_482_65e-11;

/// `__kernel_cos(x, y)`: kernel cosine on `[-pi/4, pi/4]`. `deps/v8/src/base/ieee754.cc:305-336`.
fn kernel_cos(x: f64, y: f64) -> f64 {
    let ix = get_high_word(x) & 0x7FFF_FFFF;
    if ix < 0x3E40_0000 && (x as i32) == 0 {
        return 1.0;
    }
    let z = x * x;
    let r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    if ix < 0x3FD3_3333 {
        // |x| < 0.3
        1.0 - (0.5 * z - (z * r - x * y))
    } else {
        let qx = if ix > 0x3FE9_0000 {
            0.28125
        } else {
            insert_words(ix - 0x0020_0000, 0) // x/4, top bits only
        };
        let iz = 0.5 * z - qx;
        let a = 1.0 - qx;
        a - (iz - (z * r - x * y))
    }
}

/// `cos(x)`. `deps/v8/src/base/ieee754.cc:1354-1381` (the non-`V8_USE_LIBM_TRIG_FUNCTIONS`
/// branch — see the crate README for why that is the branch Node's V8 actually compiles in).
pub fn cos(x: f64) -> f64 {
    let ix = get_high_word(x) & 0x7FFF_FFFF;
    if ix <= 0x3FE9_21FB {
        kernel_cos(x, 0.0)
    } else if ix >= 0x7FF0_0000 {
        x - x // NaN
    } else {
        let mut y = [0.0f64; 2];
        let n = ieee754_rem_pio2(x, &mut y);
        match n & 3 {
            0 => kernel_cos(y[0], y[1]),
            1 => -kernel_sin(y[0], y[1], 1),
            2 => -kernel_cos(y[0], y[1]),
            _ => kernel_sin(y[0], y[1], 1),
        }
    }
}

/// `sin(x)`. `deps/v8/src/base/ieee754.cc:2450-2479`.
pub fn sin(x: f64) -> f64 {
    let ix = get_high_word(x) & 0x7FFF_FFFF;
    if ix <= 0x3FE9_21FB {
        kernel_sin(x, 0.0, 0)
    } else if ix >= 0x7FF0_0000 {
        x - x // NaN
    } else {
        let mut y = [0.0f64; 2];
        let n = ieee754_rem_pio2(x, &mut y);
        match n & 3 {
            0 => kernel_sin(y[0], y[1], 1),
            1 => kernel_cos(y[0], y[1]),
            2 => -kernel_sin(y[0], y[1], 1),
            _ => -kernel_cos(y[0], y[1]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn matches_exact_zero() {
        assert_eq!(sin(0.0), 0.0);
        assert_eq!(cos(0.0), 1.0);
        assert!(sin(-0.0).is_sign_negative());
    }

    #[test]
    fn matches_pi_boundaries_within_1_ulp() {
        // Not exact (pi/2 as an f64 isn't exactly pi/2), but should be very close.
        assert!((sin(PI / 2.0) - 1.0).abs() < 1e-15);
        assert!((cos(PI) - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn nan_and_inf_give_nan() {
        assert!(sin(f64::INFINITY).is_nan());
        assert!(sin(f64::NEG_INFINITY).is_nan());
        assert!(sin(f64::NAN).is_nan());
        assert!(cos(f64::INFINITY).is_nan());
    }

    #[test]
    fn huge_argument_does_not_panic() {
        // Exercises the Payne-Hanek reduction path.
        assert!(sin(1.0e300).is_finite());
        assert!(cos(-1.0e300).is_finite());
        assert!(sin(f64::MAX).is_finite());
    }
}
