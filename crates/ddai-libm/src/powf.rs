// Single-precision pow, ported from the ARM optimized-routines `math/powf.c`, `math/powf_log2_data.c`,
// `math/exp2f_data.c` (the algorithm glibc 2.27+ ships as `powf`).
//
// Copyright (c) 2017-2018, Arm Limited.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
// following conditions: The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Altered: rewritten from C to Rust for DDNet-AI (safe bit operations instead of unions), the build
// configuration of glibc 2.39 on x86-64 fixed (`TOINT_INTRINSICS == 0`, `WANT_ERRNO`, round-to-nearest),
// and every `a + b * c` that GCC fuses when compiling glibc's `e_powf-fma.c` with `-mfma -mavx2` written
// as an explicit `fma(b, c, a)`, so the result is bit-identical to glibc 2.39's x86-64 FMA variant.

//! `powf`, bit-identical to glibc 2.39's `__powf_fma`.

use crate::tables::{EXP2F_POLY, EXP2F_SHIFT_SCALED, EXP2F_TAB, POWF_LOG2_POLY, POWF_LOG2_TAB};
use crate::util::{divzero_f32, fma, invalid_f32, may_uflow_f32, nan_sum_f32, oflow_f32, uflow_f32};

const POWF_LOG2_TABLE_BITS: u32 = 4;
const EXP2F_TABLE_BITS: u32 = 5;
const OFF: u32 = 0x3f33_0000;
const SIGN_BIAS: u32 = 1 << (EXP2F_TABLE_BITS + 11);

/// `asuint64 (126.0) >> 47`.
const LIM_126_TOP: u64 = 0x405f_8000_0000_0000 >> 47;
/// `0x1.fffffffd1d571p+6`: above this `|x^y| > 0x1.ffffffp127`, which overflows.
const OFLOW_LIMIT: f64 = f64::from_bits(0x405f_ffff_ffd1_d571);
/// `-150.0`.
const UFLOW_LIMIT: f64 = -150.0;
/// `-149.0`.
const MAY_UFLOW_LIMIT: f64 = -149.0;

#[inline(always)]
fn tab(i: usize) -> (f64, f64) {
    (
        f64::from_bits(POWF_LOG2_TAB[2 * i]),
        f64::from_bits(POWF_LOG2_TAB[2 * i + 1]),
    )
}

/// `log2(x)` for the (normalised) bit pattern `ix`, as a double.
#[inline(always)]
fn log2_inline(ix: u32) -> f64 {
    // x = 2^k z; where z is in range [OFF,2*OFF] and exact.
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (23 - POWF_LOG2_TABLE_BITS)) % (1 << POWF_LOG2_TABLE_BITS)) as usize;
    let top = tmp & 0xff80_0000;
    let iz = ix.wrapping_sub(top);
    let k = (top as i32) >> 23; // arithmetic shift
    let (invc, logc) = tab(i);
    let z = f64::from(f32::from_bits(iz));
    let a = |j: usize| f64::from_bits(POWF_LOG2_POLY[j]);

    // log2(x) = log1p(z/c-1)/ln2 + log2(c) + k
    let r = fma(z, invc, -1.0);
    let y0 = logc + f64::from(k);

    // Pipelined polynomial evaluation to approximate log1p(r)/ln2.
    let r2 = r * r;
    let y = fma(a(0), r, a(1));
    let p = fma(a(2), r, a(3));
    let r4 = r2 * r2;
    let q = fma(a(4), r, y0);
    let q = fma(p, r2, q);
    fma(y, r4, q)
}

/// `2^xd` with the sign bit of the result set by `sign_bias`; `xd` must be in [-1021, 1023].
#[inline(always)]
fn exp2_inline(xd: f64, sign_bias: u32) -> f64 {
    let shift = f64::from_bits(EXP2F_SHIFT_SCALED);
    let c = |j: usize| f64::from_bits(EXP2F_POLY[j]);
    // x = k/N + r with r in [-1/(2N), 1/(2N)]
    let kd = xd + shift; // Rounding to double precision is required.
    let ki = kd.to_bits();
    let kd = kd - shift; // k/N
    let r = xd - kd;

    // exp2(x) = 2^(k/N) * 2^r ~= s * (C0*r^3 + C1*r^2 + C2*r + 1)
    let mut t = EXP2F_TAB[(ki % (1 << EXP2F_TABLE_BITS)) as usize];
    let ski = ki.wrapping_add(u64::from(sign_bias));
    t = t.wrapping_add(ski << (52 - EXP2F_TABLE_BITS));
    let s = f64::from_bits(t);
    let z = fma(c(0), r, c(1));
    let r2 = r * r;
    let y = fma(c(2), r, 1.0);
    let y = fma(z, r2, y);
    y * s
}

/// Returns 0 if not int, 1 if odd int, 2 if even int. The argument is the bit representation of a
/// non-zero finite floating-point value.
#[inline(always)]
fn checkint(iy: u32) -> u32 {
    let e = (iy >> 23) & 0xff;
    if e < 0x7f {
        return 0;
    }
    if e > 0x7f + 23 {
        return 2;
    }
    if iy & ((1 << (0x7f + 23 - e)) - 1) != 0 {
        return 0;
    }
    if iy & (1 << (0x7f + 23 - e)) != 0 {
        return 1;
    }
    2
}

/// True for the bit representation of 0, infinity or NaN.
#[inline(always)]
fn zeroinfnan(ix: u32) -> bool {
    ix.wrapping_mul(2).wrapping_sub(1) >= 0xff00_0000u32.wrapping_sub(1)
}

/// glibc's `issignaling (float)` on x86 (the quiet bit is set for a quiet NaN).
#[inline(always)]
fn issignaling(x: f32) -> bool {
    (x.to_bits() ^ 0x0040_0000).wrapping_mul(2) > 0xff80_0000
}

/// `powf(x, y)`, bit-identical to glibc 2.39 (x86-64, FMA variant).
#[must_use]
pub fn powf(x: f32, y: f32) -> f32 {
    let mut sign_bias = 0u32;
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 || zeroinfnan(iy) {
        // Either (x < 0x1p-126 or inf or nan) or (y is 0 or inf or nan).
        if zeroinfnan(iy) {
            if iy.wrapping_mul(2) == 0 {
                return if issignaling(x) { nan_sum_f32(x, y) } else { 1.0 };
            }
            if ix == 0x3f80_0000 {
                return if issignaling(y) { nan_sum_f32(x, y) } else { 1.0 };
            }
            if ix.wrapping_mul(2) > 0xff00_0000u32 || iy.wrapping_mul(2) > 0xff00_0000u32 {
                return nan_sum_f32(x, y);
            }
            if ix.wrapping_mul(2) == 0x7f00_0000u32 {
                return 1.0;
            }
            if (ix.wrapping_mul(2) < 0x7f00_0000u32) == (iy & 0x8000_0000 == 0) {
                return 0.0; // |x|<1 && y==inf or |x|>1 && y==-inf.
            }
            return y * y;
        }
        if zeroinfnan(ix) {
            let mut x2 = x * x;
            if ix & 0x8000_0000 != 0 && checkint(iy) == 1 {
                x2 = -x2;
                sign_bias = 1;
            }
            if ix.wrapping_mul(2) == 0 && iy & 0x8000_0000 != 0 {
                return divzero_f32(sign_bias != 0);
            }
            return if iy & 0x8000_0000 != 0 { 1.0 / x2 } else { x2 };
        }
        // x and y are non-zero finite.
        if ix & 0x8000_0000 != 0 {
            // Finite x < 0.
            let yint = checkint(iy);
            if yint == 0 {
                return invalid_f32(x);
            }
            if yint == 1 {
                sign_bias = SIGN_BIAS;
            }
            ix &= 0x7fff_ffff;
        }
        if ix < 0x0080_0000 {
            // Normalize subnormal x so exponent becomes negative.
            ix = (x * f32::from_bits(0x4b00_0000)).to_bits(); // x * 0x1p23f
            ix &= 0x7fff_ffff;
            ix = ix.wrapping_sub(23 << 23);
        }
    }
    let logx = log2_inline(ix);
    let ylogx = f64::from(y) * logx; // Note: cannot overflow, y is single prec.
    if ((ylogx.to_bits() >> 47) & 0xffff) >= LIM_126_TOP {
        // |y*log(x)| >= 126.
        if ylogx > OFLOW_LIMIT {
            // |x^y| > 0x1.ffffffp127.
            return oflow_f32(sign_bias != 0);
        }
        // `ylogx > 0x1.fffffffa3aae2p+6`: |x^y| > 0x1.fffffep127, glibc checks whether the result rounds
        // away from 0 in the current rounding mode; in round-to-nearest `1.0f + 0x1p-25f == 1.0f`, so it
        // never does.
        if ylogx <= UFLOW_LIMIT {
            return uflow_f32(sign_bias != 0);
        }
        if ylogx < MAY_UFLOW_LIMIT {
            return may_uflow_f32(sign_bias != 0);
        }
    }
    exp2_inline(ylogx, sign_bias) as f32
}
