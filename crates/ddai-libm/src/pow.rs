// Double-precision x^y, ported from glibc 2.39: `sysdeps/ieee754/dbl-64/e_pow.c`, `e_pow_log_data.c` and
// `e_exp_data.c` (function `__pow`).
//
// Copyright (C) 2018-2024 Free Software Foundation, Inc.
// This file is part of the GNU C Library.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, see <https://www.gnu.org/licenses/>.
//
// Licence note (see NOTICE): the C original carries the GNU Lesser General Public License, version 2.1 or
// (at your option) any later version. Section 3 of that licence lets a recipient apply the terms of the
// ordinary GNU General Public License (a newer version than 2 may be named) to a copy instead, on the one
// condition that every notice that refers to the LGPL is altered to refer to the GPL and nothing else in
// the notices is changed. That is what this file does: the copyright lines above are the original ones, the
// permission paragraphs are the GPL's, and DDNet-AI as a whole is GPL-3.0.
//
// Origin: the algorithm and the tables are those of ARM Limited's optimized-routines (`math/pow.c`, `pow_log_data.c`, `exp_data.c`),
// which glibc imported (Copyright (c) 2018, Arm Limited). This file was ported from glibc's copy, not
// from upstream, so it follows glibc's licence (above). Upstream history: Apache-2.0 at first (spring
// 2018), MIT from 2018-11-12, "MIT OR Apache-2.0 WITH LLVM-exception" since 2022-02-10; all three are
// compatible with GPL-3.0, and the table values were cross-checked against current upstream in the 5.5a review.
//
// Altered on 2026-10-09: rewritten from C to Rust for DDNet-AI (safe bit operations instead of unions), the build
// configuration of glibc 2.39 on x86-64 fixed (`__FP_FAST_FMA` defined, `WANT_ROUNDING`, `WANT_ERRNO`), and
// every `a + b * c` that GCC fuses when compiling glibc's `e_pow-fma.c` with `-mfma -mavx2` written as an
// explicit `fma(b, c, a)`, so the result is bit-identical to glibc 2.39's x86-64 FMA variant.

//! `pow` (double), bit-identical to glibc 2.39's `__pow_fma` (the function `f64::powf` calls on x86-64
//! Linux, and the one V8's `Math.pow` reaches through `std::pow` in the Linux Node the TS parity
//! corpus was recorded with).

use crate::tables::{
    EXP_INVLN2N, EXP_NEGLN2HIN, EXP_NEGLN2LON, EXP_POLY, EXP_SHIFT, EXP_TAB, POW_LOG_POLY, POW_LOG_TAB,
};
use crate::util::{divzero_f64, fma, invalid_f64, nan_sum_f64, oflow_f64, uflow_f64};

const POW_LOG_TABLE_BITS: u32 = 7;
const EXP_TABLE_BITS: u32 = 7;
const OFF: u64 = 0x3fe6_9555_0000_0000;
const SIGN_BIAS: u32 = 0x800 << EXP_TABLE_BITS;
const ONE: u64 = 0x3ff0_0000_0000_0000;
const INF: u64 = 0x7ff0_0000_0000_0000;

const LN2HI: f64 = f64::from_bits(0x3fe6_2e42_fefa_3800);
const LN2LO: f64 = f64::from_bits(0x3d2e_f357_93c7_6730);

/// Top 12 bits of a double (sign and exponent bits).
#[inline(always)]
fn top12(x: f64) -> u32 {
    (x.to_bits() >> 52) as u32
}

/// `y + TAIL = log(x)` where the rounded result is `y` and `TAIL` has about 15 additional bits of
/// precision. `ix` is the bit representation of x, normalized in the subnormal range using the sign bit
/// for the exponent.
#[inline(always)]
fn log_inline(ix: u64) -> (f64, f64) {
    let a = |i: usize| f64::from_bits(POW_LOG_POLY[i]);
    // x = 2^k z; where z is in range [OFF,2*OFF) and exact.
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (52 - POW_LOG_TABLE_BITS)) % (1 << POW_LOG_TABLE_BITS)) as usize;
    let k = (tmp as i64) >> 52; // arithmetic shift
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    let z = f64::from_bits(iz);
    let kd = k as f64;

    // log(x) = k*Ln2 + log(c) + log1p(z/c-1).
    let invc = f64::from_bits(POW_LOG_TAB[3 * i]);
    let logc = f64::from_bits(POW_LOG_TAB[3 * i + 1]);
    let logctail = f64::from_bits(POW_LOG_TAB[3 * i + 2]);

    // Note: 1/c is j/N or j/N/2 where j is an integer in [N,2N) and |z/c - 1| < 1/N, so r = z/c - 1 is
    // exactly representible.
    let r = fma(z, invc, -1.0);

    // k*Ln2 + log(c) + r.
    let t1 = fma(kd, LN2HI, logc);
    let t2 = t1 + r;
    let lo1 = fma(kd, LN2LO, logctail);
    let lo2 = (t1 - t2) + r;

    // Evaluation is optimized assuming superscalar pipelined execution.
    let ar = a(0) * r; // A[0] = -0.5.
    let ar2 = r * ar;
    let ar3 = r * ar2;
    // k*Ln2 + log(c) + r + A[0]*r*r.
    let hi = t2 + ar2;
    let lo3 = fma(ar, r, -ar2);
    let lo4 = (t2 - hi) + ar2;
    // p = log1p(r) - r - A[0]*r*r.
    // ar3 * (A[1] + r*A[2] + ar2*(A[3] + r*A[4] + ar2*(A[5] + r*A[6])))
    let inner = fma(ar2, fma(r, a(6), a(5)), fma(r, a(4), a(3)));
    let poly = fma(ar2, inner, fma(r, a(2), a(1)));
    // lo = lo1 + lo2 + lo3 + lo4 + p
    let lo = fma(ar3, poly, ((lo1 + lo2) + lo3) + lo4);
    let y = hi + lo;
    let tail = (hi - y) + lo;
    (y, tail)
}

/// Handles cases that may overflow or underflow when computing the result that is `scale*(1+TMP)`
/// without intermediate rounding.
#[inline(always)]
fn specialcase(tmp: f64, sbits: u64, ki: u64) -> f64 {
    if (ki & 0x8000_0000) == 0 {
        // k > 0, the exponent of scale might have overflowed by <= 460.
        let sbits = sbits.wrapping_sub(1009u64 << 52);
        let scale = f64::from_bits(sbits);
        // 0x1p1009 * (scale + scale * tmp)
        return f64::from_bits((1023 + 1009) << 52) * fma(scale, tmp, scale);
    }
    // k < 0, need special care in the subnormal range.
    let sbits = sbits.wrapping_add(1022u64 << 52);
    // Note: sbits is signed scale.
    let scale = f64::from_bits(sbits);
    let m = scale * tmp;
    let mut y = scale + m;
    if y.abs() < 1.0 {
        // Round y to the right precision before scaling it into the subnormal range to avoid double
        // rounding that can cause 0.5+E/2 ulp error where E is the worst-case ulp error outside the
        // subnormal range.
        let one = if y < 0.0 { -1.0 } else { 1.0 };
        let lo = (scale - y) + m;
        let hi = one + y;
        let lo = ((one - hi) + y) + lo;
        y = (hi + lo) - one;
        // Fix the sign of 0.
        if y == 0.0 {
            y = f64::from_bits(sbits & 0x8000_0000_0000_0000);
        }
    }
    f64::from_bits(1u64 << 52) * y // 0x1p-1022 * y
}

/// `sign*exp(x+xtail)` where `|xtail| < 2^-8/N` and `|xtail| <= |x|`. `sign_bias` is `SIGN_BIAS` or 0.
#[inline(always)]
fn exp_inline(x: f64, xtail: f64, sign_bias: u32) -> f64 {
    let c = |i: usize| f64::from_bits(EXP_POLY[i]);
    let shift = f64::from_bits(EXP_SHIFT);
    let mut abstop = top12(x) & 0x7ff;
    // top12 (0x1p-54) = 0x3c9, top12 (512.0) = 0x408, top12 (1024.0) = 0x409.
    if abstop.wrapping_sub(0x3c9) >= 0x408 - 0x3c9 {
        if abstop.wrapping_sub(0x3c9) >= 0x8000_0000 {
            // Avoid spurious underflow for tiny x.
            // Note: 0 is common input.
            let one = 1.0 + x;
            return if sign_bias != 0 { -one } else { one };
        }
        if abstop >= 0x409 {
            // Note: inf and nan are already handled.
            return if x.to_bits() >> 63 != 0 {
                uflow_f64(sign_bias != 0)
            } else {
                oflow_f64(sign_bias != 0)
            };
        }
        // Large x is special cased below.
        abstop = 0;
    }

    // exp(x) = 2^(k/N) * exp(r), with exp(r) in [2^(-1/2N),2^(1/2N)].
    // x = ln2/N*k + r, with int k and r in [-ln2/2N, ln2/2N].
    // z = InvLn2N * x; kd = z + Shift (fused)
    let kd = fma(f64::from_bits(EXP_INVLN2N), x, shift);
    let ki = kd.to_bits();
    let kd = kd - shift;
    // r = x + kd*NegLn2hiN + kd*NegLn2loN
    let r = fma(
        kd,
        f64::from_bits(EXP_NEGLN2LON),
        fma(kd, f64::from_bits(EXP_NEGLN2HIN), x),
    );
    // The code assumes 2^-200 < |xtail| < 2^-8/N.
    let r = r + xtail;
    // 2^(k/N) ~= scale * (1 + tail).
    let idx = (2 * (ki % (1 << EXP_TABLE_BITS))) as usize;
    let top = ki.wrapping_add(u64::from(sign_bias)) << (52 - EXP_TABLE_BITS);
    let tail = f64::from_bits(EXP_TAB[idx]);
    // This is only a valid scale when -1023*N < k < 1024*N.
    let sbits = EXP_TAB[idx + 1].wrapping_add(top);
    // exp(x) = 2^(k/N) * exp(r) ~= scale + scale * (tail + exp(r) - 1).
    let r2 = r * r;
    // tail + r + r2 * (C2 + r * C3) + r2 * r2 * (C4 + r * C5)
    let tmp = fma(r2 * r2, fma(r, c(3), c(2)), fma(r2, fma(r, c(1), c(0)), tail + r));
    if abstop == 0 {
        return specialcase(tmp, sbits, ki);
    }
    let scale = f64::from_bits(sbits);
    // Note: tmp == 0 or |tmp| > 2^-200 and scale > 2^-739, so there is no spurious underflow here even
    // without fma.
    fma(scale, tmp, scale)
}

/// Returns 0 if not int, 1 if odd int, 2 if even int. The argument is the bit representation of a
/// non-zero finite floating-point value.
#[inline(always)]
fn checkint(iy: u64) -> u32 {
    let e = ((iy >> 52) & 0x7ff) as u32;
    if e < 0x3ff {
        return 0;
    }
    if e > 0x3ff + 52 {
        return 2;
    }
    if iy & ((1u64 << (0x3ff + 52 - e)) - 1) != 0 {
        return 0;
    }
    if iy & (1u64 << (0x3ff + 52 - e)) != 0 {
        return 1;
    }
    2
}

/// Returns true if the input is the bit representation of 0, infinity or nan.
#[inline(always)]
fn zeroinfnan(i: u64) -> bool {
    i.wrapping_mul(2).wrapping_sub(1) >= INF.wrapping_mul(2).wrapping_sub(1)
}

/// glibc's `issignaling_inline (double)` on x86.
#[inline(always)]
fn issignaling(x: f64) -> bool {
    (x.to_bits() ^ 0x0008_0000_0000_0000).wrapping_mul(2) > 0xfff0_0000_0000_0000
}

/// `pow(x, y)`, bit-identical to glibc 2.39 (x86-64, FMA variant).
#[must_use]
pub fn pow(x: f64, y: f64) -> f64 {
    let mut sign_bias = 0u32;
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    let mut topx = top12(x);
    let topy = top12(y);
    if topx.wrapping_sub(0x001) >= 0x7ff - 0x001 || (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be {
        // Note: if |y| > 1075 * ln2 * 2^53 ~= 0x1.749p62 then pow(x,y) = inf/0 and if |y| < 2^-54 / 1075
        // ~= 0x1.e7b6p-65 then pow(x,y) = +-1.
        // Special cases: (x < 0x1p-126 or inf or nan) or (|y| < 0x1p-65 or |y| >= 0x1p63 or nan).
        if zeroinfnan(iy) {
            if iy.wrapping_mul(2) == 0 {
                return if issignaling(x) { nan_sum_f64(x, y) } else { 1.0 };
            }
            if ix == ONE {
                return if issignaling(y) { nan_sum_f64(x, y) } else { 1.0 };
            }
            if ix.wrapping_mul(2) > INF.wrapping_mul(2) || iy.wrapping_mul(2) > INF.wrapping_mul(2) {
                return nan_sum_f64(x, y);
            }
            if ix.wrapping_mul(2) == ONE.wrapping_mul(2) {
                return 1.0;
            }
            if (ix.wrapping_mul(2) < ONE.wrapping_mul(2)) == (iy >> 63 == 0) {
                return 0.0; // |x|<1 && y==inf or |x|>1 && y==-inf.
            }
            return y * y;
        }
        if zeroinfnan(ix) {
            let mut x2 = x * x;
            if ix >> 63 != 0 && checkint(iy) == 1 {
                x2 = -x2;
                sign_bias = 1;
            }
            if ix.wrapping_mul(2) == 0 && iy >> 63 != 0 {
                return divzero_f64(sign_bias != 0);
            }
            return if iy >> 63 != 0 { 1.0 / x2 } else { x2 };
        }
        // Here x and y are non-zero finite.
        if ix >> 63 != 0 {
            // Finite x < 0.
            let yint = checkint(iy);
            if yint == 0 {
                return invalid_f64(x);
            }
            if yint == 1 {
                sign_bias = SIGN_BIAS;
            }
            ix &= 0x7fff_ffff_ffff_ffff;
            topx &= 0x7ff;
        }
        if (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be {
            // Note: sign_bias == 0 here because y is not odd.
            if ix == ONE {
                return 1.0;
            }
            if (topy & 0x7ff) < 0x3be {
                // |y| < 2^-65, x^y ~= 1 + y*log(x).
                return if ix > ONE { 1.0 + y } else { 1.0 - y };
            }
            return if (ix > ONE) == (topy < 0x800) {
                oflow_f64(false)
            } else {
                uflow_f64(false)
            };
        }
        if topx == 0 {
            // Normalize subnormal x so exponent becomes negative.
            ix = (x * f64::from_bits(0x4330_0000_0000_0000)).to_bits(); // x * 0x1p52
            ix &= 0x7fff_ffff_ffff_ffff;
            ix = ix.wrapping_sub(52u64 << 52);
        }
    }

    let (hi, lo) = log_inline(ix);
    let ehi = y * hi;
    // elo = y * lo + fma (y, hi, -ehi)
    let elo = fma(y, lo, fma(y, hi, -ehi));
    exp_inline(ehi, elo, sign_bias)
}
