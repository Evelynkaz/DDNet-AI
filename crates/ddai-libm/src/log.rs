// Double-precision natural logarithm, ported from the ARM optimized-routines `math/log.c` and
// `math/log_data.c` (the algorithm glibc 2.28+ ships as `log`).
//
// Copyright (c) 2018, Arm Limited.
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
// configuration of glibc 2.39 on x86-64 fixed (`__FP_FAST_FMA` defined, `WANT_ROUNDING`, `WANT_ERRNO`), and
// every `a + b * c` that GCC fuses when compiling glibc's `e_log-fma.c` with `-mfma -mavx2` written as an
// explicit `fma(b, c, a)`, so the result is bit-identical to glibc 2.39's x86-64 FMA variant.

//! `log`, bit-identical to glibc 2.39's `__log_fma` (the function `f64::ln` calls on x86-64 Linux).

use crate::tables::{LOG_LN2HI, LOG_LN2LO, LOG_POLY, LOG_POLY1, LOG_TAB};
use crate::util::{divzero_f64, fma, invalid_f64};

const LOG_TABLE_BITS: u32 = 7;
const N: u64 = 1 << LOG_TABLE_BITS;
const OFF: u64 = 0x3fe6_0000_0000_0000;
/// `asuint64 (1.0 - 0x1p-4)`.
const LO: u64 = 0x3fee_0000_0000_0000;
/// `asuint64 (1.0 + 0x1.09p-4)`.
const HI: u64 = 0x3ff1_0900_0000_0000;
const ONE_BITS: u64 = 0x3ff0_0000_0000_0000;
const INF_BITS: u64 = 0x7ff0_0000_0000_0000;

/// `log(x)`, bit-identical to glibc 2.39 (x86-64, FMA variant).
#[must_use]
pub fn log(x: f64) -> f64 {
    let b = |i: usize| f64::from_bits(LOG_POLY1[i]);
    let a = |i: usize| f64::from_bits(LOG_POLY[i]);
    let ln2hi = f64::from_bits(LOG_LN2HI);
    let ln2lo = f64::from_bits(LOG_LN2LO);

    let mut ix = x.to_bits();
    let top = (ix >> 48) as u32;

    if ix.wrapping_sub(LO) < HI - LO {
        // Handle close to 1.0 inputs separately.
        // Fix sign of zero with downward rounding when x==1.
        if ix == ONE_BITS {
            return 0.0;
        }
        let r = x - 1.0;
        let r2 = r * r;
        let r3 = r * r2;
        // B[1] + r*B[2] + r2*B[3] + r3*(B[4] + r*B[5] + r2*B[6] + r3*(B[7] + r*B[8] + r2*B[9] + r3*B[10]))
        let t1 = fma(r2, b(3), fma(r, b(2), b(1)));
        let t2 = fma(r2, b(6), fma(r, b(5), b(4)));
        let t3 = fma(r3, b(10), fma(r2, b(9), fma(r, b(8), b(7))));
        let q = fma(fma(t3, r3, t2), r3, t1);
        // rhi = r + w - w, with both `w` products fused.
        let rhi = fma(-134_217_728.0, r, fma(r, 134_217_728.0, r));
        let rlo = r - rhi;
        let rr = rhi * rhi;
        // w = rhi * rhi * B[0]  (B[0] == -0.5); hi = r + w; lo = r - hi + w;
        let hi = fma(rr, b(0), r);
        let mut lo = fma(rr, b(0), r - hi);
        // lo += B[0] * rlo * (rhi + r);
        lo = fma(b(0) * rlo, r + rhi, lo);
        // y = r3 * q; y += lo; y += hi;
        return fma(r3, q, lo) + hi;
    }
    if top.wrapping_sub(0x0010) >= 0x7ff0 - 0x0010 {
        // x < 0x1p-1022 or inf or nan.
        if ix.wrapping_mul(2) == 0 {
            return divzero_f64(true);
        }
        if ix == INF_BITS {
            // log(inf) == inf.
            return x;
        }
        if (top & 0x8000) != 0 || (top & 0x7ff0) == 0x7ff0 {
            return invalid_f64(x);
        }
        // x is subnormal, normalize it.
        ix = (x * f64::from_bits(0x4330_0000_0000_0000)).to_bits(); // x * 0x1p52
        ix = ix.wrapping_sub(52u64 << 52);
    }

    // x = 2^k z; where z is in range [OFF,2*OFF) and exact.
    // The range is split into N subintervals.
    // The ith subinterval contains z and c is near its center.
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (52 - LOG_TABLE_BITS)) % N) as usize;
    let k = (tmp as i64) >> 52; // arithmetic shift
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    let invc = f64::from_bits(LOG_TAB[2 * i]);
    let logc = f64::from_bits(LOG_TAB[2 * i + 1]);
    let z = f64::from_bits(iz);

    // log(x) = log1p(z/c-1) + log(c) + k*Ln2.
    // r ~= z/c - 1, |r| < 1/(2*N).
    // rounding error: 0x1p-55/N.
    let r = fma(z, invc, -1.0);
    let kd = k as f64;

    // hi + lo = r + log(c) + k*Ln2.
    let w = fma(kd, ln2hi, logc);
    let hi = w + r;
    let lo = fma(kd, ln2lo, (w - hi) + r);

    // log(x) = lo + (log1p(r) - r) + hi.
    let r2 = r * r; // rounding error: 0x1p-54/N^2.
    // y = lo + r2*A[0] + r*r2*(A[1] + r*A[2] + r2*(A[3] + r*A[4])) + hi
    let poly = fma(fma(r, a(4), a(3)), r2, fma(r, a(2), a(1)));
    let y = fma(r * r2, poly, fma(r2, a(0), lo));
    y + hi
}
