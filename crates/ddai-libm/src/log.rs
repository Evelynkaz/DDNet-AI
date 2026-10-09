// Double-precision natural logarithm, ported from glibc 2.39: `sysdeps/ieee754/dbl-64/e_log.c` and
// `e_log_data.c` (function `__log`).
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
// Origin: the algorithm and the tables are those of ARM Limited's optimized-routines (`math/log.c`, `log_data.c`),
// which glibc imported (Copyright (c) 2018, Arm Limited). This file was ported from glibc's copy, not
// from upstream, so it follows glibc's licence (above). Upstream history: Apache-2.0 at first (spring
// 2018), MIT from 2018-11-12, "MIT OR Apache-2.0 WITH LLVM-exception" since 2022-02-10; all three are
// compatible with GPL-3.0, and the table values were cross-checked against current upstream in the 5.5a review.
//
// Altered on 2026-10-09: rewritten from C to Rust for DDNet-AI (safe bit operations instead of unions), the build
// configuration of glibc 2.39 on x86-64 fixed (`__FP_FAST_FMA` defined, `WANT_ROUNDING`, `WANT_ERRNO`), and
// every `a + b * c` that GCC fuses when compiling glibc's `e_log-fma.c` with `-mfma -mavx2` written as an
// explicit `fma(b, c, a)`, so the result is bit-identical to glibc 2.39's x86-64 FMA variant.

//! `log`, bit-identical to glibc 2.39's `__log_fma` (the function `f64::ln` calls on x86-64 Linux).

use crate::tables::{LOG_LN2HI, LOG_LN2LO, LOG_POLY, LOG_POLY1, LOG_TAB};
use crate::util::{Fma, dispatch, divzero_f64, invalid_f64};

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
    dispatch!(log_impl(x))
}

fn log_impl<F: Fma>(x: f64) -> f64 {
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
        let t1 = F::fma(r2, b(3), F::fma(r, b(2), b(1)));
        let t2 = F::fma(r2, b(6), F::fma(r, b(5), b(4)));
        let t3 = F::fma(r3, b(10), F::fma(r2, b(9), F::fma(r, b(8), b(7))));
        let q = F::fma(F::fma(t3, r3, t2), r3, t1);
        // rhi = r + w - w, with both `w` products fused.
        let rhi = F::fma(-134_217_728.0, r, F::fma(r, 134_217_728.0, r));
        let rlo = r - rhi;
        let rr = rhi * rhi;
        // w = rhi * rhi * B[0]  (B[0] == -0.5); hi = r + w; lo = r - hi + w;
        let hi = F::fma(rr, b(0), r);
        let mut lo = F::fma(rr, b(0), r - hi);
        // lo += B[0] * rlo * (rhi + r);
        lo = F::fma(b(0) * rlo, r + rhi, lo);
        // y = r3 * q; y += lo; y += hi;
        return F::fma(r3, q, lo) + hi;
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
    let r = F::fma(z, invc, -1.0);
    let kd = k as f64;

    // hi + lo = r + log(c) + k*Ln2.
    let w = F::fma(kd, ln2hi, logc);
    let hi = w + r;
    let lo = F::fma(kd, ln2lo, (w - hi) + r);

    // log(x) = lo + (log1p(r) - r) + hi.
    let r2 = r * r; // rounding error: 0x1p-54/N^2.
    // y = lo + r2*A[0] + r*r2*(A[1] + r*A[2] + r2*(A[3] + r*A[4])) + hi
    let poly = F::fma(F::fma(r, a(4), a(3)), r2, F::fma(r, a(2), a(1)));
    let y = F::fma(r * r2, poly, F::fma(r2, a(0), lo));
    y + hi
}
