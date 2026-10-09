// Single-precision sin/cos, ported from the ARM optimized-routines `math/sincosf.c`, `math/sincosf.h`,
// `math/sincosf_data.c` (the algorithm glibc 2.28+ ships as `sinf`/`cosf`).
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
// Altered: rewritten from C to Rust for DDNet-AI (safe bit operations instead of unions), and every
// `a + b * c` that GCC fuses when compiling glibc's `s_sinf-fma.c` / `s_cosf-fma.c` with `-mfma -mavx2`
// is written as an explicit `fma(b, c, a)`, so the result is bit-identical to glibc 2.39's x86-64 FMA
// variant (checked exhaustively over all 2^32 inputs, see tests/glibc_probe.rs).

//! `sinf` and `cosf`, bit-identical to glibc 2.39's `__sinf_fma` / `__cosf_fma`.

use crate::tables::{INV_PIO4, SINCOSF_TABLE};
use crate::util::{fma, invalid_f32};

/// One `sincos_t` entry of `__sincosf_table` (the sign array is the same in both entries).
#[derive(Clone, Copy)]
struct Poly {
    hpi_inv: f64,
    hpi: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    s1: f64,
    s2: f64,
    s3: f64,
}

const fn poly(entry: usize) -> Poly {
    const fn at(entry: usize, i: usize) -> f64 {
        f64::from_bits(SINCOSF_TABLE[entry * 10 + i])
    }
    Poly {
        hpi_inv: at(entry, 0),
        hpi: at(entry, 1),
        c0: at(entry, 2),
        c1: at(entry, 3),
        c2: at(entry, 4),
        c3: at(entry, 5),
        c4: at(entry, 6),
        s1: at(entry, 7),
        s2: at(entry, 8),
        s3: at(entry, 9),
    }
}

/// `__sincosf_table[0]` and `[1]` (the cosine polynomial is negated in the second).
const TABLE: [Poly; 2] = [poly(0), poly(1)];
/// `sign[4]` of both entries: the sign of sine in quadrants 0..3.
const SIGN: [f64; 4] = [1.0, -1.0, -1.0, 1.0];
/// `pi63`: 2PI * 2^-64.
const PI63: f64 = f64::from_bits(0x3c1921fb54442d18);

/// Top 12 bits of the float representation with the sign bit cleared.
#[inline(always)]
fn abstop12(x: f32) -> u32 {
    (x.to_bits() >> 20) & 0x7ff
}

/// `abstop12 (pio4)` with `pio4 = 0x1.921FB6p-1f`.
const TOP_PIO4: u32 = 0x3f4;
/// `abstop12 (0x1p-12f)`.
const TOP_2M12: u32 = 0x398;
/// `abstop12 (120.0f)`.
const TOP_120: u32 = 0x42f;
/// `abstop12 (INFINITY)`.
const TOP_INF: u32 = 0x7f8;

/// Fast range reduction using a single multiply-subtract: the modulo of `x` as a value between
/// -PI/4 and PI/4 and the quadrant. Accurate for |x| <= 120.
#[inline(always)]
fn reduce_fast(x: f64, p: &Poly) -> (f64, i32) {
    // hpi_inv is prescaled by 2^24 so the quadrant ends up in bits 24..31.
    let r = x * p.hpi_inv;
    let n = ((r as i32) + 0x80_0000) >> 24;
    // `x - n * hpi`, fused.
    (fma(-f64::from(n), p.hpi, x), n)
}

/// Range reduction of `xi` (a reinterpreted float, >= 2.0f, sign ignored) to a multiple of PI/2 using a
/// 192-bit table of 4/PI and integer arithmetic. Returns the modulo (between -PI/4 and PI/4) and the
/// quadrant.
#[inline(always)]
fn reduce_large(xi: u32) -> (f64, i32) {
    let base = ((xi >> 26) & 15) as usize;
    let shift = (xi >> 23) & 7;
    let xi = ((xi & 0xff_ffff) | 0x80_0000) << shift;
    // `res0 = xi * arr[0]` multiplies two `uint32_t`: the product wraps at 32 bits.
    let res0 = u64::from(xi.wrapping_mul(INV_PIO4[base]));
    let res1 = u64::from(xi) * u64::from(INV_PIO4[base + 4]);
    let res2 = u64::from(xi) * u64::from(INV_PIO4[base + 8]);
    let res0 = (res2 >> 32) | (res0 << 32);
    let res0 = res0.wrapping_add(res1);
    let n = res0.wrapping_add(1u64 << 61) >> 62;
    let res0 = res0.wrapping_sub(n << 62);
    let x = (res0 as i64) as f64;
    (x * PI63, n as i32)
}

/// Sine (`n` even) or cosine (`n` odd) polynomial of `x` and `x2 = x * x`.
#[inline(always)]
fn sinf_poly(x: f64, x2: f64, p: &Poly, n: i32) -> f32 {
    if n & 1 == 0 {
        let x3 = x * x2;
        // `p->s2 + x2 * p->s3`
        let s1 = fma(x2, p.s3, p.s2);
        let x7 = x3 * x2;
        // `x + x3 * p->s1`
        let s = fma(x3, p.s1, x);
        // `s + x7 * s1`
        fma(x7, s1, s) as f32
    } else {
        let x4 = x2 * x2;
        // `p->c3 + x2 * p->c4`
        let c2 = fma(x2, p.c4, p.c3);
        // `p->c0 + x2 * p->c1`
        let c1 = fma(x2, p.c1, p.c0);
        let x6 = x4 * x2;
        // `c1 + x4 * p->c2`
        let c = fma(x4, p.c2, c1);
        // `c + x6 * c2`
        fma(x6, c2, c) as f32
    }
}

/// `sinf(y)`, bit-identical to glibc 2.39 (x86-64, FMA variant).
#[must_use]
pub fn sinf(y: f32) -> f32 {
    let top = abstop12(y);
    let x = f64::from(y);
    if top < TOP_PIO4 {
        let s = x * x;
        if top < TOP_2M12 {
            // Tiny y (and +-0, subnormals): sin(y) == y (glibc also forces the underflow flag here).
            return y;
        }
        sinf_poly(x, s, &TABLE[0], 0)
    } else if top < TOP_120 {
        let (x, n) = reduce_fast(x, &TABLE[0]);
        // Set up the signs for sin and cos.
        let s = SIGN[(n & 3) as usize];
        let p = if n & 2 != 0 { &TABLE[1] } else { &TABLE[0] };
        sinf_poly(x * s, x * x, p, n)
    } else if top < TOP_INF {
        let xi = y.to_bits();
        let sign = (xi >> 31) as i32;
        let (x, n) = reduce_large(xi);
        // Set up signs for sin and cos - include the original sign.
        let s = SIGN[((n + sign) & 3) as usize];
        let p = if (n + sign) & 2 != 0 { &TABLE[1] } else { &TABLE[0] };
        sinf_poly(x * s, x * x, p, n)
    } else {
        invalid_f32(y)
    }
}

/// `cosf(y)`, bit-identical to glibc 2.39 (x86-64, FMA variant).
#[must_use]
pub fn cosf(y: f32) -> f32 {
    let top = abstop12(y);
    let x = f64::from(y);
    if top < TOP_PIO4 {
        let x2 = x * x;
        if top < TOP_2M12 {
            return 1.0;
        }
        sinf_poly(x, x2, &TABLE[0], 1)
    } else if top < TOP_120 {
        let (x, n) = reduce_fast(x, &TABLE[0]);
        let s = SIGN[(n & 3) as usize];
        let p = if n & 2 != 0 { &TABLE[1] } else { &TABLE[0] };
        sinf_poly(x * s, x * x, p, n ^ 1)
    } else if top < TOP_INF {
        let xi = y.to_bits();
        let sign = (xi >> 31) as i32;
        let (x, n) = reduce_large(xi);
        let s = SIGN[((n + sign) & 3) as usize];
        let p = if (n + sign) & 2 != 0 { &TABLE[1] } else { &TABLE[0] };
        sinf_poly(x * s, x * x, p, n ^ 1)
    } else {
        invalid_f32(y)
    }
}
