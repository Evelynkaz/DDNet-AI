// `hypotf` and `hypot`, ported from glibc 2.39: `sysdeps/ieee754/flt-32/e_hypotf.c` and
// `sysdeps/ieee754/dbl-64/e_hypot.c` (the latter implements the correction of "An Improved Algorithm for
// hypot(a,b)" by Carlos F. Borges, arXiv:1904.09481, MyHypot3).
//
// Copyright (C) 2012-2024 (hypotf) and 2021-2024 (hypot) Free Software Foundation, Inc.
// This file is part of the GNU C Library.
//
// The GNU C Library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// The GNU C Library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with the GNU C Library; if not, see
// <https://www.gnu.org/licenses/>.
//
// LICENSE NOTE (see NOTICE): this file is a derived work of LGPL-2.1-or-later code. Section 3 of the
// GNU LGPL version 2.1 allows applying the terms of the ordinary GNU General Public License instead of the
// LGPL to a copy of the library; this copy is used under the terms of the GNU GPL version 3, which is the
// licence of DDNet-AI as a whole. The copyright and permission notices above are preserved.
//
// Altered: rewritten from C to Rust for DDNet-AI. x86-64 glibc builds both files without FMA (no `ifunc`
// variant exists for them), so the non-`__FP_FAST_FMA` kernel is the one ported, with the same operation
// order and no fused operation.

//! `hypotf` and `hypot`, bit-identical to glibc 2.39's.

use crate::util::{nan_sum_f32, nan_sum_f64};

/// glibc's `issignaling (float)` on x86.
#[inline(always)]
fn issignaling_f32(x: f32) -> bool {
    (x.to_bits() ^ 0x0040_0000).wrapping_mul(2) > 0xff80_0000
}

/// glibc's `issignaling_inline (double)` on x86.
#[inline(always)]
fn issignaling_f64(x: f64) -> bool {
    (x.to_bits() ^ 0x0008_0000_0000_0000).wrapping_mul(2) > 0xfff0_0000_0000_0000
}

/// `hypotf(x, y)`: `sqrt(x*x + y*y)` evaluated in double precision (the squares of two floats and their
/// sum are within a double's reach: the squares are exact, the sum rounds once) and rounded to float.
#[must_use]
pub fn hypotf(x: f32, y: f32) -> f32 {
    if !x.is_finite() || !y.is_finite() {
        if (x.is_infinite() || y.is_infinite()) && !issignaling_f32(x) && !issignaling_f32(y) {
            return f32::INFINITY;
        }
        return nan_sum_f32(x, y);
    }
    let (dx, dy) = (f64::from(x), f64::from(y));
    (dx * dx + dy * dy).sqrt() as f32
}

const SCALE: f64 = f64::from_bits(0x1a70_0000_0000_0000); // 0x1p-600
const LARGE_VAL: f64 = f64::from_bits(0x5fe0_0000_0000_0000); // 0x1p+511
const TINY_VAL: f64 = f64::from_bits(0x2340_0000_0000_0000); // 0x1p-459
const EPS: f64 = f64::from_bits(0x3c90_0000_0000_0000); // 0x1p-54

/// Hypot kernel. The inputs must be adjusted so that `ax >= ay >= 0` and squaring `ax`, `ay` and
/// `(ax - ay)` does not overflow or underflow.
#[inline(always)]
fn kernel(ax: f64, ay: f64) -> f64 {
    let mut h = (ax * ax + ay * ay).sqrt();
    let (t1, t2);
    if h <= 2.0 * ay {
        let delta = h - ay;
        t1 = ax * (2.0 * delta - ax);
        t2 = (delta - 2.0 * (ax - ay)) * delta;
    } else {
        let delta = h - ax;
        t1 = 2.0 * delta * (ax - 2.0 * ay);
        t2 = (4.0 * delta - ay) * ay + delta * delta;
    }
    h -= (t1 + t2) / (2.0 * h);
    h
}

/// `hypot(x, y)`, bit-identical to glibc 2.39.
#[must_use]
pub fn hypot(x: f64, y: f64) -> f64 {
    if !x.is_finite() || !y.is_finite() {
        if (x.is_infinite() || y.is_infinite()) && !issignaling_f64(x) && !issignaling_f64(y) {
            return f64::INFINITY;
        }
        return nan_sum_f64(x, y);
    }

    let x = x.abs();
    let y = y.abs();

    let ax = if x < y { y } else { x };
    let ay = if x < y { x } else { y };

    // If ax is huge, scale both inputs down.
    if ax > LARGE_VAL {
        if ay <= ax * EPS {
            return ax + ay;
        }
        return kernel(ax * SCALE, ay * SCALE) / SCALE;
    }

    // If ay is tiny, scale both inputs up.
    if ay < TINY_VAL {
        if ax >= ay / EPS {
            return ax + ay;
        }
        return kernel(ax / SCALE, ay / SCALE) * SCALE;
    }

    // Common case: ax is not huge and ay is not tiny.
    if ay <= ax * EPS {
        return ax + ay;
    }
    kernel(ax, ay)
}
