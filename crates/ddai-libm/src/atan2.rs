// Double-precision two-argument arctangent, ported from the IBM Accurate Mathematical Library as shipped in
// glibc 2.39: `sysdeps/ieee754/dbl-64/e_atan2.c` (function `__ieee754_atan2`), with the tables of
// `uatan.tbl`, the constants of `atnat2.h` and the `EADD`/`ESUB`/`EMULV` macros of `dla.h`.
//
// IBM Accurate Mathematical Library
// written by International Business Machines Corp.
// Copyright (C) 2001-2024 Free Software Foundation, Inc.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, see <https://www.gnu.org/licenses/>.
//
// LICENSE NOTE (see NOTICE): this file is a derived work of LGPL-2.1-or-later code. Section 3 of the
// GNU LGPL version 2.1 allows applying the terms of the ordinary GNU General Public License, version 2 or
// (as that version's "or later" clause and section 3's own text allow) any later version, instead of the
// LGPL to a copy of the library; this copy is used under the terms of the GNU GPL version 3, which is the
// licence of DDNet-AI as a whole. The copyright and permission notices above are preserved.
//
// Altered: rewritten from C to Rust for DDNet-AI (safe bit operations instead of unions), the build
// configuration of glibc 2.39 on x86-64 fixed (`__FP_FAST_FMA` defined: `EMULV` uses `fma`, no multi-
// precision fallback exists any more), and every `a + b * c` that GCC fuses when compiling glibc's
// `e_atan2-fma.c` with `-mfma -mavx2` written as an explicit `fma(b, c, a)`, so the result is
// bit-identical to glibc 2.39's x86-64 FMA variant.

//! `atan2` (double), bit-identical to glibc 2.39's `__ieee754_atan2_fma`.

use crate::tables::{
    ATAN2_CIJ, ATAN2_D3, ATAN2_D5, ATAN2_D7, ATAN2_D9, ATAN2_D11, ATAN2_D13, ATAN2_HPI, ATAN2_HPI1, ATAN2_INV16,
    ATAN2_MHPI, ATAN2_MOPI, ATAN2_MQPI, ATAN2_MTQPI, ATAN2_OPI, ATAN2_OPI1, ATAN2_QPI, ATAN2_TQPI, ATAN2_TWO500,
    ATAN2_TWOM500,
};
use crate::util::{fma, nan_sum_f64};

const fn k(bits: u64) -> f64 {
    f64::from_bits(bits)
}

const D3: f64 = k(ATAN2_D3);
const D5: f64 = k(ATAN2_D5);
const D7: f64 = k(ATAN2_D7);
const D9: f64 = k(ATAN2_D9);
const D11: f64 = k(ATAN2_D11);
const D13: f64 = k(ATAN2_D13);
const INV16: f64 = k(ATAN2_INV16);
const OPI: f64 = k(ATAN2_OPI);
const OPI1: f64 = k(ATAN2_OPI1);
const MOPI: f64 = k(ATAN2_MOPI);
const HPI: f64 = k(ATAN2_HPI);
const HPI1: f64 = k(ATAN2_HPI1);
const MHPI: f64 = k(ATAN2_MHPI);
const QPI: f64 = k(ATAN2_QPI);
const MQPI: f64 = k(ATAN2_MQPI);
const TQPI: f64 = k(ATAN2_TQPI);
const MTQPI: f64 = k(ATAN2_MTQPI);
const TWO500: f64 = k(ATAN2_TWO500);
const TWOM500: f64 = k(ATAN2_TWOM500);
const TWO52: f64 = 4_503_599_627_370_496.0; // 0x1.0p52

/// 57*16**5
const EP: i32 = 59_768_832;
/// -57*16**5
const EM: i32 = -59_768_832;

/// `cij[i][j]`.
#[inline(always)]
fn cij(i: usize, j: usize) -> f64 {
    f64::from_bits(ATAN2_CIJ[i * 7 + j])
}

/// `signArctan2 (y, z)`: `copysign (z, y)`.
#[inline(always)]
fn sign_arctan2(y: f64, z: f64) -> f64 {
    z.copysign(y)
}

/// `d3 + v*(d5 + v*(d7 + v*(d9 + v*(d11 + v*d13))))`, every step fused.
#[inline(always)]
fn poly_d(v: f64) -> f64 {
    let p = fma(v, D13, D11);
    let p = fma(v, p, D9);
    let p = fma(v, p, D7);
    let p = fma(v, p, D5);
    fma(v, p, D3)
}

/// `EADD (x, y, z, zz)`: `z = x + y; zz = |x| > |y| ? ((x - z) + y) : ((y - z) + x)`.
#[inline(always)]
fn eadd(x: f64, y: f64) -> (f64, f64) {
    let z = x + y;
    let zz = if x.abs() > y.abs() { (x - z) + y } else { (y - z) + x };
    (z, zz)
}

/// `ESUB (x, y, z, zz)`: `z = x - y; zz = |x| > |y| ? ((x - z) - y) : (x - (y + z))`.
#[inline(always)]
fn esub(x: f64, y: f64) -> (f64, f64) {
    let z = x - y;
    let zz = if x.abs() > y.abs() { (x - z) - y } else { x - (y + z) };
    (z, zz)
}

/// The table index `i = (TWO52 + 256 * u) - TWO52; i -= 16;`.
#[inline(always)]
fn table_index(u: f64) -> usize {
    let i = ((TWO52 + 256.0 * u) - TWO52) as i32 - 16;
    i as usize
}

/// `atan2(y, x)` for doubles, bit-identical to glibc 2.39 (x86-64, FMA variant).
#[must_use]
pub fn atan2(y: f64, x: f64) -> f64 {
    let xb = x.to_bits();
    let ux = (xb >> 32) as u32;
    let dx = xb as u32;
    // x = NaN
    if (ux & 0x7ff0_0000) == 0x7ff0_0000 && ((ux & 0x000f_ffff) | dx) != 0 {
        return nan_sum_f64(x, y);
    }
    let yb = y.to_bits();
    let uy = (yb >> 32) as u32;
    let dy = yb as u32;
    // y = NaN
    if (uy & 0x7ff0_0000) == 0x7ff0_0000 && ((uy & 0x000f_ffff) | dy) != 0 {
        return nan_sum_f64(y, y);
    }

    // y = +-0
    if uy == 0x0000_0000 {
        if dy == 0 {
            return if (ux & 0x8000_0000) == 0 { 0.0 } else { OPI };
        }
    } else if uy == 0x8000_0000 && dy == 0 {
        return if (ux & 0x8000_0000) == 0 { -0.0 } else { MOPI };
    }

    // x = +-0
    if x == 0.0 {
        return if (uy & 0x8000_0000) == 0 { HPI } else { MHPI };
    }

    // x = +-INF
    if ux == 0x7ff0_0000 {
        if dx == 0 {
            if uy == 0x7ff0_0000 {
                if dy == 0 {
                    return QPI;
                }
            } else if uy == 0xfff0_0000 {
                if dy == 0 {
                    return MQPI;
                }
            } else {
                return if (uy & 0x8000_0000) == 0 { 0.0 } else { -0.0 };
            }
        }
    } else if ux == 0xfff0_0000 && dx == 0 {
        if uy == 0x7ff0_0000 {
            if dy == 0 {
                return TQPI;
            }
        } else if uy == 0xfff0_0000 {
            if dy == 0 {
                return MTQPI;
            }
        } else {
            return if (uy & 0x8000_0000) == 0 { OPI } else { MOPI };
        }
    }

    // y = +-INF
    if uy == 0x7ff0_0000 {
        if dy == 0 {
            return HPI;
        }
    } else if uy == 0xfff0_0000 && dy == 0 {
        return MHPI;
    }

    // Round-to-nearest is assumed (glibc forces it with SET_RESTORE_ROUND).
    // either x/y or y/x is very close to zero
    let mut ax = x.abs();
    let mut ay = y.abs();
    let de = (uy & 0x7ff0_0000) as i32 - (ux & 0x7ff0_0000) as i32;
    if de >= EP {
        return if y > 0.0 { HPI } else { MHPI };
    } else if de <= EM {
        if x > 0.0 {
            let z = ay / ax;
            return sign_arctan2(y, z);
        }
        return if y > 0.0 { OPI } else { MOPI };
    }

    // if either x or y is extremely close to zero, scale abs(x), abs(y).
    if ax < TWOM500 || ay < TWOM500 {
        ax *= TWO500;
        ay *= TWO500;
    }

    // Likewise for large x and y.
    if ax > TWO500 || ay > TWO500 {
        ax *= TWOM500;
        ay *= TWOM500;
    }

    // x,y which are neither special nor extreme
    let (u, du) = if ay < ax {
        let u = ay / ax;
        // EMULV (ax, u, v, vv)
        let v = ax * u;
        let vv = fma(ax, u, -v);
        (u, ((ay - v) - vv) / ax)
    } else {
        let u = ax / ay;
        let v = ay * u;
        let vv = fma(ay, u, -v);
        (u, ((ax - v) - vv) / ay)
    };

    if x > 0.0 {
        // (i)   x>0, abs(y)< abs(x):  atan(ay/ax)
        if ay < ax {
            if u < INV16 {
                let v = u * u;
                // zz = du + u * v * (d3 + v * (d5 + ...))
                let zz = fma(u * v, poly_d(v), du);
                let z = u + zz;
                return sign_arctan2(y, z);
            }

            let i = table_index(u);
            let t3 = u - cij(i, 0);
            // EADD (t3, du, v, dv)
            let (v, dv) = eadd(t3, du);
            let t1 = cij(i, 1);
            let t2 = cij(i, 2);
            // zz = v*t2 + (dv*t2 + v*v*(c3 + v*(c4 + v*(c5 + v*c6))))
            let p = fma(v, cij(i, 6), cij(i, 5));
            let p = fma(v, p, cij(i, 4));
            let p = fma(v, p, cij(i, 3));
            let zz = fma(v, t2, fma(dv, t2, (v * v) * p));
            let z = t1 + zz;
            return sign_arctan2(y, z);
        }

        // (ii)  x>0, abs(x)<=abs(y):  pi/2-atan(ax/ay)
        if u < INV16 {
            let v = u * u;
            // zz = u * v * (d3 + ...). Unlike in case (i) it is NOT fused with the addition that uses it:
            // that addition sits behind the branches of ESUB/EADD, in another basic block, where GCC's
            // multiply-add contraction does not reach.
            let zz = (u * v) * poly_d(v);
            // ESUB (hpi, u, t2, cor)
            let (t2, cor) = esub(HPI, u);
            let t3 = ((HPI1 + cor) - du) - zz;
            let z = t2 + t3;
            return sign_arctan2(y, z);
        }

        let i = table_index(u);
        let v = (u - cij(i, 0)) + du;
        let p = fma(v, cij(i, 6), cij(i, 5));
        let p = fma(v, p, cij(i, 4));
        let p = fma(v, p, cij(i, 3));
        let p = fma(v, p, cij(i, 2));
        // zz = hpi1 - v * p
        let zz = fma(-v, p, HPI1);
        let t1 = HPI - cij(i, 1);
        let z = t1 + zz;
        return sign_arctan2(y, z);
    }

    // (iii) x<0, abs(x)< abs(y):  pi/2+atan(ax/ay)
    if ax < ay {
        if u < INV16 {
            let v = u * u;
            let zz = (u * v) * poly_d(v); // not fused, see case (ii)
            // EADD (hpi, u, t2, cor)
            let (t2, cor) = eadd(HPI, u);
            let t3 = ((HPI1 + cor) + du) + zz;
            let z = t2 + t3;
            return sign_arctan2(y, z);
        }

        let i = table_index(u);
        let v = (u - cij(i, 0)) + du;
        let p = fma(v, cij(i, 6), cij(i, 5));
        let p = fma(v, p, cij(i, 4));
        let p = fma(v, p, cij(i, 3));
        let p = fma(v, p, cij(i, 2));
        // zz = hpi1 + v * p
        let zz = fma(v, p, HPI1);
        let t1 = HPI + cij(i, 1);
        let z = t1 + zz;
        return sign_arctan2(y, z);
    }

    // (iv)  x<0, abs(y)<=abs(x):  pi-atan(ax/ay)
    if u < INV16 {
        let v = u * u;
        let zz = (u * v) * poly_d(v); // not fused, see case (ii)
        // ESUB (opi, u, t2, cor)
        let (t2, cor) = esub(OPI, u);
        let t3 = ((OPI1 + cor) - du) - zz;
        let z = t2 + t3;
        return sign_arctan2(y, z);
    }

    let i = table_index(u);
    let v = (u - cij(i, 0)) + du;
    let p = fma(v, cij(i, 6), cij(i, 5));
    let p = fma(v, p, cij(i, 4));
    let p = fma(v, p, cij(i, 3));
    let p = fma(v, p, cij(i, 2));
    // zz = opi1 - v * p
    let zz = fma(-v, p, OPI1);
    let t1 = OPI - cij(i, 1);
    let z = t1 + zz;
    sign_arctan2(y, z)
}
