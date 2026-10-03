//! Lane arithmetic for the batched backend: fixed-size `[f32; 8]` "cells" (8 batch lanes), written
//! as plain loops over constant trip counts so the compiler auto-vectorises them (no `unsafe`, no
//! `std::simd`, no target features: on the default x86-64 baseline an 8-lane cell is two SSE2
//! registers, on `x86-64-v3` one AVX2 register).
//!
//! Also the vector-friendly `f(V)` / `f'(V)` the batched path uses instead of libm's scalar
//! `tanhf`/`coshf` (which LLVM cannot vectorise and which would cost as much as the edge work):
//! [`activation_row`] and [`derivative_from_rate_row`] (plus the exact [`saturated_derivatives`]
//! for deep saturation), accurate to a few f32 ulp (unit tests below measure it against `f64`).

/// Batch lanes per cell.
pub(crate) const LANES: usize = 8;

/// One cell: the values of one neuron (or edge, or input) for 8 consecutive batch lanes.
pub(crate) type L8 = [f32; LANES];

pub(crate) const ZERO8: L8 = [0.0; LANES];

/// Zero-initialised cell storage whose first cell starts on a 64-byte (cache line) boundary.
///
/// Backed by a plain `Vec<f32>` (zeroed by the allocator, pages are only touched when first
/// written, so RSS follows what is actually used) plus an element offset chosen from the
/// allocation's address; viewed as cells through `as_chunks`. Safe code only.
pub(crate) struct LaneBuf {
    data: Vec<f32>,
    off: usize,
    cells: usize,
}

impl LaneBuf {
    pub(crate) fn zeroed(cells: usize) -> Self {
        // 16 spare floats: enough to slide the start to the next 64-byte boundary.
        let data = vec![0.0f32; cells * LANES + 16];
        let addr = data.as_ptr() as usize;
        let off = ((64 - addr % 64) % 64) / std::mem::size_of::<f32>();
        LaneBuf { data, off, cells }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.cells
    }

    #[inline]
    pub(crate) fn cells(&self) -> &[L8] {
        self.data[self.off..self.off + self.cells * LANES]
            .as_chunks::<LANES>()
            .0
    }

    #[inline]
    pub(crate) fn cells_mut(&mut self) -> &mut [L8] {
        let (off, n) = (self.off, self.cells);
        self.data[off..off + n * LANES].as_chunks_mut::<LANES>().0
    }

    /// Bytes of heap this buffer holds.
    pub(crate) fn bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// `a * b + c`: a fused multiply-add when the build has FMA hardware enabled (`-C
/// target-cpu=x86-64-v3` or similar -- never in this workspace's committed configuration), a plain
/// multiply and add otherwise (`f32::mul_add` without the target feature is a libm call, far
/// slower). The two differ in the last bit, so results are bitwise reproducible only within one
/// build configuration (they agree to f32 reduction-order tolerance across them).
#[inline(always)]
pub(crate) fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_feature = "fma")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_feature = "fma"))]
    {
        a * b + c
    }
}

/// Sum of the 8 lanes in a fixed pairwise order (deterministic, independent of the compiler).
#[inline(always)]
pub(crate) fn hsum8(a: &L8) -> f32 {
    let s4 = [a[0] + a[4], a[1] + a[5], a[2] + a[6], a[3] + a[7]];
    let s2 = [s4[0] + s4[2], s4[1] + s4[3]];
    s2[0] + s2[1]
}

const LOG2E: f32 = std::f32::consts::LOG2_E;
// ln 2 split in two (Cody-Waite): `LN2_HI` has few mantissa bits so `k * LN2_HI` is exact.
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
/// `1.5 * 2^23`: adding it rounds to the nearest integer and leaves that integer in the low
/// mantissa bits, so neither `round()` (libm call without SSE4.1) nor a float-to-int convert is
/// needed.
const MAGIC: f32 = 12_582_912.0;

/// `e^y` for `y` in `[-80, 80]`, relative error below ~2e-7 (Taylor degree 7 on the reduced
/// argument, `|r| <= ln2/2`, truncation ~5e-9). Callers clamp `y`.
#[inline(always)]
fn exp1(y: f32) -> f32 {
    let t = y * LOG2E + MAGIC;
    let kf = t - MAGIC;
    let k = t.to_bits() as i32 - MAGIC.to_bits() as i32;
    let r = (y - kf * LN2_HI) - kf * LN2_LO;
    let mut p = 1.0 / 5040.0f32;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    p * f32::from_bits(((k + 127) as u32) << 23)
}

/// `tanh(x)` for `0 <= x <= 7.9053`: the odd 13/6 rational minimax approximation used by Eigen and
/// TensorFlow for single-precision `tanh` (max relative error `1.3e-7` in exact arithmetic, a few
/// f32 ulp as evaluated here), ~25 flops and one division instead of an `exp`-based form.
#[inline(always)]
fn tanh_rational(x: f32) -> f32 {
    const A: [f32; 7] = [
        4.893_524_6e-3,
        6.372_619_3e-4,
        1.485_722_3e-5,
        5.122_297e-8,
        -8.604_672e-11,
        2.000_188e-13,
        -2.760_768_5e-16,
    ];
    const B: [f32; 4] = [4.893_525e-3, 2.268_434_6e-3, 1.185_347e-4, 1.198_258_4e-6];
    let x2 = x * x;
    let mut p = A[6];
    p = p * x2 + A[5];
    p = p * x2 + A[4];
    p = p * x2 + A[3];
    p = p * x2 + A[2];
    p = p * x2 + A[1];
    p = p * x2 + A[0];
    let mut q = B[3];
    q = q * x2 + B[2];
    q = q * x2 + B[1];
    q = q * x2 + B[0];
    (x * p) / q
}

/// Above this `x = V / r_max` the cheap `f'` form `(1 - t)(1 + t)` loses relative accuracy (it
/// subtracts nearly equal numbers) and [`sech2_pos`] is used instead.
const SECH2_SLOW_ABOVE: f32 = 1.5;
const TANH_X_MAX: f32 = 7.905_311;

/// `sech^2(x) = 1/cosh^2(x) = 4q/(1+q)^2` with `q = e^{-2x}`, for `x >= 0`: no subtraction of
/// nearly equal numbers anywhere (the same property the scalar `1/cosh^2` of
/// [`crate::activation::activation_derivative`] was chosen for, review F7a of task 7.2).
#[inline(always)]
fn sech2_pos(x: f32) -> f32 {
    let xc = if x < 20.0 { x } else { 20.0 };
    let q = exp1(-2.0 * xc);
    let d = 1.0 + q;
    4.0 * q / (d * d)
}

/// `f(V) = r_max * tanh(relu(V) / r_max)` for a row of values: exactly `0` for `V <= 0` (the relu
/// kink's subgradient convention of the per-sequence path), NaN for NaN. Returns whether any value
/// is deep enough in saturation (`V / r_max > 1.5`) for the derivative recomputed from the rate
/// ([`derivative_from_rate_row`]) to be inaccurate, i.e. whether the caller has to record the
/// exact derivative of those entries ([`saturated_derivatives`]).
///
/// Written over flat `f32` slices with one straight-line body so the loop vectorizer takes it;
/// the form of the selects matters (a nested `if v <= 0 {0} else if v.is_nan() {v} else {x}` makes
/// LLVM give up and run the whole loop scalar).
#[inline(always)]
pub(crate) fn activation_row(v: &[f32], r: &mut [f32], r_max: f32, inv_r_max: f32) -> bool {
    let mut sat = 0u32;
    for (&vi, ro) in v.iter().zip(r.iter_mut()) {
        let x = vi * inv_r_max;
        let xs = if x > 0.0 { x } else { 0.0 };
        let xc = if xs < TANH_X_MAX { xs } else { TANH_X_MAX };
        let t = tanh_rational(xc);
        // `fallback`: 0 for V <= 0, V itself (NaN) otherwise.
        let fallback = if vi <= 0.0 { 0.0 } else { vi };
        *ro = if vi > 0.0 { r_max * t } else { fallback };
        sat |= u32::from(xs > SECH2_SLOW_ABOVE);
    }
    sat != 0
}

/// `f'(V)` recomputed from the recorded rate `r = f(V)` instead of being recorded itself (task
/// 7.2c: one of three recorded arrays less): `1 - t^2 = (1 - t)(1 + t)` with `t = r / r_max`, and
/// exactly `0` where `r == 0` (`V <= 0`; a positive `V` whose rate underflows to zero has
/// `f' = 1` in exact arithmetic and `0` here, below anything the model feeds), NaN for NaN.
/// Absolute error ~4e-7 (the rational `tanh` of the forward pass plus the rounding of `r_max * t`
/// and of `r * (1 / r_max)`); relative `~3e-6` while `V / r_max <= 1.5`. Entries beyond that are
/// patched with the exact [`saturated_derivatives`] recorded by the forward pass.
#[inline(always)]
pub(crate) fn derivative_from_rate_row(r: &[f32], d: &mut [f32], inv_r_max: f32) {
    for (&ri, dd) in r.iter().zip(d.iter_mut()) {
        let t = ri * inv_r_max;
        *dd = if ri > 0.0 { (1.0 - t) * (1.0 + t) } else { ri };
    }
}

/// The deep-saturation fix-up, recorded by the forward pass: the exact `f'` of every entry of
/// `v` above the threshold (`V / r_max > 1.5`, V > 15 at the default `r_max`), via [`sech2_pos`]
/// -- no subtraction of nearly equal numbers, so the relative accuracy of the *small* derivatives
/// of saturated neurons (the property review F7a of task 7.2 asked for) survives even though
/// the dense `f'` array is gone. Appends `(base + index, f')` to `out`. Out of line and cold: it
/// only runs for rows with a saturated entry.
#[cold]
#[inline(never)]
pub(crate) fn saturated_derivatives(v: &[f32], base: u32, inv_r_max: f32, out: &mut Vec<(u32, f32)>) {
    for (i, &vi) in v.iter().enumerate() {
        let x = vi * inv_r_max;
        if x > SECH2_SLOW_ABOVE {
            out.push((base + i as u32, sech2_pos(x)));
        }
    }
}

/// `acc += coeff * x`, lane-wise. Out of line like the other lane reductions: inlined into the
/// per-edge loops of the backward kernels the 8-lane update came out as eight scalar mul/add pairs.
#[inline(never)]
pub(crate) fn axpy8(acc: &mut L8, coeff: f32, x: &L8) {
    for l in 0..LANES {
        acc[l] = fma(coeff, x[l], acc[l]);
    }
}

/// `sum over cells` of a row, lane-wise (kept out of line: loop-carried lane accumulators
/// vectorise reliably only behind a call boundary, see `gather_acc` in the kernels).
#[inline(never)]
pub(crate) fn lane_sum(row: &[L8]) -> L8 {
    let mut acc = ZERO8;
    for c in row {
        for l in 0..LANES {
            acc[l] += c[l];
        }
    }
    acc
}

/// For each of `K` cell rows `a[m]` (same length as `b`): `sum over cells` of `a[m][c] * b[c]`,
/// lane-wise. The `K` independent accumulator chains hide the add latency, and each cell of `b`
/// is loaded once for all of them.
#[inline(never)]
pub(crate) fn lane_dot_n<const K: usize>(a: [&[L8]; K], b: &[L8]) -> [L8; K] {
    for row in &a {
        assert_eq!(row.len(), b.len());
    }
    let mut acc = [ZERO8; K];
    for (c, y) in b.iter().enumerate() {
        for m in 0..K {
            let x = &a[m][c];
            for l in 0..LANES {
                acc[m][l] += x[l] * y[l];
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        if b == 0.0 { a.abs() } else { ((a - b) / b).abs() }
    }

    #[test]
    fn lane_buf_is_cache_line_aligned_and_zeroed() {
        for cells in [0usize, 1, 3, 8, 1000] {
            let mut b = LaneBuf::zeroed(cells);
            assert_eq!(b.len(), cells);
            assert!(b.cells().iter().all(|c| *c == ZERO8));
            if cells > 0 {
                assert_eq!(b.cells().as_ptr() as usize % 64, 0);
                b.cells_mut()[cells - 1] = [1.0; 8];
                assert_eq!(b.cells()[cells - 1], [1.0; 8]);
            }
        }
    }

    #[test]
    fn hsum8_sums_all_lanes() {
        assert_eq!(hsum8(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), 36.0);
    }

    /// `(f, f')` the way the engine produces them: `f` in the forward pass, `f'` recomputed from
    /// it in the backward pass and patched with the saturated entries the forward pass recorded.
    fn pair(vs: &[f32], r_max: f32) -> (Vec<f32>, Vec<f32>) {
        let inv = 1.0 / r_max;
        let (mut r, mut d) = (vec![0.0; vs.len()], vec![0.0; vs.len()]);
        let mut patches = Vec::new();
        if activation_row(vs, &mut r, r_max, inv) {
            saturated_derivatives(vs, 0, inv, &mut patches);
        }
        derivative_from_rate_row(&r, &mut d, inv);
        for (pos, v) in patches {
            d[pos as usize] = v;
        }
        (r, d)
    }

    /// `f(V)` and `f'(V)` against the `f64` truth over a dense grid (both signs, the kink, tiny
    /// positive values, deep saturation): relative error stays within a few f32 ulp.
    #[test]
    fn activation_pair_matches_f64_truth() {
        for r_max in [10.0f32, 1.0, 3.0] {
            let (mut worst_r, mut worst_d) = (0.0f64, 0.0f64);
            let mut vs: Vec<f32> = (-200..=8000).map(|i| i as f32 * 0.025 * r_max / 10.0).collect();
            for e in -12..=2 {
                vs.push(10f32.powi(e));
            }
            // (subnormal inputs are left out: the rational's coefficients underflow there, and
            // nothing in the model feeds them -- the backward pass flushes subnormal adjoints.)
            vs.extend([0.0, 1e-30, 1e5, 1e30]);
            // Rows of mixed content, so the saturated fix-up is exercised next to ordinary values.
            for chunk in vs.chunks(64) {
                let (r, d) = pair(chunk, r_max);
                for (l, &v) in chunk.iter().enumerate() {
                    if v <= 0.0 {
                        assert_eq!((r[l], d[l]), (0.0, 0.0), "v={v}");
                        continue;
                    }
                    let x = f64::from(v) / f64::from(r_max);
                    let want_r = f64::from(r_max) * x.tanh();
                    let er = rel(f64::from(r[l]), want_r);
                    worst_r = worst_r.max(er);
                    assert!(er < 4e-7, "r_max={r_max} v={v}: got {} want {want_r} rel {er}", r[l]);
                    let want_d = 1.0 / x.cosh().powi(2);
                    if x > 20.0 || want_d < 1e-30 {
                        assert!(f64::from(d[l]) < 1e-15, "v={v}: d {}", d[l]);
                        continue;
                    }
                    let ed = rel(f64::from(d[l]), want_d);
                    worst_d = worst_d.max(ed);
                    assert!(ed < 3e-6, "r_max={r_max} v={v}: d got {} want {want_d} rel {ed}", d[l]);
                }
            }
            eprintln!("r_max={r_max}: worst relative error vs f64: f {worst_r:.3e}, f' {worst_d:.3e}");
        }
    }

    #[test]
    fn nan_propagates_and_infinity_saturates() {
        let r_max = 10.0f32;
        let cell = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30, -1e30, 0.0, -0.0, 3.0];
        let (r, d) = pair(&cell, r_max);
        assert!(r[0].is_nan() && d[0].is_nan());
        assert!((r[1] - r_max).abs() < 3e-6 && (r[3] - r_max).abs() < 3e-6);
        assert!(d[1] < 1e-15 && d[3] < 1e-15);
        for l in [2, 4, 5, 6] {
            assert_eq!((r[l], d[l]), (0.0, 0.0), "lane {l}");
        }
        assert!(r[7] > 0.0 && d[7] > 0.0);
    }

    #[test]
    fn lane_reductions_sum_cells_lane_wise() {
        let a = [[1.0f32; 8], [2.0; 8], [3.0; 8]];
        let b = [[0.5f32; 8], [1.0; 8], [2.0; 8]];
        assert_eq!(lane_sum(&a), [6.0; 8]);
        assert_eq!(lane_dot_n::<1>([&a], &b), [[8.5; 8]]);
        assert_eq!(lane_dot_n::<2>([&a, &b], &b), [[8.5; 8], [5.25; 8]]);
        assert_eq!(lane_sum(&[]), ZERO8);
    }

    /// The vector forms must agree with the scalar functions the per-sequence path uses to within
    /// f32 rounding, on the values that actually occur.
    #[test]
    fn agrees_with_the_scalar_reference_functions() {
        use crate::activation::{activation, activation_derivative};
        let r_max = 10.0f32;
        for i in 0..2000 {
            let v = (i as f32 - 300.0) * 0.037;
            let (a8, d8) = pair(&[v], r_max);
            let (a, d) = (a8[0], d8[0]);
            let ra = activation(v, r_max);
            let rd = activation_derivative(v, r_max);
            assert!((a - ra).abs() <= 1e-6 * ra.abs().max(1e-3), "v={v}: {a} vs {ra}");
            // f' is `(1-t)(1+t)`: absolute error ~4e-7, see `derivative_from_rate_row`.
            assert!((d - rd).abs() <= 5e-7 + 1e-6 * rd.abs(), "v={v}: {d} vs {rd}");
        }
    }
}
