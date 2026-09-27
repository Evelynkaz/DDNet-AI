//! The one genuinely hot inner loop: gather-accumulate over one CSR row. Kept in its own module so
//! it's easy to benchmark/reason about in isolation from `FlyState::step_decision`'s bookkeeping.
//!
//! Acceptance criterion 5: "start with plain, cache-friendly scalar code (8-way unrolled
//! gather-accumulate)"; physics bit-exactness rules don't apply here (FMA would be allowed), but
//! this deliberately stays on that plain scalar code — two things were tried and rejected, both
//! documented here so they aren't tried again without re-reading this:
//! - **`f32::mul_add`** without a guaranteed `fma` target feature lowers `llvm.fma.f32` to a
//!   software `fmaf` call, which is much *slower* than a plain multiply-then-add. This workspace
//!   must not set `-C target-feature=+fma`/`target-cpu` globally (other crates depend on it
//!   staying unset — acceptance criterion 5), so plain `+=`/`*` (native `mulss`/`addss`, always
//!   fast) is what's used.
//! - **A runtime-dispatched `#[target_feature(enable = "fma")]` fast path** (`is_x86_feature_
//!   detected!` once per decision, then an `unsafe` call into an FMA-enabled variant of this
//!   function) was implemented and measured on the real M graph's `ddnet-ai fly bench` duty-cycle
//!   loop: it made things *worse* (median ~4.3ms vs ~3.7ms plain scalar), not better. The likely
//!   cause: a `#[target_feature]` function is a real, non-inlinable call boundary on this Rust
//!   edition, and this function is called once per CSR row (tens of thousands of times per
//!   decision on the real graphs, with typically only ~30 edges per row) — the lost inlining (and
//!   the loss of loop-level optimization across the row-loop/kernel-call boundary) outweighed
//!   whatever the fused multiply-add itself saved. This matches `docs/research/fly-lit.md`/
//!   `rust-stack.md`'s own phase-0 finding ("manual SIMD not needed for B=1" — AVX2 gather was
//!   *also* slower than plain scalar on this exact hardware/workload shape). So: plain scalar,
//!   `#[inline]`, no `unsafe` in this crate at all — acceptance criterion 5's stated preference.
//!
//! Review round 1 (F3) found two more real wins, both applied below:
//! - **`u16` presynaptic indices** when the graph has `<= 65536` neurons (both real S/M graphs
//!   qualify) instead of the `.flyg` format's native `u32` — half the bytes moved per edge for the
//!   same gather. [`FlyModel`](crate::model::FlyModel) builds a `u16` copy at load time
//!   (`narrow_pre_index`) precisely so this function can be generic over the index width via
//!   [`PreIndex`] and the caller picks the narrow path once per model, not per row.
//! - **`as_chunks::<8>()`** (`chunks_exact(8)`'s const-generic sibling — clippy's
//!   `chunks_exact_to_as_chunks` lint asks for it once the chunk size is a compile-time constant,
//!   as it is here) instead of manual `base + k` indexing: giving the compiler a
//!   fixed-size, statically-in-bounds chunk (rather than an index arithmetic expression it has to
//!   re-prove is in-bounds) measurably helped autovectorization/codegen in the reviewer's own
//!   probe (`bin/parts.rs`: M 2.56ms -> 2.11ms, S 0.29ms -> 0.17ms, combined with `u16`).

/// A presynaptic-partner index narrow enough to gather with less memory traffic than the `.flyg`
/// format's native `u32` — implemented for `u16` (the real S/M graphs both have `<= 65536`
/// neurons, see [`crate::model::FlyModel::new`]) and `u32` (the always-correct fallback for any
/// hypothetical larger graph).
pub(crate) trait PreIndex: Copy {
    fn idx(self) -> usize;
}

impl PreIndex for u16 {
    #[inline(always)]
    fn idx(self) -> usize {
        self as usize
    }
}

impl PreIndex for u32 {
    #[inline(always)]
    fn idx(self) -> usize {
        self as usize
    }
}

/// `Σ_k weights[k] * r[pre_index[k]]` for one CSR row, 8 independent accumulators over
/// `as_chunks::<8>()` (enables out-of-order/SIMD execution without needing any explicit
/// `target_feature` — see the module doc comment for why this stays plain scalar rather than
/// using `mul_add`/FMA, and for the `as_chunks`/`u16` rationale). `as_chunks` over
/// `chunks_exact(8).remainder()`: same fixed-size-chunk-plus-remainder split, but as a compile-time
/// constant, which is what clippy's `chunks_exact_to_as_chunks` lint asks for once the chunk size
/// is a literal.
///
/// `pre_index` and `weights` must be the same length; every `pre_index` entry must be
/// `< r.len()` — guaranteed by `ddai_flyg::validate` (for the `u32` instantiation; the `u16`
/// instantiation's indices are a narrowing copy of the same, validated values — see
/// `FlyModel::new`), so a violation here panics on the out-of-bounds index rather than reading
/// out of bounds (this function uses only safe indexing).
#[inline]
pub(crate) fn gather_accumulate<I: PreIndex>(pre_index: &[I], weights: &[f32], r: &[f32]) -> f32 {
    debug_assert_eq!(pre_index.len(), weights.len());
    let (idx_chunks, idx_remainder) = pre_index.as_chunks::<8>();
    let (w_chunks, w_remainder) = weights.as_chunks::<8>();

    let mut acc = [0.0f32; 8];
    for (idx_chunk, w_chunk) in idx_chunks.iter().zip(w_chunks) {
        for k in 0..8 {
            acc[k] += w_chunk[k] * r[idx_chunk[k].idx()];
        }
    }
    let mut total = acc.iter().sum::<f32>();
    for (&idx, &w) in idx_remainder.iter().zip(w_remainder) {
        total += w * r[idx.idx()];
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_f64<I: PreIndex>(pre_index: &[I], weights: &[f32], r: &[f32]) -> f64 {
        pre_index
            .iter()
            .zip(weights)
            .map(|(&idx, &w)| f64::from(w) * f64::from(r[idx.idx()]))
            .sum()
    }

    #[test]
    fn empty_row_sums_to_zero() {
        assert_eq!(gather_accumulate::<u32>(&[], &[], &[1.0, 2.0]), 0.0);
        assert_eq!(gather_accumulate::<u16>(&[], &[], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn u32_matches_naive_f64_reference_for_various_row_lengths() {
        for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 40] {
            let r: Vec<f32> = (0..len as u32 + 1).map(|i| i as f32 * 0.1).collect();
            let pre_index: Vec<u32> = (0..len as u32).collect();
            let weights: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0) * 0.37).collect();
            let got = gather_accumulate(&pre_index, &weights, &r);
            let want = reference_f64(&pre_index, &weights, &r);
            let diff = (f64::from(got) - want).abs();
            assert!(diff < 1e-4, "len={len}: got={got} want={want} diff={diff}");
        }
    }

    #[test]
    fn u16_matches_naive_f64_reference_for_various_row_lengths() {
        for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 40] {
            let r: Vec<f32> = (0..len as u32 + 1).map(|i| i as f32 * 0.1).collect();
            let pre_index: Vec<u16> = (0..len as u16).collect();
            let weights: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0) * 0.37).collect();
            let got = gather_accumulate(&pre_index, &weights, &r);
            let want = reference_f64(&pre_index, &weights, &r);
            let diff = (f64::from(got) - want).abs();
            assert!(diff < 1e-4, "len={len}: got={got} want={want} diff={diff}");
        }
    }

    #[test]
    fn u16_and_u32_agree_on_the_same_data() {
        let r: Vec<f32> = (0..20).map(|i| (i as f32) * 0.37 - 2.0).collect();
        let pre_index_u32: Vec<u32> = vec![0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 1, 2];
        let pre_index_u16: Vec<u16> = pre_index_u32.iter().map(|&x| x as u16).collect();
        let weights: Vec<f32> = (0..pre_index_u32.len()).map(|i| (i as f32 + 1.0) * 0.11).collect();
        let via_u32 = gather_accumulate(&pre_index_u32, &weights, &r);
        let via_u16 = gather_accumulate(&pre_index_u16, &weights, &r);
        assert_eq!(via_u32, via_u16);
    }
}
