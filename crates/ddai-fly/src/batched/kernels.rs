//! The hot loops of the batched backend, one chunk of neuron rows at a time: a forward substep
//! ([`fwd_chunk`]), a backward substep ([`bwd_chunk`]) and the initial-state pass ([`init_chunk`]).
//!
//! State is `[N][NB]` cells of 8 lanes ([`L8`]): row `i` of an array occupies cells
//! `i*NB .. (i+1)*NB`. A chunk task walks its rows once per **lane group** of `G` consecutive
//! cells (`G = 8`, then 4, 2, 1 for the remainder of `NB`): group-outer, row-inner. Up to 64 lanes
//! (`NB = 8`) that is a single pass per chunk; wider batches take several passes, each gathering
//! only `G` cells of every presynaptic row (the row's edge list, ~30 edges, is re-read from L1 for
//! every pass). Narrower groups (`G = 4`, `G = 2`: smaller L2 working set per pass) were measured
//! on M and were slower here: more passes cost more than the extra L2 hits save. Per neuron the
//! work has two shapes:
//! - the **gathers** (forward: over the in-edges, backward: over the out-edges): `G`
//!   accumulators stay in registers while the loop walks the edges, each edge costing one
//!   index/weight load (shared by all the lanes) plus `G` cell loads and `G*8` multiply-adds
//!   (fused on builds with FMA hardware, see [`fma`]);
//! - the **elementwise** parts (`V` update, `f(V)`, `f'(V)`, the chain rule): straight-line loops
//!   over the group's `G*8` values as a flat `f32` slice.
//!
//! No `unsafe`, no intrinsics: loops that LLVM turns into vector code (SSE2 on the default target,
//! AVX2 under `-C target-cpu=x86-64-v3`; see the crate README for the assembly check). Several
//! functions below are `#[inline(never)]` on purpose: see [`gather_acc`].
//!
//! ## What the backward pass walks
//! `dL/dalpha` of an edge `j -> i` at substep `s` is `coeff * delta_s[i] * r_{s-1}[j]` summed over
//! lanes (`delta_s = dL/dV_inf` of substep `s`, `r_{s-1}` the rate that drove it). The transposed
//! gather that produces `dL/dr_l[j]` from `delta_{l+1}` already visits exactly those `delta` rows
//! while sitting on neuron `j`, whose `r_l[j]` is at hand, so the `dL/dalpha` of substep `l + 1`
//! is accumulated right there (the rows are L1-hot from the gather): the backward pass never does
//! the second random gather of presynaptic rates a post-major `dL/dalpha` pass would need. The
//! `dL/dalpha` of substep 0 (against `f(V_init)`) is the initial-state pass's job.

use super::lanes::{
    L8, LANES, ZERO8, activation_row, axpy8, derivative_from_rate_row, fma, hsum8, lane_dot_n, lane_sum,
    saturated_derivatives,
};
use super::plan::{BatchedPlan, NO_SLOT};
use crate::kernel::PreIndex;

/// Cells per lane group of the main loop: eight 8-lane cells = 64 lanes, i.e. eight ymm
/// accumulators on AVX2 (8 independent FMA chains, enough to cover the latency) and sixteen xmm on
/// SSE2 (the register allocator shuffles accumulators between registers there, yet 8 used ~12%
/// less CPU time per step than 4 on M, and 4 less than 2).
const G_MAIN: usize = 8;

/// Magnitude below which a backward adjoint is flushed to zero. Adjoints decay geometrically
/// through the leaky integrators; left alone they end up in the subnormal range, where every
/// vector instruction touching them takes a microcode assist (~100x slower). The threshold is
/// ~7 orders of magnitude under f32 noise relative to any gradient that matters.
const FLUSH_BELOW: f32 = 1e-30;

/// `out[j][l] = sum_k w[k] * src[idx[k] * nb + g0 + j][l]`: the gather-accumulate over one
/// neuron's edge list for `G` consecutive cells, shared by the forward gather (`idx` = presynaptic
/// neurons, `src` = `r`) and the backward transposed gather (`idx` = postsynaptic neurons, `src`
/// = `dL/dV_inf`).
///
/// Deliberately **not inlined**: the accumulators are loop-carried registers, and LLVM's SLP
/// vectorizer only turns them into vector registers when their consumers are vectorisable too.
/// Inlined into the surrounding per-row code (with its input/tap branches) it kept `8 * G` scalar
/// accumulators and ran at scalar speed; behind a call boundary the loop is vectorised and the
/// result travels through memory (a few L1 loads per row, noise next to ~30 edges).
#[inline(never)]
fn gather_acc<I: PreIndex + Sync, const G: usize>(
    idx: &[I],
    w: &[f32],
    src: &[L8],
    nb: usize,
    g0: usize,
    out: &mut [L8; G],
) {
    let mut acc = [ZERO8; G];
    for (&p, &wv) in idx.iter().zip(w) {
        let base = p.idx() * nb + g0;
        let cells = &src[base..base + G];
        for j in 0..G {
            for l in 0..LANES {
                acc[j][l] = fma(wv, cells[j][l], acc[j][l]);
            }
        }
    }
    *out = acc;
}

/// The exact `f'` of the deeply saturated entries (`V / r_max > 1.5`) of one chunk, recorded by the
/// forward pass in place of the dense `f'(V)` array the first version of the backend kept (task
/// 7.2c). Everywhere else the backward pass recomputes `f'` from the recorded rate `r`
/// ([`derivative_from_rate_row`]); these few entries, whose small derivatives that formula would
/// lose to cancellation, are patched back in. Typical data has none (V > 15 at the default
/// `r_max`).
///
/// One *slot* per recorded `r` slot of the loaded segment (slot 0 = the segment's start state,
/// slot `ls + 1` = substep `ls`). A slot is stored **sparsely** -- `(key, f')` entries in `vals`,
/// `ends[k]` the end of slot `k`'s entries -- while at most half of its lanes are patched, and
/// **densely** otherwise (the whole slot's `f'` in `dense`, `dense_ends[k]` its end; recomputed
/// from `r` and overwritten with the patches when the slot closes). So a slot never takes more than
/// 4 bytes per lane -- what the dense array of the first version cost -- and the worst case
/// (everything saturated) is bounded and counted by `estimate_memory`/`memory_cap_bytes`.
///
/// Sparse entries of a slot come in the order the chunk's passes visit them -- lane groups outer,
/// rows inner, cells then lanes -- and the backward pass visits them in the same order, so it
/// consumes them with a cursor (see [`apply_patches`]). An entry's key is its flat lane index in
/// the chunk's `[row][cell][lane]` block of that slot, which is also the index into a dense slot.
#[derive(Debug, Default)]
pub(crate) struct PatchStore {
    pub vals: Vec<(u32, f32)>,
    dense: Vec<f32>,
    ends: Vec<u32>,
    dense_ends: Vec<u32>,
}

/// What the recorded slot of one chunk holds, see [`PatchStore`].
#[derive(Debug, Clone, Copy)]
pub(crate) enum SlotPatches<'a> {
    Sparse(&'a [(u32, f32)]),
    Dense(&'a [f32]),
}

impl PatchStore {
    pub(crate) fn clear(&mut self) {
        self.vals.clear();
        self.dense.clear();
        self.ends.clear();
        self.dense_ends.clear();
    }

    /// Closes the slot just recorded (`r` is the chunk's rate rows of that slot): the sparse
    /// entries are turned into a dense slot when they would take more than 4 bytes per lane.
    fn close_slot(&mut self, r: &[L8], inv_r_max: f32) {
        let start = self.ends.last().map_or(0, |&e| e as usize);
        let lanes = r.len() * LANES;
        if (self.vals.len() - start) * 2 > lanes {
            self.dense.reserve_exact(lanes);
            let d0 = self.dense.len();
            self.dense.resize(d0 + lanes, 0.0);
            let d = &mut self.dense[d0..];
            derivative_from_rate_row(r.as_flattened(), d, inv_r_max);
            for &(pos, val) in &self.vals[start..] {
                d[pos as usize] = val;
            }
            self.vals.truncate(start);
        }
        self.ends
            .push(u32::try_from(self.vals.len()).expect("patch store overflow"));
        self.dense_ends
            .push(u32::try_from(self.dense.len()).expect("patch store overflow"));
    }

    /// The patches of slot `k`.
    pub(crate) fn slot(&self, k: usize) -> SlotPatches<'_> {
        let (lo, hi) = (
            if k == 0 { 0 } else { self.ends[k - 1] as usize },
            self.ends[k] as usize,
        );
        let (dlo, dhi) = (
            if k == 0 { 0 } else { self.dense_ends[k - 1] as usize },
            self.dense_ends[k] as usize,
        );
        if dhi > dlo {
            SlotPatches::Dense(&self.dense[dlo..dhi])
        } else {
            SlotPatches::Sparse(&self.vals[lo..hi])
        }
    }

    /// Heap held (capacities).
    pub(crate) fn bytes(&self) -> usize {
        self.vals.capacity() * std::mem::size_of::<(u32, f32)>()
            + self.dense.capacity() * 4
            + (self.ends.capacity() + self.dense_ends.capacity()) * 4
    }

    /// Gives back what growth left over when the store holds more than `bound_bytes` (the
    /// geometric growth of a `Vec` can overshoot its length by up to 2x).
    pub(crate) fn trim(&mut self, bound_bytes: usize) {
        if self.bytes() > bound_bytes {
            self.vals.shrink_to_fit();
            self.dense.shrink_to_fit();
            self.ends.shrink_to_fit();
            self.dense_ends.shrink_to_fit();
        }
    }
}

/// Overwrites the entries of `d` (a group's flat `f'`, starting at flat lane index `base`) that
/// the next entries of the slot's patch list belong to; `cursor` advances past them. An entry
/// whose key is outside `[base, base + d.len())` belongs to a later group and ends the run.
#[inline(always)]
fn apply_patches(patches: &[(u32, f32)], cursor: &mut usize, base: u32, d: &mut [f32]) {
    let hi = base + d.len() as u32;
    while let Some(&(pos, val)) = patches.get(*cursor) {
        if pos < base || pos >= hi {
            break;
        }
        d[(pos - base) as usize] = val;
        *cursor += 1;
    }
}

/// `f'` of one row's group of lanes `(ri, nb, g0)` for the backward pass: recomputed from the
/// recorded rates `r` and patched with the slot's exact saturated entries, or read from the dense
/// slot.
#[inline(always)]
fn fill_derivative(
    patches: SlotPatches<'_>,
    cursor: &mut usize,
    (ri, nb, g0): (usize, usize, usize),
    r: &[f32],
    d: &mut [f32],
    inv_r_max: f32,
) {
    let base = lane_key(ri, nb, g0);
    match patches {
        SlotPatches::Dense(dense) => d.copy_from_slice(&dense[base as usize..base as usize + d.len()]),
        SlotPatches::Sparse(p) => {
            derivative_from_rate_row(r, d, inv_r_max);
            apply_patches(p, cursor, base, d);
        }
    }
}

/// Flat lane index of cell `g0` of chunk-local row `ri`.
#[inline(always)]
fn lane_key(ri: usize, nb: usize, g0: usize) -> u32 {
    ((ri * nb + g0) * LANES) as u32
}

/// Per-call constants of the model, shared by every chunk task of a region.
pub(crate) struct Ctx<'a, I: PreIndex + Sync> {
    pub plan: &'a BatchedPlan,
    pub nb: usize,
    pub row_start: &'a [u32],
    /// Post-major presynaptic index (forward gather).
    pub pre_index: &'a [I],
    /// Pre-major postsynaptic index (backward transposed gather).
    pub post_of: &'a [I],
    /// `w_ij`, post-major (forward), in the engine's neuron order.
    pub weights: &'a [f32],
    /// `w_ij`, pre-major (backward).
    pub weights_t: &'a [f32],
    pub bias: &'a [f32],
    pub decay: &'a [f32],
    pub r_max: f32,
    pub inv_r_max: f32,
    /// Whether this call's regions run on the rayon pool (see `plan::DEFAULT_PAR_MIN_EDGE_CELLS`).
    pub par: bool,
}

/// Runs `$body` once per lane group of `$nb` cells, with `$g0` bound to the group's first cell and
/// `$G` to its width as a const: groups of [`G_MAIN`] cells while they fit, then one group each of
/// 4, 2 and 1 for the remainder.
macro_rules! for_each_group {
    ($nb:expr, |$g0:ident, $G:ident| $body:expr) => {{
        let nb: usize = $nb;
        let mut $g0 = 0usize;
        while $g0 + G_MAIN <= nb {
            const $G: usize = G_MAIN;
            $body;
            $g0 += G_MAIN;
        }
        if $g0 + 4 <= nb {
            const $G: usize = 4;
            $body;
            $g0 += 4;
        }
        if $g0 + 2 <= nb {
            const $G: usize = 2;
            $body;
            $g0 += 2;
        }
        if $g0 < nb {
            const $G: usize = 1;
            $body;
        }
    }};
}

// ---------------------------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------------------------

/// What one forward chunk task owns: its rows of the membrane state (updated in place) and of the
/// substep's recorded `r` and `X = V_inf - V_prev`, plus the chunk's saturated-derivative store.
pub(crate) struct FwdTask<'a> {
    pub chunk: usize,
    pub v: &'a mut [L8],
    pub r_new: &'a mut [L8],
    pub x_new: &'a mut [L8],
    pub patches: &'a mut PatchStore,
}

/// Read-only inputs of a forward substep.
pub(crate) struct FwdRead<'a> {
    /// `r` that drives this substep, every row (`[N][NB]`).
    pub r_prev: &'a [L8],
    /// The decision's external input, `[num_inputs][NB]`.
    pub inp: &'a [L8],
}

pub(crate) fn fwd_chunk<I: PreIndex + Sync>(ctx: &Ctx<'_, I>, rd: &FwdRead<'_>, mut task: FwdTask<'_>) {
    for_each_group!(ctx.nb, |g0, G| fwd_pass::<I, G>(ctx, rd, &mut task, g0));
    task.patches.close_slot(task.r_new, ctx.inv_r_max);
}

/// One forward pass over the chunk's rows for the lane group `g0..g0+G`.
fn fwd_pass<I: PreIndex + Sync, const G: usize>(ctx: &Ctx<'_, I>, rd: &FwdRead<'_>, task: &mut FwdTask<'_>, g0: usize) {
    let ch = &ctx.plan.chunks[task.chunk];
    let nb = ctx.nb;
    let zero_in = [ZERO8; G];
    for (ri, i) in (ch.r0..ch.r1).enumerate() {
        let (s, e) = (ctx.row_start[i] as usize, ctx.row_start[i + 1] as usize);
        let mut acc = [ZERO8; G];
        gather_acc::<I, G>(&ctx.pre_index[s..e], &ctx.weights[s..e], rd.r_prev, nb, g0, &mut acc);
        let slot = ctx.plan.in_slot[i];
        let inp: &[L8] = if slot == NO_SLOT {
            &zero_in
        } else {
            &rd.inp[slot as usize * nb + g0..slot as usize * nb + g0 + G]
        };
        let cells = ri * nb + g0..ri * nb + g0 + G;
        let saturated = fwd_row_elementwise(
            acc.as_flattened(),
            inp.as_flattened(),
            task.v[cells.clone()].as_flattened_mut(),
            task.x_new[cells.clone()].as_flattened_mut(),
            task.r_new[cells.clone()].as_flattened_mut(),
            (ctx.bias[i], ctx.decay[i]),
            (ctx.r_max, ctx.inv_r_max),
        );
        if saturated {
            saturated_derivatives(
                task.v[cells].as_flattened(),
                lane_key(ri, nb, g0),
                ctx.inv_r_max,
                &mut task.patches.vals,
            );
        }
    }
}

/// What one start-of-segment activation task owns (`r = f(V)` of the segment's start state, the
/// recording's slot 0).
pub(crate) struct ActTask<'a> {
    pub chunk: usize,
    pub v: &'a [L8],
    pub r: &'a mut [L8],
    pub patches: &'a mut PatchStore,
}

/// `r = f(V)` of a chunk's rows, recording the saturated derivatives like a forward substep.
pub(crate) fn act_chunk<I: PreIndex + Sync>(ctx: &Ctx<'_, I>, task: ActTask<'_>) {
    let nb = ctx.nb;
    let rows = ctx.plan.chunks[task.chunk].r1 - ctx.plan.chunks[task.chunk].r0;
    for_each_group!(nb, |g0, G| {
        for ri in 0..rows {
            let cells = ri * nb + g0..ri * nb + g0 + G;
            let v = task.v[cells.clone()].as_flattened();
            if activation_row(v, task.r[cells].as_flattened_mut(), ctx.r_max, ctx.inv_r_max) {
                saturated_derivatives(v, lane_key(ri, nb, g0), ctx.inv_r_max, &mut task.patches.vals);
            }
        }
    });
    task.patches.close_slot(task.r, ctx.inv_r_max);
}

/// The elementwise half of a forward substep for one neuron's group of cells: `V_inf = (bias +
/// acc) + input`, `X = V_inf - V`, `V += decay * X`, then `r = f(V)`. Returns whether the group
/// has deeply saturated entries whose exact `f'` the caller has to record.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn fwd_row_elementwise(
    acc: &[f32],
    inp: &[f32],
    v: &mut [f32],
    x_out: &mut [f32],
    r_out: &mut [f32],
    (bias, decay): (f32, f32),
    (r_max, inv_r_max): (f32, f32),
) -> bool {
    // First loop: the membrane update. Second: the activation (a separate loop keeps each body
    // small enough for the vectorizer's cost model).
    for (((&a, &x_in), vv), xo) in acc.iter().zip(inp).zip(v.iter_mut()).zip(x_out.iter_mut()) {
        let vinf = (bias + a) + x_in;
        let vp = *vv;
        let xx = vinf - vp;
        *xo = xx;
        *vv = vp + decay * xx;
    }
    activation_row(v, r_out, r_max, inv_r_max)
}

// ---------------------------------------------------------------------------------------------
// backward
// ---------------------------------------------------------------------------------------------

/// What one backward chunk task owns.
pub(crate) struct BwdTask<'a> {
    pub chunk: usize,
    /// `dL/dV` carried to the previous substep (`(1 - decay) * dL/dV_total`), this chunk's rows;
    /// read and overwritten in place.
    pub fut_dv: &'a mut [L8],
    /// This substep's `dL/dV_inf`, this chunk's rows.
    pub delta_cur: &'a mut [L8],
    /// Input-gradient accumulators of this chunk's input neurons: `[k - k0][t_max][NB]`.
    pub gin: &'a mut [L8],
    /// Per-type `dL/db` / `sum(dL/dV_total * X)` partials of this chunk.
    pub acc_b: &'a mut [f32],
    pub acc_t: &'a mut [f32],
    /// `dL/dalpha` lane partials of this chunk's local shared-parameter slots (the lane sum is
    /// taken once, at the very end).
    pub acc_a: &'a mut [L8],
    /// The saturated-derivative patches of this chunk's slot for this substep's `r`.
    pub patches: SlotPatches<'a>,
}

/// One dense `dL/dr` tap of a single batch lane (the activity regulariser's).
pub(crate) struct LaneTap<'a> {
    pub lane: usize,
    pub grad: &'a [f32],
}

/// Read-only inputs of a backward substep.
pub(crate) struct BwdRead<'a> {
    /// `V_inf - V_prev` of this substep, every row.
    pub x_l: &'a [L8],
    /// `r = f(V)` after this substep (the rate that drives the *next* substep), every row; `f'`
    /// is recomputed from it.
    pub r_l: &'a [L8],
    /// `dL/dV_inf` of the substep after this one; `None` for the last substep of the window.
    pub delta_next: Option<&'a [L8]>,
    /// Output taps `[num_outputs][NB]` (already holding `grad_dn`; scaled by 0.5 here): the
    /// decision whose last substep is this one, and the one whose second-to-last it is.
    pub tap_final: Option<&'a [L8]>,
    pub tap_before: Option<&'a [L8]>,
    pub lane_taps: &'a [LaneTap<'a>],
    /// Decision this substep belongs to.
    pub decision: usize,
    pub t_max: usize,
}

/// `dL/dr` of neuron `i` for the cells `g0..g0+G`: the transposed gather of `delta_next` over its
/// out-edges (pre-major positions `ps..pe`) plus the taps.
#[inline(always)]
fn dr_group<I: PreIndex + Sync, const G: usize>(
    ctx: &Ctx<'_, I>,
    rd: &BwdRead<'_>,
    i: usize,
    (ps, pe): (usize, usize),
    g0: usize,
) -> [L8; G] {
    let nb = ctx.nb;
    let mut dr = [ZERO8; G];
    if let Some(dn) = rd.delta_next {
        gather_acc::<I, G>(&ctx.post_of[ps..pe], &ctx.weights_t[ps..pe], dn, nb, g0, &mut dr);
    }
    let out_slot = ctx.plan.out_slot[i];
    if out_slot != NO_SLOT {
        for tap in [rd.tap_final, rd.tap_before].into_iter().flatten() {
            let src = &tap[out_slot as usize * nb + g0..out_slot as usize * nb + g0 + G];
            for j in 0..G {
                for l in 0..LANES {
                    dr[j][l] += 0.5 * src[j][l];
                }
            }
        }
    }
    for t in rd.lane_taps {
        let cell = t.lane / LANES;
        if cell >= g0 && cell < g0 + G {
            dr[cell - g0][t.lane % LANES] += t.grad[ctx.plan.old_of[i] as usize];
        }
    }
    dr
}

/// `dL/dalpha` lane partials of the out-edges (pre-major positions `ps..pe`) of one neuron, for
/// the cells `g0..g0+G`: per edge `acc_a[slot] += coeff * sum_cells(delta[post] * r_row)`,
/// lane-wise. `delta_next` is the substep after the one whose rate `r_row` (row `i` of `r`) is.
/// Edges are taken four at a time: four independent accumulator chains, and `r_row` loaded once.
#[inline(always)]
fn grad_a_group<I: PreIndex + Sync, const G: usize>(
    ctx: &Ctx<'_, I>,
    delta_next: &[L8],
    r_row: &[L8],
    (ps, pe): (usize, usize),
    g0: usize,
    acc_a: &mut [L8],
) {
    if r_row.iter().all(|c| c.iter().all(|&x| x == 0.0)) {
        return; // a neuron silent in every lane contributes nothing
    }
    let nb = ctx.nb;
    let plan = ctx.plan;
    let posts = &ctx.post_of[ps..pe];
    let cells = |post: I| -> &[L8] {
        let b = post.idx() * nb + g0;
        &delta_next[b..b + G]
    };
    let mut k = ps;
    let (quads, remainder) = posts.as_chunks::<4>();
    for q in quads {
        let dots = lane_dot_n::<4>([cells(q[0]), cells(q[1]), cells(q[2]), cells(q[3])], r_row);
        for (m, dot) in dots.iter().enumerate() {
            axpy8(&mut acc_a[plan.local_sid_t[k + m] as usize], plan.coeff_t[k + m], dot);
        }
        k += 4;
    }
    for &post in remainder {
        let [dot] = lane_dot_n::<1>([cells(post)], r_row);
        axpy8(&mut acc_a[plan.local_sid_t[k] as usize], plan.coeff_t[k], &dot);
        k += 1;
    }
}

pub(crate) fn bwd_chunk<I: PreIndex + Sync>(ctx: &Ctx<'_, I>, rd: &BwdRead<'_>, mut task: BwdTask<'_>) {
    let mut cursor = 0usize;
    for_each_group!(ctx.nb, |g0, G| bwd_pass::<I, G>(ctx, rd, &mut task, g0, &mut cursor));
    if let SlotPatches::Sparse(p) = task.patches {
        debug_assert_eq!(cursor, p.len(), "saturated-derivative patches not all consumed");
    }
}

/// One backward pass over the chunk's rows for the lane group `g0..g0+G`; `cursor` walks the
/// chunk's saturated-derivative patches across the passes.
fn bwd_pass<I: PreIndex + Sync, const G: usize>(
    ctx: &Ctx<'_, I>,
    rd: &BwdRead<'_>,
    task: &mut BwdTask<'_>,
    g0: usize,
    cursor: &mut usize,
) {
    let plan = ctx.plan;
    let ch = &plan.chunks[task.chunk];
    let nb = ctx.nb;
    let mut gt_cells = [ZERO8; G];
    let mut d_cells = [ZERO8; G];
    for (ri, i) in (ch.r0..ch.r1).enumerate() {
        let (ps, pe) = (plan.out_row_start[i] as usize, plan.out_row_start[i + 1] as usize);
        let ty = plan.type_of[i] as usize;
        let dec = ctx.decay[i];
        let local = ri * nb + g0..ri * nb + g0 + G;
        let full = i * nb + g0..i * nb + g0 + G;

        let dr = dr_group::<I, G>(ctx, rd, i, (ps, pe), g0);
        fill_derivative(
            task.patches,
            cursor,
            (ri, nb, g0),
            rd.r_l[full.clone()].as_flattened(),
            d_cells.as_flattened_mut(),
            ctx.inv_r_max,
        );
        bwd_row_elementwise(
            dr.as_flattened(),
            d_cells.as_flattened(),
            rd.x_l[full.clone()].as_flattened(),
            task.fut_dv[local.clone()].as_flattened_mut(),
            task.delta_cur[local.clone()].as_flattened_mut(),
            gt_cells.as_flattened_mut(),
            (dec, 1.0 - dec),
        );
        let delta_cells = &task.delta_cur[local];
        let in_slot = plan.in_slot[i];
        if in_slot != NO_SLOT {
            let off = (plan.in_ord[i] as usize - ch.k0) * rd.t_max * nb + rd.decision * nb + g0;
            for (g, d) in task.gin[off..off + G].iter_mut().zip(delta_cells) {
                for l in 0..LANES {
                    g[l] += d[l];
                }
            }
        }
        task.acc_b[ty] += hsum8(&lane_sum(delta_cells));
        task.acc_t[ty] += hsum8(&lane_sum(&gt_cells));

        // dL/dalpha of the next substep: its delta rows (just gathered, L1-hot) against this
        // neuron's rate.
        if let Some(dn) = rd.delta_next {
            grad_a_group::<I, G>(ctx, dn, &rd.r_l[full], (ps, pe), g0, task.acc_a);
        }
    }
}

/// The chain rule through one neuron's substep, over a group of cells:
/// `dV_total = dr * f'(V) + fut`, `delta = decay * dV_total` (`dL/dV_inf`),
/// `fut' = (1 - decay) * dV_total` (carried to the previous substep) and `gt = dV_total * X` (the
/// `dL/dtau` summand), with adjoints below [`FLUSH_BELOW`] flushed to zero.
#[inline(never)]
fn bwd_row_elementwise(
    dr: &[f32],
    sd: &[f32],
    x: &[f32],
    fut: &mut [f32],
    delta: &mut [f32],
    gt: &mut [f32],
    (dec, one_minus): (f32, f32),
) {
    for (((((&r, &d), &xv), f), de), g) in dr
        .iter()
        .zip(sd)
        .zip(x)
        .zip(fut.iter_mut())
        .zip(delta.iter_mut())
        .zip(gt.iter_mut())
    {
        let dvt = r * d + *f;
        let a = dec * dvt;
        let b = one_minus * dvt;
        *de = if a.abs() < FLUSH_BELOW { 0.0 } else { a };
        *f = if b.abs() < FLUSH_BELOW { 0.0 } else { b };
        *g = dvt * xv;
    }
}

// ---------------------------------------------------------------------------------------------
// initial-state pass
// ---------------------------------------------------------------------------------------------

/// What one initial-state chunk task owns.
pub(crate) struct InitTask<'a> {
    pub chunk: usize,
    pub fut_dv: &'a [L8],
    /// `dL/dV_init` rows of this chunk, if the caller wants them.
    pub grad_v_init: Option<&'a mut [L8]>,
    pub acc_a: &'a mut [L8],
    /// The saturated-derivative patches of slot 0 of this chunk (the start state's).
    pub patches: SlotPatches<'a>,
}

/// The pass "before" the first substep: the `dL/dalpha` of substep 0 (its delta against
/// `f(V_init)`, which `rd.r_l` holds here) and, if asked, `dL/dV_init = dL/dr_init * f'(V_init) +
/// fut_dv`, where `dL/dr_init` is the transposed gather of the first substep's `dL/dV_inf` (plus
/// the `substeps == 1` boundary tap).
pub(crate) fn init_chunk<I: PreIndex + Sync>(ctx: &Ctx<'_, I>, rd: &BwdRead<'_>, mut task: InitTask<'_>) {
    let mut cursor = 0usize;
    for_each_group!(ctx.nb, |g0, G| init_pass::<I, G>(ctx, rd, &mut task, g0, &mut cursor));
    if let (SlotPatches::Sparse(p), Some(_)) = (task.patches, task.grad_v_init.as_ref()) {
        debug_assert_eq!(cursor, p.len(), "saturated-derivative patches not all consumed");
    }
}

fn init_pass<I: PreIndex + Sync, const G: usize>(
    ctx: &Ctx<'_, I>,
    rd: &BwdRead<'_>,
    task: &mut InitTask<'_>,
    g0: usize,
    cursor: &mut usize,
) {
    let plan = ctx.plan;
    let ch = &plan.chunks[task.chunk];
    let nb = ctx.nb;
    for (ri, i) in (ch.r0..ch.r1).enumerate() {
        let (ps, pe) = (plan.out_row_start[i] as usize, plan.out_row_start[i + 1] as usize);
        let full = i * nb + g0..i * nb + g0 + G;
        if let Some(gvi) = task.grad_v_init.as_deref_mut() {
            let dr = dr_group::<I, G>(ctx, rd, i, (ps, pe), g0);
            let mut sd = [ZERO8; G];
            fill_derivative(
                task.patches,
                cursor,
                (ri, nb, g0),
                rd.r_l[full.clone()].as_flattened(),
                sd.as_flattened_mut(),
                ctx.inv_r_max,
            );
            let fut = &task.fut_dv[ri * nb + g0..ri * nb + g0 + G];
            for j in 0..G {
                let out = &mut gvi[ri * nb + g0 + j];
                for l in 0..LANES {
                    out[l] = dr[j][l] * sd[j][l] + fut[j][l];
                }
            }
        }
        if let Some(dn) = rd.delta_next {
            grad_a_group::<I, G>(ctx, dn, &rd.r_l[full], (ps, pe), g0, task.acc_a);
        }
    }
}

/// Vectorisation guard. The speed of the whole backend rests on a few small loops being turned
/// into SIMD by the compiler (there is no `unsafe`/intrinsics fallback), and a refactor or a
/// toolchain bump can silently make them scalar -- every functional test stays green and training
/// gets 2-4x slower. These tests time the real kernels against the same multiply-adds written as
/// strictly ordered scalar `f32` reductions (no fast-math, so the compiler may not reorder or
/// vectorise them; several independent chains keep the scalar side throughput-bound rather than
/// latency-bound, i.e. it is about as fast as real scalar code can be), on the same data in the
/// same process. The check is a *ratio* of best-of-N timings, which is robust to a loaded machine
/// (both sides slow down together, and the minimum over repeats discards preemptions).
///
/// They need a build where the loop vectoriser runs. The workspace's `[profile.dev.package.ddai-fly]`
/// sets `opt-level = 3` for this crate, so plain `cargo test` (and CI) runs them; they would fail
/// under `opt-level` 0-1 (nothing is vectorised there), e.g. with a `CARGO_PROFILE_DEV_OPT_LEVEL`
/// override. `cargo test -p ddai-fly --release --lib vectorisation_guard` and
/// `tools/vectorisation/check.sh` (which also reads the disassembly) check the release build.
#[cfg(test)]
mod vectorisation_guard {
    use std::hint::black_box;
    use std::time::Instant;

    use super::*;
    use crate::batched::lanes::{LaneBuf, lane_dot_n};

    /// Minimum ratio of (scalar reduction time) / (kernel time) for kernels without register
    /// pressure. A 4-lane SSE2 kernel measures ~3-4x here, AVX2 more; a scalarised one ~1x or less.
    const MIN_SPEEDUP: f64 = 1.8;

    /// Many short repeats: the minimum of each side then lands in a quiet moment even on a loaded
    /// machine.
    const REPEATS: usize = 31;

    /// Best (minimum) wall time of `f` over `repeats` runs.
    fn best_of(repeats: usize, mut f: impl FnMut()) -> f64 {
        (0..repeats)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Deterministic pseudo-random floats in [-1, 1).
    fn floats(n: usize, mut state: u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn cells(n: usize, seed: u64) -> LaneBuf {
        let mut buf = LaneBuf::zeroed(n);
        for (c, chunk) in buf.cells_mut().iter_mut().zip(floats(n * LANES, seed).chunks(LANES)) {
            c.copy_from_slice(chunk);
        }
        buf
    }

    fn sum_lanes(rows: &[L8]) -> f64 {
        rows.iter().flatten().map(|&x| f64::from(x)).sum()
    }

    /// Runs `measure` (returns `(scalar, kernel)` times) up to 3 times and passes as soon as one
    /// attempt reaches `min_speedup`: a timing test on a shared machine can have a whole attempt
    /// preempted, a lost vectorisation fails every attempt.
    fn check_ratio(name: &str, min_speedup: f64, mut measure: impl FnMut() -> (f64, f64)) {
        let mut best = 0.0f64;
        for attempt in 1..=3 {
            let (scalar, kernel) = measure();
            let ratio = scalar / kernel;
            eprintln!(
                "vectorisation guard: {name} (attempt {attempt}): kernel {kernel:.4} s, scalar {scalar:.4} s, speedup {ratio:.2}x"
            );
            if ratio >= min_speedup {
                return;
            }
            best = best.max(ratio);
        }
        panic!(
            "{name} is only {best:.2}x faster than scalar code (need >= {min_speedup}x): it has \
             probably been compiled to scalar code (see the doc comment of `gather_acc`)"
        );
    }

    /// Times `gather_acc::<u16, G>` against the scalar chains over `calls` calls of 96 edges.
    fn gather_acc_ratio<const G: usize>() -> (f64, f64) {
        // 64 source rows of 8 cells (nb = 8): 16 KiB, L1-resident, so the loop is compute-bound
        // like the real kernel's hot rows.
        const NB: usize = 8;
        const ROWS: usize = 64;
        const EDGES: usize = 96;
        let src = cells(ROWS * NB, 1);
        let w = floats(EDGES, 2);
        let idx: Vec<u16> = floats(EDGES, 3)
            .iter()
            .map(|x| ((x.abs() * ROWS as f32) as usize).min(ROWS - 1) as u16)
            .collect();
        let calls = 5_000;
        let mut out_kernel = [ZERO8; G];
        let kernel = best_of(REPEATS, || {
            for _ in 0..calls {
                gather_acc::<u16, G>(
                    black_box(&idx),
                    black_box(&w),
                    black_box(src.cells()),
                    NB,
                    0,
                    &mut out_kernel,
                );
                black_box(&out_kernel);
            }
        });
        // Scalar: one strictly ordered chain per cell, summing all lanes of the cell over all
        // edges -- the same G * 8 * EDGES multiply-adds.
        let mut chains = [0.0f32; G];
        let scalar = best_of(REPEATS, || {
            for _ in 0..calls {
                let (idx, w, src) = (black_box(&idx), black_box(&w), black_box(src.cells()));
                let mut acc = [0.0f32; G];
                for (&p, &wv) in idx.iter().zip(w) {
                    let cells = &src[p as usize * NB..p as usize * NB + G];
                    for (a, cell) in acc.iter_mut().zip(cells) {
                        for &x in cell {
                            *a += wv * x;
                        }
                    }
                }
                chains = acc;
                black_box(&chains);
            }
        });
        // Same arithmetic in a different order: the totals agree to f32 rounding.
        let (total_k, total_s) = (
            sum_lanes(&out_kernel),
            chains.iter().map(|&x| f64::from(x)).sum::<f64>(),
        );
        assert!(
            (total_k - total_s).abs() <= 1e-3 * (1.0 + total_s.abs()),
            "{total_k} vs {total_s}"
        );
        (scalar, kernel)
    }

    /// `G = 4` (8 SSE2 accumulators, no register pressure): a clean ~3-4x on the SSE2 baseline,
    /// ~1x if the loop is scalar.
    #[test]
    fn gather_acc_g4_is_vectorised() {
        check_ratio("gather_acc::<u16, 4>", MIN_SPEEDUP, gather_acc_ratio::<4>);
    }

    /// `G = 8`, the width the engine uses. On the SSE2 baseline its 16 accumulators fill every
    /// xmm register and the gain over scalar is only ~1.5x (the measured price of 12% less CPU
    /// per row than G = 4, which re-reads the weights twice), too close to a scalar kernel's ~1x
    /// to bound tightly; this only catches the kernel becoming *slower* than scalar code (the
    /// `G = 4` test above and `lane_dot_n` carry the vectorisation check, `G = 8` is the same
    /// generic code).
    #[test]
    fn gather_acc_g8_is_vectorised() {
        check_ratio("gather_acc::<u16, 8>", 1.0, gather_acc_ratio::<8>);
    }

    #[test]
    fn lane_dot_n_is_vectorised() {
        check_ratio("lane_dot_n::<4>", MIN_SPEEDUP, lane_dot_ratio);
    }

    fn lane_dot_ratio() -> (f64, f64) {
        const LEN: usize = 256;
        let a: Vec<LaneBuf> = (0..4).map(|m| cells(LEN, 10 + m)).collect();
        let b = cells(LEN, 20);
        let calls = 5_000;
        let mut out_kernel = [ZERO8; 4];
        let kernel = best_of(REPEATS, || {
            for _ in 0..calls {
                let rows = [a[0].cells(), a[1].cells(), a[2].cells(), a[3].cells()];
                out_kernel = lane_dot_n::<4>(black_box(rows), black_box(b.cells()));
                black_box(&out_kernel);
            }
        });
        // Scalar: one strictly ordered chain per row (4 chains) over all cells and lanes.
        let mut chains = [0.0f32; 4];
        let scalar = best_of(REPEATS, || {
            for _ in 0..calls {
                let rows = black_box([a[0].cells(), a[1].cells(), a[2].cells(), a[3].cells()]);
                let b = black_box(b.cells());
                let mut acc = [0.0f32; 4];
                for (c, y) in b.iter().enumerate() {
                    for m in 0..4 {
                        for l in 0..LANES {
                            acc[m] += rows[m][c][l] * y[l];
                        }
                    }
                }
                chains = acc;
                black_box(&chains);
            }
        });
        let (total_k, total_s) = (
            sum_lanes(&out_kernel),
            chains.iter().map(|&x| f64::from(x)).sum::<f64>(),
        );
        assert!(
            (total_k - total_s).abs() <= 1e-3 * (1.0 + total_s.abs()),
            "{total_k} vs {total_s}"
        );
        (scalar, kernel)
    }
}
