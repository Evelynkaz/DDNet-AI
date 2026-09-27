//! Hand-written backward pass (task 7.2, acceptance criterion 1) for the exact forward model of
//! task 7.1 (`crate::state::FlyState::step_decision`/`step_decision_recording`): truncated BPTT
//! over a whole window of `t_decisions` decisions, treated as one flat chain of `t_decisions *
//! substeps_per_decision` exponential-Euler substeps (the substep/decision boundary only matters
//! for *which* input is held and *when* `dn_rates` is read out — the recurrence itself doesn't
//! care).
//!
//! ## How to produce the trajectory this consumes
//! Call [`crate::state::FlyState::step_decision_recording`] `t_decisions` times in a row into the
//! *same* [`crate::recorder::TrajectoryRecorder`] **without** calling `reset()` in between (sized
//! for at least `t_decisions * substeps_per_decision` — see [`crate::recorder::TrajectoryRecorder::new`]).
//! That gives one recorder holding the flattened trajectory [`backward`] expects; nothing new on
//! the forward side was needed for this beyond `TrajectoryRecorder` gaining `V_∞` storage (see
//! that module).
//!
//! ## The math (FLY.md §4, this crate's exact discretisation)
//! Per substep `l` (`0`-indexed into the flat chain, `v_prev` = `v` at `l-1`, or the window's
//! `v_init` for `l == 0`):
//! ```text
//! v_inf_l = bias + W · r_prev_l + input_l        (r_prev_l = f(v_prev_l), W = w_ij post-major CSR)
//! v_l     = v_prev_l + decay · (v_inf_l - v_prev_l)
//! r_l     = f(v_l)
//! ```
//! `dn_rates` for decision `t` (occupying flat indices `[t*S, t*S+S-1]`, `S = substeps_per_decision`)
//! is `0.5 * (r_{t*S+S-2} + r_{t*S+S-1})` (task 7.1's exact averaging rule; `r_{t*S+S-2}` reaches
//! back one substep further — into the *previous* decision, or the window's initial rate, when
//! `S == 1`).
//!
//! Backward walks `l` from the last substep down to `0`, maintaining two rolling "from the future"
//! buffers (no per-step allocation — every buffer lives in [`BpttScratch`], sized once):
//! - `future_delta_v`: `dL/dv_prev` contributed by substep `l+1`'s `v_{l+1} = (1-decay)·v_l +
//!   decay·v_inf_{l+1}` term (`d v_{l+1} / d v_l = 1 - decay`, elementwise per neuron);
//! - `future_dr_from_gather`: `dL/dr_l` contributed by substep `l+1` gathering `r_l` as its
//!   `r_prev` — computed via the **transposed (pre-major) CSR** (acceptance criterion 1's
//!   explicit ask): for presynaptic neuron `j`, `Σ_i w_ij · dL/dv_inf_i` over `j`'s outgoing edges
//!   (an `O(nnz)` gather indexed by post-neuron `i`, cache-friendlier than scattering into `j`
//!   while iterating post-major — see [`BackwardIndex`]).
//!
//! At each `l`: `dr_total = future_dr_from_gather + (direct taps: dn_rates readout, auxiliary
//! per-substep rate grads)`; backprop through `r_l = f(v_l)` (subgradient `0` at the relu kink —
//! [`crate::activation::activation_derivative`]) gives `delta_v_from_r`; `delta_v_total =
//! delta_v_from_r + future_delta_v` is `dL/dv_l` in full; `delta_vinf = decay · delta_v_total` is
//! `dL/dv_inf_l`, which feeds `grad_b`/`grad_theta` (`grad_theta` via the τ-clamp's own
//! subgradient, `0` when clamped — see [`d_decay_d_theta_per_type`]), `grad_inputs`, and (in
//! **post-major** order, matching forward's own CSR order — acceptance criterion 1's "accumulate
//! per-edge gradients and reduce to shared params") `grad_a` (raw `dL/dalpha`, converted to
//! `dL/da` by the softplus derivative once at the end, not per edge).

use crate::activation::{activation_derivative, sigmoid, softplus};
use crate::model::FlyModel;
use crate::recorder::TrajectoryRecorder;

/// One presynaptic-partner index viewed from the *other* side of the graph: for each neuron `j`,
/// the (postsynaptic neuron, original post-major edge index) pairs of `j`'s outgoing edges — the
/// "transposed (pre-major) CSR" acceptance criterion 1 asks the δ back-propagation to use.
/// Topology-only (built once from `model.flyg()`'s edges, never touched by `set_params` — mirrors
/// `FlyModel`'s own `narrow_pre_index`, which is topology-only for the same reason), so one
/// `BackwardIndex` is built once per `FlyModel` and reused for every training step against it.
///
/// Also caches `1/Z_i` and each type-pair's presynaptic sign — the two pieces of `FlyModel`'s own
/// (private, `set_params`-recomputed) weight formula (`w_ij = sign_j · alpha · N_ij / Z_i`) that
/// [`backward`]'s `grad_a` pass needs *decomposed* (not the combined `w_ij`, which already has
/// `alpha` baked in — see the module doc comment). `FlyModel` doesn't expose either array (no
/// training-only need existed before this task), and duplicating the ~10-line computation here
/// (from data `FlyModel::new` itself reads from, `flyg()`/`config()`, both public) is simpler and
/// safer than widening `FlyModel`'s API for one training-only consumer.
#[derive(Debug, Clone)]
pub struct BackwardIndex {
    row_start_pre: Vec<u32>,
    post_of: Vec<u32>,
    edge_id: Vec<u32>,
    /// `sign_j * N_ij / Z_i` for edge `e` (post-major, parallel to `flyg().edges.pre_index` —
    /// same order/indexing as `FlyModel::weights()`), i.e. `weights()[e] / alpha[shared_id]`
    /// *without* the numerically-risky division that would take (see [`build`]'s doc comment):
    /// everything `grad_a`'s per-edge accumulation needs except `alpha` itself, which is the one
    /// piece that actually depends on the current `a` and so can't be folded in here. Precomputed
    /// once (topology + `config.gamma` only) instead of re-deriving `sign`/`N_ij`/`1/Z_i` from
    /// `flyg()` on every edge of every substep of every call to [`backward`] — the same "precompute
    /// once, not on every hot-path call" discipline `FlyModel::recompute` already applies to
    /// `weights` itself (review round 1 of task 7.1 found redundant per-call work like this to be
    /// the dominant cost more than once).
    edge_alpha_coeff: Vec<f32>,
    /// `flyg().type_pairs[flyg().edges.type_pair_index[e]].shared_param_id` for edge `e`
    /// (post-major) — same rationale as `edge_alpha_coeff`: one indirection instead of two
    /// (`type_pair_index[e]` then `type_pairs[..].shared_param_id`) on every edge of every substep.
    edge_shared_id: Vec<u32>,
}

impl BackwardIndex {
    /// `O(nnz + num_neurons)`: counts each neuron's out-degree, prefix-sums into `row_start_pre`,
    /// then a second pass drops each edge into its presynaptic neuron's slot (a standard CSR
    /// transpose by counting sort) plus the small `inv_z`/`type_pair_sign` precomputes.
    pub fn build(model: &FlyModel) -> Self {
        let n = model.num_neurons();
        let flyg = model.flyg();
        let edges = &flyg.edges;
        let nnz = edges.pre_index.len();

        let mut row_start_pre = vec![0u32; n + 1];
        for &pre in &edges.pre_index {
            row_start_pre[pre as usize + 1] += 1;
        }
        for i in 0..n {
            row_start_pre[i + 1] += row_start_pre[i];
        }

        let mut post_of = vec![0u32; nnz];
        let mut edge_id = vec![0u32; nnz];
        let mut cursor = row_start_pre.clone();
        for post in 0..n {
            let start = edges.row_start[post] as usize;
            let end = edges.row_start[post + 1] as usize;
            for e in start..end {
                let pre = edges.pre_index[e] as usize;
                let pos = cursor[pre] as usize;
                post_of[pos] = post as u32;
                edge_id[pos] = e as u32;
                cursor[pre] += 1;
            }
        }

        let gamma = model.config().gamma;
        let inv_z: Vec<f32> = flyg
            .neuron_input_totals
            .full_connectome
            .iter()
            .map(|&z_raw| 1.0 / (z_raw.max(1) as f32).powf(gamma))
            .collect();
        let type_pair_sign: Vec<f32> = flyg
            .type_pairs
            .iter()
            .map(|tp| f32::from(flyg.types[tp.pre_type as usize].sign.as_i8()))
            .collect();

        let mut edge_alpha_coeff = vec![0.0f32; nnz];
        let mut edge_shared_id = vec![0u32; nnz];
        for (post, &inv_z_post) in inv_z.iter().enumerate().take(n) {
            let start = edges.row_start[post] as usize;
            let end = edges.row_start[post + 1] as usize;
            for e in start..end {
                let tp = edges.type_pair_index[e] as usize;
                edge_alpha_coeff[e] = type_pair_sign[tp] * (edges.synapse_count[e] as f32) * inv_z_post;
                edge_shared_id[e] = flyg.type_pairs[tp].shared_param_id;
            }
        }

        BackwardIndex {
            row_start_pre,
            post_of,
            edge_id,
            edge_alpha_coeff,
            edge_shared_id,
        }
    }

    #[inline]
    fn row(&self, pre: usize) -> (&[u32], &[u32]) {
        let start = self.row_start_pre[pre] as usize;
        let end = self.row_start_pre[pre + 1] as usize;
        (&self.post_of[start..end], &self.edge_id[start..end])
    }
}

/// A gradient contribution with respect to an arbitrary neuron's rate at an arbitrary substep of
/// an arbitrary decision within the window — the hook acceptance criterion 1 asks for ("optionally
/// w.r.t. any neuron's rate at any substep, for auxiliary heads"). `grad` is dense, `num_neurons`
/// long (dense neuron index order, like everything else in this crate); `local_substep` is
/// `0..substeps_per_decision`.
#[derive(Debug, Clone, Copy)]
pub struct ExtraRateGrad<'a> {
    pub decision: usize,
    pub local_substep: usize,
    pub grad: &'a [f32],
}

/// Every gradient [`backward`] produces, one field per trainable-parameter group (acceptance
/// criterion 1) plus the per-decision input-current gradient and (optionally) the window's
/// initial-state gradient for truncated-BPTT chaining.
#[derive(Debug, Clone)]
pub struct BpttGradients {
    /// `dL/da[shared_param_id]`, same length/order as `FlyParams::a`.
    pub grad_a: Vec<f32>,
    /// `dL/db[type_index]`, same length/order as `FlyParams::b`.
    pub grad_b: Vec<f32>,
    /// `dL/dtheta[type_index]`, same length/order as `FlyParams::theta`.
    pub grad_theta: Vec<f32>,
    /// `dL/dinput[t][k]`: one `num_inputs`-long vector per decision, same order as
    /// `step_decision`'s own `inputs` slice (`FlyModel::input_neuron_indices()`'s order).
    pub grad_inputs: Vec<Vec<f32>>,
    /// `dL/dv_init`, `num_neurons` long, present iff `backward` was asked for it
    /// (`want_v_init_grad`) — the hook for chaining truncated BPTT across windows (the *forward*
    /// state itself, `V`, is carried over regardless; this is only needed if a later window's loss
    /// should also push gradient back into an earlier window's final state, which plain truncated
    /// BPTT deliberately does *not* do by default — most callers pass `false`).
    pub grad_v_init: Option<Vec<f32>>,
}

/// Persistent scratch buffers for [`backward`], sized once from a [`FlyModel`] and reused across
/// every call against it (acceptance criterion 1's "without per-step allocation" — here read as
/// "without per-*substep*-of-the-inner-BPTT-loop allocation": [`BpttGradients`]'s own output
/// vectors are still allocated fresh per call, since they're the function's return value and
/// small relative to the `O(nnz * length)` work `backward` actually does).
#[derive(Debug, Clone)]
pub struct BpttScratch {
    future_delta_v: Vec<f32>,
    future_dr_from_gather: Vec<f32>,
    next_future_dr_from_gather: Vec<f32>,
    dr_total: Vec<f32>,
    delta_vinf: Vec<f32>,
    pending_dr_at_boundary: Vec<f32>,
    d_decay_d_theta: Vec<f32>,
}

impl BpttScratch {
    pub fn new(model: &FlyModel) -> Self {
        let n = model.num_neurons();
        let t = model.num_types();
        BpttScratch {
            future_delta_v: vec![0.0; n],
            future_dr_from_gather: vec![0.0; n],
            next_future_dr_from_gather: vec![0.0; n],
            dr_total: vec![0.0; n],
            delta_vinf: vec![0.0; n],
            pending_dr_at_boundary: vec![0.0; n],
            d_decay_d_theta: vec![0.0; t],
        }
    }
}

/// `d decay[i] / d theta[T(i)]` for every type `T`, from the model's *current* params (unlike
/// [`BackwardIndex`]'s cached arrays, this depends on `theta` and must be recomputed whenever
/// params change — i.e. fresh on every [`backward`] call, into the caller's [`BpttScratch`]
/// buffer, `O(num_types)`).
///
/// `tau = dt + softplus(theta)`, clamped to `<= tau_max` ([`crate::config::FlyConfig::tau_max_s`],
/// same clamp `FlyModel::recompute` applies). **The clamp's own kink gets the same subgradient
/// convention as the relu kink elsewhere in this crate: exactly `0`** when the clamp is active
/// (`dt + softplus(theta) >= tau_max`) — documented here because it is easy to get backwards
/// (picking the *unclamped* branch's slope there would silently teach the optimiser that raising
/// theta further keeps helping, when the forward pass has in fact stopped listening).
/// Otherwise: `d tau/d theta = sigmoid(theta)` (softplus'), `d decay/d tau = -exp(-dt/tau) * dt /
/// tau^2` (`decay = 1 - exp(-dt/tau)`), chained together.
fn d_decay_d_theta_per_type(model: &FlyModel, out: &mut [f32]) {
    let dt_s = model.config().dt_s();
    let tau_max = model.config().tau_max_s;
    for (t, &theta) in model.params().theta.iter().enumerate() {
        let tau_unclamped = dt_s + softplus(theta);
        out[t] = if tau_unclamped >= tau_max {
            0.0
        } else {
            let tau = tau_unclamped;
            let d_tau_d_theta = sigmoid(theta);
            let d_decay_d_tau = -(-dt_s / tau).exp() * dt_s / (tau * tau);
            d_decay_d_tau * d_tau_d_theta
        };
    }
}

/// Adds `weight * grad_dn[slot]` into `dr_total` at each output neuron's dense index — shared by
/// both `dn_rates` readout taps (see the module doc comment: each decision's average contributes
/// to two substeps, both via this same helper with `weight = 0.5`).
#[inline]
fn scatter_output_grad(model: &FlyModel, dr_total: &mut [f32], grad_dn: &[f32], weight: f32) {
    for (slot, &dense) in model.output_neuron_indices().iter().enumerate() {
        dr_total[dense as usize] += weight * grad_dn[slot];
    }
}

/// Runs the backward pass described in the module doc comment. `recorder` must hold exactly
/// `t_decisions * model.config().substeps_per_decision` recorded substeps (i.e. it was filled by
/// calling `step_decision_recording` that many times without an intervening `reset`, starting from
/// a state whose `V` was `v_init`); `grad_dn_rates[t]` is `dL/d(dn_rates)` for decision `t`
/// (`model.num_outputs()` long each, all-zero for a decision with no output-side loss).
///
/// Panics if `recorder.len()`, `grad_dn_rates.len()`, or `v_init.len()` don't match what the model
/// and `t_decisions` imply — a caller-assembled-trajectory mismatch, not a data error.
#[allow(clippy::too_many_arguments)]
pub fn backward(
    model: &FlyModel,
    index: &BackwardIndex,
    recorder: &TrajectoryRecorder,
    v_init: &[f32],
    t_decisions: usize,
    grad_dn_rates: &[&[f32]],
    extra_rate_grads: &[ExtraRateGrad<'_>],
    want_v_init_grad: bool,
    scratch: &mut BpttScratch,
) -> BpttGradients {
    let n = model.num_neurons();
    let s = model.config().substeps_per_decision as usize;
    let l_total = t_decisions * s;
    assert_eq!(v_init.len(), n, "backward: v_init.len() must equal num_neurons");
    assert_eq!(
        recorder.len(),
        l_total,
        "backward: recorder must hold exactly t_decisions * substeps_per_decision substeps"
    );
    assert_eq!(
        grad_dn_rates.len(),
        t_decisions,
        "backward: grad_dn_rates.len() must equal t_decisions"
    );
    for g in grad_dn_rates {
        assert_eq!(
            g.len(),
            model.num_outputs(),
            "backward: grad_dn_rates[t].len() mismatch"
        );
    }
    // Review round 1, F5: an out-of-range tap used to be silently mis-mapped (`local_substep >=
    // s` wraps into the *next* decision's substep — `decision * s + local_substep` doesn't know
    // that was a mistake, it's just arithmetic) or silently dropped (`decision >= t_decisions`
    // pushes the flat index past `l_total`, past the end of the loop below, so the tap's gradient
    // never gets added anywhere and nothing says so) — both are exactly the kind of bug that would
    // otherwise show up only as "this auxiliary head's gradient is quietly wrong", nowhere close
    // to its actual cause.
    for tap in extra_rate_grads {
        assert!(
            tap.decision < t_decisions,
            "backward: ExtraRateGrad.decision ({}) must be < t_decisions ({t_decisions})",
            tap.decision
        );
        assert!(
            tap.local_substep < s,
            "backward: ExtraRateGrad.local_substep ({}) must be < substeps_per_decision ({s})",
            tap.local_substep
        );
        assert_eq!(
            tap.grad.len(),
            n,
            "backward: ExtraRateGrad.grad.len() ({}) must equal num_neurons ({n})",
            tap.grad.len()
        );
    }

    let flyg = model.flyg();
    let edges = &flyg.edges;
    let weights = model.weights();
    let decay = model.decay();
    let r_max = model.config().r_max;
    let num_inputs = model.num_inputs();
    let input_indices = model.input_neuron_indices();

    let shared_param_count = model.params().a.len();
    let num_types = model.num_types();
    let mut grad_a = vec![0.0f32; shared_param_count];
    let mut grad_b = vec![0.0f32; num_types];
    let mut grad_theta = vec![0.0f32; num_types];
    let mut grad_inputs: Vec<Vec<f32>> = vec![vec![0.0f32; num_inputs]; t_decisions];

    d_decay_d_theta_per_type(model, &mut scratch.d_decay_d_theta);

    scratch.future_delta_v.fill(0.0);
    scratch.future_dr_from_gather.fill(0.0);
    scratch.pending_dr_at_boundary.fill(0.0);
    // The one boundary case the main loop's modular tap arithmetic structurally cannot reach:
    // decision 0's "before-last" tap targets flat index `s - 2`, which is negative (the window's
    // `v_init`/`r_init`, not any recorded substep) exactly when `s == 1` — see the module doc
    // comment's "S == 1" note. Stashed here and folded in after the main loop, once, rather than
    // special-cased inside the hot per-substep loop. `t_decisions > 0` guards an empty window
    // (review round 1, F7b): with zero decisions there is no "decision 0" at all, so
    // `grad_dn_rates[0]` would be indexing an empty slice — an empty window should just produce
    // all-zero gradients, not panic.
    if s == 1 && t_decisions > 0 {
        scatter_output_grad(model, &mut scratch.pending_dr_at_boundary, grad_dn_rates[0], 0.5);
    }

    for l in (0..l_total).rev() {
        let decision = l / s;

        // --- direct taps into dr_total (dn_rates readout, auxiliary heads) -------------------
        scratch.dr_total.copy_from_slice(&scratch.future_dr_from_gather);
        if (l + 1) % s == 0 {
            let t_final = (l + 1) / s - 1;
            scatter_output_grad(model, &mut scratch.dr_total, grad_dn_rates[t_final], 0.5);
        }
        if (l + 2) % s == 0 && (l + 2) / s >= 1 {
            let t_before = (l + 2) / s - 1;
            if t_before < t_decisions {
                scatter_output_grad(model, &mut scratch.dr_total, grad_dn_rates[t_before], 0.5);
            }
        }
        for tap in extra_rate_grads {
            // Shape/range already validated once, up front — see this function's opening checks.
            if tap.decision * s + tap.local_substep == l {
                for (d, &g) in scratch.dr_total.iter_mut().zip(tap.grad) {
                    *d += g;
                }
            }
        }

        // --- backprop through r_l = f(v_l) ----------------------------------------------------
        // `activation_derivative` (below) takes `v_l` directly (F7a: computing it from `r_l`
        // instead lost precision deep in saturation) — so, unlike an earlier version of this
        // function, there is no need to reconstruct the last substep's un-recorded resulting rate
        // at all here; only `v_l` matters.
        let v_l = recorder.v_at(l);
        let v_inf_l = recorder.v_inf_at(l);
        let v_prev_l: &[f32] = if l == 0 { v_init } else { recorder.v_at(l - 1) };
        let r_prev_l = recorder.r_at(l); // f(v_prev_l): recorder's own (V,r) pairing convention.

        for i in 0..n {
            let delta_v_from_r = scratch.dr_total[i] * activation_derivative(v_l[i], r_max);
            let delta_v_total = delta_v_from_r + scratch.future_delta_v[i];
            let delta_vinf_i = decay[i] * delta_v_total;
            scratch.delta_vinf[i] = delta_vinf_i;

            let type_index = flyg.neurons[i].type_index as usize;
            grad_b[type_index] += delta_vinf_i;
            let g_decay = delta_v_total * (v_inf_l[i] - v_prev_l[i]);
            grad_theta[type_index] += g_decay * scratch.d_decay_d_theta[type_index];

            // Reuses `future_delta_v` as this iteration's output slot for the *next* (lower) `l`
            // — safe because every entry is fully overwritten before it's read again.
            scratch.future_delta_v[i] = (1.0 - decay[i]) * delta_v_total;
        }

        for (k, &dense) in input_indices.iter().enumerate() {
            grad_inputs[decision][k] += scratch.delta_vinf[dense as usize];
        }

        // --- grad_a: post-major (same row order as forward), needs r_prev at the PRE index.
        // `edge_alpha_coeff`/`edge_shared_id` are precomputed once in `BackwardIndex::build`
        // (topology + gamma only), so this inner loop is two gathers (`r_prev_l[pre]`,
        // `edge_alpha_coeff[e]`) and one scatter-add — no per-edge `type_pairs`/`sign`/`synapse_
        // count`/`inv_z` lookups repeated on every one of the `t_decisions * substeps` calls this
        // makes per training step (that redundant work, still present in an earlier version of
        // this function, cost roughly an order of magnitude of throughput — see the crate
        // README's training-performance section for the measured before/after). --------------
        for post in 0..n {
            let start = edges.row_start[post] as usize;
            let end = edges.row_start[post + 1] as usize;
            let delta_vinf_post = scratch.delta_vinf[post];
            if delta_vinf_post == 0.0 {
                continue;
            }
            for e in start..end {
                let pre = edges.pre_index[e] as usize;
                let shared_id = index.edge_shared_id[e] as usize;
                grad_a[shared_id] += delta_vinf_post * r_prev_l[pre] * index.edge_alpha_coeff[e];
            }
        }

        // --- carried_dr for l-1: transposed (pre-major) CSR, gather over post -----------------
        for pre in 0..n {
            let (posts, edge_ids) = index.row(pre);
            let mut acc = 0.0f32;
            for (&post, &e) in posts.iter().zip(edge_ids) {
                acc += weights[e as usize] * scratch.delta_vinf[post as usize];
            }
            scratch.next_future_dr_from_gather[pre] = acc;
        }
        std::mem::swap(
            &mut scratch.future_dr_from_gather,
            &mut scratch.next_future_dr_from_gather,
        );
    }

    for (shared_id, g) in grad_a.iter_mut().enumerate() {
        *g *= sigmoid(model.params().a[shared_id]);
    }

    let grad_v_init = want_v_init_grad.then(|| {
        let mut dr_init = scratch.future_dr_from_gather.clone();
        for (d, &p) in dr_init.iter_mut().zip(&scratch.pending_dr_at_boundary) {
            *d += p;
        }
        (0..n)
            .map(|i| {
                let delta_v_from_r = dr_init[i] * activation_derivative(v_init[i], r_max);
                delta_v_from_r + scratch.future_delta_v[i]
            })
            .collect()
    });

    BpttGradients {
        grad_a,
        grad_b,
        grad_theta,
        grad_inputs,
        grad_v_init,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FlyConfig;
    use crate::params::FlyParams;
    use crate::state::FlyState;
    use crate::test_fixtures::tiny_chain_flyg;

    /// Builds a model/recorder/scratch for `tiny_chain_flyg` (3 neurons, 1 input, 1 output) with
    /// one decision of one substep recorded — just enough for `backward` to run, for the F5
    /// validation tests below (which only care that a bad `ExtraRateGrad` panics *before* any of
    /// the real computation, not about the resulting gradients' values).
    fn tiny_setup() -> (FlyModel, BackwardIndex, TrajectoryRecorder, Vec<f32>, BpttScratch) {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig {
            substeps_per_decision: 1,
            ..FlyConfig::default()
        };
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        let index = BackwardIndex::build(&model);
        let v_init = vec![0.1f32, 0.2, 0.3];
        let mut state = FlyState::new(&model);
        state.set_v(&model, &v_init);
        let mut recorder = TrajectoryRecorder::new(model.num_neurons(), 1);
        state.step_decision_recording(&model, &[0.5], &mut recorder);
        let scratch = BpttScratch::new(&model);
        (model, index, recorder, v_init, scratch)
    }

    #[test]
    #[should_panic(expected = "ExtraRateGrad.decision")]
    fn backward_panics_on_extra_rate_grad_decision_out_of_range() {
        let (model, index, recorder, v_init, mut scratch) = tiny_setup();
        let grad_dn: Vec<f32> = vec![0.0];
        let extra_grad = vec![0.0f32; model.num_neurons()];
        let taps = [ExtraRateGrad {
            decision: 1, // t_decisions == 1, so valid range is just {0}
            local_substep: 0,
            grad: &extra_grad,
        }];
        backward(
            &model,
            &index,
            &recorder,
            &v_init,
            1,
            &[&grad_dn],
            &taps,
            false,
            &mut scratch,
        );
    }

    #[test]
    #[should_panic(expected = "ExtraRateGrad.local_substep")]
    fn backward_panics_on_extra_rate_grad_local_substep_out_of_range() {
        let (model, index, recorder, v_init, mut scratch) = tiny_setup();
        let grad_dn: Vec<f32> = vec![0.0];
        let extra_grad = vec![0.0f32; model.num_neurons()];
        let taps = [ExtraRateGrad {
            decision: 0,
            local_substep: 1, // substeps_per_decision == 1, so valid range is just {0}
            grad: &extra_grad,
        }];
        backward(
            &model,
            &index,
            &recorder,
            &v_init,
            1,
            &[&grad_dn],
            &taps,
            false,
            &mut scratch,
        );
    }

    #[test]
    #[should_panic(expected = "ExtraRateGrad.grad.len()")]
    fn backward_panics_on_extra_rate_grad_wrong_length() {
        let (model, index, recorder, v_init, mut scratch) = tiny_setup();
        let grad_dn: Vec<f32> = vec![0.0];
        let too_short = vec![0.0f32; model.num_neurons() - 1];
        let taps = [ExtraRateGrad {
            decision: 0,
            local_substep: 0,
            grad: &too_short,
        }];
        backward(
            &model,
            &index,
            &recorder,
            &v_init,
            1,
            &[&grad_dn],
            &taps,
            false,
            &mut scratch,
        );
    }

    /// A tap that's in-range on every axis must not panic and must actually influence the
    /// gradient — the validation above must reject only the invalid cases, not everything.
    #[test]
    fn backward_accepts_an_in_range_extra_rate_grad_and_it_has_an_effect() {
        let (model, index, recorder, v_init, mut scratch) = tiny_setup();
        let grad_dn: Vec<f32> = vec![0.0]; // no direct output-side loss this time
        let without_tap = backward(
            &model,
            &index,
            &recorder,
            &v_init,
            1,
            &[&grad_dn],
            &[],
            false,
            &mut scratch,
        );

        let mut nonzero_grad = vec![0.0f32; model.num_neurons()];
        nonzero_grad[1] = 1.0; // the hidden neuron
        let taps = [ExtraRateGrad {
            decision: 0,
            local_substep: 0,
            grad: &nonzero_grad,
        }];
        let with_tap = backward(
            &model,
            &index,
            &recorder,
            &v_init,
            1,
            &[&grad_dn],
            &taps,
            false,
            &mut scratch,
        );

        assert_ne!(
            without_tap.grad_b, with_tap.grad_b,
            "an ExtraRateGrad tap on a real neuron must actually change the resulting gradient"
        );
    }

    /// Review round 1, F7b: `t_decisions == 0` (an empty window) together with `substeps == 1`
    /// used to panic indexing `grad_dn_rates[0]` on an empty slice — an empty window has no
    /// decisions to have a loss on, so it should just report all-zero gradients, not panic.
    #[test]
    fn backward_on_an_empty_window_returns_all_zero_gradients_without_panicking() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig {
            substeps_per_decision: 1,
            ..FlyConfig::default()
        };
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        let index = BackwardIndex::build(&model);
        let v_init = vec![0.1f32, 0.2, 0.3];
        let recorder = TrajectoryRecorder::new(model.num_neurons(), 0);
        let mut scratch = BpttScratch::new(&model);

        let result = backward(&model, &index, &recorder, &v_init, 0, &[], &[], true, &mut scratch);

        assert!(result.grad_a.iter().all(|&x| x == 0.0));
        assert!(result.grad_b.iter().all(|&x| x == 0.0));
        assert!(result.grad_theta.iter().all(|&x| x == 0.0));
        assert!(
            result.grad_inputs.is_empty(),
            "zero decisions -> zero per-decision input gradients"
        );
        assert!(result.grad_v_init.unwrap().iter().all(|&x| x == 0.0));
    }
}
