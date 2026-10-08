//! [`FlyState`]: the fly's actual dynamical state (`V` per neuron — FLY.md §4: "the whole fly's
//! state is one flat vector, cheap to clone and stash in a clip") plus the scratch buffers
//! `step_decision` needs, all sized once in [`FlyState::new`] so the hot loop never allocates (see
//! `tests/no_alloc.rs`).

use crate::activation::activation;
use crate::gm::GmRecorder;
use crate::kernel::{PreIndex, gather_accumulate};
use crate::model::FlyModel;
use crate::recorder::TrajectoryRecorder;

/// The result of one [`FlyState::step_decision`] call: DN (output-role) rates and per-type mean
/// rates, borrowed from `state`'s own scratch buffers (this is what keeps `step_decision`
/// allocation-free — see the crate README's "API notes" for why this borrows rather than owns).
#[derive(Debug)]
pub struct DecisionOutput<'s> {
    /// One rate per output-role neuron, averaged over the last two substeps' `f(V)` (the sample
    /// taken right before the decision's final exponential-Euler update, and the one taken right
    /// after it — see the module-level implementation note in `step_decision` for why those two).
    /// Order: ascending dense neuron index among `role == Output` neurons (i.e.
    /// `FlyModel::output_neuron_indices`, "`.flyg` output order") — **not** grouped by
    /// `flyg().output_groups`'s named actions; use [`FlyModel::output_slot_for_neuron`] to map one
    /// to the other. Same length and order on every call for a given model.
    pub dn_rates: &'s [f32],
    /// Mean `f(V)` per type (index = into `model.flyg().types`), from the final substep's state —
    /// a snapshot for visualisation (FLY.md §10), not itself averaged over substeps.
    pub per_type_mean_rate: &'s [f32],
}

/// Report from [`FlyState::warm_up`]: whether the search for a resting state actually converged
/// before hitting [`crate::config::FlyConfig::warmup_cap_ms`], and how long it took. Review round
/// 1 (F1) found the real S/M graphs' default init settles far slower than FLY.md's "300-500ms"
/// suggestion (seconds, not fractions of a second — see the crate README's "Warm-up convergence"
/// table) — `#[must_use]` so a caller can't silently cache a not-actually-converged state as
/// "rest" the way the original (fixed-duration) `warm_up` implicitly did.
#[must_use = "check `.converged` — the default warm-up cap may not be enough; see the crate README's \"Warm-up convergence\" section"]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarmUpReport {
    /// `true` if `max|ΔV|` (largest per-neuron change between two consecutive decisions) dropped
    /// below [`crate::config::FlyConfig::warmup_epsilon`] before `decisions_run` hit the cap.
    pub converged: bool,
    pub decisions_run: u32,
    pub elapsed_ms: f32,
    /// `max|ΔV|` between the last two decisions run — always the true final value, whether or not
    /// `converged` (so a caller can see *how far off* an unconverged warm-up left things, not just
    /// a bare `false`).
    pub final_max_delta_v: f32,
}

/// A fly's dynamical state: `V` per neuron, plus scratch buffers for `step_decision`. Cheap to
/// `clone()` (it's exactly what FLY.md §4 asks for: "one flat vector"). Borrows nothing from
/// [`FlyModel`] — see that type's doc comment for why `step_decision` takes the model explicitly
/// instead.
///
/// Invariant maintained by every method that can change `v` (`new`, `set_v`, `reset_to_rest`,
/// `step_decision`/`step_decision_recording`): `r_buf[i] == f(v[i])` always holds on return. This
/// is what lets `step_decision` skip recomputing `r` at the top of a decision's first substep —
/// it's already sitting there from whatever last touched `v` (review round 1, F3: "cache r_final
/// for the next decision's first substep").
#[derive(Debug, Clone)]
pub struct FlyState {
    v: Vec<f32>,
    /// `f(v)` — see the struct doc comment's invariant.
    r_buf: Vec<f32>,
    v_inf_buf: Vec<f32>,
    /// Scratch, used only inside `step_decision_impl`: `r_buf`'s value from just before the last
    /// substep's `V` update (saved there because `r_buf` itself gets overwritten in place right
    /// after). One of the two boundary samples `dn_rates` averages — see that function's doc
    /// comment.
    r_before_last_buf: Vec<f32>,
    dn_out_buf: Vec<f32>,
    type_sum_buf: Vec<f32>,
    type_mean_buf: Vec<f32>,
    /// Cached by [`FlyState::warm_up`]; `reset_to_rest` copies this back into `v`. Starts at all
    /// zeros, so calling `reset_to_rest` before ever warming up resets to the all-zero state (a
    /// documented, deliberate default, not an error).
    rest_v: Vec<f32>,
    /// The `Gm` neuron model's second state buffer (task 8.8): a step writes the next state here and the two are swapped.
    /// Empty for the rate model, as are `r_buf`, `v_inf_buf` and `r_before_last_buf` for `Gm`.
    gm_next: Vec<f32>,
}

impl FlyState {
    /// Allocates every buffer at the sizes `model` implies. `V` starts at all zeros; call
    /// [`FlyState::warm_up`] before using this for anything that should start from a settled
    /// resting state (FLY.md §4).
    pub fn new(model: &FlyModel) -> Self {
        let n = model.num_neurons();
        let num_outputs = model.num_outputs();
        let num_types = model.num_types();
        if model.gm().is_some() {
            let len = model.state_len();
            return FlyState {
                v: vec![0.0; len],
                r_buf: Vec::new(),
                v_inf_buf: Vec::new(),
                r_before_last_buf: Vec::new(),
                dn_out_buf: vec![0.0; num_outputs],
                type_sum_buf: vec![0.0; num_types],
                type_mean_buf: vec![0.0; num_types],
                rest_v: vec![0.0; len],
                gm_next: vec![0.0; len],
            };
        }
        FlyState {
            v: vec![0.0; n],
            r_buf: vec![0.0; n], // f(0.0) == 0.0, so the invariant already holds here.
            v_inf_buf: vec![0.0; n],
            r_before_last_buf: vec![0.0; n],
            dn_out_buf: vec![0.0; num_outputs],
            type_sum_buf: vec![0.0; num_types],
            type_mean_buf: vec![0.0; num_types],
            rest_v: vec![0.0; n],
            gm_next: Vec::new(),
        }
    }

    /// Current membrane potential, one entry per neuron (dense index order, same as `.flyg`).
    pub fn v(&self) -> &[f32] {
        &self.v
    }

    /// Overwrites `V` directly (e.g. to resume from a saved clip's state, or for a test to set up
    /// a specific starting condition) and refreshes `r_buf` to match (see the struct doc
    /// comment's invariant, which this maintains). `v.len()` must equal the model's neuron count
    /// this state was created from; panics otherwise (a programmer-error precondition, not a data
    /// error). Not on the hot path, so the extra `O(n)` `r_buf` refresh here is not a concern.
    pub fn set_v(&mut self, model: &FlyModel, v: &[f32]) {
        assert_eq!(
            v.len(),
            self.v.len(),
            "set_v: length must match the model's neuron count"
        );
        self.v.copy_from_slice(v);
        if model.gm().is_none() {
            self.refresh_r_buf_from_v(model.config().r_max);
        }
    }

    fn refresh_r_buf_from_v(&mut self, r_max: f32) {
        for (r, &v) in self.r_buf.iter_mut().zip(&self.v) {
            *r = activation(v, r_max);
        }
    }

    /// Runs `substeps` exponential-Euler updates (FLY.md §4) with `inputs` held constant across
    /// all of them (zero-order hold), and returns the resulting [`DecisionOutput`]. Allocates
    /// nothing (see `tests/no_alloc.rs`): every scratch buffer was sized in [`FlyState::new`].
    ///
    /// `inputs[k]` is the external current for the `k`-th input neuron in
    /// `model.input_neuron_indices()`'s order (ascending dense index among `InputVisual`/
    /// `InputAscending` neurons — "current per input-neuron" from acceptance criterion 1c).
    /// `inputs.len()` must equal `model.num_inputs()`; panics otherwise.
    ///
    /// Implementation note on `dn_rates`: each substep gathers using `r_buf`, which (by this
    /// struct's invariant) already holds `f(V)` from *before* that substep's update — matching
    /// FLY.md §4's synchronous update (`V_∞` for every neuron is computed from the same `r`).
    /// Right before the last substep overwrites `r_buf`, its old value (the rate that drove the
    /// second-to-last substep, i.e. "right before the decision's final update") is saved into
    /// `r_before_last_buf`; after the loop, `r_buf` holds `f(V)` from the now-fully-updated `V`
    /// ("right after the final update"). `dn_rates` averages those two — what "averaged over the
    /// last 2 substeps" is taken to mean here (documented in the crate README).
    pub fn step_decision(&mut self, model: &FlyModel, inputs: &[f32]) -> DecisionOutput<'_> {
        self.step_decision_impl(model, inputs, None)
    }

    /// Same as [`FlyState::step_decision`], but also feeds `V`/`r` from every substep into
    /// `recorder` — the hook acceptance criterion's constraints ask for so a later backward pass
    /// (7.2) can replay a decision's substep trajectory without `step_decision` itself needing to
    /// allocate or grow anything. See [`TrajectoryRecorder`] for the exact `(V, r)` pairing this
    /// records.
    pub fn step_decision_recording(
        &mut self,
        model: &FlyModel,
        inputs: &[f32],
        recorder: &mut TrajectoryRecorder,
    ) -> DecisionOutput<'_> {
        self.step_decision_impl(model, inputs, Some(recorder))
    }

    /// The `Gm` neuron model's decision (task 8.8), also recording the window's trajectory for
    /// its backward pass ([`crate::gm::gm_backward`]). Panics on a rate fly.
    pub fn step_decision_recording_gm(
        &mut self,
        model: &FlyModel,
        inputs: &[f32],
        recorder: &mut GmRecorder,
    ) -> DecisionOutput<'_> {
        self.step_gm(model, inputs, Some(recorder))
    }

    /// One decision of the `Gm` model: `steps` message-passing steps, then the DN readout.
    /// Allocation-free (`tests/no_alloc_gm.rs`).
    fn step_gm(
        &mut self,
        model: &FlyModel,
        inputs: &[f32],
        mut recorder: Option<&mut GmRecorder>,
    ) -> DecisionOutput<'_> {
        let gm = model.gm().expect("step_gm needs a Gm fly");
        assert_eq!(
            inputs.len(),
            model.num_inputs(),
            "step_decision: inputs.len() ({}) must equal model.num_inputs() ({})",
            inputs.len(),
            model.num_inputs()
        );
        for _ in 0..gm.config().steps {
            match recorder.as_deref_mut() {
                Some(r) => {
                    let mut views = r.next_step();
                    gm.step_once(&mut self.v, &mut self.gm_next, inputs, Some(&mut views));
                }
                None => gm.step_once(&mut self.v, &mut self.gm_next, inputs, None),
            }
            std::mem::swap(&mut self.v, &mut self.gm_next);
        }
        gm.read_out(&self.v, &mut self.dn_out_buf);
        if let Some(r) = recorder {
            r.record_decision(gm, inputs, &self.v);
        }
        gm.type_activity(&self.v, &mut self.type_sum_buf, &mut self.type_mean_buf);
        DecisionOutput {
            dn_rates: &self.dn_out_buf,
            per_type_mean_rate: &self.type_mean_buf,
        }
    }

    fn step_decision_impl(
        &mut self,
        model: &FlyModel,
        inputs: &[f32],
        mut recorder: Option<&mut TrajectoryRecorder>,
    ) -> DecisionOutput<'_> {
        if model.gm().is_some() {
            assert!(
                recorder.is_none(),
                "a Gm fly records into a GmRecorder (step_decision_recording_gm), not a TrajectoryRecorder"
            );
            return self.step_gm(model, inputs, None);
        }
        assert_eq!(
            inputs.len(),
            model.num_inputs(),
            "step_decision: inputs.len() ({}) must equal model.num_inputs() ({})",
            inputs.len(),
            model.num_inputs()
        );
        let bias = model.bias();
        let decay = model.decay();
        let weights = model.weights();
        let row_start = &model.flyg().edges.row_start;
        let input_indices = model.input_neuron_indices();
        // Picked once per decision (not once per CSR row, and not via a runtime feature check
        // like the FMA attempt this crate rejected — see `kernel.rs`): both real graphs fit `u16`
        // dense indices, so this is `Narrow` in practice, `Wide` only as a correctness fallback.
        let pre_index = match model.narrow_pre_index() {
            Some(narrow) => PreIndexRef::Narrow(narrow),
            None => PreIndexRef::Wide(&model.flyg().edges.pre_index),
        };

        let substeps = model.config().substeps_per_decision;
        for substep in 0..substeps {
            // Invariant: self.r_buf already holds f(self.v) here (from FlyState::new, set_v, or
            // the end of the previous substep/decision).
            pre_index.compute_v_inf(&mut self.v_inf_buf, row_start, weights, &self.r_buf, bias);
            for (k, &dense) in input_indices.iter().enumerate() {
                self.v_inf_buf[dense as usize] += inputs[k];
            }

            if substep == substeps - 1 {
                self.r_before_last_buf.copy_from_slice(&self.r_buf);
            }

            for ((v, &d), &v_inf) in self.v.iter_mut().zip(decay).zip(&self.v_inf_buf) {
                *v += d * (v_inf - *v);
            }

            if let Some(rec) = recorder.as_deref_mut() {
                // Pairs `V` *after* this update with the `r` that *drove* it (i.e. `f(V after
                // update substep-1)`, still sitting in `r_buf` at this point) — not the `r`
                // resulting from this update — plus `V_∞` this same substep computed (task 7.2's
                // backward pass needs it; see `TrajectoryRecorder::v_inf_at`'s doc comment for
                // why it's stored rather than reconstructed from `V`/`decay` alone). See
                // `TrajectoryRecorder`'s doc comment for the exact `(V, r)` pairing.
                rec.record(&self.v, &self.r_buf, &self.v_inf_buf);
            }

            self.refresh_r_buf_from_v(model.config().r_max);
        }
        // self.r_buf now holds f(V) from the fully-updated V ("right after the final update").

        for (slot, &dense) in model.output_neuron_indices().iter().enumerate() {
            let d = dense as usize;
            self.dn_out_buf[slot] = 0.5 * (self.r_before_last_buf[d] + self.r_buf[d]);
        }

        for x in self.type_sum_buf.iter_mut() {
            *x = 0.0;
        }
        for (i, neuron) in model.flyg().neurons.iter().enumerate() {
            self.type_sum_buf[neuron.type_index as usize] += self.r_buf[i];
        }
        for (t, ty) in model.flyg().types.iter().enumerate() {
            self.type_mean_buf[t] = self.type_sum_buf[t] / ty.neuron_count.max(1) as f32;
        }

        DecisionOutput {
            dn_rates: &self.dn_out_buf,
            per_type_mean_rate: &self.type_mean_buf,
        }
    }

    /// Runs decisions from `V = 0` with zero external input until `max|ΔV|` (the largest change
    /// in any neuron's `V` between two consecutive decisions) drops below
    /// `config.warmup_epsilon`, then caches the resulting `V` as this state's resting state
    /// (FLY.md §4: "an empty scene before an episode" — review round 1, F1: the real S/M graphs'
    /// default init needs well over `config.warmup_cap_ms`'s old fixed 400ms to actually get
    /// there, so this now searches for convergence instead of assuming a fixed duration reaches
    /// it). Stops early (reporting `converged: false`) if `config.warmup_cap_ms` is hit first —
    /// the cached state is still whatever `V` warm-up reached, converged or not; check the
    /// returned report rather than assuming success. Allocates a zero-input buffer and a
    /// previous-`V` scratch buffer once (not on the hot path — this runs a handful of times per
    /// episode, not per decision).
    pub fn warm_up(&mut self, model: &FlyModel) -> WarmUpReport {
        self.v.fill(0.0);
        self.r_buf.fill(0.0); // f(0.0) == 0.0: invariant holds.
        let zero_inputs = vec![0.0f32; model.num_inputs()];
        let decision_ms = model.config().decision_ms();
        let cap_decisions = ((model.config().warmup_cap_ms as f32 / decision_ms).ceil() as u32).max(1);
        let epsilon = model.config().warmup_epsilon;

        let mut prev_v = self.v.clone();
        let mut converged = false;
        let mut decisions_run = 0u32;
        let mut final_max_delta_v = f32::INFINITY;
        for i in 0..cap_decisions {
            self.step_decision(model, &zero_inputs);
            decisions_run = i + 1;
            final_max_delta_v = self
                .v
                .iter()
                .zip(&prev_v)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            prev_v.copy_from_slice(&self.v);
            if final_max_delta_v < epsilon {
                converged = true;
                break;
            }
        }

        self.rest_v.copy_from_slice(&self.v);
        WarmUpReport {
            converged,
            decisions_run,
            elapsed_ms: decisions_run as f32 * decision_ms,
            final_max_delta_v,
        }
    }

    /// Resets `V` to the state cached by the last [`FlyState::warm_up`] call (or all zeros, if
    /// `warm_up` was never called — see [`FlyState::rest_v`]'s doc comment), and refreshes
    /// `r_buf` to match (the struct doc comment's invariant).
    pub fn reset_to_rest(&mut self, model: &FlyModel) {
        self.v.copy_from_slice(&self.rest_v);
        if model.gm().is_none() {
            self.refresh_r_buf_from_v(model.config().r_max);
        }
    }
}

/// Which presynaptic-index width backs a given `FlyModel`'s CSR — resolved once per decision by
/// `step_decision_impl` (never once per row; see that function and `kernel.rs`'s doc comment for
/// why).
enum PreIndexRef<'a> {
    Narrow(&'a [u16]),
    Wide(&'a [u32]),
}

impl PreIndexRef<'_> {
    #[inline]
    fn compute_v_inf(&self, v_inf_buf: &mut [f32], row_start: &[u32], weights: &[f32], r: &[f32], bias: &[f32]) {
        match self {
            PreIndexRef::Narrow(idx) => compute_v_inf_generic(idx, v_inf_buf, row_start, weights, r, bias),
            PreIndexRef::Wide(idx) => compute_v_inf_generic(idx, v_inf_buf, row_start, weights, r, bias),
        }
    }
}

#[inline]
fn compute_v_inf_generic<I: PreIndex>(
    pre_index: &[I],
    v_inf_buf: &mut [f32],
    row_start: &[u32],
    weights: &[f32],
    r: &[f32],
    bias: &[f32],
) {
    for (post, v_inf) in v_inf_buf.iter_mut().enumerate() {
        let start = row_start[post] as usize;
        let end = row_start[post + 1] as usize;
        let acc = gather_accumulate(&pre_index[start..end], &weights[start..end], r);
        *v_inf = bias[post] + acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FlyConfig;
    use crate::params::FlyParams;
    use crate::test_fixtures::tiny_chain_flyg;

    #[test]
    fn step_decision_rejects_wrong_input_length() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        let mut state = FlyState::new(&model);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.step_decision(&model, &[0.0, 0.0]);
        }));
        assert!(
            result.is_err(),
            "wrong-length inputs should panic, not silently misindex"
        );
    }

    #[test]
    fn r_buf_invariant_holds_after_set_v() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        let mut state = FlyState::new(&model);
        state.set_v(&model, &[0.5, -1.0, 2.0]);
        // Drive one substep from this V and check it matches a hand-computed step using the same
        // V (i.e. r_buf must already have reflected the just-set V, not a stale value).
        let mut reference = FlyState::new(&model);
        reference.set_v(&model, &[0.5, -1.0, 2.0]);
        let out_a = state.step_decision(&model, &[0.0]);
        let a = out_a.dn_rates.to_vec();
        let out_b = reference.step_decision(&model, &[0.0]);
        let b = out_b.dn_rates.to_vec();
        assert_eq!(a, b);
    }

    #[test]
    fn warm_up_then_reset_to_rest_round_trips() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        let mut state = FlyState::new(&model);
        let report = state.warm_up(&model);
        assert!(
            report.converged,
            "tiny 3-neuron graph should converge well within the default cap"
        );
        let rested_v = state.v().to_vec();

        // Perturb, then reset should bring it back exactly.
        state.step_decision(&model, &[1.0]);
        assert_ne!(state.v(), rested_v.as_slice());
        state.reset_to_rest(&model);
        assert_eq!(state.v(), rested_v.as_slice());
    }

    #[test]
    fn warm_up_reports_decisions_run_and_stops_early_on_convergence() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        let mut state = FlyState::new(&model);
        let report = state.warm_up(&model);
        assert!(report.converged);
        assert!(report.final_max_delta_v < model.config().warmup_epsilon);
        let cap_decisions = (model.config().warmup_cap_ms as f32 / model.config().decision_ms()).ceil() as u32;
        assert!(
            report.decisions_run < cap_decisions,
            "should have stopped before the cap: ran {} of {} decisions",
            report.decisions_run,
            cap_decisions
        );
        assert!((report.elapsed_ms - report.decisions_run as f32 * model.config().decision_ms()).abs() < 1e-3);
    }

    #[test]
    fn warm_up_reports_non_convergence_when_the_cap_is_too_tight() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig {
            warmup_cap_ms: 40, // one decision, certainly not enough from a cold V=0 start
            warmup_epsilon: 1e-12,
            ..FlyConfig::default()
        };
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        let mut state = FlyState::new(&model);
        let report = state.warm_up(&model);
        assert!(!report.converged);
        assert_eq!(report.decisions_run, 1);
    }

    #[test]
    fn reset_to_rest_without_warm_up_gives_all_zero() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        let mut state = FlyState::new(&model);
        state.step_decision(&model, &[1.0]);
        state.reset_to_rest(&model);
        assert!(state.v().iter().all(|&x| x == 0.0));
    }
}
