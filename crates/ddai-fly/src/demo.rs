//! Learning demos (acceptance criterion 5): a synthetic binary "which side is active" supervised
//! task, trained end-to-end with this crate's own [`crate::backward::backward`] + [`crate::optim`]
//! + [`crate::train::train_step`] — first on a tiny hand-built graph (5a, `tests/train_demo.rs`),
//! then on the real S graph (5b, `tests/train_demo_real_graph.rs` and the `ddnet-ai fly train-demo`
//! CLI). `#[doc(hidden)]`, like `test_fixtures`: this is demo/reporting glue shared by tests and
//! the CLI, not part of the crate's stable API.
//!
//! **The task**, in both sizes: present a pattern that lights up either the "left" or "right" half
//! of the input neurons; the target is for a designated "left" group of output neurons to read
//! high and a "right" group to read low (or vice versa) — FLY.md §6's own example ("left vs right
//! visual input sectors → direction_left vs direction_right DN groups high"). Everything about
//! *which* neurons are "left"/"right" and *what* "high"/"low" mean is a parameter
//! ([`DirectionDemoConfig`]), not hard-coded, so the same driver runs both sizes.
//!
//! Loss: mean squared error between `dn_rates` and a per-slot target (only the two named output
//! groups' slots are constrained; every other output neuron gets zero gradient — this is
//! deliberately *not* asking the network to also learn jump/hook/fire/aim, which have no target in
//! this synthetic task). Computed from a plain (non-recording) forward pass so the same values
//! double as both the reported loss and (via `2*(pred-target)/count`) the upstream gradient
//! `train_step` needs.

use ddai_flyg::Side;

use crate::backward::BackwardIndex;
use crate::batched::{BatchedEngine, TrainBackend, train_step_batched};
use crate::model::FlyModel;
use crate::optim::{GuardedAdamConfig, GuardedAdamState, GuardedStepOutcome, clip_grad_norm, guarded_adam_step};
use crate::rng::SplitMix64;
use crate::state::FlyState;
use crate::train::{Sequence, train_step};

/// Builds a [`DirectionDemoConfig`]'s graph-derived fields (`left`/`right_inputs`,
/// `left`/`right_outputs`) straight from a real `.flyg`'s own data — the "left"/"right" input
/// sectors are `InputVisual` neurons on that soma side (mirrors `tests/stability.rs`'s left/right
/// step-response convention), and the "left"/"right" output groups are whichever `output_groups`
/// entries are named `left_action`/`right_action` (task 6.3 already builds `direction_left`/
/// `direction_right` groups on the real S/M graphs — see `ddnet-ai fly info`'s output — so this
/// isn't inventing new structure, just reading what's already there). Shared by
/// `tests/train_demo_real_graph.rs` and the `ddnet-ai fly train-demo` CLI so the two can't drift
/// apart on what "left"/"right" mean.
///
/// Panics if either named action is missing from `model.flyg().output_groups`, or if either side
/// has zero matching `InputVisual` neurons — both mean the graph doesn't actually have the
/// structure this demo assumes, which should fail loudly rather than silently run a degenerate
/// (always-losing, or gradient-free) task.
pub fn direction_inputs_outputs_from_flyg(
    model: &FlyModel,
    left_action: &str,
    right_action: &str,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let side_inputs = |side: Side| -> Vec<usize> {
        model
            .input_neuron_indices()
            .iter()
            .enumerate()
            .filter(|&(_, &dense)| model.flyg().neurons[dense as usize].side == side)
            .map(|(k, _)| k)
            .collect()
    };
    let left_inputs = side_inputs(Side::L);
    let right_inputs = side_inputs(Side::R);
    assert!(
        !left_inputs.is_empty() && !right_inputs.is_empty(),
        "expected L/R InputVisual neurons on this graph"
    );

    let output_slots_for = |action: &str| -> Vec<usize> {
        let group = model
            .flyg()
            .output_groups
            .iter()
            .find(|g| g.action == action)
            .unwrap_or_else(|| panic!("expected an output_groups entry named {action:?}"));
        group
            .members
            .iter()
            .map(|m| {
                model
                    .output_slot_for_neuron(m.neuron_index)
                    .expect("output_groups member must be a role=Output neuron")
            })
            .collect()
    };

    (
        left_inputs,
        right_inputs,
        output_slots_for(left_action),
        output_slots_for(right_action),
    )
}

/// Everything about the task's shape and the training run — no hard-coded graph-specific numbers
/// live in `run_direction_demo` itself.
#[derive(Debug, Clone)]
pub struct DirectionDemoConfig {
    /// Positions (indices into `step_decision`'s `inputs` slice / `FlyModel::input_neuron_
    /// indices()`) that make up the "left" side. A training pattern turns on a random subset of
    /// whichever side is active for that pattern (see `activation_prob`) and leaves the other side
    /// (and every non-side input) at 0.
    pub left_inputs: Vec<usize>,
    pub right_inputs: Vec<usize>,
    /// Positions in `dn_rates` (via `FlyModel::output_slot_for_neuron`) that should read `target_high`
    /// when "left" is the active side and `target_low` when "right" is.
    pub left_outputs: Vec<usize>,
    pub right_outputs: Vec<usize>,
    pub target_high: f32,
    pub target_low: f32,
    /// Probability each of the active side's input neurons is turned on for a given pattern
    /// instance (independently) — `1.0` gives exactly one fixed pattern per side (5a's "two input
    /// patterns"); `< 1.0` draws a fresh random subset each time (5b's "generalises to held-out
    /// patterns" needs this: many distinct instances of "left is active").
    pub activation_prob: f32,
    pub batch_size: usize,
    pub t_decisions: usize,
    /// Only the *last* `readout_decisions` decisions get a nonzero loss/gradient — the rest exist
    /// purely to let the signal propagate from input to output before the readout is judged (the
    /// game's own decision only needs `state.step_decision`'s two-substep average to make sense
    /// once the transient from the *previous* decision has settled a bit — a demo evaluating every
    /// single decision including the very first one, whose `dn_rates` is necessarily identical for
    /// every pattern since it's computed from a shared `v_init` before either pattern's input has
    /// had a chance to reach the output, would be judging an irreducible transient rather than
    /// whether the network learned the mapping). Must be `1 <= readout_decisions <= t_decisions`.
    pub readout_decisions: usize,
    pub steps: usize,
    pub grad_clip_norm: f32,
    pub adam: GuardedAdamConfig,
    pub seed: u64,
}

/// One training step's reported numbers (acceptance criterion 5's "report numbers and curves").
#[derive(Debug, Clone, Copy)]
pub struct StepMetric {
    pub step: usize,
    pub loss: f32,
    pub grad_norm: f32,
    /// What [`guarded_adam_step`] actually did this step (acceptance criterion 3's NaN/inf
    /// guard) — the params/optimizer state are unchanged from the previous step whenever this is
    /// anything other than [`GuardedStepOutcome::Applied`].
    pub outcome: GuardedStepOutcome,
    /// Convenience: `true` iff `outcome == GuardedStepOutcome::Applied`.
    pub applied: bool,
    /// The guard's current learning-rate backoff multiplier *after* this step (see
    /// [`GuardedAdamState::lr_scale`]) — `1.0` until the first rollback, then shrinking.
    pub lr_scale: f32,
    /// Largest `|V|` seen (over the batch's whole trajectory) this step — a stand-in for the
    /// activity-band check acceptance criterion 5b asks for ("activity stays in band"); a caller
    /// tracks this across the run and checks it never runs away.
    pub max_abs_v: f32,
}

/// Draws one training pattern: `is_left` (which side is active, alternated deterministically so a
/// batch is always balanced, not left to chance) and the resulting `num_inputs`-long input vector.
fn draw_pattern(rng: &mut SplitMix64, num_inputs: usize, config: &DirectionDemoConfig, is_left: bool) -> Vec<f32> {
    let mut v = vec![0.0f32; num_inputs];
    let side = if is_left {
        &config.left_inputs
    } else {
        &config.right_inputs
    };
    for &idx in side {
        if rng.next_f32_unit() < config.activation_prob {
            v[idx] = 1.0;
        }
    }
    v
}

/// `None` at every slot except `config.left_outputs`/`right_outputs`, which get `target_high` on
/// whichever side is active and `target_low` on the other.
fn make_target(num_outputs: usize, config: &DirectionDemoConfig, is_left: bool) -> Vec<Option<f32>> {
    let mut t = vec![None; num_outputs];
    let (high_side, low_side) = if is_left {
        (&config.left_outputs, &config.right_outputs)
    } else {
        (&config.right_outputs, &config.left_outputs)
    };
    for &idx in high_side {
        t[idx] = Some(config.target_high);
    }
    for &idx in low_side {
        t[idx] = Some(config.target_low);
    }
    t
}

/// MSE loss (only over slots with `Some` target) and its gradient w.r.t. `dn_rates`
/// (`2*(pred-target)/count`, `count` = number of constrained slots, so the loss doesn't implicitly
/// grow with how many output neurons happen to be in the two named groups).
fn mse_loss_and_grad(dn_rates: &[f32], target: &[Option<f32>]) -> (f32, Vec<f32>) {
    let count = target.iter().filter(|t| t.is_some()).count().max(1) as f32;
    let mut loss = 0.0f32;
    let grad: Vec<f32> = dn_rates
        .iter()
        .zip(target)
        .map(|(&pred, &t)| match t {
            Some(target_v) => {
                let diff = pred - target_v;
                loss += diff * diff / count;
                2.0 * diff / count
            }
            None => 0.0,
        })
        .collect();
    (loss, grad)
}

/// Runs `config.steps` batches of `config.batch_size` sequences (half "left" patterns, half
/// "right", alternated), each `config.t_decisions` decisions from `v_init`, training `model` in
/// place with Adam + global-norm gradient clipping + the NaN/inf guard (acceptance criteria 3/4):
/// [`guarded_adam_step`] validates every tentative step by applying it to `model` and running one
/// cheap forward probe (`v_init`, a fixed moderate input, one decision) before committing to it,
/// rolling back (and backing off the learning rate) if that probe — or the updated params
/// themselves — comes back non-finite. Returns one [`StepMetric`] per step, in order — the demo
/// tests/CLI turn this into a loss curve / CSV.
pub fn run_direction_demo(
    model: &mut FlyModel,
    index: &BackwardIndex,
    config: &DirectionDemoConfig,
    v_init: &[f32],
) -> Vec<StepMetric> {
    run_direction_demo_with_backend(model, index, config, v_init, TrainBackend::PerSequence)
}

/// [`run_direction_demo`] with a choice of forward/backward backend (task 7.2b): the per-sequence
/// path of task 7.2 or the batched one ([`crate::batched`]). The loss, the data, the optimiser and
/// the guard are the same; only `train_step` vs `train_step_batched` differs.
pub fn run_direction_demo_with_backend(
    model: &mut FlyModel,
    index: &BackwardIndex,
    config: &DirectionDemoConfig,
    v_init: &[f32],
    backend: TrainBackend,
) -> Vec<StepMetric> {
    let mut engine = (backend == TrainBackend::Batched).then(|| BatchedEngine::new(model));
    let mut rng = SplitMix64::new(config.seed);
    let mut guarded_state = GuardedAdamState::new(model.params());
    let num_inputs = model.num_inputs();
    let num_outputs = model.num_outputs();
    let mut metrics = Vec::with_capacity(config.steps);

    for step in 0..config.steps {
        let mut sequences = Vec::with_capacity(config.batch_size);
        let mut batch_loss = 0.0f32;
        let mut max_abs_v = 0.0f32;

        for i in 0..config.batch_size {
            let is_left = i % 2 == 0;
            let input_t = draw_pattern(&mut rng, num_inputs, config, is_left);
            let target = make_target(num_outputs, config, is_left);

            let mut probe = FlyState::new(model);
            probe.set_v(model, v_init);
            let mut inputs = Vec::with_capacity(config.t_decisions);
            let mut grad_dn = Vec::with_capacity(config.t_decisions);
            let zero_grad = vec![0.0f32; num_outputs];
            let readout_start = config.t_decisions.saturating_sub(config.readout_decisions);
            for decision in 0..config.t_decisions {
                let dn_rates_owned = probe.step_decision(model, &input_t).dn_rates.to_vec();
                max_abs_v = max_abs_v.max(probe.v().iter().fold(0.0f32, |m, &x| m.max(x.abs())));
                inputs.push(input_t.clone());
                if decision >= readout_start {
                    let (loss_t, grad_t) = mse_loss_and_grad(&dn_rates_owned, &target);
                    batch_loss += loss_t;
                    grad_dn.push(grad_t);
                } else {
                    grad_dn.push(zero_grad.clone());
                }
            }
            sequences.push(Sequence {
                v_init: v_init.to_vec(),
                inputs,
                grad_dn,
                extra_taps: vec![],
            });
        }
        batch_loss /= (config.batch_size * config.readout_decisions) as f32;

        let batch = match engine.as_mut() {
            Some(engine) => train_step_batched(model, engine, &sequences, None),
            None => train_step(model, index, &sequences, None),
        }
        .expect("shapes are self-consistent by construction");
        let mut grad = batch.grad;
        grad.scale(1.0 / config.batch_size as f32);
        let grad_norm = clip_grad_norm(&mut grad, config.grad_clip_norm);

        // `guarded_adam_step` decides whether to keep the tentative step by calling this closure
        // with the CANDIDATE params already applied to `model` (see its own doc comment) — a
        // cheap one-decision forward probe from `v_init` under a fixed, moderate input is enough
        // to catch a params update that overflowed to non-finite weights/bias/decay.
        let mut params = model.params().clone();
        let outcome = guarded_adam_step(&mut params, &grad, &mut guarded_state, &config.adam, |candidate| {
            if model.set_params(candidate.clone()).is_err() {
                return false;
            }
            let mut probe = FlyState::new(model);
            probe.set_v(model, v_init);
            let out = probe.step_decision(model, &vec![0.5f32; num_inputs]);
            out.dn_rates.iter().all(|x| x.is_finite()) && probe.v().iter().all(|x| x.is_finite())
        });
        // Whatever `guarded_adam_step` decided, `params` now holds the params `model` must reflect
        // going forward (the accepted candidate on `Applied`, or the pre-step snapshot otherwise)
        // — `model` itself may currently hold a rejected candidate (set inside the closure above),
        // so this always re-syncs it, even though it's a no-op O(nnz) recompute on the `Applied`
        // path (already done once inside the closure).
        model
            .set_params(params)
            .expect("params is always a valid, shape-matching snapshot or accepted candidate");

        metrics.push(StepMetric {
            step,
            loss: batch_loss,
            grad_norm,
            outcome,
            applied: outcome == GuardedStepOutcome::Applied,
            lr_scale: guarded_state.lr_scale,
            max_abs_v,
        });
    }

    metrics
}

/// [`evaluate_direction`]'s result: `accuracy` (fraction of probe patterns where the
/// higher-firing named group matched the active side) is a coarse pass/fail signal that can
/// ceiling at `1.0` well before training is done — real biological connectomes already lateralise
/// somewhat from wiring alone (FLY.md §6's own "Б" motivation: `LC10a -> AOTU019/025 -> DNa02` is
/// lateralised by anatomy, not by anything trained) — so `mean_margin` (the average, signed
/// `correct_side_mean - wrong_side_mean`, in rate units) is the more sensitive number for showing
/// training actually sharpened the separation even when `accuracy` was already `1.0` before it
/// started.
#[derive(Debug, Clone, Copy)]
pub struct DirectionEval {
    pub accuracy: f32,
    pub mean_margin: f32,
}

/// Probes `num_probe_patterns` fresh random patterns (from `seed`, alternating which side is
/// active) through `config.t_decisions` decisions from `v_init` and reports [`DirectionEval`] on
/// the final decision's `dn_rates` — used both for a held-out generalisation check (a caller
/// passes a `seed` never used during training, so every probe pattern's exact random mask is one
/// the network never saw) and for reporting accuracy/margin before vs. after training.
pub fn evaluate_direction(
    model: &FlyModel,
    config: &DirectionDemoConfig,
    v_init: &[f32],
    seed: u64,
    num_probe_patterns: usize,
) -> DirectionEval {
    let mut rng = SplitMix64::new(seed);
    let num_inputs = model.num_inputs();
    let mut correct = 0usize;
    let mut margin_sum = 0.0f32;
    for i in 0..num_probe_patterns {
        let is_left = i % 2 == 0;
        let input_t = draw_pattern(&mut rng, num_inputs, config, is_left);
        let mut state = FlyState::new(model);
        state.set_v(model, v_init);
        let mut dn = vec![0.0f32; model.num_outputs()];
        for _ in 0..config.t_decisions {
            dn.copy_from_slice(state.step_decision(model, &input_t).dn_rates);
        }
        let left_mean: f32 =
            config.left_outputs.iter().map(|&i| dn[i]).sum::<f32>() / config.left_outputs.len().max(1) as f32;
        let right_mean: f32 =
            config.right_outputs.iter().map(|&i| dn[i]).sum::<f32>() / config.right_outputs.len().max(1) as f32;
        let predicted_left = left_mean > right_mean;
        if predicted_left == is_left {
            correct += 1;
        }
        margin_sum += if is_left {
            left_mean - right_mean
        } else {
            right_mean - left_mean
        };
    }
    DirectionEval {
        accuracy: correct as f32 / num_probe_patterns as f32,
        mean_margin: margin_sum / num_probe_patterns as f32,
    }
}
