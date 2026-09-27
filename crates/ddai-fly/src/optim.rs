//! Training utilities (acceptance criterion 3): Adam with per-group learning rates ([`adam_step`]),
//! global-norm gradient clipping ([`clip_grad_norm`]), an optional L2 pull of `a` towards its
//! initial value ("stay near the connectome", [`add_l2_pull_to_a`]), a flyvis-style per-type
//! activity regulariser ([`activity_regularizer_rate_grad`]), and the actual NaN/inf guard
//! ([`guarded_adam_step`] — snapshots params/optimiser state, applies a tentative step, and rolls
//! it back with a learning-rate backoff if the result (or a caller-supplied validation check)
//! isn't finite; plain [`adam_step`] has no such safety net on its own). None of this depends on
//! how the gradients were produced (`crate::backward::backward` for the connectome model here, but
//! nothing here reads a `FlyModel` directly except the activity regulariser, which only needs
//! per-type neuron counts, and `guarded_adam_step`'s `validate` closure, which is free to use one
//! if the caller wants) — kept generic over `FlyParams`-shaped gradients so it stays easy to unit
//! test in isolation from the (much heavier) backward pass.

use serde::{Deserialize, Serialize};

use crate::model::FlyModel;
use crate::params::FlyParams;

/// One trainable-parameter group's gradient, matching [`FlyParams`]'s own three arrays
/// (`crate::backward::BpttGradients` reduced/summed over a batch — see `crate::train`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParamGradients {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub theta: Vec<f32>,
}

impl ParamGradients {
    pub fn zeros_like(params: &FlyParams) -> Self {
        ParamGradients {
            a: vec![0.0; params.a.len()],
            b: vec![0.0; params.b.len()],
            theta: vec![0.0; params.theta.len()],
        }
    }

    /// `true` iff every entry of every group is finite — the NaN/inf guard's actual check
    /// (acceptance criterion 3: "skips/rolls back a step and reports it").
    pub fn all_finite(&self) -> bool {
        self.a.iter().chain(&self.b).chain(&self.theta).all(|x| x.is_finite())
    }

    /// Adds `other`'s entries into `self` elementwise (all three groups must already be the same
    /// length — a batch-reduction helper for `crate::train`, not a general zip that tolerates
    /// mismatched shapes).
    pub fn add_assign(&mut self, other: &ParamGradients) {
        for (d, s) in self.a.iter_mut().zip(&other.a) {
            *d += s;
        }
        for (d, s) in self.b.iter_mut().zip(&other.b) {
            *d += s;
        }
        for (d, s) in self.theta.iter_mut().zip(&other.theta) {
            *d += s;
        }
    }

    /// Scales every entry of every group by `factor` in place (e.g. `1 / batch_size`, to average
    /// rather than sum a batch's per-sequence gradients).
    pub fn scale(&mut self, factor: f32) {
        for x in self.a.iter_mut().chain(&mut self.b).chain(&mut self.theta) {
            *x *= factor;
        }
    }

    /// `sqrt(Σ g²)` over every entry of every group, computed in `f64` (matches
    /// `crate::checkpoint`'s own "accumulate the wide way, store the narrow way" convention — a
    /// sum of many small `f32` squares is exactly the kind of reduction that benefits from it) so
    /// the *clip decision* doesn't depend on summation order/precision at the boundary the way an
    /// `f32` accumulation could.
    pub fn global_norm(&self) -> f32 {
        let sum_sq: f64 = self
            .a
            .iter()
            .chain(&self.b)
            .chain(&self.theta)
            .map(|&g| f64::from(g) * f64::from(g))
            .sum();
        sum_sq.sqrt() as f32
    }
}

/// Global-norm gradient clipping (acceptance criterion 3): rescales every entry (across all three
/// groups together, one shared scale factor — the usual "global norm" convention, not one clip per
/// group) so the combined norm is at most `max_norm`. A no-op if the norm is already `<= max_norm`
/// or not finite (a non-finite norm means at least one gradient entry is already NaN/inf; rescaling
/// by a finite factor can't fix that — [`guarded_adam_step`]'s [`ParamGradients::all_finite`] check
/// is what actually catches *that* case, and it also catches the case this function structurally
/// cannot: an update that overflows to non-finite **params** even though the gradient itself was
/// fine). Returns the pre-clipping norm, for logging.
pub fn clip_grad_norm(grads: &mut ParamGradients, max_norm: f32) -> f32 {
    let norm = grads.global_norm();
    if norm.is_finite() && norm > max_norm && max_norm > 0.0 {
        let scale = max_norm / norm;
        for x in grads.a.iter_mut().chain(&mut grads.b).chain(&mut grads.theta) {
            *x *= scale;
        }
    }
    norm
}

/// Adds an L2 pull of `a` towards `a_init` directly into `grads.a` (acceptance criterion 3's
/// "stay near the connectome"): `d/da [0.5 * weight * (a - a_init)^2] = weight * (a - a_init)`.
/// `a`/`a_init`/`grads.a` must be the same length. A no-op when `weight == 0.0` (the default —
/// this regulariser is optional).
pub fn add_l2_pull_to_a(grads: &mut ParamGradients, a: &[f32], a_init: &[f32], weight: f32) {
    if weight == 0.0 {
        return;
    }
    assert_eq!(a.len(), a_init.len());
    assert_eq!(grads.a.len(), a.len());
    for ((g, &cur), &init) in grads.a.iter_mut().zip(a).zip(a_init) {
        *g += weight * (cur - init);
    }
}

/// Adam hyper-parameters (acceptance criterion 3), separate learning rates per group — flyvis/
/// fly-lit.md §7.9 both use a much smaller learning rate for the connectome-derived scales/biases/
/// time-constants than a (not-yet-implemented, task 7.3) decoder head; since this crate only ever
/// optimises `a`/`b`/`theta`, all three default to the same "small" end of that range
/// (`2e-4`, within fly-lit.md §7.9's documented `1e-4..5e-4`), independently overridable.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AdamConfig {
    pub lr_a: f32,
    pub lr_b: f32,
    pub lr_theta: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        AdamConfig {
            lr_a: 2e-4,
            lr_b: 2e-4,
            lr_theta: 2e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// Adam's per-parameter first/second moment estimates plus the step counter (needed for bias
/// correction) — the optimiser state acceptance criterion 6 asks to be saved/loaded with the
/// checkpoint alongside `FlyParams` (see `crate::checkpoint`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdamState {
    pub step: u64,
    pub m_a: Vec<f32>,
    pub v_a: Vec<f32>,
    pub m_b: Vec<f32>,
    pub v_b: Vec<f32>,
    pub m_theta: Vec<f32>,
    pub v_theta: Vec<f32>,
}

impl AdamState {
    pub fn new(params: &FlyParams) -> Self {
        AdamState {
            step: 0,
            m_a: vec![0.0; params.a.len()],
            v_a: vec![0.0; params.a.len()],
            m_b: vec![0.0; params.b.len()],
            v_b: vec![0.0; params.b.len()],
            m_theta: vec![0.0; params.theta.len()],
            v_theta: vec![0.0; params.theta.len()],
        }
    }

    /// `true` iff every shape matches `params` — checked before a resumed checkpoint's optimiser
    /// state is trusted (acceptance criterion 6): a checkpoint's `flyg_sha256` check already rules
    /// out a mismatched graph, but this is cheap insurance against a hand-edited/corrupted
    /// checkpoint whose `Checkpoint::params` and `Checkpoint::optimizer` disagree in length.
    pub fn matches_shape(&self, params: &FlyParams) -> bool {
        self.m_a.len() == params.a.len()
            && self.v_a.len() == params.a.len()
            && self.m_b.len() == params.b.len()
            && self.v_b.len() == params.b.len()
            && self.m_theta.len() == params.theta.len()
            && self.v_theta.len() == params.theta.len()
    }

    /// `true` iff every moment estimate (`m`/`v`, all three groups) is finite (review round 1,
    /// F8 — [`guarded_adam_step`]'s own safety net): a large-but-finite gradient can overflow the
    /// second moment `v` to `inf` while the resulting parameter update itself rounds to `~0`
    /// (`m_hat / (v_hat.sqrt() + eps)` with `v_hat = inf` is `~0`, not `NaN`/`inf`) — `params`
    /// alone staying finite is not enough evidence the *state* is still usable, since every
    /// future step through that `inf` moment computes `0/inf`-shaped updates forever, freezing
    /// that parameter silently rather than erroring.
    pub fn all_finite(&self) -> bool {
        self.m_a
            .iter()
            .chain(&self.v_a)
            .chain(&self.m_b)
            .chain(&self.v_b)
            .chain(&self.m_theta)
            .chain(&self.v_theta)
            .all(|x| x.is_finite())
    }
}

/// The handful of Adam scalars that stay the same across all three groups within one
/// [`adam_step`] call, bundled so [`adam_update_group`] doesn't need ten separate arguments.
#[derive(Debug, Clone, Copy)]
struct AdamStepScalars {
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_correction1: f32,
    bias_correction2: f32,
}

#[inline]
fn adam_update_group(p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, s: AdamStepScalars) {
    for i in 0..p.len() {
        m[i] = s.beta1 * m[i] + (1.0 - s.beta1) * g[i];
        v[i] = s.beta2 * v[i] + (1.0 - s.beta2) * g[i] * g[i];
        let m_hat = m[i] / s.bias_correction1;
        let v_hat = v[i] / s.bias_correction2;
        p[i] -= lr * m_hat / (v_hat.sqrt() + s.eps);
    }
}

/// One Adam step: `state.step` is incremented first (so bias correction uses `1..`, matching the
/// standard Adam paper's convention), then each group updates independently with its own learning
/// rate. This is the bare update with no safety net — it does **not** check `grads`/the resulting
/// `params` for NaN/inf, and does not roll back a bad step; [`guarded_adam_step`] wraps this with
/// exactly that (acceptance criterion 3's actual NaN/inf guard). Callers that want the bare
/// version anyway (e.g. inside a `validate` closure that is itself deciding whether to keep a
/// tentative step) are still expected to have already run [`clip_grad_norm`]/[`add_l2_pull_to_a`].
pub fn adam_step(params: &mut FlyParams, grads: &ParamGradients, state: &mut AdamState, config: &AdamConfig) {
    assert_eq!(params.a.len(), grads.a.len());
    assert_eq!(params.b.len(), grads.b.len());
    assert_eq!(params.theta.len(), grads.theta.len());
    assert!(
        state.matches_shape(params),
        "adam_step: AdamState shape does not match FlyParams"
    );

    state.step += 1;
    // f64 exponentiation: `beta^step` for `beta` close to 1 and `step` in the thousands is exactly
    // the kind of repeated-multiplication-in-f32 scenario that accumulates visible drift; done
    // once per call (not once per parameter), so the extra precision is essentially free.
    let scalars = AdamStepScalars {
        beta1: config.beta1,
        beta2: config.beta2,
        eps: config.eps,
        bias_correction1: (1.0 - f64::from(config.beta1).powi(state.step.min(i32::MAX as u64) as i32)) as f32,
        bias_correction2: (1.0 - f64::from(config.beta2).powi(state.step.min(i32::MAX as u64) as i32)) as f32,
    };

    adam_update_group(
        &mut params.a,
        &grads.a,
        &mut state.m_a,
        &mut state.v_a,
        config.lr_a,
        scalars,
    );
    adam_update_group(
        &mut params.b,
        &grads.b,
        &mut state.m_b,
        &mut state.v_b,
        config.lr_b,
        scalars,
    );
    adam_update_group(
        &mut params.theta,
        &grads.theta,
        &mut state.m_theta,
        &mut state.v_theta,
        config.lr_theta,
        scalars,
    );
}

/// What [`guarded_adam_step`] actually did — acceptance criterion 3's "skips/rolls back a step and
/// reports it", made into a value a caller can log/count instead of a bare boolean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GuardedStepOutcome {
    /// The step was applied: `params`/`state.adam` now reflect it.
    Applied,
    /// `grads` already contained NaN/inf — never applied, nothing touched.
    SkippedNonFiniteGradient,
    /// The step was applied tentatively, then rolled back because either the resulting `params`
    /// or the caller's `validate` closure found something non-finite/unacceptable —
    /// `params`/`state.adam` are back to exactly what they were before this call.
    /// `state.lr_scale` was shrunk by `GuardedAdamConfig::backoff_factor` (floored at
    /// `min_lr_scale`), so a persistently-too-large step (e.g. a learning rate that was simply
    /// too big for this problem) backs off geometrically across repeated calls rather than
    /// retrying the exact same (already-shown-to-explode) update forever.
    RolledBack { lr_scale_after: f32 },
}

/// [`AdamState`] plus the backoff multiplier [`guarded_adam_step`] shrinks on every rollback —
/// bundled together (rather than a bare `AdamState`) because the backoff has to persist *across*
/// calls for it to do anything: a single `guarded_adam_step` call that shrinks the learning rate
/// and then throws the scale away would just retry at the original (already-diverging) rate next
/// time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardedAdamState {
    pub adam: AdamState,
    /// Every learning rate in [`GuardedAdamConfig::adam`] is multiplied by this before use.
    /// Starts at `1.0`; never rises back on its own (a caller that wants to retry at full speed
    /// after things stabilise can reset it to `1.0` explicitly — this module doesn't guess when
    /// that's safe).
    pub lr_scale: f32,
}

impl GuardedAdamState {
    pub fn new(params: &FlyParams) -> Self {
        GuardedAdamState {
            adam: AdamState::new(params),
            lr_scale: 1.0,
        }
    }
}

/// [`guarded_adam_step`]'s configuration: the underlying [`AdamConfig`] plus how aggressively to
/// back off after a rollback.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GuardedAdamConfig {
    pub adam: AdamConfig,
    /// Multiplies `lr_scale` after every rollback (e.g. `0.5` halves it each time). Must be in
    /// `(0, 1)` for backoff to actually shrink anything; `1.0` disables backoff (repeated
    /// rollbacks then just repeatedly roll back at the same, still-exploding, rate).
    pub backoff_factor: f32,
    /// Floor on `lr_scale` — keeps backoff from ever reaching exactly `0.0` (which would make
    /// every future step a silent no-op indistinguishable from "training stopped").
    pub min_lr_scale: f32,
}

impl Default for GuardedAdamConfig {
    fn default() -> Self {
        GuardedAdamConfig {
            adam: AdamConfig::default(),
            backoff_factor: 0.5,
            min_lr_scale: 1e-6,
        }
    }
}

/// The actual NaN/inf guard (acceptance criterion 3), wired all the way through — unlike a bare
/// [`adam_step`] call gated on [`ParamGradients::all_finite`] (which only catches a gradient that
/// was *already* non-finite), this also catches an update that turns an otherwise-finite gradient
/// into non-finite **params** (e.g. a learning rate large enough to overflow `f32` on its own —
/// `all_finite()` on the gradient says nothing about that), a **non-finite Adam moment** with
/// `params` still finite (review round 2, F8: a large-but-finite gradient can overflow the second
/// moment `v` to `inf` while the update itself rounds to `~0` — `params` alone would look fine
/// this step, but every future step then computes the same `~0` update through that `inf`
/// forever, silently freezing that parameter rather than erroring — see
/// [`AdamState::all_finite`]), and, optionally, a `validate` closure's own notion of "did this
/// break the model" (typically: apply the candidate params to the live `FlyModel` and run one
/// cheap forward pass, checking its output stays finite).
///
/// Sequence: (1) if `grads` isn't finite, return [`GuardedStepOutcome::SkippedNonFiniteGradient`]
/// without touching anything; (2) snapshot `params`/`state.adam`; (3) run [`adam_step`] with every
/// learning rate scaled by `state.lr_scale`; (4) if the resulting `params` are all finite *and*
/// `state.adam.all_finite()` *and* `validate(&*params)` returns `true`, keep the step and return
/// [`GuardedStepOutcome::Applied`]; (5) otherwise restore the snapshot, shrink `state.lr_scale`,
/// and return [`GuardedStepOutcome::RolledBack`].
///
/// `validate` is called at most once, only in case (4)'s check — a caller with an expensive
/// validation forward pass doesn't pay for it on the (common) path where `grads` was already
/// non-finite.
pub fn guarded_adam_step<F>(
    params: &mut FlyParams,
    grads: &ParamGradients,
    state: &mut GuardedAdamState,
    config: &GuardedAdamConfig,
    validate: F,
) -> GuardedStepOutcome
where
    F: FnOnce(&FlyParams) -> bool,
{
    if !grads.all_finite() {
        return GuardedStepOutcome::SkippedNonFiniteGradient;
    }

    let params_snapshot = params.clone();
    let adam_snapshot = state.adam.clone();

    let scaled = AdamConfig {
        lr_a: config.adam.lr_a * state.lr_scale,
        lr_b: config.adam.lr_b * state.lr_scale,
        lr_theta: config.adam.lr_theta * state.lr_scale,
        ..config.adam
    };
    adam_step(params, grads, &mut state.adam, &scaled);

    // Review round 2, F8: `params` staying finite is not enough on its own — a large-but-finite
    // gradient can overflow the second moment `v` to `inf` while `m_hat / (v_hat.sqrt() + eps)`
    // itself rounds to `~0` (an `inf` denominator, not an `inf`/`NaN` result), so the *parameter*
    // looks fine this step but every future step through that `inf` moment computes the same
    // `~0` update forever — silently freezing it rather than erroring. `state.adam.all_finite()`
    // catches that the params-only check structurally cannot.
    let params_finite = params
        .a
        .iter()
        .chain(&params.b)
        .chain(&params.theta)
        .all(|x| x.is_finite());
    if params_finite && state.adam.all_finite() && validate(params) {
        GuardedStepOutcome::Applied
    } else {
        *params = params_snapshot;
        state.adam = adam_snapshot;
        state.lr_scale = (state.lr_scale * config.backoff_factor).max(config.min_lr_scale);
        GuardedStepOutcome::RolledBack {
            lr_scale_after: state.lr_scale,
        }
    }
}

/// A flyvis-style activity regulariser (acceptance criterion 3, `docs/research/fly-lit.md` §6.1/
/// §6.5): penalises a **type's mean rate** for sitting outside `[low, high]` with a squared-hinge
/// penalty, `weight * relu(low - mean)^2` below the band and `weight * relu(mean - high)^2` above
/// it (zero inside the band, matching "penalise ... outside a band" — not flyvis's own single-
/// target-value version, which this crate's constants would make indistinguishable from a plain L2
/// pull; a band is the strictly more general, requested shape).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActivityRegularizerConfig {
    pub weight: f32,
    pub low: f32,
    pub high: f32,
}

impl Default for ActivityRegularizerConfig {
    /// `weight = 0.0`: disabled unless a caller opts in — this is a regulariser, not a correctness
    /// requirement, and its right band depends on `r_max`/the graph's own operating point (see the
    /// crate README's α-sweep table for how wide "healthy" activity actually is on the real
    /// graphs).
    fn default() -> Self {
        ActivityRegularizerConfig {
            weight: 0.0,
            low: 0.5,
            high: 5.0,
        }
    }
}

/// One decision's activity-regulariser gradient with respect to every neuron's rate, ready to feed
/// into `crate::backward::ExtraRateGrad::grad` at that decision's final substep (mean rate is a
/// per-*type* quantity; distributing `d loss / d mean_rate[T]` back to a per-*neuron* gradient
/// divides by the type's neuron count, since `mean_rate[T] = (1/count) * Σ r[i]` over that type's
/// neurons — a plain average, so every member neuron gets an equal share).
///
/// Returns an all-zero vector when `config.weight == 0.0` (still `num_neurons` long, so a caller
/// can always build an `ExtraRateGrad` without special-casing "regulariser disabled" — the all-zero
/// tap is simply a no-op once added into `backward`'s `dr_total`).
pub fn activity_regularizer_rate_grad(
    model: &FlyModel,
    per_type_mean_rate: &[f32],
    config: &ActivityRegularizerConfig,
) -> Vec<f32> {
    let n = model.num_neurons();
    let mut out = vec![0.0f32; n];
    if config.weight == 0.0 {
        return out;
    }
    assert_eq!(per_type_mean_rate.len(), model.num_types());

    let d_loss_d_mean: Vec<f32> = per_type_mean_rate
        .iter()
        .map(|&mean| {
            let below = (config.low - mean).max(0.0);
            let above = (mean - config.high).max(0.0);
            // d/dmean [w*below^2] = -2*w*below (below>0 zone); d/dmean [w*above^2] = 2*w*above.
            config.weight * (2.0 * above - 2.0 * below)
        })
        .collect();

    let flyg = model.flyg();
    for (i, neuron) in flyg.neurons.iter().enumerate() {
        let t = neuron.type_index as usize;
        let count = flyg.types[t].neuron_count.max(1) as f32;
        out[i] = d_loss_d_mean[t] / count;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FlyConfig;
    use crate::test_fixtures::tiny_chain_flyg;

    fn dummy_params() -> FlyParams {
        FlyParams {
            a: vec![0.1, 0.2, 0.3],
            b: vec![0.0, 0.0],
            theta: vec![0.0, 0.0],
        }
    }

    #[test]
    fn clip_grad_norm_is_noop_below_threshold() {
        let mut g = ParamGradients {
            a: vec![0.1, 0.2],
            b: vec![0.0],
            theta: vec![0.0],
        };
        let before = g.clone();
        let norm = clip_grad_norm(&mut g, 10.0);
        assert_eq!(g, before);
        assert!((norm - before.global_norm()).abs() < 1e-6);
    }

    #[test]
    fn clip_grad_norm_rescales_to_exactly_max_norm() {
        let mut g = ParamGradients {
            a: vec![3.0, 4.0], // norm = 5
            b: vec![0.0],
            theta: vec![0.0],
        };
        clip_grad_norm(&mut g, 1.0);
        let new_norm = g.global_norm();
        assert!((new_norm - 1.0).abs() < 1e-5, "new_norm={new_norm}");
        // Direction preserved: still 3:4.
        assert!((g.a[0] / g.a[1] - 0.75).abs() < 1e-5);
    }

    #[test]
    fn clip_grad_norm_does_not_touch_a_nonfinite_gradient() {
        let mut g = ParamGradients {
            a: vec![f32::NAN, 1.0],
            b: vec![0.0],
            theta: vec![0.0],
        };
        clip_grad_norm(&mut g, 1.0);
        assert!(g.a[0].is_nan());
        assert_eq!(g.a[1], 1.0);
    }

    #[test]
    fn all_finite_detects_nan_and_inf() {
        let mut g = ParamGradients {
            a: vec![1.0, 2.0],
            ..Default::default()
        };
        assert!(g.all_finite());
        g.a[1] = f32::NAN;
        assert!(!g.all_finite());
        g.a[1] = f32::INFINITY;
        assert!(!g.all_finite());
    }

    #[test]
    fn l2_pull_pulls_a_towards_its_init_value() {
        let a = vec![2.0, -1.0];
        let a_init = vec![0.0, 0.0];
        let mut g = ParamGradients {
            a: vec![0.0, 0.0],
            b: vec![],
            theta: vec![],
        };
        add_l2_pull_to_a(&mut g, &a, &a_init, 0.5);
        assert_eq!(g.a, vec![1.0, -0.5]);
    }

    #[test]
    fn l2_pull_is_noop_when_weight_is_zero() {
        let a = vec![2.0];
        let a_init = vec![0.0];
        let mut g = ParamGradients {
            a: vec![7.0],
            b: vec![],
            theta: vec![],
        };
        add_l2_pull_to_a(&mut g, &a, &a_init, 0.0);
        assert_eq!(g.a, vec![7.0]);
    }

    #[test]
    fn adam_step_moves_params_downhill_for_a_constant_gradient() {
        let mut params = dummy_params();
        let mut state = AdamState::new(&params);
        let config = AdamConfig::default();
        let grads = ParamGradients {
            a: vec![1.0, 1.0, 1.0],
            b: vec![0.0, 0.0],
            theta: vec![0.0, 0.0],
        };
        let before = params.a.clone();
        for _ in 0..5 {
            adam_step(&mut params, &grads, &mut state, &config);
        }
        for (b, a) in before.iter().zip(&params.a) {
            assert!(
                a < b,
                "a should have decreased under a positive constant gradient: {a} vs {b}"
            );
        }
        assert_eq!(state.step, 5);
    }

    #[test]
    fn adam_step_with_zero_gradient_does_not_move_params() {
        let mut params = dummy_params();
        let mut state = AdamState::new(&params);
        let config = AdamConfig::default();
        let grads = ParamGradients::zeros_like(&params);
        let before = params.clone();
        adam_step(&mut params, &grads, &mut state, &config);
        assert_eq!(params, before);
    }

    #[test]
    #[should_panic]
    fn adam_step_panics_on_shape_mismatch() {
        let mut params = dummy_params();
        let mut state = AdamState::new(&params);
        let config = AdamConfig::default();
        let mut grads = ParamGradients::zeros_like(&params);
        grads.a.push(0.0);
        adam_step(&mut params, &grads, &mut state, &config);
    }

    #[test]
    fn guarded_adam_step_skips_a_nonfinite_gradient_without_touching_anything() {
        let mut params = dummy_params();
        let before = params.clone();
        let mut state = GuardedAdamState::new(&params);
        let state_before = state.clone();
        let config = GuardedAdamConfig::default();
        let mut grads = ParamGradients::zeros_like(&params);
        grads.a[0] = f32::NAN;

        let outcome = guarded_adam_step(&mut params, &grads, &mut state, &config, |_| true);
        assert_eq!(outcome, GuardedStepOutcome::SkippedNonFiniteGradient);
        assert_eq!(params, before);
        assert_eq!(state, state_before);
    }

    #[test]
    fn guarded_adam_step_applies_a_sane_step() {
        let mut params = dummy_params();
        let before = params.clone();
        let mut state = GuardedAdamState::new(&params);
        let config = GuardedAdamConfig::default();
        let grads = ParamGradients {
            a: vec![1.0, 1.0, 1.0],
            b: vec![0.0, 0.0],
            theta: vec![0.0, 0.0],
        };

        let outcome = guarded_adam_step(&mut params, &grads, &mut state, &config, |_| true);
        assert_eq!(outcome, GuardedStepOutcome::Applied);
        assert_ne!(params, before, "a sane step with a nonzero gradient should move params");
        assert_eq!(
            state.lr_scale, 1.0,
            "a successful step must not touch the backoff scale"
        );
    }

    /// The exact scenario the reviewer found (F1): a learning rate so large that `adam_step`
    /// alone overflows `params` to non-finite even though the *gradient* itself was perfectly
    /// finite — `ParamGradients::all_finite()` on the gradient can never catch this, and unlike a
    /// bare `adam_step`, `guarded_adam_step` must roll it back, shrink the scale, and eventually
    /// let training recover instead of getting stuck applying (or repeatedly skipping) a NaN
    /// update forever.
    #[test]
    fn guarded_adam_step_rolls_back_and_recovers_from_an_absurd_learning_rate() {
        let mut params = dummy_params();
        let mut state = GuardedAdamState::new(&params);
        let config = GuardedAdamConfig {
            adam: AdamConfig {
                lr_a: 1e38,
                lr_b: 1e38,
                lr_theta: 1e38,
                ..AdamConfig::default()
            },
            backoff_factor: 0.1,
            min_lr_scale: 1e-9,
        };
        let grads = ParamGradients {
            a: vec![1.0, 1.0, 1.0],
            b: vec![0.0, 0.0],
            theta: vec![0.0, 0.0],
        };

        let mut rolled_back_at_least_once = false;
        let mut ever_applied = false;
        for _ in 0..40 {
            let before = params.clone();
            let outcome = guarded_adam_step(&mut params, &grads, &mut state, &config, |_| true);
            match outcome {
                GuardedStepOutcome::RolledBack { .. } => {
                    rolled_back_at_least_once = true;
                    assert_eq!(params, before, "a rolled-back step must restore params exactly");
                }
                GuardedStepOutcome::Applied => ever_applied = true,
                GuardedStepOutcome::SkippedNonFiniteGradient => {
                    panic!("the gradient here is always finite")
                }
            }
            assert!(
                params
                    .a
                    .iter()
                    .chain(&params.b)
                    .chain(&params.theta)
                    .all(|x| x.is_finite()),
                "params must never be left non-finite, ever: {params:?}"
            );
        }
        assert!(
            rolled_back_at_least_once,
            "lr=1e38 should have overflowed params at least once"
        );
        assert!(
            ever_applied,
            "after enough backoff (factor 0.1, 40 attempts) the effective lr should shrink enough to apply cleanly"
        );
        assert!(
            state.lr_scale < 1.0,
            "backoff should have shrunk lr_scale from its 1.0 starting point"
        );
    }

    #[test]
    fn guarded_adam_step_rolls_back_when_the_validate_closure_rejects_the_result() {
        let mut params = dummy_params();
        let before = params.clone();
        let mut state = GuardedAdamState::new(&params);
        let state_before = state.clone();
        let config = GuardedAdamConfig::default();
        let grads = ParamGradients {
            a: vec![1.0, 1.0, 1.0],
            b: vec![0.0, 0.0],
            theta: vec![0.0, 0.0],
        };

        let outcome = guarded_adam_step(&mut params, &grads, &mut state, &config, |_| false);
        assert_eq!(
            outcome,
            GuardedStepOutcome::RolledBack {
                lr_scale_after: config.backoff_factor
            }
        );
        assert_eq!(params, before);
        assert_eq!(state.adam, state_before.adam);
    }

    /// Review round 2, F8, the reviewer's exact repro: a gradient large enough (`1e30`) to
    /// overflow Adam's second moment `v` to `inf` while `params` themselves stay finite (the
    /// update rounds to `~0`, since `m_hat / (v_hat.sqrt() + eps)` with `v_hat = inf` is `~0`, not
    /// `NaN`/`inf`) must **not** commit — `state.adam.all_finite()` has to catch what the
    /// `params`-only check structurally cannot. Before this fix, this exact scenario returned
    /// `Applied` with `v_a[0] == inf`, after which every future step recomputed the same `~0`
    /// update through that `inf` moment forever, freezing `a[0]` silently.
    #[test]
    fn guarded_adam_step_rolls_back_a_finite_update_that_leaves_adam_state_nonfinite() {
        let mut params = dummy_params();
        let before = params.clone();
        let mut state = GuardedAdamState::new(&params);
        let state_before = state.clone();
        let config = GuardedAdamConfig::default();
        let grads = ParamGradients {
            a: vec![1e30, 1.0, 0.0], // a[0]'s gradient alone overflows v_a[0] to inf.
            b: vec![0.0, 0.0],
            theta: vec![0.0, 0.0],
        };
        assert!(
            grads.all_finite(),
            "the gradient itself must be finite for this to test F8, not the input guard"
        );

        let outcome = guarded_adam_step(&mut params, &grads, &mut state, &config, |_| true);

        assert_eq!(
            outcome,
            GuardedStepOutcome::RolledBack {
                lr_scale_after: config.backoff_factor
            },
            "a finite gradient that overflows the Adam second moment must roll back, not commit"
        );
        assert_eq!(params, before, "rolled-back params must be restored exactly");
        assert_eq!(
            state.adam, state_before.adam,
            "rolled-back Adam state must be restored exactly (never left at v=inf)"
        );
        assert!(state.adam.all_finite());

        // Recovery: a subsequent step with an ordinary, well-behaved gradient must still be able
        // to move `a[0]` — it must not be permanently frozen by the rejected attempt above.
        let normal_grads = ParamGradients {
            a: vec![1.0, 1.0, 1.0],
            b: vec![0.0, 0.0],
            theta: vec![0.0, 0.0],
        };
        let mut moved = false;
        for _ in 0..20 {
            let before_a0 = params.a[0];
            let outcome = guarded_adam_step(&mut params, &normal_grads, &mut state, &config, |_| true);
            if outcome == GuardedStepOutcome::Applied && params.a[0] != before_a0 {
                moved = true;
                break;
            }
        }
        assert!(
            moved,
            "a[0] must not be stuck forever after the rejected huge-gradient step"
        );
    }

    #[test]
    fn param_gradients_add_assign_and_scale() {
        let mut g1 = ParamGradients {
            a: vec![1.0, 2.0],
            b: vec![3.0],
            theta: vec![4.0],
        };
        let g2 = ParamGradients {
            a: vec![10.0, 20.0],
            b: vec![30.0],
            theta: vec![40.0],
        };
        g1.add_assign(&g2);
        assert_eq!(g1.a, vec![11.0, 22.0]);
        assert_eq!(g1.b, vec![33.0]);
        assert_eq!(g1.theta, vec![44.0]);
        g1.scale(0.5);
        assert_eq!(g1.a, vec![5.5, 11.0]);
    }

    #[test]
    fn activity_regularizer_is_zero_inside_the_band() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = crate::model::FlyModel::new(flyg, config, params).unwrap();
        let reg = ActivityRegularizerConfig {
            weight: 1.0,
            low: 0.0,
            high: 10.0,
        };
        let per_type_mean = vec![1.0; model.num_types()]; // inside [0,10] for every type
        let grad = activity_regularizer_rate_grad(&model, &per_type_mean, &reg);
        assert!(grad.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn activity_regularizer_gradient_is_negative_below_the_band_and_positive_above_it() {
        // `d/dmean [relu(low-mean)^2] = -2*(low-mean) < 0` when `mean < low` — raising the rate
        // *reduces* the penalty, so `dLoss/dr` must be negative there (gradient descent on the
        // parameters then pushes activity up); symmetrically, positive above the band.
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = crate::model::FlyModel::new(flyg, config, params).unwrap();
        let reg = ActivityRegularizerConfig {
            weight: 1.0,
            low: 5.0,
            high: 10.0,
        };

        let below = activity_regularizer_rate_grad(&model, &vec![0.0; model.num_types()], &reg);
        assert!(
            below.iter().all(|&x| x < 0.0),
            "below-band gradient should be negative: {below:?}"
        );

        let above = activity_regularizer_rate_grad(&model, &vec![20.0; model.num_types()], &reg);
        assert!(
            above.iter().all(|&x| x > 0.0),
            "above-band gradient should be positive: {above:?}"
        );
    }

    #[test]
    fn activity_regularizer_disabled_by_default_weight_is_all_zero() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = crate::model::FlyModel::new(flyg, config, params).unwrap();
        let reg = ActivityRegularizerConfig::default();
        let per_type_mean = vec![100.0; model.num_types()]; // way outside any sane band
        let grad = activity_regularizer_rate_grad(&model, &per_type_mean, &reg);
        assert!(grad.iter().all(|&x| x == 0.0));
    }
}
