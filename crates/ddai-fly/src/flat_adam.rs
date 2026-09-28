//! A tiny, parameter-shape-agnostic Adam optimizer over a flat `&mut [f32]` — used by
//! `crate::demo_brain`'s training loop for the encoder/decoder/world-model parameter groups, which
//! (unlike `crate::params::FlyParams`) don't share one fixed struct shape across every graph size,
//! so `crate::optim::AdamState`/`adam_step` (typed specifically to `FlyParams`) don't fit them.
//! `crate::optim::guarded_adam_step` remains what actually trains the connectome's own `a`/`b`/
//! `theta` (task spec: "train ... with 7.2's guarded optimizer step") — this module is the
//! separate, simpler piece for everything else the demo trains.
//!
//! [`FlatAdamState::step_guarded`] (review round 1, F12, CONFIRMED): an earlier revision's doc
//! comment here argued a NaN/inf rollback was unnecessary because the fly's own `guarded_adam_step`
//! call already protects the shared forward pass. That reasoning missed a real, separate failure
//! mode: [`FlatAdamState::step`] mutates its own `m`/`v` moment buffers *in place*, so one
//! non-finite gradient (a `NaN`/`inf` `grad[i]`, or a step whose resulting `params[i]` overflows to
//! `inf`) doesn't just corrupt that one step's params — it writes `NaN` into `m`/`v` themselves,
//! which **every subsequent step reads from** (`m[i] = beta1 * NaN + ... = NaN` forever after).
//! Once that happens the parameter group is permanently dead for the rest of training, independent
//! of whatever `guarded_adam_step` does for the connectome. [`step_guarded`] fixes this the same
//! way `crate::optim::guarded_adam_step` fixes it for the connectome — snapshot `params`/`m`/`v`
//! before the step, take the step, and roll all three back to the snapshot if either the gradient
//! or the resulting `params` aren't finite — just without that function's extra "probe the live
//! model" closure or learning-rate backoff (nothing here feeds back into a recurrent state the way
//! the connectome's `a`/`b`/`theta` do, so "would the resulting params still be finite" is already
//! a sufficient, cheap validity check). [`step`] itself stays available, unguarded, for a caller
//! (e.g. `demo_brain`'s MLP control) that has no such requirement.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FlatAdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for FlatAdamConfig {
    fn default() -> Self {
        FlatAdamConfig {
            lr: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// Adam's per-parameter moments for one flat parameter vector.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatAdamState {
    step: u64,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl FlatAdamState {
    pub fn new(len: usize) -> Self {
        FlatAdamState {
            step: 0,
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    /// One Adam step: `params`/`grad` must be the same length as this state was created with.
    pub fn step(&mut self, params: &mut [f32], grad: &[f32], config: &FlatAdamConfig) {
        assert_eq!(params.len(), self.m.len(), "FlatAdamState: params.len() mismatch");
        assert_eq!(grad.len(), self.m.len(), "FlatAdamState: grad.len() mismatch");
        self.step += 1;
        let bias_correction1 = (1.0 - f64::from(config.beta1).powi(self.step.min(i32::MAX as u64) as i32)) as f32;
        let bias_correction2 = (1.0 - f64::from(config.beta2).powi(self.step.min(i32::MAX as u64) as i32)) as f32;
        for i in 0..params.len() {
            self.m[i] = config.beta1 * self.m[i] + (1.0 - config.beta1) * grad[i];
            self.v[i] = config.beta2 * self.v[i] + (1.0 - config.beta2) * grad[i] * grad[i];
            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;
            params[i] -= config.lr * m_hat / (v_hat.sqrt() + config.eps);
        }
    }

    /// [`step`] with a NaN/inf guard (review round 1, F12 — see the module doc comment for why
    /// [`step`] alone can permanently corrupt this state's `m`/`v`). Returns whether the step was
    /// actually applied: `false` means `params` is unchanged (either the gradient was already
    /// non-finite, so nothing was attempted, or the tentative update produced a non-finite
    /// `params[i]`, in which case both `params` and this state's `m`/`v` are rolled back to
    /// exactly what they were before the call — the update is discarded, not "clamped").
    pub fn step_guarded(&mut self, params: &mut [f32], grad: &[f32], config: &FlatAdamConfig) -> bool {
        assert_eq!(params.len(), self.m.len(), "FlatAdamState: params.len() mismatch");
        assert_eq!(grad.len(), self.m.len(), "FlatAdamState: grad.len() mismatch");
        if !grad.iter().all(|g| g.is_finite()) {
            return false;
        }
        let params_before = params.to_vec();
        let m_before = self.m.clone();
        let v_before = self.v.clone();
        let step_before = self.step;
        self.step(params, grad, config);
        if params.iter().all(|p| p.is_finite()) {
            true
        } else {
            params.copy_from_slice(&params_before);
            self.m = m_before;
            self.v = v_before;
            self.step = step_before;
            false
        }
    }
}

/// Global-norm gradient clipping over an arbitrary number of flat slices at once (mirrors
/// `crate::optim::clip_grad_norm`'s "one shared scale across every group" convention). Returns
/// the pre-clip norm.
pub fn clip_grad_norm_multi(grads: &mut [&mut [f32]], max_norm: f32) -> f32 {
    let sum_sq: f64 = grads
        .iter()
        .flat_map(|g| g.iter())
        .map(|&x| f64::from(x) * f64::from(x))
        .sum();
    let norm = sum_sq.sqrt() as f32;
    if norm.is_finite() && norm > max_norm && max_norm > 0.0 {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            for x in g.iter_mut() {
                *x *= scale;
            }
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_moves_params_downhill_for_a_constant_gradient() {
        let mut params = vec![1.0f32, -1.0];
        let mut state = FlatAdamState::new(2);
        let config = FlatAdamConfig::default();
        let grad = vec![1.0f32, 1.0];
        let before = params.clone();
        for _ in 0..10 {
            state.step(&mut params, &grad, &config);
        }
        for (b, a) in before.iter().zip(&params) {
            assert!(a < b, "params should decrease under a positive gradient: {a} vs {b}");
        }
    }

    #[test]
    fn zero_gradient_does_not_move_params() {
        let mut params = vec![0.5f32, 0.25];
        let mut state = FlatAdamState::new(2);
        let config = FlatAdamConfig::default();
        let grad = vec![0.0f32, 0.0];
        let before = params.clone();
        state.step(&mut params, &grad, &config);
        assert_eq!(params, before);
    }

    #[test]
    #[should_panic]
    fn step_panics_on_length_mismatch() {
        let mut params = vec![0.0f32];
        let mut state = FlatAdamState::new(2);
        let config = FlatAdamConfig::default();
        state.step(&mut params, &[0.0, 0.0], &config);
    }

    #[test]
    fn clip_grad_norm_multi_rescales_to_the_max_norm() {
        let mut a = vec![3.0f32];
        let mut b = vec![4.0f32]; // combined norm = 5
        let norm = clip_grad_norm_multi(&mut [&mut a, &mut b], 1.0);
        assert!((norm - 5.0).abs() < 1e-5);
        let new_norm = (a[0] * a[0] + b[0] * b[0]).sqrt();
        assert!((new_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn clip_grad_norm_multi_is_noop_below_threshold() {
        let mut a = vec![0.1f32];
        let before = a.clone();
        clip_grad_norm_multi(&mut [&mut a], 10.0);
        assert_eq!(a, before);
    }

    #[test]
    fn step_guarded_applies_a_normal_finite_step_and_reports_it() {
        let mut params = vec![1.0f32, -1.0];
        let mut state = FlatAdamState::new(2);
        let config = FlatAdamConfig::default();
        let applied = state.step_guarded(&mut params, &[1.0, 1.0], &config);
        assert!(applied);
        assert_ne!(params, vec![1.0, -1.0]);
    }

    #[test]
    fn step_guarded_rejects_a_nan_gradient_without_touching_params_or_moments() {
        let mut params = vec![1.0f32, -1.0];
        let mut state = FlatAdamState::new(2);
        let config = FlatAdamConfig::default();
        let m_before = state.m.clone();
        let v_before = state.v.clone();
        let applied = state.step_guarded(&mut params, &[f32::NAN, 1.0], &config);
        assert!(!applied);
        assert_eq!(params, vec![1.0, -1.0]);
        assert_eq!(state.m, m_before);
        assert_eq!(state.v, v_before);
    }

    /// Review round 1, F12's actual failure mode: a plain [`FlatAdamState::step`] with a NaN
    /// gradient corrupts `m`/`v` permanently, so every later step (even with a perfectly fine
    /// gradient) stays NaN forever. [`FlatAdamState::step_guarded`] must not have this problem.
    #[test]
    fn step_guarded_recovers_where_plain_step_would_stay_permanently_nan() {
        let config = FlatAdamConfig::default();

        // Plain `step`: one bad gradient poisons every later step.
        let mut plain_params = vec![1.0f32];
        let mut plain_state = FlatAdamState::new(1);
        plain_state.step(&mut plain_params, &[f32::NAN], &config);
        assert!(plain_params[0].is_nan());
        plain_state.step(&mut plain_params, &[1.0], &config); // a perfectly fine gradient now
        assert!(
            plain_params[0].is_nan(),
            "plain `step` should stay poisoned after one NaN gradient (that's the bug F12 flags)"
        );

        // Guarded `step_guarded`: the same bad gradient is rejected, later good gradients work.
        let mut guarded_params = vec![1.0f32];
        let mut guarded_state = FlatAdamState::new(1);
        let applied_bad = guarded_state.step_guarded(&mut guarded_params, &[f32::NAN], &config);
        assert!(!applied_bad);
        assert_eq!(guarded_params[0], 1.0);
        let applied_good = guarded_state.step_guarded(&mut guarded_params, &[1.0], &config);
        assert!(applied_good);
        assert!(guarded_params[0].is_finite() && guarded_params[0] != 1.0);
    }

    #[test]
    fn step_guarded_rolls_back_when_the_resulting_params_overflow_to_infinity() {
        // A finite gradient (passes the upfront guard) but an infinite learning rate: the
        // resulting `params[i]` update itself is `inf`, which must still be caught and rolled
        // back (the guard checks the *outcome*, not just the inputs).
        let mut params = vec![2.0f32];
        let mut state = FlatAdamState::new(1);
        let config = FlatAdamConfig {
            lr: f32::INFINITY,
            ..FlatAdamConfig::default()
        };
        let m_before = state.m.clone();
        let applied = state.step_guarded(&mut params, &[1.0], &config);
        assert!(!applied);
        assert_eq!(params[0], 2.0, "params must be rolled back exactly");
        assert_eq!(state.m, m_before, "moments must be rolled back too, not left corrupted");
    }
}
