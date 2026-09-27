//! [`FlyConfig`]: the model's hyper-parameters (FLY.md §4) — everything that is *not* learned
//! (contrast with [`crate::params::FlyParams`], the learned `a`/`b`/`theta`).

use serde::{Deserialize, Serialize};

use crate::error::FlyError;

/// Hyper-parameters of the continuous-time rate model (FLY.md §4), fixed for the lifetime of a
/// [`crate::model::FlyModel`] (unlike [`crate::params::FlyParams`], which `set_params` can
/// change). Defaults match FLY.md/the acceptance criteria exactly; see each field's doc comment.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FlyConfig {
    /// `Δ`, the exponential-Euler substep size in milliseconds. FLY.md default: 10ms.
    pub dt_ms: f32,
    /// Substeps run per `step_decision` call (2 per physics tick at 50Hz, so 4 per 40ms game
    /// decision at `dt_ms = 10`). FLY.md default: 4.
    pub substeps_per_decision: u32,
    /// `γ` in `Z_i = max(1, N_i^in)^γ` (per-neuron input-current normalisation). Acceptance
    /// criterion 1a default: 1.0 (FLY.md documents the literature range `[0.5, 1]`; the
    /// acceptance criteria pin the default to the top of that range).
    pub gamma: f32,
    /// `r_max` in `f(V) = r_max * tanh(relu(V) / r_max)`. Default: 10.0.
    pub r_max: f32,
    /// `τ_max`, the upper clamp on `τ = Δ + softplus(θ)`. Default: 1.0 second.
    pub tau_max_s: f32,
    /// Upper bound (milliseconds) on how long [`crate::state::FlyState::warm_up`] will run
    /// looking for a converged resting state before giving up (FLY.md §4 suggests "300-500ms of
    /// an empty scene", but review round 1 (F1) found the real S/M graphs' default init needs
    /// well over 1s to actually settle — see the crate README's "Warm-up convergence" section —
    /// so this is a *cap*, not a fixed duration: `warm_up` stops earlier, as soon as
    /// [`FlyConfig::warmup_epsilon`] is satisfied). Default: 5000ms (5s).
    pub warmup_cap_ms: u32,
    /// `warm_up` stops once the largest `|ΔV|` (any neuron, between two consecutive decisions)
    /// drops below this. Default: 0.01 (`r_max = 10`, so this is 0.1% of the activation's full
    /// range — see the README table for what this corresponds to on the real graphs).
    pub warmup_epsilon: f32,
}

impl Default for FlyConfig {
    fn default() -> Self {
        Self {
            dt_ms: 10.0,
            substeps_per_decision: 4,
            gamma: 1.0,
            r_max: 10.0,
            tau_max_s: 1.0,
            warmup_cap_ms: 5_000,
            warmup_epsilon: 0.01,
        }
    }
}

impl FlyConfig {
    /// `Δ` in seconds (the unit every rate-constant computation actually wants).
    pub fn dt_s(&self) -> f32 {
        self.dt_ms / 1000.0
    }

    /// Wall-clock duration of one `step_decision` call in simulated time: `substeps * dt_ms`.
    /// E.g. the default (4 substeps, 10ms) gives 40ms, the game's 25Hz decision tick.
    pub fn decision_ms(&self) -> f32 {
        self.substeps_per_decision as f32 * self.dt_ms
    }

    /// Validates every field's range; returns the first violation found as
    /// [`FlyError::InvalidConfig`].
    pub fn validate(&self) -> Result<(), FlyError> {
        if !(self.dt_ms > 0.0 && self.dt_ms.is_finite()) {
            return Err(FlyError::InvalidConfig(format!(
                "dt_ms must be > 0 and finite, got {}",
                self.dt_ms
            )));
        }
        if self.substeps_per_decision == 0 {
            return Err(FlyError::InvalidConfig(
                "substeps_per_decision must be >= 1, got 0".to_string(),
            ));
        }
        if !(self.gamma.is_finite() && self.gamma >= 0.0) {
            return Err(FlyError::InvalidConfig(format!(
                "gamma must be >= 0 and finite, got {}",
                self.gamma
            )));
        }
        if !(self.r_max > 0.0 && self.r_max.is_finite()) {
            return Err(FlyError::InvalidConfig(format!(
                "r_max must be > 0 and finite, got {}",
                self.r_max
            )));
        }
        let dt_s = self.dt_s();
        if !(self.tau_max_s.is_finite() && self.tau_max_s >= dt_s) {
            return Err(FlyError::InvalidConfig(format!(
                "tau_max_s must be finite and >= dt_s ({dt_s}), got {}",
                self.tau_max_s
            )));
        }
        if self.warmup_cap_ms == 0 {
            return Err(FlyError::InvalidConfig("warmup_cap_ms must be >= 1, got 0".to_string()));
        }
        if !(self.warmup_epsilon.is_finite() && self.warmup_epsilon > 0.0) {
            return Err(FlyError::InvalidConfig(format!(
                "warmup_epsilon must be > 0 and finite, got {}",
                self.warmup_epsilon
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        FlyConfig::default().validate().expect("default config should validate");
    }

    #[test]
    fn default_decision_ms_is_40() {
        assert_eq!(FlyConfig::default().decision_ms(), 40.0);
    }

    #[test]
    fn zero_substeps_is_rejected() {
        let c = FlyConfig {
            substeps_per_decision: 0,
            ..FlyConfig::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn non_positive_dt_is_rejected() {
        let mut c = FlyConfig {
            dt_ms: 0.0,
            ..FlyConfig::default()
        };
        assert!(c.validate().is_err());
        c.dt_ms = -1.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn tau_max_below_dt_is_rejected() {
        let default = FlyConfig::default();
        let c = FlyConfig {
            tau_max_s: default.dt_s() / 2.0,
            ..default
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn negative_gamma_is_rejected() {
        let c = FlyConfig {
            gamma: -0.1,
            ..FlyConfig::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn nan_fields_are_rejected() {
        let c = FlyConfig {
            r_max: f32::NAN,
            ..FlyConfig::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn zero_warmup_cap_is_rejected() {
        let c = FlyConfig {
            warmup_cap_ms: 0,
            ..FlyConfig::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn non_positive_warmup_epsilon_is_rejected() {
        let mut c = FlyConfig {
            warmup_epsilon: 0.0,
            ..FlyConfig::default()
        };
        assert!(c.validate().is_err());
        c.warmup_epsilon = -0.1;
        assert!(c.validate().is_err());
    }
}
