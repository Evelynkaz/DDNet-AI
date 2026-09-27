//! [`FlyParams`]: the trainable parameters (FLY.md §4) — everything `set_params`/training (7.2/8)
//! actually changes. Contrast with [`crate::config::FlyConfig`] for the fixed hyper-parameters.

use ddai_flyg::{Flyg, NeuronRole};
use serde::{Deserialize, Serialize};

use crate::activation::inverse_softplus;
use crate::config::FlyConfig;
use crate::error::FlyError;
use crate::rng::seeded_for;

/// `α_init`: the value `softplus(a)` is initialised to for every shared type-pair parameter (see
/// [`FlyParams::init_default`]). Not derived from first principles (FLY.md leaves the exact
/// operating point to be tuned empirically against the real graphs — acceptance criterion 3).
/// Picked from the α-sweep table in the crate README's "Warm-up convergence" section
/// (reproducible via `tests/stability.rs`'s `alpha_sweep_experiment`), balancing three measured
/// things across both real graphs, not just "healthy activity" (review round 1, F1b — a first
/// pass picked `5.0` from activity/saturation alone and it turned out to settle in 2.4-3.4s,
/// 5-8x slower than the ones below):
/// - **settle time** (`FlyState::warm_up`'s convergence): `S` ≈ 320-360ms, `M` ≈ 880-960ms — both
///   comfortably under the review's targets (`S < 500ms`, `M < 1000ms`);
/// - **not dead/saturated**: `S`/`M` both ~98-100% of neurons firing at all, ~0% pegged near
///   `r_max`;
/// - **step-response separation**: left-only vs. right-only visual input (from converged rest,
///   sustained 1s) moves 30-40% of DN neurons by more than 0.05 and always favours the matching
///   side — lower α values settle faster but move fewer DNs (e.g. `α=1.0`: `S` 280ms/16 DNs moved,
///   `M` 280ms/20 DNs moved), higher ones move more DNs but blow the settle-time budget (e.g.
///   `α=5.0`: `M` 2520ms/73 DNs moved).
///
/// If a future graph or config change makes this dead/saturated (or unacceptably slow to settle)
/// again, re-run `alpha_sweep_experiment` and adjust this constant — not the model equations.
pub const DEFAULT_ALPHA_INIT: f32 = 2.3;

/// Mean of the `b_T` bias initialisation (FLY.md §4: "`N(0.0-0.5, 0.05)`" — a mean somewhere in
/// that range). Documented choice: the lower-middle of the range, so the resting point sits just
/// above the `f(V)` "dead zone" (`V <= 0`) without immediately saturating.
pub const DEFAULT_BIAS_MEAN: f32 = 0.2;
/// Std of the `b_T` bias initialisation.
pub const DEFAULT_BIAS_STD: f32 = 0.05;

/// `τ` target (milliseconds) for hidden-role types at initialisation (FLY.md §4: "~50ms").
pub const DEFAULT_TAU_HIDDEN_MS: f32 = 50.0;
/// `τ` target (milliseconds) for input/output-role types at initialisation — faster than hidden,
/// per FLY.md §4 ("20-30ms for input and output roles").
pub const DEFAULT_TAU_IO_MS: f32 = 25.0;

/// The trainable parameters of the rate model (FLY.md §4), all `f32`:
/// - `a[shared_param_id]`: type-pair connection-strength scale in softplus space,
///   `α = softplus(a) >= 0`.
/// - `b[type_index]`: per-type bias.
/// - `theta[type_index]`: per-type time-constant parameter, `τ = Δ + softplus(θ)` (clamped to
///   `<= τ_max` — see [`crate::config::FlyConfig::tau_max_s`]).
///
/// Lengths must match the `.flyg` a model was built from: `a.len() ==
/// flyg.summary.shared_param_count`, `b.len() == theta.len() == flyg.types.len()` — see
/// [`FlyParams::validate_shape`], which [`crate::model::FlyModel::new`]/`set_params` call before
/// accepting a value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlyParams {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub theta: Vec<f32>,
}

impl FlyParams {
    /// Checks `a`/`b`/`theta` lengths against `flyg`'s shared-param and type counts. Called by
    /// [`crate::model::FlyModel::new`]/`set_params` so a shape mismatch is a clear
    /// [`FlyError::ParamShapeMismatch`], never an out-of-bounds panic deep in the weight
    /// precompute.
    pub fn validate_shape(&self, flyg: &Flyg) -> Result<(), FlyError> {
        let expected_a = flyg.summary.shared_param_count as usize;
        if self.a.len() != expected_a {
            return Err(FlyError::ParamShapeMismatch(format!(
                "a.len() == {} but flyg.summary.shared_param_count == {expected_a}",
                self.a.len()
            )));
        }
        // Defensive: `shared_param_id` should be a dense `0..shared_param_count` range (see
        // `ddai-connectome`'s `subgraph::build` allocation), but this crate doesn't own that
        // invariant, so it checks it explicitly rather than assuming it and indexing out of
        // bounds inside the (hot, unchecked-index) weight precompute.
        for tp in &flyg.type_pairs {
            if tp.shared_param_id as usize >= self.a.len() {
                return Err(FlyError::ParamShapeMismatch(format!(
                    "type_pairs contains shared_param_id {} but a.len() == {}",
                    tp.shared_param_id,
                    self.a.len()
                )));
            }
        }
        let expected_types = flyg.types.len();
        if self.b.len() != expected_types {
            return Err(FlyError::ParamShapeMismatch(format!(
                "b.len() == {} but flyg.types.len() == {expected_types}",
                self.b.len()
            )));
        }
        if self.theta.len() != expected_types {
            return Err(FlyError::ParamShapeMismatch(format!(
                "theta.len() == {} but flyg.types.len() == {expected_types}",
                self.theta.len()
            )));
        }
        if self.a.iter().any(|x| !x.is_finite())
            || self.b.iter().any(|x| !x.is_finite())
            || self.theta.iter().any(|x| !x.is_finite())
        {
            return Err(FlyError::ParamShapeMismatch(
                "a/b/theta must not contain NaN or infinity".to_string(),
            ));
        }
        Ok(())
    }

    /// The documented deterministic default initialisation (FLY.md §4, this module's constants):
    /// - `a`: constant, `softplus(a) == `[`DEFAULT_ALPHA_INIT`] for every shared parameter.
    /// - `b`: `N(`[`DEFAULT_BIAS_MEAN`]`, `[`DEFAULT_BIAS_STD`]`)` per type, seeded deterministically
    ///   from `seed` and the type's index (same seed -> byte-identical params, forever).
    /// - `theta`: set so `τ` starts at [`DEFAULT_TAU_HIDDEN_MS`] for hidden-role types and
    ///   [`DEFAULT_TAU_IO_MS`] for input/output-role types (a type's "role" is that of any neuron
    ///   with that type — see [`type_role`]; a type is never observed with more than one role by
    ///   construction of the `.flyg` builder).
    pub fn init_default(flyg: &Flyg, config: &FlyConfig, seed: u64) -> Self {
        Self::init_default_with_alpha(flyg, config, seed, DEFAULT_ALPHA_INIT)
    }

    /// Same as [`FlyParams::init_default`], but with `alpha_init` (the initial `softplus(a)` for
    /// every shared parameter) as an explicit override instead of always using
    /// [`DEFAULT_ALPHA_INIT`] — used to produce the α-sweep table in the crate README ("Warm-up
    /// convergence" / init-tuning sections) and by `tests/stability.rs`'s `alpha_sweep_experiment`
    /// without needing to duplicate this function.
    pub fn init_default_with_alpha(flyg: &Flyg, config: &FlyConfig, seed: u64, alpha_init: f32) -> Self {
        let shared_param_count = flyg.summary.shared_param_count as usize;
        let a = vec![inverse_softplus(alpha_init); shared_param_count];

        let roles = type_roles(flyg);
        let dt_s = config.dt_s();
        let mut b = Vec::with_capacity(flyg.types.len());
        let mut theta = Vec::with_capacity(flyg.types.len());
        for (type_index, role) in roles.iter().enumerate() {
            let mut rng = seeded_for(seed, type_index as u64);
            let bias = DEFAULT_BIAS_MEAN + DEFAULT_BIAS_STD * rng.next_gaussian();
            b.push(bias);

            let tau_target_s = match role {
                Some(NeuronRole::InputVisual) | Some(NeuronRole::InputAscending) | Some(NeuronRole::Output) => {
                    DEFAULT_TAU_IO_MS / 1000.0
                }
                Some(NeuronRole::Hidden) | None => DEFAULT_TAU_HIDDEN_MS / 1000.0,
            };
            let above_dt = (tau_target_s - dt_s).max(1e-6); // theta must map to a strictly positive softplus
            theta.push(inverse_softplus(above_dt));
        }

        FlyParams { a, b, theta }
    }
}

/// The role of each type (indexed like [`Flyg::types`]), determined by scanning [`Flyg::neurons`]
/// once and remembering the role of the first neuron seen for each type. `None` for a type with no
/// neurons at all (shouldn't happen per `.flyg`'s own invariant that every type has
/// `neuron_count >= 1`, but this function doesn't re-derive that guarantee, so it stays an
/// `Option` rather than panicking on a hypothetically empty type).
fn type_roles(flyg: &Flyg) -> Vec<Option<NeuronRole>> {
    let mut roles: Vec<Option<NeuronRole>> = vec![None; flyg.types.len()];
    for neuron in &flyg.neurons {
        let slot = &mut roles[neuron.type_index as usize];
        if slot.is_none() {
            *slot = Some(neuron.role);
        }
    }
    roles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::tiny_chain_flyg;

    #[test]
    fn init_default_is_deterministic_for_the_same_seed() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let a = FlyParams::init_default(&flyg, &config, 7);
        let b = FlyParams::init_default(&flyg, &config, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn init_default_differs_across_seeds() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let a = FlyParams::init_default(&flyg, &config, 1);
        let b = FlyParams::init_default(&flyg, &config, 2);
        assert_ne!(a.b, b.b, "different seeds should give different bias draws");
    }

    #[test]
    fn init_default_passes_shape_validation() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 42);
        params
            .validate_shape(&flyg)
            .expect("init_default output should validate");
    }

    #[test]
    fn validate_shape_rejects_wrong_a_length() {
        let flyg = tiny_chain_flyg();
        let mut params = FlyParams::init_default(&flyg, &FlyConfig::default(), 1);
        params.a.push(0.0);
        assert!(matches!(
            params.validate_shape(&flyg),
            Err(FlyError::ParamShapeMismatch(_))
        ));
    }

    #[test]
    fn validate_shape_rejects_wrong_b_length() {
        let flyg = tiny_chain_flyg();
        let mut params = FlyParams::init_default(&flyg, &FlyConfig::default(), 1);
        params.b.pop();
        assert!(matches!(
            params.validate_shape(&flyg),
            Err(FlyError::ParamShapeMismatch(_))
        ));
    }

    #[test]
    fn validate_shape_rejects_nan() {
        let flyg = tiny_chain_flyg();
        let mut params = FlyParams::init_default(&flyg, &FlyConfig::default(), 1);
        params.b[0] = f32::NAN;
        assert!(matches!(
            params.validate_shape(&flyg),
            Err(FlyError::ParamShapeMismatch(_))
        ));
    }
}
