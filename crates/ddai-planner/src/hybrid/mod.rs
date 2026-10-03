//! Task 3.5: the live brain of D-041/D-042/D-048, **the fly proposes, exact search on the real
//! DDNet physics decides**.
//!
//! Where this lives: inside `ddai-planner`, not in a new crate. The search *is* the planner --
//! it scores every candidate with the planner's own `evaluate`/`scoreTick`, runs the planner's
//! shield, and shares its hidden state -- and all of that is crate-private. A separate crate
//! would have had to make the planner's internals public API for one consumer. The only piece
//! that needs a heavier dependency, `FlyProposer` (it wraps `ddai_fly::FlyBrain`), lives in
//! `ddai-fly` and implements [`Proposer`] from here, so `ddai-planner` stays light.
//!
//! Modules:
//! * [`config`]: [`HybridConfig`] and every production-only flag (each one documented there).
//! * [`proposer`]: the [`Proposer`] trait, `NoProposer`, `ScriptedProposer`, and the conversion
//!   of an action distribution into plans in the planner's plan encoding.
//! * [`threat`]: the 1vN threat model (who is a threat, input models, defensive score terms,
//!   robust choice, danger flags).
//! * [`anchors`]: geometric hook anchors (ray casts, cached per tile).
//! * [`techniques`]: the technique library of D-048 (macro-plans as candidate generators).
//! * [`engine`]: candidate scoring on worker worlds, serial or on a persistent worker pool.
//! * [`search`]: the decision procedure (pool of candidates, two-stage robust choice, adaptive
//!   budget, shield) and its telemetry.
//! * [`work`]: the work clock (deadline mode counted in physics ticks: reproducible, load-independent).
//! * [`brain`]: [`HybridBrain`], the `ddai_brain::Brain` implementation.

pub mod anchors;
pub mod brain;
pub mod config;
pub mod engine;
pub mod proposer;
pub mod search;
pub mod techniques;
pub mod threat;
pub mod work;

pub use brain::HybridBrain;
pub use config::{HybridConfig, HybridMode, RobustMode, hybrid_planner_preset, hybrid_terms};
pub use proposer::{
    ActionDistribution, NoProposer, ProposalOutcome, ProposeCtx, Proposer, ScriptedProposer, plans_from_distribution,
};
pub use search::{DecisionTelemetry, WorkCounters};
pub use techniques::Tech;
pub use work::WORK_US_PER_TEE_TICK;

/// Marker offset for an *absolute* aim angle inside a [`crate::planner::PlanStep`]. The planner
/// encodes the aim relative to the direction to the victim (`track_aim`), which is what CEM
/// samples and the opening book use; a hook at a wall must not drift when the victim moves, so a
/// technique plan writes `ABS_AIM + angle`. No relative aim ever comes near this (an aim is at most
/// a few radians), so the two encodings cannot be confused, and the TS-parity path never produces
/// one.
pub const ABS_AIM: f64 = 100.0;

/// A plan-step aim that is the absolute angle `theta` (y down, `atan2(dy, dx)`).
pub fn abs_aim(theta: f64) -> f64 {
    ABS_AIM + theta
}

/// Whether `aim` is an [`abs_aim`] encoding.
pub fn is_abs_aim(aim: f64) -> bool {
    aim > ABS_AIM / 2.0
}

/// The absolute aim angle a plan step `aim` means, given the direction to the victim (`rel_base`;
/// pass `0` when the planner does not track the aim).
pub fn resolve_aim(aim: f64, rel_base: f64) -> f64 {
    if is_abs_aim(aim) { aim - ABS_AIM } else { aim + rel_base }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_and_relative_aims_are_distinguishable() {
        for theta in [-std::f64::consts::PI, -1.0, 0.0, 1.0, std::f64::consts::PI] {
            assert!(is_abs_aim(abs_aim(theta)));
            assert!((resolve_aim(abs_aim(theta), 0.7) - theta).abs() < 1e-12);
            assert!(!is_abs_aim(theta));
            assert!((resolve_aim(theta, 0.7) - (theta + 0.7)).abs() < 1e-12);
        }
        // The widest relative aim CEM can sample (mean pi + several sigma of 1.2) stays below.
        assert!(!is_abs_aim(std::f64::consts::PI + 8.0 * 1.2));
    }
}
