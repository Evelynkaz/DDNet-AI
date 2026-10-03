//! The 1vN threat model of the hybrid brain (task 3.5, D-041): who counts as a threat, what the
//! rollouts assume each of them does, how the defensive score terms cover all of them, and how a
//! plan is chosen robustly over the combinations of those assumptions.
//!
//! **Who is a threat.** Every other free (alive, not frozen) tee within [`threat_radius`] of us,
//! except the chosen victim (the victim is scored offensively by the planner's own `scoreTick`
//! and is modelled like a threat). The radius is the hook length plus the distance the tees can
//! close during two decisions: a tee that is out of hook range now can, by the time we could
//! react to it (the decision after next), have moved into it and thrown the hook. With the
//! fastest self-propelled speed of 15 px/tick (`hook_drag_speed`, the cap on each axis) and a
//! decision every 2 ticks: `380 + 2 * 2 * 15 = 440 px`; input lag adds `lag * 15`.
//!
//! **Input models.** Each modelled opponent is either `hold` (keeps the input its visible state
//! shows, `enemyInputFromSnapshot`) or `react` (the scripted bot's reply, `scriptedAction`, aiming
//! at us). A "learned" model slots in later as a third option. A *combination* says which
//! opponents react; combination 0 (everybody holds) is the cheap model of stage 1. Only opponents
//! that can act on us are varied (the victim if it is free and within the radius, then the nearest
//! threats); with one or two of them every subset is a combination (2 or 4). The re-scoring stage
//! is skipped altogether when more than `robust.max_relevant` (default 2) can act on us: a 4 ms
//! budget affords 6-9 candidates in a 1v3/1v5 fight and re-scoring would take a third of them.
//!
//! **Defence terms.** The planner's `scoreTick` already covers the victim (hook/launch exposure,
//! being dragged over freeze) and everything global (self frozen, hazard nearness). For every
//! *other* threat [`threat_terms`] adds the same exposure terms with that threat as the source:
//! being hooked by it, being dragged toward it across a hazard, being launched by its hammer.

use crate::config::PlannerConfig;
use crate::fields::{LaunchMemo, drag_crosses_hazard, launch_flight_lands_in_hazard_memo, launch_lands_in_hazard};
use crate::plan_world::PlanWorld;
use crate::planner::LAUNCH_REACH_PX;
use crate::tuning::HOOK_LENGTH;
use crate::types::PlayerInput;
use crate::vmath::vdistance;

/// Upper bound on the extra tees a rollout models individually (the stack arrays in
/// `Planner::evaluate_impl` are this long). A 1v5 fight has 4 threats besides the victim.
pub const MAX_THREATS: usize = 8;

/// The fastest a tee moves by itself, in px per tick (`hook_drag_speed`, per axis).
pub const V_MAX_PX_PER_TICK: f64 = 15.0;

/// The extra tees of one evaluation: their ids, the inputs they hold, and which of them answer
/// with the scripted reaction instead (bit `i` of `react_mask` = threat `i`).
#[derive(Debug)]
pub struct ThreatSet {
    pub ids: Vec<i32>,
    pub inputs: Vec<PlayerInput>,
    pub react_mask: u32,
    /// Multiplier of the defensive terms of these threats ([`threat_terms`]); `1` = the same
    /// weights as for the victim.
    pub weight: f64,
    /// Whether a hook at one of these tees counts as a reachable target for the hook gate
    /// (`gateHook`); `false` keeps the hook reserved for the victim and walls.
    pub hook_targets: bool,
}

impl Clone for ThreatSet {
    fn clone(&self) -> Self {
        ThreatSet {
            ids: self.ids.clone(),
            inputs: self.inputs.clone(),
            react_mask: self.react_mask,
            weight: self.weight,
            hook_targets: self.hook_targets,
        }
    }

    /// Reuses the vectors (the derived `clone_from` would reallocate): the workers copy the
    /// decision's threats in place, so a decision costs them no allocation.
    fn clone_from(&mut self, source: &Self) {
        self.ids.clone_from(&source.ids);
        self.inputs.clone_from(&source.inputs);
        self.react_mask = source.react_mask;
        self.weight = source.weight;
        self.hook_targets = source.hook_targets;
    }
}

/// Threat radius in px (see the module docs for the derivation).
pub fn threat_radius(decision_ticks: i32, lag_ticks: u32) -> f64 {
    *HOOK_LENGTH + f64::from(2 * decision_ticks + lag_ticks as i32) * V_MAX_PX_PER_TICK
}

/// The extra per-tick score of the threats in `threat_ids` (never the victim): the exposure terms
/// of `scoreTick` with each threat as the source. Zero when we are frozen (the fight is already
/// lost then, the global terms say so) and for threats that are dead or frozen themselves.
pub(crate) fn threat_terms<W: PlanWorld>(
    world: &W,
    self_id: i32,
    threat_ids: &[i32],
    cfg: &PlannerConfig,
    mut launch_memo: Option<&mut LaunchMemo>,
) -> f64 {
    if threat_ids.is_empty() {
        return 0.0;
    }
    let Some(me) = world.get_tee(self_id) else {
        return 0.0;
    };
    if !me.alive || me.frozen {
        return 0.0;
    }
    let col = world.collision();
    let mut s = 0.0;
    for &id in threat_ids {
        let Some(t) = world.get_tee(id) else { continue };
        if !t.alive || t.frozen {
            continue;
        }
        if t.hooked_player == self_id {
            s -= cfg.hook_hold_weight * 0.75;
        }
        let separation = vdistance(me.pos, t.pos);
        if cfg.launch_exposure > 0.0 && separation < LAUNCH_REACH_PX {
            let exact = cfg.launch_exact_reach > 0.0
                && separation < cfg.launch_exact_reach
                && (me.pos.y < t.pos.y || (cfg.launch_exact_rise_vy > 0.0 && me.vel.y < -cfg.launch_exact_rise_vy));
            if exact {
                s -= (if cfg.launch_exact_weight > 0.0 {
                    cfg.launch_exact_weight
                } else {
                    cfg.launch_exposure
                }) * launch_flight_lands_in_hazard_memo(
                    launch_memo.as_deref_mut(),
                    col,
                    me.pos,
                    t.pos,
                    separation,
                    me.vel,
                );
            } else {
                s -= cfg.launch_exposure * launch_lands_in_hazard(col, me.pos, t.pos, separation);
            }
        }
        if cfg.drag_exposure > 0.0 && separation < *HOOK_LENGTH && separation >= 1.0 {
            s -= cfg.drag_exposure * drag_crosses_hazard(col, me.pos, t.pos, separation);
        }
    }
    s
}

/// The two-stage robust value of a plan: `lambda * worst + (1 - lambda) * mean` over its scores
/// under every model combination. `lambda = 1` is pure max-min, `0` the plain average.
pub fn robust_value(scores: &[f64], lambda: f64) -> f64 {
    debug_assert!(!scores.is_empty());
    let worst = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    lambda * worst + (1.0 - lambda) * mean
}

/// [`robust_value`] with a probability per model combination: `lambda * worst + (1 - lambda) *
/// expectation`, the expectation weighted by `weights` (normalised here; all zero counts as equal).
pub fn robust_value_weighted(scores: &[f64], weights: &[f64], lambda: f64) -> f64 {
    debug_assert_eq!(scores.len(), weights.len());
    let worst = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let total: f64 = weights.iter().sum();
    let mean = if total > 0.0 {
        scores.iter().zip(weights).map(|(s, w)| s * w).sum::<f64>() / total
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    };
    lambda * worst + (1.0 - lambda) * mean
}

/// The online estimate, for one opponent, of the probability that it answers like the scripted
/// bot ("react") rather than keeps its input ("hold"). Every decision the search predicts both for
/// each modelled opponent and, one decision later, compares them with what the opponent's visible
/// state shows (direction and hook): only when the two predictions differ does the observation
/// count as evidence. An opponent that stands still while the scripted bot would have rushed us
/// drifts toward `hold`, one that rushes us toward `react`. It sets how much the robust choice
/// trusts the expectation over the worst case (an idle victim must not make us play scared).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReactBelief {
    pub p: f64,
}

impl Default for ReactBelief {
    fn default() -> Self {
        ReactBelief { p: 0.5 }
    }
}

impl ReactBelief {
    pub const MIN: f64 = 0.05;
    pub const MAX: f64 = 0.95;
    const RATE: f64 = 0.2;

    /// Evidence from one observed input against the two predictions made a decision ago.
    pub fn observe(&mut self, observed: &PlayerInput, hold: &PlayerInput, react: &PlayerInput) {
        let same = |a: &PlayerInput, b: &PlayerInput| a.direction == b.direction && (a.hook != 0) == (b.hook != 0);
        let (to_react, to_hold) = (same(observed, react), same(observed, hold));
        if to_react && !to_hold {
            self.p += Self::RATE * (Self::MAX - self.p);
        } else if to_hold && !to_react {
            self.p -= Self::RATE * (self.p - Self::MIN);
        }
    }
}

/// Danger flags of one decision (D-042's extension triggers plus the rollout probe).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Danger {
    /// Free opponents (victim included) within the threat radius.
    pub opponents_in_radius: u32,
    /// Freeze or death within [`crate::fields::EDGE_GAP_PX`] of us.
    pub near_freeze: bool,
    /// Some tee holds us with its hook (its id).
    pub hooked_by: Option<i32>,
    /// A probe rollout of the warm plan ended with us out under some modelled response.
    pub probe_self_out: bool,
}

impl Danger {
    /// The explicit extension rule (D-042): two or more opponents within the radius, we are near
    /// freeze, or we are hooked by an enemy.
    pub fn flagged(&self) -> bool {
        self.opponents_in_radius >= 2 || self.near_freeze || self.hooked_by.is_some()
    }

    /// The escape generators run first when danger is flagged and either the probe already saw
    /// us go out or we are being hooked (the hook alone is the emergency).
    pub fn escape_first(&self) -> bool {
        self.probe_self_out || (self.hooked_by.is_some() && self.near_freeze)
    }

    /// Short reason list for telemetry, e.g. `"threats>=2,near_freeze"`.
    pub fn reasons(&self) -> String {
        let mut r: Vec<&str> = Vec::new();
        if self.opponents_in_radius >= 2 {
            r.push("threats>=2");
        }
        if self.near_freeze {
            r.push("near_freeze");
        }
        if self.hooked_by.is_some() {
            r.push("hooked");
        }
        if self.probe_self_out {
            r.push("probe_out");
        }
        r.join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radius_is_hook_length_plus_two_decisions_of_travel() {
        assert_eq!(threat_radius(2, 0), 380.0 + 4.0 * 15.0);
        assert_eq!(threat_radius(2, 3), 380.0 + 7.0 * 15.0);
    }

    #[test]
    fn robust_value_interpolates_between_worst_and_mean() {
        let s = [4.0, 2.0, 0.0];
        assert_eq!(robust_value(&s, 1.0), 0.0);
        assert_eq!(robust_value(&s, 0.0), 2.0);
        assert_eq!(robust_value(&s, 0.5), 1.0);
        assert_eq!(robust_value(&[3.0], 0.7), 3.0);
    }

    #[test]
    fn weighted_value_follows_the_weights_and_keeps_the_worst_case() {
        let s = [4.0, 0.0];
        assert_eq!(robust_value_weighted(&s, &[1.0, 1.0], 0.0), 2.0);
        assert_eq!(robust_value_weighted(&s, &[9.0, 1.0], 0.0), 3.6);
        assert_eq!(robust_value_weighted(&s, &[9.0, 1.0], 1.0), 0.0);
        assert_eq!(robust_value_weighted(&s, &[0.0, 0.0], 0.0), 2.0);
    }

    #[test]
    fn belief_moves_only_on_evidence_and_stays_in_range() {
        let idle = crate::types::empty_input();
        let mut rush = crate::types::empty_input();
        rush.direction = 1;
        rush.hook = 1;
        let mut b = ReactBelief::default();
        // The opponent stands still although the scripted bot would rush: toward hold.
        for _ in 0..40 {
            b.observe(&idle, &idle, &rush);
        }
        assert!(b.p < 0.06 && b.p >= ReactBelief::MIN);
        // It rushes: back toward react.
        for _ in 0..80 {
            b.observe(&rush, &idle, &rush);
        }
        assert!(b.p > 0.94 && b.p <= ReactBelief::MAX);
        // Both predictions agree: no evidence.
        let before = b.p;
        b.observe(&rush, &rush, &rush);
        b.observe(&idle, &rush, &rush);
        assert_eq!(b.p, before);
    }

    #[test]
    fn danger_flags_follow_the_documented_rule() {
        let calm = Danger::default();
        assert!(!calm.flagged());
        assert!(
            Danger {
                opponents_in_radius: 2,
                ..calm
            }
            .flagged()
        );
        assert!(
            !Danger {
                opponents_in_radius: 1,
                ..calm
            }
            .flagged()
        );
        assert!(
            Danger {
                near_freeze: true,
                ..calm
            }
            .flagged()
        );
        assert!(
            Danger {
                hooked_by: Some(3),
                ..calm
            }
            .flagged()
        );
        // The probe alone is not a trigger of the extension, it only orders the generators.
        let probe = Danger {
            probe_self_out: true,
            ..calm
        };
        assert!(!probe.flagged());
        assert!(probe.escape_first());
    }
}
