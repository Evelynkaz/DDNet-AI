//! [`HybridConfig`]: every production-only knob of the hybrid brain. The TS-parity path
//! (`Planner::decide`/`decide_once`) never sees any of it; everything here is a flag of the
//! hybrid decision, documented at its field.

use crate::config::{PlannerConfig, preset_normal};

/// The planner preset the hybrid search scores with: `preset_normal` (the live TS configuration)
/// plus the hybrid-only scoring terms (`0` everywhere else):
/// * `enemy_landing_bonus`: the victim's ballistic landing in a hazard is worth something even
///   when it happens after the rollout horizon (throws and drags);
/// * `landing_cost`: the same forecast for ourselves (`landingCost`, an existing TS term);
/// * `jumpless_anchor_bonus`: hanging on a wall hook while jumpless over a hazard (T14);
/// * `jumpless_air_cost`: ending a rollout in the air with no jump left (T13: a swing that never comes down).
pub fn hybrid_planner_preset() -> PlannerConfig {
    hybrid_terms(preset_normal())
}

/// `base` with the hybrid-only scoring terms of [`hybrid_planner_preset`] switched on.
pub fn hybrid_terms(base: PlannerConfig) -> PlannerConfig {
    PlannerConfig {
        enemy_landing_bonus: 3.0,
        landing_cost: 2.0,
        jumpless_anchor_bonus: 0.1,
        jumpless_air_cost: 0.5,
        ..base
    }
}

/// How a decision spends effort.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HybridMode {
    /// A fixed amount of work per decision: the candidate pool is complete (warm plan, proposals,
    /// book, throws, techniques) and CEM runs its configured iterations. No clock is read, the
    /// decision is a pure function of the state and the seed, and it is identical for any number
    /// of worker threads. The arena's reproducible mode.
    Fixed,
    /// Iterative deepening against a wall-clock (or injected) deadline: `budget_ms` of search
    /// (D-042: 4 ms), plus the adaptive extension when [`AdaptiveConfig`] allows it.
    Deadline { budget_ms: f64 },
}

/// How the re-scored candidates are ranked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RobustMode {
    /// `lambda * worst + (1 - lambda) * mean` over the model combinations.
    #[default]
    Mix,
    /// A plan that no modelled reply leaves us out in beats every plan some reply does; safe
    /// plans compete on the cheap-model score (they keep the attack), the others on the mix.
    SafeFirst,
}

/// The two-stage robust choice (`orig-plan`'s single hold/react opponent model generalised to
/// every combination of the modelled opponents' responses).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustConfig {
    /// `false`: choose by the stage-1 (all opponents hold) score alone.
    pub enabled: bool,
    /// How many of the best stage-1 candidates are re-scored under every response combination.
    pub top_m: usize,
    /// Weight of the worst case in the choice: `lambda * worst + (1 - lambda) * mean`. `1` is pure
    /// max-min. The default 0.5 keeps the bot attacking (pure max-min plays too safe against a
    /// bot that always attacks) while still refusing plans one plausible reply refutes.
    pub lambda: f64,
    pub mode: RobustMode,
    /// How many model combinations a plan is re-scored under at most (1-4; each one is one more
    /// rollout per re-scored plan, so with several opponents this is the main cost of stage 2).
    /// With two, the combinations are "everybody holds" and "everybody reacts".
    pub max_combos: usize,
    /// The re-scoring stage runs only when at most this many opponents can act on us (victim and
    /// threats). Beyond that a 4 ms budget affords 6-9 candidates in a 1v3/1v5 fight and the
    /// re-scoring would take a third of them (measured: more losses in 1v5, E-003), so the search
    /// decides by the cheap model and the threat terms alone.
    pub max_relevant: usize,
    /// Scale the worst-case weight by the belief that the opponents react (an opponent that looks
    /// idle counts less as a threat; measured: T5 36% -> 72%, T10 94% -> 100%, arena strength vs the
    /// planner unchanged, E-003). Without it only the expectation is weighted by the beliefs.
    pub belief_lambda: bool,
    /// A cheap robust stage for crowds (task 3.5b): with more than `max_relevant` opponents able to
    /// act, the best `crowd_top_m` candidates are re-scored under the two extremes only ("everybody
    /// holds" is the stage-1 score, "everybody reacts" the one extra rollout), instead of deciding by
    /// the cheap model alone. `false` = the 3.5 behaviour.
    pub crowd_stage: bool,
    pub crowd_top_m: usize,
}

impl Default for RobustConfig {
    fn default() -> Self {
        RobustConfig {
            enabled: true,
            top_m: 3,
            lambda: 0.5,
            mode: RobustMode::Mix,
            max_combos: 4,
            max_relevant: 2,
            belief_lambda: true,
            crowd_stage: true,
            crowd_top_m: 2,
        }
    }
}

/// D-042's adaptive budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveConfig {
    /// Allow the extension. The search always stops at the base budget when nothing is wrong.
    pub enabled: bool,
    /// Total wall time (from the start of the search) the extension may use, shield included.
    pub max_total_ms: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        AdaptiveConfig {
            enabled: true,
            max_total_ms: 15.0,
        }
    }
}

/// Early pruning (task 3.5b): a cheap short-horizon pre-score decides which of the book plans, throw
/// lines and CEM samples earn a full rollout. Techniques, the warm plan and proposals are never pruned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PruneConfig {
    pub enabled: bool,
    /// Plan steps of the pre-score rollout (a step is 3 ticks; the full plan has 9).
    pub steps: usize,
    /// Share of the pre-scored candidates that go on to a full rollout (by pre-score, among those
    /// seen so far in the decision).
    pub keep: f64,
    /// The first this many pre-scored candidates always go on (there is no distribution yet).
    pub warmup: usize,
}

impl Default for PruneConfig {
    fn default() -> Self {
        PruneConfig {
            enabled: false,
            steps: 3,
            keep: 0.5,
            warmup: 4,
        }
    }
}

/// The hybrid brain's settings.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridConfig {
    /// The planner preset the search scores with (`preset_normal` by default).
    pub planner: PlannerConfig,
    pub mode: HybridMode,
    /// Threads that score candidates, the deciding thread included. `1` = no pool, no thread is
    /// spawned. More threads make a decision faster, never different in [`HybridMode::Fixed`].
    pub workers: usize,
    /// `K`: proposals the search asks its [`crate::hybrid::Proposer`] for.
    pub proposals: usize,
    /// Run the technique library (D-048). `false` = book/CEM/proposals only.
    pub techniques: bool,
    /// Hook anchors kept per decision (angle-diverse, cached per tile).
    pub anchors: usize,
    /// At most this many throw lines (`throwLines`/`frozenThrowLines`, 12 / 28 of them) enter the
    /// pool: they are cheap to generate but each costs a rollout.
    pub throw_cap: usize,
    /// The 1vN threat model: every free opponent within the threat radius is modelled in the
    /// rollouts, scored defensively, part of the danger flags and of the shield. `false` is the
    /// "1v1 model": only the chosen victim is modelled, the others just hold their inputs.
    pub threat_model: bool,
    /// Threat radius in px; `None` derives it (`hook length + two decisions of travel`).
    pub threat_radius_px: Option<f64>,
    /// Weight of the defensive terms of the extra threats relative to the victim's (`1` = equal).
    pub threat_weight: f64,
    /// A hook at an extra threat (not the victim) passes the hook gate (T10 hooks the tee below).
    pub hook_threats: bool,
    /// Ticks between decisions (the live cadence), for the derived radius.
    pub decision_ticks: i32,
    pub robust: RobustConfig,
    pub adaptive: AdaptiveConfig,
    pub prune: PruneConfig,
    /// Share of the deadline budget reserved for the robust re-scoring (stage 2).
    pub stage2_fraction: f64,
    /// Commitment bonus of the warm plan in the final choice (task 3.5b): `0` = none. A fresh plan that waits
    /// one step before it acts ties with the warm plan that already acts and wins the tie-break under some
    /// replies, decision after decision.
    pub warm_bonus: f64,
    /// The bonus goes only to a warm plan whose first step fires (a swing that has begun is finished).
    pub warm_fire_only: bool,
    /// Two-world search (task 3.5b): the pool is scored in a reduced world (us and the victim only) and
    /// only the best few are re-scored in the full world with the threats and their modelled replies, so
    /// the 1vN model keeps the tempo of the 1v1 model and the defence of the threat model.
    pub two_world: bool,
    /// Stage 1 keeps the time stage 2 does not need (task 3.5b): its end moves later when the estimated
    /// cost of stage 2 is below its share, never earlier.
    pub stage2_dynamic: bool,
    /// ... only when at least this many opponents can act on us (the crowd stage's case; a duel's hook timing
    /// (T15a 98% -> 82%) and T18 (67% -> 47%) lose by it, E-007).
    pub stage2_dynamic_min_relevant: usize,
    /// Cap of a decision that is not extended (search + shield, ms): the search budget is
    /// `min(budget_ms, cap - shield reserve)`. `None` = the budget alone (the extension of D-042
    /// may still go up to `adaptive.max_total_ms` when danger is confirmed).
    pub decision_cap_ms: Option<f64>,
    /// Task 3.7a (D-080): the time the proposer took (`proposal_ms`) comes off the cap too, so the search
    /// budget is `min(budget_ms, cap - proposal_ms - shield reserve)`, never below 1 ms, and the extension of D-042
    /// counts from the start of the proposals. `false` = the 3.5-3.6 behaviour (the search starts its full budget
    /// after the proposals). A proposer that does nothing (`NoProposer`) costs nothing either way.
    pub proposal_in_cap: bool,
    /// Tees the rollouts simulate at most (task 3.5b, F2): us, the victim, whoever hooks us, then the
    /// nearest threats and the nearest frozen body; every other tee is dropped from the decision
    /// world, so a crowd costs like a fight. `0` = every tee (the 3.5 behaviour).
    pub max_sim_tees: usize,
    /// Shield (task 3.5b): skipped when the nearest freeze or death tile is at least this many tiles
    /// away along the tee's own path (`0` = never skip). A tee 14 tiles from any hazard cannot reach
    /// one within the shield's horizon, so the check would only cost time.
    pub shield_skip_tiles: u32,
    /// Shield (task 3.5b): the remainder of the chosen plan, rolled out exactly, is the shield's first
    /// escape (so a hook plan is not judged "no escape" by an escape model that only walks and jumps).
    pub shield_plan_escape: bool,
    /// Shield (task 3.5b): hook escapes aimed at up to this many anchors (walls, ceilings), each with
    /// and without a jump, tried after the standard escapes. `0` = none.
    pub shield_hook_anchors: usize,
    /// Shield (task 3.5b): a timed-out `escapeExists` counts as danger, so that `saferInput` gets the
    /// extension budget (needs `adaptive.enabled`). Off, only a confirmed "no escape" does.
    pub shield_timeout_danger: bool,
    /// Time the shield may use after the search, per tee in the world (ms); the search itself
    /// keeps its own budget. The shield gives up (`shield_incomplete`) when it is spent, except
    /// that a confirmed danger lets it run on up to the adaptive cap (D-042).
    pub shield_reserve_ms_per_tee: f64,
    /// Deadline mode on the **work clock**: microseconds charged per tee-tick (one physics tick of
    /// one tee; an anchor ray counts as two), instead of wall time. `None` = the wall clock (or the step clock a test
    /// injects). Reproducible and load-independent; with `workers > 1` the helpers only speculate, the result is
    /// bit-identical to `workers = 1` (task 3.7a). See [`crate::hybrid::work`].
    pub work_clock_us_per_tick: Option<f64>,
    /// Task 3.10 (finishing, opt-in): generate the offensive technique families against a frozen victim too
    /// ([`crate::hybrid::techniques::TechCaps::frozen_offence`]). `false` = T7/T8 only, as before.
    pub finish_families: bool,
    /// Keep a count of the physics ticks each phase simulates (D-045). Cheap; on by default.
    pub count_work: bool,
    /// Diagnostics: put the best candidates of every decision, with their scores, into the
    /// telemetry (`dump`). Off by default (it allocates and bloats the JSON).
    pub debug_dump: bool,
    /// Task 3.7b (the opponent model, E-017): predict what the victim does next by a small search *from the victim's seat* -- the
    /// planner's own pool of book seeds, its last plan one step on, and `mirror_samples` CEM samples, scored against what we keep
    /// doing -- and let its best plan's inputs replace "the victim holds its input" in every rollout of the cheap model. Runs only
    /// in a duel (the victim is free and within the threat radius, no other free opponent is), while we are free and the victim has
    /// not been passive (neutral direction, no hook out) for six decisions; its rollouts count against the decision cap. `false` =
    /// the 3.5-3.7a behaviour (hold). On by default since task 3.7b (D-090, E-017).
    pub mirror: bool,
    pub mirror_samples: usize,
    /// Diagnostics (task 3.7b, `crate::diag`): keep the whole pool of every decision, plans and scores, in the
    /// telemetry (`DecisionTelemetry::pool`). Off by default: it allocates and changes nothing else.
    pub debug_pool: bool,
    // --- Task 3.9 (D-096, E-020): the competitor's current planner (upstream af49dfb, "v2") inside the hybrid. Every switch below is
    // OFF by default, so the default hybrid is bit-identical to the one of task 3.7b. The four v2 planner switches
    // (`planner.hook_exact_gate`, `hook_snap_aim`, `hook_keep_flying`, `rope_ceiling_cost`) and its live scoring values
    // (`planner.launch_exposure`, `jumpless_hazard_cost`) are fields of [`HybridConfig::planner`] itself: the rollouts, the final
    // hook gate and the opponent model read them from there (see `PlannerConfig::with_version`).
    /// Task 3.9: after CEM, the best candidate is "polished" the way `polishRope` does it -- variants of it that hold the hook (and
    /// aim the throw) for its first 2, 4 or all steps (1, 2, 3 while our hook is in flight, with `planner.hook_keep_flying`) join
    /// the pool and are scored like any other candidate. At most three more rollouts, only while a hook is out or the victim is in reach.
    pub polish: bool,
    /// Task 3.9: against a frozen victim (the case of `frozen_throw_lines`), also offer the wayblock guard's wall swings (and,
    /// with `planner.air_chain`, the air chains while airborne) toward a solid wall within [`WALL_REACH_TILES`] tiles at our
    /// height (the nearer side; none without a wall). Unlike the competitor's `wallDir` (a hall's side, set by role) the side
    /// is found per decision.
    pub wall_throws: bool,
    /// Task 3.9: the planner the opponent model ("mirror") runs in the victim's seat; `None` = [`preset_normal`] (the old planner,
    /// what the model has always been). `Some(preset_normal_v2())` models the competitor's current planner. Its steps layout is
    /// always the hybrid's own.
    pub mirror_planner: Option<PlannerConfig>,
    /// Task 3.14 (E-026, opt-in): the planning world's roll through the input-lag window (the ticks between the snapshot and the first tick our decision
    /// can act on) plays the victim by what the opponent model predicted for those ticks at the last decisions (the recorded plans of
    /// [`HybridConfig::mirror`], newest first) instead of "it keeps the input its snapshot shows". Needs `mirror`; with no lag, or no prediction for a tick, the
    /// victim holds its input as before.
    pub lag_mirror: bool,
    /// Task 3.15 (E-028, opt-in): the roll through the input-lag window plays the victim by what the learned window model
    /// ([`crate::hybrid::window::WindowModel`], set with `HybridBrain::set_window_model`) predicts for each tick, instead of "it keeps the input its snapshot
    /// shows". Without a model, or with no lag, nothing changes. It takes precedence over `lag_mirror` for the ticks it predicts.
    pub window_model: bool,
    // --- Task 3.10b (D-110, E-030): closing the escapes after a freeze. Every switch below is OFF by default (the default hybrid is bit-identical to
    // the one before), and none of them changes anything while the victim is not frozen (or while we are).
    /// Task 3.10b (a), the port of upstream's `frozenTargetSteps`: while the victim is frozen with at least [`Self::frozen_steps_min_ticks`] ticks of
    /// freeze left, every plan of the decision has this many steps (the competitor's wayblock guard uses 16: 48 ticks instead of 27). `0` = off,
    /// else it must exceed `planner.steps`. A longer plan costs proportionally more per rollout, so fewer candidates fit the budget.
    pub frozen_steps: i32,
    /// Task 3.10b (a): the freeze must still last this many ticks for the longer horizon to be used (upstream's `FROZEN_PLAN_MIN_TICKS`, 30).
    pub frozen_steps_min_ticks: i32,
    /// Task 3.10b (a): the search budget (ms, the same clock as `mode`'s) of a decision that uses the longer horizon, `None` = the mode's own. The opponent
    /// model does not run while the victim is frozen, so the 1.4 ms or so it takes in a duel decision is free under the decision cap (D-042: 5 ms): a
    /// 16-step rollout costs 1.8 times a 9-step one, and a budget of 4 ms leaves the CEM nothing. Only [`HybridMode::Deadline`]; the cap still applies
    /// (`decision_cap_ms` less the shield's reserve).
    pub frozen_budget_ms: Option<f64>,
    /// Task 3.10b (a): a decision of the longer horizon does not use D-042's adaptive extension (the search's, nor the shield's `safer_input`): a 48-tick rollout
    /// sees more of our own falls than a 27-tick one, flags danger more often and the extension then runs to 15 ms (measured: work p99 9.7 ms at 2 tees against 4.3
    /// without). Default `true`; it matters only with `frozen_steps` on.
    pub frozen_no_extension: bool,
    /// Task 3.10b (b): at most this many approach-then-push plans (technique T30, [`crate::hybrid::techniques::approach_plans`]) join the pool while the
    /// victim lies frozen off the freeze with freeze left: leap over it and hook it from the far side, or walk up and hook-pull it toward the freeze. Never a hammer: a hammer hit unfreezes the tee it hits.
    /// `0` = off. Each one costs a rollout.
    pub approach_plans: usize,
}

/// The longest plan the hybrid can be asked for (`frozen_steps`).
pub const MAX_PLAN_STEPS: i32 = 32;

/// How far (tiles) a wall may be for `wall_throws` to offer the wall swings.
pub const WALL_REACH_TILES: i32 = 5;

/// A finite number above zero (a NaN or an infinity from a config file is refused).
fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

impl Default for HybridConfig {
    fn default() -> Self {
        HybridConfig {
            planner: hybrid_planner_preset(),
            mode: HybridMode::Deadline { budget_ms: 4.0 },
            workers: 1,
            proposals: 3,
            techniques: true,
            anchors: 8,
            throw_cap: 10,
            threat_model: true,
            threat_radius_px: None,
            threat_weight: 0.25,
            hook_threats: true,
            decision_ticks: 2,
            robust: RobustConfig::default(),
            adaptive: AdaptiveConfig::default(),
            prune: PruneConfig::default(),
            stage2_fraction: 0.35,
            stage2_dynamic: true,
            stage2_dynamic_min_relevant: 3,
            two_world: false,
            warm_bonus: 0.3,
            warm_fire_only: false,
            decision_cap_ms: Some(5.0),
            proposal_in_cap: true,
            max_sim_tees: 4,
            shield_skip_tiles: 14,
            shield_plan_escape: true,
            shield_hook_anchors: 3,
            shield_timeout_danger: false,
            shield_reserve_ms_per_tee: 0.25,
            work_clock_us_per_tick: None,
            finish_families: false,
            count_work: true,
            debug_dump: false,
            debug_pool: false,
            mirror: true,
            mirror_samples: 12,
            polish: false,
            wall_throws: false,
            mirror_planner: None,
            lag_mirror: false,
            window_model: false,
            frozen_steps: 0,
            frozen_steps_min_ticks: 30,
            frozen_budget_ms: None,
            frozen_no_extension: true,
            approach_plans: 0,
        }
    }
}

/// Task 3.10: the weight of the frozen-victim drag shaping the finishing switch turns on (`PlannerConfig::frozen_drag_weight`).
pub const FINISH_DRAG_WEIGHT: f64 = 20.0;

impl HybridConfig {
    /// Task 3.10 (opt-in, `--finish full`): this configuration with the finishing switches of the hybrid on -- the frozen-victim drag
    /// shaping ([`FINISH_DRAG_WEIGHT`]) only. `finish_families` (the offensive techniques against a frozen victim), `frozen_stage_weight` (the staging
    /// point) and `held_forecast_weight` (the exact forecast) stay separate knobs: no gain that holds up in E-021. The bot's target logic has its own switch
    /// (`BotConfig::finish`); `--finish full` sets both. Measured in E-021.
    pub fn with_finish(mut self) -> HybridConfig {
        self.planner.frozen_drag_weight = FINISH_DRAG_WEIGHT;
        self
    }

    /// A fixed-work (deterministic) configuration.
    pub fn fixed() -> HybridConfig {
        HybridConfig {
            mode: HybridMode::Fixed,
            adaptive: AdaptiveConfig {
                enabled: false,
                ..AdaptiveConfig::default()
            },
            ..HybridConfig::default()
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.workers == 0 {
            return Err("hybrid: workers must be at least 1".into());
        }
        if let HybridMode::Deadline { budget_ms } = self.mode
            && !positive(budget_ms)
        {
            return Err("hybrid: budget_ms must be positive".into());
        }
        if !(0.0..=1.0).contains(&self.robust.lambda) || !(0.0..1.0).contains(&self.stage2_fraction) {
            return Err("hybrid: robust.lambda in [0,1], stage2_fraction in [0,1)".into());
        }
        if self.work_clock_us_per_tick.is_some_and(|u| !positive(u)) {
            return Err("hybrid: the work clock needs us_per_tee_tick > 0".into());
        }
        if self.robust.max_relevant > crate::hybrid::threat::MAX_THREATS + 1 {
            return Err("hybrid: robust.max_relevant above the number of modelled opponents".into());
        }
        if !(1..=4).contains(&self.robust.max_combos)
            || !(self.threat_weight.is_finite() && self.threat_weight >= 0.0)
            || !positive(self.shield_reserve_ms_per_tee)
            || self.decision_cap_ms.is_some_and(|c| !positive(c))
        {
            return Err("hybrid: robust.max_combos in 1..=4, threat_weight >= 0, shield_reserve_ms_per_tee > 0, decision_cap_ms > 0".into());
        }
        if !positive(self.adaptive.max_total_ms) || self.threat_radius_px.is_some_and(|r| !positive(r)) {
            return Err("hybrid: adaptive.max_total_ms and threat_radius_px must be finite and positive".into());
        }
        if self.shield_hook_anchors > 8 {
            return Err("hybrid: shield_hook_anchors above 8".into());
        }
        if self.prune.steps == 0
            || !(self.prune.keep > 0.0 && self.prune.keep <= 1.0)
            || self.prune.steps >= self.planner.steps.max(1) as usize
        {
            return Err("hybrid: prune.steps in 1..plan steps, prune.keep in (0, 1]".into());
        }
        if self.frozen_steps != 0
            && (self.frozen_steps <= self.planner.steps
                || self.frozen_steps > MAX_PLAN_STEPS
                || self.frozen_steps_min_ticks < 0)
        {
            return Err("hybrid: frozen_steps must be 0 or in (planner.steps, 32], frozen_steps_min_ticks >= 0".into());
        }
        if self.frozen_budget_ms.is_some_and(|b| !positive(b)) {
            return Err("hybrid: frozen_budget_ms must be finite and positive".into());
        }
        if self.anchors > 36 {
            return Err("hybrid: anchors above the ray count".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_d042_shape() {
        let c = HybridConfig::default();
        assert_eq!(c.mode, HybridMode::Deadline { budget_ms: 4.0 });
        assert_eq!(c.adaptive.max_total_ms, 15.0);
        assert!(c.validate().is_ok());
        assert_eq!(HybridConfig::fixed().mode, HybridMode::Fixed);
        assert!(!HybridConfig::fixed().adaptive.enabled);
    }

    #[test]
    fn validation_rejects_nonsense() {
        let bad = |f: &dyn Fn(&mut HybridConfig)| {
            let mut c = HybridConfig::default();
            f(&mut c);
            assert!(c.validate().is_err());
        };
        bad(&|c| c.workers = 0);
        bad(&|c| c.mode = HybridMode::Deadline { budget_ms: 0.0 });
        bad(&|c| c.robust.lambda = 1.5);
        bad(&|c| c.stage2_fraction = 1.0);
        bad(&|c| c.robust.max_relevant = 20);
        bad(&|c| c.anchors = 100);
    }
}
