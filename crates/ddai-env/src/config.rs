//! Run configuration (TOML): rules, conditions (arena + players + brains), seeds. The *effective*
//! config (after CLI overrides) is what gets hashed into the run record.

use std::collections::BTreeMap;

use ddai_brain::Brain;
use ddai_planner::brains::{
    ClockKind, IdleBrain, PlannerBrain, PlannerBrainConfig, PlannerMode, PlannerPreset, ScriptedBrain,
};
use ddai_planner::hybrid::{
    HybridBrain, HybridConfig, HybridMode, NoProposer, Proposer, ScriptedProposer, hybrid_terms,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::EnvError;
use crate::arena::hex;

fn d_max_ticks() -> i32 {
    1500
}
fn d_after_ticks() -> i32 {
    150
}
fn d_decide_every() -> i32 {
    2
}
fn d_credit_ticks() -> i32 {
    50
}
fn d_games() -> u32 {
    200
}
fn d_seed() -> u64 {
    1
}
fn d_count() -> u32 {
    1
}

/// Game rules; the defaults are the phase-0 harness's (`orig-run.md` §3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rules {
    /// No one out by this tick: a timeout (`T`). 1500 ticks = 30 s.
    #[serde(default = "d_max_ticks")]
    pub max_ticks: i32,
    /// Ticks played on after the deciding tick, to judge `held`.
    #[serde(default = "d_after_ticks")]
    pub after_ticks: i32,
    /// Brains decide every this many ticks (the live snapshot cadence, 25 Hz).
    #[serde(default = "d_decide_every")]
    pub decide_every: i32,
    /// A victim counts as `credited` when the winner hooked or hammered it at most this many
    /// ticks before the onset (`BLOCK_CREDIT_TICKS`).
    #[serde(default = "d_credit_ticks")]
    pub credit_ticks: i32,
    /// Whether an opponent going out ends the game only when credited to the focal player. `None`
    /// = automatic: `false` for 1v1 (the harness rule: any onset decides), `true` for 1vN (an
    /// uncredited opponent is recorded and the game continues).
    #[serde(default)]
    pub credit_required: Option<bool>,
    /// Crowds (task 3.5b): tees from this slot on spawn anywhere on the arena's standing cells, at
    /// least `min_tiles` from every tee already placed, instead of within `max_tiles` of the focal
    /// player. `None` = every tee spawns near the focal player (the 1vN rule of E-002).
    #[serde(default)]
    pub crowd_from: Option<usize>,
    /// Crowds: the least distance in tiles between two crowd tees (default `min_tiles`, 3; a dense
    /// hub of 12-15 tees does not fit an E-002 box at that spacing).
    #[serde(default)]
    pub crowd_spacing: Option<f64>,
    /// Task 3.10 (finishing, opt-in): the focal player's target rule keeps a frozen current target until the passive forecast says it stays
    /// out for the held-block window ([`crate::sim::HoldTarget`], the arena's counterpart of the bot's `--finish target` target logic); the
    /// default rule takes a free opponent first and a frozen one last, so in a crowd the victim of a first freeze is left at once.
    #[serde(default)]
    pub hold_target: bool,
}

impl Default for Rules {
    fn default() -> Self {
        Rules {
            max_ticks: d_max_ticks(),
            after_ticks: d_after_ticks(),
            decide_every: d_decide_every(),
            credit_ticks: d_credit_ticks(),
            credit_required: None,
            crowd_from: None,
            crowd_spacing: None,
            hold_target: false,
        }
    }
}

/// The window of the held-block metric (task 3.10, D-059 amendment): 250 ticks = 5 s, more than `sv_freeze_delay` (3 s).
pub const HELD_BLOCK_TICKS: i32 = 250;

impl Rules {
    /// These rules with the held-block window: the game is played on for [`HELD_BLOCK_TICKS`] after the deciding freeze, so
    /// `GameReport::held_block` and `victim_out_ticks` mean something. An episode of a training run that should see (and be rewarded for)
    /// what happens after the first freeze uses this too (`collect_game` plays the whole window and labels it).
    pub fn held_block_window(self) -> Rules {
        Rules {
            after_ticks: self.after_ticks.max(HELD_BLOCK_TICKS),
            ..self
        }
    }

    pub fn validate(&self) -> Result<(), EnvError> {
        if self.max_ticks <= 0 || self.after_ticks < 0 || self.decide_every <= 0 || self.credit_ticks < 0 {
            return Err(EnvError::new(
                "rules: max_ticks and decide_every must be positive, the others non-negative",
            ));
        }
        Ok(())
    }
}

/// Per-condition rule overrides: every field left out inherits the run-level value.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RulesOverride {
    #[serde(default)]
    pub max_ticks: Option<i32>,
    #[serde(default)]
    pub after_ticks: Option<i32>,
    #[serde(default)]
    pub decide_every: Option<i32>,
    #[serde(default)]
    pub credit_ticks: Option<i32>,
    #[serde(default)]
    pub credit_required: Option<bool>,
    #[serde(default)]
    pub crowd_from: Option<usize>,
    #[serde(default)]
    pub crowd_spacing: Option<f64>,
    #[serde(default)]
    pub hold_target: Option<bool>,
}

impl RulesOverride {
    /// `base` with this override's set fields applied.
    pub fn apply(&self, base: &Rules) -> Rules {
        Rules {
            max_ticks: self.max_ticks.unwrap_or(base.max_ticks),
            after_ticks: self.after_ticks.unwrap_or(base.after_ticks),
            decide_every: self.decide_every.unwrap_or(base.decide_every),
            credit_ticks: self.credit_ticks.unwrap_or(base.credit_ticks),
            credit_required: self.credit_required.or(base.credit_required),
            crowd_from: self.crowd_from.or(base.crowd_from),
            crowd_spacing: self.crowd_spacing.or(base.crowd_spacing),
            hold_target: self.hold_target.unwrap_or(base.hold_target),
        }
    }
}

/// One player slot (or `count` identical slots). The focal player is the first slot of a
/// condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerSpec {
    /// `idle`, `scripted` or `planner` (built in); a CLI may add more (e.g. `fly`).
    pub brain: String,
    /// Repeat this spec for that many consecutive slots (1v3 = `count = 3` on the attacker).
    #[serde(default = "d_count")]
    pub count: u32,
    /// Input lag of this client in ticks (decisions reach the world this many ticks late).
    #[serde(default)]
    pub lag: u32,
    /// Planner: `normal` (default), `low` or `strong`; `normal-v2` and `live-v2` are the competitor's current planner
    /// (upstream af49dfb, task 3.8; the second with its `LIVE_PLANNER_CFG`); `v2-strong` is `normal-v2` with the competitor's strong
    /// mode (40 x 3 search) everywhere and `v2-strong-wb` the same on its wayblock overrides, `live-v2-strong` its `LIVE_PLANNER_CFG` with the 40 x 3 search, `strong-fixed` the old `strong` without its wall-clock budget (task 3.9; all fixed iterations). The hybrid takes the
    /// first three only (its v2 switches are `[hybrid]` knobs).
    #[serde(default)]
    pub preset: Option<String>,
    /// Planner: `fixed` (default, deterministic) or `deadline`.
    #[serde(default)]
    pub mode: Option<String>,
    /// Planner deadline mode: milliseconds per decision.
    #[serde(default)]
    pub budget_ms: Option<f64>,
    /// Planner deadline mode: `wall` (default) or `step` (deterministic fake clock, tests).
    #[serde(default)]
    pub clock: Option<String>,
    /// Planner deadline mode with `clock = "step"`: milliseconds per clock read.
    #[serde(default)]
    pub step_ms: Option<f64>,
    /// Brains loaded from a file (`fly`): path of the model.
    #[serde(default)]
    pub model: Option<String>,
    /// Model brains: `argmax` (default) or `sampled` (each head drawn from its probabilities).
    #[serde(default)]
    pub select: Option<String>,
    /// `hybrid`: the hybrid brain's own settings (`preset`, `mode`, `budget_ms`, `clock` and
    /// `step_ms` above apply to it as well).
    #[serde(default)]
    pub hybrid: Option<HybridSpec>,
    /// Display label; defaults to the brain's own name.
    #[serde(default)]
    pub label: Option<String>,
    /// Task 4.2: tell this player what the live bot's wayblock hook tells its planner (the hall's
    /// `WB_PLAN_OVERRIDES` and band) -- only on a wayblock arena (`[wayblock]` in its definition).
    #[serde(default)]
    pub wb: bool,
    /// With `wb`: strong mode (`STRONG_WB` in the hall).
    #[serde(default)]
    pub wb_strong: bool,
    /// Task 3.16 (D-115): an input lag that follows the cost of each decision ([`crate::sim::LagModel`]) instead of the fixed `lag`. Needs a brain
    /// that reports its decision cost (`PlanTelemetry::decision_us`: the hybrid; with `clock = "work"` the cost is the work clock's, reproducible).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lag_model: Option<LagModelSpec>,
}

/// The `lag_model` of a [`PlayerSpec`]; see [`crate::sim::LagModel`] for the meaning of the numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LagModelSpec {
    /// The lag (arena `lag`, ticks) of a decision that costs nothing is `ceil(base_ms / 20) - 1`; `base_ms` is the live link's `RTT + margin + the phase constant`.
    pub base_ms: f64,
    /// What the live path costs that the arena does not charge (queue hop, the driver's pick-up, the fly's proposals), ms. Default 0.
    #[serde(default)]
    pub extra_ms: f64,
    /// Snapshot arrival jitter, ms (uniform `+-`). Default 0.
    #[serde(default)]
    pub jitter_ms: f64,
    /// The decision-cost estimate before the first decision, ms. Default 6 (the live bot's start value).
    #[serde(default = "d_lag_initial")]
    pub initial_ms: f64,
    /// A deadline-aware search (opt-in): when the first slot leaves a decision at least this many ms, the brain is told to finish in them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline_floor_ms: Option<f64>,
}

fn d_lag_initial() -> f64 {
    6.0
}

impl LagModelSpec {
    /// `fixed_lag` is the player's `lag`: a model replaces it, so asking for both is a mistake.
    pub fn validate(&self, fixed_lag: u32) -> Result<(), String> {
        let ok = |x: f64| x.is_finite() && x >= 0.0;
        if !(ok(self.base_ms)
            && ok(self.extra_ms)
            && ok(self.jitter_ms)
            && ok(self.initial_ms)
            && self.deadline_floor_ms.is_none_or(|f| f.is_finite() && f > 0.0))
        {
            return Err("base_ms, extra_ms, jitter_ms and initial_ms must be finite and non-negative".into());
        }
        if fixed_lag != 0 {
            return Err(
                "the player has both `lag` and `lag_model`; the model replaces the fixed lag, set only one".into(),
            );
        }
        Ok(())
    }

    pub fn model(&self) -> crate::sim::LagModel {
        let mut m = crate::sim::LagModel::new(self.base_ms, self.extra_ms, self.jitter_ms, self.initial_ms);
        m.deadline_floor_ms = self.deadline_floor_ms;
        m
    }
}

/// The input-lag models of a condition's slots (`None` where a slot has no `lag_model`), for [`crate::game::play_game_modeled`].
pub fn lag_models_of(slots: &[PlayerSpec]) -> Vec<Option<crate::sim::LagModel>> {
    slots
        .iter()
        .map(|s| s.lag_model.as_ref().map(LagModelSpec::model))
        .collect()
}

impl PlayerSpec {
    pub fn simple(brain: &str) -> PlayerSpec {
        PlayerSpec {
            brain: brain.to_string(),
            count: 1,
            lag: 0,
            preset: None,
            mode: None,
            budget_ms: None,
            clock: None,
            step_ms: None,
            model: None,
            select: None,
            hybrid: None,
            label: None,
            wb: false,
            wb_strong: false,
            lag_model: None,
        }
    }
}

/// Settings of the `hybrid` brain (task 3.5); every field left out keeps the
/// [`HybridConfig`] default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridSpec {
    /// `none` (default), `scripted`; `fly` is added by the CLI (it needs the model file).
    #[serde(default)]
    pub proposer: Option<String>,
    /// `K`: proposals per decision.
    #[serde(default)]
    pub proposals: Option<usize>,
    /// Threads scoring candidates (the deciding thread included).
    #[serde(default)]
    pub workers: Option<usize>,
    #[serde(default)]
    pub techniques: Option<bool>,
    /// `false` = the 1v1 model (only the chosen victim is modelled).
    #[serde(default)]
    pub threat_model: Option<bool>,
    #[serde(default)]
    pub threat_radius_px: Option<f64>,
    /// Weight of the extra threats' defensive terms (default 0.25).
    #[serde(default)]
    pub threat_weight: Option<f64>,
    /// Cap of a non-extended decision (search + shield), ms; 0 = none (default 5).
    #[serde(default)]
    pub decision_cap_ms: Option<f64>,
    /// The proposer's time comes off the decision cap too (task 3.7a, D-080; default true; `false` = the 3.5-3.6 behaviour).
    #[serde(default)]
    pub proposal_in_cap: Option<bool>,
    /// Shield time after the search, ms per tee in the world (default 0.25).
    #[serde(default)]
    pub shield_reserve_ms_per_tee: Option<f64>,
    /// The cheap robust stage for crowds (default false) and how many candidates it re-scores.
    #[serde(default)]
    pub crowd_stage: Option<bool>,
    #[serde(default)]
    pub crowd_top_m: Option<usize>,
    /// Early pruning by a short-horizon pre-score (default false), its steps, kept share and warm-up.
    #[serde(default)]
    pub prune: Option<bool>,
    #[serde(default)]
    pub prune_steps: Option<usize>,
    #[serde(default)]
    pub prune_keep: Option<f64>,
    #[serde(default)]
    pub prune_warmup: Option<usize>,
    /// Tees the rollouts simulate at most (default 6; 0 = all).
    #[serde(default)]
    pub max_sim_tees: Option<usize>,
    /// Shield skipped when no hazard is this many tiles away (default 14; 0 = never).
    #[serde(default)]
    pub shield_skip_tiles: Option<u32>,
    /// Shield: the remainder of the chosen plan is its first escape (default true).
    #[serde(default)]
    pub shield_plan_escape: Option<bool>,
    /// Shield: hook escapes at this many anchors (default 3; 0 = none).
    #[serde(default)]
    pub shield_hook_anchors: Option<usize>,
    /// Shield: a timed-out escape check counts as danger (default false).
    #[serde(default)]
    pub shield_timeout_danger: Option<bool>,
    /// Whether hooking an extra threat passes the hook gate (default true).
    #[serde(default)]
    pub hook_threats: Option<bool>,
    /// Scale the worst-case weight by the reaction belief (default true).
    #[serde(default)]
    pub belief_lambda: Option<bool>,
    /// Re-score only when at most this many opponents can act on us (default 2).
    #[serde(default)]
    pub max_relevant: Option<usize>,
    /// Model combinations re-scored per plan, 1-4 (default 4).
    #[serde(default)]
    pub max_combos: Option<usize>,
    /// Two-stage robust choice on/off, its worst-case weight, and how many candidates it re-scores.
    #[serde(default)]
    pub robust: Option<bool>,
    #[serde(default)]
    pub lambda: Option<f64>,
    /// `mix` (default) or `safe` (safe plans first, see `RobustMode`).
    #[serde(default)]
    pub robust_mode: Option<String>,
    #[serde(default)]
    pub top_m: Option<usize>,
    /// The D-042 extension on/off and its total cap in ms.
    #[serde(default)]
    pub adaptive: Option<bool>,
    #[serde(default)]
    pub max_total_ms: Option<f64>,
    #[serde(default)]
    pub stage2_fraction: Option<f64>,
    /// Fine early plan steps: the first `front_steps` steps last `front_step` ticks (< `plan_step`), the rest
    /// share the remaining ticks of the horizon (the planner's `frontSteps`/`frontStep`).
    #[serde(default)]
    pub front_steps: Option<f64>,
    #[serde(default)]
    pub front_step: Option<f64>,
    /// The opponent model of task 3.7b: predict the victim's plan by a small search from its seat (default true; `false` = the victim
    /// holds its input, as before 3.7b), and how many CEM samples that search draws besides its book seeds and its last plan (default 12).
    #[serde(default)]
    pub mirror: Option<bool>,
    #[serde(default)]
    pub mirror_samples: Option<usize>,
    /// Task 3.10 (finishing, opt-in, default 0 = off): the shaping weight of a frozen victim's progress toward a freeze/death tile
    /// (`PlannerConfig::frozen_drag_weight`), and the weight of the exact passive forecast of how long the frozen victim stays out
    /// (`PlannerConfig::held_forecast_weight`).
    #[serde(default)]
    pub frozen_drag_weight: Option<f64>,
    #[serde(default)]
    pub held_forecast_weight: Option<f64>,
    /// Task 3.10 (opt-in, default false): the offensive technique families against a frozen victim too (`HybridConfig::finish_families`).
    #[serde(default)]
    pub finish_families: Option<bool>,
    /// Task 3.10 (opt-in, default 0): per-tile reward for the progress toward the staging point behind a frozen victim (`PlannerConfig::frozen_stage_weight`).
    #[serde(default)]
    pub frozen_stage_weight: Option<f64>,
    /// Task 3.10b (c, opt-in, default 0 = off): per-tick reward for the frozen victim touching a freeze tile in the rollout (`PlannerConfig::frozen_seal_weight`).
    #[serde(default)]
    pub frozen_seal_weight: Option<f64>,
    /// Task 3.10b (c, opt-in, default 0 = off): the exact-forecast check of a rollout's seal (`PlannerConfig::sealed_forecast_weight`).
    #[serde(default)]
    pub sealed_forecast_weight: Option<f64>,
    /// Task 3.10b (d, opt-in, default 0 = off): per-tick cost while both we and the victim are frozen (`PlannerConfig::mutual_freeze_cost`).
    #[serde(default)]
    pub mutual_freeze_cost: Option<f64>,
    /// Task 3.10b (opt-in, default 0 = off): the plan length while the victim is frozen with enough freeze left (`HybridConfig::frozen_steps`,
    /// upstream's `frozenTargetSteps`, 16) and the freeze ticks it needs (`HybridConfig::frozen_steps_min_ticks`, default 30).
    #[serde(default)]
    pub frozen_steps: Option<i32>,
    #[serde(default)]
    pub frozen_steps_min_ticks: Option<i32>,
    /// Task 3.10b (a, opt-in): the search budget (ms) of a decision that uses the longer horizon (`HybridConfig::frozen_budget_ms`).
    #[serde(default)]
    pub frozen_budget_ms: Option<f64>,
    /// Task 3.10b (a, default true): the long-horizon decisions skip D-042's adaptive extension (`HybridConfig::frozen_no_extension`).
    #[serde(default)]
    pub frozen_no_extension: Option<bool>,
    /// Task 3.10b (opt-in, default 0 = off): at most this many approach-then-push plans against a frozen victim off the freeze (`HybridConfig::approach_plans`).
    #[serde(default)]
    pub approach_plans: Option<usize>,
    /// Samples per CEM iteration and CEM iterations of the search (planner presets: 20 and 2): a diagnostic knob (task 3.7b).
    #[serde(default)]
    pub cem_population: Option<i32>,
    #[serde(default)]
    pub cem_iterations: Option<i32>,
    /// Plan steps and ticks per step of the search (default 9 x 3 = 27 ticks ahead): a shorter horizon
    /// makes every rollout cheaper.
    #[serde(default)]
    pub plan_steps: Option<i32>,
    #[serde(default)]
    pub plan_step_ticks: Option<i32>,
    /// Commitment bonus of the warm plan in the final choice (default 0).
    #[serde(default)]
    pub warm_bonus: Option<f64>,
    /// The warm bonus goes only to a warm plan that fires now (default false).
    #[serde(default)]
    pub warm_fire_only: Option<bool>,
    /// Two-world search: pool scored with us and the victim only, the best few re-scored with the threats
    /// (default false).
    #[serde(default)]
    pub two_world: Option<bool>,
    /// Stage 1 keeps the time stage 2 does not need (default false).
    #[serde(default)]
    pub stage2_dynamic: Option<bool>,
    /// ... only with at least this many opponents able to act (default 3).
    #[serde(default)]
    pub stage2_dynamic_min_relevant: Option<usize>,
    #[serde(default)]
    pub anchors: Option<usize>,
    #[serde(default)]
    pub throw_cap: Option<usize>,
    /// `proposer = "fly"` (CLI only): the brain config of the fly (`configs/fly/{S,M}-brain.toml`);
    /// the `.flyg` graph is the player's `model`. The fly is untrained (plumbing and cost only).
    #[serde(default)]
    pub fly_config: Option<String>,
    /// `proposer = "fly"`: a **trained** fly (an 8.2 `.bundle`) instead of the untrained one built
    /// from `model` (a `.flyg`); the bundle carries its own brain config and graph reference.
    /// CLI: `--brain hybrid:fly:<bundle>`.
    #[serde(default)]
    pub fly_model: Option<String>,
    /// `proposer = "mlp"` / `"gru"`: the trained control bundle that proposes (8.2b). CLI: `--brain hybrid:mlp:<bundle>`.
    #[serde(default)]
    pub control_model: Option<String>,
    /// Diagnostics: candidates and scores in the telemetry (`ddnet-ai arena scenarios --trace`).
    #[serde(default)]
    pub debug_dump: Option<bool>,
    /// The two hybrid-only scoring terms (defaults 3.0 and 2.0; `0` switches one off).
    #[serde(default)]
    pub enemy_landing_bonus: Option<f64>,
    #[serde(default)]
    pub landing_cost: Option<f64>,
    /// Bonus for hanging on a wall hook while jumpless over a hazard (default 0.1).
    #[serde(default)]
    pub jumpless_anchor_bonus: Option<f64>,
    /// Cost of ending a rollout airborne with no jumps left (hybrid only; default 0).
    #[serde(default)]
    pub jumpless_air_cost: Option<f64>,
    // --- Task 3.9 (D-096, E-020): the competitor's current planner (af49dfb, "v2") inside the hybrid. All default off.
    /// `true`: the four v2 planner switches below on at once (`PlannerConfig::with_version(Upstream20261002)`); the single knobs that
    /// follow then override it.
    #[serde(default)]
    pub v2: Option<bool>,
    /// The hook gate asks whether the rope meets the victim's projected position (`hookExactGate`), in the rollouts and in the chosen input.
    #[serde(default)]
    pub hook_exact_gate: Option<bool>,
    /// A throw's aim turns (at most 0.35 rad) to the projected victim position (`hookSnapAim`).
    #[serde(default)]
    pub hook_snap_aim: Option<bool>,
    /// `polishRope` also polishes while our hook is in flight (`hookKeepFlying`); only has an effect with `polish`.
    #[serde(default)]
    pub hook_keep_flying: Option<bool>,
    /// Cost per tick of being hauled by the victim's rope up into a freeze/death ceiling (`ropeCeilingCost`; v2 default 1).
    #[serde(default)]
    pub rope_ceiling_cost: Option<f64>,
    /// The competitor's live scoring values (`LIVE_PLANNER_CFG` of af49dfb: 1.5 and 0.4; the hybrid's default is 1.0 and 0.15).
    #[serde(default)]
    pub launch_exposure: Option<f64>,
    #[serde(default)]
    pub jumpless_hazard_cost: Option<f64>,
    /// `HybridConfig::polish`: after CEM, hold-the-hook variants of the best plan join the pool.
    #[serde(default)]
    pub polish: Option<bool>,
    /// `HybridConfig::wall_throws`: wall swings for a frozen victim toward a wall beside us (and, with `air_chain`, air chains).
    #[serde(default)]
    pub wall_throws: Option<bool>,
    /// Task 3.18: `HybridConfig::wall_dir` (the hall's side the wall swings throw toward, `-1`/`1`; default 0 = the nearer solid wall).
    #[serde(default)]
    pub wall_dir: Option<i32>,
    /// Task 3.18 (`--finish wb`): `HybridConfig::wb_hold` -- the wayblock hall's wall swings against a frozen victim (needs the player's `wb = true`).
    #[serde(default)]
    pub wb_hold: Option<bool>,
    #[serde(default)]
    pub air_chain: Option<bool>,
    /// The planner the opponent model runs in the victim's seat: `normal` (default), `normal-v2` or `live-v2`.
    #[serde(default)]
    pub mirror_preset: Option<String>,
    // --- Task 3.14 (E-026): duel knobs of the planner scoring (all default off = the preset's values).
    /// `PlannerConfig::hook_release_cost` (TS `hookReleaseCost`, default 0): cost of a rollout in which our hook lets go of a free, alive victim.
    #[serde(default)]
    pub hook_release_cost: Option<f64>,
    /// `PlannerConfig::self_freeze_bias` (TS `selfFreezeBias`, default 1.5; the competitor's `!try careful2` is 2.0, `bold` 1.0).
    #[serde(default)]
    pub self_freeze_bias: Option<f64>,
    /// `PlannerConfig::hook_hold_weight` (default 0.08): per-tick reward of holding the victim on our hook.
    #[serde(default)]
    pub hook_hold_weight: Option<f64>,
    /// `PlannerConfig::flip_cost` (default 0.4): cost of each change of walking direction inside a plan.
    #[serde(default)]
    pub flip_cost: Option<f64>,
    /// `PlannerConfig::launch_exact_reach` (default 70 px) / `launch_exact_weight` (default 2): the exact ballistic launch term (a hammer hit from the victim, then the flight
    /// into a freeze) applies within this reach and costs this per tick.
    #[serde(default)]
    pub launch_exact_reach: Option<f64>,
    #[serde(default)]
    pub launch_exact_weight: Option<f64>,
    /// `PlannerConfig::ceiling_guard_cost` / `ceiling_guard_px` (default 0 = off): per tick closer than `ceiling_guard_px` to a freeze ceiling above us with a free opponent in reach.
    #[serde(default)]
    pub ceiling_guard_cost: Option<f64>,
    #[serde(default)]
    pub ceiling_guard_px: Option<f64>,
    /// `HybridConfig::lag_mirror` (default off): the planning world plays the victim through the input-lag window by the opponent model's predicted plan.
    #[serde(default)]
    pub lag_mirror: Option<bool>,
    /// `PlannerConfig::duel_loss_cost` (default 0): once per rollout, the first tick we are frozen or dead costs this (a duel is lost by its first freeze).
    #[serde(default)]
    pub duel_loss_cost: Option<f64>,
    /// `PlannerConfig::duel_win_bonus` (default 0): once per rollout, the first tick the victim is frozen or dead is worth this.
    #[serde(default)]
    pub duel_win_bonus: Option<f64>,
    /// Task 3.15 (E-028): path of a trained opponent-input model (`ddai-oppnet` bundle, `~/` allowed). Switches `HybridConfig::window_model` on: the
    /// roll through the input-lag window plays the victim by the model's prediction. Default off.
    #[serde(default)]
    pub window_model: Option<String>,
    /// Confidence gate of the window model, in logits (default 0 = off): the direction, jump, hook and press heads that are less sure than this predict what the
    /// snapshot shows (hold). The aim change is a regression, not a decision: it is never gated.
    #[serde(default)]
    pub window_gate: Option<f64>,
    /// Head ablation of the window model: the comma-separated heads it may use (`dir`, `jump`, `hook`, `press`, `aim`; default all); the others predict hold.
    #[serde(default)]
    pub window_heads: Option<String>,
    /// Task 3.21 (E-036): decoding thresholds of a v2 window model, `"press=-1.5,jump=0.5,hook=0,dir_margin=1"` (logits; the keys given replace the model file's own).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_decode: Option<String>,
    /// Task 3.16 (D-115): the live knob `hybrid_budget_ms` itself (whole ms, 1 to 8): `HybridConfig::with_budget_ms`, applied after every other
    /// field of this table, so the search budget **and** the decision cap move together exactly as they do in the bot. Deadline mode only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub live_budget_ms: Option<u32>,
    // --- Task 3.19 (D-116): the reflex hammer and the hammer-safe envelope (`HybridConfig::reflex`). All default off.
    /// The reflex hammer: swing when the other tee is free and a swing along the aim at its predicted position would hit and our hammer is ready.
    #[serde(default)]
    pub reflex_hammer: Option<bool>,
    /// Pixels added to the swing's hit radius (default 0).
    #[serde(default)]
    pub reflex_slack_px: Option<f64>,
    /// The reflex swings only when the hit throws the other tee into a freeze or death tile (default false).
    #[serde(default)]
    pub reflex_hazard_only: Option<bool>,
    /// Ticks since our last swing the reflex waits for, on top of the reload timer (default 0; 16 is what a live world, which does not know the reload, needs).
    #[serde(default)]
    pub reflex_lockout_ticks: Option<i64>,
    /// The hammer-safe envelope: drop the jump (the hook climb) that a worst-case hit would carry into a freeze tile.
    #[serde(default)]
    pub envelope: Option<bool>,
    /// The other tee is a threat within this many px (default 110), when its hammer is at most `envelope_ready_ticks` (default 8) from ready.
    #[serde(default)]
    pub envelope_threat_px: Option<f64>,
    #[serde(default)]
    pub envelope_ready_ticks: Option<i64>,
    /// The worst-case kick of a hit, px/tick (default 11), and the margin under the freeze tile, px (default 16).
    #[serde(default)]
    pub envelope_kick: Option<f64>,
    #[serde(default)]
    pub envelope_margin_px: Option<f64>,
    /// Also drop the hook of a climb toward a tee above (default false).
    #[serde(default)]
    pub envelope_hook_climb: Option<bool>,
    /// Task 3.23 (D-121): the fixes of the 2026-10-08 duel against a human (`HybridConfig::duel_fixes`). All default off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duel_fixes: Option<DuelFixSpec>,
}

/// Task 3.23: `[hybrid.duel_fixes]` (see `ddai_planner::hybrid::DuelFixConfig`); a key not given keeps the default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DuelFixSpec {
    /// The fixes act only in a duel (the arena's live view is one; default true).
    #[serde(default)]
    pub duel_only: Option<bool>,
    /// Fix 1: against a static victim choose among the plans that act, while one is safe.
    #[serde(default)]
    pub static_push: Option<bool>,
    /// Fix 3: a frozen victim lying off the freeze is answered by a plan that acts, and approach plans join the pool.
    #[serde(default)]
    pub finish_push: Option<bool>,
    /// Fix 3: at most this many approach plans (technique T30) per such decision.
    #[serde(default)]
    pub finish_approach: Option<usize>,
    /// Fix 3: no hammer swing at a frozen victim.
    #[serde(default)]
    pub no_hammer_frozen: Option<bool>,
    /// Fix 2: the reacting victim of the robust stage lets go of us once it is below us while we rise.
    #[serde(default)]
    pub counter_release: Option<bool>,
    /// Fix 2: the robust stage believes the victim reacts with at least this probability while his hook holds us (0 = off).
    #[serde(default)]
    pub hooked_belief: Option<f64>,
    /// Fix 2: the best defensive techniques are re-scored by the robust stage while his hook holds us.
    #[serde(default)]
    pub protect_defence: Option<bool>,
    /// Fix 1: passive decisions before a victim counts as static (default 25, one second).
    #[serde(default)]
    pub static_after: Option<u32>,
}

impl HybridSpec {
    /// The proposer name (`none` when unset).
    pub fn proposer_name(&self) -> &str {
        self.proposer.as_deref().unwrap_or("none")
    }
}

/// Builds the [`HybridConfig`] of a `hybrid` player spec (the planner preset, the mode/budget, the
/// clock, and the [`HybridSpec`] overrides).
pub fn hybrid_config(spec: &PlayerSpec) -> Result<(HybridConfig, ClockKind), EnvError> {
    let preset = match spec.preset.as_deref().unwrap_or("normal") {
        "normal" => PlannerPreset::Normal,
        "low" => PlannerPreset::Low,
        "strong" => PlannerPreset::Strong,
        other => return Err(EnvError::new(format!("hybrid: unknown preset {other:?}"))),
    };
    let mode = match spec.mode.as_deref().unwrap_or("deadline") {
        "fixed" => HybridMode::Fixed,
        "deadline" => HybridMode::Deadline {
            budget_ms: spec.budget_ms.unwrap_or(4.0),
        },
        other => return Err(EnvError::new(format!("hybrid: unknown mode {other:?}"))),
    };
    let mut cfg = HybridConfig {
        planner: hybrid_terms(preset.config()),
        mode,
        ..HybridConfig::default()
    };
    let mut work_us = None;
    let clock = match spec.clock.as_deref().unwrap_or("wall") {
        "wall" => ClockKind::Wall,
        "work" => {
            // Deadline counted in simulated tee-ticks (reproducible, load-independent);
            // `step_ms` names the milliseconds per tee-tick (default `WORK_US_PER_TEE_TICK`, 1.25 us
            // since task 3.6; the E-003/E-007 configs pin their original 2.2 us with `step_ms = 0.0022`).
            work_us = Some(
                spec.step_ms
                    .map_or(ddai_planner::hybrid::WORK_US_PER_TEE_TICK, |ms| ms * 1000.0),
            );
            ClockKind::Wall
        }
        "step" => ClockKind::Step {
            step_ms: spec
                .step_ms
                .filter(|s| *s > 0.0)
                .ok_or_else(|| EnvError::new("hybrid: clock = \"step\" needs step_ms > 0"))?,
        },
        other => return Err(EnvError::new(format!("hybrid: unknown clock {other:?}"))),
    };
    cfg.work_clock_us_per_tick = work_us;
    if matches!(cfg.mode, HybridMode::Fixed) {
        cfg.adaptive.enabled = false;
    }
    if let Some(h) = &spec.hybrid {
        cfg.proposals = h
            .proposals
            .unwrap_or(if h.proposer_name() == "none" { 0 } else { cfg.proposals });
        if let Some(v) = h.workers {
            cfg.workers = v;
        }
        if let Some(v) = h.frozen_drag_weight {
            cfg.planner.frozen_drag_weight = v;
        }
        if let Some(v) = h.held_forecast_weight {
            cfg.planner.held_forecast_weight = v;
        }
        if let Some(v) = h.finish_families {
            cfg.finish_families = v;
        }
        if let Some(v) = h.frozen_stage_weight {
            cfg.planner.frozen_stage_weight = v;
        }
        if let Some(v) = h.mutual_freeze_cost {
            cfg.planner.mutual_freeze_cost = v;
        }
        if let Some(v) = h.sealed_forecast_weight {
            cfg.planner.sealed_forecast_weight = v;
        }
        if let Some(v) = h.frozen_seal_weight {
            cfg.planner.frozen_seal_weight = v;
        }
        if let Some(v) = h.frozen_steps {
            cfg.frozen_steps = v;
        }
        if let Some(v) = h.frozen_no_extension {
            cfg.frozen_no_extension = v;
        }
        if let Some(v) = h.frozen_budget_ms {
            cfg.frozen_budget_ms = Some(v);
        }
        if let Some(v) = h.frozen_steps_min_ticks {
            cfg.frozen_steps_min_ticks = v;
        }
        if let Some(v) = h.approach_plans {
            cfg.approach_plans = v;
        }
        if let Some(v) = h.techniques {
            cfg.techniques = v;
        }
        if let Some(v) = h.threat_model {
            cfg.threat_model = v;
        }
        cfg.threat_radius_px = h.threat_radius_px.or(cfg.threat_radius_px);
        if let Some(v) = h.threat_weight {
            cfg.threat_weight = v;
        }
        if let Some(v) = h.decision_cap_ms {
            cfg.decision_cap_ms = if v > 0.0 { Some(v) } else { None };
        }
        if let Some(v) = h.proposal_in_cap {
            cfg.proposal_in_cap = v;
        }
        if let Some(v) = h.crowd_stage {
            cfg.robust.crowd_stage = v;
        }
        if let Some(v) = h.crowd_top_m {
            cfg.robust.crowd_top_m = v;
        }
        if let Some(v) = h.prune {
            cfg.prune.enabled = v;
        }
        if let Some(v) = h.prune_steps {
            cfg.prune.steps = v;
        }
        if let Some(v) = h.prune_keep {
            cfg.prune.keep = v;
        }
        if let Some(v) = h.prune_warmup {
            cfg.prune.warmup = v;
        }
        if let Some(v) = h.max_sim_tees {
            cfg.max_sim_tees = v;
        }
        if let Some(v) = h.shield_skip_tiles {
            cfg.shield_skip_tiles = v;
        }
        if let Some(v) = h.shield_plan_escape {
            cfg.shield_plan_escape = v;
        }
        if let Some(v) = h.shield_hook_anchors {
            cfg.shield_hook_anchors = v;
        }
        if let Some(v) = h.shield_timeout_danger {
            cfg.shield_timeout_danger = v;
        }
        if let Some(v) = h.shield_reserve_ms_per_tee {
            cfg.shield_reserve_ms_per_tee = v;
        }
        if let Some(v) = h.hook_threats {
            cfg.hook_threats = v;
        }
        if let Some(v) = h.belief_lambda {
            cfg.robust.belief_lambda = v;
        }
        if let Some(v) = h.max_relevant {
            cfg.robust.max_relevant = v;
        }
        if let Some(v) = h.max_combos {
            cfg.robust.max_combos = v;
        }
        if let Some(v) = h.robust {
            cfg.robust.enabled = v;
        }
        if let Some(v) = h.lambda {
            cfg.robust.lambda = v;
        }
        match h.robust_mode.as_deref() {
            None | Some("mix") => {}
            Some("safe") => cfg.robust.mode = ddai_planner::hybrid::RobustMode::SafeFirst,
            Some(other) => return Err(EnvError::new(format!("hybrid: unknown robust_mode {other:?}"))),
        }
        if let Some(v) = h.top_m {
            cfg.robust.top_m = v;
        }
        if let Some(v) = h.adaptive {
            cfg.adaptive.enabled = v;
        }
        if let Some(v) = h.max_total_ms {
            cfg.adaptive.max_total_ms = v;
        }
        if let Some(v) = h.front_steps {
            cfg.planner.front_steps = v;
        }
        if let Some(v) = h.front_step {
            cfg.planner.front_step = v;
        }
        if let Some(v) = h.mirror {
            cfg.mirror = v;
        }
        if let Some(v) = h.mirror_samples {
            cfg.mirror_samples = v;
        }
        if let Some(v) = h.cem_population {
            cfg.planner.population = v;
        }
        if let Some(v) = h.cem_iterations {
            cfg.planner.iterations = v;
        }
        if let Some(v) = h.plan_steps {
            cfg.planner.steps = v;
        }
        if let Some(v) = h.plan_step_ticks {
            cfg.planner.plan_step = v;
        }
        if let Some(v) = h.warm_fire_only {
            cfg.warm_fire_only = v;
        }
        if let Some(v) = h.warm_bonus {
            cfg.warm_bonus = v;
        }
        if let Some(v) = h.two_world {
            cfg.two_world = v;
        }
        if let Some(v) = h.stage2_dynamic_min_relevant {
            cfg.stage2_dynamic_min_relevant = v;
        }
        if let Some(v) = h.stage2_dynamic {
            cfg.stage2_dynamic = v;
        }
        if let Some(v) = h.stage2_fraction {
            cfg.stage2_fraction = v;
        }
        if let Some(v) = h.anchors {
            cfg.anchors = v;
        }
        if let Some(v) = h.throw_cap {
            cfg.throw_cap = v;
        }
        if let Some(v) = h.debug_dump {
            cfg.debug_dump = v;
        }
        if let Some(v) = h.enemy_landing_bonus {
            cfg.planner.enemy_landing_bonus = v;
        }
        if let Some(v) = h.landing_cost {
            cfg.planner.landing_cost = v;
        }
        if let Some(v) = h.jumpless_air_cost {
            cfg.planner.jumpless_air_cost = v;
        }
        if let Some(v) = h.jumpless_anchor_bonus {
            cfg.planner.jumpless_anchor_bonus = v;
        }
        if h.v2 == Some(true) {
            cfg.planner = cfg
                .planner
                .with_version(ddai_planner::config::PlannerVersion::Upstream20261002);
        }
        if let Some(v) = h.hook_exact_gate {
            cfg.planner.hook_exact_gate = v;
        }
        if let Some(v) = h.hook_snap_aim {
            cfg.planner.hook_snap_aim = v;
        }
        if let Some(v) = h.hook_keep_flying {
            cfg.planner.hook_keep_flying = v;
        }
        if let Some(v) = h.rope_ceiling_cost {
            cfg.planner.rope_ceiling_cost = v;
        }
        if let Some(v) = h.launch_exposure {
            cfg.planner.launch_exposure = v;
        }
        if let Some(v) = h.jumpless_hazard_cost {
            cfg.planner.jumpless_hazard_cost = v;
        }
        if let Some(v) = h.air_chain {
            cfg.planner.air_chain = v;
        }
        if let Some(v) = h.hook_release_cost {
            cfg.planner.hook_release_cost = v;
        }
        if let Some(v) = h.self_freeze_bias {
            cfg.planner.self_freeze_bias = v;
        }
        if let Some(v) = h.hook_hold_weight {
            cfg.planner.hook_hold_weight = v;
        }
        if let Some(v) = h.flip_cost {
            cfg.planner.flip_cost = v;
        }
        if let Some(v) = h.launch_exact_reach {
            cfg.planner.launch_exact_reach = v;
        }
        if let Some(v) = h.launch_exact_weight {
            cfg.planner.launch_exact_weight = v;
        }
        if let Some(v) = h.lag_mirror {
            cfg.lag_mirror = v;
        }
        if let Some(v) = h.ceiling_guard_cost {
            cfg.planner.ceiling_guard_cost = v;
        }
        if let Some(v) = h.ceiling_guard_px {
            cfg.planner.ceiling_guard_px = v;
        }
        if let Some(v) = h.duel_loss_cost {
            cfg.planner.duel_loss_cost = v;
        }
        if let Some(v) = h.duel_win_bonus {
            cfg.planner.duel_win_bonus = v;
        }
        if h.window_model.is_some() {
            cfg.window_model = true;
        }
        if let Some(v) = h.reflex_hammer {
            cfg.reflex.hammer = v;
        }
        if let Some(v) = h.reflex_slack_px {
            cfg.reflex.slack_px = v;
        }
        if let Some(v) = h.reflex_hazard_only {
            cfg.reflex.hazard_only = v;
        }
        if let Some(v) = h.reflex_lockout_ticks {
            cfg.reflex.lockout_ticks = v;
        }
        if let Some(v) = h.envelope {
            cfg.reflex.envelope = v;
        }
        if let Some(v) = h.envelope_threat_px {
            cfg.reflex.threat_px = v;
        }
        if let Some(v) = h.envelope_ready_ticks {
            cfg.reflex.ready_ticks = v;
        }
        if let Some(v) = h.envelope_kick {
            cfg.reflex.kick = v;
        }
        if let Some(v) = h.envelope_margin_px {
            cfg.reflex.margin_px = v;
        }
        if let Some(v) = h.envelope_hook_climb {
            cfg.reflex.hook_climb = v;
        }
        if let Some(d) = &h.duel_fixes {
            if let Some(v) = d.duel_only {
                cfg.duel_fixes.duel_only = v;
            }
            if let Some(v) = d.static_push {
                cfg.duel_fixes.static_push = v;
            }
            if let Some(v) = d.finish_push {
                cfg.duel_fixes.finish_push = v;
            }
            if let Some(v) = d.finish_approach {
                cfg.duel_fixes.finish_approach = v;
            }
            if let Some(v) = d.no_hammer_frozen {
                cfg.duel_fixes.no_hammer_frozen = v;
            }
            if let Some(v) = d.counter_release {
                cfg.duel_fixes.counter_release = v;
            }
            if let Some(v) = d.protect_defence {
                cfg.duel_fixes.protect_defence = v;
            }
            if let Some(v) = d.hooked_belief {
                cfg.duel_fixes.hooked_belief = v;
            }
            if let Some(v) = d.static_after {
                cfg.duel_fixes.static_after = v;
            }
        }
        if let Some(v) = h.polish {
            cfg.polish = v;
        }
        if let Some(v) = h.wall_throws {
            cfg.wall_throws = v;
        }
        if let Some(v) = h.wall_dir {
            cfg.wall_dir = v;
        }
        if let Some(v) = h.wb_hold {
            cfg.wb_hold = v;
        }
        cfg.mirror_planner = match h.mirror_preset.as_deref() {
            None | Some("normal") => None,
            Some("normal-v2") => Some(ddai_planner::config::preset_normal_v2()),
            Some("live-v2") => Some(ddai_planner::config::preset_live_v2()),
            Some(other) => {
                return Err(EnvError::new(format!(
                    "hybrid: unknown mirror_preset {other:?} (normal, normal-v2, live-v2)"
                )));
            }
        };
        if let Some(ms) = h.live_budget_ms {
            if !ddai_planner::hybrid::BUDGET_MS_RANGE.contains(&ms) || !matches!(cfg.mode, HybridMode::Deadline { .. })
            {
                return Err(EnvError::new(format!(
                    "hybrid: live_budget_ms = {ms} needs deadline mode and a value in {:?}",
                    ddai_planner::hybrid::BUDGET_MS_RANGE
                )));
            }
            cfg = cfg.with_budget_ms(f64::from(ms));
        }
    } else {
        cfg.proposals = 0;
    }
    cfg.validate().map_err(EnvError::new)?;
    Ok((cfg, clock))
}

/// A proposer built in: `none` or `scripted`.
pub fn builtin_proposer(name: &str) -> Result<Box<dyn Proposer>, EnvError> {
    match name {
        "none" => Ok(Box::new(NoProposer)),
        "scripted" => Ok(Box::new(ScriptedProposer::new())),
        other => Err(EnvError::new(format!(
            "unknown proposer {other:?} (built in: none, scripted; fly needs the CLI)"
        ))),
    }
}

/// One match-up: an arena, the players and (optionally) its own game count and rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Condition {
    pub name: String,
    pub arena: String,
    #[serde(default)]
    pub games: Option<u32>,
    #[serde(default)]
    pub rules: Option<RulesOverride>,
    /// Slot 0 is the focal player (A); the rest are its opponents.
    pub players: Vec<PlayerSpec>,
    /// Task 3.19 (D-116, opt-in): the F-DDrace duel options -- the minigame's round rules and the live view of a brain (`[condition.duel]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duel: Option<crate::duel::DuelSpec>,
}

impl Condition {
    /// The players with `count` expanded, one spec per slot.
    pub fn slots(&self) -> Vec<PlayerSpec> {
        self.players
            .iter()
            .flat_map(|p| {
                let mut one = p.clone();
                one.count = 1;
                std::iter::repeat_n(one, p.count as usize)
            })
            .collect()
    }
}

/// A whole run: shared settings plus its conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub name: String,
    /// Game `g` of every condition uses seed `base_seed + g`.
    #[serde(default = "d_seed")]
    pub base_seed: u64,
    /// Default games per condition.
    #[serde(default = "d_games")]
    pub games: u32,
    #[serde(default)]
    pub rules: Rules,
    /// Directory of arena definitions (`*.toml`); default `configs/arenas`.
    #[serde(default)]
    pub arenas_dir: Option<String>,
    /// Directory the maps are read from; default `~/aiddnet/data/maps`.
    #[serde(default)]
    pub map_dir: Option<String>,
    #[serde(default)]
    pub condition: Vec<Condition>,
}

impl RunConfig {
    pub fn parse(text: &str) -> Result<RunConfig, EnvError> {
        let cfg: RunConfig = toml::from_str(text).map_err(|e| EnvError::new(format!("run config: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), EnvError> {
        self.rules.validate()?;
        let mut names = std::collections::BTreeSet::new();
        for c in &self.condition {
            if !names.insert(&c.name) {
                return Err(EnvError::new(format!("duplicate condition name {:?}", c.name)));
            }
            if c.slots().len() < 2 {
                return Err(EnvError::new(format!(
                    "condition {:?}: needs at least two players",
                    c.name
                )));
            }
            if let Some(r) = &c.rules {
                r.apply(&self.rules).validate()?;
            }
            for slot in c.slots() {
                if let Some(m) = &slot.lag_model {
                    m.validate(slot.lag)
                        .map_err(|e| EnvError::new(format!("condition {:?}: lag_model: {e}", c.name)))?;
                }
            }
            if let Some(d) = &c.duel {
                d.validate(c.slots().len())
                    .map_err(|e| EnvError::new(format!("condition {:?}: {e}", c.name)))?;
            }
        }
        Ok(())
    }

    /// The rules that apply to `c`: the run-level rules with the condition's overrides applied.
    pub fn rules_for(&self, c: &Condition) -> Rules {
        c.rules
            .as_ref()
            .map_or_else(|| self.rules.clone(), |o| o.apply(&self.rules))
    }

    /// Games to play for `c`.
    pub fn games_for(&self, c: &Condition) -> u32 {
        c.games.unwrap_or(self.games)
    }

    /// SHA-256 (hex) of the canonical JSON of this (effective) config: independent of comments,
    /// whitespace and key order in the TOML file.
    pub fn hash(&self) -> String {
        let json = serde_json::to_string(self).expect("config serializes");
        hex(&Sha256::digest(json.as_bytes()))
    }
}

/// Builds a brain for one slot.
pub type BrainFactory = dyn Fn(&PlayerSpec) -> Result<Box<dyn Brain>, EnvError> + Sync;

/// The brains this crate knows: `idle`, `scripted`, `planner`, `hybrid`.
pub fn builtin_brain(spec: &PlayerSpec) -> Result<Box<dyn Brain>, EnvError> {
    match spec.brain.as_str() {
        "idle" => Ok(Box::new(IdleBrain)),
        "scripted" => Ok(Box::new(ScriptedBrain::new())),
        "planner" => {
            let preset = PlannerPreset::parse(spec.preset.as_deref().unwrap_or("normal")).ok_or_else(|| {
                EnvError::new(format!(
                    "planner: unknown preset {:?} (normal, low, strong, normal-v2, live-v2, v2-strong, v2-strong-wb, live-v2-strong, strong-fixed)",
                    spec.preset.as_deref().unwrap_or("normal")
                ))
            })?;
            let mode = match spec.mode.as_deref().unwrap_or("fixed") {
                "fixed" => PlannerMode::Fixed,
                "deadline" => PlannerMode::Deadline {
                    budget_ms: spec
                        .budget_ms
                        .filter(|b| *b > 0.0)
                        .ok_or_else(|| EnvError::new("planner: mode = \"deadline\" needs budget_ms > 0"))?,
                },
                other => return Err(EnvError::new(format!("planner: unknown mode {other:?}"))),
            };
            let clock = match spec.clock.as_deref().unwrap_or("wall") {
                "wall" => ClockKind::Wall,
                "step" => ClockKind::Step {
                    step_ms: spec
                        .step_ms
                        .filter(|s| *s > 0.0)
                        .ok_or_else(|| EnvError::new("planner: clock = \"step\" needs step_ms > 0"))?,
                },
                other => return Err(EnvError::new(format!("planner: unknown clock {other:?}"))),
            };
            Ok(Box::new(PlannerBrain::new(PlannerBrainConfig { preset, mode, clock })))
        }
        "hybrid" => {
            let (cfg, clock) = hybrid_config(spec)?;
            let proposer = builtin_proposer(spec.hybrid.as_ref().map_or("none", HybridSpec::proposer_name))?;
            let mut brain = HybridBrain::new(cfg, clock, proposer).map_err(EnvError::new)?;
            if let Some(path) = spec.hybrid.as_ref().and_then(|h| h.window_model.as_deref()) {
                let path = match path.strip_prefix("~/") {
                    Some(rest) => std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
                    None => std::path::PathBuf::from(path),
                };
                let h = spec.hybrid.as_ref();
                let model = match ddai_oppnet::AnyPredictor::load(&path).map_err(EnvError::new)? {
                    ddai_oppnet::AnyPredictor::V1(m) => {
                        // Task 3.21: a key that belongs to the other generation is an error, not a silent no-op.
                        if h.is_some_and(|h| h.window_decode.is_some()) {
                            return Err(EnvError::new(format!(
                                "hybrid: window_decode is for a v2 window model, {} is a v1 one",
                                path.display()
                            )));
                        }
                        let gate = h.and_then(|h| h.window_gate).unwrap_or(0.0);
                        let heads = match h.and_then(|h| h.window_heads.as_deref()) {
                            Some(list) => ddai_oppnet::predictor::parse_heads(list).map_err(EnvError::new)?,
                            None => ddai_oppnet::predictor::HEAD_ALL,
                        };
                        ddai_oppnet::AnyPredictor::V1(Box::new((*m).with_gate(gate as f32).with_heads(heads)))
                    }
                    ddai_oppnet::AnyPredictor::V2(m) => {
                        if h.is_some_and(|h| h.window_gate.is_some() || h.window_heads.is_some()) {
                            return Err(EnvError::new(format!(
                                "hybrid: window_gate and window_heads are for a v1 window model, {} is a v2 one (use window_decode)",
                                path.display()
                            )));
                        }
                        let mut d = *m.decode();
                        if let Some(list) = h.and_then(|h| h.window_decode.as_deref()) {
                            ddai_oppnet::v2::predictor::apply_decode_overrides(&mut d, list).map_err(EnvError::new)?;
                        }
                        ddai_oppnet::AnyPredictor::V2(Box::new((*m).with_decode(d)))
                    }
                };
                brain.set_window_model(Box::new(model));
            }
            Ok(Box::new(brain))
        }
        other => Err(EnvError::new(format!(
            "unknown brain {other:?} (built in: idle, scripted, planner, hybrid)"
        ))),
    }
}

/// Which arena names a config uses that `available` lacks.
pub fn missing_arenas<'a, V>(cfg: &'a RunConfig, available: &BTreeMap<String, V>) -> Vec<&'a str> {
    let mut missing: Vec<&str> = cfg
        .condition
        .iter()
        .map(|c| c.arena.as_str())
        .filter(|a| !available.contains_key(*a))
        .collect();
    missing.sort_unstable();
    missing.dedup();
    missing
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: &str = r#"
name = "t"
base_seed = 7
games = 10

[[condition]]
name = "pit 1v3"
arena = "pit"
players = [
    { brain = "planner", preset = "normal", mode = "deadline", budget_ms = 2.0 },
    { brain = "scripted", count = 3 },
]
"#;

    #[test]
    fn parses_and_expands_counts() {
        let cfg = RunConfig::parse(CFG).unwrap();
        assert_eq!(cfg.base_seed, 7);
        assert_eq!(cfg.rules, Rules::default());
        let slots = cfg.condition[0].slots();
        assert_eq!(slots.len(), 4);
        assert_eq!(slots[1].brain, "scripted");
        assert!(slots.iter().all(|s| s.count == 1));
    }

    /// Task 3.6 (D-076): `clock = "work"` without `step_ms` is the current calibration; with `step_ms` it is
    /// exactly that many milliseconds per tee-tick, and `0.0022` is bit-exactly the old 2.2 us.
    #[test]
    fn work_clock_rate_defaults_to_the_calibrated_constant_and_step_ms_overrides_it() {
        let spec = |extra: &str| {
            let text = format!(
                "name = \"t\"\n[[condition]]\nname = \"c\"\narena = \"a\"\nplayers = [{{ brain = \"hybrid\", clock = \"work\"{extra} }}, {{ brain = \"scripted\" }}]\n"
            );
            RunConfig::parse(&text).unwrap().condition[0].players[0].clone()
        };
        let rate = |extra: &str| hybrid_config(&spec(extra)).unwrap().0.work_clock_us_per_tick;
        assert_eq!(rate(""), Some(ddai_planner::hybrid::WORK_US_PER_TEE_TICK));
        assert_eq!(rate(", step_ms = 0.0022").map(f64::to_bits), Some(2.2f64.to_bits()));
        assert_eq!(rate(", step_ms = 0.00125").map(f64::to_bits), Some(1.25f64.to_bits()));
    }

    /// Task 3.15 (E-028): `window_model = "<bundle>"` switches `HybridConfig::window_model` on and the brain factory loads the model; without it nothing changes,
    /// and a model file that is missing or broken is an error (never a silent fall back to the hold model).
    #[test]
    fn window_model_switch_and_loading() {
        let spec = |hybrid: &str| {
            let text = format!(
                "name = \"t\"\n[[condition]]\nname = \"c\"\narena = \"a\"\nplayers = [{{ brain = \"hybrid\", clock = \"work\", hybrid = {{ {hybrid} }} }}, {{ brain = \"scripted\" }}]\n"
            );
            RunConfig::parse(&text).unwrap().condition[0].players[0].clone()
        };
        assert!(
            !hybrid_config(&spec("workers = 1")).unwrap().0.window_model,
            "default off"
        );
        let on = spec("window_model = \"~/nowhere/m.oppnet\", window_gate = 1.5");
        assert!(hybrid_config(&on).unwrap().0.window_model);
        assert_eq!(on.hybrid.as_ref().unwrap().window_gate, Some(1.5));
        let err = builtin_brain(&on)
            .err()
            .expect("a missing model file is an error")
            .to_string();
        assert!(err.contains("m.oppnet"), "{err}");
        assert!(builtin_brain(&spec("workers = 1")).is_ok());
    }

    /// Task 3.21 review F1: the new `window_decode` key leaves the canonical JSON (so the hash) of every config that does not set it as it was; F6: a window key
    /// of the other model generation is an error.
    #[test]
    fn window_decode_keeps_old_hashes_and_keys_of_the_other_generation_are_refused() {
        let text = |hybrid: &str| {
            format!(
                "name = \"t\"\n[[condition]]\nname = \"c\"\narena = \"a\"\nplayers = [{{ brain = \"hybrid\", clock = \"work\", hybrid = {{ {hybrid} }} }}, {{ brain = \"scripted\" }}]\n"
            )
        };
        let plain = RunConfig::parse(&text("workers = 1")).unwrap();
        let json = serde_json::to_string(&plain).unwrap();
        assert!(!json.contains("window_decode"), "{json}");
        let set = RunConfig::parse(&text("workers = 1, window_decode = \"press=1\"")).unwrap();
        assert_ne!(plain.hash(), set.hash(), "a config that sets it hashes differently");

        let dir = tempfile::tempdir().unwrap();
        let (p1, p2) = (dir.path().join("a.oppnet"), dir.path().join("b.oppnet"));
        ddai_oppnet::OppBundle::new(
            ddai_oppnet::net::Mlp::new(ddai_oppnet::feature::INPUT_DIM, 4, 4, ddai_oppnet::feature::OUT_DIM, 1),
            1,
            1,
            0.0,
            "v1".into(),
        )
        .save(&p1)
        .unwrap();
        {
            use ddai_oppnet::v2::{feature as f, predictor as p};
            p::Bundle::new(
                ddai_oppnet::net::Mlp::new(f::INPUT_DIM, 4, 4, f::OUT_DIM, 1),
                p::Decode::default(),
                1,
                1,
                0.0,
                "v2".into(),
            )
            .save(&p2)
            .unwrap();
        }
        let brain = |hybrid: String| {
            let c = RunConfig::parse(&text(&hybrid)).unwrap();
            builtin_brain(&c.condition[0].players[0]).err().map(|e| e.to_string())
        };
        let (m1, m2) = (p1.display(), p2.display());
        assert!(brain(format!("window_model = \"{m1}\", window_gate = 1.0")).is_none());
        assert!(brain(format!("window_model = \"{m2}\", window_decode = \"press=1\"")).is_none());
        let e =
            brain(format!("window_model = \"{m1}\", window_decode = \"press=1\"")).expect("v1 refuses window_decode");
        assert!(e.contains("window_decode"), "{e}");
        let e = brain(format!("window_model = \"{m2}\", window_gate = 1.0")).expect("v2 refuses window_gate");
        assert!(e.contains("window_gate"), "{e}");
        let e = brain(format!("window_model = \"{m2}\", window_heads = \"dir\"")).expect("v2 refuses window_heads");
        assert!(e.contains("window_heads"), "{e}");
    }

    /// Task 3.7a (D-080): search threads now go with the work clock (the helpers only speculate), and the proposer's
    /// time comes off the decision cap unless a config says otherwise; every E-012 config parses.
    #[test]
    fn work_clock_takes_workers_and_the_proposal_cap_flag_is_read() {
        let spec = |hybrid: &str| {
            let text = format!(
                "name = \"t\"\n[[condition]]\nname = \"c\"\narena = \"a\"\nplayers = [{{ brain = \"hybrid\", clock = \"work\", hybrid = {{ {hybrid} }} }}, {{ brain = \"scripted\" }}]\n"
            );
            RunConfig::parse(&text).unwrap().condition[0].players[0].clone()
        };
        let (c, _) = hybrid_config(&spec("workers = 4")).unwrap();
        assert_eq!(
            (c.workers, c.work_clock_us_per_tick.is_some(), c.proposal_in_cap),
            (4, true, true)
        );
        assert!(c.validate().is_ok(), "the work clock no longer needs workers = 1");
        let (c, _) = hybrid_config(&spec("proposal_in_cap = false")).unwrap();
        assert!(!c.proposal_in_cap);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/arena");
        for name in [
            "e012-wall-strength.toml",
            "e012-golden-workers.toml",
            "e012-budget-sweep.toml",
            "e012-fly-ab.toml",
        ] {
            let text = std::fs::read_to_string(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
            let cfg = RunConfig::parse(&text).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            assert!(!cfg.condition.is_empty());
            for cond in &cfg.condition {
                for p in &cond.players {
                    if p.brain == "hybrid" {
                        hybrid_config(p)
                            .unwrap_or_else(|e| panic!("{name}: {e:?}"))
                            .0
                            .validate()
                            .unwrap();
                    }
                }
            }
        }
    }

    /// Task 3.9 (D-096): the v2 switches of the hybrid are `[hybrid]` knobs, all off unless named; `v2 = true` is the four planner
    /// switches of af49dfb, and a single knob after it overrides; an unknown mirror preset is refused.
    #[test]
    fn the_hybrid_v2_knobs_map_onto_the_config_and_default_to_off() {
        use ddai_planner::config::PlannerVersion;
        let cfg_of = |hybrid: &str| {
            let text = format!(
                "name = \"t\"\n[[condition]]\nname = \"c\"\narena = \"a\"\nplayers = [{{ brain = \"hybrid\", clock = \"work\", hybrid = {{ {hybrid} }} }}, {{ brain = \"scripted\" }}]\n"
            );
            let spec = RunConfig::parse(&text).unwrap().condition[0].players[0].clone();
            hybrid_config(&spec).map(|(c, _)| c)
        };
        let base = cfg_of("workers = 1").unwrap();
        assert_eq!(
            base,
            HybridConfig {
                work_clock_us_per_tick: base.work_clock_us_per_tick,
                proposals: 0, // no proposer in the arena unless named
                ..HybridConfig::default()
            }
        );
        assert_eq!(base.planner.version(), PlannerVersion::Classic);
        let all = cfg_of("v2 = true, polish = true, wall_throws = true, air_chain = true, mirror_preset = \"live-v2\"")
            .unwrap();
        assert_eq!(all.planner.version(), PlannerVersion::Upstream20261002);
        assert!(all.polish && all.wall_throws && all.planner.air_chain);
        assert_eq!(all.mirror_planner, Some(ddai_planner::config::preset_live_v2()));
        let one = cfg_of("v2 = true, hook_snap_aim = false, rope_ceiling_cost = 0.5, launch_exposure = 1.5, jumpless_hazard_cost = 0.4").unwrap();
        assert!(one.planner.hook_exact_gate && !one.planner.hook_snap_aim && one.planner.hook_keep_flying);
        assert_eq!(
            (
                one.planner.rope_ceiling_cost,
                one.planner.launch_exposure,
                one.planner.jumpless_hazard_cost
            ),
            (0.5, 1.5, 0.4)
        );
        assert!(cfg_of("mirror_preset = \"strong\"").is_err());
    }

    /// Every E-020 config parses and builds valid hybrid and planner players (the presets `v2-strong` and `v2-strong-wb` included).
    #[test]
    fn the_e020_configs_parse_and_build_their_players() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/arena");
        let mut n = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if !(name.starts_with("e020-") && name.ends_with(".toml")) {
                continue;
            }
            n += 1;
            let cfg =
                RunConfig::parse(&std::fs::read_to_string(&path).unwrap()).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            assert!(!cfg.condition.is_empty());
            for cond in &cfg.condition {
                for p in &cond.players {
                    if p.brain == "hybrid" {
                        hybrid_config(p)
                            .unwrap_or_else(|e| panic!("{name}: {e:?}"))
                            .0
                            .validate()
                            .unwrap();
                    } else {
                        builtin_brain(p).unwrap_or_else(|e| panic!("{name}: {e:?}"));
                    }
                }
            }
        }
        assert!(n >= 3, "found {n} E-020 configs");
    }

    /// Every arena config that runs a `clock = "work"` player must say which work-clock rate it means: either
    /// pin it (`step_ms = ...`; the configs of E-003/E-007 pin their original 2.2 us, so re-running them
    /// reproduces the recorded numbers) or carry the opt-in marker line `# work-clock: default-rate` to take the
    /// calibrated default (`WORK_US_PER_TEE_TICK`). Nothing depends on a file's name.
    #[test]
    fn work_clock_configs_pin_their_rate_or_opt_into_the_default() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/arena");
        let (mut pinned_2_2, mut opted_in) = (0, 0);
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if !name.ends_with(".toml") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let default_rate = text.lines().any(|l| l.trim() == "# work-clock: default-rate");
            for (i, line) in text.lines().enumerate() {
                if line.trim_start().starts_with('#') {
                    continue;
                }
                let mut rest = line;
                while let Some(at) = rest.find("clock = \"work\"") {
                    rest = &rest[at + "clock = \"work\"".len()..];
                    let pinned = rest.starts_with(", step_ms = ");
                    assert!(
                        pinned || default_rate,
                        "{name}:{}: a work-clock player needs `step_ms = ...` or the marker line `# work-clock: default-rate`: {line}",
                        i + 1
                    );
                    if pinned {
                        pinned_2_2 += usize::from(rest.starts_with(", step_ms = 0.0022"));
                    } else {
                        opted_in += 1;
                    }
                }
            }
        }
        assert!(pinned_2_2 > 100, "only {pinned_2_2} pinned work-clock players found");
        assert!(
            opted_in > 0,
            "no config uses the default rate: the marker path is untested"
        );
    }

    #[test]
    fn hash_ignores_formatting_but_not_content() {
        let a = RunConfig::parse(CFG).unwrap();
        let reformatted = CFG.replace("base_seed = 7", "# comment\nbase_seed=7");
        assert_eq!(a.hash(), RunConfig::parse(&reformatted).unwrap().hash());
        let b = RunConfig::parse(&CFG.replace("games = 10", "games = 11")).unwrap();
        assert_ne!(a.hash(), b.hash());
        assert_eq!(a.hash().len(), 64);
    }

    #[test]
    fn condition_rules_inherit_unset_fields_from_the_run_rules() {
        let text = r#"
name = "t"
[rules]
max_ticks = 900
after_ticks = 90
[[condition]]
name = "c"
arena = "pit"
rules = { after_ticks = 10 }
players = [{ brain = "idle" }, { brain = "idle" }]
[[condition]]
name = "d"
arena = "pit"
players = [{ brain = "idle" }, { brain = "idle" }]
"#;
        let cfg = RunConfig::parse(text).unwrap();
        let c = cfg.rules_for(&cfg.condition[0]);
        assert_eq!(
            (c.max_ticks, c.after_ticks, c.decide_every),
            (900, 10, 2),
            "max_ticks inherited, after_ticks overridden"
        );
        assert_eq!(cfg.rules_for(&cfg.condition[1]), cfg.rules);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(RunConfig::parse("name = \"x\"\nbogus = 1\n").is_err());
        let one_player = "name=\"x\"\n[[condition]]\nname=\"c\"\narena=\"pit\"\nplayers=[{brain=\"idle\"}]\n";
        assert!(RunConfig::parse(one_player).is_err());
        let dup = format!(
            "{CFG}\n[[condition]]\nname = \"pit 1v3\"\narena = \"pit\"\nplayers = [{{brain=\"idle\"}}, {{brain=\"idle\"}}]\n"
        );
        assert!(RunConfig::parse(&dup).is_err());
        assert!(RunConfig::parse("name=\"x\"\n[rules]\ndecide_every = 0\n").is_err());
    }

    #[test]
    fn builtin_brains_validate_their_parameters() {
        assert!(builtin_brain(&PlayerSpec::simple("idle")).is_ok());
        assert!(builtin_brain(&PlayerSpec::simple("scripted")).is_ok());
        assert!(builtin_brain(&PlayerSpec::simple("nope")).is_err());
        let mut p = PlayerSpec::simple("planner");
        assert!(builtin_brain(&p).is_ok());
        p.mode = Some("deadline".into());
        assert!(builtin_brain(&p).is_err(), "deadline needs a budget");
        p.budget_ms = Some(4.0);
        let b = builtin_brain(&p).unwrap();
        assert_eq!(b.name(), "planner-normal-4ms");
        p.preset = Some("huge".into());
        assert!(builtin_brain(&p).is_err());
    }
}
