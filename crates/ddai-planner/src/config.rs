//! `PlannerConfig`/`PLANNER_DEFAULTS` (`src/plan/planner.ts:35-315`) and the live presets built on
//! top of them (`docs/research/orig-plan.md` §1.2). Every field TS lists is here, including the
//! ones marked "dead/bot-only" in the research notes (`seek`, `pathToTarget`,
//! `settledFreezeTicks`, `blockHoldScore`, `liveTransfer`, `planOthers`, `targetHold`) — kept for
//! config-shape completeness (a caller building `LIVE_PLANNER_CFG`-equivalent overrides shouldn't
//! hit a missing field) even though `Planner` itself never reads them (documented at each field).

/// `PlannerConfig.opponentModel` (`planner.ts:88`). `Policy`/`Learned` are accepted as config
/// values (so a caller can round-trip a TS config verbatim) but behave as `Hold` here —
/// `docs/research/orig-plan.md` §1.14: both require a loaded GRU/MLP net this task's acceptance
/// criteria does not ask for (only `"hold"`/`"react"`/`opponentMix` are in scope); see
/// [`crate::planner::Planner::predict_opponent`]'s doc comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpponentModel {
    #[default]
    Hold,
    React,
    Policy,
    Learned,
}

/// `PlannerConfig.openingBook` (`planner.ts:160`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpeningBook {
    #[default]
    Classic,
    Wide,
    Movement,
    All,
}

/// `PlannerConfig.liveTransfer` (`planner.ts:143`) — bot-only (`docs/research/orig-plan.md`
/// §1.2's "исп." column: `liveTransfer` is read only by `bot.ts`, never by `planner.ts` itself).
/// Kept for config-shape completeness; [`crate::planner::Planner`] does not read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LiveTransfer {
    #[default]
    Full,
    Legacy,
}

/// `Required<PlannerConfig>` (`planner.ts:35-190`, defaults `planner.ts:222-315`). Field order and
/// names match TS (`snake_case` of the `camelCase` original); every default matches
/// `PLANNER_DEFAULTS` exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlannerConfig {
    pub steps: i32,
    pub plan_step: i32,

    pub rest_aim: bool,
    pub self_freeze_bias: f64,

    pub travel_weight: f64,

    /// Bot-only (`docs/research/orig-plan.md` §1.2: read by `bot.ts:4404`, never by
    /// `planner.ts`). Kept for config-shape completeness.
    pub seek: bool,

    pub memory_weight: f64,
    pub memory_trust: f64,

    /// Bot-only (`bot.ts`'s `pathGoal`); not read by `Planner`.
    pub path_to_target: bool,
    /// Bot-only; not read by `Planner`.
    pub settled_freeze_ticks: f64,
    /// Bot-only; not read by `Planner`.
    pub block_hold_score: f64,

    pub dead_zone_cost: f64,
    pub enemy_dead_zone_bonus: f64,
    pub front_steps: f64,
    pub front_step: f64,

    pub release_dead_hook: bool,
    pub population: i32,
    pub elite: i32,
    pub iterations: i32,
    pub seed: u32,

    pub wasted_hammer: f64,
    pub wasted_hook: f64,

    pub hammer_range_px: f64,

    pub gate_hook: bool,
    pub gate_hammer: bool,

    pub air_jump_cost: f64,

    pub jumpless_hazard_cost: f64,

    pub self_hazard_cost: f64,

    pub flip_cost: f64,

    pub wall_push_cost: f64,

    pub commit_decisions: i32,

    pub opponent_model: OpponentModel,

    pub enemy_hazard_weight: f64,
    pub frozen_weight: f64,

    pub hook_hold_weight: f64,
    pub distance_weight: f64,

    pub standoff_px: f64,

    pub self_hazard_threshold: f64,

    pub freeze_tail_weight: f64,

    pub seal_ticks: f64,

    pub frozen_target_steps: i32,

    pub frozen_throw: i32,

    pub band_cost: f64,

    pub shield: bool,

    pub policy_seeds: i32,
    pub policy_seed_jitter: f64,
    pub policy_seed_steps: i32,

    pub no_thaw: bool,
    pub no_thaw_rope: bool,

    pub hook_drag_weight: f64,

    pub enemy_hazard_from_start: bool,

    pub drag_threat: f64,
    pub launch_threat: f64,

    pub landing_cost: f64,

    pub plan_margin: f64,

    pub opponent_mix: bool,

    /// Soft, wall-clock deadline in milliseconds; `0` = fixed-iteration mode (deterministic,
    /// acceptance criterion 3's teacher-forced/parity mode). Non-zero enables D-041's
    /// deadline-driven production mode (`crate::planner::Planner::decide_deadline`).
    pub budget_ms: f64,
    /// Hard wall-clock deadline; also gates the book/policy/throw seeds and the post-processing
    /// passes (`polishRope`/`notNow`/`escapeBias`/`explain`) — see `docs/research/orig-plan.md`
    /// §1.3 step 12/§2.3.
    pub hard_ms: f64,

    pub shield_cadence: bool,

    pub explain: bool,

    /// Bot-only; not read by `Planner`.
    pub live_transfer: LiveTransfer,

    /// Bot-only; not read by `Planner`.
    pub plan_others: i32,
    /// Bot-only; not read by `Planner`.
    pub target_hold: f64,

    pub escape_bias: f64,
    pub escape_margin: f64,

    pub warm_shift_elapsed: bool,

    pub hook_release_cost: f64,

    /// Requires a loaded value net (`setValueNet`) — never set by any acceptance-criteria preset
    /// (default `0`, meaning `Planner::evaluate`'s value-net term is always skipped); ported for
    /// config-shape completeness, no value-net loader exists in this crate (out of scope, see
    /// `docs/research/orig-plan.md` §1.14/§6.1's "нет тренера").
    pub value_weight: f64,

    pub track_aim: bool,

    pub opening_book: OpeningBook,

    pub launch_exposure: f64,

    pub launch_exact_reach: f64,

    pub launch_exact_rise_vy: f64,

    pub launch_exact_weight: f64,

    /// `0` in every acceptance-criteria preset -- `OpponentProfile`-driven book branches are dead
    /// at that value (see `crate::opponent_profile`'s doc comment).
    pub opponent_read_weight: f64,

    /// Computed every decision but never read by `scoreTick` in TS (`docs/research/orig-plan.md`
    /// §11 item 1) -- kept as a config flag for shape completeness; `Planner` does not compute
    /// `travelField` at all (see `crate::fields`'s module doc comment for the full descope
    /// rationale), since doing so would cost cycles for a value nothing reads either way.
    pub route_distance: bool,

    pub drag_exposure: f64,

    pub third_tee_exposure: f64,

    pub freeze_throw: i32,

    pub jitter_cost: f64,
    pub flip_hold_ticks: f64,

    pub flip_margin: f64,

    pub edge_hold: bool,

    pub hook_seeds: bool,

    pub hook_polish: bool,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        PlannerConfig {
            steps: 9,
            plan_step: 3,
            rest_aim: false,
            self_freeze_bias: 1.5,
            travel_weight: 0.35,
            seek: true,
            memory_weight: 0.0,
            memory_trust: 0.0,
            path_to_target: true,
            settled_freeze_ticks: 0.0,
            block_hold_score: 0.0,
            dead_zone_cost: 0.0,
            enemy_dead_zone_bonus: 0.0,
            front_steps: 0.0,
            front_step: 2.0,
            release_dead_hook: false,
            population: 20,
            elite: 6,
            iterations: 2,
            seed: 1,
            wasted_hammer: 0.03,
            wasted_hook: 0.05,
            hammer_range_px: 80.0,
            gate_hook: true,
            gate_hammer: true,
            air_jump_cost: 0.0,
            jumpless_hazard_cost: 0.15,
            self_hazard_cost: 0.6,
            flip_cost: 0.4,
            wall_push_cost: 0.05,
            commit_decisions: 1,
            opponent_model: OpponentModel::Hold,
            enemy_hazard_weight: 2.0,
            frozen_weight: 0.5,
            hook_hold_weight: 0.08,
            distance_weight: 0.06,
            standoff_px: 250.0,
            self_hazard_threshold: 0.55,
            track_aim: true,
            opening_book: OpeningBook::Classic,
            launch_exposure: 1.0,
            launch_exact_reach: 70.0,
            launch_exact_rise_vy: 4.0,
            launch_exact_weight: 2.0,
            opponent_read_weight: 0.0,
            route_distance: false,
            drag_exposure: 0.5,
            third_tee_exposure: 0.0,
            freeze_throw: 0,
            jitter_cost: 0.0,
            flip_hold_ticks: 0.0,
            flip_margin: 0.6,
            edge_hold: true,
            freeze_tail_weight: 0.5,
            seal_ticks: 150.0,
            frozen_target_steps: 0,
            frozen_throw: 0,
            band_cost: 0.0,
            shield: true,
            policy_seeds: 0,
            policy_seed_jitter: 0.25,
            policy_seed_steps: 0,
            no_thaw: true,
            no_thaw_rope: false,
            hook_drag_weight: 0.0,
            enemy_hazard_from_start: true,
            drag_threat: 0.0,
            launch_threat: 0.0,
            landing_cost: 0.0,
            plan_margin: 0.0,
            opponent_mix: false,
            budget_ms: 0.0,
            hard_ms: 0.0,
            shield_cadence: false,
            explain: false,
            live_transfer: LiveTransfer::Full,
            plan_others: 0,
            target_hold: 400.0,
            escape_bias: 0.0,
            escape_margin: 1.5,
            warm_shift_elapsed: false,
            hook_release_cost: 0.0,
            value_weight: 0.0,
            hook_seeds: true,
            hook_polish: true,
        }
    }
}

/// `LIVE_PLANNER_CFG` merged over `PLANNER_DEFAULTS` (`bot.ts:590`), i.e. the "normal" preset
/// `docs/research/orig-run.md` §2.1 uses as the harness baseline (`thirdTeeExposure: 0` is
/// already the default, listed for parity with the TS object literal). `budgetMs`/`explain` are
/// left at the caller's choice (the harness's `--budget 0` vs. live `18`) — this returns the
/// config *before* that override, matching `plannerCfgNow()`'s own layering
/// (`docs/research/orig-plan.md` §1.2).
pub fn preset_normal() -> PlannerConfig {
    PlannerConfig {
        third_tee_exposure: 0.0,
        memory_trust: 0.9,
        frozen_throw: 3,
        explain: true,
        ..PlannerConfig::default()
    }
}

/// `--low-cpu`'s effective planner overrides (`cpuLoad.ts:17`, `bot.ts:2312-2322`):
/// `budgetMs: min(18, 6) = 6`, `hardMs: 11`, `commitDecisions: max(1, 2) = 2`, `explain: false`,
/// `shieldCadence: true`, layered on top of [`preset_normal`].
pub fn preset_low_cpu() -> PlannerConfig {
    PlannerConfig {
        budget_ms: 6.0,
        hard_ms: 11.0,
        commit_decisions: 2,
        explain: false,
        shield_cadence: true,
        ..preset_normal()
    }
}

/// `WB_PLAN_OVERRIDES` (`bot.ts:216-218`), applied via `setOverrides` on top of the *constructor*
/// config (`baseCfg`, i.e. whatever preset the planner was built with) — `docs/research/
/// orig-plan.md` §1.2/§1.3: `Object.assign(cfg, baseCfg, over)`. This helper takes the base config
/// to merge onto explicitly (the caller decides which preset is "WB base", exactly like TS's
/// `setOverrides` merges onto `this.baseCfg`, not always [`preset_normal`]).
pub fn wb_overrides(base: PlannerConfig) -> PlannerConfig {
    PlannerConfig {
        no_thaw_rope: true,
        frozen_throw: 3,
        air_jump_cost: 0.3,
        launch_exact_reach: 100.0,
        ..base
    }
}

/// `STRONG_WB` merged over [`wb_overrides`] (`cpuLoad.ts:19`) — only applied by the live bot when
/// inside a WB hall AND `population < 40` (`bot.ts:220,4809-4815`); exposed here as a plain
/// preset since `Planner` itself has no notion of "am I in a WB hall".
pub fn preset_strong_wb(base: PlannerConfig) -> PlannerConfig {
    PlannerConfig {
        population: 40,
        iterations: 3,
        budget_ms: 30.0,
        hard_ms: 36.0,
        ..wb_overrides(base)
    }
}

/// `PLANNER_BOLD` (`bot.ts:45`).
pub fn preset_bold(base: PlannerConfig) -> PlannerConfig {
    PlannerConfig {
        population: 64,
        iterations: 3,
        budget_ms: 18.0,
        ..base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_ts_planner_defaults_spot_check() {
        let cfg = PlannerConfig::default();
        assert_eq!(cfg.steps, 9);
        assert_eq!(cfg.plan_step, 3);
        assert_eq!(cfg.population, 20);
        assert_eq!(cfg.elite, 6);
        assert_eq!(cfg.iterations, 2);
        assert_eq!(cfg.self_freeze_bias, 1.5);
        assert_eq!(cfg.opponent_model, OpponentModel::Hold);
        assert_eq!(cfg.opening_book, OpeningBook::Classic);
        assert!(cfg.shield);
        assert!(cfg.hook_seeds);
        assert!(cfg.hook_polish);
    }

    #[test]
    fn normal_preset_matches_live_planner_cfg() {
        let cfg = preset_normal();
        assert_eq!(cfg.memory_trust, 0.9);
        assert_eq!(cfg.frozen_throw, 3);
        assert_eq!(cfg.third_tee_exposure, 0.0);
        assert!(cfg.explain);
    }

    #[test]
    fn low_cpu_preset_caps_budget_and_doubles_commit() {
        let cfg = preset_low_cpu();
        assert_eq!(cfg.budget_ms, 6.0);
        assert_eq!(cfg.hard_ms, 11.0);
        assert_eq!(cfg.commit_decisions, 2);
        assert!(!cfg.explain);
        assert!(cfg.shield_cadence);
    }

    #[test]
    fn wb_overrides_only_touch_the_named_fields() {
        let base = preset_normal();
        let wb = wb_overrides(base);
        assert!(wb.no_thaw_rope);
        assert_eq!(wb.frozen_throw, 3);
        assert_eq!(wb.air_jump_cost, 0.3);
        assert_eq!(wb.launch_exact_reach, 100.0);
        // Everything else untouched.
        assert_eq!(wb.memory_trust, base.memory_trust);
        assert_eq!(wb.population, base.population);
    }
}
