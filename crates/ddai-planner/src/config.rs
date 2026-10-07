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

    /// **Task 3.5, hybrid search only (`0` in every TS preset, so parity is untouched).** Bonus, at
    /// the end of a rollout, for a victim that is alive, not frozen and whose ballistic flight
    /// (`flightEndsInHazard`, the same forecast `landing_cost` uses for us) ends in freeze or death.
    /// A hammer throw or a drag launches the victim on an arc that lands after the 27-tick horizon
    /// (the 3.5 scenario T18: the hammer works, the freeze floor is 30+ ticks away); without this
    /// term the search cannot see the payoff and the throw scores the same as standing still.
    pub enemy_landing_bonus: f64,

    /// **Task 3.5, hybrid search only (`0` in every TS preset).** Per-tick bonus for hanging on a
    /// wall or ceiling hook while we have no jump left, are in the air and freeze or death lies
    /// below us (`docs/research/block-knowledge.md` T14, "panic hook"): the only control left over
    /// a pit is an anchor, but a 27-tick rollout sees only its ballistic end, so without a bonus
    /// an approach plan that merely survives scores as well as the safe hang.
    pub jumpless_anchor_bonus: f64,
    /// Task 3.5b (hybrid only, `0` in every TS preset): the cost of ending a rollout in the air with no
    /// jumps left -- landing refills them, hanging or swinging forever does not (T13: a swing that never
    /// comes down scores as well as a safe landing).
    pub jumpless_air_cost: f64,

    // --- Upstream af49dfb (2026-10-02), planner "v2" (task 3.8, D-095). Every field below is `false`/`0` in
    // `PlannerConfig::default()`, i.e. in the classic planner (upstream c3c619d), so it stays bit-identical; the
    // v2 behaviour is switched on by `PlannerConfig::with_version(PlannerVersion::Upstream20261002)` and the
    // wall-guard knobs.
    /// `wallDir` (`planner.ts` `PlannerConfig.wallDir`, default `0`): `-1`/`1` = a wall to that side (the wayblock
    /// guard's hall wall); adds `wallSwingLines` (and, with [`Self::air_chain`], `airChainLines`) to the frozen-throw
    /// seeds. `0` = off.
    pub wall_dir: i32,
    /// `airChain` (default `false`): with `wall_dir != 0`, also offer the in-the-air hook chains (`airChainLines`).
    pub air_chain: bool,
    /// `hookExactGate` (v2 default `true`): the hook gate asks whether the rope's flight meets the victim's projected
    /// position (`ropeIntercept`, `closestPointOnLineOrNull`) instead of the old "within 2 body widths of the line" test.
    pub hook_exact_gate: bool,
    /// `hookSnapAim` (v2 default `true`): a hook throw turns its aim (at most `SNAP_MAX_RAD`) to the projected
    /// victim position when the planned angle would just miss (`snapAim`/`interceptAim`).
    pub hook_snap_aim: bool,
    /// `hookKeepFlying` (v2 default `true`): `polishRope` also runs while our hook is in flight (`k` in 1..3 steps).
    pub hook_keep_flying: bool,
    /// `ropeCeilingCost` (v2 default `1`): cost per tick of being hooked by the victim and flying up into a
    /// freeze/death ceiling (`ceilingField`). `0` = off.
    pub rope_ceiling_cost: f64,
    /// Task 3.10 (hybrid only, `0` = off everywhere else, the TS-parity path never reads it): per-tick shaping of a **frozen** victim's
    /// progress toward a freeze/death tile (the change of `hazardNearness`, signed: dragging it nearer pays, losing ground costs). A freeze
    /// that is not kept up thaws after 150 ticks; this is the gradient that makes the rollouts prefer plans that haul the victim back into
    /// the freeze (the TS `hookDragWeight` is off for a frozen victim by design).
    pub frozen_drag_weight: f64,
    /// Task 3.10 (hybrid only, `0` = off): at the end of a rollout with the victim frozen, the exact passive forecast of
    /// [`crate::forecast::passive_forecast`] (ticks it stays out, up to the held-block horizon) is worth this much at full horizon, and
    /// an escape inside the horizon costs what it lacks of it.
    pub held_forecast_weight: f64,
    /// Task 3.10 (hybrid only, `0` = off): per-tile reward for our progress toward the *staging point* behind a frozen victim that lies off
    /// the freeze (110 px from it on the side of its nearest freeze): from there the rope hauls it back into the freeze.
    pub frozen_stage_weight: f64,
    /// Task 3.14 (hybrid only, `0` = off; the TS-parity path never reads it): **a duel is decided by the first freeze** (F-DDrace `/1vs1`: a frozen
    /// player standing still loses the round). Once per rollout, the first tick we are frozen or dead costs this much (`duel_loss_cost`) and the first
    /// tick the victim is costs `duel_win_bonus`; both are discounted like the per-tick terms (a later freeze can still be avoided by the next decision).
    /// The per-tick `frozen_weight` terms alone charge a freeze at the end of the 27-tick horizon next to nothing.
    pub duel_loss_cost: f64,
    pub duel_win_bonus: f64,
    /// Task 3.14 (hybrid only, `0` = off): **the ceiling guard**. A duel's launch is a hammer or hook haul *upward* into a freeze ceiling: the exact
    /// launch terms see it only once the opponent is within hammer reach, and `hazard_nearness` is flat in a closed arena (a freeze within 20 tiles
    /// on every side). While a free opponent is within [`CEILING_GUARD_REACH_PX`] of us, every tick spent closer than `ceiling_guard_px` to a freeze or
    /// death ceiling above us (the `rope_ceiling_cost` field) costs `ceiling_guard_cost` times how deep inside that gap we are (0 at the edge, 1 touching).
    pub ceiling_guard_cost: f64,
    pub ceiling_guard_px: f64,
}

/// The opponent distance within which the ceiling guard applies (the hook's length plus two decisions of closing speed, the hybrid's threat radius).
pub const CEILING_GUARD_REACH_PX: f64 = 440.0;

/// Which upstream planner a [`PlannerConfig`] reproduces (task 3.8, D-095).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlannerVersion {
    /// Upstream `c3c619d`: the planner ported in 3.2 (the TS-parity corpus of 19 360 + 3 604 decisions).
    #[default]
    Classic,
    /// Upstream `af49dfb` (release 2026-10-02): exact hook gate and aim snap, `polishRope` on a flying hook, rope-ceiling
    /// cost (and, behind `wall_dir`/`air_chain`, the wall swing/air chain throws, plus the passive `sealed_in`).
    Upstream20261002,
}

impl PlannerVersion {
    pub fn label(self) -> &'static str {
        match self {
            PlannerVersion::Classic => "classic",
            PlannerVersion::Upstream20261002 => "upstream-2026-10-02",
        }
    }
}

impl PlannerConfig {
    /// `self` with the version-dependent defaults of `version` (`PLANNER_DEFAULTS` of that upstream commit): the
    /// four v2 switches. Everything else is left as it was, so a preset keeps its own settings.
    pub fn with_version(self, version: PlannerVersion) -> PlannerConfig {
        match version {
            PlannerVersion::Classic => PlannerConfig {
                hook_exact_gate: false,
                hook_snap_aim: false,
                hook_keep_flying: false,
                rope_ceiling_cost: 0.0,
                ..self
            },
            PlannerVersion::Upstream20261002 => PlannerConfig {
                hook_exact_gate: true,
                hook_snap_aim: true,
                hook_keep_flying: true,
                rope_ceiling_cost: 1.0,
                ..self
            },
        }
    }

    /// The version `self` reproduces: `Upstream20261002` when any of the four v2 switches is on.
    pub fn version(&self) -> PlannerVersion {
        if self.hook_exact_gate || self.hook_snap_aim || self.hook_keep_flying || self.rope_ceiling_cost > 0.0 {
            PlannerVersion::Upstream20261002
        } else {
            PlannerVersion::Classic
        }
    }
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
            enemy_landing_bonus: 0.0,
            jumpless_anchor_bonus: 0.0,
            jumpless_air_cost: 0.0,
            wall_dir: 0,
            air_chain: false,
            hook_exact_gate: false,
            hook_snap_aim: false,
            hook_keep_flying: false,
            rope_ceiling_cost: 0.0,
            frozen_drag_weight: 0.0,
            held_forecast_weight: 0.0,
            frozen_stage_weight: 0.0,
            duel_loss_cost: 0.0,
            duel_win_bonus: 0.0,
            ceiling_guard_cost: 0.0,
            ceiling_guard_px: 0.0,
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

/// The competitor's current planner (upstream af49dfb) as the phase-0 harness runs it: [`preset_normal`] (the live config
/// of c3c619d, `thirdTeeExposure 0`, `memoryTrust 0.9`, `frozenThrow 3`) on the v2 defaults (`hookExactGate`,
/// `hookSnapAim`, `hookKeepFlying` on, `ropeCeilingCost 1`). Task 3.8, D-095.
pub fn preset_normal_v2() -> PlannerConfig {
    preset_normal().with_version(PlannerVersion::Upstream20261002)
}

/// `LIVE_PLANNER_CFG` of af49dfb (`bot.ts:678`: `thirdTeeExposure 0, memoryTrust 0.9, frozenThrow 3, launchExposure 1.5,
/// jumplessHazardCost 0.4`) on the v2 defaults: what the competitor's live bot actually plays. The two added values
/// differ from [`preset_normal_v2`] (`launchExposure 1.0`, `jumplessHazardCost 0.15` are the `PLANNER_DEFAULTS`).
pub fn preset_live_v2() -> PlannerConfig {
    PlannerConfig {
        launch_exposure: 1.5,
        jumpless_hazard_cost: 0.4,
        ..preset_normal_v2()
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

/// `WB_PLAN_OVERRIDES` of af49dfb (`bot.ts:250`): [`wb_overrides`] plus the two values it now names explicitly
/// (`launchExposure 1.0`, `jumplessHazardCost 0.15`, the defaults; they matter on top of [`preset_live_v2`], which
/// raises both).
pub fn wb_overrides_v2(base: PlannerConfig) -> PlannerConfig {
    PlannerConfig {
        launch_exposure: 1.0,
        jumpless_hazard_cost: 0.15,
        ..wb_overrides(base)
    }
}

/// The wayblock guard's plan (`wbGuardPlan`, `bot.ts:256`): `base` (a `WB_PLAN_OVERRIDES` config) plus
/// `wallDir` (`-1` left hall, `1` right), `frozenTargetSteps 16` and, with `WB_CHAIN=1`, `airChain`.
pub fn wb_guard_plan(base: PlannerConfig, wall_dir: i32, air_chain: bool) -> PlannerConfig {
    PlannerConfig {
        wall_dir,
        frozen_target_steps: 16,
        // `...(WB_CHAIN ? { airChain: true } : {})` only ever sets it: a base that has it on keeps it.
        air_chain: air_chain || base.air_chain,
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

/// [`preset_strong_wb`] on the af49dfb wayblock overrides ([`wb_overrides_v2`]).
pub fn preset_strong_wb_v2(base: PlannerConfig) -> PlannerConfig {
    PlannerConfig {
        population: 40,
        iterations: 3,
        budget_ms: 30.0,
        hard_ms: 36.0,
        ..wb_overrides_v2(base)
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
    fn classic_defaults_keep_every_v2_switch_off() {
        let d = PlannerConfig::default();
        assert_eq!(d.version(), PlannerVersion::Classic);
        assert_eq!(preset_normal().version(), PlannerVersion::Classic);
        assert_eq!(preset_low_cpu().version(), PlannerVersion::Classic);
        assert_eq!((d.wall_dir, d.air_chain), (0, false));
        assert_eq!(d.rope_ceiling_cost, 0.0);
        // Task 3.10: the finishing terms are hybrid-only; every preset the TS-parity path uses has them off.
        for c in [&d, &preset_normal(), &preset_normal_v2(), &preset_live_v2()] {
            assert_eq!(
                (c.frozen_drag_weight, c.held_forecast_weight, c.frozen_stage_weight),
                (0.0, 0.0, 0.0)
            );
            // Task 3.14: and so are the duel terms.
            assert_eq!((c.duel_loss_cost, c.duel_win_bonus), (0.0, 0.0));
            assert_eq!((c.ceiling_guard_cost, c.ceiling_guard_px), (0.0, 0.0));
        }
    }

    #[test]
    fn v2_presets_match_upstream_af49dfb() {
        let v2 = preset_normal_v2();
        assert_eq!(v2.version(), PlannerVersion::Upstream20261002);
        assert!(v2.hook_exact_gate && v2.hook_snap_aim && v2.hook_keep_flying);
        assert_eq!(v2.rope_ceiling_cost, 1.0);
        // Only the four switches differ from the classic preset.
        assert_eq!(
            v2.with_version(PlannerVersion::Classic),
            preset_normal(),
            "with_version(Classic) undoes the switches"
        );
        let live = preset_live_v2();
        assert_eq!((live.launch_exposure, live.jumpless_hazard_cost), (1.5, 0.4));
        assert_eq!((v2.launch_exposure, v2.jumpless_hazard_cost), (1.0, 0.15));
        assert_eq!(
            (live.memory_trust, live.frozen_throw, live.third_tee_exposure),
            (0.9, 3, 0.0)
        );
        let wb = wb_overrides_v2(live);
        assert_eq!((wb.launch_exposure, wb.jumpless_hazard_cost), (1.0, 0.15));
        let guard = wb_guard_plan(wb, -1, false);
        assert_eq!(
            (guard.wall_dir, guard.frozen_target_steps, guard.air_chain),
            (-1, 16, false)
        );
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
