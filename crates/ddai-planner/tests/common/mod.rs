//! Shared by the planner parity tests (`parity_planner.rs`, `parity_planner_freerun.rs`): the planner configuration a dump's
//! `preset`/`opponent`/`plannerVersion` header names. The generators (`tools/ts-trace/gen-planner-dump.mjs`,
//! `gen-planner-freerun.mjs`) build the same configs on the TS side.

#![cfg(feature = "ts-parity")]
#![allow(dead_code)]

use ddai_planner::config::{
    OpponentModel, PlannerConfig, PlannerVersion, preset_live_v2, preset_low_cpu, preset_normal, preset_strong_wb,
    wb_guard_plan, wb_overrides, wb_overrides_v2,
};

/// The config of `preset` for the dump of planner `version` (`"classic"` = c3c619d, the first corpus; `"upstream-2026-10-02"` =
/// af49dfb), with the opponent model of `opponent` and the fixed-iteration mode (no deadlines) the corpus is dumped in.
pub fn build_cfg(version: &str, preset: &str, opponent: &str) -> PlannerConfig {
    let (ver, v2) = match version {
        "classic" => (PlannerVersion::Classic, false),
        "upstream-2026-10-02" => (PlannerVersion::Upstream20261002, true),
        other => panic!("unknown planner version {other}"),
    };
    // The v2 defaults (`hookExactGate`, ... on) are the planner version's, so every named preset takes them; for `classic`
    // this changes nothing (the switches are off already).
    let base = preset_normal().with_version(ver);
    let mut cfg = match preset {
        "normal" => base,
        "low" => preset_low_cpu().with_version(ver),
        "strong" => preset_strong_wb(base),
        "wb" => wb_overrides(base),
        "live" if v2 => preset_live_v2(),
        "wblive" if v2 => wb_overrides_v2(preset_live_v2()),
        "guardl" if v2 => wb_guard_plan(wb_overrides_v2(preset_live_v2()), -1, false),
        "guardr" if v2 => wb_guard_plan(wb_overrides_v2(base), 1, false),
        "guardchain" if v2 => wb_guard_plan(wb_overrides_v2(preset_live_v2()), -1, true),
        other => panic!("unknown preset {other} for planner version {version}"),
    };
    match opponent {
        "hold" => {}
        "react" => cfg.opponent_model = OpponentModel::React,
        "mix" => cfg.opponent_mix = true,
        other => panic!("unknown opponent model {other}"),
    }
    cfg.budget_ms = 0.0;
    cfg.hard_ms = 0.0;
    cfg
}
