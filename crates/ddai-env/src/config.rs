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
}

impl Default for Rules {
    fn default() -> Self {
        Rules {
            max_ticks: d_max_ticks(),
            after_ticks: d_after_ticks(),
            decide_every: d_decide_every(),
            credit_ticks: d_credit_ticks(),
            credit_required: None,
        }
    }
}

impl Rules {
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
    /// Planner: `normal` (default), `low` or `strong`.
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
    /// Shield time after the search, ms per tee in the world (default 0.25).
    #[serde(default)]
    pub shield_reserve_ms_per_tee: Option<f64>,
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
            // `step_ms` names the milliseconds per tee-tick (default 0.0022, i.e. 2.2 us).
            work_us = Some(spec.step_ms.map_or(2.2, |ms| ms * 1000.0));
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
        if let Some(v) = h.jumpless_anchor_bonus {
            cfg.planner.jumpless_anchor_bonus = v;
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
            let preset = match spec.preset.as_deref().unwrap_or("normal") {
                "normal" => PlannerPreset::Normal,
                "low" => PlannerPreset::Low,
                "strong" => PlannerPreset::Strong,
                other => return Err(EnvError::new(format!("planner: unknown preset {other:?}"))),
            };
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
            let brain = HybridBrain::new(cfg, clock, proposer).map_err(EnvError::new)?;
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
