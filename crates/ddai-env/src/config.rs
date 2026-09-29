//! Run configuration (TOML): rules, conditions (arena + players + brains), seeds. The *effective*
//! config (after CLI overrides) is what gets hashed into the run record.

use std::collections::BTreeMap;

use ddai_brain::Brain;
use ddai_planner::brains::{
    ClockKind, IdleBrain, PlannerBrain, PlannerBrainConfig, PlannerMode, PlannerPreset, ScriptedBrain,
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
            label: None,
        }
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

/// The brains this crate knows: `idle`, `scripted`, `planner`.
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
        other => Err(EnvError::new(format!(
            "unknown brain {other:?} (built in: idle, scripted, planner)"
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
