//! Technique scenarios (D-030, D-048): fixed start states on synthetic mini-maps or Copy Love Box
//! points, a fixed horizon and a success predicate, one TOML file each
//! (`configs/scenarios/t01-*.toml` ... the T1-T18 catalogue of `docs/research/block-knowledge.md`
//! §2.1). Task 3.5 uses the success rate per brain as its technique gate.
//!
//! A scenario has tee 0 = the *subject* (the brain under test) and any number of other tees with
//! a scripted behaviour (idle, the scripted bot, or a timeline). Start-state jitter (uniform, in
//! tiles) turns one setup into many trials: a deterministic brain would otherwise score 0/1.
//! Every scenario also carries a `reference` timeline for the subject -- an open-loop solution
//! that is checked to satisfy the predicate on the exact (jitter-free) start -- which is the proof
//! that the scenario is winnable at all; a scenario without one says so in `reference_note` and is
//! reported as unverified.

use std::path::Path;

use ddai_brain::Brain;
use ddai_jsmath::Rng;
use ddai_physics::core::{HOOK_GRABBED, WEAPON_GUN, WEAPON_HAMMER};
use ddai_planner::brains::ScriptedBrain;
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::WorldEvent;
use ddai_planner::vmath::Vec2;
use serde::{Deserialize, Serialize};

use crate::EnvError;
use crate::arena::{BuiltWorld, MapSource, build_world};
use crate::brains::{ScriptStep, TimelineBrain};
use crate::observe;
use crate::sim::{PlayerSetup, Sim, default_target};
use crate::stats::wilson95;

fn d_decide_every() -> i32 {
    2
}
fn d_trials() -> u32 {
    20
}
fn d_credit() -> i32 {
    50
}

/// One tee of a scenario. Index in `tee` = slot; slot 0 is the subject.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeeDef {
    /// Start position in tile units; `x.5, y.5` is the centre of tile `(x, y)`.
    pub pos: [f64; 2],
    /// Uniform start-position jitter, +-tiles, drawn per trial.
    #[serde(default)]
    pub jitter: [f64; 2],
    /// Start velocity, pixels per tick.
    #[serde(default)]
    pub vel: [f64; 2],
    /// Starts frozen for this many ticks (0 = free).
    #[serde(default)]
    pub freeze_ticks: i32,
    /// `"spent"`: the air jump is already used (no double jump left); default: all jumps ready.
    #[serde(default)]
    pub air_jump: Option<String>,
    /// Starts hooking this slot (hook already grabbed).
    #[serde(default)]
    pub hooking: Option<usize>,
    /// Active weapon: `"hammer"` or `"gun"` (default gun, as after a spawn).
    #[serde(default)]
    pub weapon: Option<String>,
    /// Behaviour of a non-subject tee: `"idle"` (default) or `"scripted"` (the scripted bot,
    /// targeting the subject). Ignored when `script` is non-empty. Slot 0 ignores this.
    #[serde(default)]
    pub behaviour: Option<String>,
    /// A timeline of held actions (see [`ScriptStep`]).
    #[serde(default)]
    pub script: Vec<ScriptStep>,
}

/// A success predicate over the recorded trace. Ticks are world ticks; tick 0 is the start state.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Pred {
    /// All of them hold.
    All { of: Vec<Pred> },
    /// At least one holds.
    Any { of: Vec<Pred> },
    /// `tee` goes out (dead or frozen: an onset) at a tick in `from..=within`, and, when
    /// `credited_to` is given, that slot hooked/hammered it at most 50 ticks before.
    Out {
        tee: usize,
        #[serde(default)]
        from: i32,
        #[serde(default)]
        within: Option<i32>,
        #[serde(default)]
        credited_to: Option<usize>,
    },
    /// `tee` is not out at any tick `0..=until` (default: the horizon).
    NotOut {
        tee: usize,
        #[serde(default)]
        until: Option<i32>,
    },
    /// `tee` has at most `n` onsets up to `until` (default: the horizon).
    OnsetsAtMost {
        tee: usize,
        n: u32,
        #[serde(default)]
        until: Option<i32>,
    },
    /// `tee` is out at every tick in `from..=to`.
    StaysOut { tee: usize, from: i32, to: i32 },
    /// `tee` is not out at tick `tick`.
    Free { tee: usize, tick: i32 },
    /// `tee`'s hook is grabbed (on `"wall"`, a `"player"` or `"any"`) at some tick `<= within`.
    HookGrabbed {
        tee: usize,
        #[serde(default = "d_any")]
        target: String,
        within: i32,
    },
    /// `tee` stands on the ground at some tick `<= within`.
    Grounded { tee: usize, within: i32 },
    /// `tee` is inside the tile box at tick `tick`.
    InBox {
        tee: usize,
        tick: i32,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
    },
}

fn d_any() -> String {
    "any".to_string()
}

/// One scenario file.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioDef {
    /// `T1` ... `T18`.
    pub id: String,
    pub name: String,
    /// Where the technique comes from (catalogue row and source).
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub description: String,
    pub map: MapSource,
    /// Ticks played (50 = 1 s).
    pub horizon: i32,
    #[serde(default = "d_decide_every")]
    pub decide_every: i32,
    /// Trials per brain (start-state jitter draws).
    #[serde(default = "d_trials")]
    pub trials: u32,
    /// Ticks within which a touch counts as credit (the same 50 as in games).
    #[serde(default = "d_credit")]
    pub credit_ticks: i32,
    /// `true` for discipline scenarios where doing nothing is the right answer (e.g. T8).
    #[serde(default)]
    pub idle_should_pass: bool,
    /// Spawn the tees in reverse slot order (last slot first). The spawn order decides which hook
    /// is strong in a hook duel (`m_StrongWeakId`); client ids stay equal to the slot.
    #[serde(default)]
    pub spawn_reverse: bool,
    pub tee: Vec<TeeDef>,
    pub success: Pred,
    /// Open-loop solution for the subject; empty = unverified (say why in `reference_note`).
    #[serde(default)]
    pub reference: Vec<ScriptStep>,
    #[serde(default)]
    pub reference_note: String,
}

impl ScenarioDef {
    pub fn parse(text: &str) -> Result<ScenarioDef, EnvError> {
        let def: ScenarioDef = toml::from_str(text).map_err(|e| EnvError::new(format!("scenario: {e}")))?;
        def.validate()?;
        Ok(def)
    }

    pub fn validate(&self) -> Result<(), EnvError> {
        let bad = |m: &str| Err(EnvError::new(format!("scenario {}: {m}", self.id)));
        if self.tee.len() < 2 {
            return bad("needs a subject and at least one other tee");
        }
        if self.horizon <= 0 || self.decide_every <= 0 || self.trials == 0 {
            return bad("horizon, decide_every and trials must be positive");
        }
        for t in &self.tee {
            if t.hooking.is_some_and(|h| h >= self.tee.len()) {
                return bad("hooking refers to a missing slot");
            }
        }
        fn slots(p: &Pred, out: &mut Vec<usize>) {
            match p {
                Pred::All { of } | Pred::Any { of } => of.iter().for_each(|q| slots(q, out)),
                Pred::Out { tee, credited_to, .. } => {
                    out.push(*tee);
                    out.extend(credited_to);
                }
                Pred::NotOut { tee, .. }
                | Pred::OnsetsAtMost { tee, .. }
                | Pred::StaysOut { tee, .. }
                | Pred::Free { tee, .. }
                | Pred::HookGrabbed { tee, .. }
                | Pred::Grounded { tee, .. }
                | Pred::InBox { tee, .. } => out.push(*tee),
            }
        }
        let mut used = Vec::new();
        slots(&self.success, &mut used);
        if used.iter().any(|&s| s >= self.tee.len()) {
            return bad("the success predicate refers to a missing slot");
        }
        Ok(())
    }

    /// Reads every `*.toml` in `dir`, sorted by file name.
    pub fn load_dir(dir: &Path) -> Result<Vec<ScenarioDef>, EnvError> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| EnvError::new(format!("reading {}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "toml"))
            .collect();
        paths.sort();
        let mut out = Vec::new();
        for p in paths {
            let text = std::fs::read_to_string(&p).map_err(|e| EnvError::new(format!("{}: {e}", p.display())))?;
            out.push(ScenarioDef::parse(&text).map_err(|e| EnvError::new(format!("{}: {e}", p.display())))?);
        }
        Ok(out)
    }
}

/// One tee at one tick.
#[derive(Debug, Clone, Copy)]
pub struct TeeSnap {
    pub out: bool,
    /// `out` just became true.
    pub onset: bool,
    /// Who hooked/hammered this tee at most `credit_ticks` before the onset.
    pub credit: Option<usize>,
    pub pos: [f32; 2],
    pub hook_state: i32,
    pub hooked_player: i32,
    pub grounded: bool,
}

/// The recorded run of a scenario: `snaps[t][slot]`, `t` in `0..=horizon`.
#[derive(Debug, Clone)]
pub struct Trace {
    pub snaps: Vec<Vec<TeeSnap>>,
}

/// Evaluates a predicate on a recorded trace.
pub fn eval(pred: &Pred, tr: &Trace, horizon: i32) -> bool {
    pred.eval(tr, horizon)
}

impl Pred {
    fn eval(&self, tr: &Trace, horizon: i32) -> bool {
        let last = horizon.min(tr.snaps.len() as i32 - 1);
        let at = |t: i32, tee: usize| &tr.snaps[t as usize][tee];
        match self {
            Pred::All { of } => of.iter().all(|p| p.eval(tr, horizon)),
            Pred::Any { of } => of.iter().any(|p| p.eval(tr, horizon)),
            Pred::Out {
                tee,
                from,
                within,
                credited_to,
            } => ((*from).max(1)..=within.unwrap_or(last).min(last)).any(|t| {
                let s = at(t, *tee);
                s.onset && credited_to.is_none_or(|c| s.credit == Some(c))
            }),
            Pred::NotOut { tee, until } => (0..=until.unwrap_or(last).min(last)).all(|t| !at(t, *tee).out),
            Pred::OnsetsAtMost { tee, n, until } => {
                (1..=until.unwrap_or(last).min(last))
                    .filter(|&t| at(t, *tee).onset)
                    .count() as u32
                    <= *n
            }
            Pred::StaysOut { tee, from, to } => (*from..=(*to).min(last)).all(|t| at(t, *tee).out),
            Pred::Free { tee, tick } => !at((*tick).min(last), *tee).out,
            Pred::HookGrabbed { tee, target, within } => (0..=(*within).min(last)).any(|t| {
                let s = at(t, *tee);
                s.hook_state == HOOK_GRABBED
                    && match target.as_str() {
                        "wall" => s.hooked_player < 0,
                        "player" => s.hooked_player >= 0,
                        _ => true,
                    }
            }),
            Pred::Grounded { tee, within } => (0..=(*within).min(last)).any(|t| at(t, *tee).grounded),
            Pred::InBox {
                tee,
                tick,
                x0,
                y0,
                x1,
                y1,
            } => {
                let s = at((*tick).min(last), *tee);
                let (x, y) = (f64::from(s.pos[0]) / 32.0, f64::from(s.pos[1]) / 32.0);
                (*x0..=*x1).contains(&x) && (*y0..=*y1).contains(&y)
            }
        }
    }
}

/// The result of one trial.
#[derive(Debug, Clone)]
pub struct TrialOutcome {
    pub success: bool,
    pub trace: Trace,
}

fn snapshot(sim: &Sim, was_out: &[bool], touch: &[Option<(usize, i32)>], credit_ticks: i32) -> Vec<TeeSnap> {
    let now = sim.tick();
    let w = sim.pw.inner();
    sim.ids
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            let out = observe::is_out(w, id);
            let onset = out && !was_out[i];
            let credit = touch[i].and_then(|(by, t)| (onset && now - t <= credit_ticks).then_some(by));
            let obs = observe::character_observation(w, id);
            let pos = w.cores.get(id as u8).map_or([0.0, 0.0], |c| [c.pos.x, c.pos.y]);
            TeeSnap {
                out,
                onset,
                credit,
                pos,
                hook_state: obs.as_ref().map_or(-1, |o| o.hook_state),
                hooked_player: obs.as_ref().map_or(-1, |o| o.hooked_player),
                grounded: obs.as_ref().is_some_and(|o| o.grounded),
            }
        })
        .collect()
}

/// Runs one trial of `def` with `subject` as tee 0. `seed` and `trial` fix the start jitter (`jitter =
/// false` ignores it -- the exact start state, used to check the reference solution).
pub fn run_trial(
    def: &ScenarioDef,
    world: &BuiltWorld,
    subject: Box<dyn Brain>,
    subject_lag: u32,
    seed: u64,
    trial: u32,
    jitter: bool,
) -> Result<TrialOutcome, EnvError> {
    let mut pw = PhysicsWorld::from_world(world.world.clone(), world.map.clone());
    let mut rng = Rng::new(
        seed.wrapping_add(u64::from(trial).wrapping_mul(7919))
            .wrapping_mul(2_654_435_761) as u32,
    );
    let mut positions: Vec<Vec2> = Vec::new();
    for t in &def.tee {
        // Always draw two numbers per tee so the stream does not depend on which tees jitter.
        let (u, v) = (rng.next_float(), rng.next_float());
        let (jx, jy) = if jitter {
            ((u * 2.0 - 1.0) * t.jitter[0], (v * 2.0 - 1.0) * t.jitter[1])
        } else {
            (0.0, 0.0)
        };
        positions.push(Vec2 {
            x: (t.pos[0] + jx) * 32.0,
            y: (t.pos[1] + jy) * 32.0,
        });
    }
    let spawn_order: Vec<usize> = if def.spawn_reverse {
        (0..def.tee.len()).rev().collect()
    } else {
        (0..def.tee.len()).collect()
    };
    for i in spawn_order {
        pw.add_tee(i as i32, positions[i]);
    }
    for (i, t) in def.tee.iter().enumerate() {
        let mut st = pw
            .get_tee(i as i32)
            .ok_or_else(|| EnvError::new("tee vanished at spawn"))?;
        st.vel = Vec2 {
            x: t.vel[0],
            y: t.vel[1],
        };
        if t.freeze_ticks > 0 {
            st.frozen = true;
            st.freeze_ticks_left = i64::from(t.freeze_ticks);
            st.frozen_for = Some(0);
        }
        if t.air_jump.as_deref() == Some("spent") {
            st.jumped = 3;
            st.jumped_total = Some(2);
            st.jumps_left = 0;
        }
        match t.weapon.as_deref() {
            Some("hammer") => st.active_weapon = WEAPON_HAMMER,
            Some("gun") | None => st.active_weapon = WEAPON_GUN,
            Some(other) => return Err(EnvError::new(format!("scenario {}: unknown weapon {other:?}", def.id))),
        }
        if let Some(h) = t.hooking {
            st.hook_state = HOOK_GRABBED;
            st.hooked_player = h as i32;
            st.hook_pos = positions[h];
        }
        pw.apply_tee_state(i as i32, &st);
    }
    let mut players = vec![PlayerSetup {
        label: subject.name().to_string(),
        brain: subject,
        lag: subject_lag,
    }];
    for (i, t) in def.tee.iter().enumerate().skip(1) {
        let brain: Box<dyn Brain> = if !t.script.is_empty() {
            Box::new(TimelineBrain::from_script(&format!("script-{i}"), &t.script))
        } else if t.behaviour.as_deref() == Some("scripted") {
            Box::new(ScriptedBrain::new())
        } else {
            Box::new(ddai_brain::IdleBrain)
        };
        players.push(PlayerSetup {
            label: brain.name().to_string(),
            brain,
            lag: 0,
        });
    }
    let mut sim = Sim::new(pw, world.map.clone(), players, def.decide_every, seed);
    let n = def.tee.len();
    let mut was_out = vec![false; n];
    let mut touch: Vec<Option<(usize, i32)>> = vec![None; n];
    let mut snaps = Vec::with_capacity(def.horizon as usize + 1);
    let first = snapshot(&sim, &was_out, &touch, def.credit_ticks);
    for (i, s) in first.iter().enumerate() {
        was_out[i] = s.out;
    }
    snaps.push(first);
    for _ in 0..def.horizon {
        let events = sim.step(&default_target);
        let now = sim.tick();
        for e in events {
            if let WorldEvent::HammerHit { from, to } = e
                && (0..n as i32).contains(&from)
                && (0..n as i32).contains(&to)
            {
                touch[to as usize] = Some((from as usize, now));
            }
        }
        for i in 0..n {
            let h = observe::hooked_player(sim.pw.inner(), sim.ids[i]);
            if h >= 0 && (h as usize) < n {
                touch[h as usize] = Some((i, now));
            }
        }
        let snap = snapshot(&sim, &was_out, &touch, def.credit_ticks);
        for (i, s) in snap.iter().enumerate() {
            was_out[i] = s.out;
        }
        snaps.push(snap);
    }
    let trace = Trace { snaps };
    let success = def.success.eval(&trace, def.horizon);
    Ok(TrialOutcome { success, trace })
}

/// The scenario's reference solution as a brain (for the subject).
pub fn reference_brain(def: &ScenarioDef) -> Box<dyn Brain> {
    Box::new(TimelineBrain::from_script("reference", &def.reference))
}

/// Success counts of one brain on one scenario.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScenarioScore {
    pub id: String,
    pub name: String,
    pub brain: String,
    pub trials: u32,
    pub successes: u32,
    pub rate: f64,
    pub lo: f64,
    pub hi: f64,
}

impl ScenarioScore {
    pub fn new(def: &ScenarioDef, brain: &str, trials: u32, successes: u32) -> Self {
        let (lo, hi) = wilson95(f64::from(successes), f64::from(trials));
        ScenarioScore {
            id: def.id.clone(),
            name: def.name.clone(),
            brain: brain.to_string(),
            trials,
            successes,
            rate: if trials == 0 {
                0.0
            } else {
                f64::from(successes) / f64::from(trials)
            },
            lo,
            hi,
        }
    }
}

/// Loads the world of a scenario.
pub fn load_world(def: &ScenarioDef, map_dir: &Path) -> Result<BuiltWorld, EnvError> {
    build_world(&def.map, map_dir, &format!("scenario {}", def.id))
}

/// Plays `def.trials` (or `trials`) trials with a fresh subject per trial made by `make`.
pub fn score_brain(
    def: &ScenarioDef,
    world: &BuiltWorld,
    label: &str,
    subject_lag: u32,
    seed: u64,
    trials: Option<u32>,
    make: &(dyn Fn() -> Result<Box<dyn Brain>, EnvError> + Sync),
) -> Result<ScenarioScore, EnvError> {
    use rayon::prelude::*;
    let trials = trials.unwrap_or(def.trials);
    let results: Vec<Result<bool, EnvError>> = (0..trials)
        .into_par_iter()
        .map(|k| Ok(run_trial(def, world, make()?, subject_lag, seed, k, true)?.success))
        .collect();
    let mut ok = 0;
    for r in results {
        ok += u32::from(r?);
    }
    Ok(ScenarioScore::new(def, label, trials, ok))
}

/// Markdown (Russian) success table: one row per scenario, one column per brain.
pub fn markdown(scores: &[ScenarioScore], defs: &[ScenarioDef]) -> String {
    use std::fmt::Write as _;
    let mut brains: Vec<&str> = Vec::new();
    for s in scores {
        if !brains.contains(&s.brain.as_str()) {
            brains.push(&s.brain);
        }
    }
    let mut out = String::new();
    let _ = writeln!(
        out,
        "| # | Приём | эталон |{}",
        brains.iter().map(|b| format!(" {b} |")).collect::<String>()
    );
    let _ = writeln!(
        out,
        "|---|---|---|{}",
        brains.iter().map(|_| "---|").collect::<String>()
    );
    for d in defs {
        let verified = if d.reference.is_empty() { "нет" } else { "да" };
        let cells: String = brains
            .iter()
            .map(|b| {
                scores
                    .iter()
                    .find(|s| s.id == d.id && s.brain == *b)
                    .map_or(" — |".to_string(), |s| {
                        format!(
                            " {}/{} ({:.0}%; {:.0}–{:.0}) |",
                            s.successes,
                            s.trials,
                            100.0 * s.rate,
                            100.0 * s.lo,
                            100.0 * s.hi
                        )
                    })
            })
            .collect();
        let _ = writeln!(out, "| {} | {} | {verified} |{cells}", d.id, d.name);
    }
    out
}

/// A scenario run config: the brains to score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioRunConfig {
    pub name: String,
    #[serde(default = "d_seed")]
    pub seed: u64,
    /// Overrides every scenario's own trial count.
    #[serde(default)]
    pub trials: Option<u32>,
    #[serde(default)]
    pub brain: Vec<crate::config::PlayerSpec>,
}

fn d_seed() -> u64 {
    1
}

impl ScenarioRunConfig {
    pub fn parse(text: &str) -> Result<ScenarioRunConfig, EnvError> {
        let cfg: ScenarioRunConfig =
            toml::from_str(text).map_err(|e| EnvError::new(format!("scenario run config: {e}")))?;
        if cfg.brain.is_empty() {
            return Err(EnvError::new("scenario run config: no [[brain]]"));
        }
        Ok(cfg)
    }
}

/// Scores every brain on every scenario. Scenarios sharing a map file load it once per scenario
/// (they are small); trials of one brain run in parallel.
pub fn run_scenarios(
    defs: &[ScenarioDef],
    cfg: &ScenarioRunConfig,
    factory: &crate::config::BrainFactory,
    map_dir: &Path,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ScenarioScore>, EnvError> {
    let mut scores = Vec::new();
    for def in defs {
        let world = load_world(def, map_dir)?;
        for spec in &cfg.brain {
            let label = spec.label.clone().unwrap_or_else(|| spec.brain.clone());
            let score = score_brain(def, &world, &label, spec.lag, cfg.seed, cfg.trials, &|| factory(spec))?;
            progress(&format!("{} {}: {}/{}", def.id, label, score.successes, score.trials));
            scores.push(score);
        }
    }
    Ok(scores)
}
