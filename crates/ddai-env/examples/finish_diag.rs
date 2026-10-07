//! Task 3.10b: why does a won, credited block not hold? Classifies every escape after a freeze.
//!
//! Reads the JSONL of an arena run, replays the credited wins whose block did not hold (`held_block == false`) deterministically (the run's
//! config gives the brains; use a work-clock config) and follows the victim for the whole `after_ticks` window after the deciding tick.
//! The escape of a game is put in exactly one class (first match):
//!
//! | class | meaning |
//! |---|---|
//! | `quick_thaw` | the victim was free again within 25 ticks of the freeze (an unfreeze tile, or a freeze at the very end of its timer): nothing to finish |
//! | `own_freeze` | we were frozen ourselves for 30 or more ticks of the freeze window (nobody else touched us) |
//! | `third_tee` | a third tee hooked or hit us during the freeze window (we were frozen by it for 30 or more ticks, or never touched the victim) |
//! | `input_miss` | we were free, and at least 3 live decisions of the window, a fifth of them or more, chose a plan that seals the victim under the model (frozen, resting in a freeze at the end of the rollout), yet the victim got away: the aim, the hook or the commitment did not carry it out |
//! | `lost_to_scoring` | no (sustained) live sealing pick, but an oracle search from the true state (a big pool, the finishing families, 9 steps = the live horizon) finds one: a plan existed within the horizon and the live search did not pick it |
//! | `no_plan_in_horizon` | no sustained live sealing pick; only the oracle with 16-step plans finds a sealing plan |
//! | `unreachable` | no sustained live sealing pick, and the oracle finds no sealing plan at all |
//! | `no_sample` | no decision of the window qualified for the oracle (the victim not frozen with 30 or more ticks of freeze left while we were free) |
//!
//! Whether we touched the victim (our hook held it, or it was pushed within 100 px of us) is reported besides the class (`touched`).
//!
//! The oracle samples the true state at ticks `0, 12, 24, 36, 60, 90` after the freeze (while the victim is frozen with at least 30 ticks left, we are
//! free, and the escape is at least 25 ticks away). It is a fresh fixed-work [`HybridBrain`] (population 60, 3 CEM iterations, finishing families and
//! the approach family on, `debug_pool`): a pool plan counts as sealing when its rollout under the cheap model ends with the victim frozen and resting in a freeze
//! (or dead) and us never out. Two oracles run: 9-step plans and 16-step plans (`frozen_steps`).
//!
//! ```text
//! cargo run --release -p ddai-env --example finish_diag -- --config configs/arena/e030-diag.toml \
//!     --games ~/aiddnet/data/runs/E-023-3.10b/diag --out ~/aiddnet/data/runs/E-023-3.10b/diag-classes --threads 3 [--only clb-left] [--limit 100] [--no-oracle]
//! ```

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use ddai_brain::{Action, Brain, Observation, ResetContext, WorldView};
use ddai_env::arena::Arena;
use ddai_env::config::{PlayerSpec, RunConfig, builtin_brain, hybrid_config};
use ddai_env::game::play_game_watched;
use ddai_env::observe;
use ddai_env::run::{layout_of, load_arenas};
use ddai_env::sim::{PlayerSetup, Sim};
use ddai_planner::brains::ClockKind;
use ddai_planner::hybrid::{DecisionTelemetry, HybridBrain, HybridConfig, NoProposer};
use ddai_planner::plan_world::PlanWorld;
use rayon::prelude::*;
use serde_json::{Value, json};

struct Args {
    config: PathBuf,
    games: PathBuf,
    out: PathBuf,
    threads: usize,
    limit: Option<usize>,
    only: Option<String>,
    oracle: bool,
    /// Put the positions of both tees every 10 ticks of the window into every record.
    trace: bool,
    /// Ticks after the freeze at which to replay the live brain's chosen plan open loop in the true world (`inspect`).
    inspect: Vec<i32>,
    /// Put the best sealing plan of the 16-step oracle, played open loop, into every oracle sample (`pools16[].traj`).
    oracle_traj: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        config: PathBuf::new(),
        games: PathBuf::new(),
        out: PathBuf::new(),
        threads: 3,
        limit: None,
        only: None,
        oracle: true,
        trace: false,
        inspect: Vec::new(),
        oracle_traj: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--config" => a.config = PathBuf::from(v()?),
            "--games" => a.games = PathBuf::from(v()?),
            "--out" => a.out = PathBuf::from(v()?),
            "--threads" => a.threads = v()?.parse().map_err(|e| format!("--threads: {e}"))?,
            "--limit" => a.limit = Some(v()?.parse().map_err(|e| format!("--limit: {e}"))?),
            "--only" => a.only = Some(v()?),
            "--no-oracle" => a.oracle = false,
            "--trace" => a.trace = true,
            "--oracle-traj" => a.oracle_traj = true,
            "--inspect" => {
                a.inspect = v()?
                    .split(',')
                    .map(|x| x.parse().map_err(|e| format!("--inspect: {e}")))
                    .collect::<Result<_, _>>()?
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.config.as_os_str().is_empty() || a.games.as_os_str().is_empty() || a.out.as_os_str().is_empty() {
        return Err("usage: finish_diag --config <toml> --games <dir> --out <dir> [--threads N] [--limit N] [--only <condition substring>] [--no-oracle] [--trace] [--inspect dt,dt,..]".into());
    }
    Ok(a)
}

struct Orig {
    condition: String,
    game: u32,
    victim: i32,
    end_tick: i32,
}

fn read_games(dir: &Path) -> Result<Vec<Orig>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    let mut out = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).map_err(|e| format!("{}: {e}", f.display()))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line).map_err(|e| format!("{}: {e}", f.display()))?;
            let win = v["result"].as_str() == Some("W");
            let credited = v["credited"].as_bool().unwrap_or(false);
            let held = v["held_block"].as_bool().unwrap_or(false);
            if win && credited && !held && v["victim"].as_i64().unwrap_or(-1) >= 0 {
                out.push(Orig {
                    condition: v["condition"].as_str().unwrap_or_default().to_string(),
                    game: v["game"].as_u64().unwrap_or(0) as u32,
                    victim: v["victim"].as_i64().unwrap_or(-1) as i32,
                    end_tick: v["end_tick"].as_i64().unwrap_or(0) as i32,
                });
            }
        }
    }
    Ok(out)
}

fn main() -> Result<(), String> {
    let a = parse_args()?;
    let text = std::fs::read_to_string(&a.config).map_err(|e| format!("{}: {e}", a.config.display()))?;
    let cfg = RunConfig::parse(&text).map_err(|e| e.to_string())?;
    let arenas_dir = cfg
        .arenas_dir
        .as_deref()
        .map_or_else(|| PathBuf::from("configs/arenas"), PathBuf::from);
    let map_dir = cfg.map_dir.as_deref().map_or_else(
        || PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("aiddnet/data/maps"),
        PathBuf::from,
    );
    let arenas = load_arenas(&cfg, &arenas_dir, &map_dir).map_err(|e| e.to_string())?;
    let mut games = read_games(&a.games)?;
    if let Some(sub) = &a.only {
        games.retain(|g| g.condition.contains(sub.as_str()));
    }
    if let Some(n) = a.limit {
        let mut seen: BTreeMap<String, usize> = BTreeMap::new();
        games.retain(|g| {
            let c = seen.entry(g.condition.clone()).or_insert(0);
            *c += 1;
            *c <= n
        });
    }
    std::fs::create_dir_all(&a.out).map_err(|e| e.to_string())?;
    eprintln!("{} escaped games to replay", games.len());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads.max(1))
        .stack_size(64 << 20)
        .build()
        .map_err(|e| e.to_string())?;
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(a.out.join("classes.jsonl")).map_err(|e| e.to_string())?);
    let mut all: Vec<Value> = Vec::new();
    for chunk in games.chunks(12) {
        let rows: Vec<Result<Value, String>> = pool.install(|| {
            chunk
                .par_iter()
                .map(|o| trace_game(&cfg, &arenas, o, a.oracle, a.trace, &a.inspect, a.oracle_traj))
                .collect()
        });
        for r in rows {
            let r = r?;
            writeln!(out, "{r}").map_err(|e| e.to_string())?;
            all.push(r);
        }
        out.flush().map_err(|e| e.to_string())?;
    }
    summarize(&all, &a.out)
}

/// Counts per condition and class (a markdown table on stdout and in `summary.md`).
fn summarize(rows: &[Value], out: &Path) -> Result<(), String> {
    const CLASSES: [&str; 8] = [
        "quick_thaw",
        "own_freeze",
        "third_tee",
        "input_miss",
        "lost_to_scoring",
        "no_plan_in_horizon",
        "unreachable",
        "no_sample",
    ];
    let mut by: BTreeMap<String, BTreeMap<&str, u32>> = BTreeMap::new();
    for r in rows {
        let cond = r["condition"].as_str().unwrap_or("?").to_string();
        let class = r["class"].as_str().unwrap_or("?");
        let class = CLASSES.iter().find(|c| **c == class).copied().unwrap_or("?");
        *by.entry(cond.clone()).or_default().entry(class).or_default() += 1;
        *by.entry("ALL".into()).or_default().entry(class).or_default() += 1;
    }
    let mut text = String::new();
    text.push_str("| condition | escapes | ");
    text.push_str(&CLASSES.join(" | "));
    text.push_str(" |\n|---|---:|");
    text.push_str(&"---:|".repeat(CLASSES.len()));
    text.push('\n');
    for (cond, m) in &by {
        let total: u32 = m.values().sum();
        text.push_str(&format!("| {cond} | {total} |"));
        for c in CLASSES {
            text.push_str(&format!(" {} |", m.get(c).copied().unwrap_or(0)));
        }
        text.push('\n');
    }
    // Oracle detail for the games the oracle judged.
    let mut det = [0u32; 4];
    for r in rows {
        if let Some(o) = r.get("oracle") {
            if o["top9_sealed"].as_bool() == Some(true) {
                det[0] += 1;
            }
            if o["any9_sealed"].as_bool() == Some(true) {
                det[1] += 1;
            }
            if o["any16_sealed"].as_bool() == Some(true) {
                det[2] += 1;
            }
            det[3] += 1;
        }
    }
    text.push_str(&format!(
        "\noracle games: {} (own best 9-step plan seals: {}, some 9-step plan seals: {}, some 16-step plan seals: {})\n",
        det[3], det[0], det[1], det[2]
    ));
    println!("{text}");
    std::fs::write(out.join("summary.md"), text).map_err(|e| e.to_string())
}

/// One oracle: a fresh fixed-work hybrid with a big pool.
fn oracle_brain(steps_long: i32) -> Result<HybridBrain, String> {
    let mut c = HybridConfig::fixed();
    c.workers = 1;
    c.debug_pool = true;
    c.finish_families = true;
    c.approach_plans = 12;
    c.planner.population = 60;
    c.planner.iterations = 3;
    c.frozen_steps = steps_long;
    c.frozen_steps_min_ticks = 30;
    HybridBrain::new(c, ClockKind::Wall, Box::new(NoProposer))
}

#[derive(Default, Clone)]
struct OracleOut {
    any_sealed: bool,
    top_sealed: bool,
    samples: u32,
    /// The best pool entries by score and the best sealing one (label, score, sealed, first steps of the plan), for `--pools`.
    top: Vec<Value>,
    /// With `--oracle-traj`: the best sealing plan played open loop in the true world, step by step.
    traj: Vec<Value>,
}

/// Asks `brain` for a decision from the true state and reads its pool: does some plan seal the victim without us going out, and is the one it
/// picks such a plan?
fn ask(brain: &mut HybridBrain, sim: &Sim, arena: &Arena, focal: i32, victim: i32, traj: bool) -> OracleOut {
    let mut o = OracleOut::default();
    let ids = sim.ids.clone();
    let Some(obs) = observe::observation(sim.pw.inner(), &arena.map, focal, &ids, Some(victim)) else {
        return o;
    };
    let view = WorldView {
        world: sim.pw.inner(),
        self_id: focal,
        lag_ticks: 0,
        in_flight: &[],
    };
    let _ = brain.decide_in(&obs, Some(&view));
    let Some(t) = brain.last_decision() else {
        return o;
    };
    o.samples = 1;
    let sealed = |i: usize| t.pool[i].enemy_sealed[0] == Some(true) && t.pool[i].self_out[0].is_none_or(|v| v == 0);
    o.any_sealed = (0..t.pool.len()).any(sealed);
    o.top_sealed = t.pick.is_some_and(sealed);
    let mut idx: Vec<usize> = (0..t.pool.len()).filter(|&i| t.pool[i].cheap.is_some()).collect();
    idx.sort_by(|&a, &b| {
        t.pool[b]
            .cheap
            .partial_cmp(&t.pool[a].cheap)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let describe = |rank: usize, i: usize| {
        let p = &t.pool[i];
        let steps: Vec<String> = p
            .plan
            .iter()
            .step_by(2)
            .map(|s| {
                format!(
                    "{}{}{}{}",
                    s.dir,
                    if s.jump != 0 { "J" } else { "" },
                    if s.hook != 0 { "H" } else { "" },
                    if s.fire != 0 { "F" } else { "" }
                )
            })
            .collect();
        json!({"rank": rank, "src": p.src.label(), "score": p.cheap, "sealed": sealed(i), "plan": steps.join(" ")})
    };
    for (rank, &i) in idx.iter().enumerate().take(4) {
        o.top.push(describe(rank, i));
    }
    // The three best sealing plans in full (every step), whatever their rank.
    for (rank, &i) in idx.iter().enumerate().filter(|&(_, &i)| sealed(i)).take(3) {
        let p = &t.pool[i];
        let steps: Vec<String> = p
            .plan
            .iter()
            .map(|s| {
                format!(
                    "{}{}{}{}",
                    s.dir,
                    if s.jump != 0 { "J" } else { "" },
                    if s.hook != 0 { "H" } else { "" },
                    if s.fire != 0 { "F" } else { "" }
                )
            })
            .collect();
        o.top
            .push(json!({"sealing_rank": rank, "src": p.src.label(), "score": p.cheap, "full_plan": steps.join(" ")}));
    }
    if let Some((rank, &i)) = idx.iter().enumerate().find(|&(_, &i)| sealed(i))
        && rank >= 4
    {
        o.top.push(describe(rank, i));
    }
    let best_sealed: Option<Vec<ddai_planner::planner::PlanStep>> =
        idx.iter().find(|&&i| sealed(i)).map(|&i| t.pool[i].plan.clone());
    if traj && let Some(plan) = best_sealed {
        let prev_in = sim.pw.inner().characters[focal as usize]
            .as_ref()
            .map_or_else(ddai_planner::types::empty_input, |c| {
                ddai_planner::physics_adapter::from_ddnet_input(&c.latest_input)
            });
        let inputs = brain.debug_plan_inputs(sim.pw.inner(), focal, victim, prev_in, &plan);
        let mut pw = ddai_planner::physics_adapter::PhysicsWorld::from_world(sim.pw.inner().clone(), arena.map.clone());
        pw.sync_from(sim.pw.inner());
        for (k, inp) in inputs.iter().enumerate() {
            for _ in 0..3 {
                pw.set_input(focal, *inp);
                pw.step();
            }
            if let (Some(m), Some(vt)) = (pw.get_tee(focal), pw.get_tee(victim)) {
                o.traj.push(json!({
                    "s": k, "in": format!("d{} j{} h{} f{}", inp.direction, inp.jump, inp.hook, inp.fire),
                    "us": [m.pos.x.round(), m.pos.y.round()], "victim": [vt.pos.x.round(), vt.pos.y.round()], "v_ft": vt.freeze_ticks_left,
                    "touch": ddai_planner::seal::touches_freeze(pw.collision(), vt.pos.x, vt.pos.y), "hooked": m.hooked_player == victim,
                }));
            }
        }
    }
    o
}

const ORACLE_AT: [i32; 6] = [0, 12, 24, 36, 60, 90];

/// What the focal player's hybrid chose at one decision (the live brain, wrapped so that its telemetry can be read).
#[derive(Clone, Copy, Default)]
struct Chosen {
    tick: i32,
    sealed: bool,
    hook: bool,
    fire: bool,
    generated: [u32; 6],
    evaluated: [u32; 6],
    out_of_time: bool,
}

/// The focal player's brain: a hybrid that plays exactly as the one `builtin_brain` makes, and logs what it chose.
struct Watch {
    inner: Rc<RefCell<HybridBrain>>,
    log: Rc<RefCell<Vec<Chosen>>>,
}

impl Watch {
    fn note(&self, tick: i32) {
        if let Some(t) = self.inner.borrow().last_decision() {
            self.log.borrow_mut().push(chosen_of(t, tick));
        }
    }
}

fn chosen_of(t: &DecisionTelemetry, tick: i32) -> Chosen {
    let first = t.chosen_plan.first();
    Chosen {
        tick,
        sealed: t.chosen_sealed,
        hook: first.is_some_and(|s| s.hook != 0),
        fire: first.is_some_and(|s| s.fire != 0),
        generated: t.generated,
        evaluated: t.evaluated,
        out_of_time: t.out_of_time,
    }
}

impl Brain for Watch {
    fn reset(&mut self, ctx: &ResetContext) {
        self.inner.borrow_mut().reset(ctx);
    }
    fn decide(&mut self, obs: &Observation) -> Action {
        let a = self.inner.borrow_mut().decide(obs);
        self.note(obs.tick);
        a
    }
    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        let a = self.inner.borrow_mut().decide_in(obs, view);
        self.note(obs.tick);
        a
    }
    fn name(&self) -> &str {
        // The name is a `&str` of the inner brain; it never changes, so leak-free: the hybrid's name is fixed per config.
        "watch-hybrid"
    }
    fn telemetry(&self) -> Option<String> {
        self.inner.borrow().telemetry()
    }
    fn last_plan(&self) -> Option<ddai_brain::PlanTelemetry> {
        self.inner.borrow().last_plan()
    }
}

#[derive(Default, Clone)]
struct TeeRec {
    pos: (f32, f32),
    vel: (f32, f32),
    out: bool,
}

#[allow(clippy::too_many_lines)]
fn trace_game(
    cfg: &RunConfig,
    arenas: &BTreeMap<String, Arena>,
    o: &Orig,
    oracle: bool,
    trace: bool,
    inspect: &[i32],
    oracle_traj: bool,
) -> Result<Value, String> {
    let cond = cfg
        .condition
        .iter()
        .find(|c| c.name == o.condition)
        .ok_or_else(|| format!("condition {:?} not in the config", o.condition))?;
    let arena = &arenas[&cond.arena];
    let rules = cfg.rules_for(cond);
    let slots: Vec<PlayerSpec> = cond.slots();
    let log: Rc<RefCell<Vec<Chosen>>> = Rc::default();
    let mut live_brain: Option<Rc<RefCell<HybridBrain>>> = None;
    let players: Vec<PlayerSetup> = slots
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let b: Box<dyn Brain> = if i == 0 && s.brain == "hybrid" {
                let (hc, clock) = hybrid_config(s).map_err(|e| e.to_string())?;
                let inner = Rc::new(RefCell::new(HybridBrain::new(hc, clock, Box::new(NoProposer))?));
                live_brain = Some(Rc::clone(&inner));
                Box::new(Watch {
                    inner,
                    log: Rc::clone(&log),
                })
            } else {
                builtin_brain(s).map_err(|e| e.to_string())?
            };
            let label = b.name().to_string();
            Ok(PlayerSetup {
                brain: b,
                lag: s.lag,
                label,
            })
        })
        .collect::<Result<_, String>>()?;
    let n = players.len();
    let seed = cfg.base_seed.wrapping_add(u64::from(o.game));
    let victim = o.victim;
    let focal = 0i32;
    let end = o.end_tick;

    // Per-tick records of the window.
    let mut vics: Vec<TeeRec> = Vec::new();
    let mut focs: Vec<TeeRec> = Vec::new();
    let mut hooked_victim: Vec<bool> = Vec::new();
    let mut third_hook: Vec<bool> = Vec::new();
    let mut third_near: Vec<f32> = Vec::new();
    let mut freeze_left: Vec<i32> = Vec::new();
    let mut alive_v: Vec<bool> = Vec::new();
    let mut touching: Vec<bool> = Vec::new();

    let (mut brain9, mut brain16) = if oracle {
        (Some(oracle_brain(0)?), Some(oracle_brain(16)?))
    } else {
        (None, None)
    };
    let reset = |b: &mut Option<HybridBrain>| {
        if let Some(b) = b.as_mut() {
            b.reset(&ResetContext {
                map: arena.map.clone(),
                self_id: focal,
                seed,
            });
        }
    };
    reset(&mut brain9);
    reset(&mut brain16);
    let mut o9 = OracleOut::default();
    let mut pools: Vec<Value> = Vec::new();
    let mut inspects: Vec<Value> = Vec::new();
    let mut pools16: Vec<Value> = Vec::new();
    let mut o16 = OracleOut::default();
    let mut top9_any = false;
    let mut escape_seen: Option<i32> = None;
    let _ = Arc::strong_count(&arena.map);

    let rep = play_game_watched(arena, &rules, seed, layout_of(arena, o.game), players, &mut |sim, tick| {
        if tick < end {
            return true;
        }
        let dt = tick - end;
        let w = sim.pw.inner();
        let (Some(vc), Some(fc)) = (w.cores.get(victim as u8), w.cores.get(focal as u8)) else {
            vics.push(TeeRec::default());
            focs.push(TeeRec::default());
            hooked_victim.push(false);
            third_hook.push(false);
            third_near.push(1e9);
            freeze_left.push(0);
            alive_v.push(false);
            touching.push(false);
            return true;
        };
        let v_out = observe::is_out(w, victim);
        let alive = observe::is_alive(w, victim);
        let f_out = observe::is_out(w, focal);
        let fl = w.characters[victim as usize].as_ref().map_or(0, |c| c.freeze_time);
        if dt > 0 && !v_out && escape_seen.is_none() {
            escape_seen = Some(dt);
        }
        vics.push(TeeRec {
            pos: (vc.pos.x, vc.pos.y),
            vel: (vc.vel.x, vc.vel.y),
            out: v_out,
        });
        focs.push(TeeRec {
            pos: (fc.pos.x, fc.pos.y),
            vel: (fc.vel.x, fc.vel.y),
            out: f_out,
        });
        hooked_victim.push(fc.hooked_player() == victim);
        let mut th = false;
        let mut near = 1e9f32;
        for j in 0..n as i32 {
            if j == focal || j == victim {
                continue;
            }
            if let Some(c) = w.cores.get(j as u8) {
                th |= c.hooked_player() == focal && observe::is_alive(w, j) && !observe::is_out(w, j);
                if observe::is_alive(w, j) && !observe::is_out(w, j) {
                    near = near.min((c.pos.x - fc.pos.x).hypot(c.pos.y - fc.pos.y));
                }
            }
        }
        third_hook.push(th);
        third_near.push(near);
        freeze_left.push(fl);
        alive_v.push(alive);
        touching.push(ddai_planner::seal::touches_freeze(
            sim.pw.collision(),
            f64::from(vc.pos.x),
            f64::from(vc.pos.y),
        ));
        // The live brain's chosen plan, replayed open loop in the true world (victim passive): where does the victim go, step by step?
        if inspect.contains(&dt)
            && v_out
            && alive
            && !f_out
            && let Some(lb) = live_brain.as_ref()
        {
            let mut b = lb.borrow_mut();
            if let Some(t) = b.last_decision().cloned() {
                let plan = t.chosen_plan.clone();
                let prev_in = w.characters[focal as usize]
                    .as_ref()
                    .map_or_else(ddai_planner::types::empty_input, |c| ddai_planner::physics_adapter::from_ddnet_input(&c.latest_input));
                let inputs = b.debug_plan_inputs(w, focal, victim, prev_in, &plan);
                // The template is the game's own world (its server settings and zones), not a fresh one of the map.
                let mut pw = ddai_planner::physics_adapter::PhysicsWorld::from_world(w.clone(), arena.map.clone());
                pw.sync_from(w);
                let mut steps_out = Vec::new();
                for (k, inp) in inputs.iter().enumerate() {
                    for _ in 0..3 {
                        pw.set_input(focal, *inp);
                        pw.step();
                    }
                    let (me_t, v_t) = (pw.get_tee(focal), pw.get_tee(victim));
                    if let (Some(m), Some(vt)) = (me_t, v_t) {
                        steps_out.push(json!({
                            "s": k, "in": format!("d{} j{} h{} f{}", inp.direction, inp.jump, inp.hook, inp.fire),
                            "us": [m.pos.x.round(), m.pos.y.round()], "us_frozen": m.frozen,
                            "victim": [vt.pos.x.round(), vt.pos.y.round()], "v_frozen": vt.frozen, "v_ft": vt.freeze_ticks_left,
                            "v_touch": ddai_planner::seal::touches_freeze(pw.collision(), vt.pos.x, vt.pos.y),
                            "hooked": m.hooked_player == victim,
                        }));
                    }
                }
                let plan_s: Vec<String> = plan
                    .iter()
                    .map(|s| format!("{}{}{}{}", s.dir, if s.jump != 0 { "J" } else { "" }, if s.hook != 0 { "H" } else { "" }, if s.fire != 0 { "F" } else { "" }))
                    .collect();
                inspects.push(json!({
                    "dt": dt, "chosen_sealed": t.chosen_sealed, "src": t.chosen.map(|c| c.label()), "plan": plan_s.join(" "),
                    "us0": [fc.pos.x.round(), fc.pos.y.round()], "victim0": [vc.pos.x.round(), vc.pos.y.round()], "steps": steps_out,
                }));
            }
        }
        // The oracle: only while the victim is still frozen with time left and we are free.
        if oracle && ORACLE_AT.contains(&dt) && v_out && alive && !f_out && fl >= 30 {
            if let Some(b) = brain9.as_mut() {
                let r = ask(b, sim, arena, focal, victim, false);
                pools.push(json!({"dt": dt, "any": r.any_sealed, "top": r.top}));
                o9.any_sealed |= r.any_sealed;
                top9_any |= r.top_sealed;
                o9.samples += r.samples;
            }
            if let Some(b) = brain16.as_mut() {
                let r = ask(b, sim, arena, focal, victim, oracle_traj);
                pools16.push(json!({"dt": dt, "any": r.any_sealed, "traj": r.traj, "us": [fc.pos.x.round(), fc.pos.y.round()], "victim": [vc.pos.x.round(), vc.pos.y.round()], "top": r.top.iter().filter(|e| e.get("full_plan").is_some()).cloned().collect::<Vec<_>>()}));
                o16.any_sealed |= r.any_sealed;
                o16.top_sealed |= r.top_sealed;
                o16.samples += r.samples;
            }
        }
        true
    })
    .map_err(|e| e.to_string())?;

    let escape = escape_seen.or(rep.escape_tick.map(|t| t - end));
    let e = escape.unwrap_or(vics.len() as i32).max(1) as usize;
    let e = e.min(vics.len());
    let dist = |i: usize| -> f32 {
        let (v, f) = (&vics[i], &focs[i]);
        (v.pos.0 - f.pos.0).hypot(v.pos.1 - f.pos.1)
    };
    let focal_out_ticks = (0..e).filter(|&i| focs[i].out).count();
    let focal_out_first = (0..e).find(|&i| focs[i].out);
    let hook_ticks = (0..e).filter(|&i| hooked_victim[i]).count();
    let third_hook_ticks = (0..e).filter(|&i| third_hook[i]).count();
    // A push on the victim: its velocity jumped by more than 3 px in one tick while we stood within 100 px.
    let mut pushes = 0;
    for i in 1..e {
        let dv = (vics[i].vel.0 - vics[i - 1].vel.0).hypot(vics[i].vel.1 - vics[i - 1].vel.1);
        if dv > 3.0 && dist(i) < 100.0 {
            pushes += 1;
        }
    }
    // A hit on us by a third tee: our velocity jumped while a free third tee stood within 100 px.
    let mut hits_on_us = 0;
    for i in 1..e {
        let dv = (focs[i].vel.0 - focs[i - 1].vel.0).hypot(focs[i].vel.1 - focs[i - 1].vel.1);
        if dv > 3.5 && third_near[i] < 100.0 {
            hits_on_us += 1;
        }
    }
    let third_events = third_hook_ticks >= 4 || hits_on_us > 0;
    let min_dist = (0..e).map(dist).fold(f32::INFINITY, f32::min);
    let touched = hook_ticks >= 3 || pushes > 0;
    let oracle_json = oracle.then(|| {
        json!({
            "pools9": pools, "pools16": pools16, "samples9": o9.samples, "samples16": o16.samples,
            "any9_sealed": o9.any_sealed, "top9_sealed": top9_any,
            "any16_sealed": o16.any_sealed, "top16_sealed": o16.top_sealed,
        })
    });
    // The live decisions of the freeze window (ticks `end..end + e`) that chose a plan sealing the victim.
    let live: Vec<Chosen> = log
        .borrow()
        .iter()
        .copied()
        .filter(|c| c.tick >= end && c.tick < end + e as i32)
        .collect();
    let live_sealed = live.iter().filter(|c| c.sealed).count();
    let class = if e < 25 {
        "quick_thaw"
    } else if focal_out_ticks >= 30 && !third_events {
        "own_freeze"
    } else if third_events && (focal_out_ticks >= 30 || !touched) {
        "third_tee"
    } else if live_sealed >= 3 && live_sealed * 5 >= live.len() {
        "input_miss"
    } else if !oracle || o9.samples == 0 {
        "no_sample"
    } else if o9.any_sealed {
        "lost_to_scoring"
    } else if o16.any_sealed {
        "no_plan_in_horizon"
    } else {
        "unreachable"
    };
    let first_touch_off = (0..e).find(|&i| !touching[i]).is_some();
    Ok(json!({
        "condition": o.condition, "game": o.game, "victim": victim, "end_tick": end, "class": class,
        "replay_ok": rep.end_tick == end && rep.victim == victim && !rep.held_block,
        "escape_dt": escape, "freeze_left0": freeze_left.first().copied().unwrap_or(0),
        "dist0": vics.first().map(|_| dist(0).round()), "min_dist": min_dist.round(),
        "touched": touched, "live_decisions": live.len(), "live_sealed": live_sealed,
        "live_hook": live.iter().filter(|c| c.hook).count(),
        "live_generated": (0..6).map(|k| live.iter().map(|c| u64::from(c.generated[k])).sum::<u64>()).collect::<Vec<_>>(),
        "live_evaluated": (0..6).map(|k| live.iter().map(|c| u64::from(c.evaluated[k])).sum::<u64>()).collect::<Vec<_>>(),
        "live_out_of_time": live.iter().filter(|c| c.out_of_time).count(),
        "nanny": !live.is_empty() && live.iter().filter(|c| c.hook).count() * 10 >= live.len() * 7 && min_dist < 45.0, "live_fire": live.iter().filter(|c| c.fire).count(),
        "hook_ticks": hook_ticks, "pushes": pushes, "focal_out_ticks": focal_out_ticks, "focal_out_first": focal_out_first,
        "third_hook_ticks": third_hook_ticks, "hits_on_us": hits_on_us, "victim_off_freeze": first_touch_off,
        "victim_alive_end": alive_v.last().copied().unwrap_or(false),
        "oracle": oracle_json,
        "inspect": inspects,
        "trace": trace.then(|| (0..e.min(vics.len())).filter(|i| *i < 16 || i % 4 == 0).map(|i| json!({
            "dt": i, "us": [focs[i].pos.0.round(), focs[i].pos.1.round()], "victim": [vics[i].pos.0.round(), vics[i].pos.1.round()],
            "us_vel": [(focs[i].vel.0 * 10.0).round() / 10.0, (focs[i].vel.1 * 10.0).round() / 10.0],
            "victim_vel": [(vics[i].vel.0 * 10.0).round() / 10.0, (vics[i].vel.1 * 10.0).round() / 10.0],
            "us_out": focs[i].out, "hooked": hooked_victim[i], "freeze_left": freeze_left[i], "touching": touching[i],
            "sealed_pick": log.borrow().iter().any(|c| c.tick >= end + i as i32 - 1 && c.tick <= end + i as i32 && c.sealed),
        })).collect::<Vec<_>>()),
    }))
}
