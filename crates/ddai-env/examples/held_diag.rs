//! Task 3.10: why a block does not hold, in the arena.
//!
//! Reads the JSONL of an arena run, replays the games of the chosen condition(s) deterministically (the run's config gives the
//! brains; use a work-clock config) and, for every game the focal player won, follows the victim for the whole `after_ticks`
//! window that follows the deciding tick. One JSON line per game (`<out>/held.jsonl`):
//!
//! * the victim's state at the deciding tick: position, velocity, `freeze_left`, whether it touches a freeze tile, who hooks whom;
//! * the same every few ticks after it (`trace`), with our distance and whether we hold it on the rope;
//! * `escape_tick` (the first tick the victim is free again), `held_block`, and how it got free: `on_freeze` says whether the victim
//!   touched a freeze tile on the tick before it thawed (it cannot: a tee on a freeze tile is frozen), `drift_px` how far it was from
//!   the first freeze tile it was frozen on.
//!
//! ```text
//! cargo run --release -p ddai-env --example held_diag -- --config configs/arena/e021-diag.toml \
//!     --games ~/aiddnet/data/runs/E-021/diag --out ~/aiddnet/data/runs/E-021/diag-trace --threads 3 [--only clb-left] [--limit 100]
//! ```

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use ddai_env::arena::Arena;
use ddai_env::config::{PlayerSpec, RunConfig, builtin_brain};
use ddai_env::game::play_game_watched;
use ddai_env::observe;
use ddai_env::run::{layout_of, load_arenas};
use ddai_env::sim::PlayerSetup;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::seal::touches_freeze;
use rayon::prelude::*;
use serde_json::{Value, json};

struct Args {
    config: PathBuf,
    games: PathBuf,
    out: PathBuf,
    threads: usize,
    limit: Option<usize>,
    only: Option<String>,
    /// Also replay lost games (the victim is then the focal player).
    all: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        config: PathBuf::new(),
        games: PathBuf::new(),
        out: PathBuf::new(),
        threads: 3,
        limit: None,
        only: None,
        all: false,
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
            "--all" => a.all = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.config.as_os_str().is_empty() || a.games.as_os_str().is_empty() || a.out.as_os_str().is_empty() {
        return Err("usage: held_diag --config <toml> --games <dir> --out <dir> [--threads N] [--limit N] [--only <condition substring>] [--all]".into());
    }
    Ok(a)
}

struct Orig {
    condition: String,
    game: u32,
    result: String,
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
            out.push(Orig {
                condition: v["condition"].as_str().unwrap_or_default().to_string(),
                game: v["game"].as_u64().unwrap_or(0) as u32,
                result: v["result"].as_str().unwrap_or_default().to_string(),
                victim: v["victim"].as_i64().unwrap_or(-1) as i32,
                end_tick: v["end_tick"].as_i64().unwrap_or(0) as i32,
            });
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
    games.retain(|g| g.victim >= 0 && (a.all || g.result == "W"));
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
    eprintln!("{} games to replay", games.len());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads.max(1))
        .stack_size(32 << 20)
        .build()
        .map_err(|e| e.to_string())?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(a.out.join("held.jsonl")).map_err(|e| e.to_string())?);
    for chunk in games.chunks(24) {
        let lines: Vec<Result<String, String>> = pool.install(|| {
            chunk
                .par_iter()
                .map(|o| trace_game(&cfg, &arenas, o).map(|v| v.to_string()))
                .collect()
        });
        for l in lines {
            writeln!(out, "{}", l?).map_err(|e| e.to_string())?;
        }
        out.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

const SAMPLES: [i32; 16] = [0, 2, 4, 6, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250];

fn trace_game(cfg: &RunConfig, arenas: &BTreeMap<String, Arena>, o: &Orig) -> Result<Value, String> {
    let cond = cfg
        .condition
        .iter()
        .find(|c| c.name == o.condition)
        .ok_or_else(|| format!("condition {:?} not in the config", o.condition))?;
    let arena = &arenas[&cond.arena];
    let rules = cfg.rules_for(cond);
    let slots: Vec<PlayerSpec> = cond.slots();
    let players: Vec<PlayerSetup> = slots
        .iter()
        .map(|s| {
            let b = builtin_brain(s).map_err(|e| e.to_string())?;
            let label = b.name().to_string();
            Ok(PlayerSetup {
                brain: b,
                lag: s.lag,
                label,
            })
        })
        .collect::<Result<_, String>>()?;
    let seed = cfg.base_seed.wrapping_add(u64::from(o.game));
    let victim = o.victim;
    let focal = if victim == 0 { 1 } else { 0 };
    let mut trace: Vec<Value> = Vec::new();
    let mut first_frozen_pos: Option<(f32, f32)> = None;
    let mut escape: Option<i32> = None;
    let mut at_decision = Value::Null;
    let mut touching_before_escape = false;
    let mut ever_in_freeze_after = false;
    let end = o.end_tick;
    let rep = play_game_watched(arena, &rules, seed, layout_of(arena, o.game), players, &mut |sim, tick| {
        if tick < end {
            return true;
        }
        let dt = tick - end;
        let w = sim.pw.inner();
        let (Some(vc), Some(fc)) = (w.cores.get(victim as u8), w.cores.get(focal as u8)) else {
            return true;
        };
        let out = observe::is_out(w, victim);
        let alive = observe::is_alive(w, victim);
        let freeze_left = w.characters[victim as usize].as_ref().map_or(0, |c| c.freeze_time);
        let col = sim.pw.collision();
        let touching = touches_freeze(col, f64::from(vc.pos.x), f64::from(vc.pos.y));
        if out && alive && first_frozen_pos.is_none() {
            first_frozen_pos = Some((vc.pos.x, vc.pos.y));
        }
        if dt > 0 && !out && escape.is_none() {
            escape = Some(dt);
        }
        if escape.is_none() {
            touching_before_escape = touching;
        }
        ever_in_freeze_after |= dt > 0 && touching;
        let sample = json!({
            "dt": dt, "x": vc.pos.x, "y": vc.pos.y, "vx": (vc.vel.x * 10.0).round() / 10.0, "vy": (vc.vel.y * 10.0).round() / 10.0,
            "out": out, "alive": alive, "freeze_left": freeze_left, "touching_freeze": touching,
            "hooked_by_focal": fc.hooked_player() == victim, "focal_hook_state": fc.hook_state,
            "victim_hooks_focal": vc.hooked_player() == focal,
            "dist": ((vc.pos.x - fc.pos.x).hypot(vc.pos.y - fc.pos.y)).round(),
            "focal_out": observe::is_out(w, focal),
        });
        if dt == 0 {
            at_decision = sample.clone();
        }
        if SAMPLES.contains(&dt) {
            trace.push(sample);
        }
        true
    })
    .map_err(|e| e.to_string())?;
    let drift = first_frozen_pos.and_then(|(x0, y0)| {
        trace.last().map(|t| {
            let (x, y) = (t["x"].as_f64().unwrap_or(0.0), t["y"].as_f64().unwrap_or(0.0));
            (x - f64::from(x0)).hypot(y - f64::from(y0))
        })
    });
    Ok(json!({
        "condition": o.condition, "game": o.game, "result": o.result, "victim": victim, "end_tick": end,
        "held_block": rep.held_block, "escape_tick": rep.escape_tick.map(|t| t - end), "escape_dt": escape,
        "touching_freeze_before_escape": touching_before_escape, "touched_freeze_after_decision": ever_in_freeze_after,
        "drift_px": drift, "at_decision": at_decision, "trace": trace,
    }))
}
