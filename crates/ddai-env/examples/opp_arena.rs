//! Task 3.15 (E-028): plays the games of an arena config like `ddnet-ai arena run`, and also measures the **work of every decision** of slot 0 (the quantity
//! D-042 bounds: `(ticks x sim_tees + units + proposal_units) x 1.25 us`, from the brain's own telemetry), which the arena summary does not carry.
//!
//! Output: one JSONL line per game (`condition`, `game`, `result`, `credited`, `held`, `end_tick`; the format `docs/research/e026/compare.py` and
//! `docs/research/e028/compare.py` read), and per condition the credited wins, the work and the wall time of the decisions.
//!
//! ```text
//! cargo run --release -p ddai-env --example opp_arena -- --config configs/arena/e028-eval-m1-a.toml --threads 3 --jsonl ~/aiddnet/data/runs/E-028/m1-a.jsonl
//! ```

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use ddai_env::arena::Arena;
use ddai_env::config::{PlayerSpec, RunConfig, builtin_brain};
use ddai_env::game::play_game_watched;
use ddai_env::run::{layout_of, load_arenas};
use ddai_env::sim::PlayerSetup;
use ddai_planner::hybrid::WORK_US_PER_TEE_TICK;
use rayon::prelude::*;
use serde_json::{Value, json};

struct Args {
    config: PathBuf,
    threads: usize,
    games: Option<u32>,
    only: Option<String>,
    jsonl: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        config: PathBuf::new(),
        threads: 3,
        games: None,
        only: None,
        jsonl: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--config" => a.config = PathBuf::from(v()?),
            "--threads" => a.threads = v()?.parse().map_err(|e| format!("--threads: {e}"))?,
            "--games" => a.games = Some(v()?.parse().map_err(|e| format!("--games: {e}"))?),
            "--only" => a.only = Some(v()?),
            "--jsonl" => a.jsonl = Some(PathBuf::from(v()?)),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.config.as_os_str().is_empty() {
        return Err("usage: opp_arena --config <toml> [--threads N] [--games N] [--only <condition substring>] [--jsonl <file>]".into());
    }
    Ok(a)
}

/// One played game: its JSONL line, the work (ms) of each decision of slot 0, and the wall time (us) of each.
struct Row {
    line: Value,
    work_ms: Vec<f64>,
    wall_us: Vec<u32>,
    end_tick: i32,
    model_calls: u64,
}

fn play(cfg: &RunConfig, arenas: &BTreeMap<String, Arena>, cond_i: usize, g: u32) -> Result<Row, String> {
    let cond = &cfg.condition[cond_i];
    let arena = &arenas[&cond.arena];
    let rules = cfg.rules_for(cond);
    let slots: Vec<PlayerSpec> = cond.slots();
    let players: Vec<PlayerSetup> = slots
        .iter()
        .map(|s| {
            let b = builtin_brain(s).map_err(|e| e.to_string())?;
            let label = s.label.clone().unwrap_or_else(|| b.name().to_string());
            Ok(PlayerSetup {
                brain: b,
                lag: s.lag,
                label,
            })
        })
        .collect::<Result<_, String>>()?;
    let seed = cfg.base_seed.wrapping_add(u64::from(g));
    let mut work_ms = Vec::new();
    let mut model_calls = 0u64;
    let de = rules.decide_every.max(1);
    let rep = play_game_watched(arena, &rules, seed, layout_of(arena, g), players, &mut |sim, tick| {
        // The decisions of the step that just ended were made at tick `tick - 1`.
        if (tick - 1) % de == 0
            && let Some(t) = sim.players[0].brain.telemetry()
            && let Ok(v) = serde_json::from_str::<Value>(&t)
            && let Some(last) = v.get("last").filter(|l| !l.is_null())
            && let Some(w) = last.get("work")
        {
            let n = |k: &str| w.get(k).and_then(Value::as_u64).unwrap_or(0);
            let tees = last.get("sim_tees").and_then(Value::as_u64).unwrap_or(1).max(1);
            let units = n("units");
            model_calls += u64::from(units > 0);
            work_ms.push(((n("ticks") * tees + units + n("proposal_units")) as f64) * WORK_US_PER_TEE_TICK / 1000.0);
        }
        true
    })
    .map_err(|e| e.to_string())?;
    let line = json!({
        "condition": cond.name, "game": g, "result": format!("{:?}", rep.result), "credited": rep.credited, "held": rep.held,
        "held_block": rep.held_block, "end_tick": rep.end_tick, "spawns": rep.spawns,
    });
    Ok(Row {
        line,
        work_ms,
        wall_us: rep.decide_us.first().cloned().unwrap_or_default(),
        end_tick: rep.end_tick,
        model_calls,
    })
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
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
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads.max(1))
        .stack_size(32 << 20)
        .build()
        .map_err(|e| e.to_string())?;
    let mut jsonl = match &a.jsonl {
        Some(p) => Some(std::io::BufWriter::new(
            std::fs::File::create(p).map_err(|e| e.to_string())?,
        )),
        None => None,
    };
    println!(
        "| condition | games | W:L:D:T | credited wins | decided W/(W+L) | decisions | work ms p50 / p90 / p99 / max | wall us p50 / p99 | mean game ticks |"
    );
    println!("|---|---:|---|---|---:|---:|---|---|---:|");
    for (ci, cond) in cfg.condition.iter().enumerate() {
        if a.only.as_deref().is_some_and(|s| !cond.name.contains(s)) {
            continue;
        }
        let n = a.games.unwrap_or_else(|| cfg.games_for(cond));
        let res: Vec<Result<Row, String>> =
            pool.install(|| (0..n).into_par_iter().map(|g| play(&cfg, &arenas, ci, g)).collect());
        let mut rows = Vec::new();
        for r in res {
            rows.push(r?);
        }
        if let Some(f) = jsonl.as_mut() {
            for r in &rows {
                writeln!(f, "{}", r.line).map_err(|e| e.to_string())?;
            }
            f.flush().map_err(|e| e.to_string())?;
        }
        let (mut w, mut l, mut d, mut t) = (0, 0, 0, 0);
        for r in &rows {
            match r.line["result"].as_str().unwrap_or("") {
                "W" => w += 1,
                "L" => l += 1,
                "D" => d += 1,
                _ => t += 1,
            }
        }
        let cred = rows
            .iter()
            .filter(|r| r.line["result"] == "W" && r.line["credited"] == true)
            .count();
        let mut work: Vec<f64> = rows.iter().flat_map(|r| r.work_ms.iter().copied()).collect();
        let mut wall: Vec<f64> = rows
            .iter()
            .flat_map(|r| r.wall_us.iter().map(|&u| f64::from(u)))
            .collect();
        let calls: u64 = rows.iter().map(|r| r.model_calls).sum();
        let ticks: f64 = rows.iter().map(|r| f64::from(r.end_tick)).sum::<f64>() / f64::from(n);
        println!(
            "| {} | {n} | {w}:{l}:{d}:{t} | {cred} ({:.1}%) | {:.1}% | {} (model calls {calls}) | {:.2} / {:.2} / {:.2} / {:.2} | {:.0} / {:.0} | {ticks:.0} |",
            cond.name,
            100.0 * cred as f64 / f64::from(n),
            100.0 * f64::from(w) / f64::from((w + l).max(1)),
            work.len(),
            pct(&mut work, 0.5),
            pct(&mut work, 0.9),
            pct(&mut work, 0.99),
            pct(&mut work, 1.0),
            pct(&mut wall, 0.5),
            pct(&mut wall, 0.99),
        );
    }
    Ok(())
}
