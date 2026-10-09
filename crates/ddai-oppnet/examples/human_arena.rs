//! Task 3.24 (E-039), the imitation pilot, step 3: plays the games of an arena config (the 3.19 duel: F-DDrace round rules and the live view) like
//! `duel_stats`, with one addition: a hybrid whose `hybrid.proposer` is `humanprior:<head file>` gets the behaviour-cloning head of [`ddai_oppnet::bc`] as its
//! proposer. Everything else is `ddai_env`'s `builtin_brain`.
//!
//! Output: one JSONL line per game (`condition`, `game`, `result`, `credited`, `held`, `end_tick`; the format `docs/research/e028/compare.py` reads) with, for slot 0, the
//! decisions and how often the proposer's plan was generated and played (from the brain's telemetry), and per condition the credited wins.
//!
//! ```text
//! cargo run --release -p ddai-oppnet --example human_arena -- --config configs/arena/e039-prior.toml --threads 3 --jsonl out.jsonl [--games N] [--only SUBSTR]
//! ```

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use ddai_brain::Brain;
use ddai_env::arena::Arena;
use ddai_env::config::{PlayerSpec, RunConfig, builtin_brain, hybrid_config, lag_models_of};
use ddai_env::game::play_game_duel_watched;
use ddai_env::run::{layout_of, load_arenas};
use ddai_env::sim::PlayerSetup;
use ddai_oppnet::bc::{BcModel, HumanPriorProposer};
use ddai_planner::hybrid::HybridBrain;
use rayon::prelude::*;
use serde_json::{Value, json};

fn brain_for(spec: &PlayerSpec) -> Result<Box<dyn Brain>, String> {
    let proposer = spec.hybrid.as_ref().and_then(|h| h.proposer.as_deref());
    if spec.brain == "hybrid"
        && let Some(path) = proposer.and_then(|p| p.strip_prefix("humanprior:"))
    {
        let path = match path.strip_prefix("~/") {
            Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
            None => PathBuf::from(path),
        };
        let model = BcModel::load(&path)?;
        let (cfg, clock) = hybrid_config(spec).map_err(|e| e.to_string())?;
        let brain = HybridBrain::new(cfg, clock, Box::new(HumanPriorProposer::new(model)))?;
        return Ok(Box::new(brain));
    }
    builtin_brain(spec).map_err(|e| e.to_string())
}

fn play(cfg: &RunConfig, arenas: &BTreeMap<String, Arena>, cond_i: usize, g: u32) -> Result<Value, String> {
    let cond = &cfg.condition[cond_i];
    let arena = &arenas[&cond.arena];
    let rules = cfg.rules_for(cond);
    let slots: Vec<PlayerSpec> = cond.slots();
    let players: Vec<PlayerSetup> = slots
        .iter()
        .map(|s| {
            let b = brain_for(s)?;
            let label = s.label.clone().unwrap_or_else(|| b.name().to_string());
            Ok(PlayerSetup {
                brain: b,
                lag: s.lag,
                label,
            })
        })
        .collect::<Result<_, String>>()?;
    let seed = cfg.base_seed.wrapping_add(u64::from(g));
    let mut last_telemetry: Option<String> = None;
    let rep = play_game_duel_watched(
        arena,
        &rules,
        cond.duel.as_ref(),
        seed,
        layout_of(arena, g),
        players,
        lag_models_of(&slots),
        &mut |sim, _tick| {
            last_telemetry = sim.players[0].brain.telemetry();
            true
        },
    )
    .map_err(|e| e.to_string())?;
    let t: Value = last_telemetry
        .as_deref()
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .and_then(|v| v.get("totals").cloned())
        .unwrap_or(Value::Null);
    let n = |k: &str, sub: &str| t.get(k).and_then(|v| v.get(sub)).and_then(Value::as_u64).unwrap_or(0);
    Ok(json!({
        "condition": cond.name, "game": g, "result": format!("{:?}", rep.result), "credited": rep.credited, "held": rep.held,
        "held_block": rep.held_block, "end_tick": rep.end_tick,
        "decisions": t.get("decisions").and_then(Value::as_u64).unwrap_or(0),
        "proposal_generated": n("generated", "proposal"), "proposal_chosen": n("chosen", "proposal"),
        "work_proposal": t.get("work").and_then(|w| w.get("proposal")).and_then(Value::as_u64).unwrap_or(0),
    }))
}

fn main() -> Result<(), String> {
    let (mut config, mut threads, mut games, mut only, mut jsonl) =
        (PathBuf::new(), 3usize, None::<u32>, None::<String>, None::<PathBuf>);
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--config" => config = PathBuf::from(v()?),
            "--threads" => threads = v()?.parse().map_err(|e| format!("--threads: {e}"))?,
            "--games" => games = Some(v()?.parse().map_err(|e| format!("--games: {e}"))?),
            "--only" => only = Some(v()?),
            "--jsonl" => jsonl = Some(PathBuf::from(v()?)),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if config.as_os_str().is_empty() {
        return Err("usage: human_arena --config <toml> [--threads N] [--games N] [--only <condition substring>] [--jsonl <file>]".into());
    }
    let text = std::fs::read_to_string(&config).map_err(|e| format!("{}: {e}", config.display()))?;
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
        .num_threads(threads.clamp(1, 3))
        .build()
        .map_err(|e| e.to_string())?;
    let mut out = match &jsonl {
        Some(p) => Some(std::fs::File::create(p).map_err(|e| format!("{}: {e}", p.display()))?),
        None => None,
    };
    for (ci, cond) in cfg.condition.iter().enumerate() {
        if only.as_ref().is_some_and(|o| !cond.name.contains(o.as_str())) {
            continue;
        }
        let n = games.unwrap_or_else(|| cfg.games_for(cond));
        let rows: Vec<Result<Value, String>> =
            pool.install(|| (0..n).into_par_iter().map(|g| play(&cfg, &arenas, ci, g)).collect());
        let (mut w, mut l, mut cw, mut dec, mut gen_, mut ch) = (0u32, 0u32, 0u32, 0u64, 0u64, 0u64);
        for r in rows {
            let v = r?;
            match v["result"].as_str().unwrap_or("") {
                "W" => w += 1,
                "L" => l += 1,
                _ => {}
            }
            cw += u32::from(v["result"] == "W" && v["credited"] == true);
            dec += v["decisions"].as_u64().unwrap_or(0);
            gen_ += v["proposal_generated"].as_u64().unwrap_or(0);
            ch += v["proposal_chosen"].as_u64().unwrap_or(0);
            if let Some(f) = out.as_mut() {
                writeln!(f, "{v}").map_err(|e| e.to_string())?;
            }
        }
        println!(
            "{}: {n} games, W {w} / L {l}, credited wins {cw} ({:.1}%), decisions {dec}, proposal candidates {gen_}, proposal plans played {ch}",
            cond.name,
            100.0 * f64::from(cw) / f64::from(n.max(1))
        );
    }
    Ok(())
}
