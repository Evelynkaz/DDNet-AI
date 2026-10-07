//! Task 3.15 (E-028): records the games of an arena config as a dataset for the opponent-input predictor (`ddai-oppnet`).
//!
//! Plays the games of every condition (same seeds and layouts as `ddnet-ai arena run`) and keeps, for every world tick up to the first freeze
//! (which decides a 1v1 game), the snapshot-observable state of both tees, the inputs they applied, and the rays around slot 1 (the opponent).
//! One file per condition in `--out`: `<condition>.opp`, a `Vec<GameRec>` (see `ddai_oppnet::data`).
//!
//! ```text
//! cargo run --release -p ddai-env --example opp_record -- --config configs/arena/e028-data-joni.toml --out ~/aiddnet/data/runs/E-028/data/train --threads 3
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use ddai_env::arena::Arena;
use ddai_env::config::{PlayerSpec, RunConfig, builtin_brain};
use ddai_env::oppdata::record_game;
use ddai_env::run::{layout_of, load_arenas};
use ddai_env::sim::PlayerSetup;
use ddai_oppnet::data::GameRec;
use rayon::prelude::*;

struct Args {
    config: PathBuf,
    out: PathBuf,
    threads: usize,
    games: Option<u32>,
    only: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        config: PathBuf::new(),
        out: PathBuf::new(),
        threads: 3,
        games: None,
        only: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--config" => a.config = PathBuf::from(v()?),
            "--out" => a.out = PathBuf::from(v()?),
            "--threads" => a.threads = v()?.parse().map_err(|e| format!("--threads: {e}"))?,
            "--games" => a.games = Some(v()?.parse().map_err(|e| format!("--games: {e}"))?),
            "--only" => a.only = Some(v()?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.config.as_os_str().is_empty() || a.out.as_os_str().is_empty() {
        return Err(
            "usage: opp_record --config <toml> --out <dir> [--threads N] [--games N] [--only <condition substring>]"
                .into(),
        );
    }
    Ok(a)
}

fn record(cfg: &RunConfig, arenas: &BTreeMap<String, Arena>, cond_i: usize, g: u32) -> Result<(GameRec, char), String> {
    let cond = &cfg.condition[cond_i];
    let arena = &arenas[&cond.arena];
    let rules = cfg.rules_for(cond);
    let slots: Vec<PlayerSpec> = cond.slots();
    if slots.len() != 2 {
        return Err(format!(
            "condition {:?}: the recorder needs exactly two players",
            cond.name
        ));
    }
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
    let (rec, report) = record_game(arena, &rules, seed, layout_of(arena, g), players).map_err(|e| e.to_string())?;
    let r = format!("{:?}", report.result).chars().next().unwrap_or('?');
    Ok((rec, r))
}

fn file_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
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
    std::fs::create_dir_all(&a.out).map_err(|e| format!("{}: {e}", a.out.display()))?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads.max(1))
        .stack_size(32 << 20)
        .build()
        .map_err(|e| e.to_string())?;
    for (ci, cond) in cfg.condition.iter().enumerate() {
        if a.only.as_deref().is_some_and(|s| !cond.name.contains(s)) {
            continue;
        }
        let n = a.games.unwrap_or_else(|| cfg.games_for(cond));
        let t0 = std::time::Instant::now();
        let res: Vec<Result<(GameRec, char), String>> =
            pool.install(|| (0..n).into_par_iter().map(|g| record(&cfg, &arenas, ci, g)).collect());
        let mut games = Vec::new();
        let mut counts = BTreeMap::new();
        for r in res {
            let (rec, c) = r?;
            *counts.entry(c).or_insert(0u32) += 1;
            games.push(rec);
        }
        let ticks: usize = games.iter().map(|g| g.ticks.len()).sum();
        let path = a.out.join(format!("{}.opp", file_name(&cond.name)));
        ddai_oppnet::blob::write_blob(&path, &games, 3)?;
        println!(
            "{}: {n} games, {ticks} ticks, results {counts:?}, {:.0}s -> {}",
            cond.name,
            t0.elapsed().as_secs_f64(),
            path.display()
        );
    }
    Ok(())
}
