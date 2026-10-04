//! Task 3.7b (E-017): the loss diagnosis of the hybrid against the fixed planner.
//!
//! Reads the JSONL of an arena run (`ddnet-ai arena run --config configs/arena/e017-diagnosis.toml`), picks the games
//! the hybrid did not win by its own credited block (or, with `--select won`, the ones it did, as a control), replays
//! each one deterministically with a `DiagBrain` (the hybrid plus a shadowing fixed planner, `ddai_planner::diag`), and
//! writes what the analysis script needs:
//!
//! For every decision in the last `--window` ticks before the game was decided it writes (`<out>/truth.jsonl`, one line per game) the
//! hybrid's whole pool with scores, the planner's plan and its score under the hybrid's evaluator, what every plan does in the true world
//! against the opponent's actual inputs, what the evaluator would pick if it knew those inputs (the oracle), and the opponent as a planner.
//!
//! The replay of a game must be bit-identical to the original (the decision hash of slot 0 is compared and reported).
//!
//! ```text
//! cargo run --release -p ddai-env --example hybrid_losses -- --config configs/arena/e017-diagnosis.toml \
//!     --games ~/aiddnet/data/runs/E-017/diag --out ~/aiddnet/data/runs/E-017/truth --threads 3
//! ```

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use ddai_env::arena::Arena;
use ddai_env::config::{PlayerSpec, RunConfig, builtin_brain, builtin_proposer, hybrid_config};
use ddai_env::game::{GameReport, play_game};
use ddai_env::run::{layout_of, load_arenas};
use ddai_env::sim::PlayerSetup;
use ddai_planner::diag::{DecisionRecord, DiagBrain, DiagInner, DiagOptions};
use ddai_planner::hybrid::HybridBrain;
use ddai_planner::hybrid::search::{DebugScore, Source, TruthOutcome};
use ddai_planner::planner::PlanStep;
use rayon::prelude::*;
use serde_json::{Value, json};

struct Args {
    config: PathBuf,
    games: PathBuf,
    out: PathBuf,
    select: String,
    threads: usize,
    window: i32,
    limit: Option<usize>,
    only: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        config: PathBuf::new(),
        games: PathBuf::new(),
        out: PathBuf::new(),
        select: "lost".into(),
        threads: 3,
        window: 60,
        limit: None,
        only: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--config" => a.config = PathBuf::from(v()?),
            "--games" => a.games = PathBuf::from(v()?),
            "--out" => a.out = PathBuf::from(v()?),
            "--select" => a.select = v()?,
            "--threads" => a.threads = v()?.parse().map_err(|e| format!("--threads: {e}"))?,
            "--window" => a.window = v()?.parse().map_err(|e| format!("--window: {e}"))?,
            "--limit" => a.limit = Some(v()?.parse().map_err(|e| format!("--limit: {e}"))?),
            "--only" => a.only = Some(v()?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.config.as_os_str().is_empty() || a.games.as_os_str().is_empty() || a.out.as_os_str().is_empty() {
        return Err("usage: hybrid_losses --config <toml> --games <dir> --out <dir> [--select lost|won|all] [--threads N] [--window ticks] [--limit N] [--only <condition substring>]".into());
    }
    Ok(a)
}

/// One game of the original run.
struct Orig {
    condition: String,
    game: u32,
    result: String,
    credited: bool,
    end_tick: i32,
    hash: String,
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
                credited: v["credited"].as_bool().unwrap_or(false),
                end_tick: v["end_tick"].as_i64().unwrap_or(0) as i32,
                hash: v["players"][0]["hash"].as_str().unwrap_or_default().to_string(),
            });
        }
    }
    Ok(out)
}

fn plan_json(p: &[PlanStep]) -> Value {
    Value::Array(
        p.iter()
            .map(|s| json!([s.dir, s.jump, s.hook, s.fire, (s.aim * 1000.0).round() / 1000.0]))
            .collect(),
    )
}

fn action_json(a: &ddai_brain::Action) -> Value {
    json!([
        a.direction,
        i32::from(a.jump),
        i32::from(a.hook),
        i32::from(a.fire),
        a.target.x,
        a.target.y
    ])
}

fn score_json(s: &DebugScore) -> Value {
    json!({"combos": s.combos, "robust": s.robust, "self_out": s.self_out})
}

fn truth_json(t: &Option<TruthOutcome>) -> Value {
    t.as_ref().map_or(Value::Null, |t| {
        json!({
            "me_out": t.me_out_tick, "enemy_out": t.enemy_out_tick, "touch": t.touch_tick,
            "min_gap": t.min_gap_px, "end_gap_enemy": t.end_gap_enemy_px, "end_dist": t.end_dist_px,
        })
    })
}

fn truth_arr(t: &TruthOutcome) -> Value {
    json!([
        t.me_out_tick,
        t.enemy_out_tick,
        t.touch_tick,
        (t.min_gap_px * 10.0).round() / 10.0,
        (t.end_gap_enemy_px * 10.0).round() / 10.0,
        (t.end_dist_px * 10.0).round() / 10.0
    ])
}

fn tee_json(t: &ddai_planner::diag::TeeSnap) -> Value {
    json!({"x": t.x, "y": t.y, "vx": t.vx, "vy": t.vy, "frozen": t.frozen, "hook": t.hook_state,
           "hooked": t.hooked_player, "jumps": t.jumps_left, "freeze_left": t.freeze_ticks_left})
}

fn record_json(r: &DecisionRecord, inner: &DiagInner) -> Value {
    let h = &r.hybrid;
    let pool: Vec<Value> = h
        .pool
        .iter()
        .map(|c| {
            json!({
                "src": c.src.label(), "kind": Source::kind_name(c.src.kind()), "plan": plan_json(&c.plan), "cheap": c.cheap,
                "scores": c.scores, "self_out": c.self_out,
            })
        })
        .collect();
    json!({
        "idx": r.idx, "tick": r.tick, "me": tee_json(&r.me), "victim": tee_json(&r.victim),
        "hyb_action": action_json(&r.hybrid_action), "tea_action": action_json(&r.teacher_action),
        "hybrid": {
            "chosen_src": h.chosen.map_or("none", Source::label), "chosen_plan": plan_json(&h.chosen_plan),
            "pick": h.pick, "top": h.top, "weights": h.weights, "lambda": h.lambda, "danger": h.danger.reasons(),
            "extended": h.extended, "out_of_time": h.out_of_time, "shielded": h.shielded, "unsafe": h.unsafe_choice,
            "combos": h.combos, "evaluated": h.evaluated.iter().sum::<u32>(), "work_ticks": h.work.total_ticks(),
            "mirror_live_first": h.mirror_first.as_ref().map(|i| json!([i.direction, i.jump, i.hook, i.fire, i.target_x, i.target_y])),
            "react_belief": h.react_belief, "best_score": h.best_score, "robust_value": h.robust_value,
            "pool": pool,
        },
        "opp": {"actual": inner.opp_action_at(r.tick).as_ref().map(action_json), "mirror": r.mirror_action.as_ref().map(action_json), "hold": r.hold_action.as_ref().map(action_json)},
        "teacher": {"plan": r.teacher_plan.as_deref().map(plan_json), "score": r.teacher_score.as_ref().map(score_json)},
        "truth_h": truth_json(&r.truth_hybrid), "truth_t": truth_json(&r.truth_teacher),
        "pool_truth": r.truth_pool.iter().map(truth_arr).collect::<Vec<_>>(),
        "oracle": r.oracle,
        "oracle_mirror": r.oracle_mirror,
        "mirror_inputs": r.mirror_inputs.iter().map(|i| json!([i.direction, i.jump, i.hook, i.fire, i.target_x, i.target_y])).collect::<Vec<_>>(),
        "opp_future": (0..30).map(|k| inner.opp_input_at(r.tick + k).map(|i| json!([i.direction, i.jump, i.hook, i.fire, i.target_x, i.target_y]))).collect::<Vec<_>>(),
    })
}

struct Ctx<'a> {
    cfg: &'a RunConfig,
    arenas: &'a BTreeMap<String, Arena>,
}

/// Replays game `o` with a diagnosing slot 0; returns the report and the diagnosis state.
fn replay(
    ctx: &Ctx<'_>,
    o: &Orig,
    opts: DiagOptions,
) -> Result<(GameReport, std::rc::Rc<std::cell::RefCell<DiagInner>>), String> {
    let cond = ctx
        .cfg
        .condition
        .iter()
        .find(|c| c.name == o.condition)
        .ok_or_else(|| format!("condition {:?} not in the config", o.condition))?;
    let arena = &ctx.arenas[&cond.arena];
    let rules = ctx.cfg.rules_for(cond);
    let slots: Vec<PlayerSpec> = cond.slots();
    let (mut hcfg, clock) = hybrid_config(&slots[0]).map_err(|e| e.to_string())?;
    hcfg.debug_pool = true;
    let proposer =
        builtin_proposer(slots[0].hybrid.as_ref().map_or("none", |h| h.proposer_name())).map_err(|e| e.to_string())?;
    let hybrid = HybridBrain::new(hcfg, clock, proposer)?;
    let (brain, inner) = DiagBrain::new(hybrid, opts);
    let mut players = vec![PlayerSetup {
        brain: Box::new(brain),
        lag: slots[0].lag,
        label: "hybrid-diag".into(),
    }];
    for s in &slots[1..] {
        let b = builtin_brain(s).map_err(|e| e.to_string())?;
        let label = b.name().to_string();
        players.push(PlayerSetup {
            brain: b,
            lag: s.lag,
            label,
        });
    }
    let seed = ctx.cfg.base_seed.wrapping_add(u64::from(o.game));
    let rep = play_game(arena, &rules, seed, layout_of(arena, o.game), players).map_err(|e| e.to_string())?;
    Ok((rep, inner))
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
    let ctx = Ctx {
        cfg: &cfg,
        arenas: &arenas,
    };
    let mut games = read_games(&a.games)?;
    games.retain(|g| match a.select.as_str() {
        "lost" => !(g.result == "W" && g.credited),
        "won" => g.result == "W" && g.credited,
        _ => true,
    });
    if let Some(sub) = &a.only {
        games.retain(|g| g.condition.contains(sub.as_str()));
    }
    if let Some(n) = a.limit {
        // The first `n` per condition.
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
    let file = a.out.join("truth.jsonl");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&file).map_err(|e| e.to_string())?);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = games.len();
    // Chunks keep the output in game order and let it flow to disk while the run goes on.
    for chunk in games.chunks(24) {
        let lines: Vec<Result<Vec<String>, String>> = pool.install(|| {
            chunk
                .par_iter()
                .map(|o| {
                    let lines = truth_game(&ctx, o, a.window)?;
                    let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if n.is_multiple_of(25) {
                        eprintln!("{n}/{total}");
                    }
                    Ok(lines)
                })
                .collect()
        });
        for l in lines {
            for line in l? {
                writeln!(out, "{line}").map_err(|e| e.to_string())?;
            }
        }
        out.flush().map_err(|e| e.to_string())?;
    }
    eprintln!("wrote {}", file.display());
    Ok(())
}

fn truth_game(ctx: &Ctx<'_>, o: &Orig, window: i32) -> Result<Vec<String>, String> {
    let lo = (o.end_tick - window).max(0);
    let opts = DiagOptions {
        window: Some((lo, o.end_tick)),
        mirror: true,
    };
    let (rep, inner) = replay(ctx, o, opts)?;
    inner.borrow_mut().finalize();
    let inner = inner.borrow();
    let decisions: Vec<Value> = inner
        .records
        .iter()
        .filter(|r| r.tick >= lo && r.tick <= o.end_tick)
        .map(|r| record_json(r, &inner))
        .collect();
    Ok(vec![
        json!({
            "condition": o.condition, "game": o.game, "result": o.result, "credited": o.credited, "end_tick": o.end_tick,
            "replay_result": format!("{:?}", rep.result), "replay_end_tick": rep.end_tick,
            "hash_ok": rep.players[0].hash == o.hash, "decisions": decisions,
        })
        .to_string(),
    ])
}
