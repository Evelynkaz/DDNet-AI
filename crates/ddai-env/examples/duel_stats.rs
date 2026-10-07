//! Task 3.14 (E-026): how a duel is played, per player -- the numbers the diagnosis of the live 1vs1 against the competitor's bot was made of.
//!
//! Plays the games of an arena config itself (`play_game_watched`, same seeds and layouts as `ddnet-ai arena run`) and, for slot 0 (us) and
//! slot 1 (the opponent), counts per 30 s (1500 ticks) of play: hammer hits on the other tee, hook episodes on the other tee (ticks the
//! other tee is held by our hook; the share of short ones, <= 4 ticks = a tap), direction changes, and where and how each player's
//! first freeze happened (position, hooked by the other tee, hit by its hammer within 20 ticks, rising).
//!
//! ```text
//! cargo run --release -p ddai-env --example duel_stats -- --config configs/arena/e026-duel-base.toml --threads 3 [--games 100] [--only <substring>] [--jsonl <file>]
//! ```

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use ddai_env::arena::Arena;
use ddai_env::config::{PlayerSpec, RunConfig, builtin_brain};
use ddai_env::game::play_game_watched;
use ddai_env::observe;
use ddai_env::run::{layout_of, load_arenas};
use ddai_env::sim::PlayerSetup;
use ddai_planner::types::WorldEvent;
use rayon::prelude::*;
use serde_json::{Value, json};

struct Args {
    config: PathBuf,
    threads: usize,
    games: Option<u32>,
    only: Option<String>,
    jsonl: Option<PathBuf>,
    /// Record what the hybrid's decisions foresaw in the last 40 ticks before the game was decided (telemetry of slot 0 each decision).
    trace: bool,
    /// Print the last 60 ticks before the end of the first N lost games.
    dump: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        config: PathBuf::new(),
        threads: 3,
        games: None,
        only: None,
        jsonl: None,
        trace: false,
        dump: 0,
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
            "--trace" => a.trace = true,
            "--dump" => a.dump = v()?.parse().map_err(|e| format!("--dump: {e}"))?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.config.as_os_str().is_empty() {
        return Err("usage: duel_stats --config <toml> [--threads N] [--games N] [--only <condition substring>] [--jsonl <file>]".into());
    }
    Ok(a)
}

/// One played game: the condition's name, its JSONL line, what each side did, and the tick it was decided.
type Row = (String, Value, [Side; 2], i32);

/// What one player did in one game.
#[derive(Default, Clone)]
struct Side {
    hammer_hits: u32,
    hammer_fires: u32,
    /// Lengths (ticks) of the episodes in which this player's hook held the other tee.
    holds: Vec<u32>,
    /// How many of those episodes ended while the hook button was still down (the physics let go, not the player).
    forced_ends: u32,
    dir_changes: u32,
    /// Ticks the hook button was held / number of presses (a press = a rising edge of the button).
    hook_ticks: u32,
    hook_presses: u32,
    /// Slot 0 only: the decisions' verdicts of the last 40 ticks before the end -- (tick, chosen plan flagged unsafe, danger reasons, chosen label).
    verdicts: Vec<(i32, bool, String, String)>,
    /// `--dump`: one line per tick for the last 60 ticks before the game was decided (both tees, inputs, hooks, events).
    track: Vec<String>,
    /// First freeze: tick, position, hooked by the other tee, hit by its hammer within 20 ticks, vertical speed the tick before.
    freeze: Option<(i32, f32, f32, bool, bool, f32)>,
}

fn play(
    cfg: &RunConfig,
    arenas: &BTreeMap<String, Arena>,
    cond_i: usize,
    g: u32,
    trace: bool,
    dump: bool,
) -> Result<Row, String> {
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
    let mut sides: [Side; 2] = Default::default();
    let mut last_hit_on: [i32; 2] = [-1000; 2];
    let mut hold_run = [0u32; 2];
    let mut prev_dir = [0i32; 2];
    let mut prev_hook = [0i32; 2];
    let mut prev_vy = [0f32; 2];
    let mut was_out = [false; 2];
    let mut first = true;
    let mut decided = false;
    let rep = play_game_watched(arena, &rules, seed, layout_of(arena, g), players, &mut |sim, tick| {
        if first {
            sim.record_events = true;
            first = false;
        }
        if trace
            && !decided
            && (tick - 1) % 2 == 0
            && let Some(t) = sim.players[0].brain.telemetry()
                && let Ok(v) = serde_json::from_str::<Value>(&t)
                && let Some(last) = v.get("last").filter(|l| !l.is_null())
            {
                let unsafe_ = last.get("unsafe").and_then(Value::as_bool).unwrap_or(false);
                let danger = last.get("danger").map(|d| d.to_string()).unwrap_or_default();
                let chosen = last.get("chosen").and_then(Value::as_str).unwrap_or("").to_string();
                let r = &mut sides[0].verdicts;
                r.push((tick, unsafe_, danger, chosen));
                if r.len() > 20 {
                    r.remove(0);
                }
            }
        let w = sim.pw.inner();
        if dump && !decided {
            let (Some(c0), Some(c1)) = (w.cores.get(0), w.cores.get(1)) else { return true };
            let ev: Vec<String> = sim
                .last_events
                .iter()
                .filter_map(|e| match *e {
                    WorldEvent::HammerHit { from, to } => Some(format!("HIT {from}->{to}")),
                    WorldEvent::HammerFire { from, hits: 0 } => Some(format!("swing {from}")),
                    _ => None,
                })
                .collect();
            let line = format!(
                "t{tick} us ({:.0},{:.0}) v({:.1},{:.1}) in[d{} j{} h{} f{}] hook{}->{} | opp ({:.0},{:.0}) v({:.1},{:.1}) in[d{} j{} h{} f{}] hook{}->{} | dx {:.0} dy {:.0} {}",
                c0.pos.x, c0.pos.y, c0.vel.x, c0.vel.y, c0.input.direction, c0.input.jump, c0.input.hook, c0.input.fire & 1, c0.hook_state, c0.hooked_player(),
                c1.pos.x, c1.pos.y, c1.vel.x, c1.vel.y, c1.input.direction, c1.input.jump, c1.input.hook, c1.input.fire & 1, c1.hook_state, c1.hooked_player(),
                c1.pos.x - c0.pos.x, c1.pos.y - c0.pos.y, ev.join(" ")
            );
            sides[0].track.push(line);
            if sides[0].track.len() > 60 {
                sides[0].track.remove(0);
            }
        }
        for e in &sim.last_events {
            match *e {
                WorldEvent::HammerHit { from, to } if (0..2).contains(&from) && (0..2).contains(&to) => {
                    sides[from as usize].hammer_hits += 1;
                    last_hit_on[to as usize] = tick;
                }
                WorldEvent::HammerFire { from, .. } if (0..2).contains(&from) => sides[from as usize].hammer_fires += 1,
                _ => {}
            }
        }
        for i in 0..2usize {
            let o = 1 - i;
            let Some(c) = w.cores.get(i as u8) else { continue };
            let held = observe::hooked_player(w, i as i32) == o as i32;
            if held {
                hold_run[i] += 1;
            } else if hold_run[i] > 0 {
                sides[i].holds.push(hold_run[i]);
                sides[i].forced_ends += u32::from(c.input.hook != 0);
                hold_run[i] = 0;
            }
            if c.input.direction != prev_dir[i] {
                sides[i].dir_changes += 1;
            }
            prev_dir[i] = c.input.direction;
            if c.input.hook != 0 {
                sides[i].hook_ticks += 1;
                if prev_hook[i] == 0 {
                    sides[i].hook_presses += 1;
                }
            }
            prev_hook[i] = c.input.hook;
            let out = observe::is_out(w, i as i32);
            decided |= out;
            if out && !was_out[i] && sides[i].freeze.is_none() {
                let hooked_by_other = observe::hooked_player(w, o as i32) == i as i32;
                let hit = tick - last_hit_on[i] <= 20;
                sides[i].freeze = Some((tick, c.pos.x, c.pos.y, hooked_by_other, hit, prev_vy[i]));
            }
            was_out[i] = out;
            prev_vy[i] = c.vel.y;
        }
        true
    })
    .map_err(|e| e.to_string())?;
    for i in 0..2 {
        if hold_run[i] > 0 {
            sides[i].holds.push(hold_run[i]);
        }
    }
    let line = json!({
        "condition": cond.name, "game": g, "result": format!("{:?}", rep.result), "credited": rep.credited, "held": rep.held, "held_block": rep.held_block, "end_tick": rep.end_tick,
        "spawns": rep.spawns,
        "sides": sides.iter().map(|s| json!({
            "hammer_hits": s.hammer_hits, "hammer_fires": s.hammer_fires, "holds": s.holds, "dir_changes": s.dir_changes,
            "hook_ticks": s.hook_ticks, "hook_presses": s.hook_presses, "forced_ends": s.forced_ends, "freeze": s.freeze,
            "verdicts": s.verdicts,
        })).collect::<Vec<_>>(),
    });
    Ok((cond.name.clone(), line, sides, rep.end_tick))
}

fn median(v: &mut [u32]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_unstable();
    v[v.len() / 2] as f64
}

/// What the hybrid's decisions of the lost games (we froze first) said in the last 40 ticks: was the plan it chose flagged unsafe (some modelled reply freezes
/// us after it), and how long before the freeze was it last so? Printed with `--trace`.
fn foresight(rows: &[Row]) {
    let lost: Vec<_> = rows.iter().filter(|r| r.1["result"] == "L").collect();
    let mut flagged = 0;
    let mut gaps: Vec<i32> = Vec::new();
    let mut labels: BTreeMap<String, u32> = BTreeMap::new();
    for r in &lost {
        let end = r.3;
        let v = &r.2[0].verdicts;
        // The lead time of the foresight: the unbroken run of "unsafe" verdicts that ends at the last decision, from its first tick to the freeze.
        let run: Vec<_> = v.iter().rev().take_while(|x| x.1).collect();
        if let Some(first) = run.last() {
            flagged += 1;
            gaps.push(end - first.0);
        } else {
            gaps.push(0);
        }
        for x in v.iter().rev().take(3) {
            *labels.entry(x.3.clone()).or_default() += 1;
        }
    }
    gaps.sort_unstable();
    let q = |p: f64| gaps.get(((gaps.len() as f64 - 1.0) * p) as usize).copied();
    println!(
        "  lost games {}: the plan chosen by the last decision was flagged unsafe in {flagged} (the last decision's plan); how long before the freeze the unbroken run of unsafe verdicts began (ticks, 0 = none): p25 {:?} p50 {:?} p75 {:?}, >= 10 ticks: {}, >= 20 ticks: {}; the last three decisions chose: {:?}",
        lost.len(),
        q(0.25),
        q(0.5),
        q(0.75),
        gaps.iter().filter(|&&g| g >= 10).count(),
        gaps.iter().filter(|&&g| g >= 20).count(),
        labels
    );
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
    for (ci, cond) in cfg.condition.iter().enumerate() {
        if a.only.as_deref().is_some_and(|s| !cond.name.contains(s)) {
            continue;
        }
        let n = a.games.unwrap_or_else(|| cfg.games_for(cond));
        let res: Vec<Result<Row, String>> = pool.install(|| {
            (0..n)
                .into_par_iter()
                .map(|g| play(&cfg, &arenas, ci, g, a.trace, a.dump > 0))
                .collect()
        });
        let mut rows = Vec::new();
        for r in res {
            rows.push(r?);
        }
        if let Some(f) = jsonl.as_mut() {
            for (_, line, _, _) in &rows {
                writeln!(f, "{line}").map_err(|e| e.to_string())?;
            }
        }
        let ticks: f64 = rows.iter().map(|r| f64::from(r.3)).sum();
        let per30 = |x: f64| x / ticks * 1500.0;
        let (mut w, mut l, mut d, mut t) = (0, 0, 0, 0);
        for (_, line, _, _) in &rows {
            match line["result"].as_str().unwrap_or("") {
                "W" => w += 1,
                "L" => l += 1,
                "D" => d += 1,
                _ => t += 1,
            }
        }
        let cred = rows
            .iter()
            .filter(|r| r.1["result"] == "W" && r.1["credited"] == true)
            .count();
        println!(
            "== {} : {n} games, W:L:D:T = {w}:{l}:{d}:{t}, credited wins {cred} ({:.1}%), mean game {:.0} ticks",
            cond.name,
            100.0 * cred as f64 / f64::from(n),
            ticks / f64::from(n)
        );
        if a.trace {
            foresight(&rows);
        }
        let mut shown = 0;
        for r in &rows {
            if shown < a.dump && r.1["result"] == "L" {
                shown += 1;
                println!(
                    "--- lost game {} (we froze at tick {}; \"us\" = slot 0)",
                    r.1["game"], r.3
                );
                for l in &r.2[0].track {
                    println!("{l}");
                }
            }
        }
        for (i, who) in ["us (slot 0)", "opp (slot 1)"].iter().enumerate() {
            let hits: u32 = rows.iter().map(|r| r.2[i].hammer_hits).sum();
            let fires: u32 = rows.iter().map(|r| r.2[i].hammer_fires).sum();
            let dirs: u32 = rows.iter().map(|r| r.2[i].dir_changes).sum();
            let presses: u32 = rows.iter().map(|r| r.2[i].hook_presses).sum();
            let mut holds: Vec<u32> = rows.iter().flat_map(|r| r.2[i].holds.clone()).collect();
            let taps = holds.iter().filter(|&&h| h <= 4).count();
            let forced: u32 = rows.iter().map(|r| r.2[i].forced_ends).sum();
            let nh = holds.len();
            let hold_ticks: u32 = holds.iter().sum();
            let med = median(&mut holds);
            let freezes: Vec<_> = rows.iter().filter_map(|r| r.2[i].freeze).collect();
            let hooked = freezes.iter().filter(|f| f.3).count();
            let hit = freezes.iter().filter(|f| f.4).count();
            let rising = freezes.iter().filter(|f| f.5 < -2.0).count();
            let mut ys: Vec<f32> = freezes.iter().map(|f| f.2).collect();
            ys.sort_by(f32::total_cmp);
            let ymed = ys.get(ys.len() / 2).copied().unwrap_or(f32::NAN);
            println!(
                "  {who:13} hammer hits/30s {:.2} (fires {:.2}) | hook on other: eps/30s {:.2}, held ticks/30s {:.1}, median hold {med:.0}, <=4 ticks {:.0}% | hook presses/30s {:.2} | holds ended with the button still down {:.0}% | dir changes/30s {:.1} | first freeze: {} ({} hooked-by-other, {} hit<=20t, {} rising), median y {ymed:.0}",
                per30(f64::from(hits)),
                per30(f64::from(fires)),
                per30(nh as f64),
                per30(f64::from(hold_ticks)),
                100.0 * taps as f64 / nh.max(1) as f64,
                per30(f64::from(presses)),
                100.0 * f64::from(forced) / nh.max(1) as f64,
                per30(f64::from(dirs)),
                freezes.len(),
                hooked,
                hit,
                rising,
            );
        }
    }
    Ok(())
}
