//! Speed of the hybrid decision (task 3.5 criterion 5, D-045): wall time next to work counters
//! (physics ticks per phase) on Copy Love Box with 2, 4 and 6 tees, over worker counts 1..4 and
//! budgets, plus the machine's stall baseline. Heavy, so `#[ignore]`; needs the map file:
//!
//! ```text
//! cargo test -p ddai-planner --release --test hybrid_speed -- --ignored --nocapture speed_report
//! ```
//!
//! Wall time on this VM includes host stalls of about 10 ms roughly ten times a second (D-045), so
//! a p99 of wall time is a property of the machine as much as of the code; the work counters
//! (physics ticks simulated) are not affected by stalls and are what the 5 ms target is proven with.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use ddai_brain::{Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_physics::map::MapData;
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::World;
use ddai_planner::brains::{ClockKind, ScriptedBrain, input_from_action};
use ddai_planner::hybrid::{
    HybridBrain, HybridConfig, HybridMode, NoProposer, Proposer, ScriptedProposer, WORK_US_PER_TEE_TICK,
};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::{PlayerInput, empty_input};
use ddai_planner::vmath::Vec2;

fn clb() -> Option<Arc<MapData>> {
    let dir = PathBuf::from(std::env::var("HOME").ok()?).join("aiddnet/data/maps/copy-love-box");
    let f = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "map"))?;
    let m = ddai_map::load_map(&std::fs::read(f).ok()?).ok()?;
    Some(Arc::new(m.data))
}

fn char_obs(w: &World<f32>, id: i32) -> CharacterObservation {
    let core = w.cores.get(id as u8).expect("tee");
    let ch = w.characters[id as usize].as_ref().expect("character");
    let mut c = CharacterObservation::at_rest(id);
    c.pos = core.pos;
    c.vel = core.vel;
    c.hook_state = core.hook_state;
    c.hooked_player = core.hooked_player();
    c.is_frozen = ch.freeze_time > 0;
    c.freeze_ticks_remaining = ch.freeze_time;
    c.direction = core.direction;
    c
}

fn nearest(w: &World<f32>, me: i32, ids: &[i32]) -> i32 {
    let p = w.cores.get(me as u8).map(|c| c.pos);
    let mut best = (f32::INFINITY, ids[0]);
    for &i in ids.iter().filter(|&&i| i != me) {
        if let (Some(p), Some(o)) = (p, w.cores.get(i as u8)) {
            let d = (o.pos.x - p.x).powi(2) + (o.pos.y - p.y).powi(2);
            if d < best.0 {
                best = (d, i);
            }
        }
    }
    best.1
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[(((v.len() - 1) as f64) * p / 100.0).round() as usize]
}

/// The machine's stall baseline: a dense clock-reading loop with no work (as in the 3.2 review).
fn stall_baseline(ms: u64) -> (f64, f64, f64, f64) {
    let start = Instant::now();
    let mut last = start;
    let (mut n, mut over_half, mut over_two, mut max) = (0u64, 0u64, 0u64, 0.0f64);
    while start.elapsed().as_millis() < u128::from(ms) {
        let now = Instant::now();
        let d = now.duration_since(last).as_secs_f64() * 1000.0;
        last = now;
        n += 1;
        over_half += u64::from(d > 0.5);
        over_two += u64::from(d > 2.0);
        max = max.max(d);
    }
    (
        100.0 * over_half as f64 / n as f64,
        100.0 * over_two as f64 / n as f64,
        max,
        n as f64,
    )
}

struct Row {
    wall: Vec<f64>,
    ticks: Vec<f64>,
    tee_ticks: Vec<f64>,
    cands: Vec<f64>,
    fly_ms: Vec<f64>,
    search_ms: Vec<f64>,
    overshoot_ms: Vec<f64>,
    shield_ms: Vec<f64>,
    mirror_ms: Vec<f64>,
    rollout_ms: Vec<f64>,
    stage1_ticks: Vec<f64>,
    stage2_ticks: Vec<f64>,
    shield_ticks: Vec<f64>,
    ext_ticks: Vec<f64>,
    proposal_ticks: Vec<f64>,
    extended: u64,
    shield_incomplete: u64,
    decisions: u64,
    /// FNV-1a over every decision's action, work counters and candidate counts: equal digests mean
    /// bit-identical decisions (the speed-ups of task 3.6 must not change it on the work clock).
    digest: u64,
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// Plays scenes of `tees` tees (slot 0 = the hybrid, the others scripted attackers) on the left
/// wayblock hall and records `decisions` decisions of slot 0.
fn run(map: &Arc<MapData>, cfg: &HybridConfig, proposer: Box<dyn Proposer>, tees: usize, decisions: usize) -> Row {
    run_scenes(map, cfg, proposer, tees, decisions, false)
}

/// [`run`]; with `frozen_victim` the tee next to slot 0 starts frozen (150 ticks left) and the scenes go on while slot 0 is free (task 3.10: the
/// decisions of the finishing phase, where the finishing terms and families are active).
fn run_scenes(
    map: &Arc<MapData>,
    cfg: &HybridConfig,
    proposer: Box<dyn Proposer>,
    tees: usize,
    decisions: usize,
    frozen_victim: bool,
) -> Row {
    let mut row = Row {
        wall: vec![],
        ticks: vec![],
        tee_ticks: vec![],
        cands: vec![],
        fly_ms: vec![],
        search_ms: vec![],
        overshoot_ms: vec![],
        shield_ms: vec![],
        mirror_ms: vec![],
        rollout_ms: vec![],
        stage1_ticks: vec![],
        stage2_ticks: vec![],
        shield_ticks: vec![],
        ext_ticks: vec![],
        proposal_ticks: vec![],
        extended: 0,
        shield_incomplete: 0,
        decisions: 0,
        digest: 0xcbf2_9ce4_8422_2325,
    };
    let ids: Vec<i32> = (0..tees as i32).collect();
    let mut hybrid = HybridBrain::new(cfg.clone(), ClockKind::Wall, proposer).expect("config");
    let mut scene = 0u64;
    while row.decisions < decisions as u64 {
        scene += 1;
        let mut pw = PhysicsWorld::new(map.clone(), scene);
        for i in 0..tees {
            pw.add_tee(
                i as i32,
                Vec2 {
                    x: (103.5 - 2.4 * i as f64 - if i > 0 { 1.5 } else { 0.0 }) * 32.0,
                    y: 84.5 * 32.0,
                },
            );
        }
        if frozen_victim && tees > 1 {
            let mut st = pw.get_tee(1).expect("victim");
            st.frozen = true;
            st.freeze_ticks_left = 150;
            pw.apply_tee_state(1, &st);
        }
        hybrid.reset(&ResetContext {
            map: map.clone(),
            self_id: 0,
            seed: scene,
        });
        let mut scripted: Vec<ScriptedBrain> = (1..tees).map(|_| ScriptedBrain::new()).collect();
        for (k, b) in scripted.iter_mut().enumerate() {
            b.reset(&ResetContext {
                map: map.clone(),
                self_id: (k + 1) as i32,
                seed: scene * 10 + k as u64,
            });
        }
        let mut last: Vec<PlayerInput> = ids.iter().map(|_| empty_input()).collect();
        for _ in 0..60 {
            if row.decisions >= decisions as u64 {
                break;
            }
            let world = pw.inner().clone();
            let out = |id: i32| pw.get_tee(id).is_none_or(|t| t.frozen || !t.alive);
            if out(0) || (!frozen_victim && (1..tees as i32).all(out)) {
                break;
            }
            for (slot, &id) in ids.iter().enumerate() {
                if world.cores.get(id as u8).is_none() {
                    continue;
                }
                let target = if slot == 0 { nearest(&world, 0, &ids) } else { 0 };
                let obs = Observation {
                    map: map.clone(),
                    tick: world.tick,
                    self_state: char_obs(&world, id),
                    others: ids
                        .iter()
                        .filter(|&&i| i != id && world.cores.get(i as u8).is_some())
                        .map(|&i| char_obs(&world, i))
                        .collect(),
                    target_id: Some(target),
                    tuning: TuningParams::default(),
                };
                let view = WorldView {
                    world: &world,
                    self_id: id,
                    lag_ticks: 0,
                    in_flight: &[],
                };
                let action = if slot == 0 {
                    let t0 = Instant::now();
                    let a = hybrid.decide_in(&obs, Some(&view));
                    let wall = t0.elapsed().as_secs_f64() * 1000.0;
                    let t = hybrid.last_decision().expect("telemetry");
                    row.wall.push(wall);
                    row.ticks.push(t.work.total_ticks() as f64);
                    // `units`: the v2 gate's projections, priced in tee-ticks (task 3.9; zero with the switches off).
                    row.tee_ticks
                        .push((t.work.total_ticks() * u64::from(t.sim_tees.max(1)) + t.work.units) as f64);
                    row.cands.push(f64::from(t.evaluated.iter().sum::<u32>()));
                    row.fly_ms.push(t.proposal_ms);
                    row.search_ms.push(t.search_ms);
                    row.overshoot_ms.push(t.search_ms - t.budget_ms);
                    row.shield_ms.push(t.shield_ms);
                    row.mirror_ms.push(t.mirror_ms);
                    row.rollout_ms.push(t.rollout_ms);
                    row.stage1_ticks.push(t.work.stage1 as f64);
                    row.stage2_ticks.push(t.work.stage2 as f64);
                    row.shield_ticks.push(t.work.shield as f64);
                    row.ext_ticks.push(t.work.extension as f64);
                    row.proposal_ticks.push(t.work.proposal as f64);
                    row.extended += u64::from(t.extended);
                    row.shield_incomplete += u64::from(t.shield_incomplete);
                    row.decisions += 1;
                    fnv(
                        &mut row.digest,
                        format!(
                            "{a:?}|{}|{}|{:?}|{:?}",
                            t.work.total_ticks(),
                            t.sim_tees,
                            t.evaluated,
                            t.work
                        )
                        .as_bytes(),
                    );
                    a
                } else {
                    scripted[slot - 1].decide_in(&obs, Some(&view))
                };
                last[slot] = input_from_action(&action, &last[slot]);
            }
            for _ in 0..2 {
                for (slot, &id) in ids.iter().enumerate() {
                    pw.set_input(id, last[slot]);
                }
                pw.step();
            }
        }
    }
    row
}

fn deadline(budget: f64, workers: usize, adaptive: bool, proposals: usize) -> HybridConfig {
    let mut adaptive_cfg = HybridConfig::default().adaptive;
    adaptive_cfg.enabled = adaptive;
    HybridConfig {
        mode: HybridMode::Deadline { budget_ms: budget },
        workers,
        adaptive: adaptive_cfg,
        proposals,
        // The cap is for the default 4 ms decision; a bigger budget is asked for on purpose.
        decision_cap_ms: if budget > 5.0 { None } else { Some(5.0) },
        ..HybridConfig::default()
    }
}

/// Median wall microseconds of one physics tick on this machine right now (the work-counter
/// conversion factor), measured on the given tee count.
fn us_per_tick(map: &Arc<MapData>, tees: usize) -> f64 {
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    for i in 0..tees {
        pw.add_tee(
            i as i32,
            Vec2 {
                x: (103.5 - 2.4 * i as f64) * 32.0,
                y: 84.5 * 32.0,
            },
        );
    }
    let saved = pw.save_state();
    let mut samples = Vec::new();
    for _ in 0..400 {
        let t = Instant::now();
        for _ in 0..27 {
            pw.set_input(0, empty_input());
            pw.step();
        }
        samples.push(t.elapsed().as_secs_f64() * 1e6 / 27.0);
        pw.restore_state(&saved);
    }
    pct(&mut samples, 50.0)
}

fn line(label: &str, r: &mut Row, us: f64) -> String {
    let w = [pct(&mut r.wall, 50.0), pct(&mut r.wall, 90.0), pct(&mut r.wall, 99.0)];
    let k = [
        pct(&mut r.ticks, 50.0),
        pct(&mut r.ticks, 90.0),
        pct(&mut r.ticks, 99.0),
    ];
    let c = [
        pct(&mut r.cands, 50.0),
        pct(&mut r.cands, 90.0),
        pct(&mut r.cands, 99.0),
    ];
    let tt = [
        pct(&mut r.tee_ticks, 50.0) * WORK_US_PER_TEE_TICK / 1000.0,
        pct(&mut r.tee_ticks, 90.0) * WORK_US_PER_TEE_TICK / 1000.0,
        pct(&mut r.tee_ticks, 99.0) * WORK_US_PER_TEE_TICK / 1000.0,
    ];
    format!(
        "| {label} | {:.2} / {:.2} / {:.2} | {:.0} / {:.0} / {:.0} | {:.2} / {:.2} / {:.2} | {:.2} / {:.2} / {:.2} | {:.0} / {:.0} / {:.0} | {:.1}% |",
        w[0],
        w[1],
        w[2],
        k[0],
        k[1],
        k[2],
        k[0] * us / 1000.0,
        k[1] * us / 1000.0,
        k[2] * us / 1000.0,
        tt[0],
        tt[1],
        tt[2],
        c[0],
        c[1],
        c[2],
        100.0 * r.extended as f64 / r.decisions.max(1) as f64
    )
}

const HEADER: &str = "| Condition | wall ms p50/p90/p99 | work ticks p50/p90/p99 | physics-only ms (ticks x us/tick) p50/p90/p99 | work ms (tee-ticks x the work-clock calibration, WORK_US_PER_TEE_TICK) p50/p90/p99 | candidates p50/p90/p99 | extended |\n|---|---|---|---|---|---|---|";

#[test]
#[ignore = "heavy; needs the Copy Love Box map"]
fn speed_report() {
    let Some(map) = clb() else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    let n: usize = std::env::var("DDAI_SPEED_DECISIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let (h5, h2, mx, reads) = stall_baseline(2000);
    println!("\n### Stall baseline (2 s dense clock loop, no work)\n");
    println!("intervals > 0.5 ms: {h5:.4}%, > 2 ms: {h2:.4}%, max {mx:.1} ms ({reads:.0} reads)\n");
    println!(
        "(the {:.1} intervals > 2 ms of a 2 s loop mean a 4-5 ms decision meets a stall in about {:.1}% of the cases: the wall p99 below is the machine's stall tail, the work columns are not)\n",
        reads * h2 / 100.0,
        reads * h2 / 100.0 / 2.0 * 0.0045 * 100.0
    );

    println!("### Per-phase breakdown, 4 tees, 4 ms budget, 1 worker, no proposer\n");
    let us4 = us_per_tick(&map, 4);
    let mut r = run(&map, &deadline(4.0, 1, false, 0), Box::new(NoProposer), 4, n);
    println!("us/tick (4 tees) = {us4:.2}");
    println!("| phase | p50 | p90 | p99 |\n|---|---|---|---|");
    for (name, v) in [
        ("stage 1 (ticks)", &mut r.stage1_ticks),
        ("stage 2 (ticks)", &mut r.stage2_ticks),
        ("shield (ticks)", &mut r.shield_ticks),
        ("extension (ticks)", &mut r.ext_ticks),
    ] {
        println!(
            "| {name} | {:.0} | {:.0} | {:.0} |",
            pct(v, 50.0),
            pct(v, 90.0),
            pct(v, 99.0)
        );
    }
    for (name, v) in [
        ("search wall ms", &mut r.search_ms),
        ("  of which scoring rollouts ms", &mut r.rollout_ms),
        ("shield wall ms", &mut r.shield_ms),
        ("opponent model (mirror) wall ms", &mut r.mirror_ms),
    ] {
        println!(
            "| {name} | {:.2} | {:.2} | {:.2} |",
            pct(v, 50.0),
            pct(v, 90.0),
            pct(v, 99.0)
        );
    }

    println!("\n### Decision cost by tee count (1 worker)\n");
    println!("{HEADER}");
    for tees in [2usize, 4, 6] {
        let us = us_per_tick(&map, tees);
        for (label, cfg) in [
            (format!("{tees} tees, 4 ms"), deadline(4.0, 1, false, 0)),
            (format!("{tees} tees, 4 ms + adaptive 15 ms"), deadline(4.0, 1, true, 0)),
            (format!("{tees} tees, 15 ms fixed budget"), deadline(15.0, 1, false, 0)),
        ] {
            let mut r = run(&map, &cfg, Box::new(NoProposer), tees, n);
            println!("{}", line(&label, &mut r, us));
        }
    }

    println!("\n### Work clock (deterministic, load-independent): decision work by tee count and shield reserve\n");
    println!(
        "Budget 4 ms of work = {:.0} tee-ticks, no proposer; work ms = tee-ticks x {WORK_US_PER_TEE_TICK} us.\n",
        4000.0 / WORK_US_PER_TEE_TICK
    );
    println!(
        "| tees | decision cap ms | shield reserve ms/tee | adaptive | work ms p50 / p90 / p99 / max | candidates p50 | shield incomplete | search extended |\n|---|---|---|---|---|---|---|---|"
    );
    for tees in [2usize, 4, 6] {
        for (cap, reserve, adaptive, timeout_danger) in [
            (None, 0.5, false, false),
            (Some(5.0), 0.5, false, false),
            (Some(5.0), 0.25, false, false),
            (Some(5.0), 0.15, false, false),
            // The default: the extension serves a confirmed danger only.
            (Some(5.0), 0.25, true, false),
            // ... and also a shield check that timed out (task 3.5b).
            (Some(5.0), 0.25, true, true),
        ] {
            let mut cfg = deadline(4.0, 1, adaptive, 0);
            cfg.work_clock_us_per_tick = Some(WORK_US_PER_TEE_TICK);
            cfg.decision_cap_ms = cap;
            cfg.shield_reserve_ms_per_tee = reserve;
            cfg.shield_timeout_danger = timeout_danger;
            let mut r = run(&map, &cfg, Box::new(NoProposer), tees, n * 2);
            let ms = |v: &mut Vec<f64>, p: f64| pct(v, p) * WORK_US_PER_TEE_TICK / 1000.0;
            println!(
                "| {tees} | {} | {reserve} | {} | {:.2} / {:.2} / {:.2} / {:.2} | {:.0} | {:.1}% | {:.1}% |",
                cap.map_or("none".to_string(), |c| format!("{c}")),
                match (adaptive, timeout_danger) {
                    (false, _) => "off",
                    (true, false) => "on",
                    (true, true) => "on + timeout danger",
                },
                ms(&mut r.tee_ticks, 50.0),
                ms(&mut r.tee_ticks, 90.0),
                ms(&mut r.tee_ticks, 99.0),
                ms(&mut r.tee_ticks, 100.0),
                pct(&mut r.cands, 50.0),
                100.0 * r.shield_incomplete as f64 / r.decisions.max(1) as f64,
                100.0 * r.extended as f64 / r.decisions.max(1) as f64
            );
        }
    }

    println!("\n### Search overshoot over the 4 ms budget (search wall - budget, adaptive off)\n");
    println!("| tees | p50 ms | p90 ms | p99 ms | share over 0.3 ms |\n|---|---|---|---|---|");
    for tees in [2usize, 4, 6] {
        let mut r = run(&map, &deadline(4.0, 1, false, 0), Box::new(NoProposer), tees, n);
        let over = r.overshoot_ms.iter().filter(|&&o| o > 0.3).count() as f64 / r.overshoot_ms.len().max(1) as f64;
        println!(
            "| {tees} | {:.3} | {:.3} | {:.3} | {:.1}% |",
            pct(&mut r.overshoot_ms, 50.0),
            pct(&mut r.overshoot_ms, 90.0),
            pct(&mut r.overshoot_ms, 99.0),
            100.0 * over
        );
    }

    println!("\n### Worker scaling (4 ms budget, 4 tees)\n");
    println!("{HEADER}");
    for w in 1..=4usize {
        let mut r = run(&map, &deadline(4.0, w, false, 0), Box::new(NoProposer), 4, n);
        println!("{}", line(&format!("{w} workers"), &mut r, us4));
    }
    println!("\n### Worker scaling, fixed work (1 decision = the whole pool), 4 tees\n");
    println!("{HEADER}");
    for w in 1..=4usize {
        let mut cfg = HybridConfig::fixed();
        cfg.proposals = 0;
        cfg.workers = w;
        let mut r = run(&map, &cfg, Box::new(NoProposer), 4, n / 2);
        println!("{}", line(&format!("{w} workers (fixed work)"), &mut r, us4));
    }

    println!("\n### With the scripted proposer (K = 3), 4 tees, 4 ms\n");
    println!("{HEADER}");
    let mut r = run(
        &map,
        &deadline(4.0, 1, false, 3),
        Box::new(ScriptedProposer::new()),
        4,
        n,
    );
    println!("{}", line("scripted proposer K=3", &mut r, us4));
}

/// Task 4.13: `--search-threads` N = 1..4 on the wall clock, the live configuration (`HybridConfig::default()`, 4 ms search under a
/// 5 ms cap, opponent model on; `DDAI_THR_FINISH=1` adds the duel preset's `with_finish`), 2 tees close together (a duel). Rounds are
/// interleaved (1 2 3 4 1 2 3 4 ...) so that drift in the machine's load hits every N alike. Prints, per N, over all rounds: candidates
/// per decision p50 / p90, decision wall ms p50 / p90 / p99 / max, the share of decisions longer than 5 ms, and the load average before
/// and after. Needs a quiet machine and the lead's go-ahead.
///
/// ```text
/// DDAI_THR_ROUNDS=6 DDAI_SPEED_DECISIONS=300 cargo test -p ddai-planner --release --test hybrid_speed -- --ignored --nocapture threads_report
/// ```
#[test]
#[ignore = "heavy; needs the Copy Love Box map and a quiet machine"]
fn threads_report() {
    let Some(map) = clb() else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    let n: usize = std::env::var("DDAI_SPEED_DECISIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let rounds: usize = std::env::var("DDAI_THR_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let finish = std::env::var("DDAI_THR_FINISH").is_ok_and(|v| v == "1");
    let tees: usize = std::env::var("DDAI_THR_TEES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let load = || {
        std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|s| s.split_whitespace().next().map(str::to_owned))
            .unwrap_or_default()
    };
    let loads = [load()];
    let counts = [1usize, 2, 3, 4];
    let mut acc: Vec<(Vec<f64>, Vec<f64>)> = counts.iter().map(|_| (vec![], vec![])).collect();
    for _ in 0..rounds {
        for (k, &w) in counts.iter().enumerate() {
            let base = HybridConfig {
                workers: w,
                ..HybridConfig::default()
            };
            let cfg = if finish { base.with_finish() } else { base };
            let r = run(&map, &cfg, Box::new(NoProposer), tees, n);
            acc[k].0.extend(r.cands);
            acc[k].1.extend(r.wall);
        }
    }
    println!(
        "\nthreads_report: {tees} tees, finish {finish}, {rounds} rounds x {n} decisions, load {} -> {}\n",
        loads[0],
        load()
    );
    println!(
        "| threads | candidates p50 / p90 | wall ms p50 / p90 / p99 / max | decisions > 5 ms |\n|---|---|---|---|"
    );
    for (k, &w) in counts.iter().enumerate() {
        let (c, wl) = &mut acc[k];
        let over = wl.iter().filter(|&&x| x > 5.0).count() as f64 / wl.len().max(1) as f64;
        println!(
            "| {w} | {:.0} / {:.0} | {:.2} / {:.2} / {:.2} / {:.2} | {:.1}% |",
            pct(c, 50.0),
            pct(c, 90.0),
            pct(wl, 50.0),
            pct(wl, 90.0),
            pct(wl, 99.0),
            pct(wl, 100.0),
            100.0 * over
        );
    }
}

/// Task 4.13: where the wall time of a live decision goes (the live configuration, one thread): per phase p50 / p90 / p99 over
/// `DDAI_SPEED_DECISIONS` decisions of 2 tees (`DDAI_THR_TEES`), `DDAI_THR_FINISH=1` for the duel preset.
///
/// ```text
/// cargo test -p ddai-planner --release --test hybrid_speed -- --ignored --nocapture phases_report
/// ```
#[test]
#[ignore = "heavy; needs the Copy Love Box map"]
fn phases_report() {
    let Some(map) = clb() else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    let n: usize = std::env::var("DDAI_SPEED_DECISIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let tees: usize = std::env::var("DDAI_THR_TEES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let finish = std::env::var("DDAI_THR_FINISH").is_ok_and(|v| v == "1");
    let base = HybridConfig::default();
    let cfg = if finish { base.with_finish() } else { base };
    let mut r = run(&map, &cfg, Box::new(NoProposer), tees, n);
    println!("\nphases_report: {tees} tees, finish {finish}, {n} decisions\n");
    println!("| phase | p50 | p90 | p99 |\n|---|---|---|---|");
    for (name, v) in [
        ("decision wall ms", &mut r.wall),
        ("opponent model (mirror) ms", &mut r.mirror_ms),
        ("search ms", &mut r.search_ms),
        ("  of which scoring rollouts ms", &mut r.rollout_ms),
        ("shield ms", &mut r.shield_ms),
        ("candidates", &mut r.cands),
    ] {
        println!(
            "| {name} | {:.2} | {:.2} | {:.2} |",
            pct(v, 50.0),
            pct(v, 90.0),
            pct(v, 99.0)
        );
    }
}

/// The work clock's calibration (task 3.6, D-045 amended): real microseconds per tee-tick of a whole
/// decision (rollouts, scoring, shield, search bookkeeping) at the work-clock budget of 4 ms, on 2 and 4
/// tees, plus the decision digest that proves the decisions did not change. `DDAI_CAL_REPS` repetitions
/// (default 5); the minimum over repetitions is the least noisy estimate on a shared VM.
///
/// ```text
/// cargo test -p ddai-planner --release --test hybrid_speed -- --ignored --nocapture calibration_report
/// ```
#[test]
#[ignore = "heavy; needs the Copy Love Box map"]
fn calibration_report() {
    let Some(map) = clb() else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    let n: usize = std::env::var("DDAI_SPEED_DECISIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600);
    let reps: usize = std::env::var("DDAI_CAL_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    // The work-clock rate the decisions are made at (default: the old 2.2 us, so an old and a new build
    // compare on the same decisions; `DDAI_CAL_US=1.25` measures the cost at the calibrated rate).
    let cal_us: f64 = std::env::var("DDAI_CAL_US")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.2);
    println!("work clock {cal_us} us per tee-tick");
    println!(
        "\n| tees | us per tee-tick (min / median over {reps} runs) | per-decision us per tee-tick p50 / p99 | decision wall ms p50 / p99 | work ms (tee-ticks x rate) p50 / p99 | candidates p50 | tee-ticks p50 | digest |\n|---|---|---|---|---|---|---|---|"
    );
    for tees in [2usize, 4] {
        let mut cfg = deadline(4.0, 1, false, 0);
        cfg.work_clock_us_per_tick = Some(cal_us);
        let mut ratios = Vec::new();
        let mut last = None;
        for _ in 0..reps {
            let mut r = run(&map, &cfg, Box::new(NoProposer), tees, n);
            let wall: f64 = r.wall.iter().sum();
            let tt: f64 = r.tee_ticks.iter().sum();
            ratios.push(wall * 1000.0 / tt);
            if let Some((_, d)) = &last {
                assert_eq!(*d, r.digest, "decisions on the work clock must be reproducible");
            }
            last = Some((
                r.wall
                    .iter()
                    .zip(&r.tee_ticks)
                    .map(|(w, t)| w * 1000.0 / t.max(1.0))
                    .collect::<Vec<f64>>(),
                r.digest,
            ));
            if ratios.len() == reps {
                let mut per = last.as_ref().unwrap().0.clone();
                let mut rr = ratios.clone();
                println!(
                    "| {tees} | {:.3} / {:.3} | {:.3} / {:.3} | {:.2} / {:.2} | {:.2} / {:.2} | {:.0} | {:.0} | {:016x} |",
                    pct(&mut rr.clone(), 0.0),
                    pct(&mut rr, 50.0),
                    pct(&mut per, 50.0),
                    pct(&mut per, 99.0),
                    pct(&mut r.wall, 50.0),
                    pct(&mut r.wall, 99.0),
                    pct(&mut r.tee_ticks, 50.0) * cal_us / 1000.0,
                    pct(&mut r.tee_ticks, 99.0) * cal_us / 1000.0,
                    pct(&mut r.cands, 50.0),
                    pct(&mut r.tee_ticks, 50.0),
                    r.digest
                );
            }
        }
    }
}

/// A proposer that proposes nothing and costs `units` tee-ticks on the work clock: the shape of the fly (an S fly is 500).
struct Costly {
    units: u64,
}

impl Proposer for Costly {
    fn name(&self) -> &str {
        "costly"
    }
    fn propose(
        &mut self,
        _ctx: &ddai_planner::hybrid::ProposeCtx<'_>,
        _out: &mut Vec<Vec<ddai_planner::planner::PlanStep>>,
    ) {
    }
    fn work_units(&self) -> u64 {
        self.units
    }
}

/// Task 3.7b (D-042): the work of a whole decision with the opponent model on, p99 by tee count, with no proposer and with a
/// proposer the cost of the S fly (500 tee-ticks) inside the cap, the 15 ms danger extension as shipped. The bound is the spec's:
/// p99 <= 5 ms of work at 2 tees (work = tee-ticks x 1.25 us, the proposer's units included).
///
/// ```text
/// cargo test -p ddai-planner --release --test hybrid_speed -- --ignored --nocapture mirror_work_report
/// ```
#[test]
#[ignore = "heavy; needs the Copy Love Box map"]
fn mirror_work_report() {
    let Some(map) = clb() else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    let n: usize = std::env::var("DDAI_SPEED_DECISIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(800);
    println!(
        "\n| tees | opponent model | proposer | work ms p50 / p90 / p99 / max | candidates p50 | extended |\n|---|---|---|---|---|---|"
    );
    let mut worst_two_tee_p99 = 0.0f64;
    for tees in [2usize, 4, 6] {
        for mirror in [false, true] {
            for (pname, units) in [("none", 0u64), ("fly-cost 500", 500)] {
                let mut cfg = deadline(4.0, 1, true, if units > 0 { 3 } else { 0 });
                cfg.work_clock_us_per_tick = Some(WORK_US_PER_TEE_TICK);
                cfg.mirror = mirror;
                let proposer: Box<dyn Proposer> = if units > 0 {
                    Box::new(Costly { units })
                } else {
                    Box::new(NoProposer)
                };
                let mut r = run(&map, &cfg, proposer, tees, n);
                // The proposer's units are work too (`total_ticks` leaves them out): add them to every decision.
                let extra = units as f64;
                let work = |v: &Vec<f64>| -> Vec<f64> { v.iter().map(|t| t + extra).collect() };
                let mut w = work(&r.tee_ticks);
                let ms = |v: &mut Vec<f64>, p: f64| pct(v, p) * WORK_US_PER_TEE_TICK / 1000.0;
                let p99 = ms(&mut w, 99.0);
                if tees == 2 && mirror {
                    worst_two_tee_p99 = worst_two_tee_p99.max(p99);
                }
                println!(
                    "| {tees} | {} | {pname} | {:.2} / {:.2} / {:.2} / {:.2} | {:.0} | {:.1}% |",
                    if mirror { "on" } else { "off" },
                    ms(&mut w, 50.0),
                    ms(&mut w, 90.0),
                    p99,
                    ms(&mut w, 100.0),
                    pct(&mut r.cands, 50.0),
                    100.0 * r.extended as f64 / r.decisions.max(1) as f64
                );
            }
        }
    }
    assert!(
        worst_two_tee_p99 <= 5.0,
        "work p99 at 2 tees with the opponent model is {worst_two_tee_p99:.2} ms, above the 5 ms of D-042"
    );
}

/// Task 3.10, D-042: the finishing switches keep the live decision inside its budget. Decisions of the finishing phase (the victim frozen next to
/// us, 150 ticks of freeze left) on the work clock with the opponent model on: the baseline, the default switch (`HybridConfig::with_finish`, the drag
/// shaping), and -- as a diagnostic, not a promise -- every finishing knob at once (families, staging point, exact forecast); task 3.10b adds the longer
/// horizon (`frozen_steps` 16) and the approach family (`approach_plans` 12), alone and together. The work p99 at 2 tees of every arm but "all knobs"
/// must stay at most 5 ms.
///
/// ```text
/// cargo test -p ddai-planner --release --test hybrid_speed -- --ignored --nocapture finish_work_report
/// ```
#[test]
#[ignore = "heavy; needs the Copy Love Box map"]
fn finish_work_report() {
    let Some(map) = clb() else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    let n: usize = std::env::var("DDAI_SPEED_DECISIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(800);
    println!("\n| tees | finish | work ms p50 / p90 / p99 / max | candidates p50 |\n|---|---|---|---|");
    let mut worst = 0.0f64;
    for tees in [2usize, 4] {
        for mode in [
            "off",
            "default",
            "all knobs",
            "long 16",
            "approach",
            "long 16 + approach",
            "long 16 + approach + default",
            "candidate A",
        ] {
            let mut cfg = deadline(4.0, 1, true, 0);
            cfg.work_clock_us_per_tick = Some(WORK_US_PER_TEE_TICK);
            cfg.mirror = true;
            if matches!(mode, "default" | "all knobs" | "long 16 + approach + default") {
                cfg = cfg.with_finish();
            }
            // Task 3.10b (D-110): the longer horizon of a frozen victim and the approach family.
            if mode.starts_with("long 16") {
                cfg.frozen_steps = 16;
            }
            // The 3.10b duel candidate: the long horizon, 6 approach plans and the long-horizon decisions' own budget of 4.5 ms.
            if mode == "candidate A" {
                cfg.frozen_steps = 16;
                cfg.approach_plans = 6;
                cfg.frozen_budget_ms = Some(4.5);
            }
            if mode.contains("approach") {
                cfg.approach_plans = 12;
            }
            if mode == "all knobs" {
                cfg.finish_families = true;
                cfg.planner.frozen_stage_weight = 1.0;
                cfg.planner.held_forecast_weight = 20.0;
            }
            let mut r = run_scenes(&map, &cfg, Box::new(NoProposer), tees, n, true);
            let mut w = r.tee_ticks.clone();
            let ms = |v: &mut Vec<f64>, p: f64| pct(v, p) * WORK_US_PER_TEE_TICK / 1000.0;
            let p99 = ms(&mut w, 99.0);
            // The bound is the proposed arms'; "all knobs" is a diagnostic.
            if tees == 2 && mode != "all knobs" {
                worst = worst.max(p99);
            }
            println!(
                "| {tees} | {mode} | {:.2} / {:.2} / {:.2} / {:.2} | {:.0} |",
                ms(&mut w, 50.0),
                ms(&mut w, 90.0),
                p99,
                ms(&mut w, 100.0),
                pct(&mut r.cands, 50.0)
            );
        }
    }
    assert!(
        worst <= 5.0,
        "work p99 at 2 tees in the finishing phase is {worst:.2} ms, above the 5 ms of D-042"
    );
}

/// Task 3.9 (D-096, D-042): what the competitor's v2 switches inside the hybrid cost. Per variant, at 2 tees with the opponent
/// model on: the work (tee-ticks x 1.25 us, as D-042 counts it) p50 / p90 / p99 with no proposer and with a proposer the cost of the S fly
/// (500 tee-ticks) inside the cap, and the wall microseconds per tee-tick (the best of `DDAI_V2_REPS` runs of the same decisions): a
/// variant whose price per tee-tick is above the baseline's does work the tick counter does not see. The bound is D-042's: p99 of
/// work <= 5 ms at 2 tees for every variant, the fly's cost priced in. The default sample is 2 400 decisions per variant
/// (`DDAI_SPEED_DECISIONS`): the p99 of a small sample is noisy (800 decisions put `mirror on v2` with the fly at 5.35 ms, 2 400 at 4.37).
///
/// ```text
/// cargo test -p ddai-planner --release --test hybrid_speed -- --ignored --nocapture v2_work_report
/// ```
#[test]
#[ignore = "heavy; needs the Copy Love Box map"]
fn v2_work_report() {
    use ddai_planner::config::{PlannerVersion, preset_live_v2, preset_normal_v2};
    let Some(map) = clb() else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    let n: usize = std::env::var("DDAI_SPEED_DECISIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2400);
    let reps: usize = std::env::var("DDAI_V2_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    type Tweak = fn(&mut HybridConfig);
    let variants: [(&str, Tweak); 9] = [
        ("main (all off)", |_| ()),
        ("hook gate + aim snap", |c| {
            c.planner.hook_exact_gate = true;
            c.planner.hook_snap_aim = true;
        }),
        ("rope ceiling", |c| c.planner.rope_ceiling_cost = 1.0),
        ("polish (+ keep flying)", |c| {
            c.polish = true;
            c.planner.hook_keep_flying = true;
        }),
        ("wall throws + air chains", |c| {
            c.wall_throws = true;
            c.planner.air_chain = true;
        }),
        ("mirror on v2", |c| c.mirror_planner = Some(preset_normal_v2())),
        ("live scoring values", |c| {
            c.planner.launch_exposure = preset_live_v2().launch_exposure;
            c.planner.jumpless_hazard_cost = preset_live_v2().jumpless_hazard_cost;
        }),
        ("v2 planner switches (4)", |c| {
            c.planner = c.planner.with_version(PlannerVersion::Upstream20261002);
        }),
        ("all", |c| {
            c.planner = c.planner.with_version(PlannerVersion::Upstream20261002);
            c.planner.air_chain = true;
            c.polish = true;
            c.wall_throws = true;
            c.mirror_planner = Some(preset_normal_v2());
            c.planner.launch_exposure = preset_live_v2().launch_exposure;
            c.planner.jumpless_hazard_cost = preset_live_v2().jumpless_hazard_cost;
        }),
    ];
    println!(
        "\n| variant | proposer | work ms p50 / p90 / p99 / max | decisions over 5 ms | wall us per tee-tick (best of {reps}) | candidates p50 |\n|---|---|---|---|---|---|"
    );
    let mut worst = 0.0f64;
    // `DDAI_V2_VARIANTS=<substring>` runs only the variants whose name contains it.
    let only = std::env::var("DDAI_V2_VARIANTS").unwrap_or_default();
    for (name, tweak) in variants {
        if !name.contains(&only) {
            continue;
        }
        // `DDAI_FLY_UNITS=500,1000` prices the proposer (task 3.13: the E-005 fly is one view, the E-008 bundle two views).
        let fly_units: Vec<u64> = std::env::var("DDAI_FLY_UNITS")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .filter(|v: &Vec<u64>| !v.is_empty())
            .unwrap_or_else(|| vec![500]);
        let mut proposers: Vec<(String, u64)> = vec![("none".to_string(), 0)];
        proposers.extend(fly_units.iter().map(|&u| (format!("fly-cost {u}"), u)));
        for (pname, units) in proposers {
            let mut cfg = deadline(4.0, 1, true, if units > 0 { 3 } else { 0 });
            cfg.work_clock_us_per_tick = Some(WORK_US_PER_TEE_TICK);
            cfg.mirror = true;
            tweak(&mut cfg);
            let (mut best_us, mut keep) = (f64::INFINITY, None);
            for _ in 0..reps {
                let proposer: Box<dyn Proposer> = if units > 0 {
                    Box::new(Costly { units })
                } else {
                    Box::new(NoProposer)
                };
                let r = run(&map, &cfg, proposer, 2, n);
                let us = r.wall.iter().sum::<f64>() * 1000.0 / r.tee_ticks.iter().sum::<f64>().max(1.0);
                if us < best_us {
                    best_us = us;
                }
                keep = Some(r);
            }
            let mut r = keep.expect("a run");
            let extra = units as f64;
            let mut w: Vec<f64> = r.tee_ticks.iter().map(|t| t + extra).collect();
            let ms = |v: &mut Vec<f64>, p: f64| pct(v, p) * WORK_US_PER_TEE_TICK / 1000.0;
            let p99 = ms(&mut w, 99.0);
            worst = worst.max(p99);
            let over =
                100.0 * w.iter().filter(|&&t| t * WORK_US_PER_TEE_TICK / 1000.0 > 5.0).count() as f64 / w.len() as f64;
            println!(
                "| {name} | {pname} | {:.2} / {:.2} / {:.2} / {:.2} | {over:.2}% | {best_us:.3} | {:.0} |",
                ms(&mut w, 50.0),
                ms(&mut w, 90.0),
                p99,
                ms(&mut w, 100.0),
                pct(&mut r.cands, 50.0)
            );
        }
    }
    assert!(
        worst <= 5.0,
        "work p99 at 2 tees is {worst:.2} ms for some variant, above the 5 ms of D-042"
    );
}
