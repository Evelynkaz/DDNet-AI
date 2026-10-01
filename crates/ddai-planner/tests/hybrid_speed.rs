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
use ddai_planner::hybrid::{HybridBrain, HybridConfig, HybridMode, NoProposer, Proposer, ScriptedProposer};
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
    rollout_ms: Vec<f64>,
    stage1_ticks: Vec<f64>,
    stage2_ticks: Vec<f64>,
    shield_ticks: Vec<f64>,
    ext_ticks: Vec<f64>,
    proposal_ticks: Vec<f64>,
    extended: u64,
    shield_incomplete: u64,
    decisions: u64,
}

/// Plays scenes of `tees` tees (slot 0 = the hybrid, the others scripted attackers) on the left
/// wayblock hall and records `decisions` decisions of slot 0.
fn run(map: &Arc<MapData>, cfg: &HybridConfig, proposer: Box<dyn Proposer>, tees: usize, decisions: usize) -> Row {
    let mut row = Row {
        wall: vec![],
        ticks: vec![],
        tee_ticks: vec![],
        cands: vec![],
        fly_ms: vec![],
        search_ms: vec![],
        overshoot_ms: vec![],
        shield_ms: vec![],
        rollout_ms: vec![],
        stage1_ticks: vec![],
        stage2_ticks: vec![],
        shield_ticks: vec![],
        ext_ticks: vec![],
        proposal_ticks: vec![],
        extended: 0,
        shield_incomplete: 0,
        decisions: 0,
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
            if out(0) || (1..tees as i32).all(out) {
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
                    row.tee_ticks
                        .push((t.work.total_ticks() * u64::from(t.sim_tees.max(1))) as f64);
                    row.cands.push(f64::from(t.evaluated.iter().sum::<u32>()));
                    row.fly_ms.push(t.proposal_ms);
                    row.search_ms.push(t.search_ms);
                    row.overshoot_ms.push(t.search_ms - t.budget_ms);
                    row.shield_ms.push(t.shield_ms);
                    row.rollout_ms.push(t.rollout_ms);
                    row.stage1_ticks.push(t.work.stage1 as f64);
                    row.stage2_ticks.push(t.work.stage2 as f64);
                    row.shield_ticks.push(t.work.shield as f64);
                    row.ext_ticks.push(t.work.extension as f64);
                    row.proposal_ticks.push(t.work.proposal as f64);
                    row.extended += u64::from(t.extended);
                    row.shield_incomplete += u64::from(t.shield_incomplete);
                    row.decisions += 1;
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
        pct(&mut r.tee_ticks, 50.0) * 2.2 / 1000.0,
        pct(&mut r.tee_ticks, 90.0) * 2.2 / 1000.0,
        pct(&mut r.tee_ticks, 99.0) * 2.2 / 1000.0,
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

const HEADER: &str = "| Condition | wall ms p50/p90/p99 | work ticks p50/p90/p99 | physics-only ms (ticks x us/tick) p50/p90/p99 | work ms (tee-ticks x 2.2 us, the work-clock calibration) p50/p90/p99 | candidates p50/p90/p99 | extended |\n|---|---|---|---|---|---|---|";

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
    println!("Budget 4 ms of work = 1 818 tee-ticks, no proposer; work ms = tee-ticks x 2.2 us.\n");
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
            cfg.work_clock_us_per_tick = Some(2.2);
            cfg.decision_cap_ms = cap;
            cfg.shield_reserve_ms_per_tee = reserve;
            cfg.shield_timeout_danger = timeout_danger;
            let mut r = run(&map, &cfg, Box::new(NoProposer), tees, n * 2);
            let ms = |v: &mut Vec<f64>, p: f64| pct(v, p) * 2.2 / 1000.0;
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
