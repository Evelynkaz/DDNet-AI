//! Review round 2, F12/F6: per-phase step-count/wall-time breakdown of `decide_production`
//! (search / "mid" tail logic / shield), on Copy Love Box, adapted from the reviewer's own scratch
//! harness (`.../scratchpad/3.2/r2/crates/ddai-planner/tests/zz_r2_phase.rs`) so it can be re-run
//! against this crate directly. Work counters (physics steps, `escape_exists` calls -- see
//! `ddai_planner::prof`) are the load-bearing, stall-proof measurement; wall time is reported
//! alongside for context but this VM's scheduler stalls inflate it unpredictably (see the crate
//! README's D-041 section).
//!
//! `#[ignore]`d: needs the local Copy Love Box map. Run:
//! ```text
//! N=1000 B=4.0 cargo test -p ddai-planner --release --test phase_breakdown -- --ignored --nocapture
//! ```
//! `N` = decisions per (hall, tee-count) condition (default 500); `B` = comma-separated budgets in
//! ms (default `4.0`); `HALL_ONLY=1` restricts to the E-000 left-hall arena (`stand`'s `hall`
//! argument -- WB boxes L1 `{79..104,67..79}` ∪ L2 `{78..104,79..87}`), matching F15's ask that
//! quality/timing numbers say which arena they used.

use ddai_jsmath::Rng;
use ddai_planner::clock::WallClock;
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::{PlanCollision, PlanWorld};
use ddai_planner::planner::Planner;
use ddai_planner::scripted::scripted_action;
use ddai_planner::types::{PlayerInput, empty_input};
use ddai_planner::vmath::Vec2;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

const CLB: &str = "/home/ubuntu/aiddnet/data/maps/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map";

/// Standable tiles, optionally restricted to the E-000 left-hall arena (two WB boxes).
fn stand(col: &impl PlanCollision, hall: bool) -> Vec<Vec2> {
    let c = |t: i32| f64::from(t) * 32.0 + 16.0;
    let mut out = Vec::new();
    for ty in 0..col.height() - 1 {
        for tx in 0..col.width() {
            if hall
                && !(((79..=104).contains(&tx) && (67..=79).contains(&ty))
                    || ((78..=104).contains(&tx) && (79..=87).contains(&ty)))
            {
                continue;
            }
            let (px, py) = (c(tx), c(ty));
            if col.is_solid(px, py) || col.is_freeze(px, py) || col.is_death(px, py) {
                continue;
            }
            if !col.is_solid(px, py + 32.0) {
                continue;
            }
            out.push(Vec2 { x: px, y: py });
        }
    }
    out
}

fn pair(st: &[Vec2], rng: &mut Rng) -> (Vec2, Vec2) {
    let pick = |rng: &mut Rng| st[(rng.next_float() * st.len() as f64) as usize];
    for _ in 0..10_000 {
        let a = pick(rng);
        let b = pick(rng);
        let d = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt() / 32.0;
        if (3.0..=14.0).contains(&d) {
            return (a, b);
        }
    }
    (pick(rng), pick(rng))
}

fn pc(t: &[f64], p: f64) -> f64 {
    if t.is_empty() {
        return 0.0;
    }
    t[((t.len() - 1) as f64 * p).round() as usize]
}
fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
#[ignore = "needs the local Copy Love Box map -- see this file's doc comment"]
fn phase_breakdown() {
    // Review round 3, F16: `prof` is off by default now (fixed-size storage, but still only
    // written when enabled) -- this harness reads marks, so it must opt in explicitly.
    ddai_planner::prof::enable();
    let bytes = std::fs::read(CLB)
        .unwrap_or_else(|e| panic!("reading {CLB}: {e} (local map data, not committed -- see CLAUDE.md)"));
    let map = Arc::new(ddai_map::load_map(&bytes).unwrap().data);
    let w0 = PhysicsWorld::new(map.clone(), 1);
    let n: usize = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(500);
    let clock = WallClock::new();
    let budgets: Vec<f64> = std::env::var("B")
        .ok()
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![4.0]);
    let hall_only = std::env::var("HALL_ONLY").is_ok_and(|v| v == "1");
    let halls: &[bool] = if hall_only { &[true] } else { &[false, true] };
    for &hall in halls {
        let st = stand(w0.collision(), hall);
        for tees in [2usize, 6] {
            for &budget in &budgets {
                let mut shield_pairs: Vec<(f64, f64)> = vec![];
                let mut rng = Rng::new(42);
                let (mut tot, mut srch, mut mid, mut sh, mut sh_steps, mut esc) =
                    (vec![], vec![], vec![], vec![], vec![], vec![]);
                let mut games = 0u64;
                let mut shielded_n = 0u64;
                let mut incomplete_n = 0u64;
                // Review round 3, F18: the reviewer's own "missed-save" method -- on every
                // `shield_incomplete` decision, re-run the *full, unbounded* shield
                // (`escape_exists`/`safer_input`, no deadline, so it always runs to a complete
                // answer) against the same post-decision world state to see whether the bounded
                // check missed something real: `missed.0` counts decisions where the kept input
                // truly had no escape (the bounded check's timeout really did hide a danger);
                // `missed.1` counts how many of those the full `saferInput` would have fixed.
                let (mut missed_no_escape, mut missed_would_fix) = (0u64, 0u64);
                while tot.len() < n {
                    games += 1;
                    let mut w = PhysicsWorld::new(map.clone(), games);
                    let (a, b) = pair(&st, &mut rng);
                    w.add_tee(0, a);
                    w.add_tee(1, b);
                    for id in 2..tees as i32 {
                        let p = st[(rng.next_float() * st.len() as f64) as usize];
                        w.add_tee(id, p);
                        w.set_held_input(id, empty_input());
                    }
                    let mut p: Planner<PhysicsWorld> = Planner::new(ddai_planner::config::preset_normal());
                    p.reset();
                    let mut srng = Rng::new(99 + games as u32);
                    let (mut sp, mut ep) = (empty_input(), empty_input());
                    for k in 0..400 {
                        let (Some(x), Some(y)) = (w.get_tee(0), w.get_tee(1)) else {
                            break;
                        };
                        if x.frozen || y.frozen || !x.alive || !y.alive {
                            break;
                        }
                        let ei = scripted_action(&w, 1, 0, &ep, &mut srng);
                        let _ = ddai_planner::prof::take();
                        let t0 = Instant::now();
                        let si = p.decide_production(&mut w, 0, 1, sp, ei, &clock, budget);
                        let total = t0.elapsed().as_secs_f64() * 1e3;
                        let m = ddai_planner::prof::take();
                        if k > 0 && m.len() == 4 {
                            let d = |i: usize, j: usize| (m[j].0 - m[i].0).as_secs_f64() * 1e3;
                            tot.push(total);
                            srch.push(d(0, 1));
                            mid.push(d(1, 2));
                            sh.push(d(2, 3));
                            sh_steps.push((m[3].2 - m[2].2) as f64);
                            shield_pairs.push(((m[3].2 - m[2].2) as f64, d(2, 3)));
                            esc.push((m[3].3 - m[2].3) as f64);
                            if p.last_info.shielded {
                                shielded_n += 1;
                            }
                            if p.last_info.shield_incomplete {
                                incomplete_n += 1;
                                let full_others: HashMap<i32, PlayerInput> = HashMap::from([(1, ei)]);
                                if !ddai_planner::shield::escape_exists(&mut w, 0, &si, 2, &full_others) {
                                    missed_no_escape += 1;
                                    if ddai_planner::shield::safer_input(&mut w, 0, &si, 2, &full_others, Some(&sp))
                                        .is_some()
                                    {
                                        missed_would_fix += 1;
                                    }
                                }
                            }
                        }
                        w.set_input(0, si);
                        w.set_input(1, ei);
                        w.step();
                        sp = si;
                        ep = ei;
                        if tot.len() >= n {
                            break;
                        }
                    }
                }
                let (tot, srch, mid, sh, sh_steps, esc) = (
                    sorted(tot),
                    sorted(srch),
                    sorted(mid),
                    sorted(sh),
                    sorted(sh_steps),
                    sorted(esc),
                );
                eprintln!(
                    "hall={hall} tees={tees} budget={budget}: n={} total p50/p90/p99/max={:.2}/{:.2}/{:.2}/{:.2} | search p99={:.2} max={:.2} | mid p99={:.2} max={:.2} | shield wall p50/p90/p99/max={:.2}/{:.2}/{:.2}/{:.2} steps p50/p90/p99/max={}/{}/{}/{} escapeExists p99={} max={} | shielded={shielded_n} shield_incomplete={incomplete_n}",
                    tot.len(),
                    pc(&tot, 0.5),
                    pc(&tot, 0.9),
                    pc(&tot, 0.99),
                    tot[tot.len() - 1],
                    pc(&srch, 0.99),
                    srch[srch.len() - 1],
                    pc(&mid, 0.99),
                    mid[mid.len() - 1],
                    pc(&sh, 0.5),
                    pc(&sh, 0.9),
                    pc(&sh, 0.99),
                    sh[sh.len() - 1],
                    pc(&sh_steps, 0.5) as i64,
                    pc(&sh_steps, 0.9) as i64,
                    pc(&sh_steps, 0.99) as i64,
                    sh_steps[sh_steps.len() - 1] as i64,
                    pc(&esc, 0.99),
                    esc[esc.len() - 1],
                );
                shield_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                eprintln!(
                    "   top shield (steps, wall ms): {:?}",
                    &shield_pairs[..shield_pairs.len().min(5)]
                );
                eprintln!(
                    "   incomplete decisions where the full (unbounded) shield finds NO escape for the kept input: {missed_no_escape} ; of those, full saferInput WOULD have substituted: {missed_would_fix}"
                );
            }
        }
    }
}
