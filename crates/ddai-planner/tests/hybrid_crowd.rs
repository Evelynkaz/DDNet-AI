//! Crowds (3.5 review F2, task 3.5b): a 1v1 fight on Copy Love Box with idle bystanders outside the
//! threat radius, like a public-server hub. Before local-tee selection (`max_sim_tees = 0`) the
//! search budget shrank with *every* tee in the world and from 10 tees up a decision scored at most
//! two candidates; with the default cap the rollouts step only the local tees, so candidates per
//! decision stay near the 4-6-tee numbers however many tees the server shows. Heavy and needs the map:
//!
//! ```text
//! cargo test -p ddai-planner --release --test hybrid_crowd -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use ddai_brain::{Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_physics::map::MapData;
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::World;
use ddai_planner::brains::{ClockKind, ScriptedBrain, input_from_action};
use ddai_planner::hybrid::{HybridBrain, HybridConfig, NoProposer, WORK_US_PER_TEE_TICK};
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
    Some(Arc::new(ddai_map::load_map(&std::fs::read(f).ok()?).ok()?.data))
}

fn char_obs(w: &World<f32>, id: i32) -> Option<CharacterObservation> {
    let core = w.cores.get(id as u8)?;
    let ch = w.characters[id as usize].as_ref()?;
    let mut c = CharacterObservation::at_rest(id);
    c.pos = core.pos;
    c.vel = core.vel;
    c.hook_state = core.hook_state;
    c.hooked_player = core.hooked_player();
    c.is_frozen = ch.freeze_time > 0;
    c.freeze_ticks_remaining = ch.freeze_time;
    c.direction = core.direction;
    Some(c)
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[(((v.len() - 1) as f64) * p / 100.0).round() as usize]
}

struct Row {
    decisions: u64,
    budget_p50: f64,
    cands: [f64; 3],
    tiny_pool: f64,
    incomplete: f64,
    sim_tees_p50: f64,
    work_ms_p99: f64,
}

/// `attackers`: the bystanders stand near the fight and attack us (scripted bots) instead of idling far away.
fn measure(map: &Arc<MapData>, max_sim_tees: usize, bystanders: usize, attackers: bool) -> Row {
    let cfg = HybridConfig {
        proposals: 0,
        work_clock_us_per_tick: Some(WORK_US_PER_TEE_TICK),
        max_sim_tees,
        ..HybridConfig::default()
    };
    let (mut cands, mut budget, mut sim, mut work) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut dec, mut inc, mut tiny) = (0u64, 0u64, 0u64);
    for scene in 1..=6u64 {
        let mut pw = PhysicsWorld::new(map.clone(), scene);
        pw.add_tee(
            0,
            Vec2 {
                x: 103.5 * 32.0,
                y: 84.5 * 32.0,
            },
        );
        pw.add_tee(
            1,
            Vec2 {
                x: 100.6 * 32.0,
                y: 84.5 * 32.0,
            },
        );
        for k in 0..bystanders {
            pw.add_tee(
                2 + k as i32,
                Vec2 {
                    x: if attackers {
                        (96.0 - 1.1 * k as f64).max(80.0)
                    } else {
                        (86.0 - 0.9 * k as f64).max(80.0)
                    } * 32.0,
                    y: 84.5 * 32.0,
                },
            );
        }
        let ids: Vec<i32> = (0..(2 + bystanders) as i32).collect();
        let mut hb = HybridBrain::new(cfg.clone(), ClockKind::Wall, Box::new(NoProposer)).unwrap();
        hb.reset(&ResetContext {
            map: map.clone(),
            self_id: 0,
            seed: scene,
        });
        let mut sb = ScriptedBrain::new();
        sb.reset(&ResetContext {
            map: map.clone(),
            self_id: 1,
            seed: scene + 50,
        });
        let mut crowd: Vec<ScriptedBrain> = (0..bystanders).map(|_| ScriptedBrain::new()).collect();
        for (k, b) in crowd.iter_mut().enumerate() {
            b.reset(&ResetContext {
                map: map.clone(),
                self_id: (2 + k) as i32,
                seed: scene * 100 + k as u64,
            });
        }
        let mut last: Vec<PlayerInput> = ids.iter().map(|_| empty_input()).collect();
        for _ in 0..60 {
            let world = pw.inner().clone();
            if [0, 1]
                .iter()
                .any(|&i| pw.get_tee(i).is_none_or(|t| t.frozen || !t.alive))
            {
                break;
            }
            #[allow(clippy::needless_range_loop)]
            for slot in 0..ids.len() {
                if slot >= 2 && !attackers {
                    break;
                }
                if world.cores.get(slot as u8).is_none() {
                    continue;
                }
                let id = slot as i32;
                let me = char_obs(&world, id).unwrap();
                let others: Vec<CharacterObservation> = ids
                    .iter()
                    .filter(|&&i| i != id)
                    .filter_map(|&i| char_obs(&world, i))
                    .collect();
                let obs = Observation {
                    map: map.clone(),
                    tick: world.tick,
                    self_state: me,
                    others,
                    target_id: Some(if slot >= 2 { 0 } else { 1 - id }),
                    tuning: TuningParams::default(),
                };
                let view = WorldView {
                    world: &world,
                    self_id: id,
                    lag_ticks: 0,
                    in_flight: &[],
                };
                let a = if slot == 0 {
                    let a = hb.decide_in(&obs, Some(&view));
                    let t = hb.last_decision().unwrap();
                    dec += 1;
                    inc += u64::from(t.shield_incomplete);
                    let n: u32 = t.evaluated.iter().sum();
                    cands.push(f64::from(n));
                    budget.push(t.budget_ms);
                    sim.push(f64::from(t.sim_tees));
                    work.push(t.work.total_ticks() as f64 * f64::from(t.sim_tees) * WORK_US_PER_TEE_TICK / 1000.0);
                    tiny += u64::from(n <= 2);
                    a
                } else if slot == 1 {
                    sb.decide_in(&obs, Some(&view))
                } else {
                    crowd[slot - 2].decide_in(&obs, Some(&view))
                };
                last[slot] = input_from_action(&a, &last[slot]);
            }
            for _ in 0..2 {
                for (slot, &id) in ids.iter().enumerate() {
                    pw.set_input(id, last[slot]);
                }
                pw.step();
            }
        }
    }
    Row {
        decisions: dec,
        budget_p50: pct(&mut budget, 50.0),
        cands: [pct(&mut cands, 10.0), pct(&mut cands, 50.0), pct(&mut cands, 90.0)],
        tiny_pool: 100.0 * tiny as f64 / dec as f64,
        incomplete: 100.0 * inc as f64 / dec as f64,
        sim_tees_p50: pct(&mut sim, 50.0),
        work_ms_p99: pct(&mut work, 99.0),
    }
}

#[test]
#[ignore = "heavy (a minute or two in release); needs the Copy Love Box map"]
fn crowd_scaling() {
    let Some(map) = clb() else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            println!(
                "| tees in the world | max_sim_tees | decisions | search budget p50 ms | candidates p10 / p50 / p90 | pool <= 2 | shield incomplete | simulated tees p50 | work ms p99 |\n|---|---|---|---|---|---|---|---|---|"
            );
            let mut default_p50 = Vec::new();
            for (bystanders, attackers) in [
                (0usize, false),
                (2, false),
                (4, false),
                (6, false),
                (8, false),
                (10, false),
                (13, false),
                (7, true),
                (13, true),
            ] {
                for cap in [0usize, 4] {
                    let r = measure(&map, cap, bystanders, attackers);
                    println!(
                        "| {}{} | {} | {} | {:.2} | {:.0} / {:.0} / {:.0} | {:.1}% | {:.1}% | {:.0} | {:.2} |",
                        2 + bystanders,
                        if attackers { " (all attack)" } else { "" },
                        if cap == 0 { "all (3.5)".to_string() } else { cap.to_string() },
                        r.decisions,
                        r.budget_p50,
                        r.cands[0],
                        r.cands[1],
                        r.cands[2],
                        r.tiny_pool,
                        r.incomplete,
                        r.sim_tees_p50,
                        r.work_ms_p99
                    );
                    if cap == 4 && !attackers {
                        default_p50.push((2 + bystanders, r.cands[1], r.work_ms_p99));
                    }
                }
            }
            // With the cap, an idle crowd must cost like the fight alone: candidates at 15 tees stay
            // within 10% of the 2-tee number, and the work stays under the 5 ms target (plus a rollout).
            let at2 = default_p50.iter().find(|r| r.0 == 2).unwrap().1;
            let at15 = default_p50.iter().find(|r| r.0 == 15).unwrap();
            assert!(at15.1 >= 0.9 * at2, "candidates at 15 tees {} against {at2} at 2", at15.1);
            assert!(at15.2 < 5.6, "work p99 at 15 tees {} ms", at15.2);
        })
        .unwrap()
        .join()
        .unwrap();
}
