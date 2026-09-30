//! Diagnostic for review round 1, F1 (disputed, see `docs/EXPERIMENTS.md` E-003): how often does the
//! hybrid send an input that an unbounded `escape_exists` (the shield's own model: stand/walk/jump
//! escapes after the input is held) finds no escape for, and does treating a timed-out check as danger
//! change that? The reviewer's probe found 35 of 2400 (1.46%) in 1v5 on Copy Love Box.
//!
//! Finding: with the extension given to `saferInput` after *every* timeout the count fell 35 -> 4,
//! but the shield then replaced plans the exact rollouts had verified (T14 panic hook 88% -> 0%, T13
//! 88% -> 12%, 1v5 self-freezes 3 -> 10 in the arena), because its escape model has no hook. All 35 of
//! the unsafe inputs are ones the search itself judged safe. So the shield extends after a timeout
//! only when the search also flags the choice unsafe, and the count stays 35. This test records the
//! numbers and guards against the brain getting worse (more unsafe inputs with the extension on).
//! Heavy and needs the map, so `#[ignore]`:
//!
//! ```text
//! cargo test -p ddai-planner --release --test hybrid_shieldgap -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ddai_brain::{Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_physics::map::MapData;
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::World;
use ddai_planner::brains::{ClockKind, ScriptedBrain, enemy_input_from_tee, input_from_action};
use ddai_planner::hybrid::{HybridBrain, HybridConfig, NoProposer};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::shield::{Bounded, escape_exists_bounded};
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

/// (decisions checked, inputs sent with no escape, decisions that came back `shield_incomplete`)
fn probe(map: &Arc<MapData>, adaptive: bool, scenes: u64) -> (u64, u64, u64) {
    let tees = 6usize;
    let mut cfg = HybridConfig {
        proposals: 0,
        work_clock_us_per_tick: Some(2.2),
        ..HybridConfig::default()
    };
    cfg.adaptive.enabled = adaptive;
    let (mut n, mut unsafe_sent, mut incomplete) = (0u64, 0u64, 0u64);
    for scene in 1..=scenes {
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
        let ids: Vec<i32> = (0..tees as i32).collect();
        let mut hb = HybridBrain::new(cfg.clone(), ClockKind::Wall, Box::new(NoProposer)).unwrap();
        hb.reset(&ResetContext {
            map: map.clone(),
            self_id: 0,
            seed: scene,
        });
        let mut sb: Vec<ScriptedBrain> = (1..tees).map(|_| ScriptedBrain::new()).collect();
        for (k, b) in sb.iter_mut().enumerate() {
            b.reset(&ResetContext {
                map: map.clone(),
                self_id: (k + 1) as i32,
                seed: scene * 10 + k as u64,
            });
        }
        let mut last: Vec<PlayerInput> = ids.iter().map(|_| empty_input()).collect();
        for _ in 0..60 {
            let world = pw.inner().clone();
            if pw.get_tee(0).is_none_or(|t| t.frozen || !t.alive) {
                break;
            }
            let dist = |i: i32| {
                let (t, m) = (pw.get_tee(i).unwrap(), pw.get_tee(0).unwrap());
                (t.pos.x - m.pos.x).hypot(t.pos.y - m.pos.y)
            };
            let Some(tgt) = (1..tees as i32)
                .filter(|&i| pw.get_tee(i).is_some_and(|t| t.alive && !t.frozen))
                .min_by(|&a, &b| dist(a).total_cmp(&dist(b)))
            else {
                break;
            };
            for (slot, &id) in ids.iter().enumerate() {
                let Some(me) = char_obs(&world, id) else { continue };
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
                    target_id: Some(if slot == 0 { tgt } else { 0 }),
                    tuning: TuningParams::default(),
                };
                let view = WorldView {
                    world: &world,
                    self_id: id,
                    lag_ticks: 0,
                    in_flight: &[],
                };
                let a = if slot == 0 {
                    hb.decide_in(&obs, Some(&view))
                } else {
                    sb[slot - 1].decide_in(&obs, Some(&view))
                };
                last[slot] = input_from_action(&a, &last[slot]);
                if slot == 0 {
                    let t = hb.last_decision().unwrap();
                    if pw.get_tee(0).unwrap().frozen {
                        continue;
                    }
                    n += 1;
                    incomplete += u64::from(t.shield_incomplete);
                    let mut check = PhysicsWorld::from_world(world.clone(), map.clone());
                    check.sync_from(&world);
                    let mut oth = HashMap::new();
                    for &i in &ids[1..] {
                        if let Some(tt) = check.get_tee(i) {
                            oth.insert(i, enemy_input_from_tee(&tt));
                        }
                    }
                    let ok = matches!(
                        escape_exists_bounded(&mut check, 0, &last[0], 2, &oth, None),
                        Bounded::Done(true)
                    );
                    unsafe_sent += u64::from(!ok);
                }
            }
            for _ in 0..2 {
                for (slot, &id) in ids.iter().enumerate() {
                    pw.set_input(id, last[slot]);
                }
                pw.step();
            }
        }
    }
    (n, unsafe_sent, incomplete)
}

#[test]
#[ignore = "heavy (about a minute in release); needs the Copy Love Box map"]
fn the_extension_never_sends_more_inputs_without_an_escape() {
    let Some(map) = clb() else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    // The probe runs the brain on a big stack: worlds and planners are large.
    let (before, after) = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || (probe(&map, false, 40), probe(&map, true, 40)))
        .unwrap()
        .join()
        .unwrap();
    println!(
        "adaptive off: {} decisions, {} sent with no escape, {} shield_incomplete\nadaptive on:                  {} decisions, {} sent with no escape, {} shield_incomplete",
        before.0, before.1, before.2, after.0, after.1, after.2
    );
    assert!(
        after.1 <= before.1,
        "unsafe inputs sent: {} with the extension against {} without",
        after.1,
        before.1
    );
}
