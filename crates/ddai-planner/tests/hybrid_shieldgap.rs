//! Shield outcome probes (the reviewer's probes of 3.5, turned into tests in 3.5b). The shield's
//! verdict is judged by **outcomes**, not by itself: the count of sent inputs with no escape by the
//! old walk/jump model is reported, but the measure that counts is whether we actually froze or died
//! within `HORIZON` ticks after the decision -- in a 1v5 on Copy Love Box and in a T14-like jumpless
//! fall beside a wall.
//!
//! Variants (`shield_plan_escape`, `shield_hook_anchors`, `shield_timeout_danger`): `3.5` = none of
//! them (the merged behaviour), `plan` = the remainder of the chosen plan is the first escape,
//! `plan+hooks` adds anchor hook escapes, `plan+hooks+timeout` also counts a timed-out check as
//! danger. Heavy and needs the map, so `#[ignore]`:
//!
//! ```text
//! cargo test -p ddai-planner --release --test hybrid_shieldgap -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ddai_brain::{Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::World;
use ddai_planner::brains::{ClockKind, ScriptedBrain, enemy_input_from_tee, input_from_action};
use ddai_planner::hybrid::{HybridBrain, HybridConfig, NoProposer};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::shield::{Bounded, escape_exists_bounded};
use ddai_planner::types::{PlayerInput, empty_input};
use ddai_planner::vmath::Vec2;

/// A decision "led to" a freeze when we were out within this many ticks after it.
const HORIZON: i32 = 30;

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

fn variant(name: &str) -> HybridConfig {
    let mut cfg = HybridConfig {
        proposals: 0,
        work_clock_us_per_tick: Some(2.2),
        shield_plan_escape: false,
        shield_hook_anchors: 0,
        shield_timeout_danger: false,
        ..HybridConfig::default()
    };
    match name {
        "3.5" => {}
        "plan" => cfg.shield_plan_escape = true,
        "plan+hooks" => {
            cfg.shield_plan_escape = true;
            cfg.shield_hook_anchors = 3;
        }
        "plan+hooks+timeout" => {
            cfg.shield_plan_escape = true;
            cfg.shield_hook_anchors = 3;
            cfg.shield_timeout_danger = true;
        }
        other => panic!("unknown variant {other}"),
    }
    cfg
}

#[derive(Default, Debug)]
struct Outcome {
    decisions: u64,
    /// Inputs sent with no escape by the plain walk/jump model (the reviewer's metric).
    no_escape_old_model: u64,
    shield_incomplete: u64,
    substituted: u64,
    shield_ran: u64,
    plan_ok: u64,
    /// Decisions after which we were frozen or dead within `HORIZON` ticks.
    froze_within: u64,
    /// Scenes that ended with us out.
    scenes_out: u64,
    scenes: u64,
}

/// 1v5 on Copy Love Box, 6 tees, scripted attackers, `scenes` scenes of up to 60 decisions.
fn probe_1v5(map: &Arc<MapData>, cfg: &HybridConfig, scenes: u64) -> Outcome {
    let tees = 6usize;
    let mut o = Outcome::default();
    for scene in 1..=scenes {
        o.scenes += 1;
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
        // Decision ticks not yet known to be followed by a freeze.
        let mut open: Vec<i32> = Vec::new();
        let mut out = false;
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
                    o.decisions += 1;
                    o.shield_incomplete += u64::from(t.shield_incomplete);
                    o.substituted += u64::from(t.shielded);
                    o.shield_ran += u64::from(t.shield_ran);
                    o.plan_ok += u64::from(t.shield_plan_ok);
                    open.push(world.tick);
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
                    o.no_escape_old_model += u64::from(!ok);
                }
            }
            for _ in 0..2 {
                for (slot, &id) in ids.iter().enumerate() {
                    pw.set_input(id, last[slot]);
                }
                pw.step();
                if pw.get_tee(0).is_none_or(|t| t.frozen || !t.alive) && !out {
                    out = true;
                    let now = pw.inner().tick;
                    o.froze_within += open.iter().filter(|&&d| now - d <= HORIZON).count() as u64;
                }
            }
            if out {
                break;
            }
        }
        o.scenes_out += u64::from(out);
    }
    o
}

fn wall_over_freeze() -> Arc<MapData> {
    let (w, h) = (60usize, 20usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            let solid = y == 0 || x == 0 || x == w - 1 || y == h - 1 || (x <= 20 && y >= 1);
            let freeze = (21..=58).contains(&x) && (16..=18).contains(&y);
            game[y * w + x] = Tile {
                index: if solid {
                    TILE_SOLID
                } else if freeze {
                    TILE_FREEZE
                } else {
                    0
                },
                ..Tile::default()
            };
        }
    }
    Arc::new(MapData {
        width: w as u32,
        height: h as u32,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    })
}

/// A jumpless fall over freeze beside a wall (scenario T14), 60 start states: how many survive.
fn t14_like(cfg: &HybridConfig) -> (u32, u32) {
    let map = wall_over_freeze();
    let (mut trials, mut survived) = (0, 0);
    for xi in 0..5 {
        for yi in 0..3 {
            for vi in 0..4 {
                trials += 1;
                let mut pw = PhysicsWorld::new(map.clone(), 1);
                pw.add_tee(
                    0,
                    Vec2 {
                        x: (22.0 + f64::from(xi)) * 32.0,
                        y: (6.0 + 1.5 * f64::from(yi)) * 32.0,
                    },
                );
                pw.add_tee(
                    1,
                    Vec2 {
                        x: 45.5 * 32.0,
                        y: 14.5 * 32.0,
                    },
                );
                let mut st = pw.get_tee(0).unwrap();
                st.vel = Vec2 {
                    x: -1.0 + f64::from(vi),
                    y: 2.0 + f64::from(vi),
                };
                st.jumped = 3;
                st.jumped_total = Some(2);
                st.jumps_left = 0;
                pw.apply_tee_state(0, &st);
                let mut b = HybridBrain::new(cfg.clone(), ClockKind::Wall, Box::new(NoProposer)).unwrap();
                b.reset(&ResetContext {
                    map: map.clone(),
                    self_id: 0,
                    seed: 3,
                });
                let mut last = empty_input();
                let mut out = false;
                let mut log: Vec<String> = Vec::new();
                for _ in 0..40 {
                    let world = pw.inner().clone();
                    let obs = Observation {
                        map: map.clone(),
                        tick: world.tick,
                        self_state: char_obs(&world, 0).unwrap(),
                        others: vec![char_obs(&world, 1).unwrap()],
                        target_id: Some(1),
                        tuning: TuningParams::default(),
                    };
                    let view = WorldView {
                        world: &world,
                        self_id: 0,
                        lag_ticks: 0,
                        in_flight: &[],
                    };
                    let a = b.decide_in(&obs, Some(&view));
                    if std::env::var("DDAI_SHIELD_TRACE").is_ok() {
                        let t = b.last_decision().unwrap();
                        log.push(format!(
                            "  T14 trial {trials} tick {}: {} ran {} plan_ok {} shielded {} incomplete {} unsafe {} dir {} jump {} hook {}",
                            world.tick,
                            t.chosen.map_or("none", |c| c.label()),
                            t.shield_ran,
                            t.shield_plan_ok,
                            t.shielded,
                            t.shield_incomplete,
                            t.unsafe_choice,
                            a.direction,
                            a.jump,
                            a.hook
                        ));
                    }
                    last = input_from_action(&a, &last);
                    for _ in 0..2 {
                        pw.set_input(0, last);
                        pw.step();
                        out |= pw.get_tee(0).is_none_or(|t| t.frozen || !t.alive);
                    }
                    if out {
                        break;
                    }
                }
                survived += u32::from(!out);
                if out && !log.is_empty() {
                    eprintln!("FAILED trial {trials}:\n{}", log.join("\n"));
                }
            }
        }
    }
    (survived, trials)
}

const VARIANTS: [&str; 4] = ["3.5", "plan", "plan+hooks", "plan+hooks+timeout"];

#[test]
#[ignore = "heavy (minutes in release); needs the Copy Love Box map"]
fn shield_variants_by_outcome() {
    let Some(map) = clb() else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    let scenes: u64 = std::env::var("DDAI_SHIELD_SCENES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    // Worlds and planners are large: run on a big stack.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            println!("| variant | 1v5: decisions | no escape (walk/jump model) | shield incomplete | shield ran | plan escape held | substituted | froze within {HORIZON} ticks | scenes out | T14-like survived |");
            println!("|---|---|---|---|---|---|---|---|---|---|");
            for v in VARIANTS {
                let cfg = variant(v);
                let o = probe_1v5(&map, &cfg, scenes);
                let (s, t) = t14_like(&cfg);
                println!(
                    "| {v} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {s}/{t} |",
                    o.decisions,
                    o.no_escape_old_model,
                    o.shield_incomplete,
                    o.shield_ran,
                    o.plan_ok,
                    o.substituted,
                    o.froze_within,
                    o.scenes_out,
                    o.scenes
                );
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
