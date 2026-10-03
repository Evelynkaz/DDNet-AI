//! Stability of the hybrid brain (task 3.5, owner priority "speed, quality, stability"): hostile and
//! edge inputs never panic, and a long session neither grows memory nor slows down.
//!
//! * `hostile_and_edge_inputs_never_panic`: an empty world, no opponent, a dead target, a target
//!   that is not in the world, a frozen self, a dead self, 12 tees, tees at the map corners and
//!   outside it, huge velocities, input lag, every worker count.
//! * `soak_*`: a varied stream of scenes (2-15 tees, random positions, velocities, freezes,
//!   deaths, respawns, resets) decided in a row by one brain; live heap (`allocation_counter`)
//!   after the warm-up stays flat, and the work per decision stays under its bound.
//!   The short soak runs in every `cargo test`; `soak_ten_thousand_decisions` (`--ignored`) is the
//!   10 000-decision version.

use std::sync::Arc;

use allocation_counter::measure;
use ddai_brain::{Action, Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_jsmath::Rng;
use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::World;
use ddai_planner::brains::{ClockKind, input_from_action};
use ddai_planner::hybrid::{HybridBrain, HybridConfig, HybridMode, NoProposer, ScriptedProposer, WORK_US_PER_TEE_TICK};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::empty_input;
use ddai_planner::vmath::Vec2;

fn hall() -> Arc<MapData> {
    let (w, h) = (48usize, 20usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            let solid = y >= 14
                || x == 0
                || x == w - 1
                || y == 0
                || (x <= 3 && y >= 3)
                || (30..=33).contains(&x) && (5..=6).contains(&y);
            let freeze = (14..=15).contains(&y) && (10..=17).contains(&x)
                || (14..=15).contains(&y) && (26..=28).contains(&x)
                || y == 1 && (15..=25).contains(&x);
            game[y * w + x] = Tile {
                index: if freeze {
                    TILE_FREEZE
                } else if solid {
                    TILE_SOLID
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
    Some(c)
}

fn obs_for(w: &World<f32>, map: &Arc<MapData>, me: i32, ids: &[i32], target: Option<i32>) -> Option<Observation> {
    Some(Observation {
        map: map.clone(),
        tick: w.tick,
        self_state: char_obs(w, me)?,
        others: ids
            .iter()
            .filter(|&&i| i != me)
            .filter_map(|&i| char_obs(w, i))
            .collect(),
        target_id: target,
        tuning: TuningParams::default(),
    })
}

fn brain(cfg: HybridConfig) -> HybridBrain {
    HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).expect("config")
}

fn reset(b: &mut HybridBrain, map: &Arc<MapData>, seed: u64) {
    b.reset(&ResetContext {
        map: map.clone(),
        self_id: 0,
        seed,
    });
}

fn valid(a: &Action) -> bool {
    (-1..=1).contains(&a.direction)
}

fn fixed(workers: usize) -> HybridConfig {
    HybridConfig {
        workers,
        proposals: 0,
        ..HybridConfig::fixed()
    }
}

/// The test bodies hold a dozen `World` copies (~105 kB each) on the stack, more than a test
/// thread's default 2 MB; the brain itself keeps its worlds on the heap.
fn on_big_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("the test body panicked");
}

#[test]
fn hostile_and_edge_inputs_never_panic() {
    on_big_stack(hostile_inputs);
}

/// The configurations the hostile inputs run under: fixed work with 1 and 3 workers, fixed work with
/// every 3.5b option on and the tightest local-tee cap, and a wall-clock deadline with the options of
/// the speed work (dynamic stage 2, pruning) and the timeout-as-danger shield.
fn hostile_configs() -> Vec<HybridConfig> {
    let options = |mut c: HybridConfig| {
        c.prune.enabled = true;
        c.robust.crowd_stage = true;
        c.shield_timeout_danger = true;
        c.shield_hook_anchors = 8;
        c.two_world = true;
        c
    };
    vec![
        fixed(1),
        fixed(3),
        options(HybridConfig {
            max_sim_tees: 1,
            ..fixed(1)
        }),
        options(HybridConfig {
            mode: HybridMode::Deadline { budget_ms: 3.0 },
            stage2_dynamic: true,
            max_sim_tees: 3,
            proposals: 0,
            ..HybridConfig::default()
        }),
    ]
}

fn hostile_inputs() {
    let map = hall();
    for cfg in hostile_configs() {
        let mut b = brain(cfg);
        reset(&mut b, &map, 1);

        // A world with only us, and an observation with no opponent.
        let mut pw = PhysicsWorld::new(map.clone(), 1);
        pw.add_tee(0, Vec2 { x: 100.0, y: 400.0 });
        let w = pw.inner().clone();
        let mut o = obs_for(&w, &map, 0, &[0], None).unwrap();
        let view = WorldView {
            world: &w,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        assert!(valid(&b.decide_in(&o, Some(&view))));
        assert!(valid(&b.decide(&o)));
        assert!(valid(&b.decide_in(&o, None)));

        // Hostile live context: NaN and absurd spared positions, spared ids that are us, missing or negative,
        // a NaN travel goal; every tee spared (nothing to play against); then cleared again.
        b.set_spares(
            vec![
                Vec2 { x: f64::NAN, y: 1.0 },
                Vec2 { x: 1e30, y: -1e30 },
                Vec2 { x: 100.0, y: 400.0 },
            ],
            vec![Vec2 {
                x: f64::INFINITY,
                y: 0.0,
            }],
        );
        b.set_spare_ids(vec![0, 1, 99, -5]);
        b.set_travel_goal(Some(Vec2 {
            x: f64::NAN,
            y: f64::NAN,
        }));
        assert!(valid(&b.decide_in(&o, Some(&view))));
        let mut pwl = PhysicsWorld::new(map.clone(), 1);
        pwl.add_tee(0, Vec2 { x: 100.0, y: 400.0 });
        pwl.add_tee(1, Vec2 { x: 300.0, y: 400.0 });
        let wl = pwl.inner().clone();
        let ol = obs_for(&wl, &map, 0, &[0, 1], Some(1)).unwrap();
        let viewl = WorldView {
            world: &wl,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        assert!(valid(&b.decide_in(&ol, Some(&viewl))), "the only other tee is spared");
        b.set_spares(Vec::new(), Vec::new());
        b.set_spare_ids(Vec::new());
        b.set_travel_goal(None);
        assert!(valid(&b.decide_in(&ol, Some(&viewl))));

        // The observation names a target that is not in the world.
        let mut pw = PhysicsWorld::new(map.clone(), 1);
        pw.add_tee(0, Vec2 { x: 100.0, y: 400.0 });
        pw.add_tee(1, Vec2 { x: 300.0, y: 400.0 });
        let w = pw.inner().clone();
        o = obs_for(&w, &map, 0, &[0, 1], Some(9)).unwrap();
        let view = WorldView {
            world: &w,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        assert!(valid(&b.decide_in(&o, Some(&view))));

        // The target dies between the observation and the decision (the world has no tee 1).
        let mut pw2 = PhysicsWorld::new(map.clone(), 1);
        pw2.add_tee(0, Vec2 { x: 100.0, y: 400.0 });
        let w2 = pw2.inner().clone();
        let o1 = obs_for(&w, &map, 0, &[0, 1], Some(1)).unwrap();
        let view2 = WorldView {
            world: &w2,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        assert!(valid(&b.decide_in(&o1, Some(&view2))));

        // We are dead (no core) / frozen.
        let mut pw3 = PhysicsWorld::new(map.clone(), 1);
        pw3.add_tee(1, Vec2 { x: 300.0, y: 400.0 });
        let w3 = pw3.inner().clone();
        let view3 = WorldView {
            world: &w3,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        assert!(valid(&b.decide_in(&o1, Some(&view3))));
        let mut pw4 = PhysicsWorld::new(map.clone(), 1);
        pw4.add_tee(0, Vec2 { x: 100.0, y: 400.0 });
        pw4.add_tee(1, Vec2 { x: 300.0, y: 400.0 });
        let mut st = pw4.get_tee(0).unwrap();
        st.frozen = true;
        st.freeze_ticks_left = 100;
        pw4.apply_tee_state(0, &st);
        let w4 = pw4.inner().clone();
        let o4 = obs_for(&w4, &map, 0, &[0, 1], Some(1)).unwrap();
        let view4 = WorldView {
            world: &w4,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        assert!(valid(&b.decide_in(&o4, Some(&view4))));

        // Twelve tees, map corners and outside the map, huge velocities, lag with in-flight inputs.
        let mut pw5 = PhysicsWorld::new(map.clone(), 1);
        let spots = [
            (40.0, 40.0),
            (1500.0, 40.0),
            (40.0, 600.0),
            (1500.0, 600.0),
            (-500.0, 300.0),
            (4000.0, 300.0),
            (700.0, -400.0),
            (700.0, 3000.0),
            (200.0, 400.0),
            (260.0, 400.0),
            (320.0, 400.0),
            (380.0, 400.0),
        ];
        for (i, (x, y)) in spots.iter().enumerate() {
            pw5.add_tee(i as i32, Vec2 { x: *x, y: *y });
        }
        let mut st = pw5.get_tee(8).unwrap();
        st.vel = Vec2 { x: 900.0, y: -900.0 };
        pw5.apply_tee_state(8, &st);
        let w5 = pw5.inner().clone();
        let ids: Vec<i32> = (0..12).collect();
        for me in [0, 8, 10] {
            let o5 = obs_for(&w5, &map, me, &ids, Some(9)).unwrap();
            let inflight = [ddai_brain::Action::neutral().to_player_input(); 3];
            let view5 = WorldView {
                world: &w5,
                self_id: me,
                lag_ticks: 3,
                in_flight: &inflight,
            };
            assert!(valid(&b.decide_in(&o5, Some(&view5))));
        }
    }
}

/// One scene of the soak: `n` tees at random standing spots with random states.
fn random_scene(rng: &mut Rng, map: &Arc<MapData>) -> (PhysicsWorld, Vec<i32>) {
    // 2 to 15 tees: from a duel to a hub crowd (the local-tee selection keeps the cost of a crowd low).
    let n = 2 + (rng.next_float() * 14.0) as usize;
    let mut pw = PhysicsWorld::new(map.clone(), 1 + (rng.next_float() * 1000.0) as u64);
    let mut ids = Vec::new();
    for i in 0..n {
        let x = 5.0 + rng.next_float() * 40.0;
        let y = 4.0 + rng.next_float() * 9.0;
        pw.add_tee(
            i as i32,
            Vec2 {
                x: x * 32.0,
                y: y * 32.0,
            },
        );
        let mut st = pw.get_tee(i as i32).unwrap();
        st.vel = Vec2 {
            x: (rng.next_float() - 0.5) * 16.0,
            y: (rng.next_float() - 0.5) * 20.0,
        };
        if rng.next_float() < 0.15 && i > 0 {
            st.frozen = true;
            st.freeze_ticks_left = 150;
        }
        if rng.next_float() < 0.3 {
            st.jumped = 3;
            st.jumped_total = Some(2);
            st.jumps_left = 0;
        }
        pw.apply_tee_state(i as i32, &st);
        ids.push(i as i32);
    }
    (pw, ids)
}

/// Decides `decisions` decisions (the brain is tee 0 of a fresh random scene every few decisions;
/// the scene is advanced with every tee holding random inputs) and returns the largest work of one
/// decision in physics ticks.
/// Returns the largest work of one decision in tee-ticks: `(total, search)`, where `search` leaves out
/// the proposer's own work (the fly's ~1 ms is accounted separately from the 4 ms search and its
/// 15 ms extension cap, D-042).
fn soak(b: &mut HybridBrain, map: &Arc<MapData>, rng: &mut Rng, decisions: usize) -> (u64, u64) {
    let mut max_ticks = 0u64;
    let mut max_search = 0u64;
    let mut done = 0;
    while done < decisions {
        let (mut pw, ids) = random_scene(rng, map);
        if rng.next_float() < 0.25 {
            reset(b, map, (rng.next_float() * 1e6) as u64);
        }
        let mut last = empty_input();
        for _ in 0..(3 + (rng.next_float() * 25.0) as usize) {
            if done >= decisions {
                break;
            }
            let world = pw.inner().clone();
            let Some(o) = obs_for(&world, map, 0, &ids, None) else {
                break; // we died: next scene
            };
            let target = o
                .others
                .iter()
                .find(|c| !c.is_frozen)
                .or(o.others.first())
                .map(|c| c.id);
            let o = Observation { target_id: target, ..o };
            let view = WorldView {
                world: &world,
                self_id: 0,
                lag_ticks: (rng.next_float() * 3.0) as u32 % 3,
                in_flight: &[],
            };
            let view = WorldView {
                in_flight: &vec![ddai_brain::Action::neutral().to_player_input(); view.lag_ticks as usize],
                ..view
            };
            let a = b.decide_in(&o, Some(&view));
            assert!(valid(&a));
            done += 1;
            if let Some(t) = b.last_decision() {
                // Work in tee-ticks (a tick costs in proportion to the tees simulated: the local ones).
                let sim = u64::from(t.sim_tees.max(1));
                let tee_ticks = t.work.total_ticks() * sim;
                let search_ticks = (t.work.total_ticks() - t.work.proposal) * sim;
                max_search = max_search.max(search_ticks);
                if tee_ticks > max_ticks {
                    max_ticks = tee_ticks;
                    println!(
                        "  new max: {} ticks with {} tees: {:?} extended {} out_of_time {} shield_incomplete {}",
                        max_ticks,
                        ids.len(),
                        t.work,
                        t.extended,
                        t.out_of_time,
                        t.shield_incomplete
                    );
                }
            }
            last = input_from_action(&a, &last);
            for _ in 0..2 {
                pw.set_input(0, last);
                for &id in ids.iter().skip(1) {
                    let mut i = empty_input();
                    i.direction = ((rng.next_float() * 3.0) as i32 - 1).clamp(-1, 1);
                    i.jump = i32::from(rng.next_float() < 0.05);
                    i.hook = i32::from(rng.next_float() < 0.3);
                    i.target_x = (rng.next_float() - 0.5) * 600.0;
                    i.target_y = (rng.next_float() - 0.5) * 600.0;
                    pw.set_input(id, i);
                }
                pw.step();
            }
        }
    }
    (max_ticks, max_search)
}

fn soak_cfg() -> HybridConfig {
    // The 4 ms deadline on the work clock: reproducible, with the adaptive extension allowed.
    HybridConfig {
        mode: HybridMode::Deadline { budget_ms: 4.0 },
        work_clock_us_per_tick: Some(WORK_US_PER_TEE_TICK),
        ..HybridConfig::default()
    }
}

fn run_soak(total: usize, cfg: HybridConfig) {
    let map = hall();
    let mut rng = Rng::new(20260929);
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(ScriptedProposer::new())).expect("config");
    reset(&mut b, &map, 1);
    // Warm-up: caches (anchor tiles, buffers, hash maps) reach their working size.
    let warm = total / 3;
    let _ = soak(&mut b, &map, &mut rng, warm);
    let mut max_ticks = (0, 0);
    let info = measure(|| {
        max_ticks = soak(&mut b, &map, &mut rng, total - warm);
    });
    println!(
        "soak {} decisions after {} warm-up: net live heap growth {} bytes (peak {} bytes), max work of a decision {} tee-ticks, of which search without the proposer {} ({WORK_US_PER_TEE_TICK} us each)",
        total - warm,
        warm,
        info.bytes_current,
        info.bytes_max,
        max_ticks.0,
        max_ticks.1
    );
    // The scene-building garbage is freed; what the brain keeps (beliefs, anchor cache, buffers) is
    // bounded: far below one world per decision.
    assert!(
        info.bytes_current < 4 * 1024 * 1024,
        "live heap grew by {} bytes over the session",
        info.bytes_current
    );
    // The search (without the proposer, whose ~1 ms is separate) is bounded by the extension cap:
    // 15 ms on the work clock is `cap` tee-ticks (12 000 at the calibrated 1.25 us). The work clock advances
    // when a rollout finishes, so the last rollout can run past the cap: at most 27 ticks x 8 tees = 216
    // tee-ticks here.
    let cap = (15_000.0 / WORK_US_PER_TEE_TICK) as u64;
    assert!(
        max_ticks.1 < cap + 300,
        "one search simulated {} tee-ticks (cap {cap})",
        max_ticks.1
    );
    // With the proposer's own work on top (here 3 scripted rollouts, 81 ticks of the world).
    assert!(
        max_ticks.0 < cap + 1_200,
        "one decision simulated {} tee-ticks",
        max_ticks.0
    );
    let t = b.totals();
    assert!(t.decisions > 0);
    assert!(t.extended <= t.danger_flagged);
}

/// The deadline soak with the speed and shield options of 3.5b on: early pruning, dynamic stage 2, the
/// crowd stage, the shield counting a timeout as danger.
fn soak_cfg_options() -> HybridConfig {
    let mut c = soak_cfg();
    c.prune.enabled = true;
    c.stage2_dynamic = true;
    c.robust.crowd_stage = true;
    c.shield_timeout_danger = true;
    c.two_world = true;
    c
}

#[test]
fn soak_a_thousand_decisions_in_varied_scenes() {
    on_big_stack(|| run_soak(1_200, soak_cfg()));
}

#[test]
fn soak_with_the_options_of_3_5b() {
    on_big_stack(|| run_soak(1_200, soak_cfg_options()));
}

#[test]
#[ignore = "10 000 decisions: about a minute in release"]
fn soak_ten_thousand_decisions() {
    // `DDAI_SOAK_TOTAL` runs a longer session (the growth must stay flat, not scale with the length).
    let total = std::env::var("DDAI_SOAK_TOTAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    on_big_stack(move || run_soak(total, soak_cfg()));
}
