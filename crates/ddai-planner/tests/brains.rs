//! `Brain` adapters (task 8.1): `ScriptedBrain` and `PlannerBrain` on the exact world of
//! `Brain::decide_in`, and the hammer-event derivation they and the arena rely on.

use std::sync::Arc;

use ddai_brain::{Action, Brain, CharacterObservation, IVec2, Observation, ResetContext, WorldView};
use ddai_jsmath::Rng;
use ddai_physics::map::{MapData, TILE_SOLID, Tile};
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::World;
use ddai_planner::brains::{
    ClockKind, PlannerBrain, PlannerBrainConfig, PlannerMode, PlannerPreset, ScriptedBrain, action_from_input,
    enemy_input_from_tee, input_from_action,
};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::scripted::scripted_action;
use ddai_planner::types::{WorldEvent, empty_input};
use ddai_planner::vmath::Vec2;

/// A 40x16 room with a solid floor (row 10 and below), tees stand on row 9.
fn room() -> Arc<MapData> {
    let (w, h) = (40usize, 16usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            if y >= 10 || x == 0 || x == w - 1 || y == 0 {
                game[y * w + x] = Tile {
                    index: TILE_SOLID,
                    ..Tile::default()
                };
            }
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

fn world_with(map: &Arc<MapData>, tees: &[(i32, f64, f64)]) -> World<f32> {
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    for &(id, tx, ty) in tees {
        pw.add_tee(
            id,
            Vec2 {
                x: tx * 32.0,
                y: ty * 32.0,
            },
        );
    }
    pw.inner().clone()
}

fn observation(world: &World<f32>, map: &Arc<MapData>, me: i32, target: i32) -> Observation {
    let ch = |id: i32| {
        let core = world.cores.get(id as u8).unwrap();
        let mut c = CharacterObservation::at_rest(id);
        c.pos = core.pos;
        c.vel = core.vel;
        c.hooked_player = core.hooked_player();
        c
    };
    Observation {
        map: map.clone(),
        tick: world.tick,
        self_state: ch(me),
        others: vec![ch(target)],
        target_id: Some(target),
        tuning: TuningParams::default(),
    }
}

fn view(world: &World<f32>, me: i32) -> WorldView<'_> {
    WorldView {
        world,
        self_id: me,
        lag_ticks: 0,
        in_flight: &[],
    }
}

fn reset(brain: &mut dyn Brain, map: &Arc<MapData>, id: i32, seed: u64) {
    brain.reset(&ResetContext {
        map: map.clone(),
        self_id: id,
        seed,
    });
}

#[test]
fn scripted_brain_equals_scripted_action_on_the_same_world() {
    let map = room();
    let world = world_with(&map, &[(0, 10.5, 9.5), (1, 16.5, 9.5)]);
    let seed = 42u64;
    let mut brain = ScriptedBrain::new();
    reset(&mut brain, &map, 0, seed);
    let obs = observation(&world, &map, 0, 1);

    // The reference: the ported function on a PhysicsWorld holding the same state, with the
    // harness's `Rng((seed * 7919 + 17) >>> 0)`.
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    pw.sync_from(&world);
    let mut rng = Rng::new((seed * 7919 + 17) as u32);
    let mut prev = empty_input();
    for step in 0..6 {
        let want = scripted_action(&pw, 0, 1, &prev, &mut rng);
        prev = want;
        let got = brain.decide_in(&obs, Some(&view(&world, 0)));
        assert_eq!(got, action_from_input(&want), "decision {step}");
    }
}

#[test]
fn scripted_brain_is_seeded_from_the_reset_context() {
    let map = room();
    let world = world_with(&map, &[(0, 10.5, 9.5), (1, 16.5, 9.5)]);
    let obs = observation(&world, &map, 0, 1);
    let run = |seed: u64| {
        let mut b = ScriptedBrain::new();
        reset(&mut b, &map, 0, seed);
        (0..8)
            .map(|_| b.decide_in(&obs, Some(&view(&world, 0))).target)
            .collect::<Vec<IVec2>>()
    };
    assert_eq!(run(5), run(5), "same seed, same aim noise");
    assert_ne!(run(5), run(6), "the aim noise depends on the seed");
}

#[test]
fn scripted_brain_without_a_view_matches_the_view_path() {
    let map = room();
    let world = world_with(&map, &[(0, 10.5, 9.5), (1, 16.5, 9.5)]);
    let obs = observation(&world, &map, 0, 1);
    let mut a = ScriptedBrain::new();
    let mut b = ScriptedBrain::new();
    reset(&mut a, &map, 0, 9);
    reset(&mut b, &map, 0, 9);
    for _ in 0..4 {
        assert_eq!(a.decide(&obs), b.decide_in(&obs, Some(&view(&world, 0))));
    }
}

#[test]
fn scripted_brain_with_no_target_releases_fire() {
    let map = room();
    let world = world_with(&map, &[(0, 10.5, 9.5)]);
    let mut obs = observation(&world, &map, 0, 0);
    obs.others.clear();
    obs.target_id = None;
    let mut b = ScriptedBrain::new();
    reset(&mut b, &map, 0, 1);
    let a = b.decide_in(&obs, Some(&view(&world, 0)));
    assert!(!a.fire && !a.hook && a.direction == 0);
}

fn fixed_planner() -> PlannerBrain {
    PlannerBrain::new(PlannerBrainConfig::default())
}

#[test]
fn planner_brain_fixed_mode_is_deterministic() {
    let map = room();
    let world = world_with(&map, &[(0, 10.5, 9.5), (1, 16.5, 9.5)]);
    let obs = observation(&world, &map, 0, 1);
    let run = || {
        let mut b = fixed_planner();
        reset(&mut b, &map, 0, 3);
        (0..3)
            .map(|_| b.decide_in(&obs, Some(&view(&world, 0))))
            .collect::<Vec<Action>>()
    };
    assert_eq!(run(), run());
    assert_eq!(fixed_planner().name(), "planner-normal-fixed");
}

#[test]
fn planner_brain_works_without_a_view_and_reports_telemetry() {
    let map = room();
    let world = world_with(&map, &[(0, 10.5, 9.5), (1, 16.5, 9.5)]);
    let obs = observation(&world, &map, 0, 1);
    let mut b = fixed_planner();
    reset(&mut b, &map, 0, 3);
    let _ = b.decide(&obs);
    let _ = b.decide_in(&obs, None);
    let tel: serde_json::Value = serde_json::from_str(&b.telemetry().unwrap()).unwrap();
    assert_eq!(tel["decisions"], 2);
    assert!(tel["candidates"].as_u64().unwrap() > 0);
    assert_eq!(b.stats().decisions, 2);
    // reset clears the counters
    reset(&mut b, &map, 0, 3);
    assert_eq!(b.stats().decisions, 0);
}

#[test]
fn planner_brain_never_modifies_the_callers_world() {
    let map = room();
    let world = world_with(&map, &[(0, 10.5, 9.5), (1, 16.5, 9.5)]);
    let before = format!("{:?}", world.cores.get(0).unwrap().pos) + &format!("{:?}", world.cores.get(1).unwrap().pos);
    let tick = world.tick;
    let obs = observation(&world, &map, 0, 1);
    let mut b = fixed_planner();
    reset(&mut b, &map, 0, 3);
    for _ in 0..2 {
        let _ = b.decide_in(&obs, Some(&view(&world, 0)));
    }
    assert_eq!(world.tick, tick);
    assert_eq!(
        before,
        format!("{:?}", world.cores.get(0).unwrap().pos) + &format!("{:?}", world.cores.get(1).unwrap().pos)
    );
}

#[test]
fn planner_brain_lag_rolls_the_world_forward_by_the_in_flight_inputs() {
    let map = room();
    let world = world_with(&map, &[(0, 10.5, 9.5), (1, 16.5, 9.5)]);
    let obs = observation(&world, &map, 0, 1);
    let mut walk_right = Action::neutral().to_player_input();
    walk_right.direction = 1;
    let in_flight = [walk_right, walk_right, walk_right];
    let lagged = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 3,
        in_flight: &in_flight,
    };
    let mut a = fixed_planner();
    let mut b = fixed_planner();
    reset(&mut a, &map, 0, 3);
    reset(&mut b, &map, 0, 3);
    // Both must produce a valid action; with three ticks of walking already committed the search
    // starts from a different state, so it may (and for this scene does) choose differently.
    let plain = a.decide_in(&obs, Some(&view(&world, 0)));
    let with_lag = b.decide_in(&obs, Some(&lagged));
    assert!((-1..=1).contains(&plain.direction) && (-1..=1).contains(&with_lag.direction));
    assert_eq!(b.stats().decisions, 1);
}

#[test]
fn deadline_mode_with_a_step_clock_is_deterministic_and_bounded() {
    let map = room();
    let world = world_with(&map, &[(0, 10.5, 9.5), (1, 16.5, 9.5)]);
    let obs = observation(&world, &map, 0, 1);
    let cfg = PlannerBrainConfig {
        preset: PlannerPreset::Normal,
        mode: PlannerMode::Deadline { budget_ms: 2.0 },
        clock: ClockKind::Step { step_ms: 0.05 },
    };
    let run = || {
        let mut b = PlannerBrain::new(cfg);
        reset(&mut b, &map, 0, 3);
        let out: Vec<Action> = (0..3).map(|_| b.decide_in(&obs, Some(&view(&world, 0)))).collect();
        (out, b.stats())
    };
    let (a, sa) = run();
    let (b, sb) = run();
    assert_eq!(a, b);
    assert_eq!(sa, sb);
    assert_eq!(sa.decisions, 3);
    assert_eq!(PlannerBrain::new(cfg).name(), "planner-normal-2ms");
    // A tighter budget searches less.
    let tight = PlannerBrainConfig {
        mode: PlannerMode::Deadline { budget_ms: 0.2 },
        ..cfg
    };
    let mut t = PlannerBrain::new(tight);
    reset(&mut t, &map, 0, 3);
    for _ in 0..3 {
        let _ = t.decide_in(&obs, Some(&view(&world, 0)));
    }
    assert!(
        t.stats().candidates < sa.candidates,
        "{} vs {}",
        t.stats().candidates,
        sa.candidates
    );
}

#[test]
fn presets_are_distinct_planner_configs() {
    let n = PlannerPreset::Normal.config();
    let l = PlannerPreset::Low.config();
    let s = PlannerPreset::Strong.config();
    assert!(l.budget_ms > 0.0 && n.budget_ms == 0.0);
    assert!(s.population > n.population);
}

#[test]
fn enemy_input_from_tee_reads_the_wire_angle() {
    let mut tee = ddai_planner::types::blank_tee_state();
    tee.direction = -1;
    tee.hook_state = 5;
    tee.angle = 0.0;
    let i = enemy_input_from_tee(&tee);
    assert_eq!((i.direction, i.hook, i.target_x, i.target_y), (-1, 1, 300.0, 0.0));
    tee.hook_state = 0;
    tee.angle = 256.0 * std::f64::consts::PI; // wire angle of pi: straight left
    let i = enemy_input_from_tee(&tee);
    assert_eq!(i.hook, 0);
    assert_eq!(i.target_x, -300.0);
    tee.angle = 256.0 * std::f64::consts::FRAC_PI_2; // straight down (screen y grows downward)
    let i = enemy_input_from_tee(&tee);
    assert_eq!((i.target_x, i.target_y), (0.0, 300.0));
}

#[test]
fn action_and_input_conversions_keep_the_fire_press_counter() {
    let mut a = Action::neutral();
    a.fire = true;
    a.wanted_weapon = Some(0);
    let mut prev = empty_input();
    let mut counters = Vec::new();
    for _ in 0..3 {
        prev = input_from_action(&a, &prev);
        counters.push(prev.fire);
        assert!(action_from_input(&prev).fire);
    }
    assert_eq!(counters, vec![1, 3, 5], "every decision is a fresh press");
    a.fire = false;
    prev = input_from_action(&a, &prev);
    assert_eq!(prev.fire, 6, "released: the counter goes even");
    prev = input_from_action(&a, &prev);
    assert_eq!(prev.fire, 6);
    assert_eq!(prev.wanted_weapon, 1, "weapon slot 0 travels as wire 1");
    assert_eq!(action_from_input(&prev).wanted_weapon, Some(0));
}

// ---------------------------------------------------------------------------------------------
// The hammer-event derivation of `PhysicsWorld::step` (the arena's source of hammer credit).

fn hammer_input(fire: i32, weapon_wire: i32) -> ddai_planner::types::PlayerInput {
    let mut i = empty_input();
    i.target_x = 100.0;
    i.target_y = 0.0;
    i.fire = fire;
    i.wanted_weapon = weapon_wire;
    i
}

/// Steps until `pred` matches an event of a step; returns (step index, events of that step).
fn step_until(
    pw: &mut PhysicsWorld,
    n: usize,
    mut input: impl FnMut(usize) -> ddai_planner::types::PlayerInput,
) -> Vec<(usize, Vec<WorldEvent>)> {
    let mut out = Vec::new();
    for t in 0..n {
        pw.set_input(0, input(t));
        let ev = pw.step();
        if !ev.is_empty() {
            out.push((t, ev));
        }
    }
    out
}

#[test]
fn a_hammer_hit_is_reported_with_its_victim_even_though_the_victim_flies_off() {
    let map = room();
    let mut pw = PhysicsWorld::new(map, 1);
    pw.add_tee(
        0,
        Vec2 {
            x: 17.0 * 32.0 + 16.0,
            y: 9.0 * 32.0 + 16.0,
        },
    );
    pw.add_tee(
        1,
        Vec2 {
            x: 18.0 * 32.0 + 16.0,
            y: 9.0 * 32.0 + 16.0,
        },
    );
    // Switch to the hammer (wire 1) on the first steps, swing from step 2. The first hit throws B
    // ~45 px away within the same step -- beyond the 42 px a post-step check would allow.
    let mut fire = 0;
    let events = step_until(&mut pw, 12, |t| {
        if t >= 2 && t % 2 == 0 {
            fire += 2 - (fire & 1);
        }
        hammer_input(fire, 1)
    });
    let hits: Vec<_> = events
        .iter()
        .flat_map(|(t, e)| e.iter().map(move |e| (*t, *e)))
        .filter(|(_, e)| matches!(e, WorldEvent::HammerHit { .. }))
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "exactly one blow lands (the reload blocks the next): {events:?}"
    );
    assert_eq!(hits[0].1, WorldEvent::HammerHit { from: 0, to: 1 });
    assert_eq!(hits[0].0, 2);
    assert!(
        events
            .iter()
            .any(|(_, e)| e.contains(&WorldEvent::HammerFire { from: 0, hits: 1 }))
    );
}

#[test]
fn a_hammer_swing_at_nobody_is_a_miss() {
    let map = room();
    let mut pw = PhysicsWorld::new(map, 1);
    pw.add_tee(
        0,
        Vec2 {
            x: 5.0 * 32.0 + 16.0,
            y: 9.0 * 32.0 + 16.0,
        },
    );
    pw.add_tee(
        1,
        Vec2 {
            x: 30.0 * 32.0 + 16.0,
            y: 9.0 * 32.0 + 16.0,
        },
    );
    let mut fire = 0;
    let events = step_until(&mut pw, 6, |t| {
        if t >= 2 && t % 2 == 0 {
            fire += 2 - (fire & 1);
        }
        hammer_input(fire, 1)
    });
    let all: Vec<_> = events.iter().flat_map(|(_, e)| e.clone()).collect();
    assert!(all.contains(&WorldEvent::HammerFire { from: 0, hits: 0 }), "{all:?}");
    assert!(
        !all.iter().any(|e| matches!(e, WorldEvent::HammerHit { .. })),
        "{all:?}"
    );
}

/// Fire with the gun in hand next to another tee: a bullet, not a hammer blow -- no hammer events.
/// (The weapon switch requested in the same input only takes effect one input later, as in DDNet.)
#[test]
fn gun_fire_is_never_reported_as_a_hammer_event() {
    let map = room();
    let mut pw = PhysicsWorld::new(map, 1);
    pw.add_tee(
        0,
        Vec2 {
            x: 17.0 * 32.0 + 16.0,
            y: 9.0 * 32.0 + 16.0,
        },
    );
    pw.add_tee(
        1,
        Vec2 {
            x: 18.0 * 32.0 + 16.0,
            y: 9.0 * 32.0 + 16.0,
        },
    );
    let events = step_until(
        &mut pw,
        3,
        |t| if t == 0 { hammer_input(0, 0) } else { hammer_input(1, 1) },
    );
    assert!(events.is_empty(), "the first shot at t=1 is a gun shot: {events:?}");
    assert_eq!(
        pw.inner().cores.get(0).unwrap().active_weapon,
        ddai_physics::core::WEAPON_GUN
    );
}

#[test]
fn one_blow_can_hit_two_tees_and_reports_both() {
    let map = room();
    let mut pw = PhysicsWorld::new(map, 1);
    // Aiming up-right at 45 degrees puts the swing point 21 px along the diagonal from the
    // swinger; B stands 36 px to its right and C 36 px above it (tees closer than 35 px shove each
    // other apart), both well inside the 42 px reach of the swing point and 51 px from each other.
    pw.add_tee(
        0,
        Vec2 {
            x: 17.0 * 32.0 + 16.0,
            y: 9.0 * 32.0 + 16.0,
        },
    );
    pw.add_tee(
        1,
        Vec2 {
            x: 17.0 * 32.0 + 16.0 + 36.0,
            y: 9.0 * 32.0 + 16.0,
        },
    );
    pw.add_tee(
        2,
        Vec2 {
            x: 17.0 * 32.0 + 16.0,
            y: 9.0 * 32.0 + 16.0 - 36.0,
        },
    );
    let mut fire = 0;
    let events = step_until(&mut pw, 6, |t| {
        if t >= 2 && t % 2 == 0 {
            fire += 2 - (fire & 1);
        }
        let mut i = hammer_input(fire, 1);
        i.target_y = -100.0;
        i
    });
    let all: Vec<_> = events.iter().flat_map(|(_, e)| e.clone()).collect();
    assert!(all.contains(&WorldEvent::HammerHit { from: 0, to: 1 }), "{all:?}");
    assert!(all.contains(&WorldEvent::HammerHit { from: 0, to: 2 }), "{all:?}");
    assert!(all.contains(&WorldEvent::HammerFire { from: 0, hits: 2 }), "{all:?}");
}

#[test]
fn sync_from_copies_the_world_and_the_held_inputs() {
    let map = room();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    pw.add_tee(
        0,
        Vec2 {
            x: 17.0 * 32.0 + 16.0,
            y: 9.0 * 32.0 + 16.0,
        },
    );
    pw.add_tee(
        1,
        Vec2 {
            x: 24.0 * 32.0 + 16.0,
            y: 9.0 * 32.0 + 16.0,
        },
    );
    let mut walk = empty_input();
    walk.direction = 1;
    pw.set_input(1, walk);
    for _ in 0..5 {
        pw.step();
    }
    let mut copy = PhysicsWorld::new(map, 1);
    copy.sync_from(pw.inner());
    assert_eq!(copy.tick(), pw.tick());
    for id in [0, 1] {
        assert_eq!(copy.get_tee(id).unwrap().pos, pw.get_tee(id).unwrap().pos);
    }
    // The copy continues exactly like the original (tee 1 keeps walking on its held input).
    for _ in 0..5 {
        pw.step();
        copy.step();
    }
    assert_eq!(copy.get_tee(1).unwrap().pos, pw.get_tee(1).unwrap().pos);
    assert!(copy.get_tee(1).unwrap().pos.x > 24.0 * 32.0 + 16.0 + 10.0);
}
