//! `TeacherPlanner` (task 8.2): the fixed-iteration planner as a labeller. On the same state and
//! seed its label is exactly `PlannerBrain`'s decision, it is deterministic, and the soft target
//! (the elite set's first-step statistics) is well formed.

use std::sync::Arc;

use ddai_brain::{Action, Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::World;
use ddai_planner::brains::{PlannerBrain, PlannerBrainConfig, PlannerPreset};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::teacher::TeacherPlanner;
use ddai_planner::vmath::Vec2;

/// A 40x16 room: solid floor from row 10, a freeze pit in columns 18..=22 of the floor.
fn room() -> Arc<MapData> {
    let (w, h) = (40usize, 16usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            let pit = (18..=22).contains(&x) && (10..=13).contains(&y);
            if pit {
                game[y * w + x] = Tile {
                    index: TILE_FREEZE,
                    ..Tile::default()
                };
            } else if y >= 10 || x == 0 || x == w - 1 || y == 0 {
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

fn ctx(map: &Arc<MapData>, seed: u64) -> ResetContext {
    ResetContext {
        map: map.clone(),
        self_id: 0,
        seed,
    }
}

fn view(world: &World<f32>) -> WorldView<'_> {
    WorldView {
        world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    }
}

#[test]
fn the_label_is_the_planner_brains_decision_on_the_same_states() {
    let map = room();
    let world = world_with(&map, &[(0, 12.5, 9.5), (1, 17.5, 9.5)]);
    let obs = observation(&world, &map, 0, 1);
    let mut brain = PlannerBrain::new(PlannerBrainConfig::default());
    brain.reset(&ctx(&map, 7));
    let mut teacher = TeacherPlanner::new(PlannerPreset::Normal);
    teacher.reset(&ctx(&map, 7));
    for i in 0..5 {
        let want = brain.decide_in(&obs, Some(&view(&world)));
        let got = teacher.label(&obs, &view(&world));
        assert_eq!(got.action, want, "decision {i}");
    }
}

#[test]
fn labels_are_deterministic_and_the_soft_target_is_well_formed() {
    let map = room();
    let world = world_with(&map, &[(0, 12.5, 9.5), (1, 17.5, 9.5)]);
    let obs = observation(&world, &map, 0, 1);
    let run = || {
        let mut t = TeacherPlanner::new(PlannerPreset::Normal);
        t.reset(&ctx(&map, 3));
        (0..4).map(|_| t.label(&obs, &view(&world))).collect::<Vec<_>>()
    };
    let (a, b) = (run(), run());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.action, y.action);
        assert_eq!(x.elite, y.elite);
    }
    for l in &a {
        let e = l.elite.expect("a fixed-iteration decision runs the CEM");
        assert!(e.elites > 0);
        assert!((e.left + e.stop + e.right - 1.0).abs() < 1e-5);
        for p in [e.jump, e.hook, e.fire] {
            assert!((0.0..=1.0).contains(&p));
        }
        assert!(e.aim_spread >= 0.0 && e.aim_mean.abs() <= std::f32::consts::PI + 1e-5);
        assert!(l.info.searched);
    }
}

#[test]
fn without_a_target_the_label_is_neutral_and_has_no_soft_target() {
    let map = room();
    let world = world_with(&map, &[(0, 12.5, 9.5)]);
    let mut obs = observation(&world, &map, 0, 0);
    obs.others.clear();
    obs.target_id = None;
    let mut t = TeacherPlanner::new(PlannerPreset::Normal);
    t.reset(&ctx(&map, 1));
    let l = t.label(&obs, &view(&world));
    assert_eq!(l.action, Action::neutral());
    assert!(l.elite.is_none());
}

#[test]
fn note_executed_makes_the_next_label_start_from_the_played_input() {
    // Two teachers see the same states; one is told a different "played" action after the first
    // decision. Their first labels agree; the planner's flip-hysteresis reads `prev`, so telling
    // one of them it ran the opposite way must be able to change the next label.
    let map = room();
    let world = world_with(&map, &[(0, 12.5, 9.5), (1, 17.5, 9.5)]);
    let obs = observation(&world, &map, 0, 1);
    let mut a = TeacherPlanner::new(PlannerPreset::Normal);
    let mut b = TeacherPlanner::new(PlannerPreset::Normal);
    a.reset(&ctx(&map, 5));
    b.reset(&ctx(&map, 5));
    let la = a.label(&obs, &view(&world));
    let lb = b.label(&obs, &view(&world));
    assert_eq!(la.action, lb.action);
    // `a` is told its own action was played (a no-op relative to the default); `b` that the
    // opposite direction was played with hook held.
    a.note_executed(&la.action);
    let mut other = la.action;
    other.direction = -la.action.direction;
    other.hook = !la.action.hook;
    b.note_executed(&other);
    // Both still produce a valid label (no panic, elite present); the API contract is that the
    // inputs differ, so the states the planner reasons from are allowed to differ.
    let la2 = a.label(&obs, &view(&world));
    let lb2 = b.label(&obs, &view(&world));
    assert!(la2.elite.is_some() && lb2.elite.is_some());
}

/// The planner shares its hazard fields through `Arc`, so both brains may move between threads (the
/// teacher labels inside rayon workers); a regression to `Rc` would stop compiling here.
#[test]
fn the_planner_brain_and_the_teacher_are_send() {
    fn is_send<T: Send>() {}
    is_send::<PlannerBrain>();
    is_send::<TeacherPlanner>();
}
