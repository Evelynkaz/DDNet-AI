//! Task 3.15 (E-028): the lag-window model slot of the hybrid brain (`HybridConfig::window_model`).
//!
//! With the switch on, the roll of the planning world through the input-lag window plays the victim by the model's predictions; with the
//! switch off (the default) the model is never asked and the roll is the old "it keeps its input" one.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use ddai_brain::{Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_physics::core::PlayerInput as Wire;
use ddai_physics::map::{MapData, TILE_SOLID, Tile};
use ddai_physics::tuning::TuningParams;
use ddai_planner::brains::ClockKind;
use ddai_planner::hybrid::{HybridBrain, HybridConfig, NoProposer, PredictedInput, WindowCtx, WindowModel};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::vmath::Vec2;

fn hall() -> Arc<MapData> {
    let (w, h) = (40usize, 16usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            let solid = y >= 10 || x == 0 || x == w - 1 || y == 0;
            game[y * w + x] = Tile {
                index: if solid { TILE_SOLID } else { 0 },
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

fn world(map: &Arc<MapData>) -> ddai_physics::world::World<f32> {
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    for i in 0..2 {
        pw.add_tee(
            i,
            Vec2 {
                x: (17.5 + 4.0 * f64::from(i)) * 32.0,
                y: 9.5 * 32.0,
            },
        );
    }
    // A few ticks so both tees rest on the floor.
    for _ in 0..5 {
        pw.step();
    }
    pw.inner().clone()
}

fn obs(w: &ddai_physics::world::World<f32>, map: &Arc<MapData>) -> Observation {
    let ch = |id: i32| {
        let core = w.cores.get(id as u8).expect("tee");
        let mut c = CharacterObservation::at_rest(id);
        c.pos = core.pos;
        c.vel = core.vel;
        c
    };
    Observation {
        map: map.clone(),
        tick: w.tick,
        self_state: ch(0),
        others: vec![ch(1)],
        target_id: Some(1),
        tuning: TuningParams::default(),
    }
}

#[derive(Default)]
struct Probe {
    calls: Cell<u32>,
    windows: RefCell<Vec<usize>>,
    seen_ticks: RefCell<Vec<i32>>,
}

struct Stub {
    probe: Rc<Probe>,
    direction: i32,
}

impl WindowModel for Stub {
    fn name(&self) -> &str {
        "stub"
    }
    fn predict(&mut self, ctx: &WindowCtx<'_>, out: &mut [Option<PredictedInput>]) {
        self.probe.calls.set(self.probe.calls.get() + 1);
        self.probe.windows.borrow_mut().push(ctx.in_flight.len());
        self.probe.seen_ticks.borrow_mut().push(ctx.world.tick);
        assert_eq!(out.len(), ctx.in_flight.len());
        assert!(out.iter().all(Option::is_none), "the buffer arrives cleared");
        for o in out.iter_mut() {
            *o = Some(PredictedInput {
                direction: self.direction,
                jump: false,
                hook: false,
                press: false,
                aim: 0.0,
            });
        }
    }
    fn work_units(&self) -> u64 {
        7
    }
}

fn brain(window_model: bool, model: Option<Stub>) -> HybridBrain {
    let mut cfg = HybridConfig::fixed();
    cfg.workers = 1;
    cfg.window_model = window_model;
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    if let Some(m) = model {
        b.set_window_model(Box::new(m));
    }
    b.reset(&ResetContext {
        map: hall(),
        self_id: 0,
        seed: 3,
    });
    b
}

/// Decides once with a window of `lag` neutral in-flight inputs; returns the victim's core in the planning world afterwards.
fn decide(b: &mut HybridBrain, map: &Arc<MapData>, lag: usize) -> (f32, i32) {
    let w = world(map);
    let o = obs(&w, map);
    let in_flight = vec![Wire::default(); lag];
    let view = WorldView {
        world: &w,
        self_id: 0,
        lag_ticks: lag as u32,
        in_flight: &in_flight,
    };
    let _ = b.decide_in(&o, Some(&view));
    let core = b.debug_world().expect("search built").cores.get(1).expect("victim");
    (core.pos.x, core.direction)
}

#[test]
fn the_model_plays_the_victim_through_the_window() {
    let map = hall();
    let probe = Rc::new(Probe::default());
    let mut with = brain(
        true,
        Some(Stub {
            probe: probe.clone(),
            direction: 1,
        }),
    );
    let (x_model, dir_model) = decide(&mut with, &map, 3);
    let (x_hold, dir_hold) = decide(&mut brain(false, None), &map, 3);
    assert_eq!(dir_hold, 0, "without the model the victim keeps standing still");
    assert_eq!(dir_model, 1, "the last window tick applied the predicted direction");
    assert!(
        x_model > x_hold + 0.5,
        "the victim walked right through the window: {x_model} vs {x_hold}"
    );
    assert_eq!(probe.calls.get(), 1);
    assert_eq!(*probe.windows.borrow(), vec![3]);
}

#[test]
fn with_the_switch_off_the_model_is_never_asked_and_nothing_changes() {
    let map = hall();
    let probe = Rc::new(Probe::default());
    let mut b = brain(
        false,
        Some(Stub {
            probe: probe.clone(),
            direction: 1,
        }),
    );
    let off = decide(&mut b, &map, 3);
    let none = decide(&mut brain(false, None), &map, 3);
    assert_eq!(off.0.to_bits(), none.0.to_bits());
    assert_eq!(off.1, none.1);
    assert_eq!(probe.calls.get(), 0);
}

#[test]
fn the_model_sees_every_decision_even_without_a_window_and_its_cost_reaches_the_work_counters() {
    let map = hall();
    let probe = Rc::new(Probe::default());
    let mut b = brain(
        true,
        Some(Stub {
            probe: probe.clone(),
            direction: 1,
        }),
    );
    let (x0, _) = decide(&mut b, &map, 0);
    let (x_hold, _) = decide(&mut brain(false, None), &map, 0);
    assert_eq!(x0.to_bits(), x_hold.to_bits(), "no window, no change");
    assert_eq!(
        *probe.windows.borrow(),
        vec![0],
        "asked with an empty window so its history stays complete"
    );
    decide(&mut b, &map, 2);
    decide(&mut b, &map, 4);
    assert_eq!(*probe.windows.borrow(), vec![0, 2, 4]);
    assert_eq!(
        b.totals().work.units,
        3 * 7,
        "the three calls were charged, 7 tee-ticks each"
    );
}

#[test]
fn reset_reaches_the_model() {
    struct R(Rc<Cell<u32>>);
    impl WindowModel for R {
        fn name(&self) -> &str {
            "r"
        }
        fn reset(&mut self) {
            self.0.set(self.0.get() + 1);
        }
        fn predict(&mut self, _: &WindowCtx<'_>, _: &mut [Option<PredictedInput>]) {}
    }
    let n = Rc::new(Cell::new(0));
    let mut b = brain(true, None);
    b.set_window_model(Box::new(R(n.clone())));
    b.reset(&ResetContext {
        map: hall(),
        self_id: 0,
        seed: 4,
    });
    assert_eq!(n.get(), 1);
}
