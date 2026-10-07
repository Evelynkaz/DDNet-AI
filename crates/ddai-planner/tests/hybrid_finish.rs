//! Closing the escapes after a freeze (task 3.10b, D-110, E-030): the longer horizon while the victim is frozen (`HybridConfig::frozen_steps`, the port of
//! upstream's `frozenTargetSteps`) and the approach-then-push plans (`HybridConfig::approach_plans`, technique T30). Both are off by default; with them on
//! the work-clock decisions stay identical for any number of workers, and nothing changes while the victim is free.

use std::sync::Arc;

use ddai_brain::{Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::World;
use ddai_planner::brains::{ClockKind, input_from_action};
use ddai_planner::hybrid::search::Source;
use ddai_planner::hybrid::{DecisionTelemetry, HybridBrain, HybridConfig, HybridMode, NoProposer, Tech};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::vmath::Vec2;

/// 40x16 hall: floor from row 10, a freeze pit at x 20..=25 (rows 10..=11), a solid pillar on the left (x 1..=3, rows 1..=9).
fn hall() -> Arc<MapData> {
    let (w, h) = (40usize, 16usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            let solid = y >= 10 || x == 0 || x == w - 1 || y == 0 || (x <= 3 && y <= 9);
            game[y * w + x] = Tile {
                index: if solid { TILE_SOLID } else { 0 },
                ..Tile::default()
            };
        }
    }
    for y in 10..=11 {
        for x in 20..=25 {
            game[y * w + x] = Tile {
                index: TILE_FREEZE,
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

fn char_obs(w: &World<f32>, id: i32) -> CharacterObservation {
    let core = w.cores.get(id as u8).expect("tee exists");
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

fn observation(w: &World<f32>, map: &Arc<MapData>, me: i32, ids: &[i32], target: i32) -> Observation {
    Observation {
        map: map.clone(),
        tick: w.tick,
        self_state: char_obs(w, me),
        others: ids.iter().filter(|&&i| i != me).map(|&i| char_obs(w, i)).collect(),
        target_id: Some(target),
        tuning: TuningParams::default(),
    }
}

/// Us at tile (x0, 9.5), the victim at (x1, 9.5) on the floor; the pit is 20..25 tiles.
fn scene(x0: f64, x1: f64, victim_freeze: i32) -> PhysicsWorld {
    let map = hall();
    let mut pw = PhysicsWorld::new(map, 1);
    for (id, x) in [(0, x0), (1, x1)] {
        pw.add_tee(
            id,
            Vec2 {
                x: x * 32.0,
                y: 9.5 * 32.0,
            },
        );
    }
    if victim_freeze > 0 {
        pw.inner_mut().characters[1].as_mut().expect("victim").freeze_time = victim_freeze;
    }
    pw
}

fn decision(pw: &PhysicsWorld, tweak: impl FnOnce(&mut HybridConfig)) -> DecisionTelemetry {
    let mut cfg = HybridConfig::fixed();
    cfg.proposals = 0;
    cfg.debug_pool = true;
    tweak(&mut cfg);
    let map = hall();
    let world = pw.inner().clone();
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    b.reset(&ResetContext {
        map: map.clone(),
        self_id: 0,
        seed: 3,
    });
    let obs = observation(&world, &map, 0, &[0, 1], 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let _ = b.decide_in(&obs, Some(&view));
    b.last_decision().expect("telemetry").clone()
}

type Tweak = fn(&mut HybridConfig);

fn long(c: &mut HybridConfig) {
    c.frozen_steps = 16;
}

#[test]
fn every_switch_is_off_by_default_and_validated() {
    let c = HybridConfig::default();
    assert_eq!((c.frozen_steps, c.approach_plans), (0, 0));
    assert_eq!(c.frozen_steps_min_ticks, 30);
    assert_eq!(c.frozen_budget_ms, None);
    assert!(c.frozen_no_extension, "inert without frozen_steps");
    let bad_budget = HybridConfig {
        frozen_budget_ms: Some(0.0),
        ..HybridConfig::default()
    };
    assert!(bad_budget.validate().is_err());
    assert!(c.validate().is_ok());
    for bad in [1, 9, 33] {
        let c = HybridConfig {
            frozen_steps: bad,
            ..HybridConfig::default()
        };
        assert!(c.validate().is_err(), "frozen_steps {bad}");
    }
    let mut c = HybridConfig {
        frozen_steps: 16,
        approach_plans: 12,
        ..HybridConfig::default()
    };
    assert!(c.validate().is_ok());
    c.frozen_steps_min_ticks = -1;
    assert!(c.validate().is_err());
}

#[test]
fn the_longer_horizon_applies_only_to_a_frozen_victim_with_freeze_left_and_a_free_us() {
    let plan_lens = |t: &DecisionTelemetry| -> Vec<usize> { t.pool.iter().map(|p| p.plan.len()).collect() };
    // Frozen with 150 ticks left: every plan of the pool, and the chosen one, has 16 steps.
    let t = decision(&scene(5.5, 13.5, 150), long);
    assert!(!t.pool.is_empty());
    assert!(plan_lens(&t).iter().all(|&n| n == 16), "{:?}", plan_lens(&t));
    assert_eq!(t.chosen_plan.len(), 16);
    // The default hybrid plays 9 steps whatever the victim does.
    let t = decision(&scene(5.5, 13.5, 150), |_| {});
    assert!(plan_lens(&t).iter().all(|&n| n == 9));
    assert_eq!(t.chosen_plan.len(), 9);
    // Frozen with less than the 30 ticks left, or free: 9 steps.
    for freeze in [29, 0] {
        let t = decision(&scene(5.5, 13.5, freeze), long);
        assert!(plan_lens(&t).iter().all(|&n| n == 9), "freeze {freeze}");
        assert_eq!(t.chosen_plan.len(), 9);
    }
    // The threshold is the config's.
    let t = decision(&scene(5.5, 13.5, 29), |c| {
        long(c);
        c.frozen_steps_min_ticks = 20;
    });
    assert_eq!(t.chosen_plan.len(), 16);
    // We are frozen ourselves: the cheap search keeps 9.
    let mut pw = scene(5.5, 13.5, 150);
    pw.inner_mut().characters[0].as_mut().expect("us").freeze_time = 100;
    let t = decision(&pw, long);
    assert_eq!(t.chosen_plan.len(), 9);
}

#[test]
fn the_search_returns_to_nine_steps_when_the_victim_thaws_and_the_warm_plan_follows() {
    let map = hall();
    let mut cfg = HybridConfig::fixed();
    cfg.proposals = 0;
    cfg.debug_pool = true;
    long(&mut cfg);
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    b.reset(&ResetContext {
        map: map.clone(),
        self_id: 0,
        seed: 3,
    });
    let mut pw = scene(5.5, 13.5, 100);
    let mut lens = Vec::new();
    let mut prev = ddai_planner::types::empty_input();
    for _ in 0..60 {
        let world = pw.inner().clone();
        let obs = observation(&world, &map, 0, &[0, 1], 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        let a = b.decide_in(&obs, Some(&view));
        lens.push(b.last_decision().expect("telemetry").chosen_plan.len());
        // We stand still whatever it decides (the victim must thaw, not be hauled into the pit).
        let _ = input_from_action(&a, &prev);
        prev = ddai_planner::types::empty_input();
        for _ in 0..2 {
            pw.set_input(0, prev);
            pw.step();
        }
    }
    // 100 ticks of freeze: 16-step decisions while 30 or more ticks are left (the first 35 decisions), 9 after.
    assert_eq!(lens[0], 16);
    assert_eq!(*lens.last().unwrap(), 9);
    assert!(
        lens.windows(2).all(|w| !(w[0] == 9 && w[1] == 16)),
        "never back to 16: {lens:?}"
    );
    assert!(lens.iter().all(|&n| n == 9 || n == 16));
}

#[test]
fn decisions_with_the_new_switches_are_identical_for_1_and_4_workers() {
    for (name, tweak) in [
        (
            "long horizon",
            (|c: &mut HybridConfig| long(c)) as fn(&mut HybridConfig),
        ),
        ("approach", |c| c.approach_plans = 12),
        ("both", |c| {
            long(c);
            c.approach_plans = 12;
        }),
    ] {
        let run = |workers: usize| {
            let used = std::cell::Cell::new(0u32);
            let map = hall();
            let mut cfg = HybridConfig {
                mode: HybridMode::Deadline { budget_ms: 4.0 },
                proposals: 0,
                workers,
                work_clock_us_per_tick: Some(1.25),
                ..HybridConfig::default()
            };
            tweak(&mut cfg);
            let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
            b.reset(&ResetContext {
                map: map.clone(),
                self_id: 0,
                seed: 9,
            });
            let mut pw = scene(6.5, 14.5, 150);
            let mut prev = ddai_planner::types::empty_input();
            let mut log = Vec::new();
            for _ in 0..40 {
                let world = pw.inner().clone();
                let obs = observation(&world, &map, 0, &[0, 1], 1);
                let view = WorldView {
                    world: &world,
                    self_id: 0,
                    lag_ticks: 0,
                    in_flight: &[],
                };
                let a = b.decide_in(&obs, Some(&view));
                let t = b.last_decision().expect("telemetry");
                used.set(used.get() + t.spec_used);
                log.push((a, t.to_json()));
                prev = input_from_action(&a, &prev);
                for _ in 0..2 {
                    pw.set_input(0, prev);
                    pw.step();
                }
            }
            (log, used.get())
        };
        let (one, _) = run(1);
        let (four, used) = run(4);
        if let Some(k) = one.iter().zip(&four).position(|(a, b)| a != b) {
            panic!(
                "{name}: 4 workers decided differently from 1 at decision {k}:\n  1: {:?}\n  4: {:?}",
                one[k], four[k]
            );
        }
        assert!(
            used > 0,
            "{name}: the helpers supplied no rollouts, which proves nothing"
        );
    }
}

fn t30(t: &DecisionTelemetry) -> usize {
    t.pool.iter().filter(|p| p.src == Source::Tech(Tech::T30)).count()
}

#[test]
fn the_approach_family_joins_the_pool_only_for_a_frozen_victim_off_the_freeze() {
    // Off by default.
    assert_eq!(t30(&decision(&scene(6.5, 13.5, 150), |_| {})), 0);
    // On: a frozen victim 7 tiles away on the floor, the pit to its right.
    let on = |pw: &PhysicsWorld| decision(pw, |c| c.approach_plans = 12);
    let n = t30(&on(&scene(6.5, 13.5, 150)));
    assert!((1..=12).contains(&n), "{n} approach plans");
    // Not for a free victim, nor with too little freeze left, nor when we are frozen.
    assert_eq!(t30(&on(&scene(6.5, 13.5, 0))), 0);
    assert_eq!(t30(&on(&scene(6.5, 13.5, 20))), 0);
    let mut pw = scene(6.5, 13.5, 150);
    pw.inner_mut().characters[0].as_mut().expect("us").freeze_time = 100;
    assert_eq!(t30(&on(&pw)), 0);
    // Not when the victim lies on the freeze already (T7/T8 hold it there).
    assert_eq!(t30(&on(&scene(6.5, 22.5, 150))), 0);
    // The cap.
    let n3 = t30(&decision(&scene(6.5, 13.5, 150), |c| c.approach_plans = 3));
    assert!((1..=3).contains(&n3), "{n3}");
}

/// What a decision-by-decision play from a frozen victim on open floor ends in: `(sealed, ticks until it is in the freeze)`. The victim stays where the
/// physics leaves it; the hybrid plays (work clock) with `tweak`.
fn play_scene(x_us: f64, x_victim: f64, tweak: &dyn Fn(&mut HybridConfig), ticks: usize) -> (bool, Option<usize>) {
    let map = hall();
    let mut cfg = HybridConfig {
        mode: HybridMode::Deadline { budget_ms: 4.0 },
        proposals: 0,
        work_clock_us_per_tick: Some(1.25),
        ..HybridConfig::default()
    };
    tweak(&mut cfg);
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    b.reset(&ResetContext {
        map: map.clone(),
        self_id: 0,
        seed: 5,
    });
    let mut pw = scene(x_us, x_victim, 150);
    let mut prev = ddai_planner::types::empty_input();
    let mut in_freeze_at = None;
    for t in 0..ticks / 2 {
        let world = pw.inner().clone();
        let obs = observation(&world, &map, 0, &[0, 1], 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        let a = b.decide_in(&obs, Some(&view));
        prev = input_from_action(&a, &prev);
        for _ in 0..2 {
            pw.set_input(0, prev);
            pw.step();
        }
        let v = pw.get_tee(1).expect("victim");
        if in_freeze_at.is_none() && v.pos.x >= 20.0 * 32.0 && v.pos.x < 26.0 * 32.0 && v.pos.y > 9.9 * 32.0 {
            in_freeze_at = Some(2 * (t + 1));
        }
    }
    let v = pw.get_tee(1).expect("victim");
    (
        v.frozen && v.pos.x >= 19.5 * 32.0 && v.pos.x < 26.5 * 32.0,
        in_freeze_at,
    )
}

/// A scene report, not an assertion: from which starting places does the hybrid with each remedy haul a frozen victim into the freeze within 150
/// ticks (the freeze timer is 150: after it the victim walks off).
///
/// ```text
/// cargo test -p ddai-planner --release --test hybrid_finish -- --ignored --nocapture scene_report
/// ```
#[test]
#[ignore = "report"]
fn scene_report() {
    let arms: [(&str, Tweak); 5] = [
        ("off", |_| {}),
        ("long 16", |c| c.frozen_steps = 16),
        ("approach", |c| c.approach_plans = 12),
        ("both", |c| {
            c.frozen_steps = 16;
            c.approach_plans = 12;
        }),
        ("both + drag", |c| {
            c.frozen_steps = 16;
            c.approach_plans = 12;
            c.planner.frozen_drag_weight = 20.0;
        }),
    ];
    // Us left of the victim (victim between us and the pit), and us right of it (between the victim and the pit).
    let starts: Vec<(f64, f64)> = [
        (6.5, 12.5),
        (6.5, 14.5),
        (6.5, 16.5),
        (9.5, 14.5),
        (5.5, 10.5),
        (17.5, 12.5),
        (19.5, 14.5),
        (18.5, 15.5),
    ]
    .into_iter()
    .collect();
    println!(
        "\n| us / victim (tiles) | {} |\n|---|{}",
        arms.map(|a| a.0).join(" | "),
        "---|".repeat(arms.len())
    );
    let mut totals = vec![0; arms.len()];
    for (xu, xv) in &starts {
        let mut cells = Vec::new();
        for (k, (_, tweak)) in arms.iter().enumerate() {
            let (sealed, at) = play_scene(*xu, *xv, tweak, 150);
            totals[k] += usize::from(sealed);
            cells.push(format!(
                "{}{}",
                if sealed { "sealed" } else { "-" },
                at.map_or(String::new(), |t| format!(" @{t}"))
            ));
        }
        println!("| {xu} / {xv} | {} |", cells.join(" | "));
    }
    println!(
        "| sealed | {} |",
        totals.iter().map(ToString::to_string).collect::<Vec<_>>().join(" | ")
    );
}

fn clb_left() -> Option<Arc<MapData>> {
    let f = std::path::PathBuf::from(std::env::var("HOME").ok()?)
        .join("aiddnet/data/maps/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map");
    let m = ddai_map::load_map(&std::fs::read(f).ok()?).ok()?;
    Some(Arc::new(m.data))
}

/// Plays `ticks` ticks of a frozen victim on the floor of the Copy Love Box left hall (the E-030 diagnosis: it lies on the floor 3-8 tiles from the side
/// freeze, we stand beside it) and returns the tick at which it is hauled into a freeze again (its freeze timer jumps up), if it is.
fn play_clb(
    map: &Arc<MapData>,
    x_us: f64,
    x_victim: f64,
    tweak: &dyn Fn(&mut HybridConfig),
    ticks: usize,
) -> (Option<usize>, f64) {
    let mut cfg = HybridConfig {
        mode: HybridMode::Deadline { budget_ms: 4.0 },
        proposals: 0,
        work_clock_us_per_tick: Some(1.25),
        ..HybridConfig::default()
    };
    tweak(&mut cfg);
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    b.reset(&ResetContext {
        map: map.clone(),
        self_id: 0,
        seed: 5,
    });
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    for (id, x) in [(0, x_us), (1, x_victim)] {
        pw.add_tee(
            id,
            Vec2 {
                x: x * 32.0,
                y: 79.0 * 32.0,
            },
        );
    }
    for _ in 0..20 {
        pw.step();
    }
    pw.inner_mut().characters[1].as_mut().expect("victim").freeze_time = 150;
    let mut prev = ddai_planner::types::empty_input();
    let mut last_ft = 150;
    let mut sealed = None;
    for t in 0..ticks / 2 {
        let world = pw.inner().clone();
        let obs = observation(&world, map, 0, &[0, 1], 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        let a = b.decide_in(&obs, Some(&view));
        prev = input_from_action(&a, &prev);
        for k in 0..2 {
            pw.set_input(0, prev);
            pw.step();
            let ft = pw.inner().characters[1].as_ref().expect("victim").freeze_time;
            if ft > last_ft && sealed.is_none() {
                sealed = Some(2 * t + k + 1);
            }
            last_ft = ft;
        }
    }
    let v = pw.get_tee(1).expect("victim");
    (sealed, v.pos.x / 32.0)
}

/// Scene report on the real map (needs the Copy Love Box file): not an assertion.
///
/// ```text
/// cargo test -p ddai-planner --release --test hybrid_finish -- --ignored --nocapture clb_scene_report
/// ```
#[test]
#[ignore = "report; needs the Copy Love Box map"]
fn clb_scene_report() {
    let Some(map) = clb_left() else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    let arms: [(&str, Tweak); 4] = [
        ("off", |_| {}),
        ("long 16", |c| c.frozen_steps = 16),
        ("approach", |c| c.approach_plans = 12),
        ("both", |c| {
            c.frozen_steps = 16;
            c.approach_plans = 12;
        }),
    ];
    let starts = [
        (89.5, 87.5),
        (91.5, 87.5),
        (86.5, 87.5),
        (84.5, 87.5),
        (92.5, 89.5),
        (85.5, 83.5),
        (82.5, 85.5),
        (90.5, 86.5),
    ];
    println!(
        "\n| us / victim (tiles) | {} |\n|---|{}",
        arms.map(|a| a.0).join(" | "),
        "---|".repeat(arms.len())
    );
    let mut totals = vec![0; arms.len()];
    for (xu, xv) in starts {
        let mut cells = Vec::new();
        for (k, (_, tweak)) in arms.iter().enumerate() {
            let (sealed, end_x) = play_clb(&map, xu, xv, tweak, 150);
            totals[k] += usize::from(sealed.is_some());
            cells.push(format!(
                "{} (x {end_x:.0})",
                sealed.map_or("-".to_string(), |t| format!("sealed @{t}"))
            ));
        }
        println!("| {xu} / {xv} | {} |", cells.join(" | "));
    }
    println!(
        "| sealed | {} |",
        totals.iter().map(ToString::to_string).collect::<Vec<_>>().join(" | ")
    );
}
