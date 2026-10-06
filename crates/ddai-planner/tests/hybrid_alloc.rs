//! Allocation behaviour of the hybrid search (task 3.5 criterion 4), measured with
//! `allocation_counter` (per-thread counting, as in the physics/fly/world no-alloc tests):
//!
//! 1. **Scoring is allocation-free.** After warm-up, `Engine::evaluate` -- the worker path that
//!    restores the decision snapshot into the worker's own `World` buffer (`restore_from`) and
//!    rolls candidates out on it, including the scripted-reaction model of extra threats --
//!    performs **zero** heap allocations. The helper threads run exactly this code (`Shared::run`
//!    / `Worker::eval`); the per-thread counter here runs it on the calling thread (`workers = 1`).
//! 2. **A whole decision never allocates a world-sized buffer.** The rest of a decision (candidate
//!    lists, the book, the technique plans, telemetry) does allocate small vectors on the
//!    deciding thread -- that count is measured and printed -- but no allocation of a `World`
//!    (~105 kB) or a snapshot happens in steady state: the peak live bytes during a decision stay
//!    far below one world.

use std::sync::Arc;

use allocation_counter::measure;
use ddai_brain::{Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
use ddai_physics::tuning::TuningParams;
use ddai_planner::brains::ClockKind;
use ddai_planner::clock::WallClock;
use ddai_planner::fields::{hazard_field, unfreeze_field};
use ddai_planner::hybrid::engine::{Batch, Ctx, Engine};
use ddai_planner::hybrid::threat::ThreatSet;
use ddai_planner::hybrid::{HybridBrain, HybridConfig, NoProposer, ScriptedProposer};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::planner::PlanStep;
use ddai_planner::types::empty_input;
use ddai_planner::vmath::Vec2;

fn hall() -> Arc<MapData> {
    let (w, h) = (40usize, 16usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            let solid = y >= 10 || x == 0 || x == w - 1 || y == 0;
            let freeze = (10..=11).contains(&y) && (20..=25).contains(&x);
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

fn scene(map: &Arc<MapData>, n: usize) -> PhysicsWorld {
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    for i in 0..n {
        pw.add_tee(
            i as i32,
            Vec2 {
                x: (17.5 + 3.0 * i as f64) * 32.0,
                y: 9.5 * 32.0,
            },
        );
    }
    pw
}

fn plans() -> Vec<Vec<PlanStep>> {
    (0..10)
        .map(|k| {
            (0..9)
                .map(|s| PlanStep {
                    dir: (k % 3) - 1,
                    jump: i32::from(s == 1 && k > 4),
                    hook: i32::from(k % 2 == 0 && s < 6),
                    fire: i32::from(s > 5),
                    aim: 0.15 * f64::from(k - 5),
                })
                .collect()
        })
        .collect()
}

#[test]
fn worker_scoring_is_allocation_free_in_steady_state() {
    let cfg = {
        let mut c = HybridConfig::fixed();
        c.workers = 1;
        c
    };
    scoring_is_allocation_free(cfg, false);
}

/// Task 3.10: the same with the finishing terms on (the frozen-victim drag shaping and the exact passive forecast at the end of every
/// rollout of a frozen victim) and the victim frozen on open ground, so the forecast runs in every rollout.
#[test]
fn finishing_terms_keep_worker_scoring_allocation_free() {
    let cfg = {
        let mut c = HybridConfig::fixed().with_finish();
        c.workers = 1;
        c.planner.held_forecast_weight = 20.0;
        c
    };
    scoring_is_allocation_free(cfg, true);
}

fn scoring_is_allocation_free(cfg: HybridConfig, frozen_victim: bool) {
    let map = hall();
    let mut pw = scene(&map, 4);
    if frozen_victim {
        let mut st = pw.get_tee(1).expect("victim");
        st.frozen = true;
        st.freeze_ticks_left = 150;
        pw.apply_tee_state(1, &st);
    }
    let ctx = Box::new(Ctx {
        generation: 1,
        saved: Box::new(pw.save_state()),
        self_id: 0,
        victim_id: 1,
        prev: empty_input(),
        victim_input: empty_input(),
        victim_plan: vec![],
        opp_seed: 7,
        field: Arc::new(hazard_field(pw.collision())),
        unfreeze: Arc::new(unfreeze_field(pw.collision())),
        frozen_bystanders: vec![],
        frozen_bystander_vels: vec![],
        // A spared tee and a travel goal, as the live bot passes them (they must not cost allocations either).
        spares: vec![Vec2 { x: 400.0, y: 300.0 }],
        spare_vels: vec![Vec2 { x: 0.0, y: 0.0 }],
        travel_goal: Some(Vec2 { x: 900.0, y: 300.0 }),
        threats: Some(ThreatSet {
            ids: vec![2, 3],
            inputs: vec![empty_input(), empty_input()],
            react_mask: 0,
            weight: 1.0,
            hook_targets: true,
        }),
        self_freeze_bias: 1.0,
    });
    let mut engine = Engine::new(&cfg, &pw, ctx, Arc::new(WallClock::new()));
    let mut batch = Batch::default();
    let mut out = Vec::new();
    let clock = WallClock::new();
    let all_plans = plans();
    let fill = |b: &mut Batch| {
        b.clear(9);
        for (i, p) in all_plans.iter().enumerate() {
            let pi = b.push_plan(p);
            // Every model combination: victim and both threats holding or reacting.
            b.push_job(pi, (i % 8) as u32);
            b.push_job(pi, 7);
        }
    };
    // Warm-up: buffers grow to their working size.
    for _ in 0..3 {
        fill(&mut batch);
        engine.evaluate(&mut batch, &clock, None, &mut out);
    }
    assert_eq!(out.len(), 20);
    // The forecast of a frozen victim is charged on top of the rollout's own 27 ticks.
    assert!(
        out.iter()
            .all(|o| o.res.is_some() && if frozen_victim { o.ticks >= 27 } else { o.ticks == 27 })
    );
    if frozen_victim {
        assert!(
            out.iter().any(|o| o.ticks > 27),
            "the forecast ran in at least one rollout"
        );
    }
    let info = measure(|| {
        for _ in 0..25 {
            fill(&mut batch);
            engine.evaluate(&mut batch, &clock, None, &mut out);
        }
    });
    println!("steady-state scoring: {info:?}");
    assert_eq!(
        info.count_total, 0,
        "scoring 25 batches of 20 rollouts allocated: {info:?}"
    );
    // A new decision (generation bump, same buffers) does not allocate either.
    engine.with_ctx(|_| ());
    fill(&mut batch);
    engine.evaluate(&mut batch, &clock, None, &mut out);
    let info = measure(|| {
        engine.with_ctx(|_| ());
        fill(&mut batch);
        engine.evaluate(&mut batch, &clock, None, &mut out);
    });
    assert_eq!(
        info.count_total, 0,
        "reloading the decision snapshot allocated: {info:?}"
    );
}

fn obs(w: &ddai_physics::world::World<f32>, map: &Arc<MapData>, ids: &[i32]) -> Observation {
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
        others: ids.iter().filter(|&&i| i != 0).map(|&i| ch(i)).collect(),
        target_id: Some(1),
        tuning: TuningParams::default(),
    }
}

#[test]
fn a_whole_decision_allocates_no_world_sized_buffer() {
    let map = hall();
    for (workers, proposer_scripted) in [(1usize, false), (2, false), (1, true)] {
        let pw = scene(&map, 4);
        let world = pw.inner().clone();
        let mut cfg = HybridConfig::fixed();
        cfg.workers = workers;
        cfg.proposals = if proposer_scripted { 3 } else { 0 };
        let proposer: Box<dyn ddai_planner::hybrid::Proposer> = if proposer_scripted {
            Box::new(ScriptedProposer::new())
        } else {
            Box::new(NoProposer)
        };
        let mut b = HybridBrain::new(cfg, ClockKind::Wall, proposer).unwrap();
        b.reset(&ResetContext {
            map: map.clone(),
            self_id: 0,
            seed: 3,
        });
        let o = obs(&world, &map, &[0, 1, 2, 3]);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        for _ in 0..12 {
            let _ = b.decide_in(&o, Some(&view));
        }
        let (mut max_bytes, mut max_count, mut total) = (0u64, 0u64, 0u64);
        let n = 30;
        for _ in 0..n {
            let info = measure(|| {
                let _ = b.decide_in(&o, Some(&view));
            });
            max_bytes = max_bytes.max(info.bytes_max);
            max_count = max_count.max(info.count_total);
            total += info.count_total;
        }
        println!(
            "workers={workers} scripted_proposer={proposer_scripted}: peak live bytes {max_bytes}, allocations per decision max {max_count}, mean {}",
            total / n
        );
        // The candidate pool (~60 plans with their score tables) is the largest live allocation;
        // one world or snapshot alone would be ~105 kB.
        assert!(
            max_bytes < 90 * 1024,
            "a decision held {max_bytes} bytes at once: a world (~105 kB) or snapshot was allocated"
        );
    }

    // Two tees and a victim that moves, so that the opponent model runs (a victim idle for six decisions is skipped, and four tees are no duel).
    let mut pw = scene(&map, 2);
    let mut inp = empty_input();
    inp.direction = -1;
    pw.set_input(1, inp);
    pw.step();
    let world = pw.inner().clone();
    let mut b = HybridBrain::new(
        {
            let mut cfg = HybridConfig::fixed();
            cfg.workers = 1;
            cfg.mirror = true;
            cfg
        },
        ClockKind::Wall,
        Box::new(NoProposer),
    )
    .unwrap();
    b.reset(&ResetContext {
        map: map.clone(),
        self_id: 0,
        seed: 3,
    });
    let o = obs(&world, &map, &[0, 1]);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    for _ in 0..12 {
        let _ = b.decide_in(&o, Some(&view));
    }
    let (mut max_bytes, mut ran) = (0u64, 0u32);
    for _ in 0..30 {
        let info = measure(|| {
            let _ = b.decide_in(&o, Some(&view));
        });
        ran += u32::from(b.last_decision().is_some_and(|t| t.work.mirror > 0));
        max_bytes = max_bytes.max(info.bytes_max);
    }
    assert!(ran > 0, "the opponent model never ran in the 2-tee case");
    assert!(
        max_bytes < 90 * 1024,
        "a decision with the opponent model held {max_bytes} bytes at once: a world (~105 kB) or snapshot was allocated"
    );
}
