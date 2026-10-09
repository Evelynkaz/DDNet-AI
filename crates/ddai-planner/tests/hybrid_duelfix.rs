//! The duel fixes of task 3.23 (D-121, `HybridConfig::duel_fixes`) on the real hybrid brain: each one changes the decision where it should (a standing victim, a
//! frozen victim lying off the freeze, the victim's hook on us) and nothing anywhere else -- not with the switch off, not outside a duel.

use std::sync::Arc;

use ddai_brain::LiveContext;
use ddai_brain::{Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::World;
use ddai_planner::brains::ClockKind;
use ddai_planner::hybrid::{DecisionTelemetry, DuelFixConfig, HybridBrain, HybridConfig, HybridMode, NoProposer};
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

/// A hybrid brain as the arena builds it for a duel: the deadline mode on the work clock at `rate_us` microseconds per tee-tick (a starved search at 4 or more),
/// told by its caller whether a duel is on.
fn brain(rate_us: f64, fixes: DuelFixConfig, duel: bool) -> HybridBrain {
    let cfg = HybridConfig {
        mode: HybridMode::Deadline { budget_ms: 4.0 },
        work_clock_us_per_tick: Some(rate_us),
        proposals: 0,
        duel_fixes: fixes,
        ..HybridConfig::default()
    };
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    b.reset(&ResetContext {
        map: hall(),
        self_id: 0,
        seed: 3,
    });
    b.set_live_context(&LiveContext {
        duel,
        ..LiveContext::default()
    });
    b
}

/// `n` decisions in a row on the same (still) world, as a live bot decides every snapshot against a victim who does not move: the chosen first-three-step
/// plans, and the telemetry of the last decision.
/// Runs `f` on a thread with the stack the bot's decision thread has (64 MiB, `bot_cmd.rs` / `engine.rs`): a whole hybrid decision in a debug build
/// needs more than the default 2 MiB test thread has (it overflowed on Windows, STATUS_STACK_OVERFLOW). A panic in `f` is re-raised here, so the test
/// still fails with its own message.
fn with_big_stack<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|scope| {
        let handle = std::thread::Builder::new()
            .name("duelfix-test".into())
            .stack_size(64 << 20)
            .spawn_scoped(scope, f)
            .expect("spawn the test thread");
        match handle.join() {
            Ok(v) => v,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
}

fn decisions(b: &mut HybridBrain, pw: &PhysicsWorld, n: usize) -> (Vec<bool>, DecisionTelemetry) {
    let map = hall();
    let world = pw.inner().clone();
    let obs = observation(&world, &map, 0, &[0, 1], 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let mut idle = Vec::new();
    for _ in 0..n {
        let _ = b.decide_in(&obs, Some(&view));
        let t = b.last_decision().expect("telemetry");
        idle.push(ddai_planner::hybrid::duelfix::is_idle(&t.chosen_plan));
    }
    (idle, b.last_decision().expect("telemetry").clone())
}

fn all_on() -> DuelFixConfig {
    DuelFixConfig {
        static_push: true,
        counter_release: true,
        hooked_belief: 0.8,
        protect_defence: true,
        finish_push: true,
        finish_approach: 6,
        no_hammer_frozen: true,
        ..DuelFixConfig::default()
    }
}

/// Default identity against `main` rests on the goldens (`e010-golden`, `e020-golden-v2`, a live-view duel cell: bit-identical, E-038); this test is the
/// other half: with every fix **on** but no duel detected (`duel_only`), the decisions are those of the default brain.
#[test]
fn with_every_fix_on_but_no_duel_the_decisions_are_the_default_ones() {
    with_big_stack(|| {
        // A standing victim and a frozen one lying off the freeze: both situations of the fixes. Same seed, same world, the work clock: a pure function.
        let scenes = [scene(5.5, 8.5, 0), scene(5.5, 13.5, 150)];
        let plan = |t: &DecisionTelemetry| {
            t.chosen_plan
                .iter()
                .map(|s| (s.dir, s.jump, s.hook, s.fire))
                .collect::<Vec<_>>()
        };
        let mut differs = false;
        for pw in &scenes {
            let (base_idle, base) = decisions(&mut brain(8.0, DuelFixConfig::default(), true), pw, 40);
            let (idle_nd, nd) = decisions(&mut brain(8.0, all_on(), false), pw, 40);
            assert_eq!(idle_nd, base_idle, "outside a duel nothing changes");
            assert_eq!(plan(&nd), plan(&base));
            assert_eq!(nd.best_score.to_bits(), base.best_score.to_bits());
            assert!(!nd.static_push && !nd.finish_push);
            // In a duel the same fixes do change the decision (so the comparison above can fail).
            let (idle_duel, duel) = decisions(&mut brain(8.0, all_on(), true), pw, 40);
            assert!(duel.static_push || duel.finish_push, "the fixes act in a duel");
            differs |= idle_duel != base_idle || plan(&duel) != plan(&base);
        }
        assert!(differs, "in a duel the fixes change at least one of the two decisions");
    });
}

#[test]
fn a_standing_victim_is_not_answered_by_standing_still_with_the_fix() {
    with_big_stack(|| {
        let pw = scene(5.5, 8.5, 0);
        let on = DuelFixConfig {
            static_push: true,
            ..DuelFixConfig::default()
        };
        // The starved default stands still against him (the fixed point), or the test would prove nothing.
        let (base_idle, _) = decisions(&mut brain(8.0, DuelFixConfig::default(), true), &pw, 80);
        let stood = base_idle[30..].iter().filter(|&&i| i).count();
        assert!(stood >= 10, "the default idled {stood} of 50 decisions: {base_idle:?}");
        // Before the victim has been passive for `static_after` decisions the fix waits.
        let (idle, early) = decisions(&mut brain(8.0, on, true), &pw, 10);
        assert!(!early.static_push && idle.len() == 10);
        // After it, a decision is made among the plans that act: no idle choice once it is on (while a safe active plan exists).
        let (idle, last) = decisions(&mut brain(8.0, on, true), &pw, 80);
        assert!(last.static_push, "the victim has stood for 80 decisions");
        assert!(
            idle[30..].iter().all(|&i| !i),
            "no idle plan once the push is on: {idle:?}"
        );
    });
}

#[test]
fn a_frozen_victim_off_the_freeze_is_finished_with_the_fix_and_one_in_the_freeze_is_left_alone() {
    with_big_stack(|| {
        let on = DuelFixConfig {
            finish_push: true,
            finish_approach: 6,
            no_hammer_frozen: true,
            ..DuelFixConfig::default()
        };
        // Frozen on the floor 3 tiles away, 150 ticks left, off the pit: the push is on at the first decision.
        let (idle, t) = decisions(&mut brain(8.0, on, true), &scene(5.5, 8.5, 150), 1);
        assert!(t.finish_push && !t.static_push && !idle[0], "{:?}", t.chosen_plan);
        // In the pit (x 20..25): it stays frozen by itself, nothing is pushed.
        let mut pit = PhysicsWorld::new(hall(), 1);
        pit.add_tee(
            0,
            Vec2 {
                x: 14.5 * 32.0,
                y: 9.5 * 32.0,
            },
        );
        pit.add_tee(
            1,
            Vec2 {
                x: 22.5 * 32.0,
                y: 10.5 * 32.0,
            },
        );
        pit.inner_mut().characters[1].as_mut().expect("victim").freeze_time = 150;
        let (_, t) = decisions(&mut brain(8.0, on, true), &pit, 1);
        assert!(!t.finish_push, "a victim lying in the freeze is left alone");
        // About to thaw: nothing we walk to arrives in time.
        let (_, t) = decisions(&mut brain(8.0, on, true), &scene(5.5, 8.5, 20), 1);
        assert!(!t.finish_push);
    });
}

#[test]
fn the_counter_is_a_switch_of_the_robust_stage_and_changes_nothing_while_he_does_not_hook_us() {
    with_big_stack(|| {
        // Free victim in reach, no hook anywhere: the counter's floor and protection need his hook on us; the release model needs him below us while we rise.
        let pw = scene(5.5, 8.5, 0);
        let on = DuelFixConfig {
            counter_release: true,
            hooked_belief: 0.8,
            protect_defence: true,
            ..DuelFixConfig::default()
        };
        let (_, base) = decisions(&mut brain(1.25, DuelFixConfig::default(), true), &pw, 3);
        let (_, fixed) = decisions(&mut brain(1.25, on, true), &pw, 3);
        let plan = |t: &DecisionTelemetry| {
            t.chosen_plan
                .iter()
                .map(|s| (s.dir, s.jump, s.hook, s.fire))
                .collect::<Vec<_>>()
        };
        assert_eq!(plan(&base), plan(&fixed));
        assert_eq!(base.best_score.to_bits(), fixed.best_score.to_bits());
        // His hook on us: the robust stage believes he reacts (the floor of the belief) -- visible in the telemetry.
        let mut pw = scene(5.5, 8.5, 0);
        for _ in 0..12 {
            // He hooks us: aim at us, hook held (we are 96 px to his left).
            let mut him = ddai_planner::types::empty_input();
            him.hook = 1;
            him.target_x = -300.0;
            him.target_y = 0.0;
            pw.set_input(1, him);
            pw.set_input(0, ddai_planner::types::empty_input());
            pw.step();
        }
        let t1 = pw.get_tee(1).expect("his tee");
        assert_eq!(t1.hooked_player, 0, "his hook has grabbed us: {t1:?}");
        let (_, base) = decisions(&mut brain(1.25, DuelFixConfig::default(), true), &pw, 1);
        let (_, fixed) = decisions(&mut brain(1.25, on, true), &pw, 1);
        assert!(
            fixed.react_belief >= 0.8 - 1e-9 && fixed.react_belief > base.react_belief,
            "belief {} against {}",
            fixed.react_belief,
            base.react_belief
        );
    });
}
