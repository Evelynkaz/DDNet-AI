//! The hybrid brain (task 3.5): determinism across worker counts in fixed-work mode, the
//! guarantees of the decision (no clock in fixed mode, the shield always runs, proposals are only
//! candidates, the caller's world is never touched), the 1vN threat set, techniques named in
//! telemetry, the adaptive extension invariants, and the allocation behaviour of the workers.

use std::sync::Arc;

use ddai_brain::{Action, Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::World;
use ddai_planner::brains::{ClockKind, ScriptedBrain, input_from_action};
use ddai_planner::clock::{Clock, WallClock};
use ddai_planner::hybrid::config::AdaptiveConfig;
use ddai_planner::hybrid::search::{DecisionInput, HybridSearch};
use ddai_planner::hybrid::{
    HybridBrain, HybridConfig, HybridMode, NoProposer, ProposalOutcome, ProposeCtx, Proposer, ScriptedProposer,
};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::planner::PlanStep;
use ddai_planner::types::{PlayerInput, empty_input};
use ddai_planner::vmath::Vec2;

/// 40x16 hall: floor from row 10 down, a freeze pit in the floor at x 20..=25 (two rows deep), a
/// solid pillar on the left (x 1..=3, rows 1..=9). Tees stand on row 9.
fn hall() -> Arc<MapData> {
    let (w, h) = (40usize, 16usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            let idx = if y >= 10 || x == 0 || x == w - 1 || y == 0 || (x <= 3 && y <= 9) {
                TILE_SOLID
            } else {
                0
            };
            game[y * w + x] = Tile {
                index: idx,
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

fn place(pw: &mut PhysicsWorld, tees: &[(i32, f64, f64)]) {
    for &(id, tx, ty) in tees {
        pw.add_tee(
            id,
            Vec2 {
                x: tx * 32.0,
                y: ty * 32.0,
            },
        );
    }
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

fn reset(b: &mut dyn Brain, map: &Arc<MapData>, id: i32, seed: u64) {
    b.reset(&ResetContext {
        map: map.clone(),
        self_id: id,
        seed,
    });
}

fn fixed_cfg(workers: usize) -> HybridConfig {
    let mut c = HybridConfig::fixed();
    c.proposals = 0;
    c.workers = workers;
    c
}

/// Plays `decisions` decisions (every 2 ticks) of slot 0 = the hybrid brain against scripted
/// attackers and returns what the brain did and how (its telemetry per decision).
fn drive(cfg: HybridConfig, clock: ClockKind, tees: &[(i32, f64, f64)], decisions: usize) -> Vec<(Action, String)> {
    drive_with(cfg, clock, tees, decisions, |t| {
        format!(
            "{}|{}|{:?}|{:?}|{}|{}|{}",
            t.chosen.map_or("none", |c| c.label()),
            t.threat_ids.len(),
            t.evaluated,
            t.generated,
            t.work.total_ticks(),
            t.shielded,
            t.best_score.to_bits()
        )
    })
}

/// [`drive`] with the per-decision record chosen by the caller.
fn drive_with(
    cfg: HybridConfig,
    clock: ClockKind,
    tees: &[(i32, f64, f64)],
    decisions: usize,
    record: impl Fn(&ddai_planner::hybrid::DecisionTelemetry) -> String,
) -> Vec<(Action, String)> {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, tees);
    let ids: Vec<i32> = tees.iter().map(|t| t.0).collect();
    let mut hybrid = HybridBrain::new(cfg, clock, Box::new(NoProposer)).expect("valid config");
    reset(&mut hybrid, &map, 0, 7);
    let mut scripted: Vec<ScriptedBrain> = ids.iter().skip(1).map(|_| ScriptedBrain::new()).collect();
    for (k, b) in scripted.iter_mut().enumerate() {
        reset(b, &map, (k + 1) as i32, 100 + k as u64);
    }
    let mut last: Vec<PlayerInput> = ids.iter().map(|_| empty_input()).collect();
    let mut log = Vec::new();
    for _ in 0..decisions {
        let world = pw.inner().clone();
        for (slot, &id) in ids.iter().enumerate() {
            let alive = world.cores.get(id as u8).is_some();
            if !alive {
                continue;
            }
            let target = if slot == 0 {
                *ids.get(1).expect("an opponent")
            } else {
                0
            };
            let obs = observation(&world, &map, id, &ids, target);
            let view = WorldView {
                world: &world,
                self_id: id,
                lag_ticks: 0,
                in_flight: &[],
            };
            let action = if slot == 0 {
                let a = hybrid.decide_in(&obs, Some(&view));
                let tel = hybrid.last_decision().map(&record);
                log.push((a, tel.unwrap_or_default()));
                a
            } else {
                scripted[slot - 1].decide_in(&obs, Some(&view))
            };
            last[slot] = input_from_action(&action, &last[slot]);
        }
        for _ in 0..2 {
            for (slot, &id) in ids.iter().enumerate() {
                pw.set_input(id, last[slot]);
            }
            pw.step();
        }
    }
    log
}

const FOUR: [(i32, f64, f64); 4] = [(0, 17.5, 9.5), (1, 24.5, 9.5), (2, 27.5, 9.5), (3, 30.5, 9.5)];

#[test]
fn fixed_mode_decisions_are_identical_for_1_2_and_4_workers() {
    let run = |w: usize| drive(fixed_cfg(w), ClockKind::Wall, &FOUR, 24);
    let one = run(1);
    assert_eq!(one.len(), 24);
    for w in [2usize, 4] {
        let many = run(w);
        assert_eq!(one, many, "workers = {w} decided differently from workers = 1");
    }
    // ... and the same configuration twice is identical (nothing reads the clock or a global RNG).
    assert_eq!(one, run(1));
}

#[test]
fn fixed_mode_with_pruning_and_the_crowd_stage_is_identical_for_1_2_and_4_workers() {
    // The 3.5b options (early pruning, the crowd stage, hook escapes in the shield) must keep the
    // pool's merge-by-index determinism: pruning gates on results in candidate order, never on time.
    let cfg = |w: usize| {
        let mut c = fixed_cfg(w);
        c.prune.enabled = true;
        c.robust.crowd_stage = true;
        c.shield_hook_anchors = 3;
        c
    };
    let run = |w: usize| drive(cfg(w), ClockKind::Wall, &FOUR, 24);
    let one = run(1);
    for w in [2usize, 4] {
        assert_eq!(one, run(w), "workers = {w} decided differently from workers = 1");
    }
    // ... and pruning really skipped something in this fight (else the test proves nothing).
    let pruned = {
        let map = hall();
        let mut pw = PhysicsWorld::new(map.clone(), 1);
        place(&mut pw, &FOUR);
        let world = pw.inner().clone();
        let mut b = HybridBrain::new(cfg(1), ClockKind::Wall, Box::new(NoProposer)).unwrap();
        reset(&mut b, &map, 0, 7);
        let obs = observation(&world, &map, 0, &[0, 1, 2, 3], 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        let _ = b.decide_in(&obs, Some(&view));
        b.last_decision().unwrap().pruned
    };
    assert!(pruned > 0, "early pruning skipped nothing");
}

#[test]
fn an_idle_crowd_outside_the_threat_radius_is_not_simulated() {
    // Us, the victim and six bystanders 30+ tiles away: only the two fighters are in the rollouts
    // (F2), the budget keeps its 4 ms, and with `max_sim_tees = 0` (the 3.5 behaviour) all eight are.
    let map = hall();
    let tees: Vec<(i32, f64, f64)> = (0..8)
        .map(|i| match i {
            0 => (0, 5.5, 9.5),
            1 => (1, 9.5, 9.5),
            k => (k, 28.0 + f64::from(k), 9.5),
        })
        .collect();
    let ids: Vec<i32> = tees.iter().map(|t| t.0).collect();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &tees);
    let world = pw.inner().clone();
    let obs = observation(&world, &map, 0, &ids, 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let run = |cap: usize| {
        let cfg = HybridConfig {
            proposals: 0,
            max_sim_tees: cap,
            // The step clock charges every read; the budget arithmetic of this test is about the crowd, not the opponent model.
            mirror: false,
            ..HybridConfig::default()
        };
        let mut b = HybridBrain::new(cfg, ClockKind::Step { step_ms: 0.01 }, Box::new(NoProposer)).unwrap();
        reset(&mut b, &map, 0, 3);
        let _ = b.decide_in(&obs, Some(&view));
        let t = b.last_decision().unwrap();
        (t.sim_tees, t.dropped_tees, t.budget_ms)
    };
    assert_eq!(run(6), (2, 6, 4.0));
    let all = run(0);
    assert_eq!((all.0, all.1), (8, 0));
    assert!(
        all.2 < 4.0,
        "eight tees shrink the search budget under the cap: {}",
        all.2
    );
    // The caller's world is untouched.
    assert_eq!(world.cores.len(), 8);
}

#[test]
fn the_nearest_threats_stay_in_the_simulation_up_to_the_cap() {
    // Six free tees inside the radius: a cap of 4 keeps us, the victim and the two nearest threats.
    let map = hall();
    let tees = [
        (0, 10.5, 9.5),
        (1, 12.5, 9.5),
        (2, 14.0, 9.5),
        (3, 15.0, 9.5),
        (4, 16.0, 9.5),
        (5, 17.0, 9.5),
        (6, 18.0, 9.5),
    ];
    let ids: Vec<i32> = tees.iter().map(|t| t.0).collect();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &tees);
    let world = pw.inner().clone();
    let obs = observation(&world, &map, 0, &ids, 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let cfg = HybridConfig {
        proposals: 0,
        max_sim_tees: 4,
        ..HybridConfig::default()
    };
    let mut b = HybridBrain::new(cfg, ClockKind::Step { step_ms: 0.01 }, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
    let _ = b.decide_in(&obs, Some(&view));
    let t = b.last_decision().unwrap();
    assert_eq!((t.sim_tees, t.dropped_tees), (4, 3));
    assert_eq!(t.threat_ids, vec![2, 3], "the two nearest free tees");
}

#[test]
fn the_crowd_stage_rescoring_uses_two_combinations_for_three_or_more_opponents() {
    let map = hall();
    let tees = [
        (0, 10.5, 9.5),
        (1, 13.5, 9.5),
        (2, 14.5, 9.5),
        (3, 15.5, 9.5),
        (4, 16.5, 9.5),
    ];
    let ids: Vec<i32> = tees.iter().map(|t| t.0).collect();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &tees);
    let world = pw.inner().clone();
    let obs = observation(&world, &map, 0, &ids, 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let combos = |crowd: bool| {
        let mut cfg = fixed_cfg(1);
        cfg.robust.crowd_stage = crowd;
        let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
        reset(&mut b, &map, 0, 3);
        let _ = b.decide_in(&obs, Some(&view));
        b.last_decision().unwrap().combos
    };
    assert_eq!(
        combos(false),
        1,
        "3.5: four opponents are decided by the cheap model alone"
    );
    assert_eq!(combos(true), 2, "everybody holds / everybody reacts");
}

#[test]
fn the_two_world_search_is_deterministic_across_workers_and_inert_without_threats() {
    // The pool is scored with us and the victim only, the best few re-scored with the threats (task 3.5b).
    let cfg = |w: usize, two: bool| {
        let mut c = fixed_cfg(w);
        c.two_world = two;
        c
    };
    // A 1v3: identical for 1, 2 and 4 workers (the lens switches go through the same merge-by-index).
    let one = drive(cfg(1, true), ClockKind::Wall, &FOUR, 24);
    assert_eq!(one.len(), 24);
    for w in [2usize, 4] {
        assert_eq!(one, drive(cfg(w, true), ClockKind::Wall, &FOUR, 24), "workers = {w}");
    }
    // ... and it really searches differently from the single-world one when there are threats.
    assert_ne!(one, drive(cfg(1, false), ClockKind::Wall, &FOUR, 24));
    // A 1v1 has no threats: the option changes nothing.
    let duel = [(0, 17.5, 9.5), (1, 24.5, 9.5)];
    assert_eq!(
        drive(cfg(1, true), ClockKind::Wall, &duel, 16),
        drive(cfg(1, false), ClockKind::Wall, &duel, 16)
    );
}

#[test]
fn two_world_decisions_are_complete_and_safe_choices_survive_the_threats() {
    // In the two-world search the chosen plan has been scored in the full world under every combination, so
    // its safety flag means what it says; and the 1v5 hostile inputs of the soak run it too.
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &FOUR);
    let world = pw.inner().clone();
    let mut cfg = fixed_cfg(1);
    cfg.two_world = true;
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
    let obs = observation(&world, &map, 0, &[0, 1, 2, 3], 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    for _ in 0..4 {
        let _ = b.decide_in(&obs, Some(&view));
        let t = b.last_decision().unwrap();
        assert!(t.chosen.is_some());
        assert!(t.combos >= 1);
        assert!(t.work.rollouts_stage2 > 0, "no full-world re-scoring");
        assert_eq!(t.sim_tees, 4, "the shield and the re-scoring see all four tees");
    }
}

#[test]
fn fixed_mode_reads_no_clock_and_the_shield_still_runs() {
    struct NoClock;
    impl Clock for NoClock {
        fn now_ms(&self) -> f64 {
            panic!("fixed mode must not read a clock");
        }
    }
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &[(0, 19.5, 9.5), (1, 27.5, 9.5)]);
    let mut search = HybridSearch::new(
        fixed_cfg(1),
        Box::new(NoProposer),
        pw.clone_for_test(),
        Arc::new(WallClock::new()),
    );
    let world = pw.inner().clone();
    search.world_mut().sync_from(&world);
    let obs = observation(&world, &map, 0, &[0, 1], 1);
    let (out, tel) = search.decide(
        &NoClock,
        &DecisionInput {
            obs: &obs,
            self_id: 0,
            victim_id: 1,
            prev: empty_input(),
            lag_ticks: 0,
            roll_ticks: 0,
            deadline_ms: None,
            duel: false,
        },
    );
    assert!((-1..=1).contains(&out.direction));
    assert!(tel.work.stage1 > 0 && tel.work.stage2 > 0, "{:?}", tel.work);
    assert!(
        tel.work.shield > 0,
        "the shield must run on every decision: {:?}",
        tel.work
    );
    assert_eq!(tel.search_ms, 0.0);
    assert!(!tel.extended && tel.budget_ms == 0.0);
}

#[test]
fn the_callers_world_is_never_modified() {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &FOUR);
    let world = pw.inner().clone();
    let snap = |w: &World<f32>| {
        (0..4u8)
            .map(|i| format!("{:?}{:?}", w.cores.get(i).unwrap().pos, w.cores.get(i).unwrap().vel))
            .collect::<String>()
    };
    let before = (world.tick, snap(&world));
    let mut b = HybridBrain::new(fixed_cfg(2), ClockKind::Wall, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
    let ids = [0, 1, 2, 3];
    for _ in 0..3 {
        let obs = observation(&world, &map, 0, &ids, 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        let _ = b.decide_in(&obs, Some(&view));
    }
    assert_eq!((world.tick, snap(&world)), before);
}

#[test]
fn threats_are_the_free_opponents_within_the_radius_nearest_first() {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    // Victim 1 at 5 tiles; threat 2 at 8 tiles; threat 3 at 30 tiles (beyond the 440 px radius);
    // tee 4 is frozen at 6 tiles (not a threat).
    place(
        &mut pw,
        &[
            (0, 5.5, 9.5),
            (1, 10.5, 9.5),
            (2, 13.5, 9.5),
            (3, 35.5, 9.5),
            (4, 11.5, 9.5),
        ],
    );
    let mut st = pw.get_tee(4).unwrap();
    st.frozen = true;
    st.freeze_ticks_left = 200;
    pw.apply_tee_state(4, &st);
    let world = pw.inner().clone();
    let ids = [0, 1, 2, 3, 4];
    let mut b = HybridBrain::new(fixed_cfg(1), ClockKind::Wall, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
    let obs = observation(&world, &map, 0, &ids, 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let _ = b.decide_in(&obs, Some(&view));
    let t = b.last_decision().unwrap();
    assert_eq!(t.victim_id, 1);
    assert_eq!(t.threat_ids, vec![2], "only the free opponent within the radius");
    assert_eq!(t.combos, 4, "victim and one threat vary: 2^2 combinations");
    // With the 1v1 model the same scene has no threats and two combinations (victim only).
    let mut cfg = fixed_cfg(1);
    cfg.threat_model = false;
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
    let _ = b.decide_in(&obs, Some(&view));
    let t = b.last_decision().unwrap();
    assert!(t.threat_ids.is_empty());
    assert_eq!(t.combos, 2);
    assert!(b.name().ends_with("-1v1model"));
}

/// A hall like scenario T14: a solid wall on the left, freeze floor to its right.
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

/// Runs the hybrid brain as tee 0 against an idle tee 1 for `decisions` decisions from the state
/// `setup` leaves; returns the chosen labels and whether tee 0 was ever frozen or dead.
fn run_solo(
    cfg: HybridConfig,
    map: &Arc<MapData>,
    setup: impl FnOnce(&mut PhysicsWorld),
    decisions: usize,
) -> (Vec<String>, bool) {
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &[(0, 23.5, 8.0), (1, 45.5, 14.5)]);
    setup(&mut pw);
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    reset(&mut b, map, 0, 3);
    let mut last = empty_input();
    let mut labels = Vec::new();
    let mut out = false;
    for _ in 0..decisions {
        let world = pw.inner().clone();
        let obs = observation(&world, map, 0, &[0, 1], 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        let a = b.decide_in(&obs, Some(&view));
        labels.push(
            b.last_decision()
                .and_then(|t| t.chosen)
                .map_or("none", |c| c.label())
                .to_string(),
        );
        last = input_from_action(&a, &last);
        for _ in 0..2 {
            pw.set_input(0, last);
            pw.step();
            out |= pw.get_tee(0).is_none_or(|t| t.frozen || !t.alive);
        }
    }
    (labels, out)
}

fn falling_without_jumps(pw: &mut PhysicsWorld) {
    let mut st = pw.get_tee(0).unwrap();
    st.vel = Vec2 { x: 0.0, y: 3.0 };
    st.jumped = 3;
    st.jumped_total = Some(2);
    st.jumps_left = 0;
    pw.apply_tee_state(0, &st);
}

#[test]
fn a_tee_falling_without_jumps_over_freeze_hooks_the_wall_and_the_telemetry_names_it() {
    let map = wall_over_freeze();
    let (labels, out) = run_solo(fixed_cfg(1), &map, falling_without_jumps, 30);
    assert!(!out, "the tee froze: {labels:?}");
    assert!(
        labels.iter().any(|l| l.starts_with("T14")),
        "no decision named the panic hook: {labels:?}"
    );
    // Every label is a known source or technique name.
    for l in &labels {
        assert!(
            ["warm", "book", "cem", "throw", "proposal"].contains(&l.as_str())
                || l.starts_with('T')
                || l == "anchor escape",
            "{l}"
        );
    }
}

#[test]
fn a_hook_plan_is_its_own_first_escape_and_the_shield_leaves_it_alone() {
    // Task 3.5b, step 1: the remainder of the chosen plan, rolled out exactly, is the shield's first
    // escape. The shield's walk/jump escapes know no hook, so before this a falling jumpless tee
    // hooking the wall had "no escape" and a timed-out check could replace the hook by doing nothing
    // (T14: 88% -> 0%, 3.5 review round 1). Now the plan escape holds, nothing is substituted, and
    // counting a timeout as danger (`shield_timeout_danger`) is harmless.
    let map = wall_over_freeze();
    for timeout_danger in [false, true] {
        let mut cfg = HybridConfig {
            work_clock_us_per_tick: Some(2.2),
            proposals: 0,
            shield_timeout_danger: timeout_danger,
            ..HybridConfig::default()
        };
        cfg.adaptive.enabled = true;
        let mut pw = PhysicsWorld::new(map.clone(), 1);
        place(&mut pw, &[(0, 23.5, 8.0), (1, 45.5, 14.5)]);
        falling_without_jumps(&mut pw);
        let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
        reset(&mut b, &map, 0, 3);
        let mut last = empty_input();
        let mut out = false;
        for _ in 0..30 {
            let world = pw.inner().clone();
            let obs = observation(&world, &map, 0, &[0, 1], 1);
            let view = WorldView {
                world: &world,
                self_id: 0,
                lag_ticks: 0,
                in_flight: &[],
            };
            let a = b.decide_in(&obs, Some(&view));
            last = input_from_action(&a, &last);
            for _ in 0..2 {
                pw.set_input(0, last);
                pw.step();
                out |= pw.get_tee(0).is_none_or(|t| t.frozen || !t.alive);
            }
        }
        let t = b.totals();
        assert!(!out, "the tee froze (timeout danger {timeout_danger})");
        assert_eq!(
            t.shield_ran + t.shield_skipped,
            t.decisions,
            "the shield ran or was skipped (far from any hazard) on every decision (timeout danger {timeout_danger})"
        );
        assert!(t.shield_ran > 0, "the shield never ran");
        // Review F1 (rounds 1-2): a hang counts as settled only after 8 calm ticks, and this scene is one long
        // swing along the wall (13 px/tick) that never comes to rest within the check's horizon, so the plan
        // escape cannot be confirmed early: the check runs out of its reserve (incomplete), the chosen plan
        // stands and nothing is substituted. What this test pins is the outcome: the tee never freezes.
        if timeout_danger {
            // Counting a timeout as danger earns a safer-input search, and along a swing that never rests nearly
            // every check times out (24 of 30 decisions here are substituted): the reason that option is off. The
            // tee must still not freeze (asserted above).
        } else {
            assert_eq!(t.shielded, 0, "the shield substituted a verified plan");
        }
    }
}

/// Scenario T2's map: a floor, and a freeze wall (x 25, rows 6..9) a tile and a half right of the victim.
fn hammer_wall_map() -> Arc<MapData> {
    let (w, h) = (40usize, 16usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            let solid = y >= 10 || x == 0 || x == w - 1 || y == 0;
            let freeze = x == 25 && (6..=9).contains(&y);
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

/// How a test tells the brain about the spared tee.
#[derive(Clone, Copy, PartialEq)]
enum Declare {
    No,
    /// `set_spares` (positions) and `set_spare_ids`, as the live bot does.
    Inherent,
    /// Through `Brain::set_live_context` (positions and `spare_ids`, as the live bot passes them since 4.1b).
    Trait,
    /// Through `Brain::set_live_context` with the ids only, no positions: the ids alone must keep the
    /// body out of the threat set (the position match of `is_spared` cannot help here).
    TraitIdsOnly,
}

/// What a fight of scenario T2 (the victim beside a freeze wall) looked like with a third tee on the
/// hammer's line to the victim.
struct Swings {
    /// Fire presses in 20 decisions.
    fires: u32,
    /// ... of which would reach the third tee (the hammer's centre ~21 px along the aim; within 42 px).
    hits_third: u32,
    /// Threat ids at the first decision.
    threats: Vec<i32>,
    /// Tees simulated at the first decision.
    sim_tees: u32,
}

/// `in_world`: the third tee stands one tile in front of us (as a body); `declare`: how it is declared
/// spared. The live bot keeps spared tees in contact range in the world as bodies (4.1 round 2, F8) and
/// also passes their positions; with `in_world = false` only the planner's own hammer gate keeps the swings
/// off the spot.
fn swings(in_world: bool, declare: Declare) -> Swings {
    use ddai_brain::LiveContext;
    let map = hammer_wall_map();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    let third = Vec2 {
        x: 22.75 * 32.0,
        y: 9.5 * 32.0,
    };
    let mut tees = vec![(0, 22.0, 9.5), (1, 23.5, 9.5)];
    if in_world {
        tees.push((2, 22.75, 9.5));
    }
    place(&mut pw, &tees);
    let ids: Vec<i32> = tees.iter().map(|t| t.0).collect();
    let mut b = HybridBrain::new(fixed_cfg(1), ClockKind::Wall, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
    let mut out = Swings {
        fires: 0,
        hits_third: 0,
        threats: Vec::new(),
        sim_tees: 0,
    };
    let mut last = empty_input();
    for k in 0..20 {
        let world = pw.inner().clone();
        match declare {
            Declare::No => {}
            Declare::Inherent => {
                b.set_spares(vec![third], vec![Vec2 { x: 0.0, y: 0.0 }]);
                b.set_spare_ids(vec![2]);
            }
            Declare::Trait | Declare::TraitIdsOnly => {
                let p = ddai_physics::vmath::Vec2 {
                    x: third.x as f32,
                    y: third.y as f32,
                };
                let spares = [(p, ddai_physics::vmath::Vec2 { x: 0.0f32, y: 0.0 })];
                let spares: &[_] = if declare == Declare::Trait { &spares } else { &[] };
                let brain: &mut dyn Brain = &mut b;
                brain.set_live_context(&LiveContext {
                    spares,
                    spare_ids: &[2],
                    ..LiveContext::default()
                });
            }
        }
        let obs = observation(&world, &map, 0, &ids, 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        let a = b.decide_in(&obs, Some(&view));
        out.fires += u32::from(a.fire);
        if a.fire {
            let me = pw.get_tee(0).unwrap().pos;
            let len = f64::from(a.target.x).hypot(f64::from(a.target.y)).max(1.0);
            let start = Vec2 {
                x: me.x + f64::from(a.target.x) / len * 21.0,
                y: me.y + f64::from(a.target.y) / len * 21.0,
            };
            if (start.x - third.x).hypot(start.y - third.y) < 42.0 {
                out.hits_third += 1;
            }
        }
        if k == 0 {
            let t = b.last_decision().unwrap();
            out.threats = t.threat_ids.clone();
            out.sim_tees = t.sim_tees;
        }
        last = input_from_action(&a, &last);
        for _ in 0..2 {
            pw.set_input(0, last);
            pw.step();
        }
    }
    out
}

#[test]
fn a_spared_tee_in_hammer_reach_is_not_swung_at_and_is_no_threat() {
    // Task 3.5b (4.1 review F1/F8): the live bot's spared tees (friends, ignored, AFK) reach the hybrid.
    // The same fight without a third tee: the brain hammers the victim into the wall, and the swings
    // reach the spot where the third tee will stand (so the geometry really asks for the swing).
    let alone = swings(false, Declare::No);
    assert!(
        alone.fires > 0 && alone.hits_third > 0,
        "the plain brain should swing through that spot"
    );
    // A spared tee in the hammer's path that is NOT in the world: only the planner's own hammer gate,
    // fed from the live context, keeps the swings off it.
    let gone = swings(false, Declare::Inherent);
    assert_eq!(gone.hits_third, 0, "a swing toward the spared tee was fired");
    // Spared and IN the world as a body (what the bot passes now): never swung at, no threat, and still
    // simulated (it can deflect us), outside the cap.
    let body = swings(true, Declare::Inherent);
    assert_eq!(body.hits_third, 0, "a swing that would hit the spared tee was fired");
    assert!(
        body.threats.is_empty(),
        "the spared tee was treated as a threat: {:?}",
        body.threats
    );
    assert_eq!(
        body.sim_tees, 3,
        "the spared body is simulated next to us and the victim"
    );
    // The same through the trait with `spare_ids`: the swing gate and the threat exclusion both work.
    let via_trait = swings(true, Declare::Trait);
    assert_eq!(
        via_trait.hits_third, 0,
        "a swing that would hit the spared tee was fired (trait)"
    );
    assert!(via_trait.threats.is_empty(), "trait: the spared tee was a threat");
    // The ids alone (no positions) also keep the body out of the threat set; without them it is one.
    assert!(
        swings(true, Declare::TraitIdsOnly).threats.is_empty(),
        "ids only: the spared tee was a threat"
    );
    // The same third tee not spared is an ordinary threat.
    assert_eq!(swings(true, Declare::No).threats, vec![2]);
}

#[test]
fn a_spared_tee_is_never_the_target() {
    // The observation names a spared tee as the target: the brain plays against the nearest other tee.
    let map = hammer_wall_map();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &[(0, 22.0, 9.5), (1, 23.5, 9.5), (2, 22.75, 9.5)]);
    let world = pw.inner().clone();
    let mut b = HybridBrain::new(fixed_cfg(1), ClockKind::Wall, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
    b.set_spare_ids(vec![1]);
    let obs = observation(&world, &map, 0, &[0, 1, 2], 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let _ = b.decide_in(&obs, Some(&view));
    assert_eq!(
        b.last_decision().unwrap().victim_id,
        2,
        "the spared target was not replaced"
    );
}

#[test]
fn spared_tees_and_the_travel_goal_reach_every_worker_and_leave_fixed_decisions_identical() {
    // The pool's workers hold their own planners: the live context must reach them, and the decision
    // must stay identical for 1, 2 and 4 workers.
    let run = |w: usize| {
        let map = hall();
        let mut pw = PhysicsWorld::new(map.clone(), 1);
        let tees = [(0, 17.0, 9.5), (1, 19.4, 9.5), (2, 18.2, 9.5)];
        place(&mut pw, &tees);
        let world = pw.inner().clone();
        let mut b = HybridBrain::new(fixed_cfg(w), ClockKind::Wall, Box::new(NoProposer)).unwrap();
        reset(&mut b, &map, 0, 3);
        let t = pw.get_tee(2).unwrap();
        b.set_spares(vec![t.pos], vec![t.vel]);
        b.set_travel_goal(Some(Vec2 { x: 900.0, y: 300.0 }));
        let obs = observation(&world, &map, 0, &[0, 1, 2], 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        (0..4).map(|_| b.decide_in(&obs, Some(&view))).collect::<Vec<_>>()
    };
    let one = run(1);
    for w in [2usize, 4] {
        assert_eq!(one, run(w), "workers = {w}");
    }
    assert!(one.iter().all(|a| !a.fire));
}

/// A proposer that hands back one fixed plan, to check that a proposal is only a candidate.
struct Fixed(Vec<PlanStep>);
impl Proposer for Fixed {
    fn name(&self) -> &str {
        "fixed"
    }
    fn propose(&mut self, ctx: &ProposeCtx<'_>, out: &mut Vec<Vec<PlanStep>>) {
        let mut p = self.0.clone();
        p.resize(ctx.steps, *self.0.last().unwrap());
        out.push(p);
    }
}

#[test]
fn a_terrible_proposal_is_scored_and_rejected_not_obeyed() {
    // The proposal walks straight into the freeze pit; the search must not take it.
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &[(0, 18.5, 9.5), (1, 30.5, 9.5)]);
    let world = pw.inner().clone();
    let walk_in = vec![
        PlanStep {
            dir: 1,
            jump: 0,
            hook: 0,
            fire: 0,
            aim: 0.0,
        };
        9
    ];
    let mut cfg = fixed_cfg(1);
    cfg.proposals = 1;
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(Fixed(walk_in))).unwrap();
    reset(&mut b, &map, 0, 3);
    let obs = observation(&world, &map, 0, &[0, 1], 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let _ = b.decide_in(&obs, Some(&view));
    let t = b.last_decision().unwrap();
    assert_eq!(t.generated[1], 1, "the proposal entered the pool");
    assert_eq!(t.evaluated[1], 1, "and was scored like every other candidate");
    assert!(
        !matches!(t.chosen, Some(ddai_planner::hybrid::search::Source::Proposal)),
        "chosen {:?}",
        t.chosen
    );
}

#[test]
fn the_scripted_proposer_proposes_plans_of_the_planner_length() {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &[(0, 10.5, 9.5), (1, 18.5, 9.5)]);
    let world = pw.inner().clone();
    let mut cfg = fixed_cfg(1);
    cfg.proposals = 3;
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(ScriptedProposer::new())).unwrap();
    reset(&mut b, &map, 0, 3);
    let obs = observation(&world, &map, 0, &[0, 1], 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let _ = b.decide_in(&obs, Some(&view));
    let t = b.last_decision().unwrap();
    assert!(t.generated[1] >= 1, "{:?}", t.generated);
    assert_eq!(t.work.proposal, 27 * 3, "three rollouts of 9 steps x 3 ticks");
    assert!(b.name().contains("scripted"));
}

#[test]
fn deadline_budget_bounds_the_search_and_grows_it_monotonically() {
    let run = |budget: f64| {
        let mut cfg = HybridConfig {
            mode: HybridMode::Deadline { budget_ms: budget },
            proposals: 0,
            decision_cap_ms: None,
            ..HybridConfig::default()
        };
        cfg.adaptive.enabled = false;
        let map = hall();
        let mut pw = PhysicsWorld::new(map.clone(), 1);
        place(&mut pw, &FOUR);
        let world = pw.inner().clone();
        let mut b = HybridBrain::new(cfg, ClockKind::Step { step_ms: 0.05 }, Box::new(NoProposer)).unwrap();
        reset(&mut b, &map, 0, 3);
        let ids = [0, 1, 2, 3];
        let obs = observation(&world, &map, 0, &ids, 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        for _ in 0..3 {
            let _ = b.decide_in(&obs, Some(&view));
        }
        (b.totals().work.rollouts_stage1, b.totals().decisions)
    };
    let (tiny, n) = run(0.3);
    let (mid, _) = run(2.0);
    let (big, _) = run(8.0);
    assert_eq!(n, 3);
    assert!(tiny < mid && mid < big, "{tiny} {mid} {big}");
    // Even a near-zero budget scores the first candidate (a decision is always produced).
    assert!(tiny >= 3);
}

/// Task 3.16 (D-115): `Brain::set_decision_deadline_ms` lowers the cap of the next decision only, never raises it.
#[test]
fn a_decision_deadline_lowers_the_cap_of_the_next_decision_only() {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &FOUR);
    let world = pw.inner().clone();
    let cfg = HybridConfig {
        mode: HybridMode::Deadline { budget_ms: 4.0 },
        proposals: 0,
        decision_cap_ms: Some(5.0),
        shield_reserve_ms_per_tee: 0.25,
        mirror: false,
        ..HybridConfig::default()
    };
    let mut b = HybridBrain::new(cfg, ClockKind::Step { step_ms: 0.01 }, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
    let ids: Vec<i32> = FOUR.iter().map(|t| t.0).collect();
    let obs = observation(&world, &map, 0, &ids, 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let mut budget_after = |deadline: Option<f64>| {
        b.set_decision_deadline_ms(deadline);
        let _ = b.decide_in(&obs, Some(&view));
        b.last_decision().expect("telemetry").budget_ms
    };
    assert_eq!(budget_after(None), 4.0, "5 - 4 x 0.25 = 4: the asked budget fits");
    assert_eq!(budget_after(Some(3.0)), 2.0, "a 3 ms deadline: 3 - 4 x 0.25");
    assert_eq!(
        budget_after(None),
        4.0,
        "one-shot: the next decision has its own cap again"
    );
    assert_eq!(budget_after(Some(9.0)), 4.0, "a deadline above the cap changes nothing");
    assert_eq!(budget_after(Some(1.2)), 1.0, "never below the search minimum of 1 ms");
    assert_eq!(budget_after(Some(f64::NAN)), 4.0, "a garbage deadline is ignored");
    assert_eq!(budget_after(Some(-1.0)), 4.0);

    // Review F6: a decision that returns early (nothing to target) still uses up the deadline it was given.
    let mut none = obs.clone();
    none.others.clear();
    none.target_id = None;
    b.set_decision_deadline_ms(Some(3.0));
    let _ = b.decide_in(&none, Some(&view));
    let _ = b.decide_in(&obs, Some(&view));
    assert_eq!(
        b.last_decision().expect("telemetry").budget_ms,
        4.0,
        "the early return took the deadline: it must not reach the next decision"
    );
    b.set_decision_deadline_ms(Some(3.0));
    let _ = b.decide(&none);
    b.set_decision_deadline_ms(None);
    let _ = b.decide_in(&obs, Some(&view));
    assert_eq!(b.last_decision().expect("telemetry").budget_ms, 4.0);
}

#[test]
fn the_decision_cap_shortens_the_search_by_the_shield_reserve_of_many_tees() {
    // Search budget = min(budget, cap - reserve per tee * tees), never below 1 ms; `None` = as asked.
    let budget_of = |cap: Option<f64>, reserve: f64, tees: &[(i32, f64, f64)]| {
        let cfg = HybridConfig {
            mode: HybridMode::Deadline { budget_ms: 4.0 },
            proposals: 0,
            decision_cap_ms: cap,
            shield_reserve_ms_per_tee: reserve,
            // The arithmetic of the cap, without the opponent model's share of it (task 3.7b: tested apart).
            mirror: false,
            ..HybridConfig::default()
        };
        let map = hall();
        let mut pw = PhysicsWorld::new(map.clone(), 1);
        place(&mut pw, tees);
        let world = pw.inner().clone();
        let mut b = HybridBrain::new(cfg, ClockKind::Step { step_ms: 0.01 }, Box::new(NoProposer)).unwrap();
        reset(&mut b, &map, 0, 3);
        let ids: Vec<i32> = tees.iter().map(|t| t.0).collect();
        let obs = observation(&world, &map, 0, &ids, 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        let _ = b.decide_in(&obs, Some(&view));
        b.last_decision().expect("telemetry").budget_ms
    };
    assert_eq!(
        budget_of(Some(5.0), 0.25, &FOUR),
        4.0,
        "5 - 4 x 0.25 = 4: the asked budget fits"
    );
    assert_eq!(budget_of(Some(5.0), 0.5, &FOUR), 3.0, "5 - 4 x 0.5");
    assert_eq!(budget_of(Some(5.0), 2.0, &FOUR), 1.0, "never below 1 ms");
    assert_eq!(budget_of(None, 0.5, &FOUR), 4.0, "no cap: the budget as asked");
}

/// A proposer that proposes nothing and costs `units` tee-ticks on the work clock (the shape of the fly).
struct Costly {
    units: u64,
}

impl Proposer for Costly {
    fn name(&self) -> &str {
        "costly"
    }
    fn propose(&mut self, _ctx: &ProposeCtx<'_>, _out: &mut Vec<Vec<PlanStep>>) {}
    fn work_units(&self) -> u64 {
        self.units
    }
}

/// One decision of slot 0 on the work clock at 1.25 us per tee-tick with the given proposer; the telemetry of it.
fn work_decision(
    proposer: Box<dyn Proposer>,
    tweak: impl FnOnce(&mut HybridConfig),
    tees: &[(i32, f64, f64)],
) -> ddai_planner::hybrid::DecisionTelemetry {
    let mut cfg = HybridConfig {
        mode: HybridMode::Deadline { budget_ms: 4.0 },
        proposals: 3,
        work_clock_us_per_tick: Some(1.25),
        ..HybridConfig::default()
    };
    tweak(&mut cfg);
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, tees);
    let world = pw.inner().clone();
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, proposer).unwrap();
    reset(&mut b, &map, 0, 3);
    let ids: Vec<i32> = tees.iter().map(|t| t.0).collect();
    let obs = observation(&world, &map, 0, &ids, 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let _ = b.decide_in(&obs, Some(&view));
    b.last_decision().expect("telemetry").clone()
}

const TWO: [(i32, f64, f64); 2] = [(0, 17.5, 9.5), (1, 24.5, 9.5)];

#[test]
fn work_clock_decisions_are_bit_identical_for_1_2_and_4_workers() {
    // Task 3.7a: on the work clock the helpers only speculate. Everything the decision reports --
    // the action, the work counters, the candidate counts, every time the clock reads (it is a
    // counter of finished work) -- must equal the single-thread run, in every mode of the search.
    type Tweak = fn(&mut HybridConfig);
    let variants: [(&str, Tweak); 7] = [
        ("plain 4 ms", |_| ()),
        ("12 ms: CEM, stage 2 and the extension", |c| {
            c.mode = HybridMode::Deadline { budget_ms: 12.0 };
            c.decision_cap_ms = None;
        }),
        ("pruning, crowd stage, hook escapes", |c| {
            c.prune.enabled = true;
            c.robust.crowd_stage = true;
            c.mode = HybridMode::Deadline { budget_ms: 9.0 };
            c.decision_cap_ms = None;
        }),
        ("two worlds", |c| {
            c.two_world = true;
            c.mode = HybridMode::Deadline { budget_ms: 9.0 };
            c.decision_cap_ms = None;
        }),
        ("all opponents modelled", |c| {
            c.robust.max_relevant = 4;
            c.robust.max_combos = 4;
        }),
        ("opponent model, 8 samples", |c| {
            c.mirror = true;
            c.mirror_samples = 8;
        }),
        ("no opponent model (the victim holds its input)", |c| c.mirror = false),
    ];
    let mut hits_total = 0u32;
    for (name, tweak) in variants {
        for scene in [&FOUR[..], &TWO[..]] {
            let run = |workers: usize| {
                let used = std::cell::Cell::new(0u32);
                let prefetched = std::cell::Cell::new(0u32);
                let mut cfg = HybridConfig {
                    mode: HybridMode::Deadline { budget_ms: 4.0 },
                    proposals: 0,
                    workers,
                    work_clock_us_per_tick: Some(1.25),
                    ..HybridConfig::default()
                };
                tweak(&mut cfg);
                let log = drive_with(cfg, ClockKind::Wall, scene, 20, |t| {
                    used.set(used.get() + t.spec_used);
                    prefetched.set(prefetched.get() + t.spec_prefetched);
                    assert!(t.spec_used <= t.spec_prefetched);
                    t.to_json()
                });
                (log, used.get(), prefetched.get())
            };
            let (one, used1, pre1) = run(1);
            assert_eq!((used1, pre1), (0, 0), "{name}: one worker does not speculate");
            assert_eq!(one.len(), 20);
            for w in [2usize, 4] {
                let (many, used, pre) = run(w);
                if let Some(k) = one.iter().zip(&many).position(|(a, b)| a != b) {
                    panic!(
                        "{name}, {} tees: workers = {w} decided differently from workers = 1 at decision {k}:\n  1: {:?}\n  {w}: {:?}",
                        scene.len(),
                        one[k],
                        many[k]
                    );
                }
                assert_eq!(one.len(), many.len());
                assert!(pre >= used);
                hits_total += used;
            }
        }
    }
    assert!(
        hits_total > 100,
        "the helpers must really have supplied rollouts, else this proves nothing: {hits_total}"
    );
}

#[test]
fn a_decision_that_returns_early_leaves_no_verdict_behind() {
    // Review 3.7a F5/F9: `last_decision`/`last_plan` must not report the previous decision's search as the verdict of
    // a decision that never searched (no target, or the target gone from the planning world).
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &TWO);
    let world = pw.inner().clone();
    let ids = [0, 1];
    let obs = observation(&world, &map, 0, &ids, 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let mut b = HybridBrain::new(fixed_cfg(1), ClockKind::Wall, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
    let _ = b.decide_in(&obs, Some(&view));
    assert!(b.last_decision().is_some());
    assert!(
        b.last_plan().is_some_and(|p| p.searched),
        "a real decision reports its search"
    );

    // No target at all.
    let mut none = observation(&world, &map, 0, &ids, 1);
    none.target_id = None;
    none.others.clear();
    let _ = b.decide_in(&none, Some(&view));
    assert!(
        b.last_decision().is_none() && b.last_plan().is_none(),
        "no target: no verdict"
    );

    // A target the observation names but the planning world no longer has.
    let _ = b.decide_in(&obs, Some(&view));
    assert!(b.last_plan().is_some());
    let mut alone = PhysicsWorld::new(map.clone(), 1);
    place(&mut alone, &TWO[..1]);
    let alone_world = alone.inner().clone();
    let alone_view = WorldView {
        world: &alone_world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let _ = b.decide_in(&obs, Some(&alone_view));
    assert!(
        b.last_decision().is_none() && b.last_plan().is_none(),
        "target gone from the world: no verdict"
    );
    // ... and the next real decision reports again.
    let _ = b.decide_in(&obs, Some(&view));
    assert!(b.last_plan().is_some_and(|p| p.searched));
}

#[test]
fn the_proposers_time_comes_off_the_search_budget() {
    // 800 tee-ticks x 1.25 us = 1.0 ms of proposal. Four tees: shield reserve 4 x 0.25 = 1.0 ms. Cap 5.
    let with = |units: u64, in_cap: bool| {
        work_decision(
            Box::new(Costly { units }),
            |c| {
                c.proposal_in_cap = in_cap;
                c.mirror = false;
            },
            &FOUR,
        )
    };
    let t = with(800, true);
    assert_eq!(t.work.proposal_units, 800);
    assert_eq!(t.proposal_ms, 1.0, "the work clock charges the proposer's units");
    assert_eq!(t.budget_ms, 3.0, "5 - 1.0 (proposal) - 1.0 (shield reserve)");
    assert!(
        t.proposal_ms + t.search_ms + t.shield_ms <= 5.0 + 0.5,
        "proposals + search + shield stay under the cap: {} + {} + {}",
        t.proposal_ms,
        t.search_ms,
        t.shield_ms
    );
    assert_eq!(
        with(800, false).budget_ms,
        4.0,
        "the old behaviour: the full budget after the proposals"
    );
    assert_eq!(with(400, true).budget_ms, 3.5);
    assert_eq!(with(0, true).budget_ms, 4.0, "a free proposer costs nothing");
    assert_eq!(with(4000, true).budget_ms, 1.0, "never below MIN_SEARCH_MS");
    // `NoProposer` costs nothing and charges nothing, whatever the flag.
    let none = work_decision(Box::new(NoProposer), |c| c.mirror = false, &FOUR);
    assert_eq!(
        (none.proposal_ms, none.work.proposal_units, none.budget_ms),
        (0.0, 0, 4.0)
    );
}

#[test]
fn the_opponent_model_predicts_the_victims_plan_and_its_search_comes_off_the_cap() {
    // Task 3.7b: `mirror` runs a small search from the victim's seat while it is free and within the threat radius.
    let off = work_decision(Box::new(NoProposer), |c| c.mirror = false, &TWO);
    assert_eq!(
        (off.work.mirror, off.mirror_first),
        (0, None),
        "off: no search, no prediction"
    );
    let on = work_decision(Box::new(NoProposer), |c| c.mirror = true, &TWO);
    assert!(
        on.work.mirror > 0,
        "the victim's search is charged to the work counters"
    );
    assert!(on.mirror_first.is_some(), "a predicted first input");
    // Its rollouts come off the search budget (cap 5 ms - shield reserve 0.5 ms - the model's time), never below 1 ms.
    let mirror_ms = on.work.mirror as f64 * 2.0 * 1.25 / 1000.0;
    assert!(
        (on.budget_ms - (4.0f64).min(4.5 - mirror_ms)).abs() < 1e-9,
        "budget {} after {mirror_ms} ms of opponent model",
        on.budget_ms
    );
    assert!(
        on.work.total_ticks() as f64 * 2.0 * 1.25 / 1000.0 <= 5.0 + 0.6,
        "the decision, opponent model included, stays under the cap: {} ticks",
        on.work.total_ticks()
    );
    // More samples cost more.
    let more = work_decision(
        Box::new(NoProposer),
        |c| {
            c.mirror = true;
            c.mirror_samples = 16;
        },
        &TWO,
    );
    assert!(more.work.mirror > on.work.mirror);
    // The telemetry JSON carries the counter only when the model ran.
    assert!(!off.to_json().contains("\"mirror\""));
    assert!(on.to_json().contains("\"mirror\":"));
    // A fight against several (another free opponent within the radius) is a defence, not a duel: no model.
    let crowd = work_decision(Box::new(NoProposer), |c| c.mirror = true, &FOUR);
    assert_eq!((crowd.work.mirror, crowd.mirror_first), (0, None));
    // A victim outside the threat radius is not modelled (nothing it does can reach us within the plan).
    let far = work_decision(
        Box::new(NoProposer),
        |c| c.mirror = true,
        &[(0, 3.5, 9.5), (1, 36.5, 9.5)],
    );
    assert_eq!((far.work.mirror, far.mirror_first), (0, None));
}

/// Slot 0 = the hybrid, slot 1 = `opponent` (an idle tee or the scripted attacker) on the work clock: per decision the telemetry
/// of slot 0 (`mirror_first` is `Some` while the opponent model's plan is used).
fn mirror_use(opponent_idle: bool, decisions: usize) -> Vec<bool> {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    // Two tees on the safe side of the hall, 7 tiles apart (the freeze pit is at x = 20..25).
    place(&mut pw, &[(0, 4.5, 9.5), (1, 11.5, 9.5)]);
    let ids = [0, 1];
    let cfg = HybridConfig {
        mode: HybridMode::Deadline { budget_ms: 4.0 },
        proposals: 0,
        work_clock_us_per_tick: Some(1.25),
        ..HybridConfig::default()
    };
    let mut hybrid = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    reset(&mut hybrid, &map, 0, 7);
    let mut other: Box<dyn Brain> = if opponent_idle {
        Box::new(ddai_brain::IdleBrain)
    } else {
        Box::new(ScriptedBrain::new())
    };
    reset(&mut *other, &map, 1, 100);
    let mut last = [empty_input(), empty_input()];
    let mut used = Vec::new();
    for _ in 0..decisions {
        let world = pw.inner().clone();
        for (slot, &id) in ids.iter().enumerate() {
            let obs = observation(&world, &map, id, &ids, 1 - id);
            let view = WorldView {
                world: &world,
                self_id: id,
                lag_ticks: 0,
                in_flight: &[],
            };
            let action = if slot == 0 {
                let a = hybrid.decide_in(&obs, Some(&view));
                if let Some(t) = hybrid.last_decision() {
                    used.push(t.mirror_first.is_some());
                }
                a
            } else {
                other.decide_in(&obs, Some(&view))
            };
            last[slot] = input_from_action(&action, &last[slot]);
        }
        for _ in 0..2 {
            for (slot, &id) in ids.iter().enumerate() {
                pw.set_input(id, last[slot]);
            }
            pw.step();
        }
    }
    used
}

#[test]
fn the_opponent_model_is_skipped_for_an_idle_opponent_and_kept_for_an_attacker() {
    // A victim that stands with a neutral direction and no hook out for six decisions in a row is passive: "it keeps its input" is
    // right for it, and the model that expects an attack would only make us play away from a victim that never comes. It costs
    // nothing then; an attacker keeps the model in use.
    let idle = mirror_use(true, 40);
    assert!(
        idle[..5].iter().all(|&u| u),
        "the model is used until the victim has been passive for a while"
    );
    assert!(
        idle[8..].iter().all(|&u| !u),
        "idle: the plan is not used once the victim has been passive for six decisions: {idle:?}"
    );
    let attacker = mirror_use(false, 40);
    assert!(
        attacker.iter().filter(|&&u| u).count() * 10 >= attacker.len() * 8,
        "an attacker keeps the model in use: {attacker:?}"
    );
}

/// One decision of slot 0 on a step clock (every clock read costs `step_ms`), the opponent model on, two tees on the safe side.
fn step_decision(step_ms: f64) -> ddai_planner::hybrid::DecisionTelemetry {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &[(0, 4.5, 9.5), (1, 11.5, 9.5)]);
    let world = pw.inner().clone();
    let cfg = HybridConfig {
        mode: HybridMode::Deadline { budget_ms: 4.0 },
        proposals: 0,
        ..HybridConfig::default()
    };
    let mut b = HybridBrain::new(cfg, ClockKind::Step { step_ms }, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
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

#[test]
fn the_opponent_models_search_is_cut_by_its_deadline_and_the_decision_still_completes() {
    // Task 3.7b review F2: under a cap the model's search stops at its deadline (at most 2 ms) and the best plan so far is used.
    let slow = step_decision(0.05);
    assert!(slow.mirror_cut, "a slow clock cuts the model's search");
    assert!(
        slow.mirror_ms > 0.0 && slow.mirror_ms < 2.5,
        "it stopped near its deadline: {} ms",
        slow.mirror_ms
    );
    assert!(
        slow.chosen.is_some(),
        "the decision completes without or with a partial prediction"
    );
    assert!(slow.to_json().contains("\"mirror_cut\":true"));
    let fast = step_decision(0.0001);
    assert!(
        !fast.mirror_cut && fast.mirror_first.is_some(),
        "a fast clock leaves it alone"
    );
    assert!(fast.to_json().contains("\"mirror_ms\":"));
    // On the work clock the search has no deadline, whatever it costs: the arena's games stay deterministic.
    let work = work_decision(Box::new(NoProposer), |c| c.mirror_samples = 400, &TWO);
    assert!(!work.mirror_cut && work.work.mirror > 0);
    assert!(
        work.mirror_ms > 2.0,
        "the model ran past its wall-clock deadline of 2 ms on the work clock: {} ms",
        work.mirror_ms
    );
    assert!((work.mirror_ms - work.work.mirror as f64 * 2.0 * 1.25 / 1000.0).abs() < 1e-9);
}

#[test]
fn the_opponent_model_makes_the_same_game_twice() {
    let run = || {
        drive(
            {
                let mut c = fixed_cfg(1);
                c.mirror = true;
                c
            },
            ClockKind::Wall,
            &TWO,
            20,
        )
    };
    let (a, b) = (run(), run());
    assert_eq!(a, b, "the same game twice");
    assert!(a.iter().any(|(_, t)| !t.is_empty()));
}

#[test]
fn frozen_we_search_a_millisecond_and_never_extend() {
    // Frozen, the game ignores our inputs: the extension cannot achieve anything (review round 1, F4).
    // Two opponents near the pit flag danger, and "we end out" is always true while frozen.
    let map = hall();
    let tees = [(0, 19.6, 9.5), (1, 22.5, 9.0), (2, 17.5, 9.5), (3, 16.5, 9.5)];
    for frozen in [false, true] {
        let mut pw = PhysicsWorld::new(map.clone(), 1);
        place(&mut pw, &tees);
        if frozen {
            let mut st = pw.get_tee(0).unwrap();
            st.frozen = true;
            st.freeze_ticks_left = 200;
            pw.apply_tee_state(0, &st);
        }
        let world = pw.inner().clone();
        let ids = [0, 1, 2, 3];
        let cfg = HybridConfig {
            mode: HybridMode::Deadline { budget_ms: 4.0 },
            proposals: 0,
            decision_cap_ms: None,
            ..HybridConfig::default()
        };
        let mut b = HybridBrain::new(cfg, ClockKind::Step { step_ms: 0.05 }, Box::new(NoProposer)).unwrap();
        reset(&mut b, &map, 0, 3);
        let obs = observation(&world, &map, 0, &ids, 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        for _ in 0..4 {
            let _ = b.decide_in(&obs, Some(&view));
            let t = b.last_decision().unwrap();
            if frozen {
                assert!(!t.extended, "extended while frozen");
                assert!(t.budget_ms <= 1.0, "frozen search budget {}", t.budget_ms);
            } else {
                assert_eq!(t.budget_ms, 4.0);
            }
        }
    }
}

#[test]
fn max_combos_one_means_exactly_one_combination() {
    // Two opponents in the radius and one victim: with `max_combos = 2` the two extremes (2
    // combinations); with `max_combos = 1` only the cheap model. Before, 1 still gave 2.
    let map = hall();
    let tees = [(0, 5.5, 9.5), (1, 10.5, 9.5), (2, 13.5, 9.5)];
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &tees);
    let world = pw.inner().clone();
    let ids = [0, 1, 2];
    let combos = |max_combos: usize| {
        let mut cfg = fixed_cfg(1);
        cfg.robust.max_combos = max_combos;
        let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
        reset(&mut b, &map, 0, 3);
        let obs = observation(&world, &map, 0, &ids, 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        let _ = b.decide_in(&obs, Some(&view));
        b.last_decision().unwrap().combos
    };
    assert_eq!(combos(1), 1);
    assert_eq!(combos(2), 2);
    assert_eq!(combos(4), 4);
}

#[test]
fn the_extension_is_only_used_when_danger_is_flagged() {
    let cfg = HybridConfig {
        mode: HybridMode::Deadline { budget_ms: 2.0 },
        proposals: 0,
        ..HybridConfig::default()
    };
    let map = hall();
    // A calm 1v1 far from the pit, and a 1v3 at the pit's edge.
    for (tees, expect_flag) in [
        (vec![(0, 8.5, 9.5), (1, 12.5, 9.5)], false),
        (
            vec![(0, 19.6, 9.5), (1, 22.5, 9.0), (2, 17.5, 9.5), (3, 16.5, 9.5)],
            true,
        ),
    ] {
        let mut pw = PhysicsWorld::new(map.clone(), 1);
        place(&mut pw, &tees);
        let world = pw.inner().clone();
        let ids: Vec<i32> = tees.iter().map(|t| t.0).collect();
        let mut b = HybridBrain::new(cfg.clone(), ClockKind::Step { step_ms: 0.05 }, Box::new(NoProposer)).unwrap();
        reset(&mut b, &map, 0, 3);
        let obs = observation(&world, &map, 0, &ids, 1);
        let view = WorldView {
            world: &world,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        for _ in 0..4 {
            let _ = b.decide_in(&obs, Some(&view));
            let t = b.last_decision().unwrap();
            assert_eq!(t.danger.flagged(), expect_flag, "{:?}", t.danger);
            assert!(!t.extended || t.danger.flagged(), "an extension without a danger flag");
        }
        assert!(b.totals().extended <= b.totals().danger_flagged);
        if !expect_flag {
            assert_eq!(b.totals().extended, 0);
        }
    }
}

#[test]
fn telemetry_json_has_the_documented_fields() {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &FOUR);
    let world = pw.inner().clone();
    let mut b = HybridBrain::new(fixed_cfg(1), ClockKind::Wall, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
    let obs = observation(&world, &map, 0, &[0, 1, 2, 3], 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let _ = b.decide_in(&obs, Some(&view));
    let v: serde_json::Value = serde_json::from_str(&b.telemetry().unwrap()).unwrap();
    let last = &v["last"];
    for key in [
        "chosen",
        "plan",
        "victim",
        "threats",
        "danger",
        "combos",
        "generated",
        "evaluated",
        "budget_ms",
        "extended",
        "shielded",
        "shield_incomplete",
        "work",
    ] {
        assert!(!last[key].is_null(), "missing {key} in {last}");
    }
    assert_eq!(last["plan"].as_array().unwrap().len(), 9);
    assert_eq!(last["threats"].as_array().unwrap().len(), 2, "{last}");
    assert_eq!(v["totals"]["decisions"], 1);
    assert_eq!(v["workers"], 1);
    reset(&mut b, &map, 0, 3);
    let v: serde_json::Value = serde_json::from_str(&b.telemetry().unwrap()).unwrap();
    assert_eq!(v["totals"]["decisions"], 0, "reset clears the totals");
}

#[test]
fn works_without_a_world_view_from_the_observation_alone() {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &[(0, 10.5, 9.5), (1, 15.5, 9.5)]);
    let world = pw.inner().clone();
    let mut b = HybridBrain::new(fixed_cfg(1), ClockKind::Wall, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
    let obs = observation(&world, &map, 0, &[0, 1], 1);
    let a = b.decide(&obs);
    assert!((-1..=1).contains(&a.direction));
    assert_eq!(b.totals().decisions, 1);
    let a2 = b.decide_in(&obs, None);
    assert!((-1..=1).contains(&a2.direction));
}

#[test]
fn config_validation_and_step_clock_workers() {
    let c = HybridConfig {
        workers: 0,
        ..HybridConfig::default()
    };
    assert!(HybridBrain::new(c, ClockKind::Wall, Box::new(NoProposer)).is_err());
    let c = HybridConfig {
        workers: 2,
        ..HybridConfig::default()
    };
    assert!(
        HybridBrain::new(c, ClockKind::Step { step_ms: 0.1 }, Box::new(NoProposer)).is_err(),
        "worker threads cannot read an injected step clock"
    );
    for bad in [
        HybridConfig {
            decision_cap_ms: Some(0.0),
            ..HybridConfig::default()
        },
        HybridConfig {
            shield_reserve_ms_per_tee: 0.0,
            ..HybridConfig::default()
        },
        HybridConfig {
            shield_reserve_ms_per_tee: f64::NAN,
            ..HybridConfig::default()
        },
        HybridConfig {
            mode: HybridMode::Deadline { budget_ms: f64::NAN },
            ..HybridConfig::default()
        },
        HybridConfig {
            threat_weight: f64::INFINITY,
            ..HybridConfig::default()
        },
        HybridConfig {
            threat_radius_px: Some(f64::NAN),
            ..HybridConfig::default()
        },
        HybridConfig {
            adaptive: AdaptiveConfig {
                max_total_ms: f64::NAN,
                ..AdaptiveConfig::default()
            },
            ..HybridConfig::default()
        },
    ] {
        assert!(HybridBrain::new(bad, ClockKind::Wall, Box::new(NoProposer)).is_err());
    }
}

// A tiny clone helper: the search wants an owned world of its own.
trait CloneForTest {
    fn clone_for_test(&self) -> PhysicsWorld;
}
impl CloneForTest for PhysicsWorld {
    fn clone_for_test(&self) -> PhysicsWorld {
        PhysicsWorld::from_world(self.inner().clone(), self.map().clone())
    }
}

#[test]
fn the_work_clock_makes_deadline_mode_reproducible_and_load_independent() {
    let cfg = || {
        let mut c = HybridConfig {
            mode: HybridMode::Deadline { budget_ms: 3.0 },
            proposals: 0,
            work_clock_us_per_tick: Some(8.0),
            ..HybridConfig::default()
        };
        c.adaptive.enabled = true;
        c
    };
    let run = || drive(cfg(), ClockKind::Wall, &FOUR, 20);
    let quiet = run();
    // The same games while other threads burn CPU: wall time differs, work does not.
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let burners: Vec<_> = (0..6)
        .map(|_| {
            let s = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut x = 1u64;
                while !s.load(std::sync::atomic::Ordering::Relaxed) {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    std::hint::black_box(x);
                }
            })
        })
        .collect();
    let loaded = run();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for b in burners {
        b.join().unwrap();
    }
    assert_eq!(quiet, loaded, "a work-clock search must not depend on machine load");
    // ... and it is a real deadline: a bigger budget searches more.
    let quiet_ticks: u64 = quiet
        .iter()
        .map(|(_, t)| t.split('|').nth(4).unwrap().parse::<u64>().unwrap())
        .sum();
    let mut big = cfg();
    big.mode = HybridMode::Deadline { budget_ms: 9.0 };
    let big_ticks: u64 = drive(big, ClockKind::Wall, &FOUR, 20)
        .iter()
        .map(|(_, t)| t.split('|').nth(4).unwrap().parse::<u64>().unwrap())
        .sum();
    assert!(big_ticks > quiet_ticks, "{big_ticks} vs {quiet_ticks}");
}

/// How much stack one decision needs (live threads are tokio/std threads with 2 MB by default): run
/// with `DDAI_STACK_BYTES=<n>`; a run that overflows aborts, so a caller bisects `n` from outside:
/// `for n in 600000 800000 ...; do DDAI_STACK_BYTES=$n cargo test ... decision_stack -- --ignored; done`.
#[test]
#[ignore = "probe; needs DDAI_STACK_BYTES"]
fn decision_stack() {
    let Some(bytes) = std::env::var("DDAI_STACK_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    else {
        return;
    };
    let build_only = std::env::var("DDAI_STACK_PHASE").is_ok_and(|v| v == "build");
    std::thread::Builder::new()
        .stack_size(bytes)
        .spawn(move || {
            let map = hall();
            let mut pw = Box::new(PhysicsWorld::new(map.clone(), 1));
            place(&mut pw, &FOUR);
            let world = Box::new(pw.inner().clone());
            let mut b =
                Box::new(HybridBrain::new(HybridConfig::default(), ClockKind::Wall, Box::new(NoProposer)).unwrap());
            reset(&mut *b, &map, 0, 3);
            if build_only {
                return;
            }
            let ids = [0, 1, 2, 3];
            let obs = observation(&world, &map, 0, &ids, 1);
            let view = WorldView {
                world: &world,
                self_id: 0,
                lag_ticks: 0,
                in_flight: &[],
            };
            for _ in 0..3 {
                let _ = b.decide_in(&obs, Some(&view));
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Hall scene for the local-tee tests: slot 0 is us at x = 8 tiles, tee 1 the (far) victim; `extra` are
/// `(id, x tiles, vx, flying hook toward us)`. `hookers` get a grabbed hook on us. Returns the first
/// decision's telemetry: (threat ids in order, simulated tees, dropped tees).
fn local_scene(cap: usize, extra: &[(i32, f64, f64, bool)], hookers: &[i32]) -> (Vec<i32>, u32, u32) {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    let mut tees = vec![(0, 8.0, 9.5), (1, 36.0, 9.5)];
    tees.extend(extra.iter().map(|e| (e.0, e.1, 9.5)));
    place(&mut pw, &tees);
    for _ in 0..5 {
        pw.step();
    }
    let me = pw.get_tee(0).unwrap();
    for &(id, _, vx, flying) in extra {
        let mut st = pw.get_tee(id).unwrap();
        st.vel = Vec2 { x: vx, y: 0.0 };
        if hookers.contains(&id) {
            st.hook_state = ddai_planner::types::HOOK_GRABBED;
            st.hooked_player = 0;
            st.hook_pos = me.pos;
        } else if flying {
            st.hook_state = ddai_planner::types::HOOK_FLYING;
            st.hook_pos = Vec2 {
                x: (st.pos.x + me.pos.x) / 2.0,
                y: st.pos.y,
            };
            st.hook_dir = Vec2 {
                x: (me.pos.x - st.pos.x).signum(),
                y: 0.0,
            };
        }
        pw.apply_tee_state(id, &st);
    }
    let w = pw.inner().clone();
    let ids: Vec<i32> = tees.iter().map(|t| t.0).collect();
    let mut cfg = fixed_cfg(1);
    cfg.max_sim_tees = cap;
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    reset(&mut b, &map, 0, 3);
    let obs = observation(&w, &map, 0, &ids, 1);
    let view = WorldView {
        world: &w,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let _ = b.decide_in(&obs, Some(&view));
    let t = b.last_decision().unwrap();
    (t.threat_ids.clone(), t.sim_tees, t.dropped_tees)
}

#[test]
fn every_tee_that_hooks_us_stays_in_the_simulation_whatever_the_cap() {
    // Review F2: two tees (2 and 4) hold us on their hooks, an idle tee (3) stands between them and is
    // closer than tee 4. With a cap of 4 the old code found only the first hooker (`find`), then filled the
    // slots by distance: it kept tee 3 and dropped the second hooker, so its pull vanished from every
    // rollout and from the shield. Now both hookers are kept first; the neighbour is what the cap drops.
    let extra = [(2, 4.5, 0.0, false), (3, 9.2, 0.0, false), (4, 14.0, 0.0, false)];
    let (threats, sim, dropped) = local_scene(4, &extra, &[2, 4]);
    assert!(threats.contains(&2) && threats.contains(&4), "threats {threats:?}");
    assert!(
        !threats.contains(&3),
        "the cap is 4: us, the victim and the two hookers: {threats:?}"
    );
    assert_eq!((sim, dropped), (4, 1));
    // The hookers may exceed a cap that is too small for them: cap 2 keeps us, the victim and both hookers.
    let (threats2, sim2, dropped2) = local_scene(2, &extra, &[2, 4]);
    assert_eq!((sim2, dropped2), (4, 1), "{threats2:?}");
    assert!(threats2.contains(&2) && threats2.contains(&4));
    // Controls: cap 0 and cap 5 keep everybody.
    assert_eq!(local_scene(0, &extra, &[2, 4]).2, 0);
    let (threats5, sim5, dropped5) = local_scene(5, &extra, &[2, 4]);
    assert_eq!((sim5, dropped5), (5, 0), "{threats5:?}");
}

#[test]
fn threats_are_ranked_by_danger_and_distance_only_breaks_ties() {
    // A (id 2) idles 3 tiles away, B (id 3) is 9 tiles away with its hook flying at us, C (id 4) is 12 tiles
    // away running at us, D (id 5) idles 5 tiles away. Order: hook in flight, hammer reach is none here, then
    // closing speed (C), then distance (A before D).
    let extra = [
        (2, 11.0, 0.0, false),
        (3, 17.0, 0.0, true),
        (4, 20.0, -9.0, false),
        (5, 3.5, 0.0, false),
    ];
    let (threats, _, _) = local_scene(0, &extra, &[]);
    assert_eq!(threats[0], 3, "the flying hook first: {threats:?}");
    assert_eq!(threats[1], 4, "then the fastest closing tee: {threats:?}");
    assert!(
        threats.iter().position(|&i| i == 2) < threats.iter().position(|&i| i == 5),
        "{threats:?}"
    );
    // With the cap at 3 (us, victim, one more) the dangerous one is the survivor, not the nearest.
    let (t3, sim, dropped) = local_scene(3, &extra, &[]);
    assert_eq!(sim, 3);
    assert_eq!(dropped, 3);
    assert_eq!(t3[0], 3);
}

/// The hall with a freeze strip on the **front layer** only (x 50..=55, y 9): the game layer has no freeze at all.
fn front_freeze_hall() -> Arc<MapData> {
    let (w, h) = (60usize, 16usize);
    let mut game = vec![Tile::default(); w * h];
    let mut front = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            let solid = y >= 10 || x == 0 || x == w - 1 || y == 0;
            game[y * w + x] = Tile {
                index: if solid { TILE_SOLID } else { 0 },
                ..Tile::default()
            };
            if (50..=55).contains(&x) && y == 9 {
                front[y * w + x] = Tile {
                    index: TILE_FREEZE,
                    ..Tile::default()
                };
            }
        }
    }
    Arc::new(MapData {
        width: w as u32,
        height: h as u32,
        game,
        front: Some(front),
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    })
}

#[test]
fn the_shield_skip_sees_front_layer_freeze_and_the_speed() {
    // Review F5: the skip distance came from the game layer only (blind to front-layer freeze) and ignored speed.
    let map = front_freeze_hall();
    let run = |tee_x: f64, vx: f64| {
        let mut pw = PhysicsWorld::new(map.clone(), 1);
        place(&mut pw, &[(0, tee_x, 9.5), (1, 5.0, 9.5)]);
        for _ in 0..5 {
            pw.step();
        }
        let mut st = pw.get_tee(0).unwrap();
        st.vel = Vec2 { x: vx, y: 0.0 };
        pw.apply_tee_state(0, &st);
        let w = pw.inner().clone();
        let mut b = HybridBrain::new(fixed_cfg(1), ClockKind::Wall, Box::new(NoProposer)).unwrap();
        reset(&mut b, &map, 0, 3);
        let obs = observation(&w, &map, 0, &[0, 1], 1);
        let view = WorldView {
            world: &w,
            self_id: 0,
            lag_ticks: 0,
            in_flight: &[],
        };
        let _ = b.decide_in(&obs, Some(&view));
        b.last_decision().unwrap().shield_skipped
    };
    // 20 tiles from the strip, at rest: far enough, skipped.
    assert!(run(30.0, 0.0), "at rest 20 tiles away the shield should be skipped");
    // 3 tiles beside a front-layer-only strip: the game layer shows no hazard at all, the shield must run.
    assert!(!run(47.0, 0.0), "the shield was skipped beside a front-layer freeze");
    // 20 tiles away but running at it at 12 px/tick (10 tiles in the 27 ticks of a plan): not skippable.
    assert!(!run(30.0, 12.0), "the shield was skipped for a fast tee");
}

#[test]
fn the_shield_skip_field_sees_heart_pickups_as_freeze() {
    // 4.2 merged: a heart pickup freezes in its 3x3 neighbourhood (`is_freeze` of the live backend). The skip
    // field goes through `is_freeze`, so it sees them; the TS-parity `hazard_field` (game layer only) does not.
    let (w, h) = (60usize, 16usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            let solid = y >= 10 || x == 0 || x == w - 1 || y == 0;
            game[y * w + x].index = if solid { TILE_SOLID } else { 0 };
        }
    }
    game[5 * w + 30].index = ddai_physics::map::ENTITY_OFFSET + ddai_physics::map::ENTITY_HEALTH_1;
    let map = Arc::new(MapData {
        width: w as u32,
        height: h as u32,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    });
    let pw = PhysicsWorld::new(map, 1);
    let col = pw.collision();
    let full = ddai_planner::fields::hazard_field_full(col);
    let at = |tx: f64| ddai_planner::fields::hazard_tiles(&full, tx * 32.0 + 16.0, 5.0 * 32.0 + 16.0);
    assert_eq!(at(29.0), 0, "the heart's neighbour tile is a hazard tile");
    assert_eq!(at(27.0), 2);
    let ts = ddai_planner::fields::hazard_field(col);
    assert_eq!(
        ddai_planner::fields::hazard_tiles(&ts, 27.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0),
        i32::MAX,
        "the TS-parity field is blind to it by design"
    );
}

/// A proposer with a (fake) visualisation stream, to check the hybrid brain's side of it: it keeps the
/// outcome it was last told about and only has a frame after a `propose` call, like the fly's.
struct Viz {
    plan: Vec<PlanStep>,
    fresh: bool,
    told: Option<(u32, ProposalOutcome)>,
    frame: Vec<u8>,
}
impl Proposer for Viz {
    fn name(&self) -> &str {
        "viz"
    }
    fn propose(&mut self, ctx: &ProposeCtx<'_>, out: &mut Vec<Vec<PlanStep>>) {
        self.fresh = true;
        let mut p = self.plan.clone();
        p.resize(ctx.steps, *self.plan.last().unwrap());
        out.push(p);
    }
    fn viz_meta(&self) -> Option<String> {
        Some("{\"v\":1}".to_string())
    }
    fn viz_frame(&mut self, tick: u32, outcome: Option<ProposalOutcome>) -> Option<&[u8]> {
        if !std::mem::take(&mut self.fresh) {
            return None;
        }
        let o = outcome?;
        self.told = Some((tick, o));
        self.frame = vec![u8::from(o.chosen)];
        Some(&self.frame)
    }
}

#[test]
fn the_hybrid_hands_the_proposers_stream_the_searchs_verdict_and_never_a_stale_one() {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &[(0, 18.5, 9.5), (1, 30.5, 9.5)]);
    let world = pw.inner().clone();
    let walk_in = vec![
        PlanStep {
            dir: 1,
            jump: 0,
            hook: 0,
            fire: 0,
            aim: 0.0,
        };
        9
    ];
    let mut cfg = fixed_cfg(1);
    cfg.proposals = 1;
    let viz = Viz {
        plan: walk_in,
        fresh: false,
        told: None,
        frame: Vec::new(),
    };
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(viz)).unwrap();
    assert_eq!(b.viz_meta().as_deref(), Some("{\"v\":1}"), "before the search exists");
    reset(&mut b, &map, 0, 3);
    assert!(b.viz_frame(0).is_none(), "nothing decided yet");
    let obs = observation(&world, &map, 0, &[0, 1], 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    for i in 0..3u32 {
        let _ = b.decide_in(&obs, Some(&view));
        let chosen = matches!(
            b.last_decision().unwrap().chosen,
            Some(ddai_planner::hybrid::search::Source::Proposal)
        );
        let frame = b.viz_frame(100 + i).expect("a frame after a decision").to_vec();
        assert_eq!(frame, [u8::from(chosen)]);
        assert!(b.viz_frame(100 + i).is_none(), "one frame per decision");
    }
    assert_eq!(b.viz_meta().as_deref(), Some("{\"v\":1}"), "and once the search exists");
    // A decision that never reached the proposer (no target) gives no frame, however stale `last` is.
    let mut no_target = obs.clone();
    no_target.target_id = None;
    no_target.others.clear();
    let _ = b.decide_in(&no_target, Some(&view));
    assert!(b.viz_frame(200).is_none());
}

#[test]
fn a_viewer_who_subscribes_after_unwatched_decisions_never_gets_an_older_decisions_frame() {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, &[(0, 18.5, 9.5), (1, 30.5, 9.5)]);
    let world = pw.inner().clone();
    let plan = vec![
        PlanStep {
            dir: 1,
            jump: 0,
            hook: 0,
            fire: 0,
            aim: 0.0,
        };
        9
    ];
    let mut cfg = fixed_cfg(1);
    cfg.proposals = 1;
    let viz = Viz {
        plan,
        fresh: false,
        told: None,
        frame: Vec::new(),
    };
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(viz)).unwrap();
    reset(&mut b, &map, 0, 3);
    let obs = observation(&world, &map, 0, &[0, 1], 1);
    let view = WorldView {
        world: &world,
        self_id: 0,
        lag_ticks: 0,
        in_flight: &[],
    };
    let mut no_target = obs.clone();
    no_target.target_id = None;
    no_target.others.clear();
    // The proposer is consulted at decisions nobody watches (no frame is pulled) ...
    for _ in 0..3 {
        let _ = b.decide_in(&obs, Some(&view));
    }
    // ... then the viewer subscribes during a decision that never reached the proposer: no frame, however recent the
    // proposer's last proposal and the search's last verdict are.
    let _ = b.decide_in(&no_target, Some(&view));
    assert!(
        b.viz_frame(1).is_none(),
        "an older decision's frame must not be stamped with this one"
    );
    // The next decision that does consult the proposer gives a frame again.
    let _ = b.decide_in(&obs, Some(&view));
    assert!(b.viz_frame(2).is_some());
    assert!(b.viz_frame(2).is_none(), "once per decision");
}

#[test]
fn a_hybrid_without_a_streaming_proposer_has_no_stream() {
    let mut b = HybridBrain::fixed();
    assert!(b.viz_meta().is_none());
    assert!(b.viz_frame(0).is_none());
}
