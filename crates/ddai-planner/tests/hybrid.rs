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
use ddai_planner::hybrid::{HybridBrain, HybridConfig, HybridMode, NoProposer, ProposeCtx, Proposer, ScriptedProposer};
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
                let tel = hybrid.last_decision().map(|t| {
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
                });
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

#[test]
fn the_decision_cap_shortens_the_search_by_the_shield_reserve_of_many_tees() {
    // Search budget = min(budget, cap - reserve per tee * tees), never below 1 ms; `None` = as asked.
    let budget_of = |cap: Option<f64>, reserve: f64, tees: &[(i32, f64, f64)]| {
        let cfg = HybridConfig {
            mode: HybridMode::Deadline { budget_ms: 4.0 },
            proposals: 0,
            decision_cap_ms: cap,
            shield_reserve_ms_per_tee: reserve,
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
