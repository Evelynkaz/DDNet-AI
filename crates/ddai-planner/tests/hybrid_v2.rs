//! The competitor's current planner (upstream af49dfb, "v2") inside the hybrid (task 3.9, D-096, E-020): every switch is off by
//! default, and with them on the hybrid stays deterministic on the work clock for any number of workers, the polish and the wall
//! throws add the candidates they promise, and the chosen input is the one the rollouts scored.

use std::sync::Arc;

use ddai_brain::{Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::World;
use ddai_planner::brains::{ClockKind, ScriptedBrain, input_from_action};
use ddai_planner::config::{PlannerConfig, PlannerVersion, preset_live_v2, preset_normal_v2};
use ddai_planner::fields::{hazard_field, unfreeze_field};
use ddai_planner::hybrid::engine::{Batch, Ctx, Engine};
use ddai_planner::hybrid::{DecisionTelemetry, HybridBrain, HybridConfig, HybridMode, NoProposer};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::planner::PlanStep;
use ddai_planner::types::{HOOK_GRABBED, PlayerInput, empty_input};
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

/// The v2 planner switches of af49dfb on the hybrid's scoring preset.
fn v2_planner() -> PlannerConfig {
    HybridConfig::default()
        .planner
        .with_version(PlannerVersion::Upstream20261002)
}

/// Every task 3.9 switch on.
fn all_on(c: &mut HybridConfig) {
    c.planner = v2_planner();
    c.planner.air_chain = true;
    c.polish = true;
    c.wall_throws = true;
    c.mirror_planner = Some(preset_normal_v2());
}

const TWO: [(i32, f64, f64); 2] = [(0, 17.5, 9.5), (1, 24.5, 9.5)];
const FOUR: [(i32, f64, f64); 4] = [(0, 17.5, 9.5), (1, 24.5, 9.5), (2, 27.5, 9.5), (3, 30.5, 9.5)];

/// Plays `decisions` decisions (every 2 ticks) of slot 0 = the hybrid against scripted attackers; the record of each.
fn drive(
    cfg: HybridConfig,
    tees: &[(i32, f64, f64)],
    decisions: usize,
    record: impl Fn(&DecisionTelemetry) -> String,
) -> Vec<(ddai_brain::Action, String)> {
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, tees);
    let ids: Vec<i32> = tees.iter().map(|t| t.0).collect();
    let mut hybrid = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).expect("valid config");
    hybrid.reset(&ResetContext {
        map: map.clone(),
        self_id: 0,
        seed: 7,
    });
    let mut scripted: Vec<ScriptedBrain> = ids.iter().skip(1).map(|_| ScriptedBrain::new()).collect();
    for (k, b) in scripted.iter_mut().enumerate() {
        b.reset(&ResetContext {
            map: map.clone(),
            self_id: (k + 1) as i32,
            seed: 100 + k as u64,
        });
    }
    let mut last: Vec<PlayerInput> = ids.iter().map(|_| empty_input()).collect();
    let mut log = Vec::new();
    for _ in 0..decisions {
        let world = pw.inner().clone();
        for (slot, &id) in ids.iter().enumerate() {
            if world.cores.get(id as u8).is_none() {
                continue;
            }
            let target = if slot == 0 { ids[1] } else { 0 };
            let obs = observation(&world, &map, id, &ids, target);
            let view = WorldView {
                world: &world,
                self_id: id,
                lag_ticks: 0,
                in_flight: &[],
            };
            let action = if slot == 0 {
                let a = hybrid.decide_in(&obs, Some(&view));
                log.push((a, hybrid.last_decision().map(&record).unwrap_or_default()));
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

fn work_cfg(workers: usize) -> HybridConfig {
    HybridConfig {
        mode: HybridMode::Deadline { budget_ms: 4.0 },
        proposals: 0,
        workers,
        work_clock_us_per_tick: Some(1.25),
        ..HybridConfig::default()
    }
}

#[test]
fn every_switch_is_off_by_default() {
    let c = HybridConfig::default();
    assert!(!c.polish && !c.wall_throws && c.mirror_planner.is_none());
    assert_eq!(c.planner.version(), PlannerVersion::Classic);
    assert!(!c.planner.hook_exact_gate && !c.planner.hook_snap_aim && !c.planner.hook_keep_flying);
    assert_eq!(c.planner.rope_ceiling_cost, 0.0);
    assert!(!c.planner.air_chain && c.planner.wall_dir == 0);
    // The competitor's live scoring values are not the hybrid's defaults.
    assert_ne!(preset_live_v2().launch_exposure, c.planner.launch_exposure);
    assert!(c.validate().is_ok());
    let mut on = HybridConfig::default();
    all_on(&mut on);
    assert!(on.validate().is_ok());
    assert_eq!(on.planner.version(), PlannerVersion::Upstream20261002);
}

#[test]
fn work_clock_decisions_with_the_v2_switches_are_bit_identical_for_1_and_4_workers() {
    type Tweak = fn(&mut HybridConfig);
    let variants: [(&str, Tweak); 8] = [
        ("gate and snap", |c| {
            c.planner.hook_exact_gate = true;
            c.planner.hook_snap_aim = true;
        }),
        ("rope ceiling", |c| c.planner.rope_ceiling_cost = 1.0),
        ("polish", |c| {
            c.polish = true;
            c.planner.hook_keep_flying = true;
        }),
        ("wall throws and air chains", |c| {
            c.wall_throws = true;
            c.planner.air_chain = true;
        }),
        ("mirror on v2", |c| c.mirror_planner = Some(preset_normal_v2())),
        ("live scoring values", |c| {
            c.planner.launch_exposure = 1.5;
            c.planner.jumpless_hazard_cost = 0.4;
        }),
        ("all", all_on),
        ("all, 12 ms", |c| {
            all_on(c);
            c.mode = HybridMode::Deadline { budget_ms: 12.0 };
            c.decision_cap_ms = None;
        }),
    ];
    let mut hits = 0u32;
    for (name, tweak) in variants {
        for scene in [&TWO[..], &FOUR[..]] {
            let run = |workers: usize| {
                let used = std::cell::Cell::new(0u32);
                let mut cfg = work_cfg(workers);
                tweak(&mut cfg);
                let log = drive(cfg, scene, 24, |t| {
                    used.set(used.get() + t.spec_used);
                    t.to_json()
                });
                (log, used.get())
            };
            let (one, _) = run(1);
            let (four, used) = run(4);
            hits += used;
            assert_eq!(one.len(), 24);
            if let Some(k) = one.iter().zip(&four).position(|(a, b)| a != b) {
                panic!(
                    "{name}, {} tees: 4 workers decided differently from 1 at decision {k}:\n  1: {:?}\n  4: {:?}",
                    scene.len(),
                    one[k],
                    four[k]
                );
            }
        }
    }
    assert!(
        hits > 50,
        "the helpers must have supplied rollouts, else this proves nothing: {hits}"
    );
}

/// One fixed-work decision (no clock: CEM runs its 20 x 2 samples whatever they cost) of slot 0.
fn fixed_decision(
    tees: &[(i32, f64, f64)],
    freeze_victim: bool,
    tweak: impl FnOnce(&mut HybridConfig),
) -> DecisionTelemetry {
    fixed_decision_seeded(tees, freeze_victim, 3, tweak)
}

fn fixed_decision_seeded(
    tees: &[(i32, f64, f64)],
    freeze_victim: bool,
    seed: u64,
    tweak: impl FnOnce(&mut HybridConfig),
) -> DecisionTelemetry {
    let mut cfg = HybridConfig::fixed();
    cfg.proposals = 0;
    tweak(&mut cfg);
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, tees);
    if freeze_victim {
        pw.inner_mut().characters[1].as_mut().expect("victim").freeze_time = 300;
    }
    let world = pw.inner().clone();
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    b.reset(&ResetContext {
        map: map.clone(),
        self_id: 0,
        seed,
    });
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

const CEM: usize = 4;
const THROW: usize = 3;

#[test]
fn polish_adds_the_hook_variants_of_the_best_plan_only_where_a_hook_can_matter() {
    // `polishRope` tries k = 2, 4, all steps: up to three more candidates, and only when the hook gate lets the throw through. The
    // victim straight above us, where our aim already points (the first aim is straight up and the aim smoothing limits a turn).
    let above = [(0, 17.5, 9.5), (1, 17.5, 6.5)];
    let (mut offered, mut decisions) = (0, 0);
    for seed in 1..=16u64 {
        let off = fixed_decision_seeded(&above, false, seed, |c| c.mirror = false);
        let on = fixed_decision_seeded(&above, false, seed, |c| {
            c.mirror = false;
            c.polish = true;
        });
        let extra = on.generated[CEM] - off.generated[CEM];
        assert!(extra <= 3, "seed {seed}: polish added {extra} candidates");
        assert_eq!(
            on.evaluated[CEM] - off.evaluated[CEM],
            extra,
            "seed {seed}: and they are scored"
        );
        assert_eq!(
            on.generated[..CEM],
            off.generated[..CEM],
            "seed {seed}: nothing else changes"
        );
        // The fire counter shows the switch working (and is absent from the JSON of a hybrid without it).
        assert_eq!(on.polished, extra, "seed {seed}: fire counter");
        assert_eq!(off.polished, 0);
        assert_eq!(on.to_json().contains("\"polish\":"), extra > 0);
        assert!(!off.to_json().contains("\"polish\":") && !off.to_json().contains("\"wall\":"));
        offered += usize::from(extra > 0);
        decisions += 1;
    }
    assert!(
        offered >= 4,
        "the variants must be offered in a good share of {decisions} decisions: {offered}"
    );
    // The victim 28 tiles away is out of the hook's reach: nothing to polish.
    let far = [(0, 5.5, 9.5), (1, 33.5, 9.5)];
    for seed in 1..=4u64 {
        let far_off = fixed_decision_seeded(&far, false, seed, |c| c.mirror = false);
        let far_on = fixed_decision_seeded(&far, false, seed, |c| {
            c.mirror = false;
            c.polish = true;
        });
        assert_eq!(far_on.generated, far_off.generated);
    }
}

#[test]
fn polish_runs_on_a_deadline_too_not_only_in_fixed_work() {
    // The first version polished only while stage 1 had time left, which a CEM that runs to the end of it never leaves: on the work clock
    // not one decision got a variant (E-020 screening). Here: the victim straight above us, 4 ms of work, the variants appear.
    let above = [(0, 17.5, 9.5), (1, 17.5, 6.5)];
    let (mut offered, mut total) = (0u32, 0u32);
    for seed in 1..=12u64 {
        let run = |polish: bool| {
            let mut cfg = work_cfg(1);
            cfg.mirror = false;
            cfg.polish = polish;
            let map = hall();
            let mut pw = PhysicsWorld::new(map.clone(), 1);
            place(&mut pw, &above);
            let world = pw.inner().clone();
            let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
            b.reset(&ResetContext {
                map: map.clone(),
                self_id: 0,
                seed,
            });
            let obs = observation(&world, &map, 0, &[0, 1], 1);
            let view = WorldView {
                world: &world,
                self_id: 0,
                lag_ticks: 0,
                in_flight: &[],
            };
            let _ = b.decide_in(&obs, Some(&view));
            b.last_decision().expect("telemetry").generated[CEM]
        };
        let extra = run(true) - run(false);
        assert!(extra <= 3);
        offered += u32::from(extra > 0);
        total += 1;
    }
    assert!(
        offered >= 4,
        "variants offered on the work clock in {offered} of {total} decisions"
    );
}

#[test]
fn wall_throws_offer_swings_off_a_wall_beside_us_to_a_frozen_victim() {
    // Us 3 tiles from the pillar on the left (a wall within 5 tiles), the frozen victim 7 tiles away (within the hook's reach).
    let scene = [(0, 7.5, 9.5), (1, 14.5, 9.5)];
    let base = |c: &mut HybridConfig| {
        c.mirror = false;
        c.throw_cap = 4;
    };
    let off = fixed_decision(&scene, true, base);
    let on = fixed_decision(&scene, true, |c| {
        base(c);
        c.wall_throws = true;
    });
    assert_eq!(
        off.generated[THROW], 4,
        "the frozen victim gets the throw lines, capped"
    );
    let extra = on.generated[THROW] - off.generated[THROW];
    // Two jump steps x three releases = six wall swings, none of them one of the frozen throw lines.
    assert_eq!(extra, 6, "wall swings offered: {extra}");
    assert_eq!((on.wall_cands, off.wall_cands), (6, 0), "fire counter");
    assert!(on.to_json().contains("\"wall\":6") && !off.to_json().contains("\"wall\":"));
    // Air chains need the planner's `air_chain` and an airborne tee: we stand on the floor, so none.
    let chains = fixed_decision(&scene, true, |c| {
        base(c);
        c.wall_throws = true;
        c.planner.air_chain = true;
    });
    assert_eq!(chains.generated[THROW], on.generated[THROW]);
    // No wall within reach (mid-hall, the pillar 14 tiles off): nothing extra.
    let open = [(0, 17.5, 9.5), (1, 24.5, 9.5)];
    let open_off = fixed_decision(&open, true, base);
    let open_on = fixed_decision(&open, true, |c| {
        base(c);
        c.wall_throws = true;
    });
    assert_eq!(open_on.generated, open_off.generated);
    // A victim that is not frozen is not thrown at with wall swings.
    let live_on = fixed_decision(&scene, false, |c| {
        base(c);
        c.wall_throws = true;
    });
    let live_off = fixed_decision(&scene, false, base);
    assert_eq!(live_on.generated, live_off.generated);
}

/// `fixed_decision` with the live bot's wayblock hint (`LiveContext::wb`) set before the decision: `hall` = we stand in the held hall, `wall_dir` = the
/// side of its freeze wall (task 3.18).
fn fixed_decision_hinted(
    tees: &[(i32, f64, f64)],
    freeze_victim: bool,
    hall_hint: Option<i32>,
    tweak: impl FnOnce(&mut HybridConfig),
) -> DecisionTelemetry {
    let mut cfg = HybridConfig::fixed();
    cfg.proposals = 0;
    tweak(&mut cfg);
    let map = hall();
    let mut pw = PhysicsWorld::new(map.clone(), 1);
    place(&mut pw, tees);
    if freeze_victim {
        pw.inner_mut().characters[1].as_mut().expect("victim").freeze_time = 300;
    }
    let world = pw.inner().clone();
    let mut b = HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap();
    b.reset(&ResetContext {
        map: map.clone(),
        self_id: 0,
        seed: 3,
    });
    if let Some(wall_dir) = hall_hint {
        b.set_live_context(&ddai_brain::LiveContext {
            wb: ddai_brain::WbHints {
                in_hall: true,
                wall_dir,
                ..Default::default()
            },
            ..Default::default()
        });
    }
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

#[test]
fn wb_hold_throws_a_frozen_victim_toward_the_halls_wall_only_with_the_hint_and_the_switch() {
    // Mid-hall, no wall within 5 tiles (the 3.9 rule finds none), the frozen victim 7 tiles away: only the hall's own side can offer the swings.
    let open = [(0, 17.5, 9.5), (1, 24.5, 9.5)];
    let base = |c: &mut HybridConfig| {
        c.mirror = false;
        c.throw_cap = 4;
    };
    let plain = fixed_decision(&open, true, base);
    let on = fixed_decision_hinted(&open, true, Some(-1), |c| {
        base(c);
        c.wb_hold = true;
    });
    assert_eq!(
        on.generated[THROW] - plain.generated[THROW],
        6,
        "two jump steps x three releases"
    );
    assert_eq!((on.wall_cands, plain.wall_cands), (6, 0));
    // The right hall's wall side (+1) offers the same number, the mirrored lines.
    let right = fixed_decision_hinted(&open, true, Some(1), |c| {
        base(c);
        c.wb_hold = true;
    });
    assert_eq!(right.wall_cands, 6);
    // The switch off: the hint alone changes nothing (the default hybrid ignores `WbHints`, as it always did).
    let hint_only = fixed_decision_hinted(&open, true, Some(-1), base);
    assert_eq!(hint_only.generated, plain.generated);
    // The switch on without a hint (outside the hall, the lower shelf: `wall_dir` 0) changes nothing either.
    let no_hint = fixed_decision(&open, true, |c| {
        base(c);
        c.wb_hold = true;
    });
    assert_eq!(no_hint.generated, plain.generated);
    let zero = fixed_decision_hinted(&open, true, Some(0), |c| {
        base(c);
        c.wb_hold = true;
    });
    assert_eq!(zero.generated, plain.generated);
    // Only against a frozen victim.
    let free = fixed_decision_hinted(&open, false, Some(-1), |c| {
        base(c);
        c.wb_hold = true;
    });
    let free_plain = fixed_decision(&open, false, base);
    assert_eq!(free.generated, free_plain.generated);
}

#[test]
fn the_default_hybrid_ignores_the_v2_fields_it_does_not_use() {
    // Switches off = the 3.7b hybrid: the same work-clock games, with a mirror planner of `None` or an explicit `preset_normal`
    // (what `None` means) -- and a v2 planner config that changes nothing the decision reads (`wall_dir` is the competitor's own).
    let run = |tweak: fn(&mut HybridConfig)| {
        let mut cfg = work_cfg(1);
        tweak(&mut cfg);
        drive(cfg, &TWO, 24, |t| t.to_json())
    };
    let base = run(|_| ());
    let explicit = run(|c| c.mirror_planner = Some(ddai_planner::config::preset_normal()));
    assert_eq!(base, explicit);
}

/// `hall()` with a ceiling of freeze over x 10..=20 at row 6.
fn hall_with_freeze_ceiling() -> Arc<MapData> {
    let mut m = (*hall()).clone();
    for x in 10..=20 {
        m.game[6 * 40 + x] = Tile {
            index: TILE_FREEZE,
            ..Tile::default()
        };
    }
    Arc::new(m)
}

#[test]
fn the_rope_ceiling_cost_reaches_the_workers_rollouts() {
    // Tee 1 above us holds its hook on us: it hauls us up toward the freeze ceiling. The same rollout scores lower with
    // `rope_ceiling_cost` on, in the worker's planner (the ceiling field is the engine's to hand over, per map).
    let map = hall_with_freeze_ceiling();
    let score = |cost: f64| {
        let mut pw = PhysicsWorld::new(map.clone(), 1);
        place(&mut pw, &[(0, 15.5, 8.5), (1, 15.5, 4.5)]);
        let mut en = pw.get_tee(1).unwrap();
        en.hook_state = HOOK_GRABBED;
        en.hooked_player = 0;
        pw.apply_tee_state(1, &en);
        let mut cfg = HybridConfig::fixed();
        cfg.planner.rope_ceiling_cost = cost;
        let holding = PlayerInput {
            hook: 1,
            ..empty_input()
        };
        let ctx = Box::new(Ctx {
            generation: 1,
            saved: Box::new(pw.save_state()),
            self_id: 0,
            victim_id: 1,
            prev: empty_input(),
            victim_input: holding,
            victim_plan: vec![],
            opp_seed: 7,
            field: Arc::new(hazard_field(pw.collision())),
            unfreeze: Arc::new(unfreeze_field(pw.collision())),
            frozen_bystanders: vec![],
            frozen_bystander_vels: vec![],
            spares: vec![],
            spare_vels: vec![],
            travel_goal: None,
            threats: None,
            self_freeze_bias: 1.0,
            steps: 0,
            counter: false,
        });
        let mut engine = Engine::new(&cfg, &pw, ctx, Arc::new(ddai_planner::clock::WallClock::new()));
        let mut batch = Batch::default();
        batch.clear(9);
        let wait = [PlanStep {
            dir: 0,
            jump: 0,
            hook: 0,
            fire: 0,
            aim: 0.0,
        }; 9];
        let pi = batch.push_plan(&wait);
        batch.push_job(pi, 0);
        let mut out = Vec::new();
        engine.evaluate(&mut batch, &ddai_planner::clock::WallClock::new(), None, &mut out);
        out[0].res.expect("scored").score
    };
    let (off, on) = (score(0.0), score(1.0));
    assert!(on < off, "hauled up into the ceiling costs: {on} vs {off}");
    assert!(off - on <= 27.0 + 1e-9, "at most the cost per tick");
}

#[test]
fn the_gates_projections_of_a_moving_victim_are_priced_on_the_work_clock() {
    let units = |tweak: fn(&mut HybridConfig)| -> (u64, bool) {
        let mut cfg = work_cfg(1);
        tweak(&mut cfg);
        let log = drive(cfg, &TWO, 40, |t| {
            format!("{}|{}", t.work.units, t.to_json().contains("\"units\":"))
        });
        let total: u64 = log
            .iter()
            .map(|(_, r)| r.split('|').next().unwrap().parse::<u64>().unwrap())
            .sum();
        let in_json = log.iter().any(|(_, r)| r.ends_with("true"));
        (total, in_json)
    };
    assert_eq!(
        units(|_| ()),
        (0, false),
        "off: nothing priced, the JSON is the old one"
    );
    let (gate, json) = units(|c| {
        c.planner.hook_exact_gate = true;
        c.planner.hook_snap_aim = true;
    });
    assert!(
        gate > 0 && json,
        "gate and snap: the projections are charged ({gate} tee-ticks) and reported"
    );
    let (mirror, _) = units(|c| c.mirror_planner = Some(preset_normal_v2()));
    assert!(mirror > 0, "the opponent model's v2 planner is charged too ({mirror})");
}
