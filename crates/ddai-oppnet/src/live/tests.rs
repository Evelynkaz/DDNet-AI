use std::sync::Arc;

use ddai_physics::core::PlayerInput as Wire;
use ddai_physics::map::{MapData, TILE_SOLID, Tile};
use ddai_planner::hybrid::window::PredictedInput;
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::empty_input;
use ddai_planner::vmath::Vec2;

use super::*;
use crate::bundle::OppBundle;
use crate::feature::{INPUT_DIM, OUT_DIM};
use crate::live::analyze::Report;
use crate::net::Mlp;
use crate::predictor::OppPredictor;

/// A hall with a floor: two tees stand on it, 4 tiles apart.
fn hall() -> Arc<MapData> {
    let (w, h) = (60usize, 16usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            let solid = y >= 12 || x == 0 || x == w - 1 || y == 0;
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

fn pair() -> PhysicsWorld {
    let mut pw = PhysicsWorld::new(hall(), 1);
    for i in 0..2 {
        pw.add_tee(
            i,
            Vec2 {
                x: (20.5 + 4.0 * f64::from(i)) * 32.0,
                y: 11.0 * 32.0 - 14.0,
            },
        );
    }
    for _ in 0..4 {
        pw.step();
    }
    pw
}

/// A network whose output is the same at every input: tick-wise direction class `dir_class` (0 left, 1 none, 2 right) and the hook flag.
fn constant_model(dir_class: usize, hook: bool, gate: f32) -> OppPredictor {
    let mut net = Mlp::new(INPUT_DIM, 8, 8, OUT_DIM, 1);
    net.params.iter_mut().for_each(|p| *p = 0.0);
    let n = net.params.len();
    for k in 0..8 {
        let o = n - OUT_DIM + k * 7;
        net.params[o + dir_class] = 5.0;
        if hook {
            net.params[o + 4] = 5.0;
        }
    }
    let b = OppBundle::new(net, 1, 1, 0.0, "const".into());
    OppPredictor::new(b, "const").unwrap().with_gate(gate)
}

fn cfg() -> GuardConfig {
    GuardConfig {
        windows: 12,
        min_windows: 6,
        margin: 0.05,
        retry_after: 8,
        retry_margin: 0.0,
    }
}

/// A live predictor with no regime gate: the scripted opponents walk far (the gate has its own test).
fn live(model: impl Into<crate::any::AnyPredictor>) -> LiveOpp {
    LiveOpp::new(model, cfg(), [7; 32])
        .unwrap()
        .with_gate(RegimeGate::off())
}

/// Steps `ticks` ticks of the opponent (tee 1) playing `dir_of(tick)`; after every second tick calls `after(pw)`.
fn drive(pw: &mut PhysicsWorld, ticks: usize, dir_of: impl Fn(i32) -> i32, mut after: impl FnMut(&PhysicsWorld)) {
    for _ in 0..ticks {
        let t = pw.inner().tick;
        let mut i = empty_input();
        i.direction = dir_of(t);
        pw.set_input(1, i);
        pw.set_input(0, empty_input());
        pw.step();
        if pw.inner().tick % 2 == 0 {
            after(pw);
        }
    }
}

fn pair_of<'a>(world: &'a ddai_physics::world::World<f32>, tag: &'a str) -> Pair<'a> {
    Pair {
        world,
        self_id: 0,
        target: 1,
        tag,
    }
}

fn ask(l: &mut LiveOpp, pw: &PhysicsWorld, lag: usize, victim: &mut Vec<Wire>) -> WindowUse {
    let own = vec![Wire::default(); lag];
    l.window(&pair_of(pw.inner(), "c1-aabbccdd"), &own, &Wire::default(), victim)
}

#[test]
fn a_cost_counts_direction_hook_and_a_saturating_aim_error() {
    let hold = Hold {
        dir: 0,
        hook: false,
        aim: 1.0,
    };
    let actual = Actual {
        dir: 1,
        hook: true,
        aim: 1.25,
        jump_event: false,
    };
    // The model is right on the direction and the hook and 0.25 rad off: 0.5 of the aim cost.
    let model = Pm {
        dir: 1,
        hook: true,
        jump: false,
        aim: 1.0,
    };
    let (m, h) = sample_cost(&model, &hold, &actual);
    assert!((m - 0.5).abs() < 1e-6, "{m}");
    assert!(
        (h - (1.0 + 1.0 + 0.5)).abs() < 1e-6,
        "hold misses the direction, the hook and half the aim: {h}"
    );
    // The aim error saturates at 1 and wraps around 2 pi.
    let far = Pm { aim: 4.0, ..model };
    assert!((sample_cost(&far, &hold, &actual).0 - 1.0).abs() < 1e-6);
    let wrap = Actual { aim: 6.2, ..actual };
    let near_zero = Pm { aim: 0.05, ..model };
    assert!((sample_cost(&near_zero, &hold, &wrap).0 - 0.5 * (0.133 / 0.25)).abs() < 0.01);
}

#[test]
fn a_predicted_press_moves_the_fire_counter_to_the_next_odd_value() {
    let hold = Wire {
        fire: 4,
        ..Wire::default()
    };
    let p = |press| PredictedInput {
        direction: -2,
        jump: true,
        hook: true,
        press,
        aim: 0.0,
    };
    let mut fire = hold.fire;
    let a = victim_input(&p(false), &hold, &mut fire);
    assert_eq!(
        (a.fire, a.direction, a.jump, a.hook, a.target_x, a.target_y),
        (4, -1, 1, 1, 300, 0)
    );
    let b = victim_input(&p(true), &hold, &mut fire);
    assert_eq!(b.fire, 5);
    let c = victim_input(&p(true), &hold, &mut fire);
    assert_eq!(c.fire, 7, "release and press in one step");
    assert_eq!(victim_input(&p(false), &hold, &mut fire).fire, 7);
}

#[test]
fn windows_are_scored_against_the_snapshots_that_follow_and_logged() {
    let mut pw = pair();
    // The opponent walks right all the time; the model says "right and hooking", hold shows what the snapshot shows.
    let mut l = live(constant_model(2, false, 0.0));
    let mut victim = Vec::new();
    drive(
        &mut pw,
        40,
        |_| 1,
        |pw| {
            ask(&mut l, pw, 3, &mut victim);
        },
    );
    let log = String::from_utf8(l.take_log()).unwrap();
    let mut r = Report::new();
    for line in log.lines() {
        r.add_line(line);
    }
    assert_eq!(r.bad_lines, 0, "{log}");
    assert!(r.samples() > 40, "{}", r.samples());
    // Windows are 3 ticks long; the snapshots two ticks apart show the odd steps k = 1, 3, 5, 7 of each.
    let ks: Vec<u8> = (0..8).filter(|&k| r.pooled(Some(3), Some(k)).n > 0).collect();
    assert_eq!(ks, [1, 3, 5, 7], "{log}");
    // Once the opponent walks (every window after the first few), the model is right on the direction at every k.
    let k7 = r.pooled(Some(3), Some(7));
    assert!(k7.dir_model > k7.n * 9 / 10, "{k7:?}");
    for line in log.lines().skip(1).take(3) {
        assert!(
            line.contains(r#""o":"c1-aabbccdd""#) && !line.contains("name"),
            "{line}"
        );
    }
}

#[test]
fn a_window_the_model_gets_right_and_hold_gets_wrong_is_logged_as_such() {
    let mut pw = pair();
    let mut good = live(constant_model(0, false, 0.0));
    let mut victim = Vec::new();
    // The opponent walks right until tick 20 and left from then on. The snapshot at tick 20 still shows the right step (19), so hold says
    // "right" for the whole window of that decision, while the constant "left" model is right at every observed tick.
    drive(
        &mut pw,
        40,
        |t| if t < 20 { 1 } else { -1 },
        |pw| {
            ask(&mut good, pw, 3, &mut victim);
        },
    );
    let log = String::from_utf8(good.take_log()).unwrap();
    let line = log
        .lines()
        .find(|l| l.contains(r#""t":20,"#) && l.contains(r#""w":3"#))
        .unwrap_or_else(|| panic!("no window at tick 20 in\n{log}"));
    let v: serde_json::Value = serde_json::from_str(line).unwrap();
    let samples = v["s"].as_array().unwrap();
    assert_eq!(samples.len(), 4, "k = 1, 3, 5, 7: {line}");
    for s in samples {
        // [k, actual dir, model dir, hold dir, ...]
        assert_eq!(
            (s[1].as_i64(), s[2].as_i64(), s[3].as_i64()),
            (Some(-1), Some(-1), Some(1)),
            "{line}"
        );
    }
    let mut r = Report::new();
    r.add_line(line);
    let all = r.pooled(None, None);
    assert_eq!((all.dir_model, all.dir_hold), (4, 0));
    let s = good.guard_status();
    assert!(s.resolved >= 10, "{s:?}");

    // Worse than hold: the opponent walks right at a steady pace, the model says "left".
    let mut pw = pair();
    let mut bad = live(constant_model(0, false, 0.0));
    let mut states = Vec::new();
    drive(
        &mut pw,
        120,
        |_| 1,
        |pw| {
            let how = ask(&mut bad, pw, 3, &mut victim);
            states.push((how, victim.len()));
        },
    );
    let s = bad.guard_status();
    assert_eq!(s.state, GuardState::Fallback, "{s:?}");
    assert!(s.model_cost > s.hold_cost, "{s:?}");
    assert_eq!(s.fallbacks, 1);
    let mut tr = Vec::new();
    bad.take_transitions(&mut tr);
    assert_eq!(tr.len(), 1);
    assert_eq!(tr[0].to, GuardState::Fallback);
    // At the start the model drives (3 inputs for the roll), after the verdict hold does (no inputs), and the model is still called.
    assert_eq!(states.first().copied(), Some((WindowUse::Model, 3)));
    assert_eq!(states.last().copied(), Some((WindowUse::Hold, 0)));
    let c = bad.counts();
    assert!(c.used < c.predicted, "{c:?}");
    assert!(c.predicted >= 50, "the shadow keeps predicting: {c:?}");
    // The transition is in the log, for the analysis.
    let log = String::from_utf8(bad.take_log()).unwrap();
    assert!(
        log.contains(r#""ev":"guard""#) && log.contains(r#""to":"hold""#),
        "{log}"
    );
}

#[test]
fn a_model_that_says_hold_is_exactly_hold_and_never_benched() {
    // A zero network behind a wide gate answers "hold" for the direction and the hook, and the aim change 0: the null model of the 3.15 review.
    let mut pw = pair();
    let mut null = live(constant_model(1, false, 100.0));
    let mut victim = Vec::new();
    drive(
        &mut pw,
        80,
        |t| if (t / 6) % 2 == 0 { 1 } else { -1 },
        |pw| {
            ask(&mut null, pw, 4, &mut victim);
        },
    );
    let s = null.guard_status();
    assert!(s.resolved > 20);
    assert!((s.model_cost - s.hold_cost).abs() < 1e-6, "{s:?}");
    assert_eq!(s.state, GuardState::Active);
    assert_eq!(s.fallbacks, 0);
}

#[test]
fn windows_the_model_cannot_serve_are_held_and_counted() {
    let mut pw = pair();
    let mut l = live(constant_model(2, false, 0.0));
    let mut victim = vec![Wire::default(); 5];
    drive(&mut pw, 4, |_| 0, |_| {});
    assert_eq!(
        ask(&mut l, &pw, 0, &mut victim),
        WindowUse::Hold,
        "an empty window: nothing to roll"
    );
    assert!(victim.is_empty(), "the buffer is cleared on every call");
    assert_eq!(
        ask(&mut l, &pw, WINDOW_MAX + 1, &mut victim),
        WindowUse::Hold,
        "longer than the model knows"
    );
    assert_eq!(l.counts().skipped_window, 2);
    assert_eq!(l.counts().predicted, 0);
    assert_eq!(ask(&mut l, &pw, WINDOW_MAX, &mut victim), WindowUse::Model);
    assert_eq!(victim.len(), WINDOW_MAX);
}

#[test]
fn a_frozen_opponent_is_not_predicted_unless_it_thaws_inside_the_window() {
    let mut pw = pair();
    drive(&mut pw, 6, |_| 0, |_| {});
    let mut l = live(constant_model(2, false, 0.0));
    let mut victim = Vec::new();
    let freeze = |ticks: i16| {
        let mut w = pw.inner().clone();
        w.characters[1].as_mut().unwrap().freeze_time = i32::from(ticks);
        w
    };
    // Frozen for 100 more ticks: its inputs do not exist for this window.
    let long = freeze(100);
    let own = vec![Wire::default(); 3];
    let how = l.window(&pair_of(&long, "c1-aabbccdd"), &own, &Wire::default(), &mut victim);
    assert_eq!((how, victim.len()), (WindowUse::Hold, 0));
    assert_eq!((l.counts().skipped_frozen, l.counts().predicted), (1, 0));
    // Thawing at the third tick of a 3-tick window: the model is asked, and the window is not scored (it starts frozen).
    let soon = freeze(3);
    let how = l.window(&pair_of(&soon, "c1-aabbccdd"), &own, &Wire::default(), &mut victim);
    assert_eq!((how, victim.len()), (WindowUse::Model, 3));
    assert_eq!(l.counts().predicted, 1);
}

#[test]
fn a_dead_opponent_gets_no_prediction_even_when_the_same_tick_is_looked_at_twice() {
    let mut pw = pair();
    drive(&mut pw, 6, |_| 0, |_| {});
    let mut dead = pw.inner().clone();
    dead.characters[1].as_mut().unwrap().alive = false;
    let mut l = live(constant_model(2, false, 0.0));
    let mut victim = Vec::new();
    let pair = pair_of(&dead, "c1-aabbccdd");
    l.observe(&pair);
    let own = vec![Wire::default(); 3];
    assert_eq!(l.window(&pair, &own, &Wire::default(), &mut victim), WindowUse::Hold);
    assert_eq!((l.counts().predicted, l.counts().observed), (0, 0));
}

#[test]
fn another_target_or_a_dead_one_closes_the_open_windows_and_forgets_the_history() {
    let mut pw = pair();
    let mut l = live(constant_model(2, false, 0.0));
    let mut victim = Vec::new();
    drive(
        &mut pw,
        12,
        |_| 1,
        |pw| {
            ask(&mut l, pw, 3, &mut victim);
        },
    );
    let before = l.guard_status().resolved;
    // The same slot, another player (a new tag): what was open is closed with the samples it has, nothing is resolved against the newcomer.
    let own = vec![Wire::default(); 3];
    l.window(&pair_of(pw.inner(), "c1-11223344"), &own, &Wire::default(), &mut victim);
    assert!(l.guard_status().resolved >= before);
    let log = String::from_utf8(l.take_log()).unwrap();
    assert!(
        log.lines().all(|line| !line.contains("11223344")),
        "windows of the old player keep the old tag"
    );
}

#[test]
fn decisions_do_not_allocate_after_the_first() {
    let mut pw = pair();
    let mut l = live(constant_model(2, true, 0.0));
    let mut victim = Vec::with_capacity(16);
    let mut worlds = Vec::new();
    drive(&mut pw, 80, |_| 1, |pw| worlds.push(pw.inner().clone()));
    let own = vec![Wire::default(); 3];
    for w in &worlds[..20] {
        l.window(&pair_of(w, "c1-aabbccdd"), &own, &Wire::default(), &mut victim);
    }
    // New ticks every time: the snapshot is observed (open windows resolved, scored, logged, the guard fed) and the window predicted.
    let info = allocation_counter::measure(|| {
        for w in &worlds[20..] {
            l.window(&pair_of(w, "c1-aabbccdd"), &own, &Wire::default(), &mut victim);
        }
    });
    assert_eq!(info.count_total, 0, "{info:?}");
    assert!(
        l.guard_status().resolved > 20,
        "the guard was fed inside the measured loop too"
    );
}

#[test]
fn the_header_names_the_model_and_the_guard() {
    let l = live(constant_model(1, false, 0.0));
    let h = l.header_line(1_791_000_000, "127.0.0.1:8303\"x,Nick");
    assert!(
        h.ends_with('\n') && h.contains(r#""ev":"open""#) && h.contains(&"07".repeat(32)),
        "{h}"
    );
    let v: serde_json::Value = serde_json::from_str(h.trim()).unwrap();
    assert_eq!(v["guard"]["windows"], 12);
    assert_eq!(v["start"], 1_791_000_000u64);
    assert_eq!(
        v["server"], "127.0.0.1:8303xNick",
        "quotes and commas cannot get through"
    );
    assert_eq!(v["gate"]["on"], false, "the test predictor has no gate");
}

#[test]
fn a_frozen_self_opens_no_window_and_scores_nothing() {
    let mut pw = pair();
    drive(&mut pw, 6, |_| 0, |_| {});
    let mut l = live(constant_model(2, false, 0.0));
    let mut victim = Vec::new();
    let mut me_frozen = pw.inner().clone();
    me_frozen.characters[0].as_mut().unwrap().freeze_time = 100;
    let own = vec![Wire::default(); 3];
    let how = l.window(&pair_of(&me_frozen, "c1-aabbccdd"), &own, &Wire::default(), &mut victim);
    assert_eq!((how, victim.len()), (WindowUse::Hold, 0));
    assert_eq!((l.counts().skipped_own_frozen, l.counts().predicted), (1, 0));
    // A free self opens a window; the snapshot two ticks later shows us frozen (that sample is not scored), the next ones show us free.
    drive(&mut pw, 2, |_| 1, |_| {});
    assert_eq!(ask(&mut l, &pw, 3, &mut victim), WindowUse::Model);
    let t0 = pw.inner().tick;
    for step in 1..=4 {
        drive(&mut pw, 2, |_| 1, |_| {});
        let mut w = pw.inner().clone();
        if step == 1 {
            w.characters[0].as_mut().unwrap().freeze_time = 100;
        }
        l.observe(&pair_of(&w, "c1-aabbccdd"));
    }
    assert_eq!(pw.inner().tick, t0 + 8);
    let log = String::from_utf8(l.take_log()).unwrap();
    let line = log.lines().next().expect("the window was finished");
    let v: serde_json::Value = serde_json::from_str(line).unwrap();
    let ks: Vec<i64> = v["s"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s[0].as_i64().unwrap())
        .collect();
    assert_eq!(ks, [3, 5, 7], "k = 1 was seen while we were frozen: not scored: {line}");
}

#[test]
fn the_guard_judges_only_the_ticks_the_roll_uses_and_the_log_keeps_all() {
    // The opponent walks right all the time. The model says "left" at every tick: wrong everywhere. Hold is right everywhere after the first
    // tick. With a window of 3 the guard sees only k = 1 per window; the log has k = 1, 3, 5, 7.
    let mut pw = pair();
    let mut l = live(constant_model(0, false, 0.0));
    let mut victim = Vec::new();
    drive(
        &mut pw,
        60,
        |_| 1,
        |pw| {
            ask(&mut l, pw, 3, &mut victim);
        },
    );
    let status = l.guard_status();
    let log = String::from_utf8(l.take_log()).unwrap();
    let mut r = Report::new();
    for line in log.lines() {
        r.add_line(line);
    }
    let ks: Vec<u8> = (0..8).filter(|&k| r.pooled(Some(3), Some(k)).n > 0).collect();
    assert_eq!(ks, [1, 3, 5, 7], "the log keeps every tick");
    // Per resolved window the guard got one sample (k = 1): per-sample costs equal the cost of the k = 1 samples alone.
    let k1 = r.pooled(Some(3), Some(1));
    let per_sample_model = (k1.n - k1.dir_model) as f32 / k1.n as f32; // the direction miss of the model at k = 1 is 1 per sample here
    assert!(
        status.model_cost >= per_sample_model && status.model_cost < 3.0,
        "{status:?}"
    );
    assert_eq!(super::guard_ticks(3), 3);
    assert_eq!(
        super::guard_ticks(1),
        2,
        "a window of one tick is judged on k = 1, the only tick a snapshot shows"
    );
}

#[test]
fn the_regime_gate_admits_the_duel_and_refuses_a_crowd_or_a_far_target() {
    let mut pw = pair();
    drive(&mut pw, 6, |_| 0, |_| {});
    let gate = RegimeGate::default();
    let me = |w: &ddai_physics::world::World<f32>, id: u8| {
        let c = w.cores.get(id).unwrap();
        [c.pos.x, c.pos.y]
    };
    let w = pw.inner();
    assert!(gate.admits(w, 0, 1, me(w, 0), me(w, 1)), "two tees 4 tiles apart");
    // A third tee 10 tiles away is a crowd for the model.
    pw.add_tee(
        2,
        Vec2 {
            x: 30.5 * 32.0,
            y: 11.0 * 32.0 - 14.0,
        },
    );
    for _ in 0..2 {
        pw.step();
    }
    let w = pw.inner();
    assert!(!gate.admits(w, 0, 1, me(w, 0), me(w, 1)), "a third tee near");
    assert!(RegimeGate::off().admits(w, 0, 1, me(w, 0), me(w, 1)));
    // A target far away is refused.
    assert!(!gate.admits(w, 0, 1, [0.0, 0.0], [1000.0, 0.0]));
    // And the live decision counts it.
    let mut l = live(constant_model(2, false, 0.0)).with_gate(RegimeGate::default());
    let mut victim = Vec::new();
    assert_eq!(ask(&mut l, &pw, 3, &mut victim), WindowUse::Hold);
    assert_eq!((l.counts().skipped_regime, l.counts().predicted), (1, 0));
    let mut off = live(constant_model(2, false, 0.0)).with_gate(RegimeGate::off());
    assert_eq!(ask(&mut off, &pw, 3, &mut victim), WindowUse::Model);
}

/// Task 3.21: a v2 network whose output is the same at every input (direction class, hook).
fn constant_v2(dir_class: usize, hook: bool) -> crate::v2::predictor::Predictor {
    use crate::v2::feature::{HEAD_DIM, HORIZON, INPUT_DIM, OUT_DIM};
    use crate::v2::predictor::{Bundle, Decode, Predictor};
    let mut net = Mlp::new(INPUT_DIM, 8, 8, OUT_DIM, 1);
    net.params.iter_mut().for_each(|p| *p = 0.0);
    let n = net.params.len();
    for k in 0..HORIZON {
        let o = n - OUT_DIM + k * HEAD_DIM;
        net.params[o + dir_class] = 5.0;
        if hook {
            net.params[o + 4] = 5.0;
        }
    }
    Predictor::new(
        Bundle::new(net, Decode::default(), 1, 1, 0.0, "const2".into()),
        "const2",
    )
    .unwrap()
}

#[test]
fn a_v2_model_serves_windows_up_to_its_own_length_and_does_not_allocate() {
    let mut pw = pair();
    let mut l = live(constant_v2(2, true));
    let mut victim = vec![Wire::default(); 5];
    drive(&mut pw, 4, |_| 0, |_| {});
    // A window of 5 is longer than the v2 model knows (4) although the v1 model would serve it.
    assert_eq!(ask(&mut l, &pw, 5, &mut victim), WindowUse::Hold);
    assert_eq!(l.counts().skipped_window, 1);
    assert_eq!(ask(&mut l, &pw, 2, &mut victim), WindowUse::Model);
    assert_eq!(victim.len(), 2);
    assert!(
        victim.iter().all(|w| w.direction == 1 && w.hook == 1),
        "the constant model's inputs are played"
    );
    // Known pre-inputs are used by one window only and never break the pipeline; a v1 model ignores them.
    l.set_known(&[Some(Wire {
        direction: -1,
        ..Wire::default()
    })]);
    assert_eq!(ask(&mut l, &pw, 2, &mut victim), WindowUse::Model);
    let mut worlds = Vec::new();
    drive(&mut pw, 80, |_| 1, |pw| worlds.push(pw.inner().clone()));
    let own = vec![Wire::default(); 2];
    for w in &worlds[..20] {
        l.window(&pair_of(w, "c1-aabbccdd"), &own, &Wire::default(), &mut victim);
    }
    let known = [Some(Wire::default()), None];
    let info = allocation_counter::measure(|| {
        for w in &worlds[20..] {
            l.set_known(&known);
            l.window(&pair_of(w, "c1-aabbccdd"), &own, &Wire::default(), &mut victim);
        }
    });
    assert_eq!(info.count_total, 0, "{info:?}");
    let mut v1 = live(constant_model(2, true, 0.0));
    v1.set_known(&known);
    assert_eq!(
        ask(&mut v1, &pw, 2, &mut victim),
        WindowUse::Model,
        "v1 ignores known ticks"
    );
}
