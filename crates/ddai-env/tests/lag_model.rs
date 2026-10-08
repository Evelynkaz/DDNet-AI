//! Task 3.16 (D-115): the arena's input-lag model (`PlayerSpec::lag_model`, `ddai_env::sim::LagModel`). A decision is delivered
//! `max(planned, ceil((base + extra + cost) / 20 ms) - 1)` ticks after it was made, where `planned` comes from the rolling p90 of the costs
//! the brain reported before; the brain is told `planned` as its lag. Checked on a stub brain whose costs are scripted, against the tick
//! the physics world really shows the new input on.

use std::path::Path;
use std::sync::{Arc, Mutex};

use ddai_brain::{Action, Brain, Observation, PlanTelemetry, ResetContext, WorldView};
use ddai_env::arena::{Arena, ArenaDef};
use ddai_env::config::{LagModelSpec, PlayerSpec, RunConfig, hybrid_config, lag_models_of};
use ddai_env::sim::{LagModel, PlayerSetup, Sim, default_target};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::vmath::Vec2;

fn arena() -> Arena {
    let toml = r#"
name = "flat"
tag = "train"
[map]
kind = "synthetic"
width = 40
height = 16
border = true
rects = [ { x0 = 1, y0 = 10, x1 = 38, y1 = 13, tile = "solid" } ]
[spawn]
min_tiles = 0.0
max_tiles = 100.0
rows = [ { y = 9, x0 = 5, x1 = 5 }, { y = 9, x0 = 30, x1 = 30 } ]
"#;
    Arena::build(&ArenaDef::parse(toml).unwrap(), Path::new("/nonexistent")).unwrap()
}

/// Walks left on even decisions and right on odd ones, reports a scripted cost per decision, and remembers the lag it was told.
struct CostBrain {
    costs: Vec<f64>,
    n: usize,
    told: Arc<Mutex<Vec<u32>>>,
    cost_us: u32,
    /// The deadline each decision was told (`None`: none).
    deadlines: Arc<Mutex<Vec<Option<f64>>>>,
    pending_deadline: Option<f64>,
}

impl Brain for CostBrain {
    fn reset(&mut self, _ctx: &ResetContext) {}
    fn decide(&mut self, _obs: &Observation) -> Action {
        Action::neutral()
    }
    fn set_decision_deadline_ms(&mut self, ms: Option<f64>) {
        self.pending_deadline = ms;
    }
    fn decide_in(&mut self, _obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        self.told.lock().unwrap().push(view.map_or(0, |v| v.lag_ticks));
        self.deadlines.lock().unwrap().push(self.pending_deadline.take());
        let cost_ms = self.costs.get(self.n).copied().unwrap_or(1.0);
        self.cost_us = (cost_ms * 1000.0).round() as u32;
        let dir = if self.n.is_multiple_of(2) { -1 } else { 1 };
        self.n += 1;
        Action {
            direction: dir,
            ..Action::neutral()
        }
    }
    fn name(&self) -> &str {
        "cost-stub"
    }
    fn last_plan(&self) -> Option<PlanTelemetry> {
        Some(PlanTelemetry {
            decision_us: self.cost_us,
            ..PlanTelemetry::default()
        })
    }
}

struct Run {
    /// The lag the brain was told at each decision.
    told: Vec<u32>,
    /// The deadline each decision was told by the model (`None`: none).
    deadlines: Vec<Option<f64>>,
    /// The tick the world's input of tee 0 first showed each decision's direction on, minus the tick it was decided at.
    seen_after: Vec<i32>,
    model: LagModel,
}

/// Plays `decisions` decisions of the stub (every 2nd tick) with the given model.
fn run(model: LagModel, costs: Vec<f64>, decisions: usize) -> Run {
    let arena = arena();
    let told = Arc::new(Mutex::new(Vec::new()));
    let deadlines = Arc::new(Mutex::new(Vec::new()));
    let brain = CostBrain {
        costs,
        n: 0,
        told: Arc::clone(&told),
        cost_us: 0,
        deadlines: Arc::clone(&deadlines),
        pending_deadline: None,
    };
    let other = ddai_brain::IdleBrain;
    let mut pw = PhysicsWorld::from_world(arena.new_world(), arena.map.clone());
    for (i, x) in [(0, 5), (1, 30)] {
        pw.add_tee(
            i,
            Vec2 {
                x: f64::from(x as f32 * 32.0 + 16.0),
                y: f64::from(9.0f32 * 32.0 + 16.0),
            },
        );
    }
    let players = vec![
        PlayerSetup {
            brain: Box::new(brain),
            lag: 0,
            label: "stub".into(),
        },
        PlayerSetup {
            brain: Box::new(other),
            lag: 0,
            label: "idle".into(),
        },
    ];
    let mut sim = Sim::new(pw, arena.map.clone(), players, 2, 7);
    sim.set_lag_models(vec![Some(model), None]);
    // The input value of tee 0 after each step, by tick.
    let mut seen: Vec<(i32, i32)> = Vec::new();
    for _ in 0..(2 * decisions as i32 + 12) {
        let tick = sim.tick();
        sim.step(&default_target);
        let dir = sim.pw.inner().cores.get(0).map_or(0, |c| c.input.direction);
        seen.push((tick, dir));
    }
    // Decision k is made at tick 2k and flips the direction, so the k-th change of the world's input is decision k arriving (the lags here only
    // grow, so two decisions never arrive on the same tick).
    let mut changes: Vec<i32> = Vec::new();
    let mut prev = 0;
    for &(t, d) in &seen {
        if d != prev {
            changes.push(t);
            prev = d;
        }
    }
    let seen_after: Vec<i32> = (0..decisions)
        .map(|k| changes.get(k).map_or(-1, |&t| t - 2 * k as i32))
        .collect();
    let told = told.lock().unwrap().clone();
    let deadlines = deadlines.lock().unwrap().clone();
    Run {
        told,
        deadlines,
        seen_after,
        model: sim.lag_models[0].clone().unwrap(),
    }
}

#[test]
fn a_cheap_decision_gets_the_lag_of_its_estimate_and_a_slow_one_a_tick_more() {
    // base 33 + extra 0.5: a cost under 6.5 ms goes out for the 2nd tick after the snapshot (arena lag 1), up to 26.5 the 3rd (lag 2). The estimate starts at 6 ms (-> lag 1).
    let model = LagModel::new(33.0, 0.5, 0.0, 6.0);
    let costs = vec![3.0, 3.0, 3.0, 3.0, 9.0, 3.0, 3.0, 3.0, 3.0, 3.0];
    let r = run(model, costs, 10);
    // Planned: lag 1 for the first four (estimate 6, then 3), the slow decision (index 4) was planned at 1 but cost 9 -> lag 2;
    // from then on the p90 of the last costs is 9 ms for as long as it stays in the window of 64: planned 2.
    assert_eq!(r.told[..5], [1, 1, 1, 1, 1], "told {:?}", r.told);
    assert!(
        r.told[5..10].iter().all(|&l| l == 2),
        "after the slow one the estimate is 9 ms: {:?}",
        r.told
    );
    assert_eq!(r.model.later, 1, "only the slow decision was later than planned");
    assert_eq!(
        r.model.decisions as usize,
        r.told.len(),
        "every decision is counted once (the world runs a few decisions past the ten)"
    );
    assert_eq!(r.model.hist.iter().sum::<u32>(), r.model.decisions);
    assert!(r.model.hist[2] >= 6 && r.model.hist[1] >= 4, "{:?}", r.model.hist);
    // Delivery: a decision at tick T with lag L shows in the world's input on the step at tick T + L.
    assert_eq!(r.seen_after[..4], [1, 1, 1, 1], "{:?}", r.seen_after);
    assert_eq!(
        r.seen_after[4], 2,
        "the slow decision: lag 2, not the 1 it was planned for"
    );
    assert!(r.seen_after[5..10].iter().all(|&a| a == 2), "{:?}", r.seen_after);
}

#[test]
fn an_early_decision_is_held_to_the_tick_it_was_planned_for() {
    // The estimate is 6 ms -> lag 2 at base 38; a 1 ms decision would make lag 1 but is held until its planned tick.
    let model = LagModel::new(38.0, 0.0, 0.0, 6.0);
    let r = run(model, vec![1.0; 6], 6);
    assert_eq!(r.told[0], 2);
    assert_eq!(r.seen_after[0], 2, "held: delivered at the planned lag");
    // After the first fast decisions the estimate falls to 1 ms: planned 2, delivered at 2.
    assert_eq!(r.told[1], 1, "{:?}", r.told);
    assert_eq!(r.seen_after[1], 1);
    assert_eq!(r.model.later, 0);
}

#[test]
fn the_jitter_is_deterministic_bounded_and_part_of_the_plan() {
    let m = LagModel::new(30.0, 0.0, 4.0, 5.0);
    let a: Vec<f64> = (0..200).map(|t| m.plan(11, 0, t).0).collect();
    let b: Vec<f64> = (0..200).map(|t| m.plan(11, 0, t).0).collect();
    assert_eq!(a, b, "the same seed, slot and tick draw the same jitter");
    assert!(
        a.iter().all(|&p| (26.0..=34.0).contains(&p)),
        "within +-4 ms of the base"
    );
    assert!(
        a.iter().any(|&p| p < 28.0) && a.iter().any(|&p| p > 32.0),
        "and it does spread"
    );
    let other_slot: Vec<f64> = (0..200).map(|t| m.plan(11, 1, t).0).collect();
    assert_ne!(a, other_slot);
    // The planned lag follows the phase (the bot knows its own slot slack): 30 + jitter + 5 ms estimate, rounded up to ticks.
    for t in 0..200 {
        let (phase, planned) = m.plan(11, 0, t);
        assert_eq!(planned, ((phase + 5.0) / 20.0).ceil() as u32 - 1);
    }
}

#[test]
fn a_zero_jitter_model_ignores_the_seed_and_the_cost_estimate_is_the_rolling_p90() {
    let mut m = LagModel::new(34.0, 0.0, 0.0, 6.0);
    assert_eq!(m.plan(1, 0, 0), m.plan(999, 3, 77));
    assert_eq!(
        m.plan(1, 0, 0).1,
        1,
        "34 + 6 = 40 -> exactly 2 ticks after the snapshot: lag 1"
    );
    // Ten decisions of 1 ms and one of 10 ms: p90 of 11 samples is the 10th smallest (index ceil(10 * 0.9) = 9): still 1 ms.
    for _ in 0..10 {
        let (p, l) = m.plan(1, 0, 0);
        m.land(p, l, 1.0);
    }
    let (p, l) = m.plan(1, 0, 0);
    assert_eq!(l, 1, "the estimate is 1 ms: 35 -> lag 1");
    m.land(p, l, 10.0);
    assert_eq!(m.plan(1, 0, 0).1, 1, "one slow decision in 11 stays under the p90");
    let (p, l) = m.plan(1, 0, 0);
    m.land(p, l, 10.0);
    assert_eq!(m.plan(1, 0, 0).1, 2, "two in 12 are the p90: 44 ms -> lag 2");
}

#[test]
fn the_histograms_count_every_decision_once() {
    let mut m = LagModel::new(30.0, 0.0, 0.0, 6.0);
    for c in [1.0, 5.0, 12.0, 40.0, 3000.0] {
        let (p, l) = m.plan(1, 0, 0);
        m.land(p, l, c);
    }
    assert_eq!(m.decisions, 5);
    assert_eq!(m.hist.iter().sum::<u32>(), 5);
    assert_eq!(m.cost_hist.iter().sum::<u32>(), 5);
    assert_eq!(
        *m.cost_hist.last().unwrap(),
        2,
        "40 ms and 3000 ms both land in the overflow bin"
    );
    assert!(m.mean_lag() > 1.0);
}

#[test]
fn the_spec_validates_builds_a_fresh_model_per_slot_and_refuses_both_a_lag_and_a_model() {
    let spec = LagModelSpec {
        base_ms: 33.0,
        extra_ms: 0.5,
        jitter_ms: 0.0,
        initial_ms: 6.0,
        deadline_floor_ms: None,
    };
    assert!(spec.validate(0).is_ok());
    assert!(spec.validate(3).is_err(), "a model replaces the fixed lag");
    assert!(
        LagModelSpec {
            base_ms: f64::NAN,
            ..spec.clone()
        }
        .validate(0)
        .is_err()
    );
    assert!(
        LagModelSpec {
            jitter_ms: -1.0,
            ..spec.clone()
        }
        .validate(0)
        .is_err()
    );
    let mut a = PlayerSpec::simple("hybrid");
    a.lag_model = Some(spec);
    let b = PlayerSpec::simple("planner");
    let models = lag_models_of(&[a, b]);
    assert!(models[0].is_some() && models[1].is_none());
    // A condition naming both is refused when the run config is parsed.
    let text = |lag: &str| {
        format!(
            r#"
name = "t"
base_seed = 1
games = 1
[[condition]]
name = "c"
arena = "pit"
players = [{{ brain = "hybrid", {lag} lag_model = {{ base_ms = 33.0 }} }}, {{ brain = "idle" }}]
"#
        )
    };
    assert!(RunConfig::parse(&text("")).is_ok());
    let err = RunConfig::parse(&text("lag = 2,")).unwrap_err().to_string();
    assert!(err.contains("lag_model"), "{err}");
}

#[test]
fn the_live_budget_key_sets_the_budget_and_the_cap_together_and_is_refused_out_of_range() {
    let spec = |ms: u32, mode: &str| {
        let mut s = PlayerSpec::simple("hybrid");
        s.mode = Some(mode.into());
        s.clock = Some("work".into());
        s.hybrid = Some(ddai_env::config::HybridSpec {
            live_budget_ms: Some(ms),
            ..Default::default()
        });
        s
    };
    let (cfg, _) = hybrid_config(&spec(2, "deadline")).unwrap();
    assert_eq!(cfg.mode, ddai_planner::hybrid::HybridMode::Deadline { budget_ms: 2.0 });
    assert_eq!(cfg.decision_cap_ms, Some(3.0));
    // 4 is the default point: the same config as a spec that names nothing (apart from nothing else differing).
    let (four, _) = hybrid_config(&spec(4, "deadline")).unwrap();
    let mut plain = PlayerSpec::simple("hybrid");
    plain.mode = Some("deadline".into());
    plain.clock = Some("work".into());
    plain.hybrid = Some(ddai_env::config::HybridSpec::default());
    assert_eq!(four, hybrid_config(&plain).unwrap().0);
    assert!(hybrid_config(&spec(0, "deadline")).is_err());
    assert!(hybrid_config(&spec(9, "deadline")).is_err());
    assert!(
        hybrid_config(&spec(3, "fixed")).is_err(),
        "fixed work has no budget to set"
    );
}

#[test]
fn a_deadline_aware_model_tells_the_brain_the_room_the_slot_leaves_and_plans_for_it() {
    // base 36, extra 1: the slot after a free decision is 40 ms, so the room is 40 - 36 - 1 - 0.1 = 2.9 ms. With the floor at 2.5 the brain is told 2.9.
    let mut model = LagModel::new(36.0, 1.0, 0.0, 6.0);
    model.deadline_floor_ms = Some(2.5);
    assert!((model.deadline_for(36.0).unwrap() - 2.9).abs() < 1e-9);
    assert_eq!(
        model.deadline_for(37.5),
        None,
        "1.4 ms of room is under the floor: no deadline, the second slot"
    );
    assert_eq!(
        LagModel::new(36.0, 1.0, 0.0, 6.0).deadline_for(36.0),
        None,
        "off without a floor"
    );
    // The estimate (6 ms before the first decision) would plan lag 2; with the room it plans for the cut: lag 1.
    assert_eq!(model.plan(1, 0, 0).1, 1);
    assert_eq!(LagModel::new(36.0, 1.0, 0.0, 6.0).plan(1, 0, 0).1, 2);
    // In a game: the stub is told the deadline before every decision, and a decision that keeps within it is delivered at the planned lag.
    let r = run(model, vec![2.5; 6], 6);
    assert!(
        r.deadlines
            .iter()
            .take(6)
            .all(|d| d.is_some_and(|d| (d - 2.9).abs() < 1e-9)),
        "{:?}",
        r.deadlines
    );
    assert!(r.told.iter().take(6).all(|&l| l == 1), "{:?}", r.told);
    assert!(r.seen_after.iter().take(6).all(|&a| a == 1), "{:?}", r.seen_after);
    assert_eq!(r.model.later, 0);
    // A model without a floor never tells a deadline.
    let r = run(LagModel::new(36.0, 1.0, 0.0, 6.0), vec![2.5; 4], 4);
    assert!(r.deadlines.iter().all(Option::is_none));
}
