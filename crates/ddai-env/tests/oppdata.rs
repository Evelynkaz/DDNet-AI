//! Task 3.15 (E-028): the recorded dataset says what the brains did, on the arena's own timing.
//!
//! Two puppets with known timelines (ours with an input lag, the opponent without) play a few ticks of a corridor. The record's applied inputs must be the
//! timelines' decisions shifted by each player's lag; the in-flight inputs the brain is handed live must be exactly the inputs the record holds for the same
//! ticks (the trainer builds its input from the record, the brain from the view: this is the bridge between them); and the trainer's labels must be the
//! opponent's decisions.

use std::path::Path;
use std::sync::{Arc, Mutex};

use ddai_brain::{Action, Brain, Observation, ResetContext, WorldView};
use ddai_env::arena::{Arena, ArenaDef};
use ddai_env::brains::TimelineBrain;
use ddai_env::config::Rules;
use ddai_env::game::Layout;
use ddai_env::oppdata::record_game;
use ddai_env::sim::PlayerSetup;
use ddai_oppnet::feature::{HORIZON, IF_DIM, inflight_features};
use ddai_oppnet::frame::InputRec;
use ddai_oppnet::train::Corpus;
use ddai_physics::core::PlayerInput as Wire;

fn corridor() -> Arena {
    let toml = r#"
name = "corridor"
tag = "train"
[map]
kind = "synthetic"
width = 60
height = 16
border = true
rects = [{ x0 = 1, y0 = 10, x1 = 58, y1 = 13, tile = "solid" }]
[spawn]
min_tiles = 0.0
max_tiles = 100.0
rows = [{ y = 9, x0 = 20, x1 = 20 }, { y = 9, x0 = 30, x1 = 30 }]
"#;
    Arena::build(&ArenaDef::parse(toml).unwrap(), Path::new("/nonexistent")).unwrap()
}

fn act(direction: i32, jump: bool, hook: bool) -> Action {
    Action {
        direction,
        jump,
        hook,
        ..Action::neutral()
    }
}

/// The in-flight inputs a brain was handed at each decision: `(tick, inputs)`.
type Seen = Arc<Mutex<Vec<(i32, Vec<Wire>)>>>;

/// A puppet that also keeps the in-flight inputs of each decision.
struct Spy {
    inner: TimelineBrain,
    seen: Seen,
}

impl Brain for Spy {
    fn reset(&mut self, ctx: &ResetContext) {
        self.inner.reset(ctx);
    }
    fn decide(&mut self, obs: &Observation) -> Action {
        self.inner.decide(obs)
    }
    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        if let Some(v) = view {
            self.seen.lock().unwrap().push((obs.tick, v.in_flight.to_vec()));
        }
        self.inner.decide(obs)
    }
    fn name(&self) -> &str {
        "spy"
    }
}

const OURS: [(i32, i32, bool, bool); 7] = [
    (0, 0, false, false),
    (6, 1, false, false),
    (12, 1, true, false),
    (18, -1, false, true),
    (24, 0, false, false),
    (30, 1, true, true),
    (40, -1, false, false),
];
const THEIRS: [(i32, i32, bool, bool); 7] = [
    (0, 1, false, false),
    (8, -1, false, false),
    (14, -1, false, true),
    (20, 0, true, false),
    (26, 1, false, false),
    (32, -1, false, true),
    (44, 1, true, false),
];

fn script(t: &[(i32, i32, bool, bool)]) -> Vec<(i32, Action)> {
    t.iter().map(|&(from, d, j, h)| (from, act(d, j, h))).collect()
}

/// The puppet's input at decision tick `t`.
fn at(t: &[(i32, i32, bool, bool)], tick: i32) -> (i32, bool, bool) {
    let &(_, d, j, h) = t.iter().rev().find(|e| e.0 <= tick).unwrap_or(&t[0]);
    (d, j, h)
}

#[test]
fn the_record_is_the_decisions_on_the_arena_timing() {
    let arena = corridor();
    let rules = Rules {
        max_ticks: 70,
        after_ticks: 0,
        ..Rules::default()
    };
    let lag = 3u32;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let players = vec![
        PlayerSetup {
            brain: Box::new(Spy {
                inner: TimelineBrain::new("ours", script(&OURS)),
                seen: seen.clone(),
            }),
            lag,
            label: "ours".into(),
        },
        PlayerSetup {
            brain: Box::new(TimelineBrain::new("theirs", script(&THEIRS))),
            lag: 0,
            label: "theirs".into(),
        },
    ];
    let (rec, _) = record_game(&arena, &rules, 1, Layout::default(), players).unwrap();
    assert_eq!(rec.lag, [3, 0]);
    assert_eq!(rec.decide_every, 2);
    assert!(rec.ticks.len() >= 60, "{}", rec.ticks.len());
    // The input applied in the step of world tick `t` is stored with tick `t + 1`.
    let applied = |slot: usize, t: i32| rec.ticks[rec.index_of(t + 1).expect("recorded")].applied[slot];
    for t in 4..60 {
        // The opponent decides at every even tick and applies at once; we decide at even ticks and apply `lag` ticks later.
        let (d, j, h) = at(&THEIRS, t - t % 2);
        let a = applied(1, t);
        assert_eq!(
            (i32::from(a.direction), a.jump, a.hook),
            (d, j, h),
            "opponent, step {t}"
        );
        let td = t - lag as i32;
        let (d, j, h) = at(&OURS, td - td.rem_euclid(2));
        let a = applied(0, t);
        assert_eq!((i32::from(a.direction), a.jump, a.hook), (d, j, h), "us, step {t}");
    }
    // The in-flight inputs the brain was handed are the inputs the record holds for the steps T .. T + lag.
    let seen = seen.lock().unwrap();
    assert!(seen.len() > 20);
    for (tick, in_flight) in seen.iter().filter(|(t, _)| *t >= 8 && *t < 56) {
        assert_eq!(in_flight.len(), lag as usize, "tick {tick}");
        for (k, wire) in in_flight.iter().enumerate() {
            let rec_in = applied(0, tick + k as i32);
            let (mut a, mut b) = ([0.0f32; IF_DIM], [0.0f32; IF_DIM]);
            inflight_features(&InputRec::from_wire(wire), &mut a);
            inflight_features(&rec_in, &mut b);
            assert_eq!(a, b, "decision at tick {tick}, in-flight input {k}");
            assert_eq!(
                (wire.direction, wire.jump != 0, wire.hook != 0),
                (i32::from(rec_in.direction), rec_in.jump, rec_in.hook)
            );
        }
    }
    // The trainer's labels are the opponent's decisions of the next ticks.
    let corpus = Corpus::new(vec![rec]);
    let mut checked = 0;
    for r in corpus.samples() {
        let g = &corpus.games[0];
        let t = g.tick0 + r.idx as i32;
        if !(10..50).contains(&t) {
            continue;
        }
        let l = corpus.label(r);
        for k in 0..HORIZON {
            let tk = t + k as i32;
            let (d, j, h) = at(&THEIRS, tk - tk % 2);
            assert_eq!(l.valid >> k & 1, 1);
            assert_eq!(i32::from(l.dir[k]) - 1, d, "label dir, T {t} k {k}");
            assert_eq!(
                (l.jump >> k & 1 != 0, l.hook >> k & 1 != 0),
                (j, h),
                "label flags, T {t} k {k}"
            );
        }
        checked += 1;
    }
    assert!(checked > 15, "{checked}");
}
