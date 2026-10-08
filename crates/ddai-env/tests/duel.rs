//! Task 3.19 (D-116): the duel mode of the arena -- the F-DDrace round rules (`ddai_env::duel`) and the live view of a brain
//! (`ddai_env::liveview`), on hand-built situations whose answer is known by construction.

use std::path::Path;

use ddai_brain::Action;
use ddai_env::arena::{Arena, ArenaDef};
use ddai_env::brains::TimelineBrain;
use ddai_env::config::{Rules, RunConfig};
use ddai_env::duel::DuelSpec;
use ddai_env::game::{GameReport, Layout, play_game_duel_watched};
use ddai_env::sim::PlayerSetup;
use ddai_env::stats::GameResult;

/// A long floor (tees stand on row 11) with a strip of freeze tiles *in the standing row* (x 40..=44): a tee spawned on it is in a freeze tile
/// on the ground, one spawned elsewhere is not. The two spawn tiles are `a` and `b` (a game puts a tee on each, `Layout::swap` trades them;
/// the arena picks which one slot 0 gets by the seed, [`arranged`] sorts that out).
fn ring(a: i32, b: i32) -> Arena {
    let d = f64::from((a - b).abs());
    let rows = format!("    {{ y = 11, x0 = {a}, x1 = {a} }},\n    {{ y = 11, x0 = {b}, x1 = {b} }},\n");
    let toml = format!(
        r#"
name = "ring"
tag = "train"
[map]
kind = "synthetic"
width = 60
height = 20
border = true
rects = [
    {{ x0 = 1, y0 = 12, x1 = 58, y1 = 15, tile = "solid" }},
    {{ x0 = 40, y0 = 11, x1 = 44, y1 = 11, tile = "freeze" }},
]
[spawn]
min_tiles = {d}
max_tiles = {d}
rows = [
{rows}]
"#
    );
    Arena::build(&ArenaDef::parse(&toml).unwrap(), Path::new("/nonexistent")).unwrap()
}

fn act(direction: i32, hook: bool, aim: [i32; 2]) -> Action {
    Action {
        direction,
        hook,
        target: ddai_brain::IVec2::new(aim[0], aim[1]),
        ..Action::neutral()
    }
}

fn puppet(name: &str, steps: Vec<(i32, Action)>) -> PlayerSetup {
    PlayerSetup {
        brain: Box::new(TimelineBrain::new(name, steps)),
        lag: 0,
        label: name.into(),
    }
}

fn idle() -> PlayerSetup {
    puppet("idle", vec![(0, Action::neutral())])
}

/// Plays a duel game with slot 0 on tile `x0` (and slot 1 on the arena's other tile): the layout is swapped when the seed put slot 0 elsewhere.
fn arranged(
    arena: &Arena,
    x0: i32,
    spec: &DuelSpec,
    rules: &Rules,
    mut players: impl FnMut() -> Vec<PlayerSetup>,
) -> GameReport {
    for swap in [false, true] {
        let layout = Layout {
            swap,
            reverse_order: false,
        };
        let r = play_game_duel_watched(
            arena,
            rules,
            Some(spec),
            1,
            layout,
            players(),
            Vec::new(),
            &mut |_, _| true,
        )
        .unwrap();
        if (r.spawns[0][0] / 32.0).floor() as i32 == x0 {
            return r;
        }
    }
    panic!("no layout puts slot 0 on tile {x0}");
}

fn hooker() -> PlayerSetup {
    // A short pull at the other tee right after the countdown, then it lets go and stands.
    puppet(
        "hooker",
        vec![
            (0, act(0, false, [300, 0])),
            (158, act(0, true, [300, 0])),
            (164, act(0, false, [300, 0])),
        ],
    )
}

#[test]
fn a_frozen_tee_in_a_freeze_tile_on_the_ground_loses_after_a_second_when_the_other_touched_it() {
    // Slot 0 lies in the freeze strip (tile 42), slot 1 stands 6 tiles to its left, inside the hook's reach (aiming right).
    let arena = ring(42, 36);
    let r = arranged(&arena, 42, &DuelSpec::default(), &Rules::default(), || {
        vec![idle(), hooker()]
    });
    assert_eq!(
        (r.result, r.credited, r.victim),
        (GameResult::L, true, 0),
        "end {}",
        r.end_tick
    );
    // The countdown is 150 ticks, the streak starts once nothing moves it any more, a second later it is over.
    assert!(
        r.end_tick > 150 + 50 && r.end_tick < 150 + 50 + 40,
        "end {}",
        r.end_tick
    );
    assert_eq!(r.fight_start, 150);
}

#[test]
fn the_other_slot_losing_is_a_win_for_the_focal_player() {
    let arena = ring(42, 36);
    let r = arranged(&arena, 36, &DuelSpec::default(), &Rules::default(), || {
        vec![hooker(), idle()]
    });
    assert_eq!(
        (r.result, r.credited, r.victim),
        (GameResult::W, true, 1),
        "end {}",
        r.end_tick
    );
}

#[test]
fn a_round_without_a_touch_gives_no_point() {
    let arena = ring(42, 36);
    let r = arranged(&arena, 42, &DuelSpec::default(), &Rules::default(), || {
        vec![idle(), idle()]
    });
    // The frozen tee lay still in a freeze tile for a second, but nobody hooked or hit it: `IncreaseScore` counts nothing.
    assert_eq!((r.result, r.credited), (GameResult::D, false), "end {}", r.end_tick);
    assert!(r.end_tick > 150 + 50 && r.end_tick < 150 + 50 + 5, "end {}", r.end_tick);
}

#[test]
fn a_mutual_freeze_is_a_draw() {
    // Both lie in the strip: whichever the rule looks at first, the other is in a freeze tile too, so there is no point -- even though they touched.
    let arena = ring(41, 43);
    let r = arranged(&arena, 41, &DuelSpec::default(), &Rules::default(), || {
        let pull = |dir: i32| {
            puppet(
                "p",
                vec![
                    (0, act(0, false, [dir * 300, 0])),
                    (158, act(0, true, [dir * 300, 0])),
                    (162, act(0, false, [dir * 300, 0])),
                ],
            )
        };
        vec![pull(1), pull(-1)]
    });
    assert_eq!((r.result, r.credited), (GameResult::D, false), "end {}", r.end_tick);
}

#[test]
fn the_spawn_freeze_alone_does_not_end_a_round() {
    // Both tees stand still on plain floor frozen by the countdown: they are not in a freeze tile (the clips: the 150 ticks of every round run
    // through, nobody loses during them or after the thaw).
    let arena = ring(10, 14);
    let rules = Rules {
        max_ticks: 120,
        ..Rules::default()
    };
    let r = arranged(&arena, 10, &DuelSpec::default(), &rules, || vec![idle(), idle()]);
    assert_eq!((r.result, r.end_tick), (GameResult::T, 150 + 120));
}

#[test]
fn nobody_moves_in_the_countdown() {
    let arena = ring(10, 14);
    let walker = |n: &str| puppet(n, vec![(0, act(1, false, [0, -1]))]);
    let mut xs: Vec<(i32, f32, f32)> = Vec::new();
    let r = play_game_duel_watched(
        &arena,
        &Rules::default(),
        Some(&DuelSpec::default()),
        1,
        Layout::default(),
        vec![walker("a"), walker("b")],
        Vec::new(),
        &mut |sim, tick| {
            if tick == 100 || tick == 140 || tick == 200 {
                let w = sim.pw.inner();
                xs.push((tick, w.cores.get(0).unwrap().pos.x, w.cores.get(1).unwrap().pos.x));
            }
            tick < 200
        },
    )
    .unwrap();
    assert_eq!(r.result, GameResult::T);
    let x = |t: i32| xs.iter().find(|e| e.0 == t).unwrap().1;
    assert_eq!(x(100), x(140), "frozen through the countdown");
    assert!(x(200) > x(140) + 20.0, "walking after the thaw: {xs:?}");
}

#[test]
fn a_zero_countdown_starts_at_once() {
    let arena = ring(10, 14);
    let spec = DuelSpec {
        countdown_ticks: 0,
        ..DuelSpec::default()
    };
    let rules = Rules {
        max_ticks: 30,
        ..Rules::default()
    };
    let r = arranged(&arena, 10, &spec, &rules, || vec![idle(), idle()]);
    assert_eq!((r.result, r.end_tick, r.fight_start), (GameResult::T, 30, 0));
}

#[test]
fn the_duel_options_parse_validate_and_leave_the_config_hash_alone_when_absent() {
    let base = r#"
name = "t"
[[condition]]
name = "c"
arena = "pit"
players = [{ brain = "idle" }, { brain = "idle" }]
"#;
    let with = format!("{base}duel = {{ live_view = [0], grounded_ticks = 40 }}\n");
    let plain = RunConfig::parse(base).unwrap();
    let cfg = RunConfig::parse(&with).unwrap();
    let d = cfg.condition[0].duel.as_ref().unwrap();
    assert_eq!(
        (d.rounds, d.countdown_ticks, d.grounded_ticks, d.live_view.clone()),
        (true, 150, 40, vec![0])
    );
    assert!(plain.condition[0].duel.is_none());
    assert!(
        !serde_json::to_string(&plain).unwrap().contains("duel"),
        "an absent table must not change the effective config"
    );
    let bad_slot = format!("{base}duel = {{ live_view = [2] }}\n");
    assert!(RunConfig::parse(&bad_slot).is_err());
    let three = r#"
name = "t"
[[condition]]
name = "c"
arena = "pit"
players = [{ brain = "idle" }, { brain = "idle", count = 2 }]
duel = {}
"#;
    assert!(RunConfig::parse(three).is_err(), "the round rules are for two players");
}

/// The world a live seat rebuilds from the server-style snapshots equals the true world at every snapshot tick (core fields bit for bit, the
/// freeze countdown and the attack tick), through the dead reckoning the server does, a countdown, hooks, jumps, a hammer swing and a freeze.
#[test]
fn the_live_view_rebuilds_the_true_world_at_every_snapshot() {
    let arena = ring(14, 22);
    let a = puppet(
        "a",
        vec![
            (0, act(1, false, [300, -20])),
            (170, act(1, false, [300, 0])),
            (
                176,
                Action {
                    jump: true,
                    ..act(1, false, [300, -40])
                },
            ),
            (182, act(1, true, [300, 0])),
            (
                200,
                Action {
                    fire: true,
                    ..act(1, false, [300, 0])
                },
            ),
            (206, act(0, false, [300, 0])),
            (230, act(1, false, [300, 0])),
        ],
    );
    let b = puppet(
        "b",
        vec![
            (0, act(-1, false, [-300, 0])),
            (180, act(0, false, [-300, 0])),
            (
                190,
                Action {
                    jump: true,
                    ..act(-1, false, [-300, -30])
                },
            ),
            (215, act(1, false, [300, 0])),
        ],
    );
    let spec = DuelSpec {
        live_view: vec![0, 1],
        ..DuelSpec::default()
    };
    type Truth = (i32, Vec<(i32, ddai_physics::core::NetCharacterCore, i32, i32)>);
    let mut saved: Option<Truth> = None;
    let (mut compared, mut moved) = (0u32, 0u32);
    let r = play_game_duel_watched(
        &arena,
        &Rules {
            max_ticks: 400,
            ..Rules::default()
        },
        Some(&spec),
        3,
        Layout::default(),
        vec![a, b],
        Vec::new(),
        &mut |sim, tick| {
            // A decision at an even tick T fed the snapshot of the world after T steps; this call (after step T) sees `base_world().tick == T`.
            if tick % 2 == 1
                && let Some((t, ref truth)) = saved
                && t == tick - 1
            {
                for slot in 0..2usize {
                    let live = sim.live_world(slot).expect("a live seat");
                    let base = live.base_world();
                    assert_eq!(base.tick, t);
                    for (id, core, freeze, attack) in truth {
                        let got = base.cores.get(*id as u8).expect("tee in the rebuilt world");
                        assert_eq!(got.write(), *core, "slot {slot} tee {id} tick {t}");
                        let ch = base.characters[*id as usize].as_ref().unwrap();
                        assert_eq!(
                            (ch.freeze_time, ch.attack_tick),
                            (*freeze, *attack),
                            "slot {slot} tee {id} tick {t}"
                        );
                    }
                    compared += 1;
                }
            }
            let w = sim.pw.inner();
            let truth: Vec<_> = (0..2i32)
                .filter_map(|id| {
                    let c = w.cores.get(id as u8)?;
                    let ch = w.characters[id as usize].as_ref()?;
                    Some((id, c.write(), ch.freeze_time, ch.attack_tick))
                })
                .collect();
            if let Some((_, prev)) = &saved
                && prev.iter().zip(&truth).any(|(p, q)| p.1.x != q.1.x)
            {
                moved += 1;
            }
            saved = Some((tick, truth));
            true
        },
    )
    .unwrap();
    assert!(compared > 100, "{compared} snapshots compared, ended at {}", r.end_tick);
    assert!(moved > 50, "the tees moved ({moved})");
}

/// A live-view game is deterministic (same seed, same decisions).
#[test]
fn a_live_view_game_is_deterministic() {
    let arena = ring(14, 22);
    let mk = || {
        let spec = DuelSpec {
            live_view: vec![0],
            ..DuelSpec::default()
        };
        let a = puppet(
            "a",
            vec![(0, act(1, false, [300, 0])), (200, act(-1, false, [-300, 0]))],
        );
        let rules = Rules {
            max_ticks: 300,
            ..Rules::default()
        };
        arranged(&arena, 14, &spec, &rules, || {
            vec![
                puppet(
                    "a",
                    vec![(0, act(1, false, [300, 0])), (200, act(-1, false, [-300, 0]))],
                ),
                idle(),
            ]
        })
        .players[0]
            .hash
            .clone()
            + &a.label
    };
    assert_eq!(mk(), mk());
}

/// The prediction the brain decides on is the world our own inputs in flight lead to: the predicted position of our tee at `tick + lag` is where
/// the true world puts it `lag` ticks later (exactly: the snapshot's quantisation does not show in this scenario), for lags 0..=3, with a puppet
/// that changes its direction and jumps at every decision and an opponent that stands.
#[test]
fn the_live_view_predicts_our_own_tee_through_its_inputs_in_flight() {
    let arena = ring(14, 30);
    for lag in 0u32..=3 {
        let mut script = Vec::new();
        for k in 0..60 {
            let t = 150 + 2 * k;
            script.push((
                t,
                Action {
                    direction: if k % 3 == 0 { -1 } else { 1 },
                    jump: k % 7 == 0,
                    ..Action::neutral()
                },
            ));
        }
        let me = PlayerSetup {
            brain: Box::new(TimelineBrain::new("me", script)),
            lag,
            label: "me".into(),
        };
        let spec = DuelSpec {
            live_view: vec![0],
            ..DuelSpec::default()
        };
        let mut predicted: Vec<(i32, f32, f32)> = Vec::new();
        let mut truth: std::collections::BTreeMap<i32, (f32, f32)> = Default::default();
        let mut worst = 0.0f32;
        let _ = play_game_duel_watched(
            &arena,
            &Rules {
                max_ticks: 200,
                ..Rules::default()
            },
            Some(&spec),
            5,
            Layout::default(),
            vec![me, idle()],
            Vec::new(),
            &mut |sim, tick| {
                let w = sim.pw.inner();
                if let Some(c) = w.cores.get(0) {
                    truth.insert(tick, (c.pos.x, c.pos.y));
                }
                // The decision at tick T = `tick - 1` (even) has just been made.
                if (tick - 1) % 2 == 0
                    && tick > 150
                    && let Some(o) = sim.live_observation(0)
                {
                    predicted.push((tick - 1 + lag as i32, o.self_state.pos.x, o.self_state.pos.y));
                }
                true
            },
        )
        .unwrap();
        let mut compared = 0;
        for (t, x, y) in &predicted {
            if let Some(&(tx, ty)) = truth.get(t) {
                worst = worst.max((x - tx).abs()).max((y - ty).abs());
                compared += 1;
            }
        }
        assert!(compared > 40, "lag {lag}: {compared} compared");
        eprintln!("lag {lag}: {compared} predictions, worst error {worst} px");
        assert_eq!(worst, 0.0, "lag {lag}: the prediction is {worst} px off");
    }
}
