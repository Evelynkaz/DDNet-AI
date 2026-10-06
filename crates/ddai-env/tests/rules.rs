//! Rule semantics on hand-built situations: scripted "puppet" brains drive tees into known
//! outcomes on tiny synthetic arenas, so every branch of the win definition is checked against a
//! situation whose answer is known by construction.

use std::path::Path;
use std::sync::{Arc, Mutex};

use ddai_brain::{Action, Brain, IVec2, Observation, ResetContext, WorldView};
use ddai_env::arena::{Arena, ArenaDef};
use ddai_env::brains::TimelineBrain;
use ddai_env::config::Rules;
use ddai_env::game::{GameReport, Layout, play_game};
use ddai_env::sim::PlayerSetup;
use ddai_env::stats::GameResult;

/// A puppet: plays back `(from_tick, action)` pairs.
struct Puppet;

impl Puppet {
    fn boxed(script: Vec<(i32, Action)>) -> Box<dyn Brain> {
        Box::new(TimelineBrain::new("puppet", script))
    }
}

fn walk(dir: i32) -> Action {
    Action {
        direction: dir,
        ..Action::neutral()
    }
}

/// A flat corridor: floor at row 10 (tees stand on row 9), a freeze pit `pit_x0..=pit_x1` (2 deep)
/// cut into the floor, spawn slots given explicitly.
fn corridor(pit_x0: i32, pit_x1: i32, slots: &[(i32, i32)]) -> Arena {
    let rows: String = slots
        .iter()
        .map(|(x0, x1)| format!("    {{ y = 9, x0 = {x0}, x1 = {x1} }},\n"))
        .collect();
    let toml = format!(
        r#"
name = "corridor"
tag = "train"
[map]
kind = "synthetic"
width = 40
height = 16
border = true
rects = [
    {{ x0 = 1, y0 = 10, x1 = 38, y1 = 13, tile = "solid" }},
    {{ x0 = {pit_x0}, y0 = 10, x1 = {pit_x1}, y1 = 11, tile = "freeze" }},
]
[spawn]
min_tiles = 0.0
max_tiles = 100.0
rows = [
{rows}]
"#
    );
    Arena::build(&ArenaDef::parse(&toml).unwrap(), Path::new("/nonexistent")).unwrap()
}

fn setup(brain: Box<dyn Brain>) -> PlayerSetup {
    PlayerSetup {
        brain,
        lag: 0,
        label: "p".into(),
    }
}

fn play(arena: &Arena, rules: &Rules, seed: u64, swap: bool, players: Vec<Box<dyn Brain>>) -> GameReport {
    play_game(
        arena,
        rules,
        seed,
        Layout {
            swap,
            reverse_order: false,
        },
        players.into_iter().map(setup).collect(),
    )
    .unwrap()
}

/// A slot list holding exactly one tile makes the spawn deterministic regardless of seed: with
/// `min_tiles = 0` a pick of the same tile twice is accepted, so use two 1-tile rows instead.
fn arena_with_two_tiles(pit: (i32, i32), a: i32, b: i32) -> Arena {
    // Rows are read in order; with the RNG picking uniformly among 2 slots the pair (a, a) or
    // (b, b) or mixed is possible, so tests below always choose a seed whose spawn they assert.
    corridor(pit.0, pit.1, &[(a, a), (b, b)])
}

/// Finds a seed whose spawn puts slot 0 at tile `a` and slot 1 at tile `b`.
fn seed_for(arena: &Arena, a: i32, b: i32) -> u64 {
    (0..10_000u64)
        .find(|&s| {
            let t = arena.spawn_tiles(s, 2).unwrap();
            t[0].0 == a && t[1].0 == b
        })
        .expect("a seed with that spawn exists")
}

#[test]
fn only_b_out_is_a_win_only_a_out_a_loss() {
    // Pit at x 20..=21; A at x=17, B at x=24. B walks left into the pit; A stays.
    let arena = arena_with_two_tiles((20, 21), 17, 24);
    let seed = seed_for(&arena, 17, 24);
    let r = play(
        &arena,
        &Rules::default(),
        seed,
        false,
        vec![Puppet::boxed(vec![]), Puppet::boxed(vec![(0, walk(-1))])],
    );
    assert_eq!(r.result, GameResult::W, "{r:?}");
    assert_eq!(r.victim, 1);
    assert!(!r.credited, "nobody touched B: it walked into the pit on its own");
    assert!(r.end_tick > 0 && r.end_tick < 200);
    assert!(r.held, "B stays frozen for the 150 ticks that follow");
    assert_eq!(r.a_self_freezes, 0);
    assert_eq!(r.a_out_tick, None);

    // Mirror image: A walks into the pit, B stays: a loss with the victim being slot 0.
    let r = play(
        &arena,
        &Rules::default(),
        seed,
        false,
        vec![Puppet::boxed(vec![(0, walk(1))]), Puppet::boxed(vec![])],
    );
    assert_eq!(r.result, GameResult::L, "{r:?}");
    assert_eq!(r.victim, 0);
    assert_eq!(r.a_self_freezes, 1);
    assert_eq!(r.a_out_tick, Some(r.end_tick));
}

#[test]
fn both_out_on_the_same_tick_is_a_draw() {
    // Symmetric: pit at 20..=21, A at 17 walking right, B at 24 walking left (mirror positions).
    let arena = arena_with_two_tiles((20, 21), 17, 24);
    let seed = seed_for(&arena, 17, 24);
    let r = play(
        &arena,
        &Rules::default(),
        seed,
        false,
        vec![Puppet::boxed(vec![(0, walk(1))]), Puppet::boxed(vec![(0, walk(-1))])],
    );
    // The two tees are mirror images of each other so they enter the pit on the same tick.
    assert_eq!(r.result, GameResult::D, "{r:?}");
    assert_eq!(r.victim, -1);
    assert!(!r.credited && !r.held);
}

#[test]
fn nobody_out_is_a_timeout_at_max_ticks() {
    let arena = arena_with_two_tiles((20, 21), 5, 8);
    let seed = seed_for(&arena, 5, 8);
    let rules = Rules {
        max_ticks: 300,
        ..Rules::default()
    };
    let r = play(
        &arena,
        &rules,
        seed,
        false,
        vec![Puppet::boxed(vec![]), Puppet::boxed(vec![])],
    );
    assert_eq!(r.result, GameResult::T);
    assert_eq!(r.end_tick, 300);
    assert!(!r.credited && !r.held);
    assert_eq!(r.players[0].decisions, 150, "one decision per 2 ticks");
}

#[test]
fn swap_trades_the_spawns_of_the_first_two_players() {
    let arena = arena_with_two_tiles((20, 21), 5, 8);
    let seed = seed_for(&arena, 5, 8);
    let rules = Rules {
        max_ticks: 10,
        ..Rules::default()
    };
    let plain = play(
        &arena,
        &rules,
        seed,
        false,
        vec![Puppet::boxed(vec![]), Puppet::boxed(vec![])],
    );
    let swapped = play(
        &arena,
        &rules,
        seed,
        true,
        vec![Puppet::boxed(vec![]), Puppet::boxed(vec![])],
    );
    assert_eq!(plain.spawns[0], swapped.spawns[1]);
    assert_eq!(plain.spawns[1], swapped.spawns[0]);
}

/// Records what a brain was shown; used to check the `WorldView` contract.
struct Probe {
    log: Arc<Mutex<Vec<ProbeRow>>>,
    script: Vec<(i32, Action)>,
}

#[derive(Debug, Clone)]
struct ProbeRow {
    tick: i32,
    view_tick: i32,
    lag: u32,
    in_flight_dirs: Vec<i32>,
    applied_dir: i32,
    obs_x: f32,
    world_x: f32,
}

impl Brain for Probe {
    fn reset(&mut self, _ctx: &ResetContext) {}
    fn decide(&mut self, _obs: &Observation) -> Action {
        unreachable!("the arena always offers a world view")
    }
    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        let view = view.expect("the arena passes the exact world");
        let ch = view.world.characters[view.self_id as usize].as_ref().unwrap();
        let core = view.world.cores.get(view.self_id as u8).unwrap();
        self.log.lock().unwrap().push(ProbeRow {
            tick: obs.tick,
            view_tick: view.world.tick,
            lag: view.lag_ticks,
            in_flight_dirs: view.in_flight.iter().map(|i| i.direction).collect(),
            applied_dir: ch.input.direction,
            obs_x: obs.self_state.pos.x,
            world_x: core.pos.x,
        });
        self.script
            .iter()
            .rev()
            .find(|(t, _)| *t <= obs.tick)
            .map_or(Action::neutral(), |(_, a)| *a)
    }
    fn name(&self) -> &str {
        "probe"
    }
}

#[test]
fn view_is_the_exact_world_and_lag_follows_the_documented_timing() {
    let arena = arena_with_two_tiles((20, 21), 5, 8);
    let seed = seed_for(&arena, 5, 8);
    let rules = Rules {
        max_ticks: 40,
        after_ticks: 0,
        ..Rules::default()
    };
    for lag in [0u32, 1, 3, 5] {
        let log = Arc::new(Mutex::new(Vec::new()));
        // Direction +1 from tick 4 on; before that 0. Decisions happen on even ticks.
        let probe = Probe {
            log: log.clone(),
            script: vec![(4, walk(1))],
        };
        let players = vec![
            PlayerSetup {
                brain: Box::new(probe),
                lag,
                label: "probe".into(),
            },
            setup(Puppet::boxed(vec![])),
        ];
        let _ = play_game(&arena, &rules, seed, Layout::default(), players).unwrap();
        let rows = log.lock().unwrap().clone();
        assert!(rows.len() >= 15);
        for r in &rows {
            assert_eq!(r.tick, r.view_tick, "the view is the world the decision is made in");
            assert_eq!(r.obs_x, r.world_x, "the observation is derived from that same world");
            assert_eq!(r.lag, lag);
            assert_eq!(r.in_flight_dirs.len(), lag as usize);
            // The input applied by the *previous* step is what the world's character holds now:
            // the decision made at tick d takes effect from the step at world tick d + lag, so
            // the character's held direction at tick t is that of the last decision d <= t - 1 - lag... precisely:
            // held(t) = decided(d) for the latest decision tick d with d + lag <= t - 1 (the step
            // at t - 1 was the last one run), and 0 before any.
            let expected = rows
                .iter()
                .rfind(|p| p.tick + (lag as i32) < r.tick)
                .map_or(0, |p| if p.tick >= 4 { 1 } else { 0 });
            assert_eq!(r.applied_dir, expected, "lag {lag} tick {}", r.tick);
            // in_flight[k] is what step (tick + k) will apply: decided at d with d + lag <= tick + k.
            for (k, &dir) in r.in_flight_dirs.iter().enumerate() {
                let want = rows
                    .iter()
                    .rfind(|p| p.tick < r.tick && p.tick + lag as i32 <= r.tick + k as i32)
                    .map_or(0, |p| if p.tick >= 4 { 1 } else { 0 });
                assert_eq!(dir, want, "lag {lag} tick {} k {k}", r.tick);
            }
        }
    }
}

fn swing_right() -> Vec<(i32, Action)> {
    vec![
        (
            0,
            Action {
                wanted_weapon: Some(0),
                target: IVec2::new(100, 0),
                ..Action::neutral()
            },
        ),
        (
            2,
            Action {
                fire: true,
                wanted_weapon: Some(0),
                target: IVec2::new(100, 0),
                ..Action::neutral()
            },
        ),
    ]
}

/// A hammer blow that throws the victim into a freeze pit is credited to the hammerer
/// (`HammerHit` from the physics adapter's derivation), and the throw is what freezes it.
#[test]
fn hammer_throw_into_freeze_is_credited() {
    let arena = arena_with_two_tiles((20, 30), 17, 18);
    let seed = seed_for(&arena, 17, 18);
    let r = play(
        &arena,
        &Rules::default(),
        seed,
        false,
        vec![Puppet::boxed(swing_right()), Puppet::boxed(vec![])],
    );
    assert_eq!(r.result, GameResult::W, "{r:?}");
    assert_eq!(r.victim, 1);
    assert!(r.credited, "the hammer hit must be credited: {r:?}");
    assert_eq!(r.blocks_by_a, 1);
    assert_eq!(r.first_block_tick, Some(r.end_tick));
}

/// Same throw, but the victim is the hammerer's *own* team mate slot order reversed: B hammers A
/// into the pit, so it is a loss and the credit goes to the opponent.
#[test]
fn being_hammered_into_freeze_is_a_credited_loss() {
    let arena = arena_with_two_tiles((20, 30), 18, 17);
    let seed = seed_for(&arena, 18, 17);
    let r = play(
        &arena,
        &Rules::default(),
        seed,
        false,
        vec![Puppet::boxed(vec![]), Puppet::boxed(swing_right())],
    );
    assert_eq!(r.result, GameResult::L, "{r:?}");
    assert_eq!(r.victim, 0);
    assert!(r.credited, "{r:?}");
    assert_eq!(r.a_self_freezes, 1);
}

/// A hook drag across a freeze pit is credited through the hooker's `hooked_player`.
#[test]
fn hook_drag_into_freeze_is_credited() {
    let arena = corridor(24, 28, &[(20, 20), (31, 31)]);
    let seed = seed_for(&arena, 20, 31);
    let hook_and_walk_away = Action {
        direction: -1,
        hook: true,
        target: IVec2::new(300, 0),
        ..Action::neutral()
    };
    let r = play(
        &arena,
        &Rules::default(),
        seed,
        false,
        vec![Puppet::boxed(vec![(0, hook_and_walk_away)]), Puppet::boxed(vec![])],
    );
    assert_eq!(r.result, GameResult::W, "{r:?}");
    assert!(r.credited && r.held, "{r:?}");
    assert_eq!(r.blocks_by_a, 1);
    assert_eq!(r.a_self_freezes, 0);
}

/// Three players. Slot 1 walks into the pit alone (nobody touched it): in a 1vN game that is
/// recorded and the game goes on; the game then ends as a timeout.
#[test]
fn one_v_n_uncredited_opponent_out_is_recorded_and_the_game_continues() {
    let arena = corridor(20, 21, &[(3, 3), (18, 18), (5, 5)]);
    let seed = (0..10_000u64)
        .find(|&s| {
            let t = arena.spawn_tiles(s, 3).unwrap();
            t[0].0 == 3 && t[1].0 == 18 && t[2].0 == 5
        })
        .expect("a seed with that spawn exists");
    let rules = Rules {
        max_ticks: 300,
        ..Rules::default()
    };
    let r = play(
        &arena,
        &rules,
        seed,
        false,
        vec![
            Puppet::boxed(vec![]),
            Puppet::boxed(vec![(0, walk(1))]),
            Puppet::boxed(vec![]),
        ],
    );
    assert_eq!(r.result, GameResult::T, "{r:?}");
    assert_eq!(r.bystander_outs, 1, "{r:?}");
    assert_eq!(r.blocks_by_a, 0);
    assert_eq!(r.players.len(), 3);
}

/// Same three players, but the focal player itself walks into the pit: a loss, decided on its own
/// onset, regardless of the other players.
#[test]
fn one_v_n_focal_out_is_a_loss() {
    let arena = corridor(20, 21, &[(15, 15), (3, 3), (5, 5)]);
    let seed = (0..10_000u64)
        .find(|&s| {
            let t = arena.spawn_tiles(s, 3).unwrap();
            t[0].0 == 15 && t[1].0 == 3 && t[2].0 == 5
        })
        .expect("a seed with that spawn exists");
    let r = play(
        &arena,
        &Rules::default(),
        seed,
        false,
        vec![
            Puppet::boxed(vec![(0, walk(1))]),
            Puppet::boxed(vec![]),
            Puppet::boxed(vec![]),
        ],
    );
    assert_eq!(r.result, GameResult::L, "{r:?}");
    assert_eq!(r.victim, 0);
    assert!(!r.credited, "nobody touched the focal player: {r:?}");
}

/// With `credit_required = false` even in a 1vN game an uncredited opponent onset decides the
/// game (the 1v1 rule), and with `true` a 1v1 game ignores an uncredited onset.
#[test]
fn credit_required_can_be_forced_either_way() {
    let arena = corridor(20, 21, &[(3, 3), (18, 18)]);
    let seed = seed_for(&arena, 3, 18);
    let walker = || vec![Puppet::boxed(vec![]), Puppet::boxed(vec![(0, walk(1))])];
    let rules_default = Rules::default();
    let r = play(&arena, &rules_default, seed, false, walker());
    assert_eq!(r.result, GameResult::W);
    let strict = Rules {
        credit_required: Some(true),
        max_ticks: 200,
        ..Rules::default()
    };
    let r = play(&arena, &strict, seed, false, walker());
    assert_eq!(r.result, GameResult::T, "an uncredited onset must not decide: {r:?}");
    assert_eq!(r.bystander_outs, 1);
}

/// The decision hash is a pure function of the decision stream.
#[test]
fn decision_hash_follows_the_decisions() {
    let arena = arena_with_two_tiles((20, 21), 5, 8);
    let seed = seed_for(&arena, 5, 8);
    let rules = Rules {
        max_ticks: 40,
        after_ticks: 0,
        ..Rules::default()
    };
    let run = |script: Vec<(i32, Action)>| {
        play(
            &arena,
            &rules,
            seed,
            false,
            vec![Puppet::boxed(script), Puppet::boxed(vec![])],
        )
    };
    let a = run(vec![(0, walk(1))]);
    let b = run(vec![(0, walk(1))]);
    let c = run(vec![(0, walk(-1))]);
    assert_eq!(a.players[0].hash, b.players[0].hash);
    assert_ne!(a.players[0].hash, c.players[0].hash);
    assert_eq!(
        a.players[1].hash, c.players[1].hash,
        "the idle opponent's stream is unchanged"
    );
    assert_eq!(a.players[0].hash.len(), 16);
}

/// `held` needs the winner to stay out of freeze for the whole window after the deciding tick.
#[test]
fn held_is_lost_when_the_winner_goes_out_inside_the_window() {
    let arena = arena_with_two_tiles((20, 21), 17, 24);
    let seed = seed_for(&arena, 17, 24);
    let r = play(
        &arena,
        &Rules::default(),
        seed,
        false,
        vec![
            // A idles, then walks into the pit well inside the 150-tick window.
            Puppet::boxed(vec![(0, Action::neutral()), (60, walk(1))]),
            Puppet::boxed(vec![(0, walk(-1))]),
        ],
    );
    assert_eq!(r.result, GameResult::W, "A was not out when B went out: {r:?}");
    assert!(!r.held, "the winner froze inside the window: {r:?}");
    assert_eq!(r.a_self_freezes, 0, "self-freezes count only up to the deciding tick");
}

fn brief(r: &GameReport) -> String {
    format!(
        "{:?} end {} held {} held_block {} escape {:?} out_ticks {} victim {}",
        r.result, r.end_tick, r.held, r.held_block, r.escape_tick, r.victim_out_ticks, r.victim
    )
}

/// Task 3.10, `held_block`: the victim has to be out on every tick of the whole window. A victim that stays in a pit is a held block
/// for a 250-tick window; one frozen in the air that falls onto plain floor is free again 150 ticks later, inside the window: not held, and
/// `escape_tick` names the tick it was free.
#[test]
fn held_block_needs_the_victim_out_on_every_tick_of_the_window() {
    // A one-deep pit of two tiles (a tee that walks in falls to its bottom and freezes) and a freeze bar over the left spawn; A spawns on the right, B on the left.
    let toml = r#"
name = "shallow"
tag = "train"
[map]
kind = "synthetic"
width = 40
height = 16
border = true
rects = [
    { x0 = 1, y0 = 10, x1 = 38, y1 = 13, tile = "solid" },
    { x0 = 20, y0 = 10, x1 = 21, y1 = 10, tile = "freeze" },
    { x0 = 17, y0 = 7, x1 = 17, y1 = 7, tile = "freeze" },
]
[spawn]
min_tiles = 0.0
max_tiles = 100.0
rows = [
    { y = 9, x0 = 24, x1 = 24 },
    { y = 9, x0 = 17, x1 = 17 },
]
"#;
    let arena = Arena::build(&ArenaDef::parse(toml).unwrap(), Path::new("/nonexistent")).unwrap();
    let seed = seed_for(&arena, 24, 17);
    let rules = Rules {
        after_ticks: 250,
        ..Rules::default()
    };
    // B walks right into the pit and stays there.
    let r = play(
        &arena,
        &rules,
        seed,
        false,
        vec![Puppet::boxed(vec![]), Puppet::boxed(vec![(0, walk(1))])],
    );
    assert_eq!(r.result, GameResult::W, "{}", brief(&r));
    assert!(r.held && r.held_block, "B never leaves the pit: {}", brief(&r));
    assert_eq!(r.escape_tick, None);
    assert_eq!(r.victim_out_ticks, 250);

    // B jumps into a freeze bar two tiles over its head: it is frozen in the air, falls back onto the plain floor and thaws 150 ticks
    // later -- inside the window, so not a held block (`escape_tick` names the tick it was free).
    let r = play(
        &arena,
        &rules,
        seed,
        false,
        vec![
            Puppet::boxed(vec![]),
            Puppet::boxed(vec![(
                0,
                Action {
                    jump: true,
                    ..Action::neutral()
                },
            )]),
        ],
    );
    assert_eq!(r.result, GameResult::W, "{}", brief(&r));
    assert!(!r.held_block, "B thawed inside the window: {}", brief(&r));
    let t = r.escape_tick.expect("B was free again");
    assert!(
        t >= r.end_tick + 100 && t <= r.end_tick + 200,
        "about the 150 ticks of its freeze: {}",
        brief(&r)
    );
    assert!(r.victim_out_ticks >= 100 && r.victim_out_ticks < 250, "{}", brief(&r));

    // A lost game has a victim too (slot 0): held when A stays in the pit.
    let r = play(
        &arena,
        &rules,
        seed,
        false,
        vec![Puppet::boxed(vec![(0, walk(-1))]), Puppet::boxed(vec![])],
    );
    assert_eq!(r.result, GameResult::L, "{}", brief(&r));
    assert!(r.held_block);
    // A timeout has no victim: never held.
    let r = play(
        &arena,
        &rules,
        seed,
        false,
        vec![Puppet::boxed(vec![]), Puppet::boxed(vec![])],
    );
    assert_eq!(r.result, GameResult::T, "{}", brief(&r));
    assert!(!r.held_block && r.escape_tick.is_none());
}

fn play_layout(arena: &Arena, seed: u64, layout: Layout, players: Vec<Box<dyn Brain>>) -> GameReport {
    play_game(
        arena,
        &Rules::default(),
        seed,
        layout,
        players.into_iter().map(setup).collect(),
    )
    .unwrap()
}

/// Spawn order is a real asymmetry (strong/weak hook): the same two hook-and-walk-away brains give
/// a different game when the tees are spawned in the other order -- which is why the arena
/// balances it (F1).
#[test]
fn spawn_order_changes_the_game_and_the_report_says_which_order_was_played() {
    let arena = corridor(24, 28, &[(20, 20), (31, 31)]);
    let seed = seed_for(&arena, 20, 31);
    let a = Action {
        direction: -1,
        hook: true,
        target: IVec2::new(300, 0),
        ..Action::neutral()
    };
    let b = Action {
        direction: 1,
        hook: true,
        target: IVec2::new(-300, 0),
        ..Action::neutral()
    };
    let players = || vec![Puppet::boxed(vec![(0, a)]), Puppet::boxed(vec![(0, b)])];
    let normal = play_layout(&arena, seed, Layout::default(), players());
    let reversed = play_layout(
        &arena,
        seed,
        Layout {
            swap: false,
            reverse_order: true,
        },
        players(),
    );
    assert!(!normal.reverse_order && reversed.reverse_order);
    assert_eq!(
        normal.spawns, reversed.spawns,
        "positions are the same, only the order differs"
    );
    assert_eq!(normal.result, GameResult::W, "{normal:?}");
    assert_eq!(
        reversed.result,
        GameResult::L,
        "the strong hook moved to the other player: {reversed:?}"
    );
}

/// Every block of four games crosses position (swap) with spawn order.
#[test]
fn the_batch_layout_is_balanced_over_position_and_spawn_order() {
    use ddai_env::config::{PlayerSpec, builtin_brain};
    use ddai_env::run::play_indexed;
    let arena = arena_with_two_tiles((20, 21), 5, 8);
    let slots = vec![PlayerSpec::simple("idle"), PlayerSpec::simple("idle")];
    let rules = Rules {
        max_ticks: 4,
        after_ticks: 0,
        ..Rules::default()
    };
    let mut cells = std::collections::BTreeMap::new();
    for g in 0..40u32 {
        let r = play_indexed(&arena, &rules, &slots, &builtin_brain, 1, g).unwrap();
        *cells.entry((r.swap, r.reverse_order)).or_insert(0) += 1;
    }
    assert_eq!(cells.len(), 4);
    assert!(cells.values().all(|&n| n == 10), "{cells:?}");
}

/// 1vN: a loss nobody was credited for is `held` only if no opponent was out in the window.
#[test]
fn one_v_n_uncredited_loss_needs_every_opponent_to_stay_in_the_window() {
    let arena = corridor(20, 21, &[(15, 15), (5, 5), (3, 3)]);
    let seed = (0..10_000u64)
        .find(|&s| {
            let t = arena.spawn_tiles(s, 3).unwrap();
            t[0].0 == 15 && t[1].0 == 5 && t[2].0 == 3
        })
        .expect("a seed with that spawn exists");
    let quiet = play(
        &arena,
        &Rules::default(),
        seed,
        false,
        vec![
            Puppet::boxed(vec![(0, walk(1))]),
            Puppet::boxed(vec![]),
            Puppet::boxed(vec![]),
        ],
    );
    assert_eq!(quiet.result, GameResult::L);
    assert!(quiet.held, "the victim stays frozen and nobody else moves: {quiet:?}");
    let noisy = play(
        &arena,
        &Rules::default(),
        seed,
        false,
        vec![
            Puppet::boxed(vec![(0, walk(1))]),
            Puppet::boxed(vec![(0, Action::neutral()), (40, walk(1))]),
            Puppet::boxed(vec![]),
        ],
    );
    assert_eq!(noisy.result, GameResult::L);
    assert!(!noisy.held, "an opponent froze inside the window: {noisy:?}");
}

/// A killed tee is not asked for decisions any more (its client keeps sending the previous input,
/// like the harness's `prevInput`; see `Sim::step`).
#[test]
fn a_dead_player_is_not_asked_for_decisions() {
    let toml = r#"
name = "deathpit"
tag = "train"
[map]
kind = "synthetic"
width = 40
height = 16
border = true
rects = [
    { x0 = 1, y0 = 10, x1 = 38, y1 = 13, tile = "solid" },
    { x0 = 20, y0 = 10, x1 = 21, y1 = 11, tile = "death" },
]
[spawn]
min_tiles = 0.0
max_tiles = 100.0
rows = [ { y = 9, x0 = 15, x1 = 15 }, { y = 9, x0 = 3, x1 = 3 } ]
"#;
    let death = Arena::build(&ArenaDef::parse(toml).unwrap(), Path::new("/nonexistent")).unwrap();
    let rules = Rules {
        max_ticks: 400,
        after_ticks: 100,
        ..Rules::default()
    };
    let r = play_game(
        &death,
        &rules,
        1,
        Layout::default(),
        vec![setup(Puppet::boxed(vec![(0, walk(1))])), setup(Puppet::boxed(vec![]))],
    )
    .unwrap();
    assert_eq!(r.result, GameResult::L, "{r:?}");
    // 500 ticks played at most; the dead player is asked only until it died.
    assert!(r.players[0].decisions < r.players[1].decisions, "{r:?}");
}

/// Task 3.10: the facts a training reward is made of.
#[test]
fn held_outcome_pays_a_strict_held_win_only_and_the_window_helper_only_ever_lengthens() {
    use ddai_env::game::HeldOutcome;
    let o = |result, held_block, out| HeldOutcome {
        result,
        credited: true,
        held_block,
        escape_tick: (!held_block).then_some(100),
        victim_out_ticks: out,
        winner_out_in_window: false,
        focal_out_in_window: result == GameResult::L,
        window_ticks: 250,
    };
    assert!(o(GameResult::W, true, 250).strict_held_win());
    assert_eq!(o(GameResult::W, true, 250).held_return(), 1.0);
    assert_eq!(o(GameResult::L, true, 250).held_return(), -1.0);
    assert_eq!(
        o(GameResult::W, false, 125).held_return(),
        0.0,
        "a block that thawed earns nothing, however long it lasted"
    );
    assert_eq!(
        o(GameResult::L, false, 0).held_return(),
        -1.0,
        "a lost game: the focal player was out"
    );
    // The D-059 trap: a held victim nobody credited the focal player for (it froze itself) is no win.
    let uncredited = HeldOutcome {
        credited: false,
        ..o(GameResult::W, true, 250)
    };
    assert!(!uncredited.strict_held_win());
    assert_eq!(uncredited.held_return(), 0.0);
    // The focal player out inside the window (it froze after its block): not strict, and the worst outcome.
    let self_froze = HeldOutcome {
        focal_out_in_window: true,
        ..o(GameResult::W, true, 250)
    };
    assert!(!self_froze.strict_held_win());
    assert_eq!(self_froze.held_return(), -1.0);
    assert_eq!(o(GameResult::T, false, 0).held_return(), 0.0);
    assert_eq!(Rules::default().held_block_window().after_ticks, 250);
    let long = Rules {
        after_ticks: 400,
        ..Rules::default()
    };
    assert_eq!(long.held_block_window().after_ticks, 400);
}

/// Task 3.10: the crowd target rule of the finishing switch keeps a frozen current target until it is held.
#[test]
fn hold_target_keeps_a_frozen_target_on_open_floor_and_lets_a_held_one_go() {
    use ddai_env::sim::{HoldTarget, default_target};
    use ddai_physics::vmath::Vec2 as V;
    use ddai_physics::world::spawn_character;
    // The shared pit corridor: a 2-deep pit at x 20..21; the floor is row 10 (tees stand on row 9, y = 304 - 14).
    let arena = corridor(20, 21, &[(5, 5), (10, 10), (14, 14)]);
    let mut world = arena.new_world();
    let floor_y = 9.0 * 32.0 + 2.0;
    let ids = [0, 1, 2];
    spawn_character(&mut world, 0, V::new(5.5 * 32.0, floor_y));
    spawn_character(&mut world, 1, V::new(10.5 * 32.0, floor_y));
    spawn_character(&mut world, 2, V::new(14.5 * 32.0, floor_y));
    let hold = HoldTarget::new(arena.map.clone());
    // Both free: the nearest (tee 1), as the default rule.
    assert_eq!(hold.pick(&world, 0, &ids), Some(1));
    assert_eq!(default_target(&world, 0, &ids), Some(1));
    // Tee 1 freezes on open floor: the default rule goes to the free tee 2, the hold rule stays (it thaws in 150 ticks).
    world.characters[1].as_mut().unwrap().freeze_time = 150;
    assert_eq!(default_target(&world, 0, &ids), Some(2));
    assert_eq!(hold.pick(&world, 0, &ids), Some(1));
    // Other slots follow the default rule (they target the focal player).
    assert_eq!(hold.pick(&world, 1, &ids), Some(0));
    // Tee 1 lying frozen in the pit is held for the whole window: nothing left to finish, tee 2 is the target.
    let mut pit_world = arena.new_world();
    spawn_character(&mut pit_world, 0, V::new(5.5 * 32.0, floor_y));
    spawn_character(&mut pit_world, 1, V::new(20.5 * 32.0, 10.5 * 32.0));
    spawn_character(&mut pit_world, 2, V::new(14.5 * 32.0, floor_y));
    let hold = HoldTarget::new(arena.map.clone());
    // Choose tee 1 first (free), then freeze it in the pit.
    pit_world.cores.get_mut(2u8).unwrap().pos = V::new(25.5 * 32.0, floor_y);
    assert_eq!(hold.pick(&pit_world, 0, &ids), Some(1));
    pit_world.characters[1].as_mut().unwrap().freeze_time = 150;
    pit_world.cores.get_mut(2u8).unwrap().pos = V::new(14.5 * 32.0, floor_y);
    assert_eq!(
        hold.pick(&pit_world, 0, &ids),
        Some(2),
        "held in the pit: the free tee is taken"
    );
}

/// Task 3.10 (review F3): the summary's held W counts only wins by the focal player's own credited block; an opponent that froze itself into a pit and
/// stayed there is a held victim but not a held win (`held_block_w_any` keeps the raw count).
#[test]
fn the_summary_counts_a_held_win_only_when_it_was_credited() {
    use ddai_env::config::{Condition, PlayerSpec};
    use ddai_env::report::summarize;
    use ddai_env::run::ConditionRun;
    let arena = arena_with_two_tiles((20, 21), 17, 24);
    let seed = seed_for(&arena, 17, 24);
    let rules = Rules {
        after_ticks: 250,
        ..Rules::default()
    };
    // B walks into the pit on its own: a win nobody earned.
    let r = play(
        &arena,
        &rules,
        seed,
        false,
        vec![Puppet::boxed(vec![]), Puppet::boxed(vec![(0, walk(-1))])],
    );
    assert!(
        r.result == GameResult::W && !r.credited && r.held_block,
        "{}",
        brief(&r)
    );
    let run = ConditionRun {
        condition: Condition {
            name: "t".into(),
            arena: "corridor".into(),
            games: None,
            rules: None,
            players: vec![PlayerSpec::simple("idle"), PlayerSpec::simple("idle")],
        },
        arena: "corridor".into(),
        games: vec![r],
        wall_s: 0.0,
    };
    let s = summarize(&run, "train", None);
    assert_eq!((s.held_block_w, s.held_block_w_any), (0, 1));
    assert_eq!(s.credited_w, 0);
}
