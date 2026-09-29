//! Review round 2, F12: "measure whether falling back to `prev` is ever better [than keeping the
//! searched `chosen` input], and justify your choice." This is that measurement: an A/B game
//! comparison, same seeds, same everything, differing only in what `decide_production`'s caller
//! substitutes whenever `last_info.shield_incomplete` is `true` (the shield ran out of time before
//! confirming an escape or a safer alternative) -- `chosen` (this crate's actual choice, made
//! entirely at the call site by simply keeping the returned decision) vs `prev` (substituted here,
//! *without* touching `decide_production` itself, precisely so this stays an honest A/B test of
//! the policy rather than a test of two different implementations).
//!
//! To get a large enough sample of `shield_incomplete` decisions to compare, `SHIELD_RESERVE_MS`
//! would need to be tiny -- which it already effectively is relative to a busy 6-tee scene's
//! per-step cost (`phase_breakdown`'s own numbers: `shield_incomplete` fired in 334/994 and 98/902
//! decisions at tees=6, budget=4ms, hall=false/true respectively -- no artificial shrinking
//! needed).
//!
//! `#[ignore]`d: needs the local Copy Love Box map. Run:
//! ```text
//! N=150 cargo test -p ddai-planner --release --test shield_fallback_experiment -- --ignored --nocapture
//! ```

use ddai_jsmath::Rng;
use ddai_planner::clock::WallClock;
use ddai_planner::config::PlannerConfig;
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::{PlanCollision, PlanWorld};
use ddai_planner::planner::Planner;
use ddai_planner::scripted::scripted_action;
use ddai_planner::types::empty_input;
use ddai_planner::vmath::Vec2;
use std::sync::Arc;

const CLB: &str = "/home/ubuntu/aiddnet/data/maps/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map";

fn stand(col: &impl PlanCollision) -> Vec<Vec2> {
    let c = |t: i32| f64::from(t) * 32.0 + 16.0;
    let mut out = Vec::new();
    for ty in 0..col.height() - 1 {
        for tx in 0..col.width() {
            let (px, py) = (c(tx), c(ty));
            if col.is_solid(px, py) || col.is_freeze(px, py) || col.is_death(px, py) {
                continue;
            }
            if !col.is_solid(px, py + 32.0) {
                continue;
            }
            out.push(Vec2 { x: px, y: py });
        }
    }
    out
}

fn pair(st: &[Vec2], rng: &mut Rng) -> (Vec2, Vec2) {
    let pick = |rng: &mut Rng| st[(rng.next_float() * st.len() as f64) as usize];
    for _ in 0..10_000 {
        let a = pick(rng);
        let b = pick(rng);
        let d = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt() / 32.0;
        if (3.0..=14.0).contains(&d) {
            return (a, b);
        }
    }
    (pick(rng), pick(rng))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    SelfWins,
    EnemyWins,
    Draw,
    Timeout,
}
const MAX_TICKS: i32 = 800;

/// Plays one game against the scripted bot, on a 6-tee scene (2 extra stationary bystanders --
/// `phase_breakdown` measured `shield_incomplete` far more often at tees=6 than tees=2), budget 4
/// ms. `fallback_to_prev`: when `true`, every tick where `last_info.shield_incomplete` fires, the
/// input actually sent to the world is `self_prev` instead of the planner's own `chosen` return
/// value -- the *only* difference between the two arms of this experiment.
fn play_game(
    map: &Arc<ddai_physics::map::MapData>,
    stand: &[Vec2],
    seed: u64,
    clock: &WallClock,
    fallback_to_prev: bool,
) -> (Outcome, u32, u32) {
    let mut spawn_rng = Rng::new((seed.wrapping_mul(2_654_435_761) & 0xFFFF_FFFF) as u32);
    let mut world = PhysicsWorld::new(map.clone(), seed);
    let (a, b) = pair(stand, &mut spawn_rng);
    world.add_tee(0, a);
    world.add_tee(1, b);
    for id in 2..4 {
        let p = stand[(spawn_rng.next_float() * stand.len() as f64) as usize];
        world.add_tee(id, p);
        world.set_held_input(id, empty_input());
    }
    let mut planner: Planner<PhysicsWorld> = Planner::new(PlannerConfig::default());
    planner.reset();
    let mut script_rng = Rng::new((seed.wrapping_mul(7919).wrapping_add(17) & 0xFFFF_FFFF) as u32);

    let mut self_prev = empty_input();
    let mut enemy_prev = empty_input();
    let mut self_was_out = false;
    let mut enemy_was_out = false;
    let mut incomplete_n = 0u32;
    let mut substituted_n = 0u32;

    for _tick in 0..MAX_TICKS {
        let (Some(st), Some(et)) = (world.get_tee(0), world.get_tee(1)) else {
            break;
        };
        let self_out = st.frozen || !st.alive;
        let enemy_out = et.frozen || !et.alive;
        if self_out && !self_was_out && enemy_out && !enemy_was_out {
            return (Outcome::Draw, incomplete_n, substituted_n);
        }
        if self_out && !self_was_out {
            return (Outcome::EnemyWins, incomplete_n, substituted_n);
        }
        if enemy_out && !enemy_was_out {
            return (Outcome::SelfWins, incomplete_n, substituted_n);
        }
        self_was_out = self_out;
        enemy_was_out = enemy_out;

        let enemy_input = scripted_action(&world, 1, 0, &enemy_prev, &mut script_rng);
        let chosen = planner.decide_production(&mut world, 0, 1, self_prev, enemy_input, clock, 4.0);
        if planner.last_info.shield_incomplete {
            incomplete_n += 1;
        }
        let self_input = if fallback_to_prev && planner.last_info.shield_incomplete {
            substituted_n += 1;
            self_prev
        } else {
            chosen
        };

        world.set_input(0, self_input);
        world.set_input(1, enemy_input);
        world.step();
        self_prev = self_input;
        enemy_prev = enemy_input;
    }
    (Outcome::Timeout, incomplete_n, substituted_n)
}

fn games_env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[test]
#[ignore = "needs the local Copy Love Box map -- see this file's doc comment"]
fn keeping_chosen_vs_falling_back_to_prev_on_shield_incomplete() {
    let bytes = std::fs::read(CLB)
        .unwrap_or_else(|e| panic!("reading {CLB}: {e} (local map data, not committed -- see CLAUDE.md)"));
    let map = Arc::new(ddai_map::load_map(&bytes).unwrap().data);
    let world = PhysicsWorld::new(map.clone(), 1);
    let st = stand(world.collision());
    let n = games_env("N", 150);
    let clock = WallClock::new();

    for (label, fallback) in [
        ("keep chosen (this crate's actual policy)", false),
        ("fall back to prev", true),
    ] {
        let (mut wins, mut losses, mut draws, mut timeouts) = (0u32, 0u32, 0u32, 0u32);
        let (mut total_incomplete, mut total_substituted) = (0u64, 0u64);
        for g in 0..n {
            let (outcome, incomplete_n, substituted_n) = play_game(&map, &st, 1000 + g as u64, &clock, fallback);
            total_incomplete += u64::from(incomplete_n);
            total_substituted += u64::from(substituted_n);
            match outcome {
                Outcome::SelfWins => wins += 1,
                Outcome::EnemyWins => losses += 1,
                Outcome::Draw => draws += 1,
                Outcome::Timeout => timeouts += 1,
            }
        }
        eprintln!(
            "{label}: W={wins} L={losses} D={draws} timeout={timeouts} (n={n}) shield_incomplete decisions={total_incomplete} (substituted={total_substituted})"
        );
    }
}
