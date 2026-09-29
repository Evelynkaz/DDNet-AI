//! One game: a port of the phase-0 harness's `runGame` (`lib.mjs`, `orig-run.md` §3) generalised
//! to N players (1vN), running on the bit-exact `ddai_physics::World<f32>`.
//!
//! Rules (all counted in world ticks):
//! * Brains decide every `decide_every` ticks, all from the same pre-step state; a decision takes
//!   effect after the player's input lag (`lib.mjs`'s lag queue: decided at tick `T`, applied from
//!   the step at world tick `T + lag`).
//! * A tee is *out* when it is dead or frozen; an *onset* is the tick it becomes out.
//! * 1v1: the first tick with an onset decides -- only B out is `W`, only A out `L`, both `D`. No
//!   onset within `max_ticks`: `T`.
//! * 1vN (focal A vs B1..Bk): `W` when a Bi goes out *credited to A*, `L` when A goes out, `D`
//!   when both on the same tick, `T` at `max_ticks`. A Bi going out without credit to A is only
//!   recorded (`bystander_outs`) and the game goes on.
//! * `credited`: the victim was hooked or hammered by the winner at most `credit_ticks` before
//!   the onset. `held`: `after_ticks` more ticks are played; the victim is still out at the end
//!   and the winner was never out in that window.
//!
//! Hammer credit comes from `ddai_planner::physics_adapter::PhysicsWorld::step`'s
//! `HammerHit` derivation (task 3.2); hook credit from the hooker's `hooked_player` each tick.

use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::WorldEvent;
use ddai_planner::vmath::Vec2;
use serde::Serialize;

use crate::EnvError;
use crate::arena::{Arena, tile_center};
use crate::config::Rules;
use crate::observe;
use crate::sim::{PlayerSetup, Sim, default_target};
use crate::stats::{GameResult, percentile_u32};

/// How a game's seed is turned into a starting layout. Both flips exist because both matter:
/// position (which side of the hall) and spawn order (slot 0 is always client id 0, and the tee
/// spawned *last* is processed first in the tick and holds the strong hook, `m_StrongWeakId`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Layout {
    /// Trade the spawn positions of the focal player and the first opponent (harness `swap`).
    pub swap: bool,
    /// Spawn the tees in reverse slot order (last slot first); client ids stay equal to the slot.
    pub reverse_order: bool,
}

/// What one player did during a game.
#[derive(Debug, Clone, Serialize)]
pub struct PlayerReport {
    pub slot: usize,
    pub label: String,
    pub lag: u32,
    /// Number of times the brain was asked for a decision (dead tees are not asked).
    pub decisions: u32,
    /// First 16 hex digits of the SHA-1 of the decision stream (`orig-run.md` §9): for each
    /// decision tick `"{tick}:{direction},{jump},{hook},{fire},{target_x},{target_y},{wanted_weapon};"`
    /// with the wire values sent to the world. Equal for equal seeds when the brain is
    /// deterministic (not for wall-clock deadline planners).
    pub hash: String,
    /// The brain's own telemetry (`Brain::telemetry`, JSON), if any.
    pub telemetry: Option<serde_json::Value>,
}

/// Wall-clock decision times; the only non-deterministic part of a [`GameReport`].
#[derive(Debug, Clone, Serialize)]
pub struct Timing {
    /// Per player, microseconds; `None` when the player made no decision.
    pub decide_us_p50: Vec<Option<u32>>,
    pub decide_us_p99: Vec<Option<u32>>,
}

/// The record of one game (one JSONL line, plus the `condition`/`game` keys added by the batch).
#[derive(Debug, Clone, Serialize)]
pub struct GameReport {
    pub seed: u64,
    /// Sides swapped: the focal player and the first opponent traded spawn positions.
    pub swap: bool,
    /// The tees were spawned in reverse slot order (see [`Layout`]).
    pub reverse_order: bool,
    pub result: GameResult,
    /// Tick of the deciding onset (for `T`: `max_ticks`).
    pub end_tick: i32,
    pub credited: bool,
    pub held: bool,
    /// Slot of the victim (`W`: the opponent that went out, `L`: 0), `-1` for `D`/`T`.
    pub victim: i32,
    /// Spawn positions in pixels, by slot.
    pub spawns: Vec<[f32; 2]>,
    /// Tick the focal player first went out (dead or frozen), `None` if it never did before the
    /// game was decided (a censored survival).
    pub a_out_tick: Option<i32>,
    /// Onsets of the focal player up to and including the deciding tick.
    pub a_self_freezes: u32,
    /// Opponent onsets credited to the focal player up to and including the deciding tick.
    pub blocks_by_a: u32,
    /// Tick of the first such block.
    pub first_block_tick: Option<i32>,
    /// Opponent onsets *not* credited to the focal player before the game was decided (1vN only).
    pub bystander_outs: u32,
    pub players: Vec<PlayerReport>,
    pub timing: Timing,
    /// Raw decision times per player, for pooling into batch-level percentiles.
    #[serde(skip)]
    pub decide_us: Vec<Vec<u32>>,
}

/// A last-touch record: who hooked/hammered a tee, and when.
type Touch = Option<(usize, i32)>;

fn credited_touch(touch: Touch, by: usize, now: i32, credit_ticks: i32) -> bool {
    touch.is_some_and(|(b, t)| b == by && now - t <= credit_ticks)
}

/// Plays one game. `players[0]` is the focal player. Brains are reset with
/// `seed + 100000 * slot` (the harness's `seed` / `seed + 100000` for two players).
pub fn play_game(
    arena: &Arena,
    rules: &Rules,
    seed: u64,
    layout: Layout,
    players: Vec<PlayerSetup>,
) -> Result<GameReport, EnvError> {
    let n = players.len();
    if n < 2 {
        return Err(EnvError::new("a game needs at least two players"));
    }
    if n > ddai_physics::core::MAX_CLIENTS {
        return Err(EnvError::new("too many players"));
    }
    let credit_required = rules.credit_required.unwrap_or(n > 2);
    let mut spawn = arena.spawn_tiles(seed, n)?;
    if layout.swap {
        spawn.swap(0, 1);
    }
    let mut pw = PhysicsWorld::from_world(arena.new_world(), arena.map.clone());
    let mut spawns = vec![[0.0f32; 2]; n];
    let order: Vec<usize> = if layout.reverse_order {
        (0..n).rev().collect()
    } else {
        (0..n).collect()
    };
    for i in order {
        let (x, y) = (tile_center(spawn[i].0), tile_center(spawn[i].1));
        pw.add_tee(
            i as i32,
            Vec2 {
                x: f64::from(x),
                y: f64::from(y),
            },
        );
        spawns[i] = [x, y];
    }
    let mut sim = Sim::new(pw, arena.map.clone(), players, rules.decide_every, seed);
    let mut last_touch: Vec<Touch> = vec![None; n];
    let mut was_out = vec![false; n];

    let mut result: Option<GameResult> = None;
    let mut end_tick = -1;
    let mut victim: i32 = -1;
    let mut winner: Option<usize> = None;
    let mut credited = false;
    let mut winner_out_after = false;
    let mut a_out_tick: Option<i32> = None;
    let mut a_self_freezes = 0u32;
    let mut blocks_by_a = 0u32;
    let mut first_block_tick: Option<i32> = None;
    let mut bystander_outs = 0u32;

    let limit = rules.max_ticks + rules.after_ticks;
    for _ in 0..limit {
        let events = sim.step(&default_target);
        let now = sim.tick();
        for e in events {
            if let WorldEvent::HammerHit { from, to } = e
                && (0..n as i32).contains(&from)
                && (0..n as i32).contains(&to)
            {
                last_touch[to as usize] = Some((from as usize, now));
            }
        }
        for i in 0..n {
            let h = observe::hooked_player(sim.pw.inner(), sim.ids[i]);
            if h >= 0 && (h as usize) < n {
                last_touch[h as usize] = Some((i, now));
            }
        }
        let out_now: Vec<bool> = sim.ids.iter().map(|&id| observe::is_out(sim.pw.inner(), id)).collect();
        let onset: Vec<bool> = (0..n).map(|i| out_now[i] && !was_out[i]).collect();

        if result.is_none() {
            if onset[0] {
                a_self_freezes += 1;
                a_out_tick.get_or_insert(now);
            }
            // Opponent onsets, split by whether the focal player gets credit for them.
            let mut credited_b: Vec<usize> = Vec::new();
            let mut uncredited_b = 0u32;
            for j in 1..n {
                if onset[j] {
                    if credited_touch(last_touch[j], 0, now, rules.credit_ticks) {
                        credited_b.push(j);
                    } else {
                        uncredited_b += 1;
                    }
                }
            }
            blocks_by_a += credited_b.len() as u32;
            if !credited_b.is_empty() {
                first_block_tick.get_or_insert(now);
            }
            let b_decides = if credit_required {
                !credited_b.is_empty()
            } else {
                !credited_b.is_empty() || uncredited_b > 0
            };
            if credit_required {
                bystander_outs += uncredited_b;
            }
            if onset[0] || b_decides {
                end_tick = now;
                if onset[0] && b_decides {
                    result = Some(GameResult::D);
                } else if onset[0] {
                    result = Some(GameResult::L);
                    victim = 0;
                    winner = if n == 2 {
                        Some(1)
                    } else {
                        last_touch[0].and_then(|(by, t)| (now - t <= rules.credit_ticks).then_some(by))
                    };
                    credited = winner.is_some_and(|w| credited_touch(last_touch[0], w, now, rules.credit_ticks));
                } else {
                    result = Some(GameResult::W);
                    let v = (1..n).find(|&j| onset[j]).expect("an opponent went out");
                    // With credit required, prefer a credited victim (the first one, by slot).
                    let v = credited_b.first().copied().unwrap_or(v);
                    victim = v as i32;
                    winner = Some(0);
                    credited = credited_touch(last_touch[v], 0, now, rules.credit_ticks);
                }
            } else if now >= rules.max_ticks {
                result = Some(GameResult::T);
                end_tick = now;
                break;
            }
        } else {
            if victim >= 0 {
                match winner {
                    Some(w) => winner_out_after |= out_now[w],
                    // A 1vN loss nobody was credited for: "the winner never out" means no
                    // opponent at all was out during the window.
                    None => winner_out_after |= out_now[1..].iter().any(|&o| o),
                }
            }
            if now >= end_tick + rules.after_ticks {
                break;
            }
        }
        was_out.clone_from(&out_now);
    }
    let result = result.unwrap_or_else(|| {
        end_tick = sim.tick();
        GameResult::T
    });
    let held = victim >= 0 && observe::is_out(sim.pw.inner(), sim.ids[victim as usize]) && !winner_out_after;

    let mut reports = Vec::with_capacity(n);
    let mut p50 = Vec::with_capacity(n);
    let mut p99 = Vec::with_capacity(n);
    for (i, p) in sim.players.iter().enumerate() {
        p50.push(percentile_u32(&sim.decide_us[i], 50.0));
        p99.push(percentile_u32(&sim.decide_us[i], 99.0));
        reports.push(PlayerReport {
            slot: i,
            label: p.label.clone(),
            lag: p.lag,
            decisions: sim.decide_us[i].len() as u32,
            hash: sim.decision_hash(i),
            telemetry: p
                .brain
                .telemetry()
                .map(|t| serde_json::from_str(&t).unwrap_or(serde_json::Value::String(t))),
        });
    }
    Ok(GameReport {
        seed,
        swap: layout.swap,
        reverse_order: layout.reverse_order,
        result,
        end_tick,
        credited,
        held,
        victim,
        spawns,
        a_out_tick,
        a_self_freezes,
        blocks_by_a,
        first_block_tick,
        bystander_outs,
        players: reports,
        timing: Timing {
            decide_us_p50: p50,
            decide_us_p99: p99,
        },
        decide_us: sim.decide_us,
    })
}
