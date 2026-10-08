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
//! * `held_block` (task 3.10, the D-059 amendment "held block"): the strict form of `held`. Over the whole window of
//!   `after_ticks` the victim was out on **every** tick (frozen or dead, never thawed in between), whatever the winner
//!   did; `escape_tick` is the first tick of the window the victim was free again. The metric needs
//!   `after_ticks` of at least 250 (5 s, more than `sv_freeze_delay` = 3 s: a freeze that nobody keeps up thaws inside it).
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
use crate::duel::DuelSpec;
use crate::observe;
use crate::sim::{HoldTarget, PlayerSetup, Sim, default_target};
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
    /// Task 3.16: with an input-lag model ([`crate::sim::LagModel`]), the decisions by the lag in ticks they got (index = ticks, the last bin takes
    /// more) and how many of them were applied later than the brain planned for; empty without a model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub lag_hist: Vec<u32>,
    #[serde(skip_serializing_if = "is_zero")]
    pub lag_late: u32,
    /// Task 3.16: the decision costs the lag model saw, a histogram in bins of [`crate::sim::COST_BIN_MS`] (empty without a model).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub lag_cost_hist: Vec<u32>,
}

fn is_zero<T: Default + PartialEq>(n: &T) -> bool {
    *n == T::default()
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
    /// Task 3.10: the victim was out (frozen or dead) on every tick of the `after_ticks` window that follows the deciding
    /// tick (the window was played to its end); `false` for `D`/`T` games, which have no victim.
    pub held_block: bool,
    /// Task 3.10: the first tick after the deciding tick on which the victim was free again; `None` = it never was.
    pub escape_tick: Option<i32>,
    /// Task 3.10: the winner was out (frozen or dead) at some tick of the window (the `held` rule's condition; for a 1vN loss nobody was credited
    /// for: any opponent was out).
    pub winner_out_in_window: bool,
    /// Task 3.10: the focal player (slot 0) was out at some tick of the window (always true when it is the victim).
    pub focal_out_in_window: bool,
    /// Task 3.10: ticks of the `after_ticks` window the victim was out (a graded form of `held_block`, for rewards).
    pub victim_out_ticks: i32,
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
    /// Wayblock arenas: ticks the focal player spent inside the hall's band (`wbBand`), of `a_ticks`
    /// (the ticks played until the game was decided); both 0 elsewhere.
    #[serde(default)]
    pub a_band_ticks: u32,
    /// Ticks inside the held hall (`inWbHall`: the zone boxes grown by three tiles).
    #[serde(default)]
    pub a_hall_ticks: u32,
    #[serde(default)]
    pub a_ticks: u32,
    /// Opponent onsets *not* credited to the focal player before the game was decided (1vN only).
    pub bystander_outs: u32,
    /// Task 3.19: the tick the duel countdown ended (the round was played from here); 0 outside the duel rules.
    #[serde(skip_serializing_if = "is_zero")]
    pub fight_start: i32,
    pub players: Vec<PlayerReport>,
    pub timing: Timing,
    /// Raw decision times per player, for pooling into batch-level percentiles.
    #[serde(skip)]
    pub decide_us: Vec<Vec<u32>>,
}

/// What a game says about whether its block held: the facts a training reward is made of (task 3.10; the fly's training plays
/// episodes past the first freeze, `Rules::held_block_window`, and rewards what lasts).
///
/// **Which fields a reward takes.** `held_block` alone is *not* a win: it holds for any victim, an opponent that froze itself and a lost game included.
/// A held block *of the focal player* is `result == W && credited && held_block && !focal_out_in_window`
/// ([`HeldOutcome::strict_held_win`]); this is what task 8.5a's `EpisodeOutcome::held_block` computes too, so it can swap its `from_records` for
/// these facts (`credited`, `focal_out_in_window`, `held_block`, `victim_out_ticks`, `escape_tick`) without losing anything.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct HeldOutcome {
    pub result: GameResult,
    pub credited: bool,
    /// The victim was out on every tick of the window (any game with a victim; see above).
    pub held_block: bool,
    pub escape_tick: Option<i32>,
    pub victim_out_ticks: i32,
    pub winner_out_in_window: bool,
    pub focal_out_in_window: bool,
    /// The window the facts were played over (`Rules::after_ticks`).
    pub window_ticks: i32,
}

impl GameReport {
    pub fn held_outcome(&self, window_ticks: i32) -> HeldOutcome {
        HeldOutcome {
            result: self.result,
            credited: self.credited,
            held_block: self.held_block,
            escape_tick: self.escape_tick,
            victim_out_ticks: self.victim_out_ticks,
            winner_out_in_window: self.winner_out_in_window,
            focal_out_in_window: self.focal_out_in_window,
            window_ticks,
        }
    }
}

impl HeldOutcome {
    /// The focal player's own held block: a win by its credited block whose victim stayed out for the whole window while the focal player was never out.
    pub fn strict_held_win(&self) -> bool {
        self.result == GameResult::W && self.credited && self.held_block && !self.focal_out_in_window
    }

    /// The held-block return for the focal player, the same as task 8.5a's `RewardConfig` base reward: `+1` for a strict held win
    /// ([`HeldOutcome::strict_held_win`]); `-1` when the focal player was out at any time of the window (a lost game included: an own freeze is the worst
    /// outcome); `0` for everything else, a block that thawed and a freeze nobody credited the focal player for included (no graded credit: a freeze that
    /// lets go is not a block, and an opponent's self-freeze is not a skill).
    pub fn held_return(&self) -> f32 {
        if self.strict_held_win() {
            1.0
        } else if self.focal_out_in_window || self.result == GameResult::L {
            -1.0
        } else {
            0.0
        }
    }
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
    play_game_observed(arena, rules, seed, layout, players, &mut |_, _| true)
}

/// [`play_game`] with an observer called after every world tick with the players (so it can read a brain's
/// visualisation frame, `ddai_brain::Brain::viz_frame`) and the tick just played. It must not change what the
/// brains decide; it returns `false` to stop the game early (reported as a timeout `T`). The task 7.4 watch mode
/// paces the game to real time in here. With the no-op observer of [`play_game`] the game is exactly as before.
pub fn play_game_observed(
    arena: &Arena,
    rules: &Rules,
    seed: u64,
    layout: Layout,
    players: Vec<PlayerSetup>,
    observe: &mut dyn FnMut(&mut [PlayerSetup], i32) -> bool,
) -> Result<GameReport, EnvError> {
    play_game_watched(arena, rules, seed, layout, players, &mut |sim, tick| {
        observe(&mut sim.players, tick)
    })
}

/// [`play_game_observed`] whose observer gets the whole [`Sim`] (its physics world included, read-only by convention),
/// so a viewer can draw the game itself and not only a brain's frame (task 5.7: the web's demo shows the arena). The
/// same contract: the observer must not change the game, and returns `false` to stop it early.
pub fn play_game_watched(
    arena: &Arena,
    rules: &Rules,
    seed: u64,
    layout: Layout,
    players: Vec<PlayerSetup>,
    observe: &mut dyn FnMut(&mut Sim, i32) -> bool,
) -> Result<GameReport, EnvError> {
    play_game_modeled(arena, rules, seed, layout, players, Vec::new(), observe)
}

/// [`play_game_watched`] with input-lag models (task 3.16, [`crate::sim::LagModel`]) for the players that have one (`lag_models[slot]`; a short
/// vector or a `None` keeps that player's fixed lag). Without any model it is exactly [`play_game_watched`].
pub fn play_game_modeled(
    arena: &Arena,
    rules: &Rules,
    seed: u64,
    layout: Layout,
    players: Vec<PlayerSetup>,
    lag_models: Vec<Option<crate::sim::LagModel>>,
    observe: &mut dyn FnMut(&mut Sim, i32) -> bool,
) -> Result<GameReport, EnvError> {
    play_game_core(arena, rules, &[], lag_models, seed, layout, players, observe)
}

/// [`play_game_modeled`] with the duel options of a condition (task 3.19): `None` is exactly [`play_game_modeled`]; with a [`DuelSpec`] the
/// slots in `live_view` decide through the live view ([`crate::liveview`]), and with `rounds` the F-DDrace round rules replace the
/// first-freeze rule ([`crate::duel::play_duel_watched`]).
#[allow(clippy::too_many_arguments)]
pub fn play_game_duel_watched(
    arena: &Arena,
    rules: &Rules,
    duel: Option<&DuelSpec>,
    seed: u64,
    layout: Layout,
    players: Vec<PlayerSetup>,
    lag_models: Vec<Option<crate::sim::LagModel>>,
    observe: &mut dyn FnMut(&mut Sim, i32) -> bool,
) -> Result<GameReport, EnvError> {
    match duel {
        Some(d) if d.rounds => {
            crate::duel::play_duel_watched(arena, rules, d, seed, layout, players, lag_models, observe)
        }
        Some(d) => {
            d.validate(players.len())?;
            play_game_core(arena, rules, &d.live_view, lag_models, seed, layout, players, observe)
        }
        None => play_game_core(arena, rules, &[], lag_models, seed, layout, players, observe),
    }
}

#[allow(clippy::too_many_arguments)]
fn play_game_core(
    arena: &Arena,
    rules: &Rules,
    live_view: &[usize],
    lag_models: Vec<Option<crate::sim::LagModel>>,
    seed: u64,
    layout: Layout,
    players: Vec<PlayerSetup>,
    observe: &mut dyn FnMut(&mut Sim, i32) -> bool,
) -> Result<GameReport, EnvError> {
    let n = players.len();
    if n < 2 {
        return Err(EnvError::new("a game needs at least two players"));
    }
    if n > ddai_physics::core::MAX_CLIENTS {
        return Err(EnvError::new("too many players"));
    }
    let credit_required = rules.credit_required.unwrap_or(n > 2);
    let mut spawn = arena.spawn_tiles_crowd(seed, n, rules.crowd_from, rules.crowd_spacing)?;
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
    let hold = rules
        .hold_target
        .then(|| std::rc::Rc::new(HoldTarget::new(arena.map.clone())));
    let mut sim = Sim::new(pw, arena.map.clone(), players, rules.decide_every, seed);
    sim.set_lag_models(lag_models);
    for &slot in live_view {
        sim.enable_live_view(slot, seed);
    }
    let mut last_touch: Vec<Touch> = vec![None; n];
    let mut was_out = vec![false; n];

    let mut result: Option<GameResult> = None;
    let mut end_tick = -1;
    let mut victim: i32 = -1;
    let mut winner: Option<usize> = None;
    let mut credited = false;
    let mut winner_out_after = false;
    let mut focal_out_window = false;
    let mut escape_tick: Option<i32> = None;
    let mut victim_out_ticks = 0i32;
    let mut a_out_tick: Option<i32> = None;
    let mut a_self_freezes = 0u32;
    let mut blocks_by_a = 0u32;
    let mut first_block_tick: Option<i32> = None;
    let mut bystander_outs = 0u32;
    let (mut a_band_ticks, mut a_hall_ticks, mut a_ticks) = (0u32, 0u32, 0u32);

    let limit = rules.max_ticks + rules.after_ticks;
    for _ in 0..limit {
        let events = match &hold {
            Some(h) => {
                let h = std::rc::Rc::clone(h);
                sim.step(&move |w, slot, ids| h.pick(w, slot, ids))
            }
            None => sim.step(&default_target),
        };
        let now = sim.tick();
        if !observe(&mut sim, now) {
            break;
        }
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
        if result.is_none()
            && let Some(w) = &arena.wb
            && let Some(core) = sim.pw.inner().cores.get(sim.ids[0] as u8)
        {
            a_ticks += 1;
            let (x0, y0, x1, y1) = w.band;
            a_band_ticks += u32::from(core.pos.x >= x0 && core.pos.x <= x1 && core.pos.y >= y0 && core.pos.y <= y1);
            a_hall_ticks += u32::from(w.def.in_hall(
                w.side,
                (core.pos.x / 32.0).trunc() as i32,
                (core.pos.y / 32.0).trunc() as i32,
            ));
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
                if out_now[victim as usize] {
                    victim_out_ticks += 1;
                } else {
                    escape_tick.get_or_insert(now);
                }
                focal_out_window |= out_now[0];
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
    // The window must have been played to its end (an observer that stops the game early leaves it short).
    let held_block = victim >= 0 && escape_tick.is_none() && sim.tick() >= end_tick + rules.after_ticks;

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
            lag_hist: sim.lag_models[i].as_ref().map_or_else(Vec::new, |m| m.hist.to_vec()),
            lag_late: sim.lag_models[i].as_ref().map_or(0, |m| m.later),
            lag_cost_hist: sim.lag_models[i]
                .as_ref()
                .map_or_else(Vec::new, |m| m.cost_hist.clone()),
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
        held_block,
        escape_tick,
        winner_out_in_window: winner_out_after,
        focal_out_in_window: focal_out_window,
        victim_out_ticks,
        victim,
        spawns,
        a_out_tick,
        a_self_freezes,
        blocks_by_a,
        first_block_tick,
        a_band_ticks,
        a_hall_ticks,
        a_ticks,
        bystander_outs,
        fight_start: 0,
        players: reports,
        timing: Timing {
            decide_us_p50: p50,
            decide_us_p99: p99,
        },
        decide_us: sim.decide_us,
    })
}
