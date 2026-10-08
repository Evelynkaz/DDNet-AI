//! Task 3.19 (D-116): the F-DDrace `/1vs1` duel box as an arena mode -- the round rules of the minigame and the live view of the bot.
//!
//! Until now a duel game of the arena ended on the *first freeze or death* of either tee ([`crate::game`]). The minigame does not play that way
//! (`F-DDrace/src/game/server/minigames/arenas.cpp`, `CArenas::Tick`, `OnCharacterDie`, `OnCharacterSpawn`; checked against the clips of the 2026-10-07
//! test duel, `docs/research/duel-3.19.md` §2):
//!
//! * **Countdown.** Every round both tees are respawned on their spawn tiles frozen for three seconds (`Freeze(3)` = 150 ticks, no movement and no
//!   hammer for either of them); the round is played from the thaw.
//! * **Any death is a point** for the other tee. So is leaving the arena (not modelled: the box is closed).
//! * **A frozen tee on the ground loses.** A tee that is *in a freeze tile* (`m_IsFrozen`: set by `Freeze()` on every tick the tee touches one, cleared
//!   at the start of the next), on the ground (`IsGrounded`), and has not moved for more than a second (50 ticks) loses the round. The other tee gets the
//!   point unless it is itself in a freeze tile, or frozen and in the air: then it is a **draw** (no point), and the round is replayed. A freeze in the
//!   air is therefore *not* the end of a round: the frozen tee falls, may be hammered free (a hit unfreezes), thaws after 150 ticks outside a freeze tile.
//!   (The clips show it: the freeze at the ceiling of round 3 was 84 ticks before the next countdown, round 2 218, the loser lying in the pit.)
//! * **A point needs a touch.** `IncreaseScore` counts a round only if the loser's last hook/hammer contact was the winner's
//!   (`m_Killer`); otherwise the round ends without a point, like a draw.
//!
//! [`DuelSpec`] is the opt-in table of a condition (`[condition.duel]`): the round rules (`rounds`, `countdown_ticks`, `grounded_ticks`) and the
//! slots that decide through the live view (`live_view`, [`crate::liveview`]). Neither is on by default, so every existing config plays as before.

use serde::{Deserialize, Serialize};

use ddai_physics::world::World;
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::WorldEvent;
use ddai_planner::vmath::Vec2;

use crate::EnvError;
use crate::arena::{Arena, tile_center};
use crate::config::Rules;
use crate::game::{GameReport, Layout, PlayerReport, Timing};
use crate::observe;
use crate::sim::{PlayerSetup, Sim, default_target};
use crate::stats::{GameResult, percentile_u32};

fn d_true() -> bool {
    true
}
fn d_countdown() -> i32 {
    150
}
fn d_grounded() -> i32 {
    50
}

/// The `[condition.duel]` table. Every field has a default; an empty table switches the round rules on and the live view off.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DuelSpec {
    /// The F-DDrace round rules (default true). `false` = the old rule (the first freeze or death decides), the live view only.
    #[serde(default = "d_true")]
    pub rounds: bool,
    /// Ticks both tees are frozen on their spawn tiles before the round (3 s = 150).
    #[serde(default = "d_countdown")]
    pub countdown_ticks: i32,
    /// A tee in a freeze tile, on the ground and not moving for more than this many ticks loses (1 s = 50).
    #[serde(default = "d_grounded")]
    pub grounded_ticks: i32,
    /// Slots whose brain decides through the live view (`LiveWorld` from server-style snapshots, [`crate::liveview`]).
    #[serde(default)]
    pub live_view: Vec<usize>,
}

impl Default for DuelSpec {
    fn default() -> Self {
        DuelSpec {
            rounds: true,
            countdown_ticks: d_countdown(),
            grounded_ticks: d_grounded(),
            live_view: Vec::new(),
        }
    }
}

impl DuelSpec {
    pub fn validate(&self, players: usize) -> Result<(), EnvError> {
        if self.countdown_ticks < 0 || self.grounded_ticks < 1 {
            return Err(EnvError::new(
                "duel: countdown_ticks must be >= 0 and grounded_ticks >= 1",
            ));
        }
        if self.rounds && players != 2 {
            return Err(EnvError::new("duel: the round rules are for exactly two players"));
        }
        if let Some(s) = self.live_view.iter().find(|&&s| s >= players) {
            return Err(EnvError::new(format!("duel: live_view names slot {s} of {players}")));
        }
        Ok(())
    }
}

/// Why a round ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RoundWhy {
    /// A tee died (a kill tile).
    Death,
    /// A tee lay in a freeze tile on the ground, not moving, for more than `grounded_ticks`.
    Grounded,
}

/// The end of a round: who lost it and whether the other tee gets the point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundEnd {
    /// Slot of the loser.
    pub loser: usize,
    pub why: RoundWhy,
    /// `true` when the other tee is entitled to the point (a draw otherwise): the rule's own condition, before the touch check.
    pub winner_ok: bool,
}

/// The round rule of `CArenas::Tick`, tick by tick.
#[derive(Debug, Clone)]
pub struct RoundWatch {
    /// Per slot: the tick the current "in a freeze tile, on the ground, not moving" streak began (`m_aFirstGroundedFreezeTick`).
    timer: [Option<i32>; 2],
    /// Per slot: the position at the end of the previous tick (`m_PrevPos` stands for it).
    last_pos: [Option<(f32, f32)>; 2],
    grounded_ticks: i32,
}

impl RoundWatch {
    pub fn new(grounded_ticks: i32) -> RoundWatch {
        RoundWatch {
            timer: [None; 2],
            last_pos: [None; 2],
            grounded_ticks,
        }
    }

    /// Forgets the streaks (the countdown: nobody can move, so nothing may start counting).
    pub fn reset(&mut self) {
        self.timer = [None; 2];
        self.last_pos = [None; 2];
    }

    /// Whether the tee `id` is in a freeze tile this tick (`m_IsFrozen`).
    fn touches_freeze(world: &World<f32>, id: i32) -> bool {
        world
            .cores
            .get(id as u8)
            .is_some_and(|c| c.is_in_freeze || c.deep_frozen)
    }

    fn grounded(world: &World<f32>, id: i32) -> bool {
        observe::character_observation(world, id).is_some_and(|c| c.grounded)
    }

    /// One tick, evaluated on the world after the step (`now` = its tick). `ids` are the two tees' client ids by slot.
    pub fn step(&mut self, world: &World<f32>, ids: [i32; 2], now: i32) -> Option<RoundEnd> {
        for i in 0..2 {
            let j = 1 - i;
            if !observe::is_alive(world, ids[i]) {
                // OnCharacterDie: the survivor scores (if it touched the loser; the caller checks that).
                return Some(RoundEnd {
                    loser: i,
                    why: RoundWhy::Death,
                    winner_ok: observe::is_alive(world, ids[j]),
                });
            }
            let pos = world.cores.get(ids[i] as u8).map(|c| (c.pos.x, c.pos.y));
            if self.last_pos[i] != pos {
                self.timer[i] = None;
            }
            self.last_pos[i] = pos;
            match self.timer[i] {
                None => {
                    if Self::touches_freeze(world, ids[i]) && Self::grounded(world, ids[i]) {
                        self.timer[i] = Some(now);
                    }
                }
                Some(first) if first < now - self.grounded_ticks => {
                    let other_free_of_tile = !Self::touches_freeze(world, ids[j]);
                    let other_thawed_or_grounded = world.characters[ids[j] as usize]
                        .as_ref()
                        .is_some_and(|c| c.freeze_time == 0)
                        || Self::grounded(world, ids[j]);
                    return Some(RoundEnd {
                        loser: i,
                        why: RoundWhy::Grounded,
                        winner_ok: observe::is_alive(world, ids[j]) && other_free_of_tile && other_thawed_or_grounded,
                    });
                }
                Some(_) => {}
            }
        }
        None
    }
}

/// Freezes tee `id` for `seconds` the way `CCharacter::Freeze` does on a spawn.
fn spawn_freeze(pw: &mut PhysicsWorld, id: i32, seconds: i32) {
    let w = pw.inner_mut();
    let tick = w.tick;
    if let Some(slot) = w.cores.slot_of(id as u8)
        && let Some(ch) = w.characters[id as usize].as_mut()
    {
        let core = w.cores.core_at_mut(slot);
        ddai_physics::world::freeze(ch, core, tick, seconds);
    }
}

/// Plays one duel game under the round rules: a countdown, then the round until the rule ends it (or `rules.max_ticks` ticks of play, a timeout).
/// `players[0]` is the focal player; there must be exactly two. Same seeds, layouts and brains as [`crate::game::play_game_watched`], whose
/// observer contract it keeps (the observer may stop the game: a timeout).
#[allow(clippy::too_many_arguments)]
pub fn play_duel_watched(
    arena: &Arena,
    rules: &Rules,
    spec: &DuelSpec,
    seed: u64,
    layout: Layout,
    players: Vec<PlayerSetup>,
    lag_models: Vec<Option<crate::sim::LagModel>>,
    observe_fn: &mut dyn FnMut(&mut Sim, i32) -> bool,
) -> Result<GameReport, EnvError> {
    if players.len() != 2 {
        return Err(EnvError::new("a duel game needs exactly two players"));
    }
    spec.validate(2)?;
    let mut spawn = arena.spawn_tiles_crowd(seed, 2, rules.crowd_from, rules.crowd_spacing)?;
    if layout.swap {
        spawn.swap(0, 1);
    }
    let mut pw = PhysicsWorld::from_world(arena.new_world(), arena.map.clone());
    let mut spawns = vec![[0.0f32; 2]; 2];
    let order: [usize; 2] = if layout.reverse_order { [1, 0] } else { [0, 1] };
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
    let countdown = spec.countdown_ticks;
    if countdown > 0 {
        // `Freeze(3)`: the length of the countdown is the freeze, in whole seconds of ticks.
        for id in 0..2 {
            spawn_freeze(&mut pw, id, (countdown + 49) / 50);
        }
    }
    let mut sim = Sim::new(pw, arena.map.clone(), players, rules.decide_every, seed);
    sim.set_lag_models(lag_models);
    for &s in &spec.live_view {
        sim.enable_live_view(s, seed);
    }
    let ids = [sim.ids[0], sim.ids[1]];
    let mut watch = RoundWatch::new(spec.grounded_ticks);
    // Who touched whom last: `[victim] = tick` of the winner's last hook/hammer on it (slot of the toucher is the other slot in a duel).
    let mut touched: [Option<i32>; 2] = [None; 2];
    let mut was_out = [false; 2];
    let mut a_self_freezes = 0u32;
    let mut a_out_tick: Option<i32> = None;
    let mut blocks_by_a = 0u32;
    let mut first_block_tick: Option<i32> = None;
    let mut end: Option<(RoundEnd, i32)> = None;
    let limit = countdown + rules.max_ticks;
    for _ in 0..limit {
        let events = sim.step(&default_target);
        let now = sim.tick();
        if !observe_fn(&mut sim, now) {
            break;
        }
        for e in &events {
            if let WorldEvent::HammerHit { from, to } = *e
                && (0..2).contains(&from)
                && (0..2).contains(&to)
            {
                touched[to as usize] = Some(now);
            }
        }
        for id in ids {
            let h = observe::hooked_player(sim.pw.inner(), id);
            if (0..2).contains(&h) {
                touched[h as usize] = Some(now);
            }
        }
        let out_now = [
            observe::is_out(sim.pw.inner(), ids[0]),
            observe::is_out(sim.pw.inner(), ids[1]),
        ];
        if now <= countdown {
            watch.reset();
            was_out = out_now;
            continue;
        }
        if out_now[0] && !was_out[0] {
            a_self_freezes += 1;
            a_out_tick.get_or_insert(now);
        }
        if out_now[1] && !was_out[1] && touched[1].is_some_and(|t| now - t <= rules.credit_ticks) {
            blocks_by_a += 1;
            first_block_tick.get_or_insert(now);
        }
        was_out = out_now;
        if let Some(r) = watch.step(sim.pw.inner(), ids, now) {
            end = Some((r, now));
            break;
        }
    }
    let (result, end_tick, credited, victim) = match end {
        Some((r, now)) => {
            // `IncreaseScore`: no touch, no point.
            if r.winner_ok && touched[r.loser].is_some() {
                (
                    if r.loser == 1 { GameResult::W } else { GameResult::L },
                    now,
                    true,
                    r.loser as i32,
                )
            } else {
                (GameResult::D, now, false, -1)
            }
        }
        None => (GameResult::T, sim.tick(), false, -1),
    };
    let mut reports = Vec::with_capacity(2);
    let mut p50 = Vec::with_capacity(2);
    let mut p99 = Vec::with_capacity(2);
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
        held: false,
        held_block: false,
        escape_tick: None,
        winner_out_in_window: false,
        focal_out_in_window: false,
        victim_out_ticks: 0,
        victim,
        spawns,
        a_out_tick,
        a_self_freezes,
        blocks_by_a,
        first_block_tick,
        a_band_ticks: 0,
        a_hall_ticks: 0,
        a_ticks: 0,
        bystander_outs: 0,
        fight_start: countdown,
        players: reports,
        timing: Timing {
            decide_us_p50: p50,
            decide_us_p99: p99,
        },
        decide_us: sim.decide_us,
    })
}
