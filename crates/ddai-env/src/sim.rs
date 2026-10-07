//! The per-tick machinery shared by games and technique scenarios: brains deciding on a cadence,
//! per-client input lag, fire-press counters, decision hashes and timings, one physics step.
//! Rules (who won) and success predicates live in the callers.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use ddai_brain::{Action, Brain, ResetContext, WorldView};
use ddai_physics::core::PlayerInput as Wire;
use ddai_physics::map::MapData;
use ddai_planner::physics_adapter::{PhysicsWorld, from_ddnet_input};
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::WorldEvent;
use sha1::{Digest, Sha1};

use crate::arena::hex;
use crate::observe;

/// One player: a brain plus its client's input lag.
pub struct PlayerSetup {
    pub brain: Box<dyn Brain>,
    pub lag: u32,
    pub label: String,
}

/// Picks the target a player's [`ddai_brain::Observation::target_id`] names.
pub type TargetFn = dyn Fn(&ddai_physics::world::World<f32>, usize, &[i32]) -> Option<i32>;

/// The default target rule: the focal player (slot 0) targets the nearest *free* (alive, not
/// frozen) other player, else the nearest alive one; everybody else targets the focal player.
pub fn default_target(world: &ddai_physics::world::World<f32>, slot: usize, ids: &[i32]) -> Option<i32> {
    let me = ids[slot];
    if slot != 0 {
        return observe::is_alive(world, ids[0]).then_some(ids[0]);
    }
    let my_pos = world.cores.get(me as u8)?.pos;
    let mut best: Option<(bool, f32, i32)> = None;
    for &id in ids.iter().filter(|&&id| id != me) {
        if !observe::is_alive(world, id) {
            continue;
        }
        let Some(core) = world.cores.get(id as u8) else {
            continue;
        };
        let d = (core.pos.x - my_pos.x).powi(2) + (core.pos.y - my_pos.y).powi(2);
        let frozen = observe::is_out(world, id);
        // Free targets first, then by distance, then by id (a deterministic tie-break).
        if best.is_none_or(|(bf, bd, bid)| (frozen, d, id) < (bf, bd, bid)) {
            best = Some((frozen, d, id));
        }
    }
    best.map(|(_, _, id)| id)
}

/// The target rule of the finishing switch (task 3.10, `Rules::hold_target`): [`default_target`], except that the focal player's frozen
/// current target stays the target (for up to [`HOLD_TARGET_TICKS`] after it was first seen frozen) until the exact passive forecast
/// ([`ddai_planner::forecast::passive_forecast`]) says it stays out for the held-block window. The arena's counterpart of the live bot's
/// `--finish target` target logic (`ddai_bot::target`): in a crowd the default rule leaves the victim of a first freeze at once for a free
/// opponent, and a victim that lies frozen on open ground thaws in 3 s.
pub struct HoldTarget {
    /// The forecast's scratch world, synced from the game's world at each look.
    pw: std::cell::RefCell<PhysicsWorld>,
    state: std::cell::RefCell<HoldState>,
}

struct HoldState {
    current: i32,
    frozen_since: i32,
    checked_at: i32,
    held: bool,
}

/// How long after it first froze a victim is kept at most (the bot's `FINISH_MAX_HOLD_TICKS`).
pub const HOLD_TARGET_TICKS: i32 = 600;
/// "No forecast yet" (far enough back that the first look always runs, without overflowing the subtraction).
const NEVER_CHECKED: i32 = -1_000_000;
/// A forecast answer is reused for this many ticks (the bot's `SEAL_ANSWER_TICKS`).
const HOLD_ANSWER_TICKS: i32 = 6;

impl HoldTarget {
    pub fn new(map: Arc<MapData>) -> HoldTarget {
        HoldTarget {
            pw: std::cell::RefCell::new(PhysicsWorld::new(map, 1)),
            state: std::cell::RefCell::new(HoldState {
                current: -1,
                frozen_since: -1,
                checked_at: NEVER_CHECKED,
                held: false,
            }),
        }
    }

    /// A [`TargetFn`] body.
    pub fn pick(&self, world: &ddai_physics::world::World<f32>, slot: usize, ids: &[i32]) -> Option<i32> {
        if slot != 0 {
            return default_target(world, slot, ids);
        }
        let mut st = self.state.borrow_mut();
        let cur = st.current;
        if cur >= 0 && ids.contains(&cur) && observe::is_alive(world, cur) && observe::is_out(world, cur) {
            if st.frozen_since < 0 {
                st.frozen_since = world.tick;
                st.checked_at = NEVER_CHECKED;
            }
            if world.tick - st.frozen_since <= HOLD_TARGET_TICKS {
                if world.tick - st.checked_at >= HOLD_ANSWER_TICKS {
                    let mut pw = self.pw.borrow_mut();
                    pw.sync_from(world);
                    st.held = ddai_planner::forecast::passive_forecast(
                        &mut *pw,
                        cur,
                        ddai_planner::forecast::HELD_HORIZON_TICKS,
                    )
                    .held();
                    st.checked_at = world.tick;
                }
                if !st.held {
                    return Some(cur);
                }
            }
        } else {
            st.frozen_since = -1;
        }
        let t = default_target(world, slot, ids);
        if t != Some(cur) {
            st.frozen_since = -1;
        }
        st.current = t.unwrap_or(-1);
        t
    }
}

/// Action -> wire input; the fire counter continues from `prev_fire` (`true` = a fresh press each
/// decision, `false` = released: the `decodeAction`/`scriptedAction` convention).
pub fn wire_from_action(a: &Action, prev_fire: i32) -> Wire {
    let mut w = a.to_player_input();
    let held = (prev_fire & 1) != 0;
    w.fire = match (a.fire, held) {
        (true, true) => prev_fire + 2,
        (true, false) => prev_fire + 1,
        (false, true) => prev_fire + 1,
        (false, false) => prev_fire,
    };
    w
}

fn neutral_wire_with_fire(fire: i32) -> Wire {
    Wire { fire, ..neutral_wire() }
}

fn neutral_wire() -> Wire {
    Action::neutral().to_player_input()
}

/// Inputs already sent but not yet applied for the next `lag` world steps (see
/// [`WorldView`]'s timing contract).
fn in_flight_inputs(current: Wire, pending: &VecDeque<(i32, Wire)>, tick: i32, lag: u32) -> Vec<Wire> {
    let mut out = Vec::with_capacity(lag as usize);
    let mut cur = current;
    let mut it = pending.iter().peekable();
    for k in 0..lag as i32 {
        while let Some(&&(apply, input)) = it.peek() {
            if apply <= tick + k + 1 {
                cur = input;
                it.next();
            } else {
                break;
            }
        }
        out.push(cur);
    }
    out
}

/// N players in one physics world. Client id = slot.
pub struct Sim {
    pub pw: PhysicsWorld,
    pub players: Vec<PlayerSetup>,
    pub ids: Vec<i32>,
    map: Arc<MapData>,
    decide_every: i32,
    current: Vec<Wire>,
    /// The last wire input each player sent (repeated while its tee is dead, like the harness's
    /// `prevInput`).
    last_sent: Vec<Wire>,
    pending: Vec<VecDeque<(i32, Wire)>>,
    hashes: Vec<Sha1>,
    /// Wall microseconds per decision, by slot.
    pub decide_us: Vec<Vec<u32>>,
    /// Task 3.14: keep the events of the last step in [`Sim::last_events`] (diagnostics: an observer reads them). Off by default.
    pub record_events: bool,
    /// The events of the last [`Sim::step`], when `record_events` is on.
    pub last_events: Vec<WorldEvent>,
}

impl Sim {
    /// Resets every brain (`seed + 100000 * slot`, the harness's `seed` / `seed + 100000`).
    pub fn new(
        pw: PhysicsWorld,
        map: Arc<MapData>,
        mut players: Vec<PlayerSetup>,
        decide_every: i32,
        seed: u64,
    ) -> Sim {
        let n = players.len();
        for (i, p) in players.iter_mut().enumerate() {
            p.brain.reset(&ResetContext {
                map: map.clone(),
                self_id: i as i32,
                seed: seed.wrapping_add(100_000 * i as u64),
            });
        }
        Sim {
            pw,
            players,
            ids: (0..n as i32).collect(),
            map,
            decide_every,
            current: vec![neutral_wire(); n],
            last_sent: vec![neutral_wire_with_fire(0); n],
            pending: vec![VecDeque::new(); n],
            hashes: (0..n).map(|_| Sha1::new()).collect(),
            decide_us: vec![Vec::new(); n],
            record_events: false,
            last_events: Vec::new(),
        }
    }

    pub fn tick(&self) -> i32 {
        self.pw.inner().tick
    }

    /// Runs one world tick: decisions (on the cadence), input delivery, one physics step. Returns
    /// the events of the step.
    pub fn step(&mut self, target: &TargetFn) -> Vec<WorldEvent> {
        let n = self.players.len();
        let tick = self.tick();
        if tick % self.decide_every == 0 {
            for i in 0..n {
                let id = self.ids[i];
                let wire = if observe::is_alive(self.pw.inner(), id) {
                    let target_id = target(self.pw.inner(), i, &self.ids);
                    let action = match observe::observation(self.pw.inner(), &self.map, id, &self.ids, target_id) {
                        Some(obs) => {
                            let lag = self.players[i].lag;
                            let in_flight = if lag > 0 {
                                in_flight_inputs(self.current[i], &self.pending[i], tick, lag)
                            } else {
                                Vec::new()
                            };
                            let view = WorldView {
                                world: self.pw.inner(),
                                self_id: id,
                                lag_ticks: lag,
                                in_flight: &in_flight,
                            };
                            let t0 = Instant::now();
                            let a = self.players[i].brain.decide_in(&obs, Some(&view));
                            self.decide_us[i].push(u32::try_from(t0.elapsed().as_micros()).unwrap_or(u32::MAX));
                            a
                        }
                        None => Action::neutral(),
                    };
                    wire_from_action(&action, self.last_sent[i].fire)
                } else {
                    // A dead tee is not asked: its client keeps sending the previous input.
                    self.last_sent[i]
                };
                self.last_sent[i] = wire;
                self.hashes[i].update(
                    format!(
                        "{tick}:{},{},{},{},{},{},{};",
                        wire.direction,
                        wire.jump,
                        wire.hook,
                        wire.fire,
                        wire.target_x,
                        wire.target_y,
                        wire.wanted_weapon
                    )
                    .as_bytes(),
                );
                if self.players[i].lag > 0 {
                    self.pending[i].push_back((tick + 1 + self.players[i].lag as i32, wire));
                } else {
                    self.current[i] = wire;
                }
            }
        }
        for i in 0..n {
            while let Some(&(apply, input)) = self.pending[i].front() {
                if apply <= tick + 1 {
                    self.current[i] = input;
                    self.pending[i].pop_front();
                } else {
                    break;
                }
            }
            self.pw.set_input(self.ids[i], from_ddnet_input(&self.current[i]));
        }
        let events = self.pw.step();
        if self.record_events {
            self.last_events.clone_from(&events);
        }
        events
    }

    /// First 16 hex digits of the SHA-1 of slot `i`'s decision stream (`orig-run.md` §9).
    pub fn decision_hash(&self, i: usize) -> String {
        hex(&self.hashes[i].clone().finalize())[..16].to_string()
    }
}
