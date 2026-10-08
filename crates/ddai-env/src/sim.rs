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

/// Ticks of lag the model histogram counts (0..=MAX_LAG_BIN; more lands in the last bin).
pub const MAX_LAG_BIN: usize = 8;
/// The decision-cost histogram of a [`LagModel`]: bins of [`COST_BIN_MS`], the last one takes everything above.
pub const COST_BINS: usize = 129;
pub const COST_BIN_MS: f64 = 0.25;

/// Task 3.16 (D-115): an input lag that follows the decision's own cost, the arena's counterpart of the live bot's slot choice (D-063).
///
/// A live decision is aimed at an input slot, and slots are whole ticks apart. With `base_ms` the time from a snapshot's tick to the first slot
/// a decision that costs nothing could still make (the round trip, the margin and the phase of the slots in one number), a decision of `cost` ms
/// goes out for the tick `E = ceil((base + extra + cost) / 20 ms)` after the snapshot's tick, which is an arena lag of `E - 1` (see `ticks_for`):
/// the rounding up to a whole tick is the whole point (`docs/research/lag-shave.md`). The bot **plans** for the lag its rolling p90 of recent decision costs implies
/// (`planned`, the window the brain is told about) and the driver holds an early decision until its tick, so the lag a decision gets is
/// `max(planned, ceil((base + extra + cost) / 20) - 1)`: a decision slower than the estimate is applied late and the brain's world was one tick short (the live
/// `later_than_predicted`).
#[derive(Debug, Clone)]
pub struct LagModel {
    /// The lag (arena `lag`, ticks) of a decision that costs nothing is `ceil(base_ms / 20) - 1` (the input goes out for the tick `ceil(base_ms / 20)` after the snapshot's).
    pub base_ms: f64,
    /// Costs of the live path the arena does not charge (queue hop, the driver's pick-up, the fly proposer's time), added to every decision's cost.
    pub extra_ms: f64,
    /// Snapshot arrival jitter, ms: a deterministic uniform draw in `+-jitter_ms` moves `base_ms` per decision (the bot knows its own
    /// phase, so the draw enters the planned lag, too).
    pub jitter_ms: f64,
    /// The cost estimate before the first decision, ms (the live bot's 6 ms start).
    pub initial_ms: f64,
    /// A deadline-aware search (opt-in, the arena's trial of a bot that tells the brain how long the first slot allows): when the first slot is
    /// within reach with at least this many ms for the decision, the brain is told to finish in the room the slot leaves
    /// ([`ddai_brain::Brain::set_decision_deadline_ms`]) and the bot plans for that. `None`: no deadline, as always.
    pub deadline_floor_ms: Option<f64>,
    ring: Vec<f64>,
    next: usize,
    /// Decisions by the lag they got (bin `MAX_LAG_BIN` takes more).
    pub hist: [u32; MAX_LAG_BIN + 1],
    /// Decisions that cost more than the estimate allowed for: applied one tick or more after the tick the brain planned for.
    pub later: u32,
    /// Decisions made (the histogram's total).
    pub decisions: u32,
    /// The costs the decisions reported, in bins of [`COST_BIN_MS`] (the arena's counterpart of the bot's `brain` series).
    pub cost_hist: Vec<u32>,
}

/// The rolling window of decision costs behind the estimate (the bot's `ESTIMATE_WINDOW`) and its quantile (`DEFAULT_ESTIMATE_QUANTILE`).
const LAG_EST_WINDOW: usize = 64;
const LAG_EST_QUANTILE: f64 = 0.9;
/// One server tick, ms.
const TICK_MS: f64 = 20.0;

impl LagModel {
    pub fn new(base_ms: f64, extra_ms: f64, jitter_ms: f64, initial_ms: f64) -> LagModel {
        LagModel {
            base_ms,
            extra_ms,
            jitter_ms,
            initial_ms,
            deadline_floor_ms: None,
            ring: Vec::with_capacity(LAG_EST_WINDOW),
            next: 0,
            hist: [0; MAX_LAG_BIN + 1],
            later: 0,
            decisions: 0,
            cost_hist: vec![0; COST_BINS],
        }
    }

    /// The same model with its memory and counters cleared (a new game).
    pub fn fresh(&self) -> LagModel {
        LagModel {
            deadline_floor_ms: self.deadline_floor_ms,
            ..LagModel::new(self.base_ms, self.extra_ms, self.jitter_ms, self.initial_ms)
        }
    }

    /// With a deadline-aware search: the ms the first slot leaves a decision that starts now, if that is at least the floor.
    pub fn deadline_for(&self, phase: f64) -> Option<f64> {
        let floor = self.deadline_floor_ms?;
        let slack = TICK_MS * (phase / TICK_MS).ceil() - phase;
        let room = slack - self.extra_ms - 0.1;
        (room >= floor).then_some(room)
    }

    /// The rolling p90 of recent decision costs, or the initial estimate.
    fn estimate_ms(&self) -> f64 {
        if self.ring.is_empty() {
            return self.initial_ms;
        }
        let mut v = self.ring.clone();
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * LAG_EST_QUANTILE).ceil() as usize]
    }

    /// This decision's phase: `base_ms` plus the jitter draw for (`seed`, `slot`, `tick`).
    fn phase_ms(&self, seed: u64, slot: usize, tick: i32) -> f64 {
        if self.jitter_ms <= 0.0 {
            return self.base_ms;
        }
        // splitmix64 over the three: deterministic, independent of the thread and of the order games are played in.
        let mut z = seed ^ ((slot as u64) << 56) ^ (u64::from(tick as u32) << 8) ^ 0x9E37_79B9_7F4A_7C15;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = (z >> 11) as f64 / (1u64 << 53) as f64;
        self.base_ms + (2.0 * u - 1.0) * self.jitter_ms
    }

    /// The arena lag of a decision that needs `ms` from the snapshot's tick: the input goes out for the tick `E = ceil(ms / 20)` after the
    /// snapshot's, and the brain rolls the world `E - 1` ticks first -- the live bot's horizon `to_tick - tick`, and the arena's `lag` (lag 0 = the input acts
    /// in the very next step of the snapshot's state). With RTT 25 ms `E >= 2`, so the arena lag is at least 1: lag 0 is out of reach.
    fn ticks_for(ms: f64) -> u32 {
        ((ms / TICK_MS).ceil().max(1.0) as u32) - 1
    }

    /// Before the decision: the phase and the lag the brain is told to plan for.
    pub fn plan(&self, seed: u64, slot: usize, tick: i32) -> (f64, u32) {
        let phase = self.phase_ms(seed, slot, tick);
        let est = self
            .deadline_for(phase)
            .map_or_else(|| self.estimate_ms(), |room| self.estimate_ms().min(room));
        (phase, Self::ticks_for(phase + self.extra_ms + est))
    }

    /// After the decision, which cost `cost_ms` on the brain's clock: the lag it gets. Feeds the estimate and the counters.
    pub fn land(&mut self, phase: f64, planned: u32, cost_ms: f64) -> u32 {
        let wanted = Self::ticks_for(phase + self.extra_ms + cost_ms);
        let lag = planned.max(wanted);
        if self.ring.len() < LAG_EST_WINDOW {
            self.ring.push(cost_ms);
        } else {
            self.ring[self.next] = cost_ms;
            self.next = (self.next + 1) % LAG_EST_WINDOW;
        }
        self.hist[(lag as usize).min(MAX_LAG_BIN)] += 1;
        self.cost_hist[((cost_ms / COST_BIN_MS) as usize).min(COST_BINS - 1)] += 1;
        self.decisions += 1;
        self.later += u32::from(wanted > planned);
        lag
    }

    /// Mean lag in ticks over the decisions made.
    pub fn mean_lag(&self) -> f64 {
        if self.decisions == 0 {
            return 0.0;
        }
        self.hist
            .iter()
            .enumerate()
            .map(|(l, &n)| l as f64 * f64::from(n))
            .sum::<f64>()
            / f64::from(self.decisions)
    }
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
    /// Task 3.16: per player, the input-lag model that replaces the fixed `lag` (`None`: the fixed lag, as always), and the seed its jitter draws from.
    pub lag_models: Vec<Option<LagModel>>,
    seed: u64,
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
            lag_models: vec![None; n],
            seed,
        }
    }

    /// Task 3.16: gives the players input-lag models (a missing entry keeps the player's fixed lag). Call before the first step.
    pub fn set_lag_models(&mut self, mut models: Vec<Option<LagModel>>) {
        models.resize(self.players.len(), None);
        self.lag_models = models;
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
                // Task 3.16: with a lag model the lag of this decision is planned from the cost estimate and settled by its real cost below.
                let mut modelled = self.lag_models[i].as_ref().map(|m| m.plan(self.seed, i, tick));
                let mut applied_lag = modelled.map_or(self.players[i].lag, |(_, planned)| planned);
                let wire = if observe::is_alive(self.pw.inner(), id) {
                    let target_id = target(self.pw.inner(), i, &self.ids);
                    let action = match observe::observation(self.pw.inner(), &self.map, id, &self.ids, target_id) {
                        Some(obs) => {
                            let lag = applied_lag;
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
                            let deadline = self.lag_models[i]
                                .as_ref()
                                .zip(modelled)
                                .and_then(|(m, (phase, _))| m.deadline_for(phase));
                            self.players[i].brain.set_decision_deadline_ms(deadline);
                            let t0 = Instant::now();
                            let a = self.players[i].brain.decide_in(&obs, Some(&view));
                            self.decide_us[i].push(u32::try_from(t0.elapsed().as_micros()).unwrap_or(u32::MAX));
                            if let (Some(m), Some((phase, planned))) = (self.lag_models[i].as_mut(), modelled.take()) {
                                let cost_ms = self.players[i]
                                    .brain
                                    .last_plan()
                                    .map_or(0.0, |p| f64::from(p.decision_us) / 1000.0);
                                applied_lag = m.land(phase, planned, cost_ms);
                            }
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
                if applied_lag > 0 {
                    self.pending[i].push_back((tick + 1 + applied_lag as i32, wire));
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
