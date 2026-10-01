//! [`Bot`]: the live bot's state machine, sans-IO. It is fed the client's events and one
//! [`LiveWorldSnapshot`] per decision, and returns an [`Output`] — the `PlayerInput` to send and the
//! protocol actions (`Cl_Kill`, `Cl_SetTeam`). [`crate::runner`] is the thin real-time shell around it.
//!
//! # One decision per snapshot (`onSnapshot`, `docs/research/orig-bot.md` §4.2)
//!
//! Several snapshots arriving together are **collapsed** by the runner: only the newest is decided
//! (`queueSnapshot`'s `setImmediate`, §4.1); [`Bot::note_collapsed`] counts the skipped ones. The
//! pipeline for that snapshot, in the TS order minus the DROP items (chat, auto-chat, LLM, dummy,
//! partner, rescue, emotes) and the 4.2 hooks:
//!
//! 1. collision/map ready? own id known? (else nothing to do)
//! 2. a game tick lower than the last one -> `onTickReset` (map restart)
//! 3. **LiveWorld update** from the snapshot with our own in-flight inputs ([`crate::sent`]), the
//!    snapshot's projectiles included ([`ddai_world::SnapshotInput`]);
//! 4. player table, tee view, **activity clock / attribution** ([`crate::activity`]);
//! 5. own tee dead/absent -> idle input, maybe ask to join from the spectators (`maybeJoinGame`);
//! 6. first frame of a life: brain reset, wander timers, navigator notified;
//! 7. **unstick** ([`crate::unstick`]) — may request `Cl_Kill`;
//! 8. mode: `hold` idles; the navigator hook may take over; else **target selection**
//!    ([`crate::target`]) in `fight`, none in `passive`;
//! 9. no target -> **wander** (guarded); a target -> **prediction** to PredTick with our in-flight
//!    inputs over only the local tees, the **brain** (`decide_in`), then the **post-filters** (the
//!    guard for brains without their own shield, the hook veto on spared tees);
//! 10. **input encoding** ([`crate::input`]; fire as a counter, hammer by default) and out.
//!
//! # Prediction target
//!
//! A snapshot of tick `T` arrives; the driver's last `NETMSG_INPUT` was for `P = pred_tick`; the next
//! one (the one carrying this decision) goes out for `P + 1`, which the server applies in its step
//! to tick `P + 1`. So the world is predicted to `P` with the inputs already sent for `(T, P]` and the
//! brain's action takes effect on the very next step: `WorldView { lag_ticks: 0, in_flight: [] }` —
//! the in-flight inputs are already inside the predicted world, which is exact, unlike the TS's
//! `lag` guess.
//!
//! # The brain sees only the tees that matter
//!
//! Target + roped tees + the nearest few within the threat radius ([`crate::consts::THREAT_RADIUS_PX`],
//! [`crate::consts::MAX_LOCAL_OTHERS`]); the rest are cut from the predicted world. A search over
//! 8+ tees collapses (3.5's review F2).
//!
//! # Allocation
//!
//! Everything above is allocation-free per snapshot once warm, except the planner's own helpers
//! (seal check for frozen candidates, guard, veto) and the brain itself — `tests` measure it.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ddai_brain::{Action, Brain, IVec2, LiveContext, Observation, ResetContext, WorldView};
use ddai_client::LiveWorldSnapshot;
use ddai_net::generated::objects::PlayerInput as NetInput;
use ddai_physics::core::{MAX_CLIENTS, PlayerInput as PhysInput};
use ddai_physics::map::MapData;
use ddai_physics::vmath::Vec2;
use ddai_planner::physics_adapter::from_ddnet_input;
use ddai_planner::types::PlayerInput as PlannerInput;
use ddai_world::{LiveWorld, SnapshotInput, player_input_from_net};

use crate::activity::{ActivityClock, BlockEvent, BlockStats};
use crate::brains::BrainKind;
use crate::consts::*;
use crate::hooks::{HookContext, Hooks, NavStep};
use crate::input::InputEncoder;
use crate::latency::LatencyStats;
use crate::mapgrid::MapGrid;
use crate::planning::PlanScratch;
use crate::players::{PlayerTable, Salt, Tag};
use crate::relations::Relations;
use crate::sent::SentLog;
use crate::target::{PickCtx, TargetPicker, is_spared};
use crate::tees::{HOOK_IDLE, Tee, TeeSet, dist};
use crate::unstick::{KillReason, Unstick, UnstickCtx, Verdict};
use crate::wander::{Wander, WanderCtx, WanderEnv};

/// What the bot does (`mode`, §8.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Pick a target and fight (the default).
    Fight,
    /// Never pick a target: wander only.
    Passive,
    /// Do nothing (neutral input; the unstick rules still ignore us).
    Hold,
    /// A goto is in progress. Stub until task 4.2: the no-op navigator never drives, so this behaves
    /// like `Fight`.
    Goto,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Fight => "fight",
            Mode::Passive => "passive",
            Mode::Hold => "hold",
            Mode::Goto => "goto",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        [Mode::Fight, Mode::Passive, Mode::Hold, Mode::Goto]
            .into_iter()
            .find(|m| m.name() == s)
    }
}

/// Bot settings.
#[derive(Debug, Clone)]
pub struct BotConfig {
    pub brain: BrainKind,
    pub mode: Mode,
    /// `--target <name>`: fight only this player (exact folded name).
    pub fixed_target: Option<String>,
    /// Seeds the wander RNG and the brain resets.
    pub seed: u64,
    /// Per-process salt of the log hashes ([`crate::players`]).
    pub salt: Salt,
    /// Non-target tees handed to the brain at most (see the module docs).
    pub max_local_others: usize,
    pub threat_radius_px: f32,
    /// Ticks the prediction may reach past the snapshot at most.
    pub max_predict_ticks: i32,
    /// Give up asking to join the game after this many tries without becoming a player.
    pub max_join_attempts: u32,
    /// Run the seal searches on a worker thread (the live runner does; the sans-IO scenario tests
    /// keep the deterministic synchronous search). See `seal_worker`.
    pub async_seal: bool,
}

impl Default for BotConfig {
    fn default() -> Self {
        BotConfig {
            brain: BrainKind::Planner,
            mode: Mode::Fight,
            fixed_target: None,
            seed: 1,
            salt: crate::players::random_salt(),
            max_local_others: MAX_LOCAL_OTHERS,
            threat_radius_px: THREAT_RADIUS_PX,
            max_predict_ticks: MAX_PREDICT_TICKS,
            max_join_attempts: 10,
            async_seal: false,
        }
    }
}

/// What one snapshot's decision asks the shell to do.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Output {
    /// The input every `NETMSG_INPUT` embeds from now on (`Client::set_input`); `None`: leave the
    /// previous one (no decision was possible yet).
    pub input: Option<NetInput>,
    /// Send `Cl_Kill` (`Client::kill`), already cooldown-checked.
    pub kill: bool,
    /// Send `Cl_SetTeam(team)` (`Client::set_team`): `Some(0)` = join the game.
    pub set_team: Option<i32>,
    /// What this decision was aimed at, for the slot statistics (`Client::set_input_for_snapshot`).
    pub tag: Option<ddai_client::InputTag>,
}

/// Noteworthy things for the runner's log (no nicknames: [`Tag`]s only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BotEvent {
    Killed {
        tick: i32,
        reason: KillReason,
    },
    Block {
        tick: i32,
        victim: String,
    },
    BlockedBy {
        tick: i32,
        by: String,
    },
    TargetChanged {
        tick: i32,
        to: Option<String>,
    },
    Respawned {
        tick: i32,
    },
    Joining {
        tick: i32,
    },
    JoinGaveUp {
        tick: i32,
    },
    TickReset {
        from: i32,
        to: i32,
    },
    /// We were moved to the spectators after having played: the bot stops (exit code 3).
    MovedToSpectators {
        tick: i32,
    },
    RosterChanged {
        players: usize,
    },
}

/// Counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BotStats {
    pub snapshots: u64,
    /// Snapshots skipped because a newer one was already waiting ([`Bot::note_collapsed`]).
    pub collapsed: u64,
    pub decisions: u64,
    pub brain_decisions: u64,
    pub wander_decisions: u64,
    pub idle_decisions: u64,
    pub hooks_fired: u64,
    pub hammer_fires: u64,
    pub self_kills: u64,
    pub vetoed_hooks: u64,
    /// Fire presses withheld because the hammer would have hit a spared tee.
    pub vetoed_fires: u64,
    pub guarded_inputs: u64,
    pub deaths: u64,
    pub ticks_resets: u64,
}

/// The state of the bot for telemetry / the web bridge.
#[derive(Debug, Clone, PartialEq)]
pub struct Status {
    pub tick: i32,
    pub own_id: i32,
    pub target_id: i32,
    pub mode: Mode,
    pub alive: bool,
    pub frozen: bool,
    pub blocks: BlockStats,
    pub stats: BotStats,
}

/// The "asks to join" state. `attempts` is a **per-run** cap and never resets, not even when a join
/// succeeds (review round 1, F3): a server that moves us back to the spectators again and again is
/// telling us something.
#[derive(Debug, Clone, Copy, Default)]
struct JoinState {
    last_try_tick: Option<i32>,
    attempts: u32,
}

/// Why the bot asks the runner to stop (the run ends with the kick/ban exit code, 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// We had been playing and are now a spectator again: a moderator, a vote or the server moved us.
    /// That is a moderation signal (D-016): the bot stays a spectator and stops, it does not rejoin.
    MovedToSpectators,
}

impl StopReason {
    pub fn describe(self) -> &'static str {
        match self {
            StopReason::MovedToSpectators => {
                "moved to the spectators after having played: a moderation signal, not rejoining (D-016)"
            }
        }
    }
}

/// The bot.
pub struct Bot {
    cfg: BotConfig,
    relations: Relations,
    brain: Box<dyn Brain>,
    hooks: Hooks,
    players: PlayerTable,
    tees: TeeSet,
    clock: ActivityClock,
    picker: TargetPicker,
    unstick: Unstick,
    wander: Wander,
    encoder: InputEncoder,
    sent: SentLog,

    map: Option<Arc<MapData>>,
    grid: Option<MapGrid>,
    live: Option<LiveWorld>,
    plan: Option<PlanScratch>,
    obs: Option<Observation>,

    mode: Mode,
    last_tick: i32,
    was_alive: bool,
    lives: u64,
    join: JoinState,
    /// Whether our tee has existed on the current map (a later spectator state is then a move).
    played_on_map: bool,
    stop: Option<StopReason>,
    last_aim: (i32, i32),
    last_sent: PhysInput,
    /// Snapshot arrival -> this decision started (the channel hop), set by `on_snapshot`.
    queue_delay: Duration,
    /// Smoothed duration of a decision (bot thread, brain included): decides whether this snapshot's
    /// decision makes the driver's next input or the one after (see `prediction_target`).
    est_decision: Duration,

    // scratch (capacities fixed up front)
    in_flight: Vec<(i32, PhysInput)>,
    keep: Box<[bool; MAX_CLIENTS]>,
    spares: Vec<(Vec2<f32>, Vec2<f32>)>,
    spare_tees: Vec<Tee>,

    stats: BotStats,
    latency: LatencyStats,
    events: Vec<BotEvent>,
    status: Status,
}

const EVENT_CAP: usize = 64;

impl Bot {
    pub fn new(cfg: BotConfig, brain: Box<dyn Brain>, hooks: Hooks, relations: Relations) -> Bot {
        let mode = cfg.mode;
        Bot {
            players: PlayerTable::new(cfg.salt),
            tees: TeeSet::new(),
            clock: ActivityClock::new(),
            picker: TargetPicker::new(cfg.fixed_target.as_deref()),
            unstick: Unstick::new(),
            wander: Wander::new(cfg.seed),
            encoder: InputEncoder::new(),
            sent: SentLog::new(),
            map: None,
            grid: None,
            live: None,
            plan: None,
            obs: None,
            mode,
            last_tick: -1,
            was_alive: false,
            lives: 0,
            join: JoinState::default(),
            played_on_map: false,
            stop: None,
            last_aim: (0, -1),
            last_sent: PhysInput::default(),
            queue_delay: Duration::ZERO,
            est_decision: Duration::from_millis(1),
            in_flight: Vec::with_capacity(64),
            keep: Box::new([false; MAX_CLIENTS]),
            spares: Vec::with_capacity(MAX_CLIENTS),
            spare_tees: Vec::with_capacity(MAX_CLIENTS),
            stats: BotStats::default(),
            latency: LatencyStats::default(),
            events: Vec::with_capacity(EVENT_CAP),
            status: Status {
                tick: 0,
                own_id: -1,
                target_id: -1,
                mode,
                alive: false,
                frozen: false,
                blocks: BlockStats::default(),
                stats: BotStats::default(),
            },
            relations,
            brain,
            hooks,
            cfg,
        }
    }

    // ---- accessors -----------------------------------------------------------------------------

    pub fn config(&self) -> &BotConfig {
        &self.cfg
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// `!mode` / `!stop` / `!go` (task 4.3 will call this).
    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        if mode != Mode::Fight && mode != Mode::Goto {
            self.picker.set_target(-1);
        }
    }

    pub fn stats(&self) -> BotStats {
        self.stats
    }

    pub fn latency(&self) -> &LatencyStats {
        &self.latency
    }

    pub fn latency_mut(&mut self) -> &mut LatencyStats {
        &mut self.latency
    }

    pub fn players(&self) -> &PlayerTable {
        &self.players
    }

    pub fn relations(&self) -> &Relations {
        &self.relations
    }

    pub fn relations_mut(&mut self) -> &mut Relations {
        &mut self.relations
    }

    pub fn tees(&self) -> &TeeSet {
        &self.tees
    }

    pub fn map_name_known(&self) -> bool {
        self.map.is_some()
    }

    pub fn block_stats(&self) -> BlockStats {
        self.clock.stats()
    }

    pub fn target_id(&self) -> i32 {
        self.picker.target()
    }

    pub fn brain_name(&self) -> &str {
        self.brain.name()
    }

    /// The brain's telemetry JSON (allocates; call at a few Hz, not per decision).
    pub fn brain_telemetry(&self) -> Option<String> {
        self.brain.telemetry()
    }

    pub fn status(&self) -> &Status {
        &self.status
    }

    /// Wall time of every fresh `sealedIn` search (target selection).
    pub fn seal_times(&self) -> &crate::latency::Series {
        &self.picker.seal_times
    }

    /// Wall time of every fresh reachability flood (target selection).
    pub fn reach_times(&self) -> &crate::latency::Series {
        &self.picker.reach_times
    }

    pub fn reach_searches(&self) -> u64 {
        self.picker.reach_searches()
    }

    /// Drains the noteworthy events.
    pub fn drain_events(&mut self) -> std::vec::Drain<'_, BotEvent> {
        self.events.drain(..)
    }

    pub fn tag_of(&self, id: i32) -> Tag {
        self.players.tag(id)
    }

    // ---- events from the client ----------------------------------------------------------------

    /// The map the session loaded (`SessionEvent::MapLoaded`): everything map-specific is rebuilt.
    pub fn on_map_loaded(&mut self, map: Arc<MapData>) {
        self.grid = Some(MapGrid::new(&map));
        self.plan = Some(PlanScratch::new(Arc::clone(&map)));
        self.hooks.navigator.on_map(&map);
        if self.cfg.async_seal {
            match crate::seal_worker::SealWorker::spawn(Arc::clone(&map)) {
                Ok(w) => self.picker.set_seal_worker(Some(w)),
                Err(e) => {
                    tracing::warn!(error = %e, "could not start the seal worker; searching on the decision thread");
                    self.picker.set_seal_worker(None);
                }
            }
        }
        self.map = Some(map);
        self.live = None;
        self.obs = None;
        self.reset_world_state();
    }

    /// `SessionEvent::MapChanging`: the old map's state is stale from here on.
    pub fn on_map_changing(&mut self) {
        self.picker.set_seal_worker(None);
        self.map = None;
        self.grid = None;
        self.live = None;
        self.plan = None;
        self.obs = None;
        self.reset_world_state();
    }

    /// The connection dropped (the driver may bring it back): in-flight knowledge is void.
    pub fn on_disconnected(&mut self) {
        self.sent.clear();
        self.was_alive = false;
        self.last_tick = -1;
    }

    fn reset_world_state(&mut self) {
        self.players.clear();
        self.clock.reset();
        self.picker.reset();
        self.unstick.reset_ticks();
        self.sent.clear();
        self.encoder.reset_edges();
        self.was_alive = false;
        self.last_tick = -1;
        // `join` is deliberately kept: its cap is per run.
        self.played_on_map = false;
    }

    /// `SessionEvent::InputSent`.
    pub fn on_input_sent(&mut self, tick: i32, input: &NetInput) {
        self.sent.on_sent(tick, input);
    }

    /// `SessionEvent::InputTiming`.
    pub fn on_input_timing(&mut self, tick: i32, time_left_ms: i32) {
        self.sent.on_timing(tick, time_left_ms);
    }

    /// `SV_KILLMSG` (`onKill`, `bot.ts:2120`).
    ///
    /// Our own death: DDNet respawns a player right after `Cl_Kill` (`gamecontext.cpp:3008-3009`,
    /// `player.cpp:266`), so no snapshot without our tee need ever arrive. The message is therefore
    /// what ends the life (`bot.ts:2131` sets `wasAlive = false` here too): the death is counted once
    /// and the next snapshot with our tee starts a new life (brain reset, `Respawned`).
    pub fn on_kill_message(&mut self, victim: i32) {
        self.clock.on_kill(victim, self.last_tick);
        if victim == self.players.own_id().unwrap_or(-2) && self.was_alive {
            self.stats.deaths += 1;
            self.was_alive = false;
        }
    }

    /// The bot wants the run to end (see [`StopReason`]).
    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stop
    }

    /// The runner skipped `n` snapshots because newer ones were waiting.
    pub fn note_collapsed(&mut self, n: u64) {
        self.stats.collapsed += n;
    }

    /// The driver's report of one decision reaching the wire.
    pub fn note_wire_latency(&mut self, d: Duration, tick: i32, tag: Option<ddai_client::InputTag>) {
        self.latency.wire.push(d);
        if let Some(t) = tag {
            self.latency.slots.note(tick, t.first_slot, t.expected_tick);
        }
    }

    // ---- the pipeline --------------------------------------------------------------------------

    /// Decides one snapshot. `Output::default()` (no input) until a map is loaded and our id known.
    pub fn on_snapshot(&mut self, snap: &LiveWorldSnapshot) -> Output {
        let started = Instant::now();
        self.queue_delay = started.saturating_duration_since(snap.arrived);
        self.latency.queue.push(self.queue_delay);
        self.stats.snapshots += 1;
        let mut brain_time = Duration::ZERO;
        let out = self.decide(snap, &mut brain_time);
        if out.input.is_some() {
            self.stats.decisions += 1;
            let total = started.elapsed();
            self.latency.record(total, brain_time);
            // Exponential smoothing (0.2): slow brains shift the estimate within a second.
            self.est_decision = self.est_decision.mul_f32(0.8) + total.mul_f32(0.2);
        }
        out
    }

    fn decide(&mut self, snap: &LiveWorldSnapshot, brain_time: &mut Duration) -> Output {
        let mut out = Output::default();
        let (Some(map), Some(own_id)) = (self.map.clone(), snap.own_id) else {
            return out;
        };
        let tick = snap.tick;
        if tick < self.last_tick {
            self.on_tick_reset(tick);
        }
        self.last_tick = tick;
        if self.live.as_ref().is_none_or(|l| l.own_id() != own_id) {
            let live = LiveWorld::new(Arc::clone(&map), own_id, self.cfg.seed);
            self.obs = Some(live.build_observation(live.base_world(), None));
            self.live = Some(live);
        }

        // Disjoint borrows of every part, so the phases below can use them together.
        let Bot {
            cfg,
            relations,
            brain,
            hooks,
            players,
            tees,
            clock,
            picker,
            unstick,
            wander,
            encoder,
            sent,
            grid,
            live,
            plan,
            obs,
            mode,
            was_alive,
            lives,
            join,
            played_on_map,
            stop,
            last_aim,
            last_sent,
            queue_delay,
            est_decision,
            in_flight,
            keep,
            spares,
            spare_tees,
            stats,
            events,
            status,
            ..
        } = self;
        let (Some(live), Some(plan), Some(obs), Some(grid)) =
            (live.as_mut(), plan.as_mut(), obs.as_mut(), grid.as_ref())
        else {
            return out;
        };

        // 3. LiveWorld update with our own in-flight inputs.
        sent.refresh(tick);
        live.on_snapshot(SnapshotInput {
            tick,
            characters: &snap.characters,
            tuning: snap.tuning,
            switch_states: &snap.switch_states,
            teams: snap.teams.as_ref(),
            own_input_at_tick: sent.effective_at(tick),
            projectiles: &snap.projectiles,
        });

        // 4. players, tees, activity.
        if players.update(&snap.players, relations) {
            push_event(
                events,
                BotEvent::RosterChanged {
                    players: players.present().count(),
                },
            );
        }
        tees.rebuild(live.base_world(), &snap.characters);
        clock.update(tick, tees, players, own_id);
        for ev in clock.drain_events() {
            let e = match ev {
                BlockEvent::Block { victim } => BotEvent::Block {
                    tick,
                    victim: players.tag(victim).to_string(),
                },
                BlockEvent::BlockedBy { by } => BotEvent::BlockedBy {
                    tick,
                    by: players.tag(by).to_string(),
                },
            };
            push_event(events, e);
        }

        // 5. dead / absent / spectating.
        let Some(own) = tees.get(own_id).copied() else {
            if *was_alive {
                stats.deaths += 1;
            }
            *was_alive = false;
            let input = encoder.idle();
            stats.idle_decisions += 1;
            out.input = Some(input);
            if *played_on_map && players.get(own_id).is_some_and(|s| s.team == -1) {
                // F3: a spectator after having played is a moderation signal. Stay put and stop.
                if stop.is_none() {
                    *stop = Some(StopReason::MovedToSpectators);
                    push_event(events, BotEvent::MovedToSpectators { tick });
                }
            } else {
                out.set_team = join_request(cfg, players, join, events, tick, own_id);
            }
            *status = make_status(tick, own_id, *mode, None, picker.target(), clock.stats(), *stats);
            return out;
        };
        *played_on_map = true;

        // 6. first frame of a life.
        if !*was_alive {
            *was_alive = true;
            *lives += 1;
            encoder.reset_edges();
            wander.respawned();
            hooks.navigator.respawned();
            *last_sent = PhysInput::default();
            brain.reset(&ResetContext {
                map: Arc::clone(&map),
                self_id: own_id,
                seed: cfg.seed.wrapping_add(*lives),
            });
            push_event(events, BotEvent::Respawned { tick });
        }

        // 7. unstick.
        let acting = *mode != Mode::Hold;
        let wb_kill = hooks.wayblock.holding()
            && hooks.wayblock.wants_kill(
                &HookContext {
                    tick,
                    own: &own,
                    tees,
                    players,
                    grid,
                },
                unstick.frozen_for(tick, &own),
            );
        let verdict = unstick.step(&UnstickCtx {
            tick,
            own: &own,
            tees,
            players,
            grid,
            target: picker.target(),
            acting,
            in_dead_zone: hooks.navigator.in_dead_zone(own.pos),
            wayblock_wants_kill: wb_kill,
        });
        if let Verdict::Kill(reason) = verdict {
            out.kill = true;
            stats.self_kills += 1;
            push_event(events, BotEvent::Killed { tick, reason });
        }

        // 8. mode.
        if !acting {
            stats.idle_decisions += 1;
            out.input = Some(encoder.idle());
            *status = make_status(tick, own_id, *mode, Some(&own), picker.target(), clock.stats(), *stats);
            return out;
        }
        let nav = hooks.navigator.drive(&HookContext {
            tick,
            own: &own,
            tees,
            players,
            grid,
        });
        let mut navigated = None;
        match nav {
            Some(NavStep::Kill) => {
                if !out.kill && unstick.cooldown_ready(tick) {
                    unstick.note_external_kill(tick);
                    stats.self_kills += 1;
                    out.kill = true;
                }
                navigated = Some(Action::neutral());
            }
            Some(NavStep::Input(a)) => navigated = Some(a),
            None => {}
        }
        let previous_target = picker.target();
        let t_pick = Instant::now();
        let target = if navigated.is_some() || !matches!(*mode, Mode::Fight | Mode::Goto) {
            -1
        } else {
            picker.pick(
                &PickCtx {
                    tick,
                    own: &own,
                    tees,
                    players,
                    clock,
                    grid,
                    base: live.base_world(),
                },
                hooks,
                plan,
            )
        };
        if navigated.is_none() && matches!(*mode, Mode::Fight | Mode::Goto) {
            self.latency.pick.push(t_pick.elapsed());
        }
        picker.set_target(target);
        if target != previous_target {
            let to = (target >= 0).then(|| players.tag(target).to_string());
            push_event(events, BotEvent::TargetChanged { tick, to });
        }

        // 9. the action.
        let mut action;
        let mut expected_tick: Option<i32> = None;
        let own_shield = cfg.brain.has_own_shield();
        if let Some(a) = navigated {
            action = a;
            stats.idle_decisions += 1;
        } else if target < 0 {
            if cfg.brain == BrainKind::Idle {
                action = Action::neutral();
                stats.idle_decisions += 1;
            } else {
                spare_tees.clear();
                for t in tees.iter() {
                    if t.id != own.id
                        && dist(t.pos, own.pos) <= HOOK_LENGTH_PX + 64.0
                        && is_spared(t, tick, players, clock)
                    {
                        spare_tees.push(*t);
                    }
                }
                let prev_planner = from_ddnet_input(last_sent);
                let mut env = WanderEnvImpl {
                    plan,
                    world: live.base_world(),
                    own: &own,
                    tees,
                    grid,
                    prev: &prev_planner,
                    guarded: false,
                    guard_time: Duration::ZERO,
                };
                action = wander.step(
                    &WanderCtx {
                        tick,
                        own: &own,
                        grid,
                        prev_aim: *last_aim,
                        anchor_x: None,
                        look_at: None,
                    },
                    &mut env,
                );
                if env.guarded {
                    stats.guarded_inputs += 1;
                }
                // The shield belongs to the brain's share of the time (for the planner it is inside
                // the brain's own call), not to the bot's overhead.
                *brain_time += env.guard_time;
                stats.wander_decisions += 1;
            }
        } else {
            let target_tee = tees.get(target).copied();
            compute_keep(&own, target, tees, cfg, &|t| is_spared(t, tick, players, clock), keep);
            let to_tick = prediction_target(
                snap,
                cfg.max_predict_ticks,
                *queue_delay + *est_decision + DRIVER_PICKUP,
            );
            expected_tick = Some(to_tick + 1);
            sent.in_flight(tick, to_tick, in_flight);
            // The observation's target is a tee that is still in the world (it may have just died).
            let target_id = target_tee.map(|t| t.id);
            let predicted = live.predict_local_observation(to_tick, in_flight, keep, target_id, obs);

            spares.clear();
            spare_tees.clear();
            for t in tees.iter() {
                if t.id != own.id
                    && t.id != target
                    && dist(t.pos, own.pos) <= HOOK_LENGTH_PX + 64.0
                    && is_spared(t, tick, players, clock)
                {
                    spares.push((t.pos, t.vel));
                    spare_tees.push(*t);
                }
            }
            let travel_goal = match target_tee.as_ref() {
                Some(tt) => hooks.trek.goal(
                    &HookContext {
                        tick,
                        own: &own,
                        tees,
                        players,
                        grid,
                    },
                    tt,
                ),
                None => None,
            };
            brain.set_live_context(&LiveContext {
                spares: spares.as_slice(),
                travel_goal,
            });
            let view = WorldView {
                world: predicted,
                self_id: own_id,
                lag_ticks: 0,
                in_flight: &[],
            };
            let t0 = Instant::now();
            action = brain.decide_in(obs, Some(&view));
            *brain_time = t0.elapsed();
            stats.brain_decisions += 1;

            // Post-filters. The guard first (brains without their own shield), then the hook veto.
            let mut filtered = action;
            if !own_shield && grid.hazard_within(own.pos.x, own.pos.y, GUARD_HAZARD_TILES) {
                let t_guard = Instant::now();
                let prev_planner = from_ddnet_input(last_sent);
                let g = plan.guard(predicted, own_id, filtered, &prev_planner);
                if g != filtered {
                    stats.guarded_inputs += 1;
                }
                filtered = g;
                *brain_time += t_guard.elapsed();
            }
            if hook_veto(&mut filtered, &own, target_tee.as_ref(), spare_tees, players, plan) {
                stats.vetoed_hooks += 1;
            }
            if hammer_veto(
                &mut filtered,
                &own,
                obs.self_state.pos,
                (to_tick + 1 - tick).max(0) as f32,
                spare_tees,
            ) {
                stats.vetoed_fires += 1;
            }
            action = filtered;
        }

        // 10. encode and out.
        let (input, info) = encoder.encode(&action);
        if info.hook_rising {
            stats.hooks_fired += 1;
        }
        if info.fire_pressed && own.holding_hammer() {
            stats.hammer_fires += 1;
        }
        *last_aim = (input.target_x, input.target_y);
        *last_sent = player_input_from_net(input);
        out.input = Some(input);
        out.tag = Some(ddai_client::InputTag {
            first_slot: snap.pred_tick.max(snap.tick) + 1,
            expected_tick: expected_tick.unwrap_or(snap.pred_tick.max(snap.tick) + 1),
        });
        *status = make_status(tick, own_id, *mode, Some(&own), picker.target(), clock.stats(), *stats);
        out
    }

    fn on_tick_reset(&mut self, to: i32) {
        push_event(
            &mut self.events,
            BotEvent::TickReset {
                from: self.last_tick,
                to,
            },
        );
        self.stats.ticks_resets += 1;
        self.clock.reset();
        self.picker.reset();
        self.unstick.reset_ticks();
        self.sent.clear();
        self.encoder.reset_edges();
        self.was_alive = false;
    }
}

fn push_event(events: &mut Vec<BotEvent>, e: BotEvent) {
    if events.len() < EVENT_CAP {
        events.push(e);
    }
}

fn make_status(
    tick: i32,
    own_id: i32,
    mode: Mode,
    own: Option<&Tee>,
    target_id: i32,
    blocks: BlockStats,
    stats: BotStats,
) -> Status {
    Status {
        tick,
        own_id,
        target_id,
        mode,
        alive: own.is_some(),
        frozen: own.is_some_and(|t| t.frozen),
        blocks,
        stats,
    }
}

/// `maybeJoinGame` (`bot.ts:2606-2618`): while we sit in the spectators (team -1) ask to join, not
/// more often than every `JOIN_RETRY_TICKS` (3 s), and give up after `max_join_attempts` tries — a
/// server that keeps putting us back is telling us something, and D-016 says to stop, not to push.
fn join_request(
    cfg: &BotConfig,
    players: &PlayerTable,
    join: &mut JoinState,
    events: &mut Vec<BotEvent>,
    tick: i32,
    own_id: i32,
) -> Option<i32> {
    let spectating = players.get(own_id).is_some_and(|s| s.team == -1);
    if !spectating || cfg.mode == Mode::Hold || join.attempts >= cfg.max_join_attempts {
        return None;
    }
    let due = join
        .last_try_tick
        .is_none_or(|t| tick - t >= JOIN_RETRY_TICKS || tick < t);
    if !due {
        return None;
    }
    join.last_try_tick = Some(tick);
    join.attempts += 1;
    push_event(events, BotEvent::Joining { tick });
    if join.attempts >= cfg.max_join_attempts {
        push_event(events, BotEvent::JoinGaveUp { tick });
    }
    Some(0)
}

/// What the driver adds between `Client::set_input` and the socket when the input is otherwise ready
/// (one poll round, `POLL_TIMEOUT`).
const DRIVER_PICKUP: Duration = Duration::from_millis(2);
/// One server tick.
const TICK: Duration = Duration::from_millis(20);

/// The tick the prediction must reach (see the module docs): the predicted tick of the last input the
/// driver sent — plus one more for every input that will go out *before* this decision is ready. The
/// driver's next `NETMSG_INPUT` is due `next_input_in` after the snapshot arrived; a decision that will
/// be ready (`ready_in` after arrival: channel hop + estimated decision time + driver pickup) only
/// after that goes out with the tick after, so the world must be predicted one tick further, and so on.
/// Before the driver's two-snapshot bootstrap (`pred_tick == 0`) a 2-tick guess.
fn prediction_target(snap: &LiveWorldSnapshot, max_ahead: i32, ready_in: Duration) -> i32 {
    let want = if snap.pred_tick > 0 {
        let first = snap.next_input_in.unwrap_or(Duration::ZERO);
        let missed = if ready_in <= first {
            0
        } else {
            // Inputs sent at `first`, `first + 20 ms`, ... before the decision is ready.
            ((ready_in - first).as_nanos() / TICK.as_nanos()) as i32 + 1
        };
        snap.pred_tick + missed
    } else {
        snap.tick + 2
    };
    want.clamp(snap.tick, snap.tick + max_ahead)
}

/// Fills `keep` with the tees the brain gets: the target, everyone roped to us, and the nearest
/// others within the threat radius up to `max_local_others`. **Spared tees** (friends, ignored,
/// out of game, AFK) are not given to the brain unless roped to us (review round 1, F1): a brain
/// that models every tee in its world as an opponent — the hybrid's threat model — would plan
/// against, and swing at, someone it must leave alone. They reach the brain as
/// `LiveContext::spares` instead.
fn compute_keep(
    own: &Tee,
    target: i32,
    tees: &TeeSet,
    cfg: &BotConfig,
    spared: &dyn Fn(&Tee) -> bool,
    keep: &mut [bool; MAX_CLIENTS],
) {
    keep.fill(false);
    if let Some(slot) = usize::try_from(target).ok().filter(|&i| i < MAX_CLIENTS) {
        keep[slot] = true;
    }
    let mut chosen = 0usize;
    for t in tees.iter() {
        if t.id != own.id && t.id != target && (t.hooked_player == own.id || own.hooked_player == t.id) {
            keep[t.id as usize] = true;
            chosen += 1;
        }
    }
    // The nearest of the rest, by (distance, id), selected into a small fixed array.
    let cap = cfg.max_local_others;
    let mut best: [(f32, i32); 16] = [(f32::INFINITY, -1); 16];
    let cap = cap.min(best.len());
    let mut filled = 0usize;
    for t in tees.iter() {
        if t.id == own.id || t.id == target || keep[t.id as usize] || spared(t) {
            continue;
        }
        let d = dist(own.pos, t.pos);
        if d > cfg.threat_radius_px {
            continue;
        }
        if filled < cap {
            best[filled] = (d, t.id);
            filled += 1;
        } else if let Some((worst, slot)) = best[..filled]
            .iter()
            .enumerate()
            .map(|(i, &(dd, id))| ((dd, id), i))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            && (d, t.id) < worst
        {
            best[slot] = (d, t.id);
        }
    }
    let room = cap.saturating_sub(chosen);
    let mut picked = best;
    picked[..filled].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    for &(_, id) in picked[..filled.min(room)].iter() {
        keep[id as usize] = true;
    }
}

/// The hook veto (`planAction` step 9, `bot.ts:4785`): drop a hook that would catch a spared tee
/// before the target or a wall, and never keep holding a spared tee or a friend.
fn hook_veto(
    action: &mut Action,
    own: &Tee,
    target: Option<&Tee>,
    spared: &[Tee],
    players: &PlayerTable,
    plan: &PlanScratch,
) -> bool {
    if !action.hook || (spared.is_empty() && own.hooked_player < 0) {
        return false;
    }
    let holding = own.hooked_player >= 0
        && (spared.iter().any(|t| t.id == own.hooked_player)
            || players.get(own.hooked_player).is_some_and(|s| s.flags.friendly()));
    if holding || (own.hook_state == HOOK_IDLE && plan.rope_catches(own.pos, action.target, spared.iter(), target)) {
        action.hook = false;
        return true;
    }
    false
}

/// The hammer veto (review round 1, F1, the second line of defence): withhold a fire press when the
/// current weapon is the hammer and the swing would hit a spared tee. DDNet's `fire_hammer` geometry
/// (as derived in `ddai-planner`'s physics adapter and its `hammer_would_hit`): everybody within
/// `1.5 * 28` px of the point `0.75 * 28 = 21` px in front of us along the aim is hit. The input
/// takes effect `ahead` ticks after the snapshot, so the swing is checked from our predicted
/// position (`own_then`) and from where we are now, against each spared tee now, `ahead` ticks on
/// and two more ticks on (as `hammer_would_hit` does): conservative on purpose. Walls are ignored
/// (a miss through a wall is not worth the risk).
fn hammer_veto(action: &mut Action, own: &Tee, own_then: Vec2<f32>, ahead: f32, spared: &[Tee]) -> bool {
    if !action.fire || !own.holding_hammer() || spared.is_empty() {
        return false;
    }
    let (ax, ay) = (action.target.x as f32, action.target.y as f32);
    let len = ax.hypot(ay);
    if len < 1e-6 {
        return false;
    }
    let (dx, dy) = (ax / len * HAMMER_REACH_AHEAD_PX, ay / len * HAMMER_REACH_AHEAD_PX);
    let reach = HAMMER_HIT_RADIUS_PX;
    let would_hit = [own.pos, own_then].into_iter().any(|me| {
        let (sx, sy) = (me.x + dx, me.y + dy);
        spared.iter().any(|t| {
            [0.0, 2.0, ahead, ahead + 2.0]
                .into_iter()
                .any(|k| (t.pos.x + t.vel.x * k - sx).hypot(t.pos.y + t.vel.y * k - sy) < reach)
        })
    });
    if would_hit {
        action.fire = false;
    }
    would_hit
}

/// The wander step's guard and rope check, over the planner helpers.
struct WanderEnvImpl<'a> {
    plan: &'a mut PlanScratch,
    world: &'a ddai_physics::world::World<f32>,
    own: &'a Tee,
    tees: &'a TeeSet,
    grid: &'a MapGrid,
    prev: &'a PlannerInput,
    guarded: bool,
    guard_time: Duration,
}

impl WanderEnv for WanderEnvImpl<'_> {
    fn guard(&mut self, wanted: Action) -> Action {
        if !self
            .grid
            .hazard_within(self.own.pos.x, self.own.pos.y, GUARD_HAZARD_TILES)
        {
            return wanted; // no freeze or death within reach: nothing to shield from
        }
        let t = Instant::now();
        let g = self.plan.guard(self.world, self.own.id, wanted, self.prev);
        self.guard_time += t.elapsed();
        self.guarded |= g != wanted;
        g
    }

    fn rope_catches(&mut self, aim: IVec2) -> bool {
        self.plan.rope_catches(
            self.own.pos,
            aim,
            self.tees.iter().filter(|t| t.id != self.own.id),
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tee_at(id: i32, x: f32, vx: f32) -> Tee {
        Tee {
            id,
            alive: true,
            pos: Vec2 { x, y: 500.0 },
            vel: Vec2 { x: vx, y: 0.0 },
            ..Tee::DEAD
        }
    }

    fn swing_right() -> Action {
        Action {
            fire: true,
            target: IVec2::new(100, 0),
            ..Action::neutral()
        }
    }

    #[test]
    fn the_hammer_veto_uses_the_hit_circle_in_front_of_the_tee() {
        let me = tee_at(0, 1000.0, 0.0);
        // The swing is centred at 1021 with a 42 px circle: 1062 is inside, 1064 is outside.
        for (x, vetoed) in [(1062.0, true), (1064.0, false), (980.0, true), (978.0, false)] {
            let mut a = swing_right();
            let hit = hammer_veto(&mut a, &me, me.pos, 0.0, &[tee_at(1, x, 0.0)]);
            assert_eq!(hit, vetoed, "spared tee at {x}");
            assert_eq!(a.fire, !vetoed, "at {x}");
        }
    }

    #[test]
    fn the_hammer_veto_looks_ahead_along_the_spared_tees_velocity_and_our_predicted_position() {
        let me = tee_at(0, 1000.0, 0.0);
        // 90 px away, running left at 12 px/tick: two ticks on it is at 1066... still out; three
        // ticks (`ahead`) on 1054 is inside the circle.
        let mut a = swing_right();
        assert!(hammer_veto(&mut a, &me, me.pos, 3.0, &[tee_at(1, 1090.0, -12.0)]));
        let mut a = swing_right();
        assert!(!hammer_veto(&mut a, &me, me.pos, 0.0, &[tee_at(1, 1090.0, 0.0)]));
        // We are predicted to run 30 px right before the input takes effect: the swing centre moves.
        let mut a = swing_right();
        assert!(hammer_veto(
            &mut a,
            &me,
            Vec2 { x: 1030.0, y: 500.0 },
            3.0,
            &[tee_at(1, 1090.0, 0.0)]
        ));
    }

    #[test]
    fn the_hammer_veto_leaves_other_weapons_non_fire_and_zero_aim_alone() {
        let mut me = tee_at(0, 1000.0, 0.0);
        let friend = [tee_at(1, 1030.0, 0.0)];
        me.weapon = 1; // the gun
        let mut a = swing_right();
        assert!(!hammer_veto(&mut a, &me, me.pos, 0.0, &friend));
        me.weapon = 0;
        let mut a = Action {
            fire: false,
            ..swing_right()
        };
        assert!(!hammer_veto(&mut a, &me, me.pos, 0.0, &friend));
        let mut a = Action {
            target: IVec2::new(0, 0),
            ..swing_right()
        };
        assert!(!hammer_veto(&mut a, &me, me.pos, 0.0, &friend), "no aim, no geometry");
        let mut a = swing_right();
        assert!(!hammer_veto(&mut a, &me, me.pos, 0.0, &[]), "nobody spared");
    }
}
