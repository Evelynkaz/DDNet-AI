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

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ddai_brain::{Action, Brain, IVec2, LiveContext, Observation, ResetContext, WorldView};
use ddai_client::LiveWorldSnapshot;
use ddai_clip::format::{BotRec, ClipEvent, KillWhy};
use ddai_net::generated::enums::explayerflagflag;
use ddai_net::generated::objects::PlayerInput as NetInput;
use ddai_physics::core::{MAX_CLIENTS, PlayerInput as PhysInput, WEAPON_HAMMER};
use ddai_physics::map::MapData;
use ddai_physics::vmath::Vec2;
use ddai_planner::physics_adapter::from_ddnet_input;
use ddai_planner::types::PlayerInput as PlannerInput;
use ddai_world::{LiveWorld, SnapshotInput, player_input_from_net};

use crate::activity::{ActivityClock, BlockEvent, BlockStats};
use crate::brains::{BrainKind, BrainOptions};
use crate::clipper::{ClipConfig, Clipper, FrameInput};
use crate::consts::*;
use crate::hooks::{HookContext, Hooks, MapIdent, NavStep};
use crate::input::InputEncoder;
use crate::latency::{DecisionEstimator, LatencyStats};
use crate::mapgrid::MapGrid;
use crate::nav_hooks::NavHandle;
use crate::planning::PlanScratch;
use crate::players::{PlayerTable, Salt, Tag};
use crate::relations::Relations;
use crate::sent::SentLog;
use crate::target::{PickCtx, TargetPicker, is_spared};
use crate::tees::{HOOK_IDLE, Tee, TeeSet, dist};
use crate::unstick::{KillReason, Unstick, UnstickCtx, Verdict};
use crate::wander::{Wander, WanderCtx, WanderEnv};

mod apply;

/// What the bot does (`mode`, §8.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Pick a target and fight (the default).
    Fight,
    /// Never pick a target: wander only.
    Passive,
    /// Do nothing (neutral input; the unstick rules still ignore us).
    Hold,
    /// A goto / follow walk is in progress (task 4.2): the navigator hook drives the input; when the walk
    /// ends the navigator hands the bot back to the mode the walk began from. With the no-op navigator
    /// (`Hooks::default()`) nothing ever drives and this behaves like `Fight`.
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
    /// The quantile of recent decision times used as the estimate that picks the input slot
    /// (task 4.1b; 0.9 = p90).
    pub estimate_quantile: f64,
    /// **Tests only.** When set, the bot's estimate of its own decision (queue delay + decision time, the
    /// input of the slot choice in `prediction_target`) is this fixed value instead of the measured
    /// rolling quantile, so a scenario's expected prediction tick does not depend on how long the host
    /// happened to take (a ~10 ms pause on a loaded CI machine flips it). `None` (the default and every
    /// production path) measures as before. The latency statistics still record the real times.
    pub decision_time_override: Option<Duration>,
    /// **Tests only.** `Some((other, brain))`: what the two rolling estimates are fed instead of the measured duration of
    /// each decision, `brain` for a decision that ran the brain and `other` for the rest (and the queue delay counts as zero), so
    /// that a test of the slot choice can give the estimators exactly the distribution it wants without a real wall clock.
    /// `None` (the default and every production path) feeds the measured time.
    pub decision_time_feed: Option<(Duration, Duration)>,
    /// The clip recorder: where clips go, the autoclip (task 4.3).
    pub clips: ClipConfig,
    /// Where the console commands save the lists (`None`: they change only the running bot).
    pub relations_path: Option<PathBuf>,
    /// Where `!brain`, `!wb`, `!low` and `!strong` are remembered (`None`: not remembered).
    pub settings_path: Option<PathBuf>,
    /// `!low` / `!strong` as the run starts (the brain options and the navigation config carry the effect).
    pub low: bool,
    pub strong: bool,
    /// Console replies name other players by their real nickname (`--console-names`); by default by tag.
    pub console_names: bool,
    /// Task 3.10 (`--finish target`, opt-in): finish blocks -- the target selection keeps a frozen current target until it is held
    /// ([`crate::target::TargetPicker::set_finish`]).
    pub finish: bool,
    /// `--no-selfkill` (D-102): the bot never kills itself ([`Bot::set_no_selfkill`]); the owner's `!kill` stays. The runner also
    /// re-reads [`BotConfig::selfkill_marker`] once a second ([`crate::selfkill`]).
    pub no_selfkill: bool,
    /// The marker file `<data-dir>/bot/selfkill.off` (D-102): while it exists the switch is on. `None`: no marker.
    pub selfkill_marker: Option<PathBuf>,
    /// Task 3.11 (diagnosis, off by default): write the per-input trace of [`crate::trace`] here (`DDAI_INPUT_TRACE` is the
    /// same for a process that has one bot).
    pub input_trace: Option<PathBuf>,
    /// Task 3.11 (opt-in): aim a decision that is about to run the brain at the input slot the **brain decisions'** own rolling
    /// quantile says, instead of the quantile over every decision. The mixed estimate is bimodal (wandering decisions take
    /// ~0.3 ms, the brain ~5 ms): while less than a tenth of the last 64 decisions ran the brain, its p90 is the cheap mode and
    /// the first brain decisions of an engagement (the hook, the jump) are aimed one slot too early and go out a tick later than
    /// the world they were decided on assumed.
    pub kind_estimate: bool,
    /// What the driver adds between handing a decision over and it being on the wire when the slot is otherwise open
    /// ([`DRIVER_PICKUP`]; the runner lowers it to [`DRIVER_PICKUP_PRECISE`] with `ClientConfig::precise_wakeups`).
    pub driver_pickup: Duration,
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
            estimate_quantile: DEFAULT_ESTIMATE_QUANTILE,
            decision_time_override: None,
            decision_time_feed: None,
            clips: ClipConfig::default(),
            relations_path: None,
            settings_path: None,
            low: false,
            strong: false,
            console_names: false,
            finish: false,
            no_selfkill: false,
            selfkill_marker: None,
            input_trace: None,
            kind_estimate: false,
            driver_pickup: DRIVER_PICKUP,
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
    /// Send the chat command `/kill` (`Client::server_command(ServerCommand::Kill)`, D-078): a protocol `Cl_Kill` had no effect
    /// (kill protection) and the cooldown and the per-decision limit of [`crate::killfallback`] allow it.
    pub kill_command: bool,
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
    /// Task 3.10: a block of ours was still on 5 s later, or its victim died inside the window (`died`).
    BlockHeld {
        tick: i32,
        victim: String,
        died: bool,
    },
    /// Task 3.10: the victim of a block of ours was free again `after` ticks after it.
    BlockEscaped {
        tick: i32,
        victim: String,
        after: i32,
    },
    TargetChanged {
        tick: i32,
        to: Option<String>,
    },
    Respawned {
        tick: i32,
    },
    /// The `/kill` fallback was sent (D-078): `noticed` says the server's "Kill Protection enabled" line was seen, else the
    /// protocol `Cl_Kill` simply had no effect within 50 ticks.
    KillFallback {
        tick: i32,
        noticed: bool,
    },
    /// A `/kill` ended a life after this many seconds: the server's kill-protection threshold is not above it (tags only, no names).
    KillProtectionLearned {
        tick: i32,
        life_secs: u32,
    },
    /// Three `/kill`s in one life without a death: the bot stops asking for this life.
    KillFallbackGaveUp {
        tick: i32,
    },
    Joining {
        tick: i32,
    },
    JoinGaveUp {
        tick: i32,
    },
    /// The prediction horizon the decision wanted exceeded the cap (reported at most once per 10 s):
    /// the bot decides on a world that stops short of the tick its input takes effect on.
    PredictionClamped {
        tick: i32,
        wanted_ahead: i32,
        cap: i32,
    },
    TickReset {
        from: i32,
        to: i32,
    },
    /// We were moved to the spectators after having played: the bot stops (exit code 3).
    MovedToSpectators {
        tick: i32,
    },
    /// The server paused us (our own `DDNetPlayer` flag `SPEC` or `PAUSED`: the owner's `/pause` or `/spec`, task 4.9b): the bot idles.
    PausedByServer {
        tick: i32,
    },
    /// The server's pause flag cleared: the bot plays again.
    ResumedByServer {
        tick: i32,
    },
    RosterChanged {
        players: usize,
    },
    /// A clip reached the disk (task 4.3): an incident kind, `cross-fail`; no nicknames.
    ClipSaved {
        tick: i32,
        kind: String,
        severity: i32,
        path: String,
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
    /// Decisions whose prediction horizon was cut by the cap (a very long RTT; `PredictionClamped`).
    pub predict_clamped: u64,
    /// Clips that reached the disk (automatic ones and `!clip`).
    pub clips_saved: u64,
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
    clipper: Clipper,
    /// The navigation's walk label of this frame (a buffer reused by the clip recorder).
    walk_label: String,
    /// Why `Cl_Kill` is being requested by this decision (for the clip's `KillSent` event).
    kill_why: Option<KillWhy>,

    // the console commands (`Bot::command`, task 4.3)
    /// The navigation's command channel (`goto`, `home`, `wb`, `strong`), once attached.
    nav: Option<NavHandle>,
    brain_opts: BrainOptions,
    /// Counts brain replacements (`!brain`), so the driver can re-announce the brain's visualisation stream.
    brain_generation: u64,
    /// The brain decided on the latest snapshot (and its frame has not been asked for yet): only then is there a frame
    /// of *this* decision. A snapshot the bot answered without the brain (no target, a hook, a hold) has none, however
    /// recent the brain's last decision was.
    viz_fresh: bool,
    /// `!low` / `!strong`.
    low: bool,
    strong: bool,
    /// `!spec`: we asked to go to the spectators and mean to stay there (no auto-join, no moderation stop).
    wants_spectate: bool,
    /// `!join` after `!spec`: until this tick (or until the server shows us in the game: a tee, or a team other
    /// than -1) a spectator state is the operator's own doing still settling, not a moderation move.
    join_grace_until: Option<i32>,
    /// A `Cl_SetTeam` the next `on_snapshot` sends (`!spec` / `!join`).
    pending_team: Option<i32>,
    /// A `Cl_Kill` the next `on_snapshot` sends (`!kill`).
    pending_kill: bool,
    /// The `/kill` fallback after a `Cl_Kill` that had no effect (task 4.6, D-078).
    killfb: crate::killfallback::KillFallback,
    /// `--no-selfkill` (D-102): no bot-initiated `Cl_Kill` or `/kill`.
    no_selfkill: bool,
    /// `lives` as the fallback last saw it (a change is a new life).
    fb_lives: u64,
    quit: bool,
    /// The session is in the game (`SessionEvent::InGame` seen, no disconnect since): for the web status.
    connected: bool,

    map: Option<Arc<MapData>>,
    /// Name and hash of the map being loaded (set by the runner before `on_map_loaded`).
    map_ident: MapIdent,
    grid: Option<MapGrid>,
    live: Option<LiveWorld>,
    plan: Option<PlanScratch>,
    obs: Option<Observation>,

    mode: Mode,
    last_tick: i32,
    was_alive: bool,
    /// Our own `DDNetPlayer` flag `SPEC` or `PAUSED` is set (task 4.9b): the bot idles until it clears.
    paused: bool,
    lives: u64,
    join: JoinState,
    /// Whether our tee has existed on the current map (a later spectator state is then a move).
    played_on_map: bool,
    stop: Option<StopReason>,
    last_aim: (i32, i32),
    /// The tick of the last `PredictionClamped` event (rate limit).
    last_clamp_event: i32,
    last_sent: PhysInput,
    /// Snapshot arrival -> this decision started (the channel hop), set by `on_snapshot`.
    queue_delay: Duration,
    /// Conservative (rolling-quantile) duration of a decision, bot thread and brain included: decides
    /// which input slot this snapshot's decision is aimed at (see `prediction_target`).
    est_decision: Duration,
    estimator: DecisionEstimator,
    /// Task 3.11: the same estimate over the decisions that ran the brain only (`BotConfig::kind_estimate`).
    est_brain: Duration,
    estimator_brain: DecisionEstimator,

    // scratch (capacities fixed up front)
    in_flight: Vec<(i32, PhysInput)>,
    keep: Box<[bool; MAX_CLIENTS]>,
    spares: Vec<(Vec2<f32>, Vec2<f32>)>,
    spare_ids: Vec<i32>,
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
        let mut picker = TargetPicker::new(cfg.fixed_target.as_deref());
        picker.set_finish(cfg.finish);
        Bot {
            players: PlayerTable::new(cfg.salt),
            tees: TeeSet::new(),
            clock: ActivityClock::new(),
            picker,
            unstick: Unstick::new(),
            wander: Wander::new(cfg.seed),
            encoder: InputEncoder::new(),
            sent: SentLog::new(),
            clipper: {
                let mut c = Clipper::new(cfg.clips.clone(), cfg.seed);
                c.set_brain(brain.name());
                c
            },
            walk_label: String::with_capacity(64),
            kill_why: None,
            nav: None,
            brain_opts: BrainOptions::default(),
            brain_generation: 0,
            viz_fresh: false,
            low: cfg.low,
            strong: cfg.strong,
            wants_spectate: false,
            join_grace_until: None,
            pending_team: None,
            pending_kill: false,
            killfb: crate::killfallback::KillFallback::new(),
            no_selfkill: false,
            fb_lives: 0,
            connected: false,
            quit: false,
            map: None,
            map_ident: MapIdent::default(),
            grid: None,
            live: None,
            plan: None,
            obs: None,
            mode,
            last_tick: -1,
            was_alive: false,
            paused: false,
            lives: 0,
            join: JoinState::default(),
            played_on_map: false,
            stop: None,
            last_clamp_event: i32::MIN / 2,
            last_aim: (0, -1),
            last_sent: PhysInput::default(),
            queue_delay: Duration::ZERO,
            est_decision: cfg.decision_time_override.unwrap_or(Duration::from_millis(1)),
            estimator: DecisionEstimator::new(cfg.estimate_quantile, Duration::from_millis(1)),
            est_brain: cfg.decision_time_override.unwrap_or(BRAIN_ESTIMATE_INITIAL),
            estimator_brain: DecisionEstimator::new(cfg.estimate_quantile, BRAIN_ESTIMATE_INITIAL),
            in_flight: Vec::with_capacity(64),
            keep: Box::new([false; MAX_CLIENTS]),
            spares: Vec::with_capacity(MAX_CLIENTS),
            spare_ids: Vec::with_capacity(MAX_CLIENTS),
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

    /// Attaches the navigation's command channel (the runner does, with the handle the hooks were built on).
    pub fn set_nav_handle(&mut self, handle: NavHandle) {
        self.nav = Some(handle);
    }

    /// The knobs `!brain` builds the next brain with.
    pub fn set_brain_options(&mut self, opts: BrainOptions) {
        self.brain_opts = opts;
    }

    /// The operator asked to quit (`!quit`): the runner stops.
    pub fn quit_requested(&self) -> bool {
        self.quit
    }

    pub fn low(&self) -> bool {
        self.low
    }

    pub fn strong(&self) -> bool {
        self.strong
    }

    pub fn wants_spectate(&self) -> bool {
        self.wants_spectate
    }

    /// Whether the grace after `!join` is still running.
    pub fn join_grace_active(&self) -> bool {
        self.join_grace_until.is_some()
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

    /// The name of the map being played (empty before the first one).
    pub fn map_name(&self) -> &str {
        &self.map_ident.name
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

    /// Task 7.4: the layout of the brain's visualisation stream (`None`: it has none). Allocates.
    pub fn viz_meta(&self) -> Option<String> {
        self.brain.viz_meta()
    }

    /// Task 7.4: the visualisation frame of the latest decision. The driver calls it after a snapshot **only while a
    /// viewer is subscribed**; with nobody watching the brain does nothing for its stream.
    pub fn viz_frame(&mut self, tick: u32) -> Option<&[u8]> {
        if !std::mem::take(&mut self.viz_fresh) {
            return None;
        }
        self.brain.viz_frame(tick)
    }

    /// How many times the brain has been replaced since the start; changes when the stream's layout may have.
    pub fn brain_generation(&self) -> u64 {
        self.brain_generation
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

    /// `!clip [note]`: saves the ring now as `manual-<tick>[-<note>]` in the clip directory.
    pub fn save_clip(&mut self, note: &str) -> Result<crate::clipper::SavedClip, String> {
        let own = self.players.own_id().ok_or("not in the game yet")?;
        self.clipper.save_manual(note, own, &self.players, None)
    }

    /// Frames in the clip ring now.
    pub fn clip_frames(&self) -> usize {
        self.clipper.frames()
    }

    /// Waits for the clip worker to be idle and reports what it saved (tests, the end of a run).
    pub fn flush_clips(&mut self) {
        self.clipper.flush();
        self.report_saved_clips();
    }

    pub fn clips(&self) -> &Clipper {
        &self.clipper
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
        self.hooks.navigator.on_map(&map, &self.map_ident);
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
        self.clipper.set_map(&self.map_ident.name, self.map_ident.sha256);
    }

    /// The name and hash of the map the next [`Bot::on_map_loaded`] brings (the wayblock is chosen by
    /// name, the freeze memory is keyed by the hash).
    pub fn set_map_ident(&mut self, ident: MapIdent) {
        self.map_ident = ident;
    }

    /// `SessionEvent::MapChanging`: the old map's state is stale from here on.
    pub fn on_map_changing(&mut self) {
        self.hooks.navigator.on_map_changing();
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
        self.connected = false;
        self.killfb.reset();
        self.fb_lives = self.lives;
        self.clipper.reset();
        self.sent.clear();
        self.was_alive = false;
        self.paused = false;
        self.last_tick = -1;
    }

    fn reset_world_state(&mut self) {
        self.clipper.reset();
        self.players.clear();
        self.clock.reset();
        self.picker.reset();
        self.unstick.reset_ticks();
        self.sent.clear();
        self.encoder.reset_edges();
        self.was_alive = false;
        self.last_tick = -1;
        self.killfb.reset();
        self.fb_lives = self.lives;
        // The server's pause is judged again from the next snapshot (task 4.9b).
        self.paused = false;
        // `join` is deliberately kept: its cap is per run.
        self.played_on_map = false;
    }

    /// `SessionEvent::InGame`: the session entered the game.
    pub fn on_in_game(&mut self) {
        self.connected = true;
    }

    /// Whether the server has paused us (the owner's `/pause` or `/spec`): the bot idles and the web status says so.
    pub fn paused(&self) -> bool {
        self.paused
    }

    /// Whether the session is in the game (the web status's "connected").
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Switches the duel mode "never kill ourselves" (D-102) on or off. On: the unstick, the wayblock rule, the navigation's and the
    /// trek's respawn steps and the `/kill` fallback of those kills decide nothing (and routes needing a respawn are not planned). The owner's own
    /// `!kill` (console) and typed chat lines are not the bot's and stay; the console `!kill` also keeps its `/kill` fallback.
    pub fn set_no_selfkill(&mut self, off: bool) {
        self.no_selfkill = off;
        if off {
            // A `Cl_Kill` the bot sent before the switch went on is no longer awaited (no `/kill` for it).
            self.killfb.cancel_pending();
        }
        self.unstick.set_no_kill(off);
        self.hooks.navigator.set_no_selfkill(off);
    }

    /// Whether [`Bot::set_no_selfkill`] is on.
    pub fn no_selfkill(&self) -> bool {
        self.no_selfkill
    }

    /// Ticks until `Cl_Kill` is allowed again (0: now, also before any snapshot).
    pub fn kill_cooldown_ticks(&self) -> i32 {
        if self.last_tick < 0 {
            return 0;
        }
        (KILL_COOLDOWN_TICKS - (self.last_tick - self.unstick.last_kill_tick())).clamp(0, KILL_COOLDOWN_TICKS)
    }

    /// The server's system line "Kill Protection enabled ..." (read by the runner; nothing else of the chat is looked at): this
    /// life's `Cl_Kill`s are dropped, so a decision to kill sends `/kill` at once (D-078).
    pub fn on_kill_protection_notice(&mut self) {
        self.killfb.on_notice();
    }

    /// A line the owner typed on the website and that starts with `/` (a server command) was just sent (task 4.9b). It is the owner's,
    /// not the fallback's (D-078): whatever it does to our life (`/kill`, `/spec`, `/team`) is not the bot's `/kill` taking effect, so
    /// the fallback forgets its in-flight `/kill` as a candidate for the learned threshold.
    pub fn on_owner_command(&mut self) {
        self.killfb.on_owner_command();
    }

    /// The `/kill` fallback's step of one snapshot: a new life or a dead tee settles an awaited kill, a protocol kill of this
    /// snapshot is awaited, and [`Output::kill_command`] is set when the fallback is due.
    fn kill_fallback_step(&mut self, tick: i32, out: &mut Output) {
        if self.paused {
            // Paused by the server (task 4.9b): no `Cl_Kill` is decided, so nothing is awaited and no `/kill` follows.
            self.killfb.cancel_pending();
            return;
        }
        if self.lives != self.fb_lives {
            self.fb_lives = self.lives;
            if let Some(l) = self.killfb.on_life_started(tick) {
                push_event(
                    &mut self.events,
                    BotEvent::KillProtectionLearned {
                        tick,
                        life_secs: (l.life_minutes * 60.0).round().max(0.0) as u32,
                    },
                );
            }
        } else if !self.was_alive {
            self.killfb.on_dead(tick);
        }
        // D-102: under the duel switch only the owner's console `!kill` can be an awaited kill (every kill the bot decides itself is
        // suppressed); the life tracking above always runs.
        if out.kill && (!self.no_selfkill || matches!(self.kill_why, Some(KillWhy::Console))) {
            self.killfb.on_protocol_kill(tick);
        }
        let was_giving_up = self.killfb.gave_up();
        if self.killfb.poll(tick) {
            out.kill_command = true;
            let noticed = self.killfb.noticed();
            push_event(&mut self.events, BotEvent::KillFallback { tick, noticed });
        } else if !was_giving_up && self.killfb.gave_up() {
            push_event(&mut self.events, BotEvent::KillFallbackGaveUp { tick });
        }
    }

    /// `SessionEvent::InputSent`.
    pub fn on_input_sent(&mut self, tick: i32, input: &NetInput) {
        self.sent.on_sent(tick, input);
    }

    /// `SessionEvent::InputTiming`.
    pub fn on_input_timing(&mut self, tick: i32, time_left_ms: i32) {
        self.sent.on_timing(tick, time_left_ms);
    }

    /// The run is over: the hooks save what they keep (the freeze memory).
    pub fn shutdown(&mut self) {
        self.hooks.navigator.stop();
        self.clipper.finish();
        self.report_saved_clips();
    }

    /// `SV_KILLMSG` (`onKill`, `bot.ts:2120`).
    ///
    /// Our own death: DDNet respawns a player right after `Cl_Kill` (`gamecontext.cpp:3008-3009`,
    /// `player.cpp:266`), so no snapshot without our tee need ever arrive. The message is therefore
    /// what ends the life (`bot.ts:2131` sets `wasAlive = false` here too): the death is counted once
    /// and the next snapshot with our tee starts a new life (brain reset, `Respawned`).
    pub fn on_kill_message(&mut self, killer: i32, victim: i32, weapon: i32) {
        self.clipper.push_event(ClipEvent::Kill { killer, victim, weapon });
        self.clock.on_kill(victim, self.last_tick);
        self.hooks
            .navigator
            .on_kill(victim, self.players.own_id().unwrap_or(-2), self.last_tick);
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
    pub fn note_wire_latency(
        &mut self,
        d: Duration,
        tick: i32,
        tag: Option<ddai_client::InputTag>,
        handed_after: Duration,
        pickup: Duration,
    ) {
        self.latency.wire.push(d);
        self.latency.handed.push(handed_after);
        self.latency.pickup.push(pickup);
        if let Some(t) = tag {
            self.latency.slots.note(tick, t.first_slot, t.expected_tick, t.brain);
        }
    }

    // ---- the pipeline --------------------------------------------------------------------------

    /// Decides one snapshot. `Output::default()` (no input) until a map is loaded and our id known.
    pub fn on_snapshot(&mut self, snap: &LiveWorldSnapshot) -> Output {
        let started = Instant::now();
        let real_queue_delay = started.saturating_duration_since(snap.arrived);
        self.latency.queue.push(real_queue_delay);
        self.queue_delay = if self.cfg.decision_time_override.is_some() || self.cfg.decision_time_feed.is_some() {
            Duration::ZERO
        } else {
            real_queue_delay
        };
        self.stats.snapshots += 1;
        let mut brain_time = Duration::ZERO;
        let before = self.stats;
        let mut out = self.decide(snap, &mut brain_time);
        let t_finish = Instant::now();
        self.viz_fresh = self.stats.brain_decisions > before.brain_decisions;
        self.apply_pending(snap, &mut out);
        self.kill_fallback_step(snap.tick, &mut out);
        if out.kill {
            let why = self.kill_why.take().unwrap_or(KillWhy::Unstick);
            self.clipper.push_event(ClipEvent::KillSent { why: why as u8 });
        }
        self.kill_why = None;
        let t_clip = Instant::now();
        self.latency.finish.push(t_clip.saturating_duration_since(t_finish));
        self.record_clip(snap, &out, &before, started.elapsed(), brain_time);
        self.latency.clip.push(t_clip.elapsed());
        if out.input.is_some() {
            self.stats.decisions += 1;
            let total = started.elapsed();
            self.latency.record(total, brain_time);
            if self.stats.brain_decisions > before.brain_decisions
                && let Some(p) = self.brain.last_plan()
            {
                self.latency.record_plan(&p, brain_time);
            }
            // A rolling high quantile, not a mean (task 4.1b): the driver holds the decision until
            // the tick it was aimed at, so a conservative estimate only costs latency.
            let brain_decided = self.stats.brain_decisions > before.brain_decisions;
            let fed = match self.cfg.decision_time_feed {
                Some((other, brain)) => {
                    if brain_decided {
                        brain
                    } else {
                        other
                    }
                }
                None => total,
            };
            self.estimator.push(fed);
            self.est_decision = self
                .cfg
                .decision_time_override
                .unwrap_or_else(|| self.estimator.estimate());
            if brain_decided {
                self.estimator_brain.push(fed);
                self.est_brain = self
                    .cfg
                    .decision_time_override
                    .unwrap_or_else(|| self.estimator_brain.estimate());
            }
        }
        out
    }

    /// One frame of the clip ring for this snapshot (task 4.3), after the decision.
    fn record_clip(
        &mut self,
        snap: &LiveWorldSnapshot,
        out: &Output,
        before: &BotStats,
        total: Duration,
        brain_time: Duration,
    ) {
        let (Some(_), Some(own_id)) = (&self.map, snap.own_id) else {
            return;
        };
        if self.live.is_none() {
            return;
        }
        let nav = self.hooks.navigator.clip_state(&mut self.walk_label);
        let s = &self.stats;
        let mut flags = 0u16;
        let mut candidates = 0;
        if s.brain_decisions > before.brain_decisions {
            flags |= BotRec::BIT_BRAIN_DECIDED;
            if let Some(p) = self.brain.last_plan() {
                flags |= if p.searched { BotRec::BIT_SEARCHED } else { 0 }
                    | if p.out_of_time { BotRec::BIT_OUT_OF_TIME } else { 0 }
                    | if p.shielded { BotRec::BIT_SHIELDED } else { 0 }
                    | if p.shield_incomplete {
                        BotRec::BIT_SHIELD_INCOMPLETE
                    } else {
                        0
                    };
                candidates = p.candidates;
            }
        }
        for (now, was, bit) in [
            (s.wander_decisions, before.wander_decisions, BotRec::BIT_WANDER),
            (s.guarded_inputs, before.guarded_inputs, BotRec::BIT_GUARDED),
            (s.vetoed_hooks, before.vetoed_hooks, BotRec::BIT_VETOED_HOOK),
            (s.vetoed_fires, before.vetoed_fires, BotRec::BIT_VETOED_FIRE),
        ] {
            if now > was {
                flags |= bit;
            }
        }
        if nav.crossing {
            flags |= BotRec::BIT_CROSSING;
        }
        if nav.planned_freeze {
            flags |= BotRec::BIT_PLANNED_FREEZE;
        }
        if self.hooks.wayblock.holding() {
            flags |= BotRec::BIT_WB_HOLDING;
        }
        let blocks = self.clock.stats();
        let bot = BotRec {
            target: self.picker.target(),
            brain: BrainKind::ALL
                .iter()
                .position(|k| *k == self.cfg.brain)
                .map_or(255, |i| i as u8),
            flags,
            walk: 0,
            total_us: u32::try_from(total.as_micros()).unwrap_or(u32::MAX),
            brain_us: u32::try_from(brain_time.as_micros()).unwrap_or(u32::MAX),
            candidates,
            aimed_tick: out.tag.map_or(0, |t| t.expected_tick),
            blocks: u16::try_from(blocks.blocks).unwrap_or(u16::MAX),
            blocked_by: u16::try_from(blocks.blocked_by).unwrap_or(u16::MAX),
        };
        self.clipper.record(&FrameInput {
            snap,
            tees: &self.tees,
            players: &self.players,
            sent: &self.sent,
            own_id,
            bot,
            walk_label: &self.walk_label,
        });
        if let Some(note) = self.hooks.navigator.take_cross_fail() {
            self.clipper.cross_fail(&note, own_id, &self.players);
        }
        self.report_saved_clips();
    }

    /// Clips the automatic path saved since the last look become [`BotEvent::ClipSaved`]s.
    fn report_saved_clips(&mut self) {
        for c in self.clipper.take_saved() {
            self.stats.clips_saved += 1;
            push_event(
                &mut self.events,
                BotEvent::ClipSaved {
                    tick: c.tick,
                    kind: c.kind,
                    severity: c.severity,
                    path: c.path.display().to_string(),
                },
            );
        }
    }

    fn decide(&mut self, snap: &LiveWorldSnapshot, brain_time: &mut Duration) -> Output {
        let t_decide = Instant::now();
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
            last_clamp_event,
            last_sent,
            queue_delay,
            est_decision,
            est_brain,
            in_flight,
            keep,
            spares,
            spare_ids,
            spare_tees,
            stats,
            events,
            status,
            kill_why,
            wants_spectate,
            join_grace_until,
            paused,
            no_selfkill,
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
                BlockEvent::BlockedBy { by } => {
                    hooks.navigator.blocked_by(by, tick);
                    BotEvent::BlockedBy {
                        tick,
                        by: players.tag(by).to_string(),
                    }
                }
                BlockEvent::Held { victim, died } => BotEvent::BlockHeld {
                    tick,
                    victim: players.tag(victim).to_string(),
                    died,
                },
                BlockEvent::Escaped { victim, after } => BotEvent::BlockEscaped {
                    tick,
                    victim: players.tag(victim).to_string(),
                    after,
                },
            };
            push_event(events, e);
        }

        // 4b. paused by the server (task 4.9b): our own `DDNetPlayer` flag `SPEC` or `PAUSED` (the owner's `/pause` or `/spec`; to a
        // modern client the server keeps the team, so this is not the move to the spectators of D-058, which is team -1 and still stops
        // the bot). Read **before** the tee is looked for: with `sv_pauseable 1`, a practice team or an admin `force_pause` the server
        // removes a still, grounded tee in the very snapshot that sets the flag. While it is set nothing is decided: the brain is not
        // asked, the input is neutral, no kill is requested (no unstick, no navigation, no `/kill` fallback), no death is counted and no
        // respawn reported (the tee that is gone is paused, not dead), and no timer of those advances (they are reset on resume).
        let server_pause = players
            .get(own_id)
            .is_some_and(|s| s.team != -1 && s.ex_flags & (explayerflagflag::SPEC | explayerflagflag::PAUSED) != 0);
        if server_pause != *paused {
            *paused = server_pause;
            if server_pause {
                push_event(events, BotEvent::PausedByServer { tick });
            } else {
                push_event(events, BotEvent::ResumedByServer { tick });
                // Back in play: tick-based memory of the time before the pause is meaningless (the frozen clock, the stuck anchor, the
                // kill cooldown), the walk and the target are stale, and the last input was never acted on.
                unstick.reset_ticks();
                hooks.navigator.respawned();
                wander.respawned();
                picker.set_target(-1);
                encoder.reset_edges();
                *last_sent = PhysInput::default();
            }
        }
        if *paused {
            let own = tees.get(own_id).copied();
            if own.is_some() {
                *played_on_map = true;
                *join_grace_until = None;
            }
            stats.idle_decisions += 1;
            out.input = Some(encoder.idle());
            *status = make_status(
                tick,
                own_id,
                *mode,
                own.as_ref(),
                picker.target(),
                clock.stats(),
                *stats,
            );
            return out;
        }

        // 5. dead / absent / spectating.
        let t_nav = Instant::now();
        self.latency.update.push(t_nav.saturating_duration_since(t_decide));
        let Some(own) = tees.get(own_id).copied() else {
            if *was_alive {
                stats.deaths += 1;
            }
            *was_alive = false;
            let input = encoder.idle();
            stats.idle_decisions += 1;
            out.input = Some(input);
            if players.get(own_id).is_some_and(|s| s.team != -1) {
                *join_grace_until = None; // the server has us in the game again (dead or waiting to spawn)
            }
            if *wants_spectate {
                // `!spec`: the operator put us there and means it (no auto-join, no moderation stop).
            } else if join_grace_until.is_some_and(|until| tick <= until) {
                // `!join` was typed and the server has not applied it yet: keep asking, do not stop.
                out.set_team = join_request(cfg, players, join, events, tick, own_id);
            } else if *played_on_map && players.get(own_id).is_some_and(|s| s.team == -1) {
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
        *join_grace_until = None; // a tee: the join went through
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

        // 6b. the navigator's housekeeping: commands, the end of a walk, the freeze memory.
        let lag_ticks = (snap.pred_tick.max(snap.tick) + 1 - tick).max(0);
        macro_rules! hook_ctx {
            () => {
                HookContext {
                    tick,
                    own: &own,
                    tees,
                    players,
                    grid,
                    clock,
                    world: live.base_world(),
                    lag_ticks,
                    mode: *mode,
                    fixed_target: picker.fixed().is_some(),
                }
            };
        }
        let poll = hooks.navigator.poll(&hook_ctx!());
        if let Some(m) = poll.mode
            && m != *mode
        {
            *mode = m;
            if !matches!(m, Mode::Fight | Mode::Goto) {
                picker.set_target(-1);
            }
        }
        if let Some(k) = poll.knowledge {
            brain.set_map_knowledge(&k);
        }

        // 7. unstick.
        let acting = *mode != Mode::Hold;
        let wb_kill =
            hooks.wayblock.holding() && hooks.wayblock.wants_kill(&hook_ctx!(), unstick.frozen_for(tick, &own));
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
            *kill_why = Some(match reason {
                KillReason::WayBlockLying => KillWhy::WayBlockLying,
                KillReason::Overdue | KillReason::Stuck => KillWhy::Unstick,
            });
            stats.self_kills += 1;
            hooks.navigator.kill_sent(tick, false);
            push_event(events, BotEvent::Killed { tick, reason });
        }

        // 8. mode.
        if !acting {
            stats.idle_decisions += 1;
            out.input = Some(encoder.idle());
            *status = make_status(tick, own_id, *mode, Some(&own), picker.target(), clock.stats(), *stats);
            return out;
        }
        let nav = hooks.navigator.drive(&hook_ctx!());
        let mut navigated = None;
        let mut nav_guard = false;
        match nav {
            Some(NavStep::Kill { action }) => {
                if !*no_selfkill && !out.kill && unstick.cooldown_ready(tick) {
                    unstick.note_external_kill(tick);
                    hooks.navigator.kill_sent(tick, true);
                    stats.self_kills += 1;
                    out.kill = true;
                    *kill_why = Some(KillWhy::Navigation);
                }
                navigated = Some(action);
            }
            Some(NavStep::Input { action, guard }) => {
                navigated = Some(action);
                nav_guard = guard;
            }
            None => {}
        }
        let previous_target = picker.target();
        let t_pick = Instant::now();
        self.latency.nav.push(t_pick.saturating_duration_since(t_nav));
        // The foe of the wayblock walk (a player at the tube who acts against us, or one who froze us on the way)
        // is the target until the walk is back on.
        let foe = if navigated.is_none() && *mode == Mode::Fight && picker.fixed().is_none() {
            hooks.wayblock.foe_target().filter(|&id| tees.get(id).is_some())
        } else {
            None
        };
        let target = if navigated.is_some() || !matches!(*mode, Mode::Fight | Mode::Goto) {
            -1
        } else if let Some(id) = foe {
            id
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
                    lag_ticks,
                    mode: *mode,
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

        // 8b. the walk-to-the-game / seek / home rules (nothing is navigating).
        if navigated.is_none() && matches!(*mode, Mode::Fight | Mode::Goto) {
            hooks.trek.steer(&hook_ctx!(), target);
        }

        // 9. the action.
        let mut action;
        let mut expected_tick: Option<i32> = None;
        let own_shield = cfg.brain.has_own_shield();
        if let Some(a) = navigated {
            action = a;
            stats.idle_decisions += 1;
            if nav_guard && grid.hazard_within(own.pos.x, own.pos.y, GUARD_HAZARD_TILES) {
                // `driveNav`: `crossing || plannedFreeze ? want : guard(self, want)`.
                let t_guard = Instant::now();
                let prev_planner = from_ddnet_input(last_sent);
                // As the TS `guard` and the brain path do: on the world our input will act in (our own
                // in-flight inputs applied up to the tick it takes effect), not the snapshot's.
                let ready_in = *queue_delay + *est_decision + cfg.driver_pickup;
                let predicted = predict_own(live, snap, cfg, ready_in, sent, in_flight, keep, own_id, tick);
                let g = plan.guard(predicted, own_id, action, &prev_planner);
                if g != action {
                    stats.guarded_inputs += 1;
                    hooks.navigator.vetoed();
                }
                action = g;
                *brain_time += t_guard.elapsed();
            }
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
                let hint = if hooks.wayblock.holding() {
                    hooks.wayblock.wander_hint(&hook_ctx!())
                } else {
                    None
                };
                // The guard only runs near freeze or death; there it checks the world the input will act
                // in (own in-flight inputs applied), like the brain path; elsewhere nothing is predicted.
                let guard_world = if grid.hazard_within(own.pos.x, own.pos.y, GUARD_HAZARD_TILES) {
                    let ready_in = *queue_delay + *est_decision + cfg.driver_pickup;
                    predict_own(live, snap, cfg, ready_in, sent, in_flight, keep, own_id, tick)
                } else {
                    live.base_world()
                };
                let mut env = WanderEnvImpl {
                    plan,
                    world: guard_world,
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
                        anchor_x: hint.map(|h| h.anchor_x),
                        look_at: hint.and_then(|h| h.look_at).map(|(x, y)| Vec2 { x, y }),
                        still: hint.is_some_and(|h| h.still),
                        lag_ticks,
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
            let t_predict = Instant::now();
            let target_tee = tees.get(target).copied();
            compute_keep(&own, target, tees, cfg, &|t| is_spared(t, tick, players, clock), keep);
            let prediction = prediction_target(
                snap,
                cfg.max_predict_ticks,
                *queue_delay + if cfg.kind_estimate { *est_brain } else { *est_decision } + cfg.driver_pickup,
            );
            let to_tick = prediction.to_tick;
            self.latency.horizon.push(Duration::from_micros(
                u64::try_from((to_tick - tick).max(0)).unwrap_or(0),
            ));
            if prediction.clamped() {
                // The horizon wanted is beyond the cap: the decision is made on a world that stops short
                // of the tick it will take effect on (a very long RTT). Counted always, said at most
                // once per 10 s.
                stats.predict_clamped += 1;
                if tick - *last_clamp_event >= PREDICT_CLAMP_EVENT_EVERY_TICKS || tick < *last_clamp_event {
                    *last_clamp_event = tick;
                    push_event(
                        events,
                        BotEvent::PredictionClamped {
                            tick,
                            wanted_ahead: prediction.wanted_ahead,
                            cap: prediction.cap,
                        },
                    );
                }
            }
            expected_tick = Some(to_tick + 1);
            sent.in_flight(tick, to_tick, in_flight);
            // The observation's target is a tee that is still in the world (it may have just died).
            let target_id = target_tee.map(|t| t.id);

            spares.clear();
            spare_ids.clear();
            spare_tees.clear();
            for t in tees.iter() {
                if t.id != own.id
                    && t.id != target
                    && dist(t.pos, own.pos) <= HOOK_LENGTH_PX + 64.0
                    && is_spared(t, tick, players, clock)
                {
                    spares.push((t.pos, t.vel));
                    spare_ids.push(t.id);
                    spare_tees.push(*t);
                }
            }
            let travel_goal = match target_tee.as_ref() {
                Some(tt) => hooks.trek.goal(&hook_ctx!(), tt),
                None => None,
            };
            let trek_kill = hooks.trek.take_kill();
            if trek_kill && !*no_selfkill && !out.kill && unstick.cooldown_ready(tick) {
                // `trekGoal`: a respawn step on the way there.
                unstick.note_external_kill(tick);
                hooks.navigator.kill_sent(tick, false);
                stats.self_kills += 1;
                out.kill = true;
                *kill_why = Some(KillWhy::Trek);
            }
            let wb = hooks.wayblock.brain_hints(&hook_ctx!());
            brain.set_live_context(&LiveContext {
                spares: spares.as_slice(),
                spare_ids: spare_ids.as_slice(),
                travel_goal,
                wb,
            });
            let predicted = live.predict_local_observation(to_tick, in_flight, keep, target_id, obs);
            let view = WorldView {
                world: predicted,
                self_id: own_id,
                lag_ticks: 0,
                in_flight: &[],
            };
            let t0 = Instant::now();
            self.latency.predict.push(t0.saturating_duration_since(t_predict));
            action = brain.decide_in(obs, Some(&view));
            *brain_time = t0.elapsed();
            let t_post = Instant::now();
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
            self.latency.post.push(t_post.elapsed());
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
            brain: expected_tick.is_some(),
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
        self.clipper.reset();
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
/// (one poll round, `POLL_TIMEOUT`): the default of [`BotConfig::driver_pickup`].
pub const DRIVER_PICKUP: Duration = Duration::from_millis(2);
/// The same with the driver's precise wake-ups (`ClientConfig::precise_wakeups`, task 3.11): a futex wake-up and the send.
pub const DRIVER_PICKUP_PRECISE: Duration = Duration::from_micros(300);
/// What a brain decision is assumed to take before one has been measured (the hybrid's 5 ms cap and the work around it).
const BRAIN_ESTIMATE_INITIAL: Duration = Duration::from_millis(6);
/// `PredictionClamped` is reported at most this often (10 s of game ticks).
const PREDICT_CLAMP_EVENT_EVERY_TICKS: i32 = 500;
/// One server tick.
const TICK: Duration = Duration::from_millis(20);

/// One snapshot period (25 Hz).
const SNAPSHOT_PERIOD: Duration = Duration::from_millis(40);

/// Never predict farther than this many ticks past the snapshot, whatever the RTT (a 1.5 s horizon;
/// `LiveWorld` itself stops at 3 s).
pub const MAX_PREDICT_TICKS_ABSOLUTE: i32 = 75;

/// What [`prediction_target`] decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Prediction {
    /// The tick the world is predicted to.
    to_tick: i32,
    /// The horizon the decision wanted (`to_tick` before the cap) minus the snapshot tick.
    wanted_ahead: i32,
    /// The cap on the horizon in force, in ticks.
    cap: i32,
}

impl Prediction {
    fn clamped(&self) -> bool {
        self.wanted_ahead > self.cap
    }
}

/// The tick the prediction must reach (see the module docs): the predicted tick of the last input the
/// driver sent — plus one more for every input that will go out *before* this decision is ready. The
/// driver's next `NETMSG_INPUT` is due `next_input_in` after the snapshot arrived; a decision that will
/// be ready (`ready_in` after arrival: channel hop + estimated decision time + driver pickup) only
/// after that goes out with the tick after, so the world must be predicted one tick further, and so on.
/// Before the driver's two-snapshot bootstrap (`pred_tick == 0`) a 2-tick guess.
///
/// **`ready_in` is clamped below one snapshot period** (review F2a): a decision slower than that is
/// replaced by the next snapshot's before its tick comes anyway, and an estimate inflated by a stall
/// must not aim a tag farther ahead than the driver's `MAX_HOLD_TICKS` (the one way a decision could go
/// out early). With it the tag is at most two ticks past the next input.
///
/// **The horizon cap is RTT-aware** (review residual): `max_ahead` (12 ticks) is the floor, but when
/// the driver's `pred_tick` is already farther ahead of the snapshot (a long RTT: `pred_tick - tick`
/// is about the RTT in ticks plus the margin) the cap follows it plus the decision's own slots, up to
/// [`MAX_PREDICT_TICKS_ABSOLUTE`], so a 200+ ms RTT no longer under-predicts.
fn prediction_target(snap: &LiveWorldSnapshot, max_ahead: i32, ready_in: Duration) -> Prediction {
    let ready_in = ready_in.min(SNAPSHOT_PERIOD - Duration::from_millis(1));
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
    let pred_ahead = if snap.pred_tick > 0 {
        snap.pred_tick - snap.tick
    } else {
        0
    };
    // The floor, or the driver's own lead plus the most a decision can miss (2 inputs) and one tick.
    let cap = max_ahead
        .max(pred_ahead + 3)
        .min(MAX_PREDICT_TICKS_ABSOLUTE.max(max_ahead));
    Prediction {
        to_tick: want.clamp(snap.tick, snap.tick + cap),
        wanted_ahead: want - snap.tick,
        cap,
    }
}

/// Fills `keep` with the tees the brain gets: the target, everyone roped to us, and the nearest
/// others within the threat radius up to `max_local_others`.
///
/// **Spared tees** (friends, ignored, out of game, AFK) are counted **separately** (task 4.1b, review
/// F8; round 1's F1 had left them out entirely): the nearest [`MAX_SPARE_BODIES`] of them within
/// [`SPARE_BODY_RANGE_PX`] are kept in the world as *physical bodies*, so the prediction simulates
/// bumping into them, pushing them and being pulled by a rope that lands on them (collision and
/// hooking other players are on by default). They do not use up `max_local_others` slots, so a crowd
/// of friends cannot push opponents out (and the 8+-tee search collapse cannot come back: at most
/// `1 + 5 + 3` other tees); a spared tee roped to us is always kept and counts as roped. The brain
/// is told who they are through `LiveContext::spare_ids` and must not treat them as opponents.
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

    // The spared bodies: the nearest few within contact range, by (distance, id) — for brains that
    // honour `spare_ids` only (`BrainKind::honours_spare_ids`).
    if !cfg.brain.honours_spare_ids() {
        return;
    }
    // The directions we are likely to travel: our velocity, and toward the target (the plan's goal).
    let mut lanes: [Option<(f32, f32)>; 2] = [None, None];
    let speed = own.vel.x.hypot(own.vel.y);
    if speed >= SPARE_BODY_MIN_SPEED {
        lanes[0] = Some((own.vel.x / speed, own.vel.y / speed));
    }
    if let Some(tt) = tees.get(target) {
        let (dx, dy) = (tt.pos.x - own.pos.x, tt.pos.y - own.pos.y);
        let len = dx.hypot(dy);
        if len > 1.0 {
            lanes[1] = Some((dx / len, dy / len));
        }
    }
    let mut bodies: [(f32, i32); MAX_SPARE_BODIES] = [(f32::INFINITY, -1); MAX_SPARE_BODIES];
    for t in tees.iter() {
        if t.id == own.id || t.id == target || keep[t.id as usize] || !spared(t) {
            continue;
        }
        let d = dist(own.pos, t.pos);
        let in_lane = d <= SPARE_BODY_AHEAD_PX && ahead_of_us(own, t, lanes);
        if d > SPARE_BODY_RANGE_PX && !in_lane {
            continue;
        }
        // Rank: tees in our lane first (the ones we will run into), then by distance — a tee straight
        // ahead must not lose its body to side tees nearer in raw distance (review round 2, F8).
        let rank = if in_lane { d } else { d + LANE_PRIORITY_OFFSET_PX };
        // Insert into the sorted fixed array when it beats the worst entry.
        let last = MAX_SPARE_BODIES - 1;
        if (rank, t.id) < bodies[last] {
            bodies[last] = (rank, t.id);
            bodies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    for &(_, id) in bodies.iter().filter(|b| b.1 >= 0) {
        keep[id as usize] = true;
    }
}

/// Added to the distance of a spared tee outside our lane when ranking body candidates (more than any
/// distance that can qualify).
const LANE_PRIORITY_OFFSET_PX: f32 = 1000.0;

/// Whether `t` lies in a lane `SPARE_BODY_LANE_PX` wide ahead of us along one of `lanes` (unit vectors).
fn ahead_of_us(own: &Tee, t: &Tee, lanes: [Option<(f32, f32)>; 2]) -> bool {
    let (rx, ry) = (t.pos.x - own.pos.x, t.pos.y - own.pos.y);
    lanes.into_iter().flatten().any(|(dx, dy)| {
        let along = rx * dx + ry * dy;
        let lateral = (rx * dy - ry * dx).abs();
        along > 0.0 && lateral <= SPARE_BODY_LANE_PX
    })
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
/// effective weapon is the hammer (F9, below) and the swing would hit a spared tee. DDNet's `fire_hammer` geometry
/// (as derived in `ddai-planner`'s physics adapter and its `hammer_would_hit`): everybody within
/// `1.5 * 28` px of the point `0.75 * 28 = 21` px in front of us along the aim is hit. The input
/// takes effect `ahead` ticks after the snapshot, so the swing is checked from our predicted
/// position (`own_then`) and from where we are now, against each spared tee now, `ahead` ticks on
/// and two more ticks on (as `hammer_would_hit` does): conservative on purpose. Walls are ignored
/// (a miss through a wall is not worth the risk).
fn hammer_veto(action: &mut Action, own: &Tee, own_then: Vec2<f32>, ahead: f32, spared: &[Tee]) -> bool {
    // The *effective* weapon (review F9): DDNet spawns a tee with the gun active and `FireWeapon`
    // switches to the wanted weapon before it fires, so in the 1-3 ticks after a spawn a press that
    // asks for the hammer swings it while the snapshot still says "gun". The encoder always asks for
    // the hammer unless the action names another weapon.
    let swings_hammer = action.wanted_weapon.unwrap_or(WEAPON_HAMMER) == WEAPON_HAMMER || own.holding_hammer();
    if !action.fire || !swings_hammer || spared.is_empty() {
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

/// Our own tee alone, predicted to the tick an input decided now takes effect, with our inputs already
/// sent but not yet applied (`in_flight`) in the world: what `guard` needs for a navigator or wander
/// step (the brain path builds the same world with the others kept as well).
#[allow(clippy::too_many_arguments)]
fn predict_own<'a>(
    live: &'a mut LiveWorld,
    snap: &LiveWorldSnapshot,
    cfg: &BotConfig,
    ready_in: Duration,
    sent: &SentLog,
    in_flight: &mut Vec<(i32, PhysInput)>,
    keep: &mut [bool; MAX_CLIENTS],
    own_id: i32,
    tick: i32,
) -> &'a ddai_physics::world::World<f32> {
    let prediction = prediction_target(snap, cfg.max_predict_ticks, ready_in);
    sent.in_flight(tick, prediction.to_tick, in_flight);
    keep.fill(false);
    if let Some(k) = usize::try_from(own_id).ok().and_then(|i| keep.get_mut(i)) {
        *k = true;
    }
    live.predict_local(prediction.to_tick, in_flight, keep)
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

    /// Review F9: the 1-3 ticks after a spawn the snapshot still says "gun", but the press asks for the
    /// hammer (the default) and `FireWeapon` switches before it fires.
    #[test]
    fn the_hammer_veto_tests_the_effective_weapon_not_the_one_the_snapshot_reports() {
        let mut me = tee_at(0, 1000.0, 0.0);
        me.weapon = 1; // just spawned: the gun is active
        let friend = [tee_at(1, 1030.0, 0.0)];
        let mut a = swing_right(); // wanted_weapon: None = the hammer
        assert!(hammer_veto(&mut a, &me, me.pos, 0.0, &friend), "the swing is withheld");
        assert!(!a.fire);
        let mut a = Action {
            wanted_weapon: Some(WEAPON_HAMMER),
            ..swing_right()
        };
        assert!(hammer_veto(&mut a, &me, me.pos, 0.0, &friend));
        // The hammer in hand and no other weapon asked for: also a swing.
        me.weapon = WEAPON_HAMMER;
        let mut a = swing_right();
        assert!(hammer_veto(&mut a, &me, me.pos, 0.0, &friend));
    }

    #[test]
    fn the_hammer_veto_leaves_other_weapons_non_fire_and_zero_aim_alone() {
        let mut me = tee_at(0, 1000.0, 0.0);
        let friend = [tee_at(1, 1030.0, 0.0)];
        me.weapon = 1; // the gun in hand, and the action asks for the gun too: a bullet, no swing
        let mut a = Action {
            wanted_weapon: Some(1),
            ..swing_right()
        };
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

    fn snap(tick: i32, pred_tick: i32, next_input_in_ms: Option<u64>) -> LiveWorldSnapshot {
        LiveWorldSnapshot {
            tick,
            own_id: Some(0),
            characters: Vec::new(),
            tuning: ddai_net::tuning::DEFAULT_TUNE_PARAMS,
            switch_states: Vec::new(),
            teams: None,
            projectiles: Vec::new(),
            players: Vec::new(),
            pred_tick,
            next_input_in: next_input_in_ms.map(Duration::from_millis),
            arrived: Instant::now(),
        }
    }

    /// Review F2a: however slow the estimate, the tag stays within the driver's `MAX_HOLD_TICKS` of its
    /// next input (so it is never adopted early), and a stall-inflated estimate is clamped below one
    /// snapshot period.
    #[test]
    fn the_tag_distance_is_bounded_by_the_drivers_hold_limit_whatever_the_estimate() {
        for first_ms in [0u64, 1, 5, 10, 19] {
            for ready_ms in [0u64, 3, 12, 25, 39, 40, 100, 500, 60_000] {
                let sn = snap(1000, 1005, Some(first_ms));
                let p = prediction_target(&sn, 12, Duration::from_millis(ready_ms));
                // The driver's next send is for pred_tick + 1 (at the earliest); the tag is to_tick + 1.
                let distance = (p.to_tick + 1) - (sn.pred_tick + 1);
                assert!(
                    (0..=ddai_client::MAX_HOLD_TICKS).contains(&distance),
                    "first {first_ms} ms, ready {ready_ms} ms: distance {distance}"
                );
                assert!(distance <= 2, "with the estimate below one snapshot period: {distance}");
            }
        }
    }

    #[test]
    fn an_enormous_ready_estimate_is_clamped_below_one_snapshot_period() {
        let sn = snap(1000, 1005, Some(5));
        let huge = prediction_target(&sn, 12, Duration::from_secs(10));
        let at_period = prediction_target(&sn, 12, SNAPSHOT_PERIOD - Duration::from_millis(1));
        assert_eq!(huge, at_period);
    }

    /// Residual: the horizon cap follows the driver's lead (a long RTT) instead of under-predicting.
    #[test]
    fn the_horizon_cap_follows_a_long_rtt_and_says_so_only_when_it_binds() {
        // Normal link: pred_tick 5 ahead, cap 12: not clamped.
        let p = prediction_target(&snap(1000, 1005, Some(15)), 12, Duration::from_millis(8));
        assert_eq!((p.to_tick, p.clamped(), p.cap), (1005, false, 12));
        // 400 ms RTT: the driver is 25 ticks ahead; the old fixed cap of 12 would have stopped at 1012.
        let p = prediction_target(&snap(1000, 1025, Some(15)), 12, Duration::from_millis(8));
        assert_eq!(p.to_tick, 1025, "predicted to the driver's tick");
        assert!(!p.clamped());
        assert_eq!(p.cap, 28);
        // A decision that misses the next input is one tick farther, still within the RTT-aware cap.
        let p = prediction_target(&snap(1000, 1025, Some(1)), 12, Duration::from_millis(8));
        assert_eq!(p.to_tick, 1026);
        // An absurd RTT hits the absolute cap and is reported.
        let p = prediction_target(&snap(1000, 1200, Some(15)), 12, Duration::from_millis(8));
        assert!(p.clamped());
        assert_eq!(p.to_tick, 1000 + MAX_PREDICT_TICKS_ABSOLUTE);
        // Before the bootstrap: the 2-tick guess.
        let p = prediction_target(&snap(1000, 0, None), 12, Duration::from_millis(8));
        assert_eq!(p.to_tick, 1002);
    }

    fn keep_with(own: Tee, target: Tee, others: &[Tee], spared: &[i32]) -> Vec<i32> {
        let mut tees = TeeSet::new();
        tees.set_for_test(own);
        tees.set_for_test(target);
        for t in others {
            tees.set_for_test(*t);
        }
        let cfg = BotConfig::default(); // Planner honours spare_ids
        let mut keep = [false; MAX_CLIENTS];
        let spared = spared.to_vec();
        compute_keep(&own, target.id, &tees, &cfg, &|t| spared.contains(&t.id), &mut keep);
        (0..MAX_CLIENTS as i32)
            .filter(|&i| keep[i as usize] && i != target.id)
            .collect()
    }

    /// Review F3: a spared tee far ahead on our path is a body (the plan's rollouts reach 380 px), one
    /// off the lane or behind us is not, and the cap of 3 holds.
    #[test]
    fn spared_tees_ahead_along_our_velocity_or_toward_the_target_are_bodies_up_to_380_px() {
        let mut me = tee_at(0, 1000.0, 8.0); // running right
        me.pos.y = 500.0;
        let target = tee_at(1, 1900.0, 0.0);
        let at = |id, dx: f32, dy: f32| {
            let mut t = tee_at(id, 1000.0 + dx, 0.0);
            t.pos.y = 500.0 + dy;
            t
        };
        // 300 px ahead on the lane (velocity and target both to the right): kept.
        assert_eq!(keep_with(me, target, &[at(2, 300.0, 10.0)], &[2]), vec![2]);
        // 300 px behind: not. 300 px ahead but 150 px off the lane: not. 450 px ahead: beyond the reach.
        assert!(keep_with(me, target, &[at(2, -300.0, 0.0)], &[2]).is_empty());
        assert!(keep_with(me, target, &[at(2, 300.0, 150.0)], &[2]).is_empty());
        assert!(keep_with(me, target, &[at(2, 450.0, 0.0)], &[2]).is_empty());
        // Within 200 px it is kept whatever the direction (contact range).
        assert_eq!(keep_with(me, target, &[at(2, -150.0, 0.0)], &[2]), vec![2]);
        // Standing still, only the target's direction defines the lane: still toward the target.
        let still = tee_at(0, 1000.0, 0.0);
        let mut still = still;
        still.pos.y = 500.0;
        let mut tgt = tee_at(1, 1000.0, 0.0);
        tgt.pos.y = 1300.0; // straight down
        assert_eq!(
            keep_with(still, tgt, &[at(2, 5.0, 300.0)], &[2]),
            vec![2],
            "toward the target"
        );
        assert!(
            keep_with(still, tgt, &[at(2, 300.0, 0.0)], &[2]).is_empty(),
            "not toward anything"
        );
        // The cap: five spared tees on the lane, three bodies.
        let crowd: Vec<Tee> = (2..7).map(|i| at(i, 60.0 * i as f32, 0.0)).collect();
        let kept = keep_with(me, target, &crowd, &[2, 3, 4, 5, 6]);
        assert_eq!(kept, vec![2, 3, 4], "the nearest three");
    }

    /// Review round 2, F8: a tee straight ahead in our lane keeps its body slot against nearer side tees.
    #[test]
    fn a_spared_tee_in_our_lane_outranks_nearer_side_tees_for_a_body_slot() {
        let mut me = tee_at(0, 1000.0, 8.0);
        me.pos.y = 500.0;
        let target = tee_at(1, 1900.0, 0.0);
        let at = |id, dx: f32, dy: f32| {
            let mut t = tee_at(id, 1000.0 + dx, 0.0);
            t.pos.y = 500.0 + dy;
            t
        };
        // Three side tees 120-150 px to the side (beyond the lane) and one 300 px straight ahead.
        let others = [
            at(2, 0.0, 120.0),
            at(3, 0.0, 135.0),
            at(4, 0.0, 150.0),
            at(5, 300.0, 0.0),
        ];
        let kept = keep_with(me, target, &others, &[2, 3, 4, 5]);
        assert!(kept.contains(&5), "the tee in the lane is a body: {kept:?}");
        assert_eq!(kept.len(), MAX_SPARE_BODIES);
        assert_eq!(
            kept,
            vec![2, 3, 5],
            "and the two nearest side tees take the other slots"
        );
    }
}
