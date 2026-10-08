//! The real-time shell around [`Bot`]: connects with `ddai_client::Client`, turns its events into
//! bot calls, collapses snapshots, sends the bot's output back, feeds the web bridge, and stops the
//! way the rules say.
//!
//! **Snapshot collapsing** (`queueSnapshot`, `orig-bot.md` §4.1): each wake-up drains every queued
//! event; events are handled in order, but of several `LiveWorldSnapshot`s only the **newest** is
//! decided — the skipped ones are counted (`collapsed`). So the bot decides once per batch, at the
//! server's 25 Hz at most, and a slow brain can never build a backlog.
//!
//! **Stopping (D-016/D-037/D-050).** `ClientEvent::GaveUp` ends the run, and its category decides
//! the exit: a kick or ban is exit code 3, a join that never completed / a reconnect loop / attempts
//! exhausted is 4, a requested stop or the duration elapsing is 0. The runner never reconnects by
//! itself and never works around a stop: the driver already refuses to reconnect after a kick or ban,
//! and this shell just reports. One connection per server; the `ddnet-ai play` wrapper additionally
//! holds the single-instance lock.
//!
//! **Chat.** The only chat this runner can send is the typed `/kill` fallback (D-078) and a line the owner typed on the website
//! (`BotCommand::Say` from the control channel, D-094, task 4.9): the latter goes through [`OwnerChat`] (pacing, the queue, in the
//! game only) and `Client::owner_say`; its text is never logged. The audit events (`SessionEvent::OutgoingGame`, enabled by
//! [`RunnerConfig::audit_outgoing`]) are counted per label (`Cl_Say(/kill)` and `Cl_Say(owner)` apart from each other) so a test can
//! assert that no other `Cl_Say` ever reached the wire and that every `Cl_Kill` is the bot's own.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ddai_client::{Client, ClientConfig, ClientEvent, GaveUpCategory, LiveWorldSnapshot, SessionEvent, map_cache};
use ddai_net::generated::messages::GameMsg;
use ddai_net::owner_chat::OwnerSay;
use ddai_physics::map::MapData;

use crate::bot::{Bot, BotConfig, BotEvent, BotStats, Output};
use crate::brains::{BrainError, BrainOptions, make_brain};
use crate::bridge::{
    Bridge, ChatMessage, FrameChar, MapMessage, PlayerEntry, PlayerInfoEntry, PlayerInfoMessage, PlayersMessage,
    StatusMessage,
};
use crate::command::{BotCommand, CommandInbox};
use crate::console::Printer;
use crate::hooks::MapIdent;
use crate::latency::{LatencyStats, Summary};
use crate::nav_hooks::{NavConfig, NavHandle, nav_hooks};
use crate::ownerchat::{self, OwnerChat, OwnerChatStats};
use crate::relations::Relations;
use crate::selfkill::SelfKillSwitch;
use crate::trace::InputTrace;

/// Process exit codes (`ddnet-ai record` uses the same).
pub const EXIT_OK: u8 = 0;
pub const EXIT_ERROR: u8 = 1;
pub const EXIT_KICKED_OR_BANNED: u8 = 3;
pub const EXIT_JOIN_FAILED: u8 = 4;

/// In-game reconnect budget (D-050 item 5, task 4.1): the driver resets all its own budgets after any
/// connection that reached the game, so a server that lets us in and drops us again and again would
/// be an endless cycle bounded only by its 5 connections per 20 s. This caps the drops of **in-game**
/// connections per run: more than `MAX_INGAME_RECONNECTS` within `INGAME_RECONNECT_WINDOW` stops the
/// run (exit code 4) instead of reconnecting once more.
pub const MAX_INGAME_RECONNECTS: usize = 3;
pub const INGAME_RECONNECT_WINDOW: Duration = Duration::from_secs(600);

/// The sliding-window counter behind the in-game reconnect budget.
#[derive(Debug, Default)]
pub struct ReconnectBudget {
    in_game: bool,
    drops: std::collections::VecDeque<Instant>,
}

impl ReconnectBudget {
    /// A connection reached the game.
    pub fn on_in_game(&mut self) {
        self.in_game = true;
    }

    /// The connection dropped at `now`. Returns `true` when the driver may reconnect (this drop is
    /// within the budget) and `false` when the budget is exhausted and the run must stop. A drop
    /// of a connection that never reached the game is the driver's business, not counted here.
    pub fn on_dropped(&mut self, now: Instant) -> bool {
        if !std::mem::take(&mut self.in_game) {
            return true;
        }
        while self
            .drops
            .front()
            .is_some_and(|&t| now.saturating_duration_since(t) > INGAME_RECONNECT_WINDOW)
        {
            self.drops.pop_front();
        }
        self.drops.push_back(now);
        self.drops.len() <= MAX_INGAME_RECONNECTS
    }
}

/// How often the status line goes to the log and the bridge.
const STATUS_EVERY: Duration = Duration::from_millis(200);
const LOG_EVERY: Duration = Duration::from_secs(10);

/// Everything the runner needs.
pub struct RunnerConfig {
    pub server: SocketAddr,
    pub client: ClientConfig,
    pub bot: BotConfig,
    pub brain: BrainOptions,
    pub relations: Relations,
    /// Stop after this long (`None`: run until stopped).
    pub duration: Option<Duration>,
    /// Bridge socket for the web unit; `None` disables it.
    pub bridge_path: Option<PathBuf>,
    /// Put real nicknames on the bridge (`--web-names`); otherwise tags.
    pub web_names: bool,
    /// A **local-only** debug log of `tag name` pairs (`--debug-names`); `None` (the default) writes
    /// no nickname anywhere.
    pub debug_names_log: Option<PathBuf>,
    /// Count every outgoing game message (the e2e chat audit).
    pub audit_outgoing: bool,
    /// Set by a signal handler to ask for a graceful stop.
    pub shutdown: Arc<AtomicBool>,
    /// Navigation, wayblock and freeze memory (task 4.2).
    pub nav: NavConfig,
    /// How commands (`--goto`, `--follow`, task 4.3's chat commands) reach the navigation.
    pub nav_handle: NavHandle,
    /// The console / web commands waiting for the bot (task 4.3); `None`: no commands.
    pub commands: Option<CommandInbox>,
    /// Where the navigation's answers to console commands are printed (`None`: they go to the log).
    pub console_out: Option<Printer>,
}

/// What a finished run reports.
#[derive(Debug, Clone)]
pub struct RunReport {
    pub exit_code: u8,
    pub stats: BotStats,
    pub latency: LatencyStats,
    pub block_stats: crate::activity::BlockStats,
    /// Outgoing game messages by label: `(accepted, refused)`.
    pub outgoing: BTreeMap<String, (u64, u64)>,
    /// `Cl_Kill`s the bot requested, as game ticks.
    pub kill_ticks: Vec<i32>,
    /// The ticks of the snapshots whose decision sent the `/kill` fallback (D-078); each is also in the audit under
    /// `Cl_Say(/kill)`, never under `Cl_Kill`.
    pub kill_command_ticks: Vec<i32>,
    /// What the owner's website chat did (D-094): lines taken, said, refused, dropped. The text is nowhere in the report.
    pub owner_chat: OwnerChatStats,
    pub gave_up: Option<(String, GaveUpCategory)>,
    pub map_name: Option<String>,
    pub events: Vec<BotEvent>,
    /// Wall time of the fresh seal / reachability searches of target selection.
    pub seal_times: crate::latency::Series,
    pub reach_times: crate::latency::Series,
    pub margin: Option<ddai_client::MarginSummary>,
    pub elapsed: Duration,
    pub bridge_clients_dropped: u64,
}

impl RunReport {
    /// `total`/`brain`/`overhead`/`queue`/`wire` summaries as text.
    pub fn latency_text(&self) -> String {
        self.latency.report()
    }

    pub fn overhead(&self) -> Summary {
        self.latency.overhead.summary()
    }

    pub fn brain(&self) -> Summary {
        self.latency.brain.summary()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    #[error("brain: {0}")]
    Brain(#[from] BrainError),
    #[error("{0}")]
    WindowModel(String),
    #[error("bridge {path}: {source}")]
    Bridge { path: PathBuf, source: std::io::Error },
}

fn load_map(cache_dir: &std::path::Path, name: &str, sha: &[u8; 32]) -> Option<MapData> {
    let bytes = map_cache::read_cached(cache_dir, name, sha)?;
    match ddai_map::load_map(&bytes) {
        Ok(loaded) => Some(loaded.data),
        Err(e) => {
            tracing::error!(error = %e, "the cached map does not parse");
            None
        }
    }
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Runs the bot until it is stopped. Blocks the calling thread (build the brain here: it is not
/// `Send`).
pub fn run(cfg: RunnerConfig) -> Result<RunReport, RunnerError> {
    let brain = make_brain(cfg.bot.brain, &cfg.brain)?;
    let mut client_cfg = cfg.client.clone();
    // The bot's own needs: exact in-flight inputs, and the audit if asked for.
    client_cfg.emit_input_sent = true;
    client_cfg.emit_outgoing_audit = cfg.audit_outgoing;
    // The input timing / margin of the connection, every log period (task 4.4: the soak journal reads it).
    client_cfg.margin_report_every = Some(LOG_EVERY);
    let cache_dir = client_cfg.cache_dir.clone();
    let mut bridge = match &cfg.bridge_path {
        Some(path) => Some(Bridge::bind(path).map_err(|source| RunnerError::Bridge {
            path: path.clone(),
            source,
        })?),
        None => None,
    };

    let mut bot_cfg = cfg.bot.clone();
    bot_cfg.async_seal = true; // the decision path must stay cheap (D-042)
    if cfg.client.precise_wakeups && bot_cfg.driver_pickup == crate::bot::DRIVER_PICKUP {
        // The driver wakes on the hand-over instead of at its next poll (task 3.11).
        bot_cfg.driver_pickup = crate::bot::DRIVER_PICKUP_PRECISE;
    }
    let mut bot = Bot::new(
        bot_cfg,
        brain,
        nav_hooks(cfg.nav.clone(), cfg.nav_handle.clone()),
        cfg.relations.clone(),
    );
    bot.set_nav_handle(cfg.nav_handle.clone());
    bot.set_brain_options(cfg.brain.clone());
    // D-102: the duel switch (`--no-selfkill` or the marker file `bot/selfkill.off`), re-read once a second in the loop below.
    let mut selfkill = SelfKillSwitch::new(cfg.bot.no_selfkill, cfg.bot.selfkill_marker.clone(), Instant::now());
    bot.set_no_selfkill(selfkill.state().is_some());
    if let Some(why) = selfkill.state() {
        tracing::info!("self-kill: off ({})", why.name());
    }
    // Task 4.12 (D-108): the automatic duel detection is switched off by `--no-duel-detect` or the marker `bot/duel-detect.off` (re-read once a
    // second, so the owner can free a bot without a restart). The same one-second marker reader as the self-kill switch.
    let mut duel_switch = SelfKillSwitch::new(!cfg.bot.duel_detect, cfg.bot.duel_detect_marker.clone(), Instant::now());
    bot.set_duel_detect(duel_switch.state().is_none());
    if let Some(why) = duel_switch.state() {
        tracing::info!("duel detection: off ({})", why.name());
    }
    // Task 3.20 (D-112): the server's pre-inputs are played with `--preinput on` unless the marker `bot/preinput.off` exists (re-read once a second).
    let mut pre_switch = SelfKillSwitch::new(!cfg.bot.preinput, cfg.bot.preinput_marker.clone(), Instant::now());
    let pre_mode = |sw: &SelfKillSwitch| match (cfg.bot.preinput, sw.state()) {
        (false, _) => crate::bot::PreInputMode::Off,
        (true, None) => crate::bot::PreInputMode::On,
        (true, Some(_)) => crate::bot::PreInputMode::Killed,
    };
    bot.set_preinput(pre_mode(&pre_switch));
    if cfg.bot.preinput {
        tracing::info!(
            "pre-inputs: {} (--preinput; task 3.20, D-112): the other tees' real inputs the server sends ahead of their ticks play in the prediction; the marker {:?} switches it off",
            bot.preinput_status().0.name(),
            cfg.bot.preinput_marker
        );
    }
    if bot.selfkill_policy().is_smart() {
        tracing::info!("self-kill policy: smart (kill only when waiting costs more than a kill; D-108)");
    }
    tracing::info!(
        server = %cfg.server,
        brain = bot.brain_name(),
        mode = bot.mode().name(),
        wb = cfg.nav.wb_mode.name(),
        strong = cfg.nav.strong,
        "starting the bot"
    );
    // Task 3.17 (D-111): the learned window model, if asked for (only the hybrid brain uses it). A model that cannot be loaded refuses the
    // start: running an A/B without the arm that was asked for would be worse than not running.
    match (&cfg.bot.window_model, cfg.bot.brain) {
        (Some(wm), crate::brains::BrainKind::Hybrid) => {
            let wm = crate::oppnet::WindowModelConfig {
                server_tag: cfg.server.to_string(),
                ..wm.clone()
            };
            let rt = crate::oppnet::WindowModelRt::load(&wm, Instant::now()).map_err(RunnerError::WindowModel)?;
            bot.set_window_model(Some(rt));
        }
        (Some(wm), other) => tracing::info!(
            model = %wm.model.display(),
            brain = other.name(),
            "window model: ignored (it belongs to the hybrid brain)"
        ),
        (None, _) => {}
    }
    let mut client = Client::connect(cfg.server, client_cfg);

    let started = Instant::now();
    let mut report = RunReport {
        exit_code: EXIT_OK,
        stats: BotStats::default(),
        latency: LatencyStats::default(),
        block_stats: Default::default(),
        outgoing: BTreeMap::new(),
        kill_ticks: Vec::new(),
        kill_command_ticks: Vec::new(),
        owner_chat: OwnerChatStats::default(),
        gave_up: None,
        map_name: None,
        events: Vec::new(),
        seal_times: Default::default(),
        reach_times: Default::default(),
        margin: None,
        elapsed: Duration::ZERO,
        bridge_clients_dropped: 0,
    };
    let mut pending: Option<Box<LiveWorldSnapshot>> = None;
    // Task 4.9 (D-094): the owner's website lines wait here for their turn.
    let mut owner_chat = OwnerChat::new();
    let mut ended = false;
    // Set by the bot (moved to the spectators) or the reconnect budget: stop and disconnect politely.
    let mut stop_now = false;
    let diag_block_clips = std::env::var_os("DDAI_DIAG_BLOCK_CLIPS").is_some();
    let mut diag_clips_left = DIAG_BLOCK_CLIPS;
    let mut budget = ReconnectBudget::default();
    // Task 3.11: the opt-in per-input trace for timing experiments (`DDAI_INPUT_TRACE`).
    let mut trace = match &cfg.bot.input_trace {
        Some(path) => InputTrace::create(path).ok(),
        None => InputTrace::from_env(),
    };
    let mut next_status = Instant::now();
    let mut next_log = Instant::now() + LOG_EVERY;
    let mut frame_chars: Vec<FrameChar> = Vec::with_capacity(128);
    // Task 5.10: the last `PLAYERINFO` sent (serialised), so scores and pings go out again only when one changed.
    let mut last_player_info: Vec<u8> = Vec::new();
    let mut next_player_info = Instant::now();
    let mut last_tick = 0;
    // Task 7.4: the fly's visualisation stream. Its layout goes to the bridge at the start and after every brain switch;
    // frames are asked of the brain only while the bridge has a subscriber.
    let mut viz_generation = u64::MAX;

    let time_up = |started: Instant| cfg.duration.is_some_and(|d| started.elapsed() >= d);
    while !ended && !stop_now && !time_up(started) && !cfg.shutdown.load(Ordering::SeqCst) {
        if let Some(now_off) = selfkill.poll(Instant::now()) {
            bot.set_no_selfkill(now_off.is_some());
            match now_off {
                Some(why) => tracing::info!("self-kill: off ({})", why.name()),
                None => tracing::info!("self-kill: on (the marker is gone)"),
            }
        }
        if let Some(now_off) = duel_switch.poll(Instant::now()) {
            bot.set_duel_detect(now_off.is_none());
            match now_off {
                Some(why) => tracing::info!("duel detection: off ({})", why.name()),
                None => tracing::info!("duel detection: on (the marker is gone)"),
            }
        }
        if pre_switch.poll(Instant::now()).is_some() {
            bot.set_preinput(pre_mode(&pre_switch));
            tracing::info!("pre-inputs: {}", bot.preinput_status().0.name());
        }
        bot.window_model_poll(Instant::now());
        let first = client.recv_event(Duration::from_millis(20));
        let mut batch: Vec<ClientEvent> = Vec::new();
        batch.extend(first);
        batch.extend(client.events());
        for ev in batch {
            handle_event(
                ev,
                &mut bot,
                &mut bridge,
                &mut report,
                &mut pending,
                &mut ended,
                &mut stop_now,
                &mut budget,
                &cache_dir,
                &cfg,
                &mut trace,
            );
        }
        if let Some(snap) = pending.take() {
            let out = bot.on_snapshot(&snap);
            last_tick = snap.tick;
            apply_output(&client, &out, &snap, &mut report);
            if let Some(reason) = bot.stop_reason() {
                // F3: moved to the spectators after having played = a moderation signal (D-016).
                tracing::error!(reason = reason.describe(), "the bot stops and does not rejoin");
                report.gave_up = Some((reason.describe().to_string(), GaveUpCategory::KickedOrBanned));
                stop_now = true;
            }
            if let Some(b) = bridge.as_mut() {
                b.accept_pending();
                publish(b, &bot, &snap, &mut frame_chars);
                if b.fly_wanted()
                    && let Some(frame) = bot.viz_frame(u32::try_from(snap.tick.max(0)).unwrap_or(0))
                {
                    b.send_fly(frame);
                }
            }
        } else if let Some(b) = bridge.as_mut() {
            b.accept_pending();
        }
        if let Some(b) = bridge.as_mut()
            && viz_generation != bot.brain_generation()
        {
            viz_generation = bot.brain_generation();
            b.set_fly_meta(bot.viz_meta().as_deref());
        }
        // The operator's commands (and the navigation's answers to them), between two snapshots.
        if let Some(inbox) = &cfg.commands {
            while let Some(req) = inbox.try_next() {
                let reply = match req.cmd {
                    BotCommand::Say { team, text } => {
                        let result =
                            owner_chat.submit(OwnerSay::new(team, text), started.elapsed(), bot.is_connected());
                        if let Err(refusal) = &result {
                            tracing::info!(reason = refusal.code(), "owner chat refused");
                        }
                        ownerchat::reply_for(&result)
                    }
                    cmd => bot.command(cmd),
                };
                let _ = req.reply.send(reply);
            }
        }
        // The owner's lines whose turn has come (at most one per pass; none outside the game).
        if let Some(say) = owner_chat.poll(started.elapsed(), bot.is_connected()) {
            if say.text.as_str().starts_with('/') {
                // A server command typed by the owner (4.9b, `/kill` among them) is not the `/kill` fallback of D-078.
                bot.on_owner_command(say.text.as_str());
            }
            client.owner_say(say);
        }
        for line in cfg.nav_handle.drain_replies() {
            match &cfg.console_out {
                Some(print) => print(&line),
                None => tracing::info!(target: "nav", "{line}"),
            }
        }
        if bot.quit_requested() && !stop_now {
            tracing::info!("quit requested from the console");
            stop_now = true;
        }
        let mut events: Vec<BotEvent> = bot.drain_events().collect();
        for e in events.drain(..) {
            log_event(&e);
            // Task 3.10 live diagnosis (opt-in by the environment, at most `DIAG_BLOCK_CLIPS` per run): the ring as a clip when a block's fate is known,
            // `manual-<tick>-held` / `-escaped`, to read with `ddnet-ai clip held --track`.
            if diag_block_clips && diag_clips_left > 0 {
                let note = match &e {
                    BotEvent::BlockHeld { tick, .. } => Some(format!("held-{tick}")),
                    BotEvent::BlockEscaped { tick, .. } => Some(format!("escaped-{tick}")),
                    _ => None,
                };
                if let Some(note) = note {
                    diag_clips_left -= 1;
                    if let Err(err) = bot.save_clip(&note) {
                        tracing::warn!(%err, "block clip not saved");
                    }
                }
            }
            if let BotEvent::RosterChanged { .. } = &e {
                if let Some(b) = bridge.as_mut() {
                    b.send_players(&players_message(&bot, cfg.web_names));
                    let info = player_info_message(&bot, cfg.web_names);
                    last_player_info = serde_json::to_vec(&info).unwrap_or_default();
                    b.send_player_info(&info);
                }
                if let Some(path) = &cfg.debug_names_log {
                    append_debug_names(path, &bot);
                }
            }
            if report.events.len() < 4096 {
                report.events.push(e);
            }
        }
        let now = Instant::now();
        if now >= next_status {
            next_status = now + STATUS_EVERY;
            // Nobody reading the bridge: do not even build the message.
            if let Some(b) = bridge.as_mut()
                && b.clients() > 0
            {
                b.send_status(&status_message(&bot, last_tick, &cfg));
                // Scores and pings change without the roster changing: once a second, when something differs.
                if now >= next_player_info {
                    next_player_info = now + Duration::from_secs(1);
                    let info = player_info_message(&bot, cfg.web_names);
                    let json = serde_json::to_vec(&info).unwrap_or_default();
                    if json != last_player_info {
                        last_player_info = json;
                        b.send_player_info(&info);
                    }
                }
            }
        }
        if now >= next_log {
            next_log = now + LOG_EVERY;
            tracing::info!(stats = ?bot.stats(), "bot status\n{}", bot.latency().report());
            let (mode, c) = bot.preinput_status();
            if c.received > 0 || mode != crate::bot::PreInputMode::Off {
                tracing::info!(
                    mode = mode.name(),
                    received = c.received,
                    stored = c.stored,
                    ahead = c.ahead,
                    behind = c.behind,
                    stale = c.stale,
                    invalid = c.invalid,
                    used = c.used,
                    distrusted = c.distrusted,
                    lead = ?c.lead,
                    known_ahead = ?c.known_ahead,
                    "pre-inputs (lead = intended tick - latest snapshot tick, bins -4..=11)"
                );
            }
            if let Some(m) = &report.margin {
                tracing::info!(
                    count = m.count,
                    late = m.late_count,
                    stalls = m.stall_count,
                    late_fraction = m.late_fraction,
                    margin_ms = m.margin_ms,
                    adaptive = m.adaptive,
                    changes = m.margin_changes,
                    min_ms = m.min_ms.unwrap_or(-1),
                    p50_ms = m.p50_ms.unwrap_or(-1),
                    p99_ms = m.p99_ms.unwrap_or(-1),
                    superseded = m.superseded_decisions,
                    "input margin"
                );
            }
        }
    }

    if !ended {
        tracing::info!("stopping: disconnecting");
        client.disconnect();
    }
    client.join();
    if let Some(t) = trace.as_mut() {
        t.flush();
    }
    for ev in client.events() {
        if let ClientEvent::MarginSummary(m) = ev {
            report.margin = Some(m);
        } else if let ClientEvent::Session(s) = &ev
            && let SessionEvent::OutgoingGame { label, accepted } = &**s
        {
            count_outgoing(&mut report, label, *accepted);
        }
    }
    bot.shutdown();
    report.owner_chat = owner_chat.stats();
    report.stats = bot.stats();
    report.latency = bot.latency().clone();
    report.block_stats = bot.block_stats();
    report.seal_times = bot.seal_times().clone();
    report.reach_times = bot.reach_times().clone();
    report.elapsed = started.elapsed();
    if let Some(b) = &bridge {
        report.bridge_clients_dropped = b.dropped_clients();
    }
    report.exit_code = exit_code_for(report.gave_up.as_ref().map(|(_, c)| *c));
    tracing::info!(exit = report.exit_code, stats = ?report.stats, "the bot stopped\n{}", report.latency_text());
    Ok(report)
}

/// The process exit code for how the driver ended (D-016/D-037/D-050): a kick or ban is 3, a join
/// that never completed (watchdog, reconnect loop, attempts exhausted, reconnect budget, or the
/// in-game reconnect cap, or a SOCKS5 proxy's final refusal: 0x07, wrong password) is 4, being moved to the spectators after having played is 3 like a kick, a
/// requested stop or no stop at all is 0, anything else (local errors, protocol violations, redirect
/// loops) is 1. None of them is ever retried by the runner.
pub fn exit_code_for(category: Option<GaveUpCategory>) -> u8 {
    match category {
        Some(GaveUpCategory::KickedOrBanned) => EXIT_KICKED_OR_BANNED,
        Some(
            GaveUpCategory::HandshakeTimeout
            | GaveUpCategory::ReconnectLoop
            | GaveUpCategory::TooManyAttempts
            | GaveUpCategory::ReconnectBudgetExhausted
            | GaveUpCategory::ProxyRefused,
        ) => EXIT_JOIN_FAILED,
        Some(GaveUpCategory::Requested) | None => EXIT_OK,
        Some(GaveUpCategory::ProtocolViolation | GaveUpCategory::RedirectLoop | GaveUpCategory::LocalError) => {
            EXIT_ERROR
        }
    }
}

fn count_outgoing(report: &mut RunReport, label: &str, accepted: bool) {
    let e = report.outgoing.entry(label.to_string()).or_insert((0, 0));
    if accepted {
        e.0 += 1;
    } else {
        e.1 += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_event(
    ev: ClientEvent,
    bot: &mut Bot,
    bridge: &mut Option<Bridge>,
    report: &mut RunReport,
    pending: &mut Option<Box<LiveWorldSnapshot>>,
    ended: &mut bool,
    stop_now: &mut bool,
    budget: &mut ReconnectBudget,
    cache_dir: &std::path::Path,
    cfg: &RunnerConfig,
    trace: &mut Option<InputTrace>,
) {
    match ev {
        ClientEvent::LiveWorldSnapshot(snap) => {
            // Collapsing: a newer snapshot replaces one nobody decided yet.
            if pending.replace(snap).is_some() {
                bot.note_collapsed(1);
            }
        }
        ClientEvent::InputLatency {
            tick,
            since_snapshot,
            tag,
            handed_after,
            pickup,
        } => {
            if let Some(t) = trace {
                t.decision(tick, tag, since_snapshot, handed_after, pickup);
            }
            bot.note_wire_latency(since_snapshot, tick, tag, handed_after, pickup)
        }
        ClientEvent::Session(s) => match *s {
            SessionEvent::InputSent { tick, input } => {
                if let Some(t) = trace {
                    t.sent(tick, &input);
                }
                bot.on_input_sent(tick, &input)
            }
            SessionEvent::InputTiming { tick, time_left } => {
                if let Some(t) = trace {
                    t.timing(tick, time_left);
                }
                bot.on_input_timing(tick, time_left)
            }
            SessionEvent::ExGameMessage(ddai_net::generated::messages::ExGameMsg::SvPreInput(p)) => {
                bot.on_pre_input(&p)
            }
            SessionEvent::MapChanging { name, .. } => {
                tracing::info!(map = %name, "map changing");
                bot.on_map_changing();
                *pending = None;
            }
            SessionEvent::MapLoaded(loaded) => match load_map(cache_dir, &loaded.name, &loaded.sha256) {
                Some(data) => {
                    let (w, h) = (data.width, data.height);
                    bot.set_map_ident(MapIdent {
                        name: loaded.name.clone(),
                        sha256: loaded.sha256,
                    });
                    bot.on_map_loaded(Arc::new(data));
                    report.map_name = Some(loaded.name.clone());
                    if let Some(b) = bridge.as_mut() {
                        b.send_map(&MapMessage {
                            name: loaded.name.clone(),
                            sha256: hex(&loaded.sha256),
                            w,
                            h,
                        });
                    }
                    tracing::info!(map = %loaded.name, w, h, "map ready");
                }
                None => {
                    tracing::error!(map = %loaded.name, "could not read the loaded map back from the cache: the bot will not act")
                }
            },
            SessionEvent::GameMessage(GameMsg::SvKillMsg(k)) => bot.on_kill_message(k.killer, k.victim, k.weapon),
            // Chat is read-only here. The bot itself looks at one line, the server's own "Kill Protection enabled" (a system
            // line, `client_id` -1). Every line is also handed to the web unit's chat panel (task 5.10) for display, with the
            // sender and any known name inside the text replaced by tags unless `--web-names`; the text is neither kept
            // nor logged by the bot, and the bot never answers (D-007).
            SessionEvent::GameMessage(GameMsg::SvChat(c)) => {
                // Task 4.12: the F-DDrace `/1vs1` system lines (read, never kept or logged) are the duel detector's second signal.
                bot.on_chat_line(c.client_id, &c.message);
                if crate::killfallback::is_kill_protection_notice(c.client_id, &c.message) {
                    tracing::info!("the server dropped a Cl_Kill: kill protection");
                    bot.on_kill_protection_notice();
                }
                if let Some(b) = bridge.as_mut()
                    && b.clients() > 0
                {
                    b.send_chat(&chat_message(
                        bot.players(),
                        cfg.web_names,
                        c.team,
                        c.client_id,
                        &c.message,
                    ));
                }
            }
            SessionEvent::OutgoingGame { label, accepted } => count_outgoing(report, label, accepted),
            SessionEvent::InGame => {
                tracing::info!("in game");
                bot.on_in_game();
                budget.on_in_game();
            }
            SessionEvent::Disconnected { reason, by_peer } => {
                // The server's wording is free text: known names and clans become tags, and the original length is kept.
                let (shown, reason_len) = redacted_reason(bot.players(), reason.as_deref());
                tracing::warn!(reason = ?shown, reason_len, by_peer, "disconnected");
                bot.on_disconnected();
                *pending = None;
                if !budget.on_dropped(Instant::now()) && report.gave_up.is_none() {
                    let why = format!(
                        "more than {MAX_INGAME_RECONNECTS} in-game connections dropped within {} s: not reconnecting again (D-050)",
                        INGAME_RECONNECT_WINDOW.as_secs()
                    );
                    tracing::error!(%why, "the bot stops");
                    report.gave_up = Some((why, GaveUpCategory::ReconnectBudgetExhausted));
                    *stop_now = true;
                }
            }
            SessionEvent::ProtocolViolation { reason } => tracing::error!(%reason, "protocol violation"),
            SessionEvent::Anomaly(m) => tracing::warn!(%m, "anomaly"),
            _ => {}
        },
        ClientEvent::GaveUp { reason, category } => {
            tracing::error!(%reason, ?category, "the driver gave up; the bot stops and does not retry");
            report.gave_up = Some((reason, category));
            *ended = true;
        }
        ClientEvent::ReconnectAttempt { attempt, addr, backoff } => {
            tracing::warn!(attempt, %addr, ?backoff, "reconnecting");
        }
        ClientEvent::RedirectFollowed { to } => tracing::info!(%to, "following a redirect"),
        ClientEvent::RedirectRefused { reason } => tracing::error!(%reason, "redirect refused"),
        ClientEvent::ServerRequestedReconnect { addr, attempt } => {
            tracing::info!(%addr, attempt, "the server asked for a reconnect");
        }
        ClientEvent::MarginSummary(m) => report.margin = Some(m),
        ClientEvent::OwnPosition { .. } | ClientEvent::OwnTeam { .. } => {}
    }
    let _ = cfg;
}

/// A disconnect reason for the log: the text with every known player's name and clan replaced by the tag, and the original byte length.
fn redacted_reason(players: &crate::players::PlayerTable, reason: Option<&str>) -> (Option<String>, usize) {
    (reason.map(|r| players.redact(r)), reason.map_or(0, str::len))
}

fn apply_output(client: &Client, out: &Output, snap: &LiveWorldSnapshot, report: &mut RunReport) {
    if let Some(input) = out.input {
        client.set_input_for_snapshot(input, snap.arrived, out.tag);
    }
    if out.kill {
        tracing::info!(tick = snap.tick, "requesting Cl_Kill (unstick)");
        report.kill_ticks.push(snap.tick);
        client.kill();
    }
    if out.kill_command {
        tracing::info!(
            tick = snap.tick,
            "requesting /kill (fallback: the protocol Cl_Kill had no effect)"
        );
        report.kill_command_ticks.push(snap.tick);
        client.server_command(ddai_net::server_command::ServerCommand::Kill);
    }
    if let Some(team) = out.set_team {
        client.set_team(team);
    }
}

/// Clips per run the `DDAI_DIAG_BLOCK_CLIPS` diagnosis saves at most.
const DIAG_BLOCK_CLIPS: u32 = 12;

fn log_event(e: &BotEvent) {
    match e {
        BotEvent::Killed { tick, reason, why } => match why {
            Some(why) => tracing::info!(tick, ?reason, why = why.name(), "unstick: Cl_Kill (smart)"),
            None => tracing::info!(tick, ?reason, "unstick: Cl_Kill"),
        },
        BotEvent::SelfKillSkipped { tick, why } => {
            tracing::info!(
                tick,
                why = why.name(),
                "self-kill skipped: a kill the legacy timers would send is not needed"
            )
        }
        BotEvent::DuelStarted { tick, why } => tracing::warn!(
            tick,
            by = why.name(),
            "duel: an F-DDrace 1vs1 is on: self-kill off until it ends (like --no-selfkill)"
        ),
        BotEvent::DuelEvidence { tick, len } => tracing::info!(tick, "duel evidence: owner command (len {len})"),
        BotEvent::DuelEnded { tick } => tracing::info!(tick, "duel: the 1vs1 is over: self-kill as configured again"),
        BotEvent::Block { tick, victim } => tracing::info!(tick, victim, "block"),
        BotEvent::BlockedBy { tick, by } => tracing::info!(tick, by, "blocked by"),
        BotEvent::BlockHeld { tick, victim, died } => tracing::info!(tick, victim, died, "block held"),
        BotEvent::BlockEscaped { tick, victim, after } => tracing::info!(tick, victim, after, "block escaped"),
        BotEvent::TargetChanged { tick, to } => tracing::info!(tick, target = ?to, "target"),
        BotEvent::Respawned { tick } => tracing::info!(tick, "life started"),
        BotEvent::KillFallback { tick, noticed } => {
            tracing::info!(
                tick,
                noticed,
                "kill fallback: /kill sent, the protocol Cl_Kill had no effect"
            )
        }
        BotEvent::KillProtectionLearned { tick, life_secs } => tracing::info!(
            tick,
            life_minutes = f64::from(*life_secs) / 60.0,
            "kill protection: a /kill ended a life of this many minutes (the server's threshold is not above it)"
        ),
        BotEvent::KillFallbackGaveUp { tick } => tracing::warn!(
            tick,
            "kill fallback: three /kill in one life without a death; not asking again for this life"
        ),
        BotEvent::Joining { tick } => tracing::info!(tick, "in the spectators: asking to join"),
        BotEvent::JoinGaveUp { tick } => tracing::warn!(tick, "still a spectator after all tries: not asking again"),
        BotEvent::MovedToSpectators { tick } => tracing::error!(tick, "moved to the spectators after having played"),
        BotEvent::PausedByServer { tick } => {
            tracing::info!(
                tick,
                "paused by the server: owner /pause or /spec (the bot idles until it is repeated)"
            )
        }
        BotEvent::ResumedByServer { tick } => {
            tracing::info!(tick, "resumed: the server's pause is over, the bot plays again")
        }
        BotEvent::PredictionClamped {
            tick,
            wanted_ahead,
            cap,
        } => tracing::warn!(
            tick,
            wanted_ahead,
            cap,
            "the prediction horizon hit its cap (a very long RTT): decisions are made on a world that stops short of their tick"
        ),
        BotEvent::TickReset { from, to } => tracing::info!(from, to, "game tick went backwards"),
        BotEvent::RosterChanged { players } => tracing::debug!(players, "roster changed"),
        BotEvent::ClipSaved {
            tick,
            kind,
            severity,
            path,
        } => tracing::info!(tick, kind, severity, path, "clip saved"),
    }
}

fn status_message(bot: &Bot, tick: i32, cfg: &RunnerConfig) -> StatusMessage {
    let s = bot.status();
    let (total, brain, overhead) = bot.latency().status_summaries();
    let nav = cfg.nav_handle.status();
    let (window_model, window_guard) = bot.window_model_status();
    let (pre_mode, pre_counts) = bot.preinput_status();
    StatusMessage {
        tick,
        own: s.own_id,
        target: s.target_id,
        mode: bot.mode().name().to_string(),
        brain: bot.brain_name().to_string(),
        alive: s.alive,
        frozen: s.frozen,
        blocks: s.blocks.blocks,
        blocked_by: s.blocks.blocked_by,
        self_kills: s.stats.self_kills,
        decisions: s.stats.decisions,
        collapsed: s.stats.collapsed,
        decide_p50_us: total.p50_us,
        decide_p99_us: total.p99_us,
        brain_p99_us: brain.p99_us,
        overhead_p99_us: overhead.p99_us,
        telemetry: bot.brain_telemetry().and_then(|t| serde_json::from_str(&t).ok()),
        connected: bot.is_connected(),
        server: cfg.server.to_string(),
        map: bot.map_name().to_string(),
        name: cfg.client.name.clone(),
        clan: cfg.client.clan.clone(),
        skin: cfg.client.skin.clone(),
        target_tag: (s.target_id >= 0).then(|| bot.tag_of(s.target_id).to_string()),
        wb: nav.wb,
        goto: if nav.walking { nav.progress } else { String::new() },
        deaths: s.stats.deaths,
        clips_saved: bot.stats().clips_saved,
        kill_cooldown_ticks: bot.kill_cooldown_ticks(),
        paused: bot.paused(),
        finish: finish_label(cfg.bot.finish, cfg.brain.hybrid_finish, cfg.bot.finish_wb).to_string(),
        selfkill: if bot.no_selfkill() { "off" } else { "on" }.to_string(),
        wb_smart: if cfg.nav.wb_smart { "on" } else { "off" }.to_string(),
        duel: bot.duel().is_some(),
        selfkill_policy: bot.selfkill_policy().name().to_string(),
        preinput: pre_mode.name().to_string(),
        preinput_stats: preinput_stats(&pre_counts),
        window_model: window_model.to_string(),
        window_guard: window_guard.and_then(|g| serde_json::to_value(g).ok()),
    }
}

/// STATUS `preinput_stats`: the counters of the server's pre-inputs (task 3.20).
fn preinput_stats(c: &ddai_world::preinput::PreInputCounts) -> serde_json::Value {
    serde_json::json!({
        "received": c.received, "stored": c.stored, "ahead": c.ahead, "behind": c.behind, "stale": c.stale,
        "invalid": c.invalid, "duplicate": c.duplicate, "used": c.used, "distrusted": c.distrusted,
        "lead_from": ddai_world::preinput::LEAD_MIN, "lead": c.lead.to_vec(), "known_ahead": c.known_ahead.to_vec(),
    })
}

/// The finishing mode the process runs with, as `--finish` spells it: `off`, `target` (the bot's target rule), `wb` (the target rule plus the
/// wayblock hold, task 3.18) or `full` (the target rule plus the hybrid's drag shaping). Three independent switches carry it
/// ([`BotConfig::finish`], [`BrainOptions::hybrid_finish`], [`BotConfig::finish_wb`]); the drag alone (without the target rule) is not a mode
/// the command line can ask for, and reads as `off`.
fn finish_label(target_rule: bool, hybrid_drag: bool, wb_hold: bool) -> &'static str {
    match (target_rule, hybrid_drag, wb_hold) {
        (false, _, _) => "off",
        (true, false, false) => "target",
        (true, _, true) => "wb",
        (true, true, false) => "full",
    }
}

/// `--debug-names`: appends `<tag> <name>` for the current roster to a local file. Off by default;
/// this is the only place a nickname is ever written by the bot.
fn append_debug_names(path: &std::path::Path, bot: &Bot) {
    use std::io::Write;
    let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    for (id, slot) in bot.players().present() {
        let _ = writeln!(f, "{} {}", bot.players().tag(id), slot.name);
    }
}

/// The roster message: tags, or real names with `--web-names`.
fn players_message(bot: &Bot, web_names: bool) -> PlayersMessage {
    let players = bot.players();
    PlayersMessage {
        own: players.own_id().unwrap_or(-1),
        list: players
            .present()
            .map(|(id, slot)| PlayerEntry {
                id,
                name: if web_names {
                    slot.name.clone()
                } else {
                    players.tag(id).to_string()
                },
                team: slot.team,
            })
            .collect(),
    }
}

/// One chat line for the web unit: the sender as a tag (or the nickname with `--web-names`), and, without `--web-names`,
/// every known player's name and clan inside the text replaced by that player's tag.
fn chat_message(
    players: &crate::players::PlayerTable,
    web_names: bool,
    team: i32,
    cid: i32,
    text: &str,
) -> ChatMessage {
    let name = match players.get(cid).filter(|s| s.present) {
        _ if cid < 0 => String::new(),
        Some(slot) if web_names => slot.name.clone(),
        Some(_) => players.tag(cid).to_string(),
        // Someone who left before the line arrived: the id is all there is (no name is known, so none is invented).
        None => format!("c{cid}"),
    };
    // Cut first: the redaction below is O(length x names) on the bot's loop thread, so the length it sees stays bounded.
    let text = crate::bridge::cut_at(text, crate::bridge::MAX_CHAT_TEXT);
    ChatMessage {
        team,
        cid,
        name,
        text: if web_names {
            text.to_string()
        } else {
            players.redact(text)
        },
    }
}

/// How each present player looks, with their numbers (`PLAYERINFO`); the clan only with `--web-names`.
fn player_info_message(bot: &Bot, web_names: bool) -> PlayerInfoMessage {
    PlayerInfoMessage {
        list: bot
            .players()
            .present()
            .map(|(id, slot)| PlayerInfoEntry {
                id,
                clan: if web_names { slot.clan.clone() } else { String::new() },
                skin: slot.skin.clone(),
                cc: slot.use_custom_color,
                cb: slot.color_body,
                cf: slot.color_feet,
                country: slot.country,
                score: slot.score,
                // To 10 ms steps: a ping that jitters by a millisecond is not worth a message to every open page.
                ping: (slot.latency_ms + 5).div_euclid(10) * 10,
            })
            .collect(),
    }
}

/// One frame to the web unit, from the snapshot and the bot's own tee view.
fn publish(b: &mut Bridge, bot: &Bot, snap: &LiveWorldSnapshot, chars: &mut Vec<FrameChar>) {
    chars.clear();
    for cv in &snap.characters {
        let c = &cv.character;
        let tee = bot.tees().get(cv.id);
        chars.push(FrameChar {
            id: u8::try_from(cv.id.clamp(0, 255)).unwrap_or(0),
            alive: true,
            frozen: tee.is_some_and(|t| t.frozen),
            deep_frozen: tee.is_some_and(|t| t.deep_frozen),
            live_frozen: false,
            team: bot
                .players()
                .get(cv.id)
                .map_or(0, |s| u8::try_from(s.team.clamp(0, 255)).unwrap_or(0)),
            weapon: u8::try_from(c.weapon.clamp(0, 255)).unwrap_or(0),
            x: c.x,
            y: c.y,
            aim_x: cv.ddnet.map_or(0, |d| d.target_x),
            aim_y: cv.ddnet.map_or(0, |d| d.target_y),
            hook_state: c.hook_state,
            hook_x: c.hook_x,
            hook_y: c.hook_y,
            hooked_id: c.hooked_player,
        });
    }
    b.send_frame(u32::try_from(snap.tick.max(0)).unwrap_or(0), chars);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_status_names_the_finishing_mode_as_the_command_line_spells_it() {
        assert_eq!(finish_label(false, false, false), "off");
        assert_eq!(finish_label(true, false, false), "target");
        assert_eq!(finish_label(true, true, false), "full");
        assert_eq!(finish_label(true, false, true), "wb");
        // The drag without the target rule cannot be asked for (`--finish full` sets both): read as off, never as a mode of its own.
        assert_eq!(finish_label(false, true, false), "off");
        assert_eq!(finish_label(false, false, true), "off");
    }

    fn table() -> crate::players::PlayerTable {
        use crate::players::test_support::player;
        let mut t = crate::players::PlayerTable::new([9; 16]);
        t.update(
            &[
                player(0, "Muha", "Neuroset", true, 0, None),
                player(3, "Bob", "ХАОС", false, 0, None),
            ],
            &crate::relations::Relations::new(),
        );
        t
    }

    /// Task 5.10: the sender of a chat line is a tag, names inside the text become tags, the server is nobody, and a sender who left is
    /// never invented a name; with `--web-names` the nickname and the text go through as they are.
    #[test]
    fn chat_lines_for_the_web_carry_tags_not_nicknames_unless_asked() {
        let t = table();
        let bob_tag = t.tag(3).to_string();
        let m = chat_message(&t, false, 1, 3, "hello Muha, it is bob from ХАОС");
        assert_eq!((m.team, m.cid), (1, 3));
        assert_eq!(m.name, bob_tag);
        assert!(
            !m.text.contains("Muha") && !m.text.to_lowercase().contains("bob") && !m.text.contains("ХАОС"),
            "{}",
            m.text
        );
        assert!(m.text.contains(&t.tag(0).to_string()));

        let m = chat_message(&t, false, 0, -1, "Kill Protection enabled. You can use /kill later");
        assert_eq!((m.cid, m.name.as_str()), (-1, ""));
        let m = chat_message(&t, false, 0, 77, "who am i");
        assert_eq!(m.name, "c77", "an id with nobody in the slot: only the id");

        let m = chat_message(&t, true, 0, 3, "hello Muha");
        assert_eq!((m.name.as_str(), m.text.as_str()), ("Bob", "hello Muha"));
    }

    #[test]
    fn a_huge_chat_text_is_cut_before_the_names_are_looked_for() {
        let t = table();
        let long = "я".repeat(5000) + " Muha";
        let m = chat_message(&t, false, 0, 3, &long);
        assert!(m.text.len() <= crate::bridge::MAX_CHAT_TEXT, "{}", m.text.len());
        assert!(m.text.chars().all(|c| c == 'я'));
    }

    #[test]
    fn player_info_has_the_look_and_the_clan_only_with_real_names() {
        use crate::players::test_support::player;
        let mut p = player(3, "Bob", "ХАОС", false, 0, None);
        let ci = p.client_info.as_mut().unwrap();
        ci.skin = "coala".into();
        ci.use_custom_color = 1;
        ci.color_body = 0x00ff_8040;
        ci.color_feet = 5;
        ci.country = 276;
        p.info.score = 12;
        let mut t = crate::players::PlayerTable::new([9; 16]);
        let changed = t.update(&[p.clone()], &crate::relations::Relations::new());
        assert!(changed);
        // A new skin alone is not a roster change: the look goes out with the once-a-second PLAYERINFO.
        p.client_info.as_mut().unwrap().skin = "x_ninja".into();
        assert!(!t.update(&[p.clone()], &crate::relations::Relations::new()));
        assert!(!t.update(&[p], &crate::relations::Relations::new()), "nothing changed");
        let slot = t.get(3).unwrap();
        assert_eq!(
            (slot.skin.as_str(), slot.use_custom_color, slot.color_body, slot.country),
            ("x_ninja", true, 0x00ff_8040, 276)
        );
    }

    /// Review F2: a server-side rainbow (DDNet++ `/rainbow`, F-DDrace) changes a player's colour in every snapshot. That is
    /// never a roster change (which would resend PLAYERS and PLAYERINFO, log, and fill the report's event list at 50 Hz);
    /// the table still holds the newest colour for the once-a-second PLAYERINFO.
    #[test]
    fn a_colour_that_changes_every_tick_is_never_a_roster_change() {
        use crate::players::test_support::player;
        let mut p = player(3, "Bob", "ХАОС", false, 0, None);
        p.client_info.as_mut().unwrap().use_custom_color = 1;
        let mut t = crate::players::PlayerTable::new([9; 16]);
        let rel = crate::relations::Relations::new();
        assert!(t.update(&[p.clone()], &rel), "the first sight is a roster change");
        for tick in 0..500 {
            p.client_info.as_mut().unwrap().color_body = (tick * 0x0001_0100) & 0x00ff_ffff;
            assert!(!t.update(&[p.clone()], &rel), "tick {tick}");
        }
        assert_eq!(t.get(3).unwrap().color_body, (499 * 0x0001_0100) & 0x00ff_ffff);
    }

    #[test]
    fn every_way_the_driver_can_stop_has_a_distinct_honest_exit_code() {
        assert_eq!(exit_code_for(None), 0);
        assert_eq!(exit_code_for(Some(GaveUpCategory::Requested)), 0);
        assert_eq!(
            exit_code_for(Some(GaveUpCategory::KickedOrBanned)),
            3,
            "kick/ban is never 0"
        );
        for c in [
            GaveUpCategory::HandshakeTimeout,
            GaveUpCategory::ReconnectLoop,
            GaveUpCategory::TooManyAttempts,
            GaveUpCategory::ReconnectBudgetExhausted,
            GaveUpCategory::ProxyRefused,
        ] {
            assert_eq!(exit_code_for(Some(c)), 4, "{c:?}");
        }
        for c in [
            GaveUpCategory::ProtocolViolation,
            GaveUpCategory::RedirectLoop,
            GaveUpCategory::LocalError,
        ] {
            assert_eq!(exit_code_for(Some(c)), 1, "{c:?}");
        }
    }

    #[test]
    fn the_in_game_reconnect_budget_allows_three_drops_per_ten_minutes_then_stops() {
        let t0 = Instant::now();
        let mut b = ReconnectBudget::default();
        for i in 0..MAX_INGAME_RECONNECTS {
            b.on_in_game();
            assert!(
                b.on_dropped(t0 + Duration::from_secs(60 * i as u64)),
                "drop {i} is within the budget"
            );
        }
        b.on_in_game();
        assert!(
            !b.on_dropped(t0 + Duration::from_secs(200)),
            "the fourth drop in the window stops the run"
        );
    }

    #[test]
    fn old_drops_leave_the_window_and_connections_that_never_got_in_do_not_count() {
        let t0 = Instant::now();
        let mut b = ReconnectBudget::default();
        for i in 0..3 {
            b.on_in_game();
            assert!(b.on_dropped(t0 + Duration::from_secs(i)));
        }
        // 11 minutes later the window is empty again.
        b.on_in_game();
        assert!(b.on_dropped(t0 + Duration::from_secs(11 * 60)));
        // A drop of a connection that never reached the game is the driver's budget, not this one's.
        for i in 0..20 {
            assert!(b.on_dropped(t0 + Duration::from_secs(12 * 60 + i)));
        }
        b.on_in_game();
        assert!(
            b.on_dropped(t0 + Duration::from_secs(13 * 60)),
            "second drop in the new window"
        );
    }

    #[test]
    fn a_disconnect_reason_is_logged_with_names_as_tags_and_its_original_length() {
        let mut players = crate::players::PlayerTable::new([3; 16]);
        players.update(
            &[
                crate::players::test_support::player(0, "Muha", "Neuroset", true, 0, None),
                crate::players::test_support::player(4, "Spammer9", "", false, 0, None),
            ],
            &Relations::new(),
        );
        let reason = "kicked for spam by Admin: Spammer9 and friends from neuroset";
        let (shown, len) = redacted_reason(&players, Some(reason));
        let shown = shown.expect("a reason stays a reason");
        assert_eq!(len, reason.len(), "the original length");
        assert!(
            !shown.to_lowercase().contains("spammer9") && !shown.to_lowercase().contains("neuroset"),
            "{shown}"
        );
        assert!(
            shown.contains(&players.tag(4).to_string()) && shown.starts_with("kicked for spam by Admin: "),
            "{shown}"
        );
        assert_eq!(redacted_reason(&players, None), (None, 0));
        assert_eq!(
            redacted_reason(&players, Some("Server shutdown")),
            (Some("Server shutdown".to_string()), 15)
        );
    }

    /// Task 4.4 acceptance: building the bridge's status message (what the runner does 5 times a second between two snapshots)
    /// costs the decision thread at most 0.05 ms (best of 31 batches: the VM is shared). Fresh hybrid bot, **full latency rings**
    /// (the histogram scans), the navigation status and the brain's telemetry with its JSON parse.
    #[test]
    fn building_the_status_message_costs_the_decision_thread_under_fifty_microseconds() {
        let brain =
            make_brain(crate::brains::BrainKind::Hybrid, &BrainOptions::default()).expect("the hybrid brain builds");
        let mut bot = Bot::new(
            BotConfig::default(),
            brain,
            crate::hooks::Hooks::default(),
            Relations::new(),
        );
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..2 * crate::latency::RING {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let brain_us = 100 + x % 6_000;
            bot.latency_mut().record(
                Duration::from_micros(brain_us + 40 + x % 300),
                Duration::from_micros(brain_us),
            );
        }
        let cfg = RunnerConfig {
            server: "127.0.0.1:8303".parse().unwrap(),
            client: ClientConfig::default(),
            bot: BotConfig::default(),
            brain: BrainOptions::default(),
            relations: Relations::new(),
            duration: None,
            bridge_path: None,
            web_names: false,
            debug_names_log: None,
            audit_outgoing: false,
            shutdown: Arc::new(AtomicBool::new(false)),
            nav: NavConfig::default(),
            nav_handle: NavHandle::new(),
            commands: None,
            console_out: None,
        };
        let mut per_call = Vec::new();
        let mut sink = 0usize;
        for _ in 0..31 {
            let t = Instant::now();
            for _ in 0..20 {
                sink += status_message(&bot, 1, &cfg).mode.len();
            }
            per_call.push(t.elapsed() / 20);
        }
        let best = per_call.into_iter().min().unwrap();
        eprintln!("status_message: {best:?} per call (sink {sink})");
        assert!(best < Duration::from_micros(50), "status_message took {best:?}");
    }
}
