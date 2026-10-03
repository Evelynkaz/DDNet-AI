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
//! **Chat.** Nothing here can write chat: the client API has no such call, and the audit events
//! (`SessionEvent::OutgoingGame`, enabled by [`RunnerConfig::audit_outgoing`]) are counted per label so
//! a test can assert that no `Cl_Say` ever reached the wire and that every `Cl_Kill` is the bot's own.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ddai_client::{Client, ClientConfig, ClientEvent, GaveUpCategory, LiveWorldSnapshot, SessionEvent, map_cache};
use ddai_net::generated::messages::GameMsg;
use ddai_physics::map::MapData;

use crate::bot::{Bot, BotConfig, BotEvent, BotStats, Output};
use crate::brains::{BrainError, BrainOptions, make_brain};
use crate::bridge::{Bridge, FrameChar, MapMessage, PlayerEntry, PlayersMessage, StatusMessage};
use crate::command::CommandInbox;
use crate::console::Printer;
use crate::hooks::MapIdent;
use crate::latency::{LatencyStats, Summary};
use crate::nav_hooks::{NavConfig, NavHandle, nav_hooks};
use crate::relations::Relations;

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
    let mut bot = Bot::new(
        bot_cfg,
        brain,
        nav_hooks(cfg.nav.clone(), cfg.nav_handle.clone()),
        cfg.relations.clone(),
    );
    bot.set_nav_handle(cfg.nav_handle.clone());
    bot.set_brain_options(cfg.brain.clone());
    tracing::info!(
        server = %cfg.server,
        brain = bot.brain_name(),
        mode = bot.mode().name(),
        wb = cfg.nav.wb_mode.name(),
        strong = cfg.nav.strong,
        "starting the bot"
    );
    let mut client = Client::connect(cfg.server, client_cfg);

    let started = Instant::now();
    let mut report = RunReport {
        exit_code: EXIT_OK,
        stats: BotStats::default(),
        latency: LatencyStats::default(),
        block_stats: Default::default(),
        outgoing: BTreeMap::new(),
        kill_ticks: Vec::new(),
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
    let mut ended = false;
    // Set by the bot (moved to the spectators) or the reconnect budget: stop and disconnect politely.
    let mut stop_now = false;
    let mut budget = ReconnectBudget::default();
    let mut next_status = Instant::now();
    let mut next_log = Instant::now() + LOG_EVERY;
    let mut frame_chars: Vec<FrameChar> = Vec::with_capacity(128);
    let mut last_tick = 0;
    // Task 7.4: the fly's visualisation stream. Its layout goes to the bridge at the start and after every brain switch;
    // frames are asked of the brain only while the bridge has a subscriber.
    let mut viz_generation = u64::MAX;

    let time_up = |started: Instant| cfg.duration.is_some_and(|d| started.elapsed() >= d);
    while !ended && !stop_now && !time_up(started) && !cfg.shutdown.load(Ordering::SeqCst) {
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
                let reply = bot.command(req.cmd);
                let _ = req.reply.send(reply);
            }
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
            if let BotEvent::RosterChanged { .. } = &e {
                if let Some(b) = bridge.as_mut() {
                    b.send_players(&players_message(&bot, cfg.web_names));
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
            }
        }
        if now >= next_log {
            next_log = now + LOG_EVERY;
            tracing::info!(stats = ?bot.stats(), "bot status\n{}", bot.latency().report());
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
/// in-game reconnect cap) is 4, being moved to the spectators after having played is 3 like a kick, a
/// requested stop or no stop at all is 0, anything else (local errors, protocol violations, redirect
/// loops) is 1. None of them is ever retried by the runner.
pub fn exit_code_for(category: Option<GaveUpCategory>) -> u8 {
    match category {
        Some(GaveUpCategory::KickedOrBanned) => EXIT_KICKED_OR_BANNED,
        Some(
            GaveUpCategory::HandshakeTimeout
            | GaveUpCategory::ReconnectLoop
            | GaveUpCategory::TooManyAttempts
            | GaveUpCategory::ReconnectBudgetExhausted,
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
        } => bot.note_wire_latency(since_snapshot, tick, tag),
        ClientEvent::Session(s) => match *s {
            SessionEvent::InputSent { tick, input } => bot.on_input_sent(tick, &input),
            SessionEvent::InputTiming { tick, time_left } => bot.on_input_timing(tick, time_left),
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
    if let Some(team) = out.set_team {
        client.set_team(team);
    }
}

fn log_event(e: &BotEvent) {
    match e {
        BotEvent::Killed { tick, reason } => tracing::info!(tick, ?reason, "unstick: Cl_Kill"),
        BotEvent::Block { tick, victim } => tracing::info!(tick, victim, "block"),
        BotEvent::BlockedBy { tick, by } => tracing::info!(tick, by, "blocked by"),
        BotEvent::TargetChanged { tick, to } => tracing::info!(tick, target = ?to, "target"),
        BotEvent::Respawned { tick } => tracing::info!(tick, "life started"),
        BotEvent::Joining { tick } => tracing::info!(tick, "in the spectators: asking to join"),
        BotEvent::JoinGaveUp { tick } => tracing::warn!(tick, "still a spectator after all tries: not asking again"),
        BotEvent::MovedToSpectators { tick } => tracing::error!(tick, "moved to the spectators after having played"),
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
