//! `ddnet-ai record` (task 8.4a acceptance criterion 1): connects to a DDNet 20.x server via
//! `ddai-client` as a pure observer — never sends non-neutral input or chat, requests
//! `TEAM_SPECTATORS` after entering — and records every snapshot/game message into rec v1
//! (`ddai_recorder::writer::RecordingWriter`). D-027/D-038's safety switch
//! (`ddai_client::live_servers`) is checked *before* ever connecting: a non-loopback address not
//! on the owner-curated allow-list, or one requested under the wrong nick, is refused outright.

use clap::Args;
use ddai_client::view::View;
use ddai_client::{Client, ClientConfig, ClientEvent, GaveUpCategory, SessionEvent, live_servers, single_instance};
use ddai_net::generated::messages::{ExGameMsg, GameMsg};
use ddai_net::snapshot::Snapshot;
use ddai_recorder::format::{CharacterRecord, Frame, Header, PlayerRecord, RecordedGameMessage};
use ddai_recorder::writer::RecordingWriter;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing_subscriber::prelude::*;

/// Review round 1, finding F8: a distinct process exit code for "the server itself kicked or
/// banned us" — never `0` (D-016: a human should notice this, not have automation see a quiet
/// success) and never the same as the generic `ExitCode::FAILURE` (`1`) every other failure path
/// here already uses, so a caller (a supervisor script, `docs/STATUS.md`'s own incident log) can
/// tell the two apart without parsing log text.
const EXIT_KICKED_OR_BANNED: u8 = 3;
/// Task 2.3b: a distinct, non-zero exit code for the handshake watchdog giving up — same
/// motivation as [`EXIT_KICKED_OR_BANNED`] (D-016: a human should notice and investigate a stalled
/// session, not have automation see a quiet, ambiguous failure). This is exactly the incident this
/// task fixes: a session that never reached in-game, silently retried forever, and had to be
/// stopped by hand after 35s with no explanation in the log.
const EXIT_HANDSHAKE_TIMEOUT: u8 = 4;

/// `~/aiddnet/data`, per `CLAUDE.md`'s folder layout — same fallback pattern as `ddnet-ai play`.
fn default_data_dir() -> PathBuf {
    match std::env::var_os("HOME") {
        Some(home) if !home.is_empty() => PathBuf::from(home).join("aiddnet").join("data"),
        _ => PathBuf::from("data"),
    }
}

#[derive(Debug, Args)]
pub struct RecordArgs {
    /// Server to connect to (game port). Loopback addresses are always allowed; any other
    /// address must be listed in `--live-servers` under exactly `--name` (D-027/D-038).
    #[arg(long)]
    pub server: SocketAddr,
    /// Nick to record as — the observer's own `Cl_StartInfo` name. D-027/D-038 pin this to
    /// "Muha" for the one server this project is actually approved to record on; the default
    /// here matches that, but any nick is accepted for loopback testing.
    #[arg(long, default_value = "Muha")]
    pub name: String,
    /// How long to record, in seconds, before disconnecting gracefully.
    #[arg(long, default_value_t = 30)]
    pub duration: u64,
    /// Directory the recording file is written into (task usage: `--out
    /// ~/aiddnet/data/recordings/<date>/`) — created if it does not exist. The file itself is
    /// named `<map>-<unix-ms>.rec` once the map is known (see the module docs on why the name
    /// can't be chosen before that).
    #[arg(long)]
    pub out: PathBuf,
    /// Base data directory (maps cache under `<data-dir>/maps/cache`, logs under
    /// `<data-dir>/logs/record`). Defaults to `~/aiddnet/data`.
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    /// Connection silence timeout, in seconds — see `ddnet-ai play`'s identical flag.
    #[arg(long)]
    pub timeout_secs: Option<u64>,
    /// Overrides the live-servers allow-list path (testing only) — defaults to
    /// `ddai_client::live_servers::LiveServers::default_path`
    /// (`~/aiddnet/data/live-servers.toml`).
    #[arg(long)]
    pub live_servers: Option<PathBuf>,
    /// `Cl_ShowDistance`'s half-extents, in world units, both axes — D-031: "maximum
    /// `Cl_ShowDistance` (почти вся карта видна)". Large enough that no real DDNet block map (a
    /// few thousand tiles at most) can exceed it from any point within it.
    #[arg(long, default_value_t = 2_000_000)]
    pub show_distance: i32,
    /// Task 8.4a acceptance criterion 4's outgoing-input audit: if set, appends one JSON line per
    /// `NETMSG_INPUT` this process actually sends (identical format to `ddnet-ai play
    /// --input-log`) — every line must show an all-neutral input for a correct observer.
    #[arg(long)]
    pub input_log: Option<PathBuf>,
}

fn init_tracing(data_dir: &Path) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let stderr_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);

    let log_dir = data_dir.join("logs").join("record");
    match std::fs::create_dir_all(&log_dir) {
        Ok(()) => {
            let file_appender = tracing_appender::rolling::daily(&log_dir, "record.log");
            let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
            let file_layer = tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false);
            let _ = tracing_subscriber::registry()
                .with(filter)
                .with(stderr_layer)
                .with(file_layer)
                .try_init();
            Some(guard)
        }
        Err(e) => {
            eprintln!(
                "warning: could not create log directory {} ({e}); logging to stderr only",
                log_dir.display()
            );
            let _ = tracing_subscriber::registry()
                .with(filter)
                .with(stderr_layer)
                .try_init();
            None
        }
    }
}

/// Always-neutral input (task acceptance criterion 1: "it never sends input other than
/// neutral") — set once after connecting and left alone; this crate never has any code path that
/// could compute anything else, unlike `ddnet-ai play`'s brains.
fn neutral_input() -> ddai_client::PlayerInput {
    ddai_client::PlayerInput {
        direction: 0,
        target_x: 0,
        target_y: -1,
        jump: 0,
        fire: 0,
        hook: 0,
        player_flags: ddai_client::enums::playerflagflag::PLAYING,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
    }
}

/// Every character/player in `snapshot`, converted to this crate's own recorded shape — a
/// mechanical field-for-field copy of `ddai_net::view::{CharacterView,PlayerView}` (same field
/// names by construction — `ddai_recorder::format`'s doc comments explain why).
fn snapshot_to_frame(tick: i32, snapshot: &Snapshot) -> Frame {
    let view = View::new(snapshot);
    let characters = view
        .characters()
        .into_iter()
        .map(|c| CharacterRecord {
            id: c.id,
            character: c.character,
            ddnet: c.ddnet,
        })
        .collect();
    let players = view
        .players()
        .into_iter()
        .map(|p| PlayerRecord {
            id: p.id,
            info: p.info,
            client_info: p.client_info,
            ddnet: p.ddnet,
        })
        .collect();
    Frame::Snapshot {
        tick,
        characters,
        players,
    }
}

/// Task acceptance criterion 2's curated set (kill/broadcast/chat) plus the `Other` catch-all —
/// see `RecordedGameMessage`'s own docs for why anything else lands there rather than being
/// dropped.
fn game_msg_to_recorded(msg: &GameMsg) -> RecordedGameMessage {
    match msg {
        GameMsg::SvKillMsg(k) => RecordedGameMessage::Kill {
            killer: k.killer,
            victim: k.victim,
            weapon: k.weapon,
            mode_special: k.mode_special,
        },
        GameMsg::SvBroadcast(b) => RecordedGameMessage::Broadcast {
            message: b.message.clone(),
        },
        GameMsg::SvChat(c) => RecordedGameMessage::Chat {
            team: c.team,
            client_id: c.client_id,
            message: c.message.clone(),
        },
        other => RecordedGameMessage::Other {
            debug: format!("{other:?}"),
        },
    }
}

fn ex_game_msg_to_recorded(msg: &ExGameMsg) -> RecordedGameMessage {
    RecordedGameMessage::Other {
        debug: format!("{msg:?}"),
    }
}

/// Turns `name` into a safe filename component: ASCII alphanumerics and `-`/`_` pass through
/// unchanged, everything else (spaces, unicode, punctuation — real map names have all three,
/// e.g. "Copy Love Box") becomes `_`. Not `ddai_client::map_cache::is_valid_map_filename` (that
/// checks whether a *server-supplied* name is safe to use in a path at all — a different,
/// stricter, security-relevant question this module doesn't need to answer, since this filename
/// is for *our own* recording file, and any input reaching here already passed that check on the
/// client-session side before the map could even load).
fn sanitize_filename_component(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn unix_ms_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn run(args: RecordArgs) -> ExitCode {
    let data_dir = args.data_dir.clone().unwrap_or_else(default_data_dir);
    let _tracing_guard = init_tracing(&data_dir);

    // Review round 1, finding F8: refuse a second instance under the same identity against the
    // same server outright, before anything else — shared with `ddnet-ai play` (same lock scheme,
    // keyed on (address, name) — see `single_instance`'s own doc comment for why name is part of
    // the key: an address-only lock broke this project's own local multi-bot e2e tests). Held for
    // the rest of this function's lifetime; released automatically (by the OS) on any exit path,
    // including a `kill -9`.
    let _server_lock = match single_instance::acquire(args.server, &args.name) {
        Ok(lock) => lock,
        Err(e) => {
            tracing::error!(error = %e, "refusing to start");
            eprintln!("refusing to start: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Safety switch (task acceptance criterion 5, D-027/D-038): an early, friendly check against
    // the *initial* `--server` argument, so a plainly-wrong invocation fails fast with a clear
    // message before creating directories/a socket at all. This is **not** the only enforcement:
    // review round 1, finding F1 — `ddai_client::driver` itself now re-checks this same allow-list
    // before *every* connect it ever makes (including a redirect target this early check can
    // never see in advance), via `ClientConfig::live_servers` below.
    let live_servers_path = args
        .live_servers
        .clone()
        .unwrap_or_else(live_servers::LiveServers::default_path);
    let list = match live_servers::LiveServers::load_or_empty(&live_servers_path) {
        Ok(list) => list,
        Err(e) => {
            tracing::error!(error = %e, path = %live_servers_path.display(), "failed to load live-servers.toml");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = live_servers::check(args.server, &args.name, &list) {
        tracing::error!(error = %e, server = %args.server, name = %args.name, "refusing to connect: live-servers safety switch");
        eprintln!("refusing to connect: {e}");
        return ExitCode::FAILURE;
    }

    if let Err(e) = std::fs::create_dir_all(&args.out) {
        tracing::error!(error = %e, out = %args.out.display(), "failed to create --out directory");
        return ExitCode::FAILURE;
    }

    let mut config = ClientConfig {
        name: args.name.clone(),
        cache_dir: data_dir.join("maps").join("cache"),
        show_distance: (args.show_distance, args.show_distance),
        // Review round 1, finding F13: `1`, not the real client's own `0` default (see
        // `ClientConfig::show_others`'s doc comment for why this crate's `play` keeps `0`).
        show_others: 1,
        timeout: args
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(ClientConfig::default().timeout),
        emit_snapshot_data: true,
        // Review round 1, finding F7: only pay for `SessionEvent::InputSent` when `--input-log`
        // was actually requested.
        emit_input_sent: args.input_log.is_some(),
        // Review round 1, finding F1: enforced by the driver on every connect attempt from here
        // on, not just the one-shot check above.
        live_servers: list,
        ..ClientConfig::default()
    };
    // Task 2.6: a proxy only if this server's allow-list entry names one (the driver re-checks every attempt).
    if let Err(e) = crate::proxy_cmd::attach_proxy(&mut config, args.server, &data_dir) {
        tracing::error!(error = %e, "refusing to connect: proxy");
        eprintln!("refusing to connect: {e}");
        return ExitCode::from(e.exit);
    }
    let client_version = config.version_str.clone();

    let mut input_log = match &args.input_log {
        Some(path) => match std::fs::File::create(path) {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!(
                    "warning: could not create --input-log {} ({e}); not logging inputs",
                    path.display()
                );
                None
            }
        },
        None => None,
    };

    // Review round 1, finding F8: graceful shutdown on SIGINT/SIGTERM — without this, a `kill`
    // (or an operator's Ctrl-C) skips `client.disconnect()` entirely, leaving the slot connected
    // server-side until the silence timeout, and abandons the current recording segment mid-chunk
    // with no sha256 sidecar. A failure to install the handler is logged but not fatal (the
    // process still works, it just falls back to the old "no graceful signal handling" behavior).
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&shutdown_requested);
        if let Err(e) = ctrlc::set_handler(move || {
            flag.store(true, Ordering::SeqCst);
        }) {
            tracing::warn!(error = %e, "failed to install a SIGINT/SIGTERM handler");
        }
    }

    tracing::info!(server = %args.server, name = %args.name, duration = args.duration, "connecting (observer)");
    let mut client = Client::connect(args.server, config);
    client.set_input(neutral_input());

    let start = Instant::now();
    let duration = Duration::from_secs(args.duration);
    let mut ended = false;
    // Review round 1, finding F4: distinct from `ended` — `ended` only means "this function's own
    // loop should stop", which can become true for reasons the *driver* knows nothing about (e.g.
    // failing to create a recording file below); `driver_gave_up` means the driver thread is
    // already stopping on its own (a `GaveUp` event), which is the one and only condition under
    // which skipping `client.disconnect()` below is correct.
    let mut driver_gave_up = false;
    let mut gave_up_category: Option<GaveUpCategory> = None;
    let mut writer: Option<RecordingWriter> = None;
    let mut any_segment_created = false;
    let mut spectate_requested = false;
    let mut spectate_confirmed = false;
    let mut spectate_grace_deadline: Option<Instant> = None;
    let mut last_snapshot_tick: i32 = 0;
    // Review round 1, finding F1: the header names whichever address we are *actually* connected
    // to right now, not just the original `--server` argument — a redirect changes this.
    let mut current_server_addr = args.server;

    while !ended && start.elapsed() < duration && !shutdown_requested.load(Ordering::SeqCst) {
        let Some(ev) = client.recv_event(Duration::from_millis(50)) else {
            check_spectate_grace(&mut spectate_grace_deadline, spectate_confirmed);
            continue;
        };
        // Never anything but neutral — see `neutral_input`'s doc comment. Re-asserted every loop
        // iteration so no future change to this function's control flow could accidentally start
        // conditionally skipping it.
        client.set_input(neutral_input());

        match ev {
            ClientEvent::Session(session_ev) => match *session_ev {
                SessionEvent::Connected => {
                    tracing::info!("connected");
                    // Review round 1, finding F9: reset here, on *every* fresh connection —
                    // initial, reconnect, or redirect — rather than only on
                    // `ClientEvent::ReconnectAttempt`, which a server-requested reconnect or a
                    // redirect never emits at all (those go straight from the old connection to a
                    // new one without an "attempt" event in between). `Connected` fires exactly
                    // once per connection, unconditionally, so it is the one reliable place to
                    // know "this is a brand new join sequence, TEAM_SPECTATORS has not been
                    // requested on it yet".
                    spectate_requested = false;
                    spectate_confirmed = false;
                }
                SessionEvent::MapChanging { name, size, .. } => tracing::info!(map = %name, size, "map changing"),
                SessionEvent::MapLoaded(loaded) => {
                    tracing::info!(map = %loaded.name, source = ?loaded.source, "map loaded");
                    // A mid-session map change (task e2e criterion) starts a *new* recording
                    // file/segment rather than continuing the old one: rec v1's header names one
                    // map for the whole file (task acceptance criterion 2), so a file spanning two
                    // different maps would have every frame after the change describe a map the
                    // header no longer names — finishing the old segment and starting a fresh one
                    // keeps every single `.rec` file self-consistent instead.
                    finish_writer(writer.take());
                    let filename = format!("{}-{}.rec", sanitize_filename_component(&loaded.name), unix_ms_now());
                    let path = args.out.join(filename);
                    let header = Header {
                        server_address: current_server_addr.to_string(),
                        map_name: loaded.name.clone(),
                        map_sha256: loaded.sha256,
                        client_version: client_version.clone(),
                        start_time_unix_ms: unix_ms_now(),
                        observer_nick: args.name.clone(),
                    };
                    match RecordingWriter::create(&path, &header) {
                        Ok(w) => {
                            tracing::info!(path = %path.display(), "recording segment started");
                            writer = Some(w);
                            any_segment_created = true;
                        }
                        Err(e) => {
                            tracing::error!(error = %e, path = %path.display(), "failed to create recording file");
                            ended = true;
                        }
                    }
                }
                SessionEvent::InGame => {
                    tracing::info!("in game");
                    if !spectate_requested {
                        client.set_team(ddai_client::enums::team::SPECTATORS);
                        spectate_requested = true;
                        spectate_grace_deadline = Some(Instant::now() + Duration::from_secs(10));
                        tracing::info!("requested TEAM_SPECTATORS (Cl_SetTeam)");
                    }
                }
                SessionEvent::SnapshotData { tick, snapshot } => {
                    last_snapshot_tick = tick;
                    if let Some(w) = &mut writer {
                        let frame = snapshot_to_frame(tick, &snapshot);
                        if let Err(e) = w.write_frame(&frame) {
                            tracing::error!(error = %e, "failed to write snapshot frame — recording may be incomplete");
                        }
                    }
                }
                SessionEvent::Snapshot { .. } => {} // superseded by SnapshotData above
                SessionEvent::GameMessage(msg) => {
                    if let Some(w) = &mut writer {
                        let frame = Frame::GameEvent {
                            tick_hint: last_snapshot_tick,
                            message: game_msg_to_recorded(&msg),
                        };
                        if let Err(e) = w.write_frame(&frame) {
                            tracing::error!(error = %e, "failed to write game-message frame");
                        }
                    }
                }
                SessionEvent::ExGameMessage(msg) => {
                    if let Some(w) = &mut writer {
                        let frame = Frame::GameEvent {
                            tick_hint: last_snapshot_tick,
                            message: ex_game_msg_to_recorded(&msg),
                        };
                        if let Err(e) = w.write_frame(&frame) {
                            tracing::error!(error = %e, "failed to write ex-game-message frame");
                        }
                    }
                }
                SessionEvent::Tuning(t) => {
                    if let Some(w) = &mut writer {
                        let frame = Frame::GameEvent {
                            tick_hint: last_snapshot_tick,
                            message: RecordedGameMessage::Tuning(t),
                        };
                        if let Err(e) = w.write_frame(&frame) {
                            tracing::error!(error = %e, "failed to write tuning frame");
                        }
                    }
                }
                SessionEvent::Disconnected { reason, by_peer } => {
                    tracing::warn!(reason = ?reason, by_peer, "disconnected");
                }
                SessionEvent::ProtocolViolation { reason } => {
                    tracing::error!(%reason, "protocol violation — this connection will not be retried");
                }
                SessionEvent::Anomaly(msg) => tracing::warn!(%msg, "anomaly"),
                SessionEvent::InputSent { tick, input } => {
                    tracing::trace!(tick, ?input, "input sent");
                    if let Some(file) = input_log.as_mut() {
                        log_input(file, tick, &input);
                    }
                    debug_assert_eq!(input, neutral_input(), "the recorder must never send non-neutral input");
                }
                other => tracing::debug!(?other, "event"),
            },
            ClientEvent::OwnTeam { tick, team } => {
                // Review round 3, finding F21: round 2's fix logged "will retry" here but the
                // retry could never actually fire — `should_retry_set_team` required
                // `spectate_grace_deadline.is_some()`, and confirmation always sets that deadline
                // to `None` (right below, unchanged), so a demotion that follows a real
                // confirmation always found the deadline already cleared and silently sent
                // nothing, for the rest of the session, while the log claimed otherwise. Per the
                // reviewer's own explicit decision: the whole retry path is deleted, not repaired
                // — staying idle with neutral input after a demotion is already safe (task
                // acceptance criterion 1: "if spectating is refused ... stays idle and says so"),
                // and correctly reopening a bounded retry window would need real design work this
                // round chose not to spend on a case round 2's own `record.sh`/`session.sh` live
                // testing never actually observed on a real server. `own_team_outcome` is the pure
                // state transition (see its own doc comment for why it's factored out this way).
                let (outcome, new_confirmed) = own_team_outcome(spectate_confirmed, team);
                match outcome {
                    OwnTeamOutcome::Confirmed if !spectate_confirmed => {
                        tracing::info!(tick, "spectating confirmed (team == TEAM_SPECTATORS)");
                    }
                    OwnTeamOutcome::Confirmed => {}
                    OwnTeamOutcome::Demoted => {
                        tracing::warn!(tick, team, "demoted from spectators — staying idle, not re-requesting");
                    }
                    OwnTeamOutcome::StillWaiting => {
                        tracing::debug!(tick, team, "own team");
                    }
                }
                spectate_confirmed = new_confirmed;
                if spectate_confirmed {
                    spectate_grace_deadline = None;
                }
            }
            ClientEvent::ReconnectAttempt { attempt, addr, backoff } => {
                tracing::warn!(attempt, %addr, ?backoff, "reconnecting");
            }
            ClientEvent::RedirectFollowed { to } => {
                tracing::info!(%to, "following redirect");
                current_server_addr = to;
            }
            ClientEvent::RedirectRefused { reason } => tracing::error!(%reason, "redirect refused"),
            // Task 2.3b (root-cause fix): this transition used to be completely silent (no
            // `ClientEvent` at all), which is exactly how the Swarfey incident's rapid, unbounded
            // reconnect loop went unnoticed at info level — see this event's own doc comment.
            ClientEvent::ServerRequestedReconnect { addr, attempt } => {
                tracing::info!(%addr, attempt, "server requested reconnect (reconnect@ddnet.org)");
            }
            ClientEvent::GaveUp { reason, category } => {
                tracing::error!(%reason, ?category, "driver gave up");
                ended = true;
                driver_gave_up = true;
                gave_up_category = Some(category);
            }
            ClientEvent::OwnPosition { .. } => {} // a spectator has no character; not expected, harmless if seen
            // Task 2.4's `LiveWorld` is this event's real consumer; the observer recorder has none
            // (it records raw snapshots into rec v1 via `SessionEvent::SnapshotData` instead, see
            // above) — ignored here, exactly like `ddnet-ai play`'s own handling of this event.
            ClientEvent::LiveWorldSnapshot(_) | ClientEvent::InputLatency { .. } => {}
            ClientEvent::MarginSummary(s) => {
                tracing::info!(
                    count = s.count,
                    late_count = s.late_count,
                    late_fraction = s.late_fraction,
                    mean_ms = s.mean_ms,
                    "margin summary"
                );
            }
        }

        check_spectate_grace(&mut spectate_grace_deadline, spectate_confirmed);
    }

    if shutdown_requested.load(Ordering::SeqCst) {
        tracing::info!("shutdown requested (SIGINT/SIGTERM), disconnecting");
    }
    // Review round 1, finding F4: unconditionally disconnect unless the driver thread has *already*
    // ended on its own (`driver_gave_up`) — `ended` alone is not that signal (see its own comment
    // above): skipping this call whenever *this* loop merely decided to stop (duration elapsed,
    // SIGINT/SIGTERM, a local error such as failing to create a recording file) left the driver
    // thread fully connected and running forever, and `client.join()` right below would then hang
    // waiting for a thread nothing had told to stop.
    if !driver_gave_up {
        client.disconnect();
    }
    client.join();
    for ev in client.events() {
        // Final drain: only a handful of event kinds (`MarginSummary`, a last snapshot or two)
        // are expected here — reuse the exact same handling as the loop by re-dispatching through
        // a tiny local closure would duplicate a lot; instead just log what shows up, matching
        // `ddnet-ai play`'s own final-drain comment on why this still matters (the driver's last
        // events are only guaranteed queued *after* `join()` returns).
        tracing::debug!(?ev, "final drained event");
    }

    finish_writer(writer.take());

    // Review round 1, finding F8: a kick/ban gets its own distinct, non-zero exit code — checked
    // first, since it is the one outcome worth a human's attention regardless of whether a
    // recording also happened to be produced. Task 2.3b: the handshake watchdog giving up gets the
    // same treatment, for the same reason.
    if matches!(gave_up_category, Some(GaveUpCategory::KickedOrBanned)) {
        return ExitCode::from(EXIT_KICKED_OR_BANNED);
    }
    if matches!(
        gave_up_category,
        Some(
            GaveUpCategory::HandshakeTimeout
                | GaveUpCategory::ReconnectLoop
                | GaveUpCategory::TooManyAttempts
                | GaveUpCategory::ReconnectBudgetExhausted
                | GaveUpCategory::ProxyRefused
        )
    ) {
        return ExitCode::from(EXIT_HANDSHAKE_TIMEOUT);
    }
    match any_segment_created {
        true => ExitCode::SUCCESS,
        false => {
            tracing::warn!("no map was ever loaded — nothing was recorded");
            ExitCode::FAILURE
        }
    }
}

/// Finishes one recording segment, if any (a no-op given `None` — every call site passes
/// `writer.take()`, so this is safe to call unconditionally, including once at the very end of a
/// session that never saw a map change at all). Logs the segment's summary or failure; never
/// panics or propagates an error further — a failure to finish a *previous* segment must not stop
/// the observer from continuing to record the next one.
fn finish_writer(writer: Option<RecordingWriter>) {
    let Some(w) = writer else {
        return;
    };
    match w.finish() {
        Ok(summary) => {
            let hex: String = summary.whole_file_sha256.iter().map(|b| format!("{b:02x}")).collect();
            tracing::info!(
                frames_written = summary.frames_written,
                bytes_written = summary.bytes_written,
                sha256 = %hex,
                "recording segment finished"
            );
        }
        Err(e) => tracing::error!(error = %e, "failed to finish a recording segment"),
    }
}

/// Review round 3, finding F21: one `OwnTeam` event's outcome, given only whether spectating was
/// already confirmed *before* it — the retry path round 2 built on top of this (`should_retry_set_
/// team`) is deleted; see [`own_team_outcome`]'s own doc comment for why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OwnTeamOutcome {
    /// Just (re)confirmed as spectating — `team == TEAM_SPECTATORS`.
    Confirmed,
    /// A genuine demotion: was confirmed, now isn't.
    Demoted,
    /// Not confirmed, and wasn't a moment ago either — the ordinary "still waiting" case (either
    /// still inside the very first join's own grace window, or already past it) — nothing new to
    /// report or do.
    StillWaiting,
}

/// Review round 3, finding F21: a small, pure, testable state transition — factored out of
/// `record_cmd::run`'s own `OwnTeam` handler so review round 2's own precedent ("test the pure
/// predicate without a live driver/socket") still holds, but this time also directly usable to
/// drive a whole *sequence* of events and assert on the resulting trajectory (see
/// `record_cmd::tests::drive_own_team_sequence`), per the reviewer's own explicit ask to "test the
/// loop's state transitions, not just the pure predicate" — round 2's 6 tests on
/// `should_retry_set_team` exercised the predicate in isolation, with `grace_deadline_is_some:
/// true` passed directly as a test input, a state the *real* loop can never actually be in after
/// a genuine confirmation (confirming always clears the deadline to `None`, unconditionally,
/// right where `own_team_outcome`'s result is applied) — so the tests never caught that the
/// retry they were unit-testing could never fire live. Returns the outcome plus the new value
/// `spectate_confirmed` must take.
fn own_team_outcome(was_confirmed: bool, team: i32) -> (OwnTeamOutcome, bool) {
    let now_spectating = team == ddai_client::enums::team::SPECTATORS;
    match (was_confirmed, now_spectating) {
        (_, true) => (OwnTeamOutcome::Confirmed, true),
        (true, false) => (OwnTeamOutcome::Demoted, false),
        (false, false) => (OwnTeamOutcome::StillWaiting, false),
    }
}

fn check_spectate_grace(deadline: &mut Option<Instant>, confirmed: bool) {
    let Some(d) = *deadline else { return };
    if confirmed {
        *deadline = None;
        return;
    }
    if Instant::now() >= d {
        // Task acceptance criterion 1: "if spectating is refused or not possible, it stays in
        // game completely idle (neutral input) and says so in the log" — logged here; behavior
        // does not change at all (input was, and remains, always neutral either way — see
        // `neutral_input`'s doc comment), so there is nothing else to do but say so.
        tracing::warn!(
            "TEAM_SPECTATORS was not confirmed within the grace period — spectating may have been \
             refused (spam/kill protection, a mod that requires /pause, which this bot cannot use \
             per D-007) or the server may simply not have sent a fresh PlayerInfo yet; staying in \
             game with neutral input regardless"
        );
        *deadline = None; // only warn once
    }
}

/// Identical format to `play_cmd::log_input` (task 8.4a's `--input-log`) — kept as its own copy
/// rather than a shared helper: the two commands' surrounding types differ enough (this one never
/// has a `Brain` to gate on) that sharing would need a small cross-module helper for one
/// four-line function, which is not worth the indirection.
fn log_input(file: &mut std::fs::File, tick: i32, input: &ddai_client::PlayerInput) {
    use std::io::Write;
    let line = serde_json::json!({
        "tick": tick,
        "direction": input.direction,
        "target_x": input.target_x,
        "target_y": input.target_y,
        "jump": input.jump,
        "fire": input.fire,
        "hook": input.hook,
        "player_flags": input.player_flags,
        "wanted_weapon": input.wanted_weapon,
        "next_weapon": input.next_weapon,
        "prev_weapon": input.prev_weapon,
    });
    if let Err(e) = writeln!(file, "{line}") {
        tracing::warn!(error = %e, "failed to write --input-log line");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_filename_component_replaces_unsafe_characters() {
        assert_eq!(sanitize_filename_component("Copy Love Box"), "Copy_Love_Box");
        assert_eq!(sanitize_filename_component("a/b\\c:d"), "a_b_c_d");
        assert_eq!(sanitize_filename_component("safe-Name_123"), "safe-Name_123");
    }

    // --- Review round 3, finding F21 -------------------------------------------------------------

    #[test]
    fn own_team_outcome_confirms_from_not_confirmed() {
        let (outcome, confirmed) = own_team_outcome(false, ddai_client::enums::team::SPECTATORS);
        assert_eq!(outcome, OwnTeamOutcome::Confirmed);
        assert!(confirmed);
    }

    #[test]
    fn own_team_outcome_is_still_confirmed_when_already_was() {
        let (outcome, confirmed) = own_team_outcome(true, ddai_client::enums::team::SPECTATORS);
        assert_eq!(outcome, OwnTeamOutcome::Confirmed);
        assert!(confirmed);
    }

    #[test]
    fn own_team_outcome_is_a_demotion_only_when_it_was_confirmed_before() {
        let (outcome, confirmed) = own_team_outcome(true, 0);
        assert_eq!(outcome, OwnTeamOutcome::Demoted);
        assert!(!confirmed);
    }

    #[test]
    fn own_team_outcome_is_still_waiting_when_never_confirmed_yet() {
        let (outcome, confirmed) = own_team_outcome(false, 0);
        assert_eq!(outcome, OwnTeamOutcome::StillWaiting);
        assert!(!confirmed);
    }

    /// Review round 3, finding F21's own explicit ask: "add a test of the loop's state
    /// transitions, not just the pure predicate" — round 2's 6 tests on the deleted
    /// `should_retry_set_team` all called that pure function directly with hand-picked inputs,
    /// including `grace_deadline_is_some: true` after a confirmation — a combination the *real*
    /// loop can never actually produce (confirming always clears the deadline to `None` right
    /// where the outcome is applied), which is exactly how round 2 shipped a retry that could
    /// never fire live without any test catching it. This test instead drives a whole *sequence*
    /// of `OwnTeam` events through the same state-threading the real loop does (each step's
    /// `was_confirmed` is the *previous* step's own result, exactly like `spectate_confirmed`
    /// threading through `record_cmd::run`'s loop) and asserts on the resulting trajectory.
    #[test]
    fn own_team_outcome_sequence_matches_the_loops_own_state_threading() {
        const SPEC: i32 = 0; // placeholder for "not TEAM_SPECTATORS" — any non-spectator team id
        let teams = [
            ddai_client::enums::team::SPECTATORS, // first confirmation
            ddai_client::enums::team::SPECTATORS, // reconfirmation, no-op
            SPEC,                                 // a real demotion
            SPEC,                                 // still not confirmed — must NOT be another "demotion"
            ddai_client::enums::team::SPECTATORS, // re-confirmed
        ];
        let mut confirmed = false;
        let mut outcomes = Vec::new();
        for &team in &teams {
            let (outcome, new_confirmed) = own_team_outcome(confirmed, team);
            outcomes.push(outcome);
            confirmed = new_confirmed;
        }
        assert_eq!(
            outcomes,
            vec![
                OwnTeamOutcome::Confirmed,
                OwnTeamOutcome::Confirmed,
                OwnTeamOutcome::Demoted,
                OwnTeamOutcome::StillWaiting,
                OwnTeamOutcome::Confirmed,
            ]
        );
        assert!(confirmed, "the sequence ends on a confirmation");
    }

    /// Companion to the sequence test above: a *permanent* refusal (matching the reviewer's own
    /// live repro, `econ pause_game`) must produce exactly one `Demoted` (the transition itself)
    /// followed only by `StillWaiting` forever after — never another `Demoted`, and critically,
    /// nothing in this crate ever calls `client.set_team` again for any of them (finding F21: the
    /// whole point of deleting the retry path is that *no* outcome here ever triggers a resend —
    /// unlike round 2's `OwnTeamOutcome`-equivalent state, there is no field left to carry a
    /// "send now" signal at all, so this is structurally guaranteed by the return type itself, not
    /// just by the current caller's own choice not to act on one).
    #[test]
    fn a_permanent_refusal_is_one_demotion_then_only_still_waiting() {
        let mut confirmed = true; // was already spectating
        let mut outcomes = Vec::new();
        for _ in 0..50 {
            let (outcome, new_confirmed) = own_team_outcome(confirmed, 0);
            outcomes.push(outcome);
            confirmed = new_confirmed;
        }
        assert_eq!(outcomes[0], OwnTeamOutcome::Demoted);
        assert!(
            outcomes[1..].iter().all(|o| *o == OwnTeamOutcome::StillWaiting),
            "every subsequent refusal must be StillWaiting, never a repeated Demoted: {outcomes:?}"
        );
    }

    #[test]
    fn neutral_input_matches_the_wire_default_shape() {
        let n = neutral_input();
        assert_eq!(n.direction, 0);
        assert_eq!(n.jump, 0);
        assert_eq!(n.fire, 0);
        assert_eq!(n.hook, 0);
    }

    #[test]
    fn game_msg_to_recorded_curates_kill_chat_broadcast_and_falls_back_otherwise() {
        use ddai_net::generated::messages as msgs;
        let kill = game_msg_to_recorded(&GameMsg::SvKillMsg(msgs::SvKillMsg {
            killer: 1,
            victim: 2,
            weapon: 1,
            mode_special: 0,
        }));
        assert!(matches!(
            kill,
            RecordedGameMessage::Kill {
                killer: 1,
                victim: 2,
                ..
            }
        ));

        let chat = game_msg_to_recorded(&GameMsg::SvChat(msgs::SvChat {
            team: 0,
            client_id: 3,
            message: "hi".to_string(),
        }));
        assert!(matches!(chat, RecordedGameMessage::Chat { client_id: 3, .. }));

        let broadcast = game_msg_to_recorded(&GameMsg::SvBroadcast(msgs::SvBroadcast {
            message: "go".to_string(),
        }));
        assert!(matches!(broadcast, RecordedGameMessage::Broadcast { .. }));

        let fallback = game_msg_to_recorded(&GameMsg::SvMotd(msgs::SvMotd {
            message: "welcome".to_string(),
        }));
        assert!(matches!(fallback, RecordedGameMessage::Other { .. }));
    }
}
