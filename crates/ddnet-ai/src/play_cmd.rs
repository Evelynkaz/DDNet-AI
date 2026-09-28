//! `ddnet-ai play` (task 2.3): connects to a DDNet 20.x server via `ddai-client`, joins, and
//! sends inputs from one of two trivial built-in "brains" (`idle`/`circle`) — enough to prove the
//! whole join sequence, input timing, and snapshot flow actually work end to end against a real
//! server. The real bot's brain (the fly) is a later phase; this is test/demo tooling.

use clap::{Args, ValueEnum};
use ddai_client::{Client, ClientConfig, ClientEvent, PlayerInput, SessionEvent};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};
use tracing_subscriber::prelude::*;

/// `~/aiddnet/data`, per `CLAUDE.md`'s folder layout — same fallback pattern as `ddai-web`'s CLI.
fn default_data_dir() -> PathBuf {
    match std::env::var_os("HOME") {
        Some(home) if !home.is_empty() => PathBuf::from(home).join("aiddnet").join("data"),
        _ => PathBuf::from("data"),
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Brain {
    /// Sends neutral inputs the whole time (no movement) — proves the join sequence and
    /// input-timing loop work without the server ever seeing the tee move.
    Idle,
    /// Walks right, jumps, walks left, jumps, on repeat — proves inputs actually move the tee
    /// (task acceptance criterion 5/e2e scenario b: the tee's own position in snapshots changes).
    Circle,
}

#[derive(Debug, Args)]
pub struct PlayArgs {
    /// Server to connect to (game port, e.g. `127.0.0.1:8303`) — per CLAUDE.md's live-play
    /// policy this task only ever points this at 127.0.0.1.
    #[arg(long)]
    pub server: SocketAddr,
    /// Name to send in `Cl_StartInfo`.
    #[arg(long, default_value = "ddai-bot")]
    pub name: String,
    #[arg(long, value_enum, default_value = "idle")]
    pub brain: Brain,
    /// How long to stay connected, in seconds, before disconnecting gracefully.
    #[arg(long, default_value_t = 30)]
    pub duration: u64,
    /// Base data directory (maps cache under `<data-dir>/maps/cache`, logs under
    /// `<data-dir>/logs/play`). Defaults to `~/aiddnet/data`.
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    /// Connection silence timeout, in seconds, before a lost connection is detected and a
    /// reconnect is attempted — defaults to the real DDNet client's own 100s
    /// (`conn_timeout`/`ddai_net::conn::DEFAULT_TIMEOUT`). Test/tooling knob: e2e scenario (d)
    /// (server restart) passes a much shorter value so the test does not have to wait 100s for
    /// the timeout to fire.
    #[arg(long)]
    pub timeout_secs: Option<u64>,
}

fn init_tracing(data_dir: &Path) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let stderr_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);

    let log_dir = data_dir.join("logs").join("play");
    match std::fs::create_dir_all(&log_dir) {
        Ok(()) => {
            let file_appender = tracing_appender::rolling::daily(&log_dir, "play.log");
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

fn idle_input() -> PlayerInput {
    PlayerInput {
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

/// A short (1.6s) cycle: walk right (0.6s) -> jump while still moving right (0.2s) -> walk left
/// (0.6s) -> jump while still moving left (0.2s) -> repeat. Deliberately short: this is a block
/// map (the entire point of the map is to throw a careless tee into freeze), and this brain does
/// not avoid that at all — a short cycle gives the acceptance-criterion proof ("moves right, then
/// left; jumps") a real chance to complete at least once before that happens, verified live
/// against `ddnet-local` (see the crate's BUILD REPORT).
fn circle_input(elapsed: Duration) -> PlayerInput {
    const PERIOD_MS: u128 = 1600;
    let t = elapsed.as_millis() % PERIOD_MS;
    let (direction, jump) = match t {
        0..=599 => (1, 0),
        600..=799 => (1, 1),
        800..=1399 => (-1, 0),
        _ => (-1, 1),
    };
    PlayerInput {
        direction,
        target_x: direction * 32,
        target_y: -32,
        jump,
        fire: 0,
        hook: 0,
        player_flags: ddai_client::enums::playerflagflag::PLAYING,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
    }
}

pub fn run(args: PlayArgs) -> ExitCode {
    let data_dir = args.data_dir.clone().unwrap_or_else(default_data_dir);
    let _tracing_guard = init_tracing(&data_dir);

    let config = ClientConfig {
        name: args.name.clone(),
        cache_dir: data_dir.join("maps").join("cache"),
        timeout: args
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(ClientConfig::default().timeout),
        ..ClientConfig::default()
    };

    tracing::info!(server = %args.server, name = %args.name, brain = ?args.brain, duration = args.duration, "connecting");
    let mut client = Client::connect(args.server, config);

    let start = Instant::now();
    let duration = Duration::from_secs(args.duration);
    let mut in_game = false;
    let mut ended = false;
    let mut last_own_position_log: Option<Instant> = None;

    while !ended && start.elapsed() < duration {
        if let Some(ev) = client.recv_event(Duration::from_millis(50)) {
            log_event(ev, &mut in_game, &mut ended, &mut last_own_position_log);
        }

        if in_game {
            let input = match args.brain {
                Brain::Idle => idle_input(),
                Brain::Circle => circle_input(start.elapsed()),
            };
            client.set_input(input);
        }
    }

    if !ended {
        tracing::info!("duration elapsed, disconnecting");
        client.disconnect();
    }
    client.join();
    // The driver's final events (notably `MarginSummary`, sent right before its thread returns —
    // task e2e scenario h) are still sitting in the channel at this point: `join()` only waits
    // for the thread, it does not drain what it already sent. Draining here (non-blocking:
    // `try_iter` simply stops once the queue is empty) makes sure they still reach the log.
    for ev in client.events() {
        log_event(ev, &mut in_game, &mut ended, &mut last_own_position_log);
    }
    ExitCode::SUCCESS
}

fn log_event(ev: ClientEvent, in_game: &mut bool, ended: &mut bool, last_own_position_log: &mut Option<Instant>) {
    match ev {
        ClientEvent::Session(ev) => match *ev {
            SessionEvent::Connected => tracing::info!("connected"),
            SessionEvent::MapChanging { name, size, .. } => tracing::info!(map = %name, size, "map changing"),
            SessionEvent::MapLoaded(loaded) => {
                tracing::info!(map = %loaded.name, source = ?loaded.source, "map loaded")
            }
            SessionEvent::InGame => {
                *in_game = true;
                tracing::info!("in game");
            }
            SessionEvent::Snapshot { tick } => tracing::debug!(tick, "snapshot"),
            SessionEvent::Tuning(_) => tracing::debug!("tuning received"),
            SessionEvent::Disconnected { reason, by_peer } => {
                tracing::warn!(reason = ?reason, by_peer, "disconnected");
                // Whether this is retryable is the *driver's* decision to make, not a guess this
                // app-level loop makes from `by_peer` alone (review finding F2: `by_peer: true`
                // used to always mean "final" here, which broke the moment the driver started
                // reconnecting from some `by_peer: true` reasons too, e.g. a graceful "Server
                // shutdown" — see `crate::driver::should_reconnect_after_peer_close`). The one
                // authoritative "this session is over, stop the loop" signal is
                // `ClientEvent::GaveUp` (below); until that arrives, the driver may still be
                // retrying in the background, so ending this loop here (and, worse, skipping the
                // `client.disconnect()` call below because `ended` looked true) would abandon a
                // session the driver might still bring back, and `client.join()` would then block
                // for as long as the driver keeps retrying instead of the intended `--duration`.
                *in_game = false;
            }
            SessionEvent::ProtocolViolation { reason } => {
                // Always final at the driver level (a bad/hostile server, not a lost connection —
                // see `SessionEvent::ProtocolViolation`'s docs), but this loop still only reacts
                // to the authoritative `ClientEvent::GaveUp` below rather than setting `ended`
                // itself, for the same reason as the `Disconnected` arm above.
                tracing::error!(%reason, "protocol violation — this connection will not be retried");
                *in_game = false;
            }
            SessionEvent::Anomaly(msg) => tracing::warn!(%msg, "anomaly"),
            other => tracing::debug!(?other, "event"),
        },
        ClientEvent::ReconnectAttempt { attempt, addr, backoff } => {
            tracing::warn!(attempt, %addr, ?backoff, "reconnecting");
        }
        ClientEvent::RedirectFollowed { to } => tracing::info!(%to, "following redirect"),
        ClientEvent::RedirectRefused { reason } => tracing::error!(%reason, "redirect refused"),
        ClientEvent::GaveUp { reason } => {
            tracing::error!(%reason, "driver gave up");
            *ended = true;
        }
        ClientEvent::OwnPosition { tick, x, y } => {
            // Review finding F11: fires once per snapshot (up to ~50/s while in-game) — logging
            // every single one at INFO would flood the log over any real (long) play session.
            // Deliberately *not* demoted to DEBUG: `tools/e2e/session.sh`'s scenario (b) runs with
            // `RUST_LOG=info` and `tools/e2e/analyze_positions.py` parses these lines to prove
            // movement (including, per review finding F12, a brief ~200ms jump window) — DEBUG
            // would silently starve it of samples. Rate-limited to at most 10/s instead: a solid
            // 5x+ reduction over the real snapshot rate, while still comfortably sampling a 200ms
            // window multiple times.
            let now = Instant::now();
            let should_log =
                last_own_position_log.is_none_or(|last| now.duration_since(last) >= Duration::from_millis(100));
            if should_log {
                *last_own_position_log = Some(now);
                tracing::info!(tick, x, y, "own position");
            }
        }
        ClientEvent::LiveWorldSnapshot(snap) => {
            // Task 2.4's `LiveWorld` is this event's real consumer; this demo/test tool has none
            // (see this file's module doc comment) — logged at DEBUG only, same reasoning as
            // `OwnPosition` above (fires once per snapshot, up to ~50/s).
            tracing::debug!(
                tick = snap.tick,
                characters = snap.characters.len(),
                "live-world snapshot"
            );
        }
        ClientEvent::MarginSummary(s) => {
            tracing::info!(
                count = s.count,
                late_count = s.late_count,
                late_fraction = s.late_fraction,
                mean_ms = s.mean_ms,
                min_ms = ?s.min_ms,
                p50_ms = ?s.p50_ms,
                p90_ms = ?s.p90_ms,
                p99_ms = ?s.p99_ms,
                max_ms = ?s.max_ms,
                "margin summary"
            );
        }
    }
}
