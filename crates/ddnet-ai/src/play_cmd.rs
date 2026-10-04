//! `ddnet-ai play` (task 2.3): connects to a DDNet 20.x server via `ddai-client`, joins, and
//! sends inputs from one of three trivial built-in "brains" (`idle`/`circle`/`random-scripted`) —
//! enough to prove the whole join sequence, input timing, and snapshot flow actually work end to
//! end against a real server. The real bot's brain (the fly) is a later phase; this is test/demo
//! tooling. `random-scripted` (task 8.4a) additionally exists to *validate* the observer
//! recorder's offline input reconstruction (`ddnet-ai rec reconstruct`): its `--input-log` writes
//! the exact input embedded in every `NETMSG_INPUT` this process actually sent, tick by tick, as
//! ground truth to compare a reconstruction against.

use clap::{Args, ValueEnum};
use ddai_client::{Client, ClientConfig, ClientEvent, PlayerInput, SessionEvent, single_instance};
use std::fs::File;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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
    /// Task 8.4a: deterministic (seeded by `--seed`), varied inputs — direction/hook held for a
    /// few hundred ms at a time (realistic "held" behavior), jump/fire as brief single-tick
    /// pulses (realistic "tapped" behavior) — see [`random_scripted_input`]. Exists to validate
    /// `ddnet-ai rec reconstruct`'s offline input estimation against known ground truth
    /// (`--input-log`), not to play well.
    RandomScripted,
    /// Task 4.1: the live bot (`ddai-bot`) with the planner brain (debug mode, the arena baseline).
    Planner,
    /// Task 4.1: the live bot with the scripted bot of the phase-0 harness as its brain.
    Scripted,
    /// Task 4.1: the live bot with the hybrid brain (D-041/D-055: exact search with the threat model, 5 ms cap).
    Hybrid,
    /// Task 4.1: the live bot with the fly (untrained until phase 8 delivers a checkpoint).
    Fly,
}

impl Brain {
    /// Whether this brain runs through the full bot pipeline (`ddai-bot`).
    pub fn is_bot_brain(self) -> bool {
        matches!(self, Brain::Planner | Brain::Scripted | Brain::Hybrid | Brain::Fly)
    }
}

/// `--server`: an address, or `auto` (the restricted pick, see [`PlayArgs::server`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerArg {
    Auto,
    Addr(SocketAddr),
}

impl std::str::FromStr for ServerArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let t = s.trim();
        if t.is_empty() || t.eq_ignore_ascii_case("auto") || t.eq_ignore_ascii_case("авто") {
            return Ok(ServerArg::Auto);
        }
        t.parse()
            .map(ServerArg::Addr)
            .map_err(|e| format!("expected `auto` or ip:port: {e}"))
    }
}

/// Resolves `--server`: an address as given, or the restricted auto-pick over the master list (read-only).
fn resolve_server(arg: &ServerArg, name: &str, live_servers: Option<&Path>) -> Result<SocketAddr, String> {
    let ServerArg::Addr(addr) = arg else {
        let path = live_servers.map_or_else(ddai_client::live_servers::LiveServers::default_path, Path::to_path_buf);
        let list = ddai_client::live_servers::LiveServers::load_or_empty(&path).map_err(|e| e.to_string())?;
        let local: SocketAddr = ddai_client::server_list::LOCAL_SERVER
            .parse()
            .map_err(|_| "bad local address")?;
        // The master list is asked for only when a ready allow-listed server exists to choose among.
        let (pick, warning) =
            ddai_client::server_list::pick_auto_fetching(local, name, &list, crate::servers_cmd::fetch_master);
        if let Some(e) = warning {
            eprintln!("server auto: the master list is not available ({e})");
        }
        eprintln!("server auto: {} ({})", pick.addr, pick.why);
        return Ok(pick.addr);
    };
    Ok(*addr)
}

#[derive(Debug, Args)]
pub struct PlayArgs {
    /// Server to connect to (game port, e.g. `127.0.0.1:8303`), or `auto`: the most populated block server
    /// **among the allow-listed ready ones** (`live-servers.toml`, `ready = true`, our nick pinned), else the
    /// local server. `auto` never picks a public server (CLAUDE.md, D-043, D-051, D-052); an explicit
    /// address still has to pass the same allow-list gate in the client.
    #[arg(long)]
    pub server: ServerArg,
    /// Name to send in `Cl_StartInfo`.
    #[arg(long, default_value = "ddai-bot")]
    pub name: String,
    #[arg(long, value_enum, default_value = "idle")]
    pub brain: Brain,
    /// How long to stay connected, in seconds, before disconnecting gracefully. For the bot (`--brain
    /// planner|scripted|hybrid|fly`, `--bot`) `0` means no limit: it runs until SIGINT/SIGTERM, which is what a
    /// systemd unit wants.
    #[arg(long, default_value_t = 30)]
    pub duration: u64,
    /// Base data directory (maps cache under `<data-dir>/maps/cache`, logs under
    /// `<data-dir>/logs/play`). Defaults to `~/aiddnet/data`.
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    /// The live-server allow-list (`live-servers.toml`, D-067). Defaults to `~/aiddnet/data/live-servers.toml`.
    /// A server's `proxy = "<name>"` entry makes the game's UDP go through the SOCKS5 proxy in
    /// `<data-dir>/secrets/<name>-proxy.toml` (task 2.6, D-053 amendment); there is no other way to use a proxy.
    #[arg(long)]
    pub live_servers: Option<PathBuf>,
    /// Connection silence timeout, in seconds, before a lost connection is detected and a
    /// reconnect is attempted — defaults to the real DDNet client's own 100s
    /// (`conn_timeout`/`ddai_net::conn::DEFAULT_TIMEOUT`). Test/tooling knob: e2e scenario (d)
    /// (server restart) passes a much shorter value so the test does not have to wait 100s for
    /// the timeout to fire.
    #[arg(long)]
    pub timeout_secs: Option<u64>,
    /// Seed for `--brain random-scripted`'s deterministic input generator (task 8.4a). Ignored by
    /// every other brain.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Task 8.4a: if set, appends one JSON line per `NETMSG_INPUT` this process actually sends —
    /// `{"tick":..,"direction":..,"target_x":..,"target_y":..,"jump":..,"fire":..,"hook":..,
    /// "player_flags":..,"wanted_weapon":..,"next_weapon":..,"prev_weapon":..}` — the ground truth
    /// `ddnet-ai rec reconstruct --validate` compares a recording's reconstructed inputs against.
    /// Works with any brain, not just `random-scripted`.
    #[arg(long)]
    pub input_log: Option<PathBuf>,
    /// Task 4.1: run through the full bot pipeline (`ddai-bot`: target selection, unstick, filters,
    /// latency measurement, the web bridge) even with `--brain idle`. Implied by `planner`,
    /// `scripted`, `hybrid` and `fly`.
    #[arg(long)]
    pub bot: bool,
    #[command(flatten)]
    pub bot_opts: crate::bot_cmd::BotOpts,
}

const WINDOW_MS: u128 = 240;
/// One real DDNet server tick at 50Hz — review round 3, finding F20: the pulse length and the
/// window boundary are now both counted in whole *input ticks*, not raw elapsed milliseconds (see
/// [`PULSE_TICKS`]'s own doc comment for why milliseconds were the actual bug).
const TICK_MS: u128 = 20;
/// A window (see [`WINDOW_MS`]) is this many input ticks long.
const TICKS_PER_WINDOW: u128 = WINDOW_MS / TICK_MS;
/// Review round 3, finding F20: a jump/fire pulse lasts this many *input ticks* into its window —
/// round 1/2 used 2 ticks (40ms), which live e2e testing (`tools/e2e/record.sh`'s phase 2, see the
/// crate's BUILD REPORT) showed was structurally unreliable: this process's own real-time polling
/// loop (which evaluates [`random_scripted_input`] against raw `Instant::elapsed()`) and the
/// driver's actual ~50Hz per-tick send schedule are two independent clocks with no guaranteed
/// phase relationship, so a 40ms-wide "pulse active" region can contain only a *single* one of the
/// driver's own sends if the two clocks happen to be out of phase — DDNet's own server snapshots
/// every 2nd tick and resends the character core on the release tick too, so a press the driver
/// only actually sent on one real tick leaves the *next* snapshot nothing to see it on at all. A
/// live reviewer classification of every executable jump press across 8 phase-2 recordings found
/// this exactly: 2-3-tick pulses were detected 97/97 times, but 1-tick pulses only 1/15 — every
/// single miss was a 1-tick pulse. A window of length `L` sampled by a clock spaced roughly `T`
/// apart, regardless of their relative phase, always contains at least `floor(L/T)` samples — at
/// `PULSE_TICKS = 3` (60ms) that floor is already 2 even in the worst-case phase alignment, with
/// one full tick of margin past the 2 ticks live testing already found reliable.
const PULSE_TICKS: u128 = 3;

/// One window's pseudo-random decisions — factored out of [`random_scripted_input`] so
/// [`fire_counter`] (review round 1, finding F14) can replay every *earlier* window's fire
/// decision without duplicating the hash.
struct WindowDecision {
    direction: i32,
    hook: i32,
    jump_this_window: bool,
    fire_this_window: bool,
    target_x: i32,
    target_y: i32,
}

fn window_decision(window: u64, seed: u64) -> WindowDecision {
    // A tiny splitmix64-shaped hash — deterministic, dependency-free, good enough avalanche for
    // "looks varied across windows", which is all this needs.
    let mix = |x: u64| -> u64 {
        let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let h = mix(seed ^ window.wrapping_mul(0x2545F4914F6CDD1D));

    let direction = [-1, 0, 1][(h % 3) as usize];
    let hook = ((h >> 2) % 2) as i32;
    let jump_this_window = ((h >> 3) % 2) == 0;
    let fire_this_window = ((h >> 4) % 3) == 0; // fire a bit less often than jump
    let target_x = ((h >> 8) % 200) as i32 - 100;
    let target_y = ((h >> 20) % 200) as i32 - 100;
    // The real client never sends (0, 0) as an aim vector (`ClampVec2`-equivalent client-side
    // logic keeps at least a unit vector) — nudge it off the origin the same way `idle`/`circle`
    // above do (`target_y: -1` when otherwise zero).
    let target_y = if target_x == 0 && target_y == 0 { -1 } else { target_y };

    WindowDecision {
        direction,
        hook,
        jump_this_window,
        fire_this_window,
        target_x,
        target_y,
    }
}

/// Review round 1, finding F14: the real DDNet client encodes `m_Fire` as an ever-incrementing
/// counter whose bit 0 is "currently held" (`+fire` is bound to `ConKeyInputCounter`, unlike
/// `+left`/`+right`/`+jump`/`+hook`'s plain `ConKeyInputState` — `controls.cpp:78-89`:
/// `if ((*pVariable & 1) != pressed) (*pVariable)++;`), not a bare 0/1 level. The server's own
/// `CountInput(Prev, Cur)` (`gamecore.h:296-311`) walks from `Prev` to `Cur` one step at a time,
/// counting a press on every odd step and a release on every even one — sending a raw level
/// (0, then 1, then straight back to 0) makes `Cur` appear to have wrapped almost all the way
/// around from `1` back to `0`, which `CountInput` reports as a huge, spurious burst of
/// presses/releases instead of the one real press-then-release this brain actually intends.
///
/// Replicates the real counter exactly: every *earlier* window's completed fire pulse (press then
/// release) contributes exactly 2 to a running total; the current window, if it has its own fire
/// pulse, contributes 1 more while still held (odd = pressed) or 2 more once released (even).
/// O(window) per call — fine for this validation-only tool's short (tens of seconds) runs, not
/// meant for an unbounded-duration session. `tick_in_window` is a whole-tick count (review round
/// 3, finding F20 — see [`PULSE_TICKS`]'s own doc comment), not a millisecond offset.
fn fire_counter(current_window: u64, tick_in_window: u128, fire_this_window: bool, seed: u64) -> i32 {
    let mut count: i64 = 0;
    for w in 0..current_window {
        if window_decision(w, seed).fire_this_window {
            count += 2;
        }
    }
    if fire_this_window {
        count += if tick_in_window < PULSE_TICKS { 1 } else { 2 };
    }
    count as i32
}

/// Task 8.4a: a deterministic (seeded), non-repeating input generator for `--brain
/// random-scripted` — see that variant's docs. `elapsed` is quantized to whole input ticks
/// (review round 3, finding F20: [`TICK_MS`]-long ticks, not raw milliseconds — see
/// [`PULSE_TICKS`]'s own doc comment for why) and bucketed into [`TICKS_PER_WINDOW`]-tick (240ms)
/// windows: each window picks a fresh direction/hook/aim (held for the whole window, like a human
/// holding a key) and independently decides whether this window opens with a brief ([`PULSE_TICKS`]
/// -tick) jump and/or fire pulse (like a human tapping a key) — realistic enough to exercise every
/// field [`ddai_recorder`]'s reconstruction estimates, without needing an RNG crate dependency for
/// what is deliberately simple, reproducible test tooling.
fn random_scripted_input(elapsed: Duration, seed: u64) -> PlayerInput {
    let elapsed_ticks = elapsed.as_millis() / TICK_MS;
    let window = (elapsed_ticks / TICKS_PER_WINDOW) as u64;
    let tick_in_window = elapsed_ticks % TICKS_PER_WINDOW;
    let d = window_decision(window, seed);

    PlayerInput {
        direction: d.direction,
        target_x: d.target_x,
        target_y: d.target_y,
        jump: i32::from(d.jump_this_window && tick_in_window < PULSE_TICKS),
        fire: fire_counter(window, tick_in_window, d.fire_this_window, seed),
        hook: d.hook,
        player_flags: ddai_client::enums::playerflagflag::PLAYING,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
    }
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

    // Review round 1, finding F8: refuse a second instance under the same identity against the
    // same server — shared with `ddnet-ai record` (same lock scheme, keyed on (address, name) —
    // see `single_instance`'s own doc comment for why: an address-only lock broke this project's
    // own local multi-bot e2e tests, which run several distinctly-named bots against one address
    // at once). D-016: never more than one bot under the same identity on a server, from either
    // command.
    let server = match resolve_server(&args.server, &args.name, args.live_servers.as_deref()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("--server: {e}");
            return ExitCode::FAILURE;
        }
    };
    let _server_lock = match single_instance::acquire(server, &args.name) {
        Ok(lock) => lock,
        Err(e) => {
            tracing::error!(error = %e, "refusing to start");
            eprintln!("refusing to start: {e}");
            return ExitCode::FAILURE;
        }
    };

    if args.brain.is_bot_brain() || args.bot {
        return crate::bot_cmd::run(&args, &data_dir, server);
    }

    let mut config = ClientConfig {
        name: args.name.clone(),
        cache_dir: data_dir.join("maps").join("cache"),
        timeout: args
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(ClientConfig::default().timeout),
        // Review round 1, finding F7: only pay for `SessionEvent::InputSent` when `--input-log`
        // was actually requested.
        emit_input_sent: args.input_log.is_some(),
        // Review round 1, finding F1/F10: `ClientConfig::default()`'s own `live_servers` already
        // loads `~/aiddnet/data/live-servers.toml` and the driver enforces it on every connect —
        // `play` gets the same D-027/D-038 protection as `record` for free, no extra code needed
        // here.
        ..ClientConfig::default()
    };
    if let Err(e) = crate::proxy_cmd::prepare_client(&mut config, args.live_servers.as_deref(), server, &data_dir) {
        eprintln!("refusing to connect: {e}");
        return ExitCode::from(e.exit);
    }

    // Review round 1, finding F8: graceful shutdown on SIGINT/SIGTERM (see `record_cmd`'s
    // identical handler for the full rationale — a bare `kill` here skips `client.disconnect()`,
    // leaving the slot connected server-side until the silence timeout).
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&shutdown_requested);
        if let Err(e) = ctrlc::set_handler(move || {
            flag.store(true, Ordering::SeqCst);
        }) {
            tracing::warn!(error = %e, "failed to install a SIGINT/SIGTERM handler");
        }
    }

    tracing::info!(server = %server, name = %args.name, brain = ?args.brain, duration = args.duration, "connecting");
    let mut client = Client::connect(server, config);

    let mut input_log = match &args.input_log {
        Some(path) => match File::create(path) {
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

    let start = Instant::now();
    let duration = Duration::from_secs(args.duration);
    let mut in_game = false;
    let mut ended = false;
    let mut last_own_position_log: Option<Instant> = None;

    while !ended && start.elapsed() < duration && !shutdown_requested.load(Ordering::SeqCst) {
        if let Some(ev) = client.recv_event(Duration::from_millis(50)) {
            log_event(
                ev,
                &mut in_game,
                &mut ended,
                &mut last_own_position_log,
                input_log.as_mut(),
            );
        }

        if in_game {
            let input = match args.brain {
                Brain::Idle => idle_input(),
                Brain::Circle => circle_input(start.elapsed()),
                Brain::RandomScripted => random_scripted_input(start.elapsed(), args.seed),
                // Routed to `bot_cmd::run` before this loop is ever entered.
                Brain::Planner | Brain::Scripted | Brain::Hybrid | Brain::Fly => {
                    unreachable!("bot brains return early")
                }
            };
            client.set_input(input);
        }
    }

    if !ended {
        if shutdown_requested.load(Ordering::SeqCst) {
            tracing::info!("shutdown requested (SIGINT/SIGTERM), disconnecting");
        } else {
            tracing::info!("duration elapsed, disconnecting");
        }
        client.disconnect();
    }
    client.join();
    // The driver's final events (notably `MarginSummary`, sent right before its thread returns —
    // task e2e scenario h) are still sitting in the channel at this point: `join()` only waits
    // for the thread, it does not drain what it already sent. Draining here (non-blocking:
    // `try_iter` simply stops once the queue is empty) makes sure they still reach the log.
    for ev in client.events() {
        log_event(
            ev,
            &mut in_game,
            &mut ended,
            &mut last_own_position_log,
            input_log.as_mut(),
        );
    }
    ExitCode::SUCCESS
}

/// Task 8.4a: appends one JSON line for `SessionEvent::InputSent` — see `PlayArgs::input_log`'s
/// docs. Best-effort: a write failure (disk full, …) is logged once and otherwise ignored, the
/// same "never let logging/telemetry take the whole run down" spirit as `init_tracing`'s own
/// fallback.
fn log_input(file: &mut File, tick: i32, input: &PlayerInput) {
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

fn log_event(
    ev: ClientEvent,
    in_game: &mut bool,
    ended: &mut bool,
    last_own_position_log: &mut Option<Instant>,
    input_log: Option<&mut File>,
) {
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
            SessionEvent::InputSent { tick, input } => {
                tracing::trace!(tick, ?input, "input sent");
                if let Some(file) = input_log {
                    log_input(file, tick, &input);
                }
            }
            other => tracing::debug!(?other, "event"),
        },
        ClientEvent::ReconnectAttempt { attempt, addr, backoff } => {
            tracing::warn!(attempt, %addr, ?backoff, "reconnecting");
        }
        ClientEvent::RedirectFollowed { to } => tracing::info!(%to, "following redirect"),
        ClientEvent::RedirectRefused { reason } => tracing::error!(%reason, "redirect refused"),
        // Task 2.3b (root-cause fix): previously silent — see this event's own doc comment for
        // the incident this info-level line exists to prevent from happening unnoticed again.
        ClientEvent::ServerRequestedReconnect { addr, attempt } => {
            tracing::info!(%addr, attempt, "server requested reconnect (reconnect@ddnet.org)");
        }
        ClientEvent::GaveUp { reason, category } => {
            tracing::error!(%reason, ?category, "driver gave up");
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
        ClientEvent::OwnTeam { tick, team } => tracing::debug!(tick, team, "own team"),
        ClientEvent::InputLatency { .. } => {}
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
                margin_ms = s.margin_ms,
                margin_changes = s.margin_changes,
                adaptive = s.adaptive,
                "margin summary"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_scripted_input_is_a_pure_function_of_elapsed_and_seed() {
        let a = random_scripted_input(Duration::from_millis(1234), 7);
        let b = random_scripted_input(Duration::from_millis(1234), 7);
        assert_eq!(a, b);
    }

    #[test]
    fn random_scripted_input_varies_across_windows_and_seeds() {
        let mut directions = std::collections::HashSet::new();
        let mut hooks = std::collections::HashSet::new();
        let mut saw_jump = false;
        let mut saw_fire = false;
        for window in 0..200u128 {
            let elapsed = Duration::from_millis((window * 240) as u64);
            let input = random_scripted_input(elapsed, 7);
            directions.insert(input.direction);
            hooks.insert(input.hook);
            saw_jump |= input.jump == 1;
            saw_fire |= input.fire & 1 == 1; // bit 0 = currently held (F14's counter encoding)
        }
        assert_eq!(directions, [-1, 0, 1].into_iter().collect());
        assert_eq!(hooks, [0, 1].into_iter().collect());
        assert!(saw_jump, "expected at least one jump pulse over 200 windows");
        assert!(saw_fire, "expected at least one fire pulse over 200 windows");

        let with_other_seed = random_scripted_input(Duration::from_millis(1234), 99);
        let with_original_seed = random_scripted_input(Duration::from_millis(1234), 7);
        assert_ne!(
            with_other_seed, with_original_seed,
            "different seeds should (almost certainly) diverge at this sample point"
        );
    }

    /// Jump/fire are brief pulses (task 8.4a: "tapped", not "held") — never active past
    /// `PULSE_TICKS` into a window, so a rising-edge input-reconstruction estimator has a real
    /// single event to detect rather than an ambiguous multi-tick hold. `fire` is a counter (F14),
    /// so "not still active" means bit 0 clear (released), not literally `0`.
    #[test]
    fn jump_and_fire_never_last_past_the_pulse_window() {
        for window in 0..500u128 {
            let base = window * 240;
            // Tick 3 (60ms) is the first tick at or past `PULSE_TICKS` — the pulse must already
            // have ended by here (see `PULSE_TICKS`'s own doc comment for the exact boundary math).
            let late = random_scripted_input(Duration::from_millis((base + 61) as u64), 3);
            assert_eq!(
                late.jump, 0,
                "jump must not still be set past tick 3 (60ms) into a window"
            );
            assert_eq!(
                late.fire & 1,
                0,
                "fire must be released (bit 0 clear) past tick 3 (60ms) into a window"
            );
        }
    }

    /// Review round 3, finding F20's own fix: a pulse must actually cover at least 3 *distinct*
    /// sampled input ticks (0ms, 20ms, 40ms — ticks 0/1/2), not just "still be nonzero at some
    /// arbitrary millisecond" — this is the property live e2e testing needed (a real per-tick
    /// sample landing on each of the 3 ticks), and the property `jump_and_fire_never_last_past_
    /// the_pulse_window` alone cannot distinguish from a 1-tick pulse that just happens to still
    /// read as "active" at a millisecond offset within tick 0.
    #[test]
    fn pulse_covers_at_least_three_distinct_input_ticks_when_active() {
        let mut windows_checked = 0u32;
        for window in 0..500u128 {
            let base = window * 240;
            let d = window_decision(window as u64, 5);
            if !d.jump_this_window {
                continue;
            }
            windows_checked += 1;
            for tick in 0..3u128 {
                let elapsed = Duration::from_millis((base + tick * 20) as u64);
                let input = random_scripted_input(elapsed, 5);
                assert_eq!(
                    input.jump, 1,
                    "jump must still be active at tick {tick} (window {window})"
                );
            }
        }
        assert!(
            windows_checked > 0,
            "expected at least one window with a jump pulse over 500 windows"
        );
    }

    /// Review round 1, finding F14: the server's `CountInput(Prev, Cur)` walks one step at a time
    /// from `Prev` to `Cur` and infers a press/release per step — so consecutive samples' `fire`
    /// values must never differ by more than 1 (anything else would make the server see a burst
    /// of phantom presses/releases that never happened). Sampled at a 20ms (one server tick)
    /// cadence across many windows, including every pulse boundary.
    #[test]
    fn fire_counter_never_jumps_by_more_than_one_step_per_tick() {
        let mut prev: Option<i32> = None;
        for tick in 0..(500 * 12u128) {
            // 12 ticks (240ms) per window, matching `WINDOW_MS`/a real 50Hz tick.
            let elapsed = Duration::from_millis((tick * 20) as u64);
            let input = random_scripted_input(elapsed, 3);
            if let Some(p) = prev {
                assert!(
                    (input.fire - p).abs() <= 1,
                    "fire jumped from {p} to {} between two consecutive ticks (tick {tick})",
                    input.fire
                );
            }
            prev = Some(input.fire);
        }
    }

    /// The counter must actually be monotonically non-decreasing (a real client's `+fire` counter
    /// never resets or goes backward within one session) and must actually vary (not get stuck).
    #[test]
    fn fire_counter_is_monotonically_non_decreasing_and_moves() {
        let mut prev = 0;
        let mut moved = false;
        for tick in 0..(300 * 12u128) {
            let elapsed = Duration::from_millis((tick * 20) as u64);
            let input = random_scripted_input(elapsed, 11);
            assert!(input.fire >= prev, "fire counter must never decrease");
            if input.fire != prev {
                moved = true;
            }
            prev = input.fire;
        }
        assert!(moved, "expected the fire counter to change at least once");
    }

    /// The aim vector is never exactly `(0, 0)` — matches `idle`/`circle`'s own convention above
    /// (`target_y: -1`) and every real client, which never sends a zero aim vector.
    #[test]
    fn aim_vector_is_never_exactly_zero() {
        for window in 0..2000u128 {
            let input = random_scripted_input(Duration::from_millis((window * 240) as u64), 11);
            assert!(
                input.target_x != 0 || input.target_y != 0,
                "aim vector must never be exactly (0, 0), got window {window}"
            );
        }
    }
}
