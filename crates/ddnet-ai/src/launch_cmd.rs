//! `ddnet-ai launch apply|exited` (task 5.9, D-089): the **root-side helper of the web launcher**. The web only writes a small
//! request file; a root systemd path unit runs `launch apply`, which is the only code that turns a request into `systemctl`
//! calls. Everything is decided here, against fixed allow-lists, and nothing from the request is ever passed to a command line:
//!
//! - the request is read as a regular file (no symlink), size-limited, parsed with a strict schema, and **deleted before it is
//!   processed** (a request is never replayed, e.g. after a reboot);
//! - `server` is `"local"`, the exact `address` of a `ready = true` entry of `live-servers.toml`, or (task 5.12, D-099) the exact
//!   address of one of the owner's **favourites** (`<launch-dir>/favourites.json`, written by the site and validated strictly
//!   here again: a public unicast `ip:port`, a valid nick, `direct` or `proxy:<name>`, the owner's consent). A favourite is judged
//!   exactly like a `ready` entry, with the same ban memory, cool-downs and limits; the nick, the proxy and the address that
//!   reach the unit come from that entry, never from the request;
//! - the root-owned environment file for the bot unit holds only validated values from a closed character set;
//! - the unit's cgroup filter (a generated drop-in) allows loopback, plus the proxy's IPs (read from the secrets file, never
//!   printed) or the entry's server IP for a public server; for a proxy with `relay = "public"` (its UDP relay lives on
//!   another host, task 2.6b, D-090) it is the opposite: both lists are reset and **every IP of the game server is denied**,
//!   with no allow list, so the relay is reachable anywhere and the server never directly;
//! - at most one start per [`START_INTERVAL_SECS`]; a cool-down after exit 3/4; after a kick/ban on a **public** server that
//!   server (every address on the same IP) is refused until the owner has re-opened it: edited `live-servers.toml` again, or, for a
//!   favourite, pressed «Открыть снова» on the site (`reopened_at` newer than the ban). **Nothing ever switches the proxy or the IP by
//!   itself**: a favourite's proxy is the one the owner assigned, a missing or unsafe proxy file is a refusal (never a direct start
//!   or another proxy), and a ban closes the server whatever proxy the favourite names afterwards.
//!
//! `launch exited` is the bot unit's `ExecStopPost=` hook: it records how the bot ended (the ban memory above) and tells the web.
//! `launch check-proxy` (task 5.12) answers the site's «Проверить» button: it runs **unprivileged** (its own unit), reads one small
//! request naming a proxy profile, runs `proxy-check` on it and writes a result of fixed codes and numbers (never an address or a
//! credential).
//! Schemas: `docs/formats.md` §34. Never connects to any game server and never writes chat.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::net::{IpAddr, SocketAddr};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use clap::{Args, Subcommand};
use ddai_client::favourites::{self, Favourite, Favourites, Rules};
use ddai_client::live_servers::{LiveServers, is_loopback};
use ddai_client::socks5::{ProxyCheck, RelayHost, Socks5Error, Timeouts};
use ddai_web::launch::{
    Action, Brain, DurationChoice, Finish, LOCAL_SERVER, LaunchConfig, LaunchRequest, LaunchStatus, MAX_REQUEST_BYTES,
    MAX_SPARRING, Mirror, REQUEST_FILE, ReadError, START_INTERVAL_SECS, STATUS_FILE, State as RunState, WbSmart,
    bundle_run_name, no_selfkill_flag_value, parse_request, preinput_flag_value, read_regular_nofollow,
    read_regular_nofollow_with_mtime, request_is_fresh, unix_now, write_atomic,
};
use ddai_web::serverbrowser::{
    BLOCKED_FILE, BlockedEntry, BlockedFile, MAX_PROXY_CHECK_BYTES, PROXY_CHECK_REQUEST_FILE, PROXY_CHECK_RESULT_FILE,
    ProxyCheckResult, parse_proxy_check_request,
};
use serde::{Deserialize, Serialize};

const BOT_UNIT: &str = "ddnet-ai-bot.service";
const LOCAL_UNIT: &str = "ddnet-local.service";
/// The local server the «Локальный сервер» choice means.
const LOCAL_ADDR: &str = "127.0.0.1:8303";
/// The local identity of the bot (D-068); a public server's nick comes from its allow-list entry.
const LOCAL_NAME: &str = "Muha";
/// After a bot exit 3 or 4 nothing is started for this long (any server).
const COOLDOWN_AFTER_EXIT_SECS: u64 = 120;
/// How long after `stop` the bot's exit still counts as "stopped by the owner".
const STOP_GRACE_SECS: u64 = 120;
/// The state file's size cap when reading it back.
const MAX_STATE_BYTES: usize = 64 * 1024;
/// A live-servers entry's value that becomes part of a command line must be this short and plain.

#[derive(Debug, Args)]
pub struct LaunchArgs {
    #[command(subcommand)]
    pub command: LaunchCommand,
}

#[derive(Debug, Subcommand)]
pub enum LaunchCommand {
    /// Consumes the web's request file, validates it, writes the bot unit's environment file and cgroup drop-in, and runs
    /// `systemctl start|stop` (run by `ddnet-ai-launch.service` as root).
    Apply(ApplyArgs),
    /// The bot unit's `ExecStopPost=` hook: records how the bot ended (exit code from `$EXIT_STATUS`) and updates the status.
    Exited(PathArgs),
    /// The site's «Проверить» button (task 5.12): consumes `<launch-dir>/proxycheck-request.json`, runs `proxy-check` on the named
    /// profile and writes `<launch-dir>/proxycheck-result.json` (fixed codes and numbers). Unprivileged: run by its own unit.
    CheckProxy(CheckProxyArgs),
}

#[derive(Debug, Args)]
pub struct CheckProxyArgs {
    /// Base data directory (`launch/`, `secrets/`). Default `~/aiddnet/data`.
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
pub struct PathArgs {
    /// Base data directory (`live-servers.toml`, `secrets/`, `launch/`). Default `~/aiddnet/data`.
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    /// The allow-list. Default `<data-dir>/live-servers.toml`.
    #[arg(long)]
    pub live_servers: Option<PathBuf>,
    /// The root-owned launcher config (`fly_bundle`). Default `/etc/ddnet-ai/launch.toml`; missing means the default bundle.
    #[arg(long, default_value = ddai_web::launch::DEFAULT_CONFIG_PATH)]
    pub config: PathBuf,
    /// The bot unit's environment file (written here).
    #[arg(long, default_value = "/etc/ddnet-ai/bot-launch.env")]
    pub env_file: PathBuf,
    /// The bot unit's cgroup drop-in (written here).
    #[arg(long, default_value = "/etc/systemd/system/ddnet-ai-bot.service.d/50-launch.conf")]
    pub dropin: PathBuf,
    /// The helper's own memory (last start, last exit, bans). Root-only.
    #[arg(long, default_value = "/var/lib/ddnet-ai/launch-state.json")]
    pub state: PathBuf,
    /// The directory the web writes `request.json` into (the helper only reads and removes it there). Default `<data-dir>/launch`.
    #[arg(long)]
    pub launch_dir: Option<PathBuf>,
    /// The root-owned directory `status.json` is written into, for the web to read (never inside the web's own directory: the
    /// unsandboxed stop hook would follow a swapped path, review F5).
    #[arg(long, default_value = ddai_web::launch::DEFAULT_STATUS_DIR)]
    pub status_dir: PathBuf,
}

#[derive(Debug, Args)]
pub struct ApplyArgs {
    #[command(flatten)]
    pub paths: PathArgs,
    /// The request file. Default `<launch-dir>/request.json`.
    #[arg(long)]
    pub request: Option<PathBuf>,
}

struct Paths {
    data_dir: PathBuf,
    live_servers: PathBuf,
    config: PathBuf,
    env_file: PathBuf,
    dropin: PathBuf,
    state: PathBuf,
    launch_dir: PathBuf,
    status_dir: PathBuf,
}

impl PathArgs {
    fn resolve(&self) -> Paths {
        let data_dir = self.data_dir.clone().unwrap_or_else(|| match std::env::var_os("HOME") {
            Some(h) if !h.is_empty() => PathBuf::from(h).join("aiddnet").join("data"),
            _ => PathBuf::from("data"),
        });
        Paths {
            live_servers: self
                .live_servers
                .clone()
                .unwrap_or_else(|| data_dir.join("live-servers.toml")),
            launch_dir: self.launch_dir.clone().unwrap_or_else(|| data_dir.join("launch")),
            status_dir: self.status_dir.clone(),
            config: self.config.clone(),
            env_file: self.env_file.clone(),
            dropin: self.dropin.clone(),
            state: self.state.clone(),
            data_dir,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// State (root-only memory)
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct LaunchInfo {
    id: String,
    brain: Brain,
    server: String,
    duration: DurationChoice,
    sparring: u8,
    public: bool,
    /// The target was one of the owner's favourites (not an allow-list entry): the origin a ban is recorded with.
    #[serde(default)]
    favourite: bool,
    bundle: Option<String>,
    /// The finishing mode of the launch (task 5.13); older state files have none, which is `off`.
    #[serde(default)]
    finish: Finish,
    /// The smart wayblock of the launch (task 5.15); older state files have none, which is `off`.
    #[serde(default)]
    wb_smart: WbSmart,
    /// The duel switch of the launch (task 5.15); older state files have none, which is `false`.
    #[serde(default)]
    no_selfkill: bool,
    /// The opponent-input predictor of the launch (task 3.17); older state files have none, which is `false`.
    #[serde(default)]
    window_model: bool,
    /// The server's pre-inputs played in the prediction (task 3.20b); older state files have none, which is `false`.
    #[serde(default)]
    preinput: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct ExitInfo {
    at: u64,
    code: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct Block {
    at: u64,
    code: i32,
    /// Recorded for a favourite. Editing `live-servers.toml` never lifts such a block, not even for a sibling allow-list entry on the
    /// same IP (review 5.12 F3); only that favourite's own re-opening does. Older state files have none: an allow-list block.
    #[serde(default)]
    favourite: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
struct State {
    last_start_at: u64,
    last_launch: Option<LaunchInfo>,
    stop_requested_at: u64,
    last_exit: Option<ExitInfo>,
    /// Public servers (by address: an allow-list entry or a favourite) the bot was kicked or banned from, until the owner re-opens
    /// them (the allow-list edited, or the favourite's «Открыть снова»). A block closes every address on the same IP.
    blocked: BTreeMap<String, Block>,
}

impl State {
    /// The kicks and bans that still close `target`: those recorded for any address on the **same IP** (a ban for «VPN detected» or
    /// «bad ip» is the machine's, not one port's) that its way of re-opening has not lifted yet. Empty for `local`.
    fn closing_blocks(&self, target: &Target, live_servers_mtime: u64) -> Vec<&str> {
        let lifted = |block: &Block| match target.reopen {
            Reopen::Never => true,
            // The file's edit lifts only blocks recorded on allow-list entries.
            Reopen::AllowListEdit => !block.favourite && live_servers_mtime > block.at,
            Reopen::Favourite { reopened_at } => reopened_at > block.at,
        };
        if target.reopen == Reopen::Never {
            return Vec::new();
        }
        self.blocked
            .iter()
            .filter(|(key, _)| {
                key.as_str() == target.key
                    || key
                        .parse::<SocketAddr>()
                        .is_ok_and(|a| a.ip().to_canonical() == target.addr.ip().to_canonical())
            })
            .filter(|(_, block)| !lifted(block))
            .map(|(key, _)| key.as_str())
            .collect()
    }
}

/// Loads the state: `Ok(default)` when the file does not exist, `Err(())` when it exists and cannot be trusted (then every
/// start is refused until the owner has looked at it, because the ban memory may be in it).
fn load_state(path: &Path) -> Result<State, ()> {
    match read_regular_nofollow(path, MAX_STATE_BYTES) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(|_| ()),
        Err(ReadError::Missing) => Ok(State::default()),
        Err(_) => Err(()),
    }
}

/// Runs `f` on the state under an exclusive lock and saves the result. The lock is held only for this short read-modify-write,
/// never across a `systemctl` call (the bot's `ExecStopPost` hook takes the same lock).
fn with_state<T>(path: &Path, f: impl FnOnce(&mut State) -> T) -> std::io::Result<T> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut builder = std::fs::DirBuilder::new();
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.recursive(true).create(dir)?;
    let lock_path = path.with_extension("lock");
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(0o600)
        .open(&lock_path)?;
    lock.lock()?;
    // A state that exists but cannot be read is never overwritten: the ban memory may be in it.
    let mut state = load_state(path)
        .map_err(|()| std::io::Error::new(std::io::ErrorKind::InvalidData, "the launcher state file is unreadable"))?;
    let out = f(&mut state);
    let bytes = serde_json::to_vec_pretty(&state).map_err(std::io::Error::other)?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("launch-state.json");
    write_atomic(dir, name, &bytes, 0o600)?;
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Validation: request + allow-lists -> plan
// ---------------------------------------------------------------------------------------------

/// A refusal: a fixed code the web turns into text. Never carries anything from the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Refuse(&'static str);

/// How a closed (kicked or banned) server is opened again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reopen {
    /// `local` is never closed.
    Never,
    /// An allow-list entry: the owner edits `live-servers.toml` (its mtime must be newer than the ban).
    AllowListEdit,
    /// A favourite: the owner pressed «Открыть снова» on the site (its `reopened_at` must be newer than the ban).
    Favourite { reopened_at: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Target {
    /// `local`, or the entry's `address` (the state's and the status's name for it).
    key: String,
    addr: SocketAddr,
    nick: String,
    public: bool,
    reopen: Reopen,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Plan {
    target: Target,
    brain: Brain,
    duration: DurationChoice,
    sparring: u8,
    bundle: Option<PathBuf>,
    /// The hybrid's opponent model (`on` unless the request says `off`, D-090).
    mirror: Mirror,
    /// The finishing mode (`off` unless the request says otherwise, task 5.13, D-097).
    finish: Finish,
    /// The smart wayblock (`off` unless the request says otherwise, task 5.15, D-103/D-104); every brain takes it.
    wb_smart: WbSmart,
    /// The duel switch (`false` unless the request says otherwise, task 5.15, D-102); every brain takes it.
    no_selfkill: bool,
    /// The opponent-input model file (task 3.17, D-111) when the request asked for the predictor; hybrid brains only.
    window_model: Option<PathBuf>,
    /// The server's pre-inputs in the prediction (task 3.20b, D-112): `false` unless the request says otherwise; hybrid brains only.
    preinput: bool,
}

fn valid_nick(nick: &str) -> bool {
    favourites::valid_nick(nick)
}

/// The favourites as the helper sees them: the file's list, or the reason it cannot be trusted (then no favourite can be started).
type FavouritesView = Result<Favourites, &'static str>;

/// `"local"`, the one `ready = true` entry of the allow-list whose `address` is exactly `selector`, or the favourite with exactly
/// that address.
fn resolve_server(selector: &str, live: &LiveServers, favs: &FavouritesView, rules: Rules) -> Result<Target, Refuse> {
    if selector == LOCAL_SERVER {
        return Ok(Target {
            key: LOCAL_SERVER.to_string(),
            addr: LOCAL_ADDR.parse().map_err(|_| Refuse("internal"))?,
            nick: LOCAL_NAME.to_string(),
            public: false,
            reopen: Reopen::Never,
        });
    }
    let listed: Vec<_> = live.servers.iter().filter(|e| e.address == selector).collect();
    if !listed.is_empty() {
        let ready: Vec<_> = listed.iter().filter(|e| e.ready).collect();
        let Some(entry) = ready.first() else {
            return Err(Refuse("server_not_ready"));
        };
        if ready.iter().any(|e| e.nick != entry.nick || e.proxy != entry.proxy) {
            return Err(Refuse("server_ambiguous"));
        }
        let addr: SocketAddr = entry.address.parse().map_err(|_| Refuse("server_bad_entry"))?;
        if is_loopback(addr) || !valid_nick(&entry.nick) {
            return Err(Refuse("server_bad_entry"));
        }
        return Ok(Target {
            key: entry.address.clone(),
            addr,
            nick: entry.nick.clone(),
            public: true,
            reopen: Reopen::AllowListEdit,
        });
    }
    // Not an allow-list entry: only a favourite can be meant. A favourites file that cannot be trusted means none can.
    let favs = favs.as_ref().map_err(|code| Refuse(code))?;
    // A host name in the allow-list can hide the address a favourite names, and nothing here resolves names: no favourite while one
    // exists (review 5.12 F2; the owner's file has IP literals only).
    if live.has_non_literal_entry() {
        return Err(Refuse("allowlist_not_literal"));
    }
    let Some(fav) = favs.find(selector) else {
        return Err(Refuse("server_not_allowed"));
    };
    target_of_favourite(fav, live, rules)
}

/// A favourite as a start target: validated again (the file's own parse already did, this is for callers that built one), and
/// refused when the allow-list also names the same server (two statements about one server would be a guess).
fn target_of_favourite(fav: &Favourite, live: &LiveServers, rules: Rules) -> Result<Target, Refuse> {
    fav.validate(rules).map_err(|e| Refuse(e.code()))?;
    let addr = fav.socket_addr().ok_or(Refuse("server_bad_entry"))?;
    if live.listed(addr) {
        return Err(Refuse("server_ambiguous"));
    }
    Ok(Target {
        key: fav.address.clone(),
        addr,
        nick: fav.nick.clone(),
        public: true,
        reopen: Reopen::Favourite {
            reopened_at: fav.reopened_at,
        },
    })
}

/// The bundle from the root-owned config (default: `<data-dir>/runs/E-005/e005-fly/checkpoints/final.bundle`). A path of a
/// closed character set, absolute, no `..`, a regular file; the config file itself must be owned by root (or by whoever runs
/// the helper, for tests) and not writable by group or others.
fn configured_bundle(config: &Path, data_dir: &Path) -> Result<PathBuf, Refuse> {
    let path = match read_regular_nofollow(config, 4096) {
        Ok(bytes) => {
            let me = std::fs::metadata("/proc/self").map_or(0, |m| m.uid());
            let meta = std::fs::symlink_metadata(config).map_err(|_| Refuse("config_bad"))?;
            if (meta.uid() != 0 && meta.uid() != me) || meta.mode() & 0o022 != 0 {
                return Err(Refuse("config_untrusted"));
            }
            let text = String::from_utf8(bytes).map_err(|_| Refuse("config_bad"))?;
            let cfg: LaunchConfig = toml::from_str(&text).map_err(|_| Refuse("config_bad"))?;
            cfg.fly_bundle
                .unwrap_or_else(|| data_dir.join(ddai_web::launch::DEFAULT_BUNDLE_REL))
        }
        Err(ReadError::Missing) => data_dir.join(ddai_web::launch::DEFAULT_BUNDLE_REL),
        Err(_) => return Err(Refuse("config_bad")),
    };
    let text = path.to_str().ok_or(Refuse("bundle_missing"))?;
    let plain = text
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'.' | b'_' | b'-'));
    let normal = path
        .components()
        .all(|c| matches!(c, std::path::Component::RootDir | std::path::Component::Normal(_)));
    if !path.is_absolute() || !plain || !normal || path.extension().and_then(|e| e.to_str()) != Some("bundle") {
        return Err(Refuse("bundle_bad_path"));
    }
    if !path.is_file() {
        return Err(Refuse("bundle_missing"));
    }
    Ok(path)
}

/// The opponent-input model the launch card's toggle means (task 3.17, D-111): `<data-dir>/bot/models/opp-m1.oppnet`, a plain absolute path
/// that is a regular file (not a symlink, not a directory) of a sane size. The request names no path, so nothing else can be reached through it.
fn configured_window_model(data_dir: &Path) -> Result<PathBuf, Refuse> {
    let path = data_dir.join(ddai_web::launch::DEFAULT_WINDOW_MODEL_REL);
    let text = path.to_str().ok_or(Refuse("window_model_bad_path"))?;
    let plain = text
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'.' | b'_' | b'-'));
    let normal = path
        .components()
        .all(|c| matches!(c, std::path::Component::RootDir | std::path::Component::Normal(_)));
    if !path.is_absolute() || !plain || !normal {
        return Err(Refuse("window_model_bad_path"));
    }
    match std::fs::symlink_metadata(&path) {
        Ok(m) if m.file_type().is_file() && m.len() > 0 && m.len() <= MAX_WINDOW_MODEL_BYTES => Ok(path),
        _ => Err(Refuse("window_model_missing")),
    }
}

/// The model file is 0.3 MB; a file beyond this is not it.
const MAX_WINDOW_MODEL_BYTES: u64 = 4 << 20;

/// Everything a start is judged against besides the state: the owner's two lists.
struct Catalog<'a> {
    live: &'a LiveServers,
    favs: &'a FavouritesView,
    rules: Rules,
    /// Unix seconds when the owner last edited the allow-list: a ban of an allow-list entry stays until the file is newer. A
    /// favourite is re-opened by its own `reopened_at` instead.
    live_servers_mtime: u64,
    /// The opponent-input model file the launch card's toggle means (task 3.17, D-111): resolved only when a request asks for it.
    window_model: &'a dyn Fn() -> Result<PathBuf, Refuse>,
}

/// Judges a start request against the allow-lists and the rate rules. `state` is `None` when the state file cannot be trusted
/// (everything is then refused, the local server too: the file may hold a ban).
fn decide(
    req: &LaunchRequest,
    cat: &Catalog,
    state: Option<&State>,
    bundle: &dyn Fn() -> Result<PathBuf, Refuse>,
    now: u64,
) -> Result<Plan, Refuse> {
    if req.action != Action::Start {
        return Err(Refuse("bad_request"));
    }
    let (Some(brain), Some(selector), Some(duration)) = (req.brain, req.server.as_deref(), req.duration) else {
        return Err(Refuse("bad_request"));
    };
    let sparring = req.sparring.unwrap_or(0);
    if sparring > MAX_SPARRING {
        return Err(Refuse("bad_request"));
    }
    // Task 5.13: finishing was measured with the hybrid (D-097); the pure fly was not trained or measured on a held victim, so it takes
    // `off` only. The web refuses the same request first (`finish_hybrid_only`); this is the helper's own, authoritative check.
    let finish = req.finish.unwrap_or_default();
    if brain == Brain::Fly && finish.is_on() {
        return Err(Refuse("finish_hybrid_only"));
    }
    // Task 3.17 (D-111): the opponent-input model sits in the hybrid's lag window; the pure fly has none. The web refuses the same request first.
    let window_model_asked = req.window_model.unwrap_or_default();
    if brain == Brain::Fly && window_model_asked {
        return Err(Refuse("window_model_hybrid_only"));
    }
    // Task 3.20b (D-112): the pre-inputs play in the hybrid's prediction; the pure fly is not offered them. The web refuses the same request first.
    let preinput = req.preinput.unwrap_or_default();
    if brain == Brain::Fly && preinput {
        return Err(Refuse("preinput_hybrid_only"));
    }
    let target = resolve_server(selector, cat.live, cat.favs, cat.rules)?;
    if sparring > 0 && target.public {
        return Err(Refuse("sparring_local_only"));
    }
    let bundle = if brain.needs_bundle() { Some(bundle()?) } else { None };
    let window_model = if window_model_asked {
        Some((cat.window_model)()?)
    } else {
        None
    };

    // Policy: the ban memory first, then the rates.
    match state {
        None => return Err(Refuse("state_unreadable")),
        Some(st) => {
            if !st.closing_blocks(&target, cat.live_servers_mtime).is_empty() {
                return Err(Refuse("blocked_after_ban"));
            }
            if let Some(exit) = st.last_exit
                && matches!(exit.code, 3 | 4)
                && now < exit.at.saturating_add(COOLDOWN_AFTER_EXIT_SECS)
            {
                return Err(Refuse("cooldown"));
            }
            if now < st.last_start_at.saturating_add(START_INTERVAL_SECS) {
                return Err(Refuse("rate_limited"));
            }
        }
    }
    Ok(Plan {
        target,
        brain,
        duration,
        sparring,
        bundle,
        mirror: req.mirror.unwrap_or(Mirror::On),
        finish,
        wb_smart: req.wb_smart.unwrap_or_default(),
        no_selfkill: req.no_selfkill.unwrap_or_default(),
        window_model,
        preinput,
    })
}

// ---------------------------------------------------------------------------------------------
// What is written for systemd
// ---------------------------------------------------------------------------------------------

/// One `KEY="value"` line; the value is from a closed character set, so no quoting or escaping is ever needed.
fn env_line(key: &str, value: &str) -> Result<String, Refuse> {
    let plain = value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'/' | b':' | b' ' | b'[' | b']'));
    if !plain || value.len() > 512 {
        return Err(Refuse("internal"));
    }
    Ok(format!("{key}=\"{value}\"\n"))
}

/// The bot unit's environment file: only validated values.
fn render_env(id: &str, plan: &Plan) -> Result<String, Refuse> {
    let fly_args = plan
        .bundle
        .as_ref()
        .map(|b| format!("--fly-bundle {}", b.display()))
        .unwrap_or_default();
    let mut out = String::from(
        "# Written by `ddnet-ai launch apply` (root) from a validated web request. Do not edit; it is rewritten at every start.\n",
    );
    out += &env_line("BOT_LAUNCH_ID", id)?;
    out += &env_line("BOT_SERVER", &plan.target.addr.to_string())?;
    out += &env_line("BOT_NAME", &plan.target.nick)?;
    out += &env_line("BOT_BRAIN", plan.brain.play_brain())?;
    out += &env_line("BOT_DURATION", &plan.duration.seconds().to_string())?;
    out += &env_line("BOT_FLY_ARGS", &fly_args)?;
    out += &env_line("BOT_HYBRID_MIRROR", plan.mirror.flag_value())?;
    out += &env_line("BOT_FINISH", plan.finish.flag_value())?;
    out += &env_line("BOT_WB_SMART", plan.wb_smart.flag_value())?;
    out += &env_line("BOT_NO_SELFKILL", no_selfkill_flag_value(plan.no_selfkill))?;
    // Always written (empty = off): the unit passes `--window-model=${BOT_WINDOW_MODEL}` as one argument, so a stale path can never leak into a run.
    let model = plan
        .window_model
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    out += &env_line("BOT_WINDOW_MODEL", &model)?;
    // Always written (`on` or `off`): the unit passes `--preinput ${BOT_PREINPUT}`, so a stale value can never leak into a run.
    out += &env_line("BOT_PREINPUT", preinput_flag_value(plan.preinput))?;
    Ok(out)
}

/// What the bot unit's cgroup filter must be for one launch.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Filter {
    /// Loopback always, plus exactly these IPs: none for the local server, the proxy's IPs for a proxy whose relay is on
    /// the proxy's host (`relay = "proxy-host-only"`), the server's own IP for a direct public server.
    Allow(Vec<IpAddr>),
    /// `relay = "public"`: everything **except** these IPs, the game server's. No allow list: systemd lets an allow match
    /// win over a deny match (measured on this machine's systemd 255 by `tools/e2e/ipfilter_probe.sh`), so any allow
    /// entry covering the server would void the deny. Never empty (an empty deny list would filter nothing).
    DenyServer(Vec<IpAddr>),
}

/// The cgroup filter drop-in.
fn render_dropin(filter: &Filter) -> String {
    match filter {
        Filter::Allow(extra) => {
            let mut out = String::from(
                "# Written by `ddnet-ai launch apply` (root): the addresses the bot unit may talk to for the current launch. Do not edit.\n[Service]\nIPAddressAllow=\nIPAddressAllow=127.0.0.0/8 ::1\n",
            );
            for ip in extra {
                out += &format!("IPAddressAllow={ip}\n");
            }
            out
        }
        Filter::DenyServer(server) => {
            // Reset BOTH lists (the unit ships `Allow=127.0.0.0/8 ::1` and `Deny=any`), then deny each server IP.
            let mut out = String::from(
                "# Written by `ddnet-ai launch apply` (root): relay = public. The bot unit may talk to everything except the game server's own addresses and the private ranges. Do not edit.\n[Service]\nIPAddressAllow=\nIPAddressDeny=\n",
            );
            for ip in server {
                out += &format!("IPAddressDeny={ip}\n");
            }
            // Task 2.6b review F3: the code refuses a relay whose address family differs from the server's, so the other
            // family is closed here too (a v4-mapped destination is judged as IPv4 by the kernel filter, probed in
            // `tools/e2e/ipfilter_probe.sh` S8). The proxy must then be reachable over the server's family.
            if server.iter().all(IpAddr::is_ipv4) {
                out += "IPAddressDeny=::/0\n";
            }
            // The ranges the client already refuses as a relay; also keeps the cloud metadata address away from the bot.
            for range in DENY_PRIVATE_RANGES {
                out += &format!("IPAddressDeny={range}\n");
            }
            out
        }
    }
}

/// Ranges denied next to the server's IPs for `relay = "public"`: private (RFC 1918), link-local, CGNAT, and the IPv6
/// unique-local and link-local blocks. Loopback is not among them (the DNS stub, the web unit and the like stay reachable).
const DENY_PRIVATE_RANGES: [&str; 7] = [
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "169.254.0.0/16",
    "100.64.0.0/10",
    "fc00::/7",
    "fe80::/10",
];

/// The cgroup filter for a launch: nothing extra for the local server; for a public server the IPs of its proxy (read from
/// the secrets file, never printed) when the proxy's relay is on its own host, **every IP of the game server denied** when
/// the proxy file says `relay = "public"` (the relay may be anywhere), or the IP of the server itself when the entry names
/// no proxy. The server's IPs are the entry's address plus whatever the proxy file's `for_server` resolves to.
fn filter_for(plan: &Plan, live: &LiveServers, secrets_dir: &Path) -> Result<Filter, Refuse> {
    if !plan.target.public {
        return Ok(Filter::Allow(Vec::new()));
    }
    match ddai_client::proxy::resolve_for_server(plan.target.addr, &plan.target.nick, live, secrets_dir) {
        Ok(Some(proxy)) => match proxy.relay_mode() {
            ddai_client::proxy::RelayMode::ProxyHostOnly => proxy
                .resolve_ips()
                .map(Filter::Allow)
                .map_err(|_| Refuse("proxy_error")),
            ddai_client::proxy::RelayMode::Public => {
                let mut server = vec![plan.target.addr.ip().to_canonical()];
                for ip in proxy.for_server_ips() {
                    if !server.contains(&ip) {
                        server.push(ip);
                    }
                }
                Ok(Filter::DenyServer(server))
            }
        },
        Ok(None) => Ok(Filter::Allow(vec![plan.target.addr.ip().to_canonical()])),
        Err(_) => Err(Refuse("proxy_error")),
    }
}

// ---------------------------------------------------------------------------------------------
// systemctl (fixed argument lists only)
// ---------------------------------------------------------------------------------------------

fn systemctl(args: &[&str]) -> bool {
    Command::new("systemctl")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()
        .is_ok_and(|s| s.success())
}

/// `ActiveState` of a unit (`active`, `inactive`, `failed`, ...), or `unknown` when it cannot be asked.
fn active_state(unit: &str) -> String {
    Command::new("systemctl")
        .args(["show", "--property=ActiveState", "--value", unit])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string())
}

/// Whether the bot unit is configured only by the unit file and the launcher's own drop-in. Any other drop-in (for example a
/// hand-made `ExecStart=` override for a public server) would make the unit ignore the validated environment file and its
/// cgroup filter: such a start is refused, never guessed at. `DropInPaths` empty (before the first reload) is fine.
fn only_our_dropin(own: &Path) -> bool {
    let Some(out) = Command::new("systemctl")
        .args(["show", "--property=DropInPaths", "--value", BOT_UNIT])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
    else {
        return false;
    };
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .all(|p| Path::new(p) == own)
}

fn sparring_units(count: u8) -> Vec<String> {
    (1..=count.min(MAX_SPARRING))
        .map(|i| format!("ddnet-ai-sparring@{i}.service"))
        .collect()
}

fn stop_all() -> bool {
    let mut ok = systemctl(&["stop", BOT_UNIT]);
    let units = sparring_units(MAX_SPARRING);
    let mut args = vec!["stop"];
    args.extend(units.iter().map(String::as_str));
    ok &= systemctl(&args);
    ok
}

// ---------------------------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------------------------

/// Writes `path` atomically with exactly `mode` (whatever the umask: the bot unit's stop hook runs under `UMask=0077`), creating
/// its directory (`0755`) when it is missing.
fn write_root_file(path: &Path, bytes: &[u8], mode: u32) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    if !dir.is_dir() {
        let mut builder = std::fs::DirBuilder::new();
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o755);
        builder.recursive(true).create(dir)?;
        std::fs::set_permissions(dir, std::os::unix::fs::PermissionsExt::from_mode(0o755))?;
    }
    write_atomic(
        dir,
        path.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
        bytes,
        mode,
    )
}

/// Back to the plain local default for a later hand start: no one-run environment, and a loopback-only cgroup filter (a proxy IP
/// of a public run must not stay allowed). The manager picks the drop-in up at its next `daemon-reload`.
fn reset_to_local(paths: &Paths) {
    let _ = std::fs::remove_file(&paths.env_file);
    if paths.dropin.exists() {
        let _ = write_root_file(
            &paths.dropin,
            render_dropin(&Filter::Allow(Vec::new())).as_bytes(),
            0o644,
        );
    }
}

fn write_status(paths: &Paths, status: &LaunchStatus) {
    let Ok(bytes) = serde_json::to_vec_pretty(status) else {
        return;
    };
    if let Err(e) = write_root_file(&paths.status_dir.join(STATUS_FILE), &bytes, 0o644) {
        eprintln!("launch: could not write the status file: {e}");
    }
}

/// Tells the web which servers the bot was kicked or banned from and has not been re-opened (`blocked.json` next to the status, root
/// owned, world readable: addresses, times and exit codes only, never a reason text). The web shows «закрыт» and «Открыть снова»
/// from it; what actually closes a server is the memory in `state`, never this file.
fn publish_blocked(paths: &Paths, state: &State, now: u64) {
    let file = BlockedFile {
        v: 1,
        at: now,
        blocked: state
            .blocked
            .iter()
            .map(|(address, b)| BlockedEntry {
                address: address.clone(),
                at: b.at,
                code: b.code,
            })
            .collect(),
    };
    let Ok(bytes) = serde_json::to_vec_pretty(&file) else {
        return;
    };
    if let Err(e) = write_root_file(&paths.status_dir.join(BLOCKED_FILE), &bytes, 0o644) {
        eprintln!("launch: could not write the blocked list: {e}");
    }
}

fn refused(paths: &Paths, now: u64, id: Option<&str>, code: &str) {
    let mut status = LaunchStatus::new(RunState::Refused, now);
    status.request_id = id.map(str::to_string);
    status.reason = Some(code.to_string());
    write_status(paths, &status);
    eprintln!("launch: refused ({code})");
}

fn status_of(info: &LaunchInfo, state: RunState, now: u64) -> LaunchStatus {
    let mut status = LaunchStatus::new(state, now);
    status.request_id = Some(info.id.clone());
    status.brain = Some(info.brain);
    status.server = Some(info.server.clone());
    status.duration = Some(info.duration);
    status.sparring = Some(info.sparring);
    status.bundle = info.bundle.clone();
    status.finish = Some(info.finish);
    status.wb_smart = Some(info.wb_smart);
    status.no_selfkill = Some(info.no_selfkill);
    status.window_model = Some(info.window_model);
    status.preinput = Some(info.preinput);
    status
}

// ---------------------------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------------------------

/// A binary that accepts loopback favourites (a test build) must never run as the root helper: it would start a bot against a local
/// service named by a web-written file (task 5.12 review F5). `deploy/install-launcher.sh` refuses to install one as well.
fn refuse_test_build_as_root() -> Option<ExitCode> {
    let uid = std::fs::metadata("/proc/self").map_or(u32::MAX, |m| m.uid());
    if Rules::current().allow_loopback && uid == 0 {
        eprintln!("launch: refusing to run: this is a test build (+loopback-favourites) and this is the root helper");
        return Some(ExitCode::FAILURE);
    }
    None
}

pub fn run(args: LaunchArgs) -> ExitCode {
    if matches!(args.command, LaunchCommand::Apply(_) | LaunchCommand::Exited(_))
        && let Some(code) = refuse_test_build_as_root()
    {
        return code;
    }
    match args.command {
        LaunchCommand::Apply(a) => apply(&a),
        LaunchCommand::Exited(p) => exited(&p.resolve()),
        LaunchCommand::CheckProxy(a) => check_proxy(&a),
    }
}

fn mtime_secs(path: &Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs())
}

/// Removes whatever is at the request path, without following a link: a file or link by `unlink`, a directory (somebody made one
/// there, which would re-trigger the path unit for ever) with everything in it (`remove_dir_all` does not follow links).
fn discard(path: &Path) {
    let is_dir = std::fs::symlink_metadata(path).is_ok_and(|m| m.is_dir());
    let _ = if is_dir {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    };
}

fn apply(args: &ApplyArgs) -> ExitCode {
    let paths = args.paths.resolve();
    let now = unix_now();
    let request_path = args
        .request
        .clone()
        .unwrap_or_else(|| paths.launch_dir.join(REQUEST_FILE));

    // Consume the request: read it as a plain small file, then delete it before anything is done with it.
    let read = read_regular_nofollow_with_mtime(&request_path, MAX_REQUEST_BYTES);
    if matches!(read, Err(ReadError::Missing)) {
        eprintln!("launch: no request, nothing to do");
        return ExitCode::SUCCESS;
    }
    discard(&request_path);
    let (bytes, mtime) = match read {
        Ok(b) => b,
        Err(ReadError::TooLarge) => {
            refused(&paths, now, None, "request_too_large");
            return ExitCode::SUCCESS;
        }
        Err(ReadError::NotRegular) => {
            refused(&paths, now, None, "request_not_regular");
            return ExitCode::SUCCESS;
        }
        Err(_) => {
            refused(&paths, now, None, "request_unreadable");
            return ExitCode::SUCCESS;
        }
    };
    let req = match parse_request(&bytes) {
        Ok(r) => r,
        Err(e) => {
            refused(&paths, now, None, e.code());
            return ExitCode::SUCCESS;
        }
    };
    // Never carry out an old (or future-dated) request: one written while this unit was down must not act when it comes back.
    if !request_is_fresh(req.ts, mtime, now) {
        refused(&paths, now, Some(&req.id), "request_stale");
        return ExitCode::SUCCESS;
    }
    match req.action {
        Action::Stop => apply_stop(&paths, &req, now),
        Action::Start => apply_start(&paths, &req, now),
    }
}

fn apply_stop(paths: &Paths, req: &LaunchRequest, now: u64) -> ExitCode {
    let _ = with_state(&paths.state, |st| st.stop_requested_at = now);
    if let Ok(st) = load_state(&paths.state) {
        publish_blocked(paths, &st, now);
    }
    let ok = stop_all();
    reset_to_local(paths);
    let _ = systemctl(&["daemon-reload"]);
    let launch = load_state(&paths.state).ok().and_then(|s| s.last_launch);
    let mut status = match &launch {
        Some(info) => status_of(info, RunState::Stopped, now),
        None => LaunchStatus::new(RunState::Stopped, now),
    };
    status.request_id = Some(req.id.clone());
    if ok {
        status.reason = Some("stopped_by_owner".to_string());
    } else {
        status.state = RunState::Error;
        status.reason = Some("systemctl_failed".to_string());
    }
    write_status(paths, &status);
    eprintln!("launch: stop requested (ok={ok})");
    ExitCode::SUCCESS
}

fn apply_start(paths: &Paths, req: &LaunchRequest, now: u64) -> ExitCode {
    let id = Some(req.id.as_str());
    let live = match LiveServers::load_or_empty(&paths.live_servers) {
        Ok(l) => l,
        Err(_) => {
            refused(paths, now, id, "live_servers_unreadable");
            return ExitCode::SUCCESS;
        }
    };
    // Task 5.12: the owner's favourites, validated again here (the file is the web's, so nothing in it is trusted).
    let rules = Rules::current();
    let favs: FavouritesView = match favourites::load(&paths.launch_dir.join(favourites::FILE_NAME), rules) {
        Ok(f) => Ok(f),
        Err(favourites::LoadError::Unreadable) => Err("favourites_unreadable"),
        Err(favourites::LoadError::Invalid(_)) => Err("favourites_invalid"),
    };
    let live_mtime = mtime_secs(&paths.live_servers);
    let window_model = || configured_window_model(&paths.data_dir);
    let cat = Catalog {
        live: &live,
        favs: &favs,
        rules,
        live_servers_mtime: live_mtime,
        window_model: &window_model,
    };
    let state = load_state(&paths.state).ok();
    if let Some(st) = &state {
        publish_blocked(paths, st, now);
    }
    let bundle = || configured_bundle(&paths.config, &paths.data_dir);
    let plan = match decide(req, &cat, state.as_ref(), &bundle, now) {
        Ok(p) => p,
        Err(Refuse(code)) => {
            refused(paths, now, id, code);
            return ExitCode::SUCCESS;
        }
    };

    // The units: never start over a running bot, and the local server must be up for a local start.
    if matches!(
        active_state(BOT_UNIT).as_str(),
        "active" | "activating" | "reloading" | "deactivating"
    ) {
        refused(paths, now, id, "already_running");
        return ExitCode::SUCCESS;
    }
    if !only_our_dropin(&paths.dropin) {
        refused(paths, now, id, "unit_overridden");
        return ExitCode::SUCCESS;
    }
    if !plan.target.public && active_state(LOCAL_UNIT) != "active" {
        refused(paths, now, id, "local_server_down");
        return ExitCode::SUCCESS;
    }
    let secrets_dir = ddai_client::proxy::secrets_dir_for(&paths.data_dir);
    // The proxy is resolved on the allow-list plus the favourites, the one list the bot's own gate reads (`prepare_client`), so the
    // cgroup filter written here and the proxy the bot loads cannot disagree. The target was found in exactly one of them.
    let merged = match (&favs, plan.target.reopen) {
        (Ok(f), Reopen::Favourite { .. }) => live.clone().with_favourites(f).ok(),
        _ => Some(live.clone()),
    };
    let Some(merged) = merged else {
        refused(paths, now, id, "server_ambiguous");
        return ExitCode::SUCCESS;
    };
    let filter = match filter_for(&plan, &merged, &secrets_dir) {
        Ok(filter) => filter,
        Err(Refuse(code)) => {
            refused(paths, now, id, code);
            return ExitCode::SUCCESS;
        }
    };
    let (env, dropin) = match render_env(&req.id, &plan) {
        Ok(env) => (env, render_dropin(&filter)),
        Err(Refuse(code)) => {
            refused(paths, now, id, code);
            return ExitCode::SUCCESS;
        }
    };

    let info = LaunchInfo {
        id: req.id.clone(),
        brain: plan.brain,
        server: plan.target.key.clone(),
        duration: plan.duration,
        sparring: plan.sparring,
        public: plan.target.public,
        favourite: matches!(plan.target.reopen, Reopen::Favourite { .. }),
        bundle: plan.bundle.as_deref().map(bundle_run_name),
        finish: plan.finish,
        wb_smart: plan.wb_smart,
        no_selfkill: plan.no_selfkill,
        window_model: plan.window_model.is_some(),
        preinput: plan.preinput,
    };
    let fail = |code: &str| {
        let mut status = status_of(&info, RunState::Error, now);
        status.reason = Some(code.to_string());
        write_status(paths, &status);
        eprintln!("launch: error ({code})");
        ExitCode::SUCCESS
    };

    // The memory is committed BEFORE anything is written or started, and the policy is judged again on the fresh state inside the
    // lock: the exit hook may have recorded a kick or ban since the first look (review F4), and that must not be overwritten.
    let committed = with_state(&paths.state, |st| {
        decide(req, &cat, Some(st), &bundle, now)?;
        st.last_start_at = now;
        st.last_launch = Some(info.clone());
        st.stop_requested_at = 0;
        // Bans are never forgotten: one that the target's re-opening has lifted is inert (its time is older than the re-opening) and
        // keeps closing every OTHER address on the same IP until that favourite has its own re-opening.
        Ok::<(), Refuse>(())
    });
    match committed {
        Ok(Ok(())) => {}
        Ok(Err(Refuse(code))) => {
            refused(paths, now, id, code);
            return ExitCode::SUCCESS;
        }
        Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
            refused(paths, now, id, "state_unreadable");
            return ExitCode::SUCCESS;
        }
        Err(_) => return fail("state_write_failed"),
    }
    if let Ok(st) = load_state(&paths.state) {
        publish_blocked(paths, &st, now);
    }
    write_status(paths, &status_of(&info, RunState::Started, now));

    if write_root_file(&paths.env_file, env.as_bytes(), 0o644).is_err()
        || write_root_file(&paths.dropin, dropin.as_bytes(), 0o644).is_err()
    {
        return fail("write_failed");
    }
    if !systemctl(&["daemon-reload"]) {
        return fail("systemctl_failed");
    }
    // A bot that ended with 3/4 stays "failed" (RestartPreventExitStatus); an explicit, validated start clears that.
    let sparring_all = sparring_units(MAX_SPARRING);
    let mut reset = vec!["reset-failed", BOT_UNIT];
    reset.extend(sparring_all.iter().map(String::as_str));
    let _ = systemctl(&reset);

    if !systemctl(&["start", BOT_UNIT]) {
        return fail("systemctl_failed");
    }
    if plan.sparring > 0 {
        let units = sparring_units(plan.sparring);
        let mut start = vec!["start"];
        start.extend(units.iter().map(String::as_str));
        if !systemctl(&start) {
            let _ = stop_all();
            return fail("sparring_failed");
        }
    }
    eprintln!(
        "launch: started (brain {:?}, local {}, sparring {}, finish {}, wb-smart {}, no-selfkill {}, window-model {}, preinput {})",
        plan.brain,
        !plan.target.public,
        plan.sparring,
        plan.finish.flag_value(),
        plan.wb_smart.flag_value(),
        no_selfkill_flag_value(plan.no_selfkill),
        plan.window_model.is_some(),
        plan.preinput
    );
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------------------------
// check-proxy (the site's «Проверить», task 5.12)
// ---------------------------------------------------------------------------------------------

fn check_result_base(id: &str, proxy: &str, now: u64, code: &str) -> ProxyCheckResult {
    ProxyCheckResult {
        v: 1,
        id: id.to_string(),
        at: now,
        proxy: proxy.to_string(),
        ok: code == "ok",
        code: code.to_string(),
        relay: None,
        relay_mode: None,
        udp_rtt_ms: None,
        probe_sent: None,
        probe_replies: None,
    }
}

/// What the page is told about a finished check: fixed codes and numbers, never an address, a user name or a password (the
/// error's own text is not forwarded: it can name a step, and `Display` is for terminals).
fn check_result(id: &str, proxy: &str, now: u64, result: &Result<ProxyCheck, Socks5Error>) -> ProxyCheckResult {
    match result {
        Ok(c) => {
            let mut r = check_result_base(id, proxy, now, "ok");
            r.relay = Some(
                match c.relay_host {
                    RelayHost::SameAsProxy => "same_host",
                    RelayHost::Substituted => "substituted",
                    RelayHost::Remote => "remote",
                }
                .to_string(),
            );
            r.relay_mode = Some(c.mode.as_str().to_string());
            if let Some(p) = &c.probe {
                r.udp_rtt_ms = Some(u64::try_from(p.median.as_millis()).unwrap_or(u64::MAX));
                r.probe_sent = Some(p.sent);
                r.probe_replies = Some(p.replies);
            } else if let Some(s) = &c.sessions
                && let Some(Some(rtt)) = s.rtts.get(s.picked)
            {
                r.udp_rtt_ms = Some(u64::try_from(rtt.as_millis()).unwrap_or(u64::MAX));
            }
            r
        }
        Err(e) => {
            let code = match e {
                Socks5Error::UdpNotSupported => "udp_not_supported",
                Socks5Error::AuthFailed => "auth_failed",
                Socks5Error::NoAcceptableMethod | Socks5Error::UnsupportedMethod(_) => "no_auth_method",
                Socks5Error::Timeout { .. } => "timeout",
                Socks5Error::Io { .. } | Socks5Error::Closed { .. } | Socks5Error::ControlClosed => "connect_failed",
                Socks5Error::Refused { .. } => "refused",
                Socks5Error::Protocol(_) => "protocol_error",
                Socks5Error::RelayAddress(_) => "relay_refused",
                Socks5Error::ProbeFailed => "probe_failed",
            };
            check_result_base(id, proxy, now, code)
        }
    }
}

/// A profile the site made has a public IP-literal host (the site refuses anything else); one that does not is not the site's doing, and
/// the check must not connect there (review 5.12 F4). Hand-made profiles (no `managed_by`) are the owner's own and unchanged.
fn managed_host_refused(secrets_dir: &Path, name: &str, rules: Rules) -> bool {
    let Ok(path) = ddai_client::proxy::proxy_file_path(secrets_dir, name) else {
        return false;
    };
    let Ok(bytes) = read_regular_nofollow(&path, 64 * 1024) else {
        return false;
    };
    let Some(table) = String::from_utf8(bytes)
        .ok()
        .and_then(|t| t.parse::<toml::Table>().ok())
    else {
        return false;
    };
    if table.get("managed_by").and_then(toml::Value::as_str) != Some(ddai_web::serverbrowser::proxies::MANAGED_VALUE) {
        return false;
    }
    let host_ok = table
        .get("host")
        .and_then(toml::Value::as_str)
        .and_then(|h| h.parse::<std::net::IpAddr>().ok())
        .is_some_and(|ip| ddai_client::relay_rule::is_public_unicast(ip) || (rules.allow_loopback && ip.is_loopback()));
    !host_ok
}

fn check_proxy(args: &CheckProxyArgs) -> ExitCode {
    let data_dir = args.data_dir.clone().unwrap_or_else(|| match std::env::var_os("HOME") {
        Some(h) if !h.is_empty() => PathBuf::from(h).join("aiddnet").join("data"),
        _ => PathBuf::from("data"),
    });
    let launch_dir = data_dir.join("launch");
    let request_path = launch_dir.join(PROXY_CHECK_REQUEST_FILE);
    let read = read_regular_nofollow_with_mtime(&request_path, MAX_PROXY_CHECK_BYTES);
    if matches!(read, Err(ReadError::Missing)) {
        eprintln!("check-proxy: no request, nothing to do");
        return ExitCode::SUCCESS;
    }
    // Consumed before it is looked at: a request is never replayed.
    discard(&request_path);
    let now = unix_now();
    let write = |result: &ProxyCheckResult| {
        let Ok(bytes) = serde_json::to_vec_pretty(result) else {
            return;
        };
        if let Err(e) = write_atomic(&launch_dir, PROXY_CHECK_RESULT_FILE, &bytes, 0o644) {
            eprintln!("check-proxy: could not write the result: {e}");
        }
    };
    let (bytes, mtime) = match read {
        Ok(b) => b,
        Err(_) => {
            write(&check_result_base("", "", now, "bad_request"));
            return ExitCode::SUCCESS;
        }
    };
    let req = match parse_proxy_check_request(&bytes) {
        Ok(r) => r,
        Err(_) => {
            write(&check_result_base("", "", now, "bad_request"));
            return ExitCode::SUCCESS;
        }
    };
    if !request_is_fresh(req.ts, mtime, now) {
        write(&check_result_base(&req.id, &req.proxy, now, "request_stale"));
        return ExitCode::SUCCESS;
    }
    let secrets_dir = ddai_client::proxy::secrets_dir_for(&data_dir);
    if managed_host_refused(&secrets_dir, &req.proxy, Rules::current()) {
        write(&check_result_base(&req.id, &req.proxy, now, "proxy_host_refused"));
        return ExitCode::SUCCESS;
    }
    let cfg = match ddai_client::proxy::load_proxy(&secrets_dir, &req.proxy) {
        Ok(c) => c,
        Err(ddai_client::proxy::ProxyLoadError::Missing { .. }) => {
            write(&check_result_base(&req.id, &req.proxy, now, "proxy_missing"));
            return ExitCode::SUCCESS;
        }
        Err(_) => {
            write(&check_result_base(&req.id, &req.proxy, now, "proxy_file_bad"));
            return ExitCode::SUCCESS;
        }
    };
    // The check sends nothing to any game server (`ddai_client::socks5::check`): the handshake, `UDP ASSOCIATE`, and for a public
    // relay a DNS probe to a public resolver through it.
    let outcome = ddai_client::socks5::check(&cfg, &Timeouts::default());
    let result = check_result(&req.id, &req.proxy, unix_now(), &outcome);
    write(&result);
    eprintln!("check-proxy: done ({})", result.code);
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------------------------
// exited (ExecStopPost of the bot unit)
// ---------------------------------------------------------------------------------------------

/// A run counts as «time is up» when its duration was limited and it ran at least that long, less this slack (the bot's clock
/// starts a little after the launch).
const TIME_UP_SLACK_SECS: u64 = 60;

/// Whether a run started at `started_at` with `duration_secs` (0 = unlimited) ended because its time was up at `now`.
fn time_is_up(duration_secs: u64, started_at: u64, now: u64) -> bool {
    duration_secs > 0 && now >= started_at.saturating_add(duration_secs.saturating_sub(TIME_UP_SLACK_SECS))
}

/// How a bot's exit is told to the owner: `(state, reason)`. A clean exit nobody asked for is `finished` only when the time was
/// really up; otherwise (an outside SIGTERM, a hand `systemctl stop`) it is the neutral `ended`.
fn classify_exit(code: Option<i32>, stop_requested: bool, time_up: bool) -> (RunState, &'static str) {
    match code {
        Some(0) => (
            RunState::Stopped,
            if stop_requested {
                "stopped_by_owner"
            } else if time_up {
                "finished"
            } else {
                "ended"
            },
        ),
        Some(3) => (RunState::Failed, "kicked_or_banned"),
        Some(4) => (RunState::Failed, "join_failed"),
        None if stop_requested => (RunState::Stopped, "stopped_by_owner"),
        _ => (RunState::Failed, "crashed"),
    }
}

fn exited(paths: &Paths) -> ExitCode {
    let now = unix_now();
    // systemd's ExecStopPost environment: EXIT_CODE = exited|killed|dumped, EXIT_STATUS = the number (or signal name).
    let code = match std::env::var("EXIT_CODE").as_deref() {
        Ok("exited") => std::env::var("EXIT_STATUS")
            .ok()
            .and_then(|s| s.trim().parse::<i32>().ok()),
        _ => None,
    };
    let result = with_state(&paths.state, |st| {
        let stop_requested = st.stop_requested_at > 0
            && st.stop_requested_at >= st.last_start_at
            && now <= st.stop_requested_at.saturating_add(STOP_GRACE_SECS);
        let launch = st.last_launch.clone();
        let time_up = launch
            .as_ref()
            .is_some_and(|l| time_is_up(l.duration.seconds(), st.last_start_at, now));
        let (state, reason) = classify_exit(code, stop_requested, time_up);
        if let Some(c @ (3 | 4)) = code {
            st.last_exit = Some(ExitInfo { at: now, code: c });
            // A kick/ban on a public server (an allow-list entry or a favourite) stays in force until the owner re-opens it.
            if let Some(info) = &launch
                && info.public
            {
                st.blocked.insert(
                    info.server.clone(),
                    Block {
                        at: now,
                        code: c,
                        favourite: info.favourite,
                    },
                );
            }
        }
        st.stop_requested_at = 0;
        // No restart follows a normal end, a kick/ban/join failure (RestartPreventExitStatus) or a stop by the owner: the one-run
        // environment goes away, so a later hand start is the plain local default. A crash is restarted by systemd with the same.
        let no_restart = matches!(code, Some(0 | 3 | 4)) || stop_requested;
        (state, reason, launch, no_restart)
    });
    let Ok((state, reason, launch, no_restart)) = result else {
        eprintln!("launch: could not update the state");
        return ExitCode::SUCCESS;
    };
    if let Ok(st) = load_state(&paths.state) {
        publish_blocked(paths, &st, now);
    }
    if no_restart {
        reset_to_local(paths);
    }
    let mut status = match &launch {
        Some(info) => status_of(info, state, now),
        None => LaunchStatus::new(state, now),
    };
    status.reason = Some(reason.to_string());
    status.exit_code = code;
    write_status(paths, &status);
    eprintln!("launch: the bot ended ({reason}, exit {code:?})");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_web::launch::PROTOCOL_VERSION;

    fn live(toml: &str) -> LiveServers {
        toml::from_str(toml).expect("test toml")
    }

    const PUBLIC: &str =
        "[[server]]\naddress = \"203.0.113.5:8308\"\nnick = \"Muha\"\nready = true\nproxy = \"swarfey\"\n";
    const PUBLIC_NOT_READY: &str = "[[server]]\naddress = \"203.0.113.5:8308\"\nnick = \"Muha\"\nready = false\n";

    fn req(server: &str) -> LaunchRequest {
        LaunchRequest {
            v: PROTOCOL_VERSION,
            id: "0123456789abcdef".to_string(),
            ts: 0,
            action: Action::Start,
            brain: Some(Brain::HybridFly),
            server: Some(server.to_string()),
            duration: Some(DurationChoice::M15),
            sparring: Some(0),
            mirror: None,
            finish: None,
            wb_smart: None,
            no_selfkill: None,
            window_model: None,
            preinput: None,
        }
    }

    fn model_ok() -> Result<PathBuf, Refuse> {
        Ok(PathBuf::from("/data/bot/models/opp-m1.oppnet"))
    }

    fn bundle_ok() -> Result<PathBuf, Refuse> {
        Ok(PathBuf::from("/data/runs/E-005/e005-fly/checkpoints/final.bundle"))
    }

    fn decide_with(
        r: &LaunchRequest,
        l: &LiveServers,
        st: Option<&State>,
        mtime: u64,
        now: u64,
    ) -> Result<Plan, Refuse> {
        decide_fav(r, l, &Favourites::default(), st, mtime, now)
    }

    fn decide_fav(
        r: &LaunchRequest,
        l: &LiveServers,
        f: &Favourites,
        st: Option<&State>,
        mtime: u64,
        now: u64,
    ) -> Result<Plan, Refuse> {
        let favs: FavouritesView = Ok(f.clone());
        let cat = Catalog {
            live: l,
            favs: &favs,
            rules: Rules::default(),
            live_servers_mtime: mtime,
            window_model: &model_ok,
        };
        decide(r, &cat, st, &bundle_ok, now)
    }

    #[test]
    fn local_start_is_planned_with_the_local_address_and_name() {
        let plan = decide_with(&req("local"), &LiveServers::default(), Some(&State::default()), 0, 1000).unwrap();
        assert_eq!(plan.target.addr.to_string(), "127.0.0.1:8303");
        assert_eq!(plan.target.nick, "Muha");
        assert!(!plan.target.public);
        assert!(plan.bundle.is_some());
    }

    #[test]
    fn a_free_form_server_is_refused_even_if_it_is_an_address_the_owner_never_listed() {
        let l = live(PUBLIC);
        for s in [
            "127.0.0.1:8303",
            "1.2.3.4:8308",
            "203.0.113.5:8309",
            "Local",
            "",
            "203.0.113.5:8308 ",
            "../x",
        ] {
            assert_eq!(
                decide_with(&req(s), &l, Some(&State::default()), 0, 1000),
                Err(Refuse("server_not_allowed")),
                "{s:?}"
            );
        }
    }

    #[test]
    fn a_public_server_needs_ready() {
        let plan = decide_with(
            &req("203.0.113.5:8308"),
            &live(PUBLIC),
            Some(&State::default()),
            0,
            1000,
        )
        .unwrap();
        assert!(plan.target.public);
        assert_eq!(plan.target.nick, "Muha");
        assert_eq!(
            decide_with(
                &req("203.0.113.5:8308"),
                &live(PUBLIC_NOT_READY),
                Some(&State::default()),
                0,
                1000
            ),
            Err(Refuse("server_not_ready"))
        );
        // No `ready` key at all means not ready.
        let l = live("[[server]]\naddress = \"203.0.113.5:8308\"\nnick = \"Muha\"\n");
        assert_eq!(
            decide_with(&req("203.0.113.5:8308"), &l, Some(&State::default()), 0, 1000),
            Err(Refuse("server_not_ready"))
        );
    }

    #[test]
    fn bad_entries_never_become_a_command_line() {
        for entry in [
            "address = \"203.0.113.5:8308\"\nnick = \"Mu ha; rm\"\nready = true",
            "address = \"203.0.113.5:8308\"\nnick = \"\"\nready = true",
            "address = \"203.0.113.5:8308\"\nnick = \"AVeryLongNickname123\"\nready = true",
            "address = \"127.0.0.1:8308\"\nnick = \"Muha\"\nready = true",
            "address = \"example.org:8308\"\nnick = \"Muha\"\nready = true",
        ] {
            let l = live(&format!("[[server]]\n{entry}\n"));
            let addr = l.servers[0].address.clone();
            assert_eq!(
                decide_with(&req(&addr), &l, Some(&State::default()), 0, 1000),
                Err(Refuse("server_bad_entry")),
                "{entry}"
            );
        }
    }

    #[test]
    fn sparring_is_for_the_local_server_only_and_at_most_three() {
        let mut r = req("203.0.113.5:8308");
        r.sparring = Some(1);
        assert_eq!(
            decide_with(&r, &live(PUBLIC), Some(&State::default()), 0, 1000),
            Err(Refuse("sparring_local_only"))
        );
        let mut r = req("local");
        r.sparring = Some(3);
        assert_eq!(
            decide_with(&r, &live(PUBLIC), Some(&State::default()), 0, 1000)
                .unwrap()
                .sparring,
            3
        );
        r.sparring = Some(4);
        assert_eq!(
            decide_with(&r, &live(PUBLIC), Some(&State::default()), 0, 1000),
            Err(Refuse("bad_request"))
        );
    }

    #[test]
    fn a_missing_bundle_refuses_the_fly_brains_only() {
        let missing = || Err(Refuse("bundle_missing"));
        let mut r = req("local");
        let favs: FavouritesView = Ok(Favourites::default());
        let live = LiveServers::default();
        let cat = Catalog {
            live: &live,
            favs: &favs,
            rules: Rules::default(),
            live_servers_mtime: 0,
            window_model: &model_ok,
        };
        assert_eq!(
            decide(&r, &cat, Some(&State::default()), &missing, 1000),
            Err(Refuse("bundle_missing"))
        );
        r.brain = Some(Brain::Hybrid);
        assert!(decide(&r, &cat, Some(&State::default()), &missing, 1000).is_ok());
    }

    #[test]
    fn at_most_one_start_per_thirty_seconds() {
        let st = State {
            last_start_at: 1000,
            ..State::default()
        };
        let l = LiveServers::default();
        assert_eq!(
            decide_with(&req("local"), &l, Some(&st), 0, 1029),
            Err(Refuse("rate_limited"))
        );
        assert!(decide_with(&req("local"), &l, Some(&st), 0, 1030).is_ok());
        // A clock that went backwards does not unlock anything.
        assert_eq!(
            decide_with(&req("local"), &l, Some(&st), 0, 900),
            Err(Refuse("rate_limited"))
        );
    }

    #[test]
    fn after_exit_3_or_4_nothing_starts_for_a_while_but_a_crash_does_not_count() {
        let l = LiveServers::default();
        for code in [3, 4] {
            let st = State {
                last_exit: Some(ExitInfo { at: 1000, code }),
                ..State::default()
            };
            assert_eq!(
                decide_with(&req("local"), &l, Some(&st), 0, 1100),
                Err(Refuse("cooldown"))
            );
            assert!(decide_with(&req("local"), &l, Some(&st), 0, 1120).is_ok());
        }
        let st = State {
            last_exit: Some(ExitInfo { at: 1000, code: 1 }),
            ..State::default()
        };
        assert!(decide_with(&req("local"), &l, Some(&st), 0, 1001).is_ok());
    }

    #[test]
    fn a_banned_public_server_stays_refused_until_the_allow_list_is_edited_after_the_ban() {
        let l = live(PUBLIC);
        let mut st = State::default();
        st.blocked.insert(
            "203.0.113.5:8308".to_string(),
            Block {
                at: 5000,
                code: 3,
                favourite: false,
            },
        );
        // Long after the cool-down, still refused: the file is older than the ban (or the same second).
        assert_eq!(
            decide_with(&req("203.0.113.5:8308"), &l, Some(&st), 4000, 9000),
            Err(Refuse("blocked_after_ban"))
        );
        assert_eq!(
            decide_with(&req("203.0.113.5:8308"), &l, Some(&st), 5000, 9000),
            Err(Refuse("blocked_after_ban"))
        );
        // The owner re-opened it (edited the file after the ban).
        assert!(decide_with(&req("203.0.113.5:8308"), &l, Some(&st), 5001, 9000).is_ok());
        // The local server is not affected by a public server's ban.
        assert!(decide_with(&req("local"), &l, Some(&st), 4000, 9000).is_ok());
    }

    #[test]
    fn an_unreadable_state_refuses_every_start() {
        let l = live(PUBLIC);
        for s in ["203.0.113.5:8308", "local"] {
            assert_eq!(decide_with(&req(s), &l, None, 0, 9000), Err(Refuse("state_unreadable")));
        }
    }

    #[test]
    fn a_stop_is_not_a_plan() {
        let mut r = req("local");
        r.action = Action::Stop;
        assert_eq!(
            decide_with(&r, &LiveServers::default(), Some(&State::default()), 0, 1000),
            Err(Refuse("bad_request"))
        );
    }

    #[test]
    fn the_env_file_holds_only_validated_values_in_quotes() {
        let plan = decide_with(&req("local"), &LiveServers::default(), Some(&State::default()), 0, 1000).unwrap();
        let env = render_env("0123456789abcdef", &plan).unwrap();
        assert!(env.contains("BOT_SERVER=\"127.0.0.1:8303\"\n"), "{env}");
        assert!(env.contains("BOT_NAME=\"Muha\"\n"), "{env}");
        assert!(env.contains("BOT_BRAIN=\"hybrid\"\n"), "{env}");
        assert!(env.contains("BOT_DURATION=\"900\"\n"), "{env}");
        assert!(
            env.contains("BOT_FLY_ARGS=\"--fly-bundle /data/runs/E-005/e005-fly/checkpoints/final.bundle\"\n"),
            "{env}"
        );
        // Hybrid without the fly has no bundle argument at all.
        let mut r = req("local");
        r.brain = Some(Brain::Hybrid);
        r.duration = Some(DurationChoice::Unlimited);
        let plan = decide_with(&r, &LiveServers::default(), Some(&State::default()), 0, 1000).unwrap();
        let env = render_env("0123456789abcdef", &plan).unwrap();
        assert!(
            env.contains("BOT_FLY_ARGS=\"\"\n") && env.contains("BOT_DURATION=\"0\"\n"),
            "{env}"
        );
        // The opponent model is on unless the request says off (D-090); only `on` / `off` are ever written.
        assert!(env.contains("BOT_HYBRID_MIRROR=\"on\"\n"), "{env}");
        r.mirror = Some(Mirror::Off);
        let plan = decide_with(&r, &LiveServers::default(), Some(&State::default()), 0, 1000).unwrap();
        assert!(
            render_env("0123456789abcdef", &plan)
                .unwrap()
                .contains("BOT_HYBRID_MIRROR=\"off\"\n")
        );
        // Anything outside the closed character set is refused, never written.
        assert!(env_line("K", "[2001:db8::1]:8308").is_ok());
        for bad in ["a\"b", "a\nb", "a$b", "a`b", "a\\b", "a;b", "a\tb", "é"] {
            assert_eq!(env_line("K", bad), Err(Refuse("internal")), "{bad:?}");
        }
        assert_eq!(env_line("K", &"a".repeat(513)), Err(Refuse("internal")));
    }

    #[test]
    fn finishing_is_off_unless_asked_goes_to_the_env_and_the_pure_fly_takes_off_only() {
        let plan_for = |brain: Brain, finish: Option<Finish>, server: &str| {
            let mut r = req(server);
            r.brain = Some(brain);
            r.finish = finish;
            decide_with(&r, &LiveServers::default(), Some(&State::default()), 0, 1000)
        };
        let env_of = |plan: &Plan| render_env("0123456789abcdef", plan).unwrap();
        // No field (an old request) is off, and `off` is written explicitly so a stale env value can never leak through.
        let plan = plan_for(Brain::Hybrid, None, "local").unwrap();
        assert_eq!(plan.finish, Finish::Off);
        assert!(env_of(&plan).contains("BOT_FINISH=\"off\"\n"), "{}", env_of(&plan));
        // The hybrid brains take all four words (`wb`: task 3.18, D-114).
        for brain in [Brain::Hybrid, Brain::HybridFly] {
            for (finish, word) in [
                (Finish::Off, "off"),
                (Finish::Target, "target"),
                (Finish::Wb, "wb"),
                (Finish::Full, "full"),
            ] {
                let plan = plan_for(brain, Some(finish), "local").unwrap();
                assert_eq!(plan.finish, finish);
                assert!(
                    env_of(&plan).contains(&format!("BOT_FINISH=\"{word}\"\n")),
                    "{brain:?} {word}"
                );
            }
        }
        // The pure fly: `off` (or nothing) passes, `target`, `wb` and `full` are refused with their own code, nothing is planned.
        assert!(plan_for(Brain::Fly, None, "local").is_ok());
        assert!(plan_for(Brain::Fly, Some(Finish::Off), "local").is_ok());
        for finish in [Finish::Target, Finish::Wb, Finish::Full] {
            assert_eq!(
                plan_for(Brain::Fly, Some(finish), "local").unwrap_err(),
                Refuse("finish_hybrid_only")
            );
        }
        // Mirror and finishing are independent lines.
        let mut r = req("local");
        r.finish = Some(Finish::Target);
        r.mirror = Some(Mirror::Off);
        let plan = decide_with(&r, &LiveServers::default(), Some(&State::default()), 0, 1000).unwrap();
        let env = env_of(&plan);
        assert!(
            env.contains("BOT_HYBRID_MIRROR=\"off\"\n") && env.contains("BOT_FINISH=\"target\"\n"),
            "{env}"
        );
        // The status the site reads names the mode; a launch remembered before the field existed reads as off.
        let info = LaunchInfo {
            id: "0123456789abcdef".to_string(),
            brain: Brain::Hybrid,
            server: "local".to_string(),
            duration: DurationChoice::M15,
            sparring: 0,
            public: false,
            favourite: false,
            bundle: None,
            finish: Finish::Target,
            wb_smart: WbSmart::Off,
            no_selfkill: false,
            window_model: false,
            preinput: false,
        };
        assert_eq!(status_of(&info, RunState::Started, 5).finish, Some(Finish::Target));
        let mut old = serde_json::to_value(&info).unwrap();
        old.as_object_mut().unwrap().remove("finish");
        let old: LaunchInfo = serde_json::from_value(old).unwrap();
        assert_eq!(old.finish, Finish::Off);
    }

    #[test]
    fn the_smart_wayblock_and_the_duel_switch_are_off_unless_asked_go_to_the_env_and_every_brain_takes_them() {
        // Task 5.15.
        let plan_for = |brain: Brain, wb: Option<WbSmart>, ns: Option<bool>| {
            let mut r = req("local");
            r.brain = Some(brain);
            r.wb_smart = wb;
            r.no_selfkill = ns;
            decide_with(&r, &LiveServers::default(), Some(&State::default()), 0, 1000)
        };
        let env_of = |plan: &Plan| render_env("0123456789abcdef", plan).unwrap();
        // No fields (an old request): both are written explicitly as off, so a stale value in the environment can never leak into the run.
        let plan = plan_for(Brain::Hybrid, None, None).unwrap();
        assert_eq!((plan.wb_smart, plan.no_selfkill), (WbSmart::Off, false));
        let env = env_of(&plan);
        assert!(
            env.contains("BOT_WB_SMART=\"off\"\n") && env.contains("BOT_NO_SELFKILL=\"false\"\n"),
            "{env}"
        );
        // Every brain, the pure fly included, takes both; the two lines are independent of each other and of the others.
        for brain in [Brain::Hybrid, Brain::HybridFly, Brain::Fly] {
            for (wb, wb_word) in [(WbSmart::Off, "off"), (WbSmart::On, "on")] {
                for (ns, ns_word) in [(false, "false"), (true, "true")] {
                    let plan = plan_for(brain, Some(wb), Some(ns)).unwrap();
                    assert_eq!((plan.wb_smart, plan.no_selfkill), (wb, ns), "{brain:?}");
                    let env = env_of(&plan);
                    assert!(
                        env.contains(&format!("BOT_WB_SMART=\"{wb_word}\"\n"))
                            && env.contains(&format!("BOT_NO_SELFKILL=\"{ns_word}\"\n"))
                            && env.contains("BOT_FINISH=\"off\"\n"),
                        "{brain:?} {wb_word} {ns_word}: {env}"
                    );
                }
            }
        }
        // The status the site reads names both; a launch remembered before the fields existed reads as off / false.
        let info = LaunchInfo {
            id: "0123456789abcdef".to_string(),
            brain: Brain::Hybrid,
            server: "local".to_string(),
            duration: DurationChoice::M15,
            sparring: 0,
            public: false,
            favourite: false,
            bundle: None,
            finish: Finish::Off,
            wb_smart: WbSmart::On,
            no_selfkill: true,
            window_model: false,
            preinput: false,
        };
        let status = status_of(&info, RunState::Started, 5);
        assert_eq!((status.wb_smart, status.no_selfkill), (Some(WbSmart::On), Some(true)));
        let mut old = serde_json::to_value(&info).unwrap();
        old.as_object_mut().unwrap().remove("wb_smart");
        old.as_object_mut().unwrap().remove("no_selfkill");
        let old: LaunchInfo = serde_json::from_value(old).unwrap();
        assert_eq!((old.wb_smart, old.no_selfkill), (WbSmart::Off, false));
    }

    #[test]
    fn the_opponent_predictor_is_off_unless_asked_names_its_file_in_the_env_and_the_pure_fly_refuses_it() {
        // Task 3.17 (D-111).
        let plan_for = |brain: Brain, wm: Option<bool>| {
            let mut r = req("local");
            r.brain = Some(brain);
            r.window_model = wm;
            decide_with(&r, &LiveServers::default(), Some(&State::default()), 0, 1000)
        };
        let env_of = |plan: &Plan| render_env("0123456789abcdef", plan).unwrap();
        // No field (an old request) and `false`: the line is written explicitly and empty, so a stale path in the environment never leaks in.
        for wm in [None, Some(false)] {
            let plan = plan_for(Brain::Hybrid, wm).unwrap();
            assert_eq!(plan.window_model, None);
            let env = env_of(&plan);
            assert!(env.contains("BOT_WINDOW_MODEL=\"\"\n"), "{env}");
        }
        // On: the helper's own path, for both hybrids; the other lines are unchanged.
        for brain in [Brain::Hybrid, Brain::HybridFly] {
            let plan = plan_for(brain, Some(true)).unwrap();
            assert_eq!(plan.window_model, Some(PathBuf::from("/data/bot/models/opp-m1.oppnet")));
            let env = env_of(&plan);
            assert!(
                env.contains("BOT_WINDOW_MODEL=\"/data/bot/models/opp-m1.oppnet\"\n")
                    && env.contains("BOT_FINISH=\"off\"\n")
                    && env.contains("BOT_WB_SMART=\"off\"\n"),
                "{env}"
            );
        }
        // The pure fly has no lag window: refused, however the rest looks.
        assert_eq!(
            plan_for(Brain::Fly, Some(true)).unwrap_err(),
            Refuse("window_model_hybrid_only")
        );
        assert!(plan_for(Brain::Fly, Some(false)).is_ok());
        // A missing file refuses only a request that asks for it.
        let favs: FavouritesView = Ok(Favourites::default());
        let live = LiveServers::default();
        let missing = || Err(Refuse("window_model_missing"));
        let cat = Catalog {
            live: &live,
            favs: &favs,
            rules: Rules::default(),
            live_servers_mtime: 0,
            window_model: &missing,
        };
        let mut r = req("local");
        r.brain = Some(Brain::Hybrid);
        r.window_model = Some(true);
        assert_eq!(
            decide(&r, &cat, Some(&State::default()), &bundle_ok, 1000),
            Err(Refuse("window_model_missing"))
        );
        r.window_model = None;
        assert!(decide(&r, &cat, Some(&State::default()), &bundle_ok, 1000).is_ok());
        // The status the site reads names it; a launch remembered before the field existed reads as false.
        let info = LaunchInfo {
            id: "0123456789abcdef".to_string(),
            brain: Brain::Hybrid,
            server: "local".to_string(),
            duration: DurationChoice::M15,
            sparring: 0,
            public: false,
            favourite: false,
            bundle: None,
            finish: Finish::Off,
            wb_smart: WbSmart::Off,
            no_selfkill: false,
            window_model: true,
            preinput: false,
        };
        assert_eq!(status_of(&info, RunState::Started, 5).window_model, Some(true));
        let mut old = serde_json::to_value(&info).unwrap();
        old.as_object_mut().unwrap().remove("window_model");
        let old: LaunchInfo = serde_json::from_value(old).unwrap();
        assert!(!old.window_model);
    }

    #[test]
    fn the_preinput_switch_is_off_unless_asked_is_written_always_and_the_pure_fly_refuses_it() {
        // Task 3.20b (D-112).
        let plan_for = |brain: Brain, pre: Option<bool>| {
            let mut r = req("local");
            r.brain = Some(brain);
            r.preinput = pre;
            decide_with(&r, &LiveServers::default(), Some(&State::default()), 0, 1000)
        };
        let env_of = |plan: &Plan| render_env("0123456789abcdef", plan).unwrap();
        // No field (an old request) and `false`: the line is written explicitly as `off`, so a stale value in the environment never leaks in.
        for pre in [None, Some(false)] {
            let plan = plan_for(Brain::Hybrid, pre).unwrap();
            assert!(!plan.preinput);
            let env = env_of(&plan);
            assert!(env.contains("BOT_PREINPUT=\"off\"\n"), "{env}");
        }
        // On: both hybrids; the other lines are unchanged.
        for brain in [Brain::Hybrid, Brain::HybridFly] {
            let plan = plan_for(brain, Some(true)).unwrap();
            assert!(plan.preinput);
            let env = env_of(&plan);
            assert!(
                env.contains("BOT_PREINPUT=\"on\"\n")
                    && env.contains("BOT_FINISH=\"off\"\n")
                    && env.contains("BOT_WINDOW_MODEL=\"\"\n"),
                "{env}"
            );
        }
        // The pure fly is not offered the pre-inputs: refused, `false` is fine.
        assert_eq!(
            plan_for(Brain::Fly, Some(true)).unwrap_err(),
            Refuse("preinput_hybrid_only")
        );
        assert!(plan_for(Brain::Fly, Some(false)).is_ok());
        // The status the site reads names it; a launch remembered before the field existed reads as false.
        let info = LaunchInfo {
            id: "0123456789abcdef".to_string(),
            brain: Brain::Hybrid,
            server: "local".to_string(),
            duration: DurationChoice::M15,
            sparring: 0,
            public: false,
            favourite: false,
            bundle: None,
            finish: Finish::Wb,
            wb_smart: WbSmart::Off,
            no_selfkill: false,
            window_model: false,
            preinput: true,
        };
        let status = status_of(&info, RunState::Started, 5);
        assert_eq!((status.preinput, status.finish), (Some(true), Some(Finish::Wb)));
        let mut old = serde_json::to_value(&info).unwrap();
        old.as_object_mut().unwrap().remove("preinput");
        let old: LaunchInfo = serde_json::from_value(old).unwrap();
        assert!(!old.preinput);
    }

    #[test]
    fn the_model_file_must_be_a_plain_regular_file_under_the_data_dir() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path();
        let model = data.join(ddai_web::launch::DEFAULT_WINDOW_MODEL_REL);
        // Nothing there, an empty file, a directory, a symlink and an oversized file are all "missing": the bot would not start with them.
        assert_eq!(configured_window_model(data), Err(Refuse("window_model_missing")));
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"").unwrap();
        assert_eq!(
            configured_window_model(data),
            Err(Refuse("window_model_missing")),
            "empty"
        );
        std::fs::remove_file(&model).unwrap();
        std::fs::create_dir(&model).unwrap();
        assert_eq!(
            configured_window_model(data),
            Err(Refuse("window_model_missing")),
            "directory"
        );
        std::fs::remove_dir(&model).unwrap();
        let real = data.join("real.oppnet");
        std::fs::write(&real, b"x").unwrap();
        std::os::unix::fs::symlink(&real, &model).unwrap();
        assert_eq!(
            configured_window_model(data),
            Err(Refuse("window_model_missing")),
            "symlink"
        );
        std::fs::remove_file(&model).unwrap();
        std::fs::write(&model, vec![0u8; (MAX_WINDOW_MODEL_BYTES + 1) as usize]).unwrap();
        assert_eq!(
            configured_window_model(data),
            Err(Refuse("window_model_missing")),
            "oversized"
        );
        std::fs::write(&model, b"weights").unwrap();
        assert_eq!(configured_window_model(data), Ok(model));
        // A data directory with a character outside the closed set cannot be written to the environment: refused up front.
        let odd = data.join("a b");
        std::fs::create_dir_all(&odd).unwrap();
        assert_eq!(configured_window_model(&odd), Err(Refuse("window_model_bad_path")));
        assert_eq!(
            configured_window_model(Path::new("relative/dir")),
            Err(Refuse("window_model_bad_path"))
        );
    }

    #[test]
    fn the_cgroup_drop_in_resets_the_list_and_adds_only_the_given_ips() {
        let local = render_dropin(&Filter::Allow(Vec::new()));
        assert!(
            local.contains("[Service]\nIPAddressAllow=\nIPAddressAllow=127.0.0.0/8 ::1\n"),
            "{local}"
        );
        assert_eq!(local.matches("IPAddressAllow=").count(), 2);
        assert!(!local.contains("IPAddressDeny"), "{local}");
        let proxied = render_dropin(&Filter::Allow(vec![
            "198.51.100.7".parse().unwrap(),
            "2001:db8::1".parse().unwrap(),
        ]));
        assert!(
            proxied.contains("IPAddressAllow=198.51.100.7\nIPAddressAllow=2001:db8::1\n"),
            "{proxied}"
        );
    }

    /// Task 2.6b: `relay = "public"`. Both lists are reset and every server IP is denied; no allow entry may exist after
    /// the reset (an allow match beats a deny match: probed on systemd 255, `tools/e2e/ipfilter_probe.sh` S2, S5, S7).
    #[test]
    fn the_public_relay_drop_in_resets_both_lists_and_denies_the_server_and_the_private_ranges() {
        let private = [
            "IPAddressDeny=10.0.0.0/8",
            "IPAddressDeny=172.16.0.0/12",
            "IPAddressDeny=192.168.0.0/16",
            "IPAddressDeny=169.254.0.0/16",
            "IPAddressDeny=100.64.0.0/10",
            "IPAddressDeny=fc00::/7",
            "IPAddressDeny=fe80::/10",
        ];
        let body_of =
            |d: &str| -> Vec<String> { d.lines().filter(|l| !l.starts_with('#')).map(str::to_string).collect() };
        // A server with both families: its IPs, no `::/0` (the relay may be IPv6), then the private ranges.
        let drop_in = render_dropin(&Filter::DenyServer(vec![
            "203.0.113.5".parse().unwrap(),
            "2001:db8::7".parse().unwrap(),
        ]));
        let mut want: Vec<String> = [
            "[Service]",
            "IPAddressAllow=",
            "IPAddressDeny=",
            "IPAddressDeny=203.0.113.5",
            "IPAddressDeny=2001:db8::7",
        ]
        .map(String::from)
        .to_vec();
        want.extend(private.map(String::from));
        assert_eq!(body_of(&drop_in), want, "{drop_in}");
        // An IPv4-only server (Swarfey): the whole IPv6 family is denied too (F3), right after the server's IPs.
        let v4 = render_dropin(&Filter::DenyServer(vec!["203.0.113.5".parse().unwrap()]));
        let mut want: Vec<String> = [
            "[Service]",
            "IPAddressAllow=",
            "IPAddressDeny=",
            "IPAddressDeny=203.0.113.5",
            "IPAddressDeny=::/0",
        ]
        .map(String::from)
        .to_vec();
        want.extend(private.map(String::from));
        assert_eq!(body_of(&v4), want, "{v4}");
        // An IPv6-only server: nothing denies the IPv4 family (the proxy is usually reached over it).
        let v6 = render_dropin(&Filter::DenyServer(vec!["2001:db8::7".parse().unwrap()]));
        assert!(!v6.contains("0.0.0.0/0") && !v6.contains("::/0"), "{v6}");
        // Not a single non-empty allow entry, no `any`, no loopback entry, in any spelling.
        for d in [&drop_in, &v4, &v6] {
            assert!(
                !d.contains("any") && !d.contains("127.0.0.0/8") && !d.contains("::1\n"),
                "{d}"
            );
            for line in d.lines().filter(|l| l.starts_with("IPAddressAllow")) {
                assert_eq!(line, "IPAddressAllow=", "{d}");
            }
        }
    }

    fn proxy_dir(extra: &str) -> tempfile::TempDir {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("swarfey-proxy.toml");
        std::fs::write(
            &path,
            format!("host = \"198.51.100.7\"\nport = 1080\nuser = \"u\"\npass = \"p\"\nfor_server = \"203.0.113.5:8308\"\n{extra}"),
        )
        .unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        dir
    }

    fn plan_for(toml: &str) -> (Plan, LiveServers) {
        let l = live(toml);
        let plan = decide_with(&req("203.0.113.5:8308"), &l, Some(&State::default()), 0, 1000).expect("plan");
        (plan, l)
    }

    #[test]
    fn the_filter_follows_the_proxy_files_relay_mode() {
        let (plan, l) = plan_for(PUBLIC);
        let server: IpAddr = "203.0.113.5".parse().unwrap();
        // Default (and explicit "proxy-host-only"): the proxy's own IP is allowed, the server's is not mentioned.
        for extra in ["", "relay = \"proxy-host-only\"\n"] {
            let dir = proxy_dir(extra);
            assert_eq!(
                filter_for(&plan, &l, dir.path()),
                Ok(Filter::Allow(vec!["198.51.100.7".parse().unwrap()])),
                "{extra:?}"
            );
        }
        // relay = "public": the SERVER's IPs are denied (once each, even though `for_server` names the same IP), and
        // the proxy's own address is not special.
        let dir = proxy_dir("relay = \"public\"\n");
        assert_eq!(filter_for(&plan, &l, dir.path()), Ok(Filter::DenyServer(vec![server])));
        // A local launch needs nothing extra whatever the files say.
        let local = decide_with(&req("local"), &l, Some(&State::default()), 0, 1000).expect("plan");
        assert_eq!(filter_for(&local, &l, dir.path()), Ok(Filter::Allow(Vec::new())));
        // An entry without a proxy: the server's IP is allowed, as before.
        let (direct, dl) = plan_for("[[server]]\naddress = \"203.0.113.5:8308\"\nnick = \"Muha\"\nready = true\n");
        assert_eq!(filter_for(&direct, &dl, dir.path()), Ok(Filter::Allow(vec![server])));
        // A missing or unsafe proxy file is a refusal, never a filter.
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(filter_for(&plan, &l, empty.path()), Err(Refuse("proxy_error")));
    }

    #[test]
    fn a_public_relay_filter_denies_an_ipv6_server_address() {
        let l =
            live("[[server]]\naddress = \"[2001:db8::7]:8308\"\nnick = \"Muha\"\nready = true\nproxy = \"swarfey\"\n");
        let plan = decide_with(&req("[2001:db8::7]:8308"), &l, Some(&State::default()), 0, 1000).expect("plan");
        let dir = tempfile::tempdir().unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            let path = dir.path().join("swarfey-proxy.toml");
            std::fs::write(
                &path,
                "host = \"198.51.100.7\"\nport = 1080\nfor_server = \"[2001:db8::7]:8308\"\nrelay = \"public\"\n",
            )
            .unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert_eq!(
            filter_for(&plan, &l, dir.path()),
            Ok(Filter::DenyServer(vec!["2001:db8::7".parse().unwrap()]))
        );
    }

    #[test]
    fn exits_are_told_honestly() {
        assert_eq!(
            classify_exit(Some(0), true, false),
            (RunState::Stopped, "stopped_by_owner")
        );
        assert_eq!(classify_exit(Some(0), false, true), (RunState::Stopped, "finished"));
        // An outside SIGTERM or a hand stop is not «time is up».
        assert_eq!(classify_exit(Some(0), false, false), (RunState::Stopped, "ended"));
        assert_eq!(
            classify_exit(Some(3), true, false),
            (RunState::Failed, "kicked_or_banned")
        );
        assert_eq!(classify_exit(Some(4), false, false), (RunState::Failed, "join_failed"));
        assert_eq!(classify_exit(Some(1), false, false), (RunState::Failed, "crashed"));
        assert_eq!(classify_exit(None, false, false), (RunState::Failed, "crashed"));
        assert_eq!(
            classify_exit(None, true, false),
            (RunState::Stopped, "stopped_by_owner")
        );
    }

    #[test]
    fn time_is_up_only_for_a_limited_run_that_ran_its_time() {
        // 15 minutes: up from 14 minutes (60 s slack) on; never for «до остановки».
        assert!(!time_is_up(900, 1000, 1000 + 10));
        assert!(!time_is_up(900, 1000, 1000 + 839));
        assert!(time_is_up(900, 1000, 1000 + 840));
        assert!(time_is_up(900, 1000, 1000 + 905));
        assert!(!time_is_up(0, 1000, 1000 + 100_000));
        assert!(!time_is_up(30, 1000, 999), "a clock that went backwards");
    }

    #[test]
    fn the_bundle_comes_from_a_trusted_config_with_a_plain_path() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path();
        let bundle = data.join("runs/E-005/e005-fly/checkpoints/final.bundle");
        std::fs::create_dir_all(bundle.parent().unwrap()).unwrap();
        std::fs::write(&bundle, b"x").unwrap();
        // No config: the default bundle under the data dir.
        assert_eq!(configured_bundle(&data.join("none.toml"), data).unwrap(), bundle);
        let cfg = data.join("launch.toml");
        let write = |body: &str, mode: u32| {
            std::fs::write(&cfg, body).unwrap();
            std::fs::set_permissions(&cfg, std::fs::Permissions::from_mode(mode)).unwrap();
        };
        let other = data.join("other.bundle");
        std::fs::write(&other, b"x").unwrap();
        write(&format!("fly_bundle = \"{}\"\n", other.display()), 0o644);
        assert_eq!(configured_bundle(&cfg, data).unwrap(), other);
        // Group/world-writable config, unknown keys, odd paths and a missing file are refused.
        write(&format!("fly_bundle = \"{}\"\n", other.display()), 0o666);
        assert_eq!(configured_bundle(&cfg, data), Err(Refuse("config_untrusted")));
        write("fly_bundle = \"/x\"\nextra = 1\n", 0o644);
        assert_eq!(configured_bundle(&cfg, data), Err(Refuse("config_bad")));
        for bad in [
            "relative.bundle",
            "/a/../b.bundle",
            "/a b/c.bundle",
            "/a/$X.bundle",
            "/a/final.txt",
        ] {
            write(&format!("fly_bundle = \"{bad}\"\n"), 0o644);
            assert_eq!(configured_bundle(&cfg, data), Err(Refuse("bundle_bad_path")), "{bad}");
        }
        write("fly_bundle = \"/nonexistent/final.bundle\"\n", 0o644);
        assert_eq!(configured_bundle(&cfg, data), Err(Refuse("bundle_missing")));
    }

    #[test]
    fn the_state_survives_a_round_trip_and_a_corrupt_file_is_never_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub/state.json");
        with_state(&path, |st| {
            st.last_start_at = 7;
            st.blocked.insert(
                "a:1".into(),
                Block {
                    at: 9,
                    code: 3,
                    favourite: false,
                },
            );
        })
        .unwrap();
        assert_eq!(
            load_state(&path).unwrap().blocked["a:1"],
            Block {
                at: 9,
                code: 3,
                favourite: false
            }
        );
        std::fs::write(&path, b"{ not json").unwrap();
        assert!(load_state(&path).is_err());
        assert!(with_state(&path, |_| ()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"{ not json");
    }

    // --- task 5.12 (D-099): favourites -------------------------------------------------------------------------

    const FAV_ADDR: &str = "45.141.57.35:8308";
    const FAV_SIBLING: &str = "45.141.57.35:8309";
    const FAV_OTHER: &str = "93.184.216.34:8303";

    fn fav(address: &str, connection: &str) -> Favourite {
        Favourite {
            address: address.to_string(),
            name: "Some Block Server".to_string(),
            nick: "Muha".to_string(),
            connection: connection.to_string(),
            consent_at: 1_700_000_000,
            notes: String::new(),
            added_at: 1_700_000_000,
            reopened_at: 0,
        }
    }

    fn favs(list: Vec<Favourite>) -> Favourites {
        Favourites {
            v: favourites::FILE_VERSION,
            favourites: list,
        }
    }

    fn blocked_state(address: &str, at: u64) -> State {
        let mut st = State::default();
        st.blocked.insert(
            address.to_string(),
            Block {
                at,
                code: 3,
                favourite: false,
            },
        );
        st
    }

    #[test]
    fn a_favourite_is_planned_like_a_ready_entry_with_its_own_nick_and_nothing_from_the_request() {
        let mut f = fav(FAV_ADDR, "direct");
        f.nick = "Muha2".to_string();
        let plan = decide_fav(
            &req(FAV_ADDR),
            &LiveServers::default(),
            &favs(vec![f]),
            Some(&State::default()),
            0,
            1000,
        )
        .unwrap();
        assert_eq!(plan.target.addr.to_string(), FAV_ADDR);
        assert_eq!(plan.target.nick, "Muha2");
        assert!(plan.target.public);
        assert_eq!(plan.target.reopen, Reopen::Favourite { reopened_at: 0 });
        // Not a favourite: refused, whatever it looks like.
        for s in [
            FAV_OTHER,
            FAV_SIBLING,
            "45.141.57.36:8308",
            "127.0.0.1:8463",
            "Local",
            "",
            "45.141.57.35:8308 ",
        ] {
            assert_eq!(
                decide_fav(
                    &req(s),
                    &LiveServers::default(),
                    &favs(vec![fav(FAV_ADDR, "direct")]),
                    Some(&State::default()),
                    0,
                    1000
                ),
                Err(Refuse("server_not_allowed")),
                "{s:?}"
            );
        }
        // Sparring is for the local server only, favourites included.
        let mut r = req(FAV_ADDR);
        r.sparring = Some(1);
        assert_eq!(
            decide_fav(
                &r,
                &LiveServers::default(),
                &favs(vec![fav(FAV_ADDR, "direct")]),
                Some(&State::default()),
                0,
                1000
            ),
            Err(Refuse("sparring_local_only"))
        );
    }

    #[test]
    fn a_favourite_the_helper_cannot_trust_is_refused_whatever_the_file_said() {
        let live = LiveServers::default();
        let st = State::default();
        let go = |f: Favourite| decide_fav(&req(&f.address), &live, &favs(vec![f]), Some(&st), 0, 1000);
        // Built by hand (a writer bug, a hand edit): the helper validates again.
        for (bad, code) in [
            (fav("10.0.0.5:8308", "direct"), "bad_address"),
            (fav("127.0.0.1:8308", "direct"), "bad_address"),
            (fav("203.0.113.5:8308", "direct"), "bad_address"),
            (fav("example.com:8308", "direct"), "bad_address"),
            (fav(FAV_ADDR, "proxy:../x"), "bad_connection"),
            (fav(FAV_ADDR, "proxy:"), "bad_connection"),
            (fav(FAV_ADDR, "Direct"), "bad_connection"),
            (
                Favourite {
                    consent_at: 0,
                    ..fav(FAV_ADDR, "direct")
                },
                "consent_required",
            ),
            (
                Favourite {
                    nick: "Mu ha; rm".to_string(),
                    ..fav(FAV_ADDR, "direct")
                },
                "bad_nick",
            ),
            (
                Favourite {
                    nick: String::new(),
                    ..fav(FAV_ADDR, "direct")
                },
                "bad_nick",
            ),
            (
                Favourite {
                    name: "a\nb".to_string(),
                    ..fav(FAV_ADDR, "direct")
                },
                "bad_name",
            ),
        ] {
            assert_eq!(go(bad.clone()), Err(Refuse(code)), "{bad:?}");
        }
        // A favourites file that cannot be trusted: no favourite can start; local still can.
        for why in ["favourites_invalid", "favourites_unreadable"] {
            let view: FavouritesView = Err(why);
            let cat = Catalog {
                live: &live,
                favs: &view,
                rules: Rules::default(),
                live_servers_mtime: 0,
                window_model: &model_ok,
            };
            assert_eq!(
                decide(&req(FAV_ADDR), &cat, Some(&st), &bundle_ok, 1000),
                Err(Refuse(why))
            );
            assert!(decide(&req("local"), &cat, Some(&st), &bundle_ok, 1000).is_ok());
        }
        // The allow-list also names it: two statements about one server are refused.
        let both = live_from(&format!(
            "[[server]]\naddress = \"{FAV_ADDR}\"\nnick = \"Muha\"\nready = false\n"
        ));
        let list = favs(vec![fav(FAV_ADDR, "direct")]);
        let clash = decide_fav(&req(FAV_ADDR), &both, &list, Some(&st), 0, 1000);
        assert_eq!(
            clash,
            Err(Refuse("server_not_ready")),
            "the allow-list entry is judged first, as it always was"
        );
        let ready_both = live_from(&format!(
            "[[server]]\naddress = \"{FAV_ADDR}\"\nnick = \"Muha\"\nready = true\n"
        ));
        assert!(
            decide_fav(&req(FAV_ADDR), &ready_both, &list, Some(&st), 0, 1000).is_ok(),
            "an exact allow-list entry wins, nothing else changes"
        );
        // ... and a favourite that only resembles an allow-list entry on the same address under another spelling is a clash.
        let list2 = favs(vec![fav(FAV_SIBLING, "direct")]);
        let clash = live_from(&format!(
            "[[server]]\naddress = \"{FAV_SIBLING}\"\nnick = \"X\"\nready = true\n"
        ));
        // (exact string equal -> the allow-list branch.) A different spelling never reaches here: the favourite parse refuses it.
        assert!(decide_fav(&req(FAV_SIBLING), &clash, &list2, Some(&st), 0, 1000).is_ok());
        // The allow-list names the same server in another spelling of the same address (an IPv6 literal in capitals): the favourite
        // is a second statement about one server, and refused.
        let v6_live = live_from("[[server]]\naddress = \"[2A01:4F8::1]:8303\"\nnick = \"X\"\nready = true\n");
        let v6_fav = favs(vec![fav("[2a01:4f8::1]:8303", "direct")]);
        assert_eq!(
            decide_fav(&req("[2a01:4f8::1]:8303"), &v6_live, &v6_fav, Some(&st), 0, 1000),
            Err(Refuse("server_ambiguous"))
        );
        // Loopback is refused under the production rules (the e2e build accepts it, see `Rules::current`).
        let lo = fav("127.0.0.1:8463", "direct");
        let view: FavouritesView = Ok(favs(vec![lo]));
        let cat = Catalog {
            live: &live,
            favs: &view,
            rules: Rules::default(),
            live_servers_mtime: 0,
            window_model: &model_ok,
        };
        assert_eq!(
            decide(&req("127.0.0.1:8463"), &cat, Some(&st), &bundle_ok, 1000),
            Err(Refuse("bad_address"))
        );
        let cat = Catalog {
            rules: Rules { allow_loopback: true },
            ..cat
        };
        assert!(decide(&req("127.0.0.1:8463"), &cat, Some(&st), &bundle_ok, 1000).is_ok());
    }

    fn live_from(toml: &str) -> LiveServers {
        live(toml)
    }

    #[test]
    fn a_ban_closes_a_favourite_until_the_owner_reopens_it_and_no_edit_of_its_proxy_or_anything_else_does() {
        let live = LiveServers::default();
        let ban_at = 5000;
        let st = blocked_state(FAV_ADDR, ban_at);
        let mut f = fav(FAV_ADDR, "proxy:hp-1");
        let go = |f: &Favourite, mtime: u64, now: u64| {
            decide_fav(&req(FAV_ADDR), &live, &favs(vec![f.clone()]), Some(&st), mtime, now)
        };
        assert_eq!(go(&f, 0, 9000), Err(Refuse("blocked_after_ban")));
        // Whatever the owner edits afterwards (another proxy, direct, the nick) it stays closed ...
        for connection in ["proxy:hp-2", "direct", "proxy:hp-1"] {
            f.connection = connection.to_string();
            assert_eq!(go(&f, 0, 9000), Err(Refuse("blocked_after_ban")), "{connection}");
        }
        f.nick = "Other".to_string();
        assert_eq!(go(&f, 0, 9000), Err(Refuse("blocked_after_ban")));
        // ... and so does editing live-servers.toml (that re-opens allow-list entries, not favourites), even to a time after the ban.
        assert_eq!(go(&f, ban_at + 1_000_000, 9000), Err(Refuse("blocked_after_ban")));
        // Removing it and adding it again (a new entry, `reopened_at` 0) is not a re-opening.
        let fresh = fav(FAV_ADDR, "direct");
        assert_eq!(go(&fresh, 0, 9000), Err(Refuse("blocked_after_ban")));
        // A re-open at or before the ban time does not lift it; one after it does.
        f.reopened_at = ban_at;
        assert_eq!(go(&f, 0, 9000), Err(Refuse("blocked_after_ban")));
        f.reopened_at = ban_at - 1;
        assert_eq!(go(&f, 0, 9000), Err(Refuse("blocked_after_ban")));
        f.reopened_at = ban_at + 1;
        assert!(go(&f, 0, 9000).is_ok());
        // A newer ban closes it again (the earlier re-open is older than the new ban).
        let st2 = blocked_state(FAV_ADDR, ban_at + 500);
        assert_eq!(
            decide_fav(&req(FAV_ADDR), &live, &favs(vec![f.clone()]), Some(&st2), 0, 9000),
            Err(Refuse("blocked_after_ban"))
        );
    }

    #[test]
    fn a_ban_is_the_machines_not_one_ports_and_other_servers_are_unaffected() {
        let live = LiveServers::default();
        let st = blocked_state(FAV_ADDR, 5000);
        let both = favs(vec![
            fav(FAV_ADDR, "direct"),
            fav(FAV_SIBLING, "direct"),
            fav(FAV_OTHER, "direct"),
        ]);
        let go = |a: &str| decide_fav(&req(a), &live, &both, Some(&st), 0, 9000);
        assert_eq!(go(FAV_ADDR), Err(Refuse("blocked_after_ban")));
        assert_eq!(go(FAV_SIBLING), Err(Refuse("blocked_after_ban")), "the same IP");
        assert!(go(FAV_OTHER).is_ok(), "another IP");
        // The local server is never closed by anything.
        assert!(decide_fav(&req("local"), &live, &both, Some(&st), 0, 9000).is_ok());
        // The sibling's own re-open lifts the block for it (and only for the favourite that was re-opened).
        let mut reopened = fav(FAV_SIBLING, "direct");
        reopened.reopened_at = 6000;
        let list = favs(vec![fav(FAV_ADDR, "direct"), reopened]);
        assert!(decide_fav(&req(FAV_SIBLING), &live, &list, Some(&st), 0, 9000).is_ok());
        assert_eq!(
            decide_fav(&req(FAV_ADDR), &live, &list, Some(&st), 0, 9000),
            Err(Refuse("blocked_after_ban"))
        );
    }

    #[test]
    fn the_cool_down_and_the_start_interval_apply_to_favourites_too() {
        let live = LiveServers::default();
        let list = favs(vec![fav(FAV_ADDR, "direct")]);
        let st = State {
            last_exit: Some(ExitInfo { at: 1000, code: 4 }),
            ..State::default()
        };
        assert_eq!(
            decide_fav(&req(FAV_ADDR), &live, &list, Some(&st), 0, 1119),
            Err(Refuse("cooldown"))
        );
        assert!(decide_fav(&req(FAV_ADDR), &live, &list, Some(&st), 0, 1120).is_ok());
        let st = State {
            last_start_at: 1000,
            ..State::default()
        };
        assert_eq!(
            decide_fav(&req(FAV_ADDR), &live, &list, Some(&st), 0, 1029),
            Err(Refuse("rate_limited"))
        );
        assert_eq!(
            decide_fav(&req(FAV_ADDR), &live, &list, None, 0, 1029),
            Err(Refuse("state_unreadable"))
        );
    }

    #[test]
    fn a_lifted_ban_is_inert_but_still_closes_the_other_addresses_on_that_ip() {
        let live = LiveServers::default();
        let st = blocked_state(FAV_ADDR, 5000);
        let mut reopened = fav(FAV_ADDR, "direct");
        reopened.reopened_at = 6000;
        let list = favs(vec![reopened, fav(FAV_SIBLING, "direct")]);
        assert!(decide_fav(&req(FAV_ADDR), &live, &list, Some(&st), 0, 9000).is_ok());
        assert_eq!(
            decide_fav(&req(FAV_SIBLING), &live, &list, Some(&st), 0, 9000),
            Err(Refuse("blocked_after_ban")),
            "the sibling needs its own re-opening"
        );
    }

    #[test]
    fn a_favourites_proxy_is_the_one_the_owner_named_and_a_missing_one_is_a_refusal_never_a_fallback() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, extra: &str| {
            let path = dir.path().join(format!("{name}-proxy.toml"));
            std::fs::write(&path, format!("host = \"198.51.100.7\"\nport = 1080\n{extra}")).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        };
        write("hp-pub", "relay = \"public\"\n");
        write("hp-host", "");
        let plan_of = |f: Favourite| {
            let list = favs(vec![f.clone()]);
            let live = LiveServers::default();
            let plan = decide_fav(&req(&f.address), &live, &list, Some(&State::default()), 0, 1000).expect("plan");
            let merged = live.with_favourites(&list).unwrap();
            (plan, merged)
        };
        let server: IpAddr = "45.141.57.35".parse().unwrap();
        // A public relay: the server's IP is denied, nothing is allowed.
        let (plan, merged) = plan_of(fav(FAV_ADDR, "proxy:hp-pub"));
        assert_eq!(
            filter_for(&plan, &merged, dir.path()),
            Ok(Filter::DenyServer(vec![server]))
        );
        // A relay on the proxy's host: the proxy's IP is allowed.
        let (plan, merged) = plan_of(fav(FAV_ADDR, "proxy:hp-host"));
        assert_eq!(
            filter_for(&plan, &merged, dir.path()),
            Ok(Filter::Allow(vec!["198.51.100.7".parse().unwrap()]))
        );
        // Direct: the server's own IP.
        let (plan, merged) = plan_of(fav(FAV_ADDR, "direct"));
        assert_eq!(filter_for(&plan, &merged, dir.path()), Ok(Filter::Allow(vec![server])));
        // The named proxy is gone (or unsafe), while another one exists: a refusal. Not direct, not the other proxy.
        let (plan, merged) = plan_of(fav(FAV_ADDR, "proxy:gone"));
        assert_eq!(filter_for(&plan, &merged, dir.path()), Err(Refuse("proxy_error")));
        let path = dir.path().join("hp-host-proxy.toml");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let (plan, merged) = plan_of(fav(FAV_ADDR, "proxy:hp-host"));
        assert_eq!(filter_for(&plan, &merged, dir.path()), Err(Refuse("proxy_error")));
        std::fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(dir.path().join("hp-pub-proxy.toml"), &path).unwrap();
        assert_eq!(
            filter_for(&plan, &merged, dir.path()),
            Err(Refuse("proxy_error")),
            "a symlink is not followed"
        );
    }

    #[test]
    fn the_blocked_list_for_the_web_holds_addresses_times_and_codes_only() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths {
            data_dir: dir.path().to_path_buf(),
            live_servers: dir.path().join("l.toml"),
            config: dir.path().join("c.toml"),
            env_file: dir.path().join("e"),
            dropin: dir.path().join("d"),
            state: dir.path().join("s.json"),
            launch_dir: dir.path().join("launch"),
            status_dir: dir.path().join("status"),
        };
        let st = blocked_state(FAV_ADDR, 5000);
        publish_blocked(&paths, &st, 7000);
        let text = std::fs::read_to_string(paths.status_dir.join(BLOCKED_FILE)).unwrap();
        let file: BlockedFile = serde_json::from_str(&text).unwrap();
        assert_eq!(
            (
                file.v,
                file.at,
                file.blocked.len(),
                file.blocked[0].address.as_str(),
                file.blocked[0].at,
                file.blocked[0].code
            ),
            (1, 7000, 1, FAV_ADDR, 5000, 3)
        );
        assert_eq!(
            std::os::unix::fs::PermissionsExt::mode(
                &std::fs::metadata(paths.status_dir.join(BLOCKED_FILE))
                    .unwrap()
                    .permissions()
            ) & 0o777,
            0o644
        );
    }

    // --- check-proxy: the page is told codes and numbers, never a value ---------------------------------------------

    #[test]
    fn the_check_result_has_fixed_codes_and_no_address_or_credential() {
        use ddai_client::proxy::RelayMode;
        use ddai_client::socks5::{ProbeReport, SessionReport, Step};
        use std::time::Duration;
        let ok = check_result(
            "0123456789abcdef",
            "hp",
            9,
            &Ok(ProxyCheck {
                relay_host: RelayHost::Remote,
                relay_port: 4242,
                authenticated: true,
                mode: RelayMode::Public,
                probe: Some(ProbeReport {
                    sent: 5,
                    replies: 4,
                    median: Duration::from_millis(22),
                }),
                sessions: None,
            }),
        );
        assert!(ok.ok);
        assert_eq!(
            (
                ok.code.as_str(),
                ok.relay.as_deref(),
                ok.relay_mode.as_deref(),
                ok.udp_rtt_ms,
                ok.probe_sent,
                ok.probe_replies
            ),
            ("ok", Some("remote"), Some("public"), Some(22), Some(5), Some(4))
        );
        let picked = check_result(
            "0123456789abcdef",
            "hp",
            9,
            &Ok(ProxyCheck {
                relay_host: RelayHost::SameAsProxy,
                relay_port: 1,
                authenticated: false,
                mode: RelayMode::ProxyHostOnly,
                probe: None,
                sessions: Some(SessionReport {
                    rtts: vec![None, Some(Duration::from_millis(40))],
                    picked: 1,
                }),
            }),
        );
        assert_eq!(
            (picked.relay.as_deref(), picked.udp_rtt_ms),
            (Some("same_host"), Some(40))
        );
        for (e, code) in [
            (Socks5Error::UdpNotSupported, "udp_not_supported"),
            (Socks5Error::AuthFailed, "auth_failed"),
            (Socks5Error::NoAcceptableMethod, "no_auth_method"),
            (Socks5Error::UnsupportedMethod(9), "no_auth_method"),
            (Socks5Error::Timeout { step: Step::Connect }, "timeout"),
            (Socks5Error::Closed { step: Step::Connect }, "connect_failed"),
            (Socks5Error::ControlClosed, "connect_failed"),
            (
                Socks5Error::Refused {
                    code: 5,
                    meaning: "refused",
                },
                "refused",
            ),
            (Socks5Error::Protocol("x"), "protocol_error"),
            (Socks5Error::RelayAddress("private 10.1.2.3"), "relay_refused"),
            (Socks5Error::ProbeFailed, "probe_failed"),
        ] {
            let r = check_result("0123456789abcdef", "hp", 9, &Err(e));
            assert_eq!((r.ok, r.code.as_str()), (false, code));
            let text = serde_json::to_string(&r).unwrap();
            assert!(!text.contains("10.1.2.3"), "{text}");
        }
    }

    // --- review 5.12 round 1 ---------------------------------------------------------------------------------------

    #[test]
    fn a_reopening_dated_in_the_future_is_refused_by_the_helper() {
        let live = LiveServers::default();
        let st = blocked_state(FAV_ADDR, 5000);
        for future in [99_999_999_999u64, u64::MAX] {
            let mut f = fav(FAV_ADDR, "direct");
            f.reopened_at = future;
            assert_eq!(
                decide_fav(&req(FAV_ADDR), &live, &favs(vec![f]), Some(&st), 0, 9000),
                Err(Refuse("reopened_invalid")),
                "{future}"
            );
        }
        // A file that holds one is refused as a whole when it is read.
        let mut f = fav(FAV_ADDR, "direct");
        f.reopened_at = 99_999_999_999;
        let bytes = serde_json::to_vec(&favs(vec![f])).unwrap();
        assert!(Favourites::parse(&bytes, Rules::default()).is_err());
    }

    #[test]
    fn a_host_name_in_the_allow_list_closes_the_favourites_door() {
        let named = live(
            "[[server]]\naddress = \"one.one.one.one:8303\"\nnick = \"Muha\"\nready = true\nproxy = \"swarfey\"\n",
        );
        let list = favs(vec![fav("1.1.1.1:8303", "direct")]);
        assert_eq!(
            decide_fav(&req("1.1.1.1:8303"), &named, &list, Some(&State::default()), 0, 9000),
            Err(Refuse("allowlist_not_literal"))
        );
        // Literal entries only: favourites work.
        let literal = live("[[server]]\naddress = \"203.0.113.5:8308\"\nnick = \"Muha\"\nready = true\n");
        assert!(decide_fav(&req("1.1.1.1:8303"), &literal, &list, Some(&State::default()), 0, 9000).is_ok());
    }

    #[test]
    fn an_allow_list_edit_never_lifts_a_ban_recorded_for_a_favourite() {
        // The favourite 45.141.57.35:8308 was banned; the allow-list names a sibling port on the same IP.
        let sibling = live(&format!(
            "[[server]]\naddress = \"{FAV_SIBLING}\"\nnick = \"Muha\"\nready = true\n"
        ));
        let mut st = State::default();
        st.blocked.insert(
            FAV_ADDR.to_string(),
            Block {
                at: 5000,
                code: 3,
                favourite: true,
            },
        );
        // Any later edit of live-servers.toml: still closed.
        assert_eq!(
            decide_with(&req(FAV_SIBLING), &sibling, Some(&st), 1_000_000, 9000),
            Err(Refuse("blocked_after_ban"))
        );
        // The favourite's own re-opening lifts it for the favourite, and an allow-list-origin block is still lifted by the edit.
        let mut reopened = fav(FAV_ADDR, "direct");
        reopened.reopened_at = 6000;
        assert!(
            decide_fav(
                &req(FAV_ADDR),
                &LiveServers::default(),
                &favs(vec![reopened]),
                Some(&st),
                0,
                9000
            )
            .is_ok()
        );
        let st2 = blocked_state(FAV_ADDR, 5000); // origin: allow-list
        assert!(decide_with(&req(FAV_SIBLING), &sibling, Some(&st2), 1_000_000, 9000).is_ok());
        assert_eq!(
            decide_with(&req(FAV_SIBLING), &sibling, Some(&st2), 100, 9000),
            Err(Refuse("blocked_after_ban"))
        );
    }

    #[test]
    fn a_site_made_profile_host_must_be_public_and_loopback_only_in_a_test_build() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, body: &str| {
            let path = dir.path().join(format!("{name}-proxy.toml"));
            std::fs::write(&path, body).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        };
        let prod = Rules::default();
        let test = Rules { allow_loopback: true };
        write(
            "site",
            "managed_by = \"ddnet-ai-web\"\nhost = \"127.0.0.1\"\nport = 1\n",
        );
        assert!(managed_host_refused(dir.path(), "site", prod));
        assert!(!managed_host_refused(dir.path(), "site", test));
        write(
            "pub",
            "managed_by = \"ddnet-ai-web\"\nhost = \"93.184.216.34\"\nport = 1\n",
        );
        assert!(!managed_host_refused(dir.path(), "pub", prod));
        write("v6", "managed_by = \"ddnet-ai-web\"\nhost = \"::1\"\nport = 1\n");
        assert!(managed_host_refused(dir.path(), "v6", prod));
        // A hand-made profile is the owner's own, whatever its host.
        write("hand", "host = \"10.0.0.1\"\nport = 1\n");
        assert!(!managed_host_refused(dir.path(), "hand", prod));
        assert!(!managed_host_refused(dir.path(), "missing", prod));
    }
}
