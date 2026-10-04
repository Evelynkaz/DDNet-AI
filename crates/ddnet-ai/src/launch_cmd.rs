//! `ddnet-ai launch apply|exited` (task 5.9, D-089): the **root-side helper of the web launcher**. The web only writes a small
//! request file; a root systemd path unit runs `launch apply`, which is the only code that turns a request into `systemctl`
//! calls. Everything is decided here, against fixed allow-lists, and nothing from the request is ever passed to a command line:
//!
//! - the request is read as a regular file (no symlink), size-limited, parsed with a strict schema, and **deleted before it is
//!   processed** (a request is never replayed, e.g. after a reboot);
//! - `server` is `"local"` or the exact `address` of a `ready = true` entry of `live-servers.toml`; the nick, the proxy and
//!   the address that reach the unit come from that entry, never from the request;
//! - the root-owned environment file for the bot unit holds only validated values from a closed character set;
//! - the unit's cgroup filter (a generated drop-in) allows loopback, plus the proxy's IPs (read from the secrets file, never
//!   printed) or the entry's server IP for a public server; for a proxy with `relay = "public"` (its UDP relay lives on
//!   another host, task 2.6b, D-090) it is the opposite: both lists are reset and **every IP of the game server is denied**,
//!   with no allow list, so the relay is reachable anywhere and the server never directly;
//! - at most one start per [`START_INTERVAL_SECS`]; a cool-down after exit 3/4; after a kick/ban on a **public** server that
//!   server is refused until the owner has edited `live-servers.toml` again (re-opened it).
//!
//! `launch exited` is the bot unit's `ExecStopPost=` hook: it records how the bot ended (the ban memory above) and tells the web.
//! Schemas: `docs/formats.md` §34. Never connects to any game server and never writes chat.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::net::{IpAddr, SocketAddr};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use clap::{Args, Subcommand};
use ddai_client::live_servers::{LiveServers, is_loopback};
use ddai_web::launch::{
    Action, Brain, DurationChoice, LOCAL_SERVER, LaunchConfig, LaunchRequest, LaunchStatus, MAX_REQUEST_BYTES,
    MAX_SPARRING, Mirror, REQUEST_FILE, ReadError, START_INTERVAL_SECS, STATUS_FILE, State as RunState,
    bundle_run_name, parse_request, read_regular_nofollow, read_regular_nofollow_with_mtime, request_is_fresh,
    unix_now, write_atomic,
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
const MAX_NICK_LEN: usize = 15;

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
    bundle: Option<String>,
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
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
struct State {
    last_start_at: u64,
    last_launch: Option<LaunchInfo>,
    stop_requested_at: u64,
    last_exit: Option<ExitInfo>,
    /// Public servers (by allow-list address) the bot was kicked or banned from, until the owner re-opens the entry.
    blocked: BTreeMap<String, Block>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct Target {
    /// `local`, or the entry's `address` (the state's and the status's name for it).
    key: String,
    addr: SocketAddr,
    nick: String,
    public: bool,
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
}

fn valid_nick(nick: &str) -> bool {
    !nick.is_empty()
        && nick.len() <= MAX_NICK_LEN
        && nick
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// `"local"`, or the one `ready = true` entry whose `address` is exactly `selector`.
fn resolve_server(selector: &str, live: &LiveServers) -> Result<Target, Refuse> {
    if selector == LOCAL_SERVER {
        return Ok(Target {
            key: LOCAL_SERVER.to_string(),
            addr: LOCAL_ADDR.parse().map_err(|_| Refuse("internal"))?,
            nick: LOCAL_NAME.to_string(),
            public: false,
        });
    }
    let listed: Vec<_> = live.servers.iter().filter(|e| e.address == selector).collect();
    if listed.is_empty() {
        return Err(Refuse("server_not_allowed"));
    }
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
    Ok(Target {
        key: entry.address.clone(),
        addr,
        nick: entry.nick.clone(),
        public: true,
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

/// Judges a start request against the allow-lists and the rate rules. `state` is `None` when the state file cannot be trusted
/// (everything is then refused, the local server too: the file may hold a ban).
/// `live_servers_mtime` (unix seconds) is when the owner last edited the allow-list: a ban stays until the file is newer.
fn decide(
    req: &LaunchRequest,
    live: &LiveServers,
    live_servers_mtime: u64,
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
    let target = resolve_server(selector, live)?;
    if sparring > 0 && target.public {
        return Err(Refuse("sparring_local_only"));
    }
    let bundle = if brain.needs_bundle() { Some(bundle()?) } else { None };

    // Policy: the ban memory first, then the rates.
    match state {
        None => return Err(Refuse("state_unreadable")),
        Some(st) => {
            if let Some(block) = st.blocked.get(&target.key)
                && live_servers_mtime <= block.at
            {
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
    status
}

// ---------------------------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------------------------

pub fn run(args: LaunchArgs) -> ExitCode {
    match args.command {
        LaunchCommand::Apply(a) => apply(&a),
        LaunchCommand::Exited(p) => exited(&p.resolve()),
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
    let state = load_state(&paths.state).ok();
    let bundle = || configured_bundle(&paths.config, &paths.data_dir);
    let plan = match decide(
        req,
        &live,
        mtime_secs(&paths.live_servers),
        state.as_ref(),
        &bundle,
        now,
    ) {
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
    let filter = match filter_for(&plan, &live, &secrets_dir) {
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
        bundle: plan.bundle.as_deref().map(bundle_run_name),
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
        decide(req, &live, mtime_secs(&paths.live_servers), Some(st), &bundle, now)?;
        st.last_start_at = now;
        st.last_launch = Some(info.clone());
        st.stop_requested_at = 0;
        // Only a ban that `decide` has just found re-opened (the allow-list was edited after it) is still here to remove.
        st.blocked.remove(&plan.target.key);
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
        "launch: started (brain {:?}, local {}, sparring {})",
        plan.brain, !plan.target.public, plan.sparring
    );
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
            // A kick/ban on a public server stays in force until the owner edits the allow-list again.
            if let Some(info) = &launch
                && info.public
            {
                st.blocked.insert(info.server.clone(), Block { at: now, code: c });
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
        }
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
        decide(r, l, mtime, st, &bundle_ok, now)
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
        assert_eq!(
            decide(&r, &LiveServers::default(), 0, Some(&State::default()), &missing, 1000),
            Err(Refuse("bundle_missing"))
        );
        r.brain = Some(Brain::Hybrid);
        assert!(decide(&r, &LiveServers::default(), 0, Some(&State::default()), &missing, 1000).is_ok());
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
        st.blocked
            .insert("203.0.113.5:8308".to_string(), Block { at: 5000, code: 3 });
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
            st.blocked.insert("a:1".into(), Block { at: 9, code: 3 });
        })
        .unwrap();
        assert_eq!(load_state(&path).unwrap().blocked["a:1"], Block { at: 9, code: 3 });
        std::fs::write(&path, b"{ not json").unwrap();
        assert!(load_state(&path).is_err());
        assert!(with_state(&path, |_| ()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"{ not json");
    }
}
