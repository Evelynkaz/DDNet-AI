//! Server configuration and the non-loopback bind refusal (acceptance criterion 1: "refuse
//! non-loopback binds unless an explicit `--i-know-this-is-public` flag").

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use crate::auth::rate_limit::RateLimitConfig;

/// Maximum concurrent WebSocket connections per session (acceptance criterion 5).
pub const DEFAULT_MAX_WS_PER_SESSION: u32 = 4;

/// Maximum request body size for API routes, in bytes (acceptance criterion 6: "16 KiB").
pub const DEFAULT_MAX_BODY_BYTES: usize = 16 * 1024;

/// Maximum WebSocket message size, in bytes. This channel only carries small typed JSON control
/// messages for now (binary game-view frames are a later task per the goal), so this is
/// deliberately tight.
pub const DEFAULT_MAX_WS_MESSAGE_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone)]
pub struct WebConfig {
    pub listen: SocketAddr,
    pub data_dir: PathBuf,
    /// Trust `X-Forwarded-For` when the TCP peer is loopback (acceptance criterion 3).
    pub trust_proxy: bool,
    /// Bypasses the non-loopback bind refusal. Never set this in normal operation.
    pub i_know_this_is_public: bool,
    /// Whether cookies get the `Secure` attribute (and the `__Host-` name prefix). Must be false
    /// for plain-HTTP local testing (browsers refuse `Secure` cookies over plain HTTP) and should
    /// be true once Caddy terminates HTTPS in front of this server (task 5.3).
    pub cookie_secure: bool,
    pub idle_timeout: Duration,
    pub absolute_timeout: Duration,
    pub login_rate_limit: RateLimitConfig,
    pub max_ws_per_session: u32,
    pub max_body_bytes: usize,
    pub max_ws_message_bytes: usize,
    pub request_timeout: Duration,
    /// How long a trusted-device cookie stays valid (review finding F7). Also capped in practice
    /// by a password change, which invalidates every existing device's stored fingerprint
    /// regardless of this TTL — see `auth::device`.
    pub trusted_device_ttl: Duration,
    /// Task 5.2a, `ddnet-ai web --replay <dir-or-file>`: a directory of Oracle B `trace-b` files
    /// to loop through, or a single such file. `None` means the live map view has no active
    /// source — the WS still works, it just never sends `map`/`live` messages.
    pub replay_source: Option<PathBuf>,
    /// Task 4.1: the live bot's Unix socket (`ddai_bot::bridge`, `docs/formats.md` §21) as the live
    /// view's data source instead of a replay. Mutually exclusive with `replay_source`.
    pub bot_socket: Option<PathBuf>,
    /// Capacity of the hub's event channel (map, roster, source, status ...), in messages. A browser connection that falls further
    /// behind loses events and is told the state again. The default is generous; tests make it tiny to force that.
    pub event_broadcast_capacity: usize,
    /// Task 5.7: the offline demo's bridge socket (`ddnet-ai fly watch --bridge`, `docs/formats.md` §28): shown while the
    /// live bot (`bot_socket`) is not there, and given up the moment it is. Needs `bot_socket`, and must be another socket.
    pub demo_socket: Option<PathBuf>,
    /// Task 5.2a: directories `crate::live::map_resolve` may read a real `.map` file's bytes
    /// from, by filename, when resolving a real-map replay trace's map (never any path taken
    /// from request or trace content directly — see that module's doc comment). Typically
    /// `~/aiddnet/data/ddnet-server/maps` and/or `~/aiddnet/data/maps/<subdir>`.
    pub map_search_dirs: Vec<PathBuf>,
    /// Live WS frame rate cap, in Hz (acceptance criterion 1: "≤ 50 Hz; default 25 Hz, 10 Hz
    /// эконом"). A client's `sub{live: hz}` request is clamped to this.
    pub max_live_hz: f32,
    /// Task 7.4: the fly panel's frame rate cap, in Hz (the bot builds a frame every second decision, ~12.5 Hz; a phone
    /// asks for fewer). A client's `{"type":"fly","hz":N}` is clamped to this.
    pub max_fly_hz: f32,
    /// Task 5.6: the bot's control socket (`ddai_bot::control`, `docs/formats.md` §26) the owner's commands go to.
    /// Default `<data-dir>/bot/control.sock`. The web only ever *connects* to it; it never creates it.
    pub control_socket: PathBuf,
    /// Task 5.6: the friend / war / ignore lists file the editor reads and writes. Default
    /// `<data-dir>/bot/relations.json`.
    pub relations_path: PathBuf,
    /// Task 5.8: the training runs the «Обучение» tab reads (read-only), default `<data-dir>/runs`.
    pub runs_dir: PathBuf,
    /// Task 5.10: the DDNet data directory (`data/` of a DDNet 20.1 install or build) the game view's graphics are served
    /// from at `/assets/...` (`http::ddnet_assets`), read-only. `None`: the page draws without them (flat tees, no external
    /// map images). Never copied into the repository.
    pub ddnet_data_dir: Option<PathBuf>,
    /// Task 5.9: the launcher's directory (D-089): the web writes `request.json` here (the status is in
    /// `status_dir`); the root helper (`ddnet-ai launch apply`) consumes it. Default `<data-dir>/launch`.
    pub launch_dir: PathBuf,
    /// Task 5.9 (review F5): the root-owned directory the helper writes `status.json` into; the web only reads it. Default
    /// `/run/ddnet-ai`.
    pub status_dir: PathBuf,
    /// Task 5.9 (review F3): the least time between two accepted launcher requests, and the most in any 60 seconds. The path unit
    /// stops after `TriggerLimitBurst=10` in a minute, so the web stays well below it.
    pub launch_min_gap: Duration,
    pub launch_max_per_minute: usize,
    /// Task 5.9: the owner's allow-list the launcher's server choice is built from (read-only). Default
    /// `<data-dir>/live-servers.toml`.
    pub live_servers: PathBuf,
    /// Task 5.9: the root-owned launcher config (the fly bundle path), read to show which bundle is active.
    pub launch_config: PathBuf,
    /// Task 5.12 (D-099): where the master-list cache (`master.json`, `refresh.json`) is read from, read-only. Default
    /// `<data-dir>/servers`. Written by `ddnet-ai servers-cache` in its own unit; the web has no network.
    pub servers_dir: PathBuf,
    /// Task 5.12: the rules the favourites and proxy hosts are held to. Production: [`ddai_client::favourites::Rules::current`]
    /// (no loopback). Tests that run a private server on 127.0.0.1 are built with the `loopback-favourites` feature.
    pub favourite_rules: ddai_client::favourites::Rules,
    /// Task 5.12: the least time between two «обновить список» requests (the fetch itself also does nothing within
    /// [`ddai_client::server_list::MIN_REFRESH_SECS`] of the last one).
    pub refresh_min_gap: Duration,
    /// Task 5.12: the least time between two «Проверить» requests, and the most in any 60 seconds.
    pub proxycheck_min_gap: Duration,
    pub proxycheck_max_per_minute: usize,
    /// Task 5.12: the most favourite / proxy changes (every mutating route of the «Серверы» tab but launch, refresh and check) in any
    /// 60 seconds. Every attempt counts, valid or not.
    pub servers_edits_per_minute: usize,
    /// Task 4.9 (D-090): the web's own rate limit for the owner's chat lines (`POST /api/bot/say`), on top of the bot's (3 s apart,
    /// 10 a minute, a queue of 3): at most `say_burst` requests in any `say_burst_window`, and at most `say_max_per_minute` in any 60
    /// seconds. A burst of two covers a quick correction; a third in the same moment is refused here, before it reaches the bot.
    pub say_burst: usize,
    pub say_burst_window: Duration,
    pub say_max_per_minute: usize,
    /// Task 5.16 (D-120): where `GET /api/bot/status` reads the host's load average from (`/proc/loadavg`; tests point it at a file). Read-only; nothing
    /// is kept or sent anywhere.
    pub loadavg_path: PathBuf,
}

impl WebConfig {
    /// Sensible production defaults (acceptance criteria 2, 3, 5, 6): 12h idle / 7d absolute
    /// session timeout, the default login rate limits, 4 concurrent WS per session, 16 KiB body
    /// limit. `cookie_secure` defaults to `false` because this binary, run standalone, only ever
    /// speaks plain HTTP on loopback; task 5.3 (Caddy) is what makes `true` correct, and will set
    /// it explicitly.
    pub fn new(listen: SocketAddr, data_dir: PathBuf) -> Self {
        Self {
            listen,
            trust_proxy: false,
            i_know_this_is_public: false,
            cookie_secure: false,
            idle_timeout: Duration::from_secs(12 * 3600),
            absolute_timeout: Duration::from_secs(7 * 24 * 3600),
            login_rate_limit: RateLimitConfig::default(),
            max_ws_per_session: DEFAULT_MAX_WS_PER_SESSION,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_ws_message_bytes: DEFAULT_MAX_WS_MESSAGE_BYTES,
            request_timeout: Duration::from_secs(10),
            trusted_device_ttl: Duration::from_secs(90 * 24 * 3600),
            replay_source: None,
            bot_socket: None,
            demo_socket: None,
            event_broadcast_capacity: crate::live::hub::DEFAULT_EVENT_BROADCAST_CAPACITY,
            map_search_dirs: Vec::new(),
            max_live_hz: 50.0,
            max_fly_hz: 15.0,
            control_socket: data_dir.join("bot").join("control.sock"),
            relations_path: data_dir.join("bot").join("relations.json"),
            runs_dir: data_dir.join("runs"),
            ddnet_data_dir: None,
            launch_dir: data_dir.join("launch"),
            status_dir: PathBuf::from(crate::launch::DEFAULT_STATUS_DIR),
            launch_min_gap: Duration::from_secs(2),
            launch_max_per_minute: 6,
            live_servers: data_dir.join("live-servers.toml"),
            launch_config: PathBuf::from(crate::launch::DEFAULT_CONFIG_PATH),
            servers_dir: data_dir.join("servers"),
            favourite_rules: ddai_client::favourites::Rules::current(),
            refresh_min_gap: Duration::from_secs(60),
            proxycheck_min_gap: Duration::from_secs(8),
            proxycheck_max_per_minute: 6,
            servers_edits_per_minute: 30,
            say_burst: 2,
            say_burst_window: Duration::from_secs(3),
            say_max_per_minute: 10,
            loadavg_path: PathBuf::from(crate::http::bot::DEFAULT_LOADAVG_PATH),
            data_dir,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    #[error(
        "a demo socket needs the live bot's socket to stand in for (--demo-socket goes with --bot-socket, not --replay)"
    )]
    DemoNeedsBot,
    #[error("the demo socket and the live bot's socket are the same path: the demo must never be the live socket")]
    DemoIsLive,
    #[error(
        "refusing to bind to non-loopback address {addr}: pass --i-know-this-is-public to override \
         (and don't — this server has no TLS of its own; Caddy is meant to be the only public-facing hop)"
    )]
    NonLoopbackBind { addr: SocketAddr },
}

/// Refuses to bind anywhere but loopback unless `i_know_this_is_public` is set (acceptance
/// criterion 1). Checked before any socket is actually opened.
pub fn validate_listen_addr(listen: SocketAddr, i_know_this_is_public: bool) -> Result<(), ConfigError> {
    if listen.ip().is_loopback() || i_know_this_is_public {
        Ok(())
    } else {
        Err(ConfigError::NonLoopbackBind { addr: listen })
    }
}

/// Checks how the sources fit together (task 5.7): a demo stands in for a live bot, on a socket of its own.
pub fn validate_sources(config: &WebConfig) -> Result<(), ConfigError> {
    let Some(demo) = &config.demo_socket else {
        return Ok(());
    };
    match &config.bot_socket {
        None => Err(ConfigError::DemoNeedsBot),
        Some(live) if live == demo => Err(ConfigError::DemoIsLive),
        Some(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    fn addr(ip: std::net::IpAddr, port: u16) -> SocketAddr {
        SocketAddr::new(ip, port)
    }

    #[test]
    fn loopback_v4_is_allowed_by_default() {
        assert!(validate_listen_addr(addr(Ipv4Addr::LOCALHOST.into(), 7788), false).is_ok());
    }

    #[test]
    fn loopback_v6_is_allowed_by_default() {
        assert!(validate_listen_addr(addr(Ipv6Addr::LOCALHOST.into(), 7788), false).is_ok());
    }

    #[test]
    fn unspecified_v4_is_refused_by_default() {
        let a = addr(Ipv4Addr::UNSPECIFIED.into(), 7788);
        assert_eq!(
            validate_listen_addr(a, false),
            Err(ConfigError::NonLoopbackBind { addr: a })
        );
    }

    #[test]
    fn unspecified_v6_is_refused_by_default() {
        let a = addr(Ipv6Addr::UNSPECIFIED.into(), 7788);
        assert!(validate_listen_addr(a, false).is_err());
    }

    #[test]
    fn public_v4_is_refused_by_default() {
        let a = addr(Ipv4Addr::new(203, 0, 113, 5).into(), 7788);
        assert!(validate_listen_addr(a, false).is_err());
    }

    #[test]
    fn non_loopback_is_allowed_with_explicit_override() {
        let a = addr(Ipv4Addr::new(203, 0, 113, 5).into(), 7788);
        assert!(validate_listen_addr(a, true).is_ok());
    }

    #[test]
    fn config_new_has_secure_defaults() {
        let cfg = WebConfig::new(addr(Ipv4Addr::LOCALHOST.into(), 7788), PathBuf::from("/tmp/data"));
        assert!(!cfg.cookie_secure);
        assert!(!cfg.trust_proxy);
        assert!(!cfg.i_know_this_is_public);
        assert_eq!(cfg.max_ws_per_session, DEFAULT_MAX_WS_PER_SESSION);
        assert_eq!(cfg.max_body_bytes, DEFAULT_MAX_BODY_BYTES);
    }

    #[test]
    fn a_demo_socket_needs_a_different_live_socket() {
        let mut cfg = WebConfig::new(addr(Ipv4Addr::LOCALHOST.into(), 7788), PathBuf::from("/tmp/data"));
        assert_eq!(validate_sources(&cfg), Ok(()), "no demo is fine");
        cfg.demo_socket = Some(PathBuf::from("/tmp/demo.sock"));
        assert_eq!(validate_sources(&cfg), Err(ConfigError::DemoNeedsBot));
        cfg.bot_socket = Some(PathBuf::from("/tmp/demo.sock"));
        assert_eq!(validate_sources(&cfg), Err(ConfigError::DemoIsLive));
        cfg.bot_socket = Some(PathBuf::from("/tmp/live.sock"));
        assert_eq!(validate_sources(&cfg), Ok(()));
        cfg.replay_source = Some(PathBuf::from("/tmp/replays"));
        cfg.bot_socket = None;
        assert_eq!(validate_sources(&cfg), Err(ConfigError::DemoNeedsBot));
    }
}
