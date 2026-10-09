//! SOCKS5 proxy configuration and secrets (task 2.6, D-053 amendment): the redacting [`Secret`] newtype,
//! [`ProxyConfig`], and loading a proxy from `<secrets-dir>/<name>-proxy.toml`.
//!
//! The rules this module enforces:
//!
//! - A proxy is **named**: a server entry in `live-servers.toml` says `proxy = "<name>"` and only then is
//!   `<name>-proxy.toml` read ([`resolve_for_server`]). There is no other way to get a [`ProxyConfig`] into
//!   the driver for a server (the driver re-checks the binding on every connection attempt, see
//!   `crate::driver`).
//! - The file must be a regular file with mode `0600` (no group/other bits) on Unix; anything else is refused.
//! - Task 2.6b: `relay = "proxy-host-only" | "public"` (default the first) says whether a UDP relay the proxy announces
//!   on **another host** may be used ([`RelayMode`], the rule is in `crate::relay_rule`), and `session_pick = 2..=4`
//!   with a `{session}` placeholder in `user` makes the client try that many proxy sessions and keep the one with the
//!   lowest UDP round-trip time.
//! - Nothing from the file ever reaches a log line, a panic, a `Debug`/`Display` output or an error message:
//!   host, port, user and password are all held as [`Secret`]s, [`ProxyConfig`]'s `Debug` prints only the
//!   name, and every error here is built from key names and line numbers, never from values (a TOML syntax
//!   error's own text quotes the offending line, so it is **not** forwarded).

use crate::live_servers::{LiveServers, ProxyBindingError};
use std::fmt;
use std::io::Read;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

/// A string that never prints: `Debug` and `Display` both show `<redacted>`. The only way to see the
/// content is [`Secret::expose`], which the SOCKS5 handshake calls when it writes the bytes to the proxy.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    pub fn new(value: impl Into<String>) -> Self {
        Secret(value.into())
    }

    /// The raw content. Call it only where the value must go on the wire.
    pub fn expose(&self) -> &str {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

/// RFC 1929 credentials.
#[derive(Clone, PartialEq, Eq)]
pub struct ProxyAuth {
    pub user: Secret,
    pub pass: Secret,
}

impl fmt::Debug for ProxyAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ProxyAuth(<redacted>)")
    }
}

/// Where the proxy's UDP relay may be (the `relay` key of the proxy file, task 2.6b).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RelayMode {
    /// The default: datagrams go to the proxy's own IP; an announced relay on another host is replaced by the proxy's IP
    /// (the announced port is kept).
    #[default]
    ProxyHostOnly,
    /// An announced relay on another host is used when it is a public unicast address that is none of the game server's
    /// IPs and whose port is neither 0 nor the server's (`crate::relay_rule`). The unit's cgroup filter then denies the
    /// game server's IPs and nothing else (`deploy/README.md`).
    Public,
}

impl RelayMode {
    /// The value in the proxy file.
    pub fn as_str(self) -> &'static str {
        match self {
            RelayMode::ProxyHostOnly => "proxy-host-only",
            RelayMode::Public => "public",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "proxy-host-only" => Some(RelayMode::ProxyHostOnly),
            "public" => Some(RelayMode::Public),
            _ => None,
        }
    }
}

/// The placeholder in `user` that `session_pick` fills with a fresh session token.
pub const SESSION_PLACEHOLDER: &str = "{session}";
/// The most sessions `session_pick` may try (each is one proxy TCP connection).
pub const MAX_SESSION_PICK: u8 = 4;

/// A SOCKS5 proxy the driver may tunnel the game's UDP through. The address is as secret as the
/// credentials (it is the owner's, D-053), so it too is a [`Secret`] and only the `name` is ever shown.
#[derive(Clone, PartialEq, Eq)]
pub struct ProxyConfig {
    name: String,
    host: Secret,
    port: u16,
    auth: Option<ProxyAuth>,
    /// The file's own `for_server` (`host:port`): the server this proxy was first issued for (D-053). Since task 5.12 it binds
    /// nothing (the owner assigns proxies to servers on the site); its addresses are only kept away from the relay and, for
    /// `relay = "public"`, denied by the unit's cgroup filter. `None` when the file has none.
    for_server: Option<String>,
    relay: RelayMode,
    /// 0 or 1: off. 2..=4: how many sessions to try (needs [`SESSION_PLACEHOLDER`] in the user name).
    session_pick: u8,
    /// Test hook (feature `test-util` only, never set by the file parser): treat a loopback relay address as public, so
    /// tests can put the relay on another loopback alias.
    #[cfg(feature = "test-util")]
    test_loopback_relay: bool,
}

impl fmt::Debug for ProxyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProxyConfig")
            .field("name", &self.name)
            .field("endpoint", &"<redacted>")
            .field("auth", &self.auth.as_ref().map(|_| "<redacted>"))
            .field("for_server", &self.for_server.as_ref().map(|_| "<set>"))
            .field("relay", &self.relay)
            .field("session_pick", &self.session_pick)
            .finish()
    }
}

impl ProxyConfig {
    /// Builds a config by hand (tests, the e2e harness). `name` must satisfy [`valid_proxy_name`].
    pub fn new(
        name: &str,
        host: impl Into<String>,
        port: u16,
        auth: Option<(String, String)>,
    ) -> Result<Self, ProxyLoadError> {
        if !valid_proxy_name(name) {
            return Err(ProxyLoadError::InvalidName);
        }
        let host = host.into();
        if host.is_empty() {
            return Err(ProxyLoadError::Field("`host` is empty"));
        }
        if port == 0 {
            return Err(ProxyLoadError::Field("`port` must be 1..=65535"));
        }
        let auth = match auth {
            None => None,
            Some((user, pass)) => {
                if user.is_empty() || user.len() > 255 {
                    return Err(ProxyLoadError::Field("`user` must be 1..=255 bytes"));
                }
                if pass.is_empty() || pass.len() > 255 {
                    return Err(ProxyLoadError::Field("`pass` must be 1..=255 bytes"));
                }
                Some(ProxyAuth {
                    user: Secret::new(user),
                    pass: Secret::new(pass),
                })
            }
        };
        Ok(ProxyConfig {
            name: name.to_string(),
            host: Secret::new(host),
            port,
            auth,
            for_server: None,
            relay: RelayMode::default(),
            session_pick: 0,
            #[cfg(feature = "test-util")]
            test_loopback_relay: false,
        })
    }

    /// Sets the relay mode (the `relay` key of the file).
    pub fn with_relay(mut self, relay: RelayMode) -> Self {
        self.relay = relay;
        self
    }

    /// Sets how many sessions to try (the `session_pick` key). `0` and `1` mean off; `2..=MAX_SESSION_PICK` need the
    /// user name to contain [`SESSION_PLACEHOLDER`].
    pub fn with_session_pick(mut self, n: u8) -> Result<Self, ProxyLoadError> {
        if n > MAX_SESSION_PICK {
            return Err(ProxyLoadError::Field("`session_pick` must be 0..=4"));
        }
        if n >= 2
            && !self
                .auth
                .as_ref()
                .is_some_and(|a| a.user.expose().contains(SESSION_PLACEHOLDER))
        {
            return Err(ProxyLoadError::Field("`session_pick` needs `{session}` in `user`"));
        }
        self.session_pick = n;
        Ok(self)
    }

    /// Test hook: a loopback relay address counts as public (so the relay can sit on another loopback alias). Compiled
    /// only with the `test-util` feature, which the production binary never enables, and never set from a file.
    #[cfg(feature = "test-util")]
    pub fn with_test_loopback_relay(mut self) -> Self {
        self.test_loopback_relay = true;
        self
    }

    pub fn relay_mode(&self) -> RelayMode {
        self.relay
    }

    /// How many sessions to try at the start (`0` when picking is off).
    pub fn session_pick(&self) -> u8 {
        if self.session_pick >= 2 { self.session_pick } else { 0 }
    }

    /// Whether a loopback relay address may be treated as public (always `false` outside tests).
    pub(crate) fn loopback_relay_allowed(&self) -> bool {
        #[cfg(feature = "test-util")]
        {
            self.test_loopback_relay
        }
        #[cfg(not(feature = "test-util"))]
        {
            false
        }
    }

    /// The user name to send: `user` with every [`SESSION_PLACEHOLDER`] replaced by `session` (a fresh random token
    /// when `None`, so a literal `{session}` never goes on the wire).
    pub(crate) fn user_for(&self, session: Option<&str>) -> Option<String> {
        let user = self.auth.as_ref()?.user.expose();
        if !user.contains(SESSION_PLACEHOLDER) {
            return Some(user.to_string());
        }
        let fresh;
        let token = match session {
            Some(t) => t,
            None => {
                fresh = fresh_session_token(&[]);
                &fresh
            }
        };
        Some(user.replace(SESSION_PLACEHOLDER, token))
    }

    /// The IPs `for_server` resolves to, canonical (empty when the file has none or it does not resolve): the game server's
    /// other addresses, which a relay must not be either. The launcher (task 5.9) also puts them into the unit's
    /// `IPAddressDeny=` list for `relay = "public"`. Only addresses leave this type, never the text of the file.
    pub fn for_server_ips(&self) -> Vec<IpAddr> {
        let Some(for_server) = &self.for_server else {
            return Vec::new();
        };
        for_server
            .to_socket_addrs()
            .map(|it| it.map(|a| a.ip().to_canonical()).collect())
            .unwrap_or_default()
    }

    /// The port of `for_server` when it is a literal `ip:port` (never resolved): lets `proxy-check`, which has no game
    /// server in hand, apply the relay-port rule too.
    pub(crate) fn for_server_port(&self) -> Option<u16> {
        self.for_server.as_deref()?.parse::<SocketAddr>().ok().map(|a| a.port())
    }

    /// Names a server (`host:port`) whose addresses a relay must never be, as the `for_server` key of the file does.
    pub fn with_for_server(mut self, for_server: impl Into<String>) -> Self {
        self.for_server = Some(for_server.into());
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The IP addresses the proxy's `host:port` resolves to, and nothing else about it: the launcher (task 5.9, D-089) puts
    /// exactly these into the bot unit's cgroup filter (`IPAddressAllow=`), so the host name itself never leaves this type.
    /// Duplicates are removed; an empty answer is an error.
    pub fn resolve_ips(&self) -> std::io::Result<Vec<std::net::IpAddr>> {
        let mut ips: Vec<std::net::IpAddr> = Vec::new();
        for addr in (self.host.expose(), self.port).to_socket_addrs()? {
            let ip = addr.ip().to_canonical();
            if !ips.contains(&ip) {
                ips.push(ip);
            }
        }
        if ips.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "the proxy host resolves to no address",
            ));
        }
        Ok(ips)
    }

    pub(crate) fn host(&self) -> &Secret {
        &self.host
    }

    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) fn auth(&self) -> Option<&ProxyAuth> {
        self.auth.as_ref()
    }
}

/// A fresh session token: 8 characters from `[a-z0-9]`, different from every one in `used`. Randomness is from the
/// standard library's per-process hasher keys mixed with the clock: a session id needs to be unique, not secret.
pub(crate) fn fresh_session_token(used: &[String]) -> String {
    use std::hash::{BuildHasher, Hasher};
    const ALPHABET: &[u8; 36] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    loop {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        let mut n = h.finish();
        let mut token = String::with_capacity(8);
        for _ in 0..8 {
            token.push(char::from(ALPHABET[(n % 36) as usize]));
            n /= 36;
        }
        if !used.contains(&token) {
            return token;
        }
    }
}

/// A proxy name becomes part of a file name (`<name>-proxy.toml`), so it is restricted to `[A-Za-z0-9_-]`,
/// 1 to 64 characters: no path separators, no dots, nothing that could leave the secrets directory.
pub fn valid_proxy_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Why a proxy could not be loaded. No variant carries a value from the file.
#[derive(Debug, thiserror::Error)]
pub enum ProxyLoadError {
    #[error("invalid proxy name (allowed: 1-64 characters from A-Z a-z 0-9 _ -)")]
    InvalidName,
    #[error("proxy file {path} does not exist")]
    Missing { path: PathBuf },
    #[error("cannot read proxy file {path}: {kind:?}")]
    Io { path: PathBuf, kind: std::io::ErrorKind },
    #[error("proxy file {path} is not a regular file")]
    NotAFile { path: PathBuf },
    #[error(
        "proxy file {path} has mode {mode:04o}: it holds credentials and must not be accessible to group/other (chmod 600)"
    )]
    Permissions { path: PathBuf, mode: u32 },
    #[error("proxy file {path} is not valid UTF-8 TOML (syntax error{})", line_suffix(*.line))]
    Syntax { path: PathBuf, line: Option<usize> },
    #[error("proxy file {path}: {reason}")]
    Invalid { path: PathBuf, reason: &'static str },
    #[error("invalid proxy settings: {0}")]
    Field(&'static str),
}

fn line_suffix(line: Option<usize>) -> String {
    line.map(|l| format!(" at line {l}")).unwrap_or_default()
}

/// `<secrets_dir>/<name>-proxy.toml`.
pub fn proxy_file_path(secrets_dir: &Path, name: &str) -> Result<PathBuf, ProxyLoadError> {
    if !valid_proxy_name(name) {
        return Err(ProxyLoadError::InvalidName);
    }
    Ok(secrets_dir.join(format!("{name}-proxy.toml")))
}

/// `<data_dir>/secrets`: where `<name>-proxy.toml` lives (`~/aiddnet/data/secrets` in production).
pub fn secrets_dir_for(data_dir: &Path) -> PathBuf {
    data_dir.join("secrets")
}

/// Loads `<secrets_dir>/<name>-proxy.toml`.
///
/// Keys: `host` (string), `port` (integer), `user` and `pass` (strings, both or neither; `password` is
/// accepted as an alias of `pass`), `for_server` (string, `host:port`, optional), `relay` (`"proxy-host-only"`, the
/// default, or `"public"`, task 2.6b), `session_pick` (integer 0..=4, task 2.6b; 2..=4 needs `{session}` in `user`) and
/// `note` (ignored). Which server
/// gets the proxy is decided by the `proxy = "<name>"` field of the server's entry (`live-servers.toml`, or the owner's favourite,
/// task 5.12); `for_server` is no longer a binding, only extra addresses the relay rule and the cgroup filter keep away
/// ([`ProxyConfig::for_server_ips`]).
pub fn load_proxy(secrets_dir: &Path, name: &str) -> Result<ProxyConfig, ProxyLoadError> {
    let path = proxy_file_path(secrets_dir, name)?;
    // Task 5.12: the web unit can now create files in the secrets directory, and the root launcher helper reads them: a symlink
    // there is refused (`O_NOFOLLOW`) and a FIFO cannot block the reader (`O_NONBLOCK`).
    let mut file = match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(ProxyLoadError::Missing { path }),
        Err(e) if e.raw_os_error() == Some(libc::ELOOP) => return Err(ProxyLoadError::NotAFile { path }),
        Err(e) => return Err(ProxyLoadError::Io { path, kind: e.kind() }),
    };
    // The mode is read from the opened handle, so the file checked is the file read.
    let meta = file.metadata().map_err(|e| ProxyLoadError::Io {
        path: path.clone(),
        kind: e.kind(),
    })?;
    if !meta.is_file() {
        return Err(ProxyLoadError::NotAFile { path });
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = meta.permissions().mode() & 0o7777;
        if mode & 0o077 != 0 {
            return Err(ProxyLoadError::Permissions { path, mode });
        }
    }
    let mut bytes = Vec::new();
    // A proxy file is a few hundred bytes; refuse anything absurd before parsing it.
    if let Err(e) = (&mut file).take(64 * 1024).read_to_end(&mut bytes) {
        return Err(ProxyLoadError::Io { path, kind: e.kind() });
    }
    let text = match String::from_utf8(bytes) {
        Ok(t) => t,
        Err(_) => return Err(ProxyLoadError::Syntax { path, line: None }),
    };
    parse_proxy(name, &text, &path)
}

/// Parses the text of a proxy file exactly as [`load_proxy`] does (every rule: the keys, the relay mode, `session_pick` against
/// `{session}`), without touching the disk. The site validates a profile with this before it writes the file (task 5.12).
pub fn parse_proxy_text(name: &str, text: &str) -> Result<ProxyConfig, ProxyLoadError> {
    if !valid_proxy_name(name) {
        return Err(ProxyLoadError::InvalidName);
    }
    parse_proxy(name, text, Path::new("<proxy profile>"))
}

fn parse_proxy(name: &str, text: &str, path: &Path) -> Result<ProxyConfig, ProxyLoadError> {
    let table: toml::Table = match text.parse() {
        Ok(t) => t,
        Err(e) => {
            // `e`'s own text quotes the offending line: only the line number is kept.
            let line = e
                .span()
                .map(|s| text[..s.start.min(text.len())].bytes().filter(|&b| b == b'\n').count() + 1);
            return Err(ProxyLoadError::Syntax {
                path: path.to_path_buf(),
                line,
            });
        }
    };
    let invalid = |reason: &'static str| ProxyLoadError::Invalid {
        path: path.to_path_buf(),
        reason,
    };
    let get_str = |key: &str, wrong: &'static str| -> Result<Option<String>, ProxyLoadError> {
        match table.get(key) {
            None => Ok(None),
            Some(toml::Value::String(s)) => Ok(Some(s.clone())),
            Some(_) => Err(invalid(wrong)),
        }
    };
    let host = get_str("host", "`host` must be a string")?.ok_or_else(|| invalid("missing `host`"))?;
    let port = match table.get("port") {
        Some(toml::Value::Integer(p)) => u16::try_from(*p)
            .ok()
            .filter(|p| *p != 0)
            .ok_or_else(|| invalid("`port` must be 1..=65535"))?,
        Some(_) => return Err(invalid("`port` must be an integer")),
        None => return Err(invalid("missing `port`")),
    };
    let user = get_str("user", "`user` must be a string")?;
    let pass = match get_str("pass", "`pass` must be a string")? {
        Some(p) => Some(p),
        None => get_str("password", "`password` must be a string")?,
    };
    let auth = match (user, pass) {
        (None, None) => None,
        (Some(u), Some(p)) => Some((u, p)),
        (Some(_), None) => return Err(invalid("`user` is set but `pass` is missing")),
        (None, Some(_)) => return Err(invalid("`pass` is set but `user` is missing")),
    };
    let for_server = get_str("for_server", "`for_server` must be a string")?;
    if for_server.as_deref().is_some_and(str::is_empty) {
        return Err(invalid("`for_server` is empty"));
    }
    let relay = match table.get("relay") {
        None => RelayMode::default(),
        Some(toml::Value::String(s)) => {
            RelayMode::parse(s).ok_or_else(|| invalid("`relay` must be \"proxy-host-only\" or \"public\""))?
        }
        Some(_) => return Err(invalid("`relay` must be a string")),
    };
    let session_pick = match table.get("session_pick") {
        None => 0,
        Some(toml::Value::Integer(n)) => u8::try_from(*n)
            .ok()
            .filter(|n| *n <= MAX_SESSION_PICK)
            .ok_or_else(|| invalid("`session_pick` must be 0..=4"))?,
        Some(_) => return Err(invalid("`session_pick` must be an integer")),
    };
    let has_placeholder = auth.as_ref().is_some_and(|(u, _)| u.contains(SESSION_PLACEHOLDER));
    if has_placeholder && session_pick < 2 {
        return Err(invalid(
            "`user` has `{session}`: set `session_pick` to 2..=4 or remove it",
        ));
    }
    let cfg = ProxyConfig::new(name, host, port, auth)
        .map_err(|e| match e {
            ProxyLoadError::Field(reason) => invalid(reason),
            other => other,
        })?
        .with_relay(relay)
        .with_session_pick(session_pick)
        .map_err(|e| match e {
            ProxyLoadError::Field(reason) => invalid(reason),
            other => other,
        })?;
    Ok(match for_server {
        Some(f) => cfg.with_for_server(f),
        None => cfg,
    })
}

/// Why [`resolve_for_server`] refused.
#[derive(Debug, thiserror::Error)]
pub enum ProxyResolveError {
    #[error(transparent)]
    Binding(#[from] ProxyBindingError),
    #[error("the entry for {addr} names proxy {name:?}: {source}")]
    Load {
        addr: SocketAddr,
        name: String,
        source: ProxyLoadError,
    },
}

/// The proxy a connection to `addr` as `nick` must use, from the allow-list entry: `Ok(None)` for a
/// direct connection (no entry, or an entry without `proxy`), `Ok(Some(..))` when the entry names one.
/// `live_servers::check` (`ready = true`, the nick) is a separate gate that still applies.
///
/// Task 5.12 (D-099): the proxy file's `for_server` no longer has to match the server. Which proxy a server uses is the owner's
/// choice, made on the site and kept in that server's entry (a favourite's `connection`); the root helper validates it. A
/// `for_server` that a file still has only adds addresses to the ones a relay must never be and the cgroup filter denies.
pub fn resolve_for_server(
    addr: SocketAddr,
    nick: &str,
    list: &LiveServers,
    secrets_dir: &Path,
) -> Result<Option<ProxyConfig>, ProxyResolveError> {
    let Some(name) = list.proxy_binding(addr, nick)? else {
        return Ok(None);
    };
    let cfg = load_proxy(secrets_dir, name).map_err(|source| ProxyResolveError::Load {
        addr,
        name: name.to_string(),
        source,
    })?;
    Ok(Some(cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_ips_gives_the_addresses_only_and_dedups() {
        let cfg = ProxyConfig::new("p", "127.0.0.1", 1080, None).unwrap();
        assert_eq!(
            cfg.resolve_ips().unwrap(),
            vec!["127.0.0.1".parse::<std::net::IpAddr>().unwrap()]
        );
        let v6 = ProxyConfig::new("p", "::ffff:10.1.2.3", 1080, None).unwrap();
        assert_eq!(
            v6.resolve_ips().unwrap(),
            vec!["10.1.2.3".parse::<std::net::IpAddr>().unwrap()]
        );
    }

    const SECRET_USER: &str = "u-s3cr3t-user";
    const SECRET_PASS: &str = "p-s3cr3t-pass";
    const SECRET_HOST: &str = "proxy-host.secret.example";

    fn write_proxy(dir: &Path, name: &str, body: &str, mode: u32) -> PathBuf {
        let path = dir.join(format!("{name}-proxy.toml"));
        std::fs::write(&path, body).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        }
        let _ = mode;
        path
    }

    fn good_body() -> String {
        format!(
            "# comment\nhost = \"{SECRET_HOST}\"\nport = 1080\nuser = \"{SECRET_USER}\"\npass = \"{SECRET_PASS}\"\nfor_server = \"93.184.216.35:8308\"\nnote = \"y\"\n"
        )
    }

    fn assert_no_secret(text: &str) {
        for s in [SECRET_USER, SECRET_PASS, SECRET_HOST] {
            assert!(!text.contains(s), "leaked {s:?} in {text:?}");
        }
    }

    #[test]
    fn secret_debug_and_display_are_redacted() {
        let s = Secret::new("hunter2");
        assert_eq!(format!("{s}"), "<redacted>");
        assert_eq!(format!("{s:?}"), "Secret(<redacted>)");
        assert_eq!(format!("{s:#?}"), "Secret(<redacted>)");
        assert_eq!(s.expose(), "hunter2");
        let auth = ProxyAuth {
            user: Secret::new("alice"),
            pass: Secret::new("hunter2"),
        };
        let dbg = format!("{auth:?}{auth:#?}");
        assert!(!dbg.contains("alice") && !dbg.contains("hunter2"), "{dbg}");
    }

    #[test]
    fn proxy_config_debug_shows_only_the_name() {
        let cfg = ProxyConfig::new(
            "swarfey",
            SECRET_HOST,
            1080,
            Some((SECRET_USER.to_string(), SECRET_PASS.to_string())),
        )
        .unwrap();
        for text in [format!("{cfg:?}"), format!("{cfg:#?}")] {
            assert_no_secret(&text);
            assert!(!text.contains("1080"), "{text}");
            assert!(text.contains("swarfey"), "{text}");
        }
        // And through the ClientConfig that holds it.
        let client = crate::ClientConfig {
            proxy: Some(cfg),
            ..crate::ClientConfig::default()
        };
        assert_no_secret(&format!("{client:?}"));
    }

    #[test]
    fn names_that_could_leave_the_secrets_directory_are_refused() {
        for bad in [
            "",
            "..",
            "a/b",
            "a\\b",
            "a.b",
            "x y",
            "é",
            &"a".repeat(65),
            "../etc/passwd",
            "a\0b",
        ] {
            assert!(!valid_proxy_name(bad), "{bad:?}");
            assert!(matches!(
                proxy_file_path(Path::new("/s"), bad),
                Err(ProxyLoadError::InvalidName)
            ));
        }
        for ok in ["swarfey", "local-e2e", "a_B-9", &"a".repeat(64)] {
            assert!(valid_proxy_name(ok), "{ok:?}");
        }
    }

    #[test]
    fn loads_the_existing_swarfey_file_shape() {
        let dir = tempfile::tempdir().unwrap();
        write_proxy(dir.path(), "swarfey", &good_body(), 0o600);
        let cfg = load_proxy(dir.path(), "swarfey").unwrap();
        assert_eq!(cfg.name(), "swarfey");
        assert_eq!(cfg.host().expose(), SECRET_HOST);
        assert_eq!(cfg.port(), 1080);
        let auth = cfg.auth().unwrap();
        assert_eq!(auth.user.expose(), SECRET_USER);
        assert_eq!(auth.pass.expose(), SECRET_PASS);
    }

    #[test]
    fn password_is_an_alias_of_pass_and_credentials_are_optional() {
        let dir = tempfile::tempdir().unwrap();
        write_proxy(
            dir.path(),
            "a",
            "host = \"h\"\nport = 9\nuser = \"u\"\npassword = \"p\"\n",
            0o600,
        );
        assert_eq!(load_proxy(dir.path(), "a").unwrap().auth().unwrap().pass.expose(), "p");
        write_proxy(dir.path(), "b", "host = \"h\"\nport = 9\n", 0o600);
        assert!(load_proxy(dir.path(), "b").unwrap().auth().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_file_readable_by_group_or_other_is_refused_without_reading_it() {
        for mode in [0o644, 0o640, 0o604, 0o660, 0o666, 0o601] {
            let dir = tempfile::tempdir().unwrap();
            write_proxy(dir.path(), "swarfey", &good_body(), mode);
            let err = load_proxy(dir.path(), "swarfey").unwrap_err();
            assert!(matches!(err, ProxyLoadError::Permissions { .. }), "{mode:o}: {err:?}");
            assert_no_secret(&err.to_string());
            assert_no_secret(&format!("{err:?}"));
        }
        // Owner-only modes pass, including a stricter 0400.
        for mode in [0o600, 0o400] {
            let dir = tempfile::tempdir().unwrap();
            write_proxy(dir.path(), "swarfey", &good_body(), mode);
            assert!(load_proxy(dir.path(), "swarfey").is_ok(), "{mode:o}");
        }
    }

    #[test]
    fn a_missing_file_and_a_directory_are_typed_errors() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            load_proxy(dir.path(), "nope"),
            Err(ProxyLoadError::Missing { .. })
        ));
        std::fs::create_dir(dir.path().join("d-proxy.toml")).unwrap();
        let err = load_proxy(dir.path(), "d").unwrap_err();
        assert!(
            matches!(err, ProxyLoadError::NotAFile { .. } | ProxyLoadError::Io { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn malformed_content_never_echoes_a_value() {
        let dir = tempfile::tempdir().unwrap();
        let bodies = [
            // Syntax error right after a secret-bearing line.
            format!("host = \"{SECRET_HOST}\"\nport = 1080\npass = \"{SECRET_PASS}\" garbage\n"),
            // Wrong types quote the value in toml's own message.
            format!("host = \"{SECRET_HOST}\"\nport = \"{SECRET_PASS}\"\n"),
            format!("host = 5\nport = 1\nuser = \"{SECRET_USER}\"\n"),
            format!("host = \"{SECRET_HOST}\"\nport = 70000\nuser = \"{SECRET_USER}\"\npass = \"{SECRET_PASS}\"\n"),
            format!("host = \"{SECRET_HOST}\"\nport = 0\n"),
            format!("host = \"{SECRET_HOST}\"\nport = 1\nuser = \"{SECRET_USER}\"\n"),
            format!("host = \"{SECRET_HOST}\"\nport = 1\npass = \"{SECRET_PASS}\"\n"),
            format!("host = \"\"\nport = 1\nuser = \"{SECRET_USER}\"\npass = \"{SECRET_PASS}\"\n"),
            format!("port = 1\nuser = \"{SECRET_USER}\"\npass = \"{SECRET_PASS}\"\n"),
            format!("host = \"{SECRET_HOST}\"\nuser = \"{SECRET_USER}\"\npass = \"{SECRET_PASS}\"\n"),
            format!("host = \"{SECRET_HOST}\"\nport = 1\nuser = \"\"\npass = \"{SECRET_PASS}\"\n"),
            format!(
                "host = \"{SECRET_HOST}\"\nport = 1\nuser = \"{}\"\npass = \"{SECRET_PASS}\"\n",
                "x".repeat(256)
            ),
        ];
        for (i, body) in bodies.iter().enumerate() {
            write_proxy(dir.path(), "bad", body, 0o600);
            let err = load_proxy(dir.path(), "bad").unwrap_err();
            let text = format!("{err} / {err:?}");
            assert_no_secret(&text);
            assert!(!text.contains("xxxxxxxx"), "case {i}: {text}");
        }
        // Not UTF-8.
        let path = dir.path().join("bin-proxy.toml");
        std::fs::write(&path, [0xff, 0xfe, 0x00, b'h']).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert!(matches!(
            load_proxy(dir.path(), "bin"),
            Err(ProxyLoadError::Syntax { .. })
        ));
    }

    #[test]
    fn a_syntax_error_reports_its_line_only() {
        let dir = tempfile::tempdir().unwrap();
        write_proxy(
            dir.path(),
            "bad",
            &format!("host = \"h\"\nport = 1\npass = \"{SECRET_PASS}\" garbage\n"),
            0o600,
        );
        match load_proxy(dir.path(), "bad").unwrap_err() {
            ProxyLoadError::Syntax { line, .. } => assert_eq!(line, Some(3)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn resolve_uses_the_proxy_only_when_the_entry_names_it() {
        let dir = tempfile::tempdir().unwrap();
        write_proxy(dir.path(), "swarfey", &good_body(), 0o600);
        let list: LiveServers = toml::from_str(
            r#"
            [[server]]
            address = "93.184.216.35:8308"
            nick = "Muha"
            ready = true
            proxy = "swarfey"

            [[server]]
            address = "10.9.9.9:8303"
            nick = "Muha"
            ready = true
            "#,
        )
        .unwrap();
        let addr = |s: &str| -> SocketAddr { s.parse().unwrap() };
        let with = resolve_for_server(addr("93.184.216.35:8308"), "Muha", &list, dir.path()).unwrap();
        assert_eq!(with.unwrap().name(), "swarfey");
        // An entry without `proxy`, an unlisted address and loopback: direct, the file is never read.
        let empty_dir = tempfile::tempdir().unwrap();
        for a in ["10.9.9.9:8303", "127.0.0.1:8303", "1.2.3.4:5"] {
            assert!(
                resolve_for_server(addr(a), "Muha", &list, empty_dir.path())
                    .unwrap()
                    .is_none()
            );
        }
        // The entry names a proxy whose file is absent: an error, not a silent direct connection.
        let err = resolve_for_server(addr("93.184.216.35:8308"), "Muha", &list, empty_dir.path()).unwrap_err();
        assert!(matches!(err, ProxyResolveError::Load { .. }), "{err:?}");
        // Another nick has no binding either way.
        assert!(
            resolve_for_server(addr("93.184.216.35:8308"), "Other", &list, dir.path())
                .unwrap()
                .is_none()
        );
    }

    /// Task 5.12 (D-099): the file's `for_server` is read, but it is no binding any more: the owner assigns the proxy to a server
    /// on the site, and that assignment (the entry's `proxy`) is the one rule. Its addresses are only kept away from the relay.
    #[test]
    fn for_server_is_read_but_no_longer_pins_the_proxy() {
        let dir = tempfile::tempdir().unwrap();
        write_proxy(dir.path(), "swarfey", &good_body(), 0o600);
        let cfg = load_proxy(dir.path(), "swarfey").unwrap();
        assert_eq!(
            cfg.for_server_ips(),
            vec!["93.184.216.35".parse::<IpAddr>().unwrap()],
            "still read, for the deny list"
        );
        // An entry for ANOTHER server that names the proxy gets it (the old refusal is gone).
        let list: LiveServers = toml::from_str(
            "[[server]]\naddress = \"203.0.113.9:8303\"\nnick = \"Muha\"\nready = true\nproxy = \"swarfey\"\n",
        )
        .unwrap();
        let got = resolve_for_server("203.0.113.9:8303".parse().unwrap(), "Muha", &list, dir.path())
            .unwrap()
            .unwrap();
        assert_eq!(got.name(), "swarfey");
        assert!(!format!("{got:?}").contains("93.184"));
        // Bad values are rejected without echoing them.
        for body in ["for_server = 5\n", "for_server = \"\"\n"] {
            write_proxy(dir.path(), "bad", &format!("host = \"h\"\nport = 1\n{body}"), 0o600);
            assert!(matches!(
                load_proxy(dir.path(), "bad"),
                Err(ProxyLoadError::Invalid { .. })
            ));
        }
        // The proxy named by an entry must exist and load: no silent direct connection, no other proxy.
        let err = resolve_for_server(
            "203.0.113.9:8303".parse().unwrap(),
            "Muha",
            &list,
            tempfile::tempdir().unwrap().path(),
        )
        .unwrap_err();
        assert!(matches!(err, ProxyResolveError::Load { .. }), "{err:?}");
        for s in [SECRET_USER, SECRET_PASS, SECRET_HOST, "93.184.216.35"] {
            assert!(!format!("{err} {err:?}").contains(s), "{s}");
        }
    }

    #[test]
    fn a_proxy_file_that_is_a_symlink_or_a_fifo_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let real = write_proxy(dir.path(), "real", &good_body(), 0o600);
        std::os::unix::fs::symlink(&real, dir.path().join("link-proxy.toml")).unwrap();
        assert!(matches!(
            load_proxy(dir.path(), "link"),
            Err(ProxyLoadError::NotAFile { .. })
        ));
        assert!(load_proxy(dir.path(), "real").is_ok());
        // A FIFO: opened without blocking, then refused as not a regular file (never read).
        let fifo = dir.path().join("pipe-proxy.toml");
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );
        assert!(matches!(
            load_proxy(dir.path(), "pipe"),
            Err(ProxyLoadError::NotAFile { .. })
        ));
    }

    // --- task 2.6b: `relay` and `session_pick` ---------------------------------------------------------------

    fn load_body(body: &str) -> Result<ProxyConfig, ProxyLoadError> {
        let dir = tempfile::tempdir().unwrap();
        write_proxy(dir.path(), "r", body, 0o600);
        load_proxy(dir.path(), "r")
    }

    #[test]
    fn relay_defaults_to_proxy_host_only_and_reads_both_values() {
        assert_eq!(
            load_body("host = \"h\"\nport = 1\n").unwrap().relay_mode(),
            RelayMode::ProxyHostOnly
        );
        assert_eq!(
            load_body("host = \"h\"\nport = 1\nrelay = \"proxy-host-only\"\n")
                .unwrap()
                .relay_mode(),
            RelayMode::ProxyHostOnly
        );
        let public = load_body("host = \"h\"\nport = 1\nrelay = \"public\"\n").unwrap();
        assert_eq!(public.relay_mode(), RelayMode::Public);
        assert_eq!(RelayMode::Public.as_str(), "public");
        assert_eq!(RelayMode::ProxyHostOnly.as_str(), "proxy-host-only");
        // The existing real file's shape (no `relay`) is unchanged: the default.
        assert_eq!(load_body(&good_body()).unwrap().relay_mode(), RelayMode::ProxyHostOnly);
    }

    #[test]
    fn a_bad_relay_value_is_refused_without_echoing_it() {
        for body in [
            "relay = \"Public\"\n",
            "relay = \"any\"\n",
            "relay = \"\"\n",
            "relay = true\n",
            "relay = 1\n",
            "relay = [\"public\"]\n",
        ] {
            let err = load_body(&format!("host = \"h-secret-x\"\nport = 1\n{body}")).unwrap_err();
            assert!(matches!(err, ProxyLoadError::Invalid { .. }), "{body}: {err:?}");
            let text = format!("{err} {err:?}");
            assert!(!text.contains("Public") && !text.contains("any"), "{text}");
            assert!(!text.contains("h-secret-x"), "{text}");
        }
    }

    #[test]
    fn session_pick_needs_the_placeholder_and_the_placeholder_needs_session_pick() {
        let with = |user: &str, extra: &str| {
            load_body(&format!(
                "host = \"h\"\nport = 1\nuser = \"{user}\"\npass = \"p\"\n{extra}"
            ))
        };
        let ok = with("pre-{session}-post", "session_pick = 4\n").unwrap();
        assert_eq!(ok.session_pick(), 4);
        assert_eq!(with("u-{session}", "session_pick = 2\n").unwrap().session_pick(), 2);
        // 0 and 1 mean off; a plain user name is fine with them.
        assert_eq!(with("plain", "session_pick = 0\n").unwrap().session_pick(), 0);
        assert_eq!(with("plain", "session_pick = 1\n").unwrap().session_pick(), 0);
        assert_eq!(with("plain", "").unwrap().session_pick(), 0);
        // Refused: picking without a placeholder, a placeholder without picking, too many, wrong types, no credentials.
        for (user, extra) in [
            ("plain", "session_pick = 3\n"),
            ("u-{session}", ""),
            ("u-{session}", "session_pick = 1\n"),
            ("u-{session}", "session_pick = 0\n"),
            ("u-{session}", "session_pick = 5\n"),
            ("u-{session}", "session_pick = -1\n"),
            ("u-{session}", "session_pick = 300\n"),
            ("u-{session}", "session_pick = \"4\"\n"),
            ("u-{session}", "session_pick = 2.0\n"),
        ] {
            let err = with(user, extra).unwrap_err();
            assert!(matches!(err, ProxyLoadError::Invalid { .. }), "{user} {extra}: {err:?}");
        }
        let err = load_body("host = \"h\"\nport = 1\nsession_pick = 2\n").unwrap_err();
        assert!(matches!(err, ProxyLoadError::Invalid { .. }), "{err:?}");
        // The builder enforces the same.
        let plain = ProxyConfig::new("t", "h", 1, Some(("u".into(), "p".into()))).unwrap();
        assert!(plain.clone().with_session_pick(2).is_err());
        assert!(plain.clone().with_session_pick(5).is_err());
        assert!(plain.with_session_pick(1).is_ok());
    }

    #[test]
    fn the_session_placeholder_is_filled_everywhere_and_never_sent_literally() {
        let cfg = ProxyConfig::new("t", "h", 1, Some(("a-{session}-b-{session}".into(), "p".into())))
            .unwrap()
            .with_session_pick(2)
            .unwrap();
        assert_eq!(cfg.user_for(Some("tok12345")).unwrap(), "a-tok12345-b-tok12345");
        let fresh = cfg.user_for(None).unwrap();
        assert!(!fresh.contains("{session}") && fresh.starts_with("a-"), "{fresh}");
        let plain = ProxyConfig::new("t", "h", 1, Some(("alice".into(), "p".into()))).unwrap();
        assert_eq!(plain.user_for(Some("x")).unwrap(), "alice");
        assert_eq!(ProxyConfig::new("t", "h", 1, None).unwrap().user_for(None), None);
    }

    #[test]
    fn fresh_session_tokens_are_short_plain_and_distinct() {
        let mut seen: Vec<String> = Vec::new();
        for _ in 0..200 {
            let t = fresh_session_token(&seen);
            assert_eq!(t.len(), 8, "{t}");
            assert!(t.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()), "{t}");
            assert!(!seen.contains(&t));
            seen.push(t);
        }
    }

    #[test]
    fn relay_and_session_pick_show_in_debug_but_user_and_host_do_not() {
        let cfg = load_body(&format!(
            "host = \"{SECRET_HOST}\"\nport = 1080\nuser = \"{SECRET_USER}-{{session}}\"\npass = \"{SECRET_PASS}\"\nrelay = \"public\"\nsession_pick = 3\n"
        ))
        .unwrap();
        let text = format!("{cfg:?}{cfg:#?}");
        assert_no_secret(&text);
        assert!(text.contains("Public") && text.contains("session_pick"), "{text}");
    }

    #[test]
    fn the_test_hook_is_not_set_by_any_file() {
        let cfg = load_body("host = \"h\"\nport = 1\nrelay = \"public\"\n").unwrap();
        assert!(!cfg.loopback_relay_allowed());
        #[cfg(feature = "test-util")]
        assert!(cfg.with_test_loopback_relay().loopback_relay_allowed());
    }

    #[test]
    fn for_server_ips_and_port_are_read_without_printing() {
        let pinned = ProxyConfig::new("p", "h", 1, None)
            .unwrap()
            .with_for_server("[::ffff:93.184.216.35]:8308");
        assert_eq!(
            pinned.for_server_ips(),
            vec!["93.184.216.35".parse::<IpAddr>().unwrap()]
        );
        assert_eq!(pinned.for_server_port(), Some(8308));
        let none = ProxyConfig::new("p", "h", 1, None).unwrap();
        assert!(none.for_server_ips().is_empty());
        assert_eq!(none.for_server_port(), None);
        let bad = ProxyConfig::new("p", "h", 1, None)
            .unwrap()
            .with_for_server("not an address");
        assert!(bad.for_server_ips().is_empty());
        assert_eq!(bad.for_server_port(), None);
    }
}
