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
//! - Nothing from the file ever reaches a log line, a panic, a `Debug`/`Display` output or an error message:
//!   host, port, user and password are all held as [`Secret`]s, [`ProxyConfig`]'s `Debug` prints only the
//!   name, and every error here is built from key names and line numbers, never from values (a TOML syntax
//!   error's own text quotes the offending line, so it is **not** forwarded).

use crate::live_servers::{LiveServers, ProxyBindingError};
use std::fmt;
use std::io::Read;
use std::net::{SocketAddr, ToSocketAddrs};
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

/// A SOCKS5 proxy the driver may tunnel the game's UDP through. The address is as secret as the
/// credentials (it is the owner's, D-053), so it too is a [`Secret`] and only the `name` is ever shown.
#[derive(Clone, PartialEq, Eq)]
pub struct ProxyConfig {
    name: String,
    host: Secret,
    port: u16,
    auth: Option<ProxyAuth>,
    /// The file's own `for_server` (`host:port`): the one server this proxy was issued for (D-053). `None` when
    /// the file has none (hand-built configs).
    for_server: Option<String>,
}

impl fmt::Debug for ProxyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProxyConfig")
            .field("name", &self.name)
            .field("endpoint", &"<redacted>")
            .field("auth", &self.auth.as_ref().map(|_| "<redacted>"))
            .field("for_server", &self.for_server.as_ref().map(|_| "<set>"))
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
        })
    }

    /// Pins the proxy to one server (`host:port`), as the `for_server` key of the file does.
    pub fn with_for_server(mut self, for_server: impl Into<String>) -> Self {
        self.for_server = Some(for_server.into());
        self
    }

    /// Whether this proxy may be used for `target`: always, if the file named no `for_server`; otherwise only if
    /// `for_server` (an `ip:port`, or `host:port` resolved now) is `target`. A `for_server` that cannot be resolved
    /// allows nothing. This is the proxy file's own binding (D-053: the proxy is for one server only), checked in
    /// addition to the allow-list entry's `proxy = "<name>"`, so a scratch allow-list cannot route the proxy to
    /// another server.
    pub fn allows(&self, target: SocketAddr) -> bool {
        let Some(for_server) = &self.for_server else {
            return true;
        };
        for_server.to_socket_addrs().is_ok_and(|mut it| {
            it.any(|a| a.ip().to_canonical() == target.ip().to_canonical() && a.port() == target.port())
        })
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
/// accepted as an alias of `pass`), `for_server` (string, `host:port`, optional) and `note` (ignored). Which server
/// gets the proxy is decided by the `proxy = "<name>"` field of the server's entry in `live-servers.toml`; when
/// the file has `for_server` the proxy is **also** only usable for that server ([`ProxyConfig::allows`]).
pub fn load_proxy(secrets_dir: &Path, name: &str) -> Result<ProxyConfig, ProxyLoadError> {
    let path = proxy_file_path(secrets_dir, name)?;
    let mut file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(ProxyLoadError::Missing { path }),
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
    let cfg = ProxyConfig::new(name, host, port, auth).map_err(|e| match e {
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
    #[error(
        "proxy {name:?} is issued for another server (its file's `for_server`), not for {addr}: refusing to use it there (D-053)"
    )]
    NotForServer { addr: SocketAddr, name: String },
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
    if !cfg.allows(addr) {
        return Err(ProxyResolveError::NotForServer {
            addr,
            name: name.to_string(),
        });
    }
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
            "# comment\nhost = \"{SECRET_HOST}\"\nport = 1080\nuser = \"{SECRET_USER}\"\npass = \"{SECRET_PASS}\"\nfor_server = \"45.141.57.35:8308\"\nnote = \"y\"\n"
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
            address = "45.141.57.35:8308"
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
        let with = resolve_for_server(addr("45.141.57.35:8308"), "Muha", &list, dir.path()).unwrap();
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
        let err = resolve_for_server(addr("45.141.57.35:8308"), "Muha", &list, empty_dir.path()).unwrap_err();
        assert!(matches!(err, ProxyResolveError::Load { .. }), "{err:?}");
        // Another nick has no binding either way.
        assert!(
            resolve_for_server(addr("45.141.57.35:8308"), "Other", &list, dir.path())
                .unwrap()
                .is_none()
        );
    }

    /// F4: the proxy file's own `for_server` pins it to one server (D-053), whatever the allow-list says.
    #[test]
    fn for_server_pins_the_proxy_to_one_server() {
        let addr = |s: &str| -> SocketAddr { s.parse().unwrap() };
        let free = ProxyConfig::new("p", "h", 1, None).unwrap();
        assert!(
            free.allows(addr("203.0.113.9:1")),
            "a hand-built config without for_server allows any"
        );
        let pinned = ProxyConfig::new("p", "h", 1, None)
            .unwrap()
            .with_for_server("45.141.57.35:8308");
        assert!(pinned.allows(addr("45.141.57.35:8308")));
        for other in ["45.141.57.35:8309", "45.141.57.36:8308", "127.0.0.1:8303", "[::1]:8308"] {
            assert!(!pinned.allows(addr(other)), "{other}");
        }
        // IPv4-mapped IPv6 is the same endpoint.
        assert!(pinned.allows(addr("[::ffff:45.141.57.35]:8308")));
        // Unresolvable or malformed: allows nothing.
        for bad in ["not an address", "45.141.57.35", "never-resolved.invalid:8308"] {
            let p = ProxyConfig::new("p", "h", 1, None).unwrap().with_for_server(bad);
            assert!(!p.allows(addr("45.141.57.35:8308")), "{bad}");
        }
        // Not printed.
        assert!(!format!("{pinned:?}").contains("45.141"));
    }

    #[test]
    fn for_server_is_read_from_the_file_and_enforced_by_resolve() {
        let dir = tempfile::tempdir().unwrap();
        write_proxy(dir.path(), "swarfey", &good_body(), 0o600);
        let cfg = load_proxy(dir.path(), "swarfey").unwrap();
        assert!(cfg.allows("45.141.57.35:8308".parse().unwrap()));
        assert!(!cfg.allows("127.0.0.1:8303".parse().unwrap()));
        // A scratch allow-list that names the Swarfey proxy for another server is refused.
        let list: LiveServers = toml::from_str(
            "[[server]]\naddress = \"203.0.113.9:8303\"\nnick = \"Muha\"\nready = true\nproxy = \"swarfey\"\n",
        )
        .unwrap();
        let err = resolve_for_server("203.0.113.9:8303".parse().unwrap(), "Muha", &list, dir.path()).unwrap_err();
        assert!(matches!(err, ProxyResolveError::NotForServer { .. }), "{err:?}");
        assert!(err.to_string().contains("another server"));
        let text = format!("{err} {err:?}");
        for s in [SECRET_USER, SECRET_PASS, SECRET_HOST, "45.141.57.35"] {
            assert!(!text.contains(s), "{s}");
        }
        // Bad values are rejected without echoing them.
        for body in ["for_server = 5\n", "for_server = \"\"\n"] {
            write_proxy(dir.path(), "bad", &format!("host = \"h\"\nport = 1\n{body}"), 0o600);
            assert!(matches!(
                load_proxy(dir.path(), "bad"),
                Err(ProxyLoadError::Invalid { .. })
            ));
        }
        // No for_server in the file: only the allow-list binds it.
        write_proxy(dir.path(), "free", "host = \"h\"\nport = 1\n", 0o600);
        assert!(
            load_proxy(dir.path(), "free")
                .unwrap()
                .allows("1.2.3.4:5".parse().unwrap())
        );
    }
}
