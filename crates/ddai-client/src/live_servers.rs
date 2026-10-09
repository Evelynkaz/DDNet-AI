//! `~/aiddnet/data/live-servers.toml` (task 8.4a acceptance criterion 5; D-027/D-038): the
//! owner-curated allow-list of the *non-loopback* servers this project may ever connect to for
//! live play or observer recording, each pinned to the one nick approved for it (D-027: Swarfey
//! `93.184.216.35:8308`, nick "Muha"). The file lives outside this git repository (`CLAUDE.md`'s
//! "never commit ... addresses of the owner's servers" is not violated by code that only *reads*
//! a path under `~/aiddnet/data`) and is loaded at runtime, never baked into a binary.
//!
//! Loopback (`127.0.0.1`/`::1`) is always allowed and never needs an entry here — [`check`]'s
//! whole purpose is gating *non-loopback* addresses, matching the letter of the acceptance
//! criterion ("refuses any non-loopback address that is not in that list"). This module is
//! generic over "live play and recording" (the file's own header says both), not specific to the
//! observer recorder — `ddnet-ai record` (task 8.4a) is simply its first caller.

use serde::Deserialize;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};

/// One `[[server]]` entry.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct LiveServerEntry {
    /// `"host:port"`, textual, exactly as the owner wrote it (e.g. `"93.184.216.35:8308"`).
    pub address: String,
    /// The one nick approved for this server.
    pub nick: String,
    /// Free-text note (e.g. `"observer-recording"`) — informational only, never checked by
    /// [`check`]; kept so the file stays self-documenting for the human editing it.
    #[serde(default)]
    pub purpose: String,
    /// Task 4.3: the owner has said this server is ready for **live play** (the IP or the proxy is
    /// whitelisted, D-052/D-053). `--server auto` ([`crate::server_list`]) considers only entries with
    /// `ready = true`; [`check`] (an explicit `--server <addr>`, the recorder) ignores it, as before.
    /// Missing means not ready.
    #[serde(default)]
    pub ready: bool,
    /// Task 2.6 (D-053 amendment): the name of the SOCKS5 proxy this server is reached through; the proxy
    /// itself is read from `<secrets>/<name>-proxy.toml` (`crate::proxy`). A proxy is used for a server **only**
    /// when its entry names it (the driver refuses any other pairing), and the entry still needs `ready = true`
    /// like every non-loopback entry. Missing means a direct connection.
    #[serde(default)]
    pub proxy: Option<String>,
}

/// The parsed file: a flat list of [`LiveServerEntry`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
pub struct LiveServers {
    #[serde(default, rename = "server")]
    pub servers: Vec<LiveServerEntry>,
}

#[derive(Debug, thiserror::Error)]
pub enum LiveServersLoadError {
    #[error("reading {path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("parsing {path}: {source}")]
    Toml { path: PathBuf, source: toml::de::Error },
}

impl LiveServers {
    /// `~/aiddnet/data/live-servers.toml`, falling back to a relative `data/live-servers.toml` if
    /// `$HOME` isn't set — same fallback pattern as every other `default_*` helper in this crate
    /// (`session::default_cache_dir`, `ddnet-ai`'s `default_data_dir`).
    pub fn default_path() -> PathBuf {
        match std::env::var_os("HOME") {
            Some(home) if !home.is_empty() => PathBuf::from(home)
                .join("aiddnet")
                .join("data")
                .join("live-servers.toml"),
            _ => PathBuf::from("data").join("live-servers.toml"),
        }
    }

    /// A missing file is treated as "no non-loopback server is allowed" (an empty list), not an
    /// error: the safe default before the owner has ever created the file, and after task 8.4a's
    /// own local-only development (this task never creates the file's Swarfey entry as a *test*
    /// side effect — see the crate's BUILD REPORT for how it was actually created, once, by hand).
    pub fn load_or_empty(path: &Path) -> Result<Self, LiveServersLoadError> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text, path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(LiveServers::default()),
            Err(source) => Err(LiveServersLoadError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    fn parse(text: &str, path: &Path) -> Result<Self, LiveServersLoadError> {
        toml::from_str(text).map_err(|source| LiveServersLoadError::Toml {
            path: path.to_path_buf(),
            source,
        })
    }

    /// The nick pinned to `addr` by some entry, if any — entries are matched by resolving each
    /// entry's textual `address` and comparing the resulting [`SocketAddr`]s, not by comparing
    /// strings (so `"93.184.216.35:8308"` matches regardless of incidental formatting).
    /// Unparsable/unresolvable entries are skipped, not treated as errors — a typo in one entry
    /// must not make every other entry (or loopback, which never needs this file at all) stop
    /// working.
    pub fn allowed_nick(&self, addr: SocketAddr) -> Option<&str> {
        self.servers
            .iter()
            .find(|e| resolve(&e.address).is_some_and(|resolved| resolved.contains(&addr)))
            .map(|e| e.nick.as_str())
    }

    /// The entries `--server auto` may pick: the owner's `ready = true` ones.
    pub fn ready_entries(&self) -> impl Iterator<Item = &LiveServerEntry> {
        self.servers.iter().filter(|e| e.ready)
    }

    /// The proxy a connection to `addr` as `nick` must go through (task 2.6): the `proxy` of the entry that
    /// admits this connection, or `None` for a direct one.
    ///
    /// The entries considered are exactly the ones [`check`] would admit: those whose address matches, whose
    /// nick is `nick`, and (for a non-loopback address) which are `ready`. Loopback needs no entry at all, so
    /// with none it is direct. If the admitting entries disagree (one names a proxy, another does not, or two
    /// name different ones) the answer is an error, never a guess: connecting directly to a server that is only
    /// reachable through its proxy would be exactly the VPN-ban incident of D-052.
    pub fn proxy_binding(&self, addr: SocketAddr, nick: &str) -> Result<Option<&str>, ProxyBindingError> {
        let mut found: Option<Option<&str>> = None;
        for e in &self.servers {
            if e.nick != nick || !(is_loopback(addr) || e.ready) {
                continue;
            }
            if !resolve(&e.address).is_some_and(|r| r.contains(&addr)) {
                continue;
            }
            let name = e.proxy.as_deref();
            if name.is_some_and(|n| !crate::proxy::valid_proxy_name(n)) {
                return Err(ProxyBindingError::InvalidName { addr });
            }
            match found {
                None => found = Some(name),
                Some(prev) if prev == name => {}
                Some(_) => return Err(ProxyBindingError::Ambiguous { addr }),
            }
        }
        Ok(found.flatten())
    }

    /// [`LiveServers::allowed_nick`] for a `ready` entry only.
    pub fn ready_nick(&self, addr: SocketAddr) -> Option<&str> {
        self.ready_entries()
            .find(|e| resolve(&e.address).is_some_and(|resolved| resolved.contains(&addr)))
            .map(|e| e.nick.as_str())
    }
}

/// Why [`LiveServers::proxy_binding`] could not give one answer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProxyBindingError {
    #[error(
        "live-servers.toml has several entries for {addr} under this nick that disagree about `proxy`: refusing to guess"
    )]
    Ambiguous { addr: SocketAddr },
    #[error(
        "the `proxy` of the entry for {addr} is not a valid proxy name (allowed: 1-64 characters from A-Z a-z 0-9 _ -)"
    )]
    InvalidName { addr: SocketAddr },
}

fn resolve(address: &str) -> Option<Vec<SocketAddr>> {
    address.to_socket_addrs().ok().map(|it| it.collect())
}

pub fn is_loopback(addr: SocketAddr) -> bool {
    addr.ip().is_loopback()
}

/// Why [`check`] refused a connection — task acceptance criterion 5's "the recorder refuses any
/// non-loopback address that is not in that list [with the allowed nick]".
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LiveServerCheckError {
    #[error("{addr} is not a loopback address and is not listed in live-servers.toml (D-027/D-038)")]
    NotAllowed { addr: SocketAddr },
    #[error(
        "{addr} is listed in live-servers.toml, but without `ready = true`: the owner has not said this server may be connected to from here yet (D-052/D-053)"
    )]
    NotReady { addr: SocketAddr },
    #[error(
        "{addr} is listed in live-servers.toml, but only under nick {expected:?} — refusing to connect as {requested:?}"
    )]
    WrongNick {
        addr: SocketAddr,
        expected: String,
        requested: String,
    },
}

/// Task 4.3 (review F1): a non-loopback entry is also **refused unless it carries `ready = true`** — the
/// owner's explicit "this server may be connected to from here now" (D-052: the VPS is banned on Swarfey
/// until the IP or a proxy is whitelisted). This holds for every caller: `play`, `record`, `--server auto`
/// and the client driver's own gate.
///
/// Task 8.4a acceptance criterion 5: `Ok(())` for any loopback address (this task's own local
/// development never needs an entry in the file at all), or for a non-loopback address listed in
/// `list` under exactly the nick `requested_nick` — every other non-loopback address, and every
/// listed address requested under the *wrong* nick (D-027/D-038 pin one nick per server; a typo'd
/// or different `--name` is not "the approved observer"), is refused.
pub fn check(addr: SocketAddr, requested_nick: &str, list: &LiveServers) -> Result<(), LiveServerCheckError> {
    if is_loopback(addr) {
        return Ok(());
    }
    let matching: Vec<&LiveServerEntry> = list
        .servers
        .iter()
        .filter(|e| resolve(&e.address).is_some_and(|r| r.contains(&addr)))
        .collect();
    if matching.is_empty() {
        return Err(LiveServerCheckError::NotAllowed { addr });
    }
    // Only a ready entry counts; one that is merely listed is refused whatever the nick.
    let ready: Vec<&&LiveServerEntry> = matching.iter().filter(|e| e.ready).collect();
    if ready.is_empty() {
        return Err(LiveServerCheckError::NotReady { addr });
    }
    if ready.iter().any(|e| e.nick == requested_nick) {
        return Ok(());
    }
    Err(LiveServerCheckError::WrongNick {
        addr,
        expected: ready[0].nick.clone(),
        requested: requested_nick.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn swarfey_list() -> LiveServers {
        LiveServers::parse(
            r#"
            [[server]]
            address = "93.184.216.35:8308"
            nick = "Muha"
            purpose = "observer-recording"
            ready = true
            "#,
            Path::new("<test>"),
        )
        .expect("valid test TOML")
    }

    /// The owner's real file today: Swarfey listed, no `ready`.
    fn real_shaped_list() -> LiveServers {
        LiveServers::parse(
            r#"
            [[server]]
            address = "93.184.216.35:8308"
            nick = "Muha"
            purpose = "observer-recording"
            "#,
            Path::new("<test>"),
        )
        .expect("valid test TOML")
    }

    #[test]
    fn a_listed_entry_without_ready_is_refused_naming_the_flag() {
        let list = real_shaped_list();
        let err = check("93.184.216.35:8308".parse().unwrap(), "Muha", &list).unwrap_err();
        assert!(matches!(err, LiveServerCheckError::NotReady { .. }), "{err:?}");
        assert!(err.to_string().contains("ready = true"), "{err}");
        // Under any nick, and with `ready = false` written out.
        assert!(check("93.184.216.35:8308".parse().unwrap(), "Other", &list).is_err());
        let off = LiveServers::parse(
            "[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = false\n",
            Path::new("<test>"),
        )
        .unwrap();
        assert!(matches!(
            check("93.184.216.35:8308".parse().unwrap(), "Muha", &off),
            Err(LiveServerCheckError::NotReady { .. })
        ));
        // Loopback never needs an entry.
        assert!(check("127.0.0.1:8303".parse().unwrap(), "Muha", &list).is_ok());
    }

    #[test]
    fn a_ready_entry_is_found_even_when_a_not_ready_one_for_the_same_address_comes_first() {
        let list = LiveServers::parse(
            "[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Old\"\n\n[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = true\n",
            Path::new("<test>"),
        )
        .unwrap();
        assert!(check("93.184.216.35:8308".parse().unwrap(), "Muha", &list).is_ok());
        assert!(matches!(
            check("93.184.216.35:8308".parse().unwrap(), "Old", &list),
            Err(LiveServerCheckError::WrongNick { .. })
        ));
    }

    #[test]
    fn loopback_is_always_allowed_regardless_of_the_list() {
        let empty = LiveServers::default();
        assert!(check("127.0.0.1:8303".parse().unwrap(), "anyone", &empty).is_ok());
        assert!(check("[::1]:8303".parse().unwrap(), "anyone", &empty).is_ok());
    }

    #[test]
    fn a_listed_address_with_the_right_nick_is_allowed() {
        let list = swarfey_list();
        assert!(check("93.184.216.35:8308".parse().unwrap(), "Muha", &list).is_ok());
    }

    #[test]
    fn a_listed_address_with_the_wrong_nick_is_refused() {
        let list = swarfey_list();
        let err = check("93.184.216.35:8308".parse().unwrap(), "SomeoneElse", &list).unwrap_err();
        assert!(matches!(err, LiveServerCheckError::WrongNick { .. }));
    }

    #[test]
    fn an_unlisted_non_loopback_address_is_refused() {
        let list = swarfey_list();
        let err = check("1.2.3.4:8303".parse().unwrap(), "Muha", &list).unwrap_err();
        assert!(matches!(err, LiveServerCheckError::NotAllowed { .. }));
    }

    #[test]
    fn an_empty_list_refuses_every_non_loopback_address() {
        let list = LiveServers::default();
        let err = check("93.184.216.35:8308".parse().unwrap(), "Muha", &list).unwrap_err();
        assert!(matches!(err, LiveServerCheckError::NotAllowed { .. }));
    }

    #[test]
    fn missing_file_loads_as_an_empty_list_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.toml");
        let loaded = LiveServers::load_or_empty(&path).expect("missing file is not an error");
        assert_eq!(loaded, LiveServers::default());
    }

    #[test]
    fn real_file_round_trips_through_load_or_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live-servers.toml");
        std::fs::write(
            &path,
            "[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\npurpose = \"observer-recording\"\n",
        )
        .unwrap();
        let loaded = LiveServers::load_or_empty(&path).expect("valid file");
        assert_eq!(loaded.servers.len(), 1);
        assert_eq!(loaded.servers[0].address, "93.184.216.35:8308");
        assert_eq!(loaded.servers[0].nick, "Muha");
        assert!(!loaded.servers[0].ready, "a file without `ready` means not ready");
        // The owner's real file today has exactly this shape: Swarfey is listed and still refused.
        let err = check("93.184.216.35:8308".parse().unwrap(), "Muha", &loaded).unwrap_err();
        assert!(matches!(err, LiveServerCheckError::NotReady { .. }), "{err:?}");
        // Once the owner adds `ready = true` it is allowed under that nick only.
        std::fs::write(
            &path,
            "[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\npurpose = \"observer-recording\"\nready = true\n",
        )
        .unwrap();
        let loaded = LiveServers::load_or_empty(&path).expect("valid file");
        assert!(check("93.184.216.35:8308".parse().unwrap(), "Muha", &loaded).is_ok());
        assert!(check("93.184.216.35:8308".parse().unwrap(), "Other", &loaded).is_err());
    }

    #[test]
    fn malformed_toml_is_a_load_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live-servers.toml");
        std::fs::write(&path, "this is not valid toml [[[").unwrap();
        let err = LiveServers::load_or_empty(&path).unwrap_err();
        assert!(matches!(err, LiveServersLoadError::Toml { .. }));
    }

    /// A single bad/unresolvable entry must not break lookups for the rest of the file.
    #[test]
    fn an_unresolvable_entry_is_skipped_not_fatal() {
        let list = LiveServers::parse(
            r#"
            [[server]]
            address = "not a valid address at all"
            nick = "Ghost"

            [[server]]
            address = "93.184.216.35:8308"
            nick = "Muha"
            ready = true
            "#,
            Path::new("<test>"),
        )
        .expect("valid test TOML");
        assert!(check("93.184.216.35:8308".parse().unwrap(), "Muha", &list).is_ok());
    }

    fn parse(text: &str) -> LiveServers {
        LiveServers::parse(text, Path::new("<test>")).expect("valid test TOML")
    }

    fn addr(s: &str) -> SocketAddr {
        s.parse().unwrap()
    }

    #[test]
    fn the_proxy_field_is_optional_and_defaults_to_a_direct_connection() {
        let list = parse("[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = true\n");
        assert_eq!(list.servers[0].proxy, None);
        assert_eq!(list.proxy_binding(addr("93.184.216.35:8308"), "Muha"), Ok(None));
    }

    #[test]
    fn an_entry_names_its_proxy_and_only_for_the_right_address_nick_and_ready() {
        let list =
            parse("[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = true\nproxy = \"swarfey\"\n");
        assert_eq!(
            list.proxy_binding(addr("93.184.216.35:8308"), "Muha"),
            Ok(Some("swarfey"))
        );
        // Another port, another address, another nick: no binding (and `check` refuses them anyway).
        assert_eq!(list.proxy_binding(addr("93.184.216.35:8309"), "Muha"), Ok(None));
        assert_eq!(list.proxy_binding(addr("1.2.3.4:8308"), "Muha"), Ok(None));
        assert_eq!(list.proxy_binding(addr("93.184.216.35:8308"), "Other"), Ok(None));
        // Not ready: the entry does not admit the connection, so it does not bind a proxy either.
        let off = parse("[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nproxy = \"swarfey\"\n");
        assert_eq!(off.proxy_binding(addr("93.184.216.35:8308"), "Muha"), Ok(None));
        assert!(matches!(
            check(addr("93.184.216.35:8308"), "Muha", &off),
            Err(LiveServerCheckError::NotReady { .. })
        ));
    }

    #[test]
    fn a_loopback_entry_may_name_a_proxy_without_ready_and_loopback_without_an_entry_is_direct() {
        let list = parse("[[server]]\naddress = \"127.0.0.1:8303\"\nnick = \"Muha\"\nproxy = \"local\"\n");
        assert_eq!(list.proxy_binding(addr("127.0.0.1:8303"), "Muha"), Ok(Some("local")));
        assert_eq!(list.proxy_binding(addr("127.0.0.1:8303"), "Other"), Ok(None));
        assert_eq!(list.proxy_binding(addr("127.0.0.1:8313"), "Muha"), Ok(None));
        assert_eq!(
            LiveServers::default().proxy_binding(addr("127.0.0.1:8303"), "Muha"),
            Ok(None)
        );
    }

    #[test]
    fn entries_that_disagree_about_the_proxy_are_an_error_not_a_guess() {
        let mixed = parse(
            "[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = true\n\n\
             [[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = true\nproxy = \"swarfey\"\n",
        );
        assert!(matches!(
            mixed.proxy_binding(addr("93.184.216.35:8308"), "Muha"),
            Err(ProxyBindingError::Ambiguous { .. })
        ));
        let two = parse(
            "[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = true\nproxy = \"a\"\n\n\
             [[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = true\nproxy = \"b\"\n",
        );
        assert!(two.proxy_binding(addr("93.184.216.35:8308"), "Muha").is_err());
        // The same proxy twice is fine.
        let same = parse(
            "[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = true\nproxy = \"a\"\n\n\
             [[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = true\nproxy = \"a\"\n",
        );
        assert_eq!(same.proxy_binding(addr("93.184.216.35:8308"), "Muha"), Ok(Some("a")));
    }

    #[test]
    fn a_proxy_name_that_is_a_path_is_refused() {
        for bad in ["../x", "a/b", "", "a.b"] {
            let list = parse(&format!(
                "[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = true\nproxy = \"{bad}\"\n"
            ));
            assert!(
                matches!(
                    list.proxy_binding(addr("93.184.216.35:8308"), "Muha"),
                    Err(ProxyBindingError::InvalidName { .. })
                ),
                "{bad:?}"
            );
        }
    }
}
