//! The owner's favourite servers, chosen on the site (task 5.12, D-099).
//!
//! `~/aiddnet/data/launch/favourites.json` is **written by the web** (the authenticated owner picks a server in the browser) and read
//! by three parties that trust nothing in it: the root launcher helper (`ddnet-ai launch apply`), the bot's own allow-list gate
//! (`live_servers::check`, the proxy binding) and the web itself. Every one of them parses it with [`Favourites::parse`], which is
//! strict, and treats a favourite as exactly what a `ready = true` entry of `live-servers.toml` is, plus the ban memory the helper
//! keeps (D-089): a favourite is **never** a way around anything that entry would have to pass.
//!
//! What is checked, for the file and for every entry (a file with one bad entry is refused as a whole, never half-trusted):
//!
//! - size <= [`MAX_FILE_BYTES`], strict schema (`deny_unknown_fields`), version [`FILE_VERSION`], at most [`MAX_FAVOURITES`] entries;
//! - `address`: **exactly** a socket address literal `ip:port` in its canonical spelling (no host name, no spaces, no leading zeros,
//!   no `::ffff:` mapped IPv4), port 1..=65535, the IP a **public unicast** address ([`crate::relay_rule::is_public_unicast`]:
//!   no loopback, private, link-local, CGNAT, multicast, documentation or reserved range). Nothing is resolved, ever;
//! - no two entries for the same address;
//! - `nick`: 1..=15 of `[A-Za-z0-9_-]` (it becomes the bot's `--name`);
//! - `connection`: `"direct"` or `"proxy:<name>"` with a valid proxy name ([`crate::proxy::valid_proxy_name`]);
//! - `name` (display only): 1..=64 characters, trimmed, no control or direction-changing characters; `notes`: at most 200, same;
//! - `consent_at`: the owner's confirmation that the server's admin allows the bot, as a unix time, never 0.
//!
//! Loopback is refused here (the local server needs no favourite: it is `local`). Tests that need a favourite on a loopback server
//! (the e2e with a private server) say so with [`Rules::allow_loopback`]; no production code path sets it.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::live_servers::{LiveServerEntry, LiveServers};
use crate::safe_file::{self, SafeReadError};

/// How far into the future a `reopened_at` may be (clock rounding) before the favourite is refused: a re-opening can only be an act
/// the owner did already, so a value in the future would lift bans that have not happened yet (review 5.12 F1).
pub const MAX_FUTURE_SKEW_SECS: u64 = 120;

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The file's schema version.
pub const FILE_VERSION: u32 = 1;
/// The most favourites.
pub const MAX_FAVOURITES: usize = 64;
/// The file's size cap (when reading and when writing).
pub const MAX_FILE_BYTES: usize = 64 * 1024;
/// Longest display name, in characters.
pub const MAX_NAME_CHARS: usize = 64;
/// Longest note, in characters.
pub const MAX_NOTES_CHARS: usize = 200;
/// The longest bot nick (DDNet's own limit is 15 bytes).
pub const MAX_NICK_LEN: usize = 15;
/// The nick the bot uses unless the owner picks another one for a server.
pub const DEFAULT_NICK: &str = "Muha";
/// The file name inside `<data-dir>/launch/`.
pub const FILE_NAME: &str = "favourites.json";

/// The favourites file of a data directory: `<data-dir>/launch/favourites.json`.
pub fn default_path(data_dir: &Path) -> PathBuf {
    data_dir.join("launch").join(FILE_NAME)
}

/// What a validator accepts. The default is the production rule set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rules {
    /// Accept a loopback address. **Tests only** (a private server on 127.0.0.1); see [`Rules::current`].
    pub allow_loopback: bool,
}

impl Rules {
    /// The rules this build uses: the production set, except that a build with the `loopback-favourites` cargo feature (the
    /// local e2e's test binary, a `cargo test` run) also accepts loopback. A release build of the production binary has no
    /// such feature, so no flag, file or environment can make it accept a loopback favourite.
    pub fn current() -> Rules {
        Rules {
            allow_loopback: cfg!(feature = "loopback-favourites"),
        }
    }
}

/// Why a favourite or the file was refused: a fixed code, never anything the file contained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FavouriteError {
    #[error("the favourites file is not valid (size, JSON, schema, version)")]
    BadFile,
    #[error("the address must be `ip:port` with a public IP address and a port 1..=65535")]
    BadAddress,
    #[error("the display name must be 1..=64 plain characters")]
    BadName,
    #[error("the nick must be 1..=15 characters from A-Z a-z 0-9 _ -")]
    BadNick,
    #[error("the connection must be `direct` or `proxy:<name>`")]
    BadConnection,
    #[error("the owner's consent (a time) is missing")]
    NoConsent,
    #[error("the note must be at most 200 plain characters")]
    BadNotes,
    #[error("two favourites have the same address")]
    Duplicate,
    #[error("too many favourites")]
    TooMany,
    #[error("`reopened_at` is dated in the future")]
    ReopenedInFuture,
}

impl FavouriteError {
    /// The code the web and the helper report.
    pub fn code(self) -> &'static str {
        match self {
            FavouriteError::BadFile => "favourites_invalid",
            FavouriteError::BadAddress => "bad_address",
            FavouriteError::BadName => "bad_name",
            FavouriteError::BadNick => "bad_nick",
            FavouriteError::BadConnection => "bad_connection",
            FavouriteError::NoConsent => "consent_required",
            FavouriteError::BadNotes => "bad_notes",
            FavouriteError::Duplicate => "duplicate",
            FavouriteError::TooMany => "too_many",
            FavouriteError::ReopenedInFuture => "reopened_invalid",
        }
    }
}

/// One favourite server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Favourite {
    /// `ip:port`, canonical spelling.
    pub address: String,
    /// Display name (from the server list, or typed).
    pub name: String,
    /// The nick the bot plays under there.
    pub nick: String,
    /// `"direct"` or `"proxy:<name>"`: the owner's choice, and the only way a proxy is ever chosen (never automatic).
    pub connection: String,
    /// When the owner confirmed that the server's admin allows the bot (unix seconds).
    pub consent_at: u64,
    #[serde(default)]
    pub notes: String,
    /// When it was added (unix seconds).
    pub added_at: u64,
    /// When the owner last **explicitly re-opened** it after a kick or ban (unix seconds, 0 = never). The helper's ban memory
    /// keeps the favourite closed until this is newer than the ban. Only the «Открыть снова» action sets it.
    #[serde(default)]
    pub reopened_at: u64,
}

impl Favourite {
    /// The proxy this favourite is assigned to, `None` for a direct connection. Valid only after [`Favourite::validate`].
    pub fn proxy_name(&self) -> Option<&str> {
        self.connection.strip_prefix("proxy:")
    }

    /// The parsed address. Valid only after [`Favourite::validate`].
    pub fn socket_addr(&self) -> Option<SocketAddr> {
        self.address.parse().ok()
    }

    /// Checks every field (see the module docs).
    pub fn validate(&self, rules: Rules) -> Result<(), FavouriteError> {
        parse_address(&self.address, rules)?;
        if !plain_text(&self.name, 1, MAX_NAME_CHARS) {
            return Err(FavouriteError::BadName);
        }
        if !valid_nick(&self.nick) {
            return Err(FavouriteError::BadNick);
        }
        valid_connection(&self.connection)?;
        if self.consent_at == 0 {
            return Err(FavouriteError::NoConsent);
        }
        if self.reopened_at > unix_now().saturating_add(MAX_FUTURE_SKEW_SECS) {
            return Err(FavouriteError::ReopenedInFuture);
        }
        if !self.notes.is_empty() && !plain_text(&self.notes, 1, MAX_NOTES_CHARS) {
            return Err(FavouriteError::BadNotes);
        }
        Ok(())
    }
}

/// A nick the bot may play under: 1..=15 of `[A-Za-z0-9_-]`.
pub fn valid_nick(nick: &str) -> bool {
    !nick.is_empty()
        && nick.len() <= MAX_NICK_LEN
        && nick
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// `"direct"` or `"proxy:<valid name>"`.
pub fn valid_connection(connection: &str) -> Result<(), FavouriteError> {
    if connection == "direct" {
        return Ok(());
    }
    match connection.strip_prefix("proxy:") {
        Some(name) if crate::proxy::valid_proxy_name(name) => Ok(()),
        _ => Err(FavouriteError::BadConnection),
    }
}

/// Characters that change how text is laid out or hide it: bidi controls, zero-width marks, the BOM, line and paragraph separators.
pub(crate) fn is_layout_trick(c: char) -> bool {
    matches!(c,
        '\u{200B}'..='\u{200F}' | '\u{2028}'..='\u{202E}' | '\u{2060}'..='\u{2064}' | '\u{2066}'..='\u{206F}' | '\u{FEFF}' | '\u{FFF9}'..='\u{FFFB}')
}

/// Display text: `min..=max` characters, trimmed (no space at either end), no control characters, no layout tricks.
pub fn plain_text(text: &str, min: usize, max: usize) -> bool {
    let n = text.chars().count();
    n >= min && n <= max && text.trim() == text && !text.chars().any(|c| c.is_control() || is_layout_trick(c))
}

/// Parses `ip:port` in its canonical spelling and checks the address class. The only door a server address goes through.
pub fn parse_address(text: &str, rules: Rules) -> Result<SocketAddr, FavouriteError> {
    if text.len() > 64 {
        return Err(FavouriteError::BadAddress);
    }
    let addr: SocketAddr = text.parse().map_err(|_| FavouriteError::BadAddress)?;
    if addr.port() == 0 || addr.to_string() != text {
        return Err(FavouriteError::BadAddress);
    }
    if let IpAddr::V6(v6) = addr.ip()
        && v6.to_ipv4_mapped().is_some()
    {
        return Err(FavouriteError::BadAddress);
    }
    let class_ok = crate::relay_rule::is_public_unicast(addr.ip()) || (rules.allow_loopback && addr.ip().is_loopback());
    if !class_ok {
        return Err(FavouriteError::BadAddress);
    }
    Ok(addr)
}

/// The favourites file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Favourites {
    pub v: u32,
    #[serde(default)]
    pub favourites: Vec<Favourite>,
}

impl Default for Favourites {
    fn default() -> Self {
        Favourites {
            v: FILE_VERSION,
            favourites: Vec::new(),
        }
    }
}

impl Favourites {
    /// Parses and validates the file's bytes (see the module docs). Any problem refuses the whole file.
    pub fn parse(bytes: &[u8], rules: Rules) -> Result<Favourites, FavouriteError> {
        if bytes.len() > MAX_FILE_BYTES {
            return Err(FavouriteError::BadFile);
        }
        let file: Favourites = serde_json::from_slice(bytes).map_err(|_| FavouriteError::BadFile)?;
        file.validate(rules)?;
        Ok(file)
    }

    /// Checks the version, the count, every entry and that no address repeats.
    pub fn validate(&self, rules: Rules) -> Result<(), FavouriteError> {
        if self.v != FILE_VERSION {
            return Err(FavouriteError::BadFile);
        }
        if self.favourites.len() > MAX_FAVOURITES {
            return Err(FavouriteError::TooMany);
        }
        let mut seen: Vec<SocketAddr> = Vec::with_capacity(self.favourites.len());
        for f in &self.favourites {
            f.validate(rules)?;
            let addr = parse_address(&f.address, rules)?;
            if seen.contains(&addr) {
                return Err(FavouriteError::Duplicate);
            }
            seen.push(addr);
        }
        Ok(())
    }

    /// The JSON the file holds: validated first, and within the size cap, so a writer can never produce a file the readers refuse.
    pub fn to_bytes(&self, rules: Rules) -> Result<Vec<u8>, FavouriteError> {
        self.validate(rules)?;
        let bytes = serde_json::to_vec_pretty(self).map_err(|_| FavouriteError::BadFile)?;
        if bytes.len() > MAX_FILE_BYTES {
            return Err(FavouriteError::TooMany);
        }
        Ok(bytes)
    }

    /// The favourite with this exact address text.
    pub fn find(&self, address: &str) -> Option<&Favourite> {
        self.favourites.iter().find(|f| f.address == address)
    }

    /// The allow-list entries the favourites stand for: each `ready = true`, pinned to its nick, with its proxy.
    pub fn entries(&self) -> Vec<LiveServerEntry> {
        self.favourites
            .iter()
            .map(|f| LiveServerEntry {
                address: f.address.clone(),
                nick: f.nick.clone(),
                purpose: "favourite".to_string(),
                ready: true,
                proxy: f.proxy_name().map(str::to_string),
            })
            .collect()
    }
}

/// Why the favourites could not be read from disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LoadError {
    #[error("the favourites file is not a regular file, too large or unreadable")]
    Unreadable,
    #[error(transparent)]
    Invalid(#[from] FavouriteError),
}

/// Reads and validates the favourites file. A missing file is an empty list; a file that cannot be trusted is an error (callers
/// fail closed: no favourite is usable then).
pub fn load(path: &Path, rules: Rules) -> Result<Favourites, LoadError> {
    match safe_file::read_regular_nofollow(path, MAX_FILE_BYTES) {
        Ok(read) => Ok(Favourites::parse(&read.bytes, rules)?),
        Err(SafeReadError::Missing) => Ok(Favourites::default()),
        Err(_) => Err(LoadError::Unreadable),
    }
}

/// A favourite's address also listed in `live-servers.toml` (by its address, resolved the way the allow-list resolves it).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("a favourite is also an entry of live-servers.toml: refusing to guess which one rules")]
pub struct DuplicateWithAllowList;

impl LiveServers {
    /// The allow-list plus the favourites as `ready` entries (the bot's gate and the proxy binding read this). A favourite whose
    /// address is already an entry is refused: two statements about one server would be a guess.
    pub fn with_favourites(mut self, favourites: &Favourites) -> Result<LiveServers, DuplicateWithAllowList> {
        for f in &favourites.favourites {
            let Some(addr) = f.socket_addr() else {
                return Err(DuplicateWithAllowList);
            };
            // By literal, and also by resolution: the bot's own gate resolves host names, so an entry `host:port` that resolves to
            // the favourite's address is a second statement about the same server (review 5.12 F2).
            if self.listed(addr) || self.allowed_nick(addr).is_some() {
                return Err(DuplicateWithAllowList);
            }
        }
        self.servers.extend(favourites.entries());
        Ok(self)
    }

    /// Whether some entry names its server by something other than an `ip:port` literal (a host name). Favourites are refused while
    /// the allow-list holds one: a host name can hide the very address a favourite names, and neither the web nor the root helper
    /// resolves names.
    pub fn has_non_literal_entry(&self) -> bool {
        self.servers.iter().any(|e| e.address.parse::<SocketAddr>().is_err())
    }

    /// Whether some entry's `address` is the literal `ip:port` of `addr`. Host names are **not** resolved here (the helper and the web
    /// must not wait for a resolver, and a favourite is a literal anyway).
    pub fn listed(&self, addr: SocketAddr) -> bool {
        self.servers.iter().any(|e| {
            e.address
                .parse::<SocketAddr>()
                .is_ok_and(|a| a.port() == addr.port() && a.ip().to_canonical() == addr.ip().to_canonical())
        })
    }

    /// The allow-list of `live_path` plus the favourites of `favourites_path` (a missing file adds none). A favourites file that
    /// cannot be trusted, or that clashes with the allow-list, adds **none** (fail closed) and says why in the second value.
    pub fn load_with_favourites(
        live: LiveServers,
        favourites_path: &Path,
        rules: Rules,
    ) -> (LiveServers, Option<&'static str>) {
        match load(favourites_path, rules) {
            Ok(favs) => match live.clone().with_favourites(&favs) {
                Ok(merged) => (merged, None),
                Err(_) => (live, Some("favourites_clash")),
            },
            Err(LoadError::Unreadable) => (live, Some("favourites_unreadable")),
            Err(LoadError::Invalid(_)) => (live, Some("favourites_invalid")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fav(address: &str) -> Favourite {
        Favourite {
            address: address.to_string(),
            name: "Some Block Server".to_string(),
            nick: "Muha".to_string(),
            connection: "direct".to_string(),
            consent_at: 1_700_000_000,
            notes: String::new(),
            added_at: 1_700_000_000,
            reopened_at: 0,
        }
    }

    fn file(favs: Vec<Favourite>) -> Favourites {
        Favourites {
            v: FILE_VERSION,
            favourites: favs,
        }
    }

    #[test]
    fn a_good_favourite_round_trips() {
        let f = file(vec![fav("93.184.216.35:8308"), fav("[2a01:4f8::1]:8303")]);
        let bytes = f.to_bytes(Rules::default()).unwrap();
        assert_eq!(Favourites::parse(&bytes, Rules::default()).unwrap(), f);
        assert_eq!(f.favourites[0].proxy_name(), None);
        let mut p = fav("1.2.3.4:8303");
        p.connection = "proxy:hproxy-1".to_string();
        assert_eq!(p.proxy_name(), Some("hproxy-1"));
        assert!(p.validate(Rules::default()).is_ok());
    }

    #[test]
    fn only_a_canonical_public_ip_and_port_is_an_address() {
        let rules = Rules::default();
        for good in [
            "93.184.216.35:8308",
            "8.8.8.8:1",
            "[2a01:4f8::1]:65535",
            "93.184.216.34:8303",
        ] {
            assert!(parse_address(good, rules).is_ok(), "{good}");
        }
        for bad in [
            "",
            "93.184.216.35",
            "93.184.216.35:0",
            "93.184.216.35:65536",
            "93.184.216.35:-1",
            "93.184.216.35:8308 ",
            " 93.184.216.35:8308",
            "93.184.216.35:08308",
            "093.184.216.35:8308",
            "93.184.216.035:8308",
            "example.com:8303",
            "localhost:8303",
            "0x5d.184.216.35:8303",
            "1572395043:8303",
            "93.184.216.35:8308\n",
            "93.184.216.35:8308#x",
            "http://93.184.216.35:8308",
            "[::ffff:93.184.216.35]:8308",
            "[2A01:4F8::1]:8303",
            "[2a01:4f8:0:0:0:0:0:1]:8303",
            // Not public unicast.
            "127.0.0.1:8303",
            "[::1]:8303",
            "0.0.0.0:8303",
            "10.0.0.5:8303",
            "172.16.0.1:8303",
            "192.168.1.1:8303",
            "169.254.169.254:80",
            "100.64.0.1:8303",
            "224.0.0.1:8303",
            "255.255.255.255:8303",
            "203.0.113.5:8308",
            "198.51.100.7:8308",
            "240.0.0.1:8303",
            "[fe80::1]:8303",
            "[fc00::1]:8303",
            "[2001:db8::1]:8303",
            "[64:ff9b::808:808]:8303",
            "[2002:808:808::1]:8303",
        ] {
            assert_eq!(parse_address(bad, rules), Err(FavouriteError::BadAddress), "{bad:?}");
        }
        let long = format!("{}:1", "1".repeat(80));
        assert_eq!(parse_address(&long, rules), Err(FavouriteError::BadAddress));
    }

    #[test]
    fn loopback_is_only_for_tests_that_say_so() {
        let t = Rules { allow_loopback: true };
        assert!(parse_address("127.0.0.1:8463", t).is_ok());
        assert!(parse_address("127.0.0.1:8463", Rules::default()).is_err());
        // Even then nothing private or reserved but loopback gets in.
        assert!(parse_address("10.0.0.1:8463", t).is_err());
        assert!(parse_address("0.0.0.0:8463", t).is_err());
    }

    #[test]
    fn names_nicks_notes_connections_and_consent_are_checked() {
        let rules = Rules::default();
        let mut ok = fav("93.184.216.35:8308");
        assert!(ok.validate(rules).is_ok());
        for bad in [
            "",
            " x",
            "x ",
            "a\nb",
            "a\u{0}b",
            "a\u{202E}b",
            "a\u{200B}b",
            "\u{feff}a",
            &"x".repeat(65),
        ] {
            let mut f = ok.clone();
            f.name = bad.to_string();
            assert_eq!(f.validate(rules), Err(FavouriteError::BadName), "{bad:?}");
        }
        let mut f = ok.clone();
        f.name = "Сервер 日本語 🌍".to_string();
        assert!(f.validate(rules).is_ok(), "unicode names are fine");
        f.name = "я".repeat(64);
        assert!(f.validate(rules).is_ok(), "64 characters, not bytes");
        for bad in [
            "",
            "Muha Bot",
            "a/b",
            "../x",
            "Muha\n",
            "ŕ",
            &"a".repeat(16),
            "a;b",
            "a\"b",
            "a$b",
        ] {
            let mut f = ok.clone();
            f.nick = bad.to_string();
            assert_eq!(f.validate(rules), Err(FavouriteError::BadNick), "{bad:?}");
        }
        for bad in [
            "",
            "Direct",
            "proxy",
            "proxy:",
            "proxy:a/b",
            "proxy:../x",
            "proxy:a b",
            "proxy:a.b",
            "proxy:A:B",
            "PROXY:a",
            "direct ",
            "direct:x",
        ] {
            let mut f = ok.clone();
            f.connection = bad.to_string();
            assert_eq!(f.validate(rules), Err(FavouriteError::BadConnection), "{bad:?}");
        }
        let mut f = ok.clone();
        f.connection = format!("proxy:{}", "a".repeat(65));
        assert_eq!(f.validate(rules), Err(FavouriteError::BadConnection));
        ok.consent_at = 0;
        assert_eq!(ok.validate(rules), Err(FavouriteError::NoConsent));
        ok.consent_at = 5;
        ok.notes = "x".repeat(201);
        assert_eq!(ok.validate(rules), Err(FavouriteError::BadNotes));
        ok.notes = "line\nbreak".to_string();
        assert_eq!(ok.validate(rules), Err(FavouriteError::BadNotes));
        ok.notes = "admin said yes in Discord".to_string();
        assert!(ok.validate(rules).is_ok());
    }

    #[test]
    fn the_file_is_strict() {
        let rules = Rules::default();
        let good = serde_json::to_value(file(vec![fav("93.184.216.35:8308")])).unwrap();
        assert!(Favourites::parse(good.to_string().as_bytes(), rules).is_ok());
        // Unknown fields, at either level.
        let mut v = good.clone();
        v["extra"] = serde_json::json!(1);
        assert_eq!(
            Favourites::parse(v.to_string().as_bytes(), rules),
            Err(FavouriteError::BadFile)
        );
        let mut v = good.clone();
        v["favourites"][0]["ready"] = serde_json::json!(true);
        assert_eq!(
            Favourites::parse(v.to_string().as_bytes(), rules),
            Err(FavouriteError::BadFile)
        );
        // The wrong version, types and shapes.
        let mut v = good.clone();
        v["v"] = serde_json::json!(2);
        assert_eq!(
            Favourites::parse(v.to_string().as_bytes(), rules),
            Err(FavouriteError::BadFile)
        );
        let mut v = good.clone();
        v["favourites"][0]["consent_at"] = serde_json::json!("now");
        assert_eq!(
            Favourites::parse(v.to_string().as_bytes(), rules),
            Err(FavouriteError::BadFile)
        );
        let mut v = good.clone();
        v["favourites"][0]["consent_at"] = serde_json::json!(-1);
        assert_eq!(
            Favourites::parse(v.to_string().as_bytes(), rules),
            Err(FavouriteError::BadFile)
        );
        for bytes in [
            &b""[..],
            b"null",
            b"[]",
            b"{",
            b"\xff\xfe",
            b"{\"v\":1,\"favourites\":5}",
        ] {
            assert_eq!(
                Favourites::parse(bytes, rules),
                Err(FavouriteError::BadFile),
                "{bytes:?}"
            );
        }
        // One bad entry refuses the whole file.
        let mut bad = fav("10.0.0.1:8303");
        bad.name = "x".into();
        let f = file(vec![fav("93.184.216.35:8308"), bad]);
        assert_eq!(f.validate(rules), Err(FavouriteError::BadAddress));
        assert!(
            f.to_bytes(rules).is_err(),
            "a writer cannot produce a file the readers refuse"
        );
        // Size cap, duplicates (also in another spelling of the same address) and the count.
        assert_eq!(
            Favourites::parse(&vec![b' '; MAX_FILE_BYTES + 1], rules),
            Err(FavouriteError::BadFile)
        );
        let dup = file(vec![fav("93.184.216.35:8308"), fav("93.184.216.35:8308")]);
        assert_eq!(dup.validate(rules), Err(FavouriteError::Duplicate));
        let many = file(
            (0..=MAX_FAVOURITES)
                .map(|i| fav(&format!("93.184.{}.{}:8308", i / 200, 10 + i % 200)))
                .collect(),
        );
        assert_eq!(many.validate(rules), Err(FavouriteError::TooMany));
    }

    #[test]
    fn a_favourite_stands_for_a_ready_entry_with_its_proxy() {
        let mut p = fav("93.184.216.35:8308");
        p.connection = "proxy:hp".to_string();
        let f = file(vec![p, fav("1.2.3.4:8303")]);
        let list = LiveServers::default().with_favourites(&f).unwrap();
        let addr: SocketAddr = "93.184.216.35:8308".parse().unwrap();
        assert_eq!(list.ready_nick(addr), Some("Muha"));
        assert_eq!(list.proxy_binding(addr, "Muha").unwrap(), Some("hp"));
        assert_eq!(
            list.proxy_binding("1.2.3.4:8303".parse().unwrap(), "Muha").unwrap(),
            None
        );
        // The gate admits it under its nick only.
        assert!(crate::live_servers::check(addr, "Muha", &list).is_ok());
        assert!(crate::live_servers::check(addr, "Other", &list).is_err());
        // An address that is not a favourite stays refused.
        assert!(crate::live_servers::check("93.184.216.36:8308".parse().unwrap(), "Muha", &list).is_err());
    }

    #[test]
    fn a_favourite_that_is_also_an_allow_list_entry_is_a_clash() {
        let live: LiveServers = toml::from_str(
            "[[server]]\naddress = \"93.184.216.35:8308\"\nnick = \"Muha\"\nready = true\nproxy = \"swarfey\"\n",
        )
        .unwrap();
        let f = file(vec![fav("93.184.216.35:8308")]);
        assert!(live.clone().with_favourites(&f).is_err());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE_NAME);
        std::fs::write(&path, f.to_bytes(Rules::default()).unwrap()).unwrap();
        let (merged, why) = LiveServers::load_with_favourites(live.clone(), &path, Rules::default());
        assert_eq!(why, Some("favourites_clash"));
        assert_eq!(merged, live, "fail closed: no favourite is added");
    }

    #[test]
    fn a_reopening_dated_in_the_future_is_refused() {
        let rules = Rules::default();
        let mut f = fav("93.184.216.35:8308");
        f.reopened_at = 99_999_999_999;
        assert_eq!(f.validate(rules), Err(FavouriteError::ReopenedInFuture));
        assert!(file(vec![f.clone()]).to_bytes(rules).is_err());
        assert_eq!(
            Favourites::parse(serde_json::to_string(&file(vec![f])).unwrap().as_bytes(), rules),
            Err(FavouriteError::ReopenedInFuture)
        );
        let mut ok = fav("93.184.216.35:8308");
        ok.reopened_at = unix_now() + MAX_FUTURE_SKEW_SECS - 5;
        assert!(ok.validate(rules).is_ok());
        ok.reopened_at = unix_now() + MAX_FUTURE_SKEW_SECS + 60;
        assert!(ok.validate(rules).is_err());
    }

    #[test]
    fn a_host_name_entry_is_detected_and_a_name_that_resolves_to_a_favourite_is_a_clash() {
        let named: LiveServers =
            toml::from_str("[[server]]\naddress = \"example.invalid:8303\"\nnick = \"Muha\"\nready = true\n").unwrap();
        assert!(named.has_non_literal_entry());
        let literal: LiveServers =
            toml::from_str("[[server]]\naddress = \"1.2.3.4:8303\"\nnick = \"Muha\"\nready = true\n").unwrap();
        assert!(!literal.has_non_literal_entry());
        // The reviewer's case: `one.one.one.one` resolves to 1.1.1.1 (skipped when this machine cannot resolve it).
        let one: LiveServers = toml::from_str(
            "[[server]]\naddress = \"one.one.one.one:8303\"\nnick = \"Muha\"\nready = true\nproxy = \"swarfey\"\n",
        )
        .unwrap();
        let resolves = std::net::ToSocketAddrs::to_socket_addrs("one.one.one.one:8303")
            .is_ok_and(|mut it| it.any(|a| a.to_string() == "1.1.1.1:8303"));
        if resolves {
            let mut f = fav("1.1.1.1:8303");
            f.nick = "Muha2".to_string();
            assert!(
                one.clone().with_favourites(&file(vec![f.clone()])).is_err(),
                "fails closed"
            );
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(FILE_NAME);
            std::fs::write(&path, file(vec![f]).to_bytes(Rules::default()).unwrap()).unwrap();
            let (merged, why) = LiveServers::load_with_favourites(one, &path, Rules::default());
            assert_eq!(why, Some("favourites_clash"));
            assert_eq!(merged.servers.len(), 1);
        }
    }

    #[test]
    fn loading_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE_NAME);
        let rules = Rules::default();
        // Missing: none, no complaint.
        let (m, why) = LiveServers::load_with_favourites(LiveServers::default(), &path, rules);
        assert!(m.servers.is_empty() && why.is_none());
        // Garbage, a symlink and a directory: none, with the reason.
        std::fs::write(&path, b"{not json").unwrap();
        let (m, why) = LiveServers::load_with_favourites(LiveServers::default(), &path, rules);
        assert!(m.servers.is_empty());
        assert_eq!(why, Some("favourites_invalid"));
        std::fs::remove_file(&path).unwrap();
        let real = dir.path().join("real.json");
        std::fs::write(&real, file(vec![fav("93.184.216.35:8308")]).to_bytes(rules).unwrap()).unwrap();
        std::os::unix::fs::symlink(&real, &path).unwrap();
        let (m, why) = LiveServers::load_with_favourites(LiveServers::default(), &path, rules);
        assert!(m.servers.is_empty());
        assert_eq!(why, Some("favourites_unreadable"));
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert_eq!(load(&path, rules), Err(LoadError::Unreadable));
        std::fs::remove_dir(&path).unwrap();
        // A good file.
        std::fs::copy(&real, &path).unwrap();
        let (m, why) = LiveServers::load_with_favourites(LiveServers::default(), &path, rules);
        assert_eq!((m.servers.len(), why), (1, None));
    }
}
