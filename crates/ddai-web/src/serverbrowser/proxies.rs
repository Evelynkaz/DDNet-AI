//! The proxy profiles the owner manages on the site: `<data-dir>/secrets/<name>-proxy.toml`, mode 0600, the format `ddai-client`
//! already reads (`docs/formats.md` §33). **The password is write-only**: no function here returns it, no error or log line contains it
//! (errors are fixed codes), and the page never receives the user name either (it shows «задан» and keeps the old value when the
//! field is left empty on edit). The host is shown only for profiles the site made itself.
//!
//! The site only ever touches files it made: they carry `managed_by = "ddnet-ai-web"`. A hand-made file (the production
//! `swarfey-proxy.toml`) is listed read-only, its host hidden, and can be neither overwritten nor removed from the page.
//!
//! A profile is validated with the very parser the bot and the helper use ([`ddai_client::proxy::parse_proxy_text`]) before it is
//! written, and the host must be a **public unicast IP literal** (no host name, nothing private or loopback): the «Проверить» button
//! makes a machine with network rights connect to it.

use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ddai_client::favourites::Rules;
use ddai_client::proxy::{self, MAX_SESSION_PICK, ProxyLoadError, valid_proxy_name};
use serde::{Deserialize, Serialize};

use crate::launch::{read_regular_nofollow, write_atomic};

/// The key that marks a file as the site's own.
pub const MANAGED_KEY: &str = "managed_by";
pub const MANAGED_VALUE: &str = "ddnet-ai-web";
const MAX_FILE_BYTES: usize = 64 * 1024;
/// The most profiles the directory may hold (counted over every `*-proxy.toml`).
pub const MAX_PROFILES: usize = 32;
const SUFFIX: &str = "-proxy.toml";

/// What the browser sends to create or edit a profile. Unknown fields are refused.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProxyForm {
    pub name: String,
    pub host: String,
    pub port: u16,
    /// Empty or absent on an edit keeps the stored one.
    pub user: Option<String>,
    /// Write-only. Empty or absent on an edit keeps the stored one.
    pub pass: Option<String>,
    /// `proxy-host-only` (default) or `public`.
    pub relay: Option<String>,
    /// 0 or absent: off; 2..=4 needs `{session}` in the user name.
    pub session_pick: Option<u8>,
}

/// One profile as the page sees it. No user name, no password.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProxyView {
    pub name: String,
    /// Made on the site (editable and removable there); `false` for a hand-made file.
    pub managed: bool,
    /// Only for a managed profile.
    pub host: Option<String>,
    pub port: Option<u16>,
    /// A user name and a password are stored (their values are never shown).
    pub has_credentials: bool,
    pub relay: String,
    pub session_pick: u8,
    /// The bot and the helper can load it.
    pub usable: bool,
    /// Why not: `bad_mode`, `bad_file`, `missing`.
    pub problem: Option<&'static str>,
}

/// Why a change was refused: a fixed code for the page, never a value from the form or a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyError {
    BadName,
    BadHost,
    BadPort,
    BadUser,
    BadPass,
    BadRelay,
    BadSessionPick,
    /// The profile fails the parser's own rules (for example `session_pick` without `{session}`).
    Invalid,
    Exists,
    NotFound,
    /// A hand-made file: the site does not touch it.
    NotManaged,
    TooMany,
    WriteFailed,
}

impl ProxyError {
    pub fn code(self) -> &'static str {
        match self {
            ProxyError::BadName => "bad_name",
            ProxyError::BadHost => "bad_host",
            ProxyError::BadPort => "bad_port",
            ProxyError::BadUser => "bad_user",
            ProxyError::BadPass => "bad_pass",
            ProxyError::BadRelay => "bad_relay",
            ProxyError::BadSessionPick => "bad_session_pick",
            ProxyError::Invalid => "proxy_invalid",
            ProxyError::Exists => "proxy_exists",
            ProxyError::NotFound => "proxy_not_found",
            ProxyError::NotManaged => "proxy_not_managed",
            ProxyError::TooMany => "too_many_proxies",
            ProxyError::WriteFailed => "proxy_write_failed",
        }
    }
}

pub struct ProxyStore {
    dir: PathBuf,
    rules: Rules,
    lock: Mutex<()>,
}

fn plain_credential(s: &str) -> bool {
    (1..=255).contains(&s.len()) && s.bytes().all(|b| (0x20..=0x7e).contains(&b))
}

impl ProxyStore {
    pub fn new(secrets_dir: &Path, rules: Rules) -> ProxyStore {
        ProxyStore {
            dir: secrets_dir.to_path_buf(),
            rules,
            lock: Mutex::new(()),
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{name}{SUFFIX}"))
    }

    /// The names of every `<name>-proxy.toml` in the directory (valid names only), sorted.
    fn names(&self) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .filter_map(Result::ok)
            .filter_map(|e| e.file_name().into_string().ok())
            .filter_map(|f| f.strip_suffix(SUFFIX).map(str::to_string))
            .filter(|n| valid_proxy_name(n))
            .collect();
        names.sort();
        names.truncate(MAX_PROFILES * 2);
        names
    }

    /// Whether a profile with this name exists as a regular file.
    pub fn exists(&self, name: &str) -> bool {
        valid_proxy_name(name) && std::fs::symlink_metadata(self.path(name)).is_ok_and(|m| m.file_type().is_file())
    }

    fn table_of(&self, name: &str) -> Option<toml::Table> {
        let bytes = read_regular_nofollow(&self.path(name), MAX_FILE_BYTES).ok()?;
        String::from_utf8(bytes).ok()?.parse::<toml::Table>().ok()
    }

    /// Every profile, for the page.
    pub fn list(&self) -> Vec<ProxyView> {
        self.names().into_iter().map(|n| self.view(&n)).collect()
    }

    fn view(&self, name: &str) -> ProxyView {
        let table = self.table_of(name);
        let managed = table
            .as_ref()
            .and_then(|t| t.get(MANAGED_KEY))
            .and_then(toml::Value::as_str)
            == Some(MANAGED_VALUE);
        let get_str = |k: &str| table.as_ref().and_then(|t| t.get(k)).and_then(toml::Value::as_str);
        let (usable, problem) = match proxy::load_proxy(&self.dir, name) {
            Ok(_) => (true, None),
            Err(ProxyLoadError::Missing { .. }) => (false, Some("missing")),
            Err(ProxyLoadError::Permissions { .. }) => (false, Some("bad_mode")),
            Err(_) => (false, Some("bad_file")),
        };
        ProxyView {
            name: name.to_string(),
            managed,
            host: if managed {
                get_str("host").map(str::to_string)
            } else {
                None
            },
            port: if managed {
                table
                    .as_ref()
                    .and_then(|t| t.get("port"))
                    .and_then(toml::Value::as_integer)
                    .and_then(|p| u16::try_from(p).ok())
            } else {
                None
            },
            has_credentials: get_str("user").is_some() && (get_str("pass").is_some() || get_str("password").is_some()),
            relay: get_str("relay").unwrap_or("proxy-host-only").to_string(),
            session_pick: table
                .as_ref()
                .and_then(|t| t.get("session_pick"))
                .and_then(toml::Value::as_integer)
                .and_then(|n| u8::try_from(n).ok())
                .unwrap_or(0),
            usable,
            problem,
        }
    }

    fn check_host(&self, host: &str) -> Result<(), ProxyError> {
        let ip: IpAddr = host.parse().map_err(|_| ProxyError::BadHost)?;
        let ok = ddai_client::relay_rule::is_public_unicast(ip) || (self.rules.allow_loopback && ip.is_loopback());
        if ok && host.len() <= 45 {
            Ok(())
        } else {
            Err(ProxyError::BadHost)
        }
    }

    /// Validates a form and builds the file's text. `old` is the stored table when editing.
    fn compose(&self, form: &ProxyForm, old: Option<&toml::Table>) -> Result<String, ProxyError> {
        if !valid_proxy_name(&form.name) {
            return Err(ProxyError::BadName);
        }
        self.check_host(&form.host)?;
        if form.port == 0 {
            return Err(ProxyError::BadPort);
        }
        let relay = form.relay.as_deref().unwrap_or("proxy-host-only");
        if !matches!(relay, "proxy-host-only" | "public") {
            return Err(ProxyError::BadRelay);
        }
        let pick = form.session_pick.unwrap_or(0);
        if pick == 1 || pick > MAX_SESSION_PICK {
            return Err(ProxyError::BadSessionPick);
        }
        let stored = |k: &str| {
            old.and_then(|t| t.get(k))
                .and_then(toml::Value::as_str)
                .map(str::to_string)
        };
        let user = match form.user.as_deref() {
            Some(u) if !u.is_empty() => {
                if !plain_credential(u) {
                    return Err(ProxyError::BadUser);
                }
                Some(u.to_string())
            }
            _ => stored("user"),
        };
        let pass = match form.pass.as_deref() {
            Some(p) if !p.is_empty() => {
                if !plain_credential(p) {
                    return Err(ProxyError::BadPass);
                }
                Some(p.to_string())
            }
            _ => stored("pass").or_else(|| stored("password")),
        };
        let mut table = toml::Table::new();
        table.insert(MANAGED_KEY.into(), toml::Value::String(MANAGED_VALUE.into()));
        table.insert("host".into(), toml::Value::String(form.host.clone()));
        table.insert("port".into(), toml::Value::Integer(i64::from(form.port)));
        match (user, pass) {
            (Some(u), Some(p)) => {
                table.insert("user".into(), toml::Value::String(u));
                table.insert("pass".into(), toml::Value::String(p));
            }
            (None, None) => {}
            (Some(_), None) => return Err(ProxyError::BadPass),
            (None, Some(_)) => return Err(ProxyError::BadUser),
        }
        table.insert("relay".into(), toml::Value::String(relay.into()));
        if pick >= 2 {
            table.insert("session_pick".into(), toml::Value::Integer(i64::from(pick)));
        }
        let text = toml::to_string(&table).map_err(|_| ProxyError::Invalid)?;
        // The parser the bot and the helper use: every rule of the format (a `{session}` placeholder needs `session_pick`).
        proxy::parse_proxy_text(&form.name, &text).map_err(|_| ProxyError::Invalid)?;
        Ok(text)
    }

    fn write(&self, name: &str, text: &str) -> Result<(), ProxyError> {
        write_atomic(&self.dir, &format!("{name}{SUFFIX}"), text.as_bytes(), 0o600).map_err(|_| ProxyError::WriteFailed)
    }

    /// A new profile. An existing name (of any file) is refused.
    pub fn create(&self, form: &ProxyForm) -> Result<(), ProxyError> {
        let _g = self.lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !valid_proxy_name(&form.name) {
            return Err(ProxyError::BadName);
        }
        if std::fs::symlink_metadata(self.path(&form.name)).is_ok() {
            return Err(ProxyError::Exists);
        }
        if self.names().len() >= MAX_PROFILES {
            return Err(ProxyError::TooMany);
        }
        let text = self.compose(form, None)?;
        self.write(&form.name, &text)
    }

    /// Edits a profile the site made. Empty `user` / `pass` keep the stored ones.
    pub fn update(&self, form: &ProxyForm) -> Result<(), ProxyError> {
        let _g = self.lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !valid_proxy_name(&form.name) {
            return Err(ProxyError::BadName);
        }
        let old = self.managed_table(&form.name)?;
        let text = self.compose(form, Some(&old))?;
        self.write(&form.name, &text)
    }

    /// The stored table of a profile the site made, or why it may not be touched.
    fn managed_table(&self, name: &str) -> Result<toml::Table, ProxyError> {
        if !self.exists(name) {
            return Err(ProxyError::NotFound);
        }
        let table = self.table_of(name).ok_or(ProxyError::NotManaged)?;
        if table.get(MANAGED_KEY).and_then(toml::Value::as_str) != Some(MANAGED_VALUE) {
            return Err(ProxyError::NotManaged);
        }
        Ok(table)
    }

    /// Removes a profile the site made. The caller has checked that no favourite uses it.
    pub fn remove(&self, name: &str) -> Result<(), ProxyError> {
        let _g = self.lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !valid_proxy_name(name) {
            return Err(ProxyError::BadName);
        }
        self.managed_table(name)?;
        std::fs::remove_file(self.path(name)).map_err(|_| ProxyError::WriteFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    const SECRET_USER: &str = "u-s3cr3t-user-{session}";
    const SECRET_PASS: &str = "p-s3cr3t-pass";

    fn form(name: &str) -> ProxyForm {
        ProxyForm {
            name: name.into(),
            host: "198.51.100.7".into(),
            port: 1080,
            user: Some("plainuser".into()),
            pass: Some(SECRET_PASS.into()),
            relay: None,
            session_pick: None,
        }
    }

    // 198.51.100.7 is a documentation address: not public. Tests use a real-looking public one.
    fn pubform(name: &str) -> ProxyForm {
        let mut f = form(name);
        f.host = "93.184.216.34".into();
        f
    }

    fn store() -> (tempfile::TempDir, ProxyStore) {
        let dir = tempfile::tempdir().unwrap();
        let s = ProxyStore::new(dir.path(), Rules::default());
        (dir, s)
    }

    #[test]
    fn a_profile_is_written_0600_in_the_format_the_bot_reads_and_the_password_never_comes_back() {
        let (dir, s) = store();
        let mut f = pubform("hp-1");
        f.user = Some(SECRET_USER.into());
        f.relay = Some("public".into());
        f.session_pick = Some(3);
        s.create(&f).unwrap();
        let path = dir.path().join("hp-1-proxy.toml");
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        // The bot's own loader accepts it.
        let cfg = proxy::load_proxy(dir.path(), "hp-1").unwrap();
        assert_eq!(cfg.relay_mode(), proxy::RelayMode::Public);
        assert_eq!(cfg.session_pick(), 3);
        // What the page gets: no user, no password.
        let list = s.list();
        assert_eq!(list.len(), 1);
        let v = &list[0];
        assert!(v.managed && v.usable && v.has_credentials);
        assert_eq!(
            (v.host.as_deref(), v.port, v.relay.as_str(), v.session_pick),
            (Some("93.184.216.34"), Some(1080), "public", 3)
        );
        let json = serde_json::to_string(&list).unwrap();
        for secret in [SECRET_PASS, SECRET_USER, "s3cr3t"] {
            assert!(!json.contains(secret), "{json}");
        }
        // Editing with empty credentials keeps them; the file still holds them, the view still does not.
        let mut e = pubform("hp-1");
        e.user = None;
        e.pass = Some(String::new());
        e.port = 1081;
        e.relay = Some("public".into());
        e.session_pick = Some(3);
        s.update(&e).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(SECRET_PASS) && text.contains("port = 1081"), "{text}");
        assert!(!serde_json::to_string(&s.list()).unwrap().contains(SECRET_PASS));
        // A new password replaces the old one.
        let mut e = pubform("hp-1");
        e.pass = Some("another-pass".into());
        e.user = Some("plainuser".into());
        s.update(&e).unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().contains("another-pass"));
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    }

    #[test]
    fn malicious_fields_are_refused_and_nothing_is_written() {
        let (dir, s) = store();
        let bad_names = ["", "a b", "../x", "a/b", "a.b", "x".repeat(65).leak(), "a\nb", "ä"];
        for n in bad_names {
            assert_eq!(s.create(&pubform(n)).unwrap_err(), ProxyError::BadName, "{n:?}");
        }
        for h in [
            "",
            "localhost",
            "example.com",
            "127.0.0.1",
            "10.0.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "0.0.0.0",
            "::1",
            "[2a01:4f8::1]",
            "93.184.216.34:1080",
            "93.184.216.34 ",
            "http://93.184.216.34",
            "93.184.216.034",
            "198.51.100.7",
            "fe80::1",
            "100.64.0.1",
            "224.0.0.1",
            "93.184.216.34\n",
        ] {
            let mut f = pubform("ok");
            f.host = h.into();
            assert_eq!(s.create(&f).unwrap_err(), ProxyError::BadHost, "{h:?}");
        }
        let mut f = pubform("ok");
        f.port = 0;
        assert_eq!(s.create(&f).unwrap_err(), ProxyError::BadPort);
        for u in ["", "a\nb", "a\u{0}b", "ünï", &"u".repeat(256)] {
            let mut f = pubform("ok");
            f.user = Some(u.into());
            f.pass = Some("p".into());
            let e = s.create(&f).unwrap_err();
            // An empty user on create means "none", so a password without one is refused as a bad user.
            assert!(matches!(e, ProxyError::BadUser), "{u:?} {e:?}");
        }
        for p in ["a\nb", "a\u{7f}b", "ünï", &"p".repeat(256)] {
            let mut f = pubform("ok");
            f.pass = Some(p.into());
            assert_eq!(s.create(&f).unwrap_err(), ProxyError::BadPass, "{p:?}");
        }
        let mut f = pubform("ok");
        f.relay = Some("anywhere".into());
        assert_eq!(s.create(&f).unwrap_err(), ProxyError::BadRelay);
        for n in [1u8, 5, 255] {
            let mut f = pubform("ok");
            f.session_pick = Some(n);
            assert_eq!(s.create(&f).unwrap_err(), ProxyError::BadSessionPick, "{n}");
        }
        // The parser's own rules: a pick needs `{session}`; `{session}` needs a pick.
        let mut f = pubform("ok");
        f.session_pick = Some(2);
        assert_eq!(s.create(&f).unwrap_err(), ProxyError::Invalid);
        let mut f = pubform("ok");
        f.user = Some("u-{session}".into());
        assert_eq!(s.create(&f).unwrap_err(), ProxyError::Invalid);
        // A user without a password is refused too.
        let mut f = pubform("ok");
        f.pass = None;
        assert_eq!(s.create(&f).unwrap_err(), ProxyError::BadPass);
        assert!(
            std::fs::read_dir(dir.path()).unwrap().next().is_none(),
            "nothing was written"
        );
    }

    #[test]
    fn quotes_and_toml_syntax_in_values_cannot_inject_keys() {
        let (dir, s) = store();
        let mut f = pubform("inj");
        f.user = Some("u\"\nrelay = \"public".into());
        // Newlines are refused outright; quotes alone are escaped and stay inside the value.
        assert_eq!(s.create(&f).unwrap_err(), ProxyError::BadUser);
        f.user = Some("u\" relay = \"public".into());
        f.pass = Some("p\\\"q".into());
        s.create(&f).unwrap();
        let text = std::fs::read_to_string(dir.path().join("inj-proxy.toml")).unwrap();
        let table: toml::Table = text.parse().unwrap();
        assert_eq!(table["relay"].as_str(), Some("proxy-host-only"), "{text}");
        assert_eq!(table["user"].as_str(), Some("u\" relay = \"public"));
        assert_eq!(table["pass"].as_str(), Some("p\\\"q"));
    }

    #[test]
    fn existing_names_other_files_and_the_limit() {
        let (dir, s) = store();
        s.create(&pubform("one")).unwrap();
        assert_eq!(s.create(&pubform("one")).unwrap_err(), ProxyError::Exists);
        // A hand-made file: listed, never overwritten, edited or removed by the site, and its host never shown.
        let hand = dir.path().join("swarfey-proxy.toml");
        std::fs::write(
            &hand,
            "host = \"203.0.113.9\"\nport = 1\nuser = \"HAND-USER\"\npass = \"HAND-PASS\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&hand, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(s.create(&pubform("swarfey")).unwrap_err(), ProxyError::Exists);
        assert_eq!(s.update(&pubform("swarfey")).unwrap_err(), ProxyError::NotManaged);
        assert_eq!(s.remove("swarfey").unwrap_err(), ProxyError::NotManaged);
        assert!(hand.exists());
        let list = s.list();
        let v = list.iter().find(|v| v.name == "swarfey").unwrap();
        assert!(!v.managed && v.host.is_none() && v.port.is_none() && v.has_credentials && v.usable);
        let json = serde_json::to_string(&list).unwrap();
        assert!(!json.contains("203.0.113.9") && !json.contains("HAND-"), "{json}");
        // A file with another suffix (a backup) is not a profile.
        std::fs::write(dir.path().join("swarfey-proxy.toml.bak"), "x").unwrap();
        std::fs::write(dir.path().join("web-password.txt"), "x").unwrap();
        assert_eq!(s.list().len(), 2);
        // Unknown names.
        assert_eq!(s.update(&pubform("none")).unwrap_err(), ProxyError::NotFound);
        assert_eq!(s.remove("none").unwrap_err(), ProxyError::NotFound);
        assert_eq!(s.remove("../x").unwrap_err(), ProxyError::BadName);
        // Removing the site's own works.
        s.remove("one").unwrap();
        assert!(!dir.path().join("one-proxy.toml").exists());
        // The limit.
        for i in 0..MAX_PROFILES {
            let name = format!("p{i}");
            if s.create(&pubform(&name)).is_err() {
                break;
            }
        }
        assert_eq!(s.create(&pubform("overflow")).unwrap_err(), ProxyError::TooMany);
    }

    #[test]
    fn a_symlinked_or_loose_profile_is_not_followed_or_trusted() {
        let (dir, s) = store();
        let target = dir.path().join("elsewhere.toml");
        std::fs::write(
            &target,
            format!("{MANAGED_KEY} = \"{MANAGED_VALUE}\"\nhost = \"1.2.3.4\"\nport = 1\n"),
        )
        .unwrap();
        std::os::unix::fs::symlink(&target, dir.path().join("link-proxy.toml")).unwrap();
        assert!(!s.exists("link"));
        assert_eq!(s.update(&pubform("link")).unwrap_err(), ProxyError::NotFound);
        assert_eq!(s.create(&pubform("link")).unwrap_err(), ProxyError::Exists);
        let v = s.list().into_iter().find(|v| v.name == "link").unwrap();
        assert!(!v.managed && !v.usable);
        // A loose mode shows as unusable with its reason, and no value.
        s.create(&pubform("loose")).unwrap();
        std::fs::set_permissions(
            dir.path().join("loose-proxy.toml"),
            std::fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        let v = s.list().into_iter().find(|v| v.name == "loose").unwrap();
        assert_eq!((v.usable, v.problem), (false, Some("bad_mode")));
    }

    #[test]
    fn loopback_hosts_are_only_for_tests_that_say_so() {
        let dir = tempfile::tempdir().unwrap();
        let s = ProxyStore::new(dir.path(), Rules { allow_loopback: true });
        let mut f = pubform("lo");
        f.host = "127.0.0.1".into();
        assert!(s.create(&f).is_ok());
        f.name = "lo2".into();
        f.host = "10.0.0.1".into();
        assert_eq!(s.create(&f).unwrap_err(), ProxyError::BadHost);
    }
}
