//! The launcher contract (task 5.9, D-089): the **request** the web writes, the **status** the root helper
//! (`ddnet-ai launch apply`) writes back, and the small file helpers both sides share. Schemas: `docs/formats.md` §34.
//!
//! The web process never gains a privilege: it only writes [`REQUEST_FILE`] (atomically, into its own directory) and reads
//! [`STATUS_FILE`]. A root systemd path unit notices the request and runs the helper, which re-validates **every** value
//! against fixed allow-lists (nothing in the request is ever passed to a command line or trusted as an address).
//!
//! This module is plain data and file helpers: no policy lives here except the *shape* of a request (strict schema, size
//! limit). The policy (allow-lists, rate limits, bans) is the helper's, in `ddnet-ai`'s `launch_cmd`.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The request file the web writes into the launch directory.
pub const REQUEST_FILE: &str = "request.json";
/// The status file the root helper writes into the root-owned status directory ([`DEFAULT_STATUS_DIR`]), not the launch directory.
pub const STATUS_FILE: &str = "status.json";
/// A request is a few dozen bytes; anything bigger than this is refused unread.
pub const MAX_REQUEST_BYTES: usize = 1024;
/// A status file is small too.
pub const MAX_STATUS_BYTES: usize = 8 * 1024;
/// Schema version of the request and the status.
pub const PROTOCOL_VERSION: u32 = 1;
/// The one built-in server choice: the local DDNet server `127.0.0.1:8303`.
pub const LOCAL_SERVER: &str = "local";
/// Most sparring opponents (local server only).
pub const MAX_SPARRING: u8 = 3;
/// At most one start per this many seconds (the helper's rule; the web also rate-limits its own requests).
pub const START_INTERVAL_SECS: u64 = 30;
/// A request older than this (by its own `ts` or by the file's mtime), or dated in the future, is never carried out: the helper
/// refuses and discards it (a request written while the path unit was down must not start a bot when it comes back). The web
/// treats a file this old and still there as «nobody consumes it» and removes it.
pub const REQUEST_STALE_SECS: u64 = 60;
/// How far into the future a request's `ts` or mtime may be (clock rounding) before it is refused.
pub const REQUEST_FUTURE_SLACK_SECS: u64 = 5;
/// The root-owned, world-readable directory the helper writes `status.json` into (the web reads it there). Never inside a
/// directory the owner's user can rename: the root stop hook is unsandboxed and would follow a swapped path.
pub const DEFAULT_STATUS_DIR: &str = "/run/ddnet-ai";
/// The default fly bundle (run `E-005`), under `<data-dir>`; the root-owned config may name another one.
pub const DEFAULT_BUNDLE_REL: &str = "runs/E-005/e005-fly/checkpoints/final.bundle";
/// Root-owned launcher config.
pub const DEFAULT_CONFIG_PATH: &str = "/etc/ddnet-ai/launch.toml";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Start,
    Stop,
}

/// The brains the owner may pick. `Hybrid` = the default, `HybridFly` = hybrid with the fly as proposer, `Fly` = the fly alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Brain {
    #[serde(rename = "hybrid")]
    Hybrid,
    #[serde(rename = "hybrid-fly")]
    HybridFly,
    #[serde(rename = "fly")]
    Fly,
}

impl Brain {
    /// Whether the brain needs the fly bundle.
    pub fn needs_bundle(self) -> bool {
        !matches!(self, Brain::Hybrid)
    }

    /// The value of `ddnet-ai play --brain`.
    pub fn play_brain(self) -> &'static str {
        match self {
            Brain::Hybrid | Brain::HybridFly => "hybrid",
            Brain::Fly => "fly",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DurationChoice {
    #[serde(rename = "15m")]
    M15,
    #[serde(rename = "60m")]
    M60,
    #[serde(rename = "unlimited")]
    Unlimited,
}

impl DurationChoice {
    /// The value of `ddnet-ai play --duration` (`0` = no limit).
    pub fn seconds(self) -> u64 {
        match self {
            DurationChoice::M15 => 15 * 60,
            DurationChoice::M60 => 60 * 60,
            DurationChoice::Unlimited => 0,
        }
    }
}

/// The request file. Unknown fields are refused. `id` ties a status to the request that caused it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchRequest {
    pub v: u32,
    pub id: String,
    /// Unix seconds when the web made the request (see [`REQUEST_STALE_SECS`]).
    pub ts: u64,
    pub action: Action,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brain: Option<Brain>,
    /// `"local"`, or the `address` of an entry of `live-servers.toml` (matched exactly; never connected to as given).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<DurationChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparring: Option<u8>,
}

/// Whether a request made at `ts` (and written to a file last modified at `mtime`), seen at `now`, is fresh: neither older than
/// [`REQUEST_STALE_SECS`] nor dated in the future. Both times must pass, so neither a copied old file with a new mtime nor an old
/// mtime on a new body gets through.
pub fn request_is_fresh(ts: u64, mtime: u64, now: u64) -> bool {
    let ok = |t: u64| t.saturating_add(REQUEST_STALE_SECS) >= now && t <= now.saturating_add(REQUEST_FUTURE_SLACK_SECS);
    ok(ts) && ok(mtime)
}

/// Why a request file is not a request (the code is also the status `reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    #[error("the request is larger than {MAX_REQUEST_BYTES} bytes")]
    TooLarge,
    #[error("the request is not a valid request of this protocol")]
    Invalid,
}

impl ParseError {
    pub fn code(self) -> &'static str {
        match self {
            ParseError::TooLarge => "request_too_large",
            ParseError::Invalid => "bad_request",
        }
    }
}

/// `true` for an id the web makes: 8 to 32 lower-case hex digits.
pub fn valid_id(id: &str) -> bool {
    (8..=32).contains(&id.len()) && id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Parses a request: size limit, UTF-8 JSON, the strict schema, the version, the id, and that a `stop` carries nothing else and
/// a `start` carries `brain`, `server` and `duration`. Values are *not* judged here (that needs the allow-lists).
pub fn parse_request(bytes: &[u8]) -> Result<LaunchRequest, ParseError> {
    if bytes.len() > MAX_REQUEST_BYTES {
        return Err(ParseError::TooLarge);
    }
    let req: LaunchRequest = serde_json::from_slice(bytes).map_err(|_| ParseError::Invalid)?;
    if req.v != PROTOCOL_VERSION || !valid_id(&req.id) {
        return Err(ParseError::Invalid);
    }
    let complete = req.brain.is_some() && req.server.is_some() && req.duration.is_some();
    let empty = req.brain.is_none() && req.server.is_none() && req.duration.is_none() && req.sparring.is_none();
    match req.action {
        Action::Start if !complete => Err(ParseError::Invalid),
        Action::Stop if !empty => Err(ParseError::Invalid),
        _ => Ok(req),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum State {
    /// A start was accepted and `systemctl start` issued (the live bridge says whether the bot is in the game yet).
    Started,
    /// The bot is not running: stopped by the owner, or its time was up.
    Stopped,
    /// The bot exited by itself with an error (`exit_code`: 3 = kicked / banned, 4 = could not join).
    Failed,
    /// A request was refused (`reason`); nothing was changed.
    Refused,
    /// The request was fine but the helper could not carry it out.
    Error,
}

/// What the helper writes for the web (`status.json`). Nothing in it is secret; `server` is the owner's own allow-list address.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchStatus {
    pub v: u32,
    pub state: State,
    /// Unix seconds.
    pub at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// A fixed code (`rate_limited`, `blocked_after_ban`, ...); the web turns it into text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brain: Option<Brain>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<DurationChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparring: Option<u8>,
    /// The fly bundle's run name (`E-005/e005-fly`), never its path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
}

impl LaunchStatus {
    pub fn new(state: State, at: u64) -> LaunchStatus {
        LaunchStatus {
            v: PROTOCOL_VERSION,
            state,
            at,
            request_id: None,
            reason: None,
            brain: None,
            server: None,
            duration: None,
            sparring: None,
            bundle: None,
            exit_code: None,
        }
    }
}

/// The run name of a bundle path for display: `runs/<a>/<b>/...` gives `<a>/<b>`, otherwise the file stem. Never the full path.
pub fn bundle_run_name(path: &Path) -> String {
    let parts: Vec<String> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if let Some(i) = parts.iter().rposition(|p| p == "runs")
        && let (Some(a), Some(b)) = (parts.get(i + 1), parts.get(i + 2))
        && parts.len() > i + 3
    {
        return format!("{a}/{b}");
    }
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The root-owned config (`/etc/ddnet-ai/launch.toml`): `fly_bundle = "<absolute path>"`, nothing else.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchConfig {
    pub fly_bundle: Option<PathBuf>,
}

/// Why a file could not be read as a regular file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ReadError {
    #[error("the file does not exist")]
    Missing,
    #[error("not a regular file (a symlink, a directory or a special file)")]
    NotRegular,
    #[error("the file is too large")]
    TooLarge,
    #[error("the file could not be read")]
    Io,
}

/// Reads a regular file of at most `max` bytes. A symlink is refused (`O_NOFOLLOW`), a FIFO cannot block the reader
/// (`O_NONBLOCK`), and the type is checked on the opened descriptor, so the file checked is the file read.
pub fn read_regular_nofollow(path: &Path, max: usize) -> Result<Vec<u8>, ReadError> {
    read_regular_nofollow_with_mtime(path, max).map(|(bytes, _)| bytes)
}

/// [`read_regular_nofollow`], also giving the file's modification time in unix seconds (from the opened descriptor).
pub fn read_regular_nofollow_with_mtime(path: &Path, max: usize) -> Result<(Vec<u8>, u64), ReadError> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => ReadError::Missing,
            _ if e.raw_os_error() == Some(libc::ELOOP) => ReadError::NotRegular,
            _ => ReadError::Io,
        })?;
    let meta = file.metadata().map_err(|_| ReadError::Io)?;
    if !meta.is_file() {
        return Err(ReadError::NotRegular);
    }
    if meta.len() > max as u64 {
        return Err(ReadError::TooLarge);
    }
    let mtime = u64::try_from(meta.mtime()).unwrap_or(0);
    let mut bytes = Vec::with_capacity(usize::try_from(meta.len()).unwrap_or(0));
    (&mut file)
        .take(max as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| ReadError::Io)?;
    if bytes.len() > max {
        return Err(ReadError::TooLarge);
    }
    Ok((bytes, mtime))
}

/// Writes `dir/name` atomically: a temporary file in the same directory (created exclusively, never through a symlink), then
/// `rename`, which replaces whatever is at `name` (a symlink there is replaced, not followed).
pub fn write_atomic(dir: &Path, name: &str, bytes: &[u8], mode: u32) -> std::io::Result<()> {
    let suffix = crate::rand_util::encode_b64(&crate::rand_util::random_bytes::<9>());
    let tmp = dir.join(format!(".{name}.tmp-{suffix}"));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(mode)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&tmp)?;
        file.write_all(bytes)?;
        // The mode is set explicitly: the creation mode is cut by the umask (the bot unit's stop hook runs under UMask=0077, which
        // made a status unreadable for the web).
        file.set_permissions(std::fs::Permissions::from_mode(mode))?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, dir.join(name))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Unix seconds now.
pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The owner of `path` (its uid), without following a symlink.
pub fn owner_uid(path: &Path) -> Option<u32> {
    std::fs::symlink_metadata(path).ok().map(|m| m.uid())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start_json() -> serde_json::Value {
        serde_json::json!({"v":1,"id":"0123456789abcdef","ts":1000,"action":"start","brain":"hybrid-fly","server":"local","duration":"60m","sparring":2})
    }

    fn parse(v: &serde_json::Value) -> Result<LaunchRequest, ParseError> {
        parse_request(serde_json::to_vec(v).unwrap().as_slice())
    }

    #[test]
    fn a_complete_start_and_a_bare_stop_parse() {
        let r = parse(&start_json()).unwrap();
        assert_eq!(r.action, Action::Start);
        assert_eq!(r.brain, Some(Brain::HybridFly));
        assert_eq!(r.duration, Some(DurationChoice::M60));
        assert_eq!(r.sparring, Some(2));
        let stop = serde_json::json!({"v":1,"id":"0123456789abcdef","ts":1000,"action":"stop"});
        assert_eq!(parse(&stop).unwrap().action, Action::Stop);
    }

    #[test]
    fn unknown_fields_bad_enums_bad_ids_and_wrong_versions_are_refused() {
        let mut v = start_json();
        v["extra"] = serde_json::json!(1);
        assert_eq!(parse(&v), Err(ParseError::Invalid));
        for (k, bad) in [
            ("brain", "planner"),
            ("brain", "Hybrid"),
            ("duration", "2h"),
            ("action", "restart"),
            ("id", "XYZ"),
            ("id", "0123"),
            ("id", "0123456789abcdef0123456789abcdef0"),
        ] {
            let mut v = start_json();
            v[k] = serde_json::json!(bad);
            assert_eq!(parse(&v), Err(ParseError::Invalid), "{k}={bad}");
        }
        let mut v = start_json();
        v["v"] = serde_json::json!(2);
        assert_eq!(parse(&v), Err(ParseError::Invalid));
        // The wrong JSON types.
        let mut v = start_json();
        v["sparring"] = serde_json::json!(-1);
        assert_eq!(parse(&v), Err(ParseError::Invalid));
        let mut v = start_json();
        v["sparring"] = serde_json::json!(300);
        assert_eq!(parse(&v), Err(ParseError::Invalid));
        let mut v = start_json();
        v["server"] = serde_json::json!(5);
        assert_eq!(parse(&v), Err(ParseError::Invalid));
    }

    #[test]
    fn a_start_without_its_fields_and_a_stop_with_any_are_refused() {
        for key in ["brain", "server", "duration"] {
            let mut v = start_json();
            v.as_object_mut().unwrap().remove(key);
            assert_eq!(parse(&v), Err(ParseError::Invalid), "start without {key}");
        }
        let mut stop = serde_json::json!({"v":1,"id":"0123456789abcdef","ts":1000,"action":"stop"});
        stop["server"] = serde_json::json!("local");
        assert_eq!(parse(&stop), Err(ParseError::Invalid));
    }

    #[test]
    fn garbage_empty_and_oversized_input_is_refused_without_panicking() {
        for bytes in [&b""[..], b"null", b"[]", b"{", b"\xff\xfe", b"{\"v\":1}"] {
            assert_eq!(parse_request(bytes), Err(ParseError::Invalid), "{bytes:?}");
        }
        let big = vec![b' '; MAX_REQUEST_BYTES + 1];
        assert_eq!(parse_request(&big), Err(ParseError::TooLarge));
        let mut padded = serde_json::to_vec(&start_json()).unwrap();
        padded.extend(std::iter::repeat_n(b' ', MAX_REQUEST_BYTES));
        assert_eq!(parse_request(&padded), Err(ParseError::TooLarge));
    }

    #[test]
    fn a_request_must_be_fresh_by_its_ts_and_by_its_mtime() {
        let now = 1_000_000;
        assert!(request_is_fresh(now, now, now));
        assert!(request_is_fresh(now - 60, now - 60, now));
        assert!(request_is_fresh(now + 5, now + 5, now));
        assert!(!request_is_fresh(now - 61, now, now), "old ts");
        assert!(!request_is_fresh(now, now - 61, now), "old mtime");
        assert!(!request_is_fresh(now - 2 * 86400, now - 2 * 86400, now));
        assert!(!request_is_fresh(now + 6, now, now), "future ts");
        assert!(!request_is_fresh(now, now + 6, now), "future mtime");
        assert!(!request_is_fresh(0, 0, now));
    }

    #[test]
    fn the_bundle_is_shown_by_run_name_never_by_path() {
        assert_eq!(
            bundle_run_name(Path::new(
                "/home/u/aiddnet/data/runs/E-005/e005-fly/checkpoints/final.bundle"
            )),
            "E-005/e005-fly"
        );
        assert_eq!(bundle_run_name(Path::new("/opt/x/final.bundle")), "final");
    }

    #[test]
    fn durations_and_brains_map_to_the_play_flags() {
        assert_eq!(DurationChoice::M15.seconds(), 900);
        assert_eq!(DurationChoice::M60.seconds(), 3600);
        assert_eq!(DurationChoice::Unlimited.seconds(), 0);
        assert_eq!(Brain::Hybrid.play_brain(), "hybrid");
        assert_eq!(Brain::HybridFly.play_brain(), "hybrid");
        assert_eq!(Brain::Fly.play_brain(), "fly");
        assert!(!Brain::Hybrid.needs_bundle() && Brain::HybridFly.needs_bundle() && Brain::Fly.needs_bundle());
    }

    #[test]
    fn files_are_read_only_when_regular_small_and_not_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.json");
        std::fs::write(&file, b"hello").unwrap();
        assert_eq!(read_regular_nofollow(&file, 16).unwrap(), b"hello");
        assert_eq!(read_regular_nofollow(&file, 4), Err(ReadError::TooLarge));
        assert_eq!(
            read_regular_nofollow(&dir.path().join("none"), 16),
            Err(ReadError::Missing)
        );
        let link = dir.path().join("link.json");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        assert_eq!(read_regular_nofollow(&link, 16), Err(ReadError::NotRegular));
        assert_eq!(read_regular_nofollow(dir.path(), 16), Err(ReadError::NotRegular));
    }

    #[test]
    fn an_atomic_write_replaces_a_symlink_instead_of_following_it() {
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim");
        std::fs::write(&victim, b"keep").unwrap();
        let link = dir.path().join("status.json");
        std::os::unix::fs::symlink(&victim, &link).unwrap();
        write_atomic(dir.path(), "status.json", b"new", 0o644).unwrap();
        assert_eq!(std::fs::metadata(&link).unwrap().permissions().mode() & 0o777, 0o644);
        assert_eq!(std::fs::read(&victim).unwrap(), b"keep");
        assert_eq!(std::fs::read(&link).unwrap(), b"new");
        assert!(!std::fs::symlink_metadata(&link).unwrap().file_type().is_symlink());
        // No temporary file is left behind.
        let names: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(names.iter().all(|n| !n.contains(".tmp-")), "{names:?}");
    }
}
