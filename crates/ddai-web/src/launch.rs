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

/// The opponent-input model of the launch card's «Предсказатель соперника (эксперимент)» (task 3.17, D-111), relative to `<data-dir>`. The file is
/// never in git; the lead copies it there. The toggle names no path: the helper builds this one and refuses (`window_model_missing`) when it is not a
/// plain regular file, so a request can never point the bot at anything else.
pub const DEFAULT_WINDOW_MODEL_REL: &str = "bot/models/opp-m1.oppnet";
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

/// The hybrid's opponent model switch (task 3.7b, D-090): a closed list, so the helper writes only `on` or `off` to the unit's environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mirror {
    #[serde(rename = "on")]
    On,
    #[serde(rename = "off")]
    Off,
}

impl Mirror {
    /// The value of `ddnet-ai play --hybrid-mirror`.
    pub fn flag_value(self) -> &'static str {
        match self {
            Mirror::On => "on",
            Mirror::Off => "off",
        }
    }
}

/// The finishing switch (task 5.13, D-097; `ddnet-ai play --finish`): a closed list, so the helper writes only one of four words to the
/// unit's environment. `Off` is the default and what a request without the field means.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Finish {
    /// No finishing (the bot as before 3.10).
    #[default]
    #[serde(rename = "off")]
    Off,
    /// The bot keeps a frozen target until it is held, sealed or dead (the recommended live A/B).
    #[serde(rename = "target")]
    Target,
    /// `target` plus the wayblock hold (task 3.18, D-114): inside the held hall the hybrid throws a frozen victim toward the hall's freeze wall.
    /// Opt-in and an experiment (the arena gain stayed below the bar announced in advance); for wayblock play, not the duel.
    #[serde(rename = "wb")]
    Wb,
    /// `target` plus the hybrid's drag shaping for a frozen victim (did not hold up in review: not recommended).
    #[serde(rename = "full")]
    Full,
}

impl Finish {
    /// The value of `ddnet-ai play --finish`.
    pub fn flag_value(self) -> &'static str {
        match self {
            Finish::Off => "off",
            Finish::Target => "target",
            Finish::Wb => "wb",
            Finish::Full => "full",
        }
    }

    /// Whether this is not `off` (the pure fly brain refuses such a request).
    pub fn is_on(self) -> bool {
        self != Finish::Off
    }
}

/// The smart wayblock switch (task 5.15; `ddnet-ai play --wb-smart`, tasks 3.12/3.12b, D-103, D-104): a closed list of two words, so the helper
/// writes only `on` or `off` to the unit's environment. `Off` is the default and what a request without the field means. It is allowed with
/// every brain: it changes the bot's navigation and target choice (the wayblock side, the AFK rule, the tube crossings), which all three brains
/// run on, not the brain's own decision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WbSmart {
    #[default]
    #[serde(rename = "off")]
    Off,
    #[serde(rename = "on")]
    On,
}

impl WbSmart {
    /// The value of `ddnet-ai play --wb-smart`.
    pub fn flag_value(self) -> &'static str {
        match self {
            WbSmart::Off => "off",
            WbSmart::On => "on",
        }
    }
}

/// The value of `ddnet-ai play --no-selfkill=<value>` (the duel switch, task 5.15, D-102) for the request's `no_selfkill` flag: a boolean flag
/// cannot take an empty argument from the unit's environment, so the unit passes the one-argument form `--no-selfkill=${BOT_NO_SELFKILL}`.
pub fn no_selfkill_flag_value(on: bool) -> &'static str {
    if on { "true" } else { "false" }
}

/// The value of `ddnet-ai play --preinput <value>` (the server's pre-inputs in the prediction, task 3.20b, D-112) for the request's `preinput`
/// flag: always `on` or `off`, so the unit passes it as the two-word form `--preinput ${BOT_PREINPUT}` like `--wb-smart`.
pub fn preinput_flag_value(on: bool) -> &'static str {
    if on { "on" } else { "off" }
}

/// The hybrid's search threads (task 5.17, D-125; `ddnet-ai play --search-threads`): a closed list of the whole numbers 1 to 4. A JSON number
/// outside the list, a float, a string or a boolean is no request at all (the parse fails), so the helper only ever holds one of four values and
/// writes one of four static words to the unit's environment. 1 is the default and what a request without the field means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct SearchThreads(u8);

impl SearchThreads {
    /// The one-thread default (the bot's own default, D-080).
    pub const ONE: SearchThreads = SearchThreads(1);
    /// The most the list allows (the cores beyond the fourth buy little: E-012, D-123).
    pub const MAX: u8 = 4;

    /// `Some` for 1 to 4.
    pub fn new(n: u8) -> Option<SearchThreads> {
        (1..=Self::MAX).contains(&n).then_some(SearchThreads(n))
    }

    pub fn get(self) -> u8 {
        self.0
    }

    /// The value of `ddnet-ai play --search-threads`: one of four fixed words, never a formatted number.
    pub fn flag_value(self) -> &'static str {
        match self.0 {
            1 => "1",
            2 => "2",
            3 => "3",
            _ => "4",
        }
    }

    /// Whether this is more than the one-thread default (the pure fly brain, which does not search, refuses such a request).
    pub fn is_more_than_one(self) -> bool {
        self.0 > 1
    }
}

impl Default for SearchThreads {
    fn default() -> SearchThreads {
        SearchThreads::ONE
    }
}

impl TryFrom<u8> for SearchThreads {
    type Error = String;

    fn try_from(n: u8) -> Result<SearchThreads, String> {
        SearchThreads::new(n).ok_or_else(|| format!("search_threads must be 1 to {}", SearchThreads::MAX))
    }
}

impl From<SearchThreads> for u8 {
    fn from(t: SearchThreads) -> u8 {
        t.0
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
    /// The hybrid brains' opponent model (`on` when absent, D-090); other brains ignore it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirror: Option<Mirror>,
    /// The finishing mode (task 5.13, D-097); `off` when absent. Not allowed with the pure fly brain unless it is `off`
    /// (the helper refuses `finish_hybrid_only`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish: Option<Finish>,
    /// The smart wayblock (task 5.15, D-103/D-104); `off` when absent. Allowed with every brain (it is navigation and target logic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wb_smart: Option<WbSmart>,
    /// The duel switch «the bot never kills itself» (task 5.15, D-102); `false` when absent. A JSON boolean, nothing else. Allowed with
    /// every brain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_selfkill: Option<bool>,
    /// The opponent-input predictor (task 3.17, D-111; `ddnet-ai play --window-model`): a JSON boolean, `false` when absent. Only the hybrid
    /// brains take it (the model belongs to the hybrid's lag window; the helper refuses `window_model_hybrid_only` for the pure fly). It names no
    /// file: the helper uses [`DEFAULT_WINDOW_MODEL_REL`] under the data directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_model: Option<bool>,
    /// The server's pre-inputs played in the prediction (task 3.20b, D-112; `ddnet-ai play --preinput on`): a JSON boolean, `false` when absent.
    /// Only the hybrid brains take it (the helper refuses `preinput_hybrid_only` for the pure fly). It names no file and no value but the boolean:
    /// the helper writes `on` or `off` to the unit's environment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preinput: Option<bool>,
    /// The hybrid's search threads (task 5.17, D-125; `ddnet-ai play --search-threads`): a JSON integer from the closed list 1 to 4, `1` when absent.
    /// The pure fly does not search, so the helper refuses `search_threads_hybrid_only` for the fly with more than 1 (`1` is the same as absent).
    /// It names no file and no word: the helper writes one of four static digits to the unit's environment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_threads: Option<SearchThreads>,
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
    let empty = req.brain.is_none()
        && req.server.is_none()
        && req.duration.is_none()
        && req.sparring.is_none()
        && req.mirror.is_none()
        && req.finish.is_none()
        && req.wb_smart.is_none()
        && req.no_selfkill.is_none()
        && req.window_model.is_none()
        && req.preinput.is_none()
        && req.search_threads.is_none();
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
    /// The finishing mode of the launch (task 5.13); none in the status of a launch made before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish: Option<Finish>,
    /// The smart wayblock of the launch (task 5.15); none in the status of a launch made before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wb_smart: Option<WbSmart>,
    /// The duel switch of the launch (task 5.15); none in the status of a launch made before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_selfkill: Option<bool>,
    /// The opponent-input predictor of the launch (task 3.17); none in the status of a launch made before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_model: Option<bool>,
    /// The server's pre-inputs played in the prediction, for the launch (task 3.20b); none in the status of a launch made before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preinput: Option<bool>,
    /// The hybrid's search threads of the launch (task 5.17); none in the status of a launch made before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_threads: Option<SearchThreads>,
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
            finish: None,
            wb_smart: None,
            no_selfkill: None,
            window_model: None,
            preinput: None,
            search_threads: None,
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
        assert_eq!(r.mirror, None, "the opponent model is on when the request says nothing");
        let mut v = start_json();
        v["mirror"] = serde_json::json!("off");
        assert_eq!(parse(&v).unwrap().mirror, Some(Mirror::Off));
        let stop = serde_json::json!({"v":1,"id":"0123456789abcdef","ts":1000,"action":"stop"});
        assert_eq!(parse(&stop).unwrap().action, Action::Stop);
    }

    #[test]
    fn the_finishing_field_is_additive_a_closed_list_and_absent_means_off() {
        // An old request (no field) parses and means off.
        let old = parse(&start_json()).unwrap();
        assert_eq!(old.finish, None);
        assert_eq!(old.finish.unwrap_or_default(), Finish::Off);
        for (word, want, flag) in [
            ("off", Finish::Off, "off"),
            ("target", Finish::Target, "target"),
            ("wb", Finish::Wb, "wb"),
            ("full", Finish::Full, "full"),
        ] {
            let mut v = start_json();
            v["finish"] = serde_json::json!(word);
            let r = parse(&v).unwrap();
            assert_eq!(r.finish, Some(want), "{word}");
            assert_eq!(want.flag_value(), flag);
            // It survives the way the web writes it (a request round trip).
            let back = parse_request(&serde_json::to_vec(&r).unwrap()).unwrap();
            assert_eq!(back, r);
        }
        assert!(!Finish::Off.is_on() && Finish::Target.is_on() && Finish::Wb.is_on() && Finish::Full.is_on());
        // An absent field is not written (old helpers' strict schema never sees an unknown key from an unchanged request).
        let text = String::from_utf8(serde_json::to_vec(&old).unwrap()).unwrap();
        assert!(!text.contains("finish"), "{text}");
        // Nothing but the four words, in lower case: no aliases, no `on`, no injection.
        for bad in [
            serde_json::json!("on"),
            serde_json::json!("WB"),
            serde_json::json!("Wb"),
            serde_json::json!("wb "),
            serde_json::json!("wb\n"),
            serde_json::json!("wb --wb left"),
            serde_json::json!("target,wb"),
            serde_json::json!("Target"),
            serde_json::json!("TARGET"),
            serde_json::json!("target "),
            serde_json::json!(" target"),
            serde_json::json!("target\n"),
            serde_json::json!("target --report /etc/passwd"),
            serde_json::json!("target\"\nBOT_SERVER=\"1.2.3.4:5\""),
            serde_json::json!("$(id)"),
            serde_json::json!(""),
            serde_json::json!(true),
            serde_json::json!(1),
            serde_json::json!(["target"]),
            serde_json::json!({"mode": "target"}),
        ] {
            let mut v = start_json();
            v["finish"] = bad.clone();
            assert_eq!(parse(&v), Err(ParseError::Invalid), "finish={bad}");
        }
        // A stop carries nothing, finishing included.
        let mut stop = serde_json::json!({"v":1,"id":"0123456789abcdef","ts":1000,"action":"stop"});
        stop["finish"] = serde_json::json!("off");
        assert_eq!(parse(&stop), Err(ParseError::Invalid));
    }

    #[test]
    fn the_wb_smart_and_no_selfkill_fields_are_additive_closed_and_absent_means_off() {
        // Task 5.15. An old request (no fields) parses and means off / false.
        let old = parse(&start_json()).unwrap();
        assert_eq!((old.wb_smart, old.no_selfkill), (None, None));
        assert_eq!(old.wb_smart.unwrap_or_default(), WbSmart::Off);
        assert!(!old.no_selfkill.unwrap_or_default());
        // An absent field is not written (an old helper's strict schema never sees an unknown key from an unchanged request).
        let text = String::from_utf8(serde_json::to_vec(&old).unwrap()).unwrap();
        assert!(!text.contains("wb_smart") && !text.contains("no_selfkill"), "{text}");
        for (word, want) in [("off", WbSmart::Off), ("on", WbSmart::On)] {
            let mut v = start_json();
            v["wb_smart"] = serde_json::json!(word);
            let r = parse(&v).unwrap();
            assert_eq!(r.wb_smart, Some(want), "{word}");
            assert_eq!(want.flag_value(), word);
            assert_eq!(
                parse_request(&serde_json::to_vec(&r).unwrap()).unwrap(),
                r,
                "round trip"
            );
        }
        for want in [false, true] {
            let mut v = start_json();
            v["no_selfkill"] = serde_json::json!(want);
            let r = parse(&v).unwrap();
            assert_eq!(r.no_selfkill, Some(want));
            assert_eq!(
                parse_request(&serde_json::to_vec(&r).unwrap()).unwrap(),
                r,
                "round trip"
            );
            assert_eq!(no_selfkill_flag_value(want), if want { "true" } else { "false" });
        }
        // Closed values: nothing but `off|on` (lower case) and a JSON boolean; no strings, numbers, injection.
        for bad in [
            serde_json::json!("true"),
            serde_json::json!("yes"),
            serde_json::json!("On"),
            serde_json::json!("ON"),
            serde_json::json!("on "),
            serde_json::json!("on\n"),
            serde_json::json!("on --report /etc/passwd"),
            serde_json::json!("on\"\nBOT_SERVER=\"1.2.3.4:5\""),
            serde_json::json!("$(id)"),
            serde_json::json!(""),
            serde_json::json!(true),
            serde_json::json!(1),
            serde_json::json!(["on"]),
        ] {
            let mut v = start_json();
            v["wb_smart"] = bad.clone();
            assert_eq!(parse(&v), Err(ParseError::Invalid), "wb_smart={bad}");
        }
        for bad in [
            serde_json::json!("true"),
            serde_json::json!("false"),
            serde_json::json!("on"),
            serde_json::json!("true --x"),
            serde_json::json!("$(id)"),
            serde_json::json!(""),
            serde_json::json!(0),
            serde_json::json!(1),
            serde_json::json!([true]),
            serde_json::json!({"on": true}),
        ] {
            let mut v = start_json();
            v["no_selfkill"] = bad.clone();
            assert_eq!(parse(&v), Err(ParseError::Invalid), "no_selfkill={bad}");
        }
        // A stop carries nothing, these two included.
        for (key, val) in [
            ("wb_smart", serde_json::json!("off")),
            ("no_selfkill", serde_json::json!(false)),
        ] {
            let mut stop = serde_json::json!({"v":1,"id":"0123456789abcdef","ts":1000,"action":"stop"});
            stop[key] = val;
            assert_eq!(parse(&stop), Err(ParseError::Invalid), "{key}");
        }
    }

    #[test]
    fn the_window_model_field_is_an_additive_json_boolean_and_absent_means_off() {
        // Task 3.17 (D-111). An old request parses and means off; an absent field is not written.
        let old = parse(&start_json()).unwrap();
        assert_eq!(old.window_model, None);
        assert!(!old.window_model.unwrap_or_default());
        let text = String::from_utf8(serde_json::to_vec(&old).unwrap()).unwrap();
        assert!(!text.contains("window_model"), "{text}");
        for want in [false, true] {
            let mut v = start_json();
            v["window_model"] = serde_json::json!(want);
            let r = parse(&v).unwrap();
            assert_eq!(r.window_model, Some(want));
            assert_eq!(
                parse_request(&serde_json::to_vec(&r).unwrap()).unwrap(),
                r,
                "round trip"
            );
        }
        // Nothing but a JSON boolean: no path, no string, no number.
        for bad in [
            serde_json::json!("true"),
            serde_json::json!("/etc/passwd"),
            serde_json::json!("on"),
            serde_json::json!("~/aiddnet/data/bot/models/opp-m1.oppnet"),
            serde_json::json!(""),
            serde_json::json!(0),
            serde_json::json!(1),
            serde_json::json!(["x"]),
            serde_json::json!({"path": "x"}),
        ] {
            let mut v = start_json();
            v["window_model"] = bad.clone();
            assert_eq!(parse(&v), Err(ParseError::Invalid), "window_model={bad}");
        }
        // A stop carries nothing.
        let mut stop = serde_json::json!({"v":1,"id":"0123456789abcdef","ts":1000,"action":"stop"});
        stop["window_model"] = serde_json::json!(false);
        assert_eq!(parse(&stop), Err(ParseError::Invalid));
    }

    #[test]
    fn the_preinput_field_is_an_additive_json_boolean_and_absent_means_off() {
        // Task 3.20b (D-112). An old request parses and means off; an absent field is not written.
        let old = parse(&start_json()).unwrap();
        assert_eq!(old.preinput, None);
        assert!(!old.preinput.unwrap_or_default());
        let text = String::from_utf8(serde_json::to_vec(&old).unwrap()).unwrap();
        assert!(!text.contains("preinput"), "{text}");
        for want in [false, true] {
            let mut v = start_json();
            v["preinput"] = serde_json::json!(want);
            let r = parse(&v).unwrap();
            assert_eq!(r.preinput, Some(want));
            assert_eq!(
                parse_request(&serde_json::to_vec(&r).unwrap()).unwrap(),
                r,
                "round trip"
            );
            assert_eq!(preinput_flag_value(want), if want { "on" } else { "off" });
        }
        // Nothing but a JSON boolean: no word, no path, no number.
        for bad in [
            serde_json::json!("true"),
            serde_json::json!("on"),
            serde_json::json!("off"),
            serde_json::json!("on --report /etc/passwd"),
            serde_json::json!("on\"\nBOT_SERVER=\"1.2.3.4:5\""),
            serde_json::json!("/etc/passwd"),
            serde_json::json!(""),
            serde_json::json!(0),
            serde_json::json!(1),
            serde_json::json!(["x"]),
            serde_json::json!({"on": true}),
        ] {
            let mut v = start_json();
            v["preinput"] = bad.clone();
            assert_eq!(parse(&v), Err(ParseError::Invalid), "preinput={bad}");
        }
        // A stop carries nothing.
        let mut stop = serde_json::json!({"v":1,"id":"0123456789abcdef","ts":1000,"action":"stop"});
        stop["preinput"] = serde_json::json!(false);
        assert_eq!(parse(&stop), Err(ParseError::Invalid));
    }

    #[test]
    fn the_search_threads_field_is_an_additive_closed_list_of_one_to_four_and_absent_means_one() {
        // Task 5.17 (D-125). An old request parses and means one thread; an absent field is not written (byte-identical).
        let old = parse(&start_json()).unwrap();
        assert_eq!(old.search_threads, None);
        assert_eq!(old.search_threads.unwrap_or_default(), SearchThreads::ONE);
        let text = String::from_utf8(serde_json::to_vec(&old).unwrap()).unwrap();
        assert!(!text.contains("search_threads"), "{text}");
        for n in 1..=4u8 {
            let mut v = start_json();
            v["search_threads"] = serde_json::json!(n);
            let r = parse(&v).unwrap();
            assert_eq!(r.search_threads.map(SearchThreads::get), Some(n));
            assert_eq!(r.search_threads.unwrap().flag_value(), n.to_string());
            let bytes = serde_json::to_vec(&r).unwrap();
            assert!(
                String::from_utf8(bytes.clone())
                    .unwrap()
                    .contains(&format!(r#""search_threads":{n}"#))
            );
            assert_eq!(parse_request(&bytes).unwrap(), r, "round trip");
        }
        assert!(
            SearchThreads::new(0).is_none() && SearchThreads::new(5).is_none() && SearchThreads::new(255).is_none()
        );
        assert!(!SearchThreads::ONE.is_more_than_one() && SearchThreads::new(2).unwrap().is_more_than_one());
        // Nothing but the whole numbers 1 to 4: no zero, no five, no negative, no float, no word, no boolean, no injection.
        for bad in [
            serde_json::json!(0),
            serde_json::json!(5),
            serde_json::json!(8),
            serde_json::json!(16),
            serde_json::json!(255),
            serde_json::json!(256),
            serde_json::json!(-1),
            serde_json::json!(2.0),
            serde_json::json!(2.5),
            serde_json::json!(1e0),
            serde_json::json!("2"),
            serde_json::json!("auto"),
            serde_json::json!("AUTO"),
            serde_json::json!("2 --report /etc/passwd"),
            serde_json::json!("2\"\nBOT_SERVER=\"1.2.3.4:5\""),
            serde_json::json!("$(id)"),
            serde_json::json!(""),
            serde_json::json!(true),
            serde_json::json!(false),
            serde_json::json!([2]),
            serde_json::json!({"n": 2}),
        ] {
            let mut v = start_json();
            v["search_threads"] = bad.clone();
            assert_eq!(parse(&v), Err(ParseError::Invalid), "search_threads={bad}");
        }
        // A stop carries nothing.
        let mut stop = serde_json::json!({"v":1,"id":"0123456789abcdef","ts":1000,"action":"stop"});
        stop["search_threads"] = serde_json::json!(1);
        assert_eq!(parse(&stop), Err(ParseError::Invalid));
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
            ("mirror", "maybe"),
            ("mirror", "ON"),
            ("mirror", "true"),
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
        let mut stop = serde_json::json!({"v":1,"id":"0123456789abcdef","ts":1000,"action":"stop"});
        stop["mirror"] = serde_json::json!("off");
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
