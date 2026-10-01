// Ported from DDNet `src/engine/client/client.cpp` and `src/game/client/gameclient.cpp` (pinned
// rev c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"), which carry the original Teeworlds
// zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, adapted to a sans-IO shape. Exact
// file:line citations are on the individual methods below; see the module docs for the overall
// join-sequence shape and what is/isn't ported.
//
//! The sans-IO client session — task 2.3 acceptance criteria 1-4, 6.
//!
//! [`Session`] owns exactly one [`ddai_net::conn::Connection`] (one peer) plus everything above
//! it needed to actually *play*: the join sequence, snapshot assembly
//! ([`ddai_net::assembly::SnapAssembler`]), input timing ([`crate::timing::InputTiming`]), and the
//! guarded single outgoing path for game messages ([`crate::allowlist`]). Like `Connection`
//! itself, it does no socket I/O and has no wall-clock access — every method that cares about
//! time takes an explicit `now: Duration`, and [`Session::flush`] *returns* datagrams to send
//! rather than writing to a socket. [`crate::driver`] is the real-time layer that owns the actual
//! `UdpSocket` and drives this type.
//!
//! # Map acquisition and the one deliberately-synchronous exception to "sans-IO"
//!
//! [`Session`] never touches a filesystem or does HTTP — but "was this map already downloaded"
//! genuinely needs the driver to check a cache file. The handshake this crate uses: when
//! [`Session::feed`] processes `NETMSG_MAP_CHANGE`, it does **not** immediately start the
//! in-protocol download — it returns [`SessionEvent::MapChanging`] and moves to a "waiting for
//! map bytes" state that sends nothing yet. The driver gets exactly one synchronous window,
//! between that `feed()` call returning and its own next [`Session::flush`] call, to read the
//! cache file (if `sha256` is known) and hand the bytes back via [`Session::supply_cached_map`].
//! Only if that window passes without a call does [`Session::flush`] start the real
//! `NETMSG_REQUEST_MAP_DATA` download. Since the driver loop is single-threaded and processes one
//! event batch at a time, this needs no extra synchronisation — see `crate::driver::run` for the
//! caller side. HTTPS map download (acceptance criterion 2's other half, `MAP_DETAILS.url`) is
//! surfaced the same way (`url` on [`SessionEvent::MapChanging`]) but never acted on by this
//! project's own driver — see the crate root docs for why (live-play policy: 127.0.0.1 only).
//!
//! # Join sequence (file:line citations)
//!
//! 1. `Connection::connect` (`ddai_net::conn`) — TKEN handshake, [`ddai_net::conn::Event::Connected`].
//! 2. On [`ddai_net::conn::Event::Connected`]: `NETMSG_CLIENTVER` then `NETMSG_INFO`
//!    (`client.cpp:240-259`, `SendInfo`) — [`Session::on_connected`].
//! 3. Server sends `capabilities@ddnet.tw` (optional, first connection only —
//!    `client.cpp:1701-1716`), `map-details@ddnet.tw` (optional, `client.cpp:1676-1700`), then
//!    `NETMSG_MAP_CHANGE` (`client.cpp:1717-1803`) — [`Session::handle_map_change`].
//! 4. Map acquisition (see above) — [`Session::begin_protocol_download`],
//!    [`Session::handle_map_data`], [`Session::supply_cached_map`], both funnelling into
//!    [`Session::finish_map_load`] (`client.cpp:1229-1291`, `LoadMap`).
//! 5. On success: `NETMSG_READY` (`client.cpp:270-274`, `SendReady`).
//! 6. Server sends `NETMSG_CON_READY` (`client.cpp:1896-1915`; server side `server.cpp:2119-2149`,
//!    `OnNetMsgReady`/`SendConnectionReady`) — [`Session::handle_con_ready`] sends `Cl_StartInfo`
//!    (`gameclient.cpp:3224-3248`, `CGameClient::SendInfo(true)`, called from `OnConnected`,
//!    `gameclient.cpp:574-611`).
//! 7. Server sends `Sv_ReadyToEnter` (`gamecontext.cpp`/`server.cpp:2142-2143`) —
//!    [`Session::handle_ready_to_enter`] sends `NETMSG_ENTERGAME` (`client.cpp:264-268`,
//!    `SendEnterGame`) and resets [`ddai_net::assembly::SnapAssembler`]/[`crate::timing::InputTiming`]
//!    (`client.cpp:472-512`, `CClient::OnEnterGame` — task 2.2b's F1 finding: this reset is
//!    mandatory on *every* `ENTERGAME`, not just the first).
//! 8. Snapshots flow; once the first one carrying our own (`PlayerInfo::local == 1`) player
//!    arrives, `Cl_IsDDNetLegacy`/`Cl_ShowDistance`/`Cl_ShowOthers`/`Cl_EnableSpectatorCount`/
//!    `Cl_CameraInfo` are sent once (`gameclient.cpp:2289-2340,2392-2420` — see
//!    [`Session::send_post_enter_extras`] for the hand-decoded `Cl_IsDDNetLegacy` payload, task
//!    2.2b's generated struct is empty by mistake — see that method's doc comment, and for why
//!    the latter three carry fixed default-config values rather than tracked camera/HUD state).
//! 9. Also once in-game: periodic `PINGEX` origination (`client.cpp:527,2982-3004`), 0.5s after
//!    entering then every 10 minutes, when the server capability is present — see
//!    [`Session::maybe_send_ping_ex`].

use crate::allowlist;
use crate::live_servers;
use crate::map_cache;
use crate::timing::{InputTiming, MarginSummary};
use ddai_net::assembly::{self, SnapAssembler};
use ddai_net::conn::{self, Connection};
use ddai_net::delta::StaticSizes;
use ddai_net::generated::{enums::playerflagflag, messages as msgs, objects};
use ddai_net::huffman::Huffman;
use ddai_net::message::{self, ExSysMsg, Msg, Registry};
use ddai_net::packer::Packer;
use ddai_net::sysmsg::{self, SysMsg};
use ddai_net::tuning::{DEFAULT_TUNE_PARAMS, TeamsState, TuneParams};
use ddai_net::uuid::{self, MsgId};
use ddai_net::view::View;
use std::collections::VecDeque;
use std::time::Duration;

/// `NETMSGTYPE_CL_SAY = 17` = 0.6+DDNet's chat message id — this crate has *no* function that
/// builds a payload for it (decision D-007). Referenced only in doc comments/tests, not code.
pub use ddai_net::generated::messages::id::NETMSGTYPE_CL_SAY as CL_SAY_ID_FOR_DOCS_ONLY;

/// A hard cap on total downloaded/cached map bytes, independent of whatever `NETMSG_MAP_CHANGE`'s
/// `size` field (peer-controlled) claims — task acceptance criterion 2/7's "size limits against
/// hostile servers". DDNet's own client caps at 1 GiB (`client.cpp:1734`); every block map this
/// bot will ever actually play is at most a few MiB, so this is deliberately much tighter.
pub const DEFAULT_MAX_MAP_SIZE_BYTES: usize = 64 * 1024 * 1024;

/// Review finding F10: below this, a claimed map size is surfaced as [`SessionEvent::Anomaly`]
/// rather than trusted silently — a real block map (even a tiny one) has a version item, an info
/// item, at least one group/layer header and some tile data, which in practice never fits under a
/// couple hundred bytes; anything smaller is far more likely a placeholder/lobby/captcha map than
/// a real one. Deliberately well under any real map (including this file's own `tiny_map_bytes`
/// test fixture) so it only fires on the genuinely implausible case.
const MIN_PLAUSIBLE_MAP_SIZE_BYTES: usize = 256;

/// `client.cpp:527`: `time_freq() / 2` after `EnterGame` — review finding F9.
const PING_EX_INITIAL_DELAY_NS: i64 = 500_000_000;
/// `client.cpp:3004`: `600 * Freq` (10 minutes) between subsequent origination pings.
const PING_EX_INTERVAL_NS: i64 = 600 * 1_000_000_000;

/// `GAME_NETVERSION` (`game/version.h:20`) — fixed for every 0.6+DDNet client; not configurable
/// (a different value would simply fail to match on any real server).
const NETVERSION_STRING: &str = "0.6 626fce9a778df4d4";

/// The real client's own `NETMSG_CLIENTVER` string format: `"GAME_NAME GAME_RELEASE_VERSION"`
/// (`gameclient.cpp:354`, the no-git-hash branch — this bot has no build-time git hash of its own
/// to append truthfully). Review finding F6: previously a custom, obviously-non-standard
/// "DDNet-AI/0.1 (research bot; ...)" string. This field carries no server-side gating function
/// in vanilla DDNet — confirmed by reading the engine source: `server.cpp:2062` only ever
/// `str_copy`s it into a per-client info struct for display (`server.cpp:725`, e.g. a server's
/// player list), it is never compared/branched on to accept or reject a connection anywhere in
/// this codebase — so this change is about being a well-behaved, protocol-conforming client (the
/// same string any unmodified DDNet 20.1 install sends), not about defeating any detection
/// mechanism (there is none here to defeat). Still fully configurable per `ClientConfig`, and the
/// bot's actual identity/behaviour is never hidden: it never fakes chat, obeys the connection-limit
/// and no-auto-reconnect-after-kick/ban policy, and every session is logged (`CLAUDE.md`'s live-play
/// rules) regardless of what this string says.
const DEFAULT_VERSION_STR: &str = "DDNet 20.1";

/// Everything [`Session::connect`] needs to know about who we are — task acceptance criterion 1's
/// `ClientConfig`.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub name: String,
    pub clan: String,
    pub skin: String,
    pub country: i32,
    pub password: String,
    /// `Cl_ShowDistance`'s `(x, y)` half-extents, in world (game) units — sent once right after
    /// entering, in place of the old TS bot's `/showall` chat command (never sent — see the crate
    /// root docs). Default matches the task spec's suggestion of "large, e.g. 3000×2000".
    pub show_distance: (i32, i32),
    /// `NETMSG_CLIENTVER`'s version int — `DDNET_VERSION_NUMBER` is `20010` for real DDNet 20.1
    /// (`game/version.h:8`); configurable per the task spec.
    pub ddnet_version: i32,
    /// `NETMSG_CLIENTVER`'s version string.
    pub version_str: String,
    /// `cl_prediction_margin`-equivalent, milliseconds — see `crate::timing`.
    pub prediction_margin_ms: i32,
    /// See [`DEFAULT_MAX_MAP_SIZE_BYTES`].
    pub max_map_size_bytes: usize,
    /// [`ddai_net::conn::Connection`]'s silence timeout — defaults to the real
    /// [`ddai_net::conn::DEFAULT_TIMEOUT`] (100s); tests/tooling that want faster timeout
    /// detection (e.g. task 2.3's server-restart scenario) may override it.
    pub timeout: Duration,
    /// Task 2.3b's handshake watchdog: if [`crate::driver::Client`] has not reached
    /// [`SessionEvent::InGame`] within this long after it first started connecting, it disconnects
    /// gracefully (`CLOSE`) and gives up (`ClientEvent::GaveUp { category:
    /// GaveUpCategory::HandshakeTimeout, .. }`) rather than retrying silently forever. This is a
    /// self-protection measure, not a per-attempt timeout: it accumulates across *any* number of
    /// reconnects/redirects/lost-connection retries within the same logical session (mirroring the
    /// existing `reached_in_game`-driven backoff reset — see `crate::driver::run`), and only resets
    /// once `SessionEvent::InGame` actually fires. Root cause this defends against (task 2.3b's
    /// incident): a peer that keeps sending `reconnect@ddnet.org` right after every handshake used
    /// to drive an unconditional, silent, un-backed-off reconnect loop that nothing but the
    /// process-wide 5-attempts/20s rate limiter ever bounded, and that limiter does not by itself
    /// count as "noticing" anything is wrong — see `crate::driver`'s module docs.
    pub handshake_timeout: Duration,
    /// Task 2.3b: absolute cap for the watchdog above — map-download progress extends its deadline
    /// by `handshake_timeout` each time new map bytes arrive, but never past this long after the
    /// join started (or after an in-game loss). See `crate::driver::HANDSHAKE_HARD_CAP`.
    pub handshake_hard_cap: Duration,
    /// Where [`crate::driver`] looks for/writes cached maps
    /// (`<cache_dir>/<name>_<sha256-hex>.map` — see `crate::map_cache`). [`Session`] itself never
    /// touches this (sans-IO); only [`crate::driver::Client`] reads it.
    pub cache_dir: std::path::PathBuf,
    /// Task 8.4a: when `true`, every assembled snapshot also produces a
    /// [`SessionEvent::SnapshotData`] (the full, owned snapshot — every player/character, not
    /// just this client's own) alongside the existing [`SessionEvent::Snapshot`] (just the tick).
    /// `false` by default: cloning every snapshot's items is wasted work for a caller that only
    /// ever needs its own position (`ddnet-ai play`'s `ClientEvent::OwnPosition`, already served
    /// without this) — `ddnet-ai record` (the first, and so far only, caller that needs the whole
    /// snapshot) is the one place this is turned on.
    pub emit_snapshot_data: bool,
    /// Review round 1, finding F7 (task 8.4a) / task 2.4 review round 1, finding F5 — both tasks
    /// added this independently with the same name and shape; merged into one field on origin/main
    /// (task 2.4 landed first). When `true`, [`Session::send_input`] also queues a
    /// [`SessionEvent::InputSent`] for every `NETMSG_INPUT` actually sent (~50/s while in-game).
    /// `false` by default — a caller that never reads `--input-log`-style ground truth, or never
    /// needs its own exact sent-input history (e.g. `LiveWorld::predict`'s
    /// `own_inputs_in_flight`), pays nothing for it; `ddnet-ai play --brain random-scripted
    /// --input-log` and `ddnet-ai record --input-log` are task 8.4a's own callers, turned on only
    /// when `--input-log` is actually given.
    pub emit_input_sent: bool,
    /// `Cl_ShowOthers`'s `show` value (`SHOW_OTHERS_OFF=0`/`SHOW_OTHERS_ON=1`/
    /// `SHOW_OTHERS_ONLY_TEAM=2`), sent once after entering alongside `Cl_ShowDistance` — `0` by
    /// default (a fresh, default-config real client's own value, `config_variables.h:669` —
    /// task 2.3's own reasoning for why this crate mirrors that default rather than tracking real
    /// HUD state it doesn't have). Review round 1, finding F13: `ddnet-ai record` sets this to
    /// `1` — a spectator whose `Cl_SetTeam(TEAM_SPECTATORS)` request is refused (an unfamiliar mod,
    /// spam/kill-protection) falls back to *playing*, where `show_others=0`'s "only my own
    /// collision group" filtering (`character.cpp:1220-1222`, task 8.4a §16.1's own citation)
    /// would otherwise hide every other player from exactly the session this bot exists to
    /// observe.
    pub show_others: i32,
    /// Task 4.1: when `true`, every outgoing game message also queues a
    /// [`SessionEvent::OutgoingGame`] (a handful per join plus the bot's `Cl_Kill`/`Cl_SetTeam`;
    /// never per tick). `false` by default. The live bot's e2e test turns it on to audit that no
    /// chat message is ever sent (D-007).
    pub emit_outgoing_audit: bool,
    /// Review round 1, finding F1: the D-027/D-038 safety switch, checked by
    /// [`crate::driver::Client`] before **every** socket connect it ever makes for this
    /// `ClientConfig` — the very first one, every reconnect, and every redirect (a server that
    /// redirects a client to a second, non-loopback, non-listed address must not be able to
    /// bypass a check that only ran once, against the original `--server` argument). Loopback is
    /// always allowed regardless of this list's contents (see [`live_servers::check`]). Defaults
    /// to loading [`live_servers::LiveServers::default_path`]
    /// (`~/aiddnet/data/live-servers.toml`) via [`Default::default`] — a missing file is an empty
    /// list (refuses every non-loopback address, the safe direction), and so, deliberately, is a
    /// file that fails to parse (logged as a warning, never a panic or a silent "allow anything").
    pub live_servers: live_servers::LiveServers,
}

/// `~/aiddnet/data/maps/cache`, per `CLAUDE.md`'s folder layout, falling back to a relative
/// `data/maps/cache` if `$HOME` isn't set — same fallback pattern `ddai-web`'s CLI uses.
pub fn default_cache_dir() -> std::path::PathBuf {
    match std::env::var_os("HOME") {
        Some(home) if !home.is_empty() => std::path::PathBuf::from(home)
            .join("aiddnet")
            .join("data")
            .join("maps")
            .join("cache"),
        _ => std::path::PathBuf::from("data").join("maps").join("cache"),
    }
}

impl Default for ClientConfig {
    fn default() -> Self {
        ClientConfig {
            name: "ddai-bot".to_string(),
            clan: String::new(),
            skin: "default".to_string(),
            country: -1,
            password: String::new(),
            show_distance: (3000, 2000),
            ddnet_version: 20010,
            version_str: DEFAULT_VERSION_STR.to_string(),
            prediction_margin_ms: crate::timing::DEFAULT_PREDICTION_MARGIN_MS,
            max_map_size_bytes: DEFAULT_MAX_MAP_SIZE_BYTES,
            timeout: conn::DEFAULT_TIMEOUT,
            handshake_timeout: crate::driver::DEFAULT_HANDSHAKE_TIMEOUT,
            handshake_hard_cap: crate::driver::HANDSHAKE_HARD_CAP,
            cache_dir: default_cache_dir(),
            emit_snapshot_data: false,
            emit_input_sent: false,
            show_others: 0,
            emit_outgoing_audit: false,
            live_servers: default_live_servers(),
        }
    }
}

/// Loads [`live_servers::LiveServers::default_path`], falling back to an empty list (refusing
/// every non-loopback address) on any error — a missing file (`load_or_empty`'s own contract) or
/// one that fails to parse (logged here, once, as a warning: silently treating a malformed
/// allow-list as "allow everything" would be exactly backwards for a safety switch).
fn default_live_servers() -> live_servers::LiveServers {
    match live_servers::LiveServers::load_or_empty(&live_servers::LiveServers::default_path()) {
        Ok(list) => list,
        Err(e) => {
            tracing::warn!(error = %e, "failed to load live-servers.toml; defaulting to an empty allow-list");
            live_servers::LiveServers::default()
        }
    }
}

/// `CServerCapabilities` (`client.h`), as decoded by `GetServerCapabilities`
/// (`client.cpp:1602-1636`) — informational (D-024: exposed via events/getters, this bot's own
/// behaviour never branches on `sync_weapon_input` since `ClientConfig::prediction_margin_ms` is
/// used unconditionally either way — see `crate::timing`'s module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ServerCapabilities {
    pub chat_timeout_code: bool,
    pub any_player_flag: bool,
    pub ping_ex: bool,
    pub allow_dummy: bool,
    pub sync_weapon_input: bool,
}

mod servercapflag {
    pub const DDNET: i32 = 1 << 0;
    pub const CHATTIMEOUTCODE: i32 = 1 << 1;
    pub const ANYPLAYERFLAG: i32 = 1 << 2;
    pub const PINGEX: i32 = 1 << 3;
    pub const ALLOWDUMMY: i32 = 1 << 4;
    pub const SYNCWEAPONINPUT: i32 = 1 << 5;
}

impl ServerCapabilities {
    /// `GetServerCapabilities` (`client.cpp:1602-1636`), Sixup branch omitted (this project never
    /// speaks 0.7/sixup — see the crate root docs).
    fn from_version_flags(version: i32, flags: i32) -> Self {
        let ddnet = version >= 1 && (flags & servercapflag::DDNET != 0);
        let mut caps = ServerCapabilities {
            chat_timeout_code: ddnet,
            any_player_flag: true,
            ping_ex: false,
            allow_dummy: true,
            sync_weapon_input: false,
        };
        if version >= 1 {
            caps.chat_timeout_code = flags & servercapflag::CHATTIMEOUTCODE != 0;
        }
        if version >= 2 {
            caps.any_player_flag = flags & servercapflag::ANYPLAYERFLAG != 0;
        }
        if version >= 3 {
            caps.ping_ex = flags & servercapflag::PINGEX != 0;
        }
        if version >= 4 {
            caps.allow_dummy = flags & servercapflag::ALLOWDUMMY != 0;
        }
        if version >= 5 {
            caps.sync_weapon_input = flags & servercapflag::SYNCWEAPONINPUT != 0;
        }
        caps
    }
}

/// Where a loaded map's bytes came from — see [`MapLoadedEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapSource {
    Cache,
    Downloaded,
}

/// Payload of [`SessionEvent::MapLoaded`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapLoadedEvent {
    pub name: String,
    pub crc32: u32,
    pub sha256: [u8; 32],
    pub source: MapSource,
    /// `Some` only when `source == Downloaded` — the driver should write these bytes to the
    /// cache (`crate::map_cache::write_cache`) so the *next* join skips the download entirely
    /// (task acceptance criterion 2: "never re-downloaded when cached").
    pub bytes_to_cache: Option<Vec<u8>>,
}

/// Everything [`Session::feed`]/[`Session::flush`]/[`Session::take_events`] can report — task
/// acceptance criterion 1's "a stream of events (connected, map loaded, snapshot, game message,
/// tuning, kicked/banned, redirected, disconnected)".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionEvent {
    /// The TKEN handshake completed.
    Connected,
    /// `NETMSG_MAP_CHANGE` was received — see the module docs' "map acquisition" section for what
    /// the driver should do with this *before* its next [`Session::flush`] call.
    MapChanging {
        name: String,
        crc: i32,
        size: i32,
        sha256: Option<[u8; 32]>,
        url: Option<String>,
    },
    /// The map (cached or freshly downloaded) was loaded and verified; `NETMSG_READY` was sent.
    MapLoaded(MapLoadedEvent),
    /// `NETMSG_ENTERGAME` was sent; snapshots are about to start.
    InGame,
    /// A complete snapshot was assembled for `tick` — [`Session::latest_view`] returns a typed
    /// view over it.
    Snapshot { tick: i32 },
    /// Task 8.4a: the same snapshot as the preceding [`SessionEvent::Snapshot`], but as an owned
    /// [`ddai_net::snapshot::Snapshot`] rather than "go look it up via `Session::latest_view`" —
    /// only produced when [`ClientConfig::emit_snapshot_data`] is `true` (see that field's docs
    /// for why this is opt-in). Exists because [`crate::driver::Client`]'s cross-thread event
    /// channel cannot hand out a borrowed [`View`] (it would borrow `Session`'s own state, which
    /// never leaves the driver thread) — a caller that needs the *whole* snapshot (every
    /// player/character, not just its own — `ddnet-ai record`) builds `View::new(&snapshot)` over
    /// this event's payload on whichever thread it likes.
    SnapshotData {
        tick: i32,
        snapshot: ddai_net::snapshot::Snapshot,
    },
    /// A non-UUID game message the caller might care about (chat to *read* is fine — D-007 only
    /// forbids *sending* it; see `crate::allowlist`).
    GameMessage(msgs::GameMsg),
    /// A UUID (`ex`) game message.
    ExGameMessage(msgs::ExGameMsg),
    /// `Sv_TuneParams` was applied.
    Tuning(TuneParams),
    /// The connection ended for a reason genuinely worth retrying: either the peer closed us with
    /// a reason the driver's policy classifies as transient (`by_peer: true` — e.g. "Server
    /// shutdown", "This server is full", a timeout-flavoured reason; see
    /// `crate::driver::should_reconnect_after_peer_close`), or a local network-level issue
    /// (`by_peer: false` — a socket error, or `ddai_net::conn::Event::Error`, i.e. `Connection`'s
    /// own silence-timeout/too-weak-connection detection). The wire protocol has no separate
    /// "banned" message, just a `CLOSE` with a reason string, verbatim in `reason` either way.
    /// This is **not** used for local protocol violations (bad map name/size, a hash/CRC
    /// mismatch, a failed map load) — those are [`SessionEvent::ProtocolViolation`], which the
    /// driver never retries (review finding F4: retrying those forever against a server that is
    /// simply sending us garbage is not "staying connected", it's spinning).
    Disconnected { reason: Option<String>, by_peer: bool },
    /// A local, self-detected protocol violation ended the session: an invalid/hostile
    /// `NETMSG_MAP_CHANGE` (bad filename, hostile size) or a downloaded/cached map that failed
    /// verification (wrong hash/CRC, failed to parse). Always final — see
    /// [`SessionEvent::Disconnected`]'s docs for why this is a separate variant from a genuine
    /// lost connection.
    ProtocolViolation { reason: String },
    /// `reconnect@ddnet.org` — the driver should reconnect to the same address (a fresh
    /// [`Session`]/socket).
    ReconnectRequested,
    /// `redirect@ddnet.org` — the driver should reconnect to the same host, this new port,
    /// following at most once (loop protection is the driver's job — see `crate::driver`).
    RedirectRequested { port: u16 },
    /// A heuristic anomaly worth the caller's attention (task acceptance criterion 4's
    /// "captcha/lobby detection heuristics ... the caller decides") — currently: repeated
    /// snapshot CRC mismatches. Map name/size anomalies are already fully surfaced via
    /// [`SessionEvent::MapChanging`]'s own fields, so there is nothing further to add there; a
    /// caller wanting a "this looks like a captcha lobby" heuristic can inspect those directly
    /// (e.g. an implausibly small `size` for a known map name).
    Anomaly(String),
    /// Task 8.4a's own addition, and task 2.4 review round 1's finding F5, independently added the
    /// same event with the same name and shape (task 2.4 landed on `main` first — see this enum's
    /// module docs on merging). The exact `PlayerInput` just embedded in a `NETMSG_INPUT` for
    /// `tick` (the same `tick`/`input` [`Session::send_input`] just packed onto the wire). Gated
    /// by [`ClientConfig::emit_input_sent`] — task 8.4a's own use is ground truth for `ddnet-ai
    /// play --brain random-scripted`'s input log (acceptance criterion 3's validation harness:
    /// reconstructed inputs, derived offline from a *recording* of this same session, are compared
    /// against this event's log); task 2.4's own use is `LiveWorld::predict`'s
    /// `own_inputs_in_flight` ground truth — without it, a caller has no way to know what input it
    /// actually queued for a given future tick, only the *current* input `Client::set_input` last
    /// set (which may have changed since). Emitted from [`Session::flush`] (where the real send
    /// happens) but delivered through [`Session::take_events`], not `flush`'s own return value —
    /// `flush` already has a fixed, unrelated return type (bytes to send).
    InputSent { tick: i32, input: objects::PlayerInput },
    /// The server's own `NETMSG_INPUTTIMING` feedback for one `pred_tick`
    /// (`client.cpp:2084-2108`, `crate::timing::InputTiming::on_input_timing`) — `time_left`'s
    /// sign is the "did this input make its deadline" signal: negative means the server tells us
    /// our `NETMSG_INPUT` for `tick` (== a [`SessionEvent::InputSent::tick`]) missed its intended
    /// tick, so the server kept using the *previous* input for it instead. Gated by the same
    /// [`ClientConfig::emit_input_sent`] flag as [`SessionEvent::InputSent`] — the two only make
    /// sense used together, correcting `own_inputs_in_flight` for a `LiveWorld::predict` accuracy
    /// measurement (a late tick's sent input never actually took effect on that exact tick). Task
    /// 2.4's own addition — task 8.4a does not currently read this event, but it costs nothing
    /// extra beyond what `emit_input_sent` already pays for.
    InputTiming { tick: i32, time_left: i32 },
    /// Task 4.1 (D-007 audit): one outgoing *game* message went through
    /// [`Session::send_game_chunk`] — `label` is the builder's own name (`"Cl_Kill"`,
    /// `"Cl_SetTeam"`, ...), `accepted` whether [`crate::allowlist`] let it reach the wire. Gated by
    /// [`ClientConfig::emit_outgoing_audit`]; the e2e test asserts no chat label ever shows up.
    OutgoingGame { label: &'static str, accepted: bool },
}

/// Why [`Session::supply_cached_map`] refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SupplyCachedMapError {
    #[error("not currently waiting for map bytes (wrong session state)")]
    WrongState,
    #[error("cached map bytes failed verification: {0}")]
    VerifyFailed(String),
}

#[derive(Debug, Clone)]
struct PendingMapDetails {
    name: String,
    sha256: [u8; 32],
    crc: i32,
    size: i32,
    url: String,
}

#[derive(Debug, Clone)]
struct PendingMapChange {
    name: String,
    crc: i32,
    size: i32,
    sha256: Option<[u8; 32]>,
}

#[derive(Debug)]
struct MapDownload {
    name: String,
    crc: i32,
    size: i32,
    sha256: Option<[u8; 32]>,
    next_chunk: i32,
    buffer: Vec<u8>,
}

#[derive(Debug)]
enum JoinState {
    Handshaking,
    AwaitingMapChange,
    /// Waiting for the driver to (maybe) call [`Session::supply_cached_map`] — see the module
    /// docs' "map acquisition" section.
    AwaitingMapBytes(PendingMapChange),
    Downloading(MapDownload),
    AwaitingConReady,
    AwaitingReadyToEnter,
    InGame,
}

/// One entry in [`Session::recent_outgoing`] — the audit trail behind task acceptance criterion
/// 6g's "refused with an error and logged".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutgoingLogEntry {
    pub label: &'static str,
    pub accepted: bool,
}

/// How many [`OutgoingLogEntry`] to keep — bounded memory (acceptance criterion 7), plenty for
/// both real play (a handful of these messages per join) and tests.
const OUTGOING_LOG_CAP: usize = 128;

fn to_ns(d: Duration) -> i64 {
    i64::try_from(d.as_nanos()).unwrap_or(i64::MAX)
}

/// 16 pseudo-random bytes for `NETMSG_CLIENTVER`'s `connection_id` (`m_ConnectionId = RandomUuid()`,
/// `client.cpp:697`) — not security-sensitive (it is purely a server-side per-connection
/// bookkeeping tag, never a secret or capability token), so this uses `RandomState`'s
/// already-OS-seeded keys instead of pulling in a `rand`/CSPRNG dependency for one field.
fn random_bytes_16() -> [u8; 16] {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let a = RandomState::new().build_hasher().finish();
    let b = RandomState::new().build_hasher().finish();
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&a.to_le_bytes());
    out[8..].copy_from_slice(&b.to_le_bytes());
    out
}

fn player_input_to_ints(input: &objects::PlayerInput) -> [i32; objects::PlayerInput::SIZE_INTS] {
    [
        input.direction,
        input.target_x,
        input.target_y,
        input.jump,
        input.fire,
        input.hook,
        input.player_flags,
        input.wanted_weapon,
        input.next_weapon,
        input.prev_weapon,
    ]
}

/// The name registered for each [`ExSysMsg`] variant this session ever *sends* (a subset of
/// `ddai_net::uuid::REGISTERED_NAMES`) — the ex-system-message analogue of `sysmsg_id_of` below.
fn ex_sys_name(msg: &ExSysMsg) -> Option<&'static str> {
    Some(match msg {
        ExSysMsg::ClientVer { .. } => "clientver@ddnet.tw",
        // Review finding F9: this bot now also *originates* pings (`ExSysMsg::PingEx`,
        // `Session::maybe_send_ping_ex`), not just replies to the server's — same wire name in
        // both directions (`protocol_ex_msgs.h:35`: `UUID(NETMSG_PINGEX, "ping@ddnet.tw")`).
        ExSysMsg::PingEx { .. } => "ping@ddnet.tw",
        ExSysMsg::PongEx { .. } => "pong@ddnet.tw",
        ExSysMsg::ChecksumError { .. } => "checksum-error@ddnet.tw",
        ExSysMsg::ItIs { .. } => "it-is@ddnet.tw",
        ExSysMsg::IDontKnow { .. } => "i-dont-know@ddnet.tw",
        _ => return None,
    })
}

/// The numbered id for each [`SysMsg`] variant this session ever *sends*.
fn sysmsg_id_of(msg: &SysMsg) -> Option<i32> {
    Some(match msg {
        SysMsg::Info { .. } => sysmsg::id::INFO,
        SysMsg::Ready => sysmsg::id::READY,
        SysMsg::EnterGame => sysmsg::id::ENTERGAME,
        SysMsg::Input { .. } => sysmsg::id::INPUT,
        SysMsg::RequestMapData { .. } => sysmsg::id::REQUEST_MAP_DATA,
        SysMsg::PingReply => sysmsg::id::PING_REPLY,
        _ => return None,
    })
}

/// Builds a numbered (non-ex) game message payload: leading `(id<<1)|0` varint + `body`.
fn build_numbered_game_payload(id: i32, body: impl FnOnce(&mut Packer)) -> Vec<u8> {
    let mut buf = [0u8; 2048];
    let mut packer = Packer::new(&mut buf);
    uuid::pack_msg_id(&mut packer, MsgId::Numbered(id), false);
    body(&mut packer);
    packer.data().to_vec()
}

/// Builds a UUID (ex) game message payload.
fn build_ex_game_payload(name: &str, body: impl FnOnce(&mut Packer)) -> Vec<u8> {
    let mut buf = [0u8; 2048];
    let mut packer = Packer::new(&mut buf);
    let id = uuid::calculate_uuid(name);
    uuid::pack_msg_id(
        &mut packer,
        MsgId::Ex {
            uuid: id,
            resolved: None,
        },
        false,
    );
    body(&mut packer);
    packer.data().to_vec()
}

/// The sans-IO client session — see the module docs.
pub struct Session {
    config: ClientConfig,
    connection: Connection,
    huffman: Huffman,
    registry: Registry,
    snap_assembler: SnapAssembler,
    timing: InputTiming,
    state: JoinState,
    server_capabilities: ServerCapabilities,
    can_receive_capabilities: bool,
    pending_map_details: Option<PendingMapDetails>,
    connection_id: [u8; 16],
    current_input: objects::PlayerInput,
    sent_post_enter_extras: bool,
    last_snapshot_tick: Option<i32>,
    tuning: TuneParams,
    teams_state: Option<TeamsState>,
    outgoing_log: VecDeque<OutgoingLogEntry>,
    /// When to next originate a `PINGEX` (`client.cpp:527,2982-3004`) — `None` until `ENTERGAME`,
    /// review finding F9.
    next_ping_ex_at_ns: Option<i64>,
    /// Events produced outside [`Session::feed`]'s own control flow (currently
    /// [`SessionEvent::InputSent`]/[`SessionEvent::InputTiming`], queued by [`Session::send_input`]
    /// from inside [`Session::flush`], and `on_input_timing`) — drained by
    /// [`Session::take_events`], the same channel [`ddai_net::conn::Connection`]'s own out-of-band
    /// events already use.
    pending_events: VecDeque<SessionEvent>,
}

impl Session {
    pub fn new(config: ClientConfig) -> Self {
        let margin = config.prediction_margin_ms;
        let timeout = config.timeout;
        Session {
            connection: Connection::new(conn::Config {
                timeout,
                resend_requests_per_second: conn::DEFAULT_RESEND_REQUESTS_PER_SECOND,
            }),
            huffman: Huffman::new(),
            registry: Registry::new(),
            snap_assembler: SnapAssembler::new(StaticSizes::ddnet_06()),
            timing: InputTiming::new(margin),
            state: JoinState::Handshaking,
            server_capabilities: ServerCapabilities::default(),
            can_receive_capabilities: true,
            pending_map_details: None,
            connection_id: random_bytes_16(),
            current_input: objects::PlayerInput {
                direction: 0,
                target_x: 0,
                target_y: -1,
                jump: 0,
                fire: 0,
                hook: 0,
                player_flags: playerflagflag::PLAYING,
                wanted_weapon: 0,
                next_weapon: 0,
                prev_weapon: 0,
            },
            sent_post_enter_extras: false,
            last_snapshot_tick: None,
            tuning: DEFAULT_TUNE_PARAMS,
            teams_state: None,
            outgoing_log: VecDeque::new(),
            next_ping_ex_at_ns: None,
            pending_events: VecDeque::new(),
            config,
        }
    }

    /// Starts the TKEN handshake (`Connection::connect`) — the very first step of the join
    /// sequence.
    pub fn connect(&mut self, now: Duration) {
        tracing::info!("control: sending CONNECT (starting TKEN handshake)");
        self.state = JoinState::Handshaking;
        self.connection.connect(now, &self.huffman);
    }

    /// Voluntarily ends the session (`Connection::disconnect` — queues `CLOSE`, resets to
    /// offline). Does *not* itself produce a [`SessionEvent`] — the caller already knows it asked
    /// for this; call [`Session::flush`] afterwards to actually send the `CLOSE`.
    pub fn disconnect(&mut self, reason: Option<&str>) {
        tracing::info!(reason = ?reason, "control: sending CLOSE");
        self.connection.disconnect(reason, &self.huffman);
    }

    /// Sets the input to embed in every `NETMSG_INPUT` from now on, until called again — the
    /// caller (a brain) is expected to call this at least once before/around every tick it cares
    /// about; [`Session::flush`] sends whatever was set most recently, exactly like the real
    /// client sends whatever `OnSnapInput` last produced.
    pub fn set_input(&mut self, input: objects::PlayerInput) {
        self.current_input = input;
    }

    /// Sends `Cl_SetTeam` (task 8.4a acceptance criterion 1: "after entering, it joins spectators
    /// the way the real 20.1 client does" — `gamecontext.cpp:2684-2730`,
    /// `CGameContext::OnSetTeamNetMessage`, server side). `NETMSGTYPE_CL_SETTEAM` has been on
    /// [`crate::allowlist`]'s allow-list since task 2.3 (kept there "for a future public setter",
    /// per that module's doc comment) but had no builder until now — nothing else in this file
    /// ever sent it. Does not itself gate on [`Session::is_in_game`]: a caller racing this against
    /// the join sequence gets exactly what a real client sending it too early would (the server
    /// drops an `NETMSGTYPE_CL_SETTEAM` from a client it does not yet have a player slot for), not
    /// a panic or a queued-forever message.
    pub fn request_team(&mut self, team: i32, now: Duration) {
        let payload = build_numbered_game_payload(msgs::id::NETMSGTYPE_CL_SETTEAM, |p| {
            msgs::encode_cl_set_team(&msgs::ClSetTeam { team }, p);
        });
        self.send_game_chunk(payload, true, now, "Cl_SetTeam");
    }

    /// Sends `Cl_Kill` (task 4.1) — the protocol message behind the bot's unstick; chat `/kill`
    /// has no builder anywhere in this crate (D-007). `NETMSGTYPE_CL_KILL` has been on
    /// [`crate::allowlist`]'s allow-list since task 2.3. The caller owns the cooldown
    /// (`ddai-bot`'s unstick: 500 ticks, `bot.ts` `KILL_COOLDOWN_TICKS`); the server applies its
    /// own `sv_kill_delay`/kill-protection either way (`gamecontext.cpp` `OnKillNetMessage`).
    pub fn request_kill(&mut self, now: Duration) {
        let payload = build_numbered_game_payload(msgs::id::NETMSGTYPE_CL_KILL, |p| {
            msgs::encode_cl_kill(&msgs::ClKill {}, p);
        });
        self.send_game_chunk(payload, true, now, "Cl_Kill");
    }

    /// Sends `Cl_ShowDistance(x, y)` (task 4.1) — the same message [`Session::send_post_enter_extras`]
    /// sends once after entering, exposed so the bot can change its view range at runtime (D-007's
    /// replacement for the old bot's `/showall` chat command).
    pub fn request_show_distance(&mut self, x: i32, y: i32, now: Duration) {
        let payload = build_ex_game_payload("show-distance@netmsg.ddnet.tw", |p| {
            msgs::encode_cl_show_distance(&msgs::ClShowDistance { x, y }, p);
        });
        self.send_game_chunk(payload, true, now, "Cl_ShowDistance");
    }

    /// The predicted tick of the last `NETMSG_INPUT` sent (`0` before the two-snapshot bootstrap)
    /// — [`crate::timing::InputTiming::pred_tick`]. Task 4.1: the bot's prediction target.
    pub fn pred_tick(&self) -> i32 {
        self.timing.pred_tick()
    }

    /// Task 4.1: time until the next `NETMSG_INPUT` is due ([`crate::timing::InputTiming::next_input_in_ns`]).
    pub fn next_input_in(&self, now: Duration) -> Option<Duration> {
        self.timing
            .next_input_in_ns(to_ns(now))
            .map(|ns| Duration::from_nanos(u64::try_from(ns).unwrap_or(0)))
    }

    pub fn is_in_game(&self) -> bool {
        matches!(self.state, JoinState::InGame)
    }

    /// Task 2.3b: bytes of the map received so far while a download is in progress (`None`
    /// otherwise) — the driver's handshake watchdog extends its deadline while this keeps growing.
    pub fn download_progress(&self) -> Option<usize> {
        match &self.state {
            JoinState::Downloading(dl) => Some(dl.buffer.len()),
            _ => None,
        }
    }

    /// Task 2.3b: a short, stable name of the join phase this session is waiting in, for the
    /// driver's periodic "still waiting for X" info line (a stalled join must say where it stalled).
    pub fn join_phase(&self) -> &'static str {
        match self.state {
            JoinState::Handshaking => "waiting for CONNECTACCEPT (TKEN handshake; CONNECT resent every 500ms)",
            JoinState::AwaitingMapChange => "waiting for MAP_CHANGE after CLIENTVER+INFO",
            JoinState::AwaitingMapBytes(_) => "waiting for the map cache lookup",
            JoinState::Downloading(_) => "downloading the map",
            JoinState::AwaitingConReady => "waiting for CON_READY after READY",
            JoinState::AwaitingReadyToEnter => "waiting for Sv_ReadyToEnter after Cl_StartInfo",
            JoinState::InGame => "in game",
        }
    }

    pub fn server_capabilities(&self) -> ServerCapabilities {
        self.server_capabilities
    }

    pub fn tuning(&self) -> TuneParams {
        self.tuning
    }

    pub fn teams_state(&self) -> Option<TeamsState> {
        self.teams_state
    }

    /// The ack value the next `NETMSG_INPUT` will carry (`ddai_net::assembly::SnapAssembler::ack_game_tick`).
    pub fn ack_game_tick(&self) -> i32 {
        self.snap_assembler.ack_game_tick()
    }

    /// A typed view over the most recently assembled snapshot, if any.
    pub fn latest_view(&self) -> Option<View<'_>> {
        let tick = self.last_snapshot_tick?;
        self.snap_assembler.storage().get(tick).map(View::new)
    }

    /// The `NETMSG_INPUTTIMING` margin distribution observed so far (task acceptance criterion h).
    pub fn margin_summary(&self) -> MarginSummary {
        self.timing.margin_summary()
    }

    /// The last [`OUTGOING_LOG_CAP`] outgoing game messages (accepted or refused by
    /// [`crate::allowlist`]) — the audit trail/test hook behind acceptance criterion 6g.
    pub fn recent_outgoing(&self) -> impl Iterator<Item = &OutgoingLogEntry> {
        self.outgoing_log.iter()
    }

    /// Feeds one received datagram — see [`ddai_net::conn::Connection::feed`]'s docs for the
    /// general shape (never panics on malformed/hostile input).
    pub fn feed(&mut self, datagram: &[u8], now: Duration) -> Vec<SessionEvent> {
        let mut events = Vec::new();
        for ev in self.connection.feed(datagram, &self.huffman, now) {
            match ev {
                conn::Event::Connected => {
                    // Task 2.3b acceptance criterion 4: an info-level line for every connection
                    // state transition, logged here at the library level (not only via the
                    // `SessionEvent` a caller might or might not have a handler for) — this is
                    // exactly the gap the task 2.3b incident fell through: a caller-side match
                    // arm silently downgraded an unhandled event to debug level, so nothing at
                    // info level ever showed the reconnect loop happening at all.
                    tracing::info!("connection: received CONNECTACCEPT, sent ACCEPT, online (TKEN handshake complete)");
                    self.on_connected(now);
                    events.push(SessionEvent::Connected);
                }
                conn::Event::Chunk { data, vital } => {
                    let (msg, answer) = message::decode(&data, &self.registry);
                    if let Some(answer) = answer {
                        self.send_ex_system_chunk(&answer, true, now);
                    }
                    events.extend(self.handle_msg(msg, vital, now));
                }
                conn::Event::ClosedByPeer(reason) => {
                    tracing::info!(reason = %reason, "connection: closed by peer (CLOSE received)");
                    events.push(SessionEvent::Disconnected {
                        reason: if reason.is_empty() { None } else { Some(reason) },
                        by_peer: true,
                    });
                }
                conn::Event::Error(reason) => {
                    tracing::info!(reason = %reason, "connection: local error/timeout");
                    events.push(SessionEvent::Disconnected {
                        reason: Some(reason),
                        by_peer: false,
                    });
                }
            }
        }
        events
    }

    /// Runs periodic work (map-download kickoff, the predicted-tick/`NETMSG_INPUT` cadence,
    /// `Connection`'s own keepalive/resend/timeout timers) and returns every datagram that should
    /// be sent right now — see [`ddai_net::conn::Connection::flush`]'s docs for why this is
    /// separate from [`Session::feed`].
    pub fn flush(&mut self, now: Duration) -> Vec<Vec<u8>> {
        let pending_to_start = match &self.state {
            JoinState::AwaitingMapBytes(pending) => Some(pending.clone()),
            _ => None,
        };
        if let Some(pending) = pending_to_start {
            self.begin_protocol_download(pending, now);
        }

        if matches!(self.state, JoinState::InGame) {
            if let Some(tick) = self.timing.advance(to_ns(now)) {
                self.send_input(tick, now);
            }
            self.maybe_send_ping_ex(now);
        }

        self.connection.flush(&self.huffman, now)
    }

    /// Drains events produced by something other than [`Session::feed`] (a
    /// [`ddai_net::conn::Connection`]-detected timeout/too-weak-connection, observed via its own
    /// `take_events`, plus [`Session::pending_events`] — currently [`SessionEvent::InputSent`]/
    /// [`SessionEvent::InputTiming`], queued by [`Session::send_input`]/`on_input_timing`) — call
    /// this after [`Session::flush`]; `pending_events` entries, if any, come first (they were
    /// produced first, by that same `flush()` call).
    pub fn take_events(&mut self) -> Vec<SessionEvent> {
        let mut events: Vec<SessionEvent> = self.pending_events.drain(..).collect();
        events.extend(self.connection.take_events().into_iter().map(|ev| match ev {
            conn::Event::Error(reason) => {
                tracing::info!(%reason, "connection: local error/timeout (detected by flush)");
                SessionEvent::Disconnected {
                    reason: Some(reason),
                    by_peer: false,
                }
            }
            conn::Event::ClosedByPeer(reason) => {
                tracing::info!(%reason, "connection: closed by peer (CLOSE received, detected by flush)");
                SessionEvent::Disconnected {
                    reason: if reason.is_empty() { None } else { Some(reason) },
                    by_peer: true,
                }
            }
            // `Connection::take_events` never actually produces these two (see its own docs:
            // only state transitions *outside* `feed()` — timeouts and similar — end up
            // here), kept for an exhaustive match rather than a wildcard so a future
            // `ddai_net` change that *did* start emitting one of these would fail to compile
            // here instead of silently being dropped.
            conn::Event::Connected => SessionEvent::Connected,
            conn::Event::Chunk { .. } => {
                SessionEvent::Anomaly("unexpected Chunk event from Connection::take_events".to_string())
            }
        }));
        events
    }

    /// The driver calls this after seeing [`SessionEvent::MapChanging`] with `sha256.is_some()`,
    /// if it found a matching file in its cache — see the module docs. Returns the same event
    /// [`Session::flush`]/[`Session::feed`] would otherwise eventually deliver via
    /// [`SessionEvent::MapLoaded`] wrapped `Ok`, or an error if the bytes fail verification (in
    /// which case the session stays in `AwaitingMapBytes` — the *next* `flush()` call will start
    /// the ordinary protocol download as a fallback) or the session was not actually waiting for
    /// map bytes at all.
    pub fn supply_cached_map(&mut self, bytes: &[u8], now: Duration) -> Result<MapLoadedEvent, SupplyCachedMapError> {
        let pending = match &self.state {
            JoinState::AwaitingMapBytes(pending) => pending.clone(),
            _ => return Err(SupplyCachedMapError::WrongState),
        };
        self.finish_map_load(pending, bytes.to_vec(), MapSource::Cache, now)
            .map_err(SupplyCachedMapError::VerifyFailed)
    }

    // ---- internal: connection lifecycle -------------------------------------------------------

    /// `client.cpp:2712-2718`: once `Connection` reaches Online, send `NETMSG_CLIENTVER` then
    /// `NETMSG_INFO`.
    fn on_connected(&mut self, now: Duration) {
        self.state = JoinState::AwaitingMapChange;
        self.can_receive_capabilities = true;
        tracing::info!("control: sending CLIENTVER, then INFO");
        let client_ver = ExSysMsg::ClientVer {
            connection_id: self.connection_id,
            ddnet_version: self.config.ddnet_version,
            version_str: self.config.version_str.clone(),
        };
        self.send_ex_system_chunk(&client_ver, true, now);
        let info = SysMsg::Info {
            netversion: NETVERSION_STRING.to_string(),
            password: self.config.password.clone(),
        };
        self.send_system_chunk(&info, true, now);
    }

    // ---- internal: message dispatch -----------------------------------------------------------

    fn handle_msg(&mut self, msg: Msg, vital: bool, now: Duration) -> Vec<SessionEvent> {
        match msg {
            Msg::Sys(sys_msg) => self.handle_sys(sys_msg, vital, now),
            Msg::ExSys(ex_msg) => self.handle_ex_sys(ex_msg, vital, now),
            Msg::Game(game_msg) => self.handle_game(game_msg, now),
            Msg::ExGame(ex_game_msg) => vec![SessionEvent::ExGameMessage(ex_game_msg)],
            Msg::TuneParams(params) => {
                self.tuning = params;
                vec![SessionEvent::Tuning(params)]
            }
            Msg::TeamsState(state) => {
                self.teams_state = Some(state);
                Vec::new()
            }
            Msg::Invalid => Vec::new(),
        }
    }

    fn handle_sys(&mut self, sys_msg: SysMsg, vital: bool, now: Duration) -> Vec<SessionEvent> {
        match sys_msg {
            // `client.cpp:1717`/`client.cpp:1896`: the real client only acts on `MAP_CHANGE`/
            // `CON_READY` when the chunk carrying them was vital — review finding F9. A
            // non-vital chunk can be reordered/lost/duplicated by the network layer (see
            // `ddai_net::conn::Event::Chunk`'s docs), which would otherwise let a spoofed/replayed
            // non-vital datagram restart the join sequence out of turn.
            SysMsg::MapChange { name, crc, size } if vital => self.handle_map_change(name, crc, size, now),
            SysMsg::MapChange { .. } => Vec::new(),
            SysMsg::MapData { last, crc, chunk, data } => self.handle_map_data(last, crc, chunk, data, now),
            SysMsg::ConReady if vital => self.handle_con_ready(now),
            SysMsg::ConReady => Vec::new(),
            snap @ (SysMsg::Snap { .. }
            | SysMsg::SnapEmpty { .. }
            | SysMsg::SnapSingle { .. }
            | SysMsg::SnapSmall { .. }) => self.handle_snap(&snap, now),
            SysMsg::InputTiming { pred_tick, time_left } => {
                if self.config.emit_input_sent {
                    self.pending_events.push_back(SessionEvent::InputTiming {
                        tick: pred_tick,
                        time_left,
                    });
                }
                self.timing.on_input_timing(pred_tick, time_left, to_ns(now));
                Vec::new()
            }
            SysMsg::Ping => {
                // `client.cpp:1918-1924`: echo the *same* vital flag the request carried, not a
                // hardcoded one (review finding F9).
                self.send_system_chunk(&SysMsg::PingReply, vital, now);
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    /// `checksum-request@ddnet.tw`/`ping@ddnet.tw`/`capabilities@ddnet.tw`/`map-details@ddnet.tw`/
    /// `redirect@ddnet.org`/`reconnect@ddnet.org` — see the individual arms for citations.
    fn handle_ex_sys(&mut self, ex_msg: ExSysMsg, vital: bool, now: Duration) -> Vec<SessionEvent> {
        match ex_msg {
            // `client.cpp:1676`: real client only stores `MAP_DETAILS` from a vital chunk —
            // review finding F9 (same reordering/replay concern as `MAP_CHANGE` above).
            ExSysMsg::MapDetails {
                name,
                sha256,
                crc,
                size,
                url,
            } if vital => {
                self.pending_map_details = Some(PendingMapDetails {
                    name,
                    sha256,
                    crc,
                    size,
                    url,
                });
                Vec::new()
            }
            ExSysMsg::MapDetails { .. } => Vec::new(),
            // `client.cpp:1701`: same — vital-gated.
            ExSysMsg::Capabilities { version, flags } if vital => {
                // `client.cpp:1701-1716`: only the *first* one is honoured per connection.
                if self.can_receive_capabilities {
                    self.server_capabilities = ServerCapabilities::from_version_flags(version, flags);
                    self.can_receive_capabilities = false;
                }
                Vec::new()
            }
            ExSysMsg::Capabilities { .. } => Vec::new(),
            ExSysMsg::PingEx { id } => {
                // `client.cpp:1925-1930`: answer with the same id *and* the same vital flag the
                // request carried (review finding F9 — was hardcoded `false`).
                self.send_ex_system_chunk(&ExSysMsg::PongEx { id }, vital, now);
                Vec::new()
            }
            ExSysMsg::ChecksumRequest { uuid, rest } => {
                // `client.cpp:1961-1976` dispatches into `HandleChecksum` (`client.cpp:4382-4444`),
                // which returns one of four outcomes: `1` if `Start`/`Length` fail to even unpack
                // (`client.cpp:4384-4389`); `2` if they unpack but fail the bounds check
                // (`Start<0 || Length<0 || Start > INT_MAX-Length`, `client.cpp:4390-4393`); a
                // real `CHECKSUM_RESPONSE` for valid bounds that stay within the small in-memory
                // config-checksum region; and, once the requested range reaches into "hash bytes
                // of my own executable" territory (`client.cpp:4427-4444`), `3` if the client has
                // no readable own executable at all (`m_OwnExecutableSize < 0`,
                // `client.cpp:4429-4439`) or `4` if the range reaches past the end of it
                // (`client.cpp:4440-4443`).
                //
                // Review finding F14 (round 2 — round 1 sent `2` for every valid-bounds request,
                // which is simply the wrong code: `2` means "bounds check failed", not "cannot
                // answer"; a well-formed request that gets past the bounds check is *never* what
                // `2` describes in the real client, for any reason): this bot is not a build of
                // the real DDNet client at all — it has no "own executable" to hash in the sense
                // `HandleChecksum` means (out of scope, see the crate root docs), which is *exactly*
                // the real, named condition error code `3` covers ("own executable not available",
                // `client.cpp:4436` 's own comment) — so a valid-bounds request always gets `3`
                // here, honestly matching a real condition the real client itself can hit, rather
                // than a code that only ever means something else. `1`/`2` still mirror
                // `HandleChecksum`'s own unpack/bounds-check exactly, on the same fields.
                let mut unpacker = ddai_net::packer::Unpacker::new(&rest);
                let start = unpacker.get_int();
                let length = unpacker.get_int();
                let error = if unpacker.error() {
                    1
                } else if start < 0 || length < 0 || start > i32::MAX - length {
                    2
                } else {
                    3
                };
                self.send_ex_system_chunk(&ExSysMsg::ChecksumError { uuid, error }, true, now);
                Vec::new()
            }
            ExSysMsg::Redirect { port } => match u16::try_from(port) {
                Ok(port) if port != 0 => {
                    tracing::info!(port, "control: received redirect@ddnet.org");
                    vec![SessionEvent::RedirectRequested { port }]
                }
                _ => {
                    tracing::info!(
                        port,
                        "control: received redirect@ddnet.org with an invalid port, ignoring"
                    );
                    Vec::new()
                }
            },
            ExSysMsg::Reconnect => {
                // Task 2.3b acceptance criterion 4: logged here, at the library level, for the
                // same reason `conn::Event::Connected`/`ClosedByPeer`/`Error` are above — this is
                // the exact control message the incident's root cause hinges on (see
                // `crate::driver`'s `ConnectionOutcome::Reconnect` docs).
                tracing::info!("control: received reconnect@ddnet.org");
                vec![SessionEvent::ReconnectRequested]
            }
            _ => Vec::new(),
        }
    }

    fn handle_game(&mut self, game_msg: msgs::GameMsg, now: Duration) -> Vec<SessionEvent> {
        let mut events = Vec::new();
        if let msgs::GameMsg::SvReadyToEnter(_) = &game_msg {
            events.extend(self.handle_ready_to_enter(now));
        }
        events.push(SessionEvent::GameMessage(game_msg));
        events
    }

    // ---- internal: map acquisition ------------------------------------------------------------

    /// `client.cpp:1717-1803` (minus the actual byte transfer, started separately — see the
    /// module docs).
    fn handle_map_change(&mut self, name: String, crc: i32, size: i32, _now: Duration) -> Vec<SessionEvent> {
        // `client.cpp:1719-1723`: a `MAP_CHANGE` also finalises capabilities at version 0 if none
        // arrived first.
        if self.can_receive_capabilities {
            self.server_capabilities = ServerCapabilities::from_version_flags(0, 0);
            self.can_receive_capabilities = false;
        }

        let details = self.pending_map_details.take();

        // Review finding F4: a bad/hostile `MAP_CHANGE` is a *local protocol violation*, not a
        // retryable connection loss — see [`SessionEvent::ProtocolViolation`]'s docs. Emitting
        // `Disconnected{by_peer:false}` here (as this used to) made the driver treat it exactly
        // like a lost socket and retry forever against a server that is simply sending us garbage
        // or actively hostile input.
        if !map_cache::is_valid_map_filename(&name) {
            let reason = "map name is not a valid filename".to_string();
            tracing::warn!(map = %name, %reason, "control: protocol violation on MAP_CHANGE");
            self.disconnect(Some(&reason));
            return vec![SessionEvent::ProtocolViolation { reason }];
        }
        if !(0..=self.config.max_map_size_bytes as i64).contains(&i64::from(size)) {
            let reason = "invalid map size".to_string();
            tracing::warn!(map = %name, size, %reason, "control: protocol violation on MAP_CHANGE");
            self.disconnect(Some(&reason));
            return vec![SessionEvent::ProtocolViolation { reason }];
        }
        tracing::info!(map = %name, crc, size, "control: received MAP_CHANGE");

        let matched = details.filter(|d| d.name == name && d.crc == crc && d.size == size);
        let sha256 = matched.as_ref().map(|d| d.sha256);
        let url = matched.and_then(|d| if d.url.is_empty() { None } else { Some(d.url) });

        // Review finding F10: a heuristic for the "captcha/lobby wall" pattern some public
        // servers use against bots (redirecting to a tiny placeholder map with a name hinting at
        // what it is, before the real map). Not acted on (this project only ever plays
        // 127.0.0.1 — see the crate root docs), just surfaced so a human reviewing logs/`Anomaly`
        // events notices it immediately rather than the bot silently trying to "play" a lobby.
        let lower = name.to_ascii_lowercase();
        let mut anomaly = Vec::new();
        if lower.contains("lobby") || lower.contains("captcha") || lower.contains("verify") {
            anomaly.push(SessionEvent::Anomaly(format!(
                "map name '{name}' looks like a lobby/captcha wall, not a real map"
            )));
        } else if size > 0 && (size as usize) < MIN_PLAUSIBLE_MAP_SIZE_BYTES {
            anomaly.push(SessionEvent::Anomaly(format!(
                "map '{name}' claims a suspiciously small size ({size} bytes) for a real block map"
            )));
        }

        self.state = JoinState::AwaitingMapBytes(PendingMapChange {
            name: name.clone(),
            crc,
            size,
            sha256,
        });
        anomaly.push(SessionEvent::MapChanging {
            name,
            crc,
            size,
            sha256,
            url,
        });
        anomaly
    }

    /// Starts the in-protocol download — `SendMapRequest` (`client.cpp:276-291`), non-sixup path
    /// (chunk `0`).
    fn begin_protocol_download(&mut self, pending: PendingMapChange, now: Duration) {
        self.state = JoinState::Downloading(MapDownload {
            name: pending.name,
            crc: pending.crc,
            size: pending.size,
            sha256: pending.sha256,
            next_chunk: 0,
            buffer: Vec::new(),
        });
        self.send_system_chunk(&SysMsg::RequestMapData { chunk: 0 }, true, now);
    }

    /// One `NETMSG_MAP_DATA` chunk (`client.cpp:1805-1881`, non-sixup path): validated against
    /// the chunk index/CRC/size the download is already tracking, and bounded independently of
    /// whatever the server claims (task acceptance criterion 7).
    fn handle_map_data(&mut self, last: i32, crc: i32, chunk: i32, data: Vec<u8>, now: Duration) -> Vec<SessionEvent> {
        enum Step {
            Rejected,
            Continue(i32),
            Complete { pending: PendingMapChange, bytes: Vec<u8> },
        }

        let step = match &mut self.state {
            // `client.cpp:1832-1836`: the real client also rejects a zero-length chunk outright
            // (`Size <= 0`) — review finding F9. Without this, a hostile/buggy server sending an
            // empty `MAP_DATA{last: 0, ..}` forever would otherwise be silently accepted as
            // "still downloading" indefinitely (never violates the byte-size cap since it adds
            // nothing each time), spinning the download loop forever instead of it either making
            // progress or being dropped.
            JoinState::Downloading(dl)
                if !data.is_empty()
                    && crc == dl.crc
                    && chunk == dl.next_chunk
                    && dl.buffer.len() + data.len() <= self.config.max_map_size_bytes =>
            {
                dl.buffer.extend_from_slice(&data);
                dl.next_chunk += 1;
                if last != 0 {
                    Step::Complete {
                        pending: PendingMapChange {
                            name: dl.name.clone(),
                            crc: dl.crc,
                            size: dl.size,
                            sha256: dl.sha256,
                        },
                        bytes: std::mem::take(&mut dl.buffer),
                    }
                } else {
                    Step::Continue(dl.next_chunk)
                }
            }
            // Wrong chunk index/CRC, growing past the configured size cap, or a stray MAP_DATA
            // outside an active download at all — silently dropped, matching the C++ reference's
            // own `return;` for the first two (`client.cpp:1837`); the size cap is this crate's
            // own addition (task acceptance criterion 7).
            _ => Step::Rejected,
        };

        match step {
            Step::Rejected => Vec::new(),
            Step::Continue(next_chunk) => {
                self.send_system_chunk(&SysMsg::RequestMapData { chunk: next_chunk }, true, now);
                Vec::new()
            }
            Step::Complete { pending, bytes } => match self.finish_map_load(pending, bytes, MapSource::Downloaded, now)
            {
                Ok(ev) => vec![SessionEvent::MapLoaded(ev)],
                // Review finding F4: a map that fails verification (wrong hash/CRC, fails to
                // parse) is a local protocol violation, final — see
                // [`SessionEvent::ProtocolViolation`]'s docs.
                Err(reason) => {
                    self.disconnect(Some(&reason));
                    vec![SessionEvent::ProtocolViolation { reason }]
                }
            },
        }
    }

    /// `LoadMap` (`client.cpp:1229-1291`): verifies `bytes` (sha256 if known, else CRC — DDNet's
    /// own fallback, `client.cpp:1265-1284`) via `ddai_map::load_map` (a pure function — no
    /// filesystem access here, keeping this method itself sans-IO even though its *callers*
    /// exist specifically to bridge the driver's file I/O), and on success sends `NETMSG_READY`.
    fn finish_map_load(
        &mut self,
        pending: PendingMapChange,
        bytes: Vec<u8>,
        source: MapSource,
        now: Duration,
    ) -> Result<MapLoadedEvent, String> {
        // A defensive tightening beyond DDNet's own `LoadMap` (which trusts CRC/sha256 alone):
        // the bytes we are about to trust must also match the size `NETMSG_MAP_CHANGE` declared,
        // whichever source they came from (task acceptance criterion 4's map-size-anomaly spirit).
        if bytes.len() != pending.size as usize {
            return Err(format!(
                "map differs from the server: size {} != {}",
                bytes.len(),
                pending.size
            ));
        }

        let loaded = ddai_map::load_map(&bytes).map_err(|e| format!("failed to load map: {e}"))?;

        if let Some(hint) = pending.sha256 {
            if hint != loaded.sha256 {
                return Err(format!(
                    "map differs from the server: sha256 {} != {}",
                    map_cache::sha256_hex(&loaded.sha256),
                    map_cache::sha256_hex(&hint)
                ));
            }
        } else if loaded.crc32 as i32 != pending.crc {
            return Err(format!(
                "map differs from the server: crc {:08x} != {:08x}",
                loaded.crc32, pending.crc
            ));
        }

        self.state = JoinState::AwaitingConReady;
        self.send_system_chunk(&SysMsg::Ready, true, now);
        tracing::info!(map = %pending.name, source = ?source, "control: map loaded and verified, sent READY");

        let bytes_to_cache = matches!(source, MapSource::Downloaded).then_some(bytes);
        Ok(MapLoadedEvent {
            name: pending.name,
            crc32: loaded.crc32,
            sha256: loaded.sha256,
            source,
            bytes_to_cache,
        })
    }

    // ---- internal: post-map-acquisition join steps --------------------------------------------

    /// `client.cpp:1896-1907` (main-connection branch) + `gameclient.cpp:574-611`
    /// (`CGameClient::OnConnected`, which is what actually sends `Cl_StartInfo` via
    /// `SendInfo(true)`, `gameclient.cpp:603`/`3224-3248`).
    fn handle_con_ready(&mut self, now: Duration) -> Vec<SessionEvent> {
        if !matches!(self.state, JoinState::AwaitingConReady) {
            return Vec::new();
        }
        tracing::info!("control: received CON_READY, sending Cl_StartInfo");
        self.state = JoinState::AwaitingReadyToEnter;
        self.send_start_info(now);
        Vec::new()
    }

    fn send_start_info(&mut self, now: Duration) {
        let payload = build_numbered_game_payload(msgs::id::NETMSGTYPE_CL_STARTINFO, |p| {
            msgs::encode_cl_start_info(
                &msgs::ClStartInfo {
                    name: self.config.name.clone(),
                    clan: self.config.clan.clone(),
                    country: self.config.country,
                    skin: self.config.skin.clone(),
                    use_custom_color: 0,
                    color_body: 0,
                    color_feet: 0,
                },
                p,
            );
        });
        self.send_game_chunk(payload, true, now, "Cl_StartInfo");
    }

    /// `gameclient.cpp:1141-1143` (`Sv_ReadyToEnter` -> `Client()->EnterGame`) +
    /// `client.cpp:514-528` (`EnterGame`) + `client.cpp:472-512` (`OnEnterGame`, the mandatory
    /// `SnapAssembler::reset()`/`InputTiming::reset()` point — task 2.2b's F1 finding).
    fn handle_ready_to_enter(&mut self, now: Duration) -> Vec<SessionEvent> {
        if !matches!(self.state, JoinState::AwaitingReadyToEnter) {
            return Vec::new();
        }
        tracing::info!("control: received Sv_ReadyToEnter, sending NETMSG_ENTERGAME");
        self.send_system_chunk(&SysMsg::EnterGame, true, now);
        self.snap_assembler.reset();
        self.timing.reset();
        self.sent_post_enter_extras = false;
        self.last_snapshot_tick = None;
        self.state = JoinState::InGame;
        // `client.cpp:527`: armed unconditionally on every `EnterGame`, same as a real client
        // reconnecting into a new round — review finding F9.
        self.next_ping_ex_at_ns = Some(to_ns(now) + PING_EX_INITIAL_DELAY_NS);
        vec![SessionEvent::InGame]
    }

    /// `client.cpp:2982-3004` (main-connection branch): periodically originates a `PINGEX` while
    /// in-game, purely informational (server round-trip time) — review finding F9. The real
    /// client falls back to a server-browser "ping via info request" when the server lacks the
    /// `PINGEX` capability; this bot has no server-browser code path at all (out of scope), so
    /// without that capability it simply never originates a ping rather than approximating the
    /// fallback.
    fn maybe_send_ping_ex(&mut self, now: Duration) {
        let Some(next_at_ns) = self.next_ping_ex_at_ns else {
            return;
        };
        let now_ns = to_ns(now);
        if now_ns < next_at_ns {
            return;
        }
        if self.server_capabilities.ping_ex {
            let id = random_bytes_16();
            self.send_ex_system_chunk(&ExSysMsg::PingEx { id }, false, now);
        }
        self.next_ping_ex_at_ns = Some(now_ns + PING_EX_INTERVAL_NS);
    }

    fn handle_snap(&mut self, sys_msg: &SysMsg, now: Duration) -> Vec<SessionEvent> {
        let Some(event) = self.snap_assembler.feed(sys_msg) else {
            return Vec::new();
        };
        match event {
            assembly::Event::Snapshot { tick, snap } => {
                self.timing.on_snapshot(tick, to_ns(now));
                self.last_snapshot_tick = Some(tick);
                let mut events = vec![SessionEvent::Snapshot { tick }];
                if self.config.emit_snapshot_data {
                    events.push(SessionEvent::SnapshotData {
                        tick,
                        snapshot: snap.clone(),
                    });
                }
                if matches!(self.state, JoinState::InGame) && !self.sent_post_enter_extras {
                    let view = View::new(&snap);
                    if view.players().iter().any(|p| p.info.local == 1) {
                        self.send_post_enter_extras(now);
                        self.sent_post_enter_extras = true;
                    }
                }
                events
            }
            assembly::Event::CrcMismatch { tick, crc_errors, .. } => {
                tracing::warn!(tick, crc_errors, "snapshot CRC mismatch");
                if crc_errors > 5 {
                    vec![SessionEvent::Anomaly(format!(
                        "repeated snapshot CRC mismatches ({crc_errors}) around tick {tick}"
                    ))]
                } else {
                    Vec::new()
                }
            }
            assembly::Event::Resync { tick } => {
                tracing::debug!(tick, "server resync (delta base no longer available)");
                Vec::new()
            }
            assembly::Event::DeltaError { tick, error } => {
                tracing::warn!(tick, %error, "snapshot delta decode error");
                Vec::new()
            }
            assembly::Event::Stale { .. } => Vec::new(),
        }
    }

    /// `gameclient.cpp:2289-2303,2392-2407` — sent once, the first time our own player is known
    /// (`PlayerInfo::local == 1`) after entering.
    ///
    /// `Cl_IsDDNetLegacy`'s payload is hand-built here rather than via
    /// `ddai_net::generated::messages::encode_cl_is_dd_net_legacy` — that function is
    /// mechanically correct for `datasrc/network.py`'s *declared* field list (zero fields), but
    /// the real wire message carries one extra hand-appended `i32` (`DDNetVersion()`,
    /// `gameclient.cpp:2299-2301`) that both the real client and server read directly off the raw
    /// packer, bypassing the generated (un)packer entirely (`gamecontext.cpp:2733-2740`,
    /// `OnIsDDNetLegacyNetMessage`) — the exact same situation task 2.2b's F2 finding already
    /// documented and fixed for `Sv_TuneParams`/`Sv_TeamsState`, just missed for this one
    /// client-to-server message (out of scope to fix in `ddai-net` from this task; see the
    /// crate's BUILD REPORT).
    fn send_post_enter_extras(&mut self, now: Duration) {
        let ddnet_version = self.config.ddnet_version;
        let is_ddnet_legacy = build_numbered_game_payload(msgs::id::NETMSGTYPE_CL_ISDDNETLEGACY, |p| {
            p.add_int(ddnet_version);
        });
        self.send_game_chunk(is_ddnet_legacy, true, now, "Cl_IsDDNetLegacy");

        let (x, y) = self.config.show_distance;
        let show_distance = build_ex_game_payload("show-distance@netmsg.ddnet.tw", |p| {
            msgs::encode_cl_show_distance(&msgs::ClShowDistance { x, y }, p);
        });
        self.send_game_chunk(show_distance, true, now, "Cl_ShowDistance");

        // Review finding F9 (`gameclient.cpp:2305-2340,2409-2420`): a real client also sends
        // these three once, right after entering — this bot renders nothing (no camera, no HUD),
        // so rather than tracking that non-existent state it sends the exact values a fresh,
        // default-config real client would: `cl_show_others = 0`, `cl_showhud_spectator_count = 1`
        // (`config_variables.h:669,77`), and `zoom = round_truncate(1.0 * 1000) = 1000` /
        // `deadzone = 0` / `follow_factor = 0` for `cl_default_zoom = 10` (the scale's exact
        // midpoint, i.e. 1.0x/no zoom) with `cl_dyncam = 0` (`camera.cpp:623-631`,
        // `config_variables.h:112,113,171` for the deadzone/followfactor/zoom defaults this
        // mirrors). Purely informational either way (nothing server-side gates on these — see
        // `crate::allowlist`'s docs on why they're allow-listed regardless) — `show` itself is the
        // one field this crate does let a caller override (`ClientConfig::show_others`, review
        // round 1 finding F13): `ddnet-ai record` needs `1`, not the real client's own `0` default,
        // so that a spectate request the server refused (falling back to *playing*) still sees
        // every other player instead of only its own collision group.
        let show_others = build_ex_game_payload("showothers@netmsg.ddnet.tw", |p| {
            msgs::encode_cl_show_others(
                &msgs::ClShowOthers {
                    show: self.config.show_others,
                },
                p,
            );
        });
        self.send_game_chunk(show_others, true, now, "Cl_ShowOthers");

        let enable_spectator_count = build_ex_game_payload("enable-spectator-count@netmsg.ddnet.org", |p| {
            msgs::encode_cl_enable_spectator_count(&msgs::ClEnableSpectatorCount { enable: 1 }, p);
        });
        self.send_game_chunk(enable_spectator_count, true, now, "Cl_EnableSpectatorCount");

        let camera_info = build_ex_game_payload("camera-info@netmsg.ddnet.org", |p| {
            msgs::encode_cl_camera_info(
                &msgs::ClCameraInfo {
                    zoom: 1000,
                    deadzone: 0,
                    follow_factor: 0,
                },
                p,
            );
        });
        self.send_game_chunk(camera_info, true, now, "Cl_CameraInfo");
    }

    fn send_input(&mut self, tick: i32, now: Duration) {
        let ints = player_input_to_ints(&self.current_input);
        let msg = SysMsg::Input {
            ack_game_tick: self.snap_assembler.ack_game_tick(),
            pred_tick: tick,
            size: (ints.len() * 4) as i32,
            data: ints.to_vec(),
        };
        self.send_system_chunk(&msg, false, now);
        // Task 8.4a review round 1, finding F7 / task 2.4 review round 1, finding F5 — both gated
        // by `ClientConfig::emit_input_sent`, both must stay opt-in and cheap-when-off (see
        // `SessionEvent::InputSent`'s doc comment for both tasks' reasoning).
        if self.config.emit_input_sent {
            self.pending_events.push_back(SessionEvent::InputSent {
                tick,
                input: self.current_input,
            });
        }
    }

    // ---- internal: the single outgoing paths (task acceptance criterion 6g) -------------------

    /// Sends a numbered or `ex` **system** message. Never guarded by [`crate::allowlist`]: the
    /// `sys` bit puts every one of these in a namespace disjoint from `NETMSGTYPE_CL_SAY` (see
    /// `crate::allowlist`'s module docs) — this method is only ever called with a fixed,
    /// hand-written set of variants from this file, never with caller-supplied content.
    fn send_system_chunk(&mut self, msg: &SysMsg, vital: bool, now: Duration) {
        let Some(id) = sysmsg_id_of(msg) else {
            debug_assert!(
                false,
                "send_system_chunk called with a SysMsg variant this session never sends: {msg:?}"
            );
            return;
        };
        let mut buf = [0u8; 2048];
        let mut packer = Packer::new(&mut buf);
        uuid::pack_msg_id(&mut packer, MsgId::Numbered(id), true);
        sysmsg::encode(msg, &mut packer);
        if let Err(e) = self.connection.send_chunk(packer.data(), vital, now) {
            tracing::warn!(error = %e, ?msg, "failed to queue outgoing system message");
        }
    }

    fn send_ex_system_chunk(&mut self, msg: &ExSysMsg, vital: bool, now: Duration) {
        let Some(name) = ex_sys_name(msg) else {
            debug_assert!(
                false,
                "send_ex_system_chunk called with an ExSysMsg variant this session never sends: {msg:?}"
            );
            return;
        };
        let mut buf = [0u8; 2048];
        let mut packer = Packer::new(&mut buf);
        let id = uuid::calculate_uuid(name);
        uuid::pack_msg_id(
            &mut packer,
            MsgId::Ex {
                uuid: id,
                resolved: None,
            },
            true,
        );
        message::encode_ex(msg, &mut packer);
        if let Err(e) = self.connection.send_chunk(packer.data(), vital, now) {
            tracing::warn!(error = %e, ?msg, "failed to queue outgoing ex system message");
        }
    }

    /// **The** single outgoing path for game messages (task acceptance criterion 6g): every
    /// caller in this file already only ever builds an allow-listed payload, but `payload` is
    /// checked here anyway, on the raw bytes, exactly as it would check a hostile hand-built
    /// `Cl_Say` — see `crate::allowlist`'s module docs for why this is worth doing even though
    /// the type system already makes it hard to reach this function with the wrong thing.
    fn send_game_chunk(&mut self, payload: Vec<u8>, vital: bool, now: Duration, label: &'static str) {
        match allowlist::check(&payload, &self.registry) {
            Ok(()) => {
                self.log_outgoing(label, true);
                if let Err(e) = self.connection.send_chunk(&payload, vital, now) {
                    tracing::warn!(error = %e, label, "failed to queue outgoing game message");
                }
            }
            Err(e) => {
                tracing::error!(error = %e, label, "BLOCKED an outgoing game message by the allow-list guard");
                self.log_outgoing(label, false);
            }
        }
    }

    fn log_outgoing(&mut self, label: &'static str, accepted: bool) {
        if self.outgoing_log.len() >= OUTGOING_LOG_CAP {
            self.outgoing_log.pop_front();
        }
        self.outgoing_log.push_back(OutgoingLogEntry { label, accepted });
        if self.config.emit_outgoing_audit {
            self.pending_events
                .push_back(SessionEvent::OutgoingGame { label, accepted });
        }
    }

    /// Test-only hook: attempts to send a *hand-built* `Cl_Say` payload through the exact same
    /// single outgoing path every real send in this file goes through — see `tests::` below and
    /// task acceptance criterion 6g's "(1)" bullet ("a test asserts no `Cl_Say` is ever emitted
    /// ... inspect outgoing messages via a test hook").
    #[cfg(test)]
    fn try_send_hand_built_cl_say_for_testing(&mut self, now: Duration) {
        // `msgs::encode_cl_say` is `pub(crate)` inside `ddai-net` (D-007: not reachable from
        // outside that crate at all) — hand-pack the exact same wire shape directly, since the
        // whole point here is a *hand-built* Cl_Say sneaking in through the raw `Packer`/
        // `Connection::send_chunk` path this guard defends (see `crate::allowlist`'s module docs).
        let payload = build_numbered_game_payload(msgs::id::NETMSGTYPE_CL_SAY, |p| {
            p.add_int(0); // team
            p.add_string("this must never reach the wire", 0, true);
        });
        self.send_game_chunk(payload, true, now, "Cl_Say(test-only)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_map::testutil::{MapWriter, TILESLAYERFLAG_GAME, game_layer_data};
    use std::net::UdpSocket;

    fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }
    fn ms(m: u64) -> Duration {
        Duration::from_millis(m)
    }

    /// A tiny, valid `.map` file (game layer only) — enough for `ddai_map::load_map` to accept it.
    fn tiny_map_bytes() -> Vec<u8> {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_info_item(None, None, None, None, &[]);
        w.add_tile_layer(&ddai_map::testutil::TileLayerSpec {
            shape: ddai_map::testutil::TilemapShape::Full,
            item_version: 3,
            width: 4,
            height: 3,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(4, 3),
        });
        w.add_single_group_with_all_layers();
        w.finish()
    }

    /// Drives a `Session` (client role) against a hand-written server-role `Connection` over an
    /// in-memory "wire" (a `Vec<Vec<u8>>` of pending datagrams each way) — the same pattern as
    /// `ddai_net::conn`'s own `tests::handshake`, extended just far enough to exercise the join
    /// sequence this file adds on top: TKEN handshake, `CLIENTVER`/`INFO`, `CAPABILITIES`,
    /// `MAP_DETAILS`+`MAP_CHANGE`, a tiny in-protocol map download, `READY`/`CON_READY`,
    /// `Cl_StartInfo`, `Sv_ReadyToEnter`, `ENTERGAME`, and one `SnapEmpty` snapshot naming our own
    /// player. Never touches a real socket.
    struct FakeServer {
        connection: Connection,
        huffman: Huffman,
        registry: Registry,
    }

    impl FakeServer {
        fn new() -> Self {
            FakeServer {
                connection: Connection::new(conn::Config::default()),
                huffman: Huffman::new(),
                registry: Registry::new(),
            }
        }

        fn accept(&mut self, now: Duration) {
            self.connection.accept(0xcafe_babe, now, &self.huffman);
        }

        /// Feeds one datagram, returning every decoded chunk (ignoring control-only datagrams,
        /// which yield no chunks).
        fn feed(&mut self, datagram: &[u8], now: Duration) -> Vec<Msg> {
            let mut out = Vec::new();
            for ev in self.connection.feed(datagram, &self.huffman, now) {
                if let conn::Event::Chunk { data, .. } = ev {
                    let (msg, _answer) = message::decode(&data, &self.registry);
                    out.push(msg);
                }
            }
            out
        }

        fn flush(&mut self, now: Duration) -> Vec<Vec<u8>> {
            self.connection.flush(&self.huffman, now)
        }

        fn send_ex_sys(&mut self, name: &str, vital: bool, now: Duration, body: impl FnOnce(&mut Packer)) {
            let mut buf = [0u8; 4096];
            let mut packer = Packer::new(&mut buf);
            let id = uuid::calculate_uuid(name);
            uuid::pack_msg_id(
                &mut packer,
                MsgId::Ex {
                    uuid: id,
                    resolved: None,
                },
                true,
            );
            body(&mut packer);
            self.connection
                .send_chunk(packer.data(), vital, now)
                .expect("test: send_chunk");
        }

        fn send_sys_msg(&mut self, msg: &SysMsg, vital: bool, now: Duration) {
            use ddai_net::sysmsg::id;
            let numbered_id = match msg {
                SysMsg::MapChange { .. } => id::MAP_CHANGE,
                SysMsg::MapData { .. } => id::MAP_DATA,
                SysMsg::ConReady => id::CON_READY,
                SysMsg::SnapEmpty { .. } => id::SNAPEMPTY,
                SysMsg::SnapSingle { .. } => id::SNAPSINGLE,
                other => panic!("test helper: no id mapped for {other:?}"),
            };
            let mut buf = [0u8; 4096];
            let mut packer = Packer::new(&mut buf);
            uuid::pack_msg_id(&mut packer, MsgId::Numbered(numbered_id), true);
            sysmsg::encode(msg, &mut packer);
            self.connection
                .send_chunk(packer.data(), vital, now)
                .expect("test: send_chunk");
        }

        fn send_ready_to_enter(&mut self, now: Duration) {
            let mut buf = [0u8; 64];
            let mut packer = Packer::new(&mut buf);
            uuid::pack_msg_id(
                &mut packer,
                MsgId::Numbered(msgs::id::NETMSGTYPE_SV_READYTOENTER),
                false,
            );
            msgs::encode_sv_ready_to_enter(&msgs::SvReadyToEnter {}, &mut packer);
            self.connection
                .send_chunk(packer.data(), true, now)
                .expect("test: send_chunk");
        }
    }

    /// One `Character`+`PlayerInfo` snapshot naming client id 0 as our own (`local == 1`) player
    /// — the minimum a real server sends before this crate's `Cl_IsDDNetLegacy`/`Cl_ShowDistance`
    /// post-enter step fires.
    fn own_player_snapshot() -> ddai_net::snapshot::Snapshot {
        use ddai_net::snapshot::SnapshotItem;
        ddai_net::snapshot::Snapshot {
            items: vec![SnapshotItem {
                key: objects::PlayerInfo::ID << 16,
                data: vec![1, 0, 0, 0, 0], // local=1, client_id=0, team=0, score=0, latency=0
            }],
        }
    }

    /// Drives the *entire* join sequence end to end over an in-memory "wire" (no real socket): TKEN
    /// handshake, `CLIENTVER`/`INFO`, `CAPABILITIES`, `MAP_DETAILS`+`MAP_CHANGE`, a full in-protocol
    /// map download (two chunks, to also exercise the multi-chunk path), `READY`/`CON_READY`,
    /// `Cl_StartInfo`, `Sv_ReadyToEnter`, `ENTERGAME`, and one snapshot naming our own player —
    /// task acceptance criterion 2's whole join sequence, criterion 6g's "no `Cl_Say`" for a
    /// complete real session (not just the isolated unit above), and the `Cl_IsDDNetLegacy`/
    /// `Cl_ShowDistance` post-enter step.
    #[test]
    fn full_join_sequence_reaches_in_game_and_first_snapshot() {
        let mut session = Session::new(ClientConfig::default());
        let mut server = FakeServer::new();
        let mut now = secs(0);

        // 1. TKEN handshake.
        session.connect(now);
        server.accept(now);
        let c2s = session.flush(now);
        assert_eq!(c2s.len(), 1, "expected exactly one CONNECT datagram");
        now += ms(5);
        for dg in &c2s {
            assert!(server.feed(dg, now).is_empty(), "CONNECT carries no chunks");
        }
        let s2c = server.flush(now);
        assert_eq!(s2c.len(), 1, "expected exactly one CONNECTACCEPT datagram");
        now += ms(5);
        let events = session.feed(&s2c[0], now);
        assert_eq!(events, vec![SessionEvent::Connected]);

        // 2. Client -> server: ACCEPT (control) + CLIENTVER, INFO (chunks) — all in the next
        // flush (queued by `Session::on_connected`, drained together with the handshake's own
        // queued ACCEPT).
        let c2s2 = session.flush(now);
        assert!(!c2s2.is_empty());
        now += ms(5);
        let mut seen_clientver = false;
        let mut seen_info = false;
        for dg in &c2s2 {
            for msg in server.feed(dg, now) {
                match msg {
                    Msg::ExSys(ExSysMsg::ClientVer { .. }) => seen_clientver = true,
                    Msg::Sys(SysMsg::Info { netversion, .. }) => {
                        assert_eq!(netversion, "0.6 626fce9a778df4d4");
                        seen_info = true;
                    }
                    other => panic!("unexpected message from client during handshake: {other:?}"),
                }
            }
        }
        assert!(seen_clientver && seen_info, "expected both CLIENTVER and INFO");
        assert!(server.connection.is_online());

        // 3. Server -> client: CAPABILITIES, MAP_DETAILS, MAP_CHANGE.
        let map_bytes = tiny_map_bytes();
        let loaded = ddai_map::load_map(&map_bytes).unwrap();
        let map_name = "tiny".to_string();

        server.send_ex_sys("capabilities@ddnet.tw", true, now, |p| {
            p.add_int(5);
            p.add_int(0x3f);
        });
        server.send_ex_sys("map-details@ddnet.tw", true, now, |p| {
            p.add_string(&map_name, 0, true);
            p.add_raw(&loaded.sha256);
            p.add_int(loaded.crc32 as i32);
            p.add_int(map_bytes.len() as i32);
            p.add_string("", 0, true);
        });
        server.send_sys_msg(
            &SysMsg::MapChange {
                name: map_name.clone(),
                crc: loaded.crc32 as i32,
                size: map_bytes.len() as i32,
            },
            true,
            now,
        );
        let s2c2 = server.flush(now);
        assert!(!s2c2.is_empty());
        now += ms(5);
        let mut map_changing = None;
        for dg in &s2c2 {
            for ev in session.feed(dg, now) {
                if let SessionEvent::MapChanging { .. } = &ev {
                    map_changing = Some(ev);
                }
            }
        }
        match map_changing {
            Some(SessionEvent::MapChanging { name, sha256, .. }) => {
                assert_eq!(name, map_name);
                assert_eq!(sha256, Some(loaded.sha256));
            }
            other => panic!("expected MapChanging, got {other:?}"),
        }
        assert!(session.server_capabilities().sync_weapon_input);

        // 4. flush() (no cache supplied) starts the in-protocol download; feed it back in two
        // chunks to exercise the multi-chunk path.
        let half = map_bytes.len() / 2;
        let download_start = session.flush(now);
        assert!(
            !download_start.is_empty(),
            "flush() must start the download with REQUEST_MAP_DATA(chunk=0)"
        );
        now += ms(5);
        let mut saw_request_chunk_0 = false;
        for dg in &download_start {
            for msg in server.feed(dg, now) {
                if let Msg::Sys(SysMsg::RequestMapData { chunk: 0 }) = msg {
                    saw_request_chunk_0 = true;
                }
            }
        }
        assert!(saw_request_chunk_0);

        server.send_sys_msg(
            &SysMsg::MapData {
                last: 0,
                crc: loaded.crc32 as i32,
                chunk: 0,
                data: map_bytes[..half].to_vec(),
            },
            true,
            now,
        );
        for dg in server.flush(now) {
            let _ = session.feed(&dg, now);
        }
        now += ms(5);
        let chunk1_request = session.flush(now);
        assert!(!chunk1_request.is_empty());
        now += ms(5);
        for dg in &chunk1_request {
            let _ = server.feed(dg, now);
        }

        server.send_sys_msg(
            &SysMsg::MapData {
                last: 1,
                crc: loaded.crc32 as i32,
                chunk: 1,
                data: map_bytes[half..].to_vec(),
            },
            true,
            now,
        );
        let mut map_loaded = None;
        for dg in server.flush(now) {
            for ev in session.feed(&dg, now) {
                if let SessionEvent::MapLoaded(ev) = ev {
                    map_loaded = Some(ev);
                }
            }
        }
        match map_loaded {
            Some(ev) => {
                assert_eq!(ev.source, MapSource::Downloaded);
                assert_eq!(ev.sha256, loaded.sha256);
                assert_eq!(ev.bytes_to_cache.as_deref(), Some(map_bytes.as_slice()));
            }
            None => panic!("expected MapLoaded after the second chunk"),
        }
        now += ms(5);

        // 5. Client sent READY; server answers CON_READY.
        let ready = session.flush(now);
        now += ms(5);
        let mut saw_ready = false;
        for dg in &ready {
            for msg in server.feed(dg, now) {
                if let Msg::Sys(SysMsg::Ready) = msg {
                    saw_ready = true;
                }
            }
        }
        assert!(saw_ready);

        server.send_sys_msg(&SysMsg::ConReady, true, now);
        let mut seen_start_info = false;
        for dg in server.flush(now) {
            let _ = session.feed(&dg, now);
        }
        now += ms(5);
        for dg in session.flush(now) {
            for msg in server.feed(&dg, now) {
                if let Msg::Game(msgs::GameMsg::ClStartInfo(info)) = msg {
                    assert_eq!(info.name, session_default_name());
                    seen_start_info = true;
                }
            }
        }
        assert!(seen_start_info, "expected Cl_StartInfo after CON_READY");

        // 6. Server sends Sv_ReadyToEnter; client sends ENTERGAME.
        now += ms(5);
        server.send_ready_to_enter(now);
        for dg in server.flush(now) {
            let _ = session.feed(&dg, now);
        }
        assert!(session.is_in_game(), "expected InGame after Sv_ReadyToEnter");
        now += ms(5);
        let entergame = session.flush(now);
        now += ms(5);
        let mut saw_entergame = false;
        for dg in &entergame {
            for msg in server.feed(dg, now) {
                if let Msg::Sys(SysMsg::EnterGame) = msg {
                    saw_entergame = true;
                }
            }
        }
        assert!(saw_entergame);

        // 7. One snapshot naming our own player: Cl_IsDDNetLegacy + Cl_ShowDistance must follow.
        let snap = own_player_snapshot();
        let delta_ints =
            ddai_net::delta::create_delta(&ddai_net::snapshot::Snapshot::empty(), &snap, &StaticSizes::ddnet_06())
                .unwrap();
        let mut compressed = vec![0u8; delta_ints.len() * ddai_net::packer::MAX_BYTES_PACKED];
        let n = ddai_net::packer::pack_ints(&mut compressed, &delta_ints).unwrap();
        compressed.truncate(n);
        server.send_sys_msg(
            &SysMsg::SnapSingle {
                tick: 10,
                delta_tick: -1,
                crc: snap.crc() as i32,
                data: compressed,
            },
            false,
            now,
        );
        let mut saw_snapshot_event = false;
        for dg in server.flush(now) {
            for ev in session.feed(&dg, now) {
                if let SessionEvent::Snapshot { tick } = ev {
                    assert_eq!(tick, 10);
                    saw_snapshot_event = true;
                }
            }
        }
        assert!(saw_snapshot_event);

        // A second snapshot bootstraps `InputTiming`'s predicted clock (mirrors the real
        // client's own two-snapshot bootstrap, `client.cpp:2307-2320`) — `NETMSG_INPUT` only
        // starts flowing after this.
        now += ms(5);
        server.send_sys_msg(
            &SysMsg::SnapEmpty {
                tick: 11,
                delta_tick: -1,
            },
            false,
            now,
        );
        for dg in server.flush(now) {
            let _ = session.feed(&dg, now);
        }

        now += ms(5);
        let post_enter = session.flush(now);
        now += ms(5);
        let mut saw_is_ddnet_legacy = false;
        let mut saw_show_distance = false;
        let mut saw_input = false;
        let mut saw_show_others = false;
        let mut saw_enable_spectator_count = false;
        let mut saw_camera_info = false;
        for dg in &post_enter {
            for msg in server.feed(dg, now) {
                match msg {
                    Msg::Game(msgs::GameMsg::ClIsDDNetLegacy(_)) => saw_is_ddnet_legacy = true,
                    Msg::ExGame(msgs::ExGameMsg::ClShowDistance(d)) => {
                        assert_eq!((d.x, d.y), ClientConfig::default().show_distance);
                        saw_show_distance = true;
                    }
                    Msg::ExGame(msgs::ExGameMsg::ClShowOthers(_)) => saw_show_others = true,
                    Msg::ExGame(msgs::ExGameMsg::ClEnableSpectatorCount(_)) => saw_enable_spectator_count = true,
                    Msg::ExGame(msgs::ExGameMsg::ClCameraInfo(_)) => saw_camera_info = true,
                    Msg::Sys(SysMsg::Input { .. }) => saw_input = true,
                    // ClSay must never appear — the whole point of this end-to-end test.
                    Msg::Game(msgs::GameMsg::ClSay(_)) => panic!("Cl_Say was sent on the wire!"),
                    _ => {}
                }
            }
        }
        assert!(
            saw_is_ddnet_legacy,
            "expected Cl_IsDDNetLegacy after the first own-player snapshot"
        );
        assert!(
            saw_show_distance,
            "expected Cl_ShowDistance after the first own-player snapshot"
        );
        assert!(
            saw_show_others,
            "expected Cl_ShowOthers after the first own-player snapshot"
        );
        assert!(
            saw_enable_spectator_count,
            "expected Cl_EnableSpectatorCount after the first own-player snapshot"
        );
        assert!(
            saw_camera_info,
            "expected Cl_CameraInfo after the first own-player snapshot"
        );
        assert!(saw_input, "expected NETMSG_INPUT once in-game");

        // And the audit log confirms it too, independent of the wire inspection above.
        assert!(session.recent_outgoing().all(|e| e.label != "Cl_Say"));
        assert!(
            session
                .recent_outgoing()
                .any(|e| e.label == "Cl_StartInfo" && e.accepted)
        );
        assert!(
            session
                .recent_outgoing()
                .any(|e| e.label == "Cl_IsDDNetLegacy" && e.accepted)
        );
        assert!(
            session
                .recent_outgoing()
                .any(|e| e.label == "Cl_ShowDistance" && e.accepted)
        );
        assert!(
            session
                .recent_outgoing()
                .any(|e| e.label == "Cl_ShowOthers" && e.accepted)
        );
        assert!(
            session
                .recent_outgoing()
                .any(|e| e.label == "Cl_EnableSpectatorCount" && e.accepted)
        );
        assert!(
            session
                .recent_outgoing()
                .any(|e| e.label == "Cl_CameraInfo" && e.accepted)
        );

        // Task 8.4a: `Session::request_team` reaches the *real* wire, over the same connected
        // session the rest of this test already drove through a full join — not just a payload
        // built in isolation (see `request_team_is_logged_as_accepted_and_round_trips` for that).
        now += ms(5);
        session.request_team(-1, now);
        let mut saw_set_team = false;
        for dg in session.flush(now) {
            for msg in server.feed(&dg, now) {
                if let Msg::Game(msgs::GameMsg::ClSetTeam(t)) = msg {
                    assert_eq!(t.team, -1);
                    saw_set_team = true;
                }
            }
        }
        assert!(saw_set_team, "expected Cl_SetTeam(-1) to reach the server");
    }

    fn session_default_name() -> String {
        ClientConfig::default().name
    }

    #[test]
    fn cl_say_never_reaches_the_wire_even_when_directly_forced() {
        let mut session = Session::new(ClientConfig::default());
        session.try_send_hand_built_cl_say_for_testing(secs(0));
        let entries: Vec<_> = session.recent_outgoing().collect();
        assert_eq!(entries.len(), 1);
        assert!(!entries[0].accepted, "the guard must refuse the hand-built Cl_Say");
        assert_eq!(entries[0].label, "Cl_Say(test-only)");
    }

    #[test]
    fn allowed_messages_are_logged_as_accepted() {
        let mut session = Session::new(ClientConfig::default());
        session.send_start_info(secs(0));
        let entries: Vec<_> = session.recent_outgoing().collect();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].accepted);
        assert_eq!(entries[0].label, "Cl_StartInfo");
    }

    /// Task 8.4a: `Session::request_team` is the only builder for `Cl_SetTeam` — this pins down
    /// both that it reaches the allow-list guard as an accepted message and that the payload it
    /// builds decodes back to exactly the team requested (the observer recorder always requests
    /// `TEAM_SPECTATORS = -1`, but the method itself is not spectator-specific).
    #[test]
    fn request_team_is_logged_as_accepted_and_round_trips() {
        let mut session = Session::new(ClientConfig::default());
        session.request_team(-1, secs(0));
        let entries: Vec<_> = session.recent_outgoing().collect();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].accepted);
        assert_eq!(entries[0].label, "Cl_SetTeam");

        let payload = build_numbered_game_payload(msgs::id::NETMSGTYPE_CL_SETTEAM, |p| {
            msgs::encode_cl_set_team(&msgs::ClSetTeam { team: -1 }, p);
        });
        let mut unpacker = ddai_net::packer::Unpacker::new(&payload);
        let registry = Registry::new();
        let (id, sys) = uuid::unpack_msg_id(&mut unpacker, registry.uuids()).expect("decodable id");
        assert!(!sys);
        assert_eq!(id, MsgId::Numbered(msgs::id::NETMSGTYPE_CL_SETTEAM));
        let decoded = msgs::decode_cl_set_team(&mut unpacker).expect("decodable Cl_SetTeam body");
        assert_eq!(decoded.team, -1);
    }

    #[test]
    fn outgoing_log_is_bounded() {
        let mut session = Session::new(ClientConfig::default());
        for _ in 0..(OUTGOING_LOG_CAP * 3) {
            session.send_start_info(secs(0));
        }
        assert_eq!(session.recent_outgoing().count(), OUTGOING_LOG_CAP);
    }

    /// Task 2.4 review round 1, finding F5 / task 8.4a: `Session::send_input` queues a matching
    /// `SessionEvent::InputSent`, delivered through `take_events` — `LiveWorld::predict`'s own
    /// `own_inputs_in_flight` ground truth, and also the ground truth `ddnet-ai play --brain
    /// random-scripted` logs for task 8.4a's input-reconstruction accuracy report.
    #[test]
    fn send_input_is_observable_via_take_events() {
        let mut session = Session::new(ClientConfig {
            emit_input_sent: true,
            ..ClientConfig::default()
        });
        let input = objects::PlayerInput {
            direction: 1,
            target_x: 50,
            target_y: -30,
            jump: 1,
            fire: 0,
            hook: 1,
            player_flags: playerflagflag::PLAYING,
            wanted_weapon: 0,
            next_weapon: 0,
            prev_weapon: 0,
        };
        session.set_input(input);
        session.send_input(777, secs(0));
        let events = session.take_events();
        assert_eq!(events, vec![SessionEvent::InputSent { tick: 777, input }]);
        // Draining once must not repeat the same event on the next call.
        assert_eq!(session.take_events(), Vec::new());
    }

    /// `emit_input_sent` defaults to `false`, and `send_input` must not queue anything at all in
    /// that case (not just "queue it and let the caller ignore it" — the whole point is that a
    /// caller who never reads sent-input ground truth pays nothing for it).
    #[test]
    fn send_input_queues_nothing_when_emit_input_sent_is_off() {
        let mut session = Session::new(ClientConfig::default());
        assert!(!session.config.emit_input_sent, "must default to off");
        session.send_input(1, secs(0));
        assert_eq!(session.take_events(), Vec::new());
    }

    #[test]
    fn server_capabilities_from_version_flags_matches_reference_defaults() {
        let caps = ServerCapabilities::from_version_flags(0, 0);
        assert!(caps.any_player_flag);
        assert!(caps.allow_dummy);
        assert!(!caps.ping_ex);
        assert!(!caps.sync_weapon_input);
        assert!(!caps.chat_timeout_code);
    }

    #[test]
    fn server_capabilities_from_real_local_server_flags() {
        // Bit pattern the real local DDNet 20.1 server actually sends (`server.cpp:1379-1390`):
        // version 5, every flag set.
        let flags = servercapflag::DDNET
            | servercapflag::CHATTIMEOUTCODE
            | servercapflag::ANYPLAYERFLAG
            | servercapflag::PINGEX
            | servercapflag::ALLOWDUMMY
            | servercapflag::SYNCWEAPONINPUT;
        let caps = ServerCapabilities::from_version_flags(5, flags);
        assert!(caps.chat_timeout_code);
        assert!(caps.any_player_flag);
        assert!(caps.ping_ex);
        assert!(caps.allow_dummy);
        assert!(caps.sync_weapon_input);
    }

    /// Review finding F4: a bad `MAP_CHANGE` must be a final [`SessionEvent::ProtocolViolation`],
    /// never the retryable [`SessionEvent::Disconnected`] (which the driver's policy may reconnect
    /// from — see `crate::driver::should_reconnect_after_peer_close`).
    #[test]
    fn map_change_with_invalid_filename_is_a_protocol_violation() {
        let mut session = Session::new(ClientConfig::default());
        session.state = JoinState::AwaitingMapChange;
        let events = session.handle_map_change("../evil".to_string(), 0, 100, secs(0));
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], SessionEvent::ProtocolViolation{reason} if reason.contains("valid filename")));
    }

    #[test]
    fn map_change_with_hostile_size_is_a_protocol_violation() {
        let mut session = Session::new(ClientConfig::default());
        let events = session.handle_map_change("Copy Love Box".to_string(), 0, i32::MAX, secs(0));
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], SessionEvent::ProtocolViolation{reason} if reason.contains("invalid map size")));
    }

    #[test]
    fn map_change_negative_size_is_a_protocol_violation() {
        let mut session = Session::new(ClientConfig::default());
        let events = session.handle_map_change("Copy Love Box".to_string(), 0, -1, secs(0));
        assert!(matches!(&events[0], SessionEvent::ProtocolViolation { .. }));
    }

    /// Review finding F10: a suspiciously tiny claimed size surfaces an [`SessionEvent::Anomaly`]
    /// *ahead of* the (still-delivered) [`SessionEvent::MapChanging`] — informational only, never
    /// blocks the join.
    #[test]
    fn map_change_with_tiny_size_surfaces_anomaly_but_still_proceeds() {
        let mut session = Session::new(ClientConfig::default());
        let events = session.handle_map_change("Copy Love Box".to_string(), 0, 10, secs(0));
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], SessionEvent::Anomaly(msg) if msg.contains("suspiciously small")));
        assert!(matches!(&events[1], SessionEvent::MapChanging { .. }));
    }

    /// Review finding F10: a lobby/captcha-flavoured name is flagged regardless of size.
    #[test]
    fn map_change_with_lobby_like_name_surfaces_anomaly() {
        let mut session = Session::new(ClientConfig::default());
        let events = session.handle_map_change("Player Lobby".to_string(), 0, 1234, secs(0));
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], SessionEvent::Anomaly(msg) if msg.contains("lobby/captcha")));
        assert!(matches!(&events[1], SessionEvent::MapChanging { .. }));
    }

    #[test]
    fn map_change_with_matching_pending_details_surfaces_sha256_and_url() {
        let mut session = Session::new(ClientConfig::default());
        session.pending_map_details = Some(PendingMapDetails {
            name: "Copy Love Box".to_string(),
            sha256: [7u8; 32],
            crc: 42,
            size: 1234,
            url: "https://example.invalid/map".to_string(),
        });
        let events = session.handle_map_change("Copy Love Box".to_string(), 42, 1234, secs(0));
        match &events[0] {
            SessionEvent::MapChanging { sha256, url, .. } => {
                assert_eq!(*sha256, Some([7u8; 32]));
                assert_eq!(url.as_deref(), Some("https://example.invalid/map"));
            }
            other => panic!("expected MapChanging, got {other:?}"),
        }
    }

    #[test]
    fn map_change_with_mismatched_pending_details_ignores_them() {
        let mut session = Session::new(ClientConfig::default());
        session.pending_map_details = Some(PendingMapDetails {
            name: "A different map".to_string(),
            sha256: [7u8; 32],
            crc: 42,
            size: 1234,
            url: String::new(),
        });
        let events = session.handle_map_change("Copy Love Box".to_string(), 42, 1234, secs(0));
        match &events[0] {
            SessionEvent::MapChanging { sha256, url, .. } => {
                assert_eq!(*sha256, None);
                assert_eq!(*url, None);
            }
            other => panic!("expected MapChanging, got {other:?}"),
        }
    }

    #[test]
    fn download_rejects_wrong_chunk_index_and_wrong_crc() {
        let mut session = Session::new(ClientConfig::default());
        session.state = JoinState::Downloading(MapDownload {
            name: "m".to_string(),
            crc: 5,
            size: 0,
            sha256: None,
            next_chunk: 0,
            buffer: Vec::new(),
        });
        assert_eq!(session.handle_map_data(0, 5, 1, vec![1, 2, 3], secs(0)).len(), 0);
        assert_eq!(session.handle_map_data(0, 6, 0, vec![1, 2, 3], secs(0)).len(), 0);
        match &session.state {
            JoinState::Downloading(dl) => assert!(dl.buffer.is_empty(), "rejected chunks must not be accumulated"),
            other => panic!("expected still Downloading, got {other:?}"),
        }
    }

    /// Review finding F9 (`client.cpp:1832-1836`): a zero-length chunk is rejected outright, even
    /// with the right chunk index/CRC and even claiming `last != 0` — never silently "completes" a
    /// download with whatever was accumulated so far, and never advances `next_chunk` either (a
    /// hostile/buggy server repeating this forever must not spin the download loop indefinitely).
    #[test]
    fn download_rejects_empty_chunk_even_marked_last() {
        let mut session = Session::new(ClientConfig::default());
        session.state = JoinState::Downloading(MapDownload {
            name: "m".to_string(),
            crc: 5,
            size: 0,
            sha256: None,
            next_chunk: 0,
            buffer: Vec::new(),
        });
        let events = session.handle_map_data(1, 5, 0, Vec::new(), secs(0));
        assert!(events.is_empty(), "an empty chunk must never complete a download");
        match &session.state {
            JoinState::Downloading(dl) => assert_eq!(dl.next_chunk, 0, "must not advance on a rejected chunk"),
            other => panic!("expected still Downloading, got {other:?}"),
        }
    }

    #[test]
    fn download_rejects_growing_past_the_configured_size_cap() {
        let config = ClientConfig {
            max_map_size_bytes: 4,
            ..ClientConfig::default()
        };
        let mut session = Session::new(config);
        session.state = JoinState::Downloading(MapDownload {
            name: "m".to_string(),
            crc: 5,
            size: 5,
            sha256: None,
            next_chunk: 0,
            buffer: Vec::new(),
        });
        let events = session.handle_map_data(0, 5, 0, vec![1, 2, 3, 4, 5], secs(0));
        assert!(
            events.is_empty(),
            "oversized chunk must be silently dropped, not accepted"
        );
    }

    #[test]
    fn download_completion_verifies_crc_and_loads_the_map() {
        let mut session = Session::new(ClientConfig::default());
        let map_bytes = tiny_map_bytes();
        let loaded = ddai_map::load_map(&map_bytes).unwrap();
        session.state = JoinState::Downloading(MapDownload {
            name: "tiny".to_string(),
            crc: loaded.crc32 as i32,
            size: map_bytes.len() as i32,
            sha256: None,
            next_chunk: 0,
            buffer: Vec::new(),
        });
        let events = session.handle_map_data(1, loaded.crc32 as i32, 0, map_bytes.clone(), secs(0));
        assert_eq!(events.len(), 1);
        match &events[0] {
            SessionEvent::MapLoaded(ev) => {
                assert_eq!(ev.source, MapSource::Downloaded);
                assert_eq!(ev.sha256, loaded.sha256);
                assert_eq!(ev.bytes_to_cache.as_deref(), Some(map_bytes.as_slice()));
            }
            other => panic!("expected MapLoaded, got {other:?}"),
        }
        assert!(matches!(session.state, JoinState::AwaitingConReady));
    }

    #[test]
    fn download_completion_with_wrong_crc_is_a_protocol_violation() {
        let mut session = Session::new(ClientConfig::default());
        let map_bytes = tiny_map_bytes();
        session.state = JoinState::Downloading(MapDownload {
            name: "tiny".to_string(),
            crc: 0xdead,
            size: map_bytes.len() as i32,
            sha256: None,
            next_chunk: 0,
            buffer: Vec::new(),
        });
        let events = session.handle_map_data(1, 0xdead, 0, map_bytes, secs(0));
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], SessionEvent::ProtocolViolation{reason} if reason.contains("differs from the server"))
        );
    }

    #[test]
    fn supply_cached_map_in_wrong_state_is_rejected() {
        let mut session = Session::new(ClientConfig::default());
        let err = session.supply_cached_map(b"whatever", secs(0)).unwrap_err();
        assert_eq!(err, SupplyCachedMapError::WrongState);
    }

    #[test]
    fn supply_cached_map_success_transitions_to_awaiting_con_ready_without_a_download() {
        let mut session = Session::new(ClientConfig::default());
        let map_bytes = tiny_map_bytes();
        let loaded = ddai_map::load_map(&map_bytes).unwrap();
        session.state = JoinState::AwaitingMapBytes(PendingMapChange {
            name: "tiny".to_string(),
            crc: loaded.crc32 as i32,
            size: map_bytes.len() as i32,
            sha256: Some(loaded.sha256),
        });
        let ev = session.supply_cached_map(&map_bytes, secs(0)).unwrap();
        assert_eq!(ev.source, MapSource::Cache);
        assert_eq!(ev.bytes_to_cache, None, "a cache hit must not be re-cached");
        assert!(matches!(session.state, JoinState::AwaitingConReady));
        // And flush() must not have started a protocol download on top of the cache hit: state
        // already moved past `AwaitingMapBytes` before `flush()`'s own check for it ever runs, so
        // no `REQUEST_MAP_DATA` gets queued. The connection is offline in this unit test (no
        // handshake was driven), so `flush()` itself sends nothing at all either way — this only
        // asserts it does not panic and state stays put.
        let datagrams = session.flush(secs(1));
        assert!(datagrams.is_empty());
        assert!(matches!(session.state, JoinState::AwaitingConReady));
    }

    #[test]
    fn supply_cached_map_with_wrong_hash_falls_back_leaving_state_awaiting_map_bytes() {
        let mut session = Session::new(ClientConfig::default());
        let map_bytes = tiny_map_bytes();
        session.state = JoinState::AwaitingMapBytes(PendingMapChange {
            name: "tiny".to_string(),
            crc: 0,
            size: map_bytes.len() as i32,
            sha256: Some([0xffu8; 32]), // deliberately wrong
        });
        let err = session.supply_cached_map(&map_bytes, secs(0)).unwrap_err();
        assert!(matches!(err, SupplyCachedMapError::VerifyFailed(_)));
        assert!(
            matches!(session.state, JoinState::AwaitingMapBytes(_)),
            "must fall back, not get stuck disconnected"
        );
    }

    #[test]
    fn flush_starts_the_protocol_download_when_no_cache_was_supplied() {
        let mut session = Session::new(ClientConfig::default());
        session.connect(secs(0));
        // Force straight to Online so `flush()` actually emits data (bypassing the real
        // handshake, which is covered end-to-end elsewhere).
        // (Session has no public "force online" hook — drive a real handshake instead.)
        let mut server = FakeServer::new();
        server.accept(secs(0));
        for dg in session.flush(secs(0)) {
            let _ = server.feed(&dg, secs(0));
        }
        for dg in server.flush(secs(0)) {
            let _ = session.feed(&dg, secs(1));
        }
        for dg in session.flush(secs(1)) {
            let _ = server.feed(&dg, secs(1));
        }
        assert!(!session.is_in_game());

        session.state = JoinState::AwaitingMapBytes(PendingMapChange {
            name: "Copy Love Box".to_string(),
            crc: 123,
            size: 10,
            sha256: None,
        });
        let datagrams = session.flush(secs(2));
        assert!(
            !datagrams.is_empty(),
            "flush() must start the download by sending REQUEST_MAP_DATA"
        );
        assert!(matches!(session.state, JoinState::Downloading(_)));
    }

    #[test]
    fn ready_to_enter_outside_awaiting_state_is_ignored() {
        let mut session = Session::new(ClientConfig::default());
        let events = session.handle_ready_to_enter(secs(0));
        assert!(events.is_empty());
        assert!(!session.is_in_game());
    }

    #[test]
    fn con_ready_outside_awaiting_state_is_ignored() {
        let mut session = Session::new(ClientConfig::default());
        let events = session.handle_con_ready(secs(0));
        assert!(events.is_empty());
    }

    #[test]
    fn enter_game_resets_snap_assembler_and_timing() {
        let mut session = Session::new(ClientConfig::default());
        session.state = JoinState::AwaitingReadyToEnter;
        // Feed some snapshot state in first, to prove it gets wiped.
        session.snap_assembler.feed(&SysMsg::SnapEmpty {
            tick: 5000,
            delta_tick: -1,
        });
        assert_eq!(session.ack_game_tick(), 5000);
        let events = session.handle_ready_to_enter(secs(0));
        assert_eq!(events, vec![SessionEvent::InGame]);
        assert_eq!(session.ack_game_tick(), -1, "SnapAssembler must be reset on ENTERGAME");
        assert!(session.is_in_game());
    }

    /// Review finding F9 (`client.cpp:527,2982-3004`): originates a `PINGEX` 0.5s after entering,
    /// but only once the server has advertised the capability — and not a moment before the delay.
    #[test]
    fn ping_ex_is_originated_after_entering_game_when_capability_present() {
        let (mut session, mut server, mut now) = online_session_and_server();
        session.server_capabilities = ServerCapabilities {
            ping_ex: true,
            ..ServerCapabilities::default()
        };
        session.state = JoinState::AwaitingReadyToEnter;
        session.handle_ready_to_enter(now);

        let mut saw_ping_ex = false;
        for dg in session.flush(now) {
            for msg in server.feed(&dg, now) {
                if matches!(msg, Msg::ExSys(ExSysMsg::PingEx { .. })) {
                    saw_ping_ex = true;
                }
            }
        }
        assert!(!saw_ping_ex, "must not fire immediately on ENTERGAME");

        now += ms(400);
        for dg in session.flush(now) {
            for msg in server.feed(&dg, now) {
                if matches!(msg, Msg::ExSys(ExSysMsg::PingEx { .. })) {
                    saw_ping_ex = true;
                }
            }
        }
        assert!(!saw_ping_ex, "must not fire before the 0.5s delay");

        now += ms(200); // 600ms since ENTERGAME, past the 500ms delay
        for dg in session.flush(now) {
            for msg in server.feed(&dg, now) {
                if matches!(msg, Msg::ExSys(ExSysMsg::PingEx { .. })) {
                    saw_ping_ex = true;
                }
            }
        }
        assert!(
            saw_ping_ex,
            "must originate a PINGEX once past the delay, with the capability present"
        );
    }

    #[test]
    fn ping_ex_is_not_originated_without_the_capability() {
        let (mut session, mut server, mut now) = online_session_and_server();
        // `server_capabilities` defaults to `ping_ex: false` — no capabilities message was sent.
        session.state = JoinState::AwaitingReadyToEnter;
        session.handle_ready_to_enter(now);
        now += secs(1); // well past the 0.5s delay
        let mut saw_ping_ex = false;
        for dg in session.flush(now) {
            for msg in server.feed(&dg, now) {
                if matches!(msg, Msg::ExSys(ExSysMsg::PingEx { .. })) {
                    saw_ping_ex = true;
                }
            }
        }
        assert!(
            !saw_ping_ex,
            "without the PINGEX capability this bot must not originate one"
        );
    }

    /// Builds the raw bytes a well-formed `checksum-request@ddnet.tw`'s `Start`/`Length` prefix
    /// would pack as, for the two tests below.
    fn packed_two_ints(a: i32, b: i32) -> Vec<u8> {
        let mut buf = [0u8; 32];
        let mut packer = Packer::new(&mut buf);
        packer.add_int(a);
        packer.add_int(b);
        packer.data().to_vec()
    }

    /// Drives a session to `Online` (TKEN handshake only, no map join needed) so its outgoing
    /// chunks can actually be decoded by a `FakeServer` — shared setup for the two checksum tests.
    fn online_session_and_server() -> (Session, FakeServer, Duration) {
        let mut session = Session::new(ClientConfig::default());
        let mut server = FakeServer::new();
        let mut now = secs(0);
        session.connect(now);
        server.accept(now);
        for dg in session.flush(now) {
            let _ = server.feed(&dg, now);
        }
        now += ms(5);
        for dg in server.flush(now) {
            let _ = session.feed(&dg, now);
        }
        now += ms(5);
        for dg in session.flush(now) {
            let _ = server.feed(&dg, now);
        }
        assert!(server.connection.is_online(), "test setup: handshake must complete");
        (session, server, now)
    }

    /// Review finding F14 (round 2 — round 1 used error `2` here, which is simply wrong: `2`
    /// means "the bounds check itself failed", never "valid bounds but we can't answer"):
    /// a well-formed `Start`/`Length` body gets error `3` ("own executable not available",
    /// `client.cpp:4436` — a real, named condition the real client itself can hit, and honestly
    /// true for this bot, which has no such executable to hash at all).
    #[test]
    fn checksum_request_with_well_formed_body_gets_error_3() {
        let (mut session, mut server, mut now) = online_session_and_server();
        let events = session.handle_ex_sys(
            ExSysMsg::ChecksumRequest {
                uuid: [1u8; 16],
                rest: packed_two_ints(0, 16),
            },
            true,
            now,
        );
        assert!(events.is_empty());
        now += ms(5);
        let mut seen_error = None;
        for dg in session.flush(now) {
            for msg in server.feed(&dg, now) {
                if let Msg::ExSys(ExSysMsg::ChecksumError { error, .. }) = msg {
                    seen_error = Some(error);
                }
            }
        }
        assert_eq!(seen_error, Some(3));
    }

    /// Review finding F14: a well-formed body that actually fails `HandleChecksum`'s own bounds
    /// check (`Start<0 || Length<0 || Start > INT_MAX-Length`) still gets error `2`, not `3` —
    /// `Length` alone being negative is enough, independent of `Start`.
    #[test]
    fn checksum_request_with_out_of_bounds_body_gets_error_2() {
        let (mut session, mut server, mut now) = online_session_and_server();
        let events = session.handle_ex_sys(
            ExSysMsg::ChecksumRequest {
                uuid: [1u8; 16],
                rest: packed_two_ints(0, -1),
            },
            true,
            now,
        );
        assert!(events.is_empty());
        now += ms(5);
        let mut seen_error = None;
        for dg in session.flush(now) {
            for msg in server.feed(&dg, now) {
                if let Msg::ExSys(ExSysMsg::ChecksumError { error, .. }) = msg {
                    seen_error = Some(error);
                }
            }
        }
        assert_eq!(seen_error, Some(2));
    }

    /// Review finding F9/F14: a body that doesn't even contain two ints gets error `1` ("the
    /// request itself failed to unpack"), matching `HandleChecksum`'s own first check.
    #[test]
    fn checksum_request_with_malformed_body_gets_error_1() {
        let (mut session, mut server, mut now) = online_session_and_server();
        let events = session.handle_ex_sys(
            ExSysMsg::ChecksumRequest {
                uuid: [1u8; 16],
                rest: vec![],
            },
            true,
            now,
        );
        assert!(events.is_empty());
        now += ms(5);
        let mut seen_error = None;
        for dg in session.flush(now) {
            for msg in server.feed(&dg, now) {
                if let Msg::ExSys(ExSysMsg::ChecksumError { error, .. }) = msg {
                    seen_error = Some(error);
                }
            }
        }
        assert_eq!(seen_error, Some(1));
    }

    #[test]
    fn redirect_with_valid_port_emits_event() {
        let mut session = Session::new(ClientConfig::default());
        let events = session.handle_ex_sys(ExSysMsg::Redirect { port: 8304 }, true, secs(0));
        assert_eq!(events, vec![SessionEvent::RedirectRequested { port: 8304 }]);
    }

    #[test]
    fn redirect_with_invalid_port_is_ignored() {
        let mut session = Session::new(ClientConfig::default());
        assert!(
            session
                .handle_ex_sys(ExSysMsg::Redirect { port: -1 }, true, secs(0))
                .is_empty()
        );
        assert!(
            session
                .handle_ex_sys(ExSysMsg::Redirect { port: 0 }, true, secs(0))
                .is_empty()
        );
        assert!(
            session
                .handle_ex_sys(ExSysMsg::Redirect { port: 70000 }, true, secs(0))
                .is_empty()
        );
    }

    #[test]
    fn reconnect_emits_event() {
        let mut session = Session::new(ClientConfig::default());
        assert_eq!(
            session.handle_ex_sys(ExSysMsg::Reconnect, true, secs(0)),
            vec![SessionEvent::ReconnectRequested]
        );
    }

    // ---- review finding F9: vital-gating for MAP_CHANGE/CON_READY/MAP_DETAILS/CAPABILITIES ----

    #[test]
    fn map_change_from_non_vital_chunk_is_ignored() {
        let mut session = Session::new(ClientConfig::default());
        session.state = JoinState::AwaitingMapChange;
        let events = session.handle_sys(
            SysMsg::MapChange {
                name: "Copy Love Box".to_string(),
                crc: 0,
                size: 100,
            },
            false,
            secs(0),
        );
        assert!(events.is_empty());
        assert!(
            matches!(session.state, JoinState::AwaitingMapChange),
            "state must not advance on a non-vital MAP_CHANGE"
        );
    }

    #[test]
    fn con_ready_from_non_vital_chunk_is_ignored() {
        let mut session = Session::new(ClientConfig::default());
        session.state = JoinState::AwaitingConReady;
        let events = session.handle_sys(SysMsg::ConReady, false, secs(0));
        assert!(events.is_empty());
        assert!(matches!(session.state, JoinState::AwaitingConReady));
    }

    #[test]
    fn map_details_from_non_vital_chunk_is_ignored() {
        let mut session = Session::new(ClientConfig::default());
        let events = session.handle_ex_sys(
            ExSysMsg::MapDetails {
                name: "m".to_string(),
                sha256: [0u8; 32],
                crc: 0,
                size: 0,
                url: String::new(),
            },
            false,
            secs(0),
        );
        assert!(events.is_empty());
        assert!(session.pending_map_details.is_none());
    }

    #[test]
    fn capabilities_from_non_vital_chunk_is_ignored() {
        let mut session = Session::new(ClientConfig::default());
        let events = session.handle_ex_sys(
            ExSysMsg::Capabilities {
                version: 5,
                flags: 0x3f,
            },
            false,
            secs(0),
        );
        assert!(events.is_empty());
        assert_eq!(session.server_capabilities(), ServerCapabilities::default());
    }

    #[test]
    fn garbage_datagrams_never_panic_the_session() {
        let mut session = Session::new(ClientConfig::default());
        session.connect(secs(0));
        for len in 0..64 {
            let garbage = vec![0xffu8; len];
            let _ = session.feed(&garbage, secs(1)); // must not panic
        }
        let _ = session.flush(secs(2));
    }

    /// A tiny, deterministic, dependency-free xorshift PRNG — plenty for "generate a lot of
    /// varied garbage", no need for the `rand` crate just for this.
    struct Xorshift(u64);
    impl Xorshift {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_bytes(&mut self, len: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(len);
            while out.len() < len {
                out.extend_from_slice(&self.next_u64().to_le_bytes());
            }
            out.truncate(len);
            out
        }
    }

    /// Task acceptance criterion 7: "session fuzz (malformed/hostile server messages: huge map
    /// sizes, bad CRC, bogus redirects, snapshot garbage) → errors/events, never panics, bounded
    /// memory". Drives a real (in-memory) handshake to `Online` first — so the fuzzed bytes
    /// actually reach `message::decode` and every handler beyond it, not just `Connection`'s own
    /// packet-framing checks (already fuzzed independently by `ddai-net`'s own test suite) — then
    /// fires 2000 random-length, random-content vital chunks at the client, interleaved with a
    /// handful of *structured but boundary-valued* hostile messages (extreme redirect ports;
    /// Unicode/garbage/reserved-device `MAP_CHANGE` names — review finding F1 — each on their own
    /// disposable session, since those are expected to end the session) built with this file's
    /// own encoders. Passing
    /// means "did not panic and did not grow unboundedly" — checked by simply completing at all
    /// (a real panic aborts the test) and by asserting the session's own bounded structures
    /// (outgoing log, margin history) never exceed their documented caps.
    #[test]
    fn session_fuzz_never_panics_and_stays_bounded() {
        let mut session = Session::new(ClientConfig::default());
        let mut server = FakeServer::new();
        let mut now = secs(0);

        session.connect(now);
        server.accept(now);
        for dg in session.flush(now) {
            let _ = server.feed(&dg, now);
        }
        now += ms(5);
        for dg in server.flush(now) {
            let _ = session.feed(&dg, now);
        }
        now += ms(5);
        for dg in session.flush(now) {
            let _ = server.feed(&dg, now);
        }
        assert!(
            server.connection.is_online(),
            "fuzz setup: handshake must complete first"
        );

        let mut rng = Xorshift(0x9E3779B97F4A7C15);

        // Structured-but-hostile messages that do *not* terminate the connection (unlike a
        // hostile `MAP_CHANGE`, e.g. bad map name/size — already covered on their own by
        // `map_change_with_invalid_filename_disconnects`/`map_change_with_hostile_size_disconnects`
        // above; sending one here would just move the session to `Offline` a few lines into this
        // test and leave the rest of it fuzzing an already-disconnected session, testing far less
        // than intended), interleaved with pure random garbage below — the session must stay
        // `Online` throughout so the random garbage keeps actually reaching `message::decode`.
        let hostile_ex: Vec<(&str, i32)> = vec![
            ("redirect@ddnet.org", i32::MAX),
            ("redirect@ddnet.org", i32::MIN),
            ("redirect@ddnet.org", -1),
            ("redirect@ddnet.org", 0),
        ];
        for (name, port) in &hostile_ex {
            server.send_ex_sys(name, true, now, |p| p.add_int(*port));
        }
        for dg in server.flush(now) {
            let _ = session.feed(&dg, now); // must not panic on any of the above
        }
        assert!(
            server.connection.is_online(),
            "boundary-value redirects must not disconnect the session"
        );

        // Review finding F1: structured Unicode/garbage `MAP_CHANGE` names — the actual bug class
        // (a panic inside `map_cache::is_valid_map_filename` the instant a reserved-name-length
        // byte offset landed inside a multi-byte UTF-8 character) that started this finding. Each
        // case gets its own fresh, disposable session/server pair rather than reusing the long-
        // lived `session`/`server` above: a hostile `MAP_CHANGE` ends the session (matching
        // `map_change_with_invalid_filename_is_a_protocol_violation` et al. above), which would
        // otherwise cut the *main* random-garbage loop below short — see that loop's own comment.
        // Passing means simply "did not panic" (a real panic aborts the whole test either way).
        let hostile_map_change_names: &[&str] = &[
            "Blöck",          // the exact real-world name that used to panic (F1)
            "日本語マップ",   // multi-byte, no ASCII prefix at all
            "🎮 Block Party", // a 4-byte-per-codepoint emoji
            "../evil",        // path traversal — must still just be rejected, not panic
            "CON",            // reserved device name
            "CoöN",           // reserved-prefix collision candidate with a multi-byte character right after it
            "a\0b",           // embedded NUL
            "",               // empty name
            "аб",             // 2-byte Cyrillic, exactly `reserved.len()`-sized prefixes for "CON" etc.
        ];
        for name in hostile_map_change_names {
            let mut fresh_session = Session::new(ClientConfig::default());
            let mut fresh_server = FakeServer::new();
            let mut t = secs(0);
            fresh_session.connect(t);
            fresh_server.accept(t);
            for dg in fresh_session.flush(t) {
                let _ = fresh_server.feed(&dg, t);
            }
            t += ms(5);
            for dg in fresh_server.flush(t) {
                let _ = fresh_session.feed(&dg, t); // must not panic
            }
            t += ms(5);
            for dg in fresh_session.flush(t) {
                let _ = fresh_server.feed(&dg, t);
            }
            fresh_server.send_sys_msg(
                &SysMsg::MapChange {
                    name: name.to_string(),
                    crc: 0,
                    size: 100,
                },
                true,
                t,
            );
            for dg in fresh_server.flush(t) {
                let _ = fresh_session.feed(&dg, t); // the actual F1 regression check
            }
            let _ = fresh_session.flush(t); // must not panic either
        }

        for _ in 0..2000 {
            now += ms(1);
            let len = (rng.next_u64() % 300) as usize;
            let garbage = rng.next_bytes(len);
            let vital = rng.next_u64().is_multiple_of(2);
            // Not every random buffer is even a well-formed *packet* — feed it both as a raw
            // datagram (exercises `Connection`'s own framing) and, when it happens to still
            // decode as a packet, whatever chunk(s) that produced already went through
            // `Session::feed` for real inside that same call. Separately, also queue it directly
            // as an already-framed chunk server-side so it reaches `message::decode` even when
            // the random bytes would never have framed as a valid packet on their own.
            let _ = session.feed(&garbage, now);
            let _ = server
                .connection
                .send_chunk(&garbage[..garbage.len().min(1023)], vital, now);
            for dg in server.flush(now) {
                let _ = session.feed(&dg, now);
            }
            let _ = session.flush(now);
        }

        // Bounded memory: the session's own bookkeeping never grows past its documented caps,
        // no matter how much hostile input it was just fed.
        assert!(session.recent_outgoing().count() <= OUTGOING_LOG_CAP);
        let margin = session.margin_summary();
        assert!(margin.count <= 2000 + hostile_ex.len() as u64 + 10);
    }

    #[test]
    fn unbound_udp_socket_smoke_test_that_the_test_module_itself_can_touch_a_real_socket() {
        // Sanity check for `tests/e2e_local_server.rs`'s reuse of this crate's real-socket
        // patterns (review finding F3 — that file didn't exist yet when this comment was
        // written): not itself a session test, just confirms the environment allows binding
        // loopback UDP (some sandboxes do not) before other tests build on that assumption.
        let socket = UdpSocket::bind("127.0.0.1:0");
        assert!(
            socket.is_ok(),
            "binding an ephemeral loopback UDP socket must succeed in this environment"
        );
    }
}
