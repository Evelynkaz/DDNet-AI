//! The real-time driver: owns an actual `UdpSocket` and a background thread, drives a
//! [`crate::session::Session`], and implements everything the sans-IO session cannot do itself —
//! map-cache file I/O, and the live-play policy from `CLAUDE.md`/decision D-016:
//!
//! - **Connection-rate limiting**: never more than [`MAX_ATTEMPTS_PER_WINDOW`] connection attempts
//!   to the same `SocketAddr` within [`ATTEMPT_WINDOW`] — enforced by [`wait_for_attempt_slot`]
//!   against a process-wide table, so it holds across every [`Client`] in this process, not just
//!   reconnects within one.
//! - **Exponential backoff** ([`MIN_BACKOFF`] → [`MAX_BACKOFF`]) after a *lost* connection
//!   (timeout/protocol error) before reconnecting.
//! - **Redirect loop protection**: follows `redirect@ddnet.org` at most once per [`Client::connect`]
//!   call.
//! - **Reconnect-loop protection** (task 2.3b): follows `reconnect@ddnet.org` at most
//!   [`MAX_SERVER_RECONNECTS_BEFORE_IN_GAME`] time(s) before the session has been in game, then
//!   closes gracefully and gives up ([`GaveUpCategory::ReconnectLoop`]). Every request is logged.
//! - **Handshake watchdog** (task 2.3b): not in game within [`DEFAULT_HANDSHAKE_TIMEOUT`] (15 s,
//!   [`crate::session::ClientConfig::handshake_timeout`]) of connecting -> graceful `CLOSE` and
//!   give up ([`GaveUpCategory::HandshakeTimeout`]). Never a silent retry.
//! - **One socket per [`Client::connect`]**: every reconnect/redirect reuses the same local UDP
//!   port, like the real client.
//! - **Never auto-reconnects after a kick/ban** — a peer-initiated `NETMSG_CLOSE`
//!   ([`crate::session::SessionEvent::Disconnected`] with `by_peer: true`) is final *unless*
//!   [`should_reconnect_after_peer_close`] classifies its reason as transient (a graceful server
//!   restart, being full, …; review finding F2 — see that function's own docs for the exact
//!   policy and its citations). When final, the thread simply ends; the caller sees the terminal
//!   event and decides what to do next.
//!
//! [`Client`] is the crate's public entry point: `Client::connect(addr, config)`, events via
//! [`Client::try_recv_event`]/[`Client::recv_event`]/[`Client::events`], `Client::set_input`,
//! `Client::disconnect` — task acceptance criterion 1's public API.

use crate::map_cache;
use crate::session::{ClientConfig, Session, SessionEvent};
use crate::timing::MarginSummary;
use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects::PlayerInput;
use std::collections::{HashMap, VecDeque};
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

/// CLAUDE.md/D-016: never more than this many connection attempts to the same server within
/// [`ATTEMPT_WINDOW`].
pub const MAX_ATTEMPTS_PER_WINDOW: usize = 5;
pub const ATTEMPT_WINDOW: Duration = Duration::from_secs(20);
/// Backoff after a lost connection (not a redirect/reconnect-request, which are not failures).
pub const MIN_BACKOFF: Duration = Duration::from_secs(1);
pub const MAX_BACKOFF: Duration = Duration::from_secs(30);
/// Task 2.3b's handshake watchdog default — see [`crate::session::ClientConfig::handshake_timeout`]'s
/// docs for the incident this defends against. 15s comfortably covers a real, healthy join (a fresh
/// map download over loopback takes low hundreds of ms — `docs/formats.md` §14.11) while being far
/// short of anything a human waiting on a stalled session would tolerate before wondering what's
/// wrong.
pub const DEFAULT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);
/// Task 2.3b: absolute upper bound for the handshake watchdog, however much map-download progress
/// keeps extending it (the deadline moves out by `handshake_timeout` each time new map bytes arrive,
/// never past this cap measured from the start of the join / from an in-game loss).
pub const HANDSHAKE_HARD_CAP: Duration = Duration::from_secs(120);
/// Task 2.3b (review F1): total connection attempts allowed before the session has ever reached in
/// game, whatever ended the earlier ones (a server-requested reconnect, a redirect, a peer `CLOSE`
/// with a retry reason such as "This server is full", a timeout). Exceeding it is a graceful
/// `CLOSE` and `GaveUp(TooManyAttempts)`. After the first in-game the cap no longer applies (the
/// reconnect-after-loss policy and its watchdog budget take over).
pub const MAX_ATTEMPTS_BEFORE_FIRST_IN_GAME: u32 = 2;
/// Task 2.3b: how often a join that has not completed logs which step it is waiting in.
const JOIN_PROGRESS_LOG_INTERVAL: Duration = Duration::from_secs(1);
/// Task 2.3b: how many `reconnect@ddnet.org` requests the driver follows before the session has
/// ever reached in game (the counter resets on [`crate::session::SessionEvent::InGame`]). One is
/// what a healthy server that wants a single bounce needs; a second one means the peer bounces us
/// every time and the honest reaction is to stop and leave an explanation, not to loop. This is
/// the same budget the driver already applies to `redirect@ddnet.org` (one per
/// [`Client::connect`]).
pub const MAX_SERVER_RECONNECTS_BEFORE_IN_GAME: u32 = 1;
/// How often the various wait loops below (attempt-slot queueing, backoff sleeps) wake up to
/// re-check the handshake watchdog even while otherwise idle — see [`run`]'s own use of this.
const WATCHDOG_POLL_INTERVAL: Duration = Duration::from_millis(200);
/// How long `recv` blocks before the driver loop re-checks its channels/timers — this is the
/// effective granularity of [`crate::timing::InputTiming::advance`]'s cadence (see that module's
/// docs: a faster poll only lets the predicted tick advance *sooner* within its 20ms window, it
/// cannot advance further than the wall clock allows either way).
const POLL_TIMEOUT: Duration = Duration::from_millis(10);
/// Receive buffer size — comfortably above `ddai_net::packet::MAX_PACKET_SIZE` (1400).
const RECV_BUF_SIZE: usize = 2048;

/// Review finding F2 (round 2 — the round-1 substring match was unsound: an admin's free kick/ban
/// text containing e.g. "full"/"timeout"/"redirected" as an incidental word — `"Kicked (server
/// full of noobs)"`, `"You have been banned for 5 minutes (Timeout farming)"` — would have
/// reconnected from an actual kick/ban, violating D-037/CLAUDE.md's "never auto-reconnect after a
/// kick/ban" outright): whether a peer-initiated close
/// (`SessionEvent::Disconnected{by_peer: true, reason}`) should be reconnected from.
///
/// Two-step, in this exact order:
/// 1. **Kick/ban/VPN prefixes/keywords are checked first and win unconditionally** (case
///    insensitive): a reason starting with "Kicked" or "You have been banned"
///    (`server.cpp:3776,3781`/`netban.h:215`'s `MakeBanInfo`, `str_copy(aBuf, "You have been
///    banned")`), or containing "VPN" or "ban" anywhere, is *always* final — checked before any
///    reconnect-worthy text below, specifically so admin-authored free text embedding one of
///    those words never slips through.
/// 2. Only once step 1 didn't match: reconnect-worthy for a small set of *exact* 20.1 wire texts
///    (never a loose substring beyond this point either, for the same reason as step 1) —
///    `"Server shutdown"` (`server.cpp:3737`, a deliberate restart), `"This server is full"`
///    (`network_server.cpp:282`/`server.cpp:2105` — also covers a reserved-slot rejection without
///    auth, `server.cpp:2102-2106`: 20.1 sends this exact same string for that case too, no
///    distinct "reserved" wire text exists in this pinned rev), `"Timeout"`
///    (`network_conn.cpp:523-526`), `"Timeout Protection over"` (`network_conn.cpp:509-510`),
///    `"redirected"` (`server.cpp:3674`), or a reason starting with `"Too weak connection (not
///    acked for "` (`network_conn.cpp:534-537`, the count embeds `cl_reconnect_timeout`'s live
///    value so this is a prefix, not a full equality). Everything else is final.
///
/// This is a deliberate *bot-specific* policy, not literal parity with the real client's own
/// reconnect logic (`client.cpp:451-456`, which only ever auto-reconnects on "full"/"reserved" or
/// "Timeout"/"Too weak connection" substrings, unconditionally — it has no kick/ban carve-out at
/// all, because a human is expected to be watching and decide by hand either way). This bot has
/// nobody watching it live, so unlike the real client it also treats a graceful server restart and
/// a declined redirect as transient — but never at the cost of the kick/ban carve-out D-037/
/// CLAUDE.md's live-play policy requires. Any reason this doesn't specifically recognise as
/// reconnect-worthy is final: a human reviewing `docs/STATUS.md`/logs can always restart the bot
/// by hand; silently retrying against an actual ban forever cannot be undone by a human noticing
/// later.
pub fn should_reconnect_after_peer_close(reason: &str) -> bool {
    let lower = reason.to_ascii_lowercase();
    if lower.starts_with("kicked")
        || lower.starts_with("you have been banned")
        || lower.contains("vpn")
        || lower.contains("ban")
    {
        return false;
    }
    reason == "Server shutdown"
        || reason == "This server is full"
        || reason == "Timeout"
        || reason == "Timeout Protection over"
        || reason == "redirected"
        || reason.starts_with("Too weak connection (not acked for ")
}

#[cfg(test)]
mod reconnect_classification_tests {
    use super::should_reconnect_after_peer_close;

    /// Review finding F2 (round 2): a table pinning down the classification for every real
    /// disconnect-reason string cited in `should_reconnect_after_peer_close`'s docs, its
    /// case-insensitivity, a couple of reasons that must stay final (an unrecognised string, the
    /// empty string a peer `CLOSE` with no reason at all produces), and — the actual regression
    /// this round fixes — adversarial admin-authored kick/ban text that happens to *contain* one
    /// of the reconnect-worthy words: the kick/ban/VPN/"ban" check must win over all of them.
    #[test]
    fn classifies_every_cited_reason_string_correctly() {
        let reconnect_worthy = [
            "Timeout Protection over",
            "Too weak connection (not acked for 10 seconds)",
            "This server is full",
            "Server shutdown",
            "redirected",
            "Timeout",
            // Case-insensitivity of the exact-match texts themselves is *not* required — only
            // the kick/ban/VPN/"ban" carve-out is case-insensitive per the algorithm (see the
            // `SERVER SHUTDOWN`/`TIMEOUT` cases moved to `final_reasons` below).
        ];
        for reason in reconnect_worthy {
            assert!(
                should_reconnect_after_peer_close(reason),
                "{reason:?} should be reconnect-worthy"
            );
        }

        let final_reasons = [
            // Plain kicks/bans/wrong-version/wrong-password/rate-limits — unchanged from before.
            "Kicked (your name is banned)",
            "Kicked (your clan is banned: test)",
            "Kicked by console",
            "Kicked for inactivity",
            "You have been banned",
            "Wrong version. Server is running '1' and client '2'",
            "Wrong password",
            "Too many connections in a short time",
            "Too many remote console authentication tries",
            "Redirect unsupported: please reconnect",
            "", // an empty-reason CLOSE
            "some future reason we've never heard of before",
            // The exact-match texts are case-*sensitive* (only the kick/ban carve-out is not) —
            // a differently-cased variant is simply not one of the recognised exact strings.
            "SERVER SHUTDOWN",
            "TIMEOUT",
            // The actual F2 round-2 regression: admin-authored kick/ban text that *contains* a
            // reconnect-worthy word must still be final — the kick/ban/VPN/"ban" check must be
            // checked first and win, not a substring match on the reconnect-worthy list.
            "Kicked (server full of noobs)",
            "Kicked (timeout abuse)",
            "Kicked (you were redirected elsewhere)",
            "Kicked (server shutdown soon, bye)",
            "You have been banned for 5 minutes (Timeout farming)",
            "You have been banned (full)",
            "VPN detected (server full)",
            "Server shutdown for maintenance, you are banned",
            "Fully automated bots are not welcome",
        ];
        for reason in final_reasons {
            assert!(!should_reconnect_after_peer_close(reason), "{reason:?} should be final");
        }
    }
}

fn attempt_log() -> &'static Mutex<HashMap<SocketAddr, VecDeque<Instant>>> {
    static LOG: OnceLock<Mutex<HashMap<SocketAddr, VecDeque<Instant>>>> = OnceLock::new();
    LOG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Blocks (in small increments, so a concurrent [`Client::disconnect`] before a slot ever opens up
/// still leaves the process able to exit promptly) until a connection attempt to `addr` would not
/// exceed [`MAX_ATTEMPTS_PER_WINDOW`] attempts in the last [`ATTEMPT_WINDOW`], then records this
/// attempt and returns.
fn wait_for_attempt_slot(addr: SocketAddr, should_abort: &dyn Fn() -> bool) -> bool {
    let mut logged_wait = false;
    loop {
        if should_abort() {
            return false;
        }
        let wait = {
            let mut log = attempt_log().lock().unwrap_or_else(|e| e.into_inner());
            let entry = log.entry(addr).or_default();
            let now = Instant::now();
            while let Some(&front) = entry.front() {
                if now.duration_since(front) > ATTEMPT_WINDOW {
                    entry.pop_front();
                } else {
                    break;
                }
            }
            if entry.len() < MAX_ATTEMPTS_PER_WINDOW {
                entry.push_back(now);
                // Task 2.3b: the D-037 rate limit is visible at info level, granted or not.
                tracing::info!(
                    %addr,
                    attempts_in_window = entry.len(),
                    max = MAX_ATTEMPTS_PER_WINDOW,
                    window = ?ATTEMPT_WINDOW,
                    "connection-attempt slot granted"
                );
                None
            } else {
                if !logged_wait {
                    logged_wait = true;
                    tracing::warn!(
                        %addr,
                        attempts_in_window = entry.len(),
                        max = MAX_ATTEMPTS_PER_WINDOW,
                        window = ?ATTEMPT_WINDOW,
                        "connection-attempt rate limit reached (D-037), waiting for a slot"
                    );
                }
                Some((ATTEMPT_WINDOW - now.duration_since(*entry.front().unwrap())).max(Duration::from_millis(50)))
            }
        };
        match wait {
            None => return true,
            Some(d) => thread::sleep(d.min(Duration::from_millis(500))),
        }
    }
}

/// Every event a [`Client`] can deliver: every [`SessionEvent`], plus the driver's own live-play
/// bookkeeping (reconnect attempts, redirects followed/refused, giving up entirely). Not `Eq`:
/// [`ClientEvent::MarginSummary`] carries `f64` fields.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientEvent {
    /// Boxed: [`SessionEvent`] is large relative to this enum's other variants (it can carry a
    /// full downloaded map's bytes, `MapLoadedEvent::bytes_to_cache`) — boxing keeps `ClientEvent`
    /// itself small regardless, so a hot channel of e.g. `Snapshot` events stays cheap to move.
    Session(Box<SessionEvent>),
    /// About to attempt a fresh connection to `addr` (attempt `attempt`, 1-based) after a lost
    /// connection — `backoff` is how long the driver slept first.
    ReconnectAttempt {
        attempt: u32,
        addr: SocketAddr,
        backoff: Duration,
    },
    /// Following `redirect@ddnet.org` to `to` (loop protection allows at most one per
    /// [`Client::connect`] call).
    RedirectFollowed { to: SocketAddr },
    /// Task 2.3b (root-cause fix): the peer sent `reconnect@ddnet.org` and the driver is about to
    /// reconnect to `addr` immediately (no backoff — a server-requested reconnect is not a
    /// failure, same reasoning as [`ClientEvent::RedirectFollowed`]). Previously this transition
    /// produced **no** `ClientEvent` at all (unlike the sibling redirect case) — a caller had no
    /// way to tell a rapid, healthy-looking string of `Connected` events apart from a peer that
    /// keeps bouncing the client via repeated reconnect requests without ever completing the join
    /// sequence, which is exactly what happened in the incident this event fixes (see
    /// `crate::session::ClientConfig::handshake_timeout`'s docs). `attempt` is this session's own
    /// monotonically increasing connection-attempt counter (task 2.3b acceptance criterion 4: "a
    /// per-session counter of connection attempts ... logged") — distinct from the process-wide
    /// rate-limiter window `wait_for_attempt_slot` enforces.
    ServerRequestedReconnect { addr: SocketAddr, attempt: u32 },
    /// A second redirect was requested; refused (loop protection) — the driver thread ends.
    RedirectRefused { reason: String },
    /// The driver thread is ending and will not reconnect (a kick/ban, an explicit
    /// [`Client::disconnect`], or a fatal local error such as failing to bind a socket at all).
    /// `category` (review round 1, finding F8) lets a caller tell a kick/ban apart from every
    /// other final outcome without parsing `reason`'s free text — `ddnet-ai play`/`record` use it
    /// to pick a distinct process exit code for "the server itself ended this" (D-016: a human
    /// should notice and decide, not have the process quietly exit `0`).
    GaveUp { reason: String, category: GaveUpCategory },
    /// A convenience alongside every [`SessionEvent::Snapshot`]: our own tee's position, when
    /// that snapshot's `PlayerInfo::local == 1` character could be found (task e2e scenario b —
    /// "the tee's own position in snapshots changes as expected" — needs this without every
    /// caller re-deriving it from [`Session::latest_view`] itself).
    OwnPosition { tick: i32, x: i32, y: i32 },
    /// Task 8.4a: a convenience alongside every [`SessionEvent::Snapshot`], mirroring
    /// [`ClientEvent::OwnPosition`] but from `PlayerInfo::team` rather than a `Character` — unlike
    /// position, this fires whenever our own player slot is known at all, *including* while
    /// spectating (no `Character` object exists for a spectator, so `OwnPosition` alone cannot
    /// tell a caller "the spectate request was granted" from "no snapshot has arrived yet"). Used
    /// by `ddnet-ai record` to confirm/log whether `Client::set_team(TEAM_SPECTATORS)` actually
    /// took effect, without needing a new synchronous query path into the driver thread's
    /// `Session` (see this module's docs on why `Session` stays thread-local).
    OwnTeam { tick: i32, team: i32 },
    /// Emitted once, right before the driver thread ends (for any reason) — the whole session's
    /// `NETMSG_INPUTTIMING` margin distribution (task e2e scenario h: "measure the fraction of
    /// inputs arriving in time ... report the margin distribution").
    MarginSummary(MarginSummary),
    /// Synthesized alongside every [`SessionEvent::Snapshot`] (task 2.4): everything
    /// `ddai-world`'s `LiveWorld::on_snapshot` needs, bundled here because `Session` itself lives
    /// on this driver's background thread — an event is the only way its state ever reaches a
    /// caller on another thread, the same reason [`ClientEvent::OwnPosition`] exists. Boxed for
    /// the same reason as [`ClientEvent::Session`] (this is the largest variant by far: a
    /// `Vec<CharacterView>` plus a handful of `Copy` structs).
    LiveWorldSnapshot(Box<LiveWorldSnapshot>),
}

/// Payload of [`ClientEvent::LiveWorldSnapshot`] — see that variant's docs.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveWorldSnapshot {
    /// The snapshot's own game tick (matches the paired `SessionEvent::Snapshot { tick }`).
    pub tick: i32,
    /// This connection's own client id (`PlayerInfo::local == 1`), if the snapshot already
    /// includes one — matches [`ClientEvent::OwnPosition`]'s own lookup.
    pub own_id: Option<i32>,
    /// [`ddai_net::view::View::characters`]'s result.
    pub characters: Vec<ddai_net::view::CharacterView>,
    /// [`Session::tuning`]'s current value.
    pub tuning: ddai_net::tuning::TuneParams,
    /// [`ddai_net::view::View::switch_states`]'s result.
    pub switch_states: Vec<(i32, ddai_net::generated::objects::SwitchState)>,
    /// [`Session::teams_state`]'s result — `None` when no `Sv_TeamsState`/`Sv_TeamsStateLegacy`
    /// has been received yet this connection (see that method's own doc comment for the "carry
    /// forward the last known value" semantics a caller needs to replicate itself across calls).
    pub teams: Option<ddai_net::tuning::TeamsState>,
    /// [`ddai_net::view::View::projectiles`]'s result — what `ddai-world`'s
    /// `LiveWorld::set_projectiles` builds the predicted projectiles from (task 2.4b).
    pub projectiles: Vec<(i32, ddai_net::view::ProjectileView)>,
}

/// Review round 1, finding F8: categorizes [`ClientEvent::GaveUp`] so a caller can pick a distinct
/// process exit code (or otherwise branch) without matching on `reason`'s free text, which exists
/// only for a human/log to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaveUpCategory {
    /// [`Client::disconnect`] was called, or `Client` was dropped — a normal, requested stop.
    Requested,
    /// A peer-initiated close the driver's policy classified as final (kick, ban, or any other
    /// reason `should_reconnect_after_peer_close` did not recognise as transient) — CLAUDE.md/
    /// D-016: never auto-reconnect from this, and a caller should treat it as distinctly
    /// noteworthy (not the same as an ordinary "duration elapsed, we disconnected on purpose").
    KickedOrBanned,
    /// A local, self-detected protocol violation (a hostile/invalid `MAP_CHANGE`, a map that
    /// failed hash/CRC verification, …) — see [`crate::session::SessionEvent::ProtocolViolation`].
    ProtocolViolation,
    /// A second redirect was requested in the same session; refused (loop protection).
    RedirectLoop,
    /// A local, non-protocol failure before or during a connection attempt: a socket could not be
    /// bound/configured/connected, a connection-attempt-rate-limit slot never became available
    /// before a stop was requested, or (review round 1, finding F1) the D-027/D-038 live-servers
    /// safety switch refused the target address.
    LocalError,
    /// Task 2.3b: [`crate::session::ClientConfig::handshake_timeout`] elapsed without ever
    /// reaching [`crate::session::SessionEvent::InGame`] — the handshake watchdog gave up on its
    /// own rather than silently retrying forever. See that field's docs for the incident this
    /// defends against (a peer that keeps sending `reconnect@ddnet.org`/`redirect@ddnet.org` right
    /// after every handshake, or one that simply never completes the join sequence at all).
    /// Distinct from [`GaveUpCategory::LocalError`] (this is a *remote* peer's behaviour, not a
    /// local resource failure) and from [`GaveUpCategory::KickedOrBanned`] (the peer here never
    /// actually refused us — it kept letting us connect, just never let the session progress) —
    /// CLAUDE.md/D-016: this is a stop condition for a human to investigate, same as a kick/ban.
    HandshakeTimeout,
    /// Task 2.3b: the peer asked for a reconnect (`reconnect@ddnet.org`) more than
    /// [`MAX_SERVER_RECONNECTS_BEFORE_IN_GAME`] time(s) without the session ever reaching in game
    /// — the shape of the Swarfey incident (10 handshakes, no map). A stop condition for a human
    /// to investigate, same as a kick/ban.
    ReconnectLoop,
    /// Task 2.3b (review F1): [`MAX_ATTEMPTS_BEFORE_FIRST_IN_GAME`] connection attempts were made
    /// without the session ever reaching in game, whatever ended them (peer `CLOSE` "This server is
    /// full", "Server shutdown", a timeout, a reconnect request, ...).
    TooManyAttempts,
    /// Task 2.3b (review F2): the session *had* been in game, was lost, and did not get back in game
    /// within the reconnect budget (`ClientConfig::handshake_timeout`, extended by map-download
    /// progress). Distinct from [`GaveUpCategory::HandshakeTimeout`] (a join that never completed).
    ReconnectBudgetExhausted,
}

impl ClientEvent {
    /// Review finding F11 (round 2 — round 1 only covered `Snapshot`/`OwnPosition`, but
    /// `GameMessage`/`ExGameMessage` are exactly as high-frequency and just as capable of growing
    /// the queue without bound: a busy server's chat/kill-feed/pickup messages, or a flood of
    /// them, with nobody draining `events()`): whether this event may be evicted (oldest-first) to
    /// keep [`event_channel`]'s queue under [`EVENT_QUEUE_CAP`] — every per-tick/per-message event
    /// where losing an old one under sustained backpressure just means missing one stale
    /// position/tick/chat line, never a missed state transition or terminal event.
    ///
    /// `SessionEvent::SnapshotData` (task 8.4a) is droppable, exactly like the plain `Snapshot` it
    /// accompanies — `ddnet-ai record` is precisely the long, possibly unattended (Swarfey, hours)
    /// session this cap exists for, so process stability wins over completeness here: losing one
    /// snapshot under sustained backpressure (e.g. a slow disk) is far better than the driver
    /// thread's memory growing without bound. A recording is already an inherently sampled view of
    /// the game (the server itself does not snapshot every tick to every client — D-031's own
    /// "~25 Hz" estimate), so one more occasionally-missing snapshot is not a new category of gap
    /// `crate::reconstruct` has to handle.
    ///
    /// `ClientEvent::OwnTeam` (task 8.4a) is droppable for the same reason as `OwnPosition` right
    /// above it: one per snapshot, and a caller only ever cares about the *latest* value.
    ///
    /// `SessionEvent::InputSent` (task 8.4a) **is** droppable too, as of review round 1's finding
    /// F7: round 1 made it non-droppable reasoning that its only consumer (a short, actively
    /// drained `--brain random-scripted` validation run) would never see backpressure — round 1's
    /// review found this was the wrong call regardless, because non-droppable events are *never*
    /// evicted even past `EVENT_QUEUE_CAP` (by design — see `event_channel`'s own docs): at
    /// `Session::flush`'s own ~50/s rate, any consumer stall of more than a few seconds would grow
    /// the sticky queue past the cap indefinitely and, worse, starve genuinely important
    /// `SnapshotData` frames sitting behind a long backlog of old `InputSent` entries. Now paired
    /// with `ClientConfig::emit_input_sent` being off by default (no caller pays for this at all
    /// unless it asked for `--input-log`), the residual risk of a gap in a *deliberately opted-in*
    /// validation log is an acceptable trade for never letting this destabilize a long recording
    /// session.
    ///
    /// `SessionEvent::InputTiming` (task 2.4) and `ClientEvent::LiveWorldSnapshot` (task 2.4) are
    /// droppable for the identical reason (round 3, finding F12): task 2.4's own round 1/2 had
    /// given `InputSent`/`InputTiming` their own top-level, non-droppable `ClientEvent` variants —
    /// which conflicted with task 8.4a's own, independently-designed and already-reviewed
    /// `SessionEvent::InputSent` (droppable, delivered the plain `ClientEvent::Session` way).
    /// Adopting 8.4a's shape (droppable, wrapped) keeps the two tasks' events reconcilable at merge
    /// time without a name/shape clash; a caller that cares about not silently losing one (task
    /// 2.4's own `LiveWorld::on_snapshot`'s `own_input_at_tick`) must tolerate a drop by holding
    /// the previous known input — see `ddai-world`'s own BUILD REPORT (round 3) for how its e2e
    /// test does exactly that.
    fn is_droppable(&self) -> bool {
        matches!(
            self,
            ClientEvent::OwnPosition { .. } | ClientEvent::OwnTeam { .. } | ClientEvent::LiveWorldSnapshot(_)
        ) || matches!(
            self,
            ClientEvent::Session(ev)
                if matches!(
                    **ev,
                    SessionEvent::Snapshot { .. }
                        | SessionEvent::SnapshotData { .. }
                        | SessionEvent::GameMessage(_)
                        | SessionEvent::ExGameMessage(_)
                        | SessionEvent::InputSent { .. }
                        | SessionEvent::InputTiming { .. }
                )
        )
    }
}

/// Cap on buffered, not-yet-consumed [`ClientEvent`]s (review finding F11) — `std::sync::mpsc` is
/// unbounded, so a caller that stops draining [`Client`]'s events (a bug, or simply a long play
/// session nobody is watching) would otherwise let the driver thread queue events forever: memory
/// growth with no bound, for as long as the session runs, from a channel that exists specifically
/// to survive exactly that kind of unattended, long-running use (`CLAUDE.md`'s live-play policy
/// covers sessions lasting hours). See [`event_channel`] for the eviction policy this backs.
const EVENT_QUEUE_CAP: usize = 512;

/// A small MPSC-shaped channel implementing [`EVENT_QUEUE_CAP`]'s bounded-with-eviction policy —
/// review finding F11 (and F13's follow-up: round 1's single-`VecDeque` design evicted via
/// `queue.iter().position(..)`, an O(n) scan on every send once at the cap — quadratic over a
/// long flood). Once at the cap, a new send evicts the *oldest* droppable event already queued
/// ([`ClientEvent::is_droppable`], "keep-latest-drop-oldest" for snapshot/position/game-message
/// noise); every other kind of event is never evicted, so the queue can still grow slightly past
/// the cap in the rare case where it is entirely full of non-droppable events — a bounded,
/// self-limiting edge case (that many *distinct* terminal/state-transition events could ever queue
/// up at once is implausible in practice), not the unbounded growth this exists to prevent.
///
/// O(1) send/pop, not O(n): droppable and non-droppable events live in two separate `VecDeque`s
/// (so evicting the oldest droppable one is a plain `pop_front`, never a scan), each entry tagged
/// with a monotonically increasing sequence number recording its original arrival order; a pop
/// picks whichever of the two queues' fronts has the smaller sequence number, so the two streams
/// still merge back into exactly the order they were sent in. Deliberately minimal otherwise: only
/// the handful of operations [`Client`] actually needs.
mod event_channel {
    use super::{ClientEvent, EVENT_QUEUE_CAP};
    use std::collections::VecDeque;
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    #[derive(Default)]
    struct Queues {
        next_seq: u64,
        /// [`ClientEvent::is_droppable`] entries — the only ones [`Sender::send`] ever evicts.
        droppable: VecDeque<(u64, ClientEvent)>,
        /// Everything else — never evicted.
        sticky: VecDeque<(u64, ClientEvent)>,
    }

    impl Queues {
        /// Pops whichever of the two queues' fronts arrived first (smaller sequence number),
        /// preserving overall FIFO order across both — O(1).
        fn pop_front(&mut self) -> Option<ClientEvent> {
            let take_droppable = match (self.droppable.front(), self.sticky.front()) {
                (Some((d, _)), Some((s, _))) => d <= s,
                (Some(_), None) => true,
                (None, _) => false,
            };
            let (_, ev) = if take_droppable {
                self.droppable.pop_front()?
            } else {
                self.sticky.pop_front()?
            };
            Some(ev)
        }

        fn len(&self) -> usize {
            self.droppable.len() + self.sticky.len()
        }
    }

    struct Shared {
        queues: Mutex<Queues>,
        cvar: Condvar,
    }

    #[derive(Clone)]
    pub struct Sender(Arc<Shared>);

    pub struct Receiver(Arc<Shared>);

    pub fn channel() -> (Sender, Receiver) {
        let shared = Arc::new(Shared {
            queues: Mutex::new(Queues::default()),
            cvar: Condvar::new(),
        });
        (Sender(Arc::clone(&shared)), Receiver(shared))
    }

    impl Sender {
        /// Never fails, never blocks — O(1): the eviction policy above is exactly what keeps this
        /// true without an unbounded (or quadratic-under-load) queue.
        pub fn send(&self, ev: ClientEvent) {
            let mut queues = self.0.queues.lock().unwrap_or_else(|e| e.into_inner());
            if queues.len() >= EVENT_QUEUE_CAP {
                queues.droppable.pop_front();
            }
            let seq = queues.next_seq;
            queues.next_seq += 1;
            if ev.is_droppable() {
                queues.droppable.push_back((seq, ev));
            } else {
                queues.sticky.push_back((seq, ev));
            }
            self.0.cvar.notify_one();
        }
    }

    impl Receiver {
        pub fn recv_timeout(&self, timeout: Duration) -> Option<ClientEvent> {
            let deadline = Instant::now() + timeout;
            let mut queues = self.0.queues.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(ev) = queues.pop_front() {
                    return Some(ev);
                }
                let now = Instant::now();
                if now >= deadline {
                    return None;
                }
                let (locked, _timed_out) = self
                    .0
                    .cvar
                    .wait_timeout(queues, deadline - now)
                    .unwrap_or_else(|e| e.into_inner());
                queues = locked;
            }
        }

        pub fn try_recv(&self) -> Option<ClientEvent> {
            self.0.queues.lock().unwrap_or_else(|e| e.into_inner()).pop_front()
        }

        pub fn try_iter(&self) -> impl Iterator<Item = ClientEvent> + '_ {
            std::iter::from_fn(move || self.try_recv())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::driver::GaveUpCategory;
        use crate::session::SessionEvent;

        #[test]
        fn recv_timeout_returns_none_when_empty_and_the_event_when_sent() {
            let (tx, rx) = channel();
            assert_eq!(rx.recv_timeout(Duration::from_millis(20)), None);
            tx.send(ClientEvent::OwnPosition { tick: 1, x: 2, y: 3 });
            assert_eq!(
                rx.recv_timeout(Duration::from_secs(1)),
                Some(ClientEvent::OwnPosition { tick: 1, x: 2, y: 3 })
            );
        }

        #[test]
        fn fifo_order_is_preserved_under_no_pressure() {
            let (tx, rx) = channel();
            for i in 0..5 {
                tx.send(ClientEvent::OwnPosition { tick: i, x: 0, y: 0 });
            }
            for i in 0..5 {
                assert_eq!(rx.try_recv(), Some(ClientEvent::OwnPosition { tick: i, x: 0, y: 0 }));
            }
            assert_eq!(rx.try_recv(), None);
        }

        /// Review finding F11's core property: past the cap, a droppable event is evicted
        /// (oldest first) rather than the queue growing without bound.
        #[test]
        fn past_the_cap_the_oldest_droppable_event_is_evicted_not_the_queue_growing() {
            let (tx, rx) = channel();
            for i in 0..(EVENT_QUEUE_CAP + 10) {
                tx.send(ClientEvent::OwnPosition {
                    tick: i as i32,
                    x: 0,
                    y: 0,
                });
            }
            let queued = {
                let queues = tx.0.queues.lock().unwrap();
                queues.len()
            };
            assert_eq!(
                queued, EVENT_QUEUE_CAP,
                "queue must never exceed the cap for droppable events"
            );
            // The oldest entries were evicted, not the newest — so the surviving front is well
            // past tick 0.
            let front = rx.try_recv().unwrap();
            assert!(matches!(front, ClientEvent::OwnPosition { tick, .. } if tick >= 10));
        }

        /// Never-drop guarantee: non-droppable events must survive even under sustained
        /// send-pressure of droppable ones — they are only ever evicted, never the other kind.
        #[test]
        fn non_droppable_events_are_never_evicted() {
            let (tx, rx) = channel();
            tx.send(ClientEvent::Session(Box::new(SessionEvent::Connected)));
            for i in 0..(EVENT_QUEUE_CAP * 2) {
                tx.send(ClientEvent::OwnPosition {
                    tick: i as i32,
                    x: 0,
                    y: 0,
                });
            }
            let mut saw_connected = false;
            let mut drained = 0usize;
            while let Some(ev) = rx.try_recv() {
                drained += 1;
                if matches!(ev, ClientEvent::Session(ref s) if matches!(**s, SessionEvent::Connected)) {
                    saw_connected = true;
                }
            }
            assert!(
                saw_connected,
                "the one non-droppable event must never have been evicted"
            );
            assert!(
                drained <= EVENT_QUEUE_CAP + 1,
                "queue must stay bounded even with one non-droppable entry"
            );
        }

        /// Review finding F13: `GameMessage`/`ExGameMessage` are exactly as flood-prone as
        /// `Snapshot`/`OwnPosition` (a busy server's chat/kill-feed, or a hostile flood) and must
        /// be droppable too, not silently exempt from round 1's cap.
        #[test]
        fn game_messages_and_ex_game_messages_are_droppable_too() {
            use ddai_net::generated::messages as msgs;

            let (tx, rx) = channel();
            for _ in 0..(EVENT_QUEUE_CAP * 2) {
                tx.send(ClientEvent::Session(Box::new(SessionEvent::GameMessage(
                    msgs::GameMsg::SvMotd(msgs::SvMotd { message: String::new() }),
                ))));
                tx.send(ClientEvent::Session(Box::new(SessionEvent::ExGameMessage(
                    msgs::ExGameMsg::SvMyOwnMessage(msgs::SvMyOwnMessage { test: 0 }),
                ))));
            }
            let queued = tx.0.queues.lock().unwrap().len();
            assert_eq!(
                queued, EVENT_QUEUE_CAP,
                "a flood of game/ex-game messages must still respect the cap"
            );
            let _ = rx; // draining isn't the point of this test — the cap itself is
        }

        /// The new two-`VecDeque` split (review finding F13) must still merge droppable and
        /// non-droppable events back into *exactly* the order they were sent in, not just each
        /// sub-stream's own internal order (droppable-only and sticky-only order was already the
        /// trivial part) — the actual property worth pinning down is the *interleaving*.
        #[test]
        fn fifo_order_is_preserved_across_droppable_and_sticky_interleaving() {
            let (tx, rx) = channel();
            // Alternating droppable (OwnPosition) / sticky (GaveUp) sends — the exact interleaved
            // order below is what a correct merge must reproduce.
            tx.send(ClientEvent::OwnPosition { tick: 1, x: 0, y: 0 });
            tx.send(ClientEvent::Session(Box::new(SessionEvent::Connected)));
            tx.send(ClientEvent::OwnPosition { tick: 2, x: 0, y: 0 });
            tx.send(ClientEvent::OwnPosition { tick: 3, x: 0, y: 0 });
            tx.send(ClientEvent::GaveUp {
                reason: "test".to_string(),
                category: GaveUpCategory::Requested,
            });
            tx.send(ClientEvent::OwnPosition { tick: 4, x: 0, y: 0 });

            let expected = vec![
                ClientEvent::OwnPosition { tick: 1, x: 0, y: 0 },
                ClientEvent::Session(Box::new(SessionEvent::Connected)),
                ClientEvent::OwnPosition { tick: 2, x: 0, y: 0 },
                ClientEvent::OwnPosition { tick: 3, x: 0, y: 0 },
                ClientEvent::GaveUp {
                    reason: "test".to_string(),
                    category: GaveUpCategory::Requested,
                },
                ClientEvent::OwnPosition { tick: 4, x: 0, y: 0 },
            ];
            let mut actual = Vec::new();
            while let Some(ev) = rx.try_recv() {
                actual.push(ev);
            }
            assert_eq!(actual, expected);
        }

        /// Task 8.4a: `OwnTeam` is exactly as flood-prone as `OwnPosition` (one per snapshot,
        /// while a `Character` may or may not exist) and must be droppable too.
        #[test]
        fn own_team_is_droppable_too() {
            let (tx, rx) = channel();
            for i in 0..(EVENT_QUEUE_CAP + 10) {
                tx.send(ClientEvent::OwnTeam {
                    tick: i as i32,
                    team: -1,
                });
            }
            let queued = tx.0.queues.lock().unwrap().len();
            assert_eq!(
                queued, EVENT_QUEUE_CAP,
                "a flood of OwnTeam events must still respect the cap"
            );
            let front = rx.try_recv().unwrap();
            assert!(matches!(front, ClientEvent::OwnTeam { tick, .. } if tick >= 10));
        }

        /// Task 8.4a / 2.3-review carry-over finding F15: when the queue is full of *only*
        /// control (non-droppable/"sticky") events — no droppable event anywhere in it for
        /// `Sender::send`'s eviction to fall back on — none of them are ever evicted. [`Sender::send`]
        /// only ever pops from `queues.droppable`, never `queues.sticky` (see its own source
        /// above); this test pins that property down directly rather than relying on
        /// `non_droppable_events_are_never_evicted`'s coverage, which always mixes in at least one
        /// droppable event. The queue is allowed to grow past [`EVENT_QUEUE_CAP`] in this case
        /// (documented as a bounded, self-limiting edge case on [`Sender::send`]'s own docs) — the
        /// property under test is "never evicted", not "stays at the cap".
        #[test]
        fn control_events_are_never_evicted_when_the_queue_is_full_of_control_events_only() {
            let (tx, rx) = channel();
            let total = EVENT_QUEUE_CAP * 3;
            for i in 0..total {
                tx.send(ClientEvent::ReconnectAttempt {
                    attempt: i as u32,
                    addr: "127.0.0.1:8303".parse().unwrap(),
                    backoff: Duration::from_secs(1),
                });
            }
            let mut drained = Vec::new();
            while let Some(ev) = rx.try_recv() {
                drained.push(ev);
            }
            assert_eq!(
                drained.len(),
                total,
                "every control event must survive when the queue holds only control events"
            );
            for (i, ev) in drained.iter().enumerate() {
                assert!(
                    matches!(ev, ClientEvent::ReconnectAttempt { attempt, .. } if *attempt == i as u32),
                    "control events must also come out in the exact order they were sent"
                );
            }
        }
    }
}

fn default_player_input() -> PlayerInput {
    PlayerInput {
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
    }
}

enum Control {
    Disconnect,
    /// Task 8.4a: requests `Cl_SetTeam(team)` on the current connection (see
    /// [`crate::session::Session::request_team`]) — [`Client::set_team`]'s wire-side plumbing,
    /// the same shape as [`Client::set_input`]'s `input_tx` but routed through the control channel
    /// since a team change is a one-off request, not a per-tick value to keep resending.
    SetTeam(i32),
}

/// The real-time client — task acceptance criterion 1's public entry point.
pub struct Client {
    events_rx: event_channel::Receiver,
    input_tx: Sender<PlayerInput>,
    control_tx: Sender<Control>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Client {
    /// Spawns the driver thread and starts connecting to `addr`. Returns immediately — the first
    /// [`ClientEvent`] (typically [`SessionEvent::Connected`], wrapped) arrives asynchronously.
    pub fn connect(addr: SocketAddr, config: ClientConfig) -> Client {
        let (events_tx, events_rx) = event_channel::channel();
        let (input_tx, input_rx) = mpsc::channel();
        let (control_tx, control_rx) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("ddai-client".to_string())
            .spawn(move || run(addr, config, events_tx, input_rx, control_rx))
            .expect("failed to spawn the ddai-client driver thread");
        Client {
            events_rx,
            input_tx,
            control_tx,
            handle: Some(handle),
        }
    }

    /// Sets the input to embed in every `NETMSG_INPUT` from now on — task acceptance criterion
    /// 1's `set_input(PlayerInput)`. Best-effort: a driver that has already stopped simply drops
    /// this silently (nothing left to send it to).
    pub fn set_input(&self, input: PlayerInput) {
        let _ = self.input_tx.send(input);
    }

    /// Requests `Cl_SetTeam(team)` on the current connection (task 8.4a acceptance criterion 1:
    /// "after entering, it joins spectators the way the real 20.1 client does" —
    /// [`crate::session::Session::request_team`]). Best-effort and asynchronous, like
    /// [`Client::set_input`]: a driver that has already stopped simply drops this; a caller that
    /// wants confirmation should watch for [`ClientEvent::OwnTeam`] in the event stream instead of
    /// assuming this call alone means the server actually granted the change (see
    /// `gamecontext.cpp:2684-2730`'s `CanJoinTeam`/spam-protection/kill-protection checks, any of
    /// which can silently refuse it).
    pub fn set_team(&self, team: i32) {
        let _ = self.control_tx.send(Control::SetTeam(team));
    }

    /// Requests a graceful, permanent disconnect — the driver thread will not reconnect
    /// afterwards.
    pub fn disconnect(&self) {
        let _ = self.control_tx.send(Control::Disconnect);
    }

    /// Blocks for at most `timeout` waiting for the next event.
    pub fn recv_event(&self, timeout: Duration) -> Option<ClientEvent> {
        self.events_rx.recv_timeout(timeout)
    }

    pub fn try_recv_event(&self) -> Option<ClientEvent> {
        self.events_rx.try_recv()
    }

    /// Drains every event currently queued, without blocking.
    pub fn events(&self) -> impl Iterator<Item = ClientEvent> + '_ {
        self.events_rx.try_iter()
    }

    /// Blocks until the driver thread has actually exited (e.g. after [`Client::disconnect`] or a
    /// terminal [`ClientEvent::GaveUp`]). Takes `&mut self`, not `self`, deliberately: the
    /// driver's last events (notably [`ClientEvent::MarginSummary`], sent right before its thread
    /// returns) are only guaranteed to already be sitting in the channel *after* this returns —
    /// [`Client::events`]/[`Client::recv_event`] remain callable afterwards to drain them.
    pub fn join(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Client {
    /// Review finding F5: without this, a caller that drops `Client` without ever calling
    /// [`Client::disconnect`]/[`Client::join`] leaks the driver thread forever — it would keep
    /// running its full connection/reconnect/backoff loop (and the socket that goes with it)
    /// indefinitely, with nothing left able to reach it. Best-effort and non-blocking: tells the
    /// thread to stop (it notices within one `POLL_TIMEOUT`/backoff-sleep-check cycle — see
    /// `run_one_connection`/`run`'s own handling of a closed control channel, which fires even if
    /// this `send` somehow raced past it) and returns immediately. Deliberately does *not*
    /// `join()`: blocking a `Drop` on a background thread's own network I/O would trade a
    /// (bounded, self-correcting) thread leak for a much worse footgun — a drop that can hang
    /// indefinitely. A caller that needs to know the thread has actually finished must still call
    /// [`Client::join`] itself first.
    fn drop(&mut self) {
        let _ = self.control_tx.send(Control::Disconnect);
    }
}

fn send_all(socket: &UdpSocket, datagrams: Vec<Vec<u8>>) {
    for dg in datagrams {
        if let Err(e) = socket.send(&dg) {
            tracing::warn!(error = %e, "failed to send a datagram");
        }
    }
}

#[derive(Debug)]
enum ConnectionOutcome {
    Stop,
    Reconnect,
    Redirect(u16),
    LostConnection,
    KickedOrBanned,
    /// Review finding F4: a local, self-detected protocol violation
    /// ([`SessionEvent::ProtocolViolation`] — a bad/hostile `MAP_CHANGE`, a map that failed
    /// verification, …). Always final, same as [`ConnectionOutcome::KickedOrBanned`] — retrying
    /// forever against a server that is simply sending us garbage is not "staying connected", see
    /// that event's own docs.
    ProtocolViolation,
    /// Task 2.3b: the handshake watchdog expired inside this very connection attempt (still
    /// waiting on the TKEN handshake, or online but never reaching in-game) — see
    /// [`GaveUpCategory::HandshakeTimeout`]'s docs. Always final, same shape as the two outcomes
    /// above.
    HandshakeTimedOut,
}

/// Intercepts the two events the driver must act on itself ([`SessionEvent::MapChanging`]'s cache
/// lookup, [`SessionEvent::MapLoaded`]'s cache write — see `crate::session`'s module docs for why
/// this synchronous ordering works), forwards every event to `events_tx`, and translates a
/// terminal event into a [`ConnectionOutcome`] for the caller to act on.
fn handle_session_event(
    ev: SessionEvent,
    session: &mut Session,
    config: &ClientConfig,
    now: Duration,
    events_tx: &event_channel::Sender,
) -> Option<ConnectionOutcome> {
    // Acted on (and, where it produces a *derived* event, queued into `synthesized`) before the
    // triggering event itself is forwarded below — so a listener always sees e.g. `MapChanging`
    // before the `MapLoaded` it caused, not the other way around.
    let mut synthesized: Vec<ClientEvent> = Vec::new();

    if let SessionEvent::MapChanging {
        name,
        sha256: Some(sha256),
        ..
    } = &ev
        && let Some(bytes) = map_cache::read_cached(&config.cache_dir, name, sha256)
    {
        match session.supply_cached_map(&bytes, now) {
            Ok(loaded) => synthesized.push(ClientEvent::Session(Box::new(SessionEvent::MapLoaded(loaded)))),
            Err(e) => {
                tracing::warn!(error = %e, "cached map failed verification; falling back to a fresh download");
            }
        }
    }
    if let SessionEvent::MapLoaded(loaded) = &ev
        && let Some(bytes) = &loaded.bytes_to_cache
        && let Err(e) = map_cache::write_cache(&config.cache_dir, &loaded.name, &loaded.sha256, bytes)
    {
        tracing::warn!(error = %e, "failed to write the map cache");
    }
    if let SessionEvent::Snapshot { tick } = &ev
        && let Some(view) = session.latest_view()
        && let Some(own) = view.players().iter().find(|p| p.info.local == 1)
    {
        // Task 8.4a: `OwnTeam` fires whenever our player slot is known at all (spectating or
        // not); `OwnPosition` stays conditional on an actual `Character` existing, which a
        // spectator never has — see `ClientEvent::OwnTeam`'s docs.
        synthesized.push(ClientEvent::OwnTeam {
            tick: *tick,
            team: own.info.team,
        });
        if let Some(character) = view.character(own.id) {
            synthesized.push(ClientEvent::OwnPosition {
                tick: *tick,
                x: character.character.x,
                y: character.character.y,
            });
        }
    }
    if let SessionEvent::Snapshot { tick } = &ev
        && let Some(view) = session.latest_view()
    {
        let own_id = view.players().iter().find(|p| p.info.local == 1).map(|p| p.id);
        synthesized.push(ClientEvent::LiveWorldSnapshot(Box::new(LiveWorldSnapshot {
            tick: *tick,
            own_id,
            characters: view.characters(),
            tuning: session.tuning(),
            switch_states: view.switch_states(),
            teams: session.teams_state(),
            projectiles: view.projectiles(),
        })));
    }

    let outcome = match &ev {
        SessionEvent::ReconnectRequested => Some(ConnectionOutcome::Reconnect),
        SessionEvent::RedirectRequested { port } => Some(ConnectionOutcome::Redirect(*port)),
        // Review finding F2: `by_peer: true` is no longer a blanket "final" — see
        // `should_reconnect_after_peer_close`'s docs for the policy and its citations.
        SessionEvent::Disconnected { by_peer: true, reason } => {
            let reconnect_worthy = reason.as_deref().is_some_and(should_reconnect_after_peer_close);
            Some(if reconnect_worthy {
                ConnectionOutcome::LostConnection
            } else {
                ConnectionOutcome::KickedOrBanned
            })
        }
        SessionEvent::Disconnected { by_peer: false, .. } => Some(ConnectionOutcome::LostConnection),
        // Review finding F4: final, never retried — see `ConnectionOutcome::ProtocolViolation`'s
        // docs.
        SessionEvent::ProtocolViolation { .. } => Some(ConnectionOutcome::ProtocolViolation),
        _ => None,
    };
    // Review round 3, finding F12: `InputSent`/`InputTiming` used to get their own top-level
    // `ClientEvent` variant (review round 1, finding F5) instead of the generic
    // `ClientEvent::Session` wrap every other `SessionEvent` gets — reverted to match task 8.4a's
    // own shape (see `ClientEvent::is_droppable`'s doc comment for why).
    events_tx.send(ClientEvent::Session(Box::new(ev)));
    for event in synthesized {
        events_tx.send(event);
    }
    outcome
}

/// Drives one connection attempt (one socket, one [`Session`]) until it ends, for whatever
/// reason — see [`ConnectionOutcome`]. Nine parameters: this is the one place every piece of
/// per-connection state (socket, session, config, channels, the caller's latest input) has to
/// meet; grouping them into a struct would only rename this same list, not shorten it, since nothing
/// else in this crate shares that grouping.
#[allow(clippy::too_many_arguments)]
fn run_one_connection(
    socket: &UdpSocket,
    session: &mut Session,
    config: &ClientConfig,
    start: Instant,
    events_tx: &event_channel::Sender,
    input_rx: &Receiver<PlayerInput>,
    control_rx: &Receiver<Control>,
    latest_input: &mut PlayerInput,
    // Review finding F7: set to `true` the moment this connection attempt ever reaches
    // [`SessionEvent::InGame`] — `run`'s caller uses this to decide whether a later
    // `LostConnection` should reset the backoff/attempt-count state.
    reached_in_game: &mut bool,
    // Task 2.3b: the handshake watchdog's absolute deadline — accumulates across every connection
    // attempt within the same logical session (see `ClientConfig::handshake_timeout`'s docs), so
    // this single connection attempt may itself have little or no time left on it. Extended
    // (never past `hard_deadline`) while map bytes keep arriving.
    handshake_deadline: &mut Instant,
    hard_deadline: Instant,
) -> ConnectionOutcome {
    let mut buf = [0u8; RECV_BUF_SIZE];
    // Task 2.3b: while the join has not completed, say once per second (info level) which step it
    // is waiting in, so a stalled session explains itself without trace logging.
    let mut next_progress_log = Instant::now() + JOIN_PROGRESS_LOG_INTERVAL;
    let mut last_download_bytes = 0usize;
    // Task 2.3b, per the tech lead's explicit request while investigating the Swarfey incident:
    // a hard safety invariant, independent of (and in addition to) the handshake watchdog above.
    // `ddai_net::conn::Connection::feed` only ever transitions `Connecting -> Online` — see that
    // module's own docs and `conn.rs`'s `ControlMsg::ConnectAccept` arm — so a *second*
    // `SessionEvent::Connected` within the same `run_one_connection` call (i.e. without this
    // driver itself having made an explicit reconnect/redirect decision and started a *new*
    // `Session`/socket) should be structurally impossible with the code as written today. It is
    // checked anyway, unconditionally, as defense in depth: if a future change to `ddai-net`/
    // `ddai-client`, an unanticipated server behaviour (e.g. a delayed burst of `CONNECTACCEPT`
    // replies answering several of this client's own `CONNECT` resends, arriving after this
    // connection is already online), or a bug we have not found yet ever violates that invariant,
    // this must never be silently absorbed — it must stop the session loudly rather than continue
    // as if nothing happened, which is exactly the failure mode this whole task exists to close.
    let mut seen_connected = false;
    loop {
        // Task 2.3b: checked before anything else in the loop body, so a connection attempt that
        // is itself hanging (stuck in the TKEN handshake, or online but never in-game) cannot
        // sit here past the deadline just because nothing else in the loop happened to trip it.
        // A slow but progressing map download must not be killed by the watchdog: every time new
        // map bytes have arrived, push the deadline out by a full timeout, capped absolutely.
        if let Some(bytes) = session.download_progress()
            && bytes > last_download_bytes
        {
            last_download_bytes = bytes;
            let extended = (Instant::now() + config.handshake_timeout).min(hard_deadline);
            if extended > *handshake_deadline {
                *handshake_deadline = extended;
            }
        }
        if !*reached_in_game && !session.is_in_game() && Instant::now() >= next_progress_log {
            tracing::info!(
                phase = session.join_phase(),
                elapsed = ?Instant::now().duration_since(start),
                watchdog_left = ?handshake_deadline.saturating_duration_since(Instant::now()),
                map_bytes = session.download_progress(),
                "join in progress"
            );
            next_progress_log += JOIN_PROGRESS_LOG_INTERVAL;
        }
        if !*reached_in_game && Instant::now() >= *handshake_deadline {
            let now = Instant::now().duration_since(start);
            session.disconnect(Some("handshake watchdog: not in game in time"));
            send_all(socket, session.flush(now));
            return ConnectionOutcome::HandshakeTimedOut;
        }
        // Review finding F5: an explicit `Control::Disconnect` *or* the channel having been
        // dropped entirely (the `Client` was dropped without an explicit `disconnect()`/`join()`
        // — [`Client`]'s own `Drop` impl already sends `Disconnect` proactively, but treating a
        // closed channel the same way here is a second, independent layer that needs nothing from
        // that impl to work) both mean the same thing: stop, gracefully, now.
        match control_rx.try_recv() {
            Ok(Control::Disconnect) | Err(mpsc::TryRecvError::Disconnected) => {
                let now = Instant::now().duration_since(start);
                session.disconnect(Some("client requested disconnect"));
                send_all(socket, session.flush(now));
                return ConnectionOutcome::Stop;
            }
            // Task 8.4a: not terminal — send the request now (rather than waiting for the next
            // `flush()` below, which only runs after this loop's `socket.recv` — up to
            // `POLL_TIMEOUT` later) and keep driving this connection.
            Ok(Control::SetTeam(team)) => {
                let now = Instant::now().duration_since(start);
                session.request_team(team, now);
                send_all(socket, session.flush(now));
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }

        while let Ok(input) = input_rx.try_recv() {
            *latest_input = input;
        }
        session.set_input(*latest_input);

        match socket.recv(&mut buf) {
            Ok(n) => {
                let now = Instant::now().duration_since(start);
                for ev in session.feed(&buf[..n], now) {
                    if matches!(ev, SessionEvent::InGame) {
                        *reached_in_game = true;
                    }
                    if matches!(ev, SessionEvent::Connected)
                        && let Some(outcome) =
                            reject_duplicate_connected(&mut seen_connected, socket, session, now, events_tx)
                    {
                        return outcome;
                    }
                    if let Some(outcome) = handle_session_event(ev, session, config, now, events_tx) {
                        // Review finding F4: flush before returning — a terminal event (e.g. the
                        // `CLOSE` `Session::disconnect()` above just queued, or an in-flight
                        // `ACCEPT`/ack the join sequence needs to complete) must actually reach
                        // the wire before this connection's socket is dropped, not get silently
                        // discarded still sitting in `Connection`'s send queue.
                        send_all(socket, session.flush(now));
                        return outcome;
                    }
                }
            }
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
            // Review finding F2: a `connect()`-ed UDP socket surfaces the OS's own ICMP
            // Port-Unreachable as `ECONNREFUSED` on a subsequent `recv`/`send` — this is *not* the
            // application-level "the peer is gone" signal DDNet's protocol actually uses (that is
            // `Connection`'s own silence timeout, driven purely by elapsed time since
            // `last_recv_time`, exactly like the real client). A real DDNet client never even sees
            // this (it doesn't treat a transient ICMP error as fatal either); mirrored here by
            // simply ignoring it and letting the loop continue — `Connection`'s own timeout will
            // still fire on schedule if the peer really is gone, just not a full RTT sooner than
            // that from one stray ICMP packet during, say, a graceful `systemctl restart` between
            // it closing its listening socket and actually being gone.
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::ConnectionRefused | io::ErrorKind::ConnectionReset
                ) =>
            {
                tracing::debug!(error = %e, "ignoring a transient ICMP-style socket error");
            }
            Err(e) => {
                events_tx.send(ClientEvent::Session(Box::new(SessionEvent::Disconnected {
                    reason: Some(format!("socket error: {e}")),
                    by_peer: false,
                })));
                return ConnectionOutcome::LostConnection;
            }
        }

        let now = Instant::now().duration_since(start);
        send_all(socket, session.flush(now));
        for ev in session.take_events() {
            // Belt-and-suspenders, same as the identical check above: `Connection::take_events`'s
            // own docs say it never actually produces a `Connected`-mapped event, but this is
            // cheap enough to check unconditionally rather than trust that documentation forever.
            if matches!(ev, SessionEvent::Connected)
                && let Some(outcome) = reject_duplicate_connected(&mut seen_connected, socket, session, now, events_tx)
            {
                return outcome;
            }
            if let Some(outcome) = handle_session_event(ev, session, config, now, events_tx) {
                // See the comment on the identical pattern above.
                send_all(socket, session.flush(now));
                return outcome;
            }
        }
    }
}

/// Task 2.3b's hard safety invariant (see `run_one_connection`'s own doc comment on
/// `seen_connected` for the full reasoning): the first `SessionEvent::Connected` this connection
/// attempt ever sees is recorded and allowed through (returns `None`); a *second* one is treated
/// as a fatal, self-detected protocol anomaly — logged loudly, surfaced as a proper
/// `SessionEvent::ProtocolViolation` (so it reaches a caller/log the exact same way any other
/// protocol violation does, not as a special case), and the connection is torn down gracefully.
/// Returns `Some(outcome)` when the caller must stop and return that outcome immediately.
fn reject_duplicate_connected(
    seen_connected: &mut bool,
    socket: &UdpSocket,
    session: &mut Session,
    now: Duration,
    events_tx: &event_channel::Sender,
) -> Option<ConnectionOutcome> {
    if !*seen_connected {
        *seen_connected = true;
        return None;
    }
    let reason = "protocol invariant violated: a second Connected event fired within one \
                  connection attempt, without this driver making an explicit reconnect/redirect \
                  decision — stopping rather than continuing silently"
        .to_string();
    tracing::error!(%reason, "connection: duplicate Connected event — hard safety stop");
    events_tx.send(ClientEvent::Session(Box::new(SessionEvent::ProtocolViolation {
        reason: reason.clone(),
    })));
    session.disconnect(Some(&reason));
    send_all(socket, session.flush(now));
    Some(ConnectionOutcome::ProtocolViolation)
}

/// The watchdog's give-up, shared by the between-attempts and the mid-attempt paths. Before the
/// first in-game it is a join that never completed ([`GaveUpCategory::HandshakeTimeout`]); after an
/// in-game loss it is an exhausted reconnect budget ([`GaveUpCategory::ReconnectBudgetExhausted`]).
fn give_up_on_watchdog(
    events_tx: &event_channel::Sender,
    target: SocketAddr,
    attempts: u32,
    timeout: Duration,
    ever_in_game: bool,
) {
    let (reason, category) = if ever_in_game {
        (
            format!(
                "reconnect budget exhausted after in-game loss: not back in game within {timeout:?} of the loss ({attempts} connection attempt(s) in total)"
            ),
            GaveUpCategory::ReconnectBudgetExhausted,
        )
    } else {
        (
            format!("handshake watchdog: not in game within {timeout:?} after {attempts} connection attempt(s)"),
            GaveUpCategory::HandshakeTimeout,
        )
    };
    tracing::warn!(addr = %target, attempts, ?timeout, ever_in_game, "{reason} (no further retries)");
    events_tx.send(ClientEvent::GaveUp { reason, category });
}

/// Review F1: whether one more connection attempt would exceed
/// [`MAX_ATTEMPTS_BEFORE_FIRST_IN_GAME`].
fn attempt_cap_reached(ever_in_game: bool, connection_attempt: u32) -> bool {
    !ever_in_game && connection_attempt >= MAX_ATTEMPTS_BEFORE_FIRST_IN_GAME
}

fn give_up_on_attempt_cap(events_tx: &event_channel::Sender, target: SocketAddr, attempts: u32) {
    let reason = format!(
        "too many attempts: {attempts} connection attempt(s) without ever reaching in game (limit {MAX_ATTEMPTS_BEFORE_FIRST_IN_GAME}) — not retrying"
    );
    tracing::error!(addr = %target, attempts, "{reason}");
    events_tx.send(ClientEvent::GaveUp {
        reason,
        category: GaveUpCategory::TooManyAttempts,
    });
}

/// Discards anything still queued on the (reused) socket from the previous connection, so a stale
/// `CLOSE`/`CONNECTACCEPT` from the old connection can never be mistaken for a reply to the new
/// `CONNECT` (a `Connecting` connection has no token to check yet). Beyond the real client, which
/// does not do this; strictly safer.
fn drain_stale_datagrams(socket: &UdpSocket) {
    if socket.set_nonblocking(true).is_err() {
        return;
    }
    let mut buf = [0u8; RECV_BUF_SIZE];
    let mut drained = 0u32;
    while socket.recv(&mut buf).is_ok() {
        drained += 1;
    }
    let _ = socket.set_nonblocking(false);
    if drained > 0 {
        tracing::info!(
            drained,
            "driver: discarded stale datagrams from the previous connection"
        );
    }
}

/// The driver thread's body — see the module docs for the live-play policy this implements.
fn run(
    initial_addr: SocketAddr,
    config: ClientConfig,
    events_tx: event_channel::Sender,
    input_rx: Receiver<PlayerInput>,
    control_rx: Receiver<Control>,
) {
    let mut target = initial_addr;
    let mut redirects_followed = 0u32;
    // Task 2.3b: ONE socket (one local UDP port) for the whole `Client::connect` call, reused for
    // every reconnect/redirect/retry, exactly like the real client — `CClient::InitNetworkClient`
    // opens its `CNetClient` once (`client.cpp:3619`, `m_aNetClient[Conn].Open(BindAddr)`), and
    // `CClient::Connect` (`client.cpp:622-723`), which `NETMSG_RECONNECT` (`client.cpp:1979-1982`)
    // and `NETMSG_REDIRECT` (`client.cpp:1991-2004`) both call, merely `Disconnect()`s and
    // `Connect()`s that same `CNetClient`. Before this fix every attempt bound a fresh ephemeral
    // port; a server or anti-bot layer that keys its state on the client's `ip:port` then sees a
    // brand-new client on every reconnect and can never conclude the handshake.
    let socket = match UdpSocket::bind("0.0.0.0:0") {
        Ok(s) => s,
        Err(e) => {
            events_tx.send(ClientEvent::GaveUp {
                reason: format!("failed to bind a local socket: {e}"),
                category: GaveUpCategory::LocalError,
            });
            return;
        }
    };
    if let Err(e) = socket.set_read_timeout(Some(POLL_TIMEOUT)) {
        events_tx.send(ClientEvent::GaveUp {
            reason: format!("failed to configure the socket: {e}"),
            category: GaveUpCategory::LocalError,
        });
        return;
    }
    if let Ok(local) = socket.local_addr() {
        tracing::info!(
            local_port = local.port(),
            "driver: local UDP socket bound (reused for every reconnect)"
        );
    }
    // Server-requested reconnects followed since the last time we were in game — see
    // `MAX_SERVER_RECONNECTS_BEFORE_IN_GAME`.
    let mut server_reconnects_pending: u32 = 0;
    let mut backoff = MIN_BACKOFF;
    let mut latest_input = default_player_input();
    let mut reconnect_attempt: u32 = 0;
    // Task 2.3b acceptance criterion 4: "a per-session counter of connection attempts ... logged"
    // — every time this loop is about to start a new connection, for *any* reason (the very first
    // connect, a server-requested reconnect/redirect, or a retry after a lost connection). Never
    // reset (unlike `backoff`/`reconnect_attempt`, which reset on reaching in-game): a caller
    // reading the log wants "how many times has this process tried to reach the wire, ever",
    // regardless of how many of those attempts briefly succeeded in between.
    let mut connection_attempt: u32 = 0;
    // Task 2.3b's handshake watchdog: accumulates across every attempt below (reconnects,
    // redirects, lost-connection retries) and only resets once `SessionEvent::InGame` actually
    // fires — see `ClientConfig::handshake_timeout`'s docs for the incident this defends against.
    let mut handshake_deadline = Instant::now() + config.handshake_timeout;
    // Absolute cap on how far map-download progress may push `handshake_deadline` out.
    let mut hard_deadline = Instant::now() + config.handshake_hard_cap;
    // Whether any connection of this `Client::connect` call has ever reached in game.
    let mut ever_in_game = false;

    loop {
        // Task 2.3b: checked first, before anything else this loop iteration would do — a flood
        // of Reconnect/Redirect outcomes (or a long run of lost-connection backoffs, see that
        // outcome's own now-deadline-aware sleep below) must not be able to keep this loop
        // spinning past the watchdog just because each individual step happens to complete fast.
        if Instant::now() >= handshake_deadline {
            give_up_on_watchdog(
                &events_tx,
                target,
                connection_attempt,
                config.handshake_timeout,
                ever_in_game,
            );
            return;
        }

        // Review finding F5: a dropped channel means the same thing as an explicit `Disconnect`
        // here too — see `run_one_connection`'s identical check for why.
        let should_abort = || {
            matches!(
                control_rx.try_recv(),
                Ok(Control::Disconnect) | Err(mpsc::TryRecvError::Disconnected)
            ) || Instant::now() >= handshake_deadline
        };
        // Review round 1, finding F1: checked before *every* connect attempt this loop ever makes
        // — the first one, every reconnect, and every redirect target — not just once against the
        // caller's original argument. A server that redirects (or, on a future non-loopback
        // caller, a DNS/routing change that reconnect somehow lands on a different address) must
        // not be able to walk this driver onto an address D-027/D-038's allow-list never approved.
        if let Err(e) = crate::live_servers::check(target, &config.name, &config.live_servers) {
            events_tx.send(ClientEvent::GaveUp {
                reason: format!("live-servers safety switch refused {target}: {e}"),
                category: GaveUpCategory::LocalError,
            });
            return;
        }

        if !wait_for_attempt_slot(target, &should_abort) {
            // Task 2.3b: tell the two ways `should_abort` can have fired apart — the watchdog
            // expiring while queued for a slot is not the same "local" condition as the channel
            // having been dropped/an explicit disconnect, and deserves its own category/log line
            // rather than a misleading generic "local error".
            if Instant::now() >= handshake_deadline {
                continue; // the top-of-loop check above will report it uniformly.
            }
            events_tx.send(ClientEvent::GaveUp {
                reason: "disconnected while waiting for a connection-attempt slot".to_string(),
                category: GaveUpCategory::LocalError,
            });
            return;
        }
        connection_attempt += 1;
        tracing::info!(
            addr = %target,
            attempt = connection_attempt,
            "connecting"
        );

        // Re-associate the one long-lived socket with this attempt's target (a no-op for a plain
        // reconnect; a new port for a redirect). See `socket` above for why it is reused.
        if let Err(e) = socket.connect(target) {
            events_tx.send(ClientEvent::GaveUp {
                reason: format!("failed to connect the socket to {target}: {e}"),
                category: GaveUpCategory::LocalError,
            });
            return;
        }
        drain_stale_datagrams(&socket);

        let start = Instant::now();
        let mut session = Session::new(config.clone());
        session.connect(Duration::ZERO);
        send_all(&socket, session.flush(Duration::ZERO));

        let mut reached_in_game = false;
        let outcome = run_one_connection(
            &socket,
            &mut session,
            &config,
            start,
            &events_tx,
            &input_rx,
            &control_rx,
            &mut latest_input,
            &mut reached_in_game,
            &mut handshake_deadline,
            hard_deadline,
        );

        // Review finding F7: a connection that made it in-game (even briefly) before ending is
        // not "still failing to connect" — reset the backoff/attempt-count state so a *later*,
        // unrelated disconnection reconnects promptly again instead of inheriting whatever backoff
        // an earlier, now-irrelevant string of failures left behind. Applied unconditionally here
        // (harmless for the outcomes that already reset it themselves below) rather than
        // duplicated into just the `LostConnection` arm.
        //
        // Task 2.3b: the handshake watchdog resets the same way and for the same reason — once a
        // session has genuinely reached in-game, a *later*, unrelated disconnection gets a full
        // fresh `handshake_timeout` budget to reconnect within, rather than inheriting whatever is
        // left over from a budget that already did its job once.
        if reached_in_game {
            ever_in_game = true;
            backoff = MIN_BACKOFF;
            reconnect_attempt = 0;
            handshake_deadline = Instant::now() + config.handshake_timeout;
            hard_deadline = Instant::now() + config.handshake_hard_cap;
            server_reconnects_pending = 0;
        }

        match outcome {
            ConnectionOutcome::Stop => {
                events_tx.send(ClientEvent::MarginSummary(session.margin_summary()));
                return;
            }
            ConnectionOutcome::KickedOrBanned => {
                // CLAUDE.md/live-play policy: never auto-reconnect after a kick/ban (or any other
                // peer-close reason `should_reconnect_after_peer_close` didn't recognise as
                // transient — review finding F2).
                events_tx.send(ClientEvent::MarginSummary(session.margin_summary()));
                events_tx.send(ClientEvent::GaveUp {
                    reason: "disconnected by the server (kick/ban) — not reconnecting".to_string(),
                    category: GaveUpCategory::KickedOrBanned,
                });
                return;
            }
            ConnectionOutcome::ProtocolViolation => {
                // Review finding F4: final, same shape as `KickedOrBanned` above — never retried.
                events_tx.send(ClientEvent::MarginSummary(session.margin_summary()));
                events_tx.send(ClientEvent::GaveUp {
                    reason: "local protocol violation — not reconnecting".to_string(),
                    category: GaveUpCategory::ProtocolViolation,
                });
                return;
            }
            ConnectionOutcome::HandshakeTimedOut => {
                // Task 2.3b: the watchdog fired *inside* this connection attempt (still mid
                // handshake, or online but never in-game) rather than between attempts — same
                // final shape as `KickedOrBanned`/`ProtocolViolation` above.
                events_tx.send(ClientEvent::MarginSummary(session.margin_summary()));
                give_up_on_watchdog(
                    &events_tx,
                    target,
                    connection_attempt,
                    config.handshake_timeout,
                    ever_in_game,
                );
                return;
            }
            ConnectionOutcome::Reconnect => {
                // Review finding F4: the *server* asked us to reconnect, but our own connection
                // is still technically open from its point of view — gracefully close it first
                // (one bot per server: the old slot must be seen closing before/without a new one
                // appearing) rather than abandoning it mid-flight and simply opening a fresh
                // connection underneath.
                //
                // Task 2.3b (root cause of the Swarfey incident, see the BUILD REPORT): this arm
                // used to be silent and unbounded — no log, no `ClientEvent`, no backoff — so a
                // peer answering every handshake with `reconnect@ddnet.org` made this loop spin
                // until the process-wide rate limiter stalled it, and nothing above ever noticed.
                // Now: every request is logged and surfaced, and only
                // `MAX_SERVER_RECONNECTS_BEFORE_IN_GAME` of them are followed before the session
                // has been in game.
                server_reconnects_pending += 1;
                if server_reconnects_pending > MAX_SERVER_RECONNECTS_BEFORE_IN_GAME {
                    session.disconnect(Some("driver: reconnect loop, giving up"));
                    send_all(&socket, session.flush(Instant::now().duration_since(start)));
                    tracing::error!(
                        addr = %target,
                        attempts = connection_attempt,
                        requests = server_reconnects_pending,
                        "server requested reconnect again before we were ever in game: reconnect loop, giving up (no further retries)"
                    );
                    events_tx.send(ClientEvent::MarginSummary(session.margin_summary()));
                    events_tx.send(ClientEvent::GaveUp {
                        reason: format!(
                            "reconnect loop: the server sent reconnect@ddnet.org {} times without the session ever reaching in game ({} connection attempt(s))",
                            server_reconnects_pending, connection_attempt
                        ),
                        category: GaveUpCategory::ReconnectLoop,
                    });
                    return;
                }
                if attempt_cap_reached(ever_in_game, connection_attempt) {
                    session.disconnect(Some("driver: attempt budget exhausted"));
                    send_all(&socket, session.flush(Instant::now().duration_since(start)));
                    events_tx.send(ClientEvent::MarginSummary(session.margin_summary()));
                    give_up_on_attempt_cap(&events_tx, target, connection_attempt);
                    return;
                }
                session.disconnect(Some("driver: server requested reconnect"));
                send_all(&socket, session.flush(Instant::now().duration_since(start)));
                backoff = MIN_BACKOFF; // a server-requested reconnect is not a failure
                tracing::info!(
                    addr = %target,
                    attempt = connection_attempt,
                    "server requested reconnect (reconnect@ddnet.org); reconnecting once on the same socket"
                );
                events_tx.send(ClientEvent::ServerRequestedReconnect {
                    addr: target,
                    attempt: connection_attempt,
                });
                continue;
            }
            ConnectionOutcome::Redirect(port) => {
                if redirects_followed >= 1 {
                    session.disconnect(Some("driver: refusing a second redirect"));
                    send_all(&socket, session.flush(Instant::now().duration_since(start)));
                    events_tx.send(ClientEvent::MarginSummary(session.margin_summary()));
                    events_tx.send(ClientEvent::RedirectRefused {
                        reason: "refusing a second redirect in the same session (loop protection)".to_string(),
                    });
                    events_tx.send(ClientEvent::GaveUp {
                        reason: "redirect loop protection".to_string(),
                        category: GaveUpCategory::RedirectLoop,
                    });
                    return;
                }
                if attempt_cap_reached(ever_in_game, connection_attempt) {
                    session.disconnect(Some("driver: attempt budget exhausted"));
                    send_all(&socket, session.flush(Instant::now().duration_since(start)));
                    events_tx.send(ClientEvent::MarginSummary(session.margin_summary()));
                    give_up_on_attempt_cap(&events_tx, target, connection_attempt);
                    return;
                }
                // Review finding F4: same reasoning as `Reconnect` above — close the old
                // connection before opening a new one to the redirect target.
                session.disconnect(Some("driver: following a redirect"));
                send_all(&socket, session.flush(Instant::now().duration_since(start)));
                redirects_followed += 1;
                target = SocketAddr::new(target.ip(), port);
                events_tx.send(ClientEvent::RedirectFollowed { to: target });
                backoff = MIN_BACKOFF;
                continue;
            }
            ConnectionOutcome::LostConnection => {
                // Review F1: a retry-worthy peer `CLOSE` ("This server is full", "Server shutdown",
                // ...) or a timeout before the session was ever in game gets exactly one retry.
                if attempt_cap_reached(ever_in_game, connection_attempt) {
                    session.disconnect(Some("driver: attempt budget exhausted"));
                    send_all(&socket, session.flush(Instant::now().duration_since(start)));
                    events_tx.send(ClientEvent::MarginSummary(session.margin_summary()));
                    give_up_on_attempt_cap(&events_tx, target, connection_attempt);
                    return;
                }
                reconnect_attempt += 1;
                events_tx.send(ClientEvent::ReconnectAttempt {
                    attempt: reconnect_attempt,
                    addr: target,
                    backoff,
                });
                // Review finding F7: interruptible — `Client::disconnect()` (or `Client` simply
                // being dropped, review finding F5) must not have to wait out a full, up-to-30s
                // backoff sleep before this thread actually notices and stops. A deadline-based
                // loop (rather than one `recv_timeout(backoff)` call) so a `Control::SetTeam`
                // arriving mid-backoff (task 8.4a — there is no live connection to send it on
                // right now) does not end the wait early either: it is simply dropped, and the
                // remaining backoff keeps counting down.
                //
                // Task 2.3b: also capped at `WATCHDOG_POLL_INTERVAL` per wait, independent of
                // `Control::SetTeam` arriving at all — `MAX_BACKOFF` (30s) is far longer than the
                // default handshake watchdog (15s), so a single uninterrupted `recv_timeout(backoff)`
                // could sleep straight through the deadline without ever re-checking it. The
                // deadline is re-checked once per wake, same as the loop this is nested inside.
                let backoff_deadline = Instant::now() + backoff;
                loop {
                    if Instant::now() >= handshake_deadline {
                        break;
                    }
                    let remaining = backoff_deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        break;
                    }
                    match control_rx.recv_timeout(remaining.min(WATCHDOG_POLL_INTERVAL)) {
                        Ok(Control::Disconnect) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                            events_tx.send(ClientEvent::MarginSummary(session.margin_summary()));
                            return;
                        }
                        Ok(Control::SetTeam(_)) => continue,
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    }
                }
                backoff = (backoff * 2).min(MAX_BACKOFF);
                continue;
            }
        }
    }
}

#[cfg(test)]
mod duplicate_connected_invariant_tests {
    use super::*;
    use crate::session::{ClientConfig, Session};

    /// Task 2.3b's hard safety invariant (`reject_duplicate_connected`), tested directly rather
    /// than by trying to coax a real `Session`/`Connection` into actually firing `Connected`
    /// twice — per `ddai_net::conn::Connection`'s own docs (and this task's own investigation,
    /// see the BUILD REPORT), that is not something the current code can do at all, which is
    /// exactly why this is defense-in-depth: the function's own contract ("first call: pass;
    /// second call: hard stop") is what matters here, independent of whether anything can trigger
    /// it today.
    #[test]
    fn first_connected_passes_second_is_a_hard_stop() {
        let socket = UdpSocket::bind("127.0.0.1:0").expect("bind a throwaway socket");
        let (events_tx, events_rx) = event_channel::channel();
        let mut session = Session::new(ClientConfig::default());
        let mut seen_connected = false;

        let first = reject_duplicate_connected(&mut seen_connected, &socket, &mut session, Duration::ZERO, &events_tx);
        assert!(first.is_none(), "the first Connected must be allowed through");
        assert!(seen_connected);

        let second = reject_duplicate_connected(&mut seen_connected, &socket, &mut session, Duration::ZERO, &events_tx);
        assert!(
            matches!(second, Some(ConnectionOutcome::ProtocolViolation)),
            "a second Connected must be a hard ProtocolViolation stop, got {second:?}"
        );

        // A `SessionEvent::ProtocolViolation` must have been surfaced through the event channel,
        // the same path every other protocol violation uses — not a silent internal-only stop.
        let mut saw_violation = false;
        while let Some(ev) = events_rx.try_recv() {
            if let ClientEvent::Session(inner) = &ev
                && matches!(**inner, SessionEvent::ProtocolViolation { .. })
            {
                saw_violation = true;
            }
        }
        assert!(
            saw_violation,
            "expected a SessionEvent::ProtocolViolation to have been sent"
        );
    }
}

#[cfg(test)]
mod give_up_message_tests {
    use super::*;

    fn last_gave_up(events_rx: &event_channel::Receiver) -> (String, GaveUpCategory) {
        match events_rx.try_recv() {
            Some(ClientEvent::GaveUp { reason, category }) => (reason, category),
            other => panic!("expected GaveUp, got {other:?}"),
        }
    }

    /// Review F2: an outage after being in game must not be reported as a failed handshake.
    #[test]
    fn watchdog_after_an_in_game_loss_reports_an_exhausted_reconnect_budget() {
        let (tx, rx) = event_channel::channel();
        let addr: SocketAddr = "127.0.0.1:8303".parse().unwrap();

        give_up_on_watchdog(&tx, addr, 3, Duration::from_secs(15), true);
        let (reason, category) = last_gave_up(&rx);
        assert_eq!(category, GaveUpCategory::ReconnectBudgetExhausted);
        assert!(
            reason.starts_with("reconnect budget exhausted after in-game loss"),
            "{reason}"
        );
        assert!(!reason.contains("handshake watchdog"), "{reason}");

        give_up_on_watchdog(&tx, addr, 1, Duration::from_secs(15), false);
        let (reason, category) = last_gave_up(&rx);
        assert_eq!(category, GaveUpCategory::HandshakeTimeout);
        assert!(reason.starts_with("handshake watchdog: not in game"), "{reason}");
    }

    #[test]
    fn attempt_cap_applies_only_before_the_first_in_game() {
        assert!(!attempt_cap_reached(false, 0));
        assert!(!attempt_cap_reached(false, 1));
        assert!(attempt_cap_reached(false, 2));
        assert!(!attempt_cap_reached(true, 50));
    }
}
