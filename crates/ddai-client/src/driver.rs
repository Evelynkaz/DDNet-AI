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
                None
            } else {
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
    /// A second redirect was requested; refused (loop protection) — the driver thread ends.
    RedirectRefused { reason: String },
    /// The driver thread is ending and will not reconnect (a kick/ban, an explicit
    /// [`Client::disconnect`], or a fatal local error such as failing to bind a socket at all).
    GaveUp { reason: String },
    /// A convenience alongside every [`SessionEvent::Snapshot`]: our own tee's position, when
    /// that snapshot's `PlayerInfo::local == 1` character could be found (task e2e scenario b —
    /// "the tee's own position in snapshots changes as expected" — needs this without every
    /// caller re-deriving it from [`Session::latest_view`] itself).
    OwnPosition { tick: i32, x: i32, y: i32 },
    /// Emitted once, right before the driver thread ends (for any reason) — the whole session's
    /// `NETMSG_INPUTTIMING` margin distribution (task e2e scenario h: "measure the fraction of
    /// inputs arriving in time ... report the margin distribution").
    MarginSummary(MarginSummary),
}

impl ClientEvent {
    /// Review finding F11 (round 2 — round 1 only covered `Snapshot`/`OwnPosition`, but
    /// `GameMessage`/`ExGameMessage` are exactly as high-frequency and just as capable of growing
    /// the queue without bound: a busy server's chat/kill-feed/pickup messages, or a flood of
    /// them, with nobody draining `events()`): whether this event may be evicted (oldest-first) to
    /// keep [`event_channel`]'s queue under [`EVENT_QUEUE_CAP`] — every per-tick/per-message event
    /// where losing an old one under sustained backpressure just means missing one stale
    /// position/tick/chat line, never a missed state transition or terminal event.
    fn is_droppable(&self) -> bool {
        matches!(self, ClientEvent::OwnPosition { .. })
            || matches!(
                self,
                ClientEvent::Session(ev)
                    if matches!(**ev, SessionEvent::Snapshot { .. } | SessionEvent::GameMessage(_) | SessionEvent::ExGameMessage(_))
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
            });
            tx.send(ClientEvent::OwnPosition { tick: 4, x: 0, y: 0 });

            let expected = vec![
                ClientEvent::OwnPosition { tick: 1, x: 0, y: 0 },
                ClientEvent::Session(Box::new(SessionEvent::Connected)),
                ClientEvent::OwnPosition { tick: 2, x: 0, y: 0 },
                ClientEvent::OwnPosition { tick: 3, x: 0, y: 0 },
                ClientEvent::GaveUp {
                    reason: "test".to_string(),
                },
                ClientEvent::OwnPosition { tick: 4, x: 0, y: 0 },
            ];
            let mut actual = Vec::new();
            while let Some(ev) = rx.try_recv() {
                actual.push(ev);
            }
            assert_eq!(actual, expected);
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
        && let Some(own_id) = view.players().iter().find(|p| p.info.local == 1).map(|p| p.id)
        && let Some(character) = view.character(own_id)
    {
        synthesized.push(ClientEvent::OwnPosition {
            tick: *tick,
            x: character.character.x,
            y: character.character.y,
        });
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
) -> ConnectionOutcome {
    let mut buf = [0u8; RECV_BUF_SIZE];
    loop {
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
            if let Some(outcome) = handle_session_event(ev, session, config, now, events_tx) {
                // See the comment on the identical pattern above.
                send_all(socket, session.flush(now));
                return outcome;
            }
        }
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
    let mut backoff = MIN_BACKOFF;
    let mut latest_input = default_player_input();
    let mut reconnect_attempt: u32 = 0;

    loop {
        // Review finding F5: a dropped channel means the same thing as an explicit `Disconnect`
        // here too — see `run_one_connection`'s identical check for why.
        let should_abort = || {
            matches!(
                control_rx.try_recv(),
                Ok(Control::Disconnect) | Err(mpsc::TryRecvError::Disconnected)
            )
        };
        if !wait_for_attempt_slot(target, &should_abort) {
            events_tx.send(ClientEvent::GaveUp {
                reason: "disconnected while waiting for a connection-attempt slot".to_string(),
            });
            return;
        }

        let socket = match UdpSocket::bind("0.0.0.0:0") {
            Ok(s) => s,
            Err(e) => {
                events_tx.send(ClientEvent::GaveUp {
                    reason: format!("failed to bind a local socket: {e}"),
                });
                return;
            }
        };
        if let Err(e) = socket.set_read_timeout(Some(POLL_TIMEOUT)) {
            events_tx.send(ClientEvent::GaveUp {
                reason: format!("failed to configure the socket: {e}"),
            });
            return;
        }
        if let Err(e) = socket.connect(target) {
            events_tx.send(ClientEvent::GaveUp {
                reason: format!("failed to connect the socket to {target}: {e}"),
            });
            return;
        }

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
        );

        // Review finding F7: a connection that made it in-game (even briefly) before ending is
        // not "still failing to connect" — reset the backoff/attempt-count state so a *later*,
        // unrelated disconnection reconnects promptly again instead of inheriting whatever backoff
        // an earlier, now-irrelevant string of failures left behind. Applied unconditionally here
        // (harmless for the outcomes that already reset it themselves below) rather than
        // duplicated into just the `LostConnection` arm.
        if reached_in_game {
            backoff = MIN_BACKOFF;
            reconnect_attempt = 0;
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
                });
                return;
            }
            ConnectionOutcome::ProtocolViolation => {
                // Review finding F4: final, same shape as `KickedOrBanned` above — never retried.
                events_tx.send(ClientEvent::MarginSummary(session.margin_summary()));
                events_tx.send(ClientEvent::GaveUp {
                    reason: "local protocol violation — not reconnecting".to_string(),
                });
                return;
            }
            ConnectionOutcome::Reconnect => {
                // Review finding F4: the *server* asked us to reconnect, but our own connection
                // is still technically open from its point of view — gracefully close it first
                // (one bot per server: the old slot must be seen closing before/without a new one
                // appearing) rather than abandoning it mid-flight and simply opening a fresh
                // socket underneath.
                session.disconnect(Some("driver: server requested reconnect"));
                send_all(&socket, session.flush(Instant::now().duration_since(start)));
                backoff = MIN_BACKOFF; // a server-requested reconnect is not a failure
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
                    });
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
                reconnect_attempt += 1;
                events_tx.send(ClientEvent::ReconnectAttempt {
                    attempt: reconnect_attempt,
                    addr: target,
                    backoff,
                });
                // Review finding F7: interruptible — `Client::disconnect()` (or `Client` simply
                // being dropped, review finding F5) must not have to wait out a full, up-to-30s
                // backoff sleep before this thread actually notices and stops.
                match control_rx.recv_timeout(backoff) {
                    Ok(Control::Disconnect) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                        events_tx.send(ClientEvent::MarginSummary(session.margin_summary()));
                        return;
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
                backoff = (backoff * 2).min(MAX_BACKOFF);
                continue;
            }
        }
    }
}
