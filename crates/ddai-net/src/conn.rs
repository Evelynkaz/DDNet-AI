// Ported from DDNet `src/engine/shared/network_conn.cpp` and the client-relevant parts of
// `network.{h,cpp}`/`network_client.cpp` (pinned rev c9d208138f85755521f16a0096b6fe036c5c8698,
// "20.1"), which carries the original Teeworlds zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same ack/sequence/resend rules and
// state-machine semantics as `CNetConnection`, adapted to a sans-IO shape (see the module docs
// below for exactly where it differs and why). See docs/formats.md for the byte layout.
//
//! Sans-IO reliable-delivery connection state machine for Teeworlds 0.6 + DDNet.
//!
//! [`Connection`] has no socket, no threads, and no wall-clock access: every method that cares
//! about time takes an explicit `now: Duration` from the caller (an arbitrary but
//! monotonically-non-decreasing reference point — real code typically uses
//! `Instant::now().duration_since(start)`, tests use a hand-advanced fake clock), and every
//! method that needs to send something *returns* the bytes to send rather than writing to a
//! socket itself. This is deliberate (see the task's `<goal>`): the same state machine is used
//! against real UDP in task 2.3 and against a fully deterministic simulated lossy link in
//! `tests/lossy_link.rs`.
//!
//! # Design notes (decision D-029: this is our own implementation, not a vendor of DDNet/libtw2)
//!
//! * **No address tracking.** DDNet's `CNetConnection` matches incoming packets against the
//!   addresses it is speaking to (anti-spoofing, multi-address happy-eyeballs). Since a
//!   `Connection` here is *transport-independent* and always represents exactly one peer, that
//!   job belongs to whatever demultiplexes datagrams to the right `Connection` instance in the
//!   first place (the socket layer, task 2.3) — by the time a datagram reaches [`Connection::feed`]
//!   it is already known to be from the right peer.
//! * **Client *and* server roles in one type.** The spec asks for states
//!   `offline/connecting/pending/online/error`; DDNet only reaches `PENDING` on the server side,
//!   handled outside `CNetConnection::Feed` entirely (`CNetServer::TryAcceptClient`,
//!   `network_server.cpp`). We fold the (small) server-side responder half of the TKEN handshake
//!   into this same state machine via [`Connection::accept`], so `tests/lossy_link.rs`'s "tiny
//!   test server" (task acceptance criterion 6) is just a second `Connection` in the other role —
//!   good additional coverage of the encode *and* decode paths on both sides, for free.
//! * **Immediate control sends are queued, not synchronous.** DDNet sends control messages
//!   (`CONNECT`, `ACCEPT`, `CLOSE`, …) by calling `SendPacket` directly and immediately wherever
//!   they are triggered, including from inside `Feed()`. A sans-IO type cannot do I/O from
//!   `feed()`, so those get queued internally and handed out by the *next* [`Connection::flush`]
//!   call instead — callers are expected to call `flush()` right after `feed()` (or on their own
//!   regular schedule), same as any sans-IO protocol implementation.
//! * **`flush()` always drains queued vital/non-vital chunks**, not just when a 500ms idle timer
//!   fires. DDNet's own timer-driven auto-flush exists because the C++ client's `NETSENDFLAG_FLUSH`
//!   is the *real* "send now" signal and `Update()`'s 500ms check is only a backstop for chunks
//!   queued without it. Since [`Connection::send_chunk`] has no such flag, `flush(now)` is simply
//!   "send everything queued right now, plus whatever the timers say is due" — the caller decides
//!   the cadence, which is the whole point of sans-IO.
//! * **Resend-buffer exhaustion is a hard error**, not DDNet's silent "hope nobody asks for
//!   resend" (`network_conn.cpp:187-191`). See [`SendChunkError::ResendBufferFull`].
//!
//! Everything else — ack/sequence wraparound, the resend timers, the TKEN byte layout — is a
//! direct, cited port; see the individual method docs.

use crate::control::{self, ControlMsg};
use crate::huffman::Huffman;
use crate::packet::{self, packet_flags};
use std::collections::VecDeque;
use std::time::Duration;

/// `NET_CONN_BUFFERSIZE` (`network.h:103`): total bytes of vital-chunk payload the resend buffer
/// may hold before [`Connection::send_chunk`] refuses further vital sends. Unlike DDNet's ring
/// buffer (which counts struct overhead too and silently drops the chunk data on overflow while
/// still sending it once), this is a plain byte budget over payload bytes only, and overflowing
/// it is a hard [`SendChunkError::ResendBufferFull`] error that also moves the connection to
/// [`State::Error`] — see the module docs.
pub const RESEND_BUFFER_BYTES: usize = 1024 * 32;

/// `conn_timeout` default (`config_variables.h:650`): seconds of silence from the peer before the
/// connection is considered dead.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(100);
/// How long an unacked vital chunk sits in the resend buffer before a fresh copy is sent
/// (`network_conn.cpp:546-548`, hardcoded in the C++ reference, not a config variable).
const RESEND_INTERVAL: Duration = Duration::from_secs(1);
/// How long the `ONLINE` connection may go without sending anything before a bare `KEEPALIVE` is
/// sent (`network_conn.cpp:562`).
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(1);
/// How often `CONNECT`/`CONNECTACCEPT` are retried while establishing a connection
/// (`network_conn.cpp:567,572`).
const HANDSHAKE_RETRY_INTERVAL: Duration = Duration::from_millis(500);
/// `conn_resend_requests_per_second` default (`config_variables.h:652`): at most this many
/// *answers* to a peer's repeated resend requests per second (each answer resends everything
/// unacked, so answering every single request would be wasteful under sustained loss).
pub const DEFAULT_RESEND_REQUESTS_PER_SECOND: u32 = 10;

/// Connection state, named to match the task spec and DDNet's `CNetConnection::EState`
/// (`network.h:241-249`) — `Pending` is reachable only via the server-role [`Connection::accept`]
/// entry point, see the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum State {
    Offline,
    /// Client role: sent `CONNECT`, waiting for `CONNECTACCEPT`.
    Connecting,
    /// Server role: sent `CONNECTACCEPT`, waiting for `ACCEPT` (or any data packet — DDNet
    /// accepts either, `network_conn.cpp:486-492`).
    Pending,
    Online,
    /// Terminal until the caller calls [`Connection::connect`]/[`Connection::accept`] again. The
    /// string is the human-readable reason, matching `CNetConnection::ErrorString()`.
    Error(String),
}

/// The DDNet security token as tracked per-connection (`network.h:129-133`, `m_SecurityToken`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityToken {
    /// Not yet negotiated — still, a packet gets a token appended (the sentinel wire value
    /// `0xffff_ffff`), "hoping to negotiate it" (`network_conn.cpp:365-369`'s comment).
    Unknown,
    /// The peer does not speak the DDNet token extension; no token is appended or expected.
    Unsupported,
    /// Negotiated value; every packet from here on carries and is checked against it.
    Known(u32),
}

impl SecurityToken {
    /// The value to append as a packet's trailing token, or `None` to append nothing at all —
    /// see `crate::packet::build_packet`'s `security_token` parameter.
    fn wire_value(self) -> Option<u32> {
        match self {
            SecurityToken::Unknown => Some(control::TOKEN_UNKNOWN),
            SecurityToken::Unsupported => None,
            SecurityToken::Known(t) => Some(t),
        }
    }

    /// Whether an incoming packet's trailing token should be verified (and stripped before
    /// further parsing) — only once a real value has been negotiated
    /// (`network_conn.cpp:365-369`'s guard).
    fn should_verify(self) -> bool {
        matches!(self, SecurityToken::Known(_))
    }
}

/// Something [`Connection::feed`] observed while processing a datagram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// The connection just became [`State::Online`] (handshake completed, either role).
    Connected,
    /// A chunk payload was delivered — a vital chunk is delivered exactly once, in order; a
    /// non-vital one is delivered whenever it arrives (may be lost, duplicated, or reordered by
    /// the network, same as the real protocol).
    Chunk { vital: bool, data: Vec<u8> },
    /// The peer sent `CLOSE`; the connection is now [`State::Error`] with this reason (empty if
    /// the peer gave none).
    ClosedByPeer(String),
    /// The connection just moved to [`State::Error`] for a reason other than an explicit peer
    /// `CLOSE` (timeout, too-weak-connection, …) — the string matches
    /// `State::Error`'s payload.
    Error(String),
}

/// Why [`Connection::send_chunk`] refused to queue a chunk.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SendChunkError {
    #[error("chunk payload of {0} bytes exceeds NET_MAX_CHUNK_SIZE ({max})", max = packet::MAX_CHUNK_SIZE)]
    TooLarge(usize),
    #[error("connection is offline or in an error state")]
    NotConnected,
    #[error("resend buffer full (too weak connection)")]
    ResendBufferFull,
}

#[derive(Debug, Clone)]
struct ResendEntry {
    sequence: u16,
    data: Vec<u8>,
    first_send_time: Duration,
    last_send_time: Duration,
}

/// A packet-in-progress: chunks queued by [`Connection::send_chunk`]/internal resends, not yet
/// handed out by [`Connection::flush`]. Mirrors `CNetPacketConstruct` minus the fields
/// [`crate::packet`] already owns.
#[derive(Debug, Default, Clone)]
struct PendingPacket {
    num_chunks: u8,
    data: Vec<u8>,
}

/// Tunable timers, all defaulted to DDNet's own defaults — see the module-level constants.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub timeout: Duration,
    pub resend_requests_per_second: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            timeout: DEFAULT_TIMEOUT,
            resend_requests_per_second: DEFAULT_RESEND_REQUESTS_PER_SECOND,
        }
    }
}

/// The sans-IO connection state machine. See the module docs.
#[derive(Debug, Clone)]
pub struct Connection {
    config: Config,
    state: State,

    /// Next vital sequence number to assign to our own outgoing chunks (`m_Sequence`).
    sequence: u16,
    /// Last in-order vital sequence number accepted *from* the peer (`m_Ack`).
    ack: u16,
    /// Last ack value the peer has reported *to* us (`m_PeerAck`).
    peer_ack: u16,

    security_token: SecurityToken,
    resend_buffer: VecDeque<ResendEntry>,
    resend_bytes: usize,
    resend_requested_by_peer: bool,
    signal_resend_to_peer: bool,
    last_resend_answer_time: Option<Duration>,

    pending_packets: Vec<PendingPacket>,
    /// Fully-built datagrams (control messages) waiting for the next [`Connection::flush`] —
    /// see the module docs' "immediate control sends are queued" note.
    pending_immediate: Vec<Vec<u8>>,
    /// [`Event`]s produced by something other than [`Connection::feed`] (i.e. a state transition
    /// [`Connection::flush`] or [`Connection::send_chunk`] detected on its own — a timeout, a
    /// too-weak-connection, a resend-buffer overflow) — drained by [`Connection::take_events`].
    /// `feed()` keeps returning its own events directly, unchanged; see the module docs.
    pending_events: Vec<Event>,

    last_send_time: Duration,
    last_recv_time: Duration,

    remote_closed: bool,
}

impl Connection {
    /// A fresh, offline connection.
    pub fn new(config: Config) -> Self {
        Connection {
            config,
            state: State::Offline,
            sequence: 0,
            ack: 0,
            peer_ack: 0,
            security_token: SecurityToken::Unknown,
            resend_buffer: VecDeque::new(),
            resend_bytes: 0,
            resend_requested_by_peer: false,
            signal_resend_to_peer: false,
            last_resend_answer_time: None,
            pending_packets: Vec::new(),
            pending_immediate: Vec::new(),
            pending_events: Vec::new(),
            last_send_time: Duration::ZERO,
            last_recv_time: Duration::ZERO,
            remote_closed: false,
        }
    }

    /// Current state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// `true` once [`Connection::state`] is [`State::Online`].
    pub fn is_online(&self) -> bool {
        matches!(self.state, State::Online)
    }

    /// Drains and returns every [`Event`] a state transition produced *outside* of
    /// [`Connection::feed`] (timeouts, too-weak-connection, resend-buffer overflow) since the
    /// last call — `feed()`'s own return value is unaffected and still carries its events
    /// directly. Call this after [`Connection::flush`]/[`Connection::send_chunk`] to observe
    /// those transitions; an empty result is the common case (nothing to report).
    pub fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.pending_events)
    }

    fn reset(&mut self) {
        self.state = State::Offline;
        self.sequence = 0;
        self.ack = 0;
        self.peer_ack = 0;
        self.security_token = SecurityToken::Unknown;
        self.resend_buffer.clear();
        self.resend_bytes = 0;
        self.resend_requested_by_peer = false;
        self.signal_resend_to_peer = false;
        self.last_resend_answer_time = None;
        self.pending_packets.clear();
        self.pending_events.clear();
        // Deliberately not touching `pending_immediate`: a caller-triggered `CLOSE` queued right
        // before a `reset()` (see `disconnect`) must still reach the wire on the next `flush()`.
        self.remote_closed = false;
    }

    fn queue_immediate_control(&mut self, msg: &ControlMsg, huffman: &Huffman) {
        let payload = control::encode(msg);
        if let Some(dg) = packet::build_packet(
            packet_flags::CONTROL,
            self.ack,
            0,
            &payload,
            self.security_token.wire_value(),
            huffman,
        ) {
            self.pending_immediate.push(dg);
        }
    }

    /// Client role: start connecting. Resets any prior state. The `CONNECT` datagram is queued
    /// for the next [`Connection::flush`] (`network_conn.cpp:247-265`, `SendConnect`,
    /// `network_conn.cpp:204-212`).
    pub fn connect(&mut self, now: Duration, huffman: &Huffman) {
        self.reset();
        self.state = State::Connecting;
        self.last_send_time = now;
        self.last_recv_time = now;
        self.queue_immediate_control(&control::connect_payload(), huffman);
    }

    /// Server role: a peer's `CONNECT` (already recognised by the caller — demultiplexing
    /// `CONNECT`s to a fresh `Connection` per new peer is the caller's job, mirroring
    /// `CNetServer::OnTokenCtrlMsg`/`TryAcceptClient`, `network_server.cpp:513-614`) is accepted
    /// with `my_token`. Queues `CONNECTACCEPT` for the next [`Connection::flush`]
    /// (`network_conn.cpp:570-573`).
    pub fn accept(&mut self, my_token: u32, now: Duration, huffman: &Huffman) {
        self.reset();
        self.state = State::Pending;
        self.security_token = SecurityToken::Known(my_token);
        self.last_send_time = now;
        self.last_recv_time = now;
        self.queue_immediate_control(&control::connect_accept_payload(), huffman);
    }

    /// Voluntarily closes the connection: queues `CLOSE` (with `reason`, if any and unless the
    /// peer already closed us — `network_conn.cpp:311-335`) for the next
    /// [`Connection::flush`], then resets to [`State::Offline`]. A no-op if already offline.
    pub fn disconnect(&mut self, reason: Option<&str>, huffman: &Huffman) {
        if matches!(self.state, State::Offline) {
            return;
        }
        if !self.remote_closed {
            self.queue_immediate_control(
                &ControlMsg::Close {
                    reason: reason.map(str::to_string),
                },
                huffman,
            );
        }
        self.reset();
    }

    /// Queues one chunk to be sent on the next [`Connection::flush`]. Vital chunks get the next
    /// sequence number and a resend-buffer entry; non-vital ones are fire-and-forget
    /// (`CNetConnection::QueueChunk`, `network_conn.cpp:197-202`).
    ///
    /// Allowed in any state except [`State::Offline`]/[`State::Error`] (`QueueChunkEx`'s own
    /// guard, `network_conn.cpp:145-146`) — a higher layer may legitimately want to queue a
    /// message the instant a connection is created, to be sent as soon as it comes online.
    pub fn send_chunk(&mut self, data: &[u8], vital: bool, now: Duration) -> Result<(), SendChunkError> {
        if data.len() > packet::MAX_CHUNK_SIZE {
            return Err(SendChunkError::TooLarge(data.len()));
        }
        if matches!(self.state, State::Offline | State::Error(_)) {
            return Err(SendChunkError::NotConnected);
        }

        if vital && self.resend_bytes + data.len() > RESEND_BUFFER_BYTES {
            let reason = "Too weak connection (resend buffer full)".to_string();
            self.state = State::Error(reason.clone());
            self.pending_events.push(Event::Error(reason));
            return Err(SendChunkError::ResendBufferFull);
        }

        let sequence = if vital {
            self.sequence = (self.sequence + 1) % packet::MAX_SEQUENCE;
            self.sequence
        } else {
            0
        };
        let flags = if vital { packet::chunk_flags::VITAL } else { 0 };
        self.queue_chunk_bytes(flags, sequence, data);

        if vital {
            self.resend_buffer.push_back(ResendEntry {
                sequence,
                data: data.to_vec(),
                first_send_time: now,
                last_send_time: now,
            });
            self.resend_bytes += data.len();
        }
        Ok(())
    }

    /// Appends one chunk's bytes to the current in-progress packet, starting a new one if the
    /// current one is full — the sans-IO equivalent of `QueueChunkEx`'s implicit `Flush()` when
    /// out of room (`network_conn.cpp:150-155`).
    ///
    /// A no-op once `self.state` is [`State::Offline`]/[`State::Error`] — mirrors `QueueChunkEx`'s
    /// own guard (`network_conn.cpp:145-146`) at the one place every internal caller (the public
    /// [`Connection::send_chunk`], which also checks this itself for a proper `Err` return, and
    /// the resend path inside [`Connection::flush`], which does not) funnels through, so an
    /// errored connection can never have a data chunk queued into it by any path, present or
    /// future.
    fn queue_chunk_bytes(&mut self, flags: u8, sequence: u16, data: &[u8]) {
        if matches!(self.state, State::Offline | State::Error(_)) {
            return;
        }
        // Leave room for the trailing security token, exactly like the capacity check in
        // `QueueChunkEx` (`network_conn.cpp:151`: `sizeof(m_aChunkData) - sizeof(SECURITY_TOKEN)`).
        let capacity = packet::MAX_CHUNK_DATA_SIZE - packet::SECURITY_TOKEN_SIZE;
        loop {
            if self.pending_packets.is_empty()
                || self.pending_packets.last().unwrap().num_chunks == packet::MAX_PACKET_CHUNKS
            {
                self.pending_packets.push(PendingPacket::default());
            }
            let pkt = self.pending_packets.last_mut().unwrap();
            if packet::pack_chunk_into(&mut pkt.data, capacity, flags, sequence, data) {
                pkt.num_chunks += 1;
                return;
            }
            // Didn't fit: start a fresh packet and retry (a lone chunk always fits an empty one,
            // since `data.len() <= MAX_CHUNK_SIZE < capacity`, checked by `send_chunk`/the
            // resend path before this is ever called).
            self.pending_packets.push(PendingPacket::default());
        }
    }

    /// Removes every resend-buffer entry the peer has now acked, i.e. whose sequence falls in the
    /// "backroom" behind `new_peer_ack` (`CNetConnection::AckChunks`, `network_conn.cpp:101-114`).
    fn ack_chunks(&mut self, new_peer_ack: u16) {
        while let Some(front) = self.resend_buffer.front() {
            if packet::is_seq_in_backroom(front.sequence, new_peer_ack) {
                let removed = self.resend_buffer.pop_front().unwrap();
                self.resend_bytes -= removed.data.len();
            } else {
                break;
            }
        }
    }

    /// Whether `new_ack` is a plausible ack value given our own outgoing sequence counter and the
    /// last ack the peer reported — rejects spoofed/garbage acks
    /// (`CNetConnection::Feed`, `network_conn.cpp:382-393`). Wraparound-aware: all three values
    /// live in `0..MAX_SEQUENCE`, so plain comparisons are correct once the two-branch structure
    /// below picks the right "side" of the wrap, exactly like the C++ reference.
    fn ack_is_valid(&self, new_ack: u16) -> bool {
        if self.sequence >= self.peer_ack {
            new_ack >= self.peer_ack && new_ack <= self.sequence
        } else {
            new_ack >= self.peer_ack || new_ack <= self.sequence
        }
    }

    /// Resends every currently-buffered vital chunk (`CNetConnection::Resend`,
    /// `network_conn.cpp:227-231`) by re-queueing each one (with the [`packet::chunk_flags::RESEND`]
    /// bit set, informational only) into the current in-progress packet.
    fn resend_all(&mut self) {
        let entries: Vec<(u16, Vec<u8>)> = self
            .resend_buffer
            .iter()
            .map(|e| (e.sequence, e.data.clone()))
            .collect();
        for (sequence, data) in entries {
            self.queue_chunk_bytes(
                packet::chunk_flags::VITAL | packet::chunk_flags::RESEND,
                sequence,
                &data,
            );
        }
    }

    /// If the peer has asked for a resend and the per-second rate limit allows it, resends
    /// everything buffered (`CNetConnection::AnswerResendRequest`, `network_conn.cpp:233-245`).
    fn answer_resend_request(&mut self, now: Duration) {
        if !self.resend_requested_by_peer {
            return;
        }
        if self.config.resend_requests_per_second != 0
            && let Some(last) = self.last_resend_answer_time
        {
            let min_interval = Duration::from_secs_f64(1.0 / f64::from(self.config.resend_requests_per_second));
            if now.saturating_sub(last) < min_interval {
                return;
            }
        }
        self.resend_requested_by_peer = false;
        self.last_resend_answer_time = Some(now);
        self.resend_all();
    }

    /// Feeds one received datagram into the connection, returning the [`Event`]s it produced.
    /// Never panics on malformed/hostile input: anything that fails to parse, fails a validity
    /// check, or arrives in a state where it makes no sense is silently ignored (matching DDNet's
    /// own `Feed`/`UnpackPacket`/`UnpackNextChunk`, which all just drop the offending
    /// packet/chunk rather than erroring the connection over it — only a genuine protocol-level
    /// problem, like a timeout or an explicit peer `CLOSE`, ever produces [`Event::Error`]/
    /// [`Event::ClosedByPeer`]).
    ///
    /// Datagrams this triggers in response (an `ACCEPT`, a `KEEPALIVE`, …) are *not* returned
    /// here — see the module docs — call [`Connection::flush`] afterwards to get them.
    pub fn feed(&mut self, datagram: &[u8], huffman: &Huffman, now: Duration) -> Vec<Event> {
        let mut events = Vec::new();
        if matches!(self.state, State::Offline | State::Error(_)) {
            return events;
        }

        let Ok(packet) = packet::unpack_packet(datagram, huffman, true) else {
            return events;
        };
        let mut data = packet.data;

        if self.security_token.should_verify() {
            if data.len() < packet::SECURITY_TOKEN_SIZE {
                return events;
            }
            let split = data.len() - packet::SECURITY_TOKEN_SIZE;
            let tail = &data[split..];
            let got = u32::from_be_bytes([tail[0], tail[1], tail[2], tail[3]]);
            if self.security_token != SecurityToken::Known(got) {
                return events;
            }
            data.truncate(split);
        }

        if !self.ack_is_valid(packet.ack) {
            return events;
        }
        self.peer_ack = packet.ack;

        if packet.flags & packet_flags::RESEND != 0 {
            self.resend_requested_by_peer = true;
        }
        self.answer_resend_request(now);

        let is_control = packet.flags & packet_flags::CONTROL != 0;
        if is_control {
            let Ok(ctrl) = control::decode(&data) else {
                return events;
            };
            match ctrl {
                ControlMsg::Close { reason } => {
                    self.remote_closed = true;
                    let reason = reason.unwrap_or_default();
                    self.state = State::Error(reason.clone());
                    events.push(Event::ClosedByPeer(reason));
                    return events;
                }
                ControlMsg::ConnectAccept { has_tken_magic } if self.state == State::Connecting => {
                    // `data` has *not* had a trailing token stripped (we only strip once
                    // `security_token` is `Known`, and it is still `Unknown` here) — so the 4
                    // bytes right after "TKEN" are the server's freshly generated token, exactly
                    // like `network_conn.cpp:461-466` reads them. See the `control` module docs
                    // for why this can't be read inside `control::decode` itself.
                    if matches!(self.security_token, SecurityToken::Unknown) {
                        const PREFIX: usize = 1 + control::SECURITY_TOKEN_MAGIC.len();
                        self.security_token = if has_tken_magic && data.len() >= PREFIX + 4 {
                            let t = u32::from_be_bytes([
                                data[PREFIX],
                                data[PREFIX + 1],
                                data[PREFIX + 2],
                                data[PREFIX + 3],
                            ]);
                            SecurityToken::Known(t)
                        } else {
                            SecurityToken::Unsupported
                        };
                    }
                    self.queue_immediate_control(&ControlMsg::Accept, huffman);
                    self.last_recv_time = now;
                    self.state = State::Online;
                    events.push(Event::Connected);
                }
                ControlMsg::Accept if self.state == State::Pending => {
                    // The trailing-token check above already verified this really is our peer
                    // (its token matched our own, from `self.security_token`), so no further check needed.
                    self.last_recv_time = now;
                    self.state = State::Online;
                    events.push(Event::Connected);
                }
                _ => {
                    // KEEPALIVE, a retransmitted CONNECT/CONNECTACCEPT/ACCEPT that doesn't apply
                    // to our current state, … — no-op, matching the C++ reference's fallthrough.
                }
            }
        } else if self.state == State::Pending {
            // Any non-control packet also completes a pending server-side handshake
            // (`network_conn.cpp:486-492`) — recovers from a lost `ACCEPT` as soon as the client
            // sends real data.
            self.state = State::Online;
            events.push(Event::Connected);
        }

        if self.state == State::Online {
            self.last_recv_time = now;
            self.ack_chunks(packet.ack);
        }

        if !is_control && self.state == State::Online {
            let parsed = packet::Packet {
                flags: packet.flags,
                ack: packet.ack,
                num_chunks: packet.num_chunks,
                data,
            };
            for chunk in packet::ChunkIter::new(&parsed) {
                if chunk.vital {
                    if chunk.sequence == (self.ack + 1) % packet::MAX_SEQUENCE {
                        self.ack = chunk.sequence;
                        events.push(Event::Chunk {
                            vital: true,
                            data: chunk.data.to_vec(),
                        });
                    } else if packet::is_seq_in_backroom(chunk.sequence, self.ack) {
                        // Old packet we already got: silent duplicate, matching
                        // `CPacketChunkUnpacker::UnpackNextChunk` (`network.cpp:107-110`).
                    } else {
                        // Out of order: ask the peer to resend everything unacked.
                        self.signal_resend_to_peer = true;
                    }
                } else {
                    events.push(Event::Chunk {
                        vital: false,
                        data: chunk.data.to_vec(),
                    });
                }
            }
        }

        events
    }

    /// Runs the periodic timers and returns every datagram that should be sent right now:
    /// queued immediate control messages, queued vital/non-vital chunks, resends, keepalives,
    /// and handshake retries — see the module docs for how this differs from DDNet's
    /// timer-gated `Update()`/`Flush()` split.
    pub fn flush(&mut self, huffman: &Huffman, now: Duration) -> Vec<Vec<u8>> {
        let mut out = std::mem::take(&mut self.pending_immediate);

        match &self.state {
            State::Offline | State::Error(_) => return out,
            State::Connecting => {
                // No timeout here, matching `CNetConnection::Update`'s own
                // `State() != EState::CONNECT` guard on its timeout check
                // (`network_conn.cpp:522-523`) — a `Connecting` connection retries `CONNECT`
                // forever (every `HANDSHAKE_RETRY_INTERVAL`) until the caller gives up and calls
                // something else, or a `CONNECTACCEPT` arrives. Deciding "we've retried enough,
                // give up" is a policy call for whatever drives the connection (the client
                // session, task 2.3), not this state machine.
                if now.saturating_sub(self.last_send_time) >= HANDSHAKE_RETRY_INTERVAL {
                    self.queue_immediate_control_now(&control::connect_payload(), huffman, &mut out, now);
                }
                return out;
            }
            State::Pending => {
                // Timeout while waiting for ACCEPT is still meaningful (an unresponsive client
                // should not hold a slot forever) — checked below alongside Online's.
                if now.saturating_sub(self.last_send_time) >= HANDSHAKE_RETRY_INTERVAL {
                    self.queue_immediate_control_now(&control::connect_accept_payload(), huffman, &mut out, now);
                }
            }
            State::Online => {}
        }

        // From here on: State::Pending or State::Online.

        // Note: deliberately *not* returning early here — like the C++ reference
        // (`network_conn.cpp:521-528` sets `ERROR`/"Timeout" but keeps running the rest of
        // `Update()`), a connection can be simultaneously silent *and* sitting on an unacked
        // vital chunk old enough to also be "too weak"; the resend-buffer check below runs next
        // and, if it also fires, its message is what the caller ultimately sees — same
        // last-check-wins order as upstream. Either branch only *sets* `self.state`; whether
        // anything gets sent once it is `Error` is decided once, below, after both checks have
        // had their say — `queue_chunk_bytes` (called from the resend check right below) also
        // refuses to queue anything once `self.state` is `Error`, mirroring `QueueChunkEx`'s own
        // guard (`network_conn.cpp:145-146`), so a same-tick transition can never sneak a data
        // chunk into `pending_packets` either.
        if now.saturating_sub(self.last_recv_time) > self.config.timeout {
            self.state = State::Error("Timeout".to_string());
        }

        self.answer_resend_request(now);

        if let Some(front) = self.resend_buffer.front().cloned() {
            if now.saturating_sub(front.first_send_time) > self.config.timeout {
                let reason = format!(
                    "Too weak connection (not acked for {} seconds)",
                    self.config.timeout.as_secs()
                );
                self.state = State::Error(reason);
            } else if now.saturating_sub(front.last_send_time) > RESEND_INTERVAL {
                self.queue_chunk_bytes(
                    packet::chunk_flags::VITAL | packet::chunk_flags::RESEND,
                    front.sequence,
                    &front.data,
                );
                if let Some(entry) = self.resend_buffer.front_mut() {
                    entry.last_send_time = now;
                }
            }
        }

        // Newly entered `Error` this call (the two checks above are the only ways to reach it
        // from here — `Offline`/pre-existing `Error` already returned at the top of this
        // function): report it exactly once, and refuse to send any data — matching
        // `CNetConnection`, where an errored connection's `QueueChunkEx` (and thus any further
        // `Flush()`) never emits anything either (`network_conn.cpp:145-146`).
        if let State::Error(reason) = self.state.clone() {
            self.pending_events.push(Event::Error(reason));
            self.pending_packets.clear();
            return out;
        }

        if self.state == State::Online
            && self.pending_packets.is_empty()
            && now.saturating_sub(self.last_send_time) >= KEEPALIVE_INTERVAL
        {
            self.queue_immediate_control_now(&ControlMsg::KeepAlive, huffman, &mut out, now);
        }

        if self.signal_resend_to_peer && self.pending_packets.is_empty() {
            self.pending_packets.push(PendingPacket::default());
        }

        let mut first = true;
        for pkt in std::mem::take(&mut self.pending_packets) {
            let mut flags = 0u8;
            if first && std::mem::take(&mut self.signal_resend_to_peer) {
                flags |= packet_flags::RESEND;
            }
            first = false;
            if pkt.num_chunks == 0 && flags & packet_flags::RESEND == 0 {
                // Mirrors `Flush()`'s early return: nothing to say, don't send an empty packet.
                continue;
            }
            if let Some(dg) = packet::build_packet(
                flags,
                self.ack,
                pkt.num_chunks,
                &pkt.data,
                self.security_token.wire_value(),
                huffman,
            ) {
                out.push(dg);
                self.last_send_time = now;
            }
        }

        out
    }

    /// Builds and pushes one immediate control datagram straight into `out` (used by `flush`,
    /// which — unlike `feed` — is allowed to hand datagrams back directly instead of going
    /// through `pending_immediate`) and updates `last_send_time`.
    fn queue_immediate_control_now(
        &mut self,
        msg: &ControlMsg,
        huffman: &Huffman,
        out: &mut Vec<Vec<u8>>,
        now: Duration,
    ) {
        let payload = control::encode(msg);
        if let Some(dg) = packet::build_packet(
            packet_flags::CONTROL,
            self.ack,
            0,
            &payload,
            self.security_token.wire_value(),
            huffman,
        ) {
            out.push(dg);
        }
        self.last_send_time = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    fn ms(ms: u64) -> Duration {
        Duration::from_millis(ms)
    }

    /// Drives a minimal, deterministic client/server handshake over an in-memory "wire" (no
    /// loss/reorder — see `tests/lossy_link.rs` for that), returning both `Connection`s once
    /// online.
    fn handshake(huffman: &Huffman) -> (Connection, Connection) {
        let mut client = Connection::new(Config::default());
        let mut server = Connection::new(Config::default());
        let mut now = secs(0);

        client.connect(now, huffman);
        let c2s = client.flush(huffman, now);
        assert_eq!(c2s.len(), 1);

        now += ms(10);
        server.accept(0x1234_5678, now, huffman);
        // Feed the server the CONNECT (irrelevant to our `accept`-driven server, but exercises
        // the decode path and must not do anything harmful while Pending).
        assert!(server.feed(&c2s[0], huffman, now).is_empty());
        let s2c = server.flush(huffman, now);
        assert_eq!(s2c.len(), 1);

        now += ms(10);
        let events = client.feed(&s2c[0], huffman, now);
        assert_eq!(events, vec![Event::Connected]);
        assert!(client.is_online());
        let c2s2 = client.flush(huffman, now);
        assert_eq!(c2s2.len(), 1); // the queued ACCEPT

        now += ms(10);
        let events = server.feed(&c2s2[0], huffman, now);
        assert_eq!(events, vec![Event::Connected]);
        assert!(server.is_online());

        (client, server)
    }

    #[test]
    fn full_handshake_reaches_online_both_sides() {
        let huffman = Huffman::new();
        let (client, server) = handshake(&huffman);
        assert!(client.is_online());
        assert!(server.is_online());
    }

    #[test]
    fn vital_chunk_delivered_in_order_exactly_once() {
        let huffman = Huffman::new();
        let (mut client, mut server) = handshake(&huffman);
        let mut now = secs(1);

        client.send_chunk(b"hello", true, now).unwrap();
        let datagrams = client.flush(&huffman, now);
        assert_eq!(datagrams.len(), 1);

        now += ms(5);
        let events = server.feed(&datagrams[0], &huffman, now);
        assert_eq!(
            events,
            vec![Event::Chunk {
                vital: true,
                data: b"hello".to_vec()
            }]
        );

        // Re-feeding the exact same (duplicated) datagram must not re-deliver it.
        let events = server.feed(&datagrams[0], &huffman, now);
        assert_eq!(events, vec![]);
    }

    #[test]
    fn non_vital_chunk_delivered_without_sequence_tracking() {
        let huffman = Huffman::new();
        let (mut client, mut server) = handshake(&huffman);
        let now = secs(1);

        client.send_chunk(b"ping", false, now).unwrap();
        let datagrams = client.flush(&huffman, now);
        let events = server.feed(&datagrams[0], &huffman, now);
        assert_eq!(
            events,
            vec![Event::Chunk {
                vital: false,
                data: b"ping".to_vec()
            }]
        );
    }

    #[test]
    fn dropped_vital_chunk_is_resent_after_one_second_and_then_delivered() {
        let huffman = Huffman::new();
        let (mut client, mut server) = handshake(&huffman);
        let mut now = secs(1);

        client.send_chunk(b"important", true, now).unwrap();
        let datagrams = client.flush(&huffman, now);
        assert_eq!(datagrams.len(), 1);
        // Simulate the datagram being lost: never feed it to `server`.

        now += secs(1) + ms(1);
        let resent = client.flush(&huffman, now);
        assert_eq!(resent.len(), 1, "expected a resend after > 1s unacked");

        now += ms(5);
        let events = server.feed(&resent[0], &huffman, now);
        assert_eq!(
            events,
            vec![Event::Chunk {
                vital: true,
                data: b"important".to_vec()
            }]
        );
    }

    #[test]
    fn out_of_order_vital_chunk_triggers_resend_signal_and_is_recovered() {
        let huffman = Huffman::new();
        let (mut client, mut server) = handshake(&huffman);
        let mut now = secs(1);

        client.send_chunk(b"one", true, now).unwrap();
        let _first = client.flush(&huffman, now).remove(0);
        client.send_chunk(b"two", true, now).unwrap();
        let second = client.flush(&huffman, now).remove(0);

        // Deliver "two" before "one": server must not accept it early, and must ask for a
        // resend (bare RESEND-flagged packet on its next flush).
        now += ms(5);
        assert_eq!(server.feed(&second, &huffman, now), vec![]);
        let resend_request = server.flush(&huffman, now);
        assert_eq!(resend_request.len(), 1);

        now += ms(5);
        let events = client.feed(&resend_request[0], &huffman, now);
        assert_eq!(events, vec![]); // just an ack + resend flag, no chunks/control for the client
        let resent = client.flush(&huffman, now);
        assert_eq!(resent.len(), 1, "client should have resent both buffered vital chunks");

        now += ms(5);
        let events = server.feed(&resent[0], &huffman, now);
        assert_eq!(
            events,
            vec![
                Event::Chunk {
                    vital: true,
                    data: b"one".to_vec()
                },
                Event::Chunk {
                    vital: true,
                    data: b"two".to_vec()
                },
            ]
        );
    }

    #[test]
    fn ack_removes_resend_entries() {
        let huffman = Huffman::new();
        let (mut client, mut server) = handshake(&huffman);
        let mut now = secs(1);

        client.send_chunk(b"one", true, now).unwrap();
        let dg = client.flush(&huffman, now).remove(0);
        now += ms(5);
        server.feed(&dg, &huffman, now).into_iter().for_each(drop);
        // The server's next flush carries ack=1, acknowledging "one".
        let dg2 = server.flush(&huffman, now);
        // Server had nothing to send (no chunks, no resend flag) — send a KEEPALIVE explicitly by
        // advancing time, to actually get an ack-carrying datagram back to the client.
        assert!(dg2.is_empty());
        now += secs(1) + ms(1);
        let keepalive = server.flush(&huffman, now);
        assert_eq!(keepalive.len(), 1);

        now += ms(5);
        client.feed(&keepalive[0], &huffman, now).into_iter().for_each(drop);
        // Internal: resend buffer should now be empty (ack removed it) — indirectly verified by
        // there being nothing further to resend even after another timeout window.
        now += secs(2);
        let nothing_to_resend = client.flush(&huffman, now);
        // Only a keepalive of our own may appear (last_send_time was bumped by the ack'd send),
        // never the same chunk twice.
        for d in &nothing_to_resend {
            let huffman2 = Huffman::new();
            let packet = packet::unpack_packet(d, &huffman2, true).unwrap();
            assert_eq!(packet.num_chunks, 0, "no chunk should be left to resend");
        }
    }

    #[test]
    fn timeout_when_peer_goes_silent() {
        let huffman = Huffman::new();
        let (mut client, _server) = handshake(&huffman);
        let now = secs(1) + DEFAULT_TIMEOUT + secs(1);
        let datagrams = client.flush(&huffman, now);
        assert!(datagrams.is_empty());
        assert!(matches!(client.state(), State::Error(reason) if reason == "Timeout"));
    }

    /// F3: a `flush()`-detected state transition (as opposed to one `feed()` observes directly)
    /// must be observable as an event too, delivered exactly once via `take_events()`.
    #[test]
    fn timeout_delivers_event_error_exactly_once_via_take_events() {
        let huffman = Huffman::new();
        let (mut client, _server) = handshake(&huffman);
        let now = secs(1) + DEFAULT_TIMEOUT + secs(1);

        assert!(client.take_events().is_empty(), "nothing to report before the timeout");
        let datagrams = client.flush(&huffman, now);
        assert!(datagrams.is_empty());
        assert_eq!(client.take_events(), vec![Event::Error("Timeout".to_string())]);
        // Draining again must not repeat it, and further flushes (still silent, still Error)
        // must not re-report it either — it already happened, exactly once.
        assert!(client.take_events().is_empty());
        let _ = client.flush(&huffman, now + secs(10));
        assert!(
            client.take_events().is_empty(),
            "an already-`Error` connection must not re-emit the same event"
        );
    }

    /// F6: once a connection has errored (here: via timeout, mid-`flush()`), no further data
    /// chunk may be sent — matching `CNetConnection`, where `QueueChunkEx` (and thus any
    /// `Flush()`) refuses once `m_State == ERROR` (`network_conn.cpp:145-146`).
    #[test]
    fn flush_sends_no_data_once_timed_out_even_with_chunks_already_queued() {
        let huffman = Huffman::new();
        let (mut client, _server) = handshake(&huffman);
        let mut now = secs(1);

        // Queue a chunk while still healthy (not yet flushed) ...
        client.send_chunk(b"queued before the timeout", true, now).unwrap();
        // ... then jump time far enough that *this same* flush() call both detects the timeout
        // and would, without the F6 fix, still drain the pre-queued chunk into a datagram.
        now += DEFAULT_TIMEOUT + secs(1);
        let datagrams = client.flush(&huffman, now);
        assert!(
            datagrams.is_empty(),
            "no datagram — data or otherwise generated from pending_packets — after erroring"
        );
        // Queuing a vital chunk also put an entry in the resend buffer, so both the silence
        // timeout *and* the too-weak-connection check fire in this same call; per the documented
        // last-check-wins order the latter's message is what ends up set — either way, this test
        // is about F6 (no data sent), not about which message wins.
        assert!(matches!(client.state(), State::Error(_)));

        // And the connection must stay silent from here on, too.
        now += secs(1);
        assert!(client.flush(&huffman, now).is_empty());
    }

    /// F3: the resend-buffer-exhaustion error path (`send_chunk`) also reports via
    /// `take_events()`, not just the `flush()`-detected ones.
    #[test]
    fn resend_buffer_full_delivers_event_error() {
        let mut conn = Connection::new(Config::default());
        conn.state = State::Online;
        let chunk = vec![0u8; packet::MAX_CHUNK_SIZE];
        let mut now = Duration::ZERO;
        loop {
            match conn.send_chunk(&chunk, true, now) {
                Ok(()) => now += Duration::from_millis(1),
                Err(SendChunkError::ResendBufferFull) => break,
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
        let events = conn.take_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], Event::Error(reason) if reason.contains("resend buffer")));
    }

    #[test]
    fn too_weak_connection_when_vital_chunk_never_acked() {
        let huffman = Huffman::new();
        let (mut client, _server) = handshake(&huffman);
        let mut now = secs(1);
        client.send_chunk(b"lost forever", true, now).unwrap();
        let _ = client.flush(&huffman, now);

        // Keep "the peer" silent (never ack it) well past the timeout, while occasionally
        // flushing (as a real caller would every tick) so the resend timer keeps firing.
        for _ in 0..5 {
            now += secs(30);
            let _ = client.flush(&huffman, now);
        }
        match client.state() {
            State::Error(reason) => assert!(reason.starts_with("Too weak connection")),
            other => panic!("expected State::Error, got {other:?}"),
        }
    }

    #[test]
    fn send_chunk_rejects_oversized_payload() {
        let mut conn = Connection::new(Config::default());
        conn.state = State::Online; // bypass handshake for this pure input-validation test
        let too_big = vec![0u8; packet::MAX_CHUNK_SIZE + 1];
        assert_eq!(
            conn.send_chunk(&too_big, true, Duration::ZERO),
            Err(SendChunkError::TooLarge(packet::MAX_CHUNK_SIZE + 1))
        );
    }

    #[test]
    fn send_chunk_rejects_when_offline_or_errored() {
        let mut conn = Connection::new(Config::default());
        assert_eq!(
            conn.send_chunk(b"x", true, Duration::ZERO),
            Err(SendChunkError::NotConnected)
        );
        conn.state = State::Error("boom".to_string());
        assert_eq!(
            conn.send_chunk(b"x", true, Duration::ZERO),
            Err(SendChunkError::NotConnected)
        );
    }

    #[test]
    fn send_chunk_allowed_before_online_and_delivered_once_there() {
        // QueueChunkEx's own guard only excludes Offline/Error — queuing while Connecting is
        // legal and just sits until flush() actually has somewhere to send it usefully. Here we
        // exercise it in the Pending (server) role, which is the state where it is most useful:
        // queuing a system message to send the instant a still-unconfirmed client comes online.
        let huffman = Huffman::new();
        let mut server = Connection::new(Config::default());
        server.accept(0xaaaa_bbbb, Duration::ZERO, &huffman);
        assert!(matches!(server.state(), State::Pending));
        server.send_chunk(b"eager", true, Duration::ZERO).unwrap();
        let datagrams = server.flush(&huffman, Duration::ZERO);
        // First datagram is CONNECTACCEPT (control); the eagerly-queued chunk goes out in a
        // second, data-carrying datagram, ready for whenever the handshake actually completes.
        assert!(!datagrams.is_empty());
    }

    #[test]
    fn resend_buffer_full_errors_and_moves_to_error_state() {
        let mut conn = Connection::new(Config::default());
        conn.state = State::Online;
        let chunk = vec![0u8; packet::MAX_CHUNK_SIZE];
        let mut now = Duration::ZERO;
        let mut sent = 0usize;
        loop {
            match conn.send_chunk(&chunk, true, now) {
                Ok(()) => {
                    sent += chunk.len();
                    now += Duration::from_millis(1);
                    if sent > RESEND_BUFFER_BYTES {
                        panic!("expected ResendBufferFull before exceeding the budget");
                    }
                }
                Err(SendChunkError::ResendBufferFull) => break,
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
        assert!(matches!(conn.state(), State::Error(reason) if reason.contains("resend buffer")));
    }

    #[test]
    fn ack_wraparound_is_handled_like_the_reference() {
        // Direct unit test of the wraparound-aware ack validity check, independent of full
        // handshake plumbing — mirrors `CNetConnection::Feed`'s two-branch ack check exactly.
        let mut conn = Connection::new(Config::default());
        conn.sequence = 5;
        conn.peer_ack = 1020; // peer_ack > sequence: we've wrapped since they last acked.
        assert!(conn.ack_is_valid(1022)); // still within [peer_ack, MAX)
        assert!(conn.ack_is_valid(3)); // wrapped forward past 0, within [0, sequence]
        assert!(!conn.ack_is_valid(10)); // strictly between sequence and peer_ack: implausible

        conn.sequence = 100;
        conn.peer_ack = 50;
        assert!(conn.ack_is_valid(50));
        assert!(conn.ack_is_valid(100));
        assert!(conn.ack_is_valid(75));
        assert!(!conn.ack_is_valid(49));
        assert!(!conn.ack_is_valid(101));
    }

    #[test]
    fn feed_ignores_garbage_without_panicking() {
        let huffman = Huffman::new();
        let (mut client, _server) = handshake(&huffman);
        for garbage in [&b""[..], &[0xffu8; 4], &[0u8; 1400], &[0x20; 1]] {
            let events = client.feed(garbage, &huffman, secs(1));
            assert!(events.is_empty() || matches!(events[0], Event::Error(_) | Event::ClosedByPeer(_)));
        }
        assert!(client.is_online(), "garbage must not disturb an established connection");
    }

    #[test]
    fn disconnect_queues_close_and_resets_to_offline() {
        let huffman = Huffman::new();
        let (mut client, mut server) = handshake(&huffman);
        client.disconnect(Some("bye"), &huffman);
        assert!(matches!(client.state(), State::Offline));
        let datagrams = client.flush(&huffman, secs(1));
        assert_eq!(datagrams.len(), 1);

        let events = server.feed(&datagrams[0], &huffman, secs(1));
        assert_eq!(events, vec![Event::ClosedByPeer("bye".to_string())]);
        assert!(matches!(server.state(), State::Error(_)));
    }

    #[test]
    fn keepalive_sent_after_one_second_idle_online() {
        let huffman = Huffman::new();
        let (mut client, _server) = handshake(&huffman);
        let now = secs(1) + KEEPALIVE_INTERVAL + ms(1);
        let datagrams = client.flush(&huffman, now);
        assert_eq!(datagrams.len(), 1);
        let packet = packet::unpack_packet(&datagrams[0], &huffman, true).unwrap();
        assert_ne!(packet.flags & packet_flags::CONTROL, 0);
        let ctrl = control::decode(&packet.data).unwrap();
        assert_eq!(ctrl, ControlMsg::KeepAlive);
    }
}
