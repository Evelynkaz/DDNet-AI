//! Task 2.3b: reproduces the incident behind
//! `~/aiddnet/data/logs/record/swarfey-first-20260929.log` — 10 `SessionEvent::Connected` events
//! (5 @ +2.03s, 1 @ +20.00s, 4 @ +22.06s, exactly the pattern of the 5-attempts-per-20s limiter),
//! no `Disconnected`/`ReconnectAttempt`/`RedirectFollowed` in between, no map ever loaded — and
//! pins down the fix.
//!
//! Root cause: the only driver path that starts a new connection without emitting any event or
//! log line is `ConnectionOutcome::Reconnect` (`reconnect@ddnet.org`); it used to `continue`
//! unconditionally, without backoff, so a peer answering every handshake with a reconnect request
//! was followed forever, bounded only by the process-wide rate limiter. Now the driver logs and
//! surfaces every request, follows at most `MAX_SERVER_RECONNECTS_BEFORE_IN_GAME`, reuses one
//! socket (one local port) like the real client, and has a handshake watchdog as a last resort.
//!
//! Real DDNet 18.5 has no `reconnect@ddnet.org` at all (only `redirect@ddnet.org`), so the sender
//! on Swarfey is a modified server or something in front of it; here it is simulated with a small
//! local UDP test double (and, out of tree, with a patched local 18.5 server — docs/SETUP.md section 5d).

use ddai_net::conn::{self, Connection};
use ddai_net::huffman::Huffman;
use ddai_net::packer::Packer;
use ddai_net::uuid::{self, MsgId};
use std::collections::HashMap;
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

fn ex_sys_chunk(name: &str, body: impl FnOnce(&mut Packer)) -> Vec<u8> {
    let mut buf = [0u8; 256];
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
    packer.data().to_vec()
}

/// One demultiplexed peer's connection state, server role.
struct PeerState {
    connection: Connection,
    sent_reconnect: bool,
    /// `Mode::SlowMap`: `MAP_CHANGE` queued, and how many `MAP_DATA` chunks were pushed so far.
    map_chunks_sent: u32,
    next_chunk_at: Option<Instant>,
}

const SLOW_MAP_CHUNKS: u32 = 8;
const SLOW_MAP_CHUNK_INTERVAL: Duration = Duration::from_millis(150);
const SLOW_MAP_CRC: i32 = 0x1234_5678;

fn numbered_sys_chunk(msg: &ddai_net::sysmsg::SysMsg, id: i32) -> Vec<u8> {
    let mut buf = [0u8; 2048];
    let mut packer = Packer::new(&mut buf);
    uuid::pack_msg_id(&mut packer, MsgId::Numbered(id), true);
    ddai_net::sysmsg::encode(msg, &mut packer);
    packer.data().to_vec()
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Answer every completed handshake with `reconnect@ddnet.org`.
    ReconnectFlood,
    /// Complete the handshake and then never send anything else (a join that never progresses).
    Silent,
    /// Reject every `CONNECT` with a peer `CLOSE("This server is full")` (a retry-worthy reason).
    Full,
    /// Complete the handshake, announce a map, then push one `MAP_DATA` chunk every 150 ms —
    /// a slow but steadily progressing download (the last chunk is garbage, so the join ends with a
    /// `ProtocolViolation` when it completes, which is distinguishable from a watchdog give-up).
    SlowMap,
}

#[derive(Default)]
struct Observed {
    /// Source address of every completed handshake.
    connected_from: Vec<(Instant, SocketAddr)>,
    /// Number of peer-initiated `CLOSE`s (graceful disconnects) received.
    closes: u32,
    /// Number of datagrams the double treated as the start of a connection attempt.
    attempts: u32,
}

fn run_server(socket: UdpSocket, mode: Mode, stop: Arc<AtomicBool>, observed: Arc<Mutex<Observed>>) {
    socket.set_read_timeout(Some(Duration::from_millis(20))).unwrap();
    let huffman = Huffman::new();
    let start = Instant::now();
    let mut buf = [0u8; 2048];
    let mut peers: HashMap<SocketAddr, PeerState> = HashMap::new();
    let mut next_token: u32 = 0xf100_0d00;

    while !stop.load(Ordering::SeqCst) {
        match socket.recv_from(&mut buf) {
            Ok((n, from)) => {
                let now = start.elapsed();
                if mode == Mode::Full {
                    observed.lock().unwrap().attempts += 1;
                    let mut reject = Connection::new(conn::Config::default());
                    reject.accept(next_token, now, &huffman);
                    next_token = next_token.wrapping_add(1);
                    reject.disconnect(Some("This server is full"), &huffman);
                    // Only the CLOSE goes out, like a real full server (`network_server.cpp:282`),
                    // never a CONNECTACCEPT first: `accept()` queued that one, so skip it.
                    if let Some(close) = reject.flush(&huffman, now).pop() {
                        let _ = socket.send_to(&close, from);
                    }
                    continue;
                }
                // `accept()` resets the whole connection, so it must run exactly once per peer,
                // when its first datagram (the CONNECT) shows up.
                let peer = peers.entry(from).or_insert_with(|| {
                    let mut connection = Connection::new(conn::Config::default());
                    connection.accept(next_token, now, &huffman);
                    next_token = next_token.wrapping_add(1);
                    PeerState {
                        connection,
                        sent_reconnect: false,
                        map_chunks_sent: 0,
                        next_chunk_at: None,
                    }
                });
                let mut closed = false;
                for ev in peer.connection.feed(&buf[..n], &huffman, now) {
                    match ev {
                        conn::Event::Connected => {
                            observed.lock().unwrap().connected_from.push((Instant::now(), from));
                            if mode == Mode::SlowMap {
                                let change = ddai_net::sysmsg::SysMsg::MapChange {
                                    name: "slowmap".to_string(),
                                    crc: SLOW_MAP_CRC,
                                    size: (SLOW_MAP_CHUNKS * 512) as i32,
                                };
                                peer.connection
                                    .send_chunk(
                                        &numbered_sys_chunk(&change, ddai_net::sysmsg::id::MAP_CHANGE),
                                        true,
                                        now,
                                    )
                                    .expect("queue map change");
                                peer.next_chunk_at = Some(Instant::now() + SLOW_MAP_CHUNK_INTERVAL);
                            }
                            if mode == Mode::ReconnectFlood && !peer.sent_reconnect {
                                let chunk = ex_sys_chunk("reconnect@ddnet.org", |_| {});
                                peer.connection
                                    .send_chunk(&chunk, true, now)
                                    .expect("queue reconnect chunk");
                                peer.sent_reconnect = true;
                            }
                        }
                        conn::Event::ClosedByPeer(_) => {
                            observed.lock().unwrap().closes += 1;
                            closed = true;
                        }
                        _ => {}
                    }
                }
                if closed {
                    // Same source address may legitimately CONNECT again (the client reuses its
                    // socket, like the real one) — forget the old connection so it starts fresh.
                    peers.remove(&from);
                }
            }
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
            Err(e) => panic!("test double socket error: {e}"),
        }
        let now = start.elapsed();
        for peer in peers.values_mut() {
            if let Some(at) = peer.next_chunk_at
                && Instant::now() >= at
                && peer.map_chunks_sent < SLOW_MAP_CHUNKS
            {
                let last = peer.map_chunks_sent + 1 == SLOW_MAP_CHUNKS;
                let data = ddai_net::sysmsg::SysMsg::MapData {
                    last: i32::from(last),
                    crc: SLOW_MAP_CRC,
                    chunk: peer.map_chunks_sent as i32,
                    data: vec![0xab; 512],
                };
                peer.connection
                    .send_chunk(&numbered_sys_chunk(&data, ddai_net::sysmsg::id::MAP_DATA), true, now)
                    .expect("queue map data");
                peer.map_chunks_sent += 1;
                peer.next_chunk_at = Some(Instant::now() + SLOW_MAP_CHUNK_INTERVAL);
            }
        }
        for (addr, peer) in &mut peers {
            for dg in peer.connection.flush(&huffman, now) {
                let _ = socket.send_to(&dg, *addr);
            }
        }
    }
}

struct Outcome {
    gave_up: Option<ddai_client::GaveUpCategory>,
    saw_reconnect_attempt: bool,
    saw_bare_disconnected: bool,
    server_requested_reconnects: u32,
    elapsed: Duration,
    observed: Observed,
}

fn run_scenario(mode: Mode, handshake_timeout: Duration) -> Outcome {
    run_scenario_with(
        mode,
        handshake_timeout,
        ddai_client::ClientConfig::default().handshake_hard_cap,
    )
}

fn run_scenario_with(mode: Mode, handshake_timeout: Duration, handshake_hard_cap: Duration) -> Outcome {
    let socket = UdpSocket::bind("127.0.0.1:0").expect("bind test double socket");
    let addr = socket.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let observed = Arc::new(Mutex::new(Observed::default()));
    let server = {
        let stop = Arc::clone(&stop);
        let observed = Arc::clone(&observed);
        thread::spawn(move || run_server(socket, mode, stop, observed))
    };

    let config = ddai_client::ClientConfig {
        handshake_timeout,
        handshake_hard_cap,
        ..ddai_client::ClientConfig::default()
    };
    let start = Instant::now();
    let mut client = ddai_client::Client::connect(addr, config);
    let mut out = Outcome {
        gave_up: None,
        saw_reconnect_attempt: false,
        saw_bare_disconnected: false,
        server_requested_reconnects: 0,
        elapsed: Duration::ZERO,
        observed: Observed::default(),
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline && out.gave_up.is_none() {
        match client.recv_event(Duration::from_millis(50)) {
            Some(ddai_client::ClientEvent::GaveUp { category, .. }) => out.gave_up = Some(category),
            Some(ddai_client::ClientEvent::ReconnectAttempt { .. }) => out.saw_reconnect_attempt = true,
            Some(ddai_client::ClientEvent::ServerRequestedReconnect { .. }) => out.server_requested_reconnects += 1,
            Some(ddai_client::ClientEvent::Session(ev)) => {
                if matches!(*ev, ddai_client::SessionEvent::Disconnected { .. }) {
                    out.saw_bare_disconnected = true;
                }
            }
            _ => {}
        }
    }
    out.elapsed = start.elapsed();
    client.join();
    // Give the double a moment to receive the client's final CLOSE before stopping it.
    thread::sleep(Duration::from_millis(100));
    stop.store(true, Ordering::SeqCst);
    server.join().expect("test double thread panicked");
    let obs = observed.lock().unwrap();
    out.observed = Observed {
        connected_from: obs.connected_from.clone(),
        closes: obs.closes,
        attempts: obs.attempts,
    };
    out
}

/// The incident pattern: a `reconnect@ddnet.org` after every handshake. Before the fix the driver
/// followed these forever (bounded only by the 5/20s limiter), silently. Now: exactly one is
/// followed, the second stops the session with `GaveUpCategory::ReconnectLoop`, the old connection
/// is closed gracefully each time, and both handshakes come from the same local port.
#[test]
fn server_requested_reconnect_flood_stops_after_one_followed_reconnect() {
    // Watchdog long enough that the reconnect budget, not the watchdog, must be what stops it.
    let out = run_scenario(Mode::ReconnectFlood, Duration::from_secs(8));

    assert_eq!(
        out.gave_up,
        Some(ddai_client::GaveUpCategory::ReconnectLoop),
        "a reconnect loop must end with GaveUp(ReconnectLoop)"
    );
    assert_eq!(
        out.server_requested_reconnects, 1,
        "exactly one server-requested reconnect is followed (and reported)"
    );
    assert!(
        !out.saw_reconnect_attempt && !out.saw_bare_disconnected,
        "a reconnect flood is not a lost connection"
    );
    assert!(out.elapsed < Duration::from_secs(4), "took {:?}", out.elapsed);

    let from = &out.observed.connected_from;
    assert_eq!(
        from.len(),
        2,
        "server must have seen exactly 2 handshakes, saw {}",
        from.len()
    );
    assert_eq!(
        from[0].1, from[1].1,
        "the reconnect must reuse the same local UDP port, like the real client"
    );
    // The one allowed reconnect goes out promptly, like the real client's (no backoff): a gate in
    // front of the server may expect a quick return.
    let gap = from[1].0.duration_since(from[0].0);
    assert!(gap < Duration::from_millis(300), "the reconnect was delayed by {gap:?}");
    assert!(
        out.observed.closes >= 2,
        "each connection must be closed gracefully (CLOSE), saw {}",
        out.observed.closes
    );
}

/// The last-resort watchdog: a peer that completes the handshake and then never lets the join
/// progress. The client must CLOSE and give up by itself, loudly, and never retry.
#[test]
fn handshake_watchdog_stops_a_join_that_never_progresses() {
    let out = run_scenario(Mode::Silent, Duration::from_millis(400));

    assert_eq!(out.gave_up, Some(ddai_client::GaveUpCategory::HandshakeTimeout));
    assert!(!out.saw_reconnect_attempt, "the watchdog gives up, it does not retry");
    assert_eq!(
        out.observed.connected_from.len(),
        1,
        "exactly one connection, no retries"
    );
    assert!(
        out.observed.closes >= 1,
        "the client must CLOSE gracefully before giving up"
    );
    assert!(
        out.elapsed >= Duration::from_millis(400) && out.elapsed < Duration::from_secs(3),
        "took {:?}",
        out.elapsed
    );
}

/// A peer that never answers the `CONNECT` at all (the first ~2s of the Swarfey incident looked
/// like this): the watchdog must still fire during the handshake, the client must keep resending
/// `CONNECT` meanwhile (every 500ms, `network_conn.cpp:567`), close gracefully and never retry.
#[test]
fn handshake_watchdog_fires_while_the_peer_never_answers_connect() {
    let socket = UdpSocket::bind("127.0.0.1:0").expect("bind mute socket");
    let addr = socket.local_addr().unwrap();
    socket.set_read_timeout(Some(Duration::from_millis(50))).unwrap();

    let config = ddai_client::ClientConfig {
        handshake_timeout: Duration::from_millis(1800),
        ..ddai_client::ClientConfig::default()
    };
    let start = Instant::now();
    let mut client = ddai_client::Client::connect(addr, config);

    let mut datagrams = 0u32;
    let mut sources = std::collections::HashSet::new();
    let mut gave_up = None;
    let mut buf = [0u8; 2048];
    let deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < deadline && gave_up.is_none() {
        if let Ok((_, from)) = socket.recv_from(&mut buf) {
            datagrams += 1;
            sources.insert(from);
        }
        while let Some(ev) = client.recv_event(Duration::ZERO) {
            if let ddai_client::ClientEvent::GaveUp { category, .. } = ev {
                gave_up = Some(category);
            }
        }
    }
    let elapsed = start.elapsed();
    client.join();
    // The final CLOSE is flushed just before the driver thread ends.
    while let Ok((_, from)) = socket.recv_from(&mut buf) {
        datagrams += 1;
        sources.insert(from);
    }

    assert_eq!(gave_up, Some(ddai_client::GaveUpCategory::HandshakeTimeout));
    assert!(
        elapsed >= Duration::from_millis(1800) && elapsed < Duration::from_secs(4),
        "took {elapsed:?}"
    );
    // CONNECT at 0, 500, 1000, 1500 ms, plus the closing CLOSE.
    assert!(datagrams >= 4, "expected CONNECT resends, saw {datagrams} datagram(s)");
    assert_eq!(
        sources.len(),
        1,
        "one local port for the whole session, saw {sources:?}"
    );
}

/// Review F1: a server that rejects every `CONNECT` with a retry-worthy `CLOSE` ("This server is
/// full") must see at most 2 connection attempts before the client gives up on its own.
#[test]
fn a_full_server_gets_at_most_two_connection_attempts() {
    let out = run_scenario(Mode::Full, Duration::from_secs(15));

    assert_eq!(out.gave_up, Some(ddai_client::GaveUpCategory::TooManyAttempts));
    assert_eq!(
        out.observed.attempts, 2,
        "exactly 2 attempts (the limit), saw {}",
        out.observed.attempts
    );
    assert!(
        out.saw_reconnect_attempt,
        "the one retry must be announced (ReconnectAttempt)"
    );
    // Attempt 2 waits the 1s minimum backoff, then the client stops; far from the 15s watchdog.
    assert!(out.elapsed < Duration::from_secs(6), "took {:?}", out.elapsed);
}

/// Review F3: a slow but steadily progressing map download (8 chunks, 150ms apart = 1.2s) must not
/// be killed by a 400ms watchdog; it runs to completion (the garbage map then fails verification,
/// a `ProtocolViolation`, not a watchdog give-up).
#[test]
fn map_download_progress_extends_the_watchdog() {
    let out = run_scenario(Mode::SlowMap, Duration::from_millis(400));

    assert_eq!(out.gave_up, Some(ddai_client::GaveUpCategory::ProtocolViolation));
    assert!(out.elapsed >= Duration::from_millis(1000), "took {:?}", out.elapsed);
}

/// The extension is bounded: with a 700ms hard cap the same download is stopped by the watchdog
/// even though chunks are still arriving.
#[test]
fn the_watchdog_hard_cap_bounds_the_download_extension() {
    let out = run_scenario_with(Mode::SlowMap, Duration::from_millis(400), Duration::from_millis(700));

    assert_eq!(out.gave_up, Some(ddai_client::GaveUpCategory::HandshakeTimeout));
    assert!(
        out.elapsed >= Duration::from_millis(700) && out.elapsed < Duration::from_millis(1500),
        "took {:?}",
        out.elapsed
    );
}
