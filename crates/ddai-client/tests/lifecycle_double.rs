//! Real-UDP test doubles for review findings F4/F5/F7 — driven directly against
//! `ddai_client::driver::Client`, never against the real DDNet-Server, following the same "small
//! local UDP test double written in Rust" pattern `tests/redirect_double.rs` already established
//! for the redirect scenario.

use ddai_net::conn::{self, Connection};
use ddai_net::huffman::Huffman;
use ddai_net::packer::Packer;
use ddai_net::uuid::{self, MsgId};
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::thread;
use std::time::{Duration, Instant};

/// Completes the TKEN handshake (server role) against a real client, then simply returns — the
/// socket (and, once this function returns, any listener at all) goes away, leaving the client
/// with no more responses ever again. Used to force a *silence timeout* deterministically (rather
/// than a dead-port-from-the-start, which the real client — and this port, `conn.rs:643-644` — never
/// times out on while still in the initial `Connecting` state; only a connection that reached
/// `Online` first and *then* goes silent applies the ordinary silence timeout).
fn run_handshake_then_go_silent(socket: UdpSocket, deadline: Instant) {
    socket
        .set_read_timeout(Some(Duration::from_millis(50)))
        .expect("set read timeout");
    let huffman = Huffman::new();
    let mut connection = Connection::new(conn::Config::default());
    let start = Instant::now();
    let mut buf = [0u8; 2048];
    let mut peer: Option<SocketAddr> = None;

    while !connection.is_online() && Instant::now() < deadline {
        match socket.recv_from(&mut buf) {
            Ok((n, from)) => {
                peer = Some(from);
                let now = start.elapsed();
                if !connection.is_online() {
                    connection.accept(0xaaaa_bbbb, now, &huffman);
                }
                let _ = connection.feed(&buf[..n], &huffman, now);
            }
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
            Err(e) => panic!("test double socket error: {e}"),
        }
        let now = start.elapsed();
        for dg in connection.flush(&huffman, now) {
            if let Some(a) = peer {
                let _ = socket.send_to(&dg, a);
            }
        }
    }
    // Deliberately silent from here on: `socket` drops when this function returns, and nothing
    // further is ever sent back to the client.
}

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

/// Review finding F4: after the driver follows a server-requested reconnect (`reconnect@ddnet.org`
/// — the same mechanism, and the same fix, applies to a followed redirect; `redirect_double.rs`
/// covers that case separately), the *old* connection must be gracefully closed (`CLOSE` reaches
/// this test double) before a *new* connection (a fresh `CONNECT`) ever arrives. Since task 2.3b the driver reuses one
/// socket for every reconnect (like the real client), so the new `CONNECT` comes from the *same*
/// source address as the old connection; this double tracks exactly one current `Connection` at a
/// time and fails loudly the instant a datagram from a *different* address arrives while the
/// current connection has not yet seen a `CLOSE` — the real, wire-level version of "no second connection starts before the first
/// closes" (task/CLAUDE.md's "one bot per server" policy).
#[test]
fn client_closes_the_old_connection_before_reconnecting_over_real_udp() {
    let socket = UdpSocket::bind("127.0.0.1:0").expect("bind test double socket");
    socket
        .set_read_timeout(Some(Duration::from_millis(50)))
        .expect("set read timeout");
    let addr = socket.local_addr().unwrap();

    let huffman = Huffman::new();
    let start = Instant::now();
    let mut buf = [0u8; 2048];

    let mut current_peer: Option<SocketAddr> = None;
    let mut current_connection = Connection::new(conn::Config::default());
    let mut current_closed = true; // no peer yet == "closed"
    let mut sent_reconnect_for_current = false;
    let mut violation: Option<String> = None;
    let mut reconnects_sent = 0u32;

    let mut client = ddai_client::Client::connect(addr, ddai_client::ClientConfig::default());

    let deadline = Instant::now() + Duration::from_secs(10);
    // Stop once we've proven the property across (at least) one full reconnect cycle: the first
    // peer closed, and a second, distinct peer showed up cleanly afterwards — or once a violation
    // was recorded (fail fast).
    let mut second_peer_seen_cleanly = false;
    while Instant::now() < deadline && violation.is_none() && !second_peer_seen_cleanly {
        match socket.recv_from(&mut buf) {
            Ok((n, from)) => {
                let now = start.elapsed();
                // A datagram after the current connection was closed is a new connection, from the
                // same address (the driver reuses its socket) or, for other drivers, another one.
                if current_peer.is_some_and(|p| p != from || current_closed) {
                    if !current_closed {
                        violation = Some(format!(
                            "datagram from a new peer {from} arrived while the previous peer {:?} was not yet closed",
                            current_peer
                        ));
                    } else {
                        // Clean handover: start tracking the new peer from scratch.
                        current_peer = Some(from);
                        current_connection = Connection::new(conn::Config::default());
                        current_closed = false;
                        sent_reconnect_for_current = false;
                        second_peer_seen_cleanly = true;
                    }
                } else if current_peer.is_none() {
                    current_peer = Some(from);
                    current_closed = false;
                }

                if !current_connection.is_online() {
                    current_connection.accept(0xd00d_0001u32.wrapping_add(reconnects_sent), now, &huffman);
                }
                for ev in current_connection.feed(&buf[..n], &huffman, now) {
                    match ev {
                        conn::Event::Connected if !sent_reconnect_for_current => {
                            let chunk = ex_sys_chunk("reconnect@ddnet.org", |_| {});
                            current_connection
                                .send_chunk(&chunk, true, now)
                                .expect("queue reconnect chunk");
                            sent_reconnect_for_current = true;
                            reconnects_sent += 1;
                        }
                        conn::Event::ClosedByPeer(_) => {
                            current_closed = true;
                        }
                        _ => {}
                    }
                }
            }
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
            Err(e) => panic!("test double socket error: {e}"),
        }
        let now = start.elapsed();
        for dg in current_connection.flush(&huffman, now) {
            if let Some(peer_addr) = current_peer {
                let _ = socket.send_to(&dg, peer_addr);
            }
        }
    }

    client.disconnect();
    client.join();

    assert!(violation.is_none(), "{}", violation.unwrap_or_default());
    assert!(
        second_peer_seen_cleanly,
        "expected a second, distinct connection attempt to arrive cleanly after the first closed"
    );
}

/// Review finding F5: dropping a [`ddai_client::Client`] without ever calling `disconnect()`/
/// `join()` must still gracefully close the connection (not just abandon the socket/thread) —
/// [`ddai_client::Client`]'s `Drop` impl sends the same `Disconnect` signal `disconnect()` does.
#[test]
fn dropping_the_client_without_disconnect_still_sends_a_close() {
    close_after_drop(ddai_client::ClientConfig::default());
}

/// Task 3.11: the same with the precise (non-blocking) driver loop: it must stop and say goodbye like the blocking one, and the
/// socket must be back in blocking mode for the final `CLOSE` (the goodbye is a plain send; a reconnect would `recv` blocking).
#[test]
fn dropping_a_precise_client_without_disconnect_still_sends_a_close() {
    close_after_drop(ddai_client::ClientConfig {
        precise_wakeups: true,
        ..ddai_client::ClientConfig::default()
    });
}

fn close_after_drop(config: ddai_client::ClientConfig) {
    let socket = UdpSocket::bind("127.0.0.1:0").expect("bind test double socket");
    socket
        .set_read_timeout(Some(Duration::from_millis(50)))
        .expect("set read timeout");
    let addr = socket.local_addr().unwrap();

    let huffman = Huffman::new();
    let mut connection = Connection::new(conn::Config::default());
    let start = Instant::now();
    let mut buf = [0u8; 2048];
    let mut peer: Option<SocketAddr> = None;

    {
        let client = ddai_client::Client::connect(addr, config);

        let online_deadline = Instant::now() + Duration::from_secs(5);
        while !connection.is_online() && Instant::now() < online_deadline {
            match socket.recv_from(&mut buf) {
                Ok((n, from)) => {
                    peer = Some(from);
                    let now = start.elapsed();
                    if !connection.is_online() {
                        connection.accept(0xfeed_0001, now, &huffman);
                    }
                    let _ = connection.feed(&buf[..n], &huffman, now);
                }
                Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
                Err(e) => panic!("test double socket error: {e}"),
            }
            let now = start.elapsed();
            for dg in connection.flush(&huffman, now) {
                if let Some(a) = peer {
                    let _ = socket.send_to(&dg, a);
                }
            }
        }
        assert!(connection.is_online(), "test setup: handshake must complete");

        // The actual thing under test: drop, with no `disconnect()`/`join()` call at all.
        drop(client);
    }

    let mut saw_close = false;
    let close_deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < close_deadline && !saw_close {
        match socket.recv_from(&mut buf) {
            Ok((n, _from)) => {
                let now = start.elapsed();
                for ev in connection.feed(&buf[..n], &huffman, now) {
                    if matches!(ev, conn::Event::ClosedByPeer(_)) {
                        saw_close = true;
                    }
                }
            }
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
            Err(e) => panic!("test double socket error: {e}"),
        }
    }
    assert!(
        saw_close,
        "expected a CLOSE after the Client was dropped, without ever calling disconnect()"
    );
}

/// Review finding F7: `Client::disconnect()` must interrupt a backoff sleep after a lost
/// connection promptly, not wait out the full (>= 1s) backoff first. Forces a genuine silence
/// timeout (see [`run_handshake_then_go_silent`]) with a short `timeout` so the first
/// `LostConnection`/backoff-sleep is reached quickly.
#[test]
fn disconnect_interrupts_a_backoff_sleep_after_a_lost_connection() {
    let socket = UdpSocket::bind("127.0.0.1:0").expect("bind test double socket");
    let addr = socket.local_addr().unwrap();
    let server_deadline = Instant::now() + Duration::from_secs(5);
    let server_handle = thread::spawn(move || run_handshake_then_go_silent(socket, server_deadline));

    let config = ddai_client::ClientConfig {
        timeout: Duration::from_millis(100),
        ..ddai_client::ClientConfig::default()
    };
    let client = ddai_client::Client::connect(addr, config);

    let mut saw_reconnect_attempt = false;
    let attempt_deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < attempt_deadline && !saw_reconnect_attempt {
        if let Some(ddai_client::ClientEvent::ReconnectAttempt { .. }) = client.recv_event(Duration::from_millis(50)) {
            saw_reconnect_attempt = true;
        }
    }
    assert!(
        saw_reconnect_attempt,
        "expected a ReconnectAttempt once the connection timed out"
    );

    let disconnect_start = Instant::now();
    client.disconnect();

    let mut saw_margin_summary = false;
    let stop_deadline = Instant::now() + Duration::from_millis(800);
    while Instant::now() < stop_deadline && !saw_margin_summary {
        if let Some(ddai_client::ClientEvent::MarginSummary(_)) = client.recv_event(Duration::from_millis(50)) {
            saw_margin_summary = true;
        }
    }
    let elapsed = disconnect_start.elapsed();
    assert!(
        saw_margin_summary,
        "expected the driver to stop (MarginSummary) promptly after disconnect()"
    );
    assert!(
        elapsed < Duration::from_millis(800),
        "disconnect() during a backoff sleep took {elapsed:?} — MIN_BACKOFF is 1s, so this proves \
         the sleep was actually interrupted rather than waited out"
    );
    server_handle.join().expect("test double thread panicked");
}

/// Task 4.4: with `margin_report_every` set the driver sends a `MarginSummary` on that period while a connection runs,
/// not only when it ends (the soak journal samples the input timing of a connection that lasts an hour); without it
/// the summary comes only at the end.
#[test]
fn margin_summaries_arrive_periodically_only_when_asked_for() {
    let count_summaries = |every: Option<Duration>| {
        let socket = UdpSocket::bind("127.0.0.1:0").expect("bind test double socket");
        let addr = socket.local_addr().unwrap();
        let server_deadline = Instant::now() + Duration::from_secs(5);
        let server_handle = thread::spawn(move || run_handshake_then_go_silent(socket, server_deadline));
        let config = ddai_client::ClientConfig {
            // Long enough that the silent double never ends the connection during the observation window.
            timeout: Duration::from_secs(30),
            margin_report_every: every,
            ..ddai_client::ClientConfig::default()
        };
        let client = ddai_client::Client::connect(addr, config);
        let mut seen = 0;
        let end = Instant::now() + Duration::from_millis(1200);
        while Instant::now() < end {
            if let Some(ddai_client::ClientEvent::MarginSummary(_)) = client.recv_event(Duration::from_millis(20)) {
                seen += 1;
            }
        }
        client.disconnect();
        server_handle.join().expect("test double thread panicked");
        seen
    };
    assert_eq!(count_summaries(None), 0, "no periodic summary unless asked for");
    let periodic = count_summaries(Some(Duration::from_millis(100)));
    assert!(
        (5..=13).contains(&periodic),
        "one summary per 100 ms over about 1.2 s, got {periodic}"
    );
}
