//! Task 2.3 e2e scenario (e), "redirect": DDNet 20.1's local server has no admin command that
//! triggers `NETMSG_REDIRECT` (`RedirectClient` is only ever called internally, e.g. from
//! `ReconnectClient`'s own fallback — see `server.cpp:544-566`; there is no bare `redirect`
//! console command), so per the task spec this is simulated with a small local UDP test double
//! written in Rust, driven directly against `ddai_client::driver::Client` (never against the
//! real DDNet-Server) — real loopback sockets, no mocking of `ddai-client` itself.

use ddai_net::conn::{self, Connection};
use ddai_net::huffman::Huffman;
use ddai_net::packer::Packer;
use ddai_net::packet::{self, packet_flags};
use ddai_net::uuid::{self, MsgId};
use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;
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

/// A minimal TKEN-handshake-only test double: accepts one client, and as soon as the handshake
/// completes, sends `redirect@ddnet.org` with `redirect_to_port` — then keeps answering (so a
/// resent `CONNECT`/handshake retry, or the client's own `ACCEPT`, doesn't wedge anything) until
/// `deadline`. `saw_close`, if given, is set once this connection observes a `CLOSE` from the
/// client — review finding F4: the driver must gracefully close the *old* connection before
/// opening a new one to the redirect target, not just abandon it.
fn run_redirect_server(
    socket: UdpSocket,
    redirect_to_port: u16,
    deadline: Instant,
    saw_close: Option<Arc<AtomicBool>>,
) {
    socket.set_read_timeout(Some(Duration::from_millis(50))).unwrap();
    let huffman = Huffman::new();
    let mut connection = Connection::new(conn::Config::default());
    let start = Instant::now();
    let mut buf = [0u8; 2048];
    let mut peer: Option<SocketAddr> = None;
    let mut sent_redirect = false;

    while Instant::now() < deadline {
        match socket.recv_from(&mut buf) {
            Ok((n, from)) => {
                peer = Some(from);
                let now = start.elapsed();
                if !connection.is_online() {
                    connection.accept(0x1234_5678, now, &huffman);
                }
                for ev in connection.feed(&buf[..n], &huffman, now) {
                    match ev {
                        conn::Event::Connected if !sent_redirect => {
                            let chunk = ex_sys_chunk("redirect@ddnet.org", |p| p.add_int(i32::from(redirect_to_port)));
                            connection.send_chunk(&chunk, true, now).expect("queue redirect chunk");
                            sent_redirect = true;
                        }
                        conn::Event::ClosedByPeer(_) => {
                            if let Some(flag) = &saw_close {
                                flag.store(true, Ordering::SeqCst);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
            Err(e) => panic!("redirect test double socket error: {e}"),
        }
        let now = start.elapsed();
        for dg in connection.flush(&huffman, now) {
            if let Some(addr) = peer {
                let _ = socket.send_to(&dg, addr);
            }
        }
    }
}

/// A datagram is a control-oriented `CONNECT` (the very first thing any DDNet client ever sends)
/// — checked without decoding the whole handshake, just enough to prove *a fresh connection
/// attempt* reached this socket.
fn looks_like_connect(datagram: &[u8]) -> bool {
    let Ok(packet) = packet::unpack_packet(datagram, &Huffman::new(), true) else {
        return false;
    };
    if packet.flags & packet_flags::CONTROL == 0 {
        return false;
    }
    matches!(
        ddai_net::control::decode(&packet.data),
        Ok(ddai_net::control::ControlMsg::Connect { .. })
    )
}

#[test]
fn client_follows_a_redirect_to_a_new_port_over_real_udp() {
    let source_socket = UdpSocket::bind("127.0.0.1:0").expect("bind source socket");
    let target_socket = UdpSocket::bind("127.0.0.1:0").expect("bind target socket");
    let source_addr = source_socket.local_addr().unwrap();
    let target_port = target_socket.local_addr().unwrap().port();
    target_socket
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("set target read timeout");

    let saw_close = Arc::new(AtomicBool::new(false));
    let deadline = Instant::now() + Duration::from_secs(8);
    let source_handle = {
        let saw_close = Arc::clone(&saw_close);
        thread::spawn(move || run_redirect_server(source_socket, target_port, deadline, Some(saw_close)))
    };

    let mut client = ddai_client::Client::connect(source_addr, ddai_client::ClientConfig::default());

    // Proof #1: the *driver* itself reports following the redirect, to the right port.
    let mut saw_redirect_followed = false;
    let event_deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < event_deadline && !saw_redirect_followed {
        if let Some(ev) = client.recv_event(Duration::from_millis(100))
            && let ddai_client::ClientEvent::RedirectFollowed { to } = ev
        {
            assert_eq!(to.ip(), source_addr.ip());
            assert_eq!(to.port(), target_port);
            saw_redirect_followed = true;
        }
    }
    assert!(
        saw_redirect_followed,
        "expected a ClientEvent::RedirectFollowed to the target port"
    );

    // Proof #2: an actual, independent, real socket at the target port received a real fresh
    // CONNECT datagram — the redirect was followed on the wire, not just reported.
    let mut buf = [0u8; 64];
    let (n, _from) = target_socket
        .recv_from(&mut buf)
        .expect("expected a datagram to arrive at the redirect target port");
    assert!(
        looks_like_connect(&buf[..n]),
        "expected the first datagram at the target port to be CONNECT"
    );

    client.disconnect();
    client.join();
    source_handle.join().expect("redirect test double thread panicked");

    // Proof #3 (review finding F4): the *old* (source) connection was gracefully closed, not
    // just abandoned mid-flight when the driver moved on to the redirect target.
    assert!(
        saw_close.load(Ordering::SeqCst),
        "expected the source connection to see a CLOSE after the redirect was followed"
    );
}

#[test]
fn client_refuses_a_second_redirect_in_the_same_session() {
    // Source -> first target (which itself immediately redirects again) -> a third address the
    // client must NEVER reach, because loop protection allows following at most one redirect per
    // `Client::connect` call.
    let source_socket = UdpSocket::bind("127.0.0.1:0").expect("bind source socket");
    let first_target_socket = UdpSocket::bind("127.0.0.1:0").expect("bind first target socket");
    let third_socket = UdpSocket::bind("127.0.0.1:0").expect("bind third socket");

    let source_addr = source_socket.local_addr().unwrap();
    let first_target_port = first_target_socket.local_addr().unwrap().port();
    let third_port = third_socket.local_addr().unwrap().port();
    third_socket
        .set_read_timeout(Some(Duration::from_millis(500)))
        .expect("set third read timeout");

    let deadline = Instant::now() + Duration::from_secs(8);
    let source_handle = thread::spawn(move || run_redirect_server(source_socket, first_target_port, deadline, None));
    let first_target_handle =
        thread::spawn(move || run_redirect_server(first_target_socket, third_port, deadline, None));

    let mut client = ddai_client::Client::connect(source_addr, ddai_client::ClientConfig::default());

    let mut saw_refused = false;
    let mut saw_gave_up = false;
    let event_deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < event_deadline && !(saw_refused && saw_gave_up) {
        match client.recv_event(Duration::from_millis(100)) {
            Some(ddai_client::ClientEvent::RedirectRefused { .. }) => saw_refused = true,
            Some(ddai_client::ClientEvent::GaveUp { .. }) => saw_gave_up = true,
            _ => {}
        }
    }
    assert!(
        saw_refused,
        "expected the second redirect to be refused (loop protection)"
    );
    assert!(
        saw_gave_up,
        "expected the driver to give up rather than follow a second redirect"
    );

    // The third address must never see anything at all.
    let mut buf = [0u8; 64];
    let result = third_socket.recv_from(&mut buf);
    assert!(
        result.is_err(),
        "the client must never have reached the third address after refusing the second redirect"
    );

    client.join();
    source_handle.join().expect("source test double thread panicked");
    first_target_handle
        .join()
        .expect("first target test double thread panicked");
}
