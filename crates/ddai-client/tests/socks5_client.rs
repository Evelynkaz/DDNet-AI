//! Task 2.6: the real driver (`Client::connect`) through the in-test SOCKS5 server with its UDP relay, against a
//! small UDP test double of a game server. Everything is on 127.0.0.1.
//!
//! What this pins down end to end (the transport's own behaviour is unit-tested in `socks5.rs`):
//! the session really goes through the relay; a server-requested reconnect keeps the association (D-050: same
//! source port); a lost control connection is a lost connection with a typed reason and is followed by exactly one
//! new association per attempt; reply 0x07 and a wrong password stop everything after one TCP connection; and
//! the proxy/entry pairing is enforced in both directions before any connection is made.

use ddai_client::live_servers::{LiveServerEntry, LiveServers};
use ddai_client::proxy::{ProxyConfig, RelayMode};
use ddai_client::socks5_testserver::{Auth, Bnd, Config, Stall, TestSocks5Server};
use ddai_client::{Client, ClientConfig, ClientEvent, GaveUpCategory, SessionEvent};
use ddai_net::conn::{self, Connection};
use ddai_net::huffman::Huffman;
use ddai_net::packer::Packer;
use ddai_net::uuid::{self, MsgId};
use std::collections::HashMap;
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const NICK: &str = "Muha";
const PROXY: &str = "t";

fn ex_sys_chunk(name: &str) -> Vec<u8> {
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
    packer.data().to_vec()
}

#[derive(Default)]
struct Seen {
    /// Source of every datagram that reached the game double, in order.
    senders: Vec<SocketAddr>,
    /// Source of every completed handshake.
    connected_from: Vec<SocketAddr>,
    closes: u32,
}

/// A game-server double: completes the TKEN handshake with whoever talks to it, optionally answers the first
/// handshake of a peer with `reconnect@ddnet.org`, then stays online and silent.
struct GameDouble {
    addr: SocketAddr,
    seen: Arc<Mutex<Seen>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl GameDouble {
    fn start(reconnect_once: bool) -> Self {
        Self::start_on("127.0.0.1", reconnect_once)
    }

    fn start_on(ip: &str, reconnect_once: bool) -> Self {
        let socket = UdpSocket::bind((ip, 0)).unwrap();
        socket.set_read_timeout(Some(Duration::from_millis(20))).unwrap();
        let addr = socket.local_addr().unwrap();
        let seen = Arc::new(Mutex::new(Seen::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let (seen2, stop2) = (Arc::clone(&seen), Arc::clone(&stop));
        let thread = thread::spawn(move || {
            let huffman = Huffman::new();
            let start = Instant::now();
            let mut buf = [0u8; 2048];
            let mut peers: HashMap<SocketAddr, Connection> = HashMap::new();
            let mut reconnect_sent = false;
            let mut token = 0xf100_0d00u32;
            while !stop2.load(Ordering::SeqCst) {
                match socket.recv_from(&mut buf) {
                    Ok((n, from)) => {
                        let now = start.elapsed();
                        seen2.lock().unwrap().senders.push(from);
                        let connection = peers.entry(from).or_insert_with(|| {
                            let mut c = Connection::new(conn::Config::default());
                            c.accept(token, now, &huffman);
                            token = token.wrapping_add(1);
                            c
                        });
                        let mut closed = false;
                        for ev in connection.feed(&buf[..n], &huffman, now) {
                            match ev {
                                conn::Event::Connected => {
                                    seen2.lock().unwrap().connected_from.push(from);
                                    if reconnect_once && !reconnect_sent {
                                        reconnect_sent = true;
                                        connection
                                            .send_chunk(&ex_sys_chunk("reconnect@ddnet.org"), true, now)
                                            .expect("queue reconnect");
                                    }
                                }
                                conn::Event::ClosedByPeer(_) => {
                                    seen2.lock().unwrap().closes += 1;
                                    closed = true;
                                }
                                _ => {}
                            }
                        }
                        if closed {
                            peers.remove(&from);
                        }
                    }
                    Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
                    Err(e) => panic!("game double socket error: {e}"),
                }
                let now = start.elapsed();
                for (addr, c) in &mut peers {
                    for dg in c.flush(&huffman, now) {
                        let _ = socket.send_to(&dg, *addr);
                    }
                }
            }
        });
        GameDouble {
            addr,
            seen,
            stop,
            thread: Some(thread),
        }
    }
}

impl Drop for GameDouble {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn proxy_cfg(server: &TestSocks5Server, auth: Option<(&str, &str)>) -> ProxyConfig {
    let a = server.addr();
    ProxyConfig::new(
        PROXY,
        a.ip().to_string(),
        a.port(),
        auth.map(|(u, p)| (u.to_string(), p.to_string())),
    )
    .unwrap()
}

fn entry(game: SocketAddr, proxy: Option<&str>) -> LiveServers {
    LiveServers {
        servers: vec![LiveServerEntry {
            address: game.to_string(),
            nick: NICK.to_string(),
            purpose: "test".to_string(),
            ready: true,
            proxy: proxy.map(str::to_string),
        }],
    }
}

fn config(list: LiveServers, proxy: Option<ProxyConfig>) -> ClientConfig {
    ClientConfig {
        name: NICK.to_string(),
        live_servers: list,
        proxy,
        ..ClientConfig::default()
    }
}

/// Collects events until `done` says so or `limit` passes.
fn collect(client: &mut Client, limit: Duration, mut done: impl FnMut(&[ClientEvent]) -> bool) -> Vec<ClientEvent> {
    let mut events = Vec::new();
    let end = Instant::now() + limit;
    while Instant::now() < end && !done(&events) {
        if let Some(ev) = client.recv_event(Duration::from_millis(25)) {
            events.push(ev);
        }
    }
    events
}

fn wait_until(limit: Duration, mut cond: impl FnMut() -> bool) {
    let end = Instant::now() + limit;
    while Instant::now() < end && !cond() {
        thread::sleep(Duration::from_millis(10));
    }
}

fn connected_count(events: &[ClientEvent]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, ClientEvent::Session(s) if matches!(**s, SessionEvent::Connected)))
        .count()
}

fn gave_up(events: &[ClientEvent]) -> Option<(&str, GaveUpCategory)> {
    events.iter().find_map(|e| match e {
        ClientEvent::GaveUp { reason, category } => Some((reason.as_str(), *category)),
        _ => None,
    })
}

#[test]
fn the_session_goes_through_the_relay_and_never_directly() {
    let proxy = TestSocks5Server::start(Config {
        auth: Auth::UserPass("user".into(), "pw".into()),
        ..Default::default()
    });
    let game = GameDouble::start(false);
    let cfg = config(
        entry(game.addr, Some(PROXY)),
        Some(proxy_cfg(&proxy, Some(("user", "pw"))).with_for_server(game.addr.to_string())),
    );
    let mut client = Client::connect(game.addr, cfg);
    let events = collect(&mut client, Duration::from_secs(5), |e| connected_count(e) >= 1);
    assert_eq!(connected_count(&events), 1, "{events:?}");
    client.disconnect();
    client.join();
    thread::sleep(Duration::from_millis(100));
    let seen = game.seen.lock().unwrap();
    assert!(!seen.senders.is_empty());
    // Every datagram the game server ever got came from the relay's socket: not one from the client's own.
    assert!(
        seen.senders.iter().all(|s| *s == proxy.relay_addr()),
        "{:?} vs relay {}",
        seen.senders,
        proxy.relay_addr()
    );
    assert_eq!(seen.closes, 1, "the graceful CLOSE also went through the relay");
    assert_eq!(proxy.tcp_accepts(), 1);
}

/// Task 3.11: the precise (non-blocking) driver loop over the SOCKS5 transport: joins through the relay, and the graceful `CLOSE`
/// still goes through it (the association's socket is back in blocking mode when the connection ends).
#[test]
fn the_precise_loop_works_through_the_relay_too() {
    let proxy = TestSocks5Server::start(Config::default());
    let game = GameDouble::start(false);
    let mut cfg = config(
        entry(game.addr, Some(PROXY)),
        Some(proxy_cfg(&proxy, None).with_for_server(game.addr.to_string())),
    );
    cfg.precise_wakeups = true;
    let mut client = Client::connect(game.addr, cfg);
    let events = collect(&mut client, Duration::from_secs(5), |e| connected_count(e) >= 1);
    assert_eq!(connected_count(&events), 1, "{events:?}");
    client.disconnect();
    client.join();
    thread::sleep(Duration::from_millis(100));
    let seen = game.seen.lock().unwrap();
    assert!(seen.senders.iter().all(|s| *s == proxy.relay_addr()));
    assert_eq!(seen.closes, 1);
    assert_eq!(proxy.tcp_accepts(), 1);
}

#[test]
fn a_server_requested_reconnect_keeps_the_association() {
    let proxy = TestSocks5Server::start(Config::default());
    let game = GameDouble::start(true);
    let mut client = Client::connect(
        game.addr,
        config(entry(game.addr, Some(PROXY)), Some(proxy_cfg(&proxy, None))),
    );
    let events = collect(&mut client, Duration::from_secs(8), |e| connected_count(e) >= 2);
    client.disconnect();
    client.join();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, ClientEvent::ServerRequestedReconnect { .. })),
        "{events:?}"
    );
    wait_until(Duration::from_secs(2), || {
        game.seen.lock().unwrap().connected_from.len() >= 2
    });
    let seen = game.seen.lock().unwrap();
    assert_eq!(seen.connected_from.len(), 2, "{:?}", seen.connected_from);
    // Same relay socket, hence the same source port at the server, and no new TCP connection to the proxy.
    assert_eq!(seen.connected_from[0], seen.connected_from[1]);
    assert_eq!(proxy.tcp_accepts(), 1);
    assert_eq!(proxy.associations(), 1);
}

#[test]
fn losing_the_control_connection_is_a_lost_connection_followed_by_one_new_association() {
    let proxy = TestSocks5Server::start(Config::default());
    let game = GameDouble::start(false);
    let mut client = Client::connect(
        game.addr,
        config(entry(game.addr, Some(PROXY)), Some(proxy_cfg(&proxy, None))),
    );
    let first = collect(&mut client, Duration::from_secs(5), |e| connected_count(e) >= 1);
    assert_eq!(connected_count(&first), 1, "{first:?}");
    let first_relay = proxy.relay_addr();

    proxy.drop_control_connections();
    let events = collect(&mut client, Duration::from_secs(8), |e| {
        connected_count(e) >= 1 && e.iter().any(|x| matches!(x, ClientEvent::ReconnectAttempt { .. }))
    });
    // The loss is reported as a disconnect naming the control connection, never a credential.
    let reason = events.iter().find_map(|e| match e {
        ClientEvent::Session(s) => match &**s {
            SessionEvent::Disconnected { reason, by_peer: false } => reason.clone(),
            _ => None,
        },
        _ => None,
    });
    assert!(
        reason.as_deref().is_some_and(|r| r.contains("control connection")),
        "{reason:?} in {events:?}"
    );
    assert!(events.iter().any(|e| matches!(e, ClientEvent::ReconnectAttempt { .. })));
    assert_eq!(
        connected_count(&events),
        1,
        "re-connected through a new association: {events:?}"
    );
    assert_eq!(proxy.associations(), 2);
    assert_eq!(proxy.tcp_accepts(), 2, "one proxy connection per attempt");
    assert_ne!(proxy.relay_addr(), first_relay);
    // The double's own `Connected` comes a moment after the client's (it needs the client's first ack).
    wait_until(Duration::from_secs(2), || {
        game.seen.lock().unwrap().connected_from.len() >= 2
    });
    {
        let seen = game.seen.lock().unwrap();
        assert_eq!(seen.connected_from.len(), 2);
        assert_ne!(seen.connected_from[0], seen.connected_from[1]);
    }

    // A second loss before the session was ever in game exhausts the pre-game attempt cap (D-050): it stops,
    // and the proxy was contacted exactly twice.
    proxy.drop_control_connections();
    let end = collect(&mut client, Duration::from_secs(8), |e| gave_up(e).is_some());
    assert_eq!(
        gave_up(&end).map(|g| g.1),
        Some(GaveUpCategory::TooManyAttempts),
        "{end:?}"
    );
    client.join();
    assert_eq!(proxy.tcp_accepts(), 2);
}

#[test]
fn udp_not_supported_stops_after_one_proxy_connection() {
    let proxy = TestSocks5Server::start(Config {
        reply_code: 0x07,
        ..Default::default()
    });
    let game = GameDouble::start(false);
    let mut client = Client::connect(
        game.addr,
        config(entry(game.addr, Some(PROXY)), Some(proxy_cfg(&proxy, None))),
    );
    let events = collect(&mut client, Duration::from_secs(5), |e| gave_up(e).is_some());
    let (reason, category) = gave_up(&events).expect("the driver gave up");
    assert_eq!(category, GaveUpCategory::ProxyRefused);
    assert!(reason.contains("UDP not supported"), "{reason}");
    client.join();
    // Wait out the whole first backoff and then some: nothing retried.
    thread::sleep(Duration::from_millis(1500));
    assert_eq!(proxy.tcp_accepts(), 1);
    assert!(
        game.seen.lock().unwrap().senders.is_empty(),
        "nothing reached the game server"
    );
}

#[test]
fn a_wrong_password_stops_after_one_attempt_and_leaks_nothing() {
    let proxy = TestSocks5Server::start(Config {
        auth: Auth::UserPass("user".into(), "right-pw".into()),
        ..Default::default()
    });
    let game = GameDouble::start(false);
    let mut client = Client::connect(
        game.addr,
        config(
            entry(game.addr, Some(PROXY)),
            Some(proxy_cfg(&proxy, Some(("user", "WRONG-pw-xyz")))),
        ),
    );
    let events = collect(&mut client, Duration::from_secs(5), |e| gave_up(e).is_some());
    let (reason, category) = gave_up(&events).expect("the driver gave up");
    assert_eq!(category, GaveUpCategory::ProxyRefused);
    assert!(reason.contains("authentication failed"), "{reason}");
    let all = format!("{events:?}");
    assert!(!all.contains("WRONG-pw-xyz") && !all.contains("right-pw"), "{all}");
    client.join();
    thread::sleep(Duration::from_millis(1200));
    assert_eq!(proxy.tcp_accepts(), 1);
    assert!(game.seen.lock().unwrap().senders.is_empty());
}

#[test]
fn a_proxy_that_keeps_failing_is_bounded_by_the_pre_game_attempt_cap() {
    // Accepts the TCP connection, reads the greeting, closes: a transient failure each time.
    let proxy = TestSocks5Server::start(Config {
        raw_reply: Some(Vec::new()),
        ..Default::default()
    });
    let game = GameDouble::start(false);
    let mut client = Client::connect(
        game.addr,
        config(entry(game.addr, Some(PROXY)), Some(proxy_cfg(&proxy, None))),
    );
    let events = collect(&mut client, Duration::from_secs(8), |e| gave_up(e).is_some());
    assert_eq!(
        gave_up(&events).map(|g| g.1),
        Some(GaveUpCategory::TooManyAttempts),
        "{events:?}"
    );
    client.join();
    assert_eq!(proxy.tcp_accepts(), 2, "one proxy connection per attempt, two attempts");
    assert!(game.seen.lock().unwrap().senders.is_empty());
}

#[test]
fn a_stalled_proxy_times_out_instead_of_hanging_the_driver() {
    let proxy = TestSocks5Server::start(Config {
        stall: Stall::AfterGreeting,
        ..Default::default()
    });
    let game = GameDouble::start(false);
    let mut cfg = config(entry(game.addr, Some(PROXY)), Some(proxy_cfg(&proxy, None)));
    // The default handshake budget is 15 s and the proxy handshake bound 10 s; a short watchdog makes this quick.
    cfg.handshake_timeout = Duration::from_secs(2);
    let t0 = Instant::now();
    let mut client = Client::connect(game.addr, cfg);
    let events = collect(&mut client, Duration::from_secs(20), |e| gave_up(e).is_some());
    assert!(gave_up(&events).is_some(), "{events:?}");
    client.join();
    assert!(t0.elapsed() < Duration::from_secs(15), "{:?}", t0.elapsed());
}

#[test]
fn the_pairing_of_proxy_and_entry_is_enforced_in_both_directions_before_anything_is_sent() {
    // The entry names a proxy but none was configured: a direct connection would be the VPN-ban of D-052.
    expect_refused(|g| entry(g, Some(PROXY)), |_| None, "none was configured");
    // A proxy was configured but the entry does not name one (or there is no entry at all, loopback).
    expect_refused(|g| entry(g, None), |s| Some(proxy_cfg(s, None)), "does not name it");
    expect_refused(
        |_| LiveServers::default(),
        |s| Some(proxy_cfg(s, None)),
        "does not name it",
    );
    // The entry names a different proxy than the one configured.
    expect_refused(
        |g| entry(g, Some("other")),
        |s| Some(proxy_cfg(s, None)),
        "names proxy \"other\"",
    );
    // Entries that disagree.
    expect_refused(
        |g| {
            let mut l = entry(g, Some(PROXY));
            l.servers.push(entry(g, None).servers.remove(0));
            l
        },
        |s| Some(proxy_cfg(s, None)),
        "disagree",
    );
}

fn expect_refused(
    list: impl Fn(SocketAddr) -> LiveServers,
    proxy_cfg_for: impl FnOnce(&TestSocks5Server) -> Option<ProxyConfig>,
    must_say: &str,
) {
    let proxy = TestSocks5Server::start(Config::default());
    let game = GameDouble::start(false);
    let mut client = Client::connect(game.addr, config(list(game.addr), proxy_cfg_for(&proxy)));
    let events = collect(&mut client, Duration::from_secs(3), |e| gave_up(e).is_some());
    let (reason, category) = gave_up(&events).unwrap_or_else(|| panic!("not refused: {events:?}"));
    assert_eq!(category, GaveUpCategory::LocalError);
    assert!(reason.contains(must_say), "{reason}");
    client.join();
    thread::sleep(Duration::from_millis(150));
    assert_eq!(proxy.tcp_accepts(), 0, "the proxy was never contacted");
    assert!(
        game.seen.lock().unwrap().senders.is_empty(),
        "the game server was never contacted"
    );
}

#[test]
fn a_direct_entry_without_a_proxy_still_connects_directly() {
    let game = GameDouble::start(false);
    let mut client = Client::connect(game.addr, config(entry(game.addr, None), None));
    let events = collect(&mut client, Duration::from_secs(5), |e| connected_count(e) >= 1);
    assert_eq!(connected_count(&events), 1, "{events:?}");
    client.disconnect();
    client.join();
    // Loopback with no entry is direct too (the existing behaviour).
    let mut client = Client::connect(game.addr, config(LiveServers::default(), None));
    let events = collect(&mut client, Duration::from_secs(5), |e| connected_count(e) >= 1);
    assert_eq!(connected_count(&events), 1, "{events:?}");
    client.disconnect();
    client.join();
}

/// Task 5.12 (D-099): the proxy file's `for_server` is no binding any more. The entry (the owner's choice on the site) names the
/// proxy and that is the one rule: a proxy file that was once issued for another server still carries this session, and the
/// game server still only ever sees the relay.
#[test]
fn a_proxy_file_issued_for_another_server_is_used_where_the_entry_assigns_it() {
    let proxy = TestSocks5Server::start(Config::default());
    let game = GameDouble::start(false);
    let cfg = config(
        entry(game.addr, Some(PROXY)),
        Some(proxy_cfg(&proxy, None).with_for_server("45.141.57.35:8308")),
    );
    let mut client = Client::connect(game.addr, cfg);
    let events = collect(&mut client, Duration::from_secs(5), |e| connected_count(e) >= 1);
    assert_eq!(connected_count(&events), 1, "{events:?}");
    client.disconnect();
    client.join();
    thread::sleep(Duration::from_millis(100));
    let seen = game.seen.lock().unwrap();
    assert!(seen.senders.iter().all(|s| *s == proxy.relay_addr()));
}

// --- task 2.6b: `relay = "public"`: a relay on another host ---------------------------------------------------------

/// The whole driver through a relay on another loopback alias (the test hook makes loopback count as public): the
/// game server sees only the relay's address, and the graceful CLOSE goes through it too.
#[test]
fn in_public_mode_the_session_goes_through_a_relay_on_another_host() {
    let proxy = TestSocks5Server::start(Config {
        relay_ip: Some("127.0.0.2".parse().unwrap()),
        ..Default::default()
    });
    let game = GameDouble::start(false);
    let cfg = config(
        entry(game.addr, Some(PROXY)),
        Some(
            proxy_cfg(&proxy, None)
                .with_relay(RelayMode::Public)
                .with_test_loopback_relay()
                .with_for_server(game.addr.to_string()),
        ),
    );
    let mut client = Client::connect(game.addr, cfg);
    let events = collect(&mut client, Duration::from_secs(5), |e| connected_count(e) >= 1);
    assert_eq!(connected_count(&events), 1, "{events:?}");
    client.disconnect();
    client.join();
    thread::sleep(Duration::from_millis(100));
    let seen = game.seen.lock().unwrap();
    assert!(!seen.senders.is_empty());
    let relay = proxy.relay_addr();
    assert_eq!(relay.ip(), "127.0.0.2".parse::<std::net::IpAddr>().unwrap());
    assert!(
        seen.senders.iter().all(|s| *s == relay),
        "{:?} vs relay {relay}",
        seen.senders
    );
    assert_eq!(seen.closes, 1);
}

/// A hostile proxy announcing the game server's own IP as its relay: the client stops (`ProxyRefused`, exit 4 in the
/// bot) after one proxy connection, and not one datagram reaches the game server.
#[test]
fn in_public_mode_a_relay_that_is_the_game_server_stops_the_client_with_no_datagram() {
    let game = GameDouble::start_on("127.0.0.3", false);
    let proxy = TestSocks5Server::start(Config {
        bnd: Bnd::Fixed(SocketAddr::new(game.addr.ip(), 5000)),
        ..Default::default()
    });
    let cfg = config(
        entry(game.addr, Some(PROXY)),
        Some(
            proxy_cfg(&proxy, None)
                .with_relay(RelayMode::Public)
                .with_test_loopback_relay(),
        ),
    );
    let mut client = Client::connect(game.addr, cfg);
    let events = collect(&mut client, Duration::from_secs(5), |e| gave_up(e).is_some());
    let (reason, category) = gave_up(&events).expect("the driver gave up");
    assert_eq!(category, GaveUpCategory::ProxyRefused);
    assert!(reason.contains("game server itself"), "{reason}");
    client.join();
    thread::sleep(Duration::from_millis(1200));
    assert_eq!(proxy.tcp_accepts(), 1, "no retry");
    assert!(
        game.seen.lock().unwrap().senders.is_empty(),
        "nothing reached the game server"
    );
}

/// Production behaviour (no test hook): a loopback relay is not a public address, so public mode refuses it as fatal.
#[test]
fn in_public_mode_a_loopback_relay_is_refused_like_any_non_public_address() {
    let proxy = TestSocks5Server::start(Config {
        relay_ip: Some("127.0.0.2".parse().unwrap()),
        ..Default::default()
    });
    let game = GameDouble::start(false);
    let cfg = config(
        entry(game.addr, Some(PROXY)),
        Some(proxy_cfg(&proxy, None).with_relay(RelayMode::Public)),
    );
    let mut client = Client::connect(game.addr, cfg);
    let events = collect(&mut client, Duration::from_secs(5), |e| gave_up(e).is_some());
    let (reason, category) = gave_up(&events).expect("the driver gave up");
    assert_eq!(category, GaveUpCategory::ProxyRefused);
    assert!(reason.contains("loopback"), "{reason}");
    client.join();
    assert_eq!(proxy.tcp_accepts(), 1);
    assert!(game.seen.lock().unwrap().senders.is_empty());
}
