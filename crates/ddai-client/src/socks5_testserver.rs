//! An in-process SOCKS5 server with a UDP relay and fault injection (task 2.6). **Test support only**: compiled
//! for this crate's own tests and for other crates' tests through the `test-util` feature (`ddnet-ai`'s e2e
//! uses it as its loopback relay). It binds `127.0.0.1` and nothing else.
//!
//! What it does, configurable through [`Config`]:
//!
//! - authentication: none, or RFC 1929 username/password (a wrong password gets status 1);
//! - the `UDP ASSOCIATE` reply code (`0x07` for "UDP not supported", any other for a refusal);
//! - what `BND.ADDR` says: the real relay address, `0.0.0.0` with the real port, a domain name, or any fixed
//!   address;
//! - task 2.6b: the relay's own address (`relay_ip`: another loopback alias such as `127.0.0.2`, to test a relay on
//!   another host), a per-association relay delay (`relay_delays`, to give sessions different round-trip times), a
//!   fake DNS responder inside the relay (`fake_dns`: a datagram addressed to port 53 is answered by the relay itself,
//!   so a probe needs no network), and session-style user names (`Auth::UserPassSession`);
//! - stalls: say nothing after accept / the greeting / the authentication / the request, to exercise timeouts;
//! - a raw byte reply instead of a real one, for garbage-handling tests.
//!
//! And at run time ([`TestSocks5Server`]): inject datagrams to the client from the relay (garbage, fragmented,
//! misattributed) or from a foreign socket, drop every TCP control connection (which kills its association,
//! like a real proxy), and read back what clients sent: the greetings, the relayed wire datagrams, the number of
//! TCP connections accepted.
//!
//! The relay itself is the standard one: the first well-formed datagram from the TCP peer's IP fixes the client's
//! UDP address; datagrams from there are unwrapped and forwarded to their `DST` from the relay socket, and
//! anything arriving from anywhere else is wrapped with its source address and sent to the client.

use crate::socks5::parse_udp;
use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Auth {
    #[default]
    None,
    UserPass(String, String),
    /// Any user name that starts with the prefix, with this password: what a proxy with session ids in the user
    /// name looks like. Every user name presented is recorded ([`TestSocks5Server::users_seen`]).
    UserPassSession(String, String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Bnd {
    /// The real relay address.
    #[default]
    Relay,
    /// `0.0.0.0:<real relay port>`.
    Unspecified,
    /// `<name>:<real relay port>` as `ATYP=3`.
    Domain(String),
    /// An arbitrary address.
    Fixed(SocketAddr),
    /// This IP with the real relay port (a proxy behind NAT announcing an address that is not the one we reach it at).
    OtherIp(IpAddr),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Stall {
    #[default]
    None,
    AfterAccept,
    AfterGreeting,
    AfterAuth,
    AfterRequest,
}

#[derive(Debug, Clone, Default)]
pub struct Config {
    pub auth: Auth,
    pub bnd: Bnd,
    /// `REP` of the `UDP ASSOCIATE` reply; 0 means success.
    pub reply_code: u8,
    pub stall: Stall,
    /// If set: after reading the greeting, send exactly these bytes and close.
    pub raw_reply: Option<Vec<u8>>,
    /// If set: the `UDP ASSOCIATE` reply goes out one byte at a time with this pause between bytes (slowloris).
    pub drip: Option<Duration>,
    /// The version byte of the RFC 1929 reply (1 is correct; some real proxies send 5).
    pub auth_reply_version: Option<u8>,
    /// The address the relay socket binds (default `127.0.0.1`): another loopback alias puts the relay on "another
    /// host" (`Bnd::Relay` then announces it).
    pub relay_ip: Option<IpAddr>,
    /// Relay `n` (0-based, in the order associations are made) delays everything it forwards, in both directions, by
    /// `relay_delays[n % len]`. Empty: no delay.
    pub relay_delays: Vec<Duration>,
    /// A datagram addressed to port 53 is answered by the relay itself with a copy marked as a response (a fake
    /// resolver, as if the relay could reach one) instead of being forwarded.
    pub fake_dns: bool,
}

struct Relay {
    socket: UdpSocket,
    addr: SocketAddr,
    client: Arc<Mutex<Option<SocketAddr>>>,
    closed: Arc<AtomicBool>,
}

#[derive(Default)]
struct Shared {
    stop: AtomicBool,
    tcp_accepts: AtomicUsize,
    greetings: Mutex<Vec<Vec<u8>>>,
    users: Mutex<Vec<String>>,
    datagrams: Mutex<Vec<(SocketAddr, Vec<u8>)>>,
    relays: Mutex<Vec<Arc<Relay>>>,
    controls: Mutex<Vec<TcpStream>>,
}

pub struct TestSocks5Server {
    addr: SocketAddr,
    shared: Arc<Shared>,
    foreign: UdpSocket,
}

impl TestSocks5Server {
    /// Starts listening on `127.0.0.1:<free port>`.
    pub fn start(config: Config) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind the test SOCKS5 listener");
        listener.set_nonblocking(true).expect("nonblocking listener");
        let addr = listener.local_addr().expect("listener address");
        let shared = Arc::new(Shared::default());
        let accept_shared = Arc::clone(&shared);
        let config = Arc::new(config);
        thread::Builder::new()
            .name("socks5-test-accept".into())
            .spawn(move || {
                while !accept_shared.stop.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((conn, _)) => {
                            // An accepted socket inherits the listener's non-blocking mode on Windows (and the BSDs), not on Linux: the
                            // handler reads with blocking calls, so undo it.
                            let _ = conn.set_nonblocking(false);
                            accept_shared.tcp_accepts.fetch_add(1, Ordering::SeqCst);
                            if let Ok(clone) = conn.try_clone() {
                                accept_shared.controls.lock().unwrap().push(clone);
                            }
                            let shared = Arc::clone(&accept_shared);
                            let config = Arc::clone(&config);
                            let _ = thread::Builder::new()
                                .name("socks5-test-conn".into())
                                .spawn(move || handle_control(conn, &shared, &config));
                        }
                        Err(_) => thread::sleep(Duration::from_millis(5)),
                    }
                }
            })
            .expect("spawn the accept thread");
        TestSocks5Server {
            addr,
            shared,
            foreign: UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind the foreign socket"),
        }
    }

    /// The proxy's TCP address.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The UDP relay address of the most recent association. Panics if there has been none.
    pub fn relay_addr(&self) -> SocketAddr {
        self.shared
            .relays
            .lock()
            .unwrap()
            .last()
            .expect("no association has been made yet")
            .addr
    }

    /// The UDP relay address of every association so far, in order.
    pub fn relay_addrs(&self) -> Vec<SocketAddr> {
        self.shared.relays.lock().unwrap().iter().map(|r| r.addr).collect()
    }

    /// TCP connections accepted so far.
    pub fn tcp_accepts(&self) -> usize {
        self.shared.tcp_accepts.load(Ordering::SeqCst)
    }

    /// Associations (`UDP ASSOCIATE` successes) made so far.
    pub fn associations(&self) -> usize {
        self.shared.relays.lock().unwrap().len()
    }

    /// The greeting (`VER NMETHODS METHODS...`) of every connection, in order.
    pub fn greetings(&self) -> Vec<Vec<u8>> {
        self.shared.greetings.lock().unwrap().clone()
    }

    /// Every user name presented in RFC 1929 authentication, in order (right or wrong).
    pub fn users_seen(&self) -> Vec<String> {
        self.shared.users.lock().unwrap().clone()
    }

    /// Every datagram the relays received from a client, with the client's UDP source, raw (header included).
    pub fn datagrams_from_clients(&self) -> Vec<(SocketAddr, Vec<u8>)> {
        self.shared.datagrams.lock().unwrap().clone()
    }

    /// Closes every TCP control connection (and with it, as RFC 1928 says, every association).
    pub fn drop_control_connections(&self) {
        for c in self.shared.controls.lock().unwrap().drain(..) {
            let _ = c.shutdown(Shutdown::Both);
        }
        for r in self.shared.relays.lock().unwrap().iter() {
            r.closed.store(true, Ordering::SeqCst);
        }
    }

    /// Sends `bytes` to the client's UDP address from the **relay's own socket** (so the source is the
    /// relay): for garbage, fragments, wrong embedded sources. Waits up to a second for the client's UDP
    /// address to be known (it is learned from the client's first datagram).
    pub fn inject_to_client(&self, bytes: &[u8]) {
        let relay = self
            .shared
            .relays
            .lock()
            .unwrap()
            .last()
            .cloned()
            .expect("no association has been made yet");
        let client = wait_for_client(&relay);
        relay
            .socket
            .send_to(bytes, client)
            .expect("inject from the relay socket");
    }

    /// Sends `bytes` to the client's UDP address from a **different socket** (a foreign source).
    pub fn inject_from_foreign_socket(&self, bytes: &[u8]) {
        let relay = self
            .shared
            .relays
            .lock()
            .unwrap()
            .last()
            .cloned()
            .expect("no association has been made yet");
        let client = wait_for_client(&relay);
        self.foreign
            .send_to(bytes, client)
            .expect("inject from the foreign socket");
    }
}

fn wait_for_client(relay: &Relay) -> SocketAddr {
    let end = Instant::now() + Duration::from_secs(1);
    loop {
        if let Some(c) = *relay.client.lock().unwrap() {
            return c;
        }
        assert!(
            Instant::now() < end,
            "the client has not sent a UDP datagram yet, so its address is unknown"
        );
        thread::sleep(Duration::from_millis(2));
    }
}

impl Drop for TestSocks5Server {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.drop_control_connections();
    }
}

/// Blocks until the peer closes (or the server shuts the stream down): the "say nothing" stalls and the
/// post-association wait both end this way.
fn hold(conn: &mut TcpStream) {
    let mut b = [0u8; 64];
    loop {
        match conn.read(&mut b) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
    }
}

fn handle_control(mut conn: TcpStream, shared: &Arc<Shared>, cfg: &Config) {
    serve_control(&mut conn, shared, cfg);
    // `conn` has a clone in `Shared::controls`, so dropping it would not close the socket: shut it down, as a
    // real proxy closing the connection would.
    let _ = conn.shutdown(Shutdown::Both);
}

fn serve_control(conn: &mut TcpStream, shared: &Arc<Shared>, cfg: &Config) {
    let Ok(peer) = conn.peer_addr() else { return };
    if cfg.stall == Stall::AfterAccept {
        return hold(conn);
    }
    // Greeting.
    let mut head = [0u8; 2];
    if conn.read_exact(&mut head).is_err() {
        return;
    }
    let mut methods = vec![0u8; usize::from(head[1])];
    if conn.read_exact(&mut methods).is_err() {
        return;
    }
    let mut greeting = head.to_vec();
    greeting.extend_from_slice(&methods);
    shared.greetings.lock().unwrap().push(greeting);
    if let Some(raw) = &cfg.raw_reply {
        let _ = conn.write_all(raw);
        return;
    }
    let method = match &cfg.auth {
        Auth::None if methods.contains(&0x00) => 0x00,
        Auth::UserPass(..) | Auth::UserPassSession(..) if methods.contains(&0x02) => 0x02,
        _ => 0xff,
    };
    if conn.write_all(&[5, method]).is_err() || method == 0xff {
        return;
    }
    if cfg.stall == Stall::AfterGreeting {
        return hold(conn);
    }
    // RFC 1929.
    if method == 0x02 {
        let (Auth::UserPass(want_user, want_pass) | Auth::UserPassSession(want_user, want_pass)) = &cfg.auth else {
            return;
        };
        let mut ver_ulen = [0u8; 2];
        if conn.read_exact(&mut ver_ulen).is_err() {
            return;
        }
        let mut user = vec![0u8; usize::from(ver_ulen[1])];
        let mut plen = [0u8; 1];
        if conn.read_exact(&mut user).is_err() || conn.read_exact(&mut plen).is_err() {
            return;
        }
        let mut pass = vec![0u8; usize::from(plen[0])];
        if conn.read_exact(&mut pass).is_err() {
            return;
        }
        shared
            .users
            .lock()
            .unwrap()
            .push(String::from_utf8_lossy(&user).into_owned());
        let user_ok = match &cfg.auth {
            Auth::UserPassSession(..) => user.starts_with(want_user.as_bytes()),
            _ => user == want_user.as_bytes(),
        };
        let ok = ver_ulen[0] == 1 && user_ok && pass == want_pass.as_bytes();
        let version = cfg.auth_reply_version.unwrap_or(1);
        if conn.write_all(&[version, u8::from(!ok)]).is_err() || !ok {
            return;
        }
        if cfg.stall == Stall::AfterAuth {
            return hold(conn);
        }
    }
    // Request: VER CMD RSV ATYP DST.ADDR DST.PORT.
    let mut req = [0u8; 4];
    if conn.read_exact(&mut req).is_err() {
        return;
    }
    let addr_len = match req[3] {
        1 => 4,
        4 => 16,
        3 => {
            let mut l = [0u8; 1];
            if conn.read_exact(&mut l).is_err() {
                return;
            }
            usize::from(l[0])
        }
        _ => return,
    };
    let mut rest = vec![0u8; addr_len + 2];
    if conn.read_exact(&mut rest).is_err() {
        return;
    }
    if cfg.stall == Stall::AfterRequest {
        return hold(conn);
    }
    if req[1] != 3 || cfg.reply_code != 0 {
        let code = if req[1] != 3 { 0x07 } else { cfg.reply_code };
        let _ = conn.write_all(&[5, code, 0, 1, 0, 0, 0, 0, 0, 0]);
        return;
    }
    // Success: create the relay.
    let socket =
        UdpSocket::bind((cfg.relay_ip.unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST)), 0)).expect("bind the relay socket");
    socket
        .set_read_timeout(Some(Duration::from_millis(20)))
        .expect("relay timeout");
    let relay_addr = socket.local_addr().expect("relay address");
    let relay = Arc::new(Relay {
        socket,
        addr: relay_addr,
        client: Arc::new(Mutex::new(None)),
        closed: Arc::new(AtomicBool::new(false)),
    });
    let delay = {
        let mut relays = shared.relays.lock().unwrap();
        relays.push(Arc::clone(&relay));
        if cfg.relay_delays.is_empty() {
            Duration::ZERO
        } else {
            cfg.relay_delays[(relays.len() - 1) % cfg.relay_delays.len()]
        }
    };
    let fake_dns = cfg.fake_dns;
    let relay_shared = Arc::clone(shared);
    let relay_for_thread = Arc::clone(&relay);
    let _ = thread::Builder::new()
        .name("socks5-test-relay".into())
        .spawn(move || run_relay(&relay_for_thread, &relay_shared, peer.ip(), delay, fake_dns));

    let mut reply = vec![5, 0, 0];
    match &cfg.bnd {
        Bnd::Relay => push_addr(&mut reply, relay_addr),
        Bnd::Unspecified => push_addr(
            &mut reply,
            SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), relay_addr.port()),
        ),
        Bnd::Fixed(a) => push_addr(&mut reply, *a),
        Bnd::OtherIp(ip) => push_addr(&mut reply, SocketAddr::new(*ip, relay_addr.port())),
        Bnd::Domain(name) => {
            reply.push(3);
            reply.push(u8::try_from(name.len()).expect("short domain"));
            reply.extend_from_slice(name.as_bytes());
            reply.extend_from_slice(&relay_addr.port().to_be_bytes());
        }
    }
    let written = match cfg.drip {
        None => conn.write_all(&reply),
        Some(pause) => reply.iter().try_for_each(|b| {
            thread::sleep(pause);
            conn.write_all(&[*b])?;
            conn.flush()
        }),
    };
    if written.is_err() {
        relay.closed.store(true, Ordering::SeqCst);
        return;
    }
    // The association lives as long as this connection.
    hold(conn);
    relay.closed.store(true, Ordering::SeqCst);
}

fn push_addr(out: &mut Vec<u8>, addr: SocketAddr) {
    match addr.ip() {
        IpAddr::V4(ip) => {
            out.push(1);
            out.extend_from_slice(&ip.octets());
        }
        IpAddr::V6(ip) => {
            out.push(4);
            out.extend_from_slice(&ip.octets());
        }
    }
    out.extend_from_slice(&addr.port().to_be_bytes());
}

fn run_relay(relay: &Relay, shared: &Shared, client_ip: IpAddr, delay: Duration, fake_dns: bool) {
    let mut buf = vec![0u8; 4096];
    while !relay.closed.load(Ordering::SeqCst) && !shared.stop.load(Ordering::SeqCst) {
        let Ok((n, from)) = relay.socket.recv_from(&mut buf) else {
            continue;
        };
        let known = *relay.client.lock().unwrap();
        let from_client = match known {
            Some(c) => c == from,
            None => {
                // The first well-formed datagram from the TCP peer's IP fixes the client's UDP address.
                if from.ip() == client_ip && parse_udp(&buf[..n]).is_ok() {
                    *relay.client.lock().unwrap() = Some(from);
                    true
                } else {
                    false
                }
            }
        };
        if from_client {
            shared.datagrams.lock().unwrap().push((from, buf[..n].to_vec()));
            if let Ok(packet) = parse_udp(&buf[..n])
                && let crate::socks5::UdpSource::Ip(dst) = packet.source
            {
                if !delay.is_zero() {
                    thread::sleep(delay);
                }
                if fake_dns && dst.port() == 53 {
                    // The relay plays the resolver: the query with the QR bit set, from the address it was sent to.
                    let mut answer = packet.payload.to_vec();
                    if answer.len() >= 12 {
                        answer[2] |= 0x80;
                        let mut wrapped = Vec::with_capacity(answer.len() + 10);
                        crate::socks5::encode_udp_into(&mut wrapped, dst, &answer);
                        let _ = relay.socket.send_to(&wrapped, from);
                    }
                } else if dst.ip().is_loopback() {
                    // Loopback only: a test can never reach a real host through this relay.
                    let _ = relay.socket.send_to(packet.payload, dst);
                }
            }
        } else if let Some(client) = *relay.client.lock().unwrap() {
            // A reply from a game server: wrap it with the source and hand it to the client.
            if !delay.is_zero() {
                thread::sleep(delay);
            }
            let mut wrapped = Vec::with_capacity(n + 10);
            crate::socks5::encode_udp_into(&mut wrapped, from, &buf[..n]);
            let _ = relay.socket.send_to(&wrapped, client);
        }
    }
}
