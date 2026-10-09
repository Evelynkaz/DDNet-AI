//! SOCKS5 (RFC 1928) with username/password authentication (RFC 1929) and `UDP ASSOCIATE` (task 2.6):
//! the wire codec, the TCP control handshake, and [`Socks5UdpTransport`], the [`Transport`] that carries the
//! game's datagrams through the proxy's UDP relay. Standard library only.
//!
//! # Shape
//!
//! 1. [`associate`] opens the TCP control connection (every step bounded by [`Timeouts`]), negotiates no
//!    authentication or username/password, sends `UDP ASSOCIATE` and returns the relay address. Reply
//!    `0x07` becomes [`Socks5Error::UdpNotSupported`], which [`Socks5Error::is_fatal`] says stops all retrying.
//!    `BND.ADDR` is the proxy's claim and is **not trusted**. With `relay = "proxy-host-only"` (the default)
//!    datagrams always go to the proxy's own IP (the TCP peer) with the announced port: an unspecified address or
//!    the proxy's own is the normal case; any other (NAT, private, the game server's own IP, ...) is substituted
//!    and logged, never obeyed. With `relay = "public"` (task 2.6b, for proxies whose relay lives on another
//!    machine) an address on another host is obeyed only if it is a public unicast address
//!    (`crate::relay_rule::classify`), and, once the game server is known, none of the server's IPs and not the
//!    server's port ([`check_relay_is_not_target`]); a domain name is never resolved and is refused.
//! 2. [`Socks5UdpTransport`] wraps every outgoing datagram in the RFC 1928 §7 header (`RSV=0000`, `FRAG=0`,
//!    `ATYP`, `DST.ADDR`, `DST.PORT`) and sends it to the relay; an incoming datagram is accepted only if it
//!    comes from the relay's address, has `RSV=0000` and `FRAG=0`, parses cleanly, and names the game server
//!    as its source. Everything else is dropped and counted ([`DropStats`]), never an error and never a panic.
//! 3. The association lives exactly as long as the TCP control connection (RFC 1928 §6): the transport polls
//!    it on every `recv`, and when it has closed or errored the next `recv` fails with
//!    [`Socks5Error::ControlClosed`]; the driver then treats the connection as lost and goes through its normal
//!    reconnect policy (backoff, the 5-per-20-s attempt table, the pre-game attempt cap), and
//!    [`Transport::begin_attempt`] opens a *new* control connection, at most one per attempt.
//!
//! No credential, proxy address or port is ever put into an error, a log line or a `Debug` output: errors carry
//! the step and the `io::ErrorKind` only (`crate::proxy::Secret`).

use crate::proxy::{ProxyConfig, RelayMode, fresh_session_token};
use crate::relay_probe::{DNS_PROBE_TARGET, ProbePlan, probe_rtt};
use crate::relay_rule::{RelayClass, RelayTarget, classify, is_local_address};
use crate::transport::{Transport, TransportError};
use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream, ToSocketAddrs, UdpSocket};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const SOCKS_VERSION: u8 = 5;
const AUTH_VERSION: u8 = 1;
const METHOD_NO_AUTH: u8 = 0x00;
const METHOD_USERPASS: u8 = 0x02;
const METHOD_NONE_ACCEPTABLE: u8 = 0xff;
const CMD_UDP_ASSOCIATE: u8 = 0x03;
const ATYP_IPV4: u8 = 0x01;
const ATYP_DOMAIN: u8 = 0x03;
const ATYP_IPV6: u8 = 0x04;
/// RFC 1928 §6, `REP`.
const REP_SUCCEEDED: u8 = 0x00;
const REP_NOT_ALLOWED: u8 = 0x02;
const REP_COMMAND_NOT_SUPPORTED: u8 = 0x07;
const REP_ADDRESS_TYPE_NOT_SUPPORTED: u8 = 0x08;
/// The most `ATYP=3` can carry.
const MAX_DOMAIN_LEN: usize = 255;
/// Header bytes before `DST.ADDR` / after the domain length byte etc. are computed in the codec; this is the
/// largest possible header (domain of 255 bytes): 4 + 1 + 255 + 2.
const MAX_UDP_HEADER: usize = 4 + 1 + MAX_DOMAIN_LEN + 2;
/// Incoming datagram buffer: a DDNet packet is at most 1400 bytes; room for the largest header and then some.
const RECV_BUF_SIZE: usize = 2048 + MAX_UDP_HEADER;

/// The part of the handshake an error happened in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Resolve,
    Connect,
    Greeting,
    Auth,
    Associate,
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Step::Resolve => "resolving the proxy",
            Step::Connect => "connecting to the proxy",
            Step::Greeting => "method negotiation",
            Step::Auth => "username/password authentication",
            Step::Associate => "UDP ASSOCIATE",
        })
    }
}

/// Everything that can go wrong talking to a SOCKS5 proxy. Typed so a caller can tell "UDP not supported"
/// from "wrong password" from a flaky network; no variant carries a credential or an address.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Socks5Error {
    #[error("{step}: I/O error ({kind:?})")]
    Io { step: Step, kind: io::ErrorKind },
    #[error("{step}: timed out")]
    Timeout { step: Step },
    #[error("{step}: the proxy closed the connection")]
    Closed { step: Step },
    #[error(
        "the proxy accepts none of the offered authentication methods (it probably needs a username and password, or does not accept the ones given)"
    )]
    NoAcceptableMethod,
    #[error("the proxy selected an authentication method that was not offered ({0:#04x})")]
    UnsupportedMethod(u8),
    #[error("proxy authentication failed (the proxy rejected the username/password)")]
    AuthFailed,
    /// Reply `0x07` ("command not supported") to `UDP ASSOCIATE`: this proxy is TCP-only (the D-053 case).
    #[error("UDP not supported: the proxy answered UDP ASSOCIATE with reply 0x07 (command not supported)")]
    UdpNotSupported,
    #[error("the proxy refused UDP ASSOCIATE: reply {code:#04x} ({meaning})")]
    Refused { code: u8, meaning: &'static str },
    #[error("malformed reply from the proxy: {0}")]
    Protocol(&'static str),
    #[error("unusable relay address in the proxy's reply: {0}")]
    RelayAddress(&'static str),
    /// The UDP probe through the relay got no answer to any query (task 2.6b): the relay does not carry UDP for us.
    #[error("the UDP relay answered none of the DNS probe queries")]
    ProbeFailed,
    /// The TCP control connection closed or failed after the association was set up: the association is gone.
    #[error("the proxy's TCP control connection closed: the UDP association is gone")]
    ControlClosed,
}

impl Socks5Error {
    /// Whether retrying is pointless or harmful: a wrong password (repeating it can lock the account), a
    /// proxy without UDP (reply 0x07: no hammering a proxy that cannot do it), no usable authentication
    /// method, a rule-set refusal, a proxy that speaks garbage. Network trouble, timeouts and a proxy that
    /// merely failed this once are *not* fatal: the driver's backoff and attempt limits bound them.
    pub fn is_fatal(&self) -> bool {
        match self {
            Socks5Error::NoAcceptableMethod
            | Socks5Error::UnsupportedMethod(_)
            | Socks5Error::AuthFailed
            | Socks5Error::UdpNotSupported
            | Socks5Error::Protocol(_)
            | Socks5Error::RelayAddress(_) => true,
            Socks5Error::Refused { code, .. } => matches!(*code, REP_NOT_ALLOWED | REP_ADDRESS_TYPE_NOT_SUPPORTED),
            Socks5Error::Io { .. }
            | Socks5Error::Timeout { .. }
            | Socks5Error::Closed { .. }
            | Socks5Error::ProbeFailed
            | Socks5Error::ControlClosed => false,
        }
    }
}

fn io_err(step: Step, e: io::Error) -> Socks5Error {
    use io::ErrorKind::*;
    match e.kind() {
        WouldBlock | TimedOut => Socks5Error::Timeout { step },
        UnexpectedEof | ConnectionReset | ConnectionAborted | BrokenPipe => Socks5Error::Closed { step },
        kind => Socks5Error::Io { step, kind },
    }
}

fn reply_meaning(code: u8) -> &'static str {
    match code {
        0x01 => "general SOCKS server failure",
        0x02 => "connection not allowed by ruleset",
        0x03 => "network unreachable",
        0x04 => "host unreachable",
        0x05 => "connection refused",
        0x06 => "TTL expired",
        0x07 => "command not supported",
        0x08 => "address type not supported",
        _ => "unassigned",
    }
}

// ---------------------------------------------------------------------------------------------------
// UDP datagram header (RFC 1928 section 7): pure functions over bytes.
// ---------------------------------------------------------------------------------------------------

/// Where an incoming relayed datagram says it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UdpSource {
    Ip(SocketAddr),
    /// `ATYP=3`. The name is not kept: a game server is identified by IP, so a domain source can never be
    /// verified and the transport drops it.
    Domain,
}

/// A parsed relayed datagram: its claimed source and the payload after the header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdpPacket<'a> {
    pub source: UdpSource,
    pub payload: &'a [u8],
}

/// Why [`parse_udp`] rejected a datagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UdpDrop {
    /// Shorter than its own header says.
    TooShort,
    /// `RSV` is not `0000`.
    Reserved,
    /// `FRAG` is not 0: we implement no reassembly (RFC 1928 allows that), so a fragment is dropped.
    Fragmented,
    /// `ATYP` is none of 1, 3, 4.
    BadAddressType,
    /// A zero-length domain.
    BadDomain,
}

/// Writes the RFC 1928 §7 header for a datagram to `dst` followed by `payload` into `out` (cleared first).
pub fn encode_udp_into(out: &mut Vec<u8>, dst: SocketAddr, payload: &[u8]) {
    out.clear();
    out.extend_from_slice(&[0, 0, 0]);
    match dst.ip() {
        IpAddr::V4(ip) => {
            out.push(ATYP_IPV4);
            out.extend_from_slice(&ip.octets());
        }
        IpAddr::V6(ip) => {
            out.push(ATYP_IPV6);
            out.extend_from_slice(&ip.octets());
        }
    }
    out.extend_from_slice(&dst.port().to_be_bytes());
    out.extend_from_slice(payload);
}

/// Parses a datagram received from the relay. Total over arbitrary bytes: every bound is checked with
/// `get`, nothing indexes unchecked, nothing allocates, and the result borrows `buf`.
pub fn parse_udp(buf: &[u8]) -> Result<UdpPacket<'_>, UdpDrop> {
    let [rsv0, rsv1, frag, atyp, rest @ ..] = buf else {
        return Err(UdpDrop::TooShort);
    };
    if *rsv0 != 0 || *rsv1 != 0 {
        return Err(UdpDrop::Reserved);
    }
    if *frag != 0 {
        return Err(UdpDrop::Fragmented);
    }
    let port_of = |b: &[u8]| u16::from_be_bytes([b[0], b[1]]);
    match *atyp {
        ATYP_IPV4 => {
            let (addr, rest) = rest.split_first_chunk::<4>().ok_or(UdpDrop::TooShort)?;
            let (port, payload) = rest.split_first_chunk::<2>().ok_or(UdpDrop::TooShort)?;
            Ok(UdpPacket {
                source: UdpSource::Ip(SocketAddr::new(IpAddr::V4(Ipv4Addr::from(*addr)), port_of(port))),
                payload,
            })
        }
        ATYP_IPV6 => {
            let (addr, rest) = rest.split_first_chunk::<16>().ok_or(UdpDrop::TooShort)?;
            let (port, payload) = rest.split_first_chunk::<2>().ok_or(UdpDrop::TooShort)?;
            Ok(UdpPacket {
                source: UdpSource::Ip(SocketAddr::new(IpAddr::V6(Ipv6Addr::from(*addr)), port_of(port))),
                payload,
            })
        }
        ATYP_DOMAIN => {
            let (&len, rest) = rest.split_first().ok_or(UdpDrop::TooShort)?;
            if len == 0 {
                return Err(UdpDrop::BadDomain);
            }
            let rest = rest.get(usize::from(len)..).ok_or(UdpDrop::TooShort)?;
            let (_port, payload) = rest.split_first_chunk::<2>().ok_or(UdpDrop::TooShort)?;
            Ok(UdpPacket {
                source: UdpSource::Domain,
                payload,
            })
        }
        _ => Err(UdpDrop::BadAddressType),
    }
}

// ---------------------------------------------------------------------------------------------------
// TCP control handshake.
// ---------------------------------------------------------------------------------------------------

/// Bounds on the control handshake. Every blocking step uses the smaller of its own bound and what is left
/// of `total`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
    /// Name resolution and each TCP connect.
    pub connect: Duration,
    /// Each read or write of the handshake.
    pub step: Duration,
    /// The whole handshake, resolution to reply.
    pub total: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Timeouts {
            connect: Duration::from_secs(5),
            step: Duration::from_secs(5),
            total: Duration::from_secs(10),
        }
    }
}

struct Deadline {
    at: Instant,
    cap: Duration,
}

impl Deadline {
    /// The time a step may still block, or `Timeout` when the whole handshake is out of time.
    fn left(&self, step: Step, cap: Duration) -> Result<Duration, Socks5Error> {
        let left = self.at.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(Socks5Error::Timeout { step });
        }
        Ok(left.min(cap).min(self.cap))
    }
}

/// How the relay's address relates to the proxy's: what [`ProxyCheck`] reports instead of an address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelayHost {
    /// `BND.ADDR` was unspecified (`0.0.0.0`/`::`) or equal to the proxy's IP: used as announced.
    SameAsProxy,
    /// `BND.ADDR` named some other IP (a proxy behind NAT announcing its internal address, a multi-homed
    /// proxy, or a hostile one): it is **not** trusted. The proxy's own IP (from the control connection) is
    /// used with the announced port, the usual handling for a proxy behind NAT. Game datagrams therefore
    /// never go anywhere but the host we hold the control connection to.
    Substituted,
    /// `relay = "public"` only: `BND.ADDR` named another host, it is a public unicast address, and it is used as
    /// announced. Whether it is the game server is checked when the server is known ([`check_relay_is_not_target`]).
    Remote,
}

/// What a successful `UDP ASSOCIATE` produced.
#[derive(Debug)]
pub struct Established {
    /// The TCP control connection; the association lives exactly as long as it does.
    pub control: TcpStream,
    /// Where to send relayed datagrams: always on the proxy's own IP (see [`RelayHost::Substituted`]).
    pub relay: SocketAddr,
    pub relay_host: RelayHost,
    /// The proxy's IP, as seen on the control connection.
    pub proxy_ip: IpAddr,
}

/// The outcome of [`check`]: what `ddnet-ai proxy-check` prints (no address, no credential).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyCheck {
    pub relay_host: RelayHost,
    pub relay_port: u16,
    pub authenticated: bool,
    /// The rule that applied (`relay` of the proxy file).
    pub mode: RelayMode,
    /// Public mode (or session picking): the UDP probe through the relay. `None` in the default mode.
    pub probe: Option<ProbeReport>,
    /// Session picking: one entry per session tried, in order (`None` = no answer or failed), and which one won. No user
    /// names, only round-trip times.
    pub sessions: Option<SessionReport>,
}

/// The UDP probe of one relay: `replies` answers to `sent` DNS queries, with their median round-trip time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeReport {
    pub sent: u16,
    pub replies: u16,
    pub median: Duration,
}

/// The sessions `session_pick` tried (median RTT of each, `None` for one that failed or got no answer) and the index of
/// the one kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionReport {
    pub rtts: Vec<Option<Duration>>,
    pub picked: usize,
}

fn resolve_with_timeout(host: &str, port: u16, timeout: Duration) -> Result<Vec<SocketAddr>, Socks5Error> {
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Ok(vec![SocketAddr::new(ip, port)]);
    }
    let (tx, rx) = mpsc::channel();
    let host = host.to_string();
    // `getaddrinfo` cannot be interrupted, so it runs on a helper thread the handshake stops waiting for
    // when the time is up (the thread ends with the resolver call).
    std::thread::Builder::new()
        .name("ddai-socks5-resolve".into())
        .spawn(move || {
            let _ = tx.send((host.as_str(), port).to_socket_addrs().map(|it| it.collect::<Vec<_>>()));
        })
        .map_err(|e| Socks5Error::Io {
            step: Step::Resolve,
            kind: e.kind(),
        })?;
    match rx.recv_timeout(timeout) {
        Ok(Ok(addrs)) if !addrs.is_empty() => Ok(addrs),
        Ok(Ok(_)) => Err(Socks5Error::Io {
            step: Step::Resolve,
            kind: io::ErrorKind::NotFound,
        }),
        Ok(Err(e)) => Err(Socks5Error::Io {
            step: Step::Resolve,
            kind: e.kind(),
        }),
        Err(_) => Err(Socks5Error::Timeout { step: Step::Resolve }),
    }
}

/// Fills `buf` from the control connection. The timeout is recomputed before **every** `read`, so a proxy that
/// drips one byte at a time cannot stretch a step: once the total deadline has passed the step is a `Timeout`.
fn read_exact_step(
    stream: &mut TcpStream,
    dl: &Deadline,
    t: &Timeouts,
    step: Step,
    buf: &mut [u8],
) -> Result<(), Socks5Error> {
    let mut filled = 0;
    while filled < buf.len() {
        stream
            .set_read_timeout(Some(dl.left(step, t.step)?))
            .map_err(|e| io_err(step, e))?;
        match stream.read(&mut buf[filled..]) {
            Ok(0) => return Err(Socks5Error::Closed { step }),
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(io_err(step, e)),
        }
    }
    Ok(())
}

fn write_step(
    stream: &mut TcpStream,
    dl: &Deadline,
    t: &Timeouts,
    step: Step,
    bytes: &[u8],
) -> Result<(), Socks5Error> {
    stream
        .set_write_timeout(Some(dl.left(step, t.step)?))
        .map_err(|e| io_err(step, e))?;
    stream.write_all(bytes).map_err(|e| io_err(step, e))
}

/// Reads `ATYP` + address + port of a reply (RFC 1928 §6) after the first four bytes were read.
fn read_bound_address(
    stream: &mut TcpStream,
    dl: &Deadline,
    t: &Timeouts,
    atyp: u8,
) -> Result<(BoundHost, u16), Socks5Error> {
    let step = Step::Associate;
    let host = match atyp {
        ATYP_IPV4 => {
            let mut a = [0u8; 4];
            read_exact_step(stream, dl, t, step, &mut a)?;
            BoundHost::Ip(IpAddr::V4(Ipv4Addr::from(a)))
        }
        ATYP_IPV6 => {
            let mut a = [0u8; 16];
            read_exact_step(stream, dl, t, step, &mut a)?;
            BoundHost::Ip(IpAddr::V6(Ipv6Addr::from(a)))
        }
        ATYP_DOMAIN => {
            let mut len = [0u8; 1];
            read_exact_step(stream, dl, t, step, &mut len)?;
            if len[0] == 0 {
                return Err(Socks5Error::Protocol("empty domain in BND.ADDR"));
            }
            let mut name = vec![0u8; usize::from(len[0])];
            read_exact_step(stream, dl, t, step, &mut name)?;
            // The name is read to keep the stream in step and then ignored (never resolved, see `associate`).
            BoundHost::Domain
        }
        _ => return Err(Socks5Error::Protocol("unknown address type in the reply")),
    };
    let mut port = [0u8; 2];
    read_exact_step(stream, dl, t, step, &mut port)?;
    Ok((host, u16::from_be_bytes(port)))
}

enum BoundHost {
    Ip(IpAddr),
    Domain,
}

/// How long the control connection may be idle before the first TCP keepalive probe (task 4.10, D-100).
pub const KEEPALIVE_IDLE: Duration = Duration::from_secs(30);
/// The gap between keepalive probes once they have started.
pub const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);
/// Unanswered probes before the kernel declares the connection dead: a silently dead proxy is noticed after
/// `KEEPALIVE_IDLE + KEEPALIVE_RETRIES * KEEPALIVE_INTERVAL` = 105 s: the keepalive gives up only after the game's own 100 s silence timeout would have (review F1: a shorter bound turns a
/// recoverable blip into a reconnect and a ghost), instead of never.
pub const KEEPALIVE_RETRIES: u32 = 5;

/// Turns on TCP keepalive for the SOCKS5 control connection (`SO_KEEPALIVE` with a short idle time, interval and probe count, through
/// `socket2`, no `unsafe`). The association lives exactly as long as this connection (RFC 1928 §6), and the proxy provider's idle
/// timer or a NAT in between can drop an idle TCP connection without a word: the probes keep it from looking idle and make a dead one
/// fail (the next `read` returns an error, [`Socks5UdpTransport`] reports the loss and the driver reconnects through a new
/// association). Best effort: a platform that refuses an option leaves the connection as it was, with a warning.
pub fn set_control_keepalive(stream: &TcpStream) {
    let keepalive = socket2::TcpKeepalive::new()
        .with_time(KEEPALIVE_IDLE)
        .with_interval(KEEPALIVE_INTERVAL)
        .with_retries(KEEPALIVE_RETRIES);
    if let Err(e) = socket2::SockRef::from(stream).set_tcp_keepalive(&keepalive) {
        tracing::warn!(error = %e, "socks5: could not enable TCP keepalive on the control connection");
    }
}

/// Opens the control connection and completes `UDP ASSOCIATE`; see the module docs. Credentials are sent only
/// if the proxy selects username/password, and are offered only if the config has them. A `{session}` placeholder in
/// the user name gets a fresh random token.
pub fn associate(cfg: &ProxyConfig, timeouts: &Timeouts) -> Result<Established, Socks5Error> {
    associate_session(cfg, None, timeouts)
}

/// [`associate`] with an explicit session token for the `{session}` placeholder of the user name (none: a fresh one).
pub fn associate_session(
    cfg: &ProxyConfig,
    session: Option<&str>,
    timeouts: &Timeouts,
) -> Result<Established, Socks5Error> {
    let dl = Deadline {
        at: Instant::now() + timeouts.total,
        cap: timeouts.total,
    };
    // 1. Resolve and connect.
    let addrs = resolve_with_timeout(
        cfg.host().expose(),
        cfg.port(),
        dl.left(Step::Resolve, timeouts.connect)?,
    )?;
    let mut stream = None;
    let mut last_err = Socks5Error::Io {
        step: Step::Connect,
        kind: io::ErrorKind::NotFound,
    };
    for addr in addrs.iter().take(4) {
        let left = match dl.left(Step::Connect, timeouts.connect) {
            Ok(l) => l,
            Err(e) => {
                last_err = e;
                break;
            }
        };
        match TcpStream::connect_timeout(addr, left) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => {
                last_err = match io_err(Step::Connect, e) {
                    // A refused/reset connect is a plain I/O failure at this step, not "closed".
                    Socks5Error::Closed { step } => Socks5Error::Io {
                        step,
                        kind: io::ErrorKind::ConnectionRefused,
                    },
                    other => other,
                }
            }
        }
    }
    let Some(mut stream) = stream else { return Err(last_err) };
    let proxy_ip = stream
        .peer_addr()
        .map(|a| a.ip())
        .map_err(|e| io_err(Step::Connect, e))?;
    let _ = stream.set_nodelay(true);
    set_control_keepalive(&stream);

    // 2. Method negotiation (RFC 1928 §3).
    let auth = cfg.auth();
    let user = cfg.user_for(session);
    let greeting: &[u8] = if auth.is_some() {
        &[SOCKS_VERSION, 2, METHOD_NO_AUTH, METHOD_USERPASS]
    } else {
        &[SOCKS_VERSION, 1, METHOD_NO_AUTH]
    };
    write_step(&mut stream, &dl, timeouts, Step::Greeting, greeting)?;
    let mut reply = [0u8; 2];
    read_exact_step(&mut stream, &dl, timeouts, Step::Greeting, &mut reply)?;
    if reply[0] != SOCKS_VERSION {
        return Err(Socks5Error::Protocol("method reply has the wrong version"));
    }
    match reply[1] {
        METHOD_NO_AUTH => {}
        METHOD_USERPASS => {
            // Only reachable if offered, i.e. credentials exist; a proxy picking it anyway is `UnsupportedMethod`.
            let Some(auth) = auth else {
                return Err(Socks5Error::UnsupportedMethod(METHOD_USERPASS));
            };
            // 3. RFC 1929.
            let user = user.unwrap_or_default();
            let (user, pass) = (user.as_bytes(), auth.pass.expose().as_bytes());
            let mut msg = Vec::with_capacity(3 + user.len() + pass.len());
            msg.push(AUTH_VERSION);
            // Lengths are 1..=255, checked when the config is built.
            msg.push(u8::try_from(user.len()).map_err(|_| Socks5Error::Protocol("username too long"))?);
            msg.extend_from_slice(user);
            msg.push(u8::try_from(pass.len()).map_err(|_| Socks5Error::Protocol("password too long"))?);
            msg.extend_from_slice(pass);
            let sent = write_step(&mut stream, &dl, timeouts, Step::Auth, &msg);
            // Do not leave the credentials sitting in a buffer longer than needed.
            msg.iter_mut().for_each(|b| *b = 0);
            sent?;
            let mut status = [0u8; 2];
            read_exact_step(&mut stream, &dl, timeouts, Step::Auth, &mut status)?;
            // Only the status byte counts: some proxies answer with version 5 instead of 1 (curl ignores it too).
            if status[1] != 0 {
                return Err(Socks5Error::AuthFailed);
            }
        }
        METHOD_NONE_ACCEPTABLE => return Err(Socks5Error::NoAcceptableMethod),
        other => return Err(Socks5Error::UnsupportedMethod(other)),
    }

    // 4. UDP ASSOCIATE with an all-zero DST (we may not know our UDP address/port, RFC 1928 §7).
    let mut req = vec![SOCKS_VERSION, CMD_UDP_ASSOCIATE, 0];
    match proxy_ip {
        IpAddr::V4(_) => {
            req.push(ATYP_IPV4);
            req.extend_from_slice(&[0; 4 + 2]);
        }
        IpAddr::V6(_) => {
            req.push(ATYP_IPV6);
            req.extend_from_slice(&[0; 16 + 2]);
        }
    }
    write_step(&mut stream, &dl, timeouts, Step::Associate, &req)?;
    let mut head = [0u8; 4];
    read_exact_step(&mut stream, &dl, timeouts, Step::Associate, &mut head)?;
    if head[0] != SOCKS_VERSION {
        return Err(Socks5Error::Protocol("reply has the wrong version"));
    }
    match head[1] {
        REP_SUCCEEDED => {}
        REP_COMMAND_NOT_SUPPORTED => return Err(Socks5Error::UdpNotSupported),
        code => {
            return Err(Socks5Error::Refused {
                code,
                meaning: reply_meaning(code),
            });
        }
    }
    let (host, port) = read_bound_address(&mut stream, &dl, timeouts, head[3])?;
    if port == 0 {
        return Err(Socks5Error::RelayAddress("BND.PORT is 0"));
    }
    // `BND.ADDR` is the proxy's claim about itself and is not trusted (`judge_relay`).
    let announced = match host {
        BoundHost::Ip(ip) => Some(ip),
        // Rare, but legal. Never resolved: that would be a DNS query for a name the proxy chose, and the answer
        // could not be trusted either.
        BoundHost::Domain => None,
    };
    let (relay_ip, relay_host) = judge_relay(cfg, announced, proxy_ip)?;
    // From here on the control connection is only polled for closure.
    stream.set_nonblocking(true).map_err(|e| io_err(Step::Associate, e))?;
    Ok(Established {
        control: stream,
        relay: SocketAddr::new(relay_ip, port),
        relay_host,
        proxy_ip,
    })
}

/// Decides where datagrams go, from what the proxy announced (`announced`: `None` for a domain name) and the IP the
/// control connection reached (`proxy_ip`). Target-independent; the game server is checked later
/// ([`check_relay_is_not_target`]), once it is known.
///
/// - Unspecified or the proxy's own IP: the proxy's IP ([`RelayHost::SameAsProxy`]), the normal case.
/// - `relay = "proxy-host-only"`: anything else (another IP, a domain) is replaced by the proxy's IP, with a warning
///   ([`RelayHost::Substituted`]).
/// - `relay = "public"`: another IP is used as announced ([`RelayHost::Remote`]) only if it is a public unicast address;
///   a private, loopback, link-local, CGNAT, multicast, broadcast, documentation or reserved address, and any domain
///   name, is refused fatally, because a proxy that announces one cannot be told apart from one trying to make us send
///   to somewhere it chose.
fn judge_relay(
    cfg: &ProxyConfig,
    announced: Option<IpAddr>,
    proxy_ip: IpAddr,
) -> Result<(IpAddr, RelayHost), Socks5Error> {
    judge_relay_with(cfg, announced, proxy_ip, is_local_address)
}

/// [`judge_relay`] with the "is this one of this machine's own addresses" test passed in (tests).
fn judge_relay_with(
    cfg: &ProxyConfig,
    announced: Option<IpAddr>,
    proxy_ip: IpAddr,
    is_local: impl Fn(IpAddr) -> bool,
) -> Result<(IpAddr, RelayHost), Socks5Error> {
    if let Some(ip) = announced
        && (ip.is_unspecified() || ip.to_canonical() == proxy_ip.to_canonical())
    {
        return Ok((proxy_ip, RelayHost::SameAsProxy));
    }
    match cfg.relay_mode() {
        RelayMode::ProxyHostOnly => {
            tracing::warn!(
                "socks5: the proxy announced a relay address on another host; using the proxy's own address with the announced port"
            );
            Ok((proxy_ip, RelayHost::Substituted))
        }
        RelayMode::Public => {
            let Some(ip) = announced else {
                return Err(Socks5Error::RelayAddress(
                    "the relay is announced as a domain name, which is never resolved (relay = public needs a public unicast IP)",
                ));
            };
            let ip = ip.to_canonical();
            let class = classify(ip);
            // The test hook only ever applies to loopback (`ProxyConfig::loopback_relay_allowed` is false outside
            // tests and never set from a file).
            if class == RelayClass::Public {
                // Review F5 of 2.6b: not one of this machine's own (public) addresses either.
                if is_local(ip) {
                    return Err(Socks5Error::RelayAddress(
                        "the announced relay address is one of this machine's own addresses (relay = public needs another host)",
                    ));
                }
                Ok((ip, RelayHost::Remote))
            } else if cfg.loopback_relay_allowed() && ip.is_loopback() {
                Ok((ip, RelayHost::Remote))
            } else {
                Err(Socks5Error::RelayAddress(class.refusal()))
            }
        }
    }
}

/// `proxy-check`'s probe: 5 queries, each waiting at most 2 s (and never longer than one handshake step).
fn check_probe(timeouts: &Timeouts) -> ProbePlan {
    ProbePlan {
        queries: 5,
        per_query: timeouts.step.min(Duration::from_secs(2)),
    }
}

/// Session picking: fewer, shorter queries per candidate (it runs inside one connection attempt, up to
/// [`ProxyConfig::session_pick`] times): 3 queries, each waiting at most 600 ms.
fn pick_probe(timeouts: &Timeouts) -> ProbePlan {
    ProbePlan {
        queries: 3,
        per_query: timeouts.step.min(Duration::from_millis(600)),
    }
}

/// How long a whole session pick may take (review F1 of 2.6b): well under the driver's 15 s handshake watchdog, and
/// short enough that `systemctl stop` (30 s) is never kept waiting by it.
const PICK_BUDGET: Duration = Duration::from_secs(8);

/// The handshake bounds of one pick candidate: tighter than a normal handshake (3 s in all), because a healthy proxy
/// answers in tens of milliseconds and the whole pick has [`PICK_BUDGET`].
fn pick_timeouts(timeouts: &Timeouts) -> Timeouts {
    Timeouts {
        connect: timeouts.connect.min(Duration::from_secs(2)),
        step: timeouts.step.min(Duration::from_secs(2)),
        total: timeouts.total.min(Duration::from_secs(3)),
    }
}

/// A UDP socket of the relay's address family, bound to a free port.
fn bind_for(relay: SocketAddr) -> Result<UdpSocket, Socks5Error> {
    let bind: SocketAddr = match relay {
        SocketAddr::V4(_) => (Ipv4Addr::UNSPECIFIED, 0).into(),
        SocketAddr::V6(_) => (Ipv6Addr::UNSPECIFIED, 0).into(),
    };
    UdpSocket::bind(bind).map_err(|e| io_err(Step::Associate, e))
}

/// A fresh association that passed the acceptance rule against `target`, with its UDP socket, and (when `probe` is set)
/// the median UDP round-trip time through it to the neutral DNS target.
struct Candidate {
    est: Established,
    udp: UdpSocket,
    probe: Option<ProbeReport>,
}

/// `associate_session` + the target check + (optionally) the DNS probe. The target check comes **first**: the probe
/// is the first datagram this association ever carries, and it must not go to the game server.
fn establish(
    cfg: &ProxyConfig,
    session: Option<&str>,
    timeouts: &Timeouts,
    target: &RelayTarget,
    probe: Option<ProbePlan>,
) -> Result<Candidate, Socks5Error> {
    let est = associate_session(cfg, session, timeouts)?;
    check_relay_is_not_target(est.relay, est.proxy_ip, target)?;
    let udp = bind_for(est.relay)?;
    let probe = match probe {
        None => None,
        Some(plan) => {
            let result = probe_rtt(&udp, est.relay, DNS_PROBE_TARGET, plan).map_err(|e| io_err(Step::Associate, e))?;
            let median = result.median().ok_or(Socks5Error::ProbeFailed)?;
            Some(ProbeReport {
                sent: result.sent,
                replies: u16::try_from(result.rtts.len()).unwrap_or(u16::MAX),
                median,
            })
        }
    };
    Ok(Candidate { est, udp, probe })
}

/// Tries up to `cfg.session_pick()` sessions (one TCP connection each, a fresh token each), measures the UDP round-trip
/// time through each relay and keeps the lowest. Returns the winner, its token and the report.
///
/// - A fatal answer from the proxy (a wrong password, 0x07, a refused relay, ...) stops at once, so a wrong password is
///   tried once, not four times. A session that merely fails or gets no answer is skipped.
/// - The whole pick is bounded by [`PICK_BUDGET`] (8 s): each candidate has the tight [`pick_timeouts`], and a further
///   candidate is started only if the worst case of one candidate (its handshake bound plus its probe) still fits in what
///   is left. A mute relay therefore costs about two connections and a few seconds, never the watchdog.
fn pick_session(
    cfg: &ProxyConfig,
    timeouts: &Timeouts,
    target: &RelayTarget,
) -> Result<(Candidate, String, SessionReport), Socks5Error> {
    let n = usize::from(cfg.session_pick());
    let started = Instant::now();
    let t = pick_timeouts(timeouts);
    let plan = pick_probe(&t);
    let worst_candidate = t.total + plan.per_query * u32::from(plan.queries);
    let mut used: Vec<String> = Vec::new();
    let mut rtts: Vec<Option<Duration>> = Vec::new();
    let mut best: Option<(usize, Candidate, String)> = None;
    let mut last_err = Socks5Error::ProbeFailed;
    for i in 0..n {
        if i > 0 && started.elapsed() + worst_candidate > PICK_BUDGET {
            break;
        }
        let token = fresh_session_token(&used);
        used.push(token.clone());
        match establish(cfg, Some(&token), &t, target, Some(plan)) {
            Ok(c) => {
                let rtt = c.probe.map(|p| p.median);
                rtts.push(rtt);
                let better = match (&best, rtt) {
                    (None, Some(_)) => true,
                    (Some((_, b, _)), Some(r)) => b.probe.is_some_and(|bp| r < bp.median),
                    _ => false,
                };
                if better {
                    best = Some((i, c, token));
                }
            }
            Err(e) if e.is_fatal() => return Err(e),
            Err(e) => {
                rtts.push(None);
                last_err = e;
            }
        }
    }
    let Some((picked, cand, token)) = best else {
        return Err(last_err);
    };
    tracing::info!(
        tried = rtts.len(),
        picked = picked + 1,
        rtt_ms = ?rtts.iter().map(|r| r.map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))).collect::<Vec<_>>(),
        "socks5: session picking done (round-trip times only)"
    );
    Ok((cand, token, SessionReport { rtts, picked }))
}

/// The game server a proxy check measures against: the proxy file's own `for_server` (the one server it is issued for),
/// when it has one. `proxy-check` has no other idea of a server and never contacts one.
fn target_of(cfg: &ProxyConfig) -> RelayTarget {
    let mut t = RelayTarget::from_ips(cfg.for_server_ips());
    if let Some(port) = cfg.for_server_port() {
        t = t.with_port(port);
    }
    t
}

/// `ddnet-ai proxy-check`: the whole handshake (TCP connect, authentication, `UDP ASSOCIATE`, the relay address and the
/// acceptance rule) and, in `relay = "public"` mode (or with `session_pick`), a UDP reachability probe through the
/// relay: DNS queries for a neutral name to a public resolver ([`DNS_PROBE_TARGET`]) and their median round-trip time.
/// Never a datagram to a game server: the relay is checked against the proxy file's `for_server` first, and the probe
/// target is the resolver. In the default mode no UDP socket is created. The control connection is closed on return.
pub fn check(cfg: &ProxyConfig, timeouts: &Timeouts) -> Result<ProxyCheck, Socks5Error> {
    let target = target_of(cfg);
    let mode = cfg.relay_mode();
    let authenticated = cfg.auth().is_some();
    if cfg.session_pick() >= 2 {
        let (cand, _token, report) = pick_session(cfg, timeouts, &target)?;
        return Ok(ProxyCheck {
            relay_host: cand.est.relay_host,
            relay_port: cand.est.relay.port(),
            authenticated,
            mode,
            probe: cand.probe,
            sessions: Some(report),
        });
    }
    if mode == RelayMode::Public {
        let cand = establish(cfg, None, timeouts, &target, Some(check_probe(timeouts)))?;
        return Ok(ProxyCheck {
            relay_host: cand.est.relay_host,
            relay_port: cand.est.relay.port(),
            authenticated,
            mode,
            probe: cand.probe,
            sessions: None,
        });
    }
    let est = associate(cfg, timeouts)?;
    check_relay_is_not_target(est.relay, est.proxy_ip, &target)?;
    Ok(ProxyCheck {
        relay_host: est.relay_host,
        relay_port: est.relay.port(),
        authenticated,
        mode,
        probe: None,
        sessions: None,
    })
}

// ---------------------------------------------------------------------------------------------------
// The transport.
// ---------------------------------------------------------------------------------------------------

/// Counts of datagrams [`Socks5UdpTransport::recv`] dropped, by reason (since the transport was made).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DropStats {
    /// Not from the relay's address.
    pub foreign_source: u64,
    /// Header too short, reserved bits set, unknown `ATYP`, empty domain, empty or oversize payload.
    pub malformed: u64,
    /// `FRAG` not 0.
    pub fragmented: u64,
    /// A well-formed header naming some address other than the game server (or a domain).
    pub wrong_embedded_source: u64,
}

struct Association {
    control: TcpStream,
    udp: UdpSocket,
    relay: SocketAddr,
    proxy_ip: IpAddr,
}

/// A relay that is the game server itself would send our datagrams straight to it, around the proxy (the D-052
/// ban path). Refused fatally, whatever the proxy says:
///
/// - **a relay port equal to the server's port, whatever the IP.** Addresses are not compared literally: the same host
///   has several (IPv6 and IPv4, a second IPv4), and a proxy reached at one of them announcing the game port would make the
///   bot send to the game server's port on that host. Real relay ports are ephemeral; loopback tests use other ports;
/// - **a relay IP that is any of the server's IPs but is not the proxy's own** (IPv4-mapped spellings included). A relay
///   on the proxy's own IP is always the proxy's own socket, so it can be the server's IP only when the proxy runs on
///   the server's host (loopback tests), where the port rule above is what protects.
///
/// - **a relay on another host in the other address family than the server** (an IPv6 relay for an IPv4 server): it could
///   be the server's own host at its other address, which no list of the server's IPs can name.
///
/// (A relay port of 0 never gets here: `associate` refuses it.) The unit's cgroup filter is the second layer: in
/// `relay = "public"` mode it denies every game-server IP outright (`deploy/README.md`).
fn check_relay_is_not_target(relay: SocketAddr, proxy_ip: IpAddr, target: &RelayTarget) -> Result<(), Socks5Error> {
    let relay_is_proxy = relay.ip().to_canonical() == proxy_ip.to_canonical();
    if relay.port() == 0 || target.port() == Some(relay.port()) || (target.has_ip(relay.ip()) && !relay_is_proxy) {
        return Err(Socks5Error::RelayAddress("the relay address is the game server itself"));
    }
    // A relay on another host in the other address family than the server could be the server's own host at its other
    // address (review F3 of 2.6b): refused. The proxy's own IP is not judged (it is where we hold the control connection).
    if !relay_is_proxy && target.lacks_family_of(relay.ip()) {
        return Err(Socks5Error::RelayAddress(
            "the relay is in another address family than the game server (relay = public)",
        ));
    }
    Ok(())
}

/// A [`Transport`] through a SOCKS5 proxy's UDP relay. Created cheap and idle; the control connection is
/// opened by [`Transport::begin_attempt`], at most one per call.
pub struct Socks5UdpTransport {
    cfg: ProxyConfig,
    timeouts: Timeouts,
    poll: Duration,
    assoc: Option<Association>,
    target: Option<SocketAddr>,
    stats: DropStats,
    send_buf: Vec<u8>,
    recv_buf: Vec<u8>,
    /// How many associations this transport has kept (diagnostics and tests).
    associations_opened: u64,
    /// The session token session picking chose: every later association (after a loss) uses it again, so the exit stays
    /// the same and the proxy is not asked to pick again.
    session: Option<String>,
    /// A session pick has been tried (and failed without a fatal answer): never again, every later association is one
    /// plain connection per attempt with a fresh token (review F1 of 2.6b).
    pick_failed: bool,
}

impl std::fmt::Debug for Socks5UdpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Socks5UdpTransport")
            .field("proxy", &self.cfg)
            .field("associated", &self.assoc.is_some())
            .field("stats", &self.stats)
            .finish()
    }
}

impl Socks5UdpTransport {
    pub fn new(cfg: ProxyConfig, poll: Duration) -> Self {
        Self::with_timeouts(cfg, poll, Timeouts::default())
    }

    pub fn with_timeouts(cfg: ProxyConfig, poll: Duration, timeouts: Timeouts) -> Self {
        Socks5UdpTransport {
            cfg,
            timeouts,
            poll,
            assoc: None,
            target: None,
            stats: DropStats::default(),
            send_buf: Vec::with_capacity(1500),
            recv_buf: vec![0; RECV_BUF_SIZE],
            associations_opened: 0,
            session: None,
            pick_failed: false,
        }
    }

    pub fn stats(&self) -> DropStats {
        self.stats
    }

    pub fn associations_opened(&self) -> u64 {
        self.associations_opened
    }

    /// Whether an association is currently up (the control connection has not been seen closing).
    pub fn is_associated(&self) -> bool {
        self.assoc.is_some()
    }

    /// Opens an association that passed the acceptance rule against `target`. With `session_pick` and no session chosen
    /// yet, this is the pick (up to `session_pick` proxy connections, once per transport); otherwise exactly one.
    fn open(&mut self, target: &RelayTarget) -> Result<(), Socks5Error> {
        let cand = if self.cfg.session_pick() >= 2 && self.session.is_none() && !self.pick_failed {
            // Once, whatever the outcome: a pick that fails (a mute relay, a proxy that drops us) is not repeated at the
            // next attempt. This attempt is lost with the error; later ones are plain.
            match pick_session(&self.cfg, &self.timeouts, target) {
                Ok((cand, token, _report)) => {
                    self.session = Some(token);
                    cand
                }
                Err(e) => {
                    self.pick_failed = true;
                    return Err(e);
                }
            }
        } else {
            establish(&self.cfg, self.session.as_deref(), &self.timeouts, target, None)?
        };
        let Candidate { est, udp, .. } = cand;
        udp.set_read_timeout(Some(self.poll))
            .map_err(|e| io_err(Step::Associate, e))?;
        self.associations_opened += 1;
        tracing::info!(
            proxy = %self.cfg.name(),
            relay_host = ?est.relay_host,
            relay_port = est.relay.port(),
            "socks5: UDP association established"
        );
        self.assoc = Some(Association {
            control: est.control,
            udp,
            relay: est.relay,
            proxy_ip: est.proxy_ip,
        });
        Ok(())
    }

    /// Polls the control connection: `false` once it has closed or failed.
    fn control_alive(control: &mut TcpStream) -> bool {
        let mut scratch = [0u8; 64];
        // The proxy has nothing to say after the reply; bytes it sends anyway are discarded (bounded, so a
        // chatty proxy cannot keep this loop busy).
        for _ in 0..16 {
            match control.read(&mut scratch) {
                Ok(0) => return false,
                Ok(_) => continue,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return true,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => return false,
            }
        }
        true
    }

    fn lose_association(&mut self) -> io::Error {
        if self.assoc.take().is_some() {
            tracing::warn!(proxy = %self.cfg.name(), "socks5: the TCP control connection closed; the UDP association is gone");
        }
        io::Error::new(io::ErrorKind::ConnectionAborted, Socks5Error::ControlClosed)
    }
}

/// How long a transport that is going away keeps the control connection open after its last datagram.
const CLOSE_GRACE: Duration = Duration::from_millis(100);

impl Drop for Socks5UdpTransport {
    /// The association dies with the control connection (RFC 1928 §6), and a relay may discard datagrams it has
    /// not forwarded yet when that happens. The driver's last datagram is the graceful `NETMSG_CLOSE`, which
    /// frees our slot on the game server at once instead of after its silence timeout (D-016: one bot per
    /// server), so the connection is kept for a moment so the relay can pass it on. Runs on the driver thread as
    /// it ends; nothing waits for it.
    fn drop(&mut self) {
        if self.assoc.is_some() {
            std::thread::sleep(CLOSE_GRACE);
        }
    }
}

impl Transport for Socks5UdpTransport {
    fn begin_attempt(&mut self, target: SocketAddr) -> Result<(), TransportError> {
        if let Some(a) = self.assoc.as_mut()
            && !Self::control_alive(&mut a.control)
        {
            let _ = self.lose_association();
        }
        let reused = self.assoc.is_some();
        // Every address the game server is known by: the attempt's target and what the proxy file's `for_server`
        // resolves to.
        let mut known = RelayTarget::new(target);
        known.add_ips(self.cfg.for_server_ips());
        if reused {
            if let Some(a) = self.assoc.as_ref()
                && let Err(e) = check_relay_is_not_target(a.relay, a.proxy_ip, &known)
            {
                self.assoc = None;
                return Err(e.into());
            }
        } else {
            // A fresh association is checked inside `open`, before it carries a single datagram.
            self.open(&known)?;
        }
        if reused && let Some(a) = self.assoc.as_ref() {
            // Reusing the association (a server-requested reconnect or a redirect): discard what is queued
            // from the previous connection, like the direct path.
            let _ = a.udp.set_nonblocking(true);
            let mut drained = 0u32;
            while a.udp.recv_from(&mut self.recv_buf).is_ok() {
                drained += 1;
            }
            let _ = a.udp.set_nonblocking(false);
            let _ = a.udp.set_read_timeout(Some(self.poll));
            if drained > 0 {
                tracing::info!(
                    drained,
                    "driver: discarded stale datagrams from the previous connection"
                );
            }
        }
        self.target = Some(target);
        Ok(())
    }

    fn reset_after_loss(&mut self) {
        // The association cannot be vouched for after a loss (the relay may have died quietly while the TCP
        // connection looks fine): the next attempt opens a fresh one, which costs exactly that attempt's slot.
        self.assoc = None;
    }

    fn send(&mut self, datagram: &[u8]) -> io::Result<()> {
        let (Some(a), Some(target)) = (self.assoc.as_ref(), self.target) else {
            return Err(io::ErrorKind::NotConnected.into());
        };
        encode_udp_into(&mut self.send_buf, target, datagram);
        a.udp.send_to(&self.send_buf, a.relay).map(|_| ())
    }

    fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let (Some(a), Some(target)) = (self.assoc.as_mut(), self.target) else {
            return Err(io::ErrorKind::NotConnected.into());
        };
        if !Self::control_alive(&mut a.control) {
            return Err(self.lose_association());
        }
        let (n, from) = a.udp.recv_from(&mut self.recv_buf)?;
        let relay = a.relay;
        // A datagram that fills the buffer was possibly cut: never trust it.
        if n == self.recv_buf.len() {
            self.stats.malformed += 1;
            return Err(io::ErrorKind::WouldBlock.into());
        }
        if from.ip().to_canonical() != relay.ip().to_canonical() || from.port() != relay.port() {
            self.stats.foreign_source += 1;
            return Err(io::ErrorKind::WouldBlock.into());
        }
        let packet = match parse_udp(&self.recv_buf[..n]) {
            Ok(p) => p,
            Err(UdpDrop::Fragmented) => {
                self.stats.fragmented += 1;
                return Err(io::ErrorKind::WouldBlock.into());
            }
            Err(_) => {
                self.stats.malformed += 1;
                return Err(io::ErrorKind::WouldBlock.into());
            }
        };
        match packet.source {
            UdpSource::Ip(src)
                if src.ip().to_canonical() == target.ip().to_canonical() && src.port() == target.port() => {}
            _ => {
                self.stats.wrong_embedded_source += 1;
                return Err(io::ErrorKind::WouldBlock.into());
            }
        }
        // No DDNet packet is empty, and one longer than the caller's buffer would be truncated: neither
        // is a datagram the session can use.
        if packet.payload.is_empty() || packet.payload.len() > buf.len() {
            self.stats.malformed += 1;
            return Err(io::ErrorKind::WouldBlock.into());
        }
        buf[..packet.payload.len()].copy_from_slice(packet.payload);
        Ok(packet.payload.len())
    }

    fn set_nonblocking(&mut self, on: bool) -> io::Result<()> {
        match self.assoc.as_ref() {
            Some(a) => a.udp.set_nonblocking(on),
            None => Err(io::ErrorKind::NotConnected.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::RelayMode;
    use crate::socks5_testserver::{Auth, Bnd, Stall, TestSocks5Server};
    use proptest::prelude::*;

    fn v4(s: &str) -> SocketAddr {
        s.parse().unwrap()
    }

    // --- codec -----------------------------------------------------------------------------------

    #[test]
    fn encodes_the_rfc_1928_header_for_ipv4_and_ipv6() {
        let mut out = Vec::new();
        encode_udp_into(&mut out, v4("1.2.3.4:8303"), b"hi");
        assert_eq!(out, [0, 0, 0, 1, 1, 2, 3, 4, 0x20, 0x6f, b'h', b'i']);
        encode_udp_into(&mut out, "[::1]:80".parse().unwrap(), b"");
        let mut want = vec![0, 0, 0, 4];
        want.extend_from_slice(&Ipv6Addr::LOCALHOST.octets());
        want.extend_from_slice(&[0, 80]);
        assert_eq!(out, want);
    }

    #[test]
    fn parse_round_trips_what_encode_writes() {
        let mut out = Vec::new();
        for dst in [v4("9.8.7.6:65535"), "[2001:db8::1]:1".parse().unwrap()] {
            encode_udp_into(&mut out, dst, b"payload");
            let p = parse_udp(&out).unwrap();
            assert_eq!(p.source, UdpSource::Ip(dst));
            assert_eq!(p.payload, b"payload");
        }
    }

    #[test]
    fn parse_accepts_a_domain_source_but_never_names_it_an_ip() {
        let mut dg = vec![0, 0, 0, 3, 3, b'a', b'b', b'c', 0, 80];
        dg.extend_from_slice(b"xyz");
        let p = parse_udp(&dg).unwrap();
        assert_eq!(p.source, UdpSource::Domain);
        assert_eq!(p.payload, b"xyz");
    }

    #[test]
    fn parse_rejects_every_malformed_shape() {
        let cases: &[(&[u8], UdpDrop)] = &[
            (&[], UdpDrop::TooShort),
            (&[0, 0, 0], UdpDrop::TooShort),
            (&[0, 0, 0, 1, 1, 2, 3], UdpDrop::TooShort),
            (&[0, 0, 0, 1, 1, 2, 3, 4, 0], UdpDrop::TooShort),
            (&[0, 0, 0, 4, 1, 2, 3, 4, 0, 0], UdpDrop::TooShort),
            (&[0, 0, 0, 3], UdpDrop::TooShort),
            (&[0, 0, 0, 3, 5, b'a', b'b', 0, 0], UdpDrop::TooShort),
            (&[0, 0, 0, 3, 0, 0, 80], UdpDrop::BadDomain),
            (&[0, 0, 0, 2, 1, 2, 3, 4, 0, 80], UdpDrop::BadAddressType),
            (&[0, 0, 0, 0, 1, 2, 3, 4, 0, 80], UdpDrop::BadAddressType),
            (&[1, 0, 0, 1, 1, 2, 3, 4, 0, 80, 9], UdpDrop::Reserved),
            (&[0, 1, 0, 1, 1, 2, 3, 4, 0, 80, 9], UdpDrop::Reserved),
            (&[0, 0, 1, 1, 1, 2, 3, 4, 0, 80, 9], UdpDrop::Fragmented),
            (&[0, 0, 0x80, 1, 1, 2, 3, 4, 0, 80, 9], UdpDrop::Fragmented),
        ];
        for (bytes, want) in cases {
            assert_eq!(parse_udp(bytes), Err(*want), "{bytes:?}");
        }
        // Exactly header-sized: an empty payload parses (the transport drops it separately).
        assert_eq!(parse_udp(&[0, 0, 0, 1, 1, 2, 3, 4, 0, 80]).unwrap().payload, b"");
    }

    proptest! {
        /// The parser is total: arbitrary bytes never panic, and a success always borrows the tail of the input.
        #[test]
        fn parse_udp_never_panics_on_arbitrary_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..400)) {
            if let Ok(p) = parse_udp(&bytes) {
                prop_assert!(p.payload.len() <= bytes.len());
                prop_assert!(bytes.ends_with(p.payload));
            }
        }

        /// Same, biased toward plausible headers so the address-type branches are really exercised.
        #[test]
        fn parse_udp_never_panics_on_header_shaped_bytes(
            atyp in 0u8..8, frag in 0u8..3, len in 0u8..=255,
            tail in proptest::collection::vec(any::<u8>(), 0..300),
        ) {
            let mut bytes = vec![0, 0, frag, atyp, len];
            bytes.extend_from_slice(&tail);
            let _ = parse_udp(&bytes);
            let mut short = vec![0, 0, frag, atyp];
            short.extend_from_slice(&tail[..tail.len().min(5)]);
            let _ = parse_udp(&short);
        }

        #[test]
        fn encode_then_parse_is_the_identity(
            ip in any::<u32>(), port in any::<u16>(), payload in proptest::collection::vec(any::<u8>(), 0..200),
        ) {
            let dst = SocketAddr::new(IpAddr::V4(Ipv4Addr::from(ip)), port);
            let mut out = Vec::new();
            encode_udp_into(&mut out, dst, &payload);
            let p = parse_udp(&out).unwrap();
            prop_assert_eq!(p.source, UdpSource::Ip(dst));
            prop_assert_eq!(p.payload, &payload[..]);
        }
    }

    // --- error classification ----------------------------------------------------------------------

    #[test]
    fn only_unretryable_errors_are_fatal() {
        for e in [
            Socks5Error::UdpNotSupported,
            Socks5Error::AuthFailed,
            Socks5Error::NoAcceptableMethod,
            Socks5Error::UnsupportedMethod(9),
            Socks5Error::Protocol("x"),
            Socks5Error::RelayAddress("x"),
            Socks5Error::Refused { code: 2, meaning: "" },
            Socks5Error::Refused { code: 8, meaning: "" },
        ] {
            assert!(e.is_fatal(), "{e}");
        }
        for e in [
            Socks5Error::Io {
                step: Step::Connect,
                kind: io::ErrorKind::ConnectionRefused,
            },
            Socks5Error::Timeout { step: Step::Greeting },
            Socks5Error::Closed { step: Step::Auth },
            Socks5Error::ControlClosed,
            Socks5Error::Refused { code: 1, meaning: "" },
            Socks5Error::Refused { code: 5, meaning: "" },
        ] {
            assert!(!e.is_fatal(), "{e}");
        }
    }

    // --- handshake against the in-test server -----------------------------------------------------------

    fn cfg_for(server: &TestSocks5Server, auth: Option<(&str, &str)>) -> ProxyConfig {
        let a = server.addr();
        ProxyConfig::new(
            "t",
            a.ip().to_string(),
            a.port(),
            auth.map(|(u, p)| (u.to_string(), p.to_string())),
        )
        .unwrap()
    }

    fn fast() -> Timeouts {
        Timeouts {
            connect: Duration::from_millis(500),
            step: Duration::from_millis(400),
            total: Duration::from_millis(900),
        }
    }

    /// Task 4.10 (D-100): the control connection carries TCP keepalive with short idle/interval/retries, so an idle-TCP drop by the
    /// proxy provider is avoided and a silently dead connection is detected after about 105 s, never before the game's own 100 s silence timeout.
    #[test]
    fn the_control_connection_has_tcp_keepalive_with_short_timers() {
        let server = TestSocks5Server::start(Default::default());
        let est = associate(&cfg_for(&server, None), &fast()).unwrap();
        let sock = socket2::SockRef::from(&est.control);
        assert!(sock.keepalive().unwrap(), "SO_KEEPALIVE");
        assert_eq!(sock.tcp_keepalive_time().unwrap(), KEEPALIVE_IDLE);
        assert_eq!(sock.tcp_keepalive_interval().unwrap(), KEEPALIVE_INTERVAL);
        assert_eq!(sock.tcp_keepalive_retries().unwrap(), KEEPALIVE_RETRIES);
        let gives_up_after = KEEPALIVE_IDLE + KEEPALIVE_INTERVAL * KEEPALIVE_RETRIES;
        assert_eq!(gives_up_after, Duration::from_secs(105));
        assert!(
            gives_up_after >= ddai_net::conn::DEFAULT_TIMEOUT,
            "the keepalive must not give up before the game's silence timeout"
        );
    }

    #[test]
    fn associates_without_authentication() {
        let server = TestSocks5Server::start(Default::default());
        let est = associate(&cfg_for(&server, None), &fast()).unwrap();
        assert_eq!(est.relay, server.relay_addr());
        assert_eq!(est.relay_host, RelayHost::SameAsProxy);
    }

    #[test]
    fn associates_with_username_and_password_and_offers_both_methods_only_with_credentials() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            auth: Auth::UserPass("alice".into(), "wonderland".into()),
            ..Default::default()
        });
        associate(&cfg_for(&server, Some(("alice", "wonderland"))), &fast()).unwrap();
        assert_eq!(server.greetings().last().unwrap(), &vec![SOCKS_VERSION, 2, 0, 2]);
        // Without credentials only no-auth is offered, so a userpass-only proxy refuses.
        let err = associate(&cfg_for(&server, None), &fast()).unwrap_err();
        assert_eq!(err, Socks5Error::NoAcceptableMethod);
        assert_eq!(server.greetings().last().unwrap(), &vec![SOCKS_VERSION, 1, 0]);
        assert!(err.is_fatal());
    }

    #[test]
    fn a_wrong_password_is_a_fatal_auth_failure_and_credentials_are_not_in_the_error() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            auth: Auth::UserPass("alice".into(), "wonderland".into()),
            ..Default::default()
        });
        let err = associate(&cfg_for(&server, Some(("alice", "WRONG-pass-123"))), &fast()).unwrap_err();
        assert_eq!(err, Socks5Error::AuthFailed);
        assert!(err.is_fatal());
        let text = format!("{err} {err:?}");
        assert!(!text.contains("WRONG-pass-123") && !text.contains("alice"), "{text}");
    }

    #[test]
    fn reply_0x07_is_the_typed_udp_not_supported_error() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            reply_code: 0x07,
            ..Default::default()
        });
        let err = associate(&cfg_for(&server, None), &fast()).unwrap_err();
        assert_eq!(err, Socks5Error::UdpNotSupported);
        assert!(err.is_fatal());
        assert!(err.to_string().contains("UDP not supported"));
    }

    #[test]
    fn other_refusals_name_the_reply_and_only_some_are_fatal() {
        for (code, fatal) in [(0x01, false), (0x02, true), (0x05, false), (0x08, true)] {
            let server = TestSocks5Server::start(crate::socks5_testserver::Config {
                reply_code: code,
                ..Default::default()
            });
            let err = associate(&cfg_for(&server, None), &fast()).unwrap_err();
            assert!(
                matches!(err, Socks5Error::Refused { code: c, .. } if c == code),
                "{err:?}"
            );
            assert_eq!(err.is_fatal(), fatal, "{code:#x}");
        }
    }

    #[test]
    fn an_unspecified_bnd_addr_means_the_proxys_own_ip() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            bnd: Bnd::Unspecified,
            ..Default::default()
        });
        let est = associate(&cfg_for(&server, None), &fast()).unwrap();
        // The server really listens on 127.0.0.1 and announced 0.0.0.0:<port>.
        assert_eq!(est.relay, server.relay_addr());
        assert!(est.relay.ip().is_loopback());
        assert_eq!(est.relay_host, RelayHost::SameAsProxy);
    }

    #[test]
    fn a_domain_bnd_addr_is_never_resolved_and_means_the_proxys_own_ip() {
        // The name does not exist: resolving it would be a DNS query the proxy chose. It is not resolved at all.
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            bnd: Bnd::Domain("never-resolved.invalid".into()),
            ..Default::default()
        });
        let est = associate(&cfg_for(&server, None), &fast()).unwrap();
        assert_eq!(est.relay_host, RelayHost::Substituted);
        assert_eq!(est.relay, SocketAddr::new(est.proxy_ip, server.relay_addr().port()));
    }

    /// F1: `BND.ADDR` is the proxy's claim, not ours to obey: any IP but the proxy's own (or unspecified) is
    /// replaced by the proxy's IP with the announced port.
    #[test]
    fn a_bnd_addr_on_another_host_is_replaced_by_the_proxys_ip_keeping_the_port() {
        for announced in [
            "127.0.0.2:4000",
            "10.1.2.3:4000",
            "192.168.0.7:4000",
            "169.254.169.254:4000",
            "224.0.0.1:4000",
            "255.255.255.255:4000",
            "[::1]:4000",
            "[2001:db8::5]:4000",
        ] {
            let server = TestSocks5Server::start(crate::socks5_testserver::Config {
                bnd: Bnd::Fixed(announced.parse().unwrap()),
                ..Default::default()
            });
            let est = associate(&cfg_for(&server, None), &fast()).unwrap();
            assert_eq!(est.relay_host, RelayHost::Substituted, "{announced}");
            assert_eq!(
                est.relay,
                v4("127.0.0.1:4000"),
                "{announced}: the proxy's own IP, the announced port"
            );
            assert_eq!(est.proxy_ip, "127.0.0.1".parse::<IpAddr>().unwrap());
        }
        // The proxy's own IP announced explicitly is the normal case, used as announced.
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            bnd: Bnd::Fixed("127.0.0.1:4000".parse().unwrap()),
            ..Default::default()
        });
        let est = associate(&cfg_for(&server, None), &fast()).unwrap();
        assert_eq!(est.relay_host, RelayHost::SameAsProxy);
        assert_eq!(est.relay, v4("127.0.0.1:4000"));
        // Port 0 is no relay at all.
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            bnd: Bnd::Fixed("127.0.0.1:0".parse().unwrap()),
            ..Default::default()
        });
        assert!(matches!(
            associate(&cfg_for(&server, None), &fast()).unwrap_err(),
            Socks5Error::RelayAddress(_)
        ));
    }

    /// F1, the reviewer's scenario: the proxy announces the **game server's own address** as the relay. The bot's
    /// datagrams must not reach the game server from the bot's own socket.
    #[test]
    fn a_proxy_announcing_the_game_server_as_its_relay_cannot_make_the_bot_send_to_it_directly() {
        let game = UdpSocket::bind("127.0.0.3:0").unwrap();
        game.set_read_timeout(Some(Duration::from_millis(400))).unwrap();
        let game_addr = game.local_addr().unwrap();
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            // The game server's IP with another port: substituted by the proxy's IP, the port kept.
            bnd: Bnd::Fixed(SocketAddr::new(game_addr.ip(), game_addr.port().wrapping_add(1))),
            ..Default::default()
        });
        let mut t = transport_for(&server);
        t.begin_attempt(game_addr).unwrap();
        t.send(b"hello").unwrap();
        let mut buf = [0u8; 64];
        assert!(
            game.recv_from(&mut buf).is_err(),
            "the game server got a datagram straight from the bot"
        );
        // It went to the proxy's own IP (127.0.0.1) at the announced port instead, i.e. nowhere near the server.
        assert_eq!(
            t.assoc.as_ref().unwrap().relay,
            SocketAddr::new("127.0.0.1".parse().unwrap(), game_addr.port().wrapping_add(1))
        );
    }

    /// A substituted relay is a working relay: the datagrams really flow through the proxy.
    #[test]
    fn a_nat_style_announcement_still_relays_through_the_proxy() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            bnd: Bnd::OtherIp("10.9.8.7".parse().unwrap()),
            ..Default::default()
        });
        let game = GameDouble::new();
        let mut t = transport_for(&server);
        t.begin_attempt(game.addr()).unwrap();
        t.send(b"ping").unwrap();
        let (got, from) = game.recv();
        assert_eq!(got, b"ping");
        assert_eq!(from, server.relay_addr());
        game.socket.send_to(b"pong", from).unwrap();
        assert_eq!(recv_for(&mut t, Duration::from_secs(1)).unwrap(), b"pong");
    }

    #[test]
    fn a_relay_that_is_the_game_server_is_refused_fatally() {
        let ip = |s: &str| -> IpAddr { s.parse().unwrap() };
        // The relay IP equals the target's but is not the proxy's: refused whatever the port.
        for relay in ["10.0.0.9:4000", "10.0.0.9:8303"] {
            assert_eq!(
                check_relay_is_not_target(v4(relay), ip("127.0.0.1"), &RelayTarget::new(v4("10.0.0.9:8303"))),
                Err(Socks5Error::RelayAddress("the relay address is the game server itself"))
            );
        }
        // The relay is the game server's exact endpoint, even on the proxy's own host.
        assert!(
            check_relay_is_not_target(
                v4("127.0.0.1:8303"),
                ip("127.0.0.1"),
                &RelayTarget::new(v4("127.0.0.1:8303"))
            )
            .is_err()
        );
        // A proxy co-located with the game server (loopback tests) at another port, and any other host: fine.
        assert!(
            check_relay_is_not_target(
                v4("127.0.0.1:4000"),
                ip("127.0.0.1"),
                &RelayTarget::new(v4("127.0.0.1:8303"))
            )
            .is_ok()
        );
        assert!(
            check_relay_is_not_target(
                v4("203.0.113.5:4000"),
                ip("203.0.113.5"),
                &RelayTarget::new(v4("198.51.100.7:8303"))
            )
            .is_ok()
        );
        // F6: addresses are not compared literally. The proxy is reached at one address of the game server's host
        // (`127.0.0.1`), the game server is at another (`[::1]` or `127.0.0.3`), and `BND.PORT` is the game port:
        // the relay would be the game server's port on its own host. Refused whatever the IP.
        for target in ["[::1]:8303", "127.0.0.3:8303", "[::ffff:127.0.0.5]:8303"] {
            assert_eq!(
                check_relay_is_not_target(
                    v4("127.0.0.1:8303"),
                    ip("127.0.0.1"),
                    &RelayTarget::new(target.parse().unwrap())
                ),
                Err(Socks5Error::RelayAddress("the relay address is the game server itself")),
                "{target}"
            );
        }
        assert!(Socks5Error::RelayAddress("x").is_fatal());
        // Through the transport: a redirect (a reused association) to the relay's own endpoint is refused too,
        // and the association is dropped.
        let server = TestSocks5Server::start(Default::default());
        let mut t = transport_for(&server);
        t.begin_attempt(v4("127.0.0.1:9")).unwrap();
        let err = t.begin_attempt(server.relay_addr()).unwrap_err();
        assert!(err.is_fatal(), "{err:?}");
        assert!(matches!(err, TransportError::Proxy(Socks5Error::RelayAddress(_))));
        assert!(!t.is_associated());
    }

    /// F6, the reviewer's reproduction through the transport: the proxy is reached at `127.0.0.1`, the game server
    /// listens on another address of the same host (`127.0.0.3`, and `[::1]` where IPv6 loopback exists), and the
    /// proxy announces `BND.PORT` = the game port. Nothing may reach the game server from the bot's socket.
    #[test]
    fn a_relay_port_equal_to_the_game_port_on_another_address_of_its_host_is_refused() {
        let mut games = vec![UdpSocket::bind("127.0.0.3:0").unwrap()];
        if let Ok(g) = UdpSocket::bind("[::1]:0") {
            games.push(g);
        }
        for game in games {
            game.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
            let game_addr = game.local_addr().unwrap();
            let server = TestSocks5Server::start(crate::socks5_testserver::Config {
                bnd: Bnd::Fixed(SocketAddr::new("127.0.0.1".parse().unwrap(), game_addr.port())),
                ..Default::default()
            });
            let mut t = transport_for(&server);
            let err = t.begin_attempt(game_addr).unwrap_err();
            assert!(err.is_fatal(), "{game_addr}: {err:?}");
            assert!(
                matches!(err, TransportError::Proxy(Socks5Error::RelayAddress(_))),
                "{err:?}"
            );
            assert!(!t.is_associated());
            assert_eq!(t.send(b"x").unwrap_err().kind(), io::ErrorKind::NotConnected);
            let mut buf = [0u8; 16];
            assert!(
                game.recv_from(&mut buf).is_err(),
                "{game_addr}: the game server got a datagram"
            );
        }
    }

    /// F2: a proxy that drips its reply one byte at a time cannot stretch the handshake past its total bound.
    #[test]
    fn a_slowloris_proxy_cannot_stretch_the_handshake_past_the_total_bound() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            // A 20-byte domain: a reply of ~27 bytes at one byte per 300 ms is ~8 s if each read got a fresh timeout.
            bnd: Bnd::Domain("a".repeat(20)),
            drip: Some(Duration::from_millis(300)),
            ..Default::default()
        });
        let t0 = Instant::now();
        let err = associate(&cfg_for(&server, None), &fast()).unwrap_err();
        assert!(matches!(err, Socks5Error::Timeout { .. }), "{err:?}");
        assert!(
            t0.elapsed() < Duration::from_millis(1500),
            "the 0.9 s bound held for {:?}",
            t0.elapsed()
        );
    }

    /// F5: only the status byte of the RFC 1929 reply counts.
    #[test]
    fn the_auth_reply_version_byte_is_ignored_when_the_status_is_success() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            auth: Auth::UserPass("u".into(), "p".into()),
            auth_reply_version: Some(5),
            ..Default::default()
        });
        associate(&cfg_for(&server, Some(("u", "p"))), &fast()).unwrap();
        // A failure with a odd version byte is still a failure.
        let err = associate(&cfg_for(&server, Some(("u", "bad"))), &fast()).unwrap_err();
        assert_eq!(err, Socks5Error::AuthFailed);
    }

    #[test]
    fn garbage_replies_are_protocol_errors_not_panics() {
        for raw in [
            vec![0x04, 0x00],             // wrong version in the method reply
            vec![0x05],                   // truncated, then the server closes
            vec![0x05, 0x02, 0x02, 0x00], // userpass chosen but wrong auth reply version follows
        ] {
            let server = TestSocks5Server::start(crate::socks5_testserver::Config {
                raw_reply: Some(raw),
                ..Default::default()
            });
            let err = associate(&cfg_for(&server, Some(("u", "p"))), &fast()).unwrap_err();
            assert!(
                matches!(
                    err,
                    Socks5Error::Protocol(_) | Socks5Error::Closed { .. } | Socks5Error::Timeout { .. }
                ),
                "{err:?}"
            );
        }
    }

    #[test]
    fn every_stalled_step_times_out_inside_the_total_bound() {
        for stall in [
            Stall::AfterAccept,
            Stall::AfterGreeting,
            Stall::AfterAuth,
            Stall::AfterRequest,
        ] {
            let server = TestSocks5Server::start(crate::socks5_testserver::Config {
                auth: Auth::UserPass("u".into(), "p".into()),
                stall,
                ..Default::default()
            });
            let t0 = Instant::now();
            let err = associate(&cfg_for(&server, Some(("u", "p"))), &fast()).unwrap_err();
            assert!(matches!(err, Socks5Error::Timeout { .. }), "{stall:?}: {err:?}");
            assert!(!err.is_fatal());
            assert!(
                t0.elapsed() < Duration::from_millis(1500),
                "{stall:?}: {:?}",
                t0.elapsed()
            );
        }
    }

    #[test]
    fn a_dead_port_is_a_plain_connect_error() {
        // Bind and drop to find a port nobody listens on.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let cfg = ProxyConfig::new("t", "127.0.0.1", port, None).unwrap();
        let err = associate(&cfg, &fast()).unwrap_err();
        assert!(
            matches!(
                err,
                Socks5Error::Io {
                    step: Step::Connect,
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(!err.is_fatal());
    }

    #[test]
    fn check_reports_the_relay_without_creating_a_udp_socket_or_sending_datagrams() {
        let server = TestSocks5Server::start(Default::default());
        let report = check(&cfg_for(&server, None), &fast()).unwrap();
        assert_eq!(report.relay_port, server.relay_addr().port());
        assert_eq!(report.relay_host, RelayHost::SameAsProxy);
        assert!(!report.authenticated);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(server.datagrams_from_clients().len(), 0);
        assert_eq!(server.tcp_accepts(), 1);
    }

    // --- the transport ------------------------------------------------------------------------------

    /// A UDP socket standing in for the game server that echoes what it gets (uppercased first byte flip) and
    /// reports the datagram's source.
    struct GameDouble {
        socket: UdpSocket,
    }

    impl GameDouble {
        fn new() -> Self {
            let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
            socket.set_read_timeout(Some(Duration::from_millis(500))).unwrap();
            GameDouble { socket }
        }
        fn addr(&self) -> SocketAddr {
            self.socket.local_addr().unwrap()
        }
        fn recv(&self) -> (Vec<u8>, SocketAddr) {
            let mut b = [0u8; 2048];
            let (n, from) = self.socket.recv_from(&mut b).expect("game server got a datagram");
            (b[..n].to_vec(), from)
        }
    }

    fn transport_for(server: &TestSocks5Server) -> Socks5UdpTransport {
        Socks5UdpTransport::with_timeouts(cfg_for(server, None), Duration::from_millis(20), fast())
    }

    fn recv_for(t: &mut Socks5UdpTransport, within: Duration) -> Option<Vec<u8>> {
        let end = Instant::now() + within;
        let mut buf = [0u8; 2048];
        while Instant::now() < end {
            match t.recv(&mut buf) {
                Ok(n) => return Some(buf[..n].to_vec()),
                Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
                Err(e) => panic!("recv failed: {e}"),
            }
        }
        None
    }

    #[test]
    fn datagrams_make_the_round_trip_through_the_relay() {
        let server = TestSocks5Server::start(Default::default());
        let game = GameDouble::new();
        let mut t = transport_for(&server);
        t.begin_attempt(game.addr()).unwrap();
        t.send(b"hello").unwrap();
        let (got, from) = game.recv();
        assert_eq!(got, b"hello");
        // The game server sees the relay's outgoing socket, not our UDP socket.
        assert_eq!(from, server.relay_addr());
        game.socket.send_to(b"world", from).unwrap();
        assert_eq!(recv_for(&mut t, Duration::from_secs(1)).unwrap(), b"world");
        // What the relay received from us carried the RFC header for the game server.
        let (_, wire) = server.datagrams_from_clients().remove(0);
        assert_eq!(&wire[..4], &[0, 0, 0, 1]);
        assert_eq!(&wire[4..8], &[127, 0, 0, 1]);
        assert_eq!(u16::from_be_bytes([wire[8], wire[9]]), game.addr().port());
        assert_eq!(&wire[10..], b"hello");
        assert_eq!(t.stats(), DropStats::default());
    }

    #[test]
    fn works_with_username_password_and_with_a_wildcard_bnd() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            auth: Auth::UserPass("u".into(), "p".into()),
            bnd: Bnd::Unspecified,
            ..Default::default()
        });
        let game = GameDouble::new();
        let mut t =
            Socks5UdpTransport::with_timeouts(cfg_for(&server, Some(("u", "p"))), Duration::from_millis(20), fast());
        t.begin_attempt(game.addr()).unwrap();
        t.send(b"x").unwrap();
        let (_, from) = game.recv();
        game.socket.send_to(b"y", from).unwrap();
        assert_eq!(recv_for(&mut t, Duration::from_secs(1)).unwrap(), b"y");
    }

    #[test]
    fn foreign_fragmented_malformed_and_misattributed_datagrams_are_dropped_and_counted() {
        let server = TestSocks5Server::start(Default::default());
        let game = GameDouble::new();
        let mut t = transport_for(&server);
        t.begin_attempt(game.addr()).unwrap();
        t.send(b"open").unwrap();
        game.recv();
        let wrap = |dst: SocketAddr, payload: &[u8]| {
            let mut v = Vec::new();
            encode_udp_into(&mut v, dst, payload);
            v
        };
        // 1. From a socket that is not the relay, even with a perfect payload.
        server.inject_from_foreign_socket(&wrap(game.addr(), b"evil-foreign"));
        // 2. From the relay, fragmented.
        let mut frag = wrap(game.addr(), b"evil-frag");
        frag[2] = 1;
        server.inject_to_client(&frag);
        // 3. From the relay, malformed: truncated header, reserved bits, bad ATYP, empty, domain source.
        server.inject_to_client(&[0, 0, 0, 1, 127]);
        server.inject_to_client(&[0, 1, 0, 1, 127, 0, 0, 1, 0, 1, b'z']);
        server.inject_to_client(&[0, 0, 0, 9, 1, 2, 3]);
        server.inject_to_client(&[]);
        // 4. From the relay, well-formed but not from the game server (another port, another IP).
        let mut other_port = game.addr();
        other_port.set_port(other_port.port().wrapping_add(1));
        server.inject_to_client(&wrap(other_port, b"evil-port"));
        server.inject_to_client(&wrap(v4("203.0.113.5:8303"), b"evil-ip"));
        // 5. A domain source cannot be verified.
        server.inject_to_client(&[0, 0, 0, 3, 1, b'a', 0, 80, b'z']);
        // 6. Empty payload from the right source.
        server.inject_to_client(&wrap(game.addr(), b""));
        // Then one good datagram: it must be the only thing delivered.
        server.inject_to_client(&wrap(game.addr(), b"good"));
        let mut delivered = Vec::new();
        let end = Instant::now() + Duration::from_secs(1);
        while Instant::now() < end && delivered.is_empty() {
            if let Some(d) = recv_for(&mut t, Duration::from_millis(100)) {
                delivered.push(d);
            }
        }
        assert_eq!(delivered, vec![b"good".to_vec()]);
        assert!(recv_for(&mut t, Duration::from_millis(100)).is_none());
        let s = t.stats();
        assert_eq!(s.foreign_source, 1, "{s:?}");
        assert_eq!(s.fragmented, 1, "{s:?}");
        assert_eq!(s.wrong_embedded_source, 3, "{s:?}"); // other port, other ip, domain
        assert_eq!(s.malformed, 5, "{s:?}"); // short, reserved, bad atyp, empty datagram, empty payload
    }

    #[test]
    fn an_oversize_relayed_payload_is_dropped_not_truncated() {
        let server = TestSocks5Server::start(Default::default());
        let game = GameDouble::new();
        let mut t = transport_for(&server);
        t.begin_attempt(game.addr()).unwrap();
        t.send(b"open").unwrap();
        game.recv();
        let mut v = Vec::new();
        encode_udp_into(&mut v, game.addr(), &[7u8; 600]);
        server.inject_to_client(&v);
        let mut small = [0u8; 100];
        let end = Instant::now() + Duration::from_millis(300);
        while Instant::now() < end {
            if let Ok(n) = t.recv(&mut small) {
                panic!("delivered {n} bytes of an oversize datagram");
            }
        }
        assert_eq!(t.stats().malformed, 1);
    }

    #[test]
    fn dropping_the_control_connection_ends_the_association_with_a_typed_error() {
        let server = TestSocks5Server::start(Default::default());
        let game = GameDouble::new();
        let mut t = transport_for(&server);
        t.begin_attempt(game.addr()).unwrap();
        assert!(t.is_associated());
        server.drop_control_connections();
        let mut buf = [0u8; 64];
        let end = Instant::now() + Duration::from_secs(2);
        let err = loop {
            match t.recv(&mut buf) {
                Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
                Err(e) => break e,
                Ok(_) => panic!("unexpected datagram"),
            }
            assert!(Instant::now() < end, "the closed control connection was never noticed");
        };
        assert_eq!(err.kind(), io::ErrorKind::ConnectionAborted);
        let inner = err.get_ref().and_then(|e| e.downcast_ref::<Socks5Error>());
        assert_eq!(inner, Some(&Socks5Error::ControlClosed));
        assert!(!t.is_associated());
        // Sending without an association fails instead of using a dead relay.
        assert_eq!(t.send(b"x").unwrap_err().kind(), io::ErrorKind::NotConnected);
        // The next attempt opens exactly one new control connection.
        t.begin_attempt(game.addr()).unwrap();
        assert_eq!(t.associations_opened(), 2);
        assert_eq!(server.tcp_accepts(), 2);
    }

    #[test]
    fn an_association_is_reused_for_a_reconnect_but_not_after_a_loss() {
        let server = TestSocks5Server::start(Default::default());
        let game = GameDouble::new();
        let mut t = transport_for(&server);
        t.begin_attempt(game.addr()).unwrap();
        t.send(b"one").unwrap();
        let (_, first_from) = game.recv();
        // A server-requested reconnect: the same association (same relay socket, same source port at the server).
        t.begin_attempt(game.addr()).unwrap();
        t.send(b"two").unwrap();
        let (_, second_from) = game.recv();
        assert_eq!(first_from, second_from);
        assert_eq!(server.tcp_accepts(), 1);
        // A loss: a fresh association (and so one more TCP connection, one per attempt).
        t.reset_after_loss();
        assert!(!t.is_associated());
        t.begin_attempt(game.addr()).unwrap();
        assert_eq!(server.tcp_accepts(), 2);
        t.send(b"three").unwrap();
        let (_, third_from) = game.recv();
        assert_ne!(third_from, first_from);
    }

    #[test]
    fn a_redirect_keeps_the_association_and_changes_only_the_embedded_destination() {
        let server = TestSocks5Server::start(Default::default());
        let game_a = GameDouble::new();
        let game_b = GameDouble::new();
        let mut t = transport_for(&server);
        t.begin_attempt(game_a.addr()).unwrap();
        t.send(b"a").unwrap();
        game_a.recv();
        t.begin_attempt(game_b.addr()).unwrap();
        t.send(b"b").unwrap();
        assert_eq!(game_b.recv().0, b"b");
        assert_eq!(server.tcp_accepts(), 1);
        // A datagram that claims to be from the *old* server is now misattributed.
        let mut v = Vec::new();
        encode_udp_into(&mut v, game_a.addr(), b"stale");
        server.inject_to_client(&v);
        assert!(recv_for(&mut t, Duration::from_millis(200)).is_none());
        assert_eq!(t.stats().wrong_embedded_source, 1);
    }

    #[test]
    fn a_proxy_that_cannot_do_udp_leaves_the_transport_unassociated() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            reply_code: 0x07,
            ..Default::default()
        });
        let mut t = transport_for(&server);
        let err = t.begin_attempt(v4("127.0.0.1:8303")).unwrap_err();
        assert!(err.is_fatal());
        assert!(
            matches!(err, TransportError::Proxy(Socks5Error::UdpNotSupported)),
            "{err:?}"
        );
        assert!(!t.is_associated());
        assert_eq!(t.send(b"x").unwrap_err().kind(), io::ErrorKind::NotConnected);
    }

    #[test]
    fn debug_of_the_transport_hides_the_endpoint_and_credentials() {
        let server = TestSocks5Server::start(Default::default());
        let t = Socks5UdpTransport::new(
            cfg_for(&server, Some(("user-xyz", "pass-xyz"))),
            Duration::from_millis(2),
        );
        let text = format!("{t:?}");
        assert!(!text.contains("user-xyz") && !text.contains("pass-xyz"), "{text}");
        assert!(!text.contains(&server.addr().port().to_string()), "{text}");
    }

    // --- task 2.6b: a relay on another host (`relay = "public"`) ---------------------------------------------------

    fn public_cfg(server: &TestSocks5Server) -> ProxyConfig {
        cfg_for(server, None).with_relay(RelayMode::Public)
    }

    fn announcing(bnd: Bnd) -> TestSocks5Server {
        TestSocks5Server::start(crate::socks5_testserver::Config {
            bnd,
            ..Default::default()
        })
    }

    /// Every refused class, through the handshake: each is a fatal `RelayAddress` in public mode, and each is still
    /// merely substituted in the default mode (unchanged behaviour).
    #[test]
    fn public_mode_refuses_every_non_public_class_the_default_mode_substitutes() {
        for announced in [
            "0.1.2.3:4000",   // reserved "this network"
            "127.0.0.2:4000", // loopback
            "[::1]:4000",     // loopback v6
            "10.1.2.3:4000",  // private
            "172.16.5.5:4000",
            "192.168.0.7:4000",
            "[fd00::7]:4000",       // unique local
            "169.254.169.254:4000", // link-local
            "[fe80::1]:4000",
            "100.64.0.9:4000", // CGNAT
            "224.0.0.1:4000",  // multicast
            "[ff02::1]:4000",
            "255.255.255.255:4000", // broadcast
            "192.0.2.5:4000",       // documentation
            "198.51.100.5:4000",
            "203.0.113.5:4000",
            "[2001:db8::5]:4000",
            "240.0.0.1:4000", // reserved
            "198.18.0.1:4000",
            "[2002:7f00:3::1]:4000",  // 6to4 embedding 127.0.0.3
            "[64:ff9b::7f00:3]:4000", // NAT64 embedding 127.0.0.3
            "[::127.0.0.3]:4000",     // IPv4-compatible
            "[::ffff:10.0.0.1]:4000", // mapped private
        ] {
            let server = announcing(Bnd::Fixed(announced.parse().unwrap()));
            let err = associate(&public_cfg(&server), &fast()).unwrap_err();
            assert!(matches!(err, Socks5Error::RelayAddress(_)), "{announced}: {err:?}");
            assert!(err.is_fatal(), "{announced}");
            let text = err.to_string();
            assert!(text.contains("relay = public"), "{announced}: {text}");
            assert!(!text.contains("4000"), "no address or port in the error: {text}");
            // The default mode never obeys it.
            let est = associate(&cfg_for(&server, None), &fast()).unwrap();
            assert_eq!(est.relay_host, RelayHost::Substituted, "{announced}");
            assert_eq!(est.relay, v4("127.0.0.1:4000"), "{announced}");
        }
    }

    #[test]
    fn public_mode_never_resolves_a_domain_and_refuses_it() {
        let server = announcing(Bnd::Domain("never-resolved.invalid".into()));
        let err = associate(&public_cfg(&server), &fast()).unwrap_err();
        assert!(matches!(err, Socks5Error::RelayAddress(_)), "{err:?}");
        assert!(err.is_fatal());
        assert!(err.to_string().contains("domain"), "{err}");
        // Not even `localhost`.
        let server = announcing(Bnd::Domain("localhost".into()));
        assert!(matches!(
            associate(&public_cfg(&server), &fast()).unwrap_err(),
            Socks5Error::RelayAddress(_)
        ));
    }

    #[test]
    fn public_mode_accepts_a_public_address_on_another_host_and_the_normal_cases_stay_normal() {
        // No datagram is sent here: only the handshake.
        for (announced, want) in [
            ("8.8.8.8:4000", "8.8.8.8:4000"),
            ("104.171.172.27:51234", "104.171.172.27:51234"),
            ("[2606:4700:4700::1111]:4000", "[2606:4700:4700::1111]:4000"),
            // A mapped spelling is used as the IPv4 address it stands for.
            ("[::ffff:8.8.4.4]:4000", "8.8.4.4:4000"),
        ] {
            let server = announcing(Bnd::Fixed(announced.parse().unwrap()));
            let est = associate(&public_cfg(&server), &fast()).unwrap();
            assert_eq!(est.relay_host, RelayHost::Remote, "{announced}");
            assert_eq!(est.relay, want.parse::<SocketAddr>().unwrap(), "{announced}");
            assert_eq!(est.proxy_ip, "127.0.0.1".parse::<IpAddr>().unwrap());
        }
        // The proxy's own address, in any spelling, and the wildcard, are the proxy's host as before.
        for bnd in [
            Bnd::Relay,
            Bnd::Unspecified,
            Bnd::Fixed("[::ffff:127.0.0.1]:4000".parse().unwrap()),
        ] {
            let server = announcing(bnd);
            let est = associate(&public_cfg(&server), &fast()).unwrap();
            assert_eq!(est.relay_host, RelayHost::SameAsProxy);
            assert_eq!(est.relay.ip(), "127.0.0.1".parse::<IpAddr>().unwrap());
        }
        // Port 0 is no relay at all, in public mode too.
        let server = announcing(Bnd::Fixed("8.8.8.8:0".parse().unwrap()));
        assert!(matches!(
            associate(&public_cfg(&server), &fast()).unwrap_err(),
            Socks5Error::RelayAddress(_)
        ));
    }

    #[test]
    fn the_acceptance_rule_refuses_a_relay_that_is_any_of_the_servers_ips_or_its_port() {
        let proxy: IpAddr = "92.204.171.83".parse().unwrap();
        let ok = |relay: &str, target: &RelayTarget| check_relay_is_not_target(relay.parse().unwrap(), proxy, target);
        let refused = Err(Socks5Error::RelayAddress("the relay address is the game server itself"));
        let target = RelayTarget::new(v4("192.0.2.35:8308"));
        assert_eq!(ok("104.171.172.27:51234", &target), Ok(()));
        // The target's own IP, whatever the port; and the target's port, whatever the IP.
        assert_eq!(ok("192.0.2.35:51234", &target), refused);
        assert_eq!(ok("192.0.2.35:8308", &target), refused);
        assert_eq!(ok("104.171.172.27:8308", &target), refused);
        assert_eq!(ok("[2606:4700::1]:8308", &target), refused);
        // v4-mapped spellings of the target's IP.
        assert_eq!(ok("[::ffff:192.0.2.35]:51234", &target), refused);
        let mapped_target = RelayTarget::new("[::ffff:192.0.2.35]:8308".parse().unwrap());
        assert_eq!(ok("192.0.2.35:51234", &mapped_target), refused);
        // Every IP the target resolves to counts, not just the one in hand.
        let mut multi = RelayTarget::new(v4("192.0.2.35:8308"));
        multi.add_ips(["104.171.172.27".parse().unwrap(), "2001:4860::8888".parse().unwrap()]);
        assert_eq!(ok("104.171.172.27:51234", &multi), refused);
        assert_eq!(ok("[2001:4860::8888]:51234", &multi), refused);
        assert_eq!(ok("8.8.8.8:51234", &multi), Ok(()));
        // Port 0 is never a relay.
        assert_eq!(ok("8.8.8.8:0", &target), refused);
        // A known IP list without a port (`proxy-check` with an unresolvable `for_server` port) still guards the IPs.
        let ips_only = RelayTarget::from_ips(["192.0.2.35".parse().unwrap()]);
        assert_eq!(ok("192.0.2.35:1234", &ips_only), refused);
        assert_eq!(ok("8.8.8.8:8308", &ips_only), Ok(()));
    }

    /// The transport with a relay on another loopback alias (the test hook makes loopback count as public): the
    /// datagrams reach the game server from the relay's address, never from the bot's socket.
    #[test]
    fn a_relay_on_another_host_carries_the_session_in_public_mode() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            relay_ip: Some("127.0.0.2".parse().unwrap()),
            ..Default::default()
        });
        let game = GameDouble::new();
        // Without the hook, loopback is not public and the association is refused before any datagram.
        let mut strict = Socks5UdpTransport::with_timeouts(public_cfg(&server), Duration::from_millis(20), fast());
        let err = strict.begin_attempt(game.addr()).unwrap_err();
        assert!(
            matches!(err, TransportError::Proxy(Socks5Error::RelayAddress(_))),
            "{err:?}"
        );
        assert!(err.is_fatal());
        assert!(!strict.is_associated());
        assert!(
            game.socket.recv_from(&mut [0u8; 16]).is_err(),
            "nothing reached the game server"
        );

        let mut t = Socks5UdpTransport::with_timeouts(
            public_cfg(&server).with_test_loopback_relay(),
            Duration::from_millis(20),
            fast(),
        );
        t.begin_attempt(game.addr()).unwrap();
        let relay = t.assoc.as_ref().unwrap().relay;
        assert_eq!(relay, server.relay_addr());
        assert_eq!(relay.ip(), "127.0.0.2".parse::<IpAddr>().unwrap());
        t.send(b"hello").unwrap();
        let (got, from) = game.recv();
        assert_eq!(got, b"hello");
        assert_eq!(from, server.relay_addr(), "from the relay on the other host");
        game.socket.send_to(b"world", from).unwrap();
        assert_eq!(recv_for(&mut t, Duration::from_secs(1)).unwrap(), b"world");
        assert_eq!(t.stats(), DropStats::default());
    }

    /// A relay that is the game server's own IP is refused even though the class rule would let it through, and not one
    /// datagram reaches the server: the target's IPs are compared in every spelling and include `for_server`'s.
    #[test]
    fn a_remote_relay_on_the_game_servers_own_ip_is_refused_before_any_datagram() {
        let game = UdpSocket::bind("127.0.0.2:0").unwrap();
        game.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
        let game_addr = game.local_addr().unwrap();
        let hooked = |server: &TestSocks5Server| public_cfg(server).with_test_loopback_relay();
        // (a) The relay announced is the game server's IP at another port; (b) the target written v4-mapped.
        for (target, label) in [
            (game_addr, "plain"),
            (
                SocketAddr::new("::ffff:127.0.0.2".parse().unwrap(), game_addr.port()),
                "mapped",
            ),
        ] {
            let server = announcing(Bnd::Fixed(SocketAddr::new(game_addr.ip(), 5000)));
            let mut t = Socks5UdpTransport::with_timeouts(hooked(&server), Duration::from_millis(20), fast());
            let err = t.begin_attempt(target).unwrap_err();
            assert!(
                matches!(err, TransportError::Proxy(Socks5Error::RelayAddress(_))),
                "{label}: {err:?}"
            );
            assert!(err.is_fatal(), "{label}");
            assert!(!t.is_associated(), "{label}");
            assert_eq!(t.send(b"x").unwrap_err().kind(), io::ErrorKind::NotConnected);
        }
        // (c) Another IP of the same server, known only from the proxy file's `for_server`.
        let server = announcing(Bnd::Fixed("127.0.0.2:5000".parse().unwrap()));
        let mut t = Socks5UdpTransport::with_timeouts(
            hooked(&server).with_for_server("127.0.0.2:9"),
            Duration::from_millis(20),
            fast(),
        );
        let err = t.begin_attempt(v4("127.0.0.1:8303")).unwrap_err();
        assert!(
            matches!(err, TransportError::Proxy(Socks5Error::RelayAddress(_))),
            "{err:?}"
        );
        // (d) The relay's port is the game server's port, on a different IP.
        let server = announcing(Bnd::Fixed(SocketAddr::new(
            "127.0.0.3".parse().unwrap(),
            game_addr.port(),
        )));
        let mut t = Socks5UdpTransport::with_timeouts(hooked(&server), Duration::from_millis(20), fast());
        let err = t.begin_attempt(game_addr).unwrap_err();
        assert!(
            matches!(err, TransportError::Proxy(Socks5Error::RelayAddress(_))),
            "{err:?}"
        );
        assert!(
            game.recv_from(&mut [0u8; 16]).is_err(),
            "the game server got a datagram"
        );
    }

    /// A redirect reuses the association, so the acceptance rule is applied again to the NEW target: a redirect onto the
    /// IP the (remote) relay sits on drops the association and sends nothing.
    #[test]
    fn a_redirect_onto_the_relays_own_address_is_refused_and_drops_a_remote_association() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            relay_ip: Some("127.0.0.2".parse().unwrap()),
            ..Default::default()
        });
        let game = GameDouble::new();
        let mut t = Socks5UdpTransport::with_timeouts(
            public_cfg(&server).with_test_loopback_relay(),
            Duration::from_millis(20),
            fast(),
        );
        t.begin_attempt(game.addr()).unwrap();
        t.send(b"first").unwrap();
        game.recv();
        // The new target is another port on the relay's own IP: refused (the IP is the server's and not the proxy's).
        let victim = UdpSocket::bind("127.0.0.2:0").unwrap();
        victim.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
        let err = t.begin_attempt(victim.local_addr().unwrap()).unwrap_err();
        assert!(
            matches!(err, TransportError::Proxy(Socks5Error::RelayAddress(_))),
            "{err:?}"
        );
        assert!(!t.is_associated());
        assert_eq!(t.send(b"x").unwrap_err().kind(), io::ErrorKind::NotConnected);
        assert!(
            victim.recv_from(&mut [0u8; 16]).is_err(),
            "nothing reached the redirect target"
        );
    }

    /// Review F3 of 2.6b: an IPv6 relay for an IPv4-only target (and the reverse) could be the server's own host at its other
    /// address. Refused for a relay on another host; the proxy's own IP is never judged.
    #[test]
    fn a_remote_relay_in_the_other_address_family_than_the_server_is_refused() {
        let proxy: IpAddr = "92.204.171.83".parse().unwrap();
        let refused = Err(Socks5Error::RelayAddress(
            "the relay is in another address family than the game server (relay = public)",
        ));
        let v4_server = RelayTarget::new(v4("192.0.2.35:8308"));
        let check = |relay: &str, t: &RelayTarget| check_relay_is_not_target(relay.parse().unwrap(), proxy, t);
        assert_eq!(check("[2606:4700::1]:51234", &v4_server), refused);
        assert_eq!(check("104.171.172.27:51234", &v4_server), Ok(()));
        assert_eq!(
            check("[::ffff:104.171.172.27]:51234", &v4_server),
            Ok(()),
            "mapped is IPv4"
        );
        let v6_server = RelayTarget::new("[2001:4860::8888]:8308".parse().unwrap());
        assert_eq!(check("104.171.172.27:51234", &v6_server), refused);
        assert_eq!(check("[2606:4700::1]:51234", &v6_server), Ok(()));
        // The proxy's own IP is where we hold the control connection: not judged.
        assert_eq!(
            check_relay_is_not_target("92.204.171.83:51234".parse().unwrap(), proxy, &v6_server),
            Ok(())
        );
        // Through the transport (loopback stands in for public with the hook): `[::1]` for an IPv4 game server.
        let game = UdpSocket::bind("127.0.0.3:0").unwrap();
        game.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
        let server = announcing(Bnd::Fixed("[::1]:5000".parse().unwrap()));
        let mut t = Socks5UdpTransport::with_timeouts(
            public_cfg(&server).with_test_loopback_relay(),
            Duration::from_millis(20),
            fast(),
        );
        let err = t.begin_attempt(game.local_addr().unwrap()).unwrap_err();
        assert!(
            matches!(err, TransportError::Proxy(Socks5Error::RelayAddress(_))) && err.is_fatal(),
            "{err:?}"
        );
        assert!(!t.is_associated());
        assert!(game.recv_from(&mut [0u8; 16]).is_err());
    }

    /// Review F5 of 2.6b: an announced relay IP that is one of this machine's own addresses is refused.
    #[test]
    fn public_mode_refuses_an_announced_address_that_is_one_of_this_machines_own() {
        let cfg = ProxyConfig::new("t", "h", 1, None)
            .unwrap()
            .with_relay(RelayMode::Public);
        let proxy: IpAddr = "92.204.171.83".parse().unwrap();
        let announced: IpAddr = "104.171.172.27".parse().unwrap();
        // A remote public address that is not ours passes; the same address when it is ours does not.
        assert_eq!(
            judge_relay_with(&cfg, Some(announced), proxy, |_| false),
            Ok((announced, RelayHost::Remote))
        );
        assert!(matches!(
            judge_relay_with(&cfg, Some(announced), proxy, |ip| ip == announced),
            Err(Socks5Error::RelayAddress(t)) if t.contains("own addresses")
        ));
        // The real test is a bind: a documentation-range address is refused earlier by class, a public one we do not own is not local.
        assert!(!is_local_address(announced));
        // The proxy's own IP (SameAsProxy) is not subject to it.
        assert_eq!(
            judge_relay_with(&cfg, Some(proxy), proxy, |_| true),
            Ok((proxy, RelayHost::SameAsProxy))
        );
    }

    // --- task 2.6b: the UDP probe and `proxy-check` ---------------------------------------------------------------

    fn server_with_dns() -> TestSocks5Server {
        TestSocks5Server::start(crate::socks5_testserver::Config {
            fake_dns: true,
            ..Default::default()
        })
    }

    #[test]
    fn check_in_public_mode_probes_through_the_relay_and_reports_the_median_rtt() {
        let server = server_with_dns();
        let report = check(&public_cfg(&server), &fast()).unwrap();
        assert_eq!(report.mode, RelayMode::Public);
        assert_eq!(report.relay_host, RelayHost::SameAsProxy);
        let probe = report.probe.expect("public mode probes");
        assert_eq!((probe.sent, probe.replies), (5, 5));
        assert!(probe.median < Duration::from_millis(300), "{probe:?}");
        assert!(report.sessions.is_none());
        // What the relay saw: five DNS queries to port 53 and nothing else; the game server was never named.
        let wire = server.datagrams_from_clients();
        assert_eq!(wire.len(), 5);
        for (_, dg) in &wire {
            let p = parse_udp(dg).unwrap();
            let UdpSource::Ip(dst) = p.source else {
                panic!("no domain expected")
            };
            assert_eq!(dst, DNS_PROBE_TARGET);
        }
        assert_eq!(server.tcp_accepts(), 1);
    }

    #[test]
    fn check_reports_a_relay_on_another_host_and_its_rtt() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            relay_ip: Some("127.0.0.2".parse().unwrap()),
            fake_dns: true,
            relay_delays: vec![Duration::from_millis(20)],
            ..Default::default()
        });
        let report = check(&public_cfg(&server).with_test_loopback_relay(), &fast()).unwrap();
        assert_eq!(report.relay_host, RelayHost::Remote);
        let probe = report.probe.unwrap();
        assert_eq!(probe.replies, 5);
        assert!(
            probe.median >= Duration::from_millis(20),
            "the relay's delay shows: {probe:?}"
        );
    }

    #[test]
    fn check_without_the_hook_refuses_a_loopback_relay_and_never_probes() {
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            relay_ip: Some("127.0.0.2".parse().unwrap()),
            fake_dns: true,
            ..Default::default()
        });
        let err = check(&public_cfg(&server), &fast()).unwrap_err();
        assert!(matches!(err, Socks5Error::RelayAddress(_)), "{err:?}");
        std::thread::sleep(Duration::from_millis(100));
        assert!(server.datagrams_from_clients().is_empty(), "no datagram went anywhere");
    }

    #[test]
    fn check_refuses_a_relay_that_is_the_file_s_game_server_before_probing() {
        // The proxy file's `for_server` stands in for the game server: the relay at 127.0.0.2 is its IP.
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            relay_ip: Some("127.0.0.2".parse().unwrap()),
            fake_dns: true,
            ..Default::default()
        });
        let cfg = public_cfg(&server)
            .with_test_loopback_relay()
            .with_for_server("127.0.0.2:8303");
        let err = check(&cfg, &fast()).unwrap_err();
        assert!(matches!(err, Socks5Error::RelayAddress(_)), "{err:?}");
        std::thread::sleep(Duration::from_millis(100));
        assert!(server.datagrams_from_clients().is_empty());
        // And the relay's port equal to `for_server`'s port, on another IP, is refused the same way.
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            bnd: Bnd::Fixed("127.0.0.2:4000".parse().unwrap()),
            fake_dns: true,
            ..Default::default()
        });
        let cfg = public_cfg(&server)
            .with_test_loopback_relay()
            .with_for_server("127.0.0.9:4000");
        assert!(matches!(
            check(&cfg, &fast()).unwrap_err(),
            Socks5Error::RelayAddress(_)
        ));
        assert!(server.datagrams_from_clients().is_empty());
    }

    #[test]
    fn a_relay_that_answers_no_probe_query_fails_the_check_without_being_fatal() {
        // No fake DNS: the relay drops datagrams to anywhere but loopback, so nothing answers.
        let server = TestSocks5Server::start(Default::default());
        let t0 = Instant::now();
        let err = check(&public_cfg(&server), &fast()).unwrap_err();
        assert_eq!(err, Socks5Error::ProbeFailed);
        assert!(!err.is_fatal());
        assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
        // The default mode does not probe at all.
        let report = check(&cfg_for(&server, None), &fast()).unwrap();
        assert_eq!(report.mode, RelayMode::ProxyHostOnly);
        assert!(report.probe.is_none());
    }

    // --- task 2.6b: session picking -------------------------------------------------------------------------------

    fn session_server(delays: &[u64]) -> TestSocks5Server {
        TestSocks5Server::start(crate::socks5_testserver::Config {
            auth: Auth::UserPassSession("sess-".into(), "pw".into()),
            fake_dns: true,
            relay_delays: delays.iter().map(|d| Duration::from_millis(*d)).collect(),
            ..Default::default()
        })
    }

    fn session_cfg(server: &TestSocks5Server, pick: u8) -> ProxyConfig {
        let a = server.addr();
        ProxyConfig::new(
            "t",
            a.ip().to_string(),
            a.port(),
            Some(("sess-{session}".to_string(), "pw".to_string())),
        )
        .unwrap()
        .with_session_pick(pick)
        .unwrap()
    }

    #[test]
    fn session_picking_keeps_the_session_with_the_lowest_rtt_and_reuses_it_after_a_loss() {
        // Relay n gets delays[n]: the second session is the fast one.
        let server = session_server(&[150, 10, 250, 90]);
        let game = GameDouble::new();
        let mut t = Socks5UdpTransport::with_timeouts(session_cfg(&server, 4), Duration::from_millis(20), fast());
        t.begin_attempt(game.addr()).unwrap();
        // Four proxy connections, no more, each with its own fresh token in the user name; none is the placeholder.
        assert_eq!(server.tcp_accepts(), 4);
        let users = server.users_seen();
        assert_eq!(users.len(), 4);
        for u in &users {
            assert!(u.starts_with("sess-") && !u.contains('{'), "{u}");
            assert_eq!(u.len(), "sess-".len() + 8, "{u}");
        }
        let distinct: std::collections::HashSet<_> = users.iter().collect();
        assert_eq!(distinct.len(), 4, "a fresh token per session");
        // The relay kept is the fast one's.
        assert_eq!(t.assoc.as_ref().unwrap().relay, server.relay_addrs()[1]);
        assert_eq!(t.associations_opened(), 1);
        // And it carries the game.
        t.send(b"ping").unwrap();
        let (got, from) = game.recv();
        assert_eq!(got, b"ping");
        assert_eq!(from, server.relay_addrs()[1]);

        // A loss: one new proxy connection, with the SAME session token (same exit), and no new pick.
        t.reset_after_loss();
        t.begin_attempt(game.addr()).unwrap();
        assert_eq!(server.tcp_accepts(), 5);
        let users = server.users_seen();
        assert_eq!(users[4], users[1], "the winner's session is reused");
        assert_eq!(t.associations_opened(), 2);
    }

    /// Review F1 of 2.6b: a mute relay costs about two proxy connections and well under the 15 s watchdog, and the failed
    /// pick is remembered: the next attempt is one plain connection with a fresh token, not another pick.
    #[test]
    fn a_failed_pick_is_remembered_and_bounded_by_the_budget() {
        // No fake DNS: the relay answers nothing, so every candidate fails its probe.
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            auth: Auth::UserPassSession("sess-".into(), "pw".into()),
            ..Default::default()
        });
        let mut t =
            Socks5UdpTransport::with_timeouts(session_cfg(&server, 4), Duration::from_millis(20), Timeouts::default());
        let t0 = Instant::now();
        let err = t.begin_attempt(v4("127.0.0.1:9")).unwrap_err();
        let took = t0.elapsed();
        assert!(!err.is_fatal(), "{err:?}");
        assert!(took < Duration::from_secs(8), "{took:?}");
        assert!(
            server.tcp_accepts() <= 2,
            "{} proxy connections for a mute relay",
            server.tcp_accepts()
        );
        assert!(!t.is_associated());
        // The next attempt: no pick, exactly one connection, a fresh token, no probe.
        let before = server.tcp_accepts();
        t.begin_attempt(v4("127.0.0.1:9")).unwrap();
        assert_eq!(server.tcp_accepts(), before + 1);
        let users = server.users_seen();
        assert!(
            users.last().unwrap().starts_with("sess-") && !users.last().unwrap().contains('{'),
            "{users:?}"
        );
        assert_eq!(
            users.iter().collect::<std::collections::HashSet<_>>().len(),
            users.len(),
            "fresh tokens"
        );
        assert!(
            server.datagrams_from_clients().len() <= 6,
            "only probe datagrams, none for the plain connection"
        );
        // And a third attempt after a loss: still one connection, never a pick.
        t.reset_after_loss();
        t.begin_attempt(v4("127.0.0.1:9")).unwrap();
        assert_eq!(server.tcp_accepts(), before + 2);
    }

    #[test]
    fn a_session_that_fails_to_answer_is_skipped_not_fatal() {
        // The first relay answers after 700 ms, longer than the 400 ms probe wait of `fast()`: no answer at all.
        let server = session_server(&[700, 0]);
        let (cand, _token, report) =
            pick_session(&session_cfg(&server, 2), &fast(), &RelayTarget::new(v4("127.0.0.1:9"))).unwrap();
        assert_eq!(report.rtts.len(), 2);
        assert_eq!(report.rtts[0], None);
        assert!(report.rtts[1].is_some());
        assert_eq!(report.picked, 1);
        assert_eq!(cand.est.relay, server.relay_addrs()[1]);
    }

    #[test]
    fn a_wrong_password_stops_the_pick_after_one_proxy_connection() {
        let server = session_server(&[0]);
        let a = server.addr();
        let cfg = ProxyConfig::new(
            "t",
            a.ip().to_string(),
            a.port(),
            Some(("sess-{session}".to_string(), "WRONG".to_string())),
        )
        .unwrap()
        .with_session_pick(4)
        .unwrap();
        let mut t = Socks5UdpTransport::with_timeouts(cfg, Duration::from_millis(20), fast());
        let err = t.begin_attempt(v4("127.0.0.1:9")).unwrap_err();
        assert!(matches!(err, TransportError::Proxy(Socks5Error::AuthFailed)), "{err:?}");
        assert!(err.is_fatal());
        assert_eq!(
            server.tcp_accepts(),
            1,
            "a wrong password is tried once, not four times"
        );
    }

    #[test]
    fn the_pick_report_has_round_trip_times_and_no_user_names() {
        let server = session_server(&[120, 0, 240]);
        let report = check(&session_cfg(&server, 3), &fast()).unwrap();
        let sessions = report.sessions.as_ref().unwrap();
        assert_eq!(sessions.rtts.len(), 3);
        assert_eq!(sessions.picked, 1);
        assert!(report.probe.is_some());
        let text = format!("{report:?}");
        for u in server.users_seen() {
            assert!(!text.contains(&u), "{text}");
        }
        assert_eq!(server.tcp_accepts(), 3);
    }

    #[test]
    fn the_user_name_template_reaches_the_wire_filled_and_a_plain_user_is_unchanged() {
        // `associate` fills a placeholder with a fresh token even when called directly.
        let server = session_server(&[0]);
        associate(&session_cfg(&server, 2), &fast()).unwrap();
        let sent = server.users_seen();
        assert!(
            sent[0].starts_with("sess-") && !sent[0].contains("{session}"),
            "{sent:?}"
        );
        // An explicit token is used verbatim.
        associate_session(&session_cfg(&server, 2), Some("abc12345"), &fast()).unwrap();
        assert_eq!(server.users_seen()[1], "sess-abc12345");
        // No placeholder: the user name is exactly what the file says.
        let server = TestSocks5Server::start(crate::socks5_testserver::Config {
            auth: Auth::UserPass("alice".into(), "wonderland".into()),
            ..Default::default()
        });
        associate(&cfg_for(&server, Some(("alice", "wonderland"))), &fast()).unwrap();
        assert_eq!(server.users_seen(), vec!["alice".to_string()]);
    }
}
