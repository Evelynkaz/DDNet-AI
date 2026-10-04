//! A UDP reachability probe through a SOCKS5 relay (task 2.6b): a DNS query for a neutral name sent to a public DNS
//! resolver, through the relay, and the round-trip time of the answer. It is what `proxy-check` prints as the relay's
//! median RTT in `relay = "public"` mode, and what the opt-in session picking (`session_pick`) compares.
//!
//! The target is a fixed neutral one ([`DNS_PROBE_TARGET`], `1.1.1.1:53`), never a game server: the probe is sent only
//! after the relay address passed the acceptance rule (`socks5::check_relay_is_not_target`), so it cannot reach a game
//! server from the VPS's own IP either.

use crate::socks5::{UdpSource, encode_udp_into, parse_udp};
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

/// The neutral probe target: a public DNS resolver. Not a game server.
pub const DNS_PROBE_TARGET: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1)), 53);

/// How many queries and how long to wait for each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbePlan {
    pub queries: u16,
    pub per_query: Duration,
}

/// What a probe measured: the replies' round-trip times, in the order the queries went out (lost ones are missing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeResult {
    pub sent: u16,
    pub rtts: Vec<Duration>,
}

impl ProbeResult {
    /// The median RTT, or `None` when no query was answered. For an even count, the mean of the middle two.
    pub fn median(&self) -> Option<Duration> {
        median(&self.rtts)
    }
}

/// The median of `values` (the mean of the middle two when even), `None` when empty.
pub fn median(values: &[Duration]) -> Option<Duration> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort();
    let mid = v.len() / 2;
    Some(if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2
    })
}

/// A DNS query for `example.com` type `A`, recursion desired, with transaction id `id`.
fn dns_query(id: u16) -> Vec<u8> {
    let mut q = Vec::with_capacity(29);
    q.extend_from_slice(&id.to_be_bytes());
    q.extend_from_slice(&[0x01, 0x00]); // RD
    q.extend_from_slice(&[0, 1, 0, 0, 0, 0, 0, 0]); // QDCOUNT 1, the rest 0
    q.extend_from_slice(b"\x07example\x03com\x00");
    q.extend_from_slice(&[0, 1, 0, 1]); // A, IN
    q
}

/// Whether `payload` is a DNS response to the query with transaction id `id`.
fn is_dns_answer(payload: &[u8], id: u16) -> bool {
    payload.len() >= 12 && payload[..2] == id.to_be_bytes() && payload[2] & 0x80 != 0
}

/// Sends `plan.queries` DNS queries to `dns` through the relay at `relay`, one at a time, from `udp` (a socket of the
/// relay's address family; its read timeout is changed). Datagrams that are not an answer to the query in flight, or do
/// not come from the relay, are ignored. The caller must have checked `relay` against the game server **before**
/// calling this.
pub fn probe_rtt(udp: &UdpSocket, relay: SocketAddr, dns: SocketAddr, plan: ProbePlan) -> io::Result<ProbeResult> {
    let mut out = ProbeResult {
        sent: 0,
        rtts: Vec::new(),
    };
    let mut wire = Vec::with_capacity(64);
    let mut buf = [0u8; 1024];
    // A per-process base so a late answer to an earlier probe is not mistaken for this one's.
    let base = u16::try_from(std::process::id() & 0xffff).unwrap_or(1);
    for i in 0..plan.queries {
        let id = base.wrapping_add(i);
        encode_udp_into(&mut wire, dns, &dns_query(id));
        let start = Instant::now();
        udp.send_to(&wire, relay)?;
        out.sent += 1;
        let deadline = start + plan.per_query;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            udp.set_read_timeout(Some(left))?;
            match udp.recv_from(&mut buf) {
                Ok((n, from)) => {
                    if from.ip().to_canonical() != relay.ip().to_canonical() || from.port() != relay.port() {
                        continue;
                    }
                    let Ok(packet) = parse_udp(&buf[..n]) else { continue };
                    let UdpSource::Ip(src) = packet.source else { continue };
                    if src.ip().to_canonical() == dns.ip().to_canonical()
                        && src.port() == dns.port()
                        && is_dns_answer(packet.payload, id)
                    {
                        out.rtts.push(start.elapsed());
                        break;
                    }
                }
                Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => break,
                Err(e) => return Err(e),
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn the_median_of_odd_even_and_empty_lists() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[ms(5)]), Some(ms(5)));
        assert_eq!(median(&[ms(30), ms(10), ms(20)]), Some(ms(20)));
        assert_eq!(median(&[ms(40), ms(10), ms(20), ms(30)]), Some(ms(25)));
        assert_eq!(median(&[ms(7), ms(7), ms(900)]), Some(ms(7)));
    }

    #[test]
    fn the_query_is_a_well_formed_dns_question() {
        let q = dns_query(0xabcd);
        assert_eq!(&q[..2], &[0xab, 0xcd]);
        assert_eq!(q[2] & 0x80, 0, "a query, not a response");
        assert_eq!(&q[4..6], &[0, 1], "one question");
        assert_eq!(&q[12..], b"\x07example\x03com\x00\x00\x01\x00\x01");
    }

    #[test]
    fn only_a_response_with_the_matching_id_is_an_answer() {
        let mut resp = dns_query(7);
        assert!(!is_dns_answer(&resp, 7), "a query is not an answer");
        resp[2] |= 0x80;
        assert!(is_dns_answer(&resp, 7));
        assert!(!is_dns_answer(&resp, 8));
        assert!(!is_dns_answer(&resp[..11], 7));
        assert!(!is_dns_answer(&[], 7));
    }
}
