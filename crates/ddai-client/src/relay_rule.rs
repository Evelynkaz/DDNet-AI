//! The acceptance rule for a UDP relay on another host than the SOCKS5 proxy (task 2.6b, D-088 amendment): pure
//! functions over addresses, no I/O.
//!
//! A proxy's `UDP ASSOCIATE` reply announces `BND.ADDR:BND.PORT`. With `relay = "proxy-host-only"` (the default) that
//! address is never obeyed: datagrams go to the proxy's own IP. With `relay = "public"` an address on another host is
//! obeyed **only if it is a public unicast address** ([`classify`]) and, once the game server is known, **not any of the
//! server's own IPs** and with a port that is neither 0 nor the server's ([`RelayTarget`], checked by
//! `socks5::check_relay_is_not_target`). The point: a malicious or misconfigured proxy must never make the bot send game
//! UDP to the game server from the VPS's own IP (D-052: the server bans it). The unit's cgroup filter is the second,
//! system-level layer (`deploy/README.md`).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

/// What kind of address an announced relay is. Only [`RelayClass::Public`] is accepted for a relay on another host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelayClass {
    /// A public unicast address: the only class a relay on another host may have.
    Public,
    /// `0.0.0.0` or `::`.
    Unspecified,
    /// `127.0.0.0/8`, `::1`.
    Loopback,
    /// RFC 1918 (`10/8`, `172.16/12`, `192.168/16`) and unique-local IPv6 (`fc00::/7`).
    Private,
    /// `169.254/16`, `fe80::/10`.
    LinkLocal,
    /// Carrier-grade NAT, `100.64/10`.
    Cgnat,
    /// `224/4`, `ff00::/8`.
    Multicast,
    /// `255.255.255.255`.
    Broadcast,
    /// The documentation ranges (RFC 5737, RFC 3849, RFC 9637).
    Documentation,
    /// Everything else that is not a global unicast address: `0/8`, `240/4`, protocol assignments, benchmarking,
    /// 6to4 and Teredo and NAT64 (they embed an IPv4 address, which could be the game server's), site-local,
    /// IPv4-compatible, discard, and every IPv6 prefix outside `2000::/3`.
    Reserved,
}

impl RelayClass {
    /// The refusal text for this class (no address in it: a proxy's address is as secret as its credentials).
    pub fn refusal(self) -> &'static str {
        match self {
            RelayClass::Public => "the announced relay address is public",
            RelayClass::Unspecified => {
                "the announced relay address is unspecified (relay = public needs a public unicast address)"
            }
            RelayClass::Loopback => {
                "the announced relay address is a loopback address (relay = public needs a public unicast address)"
            }
            RelayClass::Private => {
                "the announced relay address is a private address (relay = public needs a public unicast address)"
            }
            RelayClass::LinkLocal => {
                "the announced relay address is a link-local address (relay = public needs a public unicast address)"
            }
            RelayClass::Cgnat => {
                "the announced relay address is in the carrier-grade NAT range (relay = public needs a public unicast address)"
            }
            RelayClass::Multicast => {
                "the announced relay address is a multicast address (relay = public needs a public unicast address)"
            }
            RelayClass::Broadcast => {
                "the announced relay address is the broadcast address (relay = public needs a public unicast address)"
            }
            RelayClass::Documentation => {
                "the announced relay address is in a documentation range (relay = public needs a public unicast address)"
            }
            RelayClass::Reserved => {
                "the announced relay address is in a reserved range (relay = public needs a public unicast address)"
            }
        }
    }
}

/// Classifies `ip`. An IPv4-mapped IPv6 address (`::ffff:a.b.c.d`) is classified as the IPv4 address it stands for.
pub fn classify(ip: IpAddr) -> RelayClass {
    match ip.to_canonical() {
        IpAddr::V4(v4) => classify_v4(v4),
        IpAddr::V6(v6) => classify_v6(v6),
    }
}

/// Whether `ip` is a public unicast address ([`RelayClass::Public`]).
pub fn is_public_unicast(ip: IpAddr) -> bool {
    classify(ip) == RelayClass::Public
}

fn classify_v4(ip: Ipv4Addr) -> RelayClass {
    let [a, b, c, _] = ip.octets();
    if ip.is_unspecified() {
        RelayClass::Unspecified
    } else if ip.is_broadcast() {
        RelayClass::Broadcast
    } else if a == 127 {
        RelayClass::Loopback
    } else if a == 10 || (a == 172 && (16..=31).contains(&b)) || (a == 192 && b == 168) {
        RelayClass::Private
    } else if a == 169 && b == 254 {
        RelayClass::LinkLocal
    } else if a == 100 && (64..=127).contains(&b) {
        RelayClass::Cgnat
    } else if (224..=239).contains(&a) {
        RelayClass::Multicast
    } else if (a, b, c) == (192, 0, 2) || (a, b, c) == (198, 51, 100) || (a, b, c) == (203, 0, 113) {
        RelayClass::Documentation
    } else if a == 0
        || a >= 240
        || (a, b, c) == (192, 0, 0)
        || (a == 198 && (18..=19).contains(&b))
        || (a, b, c) == (192, 88, 99)
    {
        RelayClass::Reserved
    } else {
        RelayClass::Public
    }
}

fn classify_v6(ip: Ipv6Addr) -> RelayClass {
    let s = ip.segments();
    if ip.is_unspecified() {
        RelayClass::Unspecified
    } else if ip.is_loopback() {
        RelayClass::Loopback
    } else if s[0] >> 8 == 0xff {
        RelayClass::Multicast
    } else if s[0] & 0xffc0 == 0xfe80 {
        RelayClass::LinkLocal
    } else if s[0] & 0xfe00 == 0xfc00 {
        RelayClass::Private
    } else if (s[0], s[1]) == (0x2001, 0x0db8) || (s[0] == 0x3fff && s[1] & 0xf000 == 0) {
        // 2001:db8::/32 (RFC 3849) and 3fff::/20 (RFC 9637).
        RelayClass::Documentation
    } else if s[0] & 0xe000 != 0x2000 {
        // Not in 2000::/3, the global unicast block: site-local, IPv4-compatible, NAT64, discard, and the rest.
        RelayClass::Reserved
    } else if (s[0] == 0x2001 && s[1] < 0x0200) || s[0] == 0x2002 {
        // 2001::/23 (protocol assignments: Teredo, benchmarking, ORCHID) and 2002::/16 (6to4): both embed or
        // tunnel an IPv4 address.
        RelayClass::Reserved
    } else {
        RelayClass::Public
    }
}

/// The game server a relay must not be: every IP it is known by, and its port. The IPs are compared in canonical form,
/// so an IPv4-mapped IPv6 spelling of the server matches its IPv4 address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelayTarget {
    ips: Vec<IpAddr>,
    /// `None` only for `proxy-check`, which has no game server in hand (it uses the proxy file's `for_server` IPs when
    /// the file has one, and then the port too).
    port: Option<u16>,
}

impl RelayTarget {
    /// One server endpoint.
    pub fn new(target: SocketAddr) -> Self {
        RelayTarget {
            ips: vec![target.ip().to_canonical()],
            port: Some(target.port()),
        }
    }

    /// No known server: only the IP list (possibly empty) and no port.
    pub fn from_ips(ips: impl IntoIterator<Item = IpAddr>) -> Self {
        let mut t = RelayTarget {
            ips: Vec::new(),
            port: None,
        };
        t.add_ips(ips);
        t
    }

    /// Adds more IPs the server is known by (a host name that resolves to several addresses).
    pub fn add_ips(&mut self, ips: impl IntoIterator<Item = IpAddr>) {
        for ip in ips {
            let ip = ip.to_canonical();
            if !self.ips.contains(&ip) {
                self.ips.push(ip);
            }
        }
    }

    /// Sets the port the relay must not have.
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Whether `ip` is one of the server's IPs (canonical form compared).
    pub fn has_ip(&self, ip: IpAddr) -> bool {
        self.ips.contains(&ip.to_canonical())
    }

    pub fn port(&self) -> Option<u16> {
        self.port
    }

    /// Whether the server has known IPs and none is of `ip`'s address family (canonical form: a mapped IPv4 address is
    /// IPv4). A relay in the other family could be another address of the server's own host, which no list of the server's
    /// IPs can name (2.6b review F3).
    pub fn lacks_family_of(&self, ip: IpAddr) -> bool {
        let v4 = ip.to_canonical().is_ipv4();
        !self.ips.is_empty() && !self.ips.iter().any(|t| t.is_ipv4() == v4)
    }
}

/// Whether `ip` is an address of this machine (a bind to it succeeds). A proxy that announces the VPS's own public address
/// as its relay would make the bot send to itself; nothing listens there, but the rule refuses it anyway.
pub fn is_local_address(ip: IpAddr) -> bool {
    std::net::UdpSocket::bind((ip, 0)).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn every_refused_class_is_named() {
        let cases: &[(&str, RelayClass)] = &[
            ("0.0.0.0", RelayClass::Unspecified),
            ("::", RelayClass::Unspecified),
            ("127.0.0.1", RelayClass::Loopback),
            ("127.255.255.254", RelayClass::Loopback),
            ("::1", RelayClass::Loopback),
            ("10.0.0.1", RelayClass::Private),
            ("10.255.255.255", RelayClass::Private),
            ("172.16.0.1", RelayClass::Private),
            ("172.31.255.255", RelayClass::Private),
            ("192.168.1.1", RelayClass::Private),
            ("fc00::1", RelayClass::Private),
            ("fd12:3456::1", RelayClass::Private),
            ("169.254.169.254", RelayClass::LinkLocal),
            ("fe80::1", RelayClass::LinkLocal),
            ("febf::1", RelayClass::LinkLocal),
            ("100.64.0.1", RelayClass::Cgnat),
            ("100.127.255.255", RelayClass::Cgnat),
            ("224.0.0.1", RelayClass::Multicast),
            ("239.255.255.255", RelayClass::Multicast),
            ("ff02::1", RelayClass::Multicast),
            ("255.255.255.255", RelayClass::Broadcast),
            ("192.0.2.1", RelayClass::Documentation),
            ("198.51.100.7", RelayClass::Documentation),
            ("203.0.113.5", RelayClass::Documentation),
            ("2001:db8::1", RelayClass::Documentation),
            ("2001:db8:ffff::1", RelayClass::Documentation),
            ("3fff::1", RelayClass::Documentation),
            ("3fff:0fff::1", RelayClass::Documentation),
            ("0.1.2.3", RelayClass::Reserved),
            ("240.0.0.1", RelayClass::Reserved),
            ("254.1.1.1", RelayClass::Reserved),
            ("192.0.0.8", RelayClass::Reserved),
            ("198.18.0.1", RelayClass::Reserved),
            ("198.19.255.255", RelayClass::Reserved),
            ("192.88.99.1", RelayClass::Reserved),
            ("fec0::1", RelayClass::Reserved),
            ("::127.0.0.3", RelayClass::Reserved),
            ("64:ff9b::7f00:3", RelayClass::Reserved),
            ("100::1", RelayClass::Reserved),
            ("2002:7f00:3::1", RelayClass::Reserved),
            ("2001::1", RelayClass::Reserved),
            ("2001:1ff::1", RelayClass::Reserved),
            ("4000::1", RelayClass::Reserved),
            ("e000::1", RelayClass::Reserved),
        ];
        for (text, want) in cases {
            assert_eq!(classify(ip(text)), *want, "{text}");
            assert!(!is_public_unicast(ip(text)), "{text}");
            assert!(!want.refusal().is_empty());
        }
    }

    #[test]
    fn this_machines_own_addresses_are_local_and_a_documentation_address_is_not() {
        assert!(is_local_address(ip("127.0.0.1")) && is_local_address(ip("127.0.0.9")));
        assert!(!is_local_address(ip("203.0.113.9")) && !is_local_address(ip("8.8.8.8")));
        assert!(!is_local_address(ip("2606:4700:4700::1111")));
    }

    #[test]
    fn public_unicast_addresses_pass() {
        for text in [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.27",
            "93.184.216.83",
            "100.63.255.255",
            "100.128.0.1",
            "172.15.255.255",
            "172.32.0.1",
            "192.0.1.1",
            "192.169.0.1",
            "198.17.255.255",
            "198.20.0.1",
            "223.255.255.255",
            "169.253.1.1",
            "2606:4700:4700::1111",
            "2620:fe::fe",
            "2001:200::1",
            "2001:4860:4860::8888",
            "3fff:1000::1",
        ] {
            assert_eq!(classify(ip(text)), RelayClass::Public, "{text}");
            assert!(is_public_unicast(ip(text)), "{text}");
        }
    }

    #[test]
    fn a_mapped_address_is_classified_as_the_ipv4_address_it_stands_for() {
        assert_eq!(classify(ip("::ffff:127.0.0.1")), RelayClass::Loopback);
        assert_eq!(classify(ip("::ffff:10.1.2.3")), RelayClass::Private);
        assert_eq!(classify(ip("::ffff:192.0.2.9")), RelayClass::Documentation);
        assert_eq!(classify(ip("::ffff:0.0.0.0")), RelayClass::Unspecified);
        assert_eq!(classify(ip("::ffff:8.8.8.8")), RelayClass::Public);
    }

    #[test]
    fn the_target_matches_every_spelling_of_each_of_its_ips() {
        let mut t = RelayTarget::new("[::ffff:192.0.2.35]:8308".parse().unwrap());
        assert!(t.has_ip(ip("192.0.2.35")));
        assert!(t.has_ip(ip("::ffff:192.0.2.35")));
        assert!(!t.has_ip(ip("192.0.2.36")));
        assert_eq!(t.port(), Some(8308));
        t.add_ips([ip("2001:db8::7"), ip("192.0.2.35"), ip("::ffff:9.9.9.9")]);
        assert!(t.has_ip(ip("2001:db8::7")));
        assert!(t.has_ip(ip("9.9.9.9")));
        assert_eq!(t.ips.len(), 3, "no duplicates");
        // The family of the relay against the family of the server's IPs.
        let v4_only = RelayTarget::new("192.0.2.35:8308".parse().unwrap());
        assert!(v4_only.lacks_family_of(ip("2606:4700::1")));
        assert!(!v4_only.lacks_family_of(ip("8.8.8.8")));
        assert!(!v4_only.lacks_family_of(ip("::ffff:8.8.8.8")), "mapped is IPv4");
        let v6_only = RelayTarget::from_ips([ip("2001:db8::7")]);
        assert!(v6_only.lacks_family_of(ip("8.8.8.8")) && !v6_only.lacks_family_of(ip("2606:4700::1")));
        t.add_ips([ip("2001:db8::7")]);
        assert!(
            !t.lacks_family_of(ip("2606:4700::1")) && !t.lacks_family_of(ip("8.8.8.8")),
            "both families known"
        );
        assert!(
            !RelayTarget::from_ips([]).lacks_family_of(ip("8.8.8.8")),
            "nothing known: nothing to compare"
        );
        let bare = RelayTarget::from_ips([ip("1.2.3.4")]);
        assert_eq!(bare.port(), None);
        assert_eq!(bare.with_port(9).port(), Some(9));
    }
}
