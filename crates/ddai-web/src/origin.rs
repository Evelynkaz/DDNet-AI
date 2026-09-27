//! Same-origin enforcement for state-changing requests and the WebSocket upgrade (acceptance
//! criterion 4), and client-IP extraction that only trusts `X-Forwarded-For` from a local,
//! explicitly configured reverse proxy (acceptance criterion 3).

use std::net::IpAddr;

/// Modern browsers send `Sec-Fetch-Site` on every request (including WebSocket upgrades); when
/// present it is authoritative and we don't need to parse `Origin` at all. When it's absent (an
/// older browser, or a non-browser HTTP client such as our own tests/tools), we fall back to
/// comparing `Origin`'s host to the request's `Host` header. If *both* are absent we reject: a
/// same-origin browser request always sends at least one of them for non-GET/non-simple
/// requests, so "neither header present" is not a case we need to allow for legitimate traffic.
pub fn is_same_origin(host_header: Option<&str>, origin_header: Option<&str>, sec_fetch_site: Option<&str>) -> bool {
    if let Some(site) = sec_fetch_site {
        return site.eq_ignore_ascii_case("same-origin") || site.eq_ignore_ascii_case("none");
    }
    match (host_header, origin_header) {
        (Some(host), Some(origin)) => {
            origin_authority(origin).is_some_and(|authority| authority.eq_ignore_ascii_case(host))
        }
        _ => false,
    }
}

/// Extracts the `host[:port]` authority from an `Origin` header value like
/// `"https://example.com"` or `"http://127.0.0.1:7788"`. `Origin` never carries a path or
/// userinfo, so a plain `"://"` split is sufficient (no need for a full URL parser here).
fn origin_authority(origin: &str) -> Option<&str> {
    origin.split_once("://").map(|(_, authority)| authority)
}

/// Resolves the client IP for rate limiting and session audit records. Trusts
/// `X-Forwarded-For`'s first entry only when `trust_proxy` is set AND the TCP peer itself is
/// loopback (i.e. only Caddy, running on the same host, could have made this connection) —
/// acceptance criterion 3. Any other case uses the real TCP peer address, so a client cannot
/// spoof its rate-limit identity by sending its own `X-Forwarded-For`.
pub fn client_ip(peer_ip: IpAddr, trust_proxy: bool, forwarded_for: Option<&str>) -> IpAddr {
    if trust_proxy
        && peer_ip.is_loopback()
        && let Some(candidate) = forwarded_for.and_then(first_forwarded_ip)
    {
        return candidate;
    }
    peer_ip
}

fn first_forwarded_ip(header_value: &str) -> Option<IpAddr> {
    header_value.split(',').next()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    #[test]
    fn sec_fetch_site_same_origin_is_allowed() {
        assert!(is_same_origin(Some("h"), None, Some("same-origin")));
    }

    #[test]
    fn sec_fetch_site_none_is_allowed() {
        assert!(is_same_origin(Some("h"), None, Some("none")));
    }

    #[test]
    fn sec_fetch_site_cross_site_is_rejected_even_with_matching_origin() {
        assert!(!is_same_origin(
            Some("127.0.0.1:7788"),
            Some("http://127.0.0.1:7788"),
            Some("cross-site")
        ));
    }

    #[test]
    fn sec_fetch_site_same_site_is_rejected() {
        // "same-site" (a sibling subdomain) is deliberately NOT treated as same-origin.
        assert!(!is_same_origin(Some("h"), None, Some("same-site")));
    }

    #[test]
    fn matching_origin_and_host_is_allowed() {
        assert!(is_same_origin(
            Some("127.0.0.1:7788"),
            Some("http://127.0.0.1:7788"),
            None
        ));
        assert!(is_same_origin(Some("example.com"), Some("https://example.com"), None));
    }

    #[test]
    fn mismatched_origin_is_rejected() {
        assert!(!is_same_origin(
            Some("127.0.0.1:7788"),
            Some("http://evil.example:1234"),
            None
        ));
    }

    #[test]
    fn origin_host_comparison_is_case_insensitive() {
        assert!(is_same_origin(Some("EXAMPLE.com"), Some("http://example.COM"), None));
    }

    #[test]
    fn missing_both_headers_is_rejected() {
        assert!(!is_same_origin(Some("127.0.0.1:7788"), None, None));
    }

    #[test]
    fn missing_origin_with_host_present_is_rejected() {
        assert!(!is_same_origin(Some("127.0.0.1:7788"), None, None));
    }

    #[test]
    fn malformed_origin_is_rejected() {
        assert!(!is_same_origin(Some("127.0.0.1:7788"), Some("not-a-url"), None));
    }

    fn v4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(a, b, c, d))
    }

    #[test]
    fn untrusted_proxy_flag_ignores_xff() {
        let peer = v4(127, 0, 0, 1);
        assert_eq!(client_ip(peer, false, Some("9.9.9.9")), peer);
    }

    #[test]
    fn trusted_proxy_from_loopback_uses_xff() {
        let peer = v4(127, 0, 0, 1);
        assert_eq!(client_ip(peer, true, Some("9.9.9.9")), v4(9, 9, 9, 9));
    }

    #[test]
    fn trusted_proxy_but_non_loopback_peer_ignores_xff() {
        let peer = v4(203, 0, 113, 5);
        assert_eq!(client_ip(peer, true, Some("9.9.9.9")), peer);
    }

    #[test]
    fn trusted_proxy_takes_first_of_multiple_xff_entries() {
        let peer = v4(127, 0, 0, 1);
        assert_eq!(client_ip(peer, true, Some("9.9.9.9, 10.0.0.1")), v4(9, 9, 9, 9));
    }

    #[test]
    fn trusted_proxy_with_unparseable_xff_falls_back_to_peer() {
        let peer = v4(127, 0, 0, 1);
        assert_eq!(client_ip(peer, true, Some("not-an-ip")), peer);
    }

    #[test]
    fn trusted_proxy_with_missing_xff_falls_back_to_peer() {
        let peer = v4(127, 0, 0, 1);
        assert_eq!(client_ip(peer, true, None), peer);
    }

    #[test]
    fn ipv6_loopback_peer_is_recognized() {
        let peer = IpAddr::V6(Ipv6Addr::LOCALHOST);
        assert_eq!(client_ip(peer, true, Some("9.9.9.9")), v4(9, 9, 9, 9));
    }
}
