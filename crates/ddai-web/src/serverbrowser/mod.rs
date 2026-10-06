//! The server browser's back end (task 5.12, D-099): the master-list cache, the owner's favourites, the proxy profiles, and the small
//! files that tie them to the units outside the web process. Schemas: `docs/formats.md` §36.
//!
//! **The web gains no privilege and no network.** It reads a cache another unit wrote, writes the favourites (`launch/favourites.json`)
//! and the proxy profiles (`secrets/<name>-proxy.toml`, 0600) into directories it already owned, and asks for work by dropping small
//! files: `launch/servers-refresh` (a path unit then runs `ddnet-ai servers-cache`, the only code that talks to the master servers) and
//! `launch/proxycheck-request.json` (another path unit runs `ddnet-ai launch check-proxy`). What the helpers answer comes back as a
//! file the web only reads. Nothing a browser sends is a path, a command line or an address the bot is told to use without the
//! root helper judging it again.
//!
//! This module holds the contract types shared with `ddnet-ai` (`launch apply`, `launch check-proxy`) and the three stores.

pub mod cache;
pub mod favourites;
pub mod proxies;

use serde::{Deserialize, Serialize};

/// The list of servers the bot was kicked or banned from (written by the root helper next to `status.json`, read by the web).
pub const BLOCKED_FILE: &str = "blocked.json";
/// The «Проверить» request (a few dozen bytes) in the launch directory, consumed by `ddnet-ai launch check-proxy`.
pub const PROXY_CHECK_REQUEST_FILE: &str = "proxycheck-request.json";
/// The check's answer, in the launch directory, written by `ddnet-ai launch check-proxy`.
pub const PROXY_CHECK_RESULT_FILE: &str = "proxycheck-result.json";
/// The file the web rewrites to ask for a fresh master list; a path unit (`PathChanged=`) runs the fetch. Its content means nothing.
pub const REFRESH_TRIGGER_FILE: &str = "servers-refresh";
/// A proxy-check request or result is small; anything bigger is refused unread.
pub const MAX_PROXY_CHECK_BYTES: usize = 2048;
/// `blocked.json` is small too.
pub const MAX_BLOCKED_BYTES: usize = 64 * 1024;

/// One kick or ban the helper remembers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockedEntry {
    /// The server's address as it is in the helper's memory.
    pub address: String,
    /// Unix seconds when it happened.
    pub at: u64,
    /// The bot's exit code (3 = kicked or banned, 4 = could not join).
    pub code: i32,
}

/// `blocked.json`: informative for the page. What really closes a server is the helper's own memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockedFile {
    pub v: u32,
    pub at: u64,
    pub blocked: Vec<BlockedEntry>,
}

/// What the web writes for «Проверить». Strict schema; the helper re-validates the name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProxyCheckRequest {
    pub v: u32,
    /// 8 to 32 lower-case hex digits ([`crate::launch::valid_id`]); ties the result to the request.
    pub id: String,
    /// Unix seconds when the web made it (see [`crate::launch::request_is_fresh`]).
    pub ts: u64,
    pub proxy: String,
}

/// What `ddnet-ai launch check-proxy` writes: fixed codes and numbers, never an address, a user name or a password.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProxyCheckResult {
    pub v: u32,
    pub id: String,
    /// Unix seconds.
    pub at: u64,
    pub proxy: String,
    pub ok: bool,
    /// `ok`, `udp_not_supported`, `auth_failed`, `no_auth_method`, `timeout`, `connect_failed`, `refused`, `relay_refused`,
    /// `probe_failed`, `protocol_error`, `proxy_missing`, `proxy_file_bad`, `request_stale`, `bad_request`.
    pub code: String,
    /// Where the UDP relay is: `same_host`, `substituted` or `remote`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay: Option<String>,
    /// The file's relay rule: `proxy-host-only` or `public`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_mode: Option<String>,
    /// Median round-trip time of the UDP probe through the relay (`relay = "public"` or session picking), in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub udp_rtt_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_sent: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_replies: Option<u16>,
}

/// Why a proxy-check request is not one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ProxyCheckParseError {
    #[error("the request is larger than {MAX_PROXY_CHECK_BYTES} bytes")]
    TooLarge,
    #[error("not a valid proxy-check request")]
    Invalid,
}

/// Parses a proxy-check request: size, strict schema, version, id, proxy name.
pub fn parse_proxy_check_request(bytes: &[u8]) -> Result<ProxyCheckRequest, ProxyCheckParseError> {
    if bytes.len() > MAX_PROXY_CHECK_BYTES {
        return Err(ProxyCheckParseError::TooLarge);
    }
    let req: ProxyCheckRequest = serde_json::from_slice(bytes).map_err(|_| ProxyCheckParseError::Invalid)?;
    if req.v != crate::launch::PROTOCOL_VERSION
        || !crate::launch::valid_id(&req.id)
        || !ddai_client::proxy::valid_proxy_name(&req.proxy)
    {
        return Err(ProxyCheckParseError::Invalid);
    }
    Ok(req)
}

/// Unix seconds now.
pub fn now() -> u64 {
    crate::launch::unix_now()
}

/// A rate limit for one family of routes: at least `min_gap` between two accepted requests and at most `per_minute` in any 60
/// seconds. [`RateGate::try_take`] records the attempt when it lets it through.
pub struct RateGate {
    min_gap: std::time::Duration,
    per_minute: usize,
    taken: std::sync::Mutex<std::collections::VecDeque<std::time::Instant>>,
}

impl RateGate {
    pub fn new(min_gap: std::time::Duration, per_minute: usize) -> RateGate {
        RateGate {
            min_gap,
            per_minute,
            taken: std::sync::Mutex::new(std::collections::VecDeque::new()),
        }
    }

    /// `true` and the attempt is recorded when it is allowed; `false` (nothing recorded) when it is too soon or too many.
    pub fn try_take(&self) -> bool {
        let mut q = self.taken.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while q
            .front()
            .is_some_and(|t| t.elapsed() >= std::time::Duration::from_secs(60))
        {
            q.pop_front();
        }
        if q.back().is_some_and(|t| t.elapsed() < self.min_gap) || q.len() >= self.per_minute {
            return false;
        }
        q.push_back(std::time::Instant::now());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> serde_json::Value {
        serde_json::json!({"v":1,"id":"0123456789abcdef","ts":5,"proxy":"hproxy-1"})
    }

    #[test]
    fn a_proxy_check_request_is_strict() {
        assert!(parse_proxy_check_request(req().to_string().as_bytes()).is_ok());
        for (k, bad) in [
            ("proxy", serde_json::json!("../x")),
            ("proxy", serde_json::json!("a b")),
            ("proxy", serde_json::json!("")),
            ("proxy", serde_json::json!("a".repeat(65))),
            ("id", serde_json::json!("XYZ")),
            ("v", serde_json::json!(2)),
            ("ts", serde_json::json!(-1)),
            ("extra", serde_json::json!(1)),
        ] {
            let mut v = req();
            v[k] = bad.clone();
            assert_eq!(
                parse_proxy_check_request(v.to_string().as_bytes()),
                Err(ProxyCheckParseError::Invalid),
                "{k}={bad}"
            );
        }
        assert_eq!(
            parse_proxy_check_request(&vec![b' '; MAX_PROXY_CHECK_BYTES + 1]),
            Err(ProxyCheckParseError::TooLarge)
        );
        for b in [&b""[..], b"null", b"{", b"\xff"] {
            assert!(parse_proxy_check_request(b).is_err());
        }
    }
}
