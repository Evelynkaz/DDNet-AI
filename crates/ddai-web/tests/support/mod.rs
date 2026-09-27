//! Test-only support code shared by the integration tests: a minimal hand-rolled HTTP/1.1 client
//! (deliberately not `reqwest` — see `Cargo.toml`'s comment-free absence of it: this crate's own
//! philosophy elsewhere is small hand-rolled protocol code over heavy dependencies, and our tests
//! only ever need `Connection: close` request/response pairs against our own loopback server) and
//! a helper that spins up a real `ddai-web` server on an ephemeral port.

#![allow(dead_code)] // Not every helper is used by every test binary that includes this module.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::time::Duration;

use ddai_web::config::WebConfig;
use ddai_web::secrets::{self, Argon2Params, SecretsPaths};

// -------------------------------------------------------------------------------------------
// Server harness
// -------------------------------------------------------------------------------------------

pub struct TestServer {
    pub addr: SocketAddr,
    pub password: String,
    pub config: WebConfig,
    _tempdir: tempfile::TempDir,
    task: tokio::task::JoinHandle<()>,
}

impl TestServer {
    pub async fn start() -> Self {
        Self::start_with(|_| {}).await
    }

    pub async fn start_with(customize: impl FnOnce(&mut WebConfig)) -> Self {
        // Weak argon2 params: tests care about behavior, not the production hashing cost, and a
        // full 0.2s hash per login attempt would make the rate-limit tests (which fire many
        // logins) slow. `start_with_argon2` is the escape hatch for the one test (review finding
        // F4) that specifically needs production-realistic hashing cost.
        let weak_params = Argon2Params {
            m_cost_kib: 8 * 1024,
            t_cost: 1,
            p_cost: 1,
        };
        Self::start_with_argon2(weak_params, customize).await
    }

    pub async fn start_with_argon2(params: Argon2Params, customize: impl FnOnce(&mut WebConfig)) -> Self {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tempdir.path());
        let generated = secrets::generate_and_store_password(&paths, params).expect("generate password");

        let mut config = WebConfig::new(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            tempdir.path().to_path_buf(),
        );
        customize(&mut config);

        let bound = ddai_web::bind(config.clone()).await.expect("bind");
        let addr = bound.local_addr;
        let task = tokio::spawn(async move {
            let _ = ddai_web::run(bound).await;
        });

        Self {
            addr,
            password: generated.plaintext,
            config,
            _tempdir: tempdir,
            task,
        }
    }

    /// Logs in and returns `(cookie_header_value, csrf_token)`, panicking if login fails. Used by
    /// tests whose focus is something downstream of a successful login.
    pub fn login(&self) -> (String, String) {
        let response = self.login_response(&self.password);
        assert_eq!(response.status, 200, "login should succeed: {response:?}");
        let cookie = extract_set_cookie_pair(&response, self.cookie_name()).expect("Set-Cookie on login");
        let csrf_token = response.json()["csrf_token"]
            .as_str()
            .expect("csrf_token in login response")
            .to_string();
        (cookie, csrf_token)
    }

    /// Logs in and additionally returns the trusted-device cookie (review finding F7), panicking
    /// if login fails.
    pub fn login_with_device(&self) -> (String, String, String) {
        let response = self.login_response(&self.password);
        assert_eq!(response.status, 200, "login should succeed: {response:?}");
        let cookie = extract_set_cookie_pair(&response, self.cookie_name()).expect("Set-Cookie on login");
        let device_cookie =
            extract_set_cookie_pair(&response, self.device_cookie_name()).expect("device Set-Cookie on login");
        let csrf_token = response.json()["csrf_token"]
            .as_str()
            .expect("csrf_token in login response")
            .to_string();
        (cookie, device_cookie, csrf_token)
    }

    /// A raw `POST /api/login` response for `password`, with the same-origin header already set.
    /// Does not assert on the outcome — callers that just want a successful login should use
    /// [`TestServer::login`]/[`TestServer::login_with_device`] instead.
    pub fn login_response(&self, password: &str) -> RawResponse {
        send(
            self.addr,
            Req::new("POST", "/api/login")
                .header("Origin", &self.origin())
                .json_body(&serde_json::json!({ "password": password })),
        )
    }

    pub fn cookie_name(&self) -> &'static str {
        ddai_web::auth::cookie::cookie_name(self.config.cookie_secure)
    }

    pub fn device_cookie_name(&self) -> &'static str {
        ddai_web::auth::cookie::device_cookie_name(self.config.cookie_secure)
    }

    pub fn origin(&self) -> String {
        format!("http://{}", self.addr)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

// -------------------------------------------------------------------------------------------
// A tiny synchronous HTTP/1.1 client
// -------------------------------------------------------------------------------------------

#[derive(Debug)]
pub struct RawResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl RawResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn headers_named(&self, name: &str) -> Vec<&str> {
        self.headers
            .iter()
            .filter(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
            .collect()
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|e| panic!("response body is not JSON ({e}): {self:?}"))
    }
}

pub struct Req {
    method: &'static str,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Req {
    pub fn new(method: &'static str, path: &str) -> Self {
        Self {
            method,
            path: path.to_string(),
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }

    pub fn cookie(self, cookie_pair: &str) -> Self {
        self.header("Cookie", cookie_pair)
    }

    pub fn body(mut self, content_type: &str, bytes: Vec<u8>) -> Self {
        self.body = bytes;
        self.header("Content-Type", content_type)
    }

    pub fn json_body(self, value: &serde_json::Value) -> Self {
        let bytes = serde_json::to_vec(value).expect("serialize JSON body");
        self.body("application/json", bytes)
    }

    pub fn form_body(self, pairs: &[(&str, &str)]) -> Self {
        let encoded = pairs
            .iter()
            .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
            .collect::<Vec<_>>()
            .join("&");
        self.body("application/x-www-form-urlencoded", encoded.into_bytes())
    }
}

fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(byte as char),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Sends `req` to `addr` over a fresh TCP connection with `Connection: close`, and parses the
/// response. Panics on any I/O or parse failure — in these tests that always indicates a bug,
/// not an expected condition. Has a generous fixed read timeout so a hung server (a real
/// possibility to reproduce while developing these tests) fails the test instead of the whole
/// run.
pub fn send(addr: SocketAddr, req: Req) -> RawResponse {
    try_send(addr, req).unwrap_or_else(|e| panic!("request failed: {e}"))
}

/// Sends a complete, valid request line and headers declaring a `Content-Length` of
/// `declared_body_len`, then sends *no body at all* and waits `wait` before reading whatever
/// response comes back. Used to trigger the server's own request timeout (review finding F5)
/// without waiting for a real client to actually hang.
///
/// Deliberately sends complete headers (unlike an earlier version of this helper, which sent an
/// incomplete request line): hyper buffers an incomplete request line/headers below the
/// application entirely, so a request that never finishes *those* never reaches the router (and
/// therefore never reaches `tower_http::timeout::TimeoutLayer`, which wraps the router service) —
/// it would just hang until this test's own `READ_TIMEOUT`. A handler that reads the body (e.g.
/// via axum's `Bytes` extractor, as `/api/login` does) only finishes extracting *inside* the
/// timed service call, once headers are already parsed and dispatched — that's the case this is
/// for.
pub fn send_incomplete_then_wait(
    addr: SocketAddr,
    method: &str,
    path: &str,
    declared_body_len: usize,
    wait: Duration,
) -> RawResponse {
    let mut stream = TcpStream::connect(addr).expect("connect");
    stream.set_read_timeout(Some(READ_TIMEOUT)).expect("set read timeout");
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {declared_body_len}\r\n\r\n"
    );
    stream.write_all(head.as_bytes()).expect("write headers");
    std::thread::sleep(wait);
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    parse_response(&raw).expect("parse response")
}

const READ_TIMEOUT: Duration = Duration::from_secs(5);

fn try_send(addr: SocketAddr, req: Req) -> std::io::Result<RawResponse> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(READ_TIMEOUT))?;

    let mut head = format!(
        "{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n",
        req.method, req.path, addr
    );
    for (k, v) in &req.headers {
        head.push_str(k);
        head.push_str(": ");
        head.push_str(v);
        head.push_str("\r\n");
    }
    if !req.body.is_empty()
        && !req
            .headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("content-length"))
    {
        head.push_str(&format!("Content-Length: {}\r\n", req.body.len()));
    }
    head.push_str("\r\n");

    stream.write_all(head.as_bytes())?;
    stream.write_all(&req.body)?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    parse_response(&raw)
}

fn parse_response(raw: &[u8]) -> std::io::Result<RawResponse> {
    let text_end = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| std::io::Error::other("no header/body separator in response"))?;
    let head = std::str::from_utf8(&raw[..text_end]).map_err(std::io::Error::other)?;
    let mut lines = head.split("\r\n");
    let status_line = lines.next().ok_or_else(|| std::io::Error::other("empty response"))?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| std::io::Error::other("malformed status line"))?
        .parse()
        .map_err(std::io::Error::other)?;

    let mut headers = Vec::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }

    let body_start = text_end + 4;
    let body = if let Some(len) = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse::<usize>().ok())
    {
        raw[body_start..(body_start + len).min(raw.len())].to_vec()
    } else {
        raw[body_start..].to_vec()
    };

    Ok(RawResponse { status, headers, body })
}

/// Extracts `name=value` (dropping cookie attributes) from a `Set-Cookie` response header, ready
/// to send back as a `Cookie` request header.
pub fn extract_set_cookie_pair(response: &RawResponse, name: &str) -> Option<String> {
    response.headers_named("set-cookie").into_iter().find_map(|value| {
        let pair = value.split(';').next()?.trim();
        let (k, _) = pair.split_once('=')?;
        (k == name).then(|| pair.to_string())
    })
}
