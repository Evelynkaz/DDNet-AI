//! The owner-only bot control (task 5.6, D-070): the routes under `/api/bot/*`, against a scripted fake bot on a real
//! Unix socket (the real bot is the 4.4 end-to-end's job). Covers what the task requires of the web side: every request
//! without a session, without CSRF or with a bad Origin is refused and reaches nothing; a command reaches the bot's socket
//! and its reply comes back; the lists editor round-trips through the file and tells the bot to reload; nothing outside
//! the closed command vocabulary gets through; and no name is ever logged.

mod support;

use std::io::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use ddai_botctl::proto::{ControlCommand, ControlReply, ControlRequest, ReplyCode};
use ddai_botctl::relations::{ListKind, Relations};
use support::{Req, TestServer, send};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;

// ---- a scripted bot on the control socket ---------------------------------------------------------

type Responder = Arc<dyn Fn(&ControlRequest) -> ControlReply + Send + Sync>;

struct FakeBot {
    /// Every request line the bot received, as text.
    lines: Arc<Mutex<Vec<String>>>,
    task: tokio::task::JoinHandle<()>,
}

impl FakeBot {
    fn start(socket: &Path, respond: impl Fn(&ControlRequest) -> ControlReply + Send + Sync + 'static) -> FakeBot {
        std::fs::create_dir_all(socket.parent().unwrap()).unwrap();
        let listener = UnixListener::bind(socket).unwrap();
        let lines = Arc::new(Mutex::new(Vec::new()));
        let respond: Responder = Arc::new(respond);
        let lines2 = lines.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    return;
                };
                let (lines, respond) = (lines2.clone(), respond.clone());
                tokio::spawn(async move {
                    let (r, mut w) = stream.into_split();
                    let mut reader = BufReader::new(r);
                    let mut line = String::new();
                    while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                        lines.lock().unwrap().push(line.trim_end().to_string());
                        let reply = match serde_json::from_str::<ControlRequest>(line.trim_end()) {
                            Ok(req) => respond(&req),
                            Err(_) => ControlReply::refused(ReplyCode::BadRequest, "bad request"),
                        };
                        let mut out = serde_json::to_vec(&reply).unwrap();
                        out.push(b'\n');
                        let _ = w.write_all(&out).await;
                        line.clear();
                    }
                });
            }
        });
        FakeBot { lines, task }
    }

    fn received(&self) -> Vec<String> {
        self.lines.lock().unwrap().clone()
    }

    fn requests(&self) -> Vec<ControlRequest> {
        self.received()
            .iter()
            .map(|l| serde_json::from_str(l).expect("the web sent a valid request"))
            .collect()
    }
}

impl Drop for FakeBot {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn ok_bot(socket: &Path) -> FakeBot {
    FakeBot::start(socket, |req| {
        ControlReply::answer(true, &format!("did {}", req.cmd.tag()))
    })
}

/// A fake bot that, like the real one, answers `reload_relations` with the digest of the lists file it reads.
fn reloading_bot(socket: &Path, lists: &Path) -> FakeBot {
    let lists = lists.to_path_buf();
    FakeBot::start(socket, move |req| match &req.cmd {
        ControlCommand::ReloadRelations {} => {
            let r = Relations::load(&lists).expect("the file the web wrote parses");
            let mut reply = ControlReply::answer(true, "lists reloaded (counts)");
            reply.data = Some(serde_json::json!({"counts": {"friend": r.len(ListKind::Friend)}, "digest": r.digest()}));
            reply
        }
        other => ControlReply::answer(true, &format!("did {}", other.tag())),
    })
}

// ---- helpers --------------------------------------------------------------------------------------

struct Login {
    cookie: String,
    csrf: String,
}

fn login(server: &TestServer) -> Login {
    let (cookie, csrf) = server.login();
    Login { cookie, csrf }
}

fn post(server: &TestServer, l: &Login, path: &str, body: &serde_json::Value) -> support::RawResponse {
    send(
        server.addr,
        Req::new("POST", path)
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &l.csrf)
            .json_body(body),
    )
}

fn get(server: &TestServer, l: &Login, path: &str) -> support::RawResponse {
    send(server.addr, Req::new("GET", path).cookie(&l.cookie))
}

fn command(server: &TestServer, l: &Login, body: serde_json::Value) -> support::RawResponse {
    post(server, l, "/api/bot/command", &body)
}

fn relations_edit(server: &TestServer, l: &Login, op: &str, kind: &str, name: &str) -> support::RawResponse {
    post(
        server,
        l,
        "/api/bot/relations",
        &serde_json::json!({"op": op, "kind": kind, "name": name}),
    )
}

fn control_socket(server: &TestServer) -> std::path::PathBuf {
    server.config.control_socket.clone()
}

fn relations_file(server: &TestServer) -> std::path::PathBuf {
    server.config.relations_path.clone()
}

// ---- refusals: no session, no CSRF, bad Origin ----------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_route_refuses_a_request_without_a_session_and_nothing_reaches_the_bot() {
    let server = TestServer::start().await;
    let bot = ok_bot(&control_socket(&server));
    for path in ["/api/bot/status", "/api/bot/relations"] {
        let r = send(server.addr, Req::new("GET", path));
        assert_eq!(r.status, 401, "{path}");
        assert_eq!(r.json()["error"], "unauthenticated");
    }
    // A POST with a correct Origin and a correct-looking CSRF header, but no session cookie.
    for (path, body) in [
        ("/api/bot/command", serde_json::json!({"type": "stop"})),
        (
            "/api/bot/relations",
            serde_json::json!({"op": "add", "kind": "friend", "name": "x"}),
        ),
    ] {
        let r = send(
            server.addr,
            Req::new("POST", path)
                .header("Origin", &server.origin())
                .header("X-CSRF-Token", "AAAA")
                .json_body(&body),
        );
        assert_eq!(r.status, 401, "{path}");
    }
    // A cookie that is not a valid session (forged value) is no session either.
    let forged = format!("{}=AAAA.BBBB", server.cookie_name());
    let r = send(server.addr, Req::new("GET", "/api/bot/status").cookie(&forged));
    assert_eq!(r.status, 401);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(bot.received().is_empty(), "nothing reached the bot");
    assert!(!relations_file(&server).exists(), "nothing was written");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_post_without_or_with_a_wrong_csrf_token_is_refused() {
    let server = TestServer::start().await;
    let bot = ok_bot(&control_socket(&server));
    let l = login(&server);
    for (path, body) in [
        ("/api/bot/command", serde_json::json!({"type": "kill"})),
        (
            "/api/bot/relations",
            serde_json::json!({"op": "add", "kind": "friend", "name": "x"}),
        ),
    ] {
        let missing = send(
            server.addr,
            Req::new("POST", path)
                .cookie(&l.cookie)
                .header("Origin", &server.origin())
                .json_body(&body),
        );
        assert_eq!(
            (missing.status, missing.json()["error"].as_str()),
            (403, Some("missing_csrf")),
            "{path}"
        );
        for wrong in ["AAAA", "", "not base64 !!", &l.csrf[..l.csrf.len() - 2]] {
            let r = send(
                server.addr,
                Req::new("POST", path)
                    .cookie(&l.cookie)
                    .header("Origin", &server.origin())
                    .header("X-CSRF-Token", wrong)
                    .json_body(&body),
            );
            assert_eq!(
                (r.status, r.json()["error"].as_str()),
                (403, Some("bad_csrf")),
                "{path} {wrong:?}"
            );
        }
    }
    // The CSRF token of another session is not this session's.
    let other = login(&server);
    let r = send(
        server.addr,
        Req::new("POST", "/api/bot/command")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &other.csrf)
            .json_body(&serde_json::json!({"type": "kill"})),
    );
    assert_eq!(r.status, 403);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(bot.received().is_empty());
    assert!(!relations_file(&server).exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_post_with_a_bad_or_missing_origin_is_refused_even_with_session_and_csrf() {
    let server = TestServer::start().await;
    let bot = ok_bot(&control_socket(&server));
    let l = login(&server);
    let body = serde_json::json!({"type": "stop"});
    let rel = serde_json::json!({"op": "add", "kind": "friend", "name": "x"});
    let cases: Vec<(&str, Vec<(&str, &str)>)> = vec![
        ("evil origin", vec![("Origin", "http://evil.example")]),
        ("null origin", vec![("Origin", "null")]),
        ("wrong port", vec![("Origin", "http://127.0.0.1:1")]),
        ("no origin at all", vec![]),
        // The lax check would let this through (Sec-Fetch-Site says same-origin); the strict one wants an Origin.
        (
            "no origin, same-origin fetch metadata",
            vec![("Sec-Fetch-Site", "same-origin")],
        ),
        ("cross-site fetch metadata", vec![("Sec-Fetch-Site", "cross-site")]),
    ];
    for (label, headers) in cases {
        for (path, b) in [("/api/bot/command", &body), ("/api/bot/relations", &rel)] {
            let mut req = Req::new("POST", path).cookie(&l.cookie).header("X-CSRF-Token", &l.csrf);
            for (k, v) in &headers {
                req = req.header(k, v);
            }
            let r = send(server.addr, req.json_body(b));
            assert_eq!(
                (r.status, r.json()["error"].as_str()),
                (403, Some("cross_origin")),
                "{label} {path}"
            );
        }
    }
    // An Origin that matches but a cross-site Sec-Fetch-Site is refused too.
    let r = send(
        server.addr,
        Req::new("POST", "/api/bot/command")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("Sec-Fetch-Site", "cross-site")
            .header("X-CSRF-Token", &l.csrf)
            .json_body(&body),
    );
    assert_eq!(r.status, 403);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(bot.received().is_empty());
    assert!(!relations_file(&server).exists());
    // The right Origin passes.
    assert_eq!(command(&server, &l, body).status, 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn behind_https_the_origin_must_be_https() {
    let server = TestServer::start_with(|c| c.cookie_secure = true).await;
    let _bot = ok_bot(&control_socket(&server));
    let l = login(&server);
    let r = post(&server, &l, "/api/bot/command", &serde_json::json!({"type": "stop"}));
    assert_eq!(r.status, 403, "http:// Origin with a secure deployment");
    let https = format!("https://{}", server.addr);
    let r = send(
        server.addr,
        Req::new("POST", "/api/bot/command")
            .cookie(&l.cookie)
            .header("Origin", &https)
            .header("X-CSRF-Token", &l.csrf)
            .json_body(&serde_json::json!({"type": "stop"})),
    );
    assert_eq!(r.status, 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_post_must_be_json() {
    let server = TestServer::start().await;
    let bot = ok_bot(&control_socket(&server));
    let l = login(&server);
    for path in ["/api/bot/command", "/api/bot/relations"] {
        let r = send(
            server.addr,
            Req::new("POST", path)
                .cookie(&l.cookie)
                .header("Origin", &server.origin())
                .header("X-CSRF-Token", &l.csrf)
                .form_body(&[("type", "stop")]),
        );
        assert_eq!(r.status, 415, "{path}");
    }
    assert!(bot.received().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn after_logout_the_routes_are_closed_again() {
    let server = TestServer::start().await;
    let _bot = ok_bot(&control_socket(&server));
    let l = login(&server);
    assert_eq!(get(&server, &l, "/api/bot/status").status, 200);
    let out = send(
        server.addr,
        Req::new("POST", "/api/logout")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &l.csrf),
    );
    assert_eq!(out.status, 200);
    assert_eq!(get(&server, &l, "/api/bot/status").status, 401);
    assert_eq!(command(&server, &l, serde_json::json!({"type": "stop"})).status, 401);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_security_headers_and_no_store_cover_the_new_routes() {
    let server = TestServer::start().await;
    let l = login(&server);
    let responses = vec![
        get(&server, &l, "/api/bot/status"),
        get(&server, &l, "/api/bot/relations"),
        command(&server, &l, serde_json::json!({"type": "stop"})),
        send(server.addr, Req::new("GET", "/api/bot/status")),
    ];
    for r in responses {
        assert!(
            r.header("content-security-policy")
                .unwrap()
                .contains("default-src 'self'")
        );
        assert_eq!(r.header("x-content-type-options"), Some("nosniff"));
        assert_eq!(r.header("x-frame-options"), Some("DENY"));
        assert_eq!(r.header("referrer-policy"), Some("no-referrer"));
        assert_eq!(r.header("cache-control"), Some("no-store"));
        assert!(r.header("access-control-allow-origin").is_none(), "no CORS");
    }
    // An oversized body is refused by the global limit (16 KiB), before any handler.
    let big = serde_json::json!({"op": "add", "kind": "friend", "name": "x".repeat(20_000)});
    assert_eq!(post(&server, &l, "/api/bot/relations", &big).status, 413);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn there_is_no_route_to_chat_to_connect_or_to_the_allowlist() {
    let server = TestServer::start().await;
    let l = login(&server);
    for path in [
        "/api/bot/say",
        "/api/bot/chat",
        "/api/bot/connect",
        "/api/bot/server",
        "/api/bot/allowlist",
        "/api/bot/live-servers",
        "/api/bot/ready",
        "/api/bot/quit",
        "/api/bot/settings",
    ] {
        for method in ["GET", "POST"] {
            let r = send(
                server.addr,
                Req::new(method, path)
                    .cookie(&l.cookie)
                    .header("Origin", &server.origin())
                    .header("X-CSRF-Token", &l.csrf)
                    .json_body(&serde_json::json!({})),
            );
            assert_eq!(r.status, 404, "{method} {path}");
        }
    }
}

// ---- commands --------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_command_reaches_the_bots_socket_and_its_reply_is_shown() {
    let server = TestServer::start().await;
    let bot = FakeBot::start(&control_socket(&server), |req| match &req.cmd {
        ControlCommand::Kill {} => ControlReply::answer(false, "reset is on cooldown"),
        other => ControlReply::answer(true, &format!("did {}", other.tag())),
    });
    let l = login(&server);
    let r = command(&server, &l, serde_json::json!({"type": "mode", "mode": "hold"}));
    assert_eq!(r.status, 200, "{r:?}");
    assert_eq!(r.json(), serde_json::json!({"ok": true, "text": "did mode:hold"}));
    let r = command(&server, &l, serde_json::json!({"type": "wb", "mode": "left"}));
    assert_eq!(r.json()["text"], "did wb:left");
    let r = command(&server, &l, serde_json::json!({"type": "goto", "x": 12, "y": 34}));
    assert_eq!(r.json()["text"], "did goto:12,34");
    // The bot's refusal is shown as it is (HTTP 200, ok false).
    let r = command(&server, &l, serde_json::json!({"type": "kill"}));
    assert_eq!(r.status, 200);
    assert_eq!(
        r.json(),
        serde_json::json!({"ok": false, "text": "reset is on cooldown"})
    );
    // What the bot received: exactly the typed requests, with an opaque session tag.
    let reqs = bot.requests();
    assert_eq!(reqs.len(), 4);
    assert_eq!(
        reqs[0].cmd,
        ControlCommand::Mode {
            mode: ddai_botctl::proto::ModeArg::Hold
        }
    );
    assert_eq!(reqs[2].cmd, ControlCommand::Goto { x: 12, y: 34 });
    let tag = reqs[0].session.clone();
    assert!(ddai_botctl::proto::valid_session_tag(&tag) && tag.len() == 16, "{tag}");
    assert!(reqs.iter().all(|r| r.session == tag), "one tag per session");
    assert!(!l.cookie.contains(&tag) && !l.csrf.contains(&tag), "not the cookie");
    // Another login is another session tag.
    let l2 = login(&server);
    command(&server, &l2, serde_json::json!({"type": "go"}));
    assert_ne!(bot.requests().last().unwrap().session, tag);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_button_of_the_page_is_a_command_the_bot_understands() {
    let server = TestServer::start().await;
    let bot = ok_bot(&control_socket(&server));
    let l = login(&server);
    let bodies = [
        serde_json::json!({"type": "mode", "mode": "fight"}),
        serde_json::json!({"type": "mode", "mode": "passive"}),
        serde_json::json!({"type": "mode", "mode": "hold"}),
        serde_json::json!({"type": "stop"}),
        serde_json::json!({"type": "go"}),
        serde_json::json!({"type": "wb", "mode": "auto"}),
        serde_json::json!({"type": "wb", "mode": "left"}),
        serde_json::json!({"type": "wb", "mode": "right"}),
        serde_json::json!({"type": "wb", "mode": "off"}),
        serde_json::json!({"type": "brain", "brain": "hybrid"}),
        serde_json::json!({"type": "brain", "brain": "planner"}),
        serde_json::json!({"type": "brain", "brain": "scripted"}),
        serde_json::json!({"type": "brain", "brain": "idle"}),
        serde_json::json!({"type": "brain", "brain": "fly"}),
        serde_json::json!({"type": "kill"}),
        serde_json::json!({"type": "clip", "note": ""}),
        serde_json::json!({"type": "clip", "note": "nice save"}),
        serde_json::json!({"type": "goto", "x": 1, "y": 2}),
        serde_json::json!({"type": "spec"}),
        serde_json::json!({"type": "join"}),
        serde_json::json!({"type": "reload_relations"}),
    ];
    for b in &bodies {
        let r = command(&server, &l, b.clone());
        assert_eq!(r.status, 200, "{b}: {r:?}");
        assert_eq!(r.json()["ok"], true, "{b}");
    }
    assert_eq!(bot.requests().len(), bodies.len());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn nothing_outside_the_closed_vocabulary_is_forwarded() {
    let server = TestServer::start().await;
    let bot = ok_bot(&control_socket(&server));
    let l = login(&server);
    for body in [
        serde_json::json!({"type": "say", "text": "hello"}),
        serde_json::json!({"type": "chat", "message": "hello"}),
        serde_json::json!({"type": "quit"}),
        serde_json::json!({"type": "target", "name": "someone"}),
        serde_json::json!({"type": "connect", "server": "1.2.3.4:8303"}),
        serde_json::json!({"type": "allowlist", "add": "1.2.3.4:8303"}),
        serde_json::json!({"type": "ready", "value": true}),
        serde_json::json!({"type": "stop", "say": "hello"}),
        serde_json::json!({"type": "mode", "mode": "goto"}),
        serde_json::json!({"type": "brain", "brain": "chatty"}),
        serde_json::json!("stop"),
        serde_json::json!([]),
        serde_json::json!(null),
    ] {
        let r = command(&server, &l, body.clone());
        assert_eq!(
            (r.status, r.json()["error"].as_str()),
            (400, Some("bad_request")),
            "{body}"
        );
    }
    // Arguments the type system cannot check.
    for body in [
        serde_json::json!({"type": "goto", "x": 100_001, "y": 0}),
        serde_json::json!({"type": "goto", "x": -2147483648, "y": 0}),
        serde_json::json!({"type": "clip", "note": "x".repeat(61)}),
        serde_json::json!({"type": "clip", "note": "a\nb"}),
    ] {
        let r = command(&server, &l, body.clone());
        assert_eq!((r.status, r.json()["error"].as_str()), (400, Some("invalid")), "{body}");
    }
    let r = send(
        server.addr,
        Req::new("POST", "/api/bot/command")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &l.csrf)
            .body("application/json", b"{not json".to_vec()),
    );
    assert_eq!(r.status, 400);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(bot.received().is_empty(), "{:?}", bot.received());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn when_the_bot_is_not_there_or_misbehaves_the_owner_is_told_why() {
    let server = TestServer::start().await;
    let l = login(&server);
    // No socket at all.
    let r = command(&server, &l, serde_json::json!({"type": "stop"}));
    assert_eq!((r.status, r.json()["error"].as_str()), (503, Some("bot_unavailable")));
    // The bot's own refusals map to the right HTTP statuses.
    let codes = Arc::new(Mutex::new(Some(ReplyCode::RateLimited)));
    let codes2 = codes.clone();
    let _bot = FakeBot::start(&control_socket(&server), move |_| match *codes2.lock().unwrap() {
        Some(code) => ControlReply::refused(code, "no"),
        None => ControlReply::answer(true, "ok"),
    });
    for (code, status) in [
        (ReplyCode::RateLimited, 429),
        (ReplyCode::Busy, 503),
        (ReplyCode::Gone, 503),
        (ReplyCode::Timeout, 504),
        (ReplyCode::BadRequest, 400),
    ] {
        *codes.lock().unwrap() = Some(code);
        let r = command(&server, &l, serde_json::json!({"type": "go"}));
        assert_eq!(r.status, status, "{code:?}");
        assert_eq!(r.json()["ok"], false);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hostile_bot_socket_cannot_hang_or_confuse_the_web() {
    let server = TestServer::start().await;
    let l = login(&server);
    let socket = control_socket(&server);
    std::fs::create_dir_all(socket.parent().unwrap()).unwrap();
    let listener = UnixListener::bind(&socket).unwrap();
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let (r, mut w) = stream.into_split();
                let mut line = String::new();
                let _ = BufReader::new(r).read_line(&mut line).await;
                let _ = w.write_all(b"<html>not a reply</html>\n").await;
            });
        }
    });
    let r = command(&server, &l, serde_json::json!({"type": "stop"}));
    assert_eq!((r.status, r.json()["error"].as_str()), (502, Some("bot_protocol")));
    task.abort();
}

// ---- the status panel ----------------------------------------------------------------------------

/// A scripted bridge: HELLO, then the given STATUS JSON every 100 ms, `count` times.
fn fake_bridge(socket: &Path, status: serde_json::Value, count: usize) -> tokio::task::JoinHandle<()> {
    std::fs::create_dir_all(socket.parent().unwrap()).unwrap();
    let listener = UnixListener::bind(socket).unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let status = status.clone();
            tokio::spawn(async move {
                let msg = |kind: u8, payload: &[u8]| {
                    let mut m = ((payload.len() + 1) as u32).to_le_bytes().to_vec();
                    m.push(kind);
                    m.extend_from_slice(payload);
                    m
                };
                let _ = stream.write_all(&msg(1, b"DDBL\x01")).await;
                for _ in 0..count {
                    let _ = stream.write_all(&msg(5, status.to_string().as_bytes())).await;
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                tokio::time::sleep(Duration::from_secs(60)).await;
            });
        }
    })
}

fn sample_status() -> serde_json::Value {
    serde_json::json!({
        "tick": 1234, "own": 3, "target": 5, "mode": "fight", "brain": "hybrid", "alive": true, "frozen": false,
        "blocks": 7, "blocked_by": 2, "self_kills": 1, "decisions": 99, "collapsed": 0,
        "decide_p50_us": 800, "decide_p99_us": 4100, "brain_p99_us": 3900, "overhead_p99_us": 200,
        "telemetry": null, "connected": true, "server": "127.0.0.1:8303", "name": "bot", "clan": "Neuroset",
        "skin": "pinky", "target_tag": "c5-0a1b2c3d", "wb": "WB: left", "goto": "", "deaths": 4,
        "clips_saved": 2, "kill_cooldown_ticks": 120
    })
}

async fn wait_for(mut f: impl FnMut() -> bool) {
    for _ in 0..100 {
        if f() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("timed out");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_status_panel_shows_the_live_bots_status_or_says_it_is_not_there() {
    let dir = tempfile::tempdir().unwrap();
    let bridge_sock = dir.path().join("live.sock");
    let _bridge = fake_bridge(&bridge_sock, sample_status(), 100);
    let sock2 = bridge_sock.clone();
    let server = TestServer::start_with(move |c| c.bot_socket = Some(sock2)).await;
    let l = login(&server);
    let mut live = serde_json::Value::Null;
    wait_for(|| {
        live = get(&server, &l, "/api/bot/status").json();
        live["live"] == true
    })
    .await;
    assert_eq!(live["bridge"], true);
    assert_eq!(live["status"]["target_tag"], "c5-0a1b2c3d");
    assert_eq!(live["status"]["kill_cooldown_ticks"], 120);
    assert_eq!(live["status"]["clan"], "Neuroset");
    assert!(live["age_ms"].as_u64().unwrap() < 3000);
    assert_eq!(live["control_socket"], false, "no control socket was created");

    // A web unit with no bridge says so, and shows no status.
    let plain = TestServer::start().await;
    let lp = login(&plain);
    let s = get(&plain, &lp, "/api/bot/status").json();
    assert_eq!(
        (s["bridge"].clone(), s["live"].clone(), s["status"].clone()),
        (false.into(), false.into(), serde_json::Value::Null)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_status_that_stops_coming_is_not_shown_as_current() {
    let dir = tempfile::tempdir().unwrap();
    let bridge_sock = dir.path().join("live.sock");
    let _bridge = fake_bridge(&bridge_sock, sample_status(), 3); // 3 statuses, then silence
    let sock2 = bridge_sock.clone();
    let server = TestServer::start_with(move |c| c.bot_socket = Some(sock2)).await;
    let l = login(&server);
    wait_for(|| get(&server, &l, "/api/bot/status").json()["live"] == true).await;
    tokio::time::sleep(Duration::from_millis(3500)).await;
    let s = get(&server, &l, "/api/bot/status").json();
    assert_eq!(s["live"], false);
    assert_eq!(s["status"], serde_json::Value::Null, "the old numbers are not shown");
    assert!(s["age_ms"].as_u64().unwrap() >= 3000);
}

// ---- the lists editor ----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_relations_add_and_remove_round_trip_through_the_file_and_the_bot_is_told_to_reload() {
    let server = TestServer::start().await;
    let bot = reloading_bot(&control_socket(&server), &relations_file(&server));
    let l = login(&server);

    let empty = get(&server, &l, "/api/bot/relations");
    assert_eq!(empty.status, 200);
    assert_eq!(empty.json()["lists"]["friend"], serde_json::json!([]));

    // Add: normalised, stored, shown, applied.
    let r = relations_edit(&server, &l, "add", "friend", "  Some   NICK ");
    assert_eq!(r.status, 200, "{r:?}");
    let j = r.json();
    assert_eq!(j["normalised"], "some nick");
    assert_eq!(j["changed"], true);
    assert_eq!(j["applied"], "applied");
    assert_eq!(j["lists"]["friend"], serde_json::json!(["some nick"]));
    let on_disk = Relations::load(&relations_file(&server)).unwrap();
    assert!(on_disk.contains(ListKind::Friend, "SOME nick"));
    assert!(!on_disk.contains(ListKind::Friend, "some"), "exact, not a substring");
    assert_eq!(
        bot.requests()
            .iter()
            .filter(|r| r.cmd == ControlCommand::ReloadRelations {})
            .count(),
        1
    );

    // GET shows it.
    assert_eq!(
        get(&server, &l, "/api/bot/relations").json()["lists"]["friend"],
        serde_json::json!(["some nick"])
    );

    // The same name again: unchanged, no reload.
    let r = relations_edit(&server, &l, "add", "friend", "SOME NICK").json();
    assert_eq!(
        (r["changed"].clone(), r["applied"].clone()),
        (false.into(), "unchanged".into())
    );
    assert_eq!(bot.requests().len(), 1);

    // War takes it off the friend list, and says so.
    let r = relations_edit(&server, &l, "add", "war", "some nick").json();
    assert_eq!(r["moved_from"], serde_json::json!(["friend"]));
    assert_eq!(r["lists"]["friend"], serde_json::json!([]));
    assert_eq!(r["lists"]["war"], serde_json::json!(["some nick"]));

    // Clans are their own lists.
    let r = relations_edit(&server, &l, "add", "clanfriend", "Good Clan").json();
    assert_eq!(r["lists"]["clanfriend"], serde_json::json!(["good clan"]));
    assert_eq!(
        Relations::load(&relations_file(&server))
            .unwrap()
            .names(ListKind::ClanFriend),
        vec!["good clan"]
    );

    // Remove by another spelling; removing what is not there changes nothing.
    let r = relations_edit(&server, &l, "remove", "war", "(2)Some Nick").json();
    assert_eq!(r["changed"], true);
    assert_eq!(r["lists"]["war"], serde_json::json!([]));
    let r = relations_edit(&server, &l, "remove", "war", "some nick").json();
    assert_eq!(
        (r["changed"].clone(), r["applied"].clone()),
        (false.into(), "unchanged".into())
    );
    assert!(bot.requests().len() >= 4);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_lists_are_saved_even_when_the_bot_is_not_running_and_say_so() {
    let server = TestServer::start().await; // no fake bot: no control socket
    let l = login(&server);
    let r = relations_edit(&server, &l, "add", "ignore", "Noisy");
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["applied"], "unavailable");
    assert_eq!(r.json()["lists"]["ignore"], serde_json::json!(["noisy"]));
    assert!(
        Relations::load(&relations_file(&server))
            .unwrap()
            .contains(ListKind::Ignore, "noisy")
    );
    // A bot that refuses or is rate limited leaves the file saved too.
    let _bot = FakeBot::start(&control_socket(&server), |_| {
        ControlReply::refused(ReplyCode::RateLimited, "slow")
    });
    let r = relations_edit(&server, &l, "add", "ignore", "Noisy Two").json();
    assert_eq!(r["applied"], "rate_limited");
    assert_eq!(r["lists"]["ignore"].as_array().unwrap().len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bad_relation_edits_are_refused_and_change_nothing() {
    let server = TestServer::start().await;
    let bot = ok_bot(&control_socket(&server));
    let l = login(&server);
    for (name, detail) in [("", "empty"), ("   ", "empty"), ("(3)", "empty"), ("a\nb", "control")] {
        let r = relations_edit(&server, &l, "add", "friend", name);
        assert_eq!((r.status, r.json()["detail"].as_str()), (400, Some(detail)), "{name:?}");
    }
    let r = relations_edit(&server, &l, "add", "friend", &"x".repeat(65));
    assert_eq!((r.status, r.json()["detail"].as_str()), (400, Some("too_long")));
    for body in [
        serde_json::json!({"op": "toggle", "kind": "friend", "name": "x"}),
        serde_json::json!({"op": "add", "kind": "enemy", "name": "x"}),
        serde_json::json!({"op": "add", "kind": "Friend", "name": "x"}),
        serde_json::json!({"op": "add", "kind": "friend"}),
        serde_json::json!({"op": "add", "kind": "friend", "name": "x", "extra": 1}),
    ] {
        let r = post(&server, &l, "/api/bot/relations", &body);
        assert_eq!(
            (r.status, r.json()["error"].as_str()),
            (400, Some("bad_request")),
            "{body}"
        );
    }
    assert!(!relations_file(&server).exists());
    assert!(bot.received().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_corrupt_lists_file_is_reported_and_never_overwritten() {
    let server = TestServer::start().await;
    let l = login(&server);
    std::fs::create_dir_all(relations_file(&server).parent().unwrap()).unwrap();
    std::fs::write(relations_file(&server), "{ not json").unwrap();
    let r = get(&server, &l, "/api/bot/relations");
    assert_eq!(
        (r.status, r.json()["error"].as_str()),
        (500, Some("relations_unreadable"))
    );
    let r = relations_edit(&server, &l, "add", "friend", "x");
    assert_eq!(r.status, 500);
    assert_eq!(std::fs::read_to_string(relations_file(&server)).unwrap(), "{ not json");
}

// ---- names are never logged ----------------------------------------------------------------------

#[derive(Clone, Default)]
struct LogBuf(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for LogBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuf {
    type Writer = LogBuf;

    fn make_writer(&'a self) -> LogBuf {
        self.clone()
    }
}

fn captured_log() -> &'static LogBuf {
    static LOG: OnceLock<LogBuf> = OnceLock::new();
    LOG.get_or_init(|| {
        let buf = LogBuf::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .with_writer(buf.clone())
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
        buf
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_name_and_no_note_ever_reaches_the_log() {
    let log = captured_log();
    let server = TestServer::start().await;
    let _bot = ok_bot(&control_socket(&server));
    let l = login(&server);
    let name = "ZZsecretnickZZ";
    relations_edit(&server, &l, "add", "friend", name);
    relations_edit(&server, &l, "add", "clanwar", "ZZsecretclanZZ");
    relations_edit(&server, &l, "remove", "friend", name);
    relations_edit(&server, &l, "add", "friend", &format!("{name}\n")); // refused
    command(
        &server,
        &l,
        serde_json::json!({"type": "clip", "note": "ZZsecretnoteZZ"}),
    );
    command(&server, &l, serde_json::json!({"type": "say", "text": "ZZsecretsayZZ"})); // refused
    get(&server, &l, "/api/bot/relations");
    tokio::time::sleep(Duration::from_millis(100)).await;
    let text = String::from_utf8_lossy(&log.0.lock().unwrap()).to_lowercase();
    assert!(
        text.contains("relations edit from the web"),
        "the capture works: {text}"
    );
    assert!(text.contains("bot command from the web"), "{text}");
    for secret in ["zzsecret", "ZZsecret".to_lowercase().as_str()] {
        assert!(!text.contains(secret), "a name or note was logged");
    }
    let _ = std::io::stdout().flush();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_web_checks_that_the_bot_reloaded_the_same_lists() {
    let server = TestServer::start().await;
    let l = login(&server);
    // A bot that reads some other file answers ok with another digest: "applied" must not be claimed.
    let other = tempfile::tempdir().unwrap();
    let mut elsewhere = Relations::new();
    elsewhere.add(ListKind::Friend, "someone else");
    let _bot = FakeBot::start(&control_socket(&server), {
        let digest = elsewhere.digest();
        move |req| {
            let mut r = ControlReply::answer(true, "lists reloaded (friend 1)");
            if req.cmd == (ControlCommand::ReloadRelations {}) {
                r.data = Some(serde_json::json!({"digest": digest}));
            }
            r
        }
    });
    let r = relations_edit(&server, &l, "add", "friend", "Pal").json();
    assert_eq!(r["applied"], "mismatch", "{r}");
    assert_eq!(r["applied_text"], "lists reloaded (friend 1)");
    drop(other);
    // An old bot that sends no digest cannot be verified: said so, not claimed.
    drop(_bot);
    std::fs::remove_file(control_socket(&server)).unwrap();
    let _bot = ok_bot(&control_socket(&server));
    let r = relations_edit(&server, &l, "add", "friend", "Pal Two").json();
    assert_eq!(r["applied"], "unverified", "{r}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_status_poll_does_not_keep_the_session_alive_but_a_real_action_does() {
    // 5.3 (review F2): a page left open must not extend the idle timeout. The page polls the status every 2 s.
    let server = TestServer::start_with(|c| c.idle_timeout = Duration::from_secs(2)).await;
    let l = login(&server);
    let started = std::time::Instant::now();
    let mut last = 200;
    while started.elapsed() < Duration::from_millis(3500) {
        last = get(&server, &l, "/api/bot/status").status;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(last, 401, "polling did not refresh the idle deadline");
    assert_eq!(get(&server, &l, "/api/bot/relations").status, 401);

    // A person's requests do refresh it (the relations list is loaded when the tab is opened).
    let l = login(&server);
    let started = std::time::Instant::now();
    while started.elapsed() < Duration::from_millis(3500) {
        assert_eq!(get(&server, &l, "/api/bot/relations").status, 200);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_json_media_type_is_parsed_exactly() {
    let server = TestServer::start().await;
    let bot = ok_bot(&control_socket(&server));
    let l = login(&server);
    let body = br#"{"type":"stop"}"#.to_vec();
    for (ct, status) in [
        ("application/json", 200),
        ("application/json; charset=utf-8", 200),
        ("Application/JSON", 200),
        ("application/jsonfoo", 415),
        ("application/json-patch+json", 415),
        ("text/plain; application/json", 415),
        ("", 415),
    ] {
        let r = send(
            server.addr,
            Req::new("POST", "/api/bot/command")
                .cookie(&l.cookie)
                .header("Origin", &server.origin())
                .header("X-CSRF-Token", &l.csrf)
                .body(ct, body.clone()),
        );
        assert_eq!(r.status, status, "{ct:?}");
    }
    assert_eq!(bot.received().len(), 3);
}
