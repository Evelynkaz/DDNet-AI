//! The owner's chat line (task 4.9, D-094): `POST /api/bot/say` against a scripted fake bot on a real Unix socket. Covers what the task
//! requires of the web side: every request without a session, without CSRF, with a bad Origin or not JSON is refused and reaches
//! nothing; a valid line reaches the bot as the typed `ControlCommand::Say` with the trimmed text; invalid lines (empty, long, control
//! characters, a leading `/`) are refused here and cost no rate-limit slot; the route has its own rate limit; the generic command route
//! does not take a chat line; the bot's refusals map to statuses; the page's two assets are served.

mod support;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ddai_botctl::proto::{ControlCommand, ControlReply, ControlRequest, ReplyCode};
use support::{Req, TestServer, send};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;

type Responder = Arc<dyn Fn(&ControlRequest) -> ControlReply + Send + Sync>;

struct FakeBot {
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

    fn requests(&self) -> Vec<ControlRequest> {
        self.lines
            .lock()
            .unwrap()
            .iter()
            .map(|l| serde_json::from_str(l).expect("the web sent a valid request"))
            .collect()
    }

    fn count(&self) -> usize {
        self.lines.lock().unwrap().len()
    }
}

impl Drop for FakeBot {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A bot that takes every line.
fn taking_bot(socket: &Path) -> FakeBot {
    FakeBot::start(socket, |_| ControlReply::answer(true, "accepted: it is being said now"))
}

struct Login {
    cookie: String,
    csrf: String,
}

fn login(server: &TestServer) -> Login {
    let (cookie, csrf) = server.login();
    Login { cookie, csrf }
}

fn say(server: &TestServer, l: &Login, body: &serde_json::Value) -> support::RawResponse {
    send(
        server.addr,
        Req::new("POST", "/api/bot/say")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &l.csrf)
            .json_body(body),
    )
}

fn control_socket(server: &TestServer) -> std::path::PathBuf {
    server.config.control_socket.clone()
}

// ---- refusals: no session, no CSRF, bad Origin, not JSON -------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_chat_line_without_a_session_is_refused_and_reaches_nothing() {
    let server = TestServer::start().await;
    let bot = taking_bot(&control_socket(&server));
    let r = send(
        server.addr,
        Req::new("POST", "/api/bot/say")
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", "AAAA")
            .json_body(&serde_json::json!({"text": "hello"})),
    );
    assert_eq!((r.status, r.json()["error"].as_str()), (401, Some("unauthenticated")));
    // a forged cookie is no session
    let forged = format!("{}=AAAA.BBBB", server.cookie_name());
    let r = send(
        server.addr,
        Req::new("POST", "/api/bot/say")
            .cookie(&forged)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", "AAAA")
            .json_body(&serde_json::json!({"text": "hello"})),
    );
    assert_eq!(r.status, 401);
    // GET is not a route for it
    let l = login(&server);
    let r = send(server.addr, Req::new("GET", "/api/bot/say").cookie(&l.cookie));
    assert_eq!(r.status, 405);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(bot.count(), 0, "nothing reached the bot");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_chat_line_without_or_with_a_wrong_csrf_token_is_refused() {
    let server = TestServer::start().await;
    let bot = taking_bot(&control_socket(&server));
    let l = login(&server);
    let body = serde_json::json!({"text": "hello"});
    let missing = send(
        server.addr,
        Req::new("POST", "/api/bot/say")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .json_body(&body),
    );
    assert_eq!(
        (missing.status, missing.json()["error"].as_str()),
        (403, Some("missing_csrf"))
    );
    for wrong in ["AAAA", "", "not base64 !!", &l.csrf[..l.csrf.len() - 2]] {
        let r = send(
            server.addr,
            Req::new("POST", "/api/bot/say")
                .cookie(&l.cookie)
                .header("Origin", &server.origin())
                .header("X-CSRF-Token", wrong)
                .json_body(&body),
        );
        assert_eq!(
            (r.status, r.json()["error"].as_str()),
            (403, Some("bad_csrf")),
            "{wrong:?}"
        );
    }
    let other = login(&server);
    let r = send(
        server.addr,
        Req::new("POST", "/api/bot/say")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &other.csrf)
            .json_body(&body),
    );
    assert_eq!(r.status, 403, "another session's token");
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(bot.count(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_chat_line_with_a_bad_or_missing_origin_is_refused_even_with_session_and_csrf() {
    let server = TestServer::start().await;
    let bot = taking_bot(&control_socket(&server));
    let l = login(&server);
    let body = serde_json::json!({"text": "hello"});
    let cases: Vec<(&str, Vec<(&str, &str)>)> = vec![
        ("evil origin", vec![("Origin", "http://evil.example")]),
        ("null origin", vec![("Origin", "null")]),
        ("wrong port", vec![("Origin", "http://127.0.0.1:1")]),
        ("no origin at all", vec![]),
        (
            "no origin, same-origin fetch metadata",
            vec![("Sec-Fetch-Site", "same-origin")],
        ),
        ("cross-site fetch metadata", vec![("Sec-Fetch-Site", "cross-site")]),
    ];
    for (label, headers) in cases {
        let mut req = Req::new("POST", "/api/bot/say")
            .cookie(&l.cookie)
            .header("X-CSRF-Token", &l.csrf);
        for (k, v) in &headers {
            req = req.header(k, v);
        }
        let r = send(server.addr, req.json_body(&body));
        assert_eq!(
            (r.status, r.json()["error"].as_str()),
            (403, Some("cross_origin")),
            "{label}"
        );
    }
    let r = send(
        server.addr,
        Req::new("POST", "/api/bot/say")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("Sec-Fetch-Site", "cross-site")
            .header("X-CSRF-Token", &l.csrf)
            .json_body(&body),
    );
    assert_eq!(r.status, 403);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(bot.count(), 0);
    assert_eq!(say(&server, &l, &body).status, 200, "the right Origin passes");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_chat_line_must_be_json() {
    let server = TestServer::start().await;
    let bot = taking_bot(&control_socket(&server));
    let l = login(&server);
    let r = send(
        server.addr,
        Req::new("POST", "/api/bot/say")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &l.csrf)
            .form_body(&[("text", "hello")]),
    );
    assert_eq!(r.status, 415);
    assert_eq!(bot.count(), 0);
}

// ---- a valid line ---------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_valid_line_reaches_the_bot_as_a_typed_say_with_the_trimmed_text() {
    let server = TestServer::start().await;
    let bot = taking_bot(&control_socket(&server));
    let l = login(&server);
    let r = say(&server, &l, &serde_json::json!({"team": true, "text": "  gg wp  "}));
    assert_eq!(r.status, 200, "{:?}", r.json());
    let body = r.json();
    assert_eq!(body["ok"], true);
    assert!(
        !body.to_string().contains("gg wp"),
        "the answer never repeats the text: {body}"
    );
    // `team` is optional and defaults to all chat; a non-ASCII line goes through untouched
    let r = say(&server, &l, &serde_json::json!({"text": "привет, мир"}));
    assert_eq!(r.status, 200);
    let reqs = bot.requests();
    assert_eq!(reqs.len(), 2);
    let ControlCommand::Say { team, text } = &reqs[0].cmd else {
        panic!("{:?}", reqs[0])
    };
    assert!(*team);
    assert_eq!(text.as_str(), "gg wp", "trimmed before it was sent");
    let ControlCommand::Say { team, text } = &reqs[1].cmd else {
        panic!("{:?}", reqs[1])
    };
    assert!(!*team);
    assert_eq!(text.as_str(), "привет, мир");
    assert!(
        ddai_botctl::proto::valid_session_tag(&reqs[0].session),
        "the audit tag, not the cookie"
    );
    assert!(!reqs[0].session.contains(&l.cookie));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_line_the_web_refuses_never_reaches_the_bot_and_costs_no_rate_limit_slot() {
    let server = TestServer::start().await;
    let bot = taking_bot(&control_socket(&server));
    let l = login(&server);
    let too_long = "x".repeat(256);
    let cases: Vec<(serde_json::Value, &str)> = vec![
        (serde_json::json!({"text": ""}), "empty"),
        (serde_json::json!({"text": "   \t "}), "empty"),
        (serde_json::json!({"text": too_long}), "too_long"),
        (serde_json::json!({"text": "a\nb"}), "control"),
        (serde_json::json!({"text": "a\u{0}b"}), "control"),
        (serde_json::json!({"text": "a\u{1b}[31m"}), "control"),
        (serde_json::json!({"text": "\u{200B}/kill"}), "control"),
        (serde_json::json!({"text": "\u{FEFF}/w someone hi"}), "control"),
        (serde_json::json!({"text": "/w x\ny"}), "control"),
        (
            serde_json::json!({"text": format!("/w x {}", "a".repeat(255))}),
            "too_long",
        ),
        (serde_json::json!({"text": "\u{FEFF}hello"}), "control"),
        (serde_json::json!({"text": "hi \u{202E}there"}), "control"),
        (
            serde_json::json!({"text": "xd sure chillerbot.png is lyfe"}),
            "reserved",
        ),
    ];
    // far more refusals than the rate limit would allow lines: none of them takes a slot
    for _ in 0..5 {
        for (body, detail) in &cases {
            let r = say(&server, &l, body);
            assert_eq!(
                (r.status, r.json()["error"].as_str(), r.json()["detail"].as_str()),
                (400, Some("invalid_text"), Some(*detail)),
                "{body}"
            );
            assert!(!r.json().to_string().contains("kill") && !r.json().to_string().contains("someone"));
        }
    }
    // malformed bodies
    for body in [
        serde_json::json!({}),
        serde_json::json!({"team": true}),
        serde_json::json!({"text": 5}),
        serde_json::json!({"text": "hi", "team": "yes"}),
        serde_json::json!({"text": "hi", "to": "everyone"}),
        serde_json::json!([]),
    ] {
        let r = say(&server, &l, &body);
        assert_eq!(
            (r.status, r.json()["error"].as_str()),
            (400, Some("bad_request")),
            "{body}"
        );
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(bot.count(), 0, "nothing reached the bot");
    assert_eq!(
        say(&server, &l, &serde_json::json!({"text": "ok"})).status,
        200,
        "and the limit is untouched"
    );
    assert_eq!(bot.count(), 1);
}

/// Task 4.9b: a line that starts with `/` is a server command the owner typed: the route takes it like any line (trimmed, the same
/// pacing, the same checks) and hands the bot the very text, in all chat or team chat.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_server_command_is_taken_like_any_line_and_reaches_the_bot_as_typed() {
    let server = TestServer::start().await;
    let bot = taking_bot(&control_socket(&server));
    let l = login(&server);
    // two lines: the web's own burst limit is two in three seconds
    let cases = [(false, "/spec", "/spec"), (false, "  /emote happy  ", "/emote happy")];
    for (team, typed, sent) in cases {
        let r = say(&server, &l, &serde_json::json!({"team": team, "text": typed}));
        assert_eq!(r.status, 200, "{typed:?}: {:?}", r.json());
        assert_eq!(r.json()["ok"], true);
        assert!(
            !r.json().to_string().contains(sent),
            "the answer never repeats the text"
        );
        let reqs = bot.requests();
        let ControlCommand::Say { team: t, text } = &reqs.last().unwrap().cmd else {
            panic!("{reqs:?}")
        };
        assert_eq!((*t, text.as_str()), (team, sent));
    }
    assert_eq!(bot.count(), 2);
}

/// Everything else that is refused stays refused behind a slash: the trap line has no slash, but invisible characters, line breaks and
/// the length limit apply to commands as to any line.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_commands_that_are_refused_are_refused_for_the_usual_reasons() {
    let server = TestServer::start().await;
    let bot = taking_bot(&control_socket(&server));
    let l = login(&server);
    for (text, detail) in [
        ("\u{200B}/kill", "control"),
        ("/kill\u{200B}", "control"),
        ("/emote\nhappy", "control"),
        ("/w x \u{0}", "control"),
    ] {
        let r = say(&server, &l, &serde_json::json!({ "text": text }));
        assert_eq!((r.status, r.json()["detail"].as_str()), (400, Some(detail)), "{text:?}");
    }
    assert_eq!(bot.count(), 0);
}

// ---- the route's own rate limit -------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_third_quick_line_is_refused_by_the_webs_own_rate_limit_before_it_reaches_the_bot() {
    let server = TestServer::start_with(|c| c.say_burst_window = Duration::from_millis(600)).await;
    let bot = taking_bot(&control_socket(&server));
    let l = login(&server);
    assert_eq!(say(&server, &l, &serde_json::json!({"text": "one"})).status, 200);
    assert_eq!(say(&server, &l, &serde_json::json!({"text": "two"})).status, 200);
    let r = say(&server, &l, &serde_json::json!({"text": "three"}));
    assert_eq!((r.status, r.json()["error"].as_str()), (429, Some("rate_limited")));
    assert_eq!(bot.count(), 2, "the third never left the web unit");
    // after the burst window the route takes lines again
    tokio::time::sleep(Duration::from_millis(700)).await;
    assert_eq!(say(&server, &l, &serde_json::json!({"text": "four"})).status, 200);
    assert_eq!(bot.count(), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_more_lines_a_minute_than_the_limit() {
    let server = TestServer::start_with(|c| {
        c.say_burst = 100;
        c.say_max_per_minute = 4;
    })
    .await;
    let bot = taking_bot(&control_socket(&server));
    let l = login(&server);
    for i in 0..4 {
        assert_eq!(
            say(&server, &l, &serde_json::json!({"text": format!("line {i}")})).status,
            200
        );
    }
    for _ in 0..3 {
        let r = say(&server, &l, &serde_json::json!({"text": "more"}));
        assert_eq!((r.status, r.json()["error"].as_str()), (429, Some("rate_limited")));
    }
    assert_eq!(bot.count(), 4);
}

// ---- the bot's answers ----------------------------------------------------------------------------

fn refusing_bot(socket: &Path, reason: &'static str) -> FakeBot {
    FakeBot::start(socket, move |_| {
        let mut r = ControlReply::answer(false, &format!("refused: {reason}"));
        r.data = Some(serde_json::json!({ "reason": reason }));
        r
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_bots_refusals_come_back_with_their_reason_and_a_fitting_status() {
    for (reason, status) in [
        ("not_in_game", 409),
        ("chat_disabled", 409),
        ("queue_full", 429),
        ("rate_limited", 429),
    ] {
        let server = TestServer::start().await;
        let _bot = refusing_bot(&control_socket(&server), reason);
        let l = login(&server);
        let r = say(&server, &l, &serde_json::json!({"text": "hello"}));
        assert_eq!(r.status, status, "{reason}");
        let body = r.json();
        assert_eq!(
            (body["ok"].as_bool(), body["reason"].as_str()),
            (Some(false), Some(reason))
        );
        assert!(body["text"].as_str().unwrap().contains(reason));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_bot_that_is_not_there_is_a_503_and_a_bot_that_rejects_the_request_a_400() {
    let server = TestServer::start().await;
    let l = login(&server);
    let r = say(&server, &l, &serde_json::json!({"text": "hello"}));
    assert_eq!((r.status, r.json()["error"].as_str()), (503, Some("bot_unavailable")));
    let server = TestServer::start().await;
    let _bot = FakeBot::start(&control_socket(&server), |_| {
        ControlReply::refused(ReplyCode::BadRequest, "the chat line is not acceptable")
    });
    let l = login(&server);
    let r = say(&server, &l, &serde_json::json!({"text": "hello"}));
    assert_eq!(r.status, 400);
    assert_eq!(r.json()["ok"], false);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_generic_command_route_does_not_take_a_chat_line() {
    let server = TestServer::start().await;
    let bot = taking_bot(&control_socket(&server));
    let l = login(&server);
    let r = send(
        server.addr,
        Req::new("POST", "/api/bot/command")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &l.csrf)
            .json_body(&serde_json::json!({"type": "say", "team": false, "text": "hello"})),
    );
    assert_eq!((r.status, r.json()["error"].as_str()), (400, Some("use_say_route")));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(bot.count(), 0);
}

// ---- the page -------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_chat_inputs_assets_are_served_and_the_page_mounts_them() {
    let server = TestServer::start().await;
    let page = send(server.addr, Req::new("GET", "/"));
    let html = String::from_utf8(page.body).unwrap();
    assert!(html.contains(r#"src="/say.js""#) && html.contains(r#"href="/say.css""#));
    // task 5.11: the input is mounted under the chat panel of the «Игра» tab, the «Бот» tab has only a pointer to it
    assert!(html.contains(r#"id="game-say-mount""#));
    assert!(!html.contains(r#"id="say-mount""#));
    assert!(html.contains(r#"id="bot-to-chat""#));
    for (path, ctype, needle) in [
        ("/say.js", "text/javascript", "SayCard"),
        ("/say.css", "text/css", ".say-card"),
    ] {
        let r = send(server.addr, Req::new("GET", path));
        assert_eq!(r.status, 200, "{path}");
        assert!(r.header("content-type").unwrap_or("").starts_with(ctype), "{path}");
        assert!(String::from_utf8(r.body).unwrap().contains(needle), "{path}");
    }
}
