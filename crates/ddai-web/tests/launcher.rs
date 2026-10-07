//! The launcher routes (task 5.9, D-089): `GET|POST /api/bot/launch`. The web writes one request file (atomically) into the launch
//! directory and reads a status file; it has no other effect. Covers: session, CSRF, Origin and content type exactly as on the
//! other mutating routes (nothing is written when any fails); the request file's content and atomicity; the same choices the
//! root helper allows (and nothing else); the web's own rate limit, a pending request and a stale one; and what the page is told.

mod support;

use std::fs;
use std::path::{Path, PathBuf};

use ddai_web::launch::{Action, Brain, DurationChoice, Finish, Mirror, WbSmart, parse_request};
use support::{Req, TestServer, send};

struct Login {
    cookie: String,
    csrf: String,
}

fn login(server: &TestServer) -> Login {
    let (cookie, csrf) = server.login();
    Login { cookie, csrf }
}

fn launch_dir(server: &TestServer) -> PathBuf {
    server.config.launch_dir.clone()
}

fn request_file(server: &TestServer) -> PathBuf {
    launch_dir(server).join("request.json")
}

fn post(server: &TestServer, l: &Login, body: &serde_json::Value) -> support::RawResponse {
    send(
        server.addr,
        Req::new("POST", "/api/bot/launch")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &l.csrf)
            .json_body(body),
    )
}

fn get(server: &TestServer, l: &Login) -> support::RawResponse {
    send(server.addr, Req::new("GET", "/api/bot/launch").cookie(&l.cookie))
}

fn start_body() -> serde_json::Value {
    serde_json::json!({"action":"start","brain":"hybrid-fly","server":"local","duration":"15m","sparring":2})
}

/// A server that does not read the machine's own `/etc/ddnet-ai/launch.toml`.
async fn plain_server() -> TestServer {
    TestServer::start_with(|c| {
        c.launch_config = c.data_dir.join("no-such-launch.toml");
        c.status_dir = c.data_dir.join("status-dir");
        c.launch_min_gap = std::time::Duration::from_millis(0);
    })
    .await
}

/// A server whose launch directory, allow-list and bundle exist, like a deployed one.
async fn deployed() -> TestServer {
    let server = plain_server().await;
    fs::create_dir_all(launch_dir(&server)).unwrap();
    fs::create_dir_all(&server.config.status_dir).unwrap();
    let bundle = server
        .config
        .data_dir
        .join("runs/E-005/e005-fly/checkpoints/final.bundle");
    fs::create_dir_all(bundle.parent().unwrap()).unwrap();
    fs::write(bundle, b"x").unwrap();
    server
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_a_session_nothing_is_served_or_written() {
    let server = deployed().await;
    let r = send(server.addr, Req::new("GET", "/api/bot/launch"));
    assert_eq!((r.status, r.json()["error"].as_str()), (401, Some("unauthenticated")));
    let r = send(
        server.addr,
        Req::new("POST", "/api/bot/launch")
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", "AAAA")
            .json_body(&start_body()),
    );
    assert_eq!(r.status, 401);
    let forged = format!("{}=AAAA.BBBB", server.cookie_name());
    let r = send(server.addr, Req::new("GET", "/api/bot/launch").cookie(&forged));
    assert_eq!(r.status, 401);
    assert!(files_in(&launch_dir(&server)).is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn csrf_origin_and_content_type_are_checked_and_nothing_is_written() {
    let server = deployed().await;
    let l = login(&server);
    let body = start_body();
    let missing = send(
        server.addr,
        Req::new("POST", "/api/bot/launch")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .json_body(&body),
    );
    assert_eq!(
        (missing.status, missing.json()["error"].as_str()),
        (403, Some("missing_csrf"))
    );
    for wrong in ["AAAA", "", &l.csrf[..l.csrf.len() - 2]] {
        let r = send(
            server.addr,
            Req::new("POST", "/api/bot/launch")
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
    for origin in [
        Some("http://evil.example"),
        Some("null"),
        Some("http://127.0.0.1:1"),
        None,
    ] {
        let mut req = Req::new("POST", "/api/bot/launch")
            .cookie(&l.cookie)
            .header("X-CSRF-Token", &l.csrf)
            .json_body(&body);
        if let Some(o) = origin {
            req = req.header("Origin", o);
        }
        let r = send(server.addr, req);
        assert_eq!(
            (r.status, r.json()["error"].as_str()),
            (403, Some("cross_origin")),
            "{origin:?}"
        );
    }
    let r = send(
        server.addr,
        Req::new("POST", "/api/bot/launch")
            .cookie(&l.cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &l.csrf)
            .body("text/plain", serde_json::to_vec(&body).unwrap()),
    );
    assert_eq!(r.status, 415);
    assert!(
        files_in(&launch_dir(&server)).is_empty(),
        "no request was written by any refused call"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_start_writes_one_request_file_atomically_with_the_chosen_values() {
    let server = deployed().await;
    let l = login(&server);
    let r = post(&server, &l, &start_body());
    assert_eq!(r.status, 202, "{r:?}");
    let id = r.json()["id"].as_str().unwrap().to_string();
    // Exactly the request file: no temporary file is left in the directory.
    assert_eq!(files_in(&launch_dir(&server)), vec!["request.json".to_string()]);
    let req = parse_request(&fs::read(request_file(&server)).unwrap()).expect("the helper's own parser accepts it");
    assert_eq!(req.id, id);
    assert_eq!(req.action, Action::Start);
    assert_eq!(req.brain, Some(Brain::HybridFly));
    assert_eq!(req.server.as_deref(), Some("local"));
    assert_eq!(req.duration, Some(DurationChoice::M15));
    assert_eq!(req.sparring, Some(2));
    assert_eq!(
        req.mirror, None,
        "no `mirror` in the form: none in the request (the helper then keeps the model on)"
    );
    assert_eq!(
        req.finish, None,
        "no `finish` in the form: none in the request (an old request; the helper reads it as off)"
    );
    let text = String::from_utf8(fs::read(request_file(&server)).unwrap()).unwrap();
    assert!(
        !text.contains("finish") && !text.contains("wb_smart") && !text.contains("no_selfkill"),
        "an unchanged form writes an unchanged request: {text}"
    );
    assert_eq!(
        (req.wb_smart, req.no_selfkill),
        (None, None),
        "no `wb_smart` / `no_selfkill` in the form: none in the request (an old request; the helper reads off / false)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_smart_wayblock_and_the_duel_switch_reach_the_request_as_closed_values_for_every_brain() {
    // Task 5.15 (D-103/D-104, D-102).
    let server = deployed().await;
    let l = login(&server);
    let with = |brain: &str, key: &str, val: serde_json::Value| {
        let mut body = start_body();
        body["brain"] = serde_json::json!(brain);
        body[key] = val;
        body
    };
    // The pure fly takes both, like the hybrid brains. (A server per brain: the site takes at most 6 accepted requests a minute.)
    for brain in ["hybrid", "hybrid-fly", "fly"] {
        let server = deployed().await;
        let l = login(&server);
        for (word, want) in [("off", WbSmart::Off), ("on", WbSmart::On)] {
            let r = post(&server, &l, &with(brain, "wb_smart", serde_json::json!(word)));
            assert_eq!(r.status, 202, "{brain} {word}: {r:?}");
            let req =
                parse_request(&fs::read(request_file(&server)).unwrap()).expect("the helper's own parser accepts it");
            assert_eq!(
                (req.brain.is_some(), req.wb_smart, req.no_selfkill),
                (true, Some(want), None),
                "{brain} {word}"
            );
            fs::remove_file(request_file(&server)).unwrap();
        }
        for want in [false, true] {
            let r = post(&server, &l, &with(brain, "no_selfkill", serde_json::json!(want)));
            assert_eq!(r.status, 202, "{brain} {want}: {r:?}");
            let req =
                parse_request(&fs::read(request_file(&server)).unwrap()).expect("the helper's own parser accepts it");
            assert_eq!((req.wb_smart, req.no_selfkill), (None, Some(want)), "{brain} {want}");
            fs::remove_file(request_file(&server)).unwrap();
        }
    }
    // Both at once, with finishing: independent fields of one request.
    let mut body = with("hybrid", "wb_smart", serde_json::json!("on"));
    body["no_selfkill"] = serde_json::json!(true);
    body["finish"] = serde_json::json!("target");
    assert_eq!(post(&server, &l, &body).status, 202);
    let req = parse_request(&fs::read(request_file(&server)).unwrap()).unwrap();
    assert_eq!(
        (req.wb_smart, req.no_selfkill, req.finish),
        (Some(WbSmart::On), Some(true), Some(Finish::Target))
    );
    fs::remove_file(request_file(&server)).unwrap();
    // Anything outside the closed values, and either on a stop: refused, nothing written (the injection attempts included).
    for (key, bad) in [
        ("wb_smart", serde_json::json!("true")),
        ("wb_smart", serde_json::json!("On")),
        ("wb_smart", serde_json::json!(true)),
        ("wb_smart", serde_json::json!(1)),
        ("wb_smart", serde_json::json!("on --report /etc/passwd")),
        ("wb_smart", serde_json::json!("on\nBOT_SERVER=\"203.0.113.5:8308\"")),
        ("wb_smart", serde_json::json!("$(id)")),
        ("no_selfkill", serde_json::json!("true")),
        ("no_selfkill", serde_json::json!("on")),
        ("no_selfkill", serde_json::json!(1)),
        ("no_selfkill", serde_json::json!("true\nBOT_NAME=\"evil\"")),
        ("no_selfkill", serde_json::json!("$(id)")),
    ] {
        let r = post(&server, &l, &with("hybrid", key, bad.clone()));
        assert_eq!(
            (r.status, r.json()["error"].as_str()),
            (400, Some("bad_request")),
            "{key}={bad}"
        );
    }
    for (key, val) in [
        ("wb_smart", serde_json::json!("off")),
        ("no_selfkill", serde_json::json!(false)),
    ] {
        let r = post(&server, &l, &serde_json::json!({"action":"stop", key: val}));
        assert_eq!(
            (r.status, r.json()["error"].as_str()),
            (400, Some("bad_request")),
            "{key}"
        );
    }
    assert!(
        files_in(&launch_dir(&server)).is_empty(),
        "a refused call writes nothing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_finishing_switch_reaches_the_request_as_one_of_three_words_and_the_pure_fly_takes_only_off() {
    // Task 5.13 (D-097).
    let server = deployed().await;
    let l = login(&server);
    let with = |brain: &str, finish: serde_json::Value| {
        let mut body = start_body();
        body["brain"] = serde_json::json!(brain);
        body["finish"] = finish;
        body
    };
    for (brain, word, want) in [
        ("hybrid", "off", Finish::Off),
        ("hybrid", "target", Finish::Target),
        ("hybrid-fly", "target", Finish::Target),
        ("hybrid-fly", "full", Finish::Full),
        ("fly", "off", Finish::Off),
    ] {
        let r = post(&server, &l, &with(brain, serde_json::json!(word)));
        assert_eq!(r.status, 202, "{brain} {word}: {r:?}");
        let req = parse_request(&fs::read(request_file(&server)).unwrap()).expect("the helper's own parser accepts it");
        assert_eq!((req.brain.is_some(), req.finish), (true, Some(want)), "{brain} {word}");
        fs::remove_file(request_file(&server)).unwrap();
    }
    // The pure fly with finishing on: refused up front, nothing written.
    for word in ["target", "full"] {
        let r = post(&server, &l, &with("fly", serde_json::json!(word)));
        assert_eq!(
            (r.status, r.json()["error"].as_str()),
            (400, Some("finish_hybrid_only")),
            "{word}"
        );
    }
    // Anything outside the three words, and a finishing on a stop: refused, nothing written (the injection attempts included).
    for bad in [
        serde_json::json!("on"),
        serde_json::json!("Target"),
        serde_json::json!("TARGET"),
        serde_json::json!(true),
        serde_json::json!(1),
        serde_json::json!("target --report /etc/passwd"),
        serde_json::json!("target\nBOT_SERVER=\"203.0.113.5:8308\""),
        serde_json::json!("$(id)"),
    ] {
        let r = post(&server, &l, &with("hybrid", bad.clone()));
        assert_eq!(
            (r.status, r.json()["error"].as_str()),
            (400, Some("bad_request")),
            "{bad}"
        );
    }
    let r = post(&server, &l, &serde_json::json!({"action":"stop","finish":"off"}));
    assert_eq!((r.status, r.json()["error"].as_str()), (400, Some("bad_request")));
    assert!(
        files_in(&launch_dir(&server)).is_empty(),
        "a refused call writes nothing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_page_gets_the_finishing_mode_of_the_helpers_status_and_an_old_status_has_none() {
    let server = deployed().await;
    let l = login(&server);
    let status = server.config.status_dir.join("status.json");
    fs::write(
        &status,
        r#"{"v":1,"state":"started","at":5,"request_id":"0123456789abcdef","brain":"hybrid","server":"local","duration":"15m","sparring":0,"finish":"target"}"#,
    )
    .unwrap();
    assert_eq!(get(&server, &l).json()["status"]["finish"], "target");
    // A status written before the field existed has none (the card then says nothing about finishing).
    fs::write(
        &status,
        r#"{"v":1,"state":"started","at":5,"brain":"hybrid","server":"local","duration":"15m","sparring":0}"#,
    )
    .unwrap();
    let j = get(&server, &l).json();
    assert_eq!(j["status"]["state"], "started");
    assert!(j["status"].get("finish").is_none(), "{j}");
    // A word outside the list makes the whole status unreadable, never a mode of its own.
    fs::write(&status, r#"{"v":1,"state":"started","at":5,"finish":"turbo"}"#).unwrap();
    assert_eq!(get(&server, &l).json()["status"], serde_json::Value::Null);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_page_gets_the_smart_wayblock_and_the_duel_switch_of_the_helpers_status_and_an_old_status_has_none() {
    // Task 5.15.
    let server = deployed().await;
    let l = login(&server);
    let status = server.config.status_dir.join("status.json");
    fs::write(
        &status,
        r#"{"v":1,"state":"started","at":5,"request_id":"0123456789abcdef","brain":"hybrid","server":"local","duration":"15m","sparring":0,"finish":"off","wb_smart":"on","no_selfkill":true}"#,
    )
    .unwrap();
    let j = get(&server, &l).json();
    assert_eq!(
        (j["status"]["wb_smart"].as_str(), j["status"]["no_selfkill"].as_bool()),
        (Some("on"), Some(true)),
        "{j}"
    );
    // A status written before the fields existed has none (the card then says nothing about them).
    fs::write(
        &status,
        r#"{"v":1,"state":"started","at":5,"brain":"hybrid","server":"local","duration":"15m","sparring":0,"finish":"off"}"#,
    )
    .unwrap();
    let j = get(&server, &l).json();
    assert_eq!(j["status"]["state"], "started");
    assert!(
        j["status"].get("wb_smart").is_none() && j["status"].get("no_selfkill").is_none(),
        "{j}"
    );
    // A value outside the closed list makes the whole status unreadable, never a mode of its own.
    for bad in [r#""wb_smart":"turbo""#, r#""no_selfkill":"yes""#] {
        fs::write(&status, format!(r#"{{"v":1,"state":"started","at":5,{bad}}}"#)).unwrap();
        assert_eq!(get(&server, &l).json()["status"], serde_json::Value::Null, "{bad}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_opponent_model_switch_reaches_the_request_and_only_as_on_or_off() {
    // Task 3.7b F9 (D-090).
    let server = deployed().await;
    let l = login(&server);
    let mut body = start_body();
    body["mirror"] = serde_json::json!("off");
    let r = post(&server, &l, &body);
    assert_eq!(r.status, 202, "{r:?}");
    let req = parse_request(&fs::read(request_file(&server)).unwrap()).unwrap();
    assert_eq!(req.mirror, Some(Mirror::Off));
    fs::remove_file(request_file(&server)).unwrap();
    for bad in [
        serde_json::json!("maybe"),
        serde_json::json!("OFF"),
        serde_json::json!(false),
    ] {
        let mut body = start_body();
        body["mirror"] = bad.clone();
        let r = post(&server, &l, &body);
        assert_eq!(
            (r.status, r.json()["error"].as_str()),
            (400, Some("bad_request")),
            "{bad}"
        );
    }
    assert!(
        files_in(&launch_dir(&server)).is_empty(),
        "a refused call writes nothing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stop_is_a_bare_request_and_a_stop_with_extras_is_refused() {
    let server = deployed().await;
    let l = login(&server);
    let r = post(&server, &l, &serde_json::json!({"action":"stop","server":"local"}));
    assert_eq!((r.status, r.json()["error"].as_str()), (400, Some("bad_request")));
    assert!(files_in(&launch_dir(&server)).is_empty());
    let r = post(&server, &l, &serde_json::json!({"action":"stop"}));
    assert_eq!(r.status, 202);
    assert_eq!(
        parse_request(&fs::read(request_file(&server)).unwrap()).unwrap().action,
        Action::Stop
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn only_the_choices_the_helper_allows_get_through() {
    let server = deployed().await;
    fs::write(
        &server.config.live_servers,
        "[[server]]\naddress=\"203.0.113.5:8308\"\nnick=\"Muha\"\nready=true\nproxy=\"swarfey\"\n[[server]]\naddress=\"203.0.113.6:8308\"\nnick=\"Muha\"\nready=false\n",
    )
    .unwrap();
    let l = login(&server);
    let bad: Vec<(serde_json::Value, &str)> = vec![
        (
            serde_json::json!({"action":"start","brain":"hybrid","server":"203.0.113.6:8308","duration":"15m"}),
            "server_not_allowed",
        ),
        (
            serde_json::json!({"action":"start","brain":"hybrid","server":"9.9.9.9:8308","duration":"15m"}),
            "server_not_allowed",
        ),
        (
            serde_json::json!({"action":"start","brain":"hybrid","server":"203.0.113.5:8308","duration":"15m","sparring":1}),
            "sparring_local_only",
        ),
        (
            serde_json::json!({"action":"start","brain":"hybrid","server":"local","duration":"15m","sparring":9}),
            "bad_request",
        ),
        (
            serde_json::json!({"action":"start","brain":"planner","server":"local","duration":"15m"}),
            "bad_request",
        ),
        (
            serde_json::json!({"action":"start","brain":"hybrid","server":"local","duration":"15m","extra":1}),
            "bad_request",
        ),
        (
            serde_json::json!({"action":"start","brain":"hybrid","duration":"15m"}),
            "bad_request",
        ),
        (serde_json::json!({"action":"restart"}), "bad_request"),
    ];
    for (body, code) in bad {
        let r = post(&server, &l, &body);
        assert_eq!((r.status, r.json()["error"].as_str()), (400, Some(code)), "{body}");
        assert!(files_in(&launch_dir(&server)).is_empty(), "{body}");
    }
    // A ready entry is a valid target.
    let r = post(
        &server,
        &l,
        &serde_json::json!({"action":"start","brain":"hybrid","server":"203.0.113.5:8308","duration":"unlimited"}),
    );
    assert_eq!(r.status, 202, "{r:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_missing_bundle_refuses_the_fly_brains_and_a_missing_launch_dir_means_unavailable() {
    let server = plain_server().await;
    let l = login(&server);
    let r = post(&server, &l, &start_body());
    assert_eq!(
        (r.status, r.json()["error"].as_str()),
        (503, Some("launcher_unavailable"))
    );
    let g = get(&server, &l);
    assert_eq!(g.status, 200);
    assert_eq!(g.json()["enabled"], false);
    fs::create_dir_all(launch_dir(&server)).unwrap();
    let r = post(&server, &l, &start_body());
    assert_eq!((r.status, r.json()["error"].as_str()), (400, Some("bundle_missing")));
    let r = post(
        &server,
        &l,
        &serde_json::json!({"action":"start","brain":"hybrid","server":"local","duration":"15m"}),
    );
    assert_eq!(r.status, 202);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waiting_request_is_not_overwritten_and_a_stale_one_is_removed_and_reported_as_a_dead_launcher() {
    let server = deployed().await;
    let l = login(&server);
    assert_eq!(post(&server, &l, &start_body()).status, 202);
    let r = post(&server, &l, &start_body());
    assert_eq!(
        (r.status, r.json()["error"].as_str()),
        (409, Some("pending")),
        "not consumed yet"
    );

    // A request somebody else left waiting (and nobody consumed) is a conflict.
    let server = deployed().await;
    let l = login(&server);
    fs::write(request_file(&server), b"{}").unwrap();
    let r = post(&server, &l, &start_body());
    assert_eq!((r.status, r.json()["error"].as_str()), (409, Some("pending")));
    assert_eq!(fs::read(request_file(&server)).unwrap(), b"{}", "left untouched");
    let g = get(&server, &l).json();
    assert_eq!(g["pending"], true);
    assert_eq!(g["launcher_down"], false);
    // After REQUEST_STALE_SECS nobody has consumed it (the path unit is not working): the web removes what it left behind and says
    // the launcher is down, until the helper writes a newer status.
    let file = fs::OpenOptions::new().write(true).open(request_file(&server)).unwrap();
    file.set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(120))
        .unwrap();
    let g = get(&server, &l).json();
    assert_eq!(
        (g["pending"].as_bool(), g["launcher_down"].as_bool()),
        (Some(false), Some(true))
    );
    assert!(!request_file(&server).exists(), "the stale request is gone");
    // A status older than the stall does not clear it; a newer one does.
    let status = server.config.status_dir.join("status.json");
    fs::write(&status, r#"{"v":1,"state":"stopped","at":1}"#).unwrap();
    assert_eq!(get(&server, &l).json()["launcher_down"], true);
    fs::write(
        &status,
        format!(
            r#"{{"v":1,"state":"stopped","at":{}}}"#,
            ddai_web::launch::unix_now() + 1
        ),
    )
    .unwrap();
    assert_eq!(get(&server, &l).json()["launcher_down"], false);
    assert_eq!(post(&server, &l, &start_body()).status, 202);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_web_stays_well_below_the_path_units_trigger_limit() {
    // Default limits: at most 6 requests in a minute (the unit's TriggerLimitBurst is 10), at least 2 s apart.
    let server = TestServer::start_with(|c| {
        c.launch_config = c.data_dir.join("no-such-launch.toml");
        c.status_dir = c.data_dir.join("status-dir");
        c.launch_min_gap = std::time::Duration::from_millis(0);
    })
    .await;
    fs::create_dir_all(launch_dir(&server)).unwrap();
    let l = login(&server);
    let stop = serde_json::json!({"action":"stop"});
    let mut accepted = 0;
    for _ in 0..10 {
        let r = post(&server, &l, &stop);
        if r.status == 202 {
            accepted += 1;
            // The helper consumes the request.
            fs::remove_file(request_file(&server)).unwrap();
        } else {
            assert_eq!((r.status, r.json()["error"].as_str()), (429, Some("rate_limited")));
        }
    }
    assert_eq!(accepted, 6);
    let server_gap = TestServer::start_with(|c| c.launch_config = c.data_dir.join("none.toml")).await;
    let lg = login(&server_gap);
    fs::create_dir_all(launch_dir(&server_gap)).unwrap();
    assert_eq!(post(&server_gap, &lg, &stop).status, 202);
    fs::remove_file(request_file(&server_gap)).unwrap();
    let r = post(&server_gap, &lg, &stop);
    assert_eq!(
        (r.status, r.json()["error"].as_str()),
        (429, Some("rate_limited")),
        "min gap 2 s"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_page_is_told_the_choices_the_bundle_by_run_name_and_the_helpers_status() {
    let server = deployed().await;
    fs::write(
        &server.config.live_servers,
        "[[server]]\naddress=\"203.0.113.5:8308\"\nnick=\"Muha\"\nready=true\n[[server]]\naddress=\"203.0.113.6:8308\"\nnick=\"Muha\"\n",
    )
    .unwrap();
    fs::write(
        server.config.status_dir.join("status.json"),
        r#"{"v":1,"state":"failed","at":1,"reason":"kicked_or_banned","exit_code":3,"server":"203.0.113.5:8308"}"#,
    )
    .unwrap();
    // A status in the web's own directory is not the helper's: ignored.
    fs::write(
        launch_dir(&server).join("status.json"),
        r#"{"v":1,"state":"started","at":9}"#,
    )
    .unwrap();
    let l = login(&server);
    let g = get(&server, &l);
    assert_eq!(g.status, 200);
    let j = g.json();
    assert_eq!(j["enabled"], true);
    let ids: Vec<&str> = j["servers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        vec!["local", "203.0.113.5:8308"],
        "local, and only the ready entries"
    );
    assert_eq!(j["brains"], serde_json::json!(["hybrid", "hybrid-fly", "fly"]));
    assert_eq!(j["max_sparring"], 3);
    assert_eq!(j["bundle"], "E-005/e005-fly");
    assert_eq!(j["bundle_present"], true);
    assert_eq!(j["status"]["reason"], "kicked_or_banned");
    assert_eq!(j["status"]["exit_code"], 3);
    assert_eq!(j["pending"], false);
    // The full path of the bundle is never sent.
    assert!(!g.json().to_string().contains(server.config.data_dir.to_str().unwrap()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_symlinked_or_garbage_status_file_is_not_shown() {
    let server = deployed().await;
    let l = login(&server);
    let victim = server.config.data_dir.join("victim.json");
    fs::write(&victim, r#"{"v":1,"state":"started","at":1}"#).unwrap();
    let status = server.config.status_dir.join("status.json");
    std::os::unix::fs::symlink(&victim, &status).unwrap();
    assert_eq!(get(&server, &l).json()["status"], serde_json::Value::Null);
    fs::remove_file(&status).unwrap();
    fs::write(&status, b"garbage").unwrap();
    assert_eq!(get(&server, &l).json()["status"], serde_json::Value::Null);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_cards_script_and_styles_are_served_and_linked_from_the_page() {
    let server = plain_server().await;
    let page = send(server.addr, Req::new("GET", "/"));
    let html = String::from_utf8_lossy(&page.body).into_owned();
    assert!(html.contains(r#"src="/launch.js""#) && html.contains(r#"href="/launch.css""#));
    assert!(html.contains(r#"id="launch-mount""#));
    for (path, ctype) in [("/launch.js", "text/javascript"), ("/launch.css", "text/css")] {
        let r = send(server.addr, Req::new("GET", path));
        assert_eq!(r.status, 200, "{path}");
        assert!(r.header("content-type").is_some_and(|c| c.starts_with(ctype)), "{path}");
        assert!(!r.body.is_empty());
    }
    // Task 5.13: the «Бот» card has a row for the finishing mode, and the launch card's script offers the three words.
    assert!(html.contains(r#"id="bs-finish""#));
    let js = String::from_utf8_lossy(&send(server.addr, Req::new("GET", "/launch.js")).body).into_owned();
    for needle in [
        "Дожим",
        "цель (рекомендуется)",
        "полный (не рекомендуется)",
        "finish_hybrid_only",
        // Task 5.15: the two more switches, their hints and the fields they send.
        "Умный ВБ",
        "Без самоубийств (дуэль)",
        "только если он мешает",
        "Для 1vs1 F-DDrace: любая смерть бота даёт очко сопернику",
        "На обычных серверах не включать",
        "кнопкой «Убить» в «Командах» (или строкой /kill на вкладке «Игра»)",
        "не бродит, а сразу начинает путь",
        "body.wb_smart",
        "body.no_selfkill",
    ] {
        assert!(js.contains(needle), "launch.js lacks {needle}");
    }
    assert!(html.contains(r#"id="bs-wbsmart""#));
}
