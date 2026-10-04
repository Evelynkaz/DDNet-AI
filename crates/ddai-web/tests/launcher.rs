//! The launcher routes (task 5.9, D-089): `GET|POST /api/bot/launch`. The web writes one request file (atomically) into the launch
//! directory and reads a status file; it has no other effect. Covers: session, CSRF, Origin and content type exactly as on the
//! other mutating routes (nothing is written when any fails); the request file's content and atomicity; the same choices the
//! root helper allows (and nothing else); the web's own rate limit, a pending request and a stale one; and what the page is told.

mod support;

use std::fs;
use std::path::{Path, PathBuf};

use ddai_web::launch::{Action, Brain, DurationChoice, parse_request};
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
}
