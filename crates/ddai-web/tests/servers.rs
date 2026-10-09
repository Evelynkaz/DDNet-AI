//! The server browser's routes (task 5.12, D-099): the cached master list, the owner's favourites, the proxy profiles, «Проверить» and
//! «Обновить», and the launcher card's new choice. Covers: session, strict Origin, CSRF, JSON and a rate limit on every mutating route
//! (nothing is written when any fails); malicious addresses, names, nicks, proxy fields and unknown fields refused with nothing
//! written; the owner's consent; ban -> closed until the explicit re-open; **the password never in any response or log line**; and that
//! the web only writes the files it is meant to.

// Task 5.5a: the server browser's refresh trigger and proxy files use POSIX permission bits and symlinks (VPS launcher deployment, D-089, D-099).
#![cfg(unix)]

mod support;

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ddai_client::favourites::{Favourites, Rules};
use ddai_client::server_list::{CACHE_FILE, MasterCache, parse_master};
use ddai_web::serverbrowser::{
    BLOCKED_FILE, BlockedEntry, BlockedFile, PROXY_CHECK_REQUEST_FILE, PROXY_CHECK_RESULT_FILE, REFRESH_TRIGGER_FILE,
    parse_proxy_check_request,
};
use support::{Req, TestServer, send};

// ---- capturing every log line of the test process, to prove no secret reaches the log ----

#[derive(Clone)]
struct LogSink(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for LogSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn logs() -> &'static Arc<Mutex<Vec<u8>>> {
    static LOGS: OnceLock<Arc<Mutex<Vec<u8>>>> = OnceLock::new();
    LOGS.get_or_init(|| {
        let sink = Arc::new(Mutex::new(Vec::new()));
        let writer = LogSink(sink.clone());
        let _ = tracing::subscriber::set_global_default(
            tracing_subscriber::fmt()
                .with_writer(move || writer.clone())
                .with_max_level(tracing::Level::TRACE)
                .finish(),
        );
        sink
    })
}

fn log_text() -> String {
    String::from_utf8_lossy(&logs().lock().unwrap()).into_owned()
}

// ---- harness ----

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

fn err(r: &support::RawResponse) -> (u16, Option<String>) {
    (r.status, r.json()["error"].as_str().map(str::to_string))
}

fn dirs(server: &TestServer) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let d = &server.config.data_dir;
    (
        d.join("launch"),
        d.join("secrets"),
        server.config.status_dir.clone(),
        server.config.servers_dir.clone(),
    )
}

/// A deployed-like server: the directories exist, the limits are loose, the rules are the production ones (no loopback).
async fn deployed() -> TestServer {
    deployed_with(|_| {}).await
}

async fn deployed_with(customize: impl FnOnce(&mut ddai_web::WebConfig)) -> TestServer {
    let _ = logs();
    let server = TestServer::start_with(|c| {
        c.launch_config = c.data_dir.join("no-such-launch.toml");
        c.status_dir = c.data_dir.join("status-dir");
        c.launch_min_gap = std::time::Duration::from_millis(0);
        c.proxycheck_min_gap = std::time::Duration::from_millis(0);
        c.refresh_min_gap = std::time::Duration::from_millis(0);
        c.servers_edits_per_minute = 1000;
        c.proxycheck_max_per_minute = 1000;
        c.favourite_rules = Rules::default();
        customize(c);
    })
    .await;
    let (launch, _, status, servers) = dirs(&server);
    fs::create_dir_all(&launch).unwrap();
    fs::create_dir_all(&status).unwrap();
    fs::create_dir_all(&servers).unwrap();
    // `secrets/` exists after `web-passwd`; the test server made it for the password.
    assert!(server.config.data_dir.join("secrets").is_dir());
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

fn add_body(address: &str) -> serde_json::Value {
    serde_json::json!({"address": address, "name": "Some Block Server", "consent": true})
}

fn favourites_on_disk(server: &TestServer) -> Favourites {
    let bytes = fs::read(dirs(server).0.join("favourites.json")).unwrap();
    Favourites::parse(&bytes, Rules::default()).expect("the file the web wrote passes the strict parser")
}

const PUB1: &str = "93.184.216.35:8308";
const PUB2: &str = "93.184.216.34:8303";

fn proxy_body(name: &str) -> serde_json::Value {
    serde_json::json!({"create": true, "name": name, "host": "93.184.216.34", "port": 1080, "user": "plainuser", "pass": PASS})
}

const PASS: &str = "p-s3cr3t-pass-VALUE";

fn write_blocked(server: &TestServer, entries: &[(&str, u64)]) {
    let file = BlockedFile {
        v: 1,
        at: 5,
        blocked: entries
            .iter()
            .map(|(a, at)| BlockedEntry {
                address: (*a).to_string(),
                at: *at,
                code: 3,
            })
            .collect(),
    };
    fs::write(dirs(server).2.join(BLOCKED_FILE), serde_json::to_vec(&file).unwrap()).unwrap();
}

// ---- auth, CSRF, Origin, content type, rate limit: on every route ----

const GETS: [&str; 3] = ["/api/servers", "/api/favourites", "/api/proxies"];
const POSTS: [&str; 8] = [
    "/api/servers/refresh",
    "/api/favourites/add",
    "/api/favourites/update",
    "/api/favourites/remove",
    "/api/favourites/reopen",
    "/api/proxies/save",
    "/api/proxies/remove",
    "/api/proxies/check",
];

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_a_session_nothing_is_served_or_written() {
    let server = deployed().await;
    for path in GETS {
        let r = send(server.addr, Req::new("GET", path));
        assert_eq!(err(&r), (401, Some("unauthenticated".into())), "{path}");
    }
    for path in POSTS {
        let r = send(
            server.addr,
            Req::new("POST", path)
                .header("Origin", &server.origin())
                .header("X-CSRF-Token", "AAAA")
                .json_body(&add_body(PUB1)),
        );
        assert_eq!(r.status, 401, "{path}");
    }
    let forged = format!("{}=AAAA.BBBB", server.cookie_name());
    assert_eq!(
        send(server.addr, Req::new("GET", "/api/servers").cookie(&forged)).status,
        401
    );
    let (launch, secrets, _, servers) = dirs(&server);
    assert!(files_in(&launch).is_empty() && files_in(&servers).is_empty());
    assert!(!files_in(&secrets).iter().any(|f| f.contains("proxy")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn csrf_origin_and_content_type_are_checked_on_every_post_and_nothing_is_written() {
    let server = deployed().await;
    let l = login(&server);
    for path in POSTS {
        let body = serde_json::json!({});
        let missing = send(
            server.addr,
            Req::new("POST", path)
                .cookie(&l.cookie)
                .header("Origin", &server.origin())
                .json_body(&body),
        );
        assert_eq!(err(&missing), (403, Some("missing_csrf".into())), "{path}");
        for wrong in ["AAAA", ""] {
            let r = send(
                server.addr,
                Req::new("POST", path)
                    .cookie(&l.cookie)
                    .header("Origin", &server.origin())
                    .header("X-CSRF-Token", wrong)
                    .json_body(&body),
            );
            assert_eq!(err(&r), (403, Some("bad_csrf".into())), "{path} {wrong:?}");
        }
        for origin in [
            Some("http://evil.example"),
            Some("null"),
            Some("http://127.0.0.1:1"),
            None,
        ] {
            let mut req = Req::new("POST", path)
                .cookie(&l.cookie)
                .header("X-CSRF-Token", &l.csrf)
                .json_body(&body);
            if let Some(o) = origin {
                req = req.header("Origin", o);
            }
            let r = send(server.addr, req);
            assert_eq!(err(&r), (403, Some("cross_origin".into())), "{path} {origin:?}");
        }
        let r = send(
            server.addr,
            Req::new("POST", path)
                .cookie(&l.cookie)
                .header("Origin", &server.origin())
                .header("X-CSRF-Token", &l.csrf)
                .body("text/plain", b"{}".to_vec()),
        );
        assert_eq!(r.status, 415, "{path}");
    }
    let (launch, secrets, _, servers) = dirs(&server);
    assert!(
        files_in(&launch).is_empty() && files_in(&servers).is_empty(),
        "no file was written by a refused call"
    );
    assert!(!files_in(&secrets).iter().any(|f| f.contains("proxy")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_mutating_route_has_its_own_rate_limit_and_every_attempt_counts() {
    let server = deployed_with(|c| c.servers_edits_per_minute = 4).await;
    let l = login(&server);
    // Invalid attempts use up the budget just as valid ones do.
    for i in 0..4 {
        let r = post(
            &server,
            &l,
            "/api/favourites/add",
            &serde_json::json!({"address": "nope", "name": "x", "consent": true}),
        );
        assert_eq!(r.status, 400, "attempt {i}");
    }
    for path in [
        "/api/favourites/add",
        "/api/favourites/update",
        "/api/favourites/remove",
        "/api/favourites/reopen",
        "/api/proxies/save",
        "/api/proxies/remove",
    ] {
        let r = post(&server, &l, path, &serde_json::json!({}));
        assert_eq!(err(&r), (429, Some("rate_limited".into())), "{path}");
    }
    // The check and the refresh have gates of their own.
    let server = deployed_with(|c| {
        c.proxycheck_min_gap = std::time::Duration::from_secs(60);
        c.refresh_min_gap = std::time::Duration::from_secs(60);
    })
    .await;
    let l = login(&server);
    assert_eq!(post(&server, &l, "/api/proxies/save", &proxy_body("hp")).status, 200);
    assert_eq!(
        post(&server, &l, "/api/proxies/check", &serde_json::json!({"name": "hp"})).status,
        202
    );
    fs::remove_file(dirs(&server).0.join(PROXY_CHECK_REQUEST_FILE)).unwrap();
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/proxies/check",
            &serde_json::json!({"name": "hp"})
        )),
        (429, Some("rate_limited".into()))
    );
    assert_eq!(
        post(&server, &l, "/api/servers/refresh", &serde_json::json!({})).status,
        202
    );
    assert_eq!(
        err(&post(&server, &l, "/api/servers/refresh", &serde_json::json!({}))),
        (429, Some("rate_limited".into()))
    );
}

// ---- the server list ----

fn write_cache(server: &TestServer, fetched_at: u64) -> MasterCache {
    let rows = parse_master(include_str!("../../ddai-client/tests/fixtures/master-servers.json")).unwrap();
    let cache = MasterCache::from_rows(&rows, fetched_at, 1);
    fs::write(dirs(server).3.join(CACHE_FILE), cache.to_bytes().unwrap()).unwrap();
    cache
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_list_comes_from_the_cache_and_a_bad_cache_is_not_shown() {
    let server = deployed().await;
    let l = login(&server);
    let r = get(&server, &l, "/api/servers");
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["problem"], "missing");
    assert_eq!(r.json()["servers"].as_array().unwrap().len(), 0);
    let cache = write_cache(&server, now() - 30);
    let r = get(&server, &l, "/api/servers");
    let j = r.json();
    assert_eq!(j["problem"], serde_json::Value::Null);
    assert_eq!(j["servers"].as_array().unwrap().len(), cache.servers.len());
    assert!(j["age_s"].as_u64().unwrap() >= 30 && j["age_s"].as_u64().unwrap() < 60);
    let row = j["servers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["address"] == PUB1)
        .unwrap();
    assert_eq!(
        (row["map"].as_str(), row["block"].as_bool(), row["v06"].as_bool()),
        (Some("Copy Love Box"), Some(true), Some(true))
    );
    // No client names ever: the cache holds counts only.
    assert!(!r.json().to_string().contains("SECRETNICK"));
    // A tampered cache (a private address) is refused as a whole.
    let mut v: serde_json::Value =
        serde_json::from_slice(&fs::read(dirs(&server).3.join(CACHE_FILE)).unwrap()).unwrap();
    v["servers"][0]["address"] = serde_json::json!("10.0.0.1:8303");
    fs::write(dirs(&server).3.join(CACHE_FILE), v.to_string()).unwrap();
    let j = get(&server, &l, "/api/servers").json();
    assert_eq!(j["problem"], "invalid");
    assert_eq!(j["servers"].as_array().unwrap().len(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refresh_rewrites_the_trigger_file_only_and_a_fresh_cache_is_not_asked_for_again() {
    let server = deployed().await;
    let l = login(&server);
    let r = post(&server, &l, "/api/servers/refresh", &serde_json::json!({}));
    assert_eq!((r.status, r.json()["asked"].as_bool()), (202, Some(true)));
    assert_eq!(files_in(&dirs(&server).0), vec![REFRESH_TRIGGER_FILE.to_string()]);
    assert!(
        dirs(&server).3.read_dir().unwrap().next().is_none(),
        "the web writes nothing into the cache directory"
    );
    // A body with anything in it is refused.
    assert_eq!(
        post(
            &server,
            &l,
            "/api/servers/refresh",
            &serde_json::json!({"url": "http://x"})
        )
        .status,
        400
    );
    // A cache from a few seconds ago: not asked for again.
    write_cache(&server, now() - 5);
    fs::remove_file(dirs(&server).0.join(REFRESH_TRIGGER_FILE)).unwrap();
    let r = post(&server, &l, "/api/servers/refresh", &serde_json::json!({}));
    assert_eq!((r.status, r.json()["asked"].as_bool()), (202, Some(false)));
    assert!(files_in(&dirs(&server).0).is_empty());
    // A symlink where the trigger goes is not followed.
    write_cache(&server, now() - 500);
    let victim = server.config.data_dir.join("victim");
    fs::write(&victim, "keep").unwrap();
    std::os::unix::fs::symlink(&victim, dirs(&server).0.join(REFRESH_TRIGGER_FILE)).unwrap();
    let r = post(&server, &l, "/api/servers/refresh", &serde_json::json!({}));
    assert_eq!(r.status, 503);
    assert_eq!(fs::read_to_string(&victim).unwrap(), "keep");
}

// ---- favourites ----

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_favourite_needs_the_owners_consent_and_is_stored_strictly() {
    let server = deployed().await;
    let l = login(&server);
    let mut body = add_body(PUB1);
    body["consent"] = serde_json::json!(false);
    assert_eq!(
        err(&post(&server, &l, "/api/favourites/add", &body)),
        (400, Some("consent_required".into()))
    );
    let mut body = add_body(PUB1);
    body.as_object_mut().unwrap().remove("consent");
    assert_eq!(post(&server, &l, "/api/favourites/add", &body).status, 400);
    assert!(files_in(&dirs(&server).0).is_empty());

    let mut body = add_body(PUB1);
    body["notes"] = serde_json::json!("admin said yes on Discord");
    body["nick"] = serde_json::json!("Muha");
    let r = post(&server, &l, "/api/favourites/add", &body);
    assert_eq!(r.status, 201, "{r:?}");
    let list = favourites_on_disk(&server);
    assert_eq!(list.favourites.len(), 1);
    let f = &list.favourites[0];
    assert_eq!(
        (f.address.as_str(), f.nick.as_str(), f.connection.as_str()),
        (PUB1, "Muha", "direct")
    );
    assert!(f.consent_at >= now() - 5 && f.reopened_at == 0, "{f:?}");
    assert_eq!(files_in(&dirs(&server).0), vec!["favourites.json".to_string()]);
    // The page sees it.
    let j = get(&server, &l, "/api/favourites").json();
    assert_eq!(j["favourites"][0]["address"], PUB1);
    assert_eq!(j["favourites"][0]["blocked"], serde_json::Value::Null);
    // Twice is a conflict; so is an address the owner's allow-list already names.
    assert_eq!(
        err(&post(&server, &l, "/api/favourites/add", &add_body(PUB1))),
        (409, Some("duplicate".into()))
    );
    fs::write(
        &server.config.live_servers,
        "[[server]]\naddress=\"93.184.216.34:8303\"\nnick=\"Muha\"\nready=false\n",
    )
    .unwrap();
    assert_eq!(
        err(&post(&server, &l, "/api/favourites/add", &add_body(PUB2))),
        (409, Some("duplicate".into()))
    );
    assert_eq!(favourites_on_disk(&server).favourites.len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn malicious_addresses_names_nicks_connections_and_extra_fields_are_refused_and_nothing_is_written() {
    let server = deployed().await;
    let l = login(&server);
    let bad_addresses = [
        "",
        "localhost:8303",
        "127.0.0.1:8303",
        "[::1]:8303",
        "10.0.0.5:8303",
        "192.168.1.1:8303",
        "169.254.169.254:80",
        "0.0.0.0:8303",
        "example.com:8303",
        "93.184.216.35",
        "93.184.216.35:0",
        "93.184.216.35:70000",
        "93.184.216.35:8308 ",
        " 93.184.216.35:8308",
        "93.184.216.35:8308\n",
        "93.184.216.35:8308/../x",
        "http://93.184.216.35:8308",
        "[::ffff:93.184.216.35]:8308",
        "093.184.216.035:8308",
        "203.0.113.5:8308",
        "100.64.0.1:8303",
        "255.255.255.255:8303",
    ];
    for a in bad_addresses {
        let r = post(&server, &l, "/api/favourites/add", &add_body(a));
        assert_eq!(err(&r), (400, Some("bad_address".into())), "{a:?}");
    }
    for (field, value, code) in [
        ("name", serde_json::json!(""), "bad_name"),
        ("name", serde_json::json!("a\nb"), "bad_name"),
        ("name", serde_json::json!("a\u{202E}b"), "bad_name"),
        ("name", serde_json::json!("x".repeat(65)), "bad_name"),
        ("nick", serde_json::json!("Mu ha"), "bad_nick"),
        ("nick", serde_json::json!("../etc"), "bad_nick"),
        ("nick", serde_json::json!("a;b"), "bad_nick"),
        ("nick", serde_json::json!("AVeryLongNickname123"), "bad_nick"),
        ("nick", serde_json::json!(""), "bad_nick"),
        ("connection", serde_json::json!("proxy"), "bad_connection"),
        ("connection", serde_json::json!("proxy:../x"), "bad_connection"),
        ("connection", serde_json::json!("proxy:a b"), "bad_connection"),
        ("connection", serde_json::json!("Direct"), "bad_connection"),
        ("connection", serde_json::json!("proxy:nosuch"), "proxy_unknown"),
        ("notes", serde_json::json!("x".repeat(201)), "bad_notes"),
        ("notes", serde_json::json!("a\nb"), "bad_notes"),
    ] {
        let mut body = add_body(PUB1);
        body[field] = value.clone();
        let r = post(&server, &l, "/api/favourites/add", &body);
        assert_eq!(err(&r), (400, Some(code.into())), "{field}={value}");
    }
    for extra in ["ready", "reopened_at", "consent_at", "added_at", "proxy_host", "id"] {
        let mut body = add_body(PUB1);
        body[extra] = serde_json::json!(1);
        assert_eq!(post(&server, &l, "/api/favourites/add", &body).status, 400, "{extra}");
    }
    assert!(files_in(&dirs(&server).0).is_empty(), "no refused call wrote a file");
    // The same refusals on update.
    assert_eq!(post(&server, &l, "/api/favourites/add", &add_body(PUB1)).status, 201);
    let before = fs::read(dirs(&server).0.join("favourites.json")).unwrap();
    for (field, value) in [
        ("nick", serde_json::json!("a b")),
        ("name", serde_json::json!("")),
        ("connection", serde_json::json!("proxy:nosuch")),
        ("notes", serde_json::json!("a\u{0}b")),
        ("consent_at", serde_json::json!(5)),
        ("reopened_at", serde_json::json!(5)),
        ("address", serde_json::json!("10.0.0.1:1")),
    ] {
        let mut body = serde_json::json!({"address": PUB1});
        body[field] = value.clone();
        assert!(
            post(&server, &l, "/api/favourites/update", &body).status >= 400,
            "{field}"
        );
    }
    assert_eq!(fs::read(dirs(&server).0.join("favourites.json")).unwrap(), before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_favourite_can_be_changed_and_removed_and_a_broken_file_is_never_overwritten() {
    let server = deployed().await;
    let l = login(&server);
    assert_eq!(post(&server, &l, "/api/proxies/save", &proxy_body("hp-1")).status, 200);
    assert_eq!(post(&server, &l, "/api/favourites/add", &add_body(PUB1)).status, 201);
    let added = favourites_on_disk(&server).favourites[0].consent_at;
    let r = post(
        &server,
        &l,
        "/api/favourites/update",
        &serde_json::json!({"address": PUB1, "connection": "proxy:hp-1", "nick": "Muha2", "notes": "n"}),
    );
    assert_eq!(r.status, 200, "{r:?}");
    let f = favourites_on_disk(&server).favourites[0].clone();
    assert_eq!(
        (f.connection.as_str(), f.nick.as_str(), f.notes.as_str()),
        ("proxy:hp-1", "Muha2", "n")
    );
    assert_eq!(
        (f.consent_at, f.reopened_at),
        (added, 0),
        "an edit neither re-consents nor re-opens"
    );
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/favourites/update",
            &serde_json::json!({"address": PUB2})
        )),
        (404, Some("not_found".into()))
    );
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/favourites/remove",
            &serde_json::json!({"address": PUB2})
        )),
        (404, Some("not_found".into()))
    );
    assert_eq!(
        post(
            &server,
            &l,
            "/api/favourites/remove",
            &serde_json::json!({"address": PUB1})
        )
        .status,
        200
    );
    assert!(favourites_on_disk(&server).favourites.is_empty());
    // A broken file: shown as such, never overwritten.
    fs::write(dirs(&server).0.join("favourites.json"), b"{not json").unwrap();
    let j = get(&server, &l, "/api/favourites").json();
    assert_eq!(j["error"], "favourites_invalid");
    assert_eq!(
        err(&post(&server, &l, "/api/favourites/add", &add_body(PUB1))),
        (409, Some("favourites_invalid".into()))
    );
    assert_eq!(fs::read(dirs(&server).0.join("favourites.json")).unwrap(), b"{not json");
}

// ---- a ban closes a favourite until the explicit re-open ----

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_ban_closes_the_favourite_until_the_owner_reopens_it_and_nothing_else_does() {
    let server = deployed().await;
    let l = login(&server);
    let start = serde_json::json!({"action":"start","brain":"hybrid","server":PUB1,"duration":"15m"});
    assert_eq!(post(&server, &l, "/api/proxies/save", &proxy_body("hp-1")).status, 200);
    assert_eq!(post(&server, &l, "/api/proxies/save", &proxy_body("hp-2")).status, 200);
    assert_eq!(post(&server, &l, "/api/favourites/add", &add_body(PUB1)).status, 201);
    // Same IP, another port: the ban is the machine's.
    assert_eq!(
        post(&server, &l, "/api/favourites/add", &add_body("93.184.216.35:8309")).status,
        201
    );
    assert_eq!(post(&server, &l, "/api/favourites/add", &add_body(PUB2)).status, 201);

    // Open: the start is accepted (a request file is written for the helper to judge again).
    assert_eq!(post(&server, &l, "/api/bot/launch", &start).status, 202);
    fs::remove_file(dirs(&server).0.join("request.json")).unwrap();

    // The helper recorded a ban.
    let ban_at = now() - 10;
    write_blocked(&server, &[(PUB1, ban_at)]);
    let j = get(&server, &l, "/api/favourites").json();
    let by = |a: &str| {
        j["favourites"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["address"] == a)
            .unwrap()
            .clone()
    };
    assert_eq!(by(PUB1)["blocked"]["at"], ban_at);
    assert_eq!(
        by("93.184.216.35:8309")["blocked"]["at"],
        ban_at,
        "the same IP is closed too"
    );
    assert_eq!(by(PUB2)["blocked"], serde_json::Value::Null);
    // The launcher card says so, and the start is refused here before it is even written.
    let g = get(&server, &l, "/api/bot/launch").json();
    let choice = g["servers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["id"] == PUB1)
        .unwrap()
        .clone();
    assert_eq!(
        (choice["kind"].as_str(), choice["blocked"].as_bool()),
        (Some("favourite"), Some(true))
    );
    let r = post(&server, &l, "/api/bot/launch", &start);
    assert_eq!(err(&r), (409, Some("blocked_after_ban".into())));
    assert!(!dirs(&server).0.join("request.json").exists());
    // Another favourite on another IP is not affected.
    let other = serde_json::json!({"action":"start","brain":"hybrid","server":PUB2,"duration":"15m"});
    assert_eq!(post(&server, &l, "/api/bot/launch", &other).status, 202);
    fs::remove_file(dirs(&server).0.join("request.json")).unwrap();

    // Changing the connection (even to another proxy) or the nick is not a re-opening: no automatic switch gets round the ban.
    for change in [
        serde_json::json!({"address": PUB1, "connection": "proxy:hp-2"}),
        serde_json::json!({"address": PUB1, "connection": "proxy:hp-1"}),
        serde_json::json!({"address": PUB1, "nick": "Other"}),
        serde_json::json!({"address": PUB1, "connection": "direct"}),
    ] {
        assert_eq!(post(&server, &l, "/api/favourites/update", &change).status, 200);
        assert_eq!(favourites_on_disk(&server).favourites[0].reopened_at, 0);
        assert_eq!(
            err(&post(&server, &l, "/api/bot/launch", &start)),
            (409, Some("blocked_after_ban".into())),
            "{change}"
        );
    }
    // Removing and adding it again does not re-open it either (the new entry has no `reopened_at`).
    assert_eq!(
        post(
            &server,
            &l,
            "/api/favourites/remove",
            &serde_json::json!({"address": PUB1})
        )
        .status,
        200
    );
    assert_eq!(post(&server, &l, "/api/favourites/add", &add_body(PUB1)).status, 201);
    assert_eq!(
        err(&post(&server, &l, "/api/bot/launch", &start)),
        (409, Some("blocked_after_ban".into()))
    );

    // Only the explicit action: it needs `confirm`, a closed favourite, and writes `reopened_at` after the ban.
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/favourites/reopen",
            &serde_json::json!({"address": PUB1, "confirm": false})
        )),
        (400, Some("confirm_required".into()))
    );
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/favourites/reopen",
            &serde_json::json!({"address": PUB1})
        )),
        (400, Some("bad_request".into()))
    );
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/favourites/reopen",
            &serde_json::json!({"address": PUB2, "confirm": true})
        )),
        (409, Some("not_blocked".into())),
        "an open favourite cannot be pre-reopened"
    );
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/favourites/reopen",
            &serde_json::json!({"address": "1.2.3.4:5", "confirm": true})
        )),
        (404, Some("not_found".into()))
    );
    assert_eq!(
        favourites_on_disk(&server)
            .favourites
            .iter()
            .map(|f| f.reopened_at)
            .max(),
        Some(0)
    );
    let r = post(
        &server,
        &l,
        "/api/favourites/reopen",
        &serde_json::json!({"address": PUB1, "confirm": true}),
    );
    assert_eq!(r.status, 200, "{r:?}");
    let f = favourites_on_disk(&server)
        .favourites
        .into_iter()
        .find(|f| f.address == PUB1)
        .unwrap();
    assert!(f.reopened_at > ban_at, "{f:?}");
    let g = get(&server, &l, "/api/bot/launch").json();
    assert_eq!(
        g["servers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["id"] == PUB1)
            .unwrap()["blocked"],
        false
    );
    assert_eq!(post(&server, &l, "/api/bot/launch", &start).status, 202);
    // The sibling port on the same IP is still closed: it has its own re-open to press.
    fs::remove_file(dirs(&server).0.join("request.json")).unwrap();
    let sibling = serde_json::json!({"action":"start","brain":"hybrid","server":"93.184.216.35:8309","duration":"15m"});
    assert_eq!(
        err(&post(&server, &l, "/api/bot/launch", &sibling)),
        (409, Some("blocked_after_ban".into()))
    );
    // A ban dated in the future would need a future re-opening, which the helper refuses: the site says so and writes nothing.
    write_blocked(&server, &[(PUB2, now() + 1000)]);
    let before = fs::read(dirs(&server).0.join("favourites.json")).unwrap();
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/favourites/reopen",
            &serde_json::json!({"address": PUB2, "confirm": true})
        )),
        (409, Some("clock_skew".into()))
    );
    assert_eq!(fs::read(dirs(&server).0.join("favourites.json")).unwrap(), before);
    // A small skew is fine and the file the site wrote still passes the strict parser.
    write_blocked(&server, &[(PUB2, now() + 30)]);
    assert_eq!(
        post(
            &server,
            &l,
            "/api/favourites/reopen",
            &serde_json::json!({"address": PUB2, "confirm": true})
        )
        .status,
        200
    );
    let f = favourites_on_disk(&server)
        .favourites
        .into_iter()
        .find(|f| f.address == PUB2)
        .unwrap();
    assert!(f.reopened_at > now() + 30 - 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_launcher_card_offers_favourites_and_refuses_what_is_not_one() {
    let server = deployed().await;
    let l = login(&server);
    assert_eq!(post(&server, &l, "/api/favourites/add", &add_body(PUB1)).status, 201);
    let g = get(&server, &l, "/api/bot/launch").json();
    let kinds: Vec<(&str, &str)> = g["servers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| (s["id"].as_str().unwrap(), s["kind"].as_str().unwrap()))
        .collect();
    assert_eq!(kinds, vec![("local", "local"), (PUB1, "favourite")]);
    assert_eq!(g["servers"][1]["name"], "Some Block Server");
    let r = post(
        &server,
        &l,
        "/api/bot/launch",
        &serde_json::json!({"action":"start","brain":"hybrid","server":PUB1,"duration":"60m"}),
    );
    assert_eq!(r.status, 202, "{r:?}");
    let req = ddai_web::launch::parse_request(&fs::read(dirs(&server).0.join("request.json")).unwrap()).unwrap();
    assert_eq!(req.server.as_deref(), Some(PUB1));
    fs::remove_file(dirs(&server).0.join("request.json")).unwrap();
    for bad in [
        PUB2,
        "93.184.216.35:8309",
        "93.184.216.36:8308",
        "127.0.0.1:8303",
        "example.com:8303",
    ] {
        let r = post(
            &server,
            &l,
            "/api/bot/launch",
            &serde_json::json!({"action":"start","brain":"hybrid","server":bad,"duration":"15m"}),
        );
        assert_eq!(err(&r), (400, Some("server_not_allowed".into())), "{bad}");
    }
    // Sparring stays local-only.
    let r = post(
        &server,
        &l,
        "/api/bot/launch",
        &serde_json::json!({"action":"start","brain":"hybrid","server":PUB1,"duration":"15m","sparring":1}),
    );
    assert_eq!(err(&r), (400, Some("sparring_local_only".into())));
    // A broken favourites file offers none and says so.
    fs::write(dirs(&server).0.join("favourites.json"), b"junk").unwrap();
    let g = get(&server, &l, "/api/bot/launch").json();
    assert_eq!(g["favourites_error"], "favourites_invalid");
    assert_eq!(g["servers"].as_array().unwrap().len(), 1);
    let r = post(
        &server,
        &l,
        "/api/bot/launch",
        &serde_json::json!({"action":"start","brain":"hybrid","server":PUB1,"duration":"15m"}),
    );
    assert_eq!(err(&r), (400, Some("server_not_allowed".into())));
}

// ---- proxies ----

fn read_mode(path: &Path) -> u32 {
    fs::metadata(path).unwrap().permissions().mode() & 0o777
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_password_is_write_only_never_in_a_response_a_log_or_an_error() {
    let server = deployed().await;
    let l = login(&server);
    let secrets = dirs(&server).1;
    let mut body = proxy_body("hp-1");
    body["user"] = serde_json::json!("secret-user-{session}");
    body["session_pick"] = serde_json::json!(3);
    body["relay"] = serde_json::json!("public");
    let r = post(&server, &l, "/api/proxies/save", &body);
    assert_eq!(r.status, 200, "{r:?}");
    let file = secrets.join("hp-1-proxy.toml");
    assert_eq!(read_mode(&file), 0o600);
    assert!(fs::read_to_string(&file).unwrap().contains(PASS));
    // The format is the one the bot reads.
    let cfg = ddai_client::proxy::load_proxy(&secrets, "hp-1").unwrap();
    assert_eq!(
        (cfg.relay_mode(), cfg.session_pick()),
        (ddai_client::proxy::RelayMode::Public, 3)
    );
    // Collect every response that could carry it.
    let mut seen = vec![r.json().to_string()];
    let list = get(&server, &l, "/api/proxies");
    seen.push(String::from_utf8_lossy(&list.body).into_owned());
    let pv = list.json()["proxies"][0].clone();
    assert_eq!(
        (
            pv["name"].as_str(),
            pv["has_credentials"].as_bool(),
            pv["usable"].as_bool()
        ),
        (Some("hp-1"), Some(true), Some(true))
    );
    assert!(pv.get("pass").is_none() && pv.get("user").is_none() && pv.get("password").is_none());
    // An edit that leaves the fields empty keeps them; every refusal is a code, never an echo.
    let mut e = proxy_body("hp-1");
    e["create"] = serde_json::json!(false);
    e["user"] = serde_json::json!("");
    e["pass"] = serde_json::json!("");
    e["relay"] = serde_json::json!("public");
    e["session_pick"] = serde_json::json!(3);
    e["port"] = serde_json::json!(1081);
    let r = post(&server, &l, "/api/proxies/save", &e);
    assert_eq!(r.status, 200, "{r:?}");
    seen.push(String::from_utf8_lossy(&r.body).into_owned());
    assert!(
        fs::read_to_string(&file).unwrap().contains(PASS),
        "an empty field keeps the stored password"
    );
    for (field, value) in [
        ("pass", serde_json::json!(format!("bad\n{PASS}"))),
        ("user", serde_json::json!(format!("bad\u{0}{PASS}"))),
        ("host", serde_json::json!(format!("10.0.0.1{PASS}"))),
        ("name", serde_json::json!(format!("../{PASS}"))),
        ("relay", serde_json::json!(PASS)),
    ] {
        let mut b = proxy_body("hp-2");
        b[field] = value;
        let r = post(&server, &l, "/api/proxies/save", &b);
        assert_eq!(r.status, 400);
        seen.push(String::from_utf8_lossy(&r.body).into_owned());
        // Also the errors of the unknown-field kind.
    }
    let mut b = proxy_body("hp-2");
    b["secret"] = serde_json::json!(PASS);
    let r = post(&server, &l, "/api/proxies/save", &b);
    assert_eq!(r.status, 400);
    seen.push(String::from_utf8_lossy(&r.body).into_owned());
    for text in &seen {
        assert!(
            !text.contains(PASS) && !text.contains("secret-user") && !text.contains("s3cr3t"),
            "{text}"
        );
    }
    let logged = log_text();
    assert!(
        logged.contains("proxy profile saved from the web"),
        "the change is audited"
    );
    for s in [PASS, "secret-user", "s3cr3t", "93.184.216.34"] {
        assert!(!logged.contains(s), "{s:?} reached the log");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn malicious_proxy_fields_are_refused_and_no_file_appears() {
    let server = deployed().await;
    let l = login(&server);
    let secrets = dirs(&server).1;
    let before = files_in(&secrets);
    for (field, value, code) in [
        ("name", serde_json::json!("../x"), "bad_name"),
        ("name", serde_json::json!("a/b"), "bad_name"),
        ("name", serde_json::json!("a.b"), "bad_name"),
        ("name", serde_json::json!(""), "bad_name"),
        ("name", serde_json::json!("x".repeat(65)), "bad_name"),
        ("host", serde_json::json!("localhost"), "bad_host"),
        ("host", serde_json::json!("example.com"), "bad_host"),
        ("host", serde_json::json!("127.0.0.1"), "bad_host"),
        ("host", serde_json::json!("169.254.169.254"), "bad_host"),
        ("host", serde_json::json!("192.168.0.1"), "bad_host"),
        ("host", serde_json::json!("93.184.216.34:80"), "bad_host"),
        ("host", serde_json::json!("[2a01:4f8::1]"), "bad_host"),
        ("host", serde_json::json!(""), "bad_host"),
        ("user", serde_json::json!("a\nrelay=\"public\""), "bad_user"),
        ("pass", serde_json::json!("a\r\nb"), "bad_pass"),
        ("relay", serde_json::json!("anywhere"), "bad_relay"),
        ("session_pick", serde_json::json!(1), "bad_session_pick"),
        ("session_pick", serde_json::json!(9), "bad_session_pick"),
        ("session_pick", serde_json::json!(2), "proxy_invalid"),
    ] {
        let mut b = proxy_body("hp");
        b[field] = value.clone();
        let r = post(&server, &l, "/api/proxies/save", &b);
        assert_eq!(err(&r), (400, Some(code.into())), "{field}={value}");
    }
    for bad in [
        serde_json::json!({"port": 0}),
        serde_json::json!({"port": 70000}),
        serde_json::json!({"port": "1080"}),
        serde_json::json!({"port": -1}),
    ] {
        let mut b = proxy_body("hp");
        for (k, v) in bad.as_object().unwrap() {
            b[k] = v.clone();
        }
        assert_eq!(post(&server, &l, "/api/proxies/save", &b).status, 400, "{bad}");
    }
    assert_eq!(files_in(&secrets), before, "no refused call left a file behind");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hand_made_proxy_file_is_listed_but_never_touched_and_a_used_proxy_is_kept() {
    let server = deployed().await;
    let l = login(&server);
    let secrets = dirs(&server).1;
    let hand = secrets.join("swarfey-proxy.toml");
    fs::write(
        &hand,
        "host = \"203.0.113.9\"\nport = 1\nuser = \"HAND-USER\"\npass = \"HAND-PASS\"\n",
    )
    .unwrap();
    fs::set_permissions(&hand, fs::Permissions::from_mode(0o600)).unwrap();
    let before = fs::read(&hand).unwrap();
    let list = get(&server, &l, "/api/proxies");
    let text = String::from_utf8_lossy(&list.body).into_owned();
    assert!(
        text.contains("swarfey") && !text.contains("203.0.113.9") && !text.contains("HAND-"),
        "{text}"
    );
    let mut create = proxy_body("swarfey");
    assert_eq!(
        err(&post(&server, &l, "/api/proxies/save", &create)),
        (409, Some("proxy_exists".into()))
    );
    create["create"] = serde_json::json!(false);
    assert_eq!(
        err(&post(&server, &l, "/api/proxies/save", &create)),
        (409, Some("proxy_not_managed".into()))
    );
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/proxies/remove",
            &serde_json::json!({"name": "swarfey"})
        )),
        (409, Some("proxy_not_managed".into()))
    );
    assert_eq!(fs::read(&hand).unwrap(), before);
    // A favourite may name it (the owner's choice; the helper checks it loads), but the site cannot remove a proxy a favourite uses.
    let r = post(&server, &l, "/api/favourites/add", &{
        let mut b = add_body(PUB1);
        b["connection"] = serde_json::json!("proxy:swarfey");
        b
    });
    assert_eq!(r.status, 201, "{r:?}");
    assert_eq!(post(&server, &l, "/api/proxies/save", &proxy_body("hp-1")).status, 200);
    assert_eq!(
        post(
            &server,
            &l,
            "/api/favourites/update",
            &serde_json::json!({"address": PUB1, "connection": "proxy:hp-1"})
        )
        .status,
        200
    );
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/proxies/remove",
            &serde_json::json!({"name": "hp-1"})
        )),
        (409, Some("proxy_in_use".into()))
    );
    assert!(secrets.join("hp-1-proxy.toml").exists());
    assert_eq!(
        post(
            &server,
            &l,
            "/api/favourites/update",
            &serde_json::json!({"address": PUB1, "connection": "direct"})
        )
        .status,
        200
    );
    assert_eq!(
        post(&server, &l, "/api/proxies/remove", &serde_json::json!({"name": "hp-1"})).status,
        200
    );
    assert!(!secrets.join("hp-1-proxy.toml").exists());
    // A favourites file nobody can read proves nothing: the proxy stays.
    assert_eq!(post(&server, &l, "/api/proxies/save", &proxy_body("hp-3")).status, 200);
    fs::write(dirs(&server).0.join("favourites.json"), b"junk").unwrap();
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/proxies/remove",
            &serde_json::json!({"name": "hp-3"})
        )),
        (409, Some("favourites_invalid".into()))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_proxy_check_is_a_small_request_for_the_check_unit_and_its_result_is_shown_for_that_request_only() {
    let server = deployed().await;
    let l = login(&server);
    let (launch, _, _, _) = dirs(&server);
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/proxies/check",
            &serde_json::json!({"name": "nosuch"})
        )),
        (404, Some("proxy_not_found".into()))
    );
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/proxies/check",
            &serde_json::json!({"name": "../x"})
        )),
        (400, Some("bad_name".into()))
    );
    assert!(files_in(&launch).is_empty());
    assert_eq!(post(&server, &l, "/api/proxies/save", &proxy_body("hp-1")).status, 200);
    let r = post(&server, &l, "/api/proxies/check", &serde_json::json!({"name": "hp-1"}));
    assert_eq!(r.status, 202, "{r:?}");
    let id = r.json()["id"].as_str().unwrap().to_string();
    assert_eq!(files_in(&launch), vec![PROXY_CHECK_REQUEST_FILE.to_string()]);
    let req = parse_proxy_check_request(&fs::read(launch.join(PROXY_CHECK_REQUEST_FILE)).unwrap())
        .expect("the helper's own parser accepts it");
    assert_eq!((req.id.as_str(), req.proxy.as_str()), (id.as_str(), "hp-1"));
    // No credential in the request.
    let raw = fs::read_to_string(launch.join(PROXY_CHECK_REQUEST_FILE)).unwrap();
    assert!(!raw.contains(PASS) && !raw.contains("93.184"), "{raw}");
    // Not asked again while it waits.
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/proxies/check",
            &serde_json::json!({"name": "hp-1"})
        )),
        (409, Some("pending".into()))
    );
    assert_eq!(get(&server, &l, "/api/proxies").json()["pending"], true);
    // The unit answers; the page shows the result of THIS request only.
    fs::remove_file(launch.join(PROXY_CHECK_REQUEST_FILE)).unwrap();
    let result = |rid: &str, name: &str| serde_json::json!({"v":1,"id":rid,"at":now(),"proxy":name,"ok":true,"code":"ok","relay":"remote","relay_mode":"public","udp_rtt_ms":22,"probe_sent":5,"probe_replies":4});
    fs::write(
        launch.join(PROXY_CHECK_RESULT_FILE),
        result("ffffffffffffffff", "hp-1").to_string(),
    )
    .unwrap();
    let j = get(&server, &l, "/api/proxies").json();
    assert_eq!(j["check"], serde_json::Value::Null, "another request's result");
    assert_eq!(j["last_check_id"], id);
    fs::write(launch.join(PROXY_CHECK_RESULT_FILE), result(&id, "hp-1").to_string()).unwrap();
    let j = get(&server, &l, "/api/proxies").json();
    assert_eq!(
        (
            j["check"]["ok"].as_bool(),
            j["check"]["udp_rtt_ms"].as_u64(),
            j["pending"].as_bool()
        ),
        (Some(true), Some(22), Some(false))
    );
    // A request nobody consumed for a minute is removed and asked again.
    let stale = launch.join(PROXY_CHECK_REQUEST_FILE);
    fs::write(&stale, b"{}").unwrap();
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(120);
    fs::File::options()
        .write(true)
        .open(&stale)
        .unwrap()
        .set_modified(old)
        .unwrap();
    assert_eq!(
        post(&server, &l, "/api/proxies/check", &serde_json::json!({"name": "hp-1"})).status,
        202
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn favourites_are_refused_while_the_allow_list_names_a_server_by_host_name() {
    let server = deployed().await;
    let l = login(&server);
    fs::write(
        &server.config.live_servers,
        "[[server]]\naddress=\"one.one.one.one:8303\"\nnick=\"Muha\"\nready=true\nproxy=\"swarfey\"\n",
    )
    .unwrap();
    assert_eq!(
        err(&post(&server, &l, "/api/favourites/add", &add_body("1.1.1.1:8303"))),
        (409, Some("allowlist_not_literal".into()))
    );
    assert!(!dirs(&server).0.join("favourites.json").exists());
    // IP literals only: fine.
    fs::write(
        &server.config.live_servers,
        "[[server]]\naddress=\"93.184.216.34:8303\"\nnick=\"Muha\"\nready=false\n",
    )
    .unwrap();
    assert_eq!(post(&server, &l, "/api/favourites/add", &add_body(PUB1)).status, 201);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_proxy_check_takes_its_rate_gate_before_it_parses_the_body() {
    let server = deployed_with(|c| c.proxycheck_min_gap = std::time::Duration::from_secs(60)).await;
    let l = login(&server);
    assert_eq!(
        post(&server, &l, "/api/proxies/check", &serde_json::json!({"junk": 1})).status,
        400
    );
    assert_eq!(
        err(&post(
            &server,
            &l,
            "/api/proxies/check",
            &serde_json::json!({"junk": 1})
        )),
        (429, Some("rate_limited".into())),
        "a refused body still counted as an attempt"
    );
}
