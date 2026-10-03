//! The «Обучение» tab's server side (task 5.8) against a real server and a scratch runs directory with synthetic data:
//! the two routes are behind the session, read only what is under the runs root, answer traversal attempts with a plain
//! refusal, cap what they read and send, and the page and its script come with the site's headers.

mod support;

use std::fs;
use std::path::Path;
use std::time::Duration;

use serde_json::json;
use support::{Req, TestServer, send};

fn write(path: &Path, content: impl AsRef<[u8]>) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn rows(rows: &[serde_json::Value]) -> String {
    rows.iter().map(|r| r.to_string() + "\n").collect()
}

/// A scratch runs directory: `E-1/run-a` (done, with a checkpoint and an eval summary), `E-1/run-b` (running).
fn scratch_runs() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let metrics = rows(&[
        json!({"kind":"train","phase":"bc","step":50,"loss":{"total":3.0,"dir":1.0,"hook":0.6},"grad_norm":2.0}),
        json!({"kind":"train","phase":"bc","step":100,"loss":{"total":2.0,"dir":0.9,"hook":0.5}}),
        json!({"kind":"eval","phase":"bc","set":"dagger-val","step":100,"report":{"dir":{"accuracy":0.7},"hook":{"auroc":0.8}}}),
        json!({"kind":"arena","phase":"bc","step":100,"eval":{"arena":"clb-left","games":300,"w":100,"l":150,"d":0,"t":50,
            "credited_w":30,"credited_win_rate":[0.1,0.07,0.14],"win_rate":[0.4,0.35,0.45],"win_rate_all":[0.33,0.28,0.38]}}),
        json!({"kind":"train","phase":"dagger-1","step":150,"loss":{"total":1.5}}),
    ]);
    let config = "bc_steps = 100\n[model]\nkind = \"fly\"\n[dagger]\nbetas = [0.5]\nsteps_per_round = 50\n";
    for (run, status) in [
        ("run-a", r#"{"phase":"done","step":150}"#),
        (
            "run-b",
            r#"{"phase":"dagger-1","step":150,"phase_step":10,"phase_steps":50}"#,
        ),
    ] {
        write(&root.join("E-1").join(run).join("status.json"), status);
        write(&root.join("E-1").join(run).join("metrics.jsonl"), &metrics);
        write(&root.join("E-1").join(run).join("config.toml"), config);
    }
    write(
        &root.join("E-1").join("run-a").join("checkpoints").join("final.bundle"),
        b"abc",
    );
    let summary = json!({"meta":{"git_commit":"0123456789abcdef"},"conditions":[
        {"name":"clb-left vs scripted","arena":"clb-left","games":100,"tally":{"w":50,"l":40,"d":0,"t":10},
         "win_rate":{"p":0.55,"lo":0.45,"hi":0.64},"win_rate_all":{"p":0.5,"lo":0.4,"hi":0.6},"credited_w":5}]});
    write(
        &root
            .join("E-1")
            .join("eval")
            .join("run-a")
            .join("arena")
            .join("summary.json"),
        summary.to_string(),
    );
    tmp
}

async fn server_for(runs: &tempfile::TempDir) -> TestServer {
    let dir = runs.path().to_path_buf();
    TestServer::start_with(move |c| c.runs_dir = dir).await
}

fn get(server: &TestServer, cookie: &str, path: &str) -> support::RawResponse {
    send(server.addr, Req::new("GET", path).cookie(cookie))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_routes_need_a_session() {
    let runs = scratch_runs();
    let server = server_for(&runs).await;
    for path in [
        "/api/train/runs",
        "/api/train/run?exp=E-1&run=run-a",
        "/api/train/runs?poll=1",
        "/api/train/run?exp=E-1&run=run-a&poll=1",
        "/api/train/run?exp=..&run=x",
    ] {
        let r = send(server.addr, Req::new("GET", path));
        assert_eq!(r.status, 401, "{path}");
        assert_eq!(r.json()["error"], "unauthenticated");
        assert_eq!(r.header("cache-control"), Some("no-store"));
        // Nothing of the data in a refusal.
        assert!(!String::from_utf8_lossy(&r.body).contains("run-a"));
    }
    // A forged / garbage session cookie is no session.
    let name = server.cookie_name();
    for cookie in [format!("{name}=garbage"), format!("{name}="), "other=1".to_string()] {
        let r = get(&server, &cookie, "/api/train/runs");
        assert_eq!(r.status, 401, "{cookie}");
    }
    // After logging out, the old cookie is refused.
    let (cookie, csrf) = server.login();
    assert_eq!(get(&server, &cookie, "/api/train/runs").status, 200);
    let logout = send(
        server.addr,
        Req::new("POST", "/api/logout")
            .cookie(&cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &csrf)
            .json_body(&json!({})),
    );
    assert_eq!(logout.status, 200, "{logout:?}");
    assert_eq!(get(&server, &cookie, "/api/train/runs").status, 401);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_listing_and_a_run_come_back_as_json_with_the_site_headers() {
    let runs = scratch_runs();
    let server = server_for(&runs).await;
    let (cookie, _) = server.login();

    let r = get(&server, &cookie, "/api/train/runs");
    assert_eq!(r.status, 200);
    assert_eq!(r.header("cache-control"), Some("no-store"));
    assert!(
        r.header("content-security-policy")
            .unwrap()
            .contains("default-src 'self'")
    );
    assert_eq!(r.header("x-content-type-options"), Some("nosniff"));
    let v = r.json();
    assert_eq!(v["root_present"], true);
    assert_eq!(v["poll_secs"], 10);
    assert_eq!(v["experiments"][0]["id"], "E-1");
    let runs_json = v["experiments"][0]["runs"].as_array().unwrap();
    assert_eq!(runs_json.len(), 2);
    assert_eq!(
        (runs_json[0]["id"].clone(), runs_json[0]["state"].clone()),
        ("run-a".into(), "done".into())
    );
    assert_eq!(runs_json[1]["state"], "running");
    assert_eq!(runs_json[1]["planned_steps"], 150);

    let r = get(&server, &cookie, "/api/train/run?exp=E-1&run=run-a");
    assert_eq!(r.status, 200);
    let v = r.json();
    assert_eq!(v["summary"]["state"], "done");
    assert_eq!(v["metrics"]["train"].as_array().unwrap().len(), 3);
    assert_eq!(v["metrics"]["phases"][1]["name"], "dagger-1");
    assert_eq!(v["metrics"]["arena"][0]["credited"]["hi"], 0.14);
    assert_eq!(v["checkpoints"][0]["name"], "final.bundle");
    assert_eq!(v["checkpoints"][0]["sha256"], "ba7816bf8f01cfea");
    assert_eq!(v["eval_summaries"][0]["conditions"][0]["credited"]["p"], 0.05);
    // No absolute path of the machine anywhere in what the page gets.
    let body = String::from_utf8(r.body).unwrap();
    assert!(!body.contains(runs.path().to_str().unwrap()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn traversal_and_bad_names_get_a_plain_refusal() {
    let runs = scratch_runs();
    let server = server_for(&runs).await;
    let (cookie, _) = server.login();
    let bad = [
        "/api/train/run?exp=..&run=x",
        "/api/train/run?exp=E-1&run=..",
        "/api/train/run?exp=E-1&run=..%2F..%2Foutside-secret.json",
        "/api/train/run?exp=E-1&run=%2e%2e",
        "/api/train/run?exp=%2Fetc&run=passwd",
        "/api/train/run?exp=E-1%2Frun-a&run=x",
        "/api/train/run?exp=E-1&run=run-a%00",
        "/api/train/run?exp=E-1&run=run%20a",
        "/api/train/run?exp=&run=run-a",
        "/api/train/run?exp=E-1",
        "/api/train/run?run=run-a",
        "/api/train/run",
    ];
    for path in bad {
        let r = get(&server, &cookie, path);
        assert_eq!(r.status, 400, "{path}: {r:?}");
        assert_eq!(r.json()["error"], "bad_name");
    }
    for path in [
        "/api/train/run?exp=E-1&run=nope",
        "/api/train/run?exp=E-9&run=run-a",
        "/api/train/run?exp=E-1&run=eval",
    ] {
        let r = get(&server, &cookie, path);
        assert_eq!(r.status, 404, "{path}");
        assert_eq!(r.json()["error"], "not_found");
    }
    // The raw path is not a way in either (no static file route reads the runs tree).
    for path in [
        "/api/train/../runs",
        "/runs/E-1/run-a/status.json",
        "/api/train/run/E-1/run-a",
    ] {
        let r = get(&server, &cookie, path);
        assert!(r.status == 404 || r.status == 400, "{path}: {}", r.status);
        assert!(!String::from_utf8_lossy(&r.body).contains("\"phase\""), "{path}");
    }
    // Only GET exists.
    let (cookie, csrf) = server.login();
    let r = send(
        server.addr,
        Req::new("POST", "/api/train/runs")
            .cookie(&cookie)
            .header("Origin", &server.origin())
            .header("X-CSRF-Token", &csrf)
            .json_body(&json!({})),
    );
    assert_eq!(r.status, 405);
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_symlinked_run_pointing_outside_is_not_served() {
    let runs = scratch_runs();
    let outside = tempfile::tempdir().unwrap();
    write(&outside.path().join("status.json"), r#"{"phase":"done","step":777}"#);
    std::os::unix::fs::symlink(outside.path(), runs.path().join("E-1").join("evil")).unwrap();
    let server = server_for(&runs).await;
    let (cookie, _) = server.login();
    let r = get(&server, &cookie, "/api/train/run?exp=E-1&run=evil");
    assert_eq!(r.status, 404);
    assert!(!String::from_utf8_lossy(&r.body).contains("777"));
    let listing = get(&server, &cookie, "/api/train/runs").json();
    let ids: Vec<_> = listing["experiments"][0]["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(ids, ["run-a", "run-b"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_missing_runs_directory_is_an_empty_listing() {
    let tmp = tempfile::tempdir().unwrap();
    let absent = tmp.path().join("absent");
    let server = TestServer::start_with(move |c| c.runs_dir = absent).await;
    let (cookie, _) = server.login();
    let v = get(&server, &cookie, "/api/train/runs").json();
    assert_eq!(v["root_present"], false);
    assert_eq!(v["experiments"].as_array().unwrap().len(), 0);
    assert_eq!(get(&server, &cookie, "/api/train/run?exp=E-1&run=run-a").status, 404);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_huge_metrics_file_is_tail_read_and_the_answer_stays_small() {
    let tmp = tempfile::tempdir().unwrap();
    let run = tmp.path().join("E-1").join("huge");
    write(&run.join("status.json"), r#"{"phase":"bc","step":99999}"#);
    // ~7 MiB of train rows (past the 4 MiB tail cap).
    let mut metrics = String::new();
    let mut step = 0u64;
    while metrics.len() < 7 * 1024 * 1024 {
        step += 1;
        metrics += &json!({"kind":"train","phase":"bc","step":step,"loss":{"total":1.0,"dir":0.5,"jump":0.5,"hook":0.5,"fire":0.5,"aim":-1.0},
            "grad_norm":1.0,"lr_mult":1.0,"decisions_per_s":700.0,"activity":0.0,"skipped":0,"unix_s":1790000000})
        .to_string();
        metrics.push('\n');
    }
    write(&run.join("metrics.jsonl"), &metrics);
    let dir = tmp.path().to_path_buf();
    let server = TestServer::start_with(move |c| c.runs_dir = dir).await;
    let (cookie, _) = server.login();
    let started = std::time::Instant::now();
    let r = get(&server, &cookie, "/api/train/run?exp=E-1&run=huge");
    assert!(started.elapsed() < Duration::from_secs(5));
    assert_eq!(r.status, 200);
    assert!(r.body.len() < 400 * 1024, "{} bytes", r.body.len());
    let v = r.json();
    assert_eq!(v["metrics"]["tail_truncated"], true);
    assert_eq!(v["metrics"]["file_len"], metrics.len());
    assert_eq!(v["metrics"]["train"].as_array().unwrap().len(), 600);
    assert_eq!(v["metrics"]["train_thinned"], true);
    assert_eq!(v["metrics"]["train"].as_array().unwrap().last().unwrap()["step"], step);
}

/// The page fires a list, a run and several comparison runs at once (`GET`s on blocking threads); more than the server's
/// scan slots (4) must queue, never come back `503`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn more_concurrent_requests_than_scan_slots_all_succeed() {
    let tmp = tempfile::tempdir().unwrap();
    let mut metrics = String::new();
    let mut step = 0u64;
    while metrics.len() < 3 * 1024 * 1024 {
        step += 1;
        metrics += &json!({"kind":"train","phase":"bc","step":step,"loss":{"total":1.0,"dir":0.5,"jump":0.5,"hook":0.5,"fire":0.5,"aim":-1.0},"pad":"x".repeat(60)}).to_string();
        metrics.push('\n');
    }
    for run in ["a", "b", "c", "d"] {
        write(&tmp.path().join("E-1").join(run).join("metrics.jsonl"), &metrics);
        write(
            &tmp.path().join("E-1").join(run).join("status.json"),
            r#"{"phase":"done","step":1}"#,
        );
    }
    let dir = tmp.path().to_path_buf();
    let server = TestServer::start_with(move |c| c.runs_dir = dir).await;
    let (cookie, _) = server.login();
    for round in 0..3 {
        let addr = server.addr;
        let mut threads = Vec::new();
        for i in 0..12 {
            let cookie = cookie.clone();
            threads.push(std::thread::spawn(move || {
                let path = if i % 3 == 0 {
                    "/api/train/runs".to_string()
                } else {
                    format!("/api/train/run?exp=E-1&run={}", ["a", "b", "c", "d"][i % 4])
                };
                send(addr, Req::new("GET", &path).cookie(&cookie)).status
            }));
        }
        let statuses: Vec<u16> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert!(statuses.iter().all(|s| *s == 200), "round {round}: {statuses:?}");
    }
}

/// F4: the session is checked before the query is parsed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_malformed_query_is_a_json_401_without_a_session_and_a_json_400_with_one() {
    let runs = scratch_runs();
    let server = server_for(&runs).await;
    for path in [
        "/api/train/run?exp=E-1&exp=..&run=x",
        "/api/train/run?exp=%ZZ&run=x",
        "/api/train/runs?poll=1&poll=2",
    ] {
        let r = send(server.addr, Req::new("GET", path));
        assert_eq!(r.status, 401, "{path}");
        assert_eq!(r.json()["error"], "unauthenticated");
        assert!(r.header("content-type").unwrap().starts_with("application/json"));
    }
    let (cookie, _) = server.login();
    let r = get(&server, &cookie, "/api/train/run?exp=E-1&exp=..&run=run-a");
    assert_eq!(r.status, 400);
    assert_eq!(r.json()["error"], "bad_name");
    // A repeated `poll` on the list is just no poll.
    assert_eq!(get(&server, &cookie, "/api/train/runs?poll=1&poll=2").status, 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_page_poll_does_not_keep_the_session_alive_but_a_person_s_request_does() {
    let runs = scratch_runs();
    let dir = runs.path().to_path_buf();
    let server = TestServer::start_with(move |c| {
        c.runs_dir = dir;
        c.idle_timeout = Duration::from_millis(1200);
    })
    .await;
    // Requests a person makes (no `poll`) refresh the idle timeout: still in after 2.4 s of 0.4 s gaps.
    let (cookie, _) = server.login();
    for _ in 0..6 {
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert_eq!(get(&server, &cookie, "/api/train/runs").status, 200);
    }
    // The page's timer (`poll=1`) does not: the same rhythm ends in 401 once the idle time has passed.
    let (cookie, _) = server.login();
    let mut statuses = Vec::new();
    for _ in 0..6 {
        tokio::time::sleep(Duration::from_millis(400)).await;
        statuses.push(get(&server, &cookie, "/api/train/runs?poll=1").status);
    }
    assert_eq!(statuses.first(), Some(&200), "{statuses:?}");
    assert_eq!(statuses.last(), Some(&401), "{statuses:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_page_and_its_script_are_served_with_the_site_headers() {
    let server = TestServer::start().await;
    let page = send(server.addr, Req::new("GET", "/"));
    assert_eq!(page.status, 200);
    let html = String::from_utf8(page.body).unwrap();
    assert!(html.contains(r#"id="tab-train""#) && html.contains(r#"id="train-view""#) && html.contains("Обучение"));
    assert!(html.contains(r#"src="/train.js""#));
    assert!(!html.contains("style=\""), "the CSP forbids inline styles");
    let js = send(server.addr, Req::new("GET", "/train.js"));
    assert_eq!(js.status, 200);
    assert!(js.header("content-type").unwrap().starts_with("text/javascript"));
    assert!(
        js.header("content-security-policy")
            .unwrap()
            .contains("script-src 'self'")
    );
    assert_eq!(js.header("x-content-type-options"), Some("nosniff"));
    let src = String::from_utf8(js.body).unwrap();
    assert!(!src.contains(".innerHTML") && !src.contains("insertAdjacentHTML"));
    assert!(!src.contains("eval(") && !src.contains("document.write"));
    assert!(!src.contains("setAttribute(\"style\"") && !src.contains("setAttribute('style'"));
    // The script is the page's own; the data needs a session, the script does not.
    assert_eq!(send(server.addr, Req::new("GET", "/api/train/runs")).status, 401);
}
