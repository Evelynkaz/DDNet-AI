//! Task 5.12 (D-099): `ddnet-ai servers-cache` as the real binary against a loopback HTTP fixture (the `--fixture-url` test switch accepts
//! plain-HTTP 127.0.0.1 only; the real masters are never contacted). What comes back is untrusted: the cache is bounded, strictly parsed,
//! public IPv4 only, and an oversized, redirected, broken or private-only answer leaves the old cache alone.

use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Command, Output};

const FIXTURE: &str = include_str!("../../ddai-client/tests/fixtures/master-servers.json");

/// A one-shot HTTP server answering every connection with `response` (a full HTTP/1.1 response) until dropped.
struct Fixture {
    port: u16,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Fixture {
    fn start(response: Vec<u8>) -> Fixture {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = stop.clone();
        std::thread::spawn(move || {
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut s, _)) => {
                        s.set_nonblocking(false).unwrap();
                        let mut buf = [0u8; 2048];
                        let _ = s.read(&mut buf);
                        let _ = s.write_all(&response);
                    }
                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
                }
            }
        });
        Fixture { port, stop }
    }

    fn ok(body: &str) -> Fixture {
        Fixture::start(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_bytes(),
        )
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}/ddnet/15/servers.json", self.port)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn run(out_dir: &std::path::Path, url: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ddnet-ai"))
        .args(["servers-cache", "--fixture-url", url, "--out"])
        .arg(out_dir)
        .output()
        .expect("run ddnet-ai servers-cache")
}

fn files(dir: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

#[test]
fn a_good_answer_becomes_the_strict_cache_and_a_second_run_within_a_minute_fetches_nothing() {
    let fx = Fixture::ok(FIXTURE);
    let dir = tempfile::tempdir().unwrap();
    let out = run(dir.path(), &fx.url());
    assert!(out.status.success(), "{out:?}");
    assert_eq!(files(dir.path()), vec!["master.json", "refresh.json"]);
    let cache =
        ddai_client::server_list::MasterCache::parse(&fs::read(dir.path().join("master.json")).unwrap()).unwrap();
    assert!(!cache.servers.is_empty() && cache.servers.iter().all(|s| !s.address.starts_with("203.0.113.")));
    assert!(String::from_utf8_lossy(&out.stdout).contains("wrote"), "{out:?}");
    let first = fs::read(dir.path().join("master.json")).unwrap();
    // A second run: the cache is a few milliseconds old.
    let out = run(dir.path(), &fx.url());
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("younger"), "{out:?}");
    assert_eq!(fs::read(dir.path().join("master.json")).unwrap(), first);
}

#[test]
fn an_oversized_redirected_failed_or_broken_answer_leaves_the_old_cache_alone() {
    let dir = tempfile::tempdir().unwrap();
    // No cache yet: each of these ends in a failure and no cache file.
    let big = "x".repeat(ddai_client::server_list::MAX_MASTER_BYTES + 1000);
    let answers: Vec<(&str, Vec<u8>)> = vec![
        (
            "oversized",
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{big}",
                big.len()
            )
            .into_bytes(),
        ),
        (
            "chunked oversized",
            format!("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{big}").into_bytes(),
        ),
        (
            "redirect",
            b"HTTP/1.1 302 Found\r\nLocation: http://10.0.0.1/x\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                .to_vec(),
        ),
        (
            "server error",
            b"HTTP/1.1 500 Oops\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
        ),
        (
            "garbage",
            b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot json!".to_vec(),
        ),
        ("not http at all", b"\x00\x01\x02".to_vec()),
    ];
    for (what, bytes) in answers {
        let fx = Fixture::start(bytes);
        let out = run(dir.path(), &fx.url());
        assert!(!out.status.success(), "{what}: {out:?}");
        assert!(!dir.path().join("master.json").exists(), "{what}");
        let note: ddai_client::server_list::RefreshStatus =
            serde_json::from_slice(&fs::read(dir.path().join("refresh.json")).unwrap()).unwrap();
        assert!(!note.ok, "{what}");
        assert!(
            matches!(note.reason.as_deref(), Some("no_master" | "bad_list")),
            "{what}: {note:?}"
        );
        // The terminal and the note never carry the url or the error text.
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stderr),
            fs::read_to_string(dir.path().join("refresh.json")).unwrap()
        );
        assert!(
            !text.contains("127.0.0.1") && !text.contains("10.0.0.1"),
            "{what}: {text}"
        );
    }
    // A good cache, then a failing run later: the good cache stays byte for byte.
    let good = Fixture::ok(FIXTURE);
    assert!(run(dir.path(), &good.url()).status.success());
    let kept = fs::read(dir.path().join("master.json")).unwrap();
    // (The minimum interval would skip the fetch; age the cache by rewriting its time.)
    let mut cache = ddai_client::server_list::MasterCache::parse(&kept).unwrap();
    cache.fetched_at -= 3600;
    let aged = cache.to_bytes().unwrap();
    fs::write(dir.path().join("master.json"), &aged).unwrap();
    let bad = Fixture::start(b"HTTP/1.1 500 Oops\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec());
    assert!(!run(dir.path(), &bad.url()).status.success());
    assert_eq!(
        fs::read(dir.path().join("master.json")).unwrap(),
        aged,
        "the old cache stays"
    );
}

#[test]
fn only_a_plain_http_loopback_fixture_or_the_four_masters_may_be_fetched() {
    let dir = tempfile::tempdir().unwrap();
    for url in [
        "https://evil.example/servers.json",
        "http://10.0.0.1:80/x",
        "file:///etc/passwd",
        "http://127.0.0.1.evil.example:80/x",
    ] {
        let out = run(dir.path(), url);
        assert!(!out.status.success(), "{url}");
        assert!(!dir.path().join("master.json").exists(), "{url}");
    }
}
