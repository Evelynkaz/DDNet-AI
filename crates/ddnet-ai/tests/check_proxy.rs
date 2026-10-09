//! Task 5.12 (D-099): `ddnet-ai launch check-proxy` — the site's «Проверить» — as the real binary against the in-process loopback SOCKS5
//! server (127.0.0.1 only; no game server, no network). The request is consumed before it is looked at; the result has fixed codes and
//! numbers and never an address, a user name or a password; a bad, stale, oversized or symlinked request and a missing or unsafe profile
//! each end in a code, never in a crash or a leak.

// Task 5.5a: relies on POSIX permission bits and symlinks; the Windows equivalents (ACLs) are tested in ddai-os.
#![cfg(unix)]

use ddai_client::socks5_testserver::{Auth, Config, TestSocks5Server};
use serde_json::{Value, json};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const USER: &str = "tester-user-4711";
const PASS: &str = "tester-pass-9274";

struct Rig {
    dir: tempfile::TempDir,
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl Rig {
    fn new() -> Rig {
        let rig = Rig {
            dir: tempfile::tempdir().unwrap(),
        };
        fs::create_dir_all(rig.launch()).unwrap();
        fs::create_dir_all(rig.dir.path().join("secrets")).unwrap();
        rig
    }
    fn launch(&self) -> PathBuf {
        self.dir.path().join("launch")
    }
    fn request(&self) -> PathBuf {
        self.launch().join("proxycheck-request.json")
    }
    fn result_path(&self) -> PathBuf {
        self.launch().join("proxycheck-result.json")
    }
    fn profile(&self, name: &str, body: &str, mode: u32) {
        let path = self.dir.path().join(format!("secrets/{name}-proxy.toml"));
        fs::write(&path, body).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
    }
    fn body_for(server: &TestSocks5Server, user: &str, pass: &str) -> String {
        format!(
            "host = \"{}\"\nport = {}\nuser = \"{user}\"\npass = \"{pass}\"\n",
            server.addr().ip(),
            server.addr().port()
        )
    }
    fn ask(&self, name: &str) {
        self.ask_raw(&serde_json::to_vec(&json!({"v":1,"id":"0123456789abcdef","ts":now(),"proxy":name})).unwrap());
    }
    fn ask_raw(&self, bytes: &[u8]) {
        fs::write(self.request(), bytes).unwrap();
    }
    fn run(&self) -> Output {
        Command::new(env!("CARGO_BIN_EXE_ddnet-ai"))
            .args(["launch", "check-proxy", "--data-dir"])
            .arg(self.dir.path())
            .output()
            .expect("run ddnet-ai launch check-proxy")
    }
    fn result(&self) -> Value {
        serde_json::from_slice(&fs::read(self.result_path()).expect("a result was written")).unwrap()
    }
}

fn all_text(out: &Output, rig: &Rig) -> String {
    format!(
        "{}{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
        fs::read_to_string(rig.result_path()).unwrap_or_default()
    )
}

#[test]
fn a_good_proxy_is_ok_the_request_is_consumed_and_nothing_secret_is_written() {
    let server = TestSocks5Server::start(Config {
        auth: Auth::UserPass(USER.into(), PASS.into()),
        ..Default::default()
    });
    let rig = Rig::new();
    rig.profile("hp", &Rig::body_for(&server, USER, PASS), 0o600);
    rig.ask("hp");
    let out = rig.run();
    assert!(out.status.success(), "{out:?}");
    assert!(!rig.request().exists(), "the request is consumed");
    let r = rig.result();
    assert_eq!(
        (
            r["ok"].as_bool(),
            r["code"].as_str(),
            r["proxy"].as_str(),
            r["id"].as_str()
        ),
        (Some(true), Some("ok"), Some("hp"), Some("0123456789abcdef"))
    );
    assert_eq!(r["relay"], "same_host");
    assert_eq!(r["relay_mode"], "proxy-host-only");
    assert_eq!(
        fs::metadata(rig.result_path()).unwrap().permissions().mode() & 0o777,
        0o644
    );
    let text = all_text(&out, &rig);
    for secret in [USER, PASS, "127.0.0.1", &server.addr().port().to_string()] {
        assert!(!text.contains(secret), "leaked {secret:?}: {text}");
    }
    // No temporary file is left behind.
    let names: Vec<String> = fs::read_dir(rig.launch())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(names.iter().all(|n| !n.contains(".tmp-")), "{names:?}");
}

#[test]
fn a_wrong_password_a_tcp_only_proxy_and_an_unreachable_proxy_are_codes() {
    let server = TestSocks5Server::start(Config {
        auth: Auth::UserPass(USER.into(), PASS.into()),
        ..Default::default()
    });
    let rig = Rig::new();
    rig.profile("hp", &Rig::body_for(&server, USER, "not-the-password"), 0o600);
    rig.ask("hp");
    let out = rig.run();
    assert!(out.status.success());
    assert_eq!(
        (rig.result()["ok"].as_bool(), rig.result()["code"].as_str()),
        (Some(false), Some("auth_failed"))
    );
    assert!(!all_text(&out, &rig).contains("not-the-password"));

    let tcp_only = TestSocks5Server::start(Config {
        reply_code: 7,
        ..Default::default()
    });
    rig.profile(
        "tcp",
        &format!(
            "host = \"{}\"\nport = {}\n",
            tcp_only.addr().ip(),
            tcp_only.addr().port()
        ),
        0o600,
    );
    rig.ask("tcp");
    assert!(rig.run().status.success());
    assert_eq!(rig.result()["code"], "udp_not_supported");

    // Nothing listens there: a connect failure, not a hang and not the address.
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    rig.profile("dead", &format!("host = \"127.0.0.1\"\nport = {port}\n"), 0o600);
    rig.ask("dead");
    let out = rig.run();
    assert_eq!(rig.result()["code"], "connect_failed", "{}", all_text(&out, &rig));
    assert!(!all_text(&out, &rig).contains(&port.to_string()));
}

#[test]
fn a_missing_loose_or_linked_profile_and_bad_requests_end_in_codes() {
    let rig = Rig::new();
    let code = |rig: &Rig| rig.result()["code"].as_str().unwrap().to_string();
    // No request: nothing happens, no result.
    let out = rig.run();
    assert!(out.status.success());
    assert!(!rig.result_path().exists());
    // A profile that does not exist.
    rig.ask("nosuch");
    assert!(rig.run().status.success());
    assert_eq!(code(&rig), "proxy_missing");
    // A profile with a loose mode, and a symlink in its place.
    rig.profile("loose", "host = \"127.0.0.1\"\nport = 1\n", 0o644);
    rig.ask("loose");
    assert!(rig.run().status.success());
    assert_eq!(code(&rig), "proxy_file_bad");
    rig.profile("real", "host = \"127.0.0.1\"\nport = 1\n", 0o600);
    std::os::unix::fs::symlink(
        rig.dir.path().join("secrets/real-proxy.toml"),
        rig.dir.path().join("secrets/link-proxy.toml"),
    )
    .unwrap();
    rig.ask("link");
    assert!(rig.run().status.success());
    assert_eq!(code(&rig), "proxy_file_bad");
    // Bad requests: junk, an unknown field, a path in the name, an oversized file, an old and a future-dated one.
    let bad: Vec<Vec<u8>> = vec![
        b"{ nope".to_vec(),
        serde_json::to_vec(&json!({"v":1,"id":"0123456789abcdef","ts":now(),"proxy":"hp","extra":1})).unwrap(),
        serde_json::to_vec(&json!({"v":1,"id":"0123456789abcdef","ts":now(),"proxy":"../secrets/real"})).unwrap(),
        serde_json::to_vec(&json!({"v":1,"id":"XYZ","ts":now(),"proxy":"real"})).unwrap(),
        serde_json::to_vec(&json!({"v":2,"id":"0123456789abcdef","ts":now(),"proxy":"real"})).unwrap(),
        vec![b' '; 4096],
    ];
    for bytes in bad {
        rig.ask_raw(&bytes);
        assert!(rig.run().status.success());
        assert_eq!(code(&rig), "bad_request");
        assert!(!rig.request().exists(), "a bad request is consumed too");
    }
    for ts in [now() - 3600, now() + 3600] {
        rig.ask_raw(&serde_json::to_vec(&json!({"v":1,"id":"0123456789abcdef","ts":ts,"proxy":"real"})).unwrap());
        assert!(rig.run().status.success());
        assert_eq!(code(&rig), "request_stale");
    }
    // A request that is a symlink (or a directory) is not followed.
    let secret = rig.dir.path().join("secret.txt");
    fs::write(&secret, "do not read").unwrap();
    std::os::unix::fs::symlink(&secret, rig.request()).unwrap();
    assert!(rig.run().status.success());
    assert_eq!(code(&rig), "bad_request");
    assert!(!rig.request().exists() && secret.exists());
    fs::create_dir(rig.request()).unwrap();
    assert!(rig.run().status.success());
    assert_eq!(code(&rig), "bad_request");
    assert!(!rig.request().exists());
    // A result file that is a symlink is replaced, not followed.
    let victim = rig.dir.path().join("victim");
    fs::write(&victim, "keep").unwrap();
    let result: &Path = &rig.result_path();
    fs::remove_file(result).unwrap();
    std::os::unix::fs::symlink(&victim, result).unwrap();
    rig.ask("nosuch");
    assert!(rig.run().status.success());
    assert_eq!(fs::read_to_string(&victim).unwrap(), "keep");
    assert_eq!(code(&rig), "proxy_missing");
}

/// Review 5.12 F4: a profile the site made (`managed_by`) must have a public IP-literal host; anything else is not the site's doing and the
/// check does not connect there. A hand-made profile is the owner's own and is checked as before.
#[test]
fn a_site_made_profile_with_a_non_public_host_is_not_connected_to() {
    let rig = Rig::new();
    let code = |rig: &Rig| rig.result()["code"].as_str().unwrap().to_string();
    for host in [
        "10.0.0.1",
        "169.254.169.254",
        "192.168.1.1",
        "example.invalid",
        "0.0.0.0",
    ] {
        rig.profile(
            "site",
            &format!("managed_by = \"ddnet-ai-web\"\nhost = \"{host}\"\nport = 1\n"),
            0o600,
        );
        rig.ask("site");
        let t = std::time::Instant::now();
        assert!(rig.run().status.success());
        assert_eq!(code(&rig), "proxy_host_refused", "{host}");
        assert!(t.elapsed().as_secs() < 3, "{host}: nothing was connected to");
    }
    // The same host in a hand-made profile is checked (here: nothing listens, a connect failure).
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    rig.profile("hand", &format!("host = \"127.0.0.1\"\nport = {port}\n"), 0o600);
    rig.ask("hand");
    assert!(rig.run().status.success());
    assert_eq!(code(&rig), "connect_failed");
}
