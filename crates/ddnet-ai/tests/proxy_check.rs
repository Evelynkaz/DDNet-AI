//! Task 2.6: `ddnet-ai proxy-check` and the proxy gate of `play`, run as the real binary against the in-process
//! loopback SOCKS5 server (127.0.0.1 only; no game server involved, no network).

use ddai_client::socks5_testserver::{Auth, Bnd, Config, TestSocks5Server};
use std::net::UdpSocket;
use std::path::Path;
use std::process::{Command, Output};
use std::time::Duration;

const USER: &str = "tester-user-4711";
const PASS: &str = "tester-pass-9274";

fn write_secret(dir: &Path, name: &str, body: &str, mode: u32) {
    let secrets = dir.join("secrets");
    std::fs::create_dir_all(&secrets).unwrap();
    let path = secrets.join(format!("{name}-proxy.toml"));
    std::fs::write(&path, body).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
}

fn proxy_body(server: &TestSocks5Server, user: &str, pass: &str) -> String {
    format!(
        "# a comment\nhost = \"{}\"\nport = {}\nuser = \"{user}\"\npass = \"{pass}\"\nfor_server = \"127.0.0.1:1\"\nnote = \"y\"\n",
        server.addr().ip(),
        server.addr().port()
    )
}

fn check(dir: &Path, name: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ddnet-ai"))
        .args(["proxy-check", "--proxy", name, "--data-dir"])
        .arg(dir)
        .output()
        .expect("run ddnet-ai proxy-check")
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

fn assert_no_secrets(server: &TestSocks5Server, out: &str) {
    for s in [USER, PASS, &server.addr().port().to_string(), "127.0.0.1"] {
        assert!(!out.contains(s), "leaked {s:?}: {out}");
    }
}

#[test]
fn ok_with_credentials_prints_ok_and_reveals_nothing() {
    let server = TestSocks5Server::start(Config {
        auth: Auth::UserPass(USER.into(), PASS.into()),
        ..Default::default()
    });
    let dir = tempfile::tempdir().unwrap();
    write_secret(dir.path(), "p", &proxy_body(&server, USER, PASS), 0o600);
    let out = check(dir.path(), "p");
    assert_eq!(out.status.code(), Some(0), "{}", text(&out));
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        stdout.contains("ok") && stdout.contains("UDP ASSOCIATE accepted"),
        "{stdout}"
    );
    assert_no_secrets(&server, &text(&out));
    // Only the proxy was contacted, with no UDP datagram anywhere.
    assert_eq!(server.tcp_accepts(), 1);
    std::thread::sleep(Duration::from_millis(100));
    assert!(server.datagrams_from_clients().is_empty());
}

#[test]
fn ok_without_credentials() {
    let server = TestSocks5Server::start(Config::default());
    let dir = tempfile::tempdir().unwrap();
    write_secret(
        dir.path(),
        "p",
        &format!("host = \"127.0.0.1\"\nport = {}\n", server.addr().port()),
        0o600,
    );
    let out = check(dir.path(), "p");
    assert_eq!(out.status.code(), Some(0), "{}", text(&out));
    assert!(String::from_utf8_lossy(&out.stdout).contains("no authentication needed"));
}

#[test]
fn a_tcp_only_proxy_reports_udp_not_supported() {
    let server = TestSocks5Server::start(Config {
        auth: Auth::UserPass(USER.into(), PASS.into()),
        reply_code: 0x07,
        ..Default::default()
    });
    let dir = tempfile::tempdir().unwrap();
    write_secret(dir.path(), "p", &proxy_body(&server, USER, PASS), 0o600);
    let out = check(dir.path(), "p");
    assert_eq!(out.status.code(), Some(3), "{}", text(&out));
    assert!(text(&out).contains("UDP not supported"), "{}", text(&out));
    assert_no_secrets(&server, &text(&out));
}

#[test]
fn a_wrong_password_reports_auth_failed() {
    let server = TestSocks5Server::start(Config {
        auth: Auth::UserPass(USER.into(), PASS.into()),
        ..Default::default()
    });
    let dir = tempfile::tempdir().unwrap();
    write_secret(dir.path(), "p", &proxy_body(&server, USER, "not-the-pass-1234"), 0o600);
    let out = check(dir.path(), "p");
    assert_eq!(out.status.code(), Some(4), "{}", text(&out));
    assert!(text(&out).contains("auth failed"), "{}", text(&out));
    assert!(!text(&out).contains("not-the-pass-1234"));
    assert_no_secrets(&server, &text(&out));
}

#[test]
fn a_group_readable_file_is_refused_without_contacting_the_proxy() {
    let server = TestSocks5Server::start(Config::default());
    let dir = tempfile::tempdir().unwrap();
    write_secret(dir.path(), "p", &proxy_body(&server, USER, PASS), 0o640);
    let out = check(dir.path(), "p");
    assert_eq!(out.status.code(), Some(1), "{}", text(&out));
    assert!(text(&out).contains("chmod 600"), "{}", text(&out));
    assert_no_secrets(&server, &text(&out));
    assert_eq!(server.tcp_accepts(), 0);
}

#[test]
fn missing_file_and_bad_names_fail_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("secrets")).unwrap();
    for name in ["absent", "../x", "a/b", ""] {
        let out = check(dir.path(), name);
        assert_eq!(out.status.code(), Some(1), "{name:?}: {}", text(&out));
        assert!(!text(&out).contains("panicked"), "{name:?}: {}", text(&out));
    }
}

/// `play` against a loopback game server whose allow-list entry names a proxy: if the proxy cannot be loaded the
/// bot refuses to start and sends nothing, never falling back to a direct connection.
#[test]
fn play_refuses_a_named_proxy_it_cannot_load_and_sends_nothing_to_the_game_server() {
    let game = UdpSocket::bind("127.0.0.1:0").unwrap();
    game.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
    let game_addr = game.local_addr().unwrap();
    let server = TestSocks5Server::start(Config::default());
    for (label, setup) in [
        ("missing file", None),
        ("group-readable file", Some(0o644u32)),
        ("world-readable file", Some(0o666)),
    ] {
        let dir = tempfile::tempdir().unwrap();
        if let Some(mode) = setup {
            write_secret(dir.path(), "named", &proxy_body(&server, USER, PASS), mode);
        }
        let list = dir.path().join("live-servers.toml");
        std::fs::write(
            &list,
            format!("[[server]]\naddress = \"{game_addr}\"\nnick = \"Muha\"\nready = true\nproxy = \"named\"\n"),
        )
        .unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_ddnet-ai"))
            .args([
                "play",
                "--server",
                &game_addr.to_string(),
                "--name",
                "Muha",
                "--brain",
                "idle",
            ])
            .args(["--duration", "3", "--data-dir"])
            .arg(dir.path())
            .arg("--live-servers")
            .arg(&list)
            .output()
            .expect("run ddnet-ai play");
        assert!(!out.status.success(), "{label}: {}", text(&out));
        assert!(text(&out).contains("refusing to connect"), "{label}: {}", text(&out));
        // (The game server is on 127.0.0.1 too, so only the credentials and the proxy's port are checked.)
        for s in [USER, PASS, &server.addr().port().to_string()] {
            assert!(!text(&out).contains(s), "{label}: leaked {s:?}: {}", text(&out));
        }
    }
    let mut buf = [0u8; 64];
    assert!(game.recv_from(&mut buf).is_err(), "a datagram reached the game server");
    assert_eq!(server.tcp_accepts(), 0);
}

/// F1: a proxy announcing a relay on another host is reported, and what is printed says it is not trusted.
#[test]
fn a_relay_announced_on_another_host_is_reported_as_substituted() {
    let server = TestSocks5Server::start(Config {
        bnd: Bnd::OtherIp("10.1.2.3".parse().unwrap()),
        ..Default::default()
    });
    let dir = tempfile::tempdir().unwrap();
    write_secret(
        dir.path(),
        "p",
        &format!("host = \"127.0.0.1\"\nport = {}\n", server.addr().port()),
        0o600,
    );
    let out = check(dir.path(), "p");
    assert_eq!(out.status.code(), Some(0), "{}", text(&out));
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        stdout.contains("NOT trusted") && stdout.contains("announced on another host"),
        "{stdout}"
    );
    assert!(!stdout.contains("10.1.2.3"), "{stdout}");
}

/// F4: the proxy file's `for_server` is enforced by `play` too: an allow-list that names the proxy for another
/// server is refused before anything is sent.
#[test]
fn play_refuses_a_proxy_issued_for_another_server() {
    let game = UdpSocket::bind("127.0.0.1:0").unwrap();
    game.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
    let game_addr = game.local_addr().unwrap();
    let server = TestSocks5Server::start(Config::default());
    let dir = tempfile::tempdir().unwrap();
    write_secret(
        dir.path(),
        "named",
        &format!(
            "host = \"127.0.0.1\"\nport = {}\nfor_server = \"45.141.57.35:8308\"\n",
            server.addr().port()
        ),
        0o600,
    );
    let list = dir.path().join("live-servers.toml");
    std::fs::write(
        &list,
        format!("[[server]]\naddress = \"{game_addr}\"\nnick = \"Muha\"\nready = true\nproxy = \"named\"\n"),
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_ddnet-ai"))
        .args([
            "play",
            "--server",
            &game_addr.to_string(),
            "--name",
            "Muha",
            "--brain",
            "idle",
        ])
        .args(["--duration", "3", "--data-dir"])
        .arg(dir.path())
        .arg("--live-servers")
        .arg(&list)
        .output()
        .expect("run ddnet-ai play");
    // Exit 4: the unit's `RestartPreventExitStatus` lists it, so systemd does not keep restarting a refusal.
    assert_eq!(out.status.code(), Some(4), "{}", text(&out));
    assert!(text(&out).contains("issued for another server"), "{}", text(&out));
    assert!(!text(&out).contains("45.141.57.35"), "{}", text(&out));
    let mut buf = [0u8; 64];
    assert!(game.recv_from(&mut buf).is_err(), "a datagram reached the game server");
    assert_eq!(server.tcp_accepts(), 0);
}
