//! Task 2.6 e2e: the bot plays on the local DDNet 20.1 server (`ddnet-local.service`, 127.0.0.1:8303, econ 8304)
//! **through a loopback SOCKS5 relay** (the in-process `TestSocks5Server`, UDP relay on 127.0.0.1). `#[ignore]`d
//! and guarded by `DDAI_E2E=1`; run it once, by hand:
//!
//! ```text
//! DDAI_E2E=1 cargo test -p ddnet-ai --test e2e_socks5 -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Private server copy instead of the local one: set `DDAI_E2E_SERVER` (e.g. `127.0.0.1:8343`),
//! `DDAI_E2E_ECON_PORT` (`8344`), `DDAI_E2E_ECON_PASSWORD_FILE` (a `.cfg` with `ec_password "..."`) and
//! `DDAI_E2E_RESTART_CMD` (a shell command that restarts that server).
//!
//! Needs: `ddnet-local.service` running, passwordless `sudo systemctl restart ddnet-local.service` (as
//! `tools/e2e/session.sh` does), `python3` (`tools/ddnet-server/econ.py`, `tools/e2e/analyze_positions.py`).
//! Loopback only; the proxy is ours. It uses a **scratch data dir with its own allowlist** (`live-servers.toml`
//! with one loopback entry that names the proxy `e2e`, and `secrets/e2e-proxy.toml`, mode 0600), never the
//! owner's `live-servers.toml`, secrets or `data/bot`. The server is put back on `Copy Love Box` at the end
//! (always, also on failure) and the value is read back.
//!
//! Phase 1, the bot (`--bot --brain scripted`, with the outgoing-message audit):
//! 1. joins through the relay: the server's `status` lists it with an address whose port is a relay socket's
//!    (so the game traffic provably came out of the relay, not directly);
//! 2. survives two map changes (`BlmapChill`, then back);
//! 3. the proxy drops its TCP control connection: the association dies, the bot reconnects through a **new** one;
//! 4. the server restarts (`systemctl restart`, a graceful `Server shutdown` close): the bot reconnects through
//!    the proxy again;
//! 5. stops on SIGTERM with exit 0, no give-up, and the audit shows **0 chat** (no `Cl_Say`, not even `/kill`).
//!
//! Phase 2: `--brain circle` through the relay moves right then left (`tools/e2e/analyze_positions.py`).
//!
//! A second test (task 2.6b, `e2e_client_through_a_relay_on_another_host_in_public_mode`) puts the relay on **another
//! loopback address** (127.0.0.2) in `relay = "public"` mode and joins a private server through it with the real driver
//! (`Client::connect`): the server's own `status` must list the client at 127.0.0.2:<relay port>. It runs in this
//! process because the test hook that makes loopback count as a public relay address exists only in the `test-util`
//! build of the library and can never be set from a proxy file, so the `ddnet-ai` binary cannot be used for it. It never
//! changes the map and never restarts the server; use a private server copy (`DDAI_E2E_SERVER=127.0.0.1:8393`,
//! `DDAI_E2E_ECON_PORT=8394`, `DDAI_E2E_ECON_PASSWORD_FILE=<cfg>`):
//!
//! ```text
//! DDAI_E2E=1 cargo test -p ddnet-ai --test e2e_socks5 e2e_client_through_a_relay -- --ignored --nocapture
//! ```

use ddai_client::live_servers::{LiveServerEntry, LiveServers};
use ddai_client::proxy::{ProxyConfig, RelayMode};
use ddai_client::socks5_testserver::{Auth, Config, TestSocks5Server};
use ddai_client::{Client, ClientConfig, ClientEvent, SessionEvent};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The game server under test: `DDAI_E2E_SERVER` (default `127.0.0.1:8303`, the local `ddnet-local.service`).
fn server_addr() -> String {
    std::env::var("DDAI_E2E_SERVER").unwrap_or_else(|_| "127.0.0.1:8303".to_string())
}
const ORIGINAL_MAP: &str = "Copy Love Box";
/// A fresh name each run: a bot killed by an earlier failed run leaves a slot on the server until its 100 s timeout.
fn nick() -> &'static str {
    static NICK: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NICK.get_or_init(|| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        format!("e2es{:04}", secs % 10_000)
    })
}
const USER: &str = "e2e-user";
const PASS: &str = "e2e-pass";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/ddnet-ai is two levels under the repo root")
        .to_path_buf()
}

/// The last `status` output, for the timeout message of [`wait_for`].
static LAST_STATUS: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

fn econ(args: &[&str]) -> String {
    let mut cmd = Command::new("python3");
    cmd.arg(repo_root().join("tools/ddnet-server/econ.py"));
    // A private server copy: `DDAI_E2E_ECON_PORT` and `DDAI_E2E_ECON_PASSWORD_FILE` (a .cfg with `ec_password "..."`).
    if let Ok(port) = std::env::var("DDAI_E2E_ECON_PORT") {
        cmd.args(["--port", &port]);
    }
    if let Ok(file) = std::env::var("DDAI_E2E_ECON_PASSWORD_FILE") {
        cmd.args(["--password-file", &file]);
    }
    let out = cmd.args(args).output().expect("run econ.py");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if args == ["status"] {
        *LAST_STATUS.lock().unwrap() = text.clone();
    }
    text
}

/// Puts the server back on the original map when the test ends, however it ends, and reads it back.
struct RestoreMap;

impl Drop for RestoreMap {
    fn drop(&mut self) {
        econ(&["change_map", ORIGINAL_MAP]);
        std::thread::sleep(Duration::from_secs(2));
        let back = econ(&["sv_map"]);
        eprintln!(
            "restore: sv_map reads back: {}",
            back.lines().find(|l| l.contains("Value:")).unwrap_or("?").trim()
        );
        assert!(
            back.contains(ORIGINAL_MAP),
            "sv_map was not restored to {ORIGINAL_MAP:?}: {back}"
        );
    }
}

/// Kills the bot if the test fails halfway, so no stray process keeps playing.
struct KillOnDrop(Option<Child>);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        if let Some(c) = self.0.as_mut() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

fn wait_for(limit: Duration, what: &str, mut cond: impl FnMut() -> bool) {
    let end = Instant::now() + limit;
    while Instant::now() < end {
        if cond() {
            return;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    panic!(
        "timed out ({limit:?}) waiting for: {what}\nlast status output:\n{}",
        LAST_STATUS.lock().unwrap()
    );
}

/// The server's own view of our client: the `addr=<{ip:port}>` of the `status` line with our name.
fn server_sees(name: &str) -> Option<String> {
    let out = econ(&["status"]);
    let line = out.lines().find(|l| l.contains(&format!("name='{name}'")))?;
    let start = line.find("addr=<{")? + "addr=<{".len();
    let end = line[start..].find("}>")? + start;
    Some(line[start..end].to_string())
}

/// Whether the server lists a client named like ours (a duplicate name gets a prefix) that connects from the relay
/// socket on `port`: the server's own proof of which path our traffic took. Found by port, not by name, because
/// after a lost control connection the old, dead slot (same name) lives on until the server's timeout.
fn server_lists_relay_port(port: u16) -> bool {
    let needle = format!("addr=<{{127.0.0.1:{port}}}>");
    econ(&["status"])
        .lines()
        .any(|l| l.contains(&needle) && l.contains(nick()))
}

fn bot_args(data: &Path, list: &Path) -> Vec<String> {
    let server = server_addr();
    [
        "play",
        "--server",
        &server,
        "--name",
        nick(),
        "--data-dir",
        data.to_str().unwrap(),
        "--live-servers",
        list.to_str().unwrap(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

#[test]
#[ignore = "touches the local ddnet-local.service (map changes, a restart); run with DDAI_E2E=1"]
fn e2e_bot_through_a_loopback_socks5_relay() {
    if std::env::var_os("DDAI_E2E").as_deref() != Some(std::ffi::OsStr::new("1")) {
        eprintln!("skipping: set DDAI_E2E=1 to run against the local server");
        return;
    }
    let _restore = RestoreMap;
    let proxy = TestSocks5Server::start(Config {
        auth: Auth::UserPass(USER.into(), PASS.into()),
        ..Default::default()
    });

    // Scratch data dir: own allowlist, own secrets.
    // Kept on failure (the logs are the evidence); removed at the end of a good run.
    let dir = tempfile::Builder::new()
        .prefix("ddai-e2e-socks5-")
        .tempdir()
        .expect("scratch dir")
        .keep();
    eprintln!("scratch data dir: {}", dir.display());
    let data = dir.join("data");
    std::fs::create_dir_all(data.join("secrets")).unwrap();
    let server = server_addr();
    let secret = data.join("secrets").join("e2e-proxy.toml");
    std::fs::write(
        &secret,
        format!(
            "host = \"{}\"\nport = {}\nuser = \"{USER}\"\npass = \"{PASS}\"\nfor_server = \"{server}\"\n",
            proxy.addr().ip(),
            proxy.addr().port()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&secret, std::fs::Permissions::from_mode(0o600)).unwrap();
    let list = data.join("live-servers.toml");
    std::fs::write(
        &list,
        format!(
            "[[server]]\naddress = \"{server}\"\nnick = \"{}\"\npurpose = \"e2e socks5\"\nready = true\nproxy = \"e2e\"\n",
            nick()
        ),
    )
    .unwrap();

    assert!(
        server_sees(nick()).is_none(),
        "a client named {} is already on the local server",
        nick()
    );

    // ---- Phase 1: the bot with the audit ------------------------------------------------------------
    let report = data.join("report.json");
    let log = data.join("bot.log");
    let mut args = bot_args(&data, &list);
    args.extend(
        [
            "--brain",
            "scripted",
            "--duration",
            "0",
            "--timeout-secs",
            "6",
            "--no-console",
            "--no-control",
            "--no-bridge",
            "--no-settings",
            "--report",
            report.to_str().unwrap(),
        ]
        .map(String::from),
    );
    let child = Command::new(env!("CARGO_BIN_EXE_ddnet-ai"))
        .args(&args)
        .env("RUST_LOG", "info")
        .env("NO_COLOR", "1")
        .stdout(Stdio::from(std::fs::File::create(&log).unwrap()))
        .stderr(Stdio::from(std::fs::File::create(data.join("bot.err")).unwrap()))
        .spawn()
        .expect("spawn the bot");
    let pid = child.id();
    let mut bot = KillOnDrop(Some(child));

    // 1. Joined, through the relay.
    wait_for(Duration::from_secs(30), "the bot to join through the relay", || {
        server_sees(nick()).is_some()
    });
    let first_addr = server_sees(nick()).unwrap();
    let relay_ports: Vec<u16> = vec![proxy.relay_addr().port()];
    eprintln!(
        "joined; server sees the client at port {}, relay port {}",
        first_addr.rsplit(':').next().unwrap(),
        relay_ports[0]
    );
    assert_eq!(
        first_addr.rsplit(':').next().unwrap().parse::<u16>().unwrap(),
        proxy.relay_addr().port(),
        "the server must see the relay's port, not the bot's own socket"
    );
    assert_eq!(proxy.tcp_accepts(), 1);
    std::thread::sleep(Duration::from_secs(4));

    // 2. Two map changes.
    econ(&["change_map", "BlmapChill"]);
    std::thread::sleep(Duration::from_secs(8));
    assert!(
        server_lists_relay_port(proxy.relay_addr().port()),
        "the bot left during the map change"
    );
    econ(&["change_map", ORIGINAL_MAP]);
    std::thread::sleep(Duration::from_secs(8));
    assert!(
        server_lists_relay_port(proxy.relay_addr().port()),
        "the bot left during the map change back"
    );
    assert_eq!(
        proxy.tcp_accepts(),
        1,
        "map changes must not touch the proxy connection"
    );

    // 3. The proxy drops the control connection: the association is gone, the bot comes back through a new one.
    let accepts_before = proxy.tcp_accepts();
    proxy.drop_control_connections();
    wait_for(
        Duration::from_secs(40),
        "a new association after the control connection was dropped",
        || proxy.associations() >= 2,
    );
    let second_relay = proxy.relay_addr();
    wait_for(
        Duration::from_secs(40),
        "the bot back in game through the new association",
        || server_lists_relay_port(second_relay.port()),
    );
    assert_eq!(
        proxy.tcp_accepts(),
        accepts_before + 1,
        "exactly one new proxy connection for the one loss"
    );
    std::thread::sleep(Duration::from_secs(4));

    // 4. A server restart: the bot reconnects through the proxy again.
    let accepts_before = proxy.tcp_accepts();
    let restart = std::env::var("DDAI_E2E_RESTART_CMD")
        .unwrap_or_else(|_| "sudo -n systemctl restart ddnet-local.service".to_string());
    let status = Command::new("sh")
        .args(["-c", &restart])
        .status()
        .expect("run the restart command");
    assert!(
        status.success(),
        "the restart command failed: {restart} (passwordless sudo is needed for the default)"
    );
    wait_for(
        Duration::from_secs(60),
        "the bot back in game after the server restart",
        || proxy.tcp_accepts() > accepts_before,
    );
    let third_relay = proxy.relay_addr();
    wait_for(
        Duration::from_secs(60),
        "the server to list the bot on the new relay port",
        || server_lists_relay_port(third_relay.port()),
    );
    std::thread::sleep(Duration::from_secs(5));

    // 5. Stop on SIGTERM: a polite disconnect and the report.
    let killed = Command::new("kill").args(["-TERM", &pid.to_string()]).status().unwrap();
    assert!(killed.success());
    let mut child = bot.0.take().unwrap();
    let end = Instant::now() + Duration::from_secs(30);
    let exit = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        assert!(Instant::now() < end, "the bot did not stop within 30 s of SIGTERM");
        std::thread::sleep(Duration::from_millis(200));
    };
    let stdout = std::fs::read_to_string(&log).unwrap_or_default();
    let stderr = std::fs::read_to_string(data.join("bot.err")).unwrap_or_default();
    assert!(
        exit.success(),
        "bot exit {exit:?}\n--- stdout\n{stdout}\n--- stderr (tail)\n{}",
        tail(&stderr, 40)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report).expect("the bot wrote its report")).unwrap();
    assert_eq!(report["exit_code"], 0, "{report}");
    assert!(report["gave_up"].is_null(), "{}", report["gave_up"]);
    assert!(
        report["stats"]["snapshots"].as_u64().unwrap() > 500,
        "{}",
        report["stats"]
    );
    assert!(
        report["stats"]["decisions"].as_u64().unwrap() > 100,
        "{}",
        report["stats"]
    );

    // The audit: 0 chat. Every outgoing message is a join/protocol one, none is `Cl_Say`, not even `/kill`.
    let outgoing = report["outgoing_game_messages"].as_object().expect("audit present");
    assert!(!outgoing.is_empty(), "the audit saw nothing: it was not on");
    for (label, v) in outgoing {
        assert!(
            !label.contains("Say") && !label.contains("Chat"),
            "chat on the wire: {label} {v}"
        );
        assert_eq!(v["refused"], 0, "the allow-list refused {label}");
    }
    eprintln!(
        "audit: {} outgoing message kinds, 0 chat: {:?}",
        outgoing.len(),
        outgoing.keys().collect::<Vec<_>>()
    );

    // The lifecycle as the driver logged it: it really went through the three associations.
    assert_eq!(
        proxy.associations(),
        3,
        "initial + after the dropped control connection + after the restart"
    );
    let all_logs = format!("{stdout}\n{stderr}");
    for secret in [USER, PASS] {
        assert!(!all_logs.contains(secret), "a credential reached the log");
    }
    eprintln!(
        "phase 1 ok: {} associations, {} proxy TCP connections",
        proxy.associations(),
        proxy.tcp_accepts()
    );

    // ---- Phase 2: movement through the relay ---------------------------------------------------------
    let circle_log = data.join("circle.log");
    let mut args = bot_args(&data, &list);
    args.extend(["--brain", "circle", "--duration", "14"].map(String::from));
    let out = Command::new(env!("CARGO_BIN_EXE_ddnet-ai"))
        .args(&args)
        .env("RUST_LOG", "info")
        .env("NO_COLOR", "1")
        .output()
        .expect("run the circle bot");
    std::fs::write(
        &circle_log,
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
    .unwrap();
    assert!(out.status.success(), "circle bot exit {:?}", out.status);
    let analysis = Command::new("python3")
        .arg(repo_root().join("tools/e2e/analyze_positions.py"))
        .arg(&circle_log)
        .output()
        .expect("run analyze_positions.py");
    let verdict = String::from_utf8_lossy(&analysis.stdout).to_string();
    assert!(
        verdict.contains("PASS"),
        "the tee did not move through the relay: {verdict}{}",
        String::from_utf8_lossy(&analysis.stderr)
    );
    eprintln!("phase 2 ok: {}", verdict.lines().last().unwrap_or(""));
    assert_eq!(proxy.associations(), 4);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "talks to a private local DDNet server; run with DDAI_E2E=1 (see the module docs)"]
fn e2e_client_through_a_relay_on_another_host_in_public_mode() {
    if std::env::var_os("DDAI_E2E").as_deref() != Some(std::ffi::OsStr::new("1")) {
        eprintln!("skipping: set DDAI_E2E=1 to run against a private local server");
        return;
    }
    let server = server_addr();
    let server_sock: std::net::SocketAddr = server.parse().expect("DDAI_E2E_SERVER is ip:port");
    assert!(server_sock.ip().is_loopback(), "the e2e talks to a local server only");
    assert!(
        server_sees(nick()).is_none(),
        "a client named {} is already on the server",
        nick()
    );
    let proxy = TestSocks5Server::start(Config {
        auth: Auth::UserPass(USER.into(), PASS.into()),
        relay_ip: Some("127.0.0.2".parse().unwrap()),
        ..Default::default()
    });
    let a = proxy.addr();
    let cfg = ClientConfig {
        name: nick().to_string(),
        live_servers: LiveServers {
            servers: vec![LiveServerEntry {
                address: server.clone(),
                nick: nick().to_string(),
                purpose: "e2e socks5 public relay".to_string(),
                ready: true,
                proxy: Some("e2e".to_string()),
            }],
        },
        proxy: Some(
            ProxyConfig::new(
                "e2e",
                a.ip().to_string(),
                a.port(),
                Some((USER.to_string(), PASS.to_string())),
            )
            .unwrap()
            .with_relay(RelayMode::Public)
            .with_test_loopback_relay()
            .with_for_server(server.clone()),
        ),
        ..ClientConfig::default()
    };
    let mut client = Client::connect(server_sock, cfg);
    let end = Instant::now() + Duration::from_secs(30);
    let (mut connected, mut positions) = (false, 0u32);
    while Instant::now() < end && !(connected && positions >= 5) {
        match client.recv_event(Duration::from_millis(100)) {
            Some(ClientEvent::Session(ev)) if matches!(*ev, SessionEvent::Connected) => connected = true,
            Some(ClientEvent::OwnPosition { .. }) => positions += 1,
            Some(ClientEvent::GaveUp { reason, .. }) => panic!("the client gave up: {reason}"),
            _ => {}
        }
    }
    assert!(
        connected && positions >= 5,
        "no join through the relay: {connected} {positions}"
    );
    let relay = proxy.relay_addr();
    assert_eq!(
        relay.ip().to_string(),
        "127.0.0.2",
        "the relay is on another host than the proxy"
    );
    // The server's own view: the client's address is the relay's (IP and port), not the bot's socket.
    let seen = server_sees(nick()).expect("the server lists the client");
    eprintln!("server sees the client at {seen}; relay is {relay}");
    assert_eq!(seen, relay.to_string(), "the server must see the relay's address");
    client.disconnect();
    client.join();
    assert_eq!(proxy.associations(), 1);
    assert_eq!(proxy.tcp_accepts(), 1);
    assert!(
        !proxy.datagrams_from_clients().is_empty(),
        "the game traffic went through the relay"
    );
}

fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}
