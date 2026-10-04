//! Task 5.9 (D-089): `ddnet-ai launch apply` / `launch exited` end to end against a **fake `systemctl`** (a shell script first on
//! `PATH` that logs its arguments and answers `show`), with every path pointed into a temporary directory. Nothing here touches
//! the real systemd, a game server or any production file: the checks are the environment file the bot unit would read, the
//! cgroup drop-in, the exact `systemctl` commands, the status the web would read, and every refusal.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::{Value, json};

const SHIM: &str = r#"#!/bin/bash
echo "$*" >> "$SHIM_DIR/calls.log"
case "$1" in
  show)
    prop="$2"; unit="${@: -1}"
    case "$prop" in
      --property=ActiveState)
        # A test hook: the exit hook records a ban right after the helper's first look at the state (review F4).
        if [ "$unit" = ddnet-ai-bot.service ] && [ -e "$SHIM_DIR/inject-ban" ]; then
          n=$(date +%s); mkdir -p "$(dirname "$SHIM_STATE")"
          echo "{\"last_exit\":{\"at\":$n,\"code\":3},\"blocked\":{\"203.0.113.5:8308\":{\"at\":$n,\"code\":3}}}" > "$SHIM_STATE"
        fi
        if [ -e "$SHIM_DIR/active-$unit" ]; then echo active; else echo inactive; fi ;;
      --property=DropInPaths) cat "$SHIM_DIR/dropins" 2>/dev/null || true ;;
    esac ;;
  *) if [ -e "$SHIM_DIR/fail-$1" ]; then echo "fake failure" >&2; exit 1; fi ;;
esac
exit 0
"#;

const PROXY_IP: &str = "198.51.100.7";
const SERVER_IP: &str = "203.0.113.5";
const PROXY_USER: &str = "u-s3cr3t-user";
const PROXY_PASS: &str = "p-s3cr3t-pass";

struct Rig {
    dir: tempfile::TempDir,
}

impl Rig {
    fn new() -> Rig {
        let rig = Rig {
            dir: tempfile::tempdir().unwrap(),
        };
        fs::create_dir_all(rig.shim()).unwrap();
        let script = rig.shim().join("systemctl");
        fs::write(&script, SHIM).unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        fs::create_dir_all(rig.data().join("launch")).unwrap();
        fs::create_dir_all(rig.data().join("secrets")).unwrap();
        let bundle = rig.data().join("runs/E-005/e005-fly/checkpoints/final.bundle");
        fs::create_dir_all(bundle.parent().unwrap()).unwrap();
        fs::write(bundle, b"bundle").unwrap();
        fs::write(rig.shim().join("active-ddnet-local.service"), b"").unwrap();
        rig
    }

    fn p(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }
    fn shim(&self) -> PathBuf {
        self.p("shim")
    }
    fn data(&self) -> PathBuf {
        self.p("data")
    }
    fn request(&self) -> PathBuf {
        self.data().join("launch/request.json")
    }
    fn live_servers(&self) -> PathBuf {
        self.data().join("live-servers.toml")
    }

    fn base(&self, sub: &[&str]) -> Command {
        // Everything runs under umask 077, like the bot unit's stop hook (UMask=0077): modes must not depend on it.
        let mut c = Command::new("sh");
        c.args(["-c", "umask 077; exec \"$@\"", "sh", env!("CARGO_BIN_EXE_ddnet-ai")]);
        c.arg("launch").args(sub);
        c.arg("--status-dir").arg(self.p("status"));
        c.arg("--data-dir").arg(self.data());
        c.arg("--config").arg(self.p("launch.toml"));
        c.arg("--env-file").arg(self.p("etc/bot-launch.env"));
        c.arg("--dropin").arg(self.p("etc/50-launch.conf"));
        c.arg("--state").arg(self.p("var/state.json"));
        c.env("SHIM_DIR", self.shim());
        c.env("SHIM_STATE", self.p("var/state.json"));
        c.env("PATH", format!("{}:/usr/bin:/bin", self.shim().display()));
        c
    }

    fn apply(&self) -> Output {
        self.base(&["apply"]).output().unwrap()
    }

    fn exited(&self, exit_code: &str, exit_status: &str) -> Output {
        self.base(&["exited"])
            .env("EXIT_CODE", exit_code)
            .env("EXIT_STATUS", exit_status)
            .output()
            .unwrap()
    }

    fn send(&self, body: &Value) -> Output {
        fs::write(self.request(), serde_json::to_vec(body).unwrap()).unwrap();
        self.apply()
    }

    fn status(&self) -> Value {
        serde_json::from_slice(&fs::read(self.p("status/status.json")).expect("a status was written")).unwrap()
    }

    fn calls(&self) -> Vec<String> {
        fs::read_to_string(self.shim().join("calls.log"))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    /// The calls that change something (everything but `show`).
    fn actions(&self) -> Vec<String> {
        self.calls().into_iter().filter(|c| !c.starts_with("show ")).collect()
    }

    fn env_file(&self) -> String {
        fs::read_to_string(self.p("etc/bot-launch.env")).unwrap()
    }

    fn dropin(&self) -> String {
        fs::read_to_string(self.p("etc/50-launch.conf")).unwrap()
    }

    fn allow_list(&self, toml: &str) {
        fs::write(self.live_servers(), toml).unwrap();
    }

    fn proxy_file(&self) {
        self.proxy_file_with("");
    }

    /// The proxy file plus extra lines (for example `relay = "public"`, task 2.6b).
    fn proxy_file_with(&self, extra: &str) {
        let path = self.data().join("secrets/swarfey-proxy.toml");
        fs::write(
            &path,
            format!(
                "host = \"{PROXY_IP}\"\nport = 1080\nuser = \"{PROXY_USER}\"\npass = \"{PROXY_PASS}\"\nfor_server = \"{SERVER_IP}:8308\"\n{extra}"
            ),
        )
        .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn start(server: &str) -> Value {
    json!({"v":1,"id":"0123456789abcdef","ts":now(),"action":"start","brain":"hybrid-fly","server":server,"duration":"60m","sparring":0})
}

fn public_entry(proxy: bool, ready: bool) -> String {
    let proxy = if proxy { "proxy = \"swarfey\"\n" } else { "" };
    format!("[[server]]\naddress = \"{SERVER_IP}:8308\"\nnick = \"Muha\"\nready = {ready}\n{proxy}")
}

fn reason(status: &Value) -> &str {
    status["reason"].as_str().unwrap_or("")
}

#[test]
fn a_local_start_with_two_sparring_writes_the_env_the_dropin_and_runs_the_commands() {
    let rig = Rig::new();
    let mut body = start("local");
    body["sparring"] = json!(2);
    let out = rig.send(&body);
    assert!(out.status.success(), "{out:?}");

    let bundle = rig.data().join("runs/E-005/e005-fly/checkpoints/final.bundle");
    let env = rig.env_file();
    assert!(env.contains("BOT_SERVER=\"127.0.0.1:8303\"\n"), "{env}");
    assert!(env.contains("BOT_NAME=\"Muha\"\n"), "{env}");
    assert!(env.contains("BOT_BRAIN=\"hybrid\"\n"), "{env}");
    assert!(env.contains("BOT_DURATION=\"3600\"\n"), "{env}");
    assert!(
        env.contains(&format!("BOT_FLY_ARGS=\"--fly-bundle {}\"\n", bundle.display())),
        "{env}"
    );
    // Every non-comment line is KEY="value".
    for line in env.lines().filter(|l| !l.starts_with('#')) {
        assert!(line.contains("=\"") && line.ends_with('"'), "{line}");
    }
    let dropin = rig.dropin();
    assert_eq!(dropin.matches("IPAddressAllow=").count(), 2, "{dropin}");
    assert!(dropin.contains("IPAddressAllow=127.0.0.0/8 ::1\n"), "{dropin}");

    assert_eq!(
        rig.actions(),
        vec![
            "daemon-reload",
            "reset-failed ddnet-ai-bot.service ddnet-ai-sparring@1.service ddnet-ai-sparring@2.service ddnet-ai-sparring@3.service",
            "start ddnet-ai-bot.service",
            "start ddnet-ai-sparring@1.service ddnet-ai-sparring@2.service",
        ]
    );
    let status = rig.status();
    assert_eq!(status["state"], "started");
    assert_eq!(status["request_id"], "0123456789abcdef");
    assert_eq!(status["brain"], "hybrid-fly");
    assert_eq!(status["server"], "local");
    assert_eq!(status["sparring"], 2);
    assert_eq!(status["bundle"], "E-005/e005-fly");
    // The request is consumed, and nothing of the bundle's path reaches the status.
    assert!(!rig.request().exists());
    assert!(!status.to_string().contains(bundle.to_str().unwrap()));
}

#[test]
fn plain_hybrid_has_no_bundle_argument_and_unlimited_means_duration_zero() {
    let rig = Rig::new();
    let mut body = start("local");
    body["brain"] = json!("hybrid");
    body["duration"] = json!("unlimited");
    assert!(rig.send(&body).status.success());
    let env = rig.env_file();
    assert!(
        env.contains("BOT_FLY_ARGS=\"\"\n") && env.contains("BOT_DURATION=\"0\"\n"),
        "{env}"
    );
    assert_eq!(rig.actions().last().unwrap(), "start ddnet-ai-bot.service");
    assert_eq!(rig.status()["bundle"], Value::Null);
}

#[test]
fn stop_stops_the_bot_and_all_sparring_units() {
    let rig = Rig::new();
    assert!(rig.send(&start("local")).status.success());
    let out = rig.send(&json!({"v":1,"id":"fedcba9876543210","ts":now(),"action":"stop"}));
    assert!(out.status.success());
    let actions = rig.actions();
    let n = actions.len();
    assert_eq!(
        actions[n - 1],
        "daemon-reload",
        "the plain local drop-in is loaded again"
    );
    assert_eq!(actions[n - 3], "stop ddnet-ai-bot.service");
    assert_eq!(
        actions[n - 2],
        "stop ddnet-ai-sparring@1.service ddnet-ai-sparring@2.service ddnet-ai-sparring@3.service"
    );
    let status = rig.status();
    assert_eq!(status["state"], "stopped");
    assert_eq!(status["reason"], "stopped_by_owner");
    assert_eq!(status["request_id"], "fedcba9876543210");
}

#[test]
fn every_bad_request_is_refused_and_changes_nothing() {
    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        (
            "unknown field",
            serde_json::to_vec(&{
                let mut v = start("local");
                v["cmd"] = json!("rm -rf /");
                v
            })
            .unwrap(),
            "bad_request",
        ),
        (
            "bad brain",
            serde_json::to_vec(&{
                let mut v = start("local");
                v["brain"] = json!("planner");
                v
            })
            .unwrap(),
            "bad_request",
        ),
        (
            "bad duration",
            serde_json::to_vec(&{
                let mut v = start("local");
                v["duration"] = json!("2h");
                v
            })
            .unwrap(),
            "bad_request",
        ),
        ("garbage", b"not json".to_vec(), "bad_request"),
        ("empty", Vec::new(), "bad_request"),
        ("oversize", vec![b' '; 4096], "request_too_large"),
        (
            "free-form public address",
            serde_json::to_vec(&start("1.2.3.4:8308")).unwrap(),
            "server_not_allowed",
        ),
        (
            "an address that is only almost the listed one",
            serde_json::to_vec(&start(&format!("{SERVER_IP}:8309"))).unwrap(),
            "server_not_allowed",
        ),
        (
            "loopback spelled out",
            serde_json::to_vec(&start("127.0.0.1:8303")).unwrap(),
            "server_not_allowed",
        ),
        (
            "listed but not ready",
            serde_json::to_vec(&start(&format!("{SERVER_IP}:8308"))).unwrap(),
            "server_not_ready",
        ),
        (
            "sparring on a public server",
            serde_json::to_vec(&{
                let mut v = start(&format!("{SERVER_IP}:8308"));
                v["sparring"] = json!(1);
                v
            })
            .unwrap(),
            "server_not_ready",
        ),
        (
            "too many sparring",
            serde_json::to_vec(&{
                let mut v = start("local");
                v["sparring"] = json!(4);
                v
            })
            .unwrap(),
            "bad_request",
        ),
    ];
    for (name, body, code) in cases {
        let rig = Rig::new();
        rig.allow_list(&public_entry(true, false));
        fs::write(rig.request(), &body).unwrap();
        let out = rig.apply();
        assert!(out.status.success(), "{name}: {out:?}");
        assert_eq!(reason(&rig.status()), code, "{name}");
        assert_eq!(rig.status()["state"], "refused", "{name}");
        assert!(rig.actions().is_empty(), "{name}: {:?}", rig.actions());
        assert!(!rig.request().exists(), "{name}: the request must be consumed");
        assert!(!rig.p("etc/bot-launch.env").exists(), "{name}");
    }
}

#[test]
fn a_symlink_request_is_refused_unread_and_removed() {
    let rig = Rig::new();
    let victim = rig.p("victim.json");
    fs::write(&victim, serde_json::to_vec(&start("local")).unwrap()).unwrap();
    std::os::unix::fs::symlink(&victim, rig.request()).unwrap();
    assert!(rig.apply().status.success());
    assert_eq!(reason(&rig.status()), "request_not_regular");
    assert!(rig.actions().is_empty());
    assert!(fs::symlink_metadata(rig.request()).is_err(), "the link is removed");
    assert!(victim.exists(), "the target is untouched");
}

#[test]
fn a_directory_in_place_of_the_request_is_refused_and_removed() {
    let rig = Rig::new();
    fs::create_dir_all(rig.request().join("sub")).unwrap();
    fs::write(rig.request().join("sub/f"), b"x").unwrap();
    assert!(rig.apply().status.success());
    assert_eq!(reason(&rig.status()), "request_not_regular");
    assert!(
        fs::symlink_metadata(rig.request()).is_err(),
        "the directory is gone, so the path unit does not spin"
    );
    assert!(rig.actions().is_empty());
}

#[test]
fn a_status_symlink_planted_in_the_status_dir_is_replaced_not_followed() {
    let rig = Rig::new();
    let victim = rig.p("victim.txt");
    fs::write(&victim, b"keep").unwrap();
    fs::create_dir_all(rig.p("status")).unwrap();
    std::os::unix::fs::symlink(&victim, rig.p("status/status.json")).unwrap();
    assert!(rig.send(&start("local")).status.success());
    assert_eq!(fs::read(&victim).unwrap(), b"keep");
    assert_eq!(rig.status()["state"], "started");
}

#[test]
fn no_request_file_is_a_quiet_no_op() {
    let rig = Rig::new();
    assert!(rig.apply().status.success());
    assert!(rig.actions().is_empty());
}

#[test]
fn a_second_start_within_thirty_seconds_is_refused() {
    let rig = Rig::new();
    assert!(rig.send(&start("local")).status.success());
    assert_eq!(rig.status()["state"], "started");
    let before = rig.actions().len();
    assert!(rig.send(&start("local")).status.success());
    assert_eq!(reason(&rig.status()), "rate_limited");
    assert_eq!(rig.actions().len(), before, "nothing more was run");
}

#[test]
fn a_running_bot_is_never_started_over_and_the_local_server_must_be_up() {
    let rig = Rig::new();
    fs::write(rig.shim().join("active-ddnet-ai-bot.service"), b"").unwrap();
    assert!(rig.send(&start("local")).status.success());
    assert_eq!(reason(&rig.status()), "already_running");
    assert!(rig.actions().is_empty());

    let rig = Rig::new();
    fs::remove_file(rig.shim().join("active-ddnet-local.service")).unwrap();
    assert!(rig.send(&start("local")).status.success());
    assert_eq!(reason(&rig.status()), "local_server_down");
    assert!(rig.actions().is_empty());
}

#[test]
fn a_foreign_dropin_on_the_bot_unit_refuses_the_start() {
    let rig = Rig::new();
    fs::write(
        rig.shim().join("dropins"),
        "/etc/systemd/system/ddnet-ai-bot.service.d/swarfey.conf\n",
    )
    .unwrap();
    assert!(rig.send(&start("local")).status.success());
    assert_eq!(reason(&rig.status()), "unit_overridden");
    assert!(rig.actions().is_empty());
    // Our own drop-in alone is fine.
    let rig = Rig::new();
    fs::write(
        rig.shim().join("dropins"),
        format!("{}\n", rig.p("etc/50-launch.conf").display()),
    )
    .unwrap();
    assert!(rig.send(&start("local")).status.success());
    assert_eq!(rig.status()["state"], "started");
}

#[test]
fn a_failing_systemctl_start_is_reported_as_an_error() {
    let rig = Rig::new();
    fs::write(rig.shim().join("fail-start"), b"").unwrap();
    assert!(rig.send(&start("local")).status.success());
    let status = rig.status();
    assert_eq!(
        (status["state"].as_str(), reason(&status)),
        (Some("error"), "systemctl_failed")
    );
}

#[test]
fn a_proxied_public_server_gets_exactly_the_proxy_ip_in_its_cgroup_filter_and_no_secret_leaks() {
    let rig = Rig::new();
    rig.allow_list(&public_entry(true, true));
    rig.proxy_file();
    let mut body = start(&format!("{SERVER_IP}:8308"));
    body["brain"] = json!("hybrid");
    let out = rig.send(&body);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());

    let env = rig.env_file();
    assert!(env.contains(&format!("BOT_SERVER=\"{SERVER_IP}:8308\"\n")), "{env}");
    assert!(env.contains("BOT_NAME=\"Muha\"\n"), "{env}");
    let dropin = rig.dropin();
    assert!(dropin.contains(&format!("IPAddressAllow={PROXY_IP}\n")), "{dropin}");
    // Not the game server's own address: a bot that tried it directly would be dropped by the kernel filter.
    assert!(!dropin.contains(SERVER_IP), "{dropin}");
    // Secrets never reach the env file, the status, the output or the command log.
    let everything = format!(
        "{env}{}{}{}{}",
        rig.status(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
        rig.calls().join("\n")
    );
    for secret in [PROXY_USER, PROXY_PASS, PROXY_IP] {
        assert!(!everything.contains(secret), "{secret} leaked");
    }
    assert_eq!(rig.status()["sparring"], 0);
    assert!(
        !rig.actions()
            .iter()
            .any(|a| a.contains("sparring@") && a.starts_with("start"))
    );
}

#[test]
fn a_direct_public_entry_gets_the_server_ip_and_a_bad_proxy_setup_is_a_refusal() {
    let rig = Rig::new();
    rig.allow_list(&public_entry(false, true));
    assert!(rig.send(&start(&format!("{SERVER_IP}:8308"))).status.success());
    assert_eq!(rig.status()["state"], "started");
    assert!(rig.dropin().contains(&format!("IPAddressAllow={SERVER_IP}\n")));

    // A proxy named by the entry but without a (safe) file: refused, nothing is written or started.
    let rig = Rig::new();
    rig.allow_list(&public_entry(true, true));
    assert!(rig.send(&start(&format!("{SERVER_IP}:8308"))).status.success());
    assert_eq!(reason(&rig.status()), "proxy_error");
    assert!(rig.actions().is_empty());
    rig.proxy_file();
    let path = rig.data().join("secrets/swarfey-proxy.toml");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(rig.send(&start(&format!("{SERVER_IP}:8308"))).status.success());
    assert_eq!(reason(&rig.status()), "proxy_error");
    assert!(rig.actions().is_empty());
}

/// Task 2.6b: a proxy whose file says `relay = "public"` (its UDP relay is on another host) gets the opposite filter:
/// both lists reset, every IP of the game server denied, no allow list, so the relay is reachable anywhere and the server
/// is not reachable directly.
#[test]
fn a_public_relay_proxy_gets_a_deny_the_server_filter_with_no_allow_list_and_no_secret_leaks() {
    let rig = Rig::new();
    rig.allow_list(&public_entry(true, true));
    rig.proxy_file_with("relay = \"public\"\n");
    let mut body = start(&format!("{SERVER_IP}:8308"));
    body["brain"] = json!("hybrid");
    let out = rig.send(&body);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    let dropin = rig.dropin();
    let lines: Vec<&str> = dropin.lines().filter(|l| !l.starts_with('#')).collect();
    assert_eq!(
        lines,
        [
            "[Service]",
            "IPAddressAllow=",
            "IPAddressDeny=",
            &format!("IPAddressDeny={SERVER_IP}"),
            // The server is IPv4: the IPv6 family is denied too (2.6b review F3), then the private ranges.
            "IPAddressDeny=::/0",
            "IPAddressDeny=10.0.0.0/8",
            "IPAddressDeny=172.16.0.0/12",
            "IPAddressDeny=192.168.0.0/16",
            "IPAddressDeny=169.254.0.0/16",
            "IPAddressDeny=100.64.0.0/10",
            "IPAddressDeny=fc00::/7",
            "IPAddressDeny=fe80::/10",
        ],
        "{dropin}"
    );
    // Neither the proxy's IP nor loopback nor `any` appears: nothing is allowed.
    assert!(
        !dropin.contains(PROXY_IP) && !dropin.contains("127.0.0.0/8") && !dropin.contains("any"),
        "{dropin}"
    );
    let env = rig.env_file();
    assert!(env.contains(&format!("BOT_SERVER=\"{SERVER_IP}:8308\"\n")), "{env}");
    let everything = format!(
        "{env}{dropin}{}{}{}{}",
        rig.status(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
        rig.calls().join("\n")
    );
    for secret in [PROXY_USER, PROXY_PASS, PROXY_IP] {
        assert!(!everything.contains(secret), "{secret} leaked");
    }
    // The units were started as usual: reload, then the bot.
    let actions = rig.actions();
    assert!(actions.iter().any(|a| a == "daemon-reload"), "{actions:?}");
    assert!(
        actions
            .iter()
            .any(|a| a.starts_with("start") && a.contains("ddnet-ai-bot")),
        "{actions:?}"
    );
}

#[test]
fn the_default_relay_mode_keeps_the_allow_the_proxy_filter_and_a_public_run_ends_with_the_plain_local_default() {
    // An explicit "proxy-host-only" is the old filter exactly.
    let rig = Rig::new();
    rig.allow_list(&public_entry(true, true));
    rig.proxy_file_with("relay = \"proxy-host-only\"\n");
    assert!(rig.send(&start(&format!("{SERVER_IP}:8308"))).status.success());
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    let dropin = rig.dropin();
    assert!(dropin.contains(&format!("IPAddressAllow={PROXY_IP}\n")), "{dropin}");
    assert!(!dropin.contains("IPAddressDeny"), "{dropin}");
    // After a deny-the-server run ends, the next hand start is back to the plain local default: no deny line left.
    let rig = Rig::new();
    rig.allow_list(&public_entry(true, true));
    rig.proxy_file_with("relay = \"public\"\n");
    assert!(rig.send(&start(&format!("{SERVER_IP}:8308"))).status.success());
    assert!(rig.dropin().contains(&format!("IPAddressDeny={SERVER_IP}\n")));
    assert!(rig.exited("exited", "0").status.success());
    let dropin = rig.dropin();
    assert!(
        dropin.contains("IPAddressAllow=127.0.0.0/8 ::1")
            && !dropin.contains("IPAddressDeny")
            && !dropin.contains(SERVER_IP),
        "{dropin}"
    );
}

#[test]
fn a_bad_relay_value_in_the_proxy_file_is_a_refusal_not_a_filter() {
    let rig = Rig::new();
    rig.allow_list(&public_entry(true, true));
    rig.proxy_file_with("relay = \"anywhere\"\n");
    assert!(rig.send(&start(&format!("{SERVER_IP}:8308"))).status.success());
    assert_eq!(reason(&rig.status()), "proxy_error");
    assert!(rig.actions().is_empty());
}

#[test]
fn a_kick_or_ban_on_a_public_server_blocks_it_until_the_allow_list_is_edited() {
    let rig = Rig::new();
    rig.allow_list(&public_entry(true, true));
    rig.proxy_file();
    let public = format!("{SERVER_IP}:8308");
    assert!(rig.send(&start(&public)).status.success());
    assert_eq!(rig.status()["state"], "started");

    // The unit's ExecStopPost hook: the bot ended with exit 3.
    assert!(rig.exited("exited", "3").status.success());
    let status = rig.status();
    assert_eq!(
        (status["state"].as_str(), reason(&status)),
        (Some("failed"), "kicked_or_banned")
    );
    assert_eq!(status["exit_code"], 3);
    assert_eq!(status["server"], public.as_str());

    let actions = rig.actions().len();
    assert!(rig.send(&start(&public)).status.success());
    assert_eq!(reason(&rig.status()), "blocked_after_ban");
    // Any start (local too) waits out the cool-down after exit 3.
    assert!(rig.send(&start("local")).status.success());
    assert_eq!(reason(&rig.status()), "cooldown");
    assert_eq!(rig.actions().len(), actions, "nothing was started");

    // The owner re-opens the entry (edits the file after the ban): the block is lifted, the cool-down still applies.
    let file = fs::OpenOptions::new().write(true).open(rig.live_servers()).unwrap();
    file.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(60))
        .unwrap();
    assert!(rig.send(&start(&public)).status.success());
    assert_eq!(reason(&rig.status()), "cooldown");
}

#[test]
fn exits_are_recorded_honestly_and_a_crash_blocks_nothing() {
    let rig = Rig::new();
    assert!(rig.send(&start("local")).status.success());
    // A clean exit right after the start of a 60-minute run is an outside SIGTERM or a hand stop, not «time is up».
    assert!(rig.exited("exited", "0").status.success());
    assert_eq!(reason(&rig.status()), "ended");
    assert_eq!(rig.status()["state"], "stopped");
    // The same exit once the 60 minutes have passed (the state's start time moved back) is «finished».
    let state_path = rig.p("var/state.json");
    let mut state: Value = serde_json::from_slice(&fs::read(&state_path).unwrap()).unwrap();
    state["last_start_at"] = json!(now() - 3600);
    fs::write(&state_path, serde_json::to_vec(&state).unwrap()).unwrap();
    assert!(rig.exited("exited", "0").status.success());
    assert_eq!(reason(&rig.status()), "finished");
    // The one-run environment is gone after a normal end.
    assert!(!rig.p("etc/bot-launch.env").exists());
    // A crash: systemd restarts the unit, with the same environment (a new rig: the start rate limit is real time).
    let rig = Rig::new();
    assert!(rig.send(&start("local")).status.success());
    assert!(rig.p("etc/bot-launch.env").exists());
    assert!(rig.exited("exited", "1").status.success());
    assert!(rig.p("etc/bot-launch.env").exists(), "kept for the restart");
    assert_eq!(reason(&rig.status()), "crashed");
    assert_eq!(rig.status()["state"], "failed");
    assert!(rig.exited("killed", "SEGV").status.success());
    assert_eq!(reason(&rig.status()), "crashed");
    // Exit 4 on the local server: a cool-down, no ban memory.
    assert!(rig.exited("exited", "4").status.success());
    assert_eq!(reason(&rig.status()), "join_failed");
    assert!(!rig.p("etc/bot-launch.env").exists());
    assert!(rig.send(&start("local")).status.success());
    assert_eq!(reason(&rig.status()), "cooldown");
}

#[test]
fn a_stop_then_the_bots_exit_reads_as_stopped_by_the_owner() {
    let rig = Rig::new();
    assert!(rig.send(&start("local")).status.success());
    assert!(
        rig.send(&json!({"v":1,"id":"fedcba9876543210","ts":now(),"action":"stop"}))
            .status
            .success()
    );
    assert!(rig.exited("exited", "0").status.success());
    assert_eq!(reason(&rig.status()), "stopped_by_owner");
}

#[test]
fn an_unreadable_state_file_refuses_starts_and_is_left_alone() {
    let rig = Rig::new();
    fs::create_dir_all(rig.p("var")).unwrap();
    fs::write(rig.p("var/state.json"), b"{ broken").unwrap();
    assert!(rig.send(&start("local")).status.success());
    assert_eq!(reason(&rig.status()), "state_unreadable");
    assert_eq!(fs::read(rig.p("var/state.json")).unwrap(), b"{ broken");
    assert!(rig.actions().is_empty());
}

fn mode_of(path: &std::path::Path) -> u32 {
    fs::metadata(path).unwrap().permissions().mode() & 0o777
}

#[test]
fn files_have_the_documented_modes_whatever_the_umask() {
    // Every command in this file runs under umask 077 (see `Rig::base`), like the bot unit's ExecStopPost hook.
    let rig = Rig::new();
    assert!(rig.send(&start("local")).status.success());
    assert_eq!(mode_of(&rig.p("status/status.json")), 0o644);
    assert_eq!(mode_of(&rig.p("status")), 0o755);
    assert_eq!(mode_of(&rig.p("etc/bot-launch.env")), 0o644);
    assert_eq!(mode_of(&rig.p("etc/50-launch.conf")), 0o644);
    assert_eq!(mode_of(&rig.p("var/state.json")), 0o600);
    // The hook (a new status after the bot ended by itself) is readable by the web too.
    fs::remove_file(rig.p("status/status.json")).unwrap();
    assert!(rig.exited("exited", "3").status.success());
    assert_eq!(
        mode_of(&rig.p("status/status.json")),
        0o644,
        "the web must be able to read what the hook writes"
    );
    assert_eq!(rig.status()["exit_code"], 3);
    // The directory is created 0755 when the hook is the first to write.
    fs::remove_dir_all(rig.p("status")).unwrap();
    assert!(rig.exited("exited", "0").status.success());
    assert_eq!(mode_of(&rig.p("status")), 0o755);
    assert_eq!(mode_of(&rig.p("status/status.json")), 0o644);
}

#[test]
fn a_stale_future_or_old_mtime_request_is_refused_discarded_and_changes_nothing() {
    let cases: Vec<(&str, u64, i64)> = vec![
        // (name, ts, mtime offset in seconds from now)
        ("old ts", now() - 120, 0),
        ("two days old", now() - 2 * 86400, -2 * 86400),
        ("future ts", now() + 600, 0),
        ("old mtime with a new body", now(), -120),
        ("future mtime", now(), 600),
    ];
    for (name, ts, mtime_offset) in cases {
        let rig = Rig::new();
        let mut body = start("local");
        body["ts"] = json!(ts);
        fs::write(rig.request(), serde_json::to_vec(&body).unwrap()).unwrap();
        let t = if mtime_offset >= 0 {
            std::time::SystemTime::now() + std::time::Duration::from_secs(mtime_offset as u64)
        } else {
            std::time::SystemTime::now() - std::time::Duration::from_secs(mtime_offset.unsigned_abs())
        };
        fs::OpenOptions::new()
            .write(true)
            .open(rig.request())
            .unwrap()
            .set_modified(t)
            .unwrap();
        assert!(rig.apply().status.success(), "{name}");
        assert_eq!(reason(&rig.status()), "request_stale", "{name}");
        assert_eq!(rig.status()["state"], "refused", "{name}");
        assert!(rig.actions().is_empty(), "{name}: {:?}", rig.actions());
        assert!(!rig.request().exists(), "{name}: discarded");
        assert!(!rig.p("etc/bot-launch.env").exists(), "{name}");
    }
    // A request without a ts is not a request of this protocol.
    let rig = Rig::new();
    let mut body = start("local");
    body.as_object_mut().unwrap().remove("ts");
    assert!(rig.send(&body).status.success());
    assert_eq!(reason(&rig.status()), "bad_request");
}

#[test]
fn a_ban_recorded_between_the_first_look_and_the_start_is_not_erased() {
    let rig = Rig::new();
    rig.allow_list(&public_entry(true, true));
    rig.proxy_file();
    fs::write(rig.shim().join("inject-ban"), b"").unwrap();
    assert!(rig.send(&start(&format!("{SERVER_IP}:8308"))).status.success());
    assert_eq!(reason(&rig.status()), "blocked_after_ban");
    assert!(rig.actions().is_empty(), "{:?}", rig.actions());
    assert!(!rig.p("etc/bot-launch.env").exists());
    let state: Value = serde_json::from_slice(&fs::read(rig.p("var/state.json")).unwrap()).unwrap();
    assert_eq!(
        state["blocked"][format!("{SERVER_IP}:8308")]["code"],
        3,
        "the ban is still in the memory"
    );
    assert!(
        state["last_start_at"].as_u64().unwrap_or(0) == 0,
        "and no start was recorded"
    );
}

#[test]
fn after_a_public_run_ends_the_next_hand_start_is_the_plain_local_default() {
    let rig = Rig::new();
    rig.allow_list(&public_entry(true, true));
    rig.proxy_file();
    assert!(rig.send(&start(&format!("{SERVER_IP}:8308"))).status.success());
    assert!(rig.dropin().contains(PROXY_IP));
    assert!(rig.exited("exited", "0").status.success());
    assert!(!rig.p("etc/bot-launch.env").exists());
    let dropin = rig.dropin();
    assert!(
        !dropin.contains(PROXY_IP) && dropin.contains("IPAddressAllow=127.0.0.0/8 ::1"),
        "{dropin}"
    );
    // The same after a stop request.
    let rig = Rig::new();
    rig.allow_list(&public_entry(true, true));
    rig.proxy_file();
    assert!(rig.send(&start(&format!("{SERVER_IP}:8308"))).status.success());
    assert!(
        rig.send(&json!({"v":1,"id":"fedcba9876543210","ts":now(),"action":"stop"}))
            .status
            .success()
    );
    assert!(!rig.p("etc/bot-launch.env").exists());
    assert!(!rig.dropin().contains(PROXY_IP));
    assert_eq!(rig.actions().last().unwrap(), "daemon-reload");
}

#[test]
fn an_ipv6_server_entry_is_allowed() {
    let rig = Rig::new();
    rig.allow_list("[[server]]\naddress = \"[2001:db8::7]:8308\"\nnick = \"Muha\"\nready = true\n");
    assert!(rig.send(&start("[2001:db8::7]:8308")).status.success());
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    assert!(rig.env_file().contains("BOT_SERVER=\"[2001:db8::7]:8308\"\n"));
    assert!(rig.dropin().contains("IPAddressAllow=2001:db8::7\n"));
}
