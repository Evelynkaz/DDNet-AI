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
/// A favourite is a public unicast address (a documentation address such as `SERVER_IP` is refused by the strict favourites rules).
const FAV_IP: &str = "93.184.216.34";
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

    /// The favourites file the site writes (`data/launch/favourites.json`): `(address, connection, reopened_at)` each.
    fn favourites(&self, list: &[(&str, &str, u64)]) {
        let favs: Vec<Value> = list
            .iter()
            .map(|(address, connection, reopened_at)| {
                json!({"address": address, "name": "Some Block Server", "nick": "Muha2", "connection": connection,
                       "consent_at": now(), "notes": "", "added_at": now(), "reopened_at": reopened_at})
            })
            .collect();
        fs::write(
            self.data().join("launch/favourites.json"),
            serde_json::to_vec(&json!({"v": 1, "favourites": favs})).unwrap(),
        )
        .unwrap();
    }

    /// A profile `<name>-proxy.toml` as the site writes it (0600).
    fn profile(&self, name: &str, extra: &str) {
        let path = self.data().join(format!("secrets/{name}-proxy.toml"));
        fs::write(
            &path,
            format!("host = \"{PROXY_IP}\"\nport = 1080\nuser = \"{PROXY_USER}\"\npass = \"{PROXY_PASS}\"\n{extra}"),
        )
        .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    }

    /// Moves the helper's memory of the last start and exit back by `secs`: the real-time cool-down and start interval are over.
    fn age_state(&self, secs: u64) {
        let path = self.p("var/state.json");
        let mut state: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let start = state["last_start_at"].as_u64().unwrap_or(0);
        state["last_start_at"] = json!(start.saturating_sub(secs));
        if let Some(at) = state["last_exit"]["at"].as_u64() {
            state["last_exit"]["at"] = json!(at.saturating_sub(secs));
        }
        fs::write(path, serde_json::to_vec(&state).unwrap()).unwrap();
    }

    fn blocked_file(&self) -> Value {
        serde_json::from_slice(&fs::read(self.p("status/blocked.json")).expect("blocked.json was written")).unwrap()
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
    // No `mirror` in the request: the opponent model stays on (D-090).
    assert!(env.contains("BOT_HYBRID_MIRROR=\"on\"\n"), "{env}");
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
fn the_opponent_model_switch_is_written_from_a_closed_list() {
    let rig = Rig::new();
    let mut body = start("local");
    body["mirror"] = json!("off");
    assert!(rig.send(&body).status.success());
    let env = rig.env_file();
    assert!(env.contains("BOT_HYBRID_MIRROR=\"off\"\n"), "{env}");
    // An explicit `on` is written as `on`; a rewritten start replaces the file.
    let rig = Rig::new();
    let mut body = start("local");
    body["mirror"] = json!("on");
    assert!(rig.send(&body).status.success());
    assert!(rig.env_file().contains("BOT_HYBRID_MIRROR=\"on\"\n"));
    // Anything else (or a mirror on a stop) is refused and changes nothing.
    for bad in [
        json!("maybe"),
        json!("ON"),
        json!(true),
        json!("off; rm -rf /"),
        json!(""),
    ] {
        let rig = Rig::new();
        let mut body = start("local");
        body["mirror"] = bad.clone();
        let out = rig.send(&body);
        assert!(out.status.success(), "{out:?}");
        assert_eq!(reason(&rig.status()), "bad_request", "{bad}");
        assert!(rig.actions().is_empty(), "{bad}: {:?}", rig.actions());
    }
    let rig = Rig::new();
    let out = rig.send(&json!({"v":1,"id":"fedcba9876543210","ts":now(),"action":"stop","mirror":"off"}));
    assert!(out.status.success(), "{out:?}");
    assert_eq!(reason(&rig.status()), "bad_request");
}

#[test]
fn the_finishing_mode_is_written_from_a_closed_list_for_the_local_server_the_allow_list_and_a_favourite() {
    // No field (an old request): `off` is written explicitly, so no value left in a unit's environment can leak into the run.
    let rig = Rig::new();
    assert!(rig.send(&start("local")).status.success());
    assert!(rig.env_file().contains("BOT_FINISH=\"off\"\n"), "{}", rig.env_file());
    assert_eq!(rig.status()["finish"], "off", "{}", rig.status());

    // The local server, both hybrid brains, all three words.
    for brain in ["hybrid", "hybrid-fly"] {
        for word in ["off", "target", "full"] {
            let rig = Rig::new();
            let mut body = start("local");
            body["brain"] = json!(brain);
            body["finish"] = json!(word);
            let out = rig.send(&body);
            assert!(out.status.success(), "{out:?}");
            assert_eq!(rig.status()["state"], "started", "{brain} {word}: {}", rig.status());
            assert!(
                rig.env_file().contains(&format!("BOT_FINISH=\"{word}\"\n")),
                "{brain} {word}: {}",
                rig.env_file()
            );
            assert_eq!(rig.status()["finish"], word);
            assert_eq!(rig.actions().last().unwrap(), "start ddnet-ai-bot.service");
            // The whole file keeps its shape: one line per key, nothing but the closed character set.
            for line in rig.env_file().lines().filter(|l| !l.starts_with('#')) {
                assert!(line.contains("=\""), "{line}");
            }
        }
    }

    // An allow-list entry (public, through a proxy).
    let rig = Rig::new();
    rig.allow_list(&public_entry(true, true));
    rig.proxy_file();
    let mut body = start(&format!("{SERVER_IP}:8308"));
    body["finish"] = json!("target");
    assert!(rig.send(&body).status.success());
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    assert!(rig.env_file().contains("BOT_FINISH=\"target\"\n"), "{}", rig.env_file());

    // A favourite.
    let rig = Rig::new();
    let addr = format!("{FAV_IP}:8303");
    rig.favourites(&[(&addr, "direct", 0)]);
    let mut body = start(&addr);
    body["finish"] = json!("target");
    assert!(rig.send(&body).status.success());
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    let env = rig.env_file();
    assert!(
        env.contains(&format!("BOT_SERVER=\"{addr}\"\n")) && env.contains("BOT_FINISH=\"target\"\n"),
        "{env}"
    );

    // The mode survives to the status the exit hook writes (it is read from the helper's memory, not from the request).
    let out = rig.exited("exited", "0");
    assert!(out.status.success(), "{out:?}");
    let status = rig.status();
    assert_eq!(
        (status["state"].as_str(), status["finish"].as_str()),
        (Some("stopped"), Some("target")),
        "{status}"
    );
}

#[test]
fn a_bad_finishing_value_is_refused_and_changes_nothing_not_even_an_injection() {
    for bad in [
        json!("on"),
        json!("Target"),
        json!("TARGET"),
        json!("target "),
        json!("target\n"),
        json!("off; rm -rf /"),
        json!("target --report /etc/passwd"),
        json!("target\"\nBOT_SERVER=\"203.0.113.5:8308"),
        json!("target\nBOT_NAME=\"evil"),
        json!("$(id)"),
        json!("`id`"),
        json!(""),
        json!(true),
        json!(1),
        json!(["target"]),
        Value::Null,
    ] {
        let rig = Rig::new();
        let mut body = start("local");
        body["finish"] = bad.clone();
        let out = rig.send(&body);
        assert!(out.status.success(), "{out:?}");
        if bad.is_null() {
            // `null` is the same as no field (serde: an absent option), i.e. `off`.
            assert_eq!(rig.status()["state"], "started", "{}", rig.status());
            assert!(rig.env_file().contains("BOT_FINISH=\"off\"\n"));
            continue;
        }
        assert_eq!(reason(&rig.status()), "bad_request", "{bad}: {}", rig.status());
        assert!(rig.actions().is_empty(), "{bad}: {:?}", rig.actions());
        assert!(
            !rig.p("etc/bot-launch.env").exists(),
            "{bad}: no environment file for a refused request"
        );
        assert!(!rig.request().exists(), "the request is consumed");
    }
    // A stop carries nothing.
    let rig = Rig::new();
    let out = rig.send(&json!({"v":1,"id":"fedcba9876543210","ts":now(),"action":"stop","finish":"off"}));
    assert!(out.status.success(), "{out:?}");
    assert_eq!(reason(&rig.status()), "bad_request");
    assert!(rig.actions().is_empty());
}

#[test]
fn the_pure_fly_takes_no_finishing_but_off() {
    for word in ["target", "full"] {
        let rig = Rig::new();
        let mut body = start("local");
        body["brain"] = json!("fly");
        body["finish"] = json!(word);
        let out = rig.send(&body);
        assert!(out.status.success(), "{out:?}");
        let status = rig.status();
        assert_eq!(
            (status["state"].as_str(), reason(&status)),
            (Some("refused"), "finish_hybrid_only"),
            "{word}"
        );
        assert!(rig.actions().is_empty(), "{word}: {:?}", rig.actions());
        assert!(!rig.p("etc/bot-launch.env").exists());
    }
    // The same request on the favourite path is refused the same way.
    let rig = Rig::new();
    let addr = format!("{FAV_IP}:8303");
    rig.favourites(&[(&addr, "direct", 0)]);
    let mut body = start(&addr);
    body["brain"] = json!("fly");
    body["finish"] = json!("target");
    assert!(rig.send(&body).status.success());
    assert_eq!(reason(&rig.status()), "finish_hybrid_only");
    assert!(rig.actions().is_empty());
    // `off` (or nothing) is fine for the fly, and the line says off.
    for finish in [Some("off"), None] {
        let rig = Rig::new();
        let mut body = start("local");
        body["brain"] = json!("fly");
        if let Some(f) = finish {
            body["finish"] = json!(f);
        }
        assert!(rig.send(&body).status.success());
        assert_eq!(rig.status()["state"], "started", "{}", rig.status());
        assert!(rig.env_file().contains("BOT_BRAIN=\"fly\"\n") && rig.env_file().contains("BOT_FINISH=\"off\"\n"));
    }
}

#[test]
fn the_smart_wayblock_and_the_duel_switch_are_written_always_from_closed_values_for_every_brain_and_server_kind() {
    // Task 5.15. No fields (an old request): both are written explicitly as off / false, so no value left in a unit's environment can leak.
    let rig = Rig::new();
    assert!(rig.send(&start("local")).status.success());
    let env = rig.env_file();
    assert!(
        env.contains("BOT_WB_SMART=\"off\"\n") && env.contains("BOT_NO_SELFKILL=\"false\"\n"),
        "{env}"
    );
    let st = rig.status();
    assert_eq!(
        (st["wb_smart"].as_str(), st["no_selfkill"].as_bool()),
        (Some("off"), Some(false)),
        "{st}"
    );

    // Every brain (the pure fly included: both are bot-level navigation, target and self-kill rules), both values of both fields.
    for brain in ["hybrid", "hybrid-fly", "fly"] {
        for (wb, ns) in [("off", false), ("on", false), ("off", true), ("on", true)] {
            let rig = Rig::new();
            let mut body = start("local");
            body["brain"] = json!(brain);
            body["wb_smart"] = json!(wb);
            body["no_selfkill"] = json!(ns);
            let out = rig.send(&body);
            assert!(out.status.success(), "{out:?}");
            let st = rig.status();
            assert_eq!(st["state"], "started", "{brain} {wb} {ns}: {st}");
            let env = rig.env_file();
            assert!(
                env.contains(&format!("BOT_WB_SMART=\"{wb}\"\n"))
                    && env.contains(&format!("BOT_NO_SELFKILL=\"{ns}\"\n"))
                    && env.contains(&format!(
                        "BOT_BRAIN=\"{}\"\n",
                        if brain == "fly" { "fly" } else { "hybrid" }
                    )),
                "{brain} {wb} {ns}: {env}"
            );
            assert_eq!(
                (st["wb_smart"].as_str(), st["no_selfkill"].as_bool()),
                (Some(wb), Some(ns)),
                "{st}"
            );
            assert_eq!(rig.actions().last().unwrap(), "start ddnet-ai-bot.service");
            for line in env.lines().filter(|l| !l.starts_with('#')) {
                assert!(line.contains("=\""), "{line}");
            }
        }
    }

    // An allow-list entry (public, through a proxy) and a favourite.
    let rig = Rig::new();
    rig.allow_list(&public_entry(true, true));
    rig.proxy_file();
    let mut body = start(&format!("{SERVER_IP}:8308"));
    body["wb_smart"] = json!("on");
    body["no_selfkill"] = json!(true);
    assert!(rig.send(&body).status.success());
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    let env = rig.env_file();
    assert!(
        env.contains("BOT_WB_SMART=\"on\"\n") && env.contains("BOT_NO_SELFKILL=\"true\"\n"),
        "{env}"
    );

    let rig = Rig::new();
    let addr = format!("{FAV_IP}:8303");
    rig.favourites(&[(&addr, "direct", 0)]);
    let mut body = start(&addr);
    body["no_selfkill"] = json!(true);
    assert!(rig.send(&body).status.success());
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    let env = rig.env_file();
    assert!(
        env.contains(&format!("BOT_SERVER=\"{addr}\"\n"))
            && env.contains("BOT_NO_SELFKILL=\"true\"\n")
            && env.contains("BOT_WB_SMART=\"off\"\n"),
        "{env}"
    );

    // Both survive to the status the exit hook writes (read from the helper's memory, not from the request).
    let out = rig.exited("exited", "0");
    assert!(out.status.success(), "{out:?}");
    let st = rig.status();
    assert_eq!(
        (
            st["state"].as_str(),
            st["wb_smart"].as_str(),
            st["no_selfkill"].as_bool()
        ),
        (Some("stopped"), Some("off"), Some(true)),
        "{st}"
    );
}

#[test]
fn a_bad_smart_wayblock_or_duel_value_is_refused_and_changes_nothing_not_even_an_injection() {
    let cases: Vec<(&str, Value)> = vec![
        ("wb_smart", json!("true")),
        ("wb_smart", json!("On")),
        ("wb_smart", json!("ON")),
        ("wb_smart", json!("on ")),
        ("wb_smart", json!("on\n")),
        ("wb_smart", json!("off; rm -rf /")),
        ("wb_smart", json!("on --report /etc/passwd")),
        ("wb_smart", json!("on\"\nBOT_SERVER=\"203.0.113.5:8308")),
        ("wb_smart", json!("on\nBOT_NAME=\"evil")),
        ("wb_smart", json!("$(id)")),
        ("wb_smart", json!("`id`")),
        ("wb_smart", json!("")),
        ("wb_smart", json!(true)),
        ("wb_smart", json!(1)),
        ("wb_smart", json!(["on"])),
        ("no_selfkill", json!("true")),
        ("no_selfkill", json!("false")),
        ("no_selfkill", json!("on")),
        ("no_selfkill", json!("true --report /etc/passwd")),
        ("no_selfkill", json!("true\"\nBOT_SERVER=\"203.0.113.5:8308")),
        ("no_selfkill", json!("$(id)")),
        ("no_selfkill", json!("")),
        ("no_selfkill", json!(0)),
        ("no_selfkill", json!(1)),
        ("no_selfkill", json!([true])),
        ("no_selfkill", json!({"on": true})),
    ];
    for (key, bad) in cases {
        let rig = Rig::new();
        let mut body = start("local");
        body[key] = bad.clone();
        let out = rig.send(&body);
        assert!(out.status.success(), "{out:?}");
        assert_eq!(reason(&rig.status()), "bad_request", "{key}={bad}: {}", rig.status());
        assert!(rig.actions().is_empty(), "{key}={bad}: {:?}", rig.actions());
        assert!(
            !rig.p("etc/bot-launch.env").exists(),
            "{key}={bad}: no environment file for a refused request"
        );
        assert!(!rig.request().exists(), "the request is consumed");
    }
    // `null` is the same as no field (serde: an absent option): the defaults.
    for key in ["wb_smart", "no_selfkill"] {
        let rig = Rig::new();
        let mut body = start("local");
        body[key] = Value::Null;
        assert!(rig.send(&body).status.success());
        assert_eq!(rig.status()["state"], "started", "{key}: {}", rig.status());
        let env = rig.env_file();
        assert!(
            env.contains("BOT_WB_SMART=\"off\"\n") && env.contains("BOT_NO_SELFKILL=\"false\"\n"),
            "{env}"
        );
    }
    // A stop carries neither.
    for (key, val) in [("wb_smart", json!("off")), ("no_selfkill", json!(false))] {
        let rig = Rig::new();
        let out = rig.send(&json!({"v":1,"id":"fedcba9876543210","ts":now(),"action":"stop", key: val}));
        assert!(out.status.success(), "{out:?}");
        assert_eq!(reason(&rig.status()), "bad_request", "{key}");
        assert!(rig.actions().is_empty());
    }
}

impl Rig {
    /// The opponent-input model file of the launch card's toggle (task 3.17): `<data>/bot/models/opp-m1.oppnet`.
    fn model(&self) -> PathBuf {
        self.data().join("bot/models/opp-m1.oppnet")
    }

    fn put_model(&self) {
        fs::create_dir_all(self.model().parent().unwrap()).unwrap();
        fs::write(self.model(), b"weights").unwrap();
    }
}

#[test]
fn the_opponent_predictor_is_written_always_empty_unless_asked_and_then_the_helpers_own_path() {
    // Task 3.17 (D-111). No field (an old request) and `false`: the line is there and empty, so no path left in a unit's environment leaks in.
    for body in [start("local"), {
        let mut b = start("local");
        b["window_model"] = json!(false);
        b
    }] {
        let rig = Rig::new();
        assert!(rig.send(&body).status.success());
        assert_eq!(rig.status()["state"], "started", "{}", rig.status());
        assert!(rig.env_file().contains("BOT_WINDOW_MODEL=\"\"\n"), "{}", rig.env_file());
        assert_eq!(rig.status()["window_model"].as_bool(), Some(false));
    }
    // On, with the file there: both hybrid brains get the one path the helper builds, nothing from the request.
    for brain in ["hybrid", "hybrid-fly"] {
        let rig = Rig::new();
        rig.put_model();
        let mut body = start("local");
        body["brain"] = json!(brain);
        body["window_model"] = json!(true);
        assert!(rig.send(&body).status.success());
        let st = rig.status();
        assert_eq!(st["state"], "started", "{brain}: {st}");
        assert_eq!(st["window_model"].as_bool(), Some(true), "{st}");
        let env = rig.env_file();
        assert!(
            env.contains(&format!("BOT_WINDOW_MODEL=\"{}\"\n", rig.model().display()))
                && env.contains("BOT_FINISH=\"off\"\n"),
            "{brain}: {env}"
        );
        assert_eq!(rig.actions().last().unwrap(), "start ddnet-ai-bot.service");
        for line in env.lines().filter(|l| !l.starts_with('#')) {
            assert!(line.contains("=\""), "{line}");
        }
        // It survives to the status the exit hook writes.
        assert!(rig.exited("exited", "0").status.success());
        assert_eq!(rig.status()["window_model"].as_bool(), Some(true));
    }
}

#[test]
fn the_opponent_predictor_is_refused_without_its_file_for_the_pure_fly_and_for_any_value_but_a_boolean() {
    // No file: refused, nothing started (the bot would not start with it either; a restart loop is worse than a refusal).
    let rig = Rig::new();
    let mut body = start("local");
    body["window_model"] = json!(true);
    assert!(rig.send(&body).status.success());
    assert_eq!(reason(&rig.status()), "window_model_missing", "{}", rig.status());
    assert!(rig.actions().is_empty() && !rig.p("etc/bot-launch.env").exists());
    // An empty file, a symlink and a directory are no model.
    for make in [
        (|r: &Rig| {
            fs::create_dir_all(r.model().parent().unwrap()).unwrap();
            fs::write(r.model(), b"").unwrap();
        }) as fn(&Rig),
        |r: &Rig| {
            fs::create_dir_all(r.model().parent().unwrap()).unwrap();
            fs::write(r.data().join("real"), b"x").unwrap();
            std::os::unix::fs::symlink(r.data().join("real"), r.model()).unwrap();
        },
        |r: &Rig| fs::create_dir_all(r.model()).unwrap(),
    ] {
        let rig = Rig::new();
        make(&rig);
        let mut body = start("local");
        body["window_model"] = json!(true);
        assert!(rig.send(&body).status.success());
        assert_eq!(reason(&rig.status()), "window_model_missing", "{}", rig.status());
        assert!(rig.actions().is_empty());
    }
    // The pure fly has no lag window: refused even with the file there; `false` is fine.
    let rig = Rig::new();
    rig.put_model();
    let mut body = start("local");
    body["brain"] = json!("fly");
    body["window_model"] = json!(true);
    assert!(rig.send(&body).status.success());
    assert_eq!(reason(&rig.status()), "window_model_hybrid_only", "{}", rig.status());
    assert!(rig.actions().is_empty());
    let rig = Rig::new();
    let mut body = start("local");
    body["brain"] = json!("fly");
    body["window_model"] = json!(false);
    assert!(rig.send(&body).status.success());
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    // Nothing but a JSON boolean, and a stop carries none: no path, no word, no injection.
    for bad in [
        json!("true"),
        json!("on"),
        json!("/etc/passwd"),
        json!("/home/ubuntu/aiddnet/data/bot/models/opp-m1.oppnet\nBOT_SERVER=\"203.0.113.5:8308\""),
        json!("$(id)"),
        json!(1),
        json!(""),
        json!(["x"]),
        json!({"path": "/etc/passwd"}),
    ] {
        let rig = Rig::new();
        rig.put_model();
        let mut body = start("local");
        body["window_model"] = bad.clone();
        assert!(rig.send(&body).status.success());
        assert_eq!(reason(&rig.status()), "bad_request", "{bad}: {}", rig.status());
        assert!(
            rig.actions().is_empty() && !rig.p("etc/bot-launch.env").exists(),
            "{bad}"
        );
    }
    let rig = Rig::new();
    let out = rig.send(&json!({"v":1,"id":"fedcba9876543210","ts":now(),"action":"stop","window_model":false}));
    assert!(out.status.success());
    assert_eq!(reason(&rig.status()), "bad_request");
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

// ---------------------------------------------------------------------------------------------------------------------
// Task 5.12 (D-099): favourites
// ---------------------------------------------------------------------------------------------------------------------

#[test]
fn a_favourite_start_takes_nick_address_and_filter_from_the_favourite_never_from_the_request() {
    let rig = Rig::new();
    let addr = format!("{FAV_IP}:8303");
    rig.favourites(&[(&addr, "direct", 0)]);
    let mut body = start(&addr);
    body["brain"] = json!("hybrid");
    let out = rig.send(&body);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    assert_eq!(rig.status()["server"], addr.as_str());
    let env = rig.env_file();
    assert!(env.contains(&format!("BOT_SERVER=\"{addr}\"\n")), "{env}");
    assert!(env.contains("BOT_NAME=\"Muha2\"\n"), "the favourite's nick: {env}");
    assert!(
        rig.dropin().contains(&format!("IPAddressAllow={FAV_IP}\n")),
        "{}",
        rig.dropin()
    );
    assert_eq!(rig.actions().last().unwrap(), "start ddnet-ai-bot.service");
    // Public: no sparring, as for every public server.
    let rig = Rig::new();
    rig.favourites(&[(&addr, "direct", 0)]);
    let mut body = start(&addr);
    body["sparring"] = json!(1);
    assert!(rig.send(&body).status.success());
    assert_eq!(reason(&rig.status()), "sparring_local_only");
    assert!(rig.actions().is_empty());
}

#[test]
fn a_favourite_through_a_proxy_gets_that_proxys_filter_and_leaks_no_secret() {
    let addr = format!("{FAV_IP}:8303");
    // Public relay: deny the server, allow nothing.
    let rig = Rig::new();
    rig.profile("hp-pub", "relay = \"public\"\n");
    rig.favourites(&[(&addr, "proxy:hp-pub", 0)]);
    let out = rig.send(&start(&addr));
    assert!(out.status.success(), "{out:?}");
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    let dropin = rig.dropin();
    assert!(
        dropin.contains(&format!("IPAddressDeny={FAV_IP}\n")) && dropin.contains("IPAddressAllow=\n"),
        "{dropin}"
    );
    assert!(
        !dropin.contains("127.0.0.0/8") && !dropin.contains(PROXY_IP),
        "{dropin}"
    );
    // The proxy's own host: allowed, the server is not mentioned.
    let rig = Rig::new();
    rig.profile("hp-host", "");
    rig.favourites(&[(&addr, "proxy:hp-host", 0)]);
    let out = rig.send(&start(&addr));
    assert!(out.status.success(), "{out:?}");
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    let dropin = rig.dropin();
    assert!(
        dropin.contains(&format!("IPAddressAllow={PROXY_IP}\n")) && !dropin.contains(FAV_IP),
        "{dropin}"
    );
    let everything = format!(
        "{}{dropin}{}{}{}{}",
        rig.env_file(),
        rig.status(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
        rig.calls().join("\n")
    );
    // The proxy's IP is in the filter (the unit may talk to it); its login and password are nowhere.
    for secret in [PROXY_USER, PROXY_PASS] {
        assert!(!everything.contains(secret), "{secret} leaked");
    }
}

#[test]
fn a_missing_or_unsafe_proxy_is_a_refusal_never_a_direct_start_or_another_proxy() {
    let addr = format!("{FAV_IP}:8303");
    let rig = Rig::new();
    rig.profile("other", "relay = \"public\"\n");
    rig.favourites(&[(&addr, "proxy:gone", 0)]);
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(reason(&rig.status()), "proxy_error");
    assert!(rig.actions().is_empty(), "nothing started: {:?}", rig.actions());
    assert!(!rig.p("etc/bot-launch.env").exists());
    // A loose mode on the named profile.
    let rig = Rig::new();
    rig.profile("hp", "");
    fs::set_permissions(
        rig.data().join("secrets/hp-proxy.toml"),
        fs::Permissions::from_mode(0o644),
    )
    .unwrap();
    rig.profile("other", "");
    rig.favourites(&[(&addr, "proxy:hp", 0)]);
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(reason(&rig.status()), "proxy_error");
    assert!(rig.actions().is_empty());
    // A bad connection value in the file is refused as such.
    let rig = Rig::new();
    rig.favourites(&[(&addr, "proxy:../secrets/x", 0)]);
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(reason(&rig.status()), "favourites_invalid");
    assert!(rig.actions().is_empty());
}

#[test]
fn what_is_not_a_valid_favourite_is_refused_and_changes_nothing() {
    let addr = format!("{FAV_IP}:8303");
    // (A loopback favourite is refused by the production build; the test binary of `cargo test` is built with the `loopback-favourites`
    // feature for the e2e, so that refusal is tested on the rules themselves: `launch_cmd` unit tests and `ddai_client::favourites`.)
    let bad_files: Vec<(&str, Vec<u8>)> = vec![
        ("not json", b"{ nope".to_vec()),
        ("unknown field", serde_json::to_vec(&json!({"v":1,"favourites":[],"x":1})).unwrap()),
        ("private address", serde_json::to_vec(&json!({"v":1,"favourites":[{"address":"10.0.0.5:8303","name":"n","nick":"Muha","connection":"direct","consent_at":5,"added_at":5}]})).unwrap()),
        ("no consent", serde_json::to_vec(&json!({"v":1,"favourites":[{"address":addr,"name":"n","nick":"Muha","connection":"direct","consent_at":0,"added_at":5}]})).unwrap()),
        ("bad nick", serde_json::to_vec(&json!({"v":1,"favourites":[{"address":addr,"name":"n","nick":"a b;c","connection":"direct","consent_at":5,"added_at":5}]})).unwrap()),
        ("extra key", serde_json::to_vec(&json!({"v":1,"favourites":[{"address":addr,"name":"n","nick":"Muha","connection":"direct","consent_at":5,"added_at":5,"ready":true}]})).unwrap()),
        ("duplicate", serde_json::to_vec(&json!({"v":1,"favourites":[
            {"address":addr,"name":"n","nick":"Muha","connection":"direct","consent_at":5,"added_at":5},
            {"address":addr,"name":"n","nick":"Muha","connection":"direct","consent_at":5,"added_at":5}]})).unwrap()),
        ("too big", vec![b' '; 70_000]),
    ];
    for (what, bytes) in bad_files {
        let rig = Rig::new();
        fs::write(rig.data().join("launch/favourites.json"), &bytes).unwrap();
        for target in [addr.as_str(), "10.0.0.5:8303", "127.0.0.1:8463"] {
            assert!(rig.send(&start(target)).status.success());
            let r = rig.status();
            assert!(
                matches!(reason(&r), "favourites_invalid" | "favourites_unreadable"),
                "{what} / {target}: {r}"
            );
            assert!(rig.actions().is_empty(), "{what}: {:?}", rig.actions());
        }
        // The local server never depends on the favourites file.
        let rig2 = Rig::new();
        fs::write(rig2.data().join("launch/favourites.json"), &bytes).unwrap();
        assert!(rig2.send(&start("local")).status.success());
        assert_eq!(rig2.status()["state"], "started", "{what}");
    }
    // A symlink where the file should be is not followed; a directory is refused.
    let rig = Rig::new();
    let real = rig.data().join("real.json");
    fs::write(&real, serde_json::to_vec(&json!({"v":1,"favourites":[]})).unwrap()).unwrap();
    std::os::unix::fs::symlink(&real, rig.data().join("launch/favourites.json")).unwrap();
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(reason(&rig.status()), "favourites_unreadable");
    // An address that is simply not a favourite.
    let rig = Rig::new();
    rig.favourites(&[(&addr, "direct", 0)]);
    for other in [
        "93.184.216.35:8303",
        "93.184.216.34:8304",
        "1.2.3.4:5",
        "example.com:8303",
        "127.0.0.1:8463",
    ] {
        assert!(rig.send(&start(other)).status.success());
        assert_eq!(reason(&rig.status()), "server_not_allowed", "{other}");
    }
    assert!(rig.actions().is_empty());
}

/// The fair-play core: a kick or ban closes a favourite (and every port on its IP) until the owner's explicit re-open, whatever the
/// favourite's proxy, nick or entry says afterwards, and the bot's own side never changes a proxy.
#[test]
fn a_ban_on_a_favourite_blocks_it_until_the_explicit_reopen_and_no_proxy_switch_gets_round_it() {
    let rig = Rig::new();
    let addr = format!("{FAV_IP}:8303");
    let sibling = format!("{FAV_IP}:8304");
    let other = "45.141.57.35:8308";
    rig.profile("hp-1", "relay = \"public\"\n");
    rig.profile("hp-2", "");
    rig.favourites(&[(&addr, "proxy:hp-1", 0), (&sibling, "direct", 0), (other, "direct", 0)]);
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());

    // The bot is kicked / banned («VPN detected»): exit 3.
    assert!(rig.exited("exited", "3").status.success());
    assert_eq!(reason(&rig.status()), "kicked_or_banned");
    let blocked = rig.blocked_file();
    assert_eq!(blocked["blocked"][0]["address"], addr.as_str());
    assert_eq!(blocked["blocked"][0]["code"], 3);
    let ban_at = blocked["blocked"][0]["at"].as_u64().unwrap();
    assert!(!blocked.to_string().contains(PROXY_PASS), "{blocked}");
    // The unit is back to the plain local default: no proxy IP, no deny line is left allowed for a hand start.
    assert!(!rig.p("etc/bot-launch.env").exists());
    let actions = rig.actions().len();

    // The cool-down and the start interval are over; the ban is not.
    rig.age_state(500);
    // 1. as it is; 2. with ANOTHER proxy; 3. direct; 4. a new nick; 5. removed and added again (reopened_at 0) - all closed.
    for (connection, why) in [
        ("proxy:hp-1", "unchanged"),
        ("proxy:hp-2", "another proxy"),
        ("direct", "direct"),
    ] {
        rig.favourites(&[(&addr, connection, 0), (&sibling, "direct", 0), (other, "direct", 0)]);
        assert!(rig.send(&start(&addr)).status.success());
        assert_eq!(reason(&rig.status()), "blocked_after_ban", "{why}: {}", rig.status());
        rig.age_state(500);
    }
    // The same IP, another port: closed too.
    assert!(rig.send(&start(&sibling)).status.success());
    assert_eq!(reason(&rig.status()), "blocked_after_ban", "the ban is the machine's");
    rig.age_state(500);
    // Editing live-servers.toml (even to a time after the ban) re-opens allow-list entries, not favourites.
    rig.allow_list("");
    fs::OpenOptions::new()
        .write(true)
        .open(rig.live_servers())
        .unwrap()
        .set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(600))
        .unwrap();
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(reason(&rig.status()), "blocked_after_ban");
    rig.age_state(500);
    // A re-open that is not newer than the ban does not lift it.
    rig.favourites(&[
        (&addr, "proxy:hp-2", ban_at),
        (&sibling, "direct", 0),
        (other, "direct", 0),
    ]);
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(reason(&rig.status()), "blocked_after_ban");
    rig.age_state(500);
    // Another IP was never affected.
    assert!(rig.send(&start(other)).status.success());
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    assert!(rig.exited("exited", "0").status.success());
    rig.age_state(500);
    assert_eq!(
        rig.actions()
            .iter()
            .filter(|a| a.starts_with("start ddnet-ai-bot"))
            .count(),
        2,
        "only the first start and the other server: {:?}",
        rig.actions()
    );
    assert!(rig.actions().len() > actions);

    // The owner presses «Открыть снова»: `reopened_at` after the ban. Only then it starts, with the proxy the owner has set now.
    rig.favourites(&[
        (&addr, "proxy:hp-2", ban_at + 1),
        (&sibling, "direct", 0),
        (other, "direct", 0),
    ]);
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
    let dropin = rig.dropin();
    assert!(
        dropin.contains(&format!("IPAddressAllow={PROXY_IP}\n")),
        "the proxy of the owner's choice: {dropin}"
    );
    // The ban stays in the memory (inert for the re-opened favourite): the sibling on the same IP needs its own re-opening.
    assert_eq!(rig.blocked_file()["blocked"][0]["address"], addr.as_str());
    assert!(rig.exited("exited", "0").status.success());
    rig.age_state(500);
    assert!(rig.send(&start(&sibling)).status.success());
    assert_eq!(reason(&rig.status()), "blocked_after_ban", "{}", rig.status());
    rig.favourites(&[
        (&addr, "proxy:hp-2", ban_at + 1),
        (&sibling, "direct", ban_at + 1),
        (other, "direct", 0),
    ]);
    assert!(rig.send(&start(&sibling)).status.success());
    assert_eq!(rig.status()["state"], "started", "{}", rig.status());
}

#[test]
fn a_cool_down_and_the_start_interval_apply_to_favourites_like_to_every_server() {
    let rig = Rig::new();
    let addr = format!("{FAV_IP}:8303");
    rig.favourites(&[(&addr, "direct", 0)]);
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(rig.status()["state"], "started");
    assert!(rig.exited("exited", "4").status.success());
    // Right after the exit: the cool-down (and the start interval), whichever server.
    assert!(rig.send(&start(&addr)).status.success());
    assert!(
        matches!(reason(&rig.status()), "cooldown" | "rate_limited" | "blocked_after_ban"),
        "{}",
        rig.status()
    );
    let n = rig.actions().len();
    assert!(rig.send(&start("local")).status.success());
    assert_eq!(reason(&rig.status()), "cooldown");
    assert_eq!(rig.actions().len(), n);
}

#[test]
fn the_helper_never_writes_into_the_launch_directory_but_its_own_request_and_the_blocked_list_is_for_the_web() {
    let rig = Rig::new();
    let addr = format!("{FAV_IP}:8303");
    rig.favourites(&[(&addr, "direct", 0)]);
    let before = fs::read(rig.data().join("launch/favourites.json")).unwrap();
    assert!(rig.send(&start(&addr)).status.success());
    assert!(rig.exited("exited", "3").status.success());
    assert_eq!(
        fs::read(rig.data().join("launch/favourites.json")).unwrap(),
        before,
        "the helper never edits the favourites"
    );
    assert_eq!(mode_of(&rig.p("status/blocked.json")), 0o644);
    assert!(!rig.data().join("launch/blocked.json").exists());
}

// ---------------------------------------------------------------------------------------------------------------------
// Review 5.12 round 1
// ---------------------------------------------------------------------------------------------------------------------

/// F1: a `reopened_at` far in the future would lift every past and future ban on that IP. The helper refuses the file.
#[test]
fn a_future_dated_reopening_lifts_nothing() {
    let rig = Rig::new();
    let addr = format!("{FAV_IP}:8303");
    rig.favourites(&[(&addr, "direct", 99_999_999_999)]);
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(reason(&rig.status()), "favourites_invalid");
    assert!(rig.actions().is_empty());
    // With a ban in the memory too.
    let rig = Rig::new();
    rig.favourites(&[(&addr, "direct", 0)]);
    assert!(rig.send(&start(&addr)).status.success());
    assert!(rig.exited("exited", "3").status.success());
    rig.age_state(500);
    rig.favourites(&[(&addr, "direct", 99_999_999_999)]);
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(reason(&rig.status()), "favourites_invalid");
    rig.favourites(&[(&addr, "direct", 0)]);
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(reason(&rig.status()), "blocked_after_ban");
}

/// F2: a host name in the allow-list can hide the address of a favourite: no favourite starts while one exists.
#[test]
fn a_host_name_in_the_allow_list_refuses_favourites() {
    let rig = Rig::new();
    let addr = format!("{FAV_IP}:8303");
    rig.allow_list(
        "[[server]]\naddress = \"one.one.one.one:8303\"\nnick = \"Muha\"\nready = true\nproxy = \"swarfey\"\n",
    );
    rig.favourites(&[(&addr, "direct", 0)]);
    assert!(rig.send(&start(&addr)).status.success());
    assert_eq!(reason(&rig.status()), "allowlist_not_literal");
    assert!(rig.actions().is_empty());
}

/// F3: editing live-servers.toml re-opens allow-list entries, never a ban recorded for a favourite (not even for a sibling port).
#[test]
fn a_live_servers_edit_does_not_lift_a_favourites_ban_for_an_allow_list_sibling() {
    let rig = Rig::new();
    let addr = format!("{FAV_IP}:8303");
    let sibling = format!("{FAV_IP}:8304");
    rig.favourites(&[(&addr, "direct", 0)]);
    assert!(rig.send(&start(&addr)).status.success());
    assert!(rig.exited("exited", "3").status.success());
    rig.age_state(500);
    rig.allow_list(&format!(
        "[[server]]\naddress = \"{sibling}\"\nnick = \"Muha\"\nready = true\n"
    ));
    fs::OpenOptions::new()
        .write(true)
        .open(rig.live_servers())
        .unwrap()
        .set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(600))
        .unwrap();
    assert!(rig.send(&start(&sibling)).status.success());
    assert_eq!(reason(&rig.status()), "blocked_after_ban");
}

/// F5: the test-only feature shows in `--version`, and as root such a build refuses to be the helper (uid 0 cannot be had here: the
/// version text is what the installers check).
#[test]
fn a_build_that_accepts_loopback_says_so_in_its_version() {
    let out = Command::new(env!("CARGO_BIN_EXE_ddnet-ai"))
        .arg("--version")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(text.contains("+loopback-favourites"), "{text}");
}
