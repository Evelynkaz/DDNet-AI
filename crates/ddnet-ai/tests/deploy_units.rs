//! Task 5.12 (D-099): the deploy files hold the safety rules of the server browser. The web unit gains **no network right** (it stays
//! loopback-only with `IPAddressDeny=any`), no new writable path and no capability; the one unit that fetches the master list is the
//! small sandboxed one; the proxy check is unprivileged; no production unit enables the test-only loopback favourites; and the installer
//! installs exactly the new units.

use std::fs;
use std::path::PathBuf;

fn deploy() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../deploy")
}

fn unit(name: &str) -> String {
    fs::read_to_string(deploy().join("systemd").join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// The unit's active settings: comments and blank lines dropped, `Key=value` pairs in order.
fn settings(text: &str) -> Vec<(String, String)> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('['))
        .filter_map(|l| {
            l.split_once('=')
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        })
        .collect()
}

fn values<'a>(s: &'a [(String, String)], key: &str) -> Vec<&'a str> {
    s.iter().filter(|(k, _)| k == key).map(|(_, v)| v.as_str()).collect()
}

#[test]
fn the_web_unit_gains_no_network_right() {
    let text = unit("ddnet-ai-web.service");
    let s = settings(&text);
    // Loopback only, everything else denied by the kernel-level cgroup filter.
    assert_eq!(values(&s, "IPAddressAllow"), vec!["127.0.0.0/8 ::1"]);
    assert_eq!(values(&s, "IPAddressDeny"), vec!["any"]);
    assert_eq!(values(&s, "RestrictAddressFamilies"), vec!["AF_INET AF_INET6 AF_UNIX"]);
    assert_eq!(values(&s, "CapabilityBoundingSet"), vec![""]);
    assert_eq!(values(&s, "AmbientCapabilities"), vec![""]);
    assert_eq!(values(&s, "NoNewPrivileges"), vec!["true"]);
    assert_eq!(values(&s, "ProtectSystem"), vec!["strict"]);
    assert_eq!(values(&s, "ProtectHome"), vec!["read-only"]);
    assert_eq!(values(&s, "User"), vec!["ubuntu"]);
    // The writable paths are the five it had: secrets (profiles), logs, bot, launch (favourites, requests). Not servers/: the cache is read-only for it.
    let rw: Vec<&str> = values(&s, "ReadWritePaths")
        .iter()
        .flat_map(|v| v.split_whitespace())
        .collect();
    assert_eq!(
        rw,
        vec![
            "/home/ubuntu/aiddnet/data/secrets",
            "/home/ubuntu/aiddnet/data/logs",
            "/home/ubuntu/aiddnet/data/bot",
            "/home/ubuntu/aiddnet/data/launch"
        ]
    );
    // It still listens on loopback only and never runs a fetch itself: no ExecStartPre/ExecStopPost, no extra command.
    assert!(text.contains("--listen 127.0.0.1:7788"));
    for key in [
        "ExecStartPre",
        "ExecStartPost",
        "ExecStop",
        "ExecStopPost",
        "ExecReload",
        "IPAddressAllow",
    ] {
        let n = values(&s, key).len();
        assert!(n <= 1, "{key}");
    }
    assert!(
        !text.contains("master1") && !text.contains("servers-cache"),
        "the web unit must not fetch the master list"
    );
}

#[test]
fn the_fetch_and_the_proxy_check_are_small_sandboxed_units_for_the_owners_user() {
    for (name, ro_secrets) in [
        ("ddnet-ai-servers.service", false),
        ("ddnet-ai-proxycheck.service", true),
    ] {
        let s = settings(&unit(name));
        assert_eq!(values(&s, "Type"), vec!["oneshot"], "{name}");
        assert_eq!(values(&s, "User"), vec!["ubuntu"], "{name}: not root");
        assert_eq!(values(&s, "CapabilityBoundingSet"), vec![""], "{name}");
        assert_eq!(values(&s, "NoNewPrivileges"), vec!["true"], "{name}");
        assert_eq!(values(&s, "ProtectSystem"), vec!["strict"], "{name}");
        // The home directory is an empty tmpfs with only what the unit needs mounted into it.
        assert_eq!(values(&s, "ProtectHome"), vec!["tmpfs"], "{name}");
        assert_eq!(
            values(&s, "RestrictAddressFamilies"),
            vec!["AF_INET AF_INET6 AF_UNIX"],
            "{name}"
        );
        assert_eq!(values(&s, "MemoryDenyWriteExecute"), vec!["true"], "{name}");
        // The machine's own networks stay out of reach: a deny list only (an allow match would win over a deny match).
        // The check unit additionally denies loopback and allows only the DNS stub (an allow match wins over a deny match).
        if name == "ddnet-ai-proxycheck.service" {
            assert_eq!(values(&s, "IPAddressAllow"), vec!["127.0.0.53/32"], "{name}");
            let deny = values(&s, "IPAddressDeny").join(" ");
            assert!(
                deny.contains("127.0.0.0/8") && deny.contains("::1"),
                "{name}: loopback denied"
            );
        } else {
            assert!(values(&s, "IPAddressAllow").is_empty(), "{name}");
        }
        let deny = values(&s, "IPAddressDeny").join(" ");
        for range in [
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "169.254.0.0/16",
            "100.64.0.0/10",
            "fc00::/7",
            "fe80::/10",
        ] {
            assert!(deny.contains(range), "{name}: {range}");
        }
        // Only the root-owned copy of the binary runs.
        assert!(
            values(&s, "ExecStart")[0].starts_with("/usr/local/libexec/ddnet-ai/ddnet-ai "),
            "{name}"
        );
        let rw = values(&s, "BindPaths");
        let ro = values(&s, "BindReadOnlyPaths");
        if ro_secrets {
            assert_eq!(rw, vec!["/home/ubuntu/aiddnet/data/launch"], "{name}");
            assert_eq!(ro, vec!["/home/ubuntu/aiddnet/data/secrets"], "{name}");
        } else {
            assert_eq!(rw, vec!["/home/ubuntu/aiddnet/data/servers"], "{name}");
            assert!(ro.is_empty(), "{name}: no secrets at all");
        }
    }
    // The fetch is the only place that names the masters, and it is the cache command.
    assert!(unit("ddnet-ai-servers.service").contains("servers-cache"));
}

#[test]
fn the_path_units_watch_the_files_the_web_writes_and_are_rate_limited() {
    let servers = settings(&unit("ddnet-ai-servers.path"));
    assert_eq!(
        values(&servers, "PathChanged"),
        vec!["/home/ubuntu/aiddnet/data/launch/servers-refresh"]
    );
    assert_eq!(values(&servers, "Unit"), vec!["ddnet-ai-servers.service"]);
    let check = settings(&unit("ddnet-ai-proxycheck.path"));
    assert_eq!(
        values(&check, "PathExists"),
        vec!["/home/ubuntu/aiddnet/data/launch/proxycheck-request.json"]
    );
    assert_eq!(values(&check, "Unit"), vec!["ddnet-ai-proxycheck.service"]);
    for s in [&servers, &check] {
        assert!(!values(s, "TriggerLimitIntervalSec").is_empty() && !values(s, "TriggerLimitBurst").is_empty());
    }
}

#[test]
fn no_unit_or_script_enables_the_test_only_loopback_favourites_and_the_installer_installs_the_new_units() {
    // The cargo feature is compile-time only; nothing in deploy/ may mention it (no flag, no environment variable).
    fn walk(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
        for e in fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, out);
            } else {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(&deploy(), &mut files);
    for f in &files {
        let Ok(text) = fs::read_to_string(f) else { continue };
        // The README and the two installers name the feature only to refuse a test build (checked below).
        if f.file_name()
            .is_some_and(|n| n == "README.md" || n == "install.sh" || n == "install-launcher.sh")
        {
            continue;
        }
        assert!(
            !text.contains("loopback-favourites") && !text.contains("LOOPBACK_FAVOURITES"),
            "{} mentions the test-only feature",
            f.display()
        );
    }
    let script = fs::read_to_string(deploy().join("install-launcher.sh")).unwrap();
    for u in [
        "ddnet-ai-servers.service",
        "ddnet-ai-servers.path",
        "ddnet-ai-proxycheck.service",
        "ddnet-ai-proxycheck.path",
    ] {
        assert!(script.contains(u), "install-launcher.sh does not install {u}");
        assert!(deploy().join("systemd").join(u).is_file(), "{u}");
    }
    // Both installers refuse a test build (it says so in `--version`).
    let install = fs::read_to_string(deploy().join("install.sh")).unwrap();
    assert!(script.contains("grep -q 'loopback-favourites'") && install.contains("grep -q 'loopback-favourites'"));
    // The installer enables the two new path units, never the fetch or the check service directly.
    assert!(script.contains("enable ddnet-ai-launch.path ddnet-ai-servers.path ddnet-ai-proxycheck.path"));
}

#[test]
fn the_bot_unit_is_unchanged_in_what_matters_no_extra_network_and_no_new_writable_path() {
    let s = settings(&unit("ddnet-ai-bot.service"));
    assert_eq!(values(&s, "IPAddressAllow"), vec!["127.0.0.0/8 ::1"]);
    assert_eq!(values(&s, "IPAddressDeny"), vec!["any"]);
    let rw: Vec<&str> = values(&s, "ReadWritePaths")
        .iter()
        .flat_map(|v| v.split_whitespace())
        .collect();
    assert!(
        rw.iter()
            .all(|p| !p.ends_with("/launch") && !p.ends_with("/secrets") && !p.ends_with("/servers")),
        "{rw:?}"
    );
}

#[test]
fn the_bot_unit_takes_the_finishing_mode_from_the_environment_and_is_off_without_the_helper() {
    let s = settings(&unit("ddnet-ai-bot.service"));
    // The default (a hand start, an env file from before task 5.13) is `off`; the helper's `BOT_FINISH` overrides it.
    assert!(
        values(&s, "Environment").contains(&"BOT_FINISH=off"),
        "{:?}",
        values(&s, "Environment")
    );
    let exec = values(&s, "ExecStart");
    assert_eq!(exec.len(), 1);
    assert!(exec[0].contains(" --finish ${BOT_FINISH} "), "{}", exec[0]);
    // Every `${BOT_*}` the command line expands has a default in the unit, so a missing value never leaves an empty argument.
    for var in exec[0]
        .split("${")
        .skip(1)
        .filter_map(|rest| rest.split_once('}').map(|(name, _)| name))
    {
        assert!(
            values(&s, "Environment")
                .iter()
                .any(|e| e.split_once('=').is_some_and(|(k, _)| k == var)),
            "{var} has no Environment= default"
        );
    }
}

#[test]
fn the_bot_unit_passes_the_smart_wayblock_and_the_duel_switch_as_single_arguments_and_both_are_off_by_default() {
    // Task 5.15 (D-103/D-104, D-102).
    let s = settings(&unit("ddnet-ai-bot.service"));
    let env = values(&s, "Environment");
    assert!(env.contains(&"BOT_WB_SMART=off"), "{env:?}");
    assert!(env.contains(&"BOT_NO_SELFKILL=false"), "{env:?}");
    let exec = values(&s, "ExecStart");
    assert_eq!(exec.len(), 1);
    // The smart wayblock takes its value as a separate word (`--wb-smart on|off`); the duel switch is a boolean flag, which cannot be given
    // an empty argument, so the unit passes the one-argument form `--no-selfkill=true|false`. systemd puts `${VAR}` into exactly one argument.
    assert!(exec[0].contains(" --wb-smart ${BOT_WB_SMART} "), "{}", exec[0]);
    assert!(exec[0].contains(" --no-selfkill=${BOT_NO_SELFKILL} "), "{}", exec[0]);
    // Never the bare flag (it would be always on), and never the old words unquoted into the line by `$VAR` splitting.
    assert!(!exec[0].contains(" --no-selfkill "), "{}", exec[0]);
    assert!(
        !exec[0].contains("$BOT_WB_SMART") && !exec[0].contains("$BOT_NO_SELFKILL"),
        "{}",
        exec[0]
    );
}

#[test]
fn the_launcher_installer_refuses_a_binary_older_than_the_unit() {
    // Task 5.15: the new bot unit passes `--no-selfkill=${BOT_NO_SELFKILL}`, which an older binary rejects, so the installer checks the
    // binary's help for the `--no-selfkill[=` form before it touches anything.
    let script = fs::read_to_string(deploy().join("install-launcher.sh")).unwrap();
    let check = script
        .find("grep -q -- '--no-selfkill\\[='")
        .expect("install-launcher.sh checks the binary for the --no-selfkill= form");
    assert!(
        script[check..]
            .lines()
            .next()
            .unwrap()
            .contains("run deploy/install.sh first")
    );
    assert!(script.contains("\"$BIN_SRC\" play --help"));
    // It is in the preconditions: before the unit files are installed and before anything is stopped or reloaded.
    for later in ["daemon-reload", "install -o root -g root -m 0644"] {
        let first_use = script.rfind(later).unwrap();
        assert!(check < first_use, "the binary check must come before `{later}`");
    }
    let unit_src = fs::read_to_string(deploy().join("systemd").join("ddnet-ai-bot.service")).unwrap();
    assert!(
        unit_src.contains("--no-selfkill=${BOT_NO_SELFKILL}"),
        "the check is for this very form"
    );
}

#[test]
fn the_bot_unit_passes_the_opponent_predictor_as_one_argument_and_it_is_off_by_default() {
    // Task 3.17 (D-111): the path of the model file, empty = off. `${VAR}` is one argument, so an empty value is `--window-model=`, which the bot reads as off.
    let s = settings(&unit("ddnet-ai-bot.service"));
    let env = values(&s, "Environment");
    assert!(env.contains(&"BOT_WINDOW_MODEL="), "{env:?}");
    let exec = values(&s, "ExecStart");
    assert_eq!(exec.len(), 1);
    assert!(exec[0].contains(" --window-model=${BOT_WINDOW_MODEL} "), "{}", exec[0]);
    // Never split by `$VAR` (a path with a space would become two words), never the bare flag, never with a space before the value.
    assert!(
        !exec[0].contains("$BOT_WINDOW_MODEL")
            && !exec[0].contains(" --window-model ")
            && !exec[0].contains("--window-model ${"),
        "{}",
        exec[0]
    );
    // The earlier switches are untouched.
    assert!(exec[0].contains(" --no-selfkill=${BOT_NO_SELFKILL} "), "{}", exec[0]);
}

#[test]
fn the_launcher_installer_refuses_a_binary_without_the_window_model_flag() {
    let script = fs::read_to_string(deploy().join("install-launcher.sh")).unwrap();
    let check = script
        .find("grep -q -- '--window-model' <<<\"$play_help\"")
        .expect("install-launcher.sh checks the binary for --window-model");
    assert!(
        script[check..]
            .lines()
            .next()
            .unwrap()
            .contains("run deploy/install.sh first")
    );
    for later in ["daemon-reload", "install -o root -g root -m 0644"] {
        let first_use = script.rfind(later).unwrap();
        assert!(check < first_use, "the binary check must come before `{later}`");
    }
}

#[test]
fn the_bot_unit_passes_the_preinput_switch_as_two_words_and_it_is_off_by_default() {
    // Task 3.20b (D-112): `--preinput ${BOT_PREINPUT}` (on|off), the form of `--wb-smart`; the helper always writes the line.
    let s = settings(&unit("ddnet-ai-bot.service"));
    let env = values(&s, "Environment");
    assert!(env.contains(&"BOT_PREINPUT=off"), "{env:?}");
    let exec = values(&s, "ExecStart");
    assert_eq!(exec.len(), 1);
    assert!(exec[0].contains(" --preinput ${BOT_PREINPUT} "), "{}", exec[0]);
    // Never split by `$VAR`, never the equals form (the bot's flag takes a separate value), never hard-coded on.
    assert!(
        !exec[0].contains("$BOT_PREINPUT") && !exec[0].contains("--preinput=") && !exec[0].contains("--preinput on"),
        "{}",
        exec[0]
    );
    // The earlier switches are untouched, and the finishing word stays one `${BOT_FINISH}` (it takes `wb` without a unit change).
    assert!(exec[0].contains(" --finish ${BOT_FINISH} "), "{}", exec[0]);
    assert!(exec[0].contains(" --window-model=${BOT_WINDOW_MODEL} "), "{}", exec[0]);
}

#[test]
fn the_launcher_installer_refuses_a_binary_without_the_preinput_flag() {
    let script = fs::read_to_string(deploy().join("install-launcher.sh")).unwrap();
    let check = script
        .find("grep -q -- '--preinput' <<<\"$play_help\"")
        .expect("install-launcher.sh checks the binary for --preinput");
    assert!(
        script[check..]
            .lines()
            .next()
            .unwrap()
            .contains("run deploy/install.sh first")
    );
    for later in ["daemon-reload", "install -o root -g root -m 0644"] {
        let first_use = script.rfind(later).unwrap();
        assert!(check < first_use, "the binary check must come before `{later}`");
    }
}

#[test]
fn the_bot_unit_has_cpu_priority_over_the_agents_with_soft_settings_only_and_keeps_its_hardening() {
    // Task 4.14 (D-124): builds, tests, arenas and training on the same VPS starved the bot's search (14 candidates per decision against 28).
    let s = settings(&unit("ddnet-ai-bot.service"));
    // CPUWeight= (systemd.resource-control: 1..10000, default 100) well above the default; exactly one value, no drop-in-style reset.
    let weight = values(&s, "CPUWeight");
    assert_eq!(weight.len(), 1, "{weight:?}");
    let w: u32 = weight[0].parse().expect("CPUWeight is a plain number");
    assert!((500..=10000).contains(&w), "CPUWeight={w}");
    // Nice= (systemd.exec: -20..19): negative, but not the extreme that would starve the game server (it shares system.slice with other units, not the bot's slice).
    let nice = values(&s, "Nice");
    assert_eq!(nice.len(), 1, "{nice:?}");
    let n: i32 = nice[0].parse().expect("Nice is a plain number");
    assert!((-10..=-1).contains(&n), "Nice={n}");
    let io = values(&s, "IOWeight");
    assert_eq!(io.len(), 1, "{io:?}");
    assert!(io[0].parse::<u32>().is_ok_and(|v| (100..=10000).contains(&v)), "{io:?}");
    // The setting that counts: a top-level slice of its own, whose weight competes with user.slice (the agents) directly.
    let slice = values(&s, "Slice");
    assert_eq!(slice, vec!["ddnetaibot.slice"]);
    let sl = settings(&unit("ddnetaibot.slice"));
    let sw: u32 = values(&sl, "CPUWeight")[0]
        .parse()
        .expect("the slice's CPUWeight is a plain number");
    assert!((500..=10000).contains(&sw), "slice CPUWeight={sw}");
    assert!(
        values(&sl, "IOWeight")[0]
            .parse::<u32>()
            .is_ok_and(|v| (100..=10000).contains(&v))
    );
    // No dash in the slice name: systemd would read `a-b.slice` as a child of `a.slice` (default weight 100), and the weight that competes
    // at the top would be that parent's, not ours.
    assert!(!slice[0].trim_end_matches(".slice").contains('-'), "{slice:?}");
    // The slice sets only weights: no limits, no quota, no real-time, no affinity, and nothing but the [Unit] and [Slice] sections.
    let keys: Vec<&str> = sl.iter().map(|(k, _)| k.as_str()).collect();
    assert!(
        keys.iter()
            .all(|k| ["Description", "Documentation", "CPUWeight", "IOWeight"].contains(k)),
        "{keys:?}"
    );
    // No real-time scheduling, ever: the policy is not set and the sandbox still forbids it.
    for key in ["CPUSchedulingPolicy", "CPUSchedulingPriority", "CPUAffinity"] {
        assert!(values(&s, key).is_empty(), "{key} must not be set");
    }
    assert_eq!(values(&s, "RestrictRealtime"), vec!["true"]);
    // Nice=-5 must not need a capability in the running unit: the bounding set stays empty (systemd applies Nice= as PID 1), and no new privileges.
    assert_eq!(values(&s, "CapabilityBoundingSet"), vec![""]);
    assert_eq!(values(&s, "AmbientCapabilities"), vec![""]);
    assert_eq!(values(&s, "NoNewPrivileges"), vec!["true"]);
    // The existing hardening is all still there.
    for (key, want) in [
        ("ProtectSystem", "strict"),
        ("ProtectHome", "read-only"),
        ("PrivateTmp", "true"),
        ("PrivateDevices", "true"),
        ("ProtectKernelTunables", "true"),
        ("ProtectKernelModules", "true"),
        ("ProtectKernelLogs", "true"),
        ("ProtectControlGroups", "true"),
        ("ProtectClock", "true"),
        ("ProtectHostname", "true"),
        ("RestrictNamespaces", "true"),
        ("RestrictSUIDSGID", "true"),
        ("LockPersonality", "true"),
        ("MemoryDenyWriteExecute", "true"),
        ("RemoveIPC", "true"),
        ("RestrictAddressFamilies", "AF_INET AF_INET6 AF_UNIX"),
        ("SystemCallArchitectures", "native"),
        ("MemoryMax", "2G"),
        ("TasksMax", "256"),
        ("UMask", "0077"),
        ("IPAddressAllow", "127.0.0.0/8 ::1"),
        ("IPAddressDeny", "any"),
    ] {
        assert_eq!(values(&s, key), vec![want], "{key}");
    }
}

#[test]
fn the_launcher_installer_installs_the_bot_slice_and_checks_that_the_bot_unit_really_has_the_cpu_priority() {
    // Task 4.14: the slice file is installed with the units (and removed by --uninstall); after `daemon-reload` the installer asks systemd what
    // it now loads (a foreign drop-in or a stale copy would show here) and compares it with the repository's files; it only warns (the install
    // itself is done, and it never starts or stops the bot).
    let script = fs::read_to_string(deploy().join("install-launcher.sh")).unwrap();
    let units_line = script.find("UNITS=(").expect("UNITS array");
    assert!(
        script[units_line..]
            .lines()
            .next()
            .unwrap()
            .contains("ddnetaibot.slice")
    );
    let uninstall = script.find("if [[ \"$UNINSTALL\" -eq 1 ]]").expect("uninstall branch");
    assert!(
        script[uninstall..].contains("ddnet-ai-proxycheck.service ddnetaibot.slice; do"),
        "--uninstall removes the slice file"
    );
    let reload = script.rfind("sudo systemctl daemon-reload").expect("daemon-reload");
    let readback = script
        .find("for key in Slice CPUWeight Nice IOWeight; do check_prop \"$BOT_UNIT\" \"$key\"; done")
        .expect("the read-back of the bot unit");
    assert!(reload < readback, "the read-back comes after the reload");
    assert!(script.contains("for key in CPUWeight IOWeight; do check_prop ddnetaibot.slice \"$key\"; done"));
    let f = script.find("check_prop() {").expect("check_prop");
    let body = &script[f..];
    let body = &body[..body.find("\n}\n").unwrap()];
    assert!(body.contains("systemctl show -p \"$key\" --value \"$unit\""), "{body}");
    assert!(
        body.contains("\"$UNIT_SRC/$unit\""),
        "the wanted value is read from the repository's file: {body}"
    );
    assert!(body.contains("WARNING"), "{body}");
    assert!(!body.contains("die "), "a mismatch warns, it does not abort: {body}");
    // Every key it reads back is present in the file it reads it from.
    let s = settings(&unit("ddnet-ai-bot.service"));
    for key in ["Slice", "CPUWeight", "Nice", "IOWeight"] {
        assert_eq!(values(&s, key).len(), 1, "{key}");
    }
    let sl = settings(&unit("ddnetaibot.slice"));
    for key in ["CPUWeight", "IOWeight"] {
        assert_eq!(values(&sl, key).len(), 1, "{key}");
    }
}

#[test]
fn the_bot_unit_passes_the_search_threads_as_two_words_and_it_is_one_by_default() {
    // Task 5.17 (D-125): `--search-threads ${BOT_SEARCH_THREADS}` (a digit 1 to 4), the form of `--preinput`; the helper always writes the line.
    let s = settings(&unit("ddnet-ai-bot.service"));
    let env = values(&s, "Environment");
    assert!(env.contains(&"BOT_SEARCH_THREADS=1"), "{env:?}");
    let exec = values(&s, "ExecStart");
    assert_eq!(exec.len(), 1);
    assert!(
        exec[0].contains(" --search-threads ${BOT_SEARCH_THREADS} "),
        "{}",
        exec[0]
    );
    // Once; never split by `$VAR`, never the equals form, never hard-coded, never `auto` (the free cores are not the owner's choice here).
    assert_eq!(exec[0].matches("--search-threads").count(), 1, "{}", exec[0]);
    assert!(
        !exec[0].contains("$BOT_SEARCH_THREADS")
            && !exec[0].contains("--search-threads=")
            && !exec[0].contains("--search-threads auto")
            && !exec[0].contains("--search-threads 1")
            && !exec[0].contains("--search-threads 3"),
        "{}",
        exec[0]
    );
    // The earlier switches are untouched.
    assert!(exec[0].contains(" --preinput ${BOT_PREINPUT} "), "{}", exec[0]);
    assert!(exec[0].contains(" --finish ${BOT_FINISH} "), "{}", exec[0]);
}

#[test]
fn the_launcher_installer_refuses_a_binary_without_the_search_threads_flag() {
    let script = fs::read_to_string(deploy().join("install-launcher.sh")).unwrap();
    let check = script
        .find("grep -q -- '--search-threads' <<<\"$play_help\"")
        .expect("install-launcher.sh checks the binary for --search-threads");
    assert!(
        script[check..]
            .lines()
            .next()
            .unwrap()
            .contains("run deploy/install.sh first")
    );
    for later in ["daemon-reload", "install -o root -g root -m 0644"] {
        let first_use = script.rfind(later).unwrap();
        assert!(check < first_use, "the binary check must come before `{later}`");
    }
}

#[test]
fn the_bot_unit_passes_the_duel_fixes_as_two_words_and_they_are_off_by_default() {
    // Task 5.18 (D-129): `--duel-fixes ${BOT_DUEL_FIXES}` (`off`, `finish` or `static,finish`), the form of `--preinput`; the helper always writes the line.
    let s = settings(&unit("ddnet-ai-bot.service"));
    let env = values(&s, "Environment");
    assert!(env.contains(&"BOT_DUEL_FIXES=off"), "{env:?}");
    let exec = values(&s, "ExecStart");
    assert_eq!(exec.len(), 1);
    assert!(exec[0].contains(" --duel-fixes ${BOT_DUEL_FIXES} "), "{}", exec[0]);
    // Once; never split by `$VAR`, never the equals form, never hard-coded (the list holds no `counter` and no `all`).
    assert_eq!(exec[0].matches("--duel-fixes").count(), 1, "{}", exec[0]);
    assert!(
        !exec[0].contains("$BOT_DUEL_FIXES")
            && !exec[0].contains("--duel-fixes=")
            && !exec[0].contains("--duel-fixes off")
            && !exec[0].contains("--duel-fixes finish")
            && !exec[0].contains("--duel-fixes static")
            && !exec[0].contains("--duel-fixes all")
            && !exec[0].contains("counter"),
        "{}",
        exec[0]
    );
    // The earlier switches are untouched.
    assert!(
        exec[0].contains(" --search-threads ${BOT_SEARCH_THREADS} "),
        "{}",
        exec[0]
    );
    assert!(exec[0].contains(" --preinput ${BOT_PREINPUT} "), "{}", exec[0]);
    assert!(exec[0].contains(" --finish ${BOT_FINISH} "), "{}", exec[0]);
}

#[test]
fn the_launcher_installer_refuses_a_binary_without_the_duel_fixes_flag() {
    let script = fs::read_to_string(deploy().join("install-launcher.sh")).unwrap();
    let check = script
        .find("grep -q -- '--duel-fixes' <<<\"$play_help\"")
        .expect("install-launcher.sh checks the binary for --duel-fixes");
    assert!(
        script[check..]
            .lines()
            .next()
            .unwrap()
            .contains("run deploy/install.sh first")
    );
    for later in ["daemon-reload", "install -o root -g root -m 0644"] {
        let first_use = script.rfind(later).unwrap();
        assert!(check < first_use, "the binary check must come before `{later}`");
    }
}
