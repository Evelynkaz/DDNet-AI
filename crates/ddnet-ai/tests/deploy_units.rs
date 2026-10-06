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
