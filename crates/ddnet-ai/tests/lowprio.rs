//! Task 4.14 (D-124): `tools/lowprio.sh <cmd...>`, the agents' recipe for running heavy work (cargo, the arena, training) at low CPU and I/O
//! priority among the agents' own work. The tests run the real script with real commands: the command's exit status, arguments, environment and
//! stdin arrive unchanged, the niceness is really raised, the command stays in the CALLER'S cgroup (a user-scope step once gave the "low
//! priority" command more CPU than a plain nice-15 process, review 4.14 F1), SCHED_IDLE is opt-in, and bad settings are refused.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/lowprio.sh")
}

fn lowprio() -> Command {
    Command::new(script())
}

/// Field 19 of /proc/<pid>/stat (the nice value) of the current process; the comm field may contain spaces, so count after the last `)`.
const NICE_SH: &str = r#"awk '{ s=$0; sub(/^.*\) /, "", s); split(s, f, " "); print f[17] }' /proc/self/stat"#;

fn own_nice() -> i32 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap();
    let after = &stat[stat.rfind(')').unwrap() + 2..];
    after.split(' ').nth(16).unwrap().parse().unwrap()
}

#[test]
fn without_a_command_it_prints_usage_and_exits_2() {
    let out = Command::new(script()).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("usage: tools/lowprio.sh <command>"));
    assert!(out.stdout.is_empty());
}

#[test]
fn the_exit_status_of_the_command_is_passed_through() {
    for code in [0, 1, 7, 42, 127] {
        let st = lowprio().args(["sh", "-c", &format!("exit {code}")]).status().unwrap();
        assert_eq!(st.code(), Some(code), "exit {code}");
    }
    // A command that cannot be found is the shell's 127, not a silent success.
    let st = lowprio().arg("/nonexistent/definitely-not-a-command").status().unwrap();
    assert_ne!(st.code(), Some(0));
}

#[test]
fn the_niceness_is_raised_by_15_and_children_inherit_it() {
    let base = own_nice();
    let want = (base + 15).min(19);
    let out = lowprio().args(["sh", "-c", NICE_SH]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), want.to_string());
    // A grandchild (what cargo -> rustc is) has it too.
    let out = lowprio()
        .args(["sh", "-c", "sh -c \"$1\"", "outer", NICE_SH])
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        want.to_string(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // The knob.
    let out = lowprio()
        .env("LOWPRIO_NICE", "3")
        .args(["sh", "-c", NICE_SH])
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        (base + 3).min(19).to_string()
    );
}

#[test]
fn arguments_environment_and_stdin_arrive_unchanged() {
    // Spaces, quotes, globs, an empty argument and a leading dash must not be re-split or re-interpreted.
    let out = lowprio()
        .args(["printf", "[%s]", "a b", "", "*", "it's \"x\"", "$HOME", "-n"])
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[a b][][*][it's \"x\"][$HOME][-n]"
    );
    let out = lowprio()
        .env("LOWPRIO_TEST_VAR", "kept value")
        .args(["sh", "-c", "printf %s \"$LOWPRIO_TEST_VAR\""])
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout), "kept value");
    let mut child = lowprio()
        .args(["cat"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"from stdin\n").unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout), "from stdin\n");
    // The working directory is the caller's.
    let out = lowprio().current_dir("/").arg("pwd").output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "/");
}

#[test]
fn the_dry_run_names_the_recipe_one_word_per_line_and_has_no_scope_step() {
    let out = Command::new(script())
        .env("LOWPRIO_DRYRUN", "1")
        .args(["cargo", "build", "--release", "a b"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let words: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_string)
        .collect();
    let n = words.iter().position(|w| w == "nice").expect("nice");
    assert_eq!(&words[n..n + 3], ["nice", "-n", "15"], "{words:?}");
    assert_eq!(
        &words[words.len() - 4..],
        ["cargo", "build", "--release", "a b"],
        "{words:?}"
    );
    // Review 4.14 F1: never a systemd scope (it moves the command out of the caller's cgroup and raises its share).
    for w in &words {
        assert!(
            !w.contains("systemd-run") && !w.contains("--scope") && !w.contains("CPUWeight"),
            "{words:?}"
        );
    }
    // SCHED_IDLE is opt-in.
    assert!(!words.contains(&"chrt".to_string()), "{words:?}");
    let out = Command::new(script())
        .env("LOWPRIO_DRYRUN", "1")
        .env("LOWPRIO_IDLE", "1")
        .arg("true")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    if text.contains("chrt") {
        assert!(text.starts_with("chrt\n--idle\n0\n"), "{text}");
    }
}

#[test]
fn bad_settings_are_refused_with_status_2_and_the_command_does_not_run() {
    for (key, bad) in [("LOWPRIO_NICE", "20"), ("LOWPRIO_NICE", "-5"), ("LOWPRIO_NICE", "x")] {
        let marker = std::env::temp_dir().join(format!("lowprio-test-{}-{key}-{bad}", std::process::id()));
        let _ = std::fs::remove_file(&marker);
        let out = lowprio()
            .env(key, bad)
            .args(["touch", marker.to_str().unwrap()])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "{key}={bad}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(key),
            "{key}={bad}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!marker.exists(), "{key}={bad}: the command must not run");
    }
}

#[test]
fn the_command_stays_in_the_callers_cgroup() {
    // Review 4.14 F1: a transient scope would move it to another cgroup, where it competes as an equal with the caller's whole session.
    let mine = std::fs::read_to_string("/proc/self/cgroup").unwrap();
    let out = lowprio().args(["cat", "/proc/self/cgroup"]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout), mine);
}

#[test]
fn sched_idle_is_applied_only_when_asked_for() {
    let probe = Command::new("chrt").args(["--idle", "0", "true"]).status();
    if !probe.is_ok_and(|s| s.success()) {
        eprintln!("SKIPPED: chrt --idle is not available here");
        return;
    }
    let policy = "awk '{ s=$0; sub(/^.*\\) /, \"\", s); split(s, f, \" \"); print f[39] }' /proc/self/stat";
    let plain = lowprio().args(["sh", "-c", policy]).output().unwrap();
    assert_eq!(
        String::from_utf8_lossy(&plain.stdout).trim(),
        "0",
        "SCHED_OTHER by default"
    );
    let idle = lowprio()
        .env("LOWPRIO_IDLE", "1")
        .args(["sh", "-c", policy])
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&idle.stdout).trim(),
        "5",
        "SCHED_IDLE with LOWPRIO_IDLE=1"
    );
    let st = lowprio()
        .env("LOWPRIO_IDLE", "1")
        .args(["sh", "-c", "exit 9"])
        .status()
        .unwrap();
    assert_eq!(st.code(), Some(9));
}
