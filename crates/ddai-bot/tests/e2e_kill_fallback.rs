//! End-to-end of the `/kill` fallback (task 4.6, D-078) against the real local DDNet 20.1 server (`ddnet-local.service`,
//! 127.0.0.1:8303). `#[ignore]`d; run with
//!
//! ```text
//! DDAI_E2E=1 cargo test -p ddai-bot --test e2e_kill_fallback -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Loopback only; the server must be empty and on `Copy Love Box`. The test sets `sv_kill_protection 1` (one minute) through
//! econ and **restores `20` afterwards, reading it back** (also when the test fails). One bot of ours plays alone:
//!
//! 1. after a life of more than a minute the console `!kill` sends the protocol `Cl_Kill`, which the server drops; the bot then
//!    sends `/kill` (exactly once for this decision) and dies;
//! 2. the next life is new: a console `!kill` a few seconds in works through the protocol alone, no `/kill`;
//! 3. after another unbroken minute the second dropped `Cl_Kill` gets its one `/kill` again, at least 500 ticks after the first.
//!
//! The outgoing audit then shows exactly two `Cl_Say(/kill)` (accepted, none refused), three `Cl_Kill`, every `/kill` preceded by
//! a decision of the bot's own, and no other chat label.

use std::io::Write as _;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ddai_bot::brains::{BrainKind, BrainOptions};
use ddai_bot::clipper::ClipConfig;
use ddai_bot::command::CommandBus;
use ddai_bot::console;
use ddai_bot::nav_hooks::{NavConfig, NavHandle, WbMode};
use ddai_bot::runner::{RunnerConfig, run};
use ddai_bot::{BotConfig, Mode, Relations};
use ddai_client::ClientConfig;
use ddai_client::session::SERVER_COMMAND_KILL_LABEL;

fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("aiddnet/data")
}

fn econ(args: &[&str]) -> String {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/ddnet-server/econ.py");
    for attempt in 0..5 {
        let out = Command::new("python3")
            .arg(&script)
            .args(args)
            .output()
            .expect("python3 econ.py");
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        if out.status.success() && !text.is_empty() {
            return text;
        }
        std::thread::sleep(Duration::from_secs(2 + attempt));
    }
    panic!("econ {args:?} failed");
}

fn value_of(text: &str) -> String {
    text.lines()
        .find_map(|l| l.split("Value: ").nth(1))
        .map(|v| v.trim().to_string())
        .unwrap_or_default()
}

/// Puts `sv_kill_protection` back to 20 (and checks `sv_map`) when the test ends, however it ends.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        econ(&["sv_kill_protection", "20"]);
        std::thread::sleep(Duration::from_secs(3));
        let kp = value_of(&econ(&["sv_kill_protection"]));
        std::thread::sleep(Duration::from_secs(3));
        let map = value_of(&econ(&["sv_map"]));
        eprintln!("[restore] sv_kill_protection = {kp}, sv_map = {map}");
        if !std::thread::panicking() {
            assert_eq!(kp, "20");
            assert_eq!(map, "Copy Love Box");
        }
    }
}

/// The scratch directory goes away when the test ends, however it ends.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Default, Clone)]
struct Said(Arc<Mutex<Vec<String>>>);

impl Said {
    fn printer(&self) -> console::Printer {
        let me = self.clone();
        Arc::new(move |text: &str| {
            for l in text.lines() {
                eprintln!("[console] {l}");
            }
            me.0.lock().unwrap().push(text.to_string());
        })
    }

    fn all(&self) -> String {
        self.0.lock().unwrap().join("\n")
    }

    /// The `deaths N` of the latest `!stats` line.
    fn deaths(&self) -> Option<u32> {
        let all = self.all();
        let at = all.rfind("deaths ")?;
        all[at + 7..].split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
    }
}

#[test]
#[ignore = "runs against the real local ddnet-local.service; DDAI_E2E=1 and --ignored"]
fn a_dropped_cl_kill_is_followed_by_exactly_one_slash_kill_and_the_bot_dies() {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set DDAI_E2E=1");
        return;
    }
    assert_eq!(
        value_of(&econ(&["sv_map"])),
        "Copy Love Box",
        "the server must be on Copy Love Box"
    );
    assert!(
        !econ(&["status"]).contains("name='"),
        "someone is on the local server: it is ours alone for this test"
    );
    let _restore = Restore;
    econ(&["sv_kill_protection", "1"]);

    let said = Said::default();
    let (sender, inbox) = CommandBus::open();
    let (pipe_in, mut pipe_out) = std::io::pipe().expect("a pipe");
    let _console = console::spawn(sender, std::io::BufReader::new(pipe_in), said.printer()).expect("console thread");
    let out = std::env::temp_dir().join(format!("ddai-e2e-killfb-{}", std::process::id()));
    std::fs::create_dir_all(&out).expect("a scratch directory");
    let _scratch = Scratch(out);

    let cfg = RunnerConfig {
        server: "127.0.0.1:8303".parse::<SocketAddr>().unwrap(),
        client: ClientConfig {
            name: "ddai-e2e-kfb".to_string(),
            cache_dir: data_dir().join("maps").join("cache"),
            adaptive_margin: true,
            ..ClientConfig::default()
        },
        bot: BotConfig {
            brain: BrainKind::Hybrid,
            mode: Mode::Fight,
            seed: 46,
            clips: ClipConfig {
                dir: None,
                autoclip: false,
                async_save: false,
            },
            console_names: true,
            ..BotConfig::default()
        },
        brain: BrainOptions {
            seed: 46,
            ..BrainOptions::default()
        },
        relations: Relations::new(),
        duration: Some(Duration::from_secs(330)),
        bridge_path: None,
        web_names: false,
        debug_names_log: None,
        audit_outgoing: true,
        shutdown: Arc::new(AtomicBool::new(false)),
        nav: NavConfig {
            memory_dir: None,
            wb_mode: WbMode::Auto,
            ..NavConfig::default()
        },
        nav_handle: NavHandle::new(),
        commands: Some(inbox),
        console_out: Some(said.printer()),
    };
    let bot = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || run(cfg).expect("the bot starts"))
        .expect("thread");

    let mut type_line = |line: &str, wait_ms: u64| {
        eprintln!("[stdin] {line}");
        writeln!(pipe_out, "{line}").expect("the console is listening");
        std::thread::sleep(Duration::from_millis(wait_ms));
    };
    let mut spawned = false;
    for _ in 0..120 {
        type_line("!where", 500);
        if said.all().contains("tile (") {
            spawned = true;
            break;
        }
    }
    assert!(spawned, "the bot never spawned");

    // Waits until the life has lasted `secs` without a death (the console's own `deaths` counter), at most `limit`.
    let unbroken = |type_line: &mut dyn FnMut(&str, u64), secs: u64, limit: u64| {
        type_line("!stats", 400);
        let mut deaths = said.deaths();
        let mut since = Instant::now();
        let end = Instant::now() + Duration::from_secs(limit);
        while Instant::now() < end {
            std::thread::sleep(Duration::from_secs(4));
            type_line("!stats", 400);
            let now = said.deaths();
            if now != deaths {
                deaths = now;
                since = Instant::now();
            }
            if since.elapsed() >= Duration::from_secs(secs) {
                return true;
            }
        }
        false
    };
    // Round 1: a life over a minute, a dropped Cl_Kill, the one /kill, a death.
    assert!(
        unbroken(&mut type_line, 70, 150),
        "never had a 70 s life without a death"
    );
    let before = said.deaths().unwrap_or(0);
    type_line("!kill", 1500);
    assert!(said.all().contains("killing, respawning"), "{}", said.all());
    std::thread::sleep(Duration::from_secs(8));
    type_line("!stats", 400);
    assert!(
        said.deaths().unwrap_or(0) > before,
        "the bot did not die after the /kill: {}",
        said.all()
    );

    // Round 2: a fresh life: the protocol kill works by itself (no kill protection yet), and no /kill follows it.
    std::thread::sleep(Duration::from_secs(12));
    let before = said.deaths().unwrap_or(0);
    let mut killed = false;
    for _ in 0..6 {
        let n = said.all().matches("killing, respawning").count();
        type_line("!kill", 1500);
        if said.all().matches("killing, respawning").count() > n {
            killed = true;
            break;
        }
        std::thread::sleep(Duration::from_secs(5));
    }
    assert!(
        killed,
        "the console kill on a fresh life was never accepted: {}",
        said.all()
    );
    std::thread::sleep(Duration::from_secs(4));
    type_line("!stats", 400);
    assert!(
        said.deaths().unwrap_or(0) > before,
        "the protocol kill of a fresh life did nothing: {}",
        said.all()
    );

    // Round 3: another life over a minute: the second /kill.
    assert!(
        unbroken(&mut type_line, 70, 150),
        "never had a second 70 s life without a death"
    );
    let before = said.deaths().unwrap_or(0);
    let mut accepted = false;
    for _ in 0..6 {
        let n = said.all().matches("killing, respawning").count();
        type_line("!kill", 1500);
        if said.all().matches("killing, respawning").count() > n {
            accepted = true;
            break;
        }
        std::thread::sleep(Duration::from_secs(5));
    }
    assert!(accepted, "{}", said.all());
    // A second `!kill` right behind it, inside the cooldown of both the Cl_Kill and the /kill: the console refuses it, and no
    // second /kill follows (checked on the wire below: at most one /kill within 500 ticks of any Cl_Kill).
    let refused_before = said.all().matches("reset is on cooldown").count();
    type_line("!kill", 600);
    assert!(
        said.all().matches("reset is on cooldown").count() > refused_before,
        "the second !kill inside the cooldown must be refused: {}",
        said.all()
    );
    std::thread::sleep(Duration::from_secs(8));
    type_line("!stats", 400);
    assert!(
        said.deaths().unwrap_or(0) > before,
        "the bot did not die after the second /kill: {}",
        said.all()
    );
    type_line("!quit", 800);
    let report = bot.join().expect("the bot thread");

    eprintln!(
        "[audit] outgoing {:?}; Cl_Kill decisions at {:?}; /kill at {:?}",
        report.outgoing, report.kill_ticks, report.kill_command_ticks
    );
    // ---- the audit -----------------------------------------------------------------------------------
    assert_eq!(report.exit_code, 0, "gave up: {:?}", report.gave_up);
    for (label, (accepted, refused)) in &report.outgoing {
        assert_eq!(*refused, 0, "the allow-list refused a {label}");
        assert!(*accepted > 0);
    }
    let chat: Vec<_> = report
        .outgoing
        .keys()
        .filter(|k| (k.contains("Say") || k.contains("Chat")) && k.as_str() != SERVER_COMMAND_KILL_LABEL)
        .collect();
    assert!(chat.is_empty(), "chat other than /kill on the wire: {chat:?}");
    let (cmds, refused) = report
        .outgoing
        .get(SERVER_COMMAND_KILL_LABEL)
        .copied()
        .unwrap_or((0, 0));
    assert_eq!((cmds, refused), (2, 0), "exactly two /kill: {:?}", report.outgoing);
    assert_eq!(report.kill_command_ticks.len(), 2, "{:?}", report.kill_command_ticks);
    let kills = report.outgoing.get("Cl_Kill").map_or(0, |(ok, _)| *ok);
    assert!(
        kills >= 3,
        "three console kills went out as Cl_Kill: {:?}",
        report.outgoing
    );
    assert_eq!(kills as usize, report.kill_ticks.len());
    // every /kill answers a Cl_Kill of the bot's own, at least 50 ticks earlier and no more than a snapshot or two over; the
    // two /kill are at least 500 ticks apart
    for c in &report.kill_command_ticks {
        let k = report
            .kill_ticks
            .iter()
            .rev()
            .find(|k| **k <= *c)
            .expect("a Cl_Kill before the /kill");
        assert!(
            (50..=60).contains(&(c - k)) || c - k <= 6,
            "the /kill {} ticks after its Cl_Kill",
            c - k
        );
    }
    assert!(
        report.kill_command_ticks[1] - report.kill_command_ticks[0] >= 500,
        "{:?}",
        report.kill_command_ticks
    );
    // at most one /kill within 500 ticks of any Cl_Kill (the cooldown under pressure), and never more than three in a life
    // (this test has three lives, two of them protected)
    for k in &report.kill_ticks {
        let in_window = report
            .kill_command_ticks
            .iter()
            .filter(|c| **c >= *k && **c <= k + 500)
            .count();
        assert!(
            in_window <= 1,
            "more than one /kill within 500 ticks of the Cl_Kill at {k}: {:?}",
            report.kill_command_ticks
        );
    }
    assert!(report.kill_command_ticks.len() <= 9, "{:?}", report.kill_command_ticks);
}
