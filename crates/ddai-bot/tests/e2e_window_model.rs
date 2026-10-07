//! Task 3.17 e2e (D-111): the learned window model in the real bot, against a **private** DDNet 20.1 server that this test starts itself (UDP
//! 127.0.0.1:8443, econ 127.0.0.1:8444, `sv_register 0`, its own scratch directory, its own random econ password, stopped afterwards), with a
//! **scripted opponent** (a second client that walks a few tiles right and left, jumps, hooks and turns
//! its aim on a fixed timeline, and kills itself every 15 s so that it is not frozen for good). `#[ignore]`d and guarded by `DDAI_E2E=1`:
//!
//! ```text
//! DDAI_E2E=1 cargo test -p ddai-bot --test e2e_window_model -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Loopback only. Nothing here touches the shared server (8303), the production units or `~/aiddnet/data/bot`: the marker, the log, the map cache and
//! the model copy live in the scratch directory (the server binary, the map and the model file are only read).
//!
//! **Phase A, the real model** (`~/aiddnet/data/runs/E-028/m1.oppnet`, or `DDAI_WINDOW_MODEL`; the phase is skipped without it): the bot (the
//! hybrid brain, `BotConfig::window_model`) loads it (its sha256 is in the bot's log), the opponent becomes its target, the model is *called* (STATUS
//! `window_guard.predicted` grows) and the guard *scores* windows against the snapshots that follow (`resolved` grows). The kill marker switches the
//! model off within a few seconds (STATUS `killed`, no more calls) and on again. After the run the log file is there, parses, carries the model's
//! sha256 in its header and has no nickname in it.
//!
//! The scripted opponent aims in circles, not at the bot, which the arena opponents never do: the real model may well lose to hold on it and be
//! benched by the guard (the phase accepts `on` and `hold` alike, and says which it was).
//!
//! **Phase B, the guard**: a deliberately wrong model (a constant "walks left"; it loses to hold, which follows what the snapshots show) with a short
//! guard span: the guard benches it (STATUS `hold`, a `guard` line in the log), and the model keeps being scored in the shadow.

use std::io::{Read, Write as _};
use std::net::{SocketAddr, TcpListener, UdpSocket};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ddai_bot::brains::{BrainKind, BrainOptions};
use ddai_bot::clipper::ClipConfig;
use ddai_bot::nav_hooks::{NavConfig, NavHandle, WbMode};
use ddai_bot::oppnet::WindowModelConfig;
use ddai_bot::runner::{RunReport, RunnerConfig, run};
use ddai_bot::{BotConfig, Mode, Relations};
use ddai_client::{Client, ClientConfig};
use ddai_net::generated::objects::PlayerInput;
use ddai_oppnet::bundle::OppBundle;
use ddai_oppnet::feature::{INPUT_DIM, OUT_DIM};
use ddai_oppnet::live::analyze::Report;
use ddai_oppnet::live::guard::GuardConfig;
use ddai_oppnet::net::Mlp;

const GAME_PORT: u16 = 8443;
const ECON_PORT: u16 = 8444;
const BOT: &str = "E2eWindowBot";
const OPPONENT: &str = "E2eScriptOpp";

// ---- the private server (a small copy of the rig of `ddnet-ai/tests/owner_chat_rig`: 127.0.0.1 only, `sv_register 0`, its own scratch) --------------

const MAP: &str = "Copy Love Box";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/ddai-bot is two levels under the repo root")
        .to_path_buf()
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

/// 24 hex digits from the OS: the private server's econ password.
fn random_hex() -> String {
    let mut bytes = [0u8; 12];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| Read::read_exact(&mut f, &mut bytes))
        .expect("/dev/urandom");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The scratch directory goes away when the test ends, however it ends (`DDAI_E2E_KEEP=1` keeps it).
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        if std::env::var("DDAI_E2E_KEEP").as_deref() != Ok("1") {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

/// The private server: stopped (econ `shutdown`, then a kill) when the test ends, however it ends.
struct PrivateServer {
    child: Child,
    econ_password: String,
}

impl PrivateServer {
    fn econ(&self, command: &str) -> Option<String> {
        let out = Command::new("python3")
            .arg(repo_root().join("tools/ddnet-server/econ.py"))
            .args([
                "--port",
                &ECON_PORT.to_string(),
                "--password",
                &self.econ_password,
                "--retries",
                "3",
            ])
            .arg(command)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).to_string())
    }
}

impl Drop for PrivateServer {
    fn drop(&mut self) {
        let _ = self.econ("shutdown");
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_private_server(scratch: &Path) -> PrivateServer {
    // The ports must be free: nothing else of ours may be answering there.
    UdpSocket::bind(("127.0.0.1", GAME_PORT)).unwrap_or_else(|e| panic!("UDP {GAME_PORT} is busy: {e}"));
    TcpListener::bind(("127.0.0.1", ECON_PORT)).unwrap_or_else(|e| panic!("TCP {ECON_PORT} is busy: {e}"));
    let binary = std::env::var("DDAI_DDNET_SERVER").map_or_else(
        |_| home().join("aiddnet/build/ddnet-20.1/build/DDNet-Server"),
        PathBuf::from,
    );
    assert!(
        binary.is_file(),
        "no DDNet-Server at {binary:?} (build it: tools/ddnet-server/build.sh)"
    );
    let maps = scratch.join("maps");
    std::fs::create_dir_all(&maps).unwrap();
    let source = home().join("aiddnet/data/ddnet-server/maps").join(format!("{MAP}.map"));
    std::fs::copy(&source, maps.join(format!("{MAP}.map"))).unwrap_or_else(|e| panic!("cannot copy {source:?}: {e}"));
    std::fs::write(
        scratch.join("storage.cfg"),
        format!(
            "add_path {}\nadd_path {}\n",
            scratch.display(),
            home().join("aiddnet/build/ddnet-20.1/src/data").display()
        ),
    )
    .unwrap();
    let econ_password = random_hex();
    let secrets = scratch.join("secrets.cfg");
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&secrets)
            .unwrap();
        writeln!(f, "ec_password \"{econ_password}\"").unwrap();
    }
    std::fs::write(
        scratch.join("server.cfg"),
        format!(
            "bindaddr 127.0.0.1\nsv_port {GAME_PORT}\nsv_register 0\nsv_ipv4only 1\n\
             sv_name \"aiddnet e2e 3.17 (private, 127.0.0.1 only)\"\nsv_map \"{MAP}\"\n\
             sv_max_clients 4\nsv_max_clients_per_ip 4\nsv_connlimit_time 0\nsv_test_cmds 0\nclear_votes\n\
             sv_high_bandwidth 0\nsv_tee_historian 0\nsv_pause_messages 1\nec_bindaddr 127.0.0.1\nec_port {ECON_PORT}\nloglevel 0\n"
        ),
    )
    .unwrap();
    // `logfile` takes at most 127 characters: the name is relative to the server's working directory, the scratch directory.
    let child = Command::new(&binary)
        .current_dir(scratch)
        .arg("-f")
        .arg(scratch.join("server.cfg"))
        .arg("-f")
        .arg(&secrets)
        .arg("bindaddr 127.0.0.1")
        .arg("sv_register 0")
        .arg("logfile server.log")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("DDNet-Server starts");
    let server = PrivateServer { child, econ_password };
    let deadline = Instant::now() + Duration::from_secs(40);
    loop {
        if let Some(out) = server.econ("status") {
            eprintln!("[server] up: {}", out.lines().next().unwrap_or(""));
            return server;
        }
        assert!(
            Instant::now() < deadline,
            "the private server never answered on econ {ECON_PORT}"
        );
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// The bot's log, captured.
#[derive(Clone, Default)]
struct LogBuf(Arc<Mutex<Vec<u8>>>);

impl LogBuf {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).to_string()
    }
}

impl std::io::Write for LogBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuf {
    type Writer = LogBuf;

    fn make_writer(&'a self) -> LogBuf {
        self.clone()
    }
}

/// The scripted opponent: a client that pumps its events and sets a new input every 20 ms from a fixed timeline: short walks right and left (it
/// stays within a few tiles of where it spawns, out of the freeze tubes), a jump, a hook, a turning aim; it kills itself every 15 s (a fresh tee
/// at the spawn, so it is not frozen for the rest of the test after the bot has blocked it once).
fn spawn_opponent(server: SocketAddr, cache: PathBuf, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let client = Client::connect(
            server,
            ClientConfig {
                name: OPPONENT.to_string(),
                cache_dir: cache,
                ..ClientConfig::default()
            },
        );
        let t0 = Instant::now();
        let mut last_kill = Instant::now();
        while !stop.load(Ordering::SeqCst) {
            let _ = client.recv_event(Duration::from_millis(20));
            let ms = t0.elapsed().as_millis() as i64;
            // 8 steps of 250 ms: right, right, left, left, jump left, stand, hook right, left.
            let step = (ms / 250) % 8;
            let a = ms as f64 / 900.0;
            let input = PlayerInput {
                direction: match step {
                    0 | 1 | 6 => 1,
                    2..=4 | 7 => -1,
                    _ => 0,
                },
                jump: i32::from(step == 4 && (ms % 250) < 100),
                hook: i32::from(step == 6),
                target_x: (a.cos() * 300.0) as i32,
                target_y: (a.sin() * 300.0) as i32,
                fire: 0,
                player_flags: 1,
                wanted_weapon: 0,
                next_weapon: 0,
                prev_weapon: 0,
            };
            client.set_input(input);
            if last_kill.elapsed() > Duration::from_secs(15) {
                client.kill();
                last_kill = Instant::now();
            }
        }
    })
}

/// Reads the bridge (`u32 LE len | u8 kind | payload`) and keeps the latest `STATUS` (kind 5).
fn tap_status(path: PathBuf, status: Arc<Mutex<Option<serde_json::Value>>>, stop: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let mut stream = loop {
            if stop.load(Ordering::SeqCst) {
                return;
            }
            match std::os::unix::net::UnixStream::connect(&path) {
                Ok(s) => break s,
                Err(_) => std::thread::sleep(Duration::from_millis(200)),
            }
        };
        let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 8192];
        while !stop.load(Ordering::SeqCst) {
            match stream.read(&mut chunk) {
                Ok(0) => return,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
                Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
                Err(_) => return,
            }
            while buf.len() >= 4 {
                let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
                if buf.len() < 4 + len {
                    break;
                }
                let msg: Vec<u8> = buf.drain(..4 + len).skip(4).collect();
                if msg.first() == Some(&5)
                    && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&msg[1..])
                {
                    *status.lock().unwrap() = Some(v);
                }
            }
        }
    });
}

struct Bot {
    status: Arc<Mutex<Option<serde_json::Value>>>,
    shutdown: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<RunReport>>,
    stop_tap: Arc<AtomicBool>,
}

impl Bot {
    fn field(&self, name: &str) -> Option<serde_json::Value> {
        self.status.lock().unwrap().as_ref().map(|s| s[name].clone())
    }

    fn guard(&self, name: &str) -> Option<f64> {
        self.field("window_guard")?.get(name)?.as_f64()
    }

    /// One line of the latest STATUS, for the test's progress output.
    fn summary(&self) -> String {
        let s = self.status.lock().unwrap();
        let Some(s) = s.as_ref() else {
            return "no STATUS yet".into();
        };
        format!(
            "tick {} alive {} frozen {} target {} blocks {} decisions {} brain {} | window_model {} guard {}",
            s["tick"],
            s["alive"],
            s["frozen"],
            s["target"],
            s["blocks"],
            s["decisions"],
            s["brain"],
            s["window_model"],
            s["window_guard"]
        )
    }

    fn wait(&self, limit: Duration, cond: impl Fn(&Bot) -> bool) -> bool {
        let end = Instant::now() + limit;
        let mut next_note = Instant::now() + Duration::from_secs(10);
        loop {
            if cond(self) {
                return true;
            }
            if Instant::now() >= end {
                return false;
            }
            if Instant::now() >= next_note {
                eprintln!("  .. {}", self.summary());
                next_note += Duration::from_secs(10);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    fn finish(mut self) -> RunReport {
        self.shutdown.store(true, Ordering::SeqCst);
        let r = self.thread.take().unwrap().join().expect("the bot thread");
        self.stop_tap.store(true, Ordering::SeqCst);
        r
    }
}

fn start_bot(scratch: &Path, model: &WindowModelConfig, seed: u64) -> Bot {
    let bot_dir = scratch.join("data").join("bot");
    std::fs::create_dir_all(&bot_dir).unwrap();
    let cache = scratch.join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let status = Arc::new(Mutex::new(None));
    let stop_tap = Arc::new(AtomicBool::new(false));
    let bridge = bot_dir.join("live.sock");
    tap_status(bridge.clone(), Arc::clone(&status), Arc::clone(&stop_tap));
    let cfg = RunnerConfig {
        server: format!("127.0.0.1:{GAME_PORT}").parse().unwrap(),
        client: ClientConfig {
            name: BOT.to_string(),
            cache_dir: cache,
            adaptive_margin: true,
            ..ClientConfig::default()
        },
        bot: BotConfig {
            brain: BrainKind::Hybrid,
            mode: Mode::Fight,
            seed,
            clips: ClipConfig {
                dir: None,
                autoclip: false,
                async_save: false,
            },
            window_model: Some(model.clone()),
            ..BotConfig::default()
        },
        brain: BrainOptions {
            seed,
            ..BrainOptions::default()
        },
        relations: Relations::new(),
        duration: Some(Duration::from_secs(240)),
        bridge_path: Some(bridge),
        web_names: false,
        debug_names_log: None,
        audit_outgoing: true,
        shutdown: Arc::clone(&shutdown),
        nav: NavConfig {
            memory_dir: None,
            wb_mode: WbMode::Off,
            ..NavConfig::default()
        },
        nav_handle: NavHandle::new(),
        commands: None,
        console_out: None,
    };
    let thread = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || run(cfg).expect("the bot starts"))
        .expect("thread");
    Bot {
        status,
        shutdown,
        thread: Some(thread),
        stop_tap,
    }
}

/// A constant network: tick-wise direction class `dir_class` (0 left, 1 none, 2 right).
fn write_constant_model(path: &Path, dir_class: usize) {
    let mut net = Mlp::new(INPUT_DIM, 8, 8, OUT_DIM, 1);
    net.params.iter_mut().for_each(|p| *p = 0.0);
    let n = net.params.len();
    for k in 0..8 {
        net.params[n - OUT_DIM + k * 7 + dir_class] = 5.0;
    }
    OppBundle::new(net, 1, 1, 0.0, "constant".into()).save(path).unwrap();
}

fn read_log(path: &Path) -> (String, Report) {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut r = Report::new();
    for l in text.lines() {
        r.add_line(l);
    }
    (text, r)
}

#[test]
#[ignore = "starts a private DDNet server on 127.0.0.1:8443/8444; DDAI_E2E=1 and --ignored"]
fn the_window_model_loads_predicts_is_guarded_and_logged_against_a_scripted_opponent() {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set DDAI_E2E=1");
        return;
    }
    let logbuf = LogBuf::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(logbuf.clone())
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("one subscriber for this test binary");

    let scratch =
        Scratch(std::env::temp_dir().join(format!("ddai-e2e-window-{}-{}", std::process::id(), random_hex())));
    std::fs::create_dir_all(&scratch.0).unwrap();
    let server = start_private_server(&scratch.0);
    let server_addr: SocketAddr = format!("127.0.0.1:{GAME_PORT}").parse().unwrap();
    let stop_opp = Arc::new(AtomicBool::new(false));
    let opp = spawn_opponent(server_addr, scratch.0.join("cache-opp"), Arc::clone(&stop_opp));
    // The scripted opponent is on the server before the bot looks for a target.
    assert!(
        (0..60).any(|_| {
            std::thread::sleep(Duration::from_millis(500));
            server.econ("status").is_some_and(|s| s.contains(OPPONENT))
        }),
        "the scripted opponent never joined"
    );

    // ---- phase A: the real model ---------------------------------------------------------------------------
    let model_path = std::env::var_os("DDAI_WINDOW_MODEL")
        .map_or_else(|| home().join("aiddnet/data/runs/E-028/m1.oppnet"), PathBuf::from);
    if model_path.is_file() {
        let data_dir = scratch.0.join("data");
        let mut cfg = WindowModelConfig::in_data_dir(model_path.clone(), &data_dir);
        cfg.guard = GuardConfig::default();
        let marker = cfg.marker.clone().unwrap();
        let log_path = cfg.log.clone().unwrap();
        let bot = start_bot(&scratch.0, &cfg, 1);
        // The model is called and the guard scores windows against the snapshots that follow.
        let ok = bot.wait(Duration::from_secs(90), |b| {
            b.guard("predicted").unwrap_or(0.0) >= 100.0 && b.guard("resolved").unwrap_or(0.0) >= 20.0
        });
        eprintln!(
            "[A] STATUS window_model {:?} window_guard {:?} target {:?}",
            bot.field("window_model"),
            bot.field("window_guard"),
            bot.field("target_tag")
        );
        assert!(
            ok,
            "the model was never called against the scripted opponent (no target?): {:?}",
            bot.field("window_guard")
        );
        let word = bot
            .field("window_model")
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap();
        assert!(word == "on" || word == "hold", "{word}");
        let sha = bot.field("window_guard").unwrap()["sha256"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(sha.len(), 64);
        assert!(
            logbuf.text().contains(&format!("sha256={sha}")),
            "the load line of the bot's log has the model's sha256"
        );
        eprintln!("[A] model sha256 {sha}");

        // The kill marker switches the model off within a few seconds, and on again.
        std::fs::write(&marker, "").unwrap();
        assert!(
            bot.wait(Duration::from_secs(6), |b| b
                .field("window_model")
                .is_some_and(|v| v == "killed")),
            "the marker did not switch the model off: {:?}",
            bot.field("window_model")
        );
        let frozen_at = bot.guard("predicted").unwrap();
        std::thread::sleep(Duration::from_secs(3));
        assert_eq!(bot.guard("predicted"), Some(frozen_at), "a killed model is not called");
        std::fs::remove_file(&marker).unwrap();
        assert!(
            bot.wait(Duration::from_secs(6), |b| b
                .field("window_model")
                .is_some_and(|v| v == "on" || v == "hold")),
            "the model did not come back when the marker went"
        );
        assert!(
            bot.wait(Duration::from_secs(20), |b| b.guard("predicted").unwrap_or(0.0)
                > frozen_at),
            "no call after the marker went"
        );
        let report = bot.finish();
        assert_eq!(report.exit_code, 0, "{:?}", report.gave_up);
        assert!(report.stats.brain_decisions > 100, "{:?}", report.stats);

        let (text, r) = read_log(&log_path);
        let header: serde_json::Value = serde_json::from_str(text.lines().next().expect("a header line")).unwrap();
        assert_eq!(header["ev"], "open");
        assert_eq!(
            header["model"].as_str(),
            Some(sha.as_str()),
            "the log says which model wrote it"
        );
        assert_eq!(r.bad_lines, 0);
        assert!(r.samples() > 100, "{} samples", r.samples());
        let lags: Vec<u8> = (0..9).filter(|&w| r.pooled(Some(w), None).n > 0).collect();
        assert!(
            !lags.is_empty() && lags.iter().all(|&w| (1..=8).contains(&w)),
            "{lags:?}"
        );
        let low = text.to_lowercase();
        assert!(
            !low.contains(&OPPONENT.to_lowercase()) && !low.contains(&BOT.to_lowercase()),
            "no nickname in the log"
        );
        eprintln!("[A] windows of lag {lags:?}\n{}", r.render(false));
    } else {
        eprintln!("[A] skipped: no model at {}", model_path.display());
    }

    // ---- phase B: a wrong model is benched by the guard ------------------------------------------------------
    let wrong = scratch.0.join("left.oppnet");
    write_constant_model(&wrong, 0);
    let data_dir_b = scratch.0.join("data-b");
    let mut cfg = WindowModelConfig::in_data_dir(wrong, &data_dir_b);
    cfg.guard = GuardConfig {
        windows: 60,
        min_windows: 30,
        margin: 0.05,
        retry_after: 100_000,
        retry_margin: 0.0,
    };
    let log_b = cfg.log.clone().unwrap();
    // The helper puts the bot's files below `scratch/data`; phase B has its own marker and log below `data-b`.
    let bot = start_bot(&scratch.0, &cfg, 2);
    let benched = bot.wait(Duration::from_secs(120), |b| {
        b.field("window_model").is_some_and(|v| v == "hold")
    });
    eprintln!(
        "[B] STATUS window_model {:?} window_guard {:?}",
        bot.field("window_model"),
        bot.field("window_guard")
    );
    assert!(
        benched,
        "the guard never benched a model that says \"left\": {:?}",
        bot.field("window_guard")
    );
    let predicted_then = bot.guard("predicted").unwrap();
    assert!(bot.guard("fallbacks").unwrap() >= 1.0);
    assert!(bot.guard("model_cost").unwrap() > bot.guard("hold_cost").unwrap());
    assert!(
        bot.wait(Duration::from_secs(60), |b| b.guard("predicted").unwrap_or(0.0)
            > predicted_then + 5.0),
        "benched, the model is still run in the shadow"
    );
    assert!(bot.guard("used").unwrap() < bot.guard("predicted").unwrap());
    let report = bot.finish();
    assert_eq!(report.exit_code, 0, "{:?}", report.gave_up);
    let (text, _) = read_log(&log_b);
    assert!(
        text.lines()
            .any(|l| l.contains(r#""ev":"guard""#) && l.contains(r#""to":"hold""#)),
        "the log has the guard's change of state"
    );
    assert!(
        logbuf.text().contains("the guard benched the model"),
        "the bot's own log says so"
    );

    stop_opp.store(true, Ordering::SeqCst);
    opp.join().unwrap();
    drop(server);
}
