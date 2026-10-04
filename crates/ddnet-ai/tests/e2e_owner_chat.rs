//! Task 4.9 e2e (D-094): the owner types on the website and the bot says it in the game chat, against a **private** DDNet 20.1 server
//! that this test starts itself (UDP 127.0.0.1:8413, econ 127.0.0.1:8414, `sv_register 0`, its own scratch directory, its own random
//! econ password, stopped afterwards). `#[ignore]`d and guarded by `DDAI_E2E=1`:
//!
//! ```text
//! DDAI_E2E=1 cargo test -p ddnet-ai --test e2e_owner_chat -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Loopback only. It never touches the shared server (8303), the production units, `~/aiddnet/data/bot`, the allow-list or the secrets:
//! the bot, the control socket, the web unit (an ephemeral port, its own password and data directory) and the map cache all live in the
//! scratch directory (the server binary and the map are only read).
//!
//! What it proves, with the server's own log as the witness of what the game server received:
//! 1. through the real web route (session, CSRF, Origin) two chat lines are said in the game chat, the second one 3 s or more behind the
//!    first, in order, with exactly their text; a team line is said as team chat;
//! 2. a third line right behind the first two is refused by the web's own rate limit (429) and is never said;
//! 3. through the control socket (bypassing the web limit) the bot's own limits hold: one line goes at once, a queue of three waits for
//!    its turns, the fifth is refused with the reason `queue_full`; the four that were taken are said, in order, 3 s apart;
//! 4. a `/`-command and an empty line are refused (400) and never said; the server's log has no other chat from the bot, and no `/kill`;
//! 5. the bot's outgoing audit counts `Cl_Say(owner)` apart (accepted, none refused) and no other chat label; its log has "owner chat
//!    sent (len N)" lines and never the text.

use std::io::Write;
use std::net::{SocketAddr, TcpListener, UdpSocket};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ddai_bot::brains::{BrainKind, BrainOptions};
use ddai_bot::clipper::ClipConfig;
use ddai_bot::command::CommandBus;
use ddai_bot::control::{AuditSink, ControlServer, MemoryAudit};
use ddai_bot::nav_hooks::{NavConfig, NavHandle, WbMode};
use ddai_bot::runner::{RunnerConfig, run};
use ddai_bot::{BotConfig, Mode, Relations};
use ddai_client::ClientConfig;
use ddai_client::session::OWNER_SAY_LABEL;
use ddai_web::config::WebConfig;
use ddai_web::secrets::{self, Argon2Params, SecretsPaths};

const GAME_PORT: u16 = 8413;
const ECON_PORT: u16 = 8414;
const BOT_NAME: &str = "E2eSay";
const MAP: &str = "Copy Love Box";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/ddnet-ai is two levels under the repo root")
        .to_path_buf()
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

fn server_binary() -> PathBuf {
    std::env::var("DDAI_DDNET_SERVER").map_or_else(
        |_| home().join("aiddnet/build/ddnet-20.1/build/DDNet-Server"),
        PathBuf::from,
    )
}

/// 24 hex digits from the OS: the private server's econ password.
fn random_hex() -> String {
    let mut bytes = [0u8; 12];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut bytes))
        .expect("/dev/urandom");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The scratch directory goes away when the test ends, however it ends.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        // `DDAI_E2E_KEEP=1` keeps it (the server's log) for a look after a failure.
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
        let script = repo_root().join("tools/ddnet-server/econ.py");
        let out = Command::new("python3")
            .arg(script)
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
    let binary = server_binary();
    assert!(
        binary.is_file(),
        "no DDNet-Server at {binary:?} (build it: tools/ddnet-server/build.sh)"
    );
    let maps = scratch.join("maps");
    std::fs::create_dir_all(&maps).unwrap();
    let source = home().join("aiddnet/data/ddnet-server/maps").join(format!("{MAP}.map"));
    std::fs::copy(&source, maps.join(format!("{MAP}.map"))).unwrap_or_else(|e| panic!("cannot copy {source:?}: {e}"));
    // Storage: the scratch directory first (maps, saves), then DDNet's own read-only data.
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
             sv_name \"aiddnet e2e 4.9 (private, 127.0.0.1 only)\"\nsv_map \"{MAP}\"\n\
             sv_max_clients 4\nsv_max_clients_per_ip 4\nsv_connlimit_time 0\nsv_test_cmds 0\nclear_votes\n\
             sv_high_bandwidth 0\nsv_tee_historian 0\nec_bindaddr 127.0.0.1\nec_port {ECON_PORT}\nloglevel 0\n"
        ),
    )
    .unwrap();
    let log = scratch.join("server.log");
    let child = Command::new(&binary)
        .current_dir(scratch)
        .arg("-f")
        .arg(scratch.join("server.cfg"))
        .arg("-f")
        .arg(&secrets)
        .arg("bindaddr 127.0.0.1")
        .arg("sv_register 0")
        .arg(format!("logfile {}", log.display()))
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

// ---- reading the server's log -------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChatLine {
    /// `chat` or `teamchat`.
    category: String,
    text: String,
    /// Seconds since midnight, from the log's own timestamp.
    at: u32,
}

/// The chat lines the server logged for our bot, in order: `YYYY-MM-DD HH:MM:SS I chat: <id>:<team>:<name>: <text>`.
fn chat_lines(log: &str) -> Vec<ChatLine> {
    let mut out = Vec::new();
    for line in log.lines() {
        for category in ["teamchat", "chat"] {
            let marker = format!(" I {category}: ");
            let Some(pos) = line.find(&marker) else { continue };
            let rest = &line[pos + marker.len()..];
            let mut parts = rest.splitn(3, ':');
            let (_id, _team, tail) = (parts.next(), parts.next(), parts.next().unwrap_or(""));
            let Some(text) = tail.strip_prefix(&format!("{BOT_NAME}: ")) else {
                break;
            };
            out.push(ChatLine {
                category: category.to_string(),
                text: text.to_string(),
                at: log_time(line).unwrap_or(0),
            });
            break;
        }
    }
    out
}

/// `HH:MM:SS` of a log line's leading timestamp (`2026-10-04 12:34:56 I chat: ...`), as seconds.
fn log_time(line: &str) -> Option<u32> {
    let time = line.split(' ').nth(1)?;
    let mut it = time.split(':').map(|p| p.parse::<u32>().ok());
    Some(it.next()?? * 3600 + it.next()?? * 60 + it.next()??)
}

fn read_log(scratch: &Path) -> String {
    std::fs::read_to_string(scratch.join("server.log")).unwrap_or_default()
}

fn wait_for_chat(scratch: &Path, count: usize, limit: Duration) -> Vec<ChatLine> {
    let end = Instant::now() + limit;
    loop {
        let lines = chat_lines(&read_log(scratch));
        if lines.len() >= count || Instant::now() >= end {
            return lines;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

// ---- capturing the bot's log ------------------------------------------------------------------------------

#[derive(Clone, Default)]
struct LogBuf(Arc<Mutex<Vec<u8>>>);

impl Write for LogBuf {
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

// ---- a tiny HTTP client for the web unit -------------------------------------------------------------------

struct Site {
    base: String,
    http: reqwest::blocking::Client,
    cookie: String,
    csrf: String,
}

impl Site {
    fn login(addr: SocketAddr, password: &str) -> Site {
        let base = format!("http://{addr}");
        let http = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(15))
            .build()
            .unwrap();
        let r = http
            .post(format!("{base}/api/login"))
            .header("Origin", &base)
            .header("Content-Type", "application/json")
            .body(serde_json::json!({ "password": password }).to_string())
            .send()
            .expect("login request");
        assert_eq!(r.status().as_u16(), 200, "login");
        let cookie = r
            .headers()
            .get_all("set-cookie")
            .iter()
            .filter_map(|v| v.to_str().ok())
            .filter_map(|v| v.split(';').next())
            .collect::<Vec<_>>()
            .join("; ");
        let csrf = serde_json::from_slice::<serde_json::Value>(&r.bytes().unwrap()).unwrap()["csrf_token"]
            .as_str()
            .unwrap()
            .to_string();
        Site {
            base,
            http,
            cookie,
            csrf,
        }
    }

    fn post(&self, path: &str, body: &serde_json::Value) -> (u16, serde_json::Value) {
        let r = self
            .http
            .post(format!("{}{path}", self.base))
            .header("Cookie", &self.cookie)
            .header("Origin", &self.base)
            .header("X-CSRF-Token", &self.csrf)
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .expect("request");
        let status = r.status().as_u16();
        let body = r.bytes().unwrap_or_default();
        (status, serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null))
    }

    fn say(&self, team: bool, text: &str) -> (u16, serde_json::Value) {
        self.post("/api/bot/say", &serde_json::json!({ "team": team, "text": text }))
    }
}

#[test]
#[ignore = "starts a private DDNet server on 127.0.0.1:8413/8414; DDAI_E2E=1 and --ignored"]
fn the_owner_types_on_the_website_and_the_bot_says_it_in_the_game_chat() {
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

    let scratch_dir = std::env::temp_dir().join(format!("ddai-e2e-ownerchat-{}", std::process::id()));
    std::fs::create_dir_all(&scratch_dir).expect("a scratch directory");
    let _scratch = Scratch(scratch_dir.clone());
    let server = start_private_server(&scratch_dir);

    // ---- the bot, with its control socket (in the scratch directory) ------------------------------------
    let (sender, inbox) = CommandBus::open();
    let control_path = scratch_dir.join("bot").join("control.sock");
    std::fs::create_dir_all(control_path.parent().unwrap()).unwrap();
    let audit = Arc::new(MemoryAudit::default());
    let _control = ControlServer::start(&control_path, sender.clone(), Arc::clone(&audit) as Arc<dyn AuditSink>)
        .expect("the control socket");
    let cache_dir = scratch_dir.join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();
    let cfg = RunnerConfig {
        server: format!("127.0.0.1:{GAME_PORT}").parse::<SocketAddr>().unwrap(),
        client: ClientConfig {
            name: BOT_NAME.to_string(),
            cache_dir,
            adaptive_margin: true,
            ..ClientConfig::default()
        },
        bot: BotConfig {
            brain: BrainKind::Hybrid,
            mode: Mode::Fight,
            seed: 49,
            clips: ClipConfig {
                dir: None,
                autoclip: false,
                async_save: false,
            },
            console_names: true,
            ..BotConfig::default()
        },
        brain: BrainOptions {
            seed: 49,
            ..BrainOptions::default()
        },
        relations: Relations::new(),
        duration: Some(Duration::from_secs(240)),
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
        console_out: None,
    };
    let bot = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || run(cfg).expect("the bot starts"))
        .expect("thread");

    // ---- the web unit (its own port, password and data directory) -------------------------------------
    let web_dir = scratch_dir.join("web");
    std::fs::create_dir_all(&web_dir).unwrap();
    let password = secrets::generate_and_store_password(
        &SecretsPaths::new(&web_dir),
        Argon2Params {
            m_cost_kib: 8 * 1024,
            t_cost: 1,
            p_cost: 1,
        },
    )
    .expect("a password")
    .plaintext;
    let mut web_config = WebConfig::new(SocketAddr::from(([127, 0, 0, 1], 0)), web_dir.clone());
    web_config.control_socket = control_path.clone();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let bound = rt.block_on(ddai_web::bind(web_config)).expect("the web unit binds");
    let web_addr = bound.local_addr;
    assert_ne!(web_addr.port(), GAME_PORT);
    rt.spawn(async move {
        let _ = ddai_web::run(bound).await;
    });
    eprintln!("[web] http://{web_addr}");
    let site = Site::login(web_addr, &password);

    // ---- wait until the bot is in the game (the console's own `!where` answers with a tile) ------------
    let mut spawned = false;
    for _ in 0..120 {
        if let Ok(reply) = sender.send_line("!where", Duration::from_secs(2))
            && reply.text.contains("tile (")
        {
            spawned = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    assert!(spawned, "the bot never spawned on the private server");
    assert!(
        server.econ("status").is_some_and(|s| s.contains(BOT_NAME)),
        "the server does not list the bot"
    );
    std::thread::sleep(Duration::from_secs(2));

    // ---- 1 + 2: two lines through the web route, a third refused by the web's own limit ----------------
    let (s1, b1) = site.say(false, "  hello from the owner  ");
    eprintln!("[web] say 1 -> {s1} {b1}");
    assert_eq!(s1, 200, "{b1}");
    let (s2, b2) = site.say(false, "second line, привет");
    eprintln!("[web] say 2 -> {s2} {b2}");
    assert_eq!(s2, 200, "{b2}");
    assert!(
        b2["text"].as_str().unwrap_or("").contains("will be said in about"),
        "the second line is told it waits for its turn: {b2}"
    );
    let (s3, b3) = site.say(false, "third line right away");
    eprintln!("[web] say 3 -> {s3} {b3}");
    assert_eq!(
        (s3, b3["error"].as_str()),
        (429, Some("rate_limited")),
        "the web's own rate limit"
    );
    // refused here, never said
    let (sc, bc) = site.say(false, "/kill");
    assert_eq!(
        (sc, bc["error"].as_str()),
        (400, Some("invalid_text")),
        "a command is refused by the web"
    );
    let (se, _) = site.say(false, "   ");
    assert_eq!(se, 400);

    let lines = wait_for_chat(&scratch_dir, 2, Duration::from_secs(15));
    eprintln!("[server] chat: {lines:?}");
    assert_eq!(lines.len(), 2, "the server received exactly two lines: {lines:?}");
    assert_eq!(
        (lines[0].category.as_str(), lines[0].text.as_str()),
        ("chat", "hello from the owner"),
        "trimmed, in all chat"
    );
    assert_eq!(
        (lines[1].category.as_str(), lines[1].text.as_str()),
        ("chat", "second line, привет")
    );
    assert!(
        lines[1].at >= lines[0].at + 2,
        "the second line came 3 s behind the first (log seconds {} and {})",
        lines[0].at,
        lines[1].at
    );
    // nothing else follows (the third line was never accepted)
    std::thread::sleep(Duration::from_secs(4));
    let lines = chat_lines(&read_log(&scratch_dir));
    assert_eq!(lines.len(), 2, "still exactly two lines after waiting: {lines:?}");

    // ---- 3: the bot's own limits through the control socket (the web's limit is not in the way) ---------
    // One JSON line per request on the bot's `control.sock`, exactly as the web unit writes it.
    let control_say = |team: bool, text: &str| -> serde_json::Value {
        use std::io::{BufRead as _, BufReader};
        let mut stream = std::os::unix::net::UnixStream::connect(&control_path).expect("control.sock");
        stream.set_read_timeout(Some(Duration::from_secs(8))).unwrap();
        let request =
            serde_json::json!({"v": 1, "session": "e2e0", "cmd": {"type": "say", "team": team, "text": text}});
        writeln!(stream, "{request}").unwrap();
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).expect("a reply");
        serde_json::from_str(&line).expect("a JSON reply")
    };
    std::thread::sleep(Duration::from_secs(2)); // the pacing gap since the last line is over
    // The first goes out at once; the next three wait for their turn (the queue holds three); the fifth has no room.
    let r_a = control_say(false, "queue one");
    let r_b = control_say(true, "queue two (team)");
    let r_c = control_say(false, "queue three");
    let r_d = control_say(false, "queue four");
    let r_e = control_say(false, "queue five");
    eprintln!("[bot] {r_a}\n[bot] {r_b}\n[bot] {r_c}\n[bot] {r_d}\n[bot] {r_e}");
    assert!(
        [&r_a, &r_b, &r_c, &r_d].iter().all(|r| r["ok"] == true),
        "four lines are taken (one said now, three waiting): {r_a} {r_b} {r_c} {r_d}"
    );
    assert_eq!(r_e["ok"], false, "the fifth is refused: {r_e}");
    assert_eq!(r_e["data"]["reason"], "queue_full");
    assert!(r_e["text"].as_str().unwrap().contains("already waiting"), "{r_e}");
    assert!(
        !r_e.to_string().contains("queue five"),
        "the refusal never repeats the line"
    );
    let lines = wait_for_chat(&scratch_dir, 6, Duration::from_secs(30));
    eprintln!("[server] chat: {lines:?}");
    assert_eq!(lines.len(), 6, "exactly six lines in all: {lines:?}");
    let got: Vec<(&str, &str)> = lines[2..]
        .iter()
        .map(|l| (l.category.as_str(), l.text.as_str()))
        .collect();
    assert_eq!(
        got,
        [
            ("chat", "queue one"),
            ("teamchat", "queue two (team)"),
            ("chat", "queue three"),
            ("chat", "queue four")
        ],
        "in order, the team line as team chat"
    );
    for w in lines[1..].windows(2) {
        assert!(w[1].at >= w[0].at + 2, "paced 3 s apart: {lines:?}");
    }
    std::thread::sleep(Duration::from_secs(4));
    let lines = chat_lines(&read_log(&scratch_dir));
    assert_eq!(lines.len(), 6, "and nothing more (the fifth was refused): {lines:?}");
    let log = read_log(&scratch_dir);
    assert!(
        !log.contains("queue five") && !log.contains("third line right away") && !log.contains("/kill"),
        "refused lines never reached the server"
    );

    // ---- stop, then the audit -----------------------------------------------------------------------
    let _ = sender.send_line("!quit", Duration::from_secs(5));
    let report = bot.join().expect("the bot thread");
    eprintln!(
        "[audit] outgoing {:?}; owner chat {:?}",
        report.outgoing, report.owner_chat
    );
    assert_eq!(report.exit_code, 0, "gave up: {:?}", report.gave_up);
    for (label, (accepted, refused)) in &report.outgoing {
        assert_eq!(*refused, 0, "the allow-list refused a {label}");
        assert!(*accepted > 0);
    }
    assert_eq!(
        report.outgoing.get(OWNER_SAY_LABEL).copied(),
        Some((6, 0)),
        "six `Cl_Say(owner)`, counted apart: {:?}",
        report.outgoing
    );
    let chat_labels: Vec<_> = report
        .outgoing
        .keys()
        .filter(|k| k.contains("Say") || k.contains("Chat"))
        .collect();
    assert_eq!(
        chat_labels,
        [&OWNER_SAY_LABEL.to_string()],
        "no other chat label (no /kill): {chat_labels:?}"
    );
    assert_eq!(report.owner_chat.sent, 6);
    assert_eq!(report.owner_chat.accepted, 6);
    assert_eq!(report.owner_chat.refused, 1, "the queue_full one");
    assert_eq!(report.owner_chat.dropped, 0);
    assert!(report.kill_command_ticks.is_empty());

    // the bot's log: one "owner chat sent (len N)" per line, and never a line's text
    let log = String::from_utf8_lossy(&logbuf.0.lock().unwrap()).to_string();
    let sent_logs = log.lines().filter(|l| l.contains("owner chat sent (len ")).count();
    assert_eq!(sent_logs, 6, "six 'owner chat sent' lines in the bot's log");
    for secret in [
        "hello from the owner",
        "second line",
        "привет",
        "queue one",
        "queue two",
        "queue three",
        "queue four",
        "queue five",
        "third line",
    ] {
        assert!(!log.contains(secret), "the bot's log has a line's text: {secret:?}");
    }
    assert!(
        log.contains("owner chat sent (len 20)"),
        "the first line is 20 bytes: {log}"
    );
    // the control audit: tags only
    let entries = audit.0.lock().unwrap().clone();
    assert_eq!(
        entries.iter().filter(|e| e.cmd == "say:all").count(),
        2 + 4,
        "two from the web, and queue one, three, four and five from the socket"
    );
    assert_eq!(entries.iter().filter(|e| e.cmd == "say:team").count(), 1);
    for e in &entries {
        let line = e.to_line();
        assert!(!line.contains("queue") && !line.contains("hello"), "{line}");
    }
    eprintln!("[e2e] done");
}
