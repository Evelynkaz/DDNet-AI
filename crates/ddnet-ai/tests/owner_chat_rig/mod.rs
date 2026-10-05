//! The shared rig of the owner-chat e2e tests (tasks 4.9 and 4.9b, D-094): a **private** DDNet 20.1 server this test starts itself
//! (127.0.0.1 only, `sv_register 0`, its own scratch directory, its own random econ password, stopped afterwards), the bot with a
//! control socket, and the web unit (an ephemeral port, its own password and data directory), all in the scratch directory.
//!
//! Loopback only. Nothing here touches the shared server (8303), the production units, `~/aiddnet/data/bot`, the allow-list or the
//! secrets (the server binary and the map are only read).
//!
//! Not a test of its own (a directory module under `tests/`): each e2e test file declares `mod owner_chat_rig;`. **One rig per test
//! binary**: the global log subscriber and the bot's once-per-process `OwnerChannel` can each be taken once.

#![allow(dead_code)] // each test file uses the part of the rig it needs

use std::io::Write;
use std::net::{SocketAddr, TcpListener, UdpSocket};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ddai_bot::brains::{BrainKind, BrainOptions};
use ddai_bot::clipper::ClipConfig;
use ddai_bot::command::{CommandBus, CommandSender};
use ddai_bot::control::{AuditSink, ControlServer, MemoryAudit};
use ddai_bot::nav_hooks::{NavConfig, NavHandle, WbMode};
use ddai_bot::runner::{RunReport, RunnerConfig, run};
use ddai_bot::{BotConfig, Mode, Relations};
use ddai_client::ClientConfig;
use ddai_web::config::WebConfig;
use ddai_web::secrets::{self, Argon2Params, SecretsPaths};

pub const MAP: &str = "Copy Love Box";

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/ddnet-ai is two levels under the repo root")
        .to_path_buf()
}

pub fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

pub fn server_binary() -> PathBuf {
    std::env::var("DDAI_DDNET_SERVER").map_or_else(
        |_| home().join("aiddnet/build/ddnet-20.1/build/DDNet-Server"),
        PathBuf::from,
    )
}

/// 24 hex digits from the OS: the private server's econ password.
pub fn random_hex() -> String {
    let mut bytes = [0u8; 12];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut bytes))
        .expect("/dev/urandom");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The scratch directory goes away when the test ends, however it ends.
pub struct Scratch(pub PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        // `DDAI_E2E_KEEP=1` keeps it (the server's log) for a look after a failure.
        if std::env::var("DDAI_E2E_KEEP").as_deref() != Ok("1") {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

/// The private server: stopped (econ `shutdown`, then a kill) when the test ends, however it ends.
pub struct PrivateServer {
    child: Child,
    pub econ_password: String,
    pub game_port: u16,
    pub econ_port: u16,
}

impl PrivateServer {
    pub fn econ(&self, command: &str) -> Option<String> {
        let script = repo_root().join("tools/ddnet-server/econ.py");
        let out = Command::new("python3")
            .arg(script)
            .args([
                "--port",
                &self.econ_port.to_string(),
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

pub fn start_private_server(
    scratch: &Path,
    game_port: u16,
    econ_port: u16,
    sv_name: &str,
    extra_cfg: &str,
) -> PrivateServer {
    // The ports must be free: nothing else of ours may be answering there.
    UdpSocket::bind(("127.0.0.1", game_port)).unwrap_or_else(|e| panic!("UDP {game_port} is busy: {e}"));
    TcpListener::bind(("127.0.0.1", econ_port)).unwrap_or_else(|e| panic!("TCP {econ_port} is busy: {e}"));
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
            "bindaddr 127.0.0.1\nsv_port {game_port}\nsv_register 0\nsv_ipv4only 1\n\
             sv_name \"{sv_name}\"\nsv_map \"{MAP}\"\n\
             sv_max_clients 4\nsv_max_clients_per_ip 4\nsv_connlimit_time 0\nsv_test_cmds 0\nclear_votes\n\
             sv_high_bandwidth 0\nsv_tee_historian 0\nsv_pause_messages 1\nec_bindaddr 127.0.0.1\nec_port {econ_port}\nloglevel 0\n{extra_cfg}\n"
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
    let server = PrivateServer {
        child,
        econ_password,
        game_port,
        econ_port,
    };
    let deadline = Instant::now() + Duration::from_secs(40);
    loop {
        if let Some(out) = server.econ("status") {
            eprintln!("[server] up: {}", out.lines().next().unwrap_or(""));
            return server;
        }
        assert!(
            Instant::now() < deadline,
            "the private server never answered on econ {econ_port}"
        );
        std::thread::sleep(Duration::from_millis(500));
    }
}

// ---- reading the server's log -------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatLine {
    /// `chat` or `teamchat`.
    pub category: String,
    pub text: String,
    /// Seconds since midnight, from the log's own timestamp.
    pub at: u32,
}

/// The chat lines the server logged for our bot, in order: `YYYY-MM-DD HH:MM:SS I chat: <id>:<team>:<name>: <text>`.
pub fn chat_lines(log: &str, bot_name: &str) -> Vec<ChatLine> {
    let mut out = Vec::new();
    for line in log.lines() {
        for category in ["teamchat", "chat"] {
            let marker = format!(" I {category}: ");
            let Some(pos) = line.find(&marker) else { continue };
            let rest = &line[pos + marker.len()..];
            let mut parts = rest.splitn(3, ':');
            let (_id, _team, tail) = (parts.next(), parts.next(), parts.next().unwrap_or(""));
            let Some(text) = tail.strip_prefix(&format!("{bot_name}: ")) else {
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

pub fn read_log(scratch: &Path) -> String {
    std::fs::read_to_string(scratch.join("server.log")).unwrap_or_default()
}

pub fn wait_for_chat(scratch: &Path, bot_name: &str, count: usize, limit: Duration) -> Vec<ChatLine> {
    let end = Instant::now() + limit;
    loop {
        let lines = chat_lines(&read_log(scratch), bot_name);
        if lines.len() >= count || Instant::now() >= end {
            return lines;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

// ---- capturing the bot's log ------------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct LogBuf(Arc<Mutex<Vec<u8>>>);

impl LogBuf {
    /// Everything the bot logged so far.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).to_string()
    }
}

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

pub struct Site {
    base: String,
    http: reqwest::blocking::Client,
    cookie: String,
    csrf: String,
}

impl Site {
    pub fn login(addr: SocketAddr, password: &str) -> Site {
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

    pub fn post(&self, path: &str, body: &serde_json::Value) -> (u16, serde_json::Value) {
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

    pub fn say(&self, team: bool, text: &str) -> (u16, serde_json::Value) {
        self.post("/api/bot/say", &serde_json::json!({ "team": team, "text": text }))
    }
}

// ---- the whole rig ---------------------------------------------------------------------------------------

/// What differs between the tests.
pub struct RigConfig<'a> {
    pub game_port: u16,
    pub econ_port: u16,
    pub bot_name: &'a str,
    pub seed: u64,
    /// The server's `sv_name`, so a look at the server list or the log says whose it is.
    pub sv_name: &'a str,
    /// The scratch directory's name (it lives in the OS temp dir).
    pub scratch_tag: &'a str,
    /// Listen on a live bridge in the scratch directory and read the chat lines it carries (the server's own system lines too: the
    /// answers to commands reach the bot only as `Sv_Chat`, and the server's log does not show them).
    pub tap_chat: bool,
    /// Extra lines for the server's config (e.g. `sv_pauseable 1`), one per line; empty for none.
    pub server_cfg: &'a str,
}

/// The rig. Fields drop in order: the web runtime, then the server (stopped through econ), then the scratch directory.
pub struct Rig {
    pub site: Site,
    pub sender: CommandSender,
    pub control_path: PathBuf,
    pub audit: Arc<MemoryAudit>,
    pub logbuf: LogBuf,
    pub bot_name: String,
    pub scratch: PathBuf,
    /// The chat lines the bot passed to the bridge, as `(client id, text)`; empty without `tap_chat`.
    pub tapped: Arc<Mutex<Vec<(i64, String)>>>,
    /// The latest `STATUS` message (kind 5) of the bridge, as JSON; `None` before the first, and without `tap_chat`.
    pub status: Arc<Mutex<Option<serde_json::Value>>>,
    bot: Option<JoinHandle<RunReport>>,
    shutdown: Arc<AtomicBool>,
    _control: ControlServer,
    _rt: tokio::runtime::Runtime,
    pub server: PrivateServer,
    _scratch: Scratch,
}

impl Rig {
    /// Starts the server, the bot, the control socket and the web unit, logs in, and waits until the bot is in the game.
    pub fn start(cfg: &RigConfig) -> Rig {
        let logbuf = LogBuf::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(logbuf.clone())
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("one subscriber for this test binary");

        let scratch_dir = std::env::temp_dir().join(format!("{}-{}", cfg.scratch_tag, std::process::id()));
        std::fs::create_dir_all(&scratch_dir).expect("a scratch directory");
        let scratch = Scratch(scratch_dir.clone());
        let server = start_private_server(&scratch_dir, cfg.game_port, cfg.econ_port, cfg.sv_name, cfg.server_cfg);

        // ---- the bot, with its control socket (in the scratch directory) -----------------------------
        let (sender, inbox) = CommandBus::open();
        let control_path = scratch_dir.join("bot").join("control.sock");
        std::fs::create_dir_all(control_path.parent().unwrap()).unwrap();
        let audit = Arc::new(MemoryAudit::default());
        let control = ControlServer::start(&control_path, sender.clone(), Arc::clone(&audit) as Arc<dyn AuditSink>)
            .expect("the control socket");
        let cache_dir = scratch_dir.join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        let bridge_path = cfg.tap_chat.then(|| scratch_dir.join("bot").join("live.sock"));
        let run_cfg = RunnerConfig {
            server: format!("127.0.0.1:{}", cfg.game_port).parse::<SocketAddr>().unwrap(),
            client: ClientConfig {
                name: cfg.bot_name.to_string(),
                cache_dir,
                adaptive_margin: true,
                ..ClientConfig::default()
            },
            bot: BotConfig {
                brain: BrainKind::Hybrid,
                mode: Mode::Fight,
                seed: cfg.seed,
                clips: ClipConfig {
                    dir: None,
                    autoclip: false,
                    async_save: false,
                },
                console_names: true,
                ..BotConfig::default()
            },
            brain: BrainOptions {
                seed: cfg.seed,
                ..BrainOptions::default()
            },
            relations: Relations::new(),
            duration: Some(Duration::from_secs(240)),
            bridge_path: bridge_path.clone(),
            web_names: false,
            debug_names_log: None,
            audit_outgoing: true,
            shutdown: Arc::clone(&shutdown),
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
            .spawn(move || run(run_cfg).expect("the bot starts"))
            .expect("thread");

        // ---- the web unit (its own port, password and data directory) --------------------------------
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
        assert_ne!(web_addr.port(), cfg.game_port);
        rt.spawn(async move {
            let _ = ddai_web::run(bound).await;
        });
        eprintln!("[web] http://{web_addr}");
        let site = Site::login(web_addr, &password);

        let tapped = Arc::new(Mutex::new(Vec::new()));
        let status = Arc::new(Mutex::new(None));
        if let Some(path) = bridge_path {
            tap_bridge_chat(path, Arc::clone(&tapped), Arc::clone(&status), Arc::clone(&shutdown));
        }
        let rig = Rig {
            site,
            sender,
            control_path,
            audit,
            logbuf,
            bot_name: cfg.bot_name.to_string(),
            scratch: scratch_dir,
            tapped,
            status,
            bot: Some(bot),
            shutdown,
            _control: control,
            _rt: rt,
            server,
            _scratch: scratch,
        };
        // ---- wait until the bot is in the game (the console's own `!where` answers with a tile) -----
        let mut spawned = false;
        for _ in 0..120 {
            if let Ok(reply) = rig.sender.send_line("!where", Duration::from_secs(2))
                && reply.text.contains("tile (")
            {
                spawned = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        assert!(spawned, "the bot never spawned on the private server");
        assert!(
            rig.server.econ("status").is_some_and(|s| s.contains(cfg.bot_name)),
            "the server does not list the bot"
        );
        std::thread::sleep(Duration::from_secs(2));
        rig
    }

    /// The chat lines the server logged for the bot, in order.
    pub fn chat(&self) -> Vec<ChatLine> {
        chat_lines(&read_log(&self.scratch), &self.bot_name)
    }

    pub fn wait_for_chat(&self, count: usize, limit: Duration) -> Vec<ChatLine> {
        wait_for_chat(&self.scratch, &self.bot_name, count, limit)
    }

    /// Waits until `cond` holds, up to `limit`.
    pub fn wait_until(&self, limit: Duration, cond: impl Fn() -> bool) -> bool {
        let end = Instant::now() + limit;
        loop {
            if cond() {
                return true;
            }
            if Instant::now() >= end {
                return false;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// A field of the latest tapped `STATUS`.
    pub fn status_field(&self, name: &str) -> Option<serde_json::Value> {
        self.status.lock().unwrap().as_ref().map(|s| s[name].clone())
    }

    /// How many tapped chat lines satisfy `pred`.
    pub fn tapped_count(&self, pred: impl Fn(i64, &str) -> bool) -> usize {
        self.tapped
            .lock()
            .unwrap()
            .iter()
            .filter(|(cid, text)| pred(*cid, text))
            .count()
    }

    /// Waits until a tapped chat line satisfies `pred`, up to `limit`.
    pub fn wait_for_tapped(&self, limit: Duration, pred: impl Fn(i64, &str) -> bool) -> bool {
        self.wait_until(limit, || self.tapped_count(&pred) > 0)
    }

    pub fn server_log(&self) -> String {
        read_log(&self.scratch)
    }

    /// Waits until the server's log has `needle`, up to `limit`.
    pub fn wait_for_log(&self, needle: &str, limit: Duration) -> bool {
        let end = Instant::now() + limit;
        loop {
            if self.server_log().contains(needle) {
                return true;
            }
            if Instant::now() >= end {
                return false;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// One JSON line on the bot's `control.sock`, exactly as the web unit writes it (the web's own rate limit is not in the way).
    pub fn control_say(&self, team: bool, text: &str) -> serde_json::Value {
        use std::io::{BufRead as _, BufReader};
        let mut stream = std::os::unix::net::UnixStream::connect(&self.control_path).expect("control.sock");
        stream.set_read_timeout(Some(Duration::from_secs(8))).unwrap();
        let request =
            serde_json::json!({"v": 1, "session": "e2e0", "cmd": {"type": "say", "team": team, "text": text}});
        writeln!(stream, "{request}").unwrap();
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).expect("a reply");
        serde_json::from_str(&line).expect("a JSON reply")
    }

    /// Whether the bot's run has ended (it stopped itself, or its time is up).
    pub fn bot_finished(&self) -> bool {
        self.bot.as_ref().is_some_and(JoinHandle::is_finished)
    }

    /// Waits for the bot's run to end on its own, up to `limit`; `None` when it is still running.
    pub fn wait_for_bot(&mut self, limit: Duration) -> Option<RunReport> {
        let end = Instant::now() + limit;
        while !self.bot_finished() {
            if Instant::now() >= end {
                return None;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        self.bot.take().map(|h| h.join().expect("the bot thread"))
    }

    /// Asks the bot to quit and returns its report.
    pub fn quit(&mut self) -> RunReport {
        let _ = self.sender.send_line("!quit", Duration::from_secs(5));
        self.bot
            .take()
            .expect("the bot thread is still ours")
            .join()
            .expect("the bot thread")
    }
}

/// A raw reader of the bot's live bridge (`u32 LE len | u8 kind | payload`, `ddai_bot::bridge`): collects the `CHAT` messages (kind 8,
/// JSON `{"team","cid","name","text"}`) and the latest `STATUS` (kind 5) until the rig is dropped.
fn tap_bridge_chat(
    path: PathBuf,
    into: Arc<Mutex<Vec<(i64, String)>>>,
    status: Arc<Mutex<Option<serde_json::Value>>>,
    stop: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        use std::io::Read as _;
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
                if msg.first() == Some(&8)
                    && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&msg[1..])
                {
                    into.lock().unwrap().push((
                        v["cid"].as_i64().unwrap_or(i64::MIN),
                        v["text"].as_str().unwrap_or("").to_string(),
                    ));
                }
            }
        }
    });
}

impl Drop for Rig {
    fn drop(&mut self) {
        // A test that failed must not leave the bot talking to the server for the rest of its 240 s.
        self.shutdown.store(true, Ordering::SeqCst);
    }
}
