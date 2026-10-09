//! A private DDNet 20.1 server for the e2e tests (127.0.0.1 only, `sv_register 0`, its own scratch directory and econ password), stopped when dropped.
#![allow(dead_code)]

use std::io::{Read, Write as _};
use std::net::SocketAddr;
use std::net::{TcpListener, UdpSocket};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---- the private server (a small copy of the rig of `ddnet-ai/tests/owner_chat_rig`: 127.0.0.1 only, `sv_register 0`, its own scratch) --------------

pub const MAP: &str = "Copy Love Box";

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/ddai-bot is two levels under the repo root")
        .to_path_buf()
}

pub fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

/// 24 hex digits from the OS: the private server's econ password.
pub fn random_hex() -> String {
    let mut bytes = [0u8; 12];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| Read::read_exact(&mut f, &mut bytes))
        .expect("/dev/urandom");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The scratch directory goes away when the test ends, however it ends (`DDAI_E2E_KEEP=1` keeps it).
pub struct Scratch(pub PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        if std::env::var("DDAI_E2E_KEEP").as_deref() != Ok("1") {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

/// The private server: stopped (econ `shutdown`, then a kill) when the test ends, however it ends.
pub struct PrivateServer {
    child: Child,
    econ_password: String,
    pub game_port: u16,
    pub econ_port: u16,
}

impl PrivateServer {
    pub fn econ(&self, command: &str) -> Option<String> {
        let out = Command::new("python3")
            .arg(repo_root().join("tools/ddnet-server/econ.py"))
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

pub fn start_private_server(scratch: &Path, game_port: u16, econ_port: u16, extra_cfg: &str) -> PrivateServer {
    // The ports must be free: nothing else of ours may be answering there.
    UdpSocket::bind(("127.0.0.1", game_port)).unwrap_or_else(|e| panic!("UDP {game_port} is busy: {e}"));
    TcpListener::bind(("127.0.0.1", econ_port)).unwrap_or_else(|e| panic!("TCP {econ_port} is busy: {e}"));
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
        use ddai_os::private::OwnerOnly;
        let mut f = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .owner_only()
            .open(&secrets)
            .unwrap();
        writeln!(f, "ec_password \"{econ_password}\"").unwrap();
    }
    std::fs::write(
        scratch.join("server.cfg"),
        format!(
            "bindaddr 127.0.0.1\nsv_port {game_port}\nsv_register 0\nsv_ipv4only 1\n\
             sv_name \"aiddnet e2e 3.17 (private, 127.0.0.1 only)\"\nsv_map \"{MAP}\"\n\
             sv_max_clients 4\nsv_max_clients_per_ip 4\nsv_connlimit_time 0\nsv_test_cmds 0\nclear_votes\n\
             sv_high_bandwidth 0\nsv_tee_historian 0\nsv_pause_messages 1\nec_bindaddr 127.0.0.1\nec_port {econ_port}\nloglevel 0\n{extra_cfg}\n"
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

/// The bot's log, captured.
#[derive(Clone, Default)]
pub struct LogBuf(Arc<Mutex<Vec<u8>>>);

impl LogBuf {
    pub fn text(&self) -> String {
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

use ddai_client::{Client, ClientConfig};
use ddai_net::generated::objects::PlayerInput;

/// The scripted opponent: a client that pumps its events and sets a new input every 20 ms from a fixed timeline: short walks right and left (it
/// stays within a few tiles of where it spawns, out of the freeze tubes), a jump, a hook, a turning aim; it kills itself every 15 s (a fresh tee
/// at the spawn, so it is not frozen for the rest of the test after the bot has blocked it once).
pub fn spawn_opponent(
    server: SocketAddr,
    cache: PathBuf,
    stop: Arc<AtomicBool>,
    name: &str,
    margin_ms: Option<i32>,
) -> std::thread::JoinHandle<()> {
    let name = name.to_string();
    std::thread::spawn(move || {
        let client = Client::connect(
            server,
            ClientConfig {
                name: name.clone(),
                cache_dir: cache,
                // The margin is how long before a tick the server has the input: the lead of its pre-inputs.
                prediction_margin_ms: margin_ms.unwrap_or(ClientConfig::default().prediction_margin_ms),
                ..ClientConfig::default()
            },
        );
        let t0 = Instant::now();
        let mut last_kill = Instant::now();
        while !stop.load(Ordering::SeqCst) {
            let _ = client.recv_event(Duration::from_millis(20));
            let ms = t0.elapsed().as_millis() as i64;
            let input = scripted_input(ms);
            client.set_input(input);
            if last_kill.elapsed() > Duration::from_secs(15) {
                client.kill();
                last_kill = Instant::now();
            }
        }
    })
}

/// The scripted opponent's input at `ms` milliseconds into its timeline (the server counts a player as AFK until it plays, and a player that is AFK
/// neither gets nor sends pre-inputs: a test client has to move, like the bot does).
pub fn scripted_input(ms: i64) -> PlayerInput {
    // 8 steps of 250 ms: right, right, left, left, jump left, stand, hook right, left.
    let step = (ms / 250) % 8;
    let a = ms as f64 / 900.0;
    PlayerInput {
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
    }
}
