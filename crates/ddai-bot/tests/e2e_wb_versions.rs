//! End-to-end of the wayblock on a **private local** DDNet 20.1 server serving one version of Copy Love Box
//! (task 4.8): the Swarfey version (468x255, the trigger) or the original (387x250). `#[ignore]`d; run with
//!
//! ```text
//! DDAI_E2E=1 DDAI_E2E_SERVER=127.0.0.1:8373 DDAI_E2E_MAP="<the map file the server serves>" \
//!   [DDAI_E2E_SECS=240] [DDAI_E2E_OPPONENTS=3] [DDAI_E2E_WB=auto|left|right] [DDAI_E2E_OUT=<dir for the log and the clips>] \
//!   cargo test --release -p ddai-bot --test e2e_wb_versions -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Loopback only (the test refuses any other address and the shared server's ports). The focal bot (`--brain hybrid`,
//! WB `auto`, fight mode) plays against a few scripted opponents. Once a second the test samples the focal tee's
//! tile and asks, with the wayblock definition `wayblock_for` gives for the served map, whether it is in a hall and
//! how close to the first spot (the guard's spot). It must find a hall, get into it, and hold it (a share of the
//! samples after the arrival). A `!clip` of the held hall and the sampled tiles go to the output directory.
//!
//! The bot never writes chat: the audit of the outgoing messages is the one of `e2e_nav`.

use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ddai_bot::brains::{BrainKind, BrainOptions};
use ddai_bot::clipper::ClipConfig;
use ddai_bot::command::CommandBus;
use ddai_bot::nav_hooks::{NavConfig, NavHandle, WbMode};
use ddai_bot::runner::{RunReport, RunnerConfig, run};
use ddai_bot::{BotConfig, Mode, Relations};
use ddai_client::ClientConfig;
use ddai_nav::wayblock::{WbSide, on_wb_spot, wayblock_for};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;

fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("aiddnet/data")
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// The private server: loopback, and never the shared one (8303/8304).
fn server() -> SocketAddr {
    let addr: SocketAddr = std::env::var("DDAI_E2E_SERVER")
        .expect("set DDAI_E2E_SERVER=127.0.0.1:<port> of a private instance")
        .parse()
        .expect("DDAI_E2E_SERVER is not an address");
    assert!(
        addr.ip().is_loopback(),
        "{addr}: this test talks to a private local server only"
    );
    assert!(
        !matches!(addr.port(), 8303 | 8304),
        "{addr}: that is the shared server; use a private instance (tools/ddnet-server/private.sh)"
    );
    addr
}

const ALLOWED_LABELS: &[&str] = &[
    "Cl_StartInfo",
    "Cl_IsDDNetLegacy",
    "Cl_ShowDistance",
    "Cl_ShowOthers",
    "Cl_EnableSpectatorCount",
    "Cl_CameraInfo",
    "Cl_Kill",
    "Cl_SetTeam",
    "Cl_Say(/kill)",
];

fn config(
    name: &str,
    kind: BrainKind,
    seed: u64,
    duration: Duration,
    handle: NavHandle,
    wb_mode: WbMode,
    stop: Arc<AtomicBool>,
) -> RunnerConfig {
    RunnerConfig {
        server: server(),
        client: ClientConfig {
            name: name.to_string(),
            cache_dir: data_dir().join("maps").join("cache"),
            adaptive_margin: true,
            ..ClientConfig::default()
        },
        bot: BotConfig {
            brain: kind,
            mode: Mode::Fight,
            seed,
            ..BotConfig::default()
        },
        brain: BrainOptions {
            seed,
            ..BrainOptions::default()
        },
        relations: Relations::new(),
        duration: Some(duration),
        bridge_path: None,
        web_names: false,
        debug_names_log: None,
        audit_outgoing: true,
        shutdown: stop,
        nav: NavConfig {
            memory_dir: None,
            wb_mode,
            ..NavConfig::default()
        },
        nav_handle: handle,
        commands: None,
        console_out: None,
    }
}

fn spawn_bot(cfg: RunnerConfig) -> std::thread::JoinHandle<RunReport> {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || run(cfg).expect("the bot starts"))
        .expect("thread")
}

fn audit(name: &str, r: &RunReport) {
    assert_eq!(r.exit_code, 0, "{name}: exit code (gave up: {:?})", r.gave_up);
    for (label, (accepted, refused)) in &r.outgoing {
        assert!(
            ALLOWED_LABELS.contains(&label.as_str()),
            "{name}: unexpected outgoing message {label}"
        );
        assert_eq!(*refused, 0, "{name}: the allow-list refused a {label}");
        assert!(*accepted > 0);
    }
    assert!(
        !r.outgoing
            .keys()
            .any(|k| (k.contains("Say") || k.contains("Chat")) && k != ddai_client::session::SERVER_COMMAND_KILL_LABEL),
        "{name}: chat on the wire"
    );
}

/// The log writer of `tracing`: every line of the bot's log goes to a file.
#[derive(Clone)]
struct LogFile(Arc<Mutex<std::fs::File>>);

impl Write for LogFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().unwrap().flush()
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogFile {
    type Writer = LogFile;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[test]
#[ignore = "runs against a private local ddnet-server; DDAI_E2E=1, DDAI_E2E_SERVER, DDAI_E2E_MAP and --ignored"]
fn the_wb_is_reached_and_held_on_the_served_version_of_copy_love_box() {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set DDAI_E2E=1");
        return;
    }
    let out = PathBuf::from(
        std::env::var("DDAI_E2E_OUT").unwrap_or_else(|_| data_dir().join("logs/4.8-live").to_string_lossy().into()),
    );
    std::fs::create_dir_all(&out).expect("the output directory");
    let log = LogFile(Arc::new(Mutex::new(
        std::fs::File::create(out.join("bot.log")).expect("the log file"),
    )));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("nav=info,ddai_bot=info")),
        )
        .with_ansi(false)
        .with_writer(log)
        .try_init();

    // The map the server serves: the definition the bot will use is the one `wayblock_for` gives for it.
    let map_path =
        PathBuf::from(std::env::var("DDAI_E2E_MAP").expect("set DDAI_E2E_MAP to the map file the server serves"));
    let map = ddai_map::load_map(&std::fs::read(&map_path).expect("the map file"))
        .expect("map")
        .data;
    let (w, h) = (map.width, map.height);
    let world = PhysicsWorld::new(Arc::new(map), 1);
    let def = wayblock_for("Copy Love Box", Some(world.collision())).expect("the served map has a wayblock");
    eprintln!("[wb] the served map is {w}x{h}; the definition is {:?}", def.name);
    eprintln!("[wb] spots left {:?}, right {:?}", def.left.spots, def.right.spots);

    let total = Duration::from_secs(env_u64("DDAI_E2E_SECS", 240));
    let opponents = env_u64("DDAI_E2E_OPPONENTS", 3);
    let wb_mode = std::env::var("DDAI_E2E_WB").ok().map_or(WbMode::Auto, |m| {
        WbMode::parse(&m).expect("DDAI_E2E_WB: auto, left, right or off")
    });
    let handle = NavHandle::new();
    let stop = Arc::new(AtomicBool::new(false));
    let (sender, inbox) = CommandBus::open();
    let mut cfg = config(
        "ddai-e2e-wbv",
        BrainKind::Hybrid,
        41,
        total,
        handle.clone(),
        wb_mode,
        Arc::clone(&stop),
    );
    cfg.bot.clips = ClipConfig {
        dir: Some(out.join("clips")),
        autoclip: true,
        async_save: true,
    };
    cfg.commands = Some(inbox);
    let focal = spawn_bot(cfg);
    // The opponents hold the same hall (`DDAI_E2E_VISIT=0`: none of them holds a wayblock and they go where the game is):
    // the focal bot has intruders in its hall, three scripted ones walking the same way in.
    let visit = std::env::var("DDAI_E2E_VISIT").is_ok_and(|v| v != "0");
    let opponent_wb = match (visit, wb_mode) {
        (false, _) => WbMode::Off,
        (true, WbMode::Right) => WbMode::Right,
        (true, _) => WbMode::Left,
    };
    let mut others = Vec::new();
    for i in 0..opponents {
        std::thread::sleep(Duration::from_millis(700));
        others.push(spawn_bot(config(
            &format!("ddai-e2e-wbs{i}"),
            BrainKind::Scripted,
            100 + i,
            total,
            NavHandle::new(),
            opponent_wb,
            Arc::new(AtomicBool::new(false)),
        )));
    }

    // Once a second: where is the focal tee?
    let mut tiles: Vec<(u64, (i32, i32))> = Vec::new();
    let started = Instant::now();
    let mut first_hall: Option<(u64, WbSide)> = None;
    let mut first_spot: Option<u64> = None;
    let mut clipped = false;
    let mut held_at_clip = 0u32;
    while started.elapsed() < total.saturating_sub(Duration::from_secs(8)) {
        std::thread::sleep(Duration::from_secs(1));
        let t = started.elapsed().as_secs();
        let Some(tile) = handle.status().tile else { continue };
        tiles.push((t, tile));
        for s in [WbSide::Left, WbSide::Right] {
            if def.in_hall(s, tile.0, tile.1) && first_hall.is_none() {
                first_hall = Some((t, s));
                eprintln!("[wb] t={t}s: in the {} hall at {tile:?}", s.name());
            }
            if def.side(s).spots.iter().any(|&p| on_wb_spot(tile, p)) && first_spot.is_none() {
                first_spot = Some(t);
                eprintln!("[wb] t={t}s: on a spot of the {} hall at {tile:?}", s.name());
            }
        }
        // The clip of the held hall: 40 s after the first spot.
        if let Some(fs) = first_spot
            && !clipped
            && t >= fs + 40
        {
            clipped = true;
            held_at_clip = tiles.len() as u32;
            let r = sender.send_line("!clip wb-held", Duration::from_secs(5));
            eprintln!("[wb] t={t}s: !clip -> {r:?}");
        }
    }
    stop.store(true, Ordering::SeqCst);
    let report = focal.join().expect("the focal bot thread");
    for o in others {
        let r = o.join().expect("an opponent thread");
        audit("opponent", &r);
    }
    audit("focal", &report);

    // ---- the evidence -------------------------------------------------------------------------------------
    let mut text = String::new();
    for (t, (tx, ty)) in &tiles {
        let side = [WbSide::Left, WbSide::Right]
            .into_iter()
            .find(|&s| def.in_hall(s, *tx, *ty))
            .map_or("-", |s| s.name());
        let spot = [WbSide::Left, WbSide::Right]
            .into_iter()
            .find_map(|s| {
                def.side(s)
                    .spots
                    .iter()
                    .position(|&p| on_wb_spot((*tx, *ty), p))
                    .map(|i| format!("{}{i}", s.name()))
            })
            .unwrap_or_else(|| "-".to_string());
        text.push_str(&format!("{t}\t{tx}\t{ty}\thall={side}\tspot={spot}\n"));
    }
    std::fs::write(out.join("tiles.tsv"), &text).expect("the tile log");
    let (arrival, side) = first_hall.expect("the bot never got into a hall");
    let after: Vec<&(u64, (i32, i32))> = tiles.iter().filter(|(t, _)| *t >= arrival).collect();
    let in_hall = after
        .iter()
        .filter(|(_, (x, y))| {
            [WbSide::Left, WbSide::Right]
                .into_iter()
                .any(|s| def.in_hall(s, *x, *y))
        })
        .count();
    let on_spot = after
        .iter()
        .filter(|(_, p)| def.side(side).spots.iter().any(|&s| on_wb_spot(*p, s)))
        .count();
    let on_first = after
        .iter()
        .filter(|(_, p)| on_wb_spot(*p, def.side(side).spots[0]))
        .count();
    eprintln!(
        "[wb] version {w}x{h} ({}): arrived in the {} hall at {arrival}s, first spot touched at {:?}s; of {} samples since then {in_hall} in a hall ({:.0}%), {on_spot} on a spot of its side, {on_first} on the first spot (the guard's)",
        def.name,
        side.name(),
        first_spot,
        after.len(),
        100.0 * in_hall as f64 / after.len().max(1) as f64,
    );
    eprintln!(
        "[wb] kills {:?}; self kills {}; blocks {} blocked_by {} deaths {}; clip taken at sample {held_at_clip}",
        report.kill_ticks,
        report.stats.self_kills,
        report.block_stats.blocks,
        report.block_stats.blocked_by,
        report.stats.deaths
    );
    assert!(first_spot.is_some(), "the bot never stood on a spot");
    assert!(
        in_hall as f64 / after.len().max(1) as f64 >= 0.5,
        "the bot should hold a hall after it got in: {in_hall} of {}",
        after.len()
    );
}
