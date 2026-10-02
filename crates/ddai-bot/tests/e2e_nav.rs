//! End-to-end navigation and wayblock against the real local DDNet 20.1 server
//! (`ddnet-local.service`, 127.0.0.1:8303) — task 4.2, acceptance criterion 8. `#[ignore]`d; run with
//!
//! ```text
//! DDAI_E2E=1 cargo test --release -p ddai-bot --test e2e_nav -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Loopback only, never chat (D-007). The server must be on `Copy Love Box` (its normal map; the test
//! does not change it).
//!
//! - **goto**: our bot (mode `hold`, so nothing else moves it) is sent to three fixed points of Copy Love
//!   Box — the left and the right WB spot (each behind a freeze tube) and a spawn tile — one after the
//!   other, through [`NavHandle`] (the API task 4.3's chat commands use). Every walk must end `arrived`
//!   with the tee within two tiles of the point, inside 120 s.
//! - **WB hold**: our planner bot (mode `fight`, wayblock `auto`) against one scripted bot for
//!   `DDAI_E2E_SECS` (default 125) seconds. It walks to its hall, holds it, and the share of the
//!   once-a-second samples inside a hall is reported (and has a floor).
//!
//! Both audit the outgoing messages: no chat (no `Cl_Say`, not even a refused one), only allowed labels,
//! every `Cl_Kill` is the bot's own request and kills are at least `KILL_COOLDOWN_TICKS` apart (the
//! navigation's respawn kills and the unstick's share one cooldown).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ddai_bot::brains::{BrainKind, BrainOptions};
use ddai_bot::consts::KILL_COOLDOWN_TICKS;
use ddai_bot::nav_hooks::{NavConfig, NavHandle, WbMode};
use ddai_bot::runner::{RunReport, RunnerConfig, run};
use ddai_bot::{BotConfig, Mode, Relations};
use ddai_client::ClientConfig;
use ddai_nav::wayblock::{WbSide, wayblocks};

fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("aiddnet/data")
}

fn secs() -> u64 {
    std::env::var("DDAI_E2E_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(125)
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
];

fn config(
    name: &str,
    kind: BrainKind,
    mode: Mode,
    seed: u64,
    duration: Duration,
    handle: NavHandle,
    stop: Arc<AtomicBool>,
) -> RunnerConfig {
    config_wb(name, kind, mode, seed, duration, handle, stop, WbMode::Auto)
}

#[allow(clippy::too_many_arguments)]
fn config_wb(
    name: &str,
    kind: BrainKind,
    mode: Mode,
    seed: u64,
    duration: Duration,
    handle: NavHandle,
    stop: Arc<AtomicBool>,
    wb_mode: WbMode,
) -> RunnerConfig {
    RunnerConfig {
        server: "127.0.0.1:8303".parse::<SocketAddr>().unwrap(),
        client: ClientConfig {
            name: name.to_string(),
            cache_dir: data_dir().join("maps").join("cache"),
            adaptive_margin: true,
            ..ClientConfig::default()
        },
        bot: BotConfig {
            brain: kind,
            mode,
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
            // The e2e must not touch the real memory files.
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
    assert!(r.gave_up.is_none(), "{name}: the driver gave up: {:?}", r.gave_up);
    for (label, (accepted, refused)) in &r.outgoing {
        assert!(
            ALLOWED_LABELS.contains(&label.as_str()),
            "{name}: unexpected outgoing message {label}"
        );
        assert_eq!(*refused, 0, "{name}: the allow-list refused a {label}");
        assert!(*accepted > 0);
    }
    assert!(
        !r.outgoing.keys().any(|k| k.contains("Say") || k.contains("Chat")),
        "{name}: chat on the wire"
    );
    let sent_kills = r.outgoing.get("Cl_Kill").map_or(0, |(ok, _)| *ok) as usize;
    assert_eq!(
        sent_kills,
        r.kill_ticks.len(),
        "{name}: Cl_Kill messages vs the bot's own kill decisions"
    );
    for w in r.kill_ticks.windows(2) {
        assert!(
            w[1] - w[0] >= KILL_COOLDOWN_TICKS.min(ddai_bot::consts::WB_KILL_COOLDOWN_TICKS),
            "{name}: kills too close: {:?}",
            r.kill_ticks
        );
    }
}

fn init_log() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
}

fn require_e2e() -> bool {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set DDAI_E2E=1");
        return false;
    }
    init_log();
    true
}

fn wait_for(deadline: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + deadline;
    while Instant::now() < end {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

#[test]
#[ignore = "runs against the real local ddnet-local.service; DDAI_E2E=1 and --ignored"]
fn goto_across_copy_love_box_to_three_fixed_points_arrives_every_time() {
    if !require_e2e() {
        return;
    }
    let clb = wayblocks()
        .into_iter()
        .find(|d| d.name == "Copy Love Box")
        .expect("CLB");
    let map_path = data_dir()
        .join("maps/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map");
    let map = ddai_map::load_map(&std::fs::read(&map_path).expect("the CLB map file"))
        .expect("map")
        .data;
    let spawn = ddai_nav::route::spawn_tiles(&map)[0];
    let points = [
        ("left WB spot", clb.left.spots[0]),
        ("right WB spot", clb.right.spots[0]),
        ("spawn", ((spawn.0 / 32.0) as i32, (spawn.1 / 32.0) as i32)),
    ];
    let handle = NavHandle::new();
    let stop = Arc::new(AtomicBool::new(false));
    let bot = spawn_bot(config(
        "ddai-e2e-nav",
        BrainKind::Idle,
        Mode::Hold,
        21,
        Duration::from_secs(900),
        handle.clone(),
        Arc::clone(&stop),
    ));
    assert!(
        wait_for(Duration::from_secs(60), || handle.status().tile.is_some()),
        "the bot never spawned"
    );
    let mut log = Vec::new();
    for (name, (tx, ty)) in points {
        let before = handle.status().walks_ended;
        let t0 = Instant::now();
        handle.goto_tile(tx, ty);
        let ended = wait_for(Duration::from_secs(120), || handle.status().walks_ended > before);
        let st = handle.status();
        let (px, py) = st.tile.unwrap_or((-99, -99));
        eprintln!(
            "[goto] {name} ({tx},{ty}): ended={ended} in {:.1}s, tile ({px},{py}), {} -- nav kills so far {}",
            t0.elapsed().as_secs_f64(),
            st.last_walk,
            st.nav_kills
        );
        log.push(format!(
            "{name}: {} in {:.1}s",
            st.last_walk,
            t0.elapsed().as_secs_f64()
        ));
        assert!(ended, "{name}: the walk never ended");
        assert!(st.last_walk.starts_with("arrived"), "{name}: {}", st.last_walk);
        assert!(
            (px - tx).abs() <= 2 && (py - ty).abs() <= 2,
            "{name}: stopped at ({px},{py}), wanted ({tx},{ty})"
        );
        // Let the bot settle between walks.
        std::thread::sleep(Duration::from_secs(1));
    }
    stop.store(true, Ordering::SeqCst);
    let report = bot.join().expect("the bot thread");
    audit("goto", &report);
    eprintln!("[goto] kills {:?}; walks: {log:?}", report.kill_ticks);
}

#[test]
#[ignore = "runs against the real local ddnet-local.service; DDAI_E2E=1 and --ignored"]
fn wb_hold_against_one_scripted_intruder_for_two_minutes() {
    if !require_e2e() {
        return;
    }
    let clb = wayblocks()
        .into_iter()
        .find(|d| d.name == "Copy Love Box")
        .expect("CLB");
    let dur = Duration::from_secs(secs());
    let handle = NavHandle::new();
    let stop = Arc::new(AtomicBool::new(false));
    let focal = spawn_bot(config(
        "ddai-e2e-wb",
        BrainKind::Planner,
        Mode::Fight,
        31,
        dur,
        handle.clone(),
        Arc::clone(&stop),
    ));
    std::thread::sleep(Duration::from_millis(700));
    // The intruder does not hold a wayblock of its own: it goes where the game is.
    let intruder = spawn_bot(config_wb(
        "ddai-e2e-wb-s",
        BrainKind::Scripted,
        Mode::Fight,
        32,
        dur,
        NavHandle::new(),
        Arc::new(AtomicBool::new(false)),
        WbMode::Off,
    ));
    // Sample the focal tee once a second.
    let (mut samples, mut in_hall) = (0u32, 0u32);
    let mut halls = [0u32; 2];
    let started = Instant::now();
    while started.elapsed() < dur {
        std::thread::sleep(Duration::from_secs(1));
        if let Some((tx, ty)) = handle.status().tile {
            samples += 1;
            for (i, s) in [WbSide::Left, WbSide::Right].into_iter().enumerate() {
                if clb.in_hall(s, tx, ty) {
                    halls[i] += 1;
                    in_hall += 1;
                    break;
                }
            }
        }
    }
    let r = focal.join().expect("the focal bot thread");
    let ri = intruder.join().expect("the intruder thread");
    audit("wb focal", &r);
    audit("wb intruder", &ri);
    for e in r.events.iter().filter(|e| {
        matches!(
            e,
            ddai_bot::BotEvent::Killed { .. } | ddai_bot::BotEvent::Respawned { .. }
        )
    }) {
        eprintln!("[wb] event {e:?}");
    }
    eprintln!("[wb] latency of the focal bot\n{}", r.latency_text());
    eprintln!(
        "[wb] fresh reach searches {:?}; seal searches {:?}",
        r.reach_times.summary(),
        r.seal_times.summary()
    );
    let st = handle.status();
    eprintln!(
        "[wb] {samples} samples, {in_hall} in a hall (left {}, right {}) = {:.0}%; walks ended {}, last: {}; wb: {}; nav kills {}; self kills {}; blocks {} blocked_by {} deaths {}",
        halls[0],
        halls[1],
        100.0 * f64::from(in_hall) / f64::from(samples.max(1)),
        st.walks_ended,
        st.last_walk,
        st.wb,
        st.nav_kills,
        r.stats.self_kills,
        r.block_stats.blocks,
        r.block_stats.blocked_by,
        r.stats.deaths
    );
    assert!(
        samples as u64 >= secs() / 2,
        "the bot hardly had a tee: {samples} samples"
    );
    assert!(
        f64::from(in_hall) / f64::from(samples) >= 0.4,
        "the bot should spend most of the two minutes in a hall: {in_hall} of {samples}"
    );
}
