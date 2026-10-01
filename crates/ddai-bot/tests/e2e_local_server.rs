//! End-to-end against the real local DDNet 20.1 server (`ddnet-local.service`, 127.0.0.1:8303) —
//! task 4.1, acceptance criterion 6. `#[ignore]`d; run with
//!
//! ```text
//! DDAI_E2E=1 cargo test --release -p ddai-bot --test e2e_local_server -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Loopback only, never chat (D-007), at most 4 connections from this process (the driver's limit is
//! 5 per 20 s per address). The server must be on `Copy Love Box` (its normal map; the test does not
//! change it).
//!
//! Scenarios, each >= 2 minutes (`DDAI_E2E_SECS`, default 125):
//!
//! - **1v1**: our bot `--brain planner` against 1 of our own `--brain scripted` bots;
//! - **1v3**: against 3 of them.
//!
//! Asserted: no panic and no kick/ban (exit code 0 everywhere); the bots ran the whole time; blocks
//! are attributed (at least one block or "blocked by" among the bots of a scenario); the outgoing-message
//! audit of every bot has **no chat** (no `Cl_Say`, not even a refused one) and only labels from the allowed
//! set; every `Cl_Kill` is the bot's own unstick request (count matches) and kills are at least
//! `KILL_COOLDOWN_TICKS` apart; the planner bot's bridge feeds the web-side reader (`BotSource`) a
//! map, players (tags, never nicknames), frames and status. Latency percentiles (overhead vs brain) are
//! printed and saved to `~/aiddnet/data/logs/4.1/e2e-latency.json`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use ddai_bot::brains::{BrainKind, BrainOptions};
use ddai_bot::consts::KILL_COOLDOWN_TICKS;
use ddai_bot::runner::{RunReport, RunnerConfig, run};
use ddai_bot::{BotConfig, Mode, Relations};
use ddai_client::ClientConfig;

fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("aiddnet/data")
}

fn secs() -> u64 {
    std::env::var("DDAI_E2E_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(125)
}

fn config(name: &str, kind: BrainKind, seed: u64, bridge: Option<PathBuf>, duration: Duration) -> RunnerConfig {
    RunnerConfig {
        server: "127.0.0.1:8303".parse::<SocketAddr>().unwrap(),
        client: ClientConfig {
            name: name.to_string(),
            cache_dir: data_dir().join("maps").join("cache"),
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
        bridge_path: bridge,
        web_names: false,
        debug_names_log: None,
        audit_outgoing: true,
        shutdown: Arc::new(AtomicBool::new(false)),
    }
}

fn spawn_bot(cfg: RunnerConfig) -> std::thread::JoinHandle<RunReport> {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || run(cfg).expect("the bot starts"))
        .expect("thread")
}

/// Labels the client may ever send (`ddai_client::allowlist` plus the join-sequence ones): anything
/// else — above all `Cl_Say` — fails the audit.
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
    // Every kill is a Cl_Kill the bot itself requested, and they respect the cooldown.
    let sent_kills = r.outgoing.get("Cl_Kill").map_or(0, |(ok, _)| *ok) as usize;
    assert_eq!(
        sent_kills,
        r.kill_ticks.len(),
        "{name}: Cl_Kill messages vs the bot's own kill decisions"
    );
    for w in r.kill_ticks.windows(2) {
        assert!(
            w[1] - w[0] >= KILL_COOLDOWN_TICKS,
            "{name}: kills too close: {:?}",
            r.kill_ticks
        );
    }
    assert!(
        r.stats.snapshots > 25 * 60,
        "{name}: only {} snapshots",
        r.stats.snapshots
    );
}

fn latency_json(name: &str, r: &RunReport) -> serde_json::Value {
    let s = |x: ddai_bot::latency::Summary| serde_json::json!({"n": x.count, "p50_us": x.p50_us, "p90_us": x.p90_us, "p99_us": x.p99_us, "max_us": x.max_us});
    serde_json::json!({
        "bot": name,
        "snapshots": r.stats.snapshots,
        "decisions": r.stats.decisions,
        "brain_decisions": r.stats.brain_decisions,
        "wander_decisions": r.stats.wander_decisions,
        "collapsed": r.stats.collapsed,
        "hooks_fired": r.stats.hooks_fired,
        "hammer_fires": r.stats.hammer_fires,
        "self_kills": r.stats.self_kills,
        "blocks": r.block_stats.blocks,
        "blocked_by": r.block_stats.blocked_by,
        "deaths": r.stats.deaths,
        "total": s(r.latency.total.summary()),
        "brain": s(r.latency.brain.summary()),
        "overhead": s(r.latency.overhead.summary()),
        "pick": s(r.latency.pick.summary()),
        "seal_search": s(r.seal_times.summary()),
        "reach_search": s(r.reach_times.summary()),
        "queue": s(r.latency.queue.summary()),
        "wire": s(r.latency.wire.summary()),
        "slots": {"decisions": r.latency.slots.decisions, "first_slot": r.latency.slots.in_first_slot, "missed_first_slot": r.latency.slots.missed_first_slot, "as_predicted": r.latency.slots.as_predicted, "later_than_predicted": r.latency.slots.later_than_predicted, "earlier_than_predicted": r.latency.slots.earlier_than_predicted},
        "late_inputs": r.margin.map(|m| m.late_fraction),
    })
}

/// The web-side reader against the planner bot's bridge for ~10 s.
fn read_bridge_through_the_web_source(socket: PathBuf) -> (bool, usize, usize, bool, String) {
    use ddai_web::live::bot_source::BotSource;
    use ddai_web::live::map_resolve::MapCache;
    use ddai_web::live::source::{FrameSource, SourceEvent};
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async move {
        let cache = Arc::new(MapCache::new());
        let source = Box::new(BotSource::new(
            socket,
            vec![data_dir().join("maps").join("cache")],
            cache,
        ));
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let (_ctl, ctl_rx) = tokio::sync::mpsc::channel(4);
        let handle = source.spawn(tx, ctl_rx);
        let (mut map, mut frames, mut players, mut status) = (false, 0usize, 0usize, false);
        let mut names = String::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while let Ok(Some(ev)) = tokio::time::timeout_at(deadline, rx.recv()).await {
            match ev {
                SourceEvent::MapChanged(m) => map = m.width > 0 && m.height > 0,
                SourceEvent::Players(p) => {
                    players = p.len();
                    names = p.iter().map(|x| x.name.clone()).collect::<Vec<_>>().join(",");
                }
                SourceEvent::Frame(_) => frames += 1,
                SourceEvent::BotStatus(_) => status = true,
                _ => {}
            }
        }
        handle.abort();
        (map, frames, players, status, names)
    })
}

fn scenario(label: &str, opponents: usize, kind: BrainKind) -> Vec<(String, RunReport)> {
    let dur = Duration::from_secs(secs());
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("live.sock");
    let planner = spawn_bot(config(
        &format!("ddai-e2e-p{opponents}"),
        kind,
        11,
        Some(socket.clone()),
        dur,
    ));
    let mut others = Vec::new();
    for i in 0..opponents {
        std::thread::sleep(Duration::from_millis(700));
        others.push(spawn_bot(config(
            &format!("ddai-e2e-s{opponents}{i}"),
            BrainKind::Scripted,
            100 + i as u64,
            None,
            dur,
        )));
    }
    // While they play: the web-side reader on the planner bot's socket.
    std::thread::sleep(Duration::from_secs(20));
    let (map, frames, players, status, names) = read_bridge_through_the_web_source(socket);
    eprintln!("[{label}] web reader: map={map} frames={frames} players={players} status={status} names={names}");
    assert!(map, "{label}: the web reader resolved the bot's map from the cache");
    assert!(frames >= 50, "{label}: only {frames} frames in 10 s");
    assert!(players > opponents, "{label}: players {players}");
    assert!(status, "{label}: no status message");
    assert!(
        names
            .split(',')
            .all(|n| n.starts_with('c') && n.contains('-') && n.len() <= 14),
        "{label}: the bridge must carry tags, not nicknames: {names}"
    );

    let mut reports = vec![(
        format!("{}(+{opponents})", kind.name()),
        planner.join().expect("the planner bot thread"),
    )];
    for (i, h) in others.into_iter().enumerate() {
        reports.push((format!("scripted{i}"), h.join().expect("a scripted bot thread")));
    }
    reports
}

fn check(label: &str, reports: &[(String, RunReport)]) -> Vec<serde_json::Value> {
    let mut json = Vec::new();
    let mut attributed = 0;
    for (name, r) in reports {
        audit(name, r);
        attributed += r.block_stats.blocks + r.block_stats.blocked_by;
        eprintln!(
            "[{label}] {name}: snapshots={} decisions={} brain={} wander={} kills={} blocks={} blocked_by={} deaths={}\n{}",
            r.stats.snapshots,
            r.stats.decisions,
            r.stats.brain_decisions,
            r.stats.wander_decisions,
            r.kill_ticks.len(),
            r.block_stats.blocks,
            r.block_stats.blocked_by,
            r.stats.deaths,
            r.latency_text()
        );
        eprintln!(
            "[{label}] {name}: fresh seal searches {:?}; fresh reach floods {:?}",
            r.seal_times.summary(),
            r.reach_times.summary()
        );
        json.push(latency_json(name, r));
    }
    assert!(
        attributed >= 1,
        "{label}: no block was attributed in {secs} s",
        secs = secs()
    );
    // The planner bot decided with a brain, and its own overhead stays small.
    let planner = &reports[0].1;
    assert!(
        planner.stats.brain_decisions > 100,
        "{label}: the planner was hardly consulted"
    );
    // D-042 targets overhead p99 <= 0.5 ms. This VM is shared (load ~11 on 8 vCPUs while these runs
    // go, four bots in one process) and pauses threads for ~10 ms at a time (D-045), so the test
    // pins the body of the distribution and reports the tail: p50 < 1 ms, p90 < 3 ms.
    let overhead = planner.latency.overhead.summary();
    assert!(
        overhead.p50_us < 1_000 && overhead.p90_us < 3_000,
        "{label}: bot overhead {overhead:?}"
    );
    json
}

fn save(name: &str, value: &serde_json::Value) {
    let dir = data_dir().join("logs").join("4.1");
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join(name), serde_json::to_vec_pretty(value).unwrap_or_default());
}

#[test]
#[ignore = "runs against the real local ddnet-local.service; DDAI_E2E=1 and --ignored"]
fn the_focal_brain_against_one_and_three_scripted_bots_for_two_minutes() {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set DDAI_E2E=1");
        return;
    }
    // `DDAI_E2E_BRAINS=planner,hybrid` (default planner) picks the focal brain(s).
    let brains = std::env::var("DDAI_E2E_BRAINS").unwrap_or_else(|_| "planner".to_string());
    let mut all = serde_json::Map::new();
    for (n, name) in brains.split(',').enumerate() {
        let kind = BrainKind::parse(name.trim()).unwrap_or_else(|| panic!("unknown brain {name}"));
        if n > 0 {
            // Stay under the 5-connections-per-20-s limit between runs.
            std::thread::sleep(Duration::from_secs(25));
        }
        let one = scenario(&format!("{name} 1v1"), 1, kind);
        let j1 = check(&format!("{name} 1v1"), &one);
        std::thread::sleep(Duration::from_secs(25));
        let three = scenario(&format!("{name} 1v3"), 3, kind);
        let j3 = check(&format!("{name} 1v3"), &three);
        all.insert(name.trim().to_string(), serde_json::json!({"1v1": j1, "1v3": j3}));
    }
    save("e2e-latency.json", &serde_json::Value::Object(all));
}
