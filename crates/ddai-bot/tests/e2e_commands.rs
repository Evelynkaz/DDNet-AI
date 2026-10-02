//! End-to-end of the console commands and the clips against the real local DDNet 20.1 server
//! (`ddnet-local.service`, 127.0.0.1:8303) — task 4.3, acceptance criterion 6. `#[ignore]`d; run with
//!
//! ```text
//! DDAI_E2E=1 cargo test -p ddai-bot --test e2e_commands -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Loopback only, never chat (D-007). The server must be on `Copy Love Box` (its normal map; the test does
//! not change it). One bot of ours is driven **by lines on a pipe read by the real console thread** (the same
//! path as stdin: `ddai_bot::console`) while two scripted bots of ours play against it:
//!
//! - `hello there` (no prefix) is refused locally, `!wb off`, `!mode`, `!target <nick>`, `!clip`, `!brain
//!   scripted` / `!brain planner`, `!goto <x> <y>`, `!kill` (twice: the second is on cooldown), `!stats`, `!where`
//!   and finally `!quit` are answered and take effect;
//! - the outgoing-message audit of the bot has **no chat** at all and every `Cl_Kill` is the bot's own (the
//!   console's included);
//! - the clips it saved (manual by `!clip`, automatic by the incident scan) are replayed **offline** on
//!   `ddai-world`/`ddai-physics` from the map cache: the report says how many steps reproduce the recorded
//!   frames bit for bit and, for every divergence, the first one with its cause.
//!
//! The clips and the replay summary are kept under `~/aiddnet/data/logs/4.3/` (never in git).

use std::io::Write as _;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ddai_bot::brains::{BrainKind, BrainOptions};
use ddai_bot::clipper::ClipConfig;
use ddai_bot::command::CommandBus;
use ddai_bot::console;
use ddai_bot::nav_hooks::{NavConfig, NavHandle, WbMode};
use ddai_bot::runner::{RunReport, RunnerConfig, run};
use ddai_bot::{BotConfig, Mode, Relations};
use ddai_client::ClientConfig;
use ddai_clip::replay::{Mode as ReplayMode, replay};
use ddai_clip::{Clip, ClipEvent};

fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("aiddnet/data")
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

fn secs() -> u64 {
    std::env::var("DDAI_E2E_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150)
}

fn require_e2e() -> bool {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set DDAI_E2E=1");
        return false;
    }
    true
}

fn runner_cfg(name: &str, kind: BrainKind, mode: Mode, seed: u64, duration: Duration, nav: NavHandle) -> RunnerConfig {
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
        shutdown: Arc::new(AtomicBool::new(false)),
        nav: NavConfig {
            memory_dir: None,
            wb_mode: WbMode::Auto,
            ..NavConfig::default()
        },
        nav_handle: nav,
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

fn wait_for(deadline: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + deadline;
    while Instant::now() < end {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

/// Where the clips and the summary go: `~/aiddnet/data/logs/4.3/e2e-<unix seconds>/`.
fn out_dir() -> PathBuf {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let d = data_dir().join("logs").join("4.3").join(format!("e2e-{ts}"));
    std::fs::create_dir_all(&d).expect("the output directory");
    d
}

fn load_map(clip: &Clip) -> Arc<ddai_physics::map::MapData> {
    let cache = data_dir().join("maps").join("cache");
    let bytes = ddai_client::map_cache::read_cached(&cache, &clip.header.map_name, &clip.header.map_sha256)
        .unwrap_or_else(|| panic!("the map {} is not in the cache", clip.header.map_name));
    Arc::new(ddai_map::load_map(&bytes).expect("the cached map parses").data)
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

    fn has(&self, needle: &str) -> bool {
        self.all().contains(needle)
    }
}

#[test]
#[ignore = "runs against the real local ddnet-local.service; DDAI_E2E=1 and --ignored"]
fn console_commands_drive_the_bot_and_its_clips_replay_bit_for_bit() {
    if !require_e2e() {
        return;
    }
    let out = out_dir();
    let clips_dir = out.join("clips");
    let total = Duration::from_secs(secs());
    let nav = NavHandle::new();
    let said = Said::default();

    let (sender, inbox) = CommandBus::open();
    let (pipe_in, mut pipe_out) = std::io::pipe().expect("a pipe");
    let _console = console::spawn(sender, std::io::BufReader::new(pipe_in), said.printer()).expect("console thread");

    let mut cfg = runner_cfg("ddai-e2e-cmd", BrainKind::Planner, Mode::Fight, 31, total, nav.clone());
    cfg.bot.clips = ClipConfig {
        dir: Some(clips_dir.clone()),
        autoclip: true,
        async_save: true,
    };
    cfg.bot.console_names = true; // our own test bots' names, asserted below
    cfg.bot.settings_path = Some(out.join("settings.toml"));
    cfg.bot.relations_path = Some(out.join("relations.json"));
    cfg.commands = Some(inbox);
    cfg.console_out = Some(said.printer());
    let bot = spawn_bot(cfg);
    let mut others = Vec::new();
    for i in 0..2 {
        std::thread::sleep(Duration::from_millis(700));
        others.push(spawn_bot(runner_cfg(
            &format!("ddai-e2e-cs{i}"),
            BrainKind::Scripted,
            Mode::Fight,
            200 + i,
            total,
            NavHandle::new(),
        )));
    }

    let mut type_line = |line: &str, wait_ms: u64| {
        eprintln!("[stdin] {line}");
        writeln!(pipe_out, "{line}").expect("the console is listening");
        std::thread::sleep(Duration::from_millis(wait_ms));
    };

    // The bot is in the game once `!where` knows a tile.
    let mut spawned = false;
    for _ in 0..120 {
        type_line("!where", 500);
        if said.has("tile (") {
            spawned = true;
            break;
        }
    }
    assert!(spawned, "the bot never spawned");

    // Not a command: refused here, goes nowhere.
    type_line("hello there everyone", 400);
    assert!(said.has("never writes in the game chat"), "{}", said.all());
    type_line("!say hello", 400);
    assert!(said.has("say: the bot never writes in the game chat"), "{}", said.all());

    // On Copy Love Box the wayblock is on by default and the bot walks to its hall: turn it off, back to fight.
    type_line("!wb off", 1000);
    assert!(said.has("WB: off"), "{}", said.all());
    type_line("!go", 600);
    type_line("!mode", 400);
    assert!(said.has("mode: fight"), "{}", said.all());
    type_line("!target ddai-e2e-cs0", 800);
    assert!(said.has("target set to 'ddai-e2e-cs0'"), "{}", said.all());
    type_line("!clip ignored-target-test", 500);
    assert!(said.has("saved "), "{}", said.all());

    // Fight a while with the target set.
    std::thread::sleep(Duration::from_secs(15));
    type_line("!target -", 400);
    assert!(said.has("target cleared"));
    type_line("!clip after-target", 500);

    // Swap the brain live, and back.
    type_line("!brain scripted", 1500);
    assert!(said.has("brain: scripted"), "{}", said.all());
    std::thread::sleep(Duration::from_secs(10));
    type_line("!clip scripted-brain", 500);
    type_line("!brain planner", 1500);
    assert!(said.has("brain: planner"), "{}", said.all());

    // A walk by command: to the spawn tile of the map (a few tiles from where it stands).
    let before = nav.status().walks_ended;
    let map = {
        let cache = data_dir().join("maps").join("cache");
        let mut found = None;
        for e in std::fs::read_dir(&cache).expect("map cache").flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if n.starts_with("Copy Love Box_") && n.ends_with(".map") {
                found = Some(ddai_map::load_map(&std::fs::read(e.path()).unwrap()).unwrap().data);
            }
        }
        found.expect("the Copy Love Box map is in the cache")
    };
    let spawn = ddai_nav::route::spawn_tiles(&map)[0];
    let (tx, ty) = ((spawn.0 / 32.0) as i32, (spawn.1 / 32.0) as i32);
    type_line("!stop", 600);
    type_line(&format!("!goto {tx} {ty}"), 1200);
    assert!(said.has("goto:"), "{}", said.all());
    let arrived = wait_for(Duration::from_secs(90), || nav.status().walks_ended > before);
    eprintln!("[goto] ended={arrived}: {}", nav.status().last_walk);
    assert!(arrived, "the walk by command never ended");
    type_line("!go", 600);
    assert!(said.has("playing"));

    // The kill command and its cooldown. The walk just made may have used the cooldown itself (a route can
    // start with a respawn step): try until the console's kill is accepted.
    let mut killed = false;
    for _ in 0..40 {
        let before = said.all().matches("killing, respawning").count();
        type_line("!kill", 1500);
        if said.all().matches("killing, respawning").count() > before {
            killed = true;
            break;
        }
        assert!(said.has("reset is on cooldown"), "{}", said.all());
        std::thread::sleep(Duration::from_secs(4));
    }
    assert!(killed, "the console kill was never accepted: {}", said.all());
    let refused = said.all().matches("reset is on cooldown").count();
    type_line("!kill", 500);
    assert!(
        said.all().matches("reset is on cooldown").count() > refused,
        "{}",
        said.all()
    );
    type_line("!clip after-kill", 500);

    // More fighting, more clips: at least five manual ones in all.
    let started = Instant::now();
    let mut n = 0;
    while started.elapsed() + Duration::from_secs(25) < total.saturating_sub(Duration::from_secs(70)) && n < 8 {
        std::thread::sleep(Duration::from_secs(9));
        n += 1;
        type_line(&format!("!clip fight-{n}"), 500);
    }
    type_line("!stats", 600);
    assert!(said.has("snapshots"), "{}", said.all());
    type_line("!where", 400);
    type_line("!quit", 800);
    assert!(said.has("disconnecting"));

    let report = bot.join().expect("the bot thread");
    let others: Vec<RunReport> = others.into_iter().map(|h| h.join().expect("a scripted bot")).collect();

    // ---- the audit: no chat, every Cl_Kill is ours --------------------------------------------------
    assert_eq!(report.exit_code, 0, "gave up: {:?}", report.gave_up);
    for (label, (accepted, refused)) in &report.outgoing {
        assert!(
            ALLOWED_LABELS.contains(&label.as_str()),
            "unexpected outgoing message {label}"
        );
        assert_eq!(*refused, 0, "the allow-list refused a {label}");
        assert!(*accepted > 0);
    }
    assert!(
        !report.outgoing.keys().any(|k| k.contains("Say") || k.contains("Chat")),
        "chat on the wire"
    );
    let sent_kills = report.outgoing.get("Cl_Kill").map_or(0, |(ok, _)| *ok) as usize;
    assert_eq!(
        sent_kills,
        report.kill_ticks.len(),
        "Cl_Kill on the wire vs the bot's own decisions"
    );
    assert!(sent_kills >= 1, "the console kill went out");
    for r in &others {
        assert!(
            !r.outgoing.keys().any(|k| k.contains("Say")),
            "chat from a scripted bot"
        );
    }

    // ---- the clips ----------------------------------------------------------------------------------
    let mut files: Vec<PathBuf> = std::fs::read_dir(&clips_dir)
        .expect("the clip directory")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "clip"))
        .collect();
    files.sort();
    let manual = files.iter().filter(|p| name_of(p).starts_with("manual-")).count();
    let auto = files.len() - manual;
    eprintln!(
        "[clips] {} files: {manual} manual, {auto} automatic, in {}",
        files.len(),
        clips_dir.display()
    );
    assert!(manual >= 5, "only {manual} manual clips");

    let mut summary = Vec::new();
    let mut seen = std::collections::BTreeSet::<&'static str>::new();
    let (mut steps, mut exact, mut free_exact_clips) = (0usize, 0usize, 0usize);
    let (mut fresh_steps, mut fresh_exact, mut by_design) = (0usize, 0usize, 0usize);
    let (mut isolated_steps, mut isolated_exact) = (0usize, 0usize);
    let mut causes = std::collections::BTreeMap::<String, usize>::new();
    for p in &files {
        let clip = Clip::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        for e in clip.frames.iter().flat_map(|f| f.events.iter()) {
            seen.insert(match e {
                ClipEvent::HookAttach { .. } => "hook-attach",
                ClipEvent::HookRelease { .. } => "hook-release",
                ClipEvent::FreezeOnset { .. } => "freeze-onset",
                ClipEvent::Kill { .. } => "kill",
                ClipEvent::KillSent { .. } => "kill-sent",
                ClipEvent::HammerFire { .. } => "hammer-fire",
                ClipEvent::HammerHit { .. } => "hammer-hit",
                ClipEvent::Respawn { .. } => "respawn",
            });
        }
        let map = load_map(&clip);
        let resync = replay(&clip, Arc::clone(&map), ReplayMode::Resync);
        let free = replay(&clip, map, ReplayMode::FreeRun);
        steps += resync.steps;
        exact += resync.exact;
        fresh_steps += resync.fresh_steps;
        fresh_exact += resync.fresh_exact;
        by_design += resync.respawn_steps;
        isolated_steps += resync.isolated_steps;
        isolated_exact += resync.isolated_exact;
        // Every step that does not reproduce has a cause. What a clip cannot know is named (a respawn or a
        // teleporter exit the server chose, another tee's unrecorded inputs); what must never appear is a step
        // nothing explains: the server changing our state for no reason the clip shows.
        let unexplained: Vec<_> = resync
            .unexplained()
            .map(|d| (d.frame, d.tick, d.field, d.recorded, d.replayed))
            .collect();
        assert!(
            unexplained.is_empty(),
            "{}: steps the bot's own physics does not reproduce and nothing explains: {unexplained:?}",
            name_of(p)
        );
        assert!(
            free.first_divergence
                .as_ref()
                .is_none_or(|d| d.cause != ddai_clip::replay::Cause::ServerCorrection),
            "{}: the free run diverges unexplained: {:?}",
            name_of(p),
            free.first_divergence
        );
        if free.first_divergence.is_none() {
            free_exact_clips += 1;
        }
        for (k, n) in resync.by_cause() {
            *causes.entry(k.to_string()).or_default() += n;
        }
        eprintln!(
            "[replay] {}: {} frames, resync {}/{} exact steps ({} skipped at deaths), free-run {}; first divergence {:?}",
            name_of(p),
            clip.frames.len(),
            resync.exact,
            resync.steps,
            resync.skipped_deaths,
            if free.first_divergence.is_none() {
                "exact"
            } else {
                "diverges"
            },
            resync.first_divergence.as_ref().map(|d| (
                d.frame,
                d.tick,
                d.field,
                d.recorded,
                d.replayed,
                format!("{:?}", d.cause)
            )),
        );
        summary.push(serde_json::json!({
            "clip": name_of(p),
            "frames": clip.frames.len(),
            "resync_steps": resync.steps, "resync_exact": resync.exact, "skipped_deaths": resync.skipped_deaths,
            "free_run_exact": free.first_divergence.is_none(), "by_cause": resync.by_cause(),
            "respawn_steps": resync.respawn_steps, "fresh_steps": resync.fresh_steps, "fresh_exact": resync.fresh_exact,
            "first_divergence": resync.first_divergence.as_ref().map(|d| serde_json::json!({
                "frame": d.frame, "tick": d.tick, "field": d.field, "recorded": d.recorded, "replayed": d.replayed, "cause": format!("{:?}", d.cause), "fields": d.fields,
            })),
        }));
    }
    eprintln!("[replay] events seen across the clips: {seen:?}");
    eprintln!(
        "[replay] resync: {exact}/{steps} steps bit-exact ({:.2}%), {by_design} by design (respawn / teleport); fresh-core steps {fresh_exact}/{fresh_steps} exact; steps with nobody near {isolated_exact}/{isolated_steps} exact; \
         free-run reproduces {free_exact_clips}/{} clips; divergence causes {causes:?}",
        100.0 * exact as f64 / steps.max(1) as f64,
        files.len()
    );
    let _ = std::fs::write(
        out.join("e2e-replay.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "clips": summary, "steps": steps, "exact": exact, "by_design": by_design, "fresh_steps": fresh_steps, "fresh_exact": fresh_exact, "isolated_steps": isolated_steps, "isolated_exact": isolated_exact,
            "free_exact_clips": free_exact_clips, "causes": causes, "events_seen": seen, "manual": manual, "auto": auto,
        }))
        .unwrap_or_default(),
    );
    for must in ["hook-attach", "freeze-onset", "kill-sent"] {
        assert!(seen.contains(must), "no {must} event in any clip: {seen:?}");
    }
    assert!(steps > 1000, "only {steps} steps compared");
    // What the code under test controls: with nobody near, the bot's own physics is reproduced bit for bit (only a
    // respawn or a real teleport may differ); no step is a server correction nothing explains; every step that
    // does not reproduce has a named cause. The overall share depends on how much the scripted opponents brawl
    // with us (their inputs are not in the clip), so it is reported, not asserted.
    assert!(isolated_steps > 500, "only {isolated_steps} steps with nobody near");
    let designed = causes.get("respawn").copied().unwrap_or(0) + causes.get("teleport").copied().unwrap_or(0);
    assert!(
        isolated_exact + designed >= isolated_steps,
        "with nobody near, only respawns and teleports may differ: {isolated_exact}/{isolated_steps}, {causes:?}"
    );
    assert_eq!(causes.get("server-correction"), None, "{causes:?}");
    let explained: usize = causes.values().sum();
    assert_eq!(
        exact + explained,
        steps,
        "every step is exact or has a named cause: {causes:?}"
    );
    eprintln!(
        "[replay] overall {:.2}% of {steps} steps bit-exact (reported, not asserted)",
        100.0 * exact as f64 / steps.max(1) as f64
    );
    if auto == 0 {
        eprintln!("[clips] note: no incident reached an automatic clip in this run");
    }
}

fn name_of(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}
