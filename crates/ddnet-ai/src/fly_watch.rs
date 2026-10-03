//! `ddnet-ai fly watch --bundle <b> --arena <a>` (task 7.4): plays offline arena games with a trained fly and
//! serves its visualisation stream on a bridge socket, so the web's «Муха» tab shows the connectome brain working
//! without a game server. Run `ddnet-ai web --bot-socket <the same socket>` next to it.
//!
//! The socket is the live bot's bridge (`ddai_bot::bridge`, `docs/formats.md` §21.2): a viewer subscribes, the game
//! then hands it the focal player's frame after every tick the fly decided in, paced to real time (50 ticks a second,
//! the fly decides at 25 Hz). With nobody subscribed the fly builds no frame at all. Since task 5.7 the stream is also
//! the game itself, so the site's «Игра» tab can show the fly on its arena: `MAP` (when the arena is on a real map
//! file), `PLAYERS` (slots only: «муха», «соперник N», never a nickname), a `FRAME` after every tick, and a small
//! `STATUS` object once a second that says what the demo is (`docs/formats.md` §28). With `--pause-idle` the game
//! stops while no client has subscribed to anything (`Bridge::watched`), so an unwatched demo costs nothing.
//! Games follow one another until `--games` is reached or Ctrl-C.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::Args;
use ddai_bot::bridge::{Bridge, FrameChar, MapMessage, PlayerEntry, PlayersMessage};
use ddai_env::arena::{Arena, load_arena_defs};
use ddai_env::config::{PlayerSpec, Rules};
use ddai_env::game::{GameReport, Layout, play_game_watched};
use ddai_env::models::ModelBrains;
use ddai_env::sim::PlayerSetup;
use ddai_physics::vmath::round_to_int;
use ddai_physics::world::World;

#[derive(Debug, Args)]
pub struct WatchArgs {
    /// The trained fly (`.bundle`), e.g. `~/aiddnet/data/runs/E-008/<run>/checkpoints/final.bundle`.
    #[arg(long)]
    pub bundle: PathBuf,
    /// The arena to play on (`ddnet-ai arena list`), e.g. `pit` or `clb-left`.
    #[arg(long)]
    pub arena: String,
    /// The fly as the hybrid's proposer (`hybrid:fly`; the tab then shows how often its proposal is played) instead of playing alone.
    #[arg(long)]
    pub hybrid: bool,
    /// `.flyg` the bundle was trained against (default: the path stored in the bundle).
    #[arg(long)]
    pub flyg: Option<PathBuf>,
    /// Unix socket to serve the stream on (`ddnet-ai web --bot-socket <it>`). Default `~/aiddnet/data/bot/fly-watch.sock`.
    #[arg(long)]
    pub bridge: Option<PathBuf>,
    /// Arena definitions (default `configs/arenas`).
    #[arg(long)]
    pub arenas_dir: Option<PathBuf>,
    /// Map directory for arenas on a real map (default `~/aiddnet/data/maps`).
    #[arg(long)]
    pub map_dir: Option<PathBuf>,
    /// The opponents' brain (`scripted`, `planner`, `idle`, ...): one slot each, see `--opponents`.
    #[arg(long, default_value = "scripted")]
    pub opponent: String,
    /// How many opponents.
    #[arg(long, default_value_t = 1)]
    pub opponents: usize,
    /// Seed of the first game; game `g` uses `seed + g` and the same layouts as the arena batches.
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    /// Stop after this many games (0: until Ctrl-C).
    #[arg(long, default_value_t = 0)]
    pub games: u32,
    /// Playback speed: 1 is real time, 0 as fast as possible (nobody can watch that; for tests).
    #[arg(long, default_value_t = 1.0)]
    pub speed: f64,
    /// Pause between two games, ms.
    #[arg(long, default_value_t = 1500)]
    pub pause_ms: u64,
    /// Stop the game while no client has subscribed (to the fly stream or to the plain "the site is open", bridge
    /// `SUBSCRIBE` bits 0 and 1): an unwatched demo then costs no CPU, and carries on where it was when somebody looks.
    /// Needs a real-time pace (`--speed` > 0). Off by default (a headless run has no viewer).
    #[arg(long)]
    pub pause_idle: bool,
}

/// How often the demo describes itself to the clients, in ticks (once a second): a client that connects mid-game is
/// told within a second (the bridge greets late joiners with `MAP` and `PLAYERS` only).
const STATUS_EVERY_TICKS: i32 = 50;
/// How long the idle pause sleeps between two looks at the clients.
const IDLE_POLL: Duration = Duration::from_millis(100);

fn expand_home(p: &Path) -> PathBuf {
    match (p.strip_prefix("~"), std::env::var_os("HOME")) {
        (Ok(rest), Some(h)) => PathBuf::from(h).join(rest),
        _ => p.to_path_buf(),
    }
}

fn home_data(sub: &str) -> PathBuf {
    std::env::var_os("HOME").map_or_else(
        || PathBuf::from(sub),
        |h| PathBuf::from(h).join("aiddnet/data").join(sub),
    )
}

/// The focal and opponent specs of a watch (slot 0 is the fly).
fn specs(args: &WatchArgs, bundle: &Path) -> Vec<PlayerSpec> {
    let arg = if args.hybrid {
        format!("hybrid:fly:{}", bundle.display())
    } else {
        format!("fly:{}", bundle.display())
    };
    let mut focal = ddai_env::models::player_from_arg(&arg);
    focal.label = Some("fly".to_string());
    let mut out = vec![focal];
    for i in 0..args.opponents {
        let mut o = ddai_env::models::player_from_arg(&args.opponent);
        o.label = Some(format!("opponent {}", i + 1));
        out.push(o);
    }
    out
}

/// The layout of game `g`: sides swapped on odd games, spawn order reversed on `g % 4 >= 2` (as `run_condition`;
/// a wayblock arena never swaps).
fn layout_of(arena: &Arena, g: u32) -> Layout {
    if arena.wb.is_some() {
        Layout {
            swap: false,
            reverse_order: g % 2 == 1,
        }
    } else {
        Layout {
            swap: g % 2 == 1,
            reverse_order: (g / 2) % 2 == 1,
        }
    }
}

/// The demo's `STATUS` object (`docs/formats.md` §28): what the site shows next to «Показ». The bundle is named the way
/// `FLYMETA` names it (`<run>/<file>`, never a path).
fn demo_status(arena: &str, meta: &str) -> String {
    let bundle = serde_json::from_str::<serde_json::Value>(meta)
        .ok()
        .and_then(|m| m["bundle"]["name"].as_str().map(str::to_string))
        .unwrap_or_default();
    serde_json::json!({ "demo": true, "arena": arena, "bundle": bundle }).to_string()
}

/// The `MAP` message of an arena on a real map file, as the web resolves it: `<name>_<sha256>.map` in its maps
/// directories, so `name` is the file name without the hash suffix. `None` for a synthetic arena (no file to show).
fn map_message(arena: &Arena) -> Option<MapMessage> {
    let sha256 = arena.map_sha256.clone()?;
    Some(MapMessage {
        name: map_name(&arena.map_source, &sha256)?,
        sha256,
        w: arena.map.width,
        h: arena.map.height,
    })
}

/// `…/Copy Love Box_<sha256>.map` -> `Copy Love Box` (a file without the hash suffix keeps its whole stem).
fn map_name(source: &str, sha256: &str) -> Option<String> {
    let stem = Path::new(source).file_stem()?.to_str()?;
    Some(stem.strip_suffix(&format!("_{sha256}")).unwrap_or(stem).to_string())
}

/// The roster: slot 0 is the fly, the rest are its opponents. Slots, not people.
fn players_message(n: usize) -> PlayersMessage {
    PlayersMessage {
        own: 0,
        list: (0..n)
            .map(|i| PlayerEntry {
                id: i as i32,
                name: if i == 0 {
                    "муха".to_string()
                } else {
                    format!("соперник {i}")
                },
                team: 0,
            })
            .collect(),
    }
}

/// Every live tee of `world` as a bridge frame character (the same fields the live bot publishes).
fn frame_chars(world: &World<f32>, n: usize, out: &mut Vec<FrameChar>) {
    out.clear();
    for i in 0..n {
        let Some(core) = world.cores.get(i as u8) else { continue };
        let Some(character) = world.characters[i].as_ref().filter(|c| c.alive) else {
            continue;
        };
        out.push(FrameChar {
            id: i as u8,
            alive: true,
            frozen: character.freeze_time > 0,
            deep_frozen: core.deep_frozen,
            live_frozen: core.live_frozen,
            team: 0,
            weapon: u8::try_from(core.active_weapon.clamp(0, 255)).unwrap_or(0),
            x: round_to_int(core.pos.x),
            y: round_to_int(core.pos.y),
            aim_x: core.input.target_x,
            aim_y: core.input.target_y,
            hook_state: core.hook_state,
            hook_x: round_to_int(core.hook_pos.x),
            hook_y: round_to_int(core.hook_pos.y),
            hooked_id: core.hooked_player(),
        });
    }
}

pub fn run(args: WatchArgs) -> ExitCode {
    match watch(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn watch(args: &WatchArgs) -> Result<(), String> {
    if args.opponents == 0 || args.opponents + 1 > ddai_physics::core::MAX_CLIENTS {
        return Err("--opponents must be between 1 and 15".to_string());
    }
    if !(args.speed >= 0.0 && args.speed.is_finite()) {
        return Err("--speed must be >= 0".to_string());
    }
    if args.pause_idle && args.speed == 0.0 {
        return Err("--pause-idle needs a real-time pace (--speed > 0)".to_string());
    }
    let bundle = expand_home(&args.bundle);
    let arenas_dir = args
        .arenas_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("configs/arenas"));
    let map_dir = args.map_dir.clone().unwrap_or_else(|| home_data("maps"));
    let defs = load_arena_defs(&arenas_dir).map_err(|e| e.to_string())?;
    let def = defs.get(&args.arena).ok_or_else(|| {
        format!(
            "unknown arena {:?} (known: {})",
            args.arena,
            defs.keys().cloned().collect::<Vec<_>>().join(", ")
        )
    })?;
    let arena = Arena::build(def, &map_dir).map_err(|e| e.to_string())?;
    let models = ModelBrains::new(args.flyg.clone().map(|p| expand_home(&p)));
    let specs = specs(args, &bundle);
    let rules = Rules::default();

    // One brain up front: its stream layout is what the bridge announces to a viewer.
    let probe = models.make(&specs[0]).map_err(|e| e.to_string())?;
    let meta = probe
        .viz_meta()
        .ok_or_else(|| "this brain has no visualisation stream".to_string())?;
    drop(probe);

    let bridge_path = args
        .bridge
        .clone()
        .map_or_else(|| home_data("bot").join("fly-watch.sock"), |p| expand_home(&p));
    let mut bridge = Bridge::bind(&bridge_path).map_err(|e| format!("bridge {}: {e}", bridge_path.display()))?;
    bridge.set_fly_meta(Some(&meta));
    let status = demo_status(&args.arena, &meta);
    let roster = players_message(specs.len());
    if let Some(m) = map_message(&arena) {
        bridge.send_map(&m);
    }
    bridge.send_players(&roster);
    eprintln!(
        "serving the fly's stream on {}\nwatch it: ddnet-ai web --bot-socket {}",
        bridge_path.display(),
        bridge_path.display()
    );

    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&shutdown);
        if let Err(e) = ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst)) {
            eprintln!("warning: no Ctrl-C handler: {e}");
        }
    }

    let tick_period = Duration::from_millis(20);
    let mut chars: Vec<FrameChar> = Vec::with_capacity(ddai_physics::core::MAX_CLIENTS);
    let mut g = 0u32;
    while !shutdown.load(Ordering::SeqCst) && (args.games == 0 || g < args.games) {
        let players: Vec<PlayerSetup> = specs
            .iter()
            .map(|spec| {
                let brain = models.make(spec).map_err(|e| e.to_string())?;
                Ok(PlayerSetup {
                    brain,
                    lag: spec.lag,
                    label: spec.label.clone().unwrap_or_default(),
                })
            })
            .collect::<Result<_, String>>()?;
        let mut started = Instant::now();
        let mut first_tick = None;
        let report = play_game_watched(
            &arena,
            &rules,
            args.seed.wrapping_add(u64::from(g)),
            layout_of(&arena, g),
            players,
            &mut |sim, tick| {
                bridge.accept_pending();
                let utick = u32::try_from(tick.max(0)).unwrap_or(0);
                if bridge.fly_wanted()
                    && let Some(frame) = sim.players[0].brain.viz_frame(utick)
                {
                    bridge.send_fly(frame);
                }
                if bridge.clients() > 0 {
                    frame_chars(sim.pw.inner(), sim.players.len(), &mut chars);
                    bridge.send_frame(utick, &chars);
                    if tick % STATUS_EVERY_TICKS == 0 {
                        bridge.send_status_json(status.as_bytes());
                    }
                }
                if args.pause_idle && args.speed > 0.0 && !bridge.watched() {
                    // Nobody looks: the game waits where it is (the clock of its pace waits with it).
                    let paused_at = Instant::now();
                    while !bridge.watched() && !shutdown.load(Ordering::SeqCst) {
                        std::thread::sleep(IDLE_POLL);
                        bridge.accept_pending();
                    }
                    started += paused_at.elapsed();
                }
                if args.speed > 0.0 {
                    let t0 = *first_tick.get_or_insert(tick);
                    let due = started + tick_period.mul_f64(f64::from(tick - t0) / args.speed);
                    if let Some(wait) = due.checked_duration_since(Instant::now()) {
                        std::thread::sleep(wait);
                    }
                }
                !shutdown.load(Ordering::SeqCst)
            },
        )
        .map_err(|e| e.to_string())?;
        println!("{}", summary(g, &report));
        g += 1;
        // Between games the stream keeps its viewers (and takes new ones).
        let until = Instant::now() + Duration::from_millis(args.pause_ms);
        while Instant::now() < until && !shutdown.load(Ordering::SeqCst) {
            bridge.accept_pending();
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    Ok(())
}

/// One line per game: the result, its length and the fly's own decision time. No names exist here (slots only).
fn summary(g: u32, r: &GameReport) -> String {
    format!(
        "game {g}: {:?} at tick {} (blocks by the fly {}, self freezes {}), fly decisions p50 {} us p99 {} us",
        r.result,
        r.end_tick,
        r.blocks_by_a,
        r.a_self_freezes,
        r.timing.decide_us_p50[0].map_or("-".to_string(), |v| v.to_string()),
        r.timing.decide_us_p99[0].map_or("-".to_string(), |v| v.to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_env::sim::PlayerSetup;

    const SHA: &str = "6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25";

    #[test]
    fn the_map_name_is_the_file_name_without_the_hash_suffix() {
        let path = format!("/data/maps/copy-love-box/Copy Love Box_{SHA}.map");
        assert_eq!(map_name(&path, SHA).as_deref(), Some("Copy Love Box"));
        assert_eq!(
            map_name("/data/maps/x/Other.map", SHA).as_deref(),
            Some("Other"),
            "no suffix: the whole stem"
        );
        assert_eq!(map_name("", SHA), None);
    }

    #[test]
    fn the_status_names_the_arena_and_the_bundle_but_no_path() {
        let meta = r#"{"v":1,"bundle":{"name":"e005-fly/final","sha256":"00"}}"#;
        let v: serde_json::Value = serde_json::from_str(&demo_status("clb-left", meta)).unwrap();
        assert_eq!(
            v,
            serde_json::json!({"demo": true, "arena": "clb-left", "bundle": "e005-fly/final"})
        );
        // A brain without a bundle: an empty name, still a valid object.
        let v: serde_json::Value = serde_json::from_str(&demo_status("pit", r#"{"v":1,"bundle":null}"#)).unwrap();
        assert_eq!(v["bundle"], "");
    }

    #[test]
    fn the_roster_has_slots_only() {
        let p = players_message(3);
        let names: Vec<&str> = p.list.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["муха", "соперник 1", "соперник 2"]);
        assert_eq!(p.own, 0);
    }

    /// A whole scripted game on the synthetic `pit`: every tick the frame has both tees, inside the map, with the tick going up.
    #[test]
    fn frames_carry_every_live_tee_of_the_game() {
        use ddai_env::arena::{Arena, load_arena_defs};
        use ddai_env::config::{PlayerSpec, Rules};
        use ddai_env::game::{Layout, play_game_watched};
        use ddai_env::models::ModelBrains;

        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/arenas");
        let defs = load_arena_defs(&dir).unwrap();
        let arena = Arena::build(&defs["pit"], Path::new("/nonexistent")).unwrap();
        assert!(
            map_message(&arena).is_none(),
            "a synthetic arena has no map file to show"
        );
        let models = ModelBrains::new(None);
        let players: Vec<PlayerSetup> = (0..2)
            .map(|i| PlayerSetup {
                brain: models.make(&PlayerSpec::simple("scripted")).unwrap(),
                lag: 0,
                label: format!("p{i}"),
            })
            .collect();
        let (w, h) = (arena.map.width as i32 * 32, arena.map.height as i32 * 32);
        let mut chars = Vec::new();
        let mut seen = Vec::new();
        let mut moved = false;
        let mut first: Option<(i32, i32)> = None;
        let rules = Rules::default();
        play_game_watched(&arena, &rules, 1, Layout::default(), players, &mut |sim, tick| {
            frame_chars(sim.pw.inner(), sim.players.len(), &mut chars);
            if tick < 200 {
                assert_eq!(chars.len(), 2, "tick {tick}");
                for c in &chars {
                    assert!(
                        (0..w).contains(&c.x) && (0..h).contains(&c.y),
                        "tick {tick}: ({}, {})",
                        c.x,
                        c.y
                    );
                    assert!(c.alive);
                }
                let p = (chars[0].x, chars[0].y);
                moved |= first.is_some_and(|f| f != p);
                first.get_or_insert(p);
                seen.push(tick);
            }
            tick < 200
        })
        .unwrap();
        assert_eq!(seen.first(), Some(&1), "the first observed tick is 1");
        assert_eq!(seen.len(), 199);
        assert!(
            seen.windows(2).all(|p| p[1] == p[0] + 1),
            "one frame per tick, in order"
        );
        assert!(moved, "the tees move");
    }
}
