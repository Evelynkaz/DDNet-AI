//! `ddnet-ai fly watch --bundle <b> --arena <a>` (task 7.4): plays offline arena games with a trained fly and
//! serves its visualisation stream on a bridge socket, so the web's «Муха» tab shows the connectome brain working
//! without a game server. Run `ddnet-ai web --bot-socket <the same socket>` next to it.
//!
//! The socket is the live bot's bridge (`ddai_bot::bridge`, `docs/formats.md` §21.2) with only the fly part used: a
//! viewer subscribes, the game then hands it the focal player's frame after every tick the fly decided in, paced to
//! real time (50 ticks a second, the fly decides at 25 Hz). With nobody subscribed the fly builds no frame at all.
//! Games follow one another until `--games` is reached or Ctrl-C.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::Args;
use ddai_bot::bridge::Bridge;
use ddai_env::arena::{Arena, load_arena_defs};
use ddai_env::config::{PlayerSpec, Rules};
use ddai_env::game::{GameReport, Layout, play_game_observed};
use ddai_env::models::ModelBrains;
use ddai_env::sim::PlayerSetup;

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
}

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
        let started = Instant::now();
        let mut first_tick = None;
        let report = play_game_observed(
            &arena,
            &rules,
            args.seed.wrapping_add(u64::from(g)),
            layout_of(&arena, g),
            players,
            &mut |players, tick| {
                bridge.accept_pending();
                if bridge.fly_wanted()
                    && let Some(frame) = players[0].brain.viz_frame(u32::try_from(tick.max(0)).unwrap_or(0))
                {
                    bridge.send_fly(frame);
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
