//! `ddnet-ai play --brain planner|scripted|hybrid|fly` (or `--bot`): the live bot (task 4.1,
//! `ddai-bot`) against a DDNet 20.x server. The same `play` command the demo brains use; this file is
//! the bot half. Loopback only until D-043's gate is met (`ClientConfig`'s live-servers allow-list
//! enforces it on every connect).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use clap::Args;
use ddai_bot::brains::{BrainKind, BrainOptions};
use ddai_bot::runner::{RunReport, RunnerConfig, RunnerError};
use ddai_bot::{BotConfig, Mode, Relations};
use ddai_client::ClientConfig;

use crate::play_cmd::{Brain, PlayArgs};

/// The bot-specific flags of `ddnet-ai play`.
#[derive(Debug, Args, Default)]
pub struct BotOpts {
    /// `fight` (default), `passive` (wander only), `hold` (stand still) or `goto` (stub until 4.2).
    #[arg(long, default_value = "fight")]
    pub mode: String,
    /// Fight only this player (exact name, folded); every other filter is ignored for them.
    #[arg(long)]
    pub target: Option<String>,
    /// Friend/war/ignore lists (never in git). Default `<data-dir>/bot/relations.json`.
    #[arg(long)]
    pub relations: Option<PathBuf>,
    /// Unix socket for the web unit's live view. Default `<data-dir>/bot/live.sock`.
    #[arg(long)]
    pub bridge: Option<PathBuf>,
    /// Do not open the live-view socket.
    #[arg(long)]
    pub no_bridge: bool,
    /// Send real nicknames to the web unit (default: salted-hash tags only).
    #[arg(long)]
    pub web_names: bool,
    /// Write real nicknames next to their tags into this local file (debug; off by default).
    #[arg(long)]
    pub debug_names: Option<PathBuf>,
    /// Write a JSON report (latency percentiles, counters, outgoing-message audit) at the end; also
    /// turns the outgoing-message audit on.
    #[arg(long)]
    pub report: Option<PathBuf>,
    /// The planner's wall budget per decision, ms (D-042).
    #[arg(long, default_value_t = 5.0)]
    pub planner_budget_ms: f64,
    /// `--brain fly`: the compiled graph.
    #[arg(long)]
    pub fly_flyg: Option<PathBuf>,
    /// `--brain fly`: the brain config (`configs/fly/S-brain.toml`).
    #[arg(long)]
    pub fly_config: Option<PathBuf>,
    /// `cl_prediction_margin` in ms (default 10): how early before a tick starts its input is due at
    /// the server. A smaller margin moves the input slot later after each snapshot, giving a slow
    /// brain time to make the slot (`docs/formats.md` §21.6); too small and inputs arrive late
    /// (`INPUTTIMING` shows it in the report's `input_margin`).
    #[arg(long)]
    pub prediction_margin_ms: Option<i32>,
    /// `Cl_ShowDistance` half-extents `X,Y` (default 3000,2000 — D-007's replacement for /showall).
    #[arg(long, value_parser = parse_pair)]
    pub show_distance: Option<(i32, i32)>,
}

fn parse_pair(s: &str) -> Result<(i32, i32), String> {
    let (a, b) = s.split_once(',').ok_or("expected X,Y")?;
    Ok((
        a.trim().parse().map_err(|e| format!("{e}"))?,
        b.trim().parse().map_err(|e| format!("{e}"))?,
    ))
}

fn kind_of(args: &PlayArgs) -> BrainKind {
    match args.brain {
        Brain::Planner => BrainKind::Planner,
        Brain::Scripted => BrainKind::Scripted,
        Brain::Hybrid => BrainKind::Hybrid,
        Brain::Fly => BrainKind::Fly,
        Brain::Idle | Brain::Circle | Brain::RandomScripted => BrainKind::Idle,
    }
}

fn default_relations(data_dir: &Path) -> PathBuf {
    data_dir.join("bot").join("relations.json")
}

/// Runs the bot; the process exit code is the runner's (0 ok, 3 kick/ban, 4 join failed).
pub fn run(args: &PlayArgs, data_dir: &Path) -> ExitCode {
    let o = &args.bot_opts;
    let Some(mode) = Mode::parse(&o.mode) else {
        eprintln!("unknown --mode {:?} (fight|passive|hold|goto)", o.mode);
        return ExitCode::FAILURE;
    };
    let relations_path = o.relations.clone().unwrap_or_else(|| default_relations(data_dir));
    let relations = match Relations::load(&relations_path) {
        Ok(r) => r,
        Err(e) => {
            // A corrupt list must not silently become "no friends" (the bot would attack them).
            eprintln!("refusing to start: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut client = ClientConfig {
        name: args.name.clone(),
        cache_dir: data_dir.join("maps").join("cache"),
        timeout: args
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(ClientConfig::default().timeout),
        ..ClientConfig::default()
    };
    if let Some(m) = o.prediction_margin_ms {
        client.prediction_margin_ms = m;
    }
    if let Some(sd) = o.show_distance {
        client.show_distance = sd;
    }
    let mut brain = BrainOptions {
        planner_budget_ms: o.planner_budget_ms,
        seed: args.seed,
        ..BrainOptions::default()
    };
    if let Some(p) = &o.fly_flyg {
        brain.fly_flyg = p.clone();
    }
    if let Some(p) = &o.fly_config {
        brain.fly_config = p.clone();
    }
    let bot = BotConfig {
        brain: kind_of(args),
        mode,
        fixed_target: o.target.clone(),
        seed: args.seed,
        ..BotConfig::default()
    };
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&shutdown);
        if let Err(e) = ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst)) {
            tracing::warn!(error = %e, "failed to install a SIGINT/SIGTERM handler");
        }
    }
    let bridge_path = if o.no_bridge {
        None
    } else {
        Some(
            o.bridge
                .clone()
                .unwrap_or_else(|| data_dir.join("bot").join("live.sock")),
        )
    };
    let cfg = RunnerConfig {
        server: args.server,
        client,
        bot,
        brain,
        relations,
        duration: Some(Duration::from_secs(args.duration)),
        bridge_path,
        web_names: o.web_names,
        debug_names_log: o.debug_names.clone(),
        audit_outgoing: o.report.is_some(),
        shutdown,
    };
    // A big stack: the planner's helpers copy whole worlds by value.
    let handle = std::thread::Builder::new()
        .name("ddai-bot".to_string())
        .stack_size(64 << 20)
        .spawn(move || ddai_bot::runner::run(cfg));
    let result = match handle {
        Ok(h) => h.join().unwrap_or_else(|_| {
            Err(RunnerError::Bridge {
                path: PathBuf::new(),
                source: std::io::Error::other("the bot thread panicked"),
            })
        }),
        Err(e) => {
            eprintln!("could not start the bot thread: {e}");
            return ExitCode::FAILURE;
        }
    };
    match result {
        Ok(report) => {
            println!("{}", summary(&report));
            if let Some(path) = &o.report
                && let Err(e) = std::fs::write(
                    path,
                    serde_json::to_vec_pretty(&report_json(&report)).unwrap_or_default(),
                )
            {
                eprintln!("could not write the report {}: {e}", path.display());
            }
            ExitCode::from(report.exit_code)
        }
        Err(e) => {
            eprintln!("the bot could not start: {e}");
            ExitCode::FAILURE
        }
    }
}

fn summary_of(name: &str, s: ddai_bot::latency::Summary) -> String {
    format!(
        "  {name:<9} n={:<6} p50={:>6} us  p90={:>6} us  p99={:>6} us  max={:>7} us",
        s.count, s.p50_us, s.p90_us, s.p99_us, s.max_us
    )
}

fn summary(r: &RunReport) -> String {
    let l = &r.latency;
    let mut out = format!(
        "bot ran {:.1}s, exit {}, {} decisions ({} collapsed), {} blocks, {} blocked-by, {} kills\n",
        r.elapsed.as_secs_f64(),
        r.exit_code,
        r.stats.decisions,
        r.stats.collapsed,
        r.block_stats.blocks,
        r.block_stats.blocked_by,
        r.kill_ticks.len()
    );
    out += &summary_of("total", l.total.summary());
    out.push('\n');
    out += &summary_of("brain", l.brain.summary());
    out.push('\n');
    out += &summary_of("overhead", l.overhead.summary());
    out.push('\n');
    out += &summary_of("pick", l.pick.summary());
    out.push('\n');
    out += &summary_of("queue", l.queue.summary());
    out.push('\n');
    out += &summary_of("wire", l.wire.summary());
    out
}

fn sum_json(s: ddai_bot::latency::Summary) -> serde_json::Value {
    serde_json::json!({"n": s.count, "p50_us": s.p50_us, "p90_us": s.p90_us, "p99_us": s.p99_us, "max_us": s.max_us})
}

/// The JSON report of a run (`--report`). No nicknames: events carry tags only.
pub fn report_json(r: &RunReport) -> serde_json::Value {
    let outgoing: serde_json::Map<String, serde_json::Value> = r
        .outgoing
        .iter()
        .map(|(k, (ok, refused))| (k.clone(), serde_json::json!({"accepted": ok, "refused": refused})))
        .collect();
    serde_json::json!({
        "exit_code": r.exit_code,
        "elapsed_s": r.elapsed.as_secs_f64(),
        "map": r.map_name,
        "gave_up": r.gave_up.as_ref().map(|(reason, cat)| serde_json::json!({"reason": reason, "category": format!("{cat:?}")})),
        "stats": {
            "snapshots": r.stats.snapshots,
            "collapsed": r.stats.collapsed,
            "decisions": r.stats.decisions,
            "brain_decisions": r.stats.brain_decisions,
            "wander_decisions": r.stats.wander_decisions,
            "idle_decisions": r.stats.idle_decisions,
            "hooks_fired": r.stats.hooks_fired,
            "hammer_fires": r.stats.hammer_fires,
            "self_kills": r.stats.self_kills,
            "vetoed_hooks": r.stats.vetoed_hooks,
            "guarded_inputs": r.stats.guarded_inputs,
            "deaths": r.stats.deaths,
        },
        "blocks": {"blocks": r.block_stats.blocks, "blocked_by": r.block_stats.blocked_by},
        "kill_ticks": r.kill_ticks,
        "outgoing_game_messages": outgoing,
        "latency_us": {
            "total": sum_json(r.latency.total.summary()),
            "brain": sum_json(r.latency.brain.summary()),
            "overhead": sum_json(r.latency.overhead.summary()),
            "pick": sum_json(r.latency.pick.summary()),
            "queue": sum_json(r.latency.queue.summary()),
            "wire": sum_json(r.latency.wire.summary()),
            "slots": {"decisions": r.latency.slots.decisions, "first_slot": r.latency.slots.in_first_slot, "missed_first_slot": r.latency.slots.missed_first_slot, "as_predicted": r.latency.slots.as_predicted, "later_than_predicted": r.latency.slots.later_than_predicted, "earlier_than_predicted": r.latency.slots.earlier_than_predicted},
        },
        "input_margin": r.margin.map(|m| serde_json::json!({
            "count": m.count, "late": m.late_count, "late_fraction": m.late_fraction, "p50_ms": m.p50_ms, "p99_ms": m.p99_ms, "min_ms": m.min_ms,
        })),
        "events": r.events.iter().map(|e| format!("{e:?}")).collect::<Vec<_>>(),
    })
}
