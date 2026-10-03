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
use ddai_bot::nav_hooks::{NavCommand, NavConfig, NavHandle, WbMode};
use ddai_bot::runner::{RunReport, RunnerConfig, RunnerError};
use ddai_bot::{BotConfig, Mode, Relations};
use ddai_client::ClientConfig;

use crate::play_cmd::{Brain, PlayArgs};

/// The bot-specific flags of `ddnet-ai play`.
#[derive(Debug, Args, Default)]
pub struct BotOpts {
    /// `fight` (default), `passive` (wander only), `hold` (stand still) or `goto` (a walk begun by
    /// `--goto`/`--follow` ends back in the mode the bot started in).
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
    /// A trained fly bundle (task 7.4): `--brain fly` plays with its weights, `--brain hybrid` gets the fly as proposer
    /// (`hybrid:fly`). The web's «Муха» tab shows the fly's activity while the bot plays with either.
    #[arg(long)]
    pub fly_bundle: Option<PathBuf>,
    /// A **fixed** `cl_prediction_margin` in ms. Without it the margin is adaptive (task 4.1b, D-063):
    /// it starts at 10 and follows the `INPUTTIMING` feedback between 3 and 20 ms (rolling p1 of
    /// `time_left` kept at about 2 ms: lowered by 1 ms at a time, only when p1 is at least 4 ms and no
    /// input was late for 10 s; raised by 2 ms when two late inputs fall in the 5 s window, or when p1
    /// is below 2 ms over at least 100 samples — one late input never raises it; late inputs sent more
    /// than 18 ms too late are stalls no margin absorbs and are ignored).
    /// The margin is how early before a tick starts its input is due at the server; a smaller one
    /// moves the input slot later after each snapshot, giving a slow brain time to make it
    /// (`docs/formats.md` §21.6); too small and inputs arrive late (`INPUTTIMING` shows it in the
    /// report's `input_margin`).
    #[arg(long)]
    pub prediction_margin_ms: Option<i32>,
    /// Walk to this tile `X,Y` as soon as the tee has spawned (task 4.2); afterwards the bot returns to
    /// its mode. Crosses freeze tubes where it must (the Copy Love Box wayblock tubes).
    #[arg(long, value_parser = parse_pair, conflicts_with = "follow")]
    pub goto: Option<(i32, i32)>,
    /// Walk to the player with this client id and keep following them while they move (task 4.2).
    #[arg(long)]
    pub follow: Option<i32>,
    /// Wayblock on Copy Love Box: `auto` (default; the side with fewer players), `left`, `right`, `off`.
    #[arg(long, default_value = "auto")]
    pub wb: String,
    /// Strong mode: inside a wayblock hall the planner searches wider (`STRONG_WB`, more CPU).
    #[arg(long)]
    pub strong: bool,
    /// Never walk to where the game is when it is dull here.
    #[arg(long)]
    pub no_seek: bool,
    /// Directory of the freeze memories (one file per map, keyed by the map's sha256). Default
    /// `<data-dir>/bot/memory`.
    #[arg(long)]
    pub memory_dir: Option<PathBuf>,
    /// Do not read or write a freeze memory.
    #[arg(long)]
    pub no_memory: bool,
    /// `Cl_ShowDistance` half-extents `X,Y` (default 3000,2000 — D-007's replacement for /showall).
    #[arg(long, value_parser = parse_pair)]
    pub show_distance: Option<(i32, i32)>,
    /// Read commands from stdin (`!help` lists them; a line without `!` or `?` goes nowhere — the bot never
    /// writes in the game chat). On by default when stdin is a terminal.
    #[arg(long)]
    pub console: bool,
    /// Console replies name other players by their real nickname. Default: by tag (`c<id>-<hash>`), so the nicknames
    /// of others never reach a journal when the console runs under a unit.
    #[arg(long)]
    pub console_names: bool,
    /// Never read stdin, even from a terminal.
    #[arg(long, conflicts_with = "console")]
    pub no_console: bool,
    /// Where clips are written (default `<data-dir>/bot/clips`; never in git).
    #[arg(long)]
    pub clips_dir: Option<PathBuf>,
    /// Do not save incident and cross-fail clips by themselves (`!clip` still works).
    #[arg(long)]
    pub no_autoclip: bool,
    /// What `!brain`, `!wb`, `!low` and `!strong` remember (default `<data-dir>/bot/settings.toml`). The
    /// command line wins over the file; a file that does not parse is renamed `.bad-<ts>` and ignored.
    #[arg(long)]
    pub settings: Option<PathBuf>,
    /// Do not read or write a settings file.
    #[arg(long, conflicts_with = "settings")]
    pub no_settings: bool,
    /// The clan sent to the server (D-068: `Neuroset` by default; also `clan = "..."` in the settings file).
    #[arg(long)]
    pub clan: Option<String>,
    /// A fixed skin name (also `skin = "..."` in the settings file). Without it the bot picks a random stock skin at
    /// every start (D-068).
    #[arg(long)]
    pub skin: Option<String>,
    /// Makes the random skin pick deterministic (tests). Ignored when `--skin` is given.
    #[arg(long)]
    pub skin_seed: Option<u64>,
    /// Unix socket of the web control channel (task 5.6, D-070): the web unit's owner-only commands and the lists
    /// editor reach the bot through it. Default `<data-dir>/bot/control.sock` (mode 0600 in a 0700 directory).
    #[arg(long)]
    pub control: Option<PathBuf>,
    /// Do not open the web control socket.
    #[arg(long, conflicts_with = "control")]
    pub no_control: bool,
    /// The control channel's audit log (command tags, session tags, outcomes; no nicknames). Default
    /// `<data-dir>/logs/bot/control-audit.log`.
    #[arg(long)]
    pub control_audit: Option<PathBuf>,
}

/// Whether `--flag` (or `--flag=value`) was typed on the command line, as opposed to being a default: a
/// value in the settings file only fills what the command line left open.
fn flag_given(flag: &str) -> bool {
    std::env::args().any(|a| a == flag || a.strip_prefix(flag).is_some_and(|r| r.starts_with('=')))
}

fn parse_pair(s: &str) -> Result<(i32, i32), String> {
    let (a, b) = s.split_once(',').ok_or("expected X,Y")?;
    Ok((
        a.trim().parse().map_err(|e| format!("{e}"))?,
        b.trim().parse().map_err(|e| format!("{e}"))?,
    ))
}

/// `--duration`: `0` means no limit (a systemd unit runs the bot until it is stopped); before task 4.4 it meant "stop at once".
fn run_for(seconds: u64) -> Option<Duration> {
    (seconds > 0).then(|| Duration::from_secs(seconds))
}

/// The one-time startup warning when `MALLOC_MMAP_THRESHOLD_` is not set (Linux, glibc): without it the bot keeps ~160 MiB more
/// memory for good after a map change to a large map (task 4.4, E-009 F1). The variable cannot be set from inside the process
/// (no `unsafe`, no `mallopt`), so a manual run has to be told.
fn mmap_threshold_warning(value: Option<&std::ffi::OsStr>) -> Option<&'static str> {
    if !cfg!(target_os = "linux") || value.is_some_and(|v| !v.is_empty()) {
        return None;
    }
    Some(
        "MALLOC_MMAP_THRESHOLD_ is not set: after a map change to a large map glibc keeps ~160 MiB of the old map's memory for good; \
         start the bot with MALLOC_MMAP_THRESHOLD_=131072 (the systemd unit does; deploy/README.md)",
    )
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
pub fn run(args: &PlayArgs, data_dir: &Path, server: std::net::SocketAddr) -> ExitCode {
    let o = &args.bot_opts;
    if let Some(warning) = mmap_threshold_warning(std::env::var_os("MALLOC_MMAP_THRESHOLD_").as_deref()) {
        tracing::warn!("{warning}");
    }
    let Some(mode) = Mode::parse(&o.mode) else {
        eprintln!("unknown --mode {:?} (fight|passive|hold|goto)", o.mode);
        return ExitCode::FAILURE;
    };
    // The settings file (task 4.3): what `!brain`, `!wb`, `!low` and `!strong` remembered. The command line wins.
    let settings_path = (!o.no_settings).then(|| {
        o.settings
            .clone()
            .unwrap_or_else(|| data_dir.join("bot").join("settings.toml"))
    });
    let settings = match settings_path.as_deref().map(ddai_bot::settings::load) {
        Some(ddai_bot::settings::Loaded::Ok(s)) => s,
        Some(ddai_bot::settings::Loaded::Corrupt { moved_to, why }) => {
            eprintln!(
                "settings: ignored ({why}){}",
                moved_to.map_or(String::new(), |p| format!("; the file is kept as {}", p.display()))
            );
            ddai_bot::settings::Settings::default()
        }
        None => ddai_bot::settings::Settings::default(),
    };
    let relations_path = o
        .relations
        .clone()
        .or_else(|| settings.relations.clone())
        .unwrap_or_else(|| default_relations(data_dir));
    let relations = match Relations::load(&relations_path) {
        Ok(r) => r,
        Err(e) => {
            // A corrupt list must not silently become "no friends" (the bot would attack them).
            eprintln!("refusing to start: {e}");
            return ExitCode::FAILURE;
        }
    };
    let identity = match ddai_bot::identity::resolve(
        ddai_bot::identity::Overrides {
            clan: o.clan.as_deref(),
            skin: o.skin.as_deref(),
            skin_seed: o.skin_seed,
        },
        &settings,
    ) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("identity: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut client = ClientConfig {
        name: args.name.clone(),
        clan: identity.clan.clone(),
        skin: identity.skin.clone(),
        cache_dir: data_dir.join("maps").join("cache"),
        timeout: args
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(ClientConfig::default().timeout),
        ..ClientConfig::default()
    };
    match o.prediction_margin_ms {
        Some(m) => client.prediction_margin_ms = m,
        None => client.adaptive_margin = true,
    }
    if let Some(sd) = o.show_distance {
        client.show_distance = sd;
    }
    let kind = if flag_given("--brain") {
        kind_of(args)
    } else {
        settings
            .brain
            .as_deref()
            .and_then(BrainKind::parse)
            .unwrap_or_else(|| kind_of(args))
    };
    let low = settings.low.unwrap_or(false);
    let strong = o.strong || (!flag_given("--strong") && settings.strong.unwrap_or(false));
    let mut brain = BrainOptions {
        planner_budget_ms: o.planner_budget_ms,
        seed: args.seed,
        planner_preset: if low {
            ddai_planner::brains::PlannerPreset::Low
        } else {
            ddai_planner::brains::PlannerPreset::Normal
        },
        ..BrainOptions::default()
    };
    if let Some(p) = &o.fly_flyg {
        brain.fly_flyg = p.clone();
    }
    if let Some(p) = &o.fly_bundle {
        brain.fly_bundle = Some(p.clone());
    }
    if let Some(p) = &o.fly_config {
        brain.fly_config = p.clone();
    }
    let bot = BotConfig {
        brain: kind,
        mode,
        fixed_target: o.target.clone(),
        seed: args.seed,
        clips: ddai_bot::clipper::ClipConfig {
            dir: Some(
                o.clips_dir
                    .clone()
                    .unwrap_or_else(|| data_dir.join("bot").join("clips")),
            ),
            autoclip: !o.no_autoclip,
            async_save: true,
        },
        relations_path: Some(relations_path.clone()),
        settings_path: settings_path.clone(),
        low,
        strong,
        console_names: o.console_names,
        ..BotConfig::default()
    };
    let wb_text = if flag_given("--wb") {
        o.wb.clone()
    } else {
        settings.wb.clone().unwrap_or_else(|| o.wb.clone())
    };
    let Some(wb_mode) = WbMode::parse(&wb_text) else {
        eprintln!("unknown --wb {wb_text:?} (auto|left|right|off)");
        return ExitCode::FAILURE;
    };
    let nav = NavConfig {
        memory_dir: if o.no_memory {
            None
        } else {
            Some(
                o.memory_dir
                    .clone()
                    .unwrap_or_else(|| data_dir.join("bot").join("memory")),
            )
        },
        wb_mode,
        strong,
        seek: !o.no_seek,
        ..NavConfig::default()
    };
    let nav_handle = NavHandle::new();
    if let Some((tx, ty)) = o.goto {
        nav_handle.send(NavCommand::Goto {
            tx,
            ty,
            through_freeze: true,
        });
    }
    if let Some(id) = o.follow {
        nav_handle.follow(id);
    }
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
    // The command bus: the console (task 4.3) and the web control socket (task 5.6) both put `BotCommand`s on it and
    // the bot answers between two snapshots. Nothing typed or sent ever reaches the chat.
    let console = o.console || (!o.no_console && ddai_bot::console::stdin_is_terminal());
    let (bus_sender, inbox) = ddai_bot::command::CommandBus::open();
    let mut bus_used = false;
    let console_out = if console {
        let print = ddai_bot::console::stdout_printer();
        match ddai_bot::console::spawn(
            bus_sender.clone(),
            std::io::BufReader::new(std::io::stdin()),
            std::sync::Arc::clone(&print),
        ) {
            Ok(_) => {
                println!("console: type !help for the commands; a line without ! or ? is not sent anywhere");
                bus_used = true;
                Some(print)
            }
            Err(e) => {
                eprintln!("could not start the console: {e}");
                None
            }
        }
    } else {
        None
    };
    // Held until the bot has stopped: dropping it closes and removes the socket.
    let _control = if o.no_control {
        None
    } else {
        let socket = o
            .control
            .clone()
            .unwrap_or_else(|| data_dir.join("bot").join(ddai_bot::control::SOCKET_NAME));
        let audit_path = o
            .control_audit
            .clone()
            .unwrap_or_else(|| data_dir.join("logs").join("bot").join("control-audit.log"));
        let audit = match ddai_bot::control::FileAudit::open(&audit_path) {
            Ok(a) => std::sync::Arc::new(a),
            Err(e) => {
                eprintln!(
                    "refusing to start: cannot open the control audit log {}: {e}",
                    audit_path.display()
                );
                return ExitCode::FAILURE;
            }
        };
        match ddai_bot::control::ControlServer::start(&socket, bus_sender.clone(), audit) {
            Ok(server) => {
                bus_used = true;
                Some(server)
            }
            Err(e) => {
                eprintln!("refusing to start: the control socket {}: {e}", socket.display());
                return ExitCode::FAILURE;
            }
        }
    };
    let commands = bus_used.then_some(inbox);
    let cfg = RunnerConfig {
        server,
        client,
        bot,
        brain,
        relations,
        duration: run_for(args.duration),
        bridge_path,
        web_names: o.web_names,
        debug_names_log: o.debug_names.clone(),
        audit_outgoing: o.report.is_some(),
        shutdown,
        nav,
        nav_handle,
        commands,
        console_out,
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
            "vetoed_fires": r.stats.vetoed_fires,
            "predict_clamped": r.stats.predict_clamped,
            "guarded_inputs": r.stats.guarded_inputs,
            "deaths": r.stats.deaths,
        },
        "blocks": {"blocks": r.block_stats.blocks, "blocked_by": r.block_stats.blocked_by},
        "kill_ticks": r.kill_ticks,
        "kill_command_ticks": r.kill_command_ticks,
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
        "input_margin": r.margin.as_ref().map(|m| serde_json::json!({
            "count": m.count, "late": m.late_count, "late_fraction": m.late_fraction, "p50_ms": m.p50_ms, "p99_ms": m.p99_ms, "min_ms": m.min_ms,
            "margin_ms": m.margin_ms, "margin_changes": m.margin_changes, "adaptive": m.adaptive, "margin_stable_ms": m.margin_stable_ms, "margin_changes_last_30s": m.margin_changes_last_30s, "superseded_decisions": m.superseded_decisions, "time_at_cap_ms": m.time_at_margin_ms.last().copied().unwrap_or(0),
        })),
        "events": r.events.iter().map(|e| format!("{e:?}")).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "linux")]
    fn the_missing_malloc_variable_is_warned_about_once_and_a_set_one_is_not() {
        use std::ffi::OsStr;
        assert!(mmap_threshold_warning(None).is_some_and(|w| w.contains("131072")));
        assert!(mmap_threshold_warning(Some(OsStr::new(""))).is_some());
        assert_eq!(mmap_threshold_warning(Some(OsStr::new("131072"))), None);
    }

    #[test]
    fn duration_zero_means_no_limit_for_the_bot() {
        assert_eq!(run_for(0), None);
        assert_eq!(run_for(1), Some(Duration::from_secs(1)));
        assert_eq!(run_for(3600), Some(Duration::from_secs(3600)));
    }
}
