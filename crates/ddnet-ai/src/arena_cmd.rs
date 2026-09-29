//! `ddnet-ai arena`: the offline evaluation arena (task 8.1, `ddai-env`).
//!
//! * `arena run --config <run.toml> --out <dir>`: play every condition of a run config, write one
//!   JSONL per condition plus `summary.json` and the Russian `summary.md`.
//! * `arena list`: the arena definitions and their standing-slot counts.
//! * `arena scenarios`: the T1-T18 technique scenarios, success rate per brain.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Subcommand};
use ddai_env::arena::{Arena, load_arena_defs};
use ddai_env::config::{RunConfig, builtin_brain};
use ddai_env::output::{RunOptions, git_info, run_config};
use ddai_env::report::measure_stall_baseline;
use ddai_env::run::load_arenas;

#[derive(Debug, Args)]
pub struct ArenaArgs {
    #[command(subcommand)]
    pub command: ArenaCommand,
}

#[derive(Debug, Subcommand)]
pub enum ArenaCommand {
    /// Plays every condition of a run config (TOML) and writes JSONL + summary files.
    Run {
        #[arg(long)]
        config: PathBuf,
        /// Output directory (created): `<condition>.jsonl`, `summary.json`, `summary.md`.
        #[arg(long)]
        out: PathBuf,
        /// Worker threads for game batches (default: 6, the shared-VM cap, or fewer cores).
        #[arg(long)]
        threads: Option<usize>,
        /// Overrides the config's default games per condition (conditions with their own `games`
        /// keep it).
        #[arg(long)]
        games: Option<u32>,
        /// Overrides the config's base seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Only conditions whose name contains this text.
        #[arg(long)]
        filter: Option<String>,
        /// Arena definition directory (default: the config's `arenas_dir`, else `configs/arenas`).
        #[arg(long)]
        arenas_dir: Option<PathBuf>,
        /// Map directory (default: the config's `map_dir`, else `~/aiddnet/data/maps`).
        #[arg(long)]
        map_dir: Option<PathBuf>,
        /// Measure the machine's idle-loop stall baseline for this many ms before the run
        /// (D-041/D-045: published next to wall-clock decision times). 0 = skip.
        #[arg(long, default_value_t = 0)]
        stall_ms: u64,
    },
    /// Technique scenarios (T1-T18): success rate per brain, reference-solution check, traces.
    Scenarios {
        /// Directory of scenario TOML files.
        #[arg(long, default_value = "configs/scenarios")]
        dir: PathBuf,
        /// Scenario run config (brains to score); omit with `--check-reference`/`--trace`.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Output directory for `scenarios.json` / `scenarios.md`.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Only scenarios whose id is in this comma-separated list (e.g. `T1,T9`).
        #[arg(long)]
        only: Option<String>,
        /// Run each scenario's reference solution on its exact start and report pass/fail.
        #[arg(long)]
        check_reference: bool,
        /// Print the per-tick trace (exact start) of this scenario id with the given `--trace-brain`.
        #[arg(long)]
        trace: Option<String>,
        /// `reference` (default), `idle`, `scripted` or `planner`.
        #[arg(long, default_value = "reference")]
        trace_brain: String,
        #[arg(long)]
        map_dir: Option<PathBuf>,
    },
    /// Lists the arena definitions (name, split, map, standing slots).
    List {
        #[arg(long, default_value = "configs/arenas")]
        arenas_dir: PathBuf,
        #[arg(long)]
        map_dir: Option<PathBuf>,
    },
}

/// The commit the binary was built from (`build.rs`), and whether the record is trustworthy as
/// "exactly that commit": `true` (dirty) when the source tree has uncommitted changes, has moved
/// to another commit since the build, or is not available here to check.
fn source_git_info() -> (String, bool) {
    let built = env!("DDAI_BUILD_GIT_COMMIT").to_string();
    let (now, dirty) = git_info(Path::new(env!("CARGO_MANIFEST_DIR")));
    let same = now == built;
    (built, dirty || !same)
}

fn default_map_dir() -> PathBuf {
    match std::env::var_os("HOME") {
        Some(h) => PathBuf::from(h).join("aiddnet/data/maps"),
        None => PathBuf::from("maps"),
    }
}

fn expand_home(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(h)) => PathBuf::from(h).join(rest),
        _ => PathBuf::from(p),
    }
}

pub fn run(args: ArenaArgs) -> ExitCode {
    let result = match args.command {
        ArenaCommand::Run {
            config,
            out,
            threads,
            games,
            seed,
            filter,
            arenas_dir,
            map_dir,
            stall_ms,
        } => run_cmd(
            &config,
            &out,
            threads,
            games,
            seed,
            filter.as_deref(),
            arenas_dir,
            map_dir,
            stall_ms,
        ),
        ArenaCommand::Scenarios {
            dir,
            config,
            out,
            only,
            check_reference,
            trace,
            trace_brain,
            map_dir,
        } => scenarios_cmd(
            &dir,
            config.as_deref(),
            out.as_deref(),
            only.as_deref(),
            check_reference,
            trace.as_deref(),
            &trace_brain,
            map_dir,
        ),
        ArenaCommand::List { arenas_dir, map_dir } => list_cmd(&arenas_dir, map_dir),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_cmd(
    config: &Path,
    out: &Path,
    threads: Option<usize>,
    games: Option<u32>,
    seed: Option<u64>,
    filter: Option<&str>,
    arenas_dir: Option<PathBuf>,
    map_dir: Option<PathBuf>,
    stall_ms: u64,
) -> Result<(), String> {
    let text = std::fs::read_to_string(config).map_err(|e| format!("{}: {e}", config.display()))?;
    let mut cfg = RunConfig::parse(&text).map_err(|e| e.to_string())?;
    if let Some(g) = games {
        cfg.games = g;
    }
    if let Some(s) = seed {
        cfg.base_seed = s;
    }
    let arenas_dir = arenas_dir
        .or_else(|| cfg.arenas_dir.as_deref().map(expand_home))
        .unwrap_or_else(|| PathBuf::from("configs/arenas"));
    let map_dir = map_dir
        .or_else(|| cfg.map_dir.as_deref().map(expand_home))
        .unwrap_or_else(default_map_dir);
    let threads = threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(6)
    });
    let arenas = load_arenas(&cfg, &arenas_dir, &map_dir).map_err(|e| e.to_string())?;
    let stall_baseline = (stall_ms > 0).then(|| measure_stall_baseline(stall_ms));
    let opts = RunOptions {
        threads,
        out_dir: Some(out),
        filter,
        git: source_git_info(),
        stall_baseline,
    };
    let summary =
        run_config(&cfg, &arenas, &builtin_brain, &opts, &mut |line| eprintln!("{line}")).map_err(|e| e.to_string())?;
    println!("{}", ddai_env::report::markdown(&summary));
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn list_cmd(arenas_dir: &Path, map_dir: Option<PathBuf>) -> Result<(), String> {
    let defs = load_arena_defs(arenas_dir).map_err(|e| e.to_string())?;
    let map_dir = map_dir.unwrap_or_else(default_map_dir);
    for def in defs.values() {
        match Arena::build(def, &map_dir) {
            Ok(a) => println!(
                "{:<18} {:<8} {:>4} standing slots  map: {}{}",
                a.name,
                a.tag.label(),
                a.slot_count(),
                a.map_source,
                a.map_sha256
                    .as_deref()
                    .map_or(String::new(), |s| format!(" sha256 {s}"))
            ),
            Err(e) => println!("{:<18} {:<8} unavailable: {e}", def.name, def.tag.label()),
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scenarios_cmd(
    dir: &Path,
    config: Option<&Path>,
    out: Option<&Path>,
    only: Option<&str>,
    check_reference: bool,
    trace: Option<&str>,
    trace_brain: &str,
    map_dir: Option<PathBuf>,
) -> Result<(), String> {
    use ddai_env::config::PlayerSpec;
    use ddai_env::scenario::{
        ScenarioDef, ScenarioRunConfig, load_world, markdown, reference_brain, run_scenarios, run_trial,
    };

    let map_dir = map_dir.unwrap_or_else(default_map_dir);
    let mut defs = ScenarioDef::load_dir(dir).map_err(|e| e.to_string())?;
    if let Some(list) = only {
        let want: Vec<&str> = list.split(',').map(str::trim).collect();
        defs.retain(|d| want.contains(&d.id.as_str()));
    }
    if let Some(id) = trace {
        let def = defs
            .iter()
            .find(|d| d.id == id)
            .ok_or_else(|| format!("no scenario {id}"))?;
        let world = load_world(def, &map_dir).map_err(|e| e.to_string())?;
        let subject = if trace_brain == "reference" {
            reference_brain(def)
        } else {
            builtin_brain(&PlayerSpec::simple(trace_brain)).map_err(|e| e.to_string())?
        };
        let (subject, log) = ddai_env::brains::RecordingBrain::new(subject);
        let outcome = run_trial(def, &world, Box::new(subject), 0, 1, 0, false).map_err(|e| e.to_string())?;
        println!("{} {}: success = {}", def.id, def.name, outcome.success);
        for (t, snap) in outcome.trace.snaps.iter().enumerate() {
            let cells: Vec<String> = snap
                .iter()
                .map(|s| {
                    format!(
                        "({:7.1},{:7.1}){}{}{}",
                        s.pos[0],
                        s.pos[1],
                        if s.out { " OUT" } else { "" },
                        if s.hook_state == 5 { " H" } else { "" },
                        s.credit.map_or(String::new(), |c| format!(" by{c}"))
                    )
                })
                .collect();
            println!("t={t:4} {}", cells.join("  |  "));
        }
        println!(
            "\nsubject decisions as a reference timeline:\n{}",
            ddai_env::brains::actions_to_toml(&log.lock().map_err(|e| e.to_string())?)
        );
        return Ok(());
    }
    if check_reference {
        let mut failed = 0;
        for def in &defs {
            if def.reference.is_empty() {
                println!("{:<4} {:<48} no reference ({})", def.id, def.name, def.reference_note);
                continue;
            }
            let world = load_world(def, &map_dir).map_err(|e| e.to_string())?;
            let ok = run_trial(def, &world, reference_brain(def), 0, 1, 0, false)
                .map_err(|e| e.to_string())?
                .success;
            failed += usize::from(!ok);
            println!(
                "{:<4} {:<48} reference {}",
                def.id,
                def.name,
                if ok { "PASS" } else { "FAIL" }
            );
        }
        return if failed == 0 {
            Ok(())
        } else {
            Err(format!("{failed} reference solutions fail"))
        };
    }
    let config = config.ok_or("--config is required (or use --check-reference / --trace)")?;
    let text = std::fs::read_to_string(config).map_err(|e| format!("{}: {e}", config.display()))?;
    let cfg = ScenarioRunConfig::parse(&text).map_err(|e| e.to_string())?;
    let scores =
        run_scenarios(&defs, &cfg, &builtin_brain, &map_dir, &mut |l| eprintln!("{l}")).map_err(|e| e.to_string())?;
    let md = markdown(&scores, &defs);
    println!("{md}");
    if let Some(out) = out {
        std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
        std::fs::write(out.join("scenarios.md"), &md).map_err(|e| e.to_string())?;
        let json = serde_json::to_string_pretty(&scores).map_err(|e| e.to_string())?;
        std::fs::write(out.join("scenarios.json"), json).map_err(|e| e.to_string())?;
    }
    Ok(())
}
