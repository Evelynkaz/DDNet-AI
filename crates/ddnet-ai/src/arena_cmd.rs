//! `ddnet-ai arena`: the offline evaluation arena (task 8.1, `ddai-env`).
//!
//! * `arena run --config <run.toml> --out <dir>`: play every condition of a run config, write one
//!   JSONL per condition plus `summary.json` and the Russian `summary.md`.
//! * `arena list`: the arena definitions and their standing-slot counts.
//! * `arena scenarios`: the T1-T18 technique scenarios, success rate per brain.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Subcommand};
use ddai_env::EnvError;
use ddai_env::arena::{Arena, load_arena_defs};
use ddai_env::config::{BrainFactory, HybridSpec, PlayerSpec, RunConfig, builtin_brain, hybrid_config};
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
        /// With `--trace`: play trial N with its start-state jitter (default: the exact start).
        #[arg(long)]
        trace_trial: Option<u32>,
        /// `reference` (default), `idle`, `scripted`, `planner` or `hybrid` (`-fixed` suffix: the
        /// deterministic mode).
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

/// The arena's brain factory: the built-in brains plus `hybrid` with the fly as proposer (which
/// needs the model file, so it lives here and not in `ddai-env`).
fn arena_brain(spec: &PlayerSpec) -> Result<Box<dyn ddai_brain::Brain>, EnvError> {
    let fly = spec.hybrid.as_ref().is_some_and(|h| h.proposer_name() == "fly");
    if spec.brain != "hybrid" || !fly {
        return builtin_brain(spec);
    }
    let h: &HybridSpec = spec.hybrid.as_ref().expect("checked above");
    let flyg = spec
        .model
        .as_deref()
        .ok_or_else(|| EnvError::new("hybrid with proposer = \"fly\" needs model = <.flyg path>"))?;
    let flyg = expand_home(flyg);
    let cfg_path = h.fly_config.as_deref().map_or_else(
        || {
            let m = flyg.file_name().is_some_and(|n| n.to_string_lossy().contains("-M-"));
            PathBuf::from(if m {
                "configs/fly/M-brain.toml"
            } else {
                "configs/fly/S-brain.toml"
            })
        },
        expand_home,
    );
    let (cfg, clock) = hybrid_config(spec)?;
    let seed = 1;
    let brain = ddai_fly::proposer::untrained_fly_brain(&flyg, &cfg_path, seed).map_err(EnvError::new)?;
    let proposer = Box::new(ddai_fly::proposer::FlyProposer::new(brain, seed));
    let hybrid = ddai_planner::hybrid::HybridBrain::new(cfg, clock, proposer).map_err(EnvError::new)?;
    Ok(Box::new(hybrid))
}

const _: fn() = || {
    let _: &BrainFactory = &arena_brain;
};

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
            trace_trial,
            trace_brain,
            map_dir,
        } => scenarios_cmd(
            &dir,
            config.as_deref(),
            out.as_deref(),
            only.as_deref(),
            check_reference,
            trace.as_deref(),
            trace_trial,
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
        run_config(&cfg, &arenas, &arena_brain, &opts, &mut |line| eprintln!("{line}")).map_err(|e| e.to_string())?;
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
    trace_trial: Option<u32>,
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
            // `planner-fixed` / `hybrid-fixed`: the deterministic modes; a bare name is the default.
            let (name, fixed) = trace_brain
                .strip_suffix("-fixed")
                .map_or((trace_brain, false), |n| (n, true));
            // `hybrid-work`: the hybrid's 4 ms deadline on the work clock.
            let (name, work) = name.strip_suffix("-work").map_or((name, false), |n| (n, true));
            let mut spec = PlayerSpec::simple(name);
            if fixed {
                spec.mode = Some("fixed".to_string());
            }
            if work {
                spec.mode = Some("deadline".to_string());
                spec.clock = Some("work".to_string());
                spec.budget_ms = Some(4.0);
            }
            if name == "hybrid" {
                // The trace is a diagnostic: show the candidates and their scores.
                spec.hybrid = Some(HybridSpec {
                    debug_dump: Some(true),
                    ..HybridSpec::default()
                });
            }
            arena_brain(&spec).map_err(|e| e.to_string())?
        };
        let (subject, log, tlog) = ddai_env::brains::RecordingBrain::with_telemetry(subject);
        let outcome = run_trial(
            def,
            &world,
            Box::new(subject),
            0,
            1,
            trace_trial.unwrap_or(0),
            trace_trial.is_some(),
        )
        .map_err(|e| e.to_string())?;
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
        for (tick, json) in tlog.lock().map_err(|e| e.to_string())?.iter() {
            // Only brains with structured telemetry (the hybrid) get a line per decision.
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(json)
                && let Some(last) = v.get("last").filter(|l| l.is_object())
            {
                if let Some(dump) = last.get("dump").and_then(|d| d.as_array()) {
                    for d in dump.iter().take(6) {
                        println!(
                            "        {:<32} {:>8} {} {}",
                            d["src"].as_str().unwrap_or("?"),
                            d["score"],
                            d["combos"],
                            d["step0"].as_str().unwrap_or("")
                        );
                    }
                }
                println!(
                    "t={tick:4} chose {:<32} danger={:<22} best={} robust={} eval={} unsafe={} shielded={}",
                    last["chosen"].as_str().unwrap_or("?"),
                    last["danger"].as_str().unwrap_or(""),
                    last["best_score"],
                    last["robust"],
                    last["evaluated"],
                    last["unsafe"],
                    last["shielded"],
                );
            }
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
        run_scenarios(&defs, &cfg, &arena_brain, &map_dir, &mut |l| eprintln!("{l}")).map_err(|e| e.to_string())?;
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
