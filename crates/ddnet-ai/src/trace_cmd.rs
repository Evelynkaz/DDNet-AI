//! Implementation of the `ddnet-ai trace` subcommand group: thin CLI glue around
//! `ddai-trace`'s map/scenario/trace formats and the `random-v1` generator. See
//! `docs/formats.md` for the file formats these commands read and write.

use clap::{Args, Subcommand};
use ddai_trace::generator::{self, Params};
use ddai_trace::{rawmap, synthetic, trace::Trace};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Args)]
pub struct TraceArgs {
    #[command(subcommand)]
    pub command: TraceCommand,
}

#[derive(Debug, Subcommand)]
pub enum TraceCommand {
    /// Writes a synthetic recipe, or a real DDNet `.map` file (task 1.4, `ddai-map`), as a
    /// rawmap v1 file. Exactly one of `--recipe`/`--map` is required.
    ExportMap {
        /// Recipe name (see `ddai_trace::synthetic::RECIPES`), e.g. "arena".
        #[arg(long, conflicts_with = "map")]
        recipe: Option<String>,
        /// A real DDNet `.map` file to read with `ddai_map::load_map`.
        #[arg(long, conflicts_with = "recipe")]
        map: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Generates a `random-v1` scenario and writes it as a scenario v2 file.
    GenScenario {
        #[arg(long)]
        recipe: String,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        ticks: u32,
        /// Number of characters, 1..=4.
        #[arg(long)]
        chars: u32,
        /// Sets the scenario's `no_weak_hook` world flag (mirrors `sv_no_weak_hook`). `random-v1`
        /// itself always generates `false`; this flag overrides that after generation, since
        /// `no_weak_hook`/tuning overrides are scenario-level knobs, not generator params (see
        /// `docs/formats.md` §4 and review round 1, finding F8).
        #[arg(long = "no-weak-hook")]
        no_weak_hook: bool,
        /// A tuning override, as `name=value_x100` (e.g. `gravity=0`, `hook_length=38000`) —
        /// `name` is a `CTuningParams` script name (`src/game/tuning.h`), `value_x100` the
        /// fixed-point integer `CTuneParam` stores internally. Repeatable.
        #[arg(long = "tune")]
        tune: Vec<String>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Compares two trace v1 files field-by-field and reports the first mismatch.
    Diff { a: PathBuf, b: PathBuf },
    /// Computes the canonical per-tick FNV-1a 64 state hash for every tick of a trace v1 file.
    Hashes {
        trace: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

pub fn run(args: TraceArgs) -> ExitCode {
    match args.command {
        TraceCommand::ExportMap { recipe, map, out } => export_map(recipe.as_deref(), map.as_deref(), &out),
        TraceCommand::GenScenario {
            recipe,
            seed,
            ticks,
            chars,
            no_weak_hook,
            tune,
            out,
        } => gen_scenario(&recipe, seed, ticks, chars, no_weak_hook, &tune, &out),
        TraceCommand::Diff { a, b } => diff(&a, &b),
        TraceCommand::Hashes { trace, out } => hashes(&trace, &out),
    }
}

fn export_map(recipe: Option<&str>, map_path: Option<&std::path::Path>, out: &PathBuf) -> ExitCode {
    let map = match (recipe, map_path) {
        (Some(recipe), None) => match synthetic::build(recipe) {
            Some(map) => map,
            None => {
                eprintln!("unknown recipe '{recipe}', expected one of {:?}", synthetic::RECIPES);
                return ExitCode::FAILURE;
            }
        },
        (None, Some(path)) => {
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("failed to read {}: {e}", path.display());
                    return ExitCode::FAILURE;
                }
            };
            match ddai_map::load_map(&bytes) {
                Ok(loaded) => loaded.data,
                Err(e) => {
                    eprintln!("failed to load map {}: {e}", path.display());
                    return ExitCode::FAILURE;
                }
            }
        }
        (Some(_), Some(_)) | (None, None) => {
            eprintln!("exactly one of --recipe/--map is required");
            return ExitCode::FAILURE;
        }
    };
    let bytes = rawmap::write(&map);
    if let Err(e) = std::fs::write(out, &bytes) {
        eprintln!("failed to write {}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    println!("wrote {} ({} bytes)", out.display(), bytes.len());
    ExitCode::SUCCESS
}

fn gen_scenario(
    recipe: &str,
    seed: u64,
    ticks: u32,
    chars: u32,
    no_weak_hook: bool,
    tune: &[String],
    out: &PathBuf,
) -> ExitCode {
    let mut scenario = match generator::random_v1(
        recipe,
        Params {
            seed,
            ticks,
            characters: chars,
        },
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to generate scenario: {e}");
            return ExitCode::FAILURE;
        }
    };
    scenario.no_weak_hook = no_weak_hook;
    for spec in tune {
        let Some((name, value)) = spec.split_once('=') else {
            eprintln!("invalid --tune '{spec}', expected 'name=value_x100'");
            return ExitCode::FAILURE;
        };
        let Ok(value_x100) = value.trim().parse::<i32>() else {
            eprintln!("invalid --tune '{spec}': '{value}' is not an integer");
            return ExitCode::FAILURE;
        };
        scenario.tuning_overrides.push(ddai_trace::scenario::TuningOverride {
            name: name.trim().to_string(),
            value_x100,
        });
    }
    let bytes = scenario.write_bytes();
    if let Err(e) = std::fs::write(out, &bytes) {
        eprintln!("failed to write {}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    println!(
        "wrote {} ({} bytes, {} ticks, {} characters)",
        out.display(),
        bytes.len(),
        ticks,
        chars
    );
    ExitCode::SUCCESS
}

fn diff(a_path: &PathBuf, b_path: &PathBuf) -> ExitCode {
    let a = match read_trace(a_path) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let b = match read_trace(b_path) {
        Ok(t) => t,
        Err(e) => return e,
    };
    match ddai_trace::trace::diff(&a, &b) {
        Err(shape_err) => {
            eprintln!("traces have different shapes: {shape_err}");
            ExitCode::FAILURE
        }
        Ok(result) => {
            if let Some(m) = &result.first_mismatch {
                println!("first mismatch: {m}");
                println!("total mismatches: {}", result.mismatch_count);
                ExitCode::FAILURE
            } else {
                println!("identical ({} ticks, {} characters)", a.ticks(), a.character_ids.len());
                ExitCode::SUCCESS
            }
        }
    }
}

fn hashes(trace_path: &PathBuf, out: &PathBuf) -> ExitCode {
    let trace = match read_trace(trace_path) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let hashes = trace.tick_hashes();
    let json = serde_json::json!({ "tick_hashes": hashes });
    let text = serde_json::to_string_pretty(&json).expect("JSON serialization of a plain hash list cannot fail");
    if let Err(e) = std::fs::write(out, text) {
        eprintln!("failed to write {}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    println!("wrote {} ({} tick hashes)", out.display(), hashes.len());
    ExitCode::SUCCESS
}

fn read_trace(path: &PathBuf) -> Result<Trace, ExitCode> {
    let bytes = std::fs::read(path).map_err(|e| {
        eprintln!("failed to read {}: {e}", path.display());
        ExitCode::FAILURE
    })?;
    Trace::read_bytes(&bytes).map_err(|e| {
        eprintln!("failed to parse trace {}: {e}", path.display());
        ExitCode::FAILURE
    })
}
