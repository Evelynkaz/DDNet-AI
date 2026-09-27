//! `ddnet-ai fly` subcommand group (task 7.1, acceptance criterion 6): `bench` (the realistic 25Hz
//! decision loop — acceptance criterion 4's methodology) and `info` (graph sizes/types/output
//! groups) for a compiled `.flyg` (`ddai-flyg`/`ddai-fly`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Args, Subcommand};
use ddai_fly::{FlyConfig, FlyModel, FlyParams, FlyState};

#[derive(Debug, Args)]
pub struct FlyArgs {
    #[command(subcommand)]
    pub command: FlyCommand,
}

#[derive(Debug, Subcommand)]
pub enum FlyCommand {
    /// Runs `decisions` realistic 25Hz-loop decisions (compute one, sleep to the next tick — see
    /// `docs/research/rust-stack.md` §4) on one pinned core, and reports median/p99/max/mean
    /// timings, the count over 5ms, `set_params` (weight precompute) time, and memory footprint.
    Bench {
        #[arg(long)]
        flyg: PathBuf,
        #[arg(long, default_value_t = 4)]
        substeps: u32,
        #[arg(long, default_value_t = 500)]
        decisions: u32,
        #[arg(long, default_value_t = 40)]
        tick_ms: u64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Prints a loaded `.flyg`'s sizes, type/edge/sign counts, and output groups.
    Info {
        #[arg(long)]
        flyg: PathBuf,
    },
}

pub fn run(args: FlyArgs) -> ExitCode {
    match args.command {
        FlyCommand::Bench {
            flyg,
            substeps,
            decisions,
            tick_ms,
            seed,
        } => bench(&flyg, substeps, decisions, tick_ms, seed),
        FlyCommand::Info { flyg } => info(&flyg),
    }
}

fn load_flyg_or_fail(path: &Path) -> Result<ddai_flyg::Flyg, ExitCode> {
    ddai_flyg::load(path).map_err(|e| {
        eprintln!("failed to load {}: {e}", path.display());
        ExitCode::FAILURE
    })
}

/// Best-effort: pins this (the current) thread to one core, per acceptance criterion 4's
/// methodology. Returns whether it actually succeeded, so the caller can note in its report that
/// the numbers may carry more host-scheduler noise on a platform/sandbox where this isn't
/// supported — never a hard failure.
fn pin_this_thread_to_one_core() -> bool {
    core_affinity::get_core_ids()
        .and_then(|mut ids| ids.pop())
        .map(core_affinity::set_for_current)
        .unwrap_or(false)
}

fn read_load_average() -> Option<String> {
    let text = std::fs::read_to_string("/proc/loadavg").ok()?;
    let fields: Vec<&str> = text.split_whitespace().take(3).collect();
    (fields.len() == 3).then(|| fields.join(" "))
}

/// Rough estimate of one `FlyState`'s scratch memory (not counting the `FlyModel` it points at,
/// which `FlyModel::memory_footprint_bytes` already covers): five neuron-length `f32` buffers
/// (`v`, `r_buf`, `v_inf_buf`, `r_final_buf`, `rest_v`), one output-length, two type-length.
fn state_footprint_bytes(model: &FlyModel) -> usize {
    let f32_bytes = std::mem::size_of::<f32>();
    (5 * model.num_neurons() + model.num_outputs() + 2 * model.num_types()) * f32_bytes
}

fn bench(flyg_path: &Path, substeps: u32, decisions: u32, tick_ms: u64, seed: u64) -> ExitCode {
    let flyg = match load_flyg_or_fail(flyg_path) {
        Ok(f) => f,
        Err(code) => return code,
    };

    let pinned = pin_this_thread_to_one_core();
    let load_avg = read_load_average();

    let config = FlyConfig {
        substeps_per_decision: substeps,
        ..FlyConfig::default()
    };
    let params = FlyParams::init_default(&flyg, &config, seed);

    let build_start = Instant::now();
    let model = match FlyModel::new(flyg, config, params.clone()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to build model: {e}");
            return ExitCode::FAILURE;
        }
    };
    let build_elapsed = build_start.elapsed();

    let mut model_for_set_params = model.clone();
    let set_params_start = Instant::now();
    if let Err(e) = model_for_set_params.set_params(params) {
        eprintln!("set_params failed: {e}");
        return ExitCode::FAILURE;
    }
    let set_params_elapsed = set_params_start.elapsed();

    let mut state = FlyState::new(&model);
    let warm_up_report = state.warm_up(&model);

    let report = ddai_fly::run_duty_cycle(&model, &mut state, decisions, Duration::from_millis(tick_ms), seed, 0.2);

    println!(
        "graph: {} ({} neurons, {} edges, {} types)",
        flyg_path.display(),
        model.num_neurons(),
        model.flyg().edges.num_edges(),
        model.num_types()
    );
    println!("thread pinned to one core: {pinned}");
    if let Some(avg) = &load_avg {
        println!("load average (1m 5m 15m) at start: {avg}");
    }
    println!(
        "warm-up: converged={} in {} decisions ({:.0}ms, final max|dV|={:.4})",
        warm_up_report.converged,
        warm_up_report.decisions_run,
        warm_up_report.elapsed_ms,
        warm_up_report.final_max_delta_v
    );
    println!("model build (load .flyg + initial precompute) time: {build_elapsed:?}");
    println!("set_params (weight precompute only) time: {set_params_elapsed:?}");
    println!(
        "memory footprint: model {:.3} MiB, one state {:.3} MiB",
        model.memory_footprint_bytes() as f64 / (1024.0 * 1024.0),
        state_footprint_bytes(&model) as f64 / (1024.0 * 1024.0)
    );
    println!();
    println!(
        "{:>10} {:>9} {:>10} {:>9} {:>9} {:>9} {:>6}",
        "decisions", "substeps", "median_ms", "p99_ms", "max_ms", "mean_ms", ">5ms"
    );
    println!(
        "{:>10} {:>9} {:>10.3} {:>9.3} {:>9.3} {:>9.3} {:>6}",
        report.num_decisions,
        report.substeps,
        report.median_ms,
        report.p99_ms,
        report.max_ms,
        report.mean_ms,
        report.over_5ms
    );

    ExitCode::SUCCESS
}

fn info(flyg_path: &Path) -> ExitCode {
    let flyg = match load_flyg_or_fail(flyg_path) {
        Ok(f) => f,
        Err(code) => return code,
    };

    let roles = flyg.summary.neurons_by_role;
    let signs = flyg.summary.sign_counts;
    println!("file: {}", flyg_path.display());
    println!(
        "neurons: {} (input_visual={}, input_ascending={}, hidden={}, output={})",
        flyg.neurons.len(),
        roles.input_visual,
        roles.input_ascending,
        roles.hidden,
        roles.output
    );
    println!("edges: {}", flyg.edges.num_edges());
    println!(
        "types: {} (excitatory={}, inhibitory={}, neutral={}, uncertain={})",
        flyg.types.len(),
        signs.excitatory,
        signs.inhibitory,
        signs.neutral,
        flyg.summary.uncertain_types
    );
    println!(
        "type_pairs: {} (distinct shared_param_id count: {})",
        flyg.type_pairs.len(),
        flyg.summary.shared_param_count
    );
    println!(
        "receptive fields: {} InputVisual neurons, {} via fallback",
        roles.input_visual, flyg.summary.rf_fallback_count
    );
    println!("input_channels: {} entries", flyg.input_channels.len());
    println!("output_groups: {} entries", flyg.output_groups.len());
    for g in &flyg.output_groups {
        println!("  {}: {} member neuron(s)", g.action, g.members.len());
    }

    ExitCode::SUCCESS
}
