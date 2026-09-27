//! `ddnet-ai fly` subcommand group: `bench` (the realistic 25Hz decision loop — acceptance
//! criterion 4's methodology, task 7.1) and `info` (graph sizes/types/output groups, task 7.1) for
//! a compiled `.flyg` (`ddai-flyg`/`ddai-fly`); `train-demo` (task 7.2, acceptance criterion 5) —
//! the same "left/right visual sector -> direction_left/right DN group" synthetic supervised demo
//! `ddai-fly`'s own `tests/train_demo_real_graph.rs` runs, exposed here so it can be run manually
//! against any compiled `.flyg` and its loss/grad-norm curve saved as CSV without going through
//! `cargo test`.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Args, Subcommand};
use ddai_fly::demo::{DirectionDemoConfig, direction_inputs_outputs_from_flyg, evaluate_direction, run_direction_demo};
use ddai_fly::optim::{AdamConfig, GuardedAdamConfig};
use ddai_fly::{BackwardIndex, FlyConfig, FlyModel, FlyParams, FlyState};

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
    /// Task 7.2, acceptance criterion 5: trains the "left/right visual sector ->
    /// direction_left/right DN group" synthetic supervised demo on a compiled `.flyg` and writes
    /// its loss/grad-norm curve as CSV. Requires the graph to have `output_groups` entries named
    /// `--left-action`/`--right-action` (the real S/M graphs' `direction_left`/`direction_right`
    /// by default — see `ddnet-ai fly info`).
    TrainDemo {
        #[arg(long)]
        flyg: PathBuf,
        #[arg(long, default_value_t = 300)]
        steps: usize,
        #[arg(long, default_value_t = 24)]
        batch_size: usize,
        #[arg(long, default_value_t = 6)]
        t_decisions: usize,
        #[arg(long, default_value_t = 2)]
        readout_decisions: usize,
        #[arg(long, default_value_t = 0.5)]
        activation_prob: f32,
        #[arg(long, default_value_t = 6.0)]
        target_high: f32,
        #[arg(long, default_value_t = 1.0)]
        target_low: f32,
        #[arg(long, default_value_t = 1.0)]
        grad_clip_norm: f32,
        #[arg(long, default_value_t = 2e-2)]
        lr: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value = "direction_left")]
        left_action: String,
        #[arg(long, default_value = "direction_right")]
        right_action: String,
        /// CSV output path. Defaults to `~/aiddnet/data/runs/7.2-demo/<flyg-stem>-direction-demo.csv`
        /// (this project's convention: experiment output lives under `~/aiddnet/data`, never in
        /// the repo — see `CLAUDE.md`).
        #[arg(long)]
        out: Option<PathBuf>,
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
        FlyCommand::TrainDemo {
            flyg,
            steps,
            batch_size,
            t_decisions,
            readout_decisions,
            activation_prob,
            target_high,
            target_low,
            grad_clip_norm,
            lr,
            seed,
            left_action,
            right_action,
            out,
        } => train_demo(TrainDemoArgs {
            flyg,
            steps,
            batch_size,
            t_decisions,
            readout_decisions,
            activation_prob,
            target_high,
            target_low,
            grad_clip_norm,
            lr,
            seed,
            left_action,
            right_action,
            out,
        }),
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

struct TrainDemoArgs {
    flyg: PathBuf,
    steps: usize,
    batch_size: usize,
    t_decisions: usize,
    readout_decisions: usize,
    activation_prob: f32,
    target_high: f32,
    target_low: f32,
    grad_clip_norm: f32,
    lr: f32,
    seed: u64,
    left_action: String,
    right_action: String,
    out: Option<PathBuf>,
}

fn default_csv_out_path(flyg_path: &Path) -> Result<PathBuf, ExitCode> {
    let home = std::env::var("HOME").map_err(|_| {
        eprintln!("HOME must be set to default --out to ~/aiddnet/data/runs/7.2-demo/");
        ExitCode::FAILURE
    })?;
    let stem = flyg_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "flyg".to_string());
    Ok(PathBuf::from(home)
        .join("aiddnet/data/runs/7.2-demo")
        .join(format!("{stem}-direction-demo.csv")))
}

fn train_demo(args: TrainDemoArgs) -> ExitCode {
    let flyg = match load_flyg_or_fail(&args.flyg) {
        Ok(f) => f,
        Err(code) => return code,
    };

    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, args.seed);
    let mut model = match FlyModel::new(flyg, config, params) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to build model: {e}");
            return ExitCode::FAILURE;
        }
    };
    let index = BackwardIndex::build(&model);

    let mut warm = FlyState::new(&model);
    let warm_report = warm.warm_up(&model);
    println!(
        "warm-up: converged={} in {} decisions ({:.0}ms)",
        warm_report.converged, warm_report.decisions_run, warm_report.elapsed_ms
    );
    if !warm_report.converged {
        eprintln!("warning: warm-up did not converge within the default cap; proceeding with its best-effort state");
    }
    let v_init = warm.v().to_vec();

    let (left_inputs, right_inputs, left_outputs, right_outputs) =
        direction_inputs_outputs_from_flyg(&model, &args.left_action, &args.right_action);
    println!(
        "task: {} left inputs, {} right inputs, {} left_outputs ({:?}), {} right_outputs ({:?})",
        left_inputs.len(),
        right_inputs.len(),
        left_outputs.len(),
        args.left_action,
        right_outputs.len(),
        args.right_action,
    );

    let demo = DirectionDemoConfig {
        left_inputs,
        right_inputs,
        left_outputs,
        right_outputs,
        target_high: args.target_high,
        target_low: args.target_low,
        activation_prob: args.activation_prob,
        batch_size: args.batch_size,
        t_decisions: args.t_decisions,
        readout_decisions: args.readout_decisions,
        steps: args.steps,
        grad_clip_norm: args.grad_clip_norm,
        adam: GuardedAdamConfig {
            adam: AdamConfig {
                lr_a: args.lr,
                lr_b: args.lr,
                lr_theta: args.lr,
                ..AdamConfig::default()
            },
            ..GuardedAdamConfig::default()
        },
        seed: args.seed,
    };

    let held_out_seed = args.seed.wrapping_add(1);
    let eval_before = evaluate_direction(&model, &demo, &v_init, held_out_seed, 60);

    let start = Instant::now();
    let metrics = run_direction_demo(&mut model, &index, &demo, &v_init);
    let elapsed = start.elapsed();

    let not_applied = metrics.iter().filter(|m| !m.applied).count();
    let rolled_back = metrics
        .iter()
        .filter(|m| matches!(m.outcome, ddai_fly::optim::GuardedStepOutcome::RolledBack { .. }))
        .count();
    let max_abs_v_ever = metrics.iter().map(|m| m.max_abs_v).fold(0.0f32, f32::max);
    let final_lr_scale = metrics.last().map(|m| m.lr_scale).unwrap_or(1.0);
    let initial_loss = metrics.first().map(|m| m.loss).unwrap_or(f32::NAN);
    let tail = metrics.len().saturating_sub(10);
    let final_loss: f32 = if metrics.len() > tail {
        metrics[tail..].iter().map(|m| m.loss).sum::<f32>() / (metrics.len() - tail) as f32
    } else {
        f32::NAN
    };
    let eval_after = evaluate_direction(&model, &demo, &v_init, held_out_seed, 60);

    println!("trained {} steps in {elapsed:?}", metrics.len());
    println!("loss: initial={initial_loss:.4}, final (last 10 avg)={final_loss:.4}");
    println!(
        "NaN/inf guard: {not_applied}/{} steps not applied ({rolled_back} rolled back), final lr_scale={final_lr_scale:.4}",
        metrics.len()
    );
    println!("max |V| observed during training: {max_abs_v_ever:.3}");
    println!(
        "held-out (seed={held_out_seed}, 60 patterns): accuracy before={:.3} after={:.3}; mean margin before={:.4} after={:.4}",
        eval_before.accuracy, eval_after.accuracy, eval_before.mean_margin, eval_after.mean_margin
    );

    let out_path = match args.out {
        Some(p) => p,
        None => match default_csv_out_path(&args.flyg) {
            Ok(p) => p,
            Err(code) => return code,
        },
    };
    if let Some(parent) = out_path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        eprintln!("failed to create {}: {e}", parent.display());
        return ExitCode::FAILURE;
    }
    match write_metrics_csv(&out_path, &metrics) {
        Ok(()) => println!("wrote loss/grad_norm curve to {}", out_path.display()),
        Err(e) => {
            eprintln!("failed to write {}: {e}", out_path.display());
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}

fn write_metrics_csv(path: &Path, metrics: &[ddai_fly::demo::StepMetric]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "step,loss,grad_norm,applied,lr_scale,max_abs_v")?;
    for m in metrics {
        writeln!(
            f,
            "{},{},{},{},{},{}",
            m.step, m.loss, m.grad_norm, m.applied, m.lr_scale, m.max_abs_v
        )?;
    }
    Ok(())
}
