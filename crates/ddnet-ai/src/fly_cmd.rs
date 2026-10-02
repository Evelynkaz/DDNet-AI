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
use ddai_fly::demo::{
    DirectionDemoConfig, direction_inputs_outputs_from_flyg, evaluate_direction, run_direction_demo_with_backend,
};
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
        /// Forward/backward backend: `per-seq` (task 7.2's per-sequence BPTT, the reference) or
        /// `batched` (task 7.2b: the whole batch at once, state `[neuron][sequence]`; same loss and
        /// gradients up to f32 summation order, much faster on the M graph).
        #[arg(long, value_enum, default_value_t = BackendArg::PerSeq)]
        backend: BackendArg,
    },
    /// Task 7.4: plays offline arena games with a trained fly and serves its visualisation stream on a bridge socket,
    /// for the web's «Муха» tab (`ddnet-ai web --bot-socket <the same socket>`): the owner watches the connectome brain
    /// work without a game server.
    Watch(crate::fly_watch::WatchArgs),
    /// Task 7.3, acceptance criterion 9: the encoder + fly + decoder learning demo — synthetic
    /// observations (a random spawn on a real block map, an opponent at a random relative
    /// position/velocity) and a scripted teacher (direction/jump/hook), trained with
    /// `guarded_adam_step` (fly) / `flat_adam` (encoder, decoder). Prints held-out accuracy
    /// before/after training, vs chance, and vs a same-size MLP control fed the same raw
    /// features; writes the per-step loss/grad-norm curve as CSV.
    BrainDemo {
        #[arg(long)]
        flyg: PathBuf,
        /// `configs/fly/{S,M}-brain.toml` — ray-grid/decoder/world-model/proprioception config.
        #[arg(long)]
        brain_config: PathBuf,
        /// A real DDNet `.map` file (task 7.3: "a random map crop from real block maps via
        /// ddai-map").
        #[arg(long)]
        map: PathBuf,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        #[arg(long, default_value_t = 4)]
        t_decisions: usize,
        #[arg(long, default_value_t = 200)]
        steps: usize,
        #[arg(long, default_value_t = 1.0)]
        grad_clip_norm: f32,
        #[arg(long, default_value_t = 2e-4)]
        lr_fly: f32,
        #[arg(long, default_value_t = 2e-2)]
        lr_encoder: f32,
        #[arg(long, default_value_t = 2e-2)]
        lr_decoder: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// CSV output path. Defaults to `~/aiddnet/data/runs/7.3-demo/<flyg-stem>-brain-demo.csv`.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

pub fn run(args: FlyArgs) -> ExitCode {
    match args.command {
        FlyCommand::Watch(watch) => crate::fly_watch::run(watch),
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
            backend,
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
            backend,
        }),
        FlyCommand::BrainDemo {
            flyg,
            brain_config,
            map,
            batch_size,
            t_decisions,
            steps,
            grad_clip_norm,
            lr_fly,
            lr_encoder,
            lr_decoder,
            seed,
            out,
        } => brain_demo(BrainDemoArgs {
            flyg,
            brain_config,
            map,
            batch_size,
            t_decisions,
            steps,
            grad_clip_norm,
            lr_fly,
            lr_encoder,
            lr_decoder,
            seed,
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
    backend: BackendArg,
}

/// `--backend` of `fly train-demo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendArg {
    /// Task 7.2's per-sequence BPTT (the reference).
    #[value(name = "per-seq")]
    PerSeq,
    /// Task 7.2b's batched backend.
    #[value(name = "batched")]
    Batched,
}

impl BackendArg {
    fn backend(self) -> ddai_fly::TrainBackend {
        match self {
            BackendArg::PerSeq => ddai_fly::TrainBackend::PerSequence,
            BackendArg::Batched => ddai_fly::TrainBackend::Batched,
        }
    }
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
    let metrics = run_direction_demo_with_backend(&mut model, &index, &demo, &v_init, args.backend.backend());
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

    println!("backend: {:?}", args.backend.backend());
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

struct BrainDemoArgs {
    flyg: PathBuf,
    brain_config: PathBuf,
    map: PathBuf,
    batch_size: usize,
    t_decisions: usize,
    steps: usize,
    grad_clip_norm: f32,
    lr_fly: f32,
    lr_encoder: f32,
    lr_decoder: f32,
    seed: u64,
    out: Option<PathBuf>,
}

fn default_brain_demo_csv_out_path(flyg_path: &Path) -> Result<PathBuf, ExitCode> {
    let home = std::env::var("HOME").map_err(|_| {
        eprintln!("HOME must be set to default --out to ~/aiddnet/data/runs/7.3-demo/");
        ExitCode::FAILURE
    })?;
    let stem = flyg_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "flyg".to_string());
    Ok(PathBuf::from(home)
        .join("aiddnet/data/runs/7.3-demo")
        .join(format!("{stem}-brain-demo.csv")))
}

fn write_brain_demo_metrics_csv(path: &Path, metrics: &[ddai_fly::demo_brain::StepMetric]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "step,loss,grad_norm,fly_applied,fly_lr_scale")?;
    for m in metrics {
        writeln!(
            f,
            "{},{},{},{},{}",
            m.step, m.loss, m.grad_norm, m.fly_applied, m.fly_lr_scale
        )?;
    }
    Ok(())
}

fn brain_demo(args: BrainDemoArgs) -> ExitCode {
    use ddai_fly::BackwardIndex;
    use ddai_fly::brain_config::load_brain_config;
    use ddai_fly::decoder::DecoderModel;
    use ddai_fly::demo_brain::{BrainDemoConfig, run_brain_demo};
    use ddai_fly::encoder::EncoderModel;

    let flyg = match load_flyg_or_fail(&args.flyg) {
        Ok(f) => f,
        Err(code) => return code,
    };
    let brain_cfg = match load_brain_config(&args.brain_config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to load {}: {e}", args.brain_config.display());
            return ExitCode::FAILURE;
        }
    };
    let map_bytes = match std::fs::read(&args.map) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read {}: {e}", args.map.display());
            return ExitCode::FAILURE;
        }
    };
    let loaded_map = match ddai_map::load_map(&map_bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to parse {}: {e}", args.map.display());
            return ExitCode::FAILURE;
        }
    };
    println!(
        "map: {} ({}x{} tiles)",
        args.map.display(),
        loaded_map.data.width,
        loaded_map.data.height
    );

    let config = ddai_fly::FlyConfig::default();
    let params = ddai_fly::FlyParams::init_default(&flyg, &config, args.seed);
    let mut model = match ddai_fly::FlyModel::new(flyg, config, params) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to build model: {e}");
            return ExitCode::FAILURE;
        }
    };
    let index = BackwardIndex::build(&model);

    let encoder = match EncoderModel::new(&model, brain_cfg.ray_grid, &brain_cfg.proprioception) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("failed to build encoder: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut encoder_params = ddai_fly::encoder::EncoderParams::init_default(encoder.num_params());
    println!(
        "encoder: {} inputs, {} (type, channel) params; visual channels with no neurons on this graph: {:?}",
        encoder.num_inputs(),
        encoder.num_params(),
        encoder.visual_channels_with_no_neurons()
    );

    let decoder = match DecoderModel::new(&model, brain_cfg.decoder) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("failed to build decoder: {e}");
            return ExitCode::FAILURE;
        }
    };
    let decoder_init = decoder.init_default_params();
    let mut decoder_params = decoder_init;

    // Calibration (task spec: "frozen per-DN calibration from a resting run"): the one documented
    // protocol every caller uses (review round 1, F10 -- an earlier revision had a *different*,
    // undocumented recipe inline here rather than going through `ddai_fly::calibrate_from_rest`,
    // the module doc comment's own documented protocol). A throwaway warm-up just for the printed
    // report below (`calibrate_from_rest` does its own warm-up internally, redundant but cheap).
    let warm_report = ddai_fly::FlyState::new(&model).warm_up(&model);
    println!(
        "warm-up: converged={} in {} decisions ({:.0}ms)",
        warm_report.converged, warm_report.decisions_run, warm_report.elapsed_ms
    );
    let calib = match ddai_fly::calibrate_from_rest(&model, args.seed, decoder.config().min_sigma) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to fit calibration: {e}");
            return ExitCode::FAILURE;
        }
    };

    let demo_cfg = BrainDemoConfig {
        batch_size: args.batch_size,
        t_decisions: args.t_decisions,
        steps: args.steps,
        grad_clip_norm: args.grad_clip_norm,
        lr_fly: args.lr_fly,
        lr_encoder: args.lr_encoder,
        lr_decoder: args.lr_decoder,
        seed: args.seed,
        ..BrainDemoConfig::default()
    };

    let start = Instant::now();
    let report = run_brain_demo(
        &mut model,
        &index,
        &encoder,
        &mut encoder_params,
        &decoder,
        &mut decoder_params,
        &calib,
        std::sync::Arc::new(loaded_map.data),
        &demo_cfg,
    );
    let elapsed = start.elapsed();

    println!(
        "trained {} steps ({} samples/step) in {elapsed:?}",
        report.metrics.len(),
        args.batch_size
    );
    println!(
        "fly pipeline trainable params: {} | MLP control trainable params: {} | held-out n={} | guarded steps skipped: {}",
        report.fly_num_params, report.mlp_num_params, report.held_out_n, report.guarded_steps_skipped
    );
    println!(
        "{:>9} {:>6} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "head", "prev.", "majority", "fly(before)", "fly(after)", "mlp", "bal.acc", "auroc"
    );
    println!(
        "{:>9} {:>6.3} {:>10.3} {:>10.3} {:>10.3} {:>8.3} {:>8.3} {:>8}",
        "direction",
        report
            .metrics_before_fly
            .direction
            .class_prevalence
            .iter()
            .cloned()
            .fold(0.0f32, f32::max),
        report.metrics_before_fly.direction.majority_baseline_accuracy,
        report.metrics_before_fly.direction.accuracy,
        report.metrics_after_fly.direction.accuracy,
        report.metrics_after_mlp.direction.accuracy,
        report.metrics_after_fly.direction.balanced_accuracy,
        "-"
    );
    println!(
        "{:>9} {:>6.3} {:>10.3} {:>10.3} {:>10.3} {:>8.3} {:>8.3} {:>8.3}",
        "jump",
        report.metrics_before_fly.jump.prevalence,
        report.metrics_before_fly.jump.majority_baseline_accuracy,
        report.metrics_before_fly.jump.accuracy,
        report.metrics_after_fly.jump.accuracy,
        report.metrics_after_mlp.jump.accuracy,
        report.metrics_after_fly.jump.balanced_accuracy,
        report.metrics_after_fly.jump.auroc
    );
    println!(
        "{:>9} {:>6.3} {:>10.3} {:>10.3} {:>10.3} {:>8.3} {:>8.3} {:>8.3}",
        "hook",
        report.metrics_before_fly.hook.prevalence,
        report.metrics_before_fly.hook.majority_baseline_accuracy,
        report.metrics_before_fly.hook.accuracy,
        report.metrics_after_fly.hook.accuracy,
        report.metrics_after_mlp.hook.accuracy,
        report.metrics_after_fly.hook.balanced_accuracy,
        report.metrics_after_fly.hook.auroc
    );
    println!(
        "95% CI (Wilson) fly(after): direction={:?} jump={:?} hook={:?}",
        report.metrics_after_fly.direction.accuracy_wilson_ci95,
        report.metrics_after_fly.jump.accuracy_wilson_ci95,
        report.metrics_after_fly.hook.accuracy_wilson_ci95,
    );

    let out_path = match args.out {
        Some(p) => p,
        None => match default_brain_demo_csv_out_path(&args.flyg) {
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
    match write_brain_demo_metrics_csv(&out_path, &report.metrics) {
        Ok(()) => println!("wrote loss/grad_norm curve to {}", out_path.display()),
        Err(e) => {
            eprintln!("failed to write {}: {e}", out_path.display());
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
