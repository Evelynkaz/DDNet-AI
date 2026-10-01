//! `ddnet-ai train`: behaviour cloning and DAgger for the fly and its controls (task 8.2).
//!
//! * `train collect --config <collect.toml>`: play arena games and label every decision with the
//!   fixed-iteration planner (D-017), appending to a teacher dataset.
//! * `train run --config <experiment.toml>`: behaviour cloning (+ DAgger rounds) of the fly or a
//!   control, resumable, writing the run directory (`config.toml`, `metrics.jsonl`, `status.json`,
//!   `state.bin`, `checkpoints/`).
//! * `train eval --config <experiment.toml> --bundle <file>`: per-head metrics of a checkpoint on
//!   the held-out sets.
//! * `train info <dataset-dir>`: what a teacher dataset holds.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Subcommand};
use ddai_env::output::git_info;
use ddai_train::experiment::{CollectConfig, expand_home, run_collect};
use ddai_train::runner::{ExperimentConfig, eval_bundle_offline, run_experiment};
use ddai_train::store::TeacherStore;

#[derive(Debug, Args)]
pub struct TrainArgs {
    #[command(subcommand)]
    pub command: TrainCommand,
}

#[derive(Debug, Subcommand)]
pub enum TrainCommand {
    /// Plays and labels the games of a collection config (TOML), appending to its dataset.
    Collect {
        #[arg(long)]
        config: PathBuf,
        /// Overrides the config's worker threads (at most 6 on the shared machine).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Runs (or resumes) an experiment: BC on teacher + human data, then DAgger rounds.
    Run {
        #[arg(long)]
        config: PathBuf,
        /// Overrides `train.threads` (at most 6 on the shared machine).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Per-head held-out metrics of a checkpoint (JSON on stdout).
    Eval {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        bundle: PathBuf,
        /// Report at the rate-matched thresholds calibrated on the validation sets (what a retrained
        /// model would store in its bundle) instead of the bundle's own thresholds.
        #[arg(long)]
        calibrate: bool,
    },
    /// The hook-head investigation on 7.3's synthetic task (task 8.2): one factor changed per arm.
    HookStudy {
        #[arg(long)]
        flyg: PathBuf,
        #[arg(long, default_value = "configs/fly/S-brain.toml")]
        brain_config: PathBuf,
        #[arg(long, default_value = "configs/fly/S-brain-dg.toml")]
        brain_config_dist_gains: PathBuf,
        #[arg(long, default_value = "configs/fly/S-brain-dg8.toml")]
        brain_config_dist_gains_8: PathBuf,
        #[arg(long)]
        train_map: PathBuf,
        #[arg(long)]
        other_map: PathBuf,
        #[arg(long, default_value_t = 200)]
        demo_steps: u64,
        #[arg(long, default_value_t = 1500)]
        long_steps: u64,
        #[arg(long, default_value_t = 6)]
        threads: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Prints what a fly checkpoint learned: calibration, decoder weights, encoder gains.
    Inspect {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        flyg: Option<PathBuf>,
    },
    /// Per-round action statistics of a teacher dataset: how often the player and the teacher hook,
    /// jump and fire, and how often they agree, per head (the student's imitation on visited states).
    Stats { dir: PathBuf },
    /// Prints a teacher dataset's manifest summary (and verifies every chunk with `--verify`).
    Info {
        dir: PathBuf,
        #[arg(long)]
        verify: bool,
    },
}

fn commit() -> String {
    let (c, dirty) = git_info(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
    if dirty { format!("{c}+dirty") } else { c }
}

pub fn run(args: TrainArgs) -> ExitCode {
    let result = match args.command {
        TrainCommand::Collect { config, threads } => collect_cmd(&config, threads),
        TrainCommand::Run { config, threads } => run_cmd(&config, threads),
        TrainCommand::Eval {
            config,
            bundle,
            calibrate,
        } => eval_cmd(&config, &bundle, calibrate),
        TrainCommand::HookStudy {
            flyg,
            brain_config,
            brain_config_dist_gains,
            brain_config_dist_gains_8,
            train_map,
            other_map,
            demo_steps,
            long_steps,
            threads,
            out,
        } => {
            let cfg = ddai_train::hook_study::HookStudyConfig {
                flyg: expand_home(&flyg.to_string_lossy()),
                brain_config,
                brain_config_dist_gains,
                brain_config_dist_gains_8,
                train_map: expand_home(&train_map.to_string_lossy()),
                other_map: expand_home(&other_map.to_string_lossy()),
                threads: threads.clamp(1, 6),
                demo_steps,
                long_steps,
                seed: 1,
            };
            ddai_train::hook_study::run_hook_study(&cfg, &mut |l| eprintln!("{l}")).and_then(|r| {
                let json = serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?;
                match out {
                    Some(p) => std::fs::write(p, json).map_err(|e| e.to_string()),
                    None => {
                        println!("{json}");
                        Ok(())
                    }
                }
            })
        }
        TrainCommand::Stats { dir } => stats_cmd(&dir),
        TrainCommand::Inspect { bundle, flyg } => inspect_cmd(&bundle, flyg.as_deref()),
        TrainCommand::Info { dir, verify } => info_cmd(&dir, verify),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn collect_cmd(path: &std::path::Path, threads: Option<usize>) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut cfg: CollectConfig = toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if let Some(t) = threads {
        cfg.threads = t;
    }
    cfg.threads = cfg.threads.clamp(1, 6);
    let t0 = std::time::Instant::now();
    let summaries = run_collect(&cfg, &commit(), &mut |l| {
        eprintln!("[{:7.1}s] {l}", t0.elapsed().as_secs_f64())
    })?;
    let steps: u64 = summaries.iter().map(|s| s.steps).sum();
    println!(
        "{}: {} jobs, {} labelled decisions in {:.1} s",
        cfg.name,
        summaries.len(),
        steps,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn info_cmd(dir: &std::path::Path, verify: bool) -> Result<(), String> {
    let store = TeacherStore::open(&expand_home(&dir.to_string_lossy())).map_err(|e| e.to_string())?;
    let m = &store.manifest;
    println!(
        "dataset {} (format {} v{}), code {}",
        m.name, m.format, m.format_version, m.code_commit
    );
    for a in &m.arenas {
        println!(
            "  arena {:<18} {:<8} map {}",
            a.name,
            a.split,
            a.map_sha256.as_deref().unwrap_or("synthetic")
        );
    }
    println!(
        "{} chunks, {} episodes, {} decisions",
        m.chunks.len(),
        m.total_episodes(),
        m.total_steps()
    );
    let mut rounds: std::collections::BTreeMap<u32, (u64, u64)> = std::collections::BTreeMap::new();
    for c in &m.chunks {
        let e = rounds.entry(c.round).or_default();
        e.0 += u64::from(c.episodes);
        e.1 += c.steps;
    }
    for (r, (eps, steps)) in rounds {
        println!("  round {r}: {eps} episodes, {steps} decisions");
    }
    if verify {
        store.verify().map_err(|e| e.to_string())?;
        println!("all chunks verified");
    }
    Ok(())
}

fn load_experiment(path: &std::path::Path, threads: Option<usize>) -> Result<ExperimentConfig, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut cfg: ExperimentConfig = toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if let Some(t) = threads {
        cfg.train.threads = t;
    }
    cfg.train.threads = cfg.train.threads.clamp(1, 6);
    Ok(cfg)
}

fn run_cmd(path: &std::path::Path, threads: Option<usize>) -> Result<(), String> {
    let cfg = load_experiment(path, threads)?;
    let t0 = std::time::Instant::now();
    run_experiment(&cfg, &mut |l| eprintln!("[{:8.1}s] {l}", t0.elapsed().as_secs_f64()))?;
    println!(
        "{}: done in {:.0} s, run directory {}",
        cfg.name,
        t0.elapsed().as_secs_f64(),
        cfg.run_dir
    );
    Ok(())
}

fn eval_cmd(path: &std::path::Path, bundle: &std::path::Path, calibrate: bool) -> Result<(), String> {
    let cfg = load_experiment(path, None)?;
    let recs = eval_bundle_offline(&cfg, bundle, calibrate, &mut |l| eprintln!("{l}"))?;
    println!("{}", serde_json::to_string_pretty(&recs).map_err(|e| e.to_string())?);
    Ok(())
}

fn inspect_cmd(bundle: &std::path::Path, flyg: Option<&std::path::Path>) -> Result<(), String> {
    let b = ddai_fly::bundle::load_bundle(bundle).map_err(|e| e.to_string())?;
    let t = ddai_fly::bundle::FlyBrainTemplate::load(bundle, flyg).map_err(|e| e.to_string())?;
    println!(
        "steps trained {}, seed {}, notes {:?}",
        b.meta.steps, b.meta.seed, b.meta.notes
    );
    let mut sig = b.calibration.sigma.clone();
    sig.sort_by(f32::total_cmp);
    let q = |p: f32| sig[((sig.len() - 1) as f32 * p) as usize];
    println!(
        "calibration: {} DNs, sigma min {:.3} / median {:.3} / max {:.3}; mu range {:.2}..{:.2}",
        sig.len(),
        q(0.0),
        q(0.5),
        q(1.0),
        b.calibration.mu.iter().cloned().fold(f32::MAX, f32::min),
        b.calibration.mu.iter().cloned().fold(f32::MIN, f32::max)
    );
    println!(
        "thresholds jump/hook/fire: {:.3}/{:.3}/{:.3} (0.500 = uncalibrated)",
        b.thresholds.jump, b.thresholds.hook, b.thresholds.fire
    );
    let d = &b.decoder_params;
    println!("decoder: dir_lr_w {:?} b {:.3}", d.direction_lr_w, d.direction_lr_b);
    println!("         stop_w {:?} b {:.3}", d.direction_stop_w, d.direction_stop_b);
    println!("         jump_w {:?} b {:.3}", d.jump_w, d.jump_b);
    println!("         hook_w {:?} b {:.3}", d.hook_w, d.hook_b);
    println!("         fire_w {:?} b {:.3}", d.fire_w, d.fire_b);
    println!(
        "         aim_pair_theta {:?} unpaired {:?}",
        d.aim_pair_theta, d.aim_unpaired_theta
    );
    let nb = t.brain_config().ray_grid.num_distance_bins;
    for a in t.encoder().assignments() {
        let p = a.param_id as usize;
        let gains = if b.encoder_params.bin_gain.is_empty() {
            String::new()
        } else {
            format!(
                " bins {:?}",
                b.encoder_params.bin_gain[p * nb..(p + 1) * nb]
                    .iter()
                    .map(|x| (x * 100.0).round() / 100.0)
                    .collect::<Vec<_>>()
            )
        };
        println!(
            "encoder {:<12} {:<20} g {:+.2} c {:+.3}{gains}",
            a.type_name, a.channel, b.encoder_params.g[p], b.encoder_params.c[p]
        );
    }
    let a_mean = b.fly_params.a.iter().sum::<f32>() / b.fly_params.a.len() as f32;
    println!(
        "fly: a mean {a_mean:.3} (init 2.2), b range {:.2}..{:.2}, theta range {:.2}..{:.2}",
        b.fly_params.b.iter().cloned().fold(f32::MAX, f32::min),
        b.fly_params.b.iter().cloned().fold(f32::MIN, f32::max),
        b.fly_params.theta.iter().cloned().fold(f32::MAX, f32::min),
        b.fly_params.theta.iter().cloned().fold(f32::MIN, f32::max)
    );
    Ok(())
}

fn stats_cmd(dir: &std::path::Path) -> Result<(), String> {
    let store = TeacherStore::open(&expand_home(&dir.to_string_lossy())).map_err(|e| e.to_string())?;
    #[derive(Default)]
    struct Acc {
        n: u64,
        student_steps: u64,
        agree: [u64; 4],
        played: [u64; 3],
        label: [u64; 3],
        outcomes: [u64; 4],
        episodes: u64,
        hook: ddai_train::play_stats::HookPlay,
    }
    let mut by_round: std::collections::BTreeMap<u32, Acc> = std::collections::BTreeMap::new();
    for ci in 0..store.manifest.chunks.len() {
        let round = store.manifest.chunks[ci].round;
        let chunk = store.read_chunk(ci).map_err(|e| e.to_string())?;
        let a = by_round.entry(round).or_default();
        for ep in &chunk.episodes {
            a.hook.add_episode(ep);
            a.episodes += 1;
            a.outcomes[ep.outcome as usize] += 1;
            for s in &ep.steps {
                if s.me.flags & ddai_dataset::types::char_flags::FROZEN != 0 {
                    continue;
                }
                a.n += 1;
                if !s.teacher_acted() && !s.noise() {
                    a.student_steps += 1;
                    a.agree[0] += u64::from(s.label.direction == s.played.direction);
                    a.agree[1] += u64::from(s.label.jump == s.played.jump);
                    a.agree[2] += u64::from(s.label.hook == s.played.hook);
                    a.agree[3] += u64::from(s.label.fire == s.played.fire);
                    a.played[0] += u64::from(s.played.jump);
                    a.played[1] += u64::from(s.played.hook);
                    a.played[2] += u64::from(s.played.fire);
                    a.label[0] += u64::from(s.label.jump);
                    a.label[1] += u64::from(s.label.hook);
                    a.label[2] += u64::from(s.label.fire);
                }
            }
        }
    }
    println!(
        "round | episodes W:L:D:T | unfrozen steps | student steps | agree dir/jump/hook/fire | student rate jump/hook/fire | teacher rate on the same states | hook start (own hook not out) student/teacher | hook release (own hook out) student/teacher | own hook out"
    );
    for (r, a) in by_round {
        let f = |x: u64, n: u64| if n == 0 { 0.0 } else { 100.0 * x as f64 / n as f64 };
        let h = a.hook.report();
        let p = |x: Option<f64>| x.map_or_else(|| "-".to_string(), |v| format!("{:.1}", 100.0 * v));
        println!(
            "{r:>5} | {} {}:{}:{}:{} | {} | {} | {:.0}/{:.0}/{:.0}/{:.0}% | {:.1}/{:.1}/{:.1}% | {:.1}/{:.1}/{:.1}% | {}/{}% | {}/{}% | {:.0}%",
            a.episodes,
            a.outcomes[0],
            a.outcomes[1],
            a.outcomes[2],
            a.outcomes[3],
            a.n,
            a.student_steps,
            f(a.agree[0], a.student_steps),
            f(a.agree[1], a.student_steps),
            f(a.agree[2], a.student_steps),
            f(a.agree[3], a.student_steps),
            f(a.played[0], a.student_steps),
            f(a.played[1], a.student_steps),
            f(a.played[2], a.student_steps),
            f(a.label[0], a.student_steps),
            f(a.label[1], a.student_steps),
            f(a.label[2], a.student_steps),
            p(h.start_student),
            p(h.start_teacher),
            p(h.release_student),
            p(h.release_teacher),
            100.0 * h.out_share,
        );
    }
    Ok(())
}
