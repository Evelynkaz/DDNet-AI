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
use ddai_train::runner::{ExperimentConfig, eval_bundle_offline, run_experiment_with};
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
        /// Overrides the config's worker threads (at most 3 on the shared machine).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Runs (or resumes) an experiment: BC on teacher + human data, then DAgger rounds.
    Run {
        #[arg(long)]
        config: PathBuf,
        /// Overrides `train.threads` (at most 3 on the shared machine).
        #[arg(long)]
        threads: Option<usize>,
        /// Resume even though the configuration differs from the one the run started with (the old
        /// config is kept as `config-before-N.toml`).
        #[arg(long)]
        allow_config_change: bool,
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
        /// Also write the bundle with the calibrated thresholds to this path (implies `--calibrate`).
        #[arg(long)]
        write_calibrated: Option<PathBuf>,
    },
    /// Writes a copy of a checkpoint with other decision thresholds (jump / hook / fire; any left out keeps the
    /// checkpoint's own). The same weights at different thresholds can then be A/B-tested in the arena or scanned
    /// for their in-play hook rates (E-005 review F9).
    SetThresholds {
        /// `fly`, `mlp` or `gru`.
        #[arg(long)]
        kind: String,
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        jump: Option<f32>,
        #[arg(long)]
        hook: Option<f32>,
        #[arg(long)]
        fire: Option<f32>,
        /// Task 8.6, the **hysteresis decode** (fly only): with the own hook key up, press when the hook probability reaches this
        /// (needs `--hook-lo`); latched on the fly's own previous hook command. The plain decode (`--hook`) is the default.
        #[arg(long, requires = "hook_lo")]
        hook_hi: Option<f32>,
        /// With the own hook key down, keep it down while the hook probability stays at or above this (below `--hook-hi` for a hysteresis;
        /// equal is the plain rule, above it the opposite of a hysteresis).
        #[arg(long, requires = "hook_hi")]
        hook_lo: Option<f32>,
        /// Back to the plain decode (the bundle's `thresholds.hook` whatever the last command was).
        #[arg(long, conflicts_with = "hook_hi")]
        plain_hook: bool,
    },
    /// Information probes of the hook decision (task 8.6): small models on the fly's encoder input, on privileged exact-state features and on the
    /// encoder input plus derived physics features, against the real flies, per latch state, on the BC's teacher-val labels of the first decisions
    /// after a freeze. JSON on stdout / `--out`.
    ProbeHook {
        /// A BC config (`teacher_base`, `teacher_data` and the graph are read from it).
        #[arg(long)]
        config: PathBuf,
        /// Reference flies, `name=bundle`, repeatable; the first one's encoder gives the input vector (all share the encoder of E-008 s2 upgraded).
        #[arg(long = "fly", required = true)]
        flies: Vec<String>,
        /// Flies whose own network state is probed (`name=bundle`, repeatable): their DN z-scores and membrane state on the hook view.
        #[arg(long = "state-fly")]
        state_flies: Vec<String>,
        #[arg(long, default_value_t = 16)]
        first: usize,
        #[arg(long, default_value_t = 4)]
        frames: usize,
        #[arg(long, default_value_t = 2)]
        seeds: usize,
        #[arg(long, default_value_t = 14000)]
        max_train_rows: usize,
        /// Only the probes on the `--state-fly` network states.
        #[arg(long)]
        state_only: bool,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        threads: usize,
    },
    /// Writes a copy of a fly checkpoint whose hook head is an **intent** head (task 8.6, bundle v4): `P(press | released)` and
    /// `P(release | held)` chosen by the fly's own previous hook command. It plays bit for bit like the original until it is trained.
    UpgradeHook {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// A model's hook behaviour in closed loop: start and release rates against the teacher's on the same
    /// states (JSON on stdout). The model plays alone; the planner labels every state it visits.
    HookPlay {
        /// `fly:<bundle>`, `mlp:<bundle>`, `gru:<bundle>`.
        #[arg(long)]
        actor: String,
        /// Comma-separated arenas (training arenas unless you mean to look at a holdout).
        #[arg(long, default_value = "clb-left,pit")]
        arenas: String,
        #[arg(long, default_value_t = 100)]
        games: u32,
        /// Opponents of slot 1.. (comma-separated; default one scripted bot).
        #[arg(long, default_value = "scripted")]
        opponents: String,
        #[arg(long, default_value_t = 7_000_000_000)]
        seed: u64,
        #[arg(long, default_value = "configs/arenas")]
        arenas_dir: PathBuf,
        #[arg(long)]
        map_dir: Option<PathBuf>,
        #[arg(long)]
        flyg: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        threads: usize,
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
        #[arg(long, default_value_t = 3)]
        threads: usize,
        /// Run the 2x2x2 factorial (distance-bin gains x 6x connectome lr x alpha 4) instead of the E-005 arm list.
        #[arg(long)]
        factorial: bool,
        /// Seed of scenes and initialisation (the study is repeated over seeds for uncertainty).
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// OpenAI-ES on the outcome of the held block, and its tools (task 8.5a): `bank`, `run`, `eval`, `compare`, `scan`.
    Es(crate::es_cmd::EsArgs),
    /// Recurrent PPO on the held-block outcome from a BC checkpoint (task 8.5b).
    Ppo(crate::ppo_cmd::PpoArgs),
    /// Writes a copy of a fly checkpoint whose encoder also reads the target opponent's state (frozen, freeze time left,
    /// velocity, hook), with ZERO weights for it: the copy plays bit for bit like the original (task 8.5a).
    UpgradeBundle {
        #[arg(long)]
        bundle: PathBuf,
        /// A TOML file with an `[opponent_state]` section (`configs/fly/S-opponent-state.toml`).
        #[arg(long)]
        opponent_state: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        flyg: Option<PathBuf>,
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
        TrainCommand::Run {
            config,
            threads,
            allow_config_change,
        } => run_cmd(&config, threads, allow_config_change),
        TrainCommand::Eval {
            config,
            bundle,
            calibrate,
            write_calibrated,
        } => eval_cmd(&config, &bundle, calibrate, write_calibrated.as_deref()),
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
            factorial,
            seed,
            out,
        } => {
            let cfg = ddai_train::hook_study::HookStudyConfig {
                flyg: expand_home(&flyg.to_string_lossy()),
                brain_config,
                brain_config_dist_gains,
                brain_config_dist_gains_8,
                train_map: expand_home(&train_map.to_string_lossy()),
                other_map: expand_home(&other_map.to_string_lossy()),
                threads: threads.clamp(1, 3),
                demo_steps,
                long_steps,
                seed,
                factorial,
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
        TrainCommand::SetThresholds {
            kind,
            bundle,
            out,
            jump,
            hook,
            fire,
            hook_hi,
            hook_lo,
            plain_hook,
        } => set_thresholds_cmd(&kind, &bundle, &out, jump, hook, fire, hook_hi.zip(hook_lo), plain_hook),
        TrainCommand::UpgradeHook { bundle, out } => upgrade_hook_cmd(&bundle, &out),
        TrainCommand::ProbeHook {
            config,
            flies,
            state_flies,
            first,
            frames,
            seeds,
            max_train_rows,
            state_only,
            out,
            threads,
        } => probe_hook_cmd(
            &config,
            &flies,
            &state_flies,
            first,
            frames,
            seeds,
            max_train_rows,
            state_only,
            out.as_deref(),
            threads,
        ),
        TrainCommand::HookPlay {
            actor,
            arenas,
            games,
            opponents,
            seed,
            arenas_dir,
            map_dir,
            flyg,
            threads,
        } => hook_play_cmd(
            &actor,
            &arenas,
            games,
            &opponents,
            seed,
            &arenas_dir,
            map_dir,
            flyg,
            threads,
        ),
        TrainCommand::Es(a) => crate::es_cmd::run(a),
        TrainCommand::Ppo(a) => crate::ppo_cmd::run(a),
        TrainCommand::UpgradeBundle {
            bundle,
            opponent_state,
            out,
            flyg,
        } => upgrade_bundle_cmd(&bundle, &opponent_state, &out, flyg.as_deref()),
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

fn upgrade_bundle_cmd(
    bundle: &std::path::Path,
    opponent_state: &std::path::Path,
    out: &std::path::Path,
    flyg: Option<&std::path::Path>,
) -> Result<(), String> {
    let b = ddai_fly::bundle::load_bundle(bundle).map_err(|e| e.to_string())?;
    let flyg_path = flyg.map_or_else(|| PathBuf::from(&b.flyg_path_hint), PathBuf::from);
    let sha = ddai_fly::bundle::sha256_hex_of_file(&flyg_path).map_err(|e| e.to_string())?;
    if sha != b.flyg_sha256 {
        return Err(format!(
            "{}: not the graph {} was built for",
            flyg_path.display(),
            bundle.display()
        ));
    }
    let g = ddai_flyg::load(&flyg_path).map_err(|e| format!("{}: {e}", flyg_path.display()))?;
    let text = std::fs::read_to_string(opponent_state).map_err(|e| format!("{}: {e}", opponent_state.display()))?;
    let up = ddai_fly::bundle::upgrade_with_opponent_state(&b, g, &text).map_err(|e| e.to_string())?;
    ddai_fly::bundle::save_bundle(out, &up).map_err(|e| e.to_string())?;
    eprintln!(
        "{}: encoder parameters {} -> {} (the new ones are zero), same decisions as {}",
        out.display(),
        b.encoder_params.g.len(),
        up.encoder_params.g.len(),
        bundle.display()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn set_thresholds_cmd(
    kind: &str,
    bundle: &std::path::Path,
    out: &std::path::Path,
    jump: Option<f32>,
    hook: Option<f32>,
    fire: Option<f32>,
    hysteresis: Option<(f32, f32)>,
    plain_hook: bool,
) -> Result<(), String> {
    let apply = |t: &mut ddai_fly::bc::HeadThresholds| -> Result<(), String> {
        t.jump = jump.unwrap_or(t.jump);
        t.hook = hook.unwrap_or(t.hook);
        t.fire = fire.unwrap_or(t.fire);
        t.validate()?;
        eprintln!("thresholds jump/hook/fire: {:.3}/{:.3}/{:.3}", t.jump, t.hook, t.fire);
        Ok(())
    };
    match kind {
        "fly" => {
            let mut b = ddai_fly::bundle::load_bundle(bundle).map_err(|e| e.to_string())?;
            apply(&mut b.thresholds)?;
            if let Some((hi, lo)) = hysteresis {
                b.hook_decode = ddai_fly::bc::HookDecode::Latched { hi, lo };
            } else if plain_hook {
                b.hook_decode = ddai_fly::bc::HookDecode::Plain;
            }
            b.hook_decode.validate()?;
            eprintln!("hook decode: {:?}", b.hook_decode);
            ddai_fly::bundle::save_bundle(out, &b).map_err(|e| e.to_string())
        }
        "mlp" | "gru" if hysteresis.is_some() || plain_hook => {
            Err("the hysteresis hook decode is for fly checkpoints only".to_string())
        }
        "mlp" | "gru" => {
            let mut b = ddai_controls::bundle::load_control_bundle(bundle).map_err(|e| e.to_string())?;
            apply(&mut b.thresholds)?;
            ddai_controls::bundle::save_control_bundle(out, &b).map_err(|e| e.to_string())
        }
        other => Err(format!("unknown kind {other:?} (fly, mlp, gru)")),
    }
}

#[allow(clippy::too_many_arguments)]
fn probe_hook_cmd(
    config: &std::path::Path,
    flies: &[String],
    state_flies: &[String],
    first: usize,
    frames: usize,
    seeds: usize,
    max_train_rows: usize,
    state_only: bool,
    out: Option<&std::path::Path>,
    threads: usize,
) -> Result<(), String> {
    let text = std::fs::read_to_string(config).map_err(|e| format!("{}: {e}", config.display()))?;
    let cfg: ddai_train::runner::ExperimentConfig =
        toml::from_str(&text).map_err(|e| format!("{}: {e}", config.display()))?;
    let env = ddai_train::experiment::load_env_with_scenarios(
        &ddai_train::experiment::expand_home(&cfg.arenas_dir),
        &ddai_train::experiment::expand_home(&cfg.map_dir),
        Some(ddai_train::experiment::expand_home(&cfg.flyg)),
        cfg.scenarios_dir
            .as_deref()
            .map(ddai_train::experiment::expand_home)
            .as_deref(),
    )?;
    let parse = |list: &[String]| -> Result<Vec<(String, PathBuf)>, String> {
        list.iter()
            .map(|f| {
                let (n, p) = f
                    .split_once('=')
                    .ok_or_else(|| format!("{f:?}: expected name=bundle"))?;
                Ok((n.to_string(), ddai_train::experiment::expand_home(p)))
            })
            .collect()
    };
    let (bundles, state_bundles) = (parse(flies)?, parse(state_flies)?);
    let spec = ddai_train::probe::ProbeSpec {
        first,
        frames,
        seeds,
        max_train_rows,
        state_only,
        no_membrane: state_only,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let res = ddai_train::probe::run_probe(
        &cfg,
        &env,
        &spec,
        &bundles,
        &state_bundles,
        threads.clamp(1, 3),
        &mut |l| eprintln!("[{:5.0}s] {l}", t0.elapsed().as_secs_f64()),
    )?;
    for r in &res {
        println!(
            "latch {} ({}): {} train rows, {} validation rows, label (key pressed) rate {:.3} / {:.3}",
            r.latch,
            if r.latch {
                "held: keep or release"
            } else {
                "released: press"
            },
            r.n_train,
            r.n_val,
            r.label_rate_train,
            r.label_rate_val
        );
        for (n, m, _, ci) in &r.probes {
            println!("   probe {n}: AUROC {m:.3} [{:.3}; {:.3}]", ci[0], ci[1]);
        }
        for (n, a, ci) in &r.flies {
            println!("   fly {n}: AUROC {a:.3} [{:.3}; {:.3}]", ci[0], ci[1]);
        }
    }
    let json = serde_json::to_string_pretty(&(spec, res)).map_err(|e| e.to_string())?;
    if let Some(p) = out {
        std::fs::write(p, json).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn upgrade_hook_cmd(bundle: &std::path::Path, out: &std::path::Path) -> Result<(), String> {
    let b = ddai_fly::bundle::load_bundle(bundle).map_err(|e| e.to_string())?;
    if b.hook_param == ddai_fly::bc::HookParam::Intent {
        return Err("the checkpoint already has an intent hook head".to_string());
    }
    let up = ddai_fly::bundle::upgrade_to_intent_hook(&b);
    ddai_fly::bundle::save_bundle(out, &up).map_err(|e| e.to_string())?;
    eprintln!(
        "wrote {} (intent hook head; decode {:?})",
        out.display(),
        up.hook_decode
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn hook_play_cmd(
    actor: &str,
    arenas: &str,
    games: u32,
    opponents: &str,
    seed: u64,
    arenas_dir: &std::path::Path,
    map_dir: Option<PathBuf>,
    flyg: Option<PathBuf>,
    threads: usize,
) -> Result<(), String> {
    let map_dir = map_dir.unwrap_or_else(|| {
        std::env::var_os("HOME").map_or_else(|| PathBuf::from("maps"), |h| PathBuf::from(h).join("aiddnet/data/maps"))
    });
    let env = ddai_train::experiment::load_env(arenas_dir, &map_dir, flyg)?;
    let split = |s: &str| {
        s.split(',')
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect::<Vec<_>>()
    };
    let evs = ddai_train::experiment::hook_play_eval(
        &env,
        actor,
        &split(arenas),
        &split(opponents),
        ddai_train::experiment::HookPlayPlan {
            games,
            base_seed: seed,
            threads: threads.clamp(1, 3),
        },
        &mut |l| eprintln!("{l}"),
    )?;
    println!("{}", serde_json::to_string_pretty(&evs).map_err(|e| e.to_string())?);
    Ok(())
}

fn collect_cmd(path: &std::path::Path, threads: Option<usize>) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut cfg: CollectConfig = toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if let Some(t) = threads {
        cfg.threads = t;
    }
    cfg.threads = cfg.threads.clamp(1, 3);
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
    cfg.train.threads = cfg.train.threads.clamp(1, 3);
    Ok(cfg)
}

fn run_cmd(path: &std::path::Path, threads: Option<usize>, allow_config_change: bool) -> Result<(), String> {
    let cfg = load_experiment(path, threads)?;
    let t0 = std::time::Instant::now();
    run_experiment_with(&cfg, allow_config_change, &mut |l| {
        eprintln!("[{:8.1}s] {l}", t0.elapsed().as_secs_f64());
    })?;
    println!(
        "{}: done in {:.0} s, run directory {}",
        cfg.name,
        t0.elapsed().as_secs_f64(),
        cfg.run_dir
    );
    Ok(())
}

fn eval_cmd(
    path: &std::path::Path,
    bundle: &std::path::Path,
    calibrate: bool,
    write_calibrated: Option<&std::path::Path>,
) -> Result<(), String> {
    let cfg = load_experiment(path, None)?;
    let recs = eval_bundle_offline(&cfg, bundle, calibrate, write_calibrated, &mut |l| eprintln!("{l}"))?;
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
        "round | episodes W:L:D:T | unfrozen steps | student steps | agree dir/jump/hook/fire | student rate jump/hook/fire | teacher rate on the same states | hook start (own hook not out) student/teacher | hook release (own hook out) student/teacher | own hook out | teacher keeps hook out: flying / on a player / on terrain"
    );
    for (r, a) in by_round {
        let f = |x: u64, n: u64| if n == 0 { 0.0 } else { 100.0 * x as f64 / n as f64 };
        let h = a.hook.report();
        let p = |x: Option<f64>| x.map_or_else(|| "-".to_string(), |v| format!("{:.1}", 100.0 * v));
        let keep = |c: &ddai_train::play_stats::HookCounts| {
            if c.n == 0 {
                "-".to_string()
            } else {
                format!("{:.0}% of {}", 100.0 * c.teacher_hook as f64 / c.n as f64, c.n)
            }
        };
        println!(
            "{r:>5} | {} {}:{}:{}:{} | {} | {} | {:.0}/{:.0}/{:.0}/{:.0}% | {:.1}/{:.1}/{:.1}% | {:.1}/{:.1}/{:.1}% | {}/{}% | {}/{}% | {:.0}% | {} / {} / {}",
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
            keep(&a.hook.out_flying),
            keep(&a.hook.out_player),
            keep(&a.hook.out_terrain),
        );
    }
    Ok(())
}
