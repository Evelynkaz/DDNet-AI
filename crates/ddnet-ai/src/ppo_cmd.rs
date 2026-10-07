//! `ddnet-ai train ppo ...`: recurrent PPO on the outcome of the held block (task 8.5b, `ddai_train::ppo`).
//!
//! * `train ppo run`: run (or resume) the PPO of a config; its checkpoints are ordinary fly bundles.
//!
//! The bank, the evaluation of a bundle (`train es eval`) and the paired comparison (`train es compare`) are those of the ES task: the same
//! episodes and seeds, so a PPO bundle is measured exactly like the ES ones and the start.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use ddai_train::ppo::{PpoConfig, ensure_demos, load, run_ppo, training_starts};

#[derive(Debug, Args)]
pub struct PpoArgs {
    #[command(subcommand)]
    pub command: PpoCommand,
}

#[derive(Debug, Subcommand)]
pub enum PpoCommand {
    /// Runs (or resumes) the PPO.
    Run {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        threads: Option<usize>,
        /// Resume even though the configuration differs from the one the run started with.
        #[arg(long)]
        allow_config_change: bool,
    },
    /// What a brain (`idle`, `scripted`, `planner`, `fly:<bundle>`) holds after the hand-over from the planner's demonstration at each offset
    /// (ticks of the planner's play): the ladder of the reverse curriculum read from outside.
    Probe {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        brain: String,
        /// Comma-separated offsets in ticks.
        #[arg(long, default_value = "0,50,100,150,200")]
        offsets: String,
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Records the planner's demonstrations of the reverse curriculum (what `run` does on its first start when the file is missing).
    Demos {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        threads: Option<usize>,
    },
}

pub fn run(args: PpoArgs) -> Result<(), String> {
    match args.command {
        PpoCommand::Run {
            config,
            threads,
            allow_config_change,
        } => {
            let text = std::fs::read_to_string(&config).map_err(|e| format!("{}: {e}", config.display()))?;
            let mut cfg = PpoConfig::parse(&text).map_err(|e| format!("{}: {e}", config.display()))?;
            if let Some(t) = threads {
                cfg.threads = t;
            }
            cfg.validate()?;
            let t0 = std::time::Instant::now();
            run_ppo(&cfg, allow_config_change, &mut |l| {
                eprintln!("[{:7.0}s] {l}", t0.elapsed().as_secs_f64())
            })?;
            Ok(())
        }
        PpoCommand::Probe {
            config,
            brain,
            offsets,
            threads,
        } => {
            let text = std::fs::read_to_string(&config).map_err(|e| format!("{}: {e}", config.display()))?;
            let mut cfg = PpoConfig::parse(&text).map_err(|e| format!("{}: {e}", config.display()))?;
            if let Some(t) = threads {
                cfg.threads = t;
            }
            cfg.validate()?;
            if !cfg.curriculum.enabled {
                return Err("the config has no [curriculum] enabled = true".into());
            }
            let offsets: Vec<i32> = offsets
                .split(',')
                .map(|s| s.trim().parse::<i32>().map_err(|e| format!("--offsets: {e}")))
                .collect::<Result<_, _>>()?;
            let loaded = load(&cfg)?;
            let pool = ddai_train::es::make_pool(cfg.threads)?;
            let starts = training_starts(&cfg, &loaded);
            let demos =
                ensure_demos(&cfg, &loaded, &starts, &pool, &mut |l| eprintln!("{l}"))?.expect("curriculum enabled");
            let spec = ddai_env::models::player_from_arg(&brain);
            let factory = loaded.env.models.factory();
            let maker = || factory(&spec);
            let rows = ddai_train::ppo::curriculum::probe_offsets(
                &loaded.env,
                &loaded.bank,
                &starts,
                &demos,
                &offsets,
                &maker,
                cfg.rollout.burn_in_ticks,
                &pool,
            )?;
            println!(
                "{brain}: held share after the hand-over (starts with a held demonstration), by class V / B / H; the focal player out in the window in brackets"
            );
            for (offset, by) in rows {
                let cell = |(n, h, o): (usize, usize, usize)| {
                    format!(
                        "{:5.1}% ({h}/{n}) [out {:4.1}%]",
                        100.0 * h as f64 / n.max(1) as f64,
                        100.0 * o as f64 / n.max(1) as f64
                    )
                };
                println!(
                    "offset {offset:3}: V {} | B {} | H {}",
                    cell(by[0]),
                    cell(by[1]),
                    cell(by[2])
                );
            }
            Ok(())
        }
        PpoCommand::Demos { config, threads } => {
            let text = std::fs::read_to_string(&config).map_err(|e| format!("{}: {e}", config.display()))?;
            let mut cfg = PpoConfig::parse(&text).map_err(|e| format!("{}: {e}", config.display()))?;
            if let Some(t) = threads {
                cfg.threads = t;
            }
            cfg.validate()?;
            if !cfg.curriculum.enabled {
                return Err("the config has no [curriculum] enabled = true".into());
            }
            let loaded = load(&cfg)?;
            let pool = ddai_train::es::make_pool(cfg.threads)?;
            let starts = training_starts(&cfg, &loaded);
            let t0 = std::time::Instant::now();
            ensure_demos(&cfg, &loaded, &starts, &pool, &mut |l| {
                eprintln!("[{:5.0}s] {l}", t0.elapsed().as_secs_f64())
            })?;
            Ok(())
        }
    }
}
