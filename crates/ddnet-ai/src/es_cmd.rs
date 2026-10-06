//! `ddnet-ai train es ...`: OpenAI-ES on the outcome of the held block (task 8.5a, `ddai_train::es`) and its tools.
//!
//! * `train es bank`: build the post-freeze start bank from arena games;
//! * `train es run`: run (or resume) the ES;
//! * `train es eval`: a brain on the config's fixed evaluation episodes (baselines, the final check), JSON with the per-item outcomes;
//! * `train es compare`: two such files, paired (McNemar);
//! * `train es scan`: how far a perturbation of each parameter group moves the fitness (the measurement behind the sigmas).

use std::path::PathBuf;

use clap::{Args, Subcommand};
use ddai_train::bank::{BankBuildSpec, build_bank};
use ddai_train::es::eval::Rate;
use ddai_train::es::stats::paired;
use ddai_train::es::{EsConfig, EvalPoint, evaluate_player, load, run_es, scan_sigmas};
use ddai_train::experiment::{expand_home, load_env};

#[derive(Debug, Args)]
pub struct EsArgs {
    #[command(subcommand)]
    pub command: EsCommand,
}

#[derive(Debug, Subcommand)]
pub enum EsCommand {
    /// Builds the post-freeze start bank from games of the given blockers against the scripted bot.
    Bank {
        #[arg(long)]
        out: PathBuf,
        /// Comma-separated arenas.
        #[arg(long)]
        arenas: String,
        /// `<brain>=<games per arena>`, repeatable: `scripted=200`, `planner=100`, `fly:<bundle>=200`.
        #[arg(long = "blocker", required = true)]
        blockers: Vec<String>,
        #[arg(long, default_value_t = 5_000_000_000)]
        seed: u64,
        #[arg(long, default_value_t = 250)]
        window: i32,
        #[arg(long, default_value = "configs/arenas")]
        arenas_dir: PathBuf,
        #[arg(long)]
        map_dir: Option<PathBuf>,
        #[arg(long)]
        flyg: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        threads: usize,
        /// Add the new starts to the bank already at `--out` (same window; give other seeds or arenas than the first build).
        #[arg(long)]
        extend: bool,
    },
    /// Labels post-freeze starts with the planner (the teacher) into a teacher dataset the BC trainer reads (the imitation arm).
    Collect {
        #[arg(long)]
        bank: PathBuf,
        /// The dataset directory (created or extended).
        #[arg(long)]
        out: PathBuf,
        /// Who plays after the freeze: `teacher` or `fly:<bundle>` (DAgger).
        #[arg(long, default_value = "teacher")]
        actor: String,
        #[arg(long, default_value_t = 0.0)]
        beta: f32,
        /// Comma-separated training arenas.
        #[arg(long, default_value = "clb-left,pit,platform")]
        arenas: String,
        /// Only the training part of the bank, only starts where an idle blocker would not hold.
        #[arg(long)]
        escapable_only: bool,
        #[arg(long, default_value_t = 1)]
        round: u32,
        #[arg(long, default_value_t = 250)]
        window: i32,
        #[arg(long, default_value_t = 50)]
        burn_in: i32,
        #[arg(long, default_value = "configs/arenas")]
        arenas_dir: PathBuf,
        #[arg(long)]
        map_dir: Option<PathBuf>,
        #[arg(long)]
        flyg: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        threads: usize,
    },
    /// Runs (or resumes) the ES.
    Run {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        threads: Option<usize>,
        /// Resume even though the configuration differs from the one the run started with.
        #[arg(long)]
        allow_config_change: bool,
    },
    /// Evaluates a brain (`idle`, `scripted`, `planner`, `fly:<bundle>`) on the config's evaluation episodes.
    Eval {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        brain: String,
        #[arg(long)]
        out: PathBuf,
        /// Overrides `eval.starts` / `eval.games` / `eval.seed_base`.
        #[arg(long)]
        starts: Option<usize>,
        #[arg(long)]
        games: Option<u32>,
        #[arg(long)]
        seed_base: Option<u64>,
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Re-tags a bank (a version 1 file, or one made before the tags changed): what an idle and a scripted blocker do from the freeze on,
    /// including whether the victim escapes after the idle blocker went out. Writes a version 2 bank.
    Retag {
        #[arg(long)]
        bank: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "configs/arenas")]
        arenas_dir: PathBuf,
        #[arg(long)]
        map_dir: Option<PathBuf>,
        #[arg(long)]
        flyg: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        threads: usize,
    },
    /// Splits the held share of `es eval` files by start type, from a (re-tagged) bank: starts where the victim escapes under an idle
    /// blocker, starts where only the idle blocker falls, the rest; own freezes next to it; paired against the first file.
    Breakdown {
        #[arg(long)]
        config: PathBuf,
        /// The re-tagged bank (default: the config's).
        #[arg(long)]
        bank: Option<PathBuf>,
        /// The `--starts` the files were made with (default: the config's `eval.starts`).
        #[arg(long)]
        starts: Option<usize>,
        files: Vec<PathBuf>,
    },
    /// Compares two `es eval` files item by item (paired, exact McNemar).
    Compare { a: PathBuf, b: PathBuf },
    /// Perturbs each parameter group alone at several scales and reports how the fitness and the held share move.
    Scan {
        #[arg(long)]
        config: PathBuf,
        /// Comma-separated scales relative to the config's sigma of the group.
        #[arg(long, default_value = "0.25,0.5,1,2,4")]
        scales: String,
        #[arg(long, default_value_t = 6)]
        perturbations: usize,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        threads: Option<usize>,
    },
}

pub fn run(args: EsArgs) -> Result<(), String> {
    match args.command {
        EsCommand::Bank {
            out,
            arenas,
            blockers,
            seed,
            window,
            arenas_dir,
            map_dir,
            flyg,
            threads,
            extend,
        } => {
            let map_dir = map_dir.unwrap_or_else(|| {
                std::env::var_os("HOME")
                    .map_or_else(|| PathBuf::from("maps"), |h| PathBuf::from(h).join("aiddnet/data/maps"))
            });
            let env = load_env(&expand_home(&arenas_dir.to_string_lossy()), &map_dir, flyg)?;
            let blockers = blockers
                .iter()
                .map(|b| {
                    let (arg, n) = b
                        .rsplit_once('=')
                        .ok_or_else(|| format!("--blocker {b:?}: expected <brain>=<games>"))?;
                    Ok((
                        arg.to_string(),
                        n.parse::<u32>().map_err(|e| format!("--blocker {b:?}: {e}"))?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let spec = BankBuildSpec {
                arenas: arenas
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
                blockers,
                base_seed: seed,
                window_ticks: window,
                threads: threads.clamp(1, 3),
            };
            let t0 = std::time::Instant::now();
            let mut bank = build_bank(&env, &spec, &mut |l| {
                eprintln!("[{:6.0}s] {l}", t0.elapsed().as_secs_f64())
            })?;
            if extend && out.exists() {
                let mut old = ddai_train::bank::Bank::load(&out)?;
                if old.window_ticks != bank.window_ticks || old.rules != bank.rules {
                    return Err("--extend: the existing bank was made with other rules or another window".into());
                }
                old.notes = format!("{}; {}", old.notes, bank.notes);
                old.starts.append(&mut bank.starts);
                bank = old;
            }
            bank.save(&out)?;
            let mut by: std::collections::BTreeMap<(String, String), (u32, u32, u32)> =
                std::collections::BTreeMap::new();
            for s in &bank.starts {
                let e = by.entry((s.arena.clone(), s.blocker.clone())).or_default();
                e.0 += 1;
                e.1 += u32::from(s.idle_held);
                e.2 += u32::from(s.scripted_held);
            }
            println!("{} starts in {}", bank.starts.len(), out.display());
            println!("arena / blocker: starts, idle holds, scripted holds");
            for ((a, b), (n, i, s)) in by {
                println!("  {a} / {b}: {n}, {i}, {s}");
            }
            Ok(())
        }
        EsCommand::Collect {
            bank,
            out,
            actor,
            beta,
            arenas,
            escapable_only,
            round,
            window,
            burn_in,
            arenas_dir,
            map_dir,
            flyg,
            threads,
        } => {
            let map_dir = map_dir.unwrap_or_else(|| {
                std::env::var_os("HOME")
                    .map_or_else(|| PathBuf::from("maps"), |h| PathBuf::from(h).join("aiddnet/data/maps"))
            });
            let env = load_env(&expand_home(&arenas_dir.to_string_lossy()), &map_dir, flyg)?;
            let bank = ddai_train::bank::Bank::load(&bank)?;
            let arenas: Vec<String> = arenas
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let starts = bank.select(&arenas, Some(false), escapable_only);
            let mut store = ddai_train::store::TeacherStore::open_or_create(
                &expand_home(&out.to_string_lossy()),
                "post-freeze",
                "8.5a",
            )
            .map_err(|e| e.to_string())?;
            let t0 = std::time::Instant::now();
            let s = ddai_train::bank_collect::collect_starts(
                &env,
                &mut store,
                &starts,
                &bank.rules,
                &actor,
                ddai_train::collect::Mixing {
                    beta,
                    ..Default::default()
                },
                window,
                burn_in,
                round,
                threads.clamp(1, 3),
                &mut |l| eprintln!("[{:5.0}s] {l}", t0.elapsed().as_secs_f64()),
            )?;
            println!("{} episodes, {} decisions", s.games, s.steps);
            Ok(())
        }
        EsCommand::Run {
            config,
            threads,
            allow_config_change,
        } => {
            let text = std::fs::read_to_string(&config).map_err(|e| format!("{}: {e}", config.display()))?;
            let mut cfg = EsConfig::parse(&text).map_err(|e| format!("{}: {e}", config.display()))?;
            if let Some(t) = threads {
                cfg.threads = t;
            }
            let t0 = std::time::Instant::now();
            run_es(&cfg, allow_config_change, &mut |l| {
                eprintln!("[{:7.0}s] {l}", t0.elapsed().as_secs_f64())
            })?;
            Ok(())
        }
        EsCommand::Eval {
            config,
            brain,
            out,
            starts,
            games,
            seed_base,
            threads,
        } => {
            let text = std::fs::read_to_string(&config).map_err(|e| format!("{}: {e}", config.display()))?;
            let mut cfg = EsConfig::parse(&text)?;
            if let Some(v) = starts {
                cfg.eval.starts = v;
            }
            if let Some(v) = games {
                cfg.eval.games = v;
            }
            if let Some(v) = seed_base {
                cfg.eval.seed_base = v;
            }
            if let Some(t) = threads {
                cfg.threads = t;
            }
            let loaded = load(&cfg)?;
            let t0 = std::time::Instant::now();
            let p = evaluate_player(&cfg, &loaded, &brain, 0)?;
            eprintln!("[{:.0}s] {brain}", t0.elapsed().as_secs_f64());
            print_point(&brain, &p);
            std::fs::write(
                &out,
                serde_json::to_string(&(brain.clone(), &p)).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())
        }
        EsCommand::Retag {
            bank,
            out,
            arenas_dir,
            map_dir,
            flyg,
            threads,
        } => {
            let map_dir = map_dir.unwrap_or_else(|| {
                std::env::var_os("HOME")
                    .map_or_else(|| PathBuf::from("maps"), |h| PathBuf::from(h).join("aiddnet/data/maps"))
            });
            let env = load_env(&expand_home(&arenas_dir.to_string_lossy()), &map_dir, flyg)?;
            let mut b = ddai_train::bank::Bank::load(&bank)?;
            let t0 = std::time::Instant::now();
            ddai_train::bank::retag_bank(&env, &mut b, threads.clamp(1, 3))?;
            b.save(&out)?;
            let n = b.starts.len();
            let count = |f: &dyn Fn(&ddai_train::bank::BankStart) -> bool| b.starts.iter().filter(|s| f(s)).count();
            println!(
                "{n} starts re-tagged in {:.0}s -> {}",
                t0.elapsed().as_secs_f64(),
                out.display()
            );
            println!("  idle holds the block:                    {}", count(&|s| s.idle_held));
            println!(
                "  victim escapes under an idle blocker:    {}",
                count(&|s| s.victim_escapes_under_idle == Some(true))
            );
            println!(
                "  idle blocker goes out:                   {}",
                count(&|s| s.idle_blocker_out == Some(true))
            );
            println!(
                "  idle blocker out and victim stays (the old 'escapable' minus the escapes): {}",
                count(&|s| s.idle_blocker_out == Some(true) && s.victim_escapes_under_idle == Some(false))
            );
            Ok(())
        }
        EsCommand::Breakdown {
            config,
            bank,
            starts,
            files,
        } => {
            let text = std::fs::read_to_string(&config).map_err(|e| format!("{}: {e}", config.display()))?;
            let cfg = EsConfig::parse(&text)?;
            let bank_path = bank.unwrap_or_else(|| expand_home(&cfg.bank));
            let bank = ddai_train::bank::Bank::load(&bank_path)?;
            let n = starts.unwrap_or(cfg.eval.starts);
            let sets: [(&str, Vec<&ddai_train::bank::BankStart>); 2] = [
                (
                    "train-val",
                    ddai_train::es::spread(&bank.select(&cfg.train_arenas, Some(true), false), n),
                ),
                (
                    "holdout",
                    ddai_train::es::spread(&bank.select(&cfg.eval.holdout_arenas, None, false), n),
                ),
            ];
            let pts: Vec<(String, EvalPoint)> = files
                .iter()
                .map(|p| {
                    serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?)
                        .map_err(|e| format!("{}: {e}", p.display()))
                })
                .collect::<Result<_, _>>()?;
            for (name, list) in &sets {
                if list.iter().any(|s| s.victim_escapes_under_idle.is_none()) {
                    return Err("the bank is not re-tagged (`train es retag`)".into());
                }
                // Classes of start: V = the victim escapes under an idle blocker; B = it does not and the idle blocker falls (the pilot's
                // "escapable" was V + B); H = the idle blocker holds the block.
                let class: Vec<char> = list
                    .iter()
                    .map(|s| {
                        if s.victim_escapes_under_idle == Some(true) {
                            'V'
                        } else if s.idle_held {
                            'H'
                        } else {
                            'B'
                        }
                    })
                    .collect();
                let cnt = |c: char| class.iter().filter(|&&x| x == c).count();
                println!(
                    "\n== {name}: {} starts: victim escapes under idle (V) {}, only the idle blocker falls (B) {}, idle holds (H) {}",
                    list.len(),
                    cnt('V'),
                    cnt('B'),
                    cnt('H')
                );
                let rate = |items: &[bool], c: &[char], want: &[char]| -> Rate {
                    if items.len() != c.len() {
                        return Rate::new(0, 0); // no per-start items in this file
                    }
                    let sel: Vec<bool> = items
                        .iter()
                        .zip(c)
                        .filter(|(_, k)| want.contains(k))
                        .map(|(x, _)| *x)
                        .collect();
                    Rate::new(sel.iter().filter(|&&x| x).count() as u32, sel.len() as u32)
                };
                for (label, p) in &pts {
                    let sm = if *name == "train-val" {
                        Some(&p.train_starts)
                    } else {
                        p.holdout_starts.as_ref()
                    };
                    let Some(sm) = sm else { continue };
                    if sm.held_items.len() != list.len() {
                        return Err(format!(
                            "{label}: {} items for {name}, the bank gives {} starts (wrong --starts or bank?)",
                            sm.held_items.len(),
                            list.len()
                        ));
                    }
                    println!(
                        "  {label}\n    held on V (PRIMARY, victim-escapable) {} | on B {} | on H {} | all {}\n    own freeze in window: on V {} | on B {} | on H {} | all {}",
                        rate(&sm.held_items, &class, &['V']).fmt_pct(),
                        rate(&sm.held_items, &class, &['B']).fmt_pct(),
                        rate(&sm.held_items, &class, &['H']).fmt_pct(),
                        sm.held.fmt_pct(),
                        rate(&sm.self_freeze_items, &class, &['V']).fmt_pct(),
                        rate(&sm.self_freeze_items, &class, &['B']).fmt_pct(),
                        rate(&sm.self_freeze_items, &class, &['H']).fmt_pct(),
                        sm.self_freeze.fmt_pct()
                    );
                }
                if let Some((l0, p0)) = pts.first() {
                    for (l1, p1) in &pts[1..] {
                        let (a, b) = if *name == "train-val" {
                            (Some(&p0.train_starts), Some(&p1.train_starts))
                        } else {
                            (p0.holdout_starts.as_ref(), p1.holdout_starts.as_ref())
                        };
                        let (Some(a), Some(b)) = (a, b) else { continue };
                        if a.self_freeze_items.len() != class.len() || b.self_freeze_items.len() != class.len() {
                            continue;
                        }
                        for (what, wa, wb) in [
                            ("held on V", &a.held_items, &b.held_items),
                            ("own freeze on V", &a.self_freeze_items, &b.self_freeze_items),
                        ] {
                            let pick = |v: &[bool]| -> Vec<bool> {
                                v.iter()
                                    .zip(&class)
                                    .filter(|(_, c)| **c == 'V')
                                    .map(|(x, _)| *x)
                                    .collect()
                            };
                            let (x, y) = (pick(wa), pick(wb));
                            let (oa, ob, pv) = paired(&x, &y);
                            println!(
                                "  paired {l0} -> {l1}, {what}: {:+.1} pp (only first {oa}, only second {ob}, McNemar p = {pv:.4})",
                                100.0
                                    * (y.iter().filter(|&&v| v).count() as f64
                                        - x.iter().filter(|&&v| v).count() as f64)
                                    / x.len().max(1) as f64
                            );
                        }
                    }
                }
            }
            Ok(())
        }
        EsCommand::Compare { a, b } => {
            let read = |p: &PathBuf| -> Result<(String, EvalPoint), String> {
                serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?)
                    .map_err(|e| format!("{}: {e}", p.display()))
            };
            let ((la, pa), (lb, pb)) = (read(&a)?, read(&b)?);
            match (&pa.spec, &pb.spec) {
                (Some(x), Some(y)) if x != y => {
                    return Err(format!(
                        "the two files were not made on the same episodes (seed base, games, starts, halls, window or bank differ):\nA {x:?}\nB {y:?}"
                    ));
                }
                (Some(_), Some(_)) => {}
                _ => eprintln!(
                    "note: a file without an eval spec (made before 8.5a's review): the episodes are not checked, only the lengths"
                ),
            }
            println!("A = {la}\nB = {lb}\n");
            let row = |name: &str, x: &[bool], y: &[bool]| {
                if x.is_empty() || x.len() != y.len() {
                    return;
                }
                let (ra, rb) = (
                    Rate::new(x.iter().filter(|&&v| v).count() as u32, x.len() as u32),
                    Rate::new(y.iter().filter(|&&v| v).count() as u32, y.len() as u32),
                );
                let (only_a, only_b, p) = paired(x, y);
                println!(
                    "{name:46} A {}  B {}  B-A {:+.1} pp  (only A {only_a}, only B {only_b}, McNemar p = {p:.4})",
                    ra.fmt_pct(),
                    rb.fmt_pct(),
                    100.0 * (rb.p - ra.p)
                );
            };
            let only = |v: &[bool], mask: &[bool]| -> Vec<bool> {
                v.iter().zip(mask).filter(|(_, m)| **m).map(|(x, _)| *x).collect()
            };
            for (name, x, y) in [
                ("train-val", Some(&pa.train_starts), Some(&pb.train_starts)),
                ("holdout", pa.holdout_starts.as_ref(), pb.holdout_starts.as_ref()),
            ] {
                let (Some(x), Some(y)) = (x, y) else { continue };
                if !x.victim_escape_items.is_empty() && x.victim_escape_items == y.victim_escape_items {
                    row(
                        &format!("{name} starts: held, victim-escapable (PRIMARY)"),
                        &only(&x.held_items, &x.victim_escape_items),
                        &only(&y.held_items, &y.victim_escape_items),
                    );
                    row(
                        &format!("{name} starts: own freeze, victim-escapable"),
                        &only(&x.self_freeze_items, &x.victim_escape_items),
                        &only(&y.self_freeze_items, &y.victim_escape_items),
                    );
                }
                if x.escapable_items == y.escapable_items {
                    row(
                        &format!("{name} starts: held, idle-does-not-hold (the pilot's 'escapable')"),
                        &only(&x.held_items, &x.escapable_items),
                        &only(&y.held_items, &y.escapable_items),
                    );
                }
                row(&format!("{name} starts: held, all"), &x.held_items, &y.held_items);
                row(
                    &format!("{name} starts: own freeze in window"),
                    &x.self_freeze_items,
                    &y.self_freeze_items,
                );
            }
            row(
                "train games: first freeze",
                &pa.train_games.credited_items,
                &pb.train_games.credited_items,
            );
            row(
                "train games: held win",
                &pa.train_games.credited_held_items,
                &pb.train_games.credited_held_items,
            );
            if let (Some(x), Some(y)) = (&pa.holdout_games, &pb.holdout_games) {
                row("holdout games: first freeze", &x.credited_items, &y.credited_items);
                row(
                    "holdout games: held win",
                    &x.credited_held_items,
                    &y.credited_held_items,
                );
            }
            Ok(())
        }
        EsCommand::Scan {
            config,
            scales,
            perturbations,
            out,
            threads,
        } => {
            let text = std::fs::read_to_string(&config).map_err(|e| format!("{}: {e}", config.display()))?;
            let mut cfg = EsConfig::parse(&text)?;
            if let Some(t) = threads {
                cfg.threads = t;
            }
            let scales: Vec<f32> = scales
                .split(',')
                .map(|s| s.trim().parse::<f32>().map_err(|e| format!("--scales: {e}")))
                .collect::<Result<_, _>>()?;
            let loaded = load(&cfg)?;
            let rows = scan_sigmas(&cfg, &loaded, &scales, perturbations, &mut |l| eprintln!("{l}"))?;
            let json = serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?;
            match out {
                Some(p) => std::fs::write(p, json).map_err(|e| e.to_string()),
                None => {
                    println!("{json}");
                    Ok(())
                }
            }
        }
    }
}

fn print_point(label: &str, p: &EvalPoint) {
    println!("{label}");
    println!(
        "  held, train-hall validation starts: {}",
        p.train_starts.held.fmt_pct()
    );
    println!(
        "    where idle would not hold:        {}",
        p.train_starts.held_escapable.fmt_pct()
    );
    println!(
        "    own freezes in the window:        {}",
        p.train_starts.self_freeze.fmt_pct()
    );
    if let Some(h) = &p.holdout_starts {
        println!("  held, holdout starts:               {}", h.held.fmt_pct());
        println!("    where idle would not hold:        {}", h.held_escapable.fmt_pct());
        println!("    own freezes in the window:        {}", h.self_freeze.fmt_pct());
    }
    println!(
        "  first freeze (credited), train halls: {}",
        p.train_games.credited.fmt_pct()
    );
    println!(
        "    credited and held:                  {}",
        p.train_games.credited_held.fmt_pct()
    );
    println!(
        "    W:L:D:T {}:{}:{}:{}",
        p.train_games.w, p.train_games.l, p.train_games.d, p.train_games.t
    );
    if let Some(g) = &p.holdout_games {
        println!("  first freeze (credited), holdout:     {}", g.credited.fmt_pct());
        println!("    credited and held:                  {}", g.credited_held.fmt_pct());
        println!("    W:L:D:T {}:{}:{}:{}", g.w, g.l, g.d, g.t);
    }
}
