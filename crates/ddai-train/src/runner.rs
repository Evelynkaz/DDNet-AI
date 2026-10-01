//! A whole experiment: behaviour cloning on the teacher (+ human) data, then DAgger rounds
//! (task 8.2, acceptance criteria 3, 4, 5, 7). One code path for the fly, the MLP and the GRU.
//!
//! ```text
//! round 0   BC on the base teacher datasets + the human mix        -> eval (offline + arena)
//! round r   the current model plays (beta-mixed with the teacher, exploration noise), the
//!           teacher labels every visited state, the data is aggregated, the model is retrained
//!           on everything (a fresh warm-up + cosine phase from the current weights)
//!                                                                  -> eval (offline + arena)
//! ```
//! Everything is resumable: `state.bin` holds parameters, Adam moments and the global step; the
//! phase a step belongs to follows from the config, and a round whose data is already in the
//! DAgger dataset is not collected again.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddai_controls::bundle::{ControlBundle, load_control_bundle};
use ddai_controls::features::input_dim;
use ddai_controls::gru::Gru;
use ddai_controls::mlp::Mlp;
use ddai_env::config::{Condition, PlayerSpec, Rules, RunConfig};
use ddai_env::models::player_from_arg;
use ddai_env::report::summarize;
use ddai_env::run::run_condition;
use ddai_fly::brain_config::parse_brain_config;
use ddai_fly::bundle::{BundleMeta, load_bundle};
use ddai_fly::rng::SplitMix64;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::experiment::{Env, JobConfig, JobSummary, expand_home, job_key, load_env, run_collect_jobs};
use crate::human::{HumanConfig, load_human};
use crate::learner::{ControlLearner, FlyLearner, FlyTrainConfig, Learner};
use crate::play_stats::round_hook_play;
use crate::seq::{Corpus, Seq};
use crate::store::TeacherStore;
use crate::teacher_data::{TeacherDataConfig, TeacherSplit, load_teacher};
use crate::trainer::{EvalSet, RunDir, TrainConfig, TrainError, Trainer};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// `fly`, `mlp` or `gru`.
    pub kind: String,
    /// Hidden size of a control (ignored by the fly).
    #[serde(default)]
    pub hidden: usize,
    /// Learning rate of a control (the fly has its own per-group rates).
    #[serde(default = "default_lr")]
    pub lr: f32,
}

fn default_lr() -> f32 {
    3e-3
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaggerConfig {
    /// `beta` of each round (probability per decision that the teacher's action is played); its
    /// length is the number of rounds.
    #[serde(default)]
    pub betas: Vec<f32>,
    #[serde(default)]
    pub noise_prob: f32,
    pub steps_per_round: u64,
    /// Games played per round (each job's `games`, `beta` is overridden per round).
    #[serde(default, rename = "job")]
    pub jobs: Vec<JobConfig>,
    /// Arena games per evaluation arena after each phase (`0` = no arena evaluation).
    #[serde(default)]
    pub eval_games: u32,
    #[serde(default)]
    pub eval_arenas: Vec<String>,
    #[serde(default = "default_lr_scale")]
    pub retrain_lr_scale: f32,
}

fn default_lr_scale() -> f32 {
    0.5
}

impl Default for DaggerConfig {
    fn default() -> Self {
        DaggerConfig {
            betas: Vec::new(),
            noise_prob: 0.0,
            steps_per_round: 0,
            jobs: Vec::new(),
            eval_games: 0,
            eval_arenas: Vec::new(),
            retrain_lr_scale: default_lr_scale(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub name: String,
    pub model: ModelSpec,
    pub flyg: String,
    pub brain_config: String,
    pub arenas_dir: String,
    pub map_dir: String,
    pub run_dir: String,
    /// Read-only teacher datasets (the shared round-0 data).
    pub teacher_base: Vec<String>,
    /// This run's own DAgger dataset (created and appended).
    pub teacher_dagger: String,
    #[serde(default)]
    pub train: TrainConfig,
    #[serde(default)]
    pub fly: FlyTrainConfig,
    #[serde(default)]
    pub teacher_data: TeacherDataConfig,
    pub human: Option<HumanConfig>,
    pub bc_steps: u64,
    #[serde(default)]
    pub eval_every: u64,
    #[serde(default)]
    pub dagger: DaggerConfig,
}

/// Everything loaded once: corpora and the held-out sets.
pub struct Data {
    pub teacher_train: Vec<Seq>,
    pub human_train: Vec<Seq>,
    pub eval_sets: Vec<EvalSet>,
}

fn by_arena(seqs: Vec<Seq>) -> BTreeMap<String, Vec<Seq>> {
    let mut out: BTreeMap<String, Vec<Seq>> = BTreeMap::new();
    for s in seqs {
        let key = match &s.source {
            crate::seq::Source::Teacher { arena, .. } => arena.clone(),
            crate::seq::Source::Human { .. } => "human".to_string(),
        };
        out.entry(key).or_default().push(s);
    }
    out
}

/// Keeps only the technique-boosted steps of human sequences scored (the rest becomes context).
fn only_boosted(seqs: Vec<Seq>) -> Vec<Seq> {
    seqs.into_iter()
        .map(|mut s| {
            for st in &mut s.steps {
                if st.weight <= 1.0 {
                    st.weight = 0.0;
                }
            }
            s
        })
        .filter(|s| s.steps.iter().any(|st| st.weight > 0.0))
        .collect()
}

fn store_chunks(store: &TeacherStore) -> Vec<usize> {
    store.chunks_of_round(None)
}

/// Loads the base and DAgger teacher data and, when `with_human`, the human data. A DAgger round
/// reloads only the teacher side: the human corpus does not change, and reloading it every round
/// would keep several copies of it alive in the allocator.
pub fn load_data(
    cfg: &ExperimentConfig,
    env: &Env,
    dagger: &TeacherStore,
    with_human: bool,
    log: &mut dyn FnMut(&str),
) -> Result<Data, String> {
    let holdout: HashSet<String> = env.holdout_names();
    let threads = cfg.train.threads;
    let mut teacher_train = Vec::new();
    let mut val = Vec::new();
    let mut holdout_seqs = Vec::new();
    for dir in &cfg.teacher_base {
        let store = TeacherStore::open(&expand_home(dir)).map_err(|e| e.to_string())?;
        let TeacherSplit {
            train,
            val: v,
            holdout: h,
        } = load_teacher(
            &store,
            &store_chunks(&store),
            &env.arenas,
            &holdout,
            &cfg.teacher_data,
            threads,
        )
        .map_err(|e| e.to_string())?;
        log(&format!(
            "teacher base {dir}: {} train / {} val / {} holdout episodes",
            train.len(),
            v.len(),
            h.len()
        ));
        teacher_train.extend(train);
        val.extend(v);
        holdout_seqs.extend(h);
    }
    let TeacherSplit {
        train,
        val: dagger_val,
        holdout: dagger_holdout,
    } = load_teacher(
        dagger,
        &store_chunks(dagger),
        &env.arenas,
        &holdout,
        &cfg.teacher_data,
        threads,
    )
    .map_err(|e| e.to_string())?;
    log(&format!(
        "teacher dagger: {} train / {} val episodes",
        train.len(),
        dagger_val.len()
    ));
    teacher_train.extend(train);
    holdout_seqs.extend(dagger_holdout);

    let mut eval_sets = vec![EvalSet {
        name: "teacher-val".into(),
        corpus: Corpus::uniform(val),
    }];
    for (arena, seqs) in by_arena(holdout_seqs) {
        eval_sets.push(EvalSet {
            name: format!("teacher-holdout:{arena}"),
            corpus: Corpus::uniform(seqs),
        });
    }
    if !dagger_val.is_empty() {
        eval_sets.push(EvalSet {
            name: "dagger-val".into(),
            corpus: Corpus::uniform(dagger_val),
        });
    }
    let mut human_train = Vec::new();
    if let (true, Some(h)) = (with_human, &cfg.human) {
        let mut h = h.clone();
        h.dataset_dirs = h
            .dataset_dirs
            .iter()
            .map(|d| expand_home(&d.to_string_lossy()))
            .collect();
        let data = load_human(&h, threads).map_err(|e| e.to_string())?;
        log(&format!(
            "human data: {}",
            serde_json::to_string(&data.stats).unwrap_or_default()
        ));
        human_train = data.train;
        eval_sets.push(EvalSet {
            name: "human-val".into(),
            corpus: Corpus::uniform(data.val.iter().map(clone_seq).collect()),
        });
        eval_sets.push(EvalSet {
            name: "human-val-tagged".into(),
            corpus: Corpus::uniform(only_boosted(data.val)),
        });
        eval_sets.push(EvalSet {
            name: "human-holdout-map".into(),
            corpus: Corpus::uniform(data.holdout_map),
        });
    }
    Ok(Data {
        teacher_train,
        human_train,
        eval_sets,
    })
}

/// Puts freshly loaded teacher evaluation sets in front of the (unchanged) human ones.
fn replace_teacher_sets(sets: &mut Vec<EvalSet>, teacher: Vec<EvalSet>) {
    sets.retain(|s| s.name.starts_with("human"));
    sets.splice(0..0, teacher);
}

fn clone_seq(s: &Seq) -> Seq {
    Seq {
        map: s.map.clone(),
        steps: s.steps.clone(),
        source: s.source.clone(),
    }
}

/// Windows of observations sampled from `corpus`, for the DN calibration.
fn calibration_windows(corpus: &Corpus, n: usize, len: usize, seed: u64) -> Vec<Vec<ddai_brain::Observation>> {
    if corpus.is_empty() {
        return Vec::new();
    }
    let mut rng = SplitMix64::new(seed ^ 0xCA11_B8A7);
    (0..n)
        .map(|_| corpus.sample_window(&mut rng, len, 0, false).observations)
        .collect()
}

fn pct_opt(x: Option<f64>) -> String {
    x.map_or_else(|| "-".to_string(), |v| format!("{:.0}%", 100.0 * v))
}

/// A fresh learner of the configured kind (calibrated on windows of `teacher` for the fly).
pub fn make_learner(cfg: &ExperimentConfig, teacher: &Corpus) -> Result<Box<dyn Learner>, String> {
    let flyg = expand_home(&cfg.flyg);
    let brain_cfg_path = expand_home(&cfg.brain_config);
    match cfg.model.kind.as_str() {
        "fly" => {
            let windows = calibration_windows(teacher, cfg.fly.calibration_windows, 32, cfg.train.seed);
            Ok(Box::new(FlyLearner::init(
                &flyg,
                &brain_cfg_path,
                cfg.train.seed,
                cfg.fly.clone(),
                &windows,
            )?))
        }
        kind @ ("mlp" | "gru") => {
            let text =
                std::fs::read_to_string(&brain_cfg_path).map_err(|e| format!("{}: {e}", brain_cfg_path.display()))?;
            let bc = parse_brain_config(&text).map_err(|e| e.to_string())?;
            let d = input_dim(&bc.ray_grid);
            if cfg.model.hidden == 0 {
                return Err("controls need model.hidden > 0".to_string());
            }
            let net: Box<dyn ddai_controls::SeqNet> = if kind == "mlp" {
                Box::new(Mlp::new(d, cfg.model.hidden, cfg.train.seed))
            } else {
                Box::new(Gru::new(d, cfg.model.hidden, cfg.train.seed))
            };
            Ok(Box::new(ControlLearner::new(net, bc.ray_grid, cfg.model.lr)))
        }
        other => Err(format!("unknown model kind {other:?} (fly, mlp, gru)")),
    }
}

/// A learner restored from a bundle file (fly or control), for resuming and for offline eval.
pub fn learner_from_bundle(cfg: &ExperimentConfig, path: &Path) -> Result<Box<dyn Learner>, String> {
    match cfg.model.kind.as_str() {
        "fly" => {
            let b = load_bundle(path).map_err(|e| e.to_string())?;
            Ok(Box::new(FlyLearner::from_bundle(
                b,
                &expand_home(&cfg.flyg),
                cfg.fly.clone(),
            )?))
        }
        _ => {
            let b: ControlBundle = load_control_bundle(path).map_err(|e| e.to_string())?;
            let net = b.build().map_err(|e| e.to_string())?;
            let mut l = ControlLearner::new(net, b.ray_grid, cfg.model.lr);
            l.set_thresholds(b.thresholds);
            Ok(Box::new(l))
        }
    }
}

/// One arena evaluation of `actor` against `opponents` on `arena`.
#[derive(Debug, Clone, Serialize)]
pub struct ArenaEval {
    pub arena: String,
    pub opponents: Vec<String>,
    pub games: u32,
    pub w: u32,
    pub l: u32,
    pub d: u32,
    pub t: u32,
    pub win_rate: Option<[f64; 3]>,
    pub win_rate_all: Option<[f64; 3]>,
    pub self_freezes_per_min: f64,
    pub blocks_per_min: f64,
    pub decide_us_p50: Option<u32>,
    pub decide_us_p99: Option<u32>,
}

pub fn arena_eval(
    env: &Env,
    actor: &PlayerSpec,
    arena: &str,
    opponents: &[String],
    games: u32,
    base_seed: u64,
    threads: usize,
) -> Result<ArenaEval, String> {
    let a = env
        .arenas
        .get(arena)
        .ok_or_else(|| format!("unknown arena {arena:?}"))?;
    let mut players = vec![actor.clone()];
    players.extend(opponents.iter().map(|o| player_from_arg(o)));
    let cond = Condition {
        name: format!("{arena} eval"),
        arena: arena.to_string(),
        games: Some(games),
        rules: None,
        players,
    };
    let rc = RunConfig {
        name: "eval".into(),
        base_seed,
        games,
        rules: Rules::default(),
        arenas_dir: None,
        map_dir: None,
        condition: vec![cond.clone()],
    };
    let factory = env.models.factory();
    let run = run_condition(&rc, &cond, a, games, &factory, threads).map_err(|e| e.to_string())?;
    let s = summarize(&run, a.tag.label(), a.map_sha256.clone());
    let p0 = s.players.first();
    Ok(ArenaEval {
        arena: arena.to_string(),
        opponents: opponents.to_vec(),
        games,
        w: s.tally.w,
        l: s.tally.l,
        d: s.tally.d,
        t: s.tally.t,
        win_rate: s.win_rate.map(|r| [r.p, r.lo, r.hi]),
        win_rate_all: s.win_rate_all.map(|r| [r.p, r.lo, r.hi]),
        self_freezes_per_min: s.self_freezes_per_min,
        blocks_per_min: s.blocks_per_min,
        decide_us_p50: p0.and_then(|p| p.decide_us_p50),
        decide_us_p99: p0.and_then(|p| p.decide_us_p99),
    })
}

fn actor_spec(cfg: &ExperimentConfig, bundle: &Path) -> PlayerSpec {
    let mut s = PlayerSpec::simple(&cfg.model.kind);
    s.model = Some(bundle.to_string_lossy().into_owned());
    s
}

fn eval_arenas(
    cfg: &ExperimentConfig,
    env: &Env,
    run: &RunDir,
    bundle: &Path,
    phase: &str,
    step: u64,
    log: &mut dyn FnMut(&str),
) -> Result<(), String> {
    if cfg.dagger.eval_games == 0 {
        return Ok(());
    }
    let spec = actor_spec(cfg, bundle);
    for (i, arena) in cfg.dagger.eval_arenas.iter().enumerate() {
        let ev = arena_eval(
            env,
            &spec,
            arena,
            &["scripted".to_string()],
            cfg.dagger.eval_games,
            9_000_000_000 + 1_000_000 * i as u64,
            cfg.train.threads,
        )?;
        log(&format!(
            "arena eval {phase} {arena}: {}:{}:{}:{} win {}",
            ev.w,
            ev.l,
            ev.d,
            ev.t,
            ev.win_rate.map_or("n/a".to_string(), |r| format!(
                "{:.1}% [{:.1}; {:.1}]",
                100.0 * r[0],
                100.0 * r[1],
                100.0 * r[2]
            ))
        ));
        run.append_metrics(&json!({"kind": "arena", "phase": phase, "step": step, "eval": ev}))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn commit_of_source() -> String {
    let (c, dirty) = ddai_env::output::git_info(Path::new(env!("CARGO_MANIFEST_DIR")));
    if dirty { format!("{c}+dirty") } else { c }
}

/// Runs (or resumes) an experiment end to end.
pub fn run_experiment(cfg: &ExperimentConfig, log: &mut dyn FnMut(&str)) -> Result<(), String> {
    let env = load_env(
        &expand_home(&cfg.arenas_dir),
        &expand_home(&cfg.map_dir),
        Some(expand_home(&cfg.flyg)),
    )?;
    let run_root = expand_home(&cfg.run_dir);
    let run = RunDir::create(&run_root).map_err(|e| e.to_string())?;
    let cfg_path = run.path("config.toml");
    if !cfg_path.exists() {
        std::fs::write(&cfg_path, toml::to_string_pretty(cfg).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    let mut dagger_store = TeacherStore::open_or_create(
        &expand_home(&cfg.teacher_dagger),
        &format!("{}-dagger", cfg.name),
        &commit_of_source(),
    )
    .map_err(|e| e.to_string())?;
    let rounds = cfg.dagger.betas.len() as u32;

    let mut data = load_data(cfg, &env, &dagger_store, true, log)?;
    let teacher_corpus = Corpus::new(std::mem::take(&mut data.teacher_train));
    let human_corpus = Corpus::new(std::mem::take(&mut data.human_train));
    log(&format!(
        "corpora: teacher {} scored decisions, human {} scored decisions",
        teacher_corpus.scored_steps(),
        human_corpus.scored_steps()
    ));

    let last_bundle = run.checkpoint("last.bundle");
    let learner = if last_bundle.exists() && run.path("state.bin").exists() {
        learner_from_bundle(cfg, &last_bundle)?
    } else {
        make_learner(cfg, &teacher_corpus)?
    };
    log(&format!(
        "model {}: {} parameters",
        learner.label(),
        learner.num_params()
    ));
    let mut trainer = Trainer::new(
        learner,
        cfg.train.clone(),
        teacher_corpus,
        human_corpus,
        Some(run.clone()),
    )
    .map_err(|e: TrainError| e.to_string())?;
    if trainer.resume().map_err(|e| e.to_string())? {
        log(&format!("resumed at step {}", trainer.step));
    }

    // Phase 0: behaviour cloning.
    if trainer.step < cfg.bc_steps {
        let remaining = cfg.bc_steps - trainer.step;
        let s = trainer
            .train_phase("bc", 0, cfg.bc_steps, remaining, &data.eval_sets, cfg.eval_every)
            .map_err(|e| e.to_string())?;
        log(&format!(
            "bc done: {} steps in {:.0}s, {:.0} decisions/s",
            s.steps, s.elapsed_s, s.decisions_per_s
        ));
        std::fs::create_dir_all(run.path("rounds")).map_err(|e| e.to_string())?;
        std::fs::copy(&last_bundle, run.path("rounds/round-0.bundle")).map_err(|e| e.to_string())?;
        eval_arenas(
            cfg,
            &env,
            &run,
            &run.path("rounds/round-0.bundle"),
            "bc",
            trainer.step,
            log,
        )?;
    } else if !run.path("rounds/round-0.bundle").exists() && last_bundle.exists() && rounds == 0 {
        std::fs::create_dir_all(run.path("rounds")).map_err(|e| e.to_string())?;
        std::fs::copy(&last_bundle, run.path("rounds/round-0.bundle")).map_err(|e| e.to_string())?;
    }

    // DAgger rounds.
    for r in 1..=rounds {
        let phase_first = cfg.bc_steps + u64::from(r - 1) * cfg.dagger.steps_per_round;
        let phase_end = phase_first + cfg.dagger.steps_per_round;
        if trainer.step >= phase_end {
            continue;
        }
        let actor_bundle = run.path(&format!("rounds/round-{}.bundle", r - 1));
        // A round is collected once *all* its jobs are in the dataset (`round_complete`); after a
        // kill between jobs the missing jobs are collected and the finished ones are kept.
        if !dagger_store.manifest.round_complete(r) {
            std::fs::create_dir_all(run.path("rounds")).map_err(|e| e.to_string())?;
            if !actor_bundle.exists() {
                std::fs::copy(&last_bundle, &actor_bundle).map_err(|e| e.to_string())?;
            }
            let spec = actor_spec(cfg, &actor_bundle);
            let actor_arg = format!("{}:{}", cfg.model.kind, spec.model.clone().unwrap_or_default());
            let beta = cfg.dagger.betas[(r - 1) as usize];
            let jobs: Vec<JobConfig> = cfg
                .dagger
                .jobs
                .iter()
                .map(|j| JobConfig {
                    beta,
                    noise_prob: cfg.dagger.noise_prob,
                    base_seed: j.base_seed + u64::from(r) * 100_000_000,
                    ..j.clone()
                })
                .collect();
            let already = jobs
                .iter()
                .filter(|j| dagger_store.manifest.job_done(r, &job_key(j)))
                .count();
            if already > 0 {
                log(&format!(
                    "round {r}: resuming collection, {already} of {} jobs already collected",
                    jobs.len()
                ));
            }
            let summaries: Vec<JobSummary> =
                run_collect_jobs(&env, &mut dagger_store, &jobs, &actor_arg, r, cfg.train.threads, log)?;
            dagger_store.mark_round_complete(r).map_err(|e| e.to_string())?;
            run.append_metrics(&json!({
                "kind": "collect", "round": r, "beta": beta, "jobs": summaries, "jobs_resumed": already,
            }))
            .map_err(|e| e.to_string())?;
            // The closed-loop hook behaviour of this round's actor against the teacher's labels on
            // the same states (start / release rates; review F2 of E-005).
            let hook = round_hook_play(&dagger_store, r).map_err(|e| e.to_string())?;
            let report = hook.report();
            log(&format!(
                "round {r}: student hook start {} / release {} (teacher {} / {}) over {} decisions",
                pct_opt(report.start_student),
                pct_opt(report.release_student),
                pct_opt(report.start_teacher),
                pct_opt(report.release_teacher),
                report.steps
            ));
            run.append_metrics(&json!({"kind": "hook_play", "round": r, "counts": hook, "report": report}))
                .map_err(|e| e.to_string())?;
            // Aggregate: base + every DAgger round so far.
            let fresh = load_data(cfg, &env, &dagger_store, false, &mut |_| {})?;
            trainer.set_teacher(Corpus::new(fresh.teacher_train));
            replace_teacher_sets(&mut data.eval_sets, fresh.eval_sets);
            log(&format!(
                "round {r}: teacher corpus now {} scored decisions",
                trainer.teacher_steps()
            ));
        } else if trainer.step == phase_first {
            let fresh = load_data(cfg, &env, &dagger_store, false, &mut |_| {})?;
            trainer.set_teacher(Corpus::new(fresh.teacher_train));
            replace_teacher_sets(&mut data.eval_sets, fresh.eval_sets);
        }
        let saved_scale = trainer.cfg.lr_scale;
        trainer.cfg.lr_scale = saved_scale * cfg.dagger.retrain_lr_scale;
        let remaining = phase_end - trainer.step;
        let s = trainer
            .train_phase(
                &format!("dagger-{r}"),
                phase_first,
                cfg.dagger.steps_per_round,
                remaining,
                &data.eval_sets,
                cfg.eval_every,
            )
            .map_err(|e| e.to_string())?;
        trainer.cfg.lr_scale = saved_scale;
        log(&format!(
            "dagger round {r} trained: {} steps in {:.0}s",
            s.steps, s.elapsed_s
        ));
        // Each phase's checkpoint gets its own path: the model loader caches by path, so evaluating
        // `last.bundle` (overwritten every phase) would silently reuse the first load.
        let round_bundle = run.path(&format!("rounds/round-{r}.bundle"));
        std::fs::copy(&last_bundle, &round_bundle).map_err(|e| e.to_string())?;
        eval_arenas(
            cfg,
            &env,
            &run,
            &round_bundle,
            &format!("dagger-{r}"),
            trainer.step,
            log,
        )?;
    }
    std::fs::copy(&last_bundle, run.checkpoint("final.bundle")).map_err(|e| e.to_string())?;
    run.write_status(&json!({"phase": "done", "step": trainer.step}))
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Offline per-head metrics of a bundle on the held-out sets of an experiment config.
///
/// With `calibrate`, the rate-matched thresholds are fitted on the validation sets first and the
/// report is at those (otherwise at the bundle's own thresholds, `0.5` for a v1 bundle).
pub fn eval_bundle_offline(
    cfg: &ExperimentConfig,
    bundle: &Path,
    calibrate: bool,
    log: &mut dyn FnMut(&str),
) -> Result<Vec<crate::trainer::EvalRecord>, String> {
    let env = load_env(
        &expand_home(&cfg.arenas_dir),
        &expand_home(&cfg.map_dir),
        Some(expand_home(&cfg.flyg)),
    )?;
    let dagger_store = TeacherStore::open_or_create(
        &expand_home(&cfg.teacher_dagger),
        &format!("{}-dagger", cfg.name),
        &commit_of_source(),
    )
    .map_err(|e| e.to_string())?;
    let data = load_data(cfg, &env, &dagger_store, true, log)?;
    let learner = learner_from_bundle(cfg, bundle)?;
    let mut trainer = Trainer::new(
        learner,
        cfg.train.clone(),
        Corpus::new(Vec::new()),
        Corpus::new(Vec::new()),
        None,
    )
    .map_err(|e| e.to_string())?;
    if calibrate {
        trainer
            .calibrate_thresholds("eval", &data.eval_sets)
            .map_err(|e| e.to_string())?;
        let t = trainer.learner().thresholds();
        log(&format!(
            "rate-matched thresholds jump/hook/fire: {:.3}/{:.3}/{:.3}",
            t.jump, t.hook, t.fire
        ));
    }
    Ok(trainer.evaluate(&data.eval_sets))
}

/// Paths a caller may want after a run.
pub fn final_bundle(cfg: &ExperimentConfig) -> PathBuf {
    expand_home(&cfg.run_dir).join("checkpoints/final.bundle")
}

#[allow(dead_code)]
fn _keep(_: Arc<()>, _: BundleMeta) {}
