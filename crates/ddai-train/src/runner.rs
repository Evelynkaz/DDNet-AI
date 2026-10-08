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

use crate::experiment::{Env, JobConfig, JobSummary, expand_home, job_key, load_env_with_scenarios, run_collect_jobs};
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
    /// Arenas the checkpoint is selected on (credited win rate); empty = the training arenas among
    /// `eval_arenas`. A holdout arena listed here is refused.
    #[serde(default)]
    pub select_arenas: Vec<String>,
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
            select_arenas: Vec::new(),
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
    /// Technique scenarios the DAgger jobs may label (`scenario = "T1"`); none by default.
    #[serde(default)]
    pub scenarios_dir: Option<String>,
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
    /// Task 8.5a: start from this checkpoint instead of a fresh model (fine-tuning on new data). A resumed run continues from its own
    /// `last.bundle` and `state.bin` as always.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_bundle: Option<String>,
}

/// Everything loaded once: corpora and the held-out sets.
pub struct Data {
    pub teacher_train: Vec<Seq>,
    pub human_train: Vec<Seq>,
    pub eval_sets: Vec<EvalSet>,
    /// Chunks of the DAgger store that are already in these corpora (a round loads only the rest).
    pub dagger_chunks: HashSet<usize>,
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
            &env.maps,
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
        &env.maps,
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
        log(&format!(
            "human data, Copy Love Box family (D-057): {:.1}% of the scored training steps, {:.1}% of the training sampling weight (after the per-demo caps and the weights)",
            100.0 * data.stats.step_share_of("Copy"),
            100.0 * data.stats.weight_share_of("Copy")
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
        dagger_chunks: store_chunks(dagger).into_iter().collect(),
    })
}

/// Loads the chunks of the DAgger store that are not in the corpora yet (a new round's, plus any left over from
/// a partly collected round that was loaded when the run started) into training / validation / holdout
/// sequences, and marks them loaded. A round used to rebuild the whole teacher side (base data included) from
/// disk while the old corpus was still alive; now the corpus only grows.
fn load_new_dagger_chunks(
    cfg: &ExperimentConfig,
    env: &Env,
    dagger: &TeacherStore,
    loaded: &mut HashSet<usize>,
) -> Result<TeacherSplit, String> {
    let fresh: Vec<usize> = store_chunks(dagger)
        .into_iter()
        .filter(|c| !loaded.contains(c))
        .collect();
    let split = load_teacher(
        dagger,
        &fresh,
        &env.maps,
        &env.holdout_names(),
        &cfg.teacher_data,
        cfg.train.threads,
    )
    .map_err(|e| e.to_string())?;
    loaded.extend(fresh);
    Ok(split)
}

/// Adds a round's validation and holdout episodes to the evaluation sets (creating the set on first use).
fn append_eval_sets(sets: &mut Vec<EvalSet>, val: Vec<Seq>, holdout: Vec<Seq>) {
    let mut add = |name: String, seqs: Vec<Seq>| {
        if seqs.is_empty() {
            return;
        }
        match sets.iter_mut().find(|s| s.name == name) {
            Some(s) => s.corpus.append_uniform(seqs),
            None => sets.push(EvalSet {
                name,
                corpus: Corpus::uniform(seqs),
            }),
        }
    };
    add("dagger-val".to_string(), val);
    for (arena, seqs) in by_arena(holdout) {
        add(format!("teacher-holdout:{arena}"), seqs);
    }
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

/// How much of the teacher corpus is technique-scenario data (`scn:T*`): episodes, scored decisions and share of the
/// sampling weight, per scenario. (E-008: what the student actually sees of each technique.)
pub fn scenario_share_line(teacher: &Corpus) -> String {
    let mut by: std::collections::BTreeMap<String, (usize, usize, f64)> = std::collections::BTreeMap::new();
    let mut total = 0.0f64;
    for s in &teacher.seqs {
        let w: f64 = s.steps.iter().map(|st| f64::from(st.weight.max(0.0))).sum();
        total += w;
        if let crate::seq::Source::Teacher { arena, .. } = &s.source
            && arena.starts_with("scn:")
        {
            let e = by.entry(arena.clone()).or_default();
            e.0 += 1;
            e.1 += s.steps.iter().filter(|st| st.weight > 0.0).count();
            e.2 += w;
        }
    }
    let scn_w: f64 = by.values().map(|e| e.2).sum();
    let detail: Vec<String> = by
        .iter()
        .map(|(k, (n, steps, w))| {
            format!(
                "{k}: {n} episodes / {steps} steps / {:.2}%",
                100.0 * w / total.max(1e-9)
            )
        })
        .collect();
    format!(
        "teacher corpus: technique scenarios are {:.2}% of the sampling weight ({})",
        100.0 * scn_w / total.max(1e-9),
        if detail.is_empty() {
            "none".to_string()
        } else {
            detail.join("; ")
        }
    )
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
            l.set_hook_view(b.hook_view);
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
    /// Games won by the player's own credited block over all games (D-059), with its Wilson interval.
    pub credited_win_rate: Option<[f64; 3]>,
    pub credited_w: u32,
    /// Task 3.10: games won by the focal player's credited block that was still on at the end of the window (`held W`), lost games whose block on us
    /// held, and the rate of the former over all games (the held-block metric; meaningful with the 250-tick window of [`arena_eval_held`]).
    pub held_block_w: u32,
    pub held_block_l: u32,
    pub held_block_rate: Option<[f64; 3]>,
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
    arena_eval_rules(
        env,
        actor,
        arena,
        opponents,
        games,
        base_seed,
        threads,
        Rules::default(),
    )
}

/// [`arena_eval`] over the held-block window (`Rules::held_block_window`, 250 ticks after the deciding freeze): the same games up to the deciding tick,
/// slower by the extra ticks, and the `held_block_*` fields of the result mean something. Opt-in (task 3.10): [`arena_eval`] keeps the 150-tick window.
pub fn arena_eval_held(
    env: &Env,
    actor: &PlayerSpec,
    arena: &str,
    opponents: &[String],
    games: u32,
    base_seed: u64,
    threads: usize,
) -> Result<ArenaEval, String> {
    arena_eval_rules(
        env,
        actor,
        arena,
        opponents,
        games,
        base_seed,
        threads,
        Rules::default().held_block_window(),
    )
}

#[allow(clippy::too_many_arguments)]
fn arena_eval_rules(
    env: &Env,
    actor: &PlayerSpec,
    arena: &str,
    opponents: &[String],
    games: u32,
    base_seed: u64,
    threads: usize,
    rules: Rules,
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
        duel: None,
    };
    let rc = RunConfig {
        name: "eval".into(),
        base_seed,
        games,
        rules,
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
        credited_win_rate: s.credited_win_rate.map(|r| [r.p, r.lo, r.hi]),
        credited_w: s.credited_w,
        held_block_w: s.held_block_w,
        held_block_l: s.held_block_l,
        held_block_rate: s.held_block_rate.map(|r| [r.p, r.lo, r.hi]),
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

/// The arenas whose evaluation of `phase` is already in `metrics.jsonl`.
fn evaluated_arenas(run: &RunDir, phase: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    if let Ok(text) = std::fs::read_to_string(run.path("metrics.jsonl")) {
        for line in text.lines() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line)
                && v["kind"] == "arena"
                && v["phase"] == phase
                && let Some(a) = v["eval"]["arena"].as_str()
            {
                out.insert(a.to_string());
            }
        }
    }
    out
}

/// Per phase (`bc`, `dagger-1`, ...) the mean credited win rate over `arenas` (every one of them must have
/// been evaluated in that phase), in the order the phases were evaluated.
pub fn credited_by_phase(run: &RunDir, arenas: &[String]) -> Vec<(String, f64)> {
    let mut by_phase: Vec<(String, Vec<(String, f64)>)> = Vec::new();
    if let Ok(text) = std::fs::read_to_string(run.path("metrics.jsonl")) {
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if v["kind"] != "arena" {
                continue;
            }
            let (Some(phase), Some(arena), Some(rate)) = (
                v["phase"].as_str(),
                v["eval"]["arena"].as_str(),
                v["eval"]["credited_win_rate"][0].as_f64(),
            ) else {
                continue;
            };
            if !arenas.iter().any(|a| a == arena) {
                continue;
            }
            match by_phase.iter_mut().find(|(p, _)| p == phase) {
                Some((_, list)) => {
                    if !list.iter().any(|(a, _)| a == arena) {
                        list.push((arena.to_string(), rate));
                    }
                }
                None => by_phase.push((phase.to_string(), vec![(arena.to_string(), rate)])),
            }
        }
    }
    by_phase
        .into_iter()
        .filter(|(_, list)| list.len() == arenas.len())
        .map(|(p, list)| (p, list.iter().map(|(_, r)| r).sum::<f64>() / list.len() as f64))
        .collect()
}

/// The bundle file a phase's evaluation was made on.
fn phase_bundle(run: &RunDir, phase: &str) -> Option<PathBuf> {
    let n: u32 = if phase == "bc" {
        0
    } else {
        phase.strip_prefix("dagger-")?.parse().ok()?
    };
    Some(run.path(&format!("rounds/round-{n}.bundle")))
}

/// The phase chosen by [`select_checkpoint`] and the table it chose from (phase, mean credited win rate).
pub type Selection = (String, Vec<(String, f64)>);

/// Picks the round to keep by the credited win rate on the **training** arenas only (never a holdout:
/// selecting on those would leak them), ties to the later round, and copies it to `checkpoints/selected.bundle`.
/// Returns the chosen phase and the table it chose from.
pub fn select_checkpoint(run: &RunDir, select_arenas: &[String]) -> Result<Option<Selection>, String> {
    let table = credited_by_phase(run, select_arenas);
    let Some((best, _)) = table
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(p, s)| (p.clone(), *s))
    else {
        return Ok(None);
    };
    // `max_by` keeps the last of equal maxima, i.e. the later round.
    let src = phase_bundle(run, &best).ok_or_else(|| format!("unknown phase {best:?}"))?;
    std::fs::copy(&src, run.checkpoint("selected.bundle")).map_err(|e| format!("{}: {e}", src.display()))?;
    Ok(Some((best, table)))
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
    let done = evaluated_arenas(run, phase);
    for (i, arena) in cfg.dagger.eval_arenas.iter().enumerate() {
        if done.contains(arena) {
            continue; // a resumed run does not repeat (or double-log) an evaluation it already made
        }
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
            "arena eval {phase} {arena}: {}:{}:{}:{} credited {} win {}",
            ev.w,
            ev.l,
            ev.d,
            ev.t,
            ev.credited_win_rate.map_or("n/a".to_string(), |r| format!(
                "{:.1}% [{:.1}; {:.1}]",
                100.0 * r[0],
                100.0 * r[1],
                100.0 * r[2]
            )),
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

/// Fields that may change between a kill and the resume without changing what the run computes (thread
/// counts, logging cadence, how many step bundles are kept, the run directory's own path).
fn normalised_for_resume(cfg: &ExperimentConfig) -> ExperimentConfig {
    let mut c = cfg.clone();
    c.train.threads = 0;
    c.train.log_every = 0;
    c.train.keep_checkpoints = 0;
    c.run_dir = String::new();
    c.eval_every = 0;
    // K = 0 and K = 1 are the same single engine (7.2c accepted that change on resume).
    c.fly.batched_subengines = c.fly.batched_subengines.max(1);
    c
}

/// The top-level sections of two configs that differ (the message of a refused resume).
fn differing_sections(old: &ExperimentConfig, new: &ExperimentConfig) -> Vec<String> {
    let (Ok(a), Ok(b)) = (toml::Value::try_from(old), toml::Value::try_from(new)) else {
        return vec!["(not comparable)".to_string()];
    };
    let (Some(a), Some(b)) = (a.as_table(), b.as_table()) else {
        return Vec::new();
    };
    let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
    keys.sort();
    keys.dedup();
    keys.into_iter().filter(|k| a.get(*k) != b.get(*k)).cloned().collect()
}

/// The fly options of task 7.2c that change the gradient (the backend, the batched sub-engine count `K`, the
/// stop-gradient burn-in), named as `fly.<key> <old> -> <new>` for the message of a refused resume.
fn gradient_option_changes(old: &FlyTrainConfig, new: &FlyTrainConfig) -> Vec<String> {
    let mut diffs = Vec::new();
    if old.backend != new.backend {
        diffs.push(format!("fly.backend {:?} -> {:?}", old.backend, new.backend));
    }
    let k = |c: &FlyTrainConfig| c.batched_subengines.max(1);
    if k(old) != k(new) {
        diffs.push(format!("fly.batched_subengines {} -> {}", k(old), k(new)));
    }
    if old.batched_stop_grad_decisions != new.batched_stop_grad_decisions {
        diffs.push(format!(
            "fly.batched_stop_grad_decisions {} -> {}",
            old.batched_stop_grad_decisions, new.batched_stop_grad_decisions
        ));
    }
    diffs
}

/// Records the configuration the run is about to execute: `config.toml` always holds the config of the
/// **current** (re)start, and every earlier different one is kept as `config-before-<n>.toml` (E-005 review F8:
/// `e005-mlp-w` ran a DAgger schedule its `config.toml` did not describe). Resuming a run that has started
/// with a config that differs in anything but [`normalised_for_resume`]'s fields is refused unless
/// `allow_change` (the new config is then recorded and the change is logged).
pub fn record_config(
    run: &RunDir,
    cfg: &ExperimentConfig,
    allow_change: bool,
    log: &mut dyn FnMut(&str),
) -> Result<(), String> {
    let path = run.path("config.toml");
    let text = toml::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    if path.exists() {
        let old_text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        if old_text == text {
            return Ok(());
        }
        let started = run.path("state.bin").exists();
        match toml::from_str::<ExperimentConfig>(&old_text) {
            Ok(old) if started && normalised_for_resume(&old) != normalised_for_resume(cfg) => {
                let mut sections = differing_sections(&normalised_for_resume(&old), &normalised_for_resume(cfg));
                let gradient = gradient_option_changes(&old.fly, &cfg.fly);
                if !gradient.is_empty() {
                    // Task 7.2c's options change the f32 summation order / the truncation of the gradient: name them.
                    sections.push(format!("gradient options: {}", gradient.join(", ")));
                }
                if !allow_change {
                    return Err(format!(
                        "refusing to resume {}: the configuration differs from the one this run started with in {sections:?} \
                         (re-run with --allow-config-change to accept it; the old config is kept)",
                        run.path("").display()
                    ));
                }
                log(&format!(
                    "WARNING: resuming with a changed configuration ({sections:?})"
                ));
            }
            Ok(_) => {}
            Err(_) if started && !allow_change => {
                return Err(format!(
                    "refusing to resume {}: its config.toml cannot be parsed to compare (re-run with --allow-config-change)",
                    run.path("").display()
                ));
            }
            Err(_) => {}
        }
        let mut n = 1;
        while run.path(&format!("config-before-{n}.toml")).exists() {
            n += 1;
        }
        std::fs::rename(&path, run.path(&format!("config-before-{n}.toml"))).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, text).map_err(|e| e.to_string())
}

/// Warnings about fly options that do nothing or train nothing (printed at start).
pub fn fly_config_warnings(cfg: &ExperimentConfig) -> Vec<String> {
    let mut w = Vec::new();
    let (fly, per_seq) = (&cfg.fly, cfg.fly.backend == ddai_fly::TrainBackend::PerSequence);
    if per_seq && fly.batched_stop_grad_decisions > 0 {
        w.push(format!(
            "fly.batched_stop_grad_decisions = {} is ignored: backend = \"per-seq\" always backpropagates the whole window",
            fly.batched_stop_grad_decisions
        ));
    }
    if per_seq && fly.batched_subengines > 1 {
        w.push(format!(
            "fly.batched_subengines = {} is ignored: it needs backend = \"batched\"",
            fly.batched_subengines
        ));
    }
    if !per_seq && fly.batched_stop_grad_decisions > 0 && cfg.train.window_len <= fly.batched_stop_grad_decisions {
        w.push(format!(
            "fly.batched_stop_grad_decisions = {} >= train.window_len = {}: every window is all burn-in, nothing is scored or trained",
            fly.batched_stop_grad_decisions, cfg.train.window_len
        ));
    }
    w
}

/// Runs (or resumes) an experiment end to end; refuses to resume under a changed configuration.
pub fn run_experiment(cfg: &ExperimentConfig, log: &mut dyn FnMut(&str)) -> Result<(), String> {
    run_experiment_with(cfg, false, log)
}

/// [`run_experiment`] with the option to accept a changed configuration on resume.
pub fn run_experiment_with(
    cfg: &ExperimentConfig,
    allow_config_change: bool,
    log: &mut dyn FnMut(&str),
) -> Result<(), String> {
    let env = load_env_with_scenarios(
        &expand_home(&cfg.arenas_dir),
        &expand_home(&cfg.map_dir),
        Some(expand_home(&cfg.flyg)),
        cfg.scenarios_dir.as_deref().map(expand_home).as_deref(),
    )?;
    let run_root = expand_home(&cfg.run_dir);
    let run = RunDir::create(&run_root).map_err(|e| e.to_string())?;
    select_arenas(cfg, &env)?; // a holdout selection arena is refused before anything is trained
    for w in fly_config_warnings(cfg) {
        log(&format!("warning: {w}"));
    }
    record_config(&run, cfg, allow_config_change, log)?;
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
    log(&scenario_share_line(&teacher_corpus));

    let last_bundle = run.checkpoint("last.bundle");
    let learner = if last_bundle.exists() && run.path("state.bin").exists() {
        learner_from_bundle(cfg, &last_bundle)?
    } else if let Some(init) = &cfg.init_bundle {
        learner_from_bundle(cfg, &expand_home(init))?
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
    } else if !run.path("rounds/round-0.bundle").exists() && last_bundle.exists() && rounds == 0 {
        std::fs::create_dir_all(run.path("rounds")).map_err(|e| e.to_string())?;
        std::fs::copy(&last_bundle, run.path("rounds/round-0.bundle")).map_err(|e| e.to_string())?;
    }

    // The BC checkpoint's arena evaluation (a resumed run completes one the kill interrupted).
    if run.path("rounds/round-0.bundle").exists() {
        eval_arenas(
            cfg,
            &env,
            &run,
            &run.path("rounds/round-0.bundle"),
            "bc",
            cfg.bc_steps,
            log,
        )?;
    }

    // DAgger rounds.
    for r in 1..=rounds {
        let phase_first = cfg.bc_steps + u64::from(r - 1) * cfg.dagger.steps_per_round;
        let phase_end = phase_first + cfg.dagger.steps_per_round;
        if trainer.step >= phase_end {
            let done_bundle = run.path(&format!("rounds/round-{r}.bundle"));
            if done_bundle.exists() {
                eval_arenas(cfg, &env, &run, &done_bundle, &format!("dagger-{r}"), phase_end, log)?;
            }
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
            let expected: Vec<String> = jobs.iter().map(job_key).collect();
            let unexpected = dagger_store.manifest.unexpected_job_keys(r, &expected);
            if !unexpected.is_empty() {
                log(&format!(
                    "WARNING: round {r} already holds jobs the current config does not list (changed seed, games or noise \
                     length?); they stay in the dataset and the listed jobs are collected next to them: {unexpected:?}"
                ));
                run.append_metrics(&json!({"kind": "warning", "round": r, "unexpected_job_keys": unexpected}))
                    .map_err(|e| e.to_string())?;
            }
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
            // Aggregate: the corpus grows by this round's episodes.
            let fresh = load_new_dagger_chunks(cfg, &env, &dagger_store, &mut data.dagger_chunks)?;
            trainer.append_teacher(fresh.train);
            append_eval_sets(&mut data.eval_sets, fresh.val, fresh.holdout);
            log(&format!(
                "round {r}: teacher corpus now {} scored decisions",
                trainer.teacher_steps()
            ));
            let scen_round: (usize, u64) = summaries
                .iter()
                .filter(|j| j.arena.starts_with("scn:"))
                .fold((0, 0), |a, j| (a.0 + j.games as usize, a.1 + j.steps));
            if scen_round.0 > 0 {
                log(&format!(
                    "round {r}: the student's technique-scenario episodes this round: {} trials, {} decisions labelled",
                    scen_round.0, scen_round.1
                ));
            }
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
    // The checkpoint to use: the round with the best credited win rate on the training arenas.
    let select = select_arenas(cfg, &env)?;
    if let Some((phase, table)) = select_checkpoint(&run, &select)? {
        let logged = std::fs::read_to_string(run.path("metrics.jsonl")).is_ok_and(|t| {
            t.lines()
                .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                .rfind(|v| v["kind"] == "selection")
                .is_some_and(|v| v["phase"] == phase.as_str() && v["table"] == json!(table))
        });
        if !logged {
            log(&format!(
                "selected {phase} by credited win rate on {select:?}: {table:?}"
            ));
            run.append_metrics(&json!({"kind": "selection", "phase": phase, "arenas": select, "table": table}))
                .map_err(|e| e.to_string())?;
        }
    }
    run.write_status(&json!({"phase": "done", "step": trainer.step}))
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// The arenas the checkpoint is selected on: `dagger.select_arenas`, else every evaluation arena that is a
/// training arena (never a holdout).
fn select_arenas(cfg: &ExperimentConfig, env: &Env) -> Result<Vec<String>, String> {
    if !cfg.dagger.select_arenas.is_empty() {
        for a in &cfg.dagger.select_arenas {
            let arena = env
                .arenas
                .get(a)
                .ok_or_else(|| format!("dagger.select_arenas: unknown arena {a:?}"))?;
            if arena.tag.label() != "train" {
                return Err(format!(
                    "dagger.select_arenas: {a:?} is a {} arena; selecting a checkpoint on it would leak it",
                    arena.tag.label()
                ));
            }
        }
        return Ok(cfg.dagger.select_arenas.clone());
    }
    Ok(cfg
        .dagger
        .eval_arenas
        .iter()
        .filter(|a| env.arenas.get(*a).is_some_and(|x| x.tag.label() == "train"))
        .cloned()
        .collect())
}

/// Offline per-head metrics of a bundle on the held-out sets of an experiment config.
///
/// With `calibrate`, the rate-matched thresholds are fitted on the validation sets first and the
/// report is at those (otherwise at the bundle's own thresholds, `0.5` for a v1 bundle).
///
/// `write_calibrated` (implies `calibrate`) also writes the bundle with those thresholds to a new file: the
/// same weights at calibrated and at `0.5` thresholds can then be A/B-tested in the arena (review F9).
pub fn eval_bundle_offline(
    cfg: &ExperimentConfig,
    bundle: &Path,
    calibrate: bool,
    write_calibrated: Option<&Path>,
    log: &mut dyn FnMut(&str),
) -> Result<Vec<crate::trainer::EvalRecord>, String> {
    let env = load_env_with_scenarios(
        &expand_home(&cfg.arenas_dir),
        &expand_home(&cfg.map_dir),
        Some(expand_home(&cfg.flyg)),
        cfg.scenarios_dir.as_deref().map(expand_home).as_deref(),
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
    if calibrate || write_calibrated.is_some() {
        trainer
            .calibrate_thresholds("eval", &data.eval_sets)
            .map_err(|e| e.to_string())?;
        let t = trainer.learner().thresholds();
        log(&format!(
            "rate-matched thresholds jump/hook/fire: {:.3}/{:.3}/{:.3}",
            t.jump, t.hook, t.fire
        ));
        if let Some(out) = write_calibrated {
            let meta = if cfg.model.kind == "fly" {
                load_bundle(bundle).map_err(|e| e.to_string())?.meta
            } else {
                load_control_bundle(bundle).map_err(|e| e.to_string())?.meta
            };
            trainer.learner().save(out, meta)?;
            log(&format!("wrote {}", out.display()));
        }
    }
    Ok(trainer.evaluate(&data.eval_sets))
}

/// Paths a caller may want after a run.
pub fn final_bundle(cfg: &ExperimentConfig) -> PathBuf {
    expand_home(&cfg.run_dir).join("checkpoints/final.bundle")
}

#[allow(dead_code)]
fn _keep(_: Arc<()>, _: BundleMeta) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ExperimentConfig {
        ExperimentConfig {
            name: "t".into(),
            model: ModelSpec {
                kind: "mlp".into(),
                hidden: 4,
                lr: 1e-3,
            },
            flyg: String::new(),
            brain_config: String::new(),
            arenas_dir: String::new(),
            map_dir: String::new(),
            scenarios_dir: None,
            run_dir: String::new(),
            teacher_base: Vec::new(),
            teacher_dagger: String::new(),
            train: TrainConfig::default(),
            fly: FlyTrainConfig::default(),
            teacher_data: TeacherDataConfig::default(),
            human: None,
            bc_steps: 10,
            eval_every: 0,
            dagger: DaggerConfig {
                betas: vec![0.5, 0.25],
                ..DaggerConfig::default()
            },
            init_bundle: None,
        }
    }

    fn arena_line(phase: &str, arena: &str, credited: f64) -> String {
        json!({"kind": "arena", "phase": phase, "step": 1, "eval": {"arena": arena, "credited_win_rate": [credited, 0.0, 1.0]}})
            .to_string()
    }

    #[test]
    fn the_checkpoint_is_selected_by_credited_wins_on_the_given_training_arenas_only() {
        let dir = tempfile::tempdir().unwrap();
        let run = RunDir::create(dir.path()).unwrap();
        std::fs::create_dir_all(run.path("rounds")).unwrap();
        for n in 0..3 {
            std::fs::write(run.path(&format!("rounds/round-{n}.bundle")), format!("bundle {n}")).unwrap();
        }
        // The holdout arena would pick round 1; the training arenas pick round 2 (round 1 lacks `pit`).
        let lines = [
            arena_line("bc", "clb-left", 0.30),
            arena_line("bc", "pit", 0.10),
            arena_line("bc", "chillblock5-ruler", 0.20),
            arena_line("dagger-1", "clb-left", 0.50),
            arena_line("dagger-1", "chillblock5-ruler", 0.90),
            arena_line("dagger-2", "clb-left", 0.45),
            arena_line("dagger-2", "pit", 0.45),
            arena_line("dagger-2", "chillblock5-ruler", 0.05),
        ];
        std::fs::write(run.path("metrics.jsonl"), lines.join("\n") + "\n").unwrap();
        let arenas = vec!["clb-left".to_string(), "pit".to_string()];
        let table = credited_by_phase(&run, &arenas);
        assert_eq!(
            table.len(),
            2,
            "a phase missing a selection arena is not a candidate: {table:?}"
        );
        assert!((table[0].1 - 0.20).abs() < 1e-9 && (table[1].1 - 0.45).abs() < 1e-9);
        let (phase, _) = select_checkpoint(&run, &arenas).unwrap().unwrap();
        assert_eq!(phase, "dagger-2");
        assert_eq!(
            std::fs::read_to_string(run.checkpoint("selected.bundle")).unwrap(),
            "bundle 2"
        );
        // Ties go to the later round; nothing to choose from is not an error.
        let tie = [
            arena_line("bc", "clb-left", 0.4),
            arena_line("dagger-1", "clb-left", 0.4),
        ];
        std::fs::write(run.path("metrics.jsonl"), tie.join("\n") + "\n").unwrap();
        let (phase, _) = select_checkpoint(&run, &["clb-left".to_string()]).unwrap().unwrap();
        assert_eq!(phase, "dagger-1");
        std::fs::write(run.path("metrics.jsonl"), "").unwrap();
        assert!(select_checkpoint(&run, &arenas).unwrap().is_none());
    }

    #[test]
    fn the_config_of_every_restart_is_recorded_and_a_changed_resume_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let run = RunDir::create(dir.path()).unwrap();
        let log = std::cell::RefCell::new(Vec::<String>::new());
        let mut sink = |l: &str| log.borrow_mut().push(l.to_string());
        let a = cfg();
        // A fresh run records its config.
        record_config(&run, &a, false, &mut sink).unwrap();
        assert_eq!(
            std::fs::read_to_string(run.path("config.toml")).unwrap(),
            toml::to_string_pretty(&a).unwrap()
        );
        // Before the run has started (no state.bin) any change just replaces the file and keeps the old one.
        let mut b = a.clone();
        b.dagger.betas = vec![0.5, 0.3, 0.15];
        record_config(&run, &b, false, &mut sink).unwrap();
        assert!(run.path("config-before-1.toml").exists());
        assert!(
            std::fs::read_to_string(run.path("config.toml"))
                .unwrap()
                .contains("0.15")
        );

        // Once the run has state, a change in what it computes is refused and the files stay as they were.
        std::fs::write(run.path("state.bin"), b"x").unwrap();
        let mut c = b.clone();
        c.dagger.betas = vec![0.5, 0.25, 0.0];
        let err = record_config(&run, &c, false, &mut sink).unwrap_err();
        assert!(err.contains("dagger") && err.contains("--allow-config-change"), "{err}");
        assert!(
            std::fs::read_to_string(run.path("config.toml"))
                .unwrap()
                .contains("0.15")
        );
        assert!(!run.path("config-before-2.toml").exists());

        // Thread counts and logging cadence are free to change on resume.
        let mut d = b.clone();
        d.train.threads = 1;
        d.train.log_every = 7;
        record_config(&run, &d, false, &mut sink).unwrap();
        assert!(
            run.path("config-before-2.toml").exists(),
            "the previous version is kept"
        );

        // The change can be accepted explicitly; it is logged and the old config kept.
        record_config(&run, &c, true, &mut sink).unwrap();
        assert!(run.path("config-before-3.toml").exists());
        assert!(
            std::fs::read_to_string(run.path("config.toml"))
                .unwrap()
                .contains("0.0")
        );
        assert!(
            log.borrow()
                .iter()
                .any(|l| l.contains("WARNING") && l.contains("dagger")),
            "{:?}",
            log.borrow()
        );
        // The same config again is a no-op.
        record_config(&run, &c, false, &mut sink).unwrap();
        assert!(!run.path("config-before-4.toml").exists());
    }
}

#[cfg(test)]
mod resume_checks {
    use ddai_fly::TrainBackend;

    use super::*;

    fn batched(k: usize, stop: usize) -> FlyTrainConfig {
        FlyTrainConfig {
            backend: TrainBackend::Batched,
            batched_subengines: k,
            batched_stop_grad_decisions: stop,
            ..FlyTrainConfig::default()
        }
    }

    fn experiment(fly: FlyTrainConfig, window_len: usize) -> ExperimentConfig {
        let text = "name = \"t\"\nflyg = \"f\"\nbrain_config = \"b\"\narenas_dir = \"a\"\nmap_dir = \"m\"\nrun_dir = \"r\"\n\
             teacher_base = []\nteacher_dagger = \"d\"\nbc_steps = 1\n[model]\nkind = \"fly\"\n";
        let mut cfg: ExperimentConfig = toml::from_str(text).expect("minimal experiment config");
        cfg.fly = fly;
        cfg.train.window_len = window_len;
        cfg
    }

    /// Task 7.2c's gradient-changing options (backend, K sub-engines, stop-gradient burn-in) are covered by
    /// `record_config` (8.2b): a resume that changes any of them is refused, and `--allow-config-change` accepts it.
    #[test]
    fn a_resume_that_changes_the_backend_k_or_stop_gradient_is_refused_by_record_config() {
        let mut sink = |_: &str| {};
        for (old, new) in [
            (FlyTrainConfig::default(), batched(1, 0)),
            (batched(1, 0), batched(4, 0)),
            (batched(1, 0), batched(1, 6)),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let run = RunDir::create(dir.path()).unwrap();
            record_config(&run, &experiment(old, 32), false, &mut sink).unwrap();
            std::fs::write(run.path("state.bin"), b"x").unwrap(); // the run has started
            let e = record_config(&run, &experiment(new.clone(), 32), false, &mut sink).unwrap_err();
            assert!(e.contains("refusing to resume") && e.contains("fly"), "{e}");
            record_config(&run, &experiment(new, 32), true, &mut sink).unwrap();
            assert!(run.path("config-before-1.toml").exists());
        }
        // K = 0 and K = 1 are the same single engine: not a change.
        let dir = tempfile::tempdir().unwrap();
        let run = RunDir::create(dir.path()).unwrap();
        record_config(&run, &experiment(batched(0, 0), 32), false, &mut sink).unwrap();
        std::fs::write(run.path("state.bin"), b"x").unwrap();
        record_config(&run, &experiment(batched(1, 0), 32), false, &mut sink).unwrap();
        // A config.toml written before 7.2c has none of the new keys: it resumes under the defaults.
        let dir = tempfile::tempdir().unwrap();
        let run = RunDir::create(dir.path()).unwrap();
        let mut value = toml::Value::try_from(experiment(FlyTrainConfig::default(), 32)).unwrap();
        let fly = value.get_mut("fly").and_then(toml::Value::as_table_mut).unwrap();
        for key in ["backend", "batched_subengines", "batched_stop_grad_decisions"] {
            fly.remove(key);
        }
        let old_text = toml::to_string_pretty(&value).unwrap();
        for key in ["batched_subengines", "batched_stop_grad_decisions"] {
            assert!(!old_text.contains(key), "{key} is still in the old config text");
        }
        std::fs::write(run.path("config.toml"), old_text).unwrap();
        std::fs::write(run.path("state.bin"), b"x").unwrap();
        record_config(&run, &experiment(FlyTrainConfig::default(), 32), false, &mut sink).unwrap();
    }

    #[test]
    fn config_warnings_catch_options_that_do_nothing_or_train_nothing() {
        assert!(fly_config_warnings(&experiment(batched(1, 0), 32)).is_empty());
        assert!(fly_config_warnings(&experiment(batched(2, 6), 32)).is_empty());
        let per_seq = FlyTrainConfig {
            batched_stop_grad_decisions: 6,
            batched_subengines: 2,
            ..FlyTrainConfig::default()
        };
        let w = fly_config_warnings(&experiment(per_seq, 32));
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].contains("per-seq") && w[1].contains("needs backend"));
        let w = fly_config_warnings(&experiment(batched(1, 6), 6));
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("all burn-in"), "{w:?}");
    }
}
