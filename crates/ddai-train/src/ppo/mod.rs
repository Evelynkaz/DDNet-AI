//! Recurrent PPO on the held-block objective, from an imitation checkpoint (task 8.5b, E-027).
//!
//! The pipeline of `docs/research/fly-training-methods.md` §2.1 on the 8.5a machinery: the fly of an E-008 bundle (upgraded with the
//! opponent-state channels) plays episodes against the scripted bot, the planner and its own earlier selves, from post-freeze bank starts
//! (the held-block task) and from the spawn (the first-freeze skill); a privileged critic and GAE turn the reward of
//! [`reward`] into advantages; the update of [`learner`] is PPO on the batched BPTT with a KL anchor to the start policy and a behaviour
//! cloning term on planner labels. Everything is a function of the seed and the configuration, at any thread count, and the run is
//! resumable from `state.bin`.
//!
//! A run directory is that of the other training runs (`config.toml`, `metrics.jsonl`, `status.json`, `state.bin`, `checkpoints/`), and
//! the metrics are lines the «Обучение» tab reads (`kind = "train"`, `kind = "arena"`, `kind = "selection"`), plus `kind = "ppo"`
//! (per iteration) and `kind = "ppo_eval"` (the held-block numbers of an evaluation point).

pub mod actor;
pub mod config;
pub mod critic;
pub mod curriculum;
pub mod learner;
pub mod reward;
pub mod rollout;

pub use self::config::PpoConfig;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use ddai_env::config::PlayerSpec;
use ddai_fly::brain::FlyBrainConfig;
use ddai_fly::bundle::{
    BundleMeta, FlyBrainTemplate, FlyBundle, load_bundle, read_zstd_postcard, save_bundle, sha256_hex_of_file,
    write_zstd_postcard,
};
use ddai_fly::rng::SplitMix64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use self::actor::{ActMode, WindowGrid};
use self::config::record_ppo_config;
use self::critic::{INPUT_DIM, MapFields};
use self::curriculum::{CurriculumState, DemoSet, bank_fingerprint, build_demos, resumed_log};
use self::learner::{BcData, PpoLearner, PpoState, UpdateStats};
use self::reward::EpisodeKind;
use self::rollout::{Episode, EpisodeSpec, RolloutCtx, play_episode_ppo};
use crate::bank::{Bank, BankStart};
use crate::es::space::with_params;
use crate::es::{EvalParams, EvalPoint, arena_line, evaluate_spec, make_pool, strip_items, unix_seconds};
use crate::experiment::{Env, expand_home};
use crate::seq::Corpus;
use crate::store::TeacherStore;
use crate::teacher_data::load_teacher;
use crate::trainer::RunDir;

/// The three classes of post-freeze starts (8.5a review, F1): what an idle blocker does from the handover on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StartClass {
    /// The victim gets free although the idle blocker stays in (the held-block task proper).
    V,
    /// The idle blocker itself goes out (frozen or dead): the fly's own hazard.
    B,
    /// An idle blocker holds the block: the geometry does the work.
    H,
}

impl StartClass {
    pub const ALL: [StartClass; 3] = [StartClass::V, StartClass::B, StartClass::H];

    pub fn index(self) -> usize {
        self as usize
    }
}

/// The class of a start; `Err` for a bank that has no tags (a version 1 file: `train es retag`).
pub fn start_class(s: &BankStart) -> Result<StartClass, String> {
    match (s.victim_escapes_under_idle, s.idle_blocker_out) {
        (Some(true), _) => Ok(StartClass::V),
        (Some(false), Some(true)) => Ok(StartClass::B),
        (Some(false), Some(false)) => Ok(StartClass::H),
        _ => Err("the bank has no victim-escape tags (a version 1 file): run `ddnet-ai train es retag` first".into()),
    }
}

/// The index drawn by `u` in `[0, sum(w))` from the weights `w` (zero-weight entries are never drawn; the last positive one catches rounding).
fn pick_weighted(w: &[f32], mut u: f32) -> usize {
    let mut last = 0;
    for (i, &wi) in w.iter().enumerate() {
        if wi <= 0.0 {
            continue;
        }
        last = i;
        if u < wi {
            return i;
        }
        u -= wi;
    }
    last
}

fn mix(a: u64, b: u64, c: u64) -> u64 {
    crate::es::noise::pair_seed(a ^ 0xBB00_BB00_BB00_BB00, b, c)
}

/// Everything a run loads once.
pub struct Loaded {
    pub env: Env,
    pub base: FlyBundle,
    pub flyg: ddai_flyg::Flyg,
    pub flyg_path: std::path::PathBuf,
    pub flyg_sha: String,
    pub bank: Bank,
    pub fields: BTreeMap<String, MapFields>,
}

pub fn load(cfg: &PpoConfig) -> Result<Loaded, String> {
    let flyg_path = expand_home(&cfg.flyg);
    let env = crate::experiment::load_env_with_scenarios(
        &expand_home(&cfg.arenas_dir),
        &expand_home(&cfg.map_dir),
        Some(flyg_path.clone()),
        cfg.scenarios_dir.as_deref().map(expand_home).as_deref(),
    )?;
    for a in &cfg.train_arenas {
        match env.arenas.get(a) {
            None => return Err(format!("unknown training arena {a:?}")),
            Some(x) if x.tag.label() != "train" => {
                return Err(format!(
                    "training arena {a:?} is tagged {} (a holdout never trains)",
                    x.tag.label()
                ));
            }
            Some(_) => {}
        }
    }
    let base = load_bundle(&expand_home(&cfg.init_bundle)).map_err(|e| e.to_string())?;
    let flyg_sha = sha256_hex_of_file(&flyg_path).map_err(|e| e.to_string())?;
    if flyg_sha != base.flyg_sha256 {
        return Err(format!(
            "{}: built for another .flyg than {}",
            cfg.init_bundle, cfg.flyg
        ));
    }
    let flyg = ddai_flyg::load(&flyg_path).map_err(|e| format!("{}: {e}", flyg_path.display()))?;
    let bank = Bank::load(&expand_home(&cfg.bank))?;
    let fields = env
        .arenas
        .iter()
        .map(|(n, a)| (n.clone(), MapFields::new(&a.map)))
        .collect();
    Ok(Loaded {
        env,
        base,
        flyg,
        flyg_path,
        flyg_sha,
        bank,
        fields,
    })
}

fn meta(cfg: &PpoConfig, iteration: u64) -> BundleMeta {
    let (commit, dirty) = ddai_env::output::git_info(Path::new(env!("CARGO_MANIFEST_DIR")));
    BundleMeta {
        seed: cfg.seed,
        git_commit: Some(if dirty { format!("{commit}+dirty") } else { commit }),
        steps: iteration,
        notes: format!("{}: recurrent PPO iteration {iteration}", cfg.name),
    }
}

/// One episode of an iteration: what to play and against whom.
#[derive(Debug, Clone, PartialEq)]
pub enum Pick {
    /// Index into the training starts.
    Post(usize),
    /// A curriculum episode: index into the training starts and the offset (ticks of the demonstration's play before the fly takes over).
    Demo(usize, i32),
    /// `(arena index into train_arenas, seed, layout game index)`.
    Game(usize, u64, u32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub pick: Pick,
    /// Index into `rollout.opponents`.
    pub opponent: usize,
    /// The league snapshot of a `past` opponent.
    pub past: Option<usize>,
}

/// The episodes of `iteration`, a function of the seed and the iteration only: `classes[c]` are the indices (into the training starts) of the
/// starts of class `c`, `n_snapshots` the earlier policies in the league.
pub fn iteration_plan(
    cfg: &PpoConfig,
    iteration: u64,
    classes: &[Vec<usize>; 3],
    demo_classes: &[Vec<usize>; 3],
    curriculum: &CurriculumState,
    n_snapshots: usize,
) -> Vec<Plan> {
    let r = &cfg.rollout;
    let total: f32 = r.opponents.iter().map(|(_, w)| *w).sum();
    let pick_opponent = |rng: &mut SplitMix64| -> (usize, Option<usize>) {
        let mut u = rng.next_f32_unit() * total;
        let mut idx = r.opponents.len() - 1;
        for (i, (_, w)) in r.opponents.iter().enumerate() {
            if u < *w {
                idx = i;
                break;
            }
            u -= *w;
        }
        let past =
            (r.opponents[idx].0 == "past" && n_snapshots > 0).then(|| (rng.next_u64() % n_snapshots as u64) as usize);
        (idx, past)
    };
    // The class weights, restricted to the classes that have starts.
    let w: Vec<f32> = (0..3)
        .map(|c| if classes[c].is_empty() { 0.0 } else { r.start_mix[c] })
        .collect();
    let w_total: f32 = w.iter().sum();
    let na = cfg.train_arenas.len();
    let mut plans = Vec::with_capacity(r.post_episodes + r.game_episodes);
    // The curriculum's share of the post-freeze episodes comes first.
    let cur = &cfg.curriculum;
    let dw: Vec<f32> = (0..3)
        .map(|c| if demo_classes[c].is_empty() { 0.0 } else { cur.mix[c] })
        .collect();
    let dw_total: f32 = dw.iter().sum();
    let n_cur = if cur.enabled && dw_total > 0.0 {
        (cur.share * r.post_episodes as f32).round() as usize
    } else {
        0
    };
    let ladders: Vec<Vec<i32>> = (0..3).map(|c| curriculum.levels(cur, c)).collect();
    for i in 0..n_cur {
        let mut rng = SplitMix64::new(mix(cfg.seed, iteration, 0x2000 + i as u64));
        let class = pick_weighted(&dw, rng.next_f32_unit() * dw_total);
        let list = &demo_classes[class];
        let idx = list[(rng.next_u64() % list.len().max(1) as u64) as usize];
        let offset = if rng.next_f32_unit() < cur.current_share {
            curriculum.offsets[class]
        } else {
            ladders[class][(rng.next_u64() % ladders[class].len() as u64) as usize]
        };
        let (opponent, past) = pick_opponent(&mut rng);
        plans.push(Plan {
            pick: Pick::Demo(idx, offset),
            opponent,
            past,
        });
    }
    for i in 0..r.post_episodes.saturating_sub(n_cur) {
        let mut rng = SplitMix64::new(mix(cfg.seed, iteration, i as u64));
        let class = pick_weighted(&w, rng.next_f32_unit() * w_total);
        let list = &classes[class];
        let idx = list[(rng.next_u64() % list.len().max(1) as u64) as usize];
        let (opponent, past) = pick_opponent(&mut rng);
        plans.push(Plan {
            pick: Pick::Post(idx),
            opponent,
            past,
        });
    }
    for j in 0..r.game_episodes {
        let mut rng = SplitMix64::new(mix(cfg.seed, iteration, 0x1000 + j as u64));
        let seed = rng.next_u64() % 1_000_000_000_000 + 3_000_000_000;
        let (opponent, past) = pick_opponent(&mut rng);
        plans.push(Plan {
            pick: Pick::Game(j % na, seed, ((iteration as usize * r.game_episodes + j) / na) as u32),
            opponent,
            past,
        });
    }
    plans
}

/// Aggregates of an iteration's rollouts, for the metrics.
#[derive(Debug, Clone, Default)]
pub struct RolloutStats {
    pub post_n: usize,
    pub post_held: usize,
    pub post_esc_n: usize,
    pub post_esc_held: usize,
    /// Per start class: episodes, held, fly out in the window.
    pub by_class: [(usize, usize, usize); 3],
    pub post_self_freeze: usize,
    pub post_return: f64,
    pub game_n: usize,
    pub game_credited: usize,
    pub game_held: usize,
    pub game_lost: usize,
    pub game_timeout: usize,
    pub game_return: f64,
    pub decisions: usize,
    pub freed_victim: usize,
    pub shaping_sum: f64,
    pub terminal_sum: f64,
    pub by_opponent: BTreeMap<String, (usize, usize)>,
    /// Curriculum episodes: played, held, the fly out.
    pub curr_n: usize,
    pub curr_held: usize,
    pub curr_out: usize,
    /// The opening of the plain V starts: the first 4 acted decisions of every such episode, and how many of them held the hook / fired (the planner
    /// hooks the victim in 80% of them, E-027).
    pub open_n: usize,
    pub open_hook: usize,
    pub open_fire: usize,
}

pub fn rollout_stats(eps: &[Episode]) -> RolloutStats {
    let mut s = RolloutStats::default();
    for e in eps {
        s.decisions += e.acted_decisions();
        s.freed_victim += usize::from(e.parts.freed_victim);
        s.shaping_sum += f64::from(e.parts.shaping);
        s.terminal_sum += f64::from(e.parts.terminal);
        let o = &e.outcome;
        if e.offset > 0 {
            // A curriculum episode: counted on its own, not in the plain start numbers.
            s.curr_n += 1;
            s.curr_held += usize::from(o.held_block);
            s.curr_out += usize::from(o.focal_out_in_window);
            continue;
        }
        match e.kind {
            EpisodeKind::Post => {
                s.post_n += 1;
                s.post_held += usize::from(o.held_block);
                s.post_self_freeze += usize::from(o.focal_out_in_window);
                s.post_return += f64::from(e.total_reward());
                if let Some(c) = e.class {
                    let b = &mut s.by_class[c.index()];
                    b.0 += 1;
                    b.1 += usize::from(o.held_block);
                    b.2 += usize::from(o.focal_out_in_window);
                    if c == StartClass::V {
                        s.post_esc_n += 1;
                        s.post_esc_held += usize::from(o.held_block);
                        for d in e.decisions.iter().filter(|d| d.acted).take(4) {
                            s.open_n += 1;
                            s.open_hook += usize::from(d.action.hook);
                            s.open_fire += usize::from(d.action.fire);
                        }
                    }
                }
                let en = s.by_opponent.entry(e.opponent.clone()).or_default();
                en.0 += 1;
                en.1 += usize::from(o.held_block);
            }
            EpisodeKind::Game => {
                s.game_n += 1;
                s.game_credited += usize::from(o.result == ddai_env::stats::GameResult::W && o.credited);
                s.game_held += usize::from(o.result == ddai_env::stats::GameResult::W && o.credited && o.held_block);
                s.game_lost += usize::from(o.result == ddai_env::stats::GameResult::L);
                s.game_timeout += usize::from(o.result == ddai_env::stats::GameResult::T);
                s.game_return += f64::from(e.total_reward());
            }
        }
    }
    s
}

fn bc_data(cfg: &PpoConfig, loaded: &Loaded, log: &mut dyn FnMut(&str)) -> Result<Option<BcData>, String> {
    if cfg.ppo.bc_coef <= 0.0 || cfg.ppo.bc_windows == 0 {
        return Ok(None);
    }
    let holdout = loaded.env.holdout_names();
    let mut train = Vec::new();
    for dir in &cfg.bc.teacher_dirs {
        let store = TeacherStore::open(&expand_home(dir)).map_err(|e| e.to_string())?;
        let split = load_teacher(
            &store,
            &store.chunks_of_round(None),
            &loaded.env.maps,
            &holdout,
            &cfg.bc.teacher_data,
            cfg.threads,
        )
        .map_err(|e| e.to_string())?;
        log(&format!("bc data {dir}: {} training episodes", split.train.len()));
        train.extend(split.train);
    }
    Ok(Some(BcData {
        corpus: Corpus::new(train),
        cfg: cfg.bc.clone(),
    }))
}

/// Evaluates the fly of a bundle on the config's evaluation episodes (deterministic play, the way the bot plays).
pub fn evaluate_bundle(
    cfg: &PpoConfig,
    loaded: &Loaded,
    pool: &rayon::ThreadPool,
    bundle: FlyBundle,
    iteration: u64,
) -> Result<EvalPoint, String> {
    let t = FlyBrainTemplate::from_parts(bundle, loaded.flyg.clone(), &loaded.flyg_sha).map_err(|e| e.to_string())?;
    let maker = || Ok(t.instantiate_played(FlyBrainConfig::default()));
    let spec = EvalParams {
        eval: &cfg.eval,
        train_arenas: &cfg.train_arenas,
        window_ticks: cfg.rollout.window_ticks,
        burn_in_ticks: cfg.rollout.burn_in_ticks,
    };
    evaluate_spec(&loaded.env, pool, &spec, &loaded.bank, &maker, iteration)
}

/// Plays the episodes of one iteration with the policy of `bundle`, in plan order whatever the thread count.
#[allow(clippy::too_many_arguments)]
pub fn rollout(
    cfg: &PpoConfig,
    loaded: &Loaded,
    pool: &rayon::ThreadPool,
    bundle: FlyBundle,
    train_starts: &[&BankStart],
    plans: &[Plan],
    snapshots: &[String],
    demos: Option<&DemoSet>,
    mode: ActMode,
) -> Result<Vec<Episode>, String> {
    let template =
        FlyBrainTemplate::from_parts(bundle, loaded.flyg.clone(), &loaded.flyg_sha).map_err(|e| e.to_string())?;
    let rules = loaded.bank.rules.clone();
    let ctx = RolloutCtx {
        arenas: &loaded.env.arenas,
        fields: &loaded.fields,
        template: &template,
        rules: &rules,
        window: cfg.rollout.window_ticks,
        burn_in_ticks: cfg.rollout.burn_in_ticks,
        grid: WindowGrid {
            chunk: cfg.ppo.chunk,
            burn_in: cfg.ppo.burn_in,
            decide_every: rules.decide_every,
        },
        aim_kappa: cfg.rollout.aim_kappa,
        temps: cfg.rollout.temperatures(),
        gamma: cfg.ppo.gamma,
        reward: &cfg.reward,
        mode,
    };
    let factory = loaded.env.models.factory();
    let scripted = || factory(&PlayerSpec::simple("scripted"));
    let results: Vec<Result<Episode, String>> = pool.install(|| {
        plans
            .par_iter()
            .map(|p| {
                let (label, arg): (String, Option<String>) = {
                    let (name, _) = &cfg.rollout.opponents[p.opponent];
                    match (name.as_str(), p.past) {
                        ("past", Some(i)) => (format!("past:{i}"), Some(format!("fly:{}", snapshots[i]))),
                        ("past", None) | ("scripted", _) => ("scripted".to_string(), None),
                        (other, _) => (other.to_string(), Some(other.to_string())),
                    }
                };
                let make_opp = || factory(&ddai_env::models::player_from_arg(arg.as_deref().unwrap_or("scripted")));
                let opponent: Option<(&str, &OpponentMaker<'_>)> =
                    arg.as_ref().map(|_| (label.as_str(), &make_opp as &OpponentMaker<'_>));
                let mut ep = match &p.pick {
                    Pick::Post(i) => play_episode_ppo(&ctx, &EpisodeSpec::Post(train_starts[*i]), &scripted, opponent),
                    Pick::Demo(i, offset) => {
                        let start = train_starts[*i];
                        let demo = demos
                            .and_then(|d| d.held_demo(start))
                            .ok_or_else(|| format!("no demonstration for start {} seed {}", start.arena, start.seed))?;
                        let (log, handover) = resumed_log(start, demo, *offset);
                        play_episode_ppo(
                            &ctx,
                            &EpisodeSpec::Resumed {
                                start,
                                log,
                                handover,
                                offset: *offset,
                            },
                            &scripted,
                            opponent,
                        )
                    }
                    Pick::Game(a, seed, g) => {
                        let arena = &loaded.env.arenas[&cfg.train_arenas[*a]];
                        play_episode_ppo(
                            &ctx,
                            &EpisodeSpec::Game {
                                arena: &cfg.train_arenas[*a],
                                seed: *seed,
                                layout: ddai_env::run::layout_of(arena, *g),
                            },
                            &scripted,
                            opponent,
                        )
                    }
                }?;
                ep.opponent = label;
                Ok(ep)
            })
            .collect()
    });
    results.into_iter().collect()
}

type OpponentMaker<'a> = dyn Fn() -> Result<Box<dyn ddai_brain::Brain>, ddai_env::EnvError> + 'a;

fn selection_score(point: &EvalPoint) -> f64 {
    point.selection_score()
}

/// Runs (or resumes) the PPO. Returns the final state.
pub fn run_ppo(cfg: &PpoConfig, allow_config_change: bool, log: &mut dyn FnMut(&str)) -> Result<PpoState, String> {
    cfg.validate()?;
    let loaded = load(cfg)?;
    let run = RunDir::create(&cfg.run_path()).map_err(|e| e.to_string())?;
    record_ppo_config(&run, cfg, allow_config_change, log)?;
    let pool = make_pool(cfg.threads)?;
    let train_starts: Vec<&BankStart> = loaded.bank.select(&cfg.train_arenas, Some(false), false);
    if train_starts.is_empty() {
        return Err("the bank has no training start on the training arenas".into());
    }
    let mut classes: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (i, s) in train_starts.iter().enumerate() {
        classes[start_class(s)?.index()].push(i);
    }
    if cfg.rollout.start_subset > 0 {
        for c in &mut classes {
            c.truncate(cfg.rollout.start_subset);
        }
    }
    let demos: Option<DemoSet> = ensure_demos(cfg, &loaded, &train_starts, &pool, log)?;
    let mut demo_classes: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    if let Some(d) = &demos {
        for (i, s) in train_starts.iter().enumerate() {
            if d.held_demo(s).is_some() {
                demo_classes[start_class(s)?.index()].push(i);
            }
        }
        if cfg.rollout.start_subset > 0 {
            for c in &mut demo_classes {
                c.truncate(cfg.rollout.start_subset);
            }
        }
        log(&format!(
            "curriculum: demonstrations for V {}, B {}, H {} training starts",
            demo_classes[0].len(),
            demo_classes[1].len(),
            demo_classes[2].len()
        ));
    }
    let bc = bc_data(cfg, &loaded, log)?;
    let mut learner = PpoLearner::new(
        cfg.ppo.clone(),
        &loaded.base,
        &loaded.flyg_path,
        cfg.rollout.aim_kappa,
        cfg.rollout.temperatures(),
        cfg.seed,
        bc,
    )?;
    let state_path = run.path("state.bin");
    let mut state: PpoState = if state_path.exists() {
        let s: PpoState = read_zstd_postcard(&state_path).map_err(|e| e.to_string())?;
        learner.from_state(&s)?;
        log(&format!("resumed at iteration {}", s.iteration));
        s
    } else {
        PpoState {
            iteration: 0,
            theta: learner.theta.clone(),
            adam: learner.adam.clone(),
            critic: learner.critic.clone(),
            critic_adam: learner.critic_adam.clone(),
            beta_kl: learner.beta_kl,
            best_score: f64::NEG_INFINITY,
            best_iter: 0,
            snapshots: Vec::new(),
            curriculum: CurriculumState::new(&cfg.curriculum),
            dagger_ids: Vec::new(),
        }
    };
    // The DAgger chunks a resumed run had added (exactly those: a chunk written after the last state is not part of it).
    if !state.dagger_ids.is_empty() && learner.has_bc() {
        let store = TeacherStore::open(&run.path("dagger")).map_err(|e| e.to_string())?;
        for &id in &state.dagger_ids {
            learner.append_bc(load_chunks(cfg, &loaded, &store, &[id])?);
        }
        log(&format!(
            "resumed {} dagger chunks into the BC corpus",
            state.dagger_ids.len()
        ));
    }
    log(&format!(
        "PPO {}: {} parameters ({} on the opponent-state channels), critic {} inputs, {} training starts (V {}, B {}, H {}), {} + {} episodes per iteration",
        cfg.name,
        learner.theta.len(),
        learner.num_new_channel_params(),
        INPUT_DIM,
        train_starts.len(),
        classes[0].len(),
        classes[1].len(),
        classes[2].len(),
        cfg.rollout.post_episodes,
        cfg.rollout.game_episodes,
    ));
    let t_start = Instant::now();
    let mut evaluated: std::collections::HashSet<u64> = std::fs::read_to_string(run.path("metrics.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v["kind"] == "ppo_eval")
        .filter_map(|v| v["iteration"].as_u64())
        .collect();
    let de = loaded.bank.rules.decide_every;

    let mut do_eval = |learner: &PpoLearner,
                       state: &mut PpoState,
                       iteration: u64,
                       log: &mut dyn FnMut(&str)|
     -> Result<(), String> {
        if !evaluated.insert(iteration) {
            return Ok(());
        }
        let t0 = Instant::now();
        let bundle = with_params(&loaded.base, &learner.theta)?;
        let point = evaluate_bundle(cfg, &loaded, &pool, bundle.clone(), iteration)?;
        let score = selection_score(&point);
        let (ng, nc, nmax) = learner.new_channel_norms();
        let mut line = serde_json::to_value(&point).map_err(|e| e.to_string())?;
        strip_items(&mut line);
        line["kind"] = json!("ppo_eval");
        line["iteration"] = json!(iteration);
        line["score"] = json!(score);
        line["eval_s"] = json!(t0.elapsed().as_secs_f64());
        line["new_g_norm"] = json!(ng);
        line["new_c_norm"] = json!(nc);
        line["new_max_abs"] = json!(nmax);
        run.append_metrics(&line).map_err(|e| e.to_string())?;
        run.append_metrics(&arena_line(&point, "ppo", "train-halls", &point.train_games))
            .map_err(|e| e.to_string())?;
        if let Some(h) = &point.holdout_games {
            run.append_metrics(&arena_line(&point, "ppo", &cfg.eval.holdout_arenas.join("+"), h))
                .map_err(|e| e.to_string())?;
        }
        save_bundle(&run.checkpoint(&format!("step-{iteration:05}.bundle")), &bundle).map_err(|e| e.to_string())?;
        if score > state.best_score {
            state.best_score = score;
            state.best_iter = iteration;
            save_bundle(&run.checkpoint("selected.bundle"), &bundle).map_err(|e| e.to_string())?;
            // Persist the selection right away (review 8.5b F3): a kill before the end of the iteration must not resume with the old best score
            // (the eval is already in the metrics and would be skipped) and let a worse checkpoint overwrite `selected.bundle`.
            write_zstd_postcard(&state_path, &*state, 3).map_err(|e| e.to_string())?;
            run.append_metrics(
                &json!({"kind": "selection", "phase": format!("ppo-{iteration}"), "arenas": cfg.train_arenas,
                "table": [[format!("ppo-{iteration}"), score]]}),
            )
            .map_err(|e| e.to_string())?;
        }
        log(&format!(
            "eval iteration {iteration}: held train-val {} (V {}) | holdout {} (V {}) | own freeze train {} | first freeze train {} | holdout {} ({:.0}s)",
            point.train_starts.held.fmt_pct(),
            point
                .train_starts
                .held_victim_escapable
                .unwrap_or(point.train_starts.held_escapable)
                .fmt_pct(),
            point.holdout_starts.as_ref().map_or("-".into(), |s| s.held.fmt_pct()),
            point.holdout_starts.as_ref().map_or("-".into(), |s| s
                .held_victim_escapable
                .unwrap_or(s.held_escapable)
                .fmt_pct()),
            point.train_starts.self_freeze.fmt_pct(),
            point.train_games.credited.fmt_pct(),
            point
                .holdout_games
                .as_ref()
                .map_or("-".into(), |g| g.credited.fmt_pct()),
            t0.elapsed().as_secs_f64()
        ));
        Ok(())
    };

    while state.iteration < cfg.iterations {
        let iteration = state.iteration;
        if iteration == 0 || (cfg.eval.every > 0 && iteration.is_multiple_of(cfg.eval.every)) {
            do_eval(&learner, &mut state, iteration, log)?;
        }
        if cfg.max_hours > 0.0 && t_start.elapsed().as_secs_f64() > f64::from(cfg.max_hours) * 3600.0 {
            log(&format!(
                "wall limit of {} h reached at iteration {iteration}",
                cfg.max_hours
            ));
            break;
        }
        let t0 = Instant::now();
        let plans = iteration_plan(
            cfg,
            iteration,
            &classes,
            &demo_classes,
            &state.curriculum,
            state.snapshots.len(),
        );
        let bundle = with_params(&loaded.base, &learner.theta)?;
        // Always the sampled policy: the update is on-policy (a deterministic rollout would make the ratio meaningless).
        let mode = ActMode::Sample;
        let episodes = rollout(
            cfg,
            &loaded,
            &pool,
            bundle,
            &train_starts,
            &plans,
            &state.snapshots,
            demos.as_ref(),
            mode,
        )?;
        let rs = rollout_stats(&episodes);
        let secs_rollout = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let us: UpdateStats = learner.update(&episodes, &loaded.fields, iteration, de, &pool)?;
        let secs_update = t1.elapsed().as_secs_f64();
        // The curriculum: did the fly hold the block often enough at the current level?
        if cfg.curriculum.enabled {
            for class in StartClass::ALL {
                let c = class.index();
                let level = state.curriculum.offsets[c];
                let at_level: Vec<bool> = episodes
                    .iter()
                    .filter(|e| e.class == Some(class) && e.offset == level && level > 0)
                    .map(|e| e.outcome.held_block)
                    .collect();
                if state
                    .curriculum
                    .advance(&cfg.curriculum, state.iteration + 1, c, &at_level)
                {
                    log(&format!(
                        "curriculum {class:?}: offset {level} -> {} (held {} of {} in this iteration at the level)",
                        state.curriculum.offsets[c],
                        at_level.iter().filter(|&&h| h).count(),
                        at_level.len()
                    ));
                }
            }
        }
        state.iteration += 1;
        state.theta.clone_from(&learner.theta);
        state.adam = learner.adam.clone();
        state.critic = learner.critic.clone();
        state.critic_adam = learner.critic_adam.clone();
        state.beta_kl = learner.beta_kl;
        if cfg.snapshot_every > 0 && state.iteration.is_multiple_of(cfg.snapshot_every) {
            let path = run.path(&format!("snapshots/it-{:05}.bundle", state.iteration));
            std::fs::create_dir_all(path.parent().expect("a parent")).map_err(|e| e.to_string())?;
            let b = with_params(&loaded.base, &learner.theta)?;
            save_bundle(&path, &b).map_err(|e| e.to_string())?;
            state.snapshots.push(path.to_string_lossy().into_owned());
        }
        // DAgger: the current fly plays, the planner labels what it visits, the labels join the BC term.
        if cfg.dagger.every > 0 && state.iteration.is_multiple_of(cfg.dagger.every) {
            let t2 = Instant::now();
            let (n, ids, seqs) = dagger_round(
                cfg,
                &loaded,
                &run,
                &learner,
                &train_starts,
                &classes,
                state.iteration,
                log,
            )?;
            learner.append_bc(seqs);
            state.dagger_ids.extend(ids);
            log(&format!(
                "dagger: {n} labelled decisions added ({:.0}s)",
                t2.elapsed().as_secs_f64()
            ));
        }
        write_zstd_postcard(&state_path, &state, 3).map_err(|e| e.to_string())?;
        let last = with_params(&loaded.base, &learner.theta).map(|mut b| {
            b.meta = meta(cfg, state.iteration);
            b
        })?;
        save_bundle(&run.checkpoint("last.bundle"), &last).map_err(|e| e.to_string())?;

        let frac = |k: usize, n: usize| if n == 0 { f64::NAN } else { k as f64 / n as f64 };
        let kl_total: f64 = us.kl_ref.iter().sum();
        run.append_metrics(&json!({"kind": "train", "phase": "ppo", "step": state.iteration,
            "loss": {"total": us.pg_loss, "pg": us.pg_loss, "value": us.value_loss, "bc": us.bc_loss}, "grad_norm": us.grad_norm}))
            .map_err(|e| e.to_string())?;
        run.append_metrics(&json!({"kind": "ppo", "iteration": state.iteration,
            "post_n": rs.post_n, "post_held": frac(rs.post_held, rs.post_n), "post_held_v": frac(rs.post_esc_held, rs.post_esc_n), "post_n_v": rs.post_esc_n,
            "post_own_v": frac(rs.by_class[0].2, rs.by_class[0].0), "post_held_b": frac(rs.by_class[1].1, rs.by_class[1].0), "post_own_b": frac(rs.by_class[1].2, rs.by_class[1].0),
            "post_held_h": frac(rs.by_class[2].1, rs.by_class[2].0), "post_own_h": frac(rs.by_class[2].2, rs.by_class[2].0),
            "post_self_freeze": frac(rs.post_self_freeze, rs.post_n), "post_return": rs.post_return / rs.post_n.max(1) as f64,
            "game_n": rs.game_n, "game_credited": frac(rs.game_credited, rs.game_n), "game_held": frac(rs.game_held, rs.game_n),
            "game_lost": frac(rs.game_lost, rs.game_n), "game_timeouts": frac(rs.game_timeout, rs.game_n), "game_return": rs.game_return / rs.game_n.max(1) as f64,
            "decisions": rs.decisions, "freed_victim": rs.freed_victim, "shaping_sum": rs.shaping_sum, "terminal_sum": rs.terminal_sum,
            "by_opponent": rs.by_opponent.iter().map(|(k, (n, h))| (k.clone(), json!([n, h]))).collect::<serde_json::Map<_, _>>(),
            "open_hook_v": frac(rs.open_hook, rs.open_n), "open_fire_v": frac(rs.open_fire, rs.open_n), "curr_offset": state.curriculum.offsets[0], "curr_offset_b": state.curriculum.offsets[1], "curr_offset_h": state.curriculum.offsets[2], "curr_n": rs.curr_n, "curr_held": frac(rs.curr_held, rs.curr_n), "curr_out": frac(rs.curr_out, rs.curr_n),
            "policy_updated": us.policy_updated, "windows": us.windows, "minibatches": us.minibatches, "epochs_run": us.epochs_run,
            "pg_loss": us.pg_loss, "ratio_mean": us.ratio_mean, "clip_frac": us.clip_frac,
            "entropy": us.entropy, "kl_ref": us.kl_ref, "kl_ref_total": kl_total, "kl_old": us.kl_old, "beta_kl": learner.beta_kl,
            "bc_loss": us.bc_loss, "grad_norm": us.grad_norm, "grad_norm_new_channels": us.grad_norm_new_channels,
            "value_loss": us.value_loss, "explained_variance": us.explained_variance, "return_mean": us.return_mean, "adv_std": us.adv_std,
            "new_g_norm": us.weight_norm_new_g, "new_c_norm": us.weight_norm_new_c, "new_max_abs": us.max_abs_new,
            "secs_rollout": secs_rollout, "secs_update": secs_update, "secs_forward": us.secs_forward, "secs_train": us.secs_train,
            "secs_critic": us.secs_critic, "iter_s": t0.elapsed().as_secs_f64()}))
            .map_err(|e| e.to_string())?;
        run.write_status(&json!({"phase": "ppo", "step": state.iteration, "phase_step": state.iteration, "phase_steps": cfg.iterations,
            "loss": us.pg_loss, "elapsed_s": t_start.elapsed().as_secs_f64(), "unix_s": unix_seconds()}))
            .map_err(|e| e.to_string())?;
        log(&format!(
            "iteration {}: curriculum offsets V {} B {} (held {:.0}% of {}) | held {:.0}% (V {:.0}%, {} of {}; own freeze on V {:.0}%, B {:.0}%) own freeze {:.0}% | games credited {:.0}% | return post {:+.2} game {:+.2} | KL ref {:.4} old {:.4} beta {:.2} | EV {:.2} | ent {:.2} | |new w| {:.3} | {} dec, {:.0}s (rollout {:.0}s, update {:.0}s)",
            state.iteration,
            state.curriculum.offsets[0],
            state.curriculum.offsets[1],
            100.0 * frac(rs.curr_held, rs.curr_n),
            rs.curr_n,
            100.0 * frac(rs.post_held, rs.post_n),
            100.0 * frac(rs.post_esc_held, rs.post_esc_n),
            rs.post_esc_held,
            rs.post_esc_n,
            100.0 * frac(rs.by_class[0].2, rs.by_class[0].0),
            100.0 * frac(rs.by_class[1].2, rs.by_class[1].0),
            100.0 * frac(rs.post_self_freeze, rs.post_n),
            100.0 * frac(rs.game_credited, rs.game_n),
            rs.post_return / rs.post_n.max(1) as f64,
            rs.game_return / rs.game_n.max(1) as f64,
            kl_total,
            us.kl_old,
            learner.beta_kl,
            us.explained_variance,
            us.entropy.iter().sum::<f64>(),
            us.weight_norm_new_g + us.weight_norm_new_c,
            rs.decisions,
            t0.elapsed().as_secs_f64(),
            secs_rollout,
            secs_update,
        ));
    }
    let it = state.iteration;
    do_eval(&learner, &mut state, it, log)?;
    write_zstd_postcard(&state_path, &state, 3).map_err(|e| e.to_string())?;
    let final_bundle = with_params(&loaded.base, &learner.theta).map(|mut b| {
        b.meta = meta(cfg, state.iteration);
        b
    })?;
    save_bundle(&run.checkpoint("final.bundle"), &final_bundle).map_err(|e| e.to_string())?;
    run.write_status(
        &json!({"phase": "done", "step": state.iteration, "phase_step": state.iteration, "phase_steps": cfg.iterations,
        "elapsed_s": t_start.elapsed().as_secs_f64(), "unix_s": unix_seconds()}),
    )
    .map_err(|e| e.to_string())?;
    Ok(state)
}

/// The demonstrations of the reverse curriculum: loaded from `curriculum.demos` when it matches the bank's training starts, else recorded (the
/// planner plays every training start once) and saved there. `None` when the curriculum is off.
pub fn ensure_demos(
    cfg: &PpoConfig,
    loaded: &Loaded,
    train_starts: &[&BankStart],
    pool: &rayon::ThreadPool,
    log: &mut dyn FnMut(&str),
) -> Result<Option<DemoSet>, String> {
    if !cfg.curriculum.enabled {
        return Ok(None);
    }
    let path = expand_home(&cfg.curriculum.demos);
    let fp = bank_fingerprint(train_starts);
    match DemoSet::load(&path) {
        Ok(d) if d.fingerprint == fp => Ok(Some(d)),
        other => {
            if other.is_ok() || path.exists() {
                log("the demonstration file does not match the bank: recording it again");
            }
            let t0 = Instant::now();
            let d = build_demos(&loaded.env, &loaded.bank, train_starts, pool, log)?;
            d.save(&path)?;
            log(&format!(
                "demonstrations recorded in {:.0}s",
                t0.elapsed().as_secs_f64()
            ));
            Ok(Some(d))
        }
    }
}

/// The training starts of a run: the training part of the bank on the training halls.
pub fn training_starts<'a>(cfg: &PpoConfig, loaded: &'a Loaded) -> Vec<&'a BankStart> {
    loaded.bank.select(&cfg.train_arenas, Some(false), false)
}

fn load_chunks(
    cfg: &PpoConfig,
    loaded: &Loaded,
    store: &TeacherStore,
    ids: &[usize],
) -> Result<Vec<crate::seq::Seq>, String> {
    let mut td = cfg.bc.teacher_data.clone();
    if !td.round_weights.iter().any(|(r, _)| *r == config::DAGGER_ROUND) {
        td.round_weights.push((config::DAGGER_ROUND, cfg.dagger.round_weight));
    }
    let holdout = loaded.env.holdout_names();
    let split = load_teacher(store, ids, &loaded.env.maps, &holdout, &td, cfg.threads).map_err(|e| e.to_string())?;
    Ok(split.train)
}

/// One DAgger round: the current fly plays `dagger.starts` training starts of the configured classes, the planner labels every state, and the
/// new chunks of the store are returned (to join the BC corpus) with the labelled decisions.
#[allow(clippy::too_many_arguments)]
fn dagger_round(
    cfg: &PpoConfig,
    loaded: &Loaded,
    run: &RunDir,
    learner: &PpoLearner,
    train_starts: &[&BankStart],
    classes: &[Vec<usize>; 3],
    iteration: u64,
    log: &mut dyn FnMut(&str),
) -> Result<(u64, Vec<usize>, Vec<crate::seq::Seq>), String> {
    use crate::bank_collect::collect_starts;
    use crate::collect::Mixing;
    let d = &cfg.dagger;
    let w: Vec<f32> = (0..3)
        .map(|c| if classes[c].is_empty() { 0.0 } else { d.mix[c] })
        .collect();
    let w_total: f32 = w.iter().sum();
    if w_total <= 0.0 {
        return Ok((0, Vec::new(), Vec::new()));
    }
    let mut picked: Vec<&BankStart> = Vec::with_capacity(d.starts);
    for i in 0..d.starts {
        let mut rng = SplitMix64::new(mix(cfg.seed, iteration, 0x3000 + i as u64));
        let class = pick_weighted(&w, rng.next_f32_unit() * w_total);
        let list = &classes[class];
        picked.push(train_starts[list[(rng.next_u64() % list.len() as u64) as usize]]);
    }
    let dir = run.path("dagger");
    std::fs::create_dir_all(dir.join("bundles")).map_err(|e| e.to_string())?;
    let bundle_path = dir.join("bundles").join(format!("fly-{iteration:05}.bundle"));
    save_bundle(&bundle_path, &with_params(&loaded.base, &learner.theta)?).map_err(|e| e.to_string())?;
    let mut store = TeacherStore::open_or_create(&dir, "ppo-dagger", "8.5b").map_err(|e| e.to_string())?;
    let before = store.manifest.chunks.len();
    let summary = collect_starts(
        &loaded.env,
        &mut store,
        &picked,
        &loaded.bank.rules,
        &format!("fly:{}", bundle_path.display()),
        Mixing::default(),
        cfg.rollout.window_ticks,
        cfg.rollout.burn_in_ticks,
        config::DAGGER_ROUND,
        cfg.threads,
        log,
    )?;
    let _ = std::fs::remove_file(&bundle_path);
    let ids: Vec<usize> = (before..store.manifest.chunks.len()).collect();
    let seqs = load_chunks(cfg, loaded, &store, &ids)?;
    Ok((summary.steps, ids, seqs))
}
