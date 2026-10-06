//! OpenAI-ES on the fly, from an imitation checkpoint, on the outcome of the held block (task 8.5a).
//!
//! One generation: `pairs` antithetic noise vectors (`theta +- sigma * eps`, drawn from a seed that is a function of
//! `(run seed, generation, pair)` only), every one of the `2 * pairs` members plays **the same episodes** (common random numbers:
//! `post_episodes` post-freeze starts from the training bank and `normal_games` full games against the scripted bot), the returns are
//! rank-shaped, the gradient estimate feeds Adam (a per-parameter step proportional to the group's sigma) with an L2 pull towards the
//! starting checkpoint (the analogue of the KL anchor of the PPO plan). Everything is deterministic given the seed and the
//! config, at any thread count, and resumable: `state.bin` holds the parameters and Adam's moments after each generation.
//!
//! The run directory is the one of the BC runs (`config.toml`, `metrics.jsonl`, `status.json`, `state.bin`, `checkpoints/`), and the
//! metrics are lines the «Обучение» tab already reads (`kind = "train"` with `loss.total` = the negated mean return, `kind = "arena"`
//! with the first-freeze rate of an evaluation), plus `kind = "es"` and `kind = "es_eval"` lines with the held-block numbers.

pub mod eval;
pub mod noise;
pub mod space;
pub mod stats;

use std::path::Path;
use std::time::Instant;

use ddai_env::config::{PlayerSpec, Rules};
use ddai_env::run::layout_of;
use ddai_fly::brain::FlyBrainConfig;
use ddai_fly::bundle::{
    BundleMeta, FlyBrainTemplate, FlyBundle, load_bundle, read_zstd_postcard, save_bundle, sha256_hex_of_file,
    write_zstd_postcard,
};
use ddai_fly::rng::SplitMix64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::bank::{Bank, BankStart, StartFilter, play_from_start};
use crate::experiment::{Env, expand_home, load_env};
use crate::heldblock::{RewardConfig, play_episode};
use crate::trainer::RunDir;

use self::eval::{BrainMaker, GamesSummary, StartsSummary, eval_games, eval_starts, summarize_games, summarize_starts};
use self::noise::{AscentAdam, gradient_estimate, members, pair_noise, pair_seed};
use self::space::{ParamSpace, SpaceConfig, flatten, with_params};
use self::stats::centered_ranks;

fn d_one() -> f32 {
    1.0
}
fn d_pairs() -> usize {
    24
}
fn d_lr_rel() -> f32 {
    0.25
}
fn d_l2() -> f32 {
    0.02
}
fn d_post() -> usize {
    16
}
fn d_normal() -> usize {
    4
}
fn d_window() -> i32 {
    crate::heldblock::WINDOW_TICKS
}
fn d_burn() -> i32 {
    crate::bank::DEFAULT_BURN_IN_TICKS
}
fn d_threads() -> usize {
    3
}
fn d_beta1() -> f32 {
    0.9
}
fn d_beta2() -> f32 {
    0.999
}

/// Shown by the web tab as the run's model kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelStub {
    pub kind: String,
}

impl Default for ModelStub {
    fn default() -> Self {
        ModelStub { kind: "fly".into() }
    }
}

/// The periodic evaluation: fixed seeds, so every point of the curve (and the baselines) plays the very same episodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvalConfig {
    /// Every this many generations (`0` = only at the start and the end).
    pub every: u64,
    /// Post-freeze starts per set (validation part of the training halls' bank, and the holdout bank).
    pub starts: usize,
    /// Full games per set (training halls, holdout halls).
    pub games: u32,
    pub holdout_arenas: Vec<String>,
    pub seed_base: u64,
}

impl Default for EvalConfig {
    fn default() -> Self {
        EvalConfig {
            every: 25,
            starts: 400,
            games: 400,
            holdout_arenas: vec!["chillblock5-ruler".into()],
            seed_base: 9_100_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EsConfig {
    pub name: String,
    #[serde(default)]
    pub model: ModelStub,
    pub flyg: String,
    /// The checkpoint to start from (upgraded with the opponent channels: `train upgrade-bundle`).
    pub init_bundle: String,
    pub arenas_dir: String,
    pub map_dir: String,
    pub run_dir: String,
    /// The post-freeze bank (`train es bank`).
    pub bank: String,
    #[serde(default)]
    pub seed: u64,
    #[serde(default = "d_pairs")]
    pub pairs: usize,
    pub generations: u64,
    #[serde(default)]
    pub space: SpaceConfig,
    /// Adam's step as a fraction of each group's sigma.
    #[serde(default = "d_lr_rel")]
    pub lr_rel: f32,
    #[serde(default = "d_beta1")]
    pub beta1: f32,
    #[serde(default = "d_beta2")]
    pub beta2: f32,
    /// Strength of the pull towards the starting parameters.
    #[serde(default = "d_l2")]
    pub l2_anchor: f32,
    #[serde(default = "d_post")]
    pub post_episodes: usize,
    #[serde(default = "d_normal")]
    pub normal_games: usize,
    /// The halls the members train on (training-tagged only).
    pub train_arenas: Vec<String>,
    /// Only post-freeze starts where an idle blocker would not hold the block (the pilot's "escapable"; mostly starts where the idle
    /// blocker falls, see `StartFilter`).
    #[serde(default)]
    pub escapable_only: bool,
    /// Only post-freeze starts where the *victim* escapes under an idle blocker (needs a bank tagged with it: `train es retag`).
    #[serde(default)]
    pub victim_escapable_only: bool,
    #[serde(default = "d_window")]
    pub window_ticks: i32,
    #[serde(default = "d_burn")]
    pub burn_in_ticks: i32,
    #[serde(default)]
    pub reward: RewardConfig,
    /// A member's fitness is `post_weight * mean(post-freeze returns) + game_weight * mean(full-game returns)`: each kind of episode counts
    /// by its weight, not by how many of them a generation plays (the first attempt summed the returns and the twelve post-freeze starts
    /// drowned the two games: the first-freeze rate collapsed, E-022).
    #[serde(default = "d_one")]
    pub post_weight: f32,
    #[serde(default = "d_one")]
    pub game_weight: f32,
    #[serde(default = "d_threads")]
    pub threads: usize,
    /// Stop (cleanly, resumable) after the generation that crosses this wall time; `0` = no limit.
    #[serde(default)]
    pub max_hours: f32,
    #[serde(default)]
    pub eval: EvalConfig,
}

impl EsConfig {
    pub fn parse(text: &str) -> Result<EsConfig, String> {
        let c: EsConfig = toml::from_str(text).map_err(|e| e.to_string())?;
        c.validate()?;
        Ok(c)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.pairs == 0 || self.generations == 0 {
            return Err("pairs and generations must be positive".into());
        }
        if self.post_episodes + self.normal_games == 0 {
            return Err("a member needs at least one episode".into());
        }
        if self.train_arenas.is_empty() {
            return Err("train_arenas is empty".into());
        }
        if self.threads == 0 || self.threads > 3 {
            return Err("threads must be 1..=3 on the shared machine".into());
        }
        Ok(())
    }
}

/// What survives a kill: everything the next generation needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EsState {
    pub generation: u64,
    pub theta: Vec<f32>,
    pub adam: AscentAdam,
    /// The best selection score so far (training halls only) and where it was.
    pub best_score: f64,
    pub best_gen: u64,
}

fn mix(a: u64, b: u64, c: u64) -> u64 {
    pair_seed(a ^ 0x5EED_5EED_5EED_5EED, b, c)
}

/// The episodes every member of generation `generation` plays.
#[derive(Debug, Clone, PartialEq)]
pub struct GenEpisodes {
    /// Indices into the training starts.
    pub post: Vec<usize>,
    /// `(arena index into train_arenas, seed, layout game index)`.
    pub games: Vec<(usize, u64, u32)>,
}

pub fn gen_episodes(cfg: &EsConfig, generation: u64, n_starts: usize) -> GenEpisodes {
    let mut rng = SplitMix64::new(mix(cfg.seed, generation, 0xB0));
    // Distinct starts when the bank is big enough (partial Fisher-Yates), else with repeats.
    let post = if n_starts == 0 {
        Vec::new()
    } else if cfg.post_episodes <= n_starts {
        let mut idx: Vec<usize> = (0..n_starts).collect();
        for i in 0..cfg.post_episodes {
            let j = i + (rng.next_u64() % (n_starts - i) as u64) as usize;
            idx.swap(i, j);
        }
        idx.truncate(cfg.post_episodes);
        idx
    } else {
        (0..cfg.post_episodes)
            .map(|_| (rng.next_u64() % n_starts as u64) as usize)
            .collect()
    };
    let na = cfg.train_arenas.len();
    let games = (0..cfg.normal_games)
        .map(|j| {
            let seed = mix(cfg.seed, generation, 0xC0 + j as u64) % 1_000_000_000_000 + 2_000_000_000;
            (j % na, seed, ((generation as usize * cfg.normal_games + j) / na) as u32)
        })
        .collect();
    GenEpisodes { post, games }
}

/// What one member made of the generation's episodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemberResult {
    pub fitness: f32,
    pub post_held: u32,
    pub post_n: u32,
    pub self_freezes: u32,
    pub games_credited: u32,
    pub games_n: u32,
    pub games_timeouts: u32,
}

struct Ctx<'a> {
    cfg: &'a EsConfig,
    env: &'a Env,
    base: &'a FlyBundle,
    flyg: &'a ddai_flyg::Flyg,
    flyg_sha: &'a str,
    train_starts: Vec<&'a BankStart>,
    pool: &'a rayon::ThreadPool,
    rules: Rules,
}

impl Ctx<'_> {
    fn template(&self, theta: &[f32]) -> Result<FlyBrainTemplate, String> {
        let bundle = with_params(self.base, theta)?;
        FlyBrainTemplate::from_parts(bundle, self.flyg.clone(), self.flyg_sha).map_err(|e| e.to_string())
    }

    fn scripted(&self) -> Result<Box<dyn ddai_brain::Brain>, ddai_env::EnvError> {
        self.env.models.factory()(&PlayerSpec::simple("scripted"))
    }

    /// Plays the generation's episodes with the fly of `theta`.
    fn member(&self, theta: &[f32], eps: &GenEpisodes) -> Result<MemberResult, String> {
        let t = self.template(theta)?;
        let c = self.cfg;
        let mut r = MemberResult {
            fitness: 0.0,
            post_held: 0,
            post_n: 0,
            self_freezes: 0,
            games_credited: 0,
            games_n: 0,
            games_timeouts: 0,
        };
        let (mut post_sum, mut game_sum) = (0.0f32, 0.0f32);
        for &i in &eps.post {
            let s = self.train_starts[i];
            let arena = &self.env.arenas[&s.arena];
            let o = play_from_start(
                arena,
                &self.rules,
                s,
                t.instantiate_played(FlyBrainConfig::default()),
                self.scripted().map_err(|e| e.to_string())?,
                c.window_ticks,
                c.burn_in_ticks,
            )
            .map_err(|e| e.to_string())?;
            post_sum += c.reward.post_freeze(&o);
            r.post_n += 1;
            r.post_held += u32::from(o.held_block);
            r.self_freezes += u32::from(o.focal_out_in_window);
        }
        for &(a, seed, g) in &eps.games {
            let arena = &self.env.arenas[&c.train_arenas[a]];
            let o = play_episode(
                arena,
                &self.rules,
                seed,
                layout_of(arena, g),
                t.instantiate_played(FlyBrainConfig::default()),
                self.scripted().map_err(|e| e.to_string())?,
                c.window_ticks,
            )
            .map_err(|e| e.to_string())?;
            game_sum += c.reward.full_game(&o);
            r.games_n += 1;
            r.games_credited += u32::from(o.result == ddai_env::stats::GameResult::W && o.credited);
            r.games_timeouts += u32::from(o.result == ddai_env::stats::GameResult::T);
        }
        if r.post_n > 0 {
            r.fitness += c.post_weight * post_sum / r.post_n as f32;
        }
        if r.games_n > 0 {
            r.fitness += c.game_weight * game_sum / r.games_n as f32;
        }
        Ok(r)
    }

    /// The fitness of every member (index `2i` / `2i + 1` the two of pair `i`), in member order whatever the thread count.
    fn population(&self, thetas: &[Vec<f32>], eps: &GenEpisodes) -> Result<Vec<MemberResult>, String> {
        let r: Vec<Result<MemberResult, String>> = self
            .pool
            .install(|| thetas.par_iter().map(|t| self.member(t, eps)).collect());
        r.into_iter().collect()
    }
}

/// A point of the learning curve: the held-block numbers and first-freeze rates of one parameter vector on the fixed evaluation episodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalPoint {
    pub generation: u64,
    /// What was evaluated on; two files are paired item by item only when these agree (`None` in the files of the pilot).
    #[serde(default)]
    pub spec: Option<EvalSpec>,
    pub train_starts: StartsSummary,
    pub holdout_starts: Option<StartsSummary>,
    pub train_games: GamesSummary,
    pub holdout_games: Option<GamesSummary>,
}

/// The episodes an [`EvalPoint`] was made on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalSpec {
    pub seed_base: u64,
    pub starts: usize,
    pub games: u32,
    pub train_arenas: Vec<String>,
    pub holdout_arenas: Vec<String>,
    pub window_ticks: i32,
    pub burn_in_ticks: i32,
    /// The bank's size and a hash of its recipes' identities (arena, seed, layout, freeze tick): the same starts in the same order.
    pub bank_starts: usize,
    pub bank_fingerprint: String,
}

impl EvalSpec {
    pub fn new(cfg: &EsConfig, bank: &Bank) -> EvalSpec {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for s in &bank.starts {
            h.update(format!("{}|{}|{}|{}|{};", s.arena, s.seed, s.swap, s.reverse_order, s.end_tick).as_bytes());
        }
        EvalSpec {
            seed_base: cfg.eval.seed_base,
            starts: cfg.eval.starts,
            games: cfg.eval.games,
            train_arenas: cfg.train_arenas.clone(),
            holdout_arenas: cfg.eval.holdout_arenas.clone(),
            window_ticks: cfg.window_ticks,
            burn_in_ticks: cfg.burn_in_ticks,
            bank_starts: bank.starts.len(),
            bank_fingerprint: ddai_env::arena::hex(&h.finalize())[..16].to_string(),
        }
    }
}

impl EvalPoint {
    /// The training-halls selection score: the held share on victim-escapable starts (the pilot's `idle does not hold` set when the bank is untagged) plus the first-freeze rate minus the own-freeze rate in the
    /// window (never a holdout number).
    pub fn selection_score(&self) -> f64 {
        self.train_starts
            .held_victim_escapable
            .unwrap_or(self.train_starts.held_escapable)
            .p
            + self.train_games.credited.p
            - self.train_starts.self_freeze.p
    }
}

/// `n` items evenly spread over `v` (all of it when `v` is shorter), in order: the same subset every time.
pub fn spread<T: Clone>(v: &[T], n: usize) -> Vec<T> {
    if v.len() <= n {
        return v.to_vec();
    }
    (0..n).map(|i| v[i * v.len() / n].clone()).collect()
}

fn evaluate_with(
    env: &Env,
    pool: &rayon::ThreadPool,
    cfg: &EsConfig,
    bank: &Bank,
    maker: &BrainMaker<'_>,
    generation: u64,
) -> Result<EvalPoint, String> {
    let e = &cfg.eval;
    let rules = &bank.rules;
    let train_val: Vec<&BankStart> = spread(&bank.select(&cfg.train_arenas, Some(true), false), e.starts);
    let hold: Vec<&BankStart> = spread(&bank.select(&e.holdout_arenas, None, false), e.starts);
    let run_starts = |s: &[&BankStart]| -> Result<StartsSummary, String> {
        let o = eval_starts(env, pool, s, rules, maker, cfg.window_ticks, cfg.burn_in_ticks)?;
        Ok(summarize_starts(s, &o))
    };
    let run_games = |arenas: &[String], seed: u64| -> Result<GamesSummary, String> {
        let o = eval_games(env, pool, arenas, e.games, seed, rules, maker, cfg.window_ticks)?;
        Ok(summarize_games(&o))
    };
    Ok(EvalPoint {
        generation,
        spec: Some(EvalSpec::new(cfg, bank)),
        train_starts: run_starts(&train_val)?,
        holdout_starts: if hold.is_empty() {
            None
        } else {
            Some(run_starts(&hold)?)
        },
        train_games: run_games(&cfg.train_arenas, e.seed_base)?,
        holdout_games: if e.holdout_arenas.is_empty() || e.games == 0 {
            None
        } else {
            Some(run_games(&e.holdout_arenas, e.seed_base + 500_000)?)
        },
    })
}

fn strip_items(v: &mut Value) {
    match v {
        Value::Object(m) => {
            m.retain(|k, _| !k.ends_with("_items"));
            for x in m.values_mut() {
                strip_items(x);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(strip_items),
        _ => {}
    }
}

fn arena_line(point: &EvalPoint, label: &str, g: &GamesSummary) -> Value {
    let rate = |r: &eval::Rate| json!([r.p, r.lo, r.hi]);
    json!({"kind": "arena", "phase": "es", "step": point.generation, "eval": {
        "arena": label, "opponents": ["scripted"], "games": g.credited.n,
        "w": g.w, "l": g.l, "d": g.d, "t": g.t,
        "credited_win_rate": rate(&g.credited), "credited_w": g.credited.k,
        "held_credited_win_rate": rate(&g.credited_held),
    }})
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Fields that may change between a kill and the resume without changing what the run computes.
fn normalised_for_resume(cfg: &EsConfig) -> EsConfig {
    let mut c = cfg.clone();
    c.threads = 0;
    c.max_hours = 0.0;
    c.run_dir = String::new();
    c.generations = 0;
    c
}

/// `config.toml` holds the config of the current (re)start, every earlier different one is kept as `config-before-<n>.toml`, and
/// resuming a started run under a config that differs in anything that matters is refused unless `allow_change` (the rules of
/// `runner::record_config`; a longer `generations` is allowed: extending a run is not changing it).
pub fn record_es_config(
    run: &RunDir,
    cfg: &EsConfig,
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
        match toml::from_str::<EsConfig>(&old_text) {
            Ok(old) if started && normalised_for_resume(&old) != normalised_for_resume(cfg) => {
                let (a, b) = (
                    toml::Value::try_from(normalised_for_resume(&old)).map_err(|e| e.to_string())?,
                    toml::Value::try_from(normalised_for_resume(cfg)).map_err(|e| e.to_string())?,
                );
                let mut sections: Vec<String> = Vec::new();
                if let (Some(a), Some(b)) = (a.as_table(), b.as_table()) {
                    for k in a.keys().chain(b.keys()) {
                        if a.get(k) != b.get(k) && !sections.contains(k) {
                            sections.push(k.clone());
                        }
                    }
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

fn meta(cfg: &EsConfig, generation: u64) -> BundleMeta {
    let (commit, dirty) = ddai_env::output::git_info(Path::new(env!("CARGO_MANIFEST_DIR")));
    BundleMeta {
        seed: cfg.seed,
        git_commit: Some(if dirty { format!("{commit}+dirty") } else { commit }),
        steps: generation,
        notes: format!("{}: OpenAI-ES generation {generation}", cfg.name),
    }
}

/// Everything the loop needs, loaded once.
pub struct Loaded {
    pub env: Env,
    pub base: FlyBundle,
    pub flyg: ddai_flyg::Flyg,
    pub flyg_sha: String,
    pub bank: Bank,
}

pub fn load(cfg: &EsConfig) -> Result<Loaded, String> {
    let flyg_path = expand_home(&cfg.flyg);
    let env = load_env(
        &expand_home(&cfg.arenas_dir),
        &expand_home(&cfg.map_dir),
        Some(flyg_path.clone()),
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
    Ok(Loaded {
        env,
        base,
        flyg,
        flyg_sha,
        bank,
    })
}

pub fn make_pool(threads: usize) -> Result<rayon::ThreadPool, String> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .stack_size(32 << 20)
        .build()
        .map_err(|e| e.to_string())
}

/// Evaluates the fly of a bundle on the config's evaluation episodes.
pub fn evaluate_bundle(
    cfg: &EsConfig,
    loaded: &Loaded,
    bundle: FlyBundle,
    generation: u64,
) -> Result<EvalPoint, String> {
    let pool = make_pool(cfg.threads)?;
    let t = FlyBrainTemplate::from_parts(bundle, loaded.flyg.clone(), &loaded.flyg_sha).map_err(|e| e.to_string())?;
    let maker = || Ok(t.instantiate_played(FlyBrainConfig::default()));
    evaluate_with(&loaded.env, &pool, cfg, &loaded.bank, &maker, generation)
}

/// The start bundle is pinned by its sha256 (`init.sha256` in the run directory): the base of the search and of the L2 anchor must not
/// change under a resumed run (replacing the file used to be accepted silently, only the parameter count was checked).
fn pin_init_bundle(run: &RunDir, cfg: &EsConfig, log: &mut dyn FnMut(&str)) -> Result<(), String> {
    let path = expand_home(&cfg.init_bundle);
    let sha = sha256_hex_of_file(&path).map_err(|e| e.to_string())?;
    let pin = run.path("init.sha256");
    match std::fs::read_to_string(&pin) {
        Ok(old) if old.trim() == sha => Ok(()),
        Ok(old) => Err(format!(
            "refusing to resume {}: the start bundle {} is not the one the run started with (sha256 {} then, {sha} now)",
            run.path("").display(),
            path.display(),
            old.trim()
        )),
        Err(_) => {
            if run.path("state.bin").exists() {
                log("note: this run predates the start-bundle pin; pinning the current file");
            }
            std::fs::write(&pin, format!("{sha}\n")).map_err(|e| e.to_string())
        }
    }
}

fn make_ctx<'a>(cfg: &'a EsConfig, loaded: &'a Loaded, pool: &'a rayon::ThreadPool) -> Result<Ctx<'a>, String> {
    let filter = if cfg.victim_escapable_only {
        StartFilter::VictimEscapes
    } else if cfg.escapable_only {
        StartFilter::IdleDoesNotHold
    } else {
        StartFilter::All
    };
    let train_starts: Vec<&BankStart> = loaded.bank.select_by(&cfg.train_arenas, Some(false), filter)?;
    if train_starts.is_empty() {
        return Err("the bank has no training start on the training arenas".into());
    }
    Ok(Ctx {
        cfg,
        env: &loaded.env,
        base: &loaded.base,
        flyg: &loaded.flyg,
        flyg_sha: &loaded.flyg_sha,
        train_starts,
        pool,
        rules: loaded.bank.rules.clone(),
    })
}

/// One row of the sigma scan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScanRow {
    pub group: String,
    pub params: usize,
    pub scale: f32,
    pub sigma: f32,
    pub fitness_base: f64,
    /// Over the perturbations: mean and sd of the fitness change against the unperturbed fly, and of the held share and credited rate.
    pub dfitness_mean: f64,
    pub dfitness_sd: f64,
    pub held_mean: f64,
    pub credited_mean: f64,
    pub best_dfitness: f64,
}

/// For every parameter group alone and every scale, `k` random perturbations of the starting checkpoint on the fixed episodes of
/// generation 0: how much the fitness moves, i.e. what noise level the ES can work at (it needs the perturbations to move the
/// return, and not so far that the fly is broken).
pub fn scan_sigmas(
    cfg: &EsConfig,
    loaded: &Loaded,
    scales: &[f32],
    k: usize,
    log: &mut dyn FnMut(&str),
) -> Result<Vec<ScanRow>, String> {
    let pool = make_pool(cfg.threads)?;
    let ctx = make_ctx(cfg, loaded, &pool)?;
    let theta0 = flatten(&loaded.base);
    let eps = gen_episodes(cfg, 0, ctx.train_starts.len());
    let base = ctx.population(std::slice::from_ref(&theta0), &eps)?[0];
    let mut rows = Vec::new();
    for group in space::GROUPS {
        let only = SpaceConfig {
            groups: vec![group.to_string()],
            ..cfg.space.clone()
        };
        let mut space = ParamSpace::new(&loaded.base, &only)?;
        space.exclude(&space::dead_bin_gains(&loaded.base, &loaded.flyg)?);
        if space.searched() == 0 {
            continue;
        }
        for &scale in scales {
            let sigma: Vec<f32> = space.sigma.iter().map(|s| s * scale).collect();
            let thetas: Vec<Vec<f32>> = (0..k)
                .map(|i| members(&theta0, &sigma, &pair_noise(cfg.seed, 0xFFFF, i as u64, space.total)).0)
                .collect();
            let res = ctx.population(&thetas, &eps)?;
            let d: Vec<f64> = res.iter().map(|r| f64::from(r.fitness - base.fitness)).collect();
            let mean = d.iter().sum::<f64>() / d.len() as f64;
            let sd = (d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / d.len() as f64).sqrt();
            let row = ScanRow {
                group: group.to_string(),
                params: space.searched(),
                scale,
                sigma: sigma.iter().cloned().fold(0.0, f32::max),
                fitness_base: f64::from(base.fitness),
                dfitness_mean: mean,
                dfitness_sd: sd,
                held_mean: res
                    .iter()
                    .map(|r| f64::from(r.post_held) / f64::from(r.post_n.max(1)))
                    .sum::<f64>()
                    / res.len() as f64,
                credited_mean: res
                    .iter()
                    .map(|r| f64::from(r.games_credited) / f64::from(r.games_n.max(1)))
                    .sum::<f64>()
                    / res.len() as f64,
                best_dfitness: d.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            };
            log(&format!(
                "{group:8} x{scale:<5} sigma {:.4}: dfitness {mean:+.3} +- {sd:.3} (best {:+.3}), held {:.1}%, credited {:.1}%",
                row.sigma,
                row.best_dfitness,
                100.0 * row.held_mean,
                100.0 * row.credited_mean
            ));
            rows.push(row);
        }
    }
    Ok(rows)
}

/// Runs (or resumes) the ES. Returns the final state.
pub fn run_es(cfg: &EsConfig, allow_config_change: bool, log: &mut dyn FnMut(&str)) -> Result<EsState, String> {
    cfg.validate()?;
    let loaded = load(cfg)?;
    let run = RunDir::create(&expand_home(&cfg.run_dir)).map_err(|e| e.to_string())?;
    record_es_config(&run, cfg, allow_config_change, log)?;
    pin_init_bundle(&run, cfg, log)?;
    let mut space = ParamSpace::new(&loaded.base, &cfg.space)?;
    space.exclude(&space::dead_bin_gains(&loaded.base, &loaded.flyg)?);
    let theta0 = flatten(&loaded.base);
    let pool = make_pool(cfg.threads)?;
    let ctx = make_ctx(cfg, &loaded, &pool)?;
    let state_path = run.path("state.bin");
    let mut state: EsState = if state_path.exists() {
        let s: EsState = read_zstd_postcard(&state_path).map_err(|e| e.to_string())?;
        if s.theta.len() != theta0.len() {
            return Err("state.bin does not match the bundle's parameter count".into());
        }
        log(&format!("resumed at generation {}", s.generation));
        s
    } else {
        EsState {
            generation: 0,
            theta: theta0.clone(),
            adam: AscentAdam::new(theta0.len(), cfg.beta1, cfg.beta2),
            best_score: f64::NEG_INFINITY,
            best_gen: 0,
        }
    };
    log(&format!(
        "ES {}: {} parameters ({} searched), {} members, {} starts + {} games per member, {} training starts in the bank",
        cfg.name,
        space.total,
        space.searched(),
        2 * cfg.pairs,
        cfg.post_episodes,
        cfg.normal_games,
        ctx.train_starts.len()
    ));
    let lr: Vec<f32> = space.sigma.iter().map(|s| s * cfg.lr_rel).collect();
    let t_start = Instant::now();
    let mut evaluated: std::collections::HashSet<u64> = std::fs::read_to_string(run.path("metrics.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v["kind"] == "es_eval")
        .filter_map(|v| v["generation"].as_u64())
        .collect();

    let mut do_eval = |state: &mut EsState, generation: u64, log: &mut dyn FnMut(&str)| -> Result<(), String> {
        if !evaluated.insert(generation) {
            return Ok(());
        }
        let t0 = Instant::now();
        let bundle = with_params(&loaded.base, &state.theta)?;
        let t = FlyBrainTemplate::from_parts(bundle.clone(), loaded.flyg.clone(), &loaded.flyg_sha)
            .map_err(|e| e.to_string())?;
        let maker = || Ok(t.instantiate_played(FlyBrainConfig::default()));
        let point = evaluate_with(&loaded.env, &pool, cfg, &loaded.bank, &maker, generation)?;
        let score = point.selection_score();
        let mut line = serde_json::to_value(&point).map_err(|e| e.to_string())?;
        strip_items(&mut line);
        line["kind"] = json!("es_eval");
        line["generation"] = json!(generation);
        line["score"] = json!(score);
        line["eval_s"] = json!(t0.elapsed().as_secs_f64());
        // What a kill must not lose goes to disk first: the checkpoints, then `state.bin` with the new best score, and only then the
        // metrics lines (a resume skips an evaluation whose line is there; with the line missing it plays the same evaluation again).
        save_bundle(&run.checkpoint(&format!("step-{generation:05}.bundle")), &bundle).map_err(|e| e.to_string())?;
        let new_best = score > state.best_score;
        if new_best {
            state.best_score = score;
            state.best_gen = generation;
            save_bundle(&run.checkpoint("selected.bundle"), &bundle).map_err(|e| e.to_string())?;
        }
        write_zstd_postcard(&state_path, state, 3).map_err(|e| e.to_string())?;
        run.append_metrics(&line).map_err(|e| e.to_string())?;
        run.append_metrics(&arena_line(&point, "train-halls", &point.train_games))
            .map_err(|e| e.to_string())?;
        if let Some(h) = &point.holdout_games {
            run.append_metrics(&arena_line(&point, &cfg.eval.holdout_arenas.join("+"), h))
                .map_err(|e| e.to_string())?;
        }
        if new_best {
            run.append_metrics(
                &json!({"kind": "selection", "phase": format!("es-{generation}"), "arenas": cfg.train_arenas,
                "table": [[format!("es-{generation}"), score]]}),
            )
            .map_err(|e| e.to_string())?;
        }
        log(&format!(
            "eval generation {generation}: held train-val {} | holdout {} | first-freeze train {} | holdout {} ({:.0}s)",
            point.train_starts.held.fmt_pct(),
            point.holdout_starts.as_ref().map_or("-".into(), |s| s.held.fmt_pct()),
            point.train_games.credited.fmt_pct(),
            point
                .holdout_games
                .as_ref()
                .map_or("-".into(), |g| g.credited.fmt_pct()),
            t0.elapsed().as_secs_f64()
        ));
        Ok(())
    };

    while state.generation < cfg.generations {
        let generation = state.generation;
        if generation == 0 || (cfg.eval.every > 0 && generation.is_multiple_of(cfg.eval.every)) {
            do_eval(&mut state, generation, log)?;
        }
        if cfg.max_hours > 0.0 && t_start.elapsed().as_secs_f64() > f64::from(cfg.max_hours) * 3600.0 {
            log(&format!(
                "wall limit of {} h reached at generation {generation}",
                cfg.max_hours
            ));
            break;
        }
        let t0 = Instant::now();
        let eps_set = gen_episodes(cfg, generation, ctx.train_starts.len());
        let noises = |i: usize| pair_noise(cfg.seed, generation, i as u64, space.total);
        let mut thetas: Vec<Vec<f32>> = Vec::with_capacity(2 * cfg.pairs);
        for i in 0..cfg.pairs {
            let (p, m) = members(&state.theta, &space.sigma, &noises(i));
            thetas.push(p);
            thetas.push(m);
        }
        let results = ctx.population(&thetas, &eps_set)?;
        let fit: Vec<f32> = results.iter().map(|r| r.fitness).collect();
        let u = centered_ranks(&fit);
        let mut g = gradient_estimate(&u, &space.sigma, noises);
        for j in 0..g.len() {
            if space.sigma[j] > 0.0 {
                g[j] -= cfg.l2_anchor * (state.theta[j] - theta0[j]) / space.sigma[j];
            }
        }
        let grad_norm = g.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
        let step_norm = state.adam.apply(&mut state.theta, &g, &lr);
        state.generation += 1;
        write_zstd_postcard(&state_path, &state, 3).map_err(|e| e.to_string())?;
        save_bundle(
            &run.checkpoint("last.bundle"),
            &with_params(&loaded.base, &state.theta).map(|mut b| {
                b.meta = meta(cfg, state.generation);
                b
            })?,
        )
        .map_err(|e| e.to_string())?;

        let n = results.len() as f64;
        let mean = fit.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let sd = (fit.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / n).sqrt();
        let sum_u = |f: &dyn Fn(&MemberResult) -> u32| results.iter().map(|r| u64::from(f(r))).sum::<u64>() as f64;
        let post_n = sum_u(&|r| r.post_n).max(1.0);
        let games_n = sum_u(&|r| r.games_n).max(1.0);
        let dist = state
            .theta
            .iter()
            .zip(&theta0)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let elapsed = t_start.elapsed().as_secs_f64();
        run.append_metrics(&json!({"kind": "train", "phase": "es", "step": state.generation,
            "loss": {"total": -mean}, "grad_norm": grad_norm}))
            .map_err(|e| e.to_string())?;
        run.append_metrics(&json!({"kind": "es", "generation": state.generation, "fitness_mean": mean, "fitness_sd": sd,
            "fitness_min": fit.iter().cloned().fold(f32::INFINITY, f32::min),
            "fitness_max": fit.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            "post_held": sum_u(&|r| r.post_held) / post_n, "self_freezes": sum_u(&|r| r.self_freezes) / post_n,
            "games_credited": sum_u(&|r| r.games_credited) / games_n, "games_timeouts": sum_u(&|r| r.games_timeouts) / games_n,
            "grad_norm": grad_norm, "step_norm": step_norm, "theta_dist": dist, "gen_s": t0.elapsed().as_secs_f64()}))
            .map_err(|e| e.to_string())?;
        run.write_status(&json!({"phase": "es", "step": state.generation, "phase_step": state.generation, "phase_steps": cfg.generations,
            "loss": -mean, "elapsed_s": elapsed, "unix_s": unix_seconds()}))
            .map_err(|e| e.to_string())?;
        log(&format!(
            "generation {}: fitness {mean:.3} +- {sd:.3}, held {:.1}%, credited {:.1}%, |step| {step_norm:.4}, |theta-theta0| {dist:.3} ({:.0}s)",
            state.generation,
            100.0 * sum_u(&|r| r.post_held) / post_n,
            100.0 * sum_u(&|r| r.games_credited) / games_n,
            t0.elapsed().as_secs_f64()
        ));
    }
    let g = state.generation;
    do_eval(&mut state, g, log)?;
    write_zstd_postcard(&state_path, &state, 3).map_err(|e| e.to_string())?;
    let final_bundle = with_params(&loaded.base, &state.theta).map(|mut b| {
        b.meta = meta(cfg, state.generation);
        b
    })?;
    save_bundle(&run.checkpoint("final.bundle"), &final_bundle).map_err(|e| e.to_string())?;
    // A run stopped by its wall limit shows as finished at the generation it reached, not as "66/400".
    let planned = state.generation.min(cfg.generations);
    run.write_status(
        &json!({"phase": "done", "step": state.generation, "phase_step": state.generation, "phase_steps": planned,
        "elapsed_s": t_start.elapsed().as_secs_f64(), "unix_s": unix_seconds()}),
    )
    .map_err(|e| e.to_string())?;
    Ok(state)
}

/// Evaluates any brain (`idle`, `scripted`, `planner`, `fly:<bundle>`, ...) in the focal seat on the config's evaluation episodes: the
/// baselines, and the final comparison of two bundles on fresh seeds.
pub fn evaluate_player(cfg: &EsConfig, loaded: &Loaded, arg: &str, generation: u64) -> Result<EvalPoint, String> {
    let pool = make_pool(cfg.threads)?;
    let spec = ddai_env::models::player_from_arg(arg);
    let factory = loaded.env.models.factory();
    let maker = || factory(&spec);
    evaluate_with(&loaded.env, &pool, cfg, &loaded.bank, &maker, generation)
}
