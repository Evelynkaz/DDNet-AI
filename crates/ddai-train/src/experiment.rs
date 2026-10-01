//! Orchestration shared by the CLI and the tests: the environment (arenas, model loader), the
//! collection of teacher data, and small helpers for building corpora and evaluation sets.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddai_env::arena::{Arena, load_arena_defs};
use ddai_env::config::{PlayerSpec, Rules};
use ddai_env::models::{ModelBrains, player_from_arg};
use serde::{Deserialize, Serialize};

use crate::collect::{CollectJob, Mixing, collect};
use crate::store::{ArenaRef, StoreError, TeacherStore};
use crate::types::Outcome;

/// `~/` expansion for paths in configs.
pub fn expand_home(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(h)) => PathBuf::from(h).join(rest),
        _ => PathBuf::from(p),
    }
}

/// Everything a run needs from the outside world: the arenas and the model brain loader.
pub struct Env {
    pub arenas: BTreeMap<String, Arena>,
    pub models: Arc<ModelBrains>,
}

/// Builds every arena definition in `arenas_dir` whose map is available (an arena whose map file
/// is missing is skipped with a note, not an error).
pub fn load_env(arenas_dir: &Path, map_dir: &Path, flyg: Option<PathBuf>) -> Result<Env, String> {
    let defs = load_arena_defs(arenas_dir).map_err(|e| e.to_string())?;
    let mut arenas = BTreeMap::new();
    for (name, def) in defs {
        match Arena::build(&def, map_dir) {
            Ok(a) => {
                arenas.insert(name, a);
            }
            Err(e) => eprintln!("note: arena {name} unavailable: {e}"),
        }
    }
    Ok(Env {
        arenas,
        models: Arc::new(ModelBrains::new(flyg)),
    })
}

impl Env {
    pub fn arena_ref(&self, name: &str) -> Option<ArenaRef> {
        self.arenas.get(name).map(|a| ArenaRef {
            name: a.name.clone(),
            map_sha256: a.map_sha256.clone(),
            split: a.tag.label().to_string(),
        })
    }

    /// Names of the arenas tagged `holdout`.
    pub fn holdout_names(&self) -> std::collections::HashSet<String> {
        self.arenas
            .values()
            .filter(|a| a.tag.label() == "holdout")
            .map(|a| a.name.clone())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobConfig {
    pub arena: String,
    pub games: u32,
    /// Opponent brains, slot 1 onwards (`scripted`, `planner`, ...).
    #[serde(default = "default_opponents")]
    pub opponents: Vec<String>,
    #[serde(default)]
    pub beta: f32,
    #[serde(default)]
    pub noise_prob: f32,
    #[serde(default = "default_noise_len")]
    pub noise_len: (u32, u32),
    pub base_seed: u64,
}

fn default_opponents() -> Vec<String> {
    vec!["scripted".to_string()]
}
fn default_noise_len() -> (u32, u32) {
    (2, 6)
}

/// A teacher-data collection run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectConfig {
    pub name: String,
    pub out: String,
    pub arenas_dir: String,
    pub map_dir: String,
    #[serde(default = "default_threads")]
    pub threads: usize,
    #[serde(default)]
    pub round: u32,
    /// Who plays the labelled slot: `teacher` or a model brain (`fly:<bundle>`, `mlp:..`, `gru:..`).
    #[serde(default = "default_actor")]
    pub actor: String,
    #[serde(default, rename = "job")]
    pub jobs: Vec<JobConfig>,
}

fn default_threads() -> usize {
    6
}
fn default_actor() -> String {
    "teacher".to_string()
}

/// Per-job outcome counts of a collection.
#[derive(Debug, Clone, Default, Serialize)]
pub struct JobSummary {
    pub arena: String,
    pub opponents: usize,
    pub games: u32,
    pub steps: u64,
    pub w: u32,
    pub l: u32,
    pub d: u32,
    pub t: u32,
}

fn spec_for(arg: &str) -> PlayerSpec {
    if arg == "teacher" {
        PlayerSpec::simple("planner")
    } else {
        player_from_arg(arg)
    }
}

/// The record of a job's setup (arena, opponents, mixing), as stored with its chunks.
pub fn job_setup(job: &JobConfig) -> String {
    format!(
        "{} vs {} beta={} noise={}",
        job.arena,
        job.opponents.join("+"),
        job.beta,
        job.noise_prob
    )
}

/// The identity of a job within a round: its setup, base seed and game count. A job whose key is
/// already in the dataset is not collected again (resuming an interrupted round).
pub fn job_key(job: &JobConfig) -> String {
    format!("{}|seed={}|games={}", job_setup(job), job.base_seed, job.games)
}

/// Plays and labels every job that is not yet in the dataset, appends the episodes to the dataset
/// at `store` (round `round`, one atomic manifest update per job) and returns the summaries of the
/// jobs it ran. A job already collected in this round (same [`job_key`]) is skipped: a collection
/// killed between jobs is finished by running it again.
pub fn run_collect_jobs(
    env: &Env,
    store: &mut TeacherStore,
    jobs: &[JobConfig],
    actor: &str,
    round: u32,
    threads: usize,
    log: &mut dyn FnMut(&str),
) -> Result<Vec<JobSummary>, String> {
    let factory = env.models.factory();
    let rules = Rules::default();
    let mut summaries = Vec::new();
    for job in jobs {
        let key = job_key(job);
        if store.manifest.job_done(round, &key) {
            log(&format!(
                "round {round}: job {key:?} is already in the dataset, skipped"
            ));
            continue;
        }
        let arena = env
            .arenas
            .get(&job.arena)
            .ok_or_else(|| format!("unknown arena {:?}", job.arena))?;
        let arena_ref = env.arena_ref(&job.arena).expect("arena exists");
        let index = store.manifest.arena_index(arena_ref);
        let cj = CollectJob {
            arena: job.arena.clone(),
            games: job.games,
            opponents: job.opponents.iter().map(|o| spec_for(o)).collect(),
            actor: (actor != "teacher").then(|| spec_for(actor)),
            mixing: Mixing {
                beta: job.beta,
                noise_prob: job.noise_prob,
                noise_len: job.noise_len,
            },
            base_seed: job.base_seed,
        };
        let episodes = collect(arena, index, &rules, &cj, &factory, threads).map_err(|e| e.to_string())?;
        let mut s = JobSummary {
            arena: job.arena.clone(),
            opponents: job.opponents.len(),
            games: job.games,
            ..JobSummary::default()
        };
        for ep in &episodes {
            s.steps += ep.steps.len() as u64;
            match ep.outcome {
                Outcome::Win => s.w += 1,
                Outcome::Loss => s.l += 1,
                Outcome::Draw => s.d += 1,
                Outcome::Timeout => s.t += 1,
            }
        }
        store
            .append(round, actor, &job_setup(job), &key, episodes)
            .map_err(|e: StoreError| e.to_string())?;
        log(&format!(
            "collected {} games on {} ({} steps) W:L:D:T {}:{}:{}:{}",
            s.games, s.arena, s.steps, s.w, s.l, s.d, s.t
        ));
        summaries.push(s);
    }
    Ok(summaries)
}

/// Runs a [`CollectConfig`] end to end (creating or extending the dataset directory).
pub fn run_collect(
    cfg: &CollectConfig,
    code_commit: &str,
    log: &mut dyn FnMut(&str),
) -> Result<Vec<JobSummary>, String> {
    let env = load_env(&expand_home(&cfg.arenas_dir), &expand_home(&cfg.map_dir), None)?;
    let mut store =
        TeacherStore::open_or_create(&expand_home(&cfg.out), &cfg.name, code_commit).map_err(|e| e.to_string())?;
    run_collect_jobs(&env, &mut store, &cfg.jobs, &cfg.actor, cfg.round, cfg.threads, log)
}
