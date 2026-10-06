//! Orchestration shared by the CLI and the tests: the environment (arenas, model loader), the
//! collection of teacher data, and small helpers for building corpora and evaluation sets.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddai_env::arena::{Arena, BuiltWorld, load_arena_defs};
use ddai_env::config::{PlayerSpec, Rules};
use ddai_env::models::{ModelBrains, player_from_arg};
use ddai_env::scenario::{ScenarioDef, load_world};
use serde::{Deserialize, Serialize};

use crate::collect::{CollectJob, Mixing, collect, collect_scenario};
use crate::seq::MapEntry;
use crate::store::{ArenaRef, StoreError, TeacherStore};
use crate::types::Outcome;

/// `~/` expansion for paths in configs.
pub fn expand_home(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(h)) => PathBuf::from(h).join(rest),
        _ => PathBuf::from(p),
    }
}

/// A technique scenario (`configs/scenarios/T*.toml`) the teacher can label trials of, with its world.
pub struct ScenarioEntry {
    pub def: ScenarioDef,
    pub world: BuiltWorld,
}

/// The name a scenario's episodes carry in a teacher dataset's arena table (`scn:T1`).
pub fn scenario_arena_name(id: &str) -> String {
    format!("scn:{id}")
}

/// Everything a run needs from the outside world: the arenas, the technique scenarios and the model brain
/// loader. `maps` has an entry for every arena and every scenario (by the name episodes are stored under),
/// shared so a map's mirrored copy is built once.
pub struct Env {
    pub arenas: BTreeMap<String, Arena>,
    pub scenarios: BTreeMap<String, ScenarioEntry>,
    pub maps: BTreeMap<String, Arc<MapEntry>>,
    pub models: Arc<ModelBrains>,
}

/// Builds every arena definition in `arenas_dir` whose map is available (an arena whose map file
/// is missing is skipped with a note, not an error).
pub fn load_env(arenas_dir: &Path, map_dir: &Path, flyg: Option<PathBuf>) -> Result<Env, String> {
    load_env_with_scenarios(arenas_dir, map_dir, flyg, None)
}

/// [`load_env`] plus the technique scenarios of `scenarios_dir` (a scenario whose map is unavailable is
/// skipped with a note).
pub fn load_env_with_scenarios(
    arenas_dir: &Path,
    map_dir: &Path,
    flyg: Option<PathBuf>,
    scenarios_dir: Option<&Path>,
) -> Result<Env, String> {
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
    let mut scenarios = BTreeMap::new();
    if let Some(dir) = scenarios_dir {
        for def in ScenarioDef::load_dir(dir).map_err(|e| e.to_string())? {
            match load_world(&def, map_dir) {
                Ok(world) => {
                    scenarios.insert(def.id.clone(), ScenarioEntry { def, world });
                }
                Err(e) => eprintln!("note: scenario {} unavailable: {e}", def.id),
            }
        }
    }
    let mut maps: BTreeMap<String, Arc<MapEntry>> = arenas
        .iter()
        .map(|(n, a)| (n.clone(), MapEntry::new(a.map.clone())))
        .collect();
    for (id, sc) in &scenarios {
        maps.insert(scenario_arena_name(id), MapEntry::new(sc.world.map.clone()));
    }
    Ok(Env {
        arenas,
        scenarios,
        maps,
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

    /// The arena-table entry of a scenario's episodes (always a training source).
    pub fn scenario_ref(&self, id: &str) -> Option<ArenaRef> {
        self.scenarios.get(id).map(|s| ArenaRef {
            name: scenario_arena_name(id),
            map_sha256: s.world.sha256.clone(),
            split: "train".to_string(),
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
    /// The arena to play on; empty for a scenario job.
    #[serde(default)]
    pub arena: String,
    /// `T1`..: label trials of this technique scenario instead of arena games (`games` = trials; the
    /// scenario fixes the map and the other tees, `opponents` is ignored).
    #[serde(default)]
    pub scenario: Option<String>,
    pub games: u32,
    /// Keep only the trials in which the scenario's success predicate held (technique demonstrations: a
    /// teacher that fails a trial is not a good example of the technique). Meant for teacher-played
    /// scenario jobs; a DAgger job labels every state the student visits whatever the outcome.
    #[serde(default)]
    pub only_success: bool,
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
    /// Task 3.10: ticks played on after the deciding freeze (`Rules::after_ticks`); `None` = the default (150). `250` is the held-block
    /// window (`Rules::held_block_window`): the episode then holds the decisions *after* the first freeze too, so a student trained on it
    /// sees what happens to a block; `HeldOutcome::held_return` pays only a strict held win and pays nothing for a freeze that thaws (`collect_game_held`).
    #[serde(default)]
    pub after_ticks: Option<i32>,
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
    /// Technique scenarios for `scenario = "T*"` jobs.
    #[serde(default)]
    pub scenarios_dir: Option<String>,
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
    match &job.scenario {
        Some(id) => format!(
            "{} beta={} noise={}{}",
            scenario_arena_name(id),
            job.beta,
            job.noise_prob,
            if job.only_success { " ok-only" } else { "" }
        ),
        None => format!(
            "{} vs {} beta={} noise={}",
            job.arena,
            job.opponents.join("+"),
            job.beta,
            job.noise_prob
        ),
    }
}

/// The identity of a job within a round: its setup, base seed, game count and noise-burst length. A job whose key is
/// already in the dataset is not collected again (resuming an interrupted round).
pub fn job_key(job: &JobConfig) -> String {
    // The window is part of the key only when set: the keys of datasets collected before 3.10 stay valid.
    let window = job.after_ticks.map_or(String::new(), |t| format!("|after={t}"));
    format!(
        "{}|seed={}|games={}|noise_len={}-{}{window}",
        job_setup(job),
        job.base_seed,
        job.games,
        job.noise_len.0,
        job.noise_len.1
    )
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
    let mut summaries = Vec::new();
    for job in jobs {
        let rules = Rules {
            after_ticks: job.after_ticks.unwrap_or_else(|| Rules::default().after_ticks),
            ..Rules::default()
        };
        let key = job_key(job);
        if store.manifest.job_done(round, &key) {
            log(&format!(
                "round {round}: job {key:?} is already in the dataset, skipped"
            ));
            continue;
        }
        let cj_actor = (actor != "teacher").then(|| spec_for(actor));
        let mixing = Mixing {
            beta: job.beta,
            noise_prob: job.noise_prob,
            noise_len: job.noise_len,
        };
        let (episodes, arena_label, opponents) = if let Some(id) = &job.scenario {
            let sc = env
                .scenarios
                .get(id)
                .ok_or_else(|| format!("unknown scenario {id:?}"))?;
            let index = store
                .manifest
                .arena_index(env.scenario_ref(id).expect("scenario exists"));
            let cj = CollectJob {
                arena: scenario_arena_name(id),
                games: job.games,
                opponents: Vec::new(),
                actor: cj_actor,
                mixing,
                base_seed: job.base_seed,
            };
            let mut eps =
                collect_scenario(&sc.def, &sc.world, index, &cj, &factory, threads).map_err(|e| e.to_string())?;
            if job.only_success {
                eps.retain(|e| e.outcome == Outcome::Win);
            }
            (eps, scenario_arena_name(id), 0)
        } else {
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
                actor: cj_actor,
                mixing,
                base_seed: job.base_seed,
            };
            let eps = collect(arena, index, &rules, &cj, &factory, threads).map_err(|e| e.to_string())?;
            (eps, job.arena.clone(), job.opponents.len())
        };
        let mut s = JobSummary {
            arena: arena_label,
            opponents,
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
    let env = load_env_with_scenarios(
        &expand_home(&cfg.arenas_dir),
        &expand_home(&cfg.map_dir),
        None,
        cfg.scenarios_dir.as_deref().map(expand_home).as_deref(),
    )?;
    let mut store =
        TeacherStore::open_or_create(&expand_home(&cfg.out), &cfg.name, code_commit).map_err(|e| e.to_string())?;
    run_collect_jobs(&env, &mut store, &cfg.jobs, &cfg.actor, cfg.round, cfg.threads, log)
}

/// What a student does with its hooks in closed loop on one arena, from games it plays alone (no teacher mixed
/// in, no noise) with the teacher labelling every state it visits.
#[derive(Debug, Clone, Serialize)]
pub struct HookPlayEval {
    pub arena: String,
    pub opponents: usize,
    pub games: u32,
    pub steps: u64,
    pub w: u32,
    pub l: u32,
    pub d: u32,
    pub t: u32,
    pub counts: crate::play_stats::HookPlay,
    pub report: crate::play_stats::HookPlayReport,
}

/// How much to play in [`hook_play_eval`].
#[derive(Debug, Clone, Copy)]
pub struct HookPlayPlan {
    pub games: u32,
    pub base_seed: u64,
    pub threads: usize,
}

/// Plays `games` games of the model brain `actor` (`fly:<bundle>`, `mlp:..`) on each arena with the labelling
/// teacher watching, and reports its start / release rates of the hook against the teacher's on the same
/// states ([`crate::play_stats`]). Nothing is written to a dataset.
pub fn hook_play_eval(
    env: &Env,
    actor: &str,
    arenas: &[String],
    opponents: &[String],
    plan: HookPlayPlan,
    log: &mut dyn FnMut(&str),
) -> Result<Vec<HookPlayEval>, String> {
    let HookPlayPlan {
        games,
        base_seed,
        threads,
    } = plan;
    let factory = env.models.factory();
    let rules = Rules::default();
    let mut out = Vec::new();
    for (i, name) in arenas.iter().enumerate() {
        let arena = env.arenas.get(name).ok_or_else(|| format!("unknown arena {name:?}"))?;
        let job = CollectJob {
            arena: name.clone(),
            games,
            opponents: opponents.iter().map(|o| spec_for(o)).collect(),
            actor: Some(spec_for(actor)),
            mixing: Mixing::default(),
            base_seed: base_seed + 1_000_000 * i as u64,
        };
        let episodes = collect(arena, 0, &rules, &job, &factory, threads).map_err(|e| e.to_string())?;
        let mut ev = HookPlayEval {
            arena: name.clone(),
            opponents: opponents.len(),
            games,
            steps: 0,
            w: 0,
            l: 0,
            d: 0,
            t: 0,
            counts: crate::play_stats::HookPlay::default(),
            report: crate::play_stats::HookPlay::default().report(),
        };
        for ep in &episodes {
            ev.steps += ep.steps.len() as u64;
            match ep.outcome {
                Outcome::Win => ev.w += 1,
                Outcome::Loss => ev.l += 1,
                Outcome::Draw => ev.d += 1,
                Outcome::Timeout => ev.t += 1,
            }
            ev.counts.add_episode(ep);
        }
        ev.report = ev.counts.report();
        log(&format!(
            "hook play {name}: {} games, start {:?} / release {:?} (teacher {:?} / {:?})",
            games,
            ev.report.start_student.map(|x| (x * 1000.0).round() / 10.0),
            ev.report.release_student.map(|x| (x * 1000.0).round() / 10.0),
            ev.report.start_teacher.map(|x| (x * 1000.0).round() / 10.0),
            ev.report.release_teacher.map(|x| (x * 1000.0).round() / 10.0),
        ));
        out.push(ev);
    }
    Ok(out)
}
