//! Teacher labels on post-freeze starts (task 8.5a, the third arm: imitation of finishing).
//!
//! The same recipes as the ES uses ([`crate::bank::BankStart`]), but the seat after the freeze is a
//! [`LabellingBrain`]: the fixed-iteration planner (the teacher, D-017) labels every state visited from the burn-in on, while the
//! action played is the teacher's (no actor: `beta` is moot) or a student's (DAgger, with `beta` of teacher actions mixed in). The
//! episodes go to a teacher dataset the ordinary BC trainer reads, and the whole window after the freeze is in them (the arena used
//! to stop there, so the dataset had no label after the first freeze at all).

use std::sync::Arc;

use ddai_env::EnvError;
use ddai_env::arena::Arena;
use ddai_env::config::{BrainFactory, PlayerSpec, Rules};
use ddai_env::game::play_game;
use ddai_env::sim::PlayerSetup;
use ddai_env::stats::GameResult;
use rayon::prelude::*;

use crate::bank::{BankStart, ReplayThenBrain};
use crate::collect::{LabellingBrain, Mixing};
use crate::experiment::{Env, JobSummary};
use crate::store::TeacherStore;
use crate::types::{Episode, Outcome};

/// One labelled episode from `start`.
#[allow(clippy::too_many_arguments)]
pub fn collect_start(
    arena: &Arena,
    arena_index: u16,
    rules: &Rules,
    start: &BankStart,
    actor: Option<&PlayerSpec>,
    mixing: Mixing,
    window: i32,
    burn_in: i32,
    factory: &BrainFactory,
) -> Result<Episode, EnvError> {
    let actor = actor.map(factory).transpose()?;
    let (labeller, log) = LabellingBrain::new(actor, mixing);
    let replay = ReplayThenBrain::new(
        Box::new(labeller),
        Arc::new(start.actions.clone()),
        start.end_tick,
        burn_in,
    );
    let rules = Rules {
        after_ticks: window,
        ..rules.clone()
    };
    let players = vec![
        PlayerSetup {
            brain: Box::new(replay),
            lag: 0,
            label: "subject".into(),
        },
        PlayerSetup {
            brain: factory(&PlayerSpec::simple("scripted"))?,
            lag: 0,
            label: "scripted".into(),
        },
    ];
    let report = play_game(arena, &rules, start.seed, start.layout(), players)?;
    if report.result != GameResult::W || report.end_tick != start.end_tick {
        return Err(EnvError::new(format!(
            "bank start {} seed {} did not replay",
            start.arena, start.seed
        )));
    }
    let steps = std::mem::take(&mut *log.lock().map_err(|_| EnvError::new("step log poisoned"))?);
    Ok(Episode {
        arena: arena_index,
        seed: start.seed,
        players: 2,
        outcome: Outcome::Win,
        end_tick: report.end_tick,
        steps,
    })
}

/// Labels every start of `starts` and appends the episodes to `store` as round `round`. `actor` is `teacher` or a model brain
/// argument (`fly:<bundle>`). Episodes come back in the order of `starts` whatever the thread count.
#[allow(clippy::too_many_arguments)]
pub fn collect_starts(
    env: &Env,
    store: &mut TeacherStore,
    starts: &[&BankStart],
    rules: &Rules,
    actor: &str,
    mixing: Mixing,
    window: i32,
    burn_in: i32,
    round: u32,
    threads: usize,
    log: &mut dyn FnMut(&str),
) -> Result<JobSummary, String> {
    let factory = env.models.factory();
    let spec = (actor != "teacher").then(|| ddai_env::models::player_from_arg(actor));
    let mut index = std::collections::BTreeMap::new();
    for s in starts {
        if !index.contains_key(&s.arena) {
            let r = env
                .arena_ref(&s.arena)
                .ok_or_else(|| format!("unknown arena {:?}", s.arena))?;
            index.insert(s.arena.clone(), store.manifest.arena_index(r));
        }
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .stack_size(32 << 20)
        .build()
        .map_err(|e| e.to_string())?;
    let results: Vec<Result<Episode, EnvError>> = pool.install(|| {
        starts
            .par_iter()
            .map(|s| {
                collect_start(
                    &env.arenas[&s.arena],
                    index[&s.arena],
                    rules,
                    s,
                    spec.as_ref(),
                    mixing,
                    window,
                    burn_in,
                    &factory,
                )
            })
            .collect()
    });
    let episodes = results
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let steps: u64 = episodes.iter().map(|e| e.steps.len() as u64).sum();
    let summary = JobSummary {
        arena: "post-freeze".into(),
        opponents: 1,
        games: episodes.len() as u32,
        steps,
        w: episodes.len() as u32,
        ..JobSummary::default()
    };
    let setup = format!("post-freeze starts, actor {actor}, beta={}", mixing.beta);
    let key = format!(
        "{setup}|starts={}|first_seed={}",
        starts.len(),
        starts.first().map_or(0, |s| s.seed)
    );
    store
        .append(round, actor, &setup, &key, episodes)
        .map_err(|e| e.to_string())?;
    log(&format!(
        "labelled {} post-freeze starts, {} decisions ({setup})",
        starts.len(),
        steps
    ));
    Ok(summary)
}
