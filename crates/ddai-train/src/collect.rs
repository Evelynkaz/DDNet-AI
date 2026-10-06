//! Teacher labelling in the arena (task 8.2, acceptance criteria 2 and 4).
//!
//! [`LabellingBrain`] wraps the labelled player of a game: at every decision the fixed-iteration
//! planner ([`TeacherPlanner`], D-017) labels the *visited state* with its action and the elite
//! set's soft target, while the action that is actually *played* comes from an actor:
//!
//! * no actor: the teacher plays (initial behaviour-cloning data);
//! * an actor (a fly, an MLP, a GRU, a scripted bot): the actor plays, optionally mixed with the
//!   teacher's action with probability `beta` per decision (DAgger's mixture policy);
//! * either way, with probability `noise_prob` per decision a short burst of random actions
//!   replaces the played action (DART-style exploration), so the dataset also covers the
//!   off-nominal states a student drifts into. The label is always the clean teacher action.
//!
//! Everything is deterministic in the game seed: the planner is fixed-iteration, the actors are
//! deterministic, and the mixing/noise draws come from an RNG seeded by the reset context. The same
//! seed therefore yields byte-identical episodes.

use std::sync::{Arc, Mutex};

use ddai_brain::{Action, Brain, IVec2, Observation, ResetContext, WorldView};
use ddai_env::arena::{Arena, BuiltWorld};
use ddai_env::config::{BrainFactory, PlayerSpec, Rules};
use ddai_env::game::{Layout, play_game};
use ddai_env::scenario::{ScenarioDef, run_trial};
use ddai_env::sim::PlayerSetup;
use ddai_env::stats::GameResult;
use ddai_fly::rng::SplitMix64;
use ddai_planner::brains::PlannerPreset;
use ddai_planner::teacher::TeacherPlanner;
use rayon::prelude::*;

use crate::EnvError;
use crate::types::{Episode, Outcome, TeacherStep, action_rec, char_rec, soft_from_elite, step_flags};

/// How the played action is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mixing {
    /// Probability per decision that the teacher's action is played instead of the actor's
    /// (ignored without an actor: the teacher then always plays).
    pub beta: f32,
    /// Probability per decision of starting a burst of random actions.
    pub noise_prob: f32,
    /// Burst length range in decisions (inclusive).
    pub noise_len: (u32, u32),
}

impl Default for Mixing {
    fn default() -> Self {
        Mixing {
            beta: 0.0,
            noise_prob: 0.0,
            noise_len: (2, 6),
        }
    }
}

/// A random action for a noise burst: any direction, jump and hook sometimes on, aim anywhere,
/// never fire.
fn random_action(rng: &mut SplitMix64) -> Action {
    let angle = rng.next_f32_unit() * std::f32::consts::TAU;
    Action {
        direction: (rng.next_u64() % 3) as i32 - 1,
        jump: rng.next_f32_unit() < 0.3,
        hook: rng.next_f32_unit() < 0.3,
        fire: false,
        target: IVec2::new(
            (angle.cos() * 1000.0).round() as i32,
            (-angle.sin() * 1000.0).round() as i32,
        ),
        wanted_weapon: None,
    }
}

pub type StepLog = Arc<Mutex<Vec<TeacherStep>>>;

/// The labelled player. See the module docs.
pub struct LabellingBrain {
    actor: Option<Box<dyn Brain>>,
    teacher: TeacherPlanner,
    mixing: Mixing,
    rng: SplitMix64,
    noise_left: u32,
    log: StepLog,
    name: String,
}

impl LabellingBrain {
    /// Returns the brain and the handle its steps are logged to (cleared at every `reset`).
    pub fn new(actor: Option<Box<dyn Brain>>, mixing: Mixing) -> (Self, StepLog) {
        let log: StepLog = Arc::new(Mutex::new(Vec::new()));
        let name = actor
            .as_ref()
            .map_or_else(|| "teacher".to_string(), |a| a.name().to_string());
        (
            LabellingBrain {
                actor,
                teacher: TeacherPlanner::new(PlannerPreset::Normal),
                mixing,
                rng: SplitMix64::new(0),
                noise_left: 0,
                log: log.clone(),
                name,
            },
            log,
        )
    }
}

impl Brain for LabellingBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        self.teacher.reset(ctx);
        if let Some(a) = &mut self.actor {
            a.reset(ctx);
        }
        self.rng = SplitMix64::new(ctx.seed ^ 0x00DA_66E4_5EED);
        self.noise_left = 0;
        self.log.lock().expect("step log").clear();
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        // Labelling needs the exact world; without one the actor (or a neutral action) plays.
        self.actor.as_mut().map_or_else(Action::neutral, |a| a.decide(obs))
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        let Some(view) = view else {
            return self.decide(obs);
        };
        let label = self.teacher.label(obs, view);
        // The actor is asked at every decision, played or not: a recurrent one must see them all.
        let actor_action = self.actor.as_mut().map(|a| a.decide_in(obs, Some(view)));
        let mut flags = 0u8;
        if label.info.searched {
            flags |= step_flags::SEARCHED;
        }
        let mut played = match actor_action {
            None => {
                flags |= step_flags::TEACHER_ACTED;
                label.action
            }
            Some(a) => {
                if self.rng.next_f32_unit() < self.mixing.beta {
                    flags |= step_flags::TEACHER_ACTED;
                    label.action
                } else {
                    a
                }
            }
        };
        if self.noise_left == 0 && self.mixing.noise_prob > 0.0 && self.rng.next_f32_unit() < self.mixing.noise_prob {
            let (lo, hi) = self.mixing.noise_len;
            let span = hi.saturating_sub(lo) + 1;
            self.noise_left = lo + (self.rng.next_u64() % u64::from(span)) as u32;
        }
        if self.noise_left > 0 {
            self.noise_left -= 1;
            played = random_action(&mut self.rng);
            flags |= step_flags::NOISE;
            flags &= !step_flags::TEACHER_ACTED;
        }
        // The teacher's own last output is its exact `prev`; only a different played action
        // needs to be told to it (round-tripping its own action would round the aim).
        if played != label.action {
            self.teacher.note_executed(&played);
        }
        self.log.lock().expect("step log").push(TeacherStep {
            tick: obs.tick,
            me: char_rec(&obs.self_state),
            others: obs.others.iter().map(char_rec).collect(),
            target: obs.target_id.map_or(-1, |t| t as i16),
            label: action_rec(&label.action),
            soft: label.elite.as_ref().map(soft_from_elite),
            played: action_rec(&played),
            flags,
        });
        played
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// One batch of games to label.
#[derive(Debug, Clone)]
pub struct CollectJob {
    pub arena: String,
    pub games: u32,
    /// Slot 1.. (`[scripted]` is 1v1, `[scripted, scripted]` is 1v2).
    pub opponents: Vec<PlayerSpec>,
    /// Who plays slot 0; `None` = the teacher itself.
    pub actor: Option<PlayerSpec>,
    pub mixing: Mixing,
    /// Game `g` uses seed `base_seed + g`.
    pub base_seed: u64,
}

fn outcome(r: GameResult) -> Outcome {
    match r {
        GameResult::W => Outcome::Win,
        GameResult::L => Outcome::Loss,
        GameResult::D => Outcome::Draw,
        GameResult::T => Outcome::Timeout,
    }
}

/// Plays game `g` of `job` (4-way balanced layout like every arena batch) and returns the labelled
/// player's episode.
pub fn collect_game(
    arena: &Arena,
    arena_index: u16,
    rules: &Rules,
    job: &CollectJob,
    g: u32,
    factory: &BrainFactory,
) -> Result<Episode, EnvError> {
    collect_game_held(arena, arena_index, rules, job, g, factory).map(|(e, _)| e)
}

/// [`collect_game`] plus what the game says about whether its block held (task 3.10): the episode holds the steps of the whole game,
/// the `rules.after_ticks` window after the deciding freeze included (use `Rules::held_block_window` for a 5 s window), and the
/// [`HeldOutcome`] gives the reward facts (`held_return()`). The stored [`Episode`] format is unchanged.
pub fn collect_game_held(
    arena: &Arena,
    arena_index: u16,
    rules: &Rules,
    job: &CollectJob,
    g: u32,
    factory: &BrainFactory,
) -> Result<(Episode, ddai_env::game::HeldOutcome), EnvError> {
    let actor = job.actor.as_ref().map(factory).transpose()?;
    let (labeller, log) = LabellingBrain::new(actor, job.mixing);
    let mut players = vec![PlayerSetup {
        brain: Box::new(labeller),
        lag: job.actor.as_ref().map_or(0, |a| a.lag),
        label: "subject".to_string(),
    }];
    for spec in &job.opponents {
        players.push(PlayerSetup {
            brain: factory(spec)?,
            lag: spec.lag,
            label: spec.label.clone().unwrap_or_else(|| spec.brain.clone()),
        });
    }
    let seed = job.base_seed.wrapping_add(u64::from(g));
    let n = players.len() as u8;
    let report = play_game(
        arena,
        rules,
        seed,
        Layout {
            swap: g % 2 == 1,
            reverse_order: (g / 2) % 2 == 1,
        },
        players,
    )?;
    let steps = std::mem::take(&mut *log.lock().expect("step log"));
    let held = report.held_outcome(rules.after_ticks);
    Ok((
        Episode {
            arena: arena_index,
            seed,
            players: n,
            outcome: outcome(report.result),
            end_tick: report.end_tick,
            steps,
        },
        held,
    ))
}

/// Plays all games of `job` on `threads` workers; the episodes come back in game order, so the
/// result does not depend on the thread count.
pub fn collect(
    arena: &Arena,
    arena_index: u16,
    rules: &Rules,
    job: &CollectJob,
    factory: &BrainFactory,
    threads: usize,
) -> Result<Vec<Episode>, EnvError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| EnvError::new(format!("thread pool: {e}")))?;
    let results: Vec<Result<Episode, EnvError>> = pool.install(|| {
        (0..job.games)
            .into_par_iter()
            .map(|g| collect_game(arena, arena_index, rules, job, g, factory))
            .collect()
    });
    results.into_iter().collect()
}

/// Labels one trial of a technique scenario: the subject (tee 0) is the [`LabellingBrain`], the other tees
/// are the scenario's own scripts. Trial `k` uses the start-state jitter of `(job.base_seed, k)`; the episode
/// is a *win* when the scenario's success predicate holds and a timeout otherwise. Deterministic in
/// `(job.base_seed, trial)` like an arena game.
pub fn collect_scenario_trial(
    def: &ScenarioDef,
    world: &BuiltWorld,
    arena_index: u16,
    job: &CollectJob,
    trial: u32,
    factory: &BrainFactory,
) -> Result<Episode, EnvError> {
    let actor = job.actor.as_ref().map(factory).transpose()?;
    let (labeller, log) = LabellingBrain::new(actor, job.mixing);
    let lag = job.actor.as_ref().map_or(0, |a| a.lag);
    let out = run_trial(def, world, Box::new(labeller), lag, job.base_seed, trial, true)?;
    let steps = std::mem::take(&mut *log.lock().expect("step log"));
    Ok(Episode {
        arena: arena_index,
        seed: job.base_seed.wrapping_add(u64::from(trial)),
        players: def.tee.len() as u8,
        outcome: if out.success { Outcome::Win } else { Outcome::Timeout },
        end_tick: def.horizon,
        steps,
    })
}

/// All `job.games` trials of a scenario on `threads` workers (in trial order, so the result does not depend
/// on the thread count).
pub fn collect_scenario(
    def: &ScenarioDef,
    world: &BuiltWorld,
    arena_index: u16,
    job: &CollectJob,
    factory: &BrainFactory,
    threads: usize,
) -> Result<Vec<Episode>, EnvError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| EnvError::new(format!("thread pool: {e}")))?;
    let results: Vec<Result<Episode, EnvError>> = pool.install(|| {
        (0..job.games)
            .into_par_iter()
            .map(|k| collect_scenario_trial(def, world, arena_index, job, k, factory))
            .collect()
    });
    results.into_iter().collect()
}
