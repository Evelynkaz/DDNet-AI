//! Evaluation of a brain on the held-block task: held share over post-freeze starts, and the first-freeze (credited) rate and held
//! wins over full games, with Wilson intervals and per-item outcomes kept so two brains can be compared **paired** (McNemar).

use ddai_brain::Brain;
use ddai_env::EnvError;
use ddai_env::config::{PlayerSpec, Rules};
use ddai_env::run::layout_of;
use ddai_env::stats::{GameResult, wilson95};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bank::{BankStart, play_from_start};
use crate::experiment::Env;
use crate::heldblock::{EpisodeOutcome, play_episode};

/// Makes a fresh brain (the focal one) for one episode.
pub type BrainMaker<'a> = dyn Fn() -> Result<Box<dyn Brain>, EnvError> + Sync + 'a;

/// A success rate with its 95% Wilson interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rate {
    pub k: u32,
    pub n: u32,
    pub p: f64,
    pub lo: f64,
    pub hi: f64,
}

impl Rate {
    pub fn new(k: u32, n: u32) -> Rate {
        let (lo, hi) = wilson95(f64::from(k), f64::from(n));
        Rate {
            k,
            n,
            p: if n == 0 { 0.0 } else { f64::from(k) / f64::from(n) },
            lo,
            hi,
        }
    }

    pub fn fmt_pct(&self) -> String {
        if self.n == 0 {
            return "n/a (0)".to_string();
        }
        format!(
            "{:.1}% [{:.1}; {:.1}] ({}/{})",
            100.0 * self.p,
            100.0 * self.lo,
            100.0 * self.hi,
            self.k,
            self.n
        )
    }
}

fn scripted(env: &Env) -> Result<Box<dyn Brain>, EnvError> {
    env.models.factory()(&PlayerSpec::simple("scripted"))
}

/// The episodes that start at `starts`, in order (every one is played with a fresh brain from `maker` taking over at the freeze).
pub fn eval_starts(
    env: &Env,
    pool: &rayon::ThreadPool,
    starts: &[&BankStart],
    rules: &Rules,
    maker: &BrainMaker<'_>,
    window: i32,
    burn_in: i32,
) -> Result<Vec<EpisodeOutcome>, String> {
    let r: Vec<Result<EpisodeOutcome, EnvError>> = pool.install(|| {
        starts
            .par_iter()
            .map(|s| {
                let arena = env
                    .arenas
                    .get(&s.arena)
                    .ok_or_else(|| EnvError::new(format!("unknown arena {:?}", s.arena)))?;
                play_from_start(arena, rules, s, maker()?, scripted(env)?, window, burn_in)
            })
            .collect()
    });
    r.into_iter().collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

/// `n` full games against the scripted bot spread over `arenas` (game `j` on arena `j % len`, seed `seed_base + j`, 4-way layouts),
/// played through the window.
#[allow(clippy::too_many_arguments)]
pub fn eval_games(
    env: &Env,
    pool: &rayon::ThreadPool,
    arenas: &[String],
    n: u32,
    seed_base: u64,
    rules: &Rules,
    maker: &BrainMaker<'_>,
    window: i32,
) -> Result<Vec<EpisodeOutcome>, String> {
    if arenas.is_empty() {
        return Ok(Vec::new());
    }
    let r: Vec<Result<EpisodeOutcome, EnvError>> = pool.install(|| {
        (0..n)
            .into_par_iter()
            .map(|j| {
                let name = &arenas[j as usize % arenas.len()];
                let arena = env
                    .arenas
                    .get(name)
                    .ok_or_else(|| EnvError::new(format!("unknown arena {name:?}")))?;
                let g = j / arenas.len() as u32;
                play_episode(
                    arena,
                    rules,
                    seed_base.wrapping_add(u64::from(j)),
                    layout_of(arena, g),
                    maker()?,
                    scripted(env)?,
                    window,
                )
            })
            .collect()
    });
    r.into_iter().collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

/// What a brain did on a set of post-freeze starts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StartsSummary {
    pub held: Rate,
    /// Starts where an idle blocker would not hold (the geometry does not do the work): the share the fly's play decides.
    pub held_escapable: Rate,
    /// The focal player was out (frozen or dead) during the window.
    pub self_freeze: Rate,
    /// Per start, in the order of the starts: held; the focal player out in the window; the start is escapable (an idle blocker
    /// would not hold it). The escapable-only held share, the primary metric, is `held_items` filtered by `escapable_items`.
    pub held_items: Vec<bool>,
    /// Per start: the focal player was out in the window (empty in the very first baseline files of E-022, made before the items existed).
    #[serde(default)]
    pub self_freeze_items: Vec<bool>,
    #[serde(default)]
    pub escapable_items: Vec<bool>,
    /// Held share on the starts where the **victim** escapes under an idle blocker (`victim_escapes_under_idle`), the primary
    /// metric since the review of 8.5a (F1): `held_escapable` also counts the starts where the idle blocker falls first. `None` in
    /// the files of the pilot and for an untagged bank.
    #[serde(default)]
    pub held_victim_escapable: Option<Rate>,
    /// Per start: the victim escapes under an idle blocker (empty when the bank is not tagged).
    #[serde(default)]
    pub victim_escape_items: Vec<bool>,
}

pub fn summarize_starts(starts: &[&BankStart], out: &[EpisodeOutcome]) -> StartsSummary {
    let held_items: Vec<bool> = out.iter().map(|o| o.held_block).collect();
    let tagged = starts.iter().all(|s| s.victim_escapes_under_idle.is_some());
    let k = |f: &dyn Fn(usize) -> bool| (0..out.len()).filter(|&i| f(i)).count() as u32;
    let esc: Vec<usize> = (0..out.len()).filter(|&i| !starts[i].idle_held).collect();
    StartsSummary {
        held: Rate::new(k(&|i| out[i].held_block), out.len() as u32),
        held_escapable: Rate::new(
            esc.iter().filter(|&&i| out[i].held_block).count() as u32,
            esc.len() as u32,
        ),
        self_freeze: Rate::new(k(&|i| out[i].focal_out_in_window), out.len() as u32),
        held_items,
        self_freeze_items: out.iter().map(|o| o.focal_out_in_window).collect(),
        escapable_items: starts.iter().map(|s| !s.idle_held).collect(),
        held_victim_escapable: tagged.then(|| {
            let ve: Vec<usize> = (0..out.len())
                .filter(|&i| starts[i].victim_escapes_under_idle == Some(true))
                .collect();
            Rate::new(
                ve.iter().filter(|&&i| out[i].held_block).count() as u32,
                ve.len() as u32,
            )
        }),
        victim_escape_items: if tagged {
            starts
                .iter()
                .map(|s| s.victim_escapes_under_idle == Some(true))
                .collect()
        } else {
            Vec::new()
        },
    }
}

/// What a brain did over full games against the scripted bot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GamesSummary {
    pub w: u32,
    pub l: u32,
    pub d: u32,
    pub t: u32,
    /// Games won by the focal player's own credited freeze (the first-freeze rate of D-059).
    pub credited: Rate,
    /// Credited wins whose block held for the whole window.
    pub credited_held: Rate,
    pub credited_items: Vec<bool>,
    pub credited_held_items: Vec<bool>,
}

pub fn summarize_games(out: &[EpisodeOutcome]) -> GamesSummary {
    let count = |r: GameResult| out.iter().filter(|o| o.result == r).count() as u32;
    let credited_items: Vec<bool> = out.iter().map(|o| o.result == GameResult::W && o.credited).collect();
    let credited_held_items: Vec<bool> = out
        .iter()
        .map(|o| o.result == GameResult::W && o.credited && o.held_block)
        .collect();
    let n = out.len() as u32;
    GamesSummary {
        w: count(GameResult::W),
        l: count(GameResult::L),
        d: count(GameResult::D),
        t: count(GameResult::T),
        credited: Rate::new(credited_items.iter().filter(|&&b| b).count() as u32, n),
        credited_held: Rate::new(credited_held_items.iter().filter(|&&b| b).count() as u32, n),
        credited_items,
        credited_held_items,
    }
}
