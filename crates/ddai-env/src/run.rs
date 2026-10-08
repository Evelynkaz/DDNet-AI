//! Batches: many games of one condition in parallel (rayon), and whole runs of many conditions.
//!
//! Reproducibility: game `g` of a condition is a pure function of `(base_seed, g, condition)`;
//! results are collected in index order, so the output is identical at any thread count (timing
//! fields aside, and provided the brains are deterministic -- a wall-clock deadline planner is
//! not).

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;

use crate::EnvError;
use crate::arena::{Arena, load_arena_defs};
use crate::config::{BrainFactory, Condition, PlayerSpec, Rules, RunConfig, lag_models_of};
use crate::duel::DuelSpec;
use crate::game::{GameReport, Layout, play_game_duel_watched};
use crate::sim::PlayerSetup;

/// Builds the players of one game from the condition's slot specs.
pub fn setups(slots: &[PlayerSpec], factory: &BrainFactory, arena: &Arena) -> Result<Vec<PlayerSetup>, EnvError> {
    slots
        .iter()
        .map(|spec| {
            let mut brain = factory(spec)?;
            if spec.wb {
                let Some(w) = &arena.wb else {
                    return Err(EnvError::new(format!(
                        "player {:?} asks for wb hints but arena {} has no [wayblock]",
                        spec.brain, arena.name
                    )));
                };
                brain = Box::new(crate::brains::WbHintBrain::new(
                    brain,
                    w.def.clone(),
                    w.side,
                    spec.wb_strong,
                ));
            }
            let label = spec.label.clone().unwrap_or_else(|| brain.name().to_string());
            Ok(PlayerSetup {
                brain,
                lag: spec.lag,
                label,
            })
        })
        .collect()
}

/// Plays game number `g` of a condition: seed `base_seed + g` and a 4-way balanced layout (a wayblock arena
/// never swaps sides: the spawn order alternates only) -- sides
/// swapped on odd games (the harness's `swap: g % 2 === 1`), spawn order reversed on games
/// `g % 4 >= 2`. Positions and spawn order (client-id/entity order, strong/weak hook) are thereby
/// crossed evenly in every block of four games.
pub fn play_indexed(
    arena: &Arena,
    rules: &Rules,
    slots: &[PlayerSpec],
    factory: &BrainFactory,
    base_seed: u64,
    g: u32,
) -> Result<GameReport, EnvError> {
    play_indexed_duel(arena, rules, None, slots, factory, base_seed, g)
}

/// [`play_indexed`] with the duel options of the condition (task 3.19); `None` is exactly [`play_indexed`].
pub fn play_indexed_duel(
    arena: &Arena,
    rules: &Rules,
    duel: Option<&DuelSpec>,
    slots: &[PlayerSpec],
    factory: &BrainFactory,
    base_seed: u64,
    g: u32,
) -> Result<GameReport, EnvError> {
    play_game_duel_watched(
        arena,
        rules,
        duel,
        base_seed.wrapping_add(u64::from(g)),
        layout_of(arena, g),
        setups(slots, factory, arena)?,
        lag_models_of(slots),
        &mut |_, _| true,
    )
}

/// The layout of game number `g` of a condition (see [`play_indexed`]).
pub fn layout_of(arena: &Arena, g: u32) -> Layout {
    if arena.wb.is_some() {
        // A wayblock hold has a holder and an intruder: slot 0 stays on the WB spot (a swap would put
        // the focal player on the intruder's tile and an intruder on the spot), so only the spawn
        // order alternates (task 4.2, review F7).
        Layout {
            swap: false,
            reverse_order: g % 2 == 1,
        }
    } else {
        Layout {
            swap: g % 2 == 1,
            reverse_order: (g / 2) % 2 == 1,
        }
    }
}

/// The outcome of one condition.
pub struct ConditionRun {
    pub condition: Condition,
    pub arena: String,
    pub games: Vec<GameReport>,
    /// Wall seconds the batch took (all threads).
    pub wall_s: f64,
}

/// Runs `games` games of `cond` on `threads` worker threads.
pub fn run_condition(
    cfg: &RunConfig,
    cond: &Condition,
    arena: &Arena,
    games: u32,
    factory: &BrainFactory,
    threads: usize,
) -> Result<ConditionRun, EnvError> {
    let rules = &cfg.rules_for(cond);
    let slots = cond.slots();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        // A hybrid decision needs 1.25-1.5 MiB of stack in release and more in debug builds: the 2 MiB default is too tight (3.7b review F1).
        .stack_size(32 << 20)
        .build()
        .map_err(|e| EnvError::new(format!("thread pool: {e}")))?;
    let t0 = Instant::now();
    let results: Vec<Result<GameReport, EnvError>> = pool.install(|| {
        (0..games)
            .into_par_iter()
            .map(|g| play_indexed_duel(arena, rules, cond.duel.as_ref(), &slots, factory, cfg.base_seed, g))
            .collect()
    });
    let games = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok(ConditionRun {
        condition: cond.clone(),
        arena: arena.name.clone(),
        games,
        wall_s: t0.elapsed().as_secs_f64(),
    })
}

/// Loads every arena the config uses (each map is read and hashed once).
pub fn load_arenas(cfg: &RunConfig, arenas_dir: &Path, map_dir: &Path) -> Result<BTreeMap<String, Arena>, EnvError> {
    let defs = load_arena_defs(arenas_dir)?;
    let mut out = BTreeMap::new();
    for c in &cfg.condition {
        if out.contains_key(&c.arena) {
            continue;
        }
        let def = defs.get(&c.arena).ok_or_else(|| {
            EnvError::new(format!(
                "condition {:?}: unknown arena {:?} (known: {})",
                c.name,
                c.arena,
                defs.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })?;
        out.insert(c.arena.clone(), Arena::build(def, map_dir)?);
    }
    Ok(out)
}
