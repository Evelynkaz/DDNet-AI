//! Teacher episodes as training sequences.
//!
//! Weights: a step where the labelled player is frozen (its input is ignored by the server) is
//! context only; every other step counts, times its round's weight (later DAgger rounds can be
//! emphasised) and, for steps played under exploration noise, `noise_weight`. The aim head is
//! scored only where the teacher hooks or fires. Episodes of the holdout arenas never enter
//! training; of the others, those whose seed is divisible by `val_mod` form the validation set
//! (unseen games on the training arenas).

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use ddai_dataset::types::char_flags;
use ddai_env::arena::Arena;
use ddai_fly::bc::HeadMask;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::seq::{MapEntry, Seq, SeqStep, Source};
use crate::store::{StoreError, TeacherStore};
use crate::types::Episode;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TeacherDataConfig {
    /// Episodes with `seed % val_mod == 0` are validation (`0` = none).
    pub val_mod: u64,
    /// Weight of steps of each round; rounds not listed use `default_round_weight`.
    pub round_weights: Vec<(u32, f32)>,
    pub default_round_weight: f32,
    /// Weight of steps played under exploration noise (their labels are still the clean teacher's).
    pub noise_weight: f32,
}

impl Default for TeacherDataConfig {
    fn default() -> Self {
        TeacherDataConfig {
            val_mod: 10,
            round_weights: Vec::new(),
            default_round_weight: 1.0,
            noise_weight: 1.0,
        }
    }
}

impl TeacherDataConfig {
    pub fn round_weight(&self, round: u32) -> f32 {
        self.round_weights
            .iter()
            .find(|(r, _)| *r == round)
            .map_or(self.default_round_weight, |(_, w)| *w)
    }
}

pub struct TeacherSplit {
    pub train: Vec<Seq>,
    pub val: Vec<Seq>,
    pub holdout: Vec<Seq>,
}

/// One arena's map entry, by arena name.
pub fn map_entries(arenas: &BTreeMap<String, Arena>) -> BTreeMap<String, Arc<MapEntry>> {
    arenas
        .iter()
        .map(|(name, a)| (name.clone(), MapEntry::new(a.map.clone())))
        .collect()
}

/// Turns one episode into a sequence (see the module docs for the weights).
pub fn episode_to_seq(ep: &Episode, map: &Arc<MapEntry>, round: u32, cfg: &TeacherDataConfig, arena: &str) -> Seq {
    let base = cfg.round_weight(round);
    let steps = ep
        .steps
        .iter()
        .map(|s| {
            let frozen = s.me.flags & (char_flags::FROZEN | char_flags::DEEP_FROZEN | char_flags::LIVE_FROZEN) != 0;
            let weight = if frozen {
                0.0
            } else if s.noise() {
                base * cfg.noise_weight
            } else {
                base
            };
            SeqStep {
                tick: s.tick,
                me: s.me,
                others: s.others.clone(),
                target: s.target,
                label: s.label,
                soft: s.soft,
                weight,
                mask: HeadMask {
                    aim: s.label.hook || s.label.fire,
                    ..HeadMask::ALL
                },
            }
        })
        .collect();
    Seq {
        map: map.clone(),
        steps,
        source: Source::Teacher {
            arena: arena.to_string(),
            round,
        },
    }
}

/// Loads the given chunks of `store` into train / validation / holdout sequences. `arenas` gives
/// the maps (by arena name, as recorded in the store's manifest); `holdout` names the arenas that
/// are evaluation-only.
pub fn load_teacher(
    store: &TeacherStore,
    chunks: &[usize],
    arenas: &BTreeMap<String, Arena>,
    holdout: &HashSet<String>,
    cfg: &TeacherDataConfig,
    threads: usize,
) -> Result<TeacherSplit, StoreError> {
    let maps = map_entries(arenas);
    let manifest = &store.manifest;
    for a in &manifest.arenas {
        if !maps.contains_key(&a.name) {
            return Err(StoreError(format!(
                "dataset uses arena {:?}, which is not available",
                a.name
            )));
        }
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| StoreError(format!("thread pool: {e}")))?;
    type Loaded = Result<Vec<(u64, bool, Seq)>, StoreError>;
    let parts: Vec<Loaded> = pool.install(|| {
        chunks
            .par_iter()
            .map(|&ci| {
                let chunk = store.read_chunk(ci)?;
                let round = manifest.chunks[ci].round;
                Ok(chunk
                    .episodes
                    .iter()
                    .map(|ep| {
                        let arena = &manifest.arenas[ep.arena as usize];
                        let map = &maps[&arena.name];
                        (
                            ep.seed,
                            holdout.contains(&arena.name),
                            episode_to_seq(ep, map, round, cfg, &arena.name),
                        )
                    })
                    .collect())
            })
            .collect()
    });
    let mut out = TeacherSplit {
        train: Vec::new(),
        val: Vec::new(),
        holdout: Vec::new(),
    };
    for p in parts {
        for (seed, is_holdout, seq) in p? {
            if is_holdout {
                out.holdout.push(seq);
            } else if cfg.val_mod > 0 && seed.is_multiple_of(cfg.val_mod) {
                out.val.push(seq);
            } else {
                out.train.push(seq);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seq::testutil::room_map;
    use crate::types::{Outcome, TeacherStep, action_rec, char_rec, step_flags};
    use ddai_brain::{Action, CharacterObservation};

    fn ep(frozen_at: usize, noise_at: usize, hook: bool) -> Episode {
        let step = |i: usize| {
            let mut me = CharacterObservation::at_rest(0);
            me.is_frozen = i == frozen_at;
            let mut a = Action::neutral();
            a.hook = hook;
            TeacherStep {
                tick: 2 * i as i32,
                me: char_rec(&me),
                others: vec![],
                target: -1,
                label: action_rec(&a),
                soft: None,
                played: action_rec(&a),
                flags: if i == noise_at {
                    step_flags::NOISE
                } else {
                    step_flags::TEACHER_ACTED
                },
            }
        };
        Episode {
            arena: 0,
            seed: 5,
            players: 2,
            outcome: Outcome::Loss,
            end_tick: 10,
            steps: (0..4).map(step).collect(),
        }
    }

    #[test]
    fn frozen_steps_are_context_noise_steps_are_reweighted_and_aim_needs_a_hook() {
        let cfg = TeacherDataConfig {
            noise_weight: 0.5,
            round_weights: vec![(2, 3.0)],
            ..TeacherDataConfig::default()
        };
        let s = episode_to_seq(&ep(1, 2, true), &room_map(), 0, &cfg, "room");
        let w: Vec<f32> = s.steps.iter().map(|x| x.weight).collect();
        assert_eq!(w, vec![1.0, 0.0, 0.5, 1.0]);
        assert!(s.steps.iter().all(|x| x.mask.aim), "a hook label makes the aim matter");
        let s = episode_to_seq(&ep(9, 9, false), &room_map(), 2, &cfg, "room");
        assert!(s.steps.iter().all(|x| x.weight == 3.0 && !x.mask.aim));
        assert!(matches!(s.source, Source::Teacher { round: 2, .. }));
    }
}
