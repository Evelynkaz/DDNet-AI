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
use ddai_fly::bc::HeadMask;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::seq::{AimMask, MapEntry, Seq, SeqStep, Source};
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
    /// Multiplier of the steps of technique-scenario episodes (`scn:T*` arenas): they are short and rare
    /// next to full games, so they can be drawn more often.
    pub scenario_weight: f32,
    /// Multiplier (task 8.5b) of the steps of the **opening** of a post-freeze episode: the decisions in the first `opening_ticks` ticks from the
    /// freeze (`Episode::end_tick`) on, in the rounds listed in `opening_rounds`. What the planner does there (it throws the hook at the frozen
    /// victim in 80% of its first decisions) decides the held block, and those few steps are otherwise 3% of the data. `1` = off.
    pub opening_boost: f32,
    pub opening_ticks: i32,
    pub opening_rounds: Vec<u32>,
    /// Which decisions score the aim head (task 8.6): every hook or fire label (the default, as before), or only throws and shots.
    pub aim_mask: AimMask,
}

impl Default for TeacherDataConfig {
    fn default() -> Self {
        TeacherDataConfig {
            val_mod: 10,
            round_weights: Vec::new(),
            default_round_weight: 1.0,
            noise_weight: 1.0,
            scenario_weight: 1.0,
            opening_boost: 1.0,
            opening_ticks: 12,
            opening_rounds: Vec::new(),
            aim_mask: AimMask::default(),
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

/// Turns one episode into a sequence (see the module docs for the weights).
pub fn episode_to_seq(ep: &Episode, map: &Arc<MapEntry>, round: u32, cfg: &TeacherDataConfig, arena: &str) -> Seq {
    let base = cfg.round_weight(round)
        * if arena.starts_with("scn:") {
            cfg.scenario_weight
        } else {
            1.0
        };
    let steps = ep
        .steps
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let frozen = s.me.flags & (char_flags::FROZEN | char_flags::DEEP_FROZEN | char_flags::LIVE_FROZEN) != 0;
            let opening = cfg.opening_rounds.contains(&round)
                && s.tick >= ep.end_tick
                && s.tick <= ep.end_tick + cfg.opening_ticks;
            let boost = if opening { cfg.opening_boost } else { 1.0 };
            let weight = if frozen {
                0.0
            } else if s.noise() {
                base * cfg.noise_weight * boost
            } else {
                base * boost
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
                    aim: cfg.aim_mask.scores(&s.label, s.me.hook_state),
                    ..HeadMask::ALL
                },
                // The latch: the hook key played at the previous decision (the episode's first has none).
                latch: i > 0 && ep.steps[i - 1].played.hook,
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

/// Loads the given chunks of `store` into train / validation / holdout sequences. `maps` gives
/// the maps (by arena or `scn:` scenario name, as recorded in the store's manifest); `holdout` names the
/// arenas that are evaluation-only.
pub fn load_teacher(
    store: &TeacherStore,
    chunks: &[usize],
    maps: &BTreeMap<String, Arc<MapEntry>>,
    holdout: &HashSet<String>,
    cfg: &TeacherDataConfig,
    threads: usize,
) -> Result<TeacherSplit, StoreError> {
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

    #[test]
    fn the_opening_of_a_post_freeze_episode_is_boosted_only_in_the_listed_rounds() {
        let cfg = TeacherDataConfig {
            opening_boost: 30.0,
            opening_ticks: 2,
            opening_rounds: vec![7],
            ..TeacherDataConfig::default()
        };
        // Steps at ticks 0, 2, 4, 6; the freeze at tick 2: the steps at ticks 2 and 4 are the opening.
        let mut e = ep(9, 9, false);
        e.end_tick = 2;
        let w = |round: u32| -> Vec<f32> {
            episode_to_seq(&e, &room_map(), round, &cfg, "room")
                .steps
                .iter()
                .map(|x| x.weight)
                .collect()
        };
        assert_eq!(w(7), vec![1.0, 30.0, 30.0, 1.0]);
        assert_eq!(w(0), vec![1.0; 4], "a round that is not listed is untouched");
        assert_eq!(
            episode_to_seq(&e, &room_map(), 7, &TeacherDataConfig::default(), "room")
                .steps
                .iter()
                .map(|x| x.weight)
                .collect::<Vec<_>>(),
            vec![1.0; 4],
            "off by default"
        );
    }

    /// The latch of a step is the hook key **played** at the previous step (the student's own command in a DAgger round, not the label),
    /// false for the episode's first; and `aim_mask = throw_or_fire` scores the aim only on throws (the label presses the hook while the
    /// observed own hook state is idle) and shots.
    #[test]
    fn the_latch_is_the_previous_played_hook_and_the_throw_mask_needs_an_idle_hook() {
        use ddai_brain::{HOOK_FLYING, HOOK_IDLE};
        let step = |i: usize, label_hook: bool, played_hook: bool, state: i32, fire: bool| {
            let mut me = CharacterObservation::at_rest(0);
            me.hook_state = state;
            let mut label = Action::neutral();
            label.hook = label_hook;
            label.fire = fire;
            let mut played = Action::neutral();
            played.hook = played_hook;
            TeacherStep {
                tick: 2 * i as i32,
                me: char_rec(&me),
                others: vec![],
                target: -1,
                label: action_rec(&label),
                soft: None,
                played: action_rec(&played),
                flags: 0,
            }
        };
        let episode = Episode {
            arena: 0,
            seed: 5,
            players: 2,
            outcome: Outcome::Loss,
            end_tick: 100,
            steps: vec![
                step(0, true, false, HOOK_IDLE, false),   // the label throws, the student did not
                step(1, true, true, HOOK_IDLE, false),    // a throw (the previous played key was up)
                step(2, true, true, HOOK_FLYING, false),  // a hold: label hooks while the hook is out
                step(3, false, false, HOOK_FLYING, true), // a shot with the hook out
                step(4, false, false, HOOK_IDLE, false),  // nothing
            ],
        };
        let cfg = TeacherDataConfig::default();
        let s = episode_to_seq(&episode, &room_map(), 0, &cfg, "room");
        assert_eq!(
            s.steps.iter().map(|x| x.latch).collect::<Vec<_>>(),
            vec![false, false, true, true, false],
            "the latch is the previous step's PLAYED hook key"
        );
        assert_eq!(
            s.steps.iter().map(|x| x.mask.aim).collect::<Vec<_>>(),
            vec![true, true, true, true, false],
            "default mask: every hook or fire label"
        );
        let throw = TeacherDataConfig {
            aim_mask: AimMask::ThrowOrFire,
            ..TeacherDataConfig::default()
        };
        let s = episode_to_seq(&episode, &room_map(), 0, &throw, "room");
        assert_eq!(
            s.steps.iter().map(|x| x.mask.aim).collect::<Vec<_>>(),
            vec![true, true, false, true, false],
            "throw mask: the held-hook decision no longer scores the aim, a shot with the hook out still does"
        );
    }
}
