//! Closed-loop hook statistics of DAgger episodes (review F2 of E-005).
//!
//! A hook head can score a good AUROC offline and still fail in play: own hook state is an input, so
//! "hook <=> my hook is already out" explains most labels, and a model that learned only that never
//! *starts* a hook and never *lets go* of one. In play that shows as a hook rate that sits at 0% or
//! at 100% depending on which side of 0.5 the start probability drifted to. This module counts, on
//! the decisions the **student** really played (not the teacher's, not exploration noise, not while
//! frozen), how often the student presses the hook against how often the teacher would have pressed
//! it on the very same states, split by whether the own hook is out (flying or grabbed).
//!
//! * start rate: `P(hook | own hook not out)`;
//! * release rate: `P(no hook | own hook out)`.

use ddai_brain::{HOOK_FLYING, HOOK_GRABBED};
use ddai_dataset::types::char_flags;
use serde::Serialize;

use crate::types::Episode;

/// Counts on the decisions in one own-hook state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct HookCounts {
    pub n: u64,
    /// Decisions where the student pressed the hook.
    pub student_hook: u64,
    /// Decisions where the teacher's label presses the hook (same states).
    pub teacher_hook: u64,
}

/// Accumulates [`HookCounts`] by own-hook state over student-played decisions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct HookPlay {
    pub not_out: HookCounts,
    pub out: HookCounts,
    /// `out` split by what the hook is doing: still flying, grabbed onto a player, grabbed onto terrain. The
    /// teacher's release decision depends on this (it lets go of a hook that has done its job), and the
    /// models' inputs carry only "out" as one scalar (E-008: why release is not learnable from them).
    pub out_flying: HookCounts,
    pub out_player: HookCounts,
    pub out_terrain: HookCounts,
}

/// Rates derived from [`HookPlay`] (`None` where the state never occurred).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct HookPlayReport {
    /// Student decisions counted (unfrozen, no noise, not the teacher's).
    pub steps: u64,
    /// Share of those decisions with the own hook out.
    pub out_share: f64,
    pub start_student: Option<f64>,
    pub start_teacher: Option<f64>,
    pub release_student: Option<f64>,
    pub release_teacher: Option<f64>,
}

fn rate(k: u64, n: u64) -> Option<f64> {
    (n > 0).then(|| k as f64 / n as f64)
}

impl HookPlay {
    pub fn add_episode(&mut self, ep: &Episode) {
        for s in &ep.steps {
            if s.me.flags & char_flags::FROZEN != 0 || s.teacher_acted() || s.noise() {
                continue;
            }
            let out = matches!(i32::from(s.me.hook_state), HOOK_FLYING | HOOK_GRABBED);
            let add = |c: &mut HookCounts| {
                c.n += 1;
                c.student_hook += u64::from(s.played.hook);
                c.teacher_hook += u64::from(s.label.hook);
            };
            add(if out { &mut self.out } else { &mut self.not_out });
            if out {
                if i32::from(s.me.hook_state) == HOOK_FLYING {
                    add(&mut self.out_flying);
                } else if s.me.hooked_player >= 0 {
                    add(&mut self.out_player);
                } else {
                    add(&mut self.out_terrain);
                }
            }
        }
    }

    pub fn merge(&mut self, other: &HookPlay) {
        for (a, b) in [
            (&mut self.not_out, &other.not_out),
            (&mut self.out, &other.out),
            (&mut self.out_flying, &other.out_flying),
            (&mut self.out_player, &other.out_player),
            (&mut self.out_terrain, &other.out_terrain),
        ] {
            a.n += b.n;
            a.student_hook += b.student_hook;
            a.teacher_hook += b.teacher_hook;
        }
    }

    pub fn report(&self) -> HookPlayReport {
        let steps = self.not_out.n + self.out.n;
        HookPlayReport {
            steps,
            out_share: if steps > 0 {
                self.out.n as f64 / steps as f64
            } else {
                0.0
            },
            start_student: rate(self.not_out.student_hook, self.not_out.n),
            start_teacher: rate(self.not_out.teacher_hook, self.not_out.n),
            release_student: rate(self.out.n - self.out.student_hook, self.out.n),
            release_teacher: rate(self.out.n - self.out.teacher_hook, self.out.n),
        }
    }
}

/// Hook statistics of the episodes the student played in `round` (all chunks of that round).
pub fn round_hook_play(store: &crate::store::TeacherStore, round: u32) -> Result<HookPlay, crate::store::StoreError> {
    let mut h = HookPlay::default();
    for ci in store.chunks_of_round(Some(round)) {
        let chunk = store.read_chunk(ci)?;
        for e in &chunk.episodes {
            h.add_episode(e);
        }
    }
    Ok(h)
}

/// Hook statistics of a set of episodes.
pub fn hook_play_of<'a>(episodes: impl IntoIterator<Item = &'a Episode>) -> HookPlay {
    let mut h = HookPlay::default();
    for e in episodes {
        h.add_episode(e);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Outcome, TeacherStep, action_rec, char_rec, step_flags};
    use ddai_brain::{Action, CharacterObservation};

    fn step(hook_state: i32, frozen: bool, flags: u8, label_hook: bool, played_hook: bool) -> TeacherStep {
        let mut me = CharacterObservation::at_rest(0);
        me.hook_state = hook_state;
        me.is_frozen = frozen;
        let act = |hook: bool| {
            action_rec(&Action {
                hook,
                ..Action::neutral()
            })
        };
        TeacherStep {
            tick: 0,
            me: char_rec(&me),
            others: vec![],
            target: -1,
            label: act(label_hook),
            soft: None,
            played: act(played_hook),
            flags,
        }
    }

    fn episode(steps: Vec<TeacherStep>) -> Episode {
        Episode {
            arena: 0,
            seed: 1,
            players: 2,
            outcome: Outcome::Timeout,
            end_tick: 10,
            steps,
        }
    }

    #[test]
    fn start_and_release_rates_split_by_own_hook_state_over_student_decisions_only() {
        let ep = episode(vec![
            // Own hook not out: the student never starts a hook, the teacher does in 2 of 4.
            step(0, false, 0, true, false),
            step(0, false, 0, true, false),
            step(0, false, 0, false, false),
            step(1, false, 0, false, false),
            // Own hook out (flying / grabbed): the student never lets go, the teacher does in 1 of 2.
            step(HOOK_FLYING, false, 0, true, true),
            step(HOOK_GRABBED, false, 0, false, true),
            // Not counted: frozen, teacher acted, noise.
            step(0, true, 0, true, true),
            step(0, false, step_flags::TEACHER_ACTED, true, true),
            step(HOOK_GRABBED, false, step_flags::NOISE, false, true),
        ]);
        let r = hook_play_of([&ep]).report();
        assert_eq!(r.steps, 6);
        assert!((r.out_share - 2.0 / 6.0).abs() < 1e-12);
        assert_eq!((r.start_student, r.start_teacher), (Some(0.0), Some(0.5)));
        assert_eq!((r.release_student, r.release_teacher), (Some(0.0), Some(0.5)));
    }

    #[test]
    fn a_hook_that_is_out_is_split_into_flying_grabbed_on_a_player_and_grabbed_on_terrain() {
        let mut flying = step(HOOK_FLYING, false, 0, true, true);
        flying.me.hooked_player = -1;
        let mut on_player = step(HOOK_GRABBED, false, 0, false, true);
        on_player.me.hooked_player = 1;
        let mut on_wall = step(HOOK_GRABBED, false, 0, false, true);
        on_wall.me.hooked_player = -1;
        let ep = episode(vec![flying, on_player.clone(), on_player, on_wall]);
        let h = hook_play_of([&ep]);
        assert_eq!((h.out.n, h.out_flying.n, h.out_player.n, h.out_terrain.n), (4, 1, 2, 1));
        // The teacher keeps a flying hook out and lets go of grabbed ones; the student never lets go.
        assert_eq!(
            (
                h.out_flying.teacher_hook,
                h.out_player.teacher_hook,
                h.out_terrain.teacher_hook
            ),
            (1, 0, 0)
        );
        assert_eq!(h.out_player.student_hook, 2);
    }

    #[test]
    fn a_state_that_never_occurs_has_no_rate_and_merging_adds_counts() {
        let only_out = episode(vec![step(HOOK_GRABBED, false, 0, true, true)]);
        let h = hook_play_of([&only_out]);
        let r = h.report();
        assert_eq!((r.start_student, r.start_teacher), (None, None));
        assert_eq!((r.release_student, r.release_teacher), (Some(0.0), Some(0.0)));
        let mut m = h;
        m.merge(&h);
        assert_eq!(m.out.n, 2);
        assert_eq!(HookPlay::default().report().steps, 0);
    }
}
