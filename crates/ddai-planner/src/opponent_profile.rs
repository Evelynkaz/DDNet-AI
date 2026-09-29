//! `OpponentProfile` (`src/bot/opponentProfile.ts`) — EMA-decayed opponent-behavior read, used by
//! `Planner::seed_plans`'s book only when `opponentReadWeight > 0` (`docs/research/orig-plan.md`
//! §1.4/§1.15). Every preset this task's acceptance criteria names (normal/low-cpu/strong/WB/bold)
//! leaves `opponentReadWeight` at its default `0`, so `Planner::decide_once` never calls
//! `OpponentProfile::observe` and `mix(value, prior) == prior` always — the book's profile-driven
//! branches (`docs/research/orig-plan.md` §1.4) are therefore dead for every configuration this
//! crate's parity tests exercise, exactly as they are for TS. Ported in full anyway (it's small
//! and self-contained, `bot/opponentProfile.ts:1-165`) rather than stubbed, so a future task that
//! *does* raise `opponentReadWeight` inherits a real implementation instead of a placeholder.

use crate::types::{HOOK_FLYING, HOOK_GRABBED, TeeState};
use crate::vmath::vdistance;
use ddai_jsmath as js;

const DECISIONS_PER_SEC: f64 = 25.0;
const FAST_HALFLIFE: f64 = 6.0 * DECISIONS_PER_SEC;
const SLOW_HALFLIFE: f64 = 30.0 * DECISIONS_PER_SEC;

fn decay(half_life: f64) -> f64 {
    js::pow(0.5, 1.0 / half_life)
}

#[derive(Debug, Clone, Copy)]
struct Rate {
    hits: f64,
    total: f64,
    k: f64,
    min_samples: f64,
}

impl Rate {
    fn new(half_life: f64, min_samples: f64) -> Self {
        Rate {
            hits: 0.0,
            total: 0.0,
            k: decay(half_life),
            min_samples,
        }
    }

    fn observe(&mut self, hit: bool) {
        self.hits = self.hits * self.k + if hit { 1.0 } else { 0.0 };
        self.total = self.total * self.k + 1.0;
    }

    fn idle(&mut self) {
        self.hits *= self.k;
        self.total *= self.k;
    }

    fn value(&self, prior: f64) -> f64 {
        if self.total < self.min_samples {
            prior
        } else {
            self.hits / self.total
        }
    }

    fn confidence(&self) -> f64 {
        js::min(1.0, self.total / (self.min_samples * 2.0))
    }

    fn reset(&mut self) {
        self.hits = 0.0;
        self.total = 0.0;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OpponentRead {
    pub aggression: f64,
    pub hook_opens_first: f64,
    pub hook_success: f64,
    pub out_of_jumps: f64,
    pub confidence: f64,
}

const DEFAULT_AGGRESSION: f64 = 0.5;
const DEFAULT_HOOK_OPENS_FIRST: f64 = 0.5;
const DEFAULT_HOOK_SUCCESS: f64 = 0.57;
const DEFAULT_OUT_OF_JUMPS: f64 = 0.3;

#[derive(Debug, Clone)]
pub struct OpponentProfile {
    aggression: Rate,
    hook_first: Rate,
    hook_success: Rate,
    out_of_jumps: Rate,

    prev_dist: f64,
    their_hook_flying: bool,
    their_hook_grabbed: bool,
    our_hook_flying: bool,

    they_opened: Option<bool>,
    engaged: bool,
}

impl Default for OpponentProfile {
    fn default() -> Self {
        OpponentProfile {
            aggression: Rate::new(FAST_HALFLIFE, 8.0),
            hook_first: Rate::new(SLOW_HALFLIFE, 4.0),
            hook_success: Rate::new(SLOW_HALFLIFE, 4.0),
            out_of_jumps: Rate::new(FAST_HALFLIFE, 8.0),
            prev_dist: -1.0,
            their_hook_flying: false,
            their_hook_grabbed: false,
            our_hook_flying: false,
            they_opened: None,
            engaged: false,
        }
    }
}

impl OpponentProfile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(&mut self, me: &TeeState, them: &TeeState, hook_range_px: f64) {
        if !me.alive || !them.alive {
            self.end_engagement();
            return;
        }
        let dist = vdistance(me.pos, them.pos);

        if !them.frozen {
            if self.prev_dist >= 0.0 {
                self.aggression.observe(dist < self.prev_dist - 1.0);
            }
            self.out_of_jumps.observe(them.jumps_left == 0);
        } else {
            self.aggression.idle();
            self.out_of_jumps.idle();
        }
        self.prev_dist = dist;

        let in_range = dist <= hook_range_px;
        if in_range && !self.engaged {
            self.engaged = true;
            self.they_opened = None;
        }

        let their_flying_now = them.hook_state == HOOK_FLYING;
        if their_flying_now && !self.their_hook_flying {
            self.their_hook_flying = true;
            self.their_hook_grabbed = false;
            if self.engaged && self.they_opened.is_none() {
                self.they_opened = Some(true);
            }
        }
        if them.hook_state == HOOK_GRABBED {
            self.their_hook_grabbed = true;
        }
        if self.their_hook_flying && them.hook_state <= 0 {
            self.hook_success.observe(self.their_hook_grabbed);
            self.their_hook_flying = false;
            self.their_hook_grabbed = false;
        }

        let our_flying_now = me.hook_state == HOOK_FLYING;
        if our_flying_now && !self.our_hook_flying {
            self.our_hook_flying = true;
            if self.engaged && self.they_opened.is_none() {
                self.they_opened = Some(false);
            }
        }
        if !our_flying_now {
            self.our_hook_flying = false;
        }

        if !in_range && self.engaged {
            self.end_engagement();
        }
    }

    fn end_engagement(&mut self) {
        if self.engaged
            && let Some(opened) = self.they_opened
        {
            self.hook_first.observe(opened);
        }
        self.engaged = false;
        self.they_opened = None;
        self.prev_dist = -1.0;
    }

    pub fn read(&self) -> OpponentRead {
        let blend = |rate: &Rate, prior: f64| prior + rate.confidence() * (rate.value(prior) - prior);
        OpponentRead {
            aggression: blend(&self.aggression, DEFAULT_AGGRESSION),
            hook_opens_first: blend(&self.hook_first, DEFAULT_HOOK_OPENS_FIRST),
            hook_success: blend(&self.hook_success, DEFAULT_HOOK_SUCCESS),
            out_of_jumps: blend(&self.out_of_jumps, DEFAULT_OUT_OF_JUMPS),
            confidence: js::max_n(&[
                self.aggression.confidence(),
                self.hook_first.confidence(),
                self.hook_success.confidence(),
                self.out_of_jumps.confidence(),
            ]),
        }
    }

    pub fn reset(&mut self) {
        self.aggression.reset();
        self.hook_first.reset();
        self.hook_success.reset();
        self.out_of_jumps.reset();
        self.prev_dist = -1.0;
        self.their_hook_flying = false;
        self.their_hook_grabbed = false;
        self.our_hook_flying = false;
        self.engaged = false;
        self.they_opened = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tee_at(x: f64, alive: bool) -> TeeState {
        let mut t = crate::types::blank_tee_state();
        t.alive = alive;
        t.pos.x = x;
        t.jumps_left = 2;
        t
    }

    #[test]
    fn fresh_profile_reads_defaults() {
        let p = OpponentProfile::new();
        let r = p.read();
        assert_eq!(r.aggression, DEFAULT_AGGRESSION);
        assert_eq!(r.hook_opens_first, DEFAULT_HOOK_OPENS_FIRST);
        assert_eq!(r.out_of_jumps, DEFAULT_OUT_OF_JUMPS);
        assert_eq!(r.confidence, 0.0);
    }

    #[test]
    fn observe_with_a_dead_participant_ends_engagement_without_panicking() {
        let mut p = OpponentProfile::new();
        let me = tee_at(0.0, true);
        let dead = tee_at(10.0, false);
        p.observe(&me, &dead, 380.0);
        assert_eq!(p.read().aggression, DEFAULT_AGGRESSION);
    }

    #[test]
    fn reset_returns_to_defaults() {
        let mut p = OpponentProfile::new();
        let me = tee_at(0.0, true);
        let them = tee_at(50.0, true);
        for _ in 0..50 {
            p.observe(&me, &them, 380.0);
        }
        p.reset();
        assert_eq!(p.read().confidence, 0.0);
    }
}
