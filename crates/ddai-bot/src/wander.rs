//! Wandering — the port of `wander` (`bot.ts:4882-4971`, `docs/research/orig-bot.md` §8.7): what the
//! bot does when it has no target (or is in `passive` mode): walk, turn at walls / freeze / drops,
//! look around, now and then jump or hook. Everything it wants goes through the **guard** (the
//! shield: `escapeExists` after holding the input two ticks, else `saferInput`), and a hook it
//! throws is vetoed when the rope would catch a tee.
//!
//! Randomness comes from a seeded xorshift ([`Rng`]), so a wandering bot is reproducible from its
//! seed (the TS used an unseeded-looking `wanderRng`). Draw order follows the TS, including its
//! short-circuits, so the distributions match: direction `0` with probability 0.22 else a fair
//! coin; segment length `25 + floor(u * 175)` ticks; a drop ahead turns it around with probability
//! 0.7; a jump starts with probability 0.03 per tick (3..10 ticks long); a hook with 0.02 (15..49).
//! The aim turns at most 0.12 rad per decision toward its target, radius 300.

use ddai_brain::{Action, IVec2};
use ddai_physics::vmath::Vec2;

use ddai_nav::navigator::{LAG_MARGIN_TICKS, WALK_BRAKE_TICKS};

use crate::consts::*;
use crate::mapgrid::MapGrid;
use crate::tees::{HOOK_IDLE, Tee};

/// A small deterministic PRNG (xorshift64*), seeded.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 scramble so nearby seeds diverge and 0 is fine.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Rng((z ^ (z >> 31)) | 1)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f32) / (1u32 << 24) as f32
    }
}

/// The two outside services a wander step needs: the shield and the rope check.
pub trait WanderEnv {
    /// The guard: returns `wanted` unchanged when an escape from freeze still exists after holding
    /// it, else the shield's safer input.
    fn guard(&mut self, wanted: Action) -> Action;
    /// Would a hook thrown along `aim` catch a tee before a wall?
    fn rope_catches(&mut self, aim: IVec2) -> bool;
}

/// What one wander step reads.
pub struct WanderCtx<'a> {
    pub tick: i32,
    pub own: &'a Tee,
    pub grid: &'a MapGrid,
    /// The aim of the input we last sent (`prevInput.target`).
    pub prev_aim: (i32, i32),
    /// Wayblock mode: stay near this x and skip the random jumps/hooks (`anchorX`; 4.2).
    pub anchor_x: Option<f32>,
    /// Wayblock mode: look toward this point (`lookAt`; 4.2).
    pub look_at: Option<Vec2<f32>>,
    /// Wayblock guard on its spot: walk to the anchor and stand there, no random turns (`still`).
    pub still: bool,
    /// The input lag in ticks: the braking distance in front of a hazard counts it (`lagTicks()`).
    pub lag_ticks: i32,
}

#[derive(Debug, Clone)]
pub struct Wander {
    dir: i32,
    until: i32,
    look: f32,
    aim: f32,
    jump_until: i32,
    hook_until: i32,
    rng: Rng,
}

impl Wander {
    pub fn new(seed: u64) -> Self {
        Wander {
            dir: 0,
            until: 0,
            look: 0.0,
            aim: 0.0,
            jump_until: 0,
            hook_until: 0,
            rng: Rng::new(seed),
        }
    }

    /// New life: the timers restart (`bot.ts:2395-2397`). The direction carries on.
    pub fn respawned(&mut self) {
        self.until = 0;
        self.jump_until = 0;
        self.hook_until = 0;
    }

    /// One decision; see [`WanderEnv`].
    pub fn step(&mut self, c: &WanderCtx<'_>, env: &mut dyn WanderEnv) -> Action {
        let (tick, me) = (c.tick, c.own);
        if tick >= self.until {
            self.dir = if self.rng.next_f32() < 0.22 {
                0
            } else if self.rng.next_f32() < 0.5 {
                -1
            } else {
                1
            };
            self.until = tick + 25 + (self.rng.next_f32() * 175.0).floor() as i32;
            self.look = self.rng.next_f32() * 2.0 - 1.0;
            self.aim = self.look * std::f32::consts::PI;
        }
        if let Some(at) = c.look_at {
            self.aim = (at.y - me.pos.y).atan2(at.x - me.pos.x) + self.look * 0.2;
        }
        let still_at = c.anchor_x.filter(|_| c.still);
        if let Some(ax) = still_at {
            let off = ax - me.pos.x;
            self.dir = if off.abs() > 16.0 { off.signum() as i32 } else { 0 };
        } else if let Some(ax) = c.anchor_x
            && self.dir != 0
            && (me.pos.x - ax).abs() > 48.0
            && (ax - me.pos.x).signum() as i32 != self.dir
        {
            self.dir = -self.dir;
            self.until = tick + 40;
        }

        // The braking distance in front of a hazard grows with the speed and the lag: look that far ahead.
        let brake_px =
            |speed: f32| 24.0 + speed * (c.lag_ticks as f32 + WALK_BRAKE_TICKS as f32 + LAG_MARGIN_TICKS as f32);
        if self.dir != 0 {
            let ahead = Vec2::new(me.pos.x + (self.dir * 40) as f32, me.pos.y);
            let far = 40.0f32.max(brake_px((me.vel.x * self.dir as f32).max(0.0)));
            let far_x = me.pos.x + self.dir as f32 * far;
            let blocked = c.grid.is_solid(ahead.x, ahead.y)
                || c.grid.is_freeze(ahead.x, ahead.y)
                || c.grid.is_death(ahead.x, ahead.y)
                || (far > 40.0 && c.grid.hazard_within_px(me.pos.x, me.pos.y, self.dir, far));
            let drop = !c.grid.is_solid(ahead.x, me.pos.y + 40.0) && !c.grid.is_solid(ahead.x, me.pos.y + 80.0);
            let hazard_below =
                c.grid.hazard_below(ahead.x, me.pos.y) || (far > 40.0 && c.grid.hazard_below(far_x, me.pos.y));
            if blocked || hazard_below || (drop && (c.still || self.rng.next_f32() < 0.7)) {
                self.dir = if c.still { 0 } else { -self.dir };
                self.until = tick + 40;
            }
        }

        // Running at a hazard faster than it can be stopped: let go of the key (on the ground), or push back.
        if me.vel.x.abs() > 0.5 {
            let going = me.vel.x.signum() as i32;
            if c.grid
                .hazard_within_px(me.pos.x, me.pos.y, going, brake_px(me.vel.x.abs()))
            {
                let grounded = c.grid.is_solid(me.pos.x + 14.0, me.pos.y + 19.0)
                    || c.grid.is_solid(me.pos.x - 14.0, me.pos.y + 19.0);
                self.dir = if grounded { 0 } else { -going };
                self.until = tick + 25;
            }
        }

        if c.anchor_x.is_none() && tick >= self.jump_until && self.rng.next_f32() < 0.03 {
            self.jump_until = tick + 3 + (self.rng.next_f32() * 8.0).floor() as i32;
        }
        if c.anchor_x.is_none() && tick >= self.hook_until && self.rng.next_f32() < 0.02 {
            self.hook_until = tick + 15 + (self.rng.next_f32() * 35.0).floor() as i32;
        }
        let mut jump = tick < self.jump_until;
        let mut hook = tick < self.hook_until;

        let want = Action {
            direction: self.dir,
            jump,
            hook,
            fire: false,
            target: IVec2::new(c.prev_aim.0, c.prev_aim.1),
            wanted_weapon: None,
        };
        let safe = env.guard(want);
        let guarded = safe != want;
        if guarded {
            self.dir = safe.direction;
            jump = safe.jump;
            hook = safe.hook;
            if hook {
                self.hook_until = self.hook_until.max(tick + 20);
                self.aim = (safe.target.y as f32).atan2(safe.target.x as f32);
            }
            self.until = tick + 25;
        }

        let cur = (c.prev_aim.1 as f32).atan2(c.prev_aim.0 as f32);
        let mut d = self.aim - cur;
        while d > std::f32::consts::PI {
            d -= 2.0 * std::f32::consts::PI;
        }
        while d < -std::f32::consts::PI {
            d += 2.0 * std::f32::consts::PI;
        }
        let a = cur + d.clamp(-WANDER_AIM_STEP, WANDER_AIM_STEP);
        let (mut tx, mut ty) = (
            (a.cos() * WANDER_AIM_RADIUS).round() as i32,
            (a.sin() * WANDER_AIM_RADIUS).round() as i32,
        );
        if guarded && hook {
            tx = safe.target.x;
            ty = safe.target.y;
        }
        if hook
            && !guarded
            && (me.hooked_player >= 0 || (me.hook_state == HOOK_IDLE && env.rope_catches(IVec2::new(tx, ty))))
        {
            hook = false;
            self.hook_until = tick;
        }
        if tx == 0 && ty == 0 {
            tx = WANDER_AIM_RADIUS as i32;
        }
        Action {
            direction: self.dir,
            jump,
            hook,
            fire: false,
            target: IVec2::new(tx, ty),
            wanted_weapon: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapgrid::test_maps::*;

    fn tee_at(x: f32, y: f32) -> Tee {
        Tee {
            id: 0,
            alive: true,
            pos: Vec2::new(x, y),
            ..Tee::DEAD
        }
    }

    /// A test environment: a guard closure and a rope answer.
    struct Env<G: FnMut(Action) -> Action> {
        guard: G,
        rope: bool,
        rope_calls: u32,
    }
    impl<G: FnMut(Action) -> Action> WanderEnv for Env<G> {
        fn guard(&mut self, wanted: Action) -> Action {
            (self.guard)(wanted)
        }
        fn rope_catches(&mut self, _aim: IVec2) -> bool {
            self.rope_calls += 1;
            self.rope
        }
    }
    fn env(rope: bool) -> Env<fn(Action) -> Action> {
        Env {
            guard: |a| a,
            rope,
            rope_calls: 0,
        }
    }

    fn run(w: &mut Wander, grid: &MapGrid, own: &Tee, ticks: i32, prev: &mut (i32, i32)) -> Vec<Action> {
        let mut out = Vec::new();
        for tick in (0..ticks).step_by(2) {
            let a = w.step(
                &WanderCtx {
                    tick,
                    own,
                    grid,
                    prev_aim: *prev,
                    anchor_x: None,
                    look_at: None,
                    still: false,
                    lag_ticks: 0,
                },
                &mut env(false),
            );
            *prev = (a.target.x, a.target.y);
            out.push(a);
        }
        out
    }

    #[test]
    fn rng_is_deterministic_uniform_enough_and_seed_dependent() {
        let (mut a, mut b, mut c) = (Rng::new(1), Rng::new(1), Rng::new(2));
        let xs: Vec<f32> = (0..1000).map(|_| a.next_f32()).collect();
        assert_eq!(xs, (0..1000).map(|_| b.next_f32()).collect::<Vec<_>>());
        assert_ne!(xs[0], c.next_f32());
        assert!(xs.iter().all(|&x| (0.0..1.0).contains(&x)));
        let mean = xs.iter().sum::<f32>() / 1000.0;
        assert!((mean - 0.5).abs() < 0.05, "{mean}");
    }

    #[test]
    fn the_same_seed_wanders_identically_and_segments_last_25_to_199_ticks() {
        let grid = MapGrid::new(&room(200, 12, &[]));
        let own = tee_at(100.0 * 32.0, 10.0 * 32.0);
        let (mut w1, mut w2) = (Wander::new(9), Wander::new(9));
        let (mut p1, mut p2) = ((0, -1), (0, -1));
        let a = run(&mut w1, &grid, &own, 2000, &mut p1);
        let b = run(&mut w2, &grid, &own, 2000, &mut p2);
        assert_eq!(a, b);
        assert!(a.iter().any(|x| x.direction == 1) && a.iter().any(|x| x.direction == -1));
        assert!(a.iter().any(|x| x.direction == 0), "it also stands still");
        // Direction only changes on segment boundaries (>= 25 ticks) in an open room.
        let mut last = a[0].direction;
        let mut since = 0;
        for x in &a[1..] {
            since += 2;
            if x.direction != last {
                assert!(since >= 24, "changed after {since} ticks");
                last = x.direction;
                since = 0;
            }
        }
    }

    #[test]
    fn it_turns_around_at_a_wall_and_at_a_freeze_below_the_edge() {
        // Wall ahead (right): x tile 6 solid column, tee at tile 4 walking right.
        let wall: Vec<(u32, u32, u8)> = (1..11).map(|y| (6, y, SOLID)).collect();
        let grid = MapGrid::new(&room(12, 12, &wall));
        let own = tee_at(4.0 * 32.0 + 28.0, 10.0 * 32.0 + 16.0);
        let mut w = Wander::new(1);
        w.dir = 1;
        w.until = 1000;
        let a = w.step(
            &WanderCtx {
                tick: 10,
                own: &own,
                grid: &grid,
                prev_aim: (0, -1),
                anchor_x: None,
                look_at: None,
                still: false,
                lag_ticks: 0,
            },
            &mut env(false),
        );
        assert_eq!(a.direction, -1, "40 px ahead is inside the wall");
        assert_eq!(w.until, 50, "and it keeps the new way for 40 ticks");
        // Freeze below the ledge ahead: floor row 8 with a gap whose bottom is freeze.
        let extra = [(6, 8, FREEZE), (5, 9, SOLID)];
        let grid = MapGrid::new(&room(12, 12, &extra));
        let own = tee_at(4.0 * 32.0 + 28.0, 7.0 * 32.0 + 16.0);
        let mut w = Wander::new(1);
        w.dir = 1;
        w.until = 1000;
        let a = w.step(
            &WanderCtx {
                tick: 10,
                own: &own,
                grid: &grid,
                prev_aim: (0, -1),
                anchor_x: None,
                look_at: None,
                still: false,
                lag_ticks: 0,
            },
            &mut env(false),
        );
        assert_eq!(a.direction, -1, "a freeze tile below the ahead column");
    }

    #[test]
    fn the_guard_can_override_direction_jump_and_hook_and_the_timers_follow() {
        let grid = MapGrid::new(&room(40, 12, &[]));
        let own = tee_at(20.0 * 32.0, 10.0 * 32.0);
        let mut w = Wander::new(3);
        w.dir = 1;
        w.until = 1000;
        let mut e = Env {
            guard: |a: Action| Action {
                direction: -1,
                jump: true,
                hook: true,
                target: IVec2::new(0, -200),
                ..a
            },
            rope: true,
            rope_calls: 0,
        };
        let a = w.step(
            &WanderCtx {
                tick: 100,
                own: &own,
                grid: &grid,
                prev_aim: (300, 0),
                anchor_x: None,
                look_at: None,
                still: false,
                lag_ticks: 0,
            },
            &mut e,
        );
        assert_eq!(e.rope_calls, 0, "a guarded hook is not vetoed again");
        assert_eq!((a.direction, a.jump, a.hook), (-1, true, true));
        assert_eq!((a.target.x, a.target.y), (0, -200), "the shield's own aim");
        assert_eq!(w.until, 125, "tick + 25");
        assert!(w.hook_until >= 120, "hook held at least 20 ticks");
    }

    #[test]
    fn a_hook_that_would_catch_a_tee_is_dropped_and_the_timer_cleared() {
        let grid = MapGrid::new(&room(40, 12, &[]));
        let own = tee_at(20.0 * 32.0, 10.0 * 32.0);
        let mut w = Wander::new(3);
        w.dir = 0;
        w.until = 1000;
        w.hook_until = 500;
        let a = w.step(
            &WanderCtx {
                tick: 100,
                own: &own,
                grid: &grid,
                prev_aim: (300, 0),
                anchor_x: None,
                look_at: None,
                still: false,
                lag_ticks: 0,
            },
            &mut env(true),
        );
        assert!(!a.hook);
        assert_eq!(w.hook_until, 100);
    }

    #[test]
    fn a_hook_while_holding_someone_is_dropped_too_and_the_aim_turns_at_most_0_12_rad() {
        let grid = MapGrid::new(&room(40, 12, &[]));
        let mut own = tee_at(20.0 * 32.0, 10.0 * 32.0);
        own.hooked_player = 5;
        let mut w = Wander::new(3);
        w.dir = 0;
        w.until = 1000;
        w.hook_until = 500;
        w.aim = std::f32::consts::PI; // far from the current aim
        let a = w.step(
            &WanderCtx {
                tick: 100,
                own: &own,
                grid: &grid,
                prev_aim: (300, 0),
                anchor_x: None,
                look_at: None,
                still: false,
                lag_ticks: 0,
            },
            &mut env(false),
        );
        assert!(!a.hook, "self.hooked_player >= 0");
        let turned = (a.target.y as f32).atan2(a.target.x as f32).abs();
        assert!((turned - 0.12).abs() < 0.01, "turned {turned} rad");
        let len = (a.target.x as f32).hypot(a.target.y as f32);
        assert!((len - 300.0).abs() < 2.0);
    }

    #[test]
    fn the_anchor_pulls_it_back_and_suppresses_random_jumps() {
        let grid = MapGrid::new(&room(80, 12, &[]));
        let own = tee_at(60.0 * 32.0, 10.0 * 32.0);
        let mut w = Wander::new(3);
        w.dir = 1; // walking away from the anchor at x = 20 tiles
        w.until = 1000;
        let a = w.step(
            &WanderCtx {
                tick: 10,
                own: &own,
                grid: &grid,
                prev_aim: (300, 0),
                anchor_x: Some(20.0 * 32.0),
                look_at: None,
                still: false,
                lag_ticks: 0,
            },
            &mut env(false),
        );
        assert_eq!(a.direction, -1, "turned back toward the anchor");
        for tick in 0..2000 {
            let a = w.step(
                &WanderCtx {
                    tick,
                    own: &own,
                    grid: &grid,
                    prev_aim: (300, 0),
                    anchor_x: Some(60.0 * 32.0),
                    look_at: None,
                    still: false,
                    lag_ticks: 0,
                },
                &mut env(false),
            );
            assert!(!a.jump && !a.hook, "anchored: no random jumps or hooks (tick {tick})");
        }
    }

    fn ctx<'a>(
        own: &'a Tee,
        grid: &'a MapGrid,
        tick: i32,
        anchor: Option<f32>,
        still: bool,
        lag: i32,
    ) -> WanderCtx<'a> {
        WanderCtx {
            tick,
            own,
            grid,
            prev_aim: (300, 0),
            anchor_x: anchor,
            look_at: None,
            still,
            lag_ticks: lag,
        }
    }

    #[test]
    fn a_guard_that_is_still_walks_to_the_anchor_and_stands_there() {
        // A floor of 40 tiles with a gap: the anchor is on the floor right before the gap.
        let grid = MapGrid::new(&room(40, 12, &[]));
        let mut w = Wander::new(5);
        w.dir = 1;
        w.until = 100_000;
        // Far from the anchor (8 tiles to its right): walks toward it, left.
        let own = tee_at(28.0 * 32.0, 10.0 * 32.0 + 16.0);
        let a = w.step(
            &ctx(&own, &grid, 10, Some(20.0 * 32.0 + 16.0), true, 0),
            &mut env(false),
        );
        assert_eq!(a.direction, -1, "toward the anchor");
        // Within 16 px of it: no direction at all, for as long as it stays there, with no random jumps or hooks.
        let own = tee_at(20.0 * 32.0 + 16.0 + 10.0, 10.0 * 32.0 + 16.0);
        for tick in 20..1200 {
            let a = w.step(
                &ctx(&own, &grid, tick, Some(20.0 * 32.0 + 16.0), true, 0),
                &mut env(false),
            );
            assert_eq!(a.direction, 0, "standing still on the anchor (tick {tick})");
            assert!(!a.jump && !a.hook);
        }
    }

    #[test]
    fn running_at_a_freeze_it_cannot_stop_in_front_of_lets_go_on_the_ground_and_pushes_back_in_the_air() {
        // A solid floor up to tile x = 14, then freeze; the tee runs right on the floor at 8 px/tick.
        let mut freeze: Vec<(u32, u32, u8)> = (14..30).map(|x| (x, 10, FREEZE)).collect();
        freeze.extend((1..14).map(|x| (x, 10, SOLID)));
        let grid = MapGrid::new(&room(40, 12, &freeze));
        let mut own = tee_at(10.0 * 32.0 + 16.0, 9.0 * 32.0 + 16.0);
        own.vel.x = 8.0;
        let mut w = Wander::new(7);
        w.dir = 1;
        w.until = 100_000;
        // Braking distance 24 + 8 * (lag 2 + 3 + 2) = 80 px: the freeze at x = 14 tiles is 4 tiles = 128 px away:
        // far enough, it keeps running.
        let a = w.step(&ctx(&own, &grid, 10, None, false, 2), &mut env(false));
        assert_eq!(a.direction, 1, "still room to stop");
        // 2 tiles from the freeze (64 px, inside the 80 px): on the ground it lets go of the key...
        let mut near = own;
        near.pos.x = 12.0 * 32.0 + 16.0;
        let a = w.step(&ctx(&near, &grid, 12, None, false, 2), &mut env(false));
        assert_eq!(a.direction, 0, "on the ground: let go");
        // ...in the air (nothing under it) it pushes back against the way it goes.
        let mut air = near;
        air.pos.y = 5.0 * 32.0;
        let mut w = Wander::new(7);
        w.dir = 1;
        w.until = 100_000;
        let a = w.step(&ctx(&air, &grid, 12, None, false, 2), &mut env(false));
        assert_eq!(a.direction, -1, "in the air: push back");
    }
}
