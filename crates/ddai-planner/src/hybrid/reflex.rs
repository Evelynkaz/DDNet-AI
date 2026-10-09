//! Task 3.19 (D-116): two rules on top of the hybrid's decision, for the close-range duel. Both are **off by default**.
//!
//! * **Reflex hammer.** The search finds a swing only when one of its few candidates happens to press fire at the right step along an aim that
//!   reaches; the clips of the 2026-10-07 test duel show it swinging in 6% of the frames where the hammer was ready and the other tee within
//!   reach (the competitor: 25%). The reflex does not ask the search: when the other tee is free (a hit unfreezes, so never a frozen one),
//!   our hammer is ready, and a swing along the aim at its predicted position would hit, it presses fire (and aims, unless the plan is
//!   launching a hook this tick, whose throw the aim steers). The reach is `character.cpp:525`: the swing finds a tee whose centre is within
//!   `14 + 28` px of the point `21` px ahead of ours.
//! * **Hammer-safe envelope.** A hammer hit throws its target up at `-8.4..-11` px/tick (`character.cpp:549-554`), so a hit on a tee that is
//!   already rising (a jump: `-13.2`/`-12`; a hook climb) sends it far above its apex, and in a box with a freeze ceiling that is the end. The
//!   envelope looks at the input the search chose: if the other tee can hit us within the lag and its reaction (free, within `threat_px`, hammer
//!   ready soon), and a worst-case hit after this input would carry our apex into a freeze tile that nothing solid shields, while the input
//!   without its jump (or hook climb) would not, the jump (the hook) is dropped.
//!
//! The functions are pure: the brain passes the planning world's tees and collision and applies the answer.

use crate::plan_world::PlanCollision;
use crate::tuning::{GRAVITY, PHYSICAL_SIZE};
use crate::types::{PlayerInput, TeeState};
use crate::vmath::Vec2;

/// Where the reflex and the envelope are switched on and tuned. The default is everything off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReflexConfig {
    /// The reflex hammer.
    pub hammer: bool,
    /// Both rules act only while the bot tells the brain a duel is on (`LiveContext::duel`); the live bot sets this, the arena (which is a duel) does not need to.
    pub duel_only: bool,
    /// Pixels added to the hit radius (negative = stricter): a swing predicted to miss by less than this is still thrown (the prediction of the other
    /// tee's position is not exact). A miss costs the 6-tick reload only.
    pub slack_px: f64,
    /// Swing only when the hit throws the other tee into a freeze or death tile: its ballistic flight after the hit (`fields::launch_flight_lands_in_hazard`,
    /// the planner's own 50-tick estimate from its position and velocity) ends in one. A hit on a tee that is not going to freeze only throws it out of reach
    /// (and, in the round rules, lets it out of the pit it lies in); the search keeps deciding those.
    pub hazard_only: bool,
    /// A swing needs this many ticks since our last one (`now - attack_tick`), on top of the reload timer. A live world cannot know whether the last swing hit
    /// (the lockout is 16 ticks after a hit, 6 after a miss) and reports the reload as 0: 16 is the safe reading there. `0` = the reload timer alone (the
    /// arena's true view knows it exactly).
    pub lockout_ticks: i64,
    /// The hammer-safe envelope.
    pub envelope: bool,
    /// The other tee counts as a threat when its centre is at most this far (px): its hammer reach plus the ground it covers within our lag and its reaction.
    pub threat_px: f64,
    /// A hit counts as ready when the other tee's hammer is at most this many ticks from ready.
    pub ready_ticks: i64,
    /// The worst-case kick of a hit (px/tick, upward): `normalize(dir + (0, -1.1)) * 10 + (0, -1)` is `-8.4` from the side, `-11` from below or above.
    pub kick: f64,
    /// Extra pixels the apex must stay below a freeze tile.
    pub margin_px: f64,
    /// Also drop the hook of a climb: our hook holds the other tee, it is above us, and the pull alone would carry the apex into the freeze.
    pub hook_climb: bool,
}

impl Default for ReflexConfig {
    fn default() -> Self {
        ReflexConfig {
            hammer: false,
            duel_only: false,
            slack_px: 0.0,
            hazard_only: false,
            lockout_ticks: 0,
            envelope: false,
            threat_px: 110.0,
            ready_ticks: 8,
            kick: 11.0,
            margin_px: 16.0,
            hook_climb: false,
        }
    }
}

impl ReflexConfig {
    pub fn validate(&self) -> Result<(), String> {
        let finite = |x: f64| x.is_finite();
        if !finite(self.slack_px) || !(-20.0..=40.0).contains(&self.slack_px) || !(0..=40).contains(&self.lockout_ticks)
        {
            return Err("hybrid: reflex slack_px in [-20, 40], lockout_ticks in 0..=40".into());
        }
        if !finite(self.threat_px) || self.threat_px <= 0.0 || self.ready_ticks < 0 || self.ready_ticks > 40 {
            return Err("hybrid: envelope threat_px > 0 and ready_ticks in 0..=40".into());
        }
        if !finite(self.kick) || !(0.0..=30.0).contains(&self.kick) || !finite(self.margin_px) {
            return Err("hybrid: envelope kick in [0, 30], margin_px finite".into());
        }
        Ok(())
    }
}

/// `WEAPON_HAMMER`.
const WEAPON_HAMMER: i32 = 0;
/// The swing's start point is this far ahead of the tee (`GetProximityRadius() * 0.75`).
const SWING_AHEAD_PX: f64 = PHYSICAL_SIZE * 0.75;
/// A swing hits a tee whose centre is nearer than this to the start point (`14 + 28`).
const SWING_RADIUS_PX: f64 = PHYSICAL_SIZE * 0.5 + PHYSICAL_SIZE;
/// The length of the aim vector the planner sends (`cos * 300`).
const AIM_LEN: f64 = 300.0;

fn can_act(t: &TeeState) -> bool {
    t.alive && !t.frozen && t.freeze_ticks_left <= 0 && t.deep_frozen != Some(true)
}

/// Whether a swing from `me` along `(ax, ay)` hits a tee at `en` (centre), by the server's own test, with `slack` pixels of tolerance.
pub fn swing_hits(me: Vec2, en: Vec2, ax: f64, ay: f64, slack: f64) -> bool {
    let len = (ax * ax + ay * ay).sqrt();
    if len < 1e-6 {
        return false;
    }
    let (sx, sy) = (me.x + ax / len * SWING_AHEAD_PX, me.y + ay / len * SWING_AHEAD_PX);
    ((en.x - sx).powi(2) + (en.y - sy).powi(2)).sqrt() < SWING_RADIUS_PX + slack
}

/// The situation a swing is decided in: the planning world's tick, the two tees at the tick the input acts in, the fire counter we sent last, and whether the
/// plan throws a new hook this tick (its aim steers the throw).
#[derive(Debug, Clone, Copy)]
pub struct SwingAt<'a> {
    pub now: i64,
    pub me: &'a TeeState,
    pub victim: &'a TeeState,
    pub prev_fire: i32,
    pub launching_hook: bool,
}

/// What the reflex did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swing {
    /// Nothing to do (or the plan already swings).
    No,
    /// Fire pressed, aim kept (the plan's own aim reaches).
    Fire,
    /// Fire pressed and aimed at the other tee.
    FireAimed,
}

/// The reflex hammer: presses fire in `out` when a swing would hit. `me` and `victim` are the planning world's tees at the tick `out` acts in;
/// `now` is the planning world's tick, `prev_fire` the fire counter we sent last. `launching_hook` tells that `out` throws a new hook this tick (its aim is the throw's).
pub fn reflex_swing<C: PlanCollision>(cfg: &ReflexConfig, col: &C, at: &SwingAt<'_>, out: &mut PlayerInput) -> Swing {
    let SwingAt {
        now,
        me,
        victim,
        prev_fire,
        launching_hook,
    } = *at;
    if !cfg.hammer || !can_act(me) || !can_act(victim) || me.active_weapon != WEAPON_HAMMER {
        return Swing::No;
    }
    // Ready: the reload timer is known exactly in the arena's true view; a live world reconstructs it (or reports 0).
    if me.reload_ticks.unwrap_or(0) > 0 || now - me.attack_tick < cfg.lockout_ticks {
        return Swing::No;
    }
    // The plan already presses fire this tick (a level `true` is a fresh press, so an odd counter is a press): leave its aim alone.
    if out.fire & 1 == 1 {
        return Swing::No;
    }
    let (dx, dy) = (victim.pos.x - me.pos.x, victim.pos.y - me.pos.y);
    if cfg.hazard_only {
        let sep = (dx * dx + dy * dy).sqrt();
        if crate::fields::launch_flight_lands_in_hazard(col, victim.pos, me.pos, sep, victim.vel) <= 0.0 {
            return Swing::No;
        }
    }
    let plan_aim_hits = swing_hits(me.pos, victim.pos, out.target_x, out.target_y, cfg.slack_px);
    let aim_hits = swing_hits(me.pos, victim.pos, dx, dy, cfg.slack_px);
    let aimed = if plan_aim_hits {
        false
    } else if aim_hits && !launching_hook {
        true
    } else {
        return Swing::No;
    };
    if aimed {
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        out.target_x = (dx / len * AIM_LEN).round();
        out.target_y = (dy / len * AIM_LEN).round();
    }
    // A fresh press: +1 when released, +2 (release and press) when held.
    out.fire = if prev_fire & 1 == 1 {
        prev_fire + 2
    } else {
        prev_fire + 1
    };
    if aimed { Swing::FireAimed } else { Swing::Fire }
}

/// What the envelope removed from the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Veto {
    pub jump: bool,
    pub hook: bool,
}

impl Veto {
    pub fn any(self) -> bool {
        self.jump || self.hook
    }
}

/// The rise (px) of a ballistic arc that starts upward at `vy` (px/tick, negative = up): `vy^2 / (2g)`.
fn rise(vy: f64) -> f64 {
    if vy >= 0.0 { 0.0 } else { vy * vy / (2.0 * *GRAVITY) }
}

/// The distance (px) up from `from` to the first freeze/death tile of the columns the tee covers, if no solid tile comes first within `limit`.
pub(crate) fn hazard_above<C: PlanCollision>(col: &C, from: Vec2, limit: f64) -> Option<f64> {
    let mut best: Option<f64> = None;
    for dx in [-10.0, 0.0, 10.0] {
        let mut d = 8.0;
        while d <= limit {
            let (x, y) = (from.x + dx, from.y - d);
            if col.is_solid(x, y) {
                break;
            }
            if col.is_hazard(x, y) {
                best = Some(best.map_or(d, |b| b.min(d)));
                break;
            }
            d += 8.0;
        }
    }
    best
}

/// Whether a worst-case hit from the other tee would carry us into a freeze after `vy` (our vertical speed after the input).
fn unsafe_after_hit<C: PlanCollision>(cfg: &ReflexConfig, col: &C, me: Vec2, vy: f64) -> bool {
    let up = rise(vy.min(0.0) - cfg.kick);
    hazard_above(col, me, up + cfg.margin_px).is_some_and(|d| d <= up + cfg.margin_px)
}

/// The hammer-safe envelope: drops the jump (and, with `hook_climb`, the hook) of `out` when a worst-case hit would carry us into a freeze and
/// the input without it would not. Returns what it dropped.
pub fn envelope<C: PlanCollision>(
    cfg: &ReflexConfig,
    col: &C,
    me: &TeeState,
    victim: &TeeState,
    out: &mut PlayerInput,
) -> Veto {
    let mut veto = Veto::default();
    if !cfg.envelope || !can_act(me) || !can_act(victim) {
        return veto;
    }
    let sep = ((victim.pos.x - me.pos.x).powi(2) + (victim.pos.y - me.pos.y).powi(2)).sqrt();
    if sep > cfg.threat_px || victim.active_weapon != WEAPON_HAMMER {
        return veto;
    }
    // His hammer must be ready within the lag and his reaction (a hammer that hit lately is locked for 16 ticks, one that missed for 6).
    if victim.reload_ticks.unwrap_or(0) > cfg.ready_ticks {
        return veto;
    }
    // What our vertical speed is after this input. A fresh jump (the key was up, a jump is left) sets the speed: 13.2 from the ground, 12 in the
    // air; the stronger is taken (the worst case).
    let jump_impulse = (out.jump != 0 && me.jumps_left > 0 && me.jumped & 1 == 0).then_some(13.2_f64);
    let vy_plain = me.vel.y;
    let vy_jump = jump_impulse.map_or(vy_plain, |imp| (-imp).min(vy_plain));
    // A hook climb: our hook holds the victim above us, it pulls up at up to the drag speed.
    let climbing = out.hook != 0 && me.hooked_player == victim.id && victim.pos.y < me.pos.y - 16.0 && cfg.hook_climb;
    let vy_climb = if climbing { vy_jump.min(-12.0) } else { vy_jump };
    if jump_impulse.is_some()
        && unsafe_after_hit(cfg, col, me.pos, vy_jump)
        && !unsafe_after_hit(cfg, col, me.pos, vy_plain)
    {
        out.jump = 0;
        veto.jump = true;
    }
    if climbing {
        let base = if veto.jump { vy_plain } else { vy_jump };
        if unsafe_after_hit(cfg, col, me.pos, vy_climb.min(base))
            && !unsafe_after_hit(cfg, col, me.pos, base.max(vy_plain))
        {
            out.hook = 0;
            veto.hook = true;
        }
    }
    veto
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan_world::PlanWorld;
    use crate::types::{blank_tee_state, empty_input};

    fn tee(id: i32, x: f64, y: f64) -> TeeState {
        let mut t = blank_tee_state();
        t.id = id;
        t.alive = true;
        t.pos = Vec2 { x, y };
        t.reload_ticks = Some(0);
        t.jumps_left = 2;
        t
    }

    fn cfg() -> ReflexConfig {
        ReflexConfig {
            hammer: true,
            envelope: true,
            hook_climb: true,
            ..ReflexConfig::default()
        }
    }

    #[test]
    fn the_swing_reach_is_the_servers() {
        let me = Vec2 { x: 0.0, y: 0.0 };
        // Straight at it: 21 + 42 = 63 px.
        assert!(swing_hits(me, Vec2 { x: 62.0, y: 0.0 }, 1.0, 0.0, 0.0));
        assert!(!swing_hits(me, Vec2 { x: 64.0, y: 0.0 }, 1.0, 0.0, 0.0));
        // Off the line of the aim the reach shrinks, behind us it is nothing.
        assert!(!swing_hits(me, Vec2 { x: 0.0, y: 60.0 }, 1.0, 0.0, 0.0));
        assert!(!swing_hits(me, Vec2 { x: -30.0, y: 0.0 }, 1.0, 0.0, 0.0));
        // The slack lets a near miss through.
        assert!(swing_hits(me, Vec2 { x: 66.0, y: 0.0 }, 1.0, 0.0, 4.0));
        assert!(
            !swing_hits(me, Vec2 { x: 62.0, y: 0.0 }, 0.0, 0.0, 0.0),
            "no aim, no swing"
        );
    }

    #[test]
    fn fire_is_pressed_aimed_at_a_free_victim_in_reach() {
        let me = tee(0, 0.0, 0.0);
        let en = tee(1, 50.0, 10.0);
        let mut out = empty_input();
        out.target_x = -300.0; // the plan looks the other way
        let s = reflex_swing(
            &cfg(),
            box_world(false).collision(),
            &SwingAt {
                now: 1000,
                me: &me,
                victim: &en,
                prev_fire: 4,
                launching_hook: false,
            },
            &mut out,
        );
        assert_eq!(s, Swing::FireAimed);
        assert_eq!(out.fire, 5, "released (even) -> a press is +1");
        assert!(out.target_x > 280.0, "aimed at it: {}", out.target_x);
        let mut held = empty_input();
        assert_eq!(
            reflex_swing(
                &cfg(),
                box_world(false).collision(),
                &SwingAt {
                    now: 1000,
                    me: &me,
                    victim: &en,
                    prev_fire: 5,
                    launching_hook: false
                },
                &mut held
            ),
            Swing::FireAimed
        );
        assert_eq!(held.fire, 7, "held (odd) -> release and press is +2");
    }

    #[test]
    fn it_keeps_a_plan_aim_that_already_reaches_and_a_hook_throw_aim_that_does_not() {
        let me = tee(0, 0.0, 0.0);
        let en = tee(1, 50.0, 0.0);
        let mut out = empty_input();
        out.target_x = 300.0;
        out.target_y = 0.0;
        assert_eq!(
            reflex_swing(
                &cfg(),
                box_world(false).collision(),
                &SwingAt {
                    now: 1000,
                    me: &me,
                    victim: &en,
                    prev_fire: 0,
                    launching_hook: true
                },
                &mut out
            ),
            Swing::Fire
        );
        assert_eq!((out.target_x, out.target_y), (300.0, 0.0));
        let mut other = empty_input();
        other.target_x = -300.0;
        assert_eq!(
            reflex_swing(
                &cfg(),
                box_world(false).collision(),
                &SwingAt {
                    now: 1000,
                    me: &me,
                    victim: &en,
                    prev_fire: 0,
                    launching_hook: true
                },
                &mut other
            ),
            Swing::No,
            "a new hook throw keeps its aim"
        );
        assert_eq!(other.fire, 0);
    }

    #[test]
    fn it_never_swings_at_a_frozen_victim_out_of_reach_with_a_reloading_hammer_or_when_off() {
        let me = tee(0, 0.0, 0.0);
        let mut en = tee(1, 50.0, 0.0);
        let mut out = empty_input();
        let mut frozen = en;
        frozen.frozen = true;
        frozen.freeze_ticks_left = 40;
        assert_eq!(
            reflex_swing(
                &cfg(),
                box_world(false).collision(),
                &SwingAt {
                    now: 1000,
                    me: &me,
                    victim: &frozen,
                    prev_fire: 0,
                    launching_hook: false
                },
                &mut out
            ),
            Swing::No
        );
        en.pos.x = 90.0;
        assert_eq!(
            reflex_swing(
                &cfg(),
                box_world(false).collision(),
                &SwingAt {
                    now: 1000,
                    me: &me,
                    victim: &en,
                    prev_fire: 0,
                    launching_hook: false
                },
                &mut out
            ),
            Swing::No
        );
        en.pos.x = 50.0;
        let mut reloading = me;
        reloading.reload_ticks = Some(9);
        assert_eq!(
            reflex_swing(
                &cfg(),
                box_world(false).collision(),
                &SwingAt {
                    now: 1000,
                    me: &reloading,
                    victim: &en,
                    prev_fire: 0,
                    launching_hook: false
                },
                &mut out
            ),
            Swing::No
        );
        let mut frozen_me = me;
        frozen_me.frozen = true;
        frozen_me.freeze_ticks_left = 10;
        assert_eq!(
            reflex_swing(
                &cfg(),
                box_world(false).collision(),
                &SwingAt {
                    now: 1000,
                    me: &frozen_me,
                    victim: &en,
                    prev_fire: 0,
                    launching_hook: false
                },
                &mut out
            ),
            Swing::No
        );
        assert_eq!(
            reflex_swing(
                &ReflexConfig::default(),
                box_world(false).collision(),
                &SwingAt {
                    now: 1000,
                    me: &me,
                    victim: &en,
                    prev_fire: 0,
                    launching_hook: false
                },
                &mut out
            ),
            Swing::No
        );
        assert_eq!(out, empty_input(), "nothing touched");
    }

    #[test]
    fn the_lockout_holds_the_swing_back_after_our_last_one() {
        let mut me = tee(0, 0.0, 0.0);
        me.attack_tick = 990;
        let en = tee(1, 50.0, 0.0);
        let locked = ReflexConfig {
            lockout_ticks: 16,
            ..cfg()
        };
        let mut out = empty_input();
        assert_eq!(
            reflex_swing(
                &locked,
                box_world(false).collision(),
                &SwingAt {
                    now: 1000,
                    me: &me,
                    victim: &en,
                    prev_fire: 0,
                    launching_hook: false
                },
                &mut out
            ),
            Swing::No,
            "10 ticks after the last swing"
        );
        assert_eq!(
            reflex_swing(
                &locked,
                box_world(false).collision(),
                &SwingAt {
                    now: 1006,
                    me: &me,
                    victim: &en,
                    prev_fire: 0,
                    launching_hook: false
                },
                &mut out
            ),
            Swing::FireAimed,
            "16 ticks after it"
        );
        let mut out = empty_input();
        assert_eq!(
            reflex_swing(
                &cfg(),
                box_world(false).collision(),
                &SwingAt {
                    now: 1000,
                    me: &me,
                    victim: &en,
                    prev_fire: 0,
                    launching_hook: false
                },
                &mut out
            ),
            Swing::FireAimed,
            "no lockout: the reload timer alone"
        );
    }

    #[test]
    fn a_plan_that_swings_already_is_left_alone() {
        let me = tee(0, 0.0, 0.0);
        let en = tee(1, 50.0, 0.0);
        let mut out = empty_input();
        out.target_x = -300.0;
        out.fire = 5; // prev 4 (released): the plan pressed
        assert_eq!(
            reflex_swing(
                &cfg(),
                box_world(false).collision(),
                &SwingAt {
                    now: 1000,
                    me: &me,
                    victim: &en,
                    prev_fire: 4,
                    launching_hook: false
                },
                &mut out
            ),
            Swing::No
        );
        assert_eq!((out.fire, out.target_x), (5, -300.0));
    }

    /// A 12 x 20 tile box: a solid floor on row 19, a freeze tile at (5, 3) in the ceiling (its bottom edge is y = 128), and a solid tile
    /// `shield` at (5, 8) when asked.
    fn box_world(shield: bool) -> crate::physics_adapter::PhysicsWorld {
        use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
        let (w, h) = (12usize, 20usize);
        let mut game = vec![Tile::default(); w * h];
        let mut set = |x: usize, y: usize, index: u8| {
            game[y * w + x] = Tile {
                index,
                ..Default::default()
            };
        };
        for x in 0..w {
            set(x, 19, TILE_SOLID);
        }
        set(5, 3, TILE_FREEZE);
        if shield {
            set(5, 8, TILE_SOLID);
        }
        let map = MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        crate::physics_adapter::PhysicsWorld::new(std::sync::Arc::new(map), 1)
    }

    fn jumping() -> PlayerInput {
        let mut o = empty_input();
        o.jump = 1;
        o
    }

    #[test]
    fn the_envelope_drops_a_jump_that_a_hit_would_carry_into_the_freeze_ceiling() {
        let world = box_world(false);
        let col = world.collision();
        // 224 px under the ceiling tile's bottom edge: a hit alone lifts us 121 px (safe), a jump and a hit 585 px (not).
        let me = tee(0, 5.0 * 32.0 + 16.0, 352.0);
        let en = tee(1, me.pos.x + 60.0, me.pos.y);
        let mut out = jumping();
        let v = envelope(&cfg(), col, &me, &en, &mut out);
        assert_eq!((v.jump, v.hook, out.jump), (true, false, 0));
        // Nothing to drop when we do not jump.
        let mut out = empty_input();
        assert!(!envelope(&cfg(), col, &me, &en, &mut out).any());
    }

    #[test]
    fn the_envelope_leaves_the_input_alone_when_it_cannot_matter() {
        let me = tee(0, 5.0 * 32.0 + 16.0, 352.0);
        let near = tee(1, me.pos.x + 60.0, me.pos.y);
        let jump = jumping();
        let kept = |cfg: &ReflexConfig, world: &crate::physics_adapter::PhysicsWorld, me: &TeeState, en: &TeeState| {
            let mut out = jump;
            envelope(cfg, world.collision(), me, en, &mut out);
            out.jump == 1
        };
        // A solid tile in the way shields the freeze.
        assert!(kept(&cfg(), &box_world(true), &me, &near));
        let open = box_world(false);
        assert!(!kept(&cfg(), &open, &me, &near), "the control");
        // The other tee is out of reach, frozen, or its hammer is locked for longer than the lag and its reaction.
        let far = tee(1, me.pos.x + 400.0, me.pos.y);
        assert!(kept(&cfg(), &open, &me, &far));
        let mut frozen = near;
        frozen.frozen = true;
        frozen.freeze_ticks_left = 30;
        assert!(kept(&cfg(), &open, &me, &frozen));
        let mut locked = near;
        locked.reload_ticks = Some(14);
        assert!(kept(&cfg(), &open, &me, &locked));
        let mut soon = near;
        soon.reload_ticks = Some(6);
        assert!(!kept(&cfg(), &open, &me, &soon));
        // No jump left, or the key already down (no fresh jump): nothing to drop either.
        let mut no_jump = me;
        no_jump.jumps_left = 0;
        assert!(kept(&cfg(), &open, &no_jump, &near));
        let mut held = me;
        held.jumped = 1;
        assert!(kept(&cfg(), &open, &held, &near));
        // And it is off by default.
        assert!(kept(&ReflexConfig::default(), &open, &me, &near));
    }

    #[test]
    fn the_envelope_drops_the_hook_of_a_climb_toward_a_tee_above_when_the_pull_is_the_danger() {
        let world = box_world(false);
        let col = world.collision();
        let mut me = tee(0, 5.0 * 32.0 + 16.0, 352.0);
        me.hooked_player = 1;
        let en = tee(1, me.pos.x + 20.0, me.pos.y - 70.0);
        let mut out = empty_input();
        out.hook = 1;
        let v = envelope(&cfg(), col, &me, &en, &mut out);
        assert_eq!((v.hook, out.hook), (true, 0), "{v:?}");
        // Not a climb when the tee is below us, or when climbs are left alone.
        let below = tee(1, me.pos.x + 20.0, me.pos.y + 40.0);
        let mut out = empty_input();
        out.hook = 1;
        assert!(!envelope(&cfg(), col, &me, &below, &mut out).any());
        let no_climb = ReflexConfig {
            hook_climb: false,
            ..cfg()
        };
        let mut out = empty_input();
        out.hook = 1;
        assert!(!envelope(&no_climb, col, &me, &en, &mut out).any());
    }

    /// D-042: the two rules cost a few microseconds a decision (a scan of three columns of at most a few dozen probes and a handful of float operations); the
    /// bound here is a hundred times that, loose enough for a loaded machine. The measured numbers are in `docs/research/duel-3.19.md`.
    #[test]
    fn the_rules_cost_microseconds_not_milliseconds() {
        let world = box_world(false);
        let col = world.collision();
        let me = tee(0, 5.0 * 32.0 + 16.0, 352.0);
        let en = tee(1, me.pos.x + 50.0, me.pos.y);
        let n = 20_000;
        let t0 = std::time::Instant::now();
        let mut sink = 0i32;
        for i in 0..n {
            let mut out = jumping();
            out.fire = 2 * (i % 7);
            let v = envelope(&cfg(), col, &me, &en, &mut out);
            let s = reflex_swing(
                &cfg(),
                col,
                &SwingAt {
                    now: 1000 + i64::from(i),
                    me: &me,
                    victim: &en,
                    prev_fire: out.fire,
                    launching_hook: false,
                },
                &mut out,
            );
            sink += out.fire + i32::from(v.jump) + i32::from(s == Swing::Fire);
        }
        let per_call = t0.elapsed().as_secs_f64() / f64::from(n) * 1e6;
        assert!(sink != 0);
        assert!(per_call < 300.0, "{per_call:.2} us per decision");
        eprintln!("reflex + envelope: {per_call:.2} us per decision");
    }

    #[test]
    fn the_hazard_only_reflex_swings_only_when_the_hit_throws_the_other_tee_into_a_freeze() {
        let world = box_world(false);
        let col = world.collision();
        let only = ReflexConfig {
            hazard_only: true,
            ..cfg()
        };
        // Under the ceiling tile (its bottom edge at y = 128, x 160..192): a hit from just below throws a tee at y = 190 up at about 11 px/tick, into the freeze.
        let me = tee(0, 5.0 * 32.0 + 16.0 - 8.0, 228.0);
        let en = tee(1, 5.0 * 32.0 + 16.0, 190.0);
        let mut out = empty_input();
        assert_ne!(
            reflex_swing(
                &only,
                col,
                &SwingAt {
                    now: 1000,
                    me: &me,
                    victim: &en,
                    prev_fire: 0,
                    launching_hook: false
                },
                &mut out
            ),
            Swing::No
        );
        // The same hit low in the box, far under the ceiling: it only throws the tee about, nothing freezes: the search keeps deciding.
        let me = tee(0, 5.0 * 32.0 + 16.0 - 8.0, 568.0);
        let en = tee(1, 5.0 * 32.0 + 16.0, 530.0);
        let mut out = empty_input();
        assert_eq!(
            reflex_swing(
                &only,
                col,
                &SwingAt {
                    now: 1000,
                    me: &me,
                    victim: &en,
                    prev_fire: 0,
                    launching_hook: false
                },
                &mut out
            ),
            Swing::No
        );
        // Without the switch it swings there too.
        let mut out = empty_input();
        assert_ne!(
            reflex_swing(
                &cfg(),
                col,
                &SwingAt {
                    now: 1000,
                    me: &me,
                    victim: &en,
                    prev_fire: 0,
                    launching_hook: false
                },
                &mut out
            ),
            Swing::No
        );
    }
}
