//! Task 3.23 (D-121): three fixes for the weaknesses the 2026-10-08 duel against a human showed. All **off by default** (a default [`DuelFixConfig`] changes
//! nothing: the decision is bit-identical to the one before).
//!
//! 1. **Static victim** (the fixed point and the AFK hole). A victim that has kept its direction neutral and its hook in for [`DuelFixConfig::static_after`]
//!    decisions, stands still and is free is "static". No plan of the 27-tick horizon blocks it (walking up to it and hauling it into a freeze takes
//!    longer), so the small terms of the score decide and the warm plan "stand still" wins every decision, for ever. With [`DuelFixConfig::static_push`]
//!    the choice is made among the plans that act, as long as one of them is safe under every modelled reply (a longer horizon was tried and did not
//!    help: half the candidates fit the budget, E-038).
//! 2. **The counter** ("hook from above, pass under, release"): the model of the victim that reacts (the scripted bot, `scripted_action`) holds its hook
//!    on us for as long as we are in rope range. A human who has pulled us up releases it when he has passed below us, and our upward speed carries us into
//!    the ceiling. [`DuelFixConfig::counter_release`] makes the reacting victim do that in every rollout (so the robust stage sees the plans that leave us
//!    rising under him). While his hook holds us the robust stage also believes he reacts ([`DuelFixConfig::hooked_belief`]) and re-scores the defensive
//!    techniques ([`DuelFixConfig::protect_defence`]). (A veto of our jump while he hangs above us on his hook was tried and made the counterfactual
//!    worse, E-038: it is not in the code.)
//! 3. **Finishing**: after a block the frozen victim is dragged into the nearest freeze it can lie in with short hook grabs
//!    ([`DuelFixConfig::finish_approach`]), not nursed and not hit (a hit unfreezes).
//!
//! The functions here are pure: the search and the brain pass the planning world's tees and collision and apply the answer.

use crate::plan_world::PlanCollision;
use crate::planner::PlanStep;
use crate::types::{PlayerInput, TeeState, WEAPON_HAMMER};

/// Where the fixes are switched on and tuned. The default is everything off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuelFixConfig {
    /// The fixes act only while the bot tells the brain a duel is on (`LiveContext::duel`; the arena's live view sets it).
    pub duel_only: bool,
    /// Fix 1: against a static victim the final choice is made among the plans that act (a plan whose first [`IDLE_STEPS`] steps are all neutral is
    /// "idle"), provided one of them is safe under every modelled reply; the best two active plans are re-scored by the robust stage even when the
    /// idle ones rank above them.
    pub static_push: bool,
    /// Fix 2: the reacting victim of the robust stage lets go of us once it is below us while we rise (`PlannerConfig::counter_release`).
    pub counter_release: bool,
    /// Fix 2: while the victim's hook holds us (or flies at us) the robust stage believes it reacts with at least this probability (`0` = the learned belief
    /// alone). The belief of an opponent that has not walked at us for a while is small, and the worst case then counts for next to nothing (weights of 0.05-0.1
    /// in the post-mortem's decisions), though a hook on us is the clearest sign there is that he acts.
    pub hooked_belief: f64,
    /// Fix 2: while the victim's hook holds us the best two defensive techniques (T9, T10, T19, ...) are re-scored by the robust stage whatever their cheap rank.
    pub protect_defence: bool,
    /// Fix 3: a frozen victim that lies off the freeze with enough freeze left is not answered by standing still: the choice is made among the plans that
    /// act (as in fix 1), and `finish_approach` approach plans ([`crate::hybrid::techniques::approach_plans`], technique T30) join the pool.
    pub finish_push: bool,
    /// Fix 3: at most this many approach plans per such decision (`0` = none).
    pub finish_approach: usize,
    /// Fix 3: no hammer swing at a frozen victim (a hit unfreezes it).
    pub no_hammer_frozen: bool,
    /// Decisions the victim must have been passive for (neutral direction, hook in) before it counts as static.
    pub static_after: u32,
}

impl Default for DuelFixConfig {
    fn default() -> Self {
        DuelFixConfig {
            duel_only: true,
            static_push: false,
            counter_release: false,
            hooked_belief: 0.0,
            protect_defence: false,
            finish_push: false,
            finish_approach: 0,
            no_hammer_frozen: false,
            static_after: 25,
        }
    }
}

impl DuelFixConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.hooked_belief.is_finite() && (0.0..=1.0).contains(&self.hooked_belief)) {
            return Err("hybrid: duel_fixes.hooked_belief in [0, 1]".into());
        }
        Ok(())
    }
}

/// The speed (px/tick, each axis) under which a tee "stands still".
pub const STATIC_SPEED: f64 = 0.5;

/// A plan is idle when its first this many steps (9 ticks) press nothing.
pub const IDLE_STEPS: usize = 3;

/// Whether `victim` is a static target: free and alive, hook in, neutral, at rest, and it has been passive for `passive` decisions.
pub fn is_static(cfg: &DuelFixConfig, victim: &TeeState, passive: u32) -> bool {
    victim.alive
        && !victim.frozen
        && victim.direction == 0
        && victim.hook_state <= 0
        && victim.vel.x.abs() < STATIC_SPEED
        && victim.vel.y.abs() < STATIC_SPEED
        && passive >= cfg.static_after
}

/// Whether the decision is one against a static victim that we should not answer by standing still.
pub fn static_push(cfg: &DuelFixConfig, me: &TeeState, victim: &TeeState, passive: u32) -> bool {
    cfg.static_push && me.alive && !me.frozen && is_static(cfg, victim, passive)
}

/// Whether `plan` does nothing in its first [`IDLE_STEPS`] steps.
pub fn is_idle(plan: &[PlanStep]) -> bool {
    plan.iter()
        .take(IDLE_STEPS)
        .all(|s| s.dir == 0 && s.jump == 0 && s.hook == 0 && s.fire == 0)
}

/// How far (px) the victim must be below us before the counter lets go of the hook: he has passed us.
pub const RELEASE_BELOW_PX: f64 = 8.0;
/// How fast (px/tick, upward) we must be rising for the release to matter (a rope pull that has not got us going yet is not the pattern).
pub const RELEASE_RISE_VY: f64 = 4.0;

/// The counter of the 2026-10-08 human: whether the victim, whose hook **holds us** (`hooked_player` is our id), lets go now -- it is below us (`y` grows
/// downwards) and we rise at more than [`RELEASE_RISE_VY`]: the speed the rope gave us carries us up without his pull. A hook that is not on us (one he fires at
/// us from below, one he holds on a wall) is never released by this rule (review 3.23 round 1, F1).
pub fn counter_releases(me: &TeeState, victim: &TeeState) -> bool {
    me.alive
        && !me.frozen
        && victim.hooked_player == me.id
        && victim.pos.y > me.pos.y + RELEASE_BELOW_PX
        && me.vel.y < -RELEASE_RISE_VY
}

/// The freeze a frozen victim must still have for the finishing to start (ticks): shorter than that nothing we walk to arrives in time.
pub const FINISH_MIN_TICKS: i64 = 30;

/// Fix 3: the victim lies frozen off the freeze: alive, frozen with at least [`FINISH_MIN_TICKS`] ticks left, at rest (it has stopped falling) and not
/// touching a freeze or death tile (in one it stays frozen by itself, and the best we can do is stand by).
pub fn lies_off_the_freeze<C: PlanCollision>(col: &C, victim: &TeeState) -> bool {
    victim.alive
        && victim.frozen
        && victim.freeze_ticks_left >= FINISH_MIN_TICKS
        && victim.vel.x.abs() < STATIC_SPEED
        && victim.vel.y.abs() < STATIC_SPEED
        && !crate::seal::touches_freeze(col, victim.pos.x, victim.pos.y)
}

/// Whether the decision finishes a frozen victim that lies off the freeze (the choice is made among the plans that act).
pub fn finish_push<C: PlanCollision>(cfg: &DuelFixConfig, col: &C, me: &TeeState, victim: &TeeState) -> bool {
    cfg.finish_push && me.alive && !me.frozen && lies_off_the_freeze(col, victim)
}

/// A swing reaches a tee within this many px of ours (`14 + 28` from the point `21` px ahead, and a step).
pub const HAMMER_REACH_PX: f64 = 70.0;

/// Fix 3: a fresh swing at a frozen victim within reach is dropped from `out` (the fire counter stays where it was, released). A hit unfreezes the tee it
/// hits, so against a frozen victim a swing is a hand-off. Returns whether it dropped one. `prev_fire` is the fire counter we sent last.
pub fn drop_hammer_at_frozen(
    cfg: &DuelFixConfig,
    me: &TeeState,
    victim: &TeeState,
    prev_fire: i32,
    out: &mut PlayerInput,
) -> bool {
    // A fresh press is the counter moving on to an odd value (`reflex.rs`).
    if !cfg.no_hammer_frozen
        || !me.alive
        || !victim.alive
        || !victim.frozen
        || out.fire == prev_fire
        || out.fire & 1 == 0
        || me.active_weapon != WEAPON_HAMMER
    {
        return false;
    }
    let (dx, dy) = (victim.pos.x - me.pos.x, victim.pos.y - me.pos.y);
    if (dx * dx + dy * dy).sqrt() > HAMMER_REACH_PX {
        return false;
    }
    out.fire = if prev_fire & 1 == 1 { prev_fire + 1 } else { prev_fire };
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan_world::PlanWorld;
    use crate::types::{blank_tee_state, empty_input};
    use crate::vmath::Vec2;

    fn tee(id: i32, x: f64, y: f64) -> TeeState {
        let mut t = blank_tee_state();
        t.id = id;
        t.alive = true;
        t.pos = Vec2 { x, y };
        t
    }

    fn on() -> DuelFixConfig {
        DuelFixConfig {
            static_push: true,
            counter_release: true,
            finish_push: true,
            no_hammer_frozen: true,
            ..DuelFixConfig::default()
        }
    }

    /// A 12 x 20 tile box: a solid floor on row 19 and a freeze tile at (5, 3) in the ceiling (its bottom edge is y = 128).
    fn box_world() -> crate::physics_adapter::PhysicsWorld {
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

    #[test]
    fn the_default_is_everything_off() {
        let c = DuelFixConfig::default();
        assert!(!c.static_push && !c.counter_release && !c.finish_push && !c.no_hammer_frozen && !c.protect_defence);
        assert!(c.hooked_belief == 0.0 && c.finish_approach == 0);
        assert!(c.duel_only);
        assert!(c.validate().is_ok());
        // Off, nothing is static, nothing is pushed, no swing is dropped.
        let (me, v) = (tee(0, 0.0, 0.0), tee(1, 40.0, 0.0));
        assert!(!static_push(&c, &me, &v, 1000));
        let world = box_world();
        let mut frozen = tee(1, 176.0, 400.0);
        frozen.frozen = true;
        frozen.freeze_ticks_left = 100;
        assert!(!finish_push(&c, world.collision(), &me, &frozen));
        let mut out = empty_input();
        out.fire = 1;
        assert!(!drop_hammer_at_frozen(&c, &me, &frozen, 0, &mut out));
        assert_eq!(out.fire, 1);
        let mut bad = c;
        bad.hooked_belief = 1.5;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn a_victim_is_static_only_when_passive_free_and_at_rest_for_long_enough() {
        let c = on();
        let me = tee(0, 0.0, 0.0);
        let still = tee(1, 40.0, 0.0);
        assert!(!static_push(&c, &me, &still, c.static_after - 1), "not for long enough");
        assert!(static_push(&c, &me, &still, c.static_after));
        let mut walking = still;
        walking.direction = 1;
        assert!(!static_push(&c, &me, &walking, 100));
        let mut hooking = still;
        hooking.hook_state = 1;
        assert!(!static_push(&c, &me, &hooking, 100));
        let mut sliding = still;
        sliding.vel = Vec2 { x: 3.0, y: 0.0 };
        assert!(!static_push(&c, &me, &sliding, 100));
        let mut frozen = still;
        frozen.frozen = true;
        assert!(
            !static_push(&c, &me, &frozen, 100),
            "a frozen victim is the finishing's case"
        );
        let mut dead = still;
        dead.alive = false;
        assert!(!static_push(&c, &me, &dead, 100));
        // Nor while we are frozen or dead.
        let mut me_frozen = me;
        me_frozen.frozen = true;
        assert!(!static_push(&c, &me_frozen, &still, 100));
    }

    #[test]
    fn a_plan_is_idle_when_its_first_three_steps_press_nothing() {
        let idle = PlanStep {
            dir: 0,
            jump: 0,
            hook: 0,
            fire: 0,
            aim: 0.0,
        };
        let walk = PlanStep { dir: 1, ..idle };
        assert!(is_idle(&[idle; 9]));
        assert!(
            is_idle(&[idle, idle, idle, walk, walk, walk, walk, walk, walk]),
            "waits 9 ticks, then acts"
        );
        assert!(!is_idle(&[idle, idle, walk, idle, idle, idle, idle, idle, idle]));
        for step in [
            PlanStep { jump: 1, ..idle },
            PlanStep { hook: 1, ..idle },
            PlanStep { fire: 1, ..idle },
        ] {
            assert!(!is_idle(&[step; 9]));
        }
    }

    #[test]
    fn the_counter_lets_go_once_he_is_below_us_and_we_rise() {
        let mut me = tee(0, 100.0, 300.0);
        me.vel = Vec2 { x: 0.0, y: -12.0 };
        let hooking = |mut t: TeeState| {
            t.hooked_player = 0;
            t
        };
        let below = hooking(tee(1, 100.0, 330.0));
        let above = hooking(tee(1, 100.0, 200.0));
        let level = hooking(tee(1, 100.0, 304.0));
        assert!(counter_releases(&me, &below));
        assert!(
            !counter_releases(&me, &above),
            "he is still above: the rope pulls us up, he holds"
        );
        assert!(!counter_releases(&me, &level), "not passed yet");
        let mut slow = me;
        slow.vel.y = -2.0;
        assert!(!counter_releases(&slow, &below), "we are not rising fast");
        let mut falling = me;
        falling.vel.y = 6.0;
        assert!(!counter_releases(&falling, &below));
        let mut frozen = me;
        frozen.frozen = true;
        assert!(!counter_releases(&frozen, &below));
        // His hook is not on us (a hook he fires at us from below, one he holds on a wall): nothing to let go of.
        for hooked in [-1, 2] {
            let mut elsewhere = below;
            elsewhere.hooked_player = hooked;
            assert!(!counter_releases(&me, &elsewhere), "hooked_player {hooked}");
        }
    }

    #[test]
    fn a_frozen_victim_off_the_freeze_is_finished_and_one_in_it_is_left_alone() {
        let world = box_world();
        let col = world.collision();
        let me = tee(0, 100.0, 560.0);
        let mut off = tee(1, 176.0, 560.0);
        off.frozen = true;
        off.freeze_ticks_left = 100;
        assert!(finish_push(&on(), col, &me, &off));
        // In the freeze tile it stays frozen by itself.
        let mut in_freeze = off;
        in_freeze.pos = Vec2 { x: 176.0, y: 110.0 };
        assert!(!finish_push(&on(), col, &me, &in_freeze));
        // Falling: wait while it falls. Thawing soon: nothing walked to arrives. Free: not the finishing's case. We frozen: no.
        let mut falling = off;
        falling.vel.y = 8.0;
        assert!(!finish_push(&on(), col, &me, &falling));
        let mut thawing = off;
        thawing.freeze_ticks_left = FINISH_MIN_TICKS - 1;
        assert!(!finish_push(&on(), col, &me, &thawing));
        let mut free = off;
        free.frozen = false;
        assert!(!finish_push(&on(), col, &me, &free));
        let mut me_frozen = me;
        me_frozen.frozen = true;
        assert!(!finish_push(&on(), col, &me_frozen, &off));
    }

    #[test]
    fn no_swing_at_a_frozen_victim_within_reach() {
        let me = tee(0, 100.0, 300.0);
        let mut frozen = tee(1, 150.0, 300.0);
        frozen.frozen = true;
        // A fresh press from a released counter, and from a held one (release and press).
        for (prev, fresh, released) in [(0, 1, 0), (1, 3, 2)] {
            let mut out = empty_input();
            out.fire = fresh;
            assert!(
                drop_hammer_at_frozen(&on(), &me, &frozen, prev, &mut out),
                "prev {prev}"
            );
            assert_eq!(out.fire, released, "prev {prev}");
        }
        // No fresh press (held, or nothing): nothing to drop.
        for (prev, now) in [(1, 1), (0, 0), (1, 2)] {
            let mut out = empty_input();
            out.fire = now;
            assert!(!drop_hammer_at_frozen(&on(), &me, &frozen, prev, &mut out));
            assert_eq!(out.fire, now);
        }
        // Out of reach, a free victim, the gun: untouched.
        let mut far = frozen;
        far.pos.x = 300.0;
        let mut free = frozen;
        free.frozen = false;
        let mut gun = me;
        gun.active_weapon = 1;
        for (m, v) in [(&me, &far), (&me, &free), (&gun, &frozen)] {
            let mut out = empty_input();
            out.fire = 1;
            assert!(!drop_hammer_at_frozen(&on(), m, v, 0, &mut out));
            assert_eq!(out.fire, 1);
        }
    }

    /// A rollout of `steps` idle steps of ours against the reacting victim (`scripted_action`), the counter on or off; returns the victim's tee after it.
    fn react_rollout(w: &mut crate::physics_adapter::PhysicsWorld, counter: bool, steps: usize) -> TeeState {
        let saved = w.save_state();
        let cfg = crate::config::PlannerConfig {
            counter_release: counter,
            ..crate::config::preset_normal()
        };
        let mut planner = crate::planner::Planner::<crate::physics_adapter::PhysicsWorld>::new(cfg);
        planner.react_this_pass = true;
        planner.keep_final = true;
        let field = crate::fields::hazard_field(w.collision());
        let unfreeze = crate::fields::unfreeze_field(w.collision());
        let idle = PlanStep {
            dir: 0,
            jump: 0,
            hook: 0,
            fire: 0,
            aim: 0.0,
        };
        let plan = vec![idle; steps];
        let score = planner.evaluate_impl(w, 0, 1, empty_input(), &plan, empty_input(), &field, &unfreeze, None);
        assert!(score.is_some());
        let him = w.get_tee(1).expect("the victim");
        w.restore_state(&saved);
        him
    }

    /// Review 3.23 round 1, F1: the counter lets go only of a hook that **holds us**. A hook he fires at us from below is kept (the victim stands on the floor
    /// 190 px under us and we rise): in the first three ticks the reacting victim throws it, counter on or off.
    #[test]
    fn a_hook_fired_at_us_from_below_is_kept_in_the_rollout() {
        let mut world = box_world();
        world.add_tee(0, Vec2 { x: 176.0, y: 400.0 });
        world.add_tee(1, Vec2 { x: 176.0, y: 585.0 });
        let mut me = world.get_tee(0).expect("us");
        me.vel = Vec2 { x: 0.0, y: -12.0 };
        world.apply_tee_state(0, &me);
        for counter in [false, true] {
            let him = react_rollout(&mut world, counter, 1);
            assert!(
                him.hook_state != 0,
                "counter {counter}: his hook was dropped before it was even thrown ({him:?})"
            );
        }
    }

    /// ... and a hook he holds on us is let go once he is below us while we rise (he passed under us with the rope on and the speed it gave us carries us up):
    /// with the counter off the rope is still on us after the first step of the rollout, with it on it is gone. Nothing differs while he is still above us.
    #[test]
    fn a_hook_held_on_us_is_released_once_he_has_passed_under_in_the_rollout() {
        let held_by = |him_y: f64, counter: bool| {
            let mut world = box_world();
            world.add_tee(0, Vec2 { x: 176.0, y: 300.0 });
            world.add_tee(1, Vec2 { x: 176.0, y: him_y });
            let mut me = world.get_tee(0).expect("us");
            me.vel = Vec2 { x: 0.0, y: -12.0 };
            world.apply_tee_state(0, &me);
            let mut him = world.get_tee(1).expect("him");
            him.hooked_player = 0;
            him.hook_state = crate::types::HOOK_GRABBED;
            him.hook_pos = Vec2 { x: 176.0, y: 300.0 };
            world.apply_tee_state(1, &him);
            let start = world.get_tee(1).expect("him");
            assert_eq!(
                (start.hooked_player, start.hook_state),
                (0, crate::types::HOOK_GRABBED),
                "the rope holds us"
            );
            react_rollout(&mut world, counter, 1)
        };
        // He is below us (120 px) and we rise at 12 px/tick: the counter lets go, the plain model holds.
        assert_eq!(
            held_by(420.0, false).hooked_player,
            0,
            "the plain model keeps the rope on us"
        );
        assert_ne!(held_by(420.0, true).hooked_player, 0, "the counter lets go");
        // He is still above us (the rope pulls us up toward him): both hold.
        assert_eq!(held_by(180.0, false).hooked_player, 0);
        assert_eq!(held_by(180.0, true).hooked_player, 0, "above us he holds");
    }
}
