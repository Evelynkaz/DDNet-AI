//! `src/plan/shield.ts` — the "shield" post-filter: checks that the chosen input still leaves an
//! escape route (`escapeExists`), and if not, looks for a safer nearby aim/input
//! (`saferInput`). Runs directly against a [`PlanWorld`] (its own save/restore, not through
//! `Planner::evaluate`) — the last thing `decide_once` does before committing to an input
//! (`docs/research/orig-plan.md` §1.3 step 28).
//!
//! Review round 2, F12: `escape_exists`/`safer_input` (below, unchanged) are exactly TS's own
//! algorithm and are what `decide_once` (the TS-parity path) still calls -- up to 6 escape
//! candidates × (`ESCAPE_TICKS` + `SETTLE_TICKS`) ≈ 126 physics steps each, and `safer_input` can
//! run up to 6 more full `escape_exists` searches on top of that (measured: up to 2,056 physics
//! steps / ~20 ms of real work in one decision, review round 2's `r2-phase.log`). That is fine for
//! `decide_once` (no deadline exists there at all, D-017) but not for `decide_production`, which
//! must stay inside its budget. [`escape_exists_bounded`]/[`safer_input_bounded`] are the same
//! algorithm with an optional deadline checked every escape candidate and every few ticks inside
//! one, returning [`Bounded::TimedOut`] instead of silently running long when time is up --
//! `Planner::decide_production` is the only caller of the bounded entry points.

use crate::clock::Clock;
use crate::plan_world::{PlanCollision, PlanWorld};
use crate::tuning::PHYSICAL_SIZE;
use crate::types::{PlayerInput, empty_input};
use ddai_jsmath as js;
use std::collections::HashMap;

const HALF: f64 = PHYSICAL_SIZE / 2.0;
const ESCAPE_TICKS: i32 = 36;
const SETTLE_TICKS: i32 = 90;
/// Speed (px/tick) under which a tee hanging on its hook is at rest.
const HANG_SPEED: f64 = 1.5;

/// Calm ticks a hang from our own tile hook needs before it counts as settled (task 3.5b, review F1): hooked to a
/// tile, speed at most [`HANG_SPEED`], not frozen, for this many ticks in a row. A tee still swinging (even one
/// that is merely grabbed at the moment) is in transit and may swing into freeze: the settle loop keeps running.
/// Within ~42 px of the anchor the hook does not pull, so a tee rising through that zone falls freely and passes
/// through speeds in `[-HANG_SPEED, HANG_SPEED]` for up to `2 * HANG_SPEED / gravity` ticks at the top of its arc
/// (7 at stock gravity); the run has to be longer than that: `ceil(2 * HANG_SPEED / gravity) + 2` (8 at stock
/// gravity 0.5; round 2, F1). A world with no gravity never settles early (the loop just runs on).
fn hang_calm_ticks(gravity: f64) -> i32 {
    if gravity <= 1e-3 {
        return SETTLE_TICKS;
    }
    ((2.0 * HANG_SPEED / gravity).ceil() as i32 + 2).min(SETTLE_TICKS)
}
const MAX_TURN_RAD: f64 = 1.5;

/// Review round 2, F12: how often (in ticks) a bounded rollout re-reads the clock -- "per escape
/// and every few steps" per the fix's own wording. `Instant::now()` is cheap (tens of ns) but not
/// free, and a rollout is 36-90 ticks long, so checking every tick would be pure overhead; every
/// 4th tick bounds the worst-case overshoot inside one rollout to a handful of physics steps
/// (measured: ~2-8 µs each, `component_cost_report`) while still keeping the clock-read count low.
const CLOCK_CHECK_STRIDE: i32 = 4;

/// Outcome of a deadline-aware check (review round 2, F12). `TimedOut` is deliberately a separate
/// case from a definitive negative result: `Planner::decide_production` treats "we ran out of
/// time before finding out" differently from "we looked and there truly is no escape/no safer
/// input" (the former keeps the searched choice and flags `shield_incomplete`; the latter is a
/// complete, honest answer with nothing to flag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bounded<T> {
    Done(T),
    TimedOut,
}

fn mk(dir: i32, jump: i32, aim_x: f64, aim_y: f64) -> PlayerInput {
    let mut e = empty_input();
    e.direction = dir;
    e.jump = jump;
    e.target_x = aim_x;
    e.target_y = aim_y;
    e
}

/// `escapes(vx, aimX, aimY)` (`shield.ts:10-32`).
fn escapes(vx: f64, aim_x: f64, aim_y: f64) -> Vec<PlayerInput> {
    let brake_dir: i32 = if js::abs(vx) < 0.5 {
        0
    } else if vx > 0.0 {
        -1
    } else {
        1
    };
    let mut out = if brake_dir == 0 {
        vec![mk(0, 0, aim_x, aim_y), mk(0, 1, aim_x, aim_y)]
    } else {
        vec![
            mk(0, 0, aim_x, aim_y),
            mk(0, 1, aim_x, aim_y),
            mk(brake_dir, 0, aim_x, aim_y),
            mk(brake_dir, 1, aim_x, aim_y),
        ]
    };
    for ax in [0, brake_dir] {
        let mut h = mk(brake_dir, 1, f64::from(ax) * 150.0, -300.0);
        h.hook = 1;
        out.push(h);
        if brake_dir == 0 {
            break;
        }
    }
    out
}

fn apply_others<W: PlanWorld>(world: &mut W, others: &HashMap<i32, PlayerInput>) {
    for (&id, inp) in others {
        world.set_input(id, *inp);
    }
}

/// `true` once every `CLOCK_CHECK_STRIDE` ticks (always checked on tick 0) when `deadline` is
/// `Some` and past; always `false` when `deadline` is `None` -- the same "only ever `None` for the
/// parity path" contract `Planner::evaluate_impl`'s own deadline parameter already relies on.
fn deadline_hit(deadline: Option<(&dyn Clock, f64)>, tick: i32) -> bool {
    if tick % CLOCK_CHECK_STRIDE != 0 {
        return false;
    }
    match deadline {
        Some((clock, deadline_ms)) => clock.now_ms() >= deadline_ms,
        None => false,
    }
}

/// Reusable snapshot buffers of the shield (task 3.5): `escape_exists` saves the world twice per
/// call, and a snapshot of the production world is ~105 kB, so the hybrid search keeps these two
/// across calls and decisions and refills them in place (`save_state_into`) instead of allocating
/// two fresh snapshots every time.
pub struct ShieldBuffers<W: PlanWorld> {
    start: Option<W::SavedState>,
    after_hold: Option<W::SavedState>,
}

impl<W: PlanWorld> Default for ShieldBuffers<W> {
    fn default() -> Self {
        ShieldBuffers {
            start: None,
            after_hold: None,
        }
    }
}

/// Fills `slot` with the world's current state, reusing the buffer once it exists.
fn save_into<W: PlanWorld>(world: &W, slot: &mut Option<W::SavedState>) {
    match slot {
        Some(s) => world.save_state_into(s),
        None => *slot = Some(world.save_state()),
    }
}

/// `escapeExists(world, selfId, input, holdTicks, others)` (`shield.ts:34-72`). Always restores
/// `world` to its state on entry before returning (TS's `finally { world.restoreState(start) }`)
/// — implemented as a plain call + restore around [`escape_exists_inner`] rather than a
/// closure/IIFE, so `world` is only ever reborrowed, never moved, across the early-return paths.
/// Unchanged from round 1: TS-parity callers (`decide_once`) always get a complete answer, never
/// `TimedOut` (see [`escape_exists_bounded`]'s doc comment for why that is a hard guarantee, not
/// just usually true).
pub fn escape_exists<W: PlanWorld>(
    world: &mut W,
    self_id: i32,
    input: &PlayerInput,
    hold_ticks: i32,
    others: &HashMap<i32, PlayerInput>,
) -> bool {
    match escape_exists_bounded(world, self_id, input, hold_ticks, others, None) {
        Bounded::Done(b) => b,
        Bounded::TimedOut => unreachable!("escape_exists_bounded(deadline: None) never times out"),
    }
}

/// Review round 2, F12: `decide_production`'s deadline-aware counterpart to [`escape_exists`].
/// `deadline: None` behaves exactly like [`escape_exists`] (and is what it's built on -- `None`
/// makes every [`deadline_hit`] check unconditionally `false`, so this is provably the same
/// control flow, not merely "the same in practice").
pub fn escape_exists_bounded<W: PlanWorld>(
    world: &mut W,
    self_id: i32,
    input: &PlayerInput,
    hold_ticks: i32,
    others: &HashMap<i32, PlayerInput>,
    deadline: Option<(&dyn Clock, f64)>,
) -> Bounded<bool> {
    escape_exists_buffered(
        world,
        self_id,
        input,
        hold_ticks,
        others,
        deadline,
        &mut ShieldBuffers::default(),
    )
}

/// [`escape_exists_bounded`] with caller-owned snapshot buffers (see [`ShieldBuffers`]): the same
/// algorithm and the same answers, without allocating a snapshot per call once the buffers exist.
pub fn escape_exists_buffered<W: PlanWorld>(
    world: &mut W,
    self_id: i32,
    input: &PlayerInput,
    hold_ticks: i32,
    others: &HashMap<i32, PlayerInput>,
    deadline: Option<(&dyn Clock, f64)>,
    bufs: &mut ShieldBuffers<W>,
) -> Bounded<bool> {
    escape_exists_ext(
        world,
        self_id,
        input,
        hold_ticks,
        others,
        deadline,
        bufs,
        &mut ShieldOpts::default(),
    )
}

/// Task 3.5b extras of the hybrid shield (the TS-parity and plain production callers use none).
#[derive(Default)]
pub struct ShieldOpts<'a> {
    /// Tried before anything else, from the decision state: does the remainder of the chosen plan,
    /// rolled out exactly, leave the tee safe? `Done(true)` ends the check with "an escape exists".
    /// Anything else falls through to the ordinary escapes (a timeout here is not a verdict).
    pub first: Option<&'a mut dyn FnMut() -> Bounded<bool>>,
    /// Escape inputs tried after the standard ones: hooks aimed at anchors (walls, ceilings).
    pub extras: &'a [PlayerInput],
}

/// [`escape_exists_buffered`] with the hybrid's extras (see [`ShieldOpts`]).
#[allow(clippy::too_many_arguments)]
pub fn escape_exists_ext<W: PlanWorld>(
    world: &mut W,
    self_id: i32,
    input: &PlayerInput,
    hold_ticks: i32,
    others: &HashMap<i32, PlayerInput>,
    deadline: Option<(&dyn Clock, f64)>,
    bufs: &mut ShieldBuffers<W>,
    opts: &mut ShieldOpts<'_>,
) -> Bounded<bool> {
    if let Some(first) = opts.first.as_mut()
        && first() == Bounded::Done(true)
    {
        return Bounded::Done(true);
    }
    save_into(world, &mut bufs.start);
    let result = escape_exists_inner(
        world,
        self_id,
        input,
        hold_ticks,
        others,
        deadline,
        &mut bufs.after_hold,
        opts.extras,
    );
    world.restore_state(bufs.start.as_ref().expect("just saved"));
    result
}

#[allow(clippy::too_many_arguments)]
fn escape_exists_inner<W: PlanWorld>(
    world: &mut W,
    self_id: i32,
    input: &PlayerInput,
    hold_ticks: i32,
    others: &HashMap<i32, PlayerInput>,
    deadline: Option<(&dyn Clock, f64)>,
    after_hold_slot: &mut Option<W::SavedState>,
    extras: &[PlayerInput],
) -> Bounded<bool> {
    crate::prof::inc_escape();
    for t in 0..hold_ticks {
        if deadline_hit(deadline, t) {
            return Bounded::TimedOut;
        }
        world.set_input(self_id, *input);
        apply_others(world, others);
        world.step();
        crate::prof::inc_step();
        let Some(me) = world.get_tee(self_id) else {
            return Bounded::Done(false);
        };
        if !me.alive || me.frozen {
            return Bounded::Done(false);
        }
    }
    escape_tail(world, self_id, input, others, deadline, after_hold_slot, &[], extras)
}

/// The escape phase of `escapeExists`, from the world's current state (the state after the hold,
/// or -- task 3.5b -- after the rest of a plan): snapshots it, then tries each escape input in turn
/// (`lead`, then `escapes()`, then `extras`: hook escapes the hybrid aims at anchors) for `ESCAPE_TICKS` and
/// lets the tee settle; `lead` inputs go before the standard ones. `input` supplies the aim and the jump-press tick the escapes are derived from.
/// With `extras` empty this is exactly the TS algorithm's second half (the TS-parity callers).
#[allow(clippy::too_many_arguments)]
pub fn escape_tail<W: PlanWorld>(
    world: &mut W,
    self_id: i32,
    input: &PlayerInput,
    others: &HashMap<i32, PlayerInput>,
    deadline: Option<(&dyn Clock, f64)>,
    after_hold_slot: &mut Option<W::SavedState>,
    lead: &[PlayerInput],
    extras: &[PlayerInput],
) -> Bounded<bool> {
    // The hybrid's extra escapes (task 3.5b) also know that a tee hanging from a hook on a tile is
    // settled; the TS-parity callers (no lead, no extras) keep the TS rule: standing, or alive at
    // the end of the settle ticks.
    let hang_ok = !lead.is_empty() || !extras.is_empty();
    save_into(world, after_hold_slot);
    let Some(me) = world.get_tee(self_id) else {
        return Bounded::Done(false);
    };

    let press_tick = i32::from(input.jump != 0);
    // Review round 3, F18 considered (and measured) reordering this list by "expected cost" --
    // permitted (not required) since `escape_exists` only ever returns whether *any* candidate
    // escape passes, never which one, so reordering cannot change its boolean result either here
    // or on the TS-parity path (`escape_exists` with `deadline: None` goes through this same
    // function). Tried: reversing `escapes()` so the hook+jump options (its most drastic, appended
    // last) go first, on the theory that grappling away is generally the most reliable way to
    // create separation. Measured effect (`phase_breakdown`, budget=4ms, n=1000/condition):
    // whole-map `shield_incomplete` improved (2 tees 127->93/1000, 6 tees 493->7/1000 -- though the
    // 6-tee number is dominated by the reserve-scaling fix below, not this reordering), but E-000
    // left-hall -- the dedicated shield-stress scenario -- got measurably *worse* at 2 tees (12->30/
    // 1000, step max 1170->809 but wall-clock cost per attempt rose enough that more decisions
    // timed out overall) -- in that tight corridor, a hook escape apparently often fails only
    // after running long (settling never reaches `standing()`, burning the full `SETTLE_TICKS`),
    // where the plain brake escape TS tries first usually resolves fast, either way. Since the
    // scenario this hurts is the one round 2's F14 built specifically to stress the shield, this
    // reordering is **not applied** -- `escapes()`'s original (TS) order is kept here too, same as
    // `safer_input`'s own loop (below) always keeps it for its different reason (its result is the
    // *specific* first alternative that passes, so reordering there would change behavior, not
    // just speed).
    for esc in lead
        .iter()
        .copied()
        .chain(escapes(me.vel.x, input.target_x, input.target_y))
        .chain(extras.iter().copied())
    {
        // Checked once per candidate unconditionally (not stride-gated): no point starting a
        // ~126-tick rollout we already know there's no time left for.
        if let Some((clock, deadline_ms)) = deadline
            && clock.now_ms() >= deadline_ms
        {
            return Bounded::TimedOut;
        }
        world.restore_state(after_hold_slot.as_ref().expect("saved after the hold"));
        let mut ok = true;
        let mut aborted = false;
        for t in 0..ESCAPE_TICKS {
            if deadline_hit(deadline, t) {
                aborted = true;
                break;
            }
            let now = if t == press_tick || esc.jump == 0 {
                esc
            } else {
                PlayerInput { jump: 0, ..esc }
            };
            world.set_input(self_id, now);
            apply_others(world, others);
            world.step();
            crate::prof::inc_step();
            let Some(cur) = world.get_tee(self_id) else {
                ok = false;
                break;
            };
            if !cur.alive || cur.frozen {
                ok = false;
                break;
            }
        }
        if aborted {
            return Bounded::TimedOut;
        }
        if ok {
            match settles_safe(world, self_id, &esc, others, deadline, hang_ok) {
                Bounded::Done(true) => return Bounded::Done(true),
                Bounded::Done(false) => {}
                Bounded::TimedOut => return Bounded::TimedOut,
            }
        }
    }
    Bounded::Done(false)
}

fn settles_safe<W: PlanWorld>(
    world: &mut W,
    self_id: i32,
    esc: &PlayerInput,
    others: &HashMap<i32, PlayerInput>,
    deadline: Option<(&dyn Clock, f64)>,
    hang_ok: bool,
) -> Bounded<bool> {
    let coast = PlayerInput {
        hook: esc.hook,
        target_x: esc.target_x,
        target_y: esc.target_y,
        ..empty_input()
    };
    let mut calm = 0;
    let calm_needed = hang_calm_ticks(world.gravity(self_id));
    for t in 0..SETTLE_TICKS {
        if deadline_hit(deadline, t) {
            return Bounded::TimedOut;
        }
        let Some(me) = world.get_tee(self_id) else {
            return Bounded::Done(false);
        };
        if !me.alive || me.frozen {
            return Bounded::Done(false);
        }
        if standing(world.collision(), me.pos.x, me.pos.y, me.vel.y) {
            return Bounded::Done(true);
        }
        // Hanging from our own hook on a tile (and still holding it) and at rest for a few ticks in a
        // row: settled, the rope keeps us up. (Tile hooks never time out; the risk is the swing before
        // the rest, so a grabbed hook alone proves nothing.)
        if hang_ok && coast.hook != 0 && me.hook_state == crate::types::HOOK_GRABBED && me.hooked_player < 0 {
            if ddai_libm::hypot(me.vel.x, me.vel.y) <= HANG_SPEED {
                calm += 1;
                if calm >= calm_needed {
                    return Bounded::Done(true);
                }
            } else {
                calm = 0;
            }
        } else {
            calm = 0;
        }
        world.set_input(self_id, coast);
        apply_others(world, others);
        world.step();
        crate::prof::inc_step();
    }
    Bounded::Done(match world.get_tee(self_id) {
        Some(me) => me.alive && !me.frozen,
        None => false,
    })
}

fn standing(col: &impl PlanCollision, x: f64, y: f64, vy: f64) -> bool {
    js::abs(vy) < 0.5 && (col.is_solid(x - HALF + 1.0, y + HALF + 2.0) || col.is_solid(x + HALF - 1.0, y + HALF + 2.0))
}

/// `saferInput(world, selfId, input, holdTicks, others, sentAim)` (`shield.ts:95-111`). Unchanged
/// from round 1 -- see [`escape_exists`]'s doc comment; the same "provably never times out when
/// `deadline: None`" argument applies here via [`safer_input_bounded`].
pub fn safer_input<W: PlanWorld>(
    world: &mut W,
    self_id: i32,
    input: &PlayerInput,
    hold_ticks: i32,
    others: &HashMap<i32, PlayerInput>,
    sent_aim: Option<&PlayerInput>,
) -> Option<PlayerInput> {
    match safer_input_bounded(world, self_id, input, hold_ticks, others, sent_aim, None) {
        Bounded::Done(v) => v,
        Bounded::TimedOut => unreachable!("safer_input_bounded(deadline: None) never times out"),
    }
}

/// Review round 2, F12: `decide_production`'s deadline-aware counterpart to [`safer_input`].
#[allow(clippy::too_many_arguments)]
pub fn safer_input_bounded<W: PlanWorld>(
    world: &mut W,
    self_id: i32,
    input: &PlayerInput,
    hold_ticks: i32,
    others: &HashMap<i32, PlayerInput>,
    sent_aim: Option<&PlayerInput>,
    deadline: Option<(&dyn Clock, f64)>,
) -> Bounded<Option<PlayerInput>> {
    safer_input_buffered(
        world,
        self_id,
        input,
        hold_ticks,
        others,
        sent_aim,
        deadline,
        &mut ShieldBuffers::default(),
    )
}

/// [`safer_input_bounded`] with caller-owned snapshot buffers (see [`ShieldBuffers`]).
#[allow(clippy::too_many_arguments)]
pub fn safer_input_buffered<W: PlanWorld>(
    world: &mut W,
    self_id: i32,
    input: &PlayerInput,
    hold_ticks: i32,
    others: &HashMap<i32, PlayerInput>,
    sent_aim: Option<&PlayerInput>,
    deadline: Option<(&dyn Clock, f64)>,
    bufs: &mut ShieldBuffers<W>,
) -> Bounded<Option<PlayerInput>> {
    safer_input_ext(world, self_id, input, hold_ticks, others, sent_aim, deadline, bufs, &[])
}

/// [`safer_input_buffered`] whose escape checks also try `extras` (hook escapes at anchors). The
/// alternatives it offers are still only the standard ones: an anchor hook is an *escape* the tee can
/// make after the held input, not a replacement for the held input.
#[allow(clippy::too_many_arguments)]
pub fn safer_input_ext<W: PlanWorld>(
    world: &mut W,
    self_id: i32,
    input: &PlayerInput,
    hold_ticks: i32,
    others: &HashMap<i32, PlayerInput>,
    sent_aim: Option<&PlayerInput>,
    deadline: Option<(&dyn Clock, f64)>,
    bufs: &mut ShieldBuffers<W>,
    extras: &[PlayerInput],
) -> Bounded<Option<PlayerInput>> {
    let Some(me) = world.get_tee(self_id) else {
        return Bounded::Done(None);
    };
    let base = match sent_aim {
        Some(aim) if aim.target_x != 0.0 || aim.target_y != 0.0 => aim,
        _ => input,
    };
    let from = js::atan2(base.target_y, base.target_x);
    for alt in escapes(me.vel.x, input.target_x, input.target_y) {
        if let Some((clock, deadline_ms)) = deadline
            && clock.now_ms() >= deadline_ms
        {
            return Bounded::TimedOut;
        }
        let mut d = js::atan2(alt.target_y, alt.target_x) - from;
        while d > js::PI {
            d -= 2.0 * js::PI;
        }
        while d < -js::PI {
            d += 2.0 * js::PI;
        }
        let a = from + js::max(-MAX_TURN_RAD, js::min(MAX_TURN_RAD, d));
        let cand = PlayerInput {
            direction: alt.direction,
            jump: alt.jump,
            hook: alt.hook,
            target_x: js::round(js::cos(a) * 300.0),
            target_y: js::round(js::sin(a) * 300.0),
            ..*input
        };
        match escape_exists_ext(
            world,
            self_id,
            &cand,
            hold_ticks,
            others,
            deadline,
            bufs,
            &mut ShieldOpts { first: None, extras },
        ) {
            Bounded::Done(true) => return Bounded::Done(Some(cand)),
            Bounded::Done(false) => {}
            Bounded::TimedOut => return Bounded::TimedOut,
        }
    }
    Bounded::Done(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hang_needs_more_calm_ticks_than_a_free_fall_apex_lasts() {
        // Stock gravity 0.5: the apex of an arc is calm (|v| <= 1.5) for 2 * 1.5 / 0.5 = 6 (up to 7) ticks.
        assert_eq!(hang_calm_ticks(0.5), 8);
        // Weaker gravity lengthens the apex, so the requirement grows; no gravity never settles early.
        assert_eq!(hang_calm_ticks(0.25), 14);
        assert_eq!(hang_calm_ticks(0.0), SETTLE_TICKS);
        assert_eq!(hang_calm_ticks(0.001), SETTLE_TICKS);
        // Stronger gravity shortens it but never below the +2 margin.
        assert_eq!(hang_calm_ticks(5.0), 3);
    }

    #[test]
    fn escapes_without_braking_gives_3_options() {
        let out = escapes(0.1, 0.0, -1.0);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn escapes_with_braking_gives_6_options() {
        let out = escapes(5.0, 0.0, -1.0);
        assert_eq!(out.len(), 6);
        assert_eq!(out[2].direction, -1); // sign(vx) > 0 -> brakeDir = -1
    }
}
