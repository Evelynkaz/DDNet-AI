//! `decodeAction` (`src/env/action.ts:19-76`) — turns the planner's `raw` 10-wide action encoding
//! into a `PlayerInput`, with aim smoothing/flick-turn and the fire press-counter convention.
//! `ACTION_SIZE`/other constants match TS exactly, cited by name.

use crate::trig;
use crate::types::{PlayerInput, WEAPON_GUN, WEAPON_HAMMER};
use ddai_jsmath as js;

pub const ACTION_SIZE: usize = 10;

pub const MAX_AIM_TURN_RAD: f64 = std::f64::consts::PI / 2.0;
pub const FLICK_TURN_RAD: f64 = std::f64::consts::PI;
pub const FLICK_THRESHOLD_RAD: f64 = 2.4;
pub const AIM_SMOOTHING: f64 = 0.75;
pub const AIM_RADIUS: f64 = 300.0;
pub const DIRECTION_SWITCH_MARGIN: f64 = 0.15;
pub const AIR_JUMP_PRESS_MARGIN: f64 = 0.4;

/// `decodeAction(raw, prev, out?, airborne = false)` (`action.ts:19-76`). `raw` must have exactly
/// [`ACTION_SIZE`] elements (indices `0..2` direction logits, `3` jump, `4` hook, `5` fire, `6..7`
/// weapon logits, `8..9` aim vector) — the planner always builds it that way (`stepToInput`/
/// `executedAim`).
pub fn decode_action(raw: &[f64; ACTION_SIZE], prev: &PlayerInput, airborne: bool) -> PlayerInput {
    let mut dst = crate::types::empty_input();

    let mut dir_idx = 0usize;
    if raw[1] > raw[dir_idx] {
        dir_idx = 1;
    }
    if raw[2] > raw[dir_idx] {
        dir_idx = 2;
    }
    let held_idx = (prev.direction + 1) as usize;
    if dir_idx != held_idx && raw[dir_idx] - raw[held_idx] < DIRECTION_SWITCH_MARGIN {
        dir_idx = held_idx;
    }
    dst.direction = dir_idx as i32 - 1;

    let jump_wanted = raw[3];
    dst.jump = if prev.jump != 0 {
        i32::from(jump_wanted > 0.0)
    } else {
        i32::from(jump_wanted > if airborne { AIR_JUMP_PRESS_MARGIN } else { 0.0 })
    };
    dst.hook = i32::from(raw[4] > 0.0);

    let held = (prev.fire & 1) != 0;
    dst.fire = if raw[5] > 0.0 {
        if held { prev.fire + 2 } else { prev.fire + 1 }
    } else if held {
        prev.fire + 1
    } else {
        prev.fire
    };

    let wanted_weapon_id = if raw[7] > raw[6] { WEAPON_GUN } else { WEAPON_HAMMER };
    dst.wanted_weapon = wanted_weapon_id + 1;
    dst.next_weapon = 0;
    dst.prev_weapon = 0;
    dst.player_flags = 0;

    let ax = if raw[8].is_finite() { raw[8] } else { 0.0 };
    let ay = if raw[9].is_finite() { raw[9] } else { 0.0 };
    let len = js::sqrt(ax * ax + ay * ay);

    let wanted = if len < 1e-6 { 0.0 } else { trig::atan2(ay, ax) };
    let prev_len = js::sqrt(prev.target_x * prev.target_x + prev.target_y * prev.target_y);
    let current = if prev_len < 1e-6 {
        wanted
    } else {
        trig::atan2(prev.target_y, prev.target_x)
    };

    let mut delta = wanted - current;
    while delta > js::PI {
        delta -= 2.0 * js::PI;
    }
    while delta < -js::PI {
        delta += 2.0 * js::PI;
    }

    let cap = if js::abs(delta) >= FLICK_THRESHOLD_RAD {
        FLICK_TURN_RAD
    } else {
        MAX_AIM_TURN_RAD
    };
    delta *= AIM_SMOOTHING;
    if delta > cap {
        delta = cap;
    } else if delta < -cap {
        delta = -cap;
    }
    let angle = current + delta;

    let mut tx = js::round(trig::cos(angle) * AIM_RADIUS);
    let mut ty = js::round(trig::sin(angle) * AIM_RADIUS);
    if tx == 0.0 && ty == 0.0 {
        tx = AIM_RADIUS;
        ty = 0.0;
    }
    dst.target_x = tx;
    dst.target_y = ty;

    dst
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::empty_input;

    #[test]
    fn holds_direction_within_switch_margin() {
        // `raw[2]` (right) barely beats `raw[1]` (held = none), under the 0.15 margin -> stays.
        let mut raw = [0.0; ACTION_SIZE];
        raw[1] = 0.5; // held (prev.direction = 0 -> heldIdx = 1)
        raw[2] = 0.6; // best, but 0.6 - 0.5 = 0.1 < 0.15
        raw[8] = 1.0;
        let prev = empty_input();
        let out = decode_action(&raw, &prev, false);
        assert_eq!(out.direction, 0);
    }

    #[test]
    fn switches_direction_past_the_margin() {
        let mut raw = [0.0; ACTION_SIZE];
        raw[1] = 0.5;
        raw[2] = 0.7; // 0.2 >= 0.15 -> switches
        raw[8] = 1.0;
        let prev = empty_input();
        let out = decode_action(&raw, &prev, false);
        assert_eq!(out.direction, 1);
    }

    #[test]
    fn zero_aim_vector_keeps_the_previous_target_direction() {
        // `raw[8..10] = (0, 0)` -> `len < 1e-6` -> `wanted = 0` ("straight right"); smoothed
        // 0.75 of the way from `prev`'s own angle (straight up, `emptyInput()`'s default) toward
        // it, capped at `MAX_AIM_TURN_RAD` -- never snaps straight to `wanted` in one decision.
        let raw = [0.0; ACTION_SIZE];
        let prev = empty_input();
        let out = decode_action(&raw, &prev, false);
        // Aim moved *toward* 0 rad from -pi/2, but did not reach it in one smoothing step.
        let prev_angle = js::atan2(prev.target_y, prev.target_x);
        let out_angle = js::atan2(out.target_y, out.target_x);
        assert!(
            js::abs(out_angle) < js::abs(prev_angle),
            "expected the aim to move toward 0 rad"
        );
        assert!(
            out_angle != 0.0,
            "one smoothing step should not reach the target angle exactly"
        );
        // The aim vector is always length `AIM_RADIUS` from the origin (rounded), so `(0, 0)`
        // is unreachable in practice -- the `(tx, ty) == (0, 0) -> (AIM_RADIUS, 0)` fallback in
        // `decode_action` exists only as a defensive guard against a `NaN`/degenerate angle.
        assert_ne!((out.target_x, out.target_y), (0.0, 0.0));
    }

    #[test]
    fn already_aimed_at_the_wanted_direction_stays_there() {
        let mut raw = [0.0; ACTION_SIZE];
        raw[8] = 1.0; // wanted = atan2(0, 1) = 0 rad, straight right.
        let mut prev = empty_input();
        prev.target_x = AIM_RADIUS;
        prev.target_y = 0.0;
        let out = decode_action(&raw, &prev, false);
        assert_eq!((out.target_x, out.target_y), (AIM_RADIUS, 0.0));
    }

    #[test]
    fn fire_counter_increments_on_new_press_and_holds_odd_while_held() {
        let mut raw = [0.0; ACTION_SIZE];
        raw[8] = 1.0;
        raw[5] = 1.0; // want fire
        let prev = empty_input(); // prev.fire = 0 (even -> not held)
        let out = decode_action(&raw, &prev, false);
        assert_eq!(out.fire, 1); // new press
        let out2 = decode_action(&raw, &out, false);
        assert_eq!(out2.fire, 3); // still held -> +2
    }
}
