//! The predictor's input vector and its targets.
//!
//! **Input** ([`INPUT_DIM`] numbers): the last [`K_HIST`] frames of the pair (opponent-centred, see [`frame_features`]),
//! [`STRIDE`] ticks apart (the decision cadence: the brain sees a new frame at every decision), the
//! geometry rays around the opponent now, our own in-flight inputs for the window, and the window length.
//!
//! **Targets** ([`HORIZON`] ticks from the snapshot tick `T`, tick `k` = the input applied in the step
//! that goes from tick `T + k` to `T + k + 1`): per tick the direction (3 classes), jump, hook, a fresh fire press, and the change of the
//! aim angle against the angle the snapshot shows.
//!
//! All the trigonometry goes through `ddai-jsmath` (V8's fdlibm): the same bits on any machine.

use ddai_jsmath as js;

use crate::frame::{InputRec, N_RAYS, TeeFrame};

/// Ticks between two history frames: the arena's decision cadence (`Rules::decide_every`) and the live bot's snapshot rate (25 Hz).
pub const STRIDE: usize = 2;
/// History frames in the input (the newest is the snapshot's own).
pub const K_HIST: usize = 8;
/// Ticks predicted.
pub const HORIZON: usize = 8;
/// Numbers per history frame.
pub const FD: usize = 29;
/// Our in-flight inputs encoded (the longest window the model is asked about).
pub const IF_SLOTS: usize = 8;
/// Numbers per in-flight input.
pub const IF_DIM: usize = 6;
/// One-hot of the window length, `0..=8`.
pub const LAG_SLOTS: usize = 9;
pub const INPUT_DIM: usize = K_HIST * FD + N_RAYS + IF_SLOTS * IF_DIM + LAG_SLOTS;
/// Outputs per predicted tick: direction (3 logits), jump, hook, press (logits), aim change (radians).
pub const HEAD_DIM: usize = 7;
pub const OUT_DIM: usize = HORIZON * HEAD_DIM;

fn clip(x: f32, lim: f32) -> f32 {
    x.clamp(-lim, lim)
}

/// Wraps an angle to `(-pi, pi]`.
pub fn wrap_angle(a: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut r = a % two_pi;
    if r > std::f64::consts::PI {
        r -= two_pi;
    } else if r <= -std::f64::consts::PI {
        r += two_pi;
    }
    r
}

/// The features of one frame of the pair (`opp` is the tee being predicted, `me` the other).
///
/// `0..15` the opponent (velocity, direction, aim, hook state, jump bits, freeze, last weapon use, grounded); `15..20` the geometry between
/// the two (offset, distance, where the opponent aims relative to the line to us); `20..29` us (velocity, direction, hook, freeze, jumps, grounded).
pub fn frame_features(me: &TeeFrame, opp: &TeeFrame, out: &mut [f32; FD]) {
    let a = f64::from(opp.angle);
    let (dx, dy) = (me.pos[0] - opp.pos[0], me.pos[1] - opp.pos[1]);
    let to_us = js::atan2(f64::from(dy), f64::from(dx));
    let rel = a - to_us;
    let hook_len = if matches!(opp.hook_state, 1..=5) {
        ((opp.hook_pos[0] - opp.pos[0]).powi(2) + (opp.hook_pos[1] - opp.pos[1]).powi(2)).sqrt() / 400.0
    } else {
        0.0
    };
    let f = |b: bool| if b { 1.0 } else { 0.0 };
    *out = [
        clip(opp.vel[0] / 10.0, 4.0),
        clip(opp.vel[1] / 10.0, 4.0),
        f32::from(opp.direction),
        js::cos(a) as f32,
        js::sin(a) as f32,
        f(opp.hook_state == 4),
        f(opp.hook_state == 5),
        f(matches!(opp.hook_state, 1..=3)),
        f(opp.hook_on_other),
        clip(hook_len, 2.0),
        f(opp.jumped & 1 != 0),
        f(opp.jumped & 2 != 0),
        f(opp.freeze_left > 0),
        f32::from(opp.attack_age.min(30)) / 30.0,
        f(opp.grounded),
        clip(dx / 320.0, 3.0),
        clip(dy / 320.0, 3.0),
        clip((dx * dx + dy * dy).sqrt() / 320.0, 3.0),
        js::cos(rel) as f32,
        js::sin(rel) as f32,
        clip(me.vel[0] / 10.0, 4.0),
        clip(me.vel[1] / 10.0, 4.0),
        f32::from(me.direction),
        f(me.hook_state == 4),
        f(me.hook_state == 5),
        f(me.hook_on_other),
        f(me.freeze_left > 0),
        f(me.jumped & 2 != 0),
        f(me.grounded),
    ];
}

/// The features of one of our in-flight inputs: direction, jump, hook, fire level, aim as a unit vector.
pub fn inflight_features(i: &InputRec, out: &mut [f32; IF_DIM]) {
    let (tx, ty) = (f32::from(i.target_x), f32::from(i.target_y));
    let n = (tx * tx + ty * ty).sqrt();
    let (c, s) = if n > 0.0 { (tx / n, ty / n) } else { (0.0, -1.0) };
    *out = [
        f32::from(i.direction),
        f32::from(u8::from(i.jump)),
        f32::from(u8::from(i.hook)),
        f32::from(u8::from(i.fire & 1 != 0)),
        c,
        s,
    ];
}

/// Builds the input vector. `hist(j)` is the feature vector of the frame `j * STRIDE` ticks before the snapshot (`j = 0` the snapshot's own);
/// `inflight` are our in-flight inputs' features (at most [`IF_SLOTS`]); the window length is `lag`.
pub fn assemble<'a>(
    x: &mut [f32; INPUT_DIM],
    hist: impl Fn(usize) -> &'a [f32; FD],
    rays: &[f32; N_RAYS],
    inflight: &[[f32; IF_DIM]],
    lag: usize,
) {
    x.fill(0.0);
    for j in 0..K_HIST {
        x[j * FD..(j + 1) * FD].copy_from_slice(hist(j));
    }
    let mut o = K_HIST * FD;
    x[o..o + N_RAYS].copy_from_slice(rays);
    o += N_RAYS;
    for (k, f) in inflight.iter().take(IF_SLOTS).enumerate() {
        x[o + k * IF_DIM..o + (k + 1) * IF_DIM].copy_from_slice(f);
    }
    o += IF_SLOTS * IF_DIM;
    x[o + lag.min(LAG_SLOTS - 1)] = 1.0;
}

/// The targets of one sample.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Label {
    /// Bit `k`: tick `k` has a label (the game was still going and the opponent was not frozen).
    pub valid: u8,
    /// Direction class `0, 1, 2` for `-1, 0, 1`.
    pub dir: [u8; HORIZON],
    /// Bit `k`: jump / hook / a fresh fire press at tick `k`.
    pub jump: u8,
    pub hook: u8,
    pub press: u8,
    /// Applied aim angle minus the angle the snapshot shows (radians, wrapped).
    pub aim_delta: [f32; HORIZON],
}

/// The aim angle of an input (`atan2(target_y, target_x)`); `None` for the null aim.
pub fn aim_of(i: &InputRec) -> Option<f64> {
    (i.target_x != 0 || i.target_y != 0).then(|| js::atan2(f64::from(i.target_y), f64::from(i.target_x)))
}

/// Whether the step with input `cur` after the step with input `prev` made a fresh fire press (the counter moved on to an odd value).
pub fn is_press(prev: &InputRec, cur: &InputRec) -> bool {
    cur.fire != prev.fire && cur.fire & 1 != 0
}

/// Adds the target of window tick `k` to `label`: `cur` is the input applied at `T + k`, `prev` the one applied at `T + k - 1`, `base_angle` the aim the snapshot
/// shows (radians).
pub fn label_tick(label: &mut Label, k: usize, prev: &InputRec, cur: &InputRec, base_angle: f64) {
    let bit = 1u8 << k;
    label.valid |= bit;
    label.dir[k] = (i32::from(cur.direction) + 1).clamp(0, 2) as u8;
    if cur.jump {
        label.jump |= bit;
    }
    if cur.hook {
        label.hook |= bit;
    }
    if is_press(prev, cur) {
        label.press |= bit;
    }
    label.aim_delta[k] = aim_of(cur).map_or(0.0, |a| wrap_angle(a - base_angle) as f32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_add_up() {
        assert_eq!(INPUT_DIM, 8 * 29 + 16 + 48 + 9);
        assert_eq!(OUT_DIM, 56);
    }

    #[test]
    fn wrap_angle_lands_in_the_half_open_interval() {
        for a in [
            -7.0,
            -3.2,
            -std::f64::consts::PI,
            0.0,
            3.0,
            std::f64::consts::PI,
            4.0,
            9.5,
        ] {
            let w = wrap_angle(a);
            assert!(
                w > -std::f64::consts::PI - 1e-12 && w <= std::f64::consts::PI + 1e-12,
                "{a} -> {w}"
            );
            assert!((js::cos(w) - js::cos(a)).abs() < 1e-9 && (js::sin(w) - js::sin(a)).abs() < 1e-9);
        }
    }

    #[test]
    fn a_press_is_a_move_of_the_counter_to_an_odd_value() {
        let f = |c: i32| InputRec {
            fire: c,
            ..Default::default()
        };
        assert!(is_press(&f(0), &f(1)));
        assert!(is_press(&f(1), &f(3)), "release and press in one step");
        assert!(!is_press(&f(1), &f(1)), "held");
        assert!(!is_press(&f(1), &f(2)), "release");
        assert!(!is_press(&f(2), &f(2)));
    }

    #[test]
    fn labels_record_levels_presses_and_the_aim_change() {
        let prev = InputRec {
            fire: 0,
            ..Default::default()
        };
        let cur = InputRec {
            direction: -1,
            jump: true,
            hook: false,
            fire: 1,
            target_x: 0,
            target_y: 300,
        };
        let mut l = Label::default();
        label_tick(&mut l, 2, &prev, &cur, 0.0);
        assert_eq!(l.valid, 0b100);
        assert_eq!((l.dir[2], l.jump, l.hook, l.press), (0, 0b100, 0, 0b100));
        assert!((f64::from(l.aim_delta[2]) - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn assemble_places_every_block_and_the_lag_one_hot() {
        let frames: Vec<[f32; FD]> = (0..K_HIST).map(|j| [j as f32 + 1.0; FD]).collect();
        let rays = [0.5f32; N_RAYS];
        let inf = [[2.0f32; IF_DIM]; 3];
        let mut x = [9.0f32; INPUT_DIM];
        assemble(&mut x, |j| &frames[j], &rays, &inf, 3);
        assert_eq!(x[0], 1.0);
        assert_eq!(x[(K_HIST - 1) * FD], K_HIST as f32);
        let r = K_HIST * FD;
        assert_eq!(x[r], 0.5);
        let i = r + N_RAYS;
        assert_eq!(
            (x[i], x[i + 3 * IF_DIM - 1], x[i + 3 * IF_DIM]),
            (2.0, 2.0, 0.0),
            "three inputs, the rest zero"
        );
        let l = i + IF_SLOTS * IF_DIM;
        assert_eq!(x[l + 3], 1.0);
        assert_eq!(x[l..].iter().sum::<f32>(), 1.0);
    }
}
