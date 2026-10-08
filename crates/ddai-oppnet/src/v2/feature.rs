//! The v2 input vector and targets (see [`crate::v2`] for what changed against [`crate::feature`]).
//!
//! **Input** ([`INPUT_DIM`] numbers): the last [`K_HIST`] frames of the pair ([`STRIDE`] ticks apart, [`FD`] numbers each), the geometry rays around the opponent
//! and around us, our in-flight inputs, the window length (one-hot `0..=`[`MAX_LAG`]) and, per window tick, the opponent's input when it is already known
//! (a pre-input; zeros otherwise, with a flag).
//!
//! **Targets**: for each of the [`HORIZON`] ticks from the snapshot tick `T` (tick `k` = the input applied in the step `T + k -> T + k + 1`): direction (3 classes),
//! jump, hook, a fresh fire press, and the change of the aim angle against the angle the snapshot shows.

use ddai_jsmath as js;

use crate::frame::{InputRec, N_RAYS, TeeFrame};

pub use crate::feature::{aim_of, inflight_features, is_press, wrap_angle};

/// Bumped when an input or output number changes its meaning.
pub const FEATURE_VERSION: u32 = 2;
/// Ticks between two history frames (the snapshot rate of the live bot).
pub const STRIDE: usize = 2;
/// History frames. Two: the abundant arena data and the clips both showed no use for more (`docs/research/opponent-predictor-v2.md`: one frame trained to the same
/// or a lower validation loss than eight, a wider network to a higher one).
pub const K_HIST: usize = 2;
/// Window ticks predicted.
pub const HORIZON: usize = 4;
/// Numbers per history frame: the 29 of [`crate::feature::frame_features`] plus 4.
pub const FD: usize = crate::feature::FD + 4;
/// Our in-flight inputs encoded.
pub const IF_SLOTS: usize = 4;
pub const IF_DIM: usize = crate::feature::IF_DIM;
/// The window length one-hot covers `0..=MAX_LAG`.
pub const MAX_LAG: usize = 4;
pub const LAG_SLOTS: usize = MAX_LAG + 1;
/// Numbers per known tick: flag, direction, jump, hook, fire level, aim cos, aim sin.
pub const KNOWN_DIM: usize = 7;
pub const INPUT_DIM: usize = K_HIST * FD + 2 * N_RAYS + IF_SLOTS * IF_DIM + LAG_SLOTS + HORIZON * KNOWN_DIM;
/// Outputs per tick: direction (3 logits), jump, hook, press (logits), aim change (radians).
pub const HEAD_DIM: usize = 7;
pub const OUT_DIM: usize = HORIZON * HEAD_DIM;

fn clip(x: f32, lim: f32) -> f32 {
    x.clamp(-lim, lim)
}

/// The features of one frame of the pair (`opp` is the tee being predicted, `me` the other).
///
/// `0..29` as [`crate::feature::frame_features`]; `29` the age of the opponent's hook (ticks / 50, at most 2); `30` how long ago we used our weapon (ticks / 30, at most 1);
/// `31, 32` the cosine and sine of the angle between our aim and the line to the opponent.
pub fn frame_features(me: &TeeFrame, opp: &TeeFrame, out: &mut [f32; FD]) {
    let mut base = [0.0f32; crate::feature::FD];
    crate::feature::frame_features(me, opp, &mut base);
    out[..crate::feature::FD].copy_from_slice(&base);
    let to_opp = js::atan2(f64::from(opp.pos[1] - me.pos[1]), f64::from(opp.pos[0] - me.pos[0]));
    let rel = f64::from(me.angle) - to_opp;
    out[crate::feature::FD] = clip(f32::from(opp.hook_tick) / 50.0, 2.0);
    out[crate::feature::FD + 1] = f32::from(me.attack_age.min(30)) / 30.0;
    out[crate::feature::FD + 2] = js::cos(rel) as f32;
    out[crate::feature::FD + 3] = js::sin(rel) as f32;
}

/// An opponent input that is already known for one window tick (a server pre-input).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KnownTick {
    pub direction: i8,
    pub jump: bool,
    pub hook: bool,
    /// The fire counter is odd.
    pub fire_held: bool,
    pub target_x: i16,
    pub target_y: i16,
}

impl KnownTick {
    pub fn from_rec(i: &InputRec) -> KnownTick {
        KnownTick {
            direction: i.direction,
            jump: i.jump,
            hook: i.hook,
            fire_held: i.fire & 1 != 0,
            target_x: i.target_x,
            target_y: i.target_y,
        }
    }

    fn features(&self, out: &mut [f32]) {
        let (tx, ty) = (f32::from(self.target_x), f32::from(self.target_y));
        let n = (tx * tx + ty * ty).sqrt();
        let (c, s) = if n > 0.0 { (tx / n, ty / n) } else { (0.0, 0.0) };
        out.copy_from_slice(&[
            1.0,
            f32::from(self.direction),
            f32::from(u8::from(self.jump)),
            f32::from(u8::from(self.hook)),
            f32::from(u8::from(self.fire_held)),
            c,
            s,
        ]);
    }
}

/// Builds the input vector. `hist(j)` is the feature vector of the frame `j * STRIDE` ticks before the snapshot (`j = 0` the snapshot's own);
/// `rays_opp` / `rays_me` the geometry around each tee at the snapshot; `inflight` our in-flight inputs' features (at most [`IF_SLOTS`]); `lag` the window length;
/// `known[k]` the opponent's input of window tick `k` when it is known.
pub fn assemble<'a>(
    x: &mut [f32; INPUT_DIM],
    hist: impl Fn(usize) -> &'a [f32; FD],
    rays_opp: &[f32; N_RAYS],
    rays_me: &[f32; N_RAYS],
    inflight: &[[f32; IF_DIM]],
    lag: usize,
    known: &[Option<KnownTick>],
) {
    x.fill(0.0);
    for j in 0..K_HIST {
        x[j * FD..(j + 1) * FD].copy_from_slice(hist(j));
    }
    let mut o = K_HIST * FD;
    x[o..o + N_RAYS].copy_from_slice(rays_opp);
    o += N_RAYS;
    x[o..o + N_RAYS].copy_from_slice(rays_me);
    o += N_RAYS;
    for (k, f) in inflight.iter().take(IF_SLOTS).enumerate() {
        x[o + k * IF_DIM..o + (k + 1) * IF_DIM].copy_from_slice(f);
    }
    o += IF_SLOTS * IF_DIM;
    x[o + lag.min(MAX_LAG)] = 1.0;
    o += LAG_SLOTS;
    for (k, kn) in known.iter().take(HORIZON).enumerate() {
        if let Some(kn) = kn {
            kn.features(&mut x[o + k * KNOWN_DIM..o + (k + 1) * KNOWN_DIM]);
        }
    }
}

/// The targets of one sample, with a validity mask per head (a live snapshot shows some heads at some ticks only, see [`crate::clipdata`]).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Label {
    /// Bit `k`: tick `k` has a label for this head.
    pub v_dir: u8,
    pub v_jump: u8,
    pub v_hook: u8,
    pub v_press: u8,
    pub v_aim: u8,
    /// Direction class `0, 1, 2` for `-1, 0, 1`.
    pub dir: [u8; HORIZON],
    /// Bit `k`: jump / hook / a fresh fire press at tick `k`.
    pub jump: u8,
    pub hook: u8,
    pub press: u8,
    /// Applied aim angle minus the angle the snapshot shows (radians, wrapped).
    pub aim_delta: [f32; HORIZON],
}

impl Label {
    /// Marks the ticks below `n` as not to be learned (their input is known).
    pub fn mask_known(&mut self, n: usize) {
        let keep = !(((1u16 << n.min(8)) - 1) as u8);
        self.v_dir &= keep;
        self.v_jump &= keep;
        self.v_hook &= keep;
        self.v_press &= keep;
        self.v_aim &= keep;
    }
}

/// Adds the arena's target of window tick `k` (all heads) to `label`: `cur` is the input applied at `T + k`, `prev` the one applied at `T + k - 1`.
pub fn label_tick(label: &mut Label, k: usize, prev: &InputRec, cur: &InputRec, base_angle: f64) {
    let bit = 1u8 << k;
    label.v_dir |= bit;
    label.v_jump |= bit;
    label.v_hook |= bit;
    label.v_press |= bit;
    label.v_aim |= bit;
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
        assert_eq!(FD, 33);
        assert_eq!(INPUT_DIM, 2 * 33 + 32 + 24 + 5 + 28);
        assert_eq!(OUT_DIM, 28);
    }

    #[test]
    fn assemble_places_every_block_and_the_known_ticks() {
        let frames: Vec<[f32; FD]> = (0..K_HIST).map(|j| [j as f32 + 1.0; FD]).collect();
        let (ro, rm) = ([0.5f32; N_RAYS], [0.25f32; N_RAYS]);
        let inf = [[2.0f32; IF_DIM]; 2];
        let known = [
            Some(KnownTick {
                direction: -1,
                jump: true,
                hook: false,
                fire_held: true,
                target_x: 0,
                target_y: 100,
            }),
            None,
        ];
        let mut x = [9.0f32; INPUT_DIM];
        assemble(&mut x, |j| &frames[j], &ro, &rm, &inf, 2, &known);
        assert_eq!(x[0], 1.0);
        assert_eq!(x[(K_HIST - 1) * FD], K_HIST as f32);
        let r = K_HIST * FD;
        assert_eq!((x[r], x[r + N_RAYS]), (0.5, 0.25));
        let i = r + 2 * N_RAYS;
        assert_eq!((x[i], x[i + 2 * IF_DIM - 1], x[i + 2 * IF_DIM]), (2.0, 2.0, 0.0));
        let l = i + IF_SLOTS * IF_DIM;
        assert_eq!(x[l + 2], 1.0);
        assert_eq!(x[l..l + LAG_SLOTS].iter().sum::<f32>(), 1.0);
        let k = l + LAG_SLOTS;
        assert_eq!(&x[k..k + KNOWN_DIM], &[1.0, -1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        assert!(
            x[k + KNOWN_DIM..].iter().all(|&v| v == 0.0),
            "an unknown tick is all zeros"
        );
    }

    #[test]
    fn the_extra_frame_features_read_the_hook_age_the_cooldown_and_our_aim() {
        let me = TeeFrame {
            alive: true,
            pos: [0.0, 0.0],
            angle: 0.0,
            attack_age: 15,
            ..TeeFrame::default()
        };
        let opp = TeeFrame {
            alive: true,
            pos: [100.0, 0.0],
            hook_tick: 25,
            ..TeeFrame::default()
        };
        let mut f = [0.0f32; FD];
        frame_features(&me, &opp, &mut f);
        let base = crate::feature::FD;
        assert!((f[base] - 0.5).abs() < 1e-6, "hook age 25 ticks");
        assert!((f[base + 1] - 0.5).abs() < 1e-6, "cooldown 15 of 30");
        assert!(
            (f[base + 2] - 1.0).abs() < 1e-6 && f[base + 3].abs() < 1e-6,
            "we aim straight at him"
        );
        let mut v1 = [0.0f32; crate::feature::FD];
        crate::feature::frame_features(&me, &opp, &mut v1);
        assert_eq!(&f[..base], &v1[..], "the first 29 are the v1 features");
    }

    #[test]
    fn masking_known_ticks_drops_their_labels() {
        let mut l = Label::default();
        let cur = InputRec {
            fire: 1,
            ..InputRec::default()
        };
        for k in 0..4 {
            label_tick(&mut l, k, &InputRec::default(), &cur, 0.0);
        }
        assert_eq!(l.v_press, 0b1111);
        l.mask_known(2);
        assert_eq!((l.v_dir, l.v_press, l.v_aim), (0b1100, 0b1100, 0b1100));
        l.mask_known(0);
        assert_eq!(l.v_dir, 0b1100);
    }
}
