//! The v2 model file ([`Bundle`]) and the live predictor ([`Predictor`]): the network behind the hybrid's `WindowModel` slot, building its input from a world with
//! the same functions the trainer uses ([`super::feature`]).
//!
//! One call per decision: it records the frame of the pair (the history the network reads) and, when the window is not empty, builds the input, runs the network and
//! decodes the first `lag` ticks. No allocation after construction.

use std::path::Path;

use ddai_planner::hybrid::window::{PredictedInput, WindowCtx, WindowModel};
use serde::{Deserialize, Serialize};

use super::feature::{
    FD, FEATURE_VERSION, HEAD_DIM, HORIZON, IF_DIM, IF_SLOTS, INPUT_DIM, K_HIST, KNOWN_DIM, KnownTick, MAX_LAG,
    OUT_DIM, STRIDE, assemble, frame_features, inflight_features, wrap_angle,
};
use crate::blob::{read_blob, write_blob};
use crate::frame::{InputRec, N_RAYS, TeeFrame, rays};
use crate::net::{Mlp, Scratch};

pub const BUNDLE_VERSION: u32 = 2;

/// How the logits become an input: the thresholds are tuned on held-out data, not fixed at the symmetric 0.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Decode {
    /// A jump is predicted when its logit is above this.
    pub jump: f32,
    pub hook: f32,
    /// The hook differs from what the snapshot shows only when its logit is more than this away from `hook` (`0`: the logit alone decides).
    pub hook_margin: f32,
    /// A fire press is predicted when its logit is above this.
    pub press: f32,
    /// The direction differs from what the snapshot shows only when its logit beats the shown direction's by this much (`0`: argmax).
    pub dir_margin: f32,
}

impl Default for Decode {
    fn default() -> Self {
        Decode {
            jump: 0.0,
            hook: 0.0,
            hook_margin: 0.0,
            press: 0.0,
            dir_margin: 0.0,
        }
    }
}

/// The hook level of a hook logit: above `d.hook` plus the margin it is out, below `d.hook` minus the margin it is in, in between whatever the snapshot shows.
pub fn decode_hook(logit: f32, d: &Decode, hold_hook: bool) -> bool {
    if d.hook_margin > 0.0 {
        if logit > d.hook + d.hook_margin {
            true
        } else if logit < d.hook - d.hook_margin {
            false
        } else {
            hold_hook
        }
    } else {
        logit > d.hook
    }
}

/// Replaces the keys named in `list` (`"press=-1.5,jump=0.5,hook=0,dir_margin=1"`) of `d`; the others keep their value.
pub fn apply_decode_overrides(d: &mut Decode, list: &str) -> Result<(), String> {
    for item in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (k, v) = item
            .split_once('=')
            .ok_or_else(|| format!("window_decode: {item:?} is not key=value"))?;
        let v: f32 = v.trim().parse().map_err(|e| format!("window_decode: {item:?}: {e}"))?;
        if !v.is_finite() {
            return Err(format!("window_decode: {item:?}: not a finite number"));
        }
        match k.trim() {
            "press" => d.press = v,
            "jump" => d.jump = v,
            "hook" => d.hook = v,
            "hook_margin" if v >= 0.0 => d.hook_margin = v,
            "dir_margin" if v >= 0.0 => d.dir_margin = v,
            "dir_margin" | "hook_margin" => return Err(format!("window_decode: {k} must not be negative")),
            other => {
                return Err(format!(
                    "window_decode: unknown key {other:?} (press, jump, hook, hook_margin, dir_margin)"
                ));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bundle {
    pub format_version: u32,
    pub feature_version: u32,
    /// `[K_HIST, STRIDE, FD, N_RAYS, IF_SLOTS, IF_DIM, HORIZON, KNOWN_DIM, MAX_LAG]` of the features the network was trained with.
    pub layout: [u32; 9],
    pub net: Mlp,
    pub decode: Decode,
    pub seed: u64,
    pub epochs: u32,
    pub val_loss: f64,
    pub notes: String,
}

pub fn current_layout() -> [u32; 9] {
    [
        K_HIST, STRIDE, FD, N_RAYS, IF_SLOTS, IF_DIM, HORIZON, KNOWN_DIM, MAX_LAG,
    ]
    .map(|v| v as u32)
}

impl Bundle {
    pub fn new(net: Mlp, decode: Decode, seed: u64, epochs: u32, val_loss: f64, notes: String) -> Bundle {
        Bundle {
            format_version: BUNDLE_VERSION,
            feature_version: FEATURE_VERSION,
            layout: current_layout(),
            net,
            decode,
            seed,
            epochs,
            val_loss,
            notes,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.format_version != BUNDLE_VERSION {
            return Err(format!(
                "opponent model: format version {} (this reader takes {BUNDLE_VERSION})",
                self.format_version
            ));
        }
        if self.feature_version != FEATURE_VERSION || self.layout != current_layout() {
            return Err("opponent model: trained with another feature layout".into());
        }
        if self.net.n_in != INPUT_DIM || self.net.n_out != OUT_DIM || !self.net.is_consistent() {
            return Err(
                "opponent model: the network does not fit the feature layout (or holds a non-finite weight)".into(),
            );
        }
        let d = &self.decode;
        if ![d.jump, d.hook, d.hook_margin, d.press, d.dir_margin]
            .iter()
            .all(|v| v.is_finite())
            || d.dir_margin < 0.0
            || d.hook_margin < 0.0
        {
            return Err(
                "opponent model: a decoding threshold is not a finite number (or the direction margin is negative)"
                    .into(),
            );
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        write_blob(path, self, 3)
    }

    pub fn load(path: &Path) -> Result<Bundle, String> {
        let b: Bundle = read_blob(path)?;
        b.validate().map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(b)
    }
}

/// History entries kept (the frames `0..K_HIST` strides back, plus slack for a skipped decision).
const RING: usize = K_HIST + 3;

/// What one call costs the work clock, in tee-ticks (1.25 us each): the whole call is a forward pass of ~100k multiply-adds (see `opp_bench`).
pub const WORK_UNITS: u64 = 16;

pub struct Predictor {
    net: Mlp,
    scratch: Scratch,
    name: String,
    ring: [(i32, [f32; FD]); RING],
    len: usize,
    head: usize,
    x: Box<[f32; INPUT_DIM]>,
    decode: Decode,
    known: [Option<KnownTick>; HORIZON],
}

impl Predictor {
    pub fn new(bundle: Bundle, name: &str) -> Result<Predictor, String> {
        bundle.validate()?;
        let scratch = Scratch::new(&bundle.net);
        Ok(Predictor {
            net: bundle.net,
            scratch,
            name: name.to_string(),
            ring: [(i32::MIN, [0.0; FD]); RING],
            len: 0,
            head: 0,
            x: Box::new([0.0; INPUT_DIM]),
            decode: bundle.decode,
            known: [None; HORIZON],
        })
    }

    pub fn load(path: &Path) -> Result<Predictor, String> {
        let name = path
            .file_stem()
            .map_or_else(|| "oppnet".to_string(), |s| s.to_string_lossy().into_owned());
        Predictor::new(Bundle::load(path)?, &name)
    }

    pub fn decode(&self) -> &Decode {
        &self.decode
    }

    /// Other decoding thresholds (tuning, ablations).
    pub fn with_decode(mut self, d: Decode) -> Predictor {
        self.decode = d;
        self
    }

    /// The opponent's real inputs for the first ticks of the next window (the server's pre-inputs): `known[k]` for tick `k`. Used by the next
    /// [`WindowModel::predict`] call only, then cleared.
    pub fn set_known(&mut self, known: &[Option<KnownTick>]) {
        self.known = [None; HORIZON];
        for (slot, k) in self.known.iter_mut().zip(known) {
            *slot = *k;
        }
    }

    /// The network's logits of the last call.
    pub fn logits(&self) -> &[f32] {
        &self.scratch.out
    }

    fn push(&mut self, tick: i32, f: [f32; FD]) {
        if self.len > 0 && self.ring[(self.head + RING - 1) % RING].0 == tick {
            self.ring[(self.head + RING - 1) % RING].1 = f;
            return;
        }
        self.ring[self.head] = (tick, f);
        self.head = (self.head + 1) % RING;
        self.len = (self.len + 1).min(RING);
    }

    fn find(&self, tick: i32) -> Option<usize> {
        (0..self.len)
            .map(|k| (self.head + RING - 1 - k) % RING)
            .find(|&i| self.ring[i].0 == tick)
    }
}

/// The direction class (`0, 1, 2` for `-1, 0, 1`) of the three direction logits `o[..3]`: the argmax, unless it beats the shown direction `hold_dir` by less than `margin`.
pub fn decode_dir(o: &[f32], margin: f32, hold_dir: u8) -> u8 {
    let best = if o[0] >= o[1] && o[0] >= o[2] {
        0
    } else if o[1] >= o[2] {
        1
    } else {
        2
    };
    if margin > 0.0 && o[best] - o[usize::from(hold_dir)] < margin {
        hold_dir
    } else {
        best as u8
    }
}

/// One tick of a prediction from the logits.
pub fn decode_tick(
    out: &[f32],
    k: usize,
    d: &Decode,
    hold_dir: u8,
    hold_hook: bool,
    base_angle: f64,
) -> PredictedInput {
    let o = &out[k * HEAD_DIM..(k + 1) * HEAD_DIM];
    let dir = decode_dir(o, d.dir_margin, hold_dir);
    PredictedInput {
        direction: i32::from(dir) - 1,
        jump: o[3] > d.jump,
        hook: decode_hook(o[4], d, hold_hook),
        press: o[5] > d.press,
        aim: wrap_angle(base_angle + f64::from(o[6])),
    }
}

impl WindowModel for Predictor {
    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self) {
        self.len = 0;
        self.head = 0;
        self.known = [None; HORIZON];
    }

    fn predict(&mut self, ctx: &WindowCtx<'_>, out: &mut [Option<PredictedInput>]) {
        let w = ctx.world;
        let (Some(me), Some(opp)) = (
            TeeFrame::from_world(w, ctx.self_id, ctx.victim_id),
            TeeFrame::from_world(w, ctx.victim_id, ctx.self_id),
        ) else {
            return;
        };
        if !me.alive || !opp.alive {
            return;
        }
        let mut f = [0.0; FD];
        frame_features(&me, &opp, &mut f);
        let now = w.tick;
        self.push(now, f);
        if out.is_empty() {
            self.known = [None; HORIZON];
            return;
        }
        let mut hist: [[f32; FD]; K_HIST] = [[0.0; FD]; K_HIST];
        let mut carry = f;
        for (j, slot) in hist.iter_mut().enumerate() {
            if let Some(i) = self.find(now - (j * STRIDE) as i32) {
                carry = self.ring[i].1;
            }
            *slot = carry;
        }
        let (mut ray_opp, mut ray_me) = ([0.0f32; N_RAYS], [0.0f32; N_RAYS]);
        rays(w, opp.pos, &mut ray_opp);
        rays(w, me.pos, &mut ray_me);
        let lag = ctx.in_flight.len();
        let mut inflight = [[0.0f32; IF_DIM]; IF_SLOTS];
        for (slot, wire) in inflight.iter_mut().zip(ctx.in_flight) {
            inflight_features(&InputRec::from_wire(wire), slot);
        }
        assemble(
            &mut self.x,
            |j| &hist[j],
            &ray_opp,
            &ray_me,
            &inflight[..lag.min(IF_SLOTS)],
            lag,
            &self.known,
        );
        self.known = [None; HORIZON];
        self.net.forward(&self.x[..], &mut self.scratch);
        debug_assert_eq!(self.scratch.out.len(), OUT_DIM);
        let base = f64::from(opp.angle);
        let hold_dir = (i32::from(opp.direction) + 1) as u8;
        let hold_hook = opp.hook_state > 0;
        for (k, slot) in out.iter_mut().enumerate().take(HORIZON) {
            *slot = Some(decode_tick(
                &self.scratch.out,
                k,
                &self.decode,
                hold_dir,
                hold_hook,
                base,
            ));
        }
    }

    fn work_units(&self) -> u64 {
        WORK_UNITS
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ddai_physics::core::PlayerInput as Wire;
    use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
    use ddai_planner::physics_adapter::PhysicsWorld;
    use ddai_planner::plan_world::PlanWorld;
    use ddai_planner::vmath::Vec2;

    use super::*;

    fn hall() -> Arc<MapData> {
        let (w, h) = (40usize, 16usize);
        let mut game = vec![Tile::default(); w * h];
        for y in 0..h {
            for x in 0..w {
                let solid = y >= 10 || x == 0 || x == w - 1 || y == 0;
                let freeze = y == 1 && (10..=20).contains(&x);
                game[y * w + x] = Tile {
                    index: if freeze {
                        TILE_FREEZE
                    } else if solid {
                        TILE_SOLID
                    } else {
                        0
                    },
                    ..Tile::default()
                };
            }
        }
        Arc::new(MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        })
    }

    fn pair() -> PhysicsWorld {
        let mut pw = PhysicsWorld::new(hall(), 1);
        for i in 0..2 {
            pw.add_tee(
                i,
                Vec2 {
                    x: (17.5 + 4.0 * f64::from(i)) * 32.0,
                    y: 9.5 * 32.0,
                },
            );
        }
        for _ in 0..4 {
            pw.step();
        }
        pw
    }

    fn predictor(seed: u64) -> Predictor {
        let net = Mlp::new(INPUT_DIM, 24, 16, OUT_DIM, seed);
        Predictor::new(Bundle::new(net, Decode::default(), seed, 1, 0.0, "t".into()), "t").unwrap()
    }

    fn ask(p: &mut Predictor, w: &ddai_physics::world::World<f32>, lag: usize) -> Vec<Option<PredictedInput>> {
        let inflight = vec![Wire::default(); lag];
        let mut out = vec![None; lag];
        p.predict(
            &WindowCtx {
                world: w,
                self_id: 0,
                victim_id: 1,
                in_flight: &inflight,
            },
            &mut out,
        );
        out
    }

    #[test]
    fn a_prediction_fills_the_window_and_is_bit_deterministic() {
        let pw = pair();
        let w = pw.inner();
        let (mut a, mut b) = (predictor(3), predictor(3));
        let (oa, ob) = (ask(&mut a, w, 2), ask(&mut b, w, 2));
        assert_eq!(oa.len(), 2);
        assert!(oa.iter().all(Option::is_some));
        assert_eq!(oa, ob);
        for (x, y) in a.logits().iter().zip(b.logits()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        assert_ne!(oa, ask(&mut predictor(4), w, 2), "another network predicts differently");
    }

    #[test]
    fn known_ticks_change_the_input_for_one_call_only() {
        let pw = pair();
        let w = pw.inner();
        let mut p = predictor(5);
        let plain = ask(&mut p, w, 3);
        p.set_known(&[Some(KnownTick {
            direction: 1,
            jump: true,
            hook: true,
            fire_held: false,
            target_x: 100,
            target_y: 0,
        })]);
        let with = ask(&mut p, w, 3);
        assert_ne!(plain, with, "the network reads the known tick");
        assert_eq!(plain, ask(&mut p, w, 3), "the knowledge was used up by the call");
    }

    #[test]
    fn thresholds_decide_the_flags_and_a_margin_keeps_the_shown_direction() {
        let mut out = vec![0.0f32; OUT_DIM];
        out[0] = 1.0; // direction -1 by 1.0 over hold 0 (class 1 at logit 0)
        out[5] = -1.0; // press logit
        out[3] = 0.4;
        let d0 = Decode::default();
        let p = decode_tick(&out, 0, &d0, 1, false, 0.0);
        assert_eq!((p.direction, p.jump, p.hook, p.press), (-1, true, false, false));
        let d = Decode {
            press: -1.5,
            jump: 0.5,
            dir_margin: 1.5,
            ..d0
        };
        let p = decode_tick(&out, 0, &d, 1, false, 0.0);
        assert_eq!(
            (p.direction, p.jump, p.press),
            (0, false, true),
            "margin 1.5 > 1.0 keeps hold; press above -1.5; jump below 0.5"
        );
        assert!((decode_tick(&out, 0, &d, 1, false, 6.0).aim - wrap_angle(6.0)).abs() < 1e-12);
    }

    #[test]
    fn a_hook_margin_keeps_what_the_snapshot_shows_between_the_thresholds() {
        let d = Decode {
            hook_margin: 1.0,
            ..Decode::default()
        };
        assert!(decode_hook(1.5, &d, false), "above the margin: out");
        assert!(!decode_hook(-1.5, &d, true), "below it: in");
        assert!(
            decode_hook(0.5, &d, true) && !decode_hook(0.5, &d, false),
            "in between: hold"
        );
        let plain = Decode::default();
        assert!(
            decode_hook(0.1, &plain, false) && !decode_hook(-0.1, &plain, true),
            "no margin: the logit decides"
        );
    }

    #[test]
    fn decode_overrides_replace_only_the_named_keys() {
        let mut d = Decode {
            jump: 1.0,
            hook: 2.0,
            hook_margin: 0.5,
            press: 3.0,
            dir_margin: 4.0,
        };
        apply_decode_overrides(&mut d, "press=-1.5, dir_margin=0, hook_margin=1").unwrap();
        assert_eq!(
            (d.jump, d.hook, d.press, d.dir_margin, d.hook_margin),
            (1.0, 2.0, -1.5, 0.0, 1.0)
        );
        assert!(apply_decode_overrides(&mut d, "press").is_err());
        assert!(apply_decode_overrides(&mut d, "foot=1").is_err());
        assert!(apply_decode_overrides(&mut d, "press=nan").is_err());
        assert!(apply_decode_overrides(&mut d, "dir_margin=-1").is_err());
        apply_decode_overrides(&mut d, "").unwrap();
    }

    #[test]
    fn a_bundle_round_trips_and_a_foreign_one_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.oppnet");
        let b = Bundle::new(
            Mlp::new(INPUT_DIM, 8, 4, OUT_DIM, 1),
            Decode::default(),
            1,
            2,
            0.5,
            "t".into(),
        );
        b.save(&p).unwrap();
        assert_eq!(Bundle::load(&p).unwrap(), b);
        let mut bad = b.clone();
        bad.layout[0] += 1;
        assert!(bad.validate().is_err());
        let mut bad = b.clone();
        bad.net.n_in += 1;
        assert!(bad.validate().is_err());
        let mut bad = b.clone();
        bad.decode.press = f32::NAN;
        assert!(bad.validate().is_err());
        let mut bad = b;
        bad.net.params[3] = f32::NAN;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn the_history_ring_replaces_a_repeated_tick_and_reset_forgets() {
        let pw = pair();
        let w = pw.inner();
        let mut p = predictor(2);
        ask(&mut p, w, 2);
        ask(&mut p, w, 2);
        assert_eq!(p.len, 1, "a second call at the same tick replaces the frame");
        p.reset();
        assert_eq!(p.len, 0);
    }
}
