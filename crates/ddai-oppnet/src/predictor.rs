//! [`OppPredictor`]: the trained network behind the hybrid's [`WindowModel`] slot.
//!
//! One call per decision: it records the frame of the pair (the history the network reads), and, when the lag window is not empty, builds the
//! input vector, runs the network and decodes the first `lag` ticks. No allocation after construction.

use ddai_planner::hybrid::window::{PredictedInput, WindowCtx, WindowModel};

use crate::bundle::OppBundle;
use crate::feature::{
    FD, HORIZON, IF_DIM, IF_SLOTS, INPUT_DIM, K_HIST, OUT_DIM, STRIDE, assemble, frame_features, inflight_features,
    wrap_angle,
};
use crate::frame::{InputRec, N_RAYS, TeeFrame, rays};
use crate::net::{Mlp, Scratch};
use crate::train::{HoldView, decode_tick_gated};

/// History entries kept (the frames `0..K_HIST` strides back, plus slack for a skipped decision).
const RING: usize = K_HIST + 3;

/// What one prediction call costs the work clock, in tee-ticks (1.25 us each = 20 us): the whole call measured 7.8 us median and 22 us p99 on the shared VM (`opp_bench`), the dense forward 9.7 us.
pub const WORK_UNITS: u64 = 16;

pub struct OppPredictor {
    net: Mlp,
    scratch: Scratch,
    name: String,
    ring: [(i32, [f32; FD]); RING],
    len: usize,
    head: usize,
    x: Box<[f32; INPUT_DIM]>,
    /// Confidence gate in logit units (`0` = off): a direction, jump, hook or press head less sure than this answers "hold" (see [`decode_tick_gated`]); the aim change is a
    /// regression with no confidence and is never gated.
    gate: f32,
    /// Which heads are used ([`HEAD_DIR`] and friends); the others answer "hold" (head ablations).
    heads: u8,
}

/// Bits of [`OppPredictor::with_heads`].
pub const HEAD_DIR: u8 = 1;
pub const HEAD_JUMP: u8 = 2;
pub const HEAD_HOOK: u8 = 4;
pub const HEAD_PRESS: u8 = 8;
pub const HEAD_AIM: u8 = 16;
pub const HEAD_ALL: u8 = 31;

/// Parses a head list such as `dir,hook,aim` (`all` = every head).
pub fn parse_heads(list: &str) -> Result<u8, String> {
    let mut m = 0;
    if list.split(',').all(|n| n.trim().is_empty()) {
        return Err("empty head list (name at least one of dir, jump, hook, press, aim, all)".into());
    }
    for name in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        m |= match name {
            "all" => HEAD_ALL,
            "dir" => HEAD_DIR,
            "jump" => HEAD_JUMP,
            "hook" => HEAD_HOOK,
            "press" => HEAD_PRESS,
            "aim" => HEAD_AIM,
            other => return Err(format!("unknown head {other:?} (dir, jump, hook, press, aim, all)")),
        };
    }
    Ok(m)
}

impl OppPredictor {
    pub fn new(bundle: OppBundle, name: &str) -> Result<OppPredictor, String> {
        bundle.validate()?;
        let scratch = Scratch::new(&bundle.net);
        Ok(OppPredictor {
            net: bundle.net,
            scratch,
            name: name.to_string(),
            ring: [(i32::MIN, [0.0; FD]); RING],
            len: 0,
            head: 0,
            x: Box::new([0.0; INPUT_DIM]),
            gate: 0.0,
            heads: HEAD_ALL,
        })
    }

    /// Uses only the heads in `mask` (see [`parse_heads`]); the others answer what the snapshot shows (hold): the direction and hook it shows, no jump, no press, no aim change.
    pub fn with_heads(mut self, mask: u8) -> OppPredictor {
        self.heads = mask & HEAD_ALL;
        self
    }

    /// Sets the confidence gate (logit margin; `0` = off).
    pub fn with_gate(mut self, margin: f32) -> OppPredictor {
        self.gate = margin.max(0.0);
        self
    }

    pub fn load(path: &std::path::Path) -> Result<OppPredictor, String> {
        let name = path
            .file_stem()
            .map_or_else(|| "oppnet".to_string(), |s| s.to_string_lossy().into_owned());
        OppPredictor::new(OppBundle::load(path)?, &name)
    }

    fn push(&mut self, tick: i32, f: [f32; FD]) {
        // A second call at the same tick (a re-decision) replaces the frame.
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

    /// The network's logits for the window, from the world and our in-flight inputs (exposed for the offline tools).
    pub fn logits(&self) -> &[f32] {
        &self.scratch.out
    }
}

impl WindowModel for OppPredictor {
    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self) {
        self.len = 0;
        self.head = 0;
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
        let mut ray = [0.0f32; N_RAYS];
        rays(w, opp.pos, &mut ray);
        let lag = ctx.in_flight.len();
        let mut inflight = [[0.0f32; IF_DIM]; IF_SLOTS];
        for (slot, wire) in inflight.iter_mut().zip(ctx.in_flight) {
            inflight_features(&InputRec::from_wire(wire), slot);
        }
        assemble(&mut self.x, |j| &hist[j], &ray, &inflight[..lag.min(IF_SLOTS)], lag);
        self.net.forward(&self.x[..], &mut self.scratch);
        debug_assert_eq!(self.scratch.out.len(), OUT_DIM);
        let base = f64::from(opp.angle);
        let hold = HoldView {
            dir: (i32::from(opp.direction) + 1) as u8,
            hook: opp.hook_state > 0,
        };
        for (k, slot) in out.iter_mut().enumerate().take(HORIZON) {
            let mut d = decode_tick_gated(&self.scratch.out, k, Some(hold), self.gate);
            if self.heads != HEAD_ALL {
                if self.heads & HEAD_DIR == 0 {
                    d.dir = hold.dir;
                }
                if self.heads & HEAD_JUMP == 0 {
                    d.jump = false;
                }
                if self.heads & HEAD_HOOK == 0 {
                    d.hook = hold.hook;
                }
                if self.heads & HEAD_PRESS == 0 {
                    d.press = false;
                }
                if self.heads & HEAD_AIM == 0 {
                    d.aim_delta = 0.0;
                }
            }
            *slot = Some(PredictedInput {
                direction: i32::from(d.dir) - 1,
                jump: d.jump,
                hook: d.hook,
                press: d.press,
                aim: wrap_angle(base + f64::from(d.aim_delta)),
            });
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

    fn predictor(seed: u64) -> OppPredictor {
        let net = Mlp::new(INPUT_DIM, 24, 16, OUT_DIM, seed);
        OppPredictor::new(OppBundle::new(net, seed, 1, 0.0, "t".into()), "t").unwrap()
    }

    fn ask(p: &mut OppPredictor, w: &ddai_physics::world::World<f32>, lag: usize) -> Vec<Option<PredictedInput>> {
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
        let (oa, ob) = (ask(&mut a, w, 3), ask(&mut b, w, 3));
        assert_eq!(oa.len(), 3);
        assert!(oa.iter().all(Option::is_some));
        assert_eq!(oa, ob);
        for (x, y) in a.logits().iter().zip(b.logits()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        assert_ne!(oa, ask(&mut predictor(4), w, 3), "another network predicts differently");
    }

    #[test]
    fn a_head_mask_turns_the_unused_heads_into_hold_and_parses() {
        assert_eq!(parse_heads("dir, hook").unwrap(), HEAD_DIR | HEAD_HOOK);
        assert_eq!(parse_heads("all").unwrap(), HEAD_ALL);
        assert!(parse_heads("dir,foot").is_err());
        assert!(
            parse_heads("").is_err() && parse_heads(" , ").is_err(),
            "an empty list is an error, not a hold-only arm"
        );
        let pw = pair();
        let w = pw.inner();
        let snap = TeeFrame::from_world(w, 1, 0).unwrap();
        // No head at all: exactly what the snapshot shows.
        let none = ask(&mut predictor(3).with_heads(0), w, 3);
        for p in none.iter().flatten() {
            assert_eq!(p.direction, i32::from(snap.direction));
            assert_eq!((p.jump, p.hook, p.press), (false, snap.hook_state > 0, false));
            assert!((p.aim - wrap_angle(f64::from(snap.angle))).abs() < 1e-9);
        }
        // All heads equals the unmasked predictor.
        assert_eq!(
            ask(&mut predictor(3).with_heads(HEAD_ALL), w, 3),
            ask(&mut predictor(3), w, 3)
        );
        // The jump head alone changes nothing but the jump.
        let full = ask(&mut predictor(3), w, 3);
        let jump_only = ask(&mut predictor(3).with_heads(HEAD_JUMP), w, 3);
        for (f, j) in full.iter().flatten().zip(jump_only.iter().flatten()) {
            assert_eq!(f.jump, j.jump);
            assert_eq!(
                (j.direction, j.hook, j.press),
                (i32::from(snap.direction), snap.hook_state > 0, false)
            );
        }
    }

    #[test]
    fn an_empty_window_only_records_the_history() {
        let pw = pair();
        let mut p = predictor(1);
        assert!(ask(&mut p, pw.inner(), 0).is_empty());
        assert_eq!(p.len, 1);
    }

    #[test]
    fn a_missing_victim_gets_no_prediction() {
        let mut p = predictor(1);
        let mut pw = PhysicsWorld::new(hall(), 1);
        pw.add_tee(0, Vec2 { x: 600.0, y: 300.0 });
        assert!(ask(&mut p, pw.inner(), 3).iter().all(Option::is_none));
        assert_eq!(p.len, 0, "no frame was recorded either");
    }

    #[test]
    fn history_enters_the_input_and_reset_forgets_it() {
        let mut pw = pair();
        let mut with_history = predictor(2);
        let first = ask(&mut with_history, pw.inner(), 3);
        let mut inp = ddai_planner::types::empty_input();
        inp.direction = 1;
        for _ in 0..2 {
            pw.set_input(1, inp);
            pw.step();
        }
        let w = pw.inner().clone();
        let second = ask(&mut with_history, &w, 3);
        let mut fresh = predictor(2);
        let alone = ask(&mut fresh, &w, 3);
        assert_ne!(second, alone, "the earlier frame changes the prediction");
        assert_ne!(first, second);
        with_history.reset();
        assert_eq!(
            ask(&mut with_history, &w, 3),
            alone,
            "after a reset the history is gone"
        );
    }

    #[test]
    fn a_prediction_allocates_nothing_once_warm() {
        let pw = pair();
        let w = pw.inner();
        let mut p = predictor(5);
        let mut out = vec![None; 3];
        let inflight = vec![Wire::default(); 3];
        for _ in 0..4 {
            p.predict(
                &WindowCtx {
                    world: w,
                    self_id: 0,
                    victim_id: 1,
                    in_flight: &inflight,
                },
                &mut out,
            );
        }
        let info = allocation_counter::measure(|| {
            p.predict(
                &WindowCtx {
                    world: w,
                    self_id: 0,
                    victim_id: 1,
                    in_flight: &inflight,
                },
                &mut out,
            );
        });
        assert_eq!(info.count_total, 0, "{info:?}");
    }

    /// The central consistency check: the input vector the trainer builds from a recorded game is, bit for bit, the one the predictor builds live
    /// from the same worlds (the history, the rays, the in-flight inputs and the window length).
    #[test]
    fn the_trainers_input_is_the_predictors_input() {
        use crate::data::{GameRec, TickRec};
        use crate::frame::{InputRec, N_RAYS, TeeFrame, rays};
        use crate::train::{Corpus, SampleRef};

        let mut pw = pair();
        let lag = 3usize;
        let mut worlds = Vec::new();
        let mut wires: Vec<[Wire; 2]> = Vec::new();
        let mut ticks = Vec::new();
        let tick0 = pw.inner().tick;
        // The first recorded tick has no step before it: its applied inputs are the neutral ones.
        let record = |pw: &PhysicsWorld, ticks: &mut Vec<TickRec>, wires: &mut Vec<[Wire; 2]>, worlds: &mut Vec<_>| {
            let w = pw.inner();
            let frames = [
                TeeFrame::from_world(w, 0, 1).unwrap(),
                TeeFrame::from_world(w, 1, 0).unwrap(),
            ];
            let wire = |id: u8| w.cores.get(id).map_or_else(Wire::default, |c| c.input);
            let mut ray = [1.0f32; N_RAYS];
            rays(w, frames[1].pos, &mut ray);
            wires.push([wire(0), wire(1)]);
            ticks.push(TickRec {
                frames,
                applied: [InputRec::from_wire(&wire(0)), InputRec::from_wire(&wire(1))],
                rays: ray,
            });
            worlds.push(w.clone());
        };
        record(&pw, &mut ticks, &mut wires, &mut worlds);
        for step in 0..40 {
            let mut a = ddai_planner::types::empty_input();
            a.direction = [1, 0, -1][step / 7 % 3];
            a.jump = i32::from(step % 11 == 3);
            a.target_x = 300.0;
            let mut b = ddai_planner::types::empty_input();
            b.direction = [-1, 1][step / 5 % 2];
            b.hook = i32::from(step % 9 > 4);
            b.target_x = -200.0;
            b.target_y = -150.0;
            b.fire = i32::from(step % 13 == 6);
            pw.set_input(0, a);
            pw.set_input(1, b);
            pw.step();
            record(&pw, &mut ticks, &mut wires, &mut worlds);
        }
        let game = GameRec {
            arena: "t".into(),
            seed: 1,
            lag: [lag as u8, 0],
            swap: false,
            decide_every: 2,
            tick0,
            ticks,
        };
        let corpus = Corpus::new(vec![game]);
        let mut p = predictor(7);
        let mut checked = 0;
        for i in 0..worlds.len() {
            let t = tick0 + i as i32;
            if t % 2 != 0 || i + lag >= worlds.len() {
                continue;
            }
            // In flight at tick `t`: our inputs applied in the steps t .. t + lag, recorded with the ticks after them.
            let inflight: Vec<Wire> = (0..lag).map(|k| wires[i + 1 + k][0]).collect();
            let mut out = vec![None; lag];
            p.predict(
                &WindowCtx {
                    world: &worlds[i],
                    self_id: 0,
                    victim_id: 1,
                    in_flight: &inflight,
                },
                &mut out,
            );
            let mut x = Box::new([0.0f32; INPUT_DIM]);
            corpus.make(SampleRef { game: 0, idx: i as u32 }, &mut x);
            for (j, (a, b)) in p.x.iter().zip(x.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "tick {t}, input {j}: {a} vs {b}");
            }
            checked += 1;
        }
        assert!(checked >= 15, "{checked}");
    }

    #[test]
    fn the_history_ring_wraps_and_finds_frames_by_tick() {
        let mut p = predictor(1);
        for t in 0..30 {
            p.push(t * 2, [t as f32; FD]);
        }
        assert_eq!(p.len, RING);
        assert_eq!(p.ring[p.find(58).unwrap()].1[0], 29.0);
        assert_eq!(
            p.ring[p.find(58 - 2 * (RING as i32 - 1)).unwrap()].1[0],
            (29 - (RING as i32 - 1)) as f32
        );
        assert!(p.find(58 - 2 * RING as i32).is_none(), "older than the ring");
        p.push(58, [99.0; FD]);
        assert_eq!(
            p.ring[p.find(58).unwrap()].1[0],
            99.0,
            "a second call at the same tick replaces the frame"
        );
        assert_eq!(p.len, RING);
    }
}
