//! Task 3.24 (E-039), the imitation pilot: a small **behaviour-cloning head** trained on what real humans do in the seconds after they freeze an
//! opponent ("after the block"), and a [`HumanPriorProposer`] that offers its action as candidate plans to the hybrid brain's search.
//!
//! The head sees the pair (the human, its victim) the way the live predictor sees a duel -- the v2 frame features, the geometry rays around both tees and the
//! human's previous input -- and predicts the human's next input: direction (3 classes), jump, hook and hammer press (levels) and the aim relative to the
//! line to the victim. Labels are the **real** inputs from the demos' `Sv_PreInput` messages ([`crate::humandata`] describes the source). Proposals are
//! candidates, never decisions: the exact search scores them with the same rollouts as everything else and plays one only if it wins (D-041), so a prior can
//! only help by being a better candidate than the search finds itself. The proposer is silent unless the victim is frozen and we are free, i.e. exactly in
//! the situations it was trained on.
//!
//! **No nicknames**: samples are numbers; a model file holds weights and a note.

use ddai_jsmath::Rng;
use ddai_planner::hybrid::proposer::{ActionDistribution, ProposeCtx, Proposer, plans_from_distribution};
use ddai_planner::planner::PlanStep;
use serde::{Deserialize, Serialize};

use crate::feature::{IF_DIM, inflight_features, wrap_angle};
use crate::frame::{InputRec, N_RAYS, TeeFrame, rays};
use crate::net::{Mlp, Scratch};
use crate::train::Adam;
use crate::v2::feature::{FD, frame_features};

/// Inputs of the head: the pair's frame features, the rays around the human and around its victim, the human's previous input.
pub const BC_IN: usize = FD + 2 * N_RAYS + IF_DIM;
/// Outputs: direction logits (3), jump, hook, press logits, cosine and sine of the aim relative to the line to the victim.
pub const BC_OUT: usize = 8;
/// Bumped when an input or output number changes its meaning.
pub const BC_VERSION: u32 = 1;

/// What the human did next, as the head's targets.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BcLabel {
    /// Direction class `0, 1, 2` for `-1, 0, 1`.
    pub dir: u8,
    pub jump: bool,
    pub hook: bool,
    /// A press of the fire key (hammer in hand).
    pub fire: bool,
    /// The aim relative to the line from the human to its victim (radians, wrapped).
    pub aim_rel: f32,
}

/// One training sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BcSample {
    /// The demo number: the unit of the train / validation / test split.
    pub session: u8,
    pub x: Vec<f32>,
    pub y: BcLabel,
}

/// The input vector of the head: `me` the human, `victim` the frozen opponent, `rays_*` the geometry around each, `prev` the human's previous input.
pub fn features(
    me: &TeeFrame,
    victim: &TeeFrame,
    rays_me: &[f32; N_RAYS],
    rays_victim: &[f32; N_RAYS],
    prev: &InputRec,
    x: &mut [f32; BC_IN],
) {
    let mut f = [0.0f32; FD];
    frame_features(me, victim, &mut f);
    x[..FD].copy_from_slice(&f);
    x[FD..FD + N_RAYS].copy_from_slice(rays_me);
    x[FD + N_RAYS..FD + 2 * N_RAYS].copy_from_slice(rays_victim);
    let mut p = [0.0f32; IF_DIM];
    inflight_features(prev, &mut p);
    x[FD + 2 * N_RAYS..].copy_from_slice(&p);
}

/// The aim label: the angle of the human's aim minus the angle to the victim.
pub fn aim_rel(input: &InputRec, me: &TeeFrame, victim: &TeeFrame) -> f32 {
    let aim = f64::from(input.target_y).atan2(f64::from(input.target_x));
    let to = f64::from(victim.pos[1] - me.pos[1]).atan2(f64::from(victim.pos[0] - me.pos[0]));
    wrap_angle(aim - to) as f32
}

#[derive(Debug, Clone, Copy)]
pub struct BcLossCfg {
    pub w_dir: f32,
    pub w_jump: f32,
    pub w_hook: f32,
    pub w_fire: f32,
    pub w_aim: f32,
    /// Positive weight of the hammer press and of the jump (rare events).
    pub pos_weight: f32,
}

impl Default for BcLossCfg {
    fn default() -> Self {
        BcLossCfg {
            w_dir: 1.0,
            w_jump: 1.0,
            w_hook: 1.0,
            w_fire: 1.0,
            w_aim: 1.0,
            pos_weight: 2.0,
        }
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Loss of one sample from the head's outputs and its derivative into `d`.
pub fn bc_loss_and_grad(out: &[f32], y: &BcLabel, cfg: &BcLossCfg, d: &mut [f32]) -> f64 {
    d.fill(0.0);
    let mut total = 0.0f64;
    let m = out[0].max(out[1]).max(out[2]);
    let e = [(out[0] - m).exp(), (out[1] - m).exp(), (out[2] - m).exp()];
    let z = e[0] + e[1] + e[2];
    for c in 0..3 {
        let p = e[c] / z;
        d[c] = cfg.w_dir * (p - f32::from(u8::from(usize::from(y.dir) == c)));
    }
    total += f64::from(cfg.w_dir) * f64::from(-(e[usize::from(y.dir)] / z).max(1e-12).ln());
    for (slot, label, w, pos_w) in [
        (3, y.jump, cfg.w_jump, cfg.pos_weight),
        (4, y.hook, cfg.w_hook, 1.0),
        (5, y.fire, cfg.w_fire, cfg.pos_weight),
    ] {
        let yy = f32::from(u8::from(label));
        let z = out[slot];
        let w = w * if label { pos_w } else { 1.0 };
        d[slot] = w * (sigmoid(z) - yy);
        total += f64::from(w) * f64::from(z.max(0.0) + (-z.abs()).exp().ln_1p() - yy * z);
    }
    let (ty, tx) = (f64::from(y.aim_rel).sin() as f32, f64::from(y.aim_rel).cos() as f32);
    for (slot, t) in [(6, tx), (7, ty)] {
        let diff = out[slot] - t;
        d[slot] = cfg.w_aim * diff;
        total += f64::from(cfg.w_aim) * 0.5 * f64::from(diff * diff);
    }
    total
}

#[derive(Debug, Clone)]
pub struct BcTrainCfg {
    pub h1: usize,
    pub h2: usize,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    pub lr_min: f32,
    pub weight_decay: f32,
    pub seed: u64,
    pub loss: BcLossCfg,
}

impl Default for BcTrainCfg {
    fn default() -> Self {
        BcTrainCfg {
            h1: 128,
            h2: 64,
            epochs: 20,
            batch: 128,
            lr: 1.5e-3,
            lr_min: 1e-4,
            weight_decay: 1e-5,
            seed: 1,
            loss: BcLossCfg::default(),
        }
    }
}

/// Mean loss over `samples`.
pub fn mean_bc_loss(m: &Mlp, samples: &[BcSample], cfg: &BcLossCfg) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut s = Scratch::new(m);
    let mut d = vec![0.0; BC_OUT];
    let mut tot = 0.0;
    for smp in samples {
        m.forward(&smp.x, &mut s);
        tot += bc_loss_and_grad(&s.out, &smp.y, cfg, &mut d);
    }
    tot / samples.len() as f64
}

/// Trains a head, keeping the epoch with the best loss on `val`. Single-threaded and deterministic.
pub fn train_bc(
    train: &[BcSample],
    val: &[BcSample],
    cfg: &BcTrainCfg,
    log: &mut dyn FnMut(&str),
) -> Result<(Mlp, f64), String> {
    if train.is_empty() || val.is_empty() {
        return Err("no samples".into());
    }
    if train.iter().chain(val).any(|s| s.x.len() != BC_IN) {
        return Err(format!("every sample must have {BC_IN} inputs"));
    }
    let mut m = Mlp::new(BC_IN, cfg.h1, cfg.h2, BC_OUT, cfg.seed);
    // the flag heads start at the base rates
    {
        let n = train.len() as f64;
        let rate = |f: &dyn Fn(&BcSample) -> bool| {
            let p = (train.iter().filter(|s| f(s)).count() as f64 + 1.0) / (n + 2.0);
            (p / (1.0 - p)).ln() as f32
        };
        let (_, _, _, _, _, b3) = m.offsets();
        m.params[b3 + 3] = rate(&|s| s.y.jump);
        m.params[b3 + 4] = rate(&|s| s.y.hook);
        m.params[b3 + 5] = rate(&|s| s.y.fire);
    }
    let mut adam = Adam::new(m.params.len());
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut rng = cfg.seed ^ 0xBC_BC_BC;
    let mut next = move || {
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let steps_per_epoch = train.len().div_ceil(cfg.batch);
    let total_steps = (steps_per_epoch * cfg.epochs).max(1);
    let mut step = 0usize;
    let mut best: Option<(f64, Mlp)> = None;
    let mut scratch = Scratch::new(&m);
    let mut d = vec![0.0f32; BC_OUT];
    let mut grad = vec![0.0f32; m.params.len()];
    let mut batch_grad = vec![0.0f32; m.params.len()];
    for epoch in 0..cfg.epochs {
        for i in (1..order.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let (mut epoch_loss, mut epoch_n) = (0.0f64, 0u64);
        for b in 0..steps_per_epoch {
            let batch = &order[b * cfg.batch..((b + 1) * cfg.batch).min(order.len())];
            grad.fill(0.0);
            for &oi in batch {
                let s = &train[oi];
                m.forward(&s.x, &mut scratch);
                epoch_loss += bc_loss_and_grad(&scratch.out, &s.y, &cfg.loss, &mut d);
                batch_grad.fill(0.0);
                m.backward(&s.x, &mut scratch, &d, &mut batch_grad);
                for (g, bg) in grad.iter_mut().zip(&batch_grad) {
                    *g += bg;
                }
                epoch_n += 1;
            }
            let inv = 1.0 / batch.len().max(1) as f32;
            grad.iter_mut().for_each(|g| *g *= inv);
            let prog = step as f32 / total_steps as f32;
            let lr = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + (std::f32::consts::PI * prog).cos());
            adam.step(&mut m.params, &grad, lr, cfg.weight_decay);
            step += 1;
        }
        let v = mean_bc_loss(&m, val, &cfg.loss);
        log(&format!(
            "epoch {:>2}: train loss {:.4}  val loss {:.4}",
            epoch + 1,
            epoch_loss / epoch_n.max(1) as f64,
            v
        ));
        if best.as_ref().is_none_or(|(b, _)| v < *b) {
            best = Some((v, m.clone()));
        }
    }
    let (v, model) = best.ok_or("no epochs")?;
    Ok((model, v))
}

/// The head's output as the action distribution the planner turns into plans; `to_victim` is the absolute angle (y down) from the human to its victim.
pub fn distribution(out: &[f32], to_victim: f64) -> ActionDistribution {
    let m = out[0].max(out[1]).max(out[2]);
    let e = [(out[0] - m).exp(), (out[1] - m).exp(), (out[2] - m).exp()];
    let z = e[0] + e[1] + e[2];
    let rel = f64::from(out[7]).atan2(f64::from(out[6]));
    ActionDistribution {
        direction: [f64::from(e[0] / z), f64::from(e[1] / z), f64::from(e[2] / z)],
        jump: f64::from(sigmoid(out[3])),
        hook: f64::from(sigmoid(out[4])),
        fire: f64::from(sigmoid(out[5])),
        aim_angle: wrap_angle(to_victim + rel),
    }
}

/// A trained head with its provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BcModel {
    pub version: u32,
    pub net: Mlp,
    /// Mean validation loss of the saved epoch.
    pub val_loss: f64,
    /// Free text without names (what it was trained on).
    pub notes: String,
}

impl BcModel {
    pub fn new(net: Mlp, val_loss: f64, notes: String) -> BcModel {
        BcModel {
            version: BC_VERSION,
            net,
            val_loss,
            notes,
        }
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        crate::blob::write_blob(path, self, 3)
    }

    pub fn load(path: &std::path::Path) -> Result<BcModel, String> {
        let m: BcModel = crate::blob::read_blob(path)?;
        if m.version != BC_VERSION || m.net.n_in != BC_IN || m.net.n_out != BC_OUT {
            return Err(format!(
                "{}: not a version {BC_VERSION} behaviour-cloning head",
                path.display()
            ));
        }
        Ok(m)
    }
}

/// The situation the head was trained on: we are free and alive, the victim is alive and frozen.
pub fn after_the_block(me: &TeeFrame, victim: &TeeFrame) -> bool {
    me.alive && me.freeze_left == 0 && victim.alive && victim.freeze_left > 0
}

/// The hybrid brain's proposer backed by a head: after a block (the victim frozen, we free) it offers `k` plans from the head's distribution, the first
/// the argmax plan, the rest samples. Before and outside a block it proposes nothing.
pub struct HumanPriorProposer {
    net: Mlp,
    scratch: Scratch,
    seed: u32,
    /// How many `propose` calls found the situation (a frozen victim, we free) and offered plans.
    pub engaged: u64,
    pub calls: u64,
}

impl HumanPriorProposer {
    pub fn new(model: BcModel) -> HumanPriorProposer {
        HumanPriorProposer {
            scratch: Scratch::new(&model.net),
            net: model.net,
            seed: 29,
            engaged: 0,
            calls: 0,
        }
    }
}

impl Proposer for HumanPriorProposer {
    fn name(&self) -> &str {
        "humanprior"
    }

    fn reset(&mut self, ctx: &ddai_brain::ResetContext) {
        self.seed = ctx.seed.wrapping_mul(7919).wrapping_add(29) as u32;
        self.engaged = 0;
        self.calls = 0;
    }

    fn propose(&mut self, ctx: &ProposeCtx<'_>, out: &mut Vec<Vec<PlanStep>>) {
        self.calls += 1;
        if ctx.k == 0 || ctx.steps == 0 {
            return;
        }
        let w = ctx.world.inner();
        let (Some(me), Some(victim)) = (
            TeeFrame::from_world(w, ctx.self_id, ctx.victim_id),
            TeeFrame::from_world(w, ctx.victim_id, ctx.self_id),
        ) else {
            return;
        };
        if !after_the_block(&me, &victim) {
            return;
        }
        let (mut r_me, mut r_victim) = ([1.0f32; N_RAYS], [1.0f32; N_RAYS]);
        rays(w, me.pos, &mut r_me);
        rays(w, victim.pos, &mut r_victim);
        let p = &ctx.prev;
        let prev = InputRec {
            direction: p.direction.clamp(-1, 1) as i8,
            jump: p.jump != 0,
            hook: p.hook != 0,
            fire: p.fire,
            target_x: p.target_x.clamp(-32768.0, 32767.0) as i16,
            target_y: p.target_y.clamp(-32768.0, 32767.0) as i16,
        };
        let mut x = [0.0f32; BC_IN];
        features(&me, &victim, &r_me, &r_victim, &prev, &mut x);
        self.net.forward(&x, &mut self.scratch);
        let to_victim = f64::from(victim.pos[1] - me.pos[1]).atan2(f64::from(victim.pos[0] - me.pos[0]));
        let d = distribution(&self.scratch.out, to_victim);
        let mut rng = Rng::new(self.seed.wrapping_add(self.engaged as u32));
        plans_from_distribution(&d, ctx.steps, ctx.k, &mut || rng.next_float(), out);
        self.engaged += 1;
    }

    fn costs_time(&self) -> bool {
        true
    }

    /// A forward pass of a ~20 k parameter network, in tee-tick equivalents of the work clock (a few microseconds).
    fn work_units(&self) -> u64 {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(dir: u8, hook: bool) -> BcLabel {
        BcLabel {
            dir,
            jump: false,
            hook,
            fire: false,
            aim_rel: 0.3,
        }
    }

    fn sample(seed: u32, positive: bool) -> BcSample {
        // an input that tells the label: x[0] carries it, the rest is noise-free filler
        let mut x = vec![0.0f32; BC_IN];
        x[0] = if positive { 1.0 } else { -1.0 };
        x[1] = (seed % 7) as f32 / 7.0;
        BcSample {
            session: 0,
            x,
            y: label(if positive { 2 } else { 0 }, positive),
        }
    }

    #[test]
    fn the_gradient_matches_finite_differences() {
        let cfg = BcLossCfg::default();
        let y = BcLabel {
            dir: 2,
            jump: true,
            hook: false,
            fire: true,
            aim_rel: -0.7,
        };
        let o: Vec<f32> = (0..BC_OUT).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut d = vec![0.0; BC_OUT];
        let base = bc_loss_and_grad(&o, &y, &cfg, &mut d);
        assert!(base.is_finite() && base > 0.0);
        let eps = 1e-3f32;
        for i in 0..BC_OUT {
            let (mut a, mut b) = (o.clone(), o.clone());
            a[i] += eps;
            b[i] -= eps;
            let mut sink = vec![0.0; BC_OUT];
            let num = ((bc_loss_and_grad(&a, &y, &cfg, &mut sink) - bc_loss_and_grad(&b, &y, &cfg, &mut sink))
                / (2.0 * f64::from(eps))) as f32;
            assert!((d[i] - num).abs() < 5e-3, "output {i}: {} vs {num}", d[i]);
        }
    }

    #[test]
    fn training_learns_a_separable_rule_and_the_distribution_follows_it() {
        let train: Vec<BcSample> = (0..400).map(|i| sample(i, i % 2 == 0)).collect();
        let val: Vec<BcSample> = (0..100).map(|i| sample(i + 1000, i % 2 == 0)).collect();
        let cfg = BcTrainCfg {
            epochs: 80,
            batch: 32,
            lr: 5e-3,
            h1: 16,
            h2: 8,
            ..BcTrainCfg::default()
        };
        let mut lines = 0;
        let (net, val_loss) = train_bc(&train, &val, &cfg, &mut |_| lines += 1).unwrap();
        assert_eq!(lines, 80);
        let before = mean_bc_loss(&Mlp::new(BC_IN, 16, 8, BC_OUT, 1), &val, &cfg.loss);
        assert!(val_loss < 0.5 * before, "{val_loss} vs untrained {before}");
        let mut s = Scratch::new(&net);
        net.forward(&sample(1, true).x, &mut s);
        let d = distribution(&s.out, 0.0);
        assert!(d.direction[2] > 0.8 && d.hook > 0.8, "{d:?}");
        assert!((d.direction.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert!((d.aim_angle - 0.3).abs() < 0.2, "aim {}", d.aim_angle);
        net.forward(&sample(1, false).x, &mut s);
        let d = distribution(&s.out, 0.0);
        assert!(d.direction[0] > 0.8 && d.hook < 0.2, "{d:?}");
        // the same data and seed give the same bytes
        let (net2, _) = train_bc(&train, &val, &cfg, &mut |_| ()).unwrap();
        assert_eq!(net.params, net2.params);
        assert!(train_bc(&[], &val, &cfg, &mut |_| ()).is_err());
    }

    #[test]
    fn the_proposer_is_for_a_free_us_and_a_frozen_victim_only() {
        let free = TeeFrame {
            alive: true,
            ..TeeFrame::default()
        };
        let frozen = TeeFrame {
            alive: true,
            freeze_left: 60,
            ..TeeFrame::default()
        };
        assert!(after_the_block(&free, &frozen));
        assert!(!after_the_block(&free, &free), "a free victim: not the situation");
        assert!(!after_the_block(&frozen, &frozen), "we are frozen too");
        assert!(!after_the_block(&frozen, &free));
        let dead = TeeFrame {
            alive: false,
            freeze_left: 60,
            ..TeeFrame::default()
        };
        assert!(!after_the_block(&free, &dead), "a dead victim");
        assert!(!after_the_block(&dead, &frozen), "we are dead");
    }

    #[test]
    fn the_aim_label_is_relative_to_the_line_to_the_victim() {
        let me = TeeFrame {
            pos: [0.0, 0.0],
            alive: true,
            ..TeeFrame::default()
        };
        let victim = TeeFrame {
            pos: [100.0, 0.0],
            alive: true,
            ..TeeFrame::default()
        };
        let straight = InputRec {
            target_x: 50,
            target_y: 0,
            ..InputRec::default()
        };
        assert!(aim_rel(&straight, &me, &victim).abs() < 1e-6);
        let up = InputRec {
            target_x: 0,
            target_y: -50,
            ..InputRec::default()
        };
        assert!((f64::from(aim_rel(&up, &me, &victim)) + std::f64::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn features_place_every_block_and_a_model_round_trips() {
        let me = TeeFrame {
            alive: true,
            pos: [10.0, 20.0],
            ..TeeFrame::default()
        };
        let victim = TeeFrame {
            alive: true,
            pos: [60.0, 20.0],
            freeze_left: 90,
            ..TeeFrame::default()
        };
        let mut x = [9.0f32; BC_IN];
        features(
            &me,
            &victim,
            &[0.5; N_RAYS],
            &[0.25; N_RAYS],
            &InputRec {
                direction: -1,
                hook: true,
                ..InputRec::default()
            },
            &mut x,
        );
        assert_eq!((x[FD], x[FD + N_RAYS]), (0.5, 0.25));
        assert_eq!((x[FD + 2 * N_RAYS], x[FD + 2 * N_RAYS + 2]), (-1.0, 1.0));
        assert!(x.iter().all(|v| v.is_finite()));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("h.bc");
        let model = BcModel::new(Mlp::new(BC_IN, 8, 4, BC_OUT, 3), 1.25, "test".into());
        model.save(&p).unwrap();
        assert_eq!(BcModel::load(&p).unwrap(), model);
        // a model of another shape is refused
        let bad = BcModel::new(Mlp::new(BC_IN + 1, 8, 4, BC_OUT, 3), 1.0, String::new());
        bad.save(&p).unwrap();
        assert!(BcModel::load(&p).is_err());
    }
}
