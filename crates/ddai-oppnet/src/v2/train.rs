//! The v2 loss, trainer and offline metrics.
//!
//! The loss has a mask per head and tick (a clip shows a head at some ticks only, a known tick is not learned), a positive weight on the fire head (a swing is
//! rare), a weight per window tick (the live window needs ticks 0 and 1 most) and a weight per source (clips are scarce). The trainer is the v1 one --
//! Adam, a cosine learning rate, four fixed gradient shards summed in order, so the result does not depend on the number of threads -- with an
//! optional starting network (fine-tuning).

use rayon::prelude::*;

use super::corpus::{Corpus, SampleRef};
use super::feature::{HEAD_DIM, HORIZON, INPUT_DIM, Label, OUT_DIM};
use super::predictor::Decode;
use crate::net::{Mlp, Scratch};
use crate::train::Adam;

#[derive(Debug, Clone, Copy)]
pub struct LossCfg {
    pub w_dir: f32,
    pub w_jump: f32,
    pub w_hook: f32,
    pub w_press: f32,
    pub w_aim: f32,
    /// Huber threshold of the aim loss (radians).
    pub huber: f32,
    /// The weight of a positive fire label against a negative one.
    pub press_pos: f32,
    /// Weight of each window tick.
    pub k_weight: [f32; HORIZON],
    /// Weight of a clip sample against an arena one.
    pub clip_weight: f32,
}

impl Default for LossCfg {
    fn default() -> Self {
        LossCfg {
            w_dir: 1.0,
            w_jump: 1.0,
            w_hook: 1.0,
            w_press: 1.0,
            w_aim: 1.0,
            huber: 0.3,
            press_pos: 3.0,
            k_weight: [1.0, 1.0, 0.5, 0.5],
            clip_weight: 1.0,
        }
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Loss of one sample from the logits `out` and its derivative into `d`; returns the loss summed over the valid ticks and heads.
pub fn loss_and_grad(out: &[f32], l: &Label, cfg: &LossCfg, src_weight: f32, d: &mut [f32]) -> f64 {
    d.fill(0.0);
    let mut total = 0.0f64;
    for k in 0..HORIZON {
        let o = &out[k * HEAD_DIM..(k + 1) * HEAD_DIM];
        let dd = &mut d[k * HEAD_DIM..(k + 1) * HEAD_DIM];
        let wk = cfg.k_weight[k] * src_weight;
        if l.v_dir >> k & 1 != 0 {
            let m = o[0].max(o[1]).max(o[2]);
            let e = [(o[0] - m).exp(), (o[1] - m).exp(), (o[2] - m).exp()];
            let z = e[0] + e[1] + e[2];
            for c in 0..3 {
                let p = e[c] / z;
                let y = f32::from(u8::from(usize::from(l.dir[k]) == c));
                dd[c] = wk * cfg.w_dir * (p - y);
            }
            total += f64::from(wk * cfg.w_dir) * f64::from(-(e[usize::from(l.dir[k])] / z).max(1e-12).ln());
        }
        for (slot, valid, bits, w, pos_w) in [
            (3, l.v_jump, l.jump, cfg.w_jump, 1.0),
            (4, l.v_hook, l.hook, cfg.w_hook, 1.0),
            (5, l.v_press, l.press, cfg.w_press, cfg.press_pos),
        ] {
            if valid >> k & 1 == 0 {
                continue;
            }
            let y = f32::from(u8::from(bits >> k & 1 != 0));
            let z = o[slot];
            let w = wk * w * if y > 0.0 { pos_w } else { 1.0 };
            dd[slot] = w * (sigmoid(z) - y);
            total += f64::from(w) * f64::from(z.max(0.0) + (-z.abs()).exp().ln_1p() - y * z);
        }
        if l.v_aim >> k & 1 != 0 {
            let diff = o[6] - l.aim_delta[k];
            let a = diff.abs();
            let w = wk * cfg.w_aim;
            if a <= cfg.huber {
                total += f64::from(w) * 0.5 * f64::from(diff * diff);
                dd[6] = w * diff;
            } else {
                total += f64::from(w) * f64::from(cfg.huber) * f64::from(a - 0.5 * cfg.huber);
                dd[6] = w * cfg.huber * diff.signum();
            }
        }
    }
    total
}

fn src_weight(cfg: &LossCfg, r: SampleRef) -> f32 {
    if r.src == 1 { cfg.clip_weight } else { 1.0 }
}

#[derive(Debug, Clone)]
pub struct TrainCfg {
    pub h1: usize,
    pub h2: usize,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    pub lr_min: f32,
    pub weight_decay: f32,
    pub seed: u64,
    pub threads: usize,
    pub loss: LossCfg,
}

impl Default for TrainCfg {
    fn default() -> Self {
        TrainCfg {
            h1: 192,
            h2: 128,
            epochs: 14,
            batch: 256,
            lr: 1.5e-3,
            lr_min: 1e-4,
            weight_decay: 1e-5,
            seed: 1,
            threads: 3,
            loss: LossCfg::default(),
        }
    }
}

const SHARDS: usize = 4;

struct Shard {
    grad: Vec<f32>,
    scratch: Scratch,
    d: Vec<f32>,
    x: Box<[f32; INPUT_DIM]>,
    loss: f64,
    n: u32,
}

/// Mean loss of `samples` (forward only; no known ticks: salt `u64::MAX` is the validation draw, which uses the corpus' own probabilities).
pub fn mean_loss(m: &Mlp, c: &Corpus, samples: &[SampleRef], cfg: &LossCfg) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let sums: Vec<f64> = samples
        .par_chunks(2048)
        .map(|ch| {
            let mut s = Scratch::new(m);
            let mut d = vec![0.0; OUT_DIM];
            let mut x = Box::new([0.0f32; INPUT_DIM]);
            let mut tot = 0.0;
            for &r in ch {
                let l = c.make(r, u64::MAX, &mut x);
                m.forward(&x[..], &mut s);
                tot += loss_and_grad(&s.out, &l, cfg, src_weight(cfg, r), &mut d);
            }
            tot
        })
        .collect();
    sums.iter().sum::<f64>() / samples.len() as f64
}

/// Sets the flag heads' biases to the logit of their base rates.
fn init_biases(m: &mut Mlp, c: &Corpus, samples: &[SampleRef]) {
    let mut pos = [[0u64; 3]; HORIZON];
    let mut n = [[0u64; 3]; HORIZON];
    for &r in samples {
        let l = c.label(r);
        for k in 0..HORIZON {
            for (s, (valid, bits)) in [(l.v_jump, l.jump), (l.v_hook, l.hook), (l.v_press, l.press)]
                .into_iter()
                .enumerate()
            {
                if valid >> k & 1 != 0 {
                    n[k][s] += 1;
                    pos[k][s] += u64::from(bits >> k & 1);
                }
            }
        }
    }
    let (_, _, _, _, _, b3) = m.offsets();
    for k in 0..HORIZON {
        for s in 0..3 {
            let p = (pos[k][s] as f64 + 1.0) / (n[k][s] as f64 + 2.0);
            m.params[b3 + k * HEAD_DIM + 3 + s] = (p / (1.0 - p)).ln() as f32;
        }
    }
}

/// Trains a network on `train`, keeping the epoch with the best loss on `val`; `init` starts from a network (fine-tuning) instead of a fresh one.
pub fn train(
    c_train: &Corpus,
    c_val: &Corpus,
    cfg: &TrainCfg,
    init: Option<&Mlp>,
    log: &mut dyn FnMut(&str),
) -> Result<(Mlp, f64), String> {
    let train_s = c_train.samples();
    let val_s = c_val.samples();
    if train_s.is_empty() || val_s.is_empty() {
        return Err("no samples".into());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.threads.max(1))
        .build()
        .map_err(|e| e.to_string())?;
    let mut m = match init {
        Some(m) => {
            if m.n_in != INPUT_DIM || m.n_out != OUT_DIM {
                return Err("the starting network does not fit the v2 layout".into());
            }
            m.clone()
        }
        None => {
            let mut m = Mlp::new(INPUT_DIM, cfg.h1, cfg.h2, OUT_DIM, cfg.seed);
            init_biases(&mut m, c_train, &train_s);
            m
        }
    };
    let mut adam = Adam::new(m.params.len());
    let mut order: Vec<usize> = (0..train_s.len()).collect();
    let mut rng = cfg.seed ^ 0xDEAD_BEEF;
    let mut next = move || {
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let steps_per_epoch = train_s.len().div_ceil(cfg.batch);
    let total_steps = (steps_per_epoch * cfg.epochs).max(1);
    let mut step = 0usize;
    let mut best: Option<(f64, Mlp)> = None;
    let mut shards: Vec<Shard> = (0..SHARDS)
        .map(|_| Shard {
            grad: vec![0.0; m.params.len()],
            scratch: Scratch::new(&m),
            d: vec![0.0; OUT_DIM],
            x: Box::new([0.0; INPUT_DIM]),
            loss: 0.0,
            n: 0,
        })
        .collect();
    let mut grad = vec![0.0f32; m.params.len()];
    for epoch in 0..cfg.epochs {
        for i in (1..order.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let (mut epoch_loss, mut epoch_n) = (0.0f64, 0u64);
        for b in 0..steps_per_epoch {
            let batch = &order[b * cfg.batch..((b + 1) * cfg.batch).min(order.len())];
            let per = batch.len().div_ceil(SHARDS);
            let model = &m;
            pool.install(|| {
                shards.par_iter_mut().enumerate().for_each(|(si, sh)| {
                    sh.grad.fill(0.0);
                    sh.loss = 0.0;
                    sh.n = 0;
                    for &oi in batch.iter().skip(si * per).take(per) {
                        let r = train_s[oi];
                        let l = c_train.make(r, epoch as u64, &mut sh.x);
                        model.forward(&sh.x[..], &mut sh.scratch);
                        sh.loss += loss_and_grad(&sh.scratch.out, &l, &cfg.loss, src_weight(&cfg.loss, r), &mut sh.d);
                        model.backward(&sh.x[..], &mut sh.scratch, &sh.d, &mut sh.grad);
                        sh.n += 1;
                    }
                });
            });
            grad.fill(0.0);
            let mut n = 0u32;
            for sh in &shards {
                for (g, s) in grad.iter_mut().zip(&sh.grad) {
                    *g += s;
                }
                epoch_loss += sh.loss;
                n += sh.n;
            }
            epoch_n += u64::from(n);
            let inv = 1.0 / n.max(1) as f32;
            grad.iter_mut().for_each(|g| *g *= inv);
            let prog = step as f32 / total_steps as f32;
            let lr = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + (std::f32::consts::PI * prog).cos());
            adam.step(&mut m.params, &grad, lr, cfg.weight_decay);
            step += 1;
        }
        let val = pool.install(|| mean_loss(&m, c_val, &val_s, &cfg.loss));
        log(&format!(
            "epoch {:>2}: train loss {:.4}  val loss {:.4}",
            epoch + 1,
            epoch_loss / epoch_n.max(1) as f64,
            val
        ));
        if best.as_ref().is_none_or(|(b, _)| val < *b) {
            best = Some((val, m.clone()));
        }
    }
    let (val, model) = best.ok_or("no epochs")?;
    Ok((model, val))
}

/// What the offline evaluation counts for one head and tick.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cell {
    pub n: u64,
    pub model: u64,
    pub hold_snap: u64,
    /// Arena only: the true previous input held.
    pub n_true: u64,
    pub hold_true: u64,
    pub model_true: u64,
}

/// Fire head scores by tick: `(logit, label)`.
#[derive(Debug, Clone, Default)]
pub struct PressScores {
    pub by_k: [Vec<(f32, bool)>; HORIZON],
}

#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub samples: u64,
    pub dir: [Cell; HORIZON],
    pub hook: [Cell; HORIZON],
    pub jump: [Cell; HORIZON],
    /// Aim error (radians): model, hold (no change), count.
    pub aim: [(f64, f64, u64); HORIZON],
    pub press: PressScores,
}

impl Metrics {
    fn merge(&mut self, o: &Metrics) {
        self.samples += o.samples;
        for k in 0..HORIZON {
            for (a, b) in [
                (&mut self.dir[k], &o.dir[k]),
                (&mut self.hook[k], &o.hook[k]),
                (&mut self.jump[k], &o.jump[k]),
            ] {
                a.n += b.n;
                a.model += b.model;
                a.hold_snap += b.hold_snap;
                a.n_true += b.n_true;
                a.hold_true += b.hold_true;
                a.model_true += b.model_true;
            }
            self.aim[k].0 += o.aim[k].0;
            self.aim[k].1 += o.aim[k].1;
            self.aim[k].2 += o.aim[k].2;
            self.press.by_k[k].extend_from_slice(&o.press.by_k[k]);
        }
    }

    /// A markdown table: accuracy of the model against the snapshot's hold (and, in the arena, the true previous input) by head and tick; the fire head's ranking.
    pub fn table(&self) -> String {
        let pct = |a: u64, n: u64| if n == 0 { f64::NAN } else { 100.0 * a as f64 / n as f64 };
        let mut s = format!("samples: {}\n\n", self.samples);
        s.push_str("| head | k | n | model % | hold(snapshot) % | n (arena) | model % | hold(true) % |\n|---|---:|---:|---:|---:|---:|---:|---:|\n");
        for (name, cells) in [("direction", &self.dir), ("hook", &self.hook), ("jump", &self.jump)] {
            for k in 0..HORIZON {
                let c = &cells[k];
                if c.n == 0 {
                    continue;
                }
                s.push_str(&format!(
                    "| {name} | {k} | {} | {:.1} | {:.1} | {} | {:.1} | {:.1} |\n",
                    c.n,
                    pct(c.model, c.n),
                    pct(c.hold_snap, c.n),
                    c.n_true,
                    pct(c.model_true, c.n_true),
                    pct(c.hold_true, c.n_true)
                ));
            }
        }
        s.push_str("\n| aim | k | n | mean abs error model (rad) | hold (rad) |\n|---|---:|---:|---:|---:|\n");
        for k in 0..HORIZON {
            let (m, h, n) = self.aim[k];
            if n > 0 {
                s.push_str(&format!(
                    "| aim | {k} | {n} | {:.3} | {:.3} |\n",
                    m / n as f64,
                    h / n as f64
                ));
            }
        }
        s.push_str("\n| fire | k | n | swings | AUC | best F1 | at logit |\n|---|---:|---:|---:|---:|---:|---:|\n");
        for k in 0..HORIZON {
            let sc = &self.press.by_k[k];
            if sc.is_empty() {
                continue;
            }
            let (f1, thr) = best_f1(sc);
            s.push_str(&format!(
                "| fire | {k} | {} | {} | {:.3} | {:.3} | {:.2} |\n",
                sc.len(),
                sc.iter().filter(|x| x.1).count(),
                auc(sc),
                f1,
                thr
            ));
        }
        s
    }
}

/// Area under the ROC curve of `(score, label)` pairs (rank-sum with mid-ranks for ties).
pub fn auc(s: &[(f32, bool)]) -> f64 {
    let mut v: Vec<(f32, bool)> = s.to_vec();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut rank_sum, mut npos) = (0.0f64, 0u64);
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1].0 == v[i].0 {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for x in &v[i..=j] {
            if x.1 {
                rank_sum += r;
                npos += 1;
            }
        }
        i = j + 1;
    }
    let nneg = v.len() as u64 - npos;
    if npos == 0 || nneg == 0 {
        return f64::NAN;
    }
    (rank_sum - npos as f64 * (npos as f64 + 1.0) / 2.0) / (npos as f64 * nneg as f64)
}

/// The best F1 over thresholds and the logit that gives it.
pub fn best_f1(s: &[(f32, bool)]) -> (f64, f32) {
    let mut v: Vec<(f32, bool)> = s.to_vec();
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    let total = v.iter().filter(|x| x.1).count() as f64;
    let (mut tp, mut best, mut thr) = (0.0f64, 0.0f64, 0.0f32);
    for (i, &(z, y)) in v.iter().enumerate() {
        if y {
            tp += 1.0;
        }
        let pred = (i + 1) as f64;
        let f1 = if tp > 0.0 { 2.0 * tp / (pred + total) } else { 0.0 };
        if f1 > best {
            best = f1;
            thr = z;
        }
    }
    (best, thr)
}

/// Offline accuracy of `m` on `samples`, decoded with `dec` (the fire head is scored by its logits, not decoded).
pub fn evaluate(m: &Mlp, c: &Corpus, samples: &[SampleRef], dec: &Decode) -> Metrics {
    let parts: Vec<Metrics> = samples
        .par_chunks(2048)
        .map(|ch| {
            let mut met = Metrics::default();
            let mut s = Scratch::new(m);
            let mut x = Box::new([0.0f32; INPUT_DIM]);
            for &r in ch {
                let l = c.make(r, u64::MAX, &mut x);
                m.forward(&x[..], &mut s);
                let (prev, frame) = c.snapshot_of(r);
                met.samples += 1;
                let snap_hook = frame.hook_state > 0;
                let snap_dir = (i32::from(frame.direction) + 1) as u8;
                for k in 0..HORIZON {
                    let o = &s.out[k * HEAD_DIM..(k + 1) * HEAD_DIM];
                    let dir = super::predictor::decode_dir(o, dec.dir_margin, snap_dir);
                    if l.v_dir >> k & 1 != 0 {
                        let c = &mut met.dir[k];
                        c.n += 1;
                        c.model += u64::from(dir == l.dir[k]);
                        c.hold_snap += u64::from(snap_dir == l.dir[k]);
                        if let Some(p) = prev {
                            c.n_true += 1;
                            c.model_true += u64::from(dir == l.dir[k]);
                            c.hold_true += u64::from((i32::from(p.direction) + 1) as u8 == l.dir[k]);
                        }
                    }
                    if l.v_hook >> k & 1 != 0 {
                        let lh = l.hook >> k & 1 != 0;
                        let c = &mut met.hook[k];
                        c.n += 1;
                        c.model += u64::from((o[4] > dec.hook) == lh);
                        c.hold_snap += u64::from(snap_hook == lh);
                        if let Some(p) = prev {
                            c.n_true += 1;
                            c.model_true += u64::from((o[4] > dec.hook) == lh);
                            c.hold_true += u64::from(p.hook == lh);
                        }
                    }
                    if l.v_jump >> k & 1 != 0 {
                        let lj = l.jump >> k & 1 != 0;
                        let c = &mut met.jump[k];
                        c.n += 1;
                        c.model += u64::from((o[3] > dec.jump) == lj);
                        c.hold_snap += u64::from(!lj);
                        if let Some(p) = prev {
                            c.n_true += 1;
                            c.model_true += u64::from((o[3] > dec.jump) == lj);
                            c.hold_true += u64::from(p.jump == lj);
                        }
                    }
                    if l.v_aim >> k & 1 != 0 {
                        let a = &mut met.aim[k];
                        a.0 += f64::from((o[6] - l.aim_delta[k]).abs());
                        a.1 += f64::from(l.aim_delta[k].abs());
                        a.2 += 1;
                    }
                    if l.v_press >> k & 1 != 0 {
                        met.press.by_k[k].push((o[5], l.press >> k & 1 != 0));
                    }
                }
            }
            met
        })
        .collect();
    let mut all = Metrics::default();
    for p in &parts {
        all.merge(p);
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::Scratch;

    fn label() -> Label {
        Label {
            v_dir: 0b1111,
            v_jump: 0b0101,
            v_hook: 0b1111,
            v_press: 0b0011,
            v_aim: 0b1010,
            dir: [0, 1, 2, 1],
            jump: 0b0001,
            hook: 0b0110,
            press: 0b0010,
            aim_delta: [0.0, 0.1, -0.7, 0.0],
        }
    }

    fn out(seed: u32) -> Vec<f32> {
        (0..OUT_DIM)
            .map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0)
            .collect()
    }

    #[test]
    fn the_gradient_matches_finite_differences_and_masked_heads_get_none() {
        let cfg = LossCfg::default();
        let l = label();
        let o = out(3);
        let mut d = vec![0.0; OUT_DIM];
        let base = loss_and_grad(&o, &l, &cfg, 1.0, &mut d);
        let eps = 1e-3f32;
        for i in 0..OUT_DIM {
            let (mut a, mut b) = (o.clone(), o.clone());
            a[i] += eps;
            b[i] -= eps;
            let mut sink = vec![0.0; OUT_DIM];
            let num = ((loss_and_grad(&a, &l, &cfg, 1.0, &mut sink) - loss_and_grad(&b, &l, &cfg, 1.0, &mut sink))
                / (2.0 * f64::from(eps))) as f32;
            assert!((d[i] - num).abs() < 5e-3, "output {i}: {} vs {num}", d[i]);
        }
        // Tick 1 has no jump label (v_jump bit 1 clear) and tick 0 no aim label: those logits get no gradient.
        assert_eq!(d[HEAD_DIM + 3], 0.0);
        assert_eq!(d[6], 0.0);
        assert!(base.is_finite() && base > 0.0);
        let mut d2 = vec![0.0; OUT_DIM];
        let doubled = loss_and_grad(&o, &l, &cfg, 2.0, &mut d2);
        assert!(
            (doubled - 2.0 * base).abs() < 1e-5 * base,
            "the source weight scales the loss"
        );
    }

    #[test]
    fn a_positive_fire_label_weighs_more() {
        let l = Label {
            v_press: 0b1,
            press: 0b1,
            ..Label::default()
        };
        let o = vec![0.0f32; OUT_DIM];
        let (mut d1, mut d3) = (vec![0.0; OUT_DIM], vec![0.0; OUT_DIM]);
        loss_and_grad(
            &o,
            &l,
            &LossCfg {
                press_pos: 1.0,
                ..LossCfg::default()
            },
            1.0,
            &mut d1,
        );
        loss_and_grad(
            &o,
            &l,
            &LossCfg {
                press_pos: 3.0,
                ..LossCfg::default()
            },
            1.0,
            &mut d3,
        );
        assert!((d3[5] / d1[5] - 3.0).abs() < 1e-5);
        let neg = Label {
            v_press: 0b1,
            press: 0,
            ..Label::default()
        };
        let (mut e1, mut e3) = (vec![0.0; OUT_DIM], vec![0.0; OUT_DIM]);
        loss_and_grad(
            &o,
            &neg,
            &LossCfg {
                press_pos: 1.0,
                ..LossCfg::default()
            },
            1.0,
            &mut e1,
        );
        loss_and_grad(
            &o,
            &neg,
            &LossCfg {
                press_pos: 3.0,
                ..LossCfg::default()
            },
            1.0,
            &mut e3,
        );
        assert_eq!(e1[5], e3[5], "a negative label is not reweighted");
    }

    #[test]
    fn auc_and_f1_on_known_scores() {
        let s = [(0.9, true), (0.8, true), (0.3, false), (0.1, false)];
        assert!((auc(&s) - 1.0).abs() < 1e-12);
        let s = [(0.9, false), (0.8, true), (0.3, true), (0.1, false)];
        assert!((auc(&s) - 0.5).abs() < 1e-12);
        let ties = [(0.5, true), (0.5, false)];
        assert!((auc(&ties) - 0.5).abs() < 1e-12);
        assert!(auc(&[(1.0, true)]).is_nan());
        let (f1, thr) = best_f1(&[(0.9, true), (0.8, true), (0.3, false), (0.1, false)]);
        assert!((f1 - 1.0).abs() < 1e-12 && (thr - 0.8).abs() < 1e-6);
    }

    #[test]
    fn init_biases_set_the_base_rate_logits() {
        use crate::frame::{InputRec, N_RAYS, TeeFrame};
        use crate::v2::corpus::{Corpus, CorpusCfg};
        use crate::v2::data::{GameRec, TickRec};
        let f = TeeFrame {
            alive: true,
            ..TeeFrame::default()
        };
        // A press every 8th tick (at t % 8 == 4): window tick 1 of the even samples at t % 8 == 2 -> a quarter of them.
        let ticks = (0..400)
            .map(|t| TickRec {
                frames: [f, f],
                applied: [
                    InputRec::default(),
                    InputRec {
                        fire: if t % 8 == 4 {
                            1
                        } else if t % 8 == 5 {
                            2
                        } else {
                            0
                        },
                        ..InputRec::default()
                    },
                ],
                rays: [[1.0; N_RAYS]; 2],
            })
            .collect();
        let g = GameRec {
            arena: "t".into(),
            seed: 1,
            lag: [2, 0],
            swap: false,
            decide_every: 2,
            tick0: 0,
            ticks,
        };
        let c = Corpus::new(vec![g], vec![], CorpusCfg::default());
        let s = c.samples();
        let mut m = Mlp::new(INPUT_DIM, 8, 8, OUT_DIM, 1);
        init_biases(&mut m, &c, &s);
        let (_, _, _, _, _, b3) = m.offsets();
        let rate = |k: usize| 1.0 / (1.0 + (-f64::from(m.params[b3 + k * HEAD_DIM + 5])).exp());
        assert!((rate(1) - 0.25).abs() < 0.03, "press base rate at tick 1: {}", rate(1));
        assert!(rate(0) < 0.02, "no even-tick press at window tick 0: {}", rate(0));
        let mut sc = Scratch::new(&m);
        m.forward(&vec![0.0; INPUT_DIM], &mut sc);
        assert!(sc.out.iter().all(|v| v.is_finite()));
    }
}
