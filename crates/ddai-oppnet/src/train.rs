//! Samples from games, the loss, the offline metrics and the trainer.
//!
//! A **sample** is a decision tick `T` of a game (`T % decide_every == 0`) with our window length `lag` (the game's own lag, so the
//! in-flight inputs are exactly the ones our client had sent). Its input is built by the same [`assemble`] the brain uses; its label
//! says what the opponent applied at `T .. T + HORIZON`.

use rayon::prelude::*;

use crate::data::GameRec;
use crate::feature::{
    FD, HEAD_DIM, HORIZON, IF_DIM, INPUT_DIM, Label, OUT_DIM, STRIDE, assemble, frame_features, inflight_features,
    label_tick,
};
use crate::frame::{InputRec, N_RAYS};
use crate::net::{Mlp, Scratch};

/// A sample's place: game and tick index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleRef {
    pub game: u32,
    pub idx: u32,
}

/// Games plus the per-tick frame features, computed once.
pub struct Corpus {
    pub games: Vec<GameRec>,
    feats: Vec<Vec<[f32; FD]>>,
}

impl Corpus {
    pub fn new(games: Vec<GameRec>) -> Corpus {
        let feats = games
            .par_iter()
            .map(|g| {
                g.ticks
                    .iter()
                    .map(|t| {
                        let mut f = [0.0; FD];
                        // The pair seen from the opponent's seat: `me` is us (slot 0), `opp` slot 1.
                        frame_features(&t.frames[0], &t.frames[1], &mut f);
                        f
                    })
                    .collect()
            })
            .collect();
        Corpus { games, feats }
    }

    /// Mean of every frame feature over all recorded ticks of the opponent seen alive (a diagnostic: compare corpora, e.g. the arena's and the live clips').
    pub fn mean_features(&self) -> Vec<f64> {
        let mut sum = vec![0.0f64; FD];
        let mut n = 0u64;
        for (g, f) in self.games.iter().zip(&self.feats) {
            for (t, v) in g.ticks.iter().zip(f) {
                if t.frames[1].alive {
                    n += 1;
                    for (s, x) in sum.iter_mut().zip(v) {
                        *s += f64::from(*x);
                    }
                }
            }
        }
        sum.iter()
            .map(|s| (s / n.max(1) as f64 * 1000.0).round() / 1000.0)
            .collect()
    }

    /// Every usable decision tick of games with a window (`lag >= 1`): the in-flight inputs exist and the first tick has a label.
    pub fn samples(&self) -> Vec<SampleRef> {
        let mut out = Vec::new();
        for (gi, g) in self.games.iter().enumerate() {
            let lag = usize::from(g.lag[0]);
            if lag == 0 {
                continue;
            }
            let de = i32::from(g.decide_every.max(1));
            for i in 0..g.ticks.len() {
                let t = g.tick0 + i as i32;
                if t % de == 0 && i + lag < g.ticks.len() && i + 1 < g.ticks.len() {
                    out.push(SampleRef {
                        game: gi as u32,
                        idx: i as u32,
                    });
                }
            }
        }
        out
    }

    /// Builds the input and the label of a sample.
    pub fn make(&self, r: SampleRef, x: &mut [f32; INPUT_DIM]) -> Label {
        let g = &self.games[r.game as usize];
        let i = r.idx as usize;
        let feats = &self.feats[r.game as usize];
        let lag = usize::from(g.lag[0]);
        let mut inflight = [[0.0f32; IF_DIM]; crate::feature::IF_SLOTS];
        let n_if = lag.min(crate::feature::IF_SLOTS);
        for (k, f) in inflight.iter_mut().enumerate().take(n_if) {
            inflight_features(&g.ticks[i + 1 + k].applied[0], f);
        }
        assemble(
            x,
            |j| &feats[i.saturating_sub(j * STRIDE)],
            &g.ticks[i].rays,
            &inflight[..n_if],
            lag,
        );
        self.label(r)
    }

    /// The label alone.
    pub fn label(&self, r: SampleRef) -> Label {
        let g = &self.games[r.game as usize];
        let i = r.idx as usize;
        let mut label = Label::default();
        let base = f64::from(g.ticks[i].frames[1].angle);
        let mut prev = g.ticks[i].applied[1];
        for k in 0..HORIZON {
            let Some(next) = g.ticks.get(i + 1 + k) else { break };
            let cur = next.applied[1];
            // A frozen or dead opponent's inputs do nothing: no label.
            let f = &next.frames[1];
            if f.alive && f.freeze_left == 0 {
                label_tick(&mut label, k, &prev, &cur, base);
            }
            prev = cur;
        }
        label
    }

    /// What the opponent applied just before the snapshot (`T - 1`) and the frame the snapshot shows.
    pub fn snapshot_of(&self, r: SampleRef) -> (InputRec, crate::frame::TeeFrame) {
        let g = &self.games[r.game as usize];
        let t = &g.ticks[r.idx as usize];
        (t.applied[1], t.frames[1])
    }

    pub fn rays_of(&self, r: SampleRef) -> [f32; N_RAYS] {
        self.games[r.game as usize].ticks[r.idx as usize].rays
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Loss weights.
#[derive(Debug, Clone, Copy)]
pub struct LossCfg {
    pub w_dir: f32,
    pub w_jump: f32,
    pub w_hook: f32,
    pub w_press: f32,
    pub w_aim: f32,
    /// Huber threshold of the aim loss (radians).
    pub huber: f32,
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
        }
    }
}

/// Loss of one sample from the logits `out`, and its derivative into `d`. Returns the loss summed over the valid ticks.
pub fn loss_and_grad(out: &[f32], l: &Label, cfg: &LossCfg, d: &mut [f32]) -> f64 {
    d.fill(0.0);
    let mut total = 0.0f64;
    for k in 0..HORIZON {
        if l.valid >> k & 1 == 0 {
            continue;
        }
        let o = &out[k * HEAD_DIM..(k + 1) * HEAD_DIM];
        let dd = &mut d[k * HEAD_DIM..(k + 1) * HEAD_DIM];
        // direction: softmax cross-entropy
        let m = o[0].max(o[1]).max(o[2]);
        let e = [(o[0] - m).exp(), (o[1] - m).exp(), (o[2] - m).exp()];
        let z = e[0] + e[1] + e[2];
        for c in 0..3 {
            let p = e[c] / z;
            let y = f32::from(u8::from(usize::from(l.dir[k]) == c));
            dd[c] = cfg.w_dir * (p - y);
        }
        total += f64::from(cfg.w_dir) * f64::from(-(e[usize::from(l.dir[k])] / z).max(1e-12).ln());
        // flags: binary cross-entropy with logits
        for (slot, bits, w) in [
            (3, l.jump, cfg.w_jump),
            (4, l.hook, cfg.w_hook),
            (5, l.press, cfg.w_press),
        ] {
            let y = f32::from(u8::from(bits >> k & 1 != 0));
            let z = o[slot];
            dd[slot] = w * (sigmoid(z) - y);
            // softplus(z) - y z, stable
            total += f64::from(w) * f64::from(z.max(0.0) + (-z.abs()).exp().ln_1p() - y * z);
        }
        // aim change: Huber
        let diff = o[6] - l.aim_delta[k];
        let a = diff.abs();
        if a <= cfg.huber {
            total += f64::from(cfg.w_aim) * 0.5 * f64::from(diff * diff);
            dd[6] = cfg.w_aim * diff;
        } else {
            total += f64::from(cfg.w_aim) * f64::from(cfg.huber) * f64::from(a - 0.5 * cfg.huber);
            dd[6] = cfg.w_aim * cfg.huber * diff.signum();
        }
    }
    total
}

/// One decoded tick of a prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decoded {
    pub dir: u8,
    pub jump: bool,
    pub hook: bool,
    pub press: bool,
    pub aim_delta: f32,
}

pub fn decode_tick(out: &[f32], k: usize) -> Decoded {
    decode_tick_gated(out, k, None, 0.0)
}

/// What a snapshot alone shows of the opponent, the fallback of a gated prediction ("hold": the direction it shows, the hook while its state is not idle, no
/// jump, no press, the aim it shows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoldView {
    /// Direction class `0, 1, 2` for `-1, 0, 1`.
    pub dir: u8,
    pub hook: bool,
}

/// [`decode_tick`] with a confidence gate: a head whose margin (the best logit minus the second best for the direction, `|logit|` for a flag) is below `margin`
/// answers with `hold` instead. The aim change (a regression, with no confidence) is never gated. `margin = 0` or no `hold` gates nothing. Logit margins, not probabilities: no `exp`, so the decision is bit-identical everywhere.
pub fn decode_tick_gated(out: &[f32], k: usize, hold: Option<HoldView>, margin: f32) -> Decoded {
    let o = &out[k * HEAD_DIM..(k + 1) * HEAD_DIM];
    let (best, second) = if o[0] >= o[1] && o[0] >= o[2] {
        (0, o[1].max(o[2]))
    } else if o[1] >= o[2] {
        (1, o[0].max(o[2]))
    } else {
        (2, o[0].max(o[1]))
    };
    let mut d = Decoded {
        dir: best as u8,
        jump: o[3] > 0.0,
        hook: o[4] > 0.0,
        press: o[5] > 0.0,
        aim_delta: o[6],
    };
    if let Some(h) = hold
        && margin > 0.0
    {
        if o[best] - second < margin {
            d.dir = h.dir;
        }
        if o[3].abs() < margin {
            d.jump = false;
        }
        if o[4].abs() < margin {
            d.hook = h.hook;
        }
        if o[5].abs() < margin {
            d.press = false;
        }
    }
    d
}

/// Adam state.
pub struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
}

impl Adam {
    pub fn new(n: usize) -> Adam {
        Adam {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }

    pub fn step(&mut self, params: &mut [f32], grad: &[f32], lr: f32, wd: f32) {
        self.t += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let c1 = 1.0 - b1.powi(self.t as i32);
        let c2 = 1.0 - b2.powi(self.t as i32);
        for i in 0..params.len() {
            let g = grad[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let mh = self.m[i] / c1;
            let vh = self.v[i] / c2;
            params[i] -= lr * (mh / (vh.sqrt() + eps) + wd * params[i]);
        }
    }
}

/// Trainer settings.
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
            h1: 160,
            h2: 128,
            epochs: 12,
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

/// Fixed number of gradient shards per batch: the sums are taken in shard order, so the result does not depend on the thread count.
const SHARDS: usize = 4;

struct Shard {
    grad: Vec<f32>,
    scratch: Scratch,
    d: Vec<f32>,
    x: Box<[f32; INPUT_DIM]>,
    loss: f64,
    n: u32,
}

/// Mean loss of `samples` (forward only).
pub fn mean_loss(m: &Mlp, c: &Corpus, samples: &[SampleRef], cfg: &LossCfg) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let chunks: Vec<&[SampleRef]> = samples.chunks(2048).collect();
    let sums: Vec<f64> = chunks
        .par_iter()
        .map(|ch| {
            let mut s = Scratch::new(m);
            let mut d = vec![0.0; OUT_DIM];
            let mut x = Box::new([0.0f32; INPUT_DIM]);
            let mut tot = 0.0;
            for &r in *ch {
                let l = c.make(r, &mut x);
                m.forward(&x[..], &mut s);
                tot += loss_and_grad(&s.out, &l, cfg, &mut d);
            }
            tot
        })
        .collect();
    sums.iter().sum::<f64>() / samples.len() as f64
}

/// Sets the flag heads' biases to the logit of their base rates in `samples`.
fn init_biases(m: &mut Mlp, c: &Corpus, samples: &[SampleRef]) {
    let mut pos = [[0u64; 3]; HORIZON];
    let mut n = [0u64; HORIZON];
    for &r in samples {
        let l = c.label(r);
        for k in 0..HORIZON {
            if l.valid >> k & 1 != 0 {
                n[k] += 1;
                for (s, bits) in [l.jump, l.hook, l.press].into_iter().enumerate() {
                    pos[k][s] += u64::from(bits >> k & 1);
                }
            }
        }
    }
    let (_, _, _, _, _, b3) = m.offsets();
    for k in 0..HORIZON {
        for (s, &hits) in pos[k].iter().enumerate() {
            let p = (hits as f64 + 1.0) / (n[k] as f64 + 2.0);
            m.params[b3 + k * HEAD_DIM + 3 + s] = (p / (1.0 - p)).ln() as f32;
        }
    }
}

/// Trains a network on `train`, keeping the one with the best loss on `val`. `log` gets one line per epoch.
pub fn train(
    c_train: &Corpus,
    c_val: &Corpus,
    cfg: &TrainCfg,
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
    let mut m = Mlp::new(INPUT_DIM, cfg.h1, cfg.h2, OUT_DIM, cfg.seed);
    init_biases(&mut m, c_train, &train_s);
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
        // Fisher-Yates with the seeded generator.
        for i in (1..order.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let mut epoch_loss = 0.0f64;
        let mut epoch_n = 0u64;
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
                        let l = c_train.make(train_s[oi], &mut sh.x);
                        model.forward(&sh.x[..], &mut sh.scratch);
                        sh.loss += loss_and_grad(&sh.scratch.out, &l, &cfg.loss, &mut sh.d);
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

/// Per-head accuracies of the model and the two baselines, by tick of the window.
#[derive(Debug, Clone, Default)]
pub struct HeadStats {
    pub n: [u64; HORIZON],
    pub model: [u64; HORIZON],
    pub hold_true: [u64; HORIZON],
    pub hold_snap: [u64; HORIZON],
    /// Of the ticks where the label differs from the true hold: how many each got right.
    pub n_change: [u64; HORIZON],
    pub model_change: [u64; HORIZON],
}

#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub samples: u64,
    pub dir: HeadStats,
    pub jump: HeadStats,
    pub hook: HeadStats,
    pub press: HeadStats,
    /// All of direction, jump and hook right at once (what moves the opponent).
    pub moves: HeadStats,
    /// Mean absolute aim-angle error (radians) summed over valid ticks, by tick: model, hold (no change).
    pub aim_model: [f64; HORIZON],
    pub aim_hold: [f64; HORIZON],
    pub aim_n: [u64; HORIZON],
}

impl Metrics {
    fn merge_head(a: &mut HeadStats, b: &HeadStats) {
        for k in 0..HORIZON {
            a.n[k] += b.n[k];
            a.model[k] += b.model[k];
            a.hold_true[k] += b.hold_true[k];
            a.hold_snap[k] += b.hold_snap[k];
            a.n_change[k] += b.n_change[k];
            a.model_change[k] += b.model_change[k];
        }
    }

    fn merge(&mut self, o: &Metrics) {
        self.samples += o.samples;
        Self::merge_head(&mut self.dir, &o.dir);
        Self::merge_head(&mut self.jump, &o.jump);
        Self::merge_head(&mut self.hook, &o.hook);
        Self::merge_head(&mut self.press, &o.press);
        Self::merge_head(&mut self.moves, &o.moves);
        for k in 0..HORIZON {
            self.aim_model[k] += o.aim_model[k];
            self.aim_hold[k] += o.aim_hold[k];
            self.aim_n[k] += o.aim_n[k];
        }
    }

    /// One markdown row per tick of the window: the accuracy of the model (`m`), of true hold (`t`) and of snapshot hold (`s`) for direction, jump, hook and all
    /// three at once.
    pub fn compact(&self) -> String {
        let pct = |a: u64, n: u64| if n == 0 { f64::NAN } else { 100.0 * a as f64 / n as f64 };
        let mut s = String::from(
            "| k | dir m / t / s | jump m / t / s | hook m / t / s | dir+jump+hook m / t / s |\n|---:|---|---|---|---|\n",
        );
        for k in 0..HORIZON {
            let cell = |h: &HeadStats| {
                format!(
                    "{:.1} / {:.1} / {:.1}",
                    pct(h.model[k], h.n[k]),
                    pct(h.hold_true[k], h.n[k]),
                    pct(h.hold_snap[k], h.n[k])
                )
            };
            s.push_str(&format!(
                "| {k} | {} | {} | {} | {} |\n",
                cell(&self.dir),
                cell(&self.jump),
                cell(&self.hook),
                cell(&self.moves)
            ));
        }
        s
    }

    /// A markdown table: per head and tick, accuracy of the model against the two holds.
    pub fn table(&self) -> String {
        let pct = |a: u64, n: u64| if n == 0 { f64::NAN } else { 100.0 * a as f64 / n as f64 };
        let mut s = format!("samples: {}\n\n", self.samples);
        s.push_str("| head | k | n | model % | hold(true) % | hold(snapshot) % | changes n | model on changes % |\n|---|---:|---:|---:|---:|---:|---:|---:|\n");
        for (name, h) in [
            ("direction", &self.dir),
            ("jump", &self.jump),
            ("hook", &self.hook),
            ("press", &self.press),
            ("dir+jump+hook", &self.moves),
        ] {
            for k in 0..HORIZON {
                s.push_str(&format!(
                    "| {name} | {k} | {} | {:.1} | {:.1} | {:.1} | {} | {:.1} |\n",
                    h.n[k],
                    pct(h.model[k], h.n[k]),
                    pct(h.hold_true[k], h.n[k]),
                    pct(h.hold_snap[k], h.n[k]),
                    h.n_change[k],
                    pct(h.model_change[k], h.n_change[k]),
                ));
            }
        }
        s.push_str("\n| aim | k | mean abs error model (rad) | hold (rad) |\n|---|---:|---:|---:|\n");
        for k in 0..HORIZON {
            let n = self.aim_n[k].max(1) as f64;
            s.push_str(&format!(
                "| aim | {k} | {:.3} | {:.3} |\n",
                self.aim_model[k] / n,
                self.aim_hold[k] / n
            ));
        }
        s
    }
}

/// Offline accuracy of `m` on `samples` against "hold the last input" (the opponent's true previous input) and against what the brain
/// holds today (`enemy_input_from_tee`: the direction the snapshot shows, the hook while its state is not idle, no jump, no fire).
pub fn evaluate(m: &Mlp, c: &Corpus, samples: &[SampleRef]) -> Metrics {
    evaluate_gated(m, c, samples, 0.0)
}

/// [`evaluate`] with the model's heads gated by a logit margin (see [`decode_tick_gated`]): a head less sure than `margin` answers like "hold (snapshot)".
pub fn evaluate_gated(m: &Mlp, c: &Corpus, samples: &[SampleRef], margin: f32) -> Metrics {
    let parts: Vec<Metrics> = samples
        .par_chunks(2048)
        .map(|ch| {
            let mut met = Metrics::default();
            let mut s = Scratch::new(m);
            let mut x = Box::new([0.0f32; INPUT_DIM]);
            for &r in ch {
                let l = c.make(r, &mut x);
                m.forward(&x[..], &mut s);
                let (prev, frame) = c.snapshot_of(r);
                met.samples += 1;
                let snap_hook = frame.hook_state > 0;
                for k in 0..HORIZON {
                    if l.valid >> k & 1 == 0 {
                        continue;
                    }
                    let snap_dir = (i32::from(frame.direction) + 1) as u8;
                    let d = decode_tick_gated(
                        &s.out,
                        k,
                        Some(HoldView {
                            dir: snap_dir,
                            hook: snap_hook,
                        }),
                        margin,
                    );
                    let (lj, lh, lp) = (l.jump >> k & 1 != 0, l.hook >> k & 1 != 0, l.press >> k & 1 != 0);
                    let dir_ok = |x: u8| x == l.dir[k];
                    let hold_dir = (i32::from(prev.direction) + 1) as u8;
                    let put = |h: &mut HeadStats, model: bool, hold: bool, snap: bool| {
                        h.n[k] += 1;
                        h.model[k] += u64::from(model);
                        h.hold_true[k] += u64::from(hold);
                        h.hold_snap[k] += u64::from(snap);
                        if !hold {
                            h.n_change[k] += 1;
                            h.model_change[k] += u64::from(model);
                        }
                    };
                    put(&mut met.dir, dir_ok(d.dir), dir_ok(hold_dir), dir_ok(snap_dir));
                    put(&mut met.jump, d.jump == lj, prev.jump == lj, !lj);
                    put(&mut met.hook, d.hook == lh, prev.hook == lh, snap_hook == lh);
                    put(&mut met.press, d.press == lp, !lp, !lp);
                    put(
                        &mut met.moves,
                        dir_ok(d.dir) && d.jump == lj && d.hook == lh,
                        dir_ok(hold_dir) && prev.jump == lj && prev.hook == lh,
                        dir_ok(snap_dir) && !lj && snap_hook == lh,
                    );
                    met.aim_model[k] += f64::from((d.aim_delta - l.aim_delta[k]).abs());
                    met.aim_hold[k] += f64::from(l.aim_delta[k].abs());
                    met.aim_n[k] += 1;
                }
            }
            met
        })
        .collect();
    let mut total = Metrics::default();
    for p in &parts {
        total.merge(p);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TickRec;
    use crate::frame::TeeFrame;

    /// A game in which slot 1 holds direction `+1` for 4 ticks, then `-1` for 4, and so on (period 8), seen from decision ticks.
    fn game(lag: u8, n: usize) -> GameRec {
        let ticks = (0..n)
            .map(|j| {
                let dir = if (j / 4) % 2 == 0 { 1 } else { -1 };
                let mut f1 = TeeFrame {
                    alive: true,
                    direction: dir as i8,
                    angle: 1.0,
                    ..Default::default()
                };
                f1.pos = [j as f32, 0.0];
                TickRec {
                    frames: [
                        TeeFrame {
                            alive: true,
                            ..Default::default()
                        },
                        f1,
                    ],
                    applied: [
                        InputRec {
                            direction: 0,
                            ..Default::default()
                        },
                        InputRec {
                            direction: dir as i8,
                            target_x: 300,
                            target_y: 0,
                            ..Default::default()
                        },
                    ],
                    rays: [1.0; N_RAYS],
                }
            })
            .collect();
        GameRec {
            arena: "t".into(),
            seed: 1,
            lag: [lag, 0],
            swap: false,
            decide_every: 2,
            tick0: 1,
            ticks,
        }
    }

    #[test]
    fn samples_are_decision_ticks_with_their_window_inside_the_game() {
        let c = Corpus::new(vec![game(3, 40), game(0, 40)]);
        let s = c.samples();
        assert!(!s.is_empty());
        assert!(s.iter().all(|r| r.game == 0), "a game with no lag has no window");
        for r in &s {
            let g = &c.games[0];
            assert_eq!((g.tick0 + r.idx as i32) % 2, 0);
            assert!(r.idx as usize + 3 < g.ticks.len());
        }
    }

    #[test]
    fn the_label_reads_the_inputs_applied_from_the_snapshot_tick_on() {
        let c = Corpus::new(vec![game(3, 40)]);
        let s = c.samples();
        // idx 7 is tick 8: the input applied in step 8 is `applied` of ticks[8], direction by (8 / 4) % 2.
        let r = *s.iter().find(|r| r.idx == 7).unwrap();
        let l = c.label(r);
        assert_eq!(l.valid, 0xff);
        for k in 0..HORIZON {
            // ticks[8 + k] holds the input applied in the step of tick 7 + k + 1... see `GameRec`: index j = applied in the step of tick0 + j - 1.
            let j = 7 + 1 + k;
            let want = if (j / 4) % 2 == 0 { 2 } else { 0 };
            assert_eq!(l.dir[k], want, "k = {k}");
        }
    }

    #[test]
    fn a_frozen_opponent_has_no_label() {
        let mut g = game(3, 40);
        for t in g.ticks.iter_mut().skip(12) {
            t.frames[1].freeze_left = 30;
        }
        let c = Corpus::new(vec![g]);
        let r = *c.samples().iter().find(|r| r.idx == 7).unwrap();
        let l = c.label(r);
        // ticks[8..=11] are free (k = 0..=3); from ticks[12] (k = 4) on the opponent is frozen.
        assert_eq!(l.valid, 0b0000_1111);
    }

    #[test]
    fn the_loss_gradient_matches_finite_differences() {
        let mut l = Label::default();
        for k in [0usize, 3, 7] {
            label_tick(
                &mut l,
                k,
                &InputRec::default(),
                &InputRec {
                    direction: 1,
                    jump: k == 3,
                    hook: true,
                    fire: 1,
                    target_x: 100,
                    target_y: 200,
                },
                0.2,
            );
        }
        let cfg = LossCfg::default();
        let out: Vec<f32> = (0..OUT_DIM).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.17).collect();
        let mut d = vec![0.0; OUT_DIM];
        loss_and_grad(&out, &l, &cfg, &mut d);
        let mut scratch = vec![0.0; OUT_DIM];
        for i in 0..OUT_DIM {
            let (mut a, mut b) = (out.clone(), out.clone());
            a[i] += 1e-2;
            b[i] -= 1e-2;
            let num =
                ((loss_and_grad(&a, &l, &cfg, &mut scratch) - loss_and_grad(&b, &l, &cfg, &mut scratch)) / 2e-2) as f32;
            assert!((d[i] - num).abs() < 2e-3, "logit {i}: {} vs {num}", d[i]);
        }
    }

    #[test]
    fn a_net_learns_a_periodic_opponent_and_beats_hold() {
        let games: Vec<GameRec> = (0..6).map(|_| game(3, 160)).collect();
        let c = Corpus::new(games);
        let cfg = TrainCfg {
            h1: 32,
            h2: 16,
            epochs: 25,
            batch: 64,
            lr: 3e-3,
            threads: 2,
            ..TrainCfg::default()
        };
        let (m, val) = train(&c, &c, &cfg, &mut |_| {}).unwrap();
        assert!(val.is_finite());
        let met = evaluate(&m, &c, &c.samples());
        // The period is 8 ticks, so hold is wrong a quarter of the time over a 8-tick window; the net knows the phase.
        let (mut model, mut hold, mut n) = (0, 0, 0);
        for k in 0..HORIZON {
            model += met.dir.model[k];
            hold += met.dir.hold_true[k];
            n += met.dir.n[k];
        }
        assert!(model as f64 / n as f64 > 0.9, "model {model}/{n}");
        assert!(model > hold + n / 10, "model {model} vs hold {hold} of {n}");
    }

    #[test]
    fn training_is_deterministic_and_independent_of_the_thread_count() {
        let c = Corpus::new((0..3).map(|_| game(3, 100)).collect());
        let run = |threads: usize| {
            let cfg = TrainCfg {
                h1: 16,
                h2: 8,
                epochs: 2,
                batch: 32,
                threads,
                ..TrainCfg::default()
            };
            train(&c, &c, &cfg, &mut |_| {}).unwrap().0
        };
        assert_eq!(run(1), run(3));
    }
}
