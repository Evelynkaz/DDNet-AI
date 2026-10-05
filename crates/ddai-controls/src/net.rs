//! What the trainer and the arena need from a control network, and the output layer both share.

use ddai_fly::bc::{HeadLogits, LossConfig, StepLoss, StepTargets};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetKind {
    Mlp,
    Gru,
}

impl NetKind {
    pub fn name(self) -> &'static str {
        match self {
            NetKind::Mlp => "mlp",
            NetKind::Gru => "gru",
        }
    }
}

/// The result of one window's forward + backward.
#[derive(Debug, Clone)]
pub struct WindowOut {
    pub loss: StepLoss,
    /// Sum of the step weights that carried a loss.
    pub weight_sum: f32,
    /// One entry per step of the window (burn-in steps included).
    pub logits: Vec<HeadLogits>,
}

/// A control network over flattened features. Parameters are one flat vector, so the trainer's
/// optimiser treats every model alike; a recurrent net starts each window from a zero state.
pub trait SeqNet: Send + Sync {
    fn kind(&self) -> NetKind;
    fn input_dim(&self) -> usize;
    fn hidden(&self) -> usize;
    fn num_params(&self) -> usize {
        self.params().len()
    }
    fn params(&self) -> &[f32];
    fn params_mut(&mut self) -> &mut [f32];
    /// Length of the recurrent state (`0` for a feed-forward net).
    fn state_size(&self) -> usize;
    /// One online step (updates `state`, which must be `state_size()` long).
    fn step(&self, state: &mut [f32], x: &[f32]) -> HeadLogits;
    /// Forward and backward over a window from a zero state; **adds** the parameter gradient of
    /// the summed (step-weighted) loss into `grad` (`num_params()` long).
    fn window_grad(&self, xs: &[Vec<f32>], targets: &[StepTargets], cfg: &LossConfig, grad: &mut [f32]) -> WindowOut;
    /// Forward only over a window from a zero state.
    fn window_logits(&self, xs: &[Vec<f32>]) -> Vec<HeadLogits> {
        let mut state = vec![0.0; self.state_size()];
        xs.iter().map(|x| self.step(&mut state, x)).collect()
    }
    fn clone_box(&self) -> Box<dyn SeqNet>;
}

/// Number of scalars the shared output layer produces.
pub const N_OUT: usize = 8;

/// Output layer layout inside a parameter vector: `[w (N_OUT x hidden)] [b (N_OUT)]`.
pub fn heads_param_len(hidden: usize) -> usize {
    N_OUT * hidden + N_OUT
}

fn to_logits(o: &[f32; N_OUT]) -> HeadLogits {
    HeadLogits {
        dir: [o[0], o[1], o[2]],
        jump: o[3],
        hook: o[4],
        fire: o[5],
        aim_c: o[6],
        aim_s: o[7],
    }
}

fn from_logits(l: &HeadLogits) -> [f32; N_OUT] {
    [l.dir[0], l.dir[1], l.dir[2], l.jump, l.hook, l.fire, l.aim_c, l.aim_s]
}

/// `logits = W h + b` for the output layer stored at the start of `p`.
pub fn heads_forward(p: &[f32], h: &[f32]) -> HeadLogits {
    let hidden = h.len();
    let (w, b) = p.split_at(N_OUT * hidden);
    let mut o = [0.0f32; N_OUT];
    for (k, ok) in o.iter_mut().enumerate() {
        *ok = b[k]
            + w[k * hidden..(k + 1) * hidden]
                .iter()
                .zip(h)
                .map(|(&a, &x)| a * x)
                .sum::<f32>();
    }
    to_logits(&o)
}

/// Backward of [`heads_forward`]: accumulates into `gp` (same layout as `p`) and returns `dL/dh`.
pub fn heads_backward(p: &[f32], h: &[f32], d: &HeadLogits, gp: &mut [f32]) -> Vec<f32> {
    let hidden = h.len();
    let (w, _) = p.split_at(N_OUT * hidden);
    let (gw, gb) = gp.split_at_mut(N_OUT * hidden);
    let d = from_logits(d);
    let mut dh = vec![0.0f32; hidden];
    for k in 0..N_OUT {
        gb[k] += d[k];
        for j in 0..hidden {
            gw[k * hidden + j] += d[k] * h[j];
            dh[j] += d[k] * w[k * hidden + j];
        }
    }
    dh
}

#[cfg(test)]
pub(crate) mod testutil {
    use ddai_fly::bc::{HeadMask, LossConfig, SoftTargets, StepTargets};

    pub fn targets(n: usize) -> Vec<StepTargets> {
        (0..n)
            .map(|t| StepTargets {
                dir: (t % 3) as u8,
                jump: t % 2 == 0,
                hook: t % 3 == 0,
                fire: t % 4 == 1,
                aim: 0.5 * t as f32 - 1.0,
                soft: (t % 2 == 1).then_some(SoftTargets {
                    dir: [0.2, 0.5, 0.3],
                    jump: 0.4,
                    hook: 0.6,
                    fire: 0.2,
                }),
                mask: HeadMask::ALL,
                weight: if t == 0 { 0.0 } else { 1.0 + 0.25 * t as f32 },
                hook_scale: 1.0,
            })
            .collect()
    }

    pub fn cfg() -> LossConfig {
        LossConfig {
            pos_weight: [1.5, 2.0, 1.2],
            ..LossConfig::default()
        }
    }

    /// Deterministic pseudo-random inputs.
    pub fn inputs(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|t| {
                (0..dim)
                    .map(|i| (((t * 31 + i * 17) % 23) as f32 / 23.0) - 0.4)
                    .collect()
            })
            .collect()
    }
}
