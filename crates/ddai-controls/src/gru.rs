//! The GRU control: one GRU layer over the flattened features, linear heads on the hidden state.
//!
//! Gates (PyTorch's convention, with the reset gate applied to the recurrent candidate term):
//! ```text
//! z = sigmoid(Wz x + bz + Uz h)      r = sigmoid(Wr x + br + Ur h)
//! n = tanh(Wn x + bn + r * (Un h + cn))      h' = (1 - z) * n + z * h
//! ```
//! Parameter layout: `[Wx (3H x D)] [Uh (3H x H)] [bx (3H)] [bh (3H)] [heads: W (8 x H), b (8)]`,
//! gate order `z, r, n` (`bh` holds only `cn` in its last third; the first two thirds are unused
//! zeros kept so the layout is uniform and the count is the textbook `3H(D + H + 2)`).

use ddai_fly::activation::sigmoid;
use ddai_fly::bc::{HeadLogits, LossConfig, StepLoss, StepTargets, head_loss_and_grad};

use crate::net::{NetKind, SeqNet, WindowOut, heads_backward, heads_forward, heads_param_len};

#[derive(Debug, Clone)]
pub struct Gru {
    input: usize,
    hidden: usize,
    p: Vec<f32>,
}

/// Saved activations of one step, for the backward pass.
struct StepCache {
    h_prev: Vec<f32>,
    z: Vec<f32>,
    r: Vec<f32>,
    n: Vec<f32>,
    /// `Un h_prev + cn` (before the reset gate).
    hn: Vec<f32>,
    h: Vec<f32>,
}

impl Gru {
    pub fn param_count(input: usize, hidden: usize) -> usize {
        3 * hidden * input + 3 * hidden * hidden + 3 * hidden + 3 * hidden + heads_param_len(hidden)
    }

    pub fn new(input: usize, hidden: usize, seed: u64) -> Self {
        let mut rng = ddai_fly::rng::SplitMix64::new(seed ^ 0x475255);
        let mut p = vec![0.0f32; Self::param_count(input, hidden)];
        let sx = (1.0 / input as f32).sqrt();
        let sh = (1.0 / hidden as f32).sqrt();
        let (wx, rest) = p.split_at_mut(3 * hidden * input);
        for v in wx {
            *v = (rng.next_f32_unit() * 2.0 - 1.0) * sx;
        }
        let (uh, rest) = rest.split_at_mut(3 * hidden * hidden);
        for v in uh {
            *v = (rng.next_f32_unit() * 2.0 - 1.0) * sh;
        }
        let heads = &mut rest[6 * hidden..];
        for v in &mut heads[..8 * hidden] {
            *v = (rng.next_f32_unit() * 2.0 - 1.0) * sh;
        }
        Gru { input, hidden, p }
    }

    pub fn from_params(input: usize, hidden: usize, p: Vec<f32>) -> Option<Self> {
        (p.len() == Self::param_count(input, hidden)).then_some(Gru { input, hidden, p })
    }

    fn offsets(&self) -> (usize, usize, usize, usize, usize) {
        let (h, d) = (self.hidden, self.input);
        let uh = 3 * h * d;
        let bx = uh + 3 * h * h;
        let bh = bx + 3 * h;
        let heads = bh + 3 * h;
        (0, uh, bx, bh, heads)
    }

    fn cell(&self, h_prev: &[f32], x: &[f32]) -> StepCache {
        let (h, d) = (self.hidden, self.input);
        let (wx_o, uh_o, bx_o, bh_o, _) = self.offsets();
        let p = &self.p;
        let mut z = vec![0.0; h];
        let mut r = vec![0.0; h];
        let mut n = vec![0.0; h];
        let mut hn = vec![0.0; h];
        let mut new_h = vec![0.0; h];
        for j in 0..h {
            let dot_x = |gate: usize| {
                let row = &p[wx_o + (gate * h + j) * d..wx_o + (gate * h + j + 1) * d];
                row.iter().zip(x).map(|(&w, &v)| w * v).sum::<f32>() + p[bx_o + gate * h + j]
            };
            let dot_h = |gate: usize| {
                let row = &p[uh_o + (gate * h + j) * h..uh_o + (gate * h + j + 1) * h];
                row.iter().zip(h_prev).map(|(&w, &v)| w * v).sum::<f32>()
            };
            z[j] = sigmoid(dot_x(0) + dot_h(0));
            r[j] = sigmoid(dot_x(1) + dot_h(1));
            hn[j] = dot_h(2) + p[bh_o + 2 * h + j];
            n[j] = (dot_x(2) + r[j] * hn[j]).tanh();
            new_h[j] = (1.0 - z[j]) * n[j] + z[j] * h_prev[j];
        }
        StepCache {
            h_prev: h_prev.to_vec(),
            z,
            r,
            n,
            hn,
            h: new_h,
        }
    }
}

impl SeqNet for Gru {
    fn kind(&self) -> NetKind {
        NetKind::Gru
    }
    fn input_dim(&self) -> usize {
        self.input
    }
    fn hidden(&self) -> usize {
        self.hidden
    }
    fn params(&self) -> &[f32] {
        &self.p
    }
    fn params_mut(&mut self) -> &mut [f32] {
        &mut self.p
    }
    fn state_size(&self) -> usize {
        self.hidden
    }

    fn step(&self, state: &mut [f32], x: &[f32]) -> HeadLogits {
        assert_eq!(x.len(), self.input, "gru: input length");
        assert_eq!(state.len(), self.hidden, "gru: state length");
        let c = self.cell(state, x);
        state.copy_from_slice(&c.h);
        heads_forward(&self.p[self.offsets().4..], &c.h)
    }

    fn window_grad(&self, xs: &[Vec<f32>], targets: &[StepTargets], cfg: &LossConfig, grad: &mut [f32]) -> WindowOut {
        assert_eq!(xs.len(), targets.len());
        assert_eq!(grad.len(), self.p.len());
        let (h, d) = (self.hidden, self.input);
        let (wx_o, uh_o, bx_o, bh_o, heads_o) = self.offsets();
        let t_len = xs.len();

        // Forward, keeping every step's activations.
        let mut caches: Vec<StepCache> = Vec::with_capacity(t_len);
        let mut h_prev = vec![0.0f32; h];
        let mut logits = Vec::with_capacity(t_len);
        for x in xs {
            let c = self.cell(&h_prev, x);
            h_prev.clone_from(&c.h);
            logits.push(heads_forward(&self.p[heads_o..], &c.h));
            caches.push(c);
        }

        // Backward through time.
        let mut loss = StepLoss::default();
        let mut weight_sum = 0.0f32;
        let mut dh_next = vec![0.0f32; h];
        for t in (0..t_len).rev() {
            let c = &caches[t];
            let tgt = &targets[t];
            let mut dh = dh_next.clone();
            if tgt.weight > 0.0 {
                let (l, dl) = head_loss_and_grad(&logits[t], tgt, cfg);
                loss.add(&l);
                weight_sum += tgt.weight;
                let dh_heads = heads_backward(&self.p[heads_o..], &c.h, &dl, &mut grad[heads_o..]);
                for (a, b) in dh.iter_mut().zip(&dh_heads) {
                    *a += b;
                }
            }
            let x = &xs[t];
            let mut dh_prev = vec![0.0f32; h];
            for j in 0..h {
                let dz = dh[j] * (c.h_prev[j] - c.n[j]);
                let dn = dh[j] * (1.0 - c.z[j]);
                dh_prev[j] += dh[j] * c.z[j];
                let d_n_pre = dn * (1.0 - c.n[j] * c.n[j]);
                let dr = d_n_pre * c.hn[j];
                let d_hn = d_n_pre * c.r[j];
                let d_z_pre = dz * c.z[j] * (1.0 - c.z[j]);
                let d_r_pre = dr * c.r[j] * (1.0 - c.r[j]);
                // (gate, d(pre-activation of the input part), d(pre-activation of the recurrent part))
                let gates = [(0usize, d_z_pre, d_z_pre), (1, d_r_pre, d_r_pre), (2, d_n_pre, d_hn)];
                for (gate, dx_pre, dh_pre) in gates {
                    let row = gate * h + j;
                    grad[bx_o + row] += dx_pre;
                    if dx_pre != 0.0 {
                        let gw = &mut grad[wx_o + row * d..wx_o + (row + 1) * d];
                        for (g, &v) in gw.iter_mut().zip(x) {
                            *g += dx_pre * v;
                        }
                    }
                    if gate == 2 {
                        grad[bh_o + 2 * h + j] += dh_pre;
                    }
                    if dh_pre != 0.0 {
                        for k in 0..h {
                            grad[uh_o + row * h + k] += dh_pre * c.h_prev[k];
                            dh_prev[k] += dh_pre * self.p[uh_o + row * h + k];
                        }
                    }
                }
            }
            dh_next = dh_prev;
        }
        WindowOut {
            loss,
            weight_sum,
            logits,
        }
    }

    fn clone_box(&self) -> Box<dyn SeqNet> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::testutil::{cfg, inputs, targets};

    fn total(net: &Gru, xs: &[Vec<f32>], t: &[StepTargets]) -> f64 {
        let mut g = vec![0.0; net.num_params()];
        f64::from(net.window_grad(xs, t, &cfg(), &mut g).loss.total)
    }

    #[test]
    fn param_count_is_the_textbook_formula_plus_heads() {
        let (d, h) = (1353usize, 2usize);
        assert_eq!(Gru::param_count(d, h), 3 * h * (d + h + 2) + 8 * h + 8);
        assert_eq!(Gru::new(10, 3, 1).num_params(), Gru::param_count(10, 3));
        assert!(Gru::from_params(10, 3, vec![0.0; 5]).is_none());
    }

    #[test]
    fn bptt_gradient_matches_finite_differences() {
        let net = Gru::new(9, 4, 5);
        let (xs, t) = (inputs(6, 9), targets(6));
        let mut g = vec![0.0; net.num_params()];
        let out = net.window_grad(&xs, &t, &cfg(), &mut g);
        assert!(out.weight_sum > 0.0);
        let max_g = g.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(max_g > 0.0);
        let eps = 1e-2f32;
        for k in (0..net.num_params()).step_by(2) {
            let (mut a, mut b) = (net.clone(), net.clone());
            a.params_mut()[k] += eps;
            b.params_mut()[k] -= eps;
            let num = ((total(&a, &xs, &t) - total(&b, &xs, &t)) / (2.0 * f64::from(eps))) as f32;
            assert!((g[k] - num).abs() < 0.03 * max_g + 3e-3, "param {k}: {} vs {num}", g[k]);
        }
    }

    #[test]
    fn online_steps_equal_the_window_forward_and_state_matters() {
        let net = Gru::new(9, 4, 5);
        let xs = inputs(5, 9);
        let win = net.window_logits(&xs);
        let mut st = vec![0.0; 4];
        for (x, l) in xs.iter().zip(&win) {
            assert_eq!(&net.step(&mut st, x), l);
        }
        // The same input after different histories gives different logits.
        let mut a = vec![0.0; 4];
        let mut b = vec![0.0; 4];
        net.step(&mut a, &xs[0]);
        net.step(&mut b, &xs[3]);
        assert_ne!(net.step(&mut a, &xs[4]), net.step(&mut b, &xs[4]));
    }
}
