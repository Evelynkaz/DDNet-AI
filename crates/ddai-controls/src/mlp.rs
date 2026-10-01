//! The MLP control: `x -> tanh(W1 x + b1) -> heads`, one hidden layer, stateless.
//!
//! Parameter layout: `[W1 (hidden x input)] [b1 (hidden)] [heads: W (8 x hidden), b (8)]`.

use ddai_fly::bc::{HeadLogits, LossConfig, StepLoss, StepTargets, head_loss_and_grad};

use crate::net::{NetKind, SeqNet, WindowOut, heads_backward, heads_forward, heads_param_len};

#[derive(Debug, Clone)]
pub struct Mlp {
    input: usize,
    hidden: usize,
    p: Vec<f32>,
}

impl Mlp {
    /// Parameter count of an MLP with these sizes.
    pub fn param_count(input: usize, hidden: usize) -> usize {
        hidden * input + hidden + heads_param_len(hidden)
    }

    /// Deterministic init from `seed`: `W1` uniform in `+-1/sqrt(input)`, the heads uniform in
    /// `+-1/sqrt(hidden)`, biases zero.
    pub fn new(input: usize, hidden: usize, seed: u64) -> Self {
        let mut rng = ddai_fly::rng::SplitMix64::new(seed ^ 0x4D4C50);
        let mut p = vec![0.0f32; Self::param_count(input, hidden)];
        let s1 = (1.0 / input as f32).sqrt();
        for v in &mut p[..hidden * input] {
            *v = (rng.next_f32_unit() * 2.0 - 1.0) * s1;
        }
        let s2 = (1.0 / hidden as f32).sqrt();
        let head_w = hidden * input + hidden;
        for v in &mut p[head_w..head_w + 8 * hidden] {
            *v = (rng.next_f32_unit() * 2.0 - 1.0) * s2;
        }
        Mlp { input, hidden, p }
    }

    pub fn from_params(input: usize, hidden: usize, p: Vec<f32>) -> Option<Self> {
        (p.len() == Self::param_count(input, hidden)).then_some(Mlp { input, hidden, p })
    }

    fn hidden_of(&self, x: &[f32]) -> Vec<f32> {
        let (w1, rest) = self.p.split_at(self.hidden * self.input);
        let b1 = &rest[..self.hidden];
        (0..self.hidden)
            .map(|j| {
                let row = &w1[j * self.input..(j + 1) * self.input];
                (b1[j] + row.iter().zip(x).map(|(&w, &v)| w * v).sum::<f32>()).tanh()
            })
            .collect()
    }

    fn heads(&self) -> &[f32] {
        &self.p[self.hidden * self.input + self.hidden..]
    }
}

impl SeqNet for Mlp {
    fn kind(&self) -> NetKind {
        NetKind::Mlp
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
        0
    }
    fn step(&self, _state: &mut [f32], x: &[f32]) -> HeadLogits {
        assert_eq!(x.len(), self.input, "mlp: input length");
        heads_forward(self.heads(), &self.hidden_of(x))
    }

    fn window_grad(&self, xs: &[Vec<f32>], targets: &[StepTargets], cfg: &LossConfig, grad: &mut [f32]) -> WindowOut {
        assert_eq!(xs.len(), targets.len());
        assert_eq!(grad.len(), self.p.len());
        let (h, i) = (self.hidden, self.input);
        let head_off = h * i + h;
        let mut loss = StepLoss::default();
        let mut weight_sum = 0.0f32;
        let mut logits = Vec::with_capacity(xs.len());
        for (x, t) in xs.iter().zip(targets) {
            let hid = self.hidden_of(x);
            let lg = heads_forward(self.heads(), &hid);
            logits.push(lg);
            if t.weight == 0.0 {
                continue;
            }
            let (l, d) = head_loss_and_grad(&lg, t, cfg);
            loss.add(&l);
            weight_sum += t.weight;
            let dh = heads_backward(self.heads(), &hid, &d, &mut grad[head_off..]);
            for j in 0..h {
                let d_pre = dh[j] * (1.0 - hid[j] * hid[j]);
                grad[h * i + j] += d_pre;
                if d_pre != 0.0 {
                    let row = &mut grad[j * i..(j + 1) * i];
                    for (g, &v) in row.iter_mut().zip(x) {
                        *g += d_pre * v;
                    }
                }
            }
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

    fn total(net: &Mlp, xs: &[Vec<f32>], t: &[StepTargets]) -> f64 {
        let mut g = vec![0.0; net.num_params()];
        f64::from(net.window_grad(xs, t, &cfg(), &mut g).loss.total)
    }

    #[test]
    fn param_count_matches_the_layout() {
        assert_eq!(Mlp::param_count(1353, 5), 5 * 1353 + 5 + 8 * 5 + 8);
        let m = Mlp::new(20, 6, 1);
        assert_eq!(m.num_params(), Mlp::param_count(20, 6));
        assert!(Mlp::from_params(20, 6, vec![0.0; 3]).is_none());
    }

    #[test]
    fn init_is_deterministic_and_seed_dependent() {
        assert_eq!(Mlp::new(30, 4, 7).params(), Mlp::new(30, 4, 7).params());
        assert_ne!(Mlp::new(30, 4, 7).params(), Mlp::new(30, 4, 8).params());
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let net = Mlp::new(12, 5, 3);
        let (xs, t) = (inputs(4, 12), targets(4));
        let mut g = vec![0.0; net.num_params()];
        let out = net.window_grad(&xs, &t, &cfg(), &mut g);
        assert!(out.weight_sum > 0.0);
        let max_g = g.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(max_g > 0.0);
        let eps = 1e-2f32;
        for k in (0..net.num_params()).step_by(3) {
            let (mut a, mut b) = (net.clone(), net.clone());
            a.params_mut()[k] += eps;
            b.params_mut()[k] -= eps;
            let num = ((total(&a, &xs, &t) - total(&b, &xs, &t)) / (2.0 * f64::from(eps))) as f32;
            assert!((g[k] - num).abs() < 0.02 * max_g + 2e-3, "param {k}: {} vs {num}", g[k]);
        }
    }

    #[test]
    fn window_logits_equal_step_by_step_and_burn_in_steps_have_no_gradient_effect() {
        let net = Mlp::new(12, 5, 3);
        let xs = inputs(3, 12);
        let batch = net.window_logits(&xs);
        let mut st = vec![];
        for (x, l) in xs.iter().zip(&batch) {
            assert_eq!(&net.step(&mut st, x), l);
        }
        // A window whose only weighted step is the last one equals that step trained alone.
        let mut t = targets(3);
        t[1].weight = 0.0;
        let mut g_win = vec![0.0; net.num_params()];
        net.window_grad(&xs, &t, &cfg(), &mut g_win);
        let mut g_single = vec![0.0; net.num_params()];
        let mut first = t[0];
        first.weight = 0.0;
        net.window_grad(&[xs[0].clone(), xs[2].clone()], &[first, t[2]], &cfg(), &mut g_single);
        // Step 0 has weight 0 in `targets`, so both windows train step 2 alone.
        for (a, b) in g_win.iter().zip(&g_single) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
