//! The network: `x -> relu(W1 x + b1) -> relu(W2 h1 + b2) -> W3 h2 + b3`, hand-written forward and backward.
//!
//! **Determinism.** Weights are stored input-major (`w[i * n_out + j]`), so a layer is a sum of
//! scaled rows: `out[j] += x[i] * w[i][j]` for `i` in order. Every output element is accumulated in the
//! same fixed order whatever the vectorisation (the loop over `j` has no cross-element reduction), a
//! zero input skips its row (adding `0 * w` changes nothing), and nothing is fused or reassociated
//! (the build has no fast-math and no `target-cpu`). The backward's dot products use eight
//! fixed accumulators ([`dot`]). So a forward pass is bit-identical on every machine, and the same
//! code is the trainer's forward and the brain's. No allocation after [`Scratch::new`].

use serde::{Deserialize, Serialize};

/// A 2-hidden-layer MLP's shape and flat parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mlp {
    pub n_in: usize,
    pub h1: usize,
    pub h2: usize,
    pub n_out: usize,
    /// `[W1 (n_in x h1)] [b1] [W2 (h1 x h2)] [b2] [W3 (h2 x n_out)] [b3]`.
    pub params: Vec<f32>,
}

/// Activations of one forward pass (and the work area of the backward).
#[derive(Debug, Clone)]
pub struct Scratch {
    pub h1: Vec<f32>,
    pub h2: Vec<f32>,
    pub out: Vec<f32>,
    d1: Vec<f32>,
    d2: Vec<f32>,
}

impl Scratch {
    pub fn new(m: &Mlp) -> Scratch {
        Scratch {
            h1: vec![0.0; m.h1],
            h2: vec![0.0; m.h2],
            out: vec![0.0; m.n_out],
            d1: vec![0.0; m.h1],
            d2: vec![0.0; m.h2],
        }
    }
}

/// `out[j] += s * row[j]`.
#[inline]
fn axpy(out: &mut [f32], s: f32, row: &[f32]) {
    for (o, &w) in out.iter_mut().zip(row) {
        *o += s * w;
    }
}

/// Dot product with eight fixed accumulators, combined in a fixed tree, then the tail in order.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0.0f32; 8];
    let ((ca, ra), (cb, rb)) = (a[..n].as_chunks::<8>(), b[..n].as_chunks::<8>());
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

impl Mlp {
    pub fn param_count(n_in: usize, h1: usize, h2: usize, n_out: usize) -> usize {
        n_in * h1 + h1 + h1 * h2 + h2 + h2 * n_out + n_out
    }

    /// He-uniform init from `seed` (SplitMix64), zero biases, the output layer scaled down.
    pub fn new(n_in: usize, h1: usize, h2: usize, n_out: usize, seed: u64) -> Mlp {
        let mut st = seed ^ 0x6f70_706e_6574;
        let mut next = move || {
            st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = st;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32) / ((1u64 << 24) as f32)
        };
        let mut params = vec![0.0f32; Self::param_count(n_in, h1, h2, n_out)];
        let m = Mlp {
            n_in,
            h1,
            h2,
            n_out,
            params: Vec::new(),
        };
        let (w1, _, w2, _, w3, _) = m.offsets();
        for (off, fan_in, len, scale) in [
            (w1, n_in, n_in * h1, 1.0),
            (w2, h1, h1 * h2, 1.0),
            (w3, h2, h2 * n_out, 0.3),
        ] {
            let lim = (6.0 / fan_in as f32).sqrt() * scale;
            for p in &mut params[off..off + len] {
                *p = (next() * 2.0 - 1.0) * lim;
            }
        }
        Mlp { params, ..m }
    }

    /// Offsets of `(W1, b1, W2, b2, W3, b3)` in [`Mlp::params`].
    pub fn offsets(&self) -> (usize, usize, usize, usize, usize, usize) {
        let w1 = 0;
        let b1 = w1 + self.n_in * self.h1;
        let w2 = b1 + self.h1;
        let b2 = w2 + self.h1 * self.h2;
        let w3 = b2 + self.h2;
        let b3 = w3 + self.h2 * self.n_out;
        (w1, b1, w2, b2, w3, b3)
    }

    pub fn is_consistent(&self) -> bool {
        self.params.len() == Self::param_count(self.n_in, self.h1, self.h2, self.n_out)
            && self.params.iter().all(|p| p.is_finite())
    }

    /// The forward pass; the logits are left in `s.out`.
    pub fn forward(&self, x: &[f32], s: &mut Scratch) {
        debug_assert_eq!(x.len(), self.n_in);
        let (w1, b1, w2, b2, w3, b3) = self.offsets();
        let p = &self.params;
        s.h1.copy_from_slice(&p[b1..b1 + self.h1]);
        for (i, &xi) in x.iter().enumerate() {
            if xi != 0.0 {
                axpy(&mut s.h1, xi, &p[w1 + i * self.h1..w1 + (i + 1) * self.h1]);
            }
        }
        for v in &mut s.h1 {
            *v = v.max(0.0);
        }
        s.h2.copy_from_slice(&p[b2..b2 + self.h2]);
        for (i, &a) in s.h1.iter().enumerate() {
            if a != 0.0 {
                axpy(&mut s.h2, a, &p[w2 + i * self.h2..w2 + (i + 1) * self.h2]);
            }
        }
        for v in &mut s.h2 {
            *v = v.max(0.0);
        }
        s.out.copy_from_slice(&p[b3..b3 + self.n_out]);
        for (i, &a) in s.h2.iter().enumerate() {
            if a != 0.0 {
                axpy(&mut s.out, a, &p[w3 + i * self.n_out..w3 + (i + 1) * self.n_out]);
            }
        }
    }

    /// Adds the gradient of a loss whose derivative at the logits is `d_out` to `grad` (same layout as the parameters). `s` must hold the
    /// activations of the forward pass on `x`.
    pub fn backward(&self, x: &[f32], s: &mut Scratch, d_out: &[f32], grad: &mut [f32]) {
        let (w1, b1, w2, b2, w3, b3) = self.offsets();
        let p = &self.params;
        for (g, d) in grad[b3..b3 + self.n_out].iter_mut().zip(d_out) {
            *g += d;
        }
        for i in 0..self.h2 {
            let a = s.h2[i];
            if a > 0.0 {
                axpy(&mut grad[w3 + i * self.n_out..w3 + (i + 1) * self.n_out], a, d_out);
                s.d2[i] = dot(&p[w3 + i * self.n_out..w3 + (i + 1) * self.n_out], d_out);
            } else {
                s.d2[i] = 0.0;
            }
        }
        for (g, d) in grad[b2..b2 + self.h2].iter_mut().zip(&s.d2) {
            *g += d;
        }
        for i in 0..self.h1 {
            let a = s.h1[i];
            if a > 0.0 {
                axpy(&mut grad[w2 + i * self.h2..w2 + (i + 1) * self.h2], a, &s.d2);
                s.d1[i] = dot(&p[w2 + i * self.h2..w2 + (i + 1) * self.h2], &s.d2);
            } else {
                s.d1[i] = 0.0;
            }
        }
        for (g, d) in grad[b1..b1 + self.h1].iter_mut().zip(&s.d1) {
            *g += d;
        }
        for (i, &xi) in x.iter().enumerate() {
            if xi != 0.0 {
                axpy(&mut grad[w1 + i * self.h1..w1 + (i + 1) * self.h1], xi, &s.d1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) as f32 / 16_777_216.0;
                if i % 5 == 0 { 0.0 } else { v * 2.0 - 1.0 }
            })
            .collect()
    }

    /// A smooth test loss: `0.5 * sum((out - t)^2)`.
    fn loss(m: &Mlp, x: &[f32], t: &[f32]) -> f64 {
        let mut s = Scratch::new(m);
        m.forward(x, &mut s);
        s.out.iter().zip(t).map(|(o, t)| 0.5 * f64::from(o - t).powi(2)).sum()
    }

    #[test]
    fn backward_matches_finite_differences() {
        let m = Mlp::new(20, 12, 9, 6, 5);
        let x = input(20, 3);
        let t = input(6, 8);
        let mut s = Scratch::new(&m);
        m.forward(&x, &mut s);
        let d: Vec<f32> = s.out.iter().zip(&t).map(|(o, t)| o - t).collect();
        let mut g = vec![0.0f32; m.params.len()];
        m.backward(&x, &mut s, &d, &mut g);
        let max_g = g.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(max_g > 0.0);
        let eps = 1e-3f32;
        let mut checked = 0;
        for k in (0..m.params.len()).step_by(7) {
            let (mut a, mut b) = (m.clone(), m.clone());
            a.params[k] += eps;
            b.params[k] -= eps;
            let num = ((loss(&a, &x, &t) - loss(&b, &x, &t)) / (2.0 * f64::from(eps))) as f32;
            // A kink of the relu can sit inside the step; allow a small absolute slack.
            assert!((g[k] - num).abs() < 0.02 * max_g + 3e-3, "param {k}: {} vs {num}", g[k]);
            checked += 1;
        }
        assert!(checked > 20);
    }

    #[test]
    fn the_sparse_forward_equals_a_dense_one() {
        let m = Mlp::new(30, 16, 8, 5, 2);
        let x = input(30, 1);
        let mut s = Scratch::new(&m);
        m.forward(&x, &mut s);
        // Dense reference in the same accumulation order, without the zero skip.
        let (w1, b1, w2, b2, w3, b3) = m.offsets();
        let p = &m.params;
        let mut h1 = p[b1..b1 + 16].to_vec();
        for (i, &xi) in x.iter().enumerate() {
            for j in 0..16 {
                h1[j] += xi * p[w1 + i * 16 + j];
            }
        }
        h1.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut h2 = p[b2..b2 + 8].to_vec();
        for (i, &a) in h1.iter().enumerate() {
            for j in 0..8 {
                h2[j] += a * p[w2 + i * 8 + j];
            }
        }
        h2.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut out = p[b3..b3 + 5].to_vec();
        for (i, &a) in h2.iter().enumerate() {
            for j in 0..5 {
                out[j] += a * p[w3 + i * 5 + j];
            }
        }
        for (a, b) in s.out.iter().zip(&out) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn init_is_deterministic_and_the_count_matches() {
        let a = Mlp::new(40, 10, 6, 4, 9);
        assert_eq!(a, Mlp::new(40, 10, 6, 4, 9));
        assert_ne!(a.params, Mlp::new(40, 10, 6, 4, 10).params);
        assert_eq!(a.params.len(), 40 * 10 + 10 + 10 * 6 + 6 + 6 * 4 + 4);
        assert!(a.is_consistent());
    }

    #[test]
    fn dot_sums_in_a_fixed_order_and_handles_the_tail() {
        let a: Vec<f32> = (0..21).map(|i| i as f32 * 0.37 - 3.0).collect();
        let b: Vec<f32> = (0..21).map(|i| 1.0 / (i as f32 + 1.5)).collect();
        let d = dot(&a, &b);
        let r: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
        assert!((f64::from(d) - r).abs() < 1e-4);
        assert_eq!(d.to_bits(), dot(&a, &b).to_bits());
    }
}
