//! The ES noise and update: antithetic Gaussian perturbations from per-pair seeds, the gradient estimate from shaped fitness, and Adam.

use ddai_fly::rng::SplitMix64;
use serde::{Deserialize, Serialize};

/// The seed of pair `pair` of generation `generation`: a function of `(run seed, generation, pair)` only, so a resumed run, and a run on any number of
/// threads, draws the very same noise.
pub fn pair_seed(seed: u64, generation: u64, pair: u64) -> u64 {
    let mut z = seed
        ^ generation.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ pair.wrapping_add(1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `n` standard normal draws for one pair; its two members are `theta + sigma * eps` and `theta - sigma * eps` (antithetic sampling).
pub fn pair_noise(seed: u64, generation: u64, pair: u64, n: usize) -> Vec<f32> {
    let mut r = SplitMix64::new(pair_seed(seed, generation, pair));
    (0..n).map(|_| r.next_gaussian()).collect()
}

/// The two members of a pair: `(theta + sigma * eps, theta - sigma * eps)`. Parameters with `sigma == 0` are not perturbed.
pub fn members(theta: &[f32], sigma: &[f32], eps: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let plus = theta.iter().zip(sigma).zip(eps).map(|((t, s), e)| t + s * e).collect();
    let minus = theta.iter().zip(sigma).zip(eps).map(|((t, s), e)| t - s * e).collect();
    (plus, minus)
}

/// The ES gradient estimate (ascent direction) from the shaped fitness `u` of the `2 * pairs` members (member `2i` is the `+` one of
/// pair `i`, member `2i + 1` the `-` one): `g = sum_i (u_{2i} - u_{2i+1}) eps_i / (2 * pairs * sigma)`, element by element (a parameter
/// with `sigma == 0` gets `0`). `eps_of(i)` regenerates pair `i`'s noise.
pub fn gradient_estimate(u: &[f32], sigma: &[f32], mut eps_of: impl FnMut(usize) -> Vec<f32>) -> Vec<f32> {
    assert!(u.len().is_multiple_of(2));
    let pairs = u.len() / 2;
    let mut g = vec![0.0f32; sigma.len()];
    for i in 0..pairs {
        let w = u[2 * i] - u[2 * i + 1];
        if w == 0.0 {
            continue;
        }
        let eps = eps_of(i);
        for ((gj, e), s) in g.iter_mut().zip(&eps).zip(sigma) {
            if *s > 0.0 {
                *gj += w * e;
            }
        }
    }
    let scale = 1.0 / (2.0 * pairs.max(1) as f32);
    for (gj, s) in g.iter_mut().zip(sigma) {
        *gj = if *s > 0.0 { *gj * scale / s } else { 0.0 };
    }
    g
}

/// Adam with a per-parameter learning rate, ascending (`params += step`); serialisable, so it resumes bit for bit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AscentAdam {
    pub step: u64,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl AscentAdam {
    pub fn new(n: usize, beta1: f32, beta2: f32) -> Self {
        AscentAdam {
            step: 0,
            m: vec![0.0; n],
            v: vec![0.0; n],
            beta1,
            beta2,
            eps: 1e-8,
        }
    }

    /// One step; returns the Euclidean norm of the parameter change.
    pub fn apply(&mut self, params: &mut [f32], grad: &[f32], lr: &[f32]) -> f32 {
        self.step += 1;
        let c1 = (1.0 - f64::from(self.beta1).powi(self.step.min(i32::MAX as u64) as i32)) as f32;
        let c2 = (1.0 - f64::from(self.beta2).powi(self.step.min(i32::MAX as u64) as i32)) as f32;
        let mut sq = 0.0f64;
        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let d = lr[i] * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + self.eps);
            params[i] += d;
            sq += f64::from(d) * f64::from(d);
        }
        sq.sqrt() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_a_function_of_seed_generation_and_pair_only() {
        let a = pair_noise(7, 3, 5, 64);
        assert_eq!(a, pair_noise(7, 3, 5, 64));
        assert_ne!(a, pair_noise(7, 3, 6, 64));
        assert_ne!(a, pair_noise(7, 4, 5, 64));
        assert_ne!(a, pair_noise(8, 3, 5, 64));
        // The first 10 draws do not depend on how many are asked for.
        assert_eq!(a[..10], pair_noise(7, 3, 5, 10)[..]);
    }

    #[test]
    fn a_pairs_members_are_antithetic_and_a_zero_sigma_parameter_is_not_moved() {
        let theta = [1.0, 2.0, 3.0, 4.0];
        let sigma = [0.1, 0.0, 0.2, 0.1];
        let eps = pair_noise(1, 0, 0, 4);
        let (p, m) = members(&theta, &sigma, &eps);
        for j in 0..4 {
            assert!(
                ((p[j] + m[j]) / 2.0 - theta[j]).abs() < 1e-6,
                "mean of the pair is theta"
            );
        }
        assert_eq!((p[1], m[1]), (2.0, 2.0));
        assert!((p[0] - m[0] - 2.0 * 0.1 * eps[0]).abs() < 1e-6);
    }

    #[test]
    fn the_estimate_points_up_a_linear_fitness() {
        // F(theta) = w . theta: the ES gradient estimate must correlate strongly with w. Rank-shaped, so only the direction.
        let n = 40;
        let w: Vec<f32> = (0..n).map(|j| ((j % 7) as f32 - 3.0) / 3.0).collect();
        let theta = vec![0.0f32; n];
        let sigma = vec![0.1f32; n];
        let pairs = 200;
        let eps: Vec<Vec<f32>> = (0..pairs).map(|i| pair_noise(11, 0, i as u64, n)).collect();
        let mut fit = Vec::new();
        for e in &eps {
            let (p, m) = members(&theta, &sigma, e);
            fit.push(p.iter().zip(&w).map(|(a, b)| a * b).sum::<f32>());
            fit.push(m.iter().zip(&w).map(|(a, b)| a * b).sum::<f32>());
        }
        let u = crate::es::stats::centered_ranks(&fit);
        let g = gradient_estimate(&u, &sigma, |i| eps[i].clone());
        let dot: f32 = g.iter().zip(&w).map(|(a, b)| a * b).sum();
        let (ng, nw) = (
            g.iter().map(|x| x * x).sum::<f32>().sqrt(),
            w.iter().map(|x| x * x).sum::<f32>().sqrt(),
        );
        assert!(dot / (ng * nw) > 0.9, "cosine {}", dot / (ng * nw));
    }

    #[test]
    fn adam_ascends_and_is_per_parameter() {
        let mut adam = AscentAdam::new(2, 0.9, 0.999);
        let mut p = [0.0f32, 0.0];
        let moved = adam.apply(&mut p, &[1.0, -1.0], &[0.1, 0.01]);
        assert!(p[0] > 0.0 && p[1] < 0.0);
        assert!((p[0].abs() / p[1].abs() - 10.0).abs() < 1e-3);
        assert!(moved > 0.0);
        // A clone resumes identically.
        let mut b = adam.clone();
        let (mut p1, mut p2) = (p, p);
        adam.apply(&mut p1, &[0.5, 0.5], &[0.1, 0.1]);
        b.apply(&mut p2, &[0.5, 0.5], &[0.1, 0.1]);
        assert_eq!(p1, p2);
    }
}
