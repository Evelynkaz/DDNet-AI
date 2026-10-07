//! The fly's **stochastic policy** over the decoded action heads (task 8.5b, recurrent PPO).
//!
//! The heads are those of the BC loss ([`crate::bc`]): `dir` (3-way softmax), `jump` / `hook` / `fire` (Bernoulli, logit
//! heads) and `aim` (the population vector `(C, S)`, whose angle `mu = atan2(S, C)` is the mean of a von Mises with the
//! fixed concentration `kappa`: the BC loss `-kappa * cos(target - mu)` is exactly its negative log-likelihood up to a
//! constant). This module adds what a policy-gradient method needs on top of them, all in closed form, with the
//! gradients with respect to the head logits checked against finite differences:
//!
//! * [`sample`]: one action from the head logits, reproducibly from a [`SplitMix64`];
//! * [`log_prob`] / [`log_prob_grad`]: the exact log-probability of an action and its gradient;
//! * [`entropy`] / [`entropy_grad`]: the entropy of the discrete heads (the von Mises head has a constant one, because
//!   `kappa` is fixed);
//! * [`kl`] / [`kl_grad`]: the exact `KL(ref || cur)` between two sets of head logits (the KL anchor of the PPO plan);
//! * [`argmax`]: the **deterministic mode** (evaluation and live play): `argmax` of the direction head, `p >= threshold` of
//!   the three binary heads, the aim angle `mu`. It is the decoding [`crate::brain::FlyBrain`] uses at
//!   `ActionSelection::Argmax`, written on the logits; `tests/policy_identity.rs`-style arena games prove it bit for bit.
//!
//! # The thresholds are part of the policy
//! A trained fly carries calibrated thresholds of its binary heads (`HeadThresholds`: the key is pressed when `p >= t`; E-005
//! review F5: a head trained with a positive-class weight presses far more often than the teacher at `0.5`). A sampling
//! policy that ignored them would play a different distribution than the thresholded play it is meant to improve, so the
//! stochastic policy is defined on the **shifted logit** `z - logit(t)`: `P(press) = sigmoid(z - logit(t))`. Its mode
//! (`P >= 0.5`) is exactly `p >= t`, i.e. the deterministic play. Every function here takes the already shifted logits
//! ([`shifted`]); the aim head has no threshold.
//!
//! # The aim only counts when it matters
//! The aim of a decision reaches the game only when the hook is **thrown** (the button pressed while the hook is idle: `hook_dir` is
//! set on the transition from `HOOK_IDLE` to `HOOK_FLYING` only, `ddai-physics::core`) or the weapon is fired. (The BC mask of the aim head,
//! `hook || fire`, is wider; it is a label rule, not what the physics reads.) Counting the aim of every held-hook decision would add one
//! von Mises score term per decision of a hold, multiplied by the advantage, for an aim that changed nothing: variance for no signal. So the
//! von Mises log-probability, the gradient and the KL term of a decision are included only when [`PolicyAction::aim_counts`] is set, which
//! the actor does from the observation ([`counts_aim`]); `sample` and `argmax` set the wider `hook || fire` until the caller refines it.

use crate::bc::{HeadLogits, HeadThresholds};
use crate::rng::SplitMix64;

/// One sampled (or arg-max) action in the policy's own terms: the direction class `0/1/2` = left/stop/right, the three
/// binary keys and the ring-convention aim angle (radians).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PolicyAction {
    pub dir: u8,
    pub jump: bool,
    pub hook: bool,
    pub fire: bool,
    pub aim: f32,
    /// The aim of this decision reaches the game (a throw or a shot) and so is part of its probability, gradient and KL.
    pub aim_counts: bool,
}

/// The aim of an action is part of its probability only when a hook is thrown or a shot fired.
pub fn aim_active(a: &PolicyAction) -> bool {
    a.aim_counts
}

/// Whether the aim of a decision reaches the game: the weapon is fired, or the hook button is pressed while the **observed** hook state is idle
/// (a throw). In the two-view fly the hook head reads the masked view, so the state has to come from the real observation.
pub fn counts_aim(hook: bool, fire: bool, hook_state_idle: bool) -> bool {
    fire || (hook && hook_state_idle)
}

/// `ln(t / (1 - t))`.
pub fn logit_of(t: f32) -> f32 {
    (t / (1.0 - t)).ln()
}

/// The head logits as the **policy** sees them: the binary heads shifted by the logit of their calibrated threshold, so that
/// `P(press) >= 0.5` exactly when `p >= threshold` (module docs).
pub fn shifted(l: &HeadLogits, th: &HeadThresholds) -> HeadLogits {
    HeadLogits {
        jump: l.jump - logit_of(th.jump),
        hook: l.hook - logit_of(th.hook),
        fire: l.fire - logit_of(th.fire),
        ..*l
    }
}

/// Temperatures of the heads of the sampled policy (see [`policy_logits`]); the aim keeps its fixed concentration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Temperatures {
    pub dir: f32,
    pub jump: f32,
    pub hook: f32,
    pub fire: f32,
}

impl Temperatures {
    pub fn uniform(t: f32) -> Temperatures {
        Temperatures {
            dir: t,
            jump: t,
            hook: t,
            fire: t,
        }
    }
}

/// The logits of the policy at `temps`: [`shifted`], then each head's logits divided by its temperature. `T < 1` sharpens towards the
/// deterministic play, and `T -> 0` is [`argmax`]: a BC-trained fly's heads are soft (its entropy is near the maximum), so the distribution it is
/// *sampled* from for the policy gradient has to be sharper than the raw heads to play anything like the thresholded decoding it is meant to
/// improve. The gradient with respect to the raw logits of a loss on these is `1 / temperature` times the one on the returned logits (for the
/// heads that are scaled; [`unscale_temperature`]).
pub fn policy_logits(l: &HeadLogits, th: &HeadThresholds, temps: &Temperatures) -> HeadLogits {
    let s = shifted(l, th);
    let kd = 1.0 / temps.dir;
    HeadLogits {
        dir: [s.dir[0] * kd, s.dir[1] * kd, s.dir[2] * kd],
        jump: s.jump / temps.jump,
        hook: s.hook / temps.hook,
        fire: s.fire / temps.fire,
        ..s
    }
}

/// Multiplies the gradient of a loss on [`policy_logits`] into the one on the raw logits (the chain rule of the temperatures).
pub fn unscale_temperature(d: &mut HeadLogits, temps: &Temperatures) {
    for v in &mut d.dir {
        *v /= temps.dir;
    }
    d.jump /= temps.jump;
    d.hook /= temps.hook;
    d.fire /= temps.fire;
}

fn softplus(x: f32) -> f32 {
    // ln(1 + e^x), stable on both sides.
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

fn sigmoid(x: f32) -> f32 {
    crate::activation::sigmoid(x)
}

/// `ln softmax(z)[i]` for all three classes.
fn log_softmax3(z: [f32; 3]) -> [f32; 3] {
    let m = z[0].max(z[1]).max(z[2]);
    let s = (z[0] - m).exp() + (z[1] - m).exp() + (z[2] - m).exp();
    let l = m + s.ln();
    [z[0] - l, z[1] - l, z[2] - l]
}

/// `ln I0(kappa)` and `I1(kappa) / I0(kappa)`, by the power series in `f64` (all terms positive: no cancellation).
fn bessel_i0_i1(kappa: f32) -> (f64, f64) {
    let x = f64::from(kappa) / 2.0;
    let (mut t0, mut i0) = (1.0f64, 1.0f64);
    let (mut t1, mut i1) = (x, x);
    for k in 1..400 {
        let kk = f64::from(k);
        t0 *= x * x / (kk * kk);
        t1 *= x * x / (kk * (kk + 1.0));
        i0 += t0;
        i1 += t1;
        if t0 < 1e-17 * i0 && t1 < 1e-17 * i1 {
            break;
        }
    }
    (i0.ln(), i1 / i0)
}

/// `ln(2 pi I0(kappa))`: the normaliser of the von Mises density.
pub fn ln_vm_norm(kappa: f32) -> f32 {
    ((2.0 * std::f64::consts::PI).ln() + bessel_i0_i1(kappa).0) as f32
}

/// `A(kappa) = I1 / I0`, the mean resultant length of a von Mises.
pub fn vm_resultant(kappa: f32) -> f32 {
    bessel_i0_i1(kappa).1 as f32
}

/// The wrapped difference `a - b` in `(-pi, pi]` (only used for the sine/cosine below, so wrapping is implicit).
fn aim_mean(l: &HeadLogits) -> Option<(f32, f32, f32)> {
    let r2 = l.aim_c * l.aim_c + l.aim_s * l.aim_s;
    if r2 < 1e-12 {
        None
    } else {
        Some((l.aim_s.atan2(l.aim_c), l.aim_c, l.aim_s))
    }
}

/// The aim mean of the logits (`atan2(S, C)`; `0` for the degenerate `(0, 0)`).
pub fn aim_mu(l: &HeadLogits) -> f32 {
    aim_mean(l).map_or(0.0, |(mu, _, _)| mu)
}

/// The deterministic mode: direction `argmax` (ties go to the last maximal class, as `FlyBrain`'s `max_by`), the binary keys at
/// `p >= threshold`, the aim `atan2(S, C)`. `l` are the **raw** logits (not [`shifted`]): the comparison is the one the
/// brain makes on the decoder's probabilities, so the result is bit-identical to `ActionSelection::Argmax`.
pub fn argmax(l: &HeadLogits, th: &HeadThresholds) -> PolicyAction {
    let probs = l.dir_probs();
    let dir = (0..3usize)
        .max_by(|&a, &b| probs[a].partial_cmp(&probs[b]).unwrap())
        .unwrap() as u8;
    PolicyAction {
        dir,
        jump: l.jump_on(th),
        hook: l.hook_on(th),
        fire: l.fire_on(th),
        aim: l.aim_angle(),
        aim_counts: l.hook_on(th) || l.fire_on(th),
    }
}

/// A von Mises sample around `mu` (Best & Fisher 1979); `kappa >= 1e-3`. The angle is wrapped to `(-pi, pi]`.
pub fn sample_von_mises(mu: f32, kappa: f32, rng: &mut SplitMix64) -> f32 {
    use std::f64::consts::PI;
    let (mu, kappa) = (f64::from(mu), f64::from(kappa));
    let u = |rng: &mut SplitMix64| f64::from(rng.next_f32_unit());
    if kappa < 1e-3 {
        return (PI * (2.0 * u(rng) - 1.0)) as f32;
    }
    let tau = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
    let rho = (tau - (2.0 * tau).sqrt()) / (2.0 * kappa);
    let r = (1.0 + rho * rho) / (2.0 * rho);
    let f = loop {
        let (u1, u2) = (u(rng), u(rng));
        let z = (PI * u1).cos();
        let f = (1.0 + r * z) / (r + z);
        let c = kappa * (r - f);
        if c * (2.0 - c) - u2 > 0.0 || (c / u2.max(1e-300)).ln() + 1.0 - c >= 0.0 {
            break f;
        }
    };
    let u3 = u(rng);
    let theta = if u3 > 0.5 {
        f.clamp(-1.0, 1.0).acos()
    } else {
        -f.clamp(-1.0, 1.0).acos()
    };
    let a = mu + theta;
    a.sin().atan2(a.cos()) as f32
}

/// One action from the policy at `l` (the **shifted** logits, [`shifted`]): every head independently, aim from a von Mises around
/// `atan2(S, C)` with concentration `aim_kappa`.
pub fn sample(l: &HeadLogits, aim_kappa: f32, rng: &mut SplitMix64) -> PolicyAction {
    let p = l.dir_probs();
    let u = rng.next_f32_unit();
    let (mut cum, mut dir) = (0.0f32, 2u8);
    for (i, &pi) in p.iter().enumerate() {
        cum += pi;
        if u < cum {
            dir = i as u8;
            break;
        }
    }
    let jump = rng.next_f32_unit() < l.jump_prob();
    let hook = rng.next_f32_unit() < l.hook_prob();
    let fire = rng.next_f32_unit() < l.fire_prob();
    let aim = sample_von_mises(aim_mu(l), aim_kappa, rng);
    PolicyAction {
        dir,
        jump,
        hook,
        fire,
        aim,
        aim_counts: hook || fire,
    }
}

/// Log-probabilities of the heads of one action.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HeadLogProbs {
    pub dir: f32,
    pub jump: f32,
    pub hook: f32,
    pub fire: f32,
    /// `0` when the aim is not [`aim_active`].
    pub aim: f32,
}

impl HeadLogProbs {
    pub fn total(&self) -> f32 {
        self.dir + self.jump + self.hook + self.fire + self.aim
    }
}

/// `ln pi(a | l)` per head, `l` the **shifted** logits.
pub fn log_prob(l: &HeadLogits, a: &PolicyAction, aim_kappa: f32) -> HeadLogProbs {
    let bern = |z: f32, on: bool| if on { -softplus(-z) } else { -softplus(z) };
    let aim = if aim_active(a) {
        match aim_mean(l) {
            Some((mu, _, _)) => aim_kappa * (a.aim - mu).cos() - ln_vm_norm(aim_kappa),
            // A degenerate mean direction: the density is flat in the angle's mean, the gradient is zero too.
            None => aim_kappa - ln_vm_norm(aim_kappa),
        }
    } else {
        0.0
    };
    HeadLogProbs {
        dir: log_softmax3(l.dir)[usize::from(a.dir.min(2))],
        jump: bern(l.jump, a.jump),
        hook: bern(l.hook, a.hook),
        fire: bern(l.fire, a.fire),
        aim,
    }
}

/// `d ln pi(a | l) / d l`: the gradient of [`log_prob`]`.total()` with respect to the (shifted) head logits; a threshold shift is
/// a constant, so it is also the gradient with respect to the raw logits.
pub fn log_prob_grad(l: &HeadLogits, a: &PolicyAction, aim_kappa: f32) -> HeadLogits {
    let p = l.dir_probs();
    let mut d = HeadLogits::default();
    for (i, (di, pi)) in d.dir.iter_mut().zip(p).enumerate() {
        *di = f32::from(u8::from(usize::from(a.dir.min(2)) == i)) - pi;
    }
    d.jump = f32::from(u8::from(a.jump)) - sigmoid(l.jump);
    d.hook = f32::from(u8::from(a.hook)) - sigmoid(l.hook);
    d.fire = f32::from(u8::from(a.fire)) - sigmoid(l.fire);
    if aim_active(a)
        && let Some((mu, c, s)) = aim_mean(l)
    {
        let r2 = c * c + s * s;
        let d_mu = aim_kappa * (a.aim - mu).sin();
        d.aim_c = d_mu * (-s / r2);
        d.aim_s = d_mu * (c / r2);
    }
    d
}

/// Entropy (nats) of the discrete heads; the von Mises head's entropy is a constant of `kappa`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HeadEntropy {
    pub dir: f32,
    pub jump: f32,
    pub hook: f32,
    pub fire: f32,
}

impl HeadEntropy {
    pub fn total(&self) -> f32 {
        self.dir + self.jump + self.hook + self.fire
    }
}

fn bern_entropy(z: f32) -> f32 {
    // H = -p ln p - (1-p) ln(1-p) = softplus(z) - p z... written stably: ln(1+e^z) - p z.
    softplus(z) - sigmoid(z) * z
}

pub fn entropy(l: &HeadLogits) -> HeadEntropy {
    let ls = log_softmax3(l.dir);
    let p = l.dir_probs();
    HeadEntropy {
        dir: -(0..3).map(|i| p[i] * ls[i]).sum::<f32>(),
        jump: bern_entropy(l.jump),
        hook: bern_entropy(l.hook),
        fire: bern_entropy(l.fire),
    }
}

/// `d H / d l` of the discrete heads.
pub fn entropy_grad(l: &HeadLogits) -> HeadLogits {
    let ls = log_softmax3(l.dir);
    let p = l.dir_probs();
    let h = entropy(l).dir;
    let bern = |z: f32| {
        let p = sigmoid(z);
        -z * p * (1.0 - p)
    };
    HeadLogits {
        dir: [-p[0] * (ls[0] + h), -p[1] * (ls[1] + h), -p[2] * (ls[2] + h)],
        jump: bern(l.jump),
        hook: bern(l.hook),
        fire: bern(l.fire),
        aim_c: 0.0,
        aim_s: 0.0,
    }
}

/// `KL(ref || cur)` per head (nats), the aim term only when `aim_on` (then `kappa * A(kappa) * (1 - cos(mu_ref - mu_cur))`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HeadKl {
    pub dir: f32,
    pub jump: f32,
    pub hook: f32,
    pub fire: f32,
    pub aim: f32,
}

impl HeadKl {
    pub fn total(&self) -> f32 {
        self.dir + self.jump + self.hook + self.fire + self.aim
    }
}

fn bern_kl(zr: f32, zc: f32) -> f32 {
    // p_r ln(p_r/p_c) + (1-p_r) ln((1-p_r)/(1-p_c)), via log-sigmoids.
    let (pr, ls_r, ls_c) = (sigmoid(zr), -softplus(-zr), -softplus(-zc));
    let (lns_r, lns_c) = (-softplus(zr), -softplus(zc)); // ln(1 - p)
    pr * (ls_r - ls_c) + (1.0 - pr) * (lns_r - lns_c)
}

pub fn kl(r: &HeadLogits, c: &HeadLogits, aim_kappa: f32, aim_on: bool) -> HeadKl {
    let (lr, lc) = (log_softmax3(r.dir), log_softmax3(c.dir));
    let pr = r.dir_probs();
    let aim = if aim_on {
        aim_kappa * vm_resultant(aim_kappa) * (1.0 - (aim_mu(r) - aim_mu(c)).cos())
    } else {
        0.0
    };
    HeadKl {
        dir: (0..3).map(|i| pr[i] * (lr[i] - lc[i])).sum(),
        jump: bern_kl(r.jump, c.jump),
        hook: bern_kl(r.hook, c.hook),
        fire: bern_kl(r.fire, c.fire),
        aim,
    }
}

/// `d KL(ref || cur) / d cur`.
pub fn kl_grad(r: &HeadLogits, c: &HeadLogits, aim_kappa: f32, aim_on: bool) -> HeadLogits {
    let (pr, pc) = (r.dir_probs(), c.dir_probs());
    let mut d = HeadLogits {
        dir: [pc[0] - pr[0], pc[1] - pr[1], pc[2] - pr[2]],
        jump: sigmoid(c.jump) - sigmoid(r.jump),
        hook: sigmoid(c.hook) - sigmoid(r.hook),
        fire: sigmoid(c.fire) - sigmoid(r.fire),
        aim_c: 0.0,
        aim_s: 0.0,
    };
    if aim_on && let Some((mu_c, cc, ss)) = aim_mean(c) {
        let mu_r = aim_mu(r);
        let r2 = cc * cc + ss * ss;
        // d/d mu_c [ kappa A (1 - cos(mu_r - mu_c)) ] = -kappa A sin(mu_r - mu_c)
        let d_mu = -aim_kappa * vm_resultant(aim_kappa) * (mu_r - mu_c).sin();
        d.aim_c = d_mu * (-ss / r2);
        d.aim_s = d_mu * (cc / r2);
    }
    d
}

/// `a + k * b`, head by head (the gradient algebra of a loss made of several terms).
pub fn axpy(a: &mut HeadLogits, k: f32, b: &HeadLogits) {
    for i in 0..3 {
        a.dir[i] += k * b.dir[i];
    }
    a.jump += k * b.jump;
    a.hook += k * b.hook;
    a.fire += k * b.fire;
    a.aim_c += k * b.aim_c;
    a.aim_s += k * b.aim_s;
}

/// `l *= k`, head by head.
pub fn scale(l: &mut HeadLogits, k: f32) {
    for v in &mut l.dir {
        *v *= k;
    }
    l.jump *= k;
    l.hook *= k;
    l.fire *= k;
    l.aim_c *= k;
    l.aim_s *= k;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(seed: u64) -> HeadLogits {
        let mut r = SplitMix64::new(seed);
        let mut g = || (r.next_f32_unit() - 0.5) * 4.0;
        HeadLogits {
            dir: [g(), g(), g()],
            jump: g(),
            hook: g(),
            fire: g(),
            aim_c: g(),
            aim_s: g(),
        }
    }

    /// Central differences of `f` against the analytic gradient `d`, for every logit.
    fn check(name: &str, l: &HeadLogits, d: &HeadLogits, f: &dyn Fn(&HeadLogits) -> f64) {
        let eps = 1e-3f32;
        type Edit = fn(&mut HeadLogits) -> &mut f32;
        let fields: [(&str, Edit, f32); 8] = [
            ("dir0", |l| &mut l.dir[0], d.dir[0]),
            ("dir1", |l| &mut l.dir[1], d.dir[1]),
            ("dir2", |l| &mut l.dir[2], d.dir[2]),
            ("jump", |l| &mut l.jump, d.jump),
            ("hook", |l| &mut l.hook, d.hook),
            ("fire", |l| &mut l.fire, d.fire),
            ("aim_c", |l| &mut l.aim_c, d.aim_c),
            ("aim_s", |l| &mut l.aim_s, d.aim_s),
        ];
        for (fname, edit, analytic) in fields {
            let (mut a, mut b) = (*l, *l);
            *edit(&mut a) += eps;
            *edit(&mut b) -= eps;
            let numeric = ((f(&a) - f(&b)) / (2.0 * f64::from(eps))) as f32;
            assert!(
                (numeric - analytic).abs() < 2e-3 * (1.0 + numeric.abs()),
                "{name}.{fname}: analytic {analytic} vs numeric {numeric}"
            );
        }
    }

    #[test]
    fn log_prob_gradient_matches_finite_differences() {
        for seed in 0..6 {
            let l = logits(seed);
            for (dir, jump, hook, fire) in [
                (0u8, true, false, false),
                (2, false, true, true),
                (1, true, true, false),
            ] {
                let a = PolicyAction {
                    dir,
                    jump,
                    hook,
                    fire,
                    aim: 0.4 + seed as f32,
                    aim_counts: counts_aim(hook, fire, true),
                };
                let d = log_prob_grad(&l, &a, 6.0);
                check("logp", &l, &d, &|x| f64::from(log_prob(x, &a, 6.0).total()));
            }
            // No hook, no fire: the aim does not count (probability and gradient).
            let quiet = PolicyAction {
                dir: 1,
                jump: true,
                hook: false,
                fire: false,
                aim: 1.0,
                aim_counts: false,
            };
            assert_eq!(log_prob(&l, &quiet, 6.0).aim, 0.0);
            let d = log_prob_grad(&l, &quiet, 6.0);
            assert_eq!((d.aim_c, d.aim_s), (0.0, 0.0));
            // A hook held while the hook is already out is not a throw: the aim does not count; a throw or a shot counts.
            assert!(!counts_aim(true, false, false) && counts_aim(true, false, true));
            assert!(counts_aim(false, true, false) && !counts_aim(false, false, true));
        }
    }

    #[test]
    fn entropy_and_kl_gradients_match_finite_differences() {
        for seed in 0..6 {
            let (l, r) = (logits(seed), logits(seed + 100));
            check("entropy", &l, &entropy_grad(&l), &|x| f64::from(entropy(x).total()));
            for aim_on in [false, true] {
                check("kl", &l, &kl_grad(&r, &l, 6.0, aim_on), &|x| {
                    f64::from(kl(&r, x, 6.0, aim_on).total())
                });
            }
            // KL(x || x) = 0 with a zero gradient, and KL is never negative.
            assert!(kl(&l, &l, 6.0, true).total().abs() < 1e-6);
            let g = kl_grad(&l, &l, 6.0, true);
            assert!(g.dir.iter().all(|x| x.abs() < 1e-6) && g.jump.abs() < 1e-6 && g.aim_c.abs() < 1e-6);
            assert!(kl(&r, &l, 6.0, true).total() >= 0.0);
        }
    }

    #[test]
    fn probabilities_normalise_and_the_normaliser_is_right() {
        // The three direction probabilities and a Bernoulli sum to one in log space.
        let l = logits(3);
        let s: f32 = (0..3)
            .map(|i| {
                log_prob(
                    &l,
                    &PolicyAction {
                        dir: i,
                        ..Default::default()
                    },
                    4.0,
                )
                .dir
                .exp()
            })
            .sum();
        assert!((s - 1.0).abs() < 1e-5);
        // The von Mises density integrates to one: sum exp(logp(theta)) dtheta over a fine grid.
        for kappa in [1.0f32, 4.0, 16.0, 60.0] {
            let n = 20_000;
            let a = |i: i32| -std::f32::consts::PI + 2.0 * std::f32::consts::PI * (i as f32 + 0.5) / n as f32;
            let total: f64 = (0..n)
                .map(|i| {
                    let act = PolicyAction {
                        hook: true,
                        aim_counts: true,
                        aim: a(i),
                        ..Default::default()
                    };
                    f64::from(log_prob(&l, &act, kappa).aim.exp())
                })
                .sum::<f64>()
                * 2.0
                * std::f64::consts::PI
                / f64::from(n);
            assert!((total - 1.0).abs() < 1e-3, "kappa {kappa}: {total}");
        }
        // A(kappa): the mean resultant length (known values: A(1) = 0.4463899658...).
        assert!((vm_resultant(1.0) - 0.446_39).abs() < 1e-4);
    }

    #[test]
    fn samples_follow_the_policy_and_are_reproducible() {
        let l = HeadLogits {
            dir: [0.5, -1.0, 1.5],
            jump: -0.7,
            hook: 0.9,
            fire: 0.2,
            aim_c: 0.3,
            aim_s: 0.8,
        };
        let kappa = 8.0;
        let n = 60_000;
        let mut rng = SplitMix64::new(11);
        let (mut cnt, mut jump, mut hook, mut fire) = ([0u32; 3], 0u32, 0u32, 0u32);
        let (mut cs, mut sn) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let a = sample(&l, kappa, &mut rng);
            cnt[usize::from(a.dir)] += 1;
            jump += u32::from(a.jump);
            hook += u32::from(a.hook);
            fire += u32::from(a.fire);
            cs += f64::from(a.aim.cos());
            sn += f64::from(a.aim.sin());
        }
        let p = l.dir_probs();
        for i in 0..3 {
            assert!(
                (f64::from(cnt[i]) / f64::from(n) - f64::from(p[i])).abs() < 0.01,
                "dir {i}"
            );
        }
        for (c, pr) in [(jump, l.jump_prob()), (hook, l.hook_prob()), (fire, l.fire_prob())] {
            assert!((f64::from(c) / f64::from(n) - f64::from(pr)).abs() < 0.01);
        }
        // The circular mean of the aim samples is mu, the resultant length A(kappa).
        let (mean_dir, r) = (sn.atan2(cs), (cs * cs + sn * sn).sqrt() / f64::from(n));
        assert!(
            (mean_dir - f64::from(aim_mu(&l))).abs() < 0.01,
            "mean {mean_dir} vs {}",
            aim_mu(&l)
        );
        assert!((r - f64::from(vm_resultant(kappa))).abs() < 0.01, "resultant {r}");
        // Same seed, same actions.
        let (mut a, mut b) = (SplitMix64::new(5), SplitMix64::new(5));
        for _ in 0..50 {
            assert_eq!(sample(&l, kappa, &mut a), sample(&l, kappa, &mut b));
        }
    }

    #[test]
    fn the_threshold_shift_makes_the_mode_the_thresholded_play() {
        let th = HeadThresholds {
            jump: 0.7,
            hook: 0.3,
            fire: 0.5,
        };
        // p just below / above the threshold, in raw logits.
        for (raw, t, press) in [
            (logit_of(0.69), 0.7, false),
            (logit_of(0.71), 0.7, true),
            (logit_of(0.31), 0.3, true),
        ] {
            let l = HeadLogits {
                jump: if t == 0.7 { raw } else { 0.0 },
                hook: if t == 0.3 { raw } else { 0.0 },
                ..Default::default()
            };
            let s = shifted(&l, &th);
            let (p_shift, on) = if t == 0.7 {
                (s.jump_prob(), l.jump_on(&th))
            } else {
                (s.hook_prob(), l.hook_on(&th))
            };
            assert_eq!(on, press);
            assert_eq!(
                p_shift >= 0.5,
                press,
                "the shifted policy's mode is the thresholded play"
            );
        }
        // Thresholds of 0.5 shift nothing.
        let l = logits(2);
        assert_eq!(shifted(&l, &HeadThresholds::default()), l);
    }

    #[test]
    fn argmax_decodes_like_the_brain() {
        let th = HeadThresholds {
            jump: 0.6,
            hook: 0.5,
            fire: 0.4,
        };
        let l = HeadLogits {
            dir: [0.1, 2.0, -1.0],
            jump: 0.3,  // p = 0.574 < 0.6
            hook: 0.01, // p = 0.5025 >= 0.5
            fire: -0.3, // p = 0.426 >= 0.4
            aim_c: -1.0,
            aim_s: 1.0,
        };
        let a = argmax(&l, &th);
        assert_eq!((a.dir, a.jump, a.hook, a.fire), (1, false, true, true));
        assert_eq!(a.aim, l.aim_angle());
    }
}
