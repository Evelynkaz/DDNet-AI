//! Acceptance criterion 2d: gradient checks, on the real S graph, for `>= 20` chosen parameters of
//! each trainable-parameter group (`a`, `b`, `theta`) plus the per-decision input currents.
//!
//! Two independent checks (review round 1, F3 — the original version only had the first, and
//! sampled indices *uniformly at random*: on the real S graph most `a`/`theta` entries have a
//! near-zero true gradient for a given random probe, so most of those 20-per-group checks were
//! silently passing against an implicit zero rather than actually exercising the analytic value):
//! - **(2d, FD)** central finite differences in `f32`, on the actual sparse `f32` model — the
//!   spec's own wording. Indices are now half the largest-`|analytic grad|` entries (guaranteed to
//!   be a real, non-trivial check) and half uniformly random (still covers the "small/zero
//!   gradient" case, just no longer *only* that); `abs_tol` is now scaled to the group's own
//!   [`ABS_TOL_GROUP_SCALE`]` * max|analytic grad|` rather than being a fixed constant that a
//!   near-zero group could trivially satisfy — see that constant's doc comment for why `1e-2`,
//!   not the `1e-3` first tried (too tight for `f32` central-FD's own noise floor at this scale).
//! - **(dual)** an `f64` **forward-mode dual-number** reference (`mod dual`, independent of
//!   `crate::backward`/`crate::model`/`crate::state` — same discipline as `tests/
//!   backward_correctness.rs`'s `mod dense`): for a handful of parameters per group, computes the
//!   exact derivative (no step size, no truncation error at all — dual arithmetic differentiates
//!   the same closed-form equations directly) and compares it to the crate's analytic `f32`
//!   backward. This is strictly stronger evidence than FD and doesn't share any code with the
//!   analytic backward pass, so it can't fail to notice a bug the same way that pass's own
//!   internals might.
//!
//! `spot_check_with_a_deliberately_zeroed_gradient_is_caught` demonstrates neither check is a
//! tautology: it corrupts one large-magnitude `grad_a` entry to `0.0` and confirms the comparison
//! actually reports a mismatch.
//!
//! `#[ignore]`d: needs the real compiled `.flyg` (outside the repo — see `tests/stability.rs`'s
//! doc comment for the project's standard convention here). Run with:
//! `cargo test -p ddai-fly --test backward_real_graph -- --ignored --nocapture`
//!
//! **Why the FD tolerance is looser than `tests/backward_correctness.rs`'s f64 checks:** central
//! finite differences computed in `f32` trade truncation error (`O(h^2)`, wants `h` large) against
//! rounding error (`O(eps/h)`, wants `h` small); `f32`'s `eps ~ 1.2e-7` puts the balance point
//! around `h ~ 1e-2` to `1e-3`, which limits the FD estimate itself to roughly 1e-3 to 1e-2
//! relative accuracy *before* accounting for `T * substeps` steps' worth of accumulated rounding
//! in the forward pass each FD evaluation reruns — so `5e-2` relative (documented, not the
//! `1e-4`/`1e-6` from the tiny hand-built graphs, which stay small enough for `f64` FD to be far
//! more precise) is the honest bar for *that* check, not a weakened one. The dual-number check has
//! no such excuse and is held to a much tighter bar (see its own section).

use std::path::PathBuf;

use ddai_fly::rng::SplitMix64;
use ddai_fly::{BackwardIndex, BpttScratch, FlyConfig, FlyModel, FlyParams, FlyState, TrajectoryRecorder, backward};

fn compiled_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set to find ~/aiddnet/data/connectome/compiled");
    PathBuf::from(home).join("aiddnet/data/connectome/compiled")
}

// ============================ forward-mode dual-number reference =============================

mod dual {
    use ddai_flyg::{Flyg, NeuronRole};

    /// `f64` value plus its derivative with respect to a single seeded scalar parameter — forward-
    /// mode automatic differentiation. Unlike `tests/backward_correctness.rs`'s central finite
    /// differences, there is no step size and no truncation error: every arithmetic operation
    /// below propagates the exact derivative through the chain rule, so `.d` at the end is exact
    /// up to ordinary `f64` rounding.
    #[derive(Clone, Copy, Debug)]
    pub struct Dual {
        pub v: f64,
        pub d: f64,
    }

    impl Dual {
        pub fn constant(v: f64) -> Self {
            Dual { v, d: 0.0 }
        }
        pub fn variable(v: f64) -> Self {
            Dual { v, d: 1.0 }
        }
    }

    impl std::ops::Add for Dual {
        type Output = Dual;
        fn add(self, o: Dual) -> Dual {
            Dual {
                v: self.v + o.v,
                d: self.d + o.d,
            }
        }
    }
    impl std::ops::Sub for Dual {
        type Output = Dual;
        fn sub(self, o: Dual) -> Dual {
            Dual {
                v: self.v - o.v,
                d: self.d - o.d,
            }
        }
    }
    impl std::ops::Mul for Dual {
        type Output = Dual;
        fn mul(self, o: Dual) -> Dual {
            Dual {
                v: self.v * o.v,
                d: self.d * o.v + self.v * o.d,
            }
        }
    }
    impl std::ops::AddAssign for Dual {
        fn add_assign(&mut self, o: Dual) {
            *self = *self + o;
        }
    }

    /// `softplus(x) = max(x,0) + ln1p(exp(-|x|))`, `d/dx = sigmoid(x)`.
    fn softplus(x: Dual) -> Dual {
        let v = x.v.max(0.0) + (-x.v.abs()).exp().ln_1p();
        let sig = if x.v >= 0.0 {
            1.0 / (1.0 + (-x.v).exp())
        } else {
            let e = x.v.exp();
            e / (1.0 + e)
        };
        Dual { v, d: sig * x.d }
    }

    /// `f(v) = r_max * tanh(relu(v) / r_max)`. Subgradient `0` at `v <= 0` — the same convention as
    /// `crate::activation::activation_derivative` (documented there); this is an
    /// independent re-derivation of the same choice, not a shared constant.
    fn activation(v: Dual, r_max: f64) -> Dual {
        if v.v <= 0.0 {
            Dual::constant(0.0)
        } else {
            let t = (v.v / r_max).tanh();
            Dual {
                v: r_max * t,
                d: (1.0 - t * t) * v.d,
            }
        }
    }

    /// `tau = min(dt + softplus(theta), tau_max)`, `decay = 1 - exp(-dt/tau)`. Subgradient `0`
    /// through the clamp when it's active — same convention as `crate::backward`'s
    /// `d_decay_d_theta_per_type`, independently re-derived here.
    fn decay_of(dt: f64, tau_max: f64, theta: Dual) -> Dual {
        let sp = softplus(theta);
        let tau_unclamped_v = dt + sp.v;
        if tau_unclamped_v >= tau_max {
            Dual::constant(1.0 - (-dt / tau_max).exp())
        } else {
            let tau_v = tau_unclamped_v;
            let tau_d = sp.d; // d(tau)/d(theta) via the chain so far; dt is a constant offset.
            let exp_v = (-dt / tau_v).exp();
            let decay_v = 1.0 - exp_v;
            let d_decay_d_tau = -exp_v * dt / (tau_v * tau_v);
            Dual {
                v: decay_v,
                d: d_decay_d_tau * tau_d,
            }
        }
    }

    fn inv_z(flyg: &Flyg, gamma: f64, i: usize) -> f64 {
        let z = (flyg.neuron_input_totals.full_connectome[i].max(1) as f64).powf(gamma);
        1.0 / z
    }

    fn input_indices(flyg: &Flyg) -> Vec<usize> {
        flyg.neurons
            .iter()
            .enumerate()
            .filter(|(_, nr)| matches!(nr.role, NeuronRole::InputVisual | NeuronRole::InputAscending))
            .map(|(i, _)| i)
            .collect()
    }

    fn output_indices(flyg: &Flyg) -> Vec<usize> {
        flyg.neurons
            .iter()
            .enumerate()
            .filter(|(_, nr)| nr.role == NeuronRole::Output)
            .map(|(i, _)| i)
            .collect()
    }

    /// Runs the exact forward model (FLY.md §4, this crate's discretisation — same equations
    /// `crate::model`/`crate::state` implement in `f32`, re-derived independently here in `f64`
    /// dual arithmetic) for `inputs.len()` decisions from `v_init`, and returns
    /// `Σ_t dot(grad_dn[t], dn_rates[t])` as a `Dual` — the same linear-functional-of-the-outputs
    /// trick `tests/backward_correctness.rs` uses, so whichever one of `a`/`b`/`theta`/`inputs` was
    /// seeded with [`Dual::variable`], `.d` on the result is the exact `dL/d(that one parameter)`.
    #[allow(clippy::too_many_arguments)]
    pub fn loss(
        flyg: &Flyg,
        dt: f64,
        tau_max: f64,
        gamma: f64,
        r_max: f64,
        substeps: usize,
        a: &[Dual],
        b: &[Dual],
        theta: &[Dual],
        v_init: &[f64],
        inputs: &[Vec<Dual>],
        grad_dn: &[Vec<f64>],
    ) -> Dual {
        let n = flyg.neurons.len();
        let ins = input_indices(flyg);
        let outs = output_indices(flyg);

        let mut v_prev: Vec<Dual> = v_init.iter().map(|&x| Dual::constant(x)).collect();
        let mut r_prev: Vec<Dual> = v_prev.iter().map(|&v| activation(v, r_max)).collect();
        let mut loss = Dual::constant(0.0);

        for (t, input_t) in inputs.iter().enumerate() {
            let mut r_before_last = r_prev.clone();
            for loc in 0..substeps {
                let mut v_inf = vec![Dual::constant(0.0); n];
                for (i, nr) in flyg.neurons.iter().enumerate() {
                    v_inf[i] = b[nr.type_index as usize];
                }
                #[allow(clippy::needless_range_loop)]
                // `post` is also used as a plain index/u32 below, not just into `v_inf`
                for post in 0..n {
                    let iz = inv_z(flyg, gamma, post);
                    for (pre, n_ij, tp) in flyg.edges.row(post as u32) {
                        let tpd = &flyg.type_pairs[tp as usize];
                        let sign = f64::from(flyg.types[tpd.pre_type as usize].sign.as_i8());
                        let alpha = softplus(a[tpd.shared_param_id as usize]);
                        let coeff = Dual::constant(sign * f64::from(n_ij) * iz);
                        v_inf[post] += (alpha * coeff) * r_prev[pre as usize];
                    }
                }
                for (k, &idx) in ins.iter().enumerate() {
                    v_inf[idx] += input_t[k];
                }
                if loc == substeps - 1 {
                    r_before_last = r_prev.clone();
                }
                let mut v_new = vec![Dual::constant(0.0); n];
                for (i, nr) in flyg.neurons.iter().enumerate() {
                    let decay = decay_of(dt, tau_max, theta[nr.type_index as usize]);
                    v_new[i] = v_prev[i] + decay * (v_inf[i] - v_prev[i]);
                }
                r_prev = v_new.iter().map(|&v| activation(v, r_max)).collect();
                v_prev = v_new;
            }
            for (&idx, &g) in outs.iter().zip(&grad_dn[t]) {
                let dn = (r_before_last[idx] + r_prev[idx]) * Dual::constant(0.5);
                loss += dn * Dual::constant(g);
            }
        }
        loss
    }
}

// =================================== shared test scaffolding ==================================

/// Runs `t_decisions` decisions (plain `step_decision`, the actual inference hot path — not the
/// recording variant) from `v_init` and returns `Σ dot(grad_dn[t], dn_rates[t])`, i.e. the same
/// linear-functional-of-the-outputs trick `tests/backward_correctness.rs` uses, here in `f32`
/// against the real model.
fn loss(model: &FlyModel, v_init: &[f32], inputs: &[Vec<f32>], grad_dn: &[Vec<f32>]) -> f64 {
    let mut state = FlyState::new(model);
    state.set_v(model, v_init);
    let mut total = 0.0f64;
    for (input_t, g) in inputs.iter().zip(grad_dn) {
        let out = state.step_decision(model, input_t);
        for (x, y) in out.dn_rates.iter().zip(g) {
            total += f64::from(*x) * f64::from(*y);
        }
    }
    total
}

fn fd_grad_a(
    model: &mut FlyModel,
    base: &FlyParams,
    index: usize,
    h: f32,
    v_init: &[f32],
    inputs: &[Vec<f32>],
    grad_dn: &[Vec<f32>],
) -> f64 {
    let mut plus = base.clone();
    plus.a[index] += h;
    model.set_params(plus).unwrap();
    let l_plus = loss(model, v_init, inputs, grad_dn);

    let mut minus = base.clone();
    minus.a[index] -= h;
    model.set_params(minus).unwrap();
    let l_minus = loss(model, v_init, inputs, grad_dn);

    model.set_params(base.clone()).unwrap();
    (l_plus - l_minus) / (2.0 * f64::from(h))
}

fn fd_grad_b(
    model: &mut FlyModel,
    base: &FlyParams,
    index: usize,
    h: f32,
    v_init: &[f32],
    inputs: &[Vec<f32>],
    grad_dn: &[Vec<f32>],
) -> f64 {
    let mut plus = base.clone();
    plus.b[index] += h;
    model.set_params(plus).unwrap();
    let l_plus = loss(model, v_init, inputs, grad_dn);

    let mut minus = base.clone();
    minus.b[index] -= h;
    model.set_params(minus).unwrap();
    let l_minus = loss(model, v_init, inputs, grad_dn);

    model.set_params(base.clone()).unwrap();
    (l_plus - l_minus) / (2.0 * f64::from(h))
}

fn fd_grad_theta(
    model: &mut FlyModel,
    base: &FlyParams,
    index: usize,
    h: f32,
    v_init: &[f32],
    inputs: &[Vec<f32>],
    grad_dn: &[Vec<f32>],
) -> f64 {
    let mut plus = base.clone();
    plus.theta[index] += h;
    model.set_params(plus).unwrap();
    let l_plus = loss(model, v_init, inputs, grad_dn);

    let mut minus = base.clone();
    minus.theta[index] -= h;
    model.set_params(minus).unwrap();
    let l_minus = loss(model, v_init, inputs, grad_dn);

    model.set_params(base.clone()).unwrap();
    (l_plus - l_minus) / (2.0 * f64::from(h))
}

fn fd_grad_input(
    model: &FlyModel,
    v_init: &[f32],
    inputs: &[Vec<f32>],
    grad_dn: &[Vec<f32>],
    t: usize,
    k: usize,
    h: f32,
) -> f64 {
    let mut plus = inputs.to_vec();
    plus[t][k] += h;
    let l_plus = loss(model, v_init, &plus, grad_dn);

    let mut minus = inputs.to_vec();
    minus[t][k] -= h;
    let l_minus = loss(model, v_init, &minus, grad_dn);

    (l_plus - l_minus) / (2.0 * f64::from(h))
}

/// `true` (rather than panicking) so a caller can also demonstrate the *failure* path (review
/// round 1, F3's "show a deliberately zeroed gradient now fails the test").
fn close_relaxed(got: f64, want: f64, rel_tol: f64, abs_tol: f64) -> bool {
    let diff = (got - want).abs();
    diff <= abs_tol + rel_tol * want.abs()
}

fn assert_close_relaxed(got: f64, want: f64, rel_tol: f64, abs_tol: f64, label: &str) {
    assert!(
        close_relaxed(got, want, rel_tol, abs_tol),
        "{label}: got {got}, want {want}, diff {} (rel_tol={rel_tol}, abs_tol={abs_tol})",
        (got - want).abs()
    );
}

/// Half the largest-`|magnitudes|` indices plus half uniformly random ones (deduplicated) — review
/// round 1, F3: sampling *only* uniformly at random meant most checks on a graph this size landed
/// on a near-zero true gradient and would have passed against an implicit zero.
fn top_and_random_indices(rng: &mut SplitMix64, magnitudes: &[f32], num_samples: usize) -> Vec<usize> {
    let half = num_samples / 2;
    let mut by_magnitude: Vec<usize> = (0..magnitudes.len()).collect();
    by_magnitude.sort_by(|&a, &b| magnitudes[b].abs().partial_cmp(&magnitudes[a].abs()).unwrap());
    let mut chosen: Vec<usize> = by_magnitude.into_iter().take(half).collect();
    let mut seen: std::collections::HashSet<usize> = chosen.iter().copied().collect();
    while chosen.len() < num_samples && seen.len() < magnitudes.len() {
        let idx = (rng.next_u64() as usize) % magnitudes.len();
        if seen.insert(idx) {
            chosen.push(idx);
        }
    }
    chosen
}

fn max_abs(xs: &[f32]) -> f32 {
    xs.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

/// Scales a group's `abs_tol` to `ABS_TOL_GROUP_SCALE * max|analytic grad|` in that group (review
/// round 1, F3). Empirically `1e-3` (the reviewer's own suggested starting point) is *tighter*
/// than `f32` central-FD's own absolute rounding-noise floor for `theta` on this graph at any `h`
/// tried (`5e-3` through `5e-2`; noise floor observed ~`1e-5`-`2e-5` regardless) — a perfectly
/// correct analytic gradient at a small-but-real magnitude could fail purely from FD's own noise,
/// not from a bug. `1e-2` clears that noise floor with margin while still being far tighter than
/// the group's own typical entry (so `spot_check_with_a_deliberately_zeroed_gradient_is_caught`
/// still fails obviously — a corruption is nowhere near `1%` of the group's largest entry, let
/// alone this scale applied to a *zeroed* value).
const ABS_TOL_GROUP_SCALE: f64 = 1e-2;

struct RealGraphFixture {
    model: FlyModel,
    base_params: FlyParams,
    v_init: Vec<f32>,
    inputs: Vec<Vec<f32>>,
    grad_dn: Vec<Vec<f32>>,
    t_decisions: usize,
}

fn build_fixture() -> RealGraphFixture {
    let path = compiled_dir().join("fly-S-v1.flyg");
    let flyg = ddai_flyg::load(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));

    let config = FlyConfig {
        substeps_per_decision: 4,
        ..FlyConfig::default()
    };
    let base_params = FlyParams::init_default(&flyg, &config, 42);
    let model = FlyModel::new(flyg, config, base_params.clone()).expect("build model from real S graph");

    let mut rng = SplitMix64::new(20_260_927);

    // Start from a converged resting state (a realistic operating point), not V=0.
    let mut warm_state = FlyState::new(&model);
    let warm_report = warm_state.warm_up(&model);
    assert!(warm_report.converged, "warm-up should converge on the real S graph");
    let v_init: Vec<f32> = warm_state.v().to_vec();

    let t_decisions = 2usize;
    let num_inputs = model.num_inputs();
    let num_outputs = model.num_outputs();

    let inputs: Vec<Vec<f32>> = (0..t_decisions)
        .map(|_| (0..num_inputs).map(|_| rng.next_f32_unit() * 0.6 - 0.3).collect())
        .collect();
    let grad_dn: Vec<Vec<f32>> = (0..t_decisions)
        .map(|_| (0..num_outputs).map(|_| rng.next_f32_unit() * 2.0 - 1.0).collect())
        .collect();

    RealGraphFixture {
        model,
        base_params,
        v_init,
        inputs,
        grad_dn,
        t_decisions,
    }
}

fn analytic_gradients(fx: &RealGraphFixture) -> ddai_fly::BpttGradients {
    let mut state = FlyState::new(&fx.model);
    state.set_v(&fx.model, &fx.v_init);
    let substeps = fx.model.config().substeps_per_decision as usize;
    let mut recorder = TrajectoryRecorder::new(fx.model.num_neurons(), fx.t_decisions * substeps);
    for input_t in &fx.inputs {
        state.step_decision_recording(&fx.model, input_t, &mut recorder);
    }
    let index = BackwardIndex::build(&fx.model);
    let mut scratch = BpttScratch::new(&fx.model);
    let grad_dn_refs: Vec<&[f32]> = fx.grad_dn.iter().map(Vec::as_slice).collect();
    backward(
        &fx.model,
        &index,
        &recorder,
        &fx.v_init,
        fx.t_decisions,
        &grad_dn_refs,
        &[],
        false,
        &mut scratch,
    )
}

// ======================================= the tests =============================================

#[test]
#[ignore]
fn spot_check_gradients_on_the_real_s_graph() {
    let mut fx = build_fixture();
    let analytic = analytic_gradients(&fx);

    let mut rng = SplitMix64::new(20_260_928);
    let num_samples = 20usize;
    // `h_param = 2e-2` (not `5e-3`): empirically (see the diagnostic sweep that motivated this,
    // reproducible by temporarily re-adding a loop over candidate `h` values here) `theta`'s FD
    // estimate at `h = 5e-3` has an *absolute* noise floor of roughly `1e-5`-`2e-5` in `f32` over
    // this graph's `T * substeps` steps — bigger than `1e-3 * max|grad_theta|` for a random
    // small-magnitude entry, so a perfectly correct analytic gradient could still fail the
    // comparison purely from FD's own rounding noise. `h = 2e-2` cuts that noise floor by roughly
    // an order of magnitude here (still a small perturbation relative to typical `a`/`b`/`theta`
    // magnitudes) without introducing enough extra truncation error to matter.
    let h_param = 2e-2f32;
    let h_input = 5e-3f32;
    let rel_tol = 5e-2;

    let abs_tol_a = ABS_TOL_GROUP_SCALE * f64::from(max_abs(&analytic.grad_a));
    let abs_tol_b = ABS_TOL_GROUP_SCALE * f64::from(max_abs(&analytic.grad_b));
    let abs_tol_theta = ABS_TOL_GROUP_SCALE * f64::from(max_abs(&analytic.grad_theta));
    let all_grad_inputs: Vec<f32> = analytic.grad_inputs.iter().flatten().copied().collect();
    let abs_tol_inputs = ABS_TOL_GROUP_SCALE * f64::from(max_abs(&all_grad_inputs));

    let a_indices = top_and_random_indices(&mut rng, &analytic.grad_a, num_samples);
    for idx in &a_indices {
        let fd = fd_grad_a(
            &mut fx.model,
            &fx.base_params,
            *idx,
            h_param,
            &fx.v_init,
            &fx.inputs,
            &fx.grad_dn,
        );
        assert_close_relaxed(
            f64::from(analytic.grad_a[*idx]),
            fd,
            rel_tol,
            abs_tol_a,
            &format!("grad_a[{idx}]"),
        );
    }

    let b_indices = top_and_random_indices(&mut rng, &analytic.grad_b, num_samples);
    for idx in &b_indices {
        let fd = fd_grad_b(
            &mut fx.model,
            &fx.base_params,
            *idx,
            h_param,
            &fx.v_init,
            &fx.inputs,
            &fx.grad_dn,
        );
        assert_close_relaxed(
            f64::from(analytic.grad_b[*idx]),
            fd,
            rel_tol,
            abs_tol_b,
            &format!("grad_b[{idx}]"),
        );
    }

    let theta_indices = top_and_random_indices(&mut rng, &analytic.grad_theta, num_samples);
    for idx in &theta_indices {
        let fd = fd_grad_theta(
            &mut fx.model,
            &fx.base_params,
            *idx,
            h_param,
            &fx.v_init,
            &fx.inputs,
            &fx.grad_dn,
        );
        assert_close_relaxed(
            f64::from(analytic.grad_theta[*idx]),
            fd,
            rel_tol,
            abs_tol_theta,
            &format!("grad_theta[{idx}]"),
        );
    }

    let num_inputs = fx.model.num_inputs();
    let flat_input_indices = top_and_random_indices(&mut rng, &all_grad_inputs, num_samples);
    for flat in &flat_input_indices {
        let t = flat / num_inputs;
        let k = flat % num_inputs;
        let fd = fd_grad_input(&fx.model, &fx.v_init, &fx.inputs, &fx.grad_dn, t, k, h_input);
        assert_close_relaxed(
            f64::from(analytic.grad_inputs[t][k]),
            fd,
            rel_tol,
            abs_tol_inputs,
            &format!("grad_inputs[{t}][{k}]"),
        );
    }

    eprintln!(
        "spot-checked {} `a`, {} `b`, {} `theta`, {} input-current gradients (half largest-|g|, half random) on the \
         real S graph ({} neurons, {} edges) — all within rel_tol={rel_tol}, abs_tol scaled to each group's own \
         {ABS_TOL_GROUP_SCALE:.0e}*max|analytic_grad| (a={abs_tol_a:.2e}, b={abs_tol_b:.2e}, theta={abs_tol_theta:.2e}, \
         inputs={abs_tol_inputs:.2e})",
        a_indices.len(),
        b_indices.len(),
        theta_indices.len(),
        flat_input_indices.len(),
        fx.model.num_neurons(),
        fx.model.flyg().edges.num_edges()
    );
}

/// Review round 1, F3: proves the check above isn't a tautology — corrupting one large-magnitude
/// `grad_a` entry to `0.0` must make the *exact same* comparison this test uses report a mismatch.
#[test]
#[ignore]
fn spot_check_with_a_deliberately_zeroed_gradient_is_caught() {
    let fx = build_fixture();
    let analytic = analytic_gradients(&fx);

    let abs_tol_a = ABS_TOL_GROUP_SCALE * f64::from(max_abs(&analytic.grad_a));
    let rel_tol = 5e-2;

    // The largest-magnitude entry: corrupting anything smaller would risk `abs_tol_a` (scaled to
    // the *group's* max) accidentally covering the corruption too.
    let (idx, &true_value) = analytic
        .grad_a
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .expect("grad_a is non-empty on the real S graph");
    assert!(
        true_value.abs() > 1e-3,
        "the largest grad_a entry ({true_value}) should be comfortably nonzero for this demonstration to mean anything"
    );

    let corrupted = 0.0f32;
    assert!(
        !close_relaxed(f64::from(corrupted), f64::from(true_value), rel_tol, abs_tol_a),
        "a deliberately zeroed grad_a[{idx}] (true value {true_value}) must NOT pass the same tolerance check \
         real gradients are held to — if it does, the check is too loose to catch a real bug"
    );
    eprintln!(
        "confirmed: grad_a[{idx}] corrupted from {true_value} to {corrupted} is correctly rejected \
         (rel_tol={rel_tol}, abs_tol={abs_tol_a:.2e})"
    );
}

/// The dual-number check (see the module doc comment and `mod dual`): exact derivatives, no FD
/// step size, for a handful of parameters per group. Tolerance is the same order as `tests/
/// backward_correctness.rs`'s (2b) f32-vs-f64-dense check (`1e-4` relative, `1e-6` absolute) since
/// there is no FD truncation error to excuse a looser bound here — the only source of disagreement
/// is `f32` vs. `f64` arithmetic over `T * substeps` steps on a ~50k-edge graph.
#[test]
#[ignore]
fn dual_number_exact_gradient_matches_analytic_on_the_real_s_graph() {
    let fx = build_fixture();
    let analytic = analytic_gradients(&fx);
    let flyg = fx.model.flyg();
    let config = fx.model.config();
    let dt = f64::from(config.dt_s());
    let tau_max = f64::from(config.tau_max_s);
    let gamma = f64::from(config.gamma);
    let r_max = f64::from(config.r_max);
    let substeps = config.substeps_per_decision as usize;

    let a64: Vec<f64> = fx.base_params.a.iter().map(|&x| f64::from(x)).collect();
    let b64: Vec<f64> = fx.base_params.b.iter().map(|&x| f64::from(x)).collect();
    let theta64: Vec<f64> = fx.base_params.theta.iter().map(|&x| f64::from(x)).collect();
    let v_init64: Vec<f64> = fx.v_init.iter().map(|&x| f64::from(x)).collect();
    let inputs64: Vec<Vec<f64>> = fx
        .inputs
        .iter()
        .map(|row| row.iter().map(|&x| f64::from(x)).collect())
        .collect();
    let grad_dn64: Vec<Vec<f64>> = fx
        .grad_dn
        .iter()
        .map(|row| row.iter().map(|&x| f64::from(x)).collect())
        .collect();

    let all_const = |xs: &[f64]| -> Vec<dual::Dual> { xs.iter().map(|&x| dual::Dual::constant(x)).collect() };
    let inputs_const = |rows: &[Vec<f64>]| -> Vec<Vec<dual::Dual>> {
        rows.iter()
            .map(|row| row.iter().map(|&x| dual::Dual::constant(x)).collect())
            .collect()
    };

    let rel_tol = 1e-4;
    let abs_tol = 1e-6;
    let mut rng = SplitMix64::new(20_260_929);
    let num_per_group = 5usize;
    let mut checked = 0usize;

    for &idx in &top_and_random_indices(&mut rng, &analytic.grad_a, num_per_group) {
        let mut a = all_const(&a64);
        a[idx] = dual::Dual::variable(a64[idx]);
        let result = dual::loss(
            flyg,
            dt,
            tau_max,
            gamma,
            r_max,
            substeps,
            &a,
            &all_const(&b64),
            &all_const(&theta64),
            &v_init64,
            &inputs_const(&inputs64),
            &grad_dn64,
        );
        assert_close_relaxed(
            f64::from(analytic.grad_a[idx]),
            result.d,
            rel_tol,
            abs_tol,
            &format!("grad_a[{idx}] (dual)"),
        );
        checked += 1;
    }

    for &idx in &top_and_random_indices(&mut rng, &analytic.grad_b, num_per_group) {
        let mut b = all_const(&b64);
        b[idx] = dual::Dual::variable(b64[idx]);
        let result = dual::loss(
            flyg,
            dt,
            tau_max,
            gamma,
            r_max,
            substeps,
            &all_const(&a64),
            &b,
            &all_const(&theta64),
            &v_init64,
            &inputs_const(&inputs64),
            &grad_dn64,
        );
        assert_close_relaxed(
            f64::from(analytic.grad_b[idx]),
            result.d,
            rel_tol,
            abs_tol,
            &format!("grad_b[{idx}] (dual)"),
        );
        checked += 1;
    }

    for &idx in &top_and_random_indices(&mut rng, &analytic.grad_theta, num_per_group) {
        let mut theta = all_const(&theta64);
        theta[idx] = dual::Dual::variable(theta64[idx]);
        let result = dual::loss(
            flyg,
            dt,
            tau_max,
            gamma,
            r_max,
            substeps,
            &all_const(&a64),
            &all_const(&b64),
            &theta,
            &v_init64,
            &inputs_const(&inputs64),
            &grad_dn64,
        );
        assert_close_relaxed(
            f64::from(analytic.grad_theta[idx]),
            result.d,
            rel_tol,
            abs_tol,
            &format!("grad_theta[{idx}] (dual)"),
        );
        checked += 1;
    }

    let num_inputs = fx.model.num_inputs();
    let all_grad_inputs: Vec<f32> = analytic.grad_inputs.iter().flatten().copied().collect();
    for &flat in &top_and_random_indices(&mut rng, &all_grad_inputs, num_per_group) {
        let t = flat / num_inputs;
        let k = flat % num_inputs;
        let mut inputs = inputs_const(&inputs64);
        inputs[t][k] = dual::Dual::variable(inputs64[t][k]);
        let result = dual::loss(
            flyg,
            dt,
            tau_max,
            gamma,
            r_max,
            substeps,
            &all_const(&a64),
            &all_const(&b64),
            &all_const(&theta64),
            &v_init64,
            &inputs,
            &grad_dn64,
        );
        assert_close_relaxed(
            f64::from(analytic.grad_inputs[t][k]),
            result.d,
            rel_tol,
            abs_tol,
            &format!("grad_inputs[{t}][{k}] (dual)"),
        );
        checked += 1;
    }

    assert_eq!(checked, num_per_group * 4);
    eprintln!(
        "dual-number exact-gradient check: {checked} parameters ({num_per_group} per group) on the real S graph \
         all within rel_tol={rel_tol}, abs_tol={abs_tol}"
    );
}
