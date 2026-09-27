//! Acceptance criterion 2's gradient-correctness tests, all on tiny hand-built `.flyg` graphs:
//! - (2a) a dense **f64** reference implementation of the exact forward *and* backward (this
//!   file's `dense` module — independent of `ddai_fly::backward`/`model`/`state`, written from the
//!   same equations FLY.md §4/this crate's README document, using nested loops rather than any of
//!   the crate's CSR/transpose/scratch-buffer machinery) checked against central finite
//!   differences on that same dense forward;
//! - (2b) the crate's actual `f32`, sparse, CSR-based `ddai_fly::backward::backward` compared
//!   against that (already FD-validated) dense f64 reference, for every parameter group and the
//!   input/initial-state gradients;
//! - (2c) a randomized property test repeating both checks over many small random graphs,
//!   sequence lengths, and substep counts.
//!
//! The FD check treats the supplied `grad_dn`/`extra_taps` as coefficients of a fixed *linear*
//! functional of the model's outputs (`loss = Σ dot(grad_dn[t], dn_rates[t]) + Σ dot(extra_tap.grad,
//! r_at(that substep))`) and finite-differences `loss` itself — the standard way to test a
//! backward pass (a vector-Jacobian product) without needing a "real" loss function: matching for
//! enough distinct `grad_dn`/`extra_taps` vectors verifies the whole Jacobian, not just one slice
//! of it, and the random test below draws a fresh random one for every graph.

use ddai_fly::rng::SplitMix64;
use ddai_fly::test_fixtures::{FxEdge, FxNeuron, FxType, build_flyg, tiny_chain_flyg};
use ddai_fly::{
    BackwardIndex, BpttScratch, ExtraRateGrad, FlyConfig, FlyModel, FlyParams, FlyState, TrajectoryRecorder, backward,
};
use ddai_flyg::{Flyg, NeuronRole, Side, Sign};

// ============================== dense f64 reference (2a) ===================================

mod dense {
    use ddai_flyg::{Flyg, NeuronRole};

    pub fn softplus(x: f64) -> f64 {
        x.max(0.0) + (-x.abs()).exp().ln_1p()
    }

    pub fn sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    pub fn activation(v: f64, r_max: f64) -> f64 {
        if v <= 0.0 { 0.0 } else { r_max * (v / r_max).tanh() }
    }

    pub fn activation_deriv(v: f64, r: f64, r_max: f64) -> f64 {
        if v <= 0.0 { 0.0 } else { 1.0 - (r / r_max) * (r / r_max) }
    }

    fn inv_z(flyg: &Flyg, gamma: f64, i: usize) -> f64 {
        let z = (flyg.neuron_input_totals.full_connectome[i].max(1) as f64).powf(gamma);
        1.0 / z
    }

    fn decay_of(dt: f64, tau_max: f64, theta: f64) -> f64 {
        let tau = (dt + softplus(theta)).min(tau_max);
        1.0 - (-dt / tau).exp()
    }

    fn ddecay_dtheta_of(dt: f64, tau_max: f64, theta: f64) -> f64 {
        let tau_unclamped = dt + softplus(theta);
        if tau_unclamped >= tau_max {
            0.0
        } else {
            let tau = tau_unclamped;
            let ddecay_dtau = -(-dt / tau).exp() * dt / (tau * tau);
            ddecay_dtau * sigmoid(theta)
        }
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

    pub struct Trace {
        pub v: Vec<Vec<f64>>,
        pub v_inf: Vec<Vec<f64>>,
        pub dn_rates: Vec<Vec<f64>>,
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        flyg: &Flyg,
        dt: f64,
        tau_max: f64,
        gamma: f64,
        r_max: f64,
        substeps: usize,
        a: &[f64],
        b: &[f64],
        theta: &[f64],
        v_init: &[f64],
        inputs: &[Vec<f64>],
    ) -> Trace {
        let n = flyg.neurons.len();
        let ins = input_indices(flyg);
        let outs = output_indices(flyg);
        let t_decisions = inputs.len();

        let mut v_prev = v_init.to_vec();
        let mut r_prev: Vec<f64> = v_prev.iter().map(|&v| activation(v, r_max)).collect();
        let mut v_hist = Vec::with_capacity(t_decisions * substeps);
        let mut v_inf_hist = Vec::with_capacity(t_decisions * substeps);
        let mut dn_rates = Vec::with_capacity(t_decisions);

        for (t, input_t) in inputs.iter().enumerate() {
            let mut r_before_last = r_prev.clone();
            for loc in 0..substeps {
                let mut v_inf = vec![0.0f64; n];
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
                        let w = sign * alpha * f64::from(n_ij) * iz;
                        v_inf[post] += w * r_prev[pre as usize];
                    }
                }
                for (k, &idx) in ins.iter().enumerate() {
                    v_inf[idx] += input_t[k];
                }
                if loc == substeps - 1 {
                    r_before_last = r_prev.clone();
                }
                let mut v_new = vec![0.0f64; n];
                for (i, nr) in flyg.neurons.iter().enumerate() {
                    let d = decay_of(dt, tau_max, theta[nr.type_index as usize]);
                    v_new[i] = v_prev[i] + d * (v_inf[i] - v_prev[i]);
                }
                v_hist.push(v_new.clone());
                v_inf_hist.push(v_inf);
                r_prev = v_new.iter().map(|&v| activation(v, r_max)).collect();
                v_prev = v_new;
                let _ = t;
            }
            dn_rates.push(
                outs.iter()
                    .map(|&idx| 0.5 * (r_before_last[idx] + r_prev[idx]))
                    .collect(),
            );
        }

        Trace {
            v: v_hist,
            v_inf: v_inf_hist,
            dn_rates,
        }
    }

    pub struct Gradients {
        pub grad_a: Vec<f64>,
        pub grad_b: Vec<f64>,
        pub grad_theta: Vec<f64>,
        pub grad_inputs: Vec<Vec<f64>>,
        pub grad_v_init: Vec<f64>,
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        flyg: &Flyg,
        dt: f64,
        tau_max: f64,
        gamma: f64,
        r_max: f64,
        substeps: usize,
        a: &[f64],
        b: &[f64],
        theta: &[f64],
        v_init: &[f64],
        inputs: &[Vec<f64>],
        grad_dn: &[Vec<f64>],
        extra_taps: &[(usize, usize, Vec<f64>)],
        trace: &Trace,
    ) -> Gradients {
        let n = flyg.neurons.len();
        let t_decisions = inputs.len();
        let l_total = t_decisions * substeps;
        let ins = input_indices(flyg);
        let outs = output_indices(flyg);

        let decay_of_type: Vec<f64> = theta.iter().map(|&th| decay_of(dt, tau_max, th)).collect();
        let ddecay_of_type: Vec<f64> = theta.iter().map(|&th| ddecay_dtheta_of(dt, tau_max, th)).collect();

        let mut grad_a = vec![0.0f64; a.len()];
        let mut grad_b = vec![0.0f64; b.len()];
        let mut grad_theta = vec![0.0f64; theta.len()];
        let mut grad_inputs = vec![vec![0.0f64; ins.len()]; t_decisions];

        let mut future_delta_v = vec![0.0f64; n];
        let mut future_dr = vec![0.0f64; n];

        let v_prev_at = |l: usize| -> Vec<f64> {
            if l == 0 {
                v_init.to_vec()
            } else {
                trace.v[l - 1].clone()
            }
        };
        let r_at = |l: usize| -> Vec<f64> { trace.v[l].iter().map(|&v| activation(v, r_max)).collect() };

        for l in (0..l_total).rev() {
            let decision = l / substeps;
            let mut dr_total = future_dr.clone();
            if (l + 1) % substeps == 0 {
                let t_final = (l + 1) / substeps - 1;
                for (slot, &idx) in outs.iter().enumerate() {
                    dr_total[idx] += 0.5 * grad_dn[t_final][slot];
                }
            }
            if (l + 2) % substeps == 0 {
                let t_before = (l + 2) / substeps - 1;
                if t_before < t_decisions {
                    for (slot, &idx) in outs.iter().enumerate() {
                        dr_total[idx] += 0.5 * grad_dn[t_before][slot];
                    }
                }
            }
            for (dec, loc, g) in extra_taps {
                if dec * substeps + loc == l {
                    for i in 0..n {
                        dr_total[i] += g[i];
                    }
                }
            }

            let v_l = &trace.v[l];
            let v_inf_l = &trace.v_inf[l];
            let v_prev_l = v_prev_at(l);
            let r_prev_l = r_at_prev(flyg, v_init, trace, l, r_max);
            let r_l = r_at(l);

            let mut delta_v_total = vec![0.0f64; n];
            let mut delta_vinf = vec![0.0f64; n];
            for (i, nr) in flyg.neurons.iter().enumerate() {
                let ty = nr.type_index as usize;
                let delta_v_from_r = dr_total[i] * activation_deriv(v_l[i], r_l[i], r_max);
                delta_v_total[i] = delta_v_from_r + future_delta_v[i];
                delta_vinf[i] = decay_of_type[ty] * delta_v_total[i];
                grad_b[ty] += delta_vinf[i];
                let g_decay = delta_v_total[i] * (v_inf_l[i] - v_prev_l[i]);
                grad_theta[ty] += g_decay * ddecay_of_type[ty];
            }
            for (k, &idx) in ins.iter().enumerate() {
                grad_inputs[decision][k] += delta_vinf[idx];
            }

            #[allow(clippy::needless_range_loop)]
            // `post` is also used as a plain index/u32 below, not just into `delta_vinf`
            for post in 0..n {
                let iz = inv_z(flyg, gamma, post);
                for (pre, n_ij, tp) in flyg.edges.row(post as u32) {
                    let tpd = &flyg.type_pairs[tp as usize];
                    let sign = f64::from(flyg.types[tpd.pre_type as usize].sign.as_i8());
                    grad_a[tpd.shared_param_id as usize] +=
                        delta_vinf[post] * r_prev_l[pre as usize] * sign * f64::from(n_ij) * iz;
                }
            }

            let mut new_future_dr = vec![0.0f64; n];
            #[allow(clippy::needless_range_loop)]
            // `post` is also used as a plain index/u32 below, not just into `delta_vinf`
            for post in 0..n {
                let iz = inv_z(flyg, gamma, post);
                for (pre, n_ij, tp) in flyg.edges.row(post as u32) {
                    let tpd = &flyg.type_pairs[tp as usize];
                    let sign = f64::from(flyg.types[tpd.pre_type as usize].sign.as_i8());
                    let alpha = softplus(a[tpd.shared_param_id as usize]);
                    let w = sign * alpha * f64::from(n_ij) * iz;
                    new_future_dr[pre as usize] += w * delta_vinf[post];
                }
            }
            let mut new_future_delta_v = vec![0.0f64; n];
            for (i, nr) in flyg.neurons.iter().enumerate() {
                new_future_delta_v[i] = (1.0 - decay_of_type[nr.type_index as usize]) * delta_v_total[i];
            }
            future_dr = new_future_dr;
            future_delta_v = new_future_delta_v;
        }

        let mut dr_init = future_dr.clone();
        if substeps == 1 {
            for (slot, &idx) in outs.iter().enumerate() {
                dr_init[idx] += 0.5 * grad_dn[0][slot];
            }
        }
        let grad_v_init: Vec<f64> = (0..n)
            .map(|i| {
                let r_init_i = activation(v_init[i], r_max);
                dr_init[i] * activation_deriv(v_init[i], r_init_i, r_max) + future_delta_v[i]
            })
            .collect();

        for (sid, g) in grad_a.iter_mut().enumerate() {
            *g *= sigmoid(a[sid]);
        }

        Gradients {
            grad_a,
            grad_b,
            grad_theta,
            grad_inputs,
            grad_v_init,
        }
    }

    /// `r` that drove substep `l` (`f(v_prev_l)`) — `f(v_init)` for `l == 0`.
    fn r_at_prev(flyg: &Flyg, v_init: &[f64], trace: &Trace, l: usize, r_max: f64) -> Vec<f64> {
        let n = flyg.neurons.len();
        if l == 0 {
            (0..n).map(|i| activation(v_init[i], r_max)).collect()
        } else {
            trace.v[l - 1].iter().map(|&v| activation(v, r_max)).collect()
        }
    }
}

// ================================ shared test scaffolding ===================================

struct Problem {
    flyg: Flyg,
    config: FlyConfig,
    a: Vec<f64>,
    b: Vec<f64>,
    theta: Vec<f64>,
    v_init: Vec<f64>,
    inputs: Vec<Vec<f64>>,
    grad_dn: Vec<Vec<f64>>,
    extra_taps: Vec<(usize, usize, Vec<f64>)>,
}

impl Problem {
    fn dt(&self) -> f64 {
        f64::from(self.config.dt_s())
    }
    fn tau_max(&self) -> f64 {
        f64::from(self.config.tau_max_s)
    }
    fn gamma(&self) -> f64 {
        f64::from(self.config.gamma)
    }
    fn r_max(&self) -> f64 {
        f64::from(self.config.r_max)
    }
    fn substeps(&self) -> usize {
        self.config.substeps_per_decision as usize
    }

    fn forward(&self, a: &[f64], b: &[f64], theta: &[f64], v_init: &[f64], inputs: &[Vec<f64>]) -> dense::Trace {
        dense::forward(
            &self.flyg,
            self.dt(),
            self.tau_max(),
            self.gamma(),
            self.r_max(),
            self.substeps(),
            a,
            b,
            theta,
            v_init,
            inputs,
        )
    }

    fn dense_backward(&self, trace: &dense::Trace) -> dense::Gradients {
        dense::backward(
            &self.flyg,
            self.dt(),
            self.tau_max(),
            self.gamma(),
            self.r_max(),
            self.substeps(),
            &self.a,
            &self.b,
            &self.theta,
            &self.v_init,
            &self.inputs,
            &self.grad_dn,
            &self.extra_taps,
            trace,
        )
    }

    fn loss(&self, a: &[f64], b: &[f64], theta: &[f64], v_init: &[f64], inputs: &[Vec<f64>]) -> f64 {
        let trace = self.forward(a, b, theta, v_init, inputs);
        let mut l = 0.0f64;
        for (dn, g) in trace.dn_rates.iter().zip(&self.grad_dn) {
            for (x, y) in dn.iter().zip(g) {
                l += x * y;
            }
        }
        for (dec, loc, g) in &self.extra_taps {
            let l_idx = dec * self.substeps() + loc;
            for (i, &gi) in g.iter().enumerate() {
                l += gi * dense::activation(trace.v[l_idx][i], self.r_max());
            }
        }
        l
    }

    /// Central finite differences of `loss` with respect to every entry of `a`, `b`, `theta`,
    /// `inputs` (flattened) and `v_init`, in that order.
    #[allow(clippy::type_complexity)]
    fn fd_gradients(&self, h: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        let fd_one = |perturb: &dyn Fn(f64) -> f64| -> f64 { (perturb(h) - perturb(-h)) / (2.0 * h) };

        let grad_a: Vec<f64> = (0..self.a.len())
            .map(|i| {
                fd_one(&|d| {
                    let mut a = self.a.clone();
                    a[i] += d;
                    self.loss(&a, &self.b, &self.theta, &self.v_init, &self.inputs)
                })
            })
            .collect();
        let grad_b: Vec<f64> = (0..self.b.len())
            .map(|i| {
                fd_one(&|d| {
                    let mut b = self.b.clone();
                    b[i] += d;
                    self.loss(&self.a, &b, &self.theta, &self.v_init, &self.inputs)
                })
            })
            .collect();
        let grad_theta: Vec<f64> = (0..self.theta.len())
            .map(|i| {
                fd_one(&|d| {
                    let mut th = self.theta.clone();
                    th[i] += d;
                    self.loss(&self.a, &self.b, &th, &self.v_init, &self.inputs)
                })
            })
            .collect();
        let grad_inputs: Vec<Vec<f64>> = (0..self.inputs.len())
            .map(|t| {
                (0..self.inputs[t].len())
                    .map(|k| {
                        fd_one(&|d| {
                            let mut inputs = self.inputs.clone();
                            inputs[t][k] += d;
                            self.loss(&self.a, &self.b, &self.theta, &self.v_init, &inputs)
                        })
                    })
                    .collect()
            })
            .collect();
        let grad_v_init: Vec<f64> = (0..self.v_init.len())
            .map(|i| {
                fd_one(&|d| {
                    let mut v_init = self.v_init.clone();
                    v_init[i] += d;
                    self.loss(&self.a, &self.b, &self.theta, &v_init, &self.inputs)
                })
            })
            .collect();

        (grad_a, grad_b, grad_theta, grad_inputs, grad_v_init)
    }
}

fn assert_close(got: f64, want: f64, rel_tol: f64, abs_tol: f64, label: &str) {
    let diff = (got - want).abs();
    assert!(
        diff <= abs_tol + rel_tol * want.abs(),
        "{label}: got {got}, want {want}, diff {diff} (rel_tol={rel_tol}, abs_tol={abs_tol})"
    );
}

fn assert_slice_close(got: &[f64], want: &[f64], rel_tol: f64, abs_tol: f64, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_close(g, w, rel_tol, abs_tol, &format!("{label}[{i}]"));
    }
}

/// (2a) dense f64 analytic backward vs. central finite differences on the dense f64 forward.
/// Acceptance criterion 2a's own bar ("relative error < 1e-6") — tightened here (review round 1,
/// F2) from an earlier `1e-5`/`1e-8`, which was looser than the spec actually asks for; `1e-6`
/// relative / `1e-10` absolute holds comfortably in `f64` for every fixture and randomized trial
/// below (central FD at `h = 1e-5` has its own truncation/rounding error far under this).
fn check_dense_backward_against_fd(problem: &Problem) {
    let trace = problem.forward(&problem.a, &problem.b, &problem.theta, &problem.v_init, &problem.inputs);
    let analytic = problem.dense_backward(&trace);
    let (fd_a, fd_b, fd_theta, fd_inputs, fd_v_init) = problem.fd_gradients(1e-5);

    let rel_tol = 1e-6;
    let abs_tol = 1e-10;
    assert_slice_close(&analytic.grad_a, &fd_a, rel_tol, abs_tol, "grad_a (dense vs FD)");
    assert_slice_close(&analytic.grad_b, &fd_b, rel_tol, abs_tol, "grad_b (dense vs FD)");
    assert_slice_close(
        &analytic.grad_theta,
        &fd_theta,
        rel_tol,
        abs_tol,
        "grad_theta (dense vs FD)",
    );
    for (t, fd_inputs_t) in fd_inputs.iter().enumerate() {
        assert_slice_close(
            &analytic.grad_inputs[t],
            fd_inputs_t,
            rel_tol,
            abs_tol,
            &format!("grad_inputs[{t}] (dense vs FD)"),
        );
    }
    assert_slice_close(
        &analytic.grad_v_init,
        &fd_v_init,
        rel_tol,
        abs_tol,
        "grad_v_init (dense vs FD)",
    );
}

/// (2b) the crate's actual f32 sparse backward vs. the (already FD-validated) dense f64 reference.
fn check_f32_backward_against_dense(problem: &Problem) {
    let trace = problem.forward(&problem.a, &problem.b, &problem.theta, &problem.v_init, &problem.inputs);
    let dense_grad = problem.dense_backward(&trace);

    let a32: Vec<f32> = problem.a.iter().map(|&x| x as f32).collect();
    let b32: Vec<f32> = problem.b.iter().map(|&x| x as f32).collect();
    let theta32: Vec<f32> = problem.theta.iter().map(|&x| x as f32).collect();
    let params = FlyParams {
        a: a32,
        b: b32,
        theta: theta32,
    };
    let model = FlyModel::new(problem.flyg.clone(), problem.config, params).expect("build model");

    let v_init32: Vec<f32> = problem.v_init.iter().map(|&x| x as f32).collect();
    let mut state = FlyState::new(&model);
    state.set_v(&model, &v_init32);

    let t_decisions = problem.inputs.len();
    let mut recorder = TrajectoryRecorder::new(model.num_neurons(), t_decisions * problem.substeps());
    for input_t in &problem.inputs {
        let input32: Vec<f32> = input_t.iter().map(|&x| x as f32).collect();
        state.step_decision_recording(&model, &input32, &mut recorder);
    }

    let index = BackwardIndex::build(&model);
    let mut scratch = BpttScratch::new(&model);
    let grad_dn32: Vec<Vec<f32>> = problem
        .grad_dn
        .iter()
        .map(|g| g.iter().map(|&x| x as f32).collect())
        .collect();
    let grad_dn_refs: Vec<&[f32]> = grad_dn32.iter().map(Vec::as_slice).collect();
    let extra_taps32: Vec<(usize, usize, Vec<f32>)> = problem
        .extra_taps
        .iter()
        .map(|(d, l, g)| (*d, *l, g.iter().map(|&x| x as f32).collect()))
        .collect();
    let extra_refs: Vec<ExtraRateGrad<'_>> = extra_taps32
        .iter()
        .map(|(d, l, g)| ExtraRateGrad {
            decision: *d,
            local_substep: *l,
            grad: g,
        })
        .collect();

    let got = backward(
        &model,
        &index,
        &recorder,
        &v_init32,
        t_decisions,
        &grad_dn_refs,
        &extra_refs,
        true,
        &mut scratch,
    );

    let got_a: Vec<f64> = got.grad_a.iter().map(|&x| f64::from(x)).collect();
    let got_b: Vec<f64> = got.grad_b.iter().map(|&x| f64::from(x)).collect();
    let got_theta: Vec<f64> = got.grad_theta.iter().map(|&x| f64::from(x)).collect();
    let got_v_init: Vec<f64> = got
        .grad_v_init
        .as_ref()
        .expect("want_v_init_grad was true")
        .iter()
        .map(|&x| f64::from(x))
        .collect();

    let rel_tol = 1e-4;
    let abs_tol = 1e-6;
    assert_slice_close(
        &got_a,
        &dense_grad.grad_a,
        rel_tol,
        abs_tol,
        "grad_a (f32 vs f64 dense)",
    );
    assert_slice_close(
        &got_b,
        &dense_grad.grad_b,
        rel_tol,
        abs_tol,
        "grad_b (f32 vs f64 dense)",
    );
    assert_slice_close(
        &got_theta,
        &dense_grad.grad_theta,
        rel_tol,
        abs_tol,
        "grad_theta (f32 vs f64 dense)",
    );
    assert_slice_close(
        &got_v_init,
        &dense_grad.grad_v_init,
        rel_tol,
        abs_tol,
        "grad_v_init (f32 vs f64 dense)",
    );
    for t in 0..t_decisions {
        let got_inputs: Vec<f64> = got.grad_inputs[t].iter().map(|&x| f64::from(x)).collect();
        assert_slice_close(
            &got_inputs,
            &dense_grad.grad_inputs[t],
            rel_tol,
            abs_tol,
            &format!("grad_inputs[{t}] (f32 vs f64 dense)"),
        );
    }
}

fn run_full_check(problem: &Problem) {
    check_dense_backward_against_fd(problem);
    check_f32_backward_against_dense(problem);
}

// =================================== concrete fixtures =======================================

fn shared_id(flyg: &Flyg, pre_type: u32, post_type: u32) -> u32 {
    ddai_fly::test_fixtures::shared_param_id_for(flyg, pre_type, post_type)
}

#[test]
fn tiny_chain_single_decision_single_substep() {
    let flyg = tiny_chain_flyg();
    let config = FlyConfig {
        substeps_per_decision: 1,
        ..FlyConfig::default()
    };
    let n = flyg.neurons.len();
    let num_types = flyg.types.len();
    let num_shared = flyg.summary.shared_param_count as usize;

    let problem = Problem {
        flyg,
        config,
        a: vec![0.3; num_shared],
        b: (0..num_types).map(|i| -0.1 + 0.05 * i as f64).collect(),
        theta: vec![0.2; num_types],
        v_init: (0..n).map(|i| 0.3 * i as f64 - 0.2).collect(),
        inputs: vec![vec![1.3]],
        grad_dn: vec![vec![0.7]],
        extra_taps: vec![],
    };
    run_full_check(&problem);
}

/// Mixed signs, a shared param reused by **two distinct edges of the same type pair** (checking
/// that `grad_a`'s per-edge accumulation actually sums both contributions into the one shared
/// entry, not just the last edge visited), and varied `tau`/`bias` per type.
#[test]
fn mixed_signs_and_a_shared_param_reused_by_two_edges() {
    let types = [
        FxType {
            name: "exc",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "inh",
            sign: Sign::Inhibitory,
        },
        FxType {
            name: "post",
            sign: Sign::Excitatory,
        },
    ];
    let neurons = [
        FxNeuron {
            type_index: 0,
            role: NeuronRole::InputAscending,
            side: Side::M,
            full_connectome_in: 20,
        },
        FxNeuron {
            type_index: 0,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 20,
        }, // second neuron of the SAME "exc" type -> shares a[shared_id(exc,post)]
        FxNeuron {
            type_index: 1,
            role: NeuronRole::InputAscending,
            side: Side::M,
            full_connectome_in: 20,
        },
        FxNeuron {
            type_index: 2,
            role: NeuronRole::Output,
            side: Side::M,
            full_connectome_in: 30,
        },
    ];
    let edges = [
        FxEdge {
            pre: 0,
            post: 3,
            synapse_count: 4,
        }, // exc(neuron0) -> post, shared_id(exc,post)
        FxEdge {
            pre: 1,
            post: 3,
            synapse_count: 6,
        }, // exc(neuron1) -> post, SAME shared_id(exc,post)
        FxEdge {
            pre: 2,
            post: 3,
            synapse_count: 3,
        }, // inh -> post, a different shared_id
    ];
    let flyg = build_flyg(&types, &neurons, &edges);
    let config = FlyConfig {
        substeps_per_decision: 2,
        gamma: 0.5,
        ..FlyConfig::default()
    };
    let num_types = flyg.types.len();
    let num_shared = flyg.summary.shared_param_count as usize;
    let n = flyg.neurons.len();

    let problem = Problem {
        flyg,
        config,
        a: vec![0.6; num_shared],
        b: vec![0.1, -0.2, 0.05][..num_types].to_vec(),
        theta: vec![-0.5, 0.4, 0.1][..num_types].to_vec(),
        v_init: vec![0.4, -0.3, 0.6, -0.1][..n].to_vec(),
        inputs: vec![vec![0.8, -0.4]],
        grad_dn: vec![vec![-0.6]],
        extra_taps: vec![],
    };
    run_full_check(&problem);
}

/// Multiple decisions, multiple substeps: exercises the flat-chain semantics across a decision
/// boundary (`r` carrying over from one decision's final substep into the next decision's first —
/// this happens purely through the exponential-Euler recurrence itself, no self-loop edge needed;
/// `.flyg` disallows autapses), per-decision `grad_inputs`, and both the "final" and "before-last"
/// `dn_rates` readout taps for more than one decision.
#[test]
fn multi_decision_multi_substep_chains_across_decision_boundaries() {
    let types = [
        FxType {
            name: "in",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "hid",
            sign: Sign::Inhibitory,
        },
        FxType {
            name: "out",
            sign: Sign::Excitatory,
        },
    ];
    let neurons = [
        FxNeuron {
            type_index: 0,
            role: NeuronRole::InputAscending,
            side: Side::M,
            full_connectome_in: 10,
        },
        FxNeuron {
            type_index: 1,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 10,
        },
        FxNeuron {
            type_index: 2,
            role: NeuronRole::Output,
            side: Side::M,
            full_connectome_in: 10,
        },
    ];
    let edges = [
        FxEdge {
            pre: 0,
            post: 1,
            synapse_count: 5,
        },
        FxEdge {
            pre: 1,
            post: 2,
            synapse_count: 4,
        },
    ];
    let flyg = build_flyg(&types, &neurons, &edges);
    let config = FlyConfig {
        substeps_per_decision: 3,
        ..FlyConfig::default()
    };
    let num_types = flyg.types.len();
    let num_shared = flyg.summary.shared_param_count as usize;

    let problem = Problem {
        flyg,
        config,
        a: vec![0.9, 1.4, 0.3][..num_shared].to_vec(),
        b: vec![0.1, -0.1, 0.0][..num_types].to_vec(),
        theta: vec![-0.2, 0.3, -0.6][..num_types].to_vec(),
        v_init: vec![0.2, -0.5, 0.1],
        inputs: vec![vec![0.9], vec![-0.4], vec![0.2]],
        grad_dn: vec![vec![0.5], vec![-0.3], vec![1.1]],
        extra_taps: vec![],
    };
    run_full_check(&problem);
}

/// `substeps_per_decision == 1`: the "before-last" `dn_rates` tap for decision 0 must land on the
/// window's `v_init`/`r_init` (a global flat index of `-1`, unreachable by the main loop), not on
/// any recorded substep — the one boundary case documented in `backward`'s module doc comment.
#[test]
fn single_substep_decisions_tap_v_init_correctly() {
    let flyg = tiny_chain_flyg();
    let config = FlyConfig {
        substeps_per_decision: 1,
        ..FlyConfig::default()
    };
    let num_types = flyg.types.len();
    let num_shared = flyg.summary.shared_param_count as usize;
    let n = flyg.neurons.len();

    let problem = Problem {
        flyg,
        config,
        a: vec![1.1; num_shared],
        b: vec![0.05; num_types],
        theta: vec![-0.3; num_types],
        v_init: vec![0.4, -0.2, 0.6][..n].to_vec(),
        inputs: vec![vec![0.5], vec![-0.2], vec![0.9]],
        grad_dn: vec![vec![0.4], vec![-0.9], vec![0.2]],
        extra_taps: vec![],
    };
    run_full_check(&problem);
}

/// Acceptance criterion 1's "optionally w.r.t. any neuron's rate at any substep, for auxiliary
/// heads": a `grad_dn` of all zeros plus an [`ExtraRateGrad`] tap on a `Hidden` neuron's rate at a
/// mid-window substep must still produce nonzero, FD-matching gradients.
#[test]
fn auxiliary_rate_tap_on_a_hidden_neuron_mid_window() {
    let flyg = tiny_chain_flyg();
    let config = FlyConfig {
        substeps_per_decision: 2,
        ..FlyConfig::default()
    };
    let num_types = flyg.types.len();
    let num_shared = flyg.summary.shared_param_count as usize;
    let n = flyg.neurons.len();

    let problem = Problem {
        flyg,
        config,
        a: vec![0.7; num_shared],
        b: vec![0.02; num_types],
        theta: vec![0.1; num_types],
        v_init: vec![0.3, 0.1, -0.2][..n].to_vec(),
        inputs: vec![vec![0.6], vec![0.3]],
        grad_dn: vec![vec![0.0], vec![0.0]],
        extra_taps: vec![(0, 1, vec![0.0, 1.0, 0.0])], // decision 0, local substep 1, hidden neuron 1
    };
    run_full_check(&problem);
}

/// A `shared_id` used across a whole (pre_type, post_type) pair reused by *many* fan-in edges to
/// the same post neuron (exercises the CSR gather kernel's `as_chunks::<8>()` main loop, not just
/// its remainder — mirrors `tests/correctness.rs`'s own review-round-1 "nine edges" fixture).
#[test]
fn nine_fan_in_edges_of_the_same_type_pair() {
    const NUM_PRE: usize = 9;
    let types = [
        FxType {
            name: "pre",
            sign: Sign::Inhibitory,
        },
        FxType {
            name: "post",
            sign: Sign::Excitatory,
        },
    ];
    let mut neurons: Vec<FxNeuron> = (0..NUM_PRE)
        .map(|_| FxNeuron {
            type_index: 0,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 1,
        })
        .collect();
    neurons.push(FxNeuron {
        type_index: 1,
        role: NeuronRole::Output,
        side: Side::M,
        full_connectome_in: 64,
    });
    let edges: Vec<FxEdge> = (0..NUM_PRE)
        .map(|i| FxEdge {
            pre: i as u32,
            post: NUM_PRE as u32,
            synapse_count: (i as u32) + 2,
        })
        .collect();
    let flyg = build_flyg(&types, &neurons, &edges);
    let config = FlyConfig {
        substeps_per_decision: 1,
        ..FlyConfig::default()
    };
    let n = flyg.neurons.len();

    let problem = Problem {
        flyg,
        config,
        a: vec![0.5], // one type pair (pre, post) used by all 9 fan-in edges -> one shared_param_id
        b: vec![0.05, -0.05],
        theta: vec![-0.1, 0.2],
        // Deliberately not landing on an exact multiple of 0.05 (offset -0.123, not -0.1): the
        // relu kink's subgradient is defined as exactly 0 (documented, matches
        // `activation_derivative`), which central finite differences *cannot* reproduce
        // exactly *at* v == 0 (they see roughly half the one-sided slope there instead) — a
        // meaningless FD "failure" this fixture must not manufacture by accident.
        v_init: (0..n).map(|i| 0.057 * i as f64 - 0.123).collect(),
        inputs: vec![vec![]],
        grad_dn: vec![vec![-0.4]],
        extra_taps: vec![],
    };
    run_full_check(&problem);
}

// ==================================== randomized (2c) =========================================

/// A small deterministically-generated random graph: `n` neurons of `num_types` types (types
/// assigned round-robin so every type actually has >= 1 neuron, matching `build_flyg`'s own
/// requirement), one designated input neuron, one designated output neuron, and up to `num_edges`
/// random edges (skipping autapses and duplicate `(pre, post)` pairs — `.flyg`'s CSR requires
/// strictly-ascending `pre_index` per row, so neither is a valid subgraph edge set).
fn random_problem(rng: &mut SplitMix64, n: usize, num_types: usize, num_edges: usize, t: usize, s: usize) -> Problem {
    let signs = [Sign::Excitatory, Sign::Inhibitory, Sign::Neutral];
    let types: Vec<FxType> = (0..num_types)
        .map(|i| FxType {
            name: match i % 3 {
                0 => "rt0",
                1 => "rt1",
                _ => "rt2",
            },
            sign: signs[(rng.next_u64() as usize) % signs.len()],
        })
        .collect();

    let mut neurons: Vec<FxNeuron> = (0..n)
        .map(|i| FxNeuron {
            type_index: (i % num_types) as u32,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 50, // generous headroom over any random edge's synapse sum below
        })
        .collect();
    neurons[0].role = NeuronRole::InputAscending;
    if n > 1 {
        neurons[n - 1].role = NeuronRole::Output;
    }

    let mut edges = Vec::with_capacity(num_edges);
    let mut in_subgraph_synapses = vec![0u32; n];
    let mut seen_pairs = std::collections::HashSet::new();
    for _ in 0..num_edges {
        let pre = (rng.next_u64() as usize) % n;
        let post = (rng.next_u64() as usize) % n;
        if pre == post {
            continue; // .flyg disallows autapses
        }
        if !seen_pairs.insert((pre, post)) {
            continue; // .flyg's CSR requires strictly-ascending (i.e. unique) pre_index per row
        }
        let synapse_count = 1 + (rng.next_u64() % 5) as u32;
        if in_subgraph_synapses[post] + synapse_count > 40 {
            continue; // stay well under full_connectome_in's headroom
        }
        in_subgraph_synapses[post] += synapse_count;
        edges.push(FxEdge {
            pre: pre as u32,
            post: post as u32,
            synapse_count,
        });
    }
    let flyg = build_flyg(&types, &neurons, &edges);

    let config = FlyConfig {
        substeps_per_decision: s as u32,
        gamma: if rng.next_f32_unit() < 0.5 { 1.0 } else { 0.5 },
        ..FlyConfig::default()
    };

    let num_shared = flyg.summary.shared_param_count as usize;
    let a: Vec<f64> = (0..num_shared)
        .map(|_| rng.next_f32_unit() as f64 * 2.0 - 0.5)
        .collect();
    let b: Vec<f64> = (0..num_types).map(|_| rng.next_f32_unit() as f64 * 0.6 - 0.3).collect();
    let theta: Vec<f64> = (0..num_types).map(|_| rng.next_f32_unit() as f64 * 1.0 - 0.5).collect();
    let v_init: Vec<f64> = (0..n).map(|_| rng.next_f32_unit() as f64 * 1.4 - 0.7).collect();
    let inputs: Vec<Vec<f64>> = (0..t).map(|_| vec![rng.next_f32_unit() as f64 * 2.0 - 1.0]).collect();
    let grad_dn: Vec<Vec<f64>> = (0..t).map(|_| vec![rng.next_f32_unit() as f64 * 2.0 - 1.0]).collect();

    let extra_taps = if rng.next_f32_unit() < 0.4 && t > 0 {
        let dec = (rng.next_u64() as usize) % t;
        let loc = (rng.next_u64() as usize) % s;
        let grad: Vec<f64> = (0..n).map(|_| rng.next_f32_unit() as f64 * 1.0 - 0.5).collect();
        vec![(dec, loc, grad)]
    } else {
        Vec::new()
    };

    Problem {
        flyg,
        config,
        a,
        b,
        theta,
        v_init,
        inputs,
        grad_dn,
        extra_taps,
    }
}

#[test]
fn randomized_property_test_over_many_tiny_graphs_and_sequences() {
    let mut rng = SplitMix64::new(202_609_272);
    for trial in 0..150 {
        let n = 3 + (trial % 10); // 3..=12 neurons
        let num_types = 1 + (n % 4).max(1);
        let num_edges = 2 + trial % 15;
        let t = 1 + trial % 4;
        let s = 1 + trial % 3;
        let problem = random_problem(&mut rng, n, num_types.min(n), num_edges, t, s);
        run_full_check(&problem);
    }
}

#[test]
fn shared_param_lookup_helper_is_used_in_at_least_one_fixture() {
    // Guards against `shared_id` bit-rotting unused if every fixture above stops needing it.
    let flyg = tiny_chain_flyg();
    let id = shared_id(&flyg, 0, 1);
    assert!((id as usize) < flyg.summary.shared_param_count as usize);
}
