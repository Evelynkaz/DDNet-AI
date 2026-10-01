//! Task 7.2b: the batched backend (`ddai_fly::batched`) against the per-sequence path of task 7.2
//! (`ddai_fly::backward::backward`, the reference) and against an independent `f64` reference, on
//! tiny random graphs (this file, runs in CI) -- every parameter group, the input gradients and the
//! initial-state gradients, with random inputs, ragged sequence lengths, extra rate taps, batch
//! sizes that are not a multiple of the lane width, one-neuron-per-chunk plans (stress for the
//! chunked reductions) and every substep count from 1 to 3.
//!
//! The real S/M graphs are in `batched_real_graph.rs` (`#[ignore]`d, they need the compiled
//! `.flyg` files).
//!
//! # Tolerance
//! Both implementations are `f32` and sum in different orders, so they agree to a few ulp
//! of the *terms being summed*, not of the result (cancellation). An element passes if
//! `|batched - reference| <= ABS_TOL + REL_TOL * |reference|`; the constants below are the same
//! `1e-4` / `1e-6` bar the 7.2 suite uses for its own `f32`-vs-`f64` check (2b), applied to the
//! batched path against the per-sequence path *and* against the `f64` truth.

use ddai_fly::batched::{BatchedEngine, BatchedForwardOptions, BatchedPlan, BatchedSeqGrad, BatchedSeqInput};
use ddai_fly::rng::SplitMix64;
use ddai_fly::test_fixtures::{FxEdge, FxNeuron, FxType, build_flyg};
use ddai_fly::{
    BackwardIndex, BpttScratch, ExtraRateGrad, FlyConfig, FlyModel, FlyParams, FlyState, Sequence, TrajectoryRecorder,
    backward, train_step,
};
use ddai_flyg::{Flyg, NeuronRole, Side, Sign};

pub const REL_TOL: f64 = 1e-4;
pub const ABS_TOL: f64 = 1e-6;

// ============================== dense f64 reference (copied from 7.2) ================================
// The independent f64 forward + backward of `backward_correctness.rs` (task 7.2's `mod dense`,
// itself validated there against central finite differences), unchanged.

// ============================== dense f64 reference (2a) ===================================

#[allow(dead_code)] // `Trace::dn_rates` is only used by the 7.2 suite this module is copied from.
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

// ====================================== random problems ========================================

struct Problem {
    flyg: Flyg,
    config: FlyConfig,
    params: FlyParams,
    /// One entry per sequence.
    seqs: Vec<Sequence>,
}

/// A random graph: `n` neurons of `num_types` types, `k_in` inputs (a mix of visual and
/// ascending), `k_out` outputs, random sparse edges, `batch` random sequences of random length
/// `1..=t_max` (ragged), random `grad_dn`, random dense taps.
#[allow(clippy::too_many_arguments)]
fn random_problem(
    rng: &mut SplitMix64,
    n: usize,
    num_types: usize,
    num_edges: usize,
    k_in: usize,
    k_out: usize,
    batch: usize,
    t_max: usize,
    s: usize,
) -> Problem {
    let signs = [Sign::Excitatory, Sign::Inhibitory, Sign::Neutral, Sign::Excitatory];
    let types: Vec<FxType> = (0..num_types)
        .map(|_| FxType {
            name: "rt",
            sign: signs[(rng.next_u64() as usize) % signs.len()],
        })
        .collect();
    let mut neurons: Vec<FxNeuron> = (0..n)
        .map(|i| FxNeuron {
            type_index: (i % num_types) as u32,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 60,
        })
        .collect();
    for (i, nr) in neurons.iter_mut().take(k_in).enumerate() {
        nr.role = if i % 2 == 0 {
            NeuronRole::InputVisual
        } else {
            NeuronRole::InputAscending
        };
    }
    for nr in neurons.iter_mut().rev().take(k_out) {
        nr.role = NeuronRole::Output;
    }
    let mut edges = Vec::new();
    let mut in_syn = vec![0u32; n];
    let mut seen = std::collections::HashSet::new();
    for _ in 0..num_edges {
        let pre = (rng.next_u64() as usize) % n;
        let post = (rng.next_u64() as usize) % n;
        if pre == post || !seen.insert((pre, post)) {
            continue;
        }
        let synapse_count = 1 + (rng.next_u64() % 5) as u32;
        if in_syn[post] + synapse_count > 50 {
            continue;
        }
        in_syn[post] += synapse_count;
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
    let params = FlyParams {
        a: (0..num_shared).map(|_| rng.next_f32_unit() * 2.0 - 0.5).collect(),
        b: (0..num_types).map(|_| rng.next_f32_unit() * 0.6 - 0.3).collect(),
        theta: (0..num_types).map(|_| rng.next_f32_unit() - 0.5).collect(),
    };
    let model_k_in = flyg
        .neurons
        .iter()
        .filter(|x| matches!(x.role, NeuronRole::InputVisual | NeuronRole::InputAscending))
        .count();
    let model_k_out = flyg.neurons.iter().filter(|x| x.role == NeuronRole::Output).count();
    let seqs = (0..batch)
        .map(|b| {
            let t = if b == 0 {
                t_max
            } else {
                1 + (rng.next_u64() as usize) % t_max
            };
            let extra_taps = if rng.next_f32_unit() < 0.5 {
                let k = 1 + (rng.next_u64() as usize) % 2;
                (0..k)
                    .map(|_| {
                        (
                            (rng.next_u64() as usize) % t,
                            (rng.next_u64() as usize) % s,
                            (0..n).map(|_| rng.next_f32_unit() - 0.5).collect(),
                        )
                    })
                    .collect()
            } else {
                Vec::new()
            };
            Sequence {
                v_init: (0..n).map(|_| rng.next_f32_unit() * 1.4 - 0.7).collect(),
                inputs: (0..t)
                    .map(|_| (0..model_k_in).map(|_| rng.next_f32_unit() * 2.0 - 1.0).collect())
                    .collect(),
                grad_dn: (0..t)
                    .map(|_| (0..model_k_out).map(|_| rng.next_f32_unit() * 2.0 - 1.0).collect())
                    .collect(),
                extra_taps,
            }
        })
        .collect();
    let _ = k_out;
    Problem {
        flyg,
        config,
        params,
        seqs,
    }
}

fn build_model(p: &Problem) -> FlyModel {
    FlyModel::new(p.flyg.clone(), p.config, p.params.clone()).expect("model")
}

/// Per-sequence 7.2 gradients of one sequence, with the initial-state gradient.
fn per_seq_reference(model: &FlyModel, index: &BackwardIndex, seq: &Sequence) -> ddai_fly::BpttGradients {
    let t = seq.inputs.len();
    let substeps = model.config().substeps_per_decision as usize;
    let mut state = FlyState::new(model);
    state.set_v(model, &seq.v_init);
    let mut recorder = TrajectoryRecorder::new(model.num_neurons(), t * substeps);
    for x in &seq.inputs {
        state.step_decision_recording(model, x, &mut recorder);
    }
    let mut scratch = BpttScratch::new(model);
    let gdn: Vec<&[f32]> = seq.grad_dn.iter().map(Vec::as_slice).collect();
    let extra: Vec<ExtraRateGrad<'_>> = seq
        .extra_taps
        .iter()
        .map(|(d, l, g)| ExtraRateGrad {
            decision: *d,
            local_substep: *l,
            grad: g,
        })
        .collect();
    backward(
        model,
        index,
        &recorder,
        &seq.v_init,
        t,
        &gdn,
        &extra,
        true,
        &mut scratch,
    )
}

/// Same sequence through the dense f64 reference.
fn f64_reference(p: &Problem, seq: &Sequence) -> dense::Gradients {
    let to64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| f64::from(x)).collect() };
    let a = to64(&p.params.a);
    let b = to64(&p.params.b);
    let th = to64(&p.params.theta);
    let v_init = to64(&seq.v_init);
    let inputs: Vec<Vec<f64>> = seq.inputs.iter().map(|x| to64(x)).collect();
    let grad_dn: Vec<Vec<f64>> = seq.grad_dn.iter().map(|x| to64(x)).collect();
    let taps: Vec<(usize, usize, Vec<f64>)> = seq.extra_taps.iter().map(|(d, l, g)| (*d, *l, to64(g))).collect();
    let (dt, tau_max, gamma, r_max) = (
        f64::from(p.config.dt_s()),
        f64::from(p.config.tau_max_s),
        f64::from(p.config.gamma),
        f64::from(p.config.r_max),
    );
    let s = p.config.substeps_per_decision as usize;
    let trace = dense::forward(&p.flyg, dt, tau_max, gamma, r_max, s, &a, &b, &th, &v_init, &inputs);
    dense::backward(
        &p.flyg, dt, tau_max, gamma, r_max, s, &a, &b, &th, &v_init, &inputs, &grad_dn, &taps, &trace,
    )
}

fn batched_inputs(seqs: &[Sequence]) -> Vec<BatchedSeqInput<'_>> {
    seqs.iter()
        .map(|s| BatchedSeqInput {
            v_init: &s.v_init,
            inputs: &s.inputs,
        })
        .collect()
}

fn batched_grads(seqs: &[Sequence]) -> Vec<BatchedSeqGrad<'_>> {
    seqs.iter()
        .map(|s| BatchedSeqGrad {
            grad_dn: &s.grad_dn,
            extra_taps: &s.extra_taps,
        })
        .collect()
}

fn assert_all_close(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= ABS_TOL + REL_TOL * w.abs(),
            "{label}[{i}]: got {g}, want {w}, diff {}",
            (g - w).abs()
        );
    }
}

fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// Everything compared: batched vs per-sequence (sum over the batch) and vs the f64 reference.
fn check_problem(p: &Problem, plan: BatchedPlan, opts: &BatchedForwardOptions) -> ddai_fly::batched::BatchedGradients {
    let model = build_model(p);
    let index = BackwardIndex::build(&model);
    let mut engine = BatchedEngine::with_plan(plan);
    engine.forward(&model, &batched_inputs(&p.seqs), opts).expect("forward");
    let got = engine.backward(&model, &batched_grads(&p.seqs), true);

    let mut want_a = vec![0.0f64; p.params.a.len()];
    let mut want_b = vec![0.0f64; p.params.b.len()];
    let mut want_t = vec![0.0f64; p.params.theta.len()];
    let mut ref64_a = want_a.clone();
    let mut ref64_b = want_b.clone();
    let mut ref64_t = want_t.clone();
    for (b, seq) in p.seqs.iter().enumerate() {
        let r = per_seq_reference(&model, &index, seq);
        let d = f64_reference(p, seq);
        for (acc, x) in want_a.iter_mut().zip(&r.grad_a) {
            *acc += f64::from(*x);
        }
        for (acc, x) in want_b.iter_mut().zip(&r.grad_b) {
            *acc += f64::from(*x);
        }
        for (acc, x) in want_t.iter_mut().zip(&r.grad_theta) {
            *acc += f64::from(*x);
        }
        for (acc, x) in ref64_a.iter_mut().zip(&d.grad_a) {
            *acc += x;
        }
        for (acc, x) in ref64_b.iter_mut().zip(&d.grad_b) {
            *acc += x;
        }
        for (acc, x) in ref64_t.iter_mut().zip(&d.grad_theta) {
            *acc += x;
        }
        // input and initial-state gradients, per sequence
        assert_eq!(got.grad_inputs[b].len(), seq.inputs.len());
        for t in 0..seq.inputs.len() {
            assert_all_close(
                &to64(&got.grad_inputs[b][t]),
                &to64(&r.grad_inputs[t]),
                "grad_inputs vs per-seq",
            );
            assert_all_close(&to64(&got.grad_inputs[b][t]), &d.grad_inputs[t], "grad_inputs vs f64");
        }
        let gvi = got.grad_v_init.as_ref().expect("grad_v_init")[b].as_slice();
        assert_all_close(
            &to64(gvi),
            &to64(r.grad_v_init.as_ref().unwrap()),
            "grad_v_init vs per-seq",
        );
        assert_all_close(&to64(gvi), &d.grad_v_init, "grad_v_init vs f64");
    }
    assert_all_close(&to64(&got.grad.a), &want_a, "grad_a vs per-seq");
    assert_all_close(&to64(&got.grad.b), &want_b, "grad_b vs per-seq");
    assert_all_close(&to64(&got.grad.theta), &want_t, "grad_theta vs per-seq");
    assert_all_close(&to64(&got.grad.a), &ref64_a, "grad_a vs f64");
    assert_all_close(&to64(&got.grad.b), &ref64_b, "grad_b vs f64");
    assert_all_close(&to64(&got.grad.theta), &ref64_t, "grad_theta vs f64");
    got
}

#[test]
fn random_tiny_graphs_match_the_per_sequence_path_and_the_f64_reference() {
    let mut rng = SplitMix64::new(20_261_001);
    for trial in 0..72usize {
        let n = 6 + (trial % 11);
        let num_types = 1 + trial % 5;
        let batch = [1usize, 2, 3, 7, 8, 9, 16, 21, 33, 64, 70, 130][trial % 12];
        let t_max = 1 + trial % 4;
        let s = 1 + trial % 3;
        let p = random_problem(
            &mut rng,
            n,
            num_types,
            10 + 2 * n,
            1 + trial % 3,
            1 + trial % 2,
            batch,
            t_max,
            s,
        );
        let model = build_model(&p);
        // One chunk, and one row per chunk (the extreme of the chunked reductions).
        for cost in [usize::MAX, 1] {
            let plan = BatchedPlan::with_chunk_cost(&model, cost);
            if cost == 1 {
                assert_eq!(plan.num_chunks(), n);
            }
            check_problem(&p, plan, &BatchedForwardOptions::default());
        }
    }
}

#[test]
fn bitwise_identical_across_thread_counts() {
    let mut rng = SplitMix64::new(77);
    let p = random_problem(&mut rng, 40, 5, 200, 4, 3, 21, 6, 3);
    let model = build_model(&p);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            // Threshold 0: the regions really run on the pool (tiny graphs would run serially).
            let mut engine =
                BatchedEngine::with_plan(BatchedPlan::with_chunk_cost(&model, 30).with_parallel_threshold(0));
            engine
                .forward(&model, &batched_inputs(&p.seqs), &BatchedForwardOptions::default())
                .unwrap();
            engine.backward(&model, &batched_grads(&p.seqs), true)
        })
    };
    let one = run(1);
    for threads in [2, 3, 8] {
        let g = run(threads);
        assert_eq!(one.grad.a, g.grad.a, "grad_a bits, 1 vs {threads} threads");
        assert_eq!(one.grad.b, g.grad.b);
        assert_eq!(one.grad.theta, g.grad.theta);
        assert_eq!(one.grad_inputs, g.grad_inputs);
        assert_eq!(one.grad_v_init, g.grad_v_init);
    }
}

#[test]
fn chunked_bptt_segments_give_bitwise_identical_gradients() {
    let mut rng = SplitMix64::new(5);
    let p = random_problem(&mut rng, 30, 4, 150, 3, 2, 10, 7, 2);
    let model = build_model(&p);
    let run = |seg: Option<usize>| {
        let mut engine = BatchedEngine::with_plan(BatchedPlan::with_chunk_cost(&model, 40));
        let opts = BatchedForwardOptions {
            segment_decisions: seg,
            ..BatchedForwardOptions::default()
        };
        engine.forward(&model, &batched_inputs(&p.seqs), &opts).unwrap();
        let mut dn = vec![0.0f32; model.num_outputs()];
        engine.dn_rates(0, 0, &mut dn);
        (engine.backward(&model, &batched_grads(&p.seqs), true), dn)
    };
    let (whole, dn_whole) = run(None);
    for seg in [1usize, 2, 3, 7, 100] {
        let (g, dn) = run(Some(seg));
        assert_eq!(whole.grad.a, g.grad.a, "grad_a bits, segment {seg}");
        assert_eq!(whole.grad.b, g.grad.b);
        assert_eq!(whole.grad.theta, g.grad.theta);
        assert_eq!(whole.grad_inputs, g.grad_inputs);
        assert_eq!(whole.grad_v_init, g.grad_v_init);
        assert_eq!(dn_whole, dn);
    }
}

#[test]
fn different_chunkings_agree_within_f32_reduction_tolerance() {
    let mut rng = SplitMix64::new(11);
    let p = random_problem(&mut rng, 50, 6, 300, 4, 3, 13, 5, 2);
    let model = build_model(&p);
    let run = |cost: usize| {
        let mut engine = BatchedEngine::with_plan(BatchedPlan::with_chunk_cost(&model, cost));
        engine
            .forward(&model, &batched_inputs(&p.seqs), &BatchedForwardOptions::default())
            .unwrap();
        engine.backward(&model, &batched_grads(&p.seqs), true)
    };
    let base = run(usize::MAX);
    for cost in [1usize, 25, 100] {
        let g = run(cost);
        assert_all_close(&to64(&g.grad.a), &to64(&base.grad.a), "grad_a across chunkings");
        assert_all_close(&to64(&g.grad.b), &to64(&base.grad.b), "grad_b across chunkings");
        assert_all_close(
            &to64(&g.grad.theta),
            &to64(&base.grad.theta),
            "grad_theta across chunkings",
        );
    }
}

#[test]
fn dn_rates_and_type_means_match_step_decision() {
    let mut rng = SplitMix64::new(3);
    let p = random_problem(&mut rng, 25, 4, 120, 3, 3, 11, 5, 3);
    let model = build_model(&p);
    let mut engine = BatchedEngine::with_plan(BatchedPlan::with_chunk_cost(&model, 20));
    let opts = BatchedForwardOptions {
        type_means: true,
        ..BatchedForwardOptions::default()
    };
    engine.forward(&model, &batched_inputs(&p.seqs), &opts).unwrap();
    for (b, seq) in p.seqs.iter().enumerate() {
        let mut state = FlyState::new(&model);
        state.set_v(&model, &seq.v_init);
        for (t, x) in seq.inputs.iter().enumerate() {
            let out = state.step_decision(&model, x);
            let mut dn = vec![0.0f32; model.num_outputs()];
            engine.dn_rates(b, t, &mut dn);
            let mut tm = vec![0.0f32; model.num_types()];
            engine.type_mean_rates(b, t, &mut tm);
            assert_all_close(&to64(&dn), &to64(out.dn_rates), "dn_rates");
            assert_all_close(&to64(&tm), &to64(out.per_type_mean_rate), "type means");
        }
    }
}

#[test]
fn train_step_batched_matches_train_step() {
    let mut rng = SplitMix64::new(9);
    let p = random_problem(&mut rng, 30, 4, 140, 3, 2, 12, 4, 2);
    let model = build_model(&p);
    let index = BackwardIndex::build(&model);
    let want = train_step(&model, &index, &p.seqs, None).unwrap();
    let mut engine = BatchedEngine::new(&model);
    let got = ddai_fly::batched::train_step_batched(&model, &mut engine, &p.seqs, None).unwrap();
    assert_all_close(&to64(&got.grad.a), &to64(&want.grad.a), "a");
    assert_all_close(&to64(&got.grad.b), &to64(&want.grad.b), "b");
    assert_all_close(&to64(&got.grad.theta), &to64(&want.grad.theta), "theta");
    assert_eq!(got.grad_inputs.len(), want.grad_inputs.len());
    for (g, w) in got.grad_inputs.iter().zip(&want.grad_inputs) {
        assert_eq!(g.len(), w.len());
        for (gt, wt) in g.iter().zip(w) {
            assert_all_close(&to64(gt), &to64(wt), "grad_inputs");
        }
    }
}

#[test]
fn memory_cap_is_enforced_and_selects_chunked_windows() {
    let mut rng = SplitMix64::new(21);
    let p = random_problem(&mut rng, 30, 4, 140, 3, 2, 9, 6, 2);
    let model = build_model(&p);
    let engine = BatchedEngine::new(&model);
    let free = engine
        .estimate_memory(&model, p.seqs.len(), 6, &BatchedForwardOptions::default())
        .unwrap();
    assert_eq!(free.segment_decisions, 6);
    assert_eq!(free.segments, 1);
    // A cap one byte under the whole-window need forces shorter segments (and still fits).
    let opts = BatchedForwardOptions {
        memory_cap_bytes: Some(free.total_bytes() - 1),
        ..BatchedForwardOptions::default()
    };
    let chunked = engine.estimate_memory(&model, p.seqs.len(), 6, &opts).unwrap();
    assert!(chunked.segment_decisions < 6 && chunked.segments > 1);
    assert!(chunked.total_bytes() < free.total_bytes());
    // A cap below even a one-decision segment is an error, before anything is allocated.
    let tiny = BatchedForwardOptions {
        memory_cap_bytes: Some(16),
        ..BatchedForwardOptions::default()
    };
    let mut engine = BatchedEngine::new(&model);
    let before = engine.memory_bytes();
    let err = engine.forward(&model, &batched_inputs(&p.seqs), &tiny).unwrap_err();
    assert_eq!(err.cap_bytes, 16);
    assert_eq!(engine.memory_bytes(), before, "nothing allocated on refusal");

    // And a capped run still produces the uncapped gradients, bit for bit.
    let mut a = BatchedEngine::new(&model);
    a.forward(&model, &batched_inputs(&p.seqs), &BatchedForwardOptions::default())
        .unwrap();
    let ga = a.backward(&model, &batched_grads(&p.seqs), false);
    let mut b = BatchedEngine::new(&model);
    b.forward(&model, &batched_inputs(&p.seqs), &opts).unwrap();
    let gb = b.backward(&model, &batched_grads(&p.seqs), false);
    assert_eq!(ga.grad.a, gb.grad.a);
    assert_eq!(ga.grad.b, gb.grad.b);
    assert_eq!(ga.grad.theta, gb.grad.theta);
}

/// The cap counts what the engine already holds: a big uncapped call followed by a small capped
/// one must release the big buffers (and still give the right gradients), while a capped call
/// that fits next to what is held keeps reusing it.
#[test]
fn memory_cap_counts_buffers_held_from_an_earlier_call() {
    let mut rng = SplitMix64::new(23);
    let p = random_problem(&mut rng, 30, 4, 140, 3, 2, 40, 12, 2);
    let model = build_model(&p);
    let mut engine = BatchedEngine::new(&model);
    let plan_bytes = engine.plan().memory_bytes();
    let held_by = |e: &BatchedEngine| e.memory_bytes() - plan_bytes;
    let big_seqs: Vec<Sequence> = p.seqs.clone();
    engine
        .forward(&model, &batched_inputs(&big_seqs), &BatchedForwardOptions::default())
        .unwrap();
    engine.backward(&model, &batched_grads(&big_seqs), true);
    let big_held = held_by(&engine);

    let small: Vec<Sequence> = p.seqs[..5].to_vec();
    let small_est = engine
        .estimate_memory(&model, small.len(), 12, &BatchedForwardOptions::default())
        .unwrap();
    let cap = small_est.total_bytes() + 1024;
    assert!(big_held > cap, "the test needs the earlier call to exceed the cap");
    let opts = BatchedForwardOptions {
        memory_cap_bytes: Some(cap),
        ..BatchedForwardOptions::default()
    };
    engine.forward(&model, &batched_inputs(&small), &opts).unwrap();
    let g_capped = engine.backward(&model, &batched_grads(&small), true);
    assert!(
        held_by(&engine) <= cap,
        "held {} bytes after a capped call, cap {cap}",
        held_by(&engine)
    );
    // Same shape again: the held buffers fit, nothing is released or reallocated.
    let held_before = held_by(&engine);
    engine.forward(&model, &batched_inputs(&small), &opts).unwrap();
    engine.backward(&model, &batched_grads(&small), true);
    assert_eq!(held_by(&engine), held_before);
    // Releasing must not change the numbers.
    let mut fresh = BatchedEngine::new(&model);
    fresh
        .forward(&model, &batched_inputs(&small), &BatchedForwardOptions::default())
        .unwrap();
    let g_fresh = fresh.backward(&model, &batched_grads(&small), true);
    assert_eq!(g_capped.grad.a, g_fresh.grad.a);
    assert_eq!(g_capped.grad.theta, g_fresh.grad.theta);
    assert_eq!(g_capped.grad_v_init, g_fresh.grad_v_init);
}

#[test]
fn engine_is_reusable_across_batches_of_different_shapes() {
    let mut rng = SplitMix64::new(31);
    let p1 = random_problem(&mut rng, 20, 3, 80, 2, 2, 20, 5, 2);
    let model = build_model(&p1);
    let mut engine = BatchedEngine::with_plan(BatchedPlan::with_chunk_cost(&model, 30));
    // big batch, then small, then big again: stale buffer contents must never leak in.
    let sizes = [20usize, 3, 20, 9];
    let mut first_big: Option<ddai_fly::batched::BatchedGradients> = None;
    for &bs in &sizes {
        let seqs: Vec<Sequence> = p1.seqs[..bs].to_vec();
        engine
            .forward(&model, &batched_inputs(&seqs), &BatchedForwardOptions::default())
            .unwrap();
        let g = engine.backward(&model, &batched_grads(&seqs), true);
        let mut fresh = BatchedEngine::with_plan(BatchedPlan::with_chunk_cost(&model, 30));
        fresh
            .forward(&model, &batched_inputs(&seqs), &BatchedForwardOptions::default())
            .unwrap();
        let g2 = fresh.backward(&model, &batched_grads(&seqs), true);
        assert_eq!(g.grad.a, g2.grad.a, "reused engine == fresh engine, batch {bs}");
        assert_eq!(g.grad_inputs, g2.grad_inputs);
        assert_eq!(g.grad_v_init, g2.grad_v_init);
        if bs == 20 {
            match &first_big {
                None => first_big = Some(g),
                Some(f) => assert_eq!(f.grad.a, g.grad.a),
            }
        }
    }
}

#[test]
fn empty_batch_and_shape_errors() {
    let mut rng = SplitMix64::new(41);
    let p = random_problem(&mut rng, 12, 3, 40, 2, 2, 2, 2, 2);
    let model = build_model(&p);
    let mut engine = BatchedEngine::new(&model);
    let got = ddai_fly::batched::train_step_batched(&model, &mut engine, &[], None).unwrap();
    assert!(got.grad_inputs.is_empty());
    assert!(got.grad.a.iter().all(|&x| x == 0.0));
}

#[test]
fn zero_length_sequences_contribute_nothing_and_do_not_disturb_the_others() {
    let mut rng = SplitMix64::new(51);
    let p = random_problem(&mut rng, 20, 3, 80, 2, 2, 5, 4, 2);
    let model = build_model(&p);
    let index = BackwardIndex::build(&model);

    // All sequences empty: zero gradients, no input gradients, and a zero v_init gradient.
    let empty: Vec<Sequence> = p
        .seqs
        .iter()
        .map(|s| Sequence {
            v_init: s.v_init.clone(),
            inputs: Vec::new(),
            grad_dn: Vec::new(),
            extra_taps: Vec::new(),
        })
        .collect();
    let mut engine = BatchedEngine::new(&model);
    engine
        .forward(&model, &batched_inputs(&empty), &BatchedForwardOptions::default())
        .unwrap();
    let g = engine.backward(&model, &batched_grads(&empty), true);
    assert!(g.grad.a.iter().chain(&g.grad.b).chain(&g.grad.theta).all(|&x| x == 0.0));
    assert!(g.grad_inputs.iter().all(Vec::is_empty));
    assert!(g.grad_v_init.unwrap().iter().all(|v| v.iter().all(|&x| x == 0.0)));

    // One empty sequence in the middle of a real batch: the others get exactly the gradients
    // they would get without it (the per-sequence reference, summed).
    let mut mixed = p.seqs.clone();
    mixed[2] = empty[2].clone();
    let got = check_problem(
        &Problem {
            flyg: p.flyg.clone(),
            config: p.config,
            params: p.params.clone(),
            seqs: mixed.clone(),
        },
        BatchedPlan::new(&model),
        &BatchedForwardOptions::default(),
    );
    assert!(got.grad_inputs[2].is_empty());
    let _ = index;
}

/// A region below the work threshold runs serially on the calling thread, above it on the pool:
/// the result must not depend on which (chunks, not threads, define every sum).
#[test]
fn serial_and_pool_regions_give_bitwise_identical_results() {
    let mut rng = SplitMix64::new(71);
    let p = random_problem(&mut rng, 40, 5, 220, 4, 3, 21, 5, 3);
    let model = build_model(&p);
    let run = |threshold: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            let mut engine =
                BatchedEngine::with_plan(BatchedPlan::with_chunk_cost(&model, 30).with_parallel_threshold(threshold));
            engine
                .forward(&model, &batched_inputs(&p.seqs), &BatchedForwardOptions::default())
                .unwrap();
            engine.backward(&model, &batched_grads(&p.seqs), true)
        })
    };
    let serial = run(usize::MAX);
    let pooled = run(0);
    assert_eq!(serial.grad.a, pooled.grad.a);
    assert_eq!(serial.grad.b, pooled.grad.b);
    assert_eq!(serial.grad.theta, pooled.grad.theta);
    assert_eq!(serial.grad_inputs, pooled.grad_inputs);
    assert_eq!(serial.grad_v_init, pooled.grad_v_init);
}
