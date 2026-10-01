//! Task 7.2b on the real S and M graphs (`#[ignore]`d: needs the compiled `.flyg` files).
//!
//! * `batched_gradients_match_the_per_sequence_path_and_f64_truth` -- the equivalence table of the
//!   README: random initial states, random inputs, random `dL/d(dn_rates)`, extra rate taps; every
//!   parameter group (`a`, `b`, `theta`, summed over the batch), every sequence's input gradients and
//!   initial-state gradient. The batched `f32` result is compared with the per-sequence `f32` result
//!   (task 7.2, the reference) **and** with an independent `f64` sparse BPTT written in this file, so
//!   the error of the batched path is shown next to the error the reference itself has.
//! * `batched_results_are_bitwise_identical_across_thread_counts_and_segmentations` -- 1 vs 8
//!   threads, and the whole window vs chunked (recomputed) BPTT windows.
//!
//! ```text
//! cargo test -p ddai-fly --release --test batched_real_graph -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use ddai_fly::batched::{BatchedEngine, BatchedForwardOptions, BatchedGradients, BatchedSeqGrad, BatchedSeqInput};
use ddai_fly::rng::SplitMix64;
use ddai_fly::{
    BackwardIndex, BpttScratch, ExtraRateGrad, FlyConfig, FlyModel, FlyParams, FlyState, Sequence, TrajectoryRecorder,
    backward,
};

fn compiled(name: &str) -> Option<PathBuf> {
    let p =
        PathBuf::from(std::env::var("HOME").ok()?).join(format!("aiddnet/data/connectome/compiled/fly-{name}-v1.flyg"));
    p.exists().then_some(p)
}

fn build_model(name: &str, seed: u64) -> Option<FlyModel> {
    let flyg = ddai_flyg::load(&compiled(name)?).unwrap();
    let config = FlyConfig::default();
    let mut params = FlyParams::init_default(&flyg, &config, seed);
    // Spread the parameters so the gradients are not all alike (a, b and theta per type/pair).
    let mut rng = SplitMix64::new(seed ^ 0xA5A5);
    for a in &mut params.a {
        *a += (rng.next_f32_unit() - 0.5) * 0.8;
    }
    for b in &mut params.b {
        *b += (rng.next_f32_unit() - 0.5) * 0.2;
    }
    for t in &mut params.theta {
        *t += (rng.next_f32_unit() - 0.5) * 1.0;
    }
    Some(FlyModel::new(flyg, config, params).unwrap())
}

fn random_sequences(model: &FlyModel, rng: &mut SplitMix64, batch: usize, t: usize) -> Vec<Sequence> {
    let (n, k_in, k_out, s) = (
        model.num_neurons(),
        model.num_inputs(),
        model.num_outputs(),
        model.config().substeps_per_decision as usize,
    );
    (0..batch)
        .map(|b| {
            let len = if b % 5 == 4 { t - t / 3 } else { t }; // a few shorter sequences
            let taps = (0..1 + b % 3)
                .map(|_| {
                    let mut g = vec![0.0f32; n];
                    for _ in 0..n / 20 {
                        g[(rng.next_u64() as usize) % n] = (rng.next_f32_unit() - 0.5) * 0.1;
                    }
                    ((rng.next_u64() as usize) % len, (rng.next_u64() as usize) % s, g)
                })
                .collect();
            Sequence {
                v_init: (0..n).map(|_| rng.next_f32_unit() * 0.8 - 0.1).collect(),
                inputs: (0..len)
                    .map(|_| (0..k_in).map(|_| rng.next_f32_unit() * 0.8 - 0.2).collect())
                    .collect(),
                grad_dn: (0..len)
                    .map(|_| (0..k_out).map(|_| rng.next_f32_unit() * 2.0 - 1.0).collect())
                    .collect(),
                extra_taps: taps,
            }
        })
        .collect()
}

// ------------------------------------------- f64 truth ---------------------------------------------

struct Grad64 {
    a: Vec<f64>,
    b: Vec<f64>,
    theta: Vec<f64>,
    inputs: Vec<Vec<f64>>,
    v_init: Vec<f64>,
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Sparse `f64` BPTT of one sequence, from the model equations of FLY.md section 4 alone (no code
/// of this crate besides the graph accessors): the forward pass stores `V`/`V_inf`, the backward
/// pass walks the substeps in reverse with the same conventions as the per-sequence path (relu
/// kink subgradient 0, `tau` clamp subgradient 0, `dn_rates` = mean of the last two `r`).
#[allow(clippy::needless_range_loop)] // `post`/`i` index the CSR and several per-neuron arrays alike
fn truth64(model: &FlyModel, seq: &Sequence) -> Grad64 {
    let flyg = model.flyg();
    let n = model.num_neurons();
    let cfg = model.config();
    let (dt, tau_max, gamma, r_max) = (
        f64::from(cfg.dt_s()),
        f64::from(cfg.tau_max_s),
        f64::from(cfg.gamma),
        f64::from(cfg.r_max),
    );
    let s = cfg.substeps_per_decision as usize;
    let t_dec = seq.inputs.len();
    let p = model.params();
    let a: Vec<f64> = p.a.iter().map(|&x| f64::from(x)).collect();
    let alpha: Vec<f64> = a.iter().map(|&x| softplus(x)).collect();
    let act = |v: f64| if v <= 0.0 { 0.0 } else { r_max * (v / r_max).tanh() };
    let dact = |v: f64| {
        if v <= 0.0 {
            0.0
        } else {
            1.0 / (v / r_max).cosh().powi(2)
        }
    };
    // Per edge: weight, shared id, sign*N/Z (the factor of dL/dalpha).
    let nnz = flyg.edges.pre_index.len();
    let mut w = vec![0.0f64; nnz];
    let mut coeff = vec![0.0f64; nnz];
    let mut sid = vec![0usize; nnz];
    for post in 0..n {
        let inv_z = 1.0 / (flyg.neuron_input_totals.full_connectome[post].max(1) as f64).powf(gamma);
        for e in flyg.edges.row_start[post] as usize..flyg.edges.row_start[post + 1] as usize {
            let tp = &flyg.type_pairs[flyg.edges.type_pair_index[e] as usize];
            let sign = f64::from(flyg.types[tp.pre_type as usize].sign.as_i8());
            coeff[e] = sign * f64::from(flyg.edges.synapse_count[e]) * inv_z;
            sid[e] = tp.shared_param_id as usize;
            w[e] = coeff[e] * alpha[sid[e]];
        }
    }
    let num_types = model.num_types();
    let bias: Vec<f64> = (0..n)
        .map(|i| f64::from(p.b[flyg.neurons[i].type_index as usize]))
        .collect();
    let tau_un: Vec<f64> = p.theta.iter().map(|&th| dt + softplus(f64::from(th))).collect();
    let decay_t: Vec<f64> = tau_un.iter().map(|&tu| 1.0 - (-dt / tu.min(tau_max)).exp()).collect();
    let ddecay_t: Vec<f64> = p
        .theta
        .iter()
        .zip(&tau_un)
        .map(|(&th, &tu)| {
            if tu >= tau_max {
                0.0
            } else {
                -(-dt / tu).exp() * dt / (tu * tu) * sigmoid(f64::from(th))
            }
        })
        .collect();
    let ins = model.input_neuron_indices();
    let outs = model.output_neuron_indices();

    // Forward.
    let l_total = t_dec * s;
    let v0: Vec<f64> = seq.v_init.iter().map(|&x| f64::from(x)).collect();
    let mut v_hist: Vec<Vec<f64>> = Vec::with_capacity(l_total);
    let mut vinf_hist: Vec<Vec<f64>> = Vec::with_capacity(l_total);
    let mut v_prev = v0.clone();
    for l in 0..l_total {
        let x = &seq.inputs[l / s];
        let r_prev: Vec<f64> = v_prev.iter().map(|&v| act(v)).collect();
        let mut vinf = bias.clone();
        for post in 0..n {
            let mut acc = 0.0;
            for e in flyg.edges.row_start[post] as usize..flyg.edges.row_start[post + 1] as usize {
                acc += w[e] * r_prev[flyg.edges.pre_index[e] as usize];
            }
            vinf[post] += acc;
        }
        for (k, &i) in ins.iter().enumerate() {
            vinf[i as usize] += f64::from(x[k]);
        }
        let v: Vec<f64> = (0..n)
            .map(|i| {
                let d = decay_t[flyg.neurons[i].type_index as usize];
                v_prev[i] + d * (vinf[i] - v_prev[i])
            })
            .collect();
        v_hist.push(v.clone());
        vinf_hist.push(vinf);
        v_prev = v;
    }

    // Backward.
    let mut grad_a = vec![0.0f64; a.len()];
    let mut grad_b = vec![0.0f64; num_types];
    let mut grad_theta = vec![0.0f64; num_types];
    let mut grad_inputs = vec![vec![0.0f64; ins.len()]; t_dec];
    let mut fut_dv = vec![0.0f64; n];
    let mut fut_dr = vec![0.0f64; n];
    let to64 = |g: &[f32]| -> Vec<f64> { g.iter().map(|&x| f64::from(x)).collect() };
    for l in (0..l_total).rev() {
        let mut dr = fut_dr.clone();
        if (l + 1) % s == 0 {
            let g = &seq.grad_dn[(l + 1) / s - 1];
            for (slot, &i) in outs.iter().enumerate() {
                dr[i as usize] += 0.5 * f64::from(g[slot]);
            }
        }
        if (l + 2) % s == 0 && (l + 2) / s - 1 < t_dec {
            let g = &seq.grad_dn[(l + 2) / s - 1];
            for (slot, &i) in outs.iter().enumerate() {
                dr[i as usize] += 0.5 * f64::from(g[slot]);
            }
        }
        for (d, loc, g) in &seq.extra_taps {
            if d * s + loc == l {
                for (x, &y) in dr.iter_mut().zip(g) {
                    *x += f64::from(y);
                }
            }
        }
        let v_prev_l: &[f64] = if l == 0 { &v0 } else { &v_hist[l - 1] };
        let r_prev_l: Vec<f64> = v_prev_l.iter().map(|&v| act(v)).collect();
        let mut dvt = vec![0.0f64; n];
        let mut delta = vec![0.0f64; n];
        for i in 0..n {
            let ty = flyg.neurons[i].type_index as usize;
            dvt[i] = dr[i] * dact(v_hist[l][i]) + fut_dv[i];
            delta[i] = decay_t[ty] * dvt[i];
            grad_b[ty] += delta[i];
            grad_theta[ty] += dvt[i] * (vinf_hist[l][i] - v_prev_l[i]) * ddecay_t[ty];
        }
        for (k, &i) in ins.iter().enumerate() {
            grad_inputs[l / s][k] += delta[i as usize];
        }
        let mut next_dr = vec![0.0f64; n];
        for post in 0..n {
            for e in flyg.edges.row_start[post] as usize..flyg.edges.row_start[post + 1] as usize {
                let pre = flyg.edges.pre_index[e] as usize;
                grad_a[sid[e]] += delta[post] * r_prev_l[pre] * coeff[e];
                next_dr[pre] += w[e] * delta[post];
            }
        }
        fut_dr = next_dr;
        for i in 0..n {
            fut_dv[i] = (1.0 - decay_t[flyg.neurons[i].type_index as usize]) * dvt[i];
        }
    }
    let _ = to64;
    for (sid_, g) in grad_a.iter_mut().enumerate() {
        *g *= sigmoid(a[sid_]);
    }
    // dL/dV_init.
    let mut dr0 = fut_dr;
    if s == 1 {
        for (slot, &i) in outs.iter().enumerate() {
            dr0[i as usize] += 0.5 * f64::from(seq.grad_dn[0][slot]);
        }
    }
    let v_init: Vec<f64> = (0..n).map(|i| dr0[i] * dact(v0[i]) + fut_dv[i]).collect();
    Grad64 {
        a: grad_a,
        b: grad_b,
        theta: grad_theta,
        inputs: grad_inputs,
        v_init,
    }
}

// -------------------------------------------- comparison -------------------------------------------

/// `(max |got - want|, max |want|)`.
fn err(got: &[f64], want: &[f64]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let d = got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
    let s = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
    (d, s)
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn per_seq_reference(model: &FlyModel, index: &BackwardIndex, seq: &Sequence) -> ddai_fly::BpttGradients {
    let t = seq.inputs.len();
    let substeps = model.config().substeps_per_decision as usize;
    let mut state = FlyState::new(model);
    state.set_v(model, &seq.v_init);
    let mut rec = TrajectoryRecorder::new(model.num_neurons(), t * substeps);
    for x in &seq.inputs {
        state.step_decision_recording(model, x, &mut rec);
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
    backward(model, index, &rec, &seq.v_init, t, &gdn, &extra, true, &mut scratch)
}

fn run_batched(
    model: &FlyModel,
    engine: &mut BatchedEngine,
    seqs: &[Sequence],
    opts: &BatchedForwardOptions,
) -> BatchedGradients {
    let ins: Vec<BatchedSeqInput<'_>> = seqs
        .iter()
        .map(|s| BatchedSeqInput {
            v_init: &s.v_init,
            inputs: &s.inputs,
        })
        .collect();
    engine.forward(model, &ins, opts).unwrap();
    let gr: Vec<BatchedSeqGrad<'_>> = seqs
        .iter()
        .map(|s| BatchedSeqGrad {
            grad_dn: &s.grad_dn,
            extra_taps: &s.extra_taps,
        })
        .collect();
    engine.backward(model, &gr, true)
}

#[test]
#[ignore]
fn batched_gradients_match_the_per_sequence_path_and_f64_truth() {
    for (name, batch, t) in [("S", 24usize, 16usize), ("M", 16, 8)] {
        let Some(model) = build_model(name, 5) else {
            eprintln!("skipped {name}: graph not found");
            continue;
        };
        let index = BackwardIndex::build(&model);
        let mut rng = SplitMix64::new(0xB47C4ED ^ batch as u64);
        let seqs = random_sequences(&model, &mut rng, batch, t);
        let mut engine = BatchedEngine::new(&model);
        let got = run_batched(&model, &mut engine, &seqs, &BatchedForwardOptions::default());

        // References, sequence by sequence.
        let mut per = (
            vec![0.0f64; model.params().a.len()],
            vec![0.0f64; model.num_types()],
            vec![0.0f64; model.num_types()],
        );
        let mut tru = per.clone();
        // [group][which comparison]: (max |diff|, scale) accumulated as maxima over sequences
        let mut e_in = [(0.0f64, 0.0f64); 3]; // batched vs per-seq, batched vs f64, per-seq vs f64
        let mut e_v0 = [(0.0f64, 0.0f64); 3];
        for (b, seq) in seqs.iter().enumerate() {
            let r = per_seq_reference(&model, &index, seq);
            let d = truth64(&model, seq);
            for (acc, x) in per.0.iter_mut().zip(&r.grad_a) {
                *acc += f64::from(*x);
            }
            for (acc, x) in per.1.iter_mut().zip(&r.grad_b) {
                *acc += f64::from(*x);
            }
            for (acc, x) in per.2.iter_mut().zip(&r.grad_theta) {
                *acc += f64::from(*x);
            }
            for (acc, x) in tru.0.iter_mut().zip(&d.a) {
                *acc += x;
            }
            for (acc, x) in tru.1.iter_mut().zip(&d.b) {
                *acc += x;
            }
            for (acc, x) in tru.2.iter_mut().zip(&d.theta) {
                *acc += x;
            }
            let upd = |slot: &mut (f64, f64), (d, s): (f64, f64)| {
                slot.0 = slot.0.max(d);
                slot.1 = slot.1.max(s);
            };
            for ti in 0..seq.inputs.len() {
                upd(
                    &mut e_in[0],
                    err(&f64s(&got.grad_inputs[b][ti]), &f64s(&r.grad_inputs[ti])),
                );
                upd(&mut e_in[1], err(&f64s(&got.grad_inputs[b][ti]), &d.inputs[ti]));
                upd(&mut e_in[2], err(&f64s(&r.grad_inputs[ti]), &d.inputs[ti]));
            }
            let gv = got.grad_v_init.as_ref().unwrap()[b].as_slice();
            let rv = r.grad_v_init.as_ref().unwrap();
            upd(&mut e_v0[0], err(&f64s(gv), &f64s(rv)));
            upd(&mut e_v0[1], err(&f64s(gv), &d.v_init));
            upd(&mut e_v0[2], err(&f64s(rv), &d.v_init));
        }
        let (ga, gb, gt) = (f64s(&got.grad.a), f64s(&got.grad.b), f64s(&got.grad.theta));
        eprintln!(
            "\n=== {name}: {} neurons, {} edges, B={batch}, T={t} (a few sequences shorter), S={}, random inputs, extra taps ===",
            model.num_neurons(),
            model.flyg().edges.num_edges(),
            model.config().substeps_per_decision
        );
        eprintln!(
            "{:<10} {:>10} | {:>22} | {:>22} | {:>22}",
            "group", "scale", "batched vs per-seq", "batched vs f64", "per-seq vs f64"
        );
        eprintln!(
            "{:<10} {:>10} | {:>10} {:>11} | {:>10} {:>11} | {:>10} {:>11}",
            "", "max|g|", "max|d|", "rel.scale", "max|d|", "rel.scale", "max|d|", "rel.scale"
        );
        let mut rows: Vec<(&str, [(f64, f64); 3])> = vec![
            ("a", [err(&ga, &per.0), err(&ga, &tru.0), err(&per.0, &tru.0)]),
            ("b", [err(&gb, &per.1), err(&gb, &tru.1), err(&per.1, &tru.1)]),
            ("theta", [err(&gt, &per.2), err(&gt, &tru.2), err(&per.2, &tru.2)]),
        ];
        rows.push(("inputs", e_in));
        rows.push(("v_init", e_v0));
        for (g, e) in &rows {
            let scale = e[1].1.max(e[0].1).max(1e-30);
            eprintln!(
                "{g:<10} {scale:>10.3e} | {:>10.2e} {:>11.2e} | {:>10.2e} {:>11.2e} | {:>10.2e} {:>11.2e}",
                e[0].0,
                e[0].0 / scale,
                e[1].0,
                e[1].0 / scale,
                e[2].0,
                e[2].0 / scale
            );
            // The documented bound: batched vs per-sequence within 1e-4 of the group's scale, and
            // no worse against the f64 truth than twice the reference's own error (plus a floor).
            assert!(
                e[0].0 / scale < 1e-4,
                "{name} {g}: batched vs per-seq {:.3e}",
                e[0].0 / scale
            );
            assert!(
                e[1].0 <= 2.0 * e[2].0 + 1e-6 * scale,
                "{name} {g}: batched error vs f64 {:.3e} exceeds twice the per-seq error {:.3e}",
                e[1].0,
                e[2].0
            );
        }
    }
}

#[test]
#[ignore]
fn batched_results_are_bitwise_identical_across_thread_counts_and_segmentations() {
    for (name, batch, t) in [("S", 21usize, 8usize), ("M", 12, 6)] {
        let Some(model) = build_model(name, 6) else {
            eprintln!("skipped {name}: graph not found");
            continue;
        };
        let mut rng = SplitMix64::new(99);
        let seqs = random_sequences(&model, &mut rng, batch, t);
        let run = |threads: usize, seg: Option<usize>| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut engine = BatchedEngine::new(&model);
                run_batched(
                    &model,
                    &mut engine,
                    &seqs,
                    &BatchedForwardOptions {
                        segment_decisions: seg,
                        ..BatchedForwardOptions::default()
                    },
                )
            })
        };
        let base = run(1, None);
        let same = |x: &BatchedGradients, label: &str| {
            assert_eq!(base.grad.a, x.grad.a, "{name} {label}: grad_a bits");
            assert_eq!(base.grad.b, x.grad.b, "{name} {label}: grad_b bits");
            assert_eq!(base.grad.theta, x.grad.theta, "{name} {label}: grad_theta bits");
            assert_eq!(base.grad_inputs, x.grad_inputs, "{name} {label}: grad_inputs bits");
            assert_eq!(base.grad_v_init, x.grad_v_init, "{name} {label}: grad_v_init bits");
            eprintln!("{name}: {label} == 1 thread, whole window (bitwise: a, b, theta, inputs, v_init)");
        };
        same(&run(8, None), "8 threads");
        same(&run(3, None), "3 threads");
        same(&run(1, Some(2)), "1 thread, 2-decision recomputed segments");
        same(&run(8, Some(1)), "8 threads, 1-decision recomputed segments");
    }
}
