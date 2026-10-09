//! Tests of the `Gm` neuron model: the f32 forward against an independent dense f64
//! implementation, the hand-written backward against central finite differences of that f64
//! forward (every parameter group, the inputs), the controls, the parameter plumbing.

use ddai_flyg::{Flyg, NeuronRole, Side, Sign};

use super::*;
use crate::config::FlyConfig;
use crate::model::FlyModel;
use crate::params::FlyParams;
use crate::rng::SplitMix64;
use crate::state::FlyState;
use crate::test_fixtures::{FxEdge, FxNeuron, FxType, build_flyg};

/// A small random graph: 3 inputs (two types), 5 hidden (three types), 3 outputs (two types, an L/R pair
/// on one), every neuron with a couple of random inputs.
fn random_graph(seed: u64) -> Flyg {
    let mut rng = SplitMix64::new(seed);
    let types = [
        FxType {
            name: "in_a",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "in_b",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "hid_e",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "hid_i",
            sign: Sign::Inhibitory,
        },
        FxType {
            name: "hid_n",
            sign: Sign::Neutral,
        },
        FxType {
            name: "out_a",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "out_b",
            sign: Sign::Inhibitory,
        },
    ];
    let roles_types: [(NeuronRole, u32, Side); 11] = [
        (NeuronRole::InputVisual, 0, Side::L),
        (NeuronRole::InputVisual, 0, Side::R),
        (NeuronRole::InputAscending, 1, Side::M),
        (NeuronRole::Hidden, 2, Side::L),
        (NeuronRole::Hidden, 2, Side::R),
        (NeuronRole::Hidden, 3, Side::M),
        (NeuronRole::Hidden, 3, Side::M),
        (NeuronRole::Hidden, 4, Side::M),
        (NeuronRole::Output, 5, Side::L),
        (NeuronRole::Output, 5, Side::R),
        (NeuronRole::Output, 6, Side::M),
    ];
    let neurons: Vec<FxNeuron> = roles_types
        .iter()
        .map(|&(role, type_index, side)| FxNeuron {
            type_index,
            role,
            side,
            full_connectome_in: 60 + rng.next_u64() % 60,
        })
        .collect();
    let n = neurons.len() as u32;
    let mut edges = Vec::new();
    for post in 0..n {
        let mut pres: Vec<u32> = Vec::new();
        for _ in 0..4 {
            let pre = (rng.next_u64() % u64::from(n)) as u32;
            if pre != post && !pres.contains(&pre) {
                pres.push(pre);
            }
        }
        for pre in pres {
            edges.push(FxEdge {
                pre,
                post,
                synapse_count: 1 + (rng.next_u64() % 9) as u32,
            });
        }
    }
    build_flyg(&types, &neurons, &edges)
}

fn gm_model(flyg: Flyg, cfg: GmConfig, seed: u64) -> FlyModel {
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 1);
    FlyModel::new(flyg, config, params)
        .unwrap()
        .with_gm(cfg, None, seed)
        .unwrap()
}

// ---------------------------------------------------------------------------------------------
// A dense f64 reference of the same equations (nested loops, no CSR, no const generics).
// ---------------------------------------------------------------------------------------------

struct P64 {
    f: Vec<Vec<f64>>,
}

impl P64 {
    fn of(p: &GmParams) -> P64 {
        P64 {
            f: p.fields()
                .iter()
                .map(|v| v.iter().map(|&x| f64::from(x)).collect())
                .collect(),
        }
    }
}

fn ss(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

/// Dense forward over a window: returns `dn_rates[t][k]`.
fn dense_forward(m: &GmModel, p: &P64, v_init: &[f64], inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (n, d, hd, o) = (m.n, m.shape.d, m.shape.hd, m.shape.o);
    let gated = o != d;
    let [eta, w1m, w1e, b1, w2, b2, wh, bg, inj, ro_w, ro_b]: [&Vec<f64>; 11] = [
        &p.f[0], &p.f[1], &p.f[2], &p.f[3], &p.f[4], &p.f[5], &p.f[6], &p.f[7], &p.f[8], &p.f[9], &p.f[10],
    ];
    let mut h = v_init.to_vec();
    let mut rates = Vec::new();
    for e_t in inputs {
        for _ in 0..m.config.steps {
            for (k, &i) in m.aff.iter().enumerate() {
                let i = i as usize;
                let slot = m.aff_slot[k] as usize;
                let old: Vec<f64> = h[i * d..(i + 1) * d].to_vec();
                for dd in 0..d {
                    let mut pre = bg[dd];
                    for kk in 0..d {
                        pre += old[kk] * wh[kk * d + dd];
                    }
                    pre += e_t[k] * inj[slot * d + dd];
                    h[i * d + dd] = ss(pre);
                }
            }
            let hi = h.clone();
            let mut next = vec![0.0f64; n * d];
            for v in 0..n {
                let mut msg = vec![0.0f64; d];
                for e in m.row_start[v] as usize..m.row_start[v + 1] as usize {
                    let u = m.pre[e] as usize;
                    for dd in 0..d {
                        msg[dd] += f64::from(m.w[e]) * hi[u * d + dd];
                    }
                }
                let key = m.key_of[v] as usize;
                let mut h1 = vec![0.0f64; hd];
                for j in 0..hd {
                    let mut pre = b1[j];
                    for k in 0..d {
                        pre += eta[key * d + k] * w1e[k * hd + j] + msg[k] * w1m[k * hd + j];
                    }
                    h1[j] = pre.max(0.0);
                }
                let mut out = vec![0.0f64; o];
                for oo in 0..o {
                    let mut s = b2[oo];
                    for j in 0..hd {
                        s += h1[j] * w2[j * o + oo];
                    }
                    out[oo] = ss(s);
                }
                for dd in 0..d {
                    next[v * d + dd] = if gated {
                        let z = 0.5 + 0.5 * out[dd];
                        hi[v * d + dd] + z * (out[d + dd] - hi[v * d + dd])
                    } else {
                        out[dd]
                    };
                }
            }
            h = next;
        }
        let mut r = Vec::new();
        for (k, &i) in m.dn.iter().enumerate() {
            let slot = m.ro_slot[k] as usize;
            let mut s = ro_b[slot];
            for dd in 0..d {
                s += ro_w[slot * d + dd] * h[i as usize * d + dd];
            }
            r.push(s);
        }
        rates.push(r);
    }
    rates
}

fn rand_vec(rng: &mut SplitMix64, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| scale * rng.next_gaussian()).collect()
}

/// Runs the f32 model over a window and its backward pass; returns `(rates, grads, grad_inputs)`.
fn run_f32(
    model: &FlyModel,
    v_init: &[f32],
    inputs: &[Vec<f32>],
    grad_dn: &[Vec<f32>],
) -> (Vec<Vec<f32>>, GmBackwardOut) {
    let gm = model.gm().unwrap();
    let mut rec = GmRecorder::new(gm, inputs.len());
    let mut state = FlyState::new(model);
    state.set_v(model, v_init);
    let mut rates = Vec::new();
    for e in inputs {
        let out = state.step_decision_recording_gm(model, e, &mut rec);
        rates.push(out.dn_rates.to_vec());
    }
    let refs: Vec<&[f32]> = grad_dn.iter().map(Vec::as_slice).collect();
    let mut scratch = GmScratch::new(gm);
    let g = gm_backward(gm, &rec, inputs.len(), &refs, &mut scratch);
    (rates, g)
}

fn loss64(rates: &[Vec<f64>], grad_dn: &[Vec<f32>]) -> f64 {
    rates
        .iter()
        .zip(grad_dn)
        .map(|(r, g)| r.iter().zip(g).map(|(a, &b)| a * f64::from(b)).sum::<f64>())
        .sum()
}

fn check_gradients(cfg: GmConfig, graph_seed: u64, tied: bool) {
    let mut flyg = random_graph(graph_seed);
    if tied {
        // The two left/right pairs share a group.
        flyg.neurons[0].group_id = Some(7);
        flyg.neurons[1].group_id = Some(7);
        flyg.neurons[3].group_id = Some(9);
        flyg.neurons[4].group_id = Some(9);
        flyg.neurons[8].group_id = Some(11);
        flyg.neurons[9].group_id = Some(11);
    }
    let model = gm_model(flyg, cfg, graph_seed);
    let gm = model.gm().unwrap();
    let mut rng = SplitMix64::new(graph_seed ^ 0xABCD);
    // Random (non-trivial) parameters: every group, including the zero-initialised biases.
    let mut params = gm.params().clone();
    for f in params.fields_mut() {
        for x in f.iter_mut() {
            *x += 0.3 * rng.next_gaussian();
        }
    }
    let mut model = model;
    model.set_gm_params(params.clone()).unwrap();
    let gm = model.gm().unwrap();
    let t = 3;
    let n_aff = gm.num_afferents();
    let n_out = gm.num_outputs();
    let inputs: Vec<Vec<f32>> = (0..t).map(|_| rand_vec(&mut rng, n_aff, 1.5)).collect();
    let grad_dn: Vec<Vec<f32>> = (0..t).map(|_| rand_vec(&mut rng, n_out, 1.0)).collect();
    let v_init = rand_vec(&mut rng, gm.state_len(), 0.4);

    let (rates32, out) = run_f32(&model, &v_init, &inputs, &grad_dn);

    // 1. The f32 forward matches the dense f64 forward.
    let p64 = P64::of(&params);
    let v64: Vec<f64> = v_init.iter().map(|&x| f64::from(x)).collect();
    let in64: Vec<Vec<f64>> = inputs
        .iter()
        .map(|r| r.iter().map(|&x| f64::from(x)).collect())
        .collect();
    let rates64 = dense_forward(gm, &p64, &v64, &in64);
    for (r32, r64) in rates32.iter().zip(&rates64) {
        for (a, b) in r32.iter().zip(r64) {
            assert!(
                (f64::from(*a) - b).abs() < 2e-4 * (1.0 + b.abs()),
                "forward: f32 {a} vs f64 {b}"
            );
        }
    }

    // 2. The backward matches central finite differences of the f64 loss.
    let eps = 1e-6;
    let mut checked = 0usize;
    let mut worst = 0.0f64;
    for (gi, analytic) in out.grads.fields().into_iter().enumerate() {
        for i in 0..analytic.len() {
            let mut plus = P64 { f: p64.f.clone() };
            let mut minus = P64 { f: p64.f.clone() };
            plus.f[gi][i] += eps;
            minus.f[gi][i] -= eps;
            let lp = loss64(&dense_forward(gm, &plus, &v64, &in64), &grad_dn);
            let lm = loss64(&dense_forward(gm, &minus, &v64, &in64), &grad_dn);
            let fd = (lp - lm) / (2.0 * eps);
            let a = f64::from(analytic[i]);
            let err = (a - fd).abs();
            worst = worst.max(err / (1.0 + fd.abs()));
            assert!(
                err < 2e-3 * (1.0 + fd.abs()),
                "group {gi} index {i}: analytic {a} vs finite difference {fd}"
            );
            checked += 1;
        }
    }
    // The input currents.
    for ti in 0..t {
        for k in 0..n_aff {
            let mut plus = in64.clone();
            let mut minus = in64.clone();
            plus[ti][k] += eps;
            minus[ti][k] -= eps;
            let lp = loss64(&dense_forward(gm, &p64, &v64, &plus), &grad_dn);
            let lm = loss64(&dense_forward(gm, &p64, &v64, &minus), &grad_dn);
            let fd = (lp - lm) / (2.0 * eps);
            let a = f64::from(out.grad_inputs[ti][k]);
            assert!(
                (a - fd).abs() < 2e-3 * (1.0 + fd.abs()),
                "input [{ti}][{k}]: analytic {a} vs finite difference {fd}"
            );
            checked += 1;
        }
    }
    assert!(checked > 100, "checked only {checked} gradients");
    assert!(worst < 2e-3, "worst relative gradient error {worst}");
}

fn cfg(d: u32, hidden: u32, steps: u32, update: GmUpdate) -> GmConfig {
    GmConfig {
        d,
        hidden,
        steps,
        update,
        ..GmConfig::default()
    }
}

#[test]
fn gradients_match_f64_finite_differences_plain_one_step() {
    check_gradients(cfg(8, 8, 1, GmUpdate::Plain), 1, false);
}

#[test]
fn gradients_match_f64_finite_differences_plain_two_steps() {
    check_gradients(cfg(8, 16, 2, GmUpdate::Plain), 2, false);
}

#[test]
fn gradients_match_f64_finite_differences_gated_two_steps() {
    check_gradients(cfg(8, 16, 2, GmUpdate::Gated), 3, false);
}

#[test]
fn gradients_match_f64_finite_differences_gated_three_steps_tied_descriptors() {
    let mut c = cfg(8, 8, 3, GmUpdate::Gated);
    c.descriptors = GmDescriptors::PerNeuronTied;
    check_gradients(c, 4, true);
}

#[test]
fn gradients_match_f64_finite_differences_d16_h32() {
    check_gradients(cfg(16, 32, 2, GmUpdate::Plain), 5, false);
    check_gradients(cfg(16, 32, 1, GmUpdate::Gated), 6, false);
}

#[test]
fn gradients_match_f64_finite_differences_for_the_controls() {
    let mut c = cfg(8, 16, 2, GmUpdate::Plain);
    c.wiring = GmWiring::DegreePreserving { seed: 3 };
    c.signs = GmSigns::Shuffled { seed: 4 };
    check_gradients(c, 7, false);
    let mut c = cfg(8, 8, 2, GmUpdate::Gated);
    c.signs = GmSigns::GluExcitatory;
    check_gradients(c, 8, false);
}

#[test]
fn recording_does_not_change_the_forward() {
    let model = gm_model(random_graph(9), cfg(8, 16, 2, GmUpdate::Gated), 9);
    let gm = model.gm().unwrap();
    let mut rng = SplitMix64::new(5);
    let inputs: Vec<Vec<f32>> = (0..4).map(|_| rand_vec(&mut rng, gm.num_afferents(), 1.0)).collect();
    let mut a = FlyState::new(&model);
    let mut b = FlyState::new(&model);
    let mut rec = GmRecorder::new(gm, 4);
    for e in &inputs {
        let ra = a.step_decision(&model, e).dn_rates.to_vec();
        let rb = b.step_decision_recording_gm(&model, e, &mut rec).dn_rates.to_vec();
        assert_eq!(ra, rb);
    }
    assert_eq!(a.v(), b.v());
}

#[test]
fn state_is_n_times_d_and_outputs_follow_the_readout() {
    let model = gm_model(random_graph(10), cfg(8, 8, 1, GmUpdate::Plain), 10);
    let gm = model.gm().unwrap();
    assert_eq!(gm.state_len(), model.num_neurons() * 8);
    assert_eq!(model.state_len(), gm.state_len());
    let mut s = FlyState::new(&model);
    assert_eq!(s.v().len(), gm.state_len());
    let out = s.step_decision(&model, &vec![0.7; model.num_inputs()]);
    assert_eq!(out.dn_rates.len(), model.num_outputs());
    assert_eq!(out.per_type_mean_rate.len(), model.num_types());
    assert!(out.dn_rates.iter().all(|x| x.is_finite()));
}

#[test]
fn warm_up_reaches_a_resting_state_and_reset_restores_it() {
    let model = gm_model(random_graph(11), cfg(8, 16, 2, GmUpdate::Plain), 11);
    let mut s = FlyState::new(&model);
    let report = s.warm_up(&model);
    assert!(report.converged, "warm-up did not converge: {report:?}");
    let rest = s.v().to_vec();
    let _ = s.step_decision(&model, &vec![1.0; model.num_inputs()]);
    assert_ne!(s.v(), rest.as_slice());
    s.reset_to_rest(&model);
    assert_eq!(s.v(), rest.as_slice());
}

#[test]
fn input_drives_the_dn_readout() {
    let model = gm_model(random_graph(12), cfg(8, 16, 2, GmUpdate::Plain), 12);
    let mut quiet = FlyState::new(&model);
    let mut loud = FlyState::new(&model);
    for _ in 0..6 {
        let _ = quiet.step_decision(&model, &vec![0.0; model.num_inputs()]);
        let _ = loud.step_decision(&model, &vec![3.0; model.num_inputs()]);
    }
    let a = quiet
        .step_decision(&model, &vec![0.0; model.num_inputs()])
        .dn_rates
        .to_vec();
    let b = loud
        .step_decision(&model, &vec![3.0; model.num_inputs()])
        .dn_rates
        .to_vec();
    assert!(
        a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-3),
        "the input does not reach the DN readout: {a:?} vs {b:?}"
    );
}

#[test]
fn params_flat_round_trip_and_set_params_recomputes() {
    let model = gm_model(random_graph(13), cfg(8, 16, 2, GmUpdate::Gated), 13);
    let gm = model.gm().unwrap();
    let flat = gm.params().to_flat();
    assert_eq!(flat.len(), gm.shape().total());
    let back = GmParams::from_flat(gm.shape(), &flat).unwrap();
    assert_eq!(&back, gm.params());
    assert!(GmParams::from_flat(gm.shape(), &flat[1..]).is_err());

    // set_params to different values gives the same model as building with them.
    let mut rng = SplitMix64::new(1);
    let mut other = gm.params().clone();
    for f in other.fields_mut() {
        for x in f.iter_mut() {
            *x += 0.2 * rng.next_gaussian();
        }
    }
    let mut a = model.clone();
    a.set_gm_params(other.clone()).unwrap();
    let b = FlyModel::new(
        random_graph(13),
        FlyConfig::default(),
        FlyParams::init_default(&random_graph(13), &FlyConfig::default(), 1),
    )
    .unwrap()
    .with_gm(cfg(8, 16, 2, GmUpdate::Gated), Some(other), 0)
    .unwrap();
    let e = vec![0.9; a.num_inputs()];
    let ra = FlyState::new(&a).step_decision(&a, &e).dn_rates.to_vec();
    let rb = FlyState::new(&b).step_decision(&b, &e).dn_rates.to_vec();
    assert_eq!(ra, rb);

    let mut bad = gm.params().clone();
    bad.eta.pop();
    assert!(a.set_gm_params(bad).is_err());
    let mut nan = gm.params().clone();
    nan.b1[0] = f32::NAN;
    assert!(a.set_gm_params(nan).is_err());
}

#[test]
fn init_is_deterministic_and_seed_dependent() {
    let m1 = gm_model(random_graph(14), GmConfig::default(), 5);
    let m2 = gm_model(random_graph(14), GmConfig::default(), 5);
    let m3 = gm_model(random_graph(14), GmConfig::default(), 6);
    assert_eq!(m1.gm().unwrap().params(), m2.gm().unwrap().params());
    assert_ne!(m1.gm().unwrap().params(), m3.gm().unwrap().params());
}

#[test]
fn config_validation_rejects_unsupported_shapes() {
    for bad in [
        GmConfig {
            d: 4,
            ..GmConfig::default()
        },
        GmConfig {
            d: 24,
            ..GmConfig::default()
        },
        GmConfig {
            hidden: 12,
            ..GmConfig::default()
        },
        GmConfig {
            steps: 0,
            ..GmConfig::default()
        },
        GmConfig {
            steps: 9,
            ..GmConfig::default()
        },
        GmConfig {
            msg_gain: 0.0,
            ..GmConfig::default()
        },
        GmConfig {
            msg_gain: f32::NAN,
            ..GmConfig::default()
        },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }
    GmConfig::default().validate().unwrap();
}

// ---------------------------------------------------------------------------------------------
// The controls.
// ---------------------------------------------------------------------------------------------

fn edge_multiset(model: &GmModel) -> Vec<(u32, u32)> {
    let mut v = Vec::new();
    for post in 0..model.n {
        for e in model.row_start[post] as usize..model.row_start[post + 1] as usize {
            v.push((model.pre[e], post as u32));
        }
    }
    v.sort_unstable();
    v
}

#[test]
fn rewiring_preserves_in_and_out_degrees_and_changes_the_edges() {
    // A larger random graph so that rewiring has room to move.
    let n = 60u32;
    let mut rng = SplitMix64::new(77);
    let types = [
        FxType {
            name: "a",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "b",
            sign: Sign::Inhibitory,
        },
    ];
    let mut neurons: Vec<FxNeuron> = (0..n)
        .map(|i| FxNeuron {
            type_index: i % 2,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 200,
        })
        .collect();
    neurons[0].role = NeuronRole::InputAscending;
    neurons[1].role = NeuronRole::Output;
    let mut edges = Vec::new();
    for post in 0..n {
        let mut seen = Vec::new();
        for _ in 0..6 {
            let pre = (rng.next_u64() % u64::from(n)) as u32;
            if pre != post && !seen.contains(&pre) {
                seen.push(pre);
                edges.push(FxEdge {
                    pre,
                    post,
                    synapse_count: 1 + (rng.next_u64() % 20) as u32,
                });
            }
        }
    }
    let flyg = build_flyg(&types, &neurons, &edges);
    let plain = gm_model(flyg.clone(), GmConfig::default(), 1);
    let mut c = GmConfig {
        wiring: GmWiring::DegreePreserving { seed: 5 },
        ..GmConfig::default()
    };
    let rew = gm_model(flyg.clone(), c, 1);
    let (p, r) = (plain.gm().unwrap(), rew.gm().unwrap());

    let degrees = |m: &GmModel| {
        let mut ind = vec![0u32; m.n];
        let mut outd = vec![0u32; m.n];
        for post in 0..m.n {
            for e in m.row_start[post] as usize..m.row_start[post + 1] as usize {
                ind[post] += 1;
                outd[m.pre[e] as usize] += 1;
            }
        }
        (ind, outd)
    };
    let (in_p, out_p) = degrees(p);
    let (in_r, out_r) = degrees(r);
    assert_eq!(in_p, in_r, "in-degrees must be preserved");
    assert_eq!(out_p, out_r, "out-degrees must be preserved");
    assert_ne!(edge_multiset(p), edge_multiset(r), "the rewired graph must differ");
    // No self loops, no duplicate edges, rows sorted by source.
    let e = edge_multiset(r);
    assert!(e.windows(2).all(|w| w[0] != w[1]), "duplicate edge");
    assert!(e.iter().all(|&(a, b)| a != b), "self loop");
    for post in 0..r.n {
        let row = &r.pre[r.row_start[post] as usize..r.row_start[post + 1] as usize];
        assert!(row.windows(2).all(|w| w[0] < w[1]), "row {post} not sorted");
    }
    // The sign of an edge is the sign of its (unchanged) source type.
    for post in 0..r.n {
        for k in r.row_start[post] as usize..r.row_start[post + 1] as usize {
            let src_type = flyg.neurons[r.pre[k] as usize].type_index;
            let want = f32::from(flyg.types[src_type as usize].sign.as_i8());
            assert_eq!(
                r.w[k].signum() * f32::from(r.w[k] != 0.0),
                want,
                "edge sign must follow the source type"
            );
        }
    }
    // Same seed, same graph; another seed, another graph.
    let again = gm_model(flyg.clone(), c, 1);
    assert_eq!(again.gm().unwrap().pre, r.pre);
    c.wiring = GmWiring::DegreePreserving { seed: 6 };
    let other = gm_model(flyg, c, 1);
    assert_ne!(other.gm().unwrap().pre, r.pre);
}

#[test]
fn out_strength_per_source_is_preserved_by_rewiring() {
    // Compare the multiset of |w|*Z_post for a fixed source before/after is not meaningful (Z_post changes),
    // so check the synapse counts directly through the helper.
    let row_start = [0u32, 2, 4, 6, 8];
    let pre = [1u32, 2, 0, 3, 0, 1, 2, 0];
    let counts = [5u32, 6, 7, 8, 9, 10, 11, 12];
    let (p2, c2) = controls::rewire_degree_preserving(&row_start, &pre, &counts, 3);
    let strength = |pre: &[u32], c: &[u32]| {
        let mut s = [0u32; 4];
        for (&p, &w) in pre.iter().zip(c) {
            s[p as usize] += w;
        }
        s
    };
    assert_eq!(strength(&pre, &counts), strength(&p2, &c2), "out-strength per source");
    let mut a = counts.to_vec();
    let mut b = c2.clone();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b, "the multiset of counts is preserved");
}

#[test]
fn shuffled_signs_are_a_permutation_of_the_type_signs() {
    let flyg = random_graph(15);
    let c = GmConfig {
        signs: GmSigns::Shuffled { seed: 2 },
        ..GmConfig::default()
    };
    let shuf = gm_model(flyg.clone(), c, 1);
    let m = shuf.gm().unwrap();
    let mut expected: Vec<f32> = flyg.types.iter().map(|t| f32::from(t.sign.as_i8())).collect();
    let before = expected.clone();
    controls::shuffle_in_place(&mut expected, 2);
    // The model's edge signs are the shuffled ones.
    for post in 0..m.n {
        for e in m.row_start[post] as usize..m.row_start[post + 1] as usize {
            let t = flyg.neurons[m.pre[e] as usize].type_index as usize;
            let got = m.w[e].signum() * f32::from(m.w[e] != 0.0);
            assert_eq!(got, expected[t], "edge from type {t}");
        }
    }
    // A permutation: the same multiset of signs; and some seed moves something.
    let (mut x, mut y) = (before.clone(), expected.clone());
    x.sort_by(f32::total_cmp);
    y.sort_by(f32::total_cmp);
    assert_eq!(x, y);
    let moved = (0..8u64).any(|seed| {
        let mut z = before.clone();
        controls::shuffle_in_place(&mut z, seed);
        z != before
    });
    assert!(moved, "shuffling never moved a sign");
}

#[test]
fn glu_excitatory_flips_only_inhibitory_glutamate_types() {
    let mut flyg = random_graph(16);
    // Make type 3 ("hid_i") glutamatergic inhibitory and type 6 GABAergic inhibitory.
    flyg.types[3].nt_class_used = NtClassUsed::Glutamate;
    flyg.types[6].nt_class_used = NtClassUsed::Gaba;
    let c = GmConfig {
        signs: GmSigns::GluExcitatory,
        ..GmConfig::default()
    };
    let model = gm_model(flyg.clone(), c, 1);
    let m = model.gm().unwrap();
    for post in 0..m.n {
        for e in m.row_start[post] as usize..m.row_start[post + 1] as usize {
            let t = flyg.neurons[m.pre[e] as usize].type_index;
            match t {
                3 => assert!(m.w[e] > 0.0, "glutamate must be excitatory"),
                6 => assert!(m.w[e] < 0.0, "GABA stays inhibitory"),
                _ => {}
            }
        }
    }
}

#[test]
fn tied_keys_share_between_homologs_and_leave_ungrouped_neurons_alone() {
    let mut flyg = random_graph(17);
    flyg.neurons[3].group_id = Some(5);
    flyg.neurons[4].group_id = Some(5);
    let (keys, n_keys) = controls::tied_keys(&flyg);
    assert_eq!(keys[3], keys[4]);
    let distinct: std::collections::HashSet<u32> = keys.iter().copied().collect();
    assert_eq!(distinct.len(), n_keys);
    assert_eq!(n_keys, flyg.neurons.len() - 1);
    // A different type with the same group id does not tie.
    flyg.neurons[8].group_id = Some(5);
    let (keys, _) = controls::tied_keys(&flyg);
    assert_ne!(keys[3], keys[8]);
}

#[test]
fn cost_report_counts_macs() {
    let model = gm_model(random_graph(18), cfg(8, 16, 2, GmUpdate::Plain), 1);
    let gm = model.gm().unwrap();
    // messages E*D + neurons N*(D*HD + HD*O) + afferents.
    let want = gm.num_edges() * 8 + 11 * (8 * 16 + 16 * 8) + 3 * (64 + 8);
    assert_eq!(gm.macs_per_step(), want);
}

// ---------------------------------------------------------------------------------------------
// What a Gm fly is refused by (review F3, F5).
// ---------------------------------------------------------------------------------------------

#[test]
#[should_panic(expected = "supports only the rate neuron model")]
fn the_batched_trainer_refuses_a_gm_fly() {
    let model = gm_model(random_graph(20), GmConfig::default(), 1);
    let _ = crate::batched::BatchedPlan::new(&model);
}

#[test]
#[should_panic(expected = "supports only the rate neuron model")]
fn the_rate_backward_pass_refuses_a_gm_fly() {
    let model = gm_model(random_graph(21), GmConfig::default(), 1);
    let index = crate::backward::BackwardIndex::build(&model);
    let rec = crate::recorder::TrajectoryRecorder::new(model.num_neurons(), 0);
    let mut scratch = crate::backward::BpttScratch::new(&model);
    let v = vec![0.0; model.num_neurons()];
    let _ = crate::backward::backward(&model, &index, &rec, &v, 0, &[], &[], false, &mut scratch);
}

/// A `Gm` checkpoint loads as a template (the arena and the BC tools use it) but is refused where a fly plays: the live bot's loader and the
/// hybrid's proposer call `require_rate`.
#[test]
fn a_gm_checkpoint_is_refused_where_a_fly_plays_but_not_in_the_arena_tools() {
    use crate::bc::HookView;
    use crate::brain_fixtures::write_tiny_fly_bundle;
    use crate::bundle::{NeuronModel, load_bundle, save_bundle};
    use crate::hook_intent_tests::{brain_config, template_of};
    let dir = tempfile::tempdir().unwrap();
    let (path, flyg_path) = write_tiny_fly_bundle(dir.path(), HookView::Shared);
    let flyg = ddai_flyg::load(&flyg_path).unwrap();
    let mut b = load_bundle(&path).unwrap();
    assert!(b.neuron_model.is_rate());
    let rate = template_of(&path, &flyg_path);
    assert!(rate.require_rate("test").is_ok());
    assert!(crate::proposer::FlyProposer::from_template(&rate, brain_config(), 1).is_ok());

    let model = b
        .build_model(flyg)
        .unwrap()
        .with_gm(GmConfig::default(), None, 3)
        .unwrap();
    b.neuron_model = NeuronModel::Gm {
        config: GmConfig::default(),
        params: model.gm().unwrap().params().clone(),
    };
    let gm_path = dir.path().join("gm.bundle");
    save_bundle(&gm_path, &b).unwrap();
    let t = template_of(&gm_path, &flyg_path); // loads: es eval / hook-eval / the BC tools use it
    let e = t.require_rate("live bot").unwrap_err().0;
    assert!(e.contains("Gm neuron model") && e.contains("live bot"), "{e}");
    let e = match crate::proposer::FlyProposer::from_template(&t, brain_config(), 1) {
        Err(e) => e.0,
        Ok(_) => panic!("the hybrid proposer must refuse a Gm fly"),
    };
    assert!(e.contains("Gm neuron model"), "{e}");
    let _ = t.instantiate(crate::brain::FlyBrainConfig::default()); // the arena can play it
}
