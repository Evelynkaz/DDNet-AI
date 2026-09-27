//! Acceptance criterion 2's correctness tests, all on tiny hand-built `.flyg` graphs constructed
//! via `ddai_fly::test_fixtures` (which itself goes through `ddai_flyg::validate` — see that
//! module's doc comment):
//! - one substep equals a hand-computed value (independent f64 reference here, tolerance 1e-6
//!   relative);
//! - zero-weight graph relaxes exponentially to `f(b)` with the right time constant;
//! - sign handling (inhibitory pre lowers post);
//! - input neuron current injection;
//! - determinism (same inputs -> bit-identical outputs across runs);
//! - mirror test (an exactly L/R-symmetric graph gives mirrored outputs for mirrored inputs).

use ddai_fly::test_fixtures::{FxEdge, FxNeuron, FxType, build_flyg, shared_param_id_for};
use ddai_fly::{FlyConfig, FlyModel, FlyParams, FlyState};
use ddai_flyg::{Flyg, NeuronRole, Side, Sign};

/// Independent f64 reference of `f(V) = r_max * tanh(relu(V) / r_max)` — deliberately not calling
/// into `ddai_fly::activation` at all, so this test can't pass merely because it shares a bug with
/// the code under test.
fn activation_f64(v: f64, r_max: f64) -> f64 {
    let relu = v.max(0.0);
    r_max * (relu / r_max).tanh()
}

/// Independent f64 reference of `softplus`'s inverse (`x` such that `softplus(x) == y`):
/// `x = ln(e^y - 1)`.
fn inverse_softplus_f64(y: f64) -> f64 {
    y.exp_m1().ln()
}

fn assert_close_rel(got: f32, want: f64, rel_tol: f64, what: &str) {
    let got64 = f64::from(got);
    let rel = (got64 - want).abs() / want.abs().max(1e-12);
    assert!(
        rel < rel_tol,
        "{what}: got {got64}, want {want}, rel diff {rel} >= {rel_tol}"
    );
}

/// A 2-neuron, 2-type, 1-edge graph: `pre -[N synapses]-> post`. `pre_role` is `Hidden` unless the
/// caller overrides it (the input-injection test uses `InputAscending`).
fn two_neuron_graph(
    pre_sign: Sign,
    pre_role: NeuronRole,
    pre_full_in: u64,
    post_full_in: u64,
    synapse_count: u32,
) -> Flyg {
    let types = [
        FxType {
            name: "pre_t",
            sign: pre_sign,
        },
        FxType {
            name: "post_t",
            // The post type's own sign is irrelevant here — sign only matters for a type acting
            // as a *pre*synaptic partner.
            sign: Sign::Excitatory,
        },
    ];
    let neurons = [
        FxNeuron {
            type_index: 0,
            role: pre_role,
            side: Side::M,
            full_connectome_in: pre_full_in,
        },
        FxNeuron {
            type_index: 1,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: post_full_in,
        },
    ];
    let edges = [FxEdge {
        pre: 0,
        post: 1,
        synapse_count,
    }];
    build_flyg(&types, &neurons, &edges)
}

#[test]
fn one_substep_matches_a_hand_computed_f64_reference() {
    let flyg = two_neuron_graph(Sign::Excitatory, NeuronRole::Hidden, 100, 8, 4);
    let config = FlyConfig {
        substeps_per_decision: 1,
        ..FlyConfig::default()
    }; // exercise exactly one substep
    let dt_s = f64::from(config.dt_s());

    let alpha = 2.0f64;
    let b_pre = 0.3f64;
    let b_post = -0.2f64;
    let tau_pre_s = 0.05f64;
    let tau_post_s = 0.08f64;

    let mut params = FlyParams::init_default(&flyg, &config, 1);
    let shared_id = shared_param_id_for(&flyg, 0, 1) as usize;
    params.a[shared_id] = inverse_softplus_f64(alpha) as f32;
    params.b[0] = b_pre as f32;
    params.b[1] = b_post as f32;
    params.theta[0] = inverse_softplus_f64(tau_pre_s - dt_s) as f32;
    params.theta[1] = inverse_softplus_f64(tau_post_s - dt_s) as f32;

    let model = FlyModel::new(flyg, config, params).unwrap();
    let mut state = FlyState::new(&model);
    let v0_pre = 0.5f64;
    let v0_post = 0.1f64;
    state.set_v(&model, &[v0_pre as f32, v0_post as f32]);

    state.step_decision(&model, &[]);

    // --- f64 reference, independent of ddai_fly's own code -------------------------------------
    let r_max = 10.0f64;
    let z_post = 8_f64.powf(1.0); // gamma = 1.0 default; max(1, ..) is a no-op since 8 > 1
    let w = 1.0 * alpha * 4.0 / z_post; // sign(pre)=+1, N_ij=4
    let r0 = activation_f64(v0_pre, r_max);

    let v_inf_pre = b_pre; // no incoming edges
    let decay_pre = -(-dt_s / tau_pre_s).exp_m1();
    let v_pre_want = v0_pre + decay_pre * (v_inf_pre - v0_pre);

    let v_inf_post = b_post + w * r0;
    let decay_post = -(-dt_s / tau_post_s).exp_m1();
    let v_post_want = v0_post + decay_post * (v_inf_post - v0_post);

    assert_close_rel(state.v()[0], v_pre_want, 1e-6, "pre neuron V after one substep");
    assert_close_rel(state.v()[1], v_post_want, 1e-6, "post neuron V after one substep");
}

/// Review round 1 (F5): the single-edge hand-computed reference above never exercises the
/// gather kernel's `chunks_exact(8)` main loop (only its 1-element remainder) and always uses
/// `γ = 1.0` (the default). This builds one postsynaptic neuron with **9** incoming edges (a full
/// 8-chunk plus a 1-element remainder) from 9 distinct presynaptic neurons of the same type
/// (varying `N_ij` per edge, not `α`, so the weight variation comes from the CSR data itself, not
/// from per-edge parameters), and `γ = 0.5` (the other end of FLY.md §4's documented range from
/// this crate's default `γ = 1.0`).
#[test]
fn one_substep_matches_a_hand_computed_reference_with_nine_edges_and_gamma_half() {
    const NUM_PRE: usize = 9;
    let types = [
        FxType {
            name: "pre_t",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "post_t",
            sign: Sign::Excitatory,
        },
    ];
    let mut neurons = Vec::new();
    for _ in 0..NUM_PRE {
        neurons.push(FxNeuron {
            type_index: 0,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 1,
        });
    }
    // Sum of synapse_counts below is 2+3+..+10 = 54, so full_connectome_in must be >= 54
    // (`in_subgraph <= full_connectome` — `.flyg`'s own invariant); 64 keeps Z_post = 64^0.5 = 8.0
    // exact for a clean hand computation.
    let post_full_in = 64u64;
    neurons.push(FxNeuron {
        type_index: 1,
        role: NeuronRole::Hidden,
        side: Side::M,
        full_connectome_in: post_full_in,
    });
    let post_index = NUM_PRE as u32;
    let synapse_counts: [u32; NUM_PRE] = [2, 3, 4, 5, 6, 7, 8, 9, 10];
    let edges: Vec<FxEdge> = (0..NUM_PRE)
        .map(|i| FxEdge {
            pre: i as u32,
            post: post_index,
            synapse_count: synapse_counts[i],
        })
        .collect();
    let flyg = build_flyg(&types, &neurons, &edges);

    let config = FlyConfig {
        substeps_per_decision: 1,
        gamma: 0.5,
        ..FlyConfig::default()
    };
    let dt_s = f64::from(config.dt_s());

    let alpha = 1.5f64;
    let b_post = 0.1f64;
    let tau_post_s = 0.06f64;

    let mut params = FlyParams::init_default(&flyg, &config, 1);
    let shared_id = shared_param_id_for(&flyg, 0, 1) as usize;
    params.a[shared_id] = inverse_softplus_f64(alpha) as f32;
    params.b[1] = b_post as f32;
    params.theta[1] = inverse_softplus_f64(tau_post_s - dt_s) as f32;

    let model = FlyModel::new(flyg, config, params).unwrap();
    let mut state = FlyState::new(&model);

    let v0_pre: [f64; NUM_PRE] = std::array::from_fn(|i| 0.1 * (i as f64 + 1.0)); // 0.1, 0.2, .., 0.9
    let v0_post = 0.05f64;
    let mut v0 = vec![0.0f32; NUM_PRE + 1];
    for i in 0..NUM_PRE {
        v0[i] = v0_pre[i] as f32;
    }
    v0[NUM_PRE] = v0_post as f32;
    state.set_v(&model, &v0);

    state.step_decision(&model, &[]);

    // --- f64 reference, independent of ddai_fly's own code -------------------------------------
    let r_max = 10.0f64;
    let z_post = (post_full_in as f64).powf(0.5);
    let acc: f64 = (0..NUM_PRE)
        .map(|i| {
            let w = alpha * f64::from(synapse_counts[i]) / z_post; // sign(pre) = +1
            let r_i = activation_f64(v0_pre[i], r_max);
            w * r_i
        })
        .sum();
    let v_inf_post = b_post + acc;
    let decay_post = -(-dt_s / tau_post_s).exp_m1();
    let v_post_want = v0_post + decay_post * (v_inf_post - v0_post);

    assert_close_rel(
        state.v()[NUM_PRE],
        v_post_want,
        1e-6,
        "post neuron V after one substep (9 edges, gamma=0.5)",
    );
}

#[test]
fn zero_weight_graph_relaxes_exponentially_to_f_of_b() {
    // A single isolated Hidden neuron: no edges at all, so V_inf is just `b` every substep.
    let types = [FxType {
        name: "solo",
        sign: Sign::Excitatory,
    }];
    let neurons = [FxNeuron {
        type_index: 0,
        role: NeuronRole::Hidden,
        side: Side::M,
        full_connectome_in: 1,
    }];
    let flyg = build_flyg(&types, &neurons, &[]);

    let config = FlyConfig {
        substeps_per_decision: 1,
        ..FlyConfig::default()
    };
    let dt_s = f64::from(config.dt_s());
    let tau_s = 0.1f64; // 100ms, 10x dt

    let mut params = FlyParams::init_default(&flyg, &config, 1);
    let b = 1.5f64;
    params.b[0] = b as f32;
    params.theta[0] = inverse_softplus_f64(tau_s - dt_s) as f32;

    let model = FlyModel::new(flyg, config, params).unwrap();
    let mut state = FlyState::new(&model);
    let v0 = -2.0f64;
    state.set_v(&model, &[v0 as f32]);

    let steps = 50;
    for k in 1..=steps {
        state.step_decision(&model, &[]);
        let want_v = b + (v0 - b) * (-(k as f64) * dt_s / tau_s).exp();
        let got_v = f64::from(state.v()[0]);
        assert!(
            (got_v - want_v).abs() < 1e-3,
            "step {k}: V = {got_v}, want {want_v} (exact exponential decay to b: no edges, so no nonlinearity is involved)"
        );
    }

    let r_final = activation_f64(f64::from(state.v()[0]), 10.0);
    let r_target = activation_f64(b, 10.0);
    // After 50 steps (5 time constants), V is still exp(-5) ~= 0.7% of the way from b, i.e. ~0.023
    // away in V-space here; f's slope near b=1.5 is close to 1, so r lags b by about the same
    // amount. 0.03 comfortably covers that analytically-expected gap while still catching a
    // badly broken convergence (e.g. off by an order of magnitude, or not converging at all).
    assert!(
        (r_final - r_target).abs() < 0.03,
        "after {steps} steps (~5 time constants) r should have converged close to f(b): r={r_final}, f(b)={r_target}"
    );
}

#[test]
fn inhibitory_pre_lowers_post_relative_to_excitatory_pre() {
    let build = |sign: Sign| -> (FlyModel, FlyState) {
        let flyg = two_neuron_graph(sign, NeuronRole::Hidden, 100, 8, 4);
        let config = FlyConfig {
            substeps_per_decision: 1,
            ..FlyConfig::default()
        };
        let mut params = FlyParams::init_default(&flyg, &config, 1);
        let shared_id = shared_param_id_for(&flyg, 0, 1) as usize;
        params.a[shared_id] = inverse_softplus_f64(2.0) as f32; // alpha = 2.0
        params.b[0] = 0.5;
        params.b[1] = 0.0;
        let model = FlyModel::new(flyg, config, params).unwrap();
        let mut state = FlyState::new(&model);
        state.set_v(&model, &[0.5, 0.1]); // same starting point for both variants
        (model, state)
    };

    let (model_exc, mut state_exc) = build(Sign::Excitatory);
    let (model_inh, mut state_inh) = build(Sign::Inhibitory);

    state_exc.step_decision(&model_exc, &[]);
    state_inh.step_decision(&model_inh, &[]);

    let v_post_exc = state_exc.v()[1];
    let v_post_inh = state_inh.v()[1];
    assert!(
        v_post_inh < v_post_exc,
        "an inhibitory presynaptic type must pull the postsynaptic neuron lower than an otherwise identical \
         excitatory one: excitatory V={v_post_exc}, inhibitory V={v_post_inh}"
    );
    // A disconnected postsynaptic neuron with the same b=0.0 would settle towards V_inf=0.0; the
    // excitatory/inhibitory edge must push it to either side of that.
    assert!(
        v_post_exc > 0.1,
        "excitatory input should push V above its own bias-only trajectory (0.1 -> 0.0)"
    );
    assert!(
        v_post_inh < 0.1,
        "inhibitory input should pull V below its own bias-only trajectory (0.1 -> 0.0)"
    );
}

#[test]
fn input_neuron_current_injection_changes_its_own_v() {
    let types = [FxType {
        name: "in_t",
        sign: Sign::Excitatory,
    }];
    let neurons = [FxNeuron {
        type_index: 0,
        role: NeuronRole::InputAscending,
        side: Side::M,
        full_connectome_in: 1,
    }];
    let flyg = build_flyg(&types, &neurons, &[]);

    let config = FlyConfig {
        substeps_per_decision: 1,
        ..FlyConfig::default()
    };
    let mut params = FlyParams::init_default(&flyg, &config, 1);
    params.b[0] = 0.0;
    let model = FlyModel::new(flyg, config, params).unwrap();

    let mut state_zero = FlyState::new(&model);
    state_zero.set_v(&model, &[0.2]);
    state_zero.step_decision(&model, &[0.0]);

    let mut state_driven = FlyState::new(&model);
    state_driven.set_v(&model, &[0.2]);
    state_driven.step_decision(&model, &[2.0]);

    assert!(
        state_driven.v()[0] > state_zero.v()[0],
        "injecting a positive current into the only (input) neuron must raise its V relative to zero input: \
         driven={}, zero={}",
        state_driven.v()[0],
        state_zero.v()[0]
    );

    // Exact check too: with no edges and b=0, V_inf == the injected current exactly.
    let dt_s = f64::from(config.dt_s());
    let theta0 = model.params().theta[0];
    let tau_s = f64::from(config.dt_s()) + f64::from(ddai_fly::activation::softplus(theta0));
    let decay = -(-dt_s / tau_s).exp_m1();
    let want = 0.2 + decay * (2.0 - 0.2);
    assert!(
        (f64::from(state_driven.v()[0]) - want).abs() < 1e-5,
        "got {}, want {want}",
        state_driven.v()[0]
    );
}

#[test]
fn determinism_same_inputs_give_bit_identical_outputs_across_independent_runs() {
    let flyg = build_flyg(
        &[
            FxType {
                name: "in_t",
                sign: Sign::Excitatory,
            },
            FxType {
                name: "hid_t",
                sign: Sign::Inhibitory,
            },
            FxType {
                name: "out_t",
                sign: Sign::Excitatory,
            },
        ],
        &[
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
        ],
        &[
            FxEdge {
                pre: 0,
                post: 1,
                synapse_count: 6,
            },
            FxEdge {
                pre: 1,
                post: 2,
                synapse_count: 4,
            },
        ],
    );

    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 123);

    let build = || {
        let model = FlyModel::new(flyg.clone(), config, params.clone()).unwrap();
        let state = FlyState::new(&model);
        (model, state)
    };
    let (model_a, mut state_a) = build();
    let (model_b, mut state_b) = build();

    // A fixed, deterministic (not actually random) input sequence, identical for both runs.
    for i in 0..20u32 {
        let x = ((i as f32) * 0.37).sin().abs();
        let out_a = state_a.step_decision(&model_a, &[x]);
        let da = out_a.dn_rates.to_vec();
        let ta = out_a.per_type_mean_rate.to_vec();
        let out_b = state_b.step_decision(&model_b, &[x]);
        let db = out_b.dn_rates.to_vec();
        let tb = out_b.per_type_mean_rate.to_vec();
        assert_eq!(
            da, db,
            "dn_rates must be bit-identical across independent runs with the same inputs"
        );
        assert_eq!(
            ta, tb,
            "per_type_mean_rate must be bit-identical across independent runs with the same inputs"
        );
    }
}

#[test]
fn mirrored_inputs_give_mirrored_outputs_on_an_l_r_symmetric_graph() {
    // in{L,R} -type In-> hid{L,R} -type Hid-> out{L,R}, **with cross-hemisphere edges** (review
    // round 1, F5: the original graph had none, so the mirror test never exercised any wiring
    // that actually crosses the midline — which is exactly the kind of wiring the real graphs
    // have plenty of, e.g. AOTU019/025's fan-out). The two halves are still an isomorphism under
    // the L<->R swap: every straight edge (inL->hidL, N=5) has a same-count mirror (inR->hidR),
    // and every crossing edge (inL->hidR, N=2) has a same-count mirror the other way
    // (inR->hidL) — so relabeling L<->R throughout maps this graph onto itself.
    let types = [
        FxType {
            name: "in_t",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "hid_t",
            sign: Sign::Inhibitory,
        },
        FxType {
            name: "out_t",
            sign: Sign::Excitatory,
        },
    ];
    // Dense indices: inL=0, inR=1, hidL=2, hidR=3, outL=4, outR=5.
    let neurons = [
        FxNeuron {
            type_index: 0,
            role: NeuronRole::InputAscending,
            side: Side::L,
            full_connectome_in: 5,
        },
        FxNeuron {
            type_index: 0,
            role: NeuronRole::InputAscending,
            side: Side::R,
            full_connectome_in: 5,
        },
        FxNeuron {
            type_index: 1,
            role: NeuronRole::Hidden,
            side: Side::L,
            // Incoming: straight (N=5) + crossing (N=2) = 7, so this must be >= 7
            // (`in_subgraph <= full_connectome`); 10 leaves headroom.
            full_connectome_in: 10,
        },
        FxNeuron {
            type_index: 1,
            role: NeuronRole::Hidden,
            side: Side::R,
            full_connectome_in: 10,
        },
        FxNeuron {
            type_index: 2,
            role: NeuronRole::Output,
            side: Side::L,
            // Incoming: straight (N=3) + crossing (N=1) = 4, so this must be >= 4.
            full_connectome_in: 6,
        },
        FxNeuron {
            type_index: 2,
            role: NeuronRole::Output,
            side: Side::R,
            full_connectome_in: 6,
        },
    ];
    let edges = [
        FxEdge {
            pre: 0,
            post: 2,
            synapse_count: 5,
        }, // inL -> hidL (straight)
        FxEdge {
            pre: 1,
            post: 3,
            synapse_count: 5,
        }, // inR -> hidR (straight)
        FxEdge {
            pre: 0,
            post: 3,
            synapse_count: 2,
        }, // inL -> hidR (crossing)
        FxEdge {
            pre: 1,
            post: 2,
            synapse_count: 2,
        }, // inR -> hidL (crossing)
        FxEdge {
            pre: 2,
            post: 4,
            synapse_count: 3,
        }, // hidL -> outL (straight)
        FxEdge {
            pre: 3,
            post: 5,
            synapse_count: 3,
        }, // hidR -> outR (straight)
        FxEdge {
            pre: 2,
            post: 5,
            synapse_count: 1,
        }, // hidL -> outR (crossing)
        FxEdge {
            pre: 3,
            post: 4,
            synapse_count: 1,
        }, // hidR -> outL (crossing)
    ];
    let flyg = build_flyg(&types, &neurons, &edges);

    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 7);
    let model = FlyModel::new(flyg, config, params).unwrap();

    assert_eq!(model.input_neuron_indices(), &[0, 1], "inL, inR in that order");
    assert_eq!(model.output_neuron_indices(), &[4, 5], "outL, outR in that order");

    let mut state_a = FlyState::new(&model); // fed (x, y)
    let mut state_b = FlyState::new(&model); // fed (y, x) -- the mirror
    let report_a = state_a.warm_up(&model);
    let report_b = state_b.warm_up(&model);
    assert!(
        report_a.converged && report_b.converged,
        "this tiny graph should converge well within the default cap"
    );

    // warm_up used zero (symmetric) input, so both states must already agree left<->right
    // (tolerance rather than exact equality: with crossing edges, a row can sum more than one
    // term, and this crate makes no bit-exactness promise about *which* floating-point op order a
    // future kernel change might use — only that the result is numerically the same).
    let tol = 1e-5;
    assert!((state_a.v()[0] - state_b.v()[1]).abs() < tol);
    assert!((state_a.v()[1] - state_b.v()[0]).abs() < tol);
    assert!((state_a.v()[2] - state_b.v()[3]).abs() < tol);
    assert!((state_a.v()[3] - state_b.v()[2]).abs() < tol);

    for i in 0..8u32 {
        let x = ((i as f32) * 0.53).sin().abs();
        let y = ((i as f32) * 0.91 + 1.0).cos().abs();

        let out_a = state_a.step_decision(&model, &[x, y]);
        let a_out_l = out_a.dn_rates[0];
        let a_out_r = out_a.dn_rates[1];

        let out_b = state_b.step_decision(&model, &[y, x]);
        let b_out_l = out_b.dn_rates[0];
        let b_out_r = out_b.dn_rates[1];

        assert!(
            (a_out_l - b_out_r).abs() < tol,
            "mirrored input must give mirrored output (L<->R swapped): a_out_l={a_out_l} b_out_r={b_out_r}"
        );
        assert!(
            (a_out_r - b_out_l).abs() < tol,
            "mirrored input must give mirrored output (L<->R swapped): a_out_r={a_out_r} b_out_l={b_out_l}"
        );
    }
}
