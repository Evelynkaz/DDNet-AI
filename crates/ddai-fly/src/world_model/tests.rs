use super::*;
use crate::brain_fixtures::{FxNeuron, FxType, build_brain_flyg};
use crate::config::FlyConfig;
use crate::params::FlyParams;
use ddai_flyg::{NeuronRole, Side, Sign};

/// 2 input neurons, 3 hidden, 1 output — enough hidden neurons for a meaningful subset without
/// any edges at all (the world model only ever reads `r_full`, an arbitrary `f(V)` vector; it
/// never touches the graph's connectivity).
fn tiny_flyg() -> ddai_flyg::Flyg {
    let types = [
        FxType {
            name: "IN_T",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "HID_T",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "OUT_T",
            sign: Sign::Excitatory,
        },
    ];
    let neurons = vec![
        FxNeuron {
            type_index: 0,
            role: NeuronRole::InputAscending,
            side: Side::M,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        },
        FxNeuron {
            type_index: 1,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        },
        FxNeuron {
            type_index: 1,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        },
        FxNeuron {
            type_index: 1,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        },
        FxNeuron {
            type_index: 2,
            role: NeuronRole::Output,
            side: Side::M,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        },
    ];
    build_brain_flyg(&types, &neurons, &[], &[], &[])
}

fn model_from(flyg: ddai_flyg::Flyg) -> FlyModel {
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 1);
    FlyModel::new(flyg, config, params).unwrap()
}

// --- Config / construction -------------------------------------------------------------------

#[test]
fn default_config_validates() {
    WorldModelConfig::default().validate().unwrap();
}

#[test]
fn negative_loss_weight_is_rejected() {
    let cfg = WorldModelConfig {
        loss_weight: -0.1,
        ..WorldModelConfig::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn default_subset_is_every_hidden_neuron() {
    let model = model_from(tiny_flyg());
    let head = WorldModelHead::new(&model, &WorldModelConfig::default()).unwrap();
    assert_eq!(head.num_subset(), 3);
    for &i in head.subset() {
        assert_eq!(model.flyg().neurons[i as usize].role, NeuronRole::Hidden);
    }
}

#[test]
fn explicit_out_of_range_subset_is_rejected() {
    let model = model_from(tiny_flyg());
    let cfg = WorldModelConfig {
        neuron_subset: Some(vec![999]),
        ..WorldModelConfig::default()
    };
    assert!(WorldModelHead::new(&model, &cfg).is_err());
}

#[test]
fn non_ascending_subset_is_rejected() {
    let cfg = WorldModelConfig {
        neuron_subset: Some(vec![2, 1]),
        ..WorldModelConfig::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn duplicate_subset_entries_are_rejected() {
    let cfg = WorldModelConfig {
        neuron_subset: Some(vec![1, 1, 2]),
        ..WorldModelConfig::default()
    };
    assert!(cfg.validate().is_err());
}

// --- Forward ----------------------------------------------------------------------------------

#[test]
fn forward_matches_a_hand_computed_value() {
    let model = model_from(tiny_flyg());
    let head = WorldModelHead::new(&model, &WorldModelConfig::default()).unwrap();
    let mut params = head.init_default_params();
    params.horizons[0].w_reg[0] = 2.0; // target 0 <- subset neuron 0
    params.horizons[0].b_reg[0] = 1.0;
    let mut r_full = vec![0.0f32; model.num_neurons()];
    r_full[head.subset()[0] as usize] = 3.0;

    let preds = world_model_forward(&head, &r_full, &params);
    assert!((preds[0].regression[0] - 7.0).abs() < 1e-6, "{:?}", preds[0]); // 2*3+1=7
}

// --- Gradient checks (task spec: "tiny graphs") ------------------------------------------------

fn loss_only(
    head: &WorldModelHead,
    r_full: &[f32],
    n: usize,
    params: &WorldModelParams,
    targets: &[HorizonTargets; 3],
) -> f64 {
    f64::from(world_model_loss_and_grad(head, r_full, n, params, targets).0)
}

#[test]
fn gradients_match_finite_differences() {
    let model = model_from(tiny_flyg());
    let head = WorldModelHead::new(&model, &WorldModelConfig::default()).unwrap();
    let mut params = head.init_default_params();
    for k in 0..3 {
        for (i, w) in params.horizons[k].w_reg.iter_mut().enumerate() {
            *w = 0.1 * (i as f32 + k as f32 - 1.0);
        }
        for (i, w) in params.horizons[k].w_bin.iter_mut().enumerate() {
            *w = 0.15 * (i as f32 - k as f32);
        }
    }
    let n = model.num_neurons();
    let r_full: Vec<f32> = (0..n).map(|i| 0.3 + 0.11 * i as f32).collect();

    // Targets are the *unperturbed* prediction plus a small, deterministic offset (not an
    // arbitrary hand-picked value): this keeps every squared-error term small (well-conditioned
    // for a small-step finite difference) regardless of how the weights above happen to scale,
    // rather than risking a large total loss whose `f32` summation rounding swamps the tiny
    // per-parameter delta a finite difference is trying to measure (a large loss magnitude with
    // a small `h` is a textbook catastrophic-cancellation setup — this was caught by exactly that
    // failure mode during development, see the crate's build report).
    let unperturbed = world_model_forward(&head, &r_full, &params);
    let mut targets: [HorizonTargets; 3] = Default::default();
    targets[0] = HorizonTargets {
        regression: Some(std::array::from_fn(|o| {
            unperturbed[0].regression[o] + 0.05 * (1 - 2 * (o % 2) as i32) as f32
        })),
        binary: Some([true, false, true]),
    };
    targets[2] = HorizonTargets {
        regression: Some(std::array::from_fn(|o| {
            unperturbed[2].regression[o] - 0.05 * (1 - 2 * (o % 2) as i32) as f32
        })),
        binary: Some([false, true, false]),
    };

    let (_loss, analytic, grad_r) = world_model_loss_and_grad(&head, &r_full, n, &params, &targets);

    let h = 1e-2f32;
    // grad_r over the whole model (including the zero-outside-subset entries).
    for i in 0..n {
        let mut plus = r_full.clone();
        plus[i] += h;
        let mut minus = r_full.clone();
        minus[i] -= h;
        let fd = (loss_only(&head, &plus, n, &params, &targets) - loss_only(&head, &minus, n, &params, &targets))
            / (2.0 * f64::from(h));
        let rel = (f64::from(grad_r[i]) - fd).abs() / fd.abs().max(1e-6);
        assert!(
            rel < 1e-3 || (f64::from(grad_r[i]) - fd).abs() < 2e-4,
            "grad_r[{i}]: analytic={} fd={fd} rel={rel}",
            grad_r[i]
        );
    }

    // w_reg / w_bin for horizon 0 and 2 (the ones with a target).
    for &k in &[0usize, 2] {
        for i in 0..params.horizons[k].w_reg.len() {
            let mut plus = params.clone();
            plus.horizons[k].w_reg[i] += h;
            let mut minus = params.clone();
            minus.horizons[k].w_reg[i] -= h;
            let fd = (loss_only(&head, &r_full, n, &plus, &targets) - loss_only(&head, &r_full, n, &minus, &targets))
                / (2.0 * f64::from(h));
            let rel = (f64::from(analytic.horizons[k].w_reg[i]) - fd).abs() / fd.abs().max(1e-6);
            assert!(
                rel < 1e-3 || (f64::from(analytic.horizons[k].w_reg[i]) - fd).abs() < 2e-4,
                "horizons[{k}].w_reg[{i}]: analytic={} fd={fd}",
                analytic.horizons[k].w_reg[i]
            );
        }
        for i in 0..params.horizons[k].w_bin.len() {
            let mut plus = params.clone();
            plus.horizons[k].w_bin[i] += h;
            let mut minus = params.clone();
            minus.horizons[k].w_bin[i] -= h;
            let fd = (loss_only(&head, &r_full, n, &plus, &targets) - loss_only(&head, &r_full, n, &minus, &targets))
                / (2.0 * f64::from(h));
            let rel = (f64::from(analytic.horizons[k].w_bin[i]) - fd).abs() / fd.abs().max(1e-6);
            assert!(
                rel < 1e-3 || (f64::from(analytic.horizons[k].w_bin[i]) - fd).abs() < 2e-4,
                "horizons[{k}].w_bin[{i}]: analytic={} fd={fd}",
                analytic.horizons[k].w_bin[i]
            );
        }
    }

    // Horizon 1 had no target: its gradient must be exactly zero.
    assert!(analytic.horizons[1].w_reg.iter().all(|&x| x == 0.0));
    assert!(analytic.horizons[1].w_bin.iter().all(|&x| x == 0.0));
}

#[test]
fn a_sabotaged_gradient_is_caught_by_the_same_check() {
    let model = model_from(tiny_flyg());
    let head = WorldModelHead::new(&model, &WorldModelConfig::default()).unwrap();
    let mut params = head.init_default_params();
    for (i, w) in params.horizons[0].w_reg.iter_mut().enumerate() {
        *w = 0.2 * (i as f32 + 1.0);
    }
    let n = model.num_neurons();
    let r_full: Vec<f32> = (0..n).map(|i| 0.4 + 0.09 * i as f32).collect();
    let mut targets: [HorizonTargets; 3] = Default::default();
    targets[0].regression = Some([0.1; NUM_REGRESSION_TARGETS]);

    let (_loss, mut analytic, _grad_r) = world_model_loss_and_grad(&head, &r_full, n, &params, &targets);
    let biggest = (0..analytic.horizons[0].w_reg.len())
        .max_by(|&a, &b| {
            analytic.horizons[0].w_reg[a]
                .abs()
                .partial_cmp(&analytic.horizons[0].w_reg[b].abs())
                .unwrap()
        })
        .unwrap();
    analytic.horizons[0].w_reg[biggest] = 0.0;

    let h = 1e-2f32;
    let mut plus = params.clone();
    plus.horizons[0].w_reg[biggest] += h;
    let mut minus = params.clone();
    minus.horizons[0].w_reg[biggest] -= h;
    let fd = (loss_only(&head, &r_full, n, &plus, &targets) - loss_only(&head, &r_full, n, &minus, &targets))
        / (2.0 * f64::from(h));
    assert!(fd.abs() > 1e-4);
    assert_ne!(analytic.horizons[0].w_reg[biggest] as f64, fd);
}

// --- Ridge probe --------------------------------------------------------------------------------

#[test]
fn ridge_probe_recovers_an_exact_linear_relationship() {
    // y0 = 2*x0 - x1 + 3, y1 = -x0 + 0.5*x1 - 1, no noise -> R^2 should be ~1 even on held-out
    // points sampled the same way.
    let mut rng = crate::rng::SplitMix64::new(7);
    let mut features = Vec::new();
    let mut targets = Vec::new();
    for _ in 0..50 {
        let x0 = rng.next_f32_unit() * 4.0 - 2.0;
        let x1 = rng.next_f32_unit() * 4.0 - 2.0;
        features.push(vec![x0, x1]);
        targets.push(vec![2.0 * x0 - x1 + 3.0, -x0 + 0.5 * x1 - 1.0]);
    }
    let probe = fit_ridge_probe(&features, &targets, 1e-6).unwrap();

    let mut held_out_features = Vec::new();
    let mut held_out_targets = Vec::new();
    for _ in 0..20 {
        let x0 = rng.next_f32_unit() * 4.0 - 2.0;
        let x1 = rng.next_f32_unit() * 4.0 - 2.0;
        held_out_features.push(vec![x0, x1]);
        held_out_targets.push(vec![2.0 * x0 - x1 + 3.0, -x0 + 0.5 * x1 - 1.0]);
    }
    let r2 = probe.r_squared(&held_out_features, &held_out_targets);
    for &v in &r2 {
        assert!(v > 0.999, "r2={r2:?}");
    }
}

#[test]
fn ridge_probe_rejects_mismatched_sample_counts() {
    let features = vec![vec![1.0, 2.0]];
    let targets = vec![vec![1.0], vec![2.0]];
    assert!(fit_ridge_probe(&features, &targets, 1e-3).is_err());
}

#[test]
fn ridge_probe_rejects_empty_input() {
    assert!(fit_ridge_probe(&[], &[], 1e-3).is_err());
}

#[test]
fn gauss_jordan_solves_a_known_small_system() {
    // [2 1; 1 3] x = [5; 10] -> x = [1, 3]
    let mut a = vec![2.0, 1.0, 1.0, 3.0];
    let mut rhs = vec![5.0, 10.0];
    let x = gauss_jordan_solve(&mut a, &mut rhs, 2, 1).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-9, "x={x:?}");
    assert!((x[1] - 3.0).abs() < 1e-9, "x={x:?}");
}

#[test]
fn gauss_jordan_reports_none_on_a_singular_matrix() {
    let mut a = vec![1.0, 2.0, 2.0, 4.0]; // rank 1
    let mut rhs = vec![1.0, 2.0];
    assert!(gauss_jordan_solve(&mut a, &mut rhs, 2, 1).is_none());
}

#[test]
fn fit_ridge_probe_rejects_negative_l2() {
    let features = vec![vec![1.0], vec![2.0]];
    let targets = vec![vec![1.0], vec![2.0]];
    assert!(fit_ridge_probe(&features, &targets, -1.0).is_err());
}
