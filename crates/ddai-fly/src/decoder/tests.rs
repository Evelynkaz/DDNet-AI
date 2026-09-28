use super::*;
use crate::brain_fixtures::{FxNeuron, FxOutputGroup, FxType, build_brain_flyg};
use crate::config::FlyConfig;
use crate::params::FlyParams;
use ddai_flyg::{NeuronRole, Side, Sign};

/// A graph shaped like the real S/M output layer's action table (review round 1, F6): every
/// action's type has **two** members per side where the head pools both sides (`jump`/`hook`/
/// `fire`/`direction_stop`), so a pooled-head test can actually tell "mean of several slots" apart
/// from "one slot" -- and the tied `direction_left`/`direction_right` pair, and an `aim` fan with
/// two same-type pairs plus one deliberately unpaired member (side `M`, label **П**) so
/// [`build_aim_structure`]'s fallback path is exercised too.
fn tiny_decoder_flyg() -> ddai_flyg::Flyg {
    let type_names = [
        "DN_LR",
        "DN_STOP",
        "DN_JUMP",
        "DN_HOOK",
        "DN_FIRE",
        "DN_AIM_A",
        "DN_AIM_B",
        "DN_AIM_UNPAIRED",
    ];
    let types: Vec<FxType> = type_names
        .iter()
        .map(|&name| FxType {
            name,
            sign: Sign::Excitatory,
        })
        .collect();
    let mut neurons = Vec::new();
    // DN_LR (type 0): tied direction pair -- one L member (-> direction_left), one R member (->
    // direction_right).
    neurons.push(FxNeuron {
        type_index: 0,
        role: NeuronRole::Output,
        side: Side::L,
        full_connectome_in: 1,
        rf: (0.0, 0.0),
    });
    neurons.push(FxNeuron {
        type_index: 0,
        role: NeuronRole::Output,
        side: Side::R,
        full_connectome_in: 1,
        rf: (0.0, 0.0),
    });
    // DN_STOP (type 1): two members (L+R), pooled.
    for side in [Side::L, Side::R] {
        neurons.push(FxNeuron {
            type_index: 1,
            role: NeuronRole::Output,
            side,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        });
    }
    // DN_JUMP (type 2): two members (L+R), pooled.
    for side in [Side::L, Side::R] {
        neurons.push(FxNeuron {
            type_index: 2,
            role: NeuronRole::Output,
            side,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        });
    }
    // DN_HOOK (type 3): two members (L+R), pooled.
    for side in [Side::L, Side::R] {
        neurons.push(FxNeuron {
            type_index: 3,
            role: NeuronRole::Output,
            side,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        });
    }
    // DN_FIRE (type 4): two members (L+R), pooled.
    for side in [Side::L, Side::R] {
        neurons.push(FxNeuron {
            type_index: 4,
            role: NeuronRole::Output,
            side,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        });
    }
    // DN_AIM_A, DN_AIM_B (types 5, 6): two tied aim pairs.
    for ti in [5u32, 6] {
        neurons.push(FxNeuron {
            type_index: ti,
            role: NeuronRole::Output,
            side: Side::L,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        });
        neurons.push(FxNeuron {
            type_index: ti,
            role: NeuronRole::Output,
            side: Side::R,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        });
    }
    // DN_AIM_UNPAIRED (type 7): one member, side M -- no same-type opposite-side partner, falls
    // back to an independent angle (label **П**).
    neurons.push(FxNeuron {
        type_index: 7,
        role: NeuronRole::Output,
        side: Side::M,
        full_connectome_in: 1,
        rf: (0.0, 0.0),
    });

    let output_groups = vec![
        FxOutputGroup {
            action: "direction_left",
            member_type_names: vec!["DN_LR"],
            side_filter: Some(Side::L),
        },
        FxOutputGroup {
            action: "direction_right",
            member_type_names: vec!["DN_LR"],
            side_filter: Some(Side::R),
        },
        FxOutputGroup {
            action: "direction_stop",
            member_type_names: vec!["DN_STOP"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "jump",
            member_type_names: vec!["DN_JUMP"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "hook",
            member_type_names: vec!["DN_HOOK"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "fire",
            member_type_names: vec!["DN_FIRE"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "aim",
            member_type_names: vec!["DN_AIM_A", "DN_AIM_B", "DN_AIM_UNPAIRED"],
            side_filter: None,
        },
    ];
    build_brain_flyg(&types, &neurons, &[], &[], &output_groups)
}

fn model_from(flyg: ddai_flyg::Flyg) -> FlyModel {
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 1);
    FlyModel::new(flyg, config, params).unwrap()
}

fn tiny_decoder() -> (FlyModel, DecoderModel) {
    let model = model_from(tiny_decoder_flyg());
    let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
    (model, decoder)
}

// --- Calibration protocol (review round 1, F10) --------------------------------------------

#[test]
fn calibrate_from_rest_produces_a_usable_calibration() {
    let (model, decoder) = tiny_decoder();
    let calib = calibrate_from_rest(&model, 7, decoder.config().min_sigma).expect("must fit");
    assert_eq!(calib.mu.len(), model.num_outputs());
    assert_eq!(calib.sigma.len(), model.num_outputs());
    assert!(calib.mu.iter().all(|x| x.is_finite()));
    assert!(
        calib
            .sigma
            .iter()
            .all(|&s| s >= decoder.config().min_sigma && s.is_finite())
    );
}

#[test]
fn calibrate_from_rest_is_deterministic_for_the_same_seed() {
    let (model, decoder) = tiny_decoder();
    let a = calibrate_from_rest(&model, 42, decoder.config().min_sigma).unwrap();
    let b = calibrate_from_rest(&model, 42, decoder.config().min_sigma).unwrap();
    assert_eq!(a, b);
}

// `tiny_decoder_flyg` has no input neurons by design (decoder-only tests don't need an encoder
// pipeline) -- with zero inputs, the per-seed jitter this protocol draws never has anywhere to
// land, so "differs across seeds" isn't a meaningful property to check *here*; it's `crate::rng`'s
// own job (`different_seeds_give_different_sequences`) and is exercised for real on a graph that
// actually has inputs by `tests/brain_demo_generalization.rs` (every seed there gets a distinct
// calibration feeding into a distinct trained decoder).

/// A small graph with real `L`/`R` input pairs (unlike `tiny_decoder_flyg`, which has none) and a
/// tied `DN_LR` direction output -- built to isolate [`calibrate_from_rest`]'s own mirror-symmetry
/// property (review round 2, F19) from the rest of the encoder/connectome pipeline, the way
/// `tests/brain_mirror.rs`'s action-level test exercises it end to end.
fn symmetric_flyg_with_inputs() -> ddai_flyg::Flyg {
    let types = [
        FxType {
            name: "VPN_A",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "HID",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "DN_LR",
            sign: Sign::Excitatory,
        },
    ];
    let neurons = vec![
        FxNeuron {
            type_index: 0,
            role: NeuronRole::InputVisual,
            side: Side::L,
            full_connectome_in: 1000,
            rf: (-45.0, 0.0),
        },
        FxNeuron {
            type_index: 0,
            role: NeuronRole::InputVisual,
            side: Side::R,
            full_connectome_in: 1000,
            rf: (45.0, 0.0),
        },
        FxNeuron {
            type_index: 1,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        },
        FxNeuron {
            type_index: 2,
            role: NeuronRole::Output,
            side: Side::L,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        },
        FxNeuron {
            type_index: 2,
            role: NeuronRole::Output,
            side: Side::R,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        },
    ];
    let edges = vec![
        crate::brain_fixtures::FxEdge {
            pre: 0,
            post: 2,
            synapse_count: 5,
        },
        crate::brain_fixtures::FxEdge {
            pre: 1,
            post: 2,
            synapse_count: 5,
        },
        crate::brain_fixtures::FxEdge {
            pre: 2,
            post: 3,
            synapse_count: 5,
        },
        crate::brain_fixtures::FxEdge {
            pre: 2,
            post: 4,
            synapse_count: 5,
        },
    ];
    let output_groups = vec![
        FxOutputGroup {
            action: "direction_left",
            member_type_names: vec!["DN_LR"],
            side_filter: Some(Side::L),
        },
        FxOutputGroup {
            action: "direction_right",
            member_type_names: vec!["DN_LR"],
            side_filter: Some(Side::R),
        },
    ];
    build_brain_flyg(&types, &neurons, &edges, &[], &output_groups)
}

/// Review round 2, F19 (CONFIRMED): `calibrate_from_rest` must give an `L`-side DN exactly the
/// same `mu`/`sigma` as its `R`-side homolog -- an earlier revision's independent per-input-neuron
/// jitter broke this (measured ~1e-3 `mu` gap, amplified by the `sigma` floor into a real
/// action-level asymmetry; see that function's doc comment). Only a partial decoder config is
/// needed here (`direction_left`/`direction_right` only) -- this test is about the calibration
/// step alone, not every head.
#[test]
fn calibrate_from_rest_gives_l_r_homologs_the_same_mu_and_sigma() {
    let flyg = symmetric_flyg_with_inputs();
    let model = model_from(flyg);
    let cfg = DecoderConfig {
        jump_action: "direction_left".to_string(),
        hook_action: "direction_left".to_string(),
        fire_action: "direction_left".to_string(),
        aim_action: "direction_left".to_string(),
        direction_actions: [
            "direction_left".to_string(),
            "direction_left".to_string(),
            "direction_right".to_string(),
        ],
        ..DecoderConfig::default()
    };
    // Only `direction_left`'s single slot and `direction_right`'s single slot matter for this
    // test; every other head is pointed at `direction_left` too purely so `DecoderModel::new`
    // resolves at all (it needs *some* member for every configured action) -- their own values
    // are never read below.
    // Built only to confirm the config above actually resolves (`DecoderModel::new` errors
    // otherwise) -- the output slots this test reads are known directly from `symmetric_flyg_
    // with_inputs`'s own fixed construction order below, not from this model.
    let _decoder = DecoderModel::new(&model, cfg).unwrap();
    let calib = calibrate_from_rest(&model, 7, 0.05).unwrap();

    // Dense output slots: index 3 = `DN_LR`-`L`, index 4 = `DN_LR`-`R` (this fixture's own
    // construction order) -- both are `Output`-role, so their output slot equals their position
    // among `Output`-role neurons in ascending dense-index order (0 and 1 respectively).
    let (slot_l, slot_r) = (0usize, 1usize);
    assert_eq!(
        calib.mu[slot_l], calib.mu[slot_r],
        "L/R DN homologs must get exactly the same mu from mirror-symmetric jitter"
    );
    assert_eq!(
        calib.sigma[slot_l], calib.sigma[slot_r],
        "L/R DN homologs must get exactly the same sigma from mirror-symmetric jitter"
    );
}

// --- Config / construction validation ------------------------------------------------------

#[test]
fn default_config_validates() {
    DecoderConfig::default().validate().unwrap();
}

#[test]
fn zero_kappa_is_rejected() {
    let cfg = DecoderConfig {
        aim_kappa: 0.0,
        ..DecoderConfig::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn unknown_action_name_is_rejected() {
    let model = model_from(tiny_decoder_flyg());
    let cfg = DecoderConfig {
        jump_action: "not_a_real_action".to_string(),
        ..DecoderConfig::default()
    };
    assert!(matches!(
        DecoderModel::new(&model, cfg),
        Err(DecoderError::UnknownAction(_))
    ));
}

/// Review round 1, F6: a type present in only one of `direction_left`/`direction_right` cannot be
/// tied -- must be a clear, named error, not a silently empty/malformed group.
#[test]
fn a_direction_type_on_only_one_side_is_rejected() {
    let type_names = ["DN_ONLY_LEFT", "DN_STOP", "DN_JUMP", "DN_HOOK", "DN_FIRE", "DN_AIM"];
    let types: Vec<FxType> = type_names
        .iter()
        .map(|&name| FxType {
            name,
            sign: Sign::Excitatory,
        })
        .collect();
    let mut neurons = vec![FxNeuron {
        type_index: 0,
        role: NeuronRole::Output,
        side: Side::L,
        full_connectome_in: 1,
        rf: (0.0, 0.0),
    }];
    for ti in 1..type_names.len() {
        neurons.push(FxNeuron {
            type_index: ti as u32,
            role: NeuronRole::Output,
            side: Side::M,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        });
    }
    let output_groups = vec![
        FxOutputGroup {
            action: "direction_left",
            member_type_names: vec!["DN_ONLY_LEFT"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "direction_right",
            // Deliberately empty on the right (an `UnknownAction`, not a
            // `DirectionTypeNotOnBothSides` -- resolving the group itself fails first when it has
            // zero members). Use a different, present type to isolate the case under test: a type
            // that appears in `direction_left` but never in `direction_right`.
            member_type_names: vec!["DN_STOP"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "direction_stop",
            member_type_names: vec!["DN_STOP"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "jump",
            member_type_names: vec!["DN_JUMP"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "hook",
            member_type_names: vec!["DN_HOOK"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "fire",
            member_type_names: vec!["DN_FIRE"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "aim",
            member_type_names: vec!["DN_AIM"],
            side_filter: None,
        },
    ];
    let flyg = build_brain_flyg(&types, &neurons, &[], &[], &output_groups);
    let model = model_from(flyg);
    assert!(matches!(
        DecoderModel::new(&model, DecoderConfig::default()),
        Err(DecoderError::DirectionTypeNotOnBothSides { .. })
    ));
}

#[test]
fn decoder_model_resolves_the_right_number_of_members_per_head() {
    let (_model, decoder) = tiny_decoder();
    assert_eq!(decoder.direction_lr.len(), 1, "one tied type");
    assert_eq!(decoder.direction_stop.len(), 1);
    assert_eq!(decoder.direction_stop[0].slots.len(), 2, "L+R pooled");
    assert_eq!(decoder.jump.len(), 1);
    assert_eq!(decoder.jump[0].slots.len(), 2);
    assert_eq!(decoder.hook[0].slots.len(), 2);
    assert_eq!(decoder.fire[0].slots.len(), 2);
    assert_eq!(decoder.aim_pairs.len(), 2, "DN_AIM_A + DN_AIM_B tied pairs");
    assert_eq!(decoder.aim_unpaired.len(), 1, "DN_AIM_UNPAIRED, side M");
}

// --- Calibration -----------------------------------------------------------------------------

#[test]
fn calibration_fits_mean_and_std_from_hand_worked_samples() {
    let samples = vec![vec![1.0, 5.0], vec![3.0, 5.0]]; // slot 0: mean 2, std 1; slot 1: constant 5
    let calib = DnCalibration::fit(2, &samples, 0.01).unwrap();
    assert!((calib.mu[0] - 2.0).abs() < 1e-6);
    assert!((calib.sigma[0] - 1.0).abs() < 1e-6);
    assert!((calib.mu[1] - 5.0).abs() < 1e-6);
    assert_eq!(
        calib.sigma[1], 0.01,
        "a constant signal must floor at min_sigma, not go to 0"
    );
}

#[test]
fn calibration_rejects_empty_or_wrong_length_samples() {
    assert!(DnCalibration::fit(2, &[], 0.01).is_err());
    assert!(DnCalibration::fit(2, &[vec![1.0]], 0.01).is_err());
}

#[test]
fn z_clips_to_the_configured_bound() {
    let calib = DnCalibration {
        mu: vec![0.0],
        sigma: vec![0.1],
    };
    let z = calib.z(&[100.0], 10.0);
    assert_eq!(z[0], 10.0);
    let z = calib.z(&[-100.0], 10.0);
    assert_eq!(z[0], -10.0);
}

// --- Forward -----------------------------------------------------------------------------------

#[test]
fn direction_probs_sum_to_one_and_are_nonnegative() {
    let (model, decoder) = tiny_decoder();
    let params = decoder.init_default_params();
    let calib = DnCalibration {
        mu: vec![0.0; model.num_outputs()],
        sigma: vec![1.0; model.num_outputs()],
    };
    let dn_rates = vec![0.3; model.num_outputs()];
    let action = decoder_forward(&decoder, &dn_rates, &calib, &params);
    let sum: f32 = action.direction_probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(action.direction_probs.iter().all(|&p| p >= 0.0));
    assert!((0.0..=1.0).contains(&action.jump_prob));
    assert!((0.0..=1.0).contains(&action.hook_prob));
    assert!((0.0..=1.0).contains(&action.fire_prob));
}

#[test]
fn aim_population_vector_points_towards_the_dominant_preferred_angle() {
    let (model, decoder) = tiny_decoder();
    let mut params = decoder.init_default_params();
    // Force the first tied pair's preferred angle to 0, the second to pi. Per `population_vector`,
    // a pair's *asymmetry* (`zl - zr`) drives the `cos(theta)` (C) term and its *sum* (`zl + zr`)
    // drives the `sin(theta)` (S) term -- so to make the vector point at the first pair's angle
    // (0), that pair needs a strongly asymmetric drive (not equal L/R, which would instead load
    // onto S), while the second pair and the unpaired member stay quiet.
    params.aim_pair_theta = vec![0.0, std::f32::consts::PI];
    params.aim_unpaired_theta = vec![std::f32::consts::PI / 2.0];
    let calib = DnCalibration {
        mu: vec![0.0; model.num_outputs()],
        sigma: vec![1.0; model.num_outputs()],
    };
    let mut dn_rates = vec![0.0; model.num_outputs()];
    dn_rates[decoder.aim_pairs[0].l_slot] = 5.0;
    dn_rates[decoder.aim_pairs[0].r_slot] = 0.1;
    dn_rates[decoder.aim_pairs[1].l_slot] = 0.1;
    dn_rates[decoder.aim_pairs[1].r_slot] = 0.1;
    dn_rates[decoder.aim_unpaired[0]] = 0.0;
    let action = decoder_forward(&decoder, &dn_rates, &calib, &params);
    assert!(action.aim_angle.abs() < 0.2, "aim_angle={}", action.aim_angle);
}

// --- Mirror symmetry (review round 1, F6, acceptance criterion 8) ------------------------------

/// Swaps every tied pair's `L`/`R` slot: `direction_lr`'s explicit `left`/`right`, `aim_pairs`'
/// `l_slot`/`r_slot`, and -- for the pooled heads (`direction_stop`/`jump`/`hook`/`fire`) -- each
/// two-member `TypeGroup`'s two slots (this fixture always pushes a type's `L` member before its
/// `R` member, so `slots == [l_slot, r_slot]`; see `tiny_decoder_flyg`). Swapping the pooled
/// heads' slots too (even though a mean is invariant to it either way) makes this the real
/// action-level analogue of a full `Observation::mirror_x` -- not just "we didn't touch those
/// slots so of course they matched".
fn mirror_z(decoder: &DecoderModel, z: &[f32]) -> Vec<f32> {
    let mut out = z.to_vec();
    for g in &decoder.direction_lr {
        for (&l, &r) in g.left.iter().zip(&g.right) {
            out.swap(l, r);
        }
    }
    for pair in &decoder.aim_pairs {
        out.swap(pair.l_slot, pair.r_slot);
    }
    for group in decoder
        .direction_stop
        .iter()
        .chain(&decoder.jump)
        .chain(&decoder.hook)
        .chain(&decoder.fire)
    {
        if group.slots.len() == 2 {
            out.swap(group.slots[0], group.slots[1]);
        }
    }
    out
}

#[test]
fn mirroring_z_gives_the_mirrored_decoded_action() {
    let (model, decoder) = tiny_decoder();
    let params = decoder.init_default_params();
    let calib = DnCalibration {
        mu: vec![0.0; model.num_outputs()],
        sigma: vec![1.0; model.num_outputs()],
    };
    // The unpaired aim member (label **П**, side `M`) has no same-type opposite-side partner to
    // tie it to, so `mirror_z` deliberately never touches its slot -- its own independent angle
    // is *not* part of the guaranteed mirror-symmetric aim structure the module doc comment
    // describes (which holds only for the tied pairs; the real S/M graphs have no unpaired aim
    // members at all, see `crate::decoder`'s module doc comment). Zeroing its rate here (`z ==
    // 0`, since `calib.mu == 0`) removes its contribution from both `C`/`S` entirely so this test
    // isolates and checks exactly the guaranteed part.
    let mut dn_rates: Vec<f32> = (0..model.num_outputs()).map(|i| 0.3 + 0.11 * i as f32).collect();
    dn_rates[decoder.aim_unpaired[0]] = 0.0;
    let z = calib.z(&dn_rates, decoder.config().z_clip);
    let mirrored_z = mirror_z(&decoder, &z);
    // Round-trip back through the (frozen, symmetric) calibration to get a "mirrored dn_rates" to
    // decode -- `mu`/`sigma` here are uniform so this is exact.
    let mirrored_dn_rates: Vec<f32> = mirrored_z
        .iter()
        .zip(&calib.mu)
        .zip(&calib.sigma)
        .map(|((&zi, &mu), &sigma)| zi * sigma + mu)
        .collect();

    let action = decoder_forward(&decoder, &dn_rates, &calib, &params);
    let mirrored_action = decoder_forward(&decoder, &mirrored_dn_rates, &calib, &params);

    assert!(
        (action.direction_probs[0] - mirrored_action.direction_probs[2]).abs() < 1e-5,
        "left(original) must equal right(mirrored): {} vs {}",
        action.direction_probs[0],
        mirrored_action.direction_probs[2]
    );
    assert!(
        (action.direction_probs[2] - mirrored_action.direction_probs[0]).abs() < 1e-5,
        "right(original) must equal left(mirrored)"
    );
    assert!(
        (action.direction_probs[1] - mirrored_action.direction_probs[1]).abs() < 1e-5,
        "stop must be side-invariant"
    );
    assert!((action.jump_prob - mirrored_action.jump_prob).abs() < 1e-5);
    assert!((action.hook_prob - mirrored_action.hook_prob).abs() < 1e-5);
    assert!((action.fire_prob - mirrored_action.fire_prob).abs() < 1e-5);

    // pi - angle, wrapped into (-pi, pi] the same way `atan2` already returns.
    let expected_mirrored_aim = {
        let a = std::f32::consts::PI - action.aim_angle;
        a.sin().atan2(a.cos()) // wrap to atan2's own range without a manual branch.
    };
    let diff = (mirrored_action.aim_angle - expected_mirrored_aim).abs();
    let wrapped = diff.min((2.0 * std::f32::consts::PI - diff).abs());
    assert!(
        wrapped < 1e-4,
        "aim_angle={} mirrored={} expected={}",
        action.aim_angle,
        mirrored_action.aim_angle,
        expected_mirrored_aim
    );
}

// --- Gradient checks (task spec: "tiny graphs, f64 reference") -----------------------------------

fn loss_only(
    decoder: &DecoderModel,
    dn_rates: &[f32],
    calib: &DnCalibration,
    params: &DecoderParams,
    targets: &DecoderTargets,
) -> f64 {
    f64::from(decoder_loss_and_grad(decoder, dn_rates, calib, params, targets).0)
}

#[test]
fn decoder_gradients_match_finite_differences_for_every_head() {
    let (model, decoder) = tiny_decoder();
    let mut params = decoder.init_default_params();
    params.direction_lr_w = vec![0.7];
    params.direction_lr_b = -0.2;
    params.direction_stop_w = vec![-0.4];
    params.direction_stop_b = 0.1;
    params.jump_w = vec![0.5];
    params.jump_b = 0.05;
    params.hook_w = vec![0.3];
    params.hook_b = -0.1;
    params.fire_w = vec![-0.6];
    params.fire_b = 0.2;
    params.aim_pair_theta = vec![0.4, 2.1];
    params.aim_unpaired_theta = vec![1.1];

    let calib = DnCalibration {
        mu: vec![0.5; model.num_outputs()],
        sigma: vec![1.3; model.num_outputs()],
    };
    // Avoid the exact aim degenerate point (population vector = (0,0)) and the exact clip
    // boundary (same discipline as `tests/backward_correctness.rs`): pick generic, nonzero rates.
    let dn_rates: Vec<f32> = (0..model.num_outputs()).map(|i| 0.4 + 0.13 * i as f32).collect();
    let targets = DecoderTargets {
        direction: Some(2),
        jump: Some(true),
        hook: Some(false),
        fire: Some(true),
        aim: Some(1.0),
    };

    let (_loss, analytic, grad_dn) = decoder_loss_and_grad(&decoder, &dn_rates, &calib, &params, &targets);

    let h = 1e-3f32;
    // `rel < rel_tol` OR `abs_diff < abs_tol`: a pure relative tolerance is too strict for a
    // small-magnitude gradient sitting close to `f32`'s own central-difference roundoff floor
    // (the loss is computed in `f32` throughout `decoder_loss_and_grad`) -- same combined
    // convention `tests/backward_correctness.rs` uses for the real-graph spot checks ("abs_tol
    // scaled to the group's max |g|"), just with a fixed empirical floor here instead of a
    // per-group scale (this test's gradients are all the same rough order of magnitude).
    let close_enough = |analytic: f64, fd: f64| -> bool {
        let rel = (analytic - fd).abs() / fd.abs().max(1e-6);
        rel < 1e-3 || (analytic - fd).abs() < 2e-4
    };

    // grad_dn (w.r.t. dn_rates directly).
    for i in 0..dn_rates.len() {
        let mut plus = dn_rates.clone();
        plus[i] += h;
        let mut minus = dn_rates.clone();
        minus[i] -= h;
        let fd = (loss_only(&decoder, &plus, &calib, &params, &targets)
            - loss_only(&decoder, &minus, &calib, &params, &targets))
            / (2.0 * f64::from(h));
        assert!(
            close_enough(f64::from(grad_dn[i]), fd),
            "grad_dn[{i}]: analytic={} fd={fd}",
            grad_dn[i]
        );
    }

    // A small helper closure so the same perturb-recompute-compare logic isn't repeated once per
    // scalar/vector field.
    macro_rules! check_scalar {
        ($field:ident) => {{
            let mut plus = params.clone();
            plus.$field += h;
            let mut minus = params.clone();
            minus.$field -= h;
            let fd = (loss_only(&decoder, &dn_rates, &calib, &plus, &targets)
                - loss_only(&decoder, &dn_rates, &calib, &minus, &targets))
                / (2.0 * f64::from(h));
            assert!(
                close_enough(f64::from(analytic.$field), fd),
                "{}: analytic={} fd={fd}",
                stringify!($field),
                analytic.$field
            );
        }};
    }
    check_scalar!(direction_lr_b);
    check_scalar!(direction_stop_b);
    check_scalar!(jump_b);
    check_scalar!(hook_b);
    check_scalar!(fire_b);

    macro_rules! check_vec {
        ($field:ident) => {{
            for i in 0..params.$field.len() {
                let mut plus = params.clone();
                plus.$field[i] += h;
                let mut minus = params.clone();
                minus.$field[i] -= h;
                let fd = (loss_only(&decoder, &dn_rates, &calib, &plus, &targets)
                    - loss_only(&decoder, &dn_rates, &calib, &minus, &targets))
                    / (2.0 * f64::from(h));
                assert!(
                    close_enough(f64::from(analytic.$field[i]), fd),
                    "{}[{i}]: analytic={} fd={fd}",
                    stringify!($field),
                    analytic.$field[i]
                );
            }
        }};
    }
    check_vec!(direction_lr_w);
    check_vec!(direction_stop_w);
    check_vec!(jump_w);
    check_vec!(hook_w);
    check_vec!(fire_w);
    check_vec!(aim_pair_theta);
    check_vec!(aim_unpaired_theta);
}

/// Review lesson from task 7.2: a gradient check must be able to fail. Sabotaging the analytic
/// gradient must break the same finite-difference comparison the check above relies on.
#[test]
fn a_sabotaged_gradient_is_caught_by_the_same_check() {
    let (model, decoder) = tiny_decoder();
    let mut params = decoder.init_default_params();
    params.direction_lr_w = vec![0.35];
    let calib = DnCalibration {
        mu: vec![0.0; model.num_outputs()],
        sigma: vec![1.0; model.num_outputs()],
    };
    let dn_rates: Vec<f32> = (0..model.num_outputs()).map(|i| 0.3 + 0.1 * i as f32).collect();
    let targets = DecoderTargets {
        direction: Some(0),
        ..Default::default()
    };
    let (_loss, mut analytic, _grad_dn) = decoder_loss_and_grad(&decoder, &dn_rates, &calib, &params, &targets);
    analytic.direction_lr_w[0] = 0.0; // sabotage

    let h = 1e-3f32;
    let mut plus = params.clone();
    plus.direction_lr_w[0] += h;
    let mut minus = params.clone();
    minus.direction_lr_w[0] -= h;
    let fd = (loss_only(&decoder, &dn_rates, &calib, &plus, &targets)
        - loss_only(&decoder, &dn_rates, &calib, &minus, &targets))
        / (2.0 * f64::from(h));
    assert!(
        fd.abs() > 1e-4,
        "the finite difference itself must be meaningfully nonzero"
    );
    assert_ne!(analytic.direction_lr_w[0] as f64, fd);
}

#[test]
fn l1_gradient_has_the_sign_of_the_weight_and_is_zero_exactly_at_zero() {
    let (_model, decoder) = tiny_decoder();
    let mut params = decoder.init_default_params();
    params.jump_w = vec![0.0];
    params.hook_w = vec![-2.0];
    let mut grads = decoder.zeros_gradients();
    add_l1_penalty(&mut grads, &params, 0.5);
    assert_eq!(grads.jump_w[0], 0.0, "L1 subgradient must be exactly 0 at w == 0");
    assert!(
        grads.hook_w[0] < 0.0,
        "L1 gradient must have the sign of a negative weight"
    );
}

#[test]
fn l1_penalty_is_a_noop_when_weight_is_zero() {
    let (_model, decoder) = tiny_decoder();
    let mut params = decoder.init_default_params();
    params.jump_w = vec![3.0];
    params.hook_w = vec![-2.0];
    let mut grads = decoder.zeros_gradients();
    add_l1_penalty(&mut grads, &params, 0.0);
    assert!(grads.jump_w.iter().chain(&grads.hook_w).all(|&x| x == 0.0));
}

/// The L1 penalty's own value (`weight * |w|`, summed over every linear head's `W`) must
/// finite-difference-match [`add_l1_penalty`]'s gradient on its own -- checked separately from the
/// classification loss precisely because [`decoder_loss_and_grad`] does **not** include it (see
/// that function's doc comment on why the two are kept apart).
#[test]
fn l1_penalty_matches_finite_differences_of_the_l1_norm() {
    let (_model, decoder) = tiny_decoder();
    let mut params = decoder.init_default_params();
    params.jump_w = vec![1.5];
    params.hook_w = vec![-0.7];
    let weight = 0.3f32;
    let mut grads = decoder.zeros_gradients();
    add_l1_penalty(&mut grads, &params, weight);

    let l1_value = |w: &[f32]| -> f64 { f64::from(weight) * w.iter().map(|x| f64::from(x.abs())).sum::<f64>() };
    let h = 1e-3f32;
    for (name, w, gw) in [
        ("jump_w", &params.jump_w, &grads.jump_w),
        ("hook_w", &params.hook_w, &grads.hook_w),
    ] {
        for i in 0..w.len() {
            let mut plus = w.clone();
            plus[i] += h;
            let mut minus = w.clone();
            minus[i] -= h;
            let fd = (l1_value(&plus) - l1_value(&minus)) / (2.0 * f64::from(h));
            let rel = (f64::from(gw[i]) - fd).abs() / fd.abs().max(1e-6);
            assert!(rel < 1e-4, "{name}[{i}]: analytic={} fd={fd} rel={rel}", gw[i]);
        }
    }
}

#[test]
fn aim_degenerate_zero_population_vector_does_not_produce_nan() {
    let (loss, dc, ds) = aim_loss_and_grad(0.0, 0.0, 1.0, 4.0);
    assert!(loss.is_finite());
    assert!(dc.is_finite() && ds.is_finite());
}

#[test]
fn no_targets_gives_zero_loss_and_zero_gradients() {
    let (model, decoder) = tiny_decoder();
    let params = decoder.init_default_params();
    let calib = DnCalibration {
        mu: vec![0.0; model.num_outputs()],
        sigma: vec![1.0; model.num_outputs()],
    };
    let dn_rates = vec![0.2; model.num_outputs()];
    let (loss, grads, grad_dn) =
        decoder_loss_and_grad(&decoder, &dn_rates, &calib, &params, &DecoderTargets::default());
    assert_eq!(loss, 0.0);
    assert!(grad_dn.iter().all(|&x| x == 0.0));
    assert!(grads.direction_lr_w.iter().all(|&x| x == 0.0));
    assert_eq!(grads.direction_lr_b, 0.0);
}

#[test]
fn decoder_forward_into_matches_decoder_forward() {
    let (model, decoder) = tiny_decoder();
    let params = decoder.init_default_params();
    let calib = DnCalibration {
        mu: vec![0.1; model.num_outputs()],
        sigma: vec![0.9; model.num_outputs()],
    };
    let dn_rates: Vec<f32> = (0..model.num_outputs()).map(|i| 0.2 + 0.05 * i as f32).collect();
    let via_alloc = decoder_forward(&decoder, &dn_rates, &calib, &params);
    let mut scratch = DecoderScratch::new(&decoder);
    let via_scratch = decoder_forward_into(&decoder, &dn_rates, &calib, &params, &mut scratch);
    assert_eq!(via_alloc, via_scratch);
}
