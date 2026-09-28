use super::*;
use crate::brain_fixtures::{FxEdge, FxInputChannel, FxNeuron, FxOutputGroup, FxType, build_brain_flyg};
use crate::config::FlyConfig;
use crate::decoder::{DecoderConfig, DecoderModel};
use crate::encoder::{EncoderModel, EncoderParams, ProprioceptionConfig, RayGridConfig};
use crate::params::FlyParams;
use crate::world_model::{WorldModelConfig, WorldModelHead};
use ddai_brain::CharacterObservation;
use ddai_flyg::{NeuronRole, Side, Sign};
use std::sync::Arc;

fn tiny_flyg() -> ddai_flyg::Flyg {
    let type_names = [
        "VPN_OPP",
        "AN_GROUND",
        "HID",
        "DN_LR",
        "DN_STOP",
        "DN_JUMP",
        "DN_HOOK",
        "DN_FIRE",
        "DN_AIM",
    ];
    let types: Vec<FxType> = type_names
        .iter()
        .map(|&name| FxType {
            name,
            sign: Sign::Excitatory,
        })
        .collect();
    let mut neurons = Vec::new();
    neurons.push(FxNeuron {
        type_index: 0,
        role: NeuronRole::InputVisual,
        side: Side::L,
        full_connectome_in: 1000,
        rf: (-45.0, 0.0),
    });
    neurons.push(FxNeuron {
        type_index: 0,
        role: NeuronRole::InputVisual,
        side: Side::R,
        full_connectome_in: 1000,
        rf: (45.0, 0.0),
    });
    neurons.push(FxNeuron {
        type_index: 1,
        role: NeuronRole::InputAscending,
        side: Side::M,
        full_connectome_in: 1000,
        rf: (0.0, 0.0),
    });
    for _ in 0..2 {
        neurons.push(FxNeuron {
            type_index: 2,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        });
    }
    // Direction: one tied L/R pair on the same type (review round 1, F6 -- matches how the real
    // .flyg's own output_groups assign each side's DN to its own action name, never both).
    neurons.push(FxNeuron {
        type_index: 3,
        role: NeuronRole::Output,
        side: Side::L,
        full_connectome_in: 1000,
        rf: (0.0, 0.0),
    });
    neurons.push(FxNeuron {
        type_index: 3,
        role: NeuronRole::Output,
        side: Side::R,
        full_connectome_in: 1000,
        rf: (0.0, 0.0),
    });
    for ti in 4..type_names.len() {
        neurons.push(FxNeuron {
            type_index: ti as u32,
            role: NeuronRole::Output,
            side: Side::M,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        });
    }
    let input_indices: Vec<u32> = (0..3).collect();
    let hidden_indices: Vec<u32> = (3..5).collect();
    let output_indices: Vec<u32> = (5..12).collect();
    let mut edges = Vec::new();
    for &pre in &input_indices {
        for &post in &hidden_indices {
            edges.push(FxEdge {
                pre,
                post,
                synapse_count: 4,
            });
        }
    }
    for &pre in &hidden_indices {
        for &post in &output_indices {
            edges.push(FxEdge {
                pre,
                post,
                synapse_count: 4,
            });
        }
    }
    let input_channels = vec![FxInputChannel {
        type_name: "VPN_OPP",
        channels: vec!["opponent_position"],
    }];
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
            member_type_names: vec!["DN_AIM"],
            side_filter: None,
        },
    ];
    build_brain_flyg(&types, &neurons, &edges, &input_channels, &output_groups)
}

fn tiny_map() -> ddai_physics::map::MapData {
    ddai_physics::map::MapData {
        width: 20,
        height: 20,
        game: vec![Default::default(); 400],
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }
}

fn sample_observation(opp_x: f32) -> Observation {
    let mut me = CharacterObservation::at_rest(0);
    me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
    let mut opp = CharacterObservation::at_rest(1);
    opp.pos = ddai_physics::vmath::Vec2::new(opp_x, 300.0);
    Observation {
        map: Arc::new(tiny_map()),
        tick: 0,
        self_state: me,
        others: vec![opp],
        target_id: None,
        tuning: ddai_physics::tuning::TuningParams::default(),
    }
}

struct Fixture {
    model: FlyModel,
    index: BackwardIndex,
    encoder: EncoderModel,
    encoder_params: EncoderParams,
    decoder: DecoderModel,
    decoder_params: crate::decoder::DecoderParams,
    calib: DnCalibration,
    world_model: WorldModelHead,
    world_model_params: WorldModelParams,
    seq: BrainSequence,
}

fn build_fixture() -> Fixture {
    build_fixture_with_wm_weight(WorldModelConfig::default().loss_weight)
}

fn build_fixture_with_wm_weight(wm_loss_weight: f32) -> Fixture {
    let flyg = tiny_flyg();
    let config = FlyConfig {
        substeps_per_decision: 2,
        ..FlyConfig::default()
    };
    let params = FlyParams::init_default(&flyg, &config, 3);
    let model = FlyModel::new(flyg, config, params).unwrap();
    let index = BackwardIndex::build(&model);

    let encoder = EncoderModel::new(
        &model,
        RayGridConfig::default(),
        &ProprioceptionConfig {
            grounded: vec!["AN_GROUND".to_string()],
            ..ProprioceptionConfig::default()
        },
    )
    .unwrap();
    let mut encoder_params = EncoderParams::init_default(encoder.num_params());
    for (i, g) in encoder_params.g.iter_mut().enumerate() {
        *g = 0.8 + 0.15 * i as f32;
    }
    for (i, c) in encoder_params.c.iter_mut().enumerate() {
        *c = 0.05 * i as f32;
    }

    let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
    let mut decoder_params = decoder.init_default_params();
    decoder_params.direction_lr_w = vec![-0.15];
    decoder_params.direction_lr_b = 0.05;
    decoder_params.direction_stop_w = vec![0.1];
    decoder_params.jump_w = vec![0.2];
    decoder_params.hook_w = vec![0.15];
    decoder_params.fire_w = vec![-0.2];
    // `tiny_flyg`'s single AIM neuron has side `M` (no same-type opposite-side partner), so it
    // resolves to `aim_unpaired`, not a tied pair.
    decoder_params.aim_unpaired_theta = vec![0.3];

    let calib = DnCalibration {
        mu: vec![0.05; model.num_outputs()],
        sigma: vec![1.1; model.num_outputs()],
    };

    let world_model = WorldModelHead::new(
        &model,
        &WorldModelConfig {
            loss_weight: wm_loss_weight,
            ..WorldModelConfig::default()
        },
    )
    .unwrap();
    let mut world_model_params = world_model.init_default_params();
    for (i, w) in world_model_params.horizons[0].w_reg.iter_mut().enumerate() {
        *w = 0.05 * (i as f32 - 4.0);
    }
    for (i, w) in world_model_params.horizons[0].w_bin.iter_mut().enumerate() {
        *w = 0.07 * (i as f32 - 1.0);
    }

    let observations: Vec<Observation> = vec![sample_observation(350.0), sample_observation(420.0)];
    // World-model targets are the *unperturbed* prediction plus a small offset (see
    // `crate::world_model::tests`'s own gradient-check rationale for why: keeps every squared
    // term small and the finite-difference well-conditioned regardless of how the weights above
    // happen to scale).
    let v_init = vec![0.05f32; model.num_neurons()];
    let unperturbed_preds = {
        // `brain_train_step` doesn't expose an intermediate decision's `r_last` directly, so
        // this replays just the forward half by hand (tiny graph, cheap) to get it.
        let mut state = FlyState::new(&model);
        state.set_v(&model, &v_init);
        let mut input_buf = vec![0.0f32; encoder.num_inputs()];
        for obs in &observations {
            let an_values = compute_proprioception_values(&obs.self_state, encoder.ray_grid_config());
            let mut features = RayGridFeatures::new(encoder.ray_grid_config());
            features.compute(obs, encoder.ray_grid_config());
            encoder.forward(&features, &an_values, &encoder_params, &mut input_buf);
            state.step_decision(&model, &input_buf);
        }
        let r_last: Vec<f32> = state.v().iter().map(|&v| activation(v, model.config().r_max)).collect();
        crate::world_model::world_model_forward(&world_model, &r_last, &world_model_params)
    };
    let target_regression: [f32; crate::world_model::NUM_REGRESSION_TARGETS] =
        std::array::from_fn(|o| unperturbed_preds[0].regression[o] + 0.03 * (1 - 2 * (o % 2) as i32) as f32);

    let targets = vec![
        DecisionTargets::default(),
        DecisionTargets {
            decoder: crate::decoder::DecoderTargets {
                direction: Some(2),
                jump: Some(true),
                hook: Some(false),
                fire: Some(true),
                aim: Some(0.7),
            },
            world_model: Some(std::array::from_fn(|k| {
                if k == 0 {
                    crate::world_model::HorizonTargets {
                        regression: Some(target_regression),
                        binary: Some([true, false, true]),
                    }
                } else {
                    crate::world_model::HorizonTargets::default()
                }
            })),
        },
    ];

    let seq = BrainSequence {
        v_init,
        observations,
        targets,
    };

    Fixture {
        model,
        index,
        encoder,
        encoder_params,
        decoder,
        decoder_params,
        calib,
        world_model,
        world_model_params,
        seq,
    }
}

fn loss_with(fx: &Fixture, model: &FlyModel) -> f32 {
    brain_train_step(
        model,
        &fx.index,
        &fx.encoder,
        &fx.encoder_params,
        &fx.decoder,
        &fx.decoder_params,
        &fx.calib,
        &fx.world_model,
        &fx.world_model_params,
        &fx.seq,
    )
    .0
}

#[test]
fn total_loss_is_finite_and_gradients_have_the_right_shapes() {
    let fx = build_fixture();
    let (loss, grads, final_v) = brain_train_step(
        &fx.model,
        &fx.index,
        &fx.encoder,
        &fx.encoder_params,
        &fx.decoder,
        &fx.decoder_params,
        &fx.calib,
        &fx.world_model,
        &fx.world_model_params,
        &fx.seq,
    );
    assert!(loss.is_finite());
    assert_eq!(grads.encoder.g.len(), fx.encoder.num_params());
    assert_eq!(grads.fly.a.len(), fx.model.params().a.len());
    assert_eq!(
        grads.decoder.direction_lr_w.len(),
        fx.decoder_params.direction_lr_w.len()
    );
    assert_eq!(
        grads.world_model.horizons[0].w_reg.len(),
        fx.world_model_params.horizons[0].w_reg.len()
    );
    assert_eq!(final_v.len(), fx.model.num_neurons());
    // Decision 0 has no targets at all -> nothing should have been added from it (checked
    // indirectly: the world-model gradient for horizon 1/2, which decision 1 also leaves
    // targetless, must be exactly zero).
    assert!(grads.world_model.horizons[1].w_reg.iter().all(|&x| x == 0.0));
}

/// The core acceptance-criterion-6 check: perturb one parameter at a time in each of the four
/// trainable groups, recompute the *whole pipeline's* loss, and compare against the analytic
/// gradient `brain_train_step` returned for it in one pass.
#[test]
fn gradients_match_finite_differences_across_every_trainable_group() {
    let fx = build_fixture();
    let (_loss, analytic, _final_v) = brain_train_step(
        &fx.model,
        &fx.index,
        &fx.encoder,
        &fx.encoder_params,
        &fx.decoder,
        &fx.decoder_params,
        &fx.calib,
        &fx.world_model,
        &fx.world_model_params,
        &fx.seq,
    );

    let h = 1e-2f32;
    let close_enough = |a: f64, fd: f64| -> bool {
        let rel = (a - fd).abs() / fd.abs().max(1e-6);
        rel < 2e-3 || (a - fd).abs() < 3e-4
    };

    // --- encoder g/c ---
    for pid in 0..fx.encoder.num_params() {
        let mut plus = fx.encoder_params.clone();
        plus.g[pid] += h;
        let mut minus = fx.encoder_params.clone();
        minus.g[pid] -= h;
        let mut fx_plus = build_fixture();
        fx_plus.encoder_params = plus;
        let mut fx_minus = build_fixture();
        fx_minus.encoder_params = minus;
        let fd = (f64::from(loss_with(&fx_plus, &fx_plus.model)) - f64::from(loss_with(&fx_minus, &fx_minus.model)))
            / (2.0 * f64::from(h));
        assert!(
            close_enough(f64::from(analytic.encoder.g[pid]), fd),
            "encoder.g[{pid}]: analytic={} fd={fd}",
            analytic.encoder.g[pid]
        );
    }

    // --- fly a (requires rebuilding the model via set_params) ---
    for pid in 0..fx.model.params().a.len().min(6) {
        // sample a handful, not necessarily all, of a potentially larger `a` — still every entry
        // for this tiny fixture, since `a.len()` is small.
        let mut fx_plus = build_fixture();
        let mut p = fx_plus.model.params().clone();
        p.a[pid] += h;
        fx_plus.model.set_params(p).unwrap();
        let mut fx_minus = build_fixture();
        let mut p = fx_minus.model.params().clone();
        p.a[pid] -= h;
        fx_minus.model.set_params(p).unwrap();
        let fd = (f64::from(loss_with(&fx_plus, &fx_plus.model)) - f64::from(loss_with(&fx_minus, &fx_minus.model)))
            / (2.0 * f64::from(h));
        assert!(
            close_enough(f64::from(analytic.fly.a[pid]), fd),
            "fly.a[{pid}]: analytic={} fd={fd}",
            analytic.fly.a[pid]
        );
    }

    // --- decoder direction_lr_w ---
    for i in 0..fx.decoder_params.direction_lr_w.len() {
        let mut fx_plus = build_fixture();
        fx_plus.decoder_params.direction_lr_w[i] += h;
        let mut fx_minus = build_fixture();
        fx_minus.decoder_params.direction_lr_w[i] -= h;
        let fd = (f64::from(loss_with(&fx_plus, &fx_plus.model)) - f64::from(loss_with(&fx_minus, &fx_minus.model)))
            / (2.0 * f64::from(h));
        assert!(
            close_enough(f64::from(analytic.decoder.direction_lr_w[i]), fd),
            "decoder.direction_lr_w[{i}]: analytic={} fd={fd}",
            analytic.decoder.direction_lr_w[i]
        );
    }

    // --- world model horizon-0 w_reg ---
    for i in 0..fx.world_model_params.horizons[0].w_reg.len() {
        let mut fx_plus = build_fixture();
        fx_plus.world_model_params.horizons[0].w_reg[i] += h;
        let mut fx_minus = build_fixture();
        fx_minus.world_model_params.horizons[0].w_reg[i] -= h;
        let fd = (f64::from(loss_with(&fx_plus, &fx_plus.model)) - f64::from(loss_with(&fx_minus, &fx_minus.model)))
            / (2.0 * f64::from(h));
        assert!(
            close_enough(f64::from(analytic.world_model.horizons[0].w_reg[i]), fd),
            "world_model.horizons[0].w_reg[{i}]: analytic={} fd={fd}",
            analytic.world_model.horizons[0].w_reg[i]
        );
    }
}

/// Review lesson from task 7.2: a gradient check must be able to fail. Sabotaging the largest
/// encoder gradient must break the finite-difference comparison above.
#[test]
fn a_sabotaged_gradient_is_caught() {
    let fx = build_fixture();
    let (_loss, mut analytic, _final_v) = brain_train_step(
        &fx.model,
        &fx.index,
        &fx.encoder,
        &fx.encoder_params,
        &fx.decoder,
        &fx.decoder_params,
        &fx.calib,
        &fx.world_model,
        &fx.world_model_params,
        &fx.seq,
    );
    let biggest = (0..analytic.encoder.g.len())
        .max_by(|&a, &b| {
            analytic.encoder.g[a]
                .abs()
                .partial_cmp(&analytic.encoder.g[b].abs())
                .unwrap()
        })
        .unwrap();
    analytic.encoder.g[biggest] = 0.0;

    let h = 1e-2f32;
    let mut fx_plus = build_fixture();
    fx_plus.encoder_params.g[biggest] += h;
    let mut fx_minus = build_fixture();
    fx_minus.encoder_params.g[biggest] -= h;
    let fd = (f64::from(loss_with(&fx_plus, &fx_plus.model)) - f64::from(loss_with(&fx_minus, &fx_minus.model)))
        / (2.0 * f64::from(h));
    // Threshold lowered from an earlier `1e-4` after fixing review round 1's F5 (world-model
    // `loss_weight` is now actually applied at its configured `0.2`, not the accidental `1.0` an
    // earlier revision effectively used): this fixture's encoder gradient is dominated by the
    // world-model coupling path, so correctly *down*-weighting it also correctly shrinks the true
    // gradient this sabotage test is trying to catch -- `1e-5` still comfortably separates a real,
    // correctly-computed gradient from noise (this fixture's actual value is `~8.7e-5`).
    assert!(
        fd.abs() > 1e-5,
        "finite difference must be meaningfully nonzero, got {fd}"
    );
    assert_ne!(analytic.encoder.g[biggest] as f64, fd);
}

/// Review round 1, F5 (major, CONFIRMED): `WorldModelConfig::loss_weight` was computed but never
/// actually multiplied into the loss/gradients `brain_train_step` returns. Checks the weight is
/// applied uniformly: `weight = 0` must drop the world-model contribution entirely (matching a
/// run with `world_model: None` on every decision), and `weight = w` must scale the world-model
/// loss/every world-model gradient array/`grad_r`'s effect on `fly`/`encoder` by exactly `w`
/// relative to `weight = 1`.
#[test]
fn world_model_loss_weight_is_actually_applied() {
    let fx_w1 = build_fixture_with_wm_weight(1.0);
    let (loss_w1, grads_w1, _) = brain_train_step(
        &fx_w1.model,
        &fx_w1.index,
        &fx_w1.encoder,
        &fx_w1.encoder_params,
        &fx_w1.decoder,
        &fx_w1.decoder_params,
        &fx_w1.calib,
        &fx_w1.world_model,
        &fx_w1.world_model_params,
        &fx_w1.seq,
    );

    let fx_w0 = build_fixture_with_wm_weight(0.0);
    let (loss_w0, grads_w0, _) = brain_train_step(
        &fx_w0.model,
        &fx_w0.index,
        &fx_w0.encoder,
        &fx_w0.encoder_params,
        &fx_w0.decoder,
        &fx_w0.decoder_params,
        &fx_w0.calib,
        &fx_w0.world_model,
        &fx_w0.world_model_params,
        &fx_w0.seq,
    );

    // `weight = 0` must produce a *strictly smaller* loss than `weight = 1` (the world-model term
    // is nonzero in this fixture -- decision 1's target regression is offset from the
    // unperturbed prediction on purpose, see `build_fixture`) and must leave the world-model
    // gradients themselves at exactly zero.
    assert!(
        loss_w0 < loss_w1 - 1e-6,
        "loss at weight=0 ({loss_w0}) should be strictly less than at weight=1 ({loss_w1})"
    );
    assert!(grads_w0.world_model.horizons[0].w_reg.iter().all(|&x| x == 0.0));

    // `weight = w` must scale the world-model gradient arrays by exactly `w` relative to
    // `weight = 1` (the fly/encoder gradients differ too, since `grad_r`'s effect on them also
    // scales -- checked separately below via the *fly* gradient, which is easiest to reason
    // about in isolation since encoder gradients mix contributions from both the decoder and the
    // world-model loss).
    let w = 0.37f32;
    let fx_w = build_fixture_with_wm_weight(w);
    let (loss_w, grads_w, _) = brain_train_step(
        &fx_w.model,
        &fx_w.index,
        &fx_w.encoder,
        &fx_w.encoder_params,
        &fx_w.decoder,
        &fx_w.decoder_params,
        &fx_w.calib,
        &fx_w.world_model,
        &fx_w.world_model_params,
        &fx_w.seq,
    );
    // Both runs share the exact same non-world-model loss (decoder targets/params/features are
    // identical across `build_fixture_with_wm_weight` calls) -- only the world-model term differs
    // -- so `loss_w - (loss_w1 - world_model_loss_at_weight_1) == w * world_model_loss_at_weight_1`
    // reduces to checking the *world-model-only* delta scales linearly: `(loss_w - loss_w0) / (loss_w1 - loss_w0) == w`.
    let fraction = (loss_w - loss_w0) / (loss_w1 - loss_w0);
    assert!(
        (fraction - w).abs() < 1e-3,
        "world-model loss did not scale linearly with weight: fraction={fraction} want={w}"
    );

    for i in 0..grads_w.world_model.horizons[0].w_reg.len() {
        let expected = w * grads_w1.world_model.horizons[0].w_reg[i];
        let actual = grads_w.world_model.horizons[0].w_reg[i];
        assert!(
            (actual - expected).abs() < 1e-4,
            "world_model.horizons[0].w_reg[{i}]: actual={actual} expected(w*grad@weight=1)={expected}"
        );
    }
}
