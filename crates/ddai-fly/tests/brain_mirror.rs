//! Acceptance criterion 8 (real S graph): flipping an `Observation` in X must give the mirrored
//! activity pattern, up to float noise, because every type's L/R copies share parameters (FLY.md
//! §4). Tested at the **encoder + connectome** level (per-`(type, side)` mean rate — the
//! granularity parameters are actually shared at: a type's L instances and R instances share one
//! `g`/`c`/`a`/`b`/`theta`, not any specific neuron-to-neuron pairing), which is the layer this
//! task's own code (the encoder) newly introduces; task 7.1/7.2's own `tests/correctness.rs`
//! already covers the connectome-only half of this (a hand-built graph, synthetic per-neuron
//! input vectors) — this file is the encoder's half, on the real graph, with real game
//! `Observation`s. `#[ignore]`d: needs `~/aiddnet/data/connectome/compiled/fly-S-v1.flyg` and
//! `configs/fly/S-brain.toml`.
//!
//! **The decoder** (review round 1, F6, CONFIRMED): an earlier revision's `DecoderParams` had no
//! *architectural* constraint tying a "left" DN group's weights to the corresponding "right" one,
//! so exact end-to-end mirror symmetry of the decoded action was an open question. Fixed by tying
//! every decoder head by construction from the real `.flyg`'s `output_groups` (see
//! `ddai_fly::decoder`'s module doc comment) — [`action_level_mirror_symmetry_on_a_synthetic_l_r_
//! symmetric_graph`] below is the action-level check this enabled: a small, fully hand-built,
//! exactly L/R-symmetric graph (encoder -> connectome -> decoder end to end, not the decoder
//! alone — `crate::decoder::tests::mirroring_z_gives_the_mirrored_decoded_action` already checks
//! the decoder's own tying math in isolation from synthetic `z`), where a mirrored observation
//! must decode to the mirrored action to within float noise.

use std::sync::Arc;

use ddai_brain::{CharacterObservation, Observation};
use ddai_fly::activation::activation;
use ddai_fly::brain_fixtures::{FxEdge, FxInputChannel, FxNeuron, FxOutputGroup, FxType, build_brain_flyg};
use ddai_fly::config::FlyConfig;
use ddai_fly::decoder::{DecoderConfig, DecoderModel, decoder_forward};
use ddai_fly::encoder::{
    EncoderModel, EncoderParams, ProprioceptionConfig, RayGridConfig, RayGridFeatures, compute_proprioception_values,
};
use ddai_fly::model::FlyModel;
use ddai_fly::params::FlyParams;
use ddai_fly::state::FlyState;
use ddai_fly::world_model::{WorldModelConfig, WorldModelHead, world_model_forward};
use ddai_flyg::{NeuronRole, Side, Sign};

fn home() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("HOME").expect("HOME must be set"))
}

fn load_real_s() -> ddai_flyg::Flyg {
    let path = home().join("aiddnet/data/connectome/compiled/fly-S-v1.flyg");
    ddai_flyg::load(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()))
}

fn load_s_brain_config() -> ddai_fly::brain_config::BrainConfig {
    // The workspace root: this test binary's `CARGO_MANIFEST_DIR` is `crates/ddai-fly`.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/fly/S-brain.toml");
    ddai_fly::brain_config::load_brain_config(&path).expect("configs/fly/S-brain.toml should parse")
}

fn synthetic_map() -> ddai_physics::map::MapData {
    // Open room, symmetric enough to not itself be the source of any asymmetry in this test
    // (mirroring the map is exact regardless — see `ddai_brain::mirror_map_data` — but an open
    // room keeps every ray's feature purely a function of the characters, easier to reason
    // about when interpreting a deviation).
    ddai_physics::map::MapData {
        width: 40,
        height: 40,
        game: vec![Default::default(); 1600],
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }
}

#[test]
#[ignore]
fn mirroring_the_observation_mirrors_per_type_per_side_activity_on_the_real_s_graph() {
    let flyg = load_real_s();
    let brain_cfg = load_s_brain_config();
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 7);
    let model = FlyModel::new(flyg, config, params).expect("build FlyModel");
    let encoder = EncoderModel::new(&model, brain_cfg.ray_grid, &brain_cfg.proprioception).expect("build EncoderModel");
    let encoder_params = EncoderParams::init_default(encoder.num_params());

    let mut me = CharacterObservation::at_rest(0);
    me.pos = ddai_physics::vmath::Vec2::new(640.0, 640.0);
    me.vel = ddai_physics::vmath::Vec2::new(120.0, -40.0);
    let mut opp = CharacterObservation::at_rest(1);
    opp.pos = ddai_physics::vmath::Vec2::new(900.0, 560.0);
    opp.vel = ddai_physics::vmath::Vec2::new(-30.0, 10.0);
    let obs = Observation {
        map: Arc::new(synthetic_map()),
        tick: 0,
        self_state: me,
        others: vec![opp],
        target_id: None,
        tuning: ddai_physics::tuning::TuningParams::default(),
    };
    let mirrored_obs = obs.mirror_x();

    // Warm up ONE state to a converged rest, then clone it for both branches — both must start
    // from *exactly* the same (already L/R-symmetric-by-construction, since it came from an
    // all-zero, side-blind initial `V` and zero input) resting state.
    let mut warm = FlyState::new(&model);
    let report = warm.warm_up(&model);
    assert!(report.converged, "warm-up must converge for this test to mean anything");

    let mut input_buf = vec![0.0f32; encoder.num_inputs()];
    let mut ray_features = ddai_fly::encoder::RayGridFeatures::new(encoder.ray_grid_config());

    let mut decide_once = |obs: &Observation| -> Vec<f32> {
        let mut state = warm.clone();
        let an_values = compute_proprioception_values(&obs.self_state, encoder.ray_grid_config());
        ray_features.compute(obs, encoder.ray_grid_config());
        encoder.forward(&ray_features, &an_values, &encoder_params, &mut input_buf);
        let _ = state.step_decision(&model, &input_buf);
        state.v().iter().map(|&v| activation(v, config.r_max)).collect()
    };

    let r_original = decide_once(&obs);
    let r_mirrored = decide_once(&mirrored_obs);

    // Per-(type, side) mean rate, for every type that has neurons on both L and R.
    let flyg = model.flyg();
    let mut by_type_side: std::collections::BTreeMap<(u32, Side), (f32, u32)> = std::collections::BTreeMap::new();
    for (i, n) in flyg.neurons.iter().enumerate() {
        if matches!(n.side, Side::L | Side::R) {
            let entry = by_type_side.entry((n.type_index, n.side)).or_insert((0.0, 0));
            entry.0 += r_original[i];
            entry.1 += 1;
        }
    }
    let mut by_type_side_mirrored: std::collections::BTreeMap<(u32, Side), (f32, u32)> =
        std::collections::BTreeMap::new();
    for (i, n) in flyg.neurons.iter().enumerate() {
        if matches!(n.side, Side::L | Side::R) {
            let entry = by_type_side_mirrored.entry((n.type_index, n.side)).or_insert((0.0, 0));
            entry.0 += r_mirrored[i];
            entry.1 += 1;
        }
    }

    let opposite = |s: Side| match s {
        Side::L => Side::R,
        Side::R => Side::L,
        other => other,
    };

    let mut max_deviation = 0.0f32;
    let mut compared = 0usize;
    for (&(type_index, side), &(sum, count)) in &by_type_side {
        let mean_original = sum / count as f32;
        if let Some(&(sum_m, count_m)) = by_type_side_mirrored.get(&(type_index, opposite(side))) {
            let mean_mirrored_opposite = sum_m / count_m as f32;
            let deviation = (mean_original - mean_mirrored_opposite).abs();
            max_deviation = max_deviation.max(deviation);
            compared += 1;
        }
    }

    eprintln!(
        "mirror symmetry (real S, {compared} type/side pairs compared): max |mean_rate(type,side,obs) - mean_rate(type,opposite_side,mirror(obs))| = {max_deviation:.6} (r_max={})",
        config.r_max
    );
    assert!(
        compared > 50,
        "expected many L/R type pairs on the real S graph, got {compared}"
    );
    assert!(
        max_deviation < 0.15,
        "mirror symmetry deviation too large: {max_deviation} (r_max={})",
        config.r_max
    );

    // Also check role-only aggregates (VPN/AN/DN) as a coarser, sanity-check summary.
    let role_of = |ti: u32| -> Option<NeuronRole> { flyg.neurons.iter().find(|n| n.type_index == ti).map(|n| n.role) };
    let mut worst_by_role: std::collections::BTreeMap<&'static str, f32> = std::collections::BTreeMap::new();
    for (&(type_index, side), &(sum, count)) in &by_type_side {
        let mean_original = sum / count as f32;
        if let Some(&(sum_m, count_m)) = by_type_side_mirrored.get(&(type_index, opposite(side))) {
            let mean_mirrored_opposite = sum_m / count_m as f32;
            let dev = (mean_original - mean_mirrored_opposite).abs();
            let role_name = match role_of(type_index) {
                Some(NeuronRole::InputVisual) => "VPN",
                Some(NeuronRole::InputAscending) => "AN",
                Some(NeuronRole::Hidden) => "Hidden",
                Some(NeuronRole::Output) => "DN",
                None => "?",
            };
            let e = worst_by_role.entry(role_name).or_insert(0.0);
            *e = e.max(dev);
        }
    }
    eprintln!("worst deviation by role: {worst_by_role:?}");
}

// --- Action-level mirror symmetry, synthetic graph (review round 1, F6) -------------------------

/// Same shape as `tests/no_alloc_brain.rs`'s `tiny_brain_flyg`: 2 VPN types (opponent/wall, `L`/`R`
/// receptive fields mirrored: `-45/+45`, `-30/+30`), 1 AN type (`M`), 2 hidden neurons (`M`), and
/// one output neuron per action -- `direction_left`/`direction_right` tied on a shared `DN_LR`
/// type (review round 1, F6), the rest single/pooled -- fully connected input -> hidden -> output
/// so a real signal reaches every decoder head.
fn symmetric_action_flyg() -> ddai_flyg::Flyg {
    let type_names = [
        "VPN_OPP",
        "VPN_WALL",
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
    let mut neurons = vec![
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
            role: NeuronRole::InputVisual,
            side: Side::L,
            full_connectome_in: 1000,
            rf: (-30.0, 0.0),
        },
        FxNeuron {
            type_index: 1,
            role: NeuronRole::InputVisual,
            side: Side::R,
            full_connectome_in: 1000,
            rf: (30.0, 0.0),
        },
        FxNeuron {
            type_index: 2,
            role: NeuronRole::InputAscending,
            side: Side::M,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        },
    ];
    for _ in 0..2 {
        neurons.push(FxNeuron {
            type_index: 3,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        });
    }
    // DN_LR: one tied L/R pair (review round 1, F6).
    neurons.push(FxNeuron {
        type_index: 4,
        role: NeuronRole::Output,
        side: Side::L,
        full_connectome_in: 1000,
        rf: (0.0, 0.0),
    });
    neurons.push(FxNeuron {
        type_index: 4,
        role: NeuronRole::Output,
        side: Side::R,
        full_connectome_in: 1000,
        rf: (0.0, 0.0),
    });
    for ti in 5..type_names.len() {
        neurons.push(FxNeuron {
            type_index: ti as u32,
            role: NeuronRole::Output,
            side: Side::M,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        });
    }

    let input_indices: Vec<u32> = (0..5).collect();
    let hidden_indices: Vec<u32> = (5..7).collect();
    let output_indices: Vec<u32> = (7..14).collect();
    let mut edges = Vec::new();
    for &pre in &input_indices {
        for &post in &hidden_indices {
            edges.push(FxEdge {
                pre,
                post,
                synapse_count: 5,
            });
        }
    }
    for &pre in &hidden_indices {
        for &post in &output_indices {
            edges.push(FxEdge {
                pre,
                post,
                synapse_count: 5,
            });
        }
    }

    let input_channels = vec![
        FxInputChannel {
            type_name: "VPN_OPP",
            channels: vec!["opponent_position"],
        },
        FxInputChannel {
            type_name: "VPN_WALL",
            channels: vec!["walls"],
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

/// Acceptance criterion 8, action level (review round 1, F6): on a small, fully hand-built,
/// exactly `L`/`R`-symmetric graph, decoding a mirrored [`Observation`] end to end (encoder ->
/// connectome -> decoder, not the decoder alone -- `crate::decoder::tests::mirroring_z_gives_
/// the_mirrored_decoded_action` already checks the decoder's own tying math from synthetic `z`)
/// must give the mirrored [`DecodedAction`] to within float noise. Non-zero, asymmetric decoder
/// weights on purpose (not `init_default_params`'s all-zero `W`) -- an all-zero decoder would pass
/// trivially (every logit is `0` regardless of any real encoder/connectome asymmetry), hiding
/// exactly the class of bug this test exists to catch.
#[test]
fn action_level_mirror_symmetry_on_a_synthetic_l_r_symmetric_graph() {
    let flyg = symmetric_action_flyg();
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 5);
    let model = FlyModel::new(flyg, config, params).expect("build FlyModel");
    let encoder = EncoderModel::new(
        &model,
        RayGridConfig::default(),
        &ProprioceptionConfig {
            grounded: vec!["AN_GROUND".to_string()],
            ..ProprioceptionConfig::default()
        },
    )
    .expect("build EncoderModel");
    let encoder_params = EncoderParams::init_default(encoder.num_params());

    let decoder = DecoderModel::new(&model, DecoderConfig::default()).expect("build DecoderModel");
    let mut decoder_params = decoder.init_default_params();
    decoder_params.direction_lr_w = vec![0.8; decoder_params.direction_lr_w.len()];
    decoder_params.direction_lr_b = 0.1;
    decoder_params.direction_stop_w = vec![0.6; decoder_params.direction_stop_w.len()];
    decoder_params.jump_w = vec![0.5; decoder_params.jump_w.len()];
    decoder_params.hook_w = vec![-0.4; decoder_params.hook_w.len()];
    decoder_params.fire_w = vec![0.3; decoder_params.fire_w.len()];
    // Review round 2, F19 (CONFIRMED): through the *real* `calibrate_from_rest` protocol, not a
    // hand-set `mu=0, sigma=1` -- an earlier revision's mirror test used the latter, so it never
    // actually exercised whether the production calibration itself preserves mirror symmetry (it
    // didn't: independent per-neuron jitter gave `L`/`R` DN homologs slightly different `mu`,
    // which the `sigma` floor amplified into a real action-level asymmetry -- see
    // `calibrate_from_rest`'s own doc comment for the fix and the measured before/after numbers).
    let calib = ddai_fly::calibrate_from_rest(&model, 5, decoder.config().min_sigma).expect("calibration must fit");

    let mut me = CharacterObservation::at_rest(0);
    me.pos = ddai_physics::vmath::Vec2::new(500.0, 640.0);
    me.vel = ddai_physics::vmath::Vec2::new(15.0, -3.0);
    let mut opp = CharacterObservation::at_rest(1);
    opp.pos = ddai_physics::vmath::Vec2::new(820.0, 560.0);
    opp.vel = ddai_physics::vmath::Vec2::new(-4.0, 2.0);
    let obs = Observation {
        map: Arc::new(synthetic_map()),
        tick: 0,
        self_state: me,
        others: vec![opp],
        target_id: None,
        tuning: ddai_physics::tuning::TuningParams::default(),
    };
    let mirrored_obs = obs.mirror_x();

    let mut warm = FlyState::new(&model);
    let report = warm.warm_up(&model);
    assert!(report.converged, "warm-up must converge for this test to mean anything");

    let decide_once = |obs: &Observation| -> ddai_fly::decoder::DecodedAction {
        let mut state = warm.clone();
        let an = compute_proprioception_values(&obs.self_state, encoder.ray_grid_config());
        let mut features = RayGridFeatures::new(encoder.ray_grid_config());
        features.compute(obs, encoder.ray_grid_config());
        let mut input_buf = vec![0.0f32; encoder.num_inputs()];
        encoder.forward(&features, &an, &encoder_params, &mut input_buf);
        let out = state.step_decision(&model, &input_buf);
        decoder_forward(&decoder, out.dn_rates, &calib, &decoder_params)
    };

    let action = decide_once(&obs);
    let mirrored_action = decide_once(&mirrored_obs);

    let dev_left_right = (action.direction_probs[0] - mirrored_action.direction_probs[2]).abs();
    let dev_right_left = (action.direction_probs[2] - mirrored_action.direction_probs[0]).abs();
    let dev_stop = (action.direction_probs[1] - mirrored_action.direction_probs[1]).abs();
    let dev_jump = (action.jump_prob - mirrored_action.jump_prob).abs();
    let dev_hook = (action.hook_prob - mirrored_action.hook_prob).abs();
    let dev_fire = (action.fire_prob - mirrored_action.fire_prob).abs();
    let max_dev = [dev_left_right, dev_right_left, dev_stop, dev_jump, dev_hook, dev_fire]
        .into_iter()
        .fold(0.0f32, f32::max);
    eprintln!(
        "action-level mirror symmetry (synthetic graph): left/right={dev_left_right:.2e} \
         right/left={dev_right_left:.2e} stop={dev_stop:.2e} jump={dev_jump:.2e} hook={dev_hook:.2e} \
         fire={dev_fire:.2e} (max={max_dev:.2e})"
    );
    assert!(
        max_dev < 1e-4,
        "action-level mirror symmetry deviation too large: {max_dev} \
         (left/right={dev_left_right} right/left={dev_right_left} stop={dev_stop} jump={dev_jump} \
         hook={dev_hook} fire={dev_fire})"
    );

    // The world-model head's regression readout (review round 1, F6: "also check"): unlike the
    // decoder's now-tied heads, `WorldModelHead`'s linear readout has no architectural L/R
    // structure at all -- it is a single dense matrix over the hidden-neuron subset, with no
    // notion of which of its `NUM_REGRESSION_TARGETS` outputs is an "x-like" component that a
    // mirrored observation ought to negate, and no mechanism tying a "flip" in the input to any
    // particular change in the output. That's a property of the *learned* weights (task 7.3
    // acceptance criterion 5 doesn't ask for mirror symmetry here, only acceptance criterion 8's
    // *action* does), not something this module can guarantee the way the decoder's tied heads now
    // do. The two values below happen to come out **equal** for this specific fixture -- not
    // because anything enforces "mirroring leaves regression unchanged", but because this
    // fixture's whole `Hidden`-role population is side-`M` and fed identically from both `L` and
    // `R` VPN inputs (the same reason pooled decoder heads are side-invariant), so it never sees
    // the L/R swap at all. A graph whose world-model subset included side-specific hidden neurons
    // would show some *other* unenforced value instead -- the point is that no relationship (equal,
    // negated, or anything else) is architecturally guaranteed either way, only whatever training
    // data happens to teach the weights.
    let world_model = WorldModelHead::new(&model, &WorldModelConfig::default()).expect("build WorldModelHead");
    let mut wm_params = world_model.init_default_params();
    for (i, w) in wm_params.horizons[0].w_reg.iter_mut().enumerate() {
        *w = 0.1 * (i as f32 - 3.0);
    }
    let decide_r_full = |obs: &Observation| -> Vec<f32> {
        let mut state = warm.clone();
        let an = compute_proprioception_values(&obs.self_state, encoder.ray_grid_config());
        let mut features = RayGridFeatures::new(encoder.ray_grid_config());
        features.compute(obs, encoder.ray_grid_config());
        let mut input_buf = vec![0.0f32; encoder.num_inputs()];
        encoder.forward(&features, &an, &encoder_params, &mut input_buf);
        let _ = state.step_decision(&model, &input_buf);
        state.v().iter().map(|&v| activation(v, config.r_max)).collect()
    };
    let r_full = decide_r_full(&obs);
    let r_full_mirrored = decide_r_full(&mirrored_obs);
    let pred = world_model_forward(&world_model, &r_full, &wm_params);
    let pred_mirrored = world_model_forward(&world_model, &r_full_mirrored, &wm_params);
    eprintln!(
        "world-model horizon-0 regression[0] (an arbitrary 'x-like' slot, no enforced meaning): \
         original={:.4} mirrored={:.4} -- equal here only because this fixture's hidden subset is \
         side-blind, not because of any architectural guarantee (see the comment above)",
        pred[0].regression[0], pred_mirrored[0].regression[0]
    );
}
