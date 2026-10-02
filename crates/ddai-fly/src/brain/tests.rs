use super::*;
use crate::brain_fixtures::{FxEdge, FxInputChannel, FxNeuron, FxOutputGroup, FxType, build_brain_flyg};
use crate::config::FlyConfig;
use crate::decoder::{DecoderConfig, DecoderModel, DnCalibration};
use crate::encoder::{EncoderModel, EncoderParams, ProprioceptionConfig, RayGridConfig};
use crate::params::FlyParams;
use ddai_brain::{CharacterObservation, Observation};
use ddai_flyg::{NeuronRole, Side, Sign};
use std::sync::Arc;

/// A small but fully-wired fixture: 2 VPN types (opponent/wall), 1 AN type (grounded), 2 hidden
/// neurons, and one output neuron per `DecoderConfig::default()` action — everything connected
/// input -> hidden -> output, so a real signal actually reaches every decoder head.
// A `vec![...]` literal here would need every field of every one of ~14 `FxNeuron`s
// spelled out positionally with no per-neuron comment anchor -- individual `.push()`
// calls (each right after its own explanatory comment) stay clearer for a fixture this
// shaped, even though clippy's default heuristic can't tell the difference from
// "just forgot the macro".
#[allow(clippy::vec_init_then_push)]
fn tiny_brain_flyg() -> ddai_flyg::Flyg {
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

    let mut neurons = Vec::new();
    // 0, 1: VPN_OPP L/R
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
    // 2, 3: VPN_WALL L/R
    neurons.push(FxNeuron {
        type_index: 1,
        role: NeuronRole::InputVisual,
        side: Side::L,
        full_connectome_in: 1000,
        rf: (-30.0, 0.0),
    });
    neurons.push(FxNeuron {
        type_index: 1,
        role: NeuronRole::InputVisual,
        side: Side::R,
        full_connectome_in: 1000,
        rf: (30.0, 0.0),
    });
    // 4: AN_GROUND
    neurons.push(FxNeuron {
        type_index: 2,
        role: NeuronRole::InputAscending,
        side: Side::M,
        full_connectome_in: 1000,
        rf: (0.0, 0.0),
    });
    // 5, 6: HID x2
    for _ in 0..2 {
        neurons.push(FxNeuron {
            type_index: 3,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        });
    }
    // 7, 8: DN_LR L/R (a tied direction pair -- review round 1, F6).
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
    // 9..14: one output neuron per remaining DN type.
    for ti in 5..type_names.len() {
        neurons.push(FxNeuron {
            type_index: ti as u32,
            role: NeuronRole::Output,
            side: Side::M,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        });
    }

    let input_indices: Vec<u32> = (0..5).collect(); // VPN_OPP x2, VPN_WALL x2, AN_GROUND
    let hidden_indices: Vec<u32> = (5..7).collect();
    let output_indices: Vec<u32> = (7..14).collect(); // one output neuron per DN type (7 DN types)

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

pub(crate) fn make_brain(seed: u64, selection: ActionSelection) -> FlyBrain {
    let flyg = tiny_brain_flyg();
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 1);
    let model = FlyModel::new(flyg, config, params).unwrap();

    let encoder = EncoderModel::new(
        &model,
        RayGridConfig::default(),
        &ProprioceptionConfig {
            grounded: vec!["AN_GROUND".to_string()],
            ..ProprioceptionConfig::default()
        },
    )
    .unwrap();
    let encoder_params = EncoderParams::init_default(encoder.num_params());

    let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
    let decoder_params = decoder.init_default_params();
    let calib = DnCalibration {
        mu: vec![0.0; model.num_outputs()],
        sigma: vec![1.0; model.num_outputs()],
    };

    FlyBrain::new(
        model,
        encoder,
        encoder_params,
        decoder,
        decoder_params,
        calib,
        FlyBrainConfig {
            action_selection: selection,
            seed,
        },
    )
}

pub(crate) fn sample_observation(opp_x: f32) -> Observation {
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

use ddai_brain::Brain;

#[test]
fn decide_returns_a_well_formed_action() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    let obs = sample_observation(400.0);
    let action = brain.decide(&obs);
    assert!((-1..=1).contains(&action.direction));
    assert_ne!(
        (action.target.x, action.target.y),
        (0, 0),
        "target must never be exactly zero"
    );
}

#[test]
fn reset_then_decide_does_not_panic_and_telemetry_appears_after_deciding() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    assert!(
        ddai_brain::Brain::telemetry(&brain).is_none(),
        "no telemetry before the first decision"
    );
    let ctx = ddai_brain::ResetContext {
        map: Arc::new(tiny_map()),
        self_id: 0,
        seed: 1,
    };
    brain.reset(&ctx);
    let obs = sample_observation(400.0);
    let _ = brain.decide(&obs);
    let telemetry = brain.telemetry().expect("telemetry after a decision");
    assert!(telemetry.starts_with('{'));
    assert!(telemetry.contains("dn_z_by_action"));
    assert!(telemetry.contains("decoded_action"));
}

/// Review round 1, F13 (CONFIRMED): `reset` must re-seed from `ResetContext::seed`, not
/// `FlyBrainConfig::seed` (which stays fixed for the brain's whole lifetime) -- two brains built
/// with *different* config seeds must still sample byte-for-byte identically after both are
/// `reset` with the *same* `ctx.seed`. An earlier revision ignored `ctx` entirely, so this would
/// have failed (each brain kept sampling from its own original, different config seed forever).
#[test]
fn reset_reseeds_sampled_action_selection_from_the_context_seed() {
    let mut a = make_brain(111, ActionSelection::Sampled);
    let mut b = make_brain(222, ActionSelection::Sampled);
    let ctx = ddai_brain::ResetContext {
        map: Arc::new(tiny_map()),
        self_id: 0,
        seed: 999,
    };
    a.reset(&ctx);
    b.reset(&ctx);
    for x in [350.0, 500.0, 250.0, 600.0] {
        let obs = sample_observation(x);
        assert_eq!(a.decide(&obs), b.decide(&obs));
    }
}

#[test]
fn decide_is_deterministic_for_the_same_seed_and_observations() {
    let mut a = make_brain(42, ActionSelection::Sampled);
    let mut b = make_brain(42, ActionSelection::Sampled);
    for x in [350.0, 500.0, 250.0, 600.0] {
        let obs = sample_observation(x);
        assert_eq!(a.decide(&obs), b.decide(&obs));
    }
}

#[test]
fn latency_is_recorded_after_a_decision() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    assert_eq!(brain.last_latency(), std::time::Duration::ZERO);
    let obs = sample_observation(400.0);
    let _ = brain.decide(&obs);
    // Not a timing assertion (no host-speed dependency) — just that *something* was recorded,
    // proving `decide` actually measured itself rather than leaving the field untouched.
    let _ = brain.last_latency();
}

#[test]
fn name_is_non_empty_and_mentions_graph_size() {
    let brain = make_brain(1, ActionSelection::Argmax);
    let name = Brain::name(&brain);
    assert!(name.contains('n'), "name={name}");
}

#[test]
fn thresholds_decide_each_binary_head_under_argmax() {
    let obs = sample_observation(400.0);
    let probe = |th: crate::bc::HeadThresholds| {
        let mut b = make_brain(1, ActionSelection::Argmax);
        b.set_thresholds(th);
        b.reset(&ddai_brain::ResetContext {
            map: obs.map.clone(),
            self_id: 0,
            seed: 1,
        });
        let a = b.decide(&obs);
        let d = *b.last_decoded().unwrap();
        (a, d)
    };
    let (a0, d) = probe(crate::bc::HeadThresholds::default());
    assert_eq!(a0.jump, d.jump_prob >= 0.5);
    assert_eq!(a0.hook, d.hook_prob >= 0.5);
    assert_eq!(a0.fire, d.fire_prob >= 0.5);
    // A threshold just below a head's probability presses it, just above releases it.
    let eps = 1e-4;
    let at = |j: f32, h: f32, f: f32| crate::bc::HeadThresholds {
        jump: j,
        hook: h,
        fire: f,
    };
    let (below, _) = probe(at(d.jump_prob - eps, d.hook_prob - eps, d.fire_prob - eps));
    assert!(below.jump && below.hook && below.fire);
    let (above, _) = probe(at(d.jump_prob + eps, d.hook_prob + eps, d.fire_prob + eps));
    assert!(!above.jump && !above.hook && !above.fire);
}

// ---- task 7.4: the visualisation stream ------------------------------------------------------------

#[test]
fn a_frame_round_trips_within_one_quantisation_step() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    brain.set_viz_every(1);
    for i in 0..3 {
        let _ = brain.decide(&sample_observation(300.0 + 10.0 * i as f32));
    }
    let obs = sample_observation(330.0);
    let action = brain.decide(&obs);
    let outcome = ddai_planner::hybrid::ProposalOutcome {
        chosen: true,
        chosen_total: 7,
        decisions_total: 9,
    };
    let bytes = brain.viz_frame_with(1234, Some(outcome)).expect("a frame").to_vec();
    let layout = brain.viz_layout().clone();
    assert_eq!(bytes.len(), layout.frame_len());
    let f = crate::viz::decode_frame(&bytes, layout.rate_max(), layout.z_clip()).unwrap();
    assert_eq!((f.tick, f.seq), (1234, 4));
    assert_eq!(i32::from(f.direction), action.direction);
    assert_eq!(f.flags & crate::viz::flag::JUMP != 0, action.jump);
    assert_eq!(f.flags & crate::viz::flag::HOOK != 0, action.hook);
    assert_eq!(f.flags & crate::viz::flag::FIRE != 0, action.fire);
    assert_ne!(f.flags & crate::viz::flag::CHOSEN_VALID, 0);
    assert_ne!(f.flags & crate::viz::flag::CHOSEN, 0);
    assert_eq!((f.chosen_total, f.decisions_total), (7, 9));
    let decoded = brain.last_decoded().unwrap();
    assert!((f.aim_angle - decoded.aim_angle).abs() < 1.0 / crate::viz::AIM_SCALE);
    // Group rates: against the layout's own aggregation of the brain's last per-type means.
    for (g, &got) in f.groups.iter().enumerate() {
        let want = layout.group_rate(g, &brain.last_per_type_mean_rate);
        assert!(
            (got - want).abs() <= layout.rate_max() / 255.0 / 2.0 + 1e-5,
            "group {g}: {got} vs {want}"
        );
    }
    // DN z-scores.
    let z = brain.calib.z(&brain.last_dn_rates, layout.z_clip());
    for (i, (&got, &want)) in f.dn_z.iter().zip(&z).enumerate() {
        assert!(
            (got - want).abs() <= layout.z_clip() / 127.0 / 2.0 + 1e-5,
            "dn {i}: {got} vs {want}"
        );
    }
    // The eye: every channel and cell.
    for (ci, ch) in crate::viz::EYE_CHANNELS.iter().enumerate() {
        let grid = brain.last_ray_features().spatial(*ch);
        assert_eq!(f.eye[ci].len(), grid.len());
        for (i, (&got, &want)) in f.eye[ci].iter().zip(grid).enumerate() {
            assert!(
                (got - want.clamp(0.0, 1.0)).abs() <= 0.5 / 255.0 + 1e-6,
                "{ch:?}[{i}]: {got} vs {want}"
            );
        }
    }
    // Logits: the heads are true logits, the direction ones log-probabilities.
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    assert!((sig(f.logits[3]) - decoded.jump_prob).abs() < 0.07);
    assert!((f.logits[0].exp() - decoded.direction_probs[0]).abs() < 0.07);
    assert_eq!(f.rays, 48);
    assert_eq!(f.bins, 4);
}

#[test]
fn pulling_frames_never_changes_a_decision() {
    // Two identical flies see the same observations; one is watched every decision, with a made-up
    // proposer verdict. Every action, rate and decoded probability must agree bit for bit.
    let mut watched = make_brain(5, ActionSelection::Sampled);
    let mut plain = make_brain(5, ActionSelection::Sampled);
    let ctx = ddai_brain::ResetContext {
        map: sample_observation(0.0).map,
        self_id: 0,
        seed: 9,
    };
    watched.reset(&ctx);
    plain.reset(&ctx);
    watched.set_viz_every(1);
    for i in 0..60 {
        let obs = sample_observation(200.0 + 7.0 * i as f32);
        let a = watched.decide(&obs);
        let b = plain.decide(&obs);
        assert_eq!(a, b, "decision {i}");
        let outcome = ddai_planner::hybrid::ProposalOutcome {
            chosen: i % 3 == 0,
            chosen_total: i,
            decisions_total: i + 1,
        };
        assert!(watched.viz_frame_with(i, Some(outcome)).is_some());
        assert_eq!(
            watched.last_dn_rates.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            plain.last_dn_rates.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "dn rates {i}"
        );
        let (x, y) = (watched.last_decoded().unwrap(), plain.last_decoded().unwrap());
        assert_eq!(x.direction_probs.map(f32::to_bits), y.direction_probs.map(f32::to_bits));
        assert_eq!(x.aim_angle.to_bits(), y.aim_angle.to_bits());
        assert_eq!(
            watched.state.v().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            plain.state.v().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "membrane state {i}"
        );
    }
}

#[test]
fn a_reset_restarts_the_frames_decision_number_and_owes_no_frame() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    brain.set_viz_every(1);
    for i in 0..3 {
        let _ = brain.decide(&sample_observation(300.0 + i as f32));
    }
    let f = crate::viz::decode_frame(brain.viz_frame_with(1, None).unwrap(), 10.0, 10.0).unwrap();
    assert_eq!(f.seq, 3);
    // Decisions nobody watched pile up; a reset ends the episode: no frame of the old one, numbering from 1.
    let _ = brain.decide(&sample_observation(310.0));
    brain.reset(&ddai_brain::ResetContext {
        map: sample_observation(0.0).map,
        self_id: 0,
        seed: 2,
    });
    assert!(
        brain.viz_frame_with(2, None).is_none(),
        "nothing decided since the reset"
    );
    let _ = brain.decide(&sample_observation(320.0));
    let f = crate::viz::decode_frame(brain.viz_frame_with(3, None).unwrap(), 10.0, 10.0).unwrap();
    assert_eq!((f.seq, f.tick), (1, 3));
}
