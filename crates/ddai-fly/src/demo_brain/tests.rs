use super::*;
use crate::brain_fixtures::{FxEdge, FxInputChannel, FxNeuron, FxOutputGroup, FxType, build_brain_flyg};
use crate::config::FlyConfig;
use crate::decoder::DecoderConfig;
use crate::encoder::{ProprioceptionConfig, RayGridConfig};
use crate::params::FlyParams;
use ddai_flyg::{NeuronRole, Side, Sign};

#[test]
fn flat_adam_config_defaults_are_positive() {
    let cfg = FlatAdamConfig::default();
    assert!(cfg.lr > 0.0);
}

#[test]
fn scripted_teacher_points_towards_the_opponent() {
    let map = std::sync::Arc::new(ddai_physics::map::MapData {
        width: 20,
        height: 20,
        game: vec![Default::default(); 400],
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    });
    let mut me = CharacterObservation::at_rest(0);
    me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
    let mut opp = CharacterObservation::at_rest(1);
    opp.pos = ddai_physics::vmath::Vec2::new(500.0, 300.0); // to the right
    let obs = Observation {
        map,
        tick: 0,
        self_state: me,
        others: vec![opp],
        target_id: None,
        tuning: ddai_physics::tuning::TuningParams::default(),
    };
    let cfg = BrainDemoConfig::default();
    let labels = scripted_teacher(&obs, &cfg);
    assert_eq!(
        labels.direction, 2,
        "opponent is to the right -> direction should be 'right'"
    );
}

#[test]
fn scripted_teacher_hooks_only_within_range_and_line_of_sight() {
    let mut game = vec![Default::default(); 400];
    // A solid wall column between self and a "close" opponent.
    for y in 0..20usize {
        game[y * 20 + 10] = ddai_physics::map::Tile {
            index: ddai_physics::map::TILE_SOLID,
            ..Default::default()
        };
    }
    let map = std::sync::Arc::new(ddai_physics::map::MapData {
        width: 20,
        height: 20,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    });
    let mut me = CharacterObservation::at_rest(0);
    me.pos = ddai_physics::vmath::Vec2::new(16.0, 300.0); // tile column 0
    let mut opp = CharacterObservation::at_rest(1);
    opp.pos = ddai_physics::vmath::Vec2::new(400.0, 300.0); // tile column 12, past the wall
    let obs = Observation {
        map,
        tick: 0,
        self_state: me,
        others: vec![opp],
        target_id: None,
        tuning: ddai_physics::tuning::TuningParams::default(),
    };
    let cfg = BrainDemoConfig::default();
    let labels = scripted_teacher(&obs, &cfg);
    assert!(!labels.hook, "a wall blocks line of sight -> must not hook");
}

#[allow(clippy::vec_init_then_push)] // see `tiny_brain_flyg`'s doc comment (same fixture shape).
fn tiny_flyg() -> ddai_flyg::Flyg {
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
    neurons.push(FxNeuron {
        type_index: 2,
        role: NeuronRole::InputAscending,
        side: Side::M,
        full_connectome_in: 1000,
        rf: (0.0, 0.0),
    });
    for _ in 0..2 {
        neurons.push(FxNeuron {
            type_index: 3,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        });
    }
    // Direction: one tied L/R pair on the same type (review round 1, F6 -- matches how the real
    // .flyg's own output_groups assign each side's DN to its own action name, never both).
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

fn tiny_map() -> ddai_physics::map::MapData {
    let mut game = vec![Default::default(); 900];
    // A border of solid tiles so `sample_open_position` has *some* solid tiles to skip (not a
    // requirement for correctness, just makes this a slightly more realistic smoke test).
    for tile in game.iter_mut().take(30) {
        *tile = ddai_physics::map::Tile {
            index: ddai_physics::map::TILE_SOLID,
            ..Default::default()
        };
    }
    ddai_physics::map::MapData {
        width: 30,
        height: 30,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }
}

#[test]
fn run_brain_demo_completes_and_produces_sane_output_on_a_tiny_graph() {
    let flyg = tiny_flyg();
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 1);
    let mut model = FlyModel::new(flyg, config, params).unwrap();
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

    let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
    let mut decoder_params = decoder.init_default_params();
    let calib = DnCalibration {
        mu: vec![0.0; model.num_outputs()],
        sigma: vec![1.0; model.num_outputs()],
    };

    let demo_cfg = BrainDemoConfig {
        batch_size: 2,
        t_decisions: 2,
        steps: 3,
        ..BrainDemoConfig::default()
    };

    let report = run_brain_demo(
        &mut model,
        &index,
        &encoder,
        &mut encoder_params,
        &decoder,
        &mut decoder_params,
        &calib,
        std::sync::Arc::new(tiny_map()),
        &demo_cfg,
    );

    assert_eq!(report.metrics.len(), demo_cfg.steps);
    assert_eq!(report.held_out_n, demo_cfg.held_out_samples);
    for m in [
        &report.metrics_before_fly,
        &report.metrics_after_fly,
        &report.metrics_after_mlp,
    ] {
        assert!((0.0..=1.0).contains(&m.direction.accuracy));
        assert!((0.0..=1.0).contains(&m.direction.majority_baseline_accuracy));
        assert!((0.0..=1.0).contains(&m.jump.accuracy));
        assert!((0.0..=1.0).contains(&m.jump.majority_baseline_accuracy));
        assert!((0.0..=1.0).contains(&m.hook.accuracy));
        assert!((0.0..=1.0).contains(&m.hook.majority_baseline_accuracy));
    }
    assert!(report.mlp_num_params > 0);
    assert!(report.fly_num_params > 0);
    for m in &report.metrics {
        assert!(m.loss.is_finite());
    }
}

#[test]
fn wilson_ci95_contains_the_point_estimate_and_widens_for_smaller_n() {
    let (lo, hi) = super::wilson_ci95(50, 100);
    assert!(lo <= 0.5 && hi >= 0.5);
    let (lo_small, hi_small) = super::wilson_ci95(5, 10);
    let (lo_large, hi_large) = super::wilson_ci95(500, 1000);
    assert!(
        hi_small - lo_small > hi_large - lo_large,
        "a smaller n must give a wider interval for the same proportion"
    );
}

#[test]
fn wilson_ci95_handles_the_boundaries_without_panicking_or_producing_nan() {
    for (k, n) in [(0, 0), (0, 10), (10, 10)] {
        let (lo, hi) = super::wilson_ci95(k, n);
        assert!(lo.is_finite() && hi.is_finite());
        assert!(lo <= hi);
        assert!((0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi));
    }
}

#[test]
fn auroc_binary_is_one_when_every_positive_outscores_every_negative() {
    let scores = vec![0.1, 0.2, 0.8, 0.9];
    let labels = vec![false, false, true, true];
    assert_eq!(super::auroc_binary(&scores, &labels), 1.0);
}

#[test]
fn auroc_binary_is_zero_point_five_for_scores_that_carry_no_information() {
    let scores = vec![0.5, 0.5, 0.5, 0.5];
    let labels = vec![false, true, false, true];
    assert!((super::auroc_binary(&scores, &labels) - 0.5).abs() < 1e-6);
}

#[test]
fn auroc_binary_is_nan_when_a_class_is_missing() {
    let scores = vec![0.1, 0.2, 0.3];
    let labels = vec![false, false, false];
    assert!(super::auroc_binary(&scores, &labels).is_nan());
}

#[test]
fn a_constant_predictor_gets_zero_balanced_accuracy_gain_over_chance_despite_high_plain_accuracy() {
    // Review round 1, F2's exact failure mode: 90 negatives, 10 positives, a predictor that
    // always says "negative" scores 90% plain accuracy but must score exactly 0.5 balanced
    // accuracy (chance) and 0.0 recall on the positive class.
    let mut probs = vec![0.1f32; 90];
    probs.extend(vec![0.1f32; 10]);
    let mut labels = vec![false; 90];
    labels.extend(vec![true; 10]);
    let m = super::binary_metrics(&probs, &labels);
    assert!((m.accuracy - 0.9).abs() < 1e-6);
    assert!(
        (m.majority_baseline_accuracy - 0.9).abs() < 1e-6,
        "matches the constant predictor exactly"
    );
    assert_eq!(m.recall_positive, 0.0);
    assert_eq!(m.recall_negative, 1.0);
    assert!((m.balanced_accuracy - 0.5).abs() < 1e-6);
}

/// A map with solid columns every 4 tiles (`x % 4 == 0`) and open ground elsewhere: unlike
/// `tiny_map` (a single solid row, so a "wall directly ahead" scenario is vanishingly rare — the
/// real degenerate case [`sample_scenario_stratified`]'s own doc comment already documents), this
/// gives self positions a genuine mix of "wall within `jump_lookahead_px`" and "clear ahead"
/// depending on which column and direction the teacher's `direction` label happens to pick, so a
/// rejection sampler actually has both classes to find.
fn columns_map_for_jump_stratification() -> ddai_physics::map::MapData {
    let (width, height) = (40u32, 20u32);
    let mut game = vec![Default::default(); (width * height) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            if x % 4 == 0 {
                game[y * width as usize + x] = ddai_physics::map::Tile {
                    index: ddai_physics::map::TILE_SOLID,
                    ..Default::default()
                };
            }
        }
    }
    ddai_physics::map::MapData {
        width,
        height,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }
}

#[test]
fn sample_scenario_stratified_moves_the_jump_positive_rate_towards_the_target() {
    let map = std::sync::Arc::new(columns_map_for_jump_stratification());
    let cfg = BrainDemoConfig {
        jump_target_positive_rate: 0.5,
        jump_stratify_max_attempts: 500,
        ..BrainDemoConfig::default()
    };
    let mut rng = SplitMix64::new(123);
    let n = 200;
    let mut positives = 0usize;
    for _ in 0..n {
        let (_obs, labels) = sample_scenario_stratified(std::sync::Arc::clone(&map), &cfg, &mut rng);
        positives += usize::from(labels.jump);
    }
    let rate = positives as f32 / n as f32;
    // Not exactly 0.5 (the rejection sampler's accept condition is on `jump` alone, and the
    // fallback path can still return an off-target draw after `jump_stratify_max_attempts`), but
    // nowhere near a real map's ~1.5% natural rate either.
    assert!(
        (0.25..=0.75).contains(&rate),
        "stratified jump-positive rate {rate} should be pulled well away from the natural rarity"
    );
}
