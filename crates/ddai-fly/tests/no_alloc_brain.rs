//! Acceptance criterion 7: "allocation-free per decision" for the *whole* `FlyBrain::decide`
//! call (encode -> `step_decision` -> decode), not just 7.1's own `step_decision` (already covered
//! by `tests/no_alloc.rs`). Same counting-allocator technique as that file; see its own doc
//! comment for the unsafe-impl justification (this is a separate integration-test binary, so its
//! own `#[global_allocator]` doesn't conflict with that file's).
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ddai_brain::{Brain, CharacterObservation, Observation};
use ddai_fly::brain::{ActionSelection, FlyBrain, FlyBrainConfig};
use ddai_fly::brain_fixtures::{FxEdge, FxInputChannel, FxNeuron, FxOutputGroup, FxType, build_brain_flyg};
use ddai_fly::config::FlyConfig;
use ddai_fly::decoder::{DecoderConfig, DecoderModel, DnCalibration};
use ddai_fly::encoder::{EncoderModel, EncoderParams, ProprioceptionConfig, RayGridConfig};
use ddai_fly::model::FlyModel;
use ddai_fly::params::FlyParams;
use ddai_flyg::{NeuronRole, Side, Sign};

struct CountingAllocator;
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

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

fn make_brain() -> FlyBrain {
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
            action_selection: ActionSelection::Argmax,
            seed: 1,
        },
    )
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

#[test]
fn decide_allocates_nothing() {
    let mut brain = make_brain();
    let obs = sample_observation(400.0);

    // Warm-up call (allowed to allocate: lazy init, etc.) before measuring, same discipline as
    // `tests/no_alloc.rs`.
    let _ = brain.decide(&obs);

    // Built *before* the counted region: constructing an `Observation` itself (an owned `Arc`
    // map, an owned `Vec` of other characters) is the caller's/server's job, not part of what
    // `decide()` must be allocation-free for — only the call being measured below is.
    let observations: Vec<Observation> = (0..100).map(|i| sample_observation(300.0 + i as f32)).collect();

    let allocs_before = ALLOC_COUNT.load(Ordering::SeqCst);
    let deallocs_before = DEALLOC_COUNT.load(Ordering::SeqCst);
    for obs in &observations {
        let action = brain.decide(obs);
        std::hint::black_box(action);
    }
    let allocs_after = ALLOC_COUNT.load(Ordering::SeqCst);
    let deallocs_after = DEALLOC_COUNT.load(Ordering::SeqCst);

    assert_eq!(
        allocs_after,
        allocs_before,
        "FlyBrain::decide must not allocate: {} allocation(s) observed over 100 calls",
        allocs_after - allocs_before
    );
    assert_eq!(
        deallocs_after,
        deallocs_before,
        "FlyBrain::decide must not deallocate either: {} deallocation(s) observed over 100 calls",
        deallocs_after - deallocs_before
    );
}
