//! Acceptance criterion "hot path: zero heap allocations per `step_decision`" — per-thread
//! allocation counting (`allocation_counter::measure`, a dev-dependency; it installs the test
//! binary's counting `#[global_allocator]` itself, so this file needs no `unsafe`), checked around
//! a window of `step_decision` calls after a warm-up call that's allowed to allocate:
//! `FlyState::new`'s own buffers are already allocated by then, but the very first call sometimes
//! still triggers unrelated one-time lazy init in the standard library/backtraces, so this test
//! warms up once and then asserts strictly on the following window.
//!
//! Task 1.10b review R2: `measure` counts only the *calling thread's* allocations, so libtest's own
//! threads and other concurrently running tests cannot leak into the window (the previous
//! process-global counter flaked in CI, and a min-of-N patch over continuing windows would have
//! hidden allocations that only happen sometimes). One window, exactly zero.

use allocation_counter::measure;
use ddai_fly::test_fixtures::{FxEdge, FxNeuron, FxType, build_flyg};
use ddai_fly::{FlyConfig, FlyModel, FlyParams, FlyState};
use ddai_flyg::{NeuronRole, Side, Sign};

fn build_graph_with_a_few_neurons() -> ddai_flyg::Flyg {
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
    let neurons = [
        FxNeuron {
            type_index: 0,
            role: NeuronRole::InputAscending,
            side: Side::M,
            full_connectome_in: 10,
        },
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
    ];
    let edges = [
        FxEdge {
            pre: 0,
            post: 2,
            synapse_count: 3,
        },
        FxEdge {
            pre: 1,
            post: 3,
            synapse_count: 3,
        },
        FxEdge {
            pre: 2,
            post: 4,
            synapse_count: 2,
        },
        FxEdge {
            pre: 3,
            post: 4,
            synapse_count: 2,
        },
    ];
    build_flyg(&types, &neurons, &edges)
}

#[test]
fn step_decision_allocates_nothing() {
    let flyg = build_graph_with_a_few_neurons();
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 1);
    let model = FlyModel::new(flyg, config, params).unwrap();
    let mut state = FlyState::new(&model);
    let inputs = vec![0.3f32, 0.6f32];

    // Warm up: `FlyState::new` above already allocated every scratch buffer, but run one call
    // before measuring anyway in case the very first call anywhere in the process trips some
    // unrelated one-time lazy init (e.g. thread-local setup) that would otherwise look like a
    // `step_decision` allocation.
    let _ = state.step_decision(&model, &inputs);

    let info = measure(|| {
        for _ in 0..100 {
            let out = state.step_decision(&model, &inputs);
            std::hint::black_box(out.dn_rates[0]);
        }
    });

    assert_eq!(
        info.count_total, 0,
        "step_decision must not allocate: {info:?} over 100 calls"
    );
    assert_eq!(
        info.count_current, 0,
        "step_decision must not deallocate either: {info:?} over 100 calls"
    );
}
