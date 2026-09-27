//! Acceptance criterion "hot path: zero heap allocations per `step_decision`" — a counting global
//! allocator wrapping the system allocator, checked around a `step_decision` call (after a warm-up
//! call that's allowed to allocate: `FlyState::new`'s own buffers are already allocated by then,
//! but the very first call sometimes still triggers unrelated one-time lazy init in the standard
//! library/backtraces, so this test warms up once, resets the counter, then asserts strictly on
//! the *next* call).
//!
//! Unsafe justification: implementing `GlobalAlloc` requires an `unsafe impl` (the trait itself is
//! `unsafe` — a safe wrapper can't express "correctly forwards every call to a real allocator").
//! This is test-only code in its own integration-test binary (never linked into the `ddai-fly`
//! library, which stays `unsafe_code = "deny"` per the workspace lint policy), and it does nothing
//! but count calls and delegate to `System` — no pointer arithmetic or manual memory management of
//! its own.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use ddai_fly::test_fixtures::{FxEdge, FxNeuron, FxType, build_flyg};
use ddai_fly::{FlyConfig, FlyModel, FlyParams, FlyState};
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

    let allocs_before = ALLOC_COUNT.load(Ordering::SeqCst);
    let deallocs_before = DEALLOC_COUNT.load(Ordering::SeqCst);
    for _ in 0..100 {
        let out = state.step_decision(&model, &inputs);
        std::hint::black_box(out.dn_rates[0]);
    }
    let allocs_after = ALLOC_COUNT.load(Ordering::SeqCst);
    let deallocs_after = DEALLOC_COUNT.load(Ordering::SeqCst);

    assert_eq!(
        allocs_after,
        allocs_before,
        "step_decision must not allocate: {} allocation(s) observed over 100 calls",
        allocs_after - allocs_before
    );
    assert_eq!(
        deallocs_after,
        deallocs_before,
        "step_decision must not deallocate either: {} deallocation(s) observed over 100 calls",
        deallocs_after - deallocs_before
    );
}
