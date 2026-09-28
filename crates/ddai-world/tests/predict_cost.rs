//! `LiveWorld::predict`'s cost (task spec, D-041: "measure its cost (µs) and avoid per-call
//! allocations" — the live decision loop's hard latency budget makes this matter). Two measures:
//!
//! - Wall-clock cost (µs), warmed up, over many iterations, for a realistic character count
//!   (6: our own tee plus 5 opponents — D-041's "up to 5+ opponents in radius").
//! - Heap allocations via a counting allocator (`stats_alloc`, the same pattern
//!   `ddai-physics/tests/world_no_alloc.rs` uses, for the same "this crate has no `unsafe`, so it
//!   cannot implement `GlobalAlloc` itself" reason). Review round 1's API note: `predict()` now
//!   uses task 1.10's `World::restore_from` (a save/restore reusing every buffer) instead of
//!   `Clone`, so the whole call — not just its step loop — is zero-allocation in steady state,
//!   verified below at both 10 and 40 stepped ticks.

use std::sync::Arc;
use std::time::Instant;

use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects;
use ddai_net::tuning::DEFAULT_TUNE_PARAMS;
use ddai_net::view::CharacterView;
use ddai_physics::core::PlayerInput;
use ddai_world::LiveWorld;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// Serializes every test in this file against `stats_alloc`'s process-global counters — same
/// reasoning as `ddai-physics/tests/world_no_alloc.rs`'s own `ALLOC_TEST_LOCK`.
static ALLOC_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn neutral_character(id: i32, x: i32, y: i32) -> CharacterView {
    let character = objects::Character {
        tick: 0, // "this already is the current tick" - no reckoning extrapolation needed.
        x,
        y,
        vel_x: 0,
        vel_y: 0,
        angle: 0,
        direction: 0,
        jumped: 0,
        hooked_player: -1,
        hook_state: -1,
        hook_tick: 0,
        hook_x: x,
        hook_y: y,
        hook_dx: 0,
        hook_dy: 0,
        player_flags: playerflagflag::PLAYING,
        health: 10,
        armor: 0,
        ammo_count: -1,
        weapon: 0,
        emote: 0,
        attack_tick: 0,
    };
    let ddnet = objects::DDNetCharacter {
        flags: 0,
        freeze_end: 0,
        jumps: 2,
        tele_checkpoint: -1,
        strong_weak_id: id,
        jumped_total: -1,
        ninja_activation_tick: -1,
        freeze_start: -1,
        target_x: 0,
        target_y: 0,
        tune_zone_override: -1,
    };
    CharacterView {
        id,
        character,
        ddnet: Some(ddnet),
    }
}

const OWN: i32 = 0;
const NUM_OPPONENTS: i32 = 5; // D-041: "up to 5+" opponents in radius.

fn build_live_world() -> LiveWorld {
    let map = Arc::new(ddai_trace::synthetic::build("arena").expect("recipe must exist"));
    let mut live = LiveWorld::new(map, OWN, 1);
    let mut characters = vec![neutral_character(OWN, 300, 300)];
    for i in 1..=NUM_OPPONENTS {
        characters.push(neutral_character(i, 300 + i * 40, 300));
    }
    live.on_snapshot(100, &characters, DEFAULT_TUNE_PARAMS, &[], None, None);
    live
}

#[test]
fn predict_wall_clock_cost_for_a_realistic_character_count() {
    let _guard = ALLOC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut live = build_live_world();

    // Warm-up: first calls may pay for page faults on freshly-grown Vec capacity.
    for _ in 0..50 {
        let _ = live.predict(live.base_tick() + 4, &[]);
    }

    const ITERS: usize = 5000;
    let mut samples_ns = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let world = live.predict(live.base_tick() + 4, &[]);
        std::hint::black_box(world.tick);
        samples_ns.push(start.elapsed().as_nanos() as u64);
    }
    samples_ns.sort_unstable();
    let mean_ns: u64 = (samples_ns.iter().sum::<u64>()) / samples_ns.len() as u64;
    let p50 = samples_ns[samples_ns.len() / 2];
    let p99 = samples_ns[samples_ns.len() * 99 / 100];
    let max = *samples_ns.last().unwrap();
    eprintln!(
        "predict(+4 ticks, {} characters): mean={:.2}us p50={:.2}us p99={:.2}us max={:.2}us",
        NUM_OPPONENTS + 1,
        mean_ns as f64 / 1000.0,
        p50 as f64 / 1000.0,
        p99 as f64 / 1000.0,
        max as f64 / 1000.0
    );
    // D-041's own budget is "the whole decision <= 5ms p99" with the fly getting ~1ms and search
    // the rest — `predict()` is one (cheap) ingredient of that search, not the whole budget. This
    // asserts on `p50`, not `p99`: on a shared, contended box (this crate's `BUILD REPORT` notes
    // several other agents' own `cargo build`/`cargo test` runs on the same machine) an occasional
    // OS scheduler preemption mid-measurement can push a handful of individual samples into the
    // millisecond range with no change to `predict()`'s own cost at all — found empirically (a
    // clean, unloaded run measured p99 ~113us; a contended one, same code, p99 ~3.1ms but p50
    // *still* ~70us) — so `p99` is reported (eprintln above) for a human reading a quiet run, but
    // is not a reliable automated gate. `p50` over `ITERS` samples is: a single preemption can only
    // ever move a small minority of samples, so the median stays representative of the *typical*
    // call even under load.
    assert!(
        p50 < 2_000_000,
        "predict() p50 ({p50} ns) is unexpectedly far past a sane micro-budget"
    );
}

#[test]
fn predict_is_fully_zero_allocation_once_warmed_up() {
    let _guard = ALLOC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // `predict`'s `self.scratch.restore_from(&self.world)` (task 1.10's `World::restore_from`,
    // review round 1's API note) reuses every one of `scratch`'s own buffers — including its
    // *private* per-call scratch buffers (`entity_order_scratch`/`range_scratch`/
    // `hammer_scratch`/`map_indices_scratch`), which `restore_from` explicitly clears in place
    // rather than replacing — instead of `Clone`'s fresh-allocate-every-`Vec`-field behavior. So,
    // once every buffer has grown to fit this world once (the warm-up loop below), every further
    // `predict()` call — restore *and* every ticked `World::step` — allocates nothing at all,
    // regardless of how many ticks it steps.
    let mut live = build_live_world();
    for _ in 0..20 {
        let _ = live.predict(live.base_tick() + 40, &[]);
    }

    let measure = |live: &mut LiveWorld, extra_ticks: i32| -> stats_alloc::Stats {
        let base_tick = live.base_tick();
        let region = Region::new(GLOBAL);
        let world = live.predict(base_tick + extra_ticks, &[]);
        std::hint::black_box(world.tick);
        region.change()
    };

    let restore_only = measure(&mut live, 0);
    eprintln!(
        "predict() restore-only (0 step ticks): {} allocations, {} bytes",
        restore_only.allocations, restore_only.bytes_allocated
    );
    let restore_plus_10 = measure(&mut live, 10);
    eprintln!(
        "predict() restore + 10 step ticks: {} allocations, {} bytes",
        restore_plus_10.allocations, restore_plus_10.bytes_allocated
    );
    let restore_plus_40 = measure(&mut live, 40);
    eprintln!(
        "predict() restore + 40 step ticks: {} allocations, {} bytes",
        restore_plus_40.allocations, restore_plus_40.bytes_allocated
    );

    for (label, stats) in [
        ("restore-only", restore_only),
        ("restore + 10 ticks", restore_plus_10),
        ("restore + 40 ticks", restore_plus_40),
    ] {
        assert_eq!(
            stats.allocations, 0,
            "{label}: expected 0 allocations, got {}",
            stats.allocations
        );
        assert_eq!(
            stats.bytes_allocated, 0,
            "{label}: expected 0 bytes allocated, got {}",
            stats.bytes_allocated
        );
    }
}

#[test]
fn own_inputs_in_flight_lookup_adds_no_allocation_of_its_own() {
    let _guard = ALLOC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut live = build_live_world();
    let in_flight: Vec<(i32, PlayerInput)> = (1..=10)
        .map(|i| {
            (
                live.base_tick() + i,
                PlayerInput {
                    direction: 1,
                    target_x: 1,
                    target_y: -1,
                    ..Default::default()
                },
            )
        })
        .collect();

    for _ in 0..20 {
        let _ = live.predict(live.base_tick() + 10, &in_flight);
        let _ = live.predict(live.base_tick() + 10, &[]);
    }

    let base_tick = live.base_tick();
    let without_in_flight = {
        let region = Region::new(GLOBAL);
        let world = live.predict(base_tick + 10, &[]);
        std::hint::black_box(world.tick);
        region.change()
    };
    let with_in_flight = {
        let region = Region::new(GLOBAL);
        let world = live.predict(base_tick + 10, &in_flight);
        std::hint::black_box(world.tick);
        region.change()
    };
    // Same restore, same 10 real ticks stepped either way — `own_inputs_in_flight`'s own linear
    // `.find()` scan (`predict`'s doc comment) must not itself cost an allocation on top of that,
    // and (task 1.10's `World::restore_from`) neither must anything else: both sides are zero.
    assert_eq!(with_in_flight.allocations, without_in_flight.allocations);
    assert_eq!(with_in_flight.bytes_allocated, without_in_flight.bytes_allocated);
    assert_eq!(with_in_flight.allocations, 0);
}
