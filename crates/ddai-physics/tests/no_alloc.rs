//! Acceptance criterion 4: "zero heap allocations inside `Tick`/`Move` (verify with a counting
//! allocator in a test)". `ddai-physics` itself contains no `unsafe` (denied workspace-wide, see
//! the root `Cargo.toml`'s `[workspace.lints.rust]`) and therefore cannot implement
//! `std::alloc::GlobalAlloc` itself — this test uses the `allocation-counter` dev-dependency,
//! whose `measure(|| ..)` counts only the *calling thread's* allocations (task 1.10b review R2:
//! the earlier process-global `stats_alloc` counters also saw libtest's own threads and other
//! concurrently running tests, so CI flaked; min-of-N over *continuing* windows was not a fix
//! either, since it hides an allocation that only happens sometimes). With per-thread counting
//! there is no cross-thread noise to average away, so this measures **one** window and asserts
//! it is exactly zero: a real allocation — unconditional or conditional on tick/state — fails
//! every run.

use allocation_counter::measure;
use ddai_physics::collision::Collision;
use ddai_physics::core::{self, CharacterCore, PlayerInput, TeamsCore, WorldCore};
use ddai_physics::map::{MapData, TILE_SOLID, Tile};
use ddai_physics::vmath::Vec2;

fn floor_map(w: i32, h: i32) -> MapData {
    let mut game = vec![
        Tile {
            index: 0,
            flags: 0,
            skip: 0,
            reserved: 0
        };
        (w * h) as usize
    ];
    for x in 0..w {
        game[x as usize].index = TILE_SOLID; // top border, for hook-to-ceiling too
        game[((h - 1) * w + x) as usize].index = TILE_SOLID;
    }
    for y in 0..h {
        game[(y * w) as usize].index = TILE_SOLID;
        game[(y * w + (w - 1)) as usize].index = TILE_SOLID;
    }
    MapData {
        width: w as u32,
        height: h as u32,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }
}

/// Drives two characters through a mix of walking, jumping, and hooking each other/the ground —
/// exercising the hook-vs-player search, player-vs-player collision, and hook-drag branches
/// inside `Tick`/`TickDeferred`/`Move`, not just the cheapest idle path.
fn active_input(tick: u32, other_target: (i32, i32)) -> PlayerInput {
    let phase = tick % 40;
    PlayerInput {
        direction: if phase < 20 { 1 } else { -1 },
        target_x: other_target.0,
        target_y: other_target.1,
        jump: i32::from(phase.is_multiple_of(15)),
        hook: i32::from(phase < 30),
        ..Default::default()
    }
}

#[test]
fn tick_move_and_quantize_perform_zero_heap_allocations() {
    let map = floor_map(20, 20);
    let collision: Collision<f32> = Collision::new(&map);
    let teams = TeamsCore::new();

    let mut a = CharacterCore::<f32>::default();
    a.reset();
    a.id = 0;
    a.pos = Vec2::new(160.0, 160.0);
    let mut b = CharacterCore::<f32>::default();
    b.reset();
    b.id = 1;
    b.pos = Vec2::new(220.0, 160.0);
    let mut world: WorldCore<f32, 4> = WorldCore::from_characters(&[(0, a), (1, b)]);

    let step = |world: &mut WorldCore<f32, 4>, tick: u32| {
        world.core_at_mut(0).input = active_input(tick, (220, 160));
        world.core_at_mut(1).input = active_input(tick + 7, (160, 160));
        core::tick(world, 0, &collision, &teams, true, true);
        core::tick(world, 1, &collision, &teams, true, true);
        core::move_character(world, 0, &collision, &teams);
        core::move_character(world, 1, &collision, &teams);
        core::quantize(world.core_at_mut(0));
        core::quantize(world.core_at_mut(1));
    };

    // Warm-up outside the measured region (first-touch page faults, etc. — irrelevant to whether
    // the *code* allocates).
    for t in 0..200 {
        step(&mut world, t);
    }

    let info = measure(|| {
        for t in 0..5_000u32 {
            step(&mut world, t);
        }
    });

    // `count_current` also goes negative for a dealloc of something allocated *before* the window,
    // so a stray `drop` inside the measured code is caught too (the old test asserted
    // `deallocations == 0` as well).
    assert_eq!(
        info.count_total, 0,
        "Tick/TickDeferred/Move/Quantize allocated: {info:?}"
    );
    assert_eq!(
        info.count_current, 0,
        "Tick/TickDeferred/Move/Quantize deallocated: {info:?}"
    );
    assert_eq!(info.bytes_total, 0, "{info:?}");
}

/// Self-check of the measurement itself (task 1.10b review R2), both directions in one test:
/// - allocations made by *other* threads while a window is open are not counted (this is exactly
///   the noise that made the process-global counter flake in CI: here a second thread allocates
///   flat out, and the measured window *waits until it has verifiably done so* inside the window,
///   so the overlap is guaranteed rather than left to the scheduler), and
/// - an allocation on the measuring thread *is* counted, including one that fires only once in
///   the middle of the window (`i == 500`), so a conditional allocation cannot slip through.
#[test]
fn measurement_ignores_other_threads_but_catches_a_conditional_allocation_on_this_one() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

    let stop = Arc::new(AtomicBool::new(false));
    let noise_allocations = Arc::new(AtomicU64::new(0));
    let noise = {
        let (stop, noise_allocations) = (Arc::clone(&stop), Arc::clone(&noise_allocations));
        std::thread::spawn(move || {
            while !stop.load(Relaxed) {
                std::hint::black_box(Vec::<u8>::with_capacity(64));
                noise_allocations.fetch_add(1, Relaxed);
            }
        })
    };

    // A window that itself allocates nothing, but only closes after the noise thread has made at
    // least 1000 allocations *during* it (bounded wait: fail loudly instead of hanging).
    let quiet = measure(|| {
        let start = noise_allocations.load(Relaxed);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while noise_allocations.load(Relaxed) < start + 1_000 {
            assert!(std::time::Instant::now() < deadline, "the noise thread never ran");
            std::thread::yield_now();
        }
    });
    // A window with one allocation, only at iteration 500, while the noise thread keeps running.
    let conditional = measure(|| {
        for i in 0..1_000u32 {
            if i == 500 {
                std::hint::black_box(Vec::<u8>::with_capacity(8));
            }
            std::hint::black_box(i);
        }
    });

    stop.store(true, Relaxed);
    noise.join().unwrap();
    assert!(noise_allocations.load(Relaxed) >= 1_000);
    assert_eq!(
        quiet.count_total, 0,
        "another thread's allocations must not be counted: {quiet:?}"
    );
    assert_eq!(
        conditional.count_total, 1,
        "the one conditional allocation must be counted: {conditional:?}"
    );
    assert_eq!(conditional.bytes_total, 8);
}
