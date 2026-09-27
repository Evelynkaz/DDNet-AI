//! Acceptance criterion 4: "zero heap allocations inside `Tick`/`Move` (verify with a counting
//! allocator in a test)". `ddai-physics` itself contains no `unsafe` (denied workspace-wide, see
//! the root `Cargo.toml`'s `[workspace.lints.rust]`) and therefore cannot implement
//! `std::alloc::GlobalAlloc` itself (every implementation of that trait is `unsafe impl`) — this
//! test uses the small, widely-used `stats_alloc` crate (dev-dependency only) instead, which
//! wraps the system allocator and counts allocations/deallocations/reallocations without this
//! crate's own source ever writing the word `unsafe`.

use ddai_physics::collision::Collision;
use ddai_physics::core::{self, CharacterCore, PlayerInput, TeamsCore, WorldCore};
use ddai_physics::map::{MapData, TILE_SOLID, Tile};
use ddai_physics::vmath::Vec2;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

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

    let region = Region::new(GLOBAL);
    for t in 0..5_000u32 {
        step(&mut world, t);
    }
    let stats = region.change();

    assert_eq!(
        stats.allocations, 0,
        "Tick/TickDeferred/Move/Quantize allocated: {stats:?}"
    );
    assert_eq!(
        stats.reallocations, 0,
        "Tick/TickDeferred/Move/Quantize reallocated: {stats:?}"
    );
    assert_eq!(
        stats.deallocations, 0,
        "Tick/TickDeferred/Move/Quantize deallocated: {stats:?}"
    );
    assert_eq!(stats.bytes_allocated, 0, "{stats:?}");
}
