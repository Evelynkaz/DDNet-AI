//! Acceptance criterion 4 (performance): "≥5M character-core ticks/s single-thread for 2
//! characters on the `freeze` recipe (floor 3M), `Clone` of a 2-character core world ≤ 1 µs".
//!
//! Run with `cargo bench -p ddai-physics` (or `--bench physics`); read the reported "time" for
//! each `bench_function`/`bench_with_input` and, for the `world_tick_two_characters` group, the
//! "thrpt" line criterion prints (elements/sec = character-ticks/sec, since each iteration
//! processes 2 characters through a full `Tick`+`TickDeferred`+`Move`+`Quantize` step).
//!
//! **Honest result (see `BUILD REPORT`/task notes): two input patterns are benchmarked, and only
//! the first clears the floor.**
//! - `ground_movement_only` (walking back and forth, no jump, no hook): ≈3.3M character-ticks/s
//!   on this hardware — clears the 3M floor, short of the 5M target.
//! - `mixed_walk_jump_hook` (walking, periodic jumps, periodic hook attempts — the more dynamic,
//!   arguably more representative pattern): measurably slower (≈1.5-1.9M character-ticks/s),
//!   **below the 3M floor**. Root cause (verified by disabling code paths one at a time): jumps
//!   and hook flight both push `m_Vel`'s magnitude up for a few ticks, and both
//!   `CCollision::MoveBox` and `IntersectLineTeleHook`'s loops run `O(velocity magnitude)`
//!   iterations each doing several `CheckPoint`/tile lookups — an *inherent* cost of the ported
//!   algorithm (this is not Rust-specific; DDNet's own C++ pays the same iteration count for the
//!   same inputs), not a bug introduced by this port. One genuine inefficiency *was* found and
//!   fixed while investigating (`crates/ddai-physics/src/core.rs`'s `move_character`: the
//!   player-pass-through loop was copying a sibling's whole `CharacterCore` (488 bytes as of
//!   review round 2's finding F2, 352 before it — including its 188-byte `TuningParams`) on
//!   every sub-step instead of the 3-4 `Copy` scalars it actually reads), but it did not close
//!   the gap for jump/hook-heavy input. Review round 2 (finding F4, waived as a hard floor,
//!   fixed as "cheap wins" regardless) additionally: precomputed a flat solid/nohook lookup
//!   table for `Collision::is_solid`/`check_point` (the hottest per-substep call), skipped a
//!   provably-redundant int→float→round_to_int round trip in `pure_map_index_from_ints`, and
//!   merged `tick`+`tick_deferred`'s extract/write-back cycle to avoid one of three per-tick
//!   `CharacterCore` copies on the common path — see `docs/formats.md`/the task's BUILD REPORT
//!   for measurements. None of these close the `mixed_walk_jump_hook` gap either: it's the
//!   `O(velocity)` loop structure itself, not a copy, that dominates that pattern. Flagged as a
//!   known performance risk for review.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ddai_physics::collision::Collision;
use ddai_physics::core::{self, CharacterCore, PlayerInput, TeamsCore, WorldCore};
use ddai_physics::map::MapData;
use ddai_physics::vmath::Vec2;
use ddai_trace::synthetic;
use std::hint::black_box;

/// Walking back and forth only — no jump, no hook. The cheapest realistic movement pattern.
fn ground_movement_input(tick: u32) -> PlayerInput {
    let phase = tick % 80;
    PlayerInput {
        direction: if phase < 40 { 1 } else { -1 },
        target_x: 1000,
        target_y: 0,
        ..Default::default()
    }
}

/// Walking, with a jump every 80 ticks and a ~40%-duty hook attempt aimed at the other
/// character — a more dynamic (and arguably more representative of real block-mode play, where
/// hooking opponents is the whole point) pattern than pure walking.
fn mixed_input(tick: u32, aim_at: (i32, i32)) -> PlayerInput {
    let phase = tick % 80;
    PlayerInput {
        direction: if phase < 40 { 1 } else { -1 },
        target_x: aim_at.0,
        target_y: aim_at.1,
        jump: i32::from(phase == 0),
        hook: i32::from(phase < 32),
        ..Default::default()
    }
}

fn two_character_world(
    map: &MapData,
    pos_a: Vec2<f32>,
    pos_b: Vec2<f32>,
) -> (Collision<f32>, TeamsCore, WorldCore<f32, 2>) {
    let collision: Collision<f32> = Collision::new(map);
    let teams = TeamsCore::new();

    let mut a = CharacterCore::<f32>::default();
    a.reset();
    a.id = 0;
    a.pos = pos_a;
    let mut b = CharacterCore::<f32>::default();
    b.reset();
    b.id = 1;
    b.pos = pos_b;

    let world: WorldCore<f32, 2> = WorldCore::from_characters(&[(0, a), (1, b)]);
    (collision, teams, world)
}

fn bench_world_tick(c: &mut Criterion) {
    let map = synthetic::build("freeze").expect("freeze recipe must exist");
    // Resting on the freeze recipe's row-18 ledge (see `ddai_trace::synthetic::freeze`), so
    // neither scenario starts with an irrelevant multi-hundred-pixel free-fall settling period.
    let ledge = Vec2::new(4.0 * 32.0 + 16.0, 18.0 * 32.0 - 14.0);
    let ledge2 = Vec2::new(6.0 * 32.0 + 16.0, 18.0 * 32.0 - 14.0);
    let tick_order = [1usize, 0]; // newest-first for 2 characters inserted in order 0, 1

    {
        let (collision, teams, mut world) = two_character_world(&map, ledge, ledge2);
        let mut tick: u32 = 0;
        let mut group = c.benchmark_group("world_tick_two_characters");
        group.throughput(Throughput::Elements(2));
        group.bench_function("ground_movement_only", |b| {
            b.iter(|| {
                world.core_at_mut(0).input = ground_movement_input(tick);
                world.core_at_mut(1).input = ground_movement_input(tick.wrapping_add(40));
                for &slot in &tick_order {
                    core::tick(&mut world, slot, &collision, &teams, true, true);
                }
                for &slot in &tick_order {
                    core::move_character(&mut world, slot, &collision, &teams);
                    core::quantize(world.core_at_mut(slot));
                }
                tick = tick.wrapping_add(1);
                black_box(world.core_at(0).pos);
            });
        });
        group.finish();
    }

    {
        let (collision, teams, mut world) = two_character_world(&map, ledge, ledge2);
        let mut tick: u32 = 0;
        let mut group = c.benchmark_group("world_tick_two_characters");
        group.throughput(Throughput::Elements(2));
        group.bench_function("mixed_walk_jump_hook", |b| {
            b.iter(|| {
                world.core_at_mut(0).input = mixed_input(tick, (ledge2.x as i32, ledge2.y as i32));
                world.core_at_mut(1).input = mixed_input(tick.wrapping_add(11), (ledge.x as i32, ledge.y as i32));
                for &slot in &tick_order {
                    core::tick(&mut world, slot, &collision, &teams, true, true);
                }
                for &slot in &tick_order {
                    core::move_character(&mut world, slot, &collision, &teams);
                    core::quantize(world.core_at_mut(slot));
                }
                tick = tick.wrapping_add(1);
                black_box(world.core_at(0).pos);
            });
        });
        group.finish();
    }
}

fn bench_clone(c: &mut Criterion) {
    let map = synthetic::build("freeze").expect("freeze recipe must exist");
    let (_collision, _teams, world) = two_character_world(
        &map,
        Vec2::new(4.0 * 32.0 + 16.0, 18.0 * 32.0 - 14.0),
        Vec2::new(6.0 * 32.0 + 16.0, 18.0 * 32.0 - 14.0),
    );

    c.bench_function("clone_two_character_world", |b| {
        b.iter(|| black_box(world.clone()));
    });
}

criterion_group!(benches, bench_world_tick, bench_clone);
criterion_main!(benches);
