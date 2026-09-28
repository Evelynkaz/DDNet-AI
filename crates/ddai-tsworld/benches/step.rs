//! Task 1.9 acceptance criterion 7: `step()` throughput with 2 tees on Copy Love Box, plus
//! `save_state`/`restore_state` cost — compared to Node timing the equivalent loop in the real
//! TS `SimWorld` (see this crate's README, "Производительность", for the numbers this produced
//! and the Node-side command that measured it; there is no pass/fail threshold in the task spec,
//! only "for the planner's budget estimate").
//!
//! Copy Love Box is not committed (`docs/CLAUDE.md` — maps are never committed); this bench
//! reads it from `~/aiddnet/data/maps/copy-love-box/` (the canonical sha256 the task spec names),
//! and is skipped with a clear message if that file isn't present on the machine running it —
//! `cargo bench` never fails outright just because the map isn't there.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ddai_tsworld::world::SimWorldOptions;
use ddai_tsworld::{PlayerInput, SimWorld, Vec2};
use std::hint::black_box;

const CANONICAL_SHA256_PREFIX: &str =
    "Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map";

fn copy_love_box_path() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let path = std::path::PathBuf::from(home)
        .join("aiddnet/data/maps/copy-love-box")
        .join(CANONICAL_SHA256_PREFIX);
    path.is_file().then_some(path)
}

fn make_world() -> Option<SimWorld> {
    let path = copy_love_box_path()?;
    let bytes = std::fs::read(&path).expect("read Copy Love Box");
    let loaded = ddai_tsworld::load_map_bytes(&bytes).expect("load Copy Love Box");
    let mut world = SimWorld::new(loaded.collision, SimWorldOptions::default());
    // Two tees near the middle of the map, walking + periodically jumping/hooking at each
    // other — same "mixed" spirit as `ddai-physics`'s own bench (see that crate's `benches/
    // physics.rs`), not a static/idle scenario that would understate real per-tick cost.
    world.add_tee(1, Vec2 { x: 200.0, y: 200.0 });
    world.add_tee(2, Vec2 { x: 260.0, y: 200.0 });
    Some(world)
}

fn mixed_input(tick: u64, aim_at: Vec2) -> PlayerInput {
    let phase = tick % 80;
    PlayerInput {
        direction: if phase < 40 { 1 } else { -1 },
        target_x: aim_at.x,
        target_y: aim_at.y,
        jump: if tick.is_multiple_of(80) { 1 } else { 0 },
        hook: if tick % 5 < 2 { 1 } else { 0 },
        fire: 0,
        player_flags: 0,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
    }
}

fn bench_step(c: &mut Criterion) {
    let Some(mut world) = make_world() else {
        eprintln!(
            "SKIPPED bench_step: Copy Love Box not found at ~/aiddnet/data/maps/copy-love-box/ — see benches/step.rs"
        );
        return;
    };

    let mut group = c.benchmark_group("copy_love_box_step");
    group.throughput(Throughput::Elements(2)); // 2 tee-ticks per step() call
    group.bench_function("mixed_walk_jump_hook", |b| {
        let mut tick: u64 = 0;
        b.iter(|| {
            world.set_input(1, mixed_input(tick, Vec2 { x: 260.0, y: 200.0 }));
            world.set_input(2, mixed_input(tick + 40, Vec2 { x: 200.0, y: 200.0 }));
            let events = world.step();
            tick += 1;
            black_box(events);
        });
    });
    group.finish();
}

fn bench_save_restore(c: &mut Criterion) {
    let Some(mut world) = make_world() else {
        eprintln!("SKIPPED bench_save_restore: Copy Love Box not found");
        return;
    };
    for i in 0..50 {
        world.set_input(1, mixed_input(i, Vec2 { x: 260.0, y: 200.0 }));
        world.set_input(2, mixed_input(i + 40, Vec2 { x: 200.0, y: 200.0 }));
        world.step();
    }

    let mut group = c.benchmark_group("copy_love_box_state");
    group.bench_function("save_state", |b| {
        b.iter(|| black_box(world.save_state()));
    });
    let saved = world.save_state();
    group.bench_function("restore_state", |b| {
        b.iter(|| world.restore_state(black_box(&saved)));
    });
    group.finish();
}

criterion_group!(benches, bench_step, bench_save_restore);
criterion_main!(benches);
