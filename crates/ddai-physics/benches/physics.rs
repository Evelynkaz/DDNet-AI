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
use ddai_physics::real::Real;
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

/// Task 1.6, spec item 7 ("measure that it costs nothing noticeable"): the `black_box`-hardened
/// `Real::powf`/`sin`/`atan2` (see `src/real.rs`'s module doc comment) against the exact same
/// calls made directly through `f32`'s own inherent methods (no `black_box`) — both loops touch
/// the same number of transcendental calls over the same varying input, so any real overhead
/// `black_box` adds would show up as a throughput difference between the two `bench_function`s.
fn bench_real_math_black_box(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_math_black_box");
    group.throughput(Throughput::Elements(1));

    group.bench_function("via_real_trait_with_black_box", |b| {
        let mut x: f32 = 0.1;
        b.iter(|| {
            x = black_box(x.powf(black_box(2.0f32)));
            x = Real::sin(x) * 0.5 + 0.5; // keep x in a stable range across iterations
            x = Real::atan2(x, 0.7f32);
            black_box(x)
        });
    });

    group.bench_function("direct_std_no_black_box", |b| {
        let mut x: f32 = 0.1;
        b.iter(|| {
            x = f32::powf(x, 2.0f32);
            x = f32::sin(x) * 0.5 + 0.5;
            x = f32::atan2(x, 0.7f32);
            black_box(x)
        });
    });

    group.finish();
}

/// Task 1.6, Stage A, spec item 6: "criterion bench on Copy Love Box and BlmapChill with 2 and 8
/// characters doing block-like input: report world-ticks/s and character-ticks/s". Loads each
/// map's rawmap bytes straight from the Oracle B corpus (`~/aiddnet/data/traces/oracle-b/v1/`,
/// outside the repo — skipped, not failed, if that directory isn't present in this environment,
/// same convention `tests/parity_oracle_b.rs` uses), places `n` characters on its own
/// `spawn_points` (falling back to the map center, spaced out, if the map has fewer spawn points
/// than characters), and drives every tick with "block-like" input: `hook` held every tick
/// (block mode's defining behavior — constantly trying to hook other players/walls),
/// `direction` alternating every 15 ticks, `jump` pulsed every 5th tick, `target_x`/`target_y`
/// aimed at the *next* character in the list (wrapping) — a deterministic, no-RNG stand-in for
/// "a player trying to hook someone" rather than a claim of realistic human play. This is
/// `World::step()` alone (no trace comparison, no I/O) — the harness-dominated
/// ~13.8k ticks/s/~41k character-ticks/s `measures_world_step_throughput_on_one_real_map_trace`
/// (`tests/parity_oracle_b.rs`) reports is a different, much lower number for exactly that
/// reason (comparison/IO-bound, not `step()`-bound); see this bench's own numbers in the task's
/// `BUILD REPORT` for the honest comparison against the ≥1M character-ticks/s target.
fn bench_world_step_on_real_maps(c: &mut Criterion) {
    let corpus_dir = {
        let home = std::env::var("HOME").expect("HOME must be set");
        std::path::PathBuf::from(home).join("aiddnet/data/traces/oracle-b/v1")
    };
    let maps: [(&str, &str); 2] = [
        ("BlmapChill", "realmap_BlmapChill__seed10001.rawmap"),
        (
            "CopyLoveBox",
            "realmap_Copy_Love_Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25__seed10001.rawmap",
        ),
    ];

    for (map_name, rawmap_file) in maps {
        let rawmap_path = corpus_dir.join(rawmap_file);
        let Ok(rawmap_bytes) = std::fs::read(&rawmap_path) else {
            eprintln!(
                "bench_world_step_on_real_maps: skipping {map_name} — {} not found (Oracle B corpus not present in this environment)",
                rawmap_path.display()
            );
            continue;
        };
        let map = match ddai_trace::rawmap::read(&rawmap_bytes) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("bench_world_step_on_real_maps: skipping {map_name} — failed to parse {rawmap_file}: {e:?}");
                continue;
            }
        };

        for &n in &[2usize, 8usize] {
            let mut world: ddai_physics::world::World<f32> = ddai_physics::world::World::from_map(&map, 1);
            world.init(std::iter::empty::<&str>()).unwrap();
            let spawn_positions: Vec<Vec2<f32>> = (0..n)
                .map(|i| {
                    world
                        .spawn_points
                        .get(i)
                        .copied()
                        .unwrap_or_else(|| Vec2::new(200.0 + (i as f32) * 64.0, 200.0))
                })
                .collect();
            let spawn_at = |i: usize| -> Vec2<f32> { spawn_positions[i] };
            for i in 0..n {
                world.players[i] = Some(ddai_physics::world::Player::new(0));
                ddai_physics::world::spawn_character(&mut world, i as i32, spawn_at(i));
            }

            let mut tick: u32 = 0;
            let mut group = c.benchmark_group(format!("world_step_{map_name}_{n}_characters"));
            group.throughput(Throughput::Elements(n as u64));
            group.bench_function("block_like_input", |b| {
                b.iter(|| {
                    let inputs: Vec<ddai_physics::world::TickInput> = (0..n)
                        .map(|i| {
                            let target = spawn_at((i + 1) % n) - spawn_at(i);
                            ddai_physics::world::TickInput {
                                id: i as u8,
                                input: PlayerInput {
                                    direction: if (tick / 15).is_multiple_of(2) { 1 } else { -1 },
                                    target_x: target.x as i32 + 1,
                                    target_y: target.y as i32 + 1,
                                    jump: i32::from(tick.is_multiple_of(5)),
                                    fire: 0,
                                    hook: 1,
                                    player_flags: 0,
                                    wanted_weapon: 0,
                                    next_weapon: 0,
                                    prev_weapon: 0,
                                },
                                kill: false,
                            }
                        })
                        .collect();
                    world.step(&inputs);
                    tick = tick.wrapping_add(1);
                    black_box(world.tick);
                });
            });
            group.finish();
        }
    }
}

/// Review round 1, finding F10: [`bench_clone`] above measures `WorldCore::clone()` (the tiny
/// task-1.3 core array), not `world::World::clone()` (this task's own type, acceptance criterion
/// 1/6's actual "save/restore" target) — on a real map, `World::collision` used to be the
/// dominant cost of a `World::clone()` (measured before the `Arc` fix: ~3.25 ms / 21.7 MB for
/// `BlmapChill`, ~350 `World::step` calls' worth). This benches the real thing, 2 characters, on
/// the same two real maps [`bench_world_step_on_real_maps`] already loads — target: a few µs.
fn bench_world_clone_on_real_maps(c: &mut Criterion) {
    let corpus_dir = {
        let home = std::env::var("HOME").expect("HOME must be set");
        std::path::PathBuf::from(home).join("aiddnet/data/traces/oracle-b/v1")
    };
    let maps: [(&str, &str); 2] = [
        ("BlmapChill", "realmap_BlmapChill__seed10001.rawmap"),
        (
            "CopyLoveBox",
            "realmap_Copy_Love_Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25__seed10001.rawmap",
        ),
    ];

    for (map_name, rawmap_file) in maps {
        let rawmap_path = corpus_dir.join(rawmap_file);
        let Ok(rawmap_bytes) = std::fs::read(&rawmap_path) else {
            eprintln!(
                "bench_world_clone_on_real_maps: skipping {map_name} — {} not found (Oracle B corpus not present in this environment)",
                rawmap_path.display()
            );
            continue;
        };
        let map = match ddai_trace::rawmap::read(&rawmap_bytes) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("bench_world_clone_on_real_maps: skipping {map_name} — failed to parse {rawmap_file}: {e:?}");
                continue;
            }
        };

        let mut world: ddai_physics::world::World<f32> = ddai_physics::world::World::from_map(&map, 1);
        world.init(std::iter::empty::<&str>()).unwrap();
        for i in 0..2 {
            let pos = world
                .spawn_points
                .get(i)
                .copied()
                .unwrap_or_else(|| Vec2::new(200.0 + (i as f32) * 64.0, 200.0));
            world.players[i] = Some(ddai_physics::world::Player::new(0));
            ddai_physics::world::spawn_character(&mut world, i as i32, pos);
        }

        c.bench_function(&format!("clone_two_character_world_{map_name}"), |b| {
            b.iter(|| black_box(world.clone()));
        });
    }
}

/// The two corpus rawmap files [`bench_world_step_on_real_maps`]/[`bench_world_clone_on_real_maps`]
/// each load independently; factored out for the task 1.10 benches below, which need the same
/// pair again. Returns `(map_name, MapData)`, skipping (not failing) a map that isn't present in
/// this environment, same convention as every other bench/test reading this corpus.
fn load_corpus_maps() -> Vec<(&'static str, MapData)> {
    let corpus_dir = {
        let home = std::env::var("HOME").expect("HOME must be set");
        std::path::PathBuf::from(home).join("aiddnet/data/traces/oracle-b/v1")
    };
    let maps: [(&str, &str); 2] = [
        ("BlmapChill", "realmap_BlmapChill__seed10001.rawmap"),
        (
            "CopyLoveBox",
            "realmap_Copy_Love_Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25__seed10001.rawmap",
        ),
    ];
    let mut out = Vec::new();
    for (map_name, rawmap_file) in maps {
        let rawmap_path = corpus_dir.join(rawmap_file);
        let Ok(rawmap_bytes) = std::fs::read(&rawmap_path) else {
            eprintln!(
                "load_corpus_maps: skipping {map_name} — {} not found (Oracle B corpus not present in this environment)",
                rawmap_path.display()
            );
            continue;
        };
        match ddai_trace::rawmap::read(&rawmap_bytes) {
            Ok(m) => out.push((map_name, m)),
            Err(e) => {
                eprintln!("load_corpus_maps: skipping {map_name} — failed to parse {rawmap_file}: {e:?}");
            }
        }
    }
    out
}

/// Builds a 2-character `World` on `map`, spawned on its own first two spawn points (falling back
/// to a fixed offset apart if it has fewer than two), same convention
/// [`bench_world_step_on_real_maps`]/[`bench_world_clone_on_real_maps`] each use inline.
fn two_character_real_world(map: &MapData) -> ddai_physics::world::World<f32> {
    let mut world: ddai_physics::world::World<f32> = ddai_physics::world::World::from_map(map, 1);
    world.init(std::iter::empty::<&str>()).unwrap();
    for i in 0..2 {
        let pos = world
            .spawn_points
            .get(i)
            .copied()
            .unwrap_or_else(|| Vec2::new(200.0 + (i as f32) * 64.0, 200.0));
        world.players[i] = Some(ddai_physics::world::Player::new(0));
        ddai_physics::world::spawn_character(&mut world, i as i32, pos);
    }
    world
}

/// Task 1.10, acceptance criterion 4: `World::restore_from`, on the same two real maps, 2
/// characters — the direct counterpart to [`bench_world_clone_on_real_maps`]'s `World::clone()`
/// measurement, target "a few µs" either way (acceptance criterion "clone or save/restore <= 5
/// µs"). The first `restore_from` call (outside the timed loop) grows every scratch `Vec` to its
/// final capacity, so the timed loop measures steady-state (zero-allocation) restores only,
/// matching the real search-loop access pattern (grow once, reuse forever).
fn bench_world_restore_on_real_maps(c: &mut Criterion) {
    for (map_name, map) in load_corpus_maps() {
        let source = two_character_real_world(&map);
        let mut scratch = source.clone();
        scratch.restore_from(&source); // warm up: grow `scratch`'s buffers to fit.

        c.bench_function(&format!("restore_two_character_world_{map_name}"), |b| {
            b.iter(|| {
                scratch.restore_from(&source);
                black_box(&scratch);
            });
        });
    }
}

/// Task 1.10, acceptance criterion 4: a "search-like" benchmark — from one saved state, run 64
/// rollouts x 30 ticks each (restoring the shared state between rollouts, not re-cloning it),
/// 2 characters, on the same two real maps. Reports rollouts/s via criterion's own throughput
/// line (`Throughput::Elements(64)`); ms/decision-equivalent (what one live decision's search
/// would cost if it needed exactly this many rollouts) isn't printed by this function itself —
/// it's `Throughput`'s "time" column (one full `b.iter()` batch of 64 rollouts) reported as-is,
/// read straight off criterion's own output rather than recomputed and printed here.
fn bench_search_like_rollouts(c: &mut Criterion) {
    const ROLLOUTS: usize = 64;
    const TICKS_PER_ROLLOUT: usize = 30;

    for (map_name, map) in load_corpus_maps() {
        let baseline = two_character_real_world(&map);
        let spawn_a = baseline
            .spawn_points
            .first()
            .copied()
            .unwrap_or(Vec2::new(200.0, 200.0));
        let spawn_b = baseline.spawn_points.get(1).copied().unwrap_or(Vec2::new(264.0, 200.0));
        let mut scratch = baseline.clone();

        let mut group = c.benchmark_group(format!("search_like_rollouts_{map_name}"));
        group.throughput(Throughput::Elements(ROLLOUTS as u64));
        group.bench_function("64_rollouts_30_ticks_2_characters", |b| {
            b.iter(|| {
                for r in 0..ROLLOUTS {
                    scratch.restore_from(&baseline);
                    for t in 0..TICKS_PER_ROLLOUT {
                        let tick = (r * TICKS_PER_ROLLOUT + t) as u32;
                        let target_a = spawn_b - spawn_a;
                        let target_b = spawn_a - spawn_b;
                        let inputs = [
                            ddai_physics::world::TickInput {
                                id: 0,
                                input: PlayerInput {
                                    direction: if (tick / 15).is_multiple_of(2) { 1 } else { -1 },
                                    target_x: target_a.x as i32 + 1,
                                    target_y: target_a.y as i32 + 1,
                                    jump: i32::from(tick.is_multiple_of(5)),
                                    hook: 1,
                                    ..Default::default()
                                },
                                kill: false,
                            },
                            ddai_physics::world::TickInput {
                                id: 1,
                                input: PlayerInput {
                                    direction: if (tick / 15).is_multiple_of(2) { -1 } else { 1 },
                                    target_x: target_b.x as i32 + 1,
                                    target_y: target_b.y as i32 + 1,
                                    jump: i32::from(tick.is_multiple_of(5)),
                                    hook: 1,
                                    ..Default::default()
                                },
                                kill: false,
                            },
                        ];
                        scratch.step(&inputs);
                    }
                }
                black_box(scratch.tick);
            });
        });
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_world_tick,
    bench_clone,
    bench_world_clone_on_real_maps,
    bench_world_restore_on_real_maps,
    bench_search_like_rollouts,
    bench_real_math_black_box,
    bench_world_step_on_real_maps
);
criterion_main!(benches);
