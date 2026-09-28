//! Review round 1, finding F9: `World::step` must perform zero heap allocations per tick in
//! *steady state* (acceptance criterion 1: "zero heap allocations per tick in steady state
//! (entity lists may use pre-reserved capacity; document any unavoidable allocation)"). Mirrors
//! `no_alloc.rs`'s pattern (a counting allocator via the `stats_alloc` dev-dependency, since
//! `ddai-physics` itself has no `unsafe` and so cannot implement `GlobalAlloc` — see that file's
//! own doc comment) but drives the full `World::step`, not just `core::tick`/`Move`/`Quantize`,
//! on a map with pickups (to exercise `pickup_tick`'s own `find_characters_in_range_into` path
//! every tick regardless of gameplay pattern) and with a mix of walking/jumping/hooking/firing/
//! hammering input (to exercise `fire_hammer`/`projectile_tick`/`create_explosion`'s own scratch
//! buffers too, not just the cheapest idle path).
//!
//! `World::step` still legitimately allocates on two discrete, rare events, *not* part of
//! steady-state per-tick cost, deliberately not exercised by this test's own input pattern:
//! `can_spawn` (only on an actual respawn — see its own doc comment) and `World::init`/
//! `apply_commands`'s `Vec<UnknownCommand>` (config parsing, once per scenario, not per tick).
//!
//! **`ALLOC_TEST_LOCK`**: `StatsAlloc`'s counters are process-global, not per-thread, and
//! `cargo test` runs every test in this file on its own thread by default — without
//! serialization, one test's own (measured or not) allocations bleed into another's `Region`
//! window, producing a spurious nonzero count that has nothing to do with `World::step` (found
//! empirically once this file grew a second `#[test]`: `--test-threads=1` alone made the
//! "failure" disappear). Every test here holds this lock for its entire body, not just its
//! measured region, since even an *unmeasured* warm-up phase running concurrently with another
//! test's `Region` window would still pollute it.
static ALLOC_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

use ddai_physics::core::PlayerInput;
use ddai_physics::vmath::Vec2;
use ddai_physics::world::{self, Player, TickInput, World};
use ddai_trace::synthetic;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;
use std::path::{Path, PathBuf};

#[path = "common/oracle_b_format.rs"]
mod oracle_b_format;
use oracle_b_format::{ScenarioV3, TraceBReader, metadata_seed};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// A mix of walking, jumping, hooking, firing (gun, so `projectile_tick` runs every tick once
/// any shot is in flight) and hammering (so `fire_hammer`'s own scratch path runs too).
fn active_input(tick: u32, other_target: (i32, i32)) -> PlayerInput {
    let phase = tick % 40;
    PlayerInput {
        direction: if phase < 20 { 1 } else { -1 },
        target_x: other_target.0,
        target_y: other_target.1,
        jump: i32::from(phase.is_multiple_of(15)),
        hook: i32::from(phase < 10),
        fire: i32::from(phase.is_multiple_of(6)),
        wanted_weapon: if phase.is_multiple_of(20) { 1 } else { 0 },
        ..Default::default()
    }
}

#[test]
fn world_step_performs_zero_heap_allocations_in_steady_state() {
    let _guard = ALLOC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // "freeze" (task 1.2's synthetic recipes) places freeze/health pickups on the map, so
    // `pickup_tick`'s `find_characters_in_range_into` scratch path runs every tick regardless of
    // where the characters are standing.
    let map = synthetic::build("freeze").expect("freeze recipe must exist");
    let mut world: World<f32> = World::from_map(&map, 12345);
    world.players[0] = Some(Player::new(0));
    world.players[1] = Some(Player::new(0));
    world::spawn_character(&mut world, 0, Vec2::new(200.0, 200.0));
    world::spawn_character(&mut world, 1, Vec2::new(260.0, 200.0));
    world::give_weapon_to(&mut world, 0, ddai_physics::core::WEAPON_GRENADE);
    world::give_weapon_to(&mut world, 1, ddai_physics::core::WEAPON_GRENADE);

    let step = |world: &mut World<f32>, tick: u32| {
        let inputs = [
            TickInput {
                id: 0,
                input: active_input(tick, (260, 200)),
                kill: false,
            },
            TickInput {
                id: 1,
                input: active_input(tick + 11, (200, 200)),
                kill: false,
            },
        ];
        world.step(&inputs);
    };

    // Warm-up outside the measured region (first-touch page faults, every scratch buffer's
    // capacity growing to fit — that growth is exactly the "pre-reserved capacity" the
    // acceptance criterion allows; it must happen *before* the measured region, not during it).
    for t in 0..500 {
        step(&mut world, t);
    }

    let region = Region::new(GLOBAL);
    for t in 500..3_000u32 {
        step(&mut world, t);
    }
    let stats = region.change();

    assert_eq!(stats.allocations, 0, "World::step allocated: {stats:?}");
    assert_eq!(stats.reallocations, 0, "World::step reallocated: {stats:?}");
    assert_eq!(stats.deallocations, 0, "World::step deallocated: {stats:?}");
    assert_eq!(stats.bytes_allocated, 0, "{stats:?}");
}

fn resolve_rawmap_path(scn_path: &Path, rawmap_path_in_file: &str) -> PathBuf {
    let p = PathBuf::from(rawmap_path_in_file);
    if p.is_file() {
        return p;
    }
    scn_path.with_file_name(p.file_name().unwrap())
}

fn tick_inputs(ids: &[u32], reference: &oracle_b_format::TraceBTick) -> Vec<TickInput> {
    let mut inputs: Vec<TickInput> = ids
        .iter()
        .zip(reference.characters.iter())
        .map(|(&id, row)| TickInput {
            id: id as u8,
            input: PlayerInput {
                direction: row.input.direction,
                target_x: row.input.target_x,
                target_y: row.input.target_y,
                jump: row.input.jump,
                fire: row.input.fire,
                hook: row.input.hook,
                player_flags: row.input.player_flags,
                wanted_weapon: row.input.wanted_weapon,
                next_weapon: row.input.next_weapon,
                prev_weapon: row.input.prev_weapon,
            },
            kill: row.input.kill != 0,
        })
        .collect();
    inputs.sort_by_key(|ti| ti.id);
    inputs
}

/// Review round 2, finding F9 \[CONFIRMED\]: the synthetic-recipe test above passed only because
/// its characters' movement pattern never actually crossed a `Collision::tile_exists` tile
/// (freeze/speedup/stopper/tele/switch/kill/etc — `Collision::get_map_indices_into`'s own doc
/// comment), so it never exercised the allocation `get_map_indices` (the un-suffixed, `Vec`-
/// returning version this crate no longer has) used to have. This one instead replays a *real*
/// map trace's *real* recorded input (Copy Love Box, seed 10001 — the reviewer's own
/// measurement target: 1652 allocations, 1276/3000 ticks, before this fix), the same corpus
/// `parity_oracle_b.rs` uses, so "the tee never touches a `tile_exists` tile" cannot happen here
/// — Copy Love Box's block layout guarantees frequent freeze/tele/speedup crossings for real
/// recorded block-mode play.
///
/// The measured region stops at the first recorded `died_this_tick` (if any): a death's
/// resulting respawn eventually calls `can_spawn`, which still legitimately allocates (a
/// discrete, rare event, not steady-state — see its own doc comment, and the reviewer's own `rev
/// alloc` instruction: "confirm 0 *outside* `can_spawn`"). Asserting a hard `0` over a window
/// that might include a real respawn would make this test flaky/trace-dependent for no reason;
/// stopping at the first death keeps the assertion both meaningful (a large stretch of a real,
/// tile-crossing block-mode trace) and exact. A first pass (unmeasured) finds that boundary; a
/// second, fresh `World` replays up to it under measurement.
#[test]
fn world_step_performs_zero_heap_allocations_replaying_a_real_map_trace() {
    let _guard = ALLOC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let home = std::env::var("HOME").expect("HOME must be set");
    let dir = PathBuf::from(home).join("aiddnet/data/traces/oracle-b/v1");
    let trb_path = dir
        .join("realmap_Copy_Love_Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25__seed10001.trb");
    if !trb_path.is_file() {
        eprintln!(
            "skipping: {} not found (Oracle B corpus not present in this environment)",
            trb_path.display()
        );
        return;
    }
    let trb_bytes = std::fs::read(&trb_path).unwrap();

    let scn_path = trb_path.with_extension("scn");
    let scn = ScenarioV3::read_bytes(&std::fs::read(&scn_path).unwrap());
    let rawmap_path = resolve_rawmap_path(&scn_path, &scn.rawmap_path);
    let rawmap_bytes = std::fs::read(&rawmap_path).unwrap();
    let map = ddai_trace::rawmap::read(&rawmap_bytes).expect("rawmap parse failed");
    let ids: Vec<u32> = scn.characters.iter().map(|c| c.id).collect();
    let seed = metadata_seed(&TraceBReader::new(&trb_bytes).metadata_json);

    let build_world = || -> World<f32> {
        let mut world: World<f32> = World::from_map(&map, seed);
        world.init(scn.cfg_lines.iter().map(|s| s.as_str())).unwrap();
        world.apply_commands(scn.cfg_lines.iter().map(|s| s.as_str())).unwrap();
        for c in &scn.characters {
            world.players[c.id as usize] = Some(Player::new(0));
            world::spawn_character(&mut world, c.id as i32, Vec2::new(c.spawn_x as f32, c.spawn_y as f32));
            if c.team != 0 {
                world::set_force_character_team(&mut world, c.id as i32, c.team);
            }
        }
        world
    };

    // First (unmeasured) pass: collect every tick's resolved input and find the first tick
    // index (0-based) any character's `died_this_tick` fires.
    let mut all_inputs: Vec<Vec<TickInput>> = Vec::new();
    let mut first_death_tick: Option<usize> = None;
    {
        let mut trace = TraceBReader::new(&trb_bytes);
        let mut idx = 0usize;
        while let Some(reference) = trace.next_tick() {
            if first_death_tick.is_none() && reference.characters.iter().any(|c| c.ddrace.died_this_tick != 0) {
                first_death_tick = Some(idx);
            }
            all_inputs.push(tick_inputs(&ids, &reference));
            idx += 1;
        }
    }
    let warm_up = 50usize;
    let measured_end = first_death_tick.unwrap_or(all_inputs.len());
    assert!(
        measured_end > warm_up,
        "trace too short or dies too early to exercise a meaningful measured region \
         (first_death_tick={first_death_tick:?}, total_ticks={})",
        all_inputs.len()
    );

    let mut world = build_world();
    for inputs in &all_inputs[..warm_up] {
        world.step(inputs);
    }

    let region = Region::new(GLOBAL);
    for inputs in &all_inputs[warm_up..measured_end] {
        world.step(inputs);
    }
    let stats = region.change();

    assert_eq!(
        stats.allocations, 0,
        "World::step allocated replaying Copy Love Box seed 10001 (ticks {warm_up}..{measured_end}, \
         first_death_tick={first_death_tick:?}): {stats:?}"
    );
    assert_eq!(stats.reallocations, 0, "World::step reallocated: {stats:?}");
    assert_eq!(stats.deallocations, 0, "World::step deallocated: {stats:?}");
    assert_eq!(stats.bytes_allocated, 0, "{stats:?}");
}
