//! Task 1.10 acceptance criterion 1 (and task 1.10b's follow-up): a per-phase `World::step`
//! breakdown, gathered via `Instant`-based instrumentation (`--features phase_profile`) since
//! this box's `perf_event_paranoid=4` blocks `perf` without a system-wide capability change out
//! of scope for either task. Run with:
//!
//!     cargo run --release -p ddai-physics --features phase_profile --example profile_world
//!
//! Two workloads:
//! - `run`: the same "block-like input" pattern `benches/physics.rs`'s
//!   `bench_world_step_on_real_maps` uses, on both corpus maps, with 2 and 8 characters, for
//!   20,000 ticks each (after a 2,000-tick warmup so falling/spawn settling doesn't skew the
//!   numbers).
//! - `run_planner_like_rollouts` (task 1.10b): the search/planner access pattern
//!   `ddai-planner`'s `physics_adapter` uses (one saved baseline `World`, `World::restore_from`
//!   before every candidate rollout, `TICKS_PER_ROLLOUT`-tick rollouts) — Copy Love Box only, 2
//!   and 6 tees, 27-tick rollouts, per the task's own profiling request.
//!
//! Both print each phase's total and percentage of the *measured* wall time (which for the
//! rollout workload includes `restore_from`'s own cost, reported separately too, not folded into
//! any `PhaseProfile` field — `restore_from` isn't part of `World::step`).
use ddai_physics::core::PlayerInput;
use ddai_physics::vmath::Vec2;
use ddai_physics::world::{Player, TickInput, World, spawn_character, take_phase_profile};

/// Shared by both workloads below: prints `p`'s fields as a percentage of `total_wall`, plus the
/// measured throughput line `header` already describes.
fn print_phase_report(p: &ddai_physics::world::PhaseProfile, total_wall: std::time::Duration) {
    let total_phases = p.projectiles
        + p.fixtures
        + p.pickups
        + p.character_pre_tick_pass
        + p.character_tick_pass
        + p.character_deferred_pass
        + p.retain
        + p.strong_weak_id_pass
        + p.switch_expiry;
    let pct = |d: std::time::Duration| 100.0 * d.as_secs_f64() / total_phases.as_secs_f64();
    println!(
        "  projectiles:            {:>10?}  ({:>5.1}%)",
        p.projectiles,
        pct(p.projectiles)
    );
    println!(
        "  fixtures:                {:>10?}  ({:>5.1}%)",
        p.fixtures,
        pct(p.fixtures)
    );
    println!(
        "  pickups:                 {:>10?}  ({:>5.1}%)",
        p.pickups,
        pct(p.pickups)
    );
    println!(
        "  character pre-tick pass: {:>10?}  ({:>5.1}%)",
        p.character_pre_tick_pass,
        pct(p.character_pre_tick_pass)
    );
    println!(
        "  character tick pass:     {:>10?}  ({:>5.1}%)",
        p.character_tick_pass,
        pct(p.character_tick_pass)
    );
    println!(
        "  character deferred pass: {:>10?}  ({:>5.1}%)",
        p.character_deferred_pass,
        pct(p.character_deferred_pass)
    );
    println!(
        "  retain:                  {:>10?}  ({:>5.1}%)",
        p.retain,
        pct(p.retain)
    );
    println!(
        "  strong_weak_id pass:     {:>10?}  ({:>5.1}%)",
        p.strong_weak_id_pass,
        pct(p.strong_weak_id_pass)
    );
    println!(
        "  switch expiry:           {:>10?}  ({:>5.1}%)",
        p.switch_expiry,
        pct(p.switch_expiry)
    );
    println!(
        "  (sum of phases: {:?}, {:.1}% of measured wall time -- remainder is step()'s own \
         input passes / restore_from, see above)",
        total_phases,
        100.0 * total_phases.as_secs_f64() / total_wall.as_secs_f64()
    );
    println!();
}

fn run(map_name: &str, rawmap_path: &std::path::Path, n: usize) {
    let Ok(rawmap_bytes) = std::fs::read(rawmap_path) else {
        eprintln!(
            "profile_world: skipping {map_name} — {} not found",
            rawmap_path.display()
        );
        return;
    };
    let map = ddai_trace::rawmap::read(&rawmap_bytes).expect("parse rawmap");
    let mut world: World<f32> = World::from_map(&map, 1);
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
    for (i, &pos) in spawn_positions.iter().enumerate() {
        world.players[i] = Some(Player::new(0));
        spawn_character(&mut world, i as i32, pos);
    }

    let make_inputs = |tick: u32| -> Vec<TickInput> {
        (0..n)
            .map(|i| {
                let target = spawn_positions[(i + 1) % n] - spawn_positions[i];
                TickInput {
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
            .collect()
    };

    // Warmup (settle falling/spawn), then reset the phase profile and time the real run.
    for tick in 0..2000u32 {
        world.step(&make_inputs(tick));
    }
    take_phase_profile();
    const TICKS: u32 = 20_000;
    let wall_start = std::time::Instant::now();
    for tick in 0..TICKS {
        world.step(&make_inputs(2000 + tick));
    }
    let wall = wall_start.elapsed();
    let p = take_phase_profile();

    println!("=== {map_name}, {n} characters, {TICKS} ticks ===");
    println!(
        "wall time: {:?} ({:.3} µs/tick, {:.0} char-ticks/s)",
        wall,
        wall.as_secs_f64() * 1e6 / TICKS as f64,
        (n as f64) * TICKS as f64 / wall.as_secs_f64()
    );
    print_phase_report(&p, wall);
}

/// Task 1.10b: the search/planner access pattern (one saved baseline, `restore_from` before
/// every candidate rollout, a fixed-length rollout) `ddai-planner`'s `physics_adapter` uses.
/// `n`: tee (character) count; `rollouts`: how many `restore_from` + `ticks_per_rollout`-tick
/// batches to run and measure (after an unmeasured warm-up batch that grows every scratch
/// buffer to its steady-state capacity).
fn run_planner_like_rollouts(map_name: &str, rawmap_path: &std::path::Path, n: usize, ticks_per_rollout: u32) {
    const ROLLOUTS: u32 = 4000;

    let Ok(rawmap_bytes) = std::fs::read(rawmap_path) else {
        eprintln!(
            "profile_world: skipping {map_name} — {} not found",
            rawmap_path.display()
        );
        return;
    };
    let map = ddai_trace::rawmap::read(&rawmap_bytes).expect("parse rawmap");
    let mut world: World<f32> = World::from_map(&map, 1);
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
    for (i, &pos) in spawn_positions.iter().enumerate() {
        world.players[i] = Some(Player::new(0));
        spawn_character(&mut world, i as i32, pos);
    }
    // Settle spawn/falling before this becomes the baseline every rollout restores to.
    for tick in 0..500u32 {
        let inputs: Vec<TickInput> = (0..n)
            .map(|i| TickInput {
                id: i as u8,
                input: PlayerInput {
                    direction: if (tick / 15).is_multiple_of(2) { 1 } else { -1 },
                    ..Default::default()
                },
                kill: false,
            })
            .collect();
        world.step(&inputs);
    }
    let baseline = world.clone();
    let mut scratch = baseline.clone();

    let make_inputs = |tick: u32| -> Vec<TickInput> {
        (0..n)
            .map(|i| {
                let target = spawn_positions[(i + 1) % n] - spawn_positions[i];
                TickInput {
                    id: i as u8,
                    input: PlayerInput {
                        direction: if (tick / 15).is_multiple_of(2) { 1 } else { -1 },
                        target_x: target.x as i32 + 1,
                        target_y: target.y as i32 + 1,
                        jump: i32::from(tick.is_multiple_of(5)),
                        hook: 1,
                        ..Default::default()
                    },
                    kill: false,
                }
            })
            .collect()
    };

    // One unmeasured warm-up rollout: grows `scratch`'s buffers to their steady-state capacity
    // (the same reasoning `bench_world_restore_on_real_maps` uses), so the measured loop below
    // times steady-state `restore_from` calls only, matching the real search-loop access pattern.
    scratch.restore_from(&baseline);
    for t in 0..ticks_per_rollout {
        scratch.step(&make_inputs(t));
    }
    take_phase_profile();

    let mut restore_time = std::time::Duration::ZERO;
    let wall_start = std::time::Instant::now();
    for _ in 0..ROLLOUTS {
        let restore_start = std::time::Instant::now();
        scratch.restore_from(&baseline);
        restore_time += restore_start.elapsed();
        for t in 0..ticks_per_rollout {
            scratch.step(&make_inputs(t));
        }
    }
    let wall = wall_start.elapsed();
    let p = take_phase_profile();

    let total_ticks = u64::from(ROLLOUTS) * u64::from(ticks_per_rollout);
    println!(
        "=== {map_name}, {n} tees, {ROLLOUTS} rollouts x {ticks_per_rollout} ticks (planner-like, restore_from between) ==="
    );
    println!(
        "wall time: {:?} ({:.3} µs/rollout, {:.0} rollouts/s, {:.0} steps/s [world ticks], {:.0} char-ticks/s)",
        wall,
        wall.as_secs_f64() * 1e6 / f64::from(ROLLOUTS),
        f64::from(ROLLOUTS) / wall.as_secs_f64(),
        total_ticks as f64 / wall.as_secs_f64(),
        (n as f64) * total_ticks as f64 / wall.as_secs_f64()
    );
    println!(
        "restore_from: {:?} total ({:.3} µs/call, {:.1}% of wall time)",
        restore_time,
        restore_time.as_secs_f64() * 1e6 / f64::from(ROLLOUTS),
        100.0 * restore_time.as_secs_f64() / wall.as_secs_f64()
    );
    print_phase_report(&p, wall);
}

fn main() {
    let home = std::env::var("HOME").expect("HOME must be set");
    let dir = std::path::PathBuf::from(home).join("aiddnet/data/traces/oracle-b/v1");
    let maps: [(&str, &str); 2] = [
        ("BlmapChill", "realmap_BlmapChill__seed10001.rawmap"),
        (
            "CopyLoveBox",
            "realmap_Copy_Love_Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25__seed10001.rawmap",
        ),
    ];
    for (name, file) in maps {
        for &n in &[2usize, 8usize] {
            run(name, &dir.join(file), n);
        }
    }

    // Task 1.10b: the planner-like rollout workload, Copy Love Box only, 2 and 6 tees.
    let copy_love_box =
        "realmap_Copy_Love_Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25__seed10001.rawmap";
    for &n in &[2usize, 6usize] {
        run_planner_like_rollouts("CopyLoveBox", &dir.join(copy_love_box), n, 27);
    }
}
