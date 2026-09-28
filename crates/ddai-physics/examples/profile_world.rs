//! Task 1.10 acceptance criterion 1: a per-phase `World::step` breakdown, gathered via
//! `Instant`-based instrumentation (`--features phase_profile`) since this box's
//! `perf_event_paranoid=4` blocks `perf` without a system-wide capability change out of this
//! task's scope. Run with:
//!
//!     cargo run --release -p ddai-physics --features phase_profile --example profile_world
//!
//! Drives the same "block-like input" pattern `benches/physics.rs`'s
//! `bench_world_step_on_real_maps` uses, on both corpus maps, with 2 and 8 characters, for 20,000
//! ticks each (after a 2,000-tick warmup so falling/spawn settling doesn't skew the numbers), and
//! prints each phase's total and percentage of `World::step`'s own wall time.
use ddai_physics::core::PlayerInput;
use ddai_physics::vmath::Vec2;
use ddai_physics::world::{Player, TickInput, World, spawn_character, take_phase_profile};

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

    let total_phases = p.projectiles
        + p.fixtures
        + p.pickups
        + p.character_pre_tick_pass
        + p.character_tick_pass
        + p.character_deferred_pass
        + p.retain
        + p.strong_weak_id_pass
        + p.switch_expiry;

    println!("=== {map_name}, {n} characters, {TICKS} ticks ===");
    println!(
        "wall time: {:?} ({:.3} µs/tick, {:.0} char-ticks/s)",
        wall,
        wall.as_secs_f64() * 1e6 / TICKS as f64,
        (n as f64) * TICKS as f64 / wall.as_secs_f64()
    );
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
        "  (sum of phases: {:?}, {:.1}% of wall time -- remainder is step()'s own input passes)",
        total_phases,
        100.0 * total_phases.as_secs_f64() / wall.as_secs_f64()
    );
    println!();
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
}
