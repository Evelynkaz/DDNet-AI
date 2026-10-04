//! Min-of-many micro-benchmark of `World::step`, robust against a loaded machine (task 1.6 stage
//! B, acceptance criterion 4): criterion's mean/median swings by several hundred percent between
//! runs on a shared box whose load average is above the core count (the `cargo bench` numbers of
//! that task's report were useless for a 3 % comparison). Preemption and cache pollution only ever
//! make a batch slower, so this times many *short* wall-clock batches (a few hundred steps, well
//! under a scheduler time slice) and reports the **minimum** — the standard noise-resistant
//! estimator — next to the 10th percentile as a sanity bound. No `unsafe`, no dependencies.
//!
//! ```text
//! cargo run --release -p ddai-physics --example world_step_cpu_time -- [batches] [steps-per-batch] [name-filter]
//! ```
//!
//! Scenarios (all 2 tees, block-like input, `hook` held, direction flipping every 15 ticks — the
//! same driver as `benches/physics.rs`'s `world_step_*_2_characters`):
//! - `CopyLoveBox` / `BlmapChill`: the two corpus block maps (the former has no stage-B entity,
//!   the latter 74 draggers, 13 turrets and 2 lights in the map).
//! - `...+laser`: the same, with both tees holding the laser rifle and firing every other tick
//!   (stage-B `CLaser` cost; the `World` of earlier stages ignores the fire).
//! - `crafted/*`: hand-made stage-B maps from the stage-B corpus (`draggers_*`, `turrets_*`,
//!   `lights_*`, `mixed_*`), tees at their scenario spawn points with the same driver.
//!
//! Prints the minimum and 10th-percentile `ns/step` and the `character-ticks/s` at the minimum.

use ddai_physics::core::{PlayerInput, WEAPON_LASER};
use ddai_physics::vmath::Vec2;
use ddai_physics::world::{self, Player, TickInput, World};
use std::path::PathBuf;

struct Scenario {
    name: String,
    world: World<f32>,
    spawns: Vec<Vec2<f32>>,
    laser: bool,
}

fn corpus(sub: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join("aiddnet/data/traces/oracle-b")
        .join(sub)
}

fn load_map(path: &std::path::Path) -> Option<ddai_physics::map::MapData> {
    let bytes = std::fs::read(path).ok()?;
    ddai_trace::rawmap::read(&bytes).ok()
}

fn build(name: &str, map: &ddai_physics::map::MapData, spawns: Option<Vec<Vec2<f32>>>, laser: bool) -> Scenario {
    let mut world: World<f32> = World::from_map(map, 1);
    world.init(std::iter::empty::<&str>()).unwrap();
    let spawns = spawns.unwrap_or_else(|| {
        (0..2)
            .map(|i| {
                world
                    .spawn_points
                    .get(i)
                    .copied()
                    .unwrap_or_else(|| Vec2::new(200.0 + i as f32 * 64.0, 200.0))
            })
            .collect()
    });
    for (i, &p) in spawns.iter().enumerate() {
        world.players[i] = Some(Player::new(0));
        world::spawn_character(&mut world, i as i32, p);
        if laser {
            world::give_weapon_to(&mut world, i as i32, WEAPON_LASER);
            let slot = world.cores.slot_of(i as u8).unwrap();
            world.cores.core_at_mut(slot).active_weapon = WEAPON_LASER;
        }
    }
    Scenario {
        name: name.to_string(),
        world,
        spawns,
        laser,
    }
}

fn inputs(tick: u32, spawns: &[Vec2<f32>], laser: bool) -> Vec<TickInput> {
    let n = spawns.len();
    (0..n)
        .map(|i| {
            let target = spawns[(i + 1) % n] - spawns[i];
            TickInput {
                id: i as u8,
                input: PlayerInput {
                    direction: if (tick / 15).is_multiple_of(2) { 1 } else { -1 },
                    target_x: target.x as i32 + 1,
                    target_y: target.y as i32 + 1,
                    jump: i32::from(tick.is_multiple_of(5)),
                    fire: if laser {
                        i32::from(tick % 2 == 1) + 2 * ((tick / 2) as i32 % 31)
                    } else {
                        0
                    },
                    hook: 1,
                    ..Default::default()
                },
                kill: false,
            }
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let batches: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let steps: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let filter = args.get(3).cloned().unwrap_or_default();
    let v1 = corpus("v1");
    let stage_b = corpus("v2-stageb");
    let clb =
        "realmap_Copy_Love_Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25__seed10001.rawmap";

    let mut scenarios: Vec<Scenario> = Vec::new();
    for (label, file) in [
        ("CopyLoveBox", clb),
        ("BlmapChill", "realmap_BlmapChill__seed10001.rawmap"),
    ] {
        if let Some(map) = load_map(&v1.join(file)) {
            scenarios.push(build(label, &map, None, false));
            scenarios.push(build(&format!("{label}+laser"), &map, None, true));
        } else {
            eprintln!("skipping {label}: {} not found", v1.join(file).display());
        }
    }
    for stem in [
        "draggers_v0_s34000",
        "turrets_v0_s35000",
        "lights_v0_s33000",
        "mixed_v0_s36000",
    ] {
        let rawmap = stage_b.join(format!("{stem}.rawmap"));
        let scn = stage_b.join(format!("{stem}.scn"));
        let (Some(map), Ok(scn_bytes)) = (load_map(&rawmap), std::fs::read(&scn)) else {
            eprintln!("skipping crafted/{stem}: not found under {}", stage_b.display());
            continue;
        };
        // First two characters' spawn points out of the scenario (see `docs/formats.md` §12.6: after
        // the header, `character_count`, then `(id, x, y, team)` quadruples).
        let spawns = scenario_spawns(&scn_bytes);
        scenarios.push(build(&format!("crafted/{stem}"), &map, Some(spawns), false));
    }

    scenarios.retain(|sc| sc.name.contains(&filter));
    println!(
        "scenario                         min ns/step  p10 ns/step   (min of {batches} batches x {steps} steps)   char-ticks/s at min"
    );
    for mut sc in scenarios {
        let mut tick = 0u32;
        // warm up: every first-fire / first-beam growth happens here
        for _ in 0..3000 {
            let input = inputs(tick, &sc.spawns, sc.laser);
            sc.world.step(&input);
            tick = tick.wrapping_add(1);
        }
        let mut samples: Vec<f64> = Vec::with_capacity(batches);
        for _ in 0..batches {
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                let input = inputs(tick, &sc.spawns, sc.laser);
                sc.world.step(&input);
                tick = tick.wrapping_add(1);
            }
            samples.push(t0.elapsed().as_nanos() as f64 / f64::from(steps));
        }
        samples.sort_by(f64::total_cmp);
        let (min, p10) = (samples[0], samples[samples.len() / 10]);
        println!(
            "{:<32} {:>10.1} {:>10.1}                {:>12.0}",
            sc.name,
            min,
            p10,
            sc.spawns.len() as f64 * 1e9 / min
        );
    }
}

/// `(spawn_x, spawn_y)` of every character of a scenario-v3 file, hand-parsed (no dependency on the
/// test-only reader).
fn scenario_spawns(b: &[u8]) -> Vec<Vec2<f32>> {
    let rd = |o: usize| i32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let mut o = 4 + 4 + 1; // magic, version, map_ref_tag
    let path_len = u16::from_le_bytes(b[o..o + 2].try_into().unwrap()) as usize;
    o += 2 + path_len + 32 + 1 + 4; // path, sha256, no_weak_hook, tuning override count
    let n = rd(o) as usize;
    o += 4;
    (0..n)
        .map(|i| {
            let base = o + i * 16;
            Vec2::new(rd(base + 4) as f32, rd(base + 8) as f32)
        })
        .collect()
}
