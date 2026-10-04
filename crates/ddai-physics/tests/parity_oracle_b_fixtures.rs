//! Task 1.6, Stage A, spec item 3: a small, committed Oracle-B golden fixture set (synthetic
//! recipes only), for fast CI parity without needing the full multi-GB corpus under
//! `~/aiddnet/data/traces/oracle-b/v1/` — mirrors task 1.2's `ddai-trace/tests/fixtures/*.json`
//! pattern exactly (same shape: `recipe`/`seed`/`ticks`/`characters`/`character_ids`/
//! `tick_hashes`/`final_tick`), just built from Oracle B's trace-b schema (28 core + 54 DDRace
//! fields per character) instead of Oracle A's core-only one.
//!
//! **Hash**: FNV-1a 64 (`ddai_trace::hash::Fnv1a64`, the exact same algorithm
//! `Trace::tick_hashes()` uses) over every character's core-then-DDRace fields, in
//! `character_ids` order, little-endian bytes — one hash per tick. [`row_hash_bytes`] is the
//! single source of truth for "which fields, in which order": both the fixture *generator*
//! (below, `#[ignore]`d, run once against the real `oracle_server` binary to produce the
//! committed JSON) and the *verifier* test (`replays_every_oracle_b_golden_fixture`, run on
//! every `cargo test`) call it, so they can never silently drift apart from each other — only
//! from the real oracle, which the generator's own one-time run against the actual binary is
//! what pins.
//!
//! **Regenerating** (only needed if a fixture goes stale, e.g. a deliberate physics change):
//! `cargo test -p ddai-physics --test parity_oracle_b_fixtures -- --ignored generate_oracle_b_golden_fixtures`
//! (needs `tools/ddnet-oracle/build/ddai_oracle_server` built — see `tools/ddnet-oracle/
//! build-server-oracle.sh`). This is this task's answer to the spec's "add golden-fixture
//! generation for Oracle B if missing": the harness itself (`oracle_server.cpp`) already has
//! everything needed (`--rawmap`/`--scenario`/`--seed`/`--out`) to produce a trace-b file for a
//! synthetic recipe; what was missing was only the *hashing/JSON-fixture* step on top, which
//! lives here instead of inside the C++ harness (per the task spec's "do not modify Oracle A/B
//! tools" — this needs no oracle_server.cpp change at all).

use ddai_physics::vmath::Vec2;
use ddai_physics::world::{self, Player, TickInput, World};
use ddai_trace::hash::Fnv1a64;
use std::path::PathBuf;

#[path = "common/oracle_b_format.rs"]
mod oracle_b_format;
use oracle_b_format::TraceBReader;

/// The single source of truth for "which fields, in which order" a golden-fixture hash covers lives
/// in `common/row_hash.rs` (shared with the stage-B fixture test).
#[path = "common/row_hash.rs"]
mod row_hash;
use row_hash::{row_hash_bytes, world_row_hash_bytes};

/// Same pure function as `parity_oracle_b.rs`'s own `weapon_variety_bonus` (duplicated rather
/// than shared — this file is already self-contained via its own `#[path]` include of
/// `oracle_b_format.rs`, and this is a two-line, unlikely-to-drift helper): `oracle_server.cpp`
/// gives every character a starting-loadout bonus, applied identically for `--rawmap`/
/// `--scenario` replay (what this file's fixtures use) as for fresh `--real-map` generation —
/// see `parity_oracle_b.rs`'s own copy for the full citation.
fn weapon_variety_bonus(seed: u64, id: u32) -> Option<i32> {
    let mut rng = ddai_trace::SplitMix64::new(seed.wrapping_mul(1_000_003).wrapping_add(97u64.wrapping_mul(id as u64)));
    match rng.next_u64() & 3 {
        0 => Some(ddai_physics::core::WEAPON_SHOTGUN),
        1 => Some(ddai_physics::core::WEAPON_GRENADE),
        2 => Some(ddai_physics::core::WEAPON_LASER),
        _ => None,
    }
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures_oracle_b")
}

fn fixture_paths() -> Vec<PathBuf> {
    let dir = fixtures_dir();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    paths.sort();
    paths
}

#[test]
fn at_least_four_oracle_b_fixtures_are_present() {
    // One per recipe, per the task spec's "synthetic recipes only".
    let paths = fixture_paths();
    assert!(
        paths.len() >= 4,
        "expected at least 4 Oracle B fixture files, found {}",
        paths.len()
    );
}

#[test]
fn oracle_b_fixtures_directory_is_small() {
    let total: u64 = fixture_paths()
        .iter()
        .map(|p| std::fs::metadata(p).unwrap().len())
        .sum();
    assert!(
        total < 300 * 1024,
        "fixtures_oracle_b/ is {total} bytes, must stay under 300 KB"
    );
}

/// The fast, always-run CI check (spec item 3): for every committed Oracle-B fixture,
/// regenerate its scenario, run it through Stage A's `World`, and compare per-tick hashes.
#[test]
fn replays_every_oracle_b_golden_fixture() {
    let paths = fixture_paths();
    assert!(
        !paths.is_empty(),
        "no Oracle B fixtures found in {}",
        fixtures_dir().display()
    );

    for path in paths {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        let json: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));

        let recipe = json["recipe"].as_str().unwrap().to_string();
        let seed = json["seed"].as_u64().unwrap();
        let ticks = json["ticks"].as_u64().unwrap() as u32;
        let characters = json["characters"].as_u64().unwrap() as u32;
        let expected_hashes: Vec<u64> = json["tick_hashes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(
            expected_hashes.len(),
            ticks as usize,
            "{}: tick_hashes/ticks length mismatch",
            path.display()
        );

        let map = ddai_trace::synthetic::build(&recipe).unwrap_or_else(|| panic!("unknown recipe '{recipe}'"));
        let scenario = ddai_trace::generator::random_v1(
            &recipe,
            ddai_trace::generator::Params {
                seed,
                ticks,
                characters,
            },
        )
        .unwrap_or_else(|e| panic!("{}: random_v1 failed: {e:?}", path.display()));

        let mut world: World<f32> = World::from_map(&map, seed);
        world.init(std::iter::empty::<&str>()).unwrap();
        for o in &scenario.tuning_overrides {
            let idx = (0..ddai_physics::tuning::TuningParams::num())
                .find(|&i| ddai_physics::tuning::TuningParams::name(i).eq_ignore_ascii_case(&o.name))
                .unwrap_or_else(|| panic!("unknown tuning parameter '{}'", o.name));
            world.tuning.zone_mut(0).set_raw(idx, o.value_x100);
        }
        for c in &scenario.characters {
            world.players[c.id as usize] = Some(Player::new(0));
            world::spawn_character(&mut world, c.id as i32, Vec2::new(c.spawn_x as f32, c.spawn_y as f32));
            if let Some(bonus) = weapon_variety_bonus(seed, c.id) {
                world::give_weapon_to(&mut world, c.id as i32, bonus);
            }
        }

        let mut prev_positions = scenario.spawn_positions();
        let mut ours_hashes = Vec::with_capacity(ticks as usize);
        // `died_this_tick`/`respawned_this_tick` tracking — same formula as `parity_oracle_b.rs`'s
        // `DeathTracking`/`compare_extended` (see `world_row_hash_bytes`'s own doc comment).
        let mut prev_alive = vec![true; scenario.characters.len()];
        let mut rec_die_tick = vec![0i32; scenario.characters.len()];
        for tick_inputs in &scenario.inputs {
            let mut resolved = Vec::with_capacity(scenario.characters.len());
            for (slot, input) in tick_inputs.iter().enumerate() {
                resolved.push(ddai_trace::scenario::resolve_input(input, slot, &prev_positions));
            }
            let inputs: Vec<TickInput> = scenario
                .characters
                .iter()
                .zip(resolved.iter())
                .map(|(c, r)| TickInput {
                    id: c.id as u8,
                    input: ddai_physics::core::PlayerInput {
                        direction: r.direction,
                        target_x: r.target_x,
                        target_y: r.target_y,
                        jump: r.jump,
                        fire: r.fire,
                        hook: r.hook,
                        player_flags: r.player_flags,
                        wanted_weapon: r.wanted_weapon,
                        next_weapon: r.next_weapon,
                        prev_weapon: r.prev_weapon,
                    },
                    kill: false,
                })
                .collect();
            let mut sorted = inputs.clone();
            sorted.sort_by_key(|ti| ti.id);
            world.step(&sorted);

            let mut h = Fnv1a64::new();
            for (slot, c) in scenario.characters.iter().enumerate() {
                if let Some(slot_idx) = world.cores.slot_of(c.id as u8) {
                    let core = world.cores.core_at(slot_idx);
                    prev_positions[slot] = (core.pos.x as i32, core.pos.y as i32);
                }
                let alive = world.characters[c.id as usize].is_some_and(|ch| ch.alive);
                let die_tick = world.players[c.id as usize].map(|p| p.die_tick).unwrap_or(0);
                let spawn_tick = world.characters[c.id as usize].map(|ch| ch.spawn_tick).unwrap_or(-1);
                let died = (prev_alive[slot] && !alive) || (alive && die_tick != rec_die_tick[slot]);
                let respawned = (!prev_alive[slot] && alive) || (alive && spawn_tick == world.tick);
                if alive {
                    rec_die_tick[slot] = die_tick;
                }
                prev_alive[slot] = alive;
                h.update(&world_row_hash_bytes(&world, c.id as i32, died, respawned));
            }
            ours_hashes.push(h.finish());
        }

        assert_eq!(
            ours_hashes.len(),
            expected_hashes.len(),
            "{}: tick count mismatch",
            path.display()
        );
        for (i, (ours, theirs)) in ours_hashes.iter().zip(expected_hashes.iter()).enumerate() {
            assert_eq!(
                ours,
                theirs,
                "{}: tick {i} hash mismatch (ours={ours:#x} theirs={theirs:#x})",
                path.display()
            );
        }
    }
}

/// Regenerates every committed fixture from a fresh Oracle B (`oracle_server`) run. `#[ignore]`d
/// — this needs the real harness binary built (see this file's module doc comment) and is meant
/// to be run explicitly, never on every `cargo test`. Only synthetic recipes (spec item 3: "small
/// committed Oracle-B golden fixtures (synthetic recipes only)").
#[test]
#[ignore]
fn generate_oracle_b_golden_fixtures() {
    // Must run the binary *in place* (not a copy elsewhere): `CStorage::FindDatadir` resolves
    // `$DATADIR` relative to argv[0]'s own path (a sibling `data/` directory next to the
    // executable, written by CMake at build time) — a copy without that sibling directory fails
    // storage init entirely ("cannot add path '$DATADIR'"), found empirically.
    let oracle_bin =
        PathBuf::from(std::env::var("HOME").unwrap()).join("aiddnet/build/oracle-b/build/ddai_oracle_server");
    assert!(
        oracle_bin.is_file(),
        "{} not found — build it first (tools/ddnet-oracle/build-server-oracle.sh)",
        oracle_bin.display()
    );
    let work_dir = std::env::temp_dir().join(format!("ddai-physics-oracle-b-fixtures-{}", std::process::id()));
    std::fs::create_dir_all(&work_dir).unwrap();

    // Small: short enough to keep the committed JSON tiny, long enough to exercise real movement
    // (jumping, hooking, freezing) — the same "interesting, not realistic" inputs `random_v1`
    // always produces for these recipes.
    const CASES: [(&str, u64, u32, u32); 4] = [
        ("arena", 301, 60, 3),
        ("freeze", 302, 60, 3),
        ("front", 303, 60, 2),
        ("tele-speedup", 304, 60, 3),
    ];

    for (recipe, seed, ticks, characters) in CASES {
        let map = ddai_trace::synthetic::build(recipe).unwrap();
        let rawmap_bytes = ddai_trace::rawmap::write(&map);
        let rawmap_path = work_dir.join(format!("{recipe}.rawmap"));
        std::fs::write(&rawmap_path, &rawmap_bytes).unwrap();

        let scenario = ddai_trace::generator::random_v1(
            recipe,
            ddai_trace::generator::Params {
                seed,
                ticks,
                characters,
            },
        )
        .unwrap();
        let scn_path = work_dir.join(format!("{recipe}.scn"));
        std::fs::write(&scn_path, scenario.write_bytes()).unwrap();

        let storage_dir = work_dir.join("storage");
        let _ = std::fs::remove_dir_all(&storage_dir);
        std::fs::create_dir_all(&storage_dir).unwrap();
        let trb_path = work_dir.join(format!("{recipe}.trb"));
        let status = std::process::Command::new(&oracle_bin)
            .current_dir(&work_dir)
            .args([
                "--storage-dir",
                storage_dir.to_str().unwrap(),
                "--rawmap",
                rawmap_path.to_str().unwrap(),
                "--scenario",
                scn_path.to_str().unwrap(),
                "--seed",
                &seed.to_string(),
                "--out",
                trb_path.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", oracle_bin.display()));
        assert!(status.success(), "oracle_server failed for recipe '{recipe}'");

        let trb_bytes = std::fs::read(&trb_path).unwrap();
        let mut trace = TraceBReader::new(&trb_bytes);
        let character_ids = trace.character_ids.clone();
        let mut tick_hashes = Vec::with_capacity(ticks as usize);
        let mut final_rows: Vec<serde_json::Value> = Vec::new();
        while let Some(tick) = trace.next_tick() {
            let mut h = Fnv1a64::new();
            final_rows.clear();
            for row in &tick.characters {
                h.update(&row_hash_bytes(&row.core, &row.ddrace));
                final_rows.push(serde_json::json!({
                    "pos_x": row.core.pos_x, "pos_y": row.core.pos_y,
                    "vel_x": row.core.vel_x, "vel_y": row.core.vel_y,
                    "alive": row.ddrace.alive, "last_weapon": row.ddrace.last_weapon,
                    "weapon_got_mask": row.ddrace.weapon_got_mask,
                }));
            }
            tick_hashes.push(h.finish());
        }

        let fixture = serde_json::json!({
            "recipe": recipe,
            "seed": seed,
            "ticks": ticks,
            "characters": characters,
            "character_ids": character_ids,
            "tick_hashes": tick_hashes,
            "final_tick_summary": final_rows,
        });
        let out_path = fixtures_dir().join(format!("{}_{seed}.json", recipe.replace('-', "_")));
        std::fs::write(&out_path, serde_json::to_string_pretty(&fixture).unwrap()).unwrap();
        eprintln!("wrote {}", out_path.display());
    }
}
