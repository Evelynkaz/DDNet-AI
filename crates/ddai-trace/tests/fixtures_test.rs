//! Golden fixtures produced by Oracle A (see `tools/ddnet-oracle/gen_fixtures.sh` and
//! `docs/formats.md`). Each fixture under `tests/fixtures/*.json` records one scenario's
//! generator params, its scenario bytes' sha256, Oracle A's per-tick canonical state hash for
//! every tick, and the full final-tick state.
//!
//! This test only checks the half of the contract that belongs to this task: that
//! `ddai_trace::generator::random_v1` run with the fixture's recorded params reproduces
//! byte-identical scenario bytes (same sha256) as what was fed to Oracle A when the fixture was
//! built. Task 1.3 (the Rust physics port) will add the other half: running its own physics on
//! the same scenario and checking its per-tick state hashes against `tick_hashes` here.

use ddai_trace::generator::{Params, random_v1};
use ddai_trace::hash::{sha256, to_hex};
use ddai_trace::scenario::{PlayerInput, TuningOverride};
use ddai_trace::trace::{CharacterCoreState, DdnetRef, Producer, ScenarioRef, Trace, TraceMetadata, TraceRow};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn fixture_paths() -> Vec<PathBuf> {
    let dir = fixtures_dir();
    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    paths.sort();
    paths
}

#[test]
fn at_least_eight_fixtures_are_present() {
    // 4 recipes x 2 seeds, per the task spec.
    let paths = fixture_paths();
    assert!(
        paths.len() >= 8,
        "expected at least 8 fixture files, found {}",
        paths.len()
    );
}

#[test]
fn fixtures_directory_is_small() {
    let total: u64 = fixture_paths().iter().map(|p| fs::metadata(p).unwrap().len()).sum();
    assert!(total < 300 * 1024, "fixtures/ is {total} bytes, must stay under 300 KB");
}

#[test]
fn regenerating_each_fixtures_scenario_reproduces_its_recorded_sha256() {
    let paths = fixture_paths();
    assert!(!paths.is_empty(), "no fixtures found in {}", fixtures_dir().display());

    for path in paths {
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        let json: Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));

        let recipe = json["recipe"]
            .as_str()
            .unwrap_or_else(|| panic!("{}: missing 'recipe'", path.display()));
        let seed = json["seed"]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: missing 'seed'", path.display()));
        let ticks = json["ticks"]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: missing 'ticks'", path.display())) as u32;
        let characters = json["characters"]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: missing 'characters'", path.display())) as u32;
        let expected_sha256 = json["scenario_sha256"]
            .as_str()
            .unwrap_or_else(|| panic!("{}: missing 'scenario_sha256'", path.display()));
        let expected_tick_hashes = json["tick_hashes"]
            .as_array()
            .unwrap_or_else(|| panic!("{}: missing 'tick_hashes'", path.display()));

        assert_eq!(
            expected_tick_hashes.len() as u32,
            ticks,
            "{}: tick_hashes length must match 'ticks'",
            path.display()
        );

        let mut scenario = random_v1(
            recipe,
            Params {
                seed,
                ticks,
                characters,
            },
        )
        .unwrap_or_else(|e| panic!("{}: random_v1({recipe}, seed={seed}) failed: {e}", path.display()));

        // `no_weak_hook`/`tuning_overrides` are applied on top of `random_v1`'s output, exactly
        // as `ddnet-ai trace gen-scenario --no-weak-hook --tune ...` does (review round 1,
        // finding F8) — most fixtures have neither (`false`/`[]`, `random_v1`'s own defaults),
        // but a couple deliberately exercise both paths.
        if let Some(true) = json["no_weak_hook"].as_bool() {
            scenario.no_weak_hook = true;
        }
        if let Some(overrides) = json["tuning_overrides"].as_array() {
            for o in overrides {
                let name = o["name"].as_str().expect("tuning_overrides[].name").to_string();
                let value_x100 = o["value_x100"].as_i64().expect("tuning_overrides[].value_x100") as i32;
                scenario.tuning_overrides.push(TuningOverride { name, value_x100 });
            }
        }

        assert_eq!(
            scenario.ticks(),
            ticks as usize,
            "{}: regenerated scenario has the wrong tick count",
            path.display()
        );
        assert_eq!(
            scenario.characters.len(),
            characters as usize,
            "{}: regenerated scenario has the wrong character count",
            path.display()
        );

        let actual_sha256 = to_hex(&sha256(&scenario.write_bytes()));
        assert_eq!(
            actual_sha256,
            expected_sha256,
            "{}: regenerating recipe='{recipe}' seed={seed} ticks={ticks} chars={characters} produced different \
             scenario bytes than what Oracle A traced (generator determinism broke, or this fixture is stale — \
             regenerate it with tools/ddnet-oracle/gen_fixtures.sh)",
            path.display()
        );
    }
}

fn json_i32(v: &Value, field: &str) -> i32 {
    v[field]
        .as_i64()
        .unwrap_or_else(|| panic!("missing/non-integer field '{field}' in {v}")) as i32
}

fn json_f32(v: &Value, field: &str) -> f32 {
    v[field]
        .as_f64()
        .unwrap_or_else(|| panic!("missing/non-numeric field '{field}' in {v}")) as f32
}

fn parse_input(v: &Value) -> PlayerInput {
    PlayerInput {
        direction: json_i32(v, "direction"),
        target_x: json_i32(v, "target_x"),
        target_y: json_i32(v, "target_y"),
        jump: json_i32(v, "jump"),
        fire: json_i32(v, "fire"),
        hook: json_i32(v, "hook"),
        player_flags: json_i32(v, "player_flags"),
        wanted_weapon: json_i32(v, "wanted_weapon"),
        next_weapon: json_i32(v, "next_weapon"),
        prev_weapon: json_i32(v, "prev_weapon"),
    }
}

fn parse_state(v: &Value) -> CharacterCoreState {
    CharacterCoreState {
        pos_x: json_f32(v, "pos_x"),
        pos_y: json_f32(v, "pos_y"),
        vel_x: json_f32(v, "vel_x"),
        vel_y: json_f32(v, "vel_y"),
        hook_pos_x: json_f32(v, "hook_pos_x"),
        hook_pos_y: json_f32(v, "hook_pos_y"),
        hook_dir_x: json_f32(v, "hook_dir_x"),
        hook_dir_y: json_f32(v, "hook_dir_y"),
        hook_tele_base_x: json_f32(v, "hook_tele_base_x"),
        hook_tele_base_y: json_f32(v, "hook_tele_base_y"),
        hook_tick: json_i32(v, "hook_tick"),
        hook_state: json_i32(v, "hook_state"),
        hooked_player: json_i32(v, "hooked_player"),
        active_weapon: json_i32(v, "active_weapon"),
        new_hook: json_i32(v, "new_hook"),
        jumped: json_i32(v, "jumped"),
        jumped_total: json_i32(v, "jumped_total"),
        jumps: json_i32(v, "jumps"),
        direction: json_i32(v, "direction"),
        angle: json_i32(v, "angle"),
        triggered_events: json_i32(v, "triggered_events"),
        colliding: json_i32(v, "colliding"),
        left_wall: json_i32(v, "left_wall"),
        move_restrictions: json_i32(v, "move_restrictions"),
        solo: json_i32(v, "solo"),
        collision_disabled: json_i32(v, "collision_disabled"),
        endless_hook: json_i32(v, "endless_hook"),
        hook_hit_disabled: json_i32(v, "hook_hit_disabled"),
    }
}

/// Review round 1, finding F10: recomputes the canonical FNV-1a 64 hash of each fixture's
/// recorded `final_tick` state (via `Trace::tick_hashes`, the same code path
/// `ddnet-ai trace hashes` uses) and checks it against the fixture's own `tick_hashes[last]` —
/// a self-consistency check that `build_fixture.py` (or a hand-edited fixture) didn't
/// transcribe the final state or the hash list inconsistently with each other.
#[test]
fn fixtures_final_tick_state_hashes_to_the_last_recorded_tick_hash() {
    let paths = fixture_paths();
    assert!(!paths.is_empty(), "no fixtures found in {}", fixtures_dir().display());

    for path in paths {
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        let json: Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));

        let character_ids: Vec<u32> = json["character_ids"]
            .as_array()
            .unwrap_or_else(|| panic!("{}: missing 'character_ids'", path.display()))
            .iter()
            .map(|v| v.as_u64().expect("character id") as u32)
            .collect();
        let expected_tick_hashes = json["tick_hashes"]
            .as_array()
            .unwrap_or_else(|| panic!("{}: missing 'tick_hashes'", path.display()));
        let expected_last_hash = expected_tick_hashes
            .last()
            .unwrap_or_else(|| panic!("{}: 'tick_hashes' is empty", path.display()))
            .as_u64()
            .unwrap();

        let final_rows_json = json["final_tick"]["rows"]
            .as_array()
            .unwrap_or_else(|| panic!("{}: missing 'final_tick.rows'", path.display()));
        assert_eq!(
            final_rows_json.len(),
            character_ids.len(),
            "{}: final_tick row count",
            path.display()
        );
        let rows: Vec<TraceRow> = final_rows_json
            .iter()
            .map(|row| TraceRow {
                input: parse_input(&row["input"]),
                state: parse_state(&row["state"]),
            })
            .collect();

        // The metadata's content is irrelevant to hashing (`Trace::tick_hashes` only reads
        // `self.rows`) — filled in only because `Trace` requires a `TraceMetadata` to exist.
        let dummy_metadata = TraceMetadata {
            producer: Producer {
                name: "fixtures_test".to_string(),
                version: "0".to_string(),
            },
            ddnet: DdnetRef {
                tag: "20.1".to_string(),
                commit: "0".repeat(40),
            },
            map_sha256: "0".repeat(64),
            scenario: ScenarioRef {
                generator: None,
                seed: None,
                scenario_sha256: "0".repeat(64),
            },
            input_schema: ddai_trace::trace::input_schema(),
            state_schema: ddai_trace::trace::state_schema(),
        };
        let single_tick_trace = Trace {
            metadata: dummy_metadata,
            character_ids,
            rows: vec![rows],
        };
        let recomputed = single_tick_trace.tick_hashes()[0];

        assert_eq!(
            recomputed,
            expected_last_hash,
            "{}: recomputed hash of final_tick.rows doesn't match tick_hashes[last] — the fixture's final state and \
             its hash list are inconsistent with each other",
            path.display()
        );
    }
}
