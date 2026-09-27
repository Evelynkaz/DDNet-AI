//! Fast CI parity test (acceptance criterion 2.a): for every golden fixture in
//! `crates/ddai-trace/tests/fixtures/` (task 1.2, produced by real Oracle A), regenerate the
//! scenario, run `core_world` in `f32`, compute the canonical per-tick state hashes, and compare
//! against the fixture: 0 mismatches expected. On failure, prints the first mismatching tick and
//! (for the last tick, which the fixture stores full field state for) a field-level diff.

mod common;

use common::{run_scenario, run_scenario_generic};
use ddai_trace::generator::{Params, random_v1};
use ddai_trace::scenario::TuningOverride;
use ddai_trace::trace::{
    CharacterCoreState, DdnetRef, Producer, ScenarioRef, Trace, TraceMetadata, TraceRow, diff, input_schema,
    state_schema,
};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    // crates/ddai-physics/tests -> crates/ddai-trace/tests/fixtures
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ddai-trace/tests/fixtures")
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

fn dummy_metadata() -> TraceMetadata {
    TraceMetadata {
        producer: Producer {
            name: "ddai-physics-parity-test".to_string(),
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
        input_schema: input_schema(),
        state_schema: state_schema(),
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

fn parse_input(v: &Value) -> ddai_trace::scenario::PlayerInput {
    ddai_trace::scenario::PlayerInput {
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

#[test]
fn rust_physics_matches_every_golden_fixture() {
    let paths = fixture_paths();
    assert!(!paths.is_empty(), "no fixtures found in {}", fixtures_dir().display());

    let mut scenarios_checked = 0u64;
    let mut total_ticks = 0u64;
    let mut total_character_ticks = 0u64;

    for path in &paths {
        let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
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

        let mut scenario = random_v1(
            recipe,
            Params {
                seed,
                ticks,
                characters,
            },
        )
        .unwrap_or_else(|e| panic!("{}: random_v1({recipe}, seed={seed}) failed: {e}", path.display()));
        if let Some(true) = json["no_weak_hook"].as_bool() {
            scenario.no_weak_hook = true;
        }
        if let Some(overrides) = json["tuning_overrides"].as_array() {
            for o in overrides {
                scenario.tuning_overrides.push(TuningOverride {
                    name: o["name"].as_str().expect("tuning_overrides[].name").to_string(),
                    value_x100: o["value_x100"].as_i64().expect("tuning_overrides[].value_x100") as i32,
                });
            }
        }

        let map = ddai_trace::synthetic::build(recipe).unwrap_or_else(|| panic!("unknown recipe '{recipe}'"));
        let rows = run_scenario(&map, &scenario);

        let character_ids: Vec<u32> = scenario.characters.iter().map(|c| c.id).collect();
        let our_trace = Trace {
            metadata: dummy_metadata(),
            character_ids: character_ids.clone(),
            rows,
        };
        let our_hashes = our_trace.tick_hashes();

        let expected_hashes: Vec<u64> = json["tick_hashes"]
            .as_array()
            .unwrap_or_else(|| panic!("{}: missing 'tick_hashes'", path.display()))
            .iter()
            .map(|v| v.as_u64().expect("tick_hashes[] must be u64"))
            .collect();

        assert_eq!(
            our_hashes.len(),
            expected_hashes.len(),
            "{}: tick count mismatch ({} vs {})",
            path.display(),
            our_hashes.len(),
            expected_hashes.len()
        );

        if our_hashes != expected_hashes {
            let first_mismatch_tick = our_hashes
                .iter()
                .zip(expected_hashes.iter())
                .position(|(a, b)| a != b)
                .unwrap();
            let mismatch_count = our_hashes
                .iter()
                .zip(expected_hashes.iter())
                .filter(|(a, b)| a != b)
                .count();
            let total = our_hashes.len();
            let ours = our_hashes[first_mismatch_tick];
            let oracle = expected_hashes[first_mismatch_tick];
            panic!(
                "{}: {mismatch_count}/{total} tick hashes differ from Oracle A; first mismatch at tick \
                 {first_mismatch_tick} (ours={ours:#x}, oracle={oracle:#x}) — recipe={recipe} seed={seed} \
                 ticks={ticks} chars={characters}",
                path.display(),
            );
        }

        // Field-level cross-check against the fixture's full `final_tick` state (the only tick a
        // fixture stores complete field data for, not just a hash) — belt-and-suspenders against
        // the (extremely unlikely, given the hash already matched) case of a hash collision
        // masking a real field mismatch.
        let final_rows_json = json["final_tick"]["rows"]
            .as_array()
            .unwrap_or_else(|| panic!("{}: missing 'final_tick.rows'", path.display()));
        let expected_final_rows: Vec<TraceRow> = final_rows_json
            .iter()
            .map(|row| TraceRow {
                input: parse_input(&row["input"]),
                state: parse_state(&row["state"]),
            })
            .collect();
        let expected_trace = Trace {
            metadata: dummy_metadata(),
            character_ids: character_ids.clone(),
            rows: vec![expected_final_rows],
        };
        let our_final_trace = Trace {
            metadata: dummy_metadata(),
            character_ids,
            rows: vec![our_trace.rows.last().unwrap().clone()],
        };
        let diff_result = diff(&our_final_trace, &expected_trace).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            diff_result.mismatch_count,
            0,
            "{}: final-tick field diff: {:?}",
            path.display(),
            diff_result.first_mismatch
        );

        scenarios_checked += 1;
        total_ticks += ticks as u64;
        total_character_ticks += ticks as u64 * characters as u64;
    }

    eprintln!(
        "parity_fixtures: {scenarios_checked} scenarios, {total_ticks} ticks, {total_character_ticks} character-ticks, 0 mismatches"
    );
}

/// Acceptance criterion 2.c ("the `f64` instantiation compiles and runs"), exercised over every
/// golden fixture scenario rather than only the one hand-written scenario in
/// `core_world`'s own `f64_instantiation_runs_a_full_scenario_without_panicking` test (review
/// round 2, finding F5) — same recipes/seeds/tuning as the `f32` parity test above, just run
/// through `CharacterCore<f64>`/`Collision<f64>`/`WorldCore<f64, _>` instead, with no
/// bit-exactness claim (see the task spec): the only assertions are "ran to completion without
/// panicking" and "produced no NaN position".
#[test]
fn f64_instantiation_runs_every_golden_fixture_without_panicking_or_nan() {
    let paths = fixture_paths();
    assert!(!paths.is_empty(), "no fixtures found in {}", fixtures_dir().display());

    let mut scenarios_checked = 0u64;

    for path in &paths {
        let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
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

        let mut scenario = random_v1(
            recipe,
            Params {
                seed,
                ticks,
                characters,
            },
        )
        .unwrap_or_else(|e| panic!("{}: random_v1({recipe}, seed={seed}) failed: {e}", path.display()));
        if let Some(true) = json["no_weak_hook"].as_bool() {
            scenario.no_weak_hook = true;
        }
        if let Some(overrides) = json["tuning_overrides"].as_array() {
            for o in overrides {
                scenario.tuning_overrides.push(TuningOverride {
                    name: o["name"].as_str().expect("tuning_overrides[].name").to_string(),
                    value_x100: o["value_x100"].as_i64().expect("tuning_overrides[].value_x100") as i32,
                });
            }
        }

        let map = ddai_trace::synthetic::build(recipe).unwrap_or_else(|| panic!("unknown recipe '{recipe}'"));
        let rows: Vec<Vec<(f64, f64)>> = run_scenario_generic::<f64>(&map, &scenario);

        assert_eq!(
            rows.len(),
            ticks as usize,
            "{}: f64 run produced the wrong tick count",
            path.display()
        );
        for (tick, tick_row) in rows.iter().enumerate() {
            for (slot, &(x, y)) in tick_row.iter().enumerate() {
                assert!(
                    !x.is_nan() && !y.is_nan(),
                    "{}: f64 physics produced a NaN position at tick {tick}, character slot {slot}",
                    path.display()
                );
            }
        }

        scenarios_checked += 1;
    }

    eprintln!(
        "f64_instantiation_runs_every_golden_fixture_without_panicking_or_nan: {scenarios_checked} scenarios, 0 panics, 0 NaNs"
    );
}
