//! Task 1.10b review R1 regression, Oracle A form: the reviewer's counterexample for the
//! `move_box` `test_box` skip (`Collision::swept_box_is_solid_free`). An 800x1500-tile map with a
//! solid wall column at tile 700 (x = 22400), one character spawned at (22039, 415) holding
//! direction +1 with `velramp_start = 1e7` (so the horizontal velocity ramp never crushes
//! `vel.x`), `air_control_speed = 4.90` and `gravity = 13.55`: the character falls at ~960
//! px/tick beside the wall with a small `vel.x`. There, with the earlier fixed 1 px pad, the skip
//! fired although the loop's float accumulation on the non-principal axis (`(n+1) * ulp(22380)/2`
//! with `n ~ 963`) drifts the box into the wall — 370 mismatched fields against the C++ Oracle A
//! (first: `pos_x` 22386 vs 22385 at tick 70) while the pre-change loop had 0.
//!
//! - `move_box_drift_scenario_matches_oracle_a_golden_hashes` (fast, always on): our per-tick state
//!   hashes for the scenario must equal the ones the real C++ Oracle A produced, stored in
//!   `fixtures_oracle_a/move_box_drift_bigwall.json` (regenerate with
//!   `generate_move_box_drift_golden`, `--ignored`, needs `tools/ddnet-oracle/build/oracle_core`).
//! - `move_box_drift_scenario_matches_live_oracle_a` (`--ignored`): runs the scenario through the
//!   live `oracle_core` and diffs field by field.

mod common;

use common::run_scenario;
use ddai_physics::map::{self, MapData, Tile};
use ddai_trace::generator::{Params, random_v1};
use ddai_trace::rawmap;
use ddai_trace::scenario::{MapRef, Scenario, ScenarioInput, TuningOverride};
use ddai_trace::trace::{Trace, diff};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const TICKS: u32 = 160;

fn bigwall_map() -> MapData {
    let (w, h, floor_row, wall_col) = (800i32, 1500i32, 1490i32, 700i32);
    let mut game = vec![
        Tile {
            index: map::TILE_AIR,
            ..Tile::default()
        };
        (w * h) as usize
    ];
    for y in 0..h {
        for x in 0..w {
            if x == 0 || x == w - 1 || y == 0 || y >= floor_row || x == wall_col {
                game[(y * w + x) as usize] = Tile {
                    index: map::TILE_SOLID,
                    ..Tile::default()
                };
            }
        }
    }
    MapData {
        width: w as u32,
        height: h as u32,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }
}

fn bigwall_scenario(map: &MapData) -> Scenario {
    let mut scn = random_v1(
        "arena",
        Params {
            seed: 1,
            ticks: TICKS,
            characters: 1,
        },
    )
    .unwrap();
    scn.characters[0].spawn_x = 22039;
    scn.characters[0].spawn_y = 415;
    scn.map_sha256 = ddai_trace::hash::sha256(&rawmap::write(map));
    scn.map_ref = MapRef::RawmapFile {
        path: "m.rawmap".into(),
    };
    for row in scn.inputs.iter_mut() {
        for inp in row.iter_mut() {
            *inp = ScenarioInput::default();
            inp.direction = 1;
            inp.aim_slot = -1;
            inp.target_x = 0;
            inp.target_y = 10;
        }
    }
    scn.tuning_overrides = vec![
        TuningOverride {
            name: "gravity".into(),
            value_x100: 1355,
        },
        TuningOverride {
            name: "velramp_start".into(),
            value_x100: 1_000_000_000,
        },
        TuningOverride {
            name: "air_control_speed".into(),
            value_x100: 490,
        },
    ];
    scn
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures_oracle_a/move_box_drift_bigwall.json")
}

fn oracle_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/ddnet-oracle/build/oracle_core")
}

/// Runs the live C++ Oracle A on the scenario; returns its recorded trace.
fn run_live_oracle(map: &MapData, scn: &Scenario) -> Trace {
    let oracle = oracle_path();
    assert!(
        oracle.is_file(),
        "{} not found — build it first (see the task spec)",
        oracle.display()
    );
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let work = std::env::temp_dir().join(format!("move-box-drift-{}-{unique}", std::process::id()));
    fs::create_dir_all(&work).unwrap();
    let (mp, sp, tp) = (work.join("m.rawmap"), work.join("s.bin"), work.join("t.bin"));
    fs::write(&mp, rawmap::write(map)).unwrap();
    fs::write(&sp, scn.write_bytes()).unwrap();
    let out = Command::new(&oracle).arg(&mp).arg(&sp).arg(&tp).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let trace = Trace::read_bytes(&fs::read(&tp).unwrap()).unwrap();
    let _ = fs::remove_dir_all(&work);
    trace
}

#[test]
fn move_box_drift_scenario_matches_oracle_a_golden_hashes() {
    let map = bigwall_map();
    let scn = bigwall_scenario(&map);
    let text = fs::read_to_string(golden_path()).expect("golden fixture missing — run generate_move_box_drift_golden");
    let json: Value = serde_json::from_str(&text).unwrap();
    let expected: Vec<u64> = json["tick_hashes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(expected.len(), TICKS as usize);

    let rows = run_scenario(&map, &scn);
    // `Trace::tick_hashes` only reads `rows`; the metadata is a placeholder.
    let trace = Trace {
        metadata: fixture_metadata(),
        character_ids: scn.characters.iter().map(|c| c.id).collect(),
        rows,
    };
    let ours = trace.tick_hashes();
    let first_bad = ours.iter().zip(&expected).position(|(a, b)| a != b);
    assert_eq!(
        first_bad, None,
        "per-tick state hashes diverge from the recorded Oracle A trace (first mismatching tick index shown)"
    );
}

fn fixture_metadata() -> ddai_trace::trace::TraceMetadata {
    use ddai_trace::trace::{DdnetRef, Producer, ScenarioRef, TraceMetadata, input_schema, state_schema};
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

#[test]
#[ignore]
fn move_box_drift_scenario_matches_live_oracle_a() {
    let map = bigwall_map();
    let scn = bigwall_scenario(&map);
    let rec = run_live_oracle(&map, &scn);
    let ours = Trace {
        metadata: rec.metadata.clone(),
        character_ids: rec.character_ids.clone(),
        rows: run_scenario(&map, &scn),
    };
    let d = diff(&ours, &rec).unwrap();
    assert_eq!(d.mismatch_count, 0, "first mismatch: {:?}", d.first_mismatch);
}

#[test]
#[ignore]
fn generate_move_box_drift_golden() {
    let map = bigwall_map();
    let scn = bigwall_scenario(&map);
    let rec = run_live_oracle(&map, &scn);
    let hashes = rec.tick_hashes();
    let json = serde_json::json!({
        "description": "Oracle A per-tick state hashes for tests/parity_move_box_drift.rs (task 1.10b review R1)",
        "ticks": TICKS,
        "tick_hashes": hashes,
    });
    fs::create_dir_all(golden_path().parent().unwrap()).unwrap();
    fs::write(golden_path(), serde_json::to_string_pretty(&json).unwrap() + "\n").unwrap();
}

/// Randomized Oracle A differential around the same map (`--ignored`; `DDAI_DRIFT_TRIES`
/// scenarios, default 400): the review's search space — spawn beside the wall, `dir = +1`, fast
/// falls from randomized gravity, `velramp_start` effectively disabled and a small
/// `air_control_speed` — plus randomized extra tuning (`air_control_accel`, ground/air friction,
/// jump impulse). Our `run_scenario` (which takes the `move_box` skip whenever it is provably safe)
/// must match the C++ Oracle A on every one: 0 mismatched fields.
#[test]
#[ignore]
fn move_box_skip_matches_live_oracle_a_on_random_large_coordinate_tunings() {
    let tries: u32 = std::env::var("DDAI_DRIFT_TRIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let map = bigwall_map();
    let mut mismatched_scenarios = 0u32;
    for k in 0..tries {
        let h = (k ^ 0x1B87_3593).wrapping_mul(2_654_435_761).rotate_left(7) ^ k.wrapping_mul(40_503);
        let mut scn = bigwall_scenario(&map);
        scn.characters[0].spawn_x = 21_000 + ((h >> 3).wrapping_mul(131) % 1_380) as i32;
        scn.characters[0].spawn_y = 100 + ((h >> 8).wrapping_mul(97) % 500) as i32;
        let dir = if h & 1 == 0 { 1 } else { -1 };
        for row in scn.inputs.iter_mut() {
            for inp in row.iter_mut() {
                inp.direction = dir;
            }
        }
        scn.tuning_overrides = vec![
            TuningOverride {
                name: "gravity".into(),
                value_x100: 1_000 + (h % 5_000) as i32,
            },
            TuningOverride {
                name: "velramp_start".into(),
                value_x100: 1_000_000_000,
            },
            TuningOverride {
                name: "air_control_speed".into(),
                value_x100: 100 + ((h >> 5) % 900) as i32,
            },
            TuningOverride {
                name: "air_control_accel".into(),
                value_x100: 50 + ((h >> 11) % 500) as i32,
            },
            TuningOverride {
                name: "ground_friction".into(),
                value_x100: 50 + ((h >> 14) % 100) as i32,
            },
        ];
        let rec = run_live_oracle(&map, &scn);
        let ours = Trace {
            metadata: rec.metadata.clone(),
            character_ids: rec.character_ids.clone(),
            rows: run_scenario(&map, &scn),
        };
        let d = diff(&ours, &rec).unwrap();
        if d.mismatch_count != 0 {
            mismatched_scenarios += 1;
            eprintln!(
                "scenario {k}: {} mismatched fields, first {:?}",
                d.mismatch_count, d.first_mismatch
            );
        }
    }
    eprintln!("{tries} scenarios, {mismatched_scenarios} mismatched");
    assert_eq!(mismatched_scenarios, 0);
}
