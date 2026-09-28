//! Bulk parity tests (acceptance criterion 2.b). `#[ignore]`d — run locally with:
//!
//! ```text
//! cargo test -p ddai-physics --release -- --ignored
//! ```
//!
//! Two tests:
//! - [`replays_the_existing_bulk_corpus`]: replays every trace already recorded under
//!   `~/aiddnet/data/traces/oracle-a/v1/` (task 1.2's `bulk_run.sh`) — regenerates each
//!   scenario from `manifest.tsv`'s recorded parameters, runs it through the Rust `f32` port,
//!   and compares against the recorded trace file, field-for-field.
//! - [`generates_and_replays_additional_scenarios_through_oracle_a`]: generates its own ≥2000
//!   fresh scenarios (all recipes, 1..=8 characters, 3000 ticks, both `no_weak_hook` values;
//!   some with the original fixed 3-override tuning, some with a random draw of extreme tuning
//!   values (velramp/elasticity/player_collision/player_hooking — review round 2, finding F1's
//!   regression slice), and roughly a third with sparse, unsorted, non-`0..N` character ids
//!   instead of `random_v1`'s own sequential ones — finding F5), runs each one through the
//!   *real* Oracle A binary (`tools/ddnet-oracle/build/oracle_core`, built by
//!   `fetch.sh`/`build.sh` — see the task spec) as a subprocess, and compares against the Rust
//!   port.
//!
//! Both report "N scenarios, T ticks, C character-ticks, 0 mismatches" on success and panic with
//! the first mismatching field on failure.

mod common;

use common::run_scenario;
use ddai_physics::map::MapData;
use ddai_trace::SplitMix64;
use ddai_trace::generator::{Params, random_v1};
use ddai_trace::rawmap;
use ddai_trace::scenario::{MapRef, Scenario, TuningOverride};
use ddai_trace::synthetic;
use ddai_trace::trace::{Trace, diff};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Resolves a scenario's `map_ref` back to a [`MapData`] — every scenario this file builds uses
/// [`MapRef::Recipe`] (`RawmapFile` is unused by any test here, so it's left unresolved).
fn build_map_ref(map_ref: &MapRef) -> MapData {
    match map_ref {
        MapRef::Recipe { name } => synthetic::build(name).unwrap_or_else(|| panic!("unknown recipe '{name}'")),
        MapRef::RawmapFile { path } => {
            panic!("this test suite never builds a RawmapFile-referencing scenario (got '{path}')")
        }
    }
}

/// Runs `scenario` through the Rust `f32` physics port and packages the result as a [`Trace`]
/// with the same shape as `recorded` (metadata/character_ids), ready for [`diff`].
fn our_trace_matching_shape_of(scenario: &Scenario, recorded: &Trace) -> Trace {
    let map = build_map_ref(&scenario.map_ref);
    let rows = run_scenario(&map, scenario);
    Trace {
        metadata: recorded.metadata.clone(),
        character_ids: recorded.character_ids.clone(),
        rows,
    }
}

/// Compares `ours` against `recorded` (already read from a trace file) and panics with a
/// descriptive message (first mismatching field, plus a summary count) if they differ.
fn assert_traces_match(ours: &Trace, recorded: &Trace, context: &str) {
    let result = diff(ours, recorded).unwrap_or_else(|e| panic!("{context}: trace shape mismatch: {e}"));
    assert_eq!(
        result.mismatch_count, 0,
        "{context}: {} field mismatches, first: {:?}",
        result.mismatch_count, result.first_mismatch
    );
}

fn traces_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set");
    PathBuf::from(home).join("aiddnet/data/traces/oracle-a/v1")
}

fn oracle_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/ddnet-oracle/build/oracle_core")
}

/// Rebuilds the exact `Scenario` a `bulk_run.sh` manifest row describes: `random_v1` with the
/// row's `recipe`/`seed`/`ticks`/`characters`, then the CLI flags `variant` implies (mirrors
/// `bulk_run.sh`'s `run_one` exactly: `main` = no flags, `nwh` = `--no-weak-hook`, `tune` =
/// the three named `--tune` overrides).
fn scenario_for_manifest_row(recipe: &str, seed: u64, ticks: u32, characters: u32, variant: &str) -> Scenario {
    let mut scenario = random_v1(
        recipe,
        Params {
            seed,
            ticks,
            characters,
        },
    )
    .unwrap_or_else(|e| panic!("random_v1({recipe}, seed={seed}, ticks={ticks}, chars={characters}) failed: {e}"));
    match variant {
        "main" => {}
        "nwh" => scenario.no_weak_hook = true,
        "tune" => scenario.tuning_overrides = tune_overrides(),
        other => panic!("unknown manifest variant '{other}'"),
    }
    scenario
}

/// The tuning overrides `bulk_run.sh`'s `tune` variant and this file's own generated "some with
/// tuning overrides" cases both use: non-default, but realistic enough that characters still
/// fall and hook each other (review round 2, finding F11 — see `bulk_run.sh`'s comment).
fn tune_overrides() -> Vec<TuningOverride> {
    vec![
        TuningOverride {
            name: "gravity".to_string(),
            value_x100: 40,
        },
        TuningOverride {
            name: "hook_length".to_string(),
            value_x100: 50000,
        },
        TuningOverride {
            name: "hook_drag_speed".to_string(),
            value_x100: 1800,
        },
    ]
}

#[test]
#[ignore]
fn replays_the_existing_bulk_corpus() {
    let dir = traces_dir();
    let manifest_path = dir.join("manifest.tsv");
    let manifest = fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        panic!(
            "failed to read {} — run tools/ddnet-oracle/bulk_run.sh first: {e}",
            manifest_path.display()
        )
    });

    let mut lines = manifest.lines();
    let header = lines.next().expect("manifest.tsv must have a header row");
    assert_eq!(
        header, "recipe\tseed\tticks\tcharacters\tvariant\ttrace_file\ttrace_sha256\tscenario_sha256",
        "manifest.tsv header changed shape — update this test's column parsing"
    );

    let mut scenarios_checked = 0u64;
    let mut total_ticks = 0u64;
    let mut total_character_ticks = 0u64;

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 8, "malformed manifest row: {line}");
        let recipe = cols[0];
        let seed: u64 = cols[1].parse().unwrap();
        let ticks: u32 = cols[2].parse().unwrap();
        let characters: u32 = cols[3].parse().unwrap();
        let variant = cols[4];
        let trace_file = cols[5];

        let scenario = scenario_for_manifest_row(recipe, seed, ticks, characters, variant);
        let recorded_bytes =
            fs::read(dir.join(trace_file)).unwrap_or_else(|e| panic!("failed to read {trace_file}: {e}"));
        let recorded =
            Trace::read_bytes(&recorded_bytes).unwrap_or_else(|e| panic!("failed to parse {trace_file}: {e}"));

        let ours = our_trace_matching_shape_of(&scenario, &recorded);
        assert_traces_match(
            &ours,
            &recorded,
            &format!("{trace_file} (recipe={recipe} seed={seed} variant={variant})"),
        );

        scenarios_checked += 1;
        total_ticks += ticks as u64;
        total_character_ticks += ticks as u64 * characters as u64;
    }

    assert!(
        scenarios_checked >= 200,
        "expected at least 200 scenarios in the bulk corpus, found {scenarios_checked}"
    );
    eprintln!(
        "replays_the_existing_bulk_corpus: {scenarios_checked} scenarios, {total_ticks} ticks, \
         {total_character_ticks} character-ticks, 0 mismatches"
    );
}

/// Deterministic, sparse, widely-spread and deliberately non-monotonic character id sets
/// (every value `< MAX_CLIENTS`, all 8 entries in a pool distinct), used in place of
/// `random_v1`'s own sequential `0..N` ids on part of the generated slice below (review round 2,
/// finding F5): `WorldCore`/`TeamsCore`'s array-index-vs-id bookkeeping (see finding F3) is only
/// exercised meaningfully when ids are sparse and out of order, which `random_v1` alone never
/// produces.
const ID_POOLS: [[u32; 8]; 5] = [
    [7, 3, 100, 1, 55, 21, 9, 127],
    [126, 0, 64, 2, 90, 45, 11, 5],
    [50, 51, 4, 99, 1, 0, 123, 60],
    [3, 125, 8, 40, 15, 2, 77, 33],
    [10, 5, 111, 20, 1, 88, 6, 45],
];

/// The exact tuning parameters and `value_x100` ranges that reproduced review round 2's finding
/// F1 (extreme `velramp_range`/`velramp_start`/`velramp_curvature` driving `VelocityRamp` to
/// `+inf`, then `0.0 * inf = NaN` through `Move()`, exposing a float→int conversion divergence
/// from x86 `cvttss2si` — see `Real::to_i32_trunc`'s doc comment). Mirrors the reviewer's own
/// differential fuzzer's `TUNES` table so this slice keeps regression-testing exactly the
/// combination that found the bug.
const EXTREME_TUNES: [(&str, i32, i32); 17] = [
    ("gravity", -50, 200),
    ("ground_control_speed", 0, 3000),
    ("ground_control_accel", 0, 1000),
    ("ground_friction", 0, 120),
    ("ground_jump_impulse", 0, 4000),
    ("air_jump_impulse", 0, 4000),
    ("air_control_speed", 0, 2000),
    ("air_control_accel", 0, 500),
    ("air_friction", 0, 110),
    ("hook_length", 0, 150000),
    ("hook_fire_speed", 0, 30000),
    ("hook_drag_accel", 0, 3000),
    ("hook_drag_speed", 0, 6000),
    ("velramp_start", 0, 200000),
    ("velramp_range", 100, 400000),
    ("velramp_curvature", 101, 500),
    ("hook_duration", 0, 400),
];

/// Draws a random combination of 1..=6 `EXTREME_TUNES` entries, plus (each with some
/// probability) `player_collision`/`player_hooking` forced to `0` or a small value, and
/// `ground_elasticity_x`/`_y` at the extremes of their valid range — the same shape of tuning
/// override set the reviewer's differential fuzzer draws (finding F1's regression coverage),
/// deterministically from `seed` so this whole test stays reproducible.
fn random_extreme_tuning(seed: u64) -> Vec<TuningOverride> {
    let mut rng = SplitMix64::new(seed);
    let mut overrides: Vec<TuningOverride> = Vec::new();
    let k = rng.range_inclusive(1, 6);
    for _ in 0..k {
        let (name, lo, hi) = EXTREME_TUNES[rng.below(EXTREME_TUNES.len() as u32) as usize];
        if overrides.iter().any(|t| t.name == name) {
            continue;
        }
        overrides.push(TuningOverride {
            name: name.to_string(),
            value_x100: rng.range_inclusive(lo, hi),
        });
    }
    if rng.chance(1, 5) {
        overrides.push(TuningOverride {
            name: "player_collision".to_string(),
            value_x100: if rng.chance(1, 2) {
                0
            } else {
                rng.range_inclusive(1, 300)
            },
        });
    }
    if rng.chance(1, 5) {
        overrides.push(TuningOverride {
            name: "player_hooking".to_string(),
            value_x100: if rng.chance(1, 2) {
                0
            } else {
                rng.range_inclusive(1, 300)
            },
        });
    }
    if rng.chance(1, 4) {
        overrides.push(TuningOverride {
            name: "ground_elasticity_x".to_string(),
            value_x100: rng.range_inclusive(-200, 200),
        });
        overrides.push(TuningOverride {
            name: "ground_elasticity_y".to_string(),
            value_x100: rng.range_inclusive(-200, 200),
        });
    }
    overrides
}

/// One freshly generated scenario's identity: enough to reconstruct it deterministically
/// (`random_v1` + the tuning overrides/id remap to apply on top).
struct GeneratedCase {
    recipe: &'static str,
    seed: u64,
    characters: u32,
    no_weak_hook: bool,
    tuning_overrides: Vec<TuningOverride>,
    /// `Some(pool)`: overwrite `random_v1`'s sequential `0..characters` ids with
    /// `pool[..characters]` (finding F5). `None`: keep `random_v1`'s own sequential ids.
    sparse_ids: Option<[u32; 8]>,
}

/// `count` cases spanning all 4 recipes evenly, cycling `characters` through 1..=8, alternating
/// `no_weak_hook`, and varying tuning overrides and character ids — covers every axis the
/// acceptance criterion names ("all recipes, 1–4 chars [now widened to 1–8, finding F5], ...
/// both no_weak_hook values, some with tuning overrides") plus review round 2's findings F1
/// (random extreme tuning, not just the original fixed 3-override set) and F5 (sparse/unsorted
/// ids, up to 8 characters) — with a single deterministic seed stream per recipe.
fn generate_cases(count: usize) -> Vec<GeneratedCase> {
    const RECIPES: [&str; 4] = ["arena", "freeze", "front", "tele-speedup"];
    let per_recipe = count.div_ceil(RECIPES.len());
    let mut cases = Vec::with_capacity(per_recipe * RECIPES.len());
    for recipe in RECIPES {
        for i in 0..per_recipe {
            // Seeds starting at 10_000 stay disjoint from bulk_run.sh's own ranges (1..=50,
            // 1001..=1005, 2001..=2005) and the fixtures' seeds (101/102/201/202).
            let seed = 10_000 + i as u64;
            let tuning_overrides = match i % 10 {
                0 => tune_overrides(),
                5 => random_extreme_tuning(seed ^ 0xA5A5_5A5A_5A5A_5A5A),
                _ => Vec::new(),
            };
            // Task 1.3 review round 2, finding F8 (fixed in task 1.6): `i % 10 == 5` (the
            // extreme-tuning branch above) only ever lands on odd `i`, so `no_weak_hook: i % 2
            // == 0` was `false` for *every* extreme-tuning case — no scenario ever exercised
            // `no_weak_hook=true` together with extreme tuning (an index-parity coincidence
            // between the two `% ` selectors, not an intentional exclusion). `(i / 2) % 2` is
            // uncorrelated with `i % 10 == 5`'s parity, so the extreme-tuning cases now cover
            // both `no_weak_hook` values as intended; other cases keep the original `i % 2`
            // derivation (no reason to change what already worked for them).
            let no_weak_hook = if i % 10 == 5 { (i / 2) % 2 == 0 } else { i % 2 == 0 };
            cases.push(GeneratedCase {
                recipe,
                seed,
                characters: 1 + (i as u32 % 8),
                no_weak_hook,
                tuning_overrides,
                sparse_ids: if i % 3 == 0 {
                    Some(ID_POOLS[i % ID_POOLS.len()])
                } else {
                    None
                },
            });
        }
    }
    cases
}

#[test]
#[ignore]
fn generates_and_replays_additional_scenarios_through_oracle_a() {
    const TICKS: u32 = 3000;
    const NUM_SCENARIOS: usize = 2000;

    let oracle = oracle_binary();
    assert!(
        oracle.is_file(),
        "{} not found — run `cd tools/ddnet-oracle && ./fetch.sh && ./build.sh` first (see the task spec)",
        oracle.display()
    );

    let work_dir = std::env::temp_dir().join(format!("ddai-physics-parity-bulk-{}", std::process::id()));
    fs::create_dir_all(&work_dir).unwrap();

    // One rawmap file per recipe, reused across every scenario on that recipe (the map itself
    // doesn't depend on seed/characters/tuning).
    let mut map_paths: HashMap<&str, PathBuf> = HashMap::new();
    for recipe in ["arena", "freeze", "front", "tele-speedup"] {
        let map = synthetic::build(recipe).unwrap();
        let path = work_dir.join(format!("{recipe}.rawmap"));
        fs::write(&path, rawmap::write(&map)).unwrap();
        map_paths.insert(recipe, path);
    }

    let cases = generate_cases(NUM_SCENARIOS);
    assert!(
        cases.len() >= NUM_SCENARIOS,
        "expected >= {NUM_SCENARIOS} generated cases, got {}",
        cases.len()
    );

    let mut scenarios_checked = 0u64;
    let mut total_character_ticks = 0u64;
    let scn_path = work_dir.join("scn.bin");
    let trace_path = work_dir.join("trace.bin");

    for (i, case) in cases.iter().enumerate() {
        let mut scenario = random_v1(
            case.recipe,
            Params {
                seed: case.seed,
                ticks: TICKS,
                characters: case.characters,
            },
        )
        .unwrap_or_else(|e| panic!("random_v1 failed for case {i}: {e}"));
        scenario.no_weak_hook = case.no_weak_hook;
        if !case.tuning_overrides.is_empty() {
            scenario.tuning_overrides = case.tuning_overrides.clone();
        }
        if let Some(pool) = case.sparse_ids {
            for (c, &id) in scenario.characters.iter_mut().zip(pool.iter()) {
                c.id = id;
            }
        }

        fs::write(&scn_path, scenario.write_bytes()).unwrap();
        let map_path = &map_paths[case.recipe];

        let output = Command::new(&oracle)
            .arg(map_path)
            .arg(&scn_path)
            .arg(&trace_path)
            .output()
            .unwrap_or_else(|e| panic!("failed to run oracle_core for case {i}: {e}"));
        assert!(
            output.status.success(),
            "oracle_core failed for case {i} (recipe={} seed={} chars={} nwh={}): {}",
            case.recipe,
            case.seed,
            case.characters,
            case.no_weak_hook,
            String::from_utf8_lossy(&output.stderr)
        );

        let recorded_bytes = fs::read(&trace_path).unwrap();
        let recorded = Trace::read_bytes(&recorded_bytes)
            .unwrap_or_else(|e| panic!("failed to parse oracle output for case {i}: {e}"));

        let ours = our_trace_matching_shape_of(&scenario, &recorded);
        assert_traces_match(
            &ours,
            &recorded,
            &format!(
                "generated case {i} (recipe={} seed={} chars={} nwh={} tune_overrides={} ids={:?})",
                case.recipe,
                case.seed,
                case.characters,
                case.no_weak_hook,
                scenario.tuning_overrides.len(),
                scenario.characters.iter().map(|c| c.id).collect::<Vec<_>>(),
            ),
        );

        scenarios_checked += 1;
        total_character_ticks += TICKS as u64 * case.characters as u64;
    }

    let _ = fs::remove_dir_all(&work_dir);

    assert!(scenarios_checked >= NUM_SCENARIOS as u64);
    eprintln!(
        "generates_and_replays_additional_scenarios_through_oracle_a: {scenarios_checked} scenarios, \
         {} ticks, {total_character_ticks} character-ticks, 0 mismatches",
        scenarios_checked * TICKS as u64
    );
}

/// Task 1.3 review round 2, finding F8 (fixed in task 1.6): pins that the extreme-tuning cases
/// (`i % 10 == 5`) cover both `no_weak_hook` values, not just one — this is a lightweight,
/// always-run guard (no C++ oracle required) for the `generate_cases` fix above; the actual
/// parity coverage claim still comes from `generates_and_replays_additional_scenarios_through_oracle_a`
/// (`#[ignore]`, requires the oracle binary).
#[test]
fn extreme_tuning_cases_cover_both_no_weak_hook_values() {
    let cases = generate_cases(2000);
    let extreme_tuned: Vec<&GeneratedCase> = cases
        .iter()
        .filter(|c| !c.tuning_overrides.is_empty() && c.tuning_overrides.len() > 3)
        .collect();
    assert!(
        !extreme_tuned.is_empty(),
        "expected at least one extreme-tuning case (i % 10 == 5) in the first 2000"
    );
    assert!(
        extreme_tuned.iter().any(|c| c.no_weak_hook),
        "no extreme-tuning case has no_weak_hook=true — F8 regressed"
    );
    assert!(
        extreme_tuned.iter().any(|c| !c.no_weak_hook),
        "no extreme-tuning case has no_weak_hook=false"
    );
}
