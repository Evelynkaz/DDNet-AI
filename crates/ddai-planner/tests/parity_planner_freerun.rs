//! Acceptance criterion 3's "free-running" check: unlike `tests/parity_planner.rs` (which resets
//! the `Planner` before every single decision), this replays one *continuous* game dumped by
//! `tools/ts-trace/gen-planner-freerun.mjs` -- one `Planner`, `set_search_seed()` once, deciding
//! every tick against the real, live world for the whole game -- so hidden cross-decision state
//! (`warm`/`committed`/`commitLeft`/`decideGaps`/`dirSince`/`oppSeed`/the CEM `Rng`/`thawScratch`+
//! `thawMemo`) must persist and evolve identically to TS's, not just any one isolated call.
//!
//! `#[ignore]`d: needs both the `ts-parity` feature and a locally-generated dump file. Run:
//! ```text
//! node tools/ts-trace/gen-planner-freerun.mjs --map ... --out ~/aiddnet/data/traces/planner-freerun/x.jsonl ...
//! DDAI_PLANNER_FREERUN_DIR=~/aiddnet/data/traces/planner-freerun \
//!   cargo test -p ddai-planner --features ts-parity --release --test parity_planner_freerun -- --ignored --nocapture
//! ```

#![cfg(feature = "ts-parity")]

use ddai_jsmath::Rng;
use ddai_planner::config::{OpponentModel, PlannerConfig};
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::planner::Planner;
use ddai_planner::scripted::scripted_action;
use ddai_planner::types::empty_input;
use ddai_planner::vmath::Vec2;
use serde::Deserialize;
use std::path::PathBuf;

fn bits_to_f64(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("hex f64 bits"))
}

#[derive(Deserialize)]
struct InputJson {
    direction: i32,
    #[serde(rename = "targetX")]
    target_x: String,
    #[serde(rename = "targetY")]
    target_y: String,
    jump: i32,
    fire: i32,
    hook: i32,
    #[serde(rename = "playerFlags")]
    player_flags: i32,
    #[serde(rename = "wantedWeapon")]
    wanted_weapon: i32,
    #[serde(rename = "nextWeapon")]
    next_weapon: i32,
    #[serde(rename = "prevWeapon")]
    prev_weapon: i32,
}

#[derive(Deserialize)]
struct LastInfoJson {
    searched: bool,
    candidates: i32,
    #[serde(rename = "outOfTime")]
    out_of_time: bool,
    #[serde(rename = "hookAt")]
    hook_at: i32,
    gated: bool,
    shielded: bool,
    #[serde(rename = "edgeHeld")]
    edge_held: bool,
    #[serde(rename = "selfOut")]
    self_out: i32,
    #[serde(rename = "enemyOut")]
    enemy_out: i32,
}

#[derive(Deserialize)]
struct SpawnJson {
    x: f64,
    y: f64,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
// Test-only deserialization enum, one value per JSON line -- not a hot path, so the size gap
// between variants (`Meta` vs the data-heavy `Case`/`Decision`) is not worth boxing fields over.
#[allow(clippy::large_enum_variant)]
enum Line {
    Meta {
        #[serde(rename = "mapPath")]
        map_path: String,
        #[serde(rename = "mapSha256")]
        map_sha256: String,
        seed: u32,
        preset: String,
        opponent: String,
        #[serde(rename = "selfSpawn")]
        self_spawn: SpawnJson,
        #[serde(rename = "enemySpawn")]
        enemy_spawn: SpawnJson,
    },
    Decision {
        tick: i64,
        decision: InputJson,
        #[serde(rename = "lastInfo")]
        last_info: LastInfoJson,
    },
}

fn build_cfg(preset: &str, opponent: &str) -> PlannerConfig {
    use ddai_planner::config::{preset_low_cpu, preset_normal, preset_strong_wb, wb_overrides};
    let mut cfg = match preset {
        "normal" => preset_normal(),
        "low" => preset_low_cpu(),
        "strong" => preset_strong_wb(preset_normal()),
        "wb" => wb_overrides(preset_normal()),
        other => panic!("unknown preset {other}"),
    };
    match opponent {
        "hold" => {}
        "react" => cfg.opponent_model = OpponentModel::React,
        "mix" => cfg.opponent_mix = true,
        other => panic!("unknown opponent model {other}"),
    }
    cfg.budget_ms = 0.0;
    cfg.hard_ms = 0.0;
    cfg
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn replay_file(path: &std::path::Path) -> (usize, Option<String>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());

    let meta_line = lines.next().expect("empty dump file");
    let Line::Meta {
        map_path,
        map_sha256,
        seed,
        preset,
        opponent,
        self_spawn,
        enemy_spawn,
    } = serde_json::from_str(meta_line).unwrap_or_else(|e| panic!("{}: bad meta line: {e}", path.display()))
    else {
        panic!("first line must be a meta line");
    };

    let bytes = std::fs::read(&map_path).unwrap_or_else(|e| panic!("reading map {map_path}: {e}"));
    let actual_sha = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(&bytes);
        hex_encode(&h.finalize())
    };
    assert_eq!(
        actual_sha, map_sha256,
        "map file changed since the dump was generated: {map_path}"
    );
    let loaded = ddai_tsworld::load_map_bytes(&bytes).unwrap_or_else(|e| panic!("loading map {map_path}: {e:?}"));

    let cfg = build_cfg(&preset, &opponent);
    let mut planner: Planner<ddai_tsworld::SimWorld> = Planner::new(cfg);
    planner.set_search_seed(seed);

    let mut world = ddai_tsworld::SimWorld::new(
        loaded.collision,
        ddai_tsworld::world::SimWorldOptions {
            respawn_delay_ticks: Some(0),
            infinite_ammo: Some(true),
            sv_hit: Some(true),
            all_weapons: None,
            no_weak_hook: None,
        },
    );
    <ddai_tsworld::SimWorld as PlanWorld>::add_tee(
        &mut world,
        0,
        Vec2 {
            x: self_spawn.x,
            y: self_spawn.y,
        },
    );
    <ddai_tsworld::SimWorld as PlanWorld>::add_tee(
        &mut world,
        1,
        Vec2 {
            x: enemy_spawn.x,
            y: enemy_spawn.y,
        },
    );

    // `new Rng((seed * 7919 + 17) >>> 0)` (gen-planner-freerun.mjs) -- exact in JS f64 for any
    // `u32 seed` (well under 2^53), so computing in `u64` then truncating to `u32` is bit-exact.
    let opp_seed = ((u64::from(seed) * 7919 + 17) & 0xFFFF_FFFF) as u32;
    let mut opp_rng = Rng::new(opp_seed);

    let mut self_prev = empty_input();
    let mut enemy_prev = empty_input();
    let mut total = 0usize;
    let mut first_mismatch = None;

    for line in lines {
        let Line::Decision {
            tick,
            decision,
            last_info,
        } = serde_json::from_str(line).unwrap_or_else(|e| panic!("{}: bad decision line: {e}\n{line}", path.display()))
        else {
            panic!("expected a decision line");
        };
        total += 1;
        let enemy_input = scripted_action(&world, 1, 0, &enemy_prev, &mut opp_rng);
        planner.set_live_tick(world.tick());
        let self_input = planner.decide(&mut world, 0, 1, self_prev, enemy_input);
        let li = planner.last_info;

        let want_target = (bits_to_f64(&decision.target_x), bits_to_f64(&decision.target_y));
        // Review round 1, F11: compare every `PlayerInput` field, not a subset.
        let got_ok = self_input.direction == decision.direction
            && self_input.target_x.to_bits() == want_target.0.to_bits()
            && self_input.target_y.to_bits() == want_target.1.to_bits()
            && self_input.jump == decision.jump
            && self_input.fire == decision.fire
            && self_input.hook == decision.hook
            && self_input.wanted_weapon == decision.wanted_weapon
            && self_input.player_flags == decision.player_flags
            && self_input.next_weapon == decision.next_weapon
            && self_input.prev_weapon == decision.prev_weapon;
        let info_ok = li.searched == last_info.searched
            && li.candidates == last_info.candidates
            && li.out_of_time == last_info.out_of_time
            && li.hook_at == last_info.hook_at
            && li.gated == last_info.gated
            && li.shielded == last_info.shielded
            && li.edge_held == last_info.edge_held
            && li.self_out == last_info.self_out
            && li.enemy_out == last_info.enemy_out;
        if (!got_ok || !info_ok) && first_mismatch.is_none() {
            first_mismatch = Some(format!(
                "{}: first mismatch at tick {tick} (decision #{total}): got {self_input:?} info(candidates={},hookAt={}) want direction={} target={:?} jump={} fire={} hook={} wanted_weapon={} info(candidates={},hookAt={})",
                path.display(),
                li.candidates,
                li.hook_at,
                decision.direction,
                want_target,
                decision.jump,
                decision.fire,
                decision.hook,
                decision.wanted_weapon,
                last_info.candidates,
                last_info.hook_at,
            ));
        }

        <ddai_tsworld::SimWorld as PlanWorld>::set_input(&mut world, 0, self_input);
        <ddai_tsworld::SimWorld as PlanWorld>::set_input(&mut world, 1, enemy_input);
        let _ = <ddai_tsworld::SimWorld as PlanWorld>::step(&mut world);
        self_prev = self_input;
        enemy_prev = enemy_input;
    }
    (total, first_mismatch)
}

#[test]
#[ignore = "needs a locally-generated free-running dump directory, see this file's doc comment"]
fn planner_matches_ts_over_a_continuous_game() {
    let dir = std::env::var("DDAI_PLANNER_FREERUN_DIR")
        .expect("set DDAI_PLANNER_FREERUN_DIR to a directory of gen-planner-freerun.mjs output");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .jsonl files in {dir}");

    let mut total = 0usize;
    let mut mismatches = 0usize;
    for f in &files {
        let (n, mism) = replay_file(f);
        total += n;
        if let Some(m) = mism {
            eprintln!("{m}");
            mismatches += 1;
        } else {
            eprintln!("{}: {n} decisions, 0 mismatches", f.display());
        }
    }
    eprintln!(
        "replayed {total} decisions across {} free-running games, {mismatches} with a mismatch",
        files.len()
    );
    assert_eq!(
        mismatches, 0,
        "{mismatches} free-running game(s) diverged from TS -- see stderr above"
    );
}
