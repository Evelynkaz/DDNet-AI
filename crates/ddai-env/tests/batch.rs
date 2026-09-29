//! Batches: reproducibility at any thread count, run-to-run identical summaries, output files,
//! and the shipped arena/scenario data.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ddai_env::arena::{Arena, Split, load_arena_defs};
use ddai_env::config::{RunConfig, builtin_brain};
use ddai_env::output::{Line, RunOptions, run_config};
use ddai_env::report::summarize;
use ddai_env::run::{load_arenas, run_condition};

fn repo(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel)
}

fn map_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join("aiddnet/data/maps"))
        .unwrap_or_default()
}

const CFG: &str = r#"
name = "batch-test"
base_seed = 100
games = 16

[rules]
max_ticks = 240
after_ticks = 60

[[condition]]
name = "pit scripted"
arena = "pit"
players = [{ brain = "scripted" }, { brain = "scripted" }]

[[condition]]
name = "platform planner-vs-scripted"
arena = "platform"
games = 6
players = [{ brain = "planner" }, { brain = "scripted" }]

[[condition]]
name = "pit lagged 1v2"
arena = "pit"
players = [{ brain = "scripted", lag = 3 }, { brain = "scripted" }, { brain = "scripted" }]
"#;

fn arenas(cfg: &RunConfig) -> BTreeMap<String, Arena> {
    load_arenas(cfg, &repo("configs/arenas"), &map_dir()).unwrap()
}

/// One JSONL line per game with the timing fields removed: what must be identical.
fn lines_without_timing(run: &ddai_env::run::ConditionRun) -> Vec<String> {
    run.games
        .iter()
        .enumerate()
        .map(|(g, report)| {
            let line = Line {
                condition: &run.condition.name,
                game: g as u32,
                report,
            };
            let mut v = serde_json::to_value(&line).unwrap();
            v.as_object_mut()
                .unwrap()
                .remove("timing")
                .expect("timing is a separate field");
            serde_json::to_string(&v).unwrap()
        })
        .collect()
}

#[test]
fn jsonl_is_identical_at_1_and_6_threads() {
    let cfg = RunConfig::parse(CFG).unwrap();
    let arenas = arenas(&cfg);
    for cond in &cfg.condition {
        let arena = &arenas[&cond.arena];
        let games = cfg.games_for(cond);
        let one = run_condition(&cfg, cond, arena, games, &builtin_brain, 1).unwrap();
        let six = run_condition(&cfg, cond, arena, games, &builtin_brain, 6).unwrap();
        let again = run_condition(&cfg, cond, arena, games, &builtin_brain, 6).unwrap();
        assert_eq!(lines_without_timing(&one), lines_without_timing(&six), "{}", cond.name);
        assert_eq!(
            lines_without_timing(&six),
            lines_without_timing(&again),
            "{}",
            cond.name
        );
        assert_eq!(one.games.len(), games as usize);
        // The two runs also agree on the aggregate numbers.
        let sa = summarize(&one, "train", None).deterministic_view();
        let sb = summarize(&six, "train", None).deterministic_view();
        assert_eq!(sa, sb, "{}", cond.name);
    }
}

#[test]
fn different_base_seeds_give_different_games() {
    let cfg = RunConfig::parse(CFG).unwrap();
    let arenas = arenas(&cfg);
    let cond = &cfg.condition[0];
    let a = run_condition(&cfg, cond, &arenas["pit"], 8, &builtin_brain, 2).unwrap();
    let mut other = cfg.clone();
    other.base_seed = 5_000;
    let b = run_condition(&other, cond, &arenas["pit"], 8, &builtin_brain, 2).unwrap();
    assert_ne!(lines_without_timing(&a), lines_without_timing(&b));
}

#[test]
fn run_config_writes_jsonl_summary_and_markdown() {
    let cfg = RunConfig::parse(CFG).unwrap();
    let arenas = arenas(&cfg);
    let dir = tempfile::tempdir().unwrap();
    let opts = RunOptions {
        threads: 2,
        out_dir: Some(dir.path()),
        filter: Some("scripted"),
        git: ("deadbeef".into(), true),
        stall_baseline: None,
    };
    let mut progress = Vec::new();
    let summary = run_config(&cfg, &arenas, &builtin_brain, &opts, &mut |l| {
        progress.push(l.to_string())
    })
    .unwrap();
    // "pit scripted" and "platform planner-vs-scripted" match the filter; the lagged one does not.
    assert_eq!(summary.conditions.len(), 2);
    assert_eq!(progress.len(), 2);
    assert_eq!(summary.meta.config_hash, cfg.hash());
    assert_eq!(summary.meta.git_commit, "deadbeef");
    assert!(summary.meta.git_dirty);
    for name in [
        "pit-scripted.jsonl",
        "platform-planner-vs-scripted.jsonl",
        "summary.json",
        "summary.md",
    ] {
        assert!(dir.path().join(name).exists(), "{name} missing");
    }
    let jsonl = std::fs::read_to_string(dir.path().join("pit-scripted.jsonl")).unwrap();
    assert_eq!(jsonl.lines().count(), 16);
    let first: serde_json::Value = serde_json::from_str(jsonl.lines().next().unwrap()).unwrap();
    for key in [
        "condition",
        "game",
        "seed",
        "result",
        "end_tick",
        "credited",
        "held",
        "victim",
        "players",
        "timing",
    ] {
        assert!(first.get(key).is_some(), "JSONL line lacks {key}");
    }
    assert!(first["players"][0]["hash"].as_str().is_some_and(|h| h.len() == 16));
    assert!(first["timing"]["decide_us_p50"].is_array() && first["timing"]["decide_us_p99"].is_array());
    let md = std::fs::read_to_string(dir.path().join("summary.md")).unwrap();
    assert!(md.contains("Винрейт") && md.contains("W:L:D:T"), "{md}");
    assert!(md.contains(&cfg.hash()));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("summary.json")).unwrap()).unwrap();
    assert!(json["conditions"][0]["win_rate"].is_object() || json["conditions"][0]["win_rate"].is_null());
    assert_eq!(
        json["conditions"][0]["tally"]["w"].as_u64().unwrap()
            + json["conditions"][0]["tally"]["l"].as_u64().unwrap()
            + json["conditions"][0]["tally"]["d"].as_u64().unwrap()
            + json["conditions"][0]["tally"]["t"].as_u64().unwrap(),
        16
    );
}

#[test]
fn lag_changes_play_but_not_the_game_count() {
    let cfg = RunConfig::parse(CFG).unwrap();
    let arenas = arenas(&cfg);
    let cond = cfg.condition.iter().find(|c| c.name == "pit lagged 1v2").unwrap();
    let run = run_condition(&cfg, cond, &arenas["pit"], 8, &builtin_brain, 2).unwrap();
    assert_eq!(run.games.len(), 8);
    assert!(run.games.iter().all(|g| g.players.len() == 3 && g.players[0].lag == 3));
}

#[test]
fn shipped_synthetic_arenas_load_and_spawn() {
    let defs = load_arena_defs(&repo("configs/arenas")).unwrap();
    for name in ["pit", "platform"] {
        let arena = Arena::build(&defs[name], &map_dir()).unwrap();
        assert_eq!(arena.tag, Split::Train);
        for seed in 0..20 {
            assert_eq!(arena.spawn_tiles(seed, 2).unwrap().len(), 2);
        }
    }
    // Tags: two train arenas + clb-left are train, the mirror hall and the second map are holdouts.
    assert_eq!(defs["clb-left"].tag, Split::Train);
    assert_eq!(defs["clb-right"].tag, Split::Holdout);
    assert_eq!(defs["chillblock5-ruler"].tag, Split::Holdout);
}

/// Real-map arenas: verified by sha256, and the CLB halls reproduce the harness's standing-slot
/// count. Skipped (with a note) when the local map files are not present, e.g. on CI.
#[test]
fn real_map_arenas_match_their_declared_maps() {
    let defs = load_arena_defs(&repo("configs/arenas")).unwrap();
    let clb = map_dir()
        .join("copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map");
    if !clb.exists() {
        eprintln!("skipping: {} not present", clb.display());
        return;
    }
    let left = Arena::build(&defs["clb-left"], &map_dir()).unwrap();
    let right = Arena::build(&defs["clb-right"], &map_dir()).unwrap();
    assert_eq!(
        left.map_sha256.as_deref(),
        Some("6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25")
    );
    // The harness's left hall has 25 standing tiles inside L1 + L2 (same rule, same boxes).
    assert_eq!(left.slot_count(), 25);
    assert_eq!(right.slot_count(), 25);
    for seed in 0..50 {
        let s = left.spawn_tiles(seed, 2).unwrap();
        assert!(s.iter().all(|t| left.slots().contains(t)));
        let d = f64::from(s[0].0 - s[1].0).hypot(f64::from(s[0].1 - s[1].1));
        assert!((3.0..=12.0).contains(&d));
    }
    if map_dir().join("chillblock5/ChillBlock5.map").exists() {
        let cb5 = Arena::build(&defs["chillblock5-ruler"], &map_dir()).unwrap();
        assert!(cb5.slot_count() >= 20);
    }
}

/// The spawn positions of the phase-0 TS harness (`gen_harness_spawns.mjs`, seeds 1..=200) are
/// reproduced exactly: same standing slots, same RNG stream, same rejection rule -- so a game seed
/// means the same starting position here as in E-000.
#[test]
fn spawns_match_the_ts_harness() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/harness_spawns.json")).expect("fixture parses");
    let defs = load_arena_defs(&repo("configs/arenas")).unwrap();
    let mut checked = 0;
    for (name, entry) in fixture.as_object().unwrap() {
        let arena = match Arena::build(&defs[name], &map_dir()) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skipping {name}: {e}");
                continue;
            }
        };
        if let Some(slots) = entry["slots"].as_u64() {
            assert_eq!(arena.slot_count() as u64, slots, "{name}: standing slots");
        }
        for (i, want) in entry["spawns"].as_array().unwrap().iter().enumerate() {
            let seed = i as u64 + 1;
            let tiles = arena.spawn_tiles(seed, 2).unwrap();
            let px = |t: (i32, i32)| {
                [
                    f64::from(ddai_env::arena::tile_center(t.0)),
                    f64::from(ddai_env::arena::tile_center(t.1)),
                ]
            };
            let got = [px(tiles[0]), px(tiles[1])].concat();
            let want: Vec<f64> = want.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            assert_eq!(got, want, "{name} seed {seed}");
            checked += 1;
        }
    }
    assert!(
        checked >= 400,
        "at least the two synthetic arenas must be checked, got {checked}"
    );
}

/// F2: in 1vN games no two players start on (or next to) the same tile, on the real halls too.
#[test]
fn one_v_n_spawns_are_spread_on_the_real_halls() {
    let defs = load_arena_defs(&repo("configs/arenas")).unwrap();
    for name in ["pit", "platform", "clb-left", "clb-right"] {
        let arena = match Arena::build(&defs[name], &map_dir()) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skipping {name}: {e}");
                continue;
            }
        };
        let (min, max) = if name == "pit" {
            (4.0, 20.0)
        } else if name == "platform" {
            (3.0, 16.0)
        } else {
            (3.0, 12.0)
        };
        for players in [3usize, 4] {
            for seed in 0..3_000u64 {
                let s = arena.spawn_tiles(seed, players).unwrap();
                for (i, &t) in s.iter().enumerate() {
                    for &u in &s[..i] {
                        let d = f64::from(t.0 - u.0).hypot(f64::from(t.1 - u.1));
                        assert!(d >= min, "{name} seed {seed}: {s:?}");
                    }
                    if i > 0 {
                        let d = f64::from(t.0 - s[0].0).hypot(f64::from(t.1 - s[0].1));
                        assert!(d <= max, "{name} seed {seed}: {s:?}");
                    }
                }
            }
        }
    }
}
