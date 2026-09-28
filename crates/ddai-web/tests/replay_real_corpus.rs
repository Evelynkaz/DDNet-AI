//! Cross-checks `TraceBReader` against a real Oracle B trace-b file from the actual corpus
//! (`~/aiddnet/data/traces/oracle-b/v1/`, task 1.5 — not committed to this repository, per
//! `CLAUDE.md`'s "never commit" list, and not guaranteed present on every machine this workspace
//! is checked out on, e.g. CI). `#[ignore]`d for that reason; run explicitly with
//! `cargo test -p ddai-web --test replay_real_corpus -- --ignored` on a machine that has the
//! corpus.
//!
//! The expected numbers below were computed independently in Python, decoding the identical
//! trace-b v2 byte layout from scratch (not calling into this crate at all) directly against the
//! raw file bytes — a genuinely separate implementation of the same format, so agreement between
//! the two is real evidence the Rust reader's field offsets are correct, not just internally
//! self-consistent.

use std::path::PathBuf;

use ddai_web::live::map_resolve;
use ddai_web::live::replay::{ReplaySource, TraceBReader};

fn corpus_file() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home).join("aiddnet/data/traces/oracle-b/v1/realmap_BlmapChill__seed10001.trb");
    path.is_file().then_some(path)
}

#[test]
#[ignore = "needs the real Oracle B corpus on disk (~/aiddnet/data/traces/oracle-b/v1/), not committed to git"]
fn real_blmapchill_trace_matches_an_independent_python_decode() {
    let Some(path) = corpus_file() else {
        panic!("corpus file not found — run on a machine with ~/aiddnet/data/traces/oracle-b/v1/");
    };

    let mut reader = TraceBReader::open(&path).expect("open real trace");
    let header = reader.header();
    assert_eq!(header.character_ids, vec![0, 1, 2]);
    assert_eq!(header.tick_count, 3000);
    assert_eq!(header.metadata.mode, "real-map");
    assert_eq!(
        header.metadata.map_sha256,
        hex_to_32("c902b2da07291266ab201054e6b6c28abd31e10b099fa5b1066b2f5a88f98240")
    );

    let mut character_ticks = 0u64;
    let mut frozen_ticks = 0u64;
    let mut died = 0u64;
    let mut respawned = 0u64;
    let mut hook_grabs = 0u64;
    let mut first_char_positions = Vec::new();
    let mut tick_index = 0u32;

    while let Some(tick) = reader.next_tick().expect("next_tick on a real file must never error") {
        for (slot, row) in tick.characters.iter().enumerate() {
            if row.ddrace_alive != 0 {
                character_ticks += 1;
                if row.ddrace_is_in_freeze != 0 {
                    frozen_ticks += 1;
                }
            }
            if row.ddrace_died_this_tick != 0 {
                died += 1;
            }
            if row.ddrace_respawned_this_tick != 0 {
                respawned += 1;
            }
            if row.core_triggered_events & 0x08 != 0 {
                hook_grabs += 1;
            }
            if slot == 0 && tick_index < 5 {
                first_char_positions.push((row.core_pos_x, row.core_pos_y));
            }
        }
        tick_index += 1;
    }

    // Independently computed in Python — see this file's doc comment.
    assert_eq!(character_ticks, 9000);
    assert_eq!(frozen_ticks, 151);
    assert_eq!(died, 1);
    assert_eq!(respawned, 1);
    assert_eq!(hook_grabs, 86);
    assert_eq!(
        first_char_positions,
        vec![
            (2253.0, 2693.0),
            (2248.0, 2683.0),
            (2242.0, 2675.0),
            (2235.0, 2668.0),
            (2228.0, 2662.0),
        ]
    );
}

fn maps_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home).join("aiddnet/data/ddnet-server/maps");
    dir.is_dir().then_some(dir)
}

#[test]
#[ignore = "needs the real local ddnet-server maps directory on disk, not committed to git"]
fn real_map_resolves_and_builds_a_scene_matching_its_trace() {
    let Some(trace_path) = corpus_file() else {
        panic!("corpus file not found — run on a machine with ~/aiddnet/data/traces/oracle-b/v1/");
    };
    let Some(maps_dir) = maps_dir() else {
        panic!("maps dir not found — run on a machine with ~/aiddnet/data/ddnet-server/maps/");
    };

    let reader = TraceBReader::open(&trace_path).expect("open real trace");
    let header = reader.header();
    assert_eq!(header.metadata.mode, "real-map");
    let hint = header.metadata.real_map_path.as_deref().expect("real_map_path");

    let (resolved_path, scene) = map_resolve::resolve_by_sha256(&[maps_dir], hint, header.metadata.map_sha256)
        .expect("should resolve the real BlmapChill.map file and load it");
    assert_eq!(resolved_path.file_name().unwrap(), "BlmapChill.map");
    // BlmapChill's real dimensions (sanity bounds, not brittle exact pixel counts): a real
    // DDRace map is always at least a few dozen tiles in both directions.
    assert!(
        scene.width > 50 && scene.height > 50,
        "{}x{}",
        scene.width,
        scene.height
    );
    assert_eq!(scene.kinds.len(), (scene.width * scene.height) as usize);
}

#[tokio::test]
#[ignore = "needs the real Oracle B corpus and local maps directory on disk, not committed to git"]
async fn replay_source_end_to_end_against_the_real_corpus() {
    use ddai_web::live::map_resolve::MapCache;
    use ddai_web::live::source::{FrameSource, SourceEvent};

    let Some(trace_path) = corpus_file() else {
        panic!("corpus file not found");
    };
    let Some(maps_dir) = maps_dir() else {
        panic!("maps dir not found");
    };

    let map_cache = std::sync::Arc::new(MapCache::new());
    let source = ReplaySource::new(&trace_path, vec![maps_dir], map_cache).expect("construct ReplaySource");

    let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(32);
    let (_control_tx, control_rx) = tokio::sync::mpsc::channel(4);
    let task = Box::new(source).spawn(events_tx, control_rx);

    let mut saw_map = false;
    let mut saw_players = false;
    let mut frame_count = 0;
    for _ in 0..20 {
        match tokio::time::timeout(std::time::Duration::from_secs(5), events_rx.recv())
            .await
            .expect("should receive an event within 5s")
            .expect("event channel should not close")
        {
            SourceEvent::MapChanged(meta) => {
                assert_eq!(meta.name, "BlmapChill");
                assert!(meta.width > 0 && meta.height > 0);
                saw_map = true;
            }
            SourceEvent::Players(players) => {
                assert_eq!(players.len(), 3);
                saw_players = true;
            }
            SourceEvent::Frame(frame) => {
                frame_count += 1;
                assert!(!frame.characters.is_empty());
            }
            SourceEvent::Error(message) => panic!("unexpected error from real corpus: {message}"),
            _ => {}
        }
        if saw_map && saw_players && frame_count > 3 {
            break;
        }
    }
    assert!(saw_map, "should have seen a MapChanged event");
    assert!(saw_players, "should have seen a Players event");
    assert!(frame_count > 0, "should have seen at least one Frame event");

    task.abort();
}

/// Review round 1, finding F2: a `recipe_*` trace (synthetic recipe, `mode: "rawmap-scenario"`)
/// must actually play — the real corpus's 160 `recipe_*` traces have no companion `.rawmap` file
/// at all (confirmed directly against the corpus directory), so the map must be rebuilt via
/// `ddai_trace::synthetic::build`, keyed off the recipe name parsed from the trace's own
/// filename (see `resolve_synthetic_map`'s doc comment for why that's the only signal that
/// exists for this corpus). No `#[ignore]`d test existed for this path before this finding.
#[test]
#[ignore = "needs the real Oracle B corpus on disk (~/aiddnet/data/traces/oracle-b/v1/), not committed to git"]
fn real_recipe_trace_plays_via_the_rebuilt_synthetic_map() {
    let home = std::env::var("HOME").expect("HOME must be set");
    let path = PathBuf::from(&home).join("aiddnet/data/traces/oracle-b/v1/recipe_arena_seed20001.trb");
    if !path.is_file() {
        panic!("corpus file not found — run on a machine with ~/aiddnet/data/traces/oracle-b/v1/");
    }

    use ddai_web::live::map_resolve::MapCache;
    use ddai_web::live::source::{FrameSource, SourceEvent};

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let map_cache = std::sync::Arc::new(MapCache::new());
        // No `--maps-dir` at all: a recipe trace must not need one (there is no real `.map` file
        // to look up for it in the first place).
        let source = ReplaySource::new(&path, Vec::new(), map_cache).expect("construct ReplaySource");

        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(32);
        let (_control_tx, control_rx) = tokio::sync::mpsc::channel(4);
        let task = Box::new(source).spawn(events_tx, control_rx);

        let mut saw_map = false;
        let mut saw_players = false;
        let mut frame_count = 0;
        for _ in 0..20 {
            match tokio::time::timeout(std::time::Duration::from_secs(5), events_rx.recv())
                .await
                .expect("should receive an event within 5s")
                .expect("event channel should not close")
            {
                SourceEvent::MapChanged(meta) => {
                    assert!(meta.width > 0 && meta.height > 0, "{meta:?}");
                    saw_map = true;
                }
                SourceEvent::Players(players) => {
                    assert_eq!(players.len(), 3);
                    saw_players = true;
                }
                SourceEvent::Frame(frame) => {
                    frame_count += 1;
                    assert!(!frame.characters.is_empty());
                }
                SourceEvent::Error(message) => panic!("recipe trace must play, not error: {message}"),
                _ => {}
            }
            if saw_map && saw_players && frame_count > 3 {
                break;
            }
        }
        assert!(saw_map, "should have seen a MapChanged event for the recipe trace");
        assert!(saw_players, "should have seen a Players event");
        assert!(frame_count > 0, "should have seen at least one Frame event");

        task.abort();
    });
}

fn hex_to_32(hex: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, out_byte) in out.iter_mut().enumerate() {
        *out_byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).expect("valid hex");
    }
    out
}
