//! End to end on synthetic bytes only: demo files -> `from_demos` -> dataset directory -> reader.
//! Checks the things the acceptance criteria demand of the whole pipeline: skipped/duplicate/
//! cache-map handling, counts, tags and filters, byte-identical reruns, and that no nickname
//! (nor the demo file name) appears anywhere in the output.

use std::fs;
use std::path::{Path, PathBuf};

use ddai_dataset::dataset::{DatasetReader, Filter};
use ddai_dataset::run::{Options, from_demos};
use ddai_dataset::synth::{self, SynthChar};
use ddai_dataset::tags::Technique;
use ddai_dataset::testutil::wire_character;
use ddai_dataset::types::{ReplayClass, SkillBucket};
use ddai_dataset::{config::Config, dataset::sha256_hex};

const NICKS: [&str; 3] = ["ZzSecretAlpha", "ZzSecretBeta", "ZzSecretGamma"];
const CLAN: &str = "ZzSecretClan";

fn ch(id: i32, tick: i32, x: i32, y: i32, weapon: i32, hook_to: Option<i32>) -> SynthChar {
    let mut w = wire_character(tick, x, y);
    w.weapon = weapon;
    if let Some(v) = hook_to {
        w.hook_state = 5;
        w.hooked_player = v;
        w.hook_x = x + 100;
    }
    SynthChar {
        id,
        name: NICKS[id as usize].to_string(),
        clan: CLAN.to_string(),
        wire: w,
    }
}

/// A hooks V from the ground over the pit; V freezes (ninja marker) and stays frozen for 60 ticks.
fn hook_and_block_snapshots() -> Vec<(i32, Vec<SynthChar>)> {
    let mut out = Vec::new();
    for i in 0..45 {
        let tick = 1000 + 2 * i;
        let hooking = (5..9).contains(&i);
        let v_x = if i < 5 { 420 } else { (420 - 15 * (i - 4)).max(330) };
        let frozen = (10..40).contains(&i);
        out.push((
            tick,
            vec![
                ch(0, tick, 250, 338, 0, hooking.then_some(1)),
                ch(1, tick, v_x, 300, if frozen { 5 } else { 0 }, None),
                ch(2, tick, 100, 338, 0, None),
            ],
        ));
    }
    out
}

fn opts(demos: &Path, out: &Path, maps: Vec<PathBuf>, threads: usize) -> Options {
    Options {
        demos_dir: demos.to_path_buf(),
        out_dir: out.to_path_buf(),
        map_dirs: maps,
        threads,
        code_commit: "test-commit".to_string(),
        name: "synthetic".to_string(),
        source: "synthetic demos".to_string(),
        limit: None,
        top_players: 5,
    }
}

fn all_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

struct Fixture {
    _tmp: tempfile::TempDir,
    demos: PathBuf,
    maps: PathBuf,
    root: PathBuf,
    demo_sha: String,
    map_sha: String,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let demos = root.join("demos");
    let maps = root.join("mapcache");
    fs::create_dir_all(demos.join("sub")).unwrap();
    fs::create_dir_all(&maps).unwrap();
    let map = synth::map_bytes(30, 12, (10, 20));
    let snaps = hook_and_block_snapshots();
    // Demo file names may carry nicknames in the real archive: they must never reach the output.
    let d1 = synth::demo_bytes(&map, true, &snaps);
    fs::write(demos.join("ZzSecretAlpha raid.demo"), &d1).unwrap();
    // A byte-identical copy under another name (a duplicate), a demo relying on the map cache, a
    // corrupt file and a file that is not a demo at all.
    fs::write(demos.join("sub").join("copy of ZzSecretBeta.demo"), &d1).unwrap();
    let shifted: Vec<_> = snaps
        .iter()
        .map(|(t, c)| {
            let chars = c
                .iter()
                .map(|sc| {
                    let mut sc = sc.clone();
                    sc.wire.tick += 100_000;
                    sc
                })
                .collect();
            (t + 100_000, chars)
        })
        .collect();
    fs::write(demos.join("nomap.demo"), synth::demo_bytes(&map, false, &shifted)).unwrap();
    fs::write(demos.join("broken.demo"), b"TWDEMO\0garbage that is far too short").unwrap();
    fs::write(demos.join("readme.txt"), b"not a demo").unwrap();
    fs::write(maps.join("cached-arena.map"), &map).unwrap();
    Fixture {
        demo_sha: sha256_hex(&d1),
        map_sha: sha256_hex(&map),
        _tmp: tmp,
        demos,
        maps,
        root,
    }
}

#[test]
fn the_whole_pipeline_on_synthetic_demos() {
    let fx = fixture();
    let out = fx.root.join("out");
    let rep = from_demos(
        &opts(&fx.demos, &out, vec![fx.maps.clone()], 3),
        &Config::default(),
        &|_| {},
    )
    .unwrap();

    // Discovery: 4 demo files (the txt is ignored); one duplicate dropped, one broken skipped.
    assert_eq!(rep.demos.total, 4);
    assert_eq!(rep.demos.duplicates, 1);
    assert_eq!(rep.demos.ok, 2);
    assert_eq!(rep.demos.skipped, 1);
    assert_eq!(rep.demos.skipped_reasons.get("demo header unreadable"), Some(&1));
    assert_eq!(rep.demos.map_embedded, 1);
    assert_eq!(
        rep.demos.map_from_cache, 1,
        "the demo without an embedded map found it by crc"
    );

    // Skill signals: A hooked V, V froze and stayed frozen 60 ticks: one credited block per demo.
    assert_eq!(rep.skill.freeze_entries, 2);
    assert_eq!(rep.skill.credited_freezes, 2);
    assert_eq!(rep.skill.blocks, 2);
    assert_eq!(rep.skill.self_freezes, 0);
    assert_eq!(rep.skill.players, 6);

    // Technique: the drag through the pit edge was detected and counted as a success.
    let t1 = rep.techniques.iter().find(|t| t.code == "T1").unwrap();
    assert_eq!((t1.active, t1.success), (2, 2), "{t1:?}");
    assert_eq!(t1.examples.len(), 2);
    assert!(t1.examples.iter().all(|e| e.demo.len() == 12));
    let not_detected: Vec<_> = rep
        .techniques
        .iter()
        .filter(|t| !t.detectable)
        .map(|t| t.code.as_str())
        .collect();
    assert!(not_detected.contains(&"T6") && not_detected.contains(&"T16"));

    // The dataset reads back.
    let reader = DatasetReader::open(&out).unwrap();
    reader.verify().unwrap();
    assert_eq!(reader.manifest.demos.len(), 3);
    assert_eq!(reader.manifest.maps.len(), 1);
    assert_eq!(reader.manifest.maps[0].sha256, fx.map_sha);
    assert_eq!(reader.manifest.code_commit, "test-commit");
    assert_eq!(reader.manifest.config_hash, Config::default().hash_hex());
    assert_eq!(reader.manifest.counts.demos_ok, 2);
    assert!(
        reader
            .manifest
            .demos
            .iter()
            .any(|d| d.sha256 == fx.demo_sha && d.map_source == "embedded")
    );
    assert!(reader.manifest.demos.iter().any(|d| d.map_source == "cache"));
    let skipped = reader.manifest.demos.iter().find(|d| d.status == "skipped").unwrap();
    assert!(skipped.reason.as_deref().unwrap().contains("header"));

    let all: Vec<_> = reader.samples(&Filter::default()).collect::<Result<_, _>>().unwrap();
    assert_eq!(all.len() as u64, reader.manifest.counts.samples);
    assert_eq!(all.len(), 2 * 44 * 3, "44 decision steps x 3 players x 2 demos");
    let one = &all[0];
    assert_eq!(hex(&one.meta.map_sha256), fx.map_sha);
    assert_eq!(one.observation.others.len(), 2);
    assert_eq!(one.observation.tuning, ddai_physics::tuning::TuningParams::default());
    assert!(all.iter().all(|s| Arc_ptr_ok(&s.observation.map)));
    assert!(all.iter().any(|s| hex(&s.meta.demo_sha256) == fx.demo_sha));

    // Tags: samples of the hooker around the drag carry T1; a tag filter returns exactly those.
    let t1_filter = Filter {
        any_tags: Technique::T1.bit(),
        ..Filter::default()
    };
    let tagged: Vec<_> = reader.samples(&t1_filter).collect::<Result<_, _>>().unwrap();
    assert!(!tagged.is_empty());
    assert!(tagged.iter().all(|s| s.meta.tags & Technique::T1.bit() != 0));
    assert!(
        tagged.iter().all(|s| s.observation.self_state.id == 0),
        "only the hooker is tagged"
    );
    let none_filter = Filter {
        none_tags: Technique::T1.bit(),
        ..Filter::default()
    };
    let rest = reader.samples(&none_filter).count();
    assert_eq!(rest + tagged.len(), all.len());

    // Frozen victim: exclude_frozen drops exactly the samples where the actor is frozen.
    let frozen = all.iter().filter(|s| s.observation.self_state.is_frozen).count();
    assert!(frozen > 0);
    let free_only = Filter {
        exclude_frozen: true,
        ..Filter::default()
    };
    assert_eq!(reader.samples(&free_only).count(), all.len() - frozen);

    // Confidence and skill filters compose; a tiny synthetic demo ranks nobody (< 30 s visible).
    let good = reader.samples(&Filter::good_play()).count();
    let conf = Filter {
        min_replay: ReplayClass::Within1px,
        ..Filter::default()
    };
    assert!(reader.samples(&conf).count() >= good);
    assert_eq!(
        good, 0,
        "nobody is ranked in a 90-tick demo, so nobody is Mid or better"
    );
    assert!(all.iter().all(|s| s.meta.skill == SkillBucket::Unranked));
    let by_map = Filter {
        maps: vec!["0".repeat(64)],
        ..Filter::default()
    };
    assert_eq!(reader.samples(&by_map).count(), 0);
    let by_map_ok = Filter {
        maps: vec![fx.map_sha.clone()],
        ..Filter::default()
    };
    assert_eq!(reader.samples(&by_map_ok).count(), all.len());

    // players.json: one row per (demo, anonymous label), no names.
    assert_eq!(reader.players.len(), 6);
    assert!(
        reader
            .players
            .iter()
            .all(|r| r.rank == 0 && r.bucket == SkillBucket::Unranked)
    );
}

#[allow(non_snake_case)]
fn Arc_ptr_ok(m: &std::sync::Arc<ddai_physics::map::MapData>) -> bool {
    m.width == 30 && m.height == 12
}

fn hex(b: &[u8; 32]) -> String {
    ddai_dataset::config::hex(b)
}

#[test]
fn no_nickname_and_no_file_name_reaches_any_output_file() {
    let fx = fixture();
    let out = fx.root.join("out");
    from_demos(
        &opts(&fx.demos, &out, vec![fx.maps.clone()], 2),
        &Config::default(),
        &|_| {},
    )
    .unwrap();
    let mut scanned = 0;
    for f in all_files(&out) {
        let bytes = fs::read(&f).unwrap();
        // Chunk files are compressed: also scan their decompressed content.
        let mut views = vec![bytes.clone()];
        if f.extension().is_some_and(|e| e == "zst") {
            views.push(zstd::decode_all(bytes.as_slice()).unwrap());
        }
        for v in views {
            let text = String::from_utf8_lossy(&v).to_lowercase();
            for needle in NICKS
                .iter()
                .chain([&CLAN, &"secret", &"raid", &"copy of", &"nomap", &"broken"])
            {
                assert!(
                    !text.contains(&needle.to_lowercase()),
                    "{} contains {needle}",
                    f.display()
                );
            }
            // The integer-packed name form (4 ints) must not survive either.
            for n in NICKS {
                assert!(
                    !v.windows(4).any(|w| w == &n.as_bytes()[..4]),
                    "{} has a name prefix",
                    f.display()
                );
            }
        }
        scanned += 1;
    }
    assert!(scanned >= 6, "manifest, players, report, map and chunks were scanned");
}

#[test]
fn two_runs_produce_identical_bytes_regardless_of_thread_count() {
    let fx = fixture();
    let a = fx.root.join("a");
    let b = fx.root.join("b");
    from_demos(
        &opts(&fx.demos, &a, vec![fx.maps.clone()], 1),
        &Config::default(),
        &|_| {},
    )
    .unwrap();
    from_demos(
        &opts(&fx.demos, &b, vec![fx.maps.clone()], 4),
        &Config::default(),
        &|_| {},
    )
    .unwrap();
    let fa = all_files(&a);
    let fb = all_files(&b);
    let rel = |base: &Path, v: &[PathBuf]| {
        v.iter()
            .map(|p| p.strip_prefix(base).unwrap().to_path_buf())
            .collect::<Vec<_>>()
    };
    assert_eq!(rel(&a, &fa), rel(&b, &fb), "same file set");
    for (x, y) in fa.iter().zip(&fb) {
        assert_eq!(fs::read(x).unwrap(), fs::read(y).unwrap(), "{} differs", x.display());
    }
}

#[test]
fn a_demo_without_a_usable_map_is_skipped_and_reported() {
    let fx = fixture();
    // No map cache directory given: the demo that relies on the cache cannot be processed.
    let out = fx.root.join("out");
    let rep = from_demos(&opts(&fx.demos, &out, vec![], 2), &Config::default(), &|_| {}).unwrap();
    assert_eq!(rep.demos.ok, 1);
    assert_eq!(rep.demos.skipped, 2);
    assert_eq!(
        rep.demos
            .skipped_reasons
            .get("no embedded map and no cached map with the demo's crc"),
        Some(&1)
    );
    let text = ddai_dataset::report::render(&rep);
    assert!(text.contains("skipped"));
}

#[test]
fn a_corrupt_chunk_file_is_detected() {
    let fx = fixture();
    let out = fx.root.join("out");
    from_demos(
        &opts(&fx.demos, &out, vec![fx.maps.clone()], 2),
        &Config::default(),
        &|_| {},
    )
    .unwrap();
    let reader = DatasetReader::open(&out).unwrap();
    let chunk = out.join(&reader.manifest.chunks[0].file);
    let mut bytes = fs::read(&chunk).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    fs::write(&chunk, bytes).unwrap();
    assert!(reader.verify().is_err());
    assert!(reader.samples(&Filter::default()).any(|r| r.is_err()));
}

#[test]
fn ingest_replaces_names_with_per_demo_labels_and_splits_stints_of_a_reused_slot() {
    use ddai_recorder::format::Frame;
    let map = synth::map_bytes(30, 12, (10, 20));
    // Slot 0 is played by "Alpha" for 6 snapshots, then by "Beta" (a nick change ends the stint).
    let snaps: Vec<_> = (0..12)
        .map(|i| {
            let tick = 500 + 2 * i;
            let mut c = ch(0, tick, 250, 338, 0, None);
            if i >= 6 {
                c.name = NICKS[1].to_string();
            }
            (tick, vec![c, ch(2, tick, 100, 338, 0, None)])
        })
        .collect();
    let bytes = synth::demo_bytes(&map, true, &snaps);
    let demo = ddai_demo::Demo::parse(&bytes).unwrap();
    let ing = ddai_dataset::ingest::ingest(&demo);
    assert_eq!(ing.frames.len(), 12);
    let mut labels = Vec::new();
    for f in &ing.frames {
        let Frame::Snapshot { players, .. } = f else {
            panic!("snapshot frame")
        };
        for p in players {
            let ci = p.client_info.as_ref().unwrap();
            assert!(ci.name.starts_with("player_"), "name was {:?}", ci.name);
            assert_eq!(ci.clan, "", "clan is blanked");
            if p.id == 0 {
                labels.push(ci.name.clone());
            }
        }
    }
    assert_eq!(labels[0], labels[5]);
    assert_ne!(
        labels[5], labels[6],
        "a new nick in the same slot is a new anonymous player"
    );
    assert_eq!(labels[6], labels[11]);
    // The pipeline sees the same split: slot 0 has two labels in the frames.
    let cfg = Config::default();
    let out = ddai_dataset::demo::process(&cfg, &std::sync::Arc::new(ddai_map::load_map(&map).unwrap().data), &ing);
    let mut ids: Vec<u16> = out
        .frames
        .iter()
        .flat_map(|f| f.chars.iter().filter(|c| c.id == 0).map(|c| c.player))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 2);
}
