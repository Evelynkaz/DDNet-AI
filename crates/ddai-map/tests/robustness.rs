//! Robustness/fuzz test (task 1.4 acceptance criterion #2): `load_map` must never panic on
//! arbitrary input, no matter how malformed, and must never allocate an absurd amount of memory
//! for a small/malicious input (see `crate::datafile`'s `MAX_ALLOC_BYTES` and `crate::loader`'s
//! `MAX_TILE_COUNT`, both cited here as what actually backs that guarantee — this test can only
//! observe the "never panics" half directly).
//!
//! Two independent seed pools:
//! - `fuzz_mutated_synthetic_maps_and_random_bytes_never_panics` (always runs, self-contained —
//!   matches this crate's "no third-party maps committed to git" constraint): mutates this
//!   crate's own [`ddai_map::testutil`] fixtures.
//! - `fuzz_mutated_real_corpus_maps_never_panics` (`#[ignore]`d: reads real DDNet maps from an
//!   external, non-repository path): additionally mutates real map bytes when run locally with
//!   the task's corpus present — see the build report for the numbers this run produced.
//!
//! Both call the exact same mutator and iteration count (120,000 — comfortably over the task's
//! "≥ 100k" floor) via [`run_fuzz`]. The mutator is a deterministic SplitMix64-seeded byte
//! mangler (same core algorithm as `crates/ddai-trace/src/prng.rs`, reimplemented here rather
//! than adding a dependency on that crate, to keep `ddai-map` decoupled from `ddai-trace`) so a
//! failure is reproducible from its printed seed.

use ddai_map::load_map;
use std::panic;

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..bound` (Lemire's multiply-high trick); `below(0) == 0`.
    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        ((self.next_u64() as u128 * bound as u128) >> 64) as usize
    }
}

fn mutate(rng: &mut SplitMix64, template: &[u8]) -> Vec<u8> {
    let mut bytes = template.to_vec();
    match rng.below(5) {
        0 => {
            // Flip a random number of random bits.
            let flips = 1 + rng.below(16);
            for _ in 0..flips {
                if bytes.is_empty() {
                    break;
                }
                let i = rng.below(bytes.len());
                let bit = rng.below(8);
                bytes[i] ^= 1 << bit;
            }
        }
        1 => {
            // Truncate to a random shorter length.
            if !bytes.is_empty() {
                let len = rng.below(bytes.len());
                bytes.truncate(len);
            }
        }
        2 => {
            // Overwrite a random contiguous run with random bytes.
            if !bytes.is_empty() {
                let start = rng.below(bytes.len());
                let run = 1 + rng.below((bytes.len() - start).min(64));
                let end = (start + run).min(bytes.len());
                for b in &mut bytes[start..end] {
                    *b = rng.below(256) as u8;
                }
            }
        }
        3 => {
            // Insert random garbage bytes at a random position (grows the file — exercises the
            // "declared sizes don't match the real length" rejection paths differently than
            // truncation/overwrite do).
            let count = 1 + rng.below(64);
            let at = if bytes.is_empty() { 0 } else { rng.below(bytes.len()) };
            let garbage: Vec<u8> = (0..count).map(|_| rng.below(256) as u8).collect();
            bytes.splice(at..at, garbage);
        }
        _ => {
            // Overwrite a random 4-byte-aligned window with an extreme i32 value — targets the
            // header's own count/size fields specifically, which is where this crate's
            // bounded-allocation logic (and DDNet's own overflow checks) actually live.
            if bytes.len() >= 4 {
                let start = rng.below(bytes.len() - 4 + 1);
                let extreme: i32 = match rng.below(4) {
                    0 => i32::MAX,
                    1 => i32::MIN,
                    2 => -1,
                    _ => 0,
                };
                bytes[start..start + 4].copy_from_slice(&extreme.to_le_bytes());
            }
        }
    }
    bytes
}

/// Real, well-formed maps this crate built itself (see [`ddai_map::testutil`]) — every tilemap
/// shape/version this crate's loader has special-cased, so a mutation has a real chance of
/// landing somewhere structurally interesting instead of always hitting the same "header
/// completely wrong" rejection.
fn synthetic_templates() -> Vec<Vec<u8>> {
    use ddai_map::testutil::{
        MapWriter, TILESLAYERFLAG_FRONT, TILESLAYERFLAG_GAME, TILESLAYERFLAG_SPEEDUP, TILESLAYERFLAG_SWITCH,
        TILESLAYERFLAG_TELE, TILESLAYERFLAG_TUNE, TileLayerSpec, TilemapShape, encode_tile_skip, game_layer_data,
    };

    let mut templates = Vec::new();

    for version in [3, 4] {
        let mut w = MapWriter::new(version);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 4,
            height: 3,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(4, 3),
        });
        w.add_single_group_with_all_layers();
        templates.push(w.finish());
    }

    // Every physics layer present, plus settings/info — the richest fixture.
    {
        let (w_, h_) = (5, 4);
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_info_item(Some("Author"), Some("v1"), Some("Credits"), Some("CC0"), &["sv_foo 1"]);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: w_,
            height: h_,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(w_, h_),
        });
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: w_,
            height: h_,
            flags: TILESLAYERFLAG_FRONT,
            data: &game_layer_data(w_, h_),
        });
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: w_,
            height: h_,
            flags: TILESLAYERFLAG_TELE,
            data: &vec![0u8; (w_ * h_ * 2) as usize],
        });
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: w_,
            height: h_,
            flags: TILESLAYERFLAG_SPEEDUP,
            data: &vec![0u8; (w_ * h_ * 6) as usize],
        });
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: w_,
            height: h_,
            flags: TILESLAYERFLAG_SWITCH,
            data: &vec![0u8; (w_ * h_ * 4) as usize],
        });
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: w_,
            height: h_,
            flags: TILESLAYERFLAG_TUNE,
            data: &vec![0u8; (w_ * h_ * 2) as usize],
        });
        w.add_single_group_with_all_layers();
        templates.push(w.finish());
    }

    // Tile-skip encoded (tilemap item version 4).
    {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        let packed = encode_tile_skip(&[(0, 0); 12]);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 4,
            width: 4,
            height: 3,
            flags: TILESLAYERFLAG_GAME,
            data: &packed,
        });
        w.add_single_group_with_all_layers();
        templates.push(w.finish());
    }

    templates
}

fn run_fuzz(templates: &[Vec<u8>], iterations: u64, seed: u64) {
    assert!(!templates.is_empty());
    let result = panic::catch_unwind(|| {
        let mut rng = SplitMix64::new(seed);
        for i in 0..iterations {
            // Every 5th iteration is pure random bytes (no template shape at all); the rest
            // mutate a random real map's bytes.
            let bytes = if i % 5 == 0 {
                let len = rng.below(4096);
                (0..len).map(|_| rng.below(256) as u8).collect::<Vec<u8>>()
            } else {
                let template = &templates[rng.below(templates.len())];
                mutate(&mut rng, template)
            };
            // Discarded on purpose: both `Ok` and `Err` are fine outcomes here, only a panic is
            // a failure — that's what `catch_unwind` is for.
            let _ = load_map(&bytes);
        }
    });
    // `catch_unwind` stops at the *first* panic (the loop never resumes), so at most one panic
    // message ever prints — no need to silence the global panic hook for 100,000 iterations'
    // worth of (nonexistent, in the passing case) noise, and doing so would be a process-global
    // change that could race with other tests' own (unrelated) panics running in parallel.
    if result.is_err() {
        panic!(
            "load_map panicked while fuzzing (seed={seed}, iterations={iterations}) — rerun with this seed to reproduce"
        );
    }
}

#[test]
fn fuzz_mutated_synthetic_maps_and_random_bytes_never_panics() {
    run_fuzz(&synthetic_templates(), 120_000, 0xC0FFEE);
}

#[test]
#[ignore = "reads real DDNet maps from an external, non-repository path — run locally with \
            `DDAI_MAP_FUZZ_CORPUS=~/aiddnet/data/research/proto-scratch/ddnet-maps cargo test \
            -p ddai-map --test robustness -- --ignored` (see the task's build report for the \
            numbers this produced)"]
fn fuzz_mutated_real_corpus_maps_never_panics() {
    let Ok(dir) = std::env::var("DDAI_MAP_FUZZ_CORPUS") else {
        eprintln!("DDAI_MAP_FUZZ_CORPUS not set, skipping");
        return;
    };
    let mut templates = Vec::new();
    for path in walk(&std::path::PathBuf::from(&dir)) {
        if let Ok(bytes) = std::fs::read(&path) {
            templates.push(bytes);
        }
        if templates.len() >= 64 {
            break; // a sample of real byte shapes is enough to seed the mutator with.
        }
    }
    assert!(!templates.is_empty(), "no .map files found under {dir}");
    run_fuzz(&templates, 120_000, 0xC0FFEE);
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else if path.extension().is_some_and(|e| e == "map") {
            out.push(path);
        }
    }
    out
}
