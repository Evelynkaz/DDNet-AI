//! Fuzzes [`ddai_demo::Demo::parse`]/[`ddai_demo::reader::TickIter`] against mutated bytes drawn
//! from real DDNet demos — review round 1 finding F4: the committed, always-on
//! `tests/robustness.rs` only mutates a small, entirely synthetic 2-tick demo (one full snapshot,
//! one no-op delta, one message, no chunk over ~30 bytes), which never exercises `>= 256`-byte
//! chunks, a delta with real *updates*, ex (UUID) types, or DDNet's UUID system messages — most of
//! the format's real complexity. This file fixes that with real bytes, at the cost of needing the
//! local demo corpus on disk (D-040) — `#[ignore]`d, like `tests/real_corpus.rs`:
//!
//! ```bash
//! cargo test -p ddai-demo --release --test fuzz_real_bytes --features ddai-demo/test-util -- --ignored --nocapture
//! ```
//!
//! Seed construction mirrors the review's own approach rather than mutating a whole real file
//! outright: this crate never parses the embedded map's own bytes at all (`ddai_map::load_map`'s
//! job, a separate crate), so mutating them would burn almost the entire mutation budget on bytes
//! this crate doesn't even look at. Instead, each seed is a small *synthetic* prelude
//! (`map_size = 0`, built the same way `tests/robustness.rs`'s does) followed by a real file's own
//! tick/chunk-stream tail (found via this crate's own [`ddai_demo::header::parse_prelude`], i.e.
//! everything after where that file's embedded map ends) — real, format-faithful Huffman/varint/
//! delta/DDNet-message bytes, with (unlike the whole-file corpus) every mutated byte actually on
//! a code path this crate exercises. Each tail is capped to a bounded prefix so a single fuzz case
//! stays cheap even though some real files' tick streams are megabytes long.

use ddai_demo::Demo;
use ddai_demo::testutil::build_prelude_bytes;
use proptest::prelude::*;
use std::path::{Path, PathBuf};

/// Real per-seed cap: large enough to cover many ticks' worth of real chunk variety, small enough
/// that 10^6+ mutate-and-parse cases stay fast.
const MAX_SEED_TAIL_BYTES: usize = 32 * 1024;

fn corpus_dirs() -> Vec<PathBuf> {
    let home = PathBuf::from(std::env::var("HOME").expect("HOME must be set"));
    vec![
        home.join("aiddnet/data/demos/chillerdragon/block-06"),
        home.join("aiddnet/data/demos/public-samples"),
    ]
}

fn collect_demo_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for entry in entries.flatten() {
            let p = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("demo")) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Builds one fuzz seed from a real file — see the module docs. `None` if the file doesn't parse
/// cleanly (e.g. the two known-corrupt archive files) or its chunk-stream tail is empty.
fn real_chunk_stream_seed(path: &Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    let prelude = ddai_demo::header::parse_prelude(&bytes).ok()?;
    let tail_start = prelude.map_offset + prelude.header.map_size as usize;
    let tail = bytes.get(tail_start..)?;
    let tail = &tail[..tail.len().min(MAX_SEED_TAIL_BYTES)];
    if tail.is_empty() {
        return None;
    }
    let mut seed = build_prelude_bytes(prelude.header.version, 0);
    seed.extend_from_slice(tail);
    Some(seed)
}

fn build_seeds() -> Vec<Vec<u8>> {
    let mut seeds = Vec::new();
    for dir in corpus_dirs() {
        for file in collect_demo_files(&dir) {
            if let Some(seed) = real_chunk_stream_seed(&file) {
                seeds.push(seed);
            }
        }
    }
    seeds
}

/// Mutates `base` at up to `num_mutations` random byte positions (flips to an arbitrary byte).
fn mutate(mut base: Vec<u8>, positions: &[(usize, u8)]) -> Vec<u8> {
    for &(pos, byte) in positions {
        if !base.is_empty() {
            let idx = pos % base.len();
            base[idx] = byte;
        }
    }
    base
}

fn parse_and_drain(bytes: &[u8]) {
    let Ok(demo) = Demo::parse(bytes) else {
        return;
    };
    let _ = demo.map_bytes();
    for tick in demo.ticks() {
        if tick.is_err() {
            break;
        }
    }
}

#[test]
#[ignore = "needs ~/aiddnet/data/demos on disk (D-040, local only)"]
fn mutated_real_chunk_streams_never_panic() {
    let seeds = build_seeds();
    assert!(
        !seeds.is_empty(),
        "expected at least one real demo to build a seed from"
    );
    eprintln!(
        "fuzz_real_bytes: {} seeds, {} bytes total",
        seeds.len(),
        seeds.iter().map(Vec::len).sum::<usize>()
    );

    // >= 10^6 cases total (review round 1 finding F4), split across a light and a heavy mutation
    // pass for the same reason `tests/robustness.rs` does: more, smaller mutations are more
    // likely to still pass the header/early chunk checks and reach deeper decode paths.
    let light = ProptestConfig::with_cases(600_000);
    let heavy = ProptestConfig::with_cases(600_000);

    let seed_index = 0..seeds.len();
    let light_mutations = prop::collection::vec((0..MAX_SEED_TAIL_BYTES.max(1), any::<u8>()), 0..8);
    let heavy_mutations = prop::collection::vec((0..MAX_SEED_TAIL_BYTES.max(1), any::<u8>()), 0..64);

    let mut runner = proptest::test_runner::TestRunner::new(light);
    runner
        .run(&(seed_index.clone(), light_mutations), |(idx, mutations)| {
            let mutated = mutate(seeds[idx].clone(), &mutations);
            parse_and_drain(&mutated);
            Ok(())
        })
        .expect("light-mutation pass over real chunk streams never panics");

    let mut runner = proptest::test_runner::TestRunner::new(heavy);
    runner
        .run(&(seed_index, heavy_mutations), |(idx, mutations)| {
            let mutated = mutate(seeds[idx].clone(), &mutations);
            parse_and_drain(&mutated);
            Ok(())
        })
        .expect("heavy-mutation pass over real chunk streams never panics");
}
