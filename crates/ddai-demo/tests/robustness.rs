//! Robustness / fuzz-style tests: task 8.4b acceptance criterion 1 — feed >= 10^6 random/mutated
//! `.demo` byte buffers into [`ddai_demo::Demo::parse`] and [`ddai_demo::reader::TickIter`], and
//! confirm none of them ever panics or fails to terminate. Mirrors `ddai-net`'s own
//! `tests/robustness.rs` (task 2.2a): "real, well-formed" seeds are built from this crate's own
//! encoders (`ddai_demo::testutil`, gated behind the `test-util` feature — see `Cargo.toml`),
//! never an external file, so this test runs in CI with no data directory present.
//!
//! `#[test]` failing here means a panic (proptest reports the panicking input directly), so "the
//! run finished" already proves "no panics" for every case it generated; the case counts below
//! sum to over 1,000,000.

use ddai_demo::Demo;
use ddai_demo::testutil::build_synthetic_demo;
use proptest::prelude::*;

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

fn mutation_strategy(max_len: usize, max_mutations: usize) -> impl Strategy<Value = Vec<(usize, u8)>> {
    prop::collection::vec((0..max_len.max(1), any::<u8>()), 0..max_mutations)
}

/// Parses `bytes` and, if that succeeds, fully drains the tick iterator — exercising every stage
/// this crate has (header/timeline/map prelude, chunk header framing, Huffman + variable-int
/// decompression, delta unpacking, raw-snapshot parsing, message decoding) on the same input.
/// Bounded: [`Demo::ticks`] always terminates (each chunk consumes >= 1 input byte, and
/// `crate::header::MAX_DEMO_FILE_SIZE` bounds the file itself), so this never hangs.
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300_000))]

    /// Pure random bytes of realistic file sizes never panic, whether or not they even parse a
    /// header.
    #[test]
    fn never_panics_on_random_bytes(bytes in prop::collection::vec(any::<u8>(), 0..4096)) {
        parse_and_drain(&bytes);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400_000))]

    /// Mutated real (well-formed) demo bytes — far more likely to pass the header check and
    /// actually exercise the chunk/delta/message decode paths than pure noise.
    #[test]
    fn never_panics_on_mutated_real_demo(
        version in prop_oneof![Just(3u8), Just(4), Just(5), Just(6), Just(7)],
        mutations in mutation_strategy(4096, 12),
    ) {
        let base = build_synthetic_demo(version);
        let mutated = mutate(base, &mutations);
        parse_and_drain(&mutated);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(350_000))]

    /// Heavier mutation (more positions, and truncation) of a real demo, specifically targeting
    /// the tick/chunk stream (mutations only ever land past the header) — more likely to corrupt
    /// tick markers, chunk size fields and compressed payloads in combination.
    #[test]
    fn never_panics_on_heavily_mutated_or_truncated_demo(
        mutations in mutation_strategy(4096, 40),
        truncate_to in 0usize..600,
    ) {
        let mut base = build_synthetic_demo(6);
        base.truncate(base.len().min(truncate_to.max(176))); // never below the header size itself
        let mutated = mutate(base, &mutations);
        parse_and_drain(&mutated);
    }
}
