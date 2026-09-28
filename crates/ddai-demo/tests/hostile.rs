//! Regression tests for review round 1 findings F2 (unbounded memory) and F5 (partial tick
//! content discarded before a stopping error) — each built from this crate's own encoders
//! (`ddai_demo::testutil`, behind the `test-util` feature), no external file needed.

use ddai_demo::reader::MAX_MESSAGES_PER_TICK;
use ddai_demo::testutil::{build_prelude_bytes, write_chunk, write_tick_marker};
use ddai_demo::{Demo, TickIterError};
use ddai_net::huffman::Huffman;
use std::time::{Duration, Instant};

/// Finding F2: a hostile file can pack a chunk header down to one byte (tick-marker flag clear,
/// type `MESSAGE`, size 0 — byte `0x40`) with nothing after it, so without a per-tick cap the
/// number of `Msg` entries accumulated between two tick markers is bounded only by the file's own
/// size — the original report used an ~8 MiB file of such bytes and hit a ~4.4 GB allocation
/// request (`Msg` is 528 bytes) before this crate capped it. This test uses a much smaller file
/// (just over the cap) and asserts both that decoding finishes quickly (no runaway CPU/memory)
/// and that the cap is enforced exactly, via [`TickIterError::TooManyMessages`].
#[test]
fn many_tiny_message_chunks_in_one_tick_are_capped_not_unbounded() {
    let mut bytes = build_prelude_bytes(6, 0);
    write_tick_marker(&mut bytes, 10, true);
    // One byte per chunk: tick-marker flag clear (0x80), type MESSAGE (2 << 5 = 0x40), size 0.
    let one_more_than_cap = MAX_MESSAGES_PER_TICK + 1;
    bytes.extend(std::iter::repeat_n(0x40u8, one_more_than_cap));

    let start = Instant::now();
    let demo = Demo::parse(&bytes).expect("well-formed prelude");
    let ticks: Vec<_> = demo.ticks().collect();
    let elapsed = start.elapsed();

    // Real DDNet traffic never comes close to this; if the cap weren't enforced, `one_more_than_
    // cap` copies of a 528-byte `Msg` would already be tens of MB, and the original (uncapped)
    // report's 8 MiB input took an OS-level abort rather than finishing at all — a generous
    // bound on wall-clock time here is a meaningful, portable stand-in for "did not try to
    // allocate/process something unbounded".
    assert!(
        elapsed < Duration::from_secs(5),
        "took {elapsed:?}, expected a fast, bounded decode"
    );

    assert_eq!(ticks.len(), 2, "one capped tick, then the deferred error");
    let tick0 = ticks[0]
        .as_ref()
        .expect("tick 10 is a valid partial tick, not an error");
    assert_eq!(tick0.tick, 10);
    assert_eq!(
        tick0.messages.len(),
        MAX_MESSAGES_PER_TICK,
        "exactly the cap's worth of messages, not one more"
    );
    assert_eq!(ticks[1], Err(TickIterError::TooManyMessages(MAX_MESSAGES_PER_TICK)));
}

/// Finding F5: a fatal decode error partway through a tick must not discard whatever that same
/// tick already decoded before the error — real DDNet delivers listener callbacks synchronously
/// per chunk, so a truncated file that cuts off mid-tick still leaves the *earlier* chunks of
/// that tick "delivered". This builds: a valid snapshot at tick 10, then at tick 11 a valid
/// message chunk followed by a chunk header that is truncated (declares an extended size and
/// then the file just ends) — the truncation must surface as its own `Err` on the *next* `Tick`,
/// not swallow tick 11's already-decoded message.
#[test]
fn a_truncated_chunk_mid_tick_yields_the_earlier_content_of_that_tick_first() {
    let huffman = Huffman::new();
    let mut bytes = build_prelude_bytes(6, 0);

    write_tick_marker(&mut bytes, 10, true);
    // Raw `CSnapshot` layout: [data_size, num_items, offsets..., data...] — one item, key
    // `1 << 16`, one data int.
    write_chunk(&mut bytes, &huffman, 1, &[8, 1, 0, 1 << 16, 7]);

    write_tick_marker(&mut bytes, 11, false);
    write_chunk(&mut bytes, &huffman, 2, &[0]); // an (undecodable, doesn't matter) message chunk

    // A chunk header that is itself truncated: type MESSAGE (2), the "read one more size byte"
    // marker (30 in the low 5 bits), then nothing — `read_chunk_header` must read past the end of
    // `bytes` to find that size byte.
    bytes.push((2 << 5) | 30);

    let demo = Demo::parse(&bytes).expect("well-formed prelude");
    let ticks: Vec<_> = demo.ticks().collect();

    assert_eq!(ticks.len(), 3, "tick 10, tick 11 (partial), then the deferred error");

    let tick10 = ticks[0].as_ref().expect("tick 10 decodes cleanly");
    assert_eq!(tick10.tick, 10);
    assert_eq!(tick10.snapshot.as_ref().expect("tick 10 has a snapshot").items.len(), 1);
    assert!(tick10.messages.is_empty());

    let tick11 = ticks[1]
        .as_ref()
        .expect("tick 11's content before the truncation is not discarded");
    assert_eq!(tick11.tick, 11);
    // The message chunk (processed before the truncation) triggers the "replay last snapshot"
    // path (`demo.cpp:801-807`), so tick 11 still carries tick 10's snapshot.
    assert_eq!(tick11.snapshot, tick10.snapshot);
    assert_eq!(
        tick11.messages.len(),
        1,
        "the one message chunk read before the truncation"
    );

    assert_eq!(ticks[2], Err(TickIterError::TruncatedChunk));
}

/// Finding F5's second half: a truncated *size byte* is a different failure mode than a
/// malformed *tick-marker value* and must be reported as such (`TruncatedChunk`, not
/// `BadTickMarker` — the two were folded into one error before this fix).
#[test]
fn a_truncated_extended_size_byte_is_reported_as_truncated_not_bad_tick_marker() {
    let mut bytes = build_prelude_bytes(6, 0);
    write_tick_marker(&mut bytes, 10, true);
    // Type MESSAGE (2), "read one more size byte" marker (30), then EOF.
    bytes.push((2 << 5) | 30);

    let demo = Demo::parse(&bytes).expect("well-formed prelude");
    let ticks: Vec<_> = demo.ticks().collect();

    assert_eq!(ticks.len(), 2);
    let tick10 = ticks[0].as_ref().expect("tick 10 itself is empty but not an error");
    assert_eq!(tick10.tick, 10);
    assert_eq!(ticks[1], Err(TickIterError::TruncatedChunk));
}
