//! Synthetic `.demo` byte builders, shared by this crate's own unit tests and, via the
//! `test-util` feature (see `Cargo.toml`), `tests/*.rs` integration tests — mirrors `ddai-map`'s
//! identical `testutil` pattern. Nothing here reads or embeds any third-party file: every byte is
//! built from this crate's/`ddai_net`'s own understanding of the format, so a demo built here can
//! be committed or shared freely (task 8.4b acceptance criterion 5's "no third-party demo
//! bytes").

use crate::header::{
    HEADER_MARKER, HEADER_SIZE, MAX_TIMELINE_MARKERS, OLD_VERSION, SHA256_EXTENSION_UUID, SHA256_VERSION,
};
use ddai_net::huffman::Huffman;
use ddai_net::packer::pack_ints;

/// Builds a minimal, well-formed prelude byte sequence for a given `version`, with `map_size`
/// bytes of (here, `0xAB`-filled) map payload appended.
pub fn build_prelude_bytes(version: u8, map_size: u32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&HEADER_MARKER);
    out.push(version);
    let mut netversion = [0u8; 64];
    netversion[..3].copy_from_slice(b"0.6");
    out.extend_from_slice(&netversion);
    let mut map_name = [0u8; 64];
    map_name[..4].copy_from_slice(b"test");
    out.extend_from_slice(&map_name);
    out.extend_from_slice(&map_size.to_be_bytes());
    out.extend_from_slice(&0xDEAD_BEEFu32.to_be_bytes()); // crc
    let mut demo_type = [0u8; 8];
    demo_type[..6].copy_from_slice(b"client");
    out.extend_from_slice(&demo_type);
    out.extend_from_slice(&0i32.to_be_bytes()); // length
    let mut timestamp = [0u8; 20];
    timestamp[..10].copy_from_slice(b"2026-01-01");
    out.extend_from_slice(&timestamp);
    assert_eq!(out.len(), HEADER_SIZE);

    if version > OLD_VERSION {
        out.extend_from_slice(&0u32.to_be_bytes()); // num timeline markers
        out.extend_from_slice(&[0u8; MAX_TIMELINE_MARKERS * 4]);
    }
    if version >= SHA256_VERSION {
        out.extend_from_slice(&SHA256_EXTENSION_UUID);
        out.extend_from_slice(&[0u8; 32]);
    }
    out.extend(std::iter::repeat_n(0xABu8, map_size as usize));
    out
}

/// Writes a tick-marker chunk (always the absolute, 4-byte-tick form — `demo.cpp:256-279`'s
/// `else` branch — never the compressed-delta forms, so callers never need to track a running
/// "last tick" of their own).
pub fn write_tick_marker(out: &mut Vec<u8>, tick: i32, keyframe: bool) {
    let mut b = 0x80u8;
    if keyframe {
        b |= 0x40;
    }
    out.push(b);
    out.extend_from_slice(&tick.to_be_bytes());
}

/// Writes one chunk (`CDemoRecorder::Write`, `demo.cpp:281-333`, minus the >= 256-byte extended
/// size-field cases — every chunk this helper builds is small): packs `ints` with
/// `ddai_net::packer::pack_ints`, Huffman-compresses the result, and writes the 2-byte
/// (`size < 256`) chunk header + compressed bytes.
pub fn write_chunk(out: &mut Vec<u8>, huffman: &Huffman, ty: u8, ints: &[i32]) {
    let mut raw = vec![0u8; ints.len() * ddai_net::packer::MAX_BYTES_PACKED];
    let n = pack_ints(&mut raw, ints).expect("scratch buffer is sized generously enough");
    let compressed = huffman.compress_vec(&raw[..n]);
    let size = compressed.len();
    assert!(size < 256, "test helper only handles small chunks (size < 256)");
    out.push(((ty & 0x3) << 5) | 30);
    out.push(size as u8);
    out.extend_from_slice(&compressed);
}

/// Encodes a message's raw payload bytes the same way `CDemoRecorder::Write` would have
/// intpack-compressed them (bytes padded to a multiple of 4, then each 4-byte group reinterpreted
/// as one little-endian `i32` — see `crate::reader`'s docs for why little-endian) — the inverse
/// of `crate::reader::TickIter`'s own bytes-from-ints reconstruction, used here only to build
/// synthetic message chunks.
fn message_bytes_to_ints(payload: &[u8]) -> Vec<i32> {
    let mut padded = payload.to_vec();
    padded.resize(payload.len().div_ceil(4) * 4, 0);
    padded
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

/// Builds a complete, well-formed synthetic `.demo` byte buffer (zero-length embedded map, two
/// ticks: a full snapshot then a no-op delta plus one chat message) — entirely from this crate's
/// own understanding of the format via `ddai_net`'s own encoders. Used as the seed for
/// `tests/robustness.rs`'s mutation fuzzing and this crate's own end-to-end unit test.
pub fn build_synthetic_demo(version: u8) -> Vec<u8> {
    let mut out = build_prelude_bytes(version, 0);
    let huffman = Huffman::new();

    // Tick 10: full snapshot (raw `CSnapshot` layout: [data_size, num_items, offsets...,
    // data...]) with one item of internal type 1 id 0 and one data int. `data_size` (8) is the
    // whole data area in bytes: one item, `key` (4 bytes) + one data int (4 bytes).
    write_tick_marker(&mut out, 10, true);
    write_chunk(&mut out, &huffman, 1, &[8, 1, 0, 1 << 16, 7]);

    // Tick 11: a no-op delta (3-int empty header: 0 deleted, 0 updated, 0 temp) plus a chat
    // message (`NETMSGTYPE_SV_CHAT` = 8, not sys: leading varint `id << 1`).
    write_tick_marker(&mut out, 11, false);
    write_chunk(&mut out, &huffman, 3, &[0, 0, 0]);
    let mut msg_bytes = vec![8 << 1]; // id=8, sys=0
    msg_bytes.push(0); // team = 0
    msg_bytes.push(0); // client_id = 0
    msg_bytes.extend_from_slice(b"hi\0");
    write_chunk(&mut out, &huffman, 2, &message_bytes_to_ints(&msg_bytes));

    out
}
