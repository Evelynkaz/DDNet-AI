//! Review round 2, finding F10: both real-traffic fixtures under `tests/fixtures/` embedded a
//! handful of genuine `MAP_DATA` chunks — third-party DDNet map bytes (`Copy Love Box`/
//! `BlmapChill`) that no test in this crate reads the content of. `examples/strip_map_data.rs`
//! (see its module docs for why it's a Rust example, not a pure-Python addition to
//! `extract_capture.py`) zeroes every such chunk's raw payload in place, changing nothing else
//! byte-for-byte (framing, sequence numbers, acks, the token, every non-map message).
//!
//! This is the permanent, CI-checked half of "show with a scan that no `MAP_DATA` payload bytes
//! remain" — `strip_map_data` itself already asserts this once, right after writing a fixture,
//! but that only runs when someone re-runs the tool by hand; this test re-derives the same scan
//! independently against whatever is actually committed, so a future accidental re-extraction
//! (dropping this step) fails CI instead of silently reintroducing third-party map bytes.

use ddai_net::huffman::Huffman;
use ddai_net::message::{self, Msg, Registry};
use ddai_net::packet::{self, ChunkHeader, packet_flags};
use ddai_net::sysmsg::SysMsg;

const FIXTURE_2_2A: &[u8] = include_bytes!("fixtures/local-capture-20260927.dat");
const FIXTURE_MAPCHANGE: &[u8] = include_bytes!("fixtures/local-capture-mapchange-20260927.dat");

fn read_ddcap2(bytes: &[u8]) -> Vec<Vec<Vec<u8>>> {
    assert_eq!(&bytes[..6], b"DDCAP2");
    let num_segments = bytes[6] as usize;
    let mut pos = 7usize;
    let mut segments = Vec::with_capacity(num_segments);
    for _ in 0..num_segments {
        let count = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        pos += 4;
        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let len = u16::from_le_bytes([bytes[pos + 1], bytes[pos + 2]]) as usize;
            pos += 3;
            records.push(bytes[pos..pos + len].to_vec());
            pos += len;
        }
        segments.push(records);
    }
    assert_eq!(pos, bytes.len(), "trailing garbage after the last fixture record");
    segments
}

fn is_connless(payload: &[u8]) -> bool {
    !payload.is_empty() && (payload[0] >> 2) & packet_flags::CONNLESS != 0
}

/// Scans every record of every segment for `MAP_DATA` chunks and returns
/// `(map_data_chunks_seen, total_data_bytes_seen, non_zero_data_bytes_seen)`.
fn scan(fixture: &[u8]) -> (usize, usize, usize) {
    let huffman = Huffman::new();
    let registry = Registry::new();
    let segments = read_ddcap2(fixture);

    let mut chunks_seen = 0usize;
    let mut total_bytes = 0usize;
    let mut non_zero_bytes = 0usize;

    for segment in &segments {
        for payload in segment {
            if is_connless(payload) {
                continue;
            }
            let Ok(parsed) = packet::unpack_packet(payload, &huffman, true) else {
                continue;
            };
            if parsed.flags & packet_flags::CONTROL != 0 || parsed.data.len() < 4 {
                continue;
            }
            let chunk_data = &parsed.data[..parsed.data.len() - 4];
            let mut pos = 0usize;
            while let Some((header, hdr_len)) = ChunkHeader::unpack_default(&chunk_data[pos..]) {
                let start = pos + hdr_len;
                let end = start + header.size as usize;
                if end > chunk_data.len() {
                    break;
                }
                let (msg, _) = message::decode(&chunk_data[start..end], &registry);
                if let Msg::Sys(SysMsg::MapData { data, .. }) = &msg {
                    chunks_seen += 1;
                    total_bytes += data.len();
                    non_zero_bytes += data.iter().filter(|&&b| b != 0).count();
                }
                pos = end;
            }
        }
    }
    (chunks_seen, total_bytes, non_zero_bytes)
}

#[test]
fn local_capture_20260927_has_no_third_party_map_bytes() {
    let (chunks, total, non_zero) = scan(FIXTURE_2_2A);
    println!("local-capture-20260927.dat: {chunks} MAP_DATA chunk(s), {total} byte(s) total, {non_zero} non-zero");
    // Still has real MAP_DATA *messages* (the mechanism this fixture documents, per
    // `tests/capture.rs`'s module docs) — only their content is gone.
    assert_eq!(
        chunks, 25,
        "expected the documented 25 MAP_DATA chunks to still be present as messages"
    );
    assert_eq!(
        total, 22375,
        "expected the documented 22,375 B of (now zeroed) MAP_DATA payload"
    );
    assert_eq!(
        non_zero, 0,
        "third-party map bytes leaked back into the committed fixture"
    );
}

#[test]
fn local_capture_mapchange_20260927_has_no_third_party_map_bytes() {
    let (chunks, total, non_zero) = scan(FIXTURE_MAPCHANGE);
    println!(
        "local-capture-mapchange-20260927.dat: {chunks} MAP_DATA chunk(s), {total} byte(s) total, {non_zero} non-zero"
    );
    assert_eq!(
        chunks, 49,
        "expected the documented 49 MAP_DATA chunks (11 BlmapChill + 38 Copy Love Box)"
    );
    assert_eq!(
        total, 43855,
        "expected the documented 43,855 B of (now zeroed) MAP_DATA payload"
    );
    assert_eq!(
        non_zero, 0,
        "third-party map bytes leaked back into the committed fixture"
    );
}

#[test]
fn both_fixtures_stay_comfortably_under_their_size_ceilings() {
    assert!(
        FIXTURE_2_2A.len() < 200_000,
        "local-capture-20260927.dat grew to {} bytes, over the 200 KB ceiling",
        FIXTURE_2_2A.len()
    );
    assert!(
        FIXTURE_MAPCHANGE.len() < 500_000,
        "local-capture-mapchange-20260927.dat grew to {} bytes, over the 500 KB ceiling",
        FIXTURE_MAPCHANGE.len()
    );
}
