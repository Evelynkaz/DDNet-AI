//! Dev tool — review round 2, finding F10: this crate's two committed real-traffic fixtures each
//! embedded a handful of genuine `MAP_DATA` chunks (third-party DDNet map bytes, `BlmapChill`/
//! `Copy Love Box`) that no test in this crate reads the content of. This tool rewrites a fixture
//! (already in this crate's own `DDCAP2` format — the same one `extract_capture.py` produces) to
//! zero every `MAP_DATA` chunk's raw payload bytes in place, keeping every other byte — packet
//! and chunk framing, sequence numbers, acks, the security token, every non-map message — exactly
//! as captured. No record is added, removed, resized, or renumbered: a `Sv_MapData` message's raw
//! `data` field is always the *last* thing packed into its chunk (see `sysmsg::encode`), so
//! zeroing its trailing `data.len()` bytes changes nothing about that chunk's declared `size`, its
//! header, or its neighbours.
//!
//! **Why this is a Rust example inside `ddai-net`, not a pure-Python addition to
//! `extract_capture.py`** (as first suggested): doing this correctly means Huffman-decompressing
//! and then re-compressing every affected packet, which needs a *decoder and encoder* for DDNet's
//! exact static Huffman table. This crate already has one — fuzzed (`tests/robustness.rs`) and
//! differentially cross-checked against `libtw2-huffman` (`tests/oracle_libtw2.rs`).
//! Re-implementing that a second time in Python (`extract_capture.py` is deliberately
//! stdlib-only, see its own docstring) purely to throw bytes away would be a second, unaudited
//! Huffman implementation for strictly more risk and no benefit. This tool reuses the exact code
//! path this crate exists to provide, and is meant to run as the second step of the same overall
//! extraction workflow `extract_capture.py` starts (pcap -> raw `.dat` -> this tool -> committed
//! `.dat`), not as a replacement for it.
//!
//! Usage: `cargo run -p ddai-net --example strip_map_data -- <in.dat> <out.dat>`
//!
//! Prints, per input record that carried at least one `MAP_DATA` chunk: how many chunks in that
//! record were zeroed and how many bytes. Exits non-zero (via `expect`/`assert`) rather than
//! silently producing a fixture that would decode differently from the input in any way other
//! than the intended zeroing — this is dev tooling for a two-fixture repository, not a general
//! archival utility, so failing loudly beats guessing.

use ddai_net::control::ctrl_msg;
use ddai_net::huffman::Huffman;
use ddai_net::message::{self, Msg, Registry};
use ddai_net::packet::{self, ChunkHeader, packet_flags};
use ddai_net::sysmsg::SysMsg;

type Segment = Vec<(u8, Vec<u8>)>;

fn read_ddcap2(bytes: &[u8]) -> Vec<Segment> {
    assert_eq!(&bytes[..6], b"DDCAP2", "not a DDCAP2 fixture");
    let num_segments = bytes[6] as usize;
    let mut pos = 7usize;
    let mut segments = Vec::with_capacity(num_segments);
    for _ in 0..num_segments {
        let count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let dir = bytes[pos];
            let len = u16::from_le_bytes([bytes[pos + 1], bytes[pos + 2]]) as usize;
            pos += 3;
            records.push((dir, bytes[pos..pos + len].to_vec()));
            pos += len;
        }
        segments.push(records);
    }
    assert_eq!(pos, bytes.len(), "trailing garbage after the last fixture record");
    segments
}

fn write_ddcap2(segments: &[Segment]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"DDCAP2");
    out.push(u8::try_from(segments.len()).expect("segment count fits in a byte"));
    for segment in segments {
        out.extend_from_slice(&u32::try_from(segment.len()).unwrap().to_le_bytes());
        for (dir, payload) in segment {
            let len = u16::try_from(payload.len()).expect("a single UDP datagram always fits in a u16");
            out.push(*dir);
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(payload);
        }
    }
    out
}

fn is_connless(payload: &[u8]) -> bool {
    !payload.is_empty() && (payload[0] >> 2) & packet_flags::CONNLESS != 0
}

/// Finds the session's negotiated security token from the server's `CONNECTACCEPT` (never the
/// client's `CONNECT`, which carries only the `TOKEN_UNKNOWN` sentinel — both have the same
/// `TKEN`-tagged shape, so the control-message type byte is what tells them apart).
fn find_token(segments: &[Segment], huffman: &Huffman) -> Option<u32> {
    for segment in segments {
        for (_dir, payload) in segment {
            if is_connless(payload) {
                continue;
            }
            let Ok(parsed) = packet::unpack_packet(payload, huffman, true) else {
                continue;
            };
            if parsed.flags & packet_flags::CONTROL == 0 {
                continue;
            }
            if parsed.data.first() == Some(&ctrl_msg::CONNECTACCEPT)
                && parsed.data.len() >= 9
                && &parsed.data[1..5] == b"TKEN"
            {
                return Some(u32::from_be_bytes([
                    parsed.data[5],
                    parsed.data[6],
                    parsed.data[7],
                    parsed.data[8],
                ]));
            }
        }
    }
    None
}

struct StripResult {
    new_payload: Vec<u8>,
    chunks_zeroed: usize,
    bytes_zeroed: usize,
}

/// Returns `Some` (with the rebuilt wire bytes) only if `payload` decoded as a token-verified,
/// non-control connection-oriented packet carrying at least one `MAP_DATA` chunk; `None` means
/// "leave this record exactly as captured" — connless probes, control packets, and every ordinary
/// record with no map data in it are never touched, so the vast majority of both fixtures'
/// records stay pristine, real DDNet-produced bytes.
fn strip_record(payload: &[u8], huffman: &Huffman, registry: &Registry, token: u32) -> Option<StripResult> {
    if is_connless(payload) {
        return None;
    }
    let parsed = packet::unpack_packet(payload, huffman, true).ok()?;
    if parsed.flags & packet_flags::CONTROL != 0 {
        return None;
    }
    if parsed.data.len() < 4 {
        return None;
    }
    let split = parsed.data.len() - 4;
    let tail = &parsed.data[split..];
    if u32::from_be_bytes([tail[0], tail[1], tail[2], tail[3]]) != token {
        return None; // never observed in our fixtures, but fail safe (leave untouched) if it ever is
    }
    let chunk_data = &parsed.data[..split];

    // Pass 1 (read-only): does this record carry a MAP_DATA chunk at all? Only records that do
    // ever lose "pristine captured bytes" status.
    let mut any_map_data = false;
    {
        let mut pos = 0usize;
        while let Some((header, hdr_len)) = ChunkHeader::unpack_default(&chunk_data[pos..]) {
            let start = pos + hdr_len;
            let end = start + header.size as usize;
            if end > chunk_data.len() {
                break;
            }
            let (msg, _) = message::decode(&chunk_data[start..end], registry);
            if matches!(msg, Msg::Sys(SysMsg::MapData { .. })) {
                any_map_data = true;
            }
            pos = end;
        }
    }
    if !any_map_data {
        return None;
    }

    // Pass 2: rebuild every chunk. Non-MAP_DATA chunks are re-packed byte-for-byte identical
    // (same header bits incl. RESEND, same sequence, same payload) — only a MAP_DATA chunk's
    // trailing `data.len()` bytes are zeroed; the leading `last`/`crc`/`chunk`/`size` ints and the
    // chunk header are untouched, so the chunk's on-wire byte length never changes.
    let mut new_chunk_data = Vec::with_capacity(chunk_data.len());
    let mut num_chunks = 0usize;
    let mut chunks_zeroed = 0usize;
    let mut bytes_zeroed = 0usize;
    let mut pos = 0usize;
    while let Some((header, hdr_len)) = ChunkHeader::unpack_default(&chunk_data[pos..]) {
        let start = pos + hdr_len;
        let end = start + header.size as usize;
        if end > chunk_data.len() {
            break;
        }
        let body = &chunk_data[start..end];
        let (msg, _) = message::decode(body, registry);
        let mut new_body = body.to_vec();
        if let Msg::Sys(SysMsg::MapData { data, .. }) = &msg {
            let tail = data.len();
            let at = new_body.len() - tail;
            for b in &mut new_body[at..] {
                *b = 0;
            }
            chunks_zeroed += 1;
            bytes_zeroed += tail;
        }
        let mut hdr_buf = [0u8; 3];
        let written = header
            .pack_default(&mut hdr_buf)
            .expect("a header we just unpacked always re-packs");
        new_chunk_data.extend_from_slice(&hdr_buf[..written]);
        new_chunk_data.extend_from_slice(&new_body);
        num_chunks += 1;
        pos = end;
    }
    assert_eq!(
        pos,
        chunk_data.len(),
        "trailing bytes after the last chunk header we could parse"
    );
    assert_eq!(
        num_chunks, parsed.num_chunks as usize,
        "chunk count changed while rebuilding"
    );

    let rebuilt = packet::build_packet(
        parsed.flags & !packet_flags::COMPRESSION,
        parsed.ack,
        parsed.num_chunks,
        &new_chunk_data,
        Some(token),
        huffman,
    )
    .expect("zeroing bytes never makes a packet larger than the original");

    Some(StripResult {
        new_payload: rebuilt,
        chunks_zeroed,
        bytes_zeroed,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, in_path, out_path] = args.as_slice() else {
        eprintln!("usage: strip_map_data <in.dat> <out.dat>");
        std::process::exit(1);
    };

    let bytes = std::fs::read(in_path).unwrap_or_else(|e| panic!("reading {in_path}: {e}"));
    let mut segments = read_ddcap2(&bytes);
    let huffman = Huffman::new();
    let registry = Registry::new();

    let token = find_token(&segments, &huffman).expect("no CONNECTACCEPT with a real token found in this fixture");
    println!("negotiated token: {token:#010x}");

    let mut records_touched = 0usize;
    let mut total_chunks_zeroed = 0usize;
    let mut total_bytes_zeroed = 0usize;
    let before_len = bytes.len();

    for (seg_idx, segment) in segments.iter_mut().enumerate() {
        for (rec_idx, (_dir, payload)) in segment.iter_mut().enumerate() {
            if let Some(result) = strip_record(payload, &huffman, &registry, token) {
                println!(
                    "  segment {seg_idx} record {rec_idx}: zeroed {} MAP_DATA chunk(s), {} byte(s) ({} -> {} wire bytes)",
                    result.chunks_zeroed,
                    result.bytes_zeroed,
                    payload.len(),
                    result.new_payload.len()
                );
                records_touched += 1;
                total_chunks_zeroed += result.chunks_zeroed;
                total_bytes_zeroed += result.bytes_zeroed;
                *payload = result.new_payload;
            }
        }
    }

    let out_bytes = write_ddcap2(&segments);
    std::fs::write(out_path, &out_bytes).unwrap_or_else(|e| panic!("writing {out_path}: {e}"));

    println!(
        "records touched: {records_touched}, MAP_DATA chunks zeroed: {total_chunks_zeroed}, bytes zeroed: {total_bytes_zeroed}"
    );
    println!("fixture size: {before_len} -> {} bytes", out_bytes.len());

    // Self-check: re-read what we just wrote and confirm no chunk in it still decodes as
    // MAP_DATA with any non-zero byte in its `data` field (the actual "scan" finding F10 asked
    // for — run automatically here, and independently again by
    // `tests/no_third_party_map_bytes.rs` against the committed fixtures).
    let reread = read_ddcap2(&out_bytes);
    let mut leftover_nonzero = 0usize;
    for segment in &reread {
        for (_dir, payload) in segment {
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
                    leftover_nonzero += data.iter().filter(|&&b| b != 0).count();
                }
                pos = end;
            }
        }
    }
    println!("post-write scan: non-zero bytes remaining across every MAP_DATA chunk: {leftover_nonzero}");
    assert_eq!(leftover_nonzero, 0, "stripping left non-zero MAP_DATA bytes behind");
}
