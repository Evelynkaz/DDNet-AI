//! Real-traffic test: task acceptance criterion 4.
//!
//! # How the fixture was produced
//!
//! 1. `sudo tcpdump -i lo -w ddnet-loopback.pcap udp port 8303` against the local
//!    `ddnet-local.service` (DDNet 20.1, `127.0.0.1:8303`, never restarted/reconfigured for this).
//! 2. In parallel: `cd ~/aiddnet/DDNet-AI && node src/bot/main.ts --server 127.0.0.1:8303
//!    --scripted --duration 20 --no-console` (the old TypeScript bot) — a full real session over
//!    2778 UDP datagrams: handshake, an **in-protocol map download** (the map was not already
//!    cached — a first, wrong assumption in an earlier version of this test/`docs/formats.md`;
//!    corrected here), `READY`/`ENTERGAME`, ~1.3k datagrams of steady-state play, and a clean
//!    client-initiated `CLOSE` at the very end.
//! 3. `tests/fixtures/extract_capture.py ddnet-loopback.pcap
//!    tests/fixtures/local-capture-20260927.dat 8303 0:40 1414:2777` — a small, dependency-free
//!    script (stdlib only) that parses the pcap's Ethernet/IPv4/UDP framing itself, keeps only
//!    the UDP payloads exchanged with port 8303, and writes out two independently-contiguous
//!    **segments** in a compact custom format (magic `b"DDCAP2"`, `u8` segment count, then per
//!    segment: `u32` LE record count, then per record: `u8` direction — 0 = client→server, 1 =
//!    server→client — `u16` LE payload length, payload bytes; parsed below in
//!    [`read_fixture`]).
//! 4. `cargo run -p ddai-net --example strip_map_data -- tests/fixtures/local-capture-20260927.dat
//!    tests/fixtures/local-capture-20260927.dat` (review round 2, finding F10) — the range above
//!    keeps the *mechanism* of the first few `REQUEST_MAP_DATA`/`MAP_DATA` round trips (step 3's
//!    own reasoning, right below), but that means 25 real chunks (22,375 B) of `Copy Love Box`'s
//!    actual, third-party map bytes were sitting in the committed fixture doing nothing for any
//!    test here. This step zeroes exactly those 25 chunks' raw payload in place — see
//!    `examples/strip_map_data.rs`'s module docs for exactly what does and does not change, and
//!    `tests/no_third_party_map_bytes.rs` for the permanent CI check that no non-zero byte of
//!    them is still in the committed file.
//!
//! **Why two segments, not one contiguous prefix** (this is the fix for a real finding — an
//! earlier version of this test took a single contiguous prefix of the first 400 packets and
//! documented it as "handshake + ~2s of steady-state snapshots"; it was actually **entirely** the
//! `TKEN` handshake plus the *start* of the map download, since that map alone took over 1400
//! packets to transfer in-protocol — there is no snapshot, no `READY`/`ENTERGAME`, and not a
//! single non-vital chunk anywhere in that range): [`SEGMENT_HANDSHAKE`] is source records `0..=40`
//! (the handshake proper, plus the first few `REQUEST_MAP_DATA`/`MAP_DATA` round trips, showing
//! that mechanism without paying for its full ~1400-packet length), and [`SEGMENT_STEADY_STATE`]
//! is source records `1414..=2777` — starting at the client's `READY`, through `CON_READY`,
//! `Cl_StartInfo`, `ENTERGAME`, then genuine steady-state play (snapshots, `INPUTTIMING`, client
//! `INPUT`, acks) all the way to the session's own clean `CLOSE`. Combined: 1405 records, ~60 KB
//! after step 4 strips the third-party map bytes (~76 KB before), comfortably under the task's
//! 200 KB ceiling. Sequence/ack continuity is only meaningful
//! *within* a segment (the excised map-download packets are real sequence numbers we no longer
//! have), so every check below tracks state per segment, resetting at each segment boundary — see
//! [`check_segment`].
//!
//! Only our own local loopback traffic between our own local server and our own bot — no public
//! server or third-party player was ever involved (`docs/SETUP.md`, `docs/STATUS.md`).
//!
//! # Two documented exceptions (not bugs — see the assertions below for exactly where and why)
//!
//! * One client→server datagram in the handshake segment (7 bytes: flags=0, num_chunks=0, no
//!   `RESEND`) fails our decoder's structural validity check — and would fail the real DDNet
//!   server's identical check too (`IsValidConnectionOrientedPacket`, `network.cpp:150-154`: a
//!   non-resend packet must carry at least one chunk). The client here is the npm `teeworlds`
//!   library the old TypeScript bot uses, not DDNet's own C++ client; it sends this once. No
//!   visible effect on the session (the ack it would have carried is repeated on the next valid
//!   packet).
//! * Two server→client datagrams in the steady-state segment are **connectionless** (the
//!   `CONNLESS` packet flag set, `10× 0xFF` legacy serverbrowse header, `"iext"`/version-string
//!   payload) — the old bot's library independently pings the server for a server-info response
//!   outside the game connection proper, on the same port. [`packet::unpack_packet`] correctly
//!   refuses to decode these as connection-oriented (that's a different function,
//!   [`packet::unpack_connless_packet`], exercised separately below) — they carry no ack/token/
//!   chunk-sequence data and are excluded from those checks accordingly.
//!
//! # What this test checks (and why it is a *stronger* check than the synthetic tests elsewhere)
//!
//! * **100% decode** (modulo the two documented, explained exceptions above): every other
//!   datagram parses successfully with [`packet::unpack_packet`] — a real DDNet 20.1 server and a
//!   real (if old/TS) DDNet client never produced anything else our decoder rejects.
//! * **Token handling matches**: the negotiated security token is read out of the real
//!   `CONNECTACCEPT` exactly the way [`ddai_net::conn::Connection`] does, and then every
//!   connection-oriented datagram after the handshake has its trailing 4 bytes checked
//!   byte-for-byte against that token.
//! * **Huffman round-trips byte-identically**: every datagram with the compression flag set is
//!   decompressed with our [`Huffman`], then *recompressed*, and the result must equal the
//!   original wire bytes exactly — checked against **real C++-produced output**, not just our own
//!   encoder's self-consistency (`tests/oracle_libtw2.rs` already covers self-consistency against
//!   an independent Rust implementation; this is the one place we cross-check against the actual
//!   DDNet 20.1 binary) — with one disclosed exception: the 25 records step 4 above rewrote to
//!   zero third-party map bytes are, by construction, *our own* `Huffman::compress_vec`'s output
//!   (of the zeroed content) rather than the original DDNet-produced bytes for those specific 25
//!   records — this check still passes for them (trivially: decompressing-then-recompressing our
//!   own already-canonical output reproduces itself), it just no longer *proves* anything about
//!   the real server's encoder for those 25. The other ~1380 records in this fixture were never
//!   touched and still are exactly what DDNet 20.1 and the old TS client actually put on the
//!   wire.
//! * **Chunk sequences/acks are consistent**: within each segment, every vital chunk's sequence
//!   number is exactly the previous one (from the same direction, in the same segment) plus one,
//!   and every ack, once we have seen at least one vital chunk from the acked direction within
//!   this segment, never claims a sequence number higher than the highest that direction has
//!   actually sent so far in this segment — a real, non-tautological check (see [`check_segment`]
//!   for why "ack < MAX_SEQUENCE" alone, true of every 10-bit field, would not be one).

use ddai_net::conn::SecurityToken;
use ddai_net::huffman::Huffman;
use ddai_net::packet::{self, packet_flags};

const FIXTURE: &[u8] = include_bytes!("fixtures/local-capture-20260927.dat");

/// Index into [`read_fixture`]'s result for the handshake + start-of-map-download segment.
const SEGMENT_HANDSHAKE: usize = 0;
/// Index into [`read_fixture`]'s result for the `READY`-through-`CLOSE` steady-state segment.
const SEGMENT_STEADY_STATE: usize = 1;

/// Relative index, within [`SEGMENT_HANDSHAKE`], of the one documented non-decoding client→server
/// datagram (see the module docs).
const HANDSHAKE_INVALID_RECORD: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    ClientToServer,
    ServerToClient,
}

struct Record {
    direction: Direction,
    payload: Vec<u8>,
}

fn read_fixture(bytes: &[u8]) -> Vec<Vec<Record>> {
    assert_eq!(&bytes[..6], b"DDCAP2", "fixture magic mismatch");
    let num_segments = bytes[6] as usize;
    let mut pos = 7usize;
    let mut segments = Vec::with_capacity(num_segments);
    for _ in 0..num_segments {
        let count = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        pos += 4;
        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let direction = match bytes[pos] {
                0 => Direction::ClientToServer,
                1 => Direction::ServerToClient,
                other => panic!("unknown direction byte {other} in fixture"),
            };
            let len = u16::from_le_bytes([bytes[pos + 1], bytes[pos + 2]]) as usize;
            pos += 3;
            let payload = bytes[pos..pos + len].to_vec();
            pos += len;
            records.push(Record { direction, payload });
        }
        segments.push(records);
    }
    assert_eq!(pos, bytes.len(), "trailing garbage after the last fixture record");
    segments
}

/// The `CONNLESS` packet flag is visible directly in the first header byte, independent of
/// whether the rest of the packet parses as connection-oriented or not — used to route the two
/// documented connectionless records (see the module docs) to [`packet::unpack_connless_packet`]
/// instead of [`packet::unpack_packet`], and to exclude them from every connection-state check
/// below (they carry no ack, token, or chunk sequence).
fn is_connless(payload: &[u8]) -> bool {
    !payload.is_empty() && (payload[0] >> 2) & packet_flags::CONNLESS != 0
}

#[test]
fn fixture_is_under_the_200kb_ceiling() {
    assert!(
        FIXTURE.len() < 200_000,
        "fixture grew to {} bytes, over the task's 200 KB ceiling",
        FIXTURE.len()
    );
}

#[test]
fn every_captured_datagram_decodes_except_two_documented_exceptions() {
    let huffman = Huffman::new();
    let segments = read_fixture(FIXTURE);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[SEGMENT_HANDSHAKE].len(), 41);
    assert_eq!(segments[SEGMENT_STEADY_STATE].len(), 1364);

    let mut hard_failures = Vec::new();
    let mut connless_seen = Vec::new();

    for (seg_idx, segment) in segments.iter().enumerate() {
        for (i, record) in segment.iter().enumerate() {
            if is_connless(&record.payload) {
                match packet::unpack_connless_packet(&record.payload) {
                    Ok(parsed) => connless_seen.push((seg_idx, i, parsed)),
                    Err(e) => hard_failures.push((seg_idx, i, record.direction, record.payload.len(), format!("{e}"))),
                }
                continue;
            }
            if seg_idx == SEGMENT_HANDSHAKE && i == HANDSHAKE_INVALID_RECORD {
                // The one documented `InvalidStructure` exception — checked precisely below,
                // not swallowed here.
                continue;
            }
            if let Err(e) = packet::unpack_packet(&record.payload, &huffman, true) {
                hard_failures.push((seg_idx, i, record.direction, record.payload.len(), format!("{e}")));
            }
        }
    }

    assert!(
        hard_failures.is_empty(),
        "undocumented decode failures:\n{hard_failures:#?}"
    );

    // The two connectionless records: both server->client, both a legacy serverbrowse response.
    assert_eq!(
        connless_seen.len(),
        2,
        "expected exactly two connectionless records, got {connless_seen:#?}"
    );
    for (seg_idx, i, parsed) in &connless_seen {
        assert_eq!(*seg_idx, SEGMENT_STEADY_STATE);
        assert_eq!(segments[*seg_idx][*i].direction, Direction::ServerToClient);
        // `iext`/`inf3`/`iex+` per `masterserver.cpp:3-10`; the response we captured is `iext`.
        assert!(
            parsed
                .data
                .windows(4)
                .any(|w| w == b"iext" || w == b"inf3" || w == b"iex+"),
            "connless payload did not look like a serverbrowse info response: {:?}",
            String::from_utf8_lossy(&parsed.data)
        );
    }

    // And the one documented structural-validity exception, checked precisely.
    let handshake = &segments[SEGMENT_HANDSHAKE];
    let bad = &handshake[HANDSHAKE_INVALID_RECORD];
    assert_eq!(bad.direction, Direction::ClientToServer);
    assert_eq!(bad.payload.len(), 7);
    assert_eq!(
        packet::unpack_packet(&bad.payload, &huffman, true).unwrap_err(),
        packet::UnpackPacketError::InvalidStructure
    );
}

#[test]
fn token_handling_matches_real_handshake() {
    let huffman = Huffman::new();
    let segments = read_fixture(FIXTURE);
    let handshake = &segments[SEGMENT_HANDSHAKE];

    // Record 0: the client's CONNECT. Verify it is exactly what we'd build ourselves.
    let connect = packet::unpack_packet(&handshake[0].payload, &huffman, true).unwrap();
    assert_ne!(connect.flags & packet_flags::CONTROL, 0);
    assert_eq!(
        connect.data.len(),
        1 + 4 + 4,
        "CONNECT: type + TKEN + trailing UNKNOWN token"
    );
    assert_eq!(&connect.data[1..5], b"TKEN");
    let unknown_token = u32::from_be_bytes([connect.data[5], connect.data[6], connect.data[7], connect.data[8]]);
    assert_eq!(
        unknown_token,
        ddai_net::control::TOKEN_UNKNOWN,
        "old TS bot also sends the UNKNOWN sentinel first"
    );

    // Record 1: the server's CONNECTACCEPT. This is where we learned the hard way (see the
    // `control` module docs) that DDNet does *not* embed the token as message data — read it the
    // same way `Connection::feed` does, straight out of the unstripped trailing bytes.
    let connect_accept = packet::unpack_packet(&handshake[1].payload, &huffman, true).unwrap();
    assert_eq!(
        connect_accept.data.len(),
        1 + 4 + 4,
        "CONNECTACCEPT: type + TKEN + trailing token (no duplicate)"
    );
    assert_eq!(&connect_accept.data[1..5], b"TKEN");
    let token = u32::from_be_bytes([
        connect_accept.data[5],
        connect_accept.data[6],
        connect_accept.data[7],
        connect_accept.data[8],
    ]);
    let security_token = SecurityToken::Known(token);

    // Record 2: the client's ACCEPT, carrying that same token as its own trailing bytes.
    let accept = packet::unpack_packet(&handshake[2].payload, &huffman, true).unwrap();
    assert_eq!(accept.data.len(), 1 + 4);
    assert_eq!(accept.data[0], ddai_net::control::ctrl_msg::ACCEPT);
    let accept_token = u32::from_be_bytes([accept.data[1], accept.data[2], accept.data[3], accept.data[4]]);
    assert_eq!(SecurityToken::Known(accept_token), security_token);

    // Every subsequent connection-oriented datagram, in *both* segments, must carry that exact
    // token as its last 4 bytes — except the documented non-decoding/connectionless records,
    // which have no such trailing bytes to check.
    let mut checked = 0usize;
    for (seg_idx, segment) in segments.iter().enumerate() {
        let start = if seg_idx == SEGMENT_HANDSHAKE { 3 } else { 0 };
        for (i, record) in segment.iter().enumerate().skip(start) {
            if is_connless(&record.payload) || (seg_idx == SEGMENT_HANDSHAKE && i == HANDSHAKE_INVALID_RECORD) {
                continue;
            }
            let parsed = packet::unpack_packet(&record.payload, &huffman, true).unwrap();
            assert!(
                parsed.data.len() >= 4,
                "segment {seg_idx} record {i} too short to carry a trailing token"
            );
            let tail = &parsed.data[parsed.data.len() - 4..];
            let got = u32::from_be_bytes([tail[0], tail[1], tail[2], tail[3]]);
            assert_eq!(
                got, token,
                "segment {seg_idx} record {i} ({:?}) trailing token mismatch",
                record.direction
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 41 - 3 - 1 + 1364 - 2);
}

#[test]
fn huffman_round_trips_every_compressed_real_datagram_byte_identically() {
    let huffman = Huffman::new();
    let segments = read_fixture(FIXTURE);

    let mut compressed_count = 0usize;
    for (seg_idx, segment) in segments.iter().enumerate() {
        for (i, record) in segment.iter().enumerate() {
            if is_connless(&record.payload) {
                continue;
            }
            let flags = record.payload[0] >> 2;
            if flags & packet_flags::COMPRESSION == 0 {
                continue;
            }
            compressed_count += 1;
            let wire_compressed = &record.payload[packet::PACKET_HEADER_SIZE..];
            let decompressed = huffman.decompress_vec(wire_compressed, packet::MAX_CHUNK_DATA_SIZE).unwrap_or_else(|| {
                panic!("segment {seg_idx} record {i}: our Huffman failed to decompress a real DDNet-compressed datagram")
            });
            let recompressed = huffman.compress_vec(&decompressed);
            assert_eq!(
                recompressed, wire_compressed,
                "segment {seg_idx} record {i}: recompressing what we decompressed did not reproduce the real server's bytes exactly"
            );
        }
    }
    assert!(
        compressed_count > 0,
        "expected at least one compressed datagram in the fixture to make this test meaningful"
    );
}

/// Per-segment chunk-sequence/ack consistency (see the module docs for why per-segment, not
/// global). Returns the number of vital chunks observed per direction (`[client, server]`), for
/// the caller to sanity-check both directions actually exercised this.
fn check_segment(huffman: &Huffman, segment: &[Record], skip_relative: Option<usize>) -> [usize; 2] {
    let mut last_vital_seq: [Option<u16>; 2] = [None, None];
    let mut max_seq_sent: [Option<u16>; 2] = [None, None]; // highest vital seq sent so far, this segment
    let mut vital_count = [0usize; 2];

    for (i, record) in segment.iter().enumerate() {
        if Some(i) == skip_relative || is_connless(&record.payload) {
            continue;
        }
        let parsed = packet::unpack_packet(&record.payload, huffman, true).unwrap();
        let dir_idx = match record.direction {
            Direction::ClientToServer => 0,
            Direction::ServerToClient => 1,
        };
        let other_idx = 1 - dir_idx;

        // Non-tautological ack check (fixes a prior version of this test that only asserted
        // `ack < MAX_SEQUENCE`, true of *every* 10-bit field): once we've seen the acked
        // direction send at least one vital chunk in this segment, an ack must never claim a
        // sequence number higher than the highest it has actually sent so far. `ack == 0` is
        // always trivially fine ("nothing acked yet"). This fixture's sessions never wrap the
        // 10-bit sequence space (well under 1024 vital chunks total per direction), so a plain
        // numeric comparison is correct here; a general-purpose version would need
        // `packet::is_seq_in_backroom`-style wraparound handling instead.
        if parsed.ack != 0
            && let Some(max_sent) = max_seq_sent[other_idx]
        {
            assert!(
                parsed.ack <= max_sent,
                "record {i} ({:?}): ack {} exceeds the highest sequence {:?} has actually sent so far ({max_sent})",
                record.direction,
                parsed.ack,
                record.direction
            );
        }

        if parsed.flags & packet_flags::CONTROL == 0 && parsed.num_chunks > 0 {
            let with_token_stripped = &parsed.data[..parsed.data.len() - 4];
            let chunk_packet = packet::Packet {
                flags: parsed.flags,
                ack: parsed.ack,
                num_chunks: parsed.num_chunks,
                data: with_token_stripped.to_vec(),
            };
            for chunk in packet::ChunkIter::new(&chunk_packet) {
                if !chunk.vital {
                    continue;
                }
                vital_count[dir_idx] += 1;
                if let Some(prev) = last_vital_seq[dir_idx] {
                    let expected = (prev + 1) % packet::MAX_SEQUENCE;
                    assert_eq!(
                        chunk.sequence, expected,
                        "record {i} ({:?}): vital sequence jumped from {prev} to {}, expected {expected}",
                        record.direction, chunk.sequence
                    );
                }
                last_vital_seq[dir_idx] = Some(chunk.sequence);
                max_seq_sent[dir_idx] = Some(chunk.sequence);
            }
        }
    }

    vital_count
}

#[test]
fn chunk_sequences_are_strictly_consecutive_per_direction_and_acks_reference_real_sends() {
    let huffman = Huffman::new();
    let segments = read_fixture(FIXTURE);

    let handshake_counts = check_segment(&huffman, &segments[SEGMENT_HANDSHAKE], Some(HANDSHAKE_INVALID_RECORD));
    assert!(
        handshake_counts[0] > 0 && handshake_counts[1] > 0,
        "expected vital chunks in both directions in the handshake segment, got {handshake_counts:?}"
    );

    let steady_state_counts = check_segment(&huffman, &segments[SEGMENT_STEADY_STATE], None);
    assert!(
        steady_state_counts[0] > 0 && steady_state_counts[1] > 0,
        "expected vital chunks in both directions in the steady-state segment, got {steady_state_counts:?}"
    );
}
