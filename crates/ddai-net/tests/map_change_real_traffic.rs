//! Real-traffic test with an actual map change — task 2.2b review round 1, finding F1's fix note
//! ("add a real-traffic test WITH a map change") plus F2's "test against the real tuning messages
//! in the fixture" and the round's request to "decode the reviewer's full map-change capture and
//! report snapshot counts per map, CRC mismatches, events" (done in full against the *original*
//! capture in this build's report; this test carries a compact, committed slice of that same
//! traffic so the finding stays covered by an automated test, not just a one-off report).
//!
//! Fixture: `tests/fixtures/local-capture-mapchange-20260927.dat` (< 500 KB, six independently-
//! contiguous segments — see `tests/fixtures/extract_capture.py`'s docs for the format) — a real
//! 70 s session (old TS bot vs `ddnet-local.service`) that changes map twice
//! (`Copy Love Box` -> `BlmapChill` -> `Copy Love Box`, `econ.py change_map`), trimmed the same way
//! task 2.2a's own fixture was: the handshake, a short pre-change steady-state window, a short
//! window right at each `MAP_CHANGE`, and a longer post-`ENTERGAME` steady-state window on each
//! map (skipping the bulk of each in-protocol map download in between, which is thousands of
//! repetitive `MAP_DATA` chunks and not needed to exercise this finding). The trimmed windows
//! still caught 49 real `MAP_DATA` chunks (11 `BlmapChill` + 38 `Copy Love Box`, 43,855 B) at
//! their edges; review round 2, finding F10: `cargo run -p ddai-net --example strip_map_data`
//! zeroed all of them in place afterwards (framing/sequence/ack/token untouched, see that
//! example's module docs) — `tests/no_third_party_map_bytes.rs` is the permanent CI check that
//! they stay zeroed.
//!
//! What this proves:
//! * without calling [`SnapAssembler::reset`] on `ENTERGAME` (task 2.3's job in a real session;
//!   this crate cannot call it for the caller), the vast majority of snapshots on a
//!   just-entered map are [`Event::Stale`] — the bug F1 reported, reproduced from a committed
//!   fixture rather than only the reviewer's own one-off capture;
//! * calling `reset()` exactly there recovers every one of them, with **0** CRC mismatches;
//! * the real `Sv_TuneParams` in this capture decode to sensible values (F2).

use ddai_net::assembly::{Event, SnapAssembler};
use ddai_net::delta::StaticSizes;
use ddai_net::huffman::Huffman;
use ddai_net::message::{self, Msg, Registry};
use ddai_net::packet::{self, packet_flags};
use ddai_net::sysmsg::SysMsg;
use std::collections::BTreeMap;

const FIXTURE: &[u8] = include_bytes!("fixtures/local-capture-mapchange-20260927.dat");
const SEG_HANDSHAKE: usize = 0;

fn read_fixture(bytes: &[u8]) -> Vec<Vec<Vec<u8>>> {
    assert_eq!(&bytes[..6], b"DDCAP2");
    let num_segments = bytes[6] as usize;
    let mut pos = 7usize;
    let mut segments = Vec::with_capacity(num_segments);
    for _ in 0..num_segments {
        let count = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        pos += 4;
        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            // direction byte at bytes[pos] is unused by this test (both directions are decoded).
            let len = u16::from_le_bytes([bytes[pos + 1], bytes[pos + 2]]) as usize;
            pos += 3;
            records.push(bytes[pos..pos + len].to_vec());
            pos += len;
        }
        segments.push(records);
    }
    segments
}

fn is_connless(payload: &[u8]) -> bool {
    !payload.is_empty() && (payload[0] >> 2) & packet_flags::CONNLESS != 0
}

fn token_from_handshake(huffman: &Huffman, handshake: &[Vec<u8>]) -> u32 {
    let connect_accept = packet::unpack_packet(&handshake[1], huffman, true).unwrap();
    assert_eq!(&connect_accept.data[1..5], b"TKEN");
    u32::from_be_bytes([
        connect_accept.data[5],
        connect_accept.data[6],
        connect_accept.data[7],
        connect_accept.data[8],
    ])
}

#[derive(Default)]
struct Outcome {
    snapshots_per_map: BTreeMap<String, u32>,
    stale_per_map: BTreeMap<String, u32>,
    crc_mismatches: u32,
    delta_errors: u32,
    resyncs: u32,
    tune_params_seen: Vec<ddai_net::tuning::TuneParams>,
}

/// Decodes the whole fixture once. `call_reset_on_entergame` toggles the fix under test (F1) —
/// `false` reproduces the bug, `true` is what a correct task 2.3 session driver does.
fn decode_fixture(call_reset_on_entergame: bool) -> Outcome {
    let huffman = Huffman::new();
    let registry = Registry::new();
    let segments = read_fixture(FIXTURE);
    let token = token_from_handshake(&huffman, &segments[SEG_HANDSHAKE]);

    let mut assembler = SnapAssembler::new(StaticSizes::ddnet_06());
    let mut outcome = Outcome::default();
    let mut current_map = String::from("?");

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
            let split = parsed.data.len() - 4;
            let tail = &parsed.data[split..];
            let got_token = u32::from_be_bytes([tail[0], tail[1], tail[2], tail[3]]);
            if got_token != token {
                continue; // a datagram from before the handshake segment's token was known, etc.
            }
            let chunk_packet = packet::Packet {
                flags: parsed.flags,
                ack: parsed.ack,
                num_chunks: parsed.num_chunks,
                data: parsed.data[..split].to_vec(),
            };
            for chunk in packet::ChunkIter::new(&chunk_packet) {
                let (msg, _answer) = message::decode(chunk.data, &registry);
                match &msg {
                    Msg::Sys(SysMsg::MapChange { name, .. }) => current_map = name.clone(),
                    Msg::Sys(SysMsg::EnterGame) if call_reset_on_entergame => assembler.reset(),
                    Msg::TuneParams(params) => outcome.tune_params_seen.push(*params),
                    _ => {}
                }
                if let Msg::Sys(sys_msg) = &msg
                    && let Some(event) = assembler.feed(sys_msg)
                {
                    match event {
                        Event::Snapshot { .. } => {
                            *outcome.snapshots_per_map.entry(current_map.clone()).or_default() += 1;
                        }
                        Event::Stale { .. } => {
                            *outcome.stale_per_map.entry(current_map.clone()).or_default() += 1;
                        }
                        Event::CrcMismatch { .. } => outcome.crc_mismatches += 1,
                        Event::DeltaError { .. } => outcome.delta_errors += 1,
                        Event::Resync { .. } => outcome.resyncs += 1,
                    }
                }
            }
        }
    }
    outcome
}

#[test]
fn without_reset_most_post_change_snapshots_are_stale_f1_repro() {
    let outcome = decode_fixture(false);
    println!("without reset(): snapshots_per_map = {:?}", outcome.snapshots_per_map);
    println!("without reset(): stale_per_map = {:?}", outcome.stale_per_map);

    // `Copy Love Box` is the map the session started on (before any change) *and* the map it
    // changes back to — its post-second-change snapshots are stale too, but the segment for the
    // very first (pre-change) window still contributes a few real ones before the bug ever has a
    // chance to strike, so we specifically check `BlmapChill` (only ever seen *after* a change).
    let blmapchill_snapshots = *outcome.snapshots_per_map.get("BlmapChill").unwrap_or(&0);
    let blmapchill_stale = *outcome.stale_per_map.get("BlmapChill").unwrap_or(&0);
    assert_eq!(
        blmapchill_snapshots, 0,
        "F1 repro: without reset(), BlmapChill must recover zero real snapshots"
    );
    assert!(
        blmapchill_stale > 20,
        "expected many stale BlmapChill snapshots, got {blmapchill_stale}"
    );
}

#[test]
fn with_reset_every_map_recovers_real_snapshots_with_zero_crc_mismatches() {
    let outcome = decode_fixture(true);
    println!("with reset(): snapshots_per_map = {:?}", outcome.snapshots_per_map);
    println!("with reset(): stale_per_map = {:?}", outcome.stale_per_map);
    println!(
        "crc_mismatches={} delta_errors={} resyncs={}",
        outcome.crc_mismatches, outcome.delta_errors, outcome.resyncs
    );

    assert_eq!(outcome.crc_mismatches, 0);
    assert_eq!(outcome.delta_errors, 0);
    let blmapchill = *outcome.snapshots_per_map.get("BlmapChill").unwrap_or(&0);
    let copy_love_box = *outcome.snapshots_per_map.get("Copy Love Box").unwrap_or(&0);
    assert!(
        blmapchill > 20,
        "expected real BlmapChill snapshots once reset() is called, got {blmapchill}"
    );
    assert!(
        copy_love_box > 20,
        "expected real Copy Love Box snapshots, got {copy_love_box}"
    );
    // `reset()` must not itself introduce spurious staleness on the map the caller is *already*
    // on (only across an actual `ENTERGAME` transition) — expect only a handful, if any, from
    // ordinary duplicate/dedup edge cases at the trimmed segment boundaries, never the dozens
    // seen without the fix.
    let total_stale: u32 = outcome.stale_per_map.values().sum();
    assert!(
        total_stale < 20,
        "expected few stale events with reset() in place, got {total_stale}"
    );
}

#[test]
fn real_tune_params_in_the_capture_decode_to_sensible_values() {
    let outcome = decode_fixture(true);
    assert!(
        !outcome.tune_params_seen.is_empty(),
        "expected at least one Sv_TuneParams in the fixture"
    );
    for params in &outcome.tune_params_seen {
        println!(
            "Sv_TuneParams: received={} gravity={} gun_curvature={} gun_speed={} hook_length={} velramp_start={}",
            params.received,
            params.gravity,
            params.gun_curvature,
            params.gun_speed,
            params.hook_length,
            params.velramp_start
        );
        // Every field the server actually sent is a full, real message (never an early-stop
        // partial one in this capture).
        assert_eq!(
            params.received,
            ddai_net::tuning::NUM_TUNE_PARAMS,
            "expected a full Sv_TuneParams payload"
        );
        // These four are DDNet's stock `tuning.h` defaults, unaffected by anything DDRace-mode
        // specific.
        assert_eq!(params.gravity, 50);
        assert_eq!(params.hook_length, 38000);
        assert_eq!(params.velramp_start, 55000);
        assert_eq!(params.player_collision, 100);
        // Review round 2, finding F9 (corrects an earlier, wrong guess in this same test that
        // blamed the map's own — zlib-compressed, unreadable by a plain `strings` scan — tuning
        // settings): gun/shotgun curvature 0 and a reduced speed for both are NOT specific to this
        // local server or this map. They are DDNet's own DDRace-gamemode defaults, set by
        // `CGameContext::ResetTuning()` (`gamecontext.cpp:4912-4918`, mirrored at `4134-4138`),
        // which every DDNet 20.1 server running DDRace (i.e. effectively all of them) calls on
        // every map load — overriding vanilla `tuning.h`'s `gun_curvature=1.25`/`gun_speed=2200`/
        // `shotgun_curvature=1.25`/`shotgun_speed=2750` with `gun_speed=1400`, `gun_curvature=0`,
        // `shotgun_speed=500`, `shotgun_speeddiff=0`, `shotgun_curvature=0` — exactly the ×100
        // fixed-point values asserted below. Consistent with this: it's identical across every map
        // in this fixture (`BlmapChill` and `Copy Love Box` alike) and across task 2.2a's own,
        // separately captured `local-capture-20260927.dat` — because the cause is DDNet's DDRace
        // game logic, not any one map or server config.
        assert_eq!(params.gun_curvature, 0);
        assert_eq!(params.gun_speed, 140000);
        assert_eq!(params.shotgun_curvature, 0);
        assert_eq!(params.shotgun_speed, 50000);
    }
}
