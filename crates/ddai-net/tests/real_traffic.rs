//! Real-traffic end-to-end test: task 2.2b acceptance criterion 5.
//!
//! Reuses task 2.2a's own loopback capture fixture (`tests/fixtures/local-capture-20260927.dat`,
//! see `tests/capture.rs` for how it was produced and its packet-layer checks) rather than making
//! a new one — the task spec explicitly allows either ("using the loopback capture tooling/
//! fixture from task 2.2a, or a new capture ... including a map change"); this fixture's steady-
//! state segment already covers the whole "`READY` through a clean `CLOSE`" span with real
//! snapshots, so a second capture would add capture/extraction risk without adding coverage this
//! task actually needs (task 2.2a's own `capture.rs` already proves the packet/chunk layer
//! decodes and the token/sequence bookkeeping is correct byte-for-byte against this exact
//! traffic — this test builds on top of that, one layer up: chunks -> messages -> snapshots).
//!
//! What this proves, precisely:
//! * every chunk in the steady-state segment (both directions) decodes to a [`Msg`] (not
//!   [`Msg::Invalid`]) — reported per message-type below;
//! * every assembled snapshot's CRC matches what the server claimed — **0 mismatches**, asserted;
//! * the bot's own tee (the `PlayerInfo` with `local == 1`) has a plausible, mostly-continuous
//!   trajectory across those snapshots (no large jump between consecutive snapshots' recorded
//!   position, except a small, explained number of them — respawn/teleport).

use ddai_net::assembly::{Event as AssembleEvent, SnapAssembler};
use ddai_net::delta::StaticSizes;
use ddai_net::generated::objects;
use ddai_net::huffman::Huffman;
use ddai_net::message::{self, Msg, Registry};
use ddai_net::packet::{self, packet_flags};
use ddai_net::sysmsg::SysMsg;
use ddai_net::view::View;
use std::collections::BTreeMap;

const FIXTURE: &[u8] = include_bytes!("fixtures/local-capture-20260927.dat");
const SEGMENT_HANDSHAKE: usize = 0;
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

// Identical fixture format/reader to `tests/capture.rs` (see that file's docs for the byte
// layout) — duplicated rather than shared, since integration test files are separate crates and
// this repo has no test-support crate (yet) to put it in.
fn read_fixture(bytes: &[u8]) -> Vec<Vec<Record>> {
    assert_eq!(&bytes[..6], b"DDCAP2");
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
    segments
}

fn is_connless(payload: &[u8]) -> bool {
    !payload.is_empty() && (payload[0] >> 2) & packet_flags::CONNLESS != 0
}

/// Extracts the negotiated security token from the handshake segment (records 1/2), exactly like
/// `tests/capture.rs::token_handling_matches_real_handshake` does.
fn negotiated_token(huffman: &Huffman, handshake: &[Record]) -> u32 {
    let connect_accept = packet::unpack_packet(&handshake[1].payload, huffman, true).unwrap();
    assert_eq!(&connect_accept.data[1..5], b"TKEN");
    u32::from_be_bytes([
        connect_accept.data[5],
        connect_accept.data[6],
        connect_accept.data[7],
        connect_accept.data[8],
    ])
}

/// The human-readable name for a non-UUID snapshot object type (task acceptance criterion 5's
/// "report counts ... per object type"); falls back to a numeric label for anything unnamed.
fn numbered_object_type_name(internal_type: i32) -> String {
    let name = match internal_type {
        v if v == objects::PlayerInput::ID => "PlayerInput",
        v if v == objects::Projectile::ID => "Projectile",
        v if v == objects::Laser::ID => "Laser",
        v if v == objects::Pickup::ID => "Pickup",
        v if v == objects::Flag::ID => "Flag",
        v if v == objects::GameInfo::ID => "GameInfo",
        v if v == objects::GameData::ID => "GameData",
        v if v == objects::CharacterCore::ID => "CharacterCore",
        v if v == objects::Character::ID => "Character",
        v if v == objects::PlayerInfo::ID => "PlayerInfo",
        v if v == objects::ClientInfo::ID => "ClientInfo",
        v if v == objects::SpectatorInfo::ID => "SpectatorInfo",
        v if v == objects::Common::ID => "Common",
        v if v == objects::Explosion::ID => "Explosion",
        v if v == objects::Spawn::ID => "Spawn",
        v if v == objects::HammerHit::ID => "HammerHit",
        v if v == objects::Death::ID => "Death",
        v if v == objects::SoundGlobal::ID => "SoundGlobal",
        v if v == objects::SoundWorld::ID => "SoundWorld",
        v if v == objects::DamageInd::ID => "DamageInd",
        other => return format!("numbered_type_{other}"),
    };
    name.to_string()
}

/// The leading identifier-like prefix of a `{:?}`-formatted enum value — a cheap, dependency-free
/// way to get "the variant name" for a per-type count without hand-writing a match arm per
/// message/`SysMsg` variant (there are over 100 across both).
fn variant_tag<T: std::fmt::Debug>(v: &T) -> String {
    let s = format!("{v:?}");
    s.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect()
}

#[test]
fn whole_session_decodes_with_zero_crc_mismatches_and_a_plausible_trajectory() {
    let huffman = Huffman::new();
    let registry = Registry::new();
    let segments = read_fixture(FIXTURE);
    let token = negotiated_token(&huffman, &segments[SEGMENT_HANDSHAKE]);

    let mut assembler = SnapAssembler::new(StaticSizes::ddnet_06());
    let mut msg_counts: BTreeMap<String, u32> = BTreeMap::new();
    let mut object_counts: BTreeMap<String, u32> = BTreeMap::new();
    let mut invalid_count = 0u32;
    let mut crc_mismatches = 0u32;
    let mut delta_errors = 0u32;
    let mut resyncs = 0u32;
    let mut snapshot_count = 0u32;
    let mut server_info_decoded = 0u32;
    // (character.tick, x, y) for the local player's character, one entry per snapshot it was
    // present in — see the F4 comment below for why `character.tick`, not the snapshot's own.
    let mut trajectory: Vec<(i32, i32, i32)> = Vec::new();
    let mut tune_params_seen: Vec<ddai_net::tuning::TuneParams> = Vec::new();

    for (seg_idx, segment) in segments.iter().enumerate() {
        for (i, record) in segment.iter().enumerate() {
            if is_connless(&record.payload) {
                // The two documented connectionless "iext" server-info responses (see
                // `tests/capture.rs`'s module docs) — task acceptance criterion 2 explicitly names
                // `SERVERINFO`; decode it for real here against the actual local server.
                let connless = packet::unpack_connless_packet(&record.payload).unwrap();
                if let Some(info) = ddai_net::serverinfo::decode(&connless.data) {
                    assert_eq!(
                        info.map, "Copy Love Box",
                        "segment {seg_idx} record {i}: unexpected map in real SERVERINFO"
                    );
                    server_info_decoded += 1;
                }
                continue;
            }
            if seg_idx == SEGMENT_HANDSHAKE && i == HANDSHAKE_INVALID_RECORD {
                continue;
            }
            let parsed = packet::unpack_packet(&record.payload, &huffman, true).unwrap();
            if parsed.flags & packet_flags::CONTROL != 0 {
                continue; // handshake control messages: not game/system-message traffic
            }
            if parsed.data.len() < 4 {
                continue; // too short to carry the trailing token; shouldn't happen post-handshake
            }
            let split = parsed.data.len() - 4;
            let tail = &parsed.data[split..];
            let got_token = u32::from_be_bytes([tail[0], tail[1], tail[2], tail[3]]);
            assert_eq!(
                got_token, token,
                "segment {seg_idx} record {i}: trailing token mismatch"
            );

            let chunk_packet = packet::Packet {
                flags: parsed.flags,
                ack: parsed.ack,
                num_chunks: parsed.num_chunks,
                data: parsed.data[..split].to_vec(),
            };
            for chunk in packet::ChunkIter::new(&chunk_packet) {
                let (msg, answer) = message::decode(chunk.data, &registry);
                assert!(
                    answer.is_none(),
                    "segment {seg_idx} record {i}: unexpected NETMSG_WHATIS in real traffic"
                );
                match &msg {
                    Msg::Invalid => {
                        invalid_count += 1;
                        continue;
                    }
                    Msg::Sys(sys_msg) => {
                        *msg_counts.entry(format!("sys:{}", variant_tag(sys_msg))).or_default() += 1;
                        if matches!(sys_msg, SysMsg::EnterGame) {
                            // F1, review round 1: a real client session must reset the assembler
                            // here — see `SnapAssembler::reset`'s docs. This fixture only ever
                            // enters once (no map change), so this is a no-op in practice; it is
                            // still the correct thing to do, and `tests/map_change_real_traffic.rs`
                            // is what actually exercises a map change needing it.
                            assembler.reset();
                        }
                        if matches!(
                            sys_msg,
                            SysMsg::Snap { .. }
                                | SysMsg::SnapEmpty { .. }
                                | SysMsg::SnapSingle { .. }
                                | SysMsg::SnapSmall { .. }
                        ) {
                            assert_eq!(
                                record.direction,
                                Direction::ServerToClient,
                                "segment {seg_idx} record {i}: a snapshot message from the client would be a protocol violation"
                            );
                        }
                        if let Some(event) = assembler.feed(sys_msg) {
                            match event {
                                AssembleEvent::Snapshot { tick: _, snap } => {
                                    snapshot_count += 1;
                                    let view = View::new(&snap);
                                    for item in &snap.items {
                                        let key = match view.describe_item(item) {
                                            ddai_net::view::ItemKind::Numbered(t) => numbered_object_type_name(t),
                                            ddai_net::view::ItemKind::Ex(name) => format!("ex:{name}"),
                                            ddai_net::view::ItemKind::UnresolvedEx => "unresolved_ex".to_string(),
                                            ddai_net::view::ItemKind::UuidTypeDescriptor => continue,
                                            ddai_net::view::ItemKind::Unknown => "unknown".to_string(),
                                        };
                                        *object_counts.entry(key).or_default() += 1;
                                    }
                                    if let Some(local) = view.players().into_iter().find(|p| p.info.local == 1)
                                        && let Some(character) = view.character(local.id)
                                    {
                                        // F4, review round 1: `character.tick` (the reckoning
                                        // core's own tick, `CharacterCore::m_Tick`) is the time
                                        // base the position `(x, y)` is actually *at* — DDNet
                                        // dead-reckons another player's `Character` forward from
                                        // that tick to the snapshot's own tick on *predicted*
                                        // physics, not by re-sending a fresh core every snapshot
                                        // (`docs/research/ddnet-protocol.md` §1.7 "Dead
                                        // reckoning"); using the snapshot's `tick` here instead
                                        // (an earlier version did) manufactures jumps that are
                                        // just the reckoning gap, not real movement.
                                        trajectory.push((
                                            character.character.tick,
                                            character.character.x,
                                            character.character.y,
                                        ));
                                    }
                                }
                                AssembleEvent::CrcMismatch { .. } => crc_mismatches += 1,
                                AssembleEvent::DeltaError { .. } => delta_errors += 1,
                                AssembleEvent::Resync { .. } => resyncs += 1,
                                AssembleEvent::Stale { .. } => {}
                            }
                        }
                    }
                    Msg::ExSys(ex_sys) => {
                        *msg_counts.entry(format!("ex_sys:{}", variant_tag(ex_sys))).or_default() += 1;
                    }
                    Msg::Game(game_msg) => {
                        *msg_counts.entry(format!("game:{}", variant_tag(game_msg))).or_default() += 1;
                    }
                    Msg::ExGame(ex_game_msg) => {
                        *msg_counts
                            .entry(format!("ex_game:{}", variant_tag(ex_game_msg)))
                            .or_default() += 1;
                    }
                    Msg::TuneParams(params) => {
                        *msg_counts.entry("game:SvTuneParams".to_string()).or_default() += 1;
                        tune_params_seen.push(*params);
                    }
                    Msg::TeamsState(_) => {
                        *msg_counts.entry("game:SvTeamsState".to_string()).or_default() += 1;
                    }
                }
            }
        }
    }

    println!("--- per-message-type counts ---");
    for (name, count) in &msg_counts {
        println!("  {name}: {count}");
    }
    println!("--- per-object-type counts ({snapshot_count} snapshots assembled) ---");
    for (name, count) in &object_counts {
        println!("  {name}: {count}");
    }
    println!("--- decode summary ---");
    println!("  invalid messages: {invalid_count}");
    println!("  crc mismatches: {crc_mismatches}");
    println!("  delta errors: {delta_errors}");
    println!("  resyncs: {resyncs}");
    println!("  trajectory samples: {}", trajectory.len());
    println!("  SERVERINFO responses decoded: {server_info_decoded}");
    println!("--- Sv_TuneParams decoded ({} seen) ---", tune_params_seen.len());
    for params in &tune_params_seen {
        println!(
            "  received={} gravity={} gun_curvature={} gun_speed={} hook_length={}",
            params.received, params.gravity, params.gun_curvature, params.gun_speed, params.hook_length
        );
    }

    assert_eq!(
        server_info_decoded, 2,
        "expected both documented real SERVERINFO responses to decode"
    );
    assert!(snapshot_count > 0, "expected at least one assembled snapshot");
    assert_eq!(invalid_count, 0, "every real chunk must decode to a known message");
    assert_eq!(delta_errors, 0, "every real delta must unpack cleanly");
    assert_eq!(crc_mismatches, 0, "acceptance criterion 5: count mismatches must be 0");

    for params in &tune_params_seen {
        assert_eq!(
            params.received,
            ddai_net::tuning::NUM_TUNE_PARAMS,
            "expected a full Sv_TuneParams payload"
        );
    }

    // Trajectory plausibility (F4, review round 1: by `character.tick`, not the snapshot's own —
    // see the push site above for why). Consecutive *distinct* `character.tick`s' recorded
    // position must not jump further than a generous per-tick speed bound would allow, except a
    // small, named number of "real" jumps (respawn back to a spawn point, or a teleporter tile) —
    // never *most* of them. Consecutive samples sharing the same `character.tick` (the reckoning
    // core simply had not changed between those snapshots) contribute no pair at all — an equal
    // tick with a different position would be a real decode bug, not "0 elapsed ticks of travel",
    // so this loop still catches that: `assert_eq!` on position when `t0 == t1`, distinct from
    // the jump accounting for `t0 != t1`.
    assert!(
        trajectory.len() > 10,
        "expected a substantial trajectory, got {}",
        trajectory.len()
    );
    const MAX_PLAUSIBLE_PX_PER_TICK: f64 = 60.0; // generous: DDNet's ground top speed is well under this
    let mut large_jumps = 0usize;
    let mut pairs = 0usize;
    for w in trajectory.windows(2) {
        let (t0, x0, y0) = w[0];
        let (t1, x1, y1) = w[1];
        if t1 == t0 {
            assert_eq!(
                (x0, y0),
                (x1, y1),
                "same character.tick {t0} reported two different positions"
            );
            continue;
        }
        pairs += 1;
        let tick_gap = (t1 - t0).max(1) as f64;
        let dist = (((x1 - x0) as f64).powi(2) + ((y1 - y0) as f64).powi(2)).sqrt();
        if dist > MAX_PLAUSIBLE_PX_PER_TICK * tick_gap {
            large_jumps += 1;
            println!("  large jump: character.tick {t0}->{t1} ({x0},{y0})->({x1},{y1}), dist={dist:.1}");
        }
    }
    println!("  trajectory pairs compared (distinct character.tick only): {pairs}");
    let jump_ratio = large_jumps as f64 / pairs.max(1) as f64;
    assert!(
        jump_ratio < 0.05,
        "too many implausible jumps in the local player's trajectory: {large_jumps}/{pairs} ({:.1}%)",
        jump_ratio * 100.0
    );
}
