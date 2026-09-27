//! Robustness / fuzz-style tests for task 2.2b's three new decode surfaces — acceptance criterion
//! 4: "fuzz-style no-panic tests (>= 10^6 mutated inputs) for message decode, delta unpack and
//! part assembly." Same style/rationale as `tests/robustness.rs` (task 2.2a): a failing `#[test]`
//! here means a panic, so "the run finished" already proves "no panics" for every case generated;
//! the `ProptestConfig::with_cases` counts below sum to over 1,000,000 across the three surfaces.

use ddai_net::delta::{self, StaticSizes};
use ddai_net::generated::objects;
use ddai_net::message::{self, Registry};
use ddai_net::packer::{Packer, Unpacker};
use ddai_net::snapshot::{Snapshot, SnapshotItem};
use ddai_net::sysmsg::{self, SysMsg};
use ddai_net::{assembly, uuid};
use proptest::prelude::*;
use std::sync::OnceLock;

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(Registry::new)
}

fn mutate(mut base: Vec<u8>, positions: &[(usize, u8)]) -> Vec<u8> {
    for &(pos, byte) in positions {
        if !base.is_empty() {
            let idx = pos % base.len();
            base[idx] = byte;
        }
    }
    base
}

fn mutation_strategy(max_len: usize) -> impl Strategy<Value = Vec<(usize, u8)>> {
    prop::collection::vec((0..max_len.max(1), any::<u8>()), 0..8)
}

// --- Message decode (`crate::message::decode`, `crate::sysmsg::decode`) ----------------------

/// A handful of real, well-formed message payloads (leading id/UUID + body), covering all four
/// decode tables — mutation seeds for the "mutated real payload" cases below.
fn sample_message_payloads() -> Vec<Vec<u8>> {
    let mut out = Vec::new();

    let mut buf = [0u8; 4096];
    let mut p = Packer::new(&mut buf);
    uuid::pack_msg_id(&mut p, uuid::MsgId::Numbered(sysmsg::id::MAP_CHANGE), true);
    sysmsg::encode(
        &SysMsg::MapChange {
            name: "Copy Love Box".to_string(),
            crc: 0x1234,
            size: 999,
        },
        &mut p,
    );
    out.push(p.data().to_vec());

    let mut buf2 = [0u8; 4096];
    let mut p2 = Packer::new(&mut buf2);
    uuid::pack_msg_id(
        &mut p2,
        uuid::MsgId::Numbered(ddai_net::generated::messages::id::NETMSGTYPE_SV_CHAT),
        false,
    );
    ddai_net::generated::messages::encode_sv_chat(
        &ddai_net::generated::messages::SvChat {
            team: 0,
            client_id: 3,
            message: "gg".to_string(),
        },
        &mut p2,
    );
    out.push(p2.data().to_vec());

    let mut buf3 = [0u8; 4096];
    let mut p3 = Packer::new(&mut buf3);
    let uuid3 = uuid::calculate_uuid("map-details@ddnet.tw");
    uuid::pack_msg_id(
        &mut p3,
        uuid::MsgId::Ex {
            uuid: uuid3,
            resolved: None,
        },
        true,
    );
    message::encode_ex(
        &message::ExSysMsg::MapDetails {
            name: "Copy Love Box".to_string(),
            sha256: [1u8; 32],
            crc: 1,
            size: 2,
            url: String::new(),
        },
        &mut p3,
    );
    out.push(p3.data().to_vec());

    let mut buf4 = [0u8; 4096];
    let mut p4 = Packer::new(&mut buf4);
    let uuid4 = uuid::calculate_uuid("showothers@netmsg.ddnet.tw");
    uuid::pack_msg_id(
        &mut p4,
        uuid::MsgId::Ex {
            uuid: uuid4,
            resolved: None,
        },
        false,
    );
    p4.add_int(1);
    out.push(p4.data().to_vec());

    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(250_000))]

    /// `crate::message::decode` never panics on arbitrary bytes.
    #[test]
    fn message_decode_never_panics_on_random_bytes(bytes in prop::collection::vec(any::<u8>(), 0..1400)) {
        let _ = message::decode(&bytes, registry());
    }

    /// `crate::sysmsg::decode` never panics on arbitrary bytes, for every known numbered id and a
    /// few unknown ones (exercising the `Unhandled` fallback too).
    #[test]
    fn sysmsg_decode_never_panics_on_random_bytes(
        bytes in prop::collection::vec(any::<u8>(), 0..600),
        id in 0i32..40,
    ) {
        let mut unpacker = Unpacker::new(&bytes);
        let _ = sysmsg::decode(id, &mut unpacker);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(250_000))]

    /// Mutating a real, well-formed message payload — more likely to pass the leading id/UUID
    /// decode and actually exercise a specific message's field-level decode logic.
    #[test]
    fn message_decode_never_panics_on_mutated_real_payloads(
        which in 0usize..4,
        mutations in mutation_strategy(200),
    ) {
        let samples = sample_message_payloads();
        let base = samples[which % samples.len()].clone();
        let mutated = mutate(base, &mutations);
        let _ = message::decode(&mutated, registry());
    }
}

// --- Delta unpack (`crate::delta::unpack_delta`) ----------------------------------------------

fn sample_base_snapshot() -> Snapshot {
    Snapshot {
        items: vec![
            SnapshotItem {
                key: objects::Flag::ID << 16,
                data: vec![100, 200, 0],
            },
            SnapshotItem {
                key: (objects::Character::ID << 16) | 3,
                data: vec![0; 22],
            },
        ],
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(250_000))]

    /// `unpack_delta` never panics on a completely arbitrary `i32` stream, against either an
    /// empty or a non-trivial base snapshot, with or without a static-size table — covers both
    /// branches deciding whether an item's size comes from the table or the wire.
    #[test]
    fn delta_unpack_never_panics_on_random_ints(
        ints in prop::collection::vec(any::<i32>(), 0..80),
        use_static_sizes in any::<bool>(),
        use_nonempty_base in any::<bool>(),
    ) {
        let sizes = if use_static_sizes { StaticSizes::ddnet_06() } else { StaticSizes::none() };
        let base = if use_nonempty_base { sample_base_snapshot() } else { Snapshot::empty() };
        let _ = delta::unpack_delta(&base, &ints, &sizes);
    }

    /// Same, but starting from a real, well-formed delta (base -> a random target) and mutating
    /// individual `i32`s — more likely to pass the header/type/id checks and actually exercise
    /// the per-item size/diff logic deeper in `unpack_delta`.
    #[test]
    fn delta_unpack_never_panics_on_mutated_real_delta(
        extra_x in any::<i32>(),
        mutation_indices in prop::collection::vec(any::<usize>(), 0..6),
        mutation_values in prop::collection::vec(any::<i32>(), 0..6),
    ) {
        let sizes = StaticSizes::ddnet_06();
        let base = sample_base_snapshot();
        let mut target = base.clone();
        target.items.push(SnapshotItem {
            key: (objects::Flag::ID << 16) | 1,
            data: vec![extra_x, 0, 0],
        });
        let Some(mut ints) = delta::create_delta(&base, &target, &sizes) else {
            return Ok(());
        };
        if !ints.is_empty() {
            for (&idx, &val) in mutation_indices.iter().zip(&mutation_values) {
                let i = idx % ints.len();
                ints[i] = val;
            }
        }
        let _ = delta::unpack_delta(&base, &ints, &sizes);
    }
}

// --- Part assembly (`crate::assembly::SnapAssembler::feed`) -----------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(250_000))]

    /// `SnapAssembler::feed` never panics on structurally arbitrary `Snap` fields — ticks,
    /// part/num_parts combinations (including out-of-range ones), and random data bytes.
    #[test]
    fn assembler_feed_never_panics_on_random_snap_fields(
        tick in any::<i32>(),
        delta_tick_diff in any::<i32>(),
        num_parts in any::<i32>(),
        part in any::<i32>(),
        crc in any::<i32>(),
        data in prop::collection::vec(any::<u8>(), 0..1200),
    ) {
        let mut asm = assembly::SnapAssembler::new(StaticSizes::ddnet_06());
        let msg = SysMsg::Snap {
            tick,
            delta_tick: tick.wrapping_sub(delta_tick_diff),
            num_parts,
            part,
            crc,
            data,
        };
        let _ = asm.feed(&msg);
    }

    /// Same, feeding a whole *sequence* of random `Snap`/`SnapEmpty`/`SnapSingle` messages into
    /// one assembler in a row — exercises cross-call state (in-progress parts bitmap, storage,
    /// ack/resync bookkeeping) under adversarial-looking input, not just single isolated calls.
    #[test]
    fn assembler_feed_never_panics_on_random_message_sequences(
        msgs in prop::collection::vec(arb_snap_like_message(), 0..12),
    ) {
        let mut asm = assembly::SnapAssembler::new(StaticSizes::ddnet_06());
        for msg in &msgs {
            let _ = asm.feed(msg);
        }
    }
}

fn arb_snap_like_message() -> impl Strategy<Value = SysMsg> {
    prop_oneof![
        (
            any::<i32>(),
            any::<i32>(),
            any::<i32>(),
            any::<i32>(),
            any::<i32>(),
            prop::collection::vec(any::<u8>(), 0..950),
        )
            .prop_map(|(tick, delta_tick, num_parts, part, crc, data)| SysMsg::Snap {
                tick,
                delta_tick,
                num_parts,
                part,
                crc,
                data,
            }),
        (any::<i32>(), any::<i32>()).prop_map(|(tick, delta_tick)| SysMsg::SnapEmpty { tick, delta_tick }),
        (
            any::<i32>(),
            any::<i32>(),
            any::<i32>(),
            prop::collection::vec(any::<u8>(), 0..950)
        )
            .prop_map(|(tick, delta_tick, crc, data)| SysMsg::SnapSingle {
                tick,
                delta_tick,
                crc,
                data,
            }),
    ]
}

// --- End-to-end: raw bytes straight into the full sys-message decoder, then the assembler -----

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    /// The full pipeline for the SNAP family specifically: arbitrary bytes decoded as a
    /// `NETMSG_SNAP`/`SNAPSINGLE`/`SNAPEMPTY` payload (`crate::sysmsg::decode`), whatever comes out
    /// fed straight into the assembler — end-to-end, no panics anywhere in between.
    #[test]
    fn sysmsg_decode_then_assemble_never_panics(
        bytes in prop::collection::vec(any::<u8>(), 0..1100),
        msg_id in prop_oneof![Just(sysmsg::id::SNAP), Just(sysmsg::id::SNAPSINGLE), Just(sysmsg::id::SNAPEMPTY), Just(sysmsg::id::SNAPSMALL)],
    ) {
        let mut asm = assembly::SnapAssembler::new(StaticSizes::ddnet_06());
        let mut unpacker = Unpacker::new(&bytes);
        let msg = sysmsg::decode(msg_id, &mut unpacker);
        let _ = asm.feed(&msg);
    }
}
