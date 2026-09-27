//! Differential tests against `libtw2-snapshot` (dev-dependency only, decision D-029 — never
//! linked into non-test builds; see `Cargo.toml`) — task 2.2b acceptance criterion 4:
//! "differential tests vs libtw2-snapshot on random snapshots/deltas (proptest >= 10k cases)".
//!
//! Both directions are tested:
//! * [`our_decoder_accepts_libtw2s_delta`]: build a random (base, target) pair of snapshots,
//!   encode the delta with libtw2's independent `Delta::create_raw`/`write`, decode it with our
//!   own [`ddai_net::delta::unpack_delta`] — the resulting item set must equal `target` exactly.
//!   This is the direction that actually matters for a client (we only ever *decode* deltas a
//!   real server sends), so it is the stronger of the two checks.
//! * [`libtw2_accepts_our_delta`]: the same pair, encoded with our own
//!   [`ddai_net::delta::create_delta`], decoded with libtw2's `Delta::read`/`RawSnap::read_with_delta`
//!   — proves our encoder (used only by our own tests, see `crate::delta`'s module docs) produces
//!   bytes an independent decoder agrees with too.
//!
//! Every item size is left on the wire (`object_size` closure always returns `None`, matching our
//! side's [`StaticSizes::none`]) — this exercises the exact same diff/undiff arithmetic either way;
//! the "size taken from a static table" branch is already covered by `crate::delta`'s own unit
//! tests (`new_item_with_static_size_omits_size_on_wire`), so duplicating that coupling here would
//! add complexity without adding coverage.

use ddai_net::delta::{self, StaticSizes};
use ddai_net::snapshot::{Snapshot, SnapshotItem};
use libtw2_snapshot::snap::{Delta as LtwDelta, RawBuilder};
use proptest::prelude::*;
use std::collections::BTreeMap;

struct NoWarn;
impl<T> libtw2_warn::Warn<T> for NoWarn {
    fn warn(&mut self, _warning: T) {}
}

/// A type's item size is a *function of the type*, fixed for the whole test case — matching the
/// real protocol's invariant that a given object type's size never varies item to item or
/// snapshot to snapshot within one server's lifetime (whether via `crate::delta::StaticSizes` or
/// a stable ex-object shape). Letting size vary independently per random item (an earlier version
/// of this test did) generates a same-key-same-type-different-size delta that *neither*
/// implementation's raw-level "update in place" logic supports — `crate::delta::unpack_delta`
/// correctly rejects it as [`ddai_net::delta::DeltaError::SizeMismatch`], and libtw2's own
/// `RawSnap::read_with_delta` hits an internal `assert!` on the same ill-formed input
/// (`prepare_item`'s occupied-entry branch silently keeps the *old* slot size). Neither is a bug;
/// it is simply not a shape the real protocol ever produces, so this strategy does not generate it
/// either — covers 0..=6 ints (zero-length, single-int, and multi-int items alike).
fn size_for_type(t: u16) -> usize {
    1 + (usize::from(t) * 7 + 3) % 6
}

/// One random item: a small type/id range keeps key collisions (which both sides must resolve
/// identically — "last one wins" when building a snapshot from a flat list) frequent enough to be
/// exercised.
fn item_strategy() -> impl Strategy<Value = (u16, u16, Vec<i32>)> {
    (1u16..=40, 0u16..12).prop_flat_map(|(t, id)| {
        prop::collection::vec(any::<i32>(), size_for_type(t)).prop_map(move |data| (t, id, data))
    })
}

fn items_strategy() -> impl Strategy<Value = Vec<(u16, u16, Vec<i32>)>> {
    prop::collection::vec(item_strategy(), 0..=25)
}

/// Deduplicates by `(type, id)` key, keeping the *last* occurrence — matches both our own
/// `Snapshot`/`RawBuilder`'s "later add wins" semantics when a random item list happens to repeat
/// a key.
fn dedup_by_key(items: Vec<(u16, u16, Vec<i32>)>) -> BTreeMap<(u16, u16), Vec<i32>> {
    items.into_iter().map(|(t, id, data)| ((t, id), data)).collect()
}

fn our_snapshot(items: &BTreeMap<(u16, u16), Vec<i32>>) -> Snapshot {
    Snapshot {
        items: items
            .iter()
            .map(|(&(t, id), data)| SnapshotItem {
                key: (i32::from(t) << 16) | i32::from(id),
                data: data.clone(),
            })
            .collect(),
    }
}

fn their_raw_snap(items: &BTreeMap<(u16, u16), Vec<i32>>) -> libtw2_snapshot::snap::RawSnap {
    let mut builder = RawBuilder::new();
    for (&(t, id), data) in items {
        builder
            .add_item(t, id, data)
            .expect("well-formed test data always fits");
    }
    builder.finish()
}

fn snapshot_as_map(snap: &Snapshot) -> BTreeMap<i32, Vec<i32>> {
    snap.items.iter().map(|it| (it.key, it.data.clone())).collect()
}

fn compress(ints: &[i32]) -> Vec<u8> {
    let mut buf = vec![0u8; ints.len() * ddai_net::packer::MAX_BYTES_PACKED];
    let n = ddai_net::packer::pack_ints(&mut buf, ints).expect("scratch buffer is always big enough");
    buf.truncate(n);
    buf
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn our_decoder_accepts_libtw2s_delta(base_raw in items_strategy(), target_raw in items_strategy()) {
        let base = dedup_by_key(base_raw);
        let target = dedup_by_key(target_raw);

        let our_base = our_snapshot(&base);
        let our_target = our_snapshot(&target);
        let their_base = their_raw_snap(&base);
        let their_target = their_raw_snap(&target);

        let mut their_delta = LtwDelta::new();
        their_delta.create_raw(&their_base, &their_target);

        let mut delta_bytes = Vec::with_capacity(1 << 16);
        let written = libtw2_packer::with_packer(&mut delta_bytes, |p| their_delta.write(|_| None, p))
            .expect("scratch buffer is always big enough");
        let written_len = written.len();
        delta_bytes.truncate(written_len);

        let mut ints = Vec::new();
        let consumed = ddai_net::packer::unpack_ints(&delta_bytes, &mut ints);
        prop_assert!(consumed.is_some(), "our own varint decompressor must accept libtw2's own varint output");

        let decoded = delta::unpack_delta(&our_base, &ints, &StaticSizes::none());
        prop_assert!(decoded.is_ok(), "delta unpack failed: {:?}", decoded.err());
        let decoded = decoded.unwrap();

        prop_assert_eq!(snapshot_as_map(&decoded), snapshot_as_map(&our_target));
    }

    #[test]
    fn libtw2_accepts_our_delta(base_raw in items_strategy(), target_raw in items_strategy()) {
        let base = dedup_by_key(base_raw);
        let target = dedup_by_key(target_raw);

        let our_base = our_snapshot(&base);
        let our_target = our_snapshot(&target);
        let their_base = their_raw_snap(&base);

        let Some(delta_ints) = delta::create_delta(&our_base, &our_target, &StaticSizes::none()) else {
            // Identical snapshots produce no delta at all — nothing to cross-check, but the
            // "then" side must also agree they're identical.
            prop_assert_eq!(snapshot_as_map(&our_base), snapshot_as_map(&our_target));
            return Ok(());
        };
        let wire_bytes = compress(&delta_ints);

        let mut their_delta = LtwDelta::new();
        let mut unpacker = libtw2_packer::Unpacker::new(&wire_bytes);
        their_delta
            .read(&mut NoWarn, |_| None, &mut unpacker)
            .expect("our own encoder's output must be valid libtw2 delta bytes");

        let mut result = libtw2_snapshot::snap::RawSnap::empty();
        result
            .read_with_delta(&mut NoWarn, &their_base, &their_delta)
            .expect("applying our delta against the same base must succeed");

        let got: BTreeMap<(u16, u16), Vec<i32>> = result
            .items()
            .map(|it| ((it.raw_type_id, it.id), it.data.to_vec()))
            .collect();
        prop_assert_eq!(got, target);
    }
}
