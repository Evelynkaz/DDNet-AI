// Ported from DDNet `src/engine/shared/snapshot.{h,cpp}` (the `CSnapshot`/`CSnapshotStorage`
// halves only — delta (un)packing is `crate::delta`; pinned rev
// c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"), which carries the original Teeworlds
// zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same item/key/CRC semantics, so it
// stays byte-for-byte compatible with the wire format every DDNet 0.6+DDNet client/server speaks.
// See docs/formats.md for the byte layout.
//
//! A decoded snapshot: a flat, ordered set of items keyed by `(internal_type << 16) | id`.
//!
//! Unlike DDNet's `CSnapshot` (a single flat byte buffer with a separate offset table, chosen for
//! C++'s manual memory management), this is a plain `Vec` of owned items — semantically identical
//! (same keys, same per-item `i32` data, same iteration order), just a more natural Rust shape.
//! Item order matters for one thing only: [`Snapshot::crc`] must sum every item's data in some
//! fixed order for determinism, and this crate always builds/decodes items in the same order
//! DDNet's own `CSnapshotBuilder` would (see `crate::delta`), so two snapshots built from the same
//! inputs always compare `==` and CRC identically.
//!
//! Unknown object types (an internal type this crate's [`crate::generated::objects`] does not
//! know, whether numbered or an unrecognised UUID) are kept here as plain, untyped items — task
//! acceptance criterion 3's "tolerant decoding": [`crate::view`] is what turns *known* types into
//! typed values; this module never drops or refuses to store an item because its type is unknown.

use crate::uuid::Uuid;

/// `CSnapshot::OFFSET_UUID_TYPE` (`snapshot.h:45`): internal types `>=` this are per-snapshot
/// UUID-resolved (ex) types, resolved via that snapshot's own `type == 0` (`NETOBJTYPE_EX`)
/// descriptor items — see [`Snapshot::ex_type_uuid`].
pub const OFFSET_UUID_TYPE: i32 = 0x4000;
/// `CSnapshot::MAX_TYPE` (`snapshot.h:46`).
pub const MAX_TYPE: i32 = 0x7fff;
/// `CSnapshot::MAX_ID` (`snapshot.h:47`).
pub const MAX_ID: i32 = 0xffff;
/// `CSnapshot::MAX_ITEMS` (`snapshot.h:48`).
pub const MAX_ITEMS: usize = 1024;
/// `CSnapshot::MAX_PARTS` (`snapshot.h:49`) — multi-part `NETMSG_SNAP` assembly, see
/// `crate::message`.
pub const MAX_PARTS: usize = 64;
/// `CSnapshot::MAX_SIZE` (`snapshot.h:50`): `MAX_PARTS * 1024`.
pub const MAX_SIZE: usize = MAX_PARTS * 1024;

/// One item: `key = (internal_type << 16) | id` (`CSnapshotItem::Key`), `data` is its raw `i32`
/// payload — for a known non-ex type this is exactly [`crate::generated::objects`]'s struct
/// fields in order; for a known ex type, resolve `internal_type` via [`Snapshot::ex_type_uuid`]
/// first (see `crate::view`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotItem {
    pub key: i32,
    pub data: Vec<i32>,
}

impl SnapshotItem {
    /// `CSnapshotItem::InternalType` (`snapshot.h:22`).
    pub fn internal_type(&self) -> i32 {
        self.key >> 16
    }

    /// `CSnapshotItem::Id` (`snapshot.h:23`).
    pub fn id(&self) -> i32 {
        self.key & 0xffff
    }
}

/// A fully decoded snapshot (the result of [`crate::delta::unpack_delta`], or of
/// [`Snapshot::empty`] as the very first delta's base — `delta_tick == -1`,
/// `network.py`/`client.cpp`'s "empty snapshot" case).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Snapshot {
    pub items: Vec<SnapshotItem>,
}

impl Snapshot {
    /// `CSnapshot::EmptySnapshot()` (`snapshot.cpp:28`) — the base for `delta_tick == -1`.
    pub fn empty() -> Self {
        Snapshot { items: Vec::new() }
    }

    /// `CSnapshot::GetItemIndex` (`snapshot.cpp:63-72`), returning the item itself rather than an
    /// index (nothing else in this crate needs the index once decoded into a `Vec`).
    pub fn item_by_key(&self, key: i32) -> Option<&SnapshotItem> {
        self.items.iter().find(|it| it.key == key)
    }

    /// Looks up an item by `(internal_type, id)` — the two halves of [`SnapshotItem::key`].
    pub fn find(&self, internal_type: i32, id: i32) -> Option<&SnapshotItem> {
        self.item_by_key((internal_type << 16) | (id & 0xffff))
    }

    /// `CSnapshot::Crc` (`snapshot.cpp:112-125`): wrapping `u32` sum of every item's data ints,
    /// across every item, key excluded — matches the C++ reference's plain `int` overflow
    /// (`unsigned` accumulator, so this is well-defined wraparound, never UB, in both languages).
    pub fn crc(&self) -> u32 {
        let mut crc: u32 = 0;
        for item in &self.items {
            for &v in &item.data {
                crc = crc.wrapping_add(v as u32);
            }
        }
        crc
    }

    /// `CSnapshot::GetExternalItemType` (`snapshot.cpp:43-61`): resolves an ex `internal_type`
    /// (`>= `[`OFFSET_UUID_TYPE`]) to its [`Uuid`] via this snapshot's own `internal_type == 0`
    /// (`NETOBJTYPE_EX`) descriptor item at `id = internal_type` — the 16-byte UUID is stored as 4
    /// big-endian `i32`s (`bytes_be_to_uint`/`uint_to_bytes_be`, `snapshot.cpp:56-59`). Returns
    /// `None` if `internal_type < OFFSET_UUID_TYPE` (not an ex type at all — matches the C++
    /// reference's short-circuit for that case, which just returns `InternalType` unchanged; the
    /// caller (`crate::view`) only calls this once it already knows it needs a UUID), the
    /// descriptor item is missing, or it is not exactly 16 bytes (4 ints).
    pub fn ex_type_uuid(&self, internal_type: i32) -> Option<Uuid> {
        if internal_type < OFFSET_UUID_TYPE {
            return None;
        }
        let descriptor = self.find(0, internal_type)?;
        // `GetExternalItemType` only rejects a *short* descriptor (`GetItemSize(...) <
        // sizeof(CUuid)`, `snapshot.cpp:51`) — one with *more* than 4 ints is accepted, only the
        // first 16 bytes are ever read (F6, review round 1: an earlier version rejected anything
        // other than exactly 4 ints, stricter than DDNet).
        if descriptor.data.len() < 4 {
            return None;
        }
        let mut bytes = [0u8; 16];
        for (i, &v) in descriptor.data[..4].iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&(v as u32).to_be_bytes());
        }
        Some(Uuid(bytes))
    }
}

/// `CSnapshotStorage` (`snapshot.cpp:635-746`): keeps recent snapshots by tick so a later delta
/// can look its base back up, purging everything older than an acked tick.
///
/// Unlike the C++ reference (an intrusive doubly-linked list of `malloc`'d nodes), this is a plain
/// `VecDeque` — same semantics (insertion must be in strictly increasing tick order, `Get` and
/// `purge_until` behave identically), simpler ownership.
#[derive(Debug, Clone, Default)]
pub struct SnapshotStorage {
    /// Kept in strictly increasing `tick` order (enforced by [`SnapshotStorage::add`]).
    entries: std::collections::VecDeque<(i32, Snapshot)>,
}

impl SnapshotStorage {
    pub fn new() -> Self {
        Self::default()
    }

    /// `CSnapshotStorage::Add` (`snapshot.cpp:681-719`). Panics only on a caller/programmer error
    /// (ticks must strictly increase — the C++ reference `dbg_assert`s the same thing), never on
    /// anything peer-controlled: a peer cannot make us call `add` at all, let alone out of order —
    /// that policy lives in whatever drives this (task 2.3's client session).
    pub fn add(&mut self, tick: i32, snap: Snapshot) {
        if let Some((last_tick, _)) = self.entries.back() {
            assert!(
                *last_tick < tick,
                "snapshots inserted into SnapshotStorage with non-increasing tick {last_tick} >= {tick}"
            );
        }
        self.entries.push_back((tick, snap));
    }

    /// `CSnapshotStorage::Get` (`snapshot.cpp:721-746`): the list is sorted by tick and the
    /// queried tick is usually one of the most recently added, so search backwards from the
    /// newest — matches the C++ reference's own stated rationale.
    pub fn get(&self, tick: i32) -> Option<&Snapshot> {
        for (t, snap) in self.entries.iter().rev() {
            if *t == tick {
                return Some(snap);
            }
            if *t < tick {
                return None; // all remaining (older) entries are even older still
            }
        }
        None
    }

    /// `CSnapshotStorage::PurgeUntil` (`snapshot.cpp:654-679`): drops every entry with
    /// `tick < until`.
    pub fn purge_until(&mut self, until: i32) {
        while let Some((t, _)) = self.entries.front() {
            if *t >= until {
                break;
            }
            self.entries.pop_front();
        }
    }

    /// `CSnapshotStorage::PurgeAll` (`snapshot.cpp:641-652`).
    pub fn purge_all(&mut self) {
        self.entries.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generated::objects;

    fn flag_item(x: i32, y: i32, team: i32) -> Snapshot {
        Snapshot {
            items: vec![SnapshotItem {
                key: objects::Flag::ID << 16,
                data: vec![x, y, team],
            }],
        }
    }

    #[test]
    fn crc_one_int() {
        // snapshot_test.cpp: Snapshot.CrcOneInt
        assert_eq!(flag_item(4, 0, 0).crc(), 4);
    }

    #[test]
    fn crc_two_ints() {
        // snapshot_test.cpp: Snapshot.CrcTwoInts
        assert_eq!(flag_item(1, 1, 0).crc(), 2);
    }

    #[test]
    fn crc_bigger_ints() {
        // snapshot_test.cpp: Snapshot.CrcBiggerInts
        assert_eq!(flag_item(99999999, 1, 1).crc(), 100000001);
    }

    #[test]
    fn crc_overflow() {
        // snapshot_test.cpp: Snapshot.CrcOverflow
        assert_eq!(flag_item(-1, 1, 1).crc(), 1); // 0xFFFFFFFF + 1 + 1 wraps to 1
    }

    #[test]
    fn storage_get() {
        // snapshot_test.cpp: Snapshot.StorageGet
        let mut storage = SnapshotStorage::new();
        storage.add(10, Snapshot::empty());
        storage.add(20, Snapshot::empty());
        storage.add(30, Snapshot::empty());
        storage.add(40, Snapshot::empty());

        assert!(storage.get(40).is_some());
        assert!(storage.get(10).is_some());
        assert!(storage.get(30).is_some());

        assert!(storage.get(50).is_none());
        assert!(storage.get(5).is_none());
        assert!(storage.get(25).is_none());
    }

    #[test]
    fn storage_purge_until_drops_only_older_entries() {
        let mut storage = SnapshotStorage::new();
        for tick in [10, 20, 30, 40] {
            storage.add(tick, Snapshot::empty());
        }
        storage.purge_until(25);
        assert!(storage.get(10).is_none());
        assert!(storage.get(20).is_none());
        assert!(storage.get(30).is_some());
        assert!(storage.get(40).is_some());
        assert_eq!(storage.len(), 2);
    }

    #[test]
    #[should_panic(expected = "non-increasing tick")]
    fn storage_add_out_of_order_panics() {
        let mut storage = SnapshotStorage::new();
        storage.add(10, Snapshot::empty());
        storage.add(10, Snapshot::empty()); // not strictly increasing
    }

    #[test]
    fn ex_type_uuid_resolves_via_descriptor_item() {
        let uuid = crate::uuid::calculate_uuid("character@netobj.ddnet.tw");
        let mut ints = [0i32; 4];
        for (i, chunk) in uuid.0.chunks(4).enumerate() {
            ints[i] = u32::from_be_bytes(chunk.try_into().unwrap()) as i32;
        }
        let snap = Snapshot {
            items: vec![SnapshotItem {
                key: 0x7fff,
                data: ints.to_vec(),
            }],
        };
        assert_eq!(snap.ex_type_uuid(0x7fff), Some(uuid));
        assert_eq!(snap.ex_type_uuid(0x7ffe), None); // no descriptor for that internal type
        assert_eq!(snap.ex_type_uuid(5), None); // not an ex type at all (< OFFSET_UUID_TYPE)
    }

    #[test]
    fn ex_type_uuid_missing_or_malformed_descriptor_returns_none_not_panic() {
        let snap = Snapshot::empty();
        assert_eq!(snap.ex_type_uuid(0x7fff), None);

        let malformed = Snapshot {
            items: vec![SnapshotItem {
                key: 0x7fff, // internal_type=0, id=0x7fff, but only 2 ints, not 4
                data: vec![1, 2],
            }],
        };
        assert_eq!(malformed.ex_type_uuid(0x7fff), None);
    }

    #[test]
    fn ex_type_uuid_accepts_a_descriptor_longer_than_4_ints() {
        // F6, review round 1: DDNet's own `GetExternalItemType` only rejects a *short*
        // descriptor (`< sizeof(CUuid)`); a longer one is accepted, extra ints ignored.
        let uuid = crate::uuid::calculate_uuid("character@netobj.ddnet.tw");
        let mut data = vec![0i32; 5];
        for (i, chunk) in uuid.0.chunks(4).enumerate() {
            data[i] = u32::from_be_bytes(chunk.try_into().unwrap()) as i32;
        }
        data[4] = 0xdead_beefu32 as i32; // trailing garbage, must be ignored
        let snap = Snapshot {
            items: vec![SnapshotItem { key: 0x7fff, data }],
        };
        assert_eq!(snap.ex_type_uuid(0x7fff), Some(uuid));
    }
}
