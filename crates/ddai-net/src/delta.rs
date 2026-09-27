// Ported from DDNet `src/engine/shared/snapshot.cpp` (`CSnapshotDelta`, pinned rev
// c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"), which carries the original Teeworlds
// zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same delta byte/int layout and
// undiff/diff arithmetic, so it stays byte-for-byte compatible with the wire format every DDNet
// 0.6+DDNet client/server speaks. See docs/formats.md for the byte layout.
//
//! Snapshot delta (de)coding — task 2.2b acceptance criterion 3.
//!
//! The wire's `NETMSG_SNAP*` payload is a *varint-compressed* stream of `i32`s
//! (`CVariableInt::Compress`/`Decompress`, already implemented in `crate::packer` as
//! [`crate::packer::pack_ints`]/[`crate::packer::unpack_ints`]); once decompressed into a plain
//! `&[i32]`, [`unpack_delta`] below is a direct, one-to-one port of `CSnapshotDelta::UnpackDelta`
//! operating on that `i32` stream — no separate byte-level parsing needed, since DDNet's own
//! `pSrcData`/`DataSize` at this layer are *already* the decompressed int buffer reinterpreted as
//! bytes (`client.cpp:2196-2209`). The delta format itself (`snapshot.cpp:517-630`):
//!
//! ```text
//! [num_deleted_items] [num_updated_items] [num_temp_items (always 0, ignored)]
//! num_deleted_items × [deleted_key]
//! num_updated_items × [type] [id] [size — only if `type` has no static size] [size × data_int]
//! ```
//!
//! `data_int` is either a brand-new item's literal value, or (if an item with the same key
//! already exists in `base`, with the same size) a per-int wrapping *diff* to add onto the base
//! item's corresponding int (`UndiffItem`) — this module has no way to tell those two cases apart
//! by looking at the bytes alone; it always tries "diff against base" first and falls back to
//! "brand new, take literally" exactly when `base` has no same-key-same-size item, matching the
//! C++ reference exactly.
//!
//! [`unpack_delta`] never panics on malformed/hostile `ints` — every read is bounds-checked and
//! every error path returns [`DeltaError`] instead (task acceptance criterion 3: "error on
//! malformed data — never panic"; criterion 4's fuzz coverage is `tests/robustness.rs`).

use crate::snapshot::{MAX_ID, MAX_ITEMS, MAX_TYPE, Snapshot, SnapshotItem};

/// `CSnapshotDelta::MAX_NETOBJSIZES` (`snapshot.h:108`): the range of *numbered* (non-ex) types
/// whose size can be looked up statically instead of read from the wire — see [`StaticSizes`].
pub const MAX_STATIC_TYPES: usize = 64;

/// A conservative cap on a single item's `i32` count, purely defensive (DDNet's own equivalent
/// bound is the whole snapshot's 64 KiB `MAX_SIZE`, enforced through `CSnapshotBuilder`'s
/// allocation checks — this crate has no fixed-size buffer to overflow, so nothing strictly
/// requires this, but an attacker-controlled "read this many ints" value should never be allowed
/// to demand an unbounded allocation regardless).
const MAX_ITEM_INTS: usize = 16 * 1024;

/// `CSnapshotDelta::m_aItemSizes` (`snapshot.h:110`): static per-(numbered-)type item sizes, in
/// `i32`s, set once at startup exactly like `SnapSetStaticsize` (`gameclient.cpp:358-360`: every
/// numbered type `0..NUM_NETOBJTYPES` gets `GetObjSize(i)`, which is `0` for type `0`
/// (`NETOBJTYPE_EX`) — a `0` here means "no static size, size is on the wire", matching the C++
/// reference's own falsy check `m_aItemSizes[Type]`).
#[derive(Debug, Clone, Copy)]
pub struct StaticSizes([u16; MAX_STATIC_TYPES]);

impl StaticSizes {
    /// No static sizes at all — every item's size comes from the wire. Valid (if slower/more
    /// wire-verbose) for any protocol version, since ex types never have a static size anyway.
    pub fn none() -> Self {
        StaticSizes([0; MAX_STATIC_TYPES])
    }

    /// The 20 numbered DDNet 0.6+DDNet object/event types' static sizes
    /// ([`crate::generated::objects`]), exactly like `gameclient.cpp:358-360`'s startup loop.
    pub fn ddnet_06() -> Self {
        use crate::generated::objects::*;
        let mut s = Self::none();
        s.set(PlayerInput::ID, PlayerInput::SIZE_INTS);
        s.set(Projectile::ID, Projectile::SIZE_INTS);
        s.set(Laser::ID, Laser::SIZE_INTS);
        s.set(Pickup::ID, Pickup::SIZE_INTS);
        s.set(Flag::ID, Flag::SIZE_INTS);
        s.set(GameInfo::ID, GameInfo::SIZE_INTS);
        s.set(GameData::ID, GameData::SIZE_INTS);
        s.set(CharacterCore::ID, CharacterCore::SIZE_INTS);
        s.set(Character::ID, Character::SIZE_INTS);
        s.set(PlayerInfo::ID, PlayerInfo::SIZE_INTS);
        s.set(ClientInfo::ID, ClientInfo::SIZE_INTS);
        s.set(SpectatorInfo::ID, SpectatorInfo::SIZE_INTS);
        s.set(Common::ID, Common::SIZE_INTS);
        s.set(Explosion::ID, Explosion::SIZE_INTS);
        s.set(Spawn::ID, Spawn::SIZE_INTS);
        s.set(HammerHit::ID, HammerHit::SIZE_INTS);
        s.set(Death::ID, Death::SIZE_INTS);
        s.set(SoundGlobal::ID, SoundGlobal::SIZE_INTS);
        s.set(SoundWorld::ID, SoundWorld::SIZE_INTS);
        s.set(DamageInd::ID, DamageInd::SIZE_INTS);
        s
    }

    /// Sets `ty`'s static size. `ty` must be `0..MAX_STATIC_TYPES` and `size` must fit a `u16` —
    /// both always true for our own fixed, compile-time-known type table, so this panics rather
    /// than erroring on violation (a programmer error, never attacker-controlled: this is never
    /// called with wire data).
    pub fn set(&mut self, ty: i32, size: usize) {
        let idx = usize::try_from(ty).expect("static type id must be non-negative");
        assert!(idx < MAX_STATIC_TYPES, "static type id {ty} out of range");
        self.0[idx] = u16::try_from(size).expect("static size must fit u16");
    }

    /// The static size for `ty`, or `None` if `ty` is out of the static range or has no static
    /// size set (matches the C++ reference's `Type < MAX_NETOBJSIZES && m_aItemSizes[Type]`).
    pub fn get(&self, ty: i32) -> Option<usize> {
        let idx = usize::try_from(ty).ok()?;
        let size = *self.0.get(idx)?;
        (size != 0).then_some(size as usize)
    }
}

/// Why [`unpack_delta`] rejected `ints` — mirrors `CSnapshotDelta::UnpackDelta`'s negative return
/// codes (`snapshot.cpp:517-631`), named instead of numbered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DeltaError {
    #[error("delta shorter than its 3-int header")]
    HeaderTooShort,
    #[error("num_deleted_items is negative")]
    NegativeDeletedCount,
    #[error("num_deleted_items claims more keys than remain in the delta")]
    DeletedKeysReadPastEnd,
    #[error("num_updated_items is negative")]
    NegativeUpdatedCount,
    #[error("an updated item's type/id header ran past the end of the delta")]
    UpdateHeaderReadPastEnd,
    #[error("an updated item's type is out of range (0..=MAX_TYPE)")]
    TypeOutOfRange,
    #[error("an updated item's id is out of range (0..=MAX_ID)")]
    IdOutOfRange,
    #[error("an updated item's explicit size ran past the end of the delta")]
    SizeReadPastEnd,
    #[error("an updated item's explicit size is negative or absurdly large")]
    InvalidSize,
    #[error("an updated item's data ran past the end of the delta")]
    ItemDataReadPastEnd,
    #[error("an updated item's size does not match the same key's existing size")]
    SizeMismatch,
    #[error("resulting snapshot would exceed MAX_ITEMS")]
    TooManyItems,
    #[error("resulting snapshot would exceed CSnapshot::MAX_SIZE (64 KiB)")]
    TooLarge,
}

/// `CSnapshotItem`'s own 4-byte `m_TypeAndId` header plus `len` `i32` data words
/// (`snapshot.h:12-26`) — one item's contribution to [`crate::snapshot::MAX_SIZE`].
fn item_bytes(len: usize) -> usize {
    4 + len * 4
}

/// The two fixed `int` header fields (`m_DataSize`, `m_NumItems`) plus one `i32` offset per item
/// (`CSnapshot::TotalSize`, `snapshot.h:37-38`) — everything in [`crate::snapshot::MAX_SIZE`]'s
/// budget *except* the items' own bytes ([`item_bytes`], summed separately so it can be tracked
/// incrementally instead of recomputed on every new item).
fn serialized_size_header(num_items: usize) -> usize {
    8 + num_items * 4
}

/// `CSnapshotDelta::UnpackDelta` (`snapshot.cpp:517-631`): reconstructs the target snapshot from
/// `base` and a delta's already-decompressed `i32` stream (see the module docs for where the
/// varint decompression happens). Never panics; every malformed/hostile shape in `ints` is a
/// [`DeltaError`], never a panic (task acceptance criterion 3/4).
pub fn unpack_delta(base: &Snapshot, ints: &[i32], sizes: &StaticSizes) -> Result<Snapshot, DeltaError> {
    if ints.len() < 3 {
        return Err(DeltaError::HeaderTooShort);
    }
    let num_deleted = ints[0];
    let num_updated = ints[1];
    // ints[2] (m_NumTempItems) is read but never used by the C++ reference either.
    let mut pos = 3usize;

    if num_deleted < 0 {
        return Err(DeltaError::NegativeDeletedCount);
    }
    let num_deleted = num_deleted as usize;
    if num_deleted > ints.len() - pos {
        return Err(DeltaError::DeletedKeysReadPastEnd);
    }
    let deleted_keys = &ints[pos..pos + num_deleted];
    pos += num_deleted;

    // Phase 1: keep every base item whose key was not deleted, in base's own order — matches
    // `snapshot.cpp:535-557`'s "copy all non deleted stuff" loop, which iterates `pFrom` in its
    // own item order and appends to the (initially empty) builder (each via `Builder.NewItem`,
    // which is where DDNet's own `CSnapshot::MAX_SIZE` check below lives, `snapshot.cpp:552,
    // 912-917` — checking the aggregate once here instead of per-item gives the same accept/
    // reject verdict, since nothing has been produced yet on a rejection either way).
    let mut items: Vec<SnapshotItem> = base
        .items
        .iter()
        .filter(|it| !deleted_keys.contains(&it.key))
        .cloned()
        .collect();
    // Running total of `serialized_size(&items)` (`CSnapshot::TotalSize`, `snapshot.h:38`),
    // maintained incrementally rather than recomputed on every new item (F3, review round 1) —
    // updates-in-place below never change it (an exact size match against the existing item was
    // already required), only a brand new item ever grows it.
    let mut data_bytes: usize = items.iter().map(|it| item_bytes(it.data.len())).sum();
    if serialized_size_header(items.len()) + data_bytes > crate::snapshot::MAX_SIZE {
        return Err(DeltaError::TooLarge);
    }

    if num_updated < 0 {
        return Err(DeltaError::NegativeUpdatedCount);
    }
    let num_updated = num_updated as usize;

    for _ in 0..num_updated {
        if pos + 2 > ints.len() {
            return Err(DeltaError::UpdateHeaderReadPastEnd);
        }
        let ty = ints[pos];
        let id = ints[pos + 1];
        pos += 2;
        if !(0..=MAX_TYPE).contains(&ty) {
            return Err(DeltaError::TypeOutOfRange);
        }
        if !(0..=MAX_ID).contains(&id) {
            return Err(DeltaError::IdOutOfRange);
        }

        let item_size = match sizes.get(ty) {
            Some(size) => size,
            None => {
                if pos + 1 > ints.len() {
                    return Err(DeltaError::SizeReadPastEnd);
                }
                let raw = ints[pos];
                pos += 1;
                if raw < 0 || raw as usize > MAX_ITEM_INTS {
                    return Err(DeltaError::InvalidSize);
                }
                raw as usize
            }
        };

        if item_size > ints.len() - pos {
            return Err(DeltaError::ItemDataReadPastEnd);
        }
        let diff_or_literal = &ints[pos..pos + item_size];
        pos += item_size;

        let key = (ty << 16) | id;
        let from_item = base.item_by_key(key);
        if let Some(from) = from_item
            && from.data.len() != item_size
        {
            return Err(DeltaError::SizeMismatch);
        }
        let new_data: Vec<i32> = match from_item {
            Some(from) => from
                .data
                .iter()
                .zip(diff_or_literal)
                .map(|(&past, &diff)| (past as u32).wrapping_add(diff as u32) as i32)
                .collect(),
            None => diff_or_literal.to_vec(),
        };

        match items.iter_mut().find(|it| it.key == key) {
            Some(existing) => {
                if existing.data.len() != item_size {
                    return Err(DeltaError::SizeMismatch);
                }
                existing.data = new_data;
            }
            None => {
                if items.len() >= MAX_ITEMS {
                    return Err(DeltaError::TooManyItems);
                }
                // `CSnapshotBuilder::NewItemRaw`'s own incremental bounds check
                // (`snapshot.cpp:912-917`): a brand new item is the only case that can grow the
                // total serialized size — check *before* pushing, so a single hostile "one
                // 16,384-int item" update, or many updates across repeated
                // `SnapAssembler::feed` calls that keep adding items without ever deleting any
                // (F3, review round 1: an earlier version had no such check at all, and a
                // reconstructed `Snapshot` could grow without bound), are both rejected exactly
                // where DDNet's own builder would refuse to allocate them, not after the fact.
                let new_item_bytes = item_bytes(new_data.len());
                if serialized_size_header(items.len() + 1) + data_bytes + new_item_bytes > crate::snapshot::MAX_SIZE {
                    return Err(DeltaError::TooLarge);
                }
                data_bytes += new_item_bytes;
                items.push(SnapshotItem { key, data: new_data });
            }
        }
    }

    Ok(Snapshot { items })
}

/// `CSnapshotDelta::CreateDelta` (`snapshot.cpp:302-383`): the encode side, used by this crate's
/// own round-trip tests (`tests/oracle_libtw2_snapshot.rs`, `#[cfg(test)]` below) — not needed by
/// a DDNet *client* at runtime (we only ever unpack deltas the server sends, never build our
/// own), but a direct, faithful port all the same, so those tests exercise real DDNet delta
/// semantics on both ends rather than only our own decoder's self-consistency.
///
/// Returns the delta's `i32` stream (still needs varint-compressing, e.g. via
/// [`crate::packer::pack_ints`], to become real wire bytes) — `None` if the two snapshots are
/// identical (`snapshot.cpp:379-381`: an empty delta is not "created" at all, matching
/// `NETMSG_SNAPEMPTY`'s condition).
pub fn create_delta(base: &Snapshot, target: &Snapshot, sizes: &StaticSizes) -> Option<Vec<i32>> {
    let mut deleted: Vec<i32> = base
        .items
        .iter()
        .filter(|it| target.item_by_key(it.key).is_none())
        .map(|it| it.key)
        .collect();
    deleted.sort_unstable();

    let mut updates: Vec<i32> = Vec::new();
    let mut num_updates = 0i32;
    for item in &target.items {
        let past = base.item_by_key(item.key);
        let include_size = sizes.get(item.internal_type()).is_none();
        let diff: Vec<i32> = match past {
            Some(p) if p.data.len() == item.data.len() => p
                .data
                .iter()
                .zip(&item.data)
                .map(|(&a, &b)| (b as u32).wrapping_sub(a as u32) as i32)
                .collect(),
            _ => item.data.clone(),
        };
        let changed = match past {
            Some(p) if p.data.len() == item.data.len() => diff.iter().any(|&d| d != 0),
            _ => true, // brand new (or size changed): always "included", like the C++ reference
        };
        if !changed {
            continue;
        }
        updates.push(item.internal_type());
        updates.push(item.id());
        if include_size {
            updates.push(item.data.len() as i32);
        }
        updates.extend_from_slice(&diff);
        num_updates += 1;
    }

    if deleted.is_empty() && num_updates == 0 {
        return None;
    }

    let mut out = Vec::with_capacity(3 + deleted.len() + updates.len());
    out.push(deleted.len() as i32);
    out.push(num_updates);
    out.push(0); // num_temp_items
    out.extend_from_slice(&deleted);
    out.extend_from_slice(&updates);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generated::objects;

    fn flag(x: i32, y: i32, id: i32) -> SnapshotItem {
        SnapshotItem {
            key: (objects::Flag::ID << 16) | id,
            data: vec![x, y, 0],
        }
    }

    #[test]
    fn empty_delta_against_empty_base_yields_empty_snapshot() {
        let sizes = StaticSizes::ddnet_06();
        let base = Snapshot::empty();
        let ints = [0, 0, 0];
        let out = unpack_delta(&base, &ints, &sizes).unwrap();
        assert_eq!(out, Snapshot::empty());
    }

    #[test]
    fn new_item_with_explicit_size_is_added() {
        let sizes = StaticSizes::none(); // force explicit sizes on the wire
        let base = Snapshot::empty();
        // 0 deleted, 1 updated, 0 temp; type=Flag::ID, id=0, size=3, data=[7,8,0]
        let ints = [0, 1, 0, objects::Flag::ID, 0, 3, 7, 8, 0];
        let out = unpack_delta(&base, &ints, &sizes).unwrap();
        assert_eq!(out.items, vec![flag(7, 8, 0)]);
    }

    #[test]
    fn new_item_with_static_size_omits_size_on_wire() {
        let sizes = StaticSizes::ddnet_06();
        let base = Snapshot::empty();
        let ints = [0, 1, 0, objects::Flag::ID, 0, 7, 8, 0]; // no explicit size field
        let out = unpack_delta(&base, &ints, &sizes).unwrap();
        assert_eq!(out.items, vec![flag(7, 8, 0)]);
    }

    #[test]
    fn update_diffs_against_base_item_with_wraparound() {
        let sizes = StaticSizes::ddnet_06();
        let base = Snapshot {
            items: vec![flag(i32::MAX, 1, 0)],
        };
        // diff so that MAX + diff wraps to MIN: diff = 1 (wrapping).
        let ints = [0, 1, 0, objects::Flag::ID, 0, 1, 0, 0];
        let out = unpack_delta(&base, &ints, &sizes).unwrap();
        assert_eq!(out.items[0].data[0], i32::MIN);
    }

    #[test]
    fn deleted_item_is_removed() {
        let sizes = StaticSizes::ddnet_06();
        let base = Snapshot {
            items: vec![flag(1, 2, 0), flag(3, 4, 1)],
        };
        let key0 = objects::Flag::ID << 16;
        let ints = [1, 0, 0, key0];
        let out = unpack_delta(&base, &ints, &sizes).unwrap();
        assert_eq!(out.items, vec![flag(3, 4, 1)]);
    }

    #[test]
    fn header_too_short_errors_not_panics() {
        let sizes = StaticSizes::ddnet_06();
        assert_eq!(
            unpack_delta(&Snapshot::empty(), &[], &sizes).unwrap_err(),
            DeltaError::HeaderTooShort
        );
        assert_eq!(
            unpack_delta(&Snapshot::empty(), &[1, 2], &sizes).unwrap_err(),
            DeltaError::HeaderTooShort
        );
    }

    #[test]
    fn negative_deleted_count_errors() {
        let sizes = StaticSizes::ddnet_06();
        assert_eq!(
            unpack_delta(&Snapshot::empty(), &[-1, 0, 0], &sizes).unwrap_err(),
            DeltaError::NegativeDeletedCount
        );
    }

    #[test]
    fn deleted_count_larger_than_remaining_errors() {
        let sizes = StaticSizes::ddnet_06();
        assert_eq!(
            unpack_delta(&Snapshot::empty(), &[5, 0, 0, 1, 2], &sizes).unwrap_err(),
            DeltaError::DeletedKeysReadPastEnd
        );
    }

    #[test]
    fn type_out_of_range_errors() {
        let sizes = StaticSizes::ddnet_06();
        let ints = [0, 1, 0, MAX_TYPE + 1, 0, 0];
        assert_eq!(
            unpack_delta(&Snapshot::empty(), &ints, &sizes).unwrap_err(),
            DeltaError::TypeOutOfRange
        );
        let ints2 = [0, 1, 0, -1, 0, 0];
        assert_eq!(
            unpack_delta(&Snapshot::empty(), &ints2, &sizes).unwrap_err(),
            DeltaError::TypeOutOfRange
        );
    }

    #[test]
    fn id_out_of_range_errors() {
        let sizes = StaticSizes::ddnet_06();
        let ints = [0, 1, 0, objects::Flag::ID, MAX_ID + 1, 0];
        assert_eq!(
            unpack_delta(&Snapshot::empty(), &ints, &sizes).unwrap_err(),
            DeltaError::IdOutOfRange
        );
    }

    #[test]
    fn negative_explicit_size_errors() {
        let sizes = StaticSizes::none();
        let ints = [0, 1, 0, objects::Flag::ID, 0, -1];
        assert_eq!(
            unpack_delta(&Snapshot::empty(), &ints, &sizes).unwrap_err(),
            DeltaError::InvalidSize
        );
    }

    #[test]
    fn absurd_explicit_size_errors_instead_of_huge_alloc() {
        let sizes = StaticSizes::none();
        let ints = [0, 1, 0, objects::Flag::ID, 0, i32::MAX];
        assert_eq!(
            unpack_delta(&Snapshot::empty(), &ints, &sizes).unwrap_err(),
            DeltaError::InvalidSize
        );
    }

    #[test]
    fn item_data_shorter_than_claimed_size_errors_not_panics() {
        let sizes = StaticSizes::none();
        let ints = [0, 1, 0, objects::Flag::ID, 0, 3, 7, 8]; // claims 3 ints, only 2 present
        assert_eq!(
            unpack_delta(&Snapshot::empty(), &ints, &sizes).unwrap_err(),
            DeltaError::ItemDataReadPastEnd
        );
    }

    #[test]
    fn size_mismatch_against_base_errors() {
        let sizes = StaticSizes::none();
        let base = Snapshot {
            items: vec![SnapshotItem {
                key: objects::Flag::ID << 16,
                data: vec![1, 2, 3],
            }],
        };
        // Same key, but claims size 2 instead of the base item's 3.
        let ints = [0, 1, 0, objects::Flag::ID, 0, 2, 5, 6];
        assert_eq!(
            unpack_delta(&base, &ints, &sizes).unwrap_err(),
            DeltaError::SizeMismatch
        );
    }

    #[test]
    fn create_then_unpack_roundtrips() {
        let sizes = StaticSizes::ddnet_06();
        let base = Snapshot {
            items: vec![flag(1, 2, 0)],
        };
        let target = Snapshot {
            items: vec![flag(1, 99, 0), flag(5, 6, 1)],
        };
        let ints = create_delta(&base, &target, &sizes).expect("should produce a non-empty delta");
        let out = unpack_delta(&base, &ints, &sizes).unwrap();
        let mut got: Vec<_> = out.items.clone();
        let mut want: Vec<_> = target.items.clone();
        got.sort_by_key(|i| i.key);
        want.sort_by_key(|i| i.key);
        assert_eq!(got, want);
    }

    #[test]
    fn identical_snapshots_produce_no_delta() {
        let sizes = StaticSizes::ddnet_06();
        let snap = Snapshot {
            items: vec![flag(1, 2, 0)],
        };
        assert_eq!(create_delta(&snap, &snap, &sizes), None);
    }

    // --- F3 (review round 1): a reconstructed snapshot must never exceed `CSnapshot::MAX_SIZE`.

    #[test]
    fn single_oversized_item_is_rejected_not_allocated() {
        // The reviewer's exact repro: one item of 16,384 ints — `serialized_size_header(1) +
        // item_bytes(16384)` = 12 + 65540 = 65552 B, just over `MAX_SIZE` (65536) — DDNet's own
        // builder would refuse this (`NewItemRaw`'s incremental check), so must we.
        let sizes = StaticSizes::none();
        let mut ints = vec![0, 1, 0, objects::Flag::ID, 0, 16384];
        ints.extend(std::iter::repeat_n(0, 16384));
        assert_eq!(
            unpack_delta(&Snapshot::empty(), &ints, &sizes).unwrap_err(),
            DeltaError::TooLarge
        );
    }

    #[test]
    fn largest_still_accepted_single_item_fits_exactly_at_the_max_size_boundary() {
        // The largest single new item a fresh (empty-base) snapshot can hold without crossing
        // `MAX_SIZE`: `serialized_size_header(1) + item_bytes(n) == MAX_SIZE` solved for `n` —
        // must succeed exactly at that boundary, and the very next size up must not (the boundary
        // itself, not an off-by-one-too-strict or too-loose rejection).
        let item_ints = (crate::snapshot::MAX_SIZE - serialized_size_header(1) - 4) / 4;
        assert_eq!(
            serialized_size_header(1) + item_bytes(item_ints),
            crate::snapshot::MAX_SIZE,
            "test setup: this must land exactly on the boundary"
        );
        let sizes = StaticSizes::none();
        let mut ints = vec![0, 1, 0, objects::Flag::ID, 0, item_ints as i32];
        ints.extend(std::iter::repeat_n(0, item_ints));
        let out = unpack_delta(&Snapshot::empty(), &ints, &sizes).unwrap();
        assert_eq!(out.items[0].data.len(), item_ints);

        let mut ints_too_big = vec![0, 1, 0, objects::Flag::ID, 0, (item_ints + 1) as i32];
        ints_too_big.extend(std::iter::repeat_n(0, item_ints + 1));
        assert_eq!(
            unpack_delta(&Snapshot::empty(), &ints_too_big, &sizes).unwrap_err(),
            DeltaError::TooLarge
        );
    }

    #[test]
    fn cumulative_growth_across_many_deltas_is_eventually_rejected() {
        // The reviewer's other repro: repeatedly applying a delta that only ever *adds* new
        // items (never deletes any) against the growing result — over enough ticks, the
        // resulting `Snapshot` must be rejected once it would cross `MAX_SIZE`, exactly like a
        // real `SnapAssembler` session would refuse it (this test drives `unpack_delta` directly,
        // the same way `assembly.rs`'s own analogous test drives the full assembler).
        let sizes = StaticSizes::none();
        let mut snap = Snapshot::empty();
        let mut rejected = false;
        for tick in 0..2000i32 {
            let mut ints = vec![0, 1, 0, objects::Flag::ID, tick & 0xffff, 64 /* explicit size */];
            ints.extend(std::iter::repeat_n(7, 64));
            match unpack_delta(&snap, &ints, &sizes) {
                Ok(next) => snap = next,
                Err(DeltaError::TooLarge) => {
                    rejected = true;
                    break;
                }
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
        assert!(rejected, "expected cumulative growth to eventually hit MAX_SIZE");
        assert!(
            serialized_size_header(snap.items.len())
                + snap.items.iter().map(|it| item_bytes(it.data.len())).sum::<usize>()
                <= crate::snapshot::MAX_SIZE
        );
    }
}
