//! Parses a `CHUNKTYPE_SNAPSHOT` chunk's decompressed `i32` stream as a raw, on-disk `CSnapshot`
//! (`engine/shared/snapshot.h:29-70`, `.cpp:23-176`, DDNet 20.1 `c9d208138f85755521f16a0096b6fe036c5c8698`),
//! which carries the original Teeworlds zlib-style notice:
//!
//!   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//!   If you are missing that file, acquire a complete release at teeworlds.com.
//!
//! into [`ddai_net::snapshot::Snapshot`].
//!
//! This has no C++ *function* to port 1:1 — `CDemoPlayer` never calls `CSnapshot::IsValid` or
//! walks items itself; it just hands the raw bytes to its listener. This module instead ports
//! `CSnapshot`'s validation/iteration *semantics* (an *altered* source version, same as every
//! other direct port in this crate: safe Rust, same byte layout and accept/reject verdicts),
//! which is exactly what any listener (the real game client, this crate) has to reproduce to make
//! sense of those bytes. `ddai_net::snapshot` only models a snapshot as a decoded
//! `Vec<SnapshotItem>` (built by `ddai_net::delta` for wire deltas) — it has no code for this
//! *on-disk* `{data_size, num_items, offsets[], data}` byte layout, so it lives here rather than
//! there.
//!
//! Wire representation: a `CHUNKTYPE_SNAPSHOT` chunk's payload, after Huffman + variable-int
//! decompression, is the raw memory image of a `CSnapshot`, reinterpreted as `i32`s (exactly the
//! same "decompressed bytes are already an int array" property `crate::reader` relies on for
//! deltas — see `CDemoPlayer::DoTick`, `demo.cpp:777-799`, which casts `m_aChunkData` straight to
//! `CSnapshot*`):
//!
//! ```text
//! ints[0]                    m_DataSize   (bytes of item data that follows the offset table)
//! ints[1]                    m_NumItems
//! ints[2 .. 2+NumItems]      Offsets[]    (byte offset of each item within the data area)
//! ints[2+NumItems ..]        item data — NumItems items back to back, each:
//!                              one i32 `m_TypeAndId` (key) + `(item span - 4) / 4` data i32s
//! ```

use crate::error::TickIterError;
use ddai_net::snapshot::{MAX_ITEMS, Snapshot, SnapshotItem};

/// `sizeof(CSnapshot)` (`snapshot.h:29-31`: two `int` fields, no vtable — `CSnapshot` declares no
/// virtual methods).
const SNAPSHOT_HEADER_INTS: usize = 2;
/// `sizeof(CSnapshotItem)` (`snapshot.h:12-26`: one `int m_TypeAndId`) — subtracted from an item's
/// byte span to get its data size (`CSnapshot::GetItemSize`, `snapshot.cpp:30-35`).
const ITEM_HEADER_BYTES: i64 = 4;
/// `CSnapshot::MAX_SIZE` (`snapshot.h:50`).
const MAX_SIZE_BYTES: i64 = ddai_net::snapshot::MAX_SIZE as i64;

/// Parses `ints` (the chunk's fully decompressed payload, already capped by
/// `crate::reader::MAX_INTS` by the caller) as a raw `CSnapshot`, applying exactly
/// `CSnapshot::IsValid` (`snapshot.cpp:140-176`)'s checks — an offset/size that fails any of them
/// is reported as [`TickIterError::TruncatedChunk`] (this crate has no separate "invalid
/// snapshot" variant — like the C++ reference's own `IsValid` check inside `DoTick`, a failure
/// here just means this chunk's data isn't usable, not that the file is unreadable, so
/// `crate::reader` treats it the same as any other single-chunk decode failure: log and move on).
///
/// Never panics: every offset/size read from `ints` is range- and alignment-checked before use,
/// and all size arithmetic uses `i64` so no attacker-chosen `i32` can overflow it.
pub fn parse_raw_snapshot(ints: &[i32]) -> Result<Snapshot, TickIterError> {
    // `ActualSize < sizeof(CSnapshot)` (`snapshot.cpp:143`).
    if ints.len() < SNAPSHOT_HEADER_INTS {
        return Err(TickIterError::TruncatedChunk);
    }
    let data_size = ints[0];
    let num_items = ints[1];

    let actual_size_bytes = ints.len() as i64 * 4;
    // `ActualSize > MAX_SIZE` (`snapshot.cpp:144`) — structurally unreachable given the caller's
    // own MAX_INTS cap, kept for a self-contained, directly-citable proof against the C++ check.
    if actual_size_bytes > MAX_SIZE_BYTES {
        return Err(TickIterError::TruncatedChunk);
    }
    // `m_NumItems < 0 || m_NumItems > MAX_ITEMS` (`snapshot.cpp:145-146`).
    if num_items < 0 || num_items as usize > MAX_ITEMS {
        return Err(TickIterError::TruncatedChunk);
    }
    // `m_DataSize < 0` (`snapshot.cpp:147`).
    if data_size < 0 {
        return Err(TickIterError::TruncatedChunk);
    }
    let num_items = num_items as usize;
    let data_size = i64::from(data_size);

    // `ActualSize != TotalSize()` (`snapshot.cpp:148`, `TotalSize = sizeof(CSnapshot) +
    // OffsetSize() + m_DataSize`, `snapshot.h:37`).
    let total_size_bytes = 8 + (num_items as i64) * 4 + data_size;
    if actual_size_bytes != total_size_bytes {
        return Err(TickIterError::TruncatedChunk);
    }

    let offsets_start = SNAPSHOT_HEADER_INTS;
    let data_start = offsets_start + num_items;

    // `pOffsets[Index] < 0 || pOffsets[Index] > m_DataSize || pOffsets[Index] % 4 != 0`
    // (`snapshot.cpp:154-163`) — validated for every item up front, exactly like the reference,
    // before any item's data is read (so every `end_off` used below — either another item's
    // already-validated offset, or `data_size` itself — is known in-range by the time it's used).
    let mut offsets = Vec::with_capacity(num_items);
    for i in 0..num_items {
        let off = i64::from(ints[offsets_start + i]);
        if off < 0 || off > data_size || off % 4 != 0 {
            return Err(TickIterError::TruncatedChunk);
        }
        offsets.push(off);
    }

    let mut items = Vec::with_capacity(num_items);
    for i in 0..num_items {
        let start_off = offsets[i];
        // `GetItemSize`: last item's span runs to `m_DataSize`; every other item's span runs to
        // the *next index's* offset, not "the next larger offset" (offsets need not be sorted —
        // `snapshot.cpp:30-35` never assumes they are, so neither do we).
        let end_off = if i + 1 == num_items { data_size } else { offsets[i + 1] };
        let span = end_off - start_off;
        // `ItemSize < 0 || ItemSize % 4 != 0` where `ItemSize = span - ITEM_HEADER_BYTES`
        // (`snapshot.cpp:167-171`) — `span < 4` is folded in here (an item always has at least
        // its own key int).
        if span < ITEM_HEADER_BYTES || span % 4 != 0 {
            return Err(TickIterError::TruncatedChunk);
        }
        let item_data_ints = ((span - ITEM_HEADER_BYTES) / 4) as usize;

        // In bounds by construction: `start_off <= data_size`, `start_off % 4 == 0`, and
        // `data_start + data_size / 4 == ints.len()` (from the `TotalSize` equality above), so
        // `item_start + 1 + item_data_ints <= ints.len()` — see the module docs.
        let item_start = data_start + (start_off / 4) as usize;
        let key = ints[item_start];
        let data = ints[item_start + 1..item_start + 1 + item_data_ints].to_vec();
        items.push(SnapshotItem { key, data });
    }

    Ok(Snapshot { items })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes a snapshot's raw `ints` layout the same way `CSnapshotBuilder::Finish` would
    /// (`snapshot.cpp:940-960`, not ported here since nothing in this crate ever *writes* a
    /// demo) — used to build well-formed inputs for the round-trip tests below.
    fn encode(items: &[(i32, &[i32])]) -> Vec<i32> {
        let mut offsets = Vec::new();
        let mut data = Vec::new();
        for &(key, item_data) in items {
            offsets.push(data.len() as i32 * 4);
            data.push(key);
            data.extend_from_slice(item_data);
        }
        let mut out = vec![data.len() as i32 * 4, items.len() as i32];
        out.extend_from_slice(&offsets);
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn round_trips_empty_snapshot() {
        let ints = encode(&[]);
        let snap = parse_raw_snapshot(&ints).unwrap();
        assert!(snap.items.is_empty());
    }

    #[test]
    fn round_trips_multiple_items() {
        let ints = encode(&[(0x0001_0007, &[1, 2, 3]), (0x0002_0003, &[]), (0x0003_0000, &[42])]);
        let snap = parse_raw_snapshot(&ints).unwrap();
        assert_eq!(snap.items.len(), 3);
        assert_eq!(snap.items[0].key, 0x0001_0007);
        assert_eq!(snap.items[0].data, vec![1, 2, 3]);
        assert_eq!(snap.items[1].key, 0x0002_0003);
        assert!(snap.items[1].data.is_empty());
        assert_eq!(snap.items[2].key, 0x0003_0000);
        assert_eq!(snap.items[2].data, vec![42]);
    }

    #[test]
    fn rejects_too_short_for_header() {
        assert_eq!(parse_raw_snapshot(&[]), Err(TickIterError::TruncatedChunk));
        assert_eq!(parse_raw_snapshot(&[0]), Err(TickIterError::TruncatedChunk));
    }

    #[test]
    fn rejects_negative_data_size() {
        assert_eq!(parse_raw_snapshot(&[-1, 0]), Err(TickIterError::TruncatedChunk));
    }

    #[test]
    fn rejects_negative_num_items() {
        assert_eq!(parse_raw_snapshot(&[0, -1]), Err(TickIterError::TruncatedChunk));
    }

    #[test]
    fn rejects_num_items_over_max() {
        let ints = vec![0, (MAX_ITEMS as i32) + 1];
        assert_eq!(parse_raw_snapshot(&ints), Err(TickIterError::TruncatedChunk));
    }

    #[test]
    fn rejects_size_mismatch() {
        // Claims data_size=4, num_items=0, but no data ints follow.
        assert_eq!(parse_raw_snapshot(&[4, 0]), Err(TickIterError::TruncatedChunk));
    }

    #[test]
    fn rejects_out_of_range_offset() {
        // data_size=4 (one item's worth), num_items=1, offset=8 (past data_size).
        let ints = vec![4, 1, 8, 0xDEAD];
        assert_eq!(parse_raw_snapshot(&ints), Err(TickIterError::TruncatedChunk));
    }

    #[test]
    fn rejects_misaligned_offset() {
        let ints = vec![4, 1, 1, 0xDEAD];
        assert_eq!(parse_raw_snapshot(&ints), Err(TickIterError::TruncatedChunk));
    }

    #[test]
    fn rejects_negative_item_span() {
        // Two items whose offsets go *backwards* (offset[1] < offset[0]) produce a negative span
        // for item 0 — DDNet's own format never asserts offsets are sorted, so this is a
        // legitimately malformed-but-in-range-individually case the span check must still catch.
        let ints = vec![8, 2, 4, 0, 0xAAAA, 0xBBBB];
        assert_eq!(parse_raw_snapshot(&ints), Err(TickIterError::TruncatedChunk));
    }

    #[test]
    fn accepts_max_items_boundary() {
        let items: Vec<(i32, &[i32])> = (0..MAX_ITEMS).map(|i| (i as i32, &[][..])).collect();
        let ints = encode(&items);
        let snap = parse_raw_snapshot(&ints).unwrap();
        assert_eq!(snap.items.len(), MAX_ITEMS);
    }
}
