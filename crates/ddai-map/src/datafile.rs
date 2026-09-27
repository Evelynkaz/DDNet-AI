//! The DDNet "datafile" container format (versions 3 and 4): the generic item/data-blob store
//! that `.map` files (and demos, though this crate only reads maps) are built on. Mirrors
//! `engine/shared/datafile.cpp`'s `CDataFileReader` (DDNet 20.1,
//! `c9d208138f85755521f16a0096b6fe036c5c8698`) field-for-field; every check below cites the
//! exact function/line it replicates.
//!
//! Byte layout (all integers little-endian; DDNet only byte-swaps on a big-endian *host*, which
//! this crate — like the C++ oracle compiled for this same x86-64 Linux host — never runs on, so
//! there is no swapping code here either; see `datafile.cpp:30-35`'s `#if
//! defined(CONF_ARCH_ENDIAN_BIG)` guard):
//!
//! ```text
//! header            [36]   magic[4] version size swaplen num_item_types num_items num_raw_data
//!                           item_size data_size   (8 i32 fields after the magic)
//! item_types[i]     [12]   type start num                      num_item_types times
//! item_offsets[i]   [4]    (relative to the start of `items`)   num_items times
//! data_offsets[i]   [4]    (relative to the start of raw data)  num_raw_data times
//! data_sizes[i]     [4]    v4 only: declared UNCOMPRESSED size  num_raw_data times
//! items             [..]   item_size bytes: concatenated {type_and_id:u32, size:i32, payload}
//! raw data          [..]   data_size bytes: concatenated blobs (v3: raw; v4: zlib-deflated)
//! ```

use crate::error::MapError;
use std::io::Read;

/// DDNet's own cap on the header+item-table allocation (`datafile.cpp:656`,
/// `constexpr int64_t MaxAllocSize = 2GiB`) — applied here to that same allocation (the parsed
/// item-type/item-offset/data-offset/data-size tables this module builds while parsing the
/// header). See [`MapError::AllocationTooLarge`]. A real, well-formed map never comes close: the
/// largest file in the task's corpus is ~10 MiB.
///
/// This does **not** bound any single *decompressed data blob* — an earlier version of this
/// module also gated `data()`'s per-blob allocation on this same constant, which review round 1
/// finding F2 pointed out can never actually fire: a v4 blob's declared size is read from the
/// file as an `i32`, so it can never exceed `i32::MAX` (~2.1 GiB), which already fits under this
/// 2 GiB cap. [`Self::data_bounded`] fixes this properly (bounding the allocation to what the
/// *caller* actually needs, not to what the file merely claims); see its own doc comment.
const MAX_ALLOC_BYTES: i64 = 2 * 1024 * 1024 * 1024;

const HEADER_SIZE: i64 = 36; // sizeof(CDatafileHeader): 4 (magic) + 8 * 4 (i32 fields).
const SIZE_OFFSET: i64 = 16; // CDatafileHeader::SizeOffset(): magic(4) + version(4) + size(4) + swaplen(4).
const MAX_ITEM_TYPE: i64 = 0xFFFF;
const ITEMTYPE_EX: u16 = 0xFFFF;
const ITEM_HEADER_SIZE: i64 = 8; // sizeof(CDatafileItem): u32 type_and_id + i32 size.
const ITEM_TYPE_ENTRY_SIZE: i64 = 12; // sizeof(CDatafileItemType): 3 * i32.

fn i32_at(bytes: &[u8], offset: usize) -> Result<i32, MapError> {
    let b = bytes.get(offset..offset + 4).ok_or(MapError::HeaderTruncated)?;
    Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[derive(Debug)]
struct ItemTypeEntry {
    type_: u16,
    start: u32,
    num: u32,
}

/// A parsed, validated datafile container, borrowing the original bytes. Every table and offset
/// has already passed the same structural checks `CDatafile::Validate` (datafile.cpp:388-501)
/// runs, so accessors here only need to guard against out-of-range *indices*, never against a
/// malformed table.
#[derive(Debug)]
pub struct Datafile<'a> {
    bytes: &'a [u8],
    item_types: Vec<ItemTypeEntry>,
    item_offsets: Vec<u32>,
    data_offsets: Vec<u32>,
    /// `Some` only for version 4 (v3 data is stored uncompressed with no declared size table).
    data_sizes: Option<Vec<u32>>,
    item_table_start: usize,
    item_size_total: u32,
    data_start: usize,
    data_size_total: u32,
}

/// One item read out of the item table: its declared id and its payload (the bytes after the
/// 8-byte `CDatafileItem` header — `pItem + 1` in DDNet's terms). The item's *type* isn't kept
/// here — every caller already knows it, having selected `index` from a [`Self::type_range`] of
/// that exact type.
pub struct RawItem<'a> {
    pub id: u16,
    pub payload: &'a [u8],
}

impl<'a> Datafile<'a> {
    /// Parses and fully validates the container (header, item-type/offset/data-offset tables,
    /// every item's own size/type/id). Does **not** decompress any data blob yet (`data()` does
    /// that lazily, matching DDNet's own lazy `GetData` — see `map.cpp:189-195`'s comment about
    /// only eagerly loading the game layer).
    pub fn parse(bytes: &'a [u8]) -> Result<Self, MapError> {
        if (bytes.len() as i64) < HEADER_SIZE {
            return Err(MapError::HeaderTruncated);
        }
        let magic = &bytes[0..4];
        // datafile.cpp:565-571: either byte order is accepted; this crate (like the C++ oracle,
        // compiled for the same little-endian host DDNet itself runs its map editor/server on)
        // never swaps either way — see this module's top doc comment.
        if magic != b"DATA" && magic != b"ATAD" {
            return Err(MapError::BadMagic);
        }
        let version = i32_at(bytes, 4)?;
        if version != 3 && version != 4 {
            return Err(MapError::UnsupportedDatafileVersion(version));
        }
        let header_size = i32_at(bytes, 8)? as i64;
        let header_swaplen = i32_at(bytes, 12)? as i64;
        let num_item_types = i32_at(bytes, 16)? as i64;
        let num_items = i32_at(bytes, 20)? as i64;
        let num_raw_data = i32_at(bytes, 24)? as i64;
        let item_size = i32_at(bytes, 28)? as i64;
        let data_size = i32_at(bytes, 32)? as i64;

        // datafile.cpp:584-596.
        if !(0..=MAX_ITEM_TYPE + 1).contains(&num_item_types) {
            return Err(MapError::InvalidHeaderField("num_item_types"));
        }
        if num_items < 0 {
            return Err(MapError::InvalidHeaderField("num_items"));
        }
        if num_raw_data < 0 {
            return Err(MapError::InvalidHeaderField("num_raw_data"));
        }
        if item_size < 0 || item_size % 4 != 0 {
            return Err(MapError::InvalidHeaderField("item_size"));
        }
        if data_size < 0 {
            return Err(MapError::InvalidHeaderField("data_size"));
        }

        // datafile.cpp:598-621: the header's own counts must add up to the file's real length —
        // this is the load-bearing bound that keeps every later allocation in this module tied
        // to the size of the input the caller already holds, not to an attacker-chosen claim.
        let mut size = num_item_types * ITEM_TYPE_ENTRY_SIZE + num_items * 4 + num_raw_data * 4;
        let size_fix = if version == 4 { num_raw_data * 4 } else { 0 };
        size += size_fix;
        size += item_size;
        if size > MAX_ALLOC_BYTES {
            return Err(MapError::AllocationTooLarge("item/type/offset tables"));
        }
        let file_len = bytes.len() as i64;
        if HEADER_SIZE + size + data_size != file_len {
            return Err(MapError::SizeMismatch);
        }

        // datafile.cpp:622-654: `m_Size`/`m_Swaplen` are redundant with the above (computed from
        // the same counts) but DDNet still cross-checks them against the file length, with a
        // legacy fix-up for v4 maps written before `m_Size`/`m_Swaplen` counted the data-sizes
        // table. Purely a validation of the header's *own* self-consistency; doesn't change how
        // this crate locates anything.
        let header_file_size = header_size + SIZE_OFFSET;
        if header_file_size != file_len && !(size_fix != 0 && header_file_size + size_fix == file_len) {
            return Err(MapError::HeaderSizeMismatch);
        }
        let header_swaplen_total = header_swaplen + SIZE_OFFSET;
        let file_size_swaplen = file_len - data_size;
        if header_swaplen_total != file_size_swaplen
            && !(header_swaplen % 4 == 0 && size_fix != 0 && header_swaplen_total + size_fix == file_size_swaplen)
        {
            return Err(MapError::HeaderSizeMismatch);
        }

        // --- layout offsets, all relative to the end of the 36-byte header --------------------
        let mut pos = HEADER_SIZE as usize;
        let item_types_bytes = (num_item_types * ITEM_TYPE_ENTRY_SIZE) as usize;
        let item_types_start = pos;
        pos += item_types_bytes;
        let item_offsets_start = pos;
        pos += (num_items * 4) as usize;
        let data_offsets_start = pos;
        pos += (num_raw_data * 4) as usize;
        let data_sizes_start = pos;
        if version == 4 {
            pos += (num_raw_data * 4) as usize;
        }
        let item_table_start = pos;
        pos += item_size as usize;
        let data_start = pos;
        // `pos + data_size == file_len` is already guaranteed by the SizeMismatch check above.

        // --- item type table: datafile.cpp:400-417 --------------------------------------------
        let mut item_types = Vec::with_capacity(num_item_types as usize);
        let mut counted_items: i64 = 0;
        let mut seen_types = std::collections::HashSet::with_capacity(num_item_types as usize);
        for i in 0..num_item_types {
            let base = item_types_start + (i * ITEM_TYPE_ENTRY_SIZE) as usize;
            let type_ = i32_at(bytes, base)?;
            let start = i32_at(bytes, base + 4)? as i64;
            let num = i32_at(bytes, base + 8)? as i64;
            if !(0..=MAX_ITEM_TYPE).contains(&(type_ as i64)) {
                return Err(MapError::InvalidItemTypeTable);
            }
            if !seen_types.insert(type_) {
                return Err(MapError::InvalidItemTypeTable);
            }
            if num <= 0 || start != counted_items {
                return Err(MapError::InvalidItemTypeTable);
            }
            counted_items += num;
            if counted_items > num_items {
                return Err(MapError::InvalidItemTypeTable);
            }
            item_types.push(ItemTypeEntry {
                type_: type_ as u16,
                start: start as u32,
                num: num as u32,
            });
        }
        if counted_items != num_items {
            return Err(MapError::InvalidItemTypeTable);
        }

        // --- item offsets: datafile.cpp:419-434 -------------------------------------------------
        let mut item_offsets = Vec::with_capacity(num_items as usize);
        let mut prev: i64 = -1;
        for i in 0..num_items {
            let off = i32_at(bytes, item_offsets_start + (i * 4) as usize)? as i64;
            if i == 0 {
                if off != 0 {
                    return Err(MapError::InvalidItemOffsets);
                }
            } else if off <= prev {
                return Err(MapError::InvalidItemOffsets);
            }
            if off >= item_size {
                return Err(MapError::InvalidItemOffsets);
            }
            prev = off;
            item_offsets.push(off as u32);
        }

        // --- data offsets: datafile.cpp:466-481 -------------------------------------------------
        let mut data_offsets = Vec::with_capacity(num_raw_data as usize);
        let mut prev: i64 = -1;
        for i in 0..num_raw_data {
            let off = i32_at(bytes, data_offsets_start + (i * 4) as usize)? as i64;
            if i == 0 {
                if off != 0 {
                    return Err(MapError::InvalidDataOffsets);
                }
            } else if off <= prev {
                return Err(MapError::InvalidDataOffsets);
            }
            if off >= data_size {
                return Err(MapError::InvalidDataOffsets);
            }
            prev = off;
            data_offsets.push(off as u32);
        }

        // --- v4 declared uncompressed data sizes: datafile.cpp:484-497 -------------------------
        let data_sizes = if version == 4 {
            let mut sizes = Vec::with_capacity(num_raw_data as usize);
            for i in 0..num_raw_data {
                let s = i32_at(bytes, data_sizes_start + (i * 4) as usize)?;
                if s < 0 {
                    return Err(MapError::InvalidDataSize);
                }
                // `s == 0` is a real (if rare) quirk DDNet tolerates at this stage — see
                // `data()`'s doc comment.
                sizes.push(s as u32);
            }
            Some(sizes)
        } else {
            None
        };

        let df = Datafile {
            bytes,
            item_types,
            item_offsets,
            data_offsets,
            data_sizes,
            item_table_start,
            item_size_total: item_size as u32,
            data_start,
            data_size_total: data_size as u32,
        };
        df.validate_items()?;
        Ok(df)
    }

    fn file_item_size(&self, index: usize) -> i64 {
        let off = self.item_offsets[index] as i64;
        if index == self.item_offsets.len() - 1 {
            self.item_size_total as i64 - off
        } else {
            self.item_offsets[index + 1] as i64 - off
        }
    }

    fn file_data_size(&self, index: usize) -> i64 {
        let off = self.data_offsets[index] as i64;
        if index == self.data_offsets.len() - 1 {
            self.data_size_total as i64 - off
        } else {
            self.data_offsets[index + 1] as i64 - off
        }
    }

    /// datafile.cpp:436-464: per item-type-table entry, every item in its range must actually
    /// declare that type, have a sane (4-byte-aligned, matching-the-file) size, and (except
    /// `ITEMTYPE_EX`, which DDNet tolerates duplicates of — datafile.cpp:448-453) a unique `Id`
    /// among items of the same type.
    fn validate_items(&self) -> Result<(), MapError> {
        let mut total: i64 = 0;
        for ty in &self.item_types {
            let mut seen_ids = std::collections::HashSet::with_capacity(ty.num as usize);
            for index in ty.start as usize..ty.start as usize + ty.num as usize {
                let file_size = self.file_item_size(index);
                if file_size < ITEM_HEADER_SIZE {
                    return Err(MapError::InvalidItem);
                }
                let base = self.item_table_start + self.item_offsets[index] as usize;
                let type_and_id = u32::from_le_bytes(
                    self.bytes[base..base + 4]
                        .try_into()
                        .map_err(|_| MapError::InvalidItem)?,
                );
                let item_type = (type_and_id >> 16) as u16;
                let item_id = (type_and_id & 0xFFFF) as u16;
                let declared_size = i32_at(self.bytes, base + 4)? as i64;
                if item_type != ty.type_ {
                    return Err(MapError::InvalidItem);
                }
                if item_type != ITEMTYPE_EX && !seen_ids.insert(item_id) {
                    return Err(MapError::InvalidItem);
                }
                if declared_size < 0 || declared_size % 4 != 0 || declared_size != file_size - ITEM_HEADER_SIZE {
                    return Err(MapError::InvalidItem);
                }
                total += file_size;
                if total > self.item_size_total as i64 {
                    return Err(MapError::InvalidItem);
                }
            }
        }
        if total != self.item_size_total as i64 {
            return Err(MapError::InvalidItem);
        }
        Ok(())
    }

    pub fn num_items(&self) -> usize {
        self.item_offsets.len()
    }

    pub fn num_data(&self) -> usize {
        self.data_offsets.len()
    }

    /// `CDataFileReader::GetType` (datafile.cpp:936-954), without the UUID/`ITEMTYPE_EX`
    /// indirection `GetInternalItemType` adds — every map item type this crate reads
    /// (`MAPITEMTYPE_VERSION`/`INFO`/`GROUP`/`LAYER`, all `< 8`) is far below `OFFSET_UUID`
    /// (`0x10000`), so that indirection's early-return fast path (datafile.cpp:896-899) is always
    /// taken for them — see `tools/ddnet-oracle/map2raw.cpp`'s header comment for the same
    /// argument on the C++ side.
    pub fn type_range(&self, type_: u16) -> (usize, usize) {
        for ty in &self.item_types {
            if ty.type_ == type_ {
                return (ty.start as usize, ty.num as usize);
            }
        }
        (0, 0)
    }

    /// `CDataFileReader::GetItem`/`GetItemSize` (datafile.cpp:919-934, 856-861). Returns an error
    /// only for an out-of-range index — every item this returns has already passed
    /// [`Self::validate_items`].
    pub fn item(&self, index: usize) -> Result<RawItem<'a>, MapError> {
        if index >= self.num_items() {
            return Err(MapError::InvalidItem);
        }
        let base = self.item_table_start + self.item_offsets[index] as usize;
        let type_and_id = u32::from_le_bytes(self.bytes[base..base + 4].try_into().unwrap());
        let size = i32_at(self.bytes, base + 4)? as usize;
        let payload = &self.bytes[base + ITEM_HEADER_SIZE as usize..base + ITEM_HEADER_SIZE as usize + size];
        Ok(RawItem {
            id: (type_and_id & 0xFFFF) as u16,
            payload,
        })
    }

    /// `CDataFileReader::GetData`/`GetDataSize` (datafile.cpp:789-801, 782-787): returns the
    /// logical (uncompressed) *length* of data blob `index`, without decompressing anything: for
    /// a version-4 file, the declared table value (datafile.cpp:716-719's
    /// `m_Info.m_pDataSizes`); for version 3, the on-disk length (data is stored uncompressed, so
    /// "logical" and "on-disk" are the same thing). Never allocates.
    ///
    /// A declared size of `0` is datafile.cpp:224-230's explicit "invalid, ignored" quirk for old
    /// v4 maps — this reports it as [`MapError::DataDecompressFailed`] rather than `Ok(0)`, since
    /// every caller of this crate that reaches for a blob's length wants to know "can I use this
    /// blob at all", and the answer there is no.
    pub fn data_len(&self, index: usize) -> Result<usize, MapError> {
        if index >= self.num_data() {
            return Err(MapError::DataDecompressFailed { index });
        }
        match &self.data_sizes {
            None => Ok(self.file_data_size(index) as usize),
            Some(sizes) => {
                if sizes[index] == 0 {
                    return Err(MapError::DataDecompressFailed { index });
                }
                Ok(sizes[index] as usize)
            }
        }
    }

    /// Reads at most `max_bytes` of data blob `index`'s logical (decompressed) content —
    /// reading directly for version 3 (already uncompressed), or zlib-inflating for version 4 —
    /// while still verifying that the blob's *actual* decompressed length matches its declared
    /// length **exactly**, mirroring `CDatafile::GetData`'s own zlib-level check
    /// (datafile.cpp:264-274: `Result != Z_OK || UncompressedSize != OriginalUncompressedSize`).
    ///
    /// This is this crate's primary bounded-allocation guarantee (task 1.4 acceptance criterion
    /// #2, review round 1 finding F2): peak extra memory for one call is `O(max_bytes)`, *never*
    /// `O(declared size)` — a data blob that declares an enormous logical size (whether honestly,
    /// via a huge legitimate map, or as a "zip bomb" probe) can inflate at most `max_bytes` worth
    /// of it into memory; the remainder (if any) is streamed through a small fixed-size buffer
    /// and only *counted*, never stored, so the length check above still runs to completion.
    /// Counting the remainder is itself bounded to `declared` bytes' worth of reads (the file's
    /// own `i32` header field, so already capped at ~2 GiB) before giving up and reporting a
    /// mismatch — still no unbounded work, only unbounded-*looking* input is rejected quickly.
    ///
    /// Unlike DDNet's own `GetData`, which returns `nullptr` on failure and lets the caller
    /// decide what that means (`CCollision::Init` just leaves the corresponding pointer null —
    /// see `crate::loader`'s doc comment on why this crate mirrors that for every physics layer
    /// except the game layer), this always returns a `Result`; converting an `Err` from this
    /// function into "layer absent" (rather than "whole map rejected") is the caller's job.
    pub fn data_bounded(&self, index: usize, max_bytes: usize) -> Result<Vec<u8>, MapError> {
        if index >= self.num_data() {
            return Err(MapError::DataDecompressFailed { index });
        }
        let file_size = self.file_data_size(index) as usize;
        let base = self.data_start + self.data_offsets[index] as usize;
        let raw = &self.bytes[base..base + file_size];

        match &self.data_sizes {
            None => {
                // Version 3: no compression at all — the on-disk bytes already are the logical
                // bytes, and `raw.len()` is bounded by the input the caller already holds (no
                // amplification is possible here regardless of `max_bytes`).
                let take = max_bytes.min(raw.len());
                Ok(raw[..take].to_vec())
            }
            Some(sizes) => {
                let declared = sizes[index] as u64;
                if declared == 0 {
                    return Err(MapError::DataDecompressFailed { index });
                }
                let keep = (max_bytes as u64).min(declared) as usize;
                let mut out = vec![0u8; keep];
                let mut decoder = flate2::read::ZlibDecoder::new(raw);
                let mut filled = 0usize;
                while filled < out.len() {
                    match decoder.read(&mut out[filled..]) {
                        Ok(0) => break,
                        Ok(n) => filled += n,
                        Err(_) => return Err(MapError::DataDecompressFailed { index }),
                    }
                }
                out.truncate(filled);

                // Drain (never storing) whatever the stream produces past `keep`, to confirm the
                // *true* total matches `declared` — this also correctly catches a blob that's
                // shorter than declared (the loop above already hit EOF, so this is one more
                // `read()` call that immediately returns `Ok(0)`) as well as one that's longer.
                let mut total = filled as u64;
                let mut discard = [0u8; 8192];
                loop {
                    if total > declared {
                        break; // already a proven mismatch; stop reading (bounds the work).
                    }
                    match decoder.read(&mut discard) {
                        Ok(0) => break,
                        Ok(n) => total += n as u64,
                        Err(_) => return Err(MapError::DataDecompressFailed { index }),
                    }
                }
                if total != declared {
                    return Err(MapError::DataDecompressFailed { index });
                }
                Ok(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::MapWriter as DatafileWriter;

    #[test]
    fn rejects_too_short_header() {
        assert_eq!(Datafile::parse(&[0u8; 10]).unwrap_err(), MapError::HeaderTruncated);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut w = DatafileWriter::new(4);
        w.add_item(0, 0, &[1, 0, 0, 0]);
        let bytes = w.finish();
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert_eq!(Datafile::parse(&bad).unwrap_err(), MapError::BadMagic);
    }

    #[test]
    fn accepts_atad_magic_alias() {
        let mut w = DatafileWriter::new(4);
        w.add_item(0, 0, &[1, 0, 0, 0]);
        let mut bytes = w.finish();
        bytes[0..4].copy_from_slice(b"ATAD");
        assert!(Datafile::parse(&bytes).is_ok());
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut w = DatafileWriter::new(5);
        w.add_item(0, 0, &[1, 0, 0, 0]);
        let bytes = w.finish();
        assert_eq!(
            Datafile::parse(&bytes).unwrap_err(),
            MapError::UnsupportedDatafileVersion(5)
        );
    }

    #[test]
    fn v3_data_is_stored_uncompressed() {
        let mut w = DatafileWriter::new(3);
        let idx = w.add_data_raw(b"hello world");
        w.add_item(0, 0, &(idx as i32).to_le_bytes());
        let bytes = w.finish();
        let df = Datafile::parse(&bytes).unwrap();
        assert_eq!(df.data_bounded(idx, 1024).unwrap(), b"hello world");
    }

    #[test]
    fn v4_data_is_zlib_compressed() {
        let mut w = DatafileWriter::new(4);
        let idx = w.add_data_compressed(b"hello world, compressed for v4");
        w.add_item(0, 0, &(idx as i32).to_le_bytes());
        let bytes = w.finish();
        let df = Datafile::parse(&bytes).unwrap();
        assert_eq!(df.data_bounded(idx, 1024).unwrap(), b"hello world, compressed for v4");
    }

    #[test]
    fn rejects_truncated_file() {
        let mut w = DatafileWriter::new(4);
        w.add_item(0, 0, &[1, 0, 0, 0]);
        let bytes = w.finish();
        let truncated = &bytes[..bytes.len() - 4];
        assert_eq!(Datafile::parse(truncated).unwrap_err(), MapError::SizeMismatch);
    }

    #[test]
    fn rejects_duplicate_item_id_within_type() {
        let mut w = DatafileWriter::new(4);
        w.add_item(0, 5, &[]);
        w.add_item(0, 5, &[]);
        let bytes = w.finish();
        assert_eq!(Datafile::parse(&bytes).unwrap_err(), MapError::InvalidItem);
    }

    #[test]
    fn allows_duplicate_ids_for_itemtype_ex() {
        let mut w = DatafileWriter::new(4);
        w.add_item(0xFFFF, 5, &[]);
        w.add_item(0xFFFF, 5, &[]);
        let bytes = w.finish();
        assert!(Datafile::parse(&bytes).is_ok());
    }

    #[test]
    fn rejects_data_index_out_of_range_read() {
        let mut w = DatafileWriter::new(4);
        w.add_item(0, 0, &[]);
        let bytes = w.finish();
        let df = Datafile::parse(&bytes).unwrap();
        assert!(df.data_bounded(0, 1024).is_err());
    }

    #[test]
    fn type_range_finds_matching_type() {
        let mut w = DatafileWriter::new(4);
        w.add_item(3, 0, &[]);
        w.add_item(3, 1, &[]);
        w.add_item(9, 0, &[]);
        let bytes = w.finish();
        let df = Datafile::parse(&bytes).unwrap();
        assert_eq!(df.type_range(3), (0, 2));
        assert_eq!(df.type_range(9), (2, 1));
        assert_eq!(df.type_range(42), (0, 0));
    }

    #[test]
    fn v4_zero_declared_size_fails_to_load_that_blob_without_rejecting_the_file() {
        let mut w = DatafileWriter::new(4);
        let idx = w.add_data_zero_size();
        w.add_item(0, 0, &(idx as i32).to_le_bytes());
        let bytes = w.finish();
        let df = Datafile::parse(&bytes).unwrap();
        assert!(df.data_bounded(idx, 1024).is_err());
    }

    #[test]
    fn v4_mismatched_declared_size_is_rejected() {
        let mut w = DatafileWriter::new(4);
        let idx = w.add_data_compressed_with_declared_size(b"abc", 999);
        w.add_item(0, 0, &(idx as i32).to_le_bytes());
        let bytes = w.finish();
        let df = Datafile::parse(&bytes).unwrap();
        assert!(df.data_bounded(idx, 1024).is_err());
    }

    #[test]
    fn data_bounded_never_allocates_more_than_max_bytes_for_a_huge_declared_size() {
        // Review round 1 finding F2: a data blob can declare an enormous logical size while its
        // actual compressed bytes are tiny — `data_bounded` must reject it (the true decompressed
        // size, `abc`'s 3 bytes, doesn't match the lie) without ever allocating anywhere near the
        // declared size.
        let mut w = DatafileWriter::new(4);
        let idx = w.add_data_compressed_with_declared_size(b"abc", (i32::MAX - 1) as u32);
        w.add_item(0, 0, &(idx as i32).to_le_bytes());
        let bytes = w.finish();
        let df = Datafile::parse(&bytes).unwrap();
        assert!(df.data_bounded(idx, 64).is_err());
    }

    #[test]
    fn data_bounded_returns_only_the_requested_prefix_of_a_larger_valid_blob() {
        let mut w = DatafileWriter::new(4);
        let idx = w.add_data_compressed(b"0123456789");
        w.add_item(0, 0, &(idx as i32).to_le_bytes());
        let bytes = w.finish();
        let df = Datafile::parse(&bytes).unwrap();
        assert_eq!(df.data_bounded(idx, 4).unwrap(), b"0123");
        assert_eq!(df.data_len(idx).unwrap(), 10);
    }

    #[test]
    fn data_bounded_detects_a_blob_longer_than_declared_even_when_the_prefix_alone_fits() {
        // The prefix we ask for (4 bytes) is satisfiable, but the *true* total (10 bytes) must
        // still be checked against `declared` (here deliberately wrong) — the mismatch has to be
        // caught by draining, not just by whether the requested prefix succeeded.
        let mut w = DatafileWriter::new(4);
        let idx = w.add_data_compressed_with_declared_size(b"0123456789", 4);
        w.add_item(0, 0, &(idx as i32).to_le_bytes());
        let bytes = w.finish();
        let df = Datafile::parse(&bytes).unwrap();
        assert!(df.data_bounded(idx, 4).is_err());
    }
}
