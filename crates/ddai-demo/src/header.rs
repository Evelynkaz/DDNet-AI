//! The `.demo` file's fixed-size prelude: `CDemoHeader`, `CTimelineMarkers`, and the optional
//! SHA256 extension that precedes the embedded map data — ported from `engine/shared/demo.{h,cpp}`
//! (DDNet 20.1, pinned rev `c9d208138f85755521f16a0096b6fe036c5c8698`), which carries the original
//! Teeworlds zlib-style notice:
//!
//!   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//!   If you are missing that file, acquire a complete release at teeworlds.com.
//!
//! This is an *altered* source version: rewritten in safe Rust, same byte layout, so it reads the
//! exact same files DDNet 20.1 itself reads/writes. See `docs/formats.md` for the annotated byte
//! layout and `crate::error::HeaderError` for the failure modes, each cited against the C++
//! function/lines it mirrors.

use crate::error::HeaderError;

/// `sizeof(CDemoHeader)` (`demo.h:34-46`): 7 (marker) + 1 (version) + 64 (netversion) + 64
/// (map name) + 4 (map size) + 4 (map crc) + 8 (type) + 4 (length) + 20 (timestamp) = 176.
pub const HEADER_SIZE: usize = 176;

/// `MAX_TIMELINE_MARKERS` (`demo.h:18`).
pub const MAX_TIMELINE_MARKERS: usize = 64;

/// This crate's own defensive cap on the whole file's size (not a DDNet-side limit — real demo
/// files this size would already be unusual): 512 MiB comfortably covers the longest known real
/// corpus demo (ChillerDragon's archive, up to ~74 minutes at 25 Hz) many times over, while still
/// bounding [`parse_prelude`]/[`crate::reader::TickIter`] against a hostile multi-gigabyte input.
pub const MAX_DEMO_FILE_SIZE: usize = 512 * 1024 * 1024;

/// `gs_aHeaderMarker` (`demo.h:21`): `"TWDEMO\0"`.
pub(crate) const HEADER_MARKER: [u8; 7] = *b"TWDEMO\0";

/// `gs_OldVersion` (`demo.cpp:31`): demos older than this are rejected outright.
pub const OLD_VERSION: u8 = 3;

/// `gs_Sha256Version` (`demo.cpp:32`): demos at or above this version carry the SHA256 extension.
pub const SHA256_VERSION: u8 = 6;

/// `gs_VersionTickCompression` (`demo.cpp:33`): demos at or above this version use the modern
/// (5-bit delta + explicit `CHUNKTICKFLAG_TICK_COMPRESSED` flag) tick encoding; older demos use
/// the legacy 6-bit-delta-if-nonzero encoding instead (see `crate::reader`).
pub const VERSION_TICK_COMPRESSION: u8 = 5;

/// `SHA256_EXTENSION` (`demo.cpp:26-28`): the UUID that marks the 32-byte SHA256 digest
/// immediately following it, for version >= [`SHA256_VERSION`] demos.
pub const SHA256_EXTENSION_UUID: [u8; 16] = [
    0x6b, 0xe6, 0xda, 0x4a, 0xce, 0xbd, 0x38, 0x0c, 0x9b, 0x5b, 0x12, 0x89, 0xc8, 0x42, 0xd7, 0x80,
];

/// A decoded `CDemoHeader` (`demo.h:34-46`). String fields are the bytes up to (not including)
/// the field's first NUL byte, decoded as UTF-8 (already validated by [`parse_prelude`] — see
/// `CDemoHeader::Valid`, `demo.cpp:39-47`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemoHeader {
    pub version: u8,
    pub netversion: String,
    pub map_name: String,
    pub map_size: u32,
    pub map_crc: u32,
    pub demo_type: String,
    /// `m_aLength`: the demo's length in seconds, as the recorder wrote it
    /// (`CDemoRecorder::Length`, `demo.h:64`: `(m_LastTickMarker - m_FirstTick) / SERVER_TICK_SPEED`).
    /// Zero for a demo that was never cleanly `Stop`ped with `EStopMode::KEEP_FILE`.
    pub length_seconds: i32,
    pub timestamp: String,
}

/// `CMapInfo` (`engine/map.h`), as populated by `CDemoPlayer::GetDemoInfo`
/// (`demo.cpp:1399-1402`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapInfo {
    /// Same string as [`DemoHeader::map_name`] (`demo.cpp:1399`: `str_copy(pMapInfo->m_aName,
    /// pDemoHeader->m_aMapName)`), kept as its own field to mirror the C++ `CMapInfo` shape.
    pub name: String,
    /// `Some` only for version >= [`SHA256_VERSION`] demos whose extension UUID matched
    /// [`SHA256_EXTENSION_UUID`] (`demo.cpp:1362-1397`) — real DDNet falls back to hashing the
    /// map bytes itself when this is `None` (`CDemoPlayer::ExtractMap`, `demo.cpp:940-950`);
    /// callers here can do the same via `ddai_map::load_map`'s own `sha256` field.
    pub sha256: Option<[u8; 32]>,
    pub crc: u32,
    pub size: u32,
}

/// Everything read from the file before the embedded map's own bytes: the header, the (not yet
/// range-validated against first/last tick — see the module docs below) timeline marker ticks,
/// and the map info. [`Prelude::map_offset`]/[`DemoHeader::map_size`] locate the embedded map's
/// raw bytes in the original buffer.
///
/// **Deliberate simplification vs. `CDemoPlayer::Load`:** real DDNet rejects the whole file
/// (`Stop("Invalid demo timeline marker")`, `demo.cpp:899-903`) if any timeline marker tick falls
/// outside `[FirstTick, LastTick]` — values only knowable after scanning the entire tick/chunk
/// stream (`ScanFile`, `demo.cpp:599-674`). Timeline markers are a cosmetic bookmark feature
/// (never consulted while decoding snapshots/messages, which is what this crate's exactness
/// testing — task 8.4b acceptance criterion 2 — checks), so this crate reads and clamps the
/// marker count to [`MAX_TIMELINE_MARKERS`] but does not perform that scan or that rejection;
/// [`crate::reader::Demo::ticks`] never consults [`Prelude::timeline_markers`] at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prelude {
    pub header: DemoHeader,
    pub timeline_markers: Vec<i32>,
    pub map: MapInfo,
    /// Byte offset of the embedded map's raw bytes within the buffer [`parse_prelude`] was
    /// given. The map itself is `data[map_offset..map_offset + header.map_size as usize]`.
    pub map_offset: usize,
}

/// Reads a byte's-worth of a fixed-size C-string field: the bytes up to (not including) the
/// first NUL, required to exist (`mem_has_null`) and to be valid UTF-8 (`str_utf8_check`) —
/// `CDemoHeader::Valid` (`demo.cpp:42-46`) applies this identically to all four string fields.
fn read_cstr_field(field: &'static str, bytes: &[u8]) -> Result<String, HeaderError> {
    let nul_pos = bytes
        .iter()
        .position(|&b| b == 0)
        .ok_or(HeaderError::BadHeaderString { field })?;
    let prefix = &bytes[..nul_pos];
    std::str::from_utf8(prefix)
        .map(str::to_owned)
        .map_err(|_| HeaderError::BadHeaderString { field })
}

/// The length half of [`MAX_DEMO_FILE_SIZE`]'s check, split out so it can be unit-tested without
/// actually allocating a half-gigabyte buffer.
fn check_file_size(len: usize) -> Result<(), HeaderError> {
    if len > MAX_DEMO_FILE_SIZE {
        return Err(HeaderError::FileTooLarge(MAX_DEMO_FILE_SIZE));
    }
    Ok(())
}

fn be_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Parses `CDemoHeader` (`demo.cpp:39-47` for validity, field layout `demo.h:34-46`) from the
/// first [`HEADER_SIZE`] bytes of `data`.
fn parse_header(data: &[u8]) -> Result<DemoHeader, HeaderError> {
    if data.len() < HEADER_SIZE {
        return Err(HeaderError::TooShortForHeader);
    }
    if data[0..7] != HEADER_MARKER {
        return Err(HeaderError::BadMagic);
    }
    let version = data[7];
    let netversion = read_cstr_field("netversion", &data[8..72])?;
    let map_name = read_cstr_field("map_name", &data[72..136])?;
    let map_size = be_u32(&data[136..140]);
    let map_crc = be_u32(&data[140..144]);
    let demo_type = read_cstr_field("demo_type", &data[144..152])?;
    // `m_aLength` is stored via `uint_to_bytes_be` (an unsigned helper) but is semantically a
    // signed second count (`CDemoRecorder::Length` returns `int`) — reinterpreting the same 4
    // big-endian bytes as `i32` reproduces exactly what `bytes_be_to_uint` followed by the
    // implicit `unsigned -> int` conversion DDNet itself does (two's complement, no UB on either
    // side for any bit pattern).
    let length_seconds = i32::from_be_bytes([data[152], data[153], data[154], data[155]]);
    let timestamp = read_cstr_field("timestamp", &data[156..176])?;

    if version < OLD_VERSION {
        return Err(HeaderError::UnsupportedVersion(version));
    }
    // `m_Sixup = str_startswith(m_aNetversion, "0.7")` (`demo.cpp:873`) — review round 1 finding
    // F10: this reader unconditionally assumes 0.6+DDNet wire semantics (`StaticSizes::
    // ddnet_06()`, `ddai_net::message`'s numbered-id tables), which sixup's renumbered ids and
    // `CSnapshotDeltaSixup` do not share; reject outright rather than silently decoding garbage.
    if netversion.starts_with("0.7") {
        return Err(HeaderError::UnsupportedSixup);
    }

    Ok(DemoHeader {
        version,
        netversion,
        map_name,
        map_size,
        map_crc,
        demo_type,
        length_seconds,
        timestamp,
    })
}

/// Parses the full prelude (header, timeline markers, optional SHA256 extension) and locates the
/// embedded map bytes — mirrors `CDemoPlayer::GetDemoInfo` (`demo.cpp:1315-1410`) plus the
/// timeline-marker read and map-offset bookkeeping from `CDemoPlayer::Load`
/// (`demo.cpp:891-905,875-877`). Never panics: every short read is a [`HeaderError`].
pub fn parse_prelude(data: &[u8]) -> Result<Prelude, HeaderError> {
    check_file_size(data.len())?;
    let header = parse_header(data)?;
    let mut pos = HEADER_SIZE;

    // `demo.cpp:1349-1359`: version > gs_OldVersion(3) reads `CTimelineMarkers` unconditionally.
    let mut timeline_markers = Vec::new();
    if header.version > OLD_VERSION {
        const TIMELINE_MARKERS_SIZE: usize = 4 + MAX_TIMELINE_MARKERS * 4;
        if data.len() < pos + TIMELINE_MARKERS_SIZE {
            return Err(HeaderError::TooShortForTimelineMarkers);
        }
        let num = be_u32(&data[pos..pos + 4]) as usize;
        let num = num.min(MAX_TIMELINE_MARKERS); // `std::clamp<int>(Num, 0, MAX_TIMELINE_MARKERS)`, `demo.cpp:895`
        pos += 4;
        for i in 0..num {
            let off = pos + i * 4;
            timeline_markers.push(i32::from_be_bytes([
                data[off],
                data[off + 1],
                data[off + 2],
                data[off + 3],
            ]));
        }
        pos += MAX_TIMELINE_MARKERS * 4;
    }

    // `demo.cpp:1362-1397`: version >= gs_Sha256Version(6) reads a 16-byte UUID; if it matches
    // SHA256_EXTENSION, a 32-byte digest follows; otherwise the 16 bytes are NOT part of the
    // extension after all (some other/future extension DDNet doesn't understand) and are left in
    // place for whatever comes next (real DDNet calls this "hoping" — `demo.cpp:1382-1396`).
    let mut sha256 = None;
    if header.version >= SHA256_VERSION {
        if data.len() < pos + 16 {
            return Err(HeaderError::TooShortForSha256Marker);
        }
        let uuid: [u8; 16] = data[pos..pos + 16].try_into().expect("slice is exactly 16 bytes");
        if uuid == SHA256_EXTENSION_UUID {
            pos += 16;
            if data.len() < pos + 32 {
                return Err(HeaderError::TooShortForSha256Digest);
            }
            let digest: [u8; 32] = data[pos..pos + 32].try_into().expect("slice is exactly 32 bytes");
            sha256 = Some(digest);
            pos += 32;
        }
        // else: leave `pos` where it is (rewound, matching `io_seek(..., -ExtensionUuidSize,
        // CURRENT)`, `demo.cpp:1387`) — those 16 bytes belong to whatever follows (the map, for
        // any actual DDNet 20.1-written file, since this crate never writes a fifth extension).
    }

    let map_offset = pos;
    let remaining = data.len().saturating_sub(map_offset) as u64;
    if header.map_size as u64 > remaining {
        return Err(HeaderError::MapSizeExceedsFile {
            declared: header.map_size,
            remaining,
        });
    }

    let map = MapInfo {
        name: header.map_name.clone(),
        sha256,
        crc: header.map_crc,
        size: header.map_size,
    };

    Ok(Prelude {
        header,
        timeline_markers,
        map,
        map_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::build_prelude_bytes;

    #[test]
    fn parses_v6_header_with_sha256() {
        let bytes = build_prelude_bytes(6, 16);
        let prelude = parse_prelude(&bytes).expect("well-formed v6 prelude");
        assert_eq!(prelude.header.version, 6);
        assert_eq!(prelude.header.netversion, "0.6");
        assert_eq!(prelude.header.map_name, "test");
        assert_eq!(prelude.header.map_size, 16);
        assert_eq!(prelude.header.map_crc, 0xDEAD_BEEF);
        assert_eq!(prelude.header.demo_type, "client");
        assert_eq!(prelude.map.sha256, Some([0u8; 32]));
        assert_eq!(prelude.map_offset, bytes.len() - 16);
        assert_eq!(&bytes[prelude.map_offset..], &[0xABu8; 16][..]);
    }

    #[test]
    fn parses_v4_header_without_sha256() {
        let bytes = build_prelude_bytes(4, 8);
        let prelude = parse_prelude(&bytes).expect("well-formed v4 prelude");
        assert_eq!(prelude.header.version, 4);
        assert_eq!(prelude.map.sha256, None);
        assert!(prelude.timeline_markers.is_empty());
    }

    #[test]
    fn parses_v3_header_without_timeline_markers() {
        let bytes = build_prelude_bytes(3, 0);
        let prelude = parse_prelude(&bytes).expect("well-formed v3 prelude");
        assert_eq!(prelude.header.version, 3);
        assert!(prelude.timeline_markers.is_empty());
        assert_eq!(prelude.map_offset, HEADER_SIZE);
    }

    #[test]
    fn rejects_sixup_netversion() {
        let mut bytes = build_prelude_bytes(6, 0);
        bytes[8..11].copy_from_slice(b"0.7"); // netversion field starts right after marker+version
        assert_eq!(parse_prelude(&bytes), Err(HeaderError::UnsupportedSixup));
    }

    #[test]
    fn rejects_oversized_file() {
        assert_eq!(check_file_size(MAX_DEMO_FILE_SIZE), Ok(()));
        assert_eq!(
            check_file_size(MAX_DEMO_FILE_SIZE + 1),
            Err(HeaderError::FileTooLarge(MAX_DEMO_FILE_SIZE))
        );
    }

    #[test]
    fn rejects_version_below_3() {
        let bytes = build_prelude_bytes(2, 0);
        assert_eq!(parse_header(&bytes), Err(HeaderError::UnsupportedVersion(2)));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = build_prelude_bytes(6, 0);
        bytes[0] = b'X';
        assert_eq!(parse_prelude(&bytes), Err(HeaderError::BadMagic));
    }

    #[test]
    fn rejects_too_short_for_header() {
        assert_eq!(parse_prelude(&[0u8; 10]), Err(HeaderError::TooShortForHeader));
        assert_eq!(parse_prelude(&[]), Err(HeaderError::TooShortForHeader));
    }

    #[test]
    fn rejects_missing_nul_terminator() {
        let mut bytes = build_prelude_bytes(6, 0);
        // Fill the whole netversion field with non-NUL bytes.
        for b in &mut bytes[8..72] {
            *b = b'a';
        }
        assert_eq!(
            parse_prelude(&bytes),
            Err(HeaderError::BadHeaderString { field: "netversion" })
        );
    }

    #[test]
    fn rejects_invalid_utf8_before_nul() {
        let mut bytes = build_prelude_bytes(6, 0);
        bytes[72] = 0xFF; // first byte of map_name: invalid UTF-8 lead byte
        bytes[73] = 0; // still NUL-terminated
        assert_eq!(
            parse_prelude(&bytes),
            Err(HeaderError::BadHeaderString { field: "map_name" })
        );
    }

    #[test]
    fn rejects_map_size_exceeding_file() {
        let mut bytes = build_prelude_bytes(6, 16);
        bytes.truncate(bytes.len() - 1); // one byte short of the declared map size
        assert_eq!(
            parse_prelude(&bytes),
            Err(HeaderError::MapSizeExceedsFile {
                declared: 16,
                remaining: 15
            })
        );
    }

    #[test]
    fn rejects_truncated_timeline_markers() {
        let bytes = build_prelude_bytes(6, 0);
        let truncated = &bytes[..HEADER_SIZE + 10];
        assert_eq!(parse_prelude(truncated), Err(HeaderError::TooShortForTimelineMarkers));
    }

    #[test]
    fn timeline_marker_count_is_clamped() {
        let mut bytes = build_prelude_bytes(6, 0);
        // Overwrite the marker count with something absurd; clamp should cap it at 64, not read
        // out of bounds or allocate anything unreasonable.
        bytes[HEADER_SIZE..HEADER_SIZE + 4].copy_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
        let prelude = parse_prelude(&bytes).expect("clamped marker count still parses");
        assert_eq!(prelude.timeline_markers.len(), MAX_TIMELINE_MARKERS);
    }

    #[test]
    fn unknown_sha256_extension_uuid_is_left_for_the_map() {
        let mut bytes = build_prelude_bytes(6, 16);
        // Corrupt one byte of the extension UUID so it no longer matches SHA256_EXTENSION_UUID —
        // real DDNet then treats those 16 bytes as belonging to whatever comes next.
        let uuid_pos = HEADER_SIZE + 4 + MAX_TIMELINE_MARKERS * 4;
        bytes[uuid_pos] ^= 0xFF;
        let prelude = parse_prelude(&bytes).expect("mismatched extension UUID still parses");
        assert_eq!(prelude.map.sha256, None);
        // The map offset now starts right at the (no longer recognised) UUID bytes, 48 bytes
        // earlier than the SHA256-matched case, and still yields exactly `map_size` bytes.
        assert_eq!(prelude.map_offset, uuid_pos);
        assert_eq!(bytes.len() - prelude.map_offset, 16 + 32 + 16); // uuid + digest + real map, all now "map data"
    }
}
