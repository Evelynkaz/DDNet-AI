//! Byte-offset constants and small parsing helpers for the map item payloads
//! (`engine/shared/datafile.h`'s items are opaque blobs; DDNet 20.1's `game/mapitems.h` defines
//! what's actually inside each one). Every constant/offset here is cited against that header.
//!
//! All these classes are plain sequences of little-endian `i32` fields (`mapitems.h`'s own
//! `static_assert`s pin their exact byte sizes — see the comments below), so reading them is just
//! indexed `i32` extraction, never a `#[repr(C)]` cast — this crate has no `unsafe` at all.

use crate::error::MapError;

pub const MAPITEMTYPE_VERSION: u16 = 0;
pub const MAPITEMTYPE_INFO: u16 = 1;
pub const MAPITEMTYPE_GROUP: u16 = 4;
pub const MAPITEMTYPE_LAYER: u16 = 5;

pub const LAYERTYPE_TILES: i32 = 2;

pub const TILESLAYERFLAG_GAME: u32 = 1 << 0;
pub const TILESLAYERFLAG_TELE: u32 = 1 << 1;
pub const TILESLAYERFLAG_SPEEDUP: u32 = 1 << 2;
pub const TILESLAYERFLAG_FRONT: u32 = 1 << 3;
pub const TILESLAYERFLAG_SWITCH: u32 = 1 << 4;
pub const TILESLAYERFLAG_TUNE: u32 = 1 << 5;
pub const ALL_PHYSICS_FLAGS: u32 = TILESLAYERFLAG_GAME
    | TILESLAYERFLAG_TELE
    | TILESLAYERFLAG_SPEEDUP
    | TILESLAYERFLAG_FRONT
    | TILESLAYERFLAG_SWITCH
    | TILESLAYERFLAG_TUNE;

/// `sizeof(CMapItemGroup_v1)` (mapitems.h:394-405): 7 `i32` fields — this crate never reads
/// `CMapItemGroup`'s clip/name extension (mapitems.h:407-417), only what map.cpp:113-126 itself
/// requires to find a group's layer range.
pub const SIZEOF_GROUP_V1: usize = 28;
/// `sizeof(CMapItemLayer)` (mapitems.h:419-425): version, type, flags.
pub const SIZEOF_LAYER: usize = 12;
/// `sizeof(CMapItemLayerTilemap_v2)` (mapitems.h:428-444, `static_assert`ed at mapitems.h:479).
/// A complete `CMapItemLayerTilemap_v2Legacy` (mapitems.h:448-456, asserted at mapitems.h:480,
/// 80 bytes) or `CMapItemLayerTilemap` (mapitems.h:467-475, asserted at mapitems.h:482, 92 bytes)
/// is this plus the 5 DDRace physics fields (each bounds-checked individually in
/// [`parse_tilemap`], not by a length floor named here — a real item can be a truncated prefix
/// of either).
pub const SIZEOF_TILEMAP_V2: usize = 60;
/// `sizeof(CMapItemLayerTilemap_v3Teeworlds)` (mapitems.h:460-464, asserted at mapitems.h:481).
pub const SIZEOF_TILEMAP_V3TEEWORLDS: usize = 72;
pub fn i32_at(bytes: &[u8], offset: usize) -> Result<i32, MapError> {
    let b = bytes.get(offset..offset + 4).ok_or(MapError::InvalidItem)?;
    Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Decodes a `m_aName`-shaped field (3 `i32`s, 12 bytes) back into raw name bytes, mirroring
/// `IntsToStr` (game/gamecore.cpp:89-115) exactly: within each `i32`, bytes are extracted
/// MSB-first (`(v>>24)&0xff`, `(v>>16)&0xff`, ...) — the *opposite* order from how the `i32`
/// itself is stored little-endian in the file — and each extracted byte is XORed with `0x80`
/// (equivalent to `byte - 128` on an 8-bit value, which is what `IntsToStr` actually computes;
/// see this function's doc for why the two are the same bit pattern). The last byte is always
/// forced to `0`, exactly like `IntsToStr`'s unconditional `pStr[StrIndex - 1] = '\0'` — a
/// version-2 item's name is never read this way at all (see [`parse_tilemap`]'s caller).
fn decode_name(payload: &[u8], offset: usize) -> [u8; 12] {
    let mut out = [0u8; 12];
    for i in 0..3 {
        // `unwrap_or(0)`: this is only ever called after the caller has already bounds-checked
        // that these 12 bytes are present (`SIZEOF_TILEMAP_V3TEEWORLDS` covers them); the
        // fallback exists only so this function itself can never panic if that invariant is
        // ever violated by a future change.
        let v = i32_at(payload, offset + i * 4).unwrap_or(0);
        let bytes = [
            (((v >> 24) & 0xff) as u8) ^ 0x80,
            (((v >> 16) & 0xff) as u8) ^ 0x80,
            (((v >> 8) & 0xff) as u8) ^ 0x80,
            ((v & 0xff) as u8) ^ 0x80,
        ];
        out[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
    out[11] = 0;
    out
}

/// `str_utf8_check` (base/str.cpp:1224-1231) on the C-string `IntsToStr` would have produced:
/// valid UTF-8 up to (not including) the first `NUL` byte — which always exists in `name` since
/// [`decode_name`] forces the last byte to `0`. Rust's own UTF-8 validation implements the same
/// Unicode conformance rules (rejects overlong encodings, surrogates, and out-of-range code
/// points) as DDNet's WHATWG-spec-based decoder, so this is a faithful, independent check, not a
/// port of DDNet's decoder loop.
fn name_is_valid_utf8(name: &[u8; 12]) -> bool {
    let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
    std::str::from_utf8(&name[..end]).is_ok()
}

/// `CMapItemGroup_v1` (mapitems.h:394-405): only `m_StartLayer`/`m_NumLayers` — this crate has no
/// use for offset/parallax/clip (rendering only).
pub struct GroupV1 {
    pub start_layer: i32,
    pub num_layers: i32,
}

pub fn parse_group_v1(payload: &[u8]) -> Result<GroupV1, MapError> {
    Ok(GroupV1 {
        start_layer: i32_at(payload, 20)?,
        num_layers: i32_at(payload, 24)?,
    })
}

/// The base `CMapItemLayer` (mapitems.h:419-425): every map item of type `MAPITEMTYPE_LAYER`
/// starts with this, regardless of what kind of layer it turns out to be.
pub fn layer_type(payload: &[u8]) -> Result<i32, MapError> {
    i32_at(payload, 4)
}

/// A `CMapItemLayerTilemap`, upgraded to the latest (92-byte) field set — mirrors
/// `CMap::UpgradeAndValidateTilesLayerItem` (map.cpp:456-567): fields the on-disk item is too
/// short to contain default to `-1` (matching `EnsureUnsetPhysicsData`'s "not used by this
/// version" meaning), *unless* the layer's own `flags` claims that role, which is a hard
/// [`MapError::TruncatedPhysicsField`] rejection instead (map.cpp:481-499).
pub struct Tilemap {
    pub item_version: i32,
    pub width: i32,
    pub height: i32,
    pub flags: u32,
    pub data: i32,
    pub tele: i32,
    pub speedup: i32,
    pub front: i32,
    pub switch: i32,
    pub tune: i32,
}

pub fn parse_tilemap(payload: &[u8], group: usize, layer: usize) -> Result<Tilemap, MapError> {
    if payload.len() < SIZEOF_TILEMAP_V2 {
        return Err(MapError::InvalidTilemapItem { group, layer });
    }
    let item_version = i32_at(payload, 12)?;
    if !(2..=4).contains(&item_version) {
        return Err(MapError::InvalidTilemapItem { group, layer });
    }
    let width = i32_at(payload, 16)?;
    let height = i32_at(payload, 20)?;
    let flags = i32_at(payload, 24)? as u32;
    let data = i32_at(payload, 56)?;

    // Which byte range covers the 5 DDRace physics fields, and whether names are expected to be
    // present, depends on the item's own `m_Version` (2 vs. 3/4) — mapitems.h:448-475,
    // map.cpp:501-564. Each field is still bounds-checked individually below (a version-2 item
    // may contain a full `CMapItemLayerTilemap_v2Legacy`, or only some prefix of it; the "full"
    // sizes named here — `SIZEOF_TILEMAP_V2LEGACY`/`SIZEOF_TILEMAP_FULL` — are what a *complete*
    // item of that shape looks like, not a length floor this function enforces up front).
    let physics_base = if item_version == 2 {
        SIZEOF_TILEMAP_V2
    } else {
        if payload.len() < SIZEOF_TILEMAP_V3TEEWORLDS {
            // map.cpp:527-533: for version 3/4, truncation below the "has a full name" size is a
            // hard error — only the *ddrace* fields beyond that may be a truncated prefix.
            return Err(MapError::InvalidTilemapItem { group, layer });
        }
        SIZEOF_TILEMAP_V3TEEWORLDS
    };

    let physics_field = |field_offset_from_base: usize, flag: u32| -> Result<i32, MapError> {
        let field_offset = physics_base + field_offset_from_base;
        if payload.len() < field_offset + 4 {
            if flags & flag != 0 {
                return Err(MapError::TruncatedPhysicsField { group, layer });
            }
            return Ok(-1);
        }
        i32_at(payload, field_offset)
    };

    let tele = physics_field(0, TILESLAYERFLAG_TELE)?;
    let speedup = physics_field(4, TILESLAYERFLAG_SPEEDUP)?;
    let front = physics_field(8, TILESLAYERFLAG_FRONT)?;
    let switch = physics_field(12, TILESLAYERFLAG_SWITCH)?;
    let tune = physics_field(16, TILESLAYERFLAG_TUNE)?;

    if (flags & ALL_PHYSICS_FLAGS).count_ones() > 1 {
        return Err(MapError::MultiplePhysicsFlags { group, layer });
    }
    if width < 2 || height < 2 {
        return Err(MapError::InvalidLayerDimensions { group, layer });
    }

    // map.cpp:331-346 (`EnsureValidName`): runs for *every* tiles layer regardless of role.
    // Version-2 items have no name field in the file at all — map.cpp:507-508 unconditionally
    // sets it to the encoding of `""` before this would run, which is trivially valid, so there
    // is nothing to check for them. For version 3/4, the name always occupies the 12 bytes right
    // after the v2 base (offset `SIZEOF_TILEMAP_V2`), which is always fully present here (the
    // truncation check above already required the whole `SIZEOF_TILEMAP_V3TEEWORLDS` prefix).
    if item_version != 2 {
        let name = decode_name(payload, SIZEOF_TILEMAP_V2);
        if !name_is_valid_utf8(&name) {
            return Err(MapError::InvalidLayerName { group, layer });
        }
    }

    // map.cpp:421-437: color range, decorative (no physics flag) layers only — a physics-flagged
    // layer's color is silently reset to the default instead (`EnsureDefaultColor`, log-only,
    // never rejects), so this check must not apply to them.
    if flags & ALL_PHYSICS_FLAGS == 0 {
        for offset in [28usize, 32, 36, 40] {
            let component = i32_at(payload, offset)?;
            if !(0..=255).contains(&component) {
                return Err(MapError::InvalidLayerColor { group, layer });
            }
        }
    }

    Ok(Tilemap {
        item_version,
        width,
        height,
        flags,
        data,
        tele,
        speedup,
        front,
        switch,
        tune,
    })
}

/// `CMapItemInfoSettings` (mapitems.h:359-373). `-1` for any data index means "absent" (matches
/// `CDataFileReader::GetDataString(-1) == ""` for the string fields, and "no settings" for
/// `settings`).
pub struct InfoSettings {
    pub author: i32,
    pub map_version: i32,
    pub credits: i32,
    pub license: i32,
    pub settings: i32,
}

pub fn parse_info_settings(payload: &[u8]) -> Option<InfoSettings> {
    // Fields beyond what's actually present default to `-1` ("absent") rather than being an
    // error — matches how a truncated `MAPITEMTYPE_INFO` item is treated (never rejects the
    // whole map; see `crate::loader`'s doc comment on why this differs from `CEditorMap::Load`,
    // which is the only real DDNet code that reads author/version/credits/license at all).
    let read = |offset: usize| -> i32 {
        if payload.len() < offset + 4 {
            -1
        } else {
            i32_at(payload, offset).unwrap_or(-1)
        }
    };
    if payload.len() < 4 {
        return None;
    }
    Some(InfoSettings {
        author: read(4),
        map_version: read(8),
        credits: read(12),
        license: read(16),
        settings: read(20),
    })
}
