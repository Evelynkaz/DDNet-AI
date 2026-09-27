//! A tiny datafile/map writer used **only** by this crate's own tests (unit tests via `cfg(test)`
//! and the `tests/` integration tests via the `test-util` feature, which `Cargo.toml`'s
//! `[dev-dependencies]` self-reference enables). It exists so this crate's test coverage never
//! needs a real, third-party `.map` file committed to the repository (constraint: "No third-party
//! maps committed to git") — every fixture `load_map` is tested against here is built by this
//! module, field-for-field, from `docs/formats.md`-cited DDNet 20.1 struct layouts.
//!
//! This writer deliberately does **not** reuse `flate2`'s encoder settings or byte layout choices
//! from `crate::datafile` — it is an independent construction of the same format, so a bug in one
//! direction (e.g. an off-by-one in size accounting) is unlikely to also be present, and equally
//! wrong, on the other side.

#![cfg(any(test, feature = "test-util"))]

use flate2::Compression;
use flate2::write::ZlibEncoder;
use std::io::Write;

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

enum DataEntry {
    Raw(Vec<u8>),
    Compressed { declared: u32, bytes: Vec<u8> },
}

/// The byte-length "shape" of a `CMapItemLayerTilemap` item, per `mapitems.h`'s version history
/// (see `ddai-map`'s crate docs and `map.cpp:456-567`'s `UpgradeAndValidateTilesLayerItem`).
#[derive(Clone, Copy)]
pub enum TilemapShape {
    /// 60 bytes (tilemap item version must be `2`): no ddrace physics fields exist in the file at
    /// all. `map.cpp` defaults every one of them to `-1` *as long as the matching
    /// `TILESLAYERFLAG_*` bit isn't set* — set `flags` to a physics role with this shape to
    /// instead exercise the `TruncatedPhysicsField` rejection.
    V2Minimal,
    /// 80 bytes (tilemap item version `2`): the full legacy DDRace layout, tele/speedup/front/
    /// switch/tune fields present.
    V2Legacy,
    /// 72 bytes (tilemap item version `3` or `4`): Teeworlds' "has a name, no ddrace fields"
    /// shape — same defaulting behavior as `V2Minimal`, via the other truncation branch.
    V3Minimal,
    /// 92 bytes (tilemap item version `3` or `4`): the shape every current DDNet map uses.
    Full,
}

/// One tiles layer to add via [`MapWriter::add_tile_layer`].
pub struct TileLayerSpec<'a> {
    pub shape: TilemapShape,
    /// The tilemap item's own `m_Version` field (distinct from the datafile header version) —
    /// `2` for `TilemapShape::{V2Minimal,V2Legacy}`, `3` or `4` for `{V3Minimal,Full}`. `4`
    /// additionally triggers tile-skip unpacking for `CTile` layers (game/front/plain) — see
    /// [`encode_tile_skip`].
    pub item_version: i32,
    pub width: i32,
    pub height: i32,
    /// One of the `TILESLAYERFLAG_*` constants above, or `0` for a plain/decorative tiles layer.
    pub flags: u32,
    /// The layer's own tile records, already encoded (row-major `CTile`/`CTeleTile`/
    /// `CSpeedupTile`/`CSwitchTile`/`CTuneTile` bytes, or a tile-skip-packed `CTile` stream for
    /// `item_version >= 4` — see [`encode_tile_skip`]). Deliberately not length-checked by this
    /// builder: a caller can pass a wrong-sized slice to build an error-case fixture.
    pub data: &'a [u8],
}

/// RLE-packs `tiles` (logical `(index, flags)` pairs, one per cell, row-major) into the
/// tile-skip-compressed `CTile` stream `ExtractTiles` (map.cpp:278-313) expects when a tilemap
/// item's `m_Version >= 4`: consecutive identical `(index, flags)` cells collapse into one source
/// record with `m_Skip = run_length - 1` (runs longer than 256 split into multiple records, since
/// `m_Skip` is a single byte).
/// Encodes a `m_aName`-shaped field (3 `i32`s, 12 bytes) exactly like `StrToInts`
/// (game/gamecore.cpp:71-87) would — the exact inverse of `mapitems::decode_name`, reimplemented
/// independently here (not calling into that private function) so a bug in one direction is
/// unlikely to be mirrored, and equally wrong, in the other. Bytes past position 11 (this
/// function only ever fills 12) or past `name`'s own length are treated as the implicit NUL
/// terminator `StrToInts` would read past the end of a real C string — i.e. `0`, which then gets
/// XORed to `0x80` same as every other unfilled position. `name` must be at most 11 bytes (the
/// 12th is always the forced terminator); this is a test-only helper, so that's simply an
/// invariant its (small, hand-written) callers must uphold, not a validated `Result`.
pub fn encode_name(name: &str) -> [u8; 12] {
    let bytes = name.as_bytes();
    debug_assert!(bytes.len() <= 11, "test-fixture layer name must fit in 11 bytes");
    let mut ints = [0i32; 3];
    for (i, chunk) in ints.iter_mut().enumerate() {
        let mut buf = [0u8; 4];
        for (c, b) in buf.iter_mut().enumerate() {
            let pos = i * 4 + c;
            if pos < bytes.len() {
                *b = bytes[pos];
            }
        }
        let stored = [buf[0] ^ 0x80, buf[1] ^ 0x80, buf[2] ^ 0x80, buf[3] ^ 0x80];
        *chunk = u32::from_be_bytes(stored) as i32;
    }
    ints[2] &= 0xFFFFFF00u32 as i32; // IntsToStr/StrToInts always force the last byte to 0.
    let mut out = [0u8; 12];
    for i in 0..3 {
        out[i * 4..i * 4 + 4].copy_from_slice(&ints[i].to_le_bytes());
    }
    out
}

pub fn encode_tile_skip(tiles: &[(u8, u8)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < tiles.len() {
        let (index, flags) = tiles[i];
        let mut run = 1usize;
        while i + run < tiles.len() && tiles[i + run] == (index, flags) && run < 256 {
            run += 1;
        }
        out.push(index);
        out.push(flags);
        out.push((run - 1) as u8);
        out.push(0); // m_MustBe0
        i += run;
    }
    out
}

pub struct MapWriter {
    version: i32,
    items: Vec<(u16, u16, Vec<u8>)>,
    datas: Vec<DataEntry>,
    layer_count: i32,
    next_group_id: u16,
    next_layer_id: u16,
}

impl MapWriter {
    pub fn new(version: i32) -> Self {
        MapWriter {
            version,
            items: Vec::new(),
            datas: Vec::new(),
            layer_count: 0,
            next_group_id: 0,
            next_layer_id: 0,
        }
    }

    // --- low-level (also exercised directly by `crate::datafile`'s own unit tests) -------------

    pub fn add_item(&mut self, type_: u16, id: u16, payload: &[u8]) -> usize {
        self.items.push((type_, id, payload.to_vec()));
        self.items.len() - 1
    }

    pub fn add_data_raw(&mut self, bytes: &[u8]) -> usize {
        self.datas.push(DataEntry::Raw(bytes.to_vec()));
        self.datas.len() - 1
    }

    pub fn add_data_compressed(&mut self, bytes: &[u8]) -> usize {
        self.add_data_compressed_with_declared_size(bytes, bytes.len() as u32)
    }

    pub fn add_data_compressed_with_declared_size(&mut self, bytes: &[u8], declared: u32) -> usize {
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(bytes).expect("in-memory zlib encode");
        let compressed = enc.finish().expect("in-memory zlib finish");
        self.datas.push(DataEntry::Compressed {
            declared,
            bytes: compressed,
        });
        self.datas.len() - 1
    }

    /// A v4 data blob whose declared uncompressed size is `0` — datafile.cpp:224-230's tolerated
    /// "invalid, ignored" quirk (the file loads; only *this* blob fails to load later).
    pub fn add_data_zero_size(&mut self) -> usize {
        self.add_data_compressed_with_declared_size(b"x", 0)
    }

    /// Adds `bytes` the way this writer's own datafile `version` stores data (raw for v3, zlib
    /// for v4) — what every higher-level helper below uses.
    pub fn add_data_auto(&mut self, bytes: &[u8]) -> usize {
        if self.version == 4 {
            self.add_data_compressed(bytes)
        } else {
            self.add_data_raw(bytes)
        }
    }

    // --- map-item level -------------------------------------------------------------------------

    pub fn add_version_item(&mut self, map_version: i32) {
        self.add_item(MAPITEMTYPE_VERSION, 0, &map_version.to_le_bytes());
    }

    /// `CMapItemInfoSettings` (mapitems.h:369-373): `author`/`map_version`/`credits`/`license`
    /// are each written as a NUL-terminated data blob if `Some`, or left as `-1` if `None` —
    /// exactly what `GetDataString(-1)` (`""`) vs. a real string distinguishes on the read side.
    pub fn add_info_item(
        &mut self,
        author: Option<&str>,
        map_version_str: Option<&str>,
        credits: Option<&str>,
        license: Option<&str>,
        settings: &[&str],
    ) {
        let mut string_index = |s: Option<&str>| -> i32 {
            match s {
                None => -1,
                Some(s) => {
                    let mut bytes = s.as_bytes().to_vec();
                    bytes.push(0);
                    self.add_data_auto(&bytes) as i32
                }
            }
        };
        let author_idx = string_index(author);
        let version_idx = string_index(map_version_str);
        let credits_idx = string_index(credits);
        let license_idx = string_index(license);
        let settings_idx = if settings.is_empty() {
            -1
        } else {
            let mut blob = Vec::new();
            for s in settings {
                blob.extend_from_slice(s.as_bytes());
                blob.push(0);
            }
            self.add_data_auto(&blob) as i32
        };

        let mut payload = Vec::with_capacity(24);
        payload.extend_from_slice(&1i32.to_le_bytes()); // m_Version
        payload.extend_from_slice(&author_idx.to_le_bytes());
        payload.extend_from_slice(&version_idx.to_le_bytes());
        payload.extend_from_slice(&credits_idx.to_le_bytes());
        payload.extend_from_slice(&license_idx.to_le_bytes());
        payload.extend_from_slice(&settings_idx.to_le_bytes());
        self.add_item(MAPITEMTYPE_INFO, 0, &payload);
    }

    /// Adds one tiles layer item (byte shape per `spec.shape`) plus its data blob, and returns
    /// this layer's index *within the LAYER item type* (0-based, insertion order) — the value
    /// [`Self::add_group`] needs for `m_StartLayer`. The name field (versions 3/4 only) is
    /// `encode_name("")` — valid UTF-8, so this never trips the name check on its own; use
    /// [`Self::add_tile_layer_with_name_bytes`] to inject a specific (possibly invalid) name.
    pub fn add_tile_layer(&mut self, spec: &TileLayerSpec) -> i32 {
        self.add_tile_layer_with_name_bytes(spec, encode_name(""))
    }

    /// Like [`Self::add_tile_layer`], but with explicit control over the 12 raw name bytes (for
    /// `TilemapShape::{V3Minimal,Full}` — the shapes that have a name field at all) — used by
    /// this crate's `InvalidLayerName` regression tests to inject a byte sequence that doesn't
    /// decode to valid UTF-8, which [`encode_name`] (by construction, since it only ever encodes
    /// an already-valid `&str`) cannot produce.
    pub fn add_tile_layer_with_name_bytes(&mut self, spec: &TileLayerSpec, name_bytes: [u8; 12]) -> i32 {
        let data_index = self.add_data_auto(spec.data) as i32;
        let role_data_index = |flag: u32| -> i32 { if spec.flags == flag { data_index } else { -1 } };
        // `m_Data` is the GAME role's data index too (mapitems.h has no separate `m_Game` field —
        // `ValidateAndUnpackTilesLayerData`, map.cpp:574-579, reads `m_Data` for both
        // `TILESLAYERFLAG_GAME` and the "no physics flag" plain/decorative case).
        let plain_data = if spec.flags == 0 || spec.flags == TILESLAYERFLAG_GAME {
            data_index
        } else {
            -1
        };

        let mut p = Vec::with_capacity(92);
        p.extend_from_slice(&0i32.to_le_bytes()); // m_Layer.m_Version (unused)
        p.extend_from_slice(&LAYERTYPE_TILES.to_le_bytes());
        p.extend_from_slice(&0i32.to_le_bytes()); // m_Layer.m_Flags
        p.extend_from_slice(&spec.item_version.to_le_bytes());
        p.extend_from_slice(&spec.width.to_le_bytes());
        p.extend_from_slice(&spec.height.to_le_bytes());
        p.extend_from_slice(&spec.flags.to_le_bytes());
        for _ in 0..4 {
            p.extend_from_slice(&255i32.to_le_bytes()); // CColor r,g,b,a
        }
        p.extend_from_slice(&(-1i32).to_le_bytes()); // m_ColorEnv
        p.extend_from_slice(&0i32.to_le_bytes()); // m_ColorEnvOffset
        p.extend_from_slice(&(-1i32).to_le_bytes()); // m_Image
        p.extend_from_slice(&plain_data.to_le_bytes()); // m_Data
        debug_assert_eq!(p.len(), 60);

        match spec.shape {
            TilemapShape::V2Minimal => {}
            TilemapShape::V2Legacy => {
                p.extend_from_slice(&role_data_index(TILESLAYERFLAG_TELE).to_le_bytes());
                p.extend_from_slice(&role_data_index(TILESLAYERFLAG_SPEEDUP).to_le_bytes());
                p.extend_from_slice(&role_data_index(TILESLAYERFLAG_FRONT).to_le_bytes());
                p.extend_from_slice(&role_data_index(TILESLAYERFLAG_SWITCH).to_le_bytes());
                p.extend_from_slice(&role_data_index(TILESLAYERFLAG_TUNE).to_le_bytes());
            }
            TilemapShape::V3Minimal => {
                p.extend_from_slice(&name_bytes);
            }
            TilemapShape::Full => {
                p.extend_from_slice(&name_bytes);
                p.extend_from_slice(&role_data_index(TILESLAYERFLAG_TELE).to_le_bytes());
                p.extend_from_slice(&role_data_index(TILESLAYERFLAG_SPEEDUP).to_le_bytes());
                p.extend_from_slice(&role_data_index(TILESLAYERFLAG_FRONT).to_le_bytes());
                p.extend_from_slice(&role_data_index(TILESLAYERFLAG_SWITCH).to_le_bytes());
                p.extend_from_slice(&role_data_index(TILESLAYERFLAG_TUNE).to_le_bytes());
            }
        }

        let id = self.next_layer_id;
        self.next_layer_id += 1;
        self.add_item(MAPITEMTYPE_LAYER, id, &p);
        let index_in_layers = self.layer_count;
        self.layer_count += 1;
        index_in_layers
    }

    /// `CMapItemGroup_v1` (28 bytes) — sufficient for `CMap::Load`'s own group size check
    /// (`>= sizeof(CMapItemGroup_v1)`); this writer never needs the clip/name fields
    /// `CMapItemGroup` adds, since they don't affect physics-layer selection.
    pub fn add_group(&mut self, start_layer: i32, num_layers: i32) {
        let mut p = Vec::with_capacity(28);
        p.extend_from_slice(&3i32.to_le_bytes()); // m_Version
        p.extend_from_slice(&0i32.to_le_bytes()); // m_OffsetX
        p.extend_from_slice(&0i32.to_le_bytes()); // m_OffsetY
        p.extend_from_slice(&100i32.to_le_bytes()); // m_ParallaxX
        p.extend_from_slice(&100i32.to_le_bytes()); // m_ParallaxY
        p.extend_from_slice(&start_layer.to_le_bytes());
        p.extend_from_slice(&num_layers.to_le_bytes());
        let id = self.next_group_id;
        self.next_group_id += 1;
        self.add_item(MAPITEMTYPE_GROUP, id, &p);
    }

    /// Convenience for the common case (one group holding every layer added so far).
    pub fn add_single_group_with_all_layers(&mut self) {
        self.add_group(0, self.layer_count);
    }

    /// Assembles the final datafile bytes: header, item-type table, item/data offset tables,
    /// (v4) declared-uncompressed-size table, items (grouped by type, ascending — matching
    /// `CDataFileWriter::Finish`, `datafile.cpp:1228-1368`), then raw data blobs in add order.
    pub fn finish(self) -> Vec<u8> {
        use std::collections::BTreeMap;

        let mut by_type: BTreeMap<u16, Vec<(u16, Vec<u8>)>> = BTreeMap::new();
        for (type_, id, payload) in self.items {
            by_type.entry(type_).or_default().push((id, payload));
        }

        let num_item_types = by_type.len() as i32;
        let num_items: i32 = by_type.values().map(|v| v.len() as i32).sum();
        let num_raw_data = self.datas.len() as i32;

        let mut item_type_table = Vec::new();
        let mut item_offsets = Vec::new();
        let mut items_bytes = Vec::new();
        let mut running_item_offset: i32 = 0;
        let mut running_item_count: i32 = 0;
        for (&type_, entries) in &by_type {
            item_type_table.extend_from_slice(&(type_ as i32).to_le_bytes());
            item_type_table.extend_from_slice(&running_item_count.to_le_bytes());
            item_type_table.extend_from_slice(&(entries.len() as i32).to_le_bytes());
            for (id, payload) in entries {
                item_offsets.extend_from_slice(&running_item_offset.to_le_bytes());
                let type_and_id = ((type_ as u32) << 16) | (*id as u32);
                items_bytes.extend_from_slice(&type_and_id.to_le_bytes());
                items_bytes.extend_from_slice(&(payload.len() as i32).to_le_bytes());
                items_bytes.extend_from_slice(payload);
                running_item_offset += 8 + payload.len() as i32;
                running_item_count += 1;
            }
        }
        let item_size = running_item_offset;

        let mut data_offsets = Vec::new();
        let mut data_sizes = Vec::new();
        let mut data_bytes = Vec::new();
        let mut running_data_offset: i32 = 0;
        for d in &self.datas {
            data_offsets.extend_from_slice(&running_data_offset.to_le_bytes());
            let bytes: &[u8] = match d {
                DataEntry::Raw(b) => b,
                DataEntry::Compressed { declared, bytes } => {
                    data_sizes.extend_from_slice(&(*declared as i32).to_le_bytes());
                    bytes
                }
            };
            data_bytes.extend_from_slice(bytes);
            running_data_offset += bytes.len() as i32;
        }
        let data_size = running_data_offset;

        let mut size: i64 = num_item_types as i64 * 12 + num_items as i64 * 4 + num_raw_data as i64 * 4;
        if self.version == 4 {
            size += num_raw_data as i64 * 4;
        }
        size += item_size as i64;
        // datafile.cpp:1261-1279 (`CDataFileWriter::Finish`): `m_Size = FileSize - SizeOffset`
        // (includes the raw data blobs), `m_Swaplen = SwapSize - SizeOffset` (does not) — both
        // relative to the header's own 16-byte `SizeOffset()` prefix (magic+version+size+swaplen
        // themselves, `datafile.h`'s `CDatafileHeader::SizeOffset`), which is *not* the same as
        // this crate's 36-byte `HEADER_SIZE` (`sizeof(CDatafileHeader)`) — `36 - 16 = 20`.
        let m_size = 20i64 + size + data_size as i64;
        let m_swaplen = 20i64 + size;

        let mut out = Vec::new();
        out.extend_from_slice(b"DATA");
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(m_size as i32).to_le_bytes());
        out.extend_from_slice(&(m_swaplen as i32).to_le_bytes());
        out.extend_from_slice(&num_item_types.to_le_bytes());
        out.extend_from_slice(&num_items.to_le_bytes());
        out.extend_from_slice(&num_raw_data.to_le_bytes());
        out.extend_from_slice(&item_size.to_le_bytes());
        out.extend_from_slice(&data_size.to_le_bytes());
        debug_assert_eq!(out.len(), 36);

        out.extend_from_slice(&item_type_table);
        out.extend_from_slice(&item_offsets);
        out.extend_from_slice(&data_offsets);
        if self.version == 4 {
            out.extend_from_slice(&data_sizes);
        }
        out.extend_from_slice(&items_bytes);
        out.extend_from_slice(&data_bytes);
        out
    }
}

/// Builds a complete, minimal-but-valid v4 map: one group, a game layer (all-zero `TILE_AIR`
/// tiles) of `width`x`height`, plus whichever `extra` physics layers the caller adds via the
/// returned [`MapWriter`] before calling `.finish()`. Most `loader`/integration tests start here.
pub fn game_layer_data(width: i32, height: i32) -> Vec<u8> {
    vec![0u8; (width * height * 4) as usize]
}
