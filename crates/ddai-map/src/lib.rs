//! Reads real DDNet `.map` files (datafile format v3/v4) into
//! [`ddai_physics::map::MapData`] — our own safe reader (no `unsafe`, no C dependencies), because
//! forking libtw2-datafile/map would bring 58 `unsafe` blocks and outdated dependencies along
//! with it (decision D-026). Correctness is proven against DDNet's own C++ loader: a small C++
//! `map2raw` tool (`tools/ddnet-oracle/map2raw.cpp`) compiled from the DDNet 20.1 sources dumps a
//! map's physics layers into the rawmap v1 format `ddai-trace` already speaks
//! (`docs/formats.md` §1); this crate's `load_map` must produce byte-identical `MapData` for
//! every map in the task's corpus (`tools/ddnet-oracle/map-corpus-check.sh`).
//!
//! ## What this crate covers, and what it deliberately doesn't
//!
//! `MapData` (game/front/tele/speedup/switch/tune tile grids + Settings strings) is exactly what
//! `ddai_physics`/`ddai-trace` need — the collision/gameplay layer built on top is a later task's
//! job (see `docs/PLAN.md` §1.3+). `MapInfo` (author/version/credits/license) is read for
//! completeness (the task's acceptance criteria ask for it in [`LoadedMap`]) but is **not**
//! cross-checked against the C++ oracle: no real DDNet *server* code path reads those fields at
//! all — only `CEditorMap::Load` (`game/editor/mapitems/map_io.cpp`) does, and `map2raw.cpp`
//! deliberately mirrors the server's `CMap::Load`/`CLayers::Init`/`CGameContext::LoadMapSettings`,
//! not the editor. `MapInfo` is verified by this crate's own unit tests instead (see
//! `crate::testutil`).
//!
//! Two DDNet loader behaviors this crate does **not** replicate bit-for-bit, both cited in detail
//! where they matter (`crate::loader`'s doc comment, `crate::datafile`'s `MAX_ALLOC_BYTES`/
//! `MAX_TILE_COUNT`): (1) a physics layer whose *own data blob* (not its item) fails to
//! decompress makes the whole map fail to load here, where real DDNet's `CCollision::Init` just
//! leaves that one layer's pointer `nullptr` and carries on — no editor-produced map exercises
//! this; (2) two bounded-allocation caps beyond DDNet's own checks, so a crafted tiny input can
//! never force a multi-gigabyte allocation (the task's fuzz/robustness requirement) — neither cap
//! comes close to tripping on any real map in the task's corpus.

mod datafile;
mod error;
mod loader;
mod mapitems;
pub mod scene;
#[cfg(any(test, feature = "test-util"))]
pub mod testutil;

pub use error::MapError;
pub use scene::{VisualScene, extract_visual_scene, extract_visual_scene_within};

use ddai_physics::map::MapData;
use sha2::{Digest, Sha256};

/// `CMapItemInfoSettings`'s author/version/credits/license strings (mapitems.h:359-373),
/// resolved to owned `String`s. `None` means the item had no data index for that field (`-1`);
/// see this crate's top-level docs for why these are not cross-checked against the C++ oracle.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MapInfo {
    pub author: Option<String>,
    pub version: Option<String>,
    pub credits: Option<String>,
    pub license: Option<String>,
}

/// Everything [`load_map`] produces: the plain-data map itself, the input's own hashes/size (so
/// a caller doesn't need to re-hash the bytes it just handed us), [`MapInfo`], and the map's
/// Settings strings (also reachable as `data.settings` — kept here too since the task's
/// acceptance criteria name it as its own field; both are always equal, populated from the same
/// read in `crate::loader`).
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedMap {
    pub data: MapData,
    pub sha256: [u8; 32],
    pub crc32: u32,
    pub size: usize,
    pub info: MapInfo,
    pub settings: Vec<String>,
}

/// Reads a DDNet `.map` file (datafile format v3 or v4) into a [`LoadedMap`]. Never panics: any
/// malformed input — truncated, corrupt, adversarially crafted, or simply not a datafile at all —
/// is reported as a [`MapError`], not a crash (see `crate::datafile`/`crate::loader`'s doc
/// comments for the bounded-allocation guarantees that back this up, and the crate's
/// `robustness` test module for how it's exercised).
/// A hard cap on how many bytes of a `MapInfo` string (author/version/credits/license) this
/// crate will read (review round 1 finding F2 — these are Rust-only display strings, see this
/// module's top-level docs, so *truncating* an oversized one is harmless, unlike Settings, where
/// a truncated blob risks a garbled command; see `loader::SETTINGS_MAX_BYTES`'s doc comment for
/// that distinction). No real map's info strings come remotely close to 64 KiB.
const MAPINFO_MAX_BYTES: usize = 64 * 1024;

pub fn load_map(bytes: &[u8]) -> Result<LoadedMap, MapError> {
    let df = datafile::Datafile::parse(bytes)?;
    let loaded = loader::load(&df)?;

    let resolve = |index: Option<i32>| -> Option<String> {
        let index = index?;
        match df.data_bounded(index as usize, MAPINFO_MAX_BYTES) {
            Ok(mut bytes) => {
                // `CDataFileReader::GetDataString` (datafile.cpp:803-824) requires the blob to
                // end with exactly one NUL and contain no earlier one; this crate is more
                // permissive (strip a single trailing NUL if present, decode losslessly
                // otherwise) rather than silently falling back to an empty string the way
                // `CEditorMap::Load`'s `ReadStringInfo` does on any such mismatch — since this
                // field is Rust-only (see this crate's top-level docs), there is no C++ behavior
                // to match bit-for-bit here, only "never panic, never silently discard real
                // author/license text a map actually has".
                if bytes.last() == Some(&0) {
                    bytes.pop();
                }
                Some(String::from_utf8_lossy(&bytes).into_owned())
            }
            Err(_) => None,
        }
    };

    let info = MapInfo {
        author: resolve(loaded.info_author),
        version: resolve(loaded.info_map_version),
        credits: resolve(loaded.info_credits),
        license: resolve(loaded.info_license),
    };

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let sha256: [u8; 32] = hasher.finalize().into();
    let crc32 = crc32fast::hash(bytes);

    Ok(LoadedMap {
        data: loaded.map,
        sha256,
        crc32,
        size: bytes.len(),
        info,
        settings: loaded.settings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{
        MapWriter, TILESLAYERFLAG_FRONT, TILESLAYERFLAG_GAME, TILESLAYERFLAG_SPEEDUP, TILESLAYERFLAG_SWITCH,
        TILESLAYERFLAG_TELE, TILESLAYERFLAG_TUNE, TileLayerSpec, TilemapShape, encode_tile_skip, game_layer_data,
    };

    fn minimal_game_layer(w: &mut MapWriter, width: i32, height: i32) {
        w.add_version_item(1);
        w.add_info_item(None, None, None, None, &[]);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width,
            height,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(width, height),
        });
        w.add_single_group_with_all_layers();
    }

    #[test]
    fn loads_a_minimal_game_only_map_v4() {
        let mut w = MapWriter::new(4);
        minimal_game_layer(&mut w, 4, 3);
        let bytes = w.finish();
        let loaded = load_map(&bytes).unwrap();
        assert_eq!(loaded.data.width, 4);
        assert_eq!(loaded.data.height, 3);
        assert_eq!(loaded.data.game.len(), 12);
        assert!(loaded.data.front.is_none());
        assert_eq!(loaded.size, bytes.len());
        assert_eq!(loaded.crc32, crc32fast::hash(&bytes));
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let expected: [u8; 32] = hasher.finalize().into();
        assert_eq!(loaded.sha256, expected);
    }

    #[test]
    fn loads_a_minimal_game_only_map_v3() {
        let mut w = MapWriter::new(3);
        minimal_game_layer(&mut w, 5, 5);
        let bytes = w.finish();
        let loaded = load_map(&bytes).unwrap();
        assert_eq!(loaded.data.width, 5);
        assert_eq!(loaded.data.height, 5);
    }

    #[test]
    fn loads_every_physics_layer_kind() {
        let (w_, h_) = (3, 3);
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: w_,
            height: h_,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(w_, h_),
        });
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: w_,
            height: h_,
            flags: TILESLAYERFLAG_FRONT,
            data: &game_layer_data(w_, h_),
        });
        let tele_data = vec![7u8, 26]; // one TeleTile record isn't enough for 9 cells — pad below.
        let mut tele_full = Vec::new();
        for _ in 0..9 {
            tele_full.extend_from_slice(&tele_data);
        }
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: w_,
            height: h_,
            flags: TILESLAYERFLAG_TELE,
            data: &tele_full,
        });
        let mut speedup_full = Vec::new();
        for _ in 0..9 {
            speedup_full.extend_from_slice(&[5u8, 0, 28, 0, (-90i16).to_le_bytes()[0], (-90i16).to_le_bytes()[1]]);
        }
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: w_,
            height: h_,
            flags: TILESLAYERFLAG_SPEEDUP,
            data: &speedup_full,
        });
        let mut switch_full = Vec::new();
        for _ in 0..9 {
            switch_full.extend_from_slice(&[1u8, 24, 0, 3]);
        }
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: w_,
            height: h_,
            flags: TILESLAYERFLAG_SWITCH,
            data: &switch_full,
        });
        let mut tune_full = Vec::new();
        for _ in 0..9 {
            tune_full.extend_from_slice(&[1u8, 68]);
        }
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: w_,
            height: h_,
            flags: TILESLAYERFLAG_TUNE,
            data: &tune_full,
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        let loaded = load_map(&bytes).unwrap();
        assert!(loaded.data.front.is_some());
        let tele = loaded.data.tele.unwrap();
        assert_eq!(tele.len(), 9);
        assert_eq!((tele[0].number, tele[0].kind), (7, 26));
        let speedup = loaded.data.speedup.unwrap();
        assert_eq!(speedup[0].force, 5);
        assert_eq!(speedup[0].angle, -90);
        let switch = loaded.data.switch.unwrap();
        assert_eq!((switch[0].number, switch[0].kind, switch[0].delay), (1, 24, 3));
        let tune = loaded.data.tune.unwrap();
        assert_eq!((tune[0].number, tune[0].kind), (1, 68));
    }

    #[test]
    fn reads_settings_and_map_info() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_info_item(
            Some("Author Name"),
            Some("v1.0"),
            Some("Credits here"),
            Some("CC0"),
            &["sv_foo 1", "sv_bar 2"],
        );
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 3,
            height: 3,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(3, 3),
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        let loaded = load_map(&bytes).unwrap();
        assert_eq!(loaded.settings, vec!["sv_foo 1".to_string(), "sv_bar 2".to_string()]);
        assert_eq!(loaded.data.settings, loaded.settings);
        assert_eq!(loaded.info.author.as_deref(), Some("Author Name"));
        assert_eq!(loaded.info.version.as_deref(), Some("v1.0"));
        assert_eq!(loaded.info.credits.as_deref(), Some("Credits here"));
        assert_eq!(loaded.info.license.as_deref(), Some("CC0"));
    }

    #[test]
    fn missing_info_item_yields_none_and_empty_settings() {
        let mut w = MapWriter::new(4);
        minimal_game_layer(&mut w, 2, 2);
        let bytes = w.finish();
        let loaded = load_map(&bytes).unwrap();
        assert_eq!(loaded.info, MapInfo::default());
        assert!(loaded.settings.is_empty());
    }

    #[test]
    fn tile_skip_encoded_game_layer_decodes_correctly() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        // 2x2: two air tiles then two solid tiles, row-major — encoded as two skip records.
        let logical = [(0u8, 0u8), (0, 0), (1, 0), (1, 0)];
        let packed = encode_tile_skip(&logical);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 4, // triggers ExtractTiles-equivalent unpacking.
            width: 2,
            height: 2,
            flags: TILESLAYERFLAG_GAME,
            data: &packed,
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        let loaded = load_map(&bytes).unwrap();
        let indices: Vec<u8> = loaded.data.game.iter().map(|t| t.index).collect();
        assert_eq!(indices, vec![0, 0, 1, 1]);
    }

    #[test]
    fn multiple_game_layers_last_one_wins() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        let mut first = game_layer_data(2, 2);
        first[0] = 9; // distinguishable from the second layer's all-zero tiles.
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 2,
            height: 2,
            flags: TILESLAYERFLAG_GAME,
            data: &first,
        });
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 2,
            height: 2,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(2, 2),
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        let loaded = load_map(&bytes).unwrap();
        // The second (last) game layer wins — its all-zero tiles, not the first's index-9 tile.
        assert_eq!(loaded.data.game[0].index, 0);
    }

    #[test]
    fn missing_game_layer_is_rejected() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 2,
            height: 2,
            flags: 0, // decorative, not game
            data: &game_layer_data(2, 2),
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert_eq!(load_map(&bytes), Err(MapError::NoGameLayer));
    }

    #[test]
    fn missing_version_item_is_rejected() {
        let mut w = MapWriter::new(4);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 2,
            height: 2,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(2, 2),
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert_eq!(load_map(&bytes), Err(MapError::MissingOrUnsupportedVersionItem));
    }

    #[test]
    fn wrong_version_item_value_is_rejected() {
        let mut w = MapWriter::new(4);
        w.add_version_item(2);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 2,
            height: 2,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(2, 2),
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert_eq!(load_map(&bytes), Err(MapError::MissingOrUnsupportedVersionItem));
    }

    #[test]
    fn width_below_two_is_rejected() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 1,
            height: 5,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(1, 5),
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert_eq!(
            load_map(&bytes),
            Err(MapError::InvalidLayerDimensions { group: 0, layer: 0 })
        );
    }

    #[test]
    fn decorative_layer_with_bad_dimensions_still_rejects_the_whole_map() {
        // Covers map.cpp:317-329 running for *every* tiles layer, not just physics ones.
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 3,
            height: 3,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(3, 3),
        });
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 1, // invalid, but this layer is purely decorative (flags = 0)
            height: 1,
            flags: 0,
            data: &[0u8; 4],
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert!(matches!(load_map(&bytes), Err(MapError::InvalidLayerDimensions { .. })));
    }

    #[test]
    fn physics_layer_smaller_than_game_is_rejected() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 4,
            height: 4,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(4, 4),
        });
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 2,
            height: 2,
            flags: TILESLAYERFLAG_TELE,
            data: &[0u8; 2 * 2 * 2],
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert_eq!(
            load_map(&bytes),
            Err(MapError::PhysicsLayerSmallerThanGame { group: 0, layer: 1 })
        );
    }

    #[test]
    fn physics_layer_larger_than_game_is_truncated_to_game_size() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 2,
            height: 2,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(2, 2),
        });
        // Tele layer declares 3x2 = 6 cells, larger than the game layer's 2x2 = 4.
        let mut tele_data = Vec::new();
        for i in 0..6u8 {
            tele_data.push(i + 1); // number
            tele_data.push(26); // TILE_TELEIN
        }
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 3,
            height: 2,
            flags: TILESLAYERFLAG_TELE,
            data: &tele_data,
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        let loaded = load_map(&bytes).unwrap();
        let tele = loaded.data.tele.unwrap();
        assert_eq!(tele.len(), 4); // truncated to the game layer's 2x2
        assert_eq!(tele[0].number, 1);
        assert_eq!(tele[3].number, 4);
    }

    #[test]
    fn duplicate_data_index_is_rejected() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        let data_idx = w.add_data_compressed(&game_layer_data(2, 2));
        // Two layer items pointing at the very same data index.
        let mk_layer = |flags: u32, data_idx: usize| -> Vec<u8> {
            let mut p = Vec::new();
            p.extend_from_slice(&0i32.to_le_bytes());
            p.extend_from_slice(&crate::testutil::LAYERTYPE_TILES.to_le_bytes());
            p.extend_from_slice(&0i32.to_le_bytes());
            p.extend_from_slice(&3i32.to_le_bytes()); // tilemap item version
            p.extend_from_slice(&2i32.to_le_bytes()); // width
            p.extend_from_slice(&2i32.to_le_bytes()); // height
            p.extend_from_slice(&flags.to_le_bytes());
            for _ in 0..4 {
                p.extend_from_slice(&255i32.to_le_bytes());
            }
            p.extend_from_slice(&(-1i32).to_le_bytes());
            p.extend_from_slice(&0i32.to_le_bytes());
            p.extend_from_slice(&(-1i32).to_le_bytes());
            p.extend_from_slice(&(data_idx as i32).to_le_bytes()); // m_Data
            p.extend_from_slice(&crate::testutil::encode_name("")); // name
            p.extend_from_slice(&(-1i32).to_le_bytes()); // tele
            p.extend_from_slice(&(-1i32).to_le_bytes()); // speedup
            p.extend_from_slice(&(-1i32).to_le_bytes()); // front
            p.extend_from_slice(&(-1i32).to_le_bytes()); // switch
            p.extend_from_slice(&(-1i32).to_le_bytes()); // tune
            p
        };
        let l0 = mk_layer(TILESLAYERFLAG_GAME, data_idx);
        w.add_item(crate::testutil::MAPITEMTYPE_LAYER, 0, &l0);
        let l1 = mk_layer(0, data_idx); // decorative layer reusing the SAME data index
        w.add_item(crate::testutil::MAPITEMTYPE_LAYER, 1, &l1);
        w.add_group(0, 2);
        let bytes = w.finish();
        assert!(matches!(load_map(&bytes), Err(MapError::DataIndexReused { .. })));
    }

    // --- review round 1 regression tests ---------------------------------------------------

    /// F1: a decorative layer's name must be valid UTF-8 (map.cpp:331-346's `EnsureValidName`,
    /// via `IntsToStr`/`str_utf8_check`) — reproduced directly against a hand-crafted map with
    /// `map2raw` (see the build report); `0xFF` is not a valid UTF-8 leading byte under any
    /// continuation-byte count.
    #[test]
    fn f1_decorative_layer_with_invalid_utf8_name_is_rejected() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 3,
            height: 3,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(3, 3),
        });
        let mut bad_name = [0u8; 12];
        bad_name[0] = 0xFF;
        w.add_tile_layer_with_name_bytes(
            &TileLayerSpec {
                shape: TilemapShape::Full,
                item_version: 3,
                width: 3,
                height: 3,
                flags: 0,
                data: &[0u8; 4 * 9],
            },
            bad_name,
        );
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert!(matches!(load_map(&bytes), Err(MapError::InvalidLayerName { .. })));
    }

    /// F1: name validity applies to *every* tiles layer, not only decorative ones (map.cpp's
    /// `EnsureValidName` runs before the physics-flag branch selects which *other* checks apply).
    #[test]
    fn f1_game_layer_with_invalid_utf8_name_is_rejected() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        let mut bad_name = [0u8; 12];
        bad_name[0] = 0xFF;
        w.add_tile_layer_with_name_bytes(
            &TileLayerSpec {
                shape: TilemapShape::Full,
                item_version: 3,
                width: 3,
                height: 3,
                flags: TILESLAYERFLAG_GAME,
                data: &game_layer_data(3, 3),
            },
            bad_name,
        );
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert!(matches!(load_map(&bytes), Err(MapError::InvalidLayerName { .. })));
    }

    /// F1: a version-2 tilemap item has no name field in the file at all (map.cpp:507-508 forces
    /// the encoding of `""` before `EnsureValidName` runs) — a `V2Minimal` layer (60 bytes, no
    /// name bytes on disk to begin with) must never trip the name check.
    #[test]
    fn f1_version_2_tilemap_item_has_no_name_field_and_is_never_checked() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::V2Minimal,
            item_version: 2,
            width: 3,
            height: 3,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(3, 3),
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert!(load_map(&bytes).is_ok());
    }

    /// F1: color range (0..=255 per component) applies to decorative layers only — a
    /// physics-flagged layer's out-of-range color is silently reset by DDNet
    /// (`EnsureDefaultColor`, log-only), never rejected.
    #[test]
    fn f1_decorative_layer_with_out_of_range_color_is_rejected() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 3,
            height: 3,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(3, 3),
        });
        let deco_data_idx = w.add_data_compressed(&[0u8; 4 * 9]);
        let mut p = Vec::new();
        p.extend_from_slice(&0i32.to_le_bytes());
        p.extend_from_slice(&crate::testutil::LAYERTYPE_TILES.to_le_bytes());
        p.extend_from_slice(&0i32.to_le_bytes());
        p.extend_from_slice(&3i32.to_le_bytes()); // tilemap item version
        p.extend_from_slice(&3i32.to_le_bytes()); // width
        p.extend_from_slice(&3i32.to_le_bytes()); // height
        p.extend_from_slice(&0i32.to_le_bytes()); // flags (decorative)
        p.extend_from_slice(&256i32.to_le_bytes()); // color.r — out of range
        p.extend_from_slice(&255i32.to_le_bytes());
        p.extend_from_slice(&255i32.to_le_bytes());
        p.extend_from_slice(&255i32.to_le_bytes());
        p.extend_from_slice(&(-1i32).to_le_bytes()); // color env
        p.extend_from_slice(&0i32.to_le_bytes());
        p.extend_from_slice(&(-1i32).to_le_bytes()); // image
        p.extend_from_slice(&(deco_data_idx as i32).to_le_bytes()); // m_Data
        p.extend_from_slice(&crate::testutil::encode_name(""));
        for _ in 0..5 {
            p.extend_from_slice(&(-1i32).to_le_bytes());
        }
        w.add_item(crate::testutil::MAPITEMTYPE_LAYER, 1, &p);
        w.add_group(0, 2);
        let bytes = w.finish();
        assert!(matches!(load_map(&bytes), Err(MapError::InvalidLayerColor { .. })));
    }

    /// F3: a bad front layer (non-zero `m_Skip`, map.cpp:695-709) makes the layer *absent*, not
    /// the whole map fail — matches `CCollision::Init`'s lazy, null-tolerant `GetData` (unlike the
    /// GAME layer, which `CMap::Load` itself force-loads eagerly and *does* reject the map for).
    /// Reproduced directly against a hand-crafted map with `map2raw` (`front_badpad.map`, see the
    /// build report): `CMap::Load` accepts it; only this crate's own (pre-fix) code rejected it.
    #[test]
    fn f3_front_layer_with_bad_padding_is_absent_not_a_load_failure() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 4,
            height: 4,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(4, 4),
        });
        let mut bad_front = game_layer_data(4, 4);
        bad_front[2] = 1; // m_Skip on the first tile, non-zero outside a skip-encoded layer.
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 4,
            height: 4,
            flags: TILESLAYERFLAG_FRONT,
            data: &bad_front,
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        let loaded = load_map(&bytes).expect("bad front data must not fail the whole map");
        assert!(loaded.data.front.is_none());
    }

    /// F3: same as above, for tele/speedup/switch/tune — an `item_version >= 4` (tile-skip) on
    /// any of these (never a valid combination — map.cpp:657-663 permits skip encoding only for
    /// game/decorative `CTile` layers) leaves the layer absent, not a load failure.
    #[test]
    fn f3_speedup_layer_with_disallowed_tile_skip_version_is_absent_not_a_load_failure() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 3,
            height: 3,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(3, 3),
        });
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 4, // disallowed for speedup
            width: 3,
            height: 3,
            flags: TILESLAYERFLAG_SPEEDUP,
            data: &[0u8; 3 * 3 * 6],
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        let loaded = load_map(&bytes).expect("disallowed tile-skip version must not fail the whole map");
        assert!(loaded.data.speedup.is_none());
    }

    /// F4: a tile-skip-encoded game layer with 1-3 trailing bytes past the last full record is
    /// still accepted — map.cpp:651's `SavedTilesSize = Size / sizeof(CTile)` floor-divides, so a
    /// short trailing remainder is never counted as "leftover" at all. Reproduced directly against
    /// a hand-crafted map with `map2raw` (`skip_trailing.map`, see the build report).
    #[test]
    fn f4_tile_skip_with_one_to_three_trailing_bytes_is_still_accepted() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        let logical = [(0u8, 0u8); 16]; // 4x4, one uniform run.
        let mut packed = encode_tile_skip(&logical);
        packed.extend_from_slice(&[0xAB, 0xCD]); // 2 trailing bytes, not a full record.
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 4,
            width: 4,
            height: 4,
            flags: TILESLAYERFLAG_GAME,
            data: &packed,
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert!(load_map(&bytes).is_ok());
    }

    /// F4 (converse): a *full extra record* left over after the destination is filled is still a
    /// real "too much tile layer data" error (map.cpp:306-312) — the fix must not have loosened
    /// this into "anything past the minimum is ignored".
    #[test]
    fn f4_tile_skip_with_a_full_extra_leftover_record_is_still_rejected() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        let mut logical = vec![(0u8, 0u8); 16]; // 4x4, exactly fills the grid.
        logical.push((1, 0)); // one more full source record than needed.
        let packed = encode_tile_skip(&logical);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 4,
            width: 4,
            height: 4,
            flags: TILESLAYERFLAG_GAME,
            data: &packed,
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert!(matches!(
            load_map(&bytes),
            Err(MapError::PhysicsLayerDataFailed { layer: "game" })
        ));
    }

    /// F5: this crate's own `MAX_TILE_COUNT` cap must apply only to layers actually decoded — a
    /// huge *decorative* layer (never decoded) must be accepted even though its tile count
    /// exceeds that cap; the real `width*height`-fits-in-`i32` check (a genuine DDNet check) still
    /// applies to it and is satisfied here (100M fits comfortably under `i32::MAX`).
    #[test]
    fn f5_huge_decorative_layer_never_decoded_is_accepted() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 3,
            height: 3,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(3, 3),
        });
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 10_000,
            height: 10_000, // 100M cells: over this crate's 32M `MAX_TILE_COUNT`, under i32::MAX.
            flags: 0,       // decorative — never decoded, so the cap must not apply.
            data: &[0u8; 4],
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert!(load_map(&bytes).is_ok());
    }

    /// F5 (converse): the GAME layer itself is *always* decoded, so if *it* exceeds
    /// `MAX_TILE_COUNT`, that must still fail (as a hard, whole-map error, matching the GAME
    /// layer's eager-load semantics) — the fix must not have disabled the cap outright.
    #[test]
    fn f5_game_layer_itself_over_max_tile_count_is_rejected() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 10_000,
            height: 10_000,
            flags: TILESLAYERFLAG_GAME,
            data: &[0u8; 4],
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        assert!(matches!(
            load_map(&bytes),
            Err(MapError::PhysicsLayerDataFailed { layer: "game" })
        ));
    }

    /// F6: `CDatafile::GetData` caches one decompressed-data processor per data *index*, shared
    /// by every consumer — if a tiles layer's data happens to share the exact index the Settings
    /// blob lives at, DDNet's observable result is "no settings" (see `loader::read_info`'s doc
    /// comment for the full mechanism, and the build report for a reproduction against a mutated
    /// real map). This crate can't replicate the underlying per-index cache, so it special-cases
    /// exactly this: any index a tiles layer also claims is never read as Settings.
    #[test]
    fn f6_settings_index_shared_with_a_tiles_layer_is_treated_as_no_settings() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        // The settings blob ("sv_x 1\0", 7 bytes) is added *first*, so it gets data index 0.
        w.add_info_item(None, None, None, None, &["sv_x 1"]);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 3,
            height: 3,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(3, 3),
        });
        // A decorative layer whose `m_Data` is set to that SAME index 0 — built by hand (not
        // `add_tile_layer`, which always allocates a *fresh* data blob) specifically so the two
        // indices collide. Uses layer id 1 (`add_tile_layer` above already used id 0).
        let mut p = Vec::new();
        p.extend_from_slice(&0i32.to_le_bytes());
        p.extend_from_slice(&crate::testutil::LAYERTYPE_TILES.to_le_bytes());
        p.extend_from_slice(&0i32.to_le_bytes());
        p.extend_from_slice(&3i32.to_le_bytes());
        p.extend_from_slice(&3i32.to_le_bytes());
        p.extend_from_slice(&3i32.to_le_bytes());
        p.extend_from_slice(&0i32.to_le_bytes()); // flags (decorative)
        for _ in 0..4 {
            p.extend_from_slice(&255i32.to_le_bytes());
        }
        p.extend_from_slice(&(-1i32).to_le_bytes());
        p.extend_from_slice(&0i32.to_le_bytes());
        p.extend_from_slice(&(-1i32).to_le_bytes());
        p.extend_from_slice(&0i32.to_le_bytes()); // m_Data = 0, same as the settings blob
        p.extend_from_slice(&crate::testutil::encode_name(""));
        for _ in 0..5 {
            p.extend_from_slice(&(-1i32).to_le_bytes());
        }
        w.add_item(crate::testutil::MAPITEMTYPE_LAYER, 1, &p);
        w.add_group(0, 2);
        let bytes = w.finish();
        let loaded = load_map(&bytes).unwrap();
        assert!(loaded.settings.is_empty());
    }

    /// F2: a Settings blob whose *declared* logical size is over this crate's `SETTINGS_MAX_BYTES`
    /// cap is treated as "no settings" outright (not truncated — a partial `sv_*` command is
    /// worse than none), without this crate ever allocating anywhere near that declared size (see
    /// `crates/ddai-map/src/datafile.rs`'s `data_bounded` tests, and the build report's measured
    /// peak-RSS numbers for the equivalent hand-crafted `.map` files run through the CLI).
    #[test]
    fn f2_settings_blob_over_the_byte_cap_is_treated_as_no_settings() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 2,
            height: 2,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(2, 2),
        });
        w.add_single_group_with_all_layers();
        // A settings blob that *declares* a huge logical size but is a tiny, honestly-compressed
        // blob underneath — like `declared_2g.map`'s scenario, just for the Settings field
        // specifically instead of a tele layer.
        let settings_idx = w.add_data_compressed_with_declared_size(b"sv_x 1\0", (i32::MAX - 1) as u32);
        let info_payload = {
            let mut p = Vec::new();
            p.extend_from_slice(&1i32.to_le_bytes());
            for _ in 0..4 {
                p.extend_from_slice(&(-1i32).to_le_bytes());
            }
            p.extend_from_slice(&(settings_idx as i32).to_le_bytes());
            p
        };
        w.add_item(crate::testutil::MAPITEMTYPE_INFO, 0, &info_payload);
        let bytes = w.finish();
        // The mismatched declared size alone would already fail `data_bounded`'s verification;
        // this asserts the *outcome* callers actually observe (empty settings, whole map still
        // loads) regardless of which specific check inside `read_info` caught it.
        let loaded = load_map(&bytes).unwrap();
        assert!(loaded.settings.is_empty());
    }

    /// F2: a `MapInfo` string (author/version/credits/license) whose declared logical size is
    /// larger than `MAPINFO_MAX_BYTES` is truncated to the cap, not rejected — unlike Settings,
    /// these are Rust-only display strings (see this crate's top-level docs), so losing the tail
    /// of an implausibly long author name is harmless.
    #[test]
    fn f2_mapinfo_string_over_the_byte_cap_is_truncated_not_rejected() {
        let long_author = "A".repeat(MAPINFO_MAX_BYTES + 100);
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_info_item(Some(&long_author), None, None, None, &[]);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 2,
            height: 2,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(2, 2),
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        let loaded = load_map(&bytes).unwrap();
        let author = loaded
            .info
            .author
            .expect("author must still be present, just truncated");
        assert_eq!(author.len(), MAPINFO_MAX_BYTES);
        assert!(long_author.starts_with(&author));
    }
}
