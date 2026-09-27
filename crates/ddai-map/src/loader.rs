//! `CMap::Load` (engine/shared/map.cpp) + `CLayers::Init(pMap, false, false)` (game/layers.cpp),
//! turning a validated [`crate::datafile::Datafile`] into a [`ddai_physics::map::MapData`].
//!
//! Two passes over every group/layer, exactly mirroring `CMap::Load`'s own two loops
//! (map.cpp:111-164 then map.cpp:172-187): the first determines every physics role's *winning*
//! (last, in group-then-layer order — see `layers.cpp:22-77`'s lack of a `break`) tiles layer
//! while running the checks `UpgradeAndValidateTilesLayerItem` does for *every* tiles layer, not
//! just physics ones (a decorative layer with bad dimensions still sinks the whole map); the
//! second re-walks the same layers now that the game layer's dimensions are known, running the
//! checks `ValidateAndUnpackTilesLayerData` does for *every* tiles layer (data-index bounds and
//! *global* uniqueness, `width*height` overflow, and — physics layers only — "at least as many
//! tiles as the game layer").
//!
//! What this module does **not** replicate, deliberately (see the crate's top-level docs and the
//! build report for task 1.4): decoding a layer's *content* (tile-skip unpacking, zero-padding
//! checks) for anything other than the six layers this crate actually keeps (the winning game/
//! front/tele/speedup/switch/tune) — DDNet itself only forces this early for the game layer
//! (map.cpp:190-195) and leaves every other layer's content check lazy (triggered only if some
//! caller — `CCollision::Init`, or here, this loader — actually asks for that data); a decorative
//! layer's or a *superseded* physics layer's data is therefore never even decompressed, matching
//! both DDNet's own behavior and what `tools/ddnet-oracle/map2raw.cpp` (which only ever reads the
//! *winning* layers too) can be compared against byte-for-byte.

use crate::datafile::Datafile;
use crate::error::MapError;
use crate::mapitems::{self, LAYERTYPE_TILES, MAPITEMTYPE_GROUP, MAPITEMTYPE_LAYER, MAPITEMTYPE_VERSION};
use ddai_physics::map::{MapData, SpeedupTile, SwitchTile, TeleTile, Tile, TuneTile};

/// A bounded-allocation cap this crate adds on top of DDNet's own `width*height` overflow check
/// (which only rejects when the product doesn't fit in an `i32` — up to ~2.1 billion tiles, see
/// this module's `TileCountOverflow` check below). Without a tighter cap,
/// [`extract_tile_skip`]'s destination buffer is sized directly from a layer's *declared*
/// `width*height` — which, for a tile-skip-encoded layer, can be reached with a tiny compressed
/// source blob (that's the whole point of run-length encoding) — so an attacker-chosen
/// width/height alone could otherwise force a multi-gigabyte allocation from a few bytes of
/// input. 128 MiB of `Tile`s (32 M tiles) is generously above the largest real map in the task's
/// corpus (`Abyss.map`, 1442x3399 ≈ 4.9 M tiles) yet still bounded — see the robustness test
/// module.
const MAX_TILE_COUNT: i64 = 32 * 1024 * 1024;

/// One physics role's winning (last-seen) tiles layer, as much as pass 2 needs to finish the job
/// pass 1 started: which data index to read, and (to decode `CTile` layers) its own declared
/// dimensions/version (a physics layer's own `width`/`height` can legitimately exceed the game
/// layer's — map.cpp:640-648 requires only "at least as many tiles", never "exactly as many" —
/// and its content must be validated/unpacked against *its own* size, not the game layer's, even
/// though only the first `game_w * game_h` records end up in [`ddai_physics::map::MapData`]).
#[derive(Clone, Copy)]
struct Winner {
    data_index: i32,
    width: i32,
    height: i32,
    item_version: i32,
}

struct Winners {
    game: Option<Winner>,
    front: Option<Winner>,
    tele: Option<Winner>,
    speedup: Option<Winner>,
    switch: Option<Winner>,
    tune: Option<Winner>,
}

/// One `LAYERTYPE_TILES` item found while walking groups (pass 1's output, reused by pass 2 so
/// the item table is only parsed once).
struct TilesLayer {
    group: usize,
    layer: usize,
    tilemap: mapitems::Tilemap,
}

pub struct Loaded {
    pub map: MapData,
    pub settings: Vec<String>,
    /// `-1`-as-`None` map-info string data indices, resolved to owned strings by
    /// [`crate::lib`]'s `load_map` (kept as indices here so this module doesn't need to know
    /// about `MapInfo`'s shape).
    pub info_author: Option<i32>,
    pub info_map_version: Option<i32>,
    pub info_credits: Option<i32>,
    pub info_license: Option<i32>,
}

pub fn load(df: &Datafile) -> Result<Loaded, MapError> {
    validate_version_item(df)?;

    let (layers_start, layers_num) = df.type_range(MAPITEMTYPE_LAYER);
    let (groups_start, groups_num) = df.type_range(MAPITEMTYPE_GROUP);

    // --- pass 1: per-layer structural validation + last-wins role selection -------------------
    let mut tiles_layers: Vec<TilesLayer> = Vec::new();
    let mut used_layer_items = std::collections::HashSet::with_capacity(layers_num);
    let mut winners = Winners {
        game: None,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
    };

    for group_idx in 0..groups_num {
        let group_item = df.item(groups_start + group_idx)?;
        if group_item.payload.len() < mapitems::SIZEOF_GROUP_V1 {
            return Err(MapError::InvalidGroup { group: group_idx });
        }
        let group = mapitems::parse_group_v1(group_item.payload)?;
        if group.start_layer < 0
            || group.num_layers < 0
            || (group.start_layer as i64) + (group.num_layers as i64) > layers_num as i64
        {
            return Err(MapError::InvalidGroup { group: group_idx });
        }
        for layer_idx in 0..group.num_layers as usize {
            let layer_item_index = layers_start + group.start_layer as usize + layer_idx;
            if !used_layer_items.insert(layer_item_index) {
                return Err(MapError::LayerReusedByTwoGroups { layer_item_index });
            }
            let layer_item = df.item(layer_item_index)?;
            if layer_item.payload.len() < mapitems::SIZEOF_LAYER {
                return Err(MapError::InvalidTilemapItem {
                    group: group_idx,
                    layer: layer_idx,
                });
            }
            if mapitems::layer_type(layer_item.payload)? != LAYERTYPE_TILES {
                continue; // quads/sounds/etc. — untouched by CMap::Load beyond the size check above.
            }
            let tilemap = mapitems::parse_tilemap(layer_item.payload, group_idx, layer_idx)?;

            let winner = Winner {
                data_index: tilemap.data,
                width: tilemap.width,
                height: tilemap.height,
                item_version: tilemap.item_version,
            };
            match tilemap.flags {
                f if f & mapitems::TILESLAYERFLAG_GAME != 0 => winners.game = Some(winner),
                f if f & mapitems::TILESLAYERFLAG_TELE != 0 => {
                    winners.tele = Some(Winner {
                        data_index: tilemap.tele,
                        ..winner
                    })
                }
                f if f & mapitems::TILESLAYERFLAG_SPEEDUP != 0 => {
                    winners.speedup = Some(Winner {
                        data_index: tilemap.speedup,
                        ..winner
                    })
                }
                f if f & mapitems::TILESLAYERFLAG_FRONT != 0 => {
                    winners.front = Some(Winner {
                        data_index: tilemap.front,
                        ..winner
                    })
                }
                f if f & mapitems::TILESLAYERFLAG_SWITCH != 0 => {
                    winners.switch = Some(Winner {
                        data_index: tilemap.switch,
                        ..winner
                    })
                }
                f if f & mapitems::TILESLAYERFLAG_TUNE != 0 => {
                    winners.tune = Some(Winner {
                        data_index: tilemap.tune,
                        ..winner
                    })
                }
                _ => {}
            }
            tiles_layers.push(TilesLayer {
                group: group_idx,
                layer: layer_idx,
                tilemap,
            });
        }
    }

    let game = winners.game.ok_or(MapError::NoGameLayer)?;
    let game_count = (game.width as i64) * (game.height as i64);

    // --- pass 2: data-index bounds/uniqueness, overflow, physics->=game-size ------------------
    let mut used_data_indices = std::collections::HashSet::with_capacity(tiles_layers.len());
    for tl in &tiles_layers {
        let physics_flag = tl.tilemap.flags & mapitems::ALL_PHYSICS_FLAGS;
        let (data_index, is_physics_non_game) = if physics_flag & mapitems::TILESLAYERFLAG_GAME != 0 {
            (tl.tilemap.data, false)
        } else if physics_flag & mapitems::TILESLAYERFLAG_TELE != 0 {
            (tl.tilemap.tele, true)
        } else if physics_flag & mapitems::TILESLAYERFLAG_SPEEDUP != 0 {
            (tl.tilemap.speedup, true)
        } else if physics_flag & mapitems::TILESLAYERFLAG_FRONT != 0 {
            (tl.tilemap.front, true)
        } else if physics_flag & mapitems::TILESLAYERFLAG_SWITCH != 0 {
            (tl.tilemap.switch, true)
        } else if physics_flag & mapitems::TILESLAYERFLAG_TUNE != 0 {
            (tl.tilemap.tune, true)
        } else {
            (tl.tilemap.data, false) // plain/decorative
        };

        if data_index < 0 || data_index as usize >= df.num_data() {
            return Err(MapError::DataIndexOutOfRange {
                group: tl.group,
                layer: tl.layer,
            });
        }
        if !used_data_indices.insert(data_index) {
            return Err(MapError::DataIndexReused {
                group: tl.group,
                layer: tl.layer,
            });
        }
        // map.cpp:630-638's overflow check (this platform's practical effect: does `width*height`
        // fit in an `i32`) applies to *every* tiles layer, decorative included — a real DDNet
        // check, so it stays here. `MAX_TILE_COUNT` (this crate's own, tighter cap) does **not**:
        // review round 1 finding F5 — a decorative layer this crate never decodes must not be
        // rejected merely for being large; that cap is applied only where a layer is actually
        // about to be decoded (`decode_ctile_layer`/`decode_small_layer`/`decode_speedup_layer`).
        let count = (tl.tilemap.width as i64) * (tl.tilemap.height as i64);
        if count > i32::MAX as i64 {
            return Err(MapError::TileCountOverflow {
                group: tl.group,
                layer: tl.layer,
            });
        }
        if is_physics_non_game && count < game_count {
            return Err(MapError::PhysicsLayerSmallerThanGame {
                group: tl.group,
                layer: tl.layer,
            });
        }
    }

    // --- decode the six layers this crate actually keeps (see this module's doc comment) ------
    // Review round 1 finding F3: only the GAME layer's own data failure is a hard map-load
    // error (map.cpp:190-195 forces it eagerly, exactly like this — a `?`). Every other physics
    // layer's data failure (bad padding, a disallowed tile-skip role, a truncated/corrupt/
    // mismatched-declared-size blob, or this crate's own `MAX_TILE_COUNT` cap) mirrors DDNet's
    // own lazy `nullptr`-on-failure (`CCollision::Init`, collision.cpp:59-86, and every downstream
    // guard like `if(m_pTele)`): the layer is simply absent from `MapData`, via `.ok()` turning an
    // `Err` into `None` — the whole map still loads.
    let game_tiles = decode_ctile_layer(df, &game, "game")?;
    let front_tiles = winners.front.and_then(|w| decode_ctile_layer(df, &w, "front").ok());
    let tele_tiles = winners.tele.and_then(|w| {
        decode_small_layer(df, &w, 2, "tele", |b| TeleTile {
            number: b[0],
            kind: b[1],
        })
        .ok()
    });
    let speedup_tiles = winners.speedup.and_then(|w| decode_speedup_layer(df, &w).ok());
    let switch_tiles = winners.switch.and_then(|w| {
        decode_small_layer(df, &w, 4, "switch", |b| SwitchTile {
            number: b[0],
            kind: b[1],
            flags: b[2],
            delay: b[3],
        })
        .ok()
    });
    let tune_tiles = winners.tune.and_then(|w| {
        decode_small_layer(df, &w, 2, "tune", |b| TuneTile {
            number: b[0],
            kind: b[1],
        })
        .ok()
    });

    // Physics layers may declare more tiles than the game layer (never fewer — checked above);
    // `CCollision`/this crate's `MapData` only ever address the first `game_w * game_h` of them
    // (`collision.cpp:59-61`'s `m_Width = GameLayer()->m_Width` and everything downstream
    // indexing off *that*, not each physics layer's own declared size).
    let n = game_count as usize;
    let truncate_tiles = |v: Vec<Tile>| -> Vec<Tile> { v.into_iter().take(n).collect() };
    let front_tiles = front_tiles.map(truncate_tiles);
    fn truncate<T>(v: Vec<T>, n: usize) -> Vec<T> {
        let mut v = v;
        v.truncate(n);
        v
    }
    let tele_tiles = tele_tiles.map(|v| truncate(v, n));
    let speedup_tiles = speedup_tiles.map(|v| truncate(v, n));
    let switch_tiles = switch_tiles.map(|v| truncate(v, n));
    let tune_tiles = tune_tiles.map(|v| truncate(v, n));

    let (settings, info_author, info_map_version, info_credits, info_license) = read_info(df, &used_data_indices)?;

    let map = MapData {
        width: game.width as u32,
        height: game.height as u32,
        game: truncate_tiles(game_tiles),
        front: front_tiles,
        tele: tele_tiles,
        speedup: speedup_tiles,
        switch: switch_tiles,
        tune: tune_tiles,
        settings: settings.clone(),
    };
    // `MapData::validate` re-checks exactly the invariant this function must already uphold
    // (every present layer has `width*height` entries) — cheap, and it turns any bug in the
    // truncation logic above into a clear panic-free error instead of a silent length mismatch
    // `ddai-trace`'s rawmap writer would otherwise panic on (see `rawmap::write`'s doc comment).
    map.validate()
        .map_err(|_| MapError::InternalInvariantViolation("layer length mismatch after truncation"))?;

    Ok(Loaded {
        map,
        settings,
        info_author,
        info_map_version,
        info_credits,
        info_license,
    })
}

/// `CMap::ValidateMapVersion` (map.cpp:255-276): `FindItemIndex(MAPITEMTYPE_VERSION, 0)` must
/// exist, be at least `sizeof(CMapItemVersion)` (4 bytes), and have `m_Version == 1`.
fn validate_version_item(df: &Datafile) -> Result<(), MapError> {
    let (start, num) = df.type_range(MAPITEMTYPE_VERSION);
    for i in start..start + num {
        let item = df.item(i)?;
        if item.id != 0 {
            continue;
        }
        if item.payload.len() < 4 {
            return Err(MapError::MissingOrUnsupportedVersionItem);
        }
        let version = mapitems::i32_at(item.payload, 0)?;
        return if version == 1 {
            Ok(())
        } else {
            Err(MapError::MissingOrUnsupportedVersionItem)
        };
    }
    Err(MapError::MissingOrUnsupportedVersionItem)
}

/// Decodes a `CTile`-shaped layer (game/front — the only two roles besides plain/decorative
/// layers that can carry either encoding): tile-skip-unpacks it when the tilemap item's own
/// `m_Version >= 4` (map.cpp:650-682's `ExtractTiles` branch), otherwise reads it 1:1 after
/// checking every record's `m_Skip`/`m_MustBe0` padding is zero (map.cpp:690-710) — front is
/// *not* one of the two roles DDNet permits tile-skip encoding for (map.cpp:657-663's
/// `LayerType != LAYERTYPE_TILES && LayerType != LAYERTYPE_GAME` check; `LayerType` is
/// `LAYERTYPE_FRONT` for a front layer, never one of those two), so this function rejects that
/// combination for `front` too, via `allow_skip`.
fn decode_ctile_layer(df: &Datafile, w: &Winner, name: &'static str) -> Result<Vec<Tile>, MapError> {
    let fail = || MapError::PhysicsLayerDataFailed { layer: name };
    let allow_skip = name == "game";
    let count_i64 = (w.width as i64) * (w.height as i64);
    // Review round 1 finding F5: this cap belongs here (right before we actually decode this
    // *specific* layer), not in the generic pass-2 structural walk, which must not reject a
    // decorative layer of this size that never reaches this function at all.
    if count_i64 > MAX_TILE_COUNT {
        return Err(fail());
    }
    let count = count_i64 as usize;
    let need = count.checked_mul(4).ok_or_else(fail)?;
    // Review round 1 finding F2: `data_len` costs nothing (no decompression), and knowing the
    // *true* declared length up front is what lets `extract_tile_skip` correctly detect leftover
    // source data without this function ever fetching more than `need` bytes into memory.
    let declared = df.data_len(w.data_index as usize).map_err(|_| fail())?;
    let raw = df.data_bounded(w.data_index as usize, need).map_err(|_| fail())?;

    if w.item_version >= 4 {
        if !allow_skip {
            return Err(fail());
        }
        return extract_tile_skip(&raw, declared, count, name);
    }

    if raw.len() < need {
        return Err(fail());
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = &raw[i * 4..i * 4 + 4];
        if b[2] != 0 || b[3] != 0 {
            // map.cpp:695-709: non-zero `m_Skip`/`m_MustBe0` outside a tile-skip-encoded layer.
            return Err(fail());
        }
        out.push(Tile {
            index: b[0],
            flags: b[1],
            skip: 0,
            reserved: 0,
        });
    }
    Ok(out)
}

/// `CMap::ExtractTiles` (map.cpp:278-313): expands a run-length-packed `CTile` stream (each
/// source record covers `m_Skip + 1` identical destination cells) into exactly `count` tiles —
/// rejecting a non-zero `m_MustBe0` or a source stream that doesn't produce *exactly* `count`
/// tiles. `declared_total_bytes` is the blob's *true* logical length (from `Datafile::data_len`,
/// which may be larger than `raw.len()` if the caller only fetched a bounded prefix) — leftover
/// source data is detected in **record units**, `declared_total_bytes / 4` vs. `src / 4`, both
/// floor-divided exactly like map.cpp:651's `SavedTilesSize = Size / sizeof(CTile)`: a 1-3 byte
/// trailing remainder is never part of `SavedTilesSize` at all, so it is never "leftover" (review
/// round 1 finding F4 — the previous `src != raw.len()` check compared bytes, not records, and so
/// rejected exactly this harmless case).
fn extract_tile_skip(
    raw: &[u8],
    declared_total_bytes: usize,
    count: usize,
    name: &'static str,
) -> Result<Vec<Tile>, MapError> {
    let fail = || MapError::PhysicsLayerDataFailed { layer: name };
    let mut out = Vec::with_capacity(count);
    let mut src = 0usize;
    while out.len() < count && src + 4 <= raw.len() {
        let index = raw[src];
        let flags = raw[src + 1];
        let skip = raw[src + 2];
        let must_be_0 = raw[src + 3];
        if must_be_0 != 0 {
            return Err(fail());
        }
        src += 4;
        let repeat = skip as usize + 1;
        for _ in 0..repeat {
            if out.len() >= count {
                break;
            }
            out.push(Tile {
                index,
                flags,
                skip: 0,
                reserved: 0,
            });
        }
    }
    if out.len() != count || src / 4 != declared_total_bytes / 4 {
        return Err(fail());
    }
    Ok(out)
}

/// Decodes a fixed-size-record physics layer with no tile-skip support and no reserved/padding
/// byte to validate (`CTeleTile`/`CSwitchTile`/`CTuneTile`) — a plain 1:1 read, rejecting
/// `item_version >= 4` (never valid for these roles; see [`decode_ctile_layer`]'s doc comment),
/// this crate's own `MAX_TILE_COUNT` cap (review round 1 finding F5 — applied here, at decode
/// time, not in the generic structural pass), and a too-short data blob.
fn decode_small_layer<T>(
    df: &Datafile,
    w: &Winner,
    record_size: usize,
    name: &'static str,
    parse: impl Fn(&[u8]) -> T,
) -> Result<Vec<T>, MapError> {
    let fail = || MapError::PhysicsLayerDataFailed { layer: name };
    if w.item_version >= 4 {
        return Err(fail());
    }
    let count_i64 = (w.width as i64) * (w.height as i64);
    if count_i64 > MAX_TILE_COUNT {
        return Err(fail());
    }
    let count = count_i64 as usize;
    let need = count.checked_mul(record_size).ok_or_else(fail)?;
    let raw = df.data_bounded(w.data_index as usize, need).map_err(|_| fail())?;
    if raw.len() < need {
        return Err(fail());
    }
    Ok(raw.chunks_exact(record_size).take(count).map(parse).collect())
}

/// `CSpeedupTile` (mapitems.h:661-669): like [`decode_small_layer`] but with an `m_MustBe0` byte
/// (offset 3) to validate, and a little-endian `i16` angle (offset 4-5) — matches
/// map.cpp:711-724's speedup-specific padding check.
fn decode_speedup_layer(df: &Datafile, w: &Winner) -> Result<Vec<SpeedupTile>, MapError> {
    let fail = || MapError::PhysicsLayerDataFailed { layer: "speedup" };
    if w.item_version >= 4 {
        return Err(fail());
    }
    let count_i64 = (w.width as i64) * (w.height as i64);
    if count_i64 > MAX_TILE_COUNT {
        return Err(fail());
    }
    let count = count_i64 as usize;
    let need = count.checked_mul(6).ok_or_else(fail)?;
    let raw = df.data_bounded(w.data_index as usize, need).map_err(|_| fail())?;
    if raw.len() < need {
        return Err(fail());
    }
    let mut out = Vec::with_capacity(count);
    let (chunks, _) = raw.as_chunks::<6>();
    for chunk in chunks.iter().take(count) {
        if chunk[3] != 0 {
            return Err(fail());
        }
        out.push(SpeedupTile {
            force: chunk[0],
            max_speed: chunk[1],
            kind: chunk[2],
            angle: i16::from_le_bytes([chunk[4], chunk[5]]),
        });
    }
    Ok(out)
}

type InfoResult = (Vec<String>, Option<i32>, Option<i32>, Option<i32>, Option<i32>);

/// A hard cap on the Settings blob's *logical* length this crate will even attempt to read
/// (review round 1 finding F2). Beyond this, the map's settings are treated as empty outright —
/// not truncated — because a truncated settings blob risks executing a garbled, partial `sv_*`
/// command rather than the complete one a real map author wrote; an empty list is the safer
/// divergence (documented in docs/formats.md §10.2). No real map in the task's corpus (largest
/// Settings blob observed: a few hundred bytes) comes remotely close to 1 MiB.
const SETTINGS_MAX_BYTES: usize = 1024 * 1024;
/// A hard cap on the *number* of settings entries this crate will produce, independent of the
/// byte cap above (review round 1 finding F2) — a blob that's *entirely* `NUL` bytes would
/// otherwise still allocate one (admittedly tiny) `String` per byte within the byte cap alone.
const SETTINGS_MAX_COUNT: usize = 64 * 1024;

/// `CGameContext::LoadMapSettings` (gamecontext.cpp:4556-4580) for the settings list — the only
/// real DDNet server code path that reads a map's Settings blob at all — plus, for
/// author/version/credits/license, the *indices* `CMapItemInfoSettings` declares (resolved to
/// strings by `crate::load_map`; see this crate's top-level docs for why no server code reads
/// those fields, so there's nothing further to mirror for them beyond "index present or not").
///
/// `used_by_tiles_layer` is pass 2's own `used_data_indices` set (review round 1 finding F6, its
/// wording corrected in review round 2 — the previous version of this comment overstated how
/// often DDNet actually rejects this case). `CDatafile::GetData` caches one decompressed-data
/// *processor* per data **index**, shared by every consumer of that index, not per call site — if
/// some tiles layer also claims the exact index the Settings blob lives at, that layer's own
/// content-validation processor (`ValidateAndUnpackTilesLayerData`) runs first, *however that
/// processor was going to behave for its own layer regardless of Settings*. If it would have
/// **rejected** the blob for that layer (too small, bad padding, ...), `GetData` returns
/// `nullptr` to *everyone*, settings-reader included — DDNet ends up with no settings, same as
/// this crate. But if it would have **accepted** the blob (a real reproduction: a real map's item
/// mutated so a decorative layer's own, perfectly valid tile data shares the Settings index — see
/// the build report), `GetData` *succeeds* and returns that layer's real tile bytes, which
/// `LoadMapSettings` then naively reinterprets as NUL-separated strings — DDNet has no
/// special-casing here at all, it just re-reads the same bytes for an unrelated purpose whenever
/// two indices coincide, however large or nonsensical the result (observed: over a million
/// "settings" reinterpreted from one real layer's tile grid).
///
/// This crate always takes the first outcome (no settings) for *any* collision, by the
/// orchestrator's explicit decision: it is the safer default (a mutated/corrupted map's colliding
/// index does not fabricate a huge, garbage settings list — see docs/formats.md §10.2's own note
/// on why an oversized Settings blob is treated as absent rather than acted on), and only crafted
/// or corrupted files can create this collision at all — no real, editor-produced map does.
fn read_info(df: &Datafile, used_by_tiles_layer: &std::collections::HashSet<i32>) -> Result<InfoResult, MapError> {
    let (start, num) = df.type_range(mapitems::MAPITEMTYPE_INFO);
    for i in start..start + num {
        let item = df.item(i)?;
        if item.id != 0 {
            continue;
        }
        let Some(info) = mapitems::parse_info_settings(item.payload) else {
            break;
        };
        let opt = |idx: i32| if idx > -1 { Some(idx) } else { None };
        let settings = match opt(info.settings) {
            None => Vec::new(),
            Some(idx) if used_by_tiles_layer.contains(&idx) => Vec::new(),
            Some(idx) => match df.data_len(idx as usize) {
                Ok(len) if len <= SETTINGS_MAX_BYTES => match df.data_bounded(idx as usize, SETTINGS_MAX_BYTES) {
                    Ok(blob) => split_nul_terminated(&blob, SETTINGS_MAX_COUNT),
                    Err(_) => Vec::new(), // gamecontext.cpp:4571 — a failed read is "no settings", not a hard error.
                },
                _ => Vec::new(), // absent, corrupt, or over the byte cap — all treated the same way.
            },
        };
        return Ok((
            settings,
            opt(info.author),
            opt(info.map_version),
            opt(info.credits),
            opt(info.license),
        ));
    }
    Ok((Vec::new(), None, None, None, None))
}

/// Splits a blob of concatenated NUL-terminated C strings (matches `CGameContext::
/// LoadMapSettings`'s `while(pNext < pSettings + Size)` loop), stopping after `max_count` entries
/// (review round 1 finding F2 — bounds the number of `String` allocations independently of the
/// blob's byte length). Decoded with [`String::from_utf8_lossy`] rather than rejected on invalid
/// UTF-8: DDNet's own reader never validates encoding here either (`Console()->ExecuteLine` takes
/// raw bytes), and this crate must never panic or error out on arbitrary map bytes (see the
/// crate's robustness test module).
fn split_nul_terminated(blob: &[u8], max_count: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, &b) in blob.iter().enumerate() {
        if out.len() >= max_count {
            break;
        }
        if b == 0 {
            out.push(String::from_utf8_lossy(&blob[start..i]).into_owned());
            start = i + 1;
        }
    }
    // A non-NUL-terminated trailing fragment (a malformed/fuzzed settings blob) is simply
    // dropped, matching the fact that `CGameContext::LoadMapSettings`'s C-string walk would never
    // "see" it as a complete setting either (`str_length` scans for the same terminator).
    out
}
