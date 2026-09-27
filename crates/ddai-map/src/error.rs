//! [`MapError`]: everything [`crate::load_map`] can fail with.
//!
//! Variants are grouped by which layer of the format they come from (datafile container, then
//! map items on top of it) and each cites the DDNet 20.1 source location whose check it mirrors,
//! so a divergence found by `tools/ddnet-oracle/map-corpus-check.sh` can be traced back to the
//! exact C++ line that produced (or didn't produce) the same rejection.

use std::fmt;

/// Everything [`crate::load_map`] can fail with. Every variant is a **rejection**, exactly
/// mirroring some DDNet 20.1 loader check returning `false` (see each variant's doc comment for
/// the `file:line` it mirrors) — this crate never panics on malformed input (see the
/// `robustness` test module), it returns one of these instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapError {
    // --- datafile container (engine/shared/datafile.cpp `CDataFileReader::Open`/`Validate`) ---
    /// Fewer than 36 bytes (`sizeof(CDatafileHeader)`) — datafile.cpp:557.
    HeaderTruncated,
    /// First 4 bytes are neither `DATA` nor `ATAD` — datafile.cpp:565-571.
    BadMagic,
    /// Header `m_Version` is neither 3 nor 4 — datafile.cpp:576-581.
    UnsupportedDatafileVersion(i32),
    /// A header count/size field is negative, or `m_ItemSize` isn't a multiple of 4 —
    /// datafile.cpp:584-596.
    InvalidHeaderField(&'static str),
    /// `sizeof(header) + Size + DataSize != file length`, where `Size` is computed from the
    /// header's own counts (datafile.cpp:598-621) — the file's declared shape doesn't match its
    /// actual length. Covers plain truncation as a special case.
    SizeMismatch,
    /// `m_Size`/`m_Swaplen` don't match the file length even after the legacy pre-`3dd1ea0`
    /// v4-map size-fix DDNet still special-cases — datafile.cpp:622-654.
    HeaderSizeMismatch,
    /// The header's own counts, taken at face value, would require an allocation past this
    /// crate's sanity cap (see `datafile::MAX_ALLOC_BYTES`) — a bounded-allocation rejection with
    /// no direct DDNet equivalent (DDNet caps only the header+item-table allocation at 2 GiB,
    /// datafile.cpp:656; this crate additionally caps *every* single allocation the same way, so
    /// a well-formed real map is never affected — see the task's robustness/fuzz requirement).
    AllocationTooLarge(&'static str),
    /// The item type table has a type outside `0..=0xFFFF`, a duplicate type, a `m_Start` that
    /// doesn't match the running item count, or the counted total doesn't match `m_NumItems` —
    /// datafile.cpp:400-417 (`CDatafile::Validate`).
    InvalidItemTypeTable,
    /// An item offset is not `0` for the first item, not strictly increasing, or not less than
    /// `m_ItemSize` — datafile.cpp:419-434.
    InvalidItemOffsets,
    /// An item is smaller than `sizeof(CDatafileItem)` (8 bytes), its type doesn't match its
    /// item-type-table entry, its `m_Size` is negative/not a multiple of 4/doesn't match the
    /// file, or two items of the same type share an `Id` (except `ITEMTYPE_EX`, which DDNet
    /// tolerates — datafile.cpp:448-453) — datafile.cpp:436-464.
    InvalidItem,
    /// A data offset is not `0` for the first entry, not strictly increasing, or not less than
    /// `m_DataSize` — datafile.cpp:466-481.
    InvalidDataOffsets,
    /// A v4 declared uncompressed data size is negative — datafile.cpp:484-497.
    InvalidDataSize,
    /// Reading (v3, raw) or inflating (v4, zlib) a data blob failed: for v4, the declared
    /// uncompressed size is `0` (datafile.cpp:224-230, an explicit "ignore, don't crash" case for
    /// old maps with this quirk) or zlib rejected the stream / produced a different size than
    /// declared (datafile.cpp:265-274).
    DataDecompressFailed { index: usize },

    // --- map items (engine/shared/map.cpp `CMap::Load` + game/layers.cpp `CLayers::Init`) -----
    /// No `MAPITEMTYPE_VERSION` item, or its `m_Version != 1` — map.cpp:257-275
    /// (`ValidateMapVersion`).
    MissingOrUnsupportedVersionItem,
    /// A `MAPITEMTYPE_GROUP` item is smaller than `CMapItemGroup_v1` (28 bytes), or its
    /// `m_StartLayer`/`m_NumLayers` reference layers out of range — map.cpp:113-126.
    InvalidGroup { group: usize },
    /// Two groups both claim the same `MAPITEMTYPE_LAYER` item index — map.cpp:127-134.
    LayerReusedByTwoGroups { layer_item_index: usize },
    /// A layer item is smaller than `CMapItemLayer` (12 bytes) — map.cpp:138-142 — or, once
    /// known to be a tiles layer, smaller than `CMapItemLayerTilemap_v2` (60 bytes) or has an
    /// unsupported `m_Version` (must be `2..=4`) — map.cpp:460-472.
    InvalidTilemapItem { group: usize, layer: usize },
    /// A tiles layer's own physics-role field (`m_Tele`/`m_Speedup`/`m_Front`/`m_Switch`/
    /// `m_Tune`) is truncated *and* its matching `TILESLAYERFLAG_*` bit is set — map.cpp:481-499.
    TruncatedPhysicsField { group: usize, layer: usize },
    /// A tiles layer sets more than one of `TILESLAYERFLAG_{GAME,TELE,SPEEDUP,FRONT,SWITCH,
    /// TUNE}` — map.cpp:474-479.
    MultiplePhysicsFlags { group: usize, layer: usize },
    /// A tiles layer's `m_Width`/`m_Height` is less than 2 — map.cpp:317-329
    /// (`EnsureTileLayerProperties`, runs for every tiles layer, not only physics ones).
    InvalidLayerDimensions { group: usize, layer: usize },
    /// A tiles layer's `m_aName` doesn't decode to valid UTF-8 — map.cpp:331-346's
    /// `EnsureValidName` (via `IntsToStr`/`str_utf8_check`), runs for every tiles layer (physics
    /// or decorative) whose tilemap item version is `3` or `4`; version-2 items have no name
    /// field at all and are exempt (map.cpp:507-508 forces the encoding of `""`, always valid).
    InvalidLayerName { group: usize, layer: usize },
    /// A decorative (no physics flag) tiles layer's `m_Color.{r,g,b,a}` is outside `0..=255` —
    /// map.cpp:428-436. Physics-flagged layers are exempt: DDNet only resets their color to the
    /// default there, never rejects (`EnsureDefaultColor`, log-only).
    InvalidLayerColor { group: usize, layer: usize },
    /// A tiles layer's data index is negative or `>= NumData()` — map.cpp:617-621.
    DataIndexOutOfRange { group: usize, layer: usize },
    /// Two tiles layers (of any kind) claim the same data index — map.cpp:623-628.
    DataIndexReused { group: usize, layer: usize },
    /// `width * height` doesn't fit in an `i32` (map.cpp:630-638's practical effect on this
    /// 64-bit platform — see `datafile.rs`'s parsing for the exact reasoning), or exceeds this
    /// crate's own, tighter `loader::MAX_TILE_COUNT` bounded-allocation cap (no direct DDNet
    /// equivalent; see that constant's doc comment — never trips on a real map in the task's
    /// corpus, whose largest is ~4.9 million tiles).
    TileCountOverflow { group: usize, layer: usize },
    /// A non-game, non-decorative physics layer has fewer tiles than the game layer —
    /// map.cpp:640-648.
    PhysicsLayerSmallerThanGame { group: usize, layer: usize },
    /// No layer had `TILESLAYERFLAG_GAME` set — map.cpp:165-169.
    NoGameLayer,
    /// The game layer's own tile data failed to load (map.cpp:190-195) or, for a present
    /// front/tele/speedup/switch/tune layer, that layer's data failed to load. Unlike DDNet's own
    /// lazy `nullptr`-on-failure (CCollision would just treat the layer as absent), this crate
    /// treats it as the whole map failing to load — see this crate's top-level docs for why.
    PhysicsLayerDataFailed { layer: &'static str },

    /// Not a DDNet check at all: [`crate::loader`] built a [`ddai_physics::map::MapData`] whose
    /// own `validate()` then disagreed with it — provably unreachable given the loader's own
    /// logic (every layer is truncated to exactly the game layer's tile count right before this
    /// would fire), kept as a returned error instead of a `panic!`/`unwrap()` only so a bug here
    /// can never violate this crate's "never panics" contract.
    InternalInvariantViolation(&'static str),
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapError::HeaderTruncated => write!(f, "file is shorter than the datafile header (36 bytes)"),
            MapError::BadMagic => write!(f, "bad datafile magic (expected b\"DATA\" or b\"ATAD\")"),
            MapError::UnsupportedDatafileVersion(v) => write!(f, "unsupported datafile version {v} (expected 3 or 4)"),
            MapError::InvalidHeaderField(field) => write!(f, "invalid datafile header field: {field}"),
            MapError::SizeMismatch => write!(f, "datafile header sizes don't add up to the file's actual length"),
            MapError::HeaderSizeMismatch => write!(f, "datafile header m_Size/m_Swaplen don't match the file length"),
            MapError::AllocationTooLarge(what) => {
                write!(f, "refusing to allocate: {what} exceeds this crate's sanity cap")
            }
            MapError::InvalidItemTypeTable => write!(f, "invalid item type table"),
            MapError::InvalidItemOffsets => write!(f, "invalid item offset table"),
            MapError::InvalidItem => write!(f, "invalid item (bad size, type mismatch, or duplicate id)"),
            MapError::InvalidDataOffsets => write!(f, "invalid data offset table"),
            MapError::InvalidDataSize => write!(f, "invalid (negative) declared data size"),
            MapError::DataDecompressFailed { index } => write!(f, "failed to load/decompress data blob {index}"),
            MapError::MissingOrUnsupportedVersionItem => {
                write!(f, "missing MAPITEMTYPE_VERSION item, or its version isn't 1")
            }
            MapError::InvalidGroup { group } => write!(f, "group {group} is invalid (truncated or bad layer range)"),
            MapError::LayerReusedByTwoGroups { layer_item_index } => {
                write!(f, "layer item {layer_item_index} is used by two groups")
            }
            MapError::InvalidTilemapItem { group, layer } => {
                write!(
                    f,
                    "tiles layer {layer} in group {group} is truncated or has an unsupported version"
                )
            }
            MapError::TruncatedPhysicsField { group, layer } => {
                write!(
                    f,
                    "tiles layer {layer} in group {group} is truncated (missing a physics field its flags require)"
                )
            }
            MapError::MultiplePhysicsFlags { group, layer } => {
                write!(
                    f,
                    "tiles layer {layer} in group {group} sets more than one physics-role flag"
                )
            }
            MapError::InvalidLayerDimensions { group, layer } => {
                write!(f, "tiles layer {layer} in group {group} has width or height < 2")
            }
            MapError::InvalidLayerName { group, layer } => {
                write!(
                    f,
                    "tiles layer {layer} in group {group} has a name that isn't valid UTF-8"
                )
            }
            MapError::InvalidLayerColor { group, layer } => {
                write!(
                    f,
                    "decorative layer {layer} in group {group} has a color component outside 0..=255"
                )
            }
            MapError::DataIndexOutOfRange { group, layer } => {
                write!(f, "tiles layer {layer} in group {group} has a data index out of range")
            }
            MapError::DataIndexReused { group, layer } => {
                write!(
                    f,
                    "tiles layer {layer} in group {group} reuses another layer's data index"
                )
            }
            MapError::TileCountOverflow { group, layer } => {
                write!(
                    f,
                    "tiles layer {layer} in group {group} has a width*height that overflows"
                )
            }
            MapError::PhysicsLayerSmallerThanGame { group, layer } => {
                write!(
                    f,
                    "physics layer {layer} in group {group} has fewer tiles than the game layer"
                )
            }
            MapError::NoGameLayer => write!(f, "map has no game layer"),
            MapError::PhysicsLayerDataFailed { layer } => write!(f, "{layer} layer data failed to load"),
            MapError::InternalInvariantViolation(what) => write!(f, "internal error (please file a bug): {what}"),
        }
    }
}

impl std::error::Error for MapError {}
