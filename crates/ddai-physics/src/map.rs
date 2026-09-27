//! Plain-data DDNet map representation.
//!
//! This module holds **only data** — tile grids and the handful of per-tile records DDNet
//! stores per physics layer — no collision or gameplay logic (that lands in a later task, see
//! `docs/PLAN.md` §1.3). It exists so `ddai-trace` (synthetic recipes, the rawmap file format)
//! and the future Rust physics port share one definition of "what a map is", and so the C++
//! oracle (`tools/ddnet-oracle`) and Rust agree byte-for-byte on tile layout.
//!
//! The tile structs mirror DDNet 20.1's `src/game/mapitems.h` field-for-field (same field order,
//! same integer widths), so a rawmap reader/writer can treat them as fixed-size records. Doc
//! comments below cite the exact `mapitems.h` class they mirror.
//!
//! `type` is a Rust keyword, so the `m_Type` field of `CTeleTile`/`CSpeedupTile`/`CSwitchTile`/
//! `CTuneTile` is named `kind` here instead.

/// A complete DDNet map, as plain data: one tile grid per physics layer, plus the map's
/// "Settings" strings (server config commands baked into the map, e.g. `sv_foo 1`).
///
/// `game` is always present (every DDNet map has a game layer); the other physics layers are
/// optional, matching DDNet: a map with no tele tiles has no tele layer at all.
#[derive(Debug, Clone, PartialEq)]
pub struct MapData {
    /// Tile grid width, in tiles.
    pub width: u32,
    /// Tile grid height, in tiles.
    pub height: u32,
    /// Game layer tiles, row-major, length `width * height`. Mirrors `CLayers::GameLayer()`.
    pub game: Vec<Tile>,
    /// Front layer tiles (hook-through, stoppers, front-only freeze/death), if the map has one.
    /// Mirrors `CLayers::FrontLayer()`.
    pub front: Option<Vec<Tile>>,
    /// Tele layer records (teleporters, checkpoints), if the map has one. Mirrors
    /// `CLayers::TeleLayer()`.
    pub tele: Option<Vec<TeleTile>>,
    /// Speedup layer records, if the map has one. Mirrors `CLayers::SpeedupLayer()`.
    pub speedup: Option<Vec<SpeedupTile>>,
    /// Switch layer records (doors, timed switches), if the map has one. Mirrors
    /// `CLayers::SwitchLayer()`.
    pub switch: Option<Vec<SwitchTile>>,
    /// Tune layer records (per-zone tuning overrides), if the map has one. Mirrors
    /// `CLayers::TuneLayer()`.
    pub tune: Option<Vec<TuneTile>>,
    /// Map "Settings" strings (server config commands), in file order.
    pub settings: Vec<String>,
}

impl MapData {
    /// Number of tiles in the grid (`width * height`), the length every present layer's `Vec`
    /// must have.
    pub fn cell_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Checks that every present layer has exactly `cell_count()` entries.
    ///
    /// This is a data-shape check, not a collision/gameplay check: it exists so a malformed map
    /// (e.g. a recipe bug, or a corrupt rawmap file) is caught immediately with a clear error
    /// instead of an out-of-bounds panic deep inside a reader or (later) the physics port.
    pub fn validate(&self) -> Result<(), MapDataError> {
        let n = self.cell_count();
        let check = |name: &'static str, len: Option<usize>| -> Result<(), MapDataError> {
            match len {
                Some(len) if len != n => Err(MapDataError::LayerLengthMismatch {
                    layer: name,
                    expected: n,
                    actual: len,
                }),
                _ => Ok(()),
            }
        };
        check("game", Some(self.game.len()))?;
        check("front", self.front.as_ref().map(Vec::len))?;
        check("tele", self.tele.as_ref().map(Vec::len))?;
        check("speedup", self.speedup.as_ref().map(Vec::len))?;
        check("switch", self.switch.as_ref().map(Vec::len))?;
        check("tune", self.tune.as_ref().map(Vec::len))?;
        Ok(())
    }
}

/// A malformed [`MapData`]: some layer's tile count doesn't match `width * height`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapDataError {
    /// `layer`'s `Vec` has `actual` entries but the map's `width * height` is `expected`.
    LayerLengthMismatch {
        /// Which layer is malformed (`"game"`, `"front"`, ...).
        layer: &'static str,
        /// `width * height`.
        expected: usize,
        /// The layer's actual `Vec` length.
        actual: usize,
    },
}

impl std::fmt::Display for MapDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MapDataError::LayerLengthMismatch {
                layer,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "layer '{layer}' has {actual} tiles, expected {expected} (width * height)"
                )
            }
        }
    }
}

impl std::error::Error for MapDataError {}

/// A single game/front layer tile. Mirrors `CTile` (`mapitems.h`) exactly: `m_Index`, `m_Flags`,
/// `m_Skip`, `m_MustBe0` (named `reserved` here — it is always `0` and carries no information).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tile {
    /// Tile id, e.g. [`TILE_SOLID`]. Mirrors `CTile::m_Index`.
    pub index: u8,
    /// Rotation/flip bits, see [`TILEFLAG_XFLIP`] etc. Mirrors `CTile::m_Flags`.
    pub flags: u8,
    /// Run-length skip hint used by the renderer/`CLayers::InitTilemapSkip`; always `0` for the
    /// maps this crate builds (we never rely on it). Mirrors `CTile::m_Skip`.
    pub skip: u8,
    /// Reserved, always `0`. Mirrors `CTile::m_MustBe0`.
    pub reserved: u8,
}

/// A single tele layer record. Mirrors `CTeleTile` (`mapitems.h`): `m_Number`, `m_Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TeleTile {
    /// Teleporter number (links a `TILE_TELEIN` to its `TILE_TELEOUT`s of the same number, and
    /// so on). Mirrors `CTeleTile::m_Number`.
    pub number: u8,
    /// Tele tile type, e.g. [`TILE_TELEIN`]. Mirrors `CTeleTile::m_Type`.
    pub kind: u8,
}

/// A single speedup layer record. Mirrors `CSpeedupTile` (`mapitems.h`): `m_Force`,
/// `m_MaxSpeed`, `m_Type`, `m_Angle` (the layout also has an `m_MustBe0` padding byte between
/// `m_Type` and `m_Angle`, omitted here since it carries no information — see `docs/formats.md`
/// for how the rawmap binary format still reserves that byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpeedupTile {
    /// Acceleration per tick applied in `angle`'s direction. Mirrors `CSpeedupTile::m_Force`.
    pub force: u8,
    /// Speed cap for the boost, `0` = tuning's default max speed. Mirrors
    /// `CSpeedupTile::m_MaxSpeed`.
    pub max_speed: u8,
    /// Speedup type: [`TILE_SPEED_BOOST_OLD`] (28) or [`TILE_SPEED_BOOST`] (29). Mirrors
    /// `CSpeedupTile::m_Type`.
    pub kind: u8,
    /// Direction in degrees. Mirrors `CSpeedupTile::m_Angle`.
    pub angle: i16,
}

/// A single switch layer record. Mirrors `CSwitchTile` (`mapitems.h`): `m_Number`, `m_Type`,
/// `m_Flags`, `m_Delay`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SwitchTile {
    /// Switch number this tile belongs to. Mirrors `CSwitchTile::m_Number`.
    pub number: u8,
    /// Switch tile type (door tile id, or a switch-control type such as
    /// `TILE_SWITCHTIMEDOPEN`). Mirrors `CSwitchTile::m_Type`.
    pub kind: u8,
    /// Rotation/flip bits for the underlying tile (door). Mirrors `CSwitchTile::m_Flags`.
    pub flags: u8,
    /// Type-dependent delay/parameter (seconds, jump count, weapon id, ...). Mirrors
    /// `CSwitchTile::m_Delay`.
    pub delay: u8,
}

/// A single tune layer record. Mirrors `CTuneTile` (`mapitems.h`): `m_Number`, `m_Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TuneTile {
    /// Tune zone number (`0` = no zone). Mirrors `CTuneTile::m_Number`.
    pub number: u8,
    /// Always `TILE_TUNE` when `number != 0`. Mirrors `CTuneTile::m_Type`.
    pub kind: u8,
}

// --- Tile ids (game/front layer `CTile::m_Index`), from DDNet 20.1 `src/game/mapitems.h`. -----
// Only the ids the synthetic recipes and the oracle actually need are named; see
// `docs/research/ddnet-physics.md` §7 for the full table.

/// Empty tile. `mapitems.h` `TILE_AIR`.
pub const TILE_AIR: u8 = 0;
/// Solid, hookable wall. `mapitems.h` `TILE_SOLID`.
pub const TILE_SOLID: u8 = 1;
/// Kills on touch (game/front). `mapitems.h` `TILE_DEATH`.
pub const TILE_DEATH: u8 = 2;
/// Solid, not hookable. `mapitems.h` `TILE_NOHOOK`.
pub const TILE_NOHOOK: u8 = 3;
/// Blocks laser (game/front); also the upper bound of `GetTile()`'s "solid" range
/// `[TILE_SOLID, TILE_NOLASER]`. `mapitems.h` `TILE_NOLASER`.
pub const TILE_NOLASER: u8 = 4;
/// Switch-layer "N jumps" tile (`CSwitchTile::m_Delay` = jump count, `255` = unlimited).
/// `mapitems.h` `TILE_JUMP`.
pub const TILE_JUMP: u8 = 7;
/// Hook-through: lets a hook that is already past cut through; body still solid.
/// `mapitems.h` `TILE_THROUGH_CUT`.
pub const TILE_THROUGH_CUT: u8 = 5;
/// Hook-through: the tile *offset* from a `THROUGH_CUT`/`THROUGH_DIR` tile in the hook's
/// direction of travel that actually lets the hook pass. `mapitems.h` `TILE_THROUGH`.
pub const TILE_THROUGH: u8 = 6;
/// Freeze on touch (game layer) / freeze indicator (front layer). `mapitems.h` `TILE_FREEZE`.
pub const TILE_FREEZE: u8 = 9;
/// Tele-in, "evil" variant (drops velocity, releases hook). Also a tele layer `m_Type` value.
/// `mapitems.h` `TILE_TELEINEVIL`.
pub const TILE_TELEINEVIL: u8 = 10;
/// Unfreeze on touch. `mapitems.h` `TILE_UNFREEZE`.
pub const TILE_UNFREEZE: u8 = 11;
/// Deep freeze on touch (can't be unfrozen by `UNFREEZE`, only by leaving deep-freeze area or
/// waiting it out). `mapitems.h` `TILE_DFREEZE`.
pub const TILE_DFREEZE: u8 = 12;
/// Ends deep freeze. `mapitems.h` `TILE_DUNFREEZE`.
pub const TILE_DUNFREEZE: u8 = 13;
/// Weapon tele-in: a projectile that flies into this tile teleports to the matching tele-out.
/// Also a tele layer `m_Type` value. `mapitems.h` `TILE_TELEINWEAPON`.
pub const TILE_TELEINWEAPON: u8 = 14;
/// Hook tele-in: a hook that flies into this tile teleports to the matching tele-out. Also a
/// tele layer `m_Type` value. `mapitems.h` `TILE_TELEINHOOK`.
pub const TILE_TELEINHOOK: u8 = 15;
/// Grants a wall-jump when touching a hookable wall while airborne (`character.cpp`, out of
/// Oracle A's core-only scope, but a `CCollision::IsWallJump` getter is still ported).
/// `mapitems.h` `TILE_WALLJUMP`.
pub const TILE_WALLJUMP: u8 = 16;
/// First tile id in the `TIME_CHECKPOINT` range (`CCollision::IsTimeCheckpoint`/
/// `IsFrontTimeCheckpoint`). `mapitems.h` `TILE_TIME_CHECKPOINT_FIRST`.
pub const TILE_TIME_CHECKPOINT_FIRST: u8 = 35;
/// Last tile id in the `TIME_CHECKPOINT` range. `mapitems.h` `TILE_TIME_CHECKPOINT_LAST`.
pub const TILE_TIME_CHECKPOINT_LAST: u8 = 59;
/// One-way stopper; which way it blocks depends on rotation flags. `mapitems.h` `TILE_STOP`.
pub const TILE_STOP: u8 = 60;
/// Two-way stopper (blocks a pair of opposite directions). `mapitems.h` `TILE_STOPS`.
pub const TILE_STOPS: u8 = 61;
/// All-way stopper. `mapitems.h` `TILE_STOPA`.
pub const TILE_STOPA: u8 = 62;
/// "Mover" tile (moves lasers/plasma along a fixed direction, `CCollision::MoverSpeed`) — normal
/// speed. `mapitems.h` `TILE_CP`.
pub const TILE_CP: u8 = 64;
/// Mover tile, "fast" variant (4x `TILE_CP`'s speed). `mapitems.h` `TILE_CP_F`.
pub const TILE_CP_F: u8 = 65;
/// Tele-in, normal variant (velocity kept). Also a tele layer `m_Type` value.
/// `mapitems.h` `TILE_TELEIN`.
pub const TILE_TELEIN: u8 = 26;
/// Tele-out destination. Also a tele layer `m_Type` value. `mapitems.h` `TILE_TELEOUT`.
pub const TILE_TELEOUT: u8 = 27;
/// Old-style speedup tile id (`CSpeedupTile::m_Type` value; superseded by [`TILE_SPEED_BOOST`]
/// but still used by many maps). `mapitems.h` `TILE_SPEED_BOOST_OLD`.
pub const TILE_SPEED_BOOST_OLD: u8 = 28;
/// Checkpoint tele-in / current-style speedup tile id (`CSpeedupTile::m_Type`); the numeric
/// value is shared between two unrelated enumerators in `mapitems.h` (`TILE_TELECHECK` and
/// `TILE_SPEED_BOOST`) — which one applies depends on which layer the value is read from (tele
/// vs speedup), never on the raw value alone. `mapitems.h` `TILE_TELECHECK` / `TILE_SPEED_BOOST`.
pub const TILE_TELECHECK: u8 = 29;
/// Same numeric id as [`TILE_TELECHECK`]; see that constant's doc comment.
pub const TILE_SPEED_BOOST: u8 = 29;
/// Checkpoint tele-out. Tele layer `m_Type` value. `mapitems.h` `TILE_TELECHECKOUT`.
pub const TILE_TELECHECKOUT: u8 = 30;
/// Checkpoint tele-in. Tele layer `m_Type` value. `mapitems.h` `TILE_TELECHECKIN`.
pub const TILE_TELECHECKIN: u8 = 31;
/// Hook-through, all rotations/hook directions (both hook and hammer body pass through).
/// `mapitems.h` `TILE_THROUGH_ALL`.
pub const TILE_THROUGH_ALL: u8 = 66;
/// Directional hook-through; which direction depends on rotation flags.
/// `mapitems.h` `TILE_THROUGH_DIR`.
pub const TILE_THROUGH_DIR: u8 = 67;
/// Checkpoint tele-in, "evil" variant. Tele layer `m_Type` value.
/// `mapitems.h` `TILE_TELECHECKINEVIL`.
pub const TILE_TELECHECKINEVIL: u8 = 63;

// --- Switch-layer control tile ids (`CSwitchTile::m_Type`, `mapitems.h`), plus the range
// `CCollision::Init` keeps verbatim in `m_pSwitch[i].m_Type` (everything else is zeroed). -------

/// Switch-layer subtract-time control tile. `mapitems.h` `TILE_SUBTRACT_TIME`.
pub const TILE_SUBTRACT_TIME: u8 = 95;
/// Enables the tele gun pickup. `mapitems.h` `TILE_ALLOW_TELE_GUN`.
pub const TILE_ALLOW_TELE_GUN: u8 = 98;
/// Enables the blue ("freeze") tele gun pickup. `mapitems.h` `TILE_ALLOW_BLUE_TELE_GUN`.
pub const TILE_ALLOW_BLUE_TELE_GUN: u8 = 99;
/// Re-enables hooking other players (switch-layer). Also the upper bound of the switch-type
/// range `CCollision::Init` keeps (`m_Type <= TILE_NPH_ENABLE`). `mapitems.h` `TILE_NPH_ENABLE`.
pub const TILE_NPH_ENABLE: u8 = 107;

// --- `CCollision::TileExists`'s two "interesting effect" game/front ranges (`mapitems.h`). ------

/// Upper bound (inclusive) of the first `TileExists` range that starts at [`TILE_FREEZE`].
/// `mapitems.h` `TILE_TELE_LASER_DISABLE`.
pub const TILE_TELE_LASER_DISABLE: u8 = 129;
/// Lower bound of `TileExists`'s second range (live freeze). `mapitems.h` `TILE_LFREEZE`.
pub const TILE_LFREEZE: u8 = 144;
/// Upper bound (inclusive) of `TileExists`'s second range. `mapitems.h` `TILE_LUNFREEZE`.
pub const TILE_LUNFREEZE: u8 = 145;

// --- Tile flags (`CTile::m_Flags` / `CSwitchTile::m_Flags`), `mapitems.h`. --------------------

/// Flip around the vertical axis. `mapitems.h` `TILEFLAG_XFLIP`.
pub const TILEFLAG_XFLIP: u8 = 1 << 0;
/// Flip around the horizontal axis. `mapitems.h` `TILEFLAG_YFLIP`.
pub const TILEFLAG_YFLIP: u8 = 1 << 1;
/// Opaque (rendering only; irrelevant to physics). `mapitems.h` `TILEFLAG_OPAQUE`.
pub const TILEFLAG_OPAQUE: u8 = 1 << 2;
/// Rotate 90°. `mapitems.h` `TILEFLAG_ROTATE`.
pub const TILEFLAG_ROTATE: u8 = 1 << 3;

/// No rotation. `mapitems.h` `ROTATION_0`.
pub const ROTATION_0: u8 = 0;
/// Rotated 90°. `mapitems.h` `ROTATION_90`.
pub const ROTATION_90: u8 = TILEFLAG_ROTATE;
/// Rotated 180° (expressed as XFLIP|YFLIP, not the rotate bit — this is how DDNet's editor and
/// `collision.cpp`'s `GetMoveRestrictionsRaw` encode it). `mapitems.h` `ROTATION_180`.
pub const ROTATION_180: u8 = TILEFLAG_XFLIP | TILEFLAG_YFLIP;
/// Rotated 270°. `mapitems.h` `ROTATION_270`.
pub const ROTATION_270: u8 = TILEFLAG_XFLIP | TILEFLAG_YFLIP | TILEFLAG_ROTATE;

/// The four "plain" rotation-flag encodings, in `0/90/180/270` order. Convenience for recipes
/// that need "every rotation" of a hook-through tile (`TILE_THROUGH_DIR`) or a hammer/laser
/// stopper. **Not** every rotation-flag byte `TILE_STOP`/`TILE_STOPS` distinguish, though: see
/// [`ROTATIONS_YFLIP`] — `collision.cpp`'s `GetMoveRestrictionsRaw` has 8 distinct `case` labels
/// for `TILE_STOP` (this array's 4, plus the 4 `TILEFLAG_YFLIP ^ ROTATION_*` ones below), and 8
/// for `TILE_STOPS` (falling into the same 2 behaviors either way, but as 8 separate branches a
/// coverage tool sees as 8 lines) — review round 1 finding F1.
pub const ROTATIONS: [u8; 4] = [ROTATION_0, ROTATION_90, ROTATION_180, ROTATION_270];

/// The other 4 rotation-flag bytes `TILE_STOP`/`TILE_STOPS` recognize as `case` labels in
/// `collision.cpp`'s `GetMoveRestrictionsRaw`, alongside [`ROTATIONS`]:
/// `TILEFLAG_YFLIP ^ ROTATION_{0,90,180,270}` = `{2, 10, 1, 9}`. Same order (0°/90°/180°/270°)
/// as [`ROTATIONS`], and — for `TILE_STOP` — the same 4 *behaviors* (`CANTMOVE_UP`/`RIGHT`/
/// `DOWN`/`LEFT`, the mirror image of `ROTATIONS`' `DOWN`/`LEFT`/`UP`/`RIGHT`); for `TILE_STOPS`
/// each maps to the same `CANTMOVE_*` pair its `ROTATIONS` counterpart 180° away does. DDNet's
/// map editor can produce either encoding for the same visual rotation depending on how a tile
/// was flipped/rotated interactively, so a real map (and this crate's `front` recipe) exercises
/// both.
pub const ROTATIONS_YFLIP: [u8; 4] = [
    TILEFLAG_YFLIP ^ ROTATION_0,
    TILEFLAG_YFLIP ^ ROTATION_90,
    TILEFLAG_YFLIP ^ ROTATION_180,
    TILEFLAG_YFLIP ^ ROTATION_270,
];

// --- Tilemap layer flags (`CMapItemLayerTilemap::m_Flags`), `mapitems.h`. ---------------------

/// Marks a tiles layer as the game layer. `mapitems.h` `TILESLAYERFLAG_GAME`.
pub const TILESLAYERFLAG_GAME: u32 = 1 << 0;
/// Marks a tiles layer as the tele layer. `mapitems.h` `TILESLAYERFLAG_TELE`.
pub const TILESLAYERFLAG_TELE: u32 = 1 << 1;
/// Marks a tiles layer as the speedup layer. `mapitems.h` `TILESLAYERFLAG_SPEEDUP`.
pub const TILESLAYERFLAG_SPEEDUP: u32 = 1 << 2;
/// Marks a tiles layer as the front layer. `mapitems.h` `TILESLAYERFLAG_FRONT`.
pub const TILESLAYERFLAG_FRONT: u32 = 1 << 3;
/// Marks a tiles layer as the switch layer. `mapitems.h` `TILESLAYERFLAG_SWITCH`.
pub const TILESLAYERFLAG_SWITCH: u32 = 1 << 4;
/// Marks a tiles layer as the tune layer. `mapitems.h` `TILESLAYERFLAG_TUNE`.
pub const TILESLAYERFLAG_TUNE: u32 = 1 << 5;

#[cfg(test)]
mod tests {
    use super::*;

    fn small_map() -> MapData {
        MapData {
            width: 2,
            height: 2,
            game: vec![Tile::default(); 4],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        }
    }

    #[test]
    fn cell_count_is_width_times_height() {
        assert_eq!(small_map().cell_count(), 4);
    }

    #[test]
    fn validate_accepts_correctly_sized_layers() {
        let mut map = small_map();
        map.front = Some(vec![Tile::default(); 4]);
        map.tele = Some(vec![TeleTile::default(); 4]);
        assert!(map.validate().is_ok());
    }

    #[test]
    fn validate_rejects_wrong_length_game_layer() {
        let mut map = small_map();
        map.game.pop();
        let err = map.validate().unwrap_err();
        assert_eq!(
            err,
            MapDataError::LayerLengthMismatch {
                layer: "game",
                expected: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn validate_rejects_wrong_length_optional_layer() {
        let mut map = small_map();
        map.speedup = Some(vec![SpeedupTile::default(); 3]);
        let err = map.validate().unwrap_err();
        assert_eq!(
            err,
            MapDataError::LayerLengthMismatch {
                layer: "speedup",
                expected: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn rotation_constants_match_mapitems_h() {
        // ROTATION_180 is XFLIP|YFLIP, not "flip the rotate bit twice" — a common porting
        // mistake, so this is pinned explicitly.
        assert_eq!(ROTATION_180, TILEFLAG_XFLIP | TILEFLAG_YFLIP);
        assert_eq!(ROTATION_270, TILEFLAG_XFLIP | TILEFLAG_YFLIP | TILEFLAG_ROTATE);
        assert_eq!(ROTATIONS, [0, 8, 3, 11]);
    }

    #[test]
    fn rotations_yflip_matches_collision_cpp_case_labels() {
        // Computed by hand from `GetMoveRestrictionsRaw`'s `TILEFLAG_YFLIP ^ ROTATION_*` case
        // labels (collision.cpp); cross-checked independently with a standalone `rustc` snippet
        // during review round 1 (finding F1).
        assert_eq!(ROTATIONS_YFLIP, [2, 10, 1, 9]);
        // No overlap with the plain rotations — otherwise `front`'s recipe test that counts
        // "one cell per distinct flag byte" would silently under-count.
        for v in ROTATIONS_YFLIP {
            assert!(!ROTATIONS.contains(&v), "{v} should not also appear in ROTATIONS");
        }
    }
}
