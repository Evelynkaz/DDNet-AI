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

// --- Task 1.6: `CCharacter::HandleTiles`/`HandleSkippableTiles`/`DDRaceTick` game/front-layer
// tile ids and switch-layer control tile ids (`src/game/mapitems.h`, verbatim numeric values). --

/// Enables endless-hook for a character touching it (game/front). `mapitems.h`
/// `TILE_EHOOK_ENABLE`.
pub const TILE_EHOOK_ENABLE: u8 = 17;
/// Disables endless-hook. `mapitems.h` `TILE_EHOOK_DISABLE`.
pub const TILE_EHOOK_DISABLE: u8 = 18;
/// Re-enables all "can't hit others" flags at once (game/front). `mapitems.h` `TILE_HIT_ENABLE`.
pub const TILE_HIT_ENABLE: u8 = 19;
/// Disables all "can hit others" flags at once (game/front). `mapitems.h` `TILE_HIT_DISABLE`.
pub const TILE_HIT_DISABLE: u8 = 20;
/// Enters a solo part (game/front; dispatched by the game controller, not `CCharacter` itself —
/// `gamemodes/ddnet.cpp:108`). `mapitems.h` `TILE_SOLO_ENABLE`.
pub const TILE_SOLO_ENABLE: u8 = 21;
/// Leaves a solo part (game/front). `mapitems.h` `TILE_SOLO_DISABLE`. Numerically identical to
/// [`TILE_SWITCHTIMEDOPEN`] — the two are never ambiguous in practice: one is only ever read from
/// the game/front layer, the other only from the switch layer.
pub const TILE_SOLO_DISABLE: u8 = 22;
/// Switch-layer: opens switch `Number` for `Delay` seconds, then it auto-closes (a timed
/// countdown pair with [`TILE_SWITCHTIMEDCLOSE`]). `mapitems.h` `TILE_SWITCHTIMEDOPEN`.
pub const TILE_SWITCHTIMEDOPEN: u8 = 22;
/// Switch-layer: closes switch `Number` for `Delay` seconds, then it auto-reopens. `mapitems.h`
/// `TILE_SWITCHTIMEDCLOSE`.
pub const TILE_SWITCHTIMEDCLOSE: u8 = 23;
/// Switch-layer: opens switch `Number` until explicitly closed. `mapitems.h` `TILE_SWITCHOPEN`.
pub const TILE_SWITCHOPEN: u8 = 24;
/// Switch-layer: closes switch `Number` until explicitly opened. `mapitems.h` `TILE_SWITCHCLOSE`.
pub const TILE_SWITCHCLOSE: u8 = 25;
/// Refills the character's jump count immediately (game/front), edge-triggered via
/// `CCharacter::m_LastRefillJumps`. `mapitems.h` `TILE_REFILL_JUMPS`.
pub const TILE_REFILL_JUMPS: u8 = 32;
/// Race-timer start (also read from the 4 diagonal "sensitivity" positions in
/// `IGameController::HandleCharacterTiles`). `mapitems.h` `TILE_START`.
pub const TILE_START: u8 = 33;
/// Race-timer finish. `mapitems.h` `TILE_FINISH`.
pub const TILE_FINISH: u8 = 34;
/// DDRace-team tile: unlocks the character's current team (game/front). `mapitems.h`
/// `TILE_UNLOCK_TEAM`.
pub const TILE_UNLOCK_TEAM: u8 = 76;
/// Switch-layer: sets `+Minutes*60+Seconds` onto the character's race timer, once per visit
/// (edge-triggered via `CCharacter::m_LastPenalty`), and propagates to every other character
/// currently on the same non-flock team. `mapitems.h` `TILE_ADD_TIME`.
pub const TILE_ADD_TIME: u8 = 79;
/// Game/front: disables player-vs-player collision for the toucher. `mapitems.h`
/// `TILE_NPC_DISABLE`.
pub const TILE_NPC_DISABLE: u8 = 88;
/// Game/front: takes away unlimited air jumps. `mapitems.h` `TILE_UNLIMITED_JUMPS_DISABLE`.
pub const TILE_UNLIMITED_JUMPS_DISABLE: u8 = 89;
/// Game/front: takes away the jetpack gun. `mapitems.h` `TILE_JETPACK_DISABLE`.
pub const TILE_JETPACK_DISABLE: u8 = 90;
/// Game/front: disables hooking other players. `mapitems.h` `TILE_NPH_DISABLE`.
pub const TILE_NPH_DISABLE: u8 = 91;
/// Enables the tele-gun pickup for the gun weapon (game/front). `mapitems.h`
/// `TILE_TELE_GUN_ENABLE`.
pub const TILE_TELE_GUN_ENABLE: u8 = 96;
/// Disables the gun-weapon tele-gun. `mapitems.h` `TILE_TELE_GUN_DISABLE`.
pub const TILE_TELE_GUN_DISABLE: u8 = 97;
/// Game/front: re-enables player-vs-player collision. `mapitems.h` `TILE_NPC_ENABLE`.
pub const TILE_NPC_ENABLE: u8 = 104;
/// Game/front: grants unlimited air jumps. `mapitems.h` `TILE_UNLIMITED_JUMPS_ENABLE`.
pub const TILE_UNLIMITED_JUMPS_ENABLE: u8 = 105;
/// Game/front: grants the jetpack gun. `mapitems.h` `TILE_JETPACK_ENABLE`.
pub const TILE_JETPACK_ENABLE: u8 = 106;
/// Enables the tele-gun pickup for the grenade weapon. `mapitems.h`
/// `TILE_TELE_GRENADE_ENABLE`.
pub const TILE_TELE_GRENADE_ENABLE: u8 = 112;
/// Disables the grenade-weapon tele-gun. `mapitems.h` `TILE_TELE_GRENADE_DISABLE`.
pub const TILE_TELE_GRENADE_DISABLE: u8 = 113;
/// Enables the tele-gun pickup for the laser weapon. `mapitems.h` `TILE_TELE_LASER_ENABLE`.
pub const TILE_TELE_LASER_ENABLE: u8 = 128;

// --- Task 1.6: `IGameController::OnEntity`/`CGameContext::CreateAllEntities` map-fixture and
// global-tile-flag ids (`src/game/mapitems.h`); `ENTITY_OFFSET` is the raw-tile-index bias
// (`GameIndex - ENTITY_OFFSET` in `CreateAllEntities`) every `ENTITY_*` id below is relative to. -

/// `CreateAllEntities`'s bias: a raw game/front/switch tile index `>= ENTITY_OFFSET` encodes an
/// `ENTITY_*` id as `raw - ENTITY_OFFSET`. `mapitems.h` `ENTITY_OFFSET` (`255 - 16*4`).
pub const ENTITY_OFFSET: u8 = 255 - 16 * 4;
/// A default-team spawn point (only collected when scanning the *initial* map load). `mapitems.h`
/// `ENTITY_SPAWN`.
pub const ENTITY_SPAWN: u8 = 1;
/// A red-team spawn point. `mapitems.h` `ENTITY_SPAWN_RED`.
pub const ENTITY_SPAWN_RED: u8 = 2;
/// A blue-team spawn point; also the upper end of the `ENTITY_SPAWN..=ENTITY_SPAWN_BLUE` scan
/// range. `mapitems.h` `ENTITY_SPAWN_BLUE`.
pub const ENTITY_SPAWN_BLUE: u8 = 3;
/// Armor pickup. `mapitems.h` `ENTITY_ARMOR_1`.
pub const ENTITY_ARMOR_1: u8 = 6;
/// "Health" pickup — in DDRace this is `POWERUP_FREEZE` (freezes on touch), not a health refill.
/// `mapitems.h` `ENTITY_HEALTH_1`.
pub const ENTITY_HEALTH_1: u8 = 7;
/// Which tiles lie inside the freeze reach of a heart pickup (`ENTITY_HEALTH_1`, game or front layer): a tee
/// whose centre is within `PICKUP_PROXIMITY_RADIUS (20) + 28 = 48` px of the pickup's centre is frozen
/// (`pickup.cpp`, `gamecontroller.cpp:288`). For a tee standing at a tile centre that is the 3x3 block
/// of tiles around the heart (`32 * sqrt(2) = 45 < 48`, `64 > 48`). Row-major, `width * height`.
///
/// **Data for the tile-based helpers only** (navigation, the shield's "rests in freeze", the bot's
/// hazard gate): the physics itself handles pickups as entities and never reads this.
pub fn pickup_freeze_mask(map: &MapData) -> Vec<bool> {
    let (w, h) = (map.width as i32, map.height as i32);
    let mut mask = vec![false; (w.max(0) * h.max(0)) as usize];
    let heart = ENTITY_OFFSET + ENTITY_HEALTH_1;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let here = map.game.get(i).is_some_and(|t| t.index == heart)
                || map
                    .front
                    .as_ref()
                    .and_then(|f| f.get(i))
                    .is_some_and(|t| t.index == heart);
            if !here {
                continue;
            }
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < w && ny < h {
                        mask[(ny * w + nx) as usize] = true;
                    }
                }
            }
        }
    }
    mask
}

/// Shotgun pickup. `mapitems.h` `ENTITY_WEAPON_SHOTGUN`.
pub const ENTITY_WEAPON_SHOTGUN: u8 = 8;
/// Grenade launcher pickup. `mapitems.h` `ENTITY_WEAPON_GRENADE`.
pub const ENTITY_WEAPON_GRENADE: u8 = 9;
/// Ninja power-up pickup. `mapitems.h` `ENTITY_POWERUP_NINJA`.
pub const ENTITY_POWERUP_NINJA: u8 = 10;
/// Laser rifle pickup. `mapitems.h` `ENTITY_WEAPON_LASER`.
pub const ENTITY_WEAPON_LASER: u8 = 11;
/// Lower end of the light-rotation-speed marker range (`ENTITY_LASER_FAST_CCW..=
/// ENTITY_LASER_FAST_CW`) — stage B (`CLight`); kept here only so [`ENTITY_LASER_SHORT`]'s
/// neighbor-scan range is unambiguous. `mapitems.h` `ENTITY_LASER_FAST_CCW`.
pub const ENTITY_LASER_FAST_CCW: u8 = 12;
/// A non-rotating light source (`AngularSpeed == 0`) — the `Ind == 0` case of the
/// `ENTITY_LASER_FAST_CCW..=ENTITY_LASER_FAST_CW` range. Stage B (`CLight`); named because task
/// 1.6's cut-rule detector treats *only* this case as static-geometry-checkable (see
/// `docs/formats.md`). `mapitems.h` `ENTITY_LASER_STOP`.
pub const ENTITY_LASER_STOP: u8 = 15;
/// Counter-clockwise light, normal speed (`pi / 180` per step). `mapitems.h`
/// `ENTITY_LASER_NORMAL_CCW`.
pub const ENTITY_LASER_NORMAL_CCW: u8 = 13;
/// Counter-clockwise light, slow speed (`pi / 360` per step). `mapitems.h`
/// `ENTITY_LASER_SLOW_CCW`.
pub const ENTITY_LASER_SLOW_CCW: u8 = 14;
/// Clockwise light, slow speed. `mapitems.h` `ENTITY_LASER_SLOW_CW`.
pub const ENTITY_LASER_SLOW_CW: u8 = 16;
/// Clockwise light, normal speed. `mapitems.h` `ENTITY_LASER_NORMAL_CW`.
pub const ENTITY_LASER_NORMAL_CW: u8 = 17;
/// Upper end of the light-rotation-speed marker range. `mapitems.h` `ENTITY_LASER_FAST_CW`.
pub const ENTITY_LASER_FAST_CW: u8 = 18;
/// Lower end of the door/light length-marker range read from a neighbor cell
/// (`ENTITY_LASER_SHORT..=ENTITY_LASER_LONG`, `Length = 32*3 + 32*(marker - ENTITY_LASER_SHORT)*3`
/// in `IGameController::OnEntity`). `mapitems.h` `ENTITY_LASER_SHORT`.
pub const ENTITY_LASER_SHORT: u8 = 19;
/// Medium door/light length marker. `mapitems.h` `ENTITY_LASER_MEDIUM`.
pub const ENTITY_LASER_MEDIUM: u8 = 20;
/// Upper end of the door/light length-marker range. `mapitems.h` `ENTITY_LASER_LONG`.
pub const ENTITY_LASER_LONG: u8 = 21;
/// Lower end of the light "closing" speed-marker range read two cells out (`aSides2[i]`):
/// the beam shrinks and grows again (`CLight::m_Speed`, `m_CurveLength = m_Length` at the start).
/// `mapitems.h` `ENTITY_LASER_C_SLOW`.
pub const ENTITY_LASER_C_SLOW: u8 = 22;
/// Normal-speed "closing" light marker. `mapitems.h` `ENTITY_LASER_C_NORMAL`.
pub const ENTITY_LASER_C_NORMAL: u8 = 23;
/// Upper end of the light "closing" speed-marker range. `mapitems.h` `ENTITY_LASER_C_FAST`.
pub const ENTITY_LASER_C_FAST: u8 = 24;
/// Lower end of the light "opening" speed-marker range (the beam starts at length `0`).
/// `mapitems.h` `ENTITY_LASER_O_SLOW`.
pub const ENTITY_LASER_O_SLOW: u8 = 25;
/// Normal-speed "opening" light marker. `mapitems.h` `ENTITY_LASER_O_NORMAL`.
pub const ENTITY_LASER_O_NORMAL: u8 = 26;
/// Upper end of the light "opening" speed-marker range. `mapitems.h` `ENTITY_LASER_O_FAST`.
pub const ENTITY_LASER_O_FAST: u8 = 27;
/// Turret (`CGun`) variant marker: explosive, not freezing. `mapitems.h` `ENTITY_PLASMAE`.
pub const ENTITY_PLASMAE: u8 = 29;
/// Turret variant marker: freezing, not explosive. `mapitems.h` `ENTITY_PLASMAF`.
pub const ENTITY_PLASMAF: u8 = 30;
/// Turret variant marker: freezing and explosive. `mapitems.h` `ENTITY_PLASMA`.
pub const ENTITY_PLASMA: u8 = 31;
/// Turret variant marker: neither freezing nor explosive. `mapitems.h` `ENTITY_PLASMAU`.
pub const ENTITY_PLASMAU: u8 = 32;
/// "Crazy shotgun", rotation-flag variant: spawns a permanently-bouncing `WEAPON_SHOTGUN`
/// `CProjectile` (explosive) at map-load time. Not a player-fired shotgun (that's a `CLaser`,
/// stage B) — this map fixture is a genuine `CProjectile`, in Stage A's scope. `mapitems.h`
/// `ENTITY_CRAZY_SHOTGUN_EX`.
pub const ENTITY_CRAZY_SHOTGUN_EX: u8 = 33;
/// "Crazy shotgun", `TILEFLAG_ROTATE`/`XFLIP|YFLIP`-variant: spawns a permanently-bouncing
/// `WEAPON_SHOTGUN` `CProjectile` (not explosive, freezing). `mapitems.h`
/// `ENTITY_CRAZY_SHOTGUN`.
pub const ENTITY_CRAZY_SHOTGUN: u8 = 34;
/// Shotgun-ammo armor pickup (drops the shotgun on touch). `mapitems.h` `ENTITY_ARMOR_SHOTGUN`.
pub const ENTITY_ARMOR_SHOTGUN: u8 = 35;
/// Grenade-ammo armor pickup. `mapitems.h` `ENTITY_ARMOR_GRENADE`.
pub const ENTITY_ARMOR_GRENADE: u8 = 36;
/// Ninja armor pickup (resets an active ninja's remaining time/velocity state). `mapitems.h`
/// `ENTITY_ARMOR_NINJA`.
pub const ENTITY_ARMOR_NINJA: u8 = 37;
/// Laser-ammo armor pickup. `mapitems.h` `ENTITY_ARMOR_LASER`.
pub const ENTITY_ARMOR_LASER: u8 = 38;
/// Lower end of the weak-dragger marker range (`ENTITY_DRAGGER_WEAK..=ENTITY_DRAGGER_STRONG`,
/// axis-aligned). Stage B (`CDragger`). `mapitems.h` `ENTITY_DRAGGER_WEAK`.
pub const ENTITY_DRAGGER_WEAK: u8 = 42;
/// Normal-strength axis-aligned dragger (strength `2`). `mapitems.h` `ENTITY_DRAGGER_NORMAL`.
pub const ENTITY_DRAGGER_NORMAL: u8 = 43;
/// Upper end of the axis-aligned dragger marker range. `mapitems.h` `ENTITY_DRAGGER_STRONG`.
pub const ENTITY_DRAGGER_STRONG: u8 = 44;
/// Lower end of the diagonal ("NW", `IgnoreWalls`) dragger marker range. `mapitems.h`
/// `ENTITY_DRAGGER_WEAK_NW`.
pub const ENTITY_DRAGGER_WEAK_NW: u8 = 45;
/// Normal-strength ignore-walls dragger. `mapitems.h` `ENTITY_DRAGGER_NORMAL_NW`.
pub const ENTITY_DRAGGER_NORMAL_NW: u8 = 46;
/// Upper end of the diagonal dragger marker range. `mapitems.h` `ENTITY_DRAGGER_STRONG_NW`.
pub const ENTITY_DRAGGER_STRONG_NW: u8 = 47;
/// A door (switch-layer only: `SwitchType - ENTITY_OFFSET == ENTITY_DOOR`) — the door's own
/// length/direction come from a neighboring switch-layer cell in the
/// [`ENTITY_LASER_SHORT`]/[`ENTITY_LASER_LONG`] range; `Number` is the switch tile's own
/// `m_Number`. Stage A (`CDoor` collision). `mapitems.h` `ENTITY_DOOR`.
pub const ENTITY_DOOR: u8 = 49;

// --- Task 1.6: `CreateAllEntities`'s map-wide global-effect tile ids (read from *any* game/front
// cell, applied once at map load — `mapitems.h`, `gamecontext.cpp:4288-4312`). ------------------

/// Anywhere on the game/front layer: sets `sv_old_laser = 1` for the whole map. `mapitems.h`
/// `TILE_OLDLASER`.
pub const TILE_OLDLASER: u8 = 71;
/// Anywhere on the game/front layer: sets `player_collision` tuning to `0` for every zone.
/// `mapitems.h` `TILE_NPC`.
pub const TILE_NPC: u8 = 72;
/// Anywhere on the game/front layer: sets `sv_endless_drag = 1` for the whole map. `mapitems.h`
/// `TILE_EHOOK`.
pub const TILE_EHOOK: u8 = 73;
/// Anywhere on the game/front layer: sets `sv_hit = 0` for the whole map. `mapitems.h`
/// `TILE_NOHIT`.
pub const TILE_NOHIT: u8 = 74;
/// Anywhere on the game/front layer: sets `player_hooking` tuning to `0` for every zone.
/// `mapitems.h` `TILE_NPH`.
pub const TILE_NPH: u8 = 75;

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
    #[test]
    fn a_heart_pickup_freezes_the_3x3_tiles_around_it_in_either_layer() {
        let (w, h) = (9usize, 7usize);
        let mut game = vec![Tile::default(); w * h];
        game[2 * w + 2].index = ENTITY_OFFSET + ENTITY_HEALTH_1;
        let mut front = vec![Tile::default(); w * h];
        front[4 * w + 7].index = ENTITY_OFFSET + ENTITY_HEALTH_1;
        // An armor pickup (index 197) freezes nothing.
        game[5 * w + 5].index = ENTITY_OFFSET + ENTITY_ARMOR_1;
        let map = MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: Some(front),
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        let mask = pickup_freeze_mask(&map);
        let on: Vec<(usize, usize)> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| mask[y * w + x])
            .collect();
        assert_eq!(on.len(), 9 + 9, "two hearts, a 3x3 block each: {on:?}");
        assert!(mask[2 * w + 2] && mask[w + 1] && mask[3 * w + 3]);
        assert!(!mask[2 * w + 4], "two tiles away");
        assert!(mask[4 * w + 7] && mask[3 * w + 8]);
        assert!(!mask[5 * w + 5], "armor is not a hazard");
    }

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
