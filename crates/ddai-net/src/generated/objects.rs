// GENERATED — do not edit by hand.
//
// Produced by `tools/ddnet-protocol-gen/generate.py` from DDNet's own protocol description
// (`datasrc/network.py` + `datasrc/datatypes.py`), commit c9d208138f85755521f16a0096b6fe036c5c8698 ("20.1").
// Regenerate with (from the repository root):
//
//   python3 tools/ddnet-protocol-gen/generate.py ~/aiddnet/build/ddnet-20.1/src
//
// Re-running against the same pinned commit's tree reproduces these files byte-for-byte (the
// script formats its own output with `rustfmt`). See `tools/ddnet-protocol-gen/README.md`.

//! Snapshot objects and events (`NETOBJTYPE_*`/`NETEVENTTYPE_*`) from `datasrc/network.py`.
//!
//! Non-UUID types 1..=20 are assigned ids in exactly the order DDNet's own
//! `datasrc/compile.py` assigns them (list order, `ex is None` only) — this order is part
//! of the wire format (it is the raw `int` stored in a snapshot item's key) and must never
//! change independent of DDNet itself. UUID (`ex`) types carry no fixed numeric id on the
//! wire at all (see `crate::snapshot`); [`EX_NAMES`] lists their UUID names in `network.py`
//! declaration order for [`crate::uuid::UuidRegistry::from_names`].

/// `CNetObj_PlayerInput` (`NETOBJTYPE_PLAYERINPUT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerInput {
    pub direction: i32,
    pub target_x: i32,
    pub target_y: i32,
    pub jump: i32,
    pub fire: i32,
    pub hook: i32,
    pub player_flags: i32,
    pub wanted_weapon: i32,
    pub next_weapon: i32,
    pub prev_weapon: i32,
}

impl PlayerInput {
    /// `NETOBJTYPE_PLAYERINPUT` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 1;
    /// Static size in `i32`s (`sizeof(CNetObj_PlayerInput) / 4`).
    pub const SIZE_INTS: usize = 10;
}

impl PlayerInput {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let direction = unpacker.get_uncompressed_int();
        let target_x = unpacker.get_uncompressed_int();
        let target_y = unpacker.get_uncompressed_int();
        let jump = unpacker.get_uncompressed_int();
        let fire = unpacker.get_uncompressed_int();
        let hook = unpacker.get_uncompressed_int();
        let mut player_flags = unpacker.get_uncompressed_int();
        if player_flags < 0 {
            player_flags = 0;
            corrections += 1;
        } else if player_flags > 256 {
            player_flags = 256;
            corrections += 1;
        }
        let wanted_weapon = unpacker.get_uncompressed_int();
        let next_weapon = unpacker.get_uncompressed_int();
        let prev_weapon = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((
            PlayerInput {
                direction,
                target_x,
                target_y,
                jump,
                fire,
                hook,
                player_flags,
                wanted_weapon,
                next_weapon,
                prev_weapon,
            },
            corrections,
        ))
    }
}

/// `CNetObj_Projectile` (`NETOBJTYPE_PROJECTILE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Projectile {
    pub x: i32,
    pub y: i32,
    pub vel_x: i32,
    pub vel_y: i32,
    pub type_: i32,
    pub start_tick: i32,
}

impl Projectile {
    /// `NETOBJTYPE_PROJECTILE` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 2;
    /// Static size in `i32`s (`sizeof(CNetObj_Projectile) / 4`).
    pub const SIZE_INTS: usize = 6;
}

impl Projectile {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let vel_x = unpacker.get_uncompressed_int();
        let vel_y = unpacker.get_uncompressed_int();
        let mut type_ = unpacker.get_uncompressed_int();
        if type_ < 0 {
            type_ = 0;
            corrections += 1;
        } else if type_ > 5 {
            type_ = 5;
            corrections += 1;
        }
        let start_tick = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((
            Projectile {
                x,
                y,
                vel_x,
                vel_y,
                type_,
                start_tick,
            },
            corrections,
        ))
    }
}

/// `CNetObj_Laser` (`NETOBJTYPE_LASER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Laser {
    pub x: i32,
    pub y: i32,
    pub from_x: i32,
    pub from_y: i32,
    pub start_tick: i32,
}

impl Laser {
    /// `NETOBJTYPE_LASER` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 3;
    /// Static size in `i32`s (`sizeof(CNetObj_Laser) / 4`).
    pub const SIZE_INTS: usize = 5;
}

impl Laser {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let from_x = unpacker.get_uncompressed_int();
        let from_y = unpacker.get_uncompressed_int();
        let start_tick = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((
            Laser {
                x,
                y,
                from_x,
                from_y,
                start_tick,
            },
            corrections,
        ))
    }
}

/// `CNetObj_Pickup` (`NETOBJTYPE_PICKUP`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pickup {
    pub x: i32,
    pub y: i32,
    pub type_: i32,
    pub subtype: i32,
}

impl Pickup {
    /// `NETOBJTYPE_PICKUP` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 4;
    /// Static size in `i32`s (`sizeof(CNetObj_Pickup) / 4`).
    pub const SIZE_INTS: usize = 4;
}

impl Pickup {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let mut type_ = unpacker.get_uncompressed_int();
        if type_ < 0 {
            type_ = 0;
            corrections += 1;
        }
        let mut subtype = unpacker.get_uncompressed_int();
        if subtype < 0 {
            subtype = 0;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((Pickup { x, y, type_, subtype }, corrections))
    }
}

/// `CNetObj_Flag` (`NETOBJTYPE_FLAG`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Flag {
    pub x: i32,
    pub y: i32,
    pub team: i32,
}

impl Flag {
    /// `NETOBJTYPE_FLAG` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 5;
    /// Static size in `i32`s (`sizeof(CNetObj_Flag) / 4`).
    pub const SIZE_INTS: usize = 3;
}

impl Flag {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let mut team = unpacker.get_uncompressed_int();
        if team < 0 {
            team = 0;
            corrections += 1;
        } else if team > 1 {
            team = 1;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((Flag { x, y, team }, corrections))
    }
}

/// `CNetObj_GameInfo` (`NETOBJTYPE_GAMEINFO`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameInfo {
    pub game_flags: i32,
    pub game_state_flags: i32,
    pub round_start_tick: i32,
    pub warmup_timer: i32,
    pub score_limit: i32,
    pub time_limit: i32,
    pub round_num: i32,
    pub round_current: i32,
}

impl GameInfo {
    /// `NETOBJTYPE_GAMEINFO` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 6;
    /// Static size in `i32`s (`sizeof(CNetObj_GameInfo) / 4`).
    pub const SIZE_INTS: usize = 8;
}

impl GameInfo {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let mut game_flags = unpacker.get_uncompressed_int();
        if game_flags < 0 {
            game_flags = 0;
            corrections += 1;
        } else if game_flags > 256 {
            game_flags = 256;
            corrections += 1;
        }
        let mut game_state_flags = unpacker.get_uncompressed_int();
        if game_state_flags < 0 {
            game_state_flags = 0;
            corrections += 1;
        } else if game_state_flags > 256 {
            game_state_flags = 256;
            corrections += 1;
        }
        let round_start_tick = unpacker.get_uncompressed_int();
        let warmup_timer = unpacker.get_uncompressed_int();
        let mut score_limit = unpacker.get_uncompressed_int();
        if score_limit < 0 {
            score_limit = 0;
            corrections += 1;
        }
        let mut time_limit = unpacker.get_uncompressed_int();
        if time_limit < 0 {
            time_limit = 0;
            corrections += 1;
        }
        let mut round_num = unpacker.get_uncompressed_int();
        if round_num < 0 {
            round_num = 0;
            corrections += 1;
        }
        let mut round_current = unpacker.get_uncompressed_int();
        if round_current < 0 {
            round_current = 0;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((
            GameInfo {
                game_flags,
                game_state_flags,
                round_start_tick,
                warmup_timer,
                score_limit,
                time_limit,
                round_num,
                round_current,
            },
            corrections,
        ))
    }
}

/// `CNetObj_GameData` (`NETOBJTYPE_GAMEDATA`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameData {
    pub teamscore_red: i32,
    pub teamscore_blue: i32,
    pub flag_carrier_red: i32,
    pub flag_carrier_blue: i32,
}

impl GameData {
    /// `NETOBJTYPE_GAMEDATA` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 7;
    /// Static size in `i32`s (`sizeof(CNetObj_GameData) / 4`).
    pub const SIZE_INTS: usize = 4;
}

impl GameData {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let teamscore_red = unpacker.get_uncompressed_int();
        let teamscore_blue = unpacker.get_uncompressed_int();
        let mut flag_carrier_red = unpacker.get_uncompressed_int();
        if flag_carrier_red < -3 {
            flag_carrier_red = -3;
            corrections += 1;
        } else if flag_carrier_red > 127 {
            flag_carrier_red = 127;
            corrections += 1;
        }
        let mut flag_carrier_blue = unpacker.get_uncompressed_int();
        if flag_carrier_blue < -3 {
            flag_carrier_blue = -3;
            corrections += 1;
        } else if flag_carrier_blue > 127 {
            flag_carrier_blue = 127;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((
            GameData {
                teamscore_red,
                teamscore_blue,
                flag_carrier_red,
                flag_carrier_blue,
            },
            corrections,
        ))
    }
}

/// `CNetObj_CharacterCore` (`NETOBJTYPE_CHARACTERCORE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharacterCore {
    pub tick: i32,
    pub x: i32,
    pub y: i32,
    pub vel_x: i32,
    pub vel_y: i32,
    pub angle: i32,
    pub direction: i32,
    pub jumped: i32,
    pub hooked_player: i32,
    pub hook_state: i32,
    pub hook_tick: i32,
    pub hook_x: i32,
    pub hook_y: i32,
    pub hook_dx: i32,
    pub hook_dy: i32,
}

impl CharacterCore {
    /// `NETOBJTYPE_CHARACTERCORE` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 8;
    /// Static size in `i32`s (`sizeof(CNetObj_CharacterCore) / 4`).
    pub const SIZE_INTS: usize = 15;
}

impl CharacterCore {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let tick = unpacker.get_uncompressed_int();
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let vel_x = unpacker.get_uncompressed_int();
        let vel_y = unpacker.get_uncompressed_int();
        let angle = unpacker.get_uncompressed_int();
        let mut direction = unpacker.get_uncompressed_int();
        if direction < -1 {
            direction = -1;
            corrections += 1;
        } else if direction > 1 {
            direction = 1;
            corrections += 1;
        }
        let mut jumped = unpacker.get_uncompressed_int();
        if jumped < 0 {
            jumped = 0;
            corrections += 1;
        } else if jumped > 3 {
            jumped = 3;
            corrections += 1;
        }
        let mut hooked_player = unpacker.get_uncompressed_int();
        if hooked_player < -1 {
            hooked_player = -1;
            corrections += 1;
        } else if hooked_player > 127 {
            hooked_player = 127;
            corrections += 1;
        }
        let mut hook_state = unpacker.get_uncompressed_int();
        if hook_state < -1 {
            hook_state = -1;
            corrections += 1;
        } else if hook_state > 5 {
            hook_state = 5;
            corrections += 1;
        }
        let hook_tick = unpacker.get_uncompressed_int();
        let hook_x = unpacker.get_uncompressed_int();
        let hook_y = unpacker.get_uncompressed_int();
        let hook_dx = unpacker.get_uncompressed_int();
        let hook_dy = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((
            CharacterCore {
                tick,
                x,
                y,
                vel_x,
                vel_y,
                angle,
                direction,
                jumped,
                hooked_player,
                hook_state,
                hook_tick,
                hook_x,
                hook_y,
                hook_dx,
                hook_dy,
            },
            corrections,
        ))
    }
}

/// `CNetObj_Character` (`NETOBJTYPE_CHARACTER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Character {
    pub tick: i32,
    pub x: i32,
    pub y: i32,
    pub vel_x: i32,
    pub vel_y: i32,
    pub angle: i32,
    pub direction: i32,
    pub jumped: i32,
    pub hooked_player: i32,
    pub hook_state: i32,
    pub hook_tick: i32,
    pub hook_x: i32,
    pub hook_y: i32,
    pub hook_dx: i32,
    pub hook_dy: i32,
    pub player_flags: i32,
    pub health: i32,
    pub armor: i32,
    pub ammo_count: i32,
    pub weapon: i32,
    pub emote: i32,
    pub attack_tick: i32,
}

impl Character {
    /// `NETOBJTYPE_CHARACTER` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 9;
    /// Static size in `i32`s (`sizeof(CNetObj_Character) / 4`).
    pub const SIZE_INTS: usize = 22;
}

impl Character {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let tick = unpacker.get_uncompressed_int();
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let vel_x = unpacker.get_uncompressed_int();
        let vel_y = unpacker.get_uncompressed_int();
        let angle = unpacker.get_uncompressed_int();
        let mut direction = unpacker.get_uncompressed_int();
        if direction < -1 {
            direction = -1;
            corrections += 1;
        } else if direction > 1 {
            direction = 1;
            corrections += 1;
        }
        let mut jumped = unpacker.get_uncompressed_int();
        if jumped < 0 {
            jumped = 0;
            corrections += 1;
        } else if jumped > 3 {
            jumped = 3;
            corrections += 1;
        }
        let mut hooked_player = unpacker.get_uncompressed_int();
        if hooked_player < -1 {
            hooked_player = -1;
            corrections += 1;
        } else if hooked_player > 127 {
            hooked_player = 127;
            corrections += 1;
        }
        let mut hook_state = unpacker.get_uncompressed_int();
        if hook_state < -1 {
            hook_state = -1;
            corrections += 1;
        } else if hook_state > 5 {
            hook_state = 5;
            corrections += 1;
        }
        let hook_tick = unpacker.get_uncompressed_int();
        let hook_x = unpacker.get_uncompressed_int();
        let hook_y = unpacker.get_uncompressed_int();
        let hook_dx = unpacker.get_uncompressed_int();
        let hook_dy = unpacker.get_uncompressed_int();
        let mut player_flags = unpacker.get_uncompressed_int();
        if player_flags < 0 {
            player_flags = 0;
            corrections += 1;
        } else if player_flags > 256 {
            player_flags = 256;
            corrections += 1;
        }
        let mut health = unpacker.get_uncompressed_int();
        if health < 0 {
            health = 0;
            corrections += 1;
        } else if health > 10 {
            health = 10;
            corrections += 1;
        }
        let mut armor = unpacker.get_uncompressed_int();
        if armor < 0 {
            armor = 0;
            corrections += 1;
        } else if armor > 10 {
            armor = 10;
            corrections += 1;
        }
        let mut ammo_count = unpacker.get_uncompressed_int();
        if ammo_count < -1 {
            ammo_count = -1;
            corrections += 1;
        } else if ammo_count > 10 {
            ammo_count = 10;
            corrections += 1;
        }
        let mut weapon = unpacker.get_uncompressed_int();
        if weapon < -1 {
            weapon = -1;
            corrections += 1;
        } else if weapon > 5 {
            weapon = 5;
            corrections += 1;
        }
        let mut emote = unpacker.get_uncompressed_int();
        if emote < 0 {
            emote = 0;
            corrections += 1;
        } else if emote > 6 {
            emote = 6;
            corrections += 1;
        }
        let mut attack_tick = unpacker.get_uncompressed_int();
        if attack_tick < 0 {
            attack_tick = 0;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((
            Character {
                tick,
                x,
                y,
                vel_x,
                vel_y,
                angle,
                direction,
                jumped,
                hooked_player,
                hook_state,
                hook_tick,
                hook_x,
                hook_y,
                hook_dx,
                hook_dy,
                player_flags,
                health,
                armor,
                ammo_count,
                weapon,
                emote,
                attack_tick,
            },
            corrections,
        ))
    }
}

/// `CNetObj_PlayerInfo` (`NETOBJTYPE_PLAYERINFO`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerInfo {
    pub local: i32,
    pub client_id: i32,
    pub team: i32,
    pub score: i32,
    pub latency: i32,
}

impl PlayerInfo {
    /// `NETOBJTYPE_PLAYERINFO` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 10;
    /// Static size in `i32`s (`sizeof(CNetObj_PlayerInfo) / 4`).
    pub const SIZE_INTS: usize = 5;
}

impl PlayerInfo {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let mut local = unpacker.get_uncompressed_int();
        if local < 0 {
            local = 0;
            corrections += 1;
        } else if local > 1 {
            local = 1;
            corrections += 1;
        }
        let mut client_id = unpacker.get_uncompressed_int();
        if client_id < 0 {
            client_id = 0;
            corrections += 1;
        } else if client_id > 127 {
            client_id = 127;
            corrections += 1;
        }
        let mut team = unpacker.get_uncompressed_int();
        if team < -1 {
            team = -1;
            corrections += 1;
        } else if team > 1 {
            team = 1;
            corrections += 1;
        }
        let score = unpacker.get_uncompressed_int();
        let latency = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((
            PlayerInfo {
                local,
                client_id,
                team,
                score,
                latency,
            },
            corrections,
        ))
    }
}

/// `CNetObj_ClientInfo` (`NETOBJTYPE_CLIENTINFO`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientInfo {
    /// Decoded from 4 wire ints (see [`crate::intstr::ints_to_str`]).
    pub name: String,
    /// Decoded from 3 wire ints (see [`crate::intstr::ints_to_str`]).
    pub clan: String,
    pub country: i32,
    /// Decoded from 6 wire ints (see [`crate::intstr::ints_to_str`]).
    pub skin: String,
    pub use_custom_color: i32,
    pub color_body: i32,
    pub color_feet: i32,
}

impl ClientInfo {
    /// `NETOBJTYPE_CLIENTINFO` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 11;
    /// Static size in `i32`s (`sizeof(CNetObj_ClientInfo) / 4`).
    pub const SIZE_INTS: usize = 17;
}

impl ClientInfo {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let __name_ints = [
            unpacker.get_uncompressed_int(),
            unpacker.get_uncompressed_int(),
            unpacker.get_uncompressed_int(),
            unpacker.get_uncompressed_int(),
        ];
        let name = crate::intstr::ints_to_str(&__name_ints);
        let __clan_ints = [
            unpacker.get_uncompressed_int(),
            unpacker.get_uncompressed_int(),
            unpacker.get_uncompressed_int(),
        ];
        let clan = crate::intstr::ints_to_str(&__clan_ints);
        let country = unpacker.get_uncompressed_int();
        let __skin_ints = [
            unpacker.get_uncompressed_int(),
            unpacker.get_uncompressed_int(),
            unpacker.get_uncompressed_int(),
            unpacker.get_uncompressed_int(),
            unpacker.get_uncompressed_int(),
            unpacker.get_uncompressed_int(),
        ];
        let skin = crate::intstr::ints_to_str(&__skin_ints);
        let mut use_custom_color = unpacker.get_uncompressed_int();
        if use_custom_color < 0 {
            use_custom_color = 0;
            corrections += 1;
        } else if use_custom_color > 1 {
            use_custom_color = 1;
            corrections += 1;
        }
        let color_body = unpacker.get_uncompressed_int();
        let color_feet = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((
            ClientInfo {
                name,
                clan,
                country,
                skin,
                use_custom_color,
                color_body,
                color_feet,
            },
            corrections,
        ))
    }
}

/// `CNetObj_SpectatorInfo` (`NETOBJTYPE_SPECTATORINFO`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpectatorInfo {
    pub spectator_id: i32,
    pub x: i32,
    pub y: i32,
}

impl SpectatorInfo {
    /// `NETOBJTYPE_SPECTATORINFO` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 12;
    /// Static size in `i32`s (`sizeof(CNetObj_SpectatorInfo) / 4`).
    pub const SIZE_INTS: usize = 3;
}

impl SpectatorInfo {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let mut spectator_id = unpacker.get_uncompressed_int();
        if spectator_id < -1 {
            spectator_id = -1;
            corrections += 1;
        } else if spectator_id > 127 {
            spectator_id = 127;
            corrections += 1;
        }
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((SpectatorInfo { spectator_id, x, y }, corrections))
    }
}

/// `CNetEvent_Common` (`NETEVENTTYPE_COMMON`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Common {
    pub x: i32,
    pub y: i32,
}

impl Common {
    /// `NETEVENTTYPE_COMMON` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 13;
    /// Static size in `i32`s (`sizeof(CNetEvent_Common) / 4`).
    pub const SIZE_INTS: usize = 2;
}

impl Common {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((Common { x, y }, corrections))
    }
}

/// `CNetEvent_Explosion` (`NETEVENTTYPE_EXPLOSION`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Explosion {
    pub x: i32,
    pub y: i32,
}

impl Explosion {
    /// `NETEVENTTYPE_EXPLOSION` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 14;
    /// Static size in `i32`s (`sizeof(CNetEvent_Explosion) / 4`).
    pub const SIZE_INTS: usize = 2;
}

impl Explosion {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((Explosion { x, y }, corrections))
    }
}

/// `CNetEvent_Spawn` (`NETEVENTTYPE_SPAWN`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spawn {
    pub x: i32,
    pub y: i32,
}

impl Spawn {
    /// `NETEVENTTYPE_SPAWN` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 15;
    /// Static size in `i32`s (`sizeof(CNetEvent_Spawn) / 4`).
    pub const SIZE_INTS: usize = 2;
}

impl Spawn {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((Spawn { x, y }, corrections))
    }
}

/// `CNetEvent_HammerHit` (`NETEVENTTYPE_HAMMERHIT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HammerHit {
    pub x: i32,
    pub y: i32,
}

impl HammerHit {
    /// `NETEVENTTYPE_HAMMERHIT` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 16;
    /// Static size in `i32`s (`sizeof(CNetEvent_HammerHit) / 4`).
    pub const SIZE_INTS: usize = 2;
}

impl HammerHit {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((HammerHit { x, y }, corrections))
    }
}

/// `CNetEvent_Death` (`NETEVENTTYPE_DEATH`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Death {
    pub x: i32,
    pub y: i32,
    pub client_id: i32,
}

impl Death {
    /// `NETEVENTTYPE_DEATH` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 17;
    /// Static size in `i32`s (`sizeof(CNetEvent_Death) / 4`).
    pub const SIZE_INTS: usize = 3;
}

impl Death {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let mut client_id = unpacker.get_uncompressed_int();
        if client_id < 0 {
            client_id = 0;
            corrections += 1;
        } else if client_id > 127 {
            client_id = 127;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((Death { x, y, client_id }, corrections))
    }
}

/// `CNetEvent_SoundGlobal` (`NETEVENTTYPE_SOUNDGLOBAL`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoundGlobal {
    pub x: i32,
    pub y: i32,
    pub sound_id: i32,
}

impl SoundGlobal {
    /// `NETEVENTTYPE_SOUNDGLOBAL` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 18;
    /// Static size in `i32`s (`sizeof(CNetEvent_SoundGlobal) / 4`).
    pub const SIZE_INTS: usize = 3;
}

impl SoundGlobal {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let mut sound_id = unpacker.get_uncompressed_int();
        if sound_id < 0 {
            sound_id = 0;
            corrections += 1;
        } else if sound_id > 40 {
            sound_id = 40;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((SoundGlobal { x, y, sound_id }, corrections))
    }
}

/// `CNetEvent_SoundWorld` (`NETEVENTTYPE_SOUNDWORLD`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoundWorld {
    pub x: i32,
    pub y: i32,
    pub sound_id: i32,
}

impl SoundWorld {
    /// `NETEVENTTYPE_SOUNDWORLD` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 19;
    /// Static size in `i32`s (`sizeof(CNetEvent_SoundWorld) / 4`).
    pub const SIZE_INTS: usize = 3;
}

impl SoundWorld {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let mut sound_id = unpacker.get_uncompressed_int();
        if sound_id < 0 {
            sound_id = 0;
            corrections += 1;
        } else if sound_id > 40 {
            sound_id = 40;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((SoundWorld { x, y, sound_id }, corrections))
    }
}

/// `CNetEvent_DamageInd` (`NETEVENTTYPE_DAMAGEIND`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DamageInd {
    pub x: i32,
    pub y: i32,
    pub angle: i32,
}

impl DamageInd {
    /// `NETEVENTTYPE_DAMAGEIND` — the fixed, non-UUID snapshot object type id.
    pub const ID: i32 = 20;
    /// Static size in `i32`s (`sizeof(CNetEvent_DamageInd) / 4`).
    pub const SIZE_INTS: usize = 3;
}

impl DamageInd {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let angle = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((DamageInd { x, y, angle }, corrections))
    }
}

/// `CNetObj_MyOwnObject` (`NETOBJTYPE_MYOWNOBJECT`).
/// UUID name: `my-own-object@heinrich5991.de`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MyOwnObject {
    pub test: i32,
}

impl MyOwnObject {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let test = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((MyOwnObject { test }, corrections))
    }
}

/// `CNetObj_DDNetCharacter` (`NETOBJTYPE_DDNETCHARACTER`).
/// UUID name: `character@netobj.ddnet.tw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DDNetCharacter {
    pub flags: i32,
    pub freeze_end: i32,
    pub jumps: i32,
    pub tele_checkpoint: i32,
    pub strong_weak_id: i32,
    pub jumped_total: i32,
    pub ninja_activation_tick: i32,
    pub freeze_start: i32,
    pub target_x: i32,
    pub target_y: i32,
    pub tune_zone_override: i32,
}

impl DDNetCharacter {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let flags = unpacker.get_uncompressed_int_or_default(0);
        let freeze_end = unpacker.get_uncompressed_int_or_default(0);
        let mut jumps = unpacker.get_uncompressed_int_or_default(2);
        if jumps < -1 {
            jumps = -1;
            corrections += 1;
        } else if jumps > 255 {
            jumps = 255;
            corrections += 1;
        }
        let tele_checkpoint = unpacker.get_uncompressed_int_or_default(-1);
        let mut strong_weak_id = unpacker.get_uncompressed_int_or_default(0);
        if strong_weak_id < 0 {
            strong_weak_id = 0;
            corrections += 1;
        } else if strong_weak_id > 127 {
            strong_weak_id = 127;
            corrections += 1;
        }
        let mut jumped_total = unpacker.get_uncompressed_int_or_default(-1);
        if jumped_total < -1 {
            jumped_total = -1;
            corrections += 1;
        } else if jumped_total > 255 {
            jumped_total = 255;
            corrections += 1;
        }
        let ninja_activation_tick = unpacker.get_uncompressed_int_or_default(-1);
        let freeze_start = unpacker.get_uncompressed_int_or_default(-1);
        let target_x = unpacker.get_uncompressed_int_or_default(0);
        let target_y = unpacker.get_uncompressed_int_or_default(0);
        let mut tune_zone_override = unpacker.get_uncompressed_int_or_default(-1);
        if tune_zone_override < -1 {
            tune_zone_override = -1;
            corrections += 1;
        } else if tune_zone_override > 255 {
            tune_zone_override = 255;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((
            DDNetCharacter {
                flags,
                freeze_end,
                jumps,
                tele_checkpoint,
                strong_weak_id,
                jumped_total,
                ninja_activation_tick,
                freeze_start,
                target_x,
                target_y,
                tune_zone_override,
            },
            corrections,
        ))
    }
}

/// `CNetObj_DDNetPlayer` (`NETOBJTYPE_DDNETPLAYER`).
/// UUID name: `player@netobj.ddnet.tw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DDNetPlayer {
    pub flags: i32,
    pub auth_level: i32,
    pub finish_time_seconds: i32,
    pub finish_time_millis: i32,
}

impl DDNetPlayer {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let flags = unpacker.get_uncompressed_int();
        let mut auth_level = unpacker.get_uncompressed_int();
        if auth_level < 0 {
            auth_level = 0;
            corrections += 1;
        } else if auth_level > 3 {
            auth_level = 3;
            corrections += 1;
        }
        let mut finish_time_seconds = unpacker.get_uncompressed_int_or_default(-2);
        if finish_time_seconds < -2 {
            finish_time_seconds = -2;
            corrections += 1;
        }
        let mut finish_time_millis = unpacker.get_uncompressed_int_or_default(0);
        if finish_time_millis < 0 {
            finish_time_millis = 0;
            corrections += 1;
        } else if finish_time_millis > 999 {
            finish_time_millis = 999;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((
            DDNetPlayer {
                flags,
                auth_level,
                finish_time_seconds,
                finish_time_millis,
            },
            corrections,
        ))
    }
}

/// `CNetObj_GameInfoEx` (`NETOBJTYPE_GAMEINFOEX`).
/// UUID name: `gameinfo@netobj.ddnet.tw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameInfoEx {
    pub flags: i32,
    pub version: i32,
    pub flags2: i32,
    pub min_team_size: i32,
    pub max_team_size: i32,
    pub num_dd_race_teams: i32,
}

impl GameInfoEx {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let flags = unpacker.get_uncompressed_int_or_default(0);
        let version = unpacker.get_uncompressed_int_or_default(0);
        let flags2 = unpacker.get_uncompressed_int_or_default(0);
        let mut min_team_size = unpacker.get_uncompressed_int_or_default(0);
        if min_team_size < 0 {
            min_team_size = 0;
            corrections += 1;
        } else if min_team_size > 128 {
            min_team_size = 128;
            corrections += 1;
        }
        let mut max_team_size = unpacker.get_uncompressed_int_or_default(0);
        if max_team_size < 0 {
            max_team_size = 0;
            corrections += 1;
        } else if max_team_size > 128 {
            max_team_size = 128;
            corrections += 1;
        }
        let num_dd_race_teams = unpacker.get_uncompressed_int_or_default(0);
        if unpacker.error() {
            return None;
        }
        Some((
            GameInfoEx {
                flags,
                version,
                flags2,
                min_team_size,
                max_team_size,
                num_dd_race_teams,
            },
            corrections,
        ))
    }
}

/// `CNetObj_DDRaceProjectile` (`NETOBJTYPE_DDRACEPROJECTILE`).
/// UUID name: `projectile@netobj.ddnet.tw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DDRaceProjectile {
    pub x: i32,
    pub y: i32,
    pub angle: i32,
    pub data: i32,
    pub type_: i32,
    pub start_tick: i32,
}

impl DDRaceProjectile {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let angle = unpacker.get_uncompressed_int();
        let data = unpacker.get_uncompressed_int();
        let mut type_ = unpacker.get_uncompressed_int();
        if type_ < 0 {
            type_ = 0;
            corrections += 1;
        } else if type_ > 5 {
            type_ = 5;
            corrections += 1;
        }
        let start_tick = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((
            DDRaceProjectile {
                x,
                y,
                angle,
                data,
                type_,
                start_tick,
            },
            corrections,
        ))
    }
}

/// `CNetObj_DDNetLaser` (`NETOBJTYPE_DDNETLASER`).
/// UUID name: `laser@netobj.ddnet.tw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DDNetLaser {
    pub to_x: i32,
    pub to_y: i32,
    pub from_x: i32,
    pub from_y: i32,
    pub start_tick: i32,
    pub owner: i32,
    pub type_: i32,
    pub switch_number: i32,
    pub subtype: i32,
    pub flags: i32,
}

impl DDNetLaser {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let to_x = unpacker.get_uncompressed_int();
        let to_y = unpacker.get_uncompressed_int();
        let from_x = unpacker.get_uncompressed_int();
        let from_y = unpacker.get_uncompressed_int();
        let start_tick = unpacker.get_uncompressed_int();
        let mut owner = unpacker.get_uncompressed_int();
        if owner < -1 {
            owner = -1;
            corrections += 1;
        } else if owner > 127 {
            owner = 127;
            corrections += 1;
        }
        let type_ = unpacker.get_uncompressed_int();
        let switch_number = unpacker.get_uncompressed_int_or_default(-1);
        let subtype = unpacker.get_uncompressed_int_or_default(-1);
        let flags = unpacker.get_uncompressed_int_or_default(0);
        if unpacker.error() {
            return None;
        }
        Some((
            DDNetLaser {
                to_x,
                to_y,
                from_x,
                from_y,
                start_tick,
                owner,
                type_,
                switch_number,
                subtype,
                flags,
            },
            corrections,
        ))
    }
}

/// `CNetObj_DDNetProjectile` (`NETOBJTYPE_DDNETPROJECTILE`).
/// UUID name: `ddnet-projectile@netobj.ddnet.tw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DDNetProjectile {
    pub x: i32,
    pub y: i32,
    pub vel_x: i32,
    pub vel_y: i32,
    pub type_: i32,
    pub start_tick: i32,
    pub owner: i32,
    pub switch_number: i32,
    pub tune_zone: i32,
    pub flags: i32,
}

impl DDNetProjectile {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let vel_x = unpacker.get_uncompressed_int();
        let vel_y = unpacker.get_uncompressed_int();
        let mut type_ = unpacker.get_uncompressed_int();
        if type_ < 0 {
            type_ = 0;
            corrections += 1;
        } else if type_ > 5 {
            type_ = 5;
            corrections += 1;
        }
        let start_tick = unpacker.get_uncompressed_int();
        let mut owner = unpacker.get_uncompressed_int();
        if owner < -1 {
            owner = -1;
            corrections += 1;
        } else if owner > 127 {
            owner = 127;
            corrections += 1;
        }
        let switch_number = unpacker.get_uncompressed_int();
        let mut tune_zone = unpacker.get_uncompressed_int();
        if tune_zone < 0 {
            tune_zone = 0;
            corrections += 1;
        } else if tune_zone > 255 {
            tune_zone = 255;
            corrections += 1;
        }
        let flags = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((
            DDNetProjectile {
                x,
                y,
                vel_x,
                vel_y,
                type_,
                start_tick,
                owner,
                switch_number,
                tune_zone,
                flags,
            },
            corrections,
        ))
    }
}

/// `CNetObj_DDNetPickup` (`NETOBJTYPE_DDNETPICKUP`).
/// UUID name: `pickup@netobj.ddnet.tw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DDNetPickup {
    pub x: i32,
    pub y: i32,
    pub type_: i32,
    pub subtype: i32,
    pub switch_number: i32,
    pub flags: i32,
}

impl DDNetPickup {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let mut type_ = unpacker.get_uncompressed_int();
        if type_ < 0 {
            type_ = 0;
            corrections += 1;
        }
        let mut subtype = unpacker.get_uncompressed_int();
        if subtype < 0 {
            subtype = 0;
            corrections += 1;
        }
        let switch_number = unpacker.get_uncompressed_int();
        let flags = unpacker.get_uncompressed_int_or_default(0);
        if unpacker.error() {
            return None;
        }
        Some((
            DDNetPickup {
                x,
                y,
                type_,
                subtype,
                switch_number,
                flags,
            },
            corrections,
        ))
    }
}

/// `CNetObj_DDNetSpectatorInfo` (`NETOBJTYPE_DDNETSPECTATORINFO`).
/// UUID name: `spectator-info@netobj.ddnet.org`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DDNetSpectatorInfo {
    pub has_camera_info: i32,
    pub zoom: i32,
    pub deadzone: i32,
    pub follow_factor: i32,
    pub spectator_count: i32,
}

impl DDNetSpectatorInfo {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let mut has_camera_info = unpacker.get_uncompressed_int();
        if has_camera_info < 0 {
            has_camera_info = 0;
            corrections += 1;
        } else if has_camera_info > 1 {
            has_camera_info = 1;
            corrections += 1;
        }
        let mut zoom = unpacker.get_uncompressed_int();
        if zoom < 0 {
            zoom = 0;
            corrections += 1;
        }
        let mut deadzone = unpacker.get_uncompressed_int();
        if deadzone < 0 {
            deadzone = 0;
            corrections += 1;
        }
        let mut follow_factor = unpacker.get_uncompressed_int();
        if follow_factor < 0 {
            follow_factor = 0;
            corrections += 1;
        }
        let mut spectator_count = unpacker.get_uncompressed_int_or_default(0);
        if spectator_count < 0 {
            spectator_count = 0;
            corrections += 1;
        } else if spectator_count > 127 {
            spectator_count = 127;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((
            DDNetSpectatorInfo {
                has_camera_info,
                zoom,
                deadzone,
                follow_factor,
                spectator_count,
            },
            corrections,
        ))
    }
}

/// `CNetObj_SpectatorCount` (`NETOBJTYPE_SPECTATORCOUNT`).
/// UUID name: `spectator-count@netobj.ddnet.org`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpectatorCount {
    pub num_spectators: i32,
}

impl SpectatorCount {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let mut num_spectators = unpacker.get_uncompressed_int();
        if num_spectators < 0 {
            num_spectators = 0;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((SpectatorCount { num_spectators }, corrections))
    }
}

/// `CNetEvent_Birthday` (`NETEVENTTYPE_BIRTHDAY`).
/// UUID name: `birthday@netevent.ddnet.org`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Birthday {
    pub x: i32,
    pub y: i32,
}

impl Birthday {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((Birthday { x, y }, corrections))
    }
}

/// `CNetEvent_Finish` (`NETEVENTTYPE_FINISH`).
/// UUID name: `finish@netevent.ddnet.org`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Finish {
    pub x: i32,
    pub y: i32,
}

impl Finish {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((Finish { x, y }, corrections))
    }
}

/// `CNetObj_MyOwnEvent` (`NETOBJTYPE_MYOWNEVENT`).
/// UUID name: `my-own-event@heinrich5991.de`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MyOwnEvent {
    pub test: i32,
}

impl MyOwnEvent {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let test = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((MyOwnEvent { test }, corrections))
    }
}

/// `CNetObj_SpecChar` (`NETOBJTYPE_SPECCHAR`).
/// UUID name: `spec-char@netobj.ddnet.tw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecChar {
    pub x: i32,
    pub y: i32,
}

impl SpecChar {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((SpecChar { x, y }, corrections))
    }
}

/// `CNetObj_SwitchState` (`NETOBJTYPE_SWITCHSTATE`).
/// UUID name: `switch-state@netobj.ddnet.tw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwitchState {
    pub highest_switch_number: i32,
    pub status: [i32; 8],
    pub switch_numbers: [i32; 4],
    pub end_ticks: [i32; 4],
}

impl SwitchState {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let highest_switch_number = unpacker.get_uncompressed_int_or_default(0);
        let status = [
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
        ];
        let switch_numbers = [
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
        ];
        let end_ticks = [
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
            unpacker.get_uncompressed_int_or_default(0),
        ];
        if unpacker.error() {
            return None;
        }
        Some((
            SwitchState {
                highest_switch_number,
                status,
                switch_numbers,
                end_ticks,
            },
            corrections,
        ))
    }
}

/// `CNetObj_EntityEx` (`NETOBJTYPE_ENTITYEX`).
/// UUID name: `entity-ex@netobj.ddnet.tw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityEx {
    pub switch_number: i32,
    pub layer: i32,
    pub entity_class: i32,
}

impl EntityEx {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let switch_number = unpacker.get_uncompressed_int();
        let layer = unpacker.get_uncompressed_int();
        let entity_class = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((
            EntityEx {
                switch_number,
                layer,
                entity_class,
            },
            corrections,
        ))
    }
}

/// `CNetObj_MapBestTime` (`NETOBJTYPE_MAPBESTTIME`).
/// UUID name: `map-best-time@netobj.ddnet.org`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapBestTime {
    pub map_best_time_seconds: i32,
    pub map_best_time_millis: i32,
}

impl MapBestTime {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let mut corrections: u32 = 0;
        let mut map_best_time_seconds = unpacker.get_uncompressed_int();
        if map_best_time_seconds < -1 {
            map_best_time_seconds = -1;
            corrections += 1;
        }
        let mut map_best_time_millis = unpacker.get_uncompressed_int();
        if map_best_time_millis < 0 {
            map_best_time_millis = 0;
            corrections += 1;
        } else if map_best_time_millis > 999 {
            map_best_time_millis = 999;
            corrections += 1;
        }
        if unpacker.error() {
            return None;
        }
        Some((
            MapBestTime {
                map_best_time_seconds,
                map_best_time_millis,
            },
            corrections,
        ))
    }
}

/// `CNetEvent_MapSoundWorld` (`NETEVENTTYPE_MAPSOUNDWORLD`).
/// UUID name: `map-sound-world@netevent.ddnet.org`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapSoundWorld {
    pub x: i32,
    pub y: i32,
    pub sound_id: i32,
}

impl MapSoundWorld {
    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s
    /// per-type case): reads each field in order (missing trailing fields, for object
    /// types that declare a default, fall back to that default rather than failing —
    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no
    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches
    /// `ClampInt`); the second element of the `Some` is how many fields were clamped.
    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut unpacker = crate::packer::Unpacker::new(&bytes);
        let corrections: u32 = 0;
        let x = unpacker.get_uncompressed_int();
        let y = unpacker.get_uncompressed_int();
        let sound_id = unpacker.get_uncompressed_int();
        if unpacker.error() {
            return None;
        }
        Some((MapSoundWorld { x, y, sound_id }, corrections))
    }
}

/// UUID names for every `ex` snapshot object/event, in `datasrc/network.py`
/// declaration order (see the module docs).
pub const EX_NAMES: &[&str] = &[
    "my-own-object@heinrich5991.de",
    "character@netobj.ddnet.tw",
    "player@netobj.ddnet.tw",
    "gameinfo@netobj.ddnet.tw",
    "projectile@netobj.ddnet.tw",
    "laser@netobj.ddnet.tw",
    "ddnet-projectile@netobj.ddnet.tw",
    "pickup@netobj.ddnet.tw",
    "spectator-info@netobj.ddnet.org",
    "spectator-count@netobj.ddnet.org",
    "birthday@netevent.ddnet.org",
    "finish@netevent.ddnet.org",
    "my-own-event@heinrich5991.de",
    "spec-char@netobj.ddnet.tw",
    "switch-state@netobj.ddnet.tw",
    "entity-ex@netobj.ddnet.tw",
    "map-best-time@netobj.ddnet.org",
    "map-sound-world@netevent.ddnet.org",
];

/// Maps an ex object/event's UUID name to a decode function over its raw item
/// data. `None` covers both "unrecognised name" and "recognised but this
/// particular item failed to decode" — the caller (the snapshot/view layer)
/// treats both the same way: the raw item is kept regardless (tolerant
/// decoding), only the *typed* view is missing that one item.
pub fn decode_ex_by_name(name: &str, data: &[i32]) -> Option<(ExObject, u32)> {
    Some(match name {
        "my-own-object@heinrich5991.de" => {
            let (v, c) = MyOwnObject::decode(data)?;
            (ExObject::MyOwnObject(v), c)
        }
        "character@netobj.ddnet.tw" => {
            let (v, c) = DDNetCharacter::decode(data)?;
            (ExObject::DDNetCharacter(v), c)
        }
        "player@netobj.ddnet.tw" => {
            let (v, c) = DDNetPlayer::decode(data)?;
            (ExObject::DDNetPlayer(v), c)
        }
        "gameinfo@netobj.ddnet.tw" => {
            let (v, c) = GameInfoEx::decode(data)?;
            (ExObject::GameInfoEx(v), c)
        }
        "projectile@netobj.ddnet.tw" => {
            let (v, c) = DDRaceProjectile::decode(data)?;
            (ExObject::DDRaceProjectile(v), c)
        }
        "laser@netobj.ddnet.tw" => {
            let (v, c) = DDNetLaser::decode(data)?;
            (ExObject::DDNetLaser(v), c)
        }
        "ddnet-projectile@netobj.ddnet.tw" => {
            let (v, c) = DDNetProjectile::decode(data)?;
            (ExObject::DDNetProjectile(v), c)
        }
        "pickup@netobj.ddnet.tw" => {
            let (v, c) = DDNetPickup::decode(data)?;
            (ExObject::DDNetPickup(v), c)
        }
        "spectator-info@netobj.ddnet.org" => {
            let (v, c) = DDNetSpectatorInfo::decode(data)?;
            (ExObject::DDNetSpectatorInfo(v), c)
        }
        "spectator-count@netobj.ddnet.org" => {
            let (v, c) = SpectatorCount::decode(data)?;
            (ExObject::SpectatorCount(v), c)
        }
        "birthday@netevent.ddnet.org" => {
            let (v, c) = Birthday::decode(data)?;
            (ExObject::Birthday(v), c)
        }
        "finish@netevent.ddnet.org" => {
            let (v, c) = Finish::decode(data)?;
            (ExObject::Finish(v), c)
        }
        "my-own-event@heinrich5991.de" => {
            let (v, c) = MyOwnEvent::decode(data)?;
            (ExObject::MyOwnEvent(v), c)
        }
        "spec-char@netobj.ddnet.tw" => {
            let (v, c) = SpecChar::decode(data)?;
            (ExObject::SpecChar(v), c)
        }
        "switch-state@netobj.ddnet.tw" => {
            let (v, c) = SwitchState::decode(data)?;
            (ExObject::SwitchState(v), c)
        }
        "entity-ex@netobj.ddnet.tw" => {
            let (v, c) = EntityEx::decode(data)?;
            (ExObject::EntityEx(v), c)
        }
        "map-best-time@netobj.ddnet.org" => {
            let (v, c) = MapBestTime::decode(data)?;
            (ExObject::MapBestTime(v), c)
        }
        "map-sound-world@netevent.ddnet.org" => {
            let (v, c) = MapSoundWorld::decode(data)?;
            (ExObject::MapSoundWorld(v), c)
        }
        _ => return None,
    })
}

/// Every ex (UUID-typed) snapshot object/event, decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExObject {
    MyOwnObject(MyOwnObject),
    DDNetCharacter(DDNetCharacter),
    DDNetPlayer(DDNetPlayer),
    GameInfoEx(GameInfoEx),
    DDRaceProjectile(DDRaceProjectile),
    DDNetLaser(DDNetLaser),
    DDNetProjectile(DDNetProjectile),
    DDNetPickup(DDNetPickup),
    DDNetSpectatorInfo(DDNetSpectatorInfo),
    SpectatorCount(SpectatorCount),
    Birthday(Birthday),
    Finish(Finish),
    MyOwnEvent(MyOwnEvent),
    SpecChar(SpecChar),
    SwitchState(SwitchState),
    EntityEx(EntityEx),
    MapBestTime(MapBestTime),
    MapSoundWorld(MapSoundWorld),
}
