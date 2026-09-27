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

//! Enums and bitflags from `datasrc/datatypes.py`.

/// `EMOTE_*` (`datasrc/datatypes.py`).
pub mod emote {
    pub const NORMAL: i32 = 0;
    pub const PAIN: i32 = 1;
    pub const HAPPY: i32 = 2;
    pub const SURPRISE: i32 = 3;
    pub const ANGRY: i32 = 4;
    pub const BLINK: i32 = 5;
    pub const NUM: usize = 6;
}

/// `POWERUP_*` (`datasrc/datatypes.py`).
pub mod powerup {
    pub const HEALTH: i32 = 0;
    pub const ARMOR: i32 = 1;
    pub const WEAPON: i32 = 2;
    pub const NINJA: i32 = 3;
    pub const ARMOR_SHOTGUN: i32 = 4;
    pub const ARMOR_GRENADE: i32 = 5;
    pub const ARMOR_NINJA: i32 = 6;
    pub const ARMOR_LASER: i32 = 7;
    pub const FREEZE: i32 = 8;
    pub const NUM: usize = 9;
}

/// `EMOTICON_*` (`datasrc/datatypes.py`).
pub mod emoticon {
    pub const OOP: i32 = 0;
    pub const EXCLAMATION: i32 = 1;
    pub const HEARTS: i32 = 2;
    pub const DROP: i32 = 3;
    pub const DOTDOT: i32 = 4;
    pub const MUSIC: i32 = 5;
    pub const SORRY: i32 = 6;
    pub const GHOST: i32 = 7;
    pub const SUSHI: i32 = 8;
    pub const SPLATTEE: i32 = 9;
    pub const DEVILTEE: i32 = 10;
    pub const ZOMG: i32 = 11;
    pub const ZZZ: i32 = 12;
    pub const WTF: i32 = 13;
    pub const EYES: i32 = 14;
    pub const QUESTION: i32 = 15;
    pub const NUM: usize = 16;
}

/// `AUTHED_*` (`datasrc/datatypes.py`).
pub mod authed {
    pub const NO: i32 = 0;
    pub const HELPER: i32 = 1;
    pub const MOD: i32 = 2;
    pub const ADMIN: i32 = 3;
    pub const NUM: usize = 4;
}

/// `ENTITYCLASS_*` (`datasrc/datatypes.py`).
pub mod entityclass {
    pub const PROJECTILE: i32 = 0;
    pub const DOOR: i32 = 1;
    pub const DRAGGER_WEAK: i32 = 2;
    pub const DRAGGER_NORMAL: i32 = 3;
    pub const DRAGGER_STRONG: i32 = 4;
    pub const GUN_NORMAL: i32 = 5;
    pub const GUN_EXPLOSIVE: i32 = 6;
    pub const GUN_FREEZE: i32 = 7;
    pub const GUN_UNFREEZE: i32 = 8;
    pub const LIGHT: i32 = 9;
    pub const PICKUP: i32 = 10;
    pub const NUM: usize = 11;
}

/// `LASERTYPE_*` (`datasrc/datatypes.py`).
pub mod lasertype {
    pub const RIFLE: i32 = 0;
    pub const SHOTGUN: i32 = 1;
    pub const DOOR: i32 = 2;
    pub const FREEZE: i32 = 3;
    pub const DRAGGER: i32 = 4;
    pub const GUN: i32 = 5;
    pub const PLASMA: i32 = 6;
    pub const NUM: usize = 7;
}

/// `LASERDRAGGERTYPE_*` (`datasrc/datatypes.py`).
pub mod laserdraggertype {
    pub const WEAK: i32 = 0;
    pub const WEAK_NW: i32 = 1;
    pub const NORMAL: i32 = 2;
    pub const NORMAL_NW: i32 = 3;
    pub const STRONG: i32 = 4;
    pub const STRONG_NW: i32 = 5;
    pub const NUM: usize = 6;
}

/// `LASERGUNTYPE_*` (`datasrc/datatypes.py`).
pub mod laserguntype {
    pub const UNFREEZE: i32 = 0;
    pub const EXPLOSIVE: i32 = 1;
    pub const FREEZE: i32 = 2;
    pub const EXPFREEZE: i32 = 3;
    pub const NUM: usize = 4;
}

/// `TEAM_*` (`datasrc/datatypes.py`).
pub mod team {
    pub const ALL: i32 = -2;
    pub const SPECTATORS: i32 = -1;
    pub const RED: i32 = 0;
    pub const BLUE: i32 = 1;
    pub const WHISPER_SEND: i32 = 2;
    pub const WHISPER_RECV: i32 = 3;
    pub const NUM: usize = 6;
}

/// `SAVESTATE_*` (`datasrc/datatypes.py`).
pub mod savestate {
    pub const PENDING: i32 = 0;
    pub const DONE: i32 = 1;
    pub const FALLBACKFILE: i32 = 2;
    pub const WARNING: i32 = 3;
    pub const ERROR: i32 = 4;
    pub const NUM: usize = 5;
}

/// `PLAYERFLAGFLAG_*` bit flags (`datasrc/datatypes.py`).
pub mod playerflagflag {
    pub const PLAYING: i32 = 1 << 0;
    pub const IN_MENU: i32 = 1 << 1;
    pub const CHATTING: i32 = 1 << 2;
    pub const SCOREBOARD: i32 = 1 << 3;
    pub const AIM: i32 = 1 << 4;
    pub const SPEC_CAM: i32 = 1 << 5;
    pub const INPUT_ABSOLUTE: i32 = 1 << 6;
    pub const INPUT_MANUAL: i32 = 1 << 7;
}

/// `GAMEFLAGFLAG_*` bit flags (`datasrc/datatypes.py`).
pub mod gameflagflag {
    pub const TEAMS: i32 = 1 << 0;
    pub const FLAGS: i32 = 1 << 1;
}

/// `GAMESTATEFLAGFLAG_*` bit flags (`datasrc/datatypes.py`).
pub mod gamestateflagflag {
    pub const GAMEOVER: i32 = 1 << 0;
    pub const SUDDENDEATH: i32 = 1 << 1;
    pub const PAUSED: i32 = 1 << 2;
    pub const RACETIME: i32 = 1 << 3;
}

/// `CHARACTERFLAGFLAG_*` bit flags (`datasrc/datatypes.py`).
pub mod characterflagflag {
    pub const SOLO: i32 = 1 << 0;
    pub const JETPACK: i32 = 1 << 1;
    pub const COLLISION_DISABLED: i32 = 1 << 2;
    pub const ENDLESS_HOOK: i32 = 1 << 3;
    pub const ENDLESS_JUMP: i32 = 1 << 4;
    pub const SUPER: i32 = 1 << 5;
    pub const HAMMER_HIT_DISABLED: i32 = 1 << 6;
    pub const SHOTGUN_HIT_DISABLED: i32 = 1 << 7;
    pub const GRENADE_HIT_DISABLED: i32 = 1 << 8;
    pub const LASER_HIT_DISABLED: i32 = 1 << 9;
    pub const HOOK_HIT_DISABLED: i32 = 1 << 10;
    pub const TELEGUN_GUN: i32 = 1 << 11;
    pub const TELEGUN_GRENADE: i32 = 1 << 12;
    pub const TELEGUN_LASER: i32 = 1 << 13;
    pub const WEAPON_HAMMER: i32 = 1 << 14;
    pub const WEAPON_GUN: i32 = 1 << 15;
    pub const WEAPON_SHOTGUN: i32 = 1 << 16;
    pub const WEAPON_GRENADE: i32 = 1 << 17;
    pub const WEAPON_LASER: i32 = 1 << 18;
    pub const WEAPON_NINJA: i32 = 1 << 19;
    pub const MOVEMENTS_DISABLED: i32 = 1 << 20;
    pub const IN_FREEZE: i32 = 1 << 21;
    pub const PRACTICE_MODE: i32 = 1 << 22;
    pub const LOCK_MODE: i32 = 1 << 23;
    pub const TEAM0_MODE: i32 = 1 << 24;
    pub const INVINCIBLE: i32 = 1 << 25;
}

/// `GAMEINFOFLAGFLAG_*` bit flags (`datasrc/datatypes.py`).
pub mod gameinfoflagflag {
    pub const TIMESCORE: i32 = 1 << 0;
    pub const GAMETYPE_RACE: i32 = 1 << 1;
    pub const GAMETYPE_FASTCAP: i32 = 1 << 2;
    pub const GAMETYPE_FNG: i32 = 1 << 3;
    pub const GAMETYPE_DDRACE: i32 = 1 << 4;
    pub const GAMETYPE_DDNET: i32 = 1 << 5;
    pub const GAMETYPE_BLOCK_WORLDS: i32 = 1 << 6;
    pub const GAMETYPE_VANILLA: i32 = 1 << 7;
    pub const GAMETYPE_PLUS: i32 = 1 << 8;
    pub const FLAG_STARTS_RACE: i32 = 1 << 9;
    pub const RACE: i32 = 1 << 10;
    pub const UNLIMITED_AMMO: i32 = 1 << 11;
    pub const DDRACE_RECORD_MESSAGE: i32 = 1 << 12;
    pub const RACE_RECORD_MESSAGE: i32 = 1 << 13;
    pub const ALLOW_EYE_WHEEL: i32 = 1 << 14;
    pub const ALLOW_HOOK_COLL: i32 = 1 << 15;
    pub const ALLOW_ZOOM: i32 = 1 << 16;
    pub const BUG_DDRACE_GHOST: i32 = 1 << 17;
    pub const BUG_DDRACE_INPUT: i32 = 1 << 18;
    pub const BUG_FNG_LASER_RANGE: i32 = 1 << 19;
    pub const BUG_VANILLA_BOUNCE: i32 = 1 << 20;
    pub const PREDICT_FNG: i32 = 1 << 21;
    pub const PREDICT_DDRACE: i32 = 1 << 22;
    pub const PREDICT_DDRACE_TILES: i32 = 1 << 23;
    pub const PREDICT_VANILLA: i32 = 1 << 24;
    pub const ENTITIES_DDNET: i32 = 1 << 25;
    pub const ENTITIES_DDRACE: i32 = 1 << 26;
    pub const ENTITIES_RACE: i32 = 1 << 27;
    pub const ENTITIES_FNG: i32 = 1 << 28;
    pub const ENTITIES_VANILLA: i32 = 1 << 29;
    pub const DONT_MASK_ENTITIES: i32 = 1 << 30;
    pub const ENTITIES_BW: i32 = 1 << 31;
}

/// `GAMEINFOFLAG2FLAG_*` bit flags (`datasrc/datatypes.py`).
pub mod gameinfoflag2flag {
    pub const ALLOW_X_SKINS: i32 = 1 << 0;
    pub const GAMETYPE_CITY: i32 = 1 << 1;
    pub const GAMETYPE_FDDRACE: i32 = 1 << 2;
    pub const ENTITIES_FDDRACE: i32 = 1 << 3;
    pub const HUD_HEALTH_ARMOR: i32 = 1 << 4;
    pub const HUD_AMMO: i32 = 1 << 5;
    pub const HUD_DDRACE: i32 = 1 << 6;
    pub const NO_WEAK_HOOK: i32 = 1 << 7;
    pub const NO_SKIN_CHANGE_FOR_FROZEN: i32 = 1 << 8;
    pub const DDRACE_TEAM: i32 = 1 << 9;
    pub const PREDICT_EVENTS: i32 = 1 << 10;
    pub const OLD_LASER: i32 = 1 << 11;
}

/// `EXPLAYERFLAGFLAG_*` bit flags (`datasrc/datatypes.py`).
pub mod explayerflagflag {
    pub const AFK: i32 = 1 << 0;
    pub const PAUSED: i32 = 1 << 1;
    pub const SPEC: i32 = 1 << 2;
}

/// `LEGACYPROJECTILEFLAGFLAG_*` bit flags (`datasrc/datatypes.py`).
pub mod legacyprojectileflagflag {
    pub const CLIENTID_BIT0: i32 = 1 << 0;
    pub const CLIENTID_BIT1: i32 = 1 << 1;
    pub const CLIENTID_BIT2: i32 = 1 << 2;
    pub const CLIENTID_BIT3: i32 = 1 << 3;
    pub const CLIENTID_BIT4: i32 = 1 << 4;
    pub const CLIENTID_BIT5: i32 = 1 << 5;
    pub const CLIENTID_BIT6: i32 = 1 << 6;
    pub const CLIENTID_BIT7: i32 = 1 << 7;
    pub const NO_OWNER: i32 = 1 << 8;
    pub const IS_DDNET: i32 = 1 << 9;
    pub const BOUNCE_HORIZONTAL: i32 = 1 << 10;
    pub const BOUNCE_VERTICAL: i32 = 1 << 11;
    pub const EXPLOSIVE: i32 = 1 << 12;
    pub const FREEZE: i32 = 1 << 13;
}

/// `PROJECTILEFLAGFLAG_*` bit flags (`datasrc/datatypes.py`).
pub mod projectileflagflag {
    pub const BOUNCE_HORIZONTAL: i32 = 1 << 0;
    pub const BOUNCE_VERTICAL: i32 = 1 << 1;
    pub const EXPLOSIVE: i32 = 1 << 2;
    pub const FREEZE: i32 = 1 << 3;
    pub const NORMALIZE_VEL: i32 = 1 << 4;
}

/// `LASERFLAGFLAG_*` bit flags (`datasrc/datatypes.py`).
pub mod laserflagflag {
    pub const NO_PREDICT: i32 = 1 << 0;
}

/// `PICKUPFLAGFLAG_*` bit flags (`datasrc/datatypes.py`).
pub mod pickupflagflag {
    pub const XFLIP: i32 = 1 << 0;
    pub const YFLIP: i32 = 1 << 1;
    pub const ROTATE: i32 = 1 << 2;
    pub const NO_PREDICT: i32 = 1 << 3;
}
