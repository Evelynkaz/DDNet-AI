// Ported from DDNet `src/game/{tuning.h,gamecore.h}` (default values + fixed-point encoding) and
// `src/game/client/gameclient.cpp:1058-1193` (`CGameClient::OnMessage`'s `Sv_TuneParams`/
// `Sv_TeamsState`/`Sv_TeamsStateLegacy` special-cased hand-parsing, pinned rev
// c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"), which carries the original Teeworlds
// zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same field order/defaults/fixed-
// point encoding and the same tolerant partial-decode semantics.
//
//! `Sv_TuneParams`/`Sv_TeamsState`/`Sv_TeamsStateLegacy` — task 2.2b review round 1, finding F2.
//!
//! `datasrc/network.py` declares both of these messages with **zero** fields
//! (`NetMessage("Sv_TuneParams", [])`, `network.py:436`) because DDNet's own client does not
//! decode them through the generated `CNetObjHandler::SecureUnpackMsg` machinery at all — it
//! special-cases both ids at the very top of `CGameClient::OnMessage`, before the generated
//! dispatch ever runs, and hand-parses the payload directly off the raw `CUnpacker`
//! (`gameclient.cpp:1058-1193`). That means `crate::generated::messages::SvTuneParams`/
//! `SvTeamsState`/`SvTeamsStateLegacy` are always empty structs — mechanically correct (there is
//! nothing else `datasrc/network.py` could have told the generator), but useless on their own.
//! This module is DDNet's hand-parsing, ported; `crate::message::decode` intercepts these three
//! ids *before* handing them to the generated dispatch, so a caller always gets one of this
//! module's real structs, never the generated empty one (see that module's `decode` for exactly
//! where).

use crate::packer::Unpacker;

/// `CTuningParams::Num()` / the number of `MACRO_TUNING_PARAM` entries in `tuning.h`.
pub const NUM_TUNE_PARAMS: usize = 47;

/// One tuning parameter set, in exactly `src/game/tuning.h`'s declaration order (that order, not
/// the field *names*, is part of the wire format: `Sv_TuneParams` is a bare sequence of varints).
/// Every value is the raw wire int — DDNet's own `×100` fixed-point encoding
/// (`CTuneParam::operator=(float)`: `m_Value = (int)(v * 100.0f)`, truncating toward zero) is left
/// as-is; converting to a real `f32` is physics' job (task 1.3), not this crate's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TuneParams {
    /// How many of the 47 fields below were actually read off the wire before the first unpack
    /// error (`0..=NUM_TUNE_PARAMS`) — every field at or after this count holds DDNet's own
    /// compile-time default (`tuning.h`'s `Value` column, [`DEFAULT_TUNE_PARAMS`]), exactly like
    /// `CTuningParams NewTuning;`'s default construction followed by a partial overwrite loop
    /// (`gameclient.cpp:1063-1086`) — never a bare `0`.
    pub received: usize,
    pub ground_control_speed: i32,
    pub ground_control_accel: i32,
    pub ground_friction: i32,
    pub ground_jump_impulse: i32,
    pub air_jump_impulse: i32,
    pub air_control_speed: i32,
    pub air_control_accel: i32,
    pub air_friction: i32,
    pub hook_length: i32,
    pub hook_fire_speed: i32,
    pub hook_drag_accel: i32,
    pub hook_drag_speed: i32,
    pub gravity: i32,
    pub velramp_start: i32,
    pub velramp_range: i32,
    pub velramp_curvature: i32,
    pub gun_curvature: i32,
    pub gun_speed: i32,
    pub gun_lifetime: i32,
    pub shotgun_curvature: i32,
    pub shotgun_speed: i32,
    pub shotgun_speeddiff: i32,
    pub shotgun_lifetime: i32,
    pub grenade_curvature: i32,
    pub grenade_speed: i32,
    pub grenade_lifetime: i32,
    pub laser_reach: i32,
    pub laser_bounce_delay: i32,
    pub laser_bounce_num: i32,
    pub laser_bounce_cost: i32,
    pub laser_damage: i32,
    pub player_collision: i32,
    pub player_hooking: i32,
    pub jetpack_strength: i32,
    pub shotgun_strength: i32,
    pub explosion_strength: i32,
    pub hammer_strength: i32,
    pub hook_duration: i32,
    pub hammer_fire_delay: i32,
    pub gun_fire_delay: i32,
    pub shotgun_fire_delay: i32,
    pub grenade_fire_delay: i32,
    pub laser_fire_delay: i32,
    pub ninja_fire_delay: i32,
    pub hammer_hit_fire_delay: i32,
    pub ground_elasticity_x: i32,
    pub ground_elasticity_y: i32,
}

/// DDNet's compile-time tuning defaults (`tuning.h`'s `Value` column), each already converted to
/// the ×100 fixed-point wire representation exactly like `CTuneParam::operator=(float)` does
/// (`v * 100.0f` truncated toward zero) — independently cross-checked against well-known DDNet
/// server `tunes` output (e.g. `gravity 0.50`, `gun_curvature 1.25`, `velramp_start 550.00`).
/// Index of `jetpack_strength` within [`DEFAULTS`]/the wire order — see
/// [`decode_sv_tune_params`]'s doc comment (review round 2, finding F8) for why this one index is
/// special-cased. Kept as a named constant, checked against `from_array`'s `v[33]` by a test, so
/// the two never silently drift apart if the field list is ever reordered.
const JETPACK_STRENGTH_INDEX: usize = 33;

const DEFAULTS: [i32; NUM_TUNE_PARAMS] = [
    1000, 200, 50, 1320, 1200, 500, 150, 95, 38000, 8000, 300, 1500, 50, 55000, 200000, 140, 125, 220000, 200, 125,
    275000, 80, 20, 700, 100000, 200, 80000, 15000, 100000, 0, 500, 100, 100, 40000, 1000, 600, 100, 125, 12500, 12500,
    50000, 50000, 80000, 80000, 32000, 0, 0,
];

impl TuneParams {
    /// Returns a copy with [`TuneParams::received`] replaced — a small test/construction
    /// convenience, never used by [`decode_sv_tune_params`] itself.
    pub const fn with_received(self, received: usize) -> Self {
        Self { received, ..self }
    }
}

/// `DEFAULTS` as a full [`TuneParams`] (`received: 0`) — what a fresh `CTuningParams` looks like
/// before any `Sv_TuneParams` message has been applied to it.
pub const DEFAULT_TUNE_PARAMS: TuneParams = from_array(0, DEFAULTS);

const fn from_array(received: usize, v: [i32; NUM_TUNE_PARAMS]) -> TuneParams {
    TuneParams {
        received,
        ground_control_speed: v[0],
        ground_control_accel: v[1],
        ground_friction: v[2],
        ground_jump_impulse: v[3],
        air_jump_impulse: v[4],
        air_control_speed: v[5],
        air_control_accel: v[6],
        air_friction: v[7],
        hook_length: v[8],
        hook_fire_speed: v[9],
        hook_drag_accel: v[10],
        hook_drag_speed: v[11],
        gravity: v[12],
        velramp_start: v[13],
        velramp_range: v[14],
        velramp_curvature: v[15],
        gun_curvature: v[16],
        gun_speed: v[17],
        gun_lifetime: v[18],
        shotgun_curvature: v[19],
        shotgun_speed: v[20],
        shotgun_speeddiff: v[21],
        shotgun_lifetime: v[22],
        grenade_curvature: v[23],
        grenade_speed: v[24],
        grenade_lifetime: v[25],
        laser_reach: v[26],
        laser_bounce_delay: v[27],
        laser_bounce_num: v[28],
        laser_bounce_cost: v[29],
        laser_damage: v[30],
        player_collision: v[31],
        player_hooking: v[32],
        jetpack_strength: v[33],
        shotgun_strength: v[34],
        explosion_strength: v[35],
        hammer_strength: v[36],
        hook_duration: v[37],
        hammer_fire_delay: v[38],
        gun_fire_delay: v[39],
        shotgun_fire_delay: v[40],
        grenade_fire_delay: v[41],
        laser_fire_delay: v[42],
        ninja_fire_delay: v[43],
        hammer_hit_fire_delay: v[44],
        ground_elasticity_x: v[45],
        ground_elasticity_y: v[46],
    }
}

/// Decodes a `Sv_TuneParams` payload (`gameclient.cpp:1058-1090`): reads up to
/// [`NUM_TUNE_PARAMS`] varints in `tuning.h` order, stopping at the first unpacker error (an
/// older/customised server may legitimately send fewer — `Client()->IsSixup()`'s 0.7
/// skip-index-30 special case does not apply to us, 0.6+DDNet only). Every field at or after that
/// point keeps its DDNet default ([`DEFAULT_TUNE_PARAMS`]) — never panics, never a bare `0` —
/// **except** `jetpack_strength` (index 33), which starts from `0`, not `tuning.h`'s `400.00`:
/// review round 2, finding F8. `CGameClient::OnMessage` sets `NewTuning.m_JetpackStrength = 0`
/// itself right before this same read loop (`gameclient.cpp:1068-1070`, comment "No jetpack on
/// DDNet incompatible servers" — jetpack is a DDNet-only extension bolted onto vanilla tuning, so
/// a server that never mentions it, or stops short of index 33, means "no jetpack", not "assume
/// the DDNet default strength"). If the wire value *is* actually read (received > 33), that real
/// value is used as normal, whatever it is.
pub fn decode_sv_tune_params(unpacker: &mut Unpacker) -> TuneParams {
    let mut values = DEFAULTS;
    values[JETPACK_STRENGTH_INDEX] = 0; // gameclient.cpp:1068-1070, see doc comment above.
    let mut received = 0usize;
    for slot in &mut values {
        let v = unpacker.get_int();
        if unpacker.error() {
            break;
        }
        *slot = v;
        received += 1;
    }
    from_array(received, values)
}

/// `TEAM_SUPER` (`teamscore.h:10`): `MAX_CLIENTS`.
const TEAM_SUPER: i32 = 128;
/// `NUM_DDRACE_TEAMS` (`teamscore.h:11`): `TEAM_SUPER + 1` — the exclusive upper bound DDNet
/// validates an incoming team id against.
const NUM_DDRACE_TEAMS: i32 = TEAM_SUPER + 1;
/// `TEAM_FLOCK` (`teamscore.h:9`): the inclusive lower bound.
const TEAM_FLOCK: i32 = 0;

/// `Sv_TeamsState`/`Sv_TeamsStateLegacy` — DDRace team assignment per client id, both ids decoded
/// identically (`gameclient.cpp:1174-1193`'s single shared branch for both `MsgId`s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TeamsState {
    /// `teams[i]` is client `i`'s DDRace team — but **only** for `i < received`. DDNet's own
    /// decode leaves every later slot at whatever the *previous* `Sv_TeamsState`/
    /// `Sv_TeamsStateLegacy` message set it to (`m_Teams`, session state this stateless decoder
    /// does not keep and has no previous value to fall back to) — so, deliberately, this struct
    /// says nothing at all about indices `>= received`; a caller that needs the DDNet-exact
    /// "carry forward the last known team" behaviour has to keep that state itself across calls.
    pub teams: [i32; 128],
    pub received: usize,
}

/// Decodes a `Sv_TeamsState`/`Sv_TeamsStateLegacy` payload. Never panics; reads up to 128 (`MAX_CLIENTS`)
/// team ids, stopping at the first unpack error or out-of-range value (`gameclient.cpp:1179-1188`:
/// `Team >= TEAM_FLOCK && Team < NUM_DDRACE_TEAMS`) — see [`TeamsState::teams`]'s docs for exactly
/// what "stopping" means for the remaining slots.
pub fn decode_teams_state(unpacker: &mut Unpacker) -> TeamsState {
    let mut teams = [0i32; 128];
    let mut received = 0usize;
    for slot in &mut teams {
        let team = unpacker.get_int();
        if unpacker.error() || !(TEAM_FLOCK..NUM_DDRACE_TEAMS).contains(&team) {
            break;
        }
        *slot = team;
        received += 1;
    }
    TeamsState { teams, received }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packer::Packer;

    #[test]
    fn decode_full_tune_params_matches_defaults_when_server_sends_defaults() {
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        for &v in &DEFAULTS {
            packer.add_int(v);
        }
        let mut unpacker = Unpacker::new(packer.data());
        let params = decode_sv_tune_params(&mut unpacker);
        assert_eq!(params.received, NUM_TUNE_PARAMS);
        assert_eq!(params, DEFAULT_TUNE_PARAMS.with_received(NUM_TUNE_PARAMS));
        assert_eq!(params.gravity, 50);
        assert_eq!(params.gun_curvature, 125);
        assert_eq!(params.velramp_start, 55000);
        assert_eq!(params.player_collision, 100);
        assert_eq!(params.ground_elasticity_y, 0);
    }

    #[test]
    fn decode_partial_tune_params_keeps_defaults_for_the_unread_tail() {
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        // Only send the first 5 params (an older/customised server sending fewer).
        for i in 0..5 {
            packer.add_int(9999 + i);
        }
        let mut unpacker = Unpacker::new(packer.data());
        let params = decode_sv_tune_params(&mut unpacker);
        assert_eq!(params.received, 5);
        assert_eq!(params.ground_control_speed, 9999);
        assert_eq!(params.air_jump_impulse, 10003);
        // Everything from index 5 onward keeps the DDNet default, not zero.
        assert_eq!(params.air_control_speed, DEFAULT_TUNE_PARAMS.air_control_speed);
        assert_eq!(params.ground_elasticity_y, DEFAULT_TUNE_PARAMS.ground_elasticity_y);
    }

    #[test]
    fn decode_empty_tune_params_is_all_defaults() {
        let mut unpacker = Unpacker::new(&[]);
        let params = decode_sv_tune_params(&mut unpacker);
        assert_eq!(params.received, 0);
        // F8: `jetpack_strength` is the one field that is NOT `DEFAULT_TUNE_PARAMS`' value here —
        // DDNet's client zeroes it before ever reading the message, so an empty/short payload
        // means "no jetpack" (0), not `tuning.h`'s compile-time default (400.00 -> 40000).
        assert_eq!(params.jetpack_strength, 0);
        assert_ne!(
            DEFAULT_TUNE_PARAMS.jetpack_strength, 0,
            "sanity: the real tuning.h default is non-zero"
        );
        assert_eq!(
            params,
            TuneParams {
                jetpack_strength: 0,
                ..DEFAULT_TUNE_PARAMS
            }
        );
    }

    #[test]
    fn decode_short_tune_params_before_jetpack_index_defaults_jetpack_to_zero_not_tuning_h() {
        // A vanilla (non-DDNet) or older server sends fewer than 34 fields — jetpack_strength is
        // never reached by the read loop, so it must fall back to 0 (F8), not `DEFAULTS[33]`
        // (40000). Everything strictly before index 33 still gets its ordinary tuning.h fallback.
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        for i in 0..10 {
            packer.add_int(1 + i);
        }
        let mut unpacker = Unpacker::new(packer.data());
        let params = decode_sv_tune_params(&mut unpacker);
        assert_eq!(params.received, 10);
        assert_eq!(params.jetpack_strength, 0);
        assert_eq!(params.player_collision, DEFAULT_TUNE_PARAMS.player_collision); // index 31, also unread
    }

    #[test]
    fn decode_tune_params_stopping_exactly_at_jetpack_index_still_defaults_it_to_zero() {
        // Sends exactly the 33 fields before jetpack_strength (indices 0..33) and then an
        // unpack error — `received == 33`, index 33 itself was never read.
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        for i in 0..33 {
            packer.add_int(i);
        }
        let mut unpacker = Unpacker::new(packer.data());
        let params = decode_sv_tune_params(&mut unpacker);
        assert_eq!(params.received, 33);
        assert_eq!(params.jetpack_strength, 0);
    }

    #[test]
    fn decode_tune_params_with_jetpack_actually_on_the_wire_uses_the_real_value() {
        // Once the message genuinely reaches index 33, whatever value the server sent wins —
        // the F8 fallback only applies when that slot was never read at all.
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        for (i, &default) in DEFAULTS.iter().enumerate() {
            packer.add_int(if i == JETPACK_STRENGTH_INDEX { 12345 } else { default });
        }
        let mut unpacker = Unpacker::new(packer.data());
        let params = decode_sv_tune_params(&mut unpacker);
        assert_eq!(params.received, NUM_TUNE_PARAMS);
        assert_eq!(params.jetpack_strength, 12345);
    }

    #[test]
    fn jetpack_strength_index_constant_matches_from_arrays_field_mapping() {
        // Ties `JETPACK_STRENGTH_INDEX` to `from_array`'s `v[33] -> jetpack_strength` mapping, so
        // the two can never silently drift apart if the field list is ever reordered.
        let mut probe = DEFAULTS;
        probe[JETPACK_STRENGTH_INDEX] = 0xBEEF;
        assert_eq!(from_array(0, probe).jetpack_strength, 0xBEEF);
    }

    #[test]
    fn decode_garbage_tune_params_never_panics() {
        for len in 0..40 {
            let bytes = vec![0xffu8; len];
            let mut unpacker = Unpacker::new(&bytes);
            let params = decode_sv_tune_params(&mut unpacker);
            assert!(params.received <= NUM_TUNE_PARAMS);
        }
    }

    #[test]
    fn decode_teams_state_full_valid() {
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        for i in 0..128 {
            packer.add_int(i % 3);
        }
        let mut unpacker = Unpacker::new(packer.data());
        let state = decode_teams_state(&mut unpacker);
        assert_eq!(state.received, 128);
        assert_eq!(state.teams[0], 0);
        assert_eq!(state.teams[1], 1);
        assert_eq!(state.teams[127], 127 % 3);
    }

    #[test]
    fn decode_teams_state_stops_at_out_of_range_value() {
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        packer.add_int(1);
        packer.add_int(2);
        packer.add_int(NUM_DDRACE_TEAMS); // one past the valid range
        packer.add_int(3); // never read
        let mut unpacker = Unpacker::new(packer.data());
        let state = decode_teams_state(&mut unpacker);
        assert_eq!(state.received, 2);
        assert_eq!(state.teams[0], 1);
        assert_eq!(state.teams[1], 2);
        assert_eq!(state.teams[2], 0);
    }

    #[test]
    fn decode_teams_state_stops_at_negative_value() {
        let mut buf = [0u8; 4096];
        let mut packer = Packer::new(&mut buf);
        packer.add_int(-1);
        let mut unpacker = Unpacker::new(packer.data());
        let state = decode_teams_state(&mut unpacker);
        assert_eq!(state.received, 0);
    }

    #[test]
    fn decode_teams_state_garbage_never_panics() {
        for len in 0..40 {
            let bytes = vec![0xffu8; len];
            let mut unpacker = Unpacker::new(&bytes);
            let state = decode_teams_state(&mut unpacker);
            assert!(state.received <= 128);
        }
    }
}
