//! Literal port of `src/core/tuning.ts` (TS, `Wranked1/DDNet-AI`, GPL-3.0).
//!
//! **Known TS quirk, reproduced on purpose (not fixed):** every `TUNING` constant is passed
//! through `tune(v) = Math.trunc(v * 100) / 100` (`tuning.ts:5-7`) in `f64` — this is a
//! rounding-to-2-decimals step done in *double* precision, at *module load time*, not the
//! server's own `f32` fixed-point tuning representation. Some products (e.g.
//! `100.0 / SERVER_TICK_SPEED = 2.0` exactly) round trivially; others do not, and the point of
//! this crate is that TS's own value is what a ported planner must match, not what the "real"
//! tuning value would be. See `docs/research/orig-plan.md` §3 and `docs/DECISIONS.md` D-035.

use ddai_jsmath as js;

/// `SERVER_TICK_SPEED` (`tuning.ts:1`).
pub const SERVER_TICK_SPEED: f64 = 50.0;
/// `PHYSICAL_SIZE` (`tuning.ts:2`).
pub const PHYSICAL_SIZE: f64 = 28.0;
/// `TICK_MS` (`tuning.ts:3`). Unused by the ported core files' own math (kept for API parity —
/// no call site in `src/core/*.ts` reads it, only documentation/comments do).
pub const TICK_MS: f64 = 20.0;

/// `tune(v)` (`tuning.ts:5-7`). Deliberately **not** `(v * 100.0).trunc() / 100.0` written
/// inline at each call site below: keeping it as one named function makes the "every constant
/// goes through this" invariant checkable by inspection, and keeps every call going through
/// `ddai_jsmath::trunc` per the crate-wide house rule.
fn tune(v: f64) -> f64 {
    js::trunc(v * 100.0) / 100.0
}

/// Mirrors the TS `TUNING` object (`tuning.ts:9-66`), field for field, in the same order. Every
/// field already has `tune()` applied — see this module's top-level doc comment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tuning {
    pub ground_control_speed: f64,
    pub ground_control_accel: f64,
    pub ground_friction: f64,
    pub ground_jump_impulse: f64,
    pub air_jump_impulse: f64,
    pub air_control_speed: f64,
    pub air_control_accel: f64,
    pub air_friction: f64,
    pub hook_length: f64,
    pub hook_fire_speed: f64,
    pub hook_drag_accel: f64,
    pub hook_drag_speed: f64,
    pub gravity: f64,

    pub velramp_start: f64,
    pub velramp_range: f64,
    pub velramp_curvature: f64,

    pub gun_curvature: f64,
    pub gun_speed: f64,
    pub gun_lifetime: f64,

    pub shotgun_curvature: f64,
    pub shotgun_speed: f64,
    pub shotgun_speeddiff: f64,
    pub shotgun_lifetime: f64,

    pub grenade_curvature: f64,
    pub grenade_speed: f64,
    pub grenade_lifetime: f64,

    pub laser_reach: f64,
    pub laser_bounce_delay: f64,
    pub laser_bounce_num: f64,
    pub laser_bounce_cost: f64,
    pub laser_damage: f64,

    pub player_collision: f64,
    pub player_hooking: f64,

    pub jetpack_strength: f64,
    pub shotgun_strength: f64,
    pub explosion_strength: f64,
    pub hammer_strength: f64,
    pub hook_duration: f64,

    pub hammer_fire_delay: f64,
    pub gun_fire_delay: f64,
    pub shotgun_fire_delay: f64,
    pub grenade_fire_delay: f64,
    pub laser_fire_delay: f64,
    pub ninja_fire_delay: f64,
    pub hammer_hit_fire_delay: f64,

    pub ground_elasticity_x: f64,
    pub ground_elasticity_y: f64,
}

/// `TUNING` (`tuning.ts:9-66`). A `const fn` is not possible here (`js::trunc` is not `const`),
/// so this is a plain function; every caller in this crate treats its result as if it were the
/// frozen TS constant (never mutated after construction — TS's `as const` object is likewise
/// never mutated by any file this crate ports).
///
/// Computed once and cached (`OnceLock`) — review finding N1: `Tuning` is 40 `f64` fields, and an
/// earlier version of this crate rebuilt it (including several `js::trunc` calls) on every single
/// call, including from inside `core_tick_deferred`'s per-pair loop — i.e. up to `O(n²)` times per
/// tick for `n` tees. The value is deterministic and never changes at runtime, so computing it
/// once is observably identical, just cheaper.
pub fn tuning() -> Tuning {
    static TUNING: std::sync::OnceLock<Tuning> = std::sync::OnceLock::new();
    *TUNING.get_or_init(build_tuning)
}

fn build_tuning() -> Tuning {
    Tuning {
        ground_control_speed: tune(10.0),
        ground_control_accel: tune(100.0 / SERVER_TICK_SPEED),
        ground_friction: tune(0.5),
        ground_jump_impulse: tune(13.2),
        air_jump_impulse: tune(12.0),
        air_control_speed: tune(250.0 / SERVER_TICK_SPEED),
        air_control_accel: tune(1.5),
        air_friction: tune(0.95),
        hook_length: tune(380.0),
        hook_fire_speed: tune(80.0),
        hook_drag_accel: tune(3.0),
        hook_drag_speed: tune(15.0),
        gravity: tune(0.5),

        velramp_start: tune(550.0),
        velramp_range: tune(2000.0),
        velramp_curvature: tune(1.4),

        gun_curvature: tune(1.25),
        gun_speed: tune(2200.0),
        gun_lifetime: tune(2.0),

        shotgun_curvature: tune(1.25),
        shotgun_speed: tune(2750.0),
        shotgun_speeddiff: tune(0.8),
        shotgun_lifetime: tune(0.2),

        grenade_curvature: tune(7.0),
        grenade_speed: tune(1000.0),
        grenade_lifetime: tune(2.0),

        laser_reach: tune(800.0),
        laser_bounce_delay: tune(150.0),
        laser_bounce_num: tune(1000.0),
        laser_bounce_cost: tune(0.0),
        laser_damage: tune(5.0),

        player_collision: tune(1.0),
        player_hooking: tune(1.0),

        jetpack_strength: tune(400.0),
        shotgun_strength: tune(10.0),
        explosion_strength: tune(6.0),
        hammer_strength: tune(1.0),
        hook_duration: tune(1.25),

        hammer_fire_delay: tune(125.0),
        gun_fire_delay: tune(125.0),
        shotgun_fire_delay: tune(500.0),
        grenade_fire_delay: tune(500.0),
        laser_fire_delay: tune(800.0),
        ninja_fire_delay: tune(800.0),
        hammer_hit_fire_delay: tune(320.0),

        ground_elasticity_x: tune(0.0),
        ground_elasticity_y: tune(0.0),
    }
}

// --- Tile ids (`tuning.ts:68-98`) --------------------------------------------------------------

pub const TILE_AIR: u8 = 0;
pub const TILE_SOLID: u8 = 1;
pub const TILE_DEATH: u8 = 2;
pub const TILE_NOHOOK: u8 = 3;
pub const TILE_FREEZE: u8 = 9;
pub const TILE_UNFREEZE: u8 = 11;

pub const TILE_DFREEZE: u8 = 12;
pub const TILE_DUNFREEZE: u8 = 13;

pub const TILE_TELEINEVIL: u8 = 10;
pub const TILE_TELEIN: u8 = 26;
pub const TILE_TELEOUT: u8 = 27;

pub const TILE_TELECHECK: u8 = 29;
pub const TILE_TELECHECKOUT: u8 = 30;
pub const TILE_TELECHECKIN: u8 = 31;
pub const TILE_TELECHECKINEVIL: u8 = 63;

pub const TILE_TELE_LASER_DISABLE: u8 = 129;
pub const TILE_LFREEZE: u8 = 144;
pub const TILE_LUNFREEZE: u8 = 145;

pub const TILE_THROUGH_CUT: u8 = 5;
pub const TILE_THROUGH: u8 = 6;
pub const TILE_THROUGH_ALL: u8 = 66;
pub const TILE_THROUGH_DIR: u8 = 67;

pub const TILE_STOP: u8 = 60;
pub const TILE_STOPS: u8 = 61;
pub const TILE_STOPA: u8 = 62;

pub const TILEFLAG_XFLIP: u8 = 1 << 0;
pub const TILEFLAG_YFLIP: u8 = 1 << 1;
pub const TILEFLAG_ROTATE: u8 = 1 << 3;
pub const ROTATION_0: u8 = 0;
pub const ROTATION_90: u8 = TILEFLAG_ROTATE;
pub const ROTATION_180: u8 = TILEFLAG_XFLIP | TILEFLAG_YFLIP;
pub const ROTATION_270: u8 = TILEFLAG_XFLIP | TILEFLAG_YFLIP | TILEFLAG_ROTATE;

pub const CANTMOVE_LEFT: i32 = 1 << 0;
pub const CANTMOVE_RIGHT: i32 = 1 << 1;
pub const CANTMOVE_UP: i32 = 1 << 2;
pub const CANTMOVE_DOWN: i32 = 1 << 3;

pub const CFLAG_SOLID: i32 = 1 << 0;
pub const CFLAG_DEATH: i32 = 1 << 1;
pub const CFLAG_NOHOOK: i32 = 1 << 2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_control_accel_rounds_to_two_exactly() {
        // 100.0 / 50.0 == 2.0 exactly, trunc(2.0 * 100.0) / 100.0 == 2.0 — a case with no
        // observable rounding, kept as a smoke test that `tuning()` doesn't accidentally shift
        // by an order of magnitude.
        assert_eq!(tuning().ground_control_accel, 2.0);
    }

    #[test]
    fn air_control_speed_matches_the_known_ts_runtime_value() {
        // 250.0 / 50.0 == 5.0 exactly as well.
        assert_eq!(tuning().air_control_speed, 5.0);
    }

    #[test]
    fn velramp_curvature_is_1_4() {
        assert_eq!(tuning().velramp_curvature, 1.4);
    }

    #[test]
    fn rotation_constants_match_ts() {
        assert_eq!(ROTATION_180, 3);
        assert_eq!(ROTATION_270, 11);
    }
}
