//! Literal port of `src/core/types.ts` (TS, `Wranked1/DDNet-AI`, GPL-3.0).

use crate::vmath::Vec2;

/// `PlayerInput` (`types.ts:4-15`). Field order matches TS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerInput {
    pub direction: i32,
    pub target_x: f64,
    pub target_y: f64,
    pub jump: i32,
    pub fire: i32,
    pub hook: i32,
    pub player_flags: i32,
    pub wanted_weapon: i32,
    pub next_weapon: i32,
    pub prev_weapon: i32,
}

/// `emptyInput()` (`types.ts:26-39`).
pub fn empty_input() -> PlayerInput {
    PlayerInput {
        direction: 0,
        target_x: 0.0,
        target_y: -1.0,
        jump: 0,
        fire: 0,
        hook: 0,
        player_flags: 0,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
    }
}

/// Copies every field of `from` into `to`, matching `world.ts`'s `copyInput` helper
/// (`world.ts:183-194`) — used everywhere TS mutates an existing `PlayerInput` object in place
/// instead of allocating a new one (`setInput`, `setHeldInput`, `preTick`'s frozen-input path,
/// the per-tick `prevInputForEdge` copy).
pub fn copy_input(from: &PlayerInput, to: &mut PlayerInput) {
    *to = *from;
}

/// `TeeState` (`types.ts:41-75`). Optional TS fields (`hookTick?`, `jumpedTotal?`, ...) become
/// `Option<_>`. Field order matches TS.
#[derive(Debug, Clone, PartialEq)]
pub struct TeeState {
    pub id: i32,
    pub alive: bool,
    pub pos: Vec2,
    pub vel: Vec2,
    pub hook_state: i32,
    pub hook_pos: Vec2,
    pub hook_dir: Vec2,
    pub hooked_player: i32,
    pub jumped: i32,
    pub jumps_left: i32,
    pub direction: i32,

    pub angle: f64,
    pub active_weapon: i32,
    pub frozen: bool,
    pub freeze_ticks_left: i64,
    pub attack_tick: i64,

    pub hook_tick: Option<i64>,
    pub jumped_total: Option<i32>,
    pub reload_ticks: Option<i64>,
    pub frozen_for: Option<i64>,
    pub deep_frozen: Option<bool>,
    pub jumps: Option<i32>,
    pub ddnet_flags: Option<i32>,
    pub since_attack: Option<i64>,
}

/// `blankTeeState()` (`types.ts:17-24`).
pub fn blank_tee_state() -> TeeState {
    TeeState {
        id: 0,
        alive: false,
        pos: Vec2 { x: 0.0, y: 0.0 },
        vel: Vec2 { x: 0.0, y: 0.0 },
        hook_state: 0,
        hook_pos: Vec2 { x: 0.0, y: 0.0 },
        hook_dir: Vec2 { x: 0.0, y: 0.0 },
        hooked_player: -1,
        jumped: 0,
        jumps_left: 0,
        direction: 0,
        angle: 0.0,
        active_weapon: 0,
        frozen: false,
        freeze_ticks_left: 0,
        attack_tick: 0,
        hook_tick: None,
        jumped_total: None,
        reload_ticks: None,
        frozen_for: None,
        deep_frozen: None,
        jumps: None,
        ddnet_flags: None,
        since_attack: None,
    }
}

pub const CHARACTERFLAG_SOLO: i32 = 1 << 0;
pub const CHARACTERFLAG_COLLISION_DISABLED: i32 = 1 << 2;
pub const CHARACTERFLAG_ENDLESS_HOOK: i32 = 1 << 3;
pub const CHARACTERFLAG_HOOK_HIT_DISABLED: i32 = 1 << 10;
pub const CHARACTERFLAG_WEAPON_NINJA: i32 = 1 << 19;

/// `wireAngleRad(angle)` (`types.ts:83-86`). Unused by the ported core files directly (kept for
/// API parity — `src/env/obs.ts` calls it on a `TeeState.angle`).
pub fn wire_angle_rad(angle: f64) -> f64 {
    let a = angle / 256.0;
    if a >= ddai_jsmath::PI {
        a - 2.0 * ddai_jsmath::PI
    } else {
        a
    }
}

/// `ProjectileState` (`types.ts:88-101`) — the *public* snapshot `SimWorld::projectiles()`
/// returns, distinct from `world.rs`'s internal `ProjectileState2` (`saveState`'s wire format).
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectileState {
    pub id: i32,
    pub kind: i32,
    pub owner: i32,
    pub pos: Vec2,
    pub vel: Vec2,
    pub dir: Vec2,
    pub start_tick: i64,
    pub spawn_pos: Vec2,
}

/// `WorldEvent` (`types.ts:103-111`).
#[derive(Debug, Clone, PartialEq)]
pub enum WorldEvent {
    HammerHit { from: i32, to: i32 },
    HammerFire { from: i32, hits: i32 },
    Explosion { pos: Vec2, owner: i32 },
    LaserHit { from: i32, to: i32, weapon: i32 },
    Freeze { id: i32, by: i32 },
    Death { id: i32, by: i32 },
}

pub const WEAPON_HAMMER: i32 = 0;
pub const WEAPON_GUN: i32 = 1;
pub const WEAPON_SHOTGUN: i32 = 2;
pub const WEAPON_GRENADE: i32 = 3;
pub const WEAPON_LASER: i32 = 4;
pub const WEAPON_NINJA: i32 = 5;
pub const NUM_WEAPONS: i32 = 6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_target_defaults_to_up() {
        let i = empty_input();
        assert_eq!(i.target_x, 0.0);
        assert_eq!(i.target_y, -1.0);
    }

    #[test]
    fn blank_tee_state_hooked_player_is_minus_one() {
        assert_eq!(blank_tee_state().hooked_player, -1);
    }

    #[test]
    fn wire_angle_rad_wraps_at_pi() {
        // angle stored as Math.trunc(x * 256); wireAngleRad divides back by 256 and wraps into
        // (-pi, pi] by subtracting a full turn when >= pi.
        let a = wire_angle_rad(256.0 * ddai_jsmath::PI);
        assert!(a < ddai_jsmath::PI);
    }
}
