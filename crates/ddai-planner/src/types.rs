//! Plain data types shared by every module here, mirroring `src/core/types.ts`'s `PlayerInput`/
//! `TeeState`/`WorldEvent` (the subset `src/plan`/`src/env` actually touch) field-for-field, so
//! the [`crate::plan_world::PlanWorld`] impls (`ts_adapter`, `physics_adapter`) are thin
//! conversions rather than reshuffles.

use crate::vmath::Vec2;

pub const WEAPON_HAMMER: i32 = 0;
pub const WEAPON_GUN: i32 = 1;

pub const HOOK_RETRACTED: i32 = -1;
pub const HOOK_IDLE: i32 = 0;
pub const HOOK_RETRACT_START: i32 = 1;
pub const HOOK_RETRACT_END: i32 = 3;
pub const HOOK_FLYING: i32 = 4;
pub const HOOK_GRABBED: i32 = 5;

/// `PlayerInput` (`types.ts:4-15`). Field order matches TS/`ddai-tsworld`.
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

/// `TeeState` (`types.ts:41-75`). Optional TS fields become `Option<_>`.
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// `WorldEvent` (`types.ts:103-111`) — only the variants `scoreTick` reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorldEvent {
    HammerHit {
        from: i32,
        to: i32,
    },
    HammerFire {
        from: i32,
        hits: i32,
    },
    Death {
        id: i32,
        by: i32,
    },
    /// Kept for structural completeness (`Explosion`/`LaserHit`/`Freeze` are part of the real
    /// `WorldEvent` union but never matched by `scoreTick`); a backend may still emit them.
    Other,
}
