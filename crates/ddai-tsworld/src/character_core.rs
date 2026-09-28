//! Literal port of `src/core/characterCore.ts` (TS, `Wranked1/DDNet-AI`, GPL-3.0).
//!
//! **Restructuring vs. TS (allowed by the task spec — "data structures may be idiomatic Rust...
//! but observable state and iteration order must equal TS"):** in TS, `CharacterCore` holds a
//! `world: CoreWorld` back-reference and calls back into it (`this.world.allCores()`,
//! `this.world.coreById(...)`) from `tick`/`tickDeferred`/`move`/`setHookedPlayer`. A field
//! holding a live, aliasable reference to its own container is not expressible in safe Rust
//! without `Rc<RefCell<_>>` (which this project avoids) or `unsafe` (forbidden by
//! `docs/CLAUDE.md`). This module therefore holds only the **plain data** every `CharacterCore`
//! carries (this file) plus the handful of pure helper functions
//! ([`saturated_add`], [`velocity_ramp`]) that don't touch other tees; the orchestration that
//! *does* need simultaneous access to several tees' cores (`tick`, `tick_deferred`, `move_`,
//! `set_hooked_player`) is ported instead as [`crate::world::SimWorld`] methods, in
//! `src/world.rs`, where that access is a matter of indexing one storage `Vec` rather than
//! following object references — see that module's doc comments for the exact TS-line mapping
//! of each such method. Every number this struct stores and every operation performed on it is
//! otherwise unchanged from TS; only *where the code that mutates it lives* moved.

use crate::types::{PlayerInput, empty_input};
use crate::vmath::Vec2;

pub const HOOK_RETRACTED: i32 = -1;
pub const HOOK_IDLE: i32 = 0;
pub const HOOK_RETRACT_START: i32 = 1;
pub const HOOK_RETRACT_END: i32 = 3;
pub const HOOK_FLYING: i32 = 4;
pub const HOOK_GRABBED: i32 = 5;

pub const COREEVENT_GROUND_JUMP: u32 = 0x01;
pub const COREEVENT_AIR_JUMP: u32 = 0x02;
pub const COREEVENT_HOOK_LAUNCH: u32 = 0x04;
pub const COREEVENT_HOOK_ATTACH_PLAYER: u32 = 0x08;
pub const COREEVENT_HOOK_ATTACH_GROUND: u32 = 0x10;
pub const COREEVENT_HOOK_HIT_NOHOOK: u32 = 0x20;
pub const COREEVENT_HOOK_RETRACT: u32 = 0x40;

/// The physical box `move()` sweeps through the map (`characterCore.ts:48`, `MOVE_SIZE`).
pub const MOVE_SIZE: Vec2 = Vec2 {
    x: crate::tuning::PHYSICAL_SIZE,
    y: crate::tuning::PHYSICAL_SIZE,
};

/// A `Set<number>` used only as `attachedPlayers` (`characterCore.ts:65`): unique `i32`s in
/// **insertion order**, exactly like a JS `Set` — `add`ing an already-present value is a no-op
/// (does not move it), `delete` removes it without shifting the remaining order. `Array.from` on
/// a JS `Set` (used by `world.ts`'s `saveState`, `world.ts:277`) yields this same order, so
/// [`Self::iter`] is what a byte-exact `saveState` port must serialize.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OrderedSet {
    items: Vec<i32>,
}

impl OrderedSet {
    pub fn new() -> Self {
        OrderedSet { items: Vec::new() }
    }

    /// `Set.prototype.add`.
    pub fn add(&mut self, v: i32) {
        if !self.items.contains(&v) {
            self.items.push(v);
        }
    }

    /// `Set.prototype.delete`.
    pub fn delete(&mut self, v: i32) {
        self.items.retain(|&x| x != v);
    }

    /// `Set.prototype.clear`.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = i32> + '_ {
        self.items.iter().copied()
    }

    pub fn contains(&self, v: i32) -> bool {
        self.items.contains(&v)
    }
}

/// `class CharacterCore` (`characterCore.ts:51-449`) — data fields only, in TS field order. See
/// the module doc comment for why the methods that need other tees' data live on `SimWorld`
/// instead.
#[derive(Debug, Clone)]
pub struct CharacterCore {
    pub id: i32,

    pub pos: Vec2,
    pub vel: Vec2,

    pub hook_pos: Vec2,
    pub hook_dir: Vec2,
    pub hook_tele_base: Vec2,
    pub hook_tick: i64,
    pub hook_state: i32,
    pub hooked_player: i32,
    pub attached_players: OrderedSet,

    pub active_weapon: i32,

    pub new_hook: bool,

    pub move_restrictions: i32,

    pub jumped: i32,
    pub jumped_total: i32,
    pub jumps: i32,

    pub direction: i32,
    pub angle: i32,
    pub input: PlayerInput,

    pub triggered_events: u32,

    pub colliding: i32,
    pub left_wall: bool,

    pub freeze_start: i64,
    pub freeze_end: i64,
    pub is_in_freeze: bool,
    pub collision_disabled: bool,
    pub hook_hit_disabled: bool,
    pub endless_hook: bool,
    pub solo: bool,
}

impl CharacterCore {
    /// `constructor(id, collision, world)` (`characterCore.ts:94-98`) plus the field initializers
    /// at declaration (`characterCore.ts:56-92`) — `collision`/`world` are not stored (see the
    /// module doc comment).
    pub fn new(id: i32) -> Self {
        CharacterCore {
            id,
            pos: Vec2 { x: 0.0, y: 0.0 },
            vel: Vec2 { x: 0.0, y: 0.0 },
            hook_pos: Vec2 { x: 0.0, y: 0.0 },
            hook_dir: Vec2 { x: 0.0, y: 0.0 },
            hook_tele_base: Vec2 { x: 0.0, y: 0.0 },
            hook_tick: 0,
            hook_state: HOOK_IDLE,
            hooked_player: -1,
            attached_players: OrderedSet::new(),
            active_weapon: 0,
            new_hook: false,
            move_restrictions: 0,
            jumped: 0,
            jumped_total: 0,
            jumps: 2,
            direction: 0,
            angle: 0,
            input: empty_input(),
            triggered_events: 0,
            colliding: 0,
            left_wall: false,
            freeze_start: 0,
            freeze_end: 0,
            is_in_freeze: false,
            collision_disabled: false,
            hook_hit_disabled: false,
            endless_hook: false,
            solo: false,
        }
    }

    /// `reset()` (`characterCore.ts:100-126`). Does **not** reset `hookedPlayer`'s effect on
    /// *other* cores' `attachedPlayers` the way `setHookedPlayer(-1)` would (TS calls
    /// `this.setHookedPlayer(-1)` here, `characterCore.ts:110`, which — since `world.coreById`
    /// looks the *other* tee up live — does reach into the other core; ported at the
    /// [`crate::world::SimWorld::reset`] call site instead, immediately before this method, so
    /// the net effect on every core's `attachedPlayers` is identical).
    pub fn reset(&mut self) {
        self.pos = Vec2 { x: 0.0, y: 0.0 };
        self.vel = Vec2 { x: 0.0, y: 0.0 };
        self.new_hook = false;
        self.move_restrictions = 0;
        self.hook_pos = Vec2 { x: 0.0, y: 0.0 };
        self.hook_dir = Vec2 { x: 0.0, y: 0.0 };
        self.hook_tele_base = Vec2 { x: 0.0, y: 0.0 };
        self.hook_tick = 0;
        self.hook_state = HOOK_IDLE;
        self.hooked_player = -1;
        self.attached_players.clear();
        self.jumped = 0;
        self.jumped_total = 0;
        self.jumps = 2;
        self.triggered_events = 0;

        self.solo = false;
        self.collision_disabled = false;
        self.endless_hook = false;
        self.hook_hit_disabled = false;
        self.freeze_start = 0;
        self.freeze_end = 0;
        self.is_in_freeze = false;

        self.input = empty_input();
    }

    /// `quantize()` (`characterCore.ts:420-434`).
    pub fn quantize(&mut self) {
        use crate::vmath::round_to_int;
        let x = round_to_int(self.pos.x);
        let y = round_to_int(self.pos.y);
        let vel_x = round_to_int(self.vel.x * 256.0);
        let vel_y = round_to_int(self.vel.y * 256.0);
        let hook_x = round_to_int(self.hook_pos.x);
        let hook_y = round_to_int(self.hook_pos.y);
        let hook_dx = round_to_int(self.hook_dir.x * 256.0);
        let hook_dy = round_to_int(self.hook_dir.y * 256.0);

        self.pos = Vec2 { x, y };
        self.vel = Vec2 {
            x: vel_x / 256.0,
            y: vel_y / 256.0,
        };
        self.hook_pos = Vec2 { x: hook_x, y: hook_y };
        self.hook_dir = Vec2 {
            x: hook_dx / 256.0,
            y: hook_dy / 256.0,
        };
    }
}

/// `saturatedAdd(min, max, current, modifier)` (`characterCore.ts:30-41`).
pub fn saturated_add(min: f64, max: f64, current: f64, modifier: f64) -> f64 {
    let mut current = current;
    if modifier < 0.0 {
        if current < min {
            return current;
        }
        current += modifier;
        if current < min {
            current = min;
        }
        return current;
    }
    if current > max {
        return current;
    }
    current += modifier;
    if current > max {
        current = max;
    }
    current
}

/// `velocityRamp(value, start, range, curvature)` (`characterCore.ts:43-46`).
pub fn velocity_ramp(value: f64, start: f64, range: f64, curvature: f64) -> f64 {
    if value < start {
        return 1.0;
    }
    1.0 / ddai_jsmath::pow(curvature, (value - start) / range)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_set_preserves_insertion_order_and_delete_does_not_reindex() {
        let mut s = OrderedSet::new();
        s.add(3);
        s.add(1);
        s.add(3); // no-op, does not move 3 to the back
        s.add(2);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![3, 1, 2]);
        s.delete(1);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![3, 2]);
        s.add(1);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![3, 2, 1]);
    }

    #[test]
    fn saturated_add_clamps_at_bounds() {
        assert_eq!(saturated_add(-10.0, 10.0, 9.0, 5.0), 10.0);
        assert_eq!(saturated_add(-10.0, 10.0, -9.0, -5.0), -10.0);
        // current already beyond max with a positive modifier: returned unchanged.
        assert_eq!(saturated_add(-10.0, 10.0, 20.0, 5.0), 20.0);
    }

    #[test]
    fn velocity_ramp_is_one_below_start() {
        assert_eq!(velocity_ramp(100.0, 550.0, 2000.0, 1.4), 1.0);
        assert!(velocity_ramp(3000.0, 550.0, 2000.0, 1.4) < 1.0);
    }

    #[test]
    fn quantize_rounds_position_and_scales_velocity() {
        let mut c = CharacterCore::new(1);
        c.pos = Vec2 { x: 10.4, y: -10.6 };
        c.vel = Vec2 { x: 1.0, y: -1.0 };
        c.quantize();
        assert_eq!(c.pos, Vec2 { x: 10.0, y: -11.0 });
        assert_eq!(c.vel, Vec2 { x: 1.0, y: -1.0 });
    }
}
