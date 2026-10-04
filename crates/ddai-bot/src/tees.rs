//! The bot's own flat view of every tee at the snapshot tick ("`world.getTee`" of the TS bot):
//! exact position/velocity/hook/freeze from the reconstructed base world (reckoning-evolved, so not
//! the quantised snapshot numbers), plus the two values only the wire `Character` carries — the aim
//! angle and the attack tick — which the activity clock and the target score need.
//!
//! Fixed-size storage, rebuilt in place each snapshot: no allocation.

use ddai_net::view::CharacterView;
use ddai_physics::core::{MAX_CLIENTS, WEAPON_HAMMER};
use ddai_physics::vmath::Vec2;
use ddai_physics::world::World;

use crate::consts::HOOK_LENGTH_PX;

/// `DEEP_FREEZE_TICKS` (`liveWorld.ts:131-193`): what a deep-frozen tee reports as time left.
pub const DEEP_FREEZE_TICKS: i32 = 150;

pub use ddai_brain::{HOOK_FLYING, HOOK_GRABBED, HOOK_IDLE, HOOK_RETRACT_END, HOOK_RETRACT_START, HOOK_RETRACTED};

/// One tee.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tee {
    pub id: i32,
    pub alive: bool,
    pub pos: Vec2<f32>,
    pub vel: Vec2<f32>,
    /// Frozen (normal or deep): it cannot act.
    pub frozen: bool,
    pub deep_frozen: bool,
    /// Ticks until it thaws (`DEEP_FREEZE_TICKS` for a deep freeze, which has no timer).
    pub freeze_ticks_left: i32,
    pub hook_state: i32,
    /// The client id its hook is attached to, or -1.
    pub hooked_player: i32,
    pub direction: i32,
    /// `Character::jumped` bits (1 = ground jump used, 2 = air jump used).
    pub jumped: i32,
    /// Wire aim angle, 1/256 rad in `0..2*pi*256`.
    pub angle: i32,
    pub attack_tick: i32,
    pub weapon: i32,
}

impl Tee {
    pub const DEAD: Tee = Tee {
        id: -1,
        alive: false,
        pos: Vec2 { x: 0.0, y: 0.0 },
        vel: Vec2 { x: 0.0, y: 0.0 },
        frozen: false,
        deep_frozen: false,
        freeze_ticks_left: 0,
        hook_state: HOOK_IDLE,
        hooked_player: -1,
        direction: 0,
        jumped: 0,
        angle: 0,
        attack_tick: 0,
        weapon: 0,
    };

    /// `wireAngleRad` (`core/types.ts:83`): the aim angle in radians, `(-pi, pi]`.
    pub fn aim_rad(&self) -> f32 {
        let a = self.angle as f32 / 256.0;
        if a >= std::f32::consts::PI {
            a - 2.0 * std::f32::consts::PI
        } else {
            a
        }
    }

    pub fn holding_hammer(&self) -> bool {
        self.weapon == WEAPON_HAMMER
    }
}

/// `inputKeysOf` (`bot.ts:321`): direction, ground-jump and "hook out" as one small integer; the
/// activity clock notices a change in it. `-1` for a frozen tee (its keys carry no information).
pub fn input_keys_of(t: &Tee) -> i32 {
    if t.frozen {
        return -1;
    }
    (t.direction + 1) | ((t.jumped & 1) << 2) | (i32::from(t.hook_state != HOOK_IDLE) << 3)
}

/// `NEUTRAL_KEYS` (`bot.ts`): the keys of a tee that holds nothing (direction 0, no jump, no hook).
const NEUTRAL_KEYS: i32 = 1;

/// `HELD_MOVE_MIN` (`bot.ts`): px a tee must have moved since the last snapshot for held keys to count
/// as activity (running, swinging on the hook) even when the keys themselves did not change.
pub const HELD_MOVE_MIN: f32 = 1.0;

/// `keysNeutral(keys)`.
pub fn keys_neutral(keys: i32) -> bool {
    keys == NEUTRAL_KEYS
}

/// Euclidean distance.
pub fn dist(a: Vec2<f32>, b: Vec2<f32>) -> f32 {
    (a.x - b.x).hypot(a.y - b.y)
}

/// All tees of one snapshot.
pub struct TeeSet {
    tees: Box<[Tee; MAX_CLIENTS]>,
    /// Alive ids, ascending.
    alive: Vec<u8>,
}

impl Default for TeeSet {
    fn default() -> Self {
        Self::new()
    }
}

impl TeeSet {
    pub fn new() -> Self {
        TeeSet {
            tees: Box::new([Tee::DEAD; MAX_CLIENTS]),
            alive: Vec::with_capacity(MAX_CLIENTS),
        }
    }

    /// Rebuilds from the base world (`LiveWorld::base_world()`) and the snapshot's characters.
    pub fn rebuild(&mut self, world: &World<f32>, characters: &[CharacterView]) {
        self.alive.clear();
        for t in self.tees.iter_mut() {
            t.alive = false;
        }
        for cv in characters {
            let Ok(idx) = usize::try_from(cv.id) else { continue };
            if idx >= MAX_CLIENTS {
                continue;
            }
            let (Some(core), Some(ch)) = (world.cores.get(idx as u8), world.characters[idx].as_ref()) else {
                continue;
            };
            let deep = core.deep_frozen;
            self.tees[idx] = Tee {
                id: cv.id,
                alive: true,
                pos: core.pos,
                vel: core.vel,
                frozen: ch.freeze_time > 0 || deep,
                deep_frozen: deep,
                freeze_ticks_left: if deep { DEEP_FREEZE_TICKS } else { ch.freeze_time.max(0) },
                hook_state: core.hook_state,
                hooked_player: core.hooked_player(),
                direction: core.direction,
                jumped: core.jumped,
                angle: cv.character.angle,
                attack_tick: cv.character.attack_tick,
                weapon: core.active_weapon,
            };
        }
        for (i, t) in self.tees.iter().enumerate() {
            if t.alive {
                self.alive.push(i as u8);
            }
        }
    }

    /// The alive tee `id`.
    pub fn get(&self, id: i32) -> Option<&Tee> {
        let t = self.tees.get(usize::try_from(id).ok()?)?;
        t.alive.then_some(t)
    }

    /// Alive tees, ascending id (the port's deterministic order; the TS used map-insertion order).
    pub fn iter(&self) -> impl Iterator<Item = &Tee> + '_ {
        self.alive.iter().map(|&i| &self.tees[i as usize])
    }

    pub fn len(&self) -> usize {
        self.alive.len()
    }

    pub fn is_empty(&self) -> bool {
        self.alive.is_empty()
    }

    /// Test helper: set a tee directly.
    #[cfg(test)]
    pub(crate) fn set_for_test(&mut self, tee: Tee) {
        let idx = tee.id as usize;
        self.tees[idx] = tee;
        self.alive.clear();
        for (i, t) in self.tees.iter().enumerate() {
            if t.alive {
                self.alive.push(i as u8);
            }
        }
    }
}

/// Whether a rope of `input`'s aim from `from` would catch `at` within hook reach (`ropeCatchAlong`,
/// `planner.ts:642-648`): distance along the aim direction, or infinity.
pub fn rope_catch_along(from: Vec2<f32>, dir: Vec2<f32>, at: Vec2<f32>) -> f32 {
    let (rx, ry) = (at.x - from.x, at.y - from.y);
    let along = rx * dir.x + ry * dir.y;
    if !(0.0..=HOOK_LENGTH_PX).contains(&along) {
        return f32::INFINITY;
    }
    if (rx * dir.y - ry * dir.x).abs() <= crate::consts::ROPE_CATCH_PX {
        along
    } else {
        f32::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tee(id: i32, x: f32, y: f32) -> Tee {
        Tee {
            id,
            alive: true,
            pos: Vec2::new(x, y),
            ..Tee::DEAD
        }
    }

    #[test]
    fn keys_encode_direction_jump_and_hook_and_freeze_hides_them() {
        let mut t = tee(0, 0.0, 0.0);
        assert_eq!(input_keys_of(&t), 1, "direction 0 -> 1");
        t.direction = 1;
        t.jumped = 1;
        t.hook_state = HOOK_FLYING;
        assert_eq!(input_keys_of(&t), 2 | 4 | 8);
        t.frozen = true;
        assert_eq!(input_keys_of(&t), -1);
    }

    #[test]
    fn aim_angle_wraps_like_wire_angle_rad() {
        let mut t = tee(0, 0.0, 0.0);
        t.angle = 0;
        assert_eq!(t.aim_rad(), 0.0);
        t.angle = (std::f32::consts::FRAC_PI_2 * 256.0) as i32;
        assert!((t.aim_rad() - std::f32::consts::FRAC_PI_2).abs() < 0.01);
        t.angle = (1.5 * std::f32::consts::PI * 256.0) as i32;
        assert!(t.aim_rad() < 0.0, "above pi wraps to negative");
        assert!((t.aim_rad() + std::f32::consts::FRAC_PI_2).abs() < 0.01);
    }

    #[test]
    fn rope_catch_needs_the_tee_on_the_line_within_reach() {
        let from = Vec2::new(0.0, 0.0);
        let right = Vec2::new(1.0, 0.0);
        assert_eq!(rope_catch_along(from, right, Vec2::new(200.0, 10.0)), 200.0);
        assert!(
            rope_catch_along(from, right, Vec2::new(200.0, 40.0)).is_infinite(),
            "off the line"
        );
        assert!(
            rope_catch_along(from, right, Vec2::new(-5.0, 0.0)).is_infinite(),
            "behind"
        );
        assert!(
            rope_catch_along(from, right, Vec2::new(400.0, 0.0)).is_infinite(),
            "beyond reach"
        );
    }

    #[test]
    fn tee_set_iterates_alive_ascending() {
        let mut s = TeeSet::new();
        s.set_for_test(tee(5, 0.0, 0.0));
        s.set_for_test(tee(2, 0.0, 0.0));
        let ids: Vec<i32> = s.iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![2, 5]);
        assert!(s.get(3).is_none());
        assert_eq!(s.len(), 2);
    }
}
