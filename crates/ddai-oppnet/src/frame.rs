//! What a live snapshot shows of one tee ([`TeeFrame`]), the input a tee applied in one step
//! ([`InputRec`]), and the geometry rays around a tee ([`rays`]).
//!
//! A [`TeeFrame`] holds only what the server's snapshot (the character core plus the DDNet extension) carries:
//! position, velocity, aim angle, direction, jump bits, hook state, freeze, the tick of the last
//! weapon use. The opponent's raw inputs (jump, hook, fire keys) are **not** in it -- the predictor
//! must infer them, as the live bot would.

use ddai_physics::core::{self, PlayerInput as WireInput};
use ddai_physics::world::World;
use serde::{Deserialize, Serialize};

/// Ray directions around a tee (unit vectors, clockwise from +x; y points down).
const S: f32 = std::f32::consts::FRAC_1_SQRT_2;
const DIRS: [(f32, f32); N_DIRS] = [
    (1.0, 0.0),
    (S, S),
    (0.0, 1.0),
    (-S, S),
    (-1.0, 0.0),
    (-S, -S),
    (0.0, -1.0),
    (S, -S),
];
pub const N_DIRS: usize = 8;
/// Two values per direction: the distance to the first solid tile and to the first freeze/death tile.
pub const N_RAYS: usize = 2 * N_DIRS;
/// Ray step (px; half a tile, so no tile is skipped) and reach (px).
pub const RAY_STEP: f32 = 16.0;
pub const RAY_REACH: f32 = 320.0;

/// One tee as a snapshot shows it.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct TeeFrame {
    pub alive: bool,
    pub pos: [f32; 2],
    /// Pixels per tick.
    pub vel: [f32; 2],
    /// Aim angle in radians in `[0, 2 pi)` (the wire's 1/256 rad).
    pub angle: f32,
    /// `-1`, `0`, `1`: the direction input of the last applied step.
    pub direction: i8,
    /// `core.jumped` bits.
    pub jumped: u8,
    pub hook_state: i8,
    /// The tee is hooking the other tee of the pair.
    pub hook_on_other: bool,
    pub hook_pos: [f32; 2],
    pub hook_tick: i16,
    /// Ticks until it thaws; `0` = free.
    pub freeze_left: i16,
    /// Ticks since its last weapon use (clipped to 120).
    pub attack_age: i16,
    pub grounded: bool,
}

impl TeeFrame {
    /// The frame of tee `id` in `world`, `other` being the tee it is paired with (for `hook_on_other`);
    /// `None` when `id` has no live character.
    pub fn from_world(world: &World<f32>, id: i32, other: i32) -> Option<TeeFrame> {
        if !(0..core::MAX_CLIENTS as i32).contains(&id) {
            return None;
        }
        let c = world.cores.get(id as u8)?;
        let ch = world.characters[id as usize].as_ref()?;
        let hooked = c.hooked_player();
        Some(TeeFrame {
            alive: ch.alive,
            pos: [c.pos.x, c.pos.y],
            vel: [c.vel.x, c.vel.y],
            angle: c.angle as f32 / 256.0,
            direction: c.direction.clamp(-1, 1) as i8,
            jumped: (c.jumped & 3) as u8,
            hook_state: c.hook_state.clamp(-1, 8) as i8,
            hook_on_other: hooked >= 0 && hooked == other,
            hook_pos: [c.hook_pos.x, c.hook_pos.y],
            hook_tick: c.hook_tick.clamp(0, i32::from(i16::MAX)) as i16,
            freeze_left: ch.freeze_time.clamp(0, i32::from(i16::MAX)) as i16,
            attack_age: (world.tick - ch.attack_tick).clamp(0, 120) as i16,
            grounded: world.collision.is_on_ground(c.pos, core::physical_size::<f32>()),
        })
    }
}

/// The input a tee applied in one world step (the wire values the physics read).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct InputRec {
    pub direction: i8,
    pub jump: bool,
    pub hook: bool,
    /// The fire press counter (odd = held).
    pub fire: i32,
    pub target_x: i16,
    pub target_y: i16,
}

impl InputRec {
    pub fn from_wire(i: &WireInput) -> InputRec {
        InputRec {
            direction: i.direction.clamp(-1, 1) as i8,
            jump: i.jump != 0,
            hook: i.hook != 0,
            fire: i.fire,
            target_x: i.target_x.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
            target_y: i.target_y.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
        }
    }
}

/// Distances (normalised to [`RAY_REACH`], `1` = nothing within it) from `pos` along [`N_DIRS`] directions to the first solid tile
/// (even indices) and the first freeze/death tile (odd indices). A fixed, allocation-free scan.
pub fn rays(world: &World<f32>, pos: [f32; 2], out: &mut [f32; N_RAYS]) {
    let col = &world.collision;
    for (d, &(dx, dy)) in DIRS.iter().enumerate() {
        let (mut solid, mut hazard) = (1.0f32, 1.0f32);
        let steps = (RAY_REACH / RAY_STEP) as i32;
        for s in 1..=steps {
            let t = RAY_STEP * s as f32;
            let (x, y) = (pos[0] + dx * t, pos[1] + dy * t);
            let (xi, yi) = (x.floor() as i32, y.floor() as i32);
            if solid >= 1.0 && col.is_solid(xi, yi) {
                solid = t / RAY_REACH;
            }
            if hazard >= 1.0 && col.hazard_tile(xi.div_euclid(32), yi.div_euclid(32)) {
                hazard = t / RAY_REACH;
            }
            if solid < 1.0 && hazard < 1.0 {
                break;
            }
        }
        out[2 * d] = solid;
        out[2 * d + 1] = hazard;
    }
}
