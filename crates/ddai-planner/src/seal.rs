//! `src/plan/seal.ts` — `touchesFreeze`, `restsInFreeze` (a simplified, non-real-physics ballistic
//! forecast of "does this tee end up resting in freeze"), and `sealedIn` (bot-only helper: "is a
//! frozen tee inescapably sealed").

use crate::plan_world::{PlanCollision, PlanWorld};
use crate::tuning::{PHYSICAL_SIZE, TILE_TELECHECKIN, TILE_TELECHECKINEVIL, TILE_TELEIN, TILE_TELEINEVIL};
use crate::types::{PlayerInput, TeeState, empty_input};
use crate::vmath::Vec2;
use ddai_jsmath as js;

const HALF: f64 = PHYSICAL_SIZE / 2.0;
const SEAL_TICKS: i32 = 90;
const REST_TICKS: i32 = 60;

/// `touchesFreeze(collision, x, y)` (`seal.ts:13-17`).
pub fn touches_freeze(col: &impl PlanCollision, x: f64, y: f64) -> bool {
    if col.is_freeze(x, y) || col.is_death(x, y) {
        return true;
    }
    let d = PHYSICAL_SIZE / 3.0;
    col.is_death(x + d, y - d) || col.is_death(x + d, y + d) || col.is_death(x - d, y - d) || col.is_death(x - d, y + d)
}

/// `restsInFreeze(collision, pos, vel)` (`seal.ts:19-67`) — a simplified ballistic forecast (not
/// real DDNet physics, `docs/research/orig-plan.md` §1.12/§11 item 14): `1` if the tee ends up
/// touching freeze/death within [`REST_TICKS`], `0` otherwise.
pub fn rests_in_freeze(col: &impl PlanCollision, pos: Vec2, vel: Vec2) -> i32 {
    let mut x = pos.x;
    let mut y = pos.y;
    let mut vx = vel.x;
    let mut vy = vel.y;
    let tele = col.has_tele();

    // `through()` (`seal.ts:26-41`): follows one tele-in hop, or reports a checkpoint-tele /
    // no-tele-here result via the return convention documented at each call site below.
    let through = |x: &mut f64, y: &mut f64, vx: &mut f64, vy: &mut f64, col: &dyn PlanCollision| -> i32 {
        let (kind, number) = col.tele_at(*x, *y);
        if number == 0 {
            return 0;
        }
        if kind == i32::from(TILE_TELEIN) || kind == i32::from(TILE_TELEINEVIL) {
            let outs = col.tele_outs_for(number);
            let Some(first) = outs.first() else { return 0 };
            *x = first.x;
            *y = first.y;
            if kind == i32::from(TILE_TELEINEVIL) {
                *vx = 0.0;
                *vy = 0.0;
            }
            return 1;
        }
        if kind == i32::from(TILE_TELECHECKIN) || kind == i32::from(TILE_TELECHECKINEVIL) {
            -1
        } else {
            0
        }
    };

    for _ in 0..REST_TICKS {
        let grounded = col.is_solid(x - HALF + 1.0, y + HALF + 1.0) || col.is_solid(x + HALF - 1.0, y + HALF + 1.0);
        if grounded && vy >= 0.0 {
            break;
        }
        vy += *crate::tuning::GRAVITY;
        vx *= *crate::tuning::AIR_FRICTION;
        let nx = x + vx;
        if col.is_solid(nx + js::sign(vx) * HALF, y) {
            vx = 0.0;
        } else {
            x = nx;
        }
        let ny = y + vy;
        if vy > 0.0 && (col.is_solid(x - HALF + 1.0, ny + HALF) || col.is_solid(x + HALF - 1.0, ny + HALF)) {
            y = js::floor((ny + HALF) / 32.0) * 32.0 - HALF - 0.01;
            vy = 0.0;
            if !tele {
                break;
            }
            let moved = through(&mut x, &mut y, &mut vx, &mut vy, col);
            if moved < 0 {
                return 0;
            }
            if moved == 0 {
                break;
            }
            continue;
        }
        if vy < 0.0 && col.is_solid(x, ny - HALF) {
            vy = 0.0;
        } else {
            y = ny;
        }
        if col.is_death(x, y) {
            return 1;
        }
        if tele && through(&mut x, &mut y, &mut vx, &mut vy, col) < 0 {
            return 0;
        }
    }
    i32::from(touches_freeze(col, x, y))
}

/// `escapes(held)` (`seal.ts:69-81`): the fixed set of escape attempts `sealedIn` tries when the
/// tee has not yet frozen long enough to skip straight to `[held]`.
fn seal_escapes(held: PlayerInput) -> Vec<PlayerInput> {
    let mut out = vec![held];
    for ax in [0.0, -1.0, 1.0] {
        let mut e = empty_input();
        e.jump = 1;
        e.hook = 1;
        e.direction = ax as i32;
        e.target_x = ax * 200.0;
        e.target_y = -300.0;
        out.push(e);
    }
    out
}

/// `sealedIn(world, id, state, held)` (`seal.ts:83-107`): bot-only helper ("would this tee, placed
/// alone with this state, still end up frozen-in-freeze after [`SEAL_TICKS`] no matter what it
/// tries"). Removes every other tee from `world` (matching TS exactly — this mutates the world it
/// is given, by design).
pub fn sealed_in<W: PlanWorld>(world: &mut W, id: i32, state: &TeeState, held: PlayerInput) -> bool {
    for other in world.all_tees() {
        if other.id != id {
            world.remove_tee(other.id);
        }
    }
    if world.get_tee(id).is_none() {
        world.add_tee(id, state.pos);
    }
    world.apply_tee_state(id, state);
    let start = world.save_state();

    let tries: Vec<PlayerInput> = if state.frozen && state.freeze_ticks_left >= i64::from(SEAL_TICKS) {
        vec![held]
    } else {
        seal_escapes(held)
    };
    for input in tries {
        world.restore_state(&start);
        for t in 0..SEAL_TICKS {
            let now = if input.jump != 0 && t % 2 == 1 {
                PlayerInput { jump: 0, ..input }
            } else {
                input
            };
            world.set_input(id, now);
            world.step();
        }
        let Some(end) = world.get_tee(id) else { continue };
        if !end.alive {
            continue;
        }
        if !end.frozen || !touches_freeze(world.collision(), end.pos.x, end.pos.y) {
            world.restore_state(&start);
            return false;
        }
    }
    world.restore_state(&start);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan_world::LineHit;
    use crate::vmath::vec2;

    struct FloorCol {
        w: i32,
        h: i32,
    }
    impl PlanCollision for FloorCol {
        fn identity(&self) -> u64 {
            0
        }
        fn width(&self) -> i32 {
            self.w
        }
        fn height(&self) -> i32 {
            self.h
        }
        fn game_tile(&self, _tx: i32, _ty: i32) -> u8 {
            0
        }
        fn is_solid(&self, _x: f64, y: f64) -> bool {
            y >= 100.0
        }
        fn is_death(&self, _x: f64, _y: f64) -> bool {
            false
        }
        fn is_freeze(&self, _x: f64, _y: f64) -> bool {
            false
        }
        fn is_un_freeze(&self, _x: f64, _y: f64) -> bool {
            false
        }
        fn is_no_hook(&self, _x: f64, _y: f64) -> bool {
            false
        }
        fn test_box(&self, _pos: Vec2, _size: Vec2) -> bool {
            false
        }
        fn intersect_line(&self, _a: Vec2, b: Vec2) -> LineHit {
            LineHit {
                collision: 0,
                out_pos: b,
                out_before_pos: b,
            }
        }
        fn intersect_line_hook(&self, a: Vec2, b: Vec2) -> LineHit {
            self.intersect_line(a, b)
        }
        fn has_tele(&self) -> bool {
            false
        }
        fn tele_at(&self, _x: f64, _y: f64) -> (i32, i32) {
            (0, 0)
        }
        fn tele_outs_for(&self, _n: i32) -> Vec<Vec2> {
            Vec::new()
        }
    }

    #[test]
    fn touches_freeze_false_on_a_plain_floor() {
        let col = FloorCol { w: 10, h: 10 };
        assert!(!touches_freeze(&col, 50.0, 50.0));
    }

    #[test]
    fn rests_in_freeze_settles_on_floor_without_freeze_is_zero() {
        let col = FloorCol { w: 10, h: 10 };
        // Starts resting right on the floor with no velocity: `grounded && vy >= 0` breaks
        // immediately, then `touchesFreeze` at the (unmoved) position is false.
        let r = rests_in_freeze(&col, vec2(50.0, 100.0 - 14.0 - 0.5), vec2(0.0, 0.0));
        assert_eq!(r, 0);
    }
}
