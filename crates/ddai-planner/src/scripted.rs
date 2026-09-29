//! `scriptedAction` (`src/env/scripted.ts:12-57`) — the scripted opponent model, used both as the
//! bot's built-in scripted brain (`docs/research/orig-plan.md` §5) and, inside the planner, as the
//! `"react"`/`opponentMix` opponent model (`docs/research/orig-plan.md` §1.14).

use crate::plan_world::{PlanCollision, PlanWorld};
use crate::types::{PlayerInput, WEAPON_GUN, WEAPON_HAMMER, empty_input};
use ddai_jsmath as js;
use ddai_jsmath::Rng;

const HOOK_MIN_RANGE: f64 = 80.0;
const HOOK_MAX_RANGE: f64 = 380.0;
const HAMMER_RANGE: f64 = 60.0;
const AIM_NOISE: f64 = 0.2;
const PHYSICAL_SIZE: f64 = crate::tuning::PHYSICAL_SIZE;

/// `scriptedAction(view, selfId, enemyId, prev, rng)` (`scripted.ts:12-57`). Draws exactly one
/// `rng.next_float()` per call (the aim-noise jitter), whether or not the enemy is alive/present.
pub fn scripted_action<W: PlanWorld>(
    world: &W,
    self_id: i32,
    enemy_id: i32,
    prev: &PlayerInput,
    rng: &mut Rng,
) -> PlayerInput {
    let mut out = empty_input();
    let self_tee = world.get_tee(self_id);
    let enemy = world.get_tee(enemy_id);
    let (Some(self_tee), Some(enemy)) = (self_tee, enemy) else {
        out.fire = if (prev.fire & 1) != 0 { prev.fire + 1 } else { prev.fire };
        return out;
    };
    if !enemy.alive {
        out.fire = if (prev.fire & 1) != 0 { prev.fire + 1 } else { prev.fire };
        return out;
    }

    let dx = enemy.pos.x - self_tee.pos.x;
    let dy = enemy.pos.y - self_tee.pos.y;
    let dist = js::hypot2(dx, dy);

    let aim_dy = if dist < HAMMER_RANGE {
        dy
    } else {
        enemy.pos.y + PHYSICAL_SIZE / 2.0 - self_tee.pos.y
    };
    let noise = (rng.next_float() - 0.5) * AIM_NOISE;
    let aim_angle = js::atan2(aim_dy, dx) + noise;
    let mut tx = js::round(js::cos(aim_angle) * 300.0);
    let ty = js::round(js::sin(aim_angle) * 300.0);
    if tx == 0.0 && ty == 0.0 {
        tx = 300.0;
    }
    out.target_x = tx;
    out.target_y = ty;

    out.direction = if dx > 5.0 {
        1
    } else if dx < -5.0 {
        -1
    } else {
        0
    };

    let ahead_x = self_tee.pos.x + if dx >= 0.0 { 1.0 } else { -1.0 } * 24.0;
    let col = world.collision();
    let wall_ahead = col.is_solid(ahead_x, self_tee.pos.y) || col.is_solid(ahead_x, self_tee.pos.y - 16.0);
    out.jump = i32::from(wall_ahead || dy < -80.0);

    let los = col.intersect_line(self_tee.pos, enemy.pos).collision == 0;
    out.hook = i32::from(dist > HOOK_MIN_RANGE && dist < HOOK_MAX_RANGE && los);

    out.wanted_weapon = (if dist < HAMMER_RANGE { WEAPON_HAMMER } else { WEAPON_GUN }) + 1;
    out.fire = if (prev.fire & 1) != 0 {
        prev.fire + 2
    } else {
        prev.fire + 1
    };

    out.next_weapon = 0;
    out.prev_weapon = 0;
    out.player_flags = 0;
    out
}
