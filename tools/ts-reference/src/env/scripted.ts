import { PHYSICAL_SIZE } from "../core/tuning.ts";
import { emptyInput } from "../core/types.ts";
import type { PlayerInput, WorldView } from "../core/types.ts";
import { WEAPON_GUN, WEAPON_HAMMER } from "../core/types.ts";
import type { Rng } from "../nn/rng.ts";

const HOOK_MIN_RANGE = 80;
const HOOK_MAX_RANGE = 380;
const HAMMER_RANGE = 60;
const AIM_NOISE = 0.2;

export function scriptedAction(
  view: WorldView,
  selfId: number,
  enemyId: number,
  prev: PlayerInput,
  rng: Rng,
): PlayerInput {
  const out = emptyInput();
  const self = view.getTee(selfId);
  const enemy = view.getTee(enemyId);
  if (!self || !enemy || !enemy.alive) {
    out.fire = (prev.fire & 1) !== 0 ? prev.fire + 1 : prev.fire;
    return out;
  }

  const dx = enemy.pos.x - self.pos.x;
  const dy = enemy.pos.y - self.pos.y;
  const dist = Math.hypot(dx, dy);

  const aimDy = dist < HAMMER_RANGE ? dy : enemy.pos.y + PHYSICAL_SIZE / 2 - self.pos.y;
  const noise = (rng.nextFloat() - 0.5) * AIM_NOISE;
  const aimAngle = Math.atan2(aimDy, dx) + noise;
  let tx = Math.round(Math.cos(aimAngle) * 300);
  let ty = Math.round(Math.sin(aimAngle) * 300);
  if (tx === 0 && ty === 0) tx = 300;
  out.targetX = tx;
  out.targetY = ty;

  out.direction = dx > 5 ? 1 : dx < -5 ? -1 : 0;

  const aheadX = self.pos.x + (dx >= 0 ? 1 : -1) * 24;
  const wallAhead =
    view.collision.isSolid(aheadX, self.pos.y) || view.collision.isSolid(aheadX, self.pos.y - 16);
  out.jump = wallAhead || dy < -80 ? 1 : 0;

  const los = view.collision.intersectLine(self.pos, enemy.pos).collision === 0;
  out.hook = dist > HOOK_MIN_RANGE && dist < HOOK_MAX_RANGE && los ? 1 : 0;

  out.wantedWeapon = (dist < HAMMER_RANGE ? WEAPON_HAMMER : WEAPON_GUN) + 1;
  out.fire = (prev.fire & 1) !== 0 ? prev.fire + 2 : prev.fire + 1;

  out.nextWeapon = 0;
  out.prevWeapon = 0;
  out.playerFlags = 0;
  return out;
}
