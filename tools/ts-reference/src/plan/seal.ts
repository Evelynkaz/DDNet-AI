import type { Collision } from "../core/collision.ts";
import type { SimWorld } from "../core/world.ts";
import type { PlayerInput, TeeState } from "../core/types.ts";
import { emptyInput } from "../core/types.ts";
import { PHYSICAL_SIZE, TILE_TELECHECKIN, TILE_TELECHECKINEVIL, TILE_TELEIN, TILE_TELEINEVIL, TUNING } from "../core/tuning.ts";
import type { Vec2 } from "../core/vmath.ts";

const HALF = PHYSICAL_SIZE / 2;

const SEAL_TICKS = 90;
const REST_TICKS = 60;

export function touchesFreeze(collision: Collision, x: number, y: number): boolean {
  if (collision.isFreeze(x, y) || collision.isDeath(x, y)) return true;
  const d = PHYSICAL_SIZE / 3;
  return collision.isDeath(x + d, y - d) || collision.isDeath(x + d, y + d) || collision.isDeath(x - d, y - d) || collision.isDeath(x - d, y + d);
}

export function restsInFreeze(collision: Collision, pos: Vec2, vel: Vec2): number {
  let x = pos.x;
  let y = pos.y;
  let vx = vel.x;
  let vy = vel.y;
  const tele = collision.hasTele();

  const through = (): number => {
    const tp = collision.teleAt(x, y);
    if (tp.number === 0) return 0;
    if (tp.type === TILE_TELEIN || tp.type === TILE_TELEINEVIL) {
      const outs = collision.teleOutsFor(tp.number);
      if (outs.length === 0) return 0;
      x = outs[0].x;
      y = outs[0].y;
      if (tp.type === TILE_TELEINEVIL) {
        vx = 0;
        vy = 0;
      }
      return 1;
    }
    return tp.type === TILE_TELECHECKIN || tp.type === TILE_TELECHECKINEVIL ? -1 : 0;
  };
  for (let t = 0; t < REST_TICKS; t++) {
    const grounded = collision.isSolid(x - HALF + 1, y + HALF + 1) || collision.isSolid(x + HALF - 1, y + HALF + 1);
    if (grounded && vy >= 0) break;
    vy += TUNING.gravity;
    vx *= TUNING.airFriction;
    const nx = x + vx;
    if (collision.isSolid(nx + Math.sign(vx) * HALF, y)) vx = 0;
    else x = nx;
    const ny = y + vy;
    if (vy > 0 && (collision.isSolid(x - HALF + 1, ny + HALF) || collision.isSolid(x + HALF - 1, ny + HALF))) {

      y = Math.floor((ny + HALF) / 32) * 32 - HALF - 0.01;
      vy = 0;
      if (!tele) break;
      const moved = through();
      if (moved < 0) return 0;
      if (moved === 0) break;
      continue;
    }
    if (vy < 0 && collision.isSolid(x, ny - HALF)) vy = 0;
    else y = ny;
    if (collision.isDeath(x, y)) return 1;
    if (tele && through() < 0) return 0;
  }
  return touchesFreeze(collision, x, y) ? 1 : 0;
}

function escapes(held: PlayerInput): PlayerInput[] {
  const out: PlayerInput[] = [held];
  for (const ax of [0, -1, 1]) {
    const e = emptyInput();
    e.jump = 1;
    e.hook = 1;
    e.direction = ax;
    e.targetX = ax * 200;
    e.targetY = -300;
    out.push(e);
  }
  return out;
}

export function sealedIn(world: SimWorld, id: number, state: TeeState, held: PlayerInput): boolean {
  for (const other of world.allTees()) if (other.id !== id) world.removeTee(other.id);
  if (world.getTee(id) === undefined) world.addTee(id, state.pos);
  world.applyTeeState(id, state);
  const start = world.saveState();
  try {

    const tries = state.frozen && state.freezeTicksLeft >= SEAL_TICKS ? [held] : escapes(held);
    for (const input of tries) {
      world.restoreState(start);
      for (let t = 0; t < SEAL_TICKS; t++) {

        const now = input.jump !== 0 && t % 2 === 1 ? { ...input, jump: 0 } : input;
        world.setInput(id, now);
        world.step();
      }
      const end = world.getTee(id);
      if (end === undefined || !end.alive) continue;
      if (!end.frozen || !touchesFreeze(world.collision, end.pos.x, end.pos.y)) return false;
    }
    return true;
  } finally {
    world.restoreState(start);
  }
}
