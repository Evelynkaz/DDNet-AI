import { CharacterCore } from "../core/characterCore.ts";
import type { CoreWorld } from "../core/characterCore.ts";
import type { Collision } from "../core/collision.ts";
import { SERVER_TICK_SPEED } from "../core/tuning.ts";
import { roundToInt } from "../core/vmath.ts";
import type { SnapCharacterCore } from "../bot/liveWorld.ts";

export const MAX_EVOLVE_TICKS = 3 * SERVER_TICK_SPEED;

const EMPTY_WORLD: CoreWorld = { allCores: () => [], coreById: () => undefined };

export function readCore(core: CharacterCore, c: SnapCharacterCore): void {
  core.pos = { x: c.x, y: c.y };
  core.vel = { x: c.vel_x / 256, y: c.vel_y / 256 };
  core.hookState = c.hook_state;
  core.hookTick = c.hook_tick;
  core.hookPos = { x: c.hook_x, y: c.hook_y };
  core.hookDir = { x: c.hook_dx / 256, y: c.hook_dy / 256 };
  core.setHookedPlayer(c.hooked_player);
  core.jumped = c.jumped;
  core.direction = c.direction;
  core.angle = c.angle;
}

export function writeCore(core: CharacterCore, tick: number): SnapCharacterCore {
  return {
    tick,
    x: roundToInt(core.pos.x),
    y: roundToInt(core.pos.y),
    vel_x: roundToInt(core.vel.x * 256),
    vel_y: roundToInt(core.vel.y * 256),
    angle: core.angle,
    direction: core.direction,
    jumped: core.jumped,
    hooked_player: core.hookedPlayer,
    hook_state: core.hookState,
    hook_tick: core.hookTick,
    hook_x: roundToInt(core.hookPos.x),
    hook_y: roundToInt(core.hookPos.y),
    hook_dx: roundToInt(core.hookDir.x * 256),
    hook_dy: roundToInt(core.hookDir.y * 256),
  };
}

export function needsEvolve(c: SnapCharacterCore, toTick: number): boolean {
  return c.tick !== 0 && c.tick < toTick && toTick - c.tick <= MAX_EVOLVE_TICKS;
}

export function evolveCore(c: SnapCharacterCore, weapon: number, toTick: number, collision: Collision): SnapCharacterCore {
  if (!needsEvolve(c, toTick)) return c;
  const core = new CharacterCore(-1, collision, EMPTY_WORLD);
  core.reset();
  readCore(core, c);
  core.activeWeapon = weapon;
  let tick = c.tick;
  while (tick < toTick) {
    tick++;
    core.tick(false);
    core.move();
    core.quantize();
  }
  return writeCore(core, toTick);
}
