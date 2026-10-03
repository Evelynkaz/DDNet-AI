import type { SimWorld } from "../core/world.ts";
import type { PlayerInput } from "../core/types.ts";
import { emptyInput } from "../core/types.ts";
import { PHYSICAL_SIZE } from "../core/tuning.ts";

const HALF = PHYSICAL_SIZE / 2;

const ESCAPE_TICKS = 36;

function escapes(vx: number, aimX: number, aimY: number): PlayerInput[] {
  const brakeDir = Math.abs(vx) < 0.5 ? 0 : vx > 0 ? -1 : 1;
  const mk = (dir: number, jump: number): PlayerInput => {
    const e = emptyInput();
    e.direction = dir;
    e.jump = jump;
    e.targetX = aimX;
    e.targetY = aimY;
    return e;
  };

  const out = brakeDir === 0 ? [mk(0, 0), mk(0, 1)] : [mk(0, 0), mk(0, 1), mk(brakeDir, 0), mk(brakeDir, 1)];

  for (const ax of [0, brakeDir]) {
    const h = mk(brakeDir, 1);
    h.hook = 1;
    h.targetX = ax * 150;
    h.targetY = -300;
    out.push(h);
    if (brakeDir === 0) break;
  }
  return out;
}

export function escapeExists(world: SimWorld, selfId: number, input: PlayerInput, holdTicks: number, others: ReadonlyMap<number, PlayerInput> = new Map()): boolean {
  const start = world.saveState();
  try {
    const setOthers = (): void => {
      for (const [id, inp] of others) world.setInput(id, inp);
    };
    for (let t = 0; t < holdTicks; t++) {
      world.setInput(selfId, input);
      setOthers();
      world.step();
      const me = world.getTee(selfId);
      if (me === undefined || !me.alive || me.frozen) return false;
    }
    const afterHold = world.saveState();
    const me = world.getTee(selfId);
    if (me === undefined) return false;

    const pressTick = input.jump ? 1 : 0;
    for (const esc of escapes(me.vel.x, input.targetX, input.targetY)) {
      world.restoreState(afterHold);
      let ok = true;
      for (let t = 0; t < ESCAPE_TICKS; t++) {

        world.setInput(selfId, t === pressTick || esc.jump === 0 ? esc : { ...esc, jump: 0 });
        setOthers();
        world.step();
        const now = world.getTee(selfId);
        if (now === undefined || !now.alive || now.frozen) {
          ok = false;
          break;
        }
      }
      if (ok && settlesSafe(world, selfId, esc, setOthers)) return true;
    }
    return false;
  } finally {
    world.restoreState(start);
  }
}

const SETTLE_TICKS = 90;

function settlesSafe(world: SimWorld, selfId: number, esc: PlayerInput, setOthers: () => void): boolean {
  const coast = { ...emptyInput(), hook: esc.hook, targetX: esc.targetX, targetY: esc.targetY };
  for (let t = 0; t < SETTLE_TICKS; t++) {
    const me = world.getTee(selfId);
    if (me === undefined || !me.alive || me.frozen) return false;
    if (standing(world, me.pos.x, me.pos.y, me.vel.y)) return true;
    world.setInput(selfId, coast);
    setOthers();
    world.step();
  }
  const me = world.getTee(selfId);
  return me !== undefined && me.alive && !me.frozen;
}

function standing(world: SimWorld, x: number, y: number, vy: number): boolean {
  const c = world.collision;
  return Math.abs(vy) < 0.5 && (c.isSolid(x - HALF + 1, y + HALF + 2) || c.isSolid(x + HALF - 1, y + HALF + 2));
}

export function saferInput(world: SimWorld, selfId: number, input: PlayerInput, holdTicks: number, others: ReadonlyMap<number, PlayerInput> = new Map(), sentAim?: { targetX: number; targetY: number }): PlayerInput | null {
  const me = world.getTee(selfId);
  if (me === undefined) return null;

  const base = sentAim !== undefined && (sentAim.targetX !== 0 || sentAim.targetY !== 0) ? sentAim : input;
  const from = Math.atan2(base.targetY, base.targetX);
  for (const alt of escapes(me.vel.x, input.targetX, input.targetY)) {

    let d = Math.atan2(alt.targetY, alt.targetX) - from;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    const a = from + Math.max(-MAX_TURN_RAD, Math.min(MAX_TURN_RAD, d));
    const cand = { ...input, direction: alt.direction, jump: alt.jump, hook: alt.hook, targetX: Math.round(Math.cos(a) * 300), targetY: Math.round(Math.sin(a) * 300) };
    if (escapeExists(world, selfId, cand, holdTicks, others)) return cand;
  }
  return null;
}

const MAX_TURN_RAD = 1.5;
