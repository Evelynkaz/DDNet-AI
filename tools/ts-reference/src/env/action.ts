import { emptyInput } from "../core/types.ts";
import type { PlayerInput } from "../core/types.ts";
import { WEAPON_GUN, WEAPON_HAMMER } from "../core/types.ts";

export const ACTION_SIZE = 10;

export const MAX_AIM_TURN_RAD = Math.PI / 2;

export const FLICK_TURN_RAD = Math.PI;
export const FLICK_THRESHOLD_RAD = 2.4;

export const AIM_SMOOTHING = 0.75;
export const AIM_RADIUS = 300;

export const DIRECTION_SWITCH_MARGIN = 0.15;

export const AIR_JUMP_PRESS_MARGIN = 0.4;

export function decodeAction(raw: Float64Array, prev: PlayerInput, out?: PlayerInput, airborne = false): PlayerInput {
  const dst = out ?? emptyInput();

  let dirIdx = 0;
  if (raw[1] > raw[dirIdx]) dirIdx = 1;
  if (raw[2] > raw[dirIdx]) dirIdx = 2;
  const heldIdx = prev.direction + 1;
  if (dirIdx !== heldIdx && raw[dirIdx] - raw[heldIdx] < DIRECTION_SWITCH_MARGIN) dirIdx = heldIdx;
  dst.direction = dirIdx - 1;

  const jumpWanted = raw[3];
  if (prev.jump !== 0) {
    dst.jump = jumpWanted > 0 ? 1 : 0;
  } else {
    dst.jump = jumpWanted > (airborne ? AIR_JUMP_PRESS_MARGIN : 0) ? 1 : 0;
  }
  dst.hook = raw[4] > 0 ? 1 : 0;

  const held = (prev.fire & 1) !== 0;
  if (raw[5] > 0) dst.fire = held ? prev.fire + 2 : prev.fire + 1;
  else dst.fire = held ? prev.fire + 1 : prev.fire;

  const wantedWeaponId = raw[7] > raw[6] ? WEAPON_GUN : WEAPON_HAMMER;

  dst.wantedWeapon = wantedWeaponId + 1;
  dst.nextWeapon = 0;
  dst.prevWeapon = 0;
  dst.playerFlags = 0;

  const ax = Number.isFinite(raw[8]) ? raw[8] : 0;
  const ay = Number.isFinite(raw[9]) ? raw[9] : 0;
  const len = Math.sqrt(ax * ax + ay * ay);

  const wanted = len < 1e-6 ? 0 : Math.atan2(ay, ax);
  const prevLen = Math.sqrt(prev.targetX * prev.targetX + prev.targetY * prev.targetY);
  const current = prevLen < 1e-6 ? wanted : Math.atan2(prev.targetY, prev.targetX);

  let delta = wanted - current;
  while (delta > Math.PI) delta -= 2 * Math.PI;
  while (delta < -Math.PI) delta += 2 * Math.PI;

  const cap = Math.abs(delta) >= FLICK_THRESHOLD_RAD ? FLICK_TURN_RAD : MAX_AIM_TURN_RAD;
  delta *= AIM_SMOOTHING;
  if (delta > cap) delta = cap;
  else if (delta < -cap) delta = -cap;
  const angle = current + delta;

  let tx = Math.round(Math.cos(angle) * AIM_RADIUS);
  let ty = Math.round(Math.sin(angle) * AIM_RADIUS);
  if (tx === 0 && ty === 0) {
    tx = AIM_RADIUS;
    ty = 0;
  }
  dst.targetX = tx;
  dst.targetY = ty;

  return dst;
}
