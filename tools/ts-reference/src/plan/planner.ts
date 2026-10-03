import { SimWorld } from "../core/world.ts";
import type { SimState } from "../core/world.ts";
import { blankTeeState, emptyInput, WEAPON_HAMMER } from "../core/types.ts";
import type { PlayerInput, TeeState, WorldEvent } from "../core/types.ts";
import { Collision } from "../core/collision.ts";
import { TILE_DEATH, TILE_FREEZE, TILE_NOHOOK, TILE_SOLID, TILE_UNFREEZE } from "../core/tuning.ts";
import { vdistance } from "../core/vmath.ts";
import type { Vec2 } from "../core/vmath.ts";
import { CFLAG_NOHOOK, PHYSICAL_SIZE, TUNING } from "../core/tuning.ts";
import { restsInFreeze } from "./seal.ts";
import { escapeExists, saferInput } from "./shield.ts";
import { deadZoneOf } from "./route.ts";
import type { FreezeMemory } from "./memory.ts";
import { Rng } from "../nn/rng.ts";
import { scriptedAction } from "../env/scripted.ts";
import { decodeAction, ACTION_SIZE } from "../env/action.ts";
import { encodeObs, OBS_SIZE } from "../env/obs.ts";
import { encodeHumanTarget } from "../train/humanImitate.ts";
import type { Mlp } from "../nn/mlp.ts";
import { RecurrentPolicy } from "../nn/gru.ts";
import { OpponentProfile } from "../bot/opponentProfile.ts";
import { HOOK_FLYING, HOOK_GRABBED, HOOK_IDLE, HOOK_RETRACT_START } from "../core/characterCore.ts";
import { frozenThrowLines, frozenThrowWorthTrying, throwLines, throwWorthTrying } from "./throwLines.ts";

const TILE_PX = 32;
const HOOK_LENGTH = TUNING.hookLength;

const AIR_JUMP_MIN_GAP_TICKS = 11;

const CADENCE_GAPS = 4;
const CADENCE_MAX_GAP = 50;
const CADENCE_MAX_TICKS = 8;
const SHIELD_MAX_HOLD = 16;

export type PlannerConfig = {
  steps?: number;
  planStep?: number;

  restAim?: boolean;
  selfFreezeBias?: number;

  travelWeight?: number;

  seek?: boolean;

  memoryWeight?: number;

  memoryTrust?: number;

  pathToTarget?: boolean;

  settledFreezeTicks?: number;
  blockHoldScore?: number;

  deadZoneCost?: number;

  enemyDeadZoneBonus?: number;
  frontSteps?: number;
  frontStep?: number;

  releaseDeadHook?: boolean;
  population?: number;
  elite?: number;
  iterations?: number;
  seed?: number;

  wastedHammer?: number;
  wastedHook?: number;

  hammerRangePx?: number;

  gateHook?: boolean;

  gateHammer?: boolean;

  airJumpCost?: number;

  jumplessHazardCost?: number;

  selfHazardCost?: number;

  flipCost?: number;

  wallPushCost?: number;

  commitDecisions?: number;

  opponentModel?: "hold" | "react" | "policy" | "learned";

  enemyHazardWeight?: number;
  frozenWeight?: number;

  hookHoldWeight?: number;
  distanceWeight?: number;

  standoffPx?: number;

  selfHazardThreshold?: number;

  freezeTailWeight?: number;

  sealTicks?: number;

  frozenTargetSteps?: number;

  frozenThrow?: number;

  bandCost?: number;

  shield?: boolean;

  policySeeds?: number;

  policySeedJitter?: number;

  policySeedSteps?: number;

  noThaw?: boolean;

  noThawRope?: boolean;

  hookDragWeight?: number;

  enemyHazardFromStart?: boolean;

  dragThreat?: number;
  launchThreat?: number;

  landingCost?: number;

  planMargin?: number;

  opponentMix?: boolean;

  budgetMs?: number;

  hardMs?: number;

  shieldCadence?: boolean;

  explain?: boolean;

  liveTransfer?: "full" | "legacy";

  planOthers?: number;

  targetHold?: number;

  escapeBias?: number;
  escapeMargin?: number;

  warmShiftElapsed?: boolean;

  hookReleaseCost?: number;

  valueWeight?: number;

  trackAim?: boolean;

  openingBook?: "classic" | "wide" | "movement" | "all";

  launchExposure?: number;

  launchExactReach?: number;

  launchExactRiseVy?: number;

  launchExactWeight?: number;

  opponentReadWeight?: number;

  routeDistance?: boolean;

  dragExposure?: number;

  thirdTeeExposure?: number;

  freezeThrow?: number;

  jitterCost?: number;
  flipHoldTicks?: number;

  flipMargin?: number;

  edgeHold?: boolean;

  hookSeeds?: boolean;

  hookPolish?: boolean;
};

export function buildStepTicks(steps: number, planStep: number, frontSteps: number, frontStep: number): number[] {
  const total = steps * planStep;
  const front = Math.max(0, Math.min(Math.trunc(frontSteps), steps - 1));
  if (front === 0 || frontStep <= 0 || frontStep >= planStep) return new Array(steps).fill(planStep);
  const fine = Math.max(1, Math.trunc(frontStep));
  const rest = steps - front;
  const left = total - front * fine;

  if (left < rest) return new Array(steps).fill(planStep);
  const base = Math.floor(left / rest);
  const extra = left - base * rest;
  const out: number[] = [];
  for (let i = 0; i < front; i++) out.push(fine);
  for (let i = 0; i < rest; i++) out.push(i >= rest - extra ? base + 1 : base);
  return out;
}

function copyInput(into: PlayerInput, from: PlayerInput): void {
  into.direction = from.direction;
  into.targetX = from.targetX;
  into.targetY = from.targetY;
  into.jump = from.jump;
  into.fire = from.fire;
  into.hook = from.hook;
  into.playerFlags = from.playerFlags;
  into.wantedWeapon = from.wantedWeapon;
  into.nextWeapon = from.nextWeapon;
  into.prevWeapon = from.prevWeapon;
}

export const PLANNER_DEFAULTS = {

  steps: 9,
  planStep: 3,
  restAim: false,
  selfFreezeBias: 1.5,
  travelWeight: 0.35,
  seek: true,
  memoryWeight: 0,
  memoryTrust: 0,

  pathToTarget: true,

  settledFreezeTicks: 0,
  blockHoldScore: 0,
  deadZoneCost: 0,
  enemyDeadZoneBonus: 0,
  frontSteps: 0,
  frontStep: 2,
  releaseDeadHook: false,
  population: 20,
  elite: 6,
  iterations: 2,
  seed: 1,
  wastedHammer: 0.03,
  wastedHook: 0.05,

  hammerRangePx: 80,
  gateHook: true,
  gateHammer: true,
  airJumpCost: 0,
  jumplessHazardCost: 0.15,

  selfHazardCost: 0.6,
  flipCost: 0.4,
  wallPushCost: 0.05,

  commitDecisions: 1,
  opponentModel: "hold" as "hold" | "react" | "policy" | "learned",
  enemyHazardWeight: 2.0,
  frozenWeight: 0.5,
  hookHoldWeight: 0.08,
  distanceWeight: 0.06,
  standoffPx: 250,
  selfHazardThreshold: 0.55,
  trackAim: true,
  openingBook: "classic" as "classic" | "wide" | "movement" | "all",
  launchExposure: 1.0,
  launchExactReach: 70,
  launchExactRiseVy: 4,
  launchExactWeight: 2,
  opponentReadWeight: 0,
  routeDistance: false,
  dragExposure: 0.5,
  thirdTeeExposure: 0,
  freezeThrow: 0,
  jitterCost: 0,
  flipHoldTicks: 0,

  flipMargin: 0.6,
  edgeHold: true,
  freezeTailWeight: 0.5,
  sealTicks: 150,
  frozenTargetSteps: 0,
  frozenThrow: 0,
  bandCost: 0,
  shield: true,
  policySeeds: 0,
  policySeedJitter: 0.25,
  policySeedSteps: 0,
  noThaw: true,
  noThawRope: false,
  hookDragWeight: 0,
  enemyHazardFromStart: true,
  dragThreat: 0,
  launchThreat: 0,
  landingCost: 0,
  planMargin: 0,
  opponentMix: false,
  budgetMs: 0,
  hardMs: 0,
  shieldCadence: false,
  explain: false,
  liveTransfer: "full" as "full" | "legacy",
  planOthers: 0,
  targetHold: 400,
  escapeBias: 0,
  escapeMargin: 1.5,
  warmShiftElapsed: false,
  hookReleaseCost: 0,
  valueWeight: 0,
  hookSeeds: true,
  hookPolish: true,
};

export type DecisionInfo = {
  searched: boolean;
  candidates: number;
  outOfTime: boolean;
  ms: number;
  selfOut: number;
  enemyOut: number;

  hookAt: number;

  gated: boolean;

  shielded?: boolean;

  edgeHeld?: boolean;
};

const THROW_LANDED_TICKS = 5;

const FROZEN_PLAN_MIN_TICKS = 30;

export type PlanStep = { dir: number; jump: number; hook: number; fire: number; aim: number };
type StepDist = { pLeft: number; pRight: number; pJump: number; pHook: number; pFire: number; aim: number; aimSpread: number };

export type HazardField = { width: number; height: number; dist: Int32Array };

const hazardFields = new WeakMap<Collision, HazardField>();
const unfreezeFields = new WeakMap<Collision, HazardField>();

export function unfreezeField(collision: Collision): HazardField {
  const cached = unfreezeFields.get(collision);
  if (cached !== undefined) return cached;
  const field = bfsField(collision, (t) => t === TILE_UNFREEZE);
  unfreezeFields.set(collision, field);
  return field;
}

function bfsField(collision: Collision, isSource: (tile: number) => boolean): HazardField {
  const { width, height, tiles } = collision;
  const dist = new Int32Array(width * height).fill(0x3fffffff);
  const queue: number[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (isSource(tiles[y * width + x])) {
        dist[y * width + x] = 0;
        queue.push(y * width + x);
      }
    }
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const idx = queue[qi];
    const x = idx % width;
    const y = (idx - x) / width;
    const d = dist[idx];
    for (let k = 0; k < 4; k++) {
      const nx = x + (k === 0 ? 1 : k === 1 ? -1 : 0);
      const ny = y + (k === 2 ? 1 : k === 3 ? -1 : 0);
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
      const ni = ny * width + nx;
      const nt = tiles[ni];
      if (nt === TILE_SOLID || nt === TILE_NOHOOK) continue;
      if (dist[ni] > d + 1) {
        dist[ni] = d + 1;
        queue.push(ni);
      }
    }
  }
  return { width, height, dist };
}

const travelScratch = new WeakMap<Collision, { dist: Int32Array; queue: Int32Array }>();

export function travelField(collision: Collision, fromX: number, fromY: number): HazardField {
  const { width, height, tiles } = collision;
  let scratch = travelScratch.get(collision);
  if (scratch === undefined) {
    scratch = { dist: new Int32Array(width * height), queue: new Int32Array(width * height) };
    travelScratch.set(collision, scratch);
  }
  const { dist, queue } = scratch;
  dist.fill(0x3fffffff);

  const sx = Math.min(width - 1, Math.max(0, Math.trunc(fromX / TILE_PX)));
  const sy = Math.min(height - 1, Math.max(0, Math.trunc(fromY / TILE_PX)));
  let head = 0;
  let tail = 0;
  const start = sy * width + sx;
  dist[start] = 0;
  queue[tail++] = start;
  while (head < tail) {
    const idx = queue[head++];
    const x = idx % width;
    const y = (idx - x) / width;
    const d = dist[idx];
    for (let k = 0; k < 4; k++) {
      const nx = x + (k === 0 ? 1 : k === 1 ? -1 : 0);
      const ny = y + (k === 2 ? 1 : k === 3 ? -1 : 0);
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
      const ni = ny * width + nx;
      const nt = tiles[ni];
      if (nt === TILE_SOLID || nt === TILE_NOHOOK || nt === TILE_FREEZE || nt === TILE_DEATH) continue;
      if (dist[ni] > d + 1) {
        dist[ni] = d + 1;
        queue[tail++] = ni;
      }
    }
  }
  return { width, height, dist };
}

export function travelDistance(field: HazardField, x: number, y: number): number {
  const tx = Math.min(field.width - 1, Math.max(0, Math.trunc(x / TILE_PX)));
  const ty = Math.min(field.height - 1, Math.max(0, Math.trunc(y / TILE_PX)));
  const d = field.dist[ty * field.width + tx];
  return d >= 0x3fffffff ? UNREACHABLE_TILES : d;
}

const UNREACHABLE_TILES = 200;

export function travelDistanceSmooth(field: HazardField, x: number, y: number): number {
  const fx = x / TILE_PX - 0.5;
  const fy = y / TILE_PX - 0.5;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;
  const at = (cx: number, cy: number): number => {
    const ix = Math.min(field.width - 1, Math.max(0, cx));
    const iy = Math.min(field.height - 1, Math.max(0, cy));
    const d = field.dist[iy * field.width + ix];
    return d >= 0x3fffffff ? UNREACHABLE_TILES : d;
  };
  const top = at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx;
  const bottom = at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx;
  return top * (1 - ty) + bottom * ty;
}

export function hazardField(collision: Collision): HazardField {
  const cached = hazardFields.get(collision);
  if (cached !== undefined) return cached;
  const { width, height, tiles } = collision;
  const dist = new Int32Array(width * height).fill(0x3fffffff);
  const queue: number[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const t = tiles[y * width + x];
      if (t === TILE_FREEZE || t === TILE_DEATH) {
        dist[y * width + x] = 0;
        queue.push(y * width + x);
      }
    }
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const idx = queue[qi];
    const x = idx % width;
    const y = (idx - x) / width;
    const d = dist[idx];
    for (let k = 0; k < 4; k++) {
      const nx = x + (k === 0 ? 1 : k === 1 ? -1 : 0);
      const ny = y + (k === 2 ? 1 : k === 3 ? -1 : 0);
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
      const ni = ny * width + nx;
      const nt = tiles[ni];
      if (nt === TILE_SOLID || nt === TILE_NOHOOK) continue;
      if (dist[ni] > d + 1) {
        dist[ni] = d + 1;
        queue.push(ni);
      }
    }
  }
  const field = { width, height, dist };
  hazardFields.set(collision, field);
  return field;
}

function wrapAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

const HAZARD_HORIZON_TILES = 20;

export function hazardNearness(field: HazardField, x: number, y: number): number {
  const tx = Math.floor(x / TILE_PX);
  const ty = Math.floor(y / TILE_PX);
  if (tx < 0 || ty < 0 || tx >= field.width || ty >= field.height) return 0;
  const d = field.dist[ty * field.width + tx];
  if (d >= HAZARD_HORIZON_TILES) return 0;
  return (HAZARD_HORIZON_TILES - d) / HAZARD_HORIZON_TILES;
}

const scoreMeBuf = blankTeeState();
const scoreEnBuf = blankTeeState();

const inDead = (dead: { width: number; cells: Uint8Array } | null, x: number, y: number): boolean => {
  if (dead === null) return false;
  const i = Math.trunc(y / TILE_PX) * dead.width + Math.trunc(x / TILE_PX);
  return i >= 0 && i < dead.cells.length && dead.cells[i] === 1;
};

function scoreTick(world: SimWorld, selfId: number, enemyId: number, events: WorldEvent[], field: HazardField, unfreeze: HazardField, cfg: Required<PlannerConfig>, drag: { prevEnemyNear: number; startEnemyNear: number; startedInDead: boolean }, travel: HazardField | null, goal: Vec2 | null, dead: { width: number; cells: Uint8Array } | null, memory: FreezeMemory | null, thirds: readonly Vec2[], band: { x0: number; y0: number; x1: number; y1: number } | null = null): number {
  const me = world.readTee(selfId, scoreMeBuf);
  const en = world.readTee(enemyId, scoreEnBuf);
  if (me === undefined || en === undefined) return -1000;
  let s = 0;
  if (!en.alive) s += 15;
  if (!me.alive) s -= 15;
  if (en.frozen) s += cfg.frozenWeight;
  if (me.frozen) s -= cfg.frozenWeight * cfg.selfFreezeBias;
  if (band !== null && cfg.bandCost > 0 && !me.frozen && me.pos.x >= band.x0 && me.pos.x <= band.x1 && me.pos.y >= band.y0 && me.pos.y <= band.y1) s -= cfg.bandCost;
  if (me.hookedPlayer === enemyId) s += cfg.hookHoldWeight;
  if (en.hookedPlayer === selfId) s -= cfg.hookHoldWeight * 0.75;
  for (const e of events) {
    if (e.kind === "hammerFire" && e.from === selfId && e.hits === 0) s -= cfg.wastedHammer;
    if (e.kind === "hammerHit" && e.from === enemyId && e.to === selfId) s -= 0.3;
    if (e.kind === "death" && e.id === enemyId) s += 15;
    if (e.kind === "death" && e.id === selfId) s -= 15;
  }

  if (me.frozen) {

    s += 0.08 * hazardNearness(unfreeze, me.pos.x, me.pos.y);
  }
  if (en.frozen) {

    s -= 0.06 * hazardNearness(unfreeze, en.pos.x, en.pos.y);
  }

  if (me.direction !== 0 && Math.abs(me.vel.x) < 0.2 && !me.frozen) s -= cfg.wallPushCost;
  if (cfg.jumplessHazardCost > 0 && !me.frozen && me.jumpsLeft === 0) s -= cfg.jumplessHazardCost * hazardNearness(field, me.pos.x, me.pos.y);
  const enNear = hazardNearness(field, en.pos.x, en.pos.y);

  const enFloor = cfg.enemyHazardFromStart ? Math.max(0.3, drag.startEnemyNear) : 0.3;
  if (enNear > enFloor) s += cfg.enemyHazardWeight * (enNear - enFloor);

  if (cfg.hookDragWeight > 0 && me.hookedPlayer === enemyId && !en.frozen && enNear > drag.prevEnemyNear) {
    s += cfg.hookDragWeight * (enNear - drag.prevEnemyNear);
  }
  drag.prevEnemyNear = enNear;

  const meNear = hazardNearness(field, me.pos.x, me.pos.y);
  if (meNear > cfg.selfHazardThreshold) {

    const trusted = memory !== null && cfg.memoryTrust > 0 ? cfg.memoryTrust * memory.safety(me.pos.x, me.pos.y) : 0;
    s -= cfg.selfHazardCost * (meNear - cfg.selfHazardThreshold) * (1 - trusted);
  }
  const separation = vdistance(me.pos, en.pos);
  if (cfg.launchExposure > 0 && !me.frozen && !en.frozen && separation < LAUNCH_REACH_PX) {

    const exact = cfg.launchExactReach > 0 && separation < cfg.launchExactReach && (me.pos.y < en.pos.y || (cfg.launchExactRiseVy > 0 && me.vel.y < -cfg.launchExactRiseVy));
    if (exact) s -= (cfg.launchExactWeight > 0 ? cfg.launchExactWeight : cfg.launchExposure) * launchFlightLandsInHazard(world.collision, me.pos, en.pos, separation, me.vel);
    else s -= cfg.launchExposure * launchLandsInHazard(world.collision, me.pos, en.pos, separation);
  }
  if (cfg.dragExposure > 0 && !me.frozen && !en.frozen && separation < HOOK_LENGTH) {
    s -= cfg.dragExposure * dragCrossesHazard(world.collision, me.pos, en.pos, separation);
  }

  if (cfg.dragThreat > 0 && !me.frozen && !en.frozen && separation < HOOK_LENGTH) {
    s += cfg.dragThreat * dragCrossesHazard(world.collision, en.pos, me.pos, separation);
  }
  if (cfg.launchThreat > 0 && !me.frozen && !en.frozen && separation < LAUNCH_REACH_PX) {
    s += cfg.launchThreat * launchLandsInHazard(world.collision, en.pos, me.pos, separation);
  }

  if (cfg.thirdTeeExposure > 0 && !me.frozen && thirds.length > 0) {
    for (const t of thirds) {
      const d = vdistance(me.pos, t);
      if (d < 1 || d >= HOOK_LENGTH) continue;
      s -= cfg.thirdTeeExposure * dragCrossesHazard(world.collision, me.pos, t, d);
    }
  }

  s -= cfg.distanceWeight * (Math.max(0, separation - cfg.standoffPx) / TILE_PX);

  if (goal !== null) s -= cfg.travelWeight * (vdistance(me.pos, goal) / TILE_PX);

  if (memory !== null && cfg.memoryWeight > 0 && me.alive && !me.frozen) {
    s -= cfg.memoryWeight * memory.risk(me.pos.x, me.pos.y);
  }

  if (dead !== null && !drag.startedInDead && (cfg.deadZoneCost > 0 || cfg.enemyDeadZoneBonus > 0)) {
    const meDead = me.alive && inDead(dead, me.pos.x, me.pos.y);
    const enDead = en.alive && inDead(dead, en.pos.x, en.pos.y);
    if (cfg.deadZoneCost > 0 && meDead && !enDead) s -= cfg.deadZoneCost;
    if (cfg.enemyDeadZoneBonus > 0 && enDead && !meDead) s += cfg.enemyDeadZoneBonus;
  }
  return s;
}

const LAUNCH_REACH_PX = 96;
const NO_VEL: Vec2 = { x: 0, y: 0 };

function dragCrossesHazard(collision: Collision, at: Vec2, from: Vec2, separation: number): number {
  const dx = (from.x - at.x) / separation;
  const dy = (from.y - at.y) / separation;
  for (const px of [TILE_PX, 2 * TILE_PX, 3 * TILE_PX]) {
    if (px >= separation) return 0;
    const x = at.x + dx * px;
    const y = at.y + dy * px;
    if (collision.isSolid(x, y)) return 0;
    if (collision.isFreeze(x, y) || collision.isDeath(x, y)) return 1;
  }
  return 0;
}

function launchLandsInHazard(collision: Collision, at: Vec2, from: Vec2, separation: number): number {
  const hx = separation > 0 ? (at.x - from.x) / separation : 0;
  const hy = separation > 0 ? (at.y - from.y) / separation : -1;

  const bx = hx;
  const by = hy - 1.1;
  const bl = Math.hypot(bx, by) || 1;
  const dx = bx / bl;
  const dy = by / bl;
  for (const px of [3 * TILE_PX, 5 * TILE_PX, 7 * TILE_PX]) {
    const x = at.x + dx * px;
    const y = at.y + dy * px;
    if (collision.isSolid(x, y)) return 0;
    if (collision.isFreeze(x, y) || collision.isDeath(x, y)) return 1;
  }
  return 0;
}

export const ROPE_CATCH_PX = PHYSICAL_SIZE + 6;
export function ropeCatchAlong(from: Vec2, dir: Vec2, at: Vec2): number {
  const rx = at.x - from.x;
  const ry = at.y - from.y;
  const along = rx * dir.x + ry * dir.y;
  if (along < 0 || along > HOOK_LENGTH) return Infinity;
  return Math.abs(rx * dir.y - ry * dir.x) <= ROPE_CATCH_PX ? along : Infinity;
}

export function launchFlightLandsInHazard(collision: Collision, at: Vec2, from: Vec2, separation: number, vel: Vec2 = NO_VEL): number {
  const hx = separation > 0 ? (at.x - from.x) / separation : 0;
  const hy = separation > 0 ? (at.y - from.y) / separation : -1;
  const bx = hx;
  const by = hy - 1.1;
  const bl = Math.hypot(bx, by) || 1;
  const k = TUNING.hammerStrength;
  let vx = vel.x + (k * 10 * bx) / bl;
  let vy = vel.y + k * (-1 + (10 * by) / bl);
  let x = at.x;
  let y = at.y;

  const half = PHYSICAL_SIZE / 2;
  let grounded = collision.isSolid(x + half, y + half + 5) || collision.isSolid(x - half, y + half + 5);
  for (let t = 0; t < LAUNCH_FLIGHT_TICKS; t++) {
    vy += TUNING.gravity;
    vx *= grounded ? TUNING.groundFriction : TUNING.airFriction;
    grounded = false;

    const speed = Math.hypot(vx, vy) * 50;
    const ramp = speed < TUNING.velrampStart ? 1 : 1 / Math.pow(TUNING.velrampCurvature, (speed - TUNING.velrampStart) / TUNING.velrampRange);
    const n = Math.max(1, Math.ceil(Math.max(Math.abs(vx), Math.abs(vy)) / LAUNCH_PROBE_STEP_PX));
    for (let i = 0; i < n; i++) {
      const sx = (vx * ramp) / n;
      if (sx !== 0) {
        const f = freeFraction(collision, x, y, sx, 0);
        x += sx * f;
        if (f < 1) vx = 0;
      }
      let landed = false;
      const sy = vy / n;
      if (sy !== 0) {
        const f = freeFraction(collision, x, y, 0, sy);
        y += sy * f;
        if (f < 1) {
          landed = vy > 0;
          vy = 0;
        }
      }

      if (collision.isFreeze(x, y) || collision.isDeath(x, y)) return 1;
      if (landed) return 0;
    }
  }
  return 0;
}

function freeFraction(collision: Collision, x: number, y: number, dx: number, dy: number): number {
  launchProbe.x = x + dx;
  launchProbe.y = y + dy;
  if (!collision.testBox(launchProbe, TEE_BOX)) return 1;
  let lo = 0;
  let hi = 1;
  for (let k = 0; k < 5; k++) {
    const mid = (lo + hi) / 2;
    launchProbe.x = x + dx * mid;
    launchProbe.y = y + dy * mid;
    if (collision.testBox(launchProbe, TEE_BOX)) hi = mid;
    else lo = mid;
  }
  return lo;
}

const LAUNCH_FLIGHT_TICKS = 50;

const LAUNCH_PROBE_STEP_PX = TILE_PX / 2;
const TEE_BOX: Vec2 = { x: PHYSICAL_SIZE, y: PHYSICAL_SIZE };
const launchProbe: Vec2 = { x: 0, y: 0 };

const FREEZE_CLOCK_TICKS = 3 * 50;

const THAW_ESCAPE_TICKS = 40;

const THAW_ESCAPES: PlayerInput[][] = (() => {
  const line = (fn: (t: number, e: PlayerInput) => void): PlayerInput[] =>
    Array.from({ length: THAW_ESCAPE_TICKS }, (_, t) => {
      const e = emptyInput();
      e.targetX = 0;
      e.targetY = -300;
      fn(t, e);
      return e;
    });
  const out: PlayerInput[][] = [line(() => {})];
  for (const d of [-1, 1]) out.push(line((_, e) => (e.direction = d)));
  for (const d of [0, -1, 1]) {
    out.push(
      line((t, e) => {
        e.direction = d;
        e.jump = t === 0 || t === 6 ? 1 : 0;
      }),
    );
  }
  for (const ax of [0, -1, 1]) {
    out.push(
      line((t, e) => {
        e.direction = ax;
        e.jump = t % 2 === 0 ? 1 : 0;
        e.hook = 1;
        e.targetX = ax * 200;
        e.targetY = -300;
      }),
    );
  }
  return out;
})();

function ropeEscapes(dx: number, dy: number): PlayerInput[][] {
  const out: PlayerInput[][] = [];
  for (const d of [0, -1, 1]) {
    for (const jumps of [false, true]) {
      out.push(
        Array.from({ length: THAW_ESCAPE_TICKS }, (_, t) => {
          const e = emptyInput();
          e.direction = d;
          e.jump = jumps && t % 2 === 0 ? 1 : 0;
          e.hook = 1;
          e.targetX = Math.round(dx) || 1;
          e.targetY = Math.round(dy);
          return e;
        }),
      );
    }
  }
  return out;
}

const FLIGHT_PROBE_TICKS = [6, 12, 18, 24];

function flightEndsInHazard(collision: Collision, pos: Vec2, vel: Vec2): number {

  const half = PHYSICAL_SIZE / 2;
  const grounded = collision.isSolid(pos.x + half, pos.y + half + 5) || collision.isSolid(pos.x - half, pos.y + half + 5);
  for (const t of FLIGHT_PROBE_TICKS) {
    const x = pos.x + vel.x * t;

    const y = grounded ? pos.y : pos.y + vel.y * t + 0.5 * TUNING.gravity * t * t;
    if (collision.isSolid(x, y)) return 0;
    if (collision.isFreeze(x, y) || collision.isDeath(x, y)) return 1;
  }
  return 0;
}

const EDGE_GAP_TILES = 3;
const EDGE_GAP_PX = EDGE_GAP_TILES * TILE_PX;

const EDGE_NEARER_PX = 4;

export function freezeGapPx(collision: Collision, x: number, y: number): number {
  const tx = Math.floor(x / TILE_PX);
  const ty = Math.floor(y / TILE_PX);
  let best = EDGE_GAP_PX;
  for (let oy = -EDGE_GAP_TILES; oy <= EDGE_GAP_TILES; oy++) {
    for (let ox = -EDGE_GAP_TILES; ox <= EDGE_GAP_TILES; ox++) {
      const left = (tx + ox) * TILE_PX;
      const top = (ty + oy) * TILE_PX;
      const cx = left + TILE_PX / 2;
      const cy = top + TILE_PX / 2;
      if (!collision.isFreeze(cx, cy) && !collision.isDeath(cx, cy)) continue;
      const dx = Math.max(left - x, 0, x - (left + TILE_PX));
      const dy = Math.max(top - y, 0, y - (top + TILE_PX));
      const d = Math.hypot(dx, dy);
      if (d < best) best = d;
    }
  }
  return best;
}

export class Planner {
  private readonly cfg: Required<PlannerConfig>;
  private rng: Rng;
  private readonly raw: Float64Array;
  private readonly aimProbe: Float64Array;
  private readonly aimProbeOut: PlayerInput;
  private saved: SimState | undefined;
  private warm: PlanStep[] | null = null;
  private committed: PlayerInput | null = null;
  private commitLeft = 0;

  private lastDecideTick = -1;
  private readonly decideGaps: number[] = [];

  private liveTick = -1;
  private lastFrozen = false;
  private readonly profile = new OpponentProfile();
  private travel: HazardField | null = null;

  private goal: Vec2 | null = null;

  private dead: { width: number; cells: Uint8Array } | null = null;

  private deadFrom: Collision | null = null;

  private memory: FreezeMemory | null = null;
  private frozenBystanders: readonly Vec2[] = [];
  private frozenBystanderVels: readonly Vec2[] = [];

  private spares: readonly Vec2[] = [];
  private spareVels: readonly Vec2[] = [];

  private thirds: Vec2[] = [];
  private band: { x0: number; y0: number; x1: number; y1: number } | null = null;
  private oppSeed = 1;
  private seedOffset = 0;
  private opponentPolicy: RecurrentPolicy | null = null;
  private opponentDirNet: Mlp | null = null;
  private oppDirIn: Float64Array = new Float64Array(0);
  private readonly oppObs = new Float64Array(OBS_SIZE);
  private seedPolicy: RecurrentPolicy | null = null;
  private valueNet: Mlp | null = null;
  private readonly valueObs = new Float64Array(OBS_SIZE);
  private readonly selfObs = new Float64Array(OBS_SIZE);
  private readonly seedRaw = new Float64Array(ACTION_SIZE);
  private predicted: PlayerInput[] = [];

  private trackRollout = false;

  private trackGap = false;
  private rolloutMinGap = EDGE_GAP_PX;

  private reactThisPass = false;

  private swingTargetFrozen = false;

  private swingTarget: TeeState | null = null;
  private thawScratch: SimWorld | null = null;
  private readonly thawMemo = new Map<string, boolean>();
  private swingCollision: Collision | null = null;
  private rolloutEnemyOut = 0;

  private rolloutEnemySealed = false;
  private rolloutSelfOut = 0;

  private readonly stepMeBuf = blankTeeState();
  private readonly stepEnBuf = blankTeeState();
  private readonly tickMeBuf = blankTeeState();
  private readonly tickEnBuf = blankTeeState();

  readonly lastInfo: DecisionInfo = { searched: false, candidates: 0, outOfTime: false, ms: 0, selfOut: -1, enemyOut: -1, hookAt: -1, gated: false };

  private lastSearchTick = -1;
  private warmShiftSteps = 1;
  private dirSince = -1;
  private dirLast = 0;
  private heldTicks = 0;

  private swingRopeOn = false;

  private stepTicks: number[];

  private readonly baseCfg: Required<PlannerConfig>;
  private overridden: Partial<PlannerConfig> | null = null;

  constructor(cfg?: PlannerConfig) {
    this.cfg = { ...PLANNER_DEFAULTS, ...cfg };
    this.baseCfg = { ...this.cfg };
    this.stepTicks = buildStepTicks(this.cfg.steps, this.cfg.planStep, this.cfg.frontSteps, this.cfg.frontStep);
    this.rng = new Rng((this.cfg.seed + this.seedOffset) >>> 0);
    this.raw = new Float64Array(ACTION_SIZE);
    this.aimProbe = new Float64Array(ACTION_SIZE);
    this.aimProbeOut = emptyInput();
  }

  setOverrides(over: Partial<PlannerConfig> | null): void {
    if (over === this.overridden) return;
    this.overridden = over;
    Object.assign(this.cfg, this.baseCfg, over ?? {});
    this.syncGrid();
  }

  config(): Required<PlannerConfig> {
    return { ...this.cfg };
  }

  setSeedPolicy(policy: RecurrentPolicy | null): void {
    this.seedPolicy = policy;
  }

  setValueNet(net: Mlp | null): void {
    this.valueNet = net;
  }

  setOpponentPolicy(policy: RecurrentPolicy | null): void {
    this.opponentPolicy = policy;
  }

  setFreezeMemory(memory: FreezeMemory | null): void {
    this.memory = memory;
  }

  setDeadZone(dead: { width: number; cells: Uint8Array } | null): void {
    this.dead = dead;
    this.deadFrom = null;
  }

  private ensureDeadZone(collision: Collision): void {
    if (this.cfg.deadZoneCost <= 0 && this.cfg.enemyDeadZoneBonus <= 0) return;
    if (this.dead !== null && this.deadFrom === collision) return;
    if (this.dead !== null && this.deadFrom === null) return;
    const cells = deadZoneOf(collision);
    this.dead = cells === null ? null : { width: collision.width, cells };
    this.deadFrom = collision;
  }

  setTravelGoal(goal: Vec2 | null): void {
    this.goal = goal === null ? null : { x: goal.x, y: goal.y };
  }

  setFrozenBystanders(tees: readonly Vec2[], vels: readonly Vec2[] = []): void {
    this.frozenBystanders = tees;
    this.frozenBystanderVels = vels;
  }

  setSpareBystanders(tees: readonly Vec2[], vels: readonly Vec2[] = []): void {
    this.spares = tees;
    this.spareVels = vels;
  }

  setBand(band: { x0: number; y0: number; x1: number; y1: number } | null): void {
    this.band = band;
  }

  setThirdTees(tees: readonly Vec2[]): void {
    this.thirds = tees.map((t) => ({ x: t.x, y: t.y }));
  }

  get travelGoal(): Vec2 | null {
    return this.goal;
  }

  setOpponentDirNet(net: Mlp | null): void {
    this.opponentDirNet = net;
    if (net !== null && this.oppDirIn.length !== OBS_SIZE + ACTION_SIZE) {
      this.oppDirIn = new Float64Array(OBS_SIZE + ACTION_SIZE);
    }
  }

  setSearchSeed(n: number): void {
    this.seedOffset = n >>> 0;
    this.reset();
  }

  reset(): void {
    this.warm = null;
    this.committed = null;
    this.commitLeft = 0;
    this.lastDecideTick = -1;
    this.decideGaps.length = 0;

    this.rng = new Rng((this.cfg.seed + this.seedOffset) >>> 0);

    this.oppSeed = (1 + this.seedOffset) >>> 0;
    this.lastFrozen = false;
    this.lastSearchTick = -1;
    this.warmShiftSteps = 1;
    this.dirSince = -1;
    this.dirLast = 0;
    this.heldTicks = 0;
    this.profile.reset();

    this.opponentPolicy?.reset();
    this.saved = undefined;
  }

  get budget(): { budgetMs: number; hardMs: number; commitDecisions: number; explain: boolean } {
    return { budgetMs: this.cfg.budgetMs, hardMs: this.cfg.hardMs, commitDecisions: this.cfg.commitDecisions, explain: this.cfg.explain };
  }

  setLiveTick(tick: number): void {
    this.liveTick = tick;
  }

  decide(world: SimWorld, selfId: number, enemyId: number, prev: PlayerInput, enemyInput: PlayerInput): PlayerInput {
    const en = world.getTee(enemyId);
    const steps = this.cfg.steps;
    const long = this.cfg.frozenTargetSteps > steps && en !== undefined && en.frozen && en.freezeTicksLeft >= FROZEN_PLAN_MIN_TICKS;
    if (!long) {
      this.syncGrid();
      return this.decideOnce(world, selfId, enemyId, prev, enemyInput);
    }
    this.cfg.steps = this.cfg.frozenTargetSteps;
    this.syncGrid();
    try {
      return this.decideOnce(world, selfId, enemyId, prev, enemyInput);
    } finally {
      this.cfg.steps = steps;
    }
  }

  private syncGrid(): void {
    const grid = buildStepTicks(this.cfg.steps, this.cfg.planStep, this.cfg.frontSteps, this.cfg.frontStep);
    if (grid.length !== this.stepTicks.length || grid.some((t, i) => t !== this.stepTicks[i])) this.stepTicks = grid;
  }

  private decideOnce(world: SimWorld, selfId: number, enemyId: number, prev: PlayerInput, enemyInput: PlayerInput): PlayerInput {
    this.oppSeed = (this.oppSeed * 1664525 + 1013904223) >>> 0;
    this.thawMemo.clear();
    const me = world.getTee(selfId);
    const en = world.getTee(enemyId);
    if (me === undefined || en === undefined || !me.alive || !en.alive) return prev;

    if (this.cfg.opponentReadWeight > 0) this.profile.observe(me, en, HOOK_LENGTH);

    const nowTick = this.liveTick >= 0 ? this.liveTick : world.tick;
    this.liveTick = -1;
    const gap = nowTick - this.lastDecideTick;
    if (this.lastDecideTick >= 0 && gap > 0 && gap <= CADENCE_MAX_GAP) {
      this.decideGaps.push(gap);
      if (this.decideGaps.length > CADENCE_GAPS) this.decideGaps.shift();
    } else if (gap !== 0) this.decideGaps.length = 0;
    this.lastDecideTick = nowTick;

    const frozenNow = me.frozen;
    const urgent = frozenNow !== this.lastFrozen || en.hookedPlayer === selfId;
    this.lastFrozen = frozenNow;
    if (!urgent && this.commitLeft > 0 && this.committed !== null) {
      this.commitLeft--;
      this.lastInfo.searched = false;
      this.lastInfo.candidates = 0;
      this.lastInfo.outOfTime = false;
      this.lastInfo.ms = 0;
      return this.maybeRelease(world, selfId, this.committed);
    }
    const startedMs = Date.now();
    let candidates = 0;

    this.heldTicks = this.dirSince < 0 ? this.cfg.flipHoldTicks : world.tick - this.dirSince;

    const field = hazardField(world.collision);
    const unfreeze = unfreezeField(world.collision);

    this.travel = this.cfg.routeDistance ? travelField(world.collision, en.pos.x, en.pos.y) : null;
    this.ensureDeadZone(world.collision);
    const aimAt = Math.atan2(en.pos.y - me.pos.y, en.pos.x - me.pos.x);

    this.warmShiftSteps = this.warmShift(world.tick);
    this.lastSearchTick = world.tick;
    const dist = this.buildDist(this.cfg.trackAim ? 0 : aimAt);
    this.saved = world.saveState(this.saved);
    this.predictOpponent(world, selfId, enemyId, prev, enemyInput);

    let best: PlanStep[] | null = null;
    let bestScore = -Infinity;

    let bestStay: PlanStep[] | null = null;
    let bestStayScore = -Infinity;

    let carry: PlanStep[] | null = null;
    let carryScore = -Infinity;
    if (this.cfg.planMargin > 0 && this.warm !== null && this.warm.length === this.cfg.steps) {

      const shift = this.warmShiftSteps;
      carry = this.warm.slice(shift).concat(this.warm.slice(this.warm.length - shift).map((s) => ({ ...s })));
      if (carry.length !== this.cfg.steps) carry = this.warm.map((s) => ({ ...s }));
      carryScore = this.evaluate(world, selfId, enemyId, prev, carry, enemyInput, field, unfreeze);
    }

    const hardline = this.cfg.hardMs > 0 ? startedMs + this.cfg.hardMs : Infinity;
    const deadline = Math.min(this.cfg.budgetMs > 0 ? Date.now() + this.cfg.budgetMs : Infinity, hardline);
    let outOfTime = false;

    const overCap = (): boolean => hardline !== Infinity && Date.now() > hardline;

    for (let it = 0; it < this.cfg.iterations && !outOfTime; it++) {

      this.reactThisPass = this.cfg.opponentMix && it > 0;
      const scored: { plan: PlanStep[]; score: number }[] = [];
      const seeds = it === 0 ? this.seedPlans(world, selfId, enemyId, field, aimAt) : [];
      for (const plan of seeds) {
        if (best !== null && overCap()) {
          outOfTime = true;
          break;
        }
        const score = this.evaluate(world, selfId, enemyId, prev, plan, enemyInput, field, unfreeze);
        scored.push({ plan, score });
        if (score > bestScore) {
          bestScore = score;
          best = plan;
        }
        if (plan[0].dir === prev.direction && score > bestStayScore) {
          bestStayScore = score;
          bestStay = plan;
        }
      }
      if (it === 0 && !outOfTime) {
        for (const plan of this.policySeedPlans(world, selfId, enemyId, prev, enemyInput, aimAt)) {
          if (best !== null && overCap()) {
            outOfTime = true;
            break;
          }
          const score = this.evaluate(world, selfId, enemyId, prev, plan, enemyInput, field, unfreeze);
          scored.push({ plan, score });
          if (score > bestScore) {
            bestScore = score;
            best = plan;
          }
          if (plan[0].dir === prev.direction && score > bestStayScore) {
            bestStayScore = score;
            bestStay = plan;
          }
        }
      }

      if (it === 0 && (this.cfg.freezeThrow > 0 || this.cfg.frozenThrow > 0) && !outOfTime && !overCap()) {
        for (const landed of this.landedThrows(world, selfId, enemyId, prev, enemyInput, field, unfreeze, aimAt, overCap)) {
          scored.push(landed);
          if (landed.score > bestScore) {
            bestScore = landed.score;
            best = landed.plan;
          }
          if (landed.plan[0].dir === prev.direction && landed.score > bestStayScore) {
            bestStayScore = landed.score;
            bestStay = landed.plan;
          }
        }
      }
      for (let i = 0; i < this.cfg.population && !outOfTime; i++) {

        if (deadline !== Infinity && (i & 3) === 3 && Date.now() > deadline) {
          outOfTime = true;
          break;
        }
        const plan = this.samplePlan(dist);
        const score = this.evaluate(world, selfId, enemyId, prev, plan, enemyInput, field, unfreeze);
        scored.push({ plan, score });
        if (score > bestScore) {
          bestScore = score;
          best = plan;
        }
        if (plan[0].dir === prev.direction && score > bestStayScore) {
          bestStayScore = score;
          bestStay = plan;
        }
      }
      candidates += scored.length;
      scored.sort((a, b) => b.score - a.score);
      this.refit(dist, scored.slice(0, this.cfg.elite).map((e) => e.plan));
    }

    world.restoreState(this.saved);
    if (best === null) return prev;

    const flips = best[0].dir !== prev.direction;

    const plainHold = this.cfg.flipMargin > 0 && bestStay !== null && flips && bestScore - bestStayScore < this.cfg.flipMargin;

    let notNowIn = false;
    if (this.cfg.edgeHold && flips && !me.frozen && !overCap() && freezeGapPx(world.collision, me.pos.x, me.pos.y) < EDGE_GAP_PX) {
      const notNow = this.notNow(world, selfId, enemyId, prev, best, enemyInput, field, unfreeze);
      if (notNow !== null) {
        notNowIn = true;
        if (notNow.score > bestStayScore) {
          bestStayScore = notNow.score;
          bestStay = notNow.plan;
        }
      }
    }
    this.lastInfo.edgeHeld = false;
    if ((this.cfg.flipMargin > 0 || notNowIn) && bestStay !== null && flips && bestScore - bestStayScore < this.cfg.flipMargin) {
      best = bestStay;

      bestScore = bestStayScore;
      this.lastInfo.edgeHeld = !plainHold;
    }

    if (carry !== null && bestScore - carryScore < this.cfg.planMargin) {
      best = carry;
      bestScore = carryScore;
    }
    this.reactThisPass = false;

    if (this.cfg.hookPolish && best[0].hook === 0 && !overCap()) {
      const polished = this.polishRope(world, selfId, enemyId, prev, enemyInput, field, unfreeze, best, bestScore);
      if (polished !== null) {
        best = polished.plan;
        bestScore = polished.score;
      }
    }

    if (this.cfg.escapeBias > 0 && !overCap()) {
      this.trackRollout = true;
      this.evaluate(world, selfId, enemyId, prev, best, enemyInput, field, unfreeze);
      this.trackRollout = false;
      if (this.rolloutSelfOut > 0) {
        const wasBias = this.cfg.selfFreezeBias;
        this.cfg.selfFreezeBias = wasBias * this.cfg.escapeBias;
        let escape: PlanStep[] | null = null;
        let escapeScore = -Infinity;
        const escDist = this.buildDist(this.cfg.trackAim ? 0 : aimAt);
        for (let i = 0; i < this.cfg.population; i++) {
          const plan = this.samplePlan(escDist);
          const score = this.evaluate(world, selfId, enemyId, prev, plan, enemyInput, field, unfreeze);
          if (score > escapeScore) {
            escapeScore = score;
            escape = plan;
          }
        }
        this.cfg.selfFreezeBias = wasBias;
        if (escape !== null) {

          const fair = this.evaluate(world, selfId, enemyId, prev, escape, enemyInput, field, unfreeze);
          this.trackRollout = true;
          this.evaluate(world, selfId, enemyId, prev, escape, enemyInput, field, unfreeze);
          this.trackRollout = false;
          if (this.rolloutSelfOut === 0 && fair >= bestScore - this.cfg.escapeMargin) {
            best = escape;
            bestScore = fair;
          }
        }
      }
    }
    this.warm = best;
    this.lastInfo.searched = true;
    this.lastInfo.candidates = candidates;
    this.lastInfo.outOfTime = outOfTime;
    this.lastInfo.ms = Date.now() - startedMs;
    this.lastInfo.hookAt = -1;

    for (let i = 0, at = 0; i < best.length; at += this.stepTicks[i], i++) {
      if (best[i].hook === 1) {
        this.lastInfo.hookAt = at;
        break;
      }
    }
    if (this.cfg.explain && !overCap()) {
      this.trackRollout = true;
      this.evaluate(world, selfId, enemyId, prev, best, enemyInput, field, unfreeze);
      this.trackRollout = false;
      this.lastInfo.selfOut = this.rolloutSelfOut;
      this.lastInfo.enemyOut = this.rolloutEnemyOut;
    } else {

      this.lastInfo.selfOut = -1;
      this.lastInfo.enemyOut = -1;
    }

    const rest = this.cfg.restAim && best[0].hook === 0 && best[0].fire === 0;
    const aim0 = rest ? aimAt : this.cfg.trackAim ? aimAt + best[0].aim : best[0].aim;

    const hookOk = !best[0].hook || this.hookAlreadyOut(world, selfId) || this.hookAllowed(world, selfId, enemyId, best[0], prev, aim0);
    this.lastInfo.gated = best[0].hook === 1 && !hookOk;

    this.swingTargetFrozen = en.frozen;
    this.swingRopeOn = me.hookedPlayer === enemyId;
    this.swingTarget = en;
    this.swingCollision = world.collision;
    let chosen = this.stepToInput(best[0], prev, vdistance(me.pos, en.pos), hookOk, me.pos, en.pos, en.vel, aim0);

    this.lastInfo.shielded = false;
    if (this.cfg.shield && !me.frozen) {
      const hold = this.shieldHold();
      const others = new Map([[enemyId, enemyInput]]);
      if (!escapeExists(world, selfId, chosen, hold, others)) {
        const safer = saferInput(world, selfId, chosen, hold, others, prev);
        if (safer !== null) {
          chosen = safer;
          this.lastInfo.shielded = true;
        }
      }
    }
    if (chosen.direction !== this.dirLast || this.dirSince < 0) {
      this.dirLast = chosen.direction;
      this.dirSince = world.tick;
    }
    this.committed = chosen;
    this.commitLeft = Math.max(0, this.cfg.commitDecisions - 1);
    return this.maybeRelease(world, selfId, chosen);
  }

  shieldHold(): number {
    const commit = Math.max(1, this.cfg.commitDecisions);
    if (!this.cfg.shieldCadence) return 2 * commit;

    const sorted = [...this.decideGaps].sort((a, b) => a - b);
    const typical = sorted.length > 0 ? sorted[Math.floor((sorted.length - 1) / 2)] : 2;
    const perDecision = Math.min(CADENCE_MAX_TICKS, Math.max(2, typical));
    return Math.min(SHIELD_MAX_HOLD, perDecision * commit);
  }

  private maybeRelease(world: SimWorld, selfId: number, input: PlayerInput): PlayerInput {
    if (!this.cfg.releaseDeadHook || input.hook === 0 || !this.hookIsDead(world, selfId)) return input;
    return { ...input, hook: 0 };
  }

  private notNow(
    world: SimWorld,
    selfId: number,
    enemyId: number,
    prev: PlayerInput,
    turning: PlanStep[],
    enemyInput: PlayerInput,
    field: HazardField,
    unfreeze: HazardField,
  ): { plan: PlanStep[]; score: number } | null {
    const later = turning.map((st, i) => ({ ...st, dir: i === 0 ? prev.direction : turning[i - 1].dir }));
    const kept = turning.map((st, i) => (i === 0 ? { ...st, dir: prev.direction } : { ...st }));
    this.trackGap = true;
    this.evaluate(world, selfId, enemyId, prev, turning, enemyInput, field, unfreeze);
    const turnGap = this.rolloutMinGap;
    let out: { plan: PlanStep[]; score: number; gap: number } | null = null;
    for (const plan of [later, kept]) {
      const score = this.evaluate(world, selfId, enemyId, prev, plan, enemyInput, field, unfreeze);
      if (out === null || score > out.score) out = { plan, score, gap: this.rolloutMinGap };
    }
    this.trackGap = false;
    if (out === null || out.gap < turnGap - EDGE_NEARER_PX) return null;
    return { plan: out.plan, score: out.score };
  }

  private landedThrows(
    world: SimWorld,
    selfId: number,
    enemyId: number,
    prev: PlayerInput,
    enemyInput: PlayerInput,
    field: HazardField,
    unfreeze: HazardField,
    aimAt: number,

    overCap: () => boolean = () => false,
  ): { plan: PlanStep[]; score: number }[] {
    const me = world.getTee(selfId);
    const en = world.getTee(enemyId);
    if (me === undefined || en === undefined) return [];
    const situation = {
      separation: vdistance(me.pos, en.pos),
      enemyHazardNearness: hazardNearness(field, en.pos.x, en.pos.y),
      meFrozen: me.frozen,
      enemyFrozen: en.frozen,
      enemyAlive: en.alive,
    };
    if (this.cfg.frozenThrow > 0 && frozenThrowWorthTrying(situation) && en.freezeTicksLeft >= FROZEN_PLAN_MIN_TICKS) {
      const kept: { plan: PlanStep[]; score: number }[] = [];
      this.trackRollout = true;
      for (const plan of frozenThrowLines(this.cfg.steps, this.cfg.trackAim ? 0 : aimAt)) {
        if (overCap()) break;
        const score = this.evaluate(world, selfId, enemyId, prev, plan, enemyInput, field, unfreeze);
        if (this.rolloutEnemySealed && this.rolloutSelfOut === 0) kept.push({ plan, score });
      }
      this.trackRollout = false;
      kept.sort((a, b) => b.score - a.score);
      return kept.slice(0, this.cfg.frozenThrow);
    }
    if (this.cfg.freezeThrow <= 0 || !throwWorthTrying(situation)) return [];

    const kept: { plan: PlanStep[]; score: number; gain: number }[] = [];
    this.trackRollout = true;
    for (const plan of throwLines(this.cfg.steps, this.cfg.trackAim ? 0 : aimAt)) {
      if (overCap()) break;
      const score = this.evaluate(world, selfId, enemyId, prev, plan, enemyInput, field, unfreeze);
      const gain = this.rolloutEnemyOut - this.rolloutSelfOut;
      if (this.rolloutEnemyOut >= THROW_LANDED_TICKS && gain > 0) kept.push({ plan, score, gain });
    }
    this.trackRollout = false;

    kept.sort((a, b) => b.gain - a.gain || b.score - a.score);
    return kept.slice(0, this.cfg.freezeThrow).map((k) => ({ plan: k.plan, score: k.score }));
  }

  private polishRope(
    world: SimWorld,
    selfId: number,
    enemyId: number,
    prev: PlayerInput,
    enemyInput: PlayerInput,
    field: HazardField,
    unfreeze: HazardField,
    best: PlanStep[],
    bestScore: number,
  ): { plan: PlanStep[]; score: number } | null {
    const me = world.getTee(selfId);
    const en = world.getTee(enemyId);
    if (me === undefined || en === undefined || !me.alive || !en.alive || me.frozen || en.frozen) return null;
    const holding = me.hookedPlayer === enemyId;

    const free = me.hookState === HOOK_IDLE;
    if (!holding && !(free && vdistance(me.pos, en.pos) < HOOK_LENGTH)) return null;

    const bearing = Math.atan2(en.pos.y - me.pos.y, en.pos.x - me.pos.x);
    const throwAim = this.cfg.trackAim ? 0 : bearing;

    if (!holding && this.cfg.gateHook && !this.hookWouldReach(world, selfId, enemyId, this.executedAim({ ...best[0], hook: 1 }, prev, bearing))) return null;
    let out: { plan: PlanStep[]; score: number } | null = null;
    const n = best.length;
    for (const k of [2, 4, n]) {
      if (k > n) continue;
      const plan = best.map((st, i) => (i < k ? { ...st, hook: 1, aim: holding ? st.aim : throwAim } : { ...st }));
      const score = this.evaluate(world, selfId, enemyId, prev, plan, enemyInput, field, unfreeze);
      if (score >= bestScore && (out === null || score > out.score)) out = { plan, score };
    }
    return out;
  }

  private warmShift(tick: number): number {
    if (!this.cfg.warmShiftElapsed) return 1;
    if (this.lastSearchTick < 0) return 1;
    const elapsed = Math.max(0, tick - this.lastSearchTick);
    let acc = 0;
    let steps = 0;
    while (steps < this.stepTicks.length && acc + this.stepTicks[steps] <= elapsed) {
      acc += this.stepTicks[steps];
      steps++;
    }
    return steps;
  }

  private buildDist(aimAt: number): StepDist[] {
    const dist: StepDist[] = [];
    for (let s = 0; s < this.cfg.steps; s++) {
      const w = this.warm !== null ? this.warm[Math.min(s + this.warmShiftSteps, this.warm.length - 1)] : null;
      if (w === null) {
        dist.push({ pLeft: 0.33, pRight: 0.33, pJump: 0.15, pHook: 0.3, pFire: 0.2, aim: aimAt, aimSpread: 1.2 });
      } else {

        dist.push({
          pLeft: w.dir === -1 ? 0.7 : 0.15,
          pRight: w.dir === 1 ? 0.7 : 0.15,
          pJump: w.jump ? 0.7 : 0.1,
          pHook: w.hook ? 0.7 : 0.15,
          pFire: w.fire ? 0.6 : 0.15,
          aim: w.aim,
          aimSpread: 0.9,
        });
      }
    }
    return dist;
  }

  private samplePlan(dist: StepDist[]): PlanStep[] {
    const plan: PlanStep[] = [];
    for (const d of dist) {
      const r = this.rng.nextFloat();
      plan.push({
        dir: r < d.pLeft ? -1 : r < d.pLeft + d.pRight ? 1 : 0,
        jump: this.rng.nextFloat() < d.pJump ? 1 : 0,
        hook: this.rng.nextFloat() < d.pHook ? 1 : 0,
        fire: this.rng.nextFloat() < d.pFire ? 1 : 0,
        aim: d.aim + this.rng.nextGaussian() * d.aimSpread,
      });
    }
    return plan;
  }

  private refit(dist: StepDist[], elites: PlanStep[][]): void {
    if (elites.length === 0) return;
    for (let s = 0; s < dist.length; s++) {
      let left = 0;
      let right = 0;
      let jump = 0;
      let hook = 0;
      let fire = 0;
      let ax = 0;
      let ay = 0;
      for (const plan of elites) {
        const st = plan[s];
        if (st.dir === -1) left++;
        else if (st.dir === 1) right++;
        jump += st.jump;
        hook += st.hook;
        fire += st.fire;
        ax += Math.cos(st.aim);
        ay += Math.sin(st.aim);
      }
      const n = elites.length;
      const d = dist[s];

      d.pLeft = 0.1 + 0.8 * (left / n);
      d.pRight = 0.1 + 0.8 * (right / n);
      d.pJump = 0.05 + 0.9 * (jump / n);
      d.pHook = 0.05 + 0.9 * (hook / n);
      d.pFire = 0.05 + 0.9 * (fire / n);
      d.aim = Math.atan2(ay / n, ax / n);
      d.aimSpread = Math.max(0.25, d.aimSpread * 0.7);
    }
  }

  private seedPlans(world: SimWorld, selfId: number, enemyId: number, field: HazardField, aimAt: number): PlanStep[][] {
    const me = world.getTee(selfId)!;
    const en = world.getTee(enemyId)!;
    const toward = Math.sign(en.pos.x - me.pos.x) || 1;

    let hazardDir = toward;
    let bestNear = 0;
    for (const a of [0, 1, 3, 4, 5, 7]) {
      const ang = (a * Math.PI) / 4;
      const near = hazardNearness(field, en.pos.x + Math.cos(ang) * 96, en.pos.y + Math.sin(ang) * 96);
      if (near > bestNear) {
        bestNear = near;
        hazardDir = Math.cos(ang) > 0 ? 1 : -1;
      }
    }

    const n = this.cfg.steps;
    const mk = (fn: (s: number) => PlanStep): PlanStep[] => Array.from({ length: n }, (_, s) => fn(s));

    const track = this.cfg.trackAim;
    const at = track ? 0 : aimAt;
    const rel = (absolute: number): number => (track ? wrapAngle(absolute - aimAt) : absolute);
    const up = rel(-Math.PI / 2 + 0.4 * toward);
    const away = rel(Math.atan2(0, -toward));
    const book: PlanStep[][] = [

      mk((s) => ({ dir: s < 3 ? toward : hazardDir, jump: s === 2 ? 1 : 0, hook: s >= 2 ? 1 : 0, fire: s > n - 4 ? 1 : 0, aim: at })),

      mk((s) => ({ dir: hazardDir, jump: 0, hook: s >= 1 && s < n / 2 ? 1 : 0, fire: s >= n / 2 && s < n / 2 + 3 ? 1 : 0, aim: at })),

      mk((s) => ({ dir: toward, jump: s % 6 === 0 ? 1 : 0, hook: 0, fire: s > 2 ? 1 : 0, aim: at })),

      mk((s) => ({ dir: toward, jump: s === 0 ? 1 : 0, hook: 1, fire: 0, aim: up })),

      mk(() => ({ dir: -toward, jump: 0, hook: 0, fire: 0, aim: at })),
    ];

    const read = this.profile.read();

    const w = Math.max(0, Math.min(1, this.cfg.opponentReadWeight));
    const mix = (v: number, prior: number): number => prior + w * (v - prior);
    {

      if (mix(read.hookOpensFirst, 0.5) > 0.6) {

        book.push(mk((s) => ({ dir: s < n / 2 ? -toward : toward, jump: s === Math.round(n / 2) ? 1 : 0, hook: 0, fire: s > n - 4 ? 1 : 0, aim: at })));
      }
      if (mix(read.aggression, 0.5) > 0.6) {

        book.push(mk((s) => ({ dir: 0, jump: 0, hook: 0, fire: s > 1 ? 1 : 0, aim: at })));
      } else if (mix(read.aggression, 0.5) < 0.4) {

        book.push(mk((s) => ({ dir: toward, jump: s % 6 === 0 ? 1 : 0, hook: 0, fire: s > 2 ? 1 : 0, aim: at })));
      }
      if (mix(read.outOfJumps, 0.3) > 0.5) {

        book.push(mk((s) => ({ dir: toward, jump: 0, hook: 0, fire: s > 0 ? 1 : 0, aim: at })));
      }
    }

    const third = Math.max(2, Math.round(n / 3));

    if (this.cfg.hookSeeds && en.alive && !en.frozen && !me.frozen && vdistance(me.pos, en.pos) < HOOK_LENGTH) {
      if (me.hookedPlayer === enemyId) {

        if (this.warm !== null && this.warm.length === n) {
          const shift = this.warmShiftSteps;
          book.push(mk((s) => ({ ...this.warm![Math.min(s + shift, n - 1)], hook: 1 })));
        }
        book.push(mk(() => ({ dir: hazardDir, jump: 0, hook: 1, fire: 0, aim: at })));
      } else if (me.hookState !== HOOK_FLYING && me.hookState !== HOOK_GRABBED) {

        book.push(mk(() => ({ dir: hazardDir, jump: 0, hook: 1, fire: 0, aim: at })));
        book.push(mk((s) => ({ dir: hazardDir, jump: 0, hook: s < third ? 1 : 0, fire: 0, aim: at })));
      }
    }

    const which = this.cfg.openingBook;
    if (which === "movement" || which === "all") this.movementSeeds(world, me, toward, at, rel, mk, book);
    if (which !== "wide" && which !== "all") return book;
    book.push(

      mk(() => ({ dir: hazardDir, jump: 0, hook: 1, fire: 0, aim: at })),

      mk((s) => ({ dir: hazardDir, jump: 0, hook: s < third ? 1 : 0, fire: 0, aim: at })),

      mk((s) => ({ dir: hazardDir, jump: s === 1 ? 1 : 0, hook: s < 2 * third ? 1 : 0, fire: 0, aim: at })),

      mk((s) => ({ dir: 0, jump: 0, hook: s < 2 * third ? 1 : 0, fire: s >= third ? 1 : 0, aim: at })),

      mk((s) => ({ dir: s < third ? 0 : hazardDir, jump: 0, hook: s < 2 * third ? 1 : 0, fire: s >= third ? 1 : 0, aim: at })),

      mk((s) => ({ dir: toward, jump: 0, hook: 0, fire: s >= 1 ? 1 : 0, aim: at })),

      mk((s) => ({ dir: s < third ? toward : -toward, jump: s === 0 || s === 2 ? 1 : 0, hook: 0, fire: s >= third + 1 ? 1 : 0, aim: at })),

      mk((s) => ({ dir: -toward, jump: s === 0 || s === 3 ? 1 : 0, hook: 0, fire: 0, aim: at })),

      mk((s) => ({ dir: -toward, jump: 0, hook: s < third ? 1 : 0, fire: 0, aim: away })),

      mk(() => ({ dir: 0, jump: 0, hook: 0, fire: 0, aim: at })),
    );
    return book;
  }

  private movementSeeds(
    world: SimWorld,
    me: { pos: Vec2; vel: Vec2 },
    toward: number,
    at: number,
    rel: (absolute: number) => number,
    mk: (fn: (s: number) => PlanStep) => PlanStep[],
    book: PlanStep[][],
  ): void {
    const n = this.cfg.steps;
    const travel = Math.abs(me.vel.x) >= 1 ? Math.sign(me.vel.x) : toward;

    const upAhead = rel(Math.atan2(-1, 0.8 * travel));
    const upAheadSteep = rel(Math.atan2(-1, 0.4 * travel));
    const ceiling = rel(Math.atan2(-1, 0.15 * travel));
    const upBehind = rel(Math.atan2(-1.2, -0.5 * travel));

    book.push(mk((s) => ({ dir: travel, jump: s === 3 ? 1 : 0, hook: s < 3 ? 1 : 0, fire: 0, aim: s < 3 ? upAhead : at })));
    book.push(mk((s) => ({ dir: travel, jump: s === 4 ? 1 : 0, hook: s < 4 ? 1 : 0, fire: 0, aim: s < 4 ? upAheadSteep : at })));

    book.push(mk((s) => ({
      dir: s < 2 ? travel : s < 4 ? -travel : travel,
      jump: s === 6 ? 1 : 0,
      hook: s < 6 ? 1 : 0,
      fire: 0,
      aim: s < 6 ? ceiling : at,
    })));

    book.push(mk((s) => ({ dir: s < 1 ? travel : -travel, jump: s === 4 ? 1 : 0, hook: s < 4 ? 1 : 0, fire: 0, aim: s < 4 ? upBehind : at })));

    const edgeStep = this.stepsToEdge(world, me, travel);
    if (edgeStep !== null && edgeStep < n - 1) {
      book.push(mk((s) => ({ dir: travel, jump: s === edgeStep || s === edgeStep + 3 ? 1 : 0, hook: 0, fire: 0, aim: at })));
    }

    book.push(mk(() => ({ dir: travel, jump: 0, hook: 0, fire: 0, aim: at })));
  }

  private stepsToEdge(world: SimWorld, me: { pos: Vec2; vel: Vec2 }, travel: number): number | null {
    const feetY = me.pos.y + PHYSICAL_SIZE / 2 + 4;
    if (!world.collision.isSolid(me.pos.x, feetY)) return null;
    const tx = Math.floor(me.pos.x / TILE_PX);
    for (let k = 1; k <= 8; k++) {
      const cx = (tx + k * travel) * TILE_PX + TILE_PX / 2;
      if (world.collision.isSolid(cx, feetY)) continue;

      const lipX = travel > 0 ? (tx + k) * TILE_PX : (tx - k + 1) * TILE_PX;
      const px = Math.abs(lipX - me.pos.x);
      const speed = Math.max(Math.abs(me.vel.x), TUNING.groundControlSpeed * 0.6);

      return this.stepAtTick(px / speed);
    }
    return null;
  }

  private stepAtTick(ticks: number): number {
    let acc = 0;
    let best = 0;
    let bestGap = Math.abs(ticks);
    for (let i = 0; i < this.stepTicks.length; i++) {
      acc += this.stepTicks[i];
      const gap = Math.abs(ticks - acc);

      if (gap <= bestGap) {
        bestGap = gap;
        best = i + 1;
      }
    }
    return best;
  }

  private hookIsDead(world: SimWorld, selfId: number): boolean {
    const me = world.getTee(selfId);
    if (me === undefined) return false;
    return me.hookState !== HOOK_IDLE && me.hookState !== HOOK_FLYING && me.hookState !== HOOK_GRABBED;
  }

  private hookAlreadyOut(world: SimWorld, selfId: number): boolean {
    const me = world.getTee(selfId);
    if (me === undefined) return false;
    return me.hookState === HOOK_FLYING || me.hookState === HOOK_GRABBED;
  }

  private hookAllowed(world: SimWorld, selfId: number, enemyId: number, step: PlanStep, prev: PlayerInput, aim: number): boolean {
    if (!this.cfg.gateHook && this.spares.length === 0) return true;
    const angle = this.executedAim(step, prev, aim);
    if (this.cfg.gateHook && !this.hookWouldReach(world, selfId, enemyId, angle)) return false;
    return !this.ropeCatchesSpare(world, selfId, enemyId, angle);
  }

  private ropeCatchesSpare(world: SimWorld, selfId: number, enemyId: number, angle: number): boolean {
    if (this.spares.length === 0) return false;
    const me = world.getTee(selfId);
    if (me === undefined) return false;
    const dir = { x: Math.cos(angle), y: Math.sin(angle) };
    const hit = world.collision.intersectLineHook(me.pos, { x: me.pos.x + dir.x * HOOK_LENGTH, y: me.pos.y + dir.y * HOOK_LENGTH });
    let stop = hit.collision !== 0 ? vdistance(me.pos, hit.outPos) : HOOK_LENGTH;
    const en = world.getTee(enemyId);
    if (en !== undefined && en.alive) stop = Math.min(stop, ropeCatchAlong(me.pos, dir, en.pos));
    for (const b of this.spares) if (ropeCatchAlong(me.pos, dir, b) < stop) return true;
    return false;
  }

  private hookWouldReach(world: SimWorld, selfId: number, enemyId: number, angle: number): boolean {
    const me = world.getTee(selfId);
    if (me === undefined) return false;
    const dir = { x: Math.cos(angle), y: Math.sin(angle) };
    const to = { x: me.pos.x + dir.x * HOOK_LENGTH, y: me.pos.y + dir.y * HOOK_LENGTH };

    const hit = world.collision.intersectLineHook(me.pos, to);
    const wallHit = hit.collision !== 0;
    if (wallHit && (hit.collision & CFLAG_NOHOOK) === 0) return true;
    const en = world.getTee(enemyId);
    if (en === undefined || !en.alive) return false;

    const wallDist = wallHit ? vdistance(me.pos, hit.outPos) : HOOK_LENGTH;

    const rel = { x: en.pos.x - me.pos.x, y: en.pos.y - me.pos.y };
    const along = rel.x * dir.x + rel.y * dir.y;
    if (along < 0 || along > HOOK_LENGTH) return false;
    const perp = Math.abs(rel.x * dir.y - rel.y * dir.x);

    if (along > wallDist) return false;
    if (perp <= PHYSICAL_SIZE * 2) return true;
    const lead = { x: en.pos.x + en.vel.x * 8 - me.pos.x, y: en.pos.y + en.vel.y * 8 - me.pos.y };
    const leadAlong = lead.x * dir.x + lead.y * dir.y;
    if (leadAlong < 0 || leadAlong > Math.min(HOOK_LENGTH, wallDist)) return false;
    return Math.abs(lead.x * dir.y - lead.y * dir.x) <= PHYSICAL_SIZE * 2;
  }

  private policySeedPlans(world: SimWorld, selfId: number, enemyId: number, prev: PlayerInput, enemyInput: PlayerInput, aimAt: number): PlanStep[][] {
    const policy = this.seedPolicy;
    const wanted = Math.max(0, Math.trunc(this.cfg.policySeeds));
    if (policy === null || wanted === 0) return [];

    this.seedRaw.set(policy.act(encodeObs(world, selfId, enemyId, this.selfObs)));
    const memory = policy.saveState();

    const line: PlanStep[] = [];
    let input = prev;
    let oppInput = enemyInput;
    for (let step = 0; step < this.cfg.steps; step++) {
      input = decodeAction(this.seedRaw, input);
      const me = world.getTee(selfId);
      const en = world.getTee(enemyId);
      let aim = Math.atan2(input.targetY, input.targetX);

      if (this.cfg.trackAim && me !== undefined && en !== undefined) {
        aim -= Math.atan2(en.pos.y - me.pos.y, en.pos.x - me.pos.x);
      }
      line.push({ dir: input.direction, jump: input.jump, hook: input.hook, fire: input.fire & 1, aim });
      if (this.predicted.length > 0) oppInput = this.predicted[Math.min(step, this.predicted.length - 1)];
      for (let t = 0; t < this.stepTicks[step]; t++) {
        world.setInput(selfId, input);
        world.setInput(enemyId, oppInput);
        world.step();
      }
      this.seedRaw.set(policy.act(encodeObs(world, selfId, enemyId, this.selfObs)));
    }
    policy.restoreState(memory);
    world.restoreState(this.saved!);

    const lead = Math.trunc(this.cfg.policySeedSteps);
    const trimmed =
      lead > 0 && lead < line.length
        ? line.slice(0, lead).concat(this.samplePlan(this.buildDist(this.cfg.trackAim ? 0 : aimAt)).slice(lead))
        : line;

    const out = [trimmed];

    for (let i = 1; i < wanted; i++) {
      const push = (i % 2 === 1 ? 1 : -1) * this.cfg.policySeedJitter * Math.ceil(i / 2);
      out.push(trimmed.map((st, i) => (lead > 0 && i >= lead ? { ...st } : { ...st, aim: st.aim + push })));
    }
    return out;
  }

  private predictedDir(world: SimWorld, selfId: number, enemyId: number, enemyInput: PlayerInput): number | null {
    const net = this.opponentDirNet;
    if (this.cfg.opponentModel !== "learned" || net === null) return null;
    const en = world.getTee(enemyId);
    if (en === undefined) return null;

    encodeObs(world, enemyId, selfId, this.oppObs);
    const x = this.oppDirIn;
    x.set(this.oppObs, 0);
    encodeHumanTarget(enemyInput, (enemyInput.fire & 1) !== 0, en.activeWeapon, x, OBS_SIZE);
    const out = net.forward(x);
    let k = 0;
    if (out[1] > out[k]) k = 1;
    if (out[2] > out[k]) k = 2;
    return k === 0 ? -1 : k === 1 ? 0 : 1;
  }

  private predictOpponent(world: SimWorld, selfId: number, enemyId: number, prev: PlayerInput, enemyInput: PlayerInput): void {
    if (this.cfg.opponentModel === "learned") {
      const dir = this.predictedDir(world, selfId, enemyId, enemyInput);
      this.predicted = dir === null ? [] : [{ ...enemyInput, direction: dir }];
      return;
    }
    const policy = this.opponentPolicy;
    if (this.cfg.opponentModel !== "policy" || policy === null) {
      this.predicted = [];
      return;
    }

    let raw = policy.act(encodeObs(world, enemyId, selfId, this.oppObs));
    const memory = policy.saveState();
    let oppInput = enemyInput;
    const out: PlayerInput[] = [];
    for (let s = 0; s < this.cfg.steps; s++) {
      oppInput = decodeAction(raw, oppInput);
      out.push(oppInput);

      for (let t = 0; t < this.stepTicks[s]; t++) {
        world.setInput(selfId, prev);
        world.setInput(enemyId, oppInput);
        world.step();
      }
      const en = world.getTee(enemyId);
      if (en === undefined || !en.alive) break;
      raw = policy.act(encodeObs(world, enemyId, selfId, this.oppObs));
    }
    world.restoreState(this.saved!);
    policy.restoreState(memory);
    this.predicted = out;
  }

  private hammerWouldHit(mePos: Vec2, enPos: Vec2, enVel: Vec2, targetX: number, targetY: number): boolean {
    const len = Math.sqrt(targetX * targetX + targetY * targetY);
    if (len < 1e-6) return false;
    const sx = mePos.x + (targetX / len) * PHYSICAL_SIZE * 0.75;
    const sy = mePos.y + (targetY / len) * PHYSICAL_SIZE * 0.75;
    const reach = PHYSICAL_SIZE * 0.5 + PHYSICAL_SIZE;
    if (Math.hypot(enPos.x - sx, enPos.y - sy) < reach) return true;
    return Math.hypot(enPos.x + enVel.x * 2 - sx, enPos.y + enVel.y * 2 - sy) < reach;
  }

  private thawEscapable(collision: Collision, en: TeeState, mePos: Vec2): boolean {
    const strict = this.cfg.noThawRope;
    const key = `${strict ? "s" : ""}${Math.round(en.pos.x / 4)},${Math.round(en.pos.y / 4)},${Math.round(en.vel.x)},${Math.round(en.vel.y)},${Math.round((en.pos.x - mePos.x) / 8)},${Math.round((en.pos.y - mePos.y) / 8)},${en.jumpsLeft},${en.freezeTicksLeft > THAW_ESCAPE_TICKS ? 1 : 0}`;
    const known = this.thawMemo.get(key);
    if (known !== undefined) return known;
    if (this.thawScratch === null || this.thawScratch.collision !== collision) {
      this.thawScratch = new SimWorld(collision, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
      this.thawScratch.addTee(0, en.pos);
    }
    if (strict && this.thawScratch.getTee(1) === undefined) this.thawScratch.addTee(1, mePos);
    if (!strict && this.thawScratch.getTee(1) !== undefined) this.thawScratch.removeTee(1);
    const scratch = this.thawScratch;
    const escapes = strict ? [...THAW_ESCAPES, ...ropeEscapes(mePos.x - en.pos.x, mePos.y - en.pos.y)] : THAW_ESCAPES;
    const sep = Math.max(1, vdistance(en.pos, mePos));
    const hx = (en.pos.x - mePos.x) / sep;
    const hy = (en.pos.y - mePos.y) / sep;
    const bl = Math.hypot(hx, hy - 1.1) || 1;
    const k = TUNING.hammerStrength;
    const push = { x: (k * 10 * hx) / bl, y: k * (-1 + (10 * (hy - 1.1)) / bl) };
    let escapable = false;
    for (const escape of escapes) {
      scratch.applyTeeState(0, { ...en, id: 0, hookState: 0, hookedPlayer: -1 });
      scratch.setHeldInput(0, emptyInput());
      if (strict) {
        scratch.applyTeeState(1, { ...blankTeeState(), id: 1, alive: true, pos: { x: mePos.x, y: mePos.y } });
        scratch.setHeldInput(1, emptyInput());
        scratch.setInput(1, emptyInput());
      }
      scratch.applyForce(0, push);
      scratch.unfreeze(0);
      let caught = false;
      for (let t = 0; t < THAW_ESCAPE_TICKS; t++) {
        scratch.setInput(0, escape[t]);
        scratch.step();
        const tee = scratch.getTee(0);
        if (tee === undefined || !tee.alive || tee.frozen) {
          caught = true;
          break;
        }
      }
      if (!caught) {
        escapable = true;
        break;
      }
    }
    this.thawMemo.set(key, escapable);
    return escapable;
  }

  private stepToInput(step: PlanStep, prev: PlayerInput, enemyDist = 0, hookOk = true, mePos?: Vec2, enPos?: Vec2, enVel?: Vec2, aim = step.aim): PlayerInput {

    const raw = this.raw;
    raw.fill(0);
    raw[0] = step.dir === -1 ? 1 : -1;
    raw[1] = step.dir === 0 ? 1 : -1;
    raw[2] = step.dir === 1 ? 1 : -1;
    raw[3] = step.jump ? 1 : -1;
    raw[4] = step.hook && hookOk ? 1 : -1;

    raw[6] = 1;
    raw[7] = -1;
    raw[8] = Math.cos(aim);
    raw[9] = Math.sin(aim);
    let canSwing = this.cfg.hammerRangePx <= 0 || enemyDist <= this.cfg.hammerRangePx;
    if (canSwing && step.fire !== 0 && this.cfg.gateHammer && mePos !== undefined && enPos !== undefined && enVel !== undefined) {

      raw[5] = -1;
      const dry = decodeAction(raw, prev);
      canSwing = this.hammerWouldHit(mePos, enPos, enVel, dry.targetX, dry.targetY);
    }

    if (canSwing && step.fire !== 0 && this.cfg.noThaw && this.swingTargetFrozen && mePos !== undefined && this.swingTarget !== null && this.swingCollision !== null) {

      if (this.cfg.noThawRope && this.swingRopeOn && step.hook !== 0) canSwing = false;
      else canSwing = !this.thawEscapable(this.swingCollision, this.swingTarget, mePos);
    }

    if (canSwing && step.fire !== 0 && this.frozenBystanders.length > 0 && mePos !== undefined && this.swingCollision !== null) {
      raw[5] = -1;
      const dry = decodeAction(raw, prev);
      for (let i = 0; i < this.frozenBystanders.length; i++) {
        const b = this.frozenBystanders[i];
        const bv = this.frozenBystanderVels[i] ?? NO_VEL;
        const d = vdistance(mePos, b);
        if (d > LAUNCH_REACH_PX * 1.5) continue;
        if (this.hammerWouldHit(mePos, b, bv, dry.targetX, dry.targetY) && launchFlightLandsInHazard(this.swingCollision, b, mePos, Math.max(1, d), bv) === 0) {
          canSwing = false;
          break;
        }
      }
    }

    if (canSwing && step.fire !== 0 && this.spares.length > 0 && mePos !== undefined) {
      raw[5] = -1;
      const dry = decodeAction(raw, prev);
      for (let i = 0; i < this.spares.length; i++) {
        const b = this.spares[i];
        if (vdistance(mePos, b) > LAUNCH_REACH_PX * 1.5) continue;
        if (this.hammerWouldHit(mePos, b, this.spareVels[i] ?? NO_VEL, dry.targetX, dry.targetY)) {
          canSwing = false;
          break;
        }
      }
    }
    raw[5] = step.fire && canSwing ? 1 : -1;
    return decodeAction(raw, prev);
  }

  private executedAim(step: PlanStep, prev: PlayerInput, aim: number): number {
    const raw = this.aimProbe;
    raw.fill(0);
    raw[0] = step.dir === -1 ? 1 : -1;
    raw[1] = step.dir === 0 ? 1 : -1;
    raw[2] = step.dir === 1 ? 1 : -1;
    raw[3] = step.jump ? 1 : -1;
    raw[4] = -1;
    raw[5] = -1;
    raw[6] = 1;
    raw[7] = -1;
    raw[8] = Math.cos(aim);
    raw[9] = Math.sin(aim);
    const probe = decodeAction(raw, prev, this.aimProbeOut);
    return Math.atan2(probe.targetY, probe.targetX);
  }

  private evaluate(
    world: SimWorld,
    selfId: number,
    enemyId: number,
    prev: PlayerInput,
    plan: PlanStep[],
    enemyInput: PlayerInput,
    field: HazardField,
    unfreeze: HazardField,
  ): number {
    let score = 0;
    let input = prev;

    let prevDir = prev.direction;
    const hold = this.cfg.flipHoldTicks;
    const jitter = this.cfg.jitterCost;
    if (hold <= 0 || jitter <= 0) {
      for (const st of plan) {
        if (st.dir !== prevDir) score -= this.cfg.flipCost;
        prevDir = st.dir;
      }
    } else {

      let run = Math.min(hold, Math.max(0, this.heldTicks));
      for (let i = 0; i < plan.length; i++) {
        const st = plan[i];

        const held = this.stepTicks[i];
        if (st.dir !== prevDir) {
          score -= this.cfg.flipCost + jitter * (1 - run / hold);
          run = held;
        } else {
          run = Math.min(hold, run + held);
        }
        prevDir = st.dir;
      }
    }

    let hookWasFlying = false;
    let hookGrabbed = false;

    let heldEnemy = this.cfg.hookReleaseCost > 0 && world.coreOf(selfId)?.hookedPlayer === enemyId;

    const oppRng = new Rng(this.oppSeed);
    let oppInput = enemyInput;
    if (this.trackRollout) {
      this.rolloutEnemyOut = 0;
      this.rolloutSelfOut = 0;
      this.rolloutEnemySealed = false;
    }
    if (this.trackGap) this.rolloutMinGap = EDGE_GAP_PX;

    const enAtStart = world.getTee(enemyId);
    const meAtStart = world.getTee(selfId);
    const enNearAtStart = enAtStart === undefined ? 0 : hazardNearness(field, enAtStart.pos.x, enAtStart.pos.y);
    const drag = {
      prevEnemyNear: enNearAtStart,
      startEnemyNear: enNearAtStart,
      startedInDead: meAtStart !== undefined && inDead(this.dead, meAtStart.pos.x, meAtStart.pos.y),
    };

    const released = emptyInput();
    let prevJumpsLeft = meAtStart?.jumpsLeft ?? 0;
    let groundJumpAt = -1;
    let rolloutTick = 0;
    for (let s = 0; s < plan.length; s++) {
      const meNowForRange = world.readTee(selfId, this.stepMeBuf);
      const enNowForRange = world.readTee(enemyId, this.stepEnBuf);
      const enemyDist = meNowForRange !== undefined && enNowForRange !== undefined ? vdistance(meNowForRange.pos, enNowForRange.pos) : 0;
      let aim = plan[s].aim;
      if (this.cfg.trackAim && meNowForRange !== undefined && enNowForRange !== undefined) {
        aim += Math.atan2(enNowForRange.pos.y - meNowForRange.pos.y, enNowForRange.pos.x - meNowForRange.pos.x);
      }
      this.swingTargetFrozen = enNowForRange?.frozen === true;
      this.swingRopeOn = meNowForRange?.hookedPlayer === enemyId;
      this.swingTarget = enNowForRange ?? null;
      this.swingCollision = world.collision;
      const hookOk = !plan[s].hook || this.hookAlreadyOut(world, selfId) || this.hookAllowed(world, selfId, enemyId, plan[s], input, aim);
      input = this.stepToInput(plan[s], input, enemyDist, hookOk, meNowForRange?.pos, enNowForRange?.pos, enNowForRange?.vel, aim);
      if (this.reactThisPass || this.cfg.opponentModel === "react") oppInput = scriptedAction(world, enemyId, selfId, oppInput, oppRng);
      else if (this.predicted.length > 0) oppInput = this.predicted[Math.min(s, this.predicted.length - 1)];
      for (let t = 0; t < this.stepTicks[s]; t++) {
        rolloutTick++;

        if (this.cfg.releaseDeadHook && input.hook !== 0 && this.hookIsDead(world, selfId)) {
          copyInput(released, input);
          released.hook = 0;
          world.setInput(selfId, released);
        } else world.setInput(selfId, input);
        world.setInput(enemyId, oppInput);
        const events = world.step();
        const meNow = world.readTee(selfId, this.tickMeBuf);
        if (meNow !== undefined) {

          if ((meNow.jumped & 2) === 0 && meNow.jumpsLeft < prevJumpsLeft) groundJumpAt = rolloutTick;
          if (prevJumpsLeft > 0 && meNow.jumpsLeft === 0 && !meNow.frozen) {
            const gap = groundJumpAt < 0 ? AIR_JUMP_MIN_GAP_TICKS : rolloutTick - groundJumpAt;

            if (gap < AIR_JUMP_MIN_GAP_TICKS) score -= this.cfg.airJumpCost * (1 - gap / AIR_JUMP_MIN_GAP_TICKS);
            groundJumpAt = -1;
          }
          prevJumpsLeft = meNow.jumpsLeft;
          if (meNow.hookState === HOOK_FLYING) hookWasFlying = true;
          if (meNow.hookState === HOOK_GRABBED) hookGrabbed = true;

          if (this.cfg.hookReleaseCost > 0) {
            const holds = meNow.hookedPlayer === enemyId;
            if (heldEnemy && !holds) {
              const enNowHeld = world.getTee(enemyId);
              if (enNowHeld !== undefined && enNowHeld.alive && !enNowHeld.frozen) score -= this.cfg.hookReleaseCost;
            }
            heldEnemy = holds;
          }
          if (hookWasFlying && !hookGrabbed && meNow.hookState >= HOOK_RETRACT_START && meNow.hookState < HOOK_FLYING) {
            score -= this.cfg.wastedHook;
            hookWasFlying = false;
          }
          if (meNow.hookState <= 0) {
            hookWasFlying = false;
            hookGrabbed = false;
          }
        }
        if (this.trackRollout) {

          const enNow = world.readTee(enemyId, this.tickEnBuf);
          if (enNow !== undefined && (enNow.frozen || !enNow.alive)) this.rolloutEnemyOut++;
          if (meNow !== undefined && (meNow.frozen || !meNow.alive)) this.rolloutSelfOut++;
        }
        if (this.trackGap && meNow !== undefined) {
          const gap = meNow.frozen || !meNow.alive ? 0 : freezeGapPx(world.collision, meNow.pos.x, meNow.pos.y);
          if (gap < this.rolloutMinGap) this.rolloutMinGap = gap;
        }

        score += scoreTick(world, selfId, enemyId, events, field, unfreeze, this.cfg, drag, this.travel, this.goal, this.dead, this.memory, this.thirds, this.band) * (1 - s / (plan.length * 2));
      }
    }
    if (this.cfg.valueWeight !== 0 && this.valueNet !== null) {

      const tail = 1 - (plan.length - 1) / (plan.length * 2);
      encodeObs(world, selfId, enemyId, this.valueObs);
      score += this.cfg.valueWeight * tail * this.valueNet.forward(this.valueObs)[0];
    }
    if (this.cfg.landingCost > 0) {
      const meEnd = world.getTee(selfId);
      if (meEnd !== undefined && meEnd.alive && !meEnd.frozen) {
        score -= this.cfg.landingCost * flightEndsInHazard(world.collision, meEnd.pos, meEnd.vel);
      }
    }
    if (this.cfg.freezeTailWeight > 0) {

      const tail = 1 - (plan.length - 1) / (plan.length * 2);
      const w = this.cfg.freezeTailWeight * this.cfg.frozenWeight * tail;
      const meEnd = world.getTee(selfId);
      const enEnd = world.getTee(enemyId);
      if (meEnd !== undefined && meEnd.frozen) score -= w * this.cfg.selfFreezeBias * meEnd.freezeTicksLeft;

      const enSealed = this.cfg.sealTicks > 0 && enEnd !== undefined && enEnd.frozen ? restsInFreeze(world.collision, enEnd.pos, enEnd.vel) : 0;
      if (enEnd !== undefined && enEnd.frozen) score += w * (this.cfg.noThaw && enSealed > 0 ? FREEZE_CLOCK_TICKS : enEnd.freezeTicksLeft);
      if (this.cfg.sealTicks > 0) {
        if (meEnd !== undefined && meEnd.frozen) score -= w * this.cfg.selfFreezeBias * this.cfg.sealTicks * restsInFreeze(world.collision, meEnd.pos, meEnd.vel);
        score += w * this.cfg.sealTicks * enSealed;
      }
    }
    if (this.trackRollout) {
      const enEnd = world.getTee(enemyId);
      this.rolloutEnemySealed = enEnd !== undefined && (!enEnd.alive || (enEnd.frozen && restsInFreeze(world.collision, enEnd.pos, enEnd.vel) > 0));
    }
    world.restoreState(this.saved!);
    return score;
  }
}

export function plannerAction(
  planner: Planner,
  world: SimWorld,
  selfId: number,
  enemyId: number,
  prev: PlayerInput,
  enemyInput: PlayerInput = emptyInput(),
): PlayerInput {
  return planner.decide(world, selfId, enemyId, prev, enemyInput);
}

export const PLANNER_WEAPON = WEAPON_HAMMER;
