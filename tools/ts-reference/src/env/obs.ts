import { clamp, vdistance } from "../core/vmath.ts";
import { PHYSICAL_SIZE } from "../core/tuning.ts";
import type { TeeState, WorldView } from "../core/types.ts";
import { WEAPON_GRENADE, WEAPON_GUN, WEAPON_HAMMER, WEAPON_LASER, WEAPON_SHOTGUN } from "../core/types.ts";
import { HOOK_FLYING, HOOK_GRABBED, HOOK_RETRACT_END, HOOK_RETRACT_START } from "../core/characterCore.ts";

const HALF = PHYSICAL_SIZE / 2;
const TERRAIN_RAYS = 20;

const TERRAIN_RANGE = 800;
const TERRAIN_STEP = 16;
const HAZARD_RAYS = 8;
const HAZARD_RANGE = 64;
const HAZARD_STEP = 8;

const GRID_W = 9;
const GRID_H = 7;

const COARSE_W = 9;
const COARSE_H = 7;
const COARSE_STEP = 3;
const TILE_PX = 32;
const ATTACK_COOLDOWN_CAP = 50;

export const OBS_SIZE = 88 + GRID_W * GRID_H + COARSE_W * COARSE_H;

function safe(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return clamp(v, -5, 5);
}

export function isGrounded(view: WorldView, tee: TeeState): boolean {
  return (
    view.collision.isSolid(tee.pos.x + HALF, tee.pos.y + HALF + 5) ||
    view.collision.isSolid(tee.pos.x - HALF, tee.pos.y + HALF + 5)
  );
}

function weaponOneHotIndex(weapon: number): number {
  switch (weapon) {
    case WEAPON_HAMMER:
      return 0;
    case WEAPON_GUN:
      return 1;
    case WEAPON_SHOTGUN:
      return 2;
    case WEAPON_GRENADE:
      return 3;
    case WEAPON_LASER:
      return 4;
    default:
      return -1;
  }
}

function hookOneHotIndex(hookState: number): number {
  if (hookState === HOOK_FLYING) return 1;
  if (hookState === HOOK_GRABBED) return 2;
  if (hookState >= HOOK_RETRACT_START && hookState <= HOOK_RETRACT_END) return 3;
  return 0;
}

export function encodeObs(view: WorldView, selfId: number, enemyId: number, out?: Float64Array): Float64Array {
  const dst = out ?? new Float64Array(OBS_SIZE);
  const self = view.getTee(selfId);
  if (!self) throw new Error(`encodeObs: no tee with id ${selfId}`);
  const enemy = view.getTee(enemyId);
  const collision = view.collision;

  let i = 0;

  dst[i++] = safe(self.vel.x / 10);
  dst[i++] = safe(self.vel.y / 10);
  dst[i++] = isGrounded(view, self) ? 1 : 0;
  dst[i++] = safe(self.jumpsLeft / 2);
  dst[i++] = self.frozen ? 1 : 0;
  dst[i++] = safe(self.freezeTicksLeft / 150);

  const hookOneHot = hookOneHotIndex(self.hookState);
  for (let h = 0; h < 4; h++) dst[i++] = h === hookOneHot ? 1 : 0;

  dst[i++] = safe((self.hookPos.x - self.pos.x) / 380);
  dst[i++] = safe((self.hookPos.y - self.pos.y) / 380);
  dst[i++] = self.hookedPlayer === enemyId ? 1 : 0;

  const weaponOneHot = weaponOneHotIndex(self.activeWeapon);
  for (let w = 0; w < 5; w++) dst[i++] = w === weaponOneHot ? 1 : 0;

  const ownAngle = self.angle / 256;
  dst[i++] = safe(Math.cos(ownAngle));
  dst[i++] = safe(Math.sin(ownAngle));

  const sinceAttack = view.tick - self.attackTick;
  dst[i++] = safe(clamp(sinceAttack, 0, ATTACK_COOLDOWN_CAP) / ATTACK_COOLDOWN_CAP);

  const enemyPos = enemy ? enemy.pos : self.pos;
  const enemyVel = enemy ? enemy.vel : self.vel;
  dst[i++] = safe((enemyPos.x - self.pos.x) / 512);
  dst[i++] = safe((enemyPos.y - self.pos.y) / 512);
  dst[i++] = safe(vdistance(self.pos, enemyPos) / 512);
  dst[i++] = safe((enemyVel.x - self.vel.x) / 10);
  dst[i++] = safe((enemyVel.y - self.vel.y) / 10);
  dst[i++] = enemy && enemy.frozen ? 1 : 0;
  dst[i++] = enemy && enemy.alive ? 1 : 0;
  dst[i++] = enemy && enemy.hookedPlayer === selfId ? 1 : 0;
  if (enemy) {
    const enemyAngle = enemy.angle / 256;
    dst[i++] = safe(Math.cos(enemyAngle));
    dst[i++] = safe(Math.sin(enemyAngle));
  } else {
    dst[i++] = 1;
    dst[i++] = 0;
  }
  const los = enemy ? collision.intersectLine(self.pos, enemy.pos).collision === 0 : false;
  dst[i++] = los ? 1 : 0;

  for (let r = 0; r < TERRAIN_RAYS; r++) {
    const angle = (r / TERRAIN_RAYS) * Math.PI * 2;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    let dist = TERRAIN_RANGE;
    let noHook = 0;
    for (let t = TERRAIN_STEP; t <= TERRAIN_RANGE; t += TERRAIN_STEP) {
      const x = self.pos.x + dx * t;
      const y = self.pos.y + dy * t;
      if (collision.isSolid(x, y)) {
        dist = t;
        noHook = collision.isNoHook(x, y) ? 1 : 0;
        break;
      }
    }
    dst[i++] = safe(dist / TERRAIN_RANGE);
    dst[i++] = noHook;
  }

  for (let r = 0; r < HAZARD_RAYS; r++) {
    const angle = (r / HAZARD_RAYS) * Math.PI * 2;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    let hazard = 0;
    for (let t = HAZARD_STEP; t <= HAZARD_RANGE; t += HAZARD_STEP) {
      const x = self.pos.x + dx * t;
      const y = self.pos.y + dy * t;
      if (collision.isFreeze(x, y) || collision.isDeath(x, y)) {
        hazard = 1;
        break;
      }
    }
    dst[i++] = hazard;
  }

  const projectiles = view.projectiles();
  let bestIdx = -1;
  let bestDist = Infinity;
  let secondIdx = -1;
  let secondDist = Infinity;
  for (let p = 0; p < projectiles.length; p++) {
    const d = vdistance(self.pos, projectiles[p].pos);
    if (d < bestDist) {
      secondIdx = bestIdx;
      secondDist = bestDist;
      bestIdx = p;
      bestDist = d;
    } else if (d < secondDist) {
      secondIdx = p;
      secondDist = d;
    }
  }
  if (bestIdx >= 0) {
    const proj = projectiles[bestIdx];
    dst[i++] = safe((proj.pos.x - self.pos.x) / 512);
    dst[i++] = safe((proj.pos.y - self.pos.y) / 512);
    dst[i++] = safe(proj.vel.x / 20);
    dst[i++] = safe(proj.vel.y / 20);
  } else {
    dst[i++] = 0;
    dst[i++] = 0;
    dst[i++] = 0;
    dst[i++] = 0;
  }
  if (secondIdx >= 0) {
    const proj = projectiles[secondIdx];
    dst[i++] = safe((proj.pos.x - self.pos.x) / 512);
    dst[i++] = safe((proj.pos.y - self.pos.y) / 512);
    dst[i++] = safe(proj.vel.x / 20);
    dst[i++] = safe(proj.vel.y / 20);
  } else {
    dst[i++] = 0;
    dst[i++] = 0;
    dst[i++] = 0;
    dst[i++] = 0;
  }

  const originX = self.pos.x - ((GRID_W - 1) / 2) * TILE_PX;
  const originY = self.pos.y - ((GRID_H - 1) / 2) * TILE_PX;
  for (let gy = 0; gy < GRID_H; gy++) {
    const py = originY + gy * TILE_PX;
    for (let gx = 0; gx < GRID_W; gx++) {
      const px = originX + gx * TILE_PX;
      let v = 0;
      if (collision.isDeath(px, py) || collision.isFreeze(px, py)) v = 1;
      else if (collision.isSolid(px, py) || collision.isNoHook(px, py)) v = 0.5;
      dst[i++] = v;
    }
  }

  const coarseOriginX = self.pos.x - ((COARSE_W - 1) / 2) * COARSE_STEP * TILE_PX;
  const coarseOriginY = self.pos.y - ((COARSE_H - 1) / 2) * COARSE_STEP * TILE_PX;
  for (let gy = 0; gy < COARSE_H; gy++) {
    for (let gx = 0; gx < COARSE_W; gx++) {
      let v = 0;
      for (let sy = 0; sy < COARSE_STEP && v < 1; sy++) {
        for (let sx = 0; sx < COARSE_STEP && v < 1; sx++) {
          const px = coarseOriginX + (gx * COARSE_STEP + sx) * TILE_PX;
          const py = coarseOriginY + (gy * COARSE_STEP + sy) * TILE_PX;
          if (collision.isDeath(px, py) || collision.isFreeze(px, py)) v = 1;
          else if (v < 0.5 && (collision.isSolid(px, py) || collision.isNoHook(px, py))) v = 0.5;
        }
      }
      dst[i++] = v;
    }
  }

  return dst;
}
