import type { Vec2 } from "./vmath.ts";
import { clamp, roundToInt, vadd, vdistance, vlength, vmul, vnormalize, vsub } from "./vmath.ts";
import type { Collision } from "./collision.ts";
import { SERVER_TICK_SPEED, TUNING } from "./tuning.ts";
import type { WorldEvent } from "./types.ts";
import { WEAPON_GRENADE, WEAPON_GUN, WEAPON_LASER, WEAPON_SHOTGUN } from "./types.ts";

export interface EntityWorld {
  readonly tick: number;
  readonly collision: Collision;
  readonly svHit: boolean;
  isAlive(id: number): boolean;
  teePos(id: number): Vec2 | undefined;

  intersectCharacter(pos0: Vec2, pos1: Vec2, radius: number, excludeId: number, onlyId: number): { id: number; pos: Vec2 } | null;

  findCharactersInRadius(pos: Vec2, radius: number): number[];

  applyForce(id: number, force: Vec2): void;
  unfreeze(id: number): void;
}

function calcPos(pos: Vec2, dir: Vec2, curvature: number, speed: number, time: number): Vec2 {
  const t = time * speed;
  return { x: pos.x + dir.x * t, y: pos.y + dir.y * t + (curvature / 10000) * (t * t) };
}

function weaponCurvatureSpeed(type: number): { curvature: number; speed: number } {
  switch (type) {
    case WEAPON_GRENADE:
      return { curvature: TUNING.grenadeCurvature, speed: TUNING.grenadeSpeed };
    case WEAPON_GUN:
      return { curvature: TUNING.gunCurvature, speed: TUNING.gunSpeed };
    default:
      return { curvature: 0, speed: 0 };
  }
}

export function isGameLayerClipped(pos: Vec2, collision: Collision): boolean {
  const tx = Math.trunc(roundToInt(pos.x) / 32);
  const ty = Math.trunc(roundToInt(pos.y) / 32);
  return tx < -200 || tx > collision.width + 200 || ty < -200 || ty > collision.height + 200;
}

export function createExplosion(world: EntityWorld, pos: Vec2, owner: number, weapon: number, noDamage: boolean, events: WorldEvent[]): void {
  events.push({ kind: "explosion", pos: { ...pos }, owner });

  const radius = 135.0;
  const innerRadius = 48.0;
  for (const id of world.findCharactersInRadius(pos, radius)) {
    const p = world.teePos(id);
    if (!p) continue;
    const diff = vsub(p, pos);
    const len = vlength(diff);
    const forceDir = len ? vnormalize(diff) : { x: 0, y: 1 };
    const falloff = 1 - clamp((len - innerRadius) / (radius - innerRadius), 0, 1);
    const strength = TUNING.explosionStrength;
    const dmg = strength * falloff;
    if (Math.trunc(dmg) === 0) continue;

    if (world.svHit || noDamage || owner === id) {
      world.applyForce(id, vmul(forceDir, dmg * 2));
    }
  }
}

export type ProjectileState2 = {
  id: number; type: number; owner: number; posX: number; posY: number; dirX: number; dirY: number;
  startTick: number; lifeSpan: number; explosive: boolean; markedForDestroy: boolean;
};

export type LaserState2 = {
  id: number; owner: number; type: number; posX: number; posY: number; dirX: number; dirY: number;
  energy: number; bounces: number; evalTick: number; zeroEnergyBounceInLastTick: boolean; markedForDestroy: boolean;
};

export class Projectile {
  readonly id: number;
  readonly type: number;
  readonly owner: number;
  readonly pos: Vec2;
  readonly dir: Vec2;
  readonly startTick: number;
  readonly explosive: boolean;
  readonly vel: Vec2;
  lifeSpan: number;
  markedForDestroy = false;

  constructor(id: number, type: number, owner: number, pos: Vec2, dir: Vec2, startTick: number, lifeSpan: number, explosive: boolean) {
    this.id = id;
    this.type = type;
    this.owner = owner;
    this.pos = { ...pos };
    this.dir = { ...dir };
    this.startTick = startTick;
    this.lifeSpan = lifeSpan;
    this.explosive = explosive;
    const { speed } = weaponCurvatureSpeed(type);
    this.vel = vmul(dir, speed);
  }

  saveState(): ProjectileState2 {
    return { id: this.id, type: this.type, owner: this.owner, posX: this.pos.x, posY: this.pos.y, dirX: this.dir.x, dirY: this.dir.y, startTick: this.startTick, lifeSpan: this.lifeSpan, explosive: this.explosive, markedForDestroy: this.markedForDestroy };
  }

  static fromState(_world: unknown, st: ProjectileState2): Projectile {
    const p = new Projectile(st.id, st.type, st.owner, { x: st.posX, y: st.posY }, { x: st.dirX, y: st.dirY }, st.startTick, st.lifeSpan, st.explosive);
    p.markedForDestroy = st.markedForDestroy;
    return p;
  }

  posAtTick(tick: number): Vec2 {
    const { curvature, speed } = weaponCurvatureSpeed(this.type);
    const time = (tick - this.startTick) / SERVER_TICK_SPEED;
    return calcPos(this.pos, this.dir, curvature, speed, time);
  }

  tick(world: EntityWorld, events: WorldEvent[]): void {
    const prevPos = this.posAtTick(world.tick - 1);
    const curPos = this.posAtTick(world.tick);

    const { collision, outPos } = world.collision.intersectLine(prevPos, curPos);
    const collide = collision !== 0;

    let targetId = -1;
    if (world.svHit) {
      const hit = world.intersectCharacter(prevPos, outPos, 6.0, this.owner, -1);
      if (hit) targetId = hit.id;
    }

    if (this.lifeSpan > -1) this.lifeSpan--;

    if (this.owner >= 0 && !world.isAlive(this.owner)) {
      this.markedForDestroy = true;
      return;
    }

    const outOfBounds = isGameLayerClipped(curPos, world.collision);
    if (targetId !== -1 || collide || outOfBounds) {
      if (this.explosive) {
        createExplosion(world, outPos, this.owner, this.type, this.owner === -1, events);
        this.markedForDestroy = true;
        return;
      }

      this.markedForDestroy = true;
      return;
    }

    if (this.lifeSpan === -1) {
      if (this.explosive) {
        createExplosion(world, outPos, this.owner, this.type, this.owner === -1, events);
      }
      this.markedForDestroy = true;
    }
  }
}

export class Laser {
  readonly id: number;
  readonly owner: number;
  readonly type: number;
  pos: Vec2;
  dir: Vec2;
  energy: number;
  bounces = 0;
  evalTick = 0;
  zeroEnergyBounceInLastTick = false;
  markedForDestroy = false;

  saveState(): LaserState2 {
    return { id: this.id, owner: this.owner, type: this.type, posX: this.pos.x, posY: this.pos.y, dirX: this.dir.x, dirY: this.dir.y, energy: this.energy, bounces: this.bounces, evalTick: this.evalTick, zeroEnergyBounceInLastTick: this.zeroEnergyBounceInLastTick, markedForDestroy: this.markedForDestroy };
  }

  static fromState(_world: unknown, st: LaserState2): Laser {
    const l = new Laser(st.id, st.owner, st.type, { x: st.posX, y: st.posY }, { x: st.dirX, y: st.dirY }, st.energy);
    l.pos = { x: st.posX, y: st.posY };
    l.dir = { x: st.dirX, y: st.dirY };
    l.energy = st.energy;
    l.bounces = st.bounces;
    l.evalTick = st.evalTick;
    l.zeroEnergyBounceInLastTick = st.zeroEnergyBounceInLastTick;
    l.markedForDestroy = st.markedForDestroy;
    return l;
  }

  constructor(id: number, owner: number, type: number, pos: Vec2, dir: Vec2, startEnergy: number) {
    this.id = id;
    this.owner = owner;
    this.type = type;
    this.pos = { ...pos };
    this.dir = { ...dir };
    this.energy = startEnergy;
  }

  private hitCharacter(world: EntityWorld, from: Vec2, to: Vec2, events: WorldEvent[]): boolean {
    const dontHitSelf = this.bounces === 0;
    const excludeId = dontHitSelf ? this.owner : -1;
    const onlyId = world.svHit ? -1 : this.owner;
    const hit = world.intersectCharacter(from, to, 0, excludeId, onlyId);
    if (!hit) return false;

    this.pos = { ...hit.pos };
    this.energy = -1;

    if (this.type === WEAPON_SHOTGUN) {
      const strength = TUNING.shotgunStrength;
      const hitPos = world.teePos(hit.id);
      if (hitPos && (hitPos.x !== from.x || hitPos.y !== from.y)) {
        world.applyForce(hit.id, vmul(vnormalize(vsub(from, hitPos)), strength));
      }

    } else if (this.type === WEAPON_LASER) {
      world.unfreeze(hit.id);
    }

    events.push({ kind: "laserHit", from: this.owner, to: hit.id, weapon: this.type });
    return true;
  }

  doBounce(world: EntityWorld, events: WorldEvent[]): void {
    this.evalTick = world.tick;
    if (this.energy < 0) {
      this.markedForDestroy = true;
      return;
    }

    const rayStart = { ...this.pos };
    const to = vadd(rayStart, vmul(this.dir, this.energy));
    const { collision, outBeforePos } = world.collision.intersectLine(rayStart, to);

    if (collision !== 0) {

      const hitTo = outBeforePos;
      if (!this.hitCharacter(world, rayStart, hitTo, events)) {
        this.pos = { ...hitTo };

        const tempPos: Vec2 = { ...this.pos };
        const tempVel: Vec2 = vmul(this.dir, 4.0);
        world.collision.movePoint(tempPos, tempVel, 1.0, null);
        this.pos = tempPos;
        this.dir = vnormalize(tempVel);

        const dist = vdistance(rayStart, this.pos);
        if (dist === 0 && this.zeroEnergyBounceInLastTick) {
          this.energy = -1;
        } else {
          this.energy -= dist + TUNING.laserBounceCost;
        }
        this.zeroEnergyBounceInLastTick = dist === 0;

        this.bounces++;
        if (this.bounces > TUNING.laserBounceNum) this.energy = -1;
      }
    } else {
      if (!this.hitCharacter(world, rayStart, to, events)) {
        this.pos = { ...to };
        this.energy = -1;
      }
    }
  }

  tick(world: EntityWorld, events: WorldEvent[]): void {
    const delayTicks = (SERVER_TICK_SPEED * TUNING.laserBounceDelay) / 1000.0;
    if (world.tick - this.evalTick > delayTicks) {
      this.doBounce(world, events);
    }
  }
}
