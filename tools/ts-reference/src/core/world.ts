import type { Vec2 } from "./vmath.ts";
import { vadd, vdistance, vmul, vnormalize, vsub, closestPointOnLineOrNull } from "./vmath.ts";
import type { Collision } from "./collision.ts";
import { clampVel } from "./collision.ts";
import { CANTMOVE_DOWN, PHYSICAL_SIZE, SERVER_TICK_SPEED, TILE_DEATH, TILE_DFREEZE, TILE_DUNFREEZE, TILE_FREEZE, TILE_LFREEZE, TILE_TELECHECK, TILE_TELECHECKIN, TILE_TELECHECKINEVIL, TILE_TELEIN, TILE_TELEINEVIL, TILE_UNFREEZE, TUNING } from "./tuning.ts";
import { COREEVENT_HOOK_RETRACT, CharacterCore, HOOK_RETRACTED } from "./characterCore.ts";
import type { CoreWorld } from "./characterCore.ts";
import type { PlayerInput, ProjectileState, TeeState, WorldEvent, WorldView } from "./types.ts";
import { NUM_WEAPONS, WEAPON_GRENADE, WEAPON_GUN, WEAPON_HAMMER, WEAPON_LASER, WEAPON_SHOTGUN, blankTeeState, emptyInput } from "./types.ts";
import { CHARACTERFLAG_COLLISION_DISABLED, CHARACTERFLAG_ENDLESS_HOOK, CHARACTERFLAG_HOOK_HIT_DISABLED, CHARACTERFLAG_SOLO } from "./types.ts";
import type { EntityWorld, LaserState2, ProjectileState2 } from "./projectile.ts";
import { Laser, Projectile, isGameLayerClipped } from "./projectile.ts";

const FREEZE_SECONDS = 3;

const INPUT_STATE_MASK = 0x3f;

function countPresses(prev: number, cur: number): number {
  prev &= INPUT_STATE_MASK;
  cur &= INPUT_STATE_MASK;
  let i = prev;
  let presses = 0;
  while (i !== cur) {
    i = (i + 1) & INPUT_STATE_MASK;
    if (i & 1) presses++;
  }
  return presses;
}

function weaponFireDelayMs(weapon: number): number {
  switch (weapon) {
    case WEAPON_HAMMER:
      return TUNING.hammerFireDelay;
    case WEAPON_GUN:
      return TUNING.gunFireDelay;
    case WEAPON_SHOTGUN:
      return TUNING.shotgunFireDelay;
    case WEAPON_GRENADE:
      return TUNING.grenadeFireDelay;
    case WEAPON_LASER:
      return TUNING.laserFireDelay;
    default:
      return 0;
  }
}

function fireDelayTicks(weapon: number): number {
  return Math.trunc((weaponFireDelayMs(weapon) * SERVER_TICK_SPEED) / 1000);
}

type WeaponSlot = { got: boolean; ammo: number };

function hasAmmo(slot: WeaponSlot): boolean {
  return slot.ammo !== 0;
}

function defaultWeapons(infiniteAmmo: boolean, allWeapons: boolean): WeaponSlot[] {
  const slots: WeaponSlot[] = [];
  for (let i = 0; i < NUM_WEAPONS; i++) slots.push({ got: false, ammo: 0 });
  const ammo = infiniteAmmo ? -1 : 10;
  slots[WEAPON_HAMMER] = { got: true, ammo: -1 };
  slots[WEAPON_GUN] = { got: true, ammo };
  if (allWeapons) {
    slots[WEAPON_SHOTGUN] = { got: true, ammo };
    slots[WEAPON_GRENADE] = { got: true, ammo };
    slots[WEAPON_LASER] = { got: true, ammo };
  }
  return slots;
}

function freezeTee(currentTick: number, rec: TeeRecord, seconds: number): boolean {
  if (seconds <= 0) return false;
  if (rec.freezeTicksLeft > seconds * SERVER_TICK_SPEED) return false;
  if (rec.freezeTicksLeft === 0 || rec.core.freezeStart < currentTick - SERVER_TICK_SPEED) {
    rec.freezeTicksLeft = seconds * SERVER_TICK_SPEED;
    rec.core.freezeStart = currentTick;
    return true;
  }
  return false;
}

function unfreezeTee(rec: TeeRecord): boolean {
  if (rec.freezeTicksLeft > 0) {
    rec.freezeTicksLeft = 0;
    rec.core.freezeStart = 0;

    rec.frozenLastTick = true;
    return true;
  }
  return false;
}

function applyJumpRules(core: CharacterCore): void {
  if (core.jumps === -1) core.jumped |= 2;
  else if (core.jumps === 0) core.jumped |= 2;
  else if (core.jumps === 1 && core.jumped > 0) core.jumped |= 2;
  else if (core.jumpedTotal < core.jumps - 1 && core.jumped > 1) core.jumped = 1;
}

function computeJumpsLeft(core: CharacterCore, collision: Collision): number {
  if (core.jumps <= 0) return 0;
  const grounded =
    collision.isSolid(core.pos.x + PHYSICAL_SIZE / 2, core.pos.y + PHYSICAL_SIZE / 2 + 5) ||
    collision.isSolid(core.pos.x - PHYSICAL_SIZE / 2, core.pos.y + PHYSICAL_SIZE / 2 + 5);
  if (grounded) return core.jumps;
  if (core.jumped & 2) return 0;
  return Math.max(0, core.jumps - 1 - core.jumpedTotal);
}

class TeeRecord {
  readonly core: CharacterCore;
  alive = true;
  spawnPos: Vec2;
  weapons: WeaponSlot[];
  queuedWeapon = -1;
  reloadTimer = 0;

  attackTick = -1000;
  freezeTicksLeft = 0;

  frozenLastTick = false;

  deepFrozen = false;

  teleCheckpoint = 0;

  moveRestrictions = 0;
  input: PlayerInput = emptyInput();
  prevInputForEdge: PlayerInput = emptyInput();

  frozenInput: PlayerInput = emptyInput();

  pendingCoreInput: PlayerInput | null = null;

  prevPos: Vec2;
  respawnAtTick: number | null = null;

  constructor(core: CharacterCore, spawnPos: Vec2, infiniteAmmo: boolean, allWeapons: boolean) {
    this.core = core;
    this.spawnPos = { ...spawnPos };
    this.prevPos = { ...spawnPos };
    this.weapons = defaultWeapons(infiniteAmmo, allWeapons);
  }
}

export type CoreState = {
  posX: number; posY: number; velX: number; velY: number;
  hookPosX: number; hookPosY: number; hookDirX: number; hookDirY: number;
  hookTeleBaseX: number; hookTeleBaseY: number;
  hookTick: number; hookState: number; hookedPlayer: number; attachedPlayers: number[];
  activeWeapon: number; newHook: boolean; jumped: number; jumpedTotal: number; jumps: number;
  direction: number; angle: number; triggeredEvents: number; colliding: number; leftWall: boolean;
  freezeStart: number; freezeEnd: number; isInFreeze: boolean; moveRestrictions: number;
};

export type TeeStateSnapshot = {
  core: CoreState;
  alive: boolean;
  freezeTicksLeft: number;
  frozenLastTick: boolean;
  deepFrozen: boolean;
  teleCheckpoint: number;
  moveRestrictions: number;
  reloadTimer: number;
  attackTick: number;
  queuedWeapon: number;
  input: PlayerInput;
  prevInputForEdge: PlayerInput;
  prevPos: Vec2;
  spawnPos: Vec2;
  respawnAtTick: number | null;
  weapons: { got: boolean; ammo: number }[];
};

export type SimState = {
  tick: number;
  nextEntityId: number;
  tees: Map<number, TeeStateSnapshot>;
  projectiles: ProjectileState2[];
  lasers: LaserState2[];
};

function copyInput(from: PlayerInput, to: PlayerInput): void {
  to.direction = from.direction;
  to.targetX = from.targetX;
  to.targetY = from.targetY;
  to.jump = from.jump;
  to.fire = from.fire;
  to.hook = from.hook;
  to.playerFlags = from.playerFlags;
  to.wantedWeapon = from.wantedWeapon;
  to.nextWeapon = from.nextWeapon;
  to.prevWeapon = from.prevWeapon;
}

export class SimWorld implements WorldView, CoreWorld, EntityWorld {
  readonly collision: Collision;
  tick = 0;
  readonly svHit: boolean;
  private readonly respawnDelayTicks: number;
  private readonly infiniteAmmo: boolean;
  private readonly allWeapons: boolean;

  readonly noWeakHook: boolean;
  private readonly tees = new Map<number, TeeRecord>();

  private order: TeeRecord[] = [];

  private byId: TeeRecord[] = [];
  private readonly tileIndices: number[] = [];
  private nextEntityId = 1;
  private projectilesList: Projectile[] = [];
  private lasersList: Laser[] = [];

  constructor(collision: Collision, options?: { respawnDelayTicks?: number; infiniteAmmo?: boolean; svHit?: boolean; allWeapons?: boolean; noWeakHook?: boolean }) {
    this.collision = collision;
    this.respawnDelayTicks = options?.respawnDelayTicks ?? 0;
    this.infiniteAmmo = options?.infiniteAmmo ?? true;
    this.allWeapons = options?.allWeapons ?? false;
    this.svHit = options?.svHit ?? true;
    this.noWeakHook = options?.noWeakHook ?? collision.noWeakHook;
  }

  addTee(id: number, spawnPos: Vec2): void {
    const core = new CharacterCore(id, this.collision, this);
    core.reset();
    core.pos = { ...spawnPos };
    core.activeWeapon = WEAPON_GUN;
    const rec = new TeeRecord(core, spawnPos, this.infiniteAmmo, this.allWeapons);
    this.tees.set(id, rec);
    this.order.unshift(rec);
    this.byId.push(rec);
    this.byId.sort((a, b) => a.core.id - b.core.id);
  }

  removeTee(id: number): void {
    this.tees.delete(id);
    this.order = this.order.filter((rec) => rec.core.id !== id);
    this.byId = this.byId.filter((rec) => rec.core.id !== id);
  }

  setInput(id: number, input: PlayerInput): void {
    const rec = this.tees.get(id);
    if (rec === undefined) return;

    const dst = rec.input;
    dst.direction = input.direction;
    dst.targetX = input.targetX;
    dst.targetY = input.targetY;

    if (dst.targetX === 0 && dst.targetY === 0) dst.targetY = -1;
    dst.jump = input.jump;
    dst.fire = input.fire;
    dst.hook = input.hook;
    dst.playerFlags = input.playerFlags;
    dst.wantedWeapon = input.wantedWeapon;
    dst.nextWeapon = input.nextWeapon;
    dst.prevWeapon = input.prevWeapon;
  }

  saveState(into?: SimState): SimState {
    const st: SimState = into ?? { tick: 0, tees: new Map(), projectiles: [], lasers: [], nextEntityId: 1 };
    st.tick = this.tick;
    st.nextEntityId = this.nextEntityId;
    for (const [id, rec] of this.tees) {
      let t = st.tees.get(id);
      if (t === undefined) {
        t = { core: {} as CoreState, alive: true, freezeTicksLeft: 0, frozenLastTick: false, deepFrozen: false, teleCheckpoint: 0, moveRestrictions: 0, reloadTimer: 0, attackTick: 0, queuedWeapon: -1, input: emptyInput(), prevInputForEdge: emptyInput(), prevPos: { x: 0, y: 0 }, spawnPos: { x: 0, y: 0 }, respawnAtTick: null, weapons: [] };
        st.tees.set(id, t);
      }
      const c = rec.core;
      t.core = {
        posX: c.pos.x, posY: c.pos.y, velX: c.vel.x, velY: c.vel.y,
        hookPosX: c.hookPos.x, hookPosY: c.hookPos.y, hookDirX: c.hookDir.x, hookDirY: c.hookDir.y,
        hookTeleBaseX: c.hookTeleBase.x, hookTeleBaseY: c.hookTeleBase.y,
        hookTick: c.hookTick, hookState: c.hookState, hookedPlayer: c.hookedPlayer,
        attachedPlayers: Array.from(c.attachedPlayers),
        activeWeapon: c.activeWeapon, newHook: c.newHook, jumped: c.jumped, jumpedTotal: c.jumpedTotal,
        jumps: c.jumps, direction: c.direction, angle: c.angle, triggeredEvents: c.triggeredEvents,
        colliding: c.colliding, leftWall: c.leftWall, freezeStart: c.freezeStart, freezeEnd: c.freezeEnd,
        isInFreeze: c.isInFreeze, moveRestrictions: c.moveRestrictions,
      };
      t.alive = rec.alive;
      t.freezeTicksLeft = rec.freezeTicksLeft;
      t.frozenLastTick = rec.frozenLastTick;
      t.deepFrozen = rec.deepFrozen;
      t.teleCheckpoint = rec.teleCheckpoint;
      t.moveRestrictions = rec.moveRestrictions;
      t.reloadTimer = rec.reloadTimer;
      t.attackTick = rec.attackTick;
      t.queuedWeapon = rec.queuedWeapon;
      t.respawnAtTick = rec.respawnAtTick;
      copyInput(rec.input, t.input);
      copyInput(rec.prevInputForEdge, t.prevInputForEdge);
      t.prevPos.x = rec.prevPos.x;
      t.prevPos.y = rec.prevPos.y;
      t.spawnPos.x = rec.spawnPos.x;
      t.spawnPos.y = rec.spawnPos.y;
      t.weapons = rec.weapons.map((w) => ({ got: w.got, ammo: w.ammo }));
    }
    st.projectiles = this.projectilesList.map((pr) => pr.saveState());
    st.lasers = this.lasersList.map((l) => l.saveState());
    return st;
  }

  restoreState(st: SimState): void {
    this.tick = st.tick;
    this.nextEntityId = st.nextEntityId;
    for (const [id, t] of st.tees) {
      const rec = this.tees.get(id);
      if (rec === undefined) continue;
      const c = rec.core;
      const sc = t.core;
      c.pos.x = sc.posX; c.pos.y = sc.posY; c.vel.x = sc.velX; c.vel.y = sc.velY;
      c.hookPos.x = sc.hookPosX; c.hookPos.y = sc.hookPosY;
      c.hookDir.x = sc.hookDirX; c.hookDir.y = sc.hookDirY;
      c.hookTeleBase.x = sc.hookTeleBaseX; c.hookTeleBase.y = sc.hookTeleBaseY;
      c.hookTick = sc.hookTick; c.hookState = sc.hookState; c.hookedPlayer = sc.hookedPlayer;
      c.attachedPlayers.clear();
      for (const a of sc.attachedPlayers) c.attachedPlayers.add(a);
      c.activeWeapon = sc.activeWeapon; c.newHook = sc.newHook; c.jumped = sc.jumped;
      c.jumpedTotal = sc.jumpedTotal; c.jumps = sc.jumps; c.direction = sc.direction;
      c.angle = sc.angle; c.triggeredEvents = sc.triggeredEvents; c.colliding = sc.colliding;
      c.leftWall = sc.leftWall; c.freezeStart = sc.freezeStart; c.freezeEnd = sc.freezeEnd;
      c.isInFreeze = sc.isInFreeze;
      c.moveRestrictions = sc.moveRestrictions;
      rec.alive = t.alive;
      rec.freezeTicksLeft = t.freezeTicksLeft;
      rec.frozenLastTick = t.frozenLastTick;
      rec.deepFrozen = t.deepFrozen;
      rec.teleCheckpoint = t.teleCheckpoint;
      rec.moveRestrictions = t.moveRestrictions;
      rec.reloadTimer = t.reloadTimer;
      rec.attackTick = t.attackTick;
      rec.queuedWeapon = t.queuedWeapon;
      rec.respawnAtTick = t.respawnAtTick;
      copyInput(t.input, rec.input);
      copyInput(t.prevInputForEdge, rec.prevInputForEdge);
      rec.prevPos.x = t.prevPos.x;
      rec.prevPos.y = t.prevPos.y;
      rec.spawnPos.x = t.spawnPos.x;
      rec.spawnPos.y = t.spawnPos.y;
      for (let i = 0; i < rec.weapons.length && i < t.weapons.length; i++) {
        rec.weapons[i].got = t.weapons[i].got;
        rec.weapons[i].ammo = t.weapons[i].ammo;
      }
      rec.pendingCoreInput = null;
    }
    this.projectilesList = st.projectiles.map((ps) => Projectile.fromState(this, ps));
    this.lasersList = st.lasers.map((ls) => Laser.fromState(this, ls));
  }

  applyTeeState(id: number, st: TeeState): void {
    const rec = this.tees.get(id);
    if (rec === undefined) return;
    const c = rec.core;
    c.pos.x = st.pos.x;
    c.pos.y = st.pos.y;
    c.vel.x = st.vel.x;
    c.vel.y = st.vel.y;
    c.hookState = st.hookState;
    c.hookPos.x = st.hookPos.x;
    c.hookPos.y = st.hookPos.y;
    c.hookDir.x = st.hookDir.x;
    c.hookDir.y = st.hookDir.y;
    c.hookedPlayer = st.hookedPlayer;

    if (st.hookTick !== undefined) c.hookTick = st.hookTick;
    c.jumped = st.jumped;

    c.jumps = st.jumps ?? 2;
    c.jumpedTotal = st.jumpedTotal ?? Math.max(0, c.jumps - st.jumpsLeft - (st.jumped & 1 ? 1 : 0));

    if (st.ddnetFlags !== undefined) {
      c.solo = (st.ddnetFlags & CHARACTERFLAG_SOLO) !== 0;
      c.collisionDisabled = (st.ddnetFlags & CHARACTERFLAG_COLLISION_DISABLED) !== 0;
      c.hookHitDisabled = (st.ddnetFlags & CHARACTERFLAG_HOOK_HIT_DISABLED) !== 0;
      c.endlessHook = (st.ddnetFlags & CHARACTERFLAG_ENDLESS_HOOK) !== 0;
    }
    c.direction = st.direction;
    c.angle = st.angle;
    c.activeWeapon = st.activeWeapon;
    rec.alive = st.alive;

    rec.attackTick = st.sinceAttack !== undefined ? this.tick - st.sinceAttack : st.attackTick;
    rec.freezeTicksLeft = st.frozen ? Math.max(1, st.freezeTicksLeft) : 0;

    if (st.reloadTicks !== undefined) rec.reloadTimer = st.reloadTicks;
    if (st.frozen && st.frozenFor !== undefined) c.freezeStart = this.tick - st.frozenFor;
    if (st.deepFrozen !== undefined) rec.deepFrozen = st.deepFrozen;

    rec.prevPos.x = st.pos.x - st.vel.x;
    rec.prevPos.y = st.pos.y - st.vel.y;

    c.moveRestrictions = this.collision.getMoveRestrictions(rec.prevPos);
    rec.moveRestrictions = this.collision.getMoveRestrictions(rec.prevPos, 18, this.collision.getMapIndex(rec.prevPos));
    rec.pendingCoreInput = null;
  }

  setHeldInput(id: number, input: PlayerInput): void {
    const rec = this.tees.get(id);
    if (rec === undefined) return;
    copyInput(input, rec.input);
    copyInput(input, rec.prevInputForEdge);
    if (rec.input.targetX === 0 && rec.input.targetY === 0) rec.input.targetY = -1;
    if (rec.prevInputForEdge.targetX === 0 && rec.prevInputForEdge.targetY === 0) rec.prevInputForEdge.targetY = -1;
  }

  lasersForTest(): readonly Laser[] {
    return this.lasersList;
  }

  coreOf(id: number): CharacterCore | undefined {
    return this.tees.get(id)?.core;
  }

  reset(): void {
    for (const rec of this.tees.values()) {
      rec.core.reset();
      rec.core.pos = { ...rec.spawnPos };
      rec.prevPos = { ...rec.spawnPos };
      rec.core.activeWeapon = WEAPON_GUN;
      rec.alive = true;
      rec.weapons = defaultWeapons(this.infiniteAmmo, this.allWeapons);
      rec.queuedWeapon = -1;
      rec.reloadTimer = 0;
      rec.attackTick = 0;
      rec.freezeTicksLeft = 0;
      rec.frozenLastTick = false;
      rec.deepFrozen = false;
      rec.teleCheckpoint = 0;
      rec.moveRestrictions = 0;
      rec.input = emptyInput();
      rec.prevInputForEdge = emptyInput();
      rec.respawnAtTick = null;
    }
    this.projectilesList = [];
    this.lasersList = [];
    this.tick = 0;
  }

  getTee(id: number): TeeState | undefined {
    const rec = this.tees.get(id);
    return rec ? this.snapshot(rec) : undefined;
  }

  allTees(): TeeState[] {
    const out: TeeState[] = [];
    for (const rec of this.tees.values()) out.push(this.snapshot(rec));
    return out;
  }

  projectiles(): ProjectileState[] {
    const out: ProjectileState[] = [];
    for (const p of this.projectilesList) {
      out.push({
        id: p.id,
        type: p.type,
        owner: p.owner,
        pos: p.posAtTick(this.tick),
        vel: { ...p.vel },
        dir: { ...p.dir },
        startTick: p.startTick,
        spawnPos: { ...p.pos },
      });
    }
    return out;
  }

  private snapshot(rec: TeeRecord): TeeState {
    return this.fill(rec, blankTeeState());
  }

  private fill(rec: TeeRecord, out: TeeState): TeeState {
    const c = rec.core;
    out.id = c.id;
    out.alive = rec.alive;
    out.pos.x = c.pos.x;
    out.pos.y = c.pos.y;
    out.vel.x = c.vel.x;
    out.vel.y = c.vel.y;
    out.hookState = c.hookState;
    out.hookPos.x = c.hookPos.x;
    out.hookPos.y = c.hookPos.y;
    out.hookDir.x = c.hookDir.x;
    out.hookDir.y = c.hookDir.y;
    out.hookedPlayer = c.hookedPlayer;
    out.jumped = c.jumped;
    out.jumpsLeft = computeJumpsLeft(c, this.collision);
    out.direction = c.direction;
    out.angle = c.angle;
    out.activeWeapon = c.activeWeapon;
    out.frozen = rec.freezeTicksLeft > 0;
    out.freezeTicksLeft = rec.freezeTicksLeft;
    out.attackTick = rec.attackTick;
    out.hookTick = c.hookTick;
    out.jumpedTotal = c.jumpedTotal;
    out.reloadTicks = rec.reloadTimer;
    out.frozenFor = rec.freezeTicksLeft > 0 ? this.tick - c.freezeStart : undefined;
    out.deepFrozen = rec.deepFrozen;
    return out;
  }

  readTee(id: number, out: TeeState): TeeState | undefined {
    const rec = this.tees.get(id);
    return rec === undefined ? undefined : this.fill(rec, out);
  }

  allCores(): Iterable<CharacterCore> {
    const cores: CharacterCore[] = [];
    for (const rec of this.byId) if (rec.alive) cores.push(rec.core);
    return cores;
  }

  coreById(id: number): CharacterCore | undefined {
    const rec = this.tees.get(id);
    return rec && rec.alive ? rec.core : undefined;
  }

  isAlive(id: number): boolean {
    return this.tees.get(id)?.alive ?? false;
  }

  teePos(id: number): Vec2 | undefined {
    const rec = this.tees.get(id);
    return rec && rec.alive ? rec.core.pos : undefined;
  }

  intersectCharacter(pos0: Vec2, pos1: Vec2, radius: number, excludeId: number, onlyId: number): { id: number; pos: Vec2 } | null {
    let closestLen = vdistance(pos0, pos1) * 100;
    let found: { id: number; pos: Vec2 } | null = null;
    for (const rec of this.order) {
      if (!rec.alive) continue;
      if (rec.core.id === excludeId) continue;
      if (onlyId !== -1 && rec.core.id !== onlyId) continue;

      const ip = closestPointOnLineOrNull(pos0, pos1, rec.core.pos);
      if (ip === null) continue;
      const len = vdistance(rec.core.pos, ip);
      if (len < PHYSICAL_SIZE + radius) {
        const len2 = vdistance(pos0, ip);
        if (len2 < closestLen) {
          closestLen = len2;
          found = { id: rec.core.id, pos: ip };
        }
      }
    }
    return found;
  }

  findCharactersInRadius(pos: Vec2, radius: number): number[] {
    const out: number[] = [];
    for (const rec of this.order) {
      if (!rec.alive) continue;
      if (vdistance(rec.core.pos, pos) < radius + PHYSICAL_SIZE) out.push(rec.core.id);
    }
    return out;
  }

  applyForce(id: number, force: Vec2): void {
    const rec = this.tees.get(id);
    if (!rec || !rec.alive) return;
    rec.core.vel = clampVel(rec.moveRestrictions, vadd(rec.core.vel, force));
  }

  unfreeze(id: number): void {
    const rec = this.tees.get(id);
    if (rec) unfreezeTee(rec);
  }

  setGrenades(list: readonly { owner: number; spawnPos: Vec2; dir: Vec2; ageTicks: number }[]): void {
    this.projectilesList = [];
    const life = Math.trunc(SERVER_TICK_SPEED * TUNING.grenadeLifetime);
    for (const g of list) {
      const owner = this.tees.has(g.owner) ? g.owner : -1;
      this.projectilesList.push(new Projectile(this.nextEntityId++, WEAPON_GRENADE, owner, g.spawnPos, g.dir, this.tick - g.ageTicks, life - g.ageTicks, true));
    }
  }

  private spawnProjectile(type: number, owner: number, pos: Vec2, dir: Vec2, lifeSpan: number, explosive: boolean): void {
    this.projectilesList.push(new Projectile(this.nextEntityId++, type, owner, pos, dir, this.tick, lifeSpan, explosive));
  }

  private spawnLaser(owner: number, type: number, pos: Vec2, dir: Vec2, energy: number, events: WorldEvent[]): void {
    const laser = new Laser(this.nextEntityId++, owner, type, pos, dir, energy);
    laser.doBounce(this, events);
    this.lasersList.push(laser);
  }

  private doWeaponSwitch(rec: TeeRecord): void {
    if (rec.reloadTimer !== 0 || rec.queuedWeapon === -1) return;
    if (!rec.weapons[rec.queuedWeapon].got) return;
    this.setWeapon(rec, rec.queuedWeapon);
  }

  private setWeapon(rec: TeeRecord, weapon: number): void {
    if (weapon === rec.core.activeWeapon) return;
    rec.queuedWeapon = -1;
    rec.core.activeWeapon = weapon;
    if (rec.core.activeWeapon < 0 || rec.core.activeWeapon >= NUM_WEAPONS) rec.core.activeWeapon = 0;
  }

  private handleWeaponSwitch(rec: TeeRecord, input: PlayerInput): void {
    let wanted = rec.core.activeWeapon;
    if (rec.queuedWeapon !== -1) wanted = rec.queuedWeapon;

    let anything = false;
    for (let i = 0; i < NUM_WEAPONS - 1; i++) if (rec.weapons[i].got) anything = true;
    if (!anything) return;

    let next = countPresses(rec.prevInputForEdge.nextWeapon, input.nextWeapon);
    let prev = countPresses(rec.prevInputForEdge.prevWeapon, input.prevWeapon);

    if (next < 128) {
      while (next) {
        wanted = (wanted + 1) % NUM_WEAPONS;
        if (rec.weapons[wanted].got) next--;
      }
    }
    if (prev < 128) {
      while (prev) {
        wanted = wanted - 1 < 0 ? NUM_WEAPONS - 1 : wanted - 1;
        if (rec.weapons[wanted].got) prev--;
      }
    }

    if (input.wantedWeapon) wanted = input.wantedWeapon - 1;

    if (wanted >= 0 && wanted < NUM_WEAPONS && wanted !== rec.core.activeWeapon && rec.weapons[wanted].got) {
      rec.queuedWeapon = wanted;
    }

    this.doWeaponSwitch(rec);
  }

  private fireWeapon(rec: TeeRecord, input: PlayerInput, events: WorldEvent[]): void {
    if (rec.reloadTimer !== 0) return;

    this.doWeaponSwitch(rec);
    const dir = vnormalize({ x: input.targetX, y: input.targetY });

    const fullAuto = rec.core.activeWeapon === WEAPON_GRENADE || rec.core.activeWeapon === WEAPON_SHOTGUN || rec.core.activeWeapon === WEAPON_LASER || rec.frozenLastTick;

    let willFire = countPresses(rec.prevInputForEdge.fire, input.fire) > 0;
    if (fullAuto && (input.fire & 1) !== 0 && rec.core.activeWeapon >= 0 && hasAmmo(rec.weapons[rec.core.activeWeapon])) {
      willFire = true;
    }
    if (!willFire) return;

    if (rec.freezeTicksLeft > 0) return;

    if (rec.core.activeWeapon < 0 || !hasAmmo(rec.weapons[rec.core.activeWeapon])) return;

    const projStartPos = vadd(rec.core.pos, vmul(dir, PHYSICAL_SIZE * 0.75));

    switch (rec.core.activeWeapon) {
      case WEAPON_HAMMER: {
        let hits = 0;
        if (this.svHit) {
          for (const other of this.order) {
            if (other === rec || !other.alive) continue;
            if (vdistance(other.core.pos, projStartPos) >= PHYSICAL_SIZE * 0.5 + PHYSICAL_SIZE) continue;

            const toTarget = vdistance(other.core.pos, rec.core.pos);
            const hitDir = toTarget > 0 ? vnormalize(vsub(other.core.pos, rec.core.pos)) : { x: 0, y: -1 };

            const strength = TUNING.hammerStrength;
            let boost = vmul(vnormalize(vadd(hitDir, { x: 0, y: -1.1 })), 10.0);

            const mr = other.moveRestrictions;
            if (mr !== 0) boost = vsub(clampVel(mr, vadd(other.core.vel, boost)), other.core.vel);
            const force = vmul(vadd({ x: 0, y: -1 }, boost), strength);
            other.core.vel = clampVel(mr, vadd(other.core.vel, force));
            unfreezeTee(other);

            events.push({ kind: "hammerHit", from: rec.core.id, to: other.core.id });
            hits++;
          }
          if (hits > 0) {
            rec.reloadTimer = Math.trunc((TUNING.hammerHitFireDelay * SERVER_TICK_SPEED) / 1000);
          }
        }
        events.push({ kind: "hammerFire", from: rec.core.id, hits });
        break;
      }
      case WEAPON_GUN: {
        const lifetime = Math.trunc(SERVER_TICK_SPEED * TUNING.gunLifetime);
        this.spawnProjectile(WEAPON_GUN, rec.core.id, projStartPos, dir, lifetime, false);
        break;
      }
      case WEAPON_SHOTGUN: {
        this.spawnLaser(rec.core.id, WEAPON_SHOTGUN, rec.core.pos, dir, TUNING.laserReach, events);
        break;
      }
      case WEAPON_GRENADE: {
        const lifetime = Math.trunc(SERVER_TICK_SPEED * TUNING.grenadeLifetime);
        this.spawnProjectile(WEAPON_GRENADE, rec.core.id, projStartPos, dir, lifetime, true);
        break;
      }
      case WEAPON_LASER: {
        this.spawnLaser(rec.core.id, WEAPON_LASER, rec.core.pos, dir, TUNING.laserReach, events);
        break;
      }
    }

    const slot = rec.weapons[rec.core.activeWeapon];
    if (slot.ammo > 0) slot.ammo--;

    rec.attackTick = this.tick;
    if (rec.reloadTimer === 0 && rec.core.activeWeapon !== -1) {
      rec.reloadTimer = fireDelayTicks(rec.core.activeWeapon);
    }
  }

  private respawnTee(rec: TeeRecord): void {

    this.order.splice(this.order.indexOf(rec), 1);
    this.order.unshift(rec);
    rec.core.reset();
    rec.core.pos = { ...rec.spawnPos };
    rec.prevPos = { ...rec.spawnPos };
    rec.core.activeWeapon = WEAPON_GUN;
    rec.alive = true;
    rec.weapons = defaultWeapons(this.infiniteAmmo, this.allWeapons);
    rec.queuedWeapon = -1;
    rec.reloadTimer = 0;
    rec.freezeTicksLeft = 0;
    rec.frozenLastTick = false;
    rec.deepFrozen = false;
    rec.teleCheckpoint = 0;
    rec.moveRestrictions = 0;
    rec.respawnAtTick = null;
  }

  kill(id: number): void {
    const rec = this.tees.get(id);
    if (rec === undefined) return;
    this.die(id, rec, id, this.pendingEvents);
  }
  private readonly pendingEvents: WorldEvent[] = [];

  private die(id: number, rec: TeeRecord, by: number, events: WorldEvent[]): void {
    if (!rec.alive) return;
    rec.alive = false;
    events.push({ kind: "death", id, by });
    if (this.respawnDelayTicks > 0) {
      rec.respawnAtTick = this.tick + this.respawnDelayTicks;
    }
  }

  private resetHook(rec: TeeRecord): void {
    const c = rec.core;
    c.setHookedPlayer(-1);
    c.hookState = HOOK_RETRACTED;
    c.triggeredEvents |= COREEVENT_HOOK_RETRACT;
    c.hookPos = { x: c.pos.x, y: c.pos.y };
  }

  private releaseHooked(id: number): void {
    for (const other of this.order) {
      const c = other.core;
      if (c.hookedPlayer !== id) continue;
      c.setHookedPlayer(-1);
      c.hookState = HOOK_RETRACTED;
      c.triggeredEvents |= COREEVENT_HOOK_RETRACT;
    }
  }

  private handleTile(rec: TeeRecord, index: number, events: WorldEvent[]): void {
    const col = this.collision;

    rec.moveRestrictions = col.getMoveRestrictions(rec.core.pos, 18, index);
    if (index < 0) return;
    const teleType = col.teleTypeAtIndex(index);
    const teleNumber = col.teleNumberAtIndex(index);
    if (teleType === TILE_TELECHECK && teleNumber !== 0) rec.teleCheckpoint = teleNumber;

    const tile = col.tiles[index];
    if (tile === TILE_FREEZE && !rec.deepFrozen) {
      if (freezeTee(this.tick, rec, FREEZE_SECONDS)) {
        events.push({ kind: "freeze", id: rec.core.id, by: -1 });
      }
    } else if (tile === TILE_UNFREEZE && !rec.deepFrozen) {
      unfreezeTee(rec);
    }
    if (tile === TILE_DFREEZE && !rec.deepFrozen) rec.deepFrozen = true;
    else if (tile === TILE_DUNFREEZE && rec.deepFrozen) rec.deepFrozen = false;

    if (rec.core.vel.y > 0 && (rec.moveRestrictions & CANTMOVE_DOWN) !== 0) {
      rec.core.jumped = 0;
      rec.core.jumpedTotal = 0;
    }
    if (rec.moveRestrictions !== 0) rec.core.vel = clampVel(rec.moveRestrictions, rec.core.vel);

    if (teleNumber === 0) return;
    if (teleType === TILE_TELEIN || teleType === TILE_TELEINEVIL) {
      const outs = col.teleOutsFor(teleNumber);
      if (outs.length === 0) return;
      rec.core.pos = { x: outs[0].x, y: outs[0].y };

      if (teleType === TILE_TELEINEVIL) {
        rec.core.vel = { x: 0, y: 0 };
        this.resetHook(rec);
        this.releaseHooked(rec.core.id);
      } else {
        this.resetHook(rec);
      }
      return;
    }
    if (teleType === TILE_TELECHECKINEVIL || teleType === TILE_TELECHECKIN) {
      const evil = teleType === TILE_TELECHECKINEVIL;
      let dest: Vec2 | undefined;
      for (let k = rec.teleCheckpoint; k >= 1 && dest === undefined; k--) {
        const outs = col.teleCheckOutsFor(k);
        if (outs.length > 0) dest = outs[0];
      }

      rec.core.pos = dest !== undefined ? { x: dest.x, y: dest.y } : { x: rec.spawnPos.x, y: rec.spawnPos.y };
      if (evil) rec.core.vel = { x: 0, y: 0 };
      this.resetHook(rec);
      if (evil) this.releaseHooked(rec.core.id);
    }
  }

  private applySpeedup(rec: TeeRecord, index: number): void {
    const speed = this.collision.speedupAt(index);
    if (speed === null) return;
    const vel = rec.core.vel;
    const force = speed.force;
    let maxSpeed = speed.maxSpeed;
    if (force === 255 && maxSpeed !== 0) {
      const k = Math.trunc(maxSpeed / 5);
      vel.x = speed.dirX * k;
      vel.y = speed.dirY * k;
      return;
    }
    const mr = rec.moveRestrictions;
    if (maxSpeed > 0 && maxSpeed < 5) maxSpeed = 5;
    if (maxSpeed > 0) {
      const oldAngle = (x: number, y: number): number => {
        let a: number;
        if (x > 0.0000001) a = -Math.atan(y / x);
        else if (x < 0.0000001) a = Math.atan(y / x) + Math.PI;
        else if (y > 0.0000001) a = Math.PI / 2;
        else a = -Math.PI / 2;
        if (a < 0) a += 2 * Math.PI;
        return a;
      };
      const speederAngle = oldAngle(speed.dirX, speed.dirY);
      const teeAngle = oldAngle(vel.x, vel.y);
      const teeSpeed = Math.sqrt(vel.x * vel.x + vel.y * vel.y);
      const speedLeft = maxSpeed / 5 - Math.cos(speederAngle - teeAngle) * teeSpeed;

      if (Number.isNaN(speedLeft)) return;
      let add: number;
      if (Math.abs(Math.trunc(speedLeft)) > force && speedLeft > 0.0000001) add = force;
      else if (Math.abs(Math.trunc(speedLeft)) > force) add = -force;
      else add = speedLeft;
      vel.x += speed.dirX * add;
      vel.y += speed.dirY * add;
    } else {
      vel.x += speed.dirX * force;
      vel.y += speed.dirY * force;
    }

    if (mr !== 0) rec.core.vel = clampVel(mr, vel);
  }

  private handleTiles(rec: TeeRecord, events: WorldEvent[]): void {
    const pos = rec.core.pos;
    const off = PHYSICAL_SIZE / 3;
    const deathHit =
      this.collision.isDeath(pos.x + off, pos.y - off) ||
      this.collision.isDeath(pos.x + off, pos.y + off) ||
      this.collision.isDeath(pos.x - off, pos.y - off) ||
      this.collision.isDeath(pos.x - off, pos.y + off);

    const centre = this.collision.getTileIndex(pos.x, pos.y);
    rec.core.isInFreeze = deathHit || centre === TILE_FREEZE || centre === TILE_DFREEZE || centre === TILE_LFREEZE || centre === TILE_DEATH;

    if (deathHit || isGameLayerClipped(pos, this.collision)) {
      this.die(rec.core.id, rec, -1, events);
      return;
    }

    const currentIndex = this.collision.getMapIndex(pos);
    if (currentIndex >= 0) this.applySpeedup(rec, currentIndex);

    const indices = this.collision.getMapIndices(rec.prevPos, pos, this.tileIndices);
    if (indices.length > 0) {
      for (let i = 0; i < indices.length; i++) {
        this.handleTile(rec, indices[i], events);
        if (!rec.alive) return;
      }
    } else {
      this.handleTile(rec, currentIndex, events);
    }
  }

  private tickEntities(events: WorldEvent[]): void {
    const nextProjectiles: Projectile[] = [];
    for (const p of this.projectilesList) {
      p.tick(this, events);
      if (!p.markedForDestroy) nextProjectiles.push(p);
    }
    this.projectilesList = nextProjectiles;

    const nextLasers: Laser[] = [];
    for (const l of this.lasersList) {
      l.tick(this, events);
      if (!l.markedForDestroy) nextLasers.push(l);
    }
    this.lasersList = nextLasers;
  }

  private preTick(rec: TeeRecord, doDeferred: boolean): void {
    const input = rec.input;

    let coreInput = input;
    if (rec.freezeTicksLeft > 0) {
      rec.freezeTicksLeft--;
      if (rec.freezeTicksLeft === 1) unfreezeTee(rec);
      const frozen = rec.frozenInput;
      copyInput(input, frozen);
      frozen.direction = 0;
      frozen.jump = 0;
      frozen.hook = 0;
      coreInput = frozen;
    }
    rec.pendingCoreInput = coreInput;

    rec.core.input = coreInput;
    rec.core.tick(true, doDeferred);
  }

  step(): WorldEvent[] {
    this.tick++;

    const events: WorldEvent[] = this.pendingEvents.splice(0);
    const order = this.order;

    for (const rec of this.byId) {
      if (!rec.alive && rec.respawnAtTick !== null && this.tick >= rec.respawnAtTick) {
        this.respawnTee(rec);
      }
    }

    this.tickEntities(events);

    for (let i = 0; i < order.length; i++) {
      const rec = order[i];
      if (!rec.alive) continue;
      const input = rec.input;

      this.handleWeaponSwitch(rec, input);
      this.fireWeapon(rec, input, events);

      copyInput(input, rec.prevInputForEdge);
    }

    const noWeakHook = this.noWeakHook;
    if (noWeakHook) {
      for (let i = 0; i < order.length; i++) {
        const rec = order[i];
        if (!rec.alive) continue;
        this.preTick(rec, false);
      }
    }
    for (let i = 0; i < order.length; i++) {
      const rec = order[i];
      if (!rec.alive) continue;
      const input = rec.input;

      if (noWeakHook) rec.core.tickDeferred();
      else this.preTick(rec, true);

      if (rec.reloadTimer > 0) rec.reloadTimer--;
      else this.fireWeapon(rec, input, events);

      rec.frozenLastTick = false;
      if (rec.deepFrozen) freezeTee(this.tick, rec, FREEZE_SECONDS);
      applyJumpRules(rec.core);

      this.handleTiles(rec, events);
      rec.prevPos.x = rec.core.pos.x;
      rec.prevPos.y = rec.core.pos.y;
    }

    for (let i = 0; i < order.length; i++) {
      const rec = order[i];
      if (!rec.alive) continue;
      rec.core.move();
      rec.core.quantize();
    }

    return events;
  }
}
