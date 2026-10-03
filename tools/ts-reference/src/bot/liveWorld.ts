import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import type { Vec2 } from "../core/vmath.ts";
import { vnormalize } from "../core/vmath.ts";
import type { Collision } from "../core/collision.ts";
import { evolveCore, needsEvolve } from "../demo/reckoning.ts";
import type { ProjectileState, TeeState, WorldView } from "../core/types.ts";
import { WEAPON_GRENADE, WEAPON_GUN, WEAPON_LASER, WEAPON_NINJA } from "../core/types.ts";
import { PHYSICAL_SIZE, SERVER_TICK_SPEED, TUNING } from "../core/tuning.ts";
import { Projectile } from "../core/projectile.ts";
import { loadMapCollision } from "../map/loadMap.ts";
import { calculateUuid, NETOBJTYPE_PROJECTILE } from "../demo/snapshot.ts";

export type SnapCharacterCore = {
  tick: number;
  x: number;
  y: number;
  vel_x: number;
  vel_y: number;
  angle: number;
  direction: number;
  jumped: number;
  hooked_player: number;
  hook_state: number;
  hook_tick: number;
  hook_x: number;
  hook_y: number;
  hook_dx: number;
  hook_dy: number;
};

export type SnapCharacter = {
  character_core: SnapCharacterCore;
  player_flags: number;
  health: number;
  armor: number;
  ammo_count: number;
  weapon: number;
  emote: number;
  attack_tick: number;
  client_id: number;
};

export type SnapDDNetCharacter = {
  m_Flags: number;
  m_FreezeEnd: number;
  m_Jumps: number;
  m_JumpedTotal?: number | null;

  m_FreezeStart?: number | null;
};

export type SnapProjectile = { x: number; y: number; vel_x: number; vel_y: number; type_: number; start_tick: number };

export type SnapLaser = { x: number; y: number; from_x: number; from_y: number; start_tick: number };

export interface SnapshotSource {
  readonly AllObjCharacter: SnapCharacter[];
  getObjExDDNetCharacter(id: number): SnapDDNetCharacter | undefined;
  readonly AllProjectiles: SnapProjectile[];
  readonly AllObjLaser: SnapLaser[];
}

export type RawSnapItem = { readonly type_id: number; readonly id: number; readonly data: readonly number[] };

export function uuidInts(name: string): number[] {
  const h = Buffer.from(calculateUuid(name), "hex");
  return [0, 4, 8, 12].map((o) => h.readInt32BE(o));
}
const DDNET_PROJECTILE_UUID = uuidInts("ddnet-projectile@netobj.ddnet.tw");
const DDNET_LASER_UUID = uuidInts("laser@netobj.ddnet.tw");

const DDNET_PLAYER_UUID = uuidInts("player@netobj.ddnet.tw");

export const EXPLAYERFLAG_AFK = 1 << 0;
export const EXPLAYERFLAG_PAUSED = 1 << 1;
export const EXPLAYERFLAG_SPEC = 1 << 2;

function exTypeId(raw: readonly RawSnapItem[], uuid: readonly number[]): number {
  for (const it of raw) {
    if (it.type_id !== 0 || it.data.length < 4) continue;
    if (it.data[0] === uuid[0] && it.data[1] === uuid[1] && it.data[2] === uuid[2] && it.data[3] === uuid[3]) return it.id;
  }
  return -1;
}

const PROJECTILEFLAG_EXPLOSIVE = 1 << 2;
const PROJECTILEFLAG_NORMALIZE_VEL = 1 << 4;

export function projectileFromItem(type: number, id: number, f: ArrayLike<number>, tick: number): ProjectileState | undefined {
  let pos: Vec2;
  let dir: Vec2;
  let weapon: number;
  let startTick: number;
  let owner = -1;
  let explosive: boolean;
  if (type === NETOBJTYPE_PROJECTILE) {
    if (f.length < 6) return undefined;
    pos = { x: f[0], y: f[1] };
    dir = { x: f[2] / 100, y: f[3] / 100 };
    weapon = f[4];
    startTick = f[5];
    explosive = weapon === WEAPON_GRENADE;
  } else {
    if (f.length < 10) return undefined;
    pos = { x: f[0] / 100, y: f[1] / 100 };
    const flags = f[9];
    dir = (flags & PROJECTILEFLAG_NORMALIZE_VEL) !== 0 ? vnormalize({ x: f[2], y: f[3] }) : { x: f[2] / 1e6, y: f[3] / 1e6 };
    weapon = f[4];
    startTick = f[5];
    owner = f[6];
    explosive = (flags & PROJECTILEFLAG_EXPLOSIVE) !== 0;
  }
  const proj = new Projectile(id, weapon, owner, pos, dir, startTick, -1, explosive);
  return { id, type: weapon, owner, pos: proj.posAtTick(tick), vel: { ...proj.vel }, startTick, spawnPos: { ...pos }, dir: { ...dir } };
}

const DEEP_FREEZE_TICKS = 3 * SERVER_TICK_SPEED;

function jumpsLeftOf(jumps: number, jumped: number, jumpedTotal: number, pos: Vec2, collision: Collision): number {
  if (jumps <= 0) return 0;
  const grounded =
    collision.isSolid(pos.x + PHYSICAL_SIZE / 2, pos.y + PHYSICAL_SIZE / 2 + 5) ||
    collision.isSolid(pos.x - PHYSICAL_SIZE / 2, pos.y + PHYSICAL_SIZE / 2 + 5);
  if (grounded) return jumps;
  if (jumped & 2) return 0;
  return Math.max(0, jumps - 1 - jumpedTotal);
}

export function decodeCharacter(
  item: SnapCharacter,
  ex: SnapDDNetCharacter | undefined,
  tick: number,
  collision: Collision,
  prev: TeeState | undefined,
): TeeState {

  const core = needsEvolve(item.character_core, tick)
    ? evolveCore(item.character_core, item.weapon, tick, collision)
    : item.character_core;
  const pos = { x: core.x, y: core.y };

  const freezeEnd = ex?.m_FreezeEnd ?? 0;
  let freezeTicksLeft = 0;
  if (freezeEnd === -1) freezeTicksLeft = DEEP_FREEZE_TICKS;
  else if (freezeEnd > 0) freezeTicksLeft = Math.max(1, freezeEnd - tick);
  const frozen = freezeTicksLeft > 0;

  let activeWeapon = item.weapon;
  if (activeWeapon === WEAPON_NINJA && frozen) activeWeapon = prev?.activeWeapon ?? WEAPON_GUN;

  const jumps = ex?.m_Jumps ?? 2;
  const jumpedTotalRaw = ex?.m_JumpedTotal;

  const jumpedTotal = typeof jumpedTotalRaw === "number" && jumpedTotalRaw >= 0 ? jumpedTotalRaw : 0;

  return {
    id: item.client_id,
    alive: true,
    pos,
    vel: { x: core.vel_x / 256, y: core.vel_y / 256 },
    hookState: core.hook_state,
    hookPos: { x: core.hook_x, y: core.hook_y },
    hookDir: { x: core.hook_dx / 256, y: core.hook_dy / 256 },
    hookedPlayer: core.hooked_player,
    jumped: core.jumped,
    jumpsLeft: jumpsLeftOf(jumps, core.jumped, jumpedTotal, pos, collision),
    direction: core.direction,
    angle: core.angle,
    activeWeapon,
    frozen,
    freezeTicksLeft,

    deepFrozen: freezeEnd === -1,
    attackTick: item.attack_tick,

    sinceAttack: Math.max(0, tick - item.attack_tick),
    hookTick: core.hook_tick,
    jumpedTotal: typeof jumpedTotalRaw === "number" && jumpedTotalRaw >= 0 ? jumpedTotalRaw : undefined,

    reloadTicks: Math.max(0, item.attack_tick + Math.trunc((TUNING.hammerFireDelay * SERVER_TICK_SPEED) / 1000) - tick),

    frozenFor: !frozen
      ? undefined
      : typeof ex?.m_FreezeStart === "number" && ex.m_FreezeStart > 0
        ? Math.max(0, tick - ex.m_FreezeStart)
        : Math.max(0, 3 * SERVER_TICK_SPEED - freezeTicksLeft),

    jumps,
    ddnetFlags: ex?.m_Flags,
  };
}

export class LiveWorld implements WorldView {
  collision: Collision;
  tick = 0;
  selfId = -1;
  private readonly tees = new Map<number, TeeState>();
  private projectilesList: ProjectileState[] = [];
  private lasersList: ProjectileState[] = [];

  private readonly flagsById = new Map<number, number>();

  constructor(collision: Collision) {
    this.collision = collision;
  }

  setCollision(c: Collision): void {
    this.collision = c;
  }

  updateFromSnapshot(unpacker: SnapshotSource, ownId: number, gameTick?: number, raw?: readonly RawSnapItem[]): void {
    this.selfId = ownId;
    const chars = unpacker.AllObjCharacter;

    let tick = gameTick ?? this.tick + 1;
    if (gameTick === undefined) {
      for (const c of chars) if (c.character_core.tick > tick) tick = c.character_core.tick;
    }
    this.tick = tick;

    for (const tee of this.tees.values()) tee.alive = false;
    for (const c of chars) {
      const id = c.client_id;
      this.tees.set(id, decodeCharacter(c, unpacker.getObjExDDNetCharacter(id), tick, this.collision, this.tees.get(id)));
    }

    this.projectilesList = [];
    const projs = unpacker.AllProjectiles;
    for (let i = 0; i < projs.length; i++) {
      const p = projs[i];
      const dir = { x: p.vel_x / 100, y: p.vel_y / 100 };
      const proj = new Projectile(i, p.type_, -1, { x: p.x, y: p.y }, dir, p.start_tick, -1, p.type_ === WEAPON_GRENADE);
      this.projectilesList.push({
        id: i,
        type: p.type_,
        owner: -1,
        pos: proj.posAtTick(tick),
        vel: { ...proj.vel },
        dir: { ...proj.dir },
        startTick: p.start_tick,
        spawnPos: { x: p.x, y: p.y },
      });
    }

    const projType = raw === undefined ? -1 : exTypeId(raw, DDNET_PROJECTILE_UUID);
    if (raw !== undefined && projType !== -1) {
      for (const it of raw) {
        if (it.type_id !== projType) continue;
        const p = projectileFromItem(projType, this.projectilesList.length, it.data, tick);
        if (p) this.projectilesList.push(p);
      }
    }

    this.lasersList = [];
    const lasers = unpacker.AllObjLaser;
    for (let i = 0; i < lasers.length; i++) {
      const l = lasers[i];
      this.lasersList.push({

        spawnPos: { x: l.from_x, y: l.from_y },
        dir: { x: 0, y: 0 },
        id: i,
        type: WEAPON_LASER,
        owner: -1,
        pos: { x: l.x, y: l.y },
        vel: { x: l.x - l.from_x, y: l.y - l.from_y },
        startTick: l.start_tick,
      });
    }

    const laserType = raw === undefined ? -1 : exTypeId(raw, DDNET_LASER_UUID);
    if (raw !== undefined && laserType !== -1) {
      for (const it of raw) {
        if (it.type_id !== laserType || it.data.length < 6) continue;
        const d = it.data;
        this.lasersList.push({
          spawnPos: { x: d[2], y: d[3] },
          dir: { x: 0, y: 0 },
          id: this.lasersList.length,
          type: WEAPON_LASER,
          owner: d[5],
          pos: { x: d[0], y: d[1] },
          vel: { x: d[0] - d[2], y: d[1] - d[3] },
          startTick: d[4],
        });
      }
    }

    this.flagsById.clear();
    const playerType = raw === undefined ? -1 : exTypeId(raw, DDNET_PLAYER_UUID);
    if (raw !== undefined && playerType !== -1) {
      for (const it of raw) {
        if (it.type_id !== playerType || it.data.length < 1) continue;
        this.flagsById.set(it.id, it.data[0]);
      }
    }
  }

  playerFlags(id: number): number {
    return this.flagsById.get(id) ?? 0;
  }

  serverAfk(id: number): boolean {
    return (this.playerFlags(id) & EXPLAYERFLAG_AFK) !== 0;
  }

  spectating(id: number): boolean {
    return (this.playerFlags(id) & EXPLAYERFLAG_SPEC) !== 0;
  }

  notPlaying(id: number): boolean {
    return (this.playerFlags(id) & (EXPLAYERFLAG_PAUSED | EXPLAYERFLAG_SPEC)) !== 0;
  }

  getTee(id: number): TeeState | undefined {
    return this.tees.get(id);
  }

  allTees(): TeeState[] {
    return Array.from(this.tees.values());
  }

  projectiles(): ProjectileState[] {
    return this.projectilesList;
  }

  lasers(): ProjectileState[] {
    return this.lasersList;
  }
}

export type MapClientLike = {
  map?: { mapBuffer?: Buffer; map_name?: string; downloading?: boolean } | undefined;
  lastMapDetails?: { map_name: string };
};

export function mapCollisionFromClient(client: MapClientLike, fallbackMapDir: string): Collision | null {
  const map = client.map;
  const rawName = map?.map_name ?? client.lastMapDetails?.map_name;
  const name = rawName ? basename(rawName) : undefined;

  if (map?.mapBuffer && map.mapBuffer.length > 0 && !map.downloading) {

    const dir = mkdtempSync(join(tmpdir(), "ddnet-ai-map-"));
    try {
      const tmp = join(dir, `${name ?? "map"}.map`);
      writeFileSync(tmp, map.mapBuffer);
      return loadMapCollision(tmp).collision;
    } catch {

    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }

  if (name !== undefined) {
    const path = join(fallbackMapDir, `${name}.map`);
    if (existsSync(path)) {
      try {
        return loadMapCollision(path).collision;
      } catch {
        return null;
      }
    }
  }
  return null;
}
