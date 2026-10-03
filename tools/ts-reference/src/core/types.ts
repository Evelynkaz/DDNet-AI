import type { Vec2 } from "./vmath.ts";
import type { Collision } from "./collision.ts";

export type PlayerInput = {
  direction: number;
  targetX: number;
  targetY: number;
  jump: number;
  fire: number;
  hook: number;
  playerFlags: number;
  wantedWeapon: number;
  nextWeapon: number;
  prevWeapon: number;
};

export function blankTeeState(): TeeState {
  return {
    id: 0, alive: false, pos: { x: 0, y: 0 }, vel: { x: 0, y: 0 }, hookState: 0,
    hookPos: { x: 0, y: 0 }, hookDir: { x: 0, y: 0 }, hookedPlayer: -1, jumped: 0,
    jumpsLeft: 0, direction: 0, angle: 0, activeWeapon: 0, frozen: false,
    freezeTicksLeft: 0, attackTick: 0,
  };
}

export function emptyInput(): PlayerInput {
  return {
    direction: 0,
    targetX: 0,
    targetY: -1,
    jump: 0,
    fire: 0,
    hook: 0,
    playerFlags: 0,
    wantedWeapon: 0,
    nextWeapon: 0,
    prevWeapon: 0,
  };
}

export type TeeState = {
  id: number;
  alive: boolean;
  pos: Vec2;
  vel: Vec2;
  hookState: number;
  hookPos: Vec2;
  hookDir: Vec2;
  hookedPlayer: number;
  jumped: number;
  jumpsLeft: number;
  direction: number;

  angle: number;
  activeWeapon: number;
  frozen: boolean;
  freezeTicksLeft: number;
  attackTick: number;

  hookTick?: number;

  jumpedTotal?: number;

  reloadTicks?: number;

  frozenFor?: number;

  deepFrozen?: boolean;

  jumps?: number;

  ddnetFlags?: number;

  sinceAttack?: number;
};

export const CHARACTERFLAG_SOLO = 1 << 0;
export const CHARACTERFLAG_COLLISION_DISABLED = 1 << 2;
export const CHARACTERFLAG_ENDLESS_HOOK = 1 << 3;
export const CHARACTERFLAG_HOOK_HIT_DISABLED = 1 << 10;
export const CHARACTERFLAG_WEAPON_NINJA = 1 << 19;

export function wireAngleRad(angle: number): number {
  const a = angle / 256;
  return a >= Math.PI ? a - 2 * Math.PI : a;
}

export type ProjectileState = {
  id: number;
  type: number;
  owner: number;

  pos: Vec2;

  vel: Vec2;

  dir: Vec2;
  startTick: number;

  spawnPos: Vec2;
};

export type WorldEvent =
  | { kind: "hammerHit"; from: number; to: number }

  | { kind: "hammerFire"; from: number; hits: number }
  | { kind: "explosion"; pos: Vec2; owner: number }

  | { kind: "laserHit"; from: number; to: number; weapon: number }
  | { kind: "freeze"; id: number; by: number }
  | { kind: "death"; id: number; by: number };

export interface WorldView {
  readonly tick: number;
  readonly collision: Collision;
  getTee(id: number): TeeState | undefined;
  allTees(): TeeState[];
  projectiles(): ProjectileState[];
}

export const WEAPON_HAMMER = 0;
export const WEAPON_GUN = 1;
export const WEAPON_SHOTGUN = 2;
export const WEAPON_GRENADE = 3;
export const WEAPON_LASER = 4;
export const WEAPON_NINJA = 5;
export const NUM_WEAPONS = 6;
