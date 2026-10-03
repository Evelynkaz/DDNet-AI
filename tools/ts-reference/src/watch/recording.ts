import type { TeeState, PlayerInput } from "../core/types.ts";
import type { Collision } from "../core/collision.ts";

export type RecFrame = {
  tick: number;
  tees: {
    id: number;
    x: number;
    y: number;
    vx: number;
    vy: number;
    angle: number;
    hookState: number;
    hookX: number;
    hookY: number;
    hookedPlayer: number;
    frozen: boolean;
    alive: boolean;
    weapon: number;

    direction?: number;
    jumped?: number;
    jumpsLeft?: number;
    freezeTicksLeft?: number;
    hookDx?: number;
    hookDy?: number;
    hookTick?: number;
    frozenFor?: number;
  }[];
  inputs: { id: number; direction: number; jump: number; hook: number; fire: number; targetX: number; targetY: number }[];
  events: string[];

  plan?: {
    target: number;
    lag: number;

    arm?: string;

    others?: number;

    lagPing?: number;
    searched: boolean;
    candidates: number;
    outOfTime: boolean;
    ms: number;
    selfOut: number;
    enemyOut: number;

    hookAt?: number;
    gated?: boolean;

    trek?: number;
  };

  walk?: string;

  plannedFreeze?: boolean;
};

export type Recording = {
  map: string;
  controller: string;
  seed: number;
  width: number;
  height: number;
  tiles: number[];

  tele?: number[][];

  selfId?: number;

  label?: string;

  players?: { id: number; name: string; clan: string; skin: string; cc?: boolean; cb?: number; cf?: number }[];
  frames: RecFrame[];
};

export function snapTee(t: TeeState): RecFrame["tees"][number] {
  return {
    id: t.id,
    x: Math.round(t.pos.x),
    y: Math.round(t.pos.y),
    vx: Number(t.vel.x.toFixed(3)),
    vy: Number(t.vel.y.toFixed(3)),
    angle: t.angle,
    hookState: t.hookState,
    hookX: Math.round(t.hookPos.x),
    hookY: Math.round(t.hookPos.y),
    hookedPlayer: t.hookedPlayer,
    frozen: t.frozen,
    alive: t.alive,
    weapon: t.activeWeapon,
    direction: t.direction,
    jumped: t.jumped,
    jumpsLeft: t.jumpsLeft,
    freezeTicksLeft: t.freezeTicksLeft,
    hookDx: Number(t.hookDir.x.toFixed(3)),
    hookDy: Number(t.hookDir.y.toFixed(3)),
    hookTick: t.hookTick,
    frozenFor: t.frozenFor,
  };
}

export function recTeeState(t: RecFrame["tees"][number], collision?: Collision): TeeState & { exact: boolean } {
  const exact = t.jumpsLeft !== undefined && t.direction !== undefined && t.hookDx !== undefined;
  let hookDx = t.hookDx ?? 0;
  let hookDy = t.hookDy ?? 0;
  if (t.hookDx === undefined) {
    const dx = t.hookX - t.x;
    const dy = t.hookY - t.y;
    const n = Math.hypot(dx, dy);
    if (n > 0) {
      hookDx = dx / n;
      hookDy = dy / n;
    }
  }

  const half = 14;
  const grounded =
    collision !== undefined && (collision.isSolid(t.x - half, t.y + half + 5) || collision.isSolid(t.x + half, t.y + half + 5));
  return {
    id: t.id,
    alive: t.alive,
    pos: { x: t.x, y: t.y },
    vel: { x: t.vx, y: t.vy },
    hookState: t.hookState,
    hookPos: { x: t.hookX, y: t.hookY },
    hookDir: { x: hookDx, y: hookDy },
    hookedPlayer: t.hookedPlayer,
    jumped: t.jumped ?? 0,
    jumpsLeft: t.jumpsLeft ?? (grounded ? 2 : 1),
    direction: t.direction ?? (t.vx > 1 ? 1 : t.vx < -1 ? -1 : 0),
    angle: t.angle,
    activeWeapon: t.weapon,
    frozen: t.frozen,
    freezeTicksLeft: t.freezeTicksLeft ?? (t.frozen ? 150 : 0),
    attackTick: 0,
    hookTick: t.hookTick,
    frozenFor: t.frozenFor ?? (t.frozen ? Math.max(0, 150 - (t.freezeTicksLeft ?? 150)) : undefined),
    exact,
  };
}

export function snapInput(id: number, i: PlayerInput): RecFrame["inputs"][number] {
  return { id, direction: i.direction, jump: i.jump, hook: i.hook, fire: i.fire, targetX: i.targetX, targetY: i.targetY };
}

export class RingRecorder {
  private readonly frames: (RecFrame | undefined)[];
  private next = 0;
  private filled = 0;
  private map: { width: number; height: number; tiles: number[]; tele: number[][] } | null = null;
  readonly capacity: number;

  constructor(capacity: number) {
    this.capacity = Math.max(1, capacity);
    this.frames = new Array<RecFrame | undefined>(this.capacity);
  }

  setMap(collision: Collision): void {
    const tele: number[][] = [];
    for (let i = 0; i < collision.width * collision.height; i++) {
      const t = collision.teleTypeAtIndex(i);
      if (t !== 0) tele.push([i, t, collision.teleNumberAtIndex(i)]);
    }
    this.map = { width: collision.width, height: collision.height, tiles: Array.from(collision.tiles), tele };
  }

  get hasMap(): boolean {
    return this.map !== null;
  }

  push(frame: RecFrame): void {
    this.frames[this.next] = frame;
    this.next = (this.next + 1) % this.capacity;
    if (this.filled < this.capacity) this.filled++;
  }

  get length(): number {
    return this.filled;
  }

  toRecording(meta: { map: string; controller: string; selfId: number; label?: string }): Recording | null {
    if (this.map === null || this.filled === 0) return null;
    const out: RecFrame[] = [];
    const start = this.filled === this.capacity ? this.next : 0;
    for (let i = 0; i < this.filled; i++) {
      const f = this.frames[(start + i) % this.capacity];
      if (f !== undefined) out.push(f);
    }
    return {
      map: meta.map,
      controller: meta.controller,
      seed: 0,
      width: this.map.width,
      height: this.map.height,
      tiles: this.map.tiles,
      tele: this.map.tele.length > 0 ? this.map.tele : undefined,
      selfId: meta.selfId,
      label: meta.label,
      frames: out,
    };
  }

  clear(): void {
    this.next = 0;
    this.filled = 0;
    this.frames.fill(undefined);
  }
}
