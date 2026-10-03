import type { Collision } from "../core/collision.ts";
import { TILE_TELEIN, TILE_TELEINEVIL, TUNING } from "../core/tuning.ts";

export type MoveKind = "walk" | "fall" | "jump" | "hook" | "kill";

export type RouteStep = {

  x: number;
  y: number;
  kind: MoveKind;

  anchorX?: number;
  anchorY?: number;

  freeze?: boolean;

  tele?: boolean;

  move?: number;

  leap?: boolean;
};

export function routeMoveKey(index: number, kind: number): number {
  return index * 8 + kind;
}

const TILE_PX = 32;

const HOOK_TILES = Math.floor(TUNING.hookLength / TILE_PX);

const JUMP_UP_REACH = [7.7, 7.3, 6.9, 6.3, 5.6, 4.4];

const JUMP_DOWN_REACH = [7.7, 8.1, 8.4, 8.8, 9.1, 9.4, 9.7];

const JUMP_MARGIN = 0.8;

const COST_WALK = 1;
const COST_FALL = 1;
const COST_JUMP = 3;
const COST_HOOK = 6;

const COST_KILL = 60;

const COST_FREEZE_FALL = 25;
const FREEZE_FALL_TILES = 3;

const COST_FREEZE_CROSS = 20;
const FREEZE_CROSS_TILES = 3;

const MAX_TRAP_TILES = 250;

const COST_DANGER = 4;

type Grid = {
  width: number;
  height: number;

  free: Uint8Array;

  hookable: Uint8Array;

  solid: Uint8Array;

  death: Uint8Array;

  unfreeze: Uint8Array;

  danger: Uint8Array;

  teleOut: Int32Array;
  teleIn: Map<number, number[]>;

  firstSolid: Int32Array;
};

const grids = new WeakMap<Collision, Grid>();

function gridOf(collision: Collision): Grid {
  const cached = grids.get(collision);
  if (cached !== undefined) return cached;
  const { width, height } = collision;
  const free = new Uint8Array(width * height);
  const hookable = new Uint8Array(width * height);
  const solid = new Uint8Array(width * height);
  const death = new Uint8Array(width * height);
  const unfreeze = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const px = x * TILE_PX + TILE_PX / 2;
      const py = y * TILE_PX + TILE_PX / 2;
      const isSolid = collision.isSolid(px, py);
      solid[i] = isSolid ? 1 : 0;
      death[i] = collision.isDeath(px, py) ? 1 : 0;
      unfreeze[i] = collision.isUnFreeze(px, py) ? 1 : 0;
      hookable[i] = isSolid && !collision.isNoHook(px, py) ? 1 : 0;
      free[i] = !isSolid && !collision.isFreeze(px, py) && !collision.isDeath(px, py) ? 1 : 0;

      if (free[i] === 0 && !isSolid) {
        const t = collision.teleTypeAtIndex(i);
        if (t === TILE_TELEIN || t === TILE_TELEINEVIL) free[i] = 1;
      }
    }
  }
  const danger = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let bad = 0;
      for (let oy = -1; oy <= 1 && bad === 0; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          const nx = x + ox;
          const ny = y + oy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const px2 = nx * TILE_PX + TILE_PX / 2;
          const py2 = ny * TILE_PX + TILE_PX / 2;
          if (collision.isFreeze(px2, py2) || collision.isDeath(px2, py2)) {
            bad = 1;
            break;
          }
        }
      }
      danger[y * width + x] = bad;
    }
  }
  const teleOut = new Int32Array(width * height).fill(-1);
  const teleIn = new Map<number, number[]>();
  if (collision.hasTele()) {
    for (let i = 0; i < width * height; i++) {
      const t = collision.teleTypeAtIndex(i);
      if (t !== TILE_TELEIN && t !== TILE_TELEINEVIL) continue;
      const outs = collision.teleOutsFor(collision.teleNumberAtIndex(i));
      if (outs.length === 0) continue;
      const oi = Math.trunc(outs[0].y / TILE_PX) * width + Math.trunc(outs[0].x / TILE_PX);
      teleOut[i] = oi;
      const list = teleIn.get(oi) ?? [];
      list.push(i);
      teleIn.set(oi, list);
    }
  }

  const firstSolid = new Int32Array(8 * width * height).fill(-1);
  for (let d = 0; d < 8; d++) {
    const dx = RAY_DX[d];
    const dy = RAY_DY[d];
    const xs = dx > 0 ? [...Array(width).keys()].reverse() : [...Array(width).keys()];
    const ys = dy > 0 ? [...Array(height).keys()].reverse() : [...Array(height).keys()];
    for (const y of ys) {
      for (const x of xs) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const ni = ny * width + nx;
        firstSolid[d * width * height + y * width + x] = solid[ni] === 1 ? ni : firstSolid[d * width * height + ni];
      }
    }
  }
  const g = { width, height, free, hookable, solid, death, unfreeze, danger, teleOut, teleIn, firstSolid };
  grids.set(collision, g);
  return g;
}

const supported = (g: Grid, x: number, y: number): boolean => y + 1 >= g.height || g.solid[(y + 1) * g.width + x] === 1;

function clearLine(g: Grid, x0: number, y0: number, x1: number, y1: number, forTee: boolean): boolean {
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let err = dx - dy;
  let x = x0;
  let y = y0;
  for (;;) {
    if (!(x === x0 && y === y0) && !(x === x1 && y === y1)) {
      const i = y * g.width + x;
      if (forTee ? g.free[i] === 0 : g.solid[i] === 1) return false;
    }
    if (x === x1 && y === y1) return true;
    const e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x += sx;
    }
    if (e2 < dx) {
      err += dx;
      y += sy;
    }
  }
}

class Heap {
  private readonly idx: Int32Array;
  private readonly cost: Int32Array;
  private size = 0;
  constructor(capacity: number) {
    this.idx = new Int32Array(capacity);
    this.cost = new Int32Array(capacity);
  }
  get empty(): boolean {
    return this.size === 0;
  }
  clear(): void {
    this.size = 0;
  }
  push(index: number, cost: number): void {
    if (this.size >= this.idx.length) return;
    let i = this.size++;
    this.idx[i] = index;
    this.cost[i] = cost;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.cost[p] <= this.cost[i]) break;
      this.swap(p, i);
      i = p;
    }
  }
  pop(): { index: number; cost: number } {
    const out = { index: this.idx[0], cost: this.cost[0] };
    this.size--;
    if (this.size > 0) {
      this.idx[0] = this.idx[this.size];
      this.cost[0] = this.cost[this.size];
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < this.size && this.cost[l] < this.cost[m]) m = l;
        if (r < this.size && this.cost[r] < this.cost[m]) m = r;
        if (m === i) break;
        this.swap(m, i);
        i = m;
      }
    }
    return out;
  }
  private swap(a: number, b: number): void {
    const i = this.idx[a];
    const c = this.cost[a];
    this.idx[a] = this.idx[b];
    this.cost[a] = this.cost[b];
    this.idx[b] = i;
    this.cost[b] = c;
  }
}

const UNREACHED = 0x3fffffff;

type Scratch = {
  dist: Int32Array;
  from_: Int32Array;
  kind: Uint8Array;
  anchor: Int32Array;

  stamp: Int32Array;
  heap: Heap;
  gen: number;
};

const scratches = new WeakMap<Collision, Scratch>();

function scratchOf(collision: Collision, n: number): Scratch {
  const cached = scratches.get(collision);
  if (cached !== undefined && cached.dist.length === n) return cached;
  const made: Scratch = {
    dist: new Int32Array(n),
    from_: new Int32Array(n),
    kind: new Uint8Array(n),
    anchor: new Int32Array(n),
    stamp: new Int32Array(n),
    heap: new Heap(n),
    gen: 0,
  };
  scratches.set(collision, made);
  return made;
}

type Visit = (nx: number, ny: number, cost: number, kind: number, anchor: number) => void;

const FREEZE_KIND = 5;

function expand(g: Grid, x: number, y: number, visit: Visit, spawns?: readonly number[], throughFreeze = true, looseCrossing = false): void {

  if (spawns !== undefined) {
    for (const si of spawns) {
      if (si !== y * g.width + x) visit(si % g.width, Math.trunc(si / g.width), COST_KILL, 4, -1);
    }
  }

  for (const dx of [-1, 1]) {
    const nx = x + dx;
    if (nx < 0 || nx >= g.width) continue;
    if (g.free[y * g.width + nx] === 0) continue;
    if (!supported(g, x, y) && !supported(g, nx, y)) continue;
    visit(nx, y, COST_WALK, 0, -1);
  }

  for (const dx of [0, -1, 1]) {
    const nx = x + dx;
    const ny = y + 1;
    if (nx < 0 || nx >= g.width || ny >= g.height) continue;
    if (g.free[ny * g.width + nx] === 0) continue;
    if (dx !== 0 && g.free[y * g.width + nx] === 0) continue;
    visit(nx, ny, COST_FALL, 1, -1);
  }

  for (let d = 1; throughFreeze && d <= FREEZE_FALL_TILES; d++) {
    const ny = y + d;
    if (ny >= g.height) break;
    const i = ny * g.width + x;
    if (g.free[i] === 1) break;
    if (g.solid[i] === 1 || g.death[i] === 1) break;
    const outY = ny + 1;
    if (outY >= g.height) break;
    if (g.free[outY * g.width + x] === 1) {
      visit(x, outY, COST_FREEZE_FALL * d, FREEZE_KIND, -1);
      break;
    }
  }

  for (const dx of throughFreeze ? [-1, 1] : []) {
    const c = freezeCrossing(g, x, y, dx, looseCrossing);
    if (c === null) continue;

    const cost = g.unfreeze[y * g.width + c.outX] === 1 ? COST_FREEZE_CROSS : COST_FREEZE_CROSS * 2;
    visit(c.outX, y, cost * c.tiles, FREEZE_KIND, -1);
  }

  if (supported(g, x, y)) {
    for (let dy = -(JUMP_UP_REACH.length - 1); dy <= JUMP_DOWN_REACH.length - 1; dy++) {
      const ny = y + dy;
      if (ny < 0 || ny >= g.height) continue;
      const reach = (dy <= 0 ? JUMP_UP_REACH[-dy] : JUMP_DOWN_REACH[dy]) * JUMP_MARGIN;
      const span = Math.trunc(reach);
      for (let dx = -span; dx <= span; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = x + dx;
        if (nx < 0 || nx >= g.width) continue;
        if (g.free[ny * g.width + nx] === 0) continue;

        if (!supported(g, nx, ny)) continue;
        if (overshootsIntoHazard(g, nx, ny, dx)) continue;
        if (!jumpClear(g, x, y, nx, ny, dx, dy, reach)) continue;
        visit(nx, ny, COST_JUMP + Math.abs(dx) + Math.max(0, -dy), 2, -1);
      }
    }
  }

  const teleOut = g.teleOut[y * g.width + x];
  if (teleOut >= 0) visit(teleOut % g.width, Math.trunc(teleOut / g.width), COST_WALK, 0, -1);

  const n = g.width * g.height;
  for (let d = 0; d < 8; d++) {
    const ai = g.firstSolid[d * n + y * g.width + x];
    if (ai < 0 || g.hookable[ai] === 0) continue;
    const ax = ai % g.width;
    const ay = (ai - ax) / g.width;
    const k = Math.max(Math.abs(ax - x), Math.abs(ay - y));
    if (Math.hypot(ax - x, ay - y) * TILE_PX > TUNING.hookLength) continue;
    const tx = ax - RAY_DX[d];
    const ty = ay - RAY_DY[d];
    if (g.free[ty * g.width + tx] === 1 && clearLine(g, x, y, tx, ty, true)) {
      visit(tx, ty, COST_HOOK + k, 3, ai);
    }
  }
}

function expandBack(g: Grid, x: number, y: number, visit: Visit, looseCrossing = false): void {
  for (const dx of [-1, 1]) {
    const ax = x + dx;
    if (ax < 0 || ax >= g.width) continue;
    if (g.free[y * g.width + ax] === 0) continue;
    if (!supported(g, x, y) && !supported(g, ax, y)) continue;
    visit(ax, y, COST_WALK, 0, -1);
  }

  for (const dx of [0, -1, 1]) {
    const ax = x + dx;
    const ay = y - 1;
    if (ax < 0 || ax >= g.width || ay < 0) continue;
    if (g.free[ay * g.width + ax] === 0) continue;
    if (dx !== 0 && g.free[y * g.width + ax] === 0) continue;
    visit(ax, ay, COST_FALL, 1, -1);
  }

  for (let dy = -(JUMP_UP_REACH.length - 1); dy <= JUMP_DOWN_REACH.length - 1; dy++) {
    const ay = y - dy;
    if (ay < 0 || ay >= g.height) continue;
    const reach = (dy <= 0 ? JUMP_UP_REACH[-dy] : JUMP_DOWN_REACH[dy]) * JUMP_MARGIN;
    const span = Math.trunc(reach);
    for (let dx = -span; dx <= span; dx++) {
      if (dx === 0 && dy === 0) continue;
      const ax = x - dx;
      if (ax < 0 || ax >= g.width) continue;
      if (g.free[ay * g.width + ax] === 0) continue;
      if (!supported(g, ax, ay)) continue;
      if (!supported(g, x, y)) continue;
      if (overshootsIntoHazard(g, x, y, dx)) continue;
      if (!jumpClear(g, ax, ay, x, y, dx, dy, reach)) continue;
      visit(ax, ay, COST_JUMP + Math.abs(dx) + Math.max(0, -dy), 2, -1);
    }
  }

  for (const dx of [-1, 1]) {
    for (let d = 1; d <= FREEZE_CROSS_TILES; d++) {
      const nx = x + dx * d;
      if (nx < 0 || nx >= g.width) break;
      const i = y * g.width + nx;
      if (g.free[i] === 1) break;
      if (g.solid[i] === 1 || g.death[i] === 1) break;
      const fromX = nx + dx;
      if (fromX < 0 || fromX >= g.width) break;
      if (g.free[y * g.width + fromX] === 1) {
        const c = freezeCrossing(g, fromX, y, -dx, looseCrossing);
        if (c !== null && c.outX === x) visit(fromX, y, COST_FREEZE_CROSS * d, 1, -1);
        break;
      }
    }
  }

  for (let d = 1; d <= FREEZE_FALL_TILES; d++) {
    const ny = y - d;
    if (ny < 0) break;
    const i = ny * g.width + x;
    if (g.free[i] === 1) break;
    if (g.solid[i] === 1 || g.death[i] === 1) break;
    const fromY = ny - 1;
    if (fromY < 0) break;
    if (g.free[fromY * g.width + x] === 1) {
      visit(x, fromY, COST_FREEZE_FALL * d, 1, -1);
      break;
    }
  }

  for (const from of g.teleIn.get(y * g.width + x) ?? []) {
    visit(from % g.width, Math.trunc(from / g.width), COST_WALK, 0, -1);
  }

  for (let d = 0; d < 8; d++) {
    const dx = RAY_DX[d];
    const dy = RAY_DY[d];
    const ax = x + dx;
    const ay = y + dy;
    if (ax < 0 || ay < 0 || ax >= g.width || ay >= g.height) continue;
    if (g.hookable[ay * g.width + ax] === 0) continue;
    for (let k = 1; k <= HOOK_TILES; k++) {
      const fx = x - dx * k;
      const fy = y - dy * k;
      if (fx < 0 || fy < 0 || fx >= g.width || fy >= g.height) break;
      if (Math.hypot(ax - fx, ay - fy) * TILE_PX > TUNING.hookLength) break;
      if (g.free[fy * g.width + fx] === 0) break;
      if (!clearLine(g, fx, fy, ax, ay, false)) break;
      if (!clearLine(g, fx, fy, x, y, true)) break;
      visit(fx, fy, COST_HOOK + k, 3, ay * g.width + ax);
    }
  }
}

function freezeCrossing(g: Grid, x: number, y: number, dx: number, loose = false): { outX: number; tiles: number } | null {
  if (!loose && !supported(g, x, y)) return null;
  for (let d = 1; d <= FREEZE_CROSS_TILES; d++) {
    const nx = x + dx * d;
    if (nx < 0 || nx >= g.width) return null;
    const i = y * g.width + nx;
    if (g.free[i] === 1) return null;
    if (g.solid[i] === 1 || g.death[i] === 1) return null;
    if (!loose && supported(g, nx, y)) return null;
    const outX = nx + dx;
    if (outX < 0 || outX >= g.width) return null;
    if (g.free[y * g.width + outX] === 1) return { outX, tiles: d };
  }
  return null;
}

const RAY_DX = [0, 1, 1, 1, 0, -1, -1, -1];
const RAY_DY = [-1, -1, 0, 1, 1, 1, 0, -1];

const OVERSHOOT_DX = 2;

function overshootsIntoHazard(g: Grid, nx: number, ny: number, dx: number): boolean {
  if (Math.abs(dx) < OVERSHOOT_DX) return false;
  const bx = nx + Math.sign(dx);
  if (bx < 0 || bx >= g.width) return false;
  const hazard = (i: number): boolean => g.free[i] === 0 && g.solid[i] === 0;
  if (hazard(ny * g.width + bx)) return true;
  return ny + 1 < g.height && g.free[ny * g.width + bx] === 1 && hazard((ny + 1) * g.width + bx);
}

function jumpClear(g: Grid, x: number, y: number, nx: number, ny: number, dx: number, dy: number, reach: number): boolean {
  if (Math.abs(dx) > reach) return false;
  const apex = Math.min(y, ny);
  for (let yy = y - 1; yy >= apex; yy--) {
    if (g.free[yy * g.width + x] === 0) return false;
  }
  if (!clearLine(g, x, apex, nx, apex, true)) return false;
  return clearLine(g, nx, apex, nx, ny, true);
}

export type RouteResult = { steps: RouteStep[]; cost: number } | null;

export function findRoute(
  collision: Collision,
  from: { x: number; y: number },
  to: { x: number; y: number },

  opts?: { nearTiles?: number; maxCost?: number; partial?: boolean; allowKill?: boolean; maxNodes?: number; throughFreeze?: boolean; avoid?: ReadonlySet<number> },
): RouteResult {
  const avoid = opts?.avoid !== undefined && opts.avoid.size > 0 ? opts.avoid : null;
  const g = gridOf(collision);
  const killSpawns =
    opts?.allowKill === true
      ? spawnTiles(collision).map((p) => Math.trunc(p.y / TILE_PX) * collision.width + Math.trunc(p.x / TILE_PX))
      : undefined;
  const near = opts?.nearTiles ?? 2;
  const maxCost = opts?.maxCost ?? 4000;

  const maxNodes = opts?.maxNodes ?? 200000;
  let popped = 0;
  const sx = Math.min(g.width - 1, Math.max(0, Math.trunc(from.x / TILE_PX)));
  const sy = Math.min(g.height - 1, Math.max(0, Math.trunc(from.y / TILE_PX)));
  const gx = Math.min(g.width - 1, Math.max(0, Math.trunc(to.x / TILE_PX)));
  const gy = Math.min(g.height - 1, Math.max(0, Math.trunc(to.y / TILE_PX)));

  const n = g.width * g.height;
  const sc = scratchOf(collision, n);
  const { dist, from_, kind, anchor, stamp, heap } = sc;
  const gen = ++sc.gen;
  heap.clear();

  const start = sy * g.width + sx;
  dist[start] = 0;
  from_[start] = -1;
  stamp[start] = gen;
  heap.push(start, 0);
  let best = -1;

  let closest = start;
  let closestGap = Math.abs(sx - gx) + Math.abs(sy - gy);

  const relax = (ni: number, cost0: number, parent: number, k: number, anc: number): void => {
    if (avoid !== null && avoid.has(routeMoveKey(ni, k))) return;
    const cost = cost0 + (g.danger[ni] === 1 ? COST_DANGER : 0);
    if (cost > maxCost) return;
    if (stamp[ni] === gen && cost >= dist[ni]) return;
    stamp[ni] = gen;
    dist[ni] = cost;
    from_[ni] = parent;
    kind[ni] = k;
    anchor[ni] = anc;
    heap.push(ni, cost);
  };

  while (!heap.empty) {
    if (++popped > maxNodes) break;
    const { index, cost } = heap.pop();
    if (stamp[index] !== gen || cost > dist[index]) continue;
    const x = index % g.width;
    const y = (index - x) / g.width;
    if (Math.abs(x - gx) <= near && Math.abs(y - gy) <= near) {
      best = index;
      break;
    }
    const gap = Math.abs(x - gx) + Math.abs(y - gy);
    if (gap < closestGap) {
      closestGap = gap;
      closest = index;
    }
    expand(
      g,
      x,
      y,
      (nx, ny, step, k, anc) => {
        relax(ny * g.width + nx, cost + step, index, k, anc);
      },
      killSpawns,
      opts?.throughFreeze !== false,
    );
  }

  if (best < 0 && opts?.partial === true) {
    const startGap = Math.abs(sx - gx) + Math.abs(sy - gy);
    if (closest !== start && closestGap < startGap * 0.66) best = closest;
  }
  if (best < 0) return null;
  const steps: RouteStep[] = [];
  const kinds: MoveKind[] = ["walk", "fall", "jump", "hook", "kill", "fall"];
  for (let i = best, next = -1; i >= 0 && i !== start; next = i, i = from_[i]) {
    const x = i % g.width;
    const y = (i - x) / g.width;
    const a = anchor[i];
    const step: RouteStep = {
      x,
      y,
      kind: kinds[kind[i]],
      anchorX: a >= 0 ? a % g.width : undefined,
      anchorY: a >= 0 ? (a - (a % g.width)) / g.width : undefined,
      move: routeMoveKey(i, kind[i]),
    };
    if (kind[i] === FREEZE_KIND) {
      step.freeze = true;

      const p = from_[i];
      if (p >= 0 && p % g.width !== x) step.leap = true;
    }
    if (next >= 0 && g.teleOut[i] === next) step.tele = true;
    steps.push(step);
  }
  steps.reverse();
  return { steps, cost: dist[best] };
}

import type { PlayerInput, TeeState } from "../core/types.ts";
import { emptyInput } from "../core/types.ts";

const REACHED_PX = 48;

const CENTRE_PX = 10;

const HOOK_MAX_TICKS = 100;

const STALL_TICKS = 100;

const FROZEN_GIVE_UP_TICKS = 25;

const FREEZE_MOVE_TICKS = 3 * 50 + 50;

const VETO_LIMIT = 12;

const THAW_LOOKAHEAD_STEPS = 8;

export type RunnerState = "running" | "arrived" | "stuck" | "replan";

export class RouteRunner {
  private steps: RouteStep[];
  private at = 0;
  private hookTicks = 0;
  private bestDist = Infinity;
  private bestTick = -1;
  private frozenTicks = 0;
  state: RunnerState = "running";

  reason = "";

  private killWanted = false;

  private killTick = -1;
  private killDied = false;

  private decisions = 0;

  private vetoes = 0;

  private readonly grid: Grid | null;

  private readonly earlyFreeze: boolean;

  constructor(route: RouteStep[], collision?: Collision, opts?: { earlyFreeze?: boolean }) {
    this.steps = route;
    this.earlyFreeze = opts?.earlyFreeze === true;
    this.grid = collision === undefined ? null : gridOf(collision);
    if (route.length === 0) this.state = "arrived";
  }

  get remaining(): number {
    return Math.max(0, this.steps.length - this.at);
  }

  get current(): RouteStep | undefined {
    return this.steps[this.at];
  }

  get failedMove(): number | undefined {
    return this.state === "stuck" ? this.steps[this.at]?.move : undefined;
  }

  get awaitingKill(): boolean {
    return this.state === "running" && this.killTick >= 0 && !this.killDied;
  }

  respawned(): void {
    if (this.killTick >= 0) this.killDied = true;
  }

  vetoed(): void {
    if (this.state !== "running") return;
    if (++this.vetoes <= VETO_LIMIT) return;
    this.state = "stuck";
    this.reason = `the guard refused step ${this.at + 1}/${this.steps.length} (${this.steps[this.at]?.kind ?? "?"}) ${this.vetoes} times: no way back from the freeze`;
  }

  freezeAhead(n: number): boolean {
    if (this.state !== "running") return false;
    for (let i = Math.max(0, this.at - 1); i < Math.min(this.steps.length, this.at + n); i++) if (this.steps[i].freeze === true) return true;
    return false;
  }

  takeKill(): boolean {
    const k = this.killWanted;
    this.killWanted = false;
    return k;
  }

  private freezePlanned(): boolean {
    return this.steps[this.at]?.freeze === true || this.steps[this.at - 1]?.freeze === true || (this.earlyFreeze && this.steps[this.at + 1]?.freeze === true);
  }

  private advance(tick: number): void {
    this.at++;
    this.vetoes = 0;
    this.killTick = -1;
    this.hookTicks = 0;
    this.bestDist = Infinity;
    this.bestTick = tick;
  }

  private reached(self: TeeState, i: number): boolean {
    const step = this.steps[i];
    return Math.hypot(self.pos.x - (step.x * TILE_PX + 16), self.pos.y - (step.y * TILE_PX + 16)) < REACHED_PX;
  }

  step(self: TeeState, tick: number, others?: readonly TeeState[]): PlayerInput {
    const out = emptyInput();
    this.decisions++;
    if (this.state === "running" && !self.alive && this.killTick >= 0) this.killDied = true;
    if (this.state !== "running" || !self.alive) return out;

    if (self.frozen) {

      this.frozenTicks++;
      if (this.frozenTicks > (this.freezePlanned() ? FREEZE_MOVE_TICKS : FROZEN_GIVE_UP_TICKS)) {
        this.state = "stuck";
        this.reason = `froze on the way at step ${this.at + 1}/${this.steps.length}`;
      }
      this.bestTick = tick;
      return out;
    }
    const thawed = this.frozenTicks > 0;
    this.frozenTicks = 0;

    if (thawed && this.freezePlanned()) {
      let on = -1;
      for (let k = this.at; k < Math.min(this.steps.length, this.at + THAW_LOOKAHEAD_STEPS); k++) if (this.reached(self, k)) on = k;
      if (on < 0) {
        this.state = "replan";
        this.reason = `came out of the planned freeze at step ${this.at + 1}/${this.steps.length} off the route`;
        return out;
      }
      while (this.at < on) this.advance(tick);
    }
    let step = this.steps[this.at];
    while (step !== undefined) {
      if (step.tele === true) {

        if (this.at + 1 < this.steps.length && this.reached(self, this.at + 1)) {
          this.advance(tick);
          step = this.steps[this.at];
          continue;
        }
        break;
      }
      if (!this.reached(self, this.at)) break;
      this.advance(tick);
      step = this.steps[this.at];
    }
    if (step === undefined) {
      this.state = "arrived";
      return out;
    }

    if (step.kind === "kill") {
      if (this.killTick < 0) {
        this.killWanted = true;
        this.killTick = tick;
        this.killDied = false;
        return out;
      }

      if (this.killDied) {
        this.state = "replan";
        this.reason = `respawned away from the spawn step ${this.at + 1}/${this.steps.length} planned`;
        return out;
      }
      if (tick - this.killTick > STALL_TICKS) {
        this.state = "stuck";
        this.reason = `the /kill at step ${this.at + 1}/${this.steps.length} never came`;
      }
      return out;
    }

    const tx = step.x * TILE_PX + 16;
    const ty = step.y * TILE_PX + 16;
    const dx = tx - self.pos.x;
    const dy = ty - self.pos.y;
    const d = Math.hypot(dx, dy);
    if (this.bestTick < 0) this.bestTick = tick;
    if (d < this.bestDist - 4) {
      this.bestDist = d;
      this.bestTick = tick;
    } else if (tick - this.bestTick > STALL_TICKS) {
      this.state = "stuck";
      this.reason = `stuck ${Math.round(d)}px from step ${this.at + 1}/${this.steps.length} (${step.kind})`;
      return out;
    }

    out.direction = dx > CENTRE_PX ? 1 : dx < -CENTRE_PX ? -1 : 0;

    if (step.kind === "hook" && step.anchorX !== undefined && step.anchorY !== undefined) {
      const ax = step.anchorX * TILE_PX + 16 - self.pos.x;
      const ay = step.anchorY * TILE_PX + 16 - self.pos.y;
      const n = Math.hypot(ax, ay) || 1;
      out.targetX = Math.round((ax / n) * 300);
      out.targetY = Math.round((ay / n) * 300);
      out.hook = 1;
      this.hookTicks++;
      if (this.hookTicks > HOOK_MAX_TICKS) {
        this.state = "stuck";
        this.reason = `the rope at step ${this.at + 1}/${this.steps.length} did not get us there`;
      }

      out.direction = ax > CENTRE_PX ? 1 : ax < -CENTRE_PX ? -1 : 0;
      return out;
    }

    out.targetX = Math.round((dx / (d || 1)) * 300);
    out.targetY = Math.round((dy / (d || 1)) * 300);

    const blocked = out.direction !== 0 && others !== undefined && teeInTheWay(self, out.direction, others) && this.hopClear(out.direction);
    if (step.kind === "jump" || step.leap === true || dy < -TILE_PX / 2 || blocked) {
      const rising = self.vel.y < -0.5;
      out.jump = rising ? 1 : this.decisions % 2 === 0 ? 1 : 0;
    }
    return out;
  }

  private hopClear(direction: number): boolean {
    const step = this.steps[this.at];
    if (step === undefined) return false;
    let x = step.x;
    for (let k = 1; k <= 2; k++) {
      const s = this.steps[this.at + k];
      if (s === undefined || s.y !== step.y || (s.x - x) * direction <= 0) return false;
      x = s.x;
    }
    const bx = x + direction;
    const next = this.steps[this.at + 3];
    if (next !== undefined && next.y === step.y && next.x === bx) return true;
    const g = this.grid;
    if (g === null || bx < 0 || bx >= g.width) return false;
    const hazard = (i: number): boolean => g.free[i] === 0 && g.solid[i] === 0;
    const i = step.y * g.width + bx;
    if (hazard(i)) return false;
    return !(step.y + 1 < g.height && g.free[i] === 1 && hazard(i + g.width));
  }
}

const IN_THE_WAY_PX = 44;

function teeInTheWay(self: TeeState, direction: number, others: readonly TeeState[]): boolean {
  for (const o of others) {
    if (o.id === self.id || !o.alive) continue;
    const ahead = (o.pos.x - self.pos.x) * direction;
    if (ahead > 0 && ahead < IN_THE_WAY_PX && Math.abs(o.pos.y - self.pos.y) < TILE_PX / 2) return true;
  }
  return false;
}

const ENTITY_SPAWN = 192;

export function routeField(collision: Collision, toX: number, toY: number): { width: number; height: number; dist: Int32Array } {
  const g = gridOf(collision);
  const n = g.width * g.height;
  const sc = scratchOf(collision, n);
  const { dist, stamp, heap } = sc;
  const gen = ++sc.gen;
  heap.clear();
  const gx = Math.min(g.width - 1, Math.max(0, Math.trunc(toX / TILE_PX)));
  const gy = Math.min(g.height - 1, Math.max(0, Math.trunc(toY / TILE_PX)));
  const start = gy * g.width + gx;
  dist[start] = 0;
  stamp[start] = gen;
  heap.push(start, 0);
  while (!heap.empty) {
    const { index: i, cost: popped } = heap.pop();
    const d = dist[i];
    if (stamp[i] !== gen || popped > d) continue;
    const x = i % g.width;
    const y = (i - x) / g.width;
    expandBack(g, x, y, (nx, ny, cost) => {
      if (nx < 0 || ny < 0 || nx >= g.width || ny >= g.height) return;
      const ni = ny * g.width + nx;
      if (g.free[ni] === 0) return;
      const nd = d + cost;
      if (stamp[ni] === gen && dist[ni] <= nd) return;
      stamp[ni] = gen;
      dist[ni] = nd;
      heap.push(ni, nd);
    });
  }

  const out = new Int32Array(n).fill(0x3fffffff);
  for (let i = 0; i < n; i++) if (stamp[i] === gen) out[i] = dist[i];
  return { width: g.width, height: g.height, dist: out };
}

export function spawnTiles(collision: Collision): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = [];
  const front = collision.frontIndex;
  for (let i = 0; i < collision.tiles.length; i++) {
    if (collision.tiles[i] !== ENTITY_SPAWN && (front === undefined || front[i] !== ENTITY_SPAWN)) continue;
    out.push({ x: (i % collision.width) * TILE_PX + TILE_PX / 2, y: Math.trunc(i / collision.width) * TILE_PX + TILE_PX / 2 });
  }
  return out;
}

const deadCache = new WeakMap<Collision, Uint8Array>();

export function deadZoneOf(collision: Collision): Uint8Array | null {
  const cached = deadCache.get(collision);
  if (cached !== undefined) return cached.length === 0 ? null : cached;
  const spawns = spawnTiles(collision);
  const cells = spawns.length === 0 ? new Uint8Array(0) : deadZone(collision, spawns);
  deadCache.set(collision, cells);
  return cells.length === 0 ? null : cells;
}

export function deadZone(collision: Collision, spawns: readonly { x: number; y: number }[]): Uint8Array {
  const g = gridOf(collision);
  const n = g.width * g.height;
  const queue = new Int32Array(n);

  const flood = (back: boolean): Uint8Array => {
    const seen = new Uint8Array(n);
    let head = 0;
    let tail = 0;
    const mark = (x: number, y: number): void => {
      if (x < 0 || y < 0 || x >= g.width || y >= g.height) return;
      const i = y * g.width + x;
      if (seen[i] === 1) return;
      seen[i] = 1;
      queue[tail++] = i;
    };
    for (const s of spawns) mark(Math.trunc(s.x / TILE_PX), Math.trunc(s.y / TILE_PX));
    while (head < tail) {
      const i = queue[head++];
      const x = i % g.width;
      const y = (i - x) / g.width;
      if (back) expandBack(g, x, y, (nx, ny) => mark(nx, ny), true);
      else expand(g, x, y, (nx, ny) => mark(nx, ny), undefined, true, true);
    }
    return seen;
  };

  const canReturn = flood(true);
  const canGetThere = flood(false);
  const dead = new Uint8Array(n);
  for (let i = 0; i < n; i++) dead[i] = g.free[i] === 1 && canGetThere[i] === 1 && canReturn[i] === 0 ? 1 : 0;

  const seen = new Uint8Array(n);
  for (let start = 0; start < n; start++) {
    if (dead[start] === 0 || seen[start] === 1) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    seen[start] = 1;
    const part: number[] = [];
    while (head < tail) {
      const i = queue[head++];
      part.push(i);
      const x = i % g.width;
      const y = (i - x) / g.width;
      for (const [ox, oy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
        const nx = x + ox;
        const ny = y + oy;
        if (nx < 0 || ny < 0 || nx >= g.width || ny >= g.height) continue;
        const j = ny * g.width + nx;
        if (dead[j] === 0 || seen[j] === 1) continue;
        seen[j] = 1;
        queue[tail++] = j;
      }
    }
    if (part.length > MAX_TRAP_TILES) for (const i of part) dead[i] = 0;
  }
  return dead;
}
