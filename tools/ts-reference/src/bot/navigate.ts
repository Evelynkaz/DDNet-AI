import type { Collision } from "../core/collision.ts";
import type { PlayerInput, TeeState } from "../core/types.ts";
import { emptyInput } from "../core/types.ts";
import type { Vec2 } from "../core/vmath.ts";
import { vdistance } from "../core/vmath.ts";
import type { HazardField } from "../plan/planner.ts";
import { travelField } from "../plan/planner.ts";
import { findRoute, RouteRunner } from "../plan/route.ts";
import { TUNING } from "../core/tuning.ts";
import type { Crossing } from "./crossing.ts";
import { inAnyBox, SwingCrosser } from "./crossing.ts";

const TILE_PX = 32;

const UNREACHABLE = 0x3fffffff;

const LOOKAHEAD_TILES = 4;

const CENTRE_PX = 8;

export const GROUND_JUMP_RISE_PX = 182;
export const DOUBLE_JUMP_RISE_PX = 330;

const RISING_VEL = -0.1;

const STALL_TICKS = 150;

const PROBE_TICKS = 100;

const TELEPORT_JUMP_PX = 96;

const MAX_HONEST_PX_PER_TICK = 12;

const MAX_TELE_GOALS = 8;

const MAX_ROUTE_REPLANS = 3;

const MAX_ROUTE_DROPS = 3;

const MAX_ALTERNATIVE_ROUTES = 4;

const MAX_CROSS_TRIES = 4;

const PLANNED_FREEZE_STEPS = 16;

const CLIMB_MIN_RISE_TILES = 3;
const CLIMB_ARC_RAYS = 13;
const CLIMB_RAY_STEP_PX = 12;
const CLIMB_GIVE_UP_TICKS = 120;
const CLIMB_ARRIVE_PX = 40;

const CLIMB_BRAKE_TICKS = 6;

const WALK_BRAKE_TICKS = 3;
const HOOK_LENGTH = TUNING.hookLength;

export type NavGoal = {
  tx: number;
  ty: number;

  label: string;

  tele: { type: number; number: number } | null;
};

export type NavPhase = "walking" | "probing" | "arrived" | "blocked";

function tileOf(px: number): number {
  return Math.trunc(px / TILE_PX);
}

function centreOf(tile: number): number {
  return tile * TILE_PX + TILE_PX / 2;
}

function distAt(field: HazardField, tx: number, ty: number): number {
  if (tx < 0 || ty < 0 || tx >= field.width || ty >= field.height) return UNREACHABLE;
  return field.dist[ty * field.width + tx];
}

function fieldTo(collision: Collision, tx: number, ty: number): HazardField {
  const live = travelField(collision, centreOf(tx), centreOf(ty));
  return { width: live.width, height: live.height, dist: new Int32Array(live.dist) };
}

const NEIGHBOURS: readonly (readonly [number, number])[] = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

export function traceRoute(field: HazardField, tx: number, ty: number, maxSteps: number): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = [];
  let cx = tx;
  let cy = ty;
  let d = distAt(field, cx, cy);
  if (d >= UNREACHABLE) return out;
  let lastDx = 0;
  let lastDy = 0;
  for (let step = 0; step < maxSteps && d > 0; step++) {
    let bestX = -1;
    let bestY = -1;
    let bestScore = -1;
    for (const [dx, dy] of NEIGHBOURS) {
      const nd = distAt(field, cx + dx, cy + dy);
      if (nd >= d) continue;
      const score = (dx === lastDx && dy === lastDy ? 4 : 0) + (dy === 0 ? 2 : 0) + (dy > 0 ? 1 : 0);
      if (score > bestScore) {
        bestScore = score;
        bestX = cx + dx;
        bestY = cy + dy;
      }
    }
    if (bestScore < 0) break;
    lastDx = bestX - cx;
    lastDy = bestY - cy;
    cx = bestX;
    cy = bestY;
    d = distAt(field, cx, cy);
    out.push({ x: cx, y: cy });
  }
  return out;
}

export function teleGoals(collision: Collision, fromX: number, fromY: number, limit = MAX_TELE_GOALS): NavGoal[] {
  if (!collision.hasTele()) return [];
  const types = collision.teleType;
  const numbers = collision.teleNumber;
  if (types === undefined) return [];

  const field = travelField(collision, fromX, fromY);
  const best = new Map<string, { goal: NavGoal; dist: number }>();
  for (let i = 0; i < types.length; i++) {
    const type = types[i];
    if (type === 0) continue;
    const tx = i % collision.width;
    const ty = (i - tx) / collision.width;
    const d = distAt(field, tx, ty);
    if (d >= UNREACHABLE) continue;
    const num = numbers === undefined ? 0 : numbers[i];
    const key = `${type}/${num}`;
    const prev = best.get(key);
    if (prev !== undefined && prev.dist <= d) continue;
    best.set(key, {
      dist: d,
      goal: { tx, ty, label: `teleporter ${type}#${num} at (${tx},${ty})`, tele: { type, number: num } },
    });
  }
  return [...best.values()]
    .sort((a, b) => a.dist - b.dist)
    .slice(0, limit)
    .map((e) => e.goal);
}

export function tileGoal(collision: Collision, tx: number, ty: number): NavGoal {
  const t = collision.teleAt(centreOf(tx), centreOf(ty));
  return {
    tx,
    ty,
    label: `(${tx},${ty})`,
    tele: t.type === 0 ? null : { type: t.type, number: t.number },
  };
}

export class Navigator {
  readonly goals: readonly NavGoal[];
  readonly collision: Collision;

  private readonly stallTicks: number;
  private readonly probeTicks: number;
  private index = 0;
  private field: HazardField | null = null;
  private phaseValue: NavPhase = "walking";
  private reason = "";
  private readonly notes: string[] = [];

  private windowBest = Infinity;
  private windowRef = Infinity;
  private windowStart = 0;
  private probeUntil = -1;
  private lastPos: Vec2 | null = null;
  private aim = 0;
  private climbAnchor: Vec2 | null = null;
  private climbStart = -1;
  private climbBestY = Infinity;
  private startTick = -1;
  private lastTick = 0;
  private steps = 0;

  private runner: RouteRunner | null = null;
  private replans = 0;

  private drops = 0;
  private dropGiveUp: string | null = null;

  private avoid = new Set<number>();
  private alternatives = 0;
  private walkRouted = false;

  private killWanted = false;

  private readonly crossings: readonly Crossing[];

  crossBudgetMs = 0;
  private toCrossing: Crossing | null = null;
  private crosser: SwingCrosser | null = null;
  private crossTries = 0;

  private crossingReach = new Map<Crossing, boolean>();
  private lag = 0;

  private readonly throughFreeze: boolean;

  constructor(
    collision: Collision,
    goals: readonly NavGoal[],
    opts?: { stallTicks?: number; probeTicks?: number; throughFreeze?: boolean; crossings?: readonly Crossing[] },
  ) {
    this.collision = collision;
    this.goals = goals;
    this.throughFreeze = opts?.throughFreeze !== false;

    this.crossings = this.throughFreeze ? (opts?.crossings ?? []) : [];
    this.stallTicks = opts?.stallTicks ?? STALL_TICKS;
    this.probeTicks = opts?.probeTicks ?? PROBE_TICKS;
    if (goals.length === 0) {
      this.phaseValue = "blocked";
      this.reason = "nowhere to go";
    }
  }

  get phase(): NavPhase {
    return this.phaseValue;
  }

  get done(): boolean {
    return this.phaseValue === "arrived" || this.phaseValue === "blocked";
  }

  get outcome(): string {
    return this.reason;
  }

  get goal(): NavGoal | null {
    return this.goals[this.index] ?? null;
  }

  takeNotes(): string[] {
    if (this.notes.length === 0) return [];
    return this.notes.splice(0, this.notes.length);
  }

  takeKill(): boolean {
    const k = this.killWanted;
    this.killWanted = false;
    return k;
  }

  vetoed(): void {
    this.runner?.vetoed();
  }

  progress(self: TeeState | null): string {
    const goal = this.goal;
    if (goal === null) return this.reason.length > 0 ? this.reason : "nowhere to go";
    const left = self === null || this.field === null ? -1 : distAt(this.field, tileOf(self.pos.x), tileOf(self.pos.y));
    const far = left < 0 ? "" : left >= UNREACHABLE ? ", no route from where it is standing" : `, ${left} tiles to go`;
    const which = this.goals.length > 1 ? ` (candidate ${this.index + 1} of ${this.goals.length})` : "";
    return `${this.phaseValue === "probing" ? "standing on" : "walking to"} ${goal.label}${far}${which}`;
  }

  brief(self: TeeState | null, name?: string): string {
    const goal = this.goal;
    if (goal === null) return "goto ✗";
    const left = self === null || this.field === null ? -1 : distAt(this.field, tileOf(self.pos.x), tileOf(self.pos.y));
    const far = left < 0 ? "" : left >= UNREACHABLE ? " ??" : ` ${left}t`;
    return `${this.phaseValue === "probing" ? "on" : "goto"} ${name ?? `(${goal.tx},${goal.ty})`}${far}`;
  }

  tilesLeft(self: TeeState | null): number {
    if (self === null || this.field === null) return -1;
    const left = distAt(this.field, tileOf(self.pos.x), tileOf(self.pos.y));
    return left >= UNREACHABLE ? -1 : left;
  }

  private note(text: string): void {
    this.notes.push(text);
  }

  private finish(phase: "arrived" | "blocked", reason: string): void {
    this.phaseValue = phase;
    this.reason = reason;
    this.note(reason);
  }

  private nextGoal(why: string, tick: number): void {
    this.note(why);
    this.index++;
    this.field = null;
    this.runner = null;
    this.replans = 0;
    this.drops = 0;
    this.dropGiveUp = null;
    this.avoid = new Set();
    this.alternatives = 0;
    this.walkRouted = false;
    this.toCrossing = null;
    this.crosser = null;
    this.crossTries = 0;
    this.crossingReach = new Map();
    this.windowBest = Infinity;
    this.windowRef = Infinity;
    this.windowStart = tick;
    this.probeUntil = -1;
    if (this.index >= this.goals.length) {

      this.phaseValue = "blocked";
      this.reason = this.goals.length > 1 ? `${why}; nothing else to try` : why;
      if (this.goals.length > 1) this.note("no route to any of them");
    }
  }

  step(self: TeeState, tick: number, others?: readonly TeeState[], lag = 0): PlayerInput {
    this.lag = lag;
    const sinceLast = this.startTick < 0 ? 1 : Math.max(1, tick - this.lastTick);
    this.lastTick = tick;
    this.steps++;
    if (this.startTick < 0) {
      this.startTick = tick;
      this.windowStart = tick;
    }

    if (tick < this.windowStart) this.windowStart = tick;

    if (this.climbStart > tick) this.climbStart = tick;
    if (this.probeUntil > tick + this.probeTicks) this.probeUntil = tick + this.probeTicks;

    const was = this.lastPos;
    const moved = was !== null ? vdistance(was, self.pos) : 0;
    this.lastPos = { x: self.pos.x, y: self.pos.y };
    if (was !== null && moved >= Math.max(TELEPORT_JUMP_PX, sinceLast * MAX_HONEST_PX_PER_TICK) && !this.done) {
      const goal = this.goal;
      const here = `tile (${tileOf(self.pos.x)},${tileOf(self.pos.y)})`;
      const fromGoal = goal === null ? Infinity : vdistance(was, { x: centreOf(goal.tx), y: centreOf(goal.ty) });
      if (goal !== null && goal.tele !== null && fromGoal <= TILE_PX * 1.5) {
        this.finish("arrived", `teleported to ${here} -- type ${goal.tele.type} is an entrance`);
        return emptyInput();
      }
      this.windowBest = Infinity;
      this.windowRef = Infinity;
      this.windowStart = tick;
      if (this.runner?.awaitingKill === true) {

        this.runner.respawned();
        this.note(`respawned at ${here}`);
      } else if (this.crosser !== null) {

        this.crosser = null;
        this.field = null;
        this.note(`moved to ${here} in the middle of the swing; starting over from here`);
      } else if (this.runner?.current?.tele === true) {

        this.note(`teleported to ${here}`);
      } else if (this.runner !== null) {

        this.dropRoute(`moved to ${here} off it`);
        this.note(`something moved us to ${here} (not the doorway we were walking to); planning again from here`);
      } else {
        this.note(`something moved us to ${here} (not the doorway we were walking to); carrying on`);
      }
    }
    if (this.done) return emptyInput();

    const goal = this.goal;
    if (goal === null) {
      this.finish("blocked", "nowhere to go");
      return emptyInput();
    }
    if (this.dropGiveUp !== null) {
      this.nextGoal(`route to ${goal.label} broke off: ${this.dropGiveUp}`, tick);
      return emptyInput();
    }

    const crossed = this.stepCrossing(self, tick, goal);
    if (crossed !== null) return crossed;

    if (this.field === null && self.frozen) return emptyInput();

    if (this.field === null) {
      this.field = fieldTo(this.collision, goal.tx, goal.ty);
      this.windowBest = Infinity;
      this.windowRef = Infinity;
      this.windowStart = tick;
      const here = distAt(this.field, tileOf(self.pos.x), tileOf(self.pos.y));

      if (here >= UNREACHABLE || this.walkRouted) {

        const route = findRoute(this.collision, self.pos, { x: centreOf(goal.tx), y: centreOf(goal.ty) }, { nearTiles: 2, allowKill: true, throughFreeze: this.throughFreeze, avoid: this.avoid });
        if (route !== null && route.steps.length > 0) {
          this.runner = new RouteRunner(route.steps, this.collision, { earlyFreeze: this.crossings.length > 0 });
          const hooks = route.steps.filter((s) => s.kind === "hook").length;
          this.note(`no walk to ${goal.label}; going by route: ${route.steps.length} steps${hooks > 0 ? `, ${hooks} on the rope` : ""}`);
        } else if (this.startCrossing(self, goal)) {
          return emptyInput();
        } else {

          this.nextGoal(`no route to ${goal.label}: it is walled off from here`, tick);
          return emptyInput();
        }
      } else {
        this.note(`heading for ${goal.label}, ${here} tiles away`);
      }
    }

    const runner = this.runner;
    if (runner !== null) {
      if (runner.state === "running") {
        const out = runner.step(self, tick, others);
        if (runner.takeKill()) this.killWanted = true;
        this.steps++;
        return out;
      }
      this.runner = null;
      if (runner.state === "arrived") {

        if (this.toCrossing !== null) {
          this.field = null;
          return emptyInput();
        }
        this.finish("arrived", `walked the route to ${goal.label}`);
        return emptyInput();
      }

      if (runner.state === "replan" && this.replans < MAX_ROUTE_REPLANS) {
        this.replans++;
        this.note(`route to ${goal.label}: ${runner.reason}; planning again from here`);
        this.field = null;
        return emptyInput();
      }

      const failed = runner.failedMove;
      if (failed !== undefined && this.alternatives < MAX_ALTERNATIVE_ROUTES) {
        this.alternatives++;
        this.avoid.add(failed);
        this.note(`route to ${goal.label}: ${runner.reason}; looking for another way (${this.alternatives}/${MAX_ALTERNATIVE_ROUTES})`);
        this.field = null;
        return emptyInput();
      }
      this.nextGoal(`route to ${goal.label} broke off: ${runner.reason}`, tick);
      return emptyInput();
    }

    const field = this.field;
    const tx = tileOf(self.pos.x);
    const ty = tileOf(self.pos.y);
    const here = distAt(field, tx, ty);

    if (tx === goal.tx && ty === goal.ty) {
      if (goal.tele === null) {
        this.finish("arrived", `arrived at ${goal.label}`);
        return emptyInput();
      }

      if (this.probeUntil < 0) {
        this.probeUntil = tick + this.probeTicks;
        this.phaseValue = "probing";
        this.note(`standing on ${goal.label} to see whether it moves us`);
      }
    }

    if (this.probeUntil >= 0 && tick >= this.probeUntil) {
      this.phaseValue = "walking";
      const dead = goal.label;
      if (this.index + 1 >= this.goals.length) {
        this.finish("arrived", `reached ${dead}; it did not move us, and there is no other doorway from here`);
        return emptyInput();
      }
      this.nextGoal(`${dead} did not move us; trying the next one`, tick);
      return emptyInput();
    }

    if (here < this.windowBest) this.windowBest = here;
    if (this.probeUntil < 0 && tick - this.windowStart >= this.stallTicks) {
      if (this.windowBest >= this.windowRef && !this.walkRouted) {

        this.walkRouted = true;
        const route = findRoute(this.collision, self.pos, { x: centreOf(goal.tx), y: centreOf(goal.ty) }, { nearTiles: 2, allowKill: true, throughFreeze: this.throughFreeze, avoid: this.avoid });
        if (route !== null && route.steps.length > 0) {
          this.runner = new RouteRunner(route.steps, this.collision, { earlyFreeze: this.crossings.length > 0 });
          const hooks = route.steps.filter((st) => st.kind === "hook").length;
          this.note(`walking to ${goal.label} stalled ${here >= UNREACHABLE ? "off the flood" : `${here} tiles away`}; going by route: ${route.steps.length} steps${hooks > 0 ? `, ${hooks} on the rope` : ""}`);
          return emptyInput();
        }
      }
      if (this.windowBest >= this.windowRef) {
        this.nextGoal(
          `stuck ${here >= UNREACHABLE ? "off the route" : `${here} tiles from ${goal.label}`}: ` +
            `no closer in ${((tick - this.windowStart) / 50).toFixed(0)}s -- the rest of that route needs more than walking`,
          tick,
        );
        return emptyInput();
      }
      this.windowRef = this.windowBest;
      this.windowBest = here;
      this.windowStart = tick;
    }

    return this.follow(self, field, tx, ty, here, tick);
  }

  private follow(self: TeeState, field: HazardField, tx: number, ty: number, here: number, tick: number): PlayerInput {
    const input = emptyInput();

    let route = traceRoute(field, tx, ty, LOOKAHEAD_TILES);
    if (here >= UNREACHABLE || route.length === 0) {

      const rescue = this.nearestOnRoute(field, tx, ty);
      if (rescue === null) return input;
      route = [rescue];
    }

    const wp = route[Math.min(LOOKAHEAD_TILES, route.length) - 1];
    const next = route[0];

    const wantX = centreOf(wp.x);
    let direction = 0;
    if (self.pos.x < wantX - CENTRE_PX) direction = 1;
    else if (self.pos.x > wantX + CENTRE_PX) direction = -1;

    let wantUp = next.y < ty;

    const brakePx = 24 + Math.max(0, self.vel.x * direction) * (this.lag + WALK_BRAKE_TICKS);
    if (direction !== 0 && this.hazardWithin(self, direction, brakePx)) {
      direction = self.vel.x * direction > 0.5 ? -direction : 0;
      wantUp = false;
    }

    const riseTiles = (ty - route[route.length - 1].y);
    const wantClimb = riseTiles >= CLIMB_MIN_RISE_TILES && !this.hazardAhead(self, direction);

    if (this.climbAnchor !== null || wantClimb) {
      const climb = this.climb(self, tick, direction, wantX);
      if (climb !== null) {
        this.aimAt(climb.aimX, climb.aimY, self, input);
        input.direction = climb.direction;
        input.jump = climb.jump ? 1 : 0;
        input.hook = climb.hook ? 1 : 0;
        return input;
      }
    }

    input.direction = direction;
    input.jump = wantUp && (self.vel.y < RISING_VEL || this.steps % 2 === 0) ? 1 : 0;
    input.hook = 0;
    input.fire = 0;

    this.aimAt(wantX, centreOf(wp.y), self, input);
    return input;
  }

  private aimAt(x: number, y: number, self: TeeState, input: PlayerInput): void {
    const want = Math.atan2(y - self.pos.y, x - self.pos.x);
    let d = want - this.aim;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    this.aim += Math.max(-0.12, Math.min(0.12, d));
    const ax = Math.round(Math.cos(this.aim) * 300);
    const ay = Math.round(Math.sin(this.aim) * 300);
    input.targetX = ax === 0 && ay === 0 ? 300 : ax;
    input.targetY = ay;
  }

  private climb(
    self: TeeState,
    tick: number,
    direction: number,
    wantX: number,
  ): { aimX: number; aimY: number; direction: number; jump: boolean; hook: boolean } | null {
    if (this.climbAnchor === null) {
      const anchor = this.findAnchor(self, direction);
      if (anchor === null) return null;
      this.climbAnchor = anchor;
      this.climbStart = tick;
      this.climbBestY = self.pos.y;
      this.note(`climbing to (${Math.round(anchor.x / TILE_PX)}, ${Math.round(anchor.y / TILE_PX)})`);
    }
    const anchor = this.climbAnchor;

    if (self.pos.y < this.climbBestY - 1) this.climbBestY = self.pos.y;
    const stalled = tick - this.climbStart > CLIMB_GIVE_UP_TICKS;
    const arrived = self.pos.y <= anchor.y + CLIMB_ARRIVE_PX;
    if (arrived || stalled) {
      this.climbAnchor = null;
      if (stalled) this.note("climb stalled, back to walking");

      return { aimX: wantX, aimY: self.pos.y - 200, direction, jump: arrived, hook: false };
    }

    const lean = direction !== 0 && this.hazardWithin(self, direction, 24 + Math.abs(self.vel.x) * CLIMB_BRAKE_TICKS) ? (self.vel.x * direction > 0.5 ? -direction : 0) : direction;
    return { aimX: anchor.x, aimY: anchor.y, direction: lean, jump: false, hook: true };
  }

  private findAnchor(self: TeeState, direction: number): Vec2 | null {
    let best: Vec2 | null = null;
    let bestScore = -Infinity;
    for (let r = 0; r < CLIMB_ARC_RAYS; r++) {

      const angle = -Math.PI + (Math.PI * (r + 0.5)) / CLIMB_ARC_RAYS;
      const dx = Math.cos(angle);
      const dy = Math.sin(angle);
      for (let t = TILE_PX; t <= HOOK_LENGTH; t += CLIMB_RAY_STEP_PX) {
        const x = self.pos.x + dx * t;
        const y = self.pos.y + dy * t;
        if (!this.collision.isSolid(x, y)) continue;
        if (this.collision.isNoHook(x, y)) break;
        const rise = self.pos.y - y;
        if (rise < TILE_PX) break;

        const score = rise + (direction !== 0 && Math.sign(x - self.pos.x) === direction ? 40 : 0);
        if (score > bestScore) {
          bestScore = score;
          best = { x, y };
        }
        break;
      }
    }
    return best;
  }

  private hazardAhead(self: TeeState, direction: number): boolean {
    return this.hazardWithin(self, direction, 24);
  }

  private hazardWithin(self: TeeState, direction: number, px: number): boolean {
    const offsets: number[] = [];
    for (let a = Math.min(24, px); a < px; a += TILE_PX) offsets.push(a);
    offsets.push(px);
    for (const ahead of offsets) {
      const x = self.pos.x + direction * ahead;
      if (this.collision.isSolid(x, self.pos.y)) return false;
      for (let dy = 0; dy <= 4; dy++) {
        const y = self.pos.y + dy * TILE_PX;
        if (this.collision.isFreeze(x, y) || this.collision.isDeath(x, y)) return true;
        if (this.collision.isSolid(x, y)) break;
      }
    }
    return false;
  }

  private stepCrossing(self: TeeState, tick: number, goal: NavGoal): PlayerInput | null {
    const c = this.toCrossing;
    if (c === null) return null;
    if (this.crosser === null) {
      if (self.frozen || !inAnyBox(c.from, tileOf(self.pos.x), tileOf(self.pos.y))) return null;
      this.runner = null;
      this.crosser = new SwingCrosser(this.collision, c);
      this.note(`at the start of ${c.label}: swinging through on the rope (try ${this.crossTries + 1} of ${MAX_CROSS_TRIES})`);
    }
    const crosser = this.crosser;
    crosser.budgetMs = this.crossBudgetMs;
    const was = crosser.doing;
    const out = crosser.step(self, tick, this.lag);
    if (crosser.doing !== was && !crosser.done && crosser.phase !== "approach") this.note(`${c.label}: ${crosser.doing} (lag ${this.lag})`);
    if (crosser.phase === "arrived") {
      this.note(`${crosser.reason}; on to ${goal.label}`);
      this.crosser = null;
      this.toCrossing = null;
      this.field = null;
      this.runner = null;
      this.walkRouted = false;
      this.windowBest = Infinity;
      this.windowRef = Infinity;
      this.windowStart = tick;
      return emptyInput();
    }
    if (crosser.phase === "failed") {
      this.crosser = null;
      this.crossTries++;
      if (this.crossTries >= MAX_CROSS_TRIES) {
        this.nextGoal(`no way through ${c.label} to ${goal.label}: ${crosser.reason}, ${this.crossTries} times`, tick);
        return emptyInput();
      }

      this.note(`${c.label}: ${crosser.reason}; trying again from the spawn`);
      this.field = null;
      return emptyInput();
    }
    this.windowStart = tick;
    return out;
  }

  private startCrossing(self: TeeState, goal: NavGoal): boolean {
    if (this.crossings.length === 0) return false;
    const tx = tileOf(self.pos.x);
    const ty = tileOf(self.pos.y);
    let best: Crossing | null = null;
    let bestRoute: ReturnType<typeof findRoute> = null;
    for (const c of this.crossings) {
      let reach = this.crossingReach.get(c);
      if (reach === undefined) {
        const far = c.hallTile ?? c.exitTile;
        const on = findRoute(this.collision, { x: centreOf(far.tx), y: centreOf(far.ty) }, { x: centreOf(goal.tx), y: centreOf(goal.ty) }, { nearTiles: 2, throughFreeze: this.throughFreeze });
        reach = on !== null;
        this.crossingReach.set(c, reach);
      }
      if (!reach) continue;
      if (inAnyBox(c.from, tx, ty)) {
        best = c;
        bestRoute = null;
        break;
      }
      const way = findRoute(this.collision, self.pos, { x: centreOf(c.start.tx), y: centreOf(c.start.ty) }, { nearTiles: 1, allowKill: true, throughFreeze: this.throughFreeze, avoid: this.avoid });
      if (way === null || way.steps.length === 0) continue;
      if (bestRoute === null || way.cost < bestRoute.cost) {
        best = c;
        bestRoute = way;
      }
    }
    if (best === null) return false;
    this.toCrossing = best;
    if (bestRoute !== null) {
      this.runner = new RouteRunner(bestRoute.steps, this.collision, { earlyFreeze: this.crossings.length > 0 });
      this.note(`no way to ${goal.label} but through ${best.label}; going to its start first: ${bestRoute.steps.length} steps`);
    }
    return true;
  }

  private nearestOnRoute(field: HazardField, tx: number, ty: number): { x: number; y: number } | null {
    let best: { x: number; y: number } | null = null;
    let bestD = UNREACHABLE;
    for (let dy = -3; dy <= 3; dy++) {
      for (let dx = -3; dx <= 3; dx++) {
        const d = distAt(field, tx + dx, ty + dy);
        if (d < bestD) {
          bestD = d;
          best = { x: tx + dx, y: ty + dy };
        }
      }
    }
    return best;
  }

  respawned(): void {

    this.crosser = null;

    if (this.runner?.awaitingKill === true) this.runner.respawned();
    else if (this.runner !== null) this.dropRoute("died on it");
    this.climbAnchor = null;
    this.climbStart = -1;
    this.climbBestY = Infinity;
    this.lastPos = null;
    this.windowBest = Infinity;
    this.windowRef = Infinity;
    this.windowStart = this.lastTick;
  }

  private dropRoute(why: string): void {
    this.runner = null;
    this.field = null;
    if (++this.drops > MAX_ROUTE_DROPS) this.dropGiveUp = `${why} ${this.drops} times`;
  }

  cancel(why: string): void {
    this.climbAnchor = null;
    this.climbStart = -1;
    this.climbBestY = Infinity;
    if (this.done) return;
    this.finish("blocked", why);
  }

  get crossing(): boolean {
    return this.crosser !== null;
  }

  get plannedFreeze(): boolean {
    return this.crossings.length > 0 && this.runner !== null && this.runner.freezeAhead(PLANNED_FREEZE_STEPS);
  }

  get elapsedTicks(): number {
    return this.startTick < 0 ? 0 : this.lastTick - this.startTick;
  }
}
