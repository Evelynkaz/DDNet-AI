import type { Collision } from "../core/collision.ts";
import { SimWorld } from "../core/world.ts";
import type { PlayerInput, TeeState } from "../core/types.ts";
import { emptyInput } from "../core/types.ts";
import { HOOK_GRABBED } from "../core/characterCore.ts";
import { TUNING } from "../core/tuning.ts";

export type TileBox = { x0: number; y0: number; x1: number; y1: number };

export function inBox(b: TileBox, tx: number, ty: number): boolean {
  return tx >= b.x0 && tx <= b.x1 && ty >= b.y0 && ty <= b.y1;
}

export function inAnyBox(boxes: readonly TileBox[], tx: number, ty: number): boolean {
  for (const b of boxes) if (inBox(b, tx, ty)) return true;
  return false;
}

export function shiftBox(b: TileBox, dx: number, dy: number): TileBox {
  return { x0: b.x0 + dx, y0: b.y0 + dy, x1: b.x1 + dx, y1: b.y1 + dy };
}

export type Crossing = {
  label: string;

  from: readonly TileBox[];

  chamber: TileBox;

  start: { tx: number; ty: number };

  anchors: readonly { tx: number; ty: number }[];

  landing: readonly TileBox[];

  exit: readonly TileBox[];

  exitTile: { tx: number; ty: number };

  hall?: readonly TileBox[];

  hallTile?: { tx: number; ty: number };

  toward: -1 | 1;
};

export function shiftCrossing(c: Crossing, dx: number, dy: number): Crossing {
  return {
    ...c,
    from: c.from.map((b) => shiftBox(b, dx, dy)),
    chamber: shiftBox(c.chamber, dx, dy),
    start: { tx: c.start.tx + dx, ty: c.start.ty + dy },
    anchors: c.anchors.map((a) => ({ tx: a.tx + dx, ty: a.ty + dy })),
    landing: c.landing.map((b) => shiftBox(b, dx, dy)),
    exit: c.exit.map((b) => shiftBox(b, dx, dy)),
    exitTile: { tx: c.exitTile.tx + dx, ty: c.exitTile.ty + dy },
    hall: c.hall?.map((b) => shiftBox(b, dx, dy)),
    hallTile: c.hallTile === undefined ? undefined : { tx: c.hallTile.tx + dx, ty: c.hallTile.ty + dy },
  };
}

const TILE_PX = 32;
const centre = (t: number): number => t * TILE_PX + TILE_PX / 2;
const tileOf = (px: number): number => Math.trunc(px / TILE_PX);

const HOLDS = [10, 16, 24, 40];

const PUSH_AFTER_TICKS = 30;

const SETTLE_TICKS = 110;

const HOP_RUNS = [10, 20, 40];
const HOP_JUMP_AT = [0, 2, 4, 6, 8, 10, 14];
const HOP_JUMP_HOLD = [6, 14];
const HOP_TICKS = 90;

const DROP_RUNS = [0, 6, 12, 24];
const DROP_TICKS = 260;

const AIR_RUNS = [4, 10, 20, 40];
const AIR_JUMP_AT = [-1, 0, 3];
const MAX_HOPS = 3;

const SPREAD_TICKS = 2;

const SPREAD_EARLY_TICKS = 1;

const NUDGE_PX = 6;
const NUDGE_Y_PX = 4;

const HOP_CLEAR_PX = 6;

const ARRIVED_VX = 3;

const APPROACH_DEPTH_TILES = 3;

const APPROACH_GIVE_UP_TICKS = 150;

type Swing = { kind: "swing"; anchor: { tx: number; ty: number }; hold: number; dir: number };

type Hop = { kind: "hop"; what: string; dir: number; run: number; jumpAt: number; jumpHold: number; ticks: number };
type Program = (Swing | Hop) & { startTick: number; froze: boolean };

export type CrossPhase = "approach" | "swinging" | "hopping" | "arrived" | "failed";

export class SwingCrosser {
  readonly crossing: Crossing;
  private readonly collision: Collision;
  private sim: SimWorld | null = null;
  private program: Program | null = null;
  private approachDir = 0;
  private startTick = -1;
  private lastTick = -1;
  private cadence = 1;
  private hops = 0;
  private phaseValue: CrossPhase = "approach";
  private why = "";

  tried = 0;

  budgetMs = 0;
  private until = Infinity;

  private readonly laps = new Map<string, { at: number; tried: number }>();

  private outOfTime = false;
  stops = 0;

  constructor(collision: Collision, crossing: Crossing) {
    this.collision = collision;
    this.crossing = crossing;
  }

  get phase(): CrossPhase {
    return this.phaseValue;
  }

  get done(): boolean {
    return this.phaseValue === "arrived" || this.phaseValue === "failed";
  }

  get reason(): string {
    return this.why;
  }

  get doing(): string {
    const p = this.program;
    if (p === null) return this.phaseValue;
    if (p.kind === "hop") {
      const push = p.dir === 0 || p.run === 0 ? "" : ` pushing ${p.dir === this.crossing.toward ? "on" : "back"} ${p.run} ticks`;
      return `${p.what}${push}${p.jumpAt < 0 ? "" : `, jump at ${p.jumpAt} for ${p.jumpHold}`}`;
    }
    return `rope on (${p.anchor.tx},${p.anchor.ty}) for ${p.hold} ticks${p.dir === 0 ? ", swinging free" : ", pushing on"}`;
  }

  private arrivedAt(self: TeeState): boolean {
    if (self.frozen) return false;
    const hall = this.crossing.hall;
    if (hall !== undefined) return Math.abs(self.vel.x) <= 2 && this.supported(self) && inAnyBox(hall, tileOf(self.pos.x), tileOf(self.pos.y));
    return this.inPassage(self);
  }

  private inPassage(self: TeeState): boolean {
    return !self.frozen && Math.abs(self.vel.x) <= ARRIVED_VX && inAnyBox(this.crossing.exit, tileOf(self.pos.x), tileOf(self.pos.y));
  }

  private throughAt(self: TeeState): boolean {
    return this.arrivedAt(self) || this.inPassage(self);
  }

  private inFreeze(self: TeeState): boolean {
    for (const dx of [-14, 14]) for (const dy of [-14, 14]) if (this.collision.isFreeze(self.pos.x + dx, self.pos.y + dy)) return true;
    return false;
  }

  private nearFreeze(self: TeeState, px: number): boolean {
    const r = 14 + px;
    for (const dx of [-r, 0, r]) for (const dy of [-r, 0, r]) if (this.collision.isFreeze(self.pos.x + dx, self.pos.y + dy)) return true;
    return false;
  }

  private inChamberFloor(self: TeeState): boolean {
    const ch = this.crossing.chamber;
    const tx = tileOf(self.pos.x);
    return tx >= ch.x0 && tx <= ch.x1 && tileOf(self.pos.y) > ch.y1;
  }

  private landedAt(self: TeeState): boolean {
    return !self.frozen && Math.abs(self.vel.x) <= 1 && this.supported(self) && inAnyBox(this.crossing.landing, tileOf(self.pos.x), tileOf(self.pos.y));
  }

  step(self: TeeState, tick: number, lag = 0): PlayerInput {
    const input = emptyInput();
    if (this.done) return input;
    this.until = this.budgetMs > 0 ? Date.now() + this.budgetMs : Infinity;
    this.outOfTime = false;
    if (this.startTick < 0 || tick < this.startTick) this.startTick = tick;
    if (this.lastTick >= 0 && tick > this.lastTick) this.cadence = Math.min(4, tick - this.lastTick);
    this.lastTick = tick;
    if (!self.alive) return this.fail("died on the way", input);
    if (this.arrivedAt(self)) {
      this.phaseValue = "arrived";
      this.why = `through ${this.crossing.label}`;
      return input;
    }
    const tx = tileOf(self.pos.x);
    const ty = tileOf(self.pos.y);
    if (self.frozen) {

      if (Math.hypot(self.vel.x, self.vel.y) < 0.5 && this.supported(self) && (this.inFreeze(self) || this.inChamberFloor(self))) return this.fail(`lies frozen at (${tx},${ty})`, input);

      if (this.program !== null) {
        this.program.froze = true;
        return this.programInput(self, this.program, tick - this.program.startTick, input);
      }
      return input;
    }

    if (this.program !== null && !this.supported(self)) {
      const p = this.program;
      const t = tick - p.startTick;
      const roped = p.kind === "swing" && t < p.hold && self.hookState === HOOK_GRABBED;
      if (!roped && t > 0 && !this.works(self, p, t, lag)) {
        const fix = this.searchHop(self, lag, true);
        if (fix !== null) this.program = { ...fix, startTick: tick, froze: p.froze };
      }
    }
    if (this.program !== null) {
      const p = this.program;
      const t = tick - p.startTick;
      if (p.kind === "swing") {

        if (!p.froze && t > 12 && t < p.hold && self.hookState !== HOOK_GRABBED) this.program = null;

        else if (t >= p.hold && (this.landedAt(self) || this.inPassage(self))) this.program = null;
        else if (t > p.hold + PUSH_AFTER_TICKS + SETTLE_TICKS) return this.fail(`the swing ran out at (${tx},${ty})`, input);
        else if (t < p.hold + PUSH_AFTER_TICKS || !this.inFrom(tx, ty)) return this.programInput(self, p, t, input);

        else this.program = null;
      } else {
        const dropping = p.ticks > HOP_TICKS;
        if (t > 4 && !dropping && (this.landedAt(self) || this.inPassage(self))) this.program = null;
        else if (t > p.ticks) return this.fail(`the ${dropping ? "drop" : "hop"} ran out at (${tx},${ty})`, input);
        else return this.programInput(self, p, t, input);
      }
    }

    if (this.crossing.hall !== undefined && this.inPassage(self)) {
      const drop = this.searchDrop(self, lag);

      if (drop === null && this.outOfTime) return input;
      if (drop === null) {
        this.phaseValue = "arrived";
        this.why = `through ${this.crossing.label}, at the foot of the passage`;
        return input;
      }
      this.program = { ...drop, startTick: tick, froze: false };
      this.phaseValue = "hopping";
      return this.programInput(self, this.program, 0, input);
    }
    if (this.landedAt(self)) {
      if (this.hops >= MAX_HOPS) return this.fail(`${this.hops} hops and still at (${tx},${ty})`, input);
      const hop = this.searchHop(self, lag);
      if (hop === null && this.outOfTime) return input;
      if (hop === null) return this.fail(`no hop from (${tx},${ty}) that clears the freeze`, input);
      this.hops++;
      this.program = { ...hop, startTick: tick, froze: false };
      this.phaseValue = "hopping";
      return this.programInput(self, this.program, 0, input);
    }
    if (tick - this.startTick > APPROACH_GIVE_UP_TICKS) return this.fail("no swing through from where it got to", input);

    if (!this.nearFrom(tx, ty)) return this.fail(`off the start of it at (${tx},${ty})`, input);
    {

      const ch = this.crossing.chamber;
      const toward = this.crossing.toward;
      const nearEdge = toward < 0 ? ch.x0 : ch.x1;
      const depth = (tx - nearEdge) * -toward;
      if (tx < ch.x0 - 3 || tx > ch.x1 + 3) this.approachDir = tx < ch.x0 ? 1 : -1;
      else if (depth < APPROACH_DEPTH_TILES && (this.supported(self) || this.approachDir === -toward)) this.approachDir = -toward;
      else this.approachDir = toward;
    }
    const found = this.searchSwing(self, lag);
    if (found !== null) {
      this.program = { ...found, startTick: tick, froze: false };
      this.phaseValue = "swinging";
      return this.programInput(self, this.program, 0, input);
    }
    this.approachInput(input);
    return input;
  }

  private fail(why: string, input: PlayerInput): PlayerInput {
    this.phaseValue = "failed";
    this.why = why;
    this.program = null;
    return input;
  }

  private inFrom(tx: number, ty: number): boolean {
    return inAnyBox(this.crossing.from, tx, ty);
  }

  private nearFrom(tx: number, ty: number): boolean {
    for (const b of this.crossing.from) if (tx >= b.x0 - 2 && tx <= b.x1 + 2 && ty >= b.y0 - 2 && ty <= b.y1 + 2) return true;
    return false;
  }

  private supported(self: TeeState): boolean {
    return this.collision.isSolid(self.pos.x, self.pos.y + 17) || this.collision.isSolid(self.pos.x - 14, self.pos.y + 17) || this.collision.isSolid(self.pos.x + 14, self.pos.y + 17);
  }

  private approachInput(input: PlayerInput): PlayerInput {
    input.direction = this.approachDir;
    input.hook = 0;
    input.jump = 0;
    input.targetX = this.approachDir * 300;
    input.targetY = 0;
    return input;
  }

  private programInput(self: TeeState, p: Swing | Hop, t: number, input: PlayerInput): PlayerInput {
    const toward = this.crossing.toward;
    if (p.kind === "hop") {
      input.direction = t < p.run ? p.dir : 0;
      input.jump = p.jumpAt >= 0 && t >= p.jumpAt && t < p.jumpAt + p.jumpHold ? 1 : 0;
      input.hook = 0;
      input.targetX = toward * 300;
      input.targetY = 0;
      return input;
    }
    const holding = t < p.hold;
    input.hook = holding ? 1 : 0;
    input.jump = 0;
    input.direction = t < p.hold + PUSH_AFTER_TICKS ? p.dir : 0;
    input.targetX = holding ? Math.round(centre(p.anchor.tx) - self.pos.x) : toward * 300;
    input.targetY = holding ? Math.round(centre(p.anchor.ty) - self.pos.y) : 0;
    if (input.targetX === 0 && input.targetY === 0) input.targetY = -1;
    return input;
  }

  private simFrom(self: TeeState): SimWorld {
    if (this.sim === null) {
      this.sim = new SimWorld(this.collision, { respawnDelayTicks: 0 });
      this.sim.addTee(0, { x: self.pos.x, y: self.pos.y });
    }
    return this.sim;
  }

  private robust(self: TeeState, p: Swing | Hop, lag: number, done: (me: TeeState) => boolean, ticks: number, approaching: boolean): boolean {
    const sim = this.simFrom(self);

    const lags = [lag];
    for (let d = Math.max(0, lag - SPREAD_EARLY_TICKS); d <= lag + SPREAD_TICKS; d++) if (d !== lag) lags.push(d);
    for (const d of lags) {
      sim.applyTeeState(0, { ...self, id: 0 });
      if (!this.rollout(sim, p, d, done, ticks, approaching)) return false;
    }
    for (const [dx, dy] of [
      [-NUDGE_PX, 0],
      [NUDGE_PX, 0],
      [0, -NUDGE_Y_PX],
      [0, NUDGE_Y_PX],
    ]) {

      const x = self.pos.x + dx;
      const y = self.pos.y + dy;
      if (this.collision.isSolid(x - 14, y - 14) || this.collision.isSolid(x + 14, y - 14) || this.collision.isSolid(x - 14, y + 14) || this.collision.isSolid(x + 14, y + 14)) continue;
      sim.applyTeeState(0, { ...self, id: 0, pos: { x, y } });
      if (!this.rollout(sim, p, lag, done, ticks, approaching)) return false;
    }
    return true;
  }

  private firstThat<T>(kind: string, list: readonly T[], works: (p: T) => boolean): T | null {
    const n = list.length;
    const lap = this.laps.get(kind);
    this.laps.delete(kind);
    if (n === 0) return null;

    const from = lap === undefined ? 0 : lap.at % n;
    let tried = lap === undefined ? 0 : Math.min(lap.tried, n - 1);
    for (let k = 0; tried < n; k++, tried++) {
      const i = (from + k) % n;
      if (k > 0 && Date.now() > this.until) {
        this.laps.set(kind, { at: i, tried });
        this.outOfTime = true;
        this.stops++;
        return null;
      }
      if (works(list[i])) {
        this.laps.clear();
        return list[i];
      }
    }
    return null;
  }

  private searchSwing(self: TeeState, lag: number): Swing | null {
    const toward = this.crossing.toward;
    const done = (me: TeeState): boolean => this.throughAt(me) || this.landedAt(me);
    const list: Swing[] = [];
    for (const anchor of this.crossing.anchors) {

      const reach = Math.hypot(centre(anchor.tx) - self.pos.x, centre(anchor.ty) - self.pos.y);
      if (reach > TUNING.hookLength + (lag + SPREAD_TICKS) * 16) continue;
      for (const hold of HOLDS) for (const dir of [toward, 0]) list.push({ kind: "swing", anchor, hold, dir });
    }
    return this.firstThat("swing", list, (p) => this.robust(self, p, lag, done, lag + p.hold + SETTLE_TICKS, true));
  }

  private searchHop(self: TeeState, lag: number, inAir = false): Hop | null {
    const toward = this.crossing.toward;
    const from = self.pos.x;
    const done = inAir
      ? (me: TeeState): boolean => this.throughAt(me) || this.landedAt(me)
      : (me: TeeState): boolean => this.throughAt(me) || (this.landedAt(me) && (me.pos.x - from) * toward > TILE_PX);
    const dirs = inAir ? [toward, 0, -toward] : [toward];
    const runs = inAir ? AIR_RUNS : HOP_RUNS;
    const jumps = inAir ? AIR_JUMP_AT : HOP_JUMP_AT;

    const passes: ((jumpAt: number) => boolean)[] = inAir ? [(j) => j < 0, (j) => j >= 0] : [() => true];
    const list: Hop[] = [];
    for (const pass of passes) {
      for (const dir of dirs) {
        for (const run of runs) {
          for (const jumpAt of jumps) {
            if (!pass(jumpAt) || jumpAt >= run) continue;
            for (const jumpHold of jumpAt < 0 ? [0] : HOP_JUMP_HOLD) list.push({ kind: "hop", what: inAir ? "in the air" : "hop", dir, run, jumpAt, jumpHold, ticks: HOP_TICKS });
          }
        }
      }
    }
    return this.firstThat(inAir ? "air" : "hop", list, (p) => this.robust(self, p, lag, done, lag + HOP_TICKS, false));
  }

  private searchDrop(self: TeeState, lag: number): Hop | null {
    const done = (me: TeeState): boolean => this.arrivedAt(me);
    const toward = this.crossing.toward;
    const list: Hop[] = [];
    for (const run of DROP_RUNS) {
      for (const dir of run === 0 ? [0] : [-toward, toward]) list.push({ kind: "hop", what: "drop into the hall", dir, run, jumpAt: -1, jumpHold: 0, ticks: DROP_TICKS });
    }
    return this.firstThat("drop", list, (p) => this.robust(self, p, lag, done, lag + DROP_TICKS, false));
  }

  private works(self: TeeState, p: Swing | Hop, t: number, lag: number): boolean {
    const sim = this.simFrom(self);
    sim.applyTeeState(0, { ...self, id: 0 });
    const dropping = p.kind === "hop" && p.ticks > HOP_TICKS;
    const done = (me: TeeState): boolean => (dropping ? this.arrivedAt(me) : this.throughAt(me) || this.landedAt(me));
    const ticks = p.kind === "swing" ? Math.max(lag + 20, lag + p.hold + SETTLE_TICKS - t) : Math.max(lag + 20, lag + p.ticks - t);
    return this.rollout(sim, p, lag, done, ticks, false, t);
  }

  private rollout(sim: SimWorld, p: Swing | Hop, lag: number, done: (me: TeeState) => boolean, ticks: number, approaching: boolean, t0 = 0): boolean {
    this.tried++;
    const first = sim.getTee(0);
    let held = approaching ? this.approachInput(emptyInput()) : t0 > 0 && first !== undefined ? this.programInput(first, p, t0 - 1, emptyInput()) : emptyInput();
    const pending: { at: number; input: PlayerInput }[] = [];
    for (let t = 0; t < ticks; t++) {
      const me = sim.getTee(0);
      if (me === undefined || !me.alive) return false;
      if (t % this.cadence === 0) pending.push({ at: t + lag, input: this.programInput(me, p, t0 + t, emptyInput()) });
      while (pending.length > 0 && pending[0].at <= t) held = (pending.shift() as { input: PlayerInput }).input;
      sim.setInput(0, held);
      sim.step();
      const now = sim.getTee(0);
      if (now === undefined) return false;

      if (t > lag + 4 && done(now)) return true;

      if (t > lag + 8 && now.frozen && (this.inChamberFloor(now) || (Math.hypot(now.vel.x, now.vel.y) < 0.3 && this.inFreeze(now)))) return false;

      if (p.kind === "hop" && p.ticks <= HOP_TICKS && t > lag && (now.frozen || (p.what === "in the air" && this.nearFreeze(now, HOP_CLEAR_PX)))) return false;
      if (now.frozen && t > lag) {

      }
    }
    return false;
  }
}
