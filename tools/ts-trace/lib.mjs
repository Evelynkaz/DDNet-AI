// Shared code for the trace-ts v1 generator (task 1.9, docs/formats.md "trace-ts v1"). Runs the
// REAL, unmodified `tools/ts-reference/src/core/*.ts`/`.../src/map/loadMap.ts` (the frozen TS
// reference, relocated from the repo root's `src/` by task 5.4; via Node 24's built-in TS type
// stripping — no compilation step, no copy of the sources) and dumps every field of the
// resulting state as a JSON-lines trace `ddai-tsworld` (Rust) replays and compares bit-for-bit.
//
// Node-stdlib only (fs/path/url/crypto/child_process), no npm dependencies — this script is
// committed and is part of the proof, not a throwaway (see tools/jsmath-oracle/probe.mjs for the
// same house convention).

import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = join(HERE, "..", "..");
/** Root of the frozen TS reference (`tools/ts-reference`); the generators import `<TS_REF>/src/...`. */
export const TS_REF = join(REPO_ROOT, "tools", "ts-reference");
export const CORE_DIR = join(TS_REF, "src", "core");

// --- imports of the REAL tools/ts-reference/src/core/*.ts (never copied, never edited) ---------
export { SimWorld } from "../ts-reference/src/core/world.ts";
export { Collision } from "../ts-reference/src/core/collision.ts";
export { loadMapCollision } from "../ts-reference/src/map/loadMap.ts";
export * as types from "../ts-reference/src/core/types.ts";

// --- f64 <-> hex bit pattern (matches ddai-tsworld's `f64::to_bits`/`from_bits`) ----------------

const f64buf = new ArrayBuffer(8);
const f64view = new DataView(f64buf);

/** Exact IEEE-754 bit pattern of `x`, as 16 lowercase hex chars (big-endian bit value). */
export function f64Bits(x) {
  f64view.setFloat64(0, x, false);
  const hi = f64view.getUint32(0, false);
  const lo = f64view.getUint32(4, false);
  return hi.toString(16).padStart(8, "0") + lo.toString(16).padStart(8, "0");
}

/** Inverse of {@link f64Bits}: the exact `f64` whose bit pattern is this 16-hex-char string. */
export function bitsToF64(s) {
  const hi = parseInt(s.slice(0, 8), 16) >>> 0;
  const lo = parseInt(s.slice(8, 16), 16) >>> 0;
  f64view.setUint32(0, hi, false);
  f64view.setUint32(4, lo, false);
  return f64view.getFloat64(0, false);
}

// --- SplitMix64 (public domain, Vigna) — see docs/formats.md §4's Rust port for the same
// algorithm; this is an independent JS implementation of the identical spec, not shared code —
// two implementations of the same well-specified generator agreeing is a stronger determinism
// check than one implementation reused twice would be. Uses BigInt for exact 64-bit wraparound.

const MASK64 = (1n << 64n) - 1n;

export class SplitMix64 {
  constructor(seed) {
    this.state = BigInt.asUintN(64, BigInt(seed));
  }
  nextU64() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return (z ^ (z >> 31n)) & MASK64;
  }
  /** Uniform in `[0, bound)`, Lemire's multiply-high (matches docs/formats.md §4's `below`). */
  below(bound) {
    if (bound <= 0) return 0;
    const b = BigInt(bound);
    const z = this.nextU64();
    return Number((z * b) >> 64n);
  }
  rangeInclusive(lo, hi) {
    return lo + this.below(hi - lo + 1);
  }
  /** True with probability `num/den` (integers, `num <= den`). */
  chance(num, den) {
    return this.below(den) < num;
  }
  /** Uniform float in `[lo, hi)`. Not part of docs/formats.md §4 (that generator only needs
   * integers); used here for target-vector noise. */
  uniformFloat(lo, hi) {
    // `nextU64() / 2^64` as a double in [0,1) — 53 significant bits is already more precision
    // than an `f64` mantissa can hold, so truncating the low bits away is lossless for our
    // purpose (a broad continuous range) and avoids a BigInt->Number precision-loss surprise.
    const top53 = this.nextU64() >> 11n;
    const unit = Number(top53) / Number(1n << 53n);
    return lo + unit * (hi - lo);
  }
}

// --- TS core provenance (deterministic -- never network, never wall clock, never a git query) ---

/**
 * The last commit that changed the TS core (`core/` + `map/`) before task 5.4 relocated the sources from the repo
 * root's `src/` to `tools/ts-reference/src/` (the files are byte-identical since; `git log --follow` on any of them
 * reaches this commit). It used to be `git log -1 -- src/core src/map`; a pinned value keeps the committed traces'
 * `tsCoreCommit` header byte-identical and does not depend on a full-history checkout.
 */
export const TS_CORE_COMMIT = "dd8c1e3d8c3a3dd15d1f7834d3a64e3ba9dba8e1";
export function tsCoreCommit() {
  return TS_CORE_COMMIT;
}

export function sha256File(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

// --- PlayerInput generation (docs/formats.md §4's "random-v1" idea, adapted to TS's
// `PlayerInput` fields/semantics — see this crate's README for the differences from §4's own
// Rust generator, which targets Oracle A's core-only fields, not TS's weapon/freeze/tele fields).

/** One independent per-tee input state machine. */
export class TeeInputGen {
  constructor(rng, teeIds, selfIndex) {
    this.rng = rng;
    this.teeIds = teeIds;
    this.selfIndex = selfIndex;
    this.direction = 0;
    this.directionTicksLeft = rng.rangeInclusive(5, 40);
    this.jumpHeld = false;
    this.jumpTicksLeft = rng.rangeInclusive(5, 30);
    this.hookHeld = false;
    this.hookTicksLeft = rng.rangeInclusive(3, 30);
    this.aimAtBuddy = teeIds.length > 1 && rng.chance(2, 3);
    this.buddyIndex = this.aimAtBuddy ? this.pickBuddyIndex() : -1;
    this.targetX = 0;
    this.targetY = -1000;
    this.aimTicksLeft = rng.rangeInclusive(10, 50);
    this.wantedWeapon = 0;
    this.fireToggle = 0;
  }

  pickBuddyIndex() {
    let idx = this.selfIndex;
    while (idx === this.selfIndex) idx = this.rng.below(this.teeIds.length);
    return idx;
  }

  /** Advances this tee's input state machine by one tick and returns a fresh `PlayerInput`-shaped
   * plain object (not yet run through `SimWorld.setInput`'s `(0,0)->(0,-1)` normalization — the
   * caller passes it to `setInput` itself, exactly like a real caller would). `positions` is the
   * current tick's tee positions (id -> {x,y}), needed for "aim at buddy". */
  next(positions) {
    const rng = this.rng;

    if (this.directionTicksLeft-- <= 0) {
      this.direction = rng.below(3) - 1;
      this.directionTicksLeft = rng.rangeInclusive(5, 40);
    }

    if (this.jumpHeld) {
      if (this.jumpTicksLeft-- <= 0) {
        this.jumpHeld = false;
        this.jumpTicksLeft = rng.rangeInclusive(5, 30);
      }
    } else if (this.jumpTicksLeft-- <= 0) {
      this.jumpHeld = true;
      this.jumpTicksLeft = rng.rangeInclusive(1, 3);
      if (rng.chance(2, 5)) {
        // Schedule a quick second short press (double-jump attempt shape).
        this._secondJumpIn = rng.rangeInclusive(2, 5);
      }
    }

    if (this.hookHeld) {
      const timeout = rng.chance(3, 20) ? rng.rangeInclusive(61, 120) : rng.rangeInclusive(1, 60);
      if (this._hookTimeout === undefined) this._hookTimeout = timeout;
      if (this._hookElapsed === undefined) this._hookElapsed = 0;
      this._hookElapsed++;
      if (this._hookElapsed >= this._hookTimeout) {
        this.hookHeld = false;
        this._hookTimeout = undefined;
        this._hookElapsed = undefined;
        this.hookTicksLeft = rng.rangeInclusive(3, 30);
      }
    } else if (this.hookTicksLeft-- <= 0) {
      this.hookHeld = true;
    }

    const aimNoise = this.hookHeld && rng.chance(1, 10);
    if (this.aimTicksLeft-- <= 0 || aimNoise) {
      this.aimTicksLeft = rng.rangeInclusive(10, 50);
      if (this.aimAtBuddy && positions[this.teeIds[this.buddyIndex]]) {
        const buddy = positions[this.teeIds[this.buddyIndex]];
        this.targetX = buddy.x + rng.rangeInclusive(-30, 30);
        this.targetY = buddy.y + rng.rangeInclusive(-30, 30);
      } else {
        const angle = rng.uniformFloat(0, 2 * Math.PI);
        this.targetX = Math.cos(angle) * 1000;
        this.targetY = Math.sin(angle) * 1000;
      }
      if (this.targetX === 0 && this.targetY === 0) this.targetY = -1000;
    }

    if (rng.chance(1, 20)) {
      this.wantedWeapon = rng.below(6);
    }

    // fire: bump the press-count's low bit each tick with some probability (mirrors a real
    // client's edge-triggered press counter).
    if (rng.chance(1, 4)) this.fireToggle = (this.fireToggle + 1) & 0x3f;

    let jump = this.jumpHeld ? 1 : 0;
    if (this._secondJumpIn !== undefined) {
      if (--this._secondJumpIn <= 0) {
        jump = 1;
        this._secondJumpIn = undefined;
      }
    }

    return {
      direction: this.direction,
      targetX: this.targetX,
      targetY: this.targetY,
      jump,
      fire: this.fireToggle,
      hook: this.hookHeld ? 1 : 0,
      playerFlags: 0,
      wantedWeapon: this.wantedWeapon,
      nextWeapon: 0,
      prevWeapon: 0,
    };
  }
}

// --- state -> JSON (flat field names matching TS's own `CoreState`/`TeeStateSnapshot` shape) --

export function inputJson(inp) {
  return {
    direction: inp.direction,
    targetX: f64Bits(inp.targetX),
    targetY: f64Bits(inp.targetY),
    jump: inp.jump,
    fire: inp.fire,
    hook: inp.hook,
    playerFlags: inp.playerFlags,
    wantedWeapon: inp.wantedWeapon,
    nextWeapon: inp.nextWeapon,
    prevWeapon: inp.prevWeapon,
  };
}

function coreStateJson(c) {
  return {
    posX: f64Bits(c.posX),
    posY: f64Bits(c.posY),
    velX: f64Bits(c.velX),
    velY: f64Bits(c.velY),
    hookPosX: f64Bits(c.hookPosX),
    hookPosY: f64Bits(c.hookPosY),
    hookDirX: f64Bits(c.hookDirX),
    hookDirY: f64Bits(c.hookDirY),
    hookTeleBaseX: f64Bits(c.hookTeleBaseX),
    hookTeleBaseY: f64Bits(c.hookTeleBaseY),
    hookTick: c.hookTick,
    hookState: c.hookState,
    hookedPlayer: c.hookedPlayer,
    attachedPlayers: c.attachedPlayers,
    activeWeapon: c.activeWeapon,
    newHook: c.newHook,
    jumped: c.jumped,
    jumpedTotal: c.jumpedTotal,
    jumps: c.jumps,
    direction: c.direction,
    angle: c.angle,
    triggeredEvents: c.triggeredEvents,
    colliding: c.colliding,
    leftWall: c.leftWall,
    freezeStart: c.freezeStart,
    freezeEnd: c.freezeEnd,
    isInFreeze: c.isInFreeze,
    moveRestrictions: c.moveRestrictions,
  };
}

function teeSnapshotJson(t) {
  return {
    core: coreStateJson(t.core),
    alive: t.alive,
    freezeTicksLeft: t.freezeTicksLeft,
    frozenLastTick: t.frozenLastTick,
    deepFrozen: t.deepFrozen,
    teleCheckpoint: t.teleCheckpoint,
    moveRestrictions: t.moveRestrictions,
    reloadTimer: t.reloadTimer,
    attackTick: t.attackTick,
    queuedWeapon: t.queuedWeapon,
    input: inputJson(t.input),
    prevInputForEdge: inputJson(t.prevInputForEdge),
    prevPosX: f64Bits(t.prevPos.x),
    prevPosY: f64Bits(t.prevPos.y),
    spawnPosX: f64Bits(t.spawnPos.x),
    spawnPosY: f64Bits(t.spawnPos.y),
    respawnAtTick: t.respawnAtTick,
    weapons: t.weapons.map((w) => ({ got: w.got, ammo: w.ammo })),
  };
}

function projectileJson(p) {
  return {
    id: p.id,
    type: p.type,
    owner: p.owner,
    posX: f64Bits(p.posX),
    posY: f64Bits(p.posY),
    dirX: f64Bits(p.dirX),
    dirY: f64Bits(p.dirY),
    startTick: p.startTick,
    lifeSpan: p.lifeSpan,
    explosive: p.explosive,
    markedForDestroy: p.markedForDestroy,
  };
}

function laserJson(l) {
  return {
    id: l.id,
    owner: l.owner,
    type: l.type,
    posX: f64Bits(l.posX),
    posY: f64Bits(l.posY),
    dirX: f64Bits(l.dirX),
    dirY: f64Bits(l.dirY),
    energy: f64Bits(l.energy),
    bounces: l.bounces,
    evalTick: l.evalTick,
    zeroEnergyBounceInLastTick: l.zeroEnergyBounceInLastTick,
    markedForDestroy: l.markedForDestroy,
  };
}

/** `world.saveState()`'s result -> the trace's flat, hex-bit JSON shape. `st.tees` is TS's own
 * `Map<number, TeeStateSnapshot>` — serialized as a `[[id, snapshot], ...]` array (in the map's
 * own iteration order) so the trace file needs no special "object with numeric string keys"
 * handling on the reading side. */
export function simStateJson(st) {
  return {
    tick: st.tick,
    nextEntityId: st.nextEntityId,
    tees: Array.from(st.tees.entries()).map(([id, t]) => [id, teeSnapshotJson(t)]),
    projectiles: st.projectiles.map(projectileJson),
    lasers: st.lasers.map(laserJson),
  };
}

/** The public `TeeState` shape (`types.ts:41-75`) -> hex-bit JSON, for `applyTeeState`'s op-script
 * argument (op-scripts record the *argument*, not just an id, so a replayer can call
 * `apply_tee_state` with the identical values — see `gen-opscript.mjs`). Optional fields become
 * `null` when TS's own field is `undefined` (JSON has no `undefined`). */
export function teeStateJson(st) {
  return {
    id: st.id,
    alive: st.alive,
    posX: f64Bits(st.pos.x),
    posY: f64Bits(st.pos.y),
    velX: f64Bits(st.vel.x),
    velY: f64Bits(st.vel.y),
    hookState: st.hookState,
    hookPosX: f64Bits(st.hookPos.x),
    hookPosY: f64Bits(st.hookPos.y),
    hookDirX: f64Bits(st.hookDir.x),
    hookDirY: f64Bits(st.hookDir.y),
    hookedPlayer: st.hookedPlayer,
    jumped: st.jumped,
    jumpsLeft: st.jumpsLeft,
    direction: st.direction,
    angle: f64Bits(st.angle),
    activeWeapon: st.activeWeapon,
    frozen: st.frozen,
    freezeTicksLeft: st.freezeTicksLeft,
    attackTick: st.attackTick,
    hookTick: st.hookTick ?? null,
    jumpedTotal: st.jumpedTotal ?? null,
    reloadTicks: st.reloadTicks ?? null,
    frozenFor: st.frozenFor ?? null,
    deepFrozen: st.deepFrozen ?? null,
    jumps: st.jumps ?? null,
    ddnetFlags: st.ddnetFlags ?? null,
    sinceAttack: st.sinceAttack ?? null,
  };
}

export function vec2Json(v) {
  return { x: f64Bits(v.x), y: f64Bits(v.y) };
}

/** `WorldEvent` -> JSON, hex-bit-encoding every `f64` field (review finding F1: `step()`'s
 * returned events *are* part of the parity target — the planner consumes them directly,
 * `src/plan/livePlan.ts`'s `rolled.push(sim.step())` — so `ddai-tsworld`'s replay must compare
 * them field-for-field like everything else, not treat them as inert diagnostics). Only
 * `explosion`'s `pos` is a float; every other event field is already an integer id/count/enum. */
export function eventJson(e) {
  if (e.kind === "explosion") {
    return { kind: e.kind, pos: vec2Json(e.pos), owner: e.owner };
  }
  return e;
}

/** `Collision`'s `order`, as ids (`world["order"]` reads the private runtime field — see
 * `crates/ddai-jsmath/src/rng.rs`'s doc comment on why TS `private` is a compile-time-only
 * annotation that Node's type stripping erases, making the field an ordinary reachable property
 * at runtime; this file, like that one, never edits `src/`, only reads a field from outside). */
export function orderIds(world) {
  return world.order.map((rec) => rec.core.id);
}

export function byIdIds(world) {
  return world.byId.map((rec) => rec.core.id);
}
