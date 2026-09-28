#!/usr/bin/env node
// trace-ts v1 "opscript" generator (docs/formats.md "trace-ts v1"). Drives the REAL `SimWorld`
// API (`step`, `saveState`/`restoreState`, `applyTeeState`, `setHeldInput`, `applyForce`,
// `unfreeze`, `kill`, `addTee`/`removeTee`, `setGrenades`) through a seeded random sequence, and
// dumps the resulting `SimState` after every op.
//
// Usage:
//   node gen-opscript.mjs --map <path.map> --seed <u64> --ops <n> --tees <n> --out <path.jsonl>
//     [--respawn-delay <ticks>] [--infinite-ammo <0|1>] [--sv-hit <0|1>] [--all-weapons <0|1>]
//     [--no-weak-hook <0|1>]

import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import {
  SimWorld,
  loadMapCollision,
  SplitMix64,
  simStateJson,
  teeStateJson,
  inputJson,
  vec2Json,
  eventJson,
  orderIds,
  byIdIds,
  sha256File,
  tsCoreCommit,
} from "./lib.mjs";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      out[argv[i].slice(2)] = argv[i + 1];
      i++;
    }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
const mapPath = args.map;
const seed = BigInt(args.seed ?? "1");
const opCount = Number(args.ops ?? "10000");
const maxTees = Number(args.tees ?? "4");
const outPath = args.out;
if (!mapPath || !outPath) {
  console.error("usage: gen-opscript.mjs --map <path> --seed <u64> --ops <n> --tees <n> --out <path>");
  process.exit(2);
}

// Review finding F8: previously hardcoded, so the whole bulk corpus only ever exercised one
// option combination — never the production-realistic `respawnDelayTicks: 0` /
// `allWeapons: false` / `svHit: false` / `infiniteAmmo: false` / `noWeakHook: true` mix real
// server configs actually run with (see `bot.ts` / `planner.ts` defaults). Now CLI-driven, so
// `run-corpus.sh` can vary it per generated file the same way it already varies seed/map/tees.
const options = {
  respawnDelayTicks: Number(args["respawn-delay"] ?? "150"),
  infiniteAmmo: (args["infinite-ammo"] ?? "1") !== "0",
  svHit: (args["sv-hit"] ?? "1") !== "0",
  allWeapons: (args["all-weapons"] ?? "1") !== "0",
  noWeakHook: (args["no-weak-hook"] ?? "0") !== "0",
};
const loaded = loadMapCollision(mapPath);
const world = new SimWorld(loaded.collision, options);
const rng = new SplitMix64(seed);

const freeCells = [];
for (let y = 0; y < loaded.height; y++) {
  for (let x = 0; x < loaded.width; x++) {
    const i = y * loaded.width + x;
    if (loaded.collision.tiles[i] !== 0) continue;
    freeCells.push({ x: x * 32 + 16, y: y * 32 + 16 });
  }
}
function randomSpawn() {
  return freeCells[rng.below(freeCells.length)];
}

let nextId = 1;
const liveIds = [];
// Every id ever assigned (live or not) — review finding F8: sampled from by the "addTee" case
// below to occasionally reuse an id instead of always minting a fresh one, so the random corpus
// also exercises F4 (duplicate-id orphan, id still live) and F7 (reused id after removal) at
// scale, not just via the hand-scripted `gen-regression.mjs` fixtures.
const usedIds = [];

function randomInput() {
  const angle = rng.uniformFloat(0, 2 * Math.PI);
  return {
    direction: rng.below(3) - 1,
    targetX: Math.cos(angle) * 1000,
    targetY: Math.sin(angle) * 1000,
    jump: rng.below(2),
    fire: rng.below(64),
    hook: rng.below(2),
    playerFlags: 0,
    wantedWeapon: rng.below(6),
    nextWeapon: 0,
    prevWeapon: 0,
  };
}

/** Builds a "random but valid" `TeeState` by perturbing a real, currently-alive tee's own
 * snapshot (or a freshly spawned default one if none is alive yet) — see this tool's README for
 * why "random but valid" means "physically plausible", not "arbitrary bit patterns": a handful of
 * `applyTeeState`'s own fields derive others (`jumpedTotal` from `jumpsLeft` when absent,
 * `attackTick` from `sinceAttack` when present), so basing perturbations on a real snapshot keeps
 * every op meaningful for the planner-parity goal this trace ultimately serves. */
function randomTeeState(baseId) {
  const base = world.getTee(baseId) ?? {
    id: baseId,
    alive: true,
    pos: randomSpawn(),
    vel: { x: 0, y: 0 },
    hookState: 0,
    hookPos: { x: 0, y: 0 },
    hookDir: { x: 0, y: 0 },
    hookedPlayer: -1,
    jumped: 0,
    jumpsLeft: 2,
    direction: 0,
    angle: 0,
    activeWeapon: 1,
    frozen: false,
    freezeTicksLeft: 0,
    attackTick: 0,
  };
  const st = { ...base, pos: { ...base.pos }, vel: { ...base.vel }, hookPos: { ...base.hookPos }, hookDir: { ...base.hookDir } };
  st.pos.x += rng.rangeInclusive(-50, 50);
  st.pos.y += rng.rangeInclusive(-50, 50);
  st.vel.x = rng.rangeInclusive(-100, 100) / 10;
  st.vel.y = rng.rangeInclusive(-100, 100) / 10;
  if (rng.chance(1, 3)) {
    st.frozen = true;
    st.freezeTicksLeft = rng.rangeInclusive(1, 150);
  } else {
    st.frozen = false;
    st.freezeTicksLeft = 0;
  }
  if (rng.chance(1, 4)) st.hookState = rng.below(6) - 1;
  if (rng.chance(1, 5)) st.deepFrozen = rng.chance(1, 2);
  if (rng.chance(1, 5)) st.jumps = rng.below(3);
  return st;
}

mkdirSync(dirname(outPath), { recursive: true });

const savedStates = [];

// Seed with two initial tees (as explicit, recorded `addTee` ops — see this file's op loop below
// — not a silent pre-step) so most of the random op sequence has something to act on right away.
function addTeeOp() {
  const id = nextId++;
  const pos = randomSpawn();
  world.addTee(id, pos);
  liveIds.push(id);
  usedIds.push(id);
  lines.push(
    JSON.stringify({
      op: "addTee",
      id,
      pos: vec2Json(pos),
      events: [],
      order: orderIds(world),
      byId: byIdIds(world),
      state: simStateJson(world.saveState()),
    }),
  );
}

const lines = [];
lines.push(
  JSON.stringify({
    traceVersion: 1,
    kind: "opscript",
    generatorVersion: "ts-trace 1",
    node: process.version,
    v8: process.versions.v8,
    tsCoreCommit: tsCoreCommit(),
    mapPath,
    mapSha256: sha256File(mapPath),
    seed: seed.toString(),
    ops: opCount,
    options,
  }),
);
for (let i = 0; i < Math.min(2, maxTees); i++) addTeeOp();

// Weighted op menu (name, weight). `step` dominates so op scripts still look like real usage
// mixed with API calls, per the task's own list.
const OPS = [
  ["step", 30],
  ["saveState", 8],
  ["saveInto", 4],
  ["restoreState", 6],
  ["applyTeeState", 10],
  ["setHeldInput", 15],
  ["applyForce", 8],
  ["unfreeze", 5],
  ["kill", 4],
  ["addTee", 3],
  ["removeTee", 3],
  ["setGrenades", 4],
  ["reset", 2],
];
const totalWeight = OPS.reduce((a, [, w]) => a + w, 0);
function pickOp() {
  let r = rng.below(totalWeight);
  for (const [name, w] of OPS) {
    if (r < w) return name;
    r -= w;
  }
  return "step";
}

for (let i = 0; i < opCount; i++) {
  const op = pickOp();
  let record = { op };

  switch (op) {
    case "step": {
      const events = world.step();
      record.events = events.map(eventJson);
      break;
    }
    case "saveState": {
      savedStates.push(world.saveState());
      record.savedIndex = savedStates.length - 1;
      break;
    }
    case "saveInto": {
      // Review finding F8: `saveState(into)` (F7's own quirk — never clearing stale entries) was
      // never exercised by the random generator at all, only by hand-scripted regression
      // fixtures. `slot` is the field name `trace_ts.rs`'s `OpLine.slot` reads (see F7's fixture
      // and comments there for why this must not be `savedIndex`/`restoredIndex`).
      if (savedStates.length === 0) {
        record = { op: "noop" };
        break;
      }
      const idx = rng.below(savedStates.length);
      world.saveState(savedStates[idx]);
      record.slot = idx;
      break;
    }
    case "reset": {
      world.reset();
      break;
    }
    case "restoreState": {
      if (savedStates.length === 0) {
        // Bug found while scaling the corpus up for review finding F8: this fallback (no saved
        // state exists yet, so a real `step()` runs instead) used to call `world.step()` and
        // throw its return value away, always recording `events: []` for that line — even on a
        // tick where a real event (e.g. a death queued by an earlier `kill()`) actually fired.
        // The Rust side, replaying the same (correctly labeled) `"step"` op, computes that same
        // real event from its own `world.step()` — so the discrepancy was entirely this
        // generator discarding TS's own true output, not a Rust parity bug (confirmed by
        // replaying the exact op sequence directly against `SimWorld` outside the generator).
        // Fixed by recording events exactly like the `"step"` case below.
        record = { op: "step" };
        const events = world.step();
        record.events = events.map(eventJson);
        break;
      }
      const idx = rng.below(savedStates.length);
      world.restoreState(savedStates[idx]);
      record.restoredIndex = idx;
      break;
    }
    case "applyTeeState": {
      if (liveIds.length === 0) {
        record = { op: "noop" };
        break;
      }
      const id = liveIds[rng.below(liveIds.length)];
      const st = randomTeeState(id);
      world.applyTeeState(id, st);
      record.id = id;
      record.teeState = teeStateJson(st);
      break;
    }
    case "setHeldInput": {
      if (liveIds.length === 0) {
        record = { op: "noop" };
        break;
      }
      const id = liveIds[rng.below(liveIds.length)];
      const inp = randomInput();
      world.setHeldInput(id, inp);
      record.id = id;
      record.input = inputJson(inp);
      break;
    }
    case "applyForce": {
      if (liveIds.length === 0) {
        record = { op: "noop" };
        break;
      }
      const id = liveIds[rng.below(liveIds.length)];
      const fx = rng.rangeInclusive(-500, 500) / 10;
      const fy = rng.rangeInclusive(-500, 500) / 10;
      world.applyForce(id, { x: fx, y: fy });
      record.id = id;
      record.force = vec2Json({ x: fx, y: fy });
      break;
    }
    case "unfreeze": {
      if (liveIds.length === 0) {
        record = { op: "noop" };
        break;
      }
      const id = liveIds[rng.below(liveIds.length)];
      world.unfreeze(id);
      record.id = id;
      break;
    }
    case "kill": {
      if (liveIds.length === 0) {
        record = { op: "noop" };
        break;
      }
      const id = liveIds[rng.below(liveIds.length)];
      world.kill(id);
      record.id = id;
      break;
    }
    case "addTee": {
      // Review finding F8: 1-in-5, reuse a previously-used id instead of minting a fresh one —
      // if it's still live, this is F4's duplicate-`addTee` orphan quirk; if it was removed, this
      // is F7's "reused id after removal" quirk. Bypasses the `maxTees` gate below when the id is
      // already live (re-adding it doesn't grow the *current* tee count, only `order`/`byId`).
      if (rng.chance(1, 5) && usedIds.length > 0) {
        const id = usedIds[rng.below(usedIds.length)];
        const pos = randomSpawn();
        world.addTee(id, pos);
        if (!liveIds.includes(id)) liveIds.push(id);
        record.id = id;
        record.pos = vec2Json(pos);
        break;
      }
      if (liveIds.length >= maxTees) {
        record = { op: "noop" };
        break;
      }
      const id = nextId++;
      const pos = randomSpawn();
      world.addTee(id, pos);
      liveIds.push(id);
      usedIds.push(id);
      record.id = id;
      record.pos = vec2Json(pos);
      break;
    }
    case "removeTee": {
      if (liveIds.length === 0) {
        record = { op: "noop" };
        break;
      }
      const idx = rng.below(liveIds.length);
      const id = liveIds[idx];
      world.removeTee(id);
      liveIds.splice(idx, 1);
      record.id = id;
      break;
    }
    case "setGrenades": {
      const count = rng.below(3);
      const list = [];
      for (let k = 0; k < count; k++) {
        const angle = rng.uniformFloat(0, 2 * Math.PI);
        list.push({
          owner: liveIds.length > 0 ? liveIds[rng.below(liveIds.length)] : -1,
          spawnPos: randomSpawn(),
          dir: { x: Math.cos(angle), y: Math.sin(angle) },
          ageTicks: rng.below(20),
        });
      }
      world.setGrenades(list);
      record.count = count;
      record.list = list.map((g) => ({
        owner: g.owner,
        spawnPos: vec2Json(g.spawnPos),
        dir: vec2Json(g.dir),
        ageTicks: g.ageTicks,
      }));
      break;
    }
    default:
      break;
  }

  // `order`/`byId` on every line (review finding F1: the duplicate-`addTee`-id orphan quirk, and
  // any other order-affecting bug, is only visible in these, not in `state` — see also
  // `gen-episode.mjs`, which already recorded them per tick).
  record.events ??= [];
  record.order = orderIds(world);
  record.byId = byIdIds(world);
  record.state = simStateJson(world.saveState());
  lines.push(JSON.stringify(record));
}

writeFileSync(outPath, lines.join("\n") + "\n");
console.error(`wrote ${lines.length - 1} ops to ${outPath}`);
