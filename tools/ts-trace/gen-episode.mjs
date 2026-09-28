#!/usr/bin/env node
// trace-ts v1 "episode" generator (docs/formats.md, new "trace-ts v1" section). Drives the REAL
// `src/core/world.ts` `SimWorld` (via `../../src/map/loadMap.ts`'s real file-based loader — see
// this crate's README for why synthetic maps are also real `.map` files, not a JS-side recipe
// re-implementation) with seeded per-tee input, and writes one JSON object per line: a metadata
// line, then one line per tick.
//
// Usage:
//   node gen-episode.mjs --map <path.map> --seed <u64> --ticks <n> --tees <n> --out <path.jsonl>
//     [--respawn-delay <ticks>] [--infinite-ammo <0|1>] [--sv-hit <0|1>] [--all-weapons <0|1>]
//     [--no-weak-hook <0|1>]

import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import {
  SimWorld,
  loadMapCollision,
  SplitMix64,
  TeeInputGen,
  simStateJson,
  inputJson,
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
const ticks = Number(args.ticks ?? "1000");
const teeCount = Number(args.tees ?? "2");
const outPath = args.out;
if (!mapPath || !outPath) {
  console.error("usage: gen-episode.mjs --map <path> --seed <u64> --ticks <n> --tees <n> --out <path>");
  process.exit(2);
}

const options = {
  respawnDelayTicks: Number(args["respawn-delay"] ?? "150"),
  infiniteAmmo: (args["infinite-ammo"] ?? "1") !== "0",
  svHit: (args["sv-hit"] ?? "1") !== "0",
  allWeapons: (args["all-weapons"] ?? "1") !== "0",
  // Review finding F8: production servers do not run with every weapon and infinite ammo — this
  // flag (default off, matching the pre-existing behavior) lets the corpus also cover a
  // `noWeakHook: true` map/option combination.
  noWeakHook: (args["no-weak-hook"] ?? "0") !== "0",
};

const loaded = loadMapCollision(mapPath);
const world = new SimWorld(loaded.collision, options);

// --- spawn placement: free (TILE_AIR, in game AND front-if-present) cells, row-major, matching
// docs/formats.md §4's "Спавн персонажей" idea (first tee: random free cell; each next tee: 1/2
// chance within Chebyshev distance 1..=3 of an already-placed tee, else independently random;
// positions never repeat).
const freeCells = [];
for (let y = 0; y < loaded.height; y++) {
  for (let x = 0; x < loaded.width; x++) {
    const i = y * loaded.width + x;
    if (loaded.collision.tiles[i] !== 0) continue;
    if (loaded.collision.frontIndex !== undefined && loaded.collision.frontIndex[i] !== 0) continue;
    freeCells.push({ x, y });
  }
}
if (freeCells.length < teeCount) {
  console.error(`map has only ${freeCells.length} free cells, need ${teeCount}`);
  process.exit(2);
}

const rng = new SplitMix64(seed);
const used = new Set();
const placedCells = [];
function cellKey(c) {
  return c.y * loaded.width + c.x;
}
function pickRandomFreeCell() {
  let tries = 0;
  while (tries++ < 100000) {
    const c = freeCells[rng.below(freeCells.length)];
    if (!used.has(cellKey(c))) return c;
  }
  throw new Error("could not find a free cell");
}
function pickNearCell(base) {
  const candidates = freeCells.filter((c) => {
    if (used.has(cellKey(c))) return false;
    const d = Math.max(Math.abs(c.x - base.x), Math.abs(c.y - base.y));
    return d >= 1 && d <= 3;
  });
  if (candidates.length === 0) return pickRandomFreeCell();
  return candidates[rng.below(candidates.length)];
}

const teeIds = [];
for (let i = 0; i < teeCount; i++) {
  const id = i + 1;
  let cell;
  if (i === 0) {
    cell = pickRandomFreeCell();
  } else if (rng.chance(1, 2)) {
    cell = pickNearCell(placedCells[rng.below(placedCells.length)]);
  } else {
    cell = pickRandomFreeCell();
  }
  used.add(cellKey(cell));
  placedCells.push(cell);
  teeIds.push(id);
  world.addTee(id, { x: cell.x * 32 + 16, y: cell.y * 32 + 16 });
}

const inputGens = teeIds.map((id, idx) => new TeeInputGen(rng, teeIds, idx));

mkdirSync(dirname(outPath), { recursive: true });
const lines = [];
lines.push(
  JSON.stringify({
    traceVersion: 1,
    kind: "episode",
    generatorVersion: "ts-trace 1",
    node: process.version,
    v8: process.versions.v8,
    tsCoreCommit: tsCoreCommit(),
    mapPath,
    mapSha256: sha256File(mapPath),
    seed: seed.toString(),
    tees: teeIds,
    ticks,
    options,
  }),
);

for (let tick = 1; tick <= ticks; tick++) {
  const positions = {};
  for (const id of teeIds) {
    const t = world.getTee(id);
    if (t) positions[id] = t.pos;
  }
  const appliedInputs = [];
  for (let i = 0; i < teeIds.length; i++) {
    const id = teeIds[i];
    if (!world.isAlive(id)) continue;
    const inp = inputGens[i].next(positions);
    world.setInput(id, inp);
    appliedInputs.push({ id, ...inputJson(inp) });
  }

  const events = world.step();

  lines.push(
    JSON.stringify({
      tick,
      order: orderIds(world),
      byId: byIdIds(world),
      events: events.map(eventJson),
      inputs: appliedInputs,
      state: simStateJson(world.saveState()),
    }),
  );
}

writeFileSync(outPath, lines.join("\n") + "\n");
console.error(`wrote ${lines.length - 1} ticks to ${outPath}`);
