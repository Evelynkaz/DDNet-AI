#!/usr/bin/env node
// Task 3.2 acceptance criterion 3, "free-running" check: unlike gen-planner-dump.mjs (which
// resets the `Planner` before every single decision, so hidden cross-decision state never has to
// round-trip through this file), this drives ONE continuous game -- a single `Planner` instance,
// `reset()` once at the start, deciding every tick against the real, live `SimWorld` (`sync:
// "direct"`, `docs/research/orig-plan.md` §2.1: `decide()` itself saves/restores the world it is
// given, so handing it the live world directly is safe and needs no snapshot-sync layer) against
// a scripted opponent -- so `warm`/`committed`/`commitLeft`/`decideGaps`/`dirSince`/`dirLast`/
// `oppSeed`/the CEM `Rng` (including its carried Gaussian spare)/`thawScratch`+`thawMemo` all
// persist and evolve turn to turn exactly like a real game, and a Rust replayer must reproduce
// that evolution decision-for-decision to match, not just one isolated call.
//
// Usage:
//   node gen-planner-freerun.mjs --map <path|synthetic:name> --seed <u32> --ticks <n>
//     --preset <normal|low|strong|wb> --opponent <hold|react|mix> --out <path.jsonl>

import { writeFileSync, mkdirSync, appendFileSync } from "node:fs";
import { dirname } from "node:path";
import { SimWorld, loadMapCollision, teeStateJson, inputJson, sha256File, tsCoreCommit, TS_REF } from "./lib.mjs";

const { Planner } = await import(`${TS_REF}/src/plan/planner.ts`);
const { scriptedAction } = await import(`${TS_REF}/src/env/scripted.ts`);
const { Rng } = await import(`${TS_REF}/src/nn/rng.ts`);
const { emptyInput } = await import(`${TS_REF}/src/core/types.ts`);

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
const mapArg = args.map;
const seed = Number(args.seed ?? "1") >>> 0;
const ticks = Number(args.ticks ?? "600");
const presetName = args.preset ?? "normal";
const opponentName = args.opponent ?? "hold";
const outPath = args.out;
if (!mapArg || !outPath) {
  console.error("usage: gen-planner-freerun.mjs --map <path|synthetic:name> --seed <u32> --ticks <n> --preset <p> --opponent <o> --out <path>");
  process.exit(2);
}

const PRESET_BASE = { thirdTeeExposure: 0, memoryTrust: 0.9, frozenThrow: 3, explain: true };
const WB_OVER = { noThawRope: true, frozenThrow: 3, airJumpCost: 0.3, launchExactReach: 100 };
function presetConfig(name) {
  switch (name) {
    case "normal":
      return { ...PRESET_BASE };
    case "low":
      return { ...PRESET_BASE, explain: false, commitDecisions: 2, shieldCadence: true };
    case "strong":
      return { ...PRESET_BASE, ...WB_OVER, population: 40, iterations: 3 };
    case "wb":
      return { ...PRESET_BASE, ...WB_OVER };
    default:
      throw new Error(`unknown preset ${name}`);
  }
}
function opponentConfig(name) {
  switch (name) {
    case "hold":
      return {};
    case "react":
      return { opponentModel: "react" };
    case "mix":
      return { opponentMix: true };
    default:
      throw new Error(`unknown opponent model ${name}`);
  }
}
const cfg = { ...presetConfig(presetName), ...opponentConfig(opponentName), budgetMs: 0, hardMs: 0 };

let mapPath = mapArg.startsWith("synthetic:")
  ? `${process.env.DDAI_DATA_DIR ?? "/home/ubuntu/aiddnet/data"}/maps/synthetic/${mapArg.slice("synthetic:".length)}.map`
  : mapArg;
const { collision: col } = loadMapCollision(mapPath);
const mapSha256 = sha256File(mapPath);

function tileCenter(t) {
  return t * 32 + 16;
}
function findStandable(col) {
  const out = [];
  for (let ty = 0; ty < col.height - 1; ty++) {
    for (let tx = 0; tx < col.width; tx++) {
      const px = tileCenter(tx);
      const py = tileCenter(ty);
      if (col.isSolid(px, py) || col.isFreeze(px, py) || col.isDeath(px, py)) continue;
      if (!col.isSolid(tileCenter(tx), py + 32)) continue;
      out.push({ x: px, y: py });
    }
  }
  return out;
}
const stand = findStandable(col);
const pickRng = new Rng((seed * 2654435761) >>> 0);
function pickSpawn() {
  const p = stand[Math.floor(pickRng.nextFloat() * stand.length)];
  return { x: p.x, y: p.y };
}
const selfSpawn = pickSpawn();
const enemySpawn = pickSpawn();

const world = new SimWorld(col, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
// bot.ts's own order: addTee(ownId) then addTee(targetId) -- the target ends up first in `order`.
world.addTee(0, selfSpawn);
world.addTee(1, enemySpawn);

const planner = new Planner(cfg);
planner.setSearchSeed(seed >>> 0);
const scriptRng = new Rng((seed * 7919 + 17) >>> 0);

let selfPrev = emptyInput();
let enemyPrev = emptyInput();
const decisions = [];

for (let t = 0; t < ticks; t++) {
  const self = world.getTee(0);
  const enemy = world.getTee(1);
  if (self === undefined || enemy === undefined || !self.alive || !enemy.alive) break;

  const enemyInput = scriptedAction(world, 1, 0, enemyPrev, scriptRng);
  planner.setLiveTick(world.tick);
  const selfInput = planner.decide(world, 0, 1, selfPrev, enemyInput);
  const li = planner.lastInfo;

  decisions.push({
    tick: world.tick,
    decision: inputJson(selfInput),
    lastInfo: {
      searched: li.searched,
      candidates: li.candidates,
      outOfTime: li.outOfTime,
      hookAt: li.hookAt,
      gated: li.gated,
      shielded: li.shielded ?? false,
      edgeHeld: li.edgeHeld ?? false,
      selfOut: li.selfOut,
      enemyOut: li.enemyOut,
    },
  });

  world.setInput(0, selfInput);
  world.setInput(1, enemyInput);
  world.step();
  selfPrev = selfInput;
  enemyPrev = enemyInput;
}

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(
  outPath,
  JSON.stringify({
    kind: "meta",
    version: 1,
    mapPath,
    mapSha256,
    seed,
    preset: presetName,
    opponent: opponentName,
    config: cfg,
    selfSpawn,
    enemySpawn,
    tsCoreCommit: tsCoreCommit(),
    node: process.version,
    v8: process.versions.v8,
  }) + "\n",
);
for (const d of decisions) appendFileSync(outPath, JSON.stringify({ kind: "decision", ...d }) + "\n");
process.stderr.write(`${outPath}: ${decisions.length} decisions over ${ticks} ticks (self alive=${world.getTee(0)?.alive}, enemy alive=${world.getTee(1)?.alive})\n`);
