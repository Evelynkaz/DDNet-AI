#!/usr/bin/env node
// Task 3.8: parity dumps for the pieces upstream af49dfb (release 2026-10-02) added to the planner, one by one, next to the
// full-planner dumps of `gen-planner-dump.mjs` (which exercise them only through decisions): `ceilingField` (the whole grid),
// `ropeIntercept` (random victim states, incl. against walls and standing still), `wallSwingLines`/`airChainLines`
// (step layouts of `buildStepTicks`, both walls, with and without the air jump) and `sealedIn(..., passive)` (frozen and free
// tees at freeze edges). Needs the af49dfb reference: `DDAI_TS_REF=<dir with src/ of af49dfb>`.
//
// Usage: DDAI_TS_REF=... node gen-v2-component-dump.mjs --map <path|synthetic:name> --seed <u32> --cases <n> --out <path.jsonl>

import { writeFileSync, mkdirSync, appendFileSync } from "node:fs";
import { dirname } from "node:path";
import { SimWorld, loadMapCollision, f64Bits, sha256File, inputJson, TS_REF } from "./lib.mjs";

const planner = await import(`${TS_REF}/src/plan/planner.ts`);
const { ceilingField, ropeIntercept, buildStepTicks } = planner;
const { wallSwingLines, airChainLines } = await import(`${TS_REF}/src/plan/throwLines.ts`);
const { sealedIn } = await import(`${TS_REF}/src/plan/seal.ts`);
const { Rng } = await import(`${TS_REF}/src/nn/rng.ts`);
const { emptyInput } = await import(`${TS_REF}/src/core/types.ts`);
if (typeof ceilingField !== "function" || typeof wallSwingLines !== "function") {
  console.error(`${TS_REF} has no af49dfb planner (ceilingField/wallSwingLines missing): set DDAI_TS_REF`);
  process.exit(2);
}

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
const wantCases = Number(args.cases ?? "1500");
const outPath = args.out;
if (!mapArg || !outPath) {
  console.error("usage: gen-v2-component-dump.mjs --map <path|synthetic:name> --seed <u32> --cases <n> --out <path>");
  process.exit(2);
}
const mapPath = mapArg.startsWith("synthetic:")
  ? `${process.env.DDAI_DATA_DIR ?? "/home/ubuntu/aiddnet/data"}/maps/synthetic/${mapArg.slice("synthetic:".length)}.map`
  : mapArg;
const { collision: col } = loadMapCollision(mapPath);
const mapSha256 = sha256File(mapPath);

const tileCenter = (t) => t * 32 + 16;
const stand = [];
for (let ty = 0; ty < col.height - 1; ty++) {
  for (let tx = 0; tx < col.width; tx++) {
    const px = tileCenter(tx);
    const py = tileCenter(ty);
    if (col.isSolid(px, py) || col.isFreeze(px, py) || col.isDeath(px, py)) continue;
    if (!col.isSolid(px, py + 32)) continue;
    stand.push({ x: px, y: py });
  }
}
// Standable tiles with a freeze/death tile within one tile, and the direction to it (as in gen-component-dump.mjs).
const edge = [];
for (const p of stand) {
  const tx = Math.trunc(p.x / 32);
  const ty = Math.trunc(p.y / 32);
  let best = null;
  let bestD = Infinity;
  for (let oy = -1; oy <= 1; oy++) {
    for (let ox = -1; ox <= 1; ox++) {
      const nx = tx + ox;
      const ny = ty + oy;
      if (nx < 0 || ny < 0 || nx >= col.width || ny >= col.height) continue;
      if (!col.isFreeze(nx * 32 + 16, ny * 32 + 16)) continue;
      const d = ox * ox + oy * oy;
      if (d < bestD) {
        bestD = d;
        best = { ox, oy };
      }
    }
  }
  if (best !== null) edge.push({ x: p.x, y: p.y, dx: best.ox, dy: best.oy });
}

const rng = new Rng((seed * 2654435761 + 11) >>> 0);
const planJson = (lines) => lines.map((l) => l.map((s) => ({ dir: s.dir, jump: s.jump, hook: s.hook, fire: s.fire, aim: f64Bits(s.aim) })));
const out = [];

// --- ceilingField: the whole grid -------------------------------------------------------------------------------------
{
  const f = ceilingField(col);
  out.push({ kind: "ceiling", width: col.width, height: col.height, dist: Array.from(f) });
}

// --- ropeIntercept: from a standable tile to a victim somewhere within hook reach, moving or not -----------------------
let moving = 0;
for (let i = 0; i < wantCases; i++) {
  const a = stand[Math.floor(rng.nextFloat() * stand.length)];
  // Victim: near `a` half of the time (a rope's reach), anywhere otherwise (walls in between, off the beaten path).
  const near = rng.nextFloat() < 0.6;
  const b = near ? { x: a.x + (rng.nextFloat() * 2 - 1) * 380, y: a.y + (rng.nextFloat() * 2 - 1) * 260 } : stand[Math.floor(rng.nextFloat() * stand.length)];
  const still = rng.nextFloat() < 0.15;
  const vel = still ? { x: 0, y: 0 } : { x: (rng.nextFloat() * 2 - 1) * 24, y: (rng.nextFloat() * 2 - 1) * 24 };
  if (!still) moving++;
  const from = { x: a.x + (rng.nextFloat() * 2 - 1) * 8, y: a.y + (rng.nextFloat() * 2 - 1) * 8 };
  const r = ropeIntercept(from, { pos: b, vel }, col);
  out.push({ kind: "intercept", fromX: f64Bits(from.x), fromY: f64Bits(from.y), posX: f64Bits(b.x), posY: f64Bits(b.y), velX: f64Bits(vel.x), velY: f64Bits(vel.y), x: f64Bits(r.x), y: f64Bits(r.y) });
}

// --- wallSwingLines / airChainLines: step layouts of buildStepTicks, both walls, the air jump or not -------------------
const layouts = [
  [9, 3, 0, 2],
  [16, 3, 0, 2],
  [16, 3, 3, 1],
  [12, 2, 0, 2],
  [9, 3, 2, 1],
  [5, 4, 0, 2],
  [3, 3, 0, 2],
  [1, 3, 0, 2],
];
for (const [steps, planStep, frontSteps, frontStep] of layouts) {
  const ticks = buildStepTicks(steps, planStep, frontSteps, frontStep);
  for (const at of [0, 0.7, -1.9]) {
    for (const wallDir of [-1, 1, 0, -3]) {
      out.push({ kind: "wallswing", steps, stepTicks: ticks, at: f64Bits(at), wallDir, lines: planJson(wallSwingLines(steps, ticks, at, wallDir)) });
      for (const airJump of [true, false]) {
        out.push({ kind: "airchain", steps, stepTicks: ticks, at: f64Bits(at), wallDir, airJump, lines: planJson(airChainLines(steps, ticks, at, wallDir, airJump)) });
      }
    }
  }
}
// Degenerate layouts the guards return early on.
out.push({ kind: "wallswing", steps: 9, stepTicks: [], at: f64Bits(0), wallDir: 1, lines: [] });
out.push({ kind: "airchain", steps: 9, stepTicks: [], at: f64Bits(0), wallDir: 1, airJump: true, lines: [] });
out.push({ kind: "wallswing", steps: 0, stepTicks: [3], at: f64Bits(0), wallDir: 1, lines: [] });

// --- sealedIn(passive): a free or frozen tee next to freeze, holding an input ------------------------------------------
let sealedTrue = 0;
let passiveDiffers = 0;
const sealN = Math.min(wantCases, 800);
for (let i = 0; i < sealN; i++) {
  const near = edge.length > 0 && rng.nextFloat() < 0.8;
  const p = near ? edge[Math.floor(rng.nextFloat() * edge.length)] : stand[Math.floor(rng.nextFloat() * stand.length)];
  let vel;
  if (near) {
    const norm = Math.hypot(p.dx, p.dy) || 1;
    const speed = rng.nextFloat() * 10;
    vel = { x: (p.dx / norm) * speed, y: (p.dy / norm) * speed - (p.dy === 0 ? rng.nextFloat() * 6 : 0) };
  } else {
    vel = { x: (rng.nextFloat() * 2 - 1) * 6, y: 0 };
  }
  const frozen = rng.nextFloat() < 0.7;
  const freezeTicksLeft = frozen ? [0, 10, 30, 60, 89, 90, 120, 200][Math.floor(rng.nextFloat() * 8)] : 0;
  const dir = [-1, 0, 1][Math.floor(rng.nextFloat() * 3)];
  const jump = rng.nextFloat() < 0.3 ? 1 : 0;
  const held = { ...emptyInput(), direction: dir, jump };
  const run = (passive) => {
    const w = new SimWorld(col, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
    w.addTee(0, { x: p.x, y: p.y });
    w.addTee(1, { x: stand[0].x, y: stand[0].y });
    const base = w.getTee(0);
    const state = { ...base, pos: { x: p.x, y: p.y }, vel, frozen, freezeTicksLeft };
    return passive === undefined ? sealedIn(w, 0, state, held) : sealedIn(w, 0, state, held, passive);
  };
  const plain = run(undefined);
  const passive = run(true);
  const explicitFalse = run(false);
  if (plain !== explicitFalse) throw new Error("sealedIn(passive=false) differs from the default");
  if (passive) sealedTrue++;
  if (passive !== plain) passiveDiffers++;
  out.push({ kind: "sealpassive", posX: f64Bits(p.x), posY: f64Bits(p.y), velX: f64Bits(vel.x), velY: f64Bits(vel.y), frozen, freezeTicksLeft, input: inputJson(held), plain, passive });
}

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify({ kind: "meta", version: 1, mapPath, mapSha256, seed, tsRef: TS_REF }) + "\n");
for (const c of out) appendFileSync(outPath, JSON.stringify(c) + "\n");
process.stderr.write(`${outPath}: ${out.length} cases (${moving} moving intercepts, ${sealN} sealedIn cases: ${sealedTrue} passive-sealed, ${passiveDiffers} where passive differs from plain)\n`);
