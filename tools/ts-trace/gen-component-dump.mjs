#!/usr/bin/env node
// Task 3.2 acceptance criterion 4: dedicated parity dumps for the smaller pure/near-pure
// components (`scriptedAction`, `seal.ts`'s `restsInFreeze`/`touchesFreeze`/`sealedIn`,
// `throwLines.ts`'s `throwLines`/`frozenThrowLines`, `shield.ts`'s `escapeExists`/`saferInput`,
// the BFS `hazardField`/`unfreezeField` grids) -- separate from the full-planner dumps
// (`gen-planner-dump.mjs`), which already exercise every one of these indirectly (any divergence
// in any of them would change a decision or a score and show up there), but AC4 asks for each to
// have its own dedicated dump/compare.
//
// Review round 1, F5: an earlier revision of this file dumped `restsInFreeze` only from organic
// driver-trajectory states, which are ~never actually near a freeze tile (the scripted driver
// bots already avoid hazards) -- so "current dumps are ~all zeros". This revision additionally
// biases a fraction of `seal`/`shield` cases toward freeze-adjacent tiles (same technique
// `gen-planner-dump.mjs`'s `--danger-bias`/freeze-edge injection uses) and adds `touchesFreeze`,
// `sealedIn`, `escapeExists`, `saferInput`, and the full `hazardField`/`unfreezeField` grids,
// which had no dedicated dump at all before.
//
// Usage: node gen-component-dump.mjs --map <path|synthetic:name> --seed <u32> --cases <n> --out <path.jsonl>

import { writeFileSync, mkdirSync, appendFileSync } from "node:fs";
import { dirname } from "node:path";
import { SimWorld, loadMapCollision, teeStateJson, inputJson, f64Bits, sha256File, TS_REF } from "./lib.mjs";

const { scriptedAction } = await import(`${TS_REF}/src/env/scripted.ts`);
const { restsInFreeze, touchesFreeze, sealedIn } = await import(`${TS_REF}/src/plan/seal.ts`);
const { throwLines, frozenThrowLines } = await import(`${TS_REF}/src/plan/throwLines.ts`);
const { escapeExists, saferInput } = await import(`${TS_REF}/src/plan/shield.ts`);
const { hazardField, unfreezeField } = await import(`${TS_REF}/src/plan/planner.ts`);
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
const wantCases = Number(args.cases ?? "500");
const outPath = args.out;
if (!mapArg || !outPath) {
  console.error("usage: gen-component-dump.mjs --map <path|synthetic:name> --seed <u32> --cases <n> --out <path>");
  process.exit(2);
}

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

// Standable tiles touching a freeze tile (radius 1, Chebyshev), plus the direction (in tiles) to
// it -- review round 1, F5: "restsInFreeze states chosen near freeze tiles (current dumps are
// ~all zeros)".
function findFreezeEdge(col, stand) {
  const out = [];
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
    if (best !== null) out.push({ x: p.x, y: p.y, dx: best.ox, dy: best.oy });
  }
  return out;
}
const freezeEdge = findFreezeEdge(col, stand);
process.stderr.write(`${outPath}: ${stand.length} standable tiles, ${freezeEdge.length} freeze-adjacent\n`);

const driveRng = new Rng((seed * 2654435761) >>> 0);
const scriptRngA = new Rng((seed * 7919 + 17) >>> 0);
const scriptRngB = new Rng((seed * 7919 + 19) >>> 0);
const driver = new SimWorld(col, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
function pickSpawn() {
  const p = stand[Math.floor(driveRng.nextFloat() * stand.length)];
  return { x: p.x, y: p.y };
}
driver.addTee(0, pickSpawn());
driver.addTee(1, pickSpawn());

let prevA = emptyInput();
let prevB = emptyInput();
const scriptedCases = [];
const sealCases = [];
const throwCases = [];

// throwLines/frozenThrowLines: pure functions of (steps, at) -- a handful of representative
// (steps, at) pairs is exhaustive (no world/RNG dependence at all).
for (const steps of [5, 9, 12]) {
  for (const at of [0, 0.7, -1.9, Math.PI - 0.01]) {
    throwCases.push({ kind: "throwlines", steps, at: f64Bits(at), lines: throwLines(steps, at).map((l) => l.map((s) => ({ dir: s.dir, jump: s.jump, hook: s.hook, fire: s.fire, aim: f64Bits(s.aim) }))) });
    throwCases.push({
      kind: "frozenthrowlines",
      steps,
      at: f64Bits(at),
      lines: frozenThrowLines(steps, at).map((l) => l.map((s) => ({ dir: s.dir, jump: s.jump, hook: s.hook, fire: s.fire, aim: f64Bits(s.aim) }))),
    });
  }
}

// --- hazard/unfreeze BFS fields (review round 1, F5): the full grid, once -- an exact,
// bit-for-bit (well, plain-integer-for-integer: `dist` is `Int32Array`) comparison target for
// `fields::hazard_field`/`unfreeze_field`, which previously had no dedicated dump at all. -------
const hf = hazardField(col);
const uf = unfreezeField(col);
const fieldsCase = { kind: "fields", width: hf.width, height: hf.height, hazardDist: Array.from(hf.dist), unfreezeDist: Array.from(uf.dist) };

// --- touchesFreeze: freeze-adjacent tiles (some literally touch a freeze/death tile -> true,
// their standable neighbor doesn't -> mixed) plus a batch of plain standable tiles (false). -----
const touchesCases = [];
for (const p of freezeEdge) {
  touchesCases.push({ kind: "touchesfreeze", x: f64Bits(p.x), y: f64Bits(p.y), touches: touchesFreeze(col, p.x, p.y) });
  const fx = p.x + p.dx * 32;
  const fy = p.y + p.dy * 32;
  touchesCases.push({ kind: "touchesfreeze", x: f64Bits(fx), y: f64Bits(fy), touches: touchesFreeze(col, fx, fy) });
}
for (const p of stand.slice(0, Math.min(stand.length, 200))) {
  touchesCases.push({ kind: "touchesfreeze", x: f64Bits(p.x), y: f64Bits(p.y), touches: touchesFreeze(col, p.x, p.y) });
}

// --- shield.ts (escapeExists/saferInput) + seal.ts's sealedIn: review round 1, F5 -- "current
// dumps are ~all zeros" because organic driver-trajectory states are almost never actually near a
// freeze tile (the scripted driver already avoids hazards); bias 70% of these cases to a
// freeze-adjacent tile with velocity aimed at the freeze (same technique
// `gen-planner-dump.mjs`'s freeze-edge injection uses for `shielded`), so `escapeExists` actually
// has a mix of "yes" and "no" verdicts to be right or wrong about, not just always-yes. ----------
const shieldCases = [];
const wantShield = Math.min(wantCases, 400);
const shieldRng = new Rng((seed * 104729 + 3) >>> 0);
for (let i = 0; i < wantShield; i++) {
  const nearFreeze = freezeEdge.length > 0 && shieldRng.nextFloat() < 0.7;
  const p = nearFreeze ? freezeEdge[Math.floor(shieldRng.nextFloat() * freezeEdge.length)] : stand[Math.floor(shieldRng.nextFloat() * stand.length)];
  const enemyP = stand[Math.floor(shieldRng.nextFloat() * stand.length)];
  let vel;
  if (nearFreeze) {
    const norm = Math.hypot(p.dx, p.dy) || 1;
    vel = { x: (p.dx / norm) * 8, y: (p.dy / norm) * 8 };
  } else {
    vel = { x: (shieldRng.nextFloat() * 2 - 1) * 6, y: 0 };
  }
  const dir = vel.x > 0 ? 1 : vel.x < 0 ? -1 : 0;
  const holdInput = { ...emptyInput(), direction: dir };
  const holdTicks = 4;

  const sim = new SimWorld(col, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
  sim.addTee(0, { x: p.x, y: p.y });
  sim.addTee(1, { x: enemyP.x, y: enemyP.y });
  const base0 = sim.getTee(0);
  sim.applyTeeState(0, { ...base0, pos: { x: p.x, y: p.y }, vel, frozen: false, freezeTicksLeft: 0 });
  const others = new Map([[1, emptyInput()]]);
  const escExists = escapeExists(sim, 0, holdInput, holdTicks, others);
  const safer = saferInput(sim, 0, holdInput, holdTicks, others, undefined);

  // sealedIn strips every tee but `id` from its world -- give it its own fresh one, same self
  // state. `startFrozen` (already deep-frozen) exercises `sealedIn`'s `[held]`-only branch;
  // otherwise it exercises the `escapes(held)` branch (freezes only partway through the rollout).
  const startFrozen = nearFreeze && shieldRng.nextFloat() < 0.3;
  const sealWorld = new SimWorld(col, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
  sealWorld.addTee(0, { x: p.x, y: p.y });
  const base0b = sealWorld.getTee(0);
  const state = { ...base0b, pos: { x: p.x, y: p.y }, vel, frozen: startFrozen, freezeTicksLeft: startFrozen ? 120 : 0 };
  const sealed = sealedIn(sealWorld, 0, state, holdInput);

  shieldCases.push({
    kind: "shield",
    posX: f64Bits(p.x),
    posY: f64Bits(p.y),
    velX: f64Bits(vel.x),
    velY: f64Bits(vel.y),
    frozen: startFrozen,
    freezeTicksLeft: startFrozen ? 120 : 0,
    enemyX: f64Bits(enemyP.x),
    enemyY: f64Bits(enemyP.y),
    input: inputJson(holdInput),
    holdTicks,
    escapeExists: escExists,
    saferInput: safer === null ? null : inputJson(safer),
    sealedIn: sealed,
  });
}

for (let tick = 0; scriptedCases.length < wantCases && tick < 200000; tick++) {
  const selfBefore = driver.getTee(0);
  const enemyBefore = driver.getTee(1);
  if (selfBefore === undefined || enemyBefore === undefined) break;
  if (!selfBefore.alive || !enemyBefore.alive) {
    driver.applyTeeState(0, { ...selfBefore, alive: true, freezeTicksLeft: 0, hookState: 0, hookedPlayer: -1, pos: pickSpawn(), vel: { x: 0, y: 0 } });
    driver.applyTeeState(1, { ...enemyBefore, alive: true, freezeTicksLeft: 0, hookState: 0, hookedPlayer: -1, pos: pickSpawn(), vel: { x: 0, y: 0 } });
    prevA = emptyInput();
    prevB = emptyInput();
    continue;
  }

  // scriptedAction: dump the RNG state before the call (both s0..s3, haveSpare, spare -- the
  // full public surface `ddai_jsmath::Rng` exposes too) + inputs, then the decoded output.
  const rngStateBefore = { s0: scriptRngA.s0, s1: scriptRngA.s1, s2: scriptRngA.s2, s3: scriptRngA.s3, haveSpare: scriptRngA.haveSpare, spare: f64Bits(scriptRngA.spare) };
  const outA = scriptedAction(driver, 0, 1, prevA, scriptRngA);
  scriptedCases.push({ kind: "scripted", tick, self: teeStateJson(selfBefore), enemy: teeStateJson(enemyBefore), prev: inputJson(prevA), rngBefore: rngStateBefore, out: inputJson(outA) });

  // restsInFreeze: same self/enemy states, both directions (self ballistic from its own pos/vel,
  // and from the enemy's, to vary the sampled trajectories).
  sealCases.push({ kind: "seal", tick, posX: f64Bits(selfBefore.pos.x), posY: f64Bits(selfBefore.pos.y), velX: f64Bits(selfBefore.vel.x), velY: f64Bits(selfBefore.vel.y), rests: restsInFreeze(col, selfBefore.pos, selfBefore.vel) });
  sealCases.push({ kind: "seal", tick, posX: f64Bits(enemyBefore.pos.x), posY: f64Bits(enemyBefore.pos.y), velX: f64Bits(enemyBefore.vel.x), velY: f64Bits(enemyBefore.vel.y), rests: restsInFreeze(col, enemyBefore.pos, enemyBefore.vel) });

  // Review round 1, F5: also probe `restsInFreeze` from a freeze-adjacent tile with velocity
  // aimed at the freeze -- the organic driver states above almost never land anywhere near one.
  //
  // Review round 2, F14: this still dumped ~all `rests=false` on Copy Love Box specifically
  // (0/1200, confirmed by direct measurement) -- `restsInFreeze`'s own loop breaks out on tick 0
  // whenever the tee starts grounded (`isSolid` under it) with `vel.y >= 0` (`seal.ts`'s `grounded
  // && vy >= 0` check, before any position update), so a *purely horizontal* nudge from a
  // standable, freeze-adjacent tile (`p.dy === 0`, i.e. the nearest freeze tile is directly to one
  // side, not above/below) can never explore a single tick of ballistic flight -- it always
  // returns `touchesFreeze` of the untouched starting point, which is essentially never true for a
  // tile that is itself standable ground. CLB's freeze-adjacent tiles happen to be predominantly
  // this `dy === 0` shape (measured: 38/84); BlmapChill's happen not to be, which is why only CLB
  // showed the failure. Fix: give `dy === 0` cases a small upward kick (as if just knocked/jumped,
  // matching how `restsInFreeze` is actually invoked in `planner.ts`/`bot.ts` -- always on a tee
  // whose velocity is its true post-hit trajectory, not a resting one) so the sim actually runs;
  // `dy !== 0` cases already carry a real vertical component and are left as they were. Measured
  // effect (dedicated `Rng`-seeded probe, `n=1200`, this same selection/velocity scheme): CLB
  // 0/1200 -> 269/1200 true; BlmapChill (which this probe alone -- not counting the organic-tick
  // lines above, which already contribute non-zero trues there -- previously gave only 3/1200 true
  // for) 3/1200 -> 215/1200 true.
  if (freezeEdge.length > 0) {
    const p = freezeEdge[Math.floor(driveRng.nextFloat() * freezeEdge.length)];
    const norm = Math.hypot(p.dx, p.dy) || 1;
    const kick = driveRng.nextFloat();
    let vy = (p.dy / norm) * 10;
    if (p.dy === 0) vy -= 4 + kick * 4;
    const vel = { x: (p.dx / norm) * 10, y: vy };
    sealCases.push({ kind: "seal", tick, posX: f64Bits(p.x), posY: f64Bits(p.y), velX: f64Bits(vel.x), velY: f64Bits(vel.y), rests: restsInFreeze(col, { x: p.x, y: p.y }, vel) });
  }

  const outB = scriptedAction(driver, 1, 0, prevB, scriptRngB);
  driver.setInput(0, outA);
  driver.setInput(1, outB);
  driver.step();
  prevA = outA;
  prevB = outB;
}

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify({ kind: "meta", version: 1, mapPath, mapSha256, seed }) + "\n");
appendFileSync(outPath, JSON.stringify(fieldsCase) + "\n");
for (const c of [...throwCases, ...scriptedCases, ...sealCases, ...touchesCases, ...shieldCases]) appendFileSync(outPath, JSON.stringify(c) + "\n");
const shielded = shieldCases.filter((c) => !c.escapeExists).length;
process.stderr.write(
  `${outPath}: ${throwCases.length} throwLines cases, ${scriptedCases.length} scripted cases, ${sealCases.length} seal cases, ` +
    `${touchesCases.length} touchesFreeze cases, ${shieldCases.length} shield cases (${shielded} with escapeExists=false)\n`,
);
