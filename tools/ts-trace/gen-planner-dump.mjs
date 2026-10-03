#!/usr/bin/env node
// Task 3.2 planner-parity dump generator. Drives the REAL, unmodified `src/plan/planner.ts`
// `Planner` (never `src/` edited, never copied) the way `bot.ts:planAction` does (a private
// `SimWorld` holding only self+enemy, `addTee(self)` then `addTee(enemy)` -- unshift order
// matters, `docs/research/orig-plan.md` §0), and dumps one JSON object per line: the full input
// to one `decide()` call (self/enemy `TeeState`, `prevInput`, `enemyInput`, the config used) and
// its full output (the chosen `PlayerInput` + `lastInfo`).
//
// Every case is independently teacher-forced and reproducible: the `Planner` is freshly
// constructed and `reset()` right before each `decide()` call, so no hidden cross-decision state
// (RNG position, `warm`, `oppSeed`, ...) needs to be serialized -- both TS and a Rust replayer
// start from the exact same well-known state (`new Planner(cfg)` then `reset()`) and only differ
// in the external inputs this file records. `budgetMs`/`hardMs` are always forced to `0` (D-017:
// fixed-iteration mode, deterministic regardless of CPU load), even for presets whose *live*
// values are nonzero.
//
// The (self, enemy) world states fed into each decision come from driving a *separate* simulation
// (two `scriptedAction` bots playing each other) forward and sampling its state every few ticks --
// this only exists to produce a diverse, "natural" distribution of positions/velocities/hook
// states/freezes to decide *from*; it is not itself part of the parity target.
//
// Usage:
//   node gen-planner-dump.mjs --map <path.map|synthetic:arena|synthetic:freeze> --seed <u32>
//     --cases <n> --preset <normal|low|strong|wb> --opponent <hold|react|mix> --out <path.jsonl>
//     [--sample-every <ticks=7>] [--max-ticks <n=200000>]

import { writeFileSync, mkdirSync, appendFileSync } from "node:fs";
import { dirname } from "node:path";
import { SimWorld, loadMapCollision, teeStateJson, inputJson, f64Bits, bitsToF64, sha256File, tsCoreCommit, TS_REF } from "./lib.mjs";

const { Planner } = await import(`${TS_REF}/src/plan/planner.ts`);
const { scriptedAction } = await import(`${TS_REF}/src/env/scripted.ts`);
const { Rng } = await import(`${TS_REF}/src/nn/rng.ts`);
const { emptyInput } = await import(`${TS_REF}/src/core/types.ts`);
const { FreezeMemory } = await import(`${TS_REF}/src/plan/memory.ts`);

// Review round 1, F4: every candidate `Planner.prototype.evaluate` scores (book seeds, CEM
// population, `notNow`, `landedThrows`, `polishRope`, the final re-score, the `planMargin` carry
// candidate -- in call order) gets appended here while `collectingCandidates` is on, mirroring
// the Rust port's `Planner::start_candidate_log`/`take_candidate_log` (`planner.rs`) so a replay
// test can compare *every* candidate's `(plan, score)` bit-for-bit, not just the final winner.
let collectingCandidates = false;
let candidateLog = [];
const origEvaluate = Planner.prototype.evaluate;
Planner.prototype.evaluate = function patchedEvaluate(...args) {
  const score = origEvaluate.apply(this, args);
  if (collectingCandidates) {
    const plan = args[4];
    candidateLog.push({
      plan: plan.map((s) => ({ dir: s.dir, jump: s.jump, hook: s.hook, fire: s.fire, aimBits: f64Bits(s.aim) })),
      scoreBits: f64Bits(score),
    });
  }
  return score;
};

// A snapshot of every piece of hidden state the real `Planner` carries between decisions
// (`docs/research/orig-plan.md` §1.15/§2.5), read directly off the instance -- TS `private` is
// compile-time only, so these are plain runtime properties. Mirrors
// `ddai_planner::planner::Planner::debug_state`/`PlannerDebugState` field-for-field.
function hiddenSnapshot(planner) {
  const rng = planner["rng"];
  const warm = planner["warm"];
  const committed = planner["committed"];
  return {
    rngS0: rng["s0"] >>> 0,
    rngS1: rng["s1"] >>> 0,
    rngS2: rng["s2"] >>> 0,
    rngS3: rng["s3"] >>> 0,
    rngHaveSpare: rng["haveSpare"],
    rngSpareBits: f64Bits(rng["spare"]),
    oppSeed: planner["oppSeed"] >>> 0,
    warm: warm === null ? null : warm.map((s) => ({ dir: s.dir, jump: s.jump, hook: s.hook, fire: s.fire, aimBits: f64Bits(s.aim) })),
    committed: committed === null ? null : inputJson(committed),
    commitLeft: planner["commitLeft"],
    lastDecideTick: planner["lastDecideTick"],
    decideGaps: [...planner["decideGaps"]],
    lastFrozen: planner["lastFrozen"],
    dirSince: planner["dirSince"],
    dirLast: planner["dirLast"],
  };
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
const wantCases = Number(args.cases ?? "250");
const presetName = args.preset ?? "normal";
const opponentName = args.opponent ?? "hold";
const outPath = args.out;
const sampleEvery = Number(args["sample-every"] ?? "7");
const maxTicks = Number(args["max-ticks"] ?? "200000");
// Review round 1, F4: "baseline" is the original case shape (nothing extra set on the `Planner`);
// "full" additionally exercises `setTravelGoal`/`setThirdTees`/`setFrozenBystanders`/
// `setSpareBystanders`/`setBand`/`setFreezeMemory`/`setOverrides` together and adds 4 extra tees
// into the decide-time `sim` (2 frozen, 2 free -- 6 tees total, `bot.ts:4748-4779`'s own call
// pattern: every one of those setters runs on every real `planAction`).
const scenarioKind = args.scenario ?? "baseline";
if (scenarioKind !== "baseline" && scenarioKind !== "full") {
  console.error(`unknown --scenario ${scenarioKind} (want baseline|full)`);
  process.exit(2);
}
// Fraction of driver spawn picks biased toward a tile within a few tiles of a freeze tile, so the
// dumped corpus actually contains `shielded=true` cases at a realistic rate (review: "raise
// shielded share to >= 10%"; baseline corpora had it at ~0.1%).
const dangerBias = Number(args["danger-bias"] ?? "0");
if (!mapArg || !outPath) {
  console.error(
    "usage: gen-planner-dump.mjs --map <path|synthetic:name> --seed <u32> --cases <n> --preset <p> --opponent <o> --out <path> [--scenario baseline|full] [--danger-bias 0..1]",
  );
  process.exit(2);
}

// --- presets (docs/research/orig-plan.md §1.2) -- budgetMs/hardMs always forced to 0 (D-017). ---
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

// --- map ------------------------------------------------------------------------------------
let col;
let mapSha256 = null;
let mapPath = null;
if (mapArg.startsWith("synthetic:")) {
  const kind = mapArg.slice("synthetic:".length);
  // Data (maps, demos, ...) lives outside the repo, `~/aiddnet/data` (CLAUDE.md) -- not derived
  // from `REPO_ROOT`, which is this checkout's own path (a worktree, not always `~/aiddnet/DDNet-AI`).
  mapPath = `${process.env.DDAI_DATA_DIR ?? "/home/ubuntu/aiddnet/data"}/maps/synthetic/${kind}.map`;
} else {
  mapPath = mapArg;
}
{
  const loaded = loadMapCollision(mapPath);
  col = loaded.collision;
  mapSha256 = sha256File(mapPath);
}

// --- standable tiles (any non-hazard tile with solid ground directly below) -----------------
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
// Review round 2, F14/F15: restrict every spawn (self, enemy, bystanders, goal/thirds/band/...)
// to the E-000 left-hall arena (WB boxes L1 `{79..104,67..79}` union L2 `{78..104,79..87}`) when
// `--hall` is passed, so a shield-focused dump is directly comparable to `phase_breakdown.rs`'s
// own `hall=true` measurements and to any future E-000 numbers.
const hallOnly = args.hall === "1" || args.hall === "true";
function inHall(px, py) {
  const tx = Math.trunc(px / 32);
  const ty = Math.trunc(py / 32);
  return (tx >= 79 && tx <= 104 && ty >= 67 && ty <= 79) || (tx >= 78 && tx <= 104 && ty >= 79 && ty <= 87);
}
const stand = findStandable(col).filter((p) => !hallOnly || inHall(p.x, p.y));
if (stand.length < 2) {
  console.error(`no standable tiles found in ${mapPath}${hallOnly ? " (--hall restricted)" : ""}`);
  process.exit(1);
}

// Standable tiles within `radius` tiles (Chebyshev) of a freeze tile -- close enough that
// `shield.ts`'s `escapeExists`/`saferInput` actually has something to react to. Also returns, for
// each tile, the offset (in tiles) to the nearest freeze tile found, so a caller can point a
// tee's velocity at it.
function findNearFreeze(col, stand, radius) {
  const out = [];
  for (const p of stand) {
    const tx = Math.trunc(p.x / 32);
    const ty = Math.trunc(p.y / 32);
    let best = null;
    let bestD = Infinity;
    for (let oy = -radius; oy <= radius; oy++) {
      for (let ox = -radius; ox <= radius; ox++) {
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
const dangerStand = dangerBias > 0 || scenarioKind === "full" ? findNearFreeze(col, stand, 6) : [];
// Radius 1 (true adjacency, vs. 6 for `dangerStand`'s general spawn bias): close enough that
// pointing velocity at the freeze tile genuinely risks entering it within `shieldHold()`'s
// few-tick window -- review round 1, F4: "raise shielded share to a realistic level (>= 10%)"
// (baseline corpora had ~0.1%). Speed 14 (tuned empirically against Copy Love Box: 8/10/12/14/16
// all land in the same ballpark, 14 measured highest) is fast enough to be a real committed move,
// not so fast the tee's own momentum carries it clean past the freeze tile before the next tick.
const freezeEdgeStand = scenarioKind === "full" ? findNearFreeze(col, stand, 1) : [];

// --- driver simulation: two scripted bots playing each other, to get a diverse, natural
// distribution of (self, enemy) states to decide from. -----------------------------------------
const driveRng = new Rng((seed * 2654435761) >>> 0);
const scriptRngA = new Rng((seed * 7919 + 17) >>> 0);
const scriptRngB = new Rng((seed * 7919 + 19) >>> 0);
const driver = new SimWorld(col, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
function pickSpawn() {
  const useDanger = dangerStand.length > 0 && driveRng.nextFloat() < dangerBias;
  const pool = useDanger ? dangerStand : stand;
  const p = pool[Math.floor(driveRng.nextFloat() * pool.length)];
  return { x: p.x, y: p.y };
}

// Review round 1, F4 scenario machinery: builds a fully-concrete, JSON-dumpable description of
// everything set on the `Planner` (and the extra tees added to the decide-time `sim`) for one
// case -- a Rust replay applies these exact values (never recomputes them), so a case is still
// self-contained/teacher-forced despite the extra state (`WB_OVER` reuses the same object the
// "strong"/"wb" presets bake into `cfg` above -- `setOverrides` at runtime is bot.ts's own
// `wbPlanOverrides(self)` pattern, exercised here regardless of which preset the case otherwise
// uses).
function vecBits(p) {
  return { xBits: f64Bits(p.x), yBits: f64Bits(p.y) };
}
function buildScenario(rng) {
  if (scenarioKind === "baseline") {
    return { kind: "baseline", goal: null, thirds: [], frozenBystanders: [], spareBystanders: [], band: null, memoryNotes: [], overrides: null, extraTees: [] };
  }
  const pick = () => stand[Math.floor(rng.nextFloat() * stand.length)];
  const pickDanger = () => (dangerStand.length > 0 ? dangerStand[Math.floor(rng.nextFloat() * dangerStand.length)] : pick());
  const goal = vecBits(pick());
  const thirds = [pick(), pick()].map(vecBits);
  const frozenBystanders = [pick(), pick()].map((p) => ({ ...vecBits(p), vxBits: f64Bits(0), vyBits: f64Bits(0) }));
  const spareBystanders = [pick()].map((p) => ({ ...vecBits(p), vxBits: f64Bits(0), vyBits: f64Bits(0) }));
  const a = pick();
  const b = pick();
  const band = {
    x0Bits: f64Bits(Math.min(a.x, b.x) - 300),
    y0Bits: f64Bits(Math.min(a.y, b.y) - 200),
    x1Bits: f64Bits(Math.max(a.x, b.x) + 300),
    y1Bits: f64Bits(Math.max(a.y, b.y) + 200),
  };
  const memoryNotes = [pickDanger(), pickDanger(), pick()].map((p) => ({ ...vecBits(p), kind: "note" }));
  const extraTees = [];
  for (let i = 0; i < 4; i++) {
    const p = pick();
    extraTees.push({ id: 2 + i, ...vecBits(p), frozen: i < 2, freezeTicksLeft: i < 2 ? 200 : 0 });
  }
  return { kind: "full", goal, thirds, frozenBystanders, spareBystanders, band, memoryNotes, overrides: "wb", extraTees };
}
function applyScenario(planner, sim, sc) {
  const bitsToXy = (o, xk, yk) => ({ x: bitsToF64(o[xk]), y: bitsToF64(o[yk]) });
  planner.setTravelGoal(sc.goal === null ? null : bitsToXy(sc.goal, "xBits", "yBits"));
  planner.setThirdTees(sc.thirds.map((p) => bitsToXy(p, "xBits", "yBits")));
  planner.setFrozenBystanders(
    sc.frozenBystanders.map((p) => bitsToXy(p, "xBits", "yBits")),
    sc.frozenBystanders.map((p) => bitsToXy(p, "vxBits", "vyBits")),
  );
  planner.setSpareBystanders(
    sc.spareBystanders.map((p) => bitsToXy(p, "xBits", "yBits")),
    sc.spareBystanders.map((p) => bitsToXy(p, "vxBits", "vyBits")),
  );
  planner.setBand(
    sc.band === null
      ? null
      : { x0: bitsToF64(sc.band.x0Bits), y0: bitsToF64(sc.band.y0Bits), x1: bitsToF64(sc.band.x1Bits), y1: bitsToF64(sc.band.y1Bits) },
  );
  if (sc.memoryNotes.length > 0) {
    const mem = new FreezeMemory(col.width, col.height);
    for (const n of sc.memoryNotes) {
      const p = bitsToXy(n, "xBits", "yBits");
      if (n.kind === "note") mem.note(p.x, p.y);
      else mem.notePass(p.x, p.y);
    }
    planner.setFreezeMemory(mem);
  } else {
    planner.setFreezeMemory(null);
  }
  planner.setOverrides(sc.overrides === "wb" ? WB_OVER : null);
  for (const t of sc.extraTees) {
    const p = bitsToXy(t, "xBits", "yBits");
    if (sim.getTee(t.id) === undefined) sim.addTee(t.id, p);
    const base = sim.getTee(t.id);
    sim.applyTeeState(t.id, { ...base, frozen: t.frozen, freezeTicksLeft: t.freezeTicksLeft, pos: p, vel: { x: 0, y: 0 } });
  }
}
driver.addTee(0, pickSpawn());
driver.addTee(1, pickSpawn());

let prevA = emptyInput();
let prevB = emptyInput();
const cases = [];
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
    tsCoreCommit: tsCoreCommit(),
    node: process.version,
    v8: process.versions.v8,
  }) + "\n",
);

for (let tick = 0; cases.length < wantCases && tick < maxTicks; tick++) {
  const selfBefore = driver.getTee(0);
  const enemyBefore = driver.getTee(1);
  if (selfBefore === undefined || enemyBefore === undefined) break;
  // Respawn drift: if either tee died (svHit/no kill tiles here, but be defensive), reset both to
  // fresh spawns so the driver simulation keeps producing useful states for the whole run.
  if (!selfBefore.alive || !enemyBefore.alive) {
    driver.applyTeeState(0, { ...selfBefore, alive: true, freezeTicksLeft: 0, hookState: 0, hookedPlayer: -1, pos: pickSpawn(), vel: { x: 0, y: 0 } });
    driver.applyTeeState(1, { ...enemyBefore, alive: true, freezeTicksLeft: 0, hookState: 0, hookedPlayer: -1, pos: pickSpawn(), vel: { x: 0, y: 0 } });
    prevA = emptyInput();
    prevB = emptyInput();
    continue;
  }
  const outA = scriptedAction(driver, 0, 1, prevA, scriptRngA);
  const outB = scriptedAction(driver, 1, 0, prevB, scriptRngB);
  driver.setInput(0, outA);
  driver.setInput(1, outB);
  driver.step();
  prevA = outA;
  prevB = outB;

  if (tick % sampleEvery !== 0) continue;
  let self = driver.getTee(0);
  const enemy = driver.getTee(1);
  if (self === undefined || enemy === undefined || !self.alive || !enemy.alive) continue;

  const scenarioRng = new Rng((((tick + 1) * 2654435761) ^ (seed * 97)) >>> 0);

  // Review round 1, F4: "raise shielded share to a realistic level (>= 10%)". Organic
  // driver-trajectory sampling almost never lands `self` somewhere `shield.ts`'s `escapeExists`
  // actually rejects (the scripted driver already avoids hazards well before getting that close),
  // so for a "full"-scenario case, override `self`'s position/velocity (a local copy fed to the
  // decide-time `sim` only -- the driver's own world/trajectory is untouched) to sit right at a
  // freeze-adjacent tile, moving toward the freeze -- a real, if engineered, "about to be
  // shielded" state, not a hand-picked outcome (whether `escapeExists` actually rejects it, and
  // hence whether `lastInfo.shielded` ends up true, is still entirely up to the real TS code).
  // Review round 2, F14: raised from 0.7 -- sabotage C (shield `ESCAPE_TICKS` 36->35) needs an
  // escape maneuver that's genuinely marginal (finishes right around tick 35/36) to have a chance
  // of flipping, which only happens near freeze at all; more freeze-adjacent placements means more
  // chances at that exact margin, not a guarantee of it.
  const freezeEdgeProb = args["force-freeze-edge"] === "1" ? 1.0 : 0.9;
  if (scenarioKind === "full" && freezeEdgeStand.length > 0 && scenarioRng.nextFloat() < freezeEdgeProb) {
    const p = freezeEdgeStand[Math.floor(scenarioRng.nextFloat() * freezeEdgeStand.length)];
    const norm = Math.hypot(p.dx, p.dy) || 1;
    const speed = 14;
    self = {
      ...self,
      pos: { x: p.x, y: p.y },
      vel: { x: (p.dx / norm) * speed, y: (p.dy / norm) * speed },
      frozen: false,
      freezeTicksLeft: 0,
    };
  }

  // --- the actual parity-relevant call: fresh Planner, fresh SimWorld, addTee(self) then
  // addTee(enemy) (bot.ts's own order, `docs/research/orig-plan.md` §0 step 1). ----------------
  const planner = new Planner(cfg);
  planner.reset();
  const sim = new SimWorld(col, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
  sim.addTee(0, self.pos);
  sim.addTee(1, enemy.pos);
  sim.applyTeeState(0, self);
  sim.applyTeeState(1, enemy);

  const scenario = buildScenario(scenarioRng);
  applyScenario(planner, sim, scenario);
  planner.setLiveTick(sim.tick);

  collectingCandidates = true;
  candidateLog = [];
  const decision = planner.decide(sim, 0, 1, prevA, outB);
  collectingCandidates = false;
  const li = planner.lastInfo;
  const hidden = hiddenSnapshot(planner);

  cases.push({
    kind: "case",
    tick,
    self: teeStateJson(self),
    enemy: teeStateJson(enemy),
    prevInput: inputJson(prevA),
    enemyInput: inputJson(outB),
    scenario,
    decision: inputJson(decision),
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
    candidateLog,
    hidden,
  });
  if (cases.length % 25 === 0) {
    process.stderr.write(`${outPath}: ${cases.length}/${wantCases} (tick ${tick})\n`);
  }
}

for (const c of cases) appendFileSync(outPath, JSON.stringify(c) + "\n");
process.stderr.write(`${outPath}: wrote ${cases.length} cases\n`);
if (cases.length < wantCases) {
  process.stderr.write(`WARNING: only got ${cases.length}/${wantCases} cases (maxTicks reached)\n`);
}
