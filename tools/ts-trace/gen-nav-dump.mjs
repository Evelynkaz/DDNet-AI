#!/usr/bin/env node
// Task 4.2 acceptance criterion 2: parity dumps for the deterministic navigation code of the REAL
// TS sources — `route.ts`'s `findRoute`/`deadZone`/`spawnTiles`, `crossing.ts`'s search choice,
// `wayblock.ts`'s side chooser/spot choice (the last two are added by their own sections below).
// `ddai-nav`'s `tests/parity_nav.rs` replays every line against the Rust port (feature `ts-parity`)
// and requires zero mismatches.
//
// Usage: node gen-nav-dump.mjs --map <path> --seed <u32> --routes <n> --out <path.jsonl>
//        [--section routes|deadzone|all]

import { writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { mkdirSync } from "node:fs";
import { loadMapCollision, sha256File, TS_REF, SimWorld, f64Bits, teeStateJson } from "./lib.mjs";
import { createRequire } from "node:module";

const { findRoute, deadZone, spawnTiles } = await import(`${TS_REF}/src/plan/route.ts`);
const { Rng } = await import(`${TS_REF}/src/nn/rng.ts`);
const { emptyInput } = await import(`${TS_REF}/src/core/types.ts`);
const { WAYBLOCKS, WbSideChooser, wayblockFor, sideAt, inWbZone, inWbHall, inWbLeash, wbWalkAllowed } = await import(`${TS_REF}/src/bot/wayblock.ts`);
const { SwingCrosser } = await import(`${TS_REF}/src/bot/crossing.ts`);
const { Navigator, tileGoal } = await import(`${TS_REF}/src/bot/navigate.ts`);
// `bot.ts` needs `node_modules` (the `teeworlds` package): run `npm ci` in `tools/ts-reference` first
// (see its README).
const { DdnetBot } = await import(`${TS_REF}/src/bot/bot.ts`);

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) if (argv[i].startsWith("--")) out[argv[i].slice(2)] = argv[++i];
  return out;
}
const args = parseArgs(process.argv.slice(2));
const mapPath = args.map;
const seed = Number(args.seed ?? "1") >>> 0;
const nRoutes = Number(args.routes ?? "1000");
const outPath = args.out;
const section = args.section ?? "all";
if (!mapPath || !outPath) {
  console.error("usage: gen-nav-dump.mjs --map <path> --seed <u32> --routes <n> --out <path> [--section routes|deadzone|all]");
  process.exit(2);
}
const { collision: col } = loadMapCollision(mapPath);
const mapSha256 = sha256File(mapPath);
const W = col.width;
const H = col.height;

const free = (tx, ty) => {
  const px = tx * 32 + 16;
  const py = ty * 32 + 16;
  return !col.isSolid(px, py) && !col.isFreeze(px, py) && !col.isDeath(px, py);
};
const standable = [];
const freeTiles = [];
for (let ty = 0; ty < H - 1; ty++) {
  for (let tx = 0; tx < W; tx++) {
    if (!free(tx, ty)) continue;
    freeTiles.push({ tx, ty });
    if (col.isSolid(tx * 32 + 16, ty * 32 + 48)) standable.push({ tx, ty });
  }
}
process.stderr.write(`${mapPath}: ${W}x${H}, ${standable.length} standable, ${freeTiles.length} free\n`);

const rng = new Rng((seed * 2654435761) >>> 0);
const pick = (list) => list[Math.floor(rng.nextFloat() * list.length)];
const centre = (t) => t * 32 + 16;
const lines = [];
lines.push(JSON.stringify({ kind: "meta", mapPath, mapSha256, width: W, height: H, seed }));

function stepJson(s) {
  return { x: s.x, y: s.y, kind: s.kind, ax: s.anchorX ?? -1, ay: s.anchorY ?? -1, freeze: s.freeze === true, tele: s.tele === true, move: s.move ?? -1, leap: s.leap === true };
}
function resultJson(r) {
  return r === null ? null : { cost: r.cost, steps: r.steps.map(stepJson) };
}

if (section === "all" || section === "spawns" || section === "deadzone") {
  const spawns = spawnTiles(col);
  lines.push(JSON.stringify({ kind: "spawns", tiles: spawns.map((p) => [p.x, p.y]) }));
}
if (section === "all" || section === "deadzone") {
  const spawns = spawnTiles(col);
  const cells = deadZone(col, spawns);
  let n = 0;
  for (const v of cells) n += v;
  // run-length encoded 0/1 cells: [first value, run lengths...]
  const runs = [];
  let cur = cells[0];
  let len = 0;
  for (const v of cells) {
    if (v === cur) len++;
    else {
      runs.push(len);
      cur = v;
      len = 1;
    }
  }
  runs.push(len);
  lines.push(JSON.stringify({ kind: "deadzone", width: W, height: H, first: cells[0], runs, count: n }));
  process.stderr.write(`deadZone: ${n} dead tiles, ${spawns.length} spawns\n`);
}

if (section === "all" || section === "routes") {
  const OPTS = [
    { nearTiles: 2 },
    { nearTiles: 2, throughFreeze: false },
    { nearTiles: 1, allowKill: true },
    { nearTiles: 3, partial: true, allowKill: true, throughFreeze: false },
    { nearTiles: 3, partial: false, maxNodes: 4000, throughFreeze: false },
    { nearTiles: 2, throughFreeze: true, allowKill: true },
    { nearTiles: 3, maxNodes: 20000, throughFreeze: false },
  ];
  let found = 0;
  let hooks = 0;
  let freezes = 0;
  for (let q = 0; q < nRoutes; q++) {
    const from = pick(rng.nextFloat() < 0.8 ? standable : freeTiles);
    // goals: mostly within a few dozen tiles (routes exist), sometimes anywhere
    let to;
    if (rng.nextFloat() < 0.75) {
      const r = 8 + Math.floor(rng.nextFloat() * 50);
      for (let k = 0; k < 40; k++) {
        const cand = pick(standable);
        if (Math.abs(cand.tx - from.tx) <= r && Math.abs(cand.ty - from.ty) <= r) {
          to = cand;
          break;
        }
      }
    }
    to = to ?? pick(rng.nextFloat() < 0.8 ? standable : freeTiles);
    const base = { ...OPTS[q % OPTS.length] };
    let avoid = null;
    if (q % 5 === 4) {
      // an `avoid` set: route once, then forbid a few of its moves (the navigator's alternative-route path)
      const first = findRoute(col, { x: centre(from.tx), y: centre(from.ty) }, { x: centre(to.tx), y: centre(to.ty) }, base);
      if (first !== null && first.steps.length > 3) {
        avoid = [];
        const n = 1 + Math.floor(rng.nextFloat() * 3);
        for (let k = 0; k < n; k++) avoid.push(first.steps[Math.floor(rng.nextFloat() * first.steps.length)].move);
      }
    }
    const opts = { ...base };
    if (avoid !== null) opts.avoid = new Set(avoid);
    const fromPx = { x: centre(from.tx) + (rng.nextFloat() < 0.3 ? Math.floor(rng.nextFloat() * 20) - 10 : 0), y: centre(from.ty) };
    const toPx = { x: centre(to.tx), y: centre(to.ty) };
    const res = findRoute(col, fromPx, toPx, opts);
    if (res !== null) {
      found++;
      for (const s of res.steps) {
        if (s.kind === "hook") hooks++;
        if (s.freeze === true) freezes++;
      }
    }
    lines.push(
      JSON.stringify({
        kind: "route",
        from: [fromPx.x, fromPx.y],
        to: [toPx.x, toPx.y],
        opts: { nearTiles: base.nearTiles, partial: base.partial === true, allowKill: base.allowKill === true, throughFreeze: base.throughFreeze !== false, maxNodes: base.maxNodes ?? 200000, avoid: avoid ?? [] },
        result: resultJson(res),
      }),
    );
  }
  process.stderr.write(`routes: ${nRoutes} queries, ${found} found, ${hooks} hook steps, ${freezes} freeze steps\n`);
}

// --- wayblock.ts: `wayblockFor`, the zone predicates, the side chooser, `wbSpot` ---------------------
if (section === "all" || section === "wb") {
  const names = ["Copy Love Box", "Copy Love Box JoniTee", "  copy love box ", "COPY LOVE BOX JONITEE", "BlmapChill", "ChillBlock5", ""];
  lines.push(
    JSON.stringify({
      kind: "wayblockfor",
      cases: names.map((n) => {
        const withCol = wayblockFor(n, col);
        const noCol = wayblockFor(n);
        return { name: n, withCol: withCol === null ? null : withCol.name, noCol: noCol === null ? null : noCol.name };
      }),
    }),
  );
  const def = WAYBLOCKS[0];
  // zone predicates on tiles in and around both halls and the tubes, plus random ones
  const zr = new Rng((seed * 7919 + 5) >>> 0);
  const tilesWb = [];
  for (let i = 0; i < 4000; i++) {
    const near = zr.nextFloat() < 0.8;
    const tx = near ? 70 + Math.floor(zr.nextFloat() * 100) : Math.floor(zr.nextFloat() * W);
    const ty = near ? Math.floor(zr.nextFloat() * 100) : Math.floor(zr.nextFloat() * H);
    tilesWb.push([tx, ty]);
  }
  for (const d of WAYBLOCKS) {
    const rows = tilesWb.map(([tx, ty]) => [tx, ty, sideAt(d, tx, ty) ?? "-", inWbZone(d, "left", tx, ty) ? 1 : 0, inWbZone(d, "right", tx, ty) ? 1 : 0, inWbHall(d, "left", tx, ty) ? 1 : 0, inWbHall(d, "right", tx, ty) ? 1 : 0, inWbLeash(d, "left", tx, ty) ? 1 : 0, inWbLeash(d, "right", tx, ty) ? 1 : 0, wbWalkAllowed(d, tx, ty) ? 1 : 0]);
    lines.push(JSON.stringify({ kind: "wbzones", def: d.name, rows }));
  }
  // the side chooser: random update/adopt/reset sequences
  const cr = new Rng((seed * 104729 + 11) >>> 0);
  for (let q = 0; q < 1500; q++) {
    const chooser = new WbSideChooser();
    const ops = [];
    let tick = Math.floor(cr.nextFloat() * 2000);
    const n = 6 + Math.floor(cr.nextFloat() * 20);
    for (let k = 0; k < n; k++) {
      tick += Math.floor(cr.nextFloat() * (cr.nextFloat() < 0.2 ? 600 : 120)) - (cr.nextFloat() < 0.03 ? 300 : 0);
      const r = cr.nextFloat();
      if (r < 0.06) {
        chooser.reset();
        ops.push({ op: "reset", side: chooser.side });
      } else if (r < 0.12) {
        const side = cr.nextFloat() < 0.5 ? "left" : "right";
        chooser.adopt(side);
        ops.push({ op: "adopt", arg: side, side: chooser.side });
      } else {
        const counts = { left: Math.floor(cr.nextFloat() * 4), right: Math.floor(cr.nextFloat() * 4) };
        const h = cr.nextFloat();
        const here = h < 0.34 ? null : h < 0.67 ? "left" : "right";
        const nearer = cr.nextFloat() < 0.5 ? "left" : "right";
        const out = chooser.update(counts, here, tick, nearer);
        ops.push({ op: "update", left: counts.left, right: counts.right, here, tick, nearer, side: out });
      }
    }
    lines.push(JSON.stringify({ kind: "wbchooser", ops }));
  }
  // wbSpot: the REAL DdnetBot method, on a stand-in `this`

  const sr = new Rng((seed * 15485863 + 7) >>> 0);
  for (let q = 0; q < 2000; q++) {
    const sideName = sr.nextFloat() < 0.5 ? "left" : "right";
    const sd = sideName === "left" ? def.left : def.right;
    const tees = [];
    const nT = Math.floor(sr.nextFloat() * 6);
    for (let i = 0; i < nT; i++) {
      const spot = sd.spots[Math.floor(sr.nextFloat() * sd.spots.length)];
      const onSpot = sr.nextFloat() < 0.6;
      const x = (onSpot ? spot.tx : spot.tx + Math.floor(sr.nextFloat() * 14) - 7) * 32 + 16 + Math.floor(sr.nextFloat() * 40) - 20;
      const y = (onSpot ? spot.ty : spot.ty + Math.floor(sr.nextFloat() * 10) - 5) * 32 + 16 + Math.floor(sr.nextFloat() * 20) - 10;
      tees.push({ id: i + 1, alive: sr.nextFloat() < 0.9, frozen: sr.nextFloat() < 0.25, pos: { x, y } });
    }
    const friends = new Set(tees.filter(() => sr.nextFloat() < 0.4).map((t) => t.id));
    const ownId = 0;
    let here;
    const hr = sr.nextFloat();
    if (hr < 0.3) here = undefined;
    else {
      const z = sd.zone[Math.floor(sr.nextFloat() * sd.zone.length)];
      here = hr < 0.8 ? { tx: z.x0 + Math.floor(sr.nextFloat() * (z.x1 - z.x0 + 1)), ty: z.y0 + Math.floor(sr.nextFloat() * (z.y1 - z.y0 + 1)) } : { tx: sd.spots[0].tx + Math.floor(sr.nextFloat() * 9) - 4, ty: sd.spots[0].ty + Math.floor(sr.nextFloat() * 14) - 2 };
    }
    const fake = { world: { allTees: () => tees }, isPartnerNow: () => false, isFriendId: (id) => friends.has(id) };
    const spot = DdnetBot.prototype.wbSpot.call(fake, ownId, def, sideName, here);
    lines.push(
      JSON.stringify({
        kind: "wbspot",
        side: sideName,
        here: here === undefined ? null : [here.tx, here.ty],
        tees: tees.map((t) => ({ id: t.id, alive: t.alive, frozen: t.frozen, x: f64Bits(t.pos.x), y: f64Bits(t.pos.y) })),
        friends: [...friends],
        spot: [spot.tx, spot.ty],
      }),
    );
  }
}

// --- crossing.ts: whole crossing traces (every input, tick by tick) -------------------------------
if (section === "all" || section === "cross") {
  const def = wayblockFor("Copy Love Box", col);
  if (def === null) process.stderr.write("cross: this map has no wayblock definition; skipped\n");
  else {
    const xr = new Rng((seed * 49979687 + 13) >>> 0);
    const nCross = Number(args.crossings ?? "60");
    let arrived = 0;
    let failed = 0;
    for (let q = 0; q < nCross; q++) {
      const c = def.crossings[q % 2];
      const lag = [0, 2, 4, 1][Math.floor(xr.nextFloat() * 4)];
      // start: a standable tile in the `from` boxes near the tube start (as the navigator arrives)
      const cands = [];
      for (const b of c.from) for (let ty = b.y0; ty <= b.y1; ty++) for (let tx = b.x0; tx <= b.x1; tx++) {
        if (Math.abs(tx - c.start.tx) > 12 || Math.abs(ty - c.start.ty) > 6) continue;
        const px = tx * 32 + 16;
        const py = ty * 32 + 16;
        if (col.isSolid(px, py) || col.isFreeze(px, py) || col.isDeath(px, py)) continue;
        if (col.isSolid(px, py + 32)) cands.push({ tx, ty });
      }
      if (cands.length === 0) continue;
      const st = cands[Math.floor(xr.nextFloat() * cands.length)];
      const world = new SimWorld(col, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
      const pos = { x: st.tx * 32 + 16, y: st.ty * 32 + 16 };
      world.addTee(0, pos);
      const crosser = new SwingCrosser(col, c);
      const trace = [];
      let tick = 1000;
      let ended = "";
      for (let t = 0; t < 700; t++, tick++) {
        const me = world.getTee(0);
        const inp = crosser.step(me, tick, lag);
        trace.push([inp.direction, f64Bits(inp.targetX), f64Bits(inp.targetY), inp.jump, inp.hook]);
        if (crosser.done) {
          ended = crosser.phase;
          break;
        }
        world.setInput(0, inp);
        world.step();
      }
      if (ended === "arrived") arrived++;
      else failed++;
      lines.push(
        JSON.stringify({ kind: "crosstrace", crossing: q % 2, lag, start: [pos.x, pos.y], ended, reason: crosser.reason, rollouts: crosser.tried, ticks: trace.length, trace }),
      );
    }
    process.stderr.write(`crossings: ${nCross} traces, ${arrived} arrived, ${failed} did not\n`);
  }
}

// --- navigate.ts: whole navigator runs, tick by tick ------------------------------------------------
// The tee starts at a tile, the REAL `Navigator` drives it in the REAL `SimWorld` (f64) until it
// is done or `maxTicks` pass; a `Cl_Kill` request (`takeKill`) respawns the tee at the next spawn tile
// (a fixed cycle) with a clean state. Every input is recorded; `ddai-nav` replays the same run on the
// same world and must produce the identical stream, notes, outcome and kills.
export function respawnState(pos) {
  return {
    id: 0, alive: true, pos: { x: pos.x, y: pos.y }, vel: { x: 0, y: 0 }, hookState: 0, hookPos: { x: pos.x, y: pos.y }, hookDir: { x: 0, y: 0 }, hookedPlayer: -1,
    jumped: 0, jumpsLeft: 2, direction: 0, angle: 0, activeWeapon: 1, frozen: false, freezeTicksLeft: 0, attackTick: 0, deepFrozen: false,
  };
}
// `unstick`: like the live bot, a tee frozen for 400 ticks in a row is killed (`Cl_Kill`) and respawned.
const FROZEN_UNSTICK_TICKS = 400;
function runNav(startTile, goalTile, throughFreeze, withCrossings, maxTicks, wantTrace, unstick = false) {
  const spawns = spawnTiles(col);
  const world = new SimWorld(col, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
  const start = { x: centre(startTile.tx), y: centre(startTile.ty) };
  world.addTee(0, start);
  const def = wayblockFor("Copy Love Box", col);
  const nav = new Navigator(col, [tileGoal(col, goalTile.tx, goalTile.ty)], { throughFreeze, crossings: withCrossings && def !== null ? def.crossings : undefined });
  const trace = [];
  const kills = [];
  const notes = [];
  let tick = 1000;
  let freezes = 0;
  let wasFrozen = false;
  let nSpawn = 0;
  let frozenRun = 0;
  let t = 0;
  for (; t < maxTicks; t++, tick++) {
    const me = world.getTee(0);
    if (me.frozen && !wasFrozen) freezes++;
    wasFrozen = me.frozen;
    frozenRun = me.frozen ? frozenRun + 1 : 0;
    if (unstick && frozenRun >= FROZEN_UNSTICK_TICKS) {
      frozenRun = 0;
      kills.push(t);
      const sp = spawns.length === 0 ? start : spawns[nSpawn++ % spawns.length];
      world.applyTeeState(0, respawnState({ x: sp.x, y: sp.y }));
      nav.respawned();
      continue;
    }
    const inp = nav.step(me, tick, [], 0);
    for (const n of nav.takeNotes()) notes.push(n);
    if (wantTrace) trace.push([inp.direction, f64Bits(inp.targetX), f64Bits(inp.targetY), inp.jump, inp.hook, inp.fire]);
    if (nav.takeKill()) {
      kills.push(t);
      const sp = spawns.length === 0 ? start : spawns[nSpawn++ % spawns.length];
      world.applyTeeState(0, respawnState({ x: sp.x, y: sp.y }));
      nav.respawned();
      continue;
    }
    if (nav.done) break;
    world.setInput(0, inp);
    world.step();
  }
  const me = world.getTee(0);
  const d = Math.hypot(me.pos.x - centre(goalTile.tx), me.pos.y - centre(goalTile.ty));
  return { phase: nav.phase, outcome: nav.outcome, ticks: t, kills, freezes, notes, trace, dist: d, pos: [me.pos.x, me.pos.y] };
}
if (section === "all" || section === "nav") {
  const nr = new Rng((seed * 32452843 + 17) >>> 0);
  const nNav = Number(args.navs ?? "40");
  const maxTicks = Number(args.maxticks ?? "3000");
  let arrived = 0;
  for (let q = 0; q < nNav; q++) {
    const from = pick(standable);
    const throughFreeze = q % 3 !== 2;
    let to;
    const r = 6 + Math.floor(nr.nextFloat() * 70);
    for (let k = 0; k < 200; k++) {
      const cand = pick(standable);
      if (Math.abs(cand.tx - from.tx) <= r && Math.abs(cand.ty - from.ty) <= r && (cand.tx !== from.tx || cand.ty !== from.ty)) {
        // only pairs the route graph connects (the same search the navigator falls back to)
        const way = findRoute(col, { x: centre(from.tx), y: centre(from.ty) }, { x: centre(cand.tx), y: centre(cand.ty) }, { nearTiles: 2, allowKill: true, throughFreeze });
        if (way !== null) {
          to = cand;
          break;
        }
      }
    }
    if (to === undefined) continue;
    const withCrossings = throughFreeze;
    const res = runNav(from, to, throughFreeze, withCrossings, maxTicks, true);
    if (res.phase === "arrived") arrived++;
    lines.push(JSON.stringify({ kind: "navtrace", from: [from.tx, from.ty], to: [to.tx, to.ty], throughFreeze, withCrossings, maxTicks, phase: res.phase, outcome: res.outcome, ticks: res.ticks, kills: res.kills, notes: res.notes, trace: res.trace }));
  }
  // Copy Love Box: from every spawn tile to every wayblock spot (through the tubes, as the WB walk does)
  const wbDef = wayblockFor("Copy Love Box", col);
  if (wbDef !== null && args.wbhall !== "0") {
    const sp = spawnTiles(col).map((p) => ({ tx: Math.trunc(p.x / 32), ty: Math.trunc(p.y / 32) }));
    for (const from of sp) {
      for (const side of [wbDef.left, wbDef.right]) {
        for (const to of side.spots) {
          const res = runNav(from, to, true, true, 4000, true);
          if (res.phase === "arrived") arrived++;
          lines.push(JSON.stringify({ kind: "navtrace", from: [from.tx, from.ty], to: [to.tx, to.ty], throughFreeze: true, withCrossings: true, maxTicks: 4000, phase: res.phase, outcome: res.outcome, ticks: res.ticks, kills: res.kills, notes: res.notes, trace: res.trace }));
        }
      }
    }
  }
  process.stderr.write(`navigator: ${nNav} random runs plus the WB hall runs, ${arrived} arrived\n`);
}

// --- arrival-rate pairs: the TS navigator's own results (no traces) ---------------------------------
// `ddai-nav`'s `tests/arrival_vs_ts.rs` runs the same pairs with the Rust navigator on `World<f32>`.
if (section === "arrival") {
  const per = Number(args.pairs ?? "200");
  const ar = new Rng((seed * 86028121 + 19) >>> 0);
  const wbDef = wayblockFor("Copy Love Box", col);
  const limit = Number(args.maxticks ?? "6000");
  const spawnPx = spawnTiles(col).map((p) => ({ tx: Math.trunc(p.x / 32), ty: Math.trunc(p.y / 32) }));
  let total = 0;
  for (const throughFreeze of [true, false]) {
    let made = 0;
    const cats = [];
    // CLB: the halls and the tubes first (spawn -> spots, spots -> spawn), with the tube crossings
    // (only with through-freeze on: the tubes are freeze)
    if (wbDef !== null && throughFreeze) {
      for (const from of spawnPx) for (const side of [wbDef.left, wbDef.right]) for (const to of side.spots) cats.push({ cat: "wb-in", from, to });
      for (const side of [wbDef.left, wbDef.right]) for (const from of side.spots) for (const to of spawnPx) cats.push({ cat: "wb-out", from, to });
    }
    while (cats.length < per) {
      const from = pick(standable);
      let to;
      const r = 6 + Math.floor(ar.nextFloat() * 90);
      for (let k = 0; k < 300; k++) {
        const cand = pick(standable);
        if (Math.abs(cand.tx - from.tx) <= r && Math.abs(cand.ty - from.ty) <= r && (cand.tx !== from.tx || cand.ty !== from.ty)) {
          const way = findRoute(col, { x: centre(from.tx), y: centre(from.ty) }, { x: centre(cand.tx), y: centre(cand.ty) }, { nearTiles: 2, allowKill: true, throughFreeze });
          if (way !== null) {
            to = cand;
            break;
          }
        }
      }
      if (to !== undefined) cats.push({ cat: "random", from, to });
    }
    for (const c of cats.slice(0, Math.max(per, cats.length))) {
      const res = runNav(c.from, c.to, throughFreeze, throughFreeze, limit, false, true);
      total++;
      lines.push(JSON.stringify({ kind: "arrival", cat: c.cat, from: [c.from.tx, c.from.ty], to: [c.to.tx, c.to.ty], throughFreeze, maxTicks: limit, phase: res.phase, ticks: res.ticks, kills: res.kills.length, freezes: res.freezes, dist: res.dist, outcome: res.outcome }));
      made++;
    }
    process.stderr.write(`arrival (throughFreeze ${throughFreeze}): ${made} pairs\n`);
  }
}

// --- follow mode: the REAL `steerFollow` + `Navigator` against a scripted moving target ---------------
// The target walks a dumped pixel path at a fixed speed, pausing at waypoints; the follower is the
// real navigator driven by the real `DdnetBot.prototype.steerFollow` (on a stand-in `this`).
// The scripted target alone (no follower) in the TS world: does it ever jump more than 96 px in one tick?
// (A physical teleport of the kinematic target: the live world does it where the TS world does not, so a
// script that does it in either world is not a fair follow test; the Rust side checks its own world the
// same way, `ddai_nav::harness::target_jumps`.)
function targetJumps(path, dwellAt, speed, dwell, limit) {
  const world = new SimWorld(col, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
  world.addTee(1, { x: path[0][0], y: path[0][1] });
  let pi = 0;
  let pause = 0;
  let last = world.getTee(1).pos;
  for (let t = 0; t < limit; t++) {
    const tee1 = world.getTee(1);
    if (pause > 0) pause--;
    else if (pi < path.length - 1) {
      const tx = path[pi + 1][0];
      const ty = path[pi + 1][1];
      const d = Math.hypot(tx - tee1.pos.x, ty - tee1.pos.y);
      if (d <= speed) {
        pi++;
        world.applyTeeState(1, { ...tee1, pos: { x: tx, y: ty }, vel: { x: 0, y: 0 } });
        if (dwellAt.includes(pi)) pause = dwell;
      } else world.applyTeeState(1, { ...tee1, pos: { x: tee1.pos.x + ((tx - tee1.pos.x) / d) * speed, y: tee1.pos.y + ((ty - tee1.pos.y) / d) * speed }, vel: { x: 0, y: 0 } });
    }
    world.step();
    const now = world.getTee(1).pos;
    if (Math.hypot(now.x - last.x, now.y - last.y) > 96) return true;
    last = now;
  }
  return false;
}

function runFollow(fromTile, path, dwellAt, speed, dwell, throughFreeze, limit) {
  const spawns = spawnTiles(col);
  const world = new SimWorld(col, { svHit: true, respawnDelayTicks: 0, infiniteAmmo: true });
  const start = { x: centre(fromTile.tx), y: centre(fromTile.ty) };
  world.addTee(0, start);
  world.addTee(1, { x: path[0][0], y: path[0][1] });
  const def = wayblockFor("Copy Love Box", col);
  const navOpts = { throughFreeze, crossings: throughFreeze && def !== null ? def.crossings : undefined };
  const events = [];
  const fake = {
    nav: null,
    follow: null,
    navReturnMode: "fight",
    wbDef: def,
    mode: "goto",
    emit: (_k, m) => events.push(m),
    log: () => {},
    world: { tick: 1000, collision: col, getTee: (id) => world.getTee(id), notPlaying: () => false, allTees: () => world.allTees() },
    client: { SnapshotUnpacker: { AllObjClientInfo: [{ id: 1, name: "p1", clan: "" }] } },
    afterArrival: () => "",
    endNav() {
      this.nav = null;
      this.follow = null;
    },
  };
  const P = DdnetBot.prototype;
  fake.followTile = (pos) => P.followTile.call(fake, pos);
  fake.followGoal = (tile, name) => P.followGoal.call(fake, tile, name);
  fake.navOptsFor = (o) => P.navOptsFor.call(fake, o);
  const t0 = world.getTee(1);
  const goal = fake.followTile(t0.pos) ?? { tx: Math.trunc(t0.pos.x / 32), ty: Math.trunc(t0.pos.y / 32) };
  fake.nav = new Navigator(col, [fake.followGoal(goal, "p1")], fake.navOptsFor(navOpts));
  fake.follow = { id: 1, name: "p1", tx: goal.tx, ty: goal.ty, routedTick: 1000, seenTick: 1000, startTick: 1000, best: Math.hypot(start.x - t0.pos.x, start.y - t0.pos.y), bestTick: 1000, fails: 0, blockedAt: -1, deaths: 0, selfDeaths: 0, last: { x: t0.pos.x, y: t0.pos.y }, waiting: false, navOpts };
  let pi = 0;
  let pause = 0;
  let frozenRun = 0;
  let kills = 0;
  let freezes = 0;
  let wasFrozen = false;
  let nSpawn = 0;
  let tick = 1000;
  let t = 0;
  let ended = "";
  for (; t < limit; t++, tick++) {
    fake.world.tick = tick;
    // the target
    const tee1 = world.getTee(1);
    if (pause > 0) pause--;
    else if (pi < path.length - 1) {
      const tx = path[pi + 1][0];
      const ty = path[pi + 1][1];
      const d = Math.hypot(tx - tee1.pos.x, ty - tee1.pos.y);
      if (d <= speed) {
        pi++;
        world.applyTeeState(1, { ...tee1, pos: { x: tx, y: ty }, vel: { x: 0, y: 0 } });
        if (dwellAt.includes(pi)) pause = dwell;
      } else world.applyTeeState(1, { ...tee1, pos: { x: tee1.pos.x + ((tx - tee1.pos.x) / d) * speed, y: tee1.pos.y + ((ty - tee1.pos.y) / d) * speed }, vel: { x: 0, y: 0 } });
    }
    const me = world.getTee(0);
    if (me.frozen && !wasFrozen) freezes++;
    wasFrozen = me.frozen;
    frozenRun = me.frozen ? frozenRun + 1 : 0;
    if (frozenRun >= FROZEN_UNSTICK_TICKS) {
      frozenRun = 0;
      kills++;
      const sp = spawns.length === 0 ? start : spawns[nSpawn++ % spawns.length];
      world.applyTeeState(0, respawnState({ x: sp.x, y: sp.y }));
      if (fake.nav !== null) fake.nav.respawned();
      continue;
    }
    const verdict = P.steerFollow.call(fake, me);
    if (fake.follow === null) {
      ended = events[events.length - 1] ?? "";
      break;
    }
    let inp;
    if (verdict === "go" && fake.nav !== null) {
      inp = fake.nav.step(me, tick, [], 0);
      for (const n of fake.nav.takeNotes()) void n;
      if (fake.nav.takeKill()) {
        kills++;
        const sp = spawns.length === 0 ? start : spawns[nSpawn++ % spawns.length];
        world.applyTeeState(0, respawnState({ x: sp.x, y: sp.y }));
        fake.nav.respawned();
        continue;
      }
    } else inp = emptyInput();
    world.setInput(0, inp);
    world.step();
  }
  const me = world.getTee(0);
  const tee1 = world.getTee(1);
  return { ended, ticks: t, kills, freezes, dist: Math.hypot(me.pos.x - tee1.pos.x, me.pos.y - tee1.pos.y), events };
}
if (section === "follow") {
  const { DdnetBot: _unused } = { DdnetBot };
  const fr = new Rng((seed * 67867979 + 23) >>> 0);
  const nFollow = Number(args.pairs ?? "200");
  const limit = Number(args.maxticks ?? "6000");
  let made = 0;
  let wins = 0;
  let skippedJumps = 0;
  for (const throughFreeze of [true, false]) {
    let m = 0;
    let guard = 0;
    while (m < nFollow && guard++ < nFollow * 40) {
      const from = pick(standable);
      // waypoints of the target: 3 connected standable tiles
      const wps = [pick(standable)];
      for (let k = 0; k < 2; k++) {
        for (let tries = 0; tries < 80; tries++) {
          const cand = pick(standable);
          const prev = wps[wps.length - 1];
          if (Math.abs(cand.tx - prev.tx) > 60 || Math.abs(cand.ty - prev.ty) > 40) continue;
          const way = findRoute(col, { x: centre(prev.tx), y: centre(prev.ty) }, { x: centre(cand.tx), y: centre(cand.ty) }, { nearTiles: 1, throughFreeze: false });
          if (way !== null) {
            wps.push(cand);
            break;
          }
        }
      }
      if (wps.length < 2) continue;
      // the target's path in px: the route steps between consecutive waypoints
      const path = [[centre(wps[0].tx), centre(wps[0].ty)]];
      const dwellAt = [];
      let ok = true;
      for (let k = 0; k + 1 < wps.length; k++) {
        const way = findRoute(col, { x: centre(wps[k].tx), y: centre(wps[k].ty) }, { x: centre(wps[k + 1].tx), y: centre(wps[k + 1].ty) }, { nearTiles: 0, throughFreeze: false });
        // A teleporter on the target's way would teleport it physically in the live world (to an exit
        // of its own choosing) and not in the TS world: such scripts are not comparable, skip them.
        if (way === null || way.steps.some((st) => st.tele === true)) {
          ok = false;
          break;
        }
        for (const st of way.steps) path.push([centre(st.x), centre(st.y)]);
        dwellAt.push(path.length - 1);
      }
      if (!ok) continue;
      // only follows the follower's route graph can connect at the start
      const startWay = findRoute(col, { x: centre(from.tx), y: centre(from.ty) }, { x: path[0][0], y: path[0][1] }, { nearTiles: 2, allowKill: true, throughFreeze });
      if (startWay === null) continue;
      const speed = 3 + Math.floor(fr.nextFloat() * 3);
      const dwell = 80;
      if (targetJumps(path, dwellAt, speed, dwell, limit)) {
        skippedJumps++;
        continue;
      }
      const res = runFollow(from, path, dwellAt, speed, dwell, throughFreeze, limit);
      const arrived = res.ended.includes("arrived at p1");
      if (arrived) wins++;
      m++;
      made++;
      lines.push(JSON.stringify({ kind: "follow", from: [from.tx, from.ty], path, dwellAt, speed, dwell, throughFreeze, maxTicks: limit, arrived, ticks: res.ticks, kills: res.kills, freezes: res.freezes, dist: res.dist, ended: res.ended }));
    }
  }
  process.stderr.write(`follow: ${made} runs, ${wins} arrived, ${skippedJumps} scripts dropped (the target alone jumps > 96 px in the TS world)\n`);
}

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, lines.join("\n") + "\n");
