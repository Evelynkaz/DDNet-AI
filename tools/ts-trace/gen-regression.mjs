#!/usr/bin/env node
// Hand-scripted (non-random) regression fixtures for specific reviewer-found bugs (task 1.9,
// round 2 review). Each `--case` runs a fixed, deterministic op sequence against the REAL TS
// `SimWorld` and dumps it in the same opscript wire format `gen-opscript.mjs` uses, so
// `ddai_tsworld::trace_ts::replay_opscript_with_map_bytes` can replay it unchanged.
//
// Usage: node gen-regression.mjs --case <name> --map <path> --out <path.jsonl>

import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import {
  SimWorld,
  loadMapCollision,
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

const a = parseArgs(process.argv.slice(2));
const caseName = a.case;
const mapPath = a.map;
const outPath = a.out;
if (!caseName || !mapPath || !outPath) {
  console.error("usage: gen-regression.mjs --case <name> --map <path> --out <path.jsonl>");
  process.exit(2);
}

const baseOptions = { respawnDelayTicks: 0, infiniteAmmo: true, svHit: true, allWeapons: true };
// Per-case option overrides, merged with `baseOptions` *before* `SimWorld` is constructed below —
// unlike the base build of this file, `meta()` no longer accepts an `extraOptions` argument that
// only patched the *recorded* metadata line: that was a real bug (caught while cross-checking the
// f5 fixture's own recorded TS output against its intended `respawnDelayTicks: 2` — the emitted
// trace showed `respawnAtTick: null` throughout, i.e. TS's real `respawnDelayTicks` was still the
// module-level default of `0`, because `world` had already been constructed with `baseOptions`
// before the case's `meta({ respawnDelayTicks: 2 })` call ever ran; `meta()`'s merge only affected
// the JSON *label*, not the simulation). Fixed by resolving the case's option overrides first and
// constructing `world` from the final merged options, so the metadata line and the actual `world`
// behavior can never diverge.
const CASE_OPTIONS = {
  "f5-respawn-detaches-from-old-hook-target": { respawnDelayTicks: 2 },
};
const options = { ...baseOptions, ...(CASE_OPTIONS[caseName] ?? {}) };
const loaded = loadMapCollision(mapPath);
const world = new SimWorld(loaded.collision, options);

const lines = [];
function meta() {
  lines.push(
    JSON.stringify({
      traceVersion: 1,
      kind: "opscript",
      generatorVersion: "ts-trace regression 1",
      node: process.version,
      v8: process.versions.v8,
      tsCoreCommit: tsCoreCommit(),
      mapPath,
      mapSha256: sha256File(mapPath),
      seed: "0",
      ops: 0, // filled in after
      options,
    }),
  );
}
function emit(op, extra) {
  lines.push(
    JSON.stringify({
      op,
      ...(extra ?? {}),
      events: (extra?.events ?? []).map(eventJson),
      order: orderIds(world),
      byId: byIdIds(world),
      state: simStateJson(world.saveState()),
    }),
  );
}
function op_addTee(id, pos) {
  world.addTee(id, pos);
  emit("addTee", { id, pos: vec2Json(pos) });
}
// Matches `gen-opscript.mjs`'s own wire format: `step` carries no inputs of its own — held input
// is set via separate `setHeldInput` ops beforehand (`SimWorld.setHeldInput` persists until
// changed again, so one call covers many subsequent `step`s, exactly like a real client holding a
// key down).
function op_step() {
  const ev = world.step();
  emit("step", { events: ev });
}
function op_setHeldInput(id, inp) {
  world.setHeldInput(id, inp);
  emit("setHeldInput", { id, input: inputJson(inp) });
}
function op_kill(id) {
  world.kill(id);
  emit("kill", { id });
}
function op_removeTee(id) {
  world.removeTee(id);
  emit("removeTee", { id });
}
const savedSlots = [];
function op_saveState() {
  savedSlots.push(world.saveState());
  emit("saveState", {});
  return savedSlots.length - 1;
}
function op_saveInto(slot) {
  world.saveState(savedSlots[slot]);
  emit("saveInto", { slot });
}
function op_restoreState(slot) {
  world.restoreState(savedSlots[slot]);
  // Field name must match `gen-opscript.mjs`'s own `"restoreState"` case (`record.restoredIndex`)
  // — `trace_ts.rs`'s `OpLine.restored_index` (camelCase `restoredIndex`) is what the Rust replay
  // actually reads; `{ slot }` alone would silently deserialize to `None` and panic at replay time
  // ("restoreState needs restoredIndex"), which is exactly what happened before this fix.
  emit("restoreState", { restoredIndex: slot });
}
function op_applyTeeState(id, st) {
  world.applyTeeState(id, st);
  // Field name must match `gen-opscript.mjs`'s own `"applyTeeState"` case (`record.teeState`) —
  // `trace_ts.rs`'s `OpLine.tee_state` (camelCase `teeState`) is what the Rust replay reads.
  emit("applyTeeState", { id, teeState: teeStateJson(st) });
}

const hookInput = (tx, ty) => ({
  direction: 0,
  targetX: tx,
  targetY: ty,
  jump: 0,
  fire: 0,
  hook: 1,
  playerFlags: 0,
  wantedWeapon: 0,
  nextWeapon: 0,
  prevWeapon: 0,
});
const idleInput = () => ({
  direction: 0,
  targetX: 0,
  targetY: -1,
  jump: 0,
  fire: 0,
  hook: 0,
  playerFlags: 0,
  wantedWeapon: 0,
  nextWeapon: 0,
  prevWeapon: 0,
});

switch (caseName) {
  // F2: kill()'s death event must survive into the *next* step()'s returned events, not be
  // discarded.
  case "f2-kill-event-survives": {
    meta();
    op_addTee(1, { x: 160, y: 160 });
    op_setHeldInput(1, idleInput());
    op_step();
    op_kill(1);
    op_step();
    break;
  }

  // F3(a): a hook already `FLYING` with `hookedPlayer` already set to something other than `-1`
  // (an inconsistent state only reachable via `applyTeeState`, which never goes through
  // `setHookedPlayer`) must never grab a new target this tick — TS's own scan condition
  // (`hookedPlayer === -1 || d < bestDistance`, `bestDistance` starting at `0`) can never accept
  // any real (non-negative) distance once `hookedPlayer` is already non-`-1`.
  case "f3-flying-with-hooked-player-never-grabs": {
    meta();
    op_addTee(1, { x: 400, y: 400 });
    op_addTee(2, { x: 500, y: 400 });
    op_applyTeeState(1, {
      id: 1,
      alive: true,
      pos: { x: 400, y: 400 },
      vel: { x: 0, y: 0 },
      hookState: 4, // HOOK_FLYING
      hookPos: { x: 442, y: 400 },
      hookDir: { x: 1, y: 0 },
      hookedPlayer: 2, // already set, inconsistent with FLYING — TS never clears this via applyTeeState
      jumped: 0,
      jumpsLeft: 2,
      direction: 0,
      angle: 0,
      activeWeapon: 1,
      frozen: false,
      freezeTicksLeft: 0,
      attackTick: 0,
    });
    op_setHeldInput(1, hookInput(1, 0));
    op_setHeldInput(2, idleInput());
    op_step();
    break;
  }

  // F4: a duplicate `addTee` (same id, still live) leaves the old record as an orphan that keeps
  // ticking (falling under gravity, etc.) as a *distinct* object — `getTee`/`saveState` only ever
  // see the newest one, but the orphan still occupies a slot in `order`/`byId` and participates
  // in every other tee's `allCores()`-based loops.
  case "f4-duplicate-addtee-orphan-still-ticks": {
    meta();
    op_addTee(1, { x: 400, y: 300 });
    op_addTee(1, { x: 600, y: 300 }); // duplicate id — orphans the first record
    op_addTee(2, { x: 300, y: 300 });
    op_setHeldInput(1, idleInput());
    op_setHeldInput(2, idleInput());
    for (let i = 0; i < 20; i++) op_step();
    break;
  }

  // F5: a tee that dies while hooking someone, then respawns, must detach from that hook target
  // (`core.reset()` calls `setHookedPlayer(-1)` as its first action, `characterCore.ts:100,110`)
  // — the target's `attachedPlayers` must end up empty again, not keep a stale entry.
  case "f5-respawn-detaches-from-old-hook-target": {
    meta(); // respawnDelayTicks: 2 comes from CASE_OPTIONS above, already baked into `world`.
    op_addTee(1, { x: 160, y: 160 });
    op_addTee(2, { x: 260, y: 160 });
    op_setHeldInput(1, hookInput(100, 0));
    op_setHeldInput(2, idleInput());
    // Drive tee1's hook onto tee2 for a few ticks (holding hook, aimed at tee2).
    for (let i = 0; i < 25; i++) op_step();
    op_kill(1);
    // respawnDelayTicks=2: a couple of steps later tee1 respawns and must detach from tee2.
    for (let i = 0; i < 4; i++) op_step();
    break;
  }

  // F7: `saveState(into)` never clears `into.tees` first — it updates existing entries in place
  // and never removes a stale one for an id no longer live. `removeTee` then `addTee` with the
  // same id, saved into an *existing* `into`, then restored, must reproduce that exact quirk.
  case "f7-save-into-keeps-stale-entries": {
    meta();
    op_addTee(1, { x: 160, y: 160 });
    op_addTee(2, { x: 260, y: 160 });
    op_setHeldInput(1, idleInput());
    op_setHeldInput(2, idleInput());
    for (let i = 0; i < 5; i++) op_step();
    const slot0 = op_saveState(); // captures both 1 and 2
    op_removeTee(2);
    op_step();
    op_saveInto(slot0); // update slot0 in place: 1 updated, 2 (no longer live) left stale
    op_addTee(2, { x: 300, y: 300 }); // a *new* tee 2
    op_setHeldInput(2, idleInput());
    for (let i = 0; i < 3; i++) op_step();
    op_restoreState(slot0); // applies the *stale* tee-2 snapshot onto the new tee 2
    break;
  }

  default:
    console.error(`unknown case ${caseName}`);
    process.exit(2);
}

// Patch the metadata line's `ops` count now that we know it.
const metaObj = JSON.parse(lines[0]);
metaObj.ops = lines.length - 1;
lines[0] = JSON.stringify(metaObj);

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, lines.join("\n") + "\n");
console.error(`wrote ${lines.length - 1} ops to ${outPath}`);
