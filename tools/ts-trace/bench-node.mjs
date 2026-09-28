#!/usr/bin/env node
// Task 1.9 acceptance criterion 7: times the real TS `SimWorld.step()` on Copy Love Box with 2
// tees, for comparison with `crates/ddai-tsworld/benches/step.rs`'s criterion numbers (see this
// crate's README, "Производительность", for the side-by-side result).
//
// Usage: node bench-node.mjs <path/to/Copy Love Box_....map>

import { SimWorld, loadMapCollision } from "./lib.mjs";

const mapPath = process.argv[2];
if (!mapPath) {
  console.error("usage: bench-node.mjs <copy-love-box.map>");
  process.exit(2);
}

const loaded = loadMapCollision(mapPath);
const world = new SimWorld(loaded.collision);
world.addTee(1, { x: 200, y: 200 });
world.addTee(2, { x: 260, y: 200 });

function mixedInput(tick, aimAt) {
  const phase = tick % 80;
  return {
    direction: phase < 40 ? 1 : -1,
    targetX: aimAt.x,
    targetY: aimAt.y,
    jump: tick % 80 === 0 ? 1 : 0,
    fire: 0,
    hook: tick % 5 < 2 ? 1 : 0,
    playerFlags: 0,
    wantedWeapon: 0,
    nextWeapon: 0,
    prevWeapon: 0,
  };
}

const WARMUP = 200_000;
const ITERS = 2_000_000;

for (let tick = 0; tick < WARMUP; tick++) {
  world.setInput(1, mixedInput(tick, { x: 260, y: 200 }));
  world.setInput(2, mixedInput(tick + 40, { x: 200, y: 200 }));
  world.step();
}

const t0 = process.hrtime.bigint();
for (let tick = 0; tick < ITERS; tick++) {
  world.setInput(1, mixedInput(tick, { x: 260, y: 200 }));
  world.setInput(2, mixedInput(tick + 40, { x: 200, y: 200 }));
  world.step();
}
const t1 = process.hrtime.bigint();
const nsPerStep = Number(t1 - t0) / ITERS;
console.log(`step(): ${nsPerStep.toFixed(1)} ns/call, ${(1e9 / nsPerStep / 1e6).toFixed(2)} M steps/s (2 tees, ${ITERS} iters)`);

// saveState/restoreState cost.
const SR_ITERS = 500_000;
let saved = world.saveState();
const s0 = process.hrtime.bigint();
for (let i = 0; i < SR_ITERS; i++) {
  saved = world.saveState(saved);
}
const s1 = process.hrtime.bigint();
console.log(`saveState(): ${(Number(s1 - s0) / SR_ITERS).toFixed(1)} ns/call`);

const r0 = process.hrtime.bigint();
for (let i = 0; i < SR_ITERS; i++) {
  world.restoreState(saved);
}
const r1 = process.hrtime.bigint();
console.log(`restoreState(): ${(Number(r1 - r0) / SR_ITERS).toFixed(1)} ns/call`);
