#!/usr/bin/env node
// Runs the real `Rng` (imported read-only from `src/nn/rng.ts`, per the task's constraint of
// never modifying or copying the old TS code — Node 24 runs `.ts` directly via type stripping,
// see `docs/research/orig-plan.md` "Можно ли запустить TS детерминированно") over a batch of
// seeds and a fixed, interleaved draw pattern, and writes the resulting f64 sequence for the Rust
// side to compare bit-for-bit.
//
// Usage: node rng_probe.mjs <seeds.u32> <draws-per-seed> <out.f64>
//
// `seeds.u32`: little-endian u32 seeds, one per `Rng`.
// Draw pattern per seed (index i in 0..draws-per-seed, matching
// crates/ddai-jsmath/tests/oracle.rs's `rng_op_for_draw`): i%5 in {0,1} -> nextU32 (widened to
// f64, exactly, since it's an integer < 2^53), {2,3} -> nextFloat, {4} -> nextGaussian (chosen so
// consecutive nextGaussian calls are 5 draws apart, alternating between computing a fresh
// Box-Muller pair and returning the carried spare, exercising both branches).
// `out.f64`: little-endian f64, `seeds.length * draws-per-seed` values, row-major per seed.

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const { Rng } = await import(join(here, "..", "..", "src", "nn", "rng.ts"));

function main() {
  const [seedsPath, drawsArg, outPath] = process.argv.slice(2);
  if (!seedsPath || !drawsArg || !outPath) {
    console.error("usage: node rng_probe.mjs <seeds.u32> <draws-per-seed> <out.f64>");
    process.exit(1);
  }
  const draws = Number.parseInt(drawsArg, 10);

  const seedsBuf = readFileSync(seedsPath);
  const seeds = new Uint32Array(seedsBuf.buffer, seedsBuf.byteOffset, seedsBuf.byteLength / 4);

  const out = new Float64Array(seeds.length * draws);
  let k = 0;
  for (let s = 0; s < seeds.length; s++) {
    const rng = new Rng(seeds[s]);
    for (let i = 0; i < draws; i++) {
      const op = i % 5;
      let v;
      if (op === 0 || op === 1) v = rng.nextU32();
      else if (op === 2 || op === 3) v = rng.nextFloat();
      else v = rng.nextGaussian();
      out[k++] = v;
    }
  }

  writeFileSync(outPath, Buffer.from(out.buffer, out.byteOffset, out.byteLength));
  console.error(
    `rng_probe.mjs: node ${process.version}, v8 ${process.versions.v8}: ${seeds.length} seeds x ${draws} draws`,
  );
}

main();
