// Regenerates harness_spawns.json: the spawn positions the phase-0 TS harness
// (~/aiddnet/data/research/harness, docs/research/orig-run.md) produces for seeds 1..200 on each
// arena, as `[ax, ay, bx, by]` pixel positions. ddai-env's `spawn_tiles` must reproduce them.
//   node tests/fixtures/gen_harness_spawns.mjs > tests/fixtures/harness_spawns.json
import { arenaPit, arenaPlatform, Rng } from "/home/ubuntu/aiddnet/data/research/harness/lib.mjs";
import { mapArena } from "/home/ubuntu/aiddnet/data/research/harness/maps.mjs";
const arenas = { pit: arenaPit(), platform: arenaPlatform(), "clb-left": mapArena("clb"), "clb-right": mapArena("clb-right") };
const out = {};
for (const [name, a] of Object.entries(arenas)) {
  out[name] = { slots: a.stand ? a.stand.length : null, spawns: [] };
  for (let seed = 1; seed <= 200; seed++) {
    const rng = new Rng((seed * 2654435761) >>> 0);
    const [pa, pb] = a.spawn(rng);
    out[name].spawns.push([pa.x, pa.y, pb.x, pb.y]);
  }
}
console.log(JSON.stringify(out));
