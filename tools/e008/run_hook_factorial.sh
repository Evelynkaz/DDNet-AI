#!/usr/bin/env bash
# The 2x2x2 hook-study factorial (E-005 review F6), 3 seeds, 1 thread; a seed whose JSON exists is skipped.
#   tools/e008/run_hook_factorial.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
bin=${DDNET_AI:-./target/release/ddnet-ai}
out=~/aiddnet/data/runs/E-008/hook-factorial
mkdir -p "$out"
maps=~/aiddnet/data/maps/cache
for seed in 1 2 3; do
  [ -f "$out/seed-$seed.json" ] && continue
  RAYON_NUM_THREADS=1 $bin train hook-study --factorial --seed $seed --threads 1 \
    --flyg ~/aiddnet/data/connectome/compiled/fly-S-v1.flyg \
    --train-map "$maps/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map" \
    --other-map "$maps/BlmapChill_c902b2da07291266ab201054e6b6c28abd31e10b099fa5b1066b2f5a88f98240.map" \
    --out "$out/seed-$seed.json" > "$out/seed-$seed.log" 2>&1
done
