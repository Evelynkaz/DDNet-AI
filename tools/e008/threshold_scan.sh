#!/usr/bin/env bash
# In-play sensitivity of the hook threshold (E-005 review F9: "consider matching the hook threshold on the in-play start
# rates"): the same weights at several hook thresholds, 100 games on clb-left and pit with the teacher watching, and the
# 1v1 credited win rate on clb-left. Resumable (existing outputs are skipped).
#   tools/e008/threshold_scan.sh <run-name> [thresholds...]
set -euo pipefail
name=$1; shift
ths=${*:-0.3 0.4 0.5 0.6 0.7}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
bin=${DDNET_AI:-./target/release/ddnet-ai}
case $name in *fly*) kind=fly ;; *mlpw*) kind=mlp ;; *gruw*) kind=gru ;; *) exit 1 ;; esac
src=~/aiddnet/data/runs/E-008/$name/checkpoints/selected.bundle
out=~/aiddnet/data/runs/E-008/eval/$name/threshold-scan
mkdir -p "$out"
export RAYON_NUM_THREADS=${THREADS:-1}
for t in $ths; do
  b=$out/hook-$t.bundle
  [ -f "$b" ] || $bin train set-thresholds --kind $kind --bundle "$src" --out "$b" --hook "$t" 2> /dev/null
  [ -f "$out/hook-play-$t.json" ] || $bin train hook-play --actor "$kind:$b" --arenas clb-left,pit --games 100 --threads ${THREADS:-1} \
    > "$out/hook-play-$t.json" 2> "$out/hook-play-$t.log"
  [ -f "$out/arena-$t/summary.json" ] || $bin arena run --config configs/arena/e008-eval.toml --brain "$kind:$b" --filter "clb-left vs scripted" \
    --out "$out/arena-$t" --threads ${THREADS:-1} > /dev/null 2> "$out/arena-$t.log"
done
echo "$name scanned"
