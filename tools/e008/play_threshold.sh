#!/usr/bin/env bash
# F9, in-play hook threshold: matches the hook threshold on the in-play START rate (tools/e008/play_threshold.py, training
# arenas only), then plays the matrix of tools/e008/eval_run.sh with the matched bundle (`arena-play`), next to the
# offline-calibrated bundle (`arena`) and the same weights at 0.5 (`arena-half`). Resumable.
#   DDNET_AI=<binary> tools/e008/play_threshold.sh <run-name> [threads]
set -euo pipefail
name=$1; threads=${2:-2}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
bin=${DDNET_AI:-./target/release/ddnet-ai}
out=~/aiddnet/data/runs/E-008/eval/$name
case $name in *fly*) kind=fly ;; *mlpw*) kind=mlp ;; *gruw*) kind=gru ;; *) exit 1 ;; esac
[ -f "$out/play-threshold.json" ] || DDNET_AI=$bin python3 tools/e008/play_threshold.py "$name" --threads "$threads"
[ -f "$out/arena-play/summary.json" ] || $bin arena run --config configs/arena/e008-eval.toml --brain "$kind:$out/selected-play.bundle" \
  --out "$out/arena-play" --threads "$threads" > "$out/arena-play.stdout" 2> "$out/arena-play.log"
echo "$name play-matched"
