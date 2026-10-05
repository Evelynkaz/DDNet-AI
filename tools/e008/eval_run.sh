#!/usr/bin/env bash
# E-008 evaluation of one finished run: offline per-head metrics (hook by own-hook state, calibrated and not), the
# in-play hook start/release rates, the arena matrix (credited win rate first, D-059) and T1-T18, all on the run's
# SELECTED checkpoint. Resumable: a step whose output exists is skipped.
#   tools/e008/eval_run.sh <run-name> [threads]
set -euo pipefail
name=$1
threads=${2:-2}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
bin=${DDNET_AI:-./target/release/ddnet-ai}
run=~/aiddnet/data/runs/E-008/$name
out=~/aiddnet/data/runs/E-008/eval/$name
mkdir -p "$out"
case $name in *fly*) kind=fly ;; *mlpw*) kind=mlp ;; *gruw*) kind=gru ;; *) echo "unknown model kind in $name"; exit 1 ;; esac
bundle=$run/checkpoints/selected.bundle
[ -f "$bundle" ] || { echo "no selected bundle for $name"; exit 1; }
cfg=configs/train/$name.toml
noh=$out/config-noh.toml
python3 tools/e008/strip_human.py "$cfg" "$noh"
export RAYON_NUM_THREADS=$threads

# `to FILE CMD...`: runs CMD with its stdout going to FILE unless FILE is already there and not empty; the output is
# written to FILE.part and renamed only when CMD succeeded, so a killed evaluation leaves nothing that a resume would skip.
to() { local f=$1; shift; [ -s "$f" ] || { "$@" > "$f.part" && mv "$f.part" "$f"; }; }

to "$out/offline.json" $bin train eval --config "$noh" --bundle "$bundle" 2> "$out/offline.log"
to "$out/offline-calibrated.json" $bin train eval --config "$noh" --bundle "$bundle" --calibrate \
  --write-calibrated "$out/selected-calibrated.bundle" 2> "$out/offline-calibrated.log"
to "$out/hook-play.json" $bin train hook-play --actor "$kind:$bundle" --arenas clb-left,pit --games 100 \
  --threads "$threads" 2> "$out/hook-play.log"
[ -f "$out/arena/summary.json" ] || $bin arena run --config configs/arena/e008-eval.toml --brain "$kind:$bundle" \
  --out "$out/arena" --threads "$threads" > "$out/arena.stdout" 2> "$out/arena.log"
[ -f "$out/scenarios/scenarios.json" ] || $bin arena scenarios --brain "$kind:$bundle" --trials 100 \
  --out "$out/scenarios" > "$out/scenarios.stdout" 2> "$out/scenarios.log"
# F9 A/B: the shipped bundle carries thresholds calibrated at the end of every phase (rate-matched on the even
# validation windows); the same weights at 0.5 are the control. In-play hook rates and the arena matrix for both.
half=$out/selected-half.bundle
[ -f "$half" ] || $bin train set-thresholds --kind "$kind" --bundle "$bundle" --out "$half" --jump 0.5 --hook 0.5 --fire 0.5 2> "$out/set-half.log"
to "$out/hook-play-half.json" $bin train hook-play --actor "$kind:$half" --arenas clb-left,pit --games 100 \
  --threads "$threads" 2> "$out/hook-play-half.log"
[ -f "$out/arena-half/summary.json" ] || $bin arena run --config configs/arena/e008-eval.toml --brain "$kind:$half" \
  --out "$out/arena-half" --threads "$threads" > "$out/arena-half.stdout" 2> "$out/arena-half.log"
echo "$name evaluated"
