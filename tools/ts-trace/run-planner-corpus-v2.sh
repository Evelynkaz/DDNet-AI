#!/usr/bin/env bash
# Task 3.8: the planner-parity corpus of upstream af49dfb (release 2026-10-02) -- teacher-forced decisions (gen-planner-dump.mjs) on both
# Copy Love Box versions (the canonical 387x250 and Swarfey's 468x255), BlmapChill and ChillBlock5, free-running games
# (gen-planner-freerun.mjs) on the same maps plus a synthetic arena, and the per-component dumps (gen-v2-component-dump.mjs). Needs the
# reference extracted (`tools/ts-trace/README.md`, "Два эталона"): DDAI_TS_REF=~/aiddnet/data/scratch/ts-af49dfb. Output (not committed):
# $OUT_DIR (default ~/aiddnet/data/traces/planner-af49dfb) with tf/, fr/, components/. Runs 3 generators at a time (JOBS).
#
# Cases (defaults): baseline 4 maps x 5 presets x 3 opponents x 120 = 7200; full (goal/thirds/bystanders/band/memory/overrides by role,
# frozen enemies, enemies hooking us) 4 maps x 4 presets x 2 opponents x 250 = 8000; guard stress (hall, frozen enemy) 2 maps x 3
# guard presets x 2 opponents x 300 = 3600; own hook in flight 3 maps x 2 presets x 150 = 900; free-run games 4 maps x 5 presets x 2 + 4 synthetic = 44 games x 600 ticks.
set -euo pipefail
cd "$(dirname "$0")"
: "${DDAI_TS_REF:=$HOME/aiddnet/data/scratch/ts-af49dfb}"
export DDAI_TS_REF
[ -f "$DDAI_TS_REF/src/plan/planner.ts" ] || { echo "no af49dfb reference in $DDAI_TS_REF (see tools/ts-trace/README.md)" >&2; exit 2; }
OUT_DIR="${OUT_DIR:-$HOME/aiddnet/data/traces/planner-af49dfb}"
JOBS="${JOBS:-3}"
BASE_CASES="${BASE_CASES:-120}"
FULL_CASES="${FULL_CASES:-250}"
GUARD_CASES="${GUARD_CASES:-300}"
FR_TICKS="${FR_TICKS:-600}"
rm -rf "$OUT_DIR/tf" "$OUT_DIR/fr"
mkdir -p "$OUT_DIR/tf" "$OUT_DIR/fr" "$OUT_DIR/components"

M="$HOME/aiddnet/data/maps"
declare -A MAPS=(
  [clb]="$M/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map"
  [clbswarfey]="$M/cache/Copy Love Box_23f188bba2a0a98f6005767481912f2b24dd55ea4980735fe31f81a090af94d7.map"
  [blmapchill]="$M/cache/BlmapChill_c902b2da07291266ab201054e6b6c28abd31e10b099fa5b1066b2f5a88f98240.map"
  [chillblock5]="$M/cache/ChillBlock5_44f8343a686a05afc8f61e7ca90d95bf8fc97f4074b0f3f6d5f8cff60d1cb378.map"
)
MAP_NAMES=(clb clbswarfey blmapchill chillblock5)
JOBFILE="$(mktemp)"
trap 'rm -f "$JOBFILE"' EXIT
q() { printf '%q' "$1"; }
seed=100
job() { echo "node $*" >>"$JOBFILE"; }

for m in "${MAP_NAMES[@]}"; do
  for preset in normal live low strong wb; do
    for opp in hold react mix; do
      seed=$((seed + 1))
      job gen-planner-dump.mjs --map "$(q "${MAPS[$m]}")" --seed $seed --cases "$BASE_CASES" --preset $preset --opponent $opp --scenario baseline \
        --out "$OUT_DIR/tf/${m}_${preset}_${opp}.jsonl"
    done
  done
done
for m in "${MAP_NAMES[@]}"; do
  for preset in normal live wblive guardl; do
    for opp in hold react; do
      seed=$((seed + 1))
      job gen-planner-dump.mjs --map "$(q "${MAPS[$m]}")" --seed $seed --cases "$FULL_CASES" --preset $preset --opponent $opp --scenario full \
        --danger-bias 0.6 --frozen-enemy 0.3 --hook-us 0.25 --out "$OUT_DIR/tf/${m}_${preset}_${opp}_full.jsonl"
    done
  done
done
for m in clb clbswarfey; do
  for preset in guardl guardr guardchain; do
    for opp in hold react; do
      seed=$((seed + 1))
      job gen-planner-dump.mjs --map "$(q "${MAPS[$m]}")" --seed $seed --cases "$GUARD_CASES" --preset $preset --opponent $opp --scenario full \
        --danger-bias 0.6 --hall 1 --frozen-enemy 0.6 --hook-us 0.15 --out "$OUT_DIR/tf/${m}_${preset}_${opp}_guard.jsonl"
    done
  done
done
# Our own hook in flight toward the enemy (`hookKeepFlying`'s branch of polishRope): 3 maps x 2 presets x 150 = 900. Seeds 901.. as dumped on 2026-10-04.
fseed=900
for spec in "clb:clb" "blm:blmapchill" "cb5:chillblock5"; do
  short="${spec%%:*}"; mapname="${spec##*:}"
  for preset in normal live; do
    fseed=$((fseed + 1))
    job gen-planner-dump.mjs --map "$(q "${MAPS[$mapname]}")" --seed $fseed --cases 150 --preset $preset --opponent react --scenario baseline \
      --hook-flying 0.8 --out "$OUT_DIR/tf/${short}_${preset}_react_flying.jsonl"
  done
done
for m in "${MAP_NAMES[@]}"; do
  for preset in normal live guardl guardchain strong; do
    for opp in hold react; do
      seed=$((seed + 1))
      job gen-planner-freerun.mjs --map "$(q "${MAPS[$m]}")" --seed $seed --ticks "$FR_TICKS" --preset $preset --opponent $opp \
        --out "$OUT_DIR/fr/${m}_${preset}_${opp}.jsonl"
    done
  done
done
for n in 1 2 3 4; do
  seed=$((seed + 1))
  job gen-planner-freerun.mjs --map synthetic:arena --seed $seed --ticks "$FR_TICKS" --preset $([ $n -le 2 ] && echo live || echo guardl) --opponent react \
    --out "$OUT_DIR/fr/synth_$n.jsonl"
done
for m in "${MAP_NAMES[@]}"; do
  job gen-v2-component-dump.mjs --map "$(q "${MAPS[$m]}")" --seed 1 --cases 1500 --out "$OUT_DIR/components/$m.jsonl"
done

echo "$(wc -l <"$JOBFILE") generator runs, $JOBS at a time" >&2
xargs -d '\n' -P "$JOBS" -I{} bash -c '{} 2>>"'"$OUT_DIR"'/gen.log"' <"$JOBFILE"
echo "done: $(cat "$OUT_DIR"/tf/*.jsonl | grep -c '"kind":"case"') teacher-forced decisions, $(cat "$OUT_DIR"/fr/*.jsonl | grep -c '"kind":"decision"') free-run decisions in $OUT_DIR" >&2
