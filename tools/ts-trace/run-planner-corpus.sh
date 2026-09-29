#!/usr/bin/env bash
# Task 3.2 acceptance criterion 3: generates the full planner-parity dump corpus (>= 10k
# decisions across >= 3 maps, every preset, every in-scope opponent model) via
# gen-planner-dump.mjs. Output goes to $OUT_DIR (default ~/aiddnet/data/traces/planner, not
# committed -- CLAUDE.md).
#
# Review round 1, F4: a second sweep with `--scenario full --danger-bias` adds cases that exercise
# `setTravelGoal`/`setThirdTees`/`setFrozenBystanders`/`setSpareBystanders`/`setBand`/
# `setFreezeMemory`/`setOverrides` together plus 6 tees in the decide-time sim (`bot.ts:4748-4779`'s
# own call pattern), and spawns are biased toward freeze-adjacent tiles so `shielded=true` actually
# shows up at a realistic rate (baseline corpora had it at ~0.1%).
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-$HOME/aiddnet/data/traces/planner}"
CASES_PER_COMBO="${CASES_PER_COMBO:-220}"
# Review round 2, F14: raised from 150 -- more full-scenario volume means more chances of hitting
# an escape rollout genuinely marginal enough for sabotage C (ESCAPE_TICKS 36->35) to flip.
FULL_CASES_PER_COMBO="${FULL_CASES_PER_COMBO:-400}"
SHIELD_STRESS_CASES_PER_COMBO="${SHIELD_STRESS_CASES_PER_COMBO:-400}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

DATA_MAPS="$HOME/aiddnet/data/maps"
declare -A MAPS=(
  [arena]="synthetic:arena"
  [freeze]="synthetic:freeze"
  [clb]="$DATA_MAPS/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map"
  [blmapchill]="$DATA_MAPS/cache/BlmapChill_c902b2da07291266ab201054e6b6c28abd31e10b099fa5b1066b2f5a88f98240.map"
)
PRESETS=(normal low strong wb)
OPPONENTS=(hold react mix)

seed=1
for map_name in "${!MAPS[@]}"; do
  for preset in "${PRESETS[@]}"; do
    for opponent in "${OPPONENTS[@]}"; do
      out="$OUT_DIR/${map_name}_${preset}_${opponent}.jsonl"
      echo "== $map_name $preset $opponent (seed $seed) baseline ==" >&2
      node gen-planner-dump.mjs \
        --map "${MAPS[$map_name]}" \
        --seed "$seed" \
        --cases "$CASES_PER_COMBO" \
        --preset "$preset" \
        --opponent "$opponent" \
        --scenario baseline \
        --out "$out"
      seed=$((seed + 1))
    done
  done
done

FULL_PRESETS=(normal wb)
FULL_OPPONENTS=(hold react)
for map_name in "${!MAPS[@]}"; do
  for preset in "${FULL_PRESETS[@]}"; do
    for opponent in "${FULL_OPPONENTS[@]}"; do
      out="$OUT_DIR/${map_name}_${preset}_${opponent}_full.jsonl"
      echo "== $map_name $preset $opponent (seed $seed) full ==" >&2
      node gen-planner-dump.mjs \
        --map "${MAPS[$map_name]}" \
        --seed "$seed" \
        --cases "$FULL_CASES_PER_COMBO" \
        --preset "$preset" \
        --opponent "$opponent" \
        --scenario full \
        --danger-bias 0.6 \
        --out "$out"
      seed=$((seed + 1))
    done
  done
done


# Review round 2, F14: dedicated shield-stress sweep -- Copy Love Box only, E-000 left hall
# (`--hall`), `self` teleported to a freeze-adjacent tile on every case (`--force-freeze-edge`),
# so `shielded=true`/sabotage C have the best realistic odds this teacher-forced technique gets
# (see the crate README's D-041 section for why this still doesn't reach the requested >= 10%).
CLB_PATH="$DATA_MAPS/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map"
SHIELD_PRESETS=(normal wb strong)
SHIELD_OPPONENTS=(hold react)
for preset in "${SHIELD_PRESETS[@]}"; do
  for opponent in "${SHIELD_OPPONENTS[@]}"; do
    out="$OUT_DIR/clb_${preset}_${opponent}_shieldstress.jsonl"
    echo "== clb $preset $opponent (seed $seed) shield-stress ==" >&2
    node gen-planner-dump.mjs \
      --map "$CLB_PATH" \
      --seed "$seed" \
      --cases "$SHIELD_STRESS_CASES_PER_COMBO" \
      --preset "$preset" \
      --opponent "$opponent" \
      --scenario full \
      --danger-bias 0.9 \
      --hall 1 \
      --force-freeze-edge 1 \
      --out "$out"
    seed=$((seed + 1))
  done
done

echo "done: $(cat "$OUT_DIR"/*.jsonl | grep -c '"kind":"case"') total cases in $OUT_DIR" >&2
