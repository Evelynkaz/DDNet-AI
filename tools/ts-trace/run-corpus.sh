#!/usr/bin/env bash
# Generates and immediately replay-checks the task 1.9 trace-ts v1 bulk corpus, **streaming**:
# each generated file is diffed against `ddai_tsworld::SimWorld` via the `ts-diff` binary right
# after it's written, its coverage counters are folded into this map's running total, and the file
# is deleted before the next one is generated — at no point does the corpus sit on disk as a
# whole (review finding F8's "regenerate at the required scale" instruction, done without the
# multi-gigabyte footprint a "generate everything, then diff everything" pass would leave behind).
# Nothing this script writes is ever committed (`docs/CLAUDE.md` "Никогда не коммитить" — trace
# data lives under `~/aiddnet/data/traces/`, never under the repository).
#
# Run from the repository root: tools/ts-trace/run-corpus.sh [--quick|--real-only]
#   --quick:     a much smaller pass (for a fast local sanity check), still covering every map and
#                option mix at least once.
#   --real-only: skips the synthetic-map episodes and opscripts (already verified separately) and
#                only runs the real-map episodes, at full scale — review finding F10: AC 4a names
#                specific real maps (Copy Love Box, BlmapChill/ChillBlock5) that need their own
#                ≥1000×1000 coverage, not just an aggregate total across every map.
#
# Requires `cargo build -p ddai-tsworld --bin ts-diff --release` to have been run already.

set -euo pipefail
cd "$(dirname "$0")/../.."

TSDIFF="./target/release/ts-diff"
if [ ! -x "$TSDIFF" ]; then
  echo "missing $TSDIFF — run: cargo build -p ddai-tsworld --bin ts-diff --release" >&2
  exit 1
fi

SYNTH_MAPS_DIR="$HOME/aiddnet/data/maps/synthetic"
for m in arena freeze front tele-speedup; do
  if [ ! -f "$SYNTH_MAPS_DIR/$m.map" ]; then
    echo "missing $SYNTH_MAPS_DIR/$m.map — run: cargo run -p ddai-tsworld --example gen_fixture_maps" >&2
    exit 1
  fi
done

REAL_MAPS=(
  "Copy Love Box=$HOME/aiddnet/data/maps/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map"
  "BlmapChill=$HOME/aiddnet/data/ddnet-server/maps/BlmapChill.map"
  "ChillBlock5=$HOME/aiddnet/data/ddnet-server/maps/ChillBlock5.map"
)

QUICK=0
REAL_ONLY=0
case "${1:-}" in
  --quick) QUICK=1 ;;
  --real-only) REAL_ONLY=1 ;;
esac

if [ "$QUICK" = 1 ]; then
  SYNTH_EPISODES_PER_MAP=6
  SYNTH_TICKS=200
  REAL_EPISODES_PER_MAP=2
  REAL_TICKS=150
  OPSCRIPTS=6
  OPS_PER_SCRIPT=2000
else
  SYNTH_EPISODES_PER_MAP="${SYNTH_EPISODES_PER_MAP:-300}"
  SYNTH_TICKS="${SYNTH_TICKS:-1000}"
  # Review finding F10: AC 4a names Copy Love Box and a real block map specifically — bumped from
  # 40×500 to the same ≥1000×1000 scale the synthetic maps get, per-map (not just in aggregate).
  REAL_EPISODES_PER_MAP="${REAL_EPISODES_PER_MAP:-1000}"
  REAL_TICKS="${REAL_TICKS:-1000}"
  OPSCRIPTS="${OPSCRIPTS:-30}"
  OPS_PER_SCRIPT="${OPS_PER_SCRIPT:-10000}"
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/ts-corpus.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

# --- production-realistic option mixes (review finding F8) --------------------------------------
# Index 0 is this tool's original always-on default (kept so the corpus still covers it); indices
# 1-2 are the two production-realistic combinations the review asked for — a "hard block server"
# mix (no respawn delay, only the default weapon, hits count, ammo is finite) and its
# `noWeakHook: true` variant. `OPT_NAMES` labels each for the coverage report.
OPT_NAMES=("legacy-default" "block-server" "block-server-noweakhook")
opt_flags() {
  case "$1" in
    0) echo "--respawn-delay 150 --infinite-ammo 1 --sv-hit 1 --all-weapons 1 --no-weak-hook 0" ;;
    1) echo "--respawn-delay 0 --infinite-ammo 0 --sv-hit 0 --all-weapons 0 --no-weak-hook 0" ;;
    2) echo "--respawn-delay 0 --infinite-ammo 0 --sv-hit 0 --all-weapons 0 --no-weak-hook 1" ;;
  esac
}

# --- running totals (grand total, folded from every per-map subtotal) ---------------------------
declare -A TOTAL=(
  [files]=0 [mismatch_files]=0 [steps]=0 [freeze]=0 [hfire]=0 [hhit]=0 [hook]=0 [death]=0
  [respawn]=0 [maxproj]=0 [maxlaser]=0
)

declare -A MAPTOTAL_files MAPTOTAL_mismatch MAPTOTAL_steps MAPTOTAL_freeze MAPTOTAL_hfire \
  MAPTOTAL_hhit MAPTOTAL_hook MAPTOTAL_death MAPTOTAL_respawn MAPTOTAL_maxproj MAPTOTAL_maxlaser

init_map_totals() {
  local m="$1"
  MAPTOTAL_files[$m]=0; MAPTOTAL_mismatch[$m]=0; MAPTOTAL_steps[$m]=0; MAPTOTAL_freeze[$m]=0
  MAPTOTAL_hfire[$m]=0; MAPTOTAL_hhit[$m]=0; MAPTOTAL_hook[$m]=0; MAPTOTAL_death[$m]=0
  MAPTOTAL_respawn[$m]=0; MAPTOTAL_maxproj[$m]=0; MAPTOTAL_maxlaser[$m]=0
}

# Runs `ts-diff --corpus` on a directory holding exactly one just-generated file, folds its
# printed coverage summary into both the per-map and grand totals, and echoes 1 line ("OK" or
# "MISMATCH <path>") so the caller can decide whether to abort.
diff_one_streaming() {
  local file="$1" map="$2"
  local batch="$WORKDIR/batch"
  rm -rf "$batch"
  mkdir -p "$batch"
  mv "$file" "$batch/"
  local out
  out="$("$TSDIFF" --corpus "$batch" 2>&1)" || true
  rm -rf "$batch"

  local steps mism freeze hfire hhit hook death respawn maxproj maxlaser
  steps=$(echo "$out" | grep -oP 'total steps.*: \K[0-9]+')
  mism=$(echo "$out" | grep -oP 'files with >=1 mismatch: \K[0-9]+')
  freeze=$(echo "$out" | grep -oP 'freeze events:\s+\K[0-9]+')
  hfire=$(echo "$out" | grep -oP 'hammer fire events:\s+\K[0-9]+')
  hhit=$(echo "$out" | grep -oP 'hammer hit events:\s+\K[0-9]+')
  hook=$(echo "$out" | grep -oP 'hook-grabbed ticks:\s+\K[0-9]+')
  death=$(echo "$out" | grep -oP 'death events:\s+\K[0-9]+')
  respawn=$(echo "$out" | grep -oP 'respawns observed:\s+\K[0-9]+')
  maxproj=$(echo "$out" | grep -oP 'max projectiles live:\K[0-9]+')
  maxlaser=$(echo "$out" | grep -oP 'max lasers live:\s*\K[0-9]+')

  TOTAL[files]=$((TOTAL[files] + 1))
  TOTAL[mismatch_files]=$((TOTAL[mismatch_files] + mism))
  TOTAL[steps]=$((TOTAL[steps] + steps))
  TOTAL[freeze]=$((TOTAL[freeze] + freeze))
  TOTAL[hfire]=$((TOTAL[hfire] + hfire))
  TOTAL[hhit]=$((TOTAL[hhit] + hhit))
  TOTAL[hook]=$((TOTAL[hook] + hook))
  TOTAL[death]=$((TOTAL[death] + death))
  TOTAL[respawn]=$((TOTAL[respawn] + respawn))
  [ "$maxproj" -gt "${TOTAL[maxproj]}" ] && TOTAL[maxproj]=$maxproj
  [ "$maxlaser" -gt "${TOTAL[maxlaser]}" ] && TOTAL[maxlaser]=$maxlaser

  MAPTOTAL_files[$map]=$((MAPTOTAL_files[$map] + 1))
  MAPTOTAL_mismatch[$map]=$((MAPTOTAL_mismatch[$map] + mism))
  MAPTOTAL_steps[$map]=$((MAPTOTAL_steps[$map] + steps))
  MAPTOTAL_freeze[$map]=$((MAPTOTAL_freeze[$map] + freeze))
  MAPTOTAL_hfire[$map]=$((MAPTOTAL_hfire[$map] + hfire))
  MAPTOTAL_hhit[$map]=$((MAPTOTAL_hhit[$map] + hhit))
  MAPTOTAL_hook[$map]=$((MAPTOTAL_hook[$map] + hook))
  MAPTOTAL_death[$map]=$((MAPTOTAL_death[$map] + death))
  MAPTOTAL_respawn[$map]=$((MAPTOTAL_respawn[$map] + respawn))
  [ "$maxproj" -gt "${MAPTOTAL_maxproj[$map]:-0}" ] && MAPTOTAL_maxproj[$map]=$maxproj
  [ "$maxlaser" -gt "${MAPTOTAL_maxlaser[$map]:-0}" ] && MAPTOTAL_maxlaser[$map]=$maxlaser

  if [ "$mism" != "0" ]; then
    echo "MISMATCH in file generated for map=$map:"
    echo "$out"
    return 1
  fi
  return 0
}

print_map_report() {
  local m="$1"
  echo "  [$m] files=${MAPTOTAL_files[$m]} steps=${MAPTOTAL_steps[$m]} mismatches=${MAPTOTAL_mismatch[$m]:-0} | freeze=${MAPTOTAL_freeze[$m]} hammerFire=${MAPTOTAL_hfire[$m]} hammerHit=${MAPTOTAL_hhit[$m]} hookGrabbed=${MAPTOTAL_hook[$m]} deaths=${MAPTOTAL_death[$m]} respawns=${MAPTOTAL_respawn[$m]} maxProj=${MAPTOTAL_maxproj[$m]:-0} maxLaser=${MAPTOTAL_maxlaser[$m]:-0}"
}

if [ "$REAL_ONLY" != 1 ]; then
  echo "== synthetic map episodes (production option mix included) =="
  for m in arena freeze front tele-speedup; do
    init_map_totals "$m"
    for ((i = 0; i < SYNTH_EPISODES_PER_MAP; i++)); do
      seed=$((1000 + i))
      tees=$(((i % 4) + 2)) # 2..5 tees
      optidx=$((i % 3))
      out="$WORKDIR/ep_${m}_${seed}.jsonl"
      node tools/ts-trace/gen-episode.mjs --map "$SYNTH_MAPS_DIR/$m.map" --seed "$seed" --ticks "$SYNTH_TICKS" --tees "$tees" $(opt_flags "$optidx") --out "$out"
      diff_one_streaming "$out" "$m"
    done
    print_map_report "$m"
  done
fi

echo "== real map episodes =="
for entry in "${REAL_MAPS[@]}"; do
  name="${entry%%=*}"
  path="${entry#*=}"
  mapkey="real-$name"
  init_map_totals "$mapkey"
  for ((i = 0; i < REAL_EPISODES_PER_MAP; i++)); do
    seed=$((2000 + i))
    tees=$(((i % 4) + 2))
    optidx=$((i % 3))
    out="$WORKDIR/ep_${mapkey// /_}_${seed}.jsonl"
    node tools/ts-trace/gen-episode.mjs --map "$path" --seed "$seed" --ticks "$REAL_TICKS" --tees "$tees" $(opt_flags "$optidx") --out "$out"
    diff_one_streaming "$out" "$mapkey"
  done
  print_map_report "$mapkey"
done

if [ "$REAL_ONLY" != 1 ]; then
  echo "== opscripts (events/order/byId comparisons, production option mix, reset/saveInto/dup-ids) =="
  init_map_totals "opscripts"
  for ((i = 0; i < OPSCRIPTS; i++)); do
    seed=$((3000 + i))
    case $((i % 4)) in
      0) m="arena" ;;
      1) m="freeze" ;;
      2) m="front" ;;
      3) m="tele-speedup" ;;
    esac
    optidx=$((i % 3))
    tees=$(((i % 4) + 2))
    out="$WORKDIR/ops_${seed}_${m}.jsonl"
    node tools/ts-trace/gen-opscript.mjs --map "$SYNTH_MAPS_DIR/$m.map" --seed "$seed" --ops "$OPS_PER_SCRIPT" --tees "$tees" $(opt_flags "$optidx") --out "$out"
    diff_one_streaming "$out" "opscripts"
  done
  print_map_report "opscripts"
fi

echo
echo "=== GRAND TOTAL ==="
echo "files replayed:        ${TOTAL[files]}"
echo "total steps (ticks/ops): ${TOTAL[steps]}"
echo "files with mismatches:  ${TOTAL[mismatch_files]}"
echo "freeze events:          ${TOTAL[freeze]}"
echo "hammer fire events:     ${TOTAL[hfire]}"
echo "hammer hit events:      ${TOTAL[hhit]}"
echo "hook-grabbed ticks:     ${TOTAL[hook]}"
echo "death events:           ${TOTAL[death]}"
echo "respawns observed:      ${TOTAL[respawn]}"
echo "max projectiles live:   ${TOTAL[maxproj]}"
echo "max lasers live:        ${TOTAL[maxlaser]}"

if [ "${TOTAL[mismatch_files]}" != "0" ]; then
  echo "FAILED: ${TOTAL[mismatch_files]} file(s) had a mismatch" >&2
  exit 1
fi
echo "OK: 0 mismatches across ${TOTAL[files]} streamed files / ${TOTAL[steps]} total steps"
