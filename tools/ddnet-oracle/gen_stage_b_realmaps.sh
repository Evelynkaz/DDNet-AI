#!/usr/bin/env bash
# Stage-B regeneration of the real-map traces that matter for turrets (task 1.6 stage B,
# docs/formats.md section 12.8): the two corpus maps with `CGun` turrets, BlmapChill (13) and
# blmapV5_ddpp (114), re-run with the trace-b v3 harness (which dumps `CPlasma` as entity kind 7
# and the dragged client of every dragger beam) over the SAME seeds, character counts and cfg/team
# variations as bulk_run_server.sh used for v1, so a v2-stageb trace follows the same input
# stream as its v1 sibling until the dump contents differ.
#
# Every trace is produced twice and the two files must be byte-identical (determinism check); a
# SHA256SUMS.realmaps with the digests is written next to them (gen_stage_b.py rewrites the merged
# SHA256SUMS over every .trb/.scn/.rawmap in the directory).
#
# Usage: tools/ddnet-oracle/gen_stage_b_realmaps.sh OUT_DIR [SEEDS_PER_MAP]
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="$(mkdir -p "${1:?usage: gen_stage_b_realmaps.sh OUT_DIR [SEEDS_PER_MAP]}" && cd "$1" && pwd)"
SEEDS_PER_MAP="${2:-40}"
ORACLE_B_BIN="${ORACLE_B_BIN:-$HOME/aiddnet/build/oracle-b/build/ddai_oracle_server}"
TICKS=3000
MAPS_DIR="$HOME/aiddnet/data/research/physics-scratch/maps"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

[[ -x "$ORACLE_B_BIN" ]] || { echo "oracle not built: $ORACLE_B_BIN" >&2; exit 1; }

printf 'sv_solo_server 1\n' >"$WORK_DIR/solo.cfg"
printf 'sv_hit 0\n' >"$WORK_DIR/nohit.cfg"

: >"$OUT_DIR/SHA256SUMS.realmaps"
for map_name in BlmapChill blmapV5_ddpp; do
  map_path="$MAPS_DIR/$map_name.map"
  [[ -f "$map_path" ]] || { echo "missing $map_path" >&2; exit 1; }
  for i in $(seq 1 "$SEEDS_PER_MAP"); do
    seed=$((10000 + i))
    chars=$((2 + (i % 3)))
    extra_args=()
    if (( i % 10 == 0 )); then
      extra_args+=(--cfg "$WORK_DIR/solo.cfg")
    elif (( i % 10 == 5 )); then
      extra_args+=(--cfg "$WORK_DIR/nohit.cfg")
    fi
    if (( i % 7 == 0 )); then
      for c in $(seq 0 $((chars - 1))); do
        extra_args+=(--team "$c=3")
      done
    fi
    name="realmap_${map_name}_seed${seed}"
    sums=()
    for attempt in 1 2; do
      rm -rf "$WORK_DIR/storage"
      "$ORACLE_B_BIN" --storage-dir "$WORK_DIR/storage" --real-map "$map_path" \
        --seed "$seed" --ticks "$TICKS" --chars "$chars" \
        --emit-scenario-v3 "$OUT_DIR/$name.scn" --out "$OUT_DIR/$name.trb" \
        "${extra_args[@]}" >/dev/null 2>"$WORK_DIR/stderr.log" || { cat "$WORK_DIR/stderr.log" >&2; exit 1; }
      sums+=("$(sha256sum "$OUT_DIR/$name.trb" | cut -d' ' -f1)")
    done
    [[ "${sums[0]}" == "${sums[1]}" ]] || { echo "NON-DETERMINISTIC: $name ${sums[*]}" >&2; exit 1; }
    echo "${sums[0]}  $name.trb" >>"$OUT_DIR/SHA256SUMS.realmaps"
    echo "[gen_stage_b_realmaps] $name ok ${sums[0]:0:12}" >&2
  done
done

# Byte-identical rawmaps (one per map) become hard links, like bulk_run_server.sh does.
python3 - "$OUT_DIR" <<'PYEOF'
import glob, hashlib, os, sys, collections
groups = collections.defaultdict(list)
for path in glob.glob(os.path.join(sys.argv[1], "realmap_*.rawmap")):
    groups[hashlib.sha256(open(path, "rb").read()).hexdigest()].append(path)
for paths in groups.values():
    for p in paths[1:]:
        os.remove(p)
        os.link(paths[0], p)
PYEOF
