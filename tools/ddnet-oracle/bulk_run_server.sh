#!/usr/bin/env bash
# Oracle B bulk corpus generator (task 1.5 spec, acceptance criterion 6, revised in round-2
# review fixes F5/F7): >= 500 scenarios over the 7 real maps named in the task spec (Copy Love
# Box, BlmapChill, ChillBlock5, blmapV5_ddpp, Blockdale, BlockField, blmapV3multistarbox) plus
# the 4 synthetic recipes (arena/freeze/front/tele-speedup) via raw2map, writing trace-b v1
# files into ~/aiddnet/data/traces/oracle-b/v1/ (outside the repository, per project
# convention -- CLAUDE.md's "Раскладка папок"). Every real-map scenario is a "block scenario"
# (deliberate hammer hits and hooks aimed at the nearest other character, movement biased
# toward it, spawn biased near freeze edges -- docs/formats.md section 9.3) by construction of
# the harness-side generator, so no separate flag/mode is needed for that half of criterion 6.
#
# Round-2 review changes from the first corpus:
#   - F7 (round 1): a `--cfg`-based variation slice (every 10th seed per real map) exercises
#     sv_solo_server/sv_team/sv_hit so `solo`/`nonzero_team` coverage isn't left entirely to
#     incidental server behavior.
#   - F5: every real-map scenario also gets `--emit-scenario-v3`, writing a companion
#     (rawmap + scenario v3) pair next to its trace so task 1.6 can replay every one of these
#     traces' exact resolved inputs without this harness's generator or the original .map file.
#   - F11 (round 2, BLOCKER): a round-2 pass mistakenly read "shorter episodes" (round 1's
#     coverage feedback) as "shorten the WHOLE trace" and set TICKS=600, dropping the corpus to
#     650 * 600 = 390k ticks (1.16M character-ticks) -- well under the spec's ">= 500 scenarios
#     x 3000 ticks" (>= 1.5M ticks). The actual intent was sub-episodes WITHIN a 3000-tick trace
#     (via kill/respawn, which F12 below now provides in a replay-safe way) -- not a shorter
#     trace. Reverted to TICKS=3000; scenario counts kept the same (70/map, 40/recipe -- still
#     >= 500 total) since memory stayed low even at 3000 ticks (measured: ~150 MiB peak RSS for
#     a 4-character/3000-tick real-map run -- see the build report) and disk headroom is ample
#     (measured corpus size in the build report).
#   - F12 (round 2): kill is now a per-tick INPUT bit (scenario v3, docs/formats.md section 8/
#     9.3/9.6), applied through the real `OnKillNetMessage` path -- not a silent harness action
#     -- so a long freeze no longer has to occupy the rest of a 3000-tick trace uneventfully.
#   - Coverage addition (round 2, orchestrator addition to F7): real-map spawns are now biased
#     (~1/4) toward teleporter-in/switch/tune-zone tiles when a map has any, and ~3/4 of
#     characters get shotgun/grenade/laser at spawn (docs/formats.md section 9.3).
#
# Usage: tools/ddnet-oracle/bulk_run_server.sh [out_dir]
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."

OUT_DIR="${1:-$HOME/aiddnet/data/traces/oracle-b/v1}"
ORACLE_B_BIN="${ORACLE_B_BIN:-$HOME/aiddnet/build/oracle-b/build/ddai_oracle_server}"
DDNET_AI_BIN="$REPO_ROOT/target/release/ddnet-ai"
TICKS=3000
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

log() { printf '[bulk_run_server.sh] %s\n' "$*" >&2; }
die() { printf '[bulk_run_server.sh] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -x "$ORACLE_B_BIN" ]] || die "$ORACLE_B_BIN not found -- run tools/ddnet-oracle/build-server-oracle.sh first"
[[ -x "$DDNET_AI_BIN" ]] || die "$DDNET_AI_BIN not found -- run 'cargo build --release -p ddnet-ai' first"

mkdir -p "$OUT_DIR"

REAL_MAPS_DIR_1="$HOME/aiddnet/data/research/physics-scratch/maps"
REAL_MAPS_DIR_2="$HOME/aiddnet/data/maps/copy-love-box"
declare -a REAL_MAPS=()
[[ -f "$REAL_MAPS_DIR_1/BlmapChill.map" ]] && REAL_MAPS+=("$REAL_MAPS_DIR_1/BlmapChill.map")
[[ -f "$REAL_MAPS_DIR_1/ChillBlock5.map" ]] && REAL_MAPS+=("$REAL_MAPS_DIR_1/ChillBlock5.map")
[[ -f "$REAL_MAPS_DIR_1/blmapV5_ddpp.map" ]] && REAL_MAPS+=("$REAL_MAPS_DIR_1/blmapV5_ddpp.map")
[[ -f "$REAL_MAPS_DIR_1/Blockdale.map" ]] && REAL_MAPS+=("$REAL_MAPS_DIR_1/Blockdale.map")
[[ -f "$REAL_MAPS_DIR_1/BlockField.map" ]] && REAL_MAPS+=("$REAL_MAPS_DIR_1/BlockField.map")
[[ -f "$REAL_MAPS_DIR_1/blmapV3multistarbox.map" ]] && REAL_MAPS+=("$REAL_MAPS_DIR_1/blmapV3multistarbox.map")
# Copy Love Box: pinned by sha256, NOT by lexicographic/first-found filename (F4, round-1
# review): SOURCES.txt in that directory names 6e79ef43... (archived 2020-09-03) as "Primary
# file used by the harness ... All six pass ... wayblockFor(...) except 100590d0 (2021 build,
# same size, kept for reference)" -- a naive `find | sort | head -1` picks 100590d0 (it sorts
# before 51a2.../68f2.../6e79.../c15e.../cd1d... lexicographically), which is exactly the ONE
# file SOURCES.txt says NOT to use. Refuse to proceed on a mismatch rather than silently using
# whatever happens to be on disk.
readonly CLB_SHA256="6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25"
CLB_MAP="$REAL_MAPS_DIR_2/Copy Love Box_${CLB_SHA256}.map"
if [[ -f "$CLB_MAP" ]]; then
  actual_sha256="$(sha256sum "$CLB_MAP" | awk '{print $1}')"
  [[ "$actual_sha256" == "$CLB_SHA256" ]] || die "Copy Love Box file $CLB_MAP has sha256 $actual_sha256, expected $CLB_SHA256 (refusing to use a possibly-wrong/corrupted map)"
  REAL_MAPS+=("$CLB_MAP")
else
  log "WARNING: pinned Copy Love Box file (sha256 $CLB_SHA256) not found at $CLB_MAP -- skipping this map"
fi

[[ "${#REAL_MAPS[@]}" -ge 1 ]] || die "no real maps found under $REAL_MAPS_DIR_1 or $REAL_MAPS_DIR_2"
log "using ${#REAL_MAPS[@]} real maps: ${REAL_MAPS[*]}"

RECIPES=(arena freeze front tele-speedup)
for recipe in "${RECIPES[@]}"; do
  "$DDNET_AI_BIN" trace export-map --recipe "$recipe" --out "$WORK_DIR/$recipe.rawmap" >/dev/null
done

# F7: variation cfg files -- sv_solo_server/sv_hit exercised on a fraction of real-map
# scenarios (every 10th seed) so `solo`/hit-related coverage isn't left purely incidental.
SOLO_CFG="$WORK_DIR/solo.cfg"
printf 'sv_solo_server 1\n' >"$SOLO_CFG"
NOHIT_CFG="$WORK_DIR/nohit.cfg"
printf 'sv_hit 0\n' >"$NOHIT_CFG"

SCENARIOS_PER_REAL_MAP=70
SCENARIOS_PER_RECIPE=40

COUNT=0
TOTAL_BYTES=0
START_TS=$(date +%s)

COV_FIELDS=(character_ticks frozen_ticks deep_frozen_ticks live_frozen_ticks freeze_entries freeze_exits \
  speedup_ticks stopper_ticks tune_zone_ticks teleport_ticks hammer_swings hammer_hits hook_grabs \
  switch_toggles solo_ticks jetpack_ticks endless_jump_ticks super_ticks collision_disabled_ticks \
  hook_hit_disabled_ticks nonzero_team_ticks died respawned)

run_one() {
  local group="$1"
  local out_name="$2"
  shift 2
  local cov_file="$WORK_DIR/cov.json"
  "$ORACLE_B_BIN" "$@" --out "$OUT_DIR/$out_name" --coverage-out "$cov_file" >/dev/null 2>"$WORK_DIR/stderr.log" \
    || { cat "$WORK_DIR/stderr.log" >&2; die "run failed for $out_name"; }
  COUNT=$((COUNT + 1))
  local sz
  sz=$(stat -c%s "$OUT_DIR/$out_name")
  TOTAL_BYTES=$((TOTAL_BYTES + sz))
  python3 - "$group" "$cov_file" "${COV_FIELDS[@]}" <<'PYEOF' >>"$WORK_DIR/cov_totals.tsv"
import json, sys
group = sys.argv[1]
d = json.load(open(sys.argv[2]))
names = sys.argv[3:]
print(group + "\t" + "\t".join(str(d[k]) for k in names))
PYEOF
}

: > "$WORK_DIR/cov_totals.tsv"

for map_path in "${REAL_MAPS[@]}"; do
  map_base="$(basename "$map_path" .map)"
  map_slug="$(echo "$map_base" | tr -c 'A-Za-z0-9_' '_')"
  for i in $(seq 1 "$SCENARIOS_PER_REAL_MAP"); do
    seed=$((10000 + i))
    chars=$((2 + (i % 3)))
    storage_dir="$WORK_DIR/storage"
    rm -rf "$storage_dir"
    extra_args=()
    if (( i % 10 == 0 )); then
      extra_args+=(--cfg "$SOLO_CFG")
    elif (( i % 10 == 5 )); then
      extra_args+=(--cfg "$NOHIT_CFG")
    fi
    if (( i % 7 == 0 )); then
      # sv_team-style variation: force every character in this scenario onto a shared
      # non-zero DDRace team so `nonzero_team_ticks` coverage isn't purely incidental.
      for c in $(seq 0 $((chars - 1))); do
        extra_args+=(--team "$c=3")
      done
    fi
    run_one "realmap_$map_slug" "realmap_${map_slug}_seed${seed}.trb" \
      --storage-dir "$storage_dir" \
      --real-map "$map_path" \
      --seed "$seed" --ticks "$TICKS" --chars "$chars" \
      --emit-scenario-v3 "$OUT_DIR/realmap_${map_slug}_seed${seed}.scn" \
      "${extra_args[@]}"
  done
  log "real map $map_base: $SCENARIOS_PER_REAL_MAP scenarios done (running total: $COUNT)"
done

for recipe in "${RECIPES[@]}"; do
  chars_for_recipe=3
  [[ "$recipe" == "tele-speedup" ]] && chars_for_recipe=4
  for i in $(seq 1 "$SCENARIOS_PER_RECIPE"); do
    seed=$((20000 + i))
    "$DDNET_AI_BIN" trace gen-scenario --recipe "$recipe" --seed "$seed" --ticks "$TICKS" --chars "$chars_for_recipe" \
      --out "$WORK_DIR/scn.scn" >/dev/null
    storage_dir="$WORK_DIR/storage"
    rm -rf "$storage_dir"
    run_one "recipe_$recipe" "recipe_${recipe//-/_}_seed${seed}.trb" \
      --storage-dir "$storage_dir" \
      --rawmap "$WORK_DIR/$recipe.rawmap" \
      --scenario "$WORK_DIR/scn.scn" \
      --seed "$seed"
  done
  log "recipe $recipe: $SCENARIOS_PER_RECIPE scenarios done (running total: $COUNT)"
done

END_TS=$(date +%s)

# Dedupe rawmap files by content (F5's map2raw output is a pure function of the loaded map, so
# every scenario generated on the SAME map produces a byte-identical .rawmap) -- replacing the
# duplicates with hard links keeps every .scn's own `map_ref_string` pointing at a file that
# still exists under the exact same name/path, just backed by shared storage. Cuts a multi-GiB
# corpus down to the handful of genuinely distinct maps' worth of rawmap bytes.
log "deduplicating rawmap files (hard links for byte-identical content)..."
python3 - "$OUT_DIR" <<'PYEOF'
import hashlib, os, sys, glob, collections
out_dir = sys.argv[1]
groups = collections.defaultdict(list)
for path in glob.glob(os.path.join(out_dir, "realmap_*.rawmap")):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    groups[h.hexdigest()].append(path)
saved = 0
for paths in groups.values():
    canonical = paths[0]
    sz = os.path.getsize(canonical)
    for p in paths[1:]:
        os.remove(p)
        os.link(canonical, p)
        saved += sz
print(f"[bulk_run_server.sh] {len(groups)} distinct rawmap(s), reclaimed {saved // 1024 // 1024} MiB via hard links", file=sys.stderr)
PYEOF

log "generated $COUNT trace files, $((TOTAL_BYTES / 1024 / 1024)) MiB total (pre-dedup), in $((END_TS - START_TS))s -> $OUT_DIR"

python3 - "$WORK_DIR/cov_totals.tsv" "$OUT_DIR/coverage_summary.json" "${COV_FIELDS[@]}" <<'PYEOF'
import sys, json, collections
names = sys.argv[3:]
per_group = collections.OrderedDict()
totals = [0]*len(names)
n = 0
with open(sys.argv[1]) as f:
    for line in f:
        parts = line.strip().split("\t")
        group = parts[0]
        vals = [int(x) for x in parts[1:]]
        g = per_group.setdefault(group, [0]*len(names))
        for i, v in enumerate(vals):
            g[i] += v
            totals[i] += v
        n += 1
print(f"[bulk_run_server.sh] per-map/recipe coverage ({n} scenarios total):", file=sys.stderr)
for group, vals in per_group.items():
    print(f"  {group}: " + ", ".join(f"{name}={v}" for name, v in zip(names, vals)), file=sys.stderr)
print("[bulk_run_server.sh] aggregate coverage:", file=sys.stderr)
for name, total in zip(names, totals):
    print(f"  {name}: {total}", file=sys.stderr)
summary = {"scenario_count": n, "aggregate": dict(zip(names, totals)), "per_group": {g: dict(zip(names, v)) for g, v in per_group.items()}}
with open(sys.argv[2], "w") as f:
    json.dump(summary, f, indent=2)
print(f"[bulk_run_server.sh] wrote {sys.argv[2]}", file=sys.stderr)
PYEOF

[[ "$COUNT" -ge 500 ]] || die "generated only $COUNT scenarios, expected >= 500"
log "PASS: $COUNT >= 500 scenarios generated"
