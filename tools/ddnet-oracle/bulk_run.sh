#!/usr/bin/env bash
# Runs Oracle A over a large batch of scenarios (default: 4 recipes x 50 seeds = 200 scenarios,
# 3000 ticks x 3 characters each, plus a small `no_weak_hook`/tuning-override slice — review
# round 1, finding F8) into ~/aiddnet/data/traces/oracle-a/v1/ (outside the repo, per the task
# spec) for task 1.3's bulk parity test. Reports total size and wall time.
#
# Usage: ./bulk_run.sh [num_seeds] [ticks] [chars]
set -euo pipefail
cd "$(dirname "$0")"

readonly NUM_SEEDS="${1:-50}"
readonly TICKS="${2:-3000}"
readonly CHARS="${3:-3}"
readonly REPO_ROOT="$(cd ../.. && pwd)"
readonly DDNET_AI_BIN="$REPO_ROOT/target/release/ddnet-ai"
readonly ORACLE="build/oracle_core"
readonly OUT_DIR="$HOME/aiddnet/data/traces/oracle-a/v1"
readonly WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

if [ ! -x "$ORACLE" ]; then
	echo "bulk_run.sh: $ORACLE not built — run ./fetch.sh && ./build.sh first" >&2
	exit 1
fi
echo "bulk_run.sh: building ddnet-ai (release)"
(cd "$REPO_ROOT" && source "$HOME/.cargo/env" && cargo build --release -q -p ddnet-ai)

# Clear any stale corpus from a previous (possibly pre-fix) run so its traces are never mistaken
# for output of the current code.
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
manifest="$OUT_DIR/manifest.tsv"
echo -e "recipe\tseed\tticks\tcharacters\tvariant\ttrace_file\ttrace_sha256\tscenario_sha256" >"$manifest"

readonly RECIPES=(arena freeze front tele-speedup)
declare -A MAP_PATH
for recipe in "${RECIPES[@]}"; do
	MAP_PATH[$recipe]="$WORK_DIR/$recipe.rawmap"
	"$DDNET_AI_BIN" trace export-map --recipe "$recipe" --out "${MAP_PATH[$recipe]}" >/dev/null
done

run_one() {
	local recipe="$1" seed="$2" variant="$3"
	shift 3
	local scn="$WORK_DIR/scn.bin"
	local trace_name="${recipe}_${variant}_${seed}.trace"
	local trace_path="$OUT_DIR/$trace_name"
	"$DDNET_AI_BIN" trace gen-scenario --recipe "$recipe" --seed "$seed" --ticks "$TICKS" --chars "$CHARS" "$@" --out "$scn" >/dev/null
	"$ORACLE" "${MAP_PATH[$recipe]}" "$scn" "$trace_path" --generator random-v1 --seed "$seed" 2>>"$OUT_DIR/oracle.log"
	local trace_sha256 scenario_sha256
	trace_sha256="$(sha256sum "$trace_path" | cut -d' ' -f1)"
	scenario_sha256="$(sha256sum "$scn" | cut -d' ' -f1)"
	echo -e "$recipe\t$seed\t$TICKS\t$CHARS\t$variant\t$trace_name\t$trace_sha256\t$scenario_sha256" >>"$manifest"
}

start_time=$(date +%s.%N)
count=0
for recipe in "${RECIPES[@]}"; do
	for seed in $(seq 1 "$NUM_SEEDS"); do
		count=$((count + 1))
		run_one "$recipe" "$seed" main
	done
done

# `no_weak_hook`/tuning-override slice: a handful of scenarios per recipe exercising the world
# flag and tuning-override paths, which the main loop above never sets (review round 1, finding
# F8 — before this, `no_weak_hook=true` and any tuning override never appeared anywhere in the
# bulk corpus or the fixtures, despite the oracle supporting both since task 1.2's first draft).
readonly NWH_SEEDS=5
for recipe in "${RECIPES[@]}"; do
	for seed in $(seq 1001 $((1000 + NWH_SEEDS))); do
		count=$((count + 1))
		run_one "$recipe" "$seed" nwh --no-weak-hook --tune gravity=0
	done
done
end_time=$(date +%s.%N)

elapsed=$(echo "$end_time - $start_time" | bc)
total_bytes=$(du -cb "$OUT_DIR"/*.trace | tail -1 | cut -f1)
main_count=$((${#RECIPES[@]} * NUM_SEEDS))
nwh_count=$((${#RECIPES[@]} * NWH_SEEDS))
echo "bulk_run.sh: $count scenarios total ($main_count main + $nwh_count no_weak_hook/tune), $TICKS ticks, $CHARS characters each"
echo "bulk_run.sh: wall time: ${elapsed}s"
echo "bulk_run.sh: total trace size: $total_bytes bytes ($(du -sh "$OUT_DIR" | cut -f1))"
echo "bulk_run.sh: manifest: $manifest"
