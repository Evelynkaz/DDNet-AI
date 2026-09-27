#!/usr/bin/env bash
# Regenerates crates/ddai-trace/tests/fixtures/*.json — the golden fixtures task 1.3 will check
# Rust physics against. Each fixture records one scenario's generator params, its scenario
# bytes' sha256, the per-tick canonical FNV-1a 64 state hash for every tick (produced by Oracle
# A), and the full final-tick state — see docs/formats.md and
# crates/ddai-trace/tests/fixtures_test.rs.
#
# Usage: ./gen_fixtures.sh   (run after ./fetch.sh && ./build.sh)
set -euo pipefail
cd "$(dirname "$0")"

readonly REPO_ROOT="$(cd ../.. && pwd)"
readonly DDNET_AI_BIN="$REPO_ROOT/target/release/ddnet-ai"
readonly ORACLE="build/oracle_core"
readonly FIXTURES_DIR="$REPO_ROOT/crates/ddai-trace/tests/fixtures"
readonly WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

if [ ! -x "$ORACLE" ]; then
	echo "gen_fixtures.sh: $ORACLE not built — run ./fetch.sh && ./build.sh first" >&2
	exit 1
fi

echo "gen_fixtures.sh: building ddnet-ai (release)"
(cd "$REPO_ROOT" && source "$HOME/.cargo/env" && cargo build --release -q -p ddnet-ai)
mkdir -p "$FIXTURES_DIR"

readonly TICKS=600
# arena/freeze/front use 3 characters; tele-speedup uses 4 (task spec: "1 recipe may use 4").
declare -A CHARS=([arena]=3 [freeze]=3 [front]=3 [tele-speedup]=4)

gen_one() {
	local recipe="$1" seed="$2" chars="$3" fixture_name="$4"
	shift 4
	local map_path="$WORK_DIR/$recipe.rawmap"
	local scn_path="$WORK_DIR/${fixture_name}.scn"
	local trace_path="$WORK_DIR/${fixture_name}.trace"
	"$DDNET_AI_BIN" trace gen-scenario --recipe "$recipe" --seed "$seed" --ticks "$TICKS" --chars "$chars" "$@" --out "$scn_path" >/dev/null
	"$ORACLE" "$map_path" "$scn_path" "$trace_path" --generator random-v1 --seed "$seed"

	local scenario_sha256 hashes_path
	scenario_sha256="$(sha256sum "$scn_path" | cut -d' ' -f1)"
	hashes_path="$WORK_DIR/${fixture_name}.hashes.json"
	"$DDNET_AI_BIN" trace hashes "$trace_path" --out "$hashes_path" >/dev/null

	# `build_fixture.py` accepts the exact same `--no-weak-hook`/`--tune NAME=VALUE` flags as
	# `ddnet-ai trace gen-scenario` (on purpose) — "$@" is passed through unchanged to both, so
	# the fixture always records exactly what the scenario was actually generated with.
	python3 "$(dirname "$0")/build_fixture.py" \
		--recipe "$recipe" --seed "$seed" --ticks "$TICKS" --chars "$chars" \
		--scenario-sha256 "$scenario_sha256" \
		--trace "$trace_path" --hashes "$hashes_path" \
		"$@" \
		--out "$FIXTURES_DIR/${fixture_name}.json"
	echo "gen_fixtures.sh: wrote $FIXTURES_DIR/${fixture_name}.json"
}

for recipe in arena freeze front tele-speedup; do
	chars="${CHARS[$recipe]}"
	"$DDNET_AI_BIN" trace export-map --recipe "$recipe" --out "$WORK_DIR/$recipe.rawmap" >/dev/null

	for seed in 101 102; do
		gen_one "$recipe" "$seed" "$chars" "${recipe}_${seed}"
	done
done

# no_weak_hook / tuning-override fixtures (review round 1, finding F8): before this, neither
# path was represented anywhere in the committed fixtures. `arena_tune_202`'s overrides (review
# round 2, finding F11) are non-default but realistic — `gravity=40`/`hook_length=50000`/
# `hook_drag_speed=1800` all differ from their 20.1 defaults (tuning.h: 50/38000/1500) while
# still letting characters fall and hook the ground/each other; the old `--tune gravity=0
# --tune hook_length=38000` zeroed gravity (so nobody ever fell into anyone) and its
# `hook_length` override was silently the default, leaving this fixture with 0 hook-on-player
# events.
gen_one arena 201 3 arena_nwh_201 --no-weak-hook
gen_one arena 202 3 arena_tune_202 --tune gravity=40 --tune hook_length=50000 --tune hook_drag_speed=1800

du -ch "$FIXTURES_DIR"/*.json | tail -1
