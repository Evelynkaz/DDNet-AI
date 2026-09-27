#!/usr/bin/env bash
# Builds Oracle A, generates >= 20 scenarios (every synthetic recipe, several seeds/character
# counts each), and for every one of them runs the oracle twice and checks the two trace files
# are byte-identical (sha256) — Oracle A's determinism requirement (task spec, criterion 4d).
# Prints each run's throughput (tee-ticks/second, criterion 4e).
#
# Usage: ./selftest.sh
set -euo pipefail
cd "$(dirname "$0")"

readonly REPO_ROOT="$(cd ../.. && pwd)"
readonly DDNET_AI_BIN="$REPO_ROOT/target/release/ddnet-ai"
readonly WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "selftest.sh: building ddnet-ai (release)"
(cd "$REPO_ROOT" && source "$HOME/.cargo/env" && cargo build --release -q -p ddnet-ai)

echo "selftest.sh: fetching + building Oracle A"
./fetch.sh
./build.sh
readonly ORACLE="build/oracle_core"

readonly RECIPES=(arena freeze front tele-speedup)
readonly SEEDS=(1 2 3 4 5)
readonly TICKS=300

# One rawmap export per recipe (recipes are deterministic, so one export covers every seed).
declare -A MAP_PATH
for recipe in "${RECIPES[@]}"; do
	MAP_PATH[$recipe]="$WORK_DIR/$recipe.rawmap"
	"$DDNET_AI_BIN" trace export-map --recipe "$recipe" --out "${MAP_PATH[$recipe]}"
done

total=0
fail=0
for recipe in "${RECIPES[@]}"; do
	for seed in "${SEEDS[@]}"; do
		# Cycle through 1..4 characters across seeds so the character-count parameter is
		# exercised too, without needing a fifth loop dimension.
		chars=$(((seed - 1) % 4 + 1))
		total=$((total + 1))
		scn="$WORK_DIR/${recipe}_${seed}.scn"
		"$DDNET_AI_BIN" trace gen-scenario --recipe "$recipe" --seed "$seed" --ticks "$TICKS" --chars "$chars" --out "$scn" >/dev/null

		t1="$WORK_DIR/${recipe}_${seed}_a.trace"
		t2="$WORK_DIR/${recipe}_${seed}_b.trace"
		"$ORACLE" "${MAP_PATH[$recipe]}" "$scn" "$t1" --generator random-v1 --seed "$seed"
		"$ORACLE" "${MAP_PATH[$recipe]}" "$scn" "$t2" --generator random-v1 --seed "$seed"

		sha1="$(sha256sum "$t1" | cut -d' ' -f1)"
		sha2="$(sha256sum "$t2" | cut -d' ' -f1)"
		if [ "$sha1" = "$sha2" ]; then
			echo "selftest.sh: OK   $recipe seed=$seed chars=$chars ticks=$TICKS sha256=$sha1"
		else
			echo "selftest.sh: FAIL $recipe seed=$seed chars=$chars: sha256 mismatch ($sha1 != $sha2)" >&2
			fail=$((fail + 1))
		fi
	done
done

echo "selftest.sh: $total scenarios, $fail determinism failures"
if [ "$fail" -ne 0 ]; then
	exit 1
fi
echo "selftest.sh: all scenarios deterministic"
