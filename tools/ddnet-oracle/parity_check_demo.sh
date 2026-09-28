#!/usr/bin/env bash
# Compares `ddai-demo` (via `ddnet-ai demo dump --raw`) against `demo2json` (task 8.4b's C++
# oracle over REAL DDNet 20.1's own demo-reading code) across a corpus of real `.demo` files:
# task 8.4b acceptance criterion 2 — "check byte-identical item data on every demo of
# ChillerDragon's archive and the samples in public-samples".
#
# Usage: ./parity_check_demo.sh <dir-or-file>...
#   ./parity_check_demo.sh ~/aiddnet/data/demos/chillerdragon/block-06 ~/aiddnet/data/demos/public-samples
#
# Requires `./build.sh` to have already built `build/demo2json`, and `cargo build --release -p
# ddnet-ai` to have already built `../../target/release/ddnet-ai` (this script does not build
# either itself — both are slow to (re)build and this script is meant to be re-run standalone).
set -uo pipefail
cd "$(dirname "$0")"

readonly DEMO2JSON="build/demo2json"
readonly DDNET_AI="../../target/release/ddnet-ai"
readonly TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Same rationale as map-corpus-check.sh: `demo2json` links REAL, unmodified DDNet demo-reading
# code, which has no bounded-allocation guarantees of its own; this VPS is shared, so every
# invocation of either tool runs under a hard memory cap and wall-clock timeout. A capped run's
# exit status is 0 (success), 1 (the tool's own clean rejection/`Fail()`), or anything else
# (timeout `124`, or killed by the cap/a crash `128+signal`) — the last of which means the tool
# hit *its own* resource limit, reported separately for the C++ oracle (expected/documented,
# never counted as a disagreement) but always a real bug if it's `ddnet-ai`.
readonly MEM_LIMIT_BYTES=2147483648 # 2 GiB
readonly RUN_TIMEOUT=60s
run_capped() {
	timeout --signal=KILL "$RUN_TIMEOUT" prlimit "--as=$MEM_LIMIT_BYTES" "$@"
}

if [ ! -x "$DEMO2JSON" ]; then
	echo "parity_check_demo.sh: $DEMO2JSON not found — run ./build.sh first" >&2
	exit 1
fi
if [ ! -x "$DDNET_AI" ]; then
	echo "parity_check_demo.sh: $DDNET_AI not found — run 'cargo build --release -p ddnet-ai' first" >&2
	exit 1
fi
if [ "$#" -eq 0 ]; then
	echo "usage: $0 <dir-or-file>..." >&2
	exit 1
fi

# Finds every `.demo` file under each argument (or the argument itself, if it's already a file),
# skipping dot-directories (ChillerDragon's archive is checked out as its own `.git` repo).
mapfile -t files < <(for arg in "$@"; do
	if [ -f "$arg" ]; then
		printf '%s\n' "$arg"
	else
		find "$arg" -type f -name '*.demo' -not -path '*/.*'
	fi
done | sort)

echo "parity_check_demo.sh: checking ${#files[@]} demo(s)..."

total=0
agree=0
both_rejected=0
disagree_bytes=0
disagree_accept=0
cpp_resource_limited=0
rust_crashed=0
total_ticks=0
declare -a disagreement_lines=()
declare -a resource_limited_lines=()
declare -a rust_crash_lines=()

for f in "${files[@]}"; do
	total=$((total + 1))
	cpp_out="$TMP/cpp.jsonl"
	rust_out="$TMP/rust.jsonl"
	rm -f "$cpp_out" "$rust_out"

	cpp_err=$(run_capped "$DEMO2JSON" "$f" "$cpp_out" 2>&1 >/dev/null)
	cpp_status=$?
	rust_err=$(run_capped "$DDNET_AI" demo dump --raw "$f" >"$rust_out" 2>&1)
	rust_status=$?

	if [ "$rust_status" -gt 1 ]; then
		rust_crashed=$((rust_crashed + 1))
		rust_crash_lines+=("RUST/ddai-demo CRASHED: $f — exit=$rust_status ($rust_err)")
		continue
	fi
	if [ "$cpp_status" -gt 1 ]; then
		cpp_resource_limited=$((cpp_resource_limited + 1))
		resource_limited_lines+=("CPP (demo2json) RESOURCE-LIMITED: $f — demo2json exit=$cpp_status ($cpp_err) ; ddai-demo exit=$rust_status")
		continue
	fi
	if [ "$cpp_status" -ne 0 ] && [ "$rust_status" -ne 0 ]; then
		both_rejected=$((both_rejected + 1))
		continue
	fi
	if [ "$cpp_status" -ne 0 ] || [ "$rust_status" -ne 0 ]; then
		disagree_accept=$((disagree_accept + 1))
		disagreement_lines+=("ACCEPT/REJECT MISMATCH: $f — demo2json exit=$cpp_status ($cpp_err) ; ddai-demo exit=$rust_status ($rust_err)")
		continue
	fi

	# demo2json's first line is the header (not produced by `demo dump --raw`, which is
	# per-tick items only) — strip it before comparing.
	tail -n +2 "$cpp_out" >"$TMP/cpp_ticks.jsonl"

	if cmp -s "$TMP/cpp_ticks.jsonl" "$rust_out"; then
		agree=$((agree + 1))
		total_ticks=$((total_ticks + $(wc -l <"$rust_out")))
	else
		disagree_bytes=$((disagree_bytes + 1))
		cpp_lines=$(wc -l <"$TMP/cpp_ticks.jsonl")
		rust_lines=$(wc -l <"$rust_out")
		first_diff=$(diff "$TMP/cpp_ticks.jsonl" "$rust_out" 2>&1 | head -5 || true)
		disagreement_lines+=("MISMATCH: $f — demo2json ${cpp_lines} ticks, ddai-demo ${rust_lines} ticks
$first_diff")
	fi
done

echo
echo "=== parity_check_demo.sh summary ==="
echo "total demos checked:            $total"
echo "agree (items byte-identical):   $agree"
echo "agree (both rejected):          $both_rejected"
echo "disagree (item mismatch):       $disagree_bytes"
echo "disagree (accept/reject):       $disagree_accept"
echo "cpp (demo2json) resource-limited: $cpp_resource_limited  (not a disagreement — see above)"
echo "RUST/ddai-demo crashed:          $rust_crashed  (ALWAYS a bug if nonzero)"
echo "total ticks compared (agreeing demos): $total_ticks"
echo

if [ "${#resource_limited_lines[@]}" -gt 0 ]; then
	echo "=== cpp (demo2json) resource-limited (informational only) ==="
	for line in "${resource_limited_lines[@]}"; do
		echo "$line"
	done
	echo
fi

if [ "${#rust_crash_lines[@]}" -gt 0 ]; then
	echo "=== RUST/ddai-demo CRASHED (this is always a bug) ==="
	for line in "${rust_crash_lines[@]}"; do
		echo "$line"
	done
	echo
fi

if [ "${#disagreement_lines[@]}" -gt 0 ]; then
	echo "=== disagreements ==="
	for line in "${disagreement_lines[@]}"; do
		echo "$line"
	done
fi

if [ "${#disagreement_lines[@]}" -gt 0 ] || [ "${#rust_crash_lines[@]}" -gt 0 ]; then
	exit 1
fi

echo "parity_check_demo.sh: 100% agreement across $total demos ($cpp_resource_limited demo2json resource-limited, informational)."
exit 0
