#!/usr/bin/env bash
# Compares `ddai-map::load_map` (via `ddnet-ai trace export-map --map`) against `map2raw` (task
# 1.2/1.4's C++ oracle over the REAL DDNet 20.1 loader) across a corpus of real `.map` files:
# task 1.4's acceptance criterion #5 — "Rust load_map -> rawmap bytes identical to map2raw
# output. Any map DDNet itself rejects must also be rejected by us (and vice versa)".
#
# Usage: ./map-corpus-check.sh <dir-or-file>...
#   ./map-corpus-check.sh ~/aiddnet/data/research/proto-scratch/ddnet-maps \
#       ~/aiddnet/data/maps/copy-love-box ~/aiddnet/data/research/physics-scratch/maps
#
# `--mutate [--seed N] [--iterations-per-map N] <map-file>...` switches to the *differential
# mutation* mode instead (review round 1: corpus maps alone are all real, well-formed maps that
# both loaders accept, so they never exercise the reject/accept-parity half of criterion #5 at
# all — this mode randomly corrupts a handful of fields inside a real map's own LAYER items and
# checks `map2raw`/`ddai-map` still agree, byte-for-byte or on rejection, on the mutated result).
# Delegates to `map_mutation_fuzz.py`, which has the exact mutation strategy and known-divergence
# notes; see that file's own docstring.
#
# Requires `./build.sh` to have already built `build/map2raw`, and `cargo build --release -p
# ddnet-ai` to have already built `../../target/release/ddnet-ai` (this script does not build
# either itself — both are slow to (re)build and this script is meant to be re-run standalone).
set -uo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "--mutate" ]; then
	shift
	exec python3 map_mutation_fuzz.py "$@"
fi

readonly MAP2RAW="build/map2raw"
readonly DDNET_AI="../../target/release/ddnet-ai"
readonly TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# `map2raw` links REAL, unmodified DDNet loading code, which — like real DDNet itself — has no
# bounded-allocation guarantees at all (see docs/formats.md §10.2; `ddai-map`'s own bounded reads,
# criterion #2, are exactly what this crate adds *on top of* that), and even `ddnet-ai`/`ddai-map`
# gets the exact same external cap for defense-in-depth/consistency (review round 2 finding F7):
# a bug in its bounded-allocation design should surface as a crash here, not as unbounded host
# memory use. This VPS is shared with several other builders, so EVERY invocation of EITHER tool
# below runs under a hard virtual-memory cap and a wall-clock timeout (an incident during review
# round 1's differential mutation fuzzing hit ~6.8 GB RSS on one such input) — `run_capped` is the
# only thing in this script allowed to invoke either binary directly. A capped run's exit status
# is `0` (accepted), `1` (the tool's own clean rejection), or anything else (timeout — `124` — or
# killed by the cap/a crash — `128+signal`), the last of which means the tool hit *its own*
# resource limit rather than reaching a considered answer; see the per-status handling below for
# why that's expected (and reported, not counted as a disagreement) for `map2raw` specifically,
# but always a real bug for `ddnet-ai`.
readonly MEM_LIMIT_BYTES=2147483648 # 2 GiB
readonly RUN_TIMEOUT=20s
run_capped() {
	timeout --signal=KILL "$RUN_TIMEOUT" prlimit "--as=$MEM_LIMIT_BYTES" "$@"
}

if [ ! -x "$MAP2RAW" ]; then
	echo "map-corpus-check.sh: $MAP2RAW not found — run ./build.sh first" >&2
	exit 1
fi
if [ ! -x "$DDNET_AI" ]; then
	echo "map-corpus-check.sh: $DDNET_AI not found — run 'cargo build --release -p ddnet-ai' first" >&2
	exit 1
fi

if [ "$#" -eq 0 ]; then
	echo "usage: $0 <dir-or-file>..." >&2
	exit 1
fi

total=0
agree=0
both_rejected=0
disagree_bytes=0
disagree_accept=0
cpp_resource_limited=0
rust_crashed=0
declare -a disagreement_lines=()
declare -a resource_limited_lines=()
declare -a rust_crash_lines=()

# Finds every `.map` file under each argument (or the argument itself, if it's already a file).
mapfile -t files < <(for arg in "$@"; do
	if [ -f "$arg" ]; then
		printf '%s\n' "$arg"
	else
		find "$arg" -type f -name '*.map'
	fi
done | sort)

echo "map-corpus-check.sh: checking ${#files[@]} map(s)..."

for f in "${files[@]}"; do
	total=$((total + 1))
	cpp_out="$TMP/cpp.rawmap"
	rust_out="$TMP/rust.rawmap"
	rm -f "$cpp_out" "$rust_out"

	cpp_err=$(run_capped "$MAP2RAW" "$f" "$cpp_out" 2>&1 >/dev/null)
	cpp_status=$?
	rust_err=$(run_capped "$DDNET_AI" trace export-map --map "$f" --out "$rust_out" 2>&1 >/dev/null)
	rust_status=$?

	# `ddnet-ai`/`ddai-map` hitting its own resource limit or crashing is a real bug (acceptance
	# criterion #2: never panics, bounded memory) — always reported and always fails the run,
	# checked first so it's never masked by a coincidental `map2raw` outcome on the same map.
	if [ "$rust_status" -gt 1 ]; then
		rust_crashed=$((rust_crashed + 1))
		rust_crash_lines+=("RUST/ddai-map CRASHED: $f — exit=$rust_status ($rust_err)")
		continue
	fi
	# Any `map2raw` status other than 0 (accepted) or 1 (its own clean `Fail()` — a real "DDNet
	# rejects this") means the oracle hit its OWN resource limit, not a considered rejection —
	# reported separately, and only ever compared against Rust informationally, never counted as
	# a parity disagreement (`ddai-map` being bounded/successful where the disposable oracle
	# isn't is the *expected*, documented shape of this divergence — docs/formats.md §10.2).
	if [ "$cpp_status" -gt 1 ]; then
		cpp_resource_limited=$((cpp_resource_limited + 1))
		resource_limited_lines+=("CPP (map2raw) RESOURCE-LIMITED: $f — map2raw exit=$cpp_status ($cpp_err) ; ddai-map exit=$rust_status ($rust_err)")
		continue
	fi
	if [ "$cpp_status" -ne 0 ] && [ "$rust_status" -ne 0 ]; then
		both_rejected=$((both_rejected + 1))
		continue
	fi
	if [ "$cpp_status" -ne 0 ] || [ "$rust_status" -ne 0 ]; then
		disagree_accept=$((disagree_accept + 1))
		disagreement_lines+=("ACCEPT/REJECT MISMATCH: $f — map2raw exit=$cpp_status ($cpp_err) ; ddai-map exit=$rust_status ($rust_err)")
		continue
	fi
	if cmp -s "$cpp_out" "$rust_out"; then
		agree=$((agree + 1))
	else
		disagree_bytes=$((disagree_bytes + 1))
		cpp_size=$(stat -c%s "$cpp_out")
		rust_size=$(stat -c%s "$rust_out")
		first_diff=$(cmp "$cpp_out" "$rust_out" 2>&1 || true)
		disagreement_lines+=("BYTE MISMATCH: $f — map2raw ${cpp_size}B, ddai-map ${rust_size}B — $first_diff")
	fi
done

echo
echo "=== map-corpus-check.sh summary ==="
echo "total maps checked:            $total"
echo "agree (bytes identical):       $agree"
echo "agree (both rejected):         $both_rejected"
echo "disagree (byte mismatch):      $disagree_bytes"
echo "disagree (accept/reject):      $disagree_accept"
echo "cpp (map2raw) resource-limited: $cpp_resource_limited  (not a disagreement — see above)"
echo "RUST/ddai-map crashed:          $rust_crashed  (ALWAYS a bug if nonzero)"
echo

if [ "${#resource_limited_lines[@]}" -gt 0 ]; then
	echo "=== cpp (map2raw) resource-limited (informational only) ==="
	for line in "${resource_limited_lines[@]}"; do
		echo "$line"
	done
	echo
fi

if [ "${#rust_crash_lines[@]}" -gt 0 ]; then
	echo "=== RUST/ddai-map CRASHED (this is always a bug) ==="
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

echo "map-corpus-check.sh: 100% agreement across $total maps ($cpp_resource_limited map2raw resource-limited, informational)."
exit 0
