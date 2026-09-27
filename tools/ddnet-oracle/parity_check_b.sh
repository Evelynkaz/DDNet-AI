#!/usr/bin/env bash
# Oracle A / Oracle B consistency check (task 1.5 spec, acceptance criterion 7): "on scenarios
# without any tiles/weapons effects (arena recipe, fire never pressed), the core fields of
# Oracle B equal Oracle A's -- show a check over >= 20 scenarios, or explain every expected
# difference".
#
# For each of >= 20 seeds: generates an arena-recipe rawmap+scenario via the existing Rust CLI
# (crates/ddnet-ai, unchanged), zeroes every tick's `fire` field (scripts/zero_fire.py -- so
# HandleWeapons()/FireWeapon() never actually fires, matching the acceptance criterion's "fire
# never pressed"), runs the SAME resulting scenario file through both oracles, and uses Oracle
# B's own `--compare-oracle-a` mode to diff all 28 documented core fields EXCEPT `active_weapon`
# (excluded on purpose -- see oracle_server.cpp's comment at the comparison site: Oracle A's
# core-only tick never advances this field at all, by construction, so it can never agree with
# Oracle B's real `CCharacter::Spawn()`/weapon-switch logic; every other field IS compared).
#
# Usage: tools/ddnet-oracle/parity_check_b.sh [num_seeds]
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."

NUM_SEEDS="${1:-20}"
ORACLE_A_BIN="$SCRIPT_DIR/build/oracle_core"
ORACLE_B_BIN="${ORACLE_B_BIN:-$HOME/aiddnet/build/oracle-b/build/ddai_oracle_server}"
DDNET_AI_BIN="$REPO_ROOT/target/release/ddnet-ai"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

log() { printf '[parity_check_b.sh] %s\n' "$*" >&2; }
die() { printf '[parity_check_b.sh] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -x "$ORACLE_A_BIN" ]] || die "$ORACLE_A_BIN not found -- run tools/ddnet-oracle/fetch.sh && build.sh first"
[[ -x "$ORACLE_B_BIN" ]] || die "$ORACLE_B_BIN not found -- run tools/ddnet-oracle/build-server-oracle.sh first"
[[ -x "$DDNET_AI_BIN" ]] || die "$DDNET_AI_BIN not found -- run 'cargo build --release -p ddnet-ai' first"

"$DDNET_AI_BIN" trace export-map --recipe arena --out "$WORK_DIR/arena.rawmap" >/dev/null

TOTAL_MISMATCHES=0
FAILED_SEEDS=()
for seed in $(seq 1 "$NUM_SEEDS"); do
  chars=$((2 + seed % 3)) # 2..4 characters, varied across seeds
  ticks=600
  "$DDNET_AI_BIN" trace gen-scenario --recipe arena --seed "$seed" --ticks "$ticks" --chars "$chars" \
    --out "$WORK_DIR/s_$seed.scn" >/dev/null
  python3 "$SCRIPT_DIR/zero_fire.py" "$WORK_DIR/s_$seed.scn" "$WORK_DIR/s_${seed}_nofire.scn"

  "$ORACLE_A_BIN" "$WORK_DIR/arena.rawmap" "$WORK_DIR/s_${seed}_nofire.scn" "$WORK_DIR/a_$seed.trace" \
    --generator random-v1 --seed "$seed" >/dev/null

  storage_dir="$WORK_DIR/storage_$seed"
  set +e
  output=$("$ORACLE_B_BIN" \
    --storage-dir "$storage_dir" \
    --rawmap "$WORK_DIR/arena.rawmap" \
    --scenario "$WORK_DIR/s_${seed}_nofire.scn" \
    --seed "$seed" \
    --out "$WORK_DIR/b_$seed.trb" \
    --compare-oracle-a "$WORK_DIR/a_$seed.trace" 2>&1)
  rc=$?
  set -e

  mismatch_line=$(echo "$output" | grep -F 'core-field mismatches' || true)
  mismatches=$(echo "$mismatch_line" | grep -oE '^[^:]*: --compare-oracle-a: [0-9]+' | grep -oE '[0-9]+$' || echo "?")
  log "seed=$seed chars=$chars rc=$rc mismatches=$mismatches"
  if [[ "$rc" -ne 0 ]]; then
    FAILED_SEEDS+=("$seed")
    echo "$output" | grep -F 'first mismatch' >&2 || true
    TOTAL_MISMATCHES=$((TOTAL_MISMATCHES + 1))
  fi
done

log "checked $NUM_SEEDS seeds (arena recipe, fire zeroed)"
if [[ "${#FAILED_SEEDS[@]}" -eq 0 ]]; then
  log "PASS: 0 unexplained core-field mismatches across $NUM_SEEDS scenarios (active_weapon excluded, see header)"
  exit 0
else
  die "FAIL: seeds with mismatches: ${FAILED_SEEDS[*]}"
fi
