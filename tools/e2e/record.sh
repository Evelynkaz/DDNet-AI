#!/usr/bin/env bash
# tools/e2e/record.sh — task 8.4a acceptance criterion 4: end-to-end observer-recorder test
# against the LOCAL DDNet 20.1 server (`ddnet-local.service`, 127.0.0.1:8303, econ
# 127.0.0.1:8304) — see CLAUDE.md's live-play policy: this script must never touch any address
# other than 127.0.0.1 (the live-servers.toml safety switch also refuses anything else, but this
# script does not even try).
#
# Phase 1: starts 3 scripted `ddnet-ai play` clients (two `--brain circle`, one `--brain
# random-scripted` with `--input-log` as ground truth) plus `ddnet-ai record` as a fourth,
# pure-observer client; changes the map mid-session (BlmapChill, then back to Copy Love Box); does
# a graceful server restart mid-session; lets everything run to completion; then verifies the
# *last* recorded segment (the one after the restart — see `record_cmd.rs`'s doc comment on why a
# map change starts a fresh segment) decodes, reports its input-reconstruction accuracy against the
# random-scripted bot's logged ground truth (a loose, "did reconstruction run at all" check),
# confirms the recorder's own outgoing inputs were all neutral, and (delegated to the Rust unit
# test suite, the same pattern task 2.3's `session.sh` scenario (g) already established) that
# `Cl_Say` has no reachable send path in this crate at all.
#
# Phase 2 (review round 2, finding F16): the two circle bots make phase 1 structurally unable to
# produce enough non-frozen ("free") ticks to judge per-field accuracy from — so the stricter
# free-tick gate runs separately here, against the validation bot running ALONE (no circle bots)
# on ChillBlock5, a map/scenario combination measured to spend most of its time unfrozen. Review
# round 3, finding F20: pools 3 independent, short (30s) sub-runs — each its own fresh connection,
# so the validator respawns at its map start point every time — and gates on the *summed* raw
# counts, not any one sub-run's own fraction, removing the "given enough time it wanders into
# freeze and never comes back" flake a single longer run had.
#
# Restores the server to map "Copy Love Box" on exit, always (even on failure/Ctrl-C).
#
# Usage: tools/e2e/record.sh

set -uo pipefail

# shellcheck disable=SC1090
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DDAI_DATA_DIR:-$HOME/aiddnet/data}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$DATA_DIR/logs/e2e-8.4a/$STAMP"
RECORDINGS_DIR="$RUN_DIR/recordings"
mkdir -p "$RUN_DIR" "$RECORDINGS_DIR"

ECON=(python3 "$REPO_ROOT/tools/ddnet-server/econ.py")
BOT="$REPO_ROOT/target/debug/ddnet-ai"
ORIGINAL_MAP="Copy Love Box"
# Phase 1's own duration — next to two `--brain circle` bots on a genuine block map, `RSValid` (a
# `random-scripted` brain with no evasion behavior) spends most of its time frozen (measured 34/95
# and 9/266 free ticks at 45s/90s respectively across two live tuning runs — duration alone is not
# a reliable way to raise this). Review round 2, finding F16: the free-tick accuracy gate no longer
# runs against phase 1 at all — it moved to a dedicated phase 2 (below) that removes the circle
# bots entirely, which is the actual fix, not just "wait longer next to what causes the problem".
# Phase 1 itself still keeps its own, looser check (`reconstruct_accuracy`, >= 50 overlapping
# ticks) — segmenting/decoding/map-change/restart/audit coverage, unrelated to per-field accuracy.
DURATION=45

declare -a RESULTS=()
declare -a BOT_PIDS=()
PASS_COUNT=0
FAIL_COUNT=0

pass() {
    echo "PASS $1: $2"
    RESULTS+=("PASS $1: $2")
    PASS_COUNT=$((PASS_COUNT + 1))
}
fail() {
    echo "FAIL $1: $2"
    RESULTS+=("FAIL $1: $2")
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

cleanup() {
    echo "cleaning up: killing any still-running bot processes ..."
    for pid in "${BOT_PIDS[@]:-}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done
    echo "restoring map to '$ORIGINAL_MAP' ..."
    "${ECON[@]}" change_map "$ORIGINAL_MAP" >"$RUN_DIR/restore-map.log" 2>&1
    sleep 1
    systemctl is-active --quiet ddnet-local.service || sudo systemctl start ddnet-local.service
}
trap cleanup EXIT

echo "building ddnet-ai (debug) ..."
(cd "$REPO_ROOT" && cargo build -p ddnet-ai --bin ddnet-ai) >"$RUN_DIR/build.log" 2>&1 || {
    fail all "cargo build failed, see $RUN_DIR/build.log"
    exit 1
}

TRUTH_LOG="$RUN_DIR/truth-inputs.jsonl"
RECORDER_INPUT_LOG="$RUN_DIR/recorder-inputs.jsonl"

echo "starting 2 circle bots, 1 random-scripted validation bot, and the recorder ..."
"$BOT" play --server 127.0.0.1:8303 --name e2aC1 --brain circle --duration "$DURATION" \
    --data-dir "$DATA_DIR" >"$RUN_DIR/circle1.log" 2>&1 &
BOT_PIDS+=($!)
"$BOT" play --server 127.0.0.1:8303 --name e2aC2 --brain circle --duration "$DURATION" \
    --data-dir "$DATA_DIR" >"$RUN_DIR/circle2.log" 2>&1 &
BOT_PIDS+=($!)
"$BOT" play --server 127.0.0.1:8303 --name RSValid --brain random-scripted --seed 7 \
    --duration "$DURATION" --input-log "$TRUTH_LOG" --data-dir "$DATA_DIR" >"$RUN_DIR/rsvalid.log" 2>&1 &
BOT_PIDS+=($!)
RUST_LOG=info "$BOT" record --server 127.0.0.1:8303 --name RecE2E --duration "$DURATION" \
    --out "$RECORDINGS_DIR" --input-log "$RECORDER_INPUT_LOG" --data-dir "$DATA_DIR" \
    >"$RUN_DIR/record.log" 2>&1 &
BOT_PIDS+=($!)

sleep 8
echo "changing map to BlmapChill mid-session ..."
"${ECON[@]}" change_map "BlmapChill" >"$RUN_DIR/econ-blmapchill.log" 2>&1
sleep 8
echo "changing map back to '$ORIGINAL_MAP' mid-session ..."
"${ECON[@]}" change_map "$ORIGINAL_MAP" >"$RUN_DIR/econ-restore.log" 2>&1
sleep 5
echo "restarting ddnet-local.service (graceful) mid-session ..."
sudo systemctl restart ddnet-local.service

for pid in "${BOT_PIDS[@]}"; do
    wait "$pid" 2>/dev/null
done
BOT_PIDS=()

echo "=== all bots finished; verifying ==="

# --- (1) the recording exists and decodes ------------------------------------------------------
LAST_REC="$(find "$RECORDINGS_DIR" -maxdepth 1 -name '*.rec' -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | head -1 | cut -d' ' -f2- || true)"
if [ -z "$LAST_REC" ]; then
    fail rec_exists "no .rec file was ever written to $RECORDINGS_DIR (see record.log)"
else
    pass rec_exists "$(basename "$LAST_REC") (see $RECORDINGS_DIR for every segment)"
fi

if [ -n "$LAST_REC" ]; then
    inspect_log="$RUN_DIR/inspect.log"
    if "$BOT" rec inspect "$LAST_REC" --verify >"$inspect_log" 2>&1; then
        pass rec_decodes "decodes and sha256 sidecar verifies (see inspect.log)"
    else
        fail rec_decodes "ddnet-ai rec inspect --verify failed (see inspect.log)"
    fi
    cat "$inspect_log"
fi

# --- (2) input-reconstruction accuracy report (phase 1: general "did reconstruction run at all
#         and land in the right ballpark" check — the strict per-field free-tick gate moved to
#         phase 2 below, review round 2, finding F16) ------------------------------------------
if [ -n "$LAST_REC" ] && [ -s "$TRUTH_LOG" ]; then
    validate_log="$RUN_DIR/validate.log"
    if "$BOT" rec reconstruct "$LAST_REC" --validate "$TRUTH_LOG" --player-name RSValid >"$validate_log" 2>&1; then
        cat "$validate_log"
        compared="$(grep -oP 'compared ticks:\s+\K[0-9]+' "$validate_log" || echo 0)"
        direction_acc="$(grep -oP 'direction:\s+\K[0-9.]+' "$validate_log" || echo 0)"
        aim_acc="$(grep -oP 'aim:\s+\K[0-9.]+' "$validate_log" || echo 0)"
        if [ "$compared" -ge 50 ]; then
            pass reconstruct_accuracy "compared $compared ticks (direction=$direction_acc, aim=$aim_acc; see validate.log for the full per-field report)"
        else
            fail reconstruct_accuracy "only $compared overlapping ticks between the recording's last segment and the logged ground truth (see validate.log) — too few to judge accuracy from"
        fi
    else
        fail reconstruct_accuracy "ddnet-ai rec reconstruct --validate failed (see validate.log)"
    fi
else
    fail reconstruct_accuracy "missing recording or empty truth log — cannot validate (see rsvalid.log, record.log)"
fi

# --- (3) outgoing-input audit: zero non-neutral inputs from the recorder ------------------------
if [ -s "$RECORDER_INPUT_LOG" ]; then
    audit_report="$(python3 - "$RECORDER_INPUT_LOG" <<'PYEOF'
import json, sys
path = sys.argv[1]
total = 0
bad = 0
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        rec = json.loads(line)
        if rec["direction"] != 0 or rec["jump"] != 0 or rec["fire"] != 0 or rec["hook"] != 0:
            bad += 1
print(f"{total} {bad}")
PYEOF
)"
    read -r total bad <<<"$audit_report"
    if [ "$total" -gt 0 ] && [ "$bad" -eq 0 ]; then
        pass outgoing_audit_neutral "$total logged NETMSG_INPUT sends from the recorder, 0 non-neutral (see recorder-inputs.jsonl)"
    else
        fail outgoing_audit_neutral "$bad/$total logged sends were non-neutral (see recorder-inputs.jsonl)"
    fi
else
    fail outgoing_audit_neutral "recorder's --input-log is empty or missing (see record.log)"
fi

# --- (4) outgoing-input audit: zero chat (delegated to the Rust unit-test suite, task 2.3
#         scenario (g)'s own precedent — Cl_Say has no reachable send path in this crate at all) --
chat_log="$RUN_DIR/no-chat.log"
if (cd "$REPO_ROOT" && cargo test -p ddai-client --lib -- --nocapture) >"$chat_log" 2>&1; then
    pass outgoing_audit_chat "Cl_Say never reaches the wire (allow-list + full-join-sequence unit tests — see no-chat.log)"
else
    fail outgoing_audit_chat "a ddai-client unit test failed (see no-chat.log)"
fi

# --- Phase 2 (review round 2, finding F16; pooling per review round 3, finding F20 step (3)) -----
# Phase 1 above (segmenting/decoding/map-change/restart/audits) deliberately keeps the two
# `--brain circle` bots — that mixed-traffic scenario is exactly what task 8.4a's own e2e is meant
# to prove — but it structurally cannot produce enough free (non-frozen) ticks to also judge
# per-field accuracy from (the validator spends most of its time frozen next to them). The actual
# fix: a second, focused phase with the validator ALONE (no circle bots at all) on ChillBlock5 — a
# map the review round 2 reviewer measured at 131 free ticks in just 30s alone.
#
# Review round 3, finding F20 step (3): one long phase-2 run still had a "free-ticks flake" —
# `--brain random-scripted` has no freeze-avoidance at all, so *given enough time* it can wander
# into a stretch of the map it does not get back out of, dragging the free-tick count below the
# minimum on an unlucky run (live-measured: as low as 80 on one single 120s run, comfortably above
# 100 on others). Fixed by pooling `PHASE2_SUBRUNS` independent, short (`PHASE2_SUBRUN_DURATION`s)
# sub-runs instead of one long one — each is its own fresh `ddnet-ai play`/`record` connection pair
# (the validator respawns at its map start point every time, so no single sub-run's bad luck can
# compound across the whole phase), and the final gate sums each sub-run's own raw matched/total
# counts (never averages already-computed fractions, which would misweight a small-sample sub-run
# equally to a large one) before computing the final pooled fractions.
PHASE2_SUBRUN_DURATION=30
PHASE2_SUBRUNS=3
PHASE2_MAP="ChillBlock5"

# Review round 1's stricter gate: minimum per-field accuracies on *non-frozen* ("free") ticks only
# (a frozen tick has Direction/Jump/Hook forced to 0 by the server itself, per DDRaceTick — no
# reconstruction, however good, can be judged on those), on at least 100 free ticks — below that,
# a single lucky/unlucky run's noise could swing the fraction past any threshold either way.
MIN_FREE_TICKS=100
MIN_DIRECTION_FREE_ACC="0.95"
MIN_HOOK_FREE_ACC="0.90"
MIN_JUMP_FREE_RECALL="0.90"
# Review round 3, finding F20 step (2): keep the 0.90 recall threshold, but ALSO require at least
# this many executable jump presses pooled across all sub-runs — fewer is an outright FAIL
# ("insufficient sample"), never treated as a pass no matter what the recall fraction reads (a
# perfect but tiny sample, e.g. 2/2 = 1.0000, proves nothing). Deliberately NOT a Wilson lower
# bound: a perfect 16/16 has a 95% Wilson lower bound of about 0.81, which would fail a perfect
# run — exactly what the reviewer's own decision says not to do.
MIN_JUMP_EXECUTABLE_N=12
MIN_FIRE_FREE_PRECISION="0.90"
MIN_FIRE_FREE_RECALL="0.90"

echo "=== phase 2 (finding F16/F20): $PHASE2_MAP, validator alone + fresh recorder, $PHASE2_SUBRUNS x ${PHASE2_SUBRUN_DURATION}s pooled sub-runs ==="
echo "changing map to $PHASE2_MAP for phase 2 ..."
"${ECON[@]}" change_map "$PHASE2_MAP" >"$RUN_DIR/econ-phase2-map.log" 2>&1
sleep 2

# Pooled raw counts (review round 3, finding F20 step (3)) — summed across every sub-run that
# actually produced a validatable recording; a sub-run that fails outright (no recording, no
# truth log, or `rec reconstruct --validate` itself erroring) contributes nothing to the pool and
# is reported as its own separate failure, not silently skipped.
total_direction_matched=0
total_direction_total=0
total_hook_matched=0
total_hook_total=0
total_jump_matched=0
total_jump_n=0
total_fire_truth_matched=0
total_fire_truth_n=0
total_fire_recon_matched=0
total_fire_recon_n=0
subruns_ok=0

for i in $(seq 1 "$PHASE2_SUBRUNS"); do
    echo "--- phase 2 sub-run $i/$PHASE2_SUBRUNS (${PHASE2_SUBRUN_DURATION}s, fresh connection) ---"
    SUB_RECORDINGS_DIR="$RUN_DIR/recordings-phase2-sub$i"
    mkdir -p "$SUB_RECORDINGS_DIR"
    SUB_TRUTH_LOG="$RUN_DIR/phase2-sub$i-truth-inputs.jsonl"
    VALIDATOR_NAME="RSV2$i"
    RECORDER_NAME="REC2$i"

    "$BOT" play --server 127.0.0.1:8303 --name "$VALIDATOR_NAME" --brain random-scripted --seed "$((7 + i))" \
        --duration "$PHASE2_SUBRUN_DURATION" --input-log "$SUB_TRUTH_LOG" --data-dir "$DATA_DIR" \
        >"$RUN_DIR/phase2-sub$i-rsvalid.log" 2>&1 &
    BOT_PIDS+=($!)
    RUST_LOG=info "$BOT" record --server 127.0.0.1:8303 --name "$RECORDER_NAME" --duration "$PHASE2_SUBRUN_DURATION" \
        --out "$SUB_RECORDINGS_DIR" --data-dir "$DATA_DIR" >"$RUN_DIR/phase2-sub$i-record.log" 2>&1 &
    BOT_PIDS+=($!)

    for pid in "${BOT_PIDS[@]}"; do
        wait "$pid" 2>/dev/null
    done
    BOT_PIDS=()

    SUB_LAST_REC="$(find "$SUB_RECORDINGS_DIR" -maxdepth 1 -name '*.rec' -printf '%T@ %p\n' 2>/dev/null |
        sort -rn | head -1 | cut -d' ' -f2- || true)"

    if [ -z "$SUB_LAST_REC" ] || [ ! -s "$SUB_TRUTH_LOG" ]; then
        fail "phase2_subrun_${i}" "missing recording or empty truth log (see phase2-sub$i-rsvalid.log, phase2-sub$i-record.log)"
        continue
    fi

    sub_validate_log="$RUN_DIR/phase2-sub$i-validate.log"
    if ! "$BOT" rec reconstruct "$SUB_LAST_REC" --validate "$SUB_TRUTH_LOG" --player-name "$VALIDATOR_NAME" \
        >"$sub_validate_log" 2>&1; then
        fail "phase2_subrun_${i}" "ddnet-ai rec reconstruct --validate failed (see phase2-sub$i-validate.log)"
        cat "$sub_validate_log"
        continue
    fi
    cat "$sub_validate_log"

    d_m="$(grep -oP '^direction_free_matched:\s+\K[0-9]+' "$sub_validate_log" || echo 0)"
    d_t="$(grep -oP '^direction_free_total:\s+\K[0-9]+' "$sub_validate_log" || echo 0)"
    h_m="$(grep -oP '^hook_free_matched:\s+\K[0-9]+' "$sub_validate_log" || echo 0)"
    h_t="$(grep -oP '^hook_free_total:\s+\K[0-9]+' "$sub_validate_log" || echo 0)"
    j_m="$(grep -oP '^jump_executable_matched:\s+\K[0-9]+' "$sub_validate_log" || echo 0)"
    j_n="$(grep -oP '^jump_executable_n:\s+\K[0-9]+' "$sub_validate_log" || echo 0)"
    ft_m="$(grep -oP '^fire_free_truth_matched:\s+\K[0-9]+' "$sub_validate_log" || echo 0)"
    ft_n="$(grep -oP '^fire_free_truth_n:\s+\K[0-9]+' "$sub_validate_log" || echo 0)"
    fr_m="$(grep -oP '^fire_free_recon_matched:\s+\K[0-9]+' "$sub_validate_log" || echo 0)"
    fr_n="$(grep -oP '^fire_free_recon_n:\s+\K[0-9]+' "$sub_validate_log" || echo 0)"

    total_direction_matched=$((total_direction_matched + d_m))
    total_direction_total=$((total_direction_total + d_t))
    total_hook_matched=$((total_hook_matched + h_m))
    total_hook_total=$((total_hook_total + h_t))
    total_jump_matched=$((total_jump_matched + j_m))
    total_jump_n=$((total_jump_n + j_n))
    total_fire_truth_matched=$((total_fire_truth_matched + ft_m))
    total_fire_truth_n=$((total_fire_truth_n + ft_n))
    total_fire_recon_matched=$((total_fire_recon_matched + fr_m))
    total_fire_recon_n=$((total_fire_recon_n + fr_n))
    subruns_ok=$((subruns_ok + 1))
    pass "phase2_subrun_${i}" "recorded and validated (direction $d_m/$d_t, hook $h_m/$h_t, jump $j_m/$j_n, fire_truth $ft_m/$ft_n, fire_recon $fr_m/$fr_n)"
done

echo "=== phase 2 finished; verifying the pooled free-tick gate ($subruns_ok/$PHASE2_SUBRUNS sub-runs usable) ==="

pool_summary="pooled: direction=$total_direction_matched/$total_direction_total hook=$total_hook_matched/$total_hook_total jump=$total_jump_matched/$total_jump_n fire_truth=$total_fire_truth_matched/$total_fire_truth_n fire_recon=$total_fire_recon_matched/$total_fire_recon_n"
echo "$pool_summary"

if [ "$subruns_ok" -eq 0 ]; then
    fail reconstruct_free_gate "phase 2: every sub-run failed — nothing to pool ($pool_summary)"
elif [ "$total_direction_total" -lt "$MIN_FREE_TICKS" ]; then
    fail reconstruct_free_gate "phase 2: only $total_direction_total pooled non-frozen ticks (need >= $MIN_FREE_TICKS) ($pool_summary)"
elif [ "$total_jump_n" -lt "$MIN_JUMP_EXECUTABLE_N" ]; then
    fail reconstruct_free_gate "phase 2: only $total_jump_n pooled executable jump presses (need >= $MIN_JUMP_EXECUTABLE_N) — insufficient sample, not judged on recall at all ($pool_summary)"
elif python3 -c "
import sys
d_m, d_t, h_m, h_t, j_m, j_n, ft_m, ft_n, fr_m, fr_n = (int(x) for x in sys.argv[1:11])
direction = d_m / d_t if d_t else 1.0
hook = h_m / h_t if h_t else 1.0
jump_recall = j_m / j_n if j_n else 1.0
fire_recall = ft_m / ft_n if ft_n else 1.0
fire_precision = fr_m / fr_n if fr_n else 1.0
ok = (
    direction >= float('$MIN_DIRECTION_FREE_ACC')
    and hook >= float('$MIN_HOOK_FREE_ACC')
    and jump_recall >= float('$MIN_JUMP_FREE_RECALL')
    and fire_precision >= float('$MIN_FIRE_FREE_PRECISION')
    and fire_recall >= float('$MIN_FIRE_FREE_RECALL')
)
print(f'direction={direction:.4f} hook={hook:.4f} jump_recall={jump_recall:.4f} fire_precision={fire_precision:.4f} fire_recall={fire_recall:.4f}')
sys.exit(0 if ok else 1)
" "$total_direction_matched" "$total_direction_total" "$total_hook_matched" "$total_hook_total" \
    "$total_jump_matched" "$total_jump_n" "$total_fire_truth_matched" "$total_fire_truth_n" \
    "$total_fire_recon_matched" "$total_fire_recon_n" >"$RUN_DIR/phase2-pooled-fractions.log"; then
    cat "$RUN_DIR/phase2-pooled-fractions.log"
    pass reconstruct_free_gate "phase 2 ($PHASE2_MAP, $subruns_ok pooled sub-runs) meets minimum free-tick accuracies (direction>=$MIN_DIRECTION_FREE_ACC, hook>=$MIN_HOOK_FREE_ACC, jump_recall>=$MIN_JUMP_FREE_RECALL on >=$MIN_JUMP_EXECUTABLE_N executable presses, fire_precision>=$MIN_FIRE_FREE_PRECISION, fire_recall>=$MIN_FIRE_FREE_RECALL; $pool_summary)"
else
    cat "$RUN_DIR/phase2-pooled-fractions.log"
    fail reconstruct_free_gate "phase 2 ($PHASE2_MAP, $subruns_ok pooled sub-runs) below a minimum pooled accuracy threshold ($pool_summary; see phase2-pooled-fractions.log)"
fi

echo
echo "=== summary ==="
for r in "${RESULTS[@]}"; do
    echo "$r"
done
echo "$PASS_COUNT passed, $FAIL_COUNT failed"
echo "logs: $RUN_DIR"

[ "$FAIL_COUNT" -eq 0 ]
