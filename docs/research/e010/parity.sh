#!/bin/bash
cd ~/aiddnet/wt/task-3.6
source ~/.cargo/env; export RUSTC_WRAPPER=sccache SCCACHE_DIR=~/aiddnet/data/cache/sccache SCCACHE_CACHE_SIZE=20G CARGO_BUILD_JOBS=3 RUST_TEST_THREADS=3
T=~/aiddnet/data/traces
F="--features ts-parity --release --locked"
for m in clb blmapchill; do
  echo "== components $m"
  DDAI_COMPONENT_DUMP=$T/planner-components/$m.jsonl cargo test -p ddai-planner $F --test parity_components -- --ignored --nocapture 2>&1 | grep -E "mismatch|cases|test result"
done
echo "== planner dumps"
DDAI_PLANNER_DUMP_DIR=$T/planner cargo test -p ddai-planner $F --test parity_planner -- --ignored --nocapture 2>&1 | grep -E "replayed|test result|mismatch"
echo "== freerun"
DDAI_PLANNER_FREERUN_DIR=$T/planner-freerun cargo test -p ddai-planner $F --test parity_planner_freerun -- --ignored --nocapture 2>&1 | grep -E "replayed|test result|mismatch"
echo "== planner lib+tests with ts-parity (non ignored)"
cargo test -p ddai-planner -p ddai-nav $F 2>&1 | grep -E "^test result|FAILED|failed"
echo "== physics integration suites incl. ignored (live Oracle A, Oracle B full corpus)"
for t in parity_bulk parity_fixtures parity_move_box_drift parity_oracle_b parity_oracle_b_fixtures no_alloc world_no_alloc world_smoke; do
  echo "-- $t"; cargo test -p ddai-physics --release --locked --test $t -- --include-ignored 2>&1 | grep -E "^test result|FAILED|panicked|mismatch"
done
echo "== done $(date +%T)"
