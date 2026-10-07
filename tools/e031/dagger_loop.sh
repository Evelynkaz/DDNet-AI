#!/usr/bin/env bash
# E-031 (task 8.7): DAgger rounds of one readout arm, each = collect with the previous round's head -> BC of the readout again from the arm's initial bundle on the
# old data plus every round's labels -> closed-loop evaluation. usage: dagger_loop.sh <binary> <arm> <init bundle> <first played bundle> <first round> <last round>
# (round k plays the bundle of round k-1; round 1 plays <first played bundle>; the store ~/aiddnet/data/datasets/teacher/e031-<arm>-store accumulates).
set -eu
BIN=$1; ARM=$2; INIT=$3; PLAY=$4; FIRST=$5; LAST=$6
cd "$(dirname "$0")/../.."
R=~/aiddnet/data/runs/E-031
STORE=~/aiddnet/data/datasets/teacher/e031-$ARM-store
for k in $(seq "$FIRST" "$LAST"); do
  tools/e031/dagger_round.sh "$BIN" "$ARM-d$k" "$PLAY" "$STORE"
  tools/e031/gen_configs.sh "d$k-$ARM" "$INIT" "$STORE"
  tools/e031/bc_arms.sh "$BIN" "d$k-$ARM"
  cp "$R/ro-d$k-$ARM/checkpoints/final.bundle" "$R/bundles/d$k-$ARM.bundle"
  tools/e031/eval.sh "$BIN" "$R/final" "d$k-$ARM=$R/bundles/d$k-$ARM.bundle"
  PLAY=$R/bundles/d$k-$ARM.bundle
done
