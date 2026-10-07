#!/usr/bin/env bash
# E-031 (task 8.7): one DAgger round of a readout arm. The fly of <bundle> plays the post-freeze windows of the training starts of the bank and the planner
# labels every state (`train es collect`, round 100 = the loop's DAgger round, sampling weight 6 in the BC configs); then the arm's BC is run again from
# its initial bundle on the old data plus every store of the arm.
#   usage: dagger_round.sh <binary> <arm> <bundle played> <store>      (the store is created or extended)
set -eu
BIN=$1; ARM=$2; BUNDLE=$3; STORE=$4
cd "$(dirname "$0")/../.."
"$BIN" train es collect --bank ~/aiddnet/data/runs/E-022/banks/bank-v2.bank --out "$STORE" --actor "fly:$BUNDLE" --beta 0 \
  --arenas clb-left,pit,platform --escapable-only --round 100 --window 250 --burn-in 50 --threads 3 \
  > ~/aiddnet/data/runs/E-031/collect-$ARM.log 2>&1
tail -1 ~/aiddnet/data/runs/E-031/collect-$ARM.log
