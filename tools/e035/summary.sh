#!/usr/bin/env bash
# E-035: paired comparison of each arm against the reference eval file (default s2: ~/aiddnet/data/runs/E-031/final/s2.json): the lines of `train es compare` the go criterion is made of.
#   usage: summary.sh <binary> <final dir> <reference name> <arm> ...
BIN=$1; DIR=$2; REF=$3; shift 3
for a in "$@"; do
  echo "== $a vs $REF"
  "$BIN" train es compare "$DIR/$REF.json" "$DIR/$a.json" 2>&1 | grep -E "holdout starts: (held, victim-escapable|own freeze, victim-escapable|held, all|own freeze in window)|holdout games: first freeze|train games: first freeze|train-val starts: held, victim" | sed -E 's/\(only A [0-9]+, only B [0-9]+, /(/; s/ +A / A /; s/ +B / B /' | cut -c1-230
done
