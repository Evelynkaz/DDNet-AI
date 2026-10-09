#!/usr/bin/env bash
# E-035 (task 8.8): the paired comparison lines of every arm against the s2 fly (the pre-registered reference) and against the equal-budget rate fly from scratch.
#   usage: report.sh <binary> <final dir> <arm> ...
BIN=$1; DIR=$2; shift 2
cd "$(dirname "$0")/../.."
for ref in s2 rate; do
  echo "######## reference: $ref"
  tools/e035/summary.sh "$BIN" "$DIR" "$ref" "$@"
done
