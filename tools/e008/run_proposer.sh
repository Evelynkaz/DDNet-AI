#!/usr/bin/env bash
# Plays the conditions of the proposer comparison one at a time (each into its own output directory, so a killed cell is
# simply played again; a cell whose summary.json exists is skipped). The queue is a file with one condition name per line
# (the "<shape> | <proposer>" names of the config); the first line is taken when a cell starts, so the queue can be
# edited while it runs.
#   DDNET_AI=<merged build> THREADS=2 LOGTAG=prop tools/e008/run_proposer.sh <config.toml> <queue-file> [out-root]
# NB: `--filter` is a substring match: a name that is a prefix of another ("...-K1" and "...-K10") also plays the longer one
# into its directory; queue the longer name first or use unique names.
# Merge the cells with: tools/e008/proposer_tables.py <out-root>/*
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
cfg=$1; queue=$2
out=${3:-~/aiddnet/data/runs/E-008/proposer}
bin=${DDNET_AI:-./target/release/ddnet-ai}
threads=${THREADS:-2}
tag=${LOGTAG:-prop}
log=~/aiddnet/data/logs/8.2b
mkdir -p "$out"
while true; do
  name=$(grep -v '^\s*#' "$queue" 2>/dev/null | grep -v '^\s*$' | head -1 || true)
  [ -n "$name" ] || { echo "$(date +%T) queue empty" >> "$log/$tag.log"; break; }
  python3 tools/e008/pop_queue.py "$queue" "$name"
  slug=$(echo "$name" | tr -c 'A-Za-z0-9\n' '_')
  if [ ! -f "$out/$slug/summary.json" ]; then
    echo "$(date +%T) cell $name" >> "$log/$tag.log"
    $bin arena run --config "$cfg" ${GAMES:+--games $GAMES} --filter "$name" --out "$out/$slug" --threads "$threads" > "$log/prop-$slug.out" 2> "$log/prop-$slug.err" \
      || { echo "$(date +%T) FAILED $name" >> "$log/$tag.log"; continue; }
  fi
  echo "$(date +%T) done $name" >> "$log/$tag.log"
done
