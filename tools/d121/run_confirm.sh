#!/bin/bash
# Task 3.23 (E-038): the pre-registered confirmation runs of docs/research/duel-fixes-3.23.md §0, one after the other (3 threads, nice 15).
# usage: tools/d121/run_confirm.sh   (from the repository root, after `cargo build --release -p ddai-env -p ddnet-ai --bins --examples`)
set -u
E=${E:-$HOME/aiddnet/data/runs/E-038}
CLIPS=$HOME/aiddnet/data/scratch/pm-1008/cfclips
N="nice -n 15"
mkdir -p "$E/confirm"
# fix 1: the AFK box
$N target/release/examples/duel_stats --config configs/arena/d121-afk-confirm.toml --threads 3 --jsonl "$E/confirm/afk.jsonl" > "$E/confirm/afk.txt" 2>&1
# fix 3: the finishing scenarios
$N target/release/ddnet-ai arena scenarios --dir configs/scenarios-d121 --config configs/arena/d121-scenarios-confirm.toml --out "$E/confirm/scenarios" > "$E/confirm/scenarios.txt" 2>&1
# fix 2: the counterfactual from the 10 losses, fresh seeds 45001.., 32 seeds, starts at -40/-30/-20/-10 ticks
export PM_FREEZE="duel-loss-38377562.clip:38377390,duel-loss-38381654.clip:38381460,duel-loss-38382552.clip:38382296,duel-loss-38385524.clip:38385226,duel-loss-38386672.clip:38386304,duel-loss-38387718.clip:38387528,duel-loss-38388104.clip:38387934,duel-loss-38388810.clip:38388646,duel-loss-38389290.clip:38389114,duel-loss-38391512.clip:38391310"
export PM_DRAG=20 PM_SEED0=45001
for arm in "base:" "counter:counter,belief=0.8,protect" ; do
  name=${arm%%:*}; fixes=${arm#*:}
  if [ -n "$fixes" ]; then export PM_FIXES="$fixes"; else unset PM_FIXES; fi
  $N target/release/examples/pm_counterfactual --clips "$CLIPS" --cells "ol:2:4" --seeds 32 --offsets 40,30,20,10 --horizon 150 --threads 3 \
    --jsonl "$E/confirm/cf-$name.jsonl" > "$E/confirm/cf-$name.txt" 2>&1
done
unset PM_FIXES
# all arms against live-v2
for his in 0 1 2; do
  $N target/release/examples/duel_stats --config configs/arena/d121-confirm-2$his.toml --threads 3 --jsonl "$E/confirm/duel-2$his.jsonl" > "$E/confirm/duel-2$his.txt" 2>&1
done
echo done > "$E/confirm/DONE"
