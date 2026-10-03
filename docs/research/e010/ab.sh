#!/bin/bash
# usage: ab.sh BASE_BIN NEW_BIN [rounds] ; alternates, prints tee/min per round
B=$1; N=$2; R=${3:-3}
cd ~/aiddnet/wt/task-3.6
for i in $(seq 1 $R); do
  for w in base new; do
    bin=$B; [ $w = new ] && bin=$N
    echo "round $i $w load=$(cut -d' ' -f1 /proc/loadavg)"
    DDAI_CAL_REPS=3 DDAI_SPEED_DECISIONS=${DEC:-400} $bin --ignored --nocapture calibration_report 2>&1 | grep -E "^\| (2|4) "
  done
done
