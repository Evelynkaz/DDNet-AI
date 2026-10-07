#!/usr/bin/env python3
"""E-031 (task 8.7): the last `eval` record of each BC run's metrics.jsonl: hook-head AUROC and NLL per latch state (press = key up, hold = key down)
on the teacher-val set (all decisions of the held-out games) and the hold-out halls, and the per-run thresholds.
  offline_table.py <run dir> [<run dir> ...]
"""
import json
import sys

for run in sys.argv[1:]:
    evals, th = {}, None
    for line in open(f"{run}/metrics.jsonl"):
        d = json.loads(line)
        if d["kind"] == "eval":
            evals[d.get("set")] = d["report"]
        if d["kind"] == "thresholds":
            th = d
    print(f"\n{run.rstrip('/').split('/')[-1]}  (hook threshold {th['hook'] if th else '-':.3f})")
    for name, r in evals.items():
        h = r["hook_by_latch"]
        print(
            f"  {name:34s} press AUROC {h['released']['auroc']:.3f} (NLL {h['released']['nll']:.3f}, n {h['released']['n']}) | "
            f"hold AUROC {h['held']['auroc']:.3f} (NLL {h['held']['nll']:.3f}, n {h['held']['n']}) | "
            f"press rate label/model {h['press_label_rate']:.2f}/{h['press_pred_rate']:.2f}"
        )
