#!/usr/bin/env python3
"""E-031 (task 8.7): one table of the closed-loop evaluation of the arms (files of `eval.sh`): V held, own freeze on V and on all starts, held (all),
first freeze (credited) on the train halls and on the holdout hall; and, from the hook-eval files, the hook on V starts (start rate at idle, opening key, press within 8 ticks).
  table.py <final dir> name ...
"""
import json
import sys

R = sys.argv[1].rstrip("/") + "/"


def pct(d):
    return f"{100 * d['p']:.1f}% ({d['k']}/{d['n']})"


print("| arm | V held (train-val halls) | V held | V own freeze | held (all) | own freeze (all) | first freeze holdout | first freeze train | V: start rate / opening key / press<=8t |")
print("|---|---|---|---|---|---|---|---|---|")
for a in sys.argv[2:]:
    _, p = json.load(open(R + a + ".json"))
    h = p["holdout_starts"]
    tv = p["train_starts"]
    # own freeze on V-escapable starts: the items are in the file
    vi = h["victim_escape_items"]
    sf = [x for x, m in zip(h["self_freeze_items"], vi) if m]
    own_v = f"{100 * sum(sf) / len(sf):.1f}% ({sum(sf)}/{len(sf)})"
    g = p["holdout_games"]["credited"]
    gt = p["train_games"]["credited"]
    try:
        _, sets = json.load(open(R + "hook-" + a + ".json"))
        recs = dict(sets)["holdout"]
        v = [r for r in recs if r["class"] == "V"]
        idle = [x for r in v for x in r["decisions"] if x["state"] == 0]
        rate = 100 * sum(x["hook"] for x in idle) / max(1, len(idle))
        op = [x for r in v for x in r["decisions"][:4]]
        okey = 100 * sum(x["hook"] for x in op) / max(1, len(op))
        first = [next((x["tick"] for x in r["decisions"] if x["hook"]), None) for r in v]
        p8 = 100 * sum(1 for f in first if f is not None and f <= 8) / len(v)
        hk = f"{rate:.1f}% / {okey:.1f}% / {p8:.1f}%"
    except Exception as e:  # noqa: BLE001
        hk = f"({e})"
    print(f"| {a} | {pct(tv['held_victim_escapable'])} | {pct(h['held_victim_escapable'])} | {own_v} | {pct(h['held'])} | {pct(h['self_freeze'])} | {pct(g)} | {pct(gt)} | {hk} |")
