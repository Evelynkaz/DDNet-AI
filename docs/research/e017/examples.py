#!/usr/bin/env python3
"""E-017: annotated example windows (local files, never committed): for the first lost games of each decisive-decision class, the last
decisions before the loss with what the hybrid picked, what the planner would have played, what the true world says of both, and the
plan of the hybrid's pool that the evaluator told the opponent's actual inputs would have chosen.

  examples.py truth.jsonl OUT_DIR [--per-class 2]
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("truth")
ap.add_argument("out")
ap.add_argument("--per-class", type=int, default=2)
ap.add_argument("--window", type=int, default=24, help="ticks before the end shown")
args = ap.parse_args()

# Reuse the classifier's `analyse`.
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "classify2.py")).read()
head = src[: src.index("def report(")]
ns = {"__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "classify2.py")}
exec(compile(head.replace("args = ap.parse_args()", 'args = ap.parse_args(["x", "--window", "60"])'), "classify2_head", "exec"), ns)
analyse = ns["analyse"]


def act(a):
    return None if a is None else {"dir": a[0], "jump": a[1], "hook": a[2], "fire": a[3], "aim": [a[4], a[5]]}


def step_str(p):
    return "".join(
        ("L" if s[0] < 0 else "R" if s[0] > 0 else "-") + ("j" if s[1] else "") + ("H" if s[2] else "") + ("F" if s[3] else "") + " "
        for s in p
    )


os.makedirs(args.out, exist_ok=True)
seen = {}
for line in open(args.truth):
    g = json.loads(line)
    f = analyse(g)
    k = f["class"]
    if seen.get(k, 0) >= args.per_class:
        continue
    seen[k] = seen.get(k, 0) + 1
    name = f"{g['condition'].split(':')[0].replace(' ', '_')}-g{g['game']}-{k.replace('/', '_')}"
    rows = []
    for d in g["decisions"]:
        if g["end_tick"] - d["tick"] > args.window:
            continue
        h = d["hybrid"]
        pick = h["pick"]
        pool = h["pool"]
        pt = d["pool_truth"]
        o = d.get("oracle") or []
        best = None
        if o and len(o) >= len(pool):
            sc = [(o[i], i) for i in range(len(pool)) if o[i] is not None]
            if sc:
                best = max(sc)[1]
        rows.append(
            {
                "tick": d["tick"],
                "me": [round(d["me"]["x"]), round(d["me"]["y"]), d["me"]["frozen"], d["me"]["hooked"]],
                "victim": [round(d["victim"]["x"]), round(d["victim"]["y"]), d["victim"]["frozen"], d["victim"]["hooked"]],
                "danger": h["danger"],
                "hybrid": {
                    "action": act(d["hyb_action"]),
                    "source": h["chosen_src"],
                    "plan": step_str(h["chosen_plan"]),
                    "truth_me_out_tick": d["truth_h"]["me_out"] if d["truth_h"] else None,
                },
                "planner": {"action": act(d["tea_action"]), "plan": step_str(d["teacher"]["plan"] or []),
                            "truth_me_out_tick": d["truth_t"]["me_out"] if d["truth_t"] else None},
                "oracle_choice": None
                if best is None
                else {"pool_index": best, "source": pool[best]["src"], "plan": step_str(pool[best]["plan"]), "truth_me_out_tick": pt[best][0]},
                "opponent_actual": act(d["opp"]["actual"]) if d.get("opp") else None,
                "opponent_hold_model": act(d["opp"]["hold"]) if d.get("opp") else None,
                "opponent_planner_mirror": act(d["opp"]["mirror"]) if d.get("opp") else None,
            }
        )
    out = {
        "game": {"condition": g["condition"], "game": g["game"], "result": g["result"], "end_tick": g["end_tick"]},
        "class": k,
        "decisive_tick": f.get("tick"),
        "note": "plan strings: one cell per 3-tick step, L/R/- direction, j jump, H hook, F fire; truth_me_out_tick = tick of the plan at which we are frozen in the true world against the opponent's actual inputs (-1: never)",
        "decisions": rows,
    }
    p = os.path.join(args.out, name + ".json")
    json.dump(out, open(p, "w"), indent=1)
    print(p)
