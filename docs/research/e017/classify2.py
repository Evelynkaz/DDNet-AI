#!/usr/bin/env python3
"""E-017: where does the hybrid lose? The decision tree of spec 3.7b, on the full-pool truth data.

Input: `truth.jsonl` of `hybrid_losses --mode truth` (the v1 format: per decision in the window the hybrid's pool with its scores,
the fixed planner's plan with its score under the hybrid's evaluator, what every plan does in the true world against the opponent's
actual inputs (`pool_truth`, `truth_h`, `truth_t`), and the hybrid's evaluator told those inputs (`oracle`)).

A decision *can be repaired* when the pick ends with us out (frozen or dead) within its 27-tick plan in the true world while some other
plan -- of the hybrid's pool or the planner's -- does not. The *decisive* decision of a lost game is the last repairable one in the
window (the last chance). Its class:
  generation    only the planner's plan avoids the freeze: the plan was not in the pool;
  selection     some pool plan avoids it, and
     choice       the hybrid's own score ranks such a plan above the pick (stage 1 score) -- the robust choice or its hysteresis
                  passed it over;
     opponent     it does not, but the same evaluator told the opponent's actual inputs would choose a plan that avoids the freeze
                  (the opponent model is the gap);
     evaluator    not even then: the evaluator (shaping, horizon) mis-ranks it;
  doomed        no decision of the window can be repaired by any plan in hand (the position was lost earlier or by a longer horizon).
A loss without a freeze in the plans of the window (a timeout or draw) is reported apart.
Usage: classify2.py [--window TICKS] label=truth.jsonl ...
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import pct  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("truth", nargs="+")
ap.add_argument("--window", type=int, default=60)
ap.add_argument("--json")
args = ap.parse_args()


def out_me(t):
    """A truth array [me_out, enemy_out, touch, min_gap, end_gap_enemy, end_dist] (or the dict of truth_h/truth_t): are we out?"""
    if t is None:
        return None
    if isinstance(t, dict):
        return t["me_out"] >= 0
    return t[0] >= 0


def me_tick(t):
    return t["me_out"] if isinstance(t, dict) else t[0]


def enemy_tick(t):
    return t["enemy_out"] if isinstance(t, dict) else t[1]


def rank(t):
    me, en = me_tick(t), enemy_tick(t)
    if en >= 0 and (me < 0 or en < me):
        return 1
    if me >= 0 and (en < 0 or me < en):
        return -1
    return 0


def analyse(g):
    """The decisive decision of one game, its class and the facts about it."""
    best = None
    n_rep = 0
    for d in g["decisions"]:
        if g["end_tick"] - d["tick"] > args.window:
            continue
        h = d["hybrid"]
        pick = h["pick"]
        pool, pt = h["pool"], d["pool_truth"]
        th = d["truth_h"]
        if pick is None or th is None or len(pt) != len(pool):
            continue
        if not out_me(th):
            continue
        alt_pool = [i for i in range(len(pool)) if i != pick and not out_me(pt[i])]
        tt = d["truth_t"]
        alt_teacher = tt is not None and not out_me(tt)
        if not alt_pool and not alt_teacher:
            continue
        n_rep += 1
        best = (d, alt_pool, alt_teacher)  # decisions are in time order: the last one wins
    if best is None:
        return {"class": "doomed", "repairable_decisions": 0}
    d, alt_pool, alt_teacher = best
    h = d["hybrid"]
    pool = h["pool"]
    pick = h["pick"]
    facts = {"repairable_decisions": n_rep, "to_end": g["end_tick"] - d["tick"], "danger": h["danger"], "tick": d["tick"]}
    if not alt_pool:
        facts["class"] = "generation"
        return facts
    pick_cheap = pool[pick]["cheap"]
    higher = [i for i in alt_pool if pool[i]["cheap"] is not None and pick_cheap is not None and pool[i]["cheap"] > pick_cheap + 1e-9]
    if higher:
        facts["class"] = "selection/choice"
        return facts
    oracle = d["oracle"]
    if oracle and len(oracle) >= len(pool):
        pool_scores = [(oracle[i], i) for i in range(len(pool)) if oracle[i] is not None]
        if pool_scores:
            top = max(pool_scores)[1]
            if not out_me(d["pool_truth"][top]):
                facts["class"] = "selection/opponent"
                return facts
    facts["class"] = "selection/evaluator"
    return facts


def situation(d_facts, g):
    dg = d_facts.get("danger", "")
    if "hooked" in dg:
        return "hooked by the opponent"
    if "near_freeze" in dg:
        return "edge / freeze near"
    return "other"


CLASSES = ["generation", "selection/choice", "selection/opponent", "selection/evaluator", "doomed"]


def report(label, games):
    by = defaultdict(list)
    for g in games:
        by[g["condition"]].append(g)
    print(f"\n# {label}: {len(games)} games (window {args.window} ticks)")
    print(f"replay hash mismatches: {sum(1 for g in games if not g['hash_ok'])}/{len(games)}")
    result = {}
    for cond, gs in list(sorted(by.items())) + [("ALL HALLS", games)]:
        c = Counter()
        sit = Counter()
        rep = []
        for g in gs:
            f = analyse(g)
            c[f["class"]] += 1
            rep.append(f["repairable_decisions"])
            if f["class"] != "doomed":
                sit[(f["class"], situation(f, g))] += 1
        n = len(gs)
        print(f"\n## {cond}  (games {n}; results {dict(Counter(g['result'] for g in gs))})")
        for k in CLASSES:
            print(f"  {k:22s} {pct(c[k], n)}")
        repairable = n - c["doomed"]
        print(f"  repairable at some decision of the window: {pct(repairable, n)}; mean repairable decisions per game {sum(rep)/max(1,n):.1f}")
        if sit:
            print("  situations of the decisive decision (class x danger flag):")
            for (k, s), v in sorted(sit.items(), key=lambda x: (-x[1], x[0])):
                print(f"    {k:22s} {s:24s} {v}")
        result[cond] = dict(c)
    return result


out = {}
for spec in args.truth:
    label, _, path = spec.rpartition("=")
    label = label or path
    games = [json.loads(l) for l in open(path) if l.strip()]
    out[label] = report(label, games)
if args.json:
    json.dump(out, open(args.json, "w"), indent=1)
