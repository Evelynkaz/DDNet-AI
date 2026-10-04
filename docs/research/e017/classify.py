#!/usr/bin/env python3
"""E-017: classify the decisive decisions of the hybrid's lost / uncredited games.

Input: `truth.jsonl` written by `cargo run --release -p ddai-env --example hybrid_losses -- --mode truth` (one line per
replayed game; per decision in the window: the hybrid's pool with scores, the fixed planner's plan, its score under the
hybrid's evaluator, and the true-world rollouts of both plans against the opponent's actual inputs).

Per decision, three questions (spec 3.7b):
  P  is the planner's plan in the hybrid's pool?          (plan match, see `same_plan`)
  S  does the hybrid's evaluator rank the planner's plan above the pick?   (robust value, then cheap score)
  T  does the true world say the planner's plan is better?  (outcome rank, then the soft margin)
and the classes:
  generation gap  - the plan is absent from the pool, the hybrid's evaluator would have preferred it, and the true world
                    agrees it is better;
  evaluation gap  - the true world says the planner's plan is better but the hybrid's evaluator ranks the pick at least as
                    high (the plan is in the pool, or it is absent and would not have won anyway);
  other           - the true world does not say the planner's plan is better (horizon, opponent model, ties, no better move).
A game is attributed to its decisive decision: the one in the window with the largest true-world advantage of the planner.
Usage: classify.py truth.jsonl [--window TICKS] [--match-steps K]
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from stats import pct  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("truth", nargs="+", help="truth.jsonl files, optionally as label=path")
ap.add_argument("--window", type=int, default=60, help="ticks before the end that count (default 60)")
ap.add_argument("--match-steps", type=int, default=3, help="plan steps compared for 'the same move'")
ap.add_argument("--aim-tol", type=float, default=0.30)
ap.add_argument("--soft", type=float, default=32.0, help="px margin of the soft truth comparison")
ap.add_argument("--json", help="write per-game classes here")
args = ap.parse_args()


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def rel_aim(aim, base):
    """Plan aim as an angle relative to the direction to the victim (absolute aims are 100 + angle)."""
    return wrap(aim - 100.0 - base) if aim > 50 else aim


def same_plan(a, b, base, k, tol):
    """The first k steps agree on dir/jump/hook/fire, and on the aim wherever it matters (hook or fire)."""
    for s in range(min(k, len(a), len(b))):
        x, y = a[s], b[s]
        if x[:4] != y[:4]:
            return False
        if x[2] or x[3]:
            if abs(wrap(rel_aim(x[4], base) - rel_aim(y[4], base))) > tol:
                return False
    return True


def rank(t):
    """Outcome of a rollout: +1 the victim went out first, -1 we did, 0 otherwise (both on one tick counts 0)."""
    if t is None:
        return None
    me, en = t["me_out"], t["enemy_out"]
    if en >= 0 and (me < 0 or en < me):
        return 1
    if me >= 0 and (en < 0 or me < en):
        return -1
    return 0


def soft(t):
    """A margin for rollouts with the same outcome: farther from a freeze ourselves, nearer for the victim (px)."""
    return min(t["min_gap"], 200.0) + (200.0 - min(t["end_gap_enemy"], 200.0))


def advantage(th, tx):
    """True-world advantage of plan `tx` over the pick `th` (>= 999: a better outcome; else the soft margin, px)."""
    rh, rx = rank(th), rank(tx)
    if rh is None or rx is None:
        return None
    if rx != rh:
        return (rx - rh) * 1000.0
    return soft(tx) - soft(th)


def better(adv):
    return adv is not None and (adv >= 999.0 or (-999.0 < adv and adv >= args.soft))


def worse(adv):
    return adv is not None and (adv <= -999.0 or (adv > -999.0 and adv <= -args.soft))


def chosen_values(h):
    pick = h["pick"]
    if pick is None:
        return None
    c = h["pool"][pick]
    sc = c["scores"]
    full = [s for s in sc if s is not None]
    if len(full) < max(1, h["combos"]):
        robust = c["cheap"]
    else:
        w = h["weights"][: h["combos"]]
        tot = sum(w)
        mean = sum(s * x for s, x in zip(sc, w)) / tot if tot > 0 else sum(sc[: h["combos"]]) / h["combos"]
        robust = h["lambda"] * min(sc[: h["combos"]]) + (1 - h["lambda"]) * mean
    return robust


def decision_facts(d, end_tick):
    h = d["hybrid"]
    me, vi = d["me"], d["victim"]
    base = math.atan2(vi["y"] - me["y"], vi["x"] - me["x"])
    tp = d["teacher"]["plan"]
    facts = {"tick": d["tick"], "to_end": end_tick - d["tick"]}
    if tp is None or h["pick"] is None:
        facts["valid"] = False
        return facts
    chosen = h["pool"][h["pick"]]["plan"]
    facts["valid"] = True
    facts["same_first"] = d["hyb_action"] == d["tea_action"]
    facts["same_plan_k"] = same_plan(chosen, tp, base, args.match_steps, args.aim_tol)
    facts["in_pool"] = any(same_plan(c["plan"], tp, base, args.match_steps, args.aim_tol) for c in h["pool"])
    facts["in_pool_full"] = any(same_plan(c["plan"], tp, base, len(tp), args.aim_tol) for c in h["pool"])
    ts = d["teacher"]["score"]
    cv = chosen_values(h)
    facts["eval_pref"] = None
    if ts is not None and cv is not None and ts["robust"] == ts["robust"]:
        facts["eval_pref"] = ts["robust"] > cv + 1e-9
        facts["eval_margin"] = ts["robust"] - cv
    th, tt = d.get("truth_h"), d.get("truth_t")
    rh, rt = rank(th), rank(tt)
    facts["rank_h"], facts["rank_t"] = rh, rt
    facts["adv"] = advantage(th, tt)
    facts["truth_better"] = better(facts["adv"])
    ar = d.get("alt_runner")
    ax = d.get("alt_random")
    facts["adv_runner"] = advantage(th, ar["truth"]) if ar else None
    facts["adv_random"] = advantage(th, ax["truth"]) if ax else None
    facts["danger"] = h["danger"]
    facts["src"] = h["chosen_src"]
    facts["me"], facts["victim"] = me, vi
    facts["dist"] = math.hypot(vi["x"] - me["x"], vi["y"] - me["y"])
    return facts


def dec_class(f):
    if not f.get("valid"):
        return "invalid"
    if f["same_plan_k"]:
        return "same"
    if not f["truth_better"]:
        return "other"
    if not f["in_pool"] and f["eval_pref"]:
        return "generation"
    return "evaluation"


def situation(f):
    me, vi, dg = f["me"], f["victim"], f["danger"]
    if vi["hooked"] == 0 or "hooked" in dg:
        return "hooked-by-enemy"
    if me["hooked"] == 1:
        return "hooking-enemy"
    if me["frozen"]:
        return "frozen"
    if "near_freeze" in dg:
        return "edge/freeze-near"
    if f["dist"] > 300:
        return "approach"
    return "close-fight"




def load_games(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def report(label, games):
    by_cond = defaultdict(list)
    for g in games:
        by_cond[g["condition"]].append(g)
    out_games = []
    print(f"\n# {label}: {len(games)} games, window {args.window} ticks, match {args.match_steps} steps")
    hash_bad = sum(1 for g in games if not g["hash_ok"])
    print(f"replay hash mismatches: {hash_bad}/{len(games)}")
    groups = list(sorted(by_cond.items())) + [("ALL HALLS", games)]
    for cond, gs in groups:
        print(f"\n## {cond}  (games: {len(gs)}; results {dict(Counter(g['result'] for g in gs))})")
        cls_games = Counter()
        sit_games = Counter()
        dec_cls = Counter()
        pool_stats = Counter()
        ctrl = Counter()
        cross = Counter()
        for g in gs:
            facts = [decision_facts(d, g["end_tick"]) for d in g["decisions"]]
            facts = [f for f in facts if f["to_end"] <= args.window]
            for f in facts:
                c = dec_class(f)
                dec_cls[c] += 1
                if f.get("valid"):
                    pool_stats["decisions"] += 1
                    pool_stats["same_first"] += f["same_first"]
                    pool_stats["in_pool"] += f["in_pool"]
                    pool_stats["in_pool_full"] += f["in_pool_full"]
                    pool_stats["eval_pref"] += bool(f["eval_pref"])
                    pool_stats["truth_better"] += f["truth_better"]
                    pool_stats["differs"] += not f["same_plan_k"]
                    if not f["same_plan_k"]:
                        cross[(f["in_pool"], bool(f["eval_pref"]), f["truth_better"])] += 1
                    for name, key in (("teacher", "adv"), ("runner", "adv_runner"), ("random", "adv_random")):
                        a = f[key]
                        if a is not None:
                            ctrl[name + "_n"] += 1
                            ctrl[name + "_better"] += better(a)
                            ctrl[name + "_worse"] += worse(a)
            valid = [f for f in facts if f.get("valid") and f["adv"] is not None]
            best = max(valid, key=lambda f: (f["adv"], -f["to_end"]), default=None)
            if best is None:
                gc, sit = "no-data", "-"
            elif best["truth_better"]:
                gc, sit = dec_class(best), situation(best)
            else:
                gc, sit = "no-better-move", situation(best)
            cls_games[gc] += 1
            sit_games[(gc, sit)] += 1
            out_games.append({"condition": cond, "game": g["game"], "class": gc, "situation": sit, "result": g["result"]})
        n = len(gs)
        print("decisions in the window:", dict(dec_cls))
        if pool_stats["decisions"]:
            d = pool_stats["decisions"]
            print(
                f"  planner plan differs from the pick: {pct(pool_stats['differs'], d)}; absent from the pool (k={args.match_steps}): "
                f"{pct(d - pool_stats['in_pool'], d)}; evaluator prefers the planner's: {pct(pool_stats['eval_pref'], d)}; "
                f"truth prefers it: {pct(pool_stats['truth_better'], d)}"
            )
            for name in ("teacher", "runner", "random"):
                k = ctrl[name + "_n"]
                if k:
                    print(
                        f"  control {name:8s}: truth-better {pct(ctrl[name + '_better'], k)}, truth-worse {pct(ctrl[name + '_worse'], k)}"
                    )
        if cross:
            tot = sum(cross.values())
            print(f"  cross-tab over the {tot} decisions where the plans differ (in pool / evaluator prefers planner / truth prefers planner):")
            for k in sorted(cross):
                print(f"    in_pool={int(k[0])} eval_pref={int(k[1])} truth_better={int(k[2])}: {pct(cross[k], tot)}")
        print("games by decisive-decision class:")
        for c in ["generation", "evaluation", "other", "no-better-move", "no-data"]:
            print(f"  {c:16s} {pct(cls_games[c], n)}")
        print("situations (class x situation):")
        for (c, s), k in sorted(sit_games.items(), key=lambda x: -x[1]):
            print(f"  {c:16s} {s:18s} {k}")
    return out_games


all_out = {}
for spec in args.truth:
    label, _, path = spec.rpartition("=")
    label = label or path
    all_out[label] = report(label, load_games(path))
if args.json:
    json.dump(all_out, open(args.json, "w"))
