#!/usr/bin/env python3
"""E-026: the arms of a `duel_stats --jsonl` file (or the `*.jsonl` of `ddnet-ai arena run`) against a baseline arm, game by game.

  compare.py FILE_OR_DIR [--base "substring of the baseline condition"]

Per arm: W:L:D:T, credited wins (Wilson), the share of decided games W/(W+L) with its sign-test p against 50%, and -- paired by game index with the
baseline arm (same seeds and layouts) -- the games only this arm won (credited), only the baseline won, and the exact McNemar p.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e017"))
from stats import mcnemar_exact, pct  # noqa: E402


def load(path):
    files = sorted(glob.glob(os.path.join(path, "*.jsonl"))) if os.path.isdir(path) else [path]
    arms = defaultdict(dict)
    for f in files:
        for line in open(f):
            if line.strip():
                try:
                    v = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a run that was stopped leaves a half-written last line
                arms[v["condition"]][v["game"]] = (v["result"], bool(v["credited"]), bool(v.get("held", False)))
    return arms


def win(g):
    return g[0] == "W" and g[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--base", default="base")
    args = ap.parse_args()
    arms = load(args.path)
    base_key = next((k for k in arms if args.base in k), None)
    print("| arm | W:L:D:T | credited wins | held wins (150 ticks) | decided W/(W+L) [sign-test p vs 50%] | only arm / only base (credited) | McNemar p |")
    print("|---|---|---|---|---|---|---|")
    for k, games in arms.items():
        c = defaultdict(int)
        for g in games.values():
            c[g[0]] += 1
        n = len(games)
        w, l = c["W"], c["L"]
        dec = f"{100*w/(w+l):.1f}% [{mcnemar_exact(w, l):.3g}]" if w + l else "-"
        pair = ""
        p = ""
        if base_key and k != base_key:
            bg = arms[base_key]
            idx = sorted(set(games) & set(bg))
            oa = sum(win(games[i]) and not win(bg[i]) for i in idx)
            ob = sum(win(bg[i]) and not win(games[i]) for i in idx)
            pair = f"{oa} / {ob}"
            p = f"{mcnemar_exact(oa, ob):.3g}"
        held = sum(g[0] == "W" and g[2] for g in games.values())
        print(f"| {k} | {w}:{l}:{c['D']}:{c['T']} | {pct(sum(win(g) for g in games.values()), n)} | {held} ({100*held/n:.1f}%) | {dec} | {pair} | {p} |")


if __name__ == "__main__":
    main()
