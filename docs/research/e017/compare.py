#!/usr/bin/env python3
"""E-017: paired comparison of two arena runs (same config seeds): credited wins (D-059) with Wilson intervals and the
exact McNemar test of the paired credited-win indicator.

  compare.py A_DIR B_DIR [--label-a main --label-b new] [--map "A condition substring=B condition substring"]

The two directories hold `*.jsonl` of `ddnet-ai arena run` (one line per game, `condition`, `game`, `result`,
`credited`). Conditions are paired by file name unless `--pair` lists `a-file-stem=b-file-stem` pairs; games are paired by
their index. Prints one markdown table.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import mcnemar_exact, pct, wilson  # noqa: E402


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        games = {}
        for line in open(f):
            if line.strip():
                v = json.loads(line)
                games[v["game"]] = (v["result"], bool(v["credited"]), v["a_self_freezes"], v.get("blocks_by_a", 0), v["end_tick"])
        out[os.path.basename(f)[:-6]] = games
    return out


def win(g):
    return g[0] == "W" and g[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--pair", action="append", default=[], help="a-stem=b-stem (default: equal file names)")
    ap.add_argument("--pool", action="store_true", help="also print the pooled row over all pairs")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    pairs = [tuple(p.split("=", 1)) for p in args.pair] or [(k, k) for k in a if k in b]
    print(f"| condition | n | {args.label_a} credited | {args.label_b} credited | {args.label_a} only | {args.label_b} only | McNemar p |")
    print("|---|---:|---|---|---:|---:|---:|")
    tot = defaultdict(int)
    for ka, kb in pairs:
        ga, gb = a[ka], b[kb]
        idx = sorted(set(ga) & set(gb))
        n = len(idx)
        wa = sum(win(ga[i]) for i in idx)
        wb = sum(win(gb[i]) for i in idx)
        only_a = sum(win(ga[i]) and not win(gb[i]) for i in idx)
        only_b = sum(win(gb[i]) and not win(ga[i]) for i in idx)
        p = mcnemar_exact(only_a, only_b)
        print(f"| {ka} | {n} | {pct(wa, n)} | {pct(wb, n)} | {only_a} | {only_b} | {p:.4f} |")
        for k, v in (("n", n), ("wa", wa), ("wb", wb), ("oa", only_a), ("ob", only_b)):
            tot[k] += v
    if args.pool and len(pairs) > 1:
        p = mcnemar_exact(tot["oa"], tot["ob"])
        print(f"| **all** | {tot['n']} | {pct(tot['wa'], tot['n'])} | {pct(tot['wb'], tot['n'])} | {tot['oa']} | {tot['ob']} | {p:.4f} |")


main()
