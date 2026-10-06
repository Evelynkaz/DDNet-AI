#!/usr/bin/env python3
"""Task 3.10 (E-021): the held-block metric of arena runs, and the paired comparison of two arms.

A game is a *held-block win* when the focal player won it with its own credited block (`result == W`, `credited`) and the victim was out
(frozen or dead) on every tick of the `after_ticks` window that follows (`held_block`, needs a run with `after_ticks = 250`); a *held-block
loss* is a lost game in which the focal player itself stayed out for the whole window. For every condition it prints

* `W:L` with credited wins, held-block wins and held-block losses (Wilson 95%), and the held-block share of the decided games
  `held W / (held W + held L)`;
* `held W / all games` (the held analogue of the D-059 credited rate);
* the **paired** comparison of two arms on the same games (same seeds and layouts): exact McNemar on the indicator "held-block win",
  and a sign test on the per-game net `held win - held loss` over the games where the arms differ.

  heldcompare.py DIR                       # one arm, every condition
  heldcompare.py A_DIR B_DIR [--pair a-stem=b-stem ...] [--label-a base --label-b new] [--pool]

`A_DIR` and `B_DIR` may be the same directory (two arms of one run: pair their conditions with `--pair`).
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e017"))
from stats import mcnemar_exact, pct, wilson  # noqa: E402


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        games = {}
        for line in open(f):
            if line.strip():
                v = json.loads(line)
                games[v["game"]] = v
        out[os.path.basename(f)[:-6]] = games
    return out


def hw(g):
    return g["result"] == "W" and g["credited"] and g.get("held_block", False)


def hl(g):
    return g["result"] == "L" and g.get("held_block", False)


def net(g):
    return int(hw(g)) - int(hl(g))


def sign_test(b, c):
    return mcnemar_exact(b, c)


def one(name, games):
    n = len(games)
    w = sum(g["result"] == "W" for g in games.values())
    l = sum(g["result"] == "L" for g in games.values())
    cw = sum(g["result"] == "W" and g["credited"] for g in games.values())
    h_w = sum(hw(g) for g in games.values())
    h_l = sum(hl(g) for g in games.values())
    return n, w, l, cw, h_w, h_l


def row(name, t):
    n, w, l, cw, h_w, h_l = t
    dec = h_w + h_l
    return (f"| {name} | {n} | {w}:{l} | {pct(cw, n)} | {h_w}:{h_l} | {pct(h_w, dec)} | {pct(h_w, n)} | {pct(h_w, w) if w else '-'} |")


HEAD = ("| condition | games | W:L | credited / all | held W:L | held W / held decided | held W / all | held W / W |\n"
        "|---|---:|---:|---|---:|---|---|---|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--pair", action="append", default=[], help="a-stem=b-stem (default: equal file names)")
    ap.add_argument("--pool", action="store_true", help="also print the pooled row over all pairs")
    args = ap.parse_args()
    a = load(args.a)
    if args.b is None:
        print(HEAD)
        tot = [0] * 6
        for k, g in a.items():
            t = one(k, g)
            print(row(k, t))
            tot = [x + y for x, y in zip(tot, t)]
        if len(a) > 1:
            print(row("**all**", tuple(tot)))
        return
    b = load(args.b)
    pairs = [tuple(p.split("=", 1)) for p in args.pair] or [(k, k) for k in a if k in b]
    print(f"{args.label_a}:")
    print(HEAD)
    ta = [0] * 6
    tb = [0] * 6
    rows_b = []
    pooled = defaultdict(int)
    for ka, kb in pairs:
        ga, gb = a[ka], b[kb]
        idx = sorted(set(ga) & set(gb))
        sa = one(ka, {i: ga[i] for i in idx})
        sb = one(kb, {i: gb[i] for i in idx})
        print(row(ka, sa))
        rows_b.append(row(kb, sb))
        ta = [x + y for x, y in zip(ta, sa)]
        tb = [x + y for x, y in zip(tb, sb)]
        for k, f in (("hw_only_a", lambda i: hw(ga[i]) and not hw(gb[i])), ("hw_only_b", lambda i: hw(gb[i]) and not hw(ga[i])),
                     ("cw_only_a", lambda i: ga[i]["result"] == "W" and ga[i]["credited"] and not (gb[i]["result"] == "W" and gb[i]["credited"])),
                     ("cw_only_b", lambda i: gb[i]["result"] == "W" and gb[i]["credited"] and not (ga[i]["result"] == "W" and ga[i]["credited"])),
                     ("net_up", lambda i: net(gb[i]) > net(ga[i])), ("net_down", lambda i: net(gb[i]) < net(ga[i]))):
            pooled[(ka, k)] = sum(f(i) for i in idx)
    if args.pool and len(pairs) > 1:
        print(row("**all**", tuple(ta)))
    print(f"\n{args.label_b}:")
    print(HEAD)
    print("\n".join(rows_b))
    if args.pool and len(pairs) > 1:
        print(row("**all**", tuple(tb)))
    print("\npaired (same games):")
    print(f"| condition | held W only {args.label_a} | held W only {args.label_b} | McNemar p (held W) | credited only {args.label_a} | credited only {args.label_b} | McNemar p (credited) | net up | net down | sign p (net) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot = defaultdict(int)
    for ka, kb in pairs:
        g = {k: pooled[(ka, k)] for k in ("hw_only_a", "hw_only_b", "cw_only_a", "cw_only_b", "net_up", "net_down")}
        print(f"| {ka} | {g['hw_only_a']} | {g['hw_only_b']} | {mcnemar_exact(g['hw_only_a'], g['hw_only_b']):.4f} | {g['cw_only_a']} | {g['cw_only_b']} | "
              f"{mcnemar_exact(g['cw_only_a'], g['cw_only_b']):.4f} | {g['net_up']} | {g['net_down']} | {sign_test(g['net_up'], g['net_down']):.4f} |")
        for k, v in g.items():
            tot[k] += v
    if args.pool and len(pairs) > 1:
        g = tot
        print(f"| **all** | {g['hw_only_a']} | {g['hw_only_b']} | {mcnemar_exact(g['hw_only_a'], g['hw_only_b']):.4f} | {g['cw_only_a']} | {g['cw_only_b']} | "
              f"{mcnemar_exact(g['cw_only_a'], g['cw_only_b']):.4f} | {g['net_up']} | {g['net_down']} | {sign_test(g['net_up'], g['net_down']):.4f} |")


if __name__ == "__main__":
    main()
