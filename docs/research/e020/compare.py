#!/usr/bin/env python3
"""E-020: report of `ddnet-ai arena run` directories in E-019's style -- credited wins (D-059) AND W:L:D:T with the share of decided
games W/(W+L), Wilson 95% intervals, the exact sign test of that share against 1/2, and, against a baseline run of the same games,
the paired exact McNemar tests (credited wins; and decided games: games both arms decided, discordant W/L).

  compare.py [--base DIR --base-label main] ARM_LABEL=DIR[@substring] [ARM_LABEL=DIR[@substring] ...] [--halls]

Each DIR holds one `*.jsonl` per hall (`<hall>-....jsonl`, the hall = the file name's `clb-left`/`clb-right`/`chillblock5` prefix); `@substring` picks one arm out of a directory several arms share (E-020's screening and fresh configs name their arms in the condition). Games pair by
(hall, game index) -- the same seed and layout in every arm of a config family. The report is a markdown table per arm; `--halls`
adds the per-hall rows.
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e017"))
from stats import mcnemar_exact, wilson  # noqa: E402


HALLS = ("clb-left", "clb-right", "chillblock5")


def hall_of(path):
    stem = os.path.basename(path)[:-6]
    for h in HALLS:
        if stem.startswith(h):
            return h
    return stem.split("-hybrid")[0]


def load(spec):
    """`DIR` or `DIR@substring`: only the files whose name contains the substring (several arms share a directory)."""
    d, _, sub = spec.partition("@")
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        if sub and sub not in os.path.basename(f):
            continue
        games = out.setdefault(hall_of(f), {})
        for line in open(f):
            if line.strip():
                v = json.loads(line)
                games[v["game"]] = (v["result"], bool(v["credited"]), bool(v.get("held", False)))
    return out


def sign_p(w, l):
    return mcnemar_exact(w, l)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:.1f}% [{100*lo:.1f}; {100*hi:.1f}]" if n else "-"


def counts(games):
    c = {"W": 0, "L": 0, "D": 0, "T": 0}
    cred = 0
    for r, cr, _h in games.values():
        c[r] += 1
        cred += r == "W" and cr
    return c, cred


def held_wins(games):
    """Wins whose block was held (D-059 `held`: the victim still out `after_ticks` later and the winner never out)."""
    return sum(1 for r, cr, h in games.values() if r == "W" and cr and h)


def row(label, games, base=None):
    c, cred = counts(games)
    n = len(games)
    dec = c["W"] + c["L"]
    hw = held_wins(games)
    cells = [label, str(n), f"{c['W']}:{c['L']}:{c['D']}:{c['T']}", f"{cred}/{n} = {fmt(cred, n)}",
             f"{fmt(c['W'], dec)}" if dec else "-", f"{sign_p(c['W'], c['L']):.2g}",
             f"{hw}/{n} = {fmt(hw, n)}", f"{hw}:{c['L']}", fmt(hw, hw + c["L"]) if hw + c["L"] else "-",
             f"{fmt(hw, c['W'])}" if c["W"] else "-"]
    if base is not None:
        idx = sorted(set(games) & set(base))
        win = lambda g: g[0] == "W" and g[1]
        oa = sum(win(base[i]) and not win(games[i]) for i in idx)
        ob = sum(win(games[i]) and not win(base[i]) for i in idx)
        held = lambda g: g[0] == "W" and g[1] and g[2]
        ha = sum(held(base[i]) and not held(games[i]) for i in idx)
        hb = sum(held(games[i]) and not held(base[i]) for i in idx)
        both = [i for i in idx if base[i][0] in "WL" and games[i][0] in "WL"]
        da = sum(base[i][0] == "W" and games[i][0] == "L" for i in both)
        db = sum(base[i][0] == "L" and games[i][0] == "W" for i in both)
        cells += [str(oa), str(ob), f"{mcnemar_exact(oa, ob):.3g}", f"{da}/{db}", f"{mcnemar_exact(da, db):.3g}",
                  f"{ha}/{hb}", f"{mcnemar_exact(ha, hb):.3g}"]
    return "| " + " | ".join(cells) + " |"


def merge(halls):
    out = {}
    for h, games in halls.items():
        for g, v in games.items():
            out[(h, g)] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--base-label", default="main")
    ap.add_argument("--halls", action="store_true")
    ap.add_argument("arms", nargs="+", help="LABEL=DIR[@substring]")
    a = ap.parse_args()
    base_halls = load(a.base) if a.base else None
    head = "| arm | games | W:L:D:T | credited wins [Wilson] | decided share W/(W+L) [Wilson] | sign p | held wins (of all games) | held W:L | held share H/(H+L) | held / wins |"
    sep = "|---|---:|---|---|---|---:|---|---|---|---|"
    if base_halls:
        head += f" only {a.base_label} | only arm | McNemar p (credited) | decided W→L / L→W | sign p (decided, paired) | only base held / only arm held | McNemar p (held) |"
        sep += "---:|---:|---:|---|---:|---|---:|"
    print(head)
    print(sep)
    if base_halls:
        print(row(a.base_label, merge(base_halls)))
    for spec in a.arms:
        label, d = spec.split("=", 1)
        halls = load(d)
        print(row(label, merge(halls), merge(base_halls) if base_halls else None))
        if a.halls:
            for h in sorted(halls):
                b = base_halls.get(h) if base_halls else None
                print(row(f"  {label} / {h}", {(h, g): v for g, v in halls[h].items()},
                          {(h, g): v for g, v in b.items()} if b else None))


main()
