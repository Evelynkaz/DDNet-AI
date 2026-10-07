#!/usr/bin/env python3
"""E-025: report of `ddnet-ai arena run` directories of the duel experiments (task 3.13): credited wins (D-059) with Wilson 95% intervals,
W:L:D:T, the decided share W/(W+L) with its exact sign test, held wins (credited and the block still held `after_ticks` later), and, against
a base arm on the same games, the paired exact McNemar tests (credited wins, held wins, decided games).

  compare.py DIR[+OFFSET] [DIR[+OFFSET] ...] [--base ARM] [--arenas] [--only ARM,ARM] [--flip]

Conditions are named `<arena>: <arm> vs <opponent>` (the JSONL `condition` field). Games pair by (arena, game index), the same seed and layout
in every arm. Several DIRs are merged (one config may be split over several runs). `--flip`: our bot sat in slot 1 (the competitor held the
WB spot as slot 0): our wins are the focal player's losses, so only W:L and the decided share are meaningful there.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e017"))
from stats import mcnemar_exact, wilson  # noqa: E402


def load(dirs, flip=False, limit=None, keep=None):
    """{arm: {arena: {game: (result, credited, held)}}}"""
    out = {}
    for spec in dirs:
        # `DIR+N`: the games of DIR started at base seed (first seed + N), so its game g is game g + N of the baseline run
        d, _, off = spec.partition("+")
        off = int(off or 0)
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            for line in open(f):
                if not line.strip():
                    continue
                v = json.loads(line)
                v["game"] += off
                if limit is not None and v["game"] >= limit:
                    continue
                arena, rest = v["condition"].split(": ", 1)
                if keep and arena not in keep:
                    continue
                arm = rest.split(" vs ")[0]
                r = v["result"]
                if flip:
                    r = {"W": "L", "L": "W"}.get(r, r)
                out.setdefault(arm, {}).setdefault(arena, {})[v["game"]] = (r, bool(v["credited"]) and not flip, bool(v.get("held", False)) and not flip)
    return out


def fmt(k, n):
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{100*k/n:.1f}% [{100*lo:.1f}; {100*hi:.1f}]"


def pool(arms_arena, arenas=None):
    return {(a, g): v for a, games in arms_arena.items() if arenas is None or a in arenas for g, v in games.items()}


def row(label, games, base=None, flip=False):
    c = {"W": 0, "L": 0, "D": 0, "T": 0}
    cred = hw = 0
    for r, cr, h in games.values():
        c[r] += 1
        cred += r == "W" and cr
        hw += r == "W" and cr and h
    n = len(games)
    dec = c["W"] + c["L"]
    cells = [label, str(n), f"{c['W']}:{c['L']}:{c['D']}:{c['T']}"]
    cells.append("-" if flip else f"{cred} = {fmt(cred, n)}")
    cells.append(f"{fmt(c['W'], dec)}")
    cells.append(f"{mcnemar_exact(c['W'], c['L']):.2g}")
    cells.append("-" if flip else f"{hw} = {fmt(hw, n)}")
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
        cells += [f"{ob - oa:+d} ({ob}/{oa}) p={mcnemar_exact(oa, ob):.2g}", f"{hb - ha:+d} p={mcnemar_exact(ha, hb):.2g}",
                  f"{db}/{da} p={mcnemar_exact(da, db):.2g}"]
    return "| " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--base", default="main")
    ap.add_argument("--arenas", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--keep", help="comma-separated arenas to keep (default: all)")
    ap.add_argument("--limit", type=int, help="only games with index < LIMIT (the baseline run has more games than the arms)")
    ap.add_argument("--first", help="comma-separated arms to list first after the base (default: alphabetical)")
    a = ap.parse_args()
    data = load(a.dirs, a.flip, a.limit, a.keep.split(",") if a.keep else None)
    names = [x for x in data if not a.only or x in a.only.split(",") or x == a.base]
    names.sort(key=lambda x: (x != a.base, x))
    base = data.get(a.base)
    head = "| arm | games | W:L:D:T | credited [Wilson] | decided W/(W+L) [Wilson] | sign p | held wins (credited, of all games) |"
    sep = "|---|---:|---|---|---|---:|---|"
    if base:
        head += f" credited vs {a.base} (only arm/only base), McNemar | held vs {a.base} | decided L→W / W→L |"
        sep += "---|---|---|"
    print(head)
    print(sep)
    for arm in names:
        arenas = data[arm]
        print(row(arm, pool(arenas), pool(base) if base and arm != a.base else None, a.flip))
        if a.arenas:
            for ar in sorted(arenas):
                b = {(ar, g): v for g, v in base.get(ar, {}).items()} if base and arm != a.base else None
                print(row(f"  {arm} / {ar}", {(ar, g): v for g, v in arenas[ar].items()}, b, a.flip))


main()
