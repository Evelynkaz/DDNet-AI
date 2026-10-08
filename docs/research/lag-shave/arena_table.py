#!/usr/bin/env python3
"""Task 3.16 (E-032): the arena cells of `duel_stats --jsonl` files, pooled over seed blocks and paired against a base cell.

  arena_table.py RUN_DIR

The files of RUN_DIR are `<name>.jsonl`; `Xb2-` .. `Xb5-<name>.jsonl` are the replicate blocks that started at seeds 13201 / 13401 / 13601 / 13801
(configs `e032-*-b2.toml` .. `-b5.toml`), every other file is block 13001. The base of a cell: for `B3` / `B2` cells the `B4` cell of the same name (same lag or margin: the budget alone); for the other fixed-lag
cells `B4 lag 2/<theirs>` (our lag 2 is what the bot has live), for the other coupled cells `B4 m9 vs <theirs>` (the adaptive margin of the sessions). A cell is pooled over the blocks that played it; the paired columns count, over the (block, game)
pairs both cells played, the credited wins only this cell has / only the base has, and the exact McNemar p. For the coupled cells (`lag_hist`) the
mean lag and the share of decisions at 2 ticks are printed from the pooled histograms.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e017"))
from stats import mcnemar_exact, pct  # noqa: E402


def load(d):
    cells = defaultdict(dict)  # condition -> {(block, game): (result, credited)}
    lag = defaultdict(lambda: defaultdict(int))
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        name = os.path.basename(f)
        block = {"Xb2": "13201", "Xb3": "13401", "Xb4": "13601", "Xb5": "13801"}.get(name.split("-")[0], "13001")
        for line in open(f):
            if not line.strip():
                continue
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            cells[v["condition"]][(block, v["game"])] = (v["result"], bool(v["credited"]))
            for i, n in enumerate(v.get("lag_hist") or []):
                lag[v["condition"]][i] += n
    return cells, lag


def win(g):
    return g[0] == "W" and g[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    a = ap.parse_args()
    cells, lag = load(a.dir)
    print("| cell | games | W:L:D:T | credited wins | decisions at lag 1 / 2 / 3 | mean lag | vs base: only cell / only base | McNemar p |")
    print("|---|---|---|---|---|---|---|---|")
    for k in sorted(cells):
        g = cells[k]
        theirs = k.rstrip().split("/")[-1] if "lag " in k else k.split(" vs ")[-1]
        if k.startswith(("B3 ", "B2 ")):
            base_name = "B4 " + k.split(" ", 1)[1]  # a smaller budget is paired with the budget 4 of the SAME cell (same lag or margin)
        else:
            base_name = f"B4 m9 vs {theirs}" if " m" in k else f"B4 lag 2/{theirs}"
        base = cells.get(base_name)
        c = defaultdict(int)
        for r in g.values():
            c[r[0]] += 1
        w = sum(win(r) for r in g.values())
        h = lag.get(k)
        lagcol = mean = ""
        if h:
            n = sum(h.values())
            lagcol = " / ".join(f"{100 * h[i] / n:.0f}%" for i in (1, 2, 3))
            mean = f"{sum(i * x for i, x in h.items()) / n:.2f}"
        pair = p = ""
        if base and k != base_name:
            idx = sorted(set(g) & set(base))
            oa = sum(win(g[i]) and not win(base[i]) for i in idx)
            ob = sum(win(base[i]) and not win(g[i]) for i in idx)
            pair = f"{oa} / {ob} (n = {len(idx)})"
            p = f"{mcnemar_exact(oa, ob):.3f}"
        print(f"| {k} | {len(g)} | {c['W']}:{c['L']}:{c['D']}:{c['T']} | {pct(w, len(g))} | {lagcol} | {mean} | {pair} | {p} |")


if __name__ == "__main__":
    main()
