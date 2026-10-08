#!/usr/bin/env python3
"""Task 3.19 (D-116, E-034): paired statistics of `duel_stats --jsonl` output.

    python3 tools/d115/summarize.py games.jsonl [--base base] [--md]

Conditions are named "<our lag>/<his lag> <arm>" (tools/d115/gen_configs.py). Game `g` of every condition has the same seed and layout, so the arms of a
cell are paired by game index. For every arm and cell: W:L:D:T, credited wins with the Wilson 95% interval, the decided share W/(W+L) with its interval,
the paired difference to the base arm in percentage points of games (arm wins minus base wins over all games) with the exact McNemar test on the games
only one of them won, and the same pooled over the cells. A "win" is the focal player's credited win.
"""
import json
import math
import sys
from collections import defaultdict

Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    m = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    s = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * s)


def paired_diff(a_wins, b_wins):
    """Arm a against base b over the same games: (only a, only b, difference in pp of games, 95% half-width in pp)."""
    n = len(a_wins)
    only_a = sum(1 for x, y in zip(a_wins, b_wins) if x and not y)
    only_b = sum(1 for x, y in zip(a_wins, b_wins) if y and not x)
    d = (only_a - only_b) / n
    # variance of the paired difference of proportions
    var = ((only_a + only_b) / n - d * d) / n
    return only_a, only_b, 100 * d, 100 * Z * math.sqrt(max(var, 0.0))


def pairs(rows, spec):
    """`--pairs a:b,c:d`: arm a against arm b (not only against the base arm), per cell and pooled."""
    pooled = defaultdict(lambda: ([], []))
    print("| contrast | cell | n | arm wins | other wins | only arm / only other | diff pp | McNemar p |")
    print("|---|---|---:|---:|---:|---|---:|---:|")
    for pair in spec.split(","):
        a_name, b_name = pair.split(":")
        for cell in sorted(rows):
            arms = rows[cell]
            if a_name not in arms or b_name not in arms:
                continue
            common = sorted(set(arms[a_name]) & set(arms[b_name]))
            a = [arms[a_name][g]["result"] == "W" for g in common]
            b = [arms[b_name][g]["result"] == "W" for g in common]
            oa, ob, dpp, hw = paired_diff(a, b)
            pooled[pair][0].extend(a)
            pooled[pair][1].extend(b)
            print(f"| {a_name} vs {b_name} | {cell} | {len(common)} | {sum(a)} | {sum(b)} | {oa} / {ob} | {dpp:+.1f} ± {hw:.1f} | {mcnemar_exact(oa, ob):.3f} |")
        a, b = pooled[pair]
        if a:
            oa, ob, dpp, hw = paired_diff(a, b)
            print(f"| **{pair.replace(':', ' vs ')}** | **pooled** | {len(a)} | {sum(a)} | {sum(b)} | {oa} / {ob} | **{dpp:+.1f} ± {hw:.1f}** | **{mcnemar_exact(oa, ob):.4f}** |")


def main():
    path = sys.argv[1]
    base_arm = "base"
    if "--base" in sys.argv:
        base_arm = sys.argv[sys.argv.index("--base") + 1]
    rows = defaultdict(dict)  # cell -> arm -> {game: result row}
    for line in open(path):
        r = json.loads(line)
        cell, _, arm = r["condition"].partition(" ")
        rows[cell].setdefault(arm, {})[r["game"]] = r
    pooled = defaultdict(lambda: ([], []))
    if "--pairs" in sys.argv:
        pairs(rows, sys.argv[sys.argv.index("--pairs") + 1])
        return
    print("| cell | arm | n | W:L:D:T | credited wins | decided share W/(W+L) | vs base: only arm / only base | diff pp | McNemar p |")
    print("|---|---|---:|---|---|---|---|---:|---:|")
    for cell in sorted(rows):
        arms = rows[cell]
        base = arms.get(base_arm)
        for arm, games in arms.items():
            idx = sorted(games)
            res = [games[g]["result"] for g in idx]
            w, l, d, t = (res.count(x) for x in "WLDT")
            wins = [r == "W" for r in res]
            lo, hi = wilson(w, len(res))
            dl, dh = wilson(w, w + l)
            cmp_txt, diff_txt, p_txt = "", "", ""
            if base and arm != base_arm:
                common = sorted(set(idx) & set(base))
                a = [games[g]["result"] == "W" for g in common]
                b = [base[g]["result"] == "W" for g in common]
                oa, ob, dpp, hw = paired_diff(a, b)
                cmp_txt = f"{oa} / {ob}"
                diff_txt = f"{dpp:+.1f} ± {hw:.1f}"
                p_txt = f"{mcnemar_exact(oa, ob):.3f}"
                pa, pb = pooled[arm]
                pa.extend(a)
                pb.extend(b)
            print(
                f"| {cell} | {arm} | {len(res)} | {w}:{l}:{d}:{t} | {100 * w / len(res):.1f}% [{100 * lo:.1f}; {100 * hi:.1f}] | "
                f"{100 * w / max(1, w + l):.1f}% [{100 * dl:.1f}; {100 * dh:.1f}] | {cmp_txt} | {diff_txt} | {p_txt} |"
            )
    if pooled:
        print()
        print("Pooled over the cells (paired by game):")
        print("| arm | n pairs | only arm / only base | diff pp | McNemar p |")
        print("|---|---:|---|---:|---:|")
        for arm, (a, b) in pooled.items():
            oa, ob, dpp, hw = paired_diff(a, b)
            print(f"| {arm} | {len(a)} | {oa} / {ob} | {dpp:+.1f} ± {hw:.1f} | {mcnemar_exact(oa, ob):.3f} |")


main()
