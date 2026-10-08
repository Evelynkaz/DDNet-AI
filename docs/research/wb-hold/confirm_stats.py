#!/usr/bin/env python3
"""Task 3.18 (E-033): the pre-registered go bar of docs/research/wb-hold.md section 9, computed from the JSONL of the confirmation runs.

    confirm_stats.py RUN_DIR [RUN_DIR ...]      # every dir holds `<arena> <opponents> base.jsonl` / `... wbhold.jsonl` pairs

A condition is a pair of files named `<prefix>-base.jsonl` / `<prefix>-wbhold.jsonl` (the arena writes the condition name with spaces replaced by `-`). For
each pair it pairs the games by index (same seeds and layouts) and reports

* held W (the focal player won with its own credited block and the victim was out on every tick of the 250-tick window), the paired exact McNemar test,
  and the exact one-sided sign test for "wbhold is worse";
* credited wins and losses (must not move: `wb_hold` acts only after the first block);
* `focal_out_in_window` (we were out at some tick of the window) and our own freezes before the deciding tick;
* exploratory (not part of the bar): the victim's free ticks in the 250-tick window of a credited win, and the share of credited wins whose victim was free for at most
  10 ticks in all ("tolerant hold": a victim that thawed and was thrown back into the freeze at once is held in the human sense, not in the strict one);
* the search cost: `work.ticks` per decision and the share of decisions with the hall's wall swings (`generated.wall`).

Then the bar: pooled over the clb conditions (everything whose name starts with `clb-`), the other conditions (`joni-`) reported separately.
"""
import glob
import json
import math
import os
import sys


def binom_two_sided(b, c):
    """Exact McNemar: p of a split b:c of the discordant pairs under p = 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def binom_one_sided_less(k, n):
    """P(X <= k), X ~ Bin(n, 1/2)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(path):
    out = {}
    for line in open(path):
        if line.strip():
            v = json.loads(line)
            out[v["game"]] = v
    return out


def held_win(g):
    return g["result"] == "W" and g["credited"] and g.get("held_block", False)


def cost(games):
    ticks = decisions = wall_dec = 0
    for g in games:
        p = g["players"][0]
        t = p.get("telemetry", {}).get("totals", {})
        decisions += t.get("decisions", p.get("decisions", 0))
        ticks += t.get("work", {}).get("ticks", 0)
        wall_dec += t.get("generated", {}).get("wall", 0)
    return ticks, decisions, wall_dec


def main():
    pairs = []
    for d in sys.argv[1:]:
        for f in sorted(glob.glob(os.path.join(d, "*-base.jsonl"))):
            g = f.replace("-base.jsonl", "-wbhold.jsonl")
            if os.path.exists(g):
                pairs.append((os.path.basename(f)[: -len("-base.jsonl")], f, g))
    rows = []
    for name, fa, fb in pairs:
        a, b = load(fa), load(fb)
        assert a.keys() == b.keys(), name
        ids = sorted(a)
        ha = [held_win(a[i]) for i in ids]
        hb = [held_win(b[i]) for i in ids]
        only_a = sum(1 for x, y in zip(ha, hb) if x and not y)
        only_b = sum(1 for x, y in zip(ha, hb) if y and not x)
        cred_a = sum(1 for i in ids if a[i]["result"] == "W" and a[i]["credited"])
        cred_b = sum(1 for i in ids if b[i]["result"] == "W" and b[i]["credited"])
        wl_a = (sum(a[i]["result"] == "W" for i in ids), sum(a[i]["result"] == "L" for i in ids))
        wl_b = (sum(b[i]["result"] == "W" for i in ids), sum(b[i]["result"] == "L" for i in ids))
        fo_a = sum(1 for i in ids if a[i]["focal_out_in_window"])
        fo_b = sum(1 for i in ids if b[i]["focal_out_in_window"])
        sf_a = sum(a[i]["a_self_freezes"] for i in ids)
        sf_b = sum(b[i]["a_self_freezes"] for i in ids)
        ca, cb = cost(a.values()), cost(b.values())
        free_a = [250 - a[i]["victim_out_ticks"] for i in ids if a[i]["result"] == "W" and a[i]["credited"]]
        free_b = [250 - b[i]["victim_out_ticks"] for i in ids if b[i]["result"] == "W" and b[i]["credited"]]
        rows.append(
            dict(name=name, free=(free_a, free_b), n=len(ids), ha=sum(ha), hb=sum(hb), only_a=only_a, only_b=only_b, cred=(cred_a, cred_b), wl=(wl_a, wl_b),
                 fo=(fo_a, fo_b), sf=(sf_a, sf_b), ca=ca, cb=cb)
        )
    if not rows:
        print("no pairs found")
        return 1
    print("| condition | games | held W base | held W wbhold | only base | only wbhold | McNemar p | worse (1-sided sign p) | credited b/w | W:L base | W:L wbhold | out in window b/w | self-freezes b/w |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|---|---|---|---|---|")
    for r in rows:
        worse = binom_one_sided_less(r["only_b"], r["only_a"] + r["only_b"])
        print(
            f"| {r['name']} | {r['n']} | {r['ha']} ({100 * r['ha'] / r['n']:.1f}%) | {r['hb']} ({100 * r['hb'] / r['n']:.1f}%) | {r['only_a']} | {r['only_b']} | "
            f"{binom_two_sided(r['only_a'], r['only_b']):.4f} | {worse:.4f} | {r['cred'][0]}/{r['cred'][1]} | {r['wl'][0][0]}:{r['wl'][0][1]} | "
            f"{r['wl'][1][0]}:{r['wl'][1][1]} | {r['fo'][0]}/{r['fo'][1]} | {r['sf'][0]}/{r['sf'][1]} |"
        )

    def pooled(label, sel):
        rs = [r for r in rows if sel(r["name"])]
        if not rs:
            return None
        n = sum(r["n"] for r in rs)
        ha, hb = sum(r["ha"] for r in rs), sum(r["hb"] for r in rs)
        oa, ob = sum(r["only_a"] for r in rs), sum(r["only_b"] for r in rs)
        d = 100.0 * (hb - ha) / n
        lo_a, hi_a = wilson(ha, n)
        lo_b, hi_b = wilson(hb, n)
        p = binom_two_sided(oa, ob)
        fo = (sum(r["fo"][0] for r in rs), sum(r["fo"][1] for r in rs))
        cred = (sum(r["cred"][0] for r in rs), sum(r["cred"][1] for r in rs))
        wl = (
            (sum(r["wl"][0][0] for r in rs), sum(r["wl"][0][1] for r in rs)),
            (sum(r["wl"][1][0] for r in rs), sum(r["wl"][1][1] for r in rs)),
        )
        ta = (sum(r["ca"][0] for r in rs), sum(r["ca"][1] for r in rs), sum(r["ca"][2] for r in rs))
        tb = (sum(r["cb"][0] for r in rs), sum(r["cb"][1] for r in rs), sum(r["cb"][2] for r in rs))
        fa = [x for r in rs for x in r["free"][0]]
        fb = [x for r in rs for x in r["free"][1]]
        print()
        print(f"**{label}** ({len(rs)} conditions, {n} paired games per arm)")
        if fa and fb:
            print(f"- exploratory, credited wins {len(fa)}: victim free ticks of the window mean {sum(fa) / len(fa):.1f} -> {sum(fb) / len(fb):.1f}; "
                  f"free <= 10 ticks (tolerant hold): {100 * sum(1 for x in fa if x <= 10) / len(fa):.1f}% -> {100 * sum(1 for x in fb if x <= 10) / len(fb):.1f}%")
        # Paired Wald interval of the difference: the variance of the paired difference is (oa + ob - (ob - oa)^2 / n) / n^2.
        se = math.sqrt(max(oa + ob - (ob - oa) ** 2 / n, 0.0)) / n
        ci = (d - 196.0 * se, d + 196.0 * se)
        print(f"- held W / games: base {ha} ({100 * ha / n:.1f}% [{100 * lo_a:.1f}; {100 * hi_a:.1f}]) -> wbhold {hb} ({100 * hb / n:.1f}% [{100 * lo_b:.1f}; {100 * hi_b:.1f}]), "
              f"difference {d:+.2f} pp [paired Wald 95%: {ci[0]:.2f}; {ci[1]:.2f}]; discordant {oa}:{ob}, exact McNemar p = {p:.2e}")
        print(f"- credited wins {cred[0]} -> {cred[1]}; W:L {wl[0][0]}:{wl[0][1]} -> {wl[1][0]}:{wl[1][1]}")
        print(f"- we were out in the window: {fo[0]} -> {fo[1]} games ({100 * (fo[1] - fo[0]) / n:+.2f} pp)")
        if ta[1] and tb[1]:
            print(f"- work ticks per decision: {ta[0] / ta[1]:.0f} -> {tb[0] / tb[1]:.0f} ({100 * ((tb[0] / tb[1]) / (ta[0] / ta[1]) - 1):+.1f}%); "
                  f"wall-swing candidates per decision: {ta[2] / ta[1]:.3f} -> {tb[2] / tb[1]:.3f}")
        return dict(d=d, p=p, n=n, fo=100 * (fo[1] - fo[0]) / n, cred=cred, wl=wl)

    clb = pooled("clb (the bar)", lambda s: s.startswith("clb-"))
    pooled("joni", lambda s: s.startswith("joni-"))
    pooled("clb scripted", lambda s: s.startswith("clb-") and "scripted" in s)
    pooled("clb live-v2", lambda s: s.startswith("clb-") and "live-v2" in s)
    print()
    if clb:
        worse_any = [
            r["name"]
            for r in rows
            if binom_one_sided_less(r["only_b"], r["only_a"] + r["only_b"]) < 0.05 and r["only_b"] < r["only_a"]
        ]
        print("bar 1 (pooled clb >= +4.0 pp and p < 0.01):", "PASS" if clb["d"] >= 4.0 and clb["p"] < 0.01 else "FAIL")
        print("bar 2 (no condition significantly worse):", "PASS" if not worse_any else f"FAIL {worse_any}")
        n = clb["n"]
        dcred = 100.0 * (clb["cred"][1] - clb["cred"][0]) / n
        dw = 100.0 * (clb["wl"][1][0] - clb["wl"][0][0]) / n
        dl = 100.0 * (clb["wl"][1][1] - clb["wl"][0][1]) / n
        ok3 = abs(dcred) <= 1.0 and abs(dw) <= 1.0 and abs(dl) <= 1.0
        print(f"bar 3 (credited wins, W and L each within +-1 pp of all games; now {dcred:+.2f}, {dw:+.2f}, {dl:+.2f} pp):", "PASS" if ok3 else "FAIL")
        print("bar 4 (we out in the window: at most +1 pp):", "PASS" if clb["fo"] <= 1.0 else "FAIL")
        print("bars 5-6: cost and identity are read from the lines above / the identity check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
