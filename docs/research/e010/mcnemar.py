#!/usr/bin/env python3
"""Paired old-vs-new comparison of credited wins (D-059) per condition pair: Wilson CIs, W:L:D:T, McNemar exact test.
usage: mcnemar.py RUN_DIR   (conditions named '... OLD ...' and '... NEW ...'; files <slug>.jsonl)"""
import json, math, sys, os, re, glob

def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0, c - h), min(1, c + h))

def binom_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # exact two-sided: 2 * P(X <= k), X ~ Bin(n, 1/2), capped at 1
    s = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * s)

def load(path):
    games = {}
    for line in open(path):
        g = json.loads(line)
        games[g['game']] = g
    return games

def stats(games):
    n = len(games)
    tally = {'W': 0, 'L': 0, 'D': 0, 'T': 0}
    cw = 0; sf = 0
    for g in games.values():
        tally[g['result']] += 1
        cw += int(g['result'] == 'W' and g['credited'])
        sf += g['a_self_freezes']
    return n, tally, cw, sf

def fmt(p):
    return f"{100 * p[0]:.1f} [{100 * p[1]:.1f}; {100 * p[2]:.1f}]"

def main(run):
    files = {}
    for f in glob.glob(os.path.join(run, '*.jsonl')):
        first = json.loads(open(f).readline())
        files[first['condition']] = f
    names = sorted(files)
    pairs = []
    for nm in names:
        if ' OLD' in nm:
            new = nm.replace(' OLD', ' NEW')
            if new in files:
                pairs.append((nm, new))
    print('| condition | OLD credited % [CI] | OLD W:L:D:T | NEW credited % [CI] | NEW W:L:D:T | delta (pp) | old-only / new-only wins | McNemar p | self-freezes old (n) | self-freezes new (n) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    pooled = {}
    for o, nw in pairs:
        go, gn = load(files[o]), load(files[nw])
        common = sorted(set(go) & set(gn))
        no, to, cwo, sfo = stats(go); nn, tn, cwn, sfn = stats(gn)
        b = c = 0
        for k in common:
            wo = go[k]['result'] == 'W' and go[k]['credited']
            wn = gn[k]['result'] == 'W' and gn[k]['credited']
            b += int(wo and not wn); c += int(wn and not wo)
        label = nw.replace(' NEW', '')
        key = label.split(': ', 1)[-1] if ': ' in label else None
        if key and 'vs ' in key:
            a = pooled.setdefault(key, dict(no=0, cwo=0, nn=0, cwn=0, b=0, c=0))
            a['no'] += no; a['cwo'] += cwo; a['nn'] += nn; a['cwn'] += cwn; a['b'] += b; a['c'] += c
        print(f"| {label} | {fmt(wilson(cwo, no))} | {to['W']}:{to['L']}:{to['D']}:{to['T']} | {fmt(wilson(cwn, nn))} | {tn['W']}:{tn['L']}:{tn['D']}:{tn['T']} | {100 * (cwn / nn - cwo / no):+.1f} | {b} / {c} | {binom_two_sided(b, c):.3f} | {sfo} | {sfn} |")
    pooled_rows(pooled)

def pooled_rows(pooled):
    for key, a in pooled.items():
        print(f"| ALL HALLS: {key} | {fmt(wilson(a['cwo'], a['no']))} | | {fmt(wilson(a['cwn'], a['nn']))} | | {100 * (a['cwn'] / a['nn'] - a['cwo'] / a['no']):+.1f} | {a['b']} / {a['c']} | {binom_two_sided(a['b'], a['c']):.4f} | | |")

if __name__ == '__main__':
    main(sys.argv[1])
