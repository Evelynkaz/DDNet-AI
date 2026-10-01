#!/usr/bin/env python3
"""Markdown (Russian headers) tables of E-005 from the eval directories written by eval_all.sh.

  tools/e005/tables.py <eval-root> label1 label2 ...
"""
import json, sys, os

root = sys.argv[1]
labels = sys.argv[2:]

def load(label):
    with open(os.path.join(root, label, 'arena', 'summary.json')) as f:
        return {c['name']: c for c in json.load(f)['conditions']}

def cell(c):
    t = c['tally']
    wr = c['win_rate']
    if wr is None:
        return f"{t['w']}:{t['l']}:{t['d']}:{t['t']} —"
    a = c['win_rate_all']
    return (f"{t['w']}:{t['l']}:{t['d']}:{t['t']} **{100*wr['p']:.1f}** [{100*wr['lo']:.1f}; {100*wr['hi']:.1f}]"
            f" / {100*a['p']:.1f}")

data = {l: load(l) for l in labels}
names = list(next(iter(data.values())).keys())
print('| Условие | ' + ' | '.join(labels) + ' |')
print('|---|' + '---|' * len(labels))
for n in names:
    print(f'| {n} | ' + ' | '.join(cell(data[l][n]) if n in data[l] else '—' for l in labels) + ' |')
print()
print('Самозаморозки/мин, блоки/мин, решение p50/p99 мкс (по условию «clb-left vs scripted» и «chillblock5-ruler vs scripted»):')
print()
print('| Мозг | условие | самозамор./мин | блоков/мин | p50, мкс | p99, мкс |')
print('|---|---|---|---|---|---|')
for l in labels:
    for n in ('clb-left vs scripted', 'chillblock5-ruler vs scripted'):
        c = data[l].get(n)
        if c:
            p = c['players'][0]
            print(f"| {l} | {n} | {c['self_freezes_per_min']:.2f} | {c['blocks_per_min']:.2f} | {p['decide_us_p50']} | {p['decide_us_p99']} |")
