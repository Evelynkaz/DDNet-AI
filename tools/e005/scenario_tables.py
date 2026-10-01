#!/usr/bin/env python3
"""Markdown table of the T1-T18 technique scenarios for several brains.

  tools/e005/scenario_tables.py <eval-root> label1 label2 ...
"""
import json, sys, os

root, labels = sys.argv[1], sys.argv[2:]
rows = {}
for l in labels:
    with open(os.path.join(root, l, 'scenarios', 'scenarios.json')) as f:
        for s in json.load(f):
            rows.setdefault((s['id'], s['name']), {})[l] = s
print('| # | Приём | ' + ' | '.join(labels) + ' |')
print('|---|---|' + '---|' * len(labels))
tot = {l: [0, 0] for l in labels}
for (i, name), by in sorted(rows.items(), key=lambda kv: (int(''.join(ch for ch in kv[0][0] if ch.isdigit())), kv[0][0])):
    cells = []
    for l in labels:
        s = by.get(l)
        if s:
            cells.append(f"{s['successes']}/{s['trials']} ({100*s['rate']:.0f}%; {100*s['lo']:.0f}–{100*s['hi']:.0f})")
            tot[l][0] += s['successes']
            tot[l][1] += s['trials']
        else:
            cells.append('—')
    print(f'| {i} | {name} | ' + ' | '.join(cells) + ' |')
print('| | **Всего проб** | ' + ' | '.join(f"{a}/{b} ({100*a/max(b,1):.0f}%)" for a, b in tot.values()) + ' |')
