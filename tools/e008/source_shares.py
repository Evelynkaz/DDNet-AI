#!/usr/bin/env python3
"""Candidate sources of the hybrid per proposer (E-008 part 4b, "why does no proposer help"): per decision, the candidates
generated and evaluated by source and the share of decisions whose chosen plan came from each source, from the telemetry
sums of the arena summaries.
  tools/e008/source_shares.py "<shape>" <arena-out-dir> [<arena-out-dir> ...]"""
import glob, json, os, sys
shape = sys.argv[1]
SRC = ['proposal', 'book', 'cem', 'technique', 'throw', 'warm']
rows = []
for d in sys.argv[2:]:
    f = os.path.join(os.path.expanduser(d), 'summary.json')
    if not os.path.exists(f):
        continue
    for c in json.load(open(f))['conditions']:
        sh, _, p = c['name'].partition(' | ')
        if sh != shape or p == 'idle':
            continue
        ts = c['players'][0]['telemetry_sum']
        dec = ts['totals.decisions']
        g = lambda k: ts.get('totals.' + k, 0) / dec
        rows.append((p, g('work.ticks'), [g('generated.' + s) for s in SRC], [100 * g('chosen.' + s) for s in SRC]))
print('| proposer | работа, тиков/решение | кандидатов на решение: ' + ' / '.join(SRC) + ' | выбрано из источника, % решений: ' + ' / '.join(SRC) + ' |')
print('|---|---:|---|---|')
for p, w, gen, ch in rows:
    print(f'| {p} | {w:.0f} | ' + ' / '.join(f'{x:.1f}' for x in gen) + ' | ' + ' / '.join(f'{x:.1f}' for x in ch) + ' |')
