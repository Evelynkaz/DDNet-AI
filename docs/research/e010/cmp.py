#!/usr/bin/env python3
"""Compare two arena output dirs game by game, ignoring wall timings. Exit 1 on any difference."""
import json, sys, glob, os
def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.endswith('_ms') and k != 'timing'}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o
a, b = sys.argv[1], sys.argv[2]
bad = n = 0
for f in sorted(glob.glob(os.path.join(a, '*.jsonl'))):
    g = os.path.join(b, os.path.basename(f))
    la = [strip(json.loads(l)) for l in open(f)]
    lb = [strip(json.loads(l)) for l in open(g)]
    if len(la) != len(lb):
        print('LEN', f, len(la), len(lb)); bad += 1; continue
    for x, y in zip(la, lb):
        n += 1
        if x != y:
            bad += 1
            print('DIFF', os.path.basename(f), x['game'], [p['hash'] for p in x['players']], [p['hash'] for p in y['players']])
print(f'{n} games compared, {bad} differ')
sys.exit(1 if bad else 0)
