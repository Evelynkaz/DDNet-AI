#!/usr/bin/env python3
"""Ranks evaluated E-008 runs by the mean credited win rate over the TRAINING-arena conditions of the evaluation matrix
(`arena_tag == "train"`: clb-left, clb-left 1v2, clb-left 1v3; the holdout halls clb-right and chillblock5-ruler never take
part). Used to name the "best trained fly / MLP-w" for the proposer comparison without looking at the holdout.
  tools/e008/rank_runs.py fly|mlpw [name-substring]"""
import glob, json, os, sys
ROOT = os.path.expanduser('~/aiddnet/data/runs/E-008/eval')
kind = sys.argv[1]
sub = sys.argv[2] if len(sys.argv) > 2 else ''
rows = []
for d in sorted(glob.glob(os.path.join(ROOT, f'*-{kind}-*'))):
    name = os.path.basename(d)
    if sub not in name:
        continue
    for variant in ('arena', 'arena-half'):
        p = os.path.join(d, variant, 'summary.json')
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        tr = [c for c in s['conditions'] if c['arena_tag'] == 'train']
        if len(tr) < 3 or any(c['games'] < 1000 for c in tr):
            continue
        rows.append((sum(c['credited_w'] / c['games'] for c in tr) / len(tr), name, variant, [c['credited_w'] / c['games'] for c in tr]))
for m, n, v, cs in sorted(rows, reverse=True):
    print(f"{100*m:5.1f}  {n} [{v}]  " + ' / '.join(f'{100*c:.1f}' for c in cs))
