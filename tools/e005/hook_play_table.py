#!/usr/bin/env python3
"""Per-round hook start / release rates of the student against the teacher (review F2 of E-005), from `ddnet-ai train stats`.

  tools/e005/hook_play_table.py <bin> run1 run2 ...      (datasets under ~/aiddnet/data/datasets/teacher/<run>-dagger)
Cell: `start student/teacher · release student/teacher · own hook out` in percent.
"""
import os, re, subprocess, sys

binary, runs = sys.argv[1], sys.argv[2:]
root = os.path.expanduser('~/aiddnet/data/datasets/teacher')
print('| Запуск | раунд 1 | раунд 2 | раунд 3 | раунд 4 | раунд 5 |')
print('|---|---|---|---|---|---|')
for run in runs:
    out = subprocess.run([binary, 'train', 'stats', f'{root}/{run}-dagger'], capture_output=True, text=True).stdout
    cells = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 10 or not parts[0].isdigit():
            continue
        start, release, out_share = parts[7], parts[8], parts[9]
        cells[int(parts[0])] = f'{start.replace("%","")} · {release.replace("%","")} · {out_share}'
    print(f'| {run} | ' + ' | '.join(cells.get(r, '—') for r in range(1, 6)) + ' |')
