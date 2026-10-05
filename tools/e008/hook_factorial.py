#!/usr/bin/env python3
"""Analyses the 2x2x2 hook-study factorial (E-005 review F6): distance-bin gains x connectome lr 6x x alpha_init 4.

  tools/e008/hook_factorial.py <study-seed1.json> <study-seed2.json> ...   (output of `train hook-study --factorial`)

Prints the per-cell hook AUROC (mean +- sd over seeds, same map / other map), the main effect of each factor
(difference of the mean AUROC with the factor on versus off, averaged over the other two factors; per-seed
effects give the mean and its standard error) and the three two-way interactions. With 3 seeds the standard
error is crude: effects are called "clear" only when they exceed two standard errors.
"""
import json, math, sys, itertools

cells = {}  # (gains, lr6, a4) -> {'same': [..], 'other': [..]}
for path in sys.argv[1:]:
    for arm in json.load(open(path)):
        name = arm['arm']
        if not name.startswith('f: '):
            continue
        key = ('no gains' not in name, 'lr 6x' in name, 'alpha 4' in name)
        d = cells.setdefault(key, {'same': [], 'other': []})
        d['same'].append(arm['same_map']['hook']['auroc'])
        d['other'].append(arm['other_map']['hook']['auroc'])

def ms(x):
    m = sum(x) / len(x)
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else float('nan')
    return m, sd

def label(k):
    return f"gains {'on ' if k[0] else 'off'} | lr {'6x' if k[1] else '1x'} | alpha {'4      ' if k[2] else 'default'}"

n_seeds = min(len(v['same']) for v in cells.values())
print(f'seeds per cell: {n_seeds}')
print('\n| cell | AUROC same map (mean +- sd) | AUROC other map |')
print('|---|---|---|')
for k in sorted(cells):
    a, b = ms(cells[k]['same']), ms(cells[k]['other'])
    print(f"| {label(k)} | {a[0]:.3f} +- {a[1]:.3f} | {b[0]:.3f} +- {b[1]:.3f} |")

def effect(which, idx, fn):
    """Per-seed contrast: mean over cells where fn(key) is +1 minus mean over cells where it is -1."""
    per_seed = []
    for s in range(n_seeds):
        plus = [cells[k][which][s] for k in cells if fn(k) > 0]
        minus = [cells[k][which][s] for k in cells if fn(k) < 0]
        per_seed.append(sum(plus) / len(plus) - sum(minus) / len(minus))
    m, sd = ms(per_seed)
    se = sd / math.sqrt(n_seeds) if n_seeds > 1 else float('nan')
    return m, se

factors = {
    'distance-bin gains': lambda k: 1 if k[0] else -1,
    'connectome lr 6x': lambda k: 1 if k[1] else -1,
    'alpha_init 4': lambda k: 1 if k[2] else -1,
}
print('\n| effect (AUROC change, factor on - off) | same map | other map |')
print('|---|---|---|')
for name, fn in factors.items():
    row = []
    for which in ('same', 'other'):
        m, se = effect(which, 0, fn)
        flag = 'clear' if abs(m) > 2 * se else 'within noise'
        row.append(f'{m:+.3f} +- {se:.3f} ({flag})')
    print(f'| {name} | {row[0]} | {row[1]} |')
for (n1, f1), (n2, f2) in itertools.combinations(factors.items(), 2):
    row = []
    for which in ('same', 'other'):
        m, se = effect(which, 0, lambda k, f1=f1, f2=f2: f1(k) * f2(k))
        row.append(f'{m:+.3f} +- {se:.3f} ({"clear" if abs(m) > 2 * se else "within noise"})')
    print(f'| interaction {n1} x {n2} | {row[0]} | {row[1]} |')

# The arm E-005 never ran: everything else at the accepted settings except the gains.
k_acc, k_nogain = (True, True, True), (False, True, True)
if k_acc in cells and k_nogain in cells:
    for which in ('same', 'other'):
        d = [a - b for a, b in zip(cells[k_acc][which], cells[k_nogain][which])]
        m, sd = ms(d)
        se = sd / math.sqrt(len(d)) if len(d) > 1 else float('nan')
        print(f"\nWithout the gains at lr 6x, alpha 4 ({which} map): {ms(cells[k_nogain][which])[0]:.3f}; "
              f"the gains add {m:+.3f} +- {se:.3f}")
