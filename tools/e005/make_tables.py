#!/usr/bin/env python3
"""Russian markdown tables of E-005 (arena matrix, DAgger rounds, per-head offline metrics).

  tools/e005/make_tables.py <section> [labels...]
sections: arena1 | arenaP | arenaN | credited1 | creditedN | rounds | offline | misc
Reads ~/aiddnet/data/runs/E-005/{eval,offline,<run>}/.
"""
import json, os, sys

ROOT = os.path.expanduser('~/aiddnet/data/runs/E-005')

def pct(x):
    return f'{100*x:.1f}'

def arena_cell(c):
    t = c['tally']; wr = c['win_rate']
    if not wr:
        return f"{t['w']}:{t['l']}:{t['d']}:{t['t']}"
    return f"{pct(wr['p'])} [{pct(wr['lo'])}; {pct(wr['hi'])}] ({t['w']}:{t['l']}:{t['d']}:{t['t']})"

def wilson(k, n, z=1.959963984540054):
    """95% Wilson interval of k successes in n trials (the arena's own formula)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))

def credited_of(c):
    """Credited win rate of a condition summary: games won by the focal player's own credited block over
    all games (D-059). Recomputed from `credited_w` for summaries written before the arena reported it."""
    n = c['games']
    k = c['credited_w']
    lo, hi = wilson(k, n)
    return k / n, lo, hi

def load_arena(label):
    with open(f'{ROOT}/eval/{label}/arena/summary.json') as f:
        return {c['name']: c for c in json.load(f)['conditions']}

def section_arena(labels, names):
    data = {l: load_arena(l) for l in labels}
    print('| Условие | ' + ' | '.join(labels) + ' |')
    print('|---|' + '---|' * len(labels))
    for n in names:
        print(f'| {n} | ' + ' | '.join(arena_cell(data[l][n]) if n in data[l] else '—' for l in labels) + ' |')

def section_credited(labels, names):
    """One table per condition: credited-win rate (headline, D-059), blocks/min, self-freezes/min and
    W/(W+L+D) (secondary); `idle` and `scripted` (scripted vs scripted) are the baselines."""
    data = {l: load_arena(l) for l in labels}
    for n in names:
        print(f'\n**{n}**\n')
        print('| Мозг | credited-побед, % [ДИ] | блоков/мин | самозаморозок/мин | W/(W+L+D), % [ДИ] | W:L:D:T |')
        print('|---|---|---|---|---|---|')
        for l in labels:
            c = data[l].get(n)
            if c is None:
                continue
            p, lo, hi = credited_of(c)
            t = c['tally']; wr = c['win_rate']
            sec = f"{pct(wr['p'])} [{pct(wr['lo'])}; {pct(wr['hi'])}]" if wr else '—'
            print(f"| {l} | **{pct(p)}** [{pct(lo)}; {pct(hi)}] | {c['blocks_per_min']:.2f} | "
                  f"{c['self_freezes_per_min']:.2f} | {sec} | {t['w']}:{t['l']}:{t['d']}:{t['t']} |")

def section_matrix(labels, names, what):
    """Rows = conditions, columns = brains; `what` is credited | blocks | freezes (D-059 columns)."""
    data = {l: load_arena(l) for l in labels}
    def cell(c):
        if what == 'credited':
            p, lo, hi = credited_of(c)
            return f"**{pct(p)}** [{pct(lo)}; {pct(hi)}]"
        if what == 'blocks':
            return f"{c['blocks_per_min']:.2f}"
        if what == 'bf':
            return f"{c['blocks_per_min']:.1f} / {c['self_freezes_per_min']:.1f}"
        return f"{c['self_freezes_per_min']:.2f}"
    print('| Условие | ' + ' | '.join(labels) + ' |')
    print('|---|' + '---|' * len(labels))
    for n in names:
        print(f'| {n} | ' + ' | '.join(cell(data[l][n]) if n in data[l] else '—' for l in labels) + ' |')

def section_rounds(runs, arenas):
    print('| Запуск | Фаза | ' + ' | '.join(arenas) + ' |')
    print('|---|---|' + '---|' * len(arenas))
    for run in runs:
        rows = {}
        for line in open(f'{ROOT}/{run}/metrics.jsonl'):
            d = json.loads(line)
            if d.get('kind') == 'arena':
                e = d['eval']
                wr = e['win_rate']
                rows.setdefault(d['phase'], {})[e['arena']] = f"{pct(wr[0])} [{pct(wr[1])}; {pct(wr[2])}]" if wr else '—'
        for ph, r in rows.items():
            print(f'| {run} | {ph} | ' + ' | '.join(r.get(a, '—') for a in arenas) + ' |')

def section_offline(runs, sets):
    for s in sets:
        print(f'\n**{s}**\n')
        print('| Запуск | n | dir acc (баз.) | dir top-2 | jump AUROC | hook AUROC / bal.acc | fire AUROC | aim мед., ° | dir+jump+hook (top-2) |')
        print('|---|---|---|---|---|---|---|---|---|')
        for n in runs:
            d = {r['set']: r['report'] for r in json.load(open(f'{ROOT}/offline/{n}.json'))}
            r = d.get(s)
            if not r:
                continue
            print(f"| {n} | {r['steps']} | {r['dir']['accuracy']:.3f} ({r['dir']['majority_baseline']:.3f}) | {r['dir']['top2_accuracy']:.3f} | "
                  f"{r['jump']['auroc']:.3f} | {r['hook']['auroc']:.3f} / {r['hook']['balanced_accuracy']:.3f} | {r['fire']['auroc']:.3f} | "
                  f"{r['aim']['median_error_deg']:.1f} | {r['joint_dir_jump_hook']:.3f} ({r['joint_top2_dir_jump_hook']:.3f}) |")

def section_misc(labels):
    data = {l: load_arena(l) for l in labels}
    print('| Мозг | условие | самозамор./мин | блоков/мин | p50 / p99 решения, мкс (6 потоков арены) |')
    print('|---|---|---|---|---|')
    for l in labels:
        for n in ('clb-left vs scripted', 'chillblock5-ruler vs scripted'):
            c = data[l][n]; p = c['players'][0]
            print(f"| {l} | {n} | {c['self_freezes_per_min']:.2f} | {c['blocks_per_min']:.2f} | {p['decide_us_p50']} / {p['decide_us_p99']} |")

if __name__ == '__main__':
    sec, args = sys.argv[1], sys.argv[2:]
    one = ['clb-left vs scripted', 'clb-right vs scripted', 'chillblock5-ruler vs scripted', 'pit vs scripted', 'platform vs scripted']
    pl = ['clb-left vs planner', 'clb-right vs planner', 'chillblock5-ruler vs planner']
    multi = [f'{a} 1v{k} vs scripted' for a in ('clb-left', 'clb-right', 'chillblock5-ruler') for k in (2, 3)]
    if sec == 'arena1': section_arena(args, one)
    elif sec == 'credited1': section_credited(args, one)
    elif sec == 'creditedN': section_credited(args, multi)
    elif sec in ('cm1', 'cmN', 'bf1', 'bfN'):
        what = {'cm': 'credited', 'bf': 'bf'}[sec[:2]]
        section_matrix(args, one if sec.endswith('1') else multi, what)
    elif sec == 'arenaP': section_arena(args, pl)
    elif sec == 'arenaN': section_arena(args, multi)
    elif sec == 'rounds': section_rounds(args, ['clb-left', 'clb-right', 'chillblock5-ruler', 'pit', 'platform'])
    elif sec == 'offline': section_offline(args, ['teacher-val', 'teacher-holdout:chillblock5-ruler', 'teacher-holdout:clb-right', 'human-val', 'human-val-tagged'])
    elif sec == 'misc': section_misc(args)
