#!/usr/bin/env python3
"""Russian markdown tables of the proposer comparison (E-008 part 4) from `arena run` output directories.

  tools/e008/proposer_tables.py <arena-out-dir> [<arena-out-dir> ...] [--baseline none] [--out file.md]

Conditions are named "<shape> | <proposer>" (tools/e008/gen_proposer_config.py). Per shape and proposer: games, the
credited win rate with a Wilson CI (D-059), blocks/min and self-freezes/min, the share of decisions whose chosen plan
came from the proposer, the wall-clock decision time p50/p99 of the focal player (includes VM pauses, D-045) and the
search work per decision, and a paired exact McNemar test of "credited win" against the baseline proposer (games are
paired by game index: the same seed, spawn order and swap). Several directories are merged (shape, proposer) by name.
"""
import argparse, json, math, os, sys
from collections import defaultdict


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact binomial p-value of the discordant pair counts (b: only A won, c: only the baseline won)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(dirs):
    summ, games = {}, defaultdict(dict)
    for d in dirs:
        if not os.path.exists(os.path.join(d, 'summary.json')):  # a cell still being played
            continue
        s = json.load(open(os.path.join(d, 'summary.json')))
        for c in s['conditions']:
            summ[c['name']] = c
        for fn in os.listdir(d):
            if not fn.endswith('.jsonl'):
                continue
            for line in open(os.path.join(d, fn)):
                g = json.loads(line)
                games[g['condition']][g['game']] = g
    return summ, games


def split(name):
    shape, _, prop = name.partition(' | ')
    return shape, prop


def fmt_pct(p):
    return '—' if p is None or (isinstance(p, float) and math.isnan(p)) else f'{100 * p:.1f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--baseline', default='none')
    ap.add_argument('--out')
    a = ap.parse_args()
    summ, games = load([os.path.expanduser(d) for d in a.dirs])
    order = ['clb-left vs planner', 'clb-right vs planner', 'chillblock5-ruler vs planner', 'clb-left', 'clb-right',
             'chillblock5-ruler', 'clb-left 1v2', 'clb-left 1v3']
    shapes = sorted({split(n)[0] for n in summ}, key=lambda s: order.index(s) if s in order else len(order))
    kind = lambda p: (['idle', 'none', 'untrained-fly'].index(p) if p in ('idle', 'none', 'untrained-fly')
                      else 3 if p.startswith('fly:') else 4, p)
    lines = []
    for shape in shapes:
        names = sorted((n for n in summ if split(n)[0] == shape), key=lambda n: kind(split(n)[1]))
        lines.append(f'\n**{shape}**\n')
        lines.append('| proposer | игр | credited-побед, % [ДИ Уилсона] | Δ к ' + a.baseline + ', п.п. | McNemar p (только он / только '
                     + a.baseline + ') | блоков/мин | самозамор./мин | выбран proposer'
                     'ом, % решений | решение p50 / p99, мс | работа, тиков/решение |')
        lines.append('|---|---:|---|---|---|---:|---:|---:|---|---:|')
        base_name = f'{shape} | {a.baseline}'
        base = games.get(base_name, {})
        for n in names:
            c = summ[n]
            prop = split(n)[1]
            k, g = c['credited_w'], c['games']
            p, lo, hi = wilson(k, g)
            delta = mc = '—'
            if n != base_name and base:
                mine = games[n]
                b = cc = 0
                common = 0
                for gi, gm in mine.items():
                    bm = base.get(gi)
                    if bm is None:
                        continue
                    common += 1
                    x = bool(gm['credited']) and gm['result'] == 'W'
                    y = bool(bm['credited']) and bm['result'] == 'W'
                    b += x and not y
                    cc += y and not x
                kb = sum(1 for gm in base.values() if gm['credited'] and gm['result'] == 'W')
                delta = f'{100 * (p - kb / len(base)):+.1f}'
                mc = f'{mcnemar_exact(b, cc):.3g} ({b} / {cc}; {common} пар)'
            pl = c['players'][0]
            ts = pl.get('telemetry_sum', {})
            dec = ts.get('totals.decisions', 0) or pl['decisions']
            chosen = ts.get('totals.chosen.proposal')
            share = '—' if chosen is None or not dec else f'{100 * chosen / dec:.1f}'
            work = ts.get('totals.work.ticks')
            wtxt = '—' if work is None or not dec else f'{work / dec:.0f}'
            lines.append(
                f"| {prop} | {g} | {100 * p:.1f} [{100 * lo:.1f}; {100 * hi:.1f}] | {delta} | {mc} | {c['blocks_per_min']:.2f} | "
                f"{c['self_freezes_per_min']:.2f} | {share} | {pl['decide_us_p50'] / 1000:.1f} / {pl['decide_us_p99'] / 1000:.1f} | {wtxt} |")
    # pooled over the halls of a group (all games of a proposer in the group against the same games of the baseline)
    groups = [('против planner', [x for x in shapes if x.endswith('vs planner')]),
              ('против scripted', [x for x in shapes if not x.endswith('vs planner')])]
    lines.append('\n**Сводно по группам условий** (те же игры, что у ' + a.baseline + '; McNemar по парам игр; условия не независимы друг от друга: общая база)\n')
    lines.append('| группа | proposer | игр | credited-побед, % | Δ к ' + a.baseline + ', п.п. | McNemar p (только он / только ' + a.baseline + ') |')
    lines.append('|---|---|---:|---:|---|---|')
    for gname, gshapes in groups:
        props = []
        for n in summ:
            sh, pr = split(n)
            if sh in gshapes and pr not in props:
                props.append(pr)
        props.sort(key=kind)
        for pr in props:
            tot = k = bk = b = c = 0
            for sh in gshapes:
                mine, base = games.get(f'{sh} | {pr}', {}), games.get(f'{sh} | {a.baseline}', {})
                for gi, gm in mine.items():
                    bm = base.get(gi)
                    if bm is None:
                        continue
                    x = bool(gm['credited']) and gm['result'] == 'W'
                    y = bool(bm['credited']) and bm['result'] == 'W'
                    tot += 1; k += x; bk += y; b += x and not y; c += y and not x
            if not tot:
                continue
            delta = '—' if pr == a.baseline else f'{100 * (k - bk) / tot:+.2f}'
            mc = '—' if pr == a.baseline else f'{mcnemar_exact(b, c):.3g} ({b} / {c})'
            lines.append(f'| {gname} ({len(gshapes)} усл.) | {pr} | {tot} | {100 * k / tot:.1f} | {delta} | {mc} |')
    text = '\n'.join(lines) + '\n'
    if a.out:
        open(a.out, 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
