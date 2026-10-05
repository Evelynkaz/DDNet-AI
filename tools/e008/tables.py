#!/usr/bin/env python3
"""Russian markdown tables of E-008 from ~/aiddnet/data/runs/E-008/{<run>,eval/<run>}.

  tools/e008/tables.py arms <model> [arm ...]        # phase 1: one row per arm, seeds pooled (mean and min..max);
                                                     # <model> = p2:mlpw / p2:fly for the data arms of phase 2
  tools/e008/tables.py runs <run> [run ...]          # one row per run (every seed)
  tools/e008/tables.py rounds <run> [run ...]        # in-play hook start/release per DAgger round and the selected round
D-059: the credited win rate leads; W/(W+L+D) is not shown here (see the arena summaries).
"""
import json, math, os, re, sys
from glob import glob

ROOT = os.path.expanduser('~/aiddnet/data/runs/E-008')
CONDS = ['clb-left vs scripted', 'clb-right vs scripted', 'chillblock5-ruler vs scripted', 'clb-left 1v2 vs scripted', 'clb-left 1v3 vs scripted']
SHORT = ['clb-left', 'clb-right', 'ChillBlock5', '1v2', '1v3']
OFFENSIVE = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T14', 'T15a', 'T15b', 'T16']


def jl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except ValueError:  # an eval still writing the file
        return None


def credited(c):
    n = c['games']
    k = c['credited_w']
    p = k / n
    z = 1.959963984540054
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, centre - half, centre + half


def run_info(name):
    ev = os.path.join(ROOT, 'eval', name)
    info = {'name': name}
    arena = load_json(os.path.join(ev, 'arena', 'summary.json'))
    info['arena'] = {c['name']: c for c in arena['conditions']} if arena else {}
    half = load_json(os.path.join(ev, 'arena-half', 'summary.json'))
    info['arena_half'] = {c['name']: c for c in half['conditions']} if half else {}
    info['hook_play'] = load_json(os.path.join(ev, 'hook-play.json')) or []
    info['hook_play_half'] = load_json(os.path.join(ev, 'hook-play-half.json')) or []
    off = load_json(os.path.join(ev, 'offline.json'))
    info['offline'] = {r['set']: r['report'] for r in off} if off else {}
    sc = load_json(os.path.join(ev, 'scenarios', 'scenarios.json'))
    info['scen'] = {s['id']: s['successes'] for s in sc} if sc else {}
    metrics = jl(os.path.join(ROOT, name, 'metrics.jsonl'))
    info['rounds'] = [m for m in metrics if m.get('kind') == 'hook_play']
    sel = [m for m in metrics if m.get('kind') == 'selection']
    info['selected'] = sel[-1]['phase'] if sel else None
    info['selection_table'] = sel[-1]['table'] if sel else []
    info['self_freeze'] = None
    return info


def pooled_hook(hp):
    """Student and teacher start/release rates pooled over the arenas of a hook-play JSON."""
    c = {'no': [0, 0, 0], 'out': [0, 0, 0]}
    for ev in hp:
        for key, sub in (('no', 'not_out'), ('out', 'out')):
            v = ev['counts'][sub]
            c[key][0] += v['n']; c[key][1] += v['student_hook']; c[key][2] += v['teacher_hook']
    start_s = c['no'][1] / c['no'][0] if c['no'][0] else float('nan')
    start_t = c['no'][2] / c['no'][0] if c['no'][0] else float('nan')
    rel_s = 1 - c['out'][1] / c['out'][0] if c['out'][0] else float('nan')
    rel_t = 1 - c['out'][2] / c['out'][0] if c['out'][0] else float('nan')
    out_share = c['out'][0] / (c['out'][0] + c['no'][0]) if (c['out'][0] + c['no'][0]) else float('nan')
    return start_s, start_t, rel_s, rel_t, out_share


def pct(x, d=0):
    return '—' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{100 * x:.{d}f}'


def spread(vals, d=1):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return '—'
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return f'{100 * m:.{d}f}'
    return f'{100 * m:.{d}f} ({100 * min(vals):.0f}…{100 * max(vals):.0f})'


def arm_rows(model, arms):
    phase = 'p1'
    if ':' in model:  # `p2:mlpw`: the data arms of phase 2
        phase, model = model.split(':')
    runs = sorted(glob(os.path.join(ROOT, f'e008-{phase}-{model}-*-s*')))
    by_arm = {}
    for r in runs:
        n = os.path.basename(r)
        m = re.match(rf'e008-{phase}-{model}-(.+)-s(\d+)$', n)
        if m:
            by_arm.setdefault(m.group(1), []).append(n)
    print('| Арена | сидов | старт хука в игре, % (ученик / учитель) | отпускание, % (ученик / учитель) | доля решений с выпущенным хуком | ' +
          ' | '.join(f'credited {s}' for s in SHORT) + ' | блоков/мин (clb-left) | самозамор./мин (clb-left) | T1–T7, T14–T16 (сумма из 1100) |')
    print('|---|---|---|---|---|' + '---|' * len(SHORT) + '---|---|---|')
    for arm in arms or sorted(by_arm):
        names = by_arm.get(arm, [])
        infos = [run_info(n) for n in names if run_info(n)['arena']]
        if not infos:
            continue
        hooks = [pooled_hook(i['hook_play']) for i in infos if i['hook_play']]
        def avg(idx):
            v = [h[idx] for h in hooks if not math.isnan(h[idx])]
            return sum(v) / len(v) if v else float('nan')
        cells = []
        for c in CONDS:
            cells.append(spread([credited(i['arena'][c])[0] for i in infos if c in i['arena']]))
        bl = [i['arena'][CONDS[0]]['blocks_per_min'] for i in infos if CONDS[0] in i['arena']]
        sf = [i['arena'][CONDS[0]]['self_freezes_per_min'] for i in infos if CONDS[0] in i['arena']]
        off = [sum(i['scen'].get(t, 0) for t in OFFENSIVE) for i in infos if i['scen']]
        print(f'| {arm} | {len(infos)} | {pct(avg(0))} / {pct(avg(1))} | {pct(avg(2))} / {pct(avg(3))} | {pct(avg(4))} | ' +
              ' | '.join(cells) + f' | {sum(bl)/len(bl):.2f} | {sum(sf)/len(sf):.2f} | ' +
              (f'{sum(off)/len(off):.0f} ({min(off)}…{max(off)})' if off else '—') + ' |')


def run_rows(names):
    print('| Запуск | выбран | старт ученик/учитель | отпускание ученик/учитель | ' + ' | '.join(f'credited {s} [ДИ]' for s in SHORT) + ' |')
    print('|---|---|---|---|' + '---|' * len(SHORT))
    for n in names:
        i = run_info(n)
        if not i['arena']:
            continue
        s = pooled_hook(i['hook_play']) if i['hook_play'] else (float('nan'),) * 5
        cells = []
        for c in CONDS:
            if c in i['arena']:
                p, lo, hi = credited(i['arena'][c])
                cells.append(f'{100*p:.1f} [{100*lo:.0f}; {100*hi:.0f}]')
            else:
                cells.append('—')
        print(f"| {n} | {i['selected']} | {pct(s[0])} / {pct(s[1])} | {pct(s[2])} / {pct(s[3])} | " + ' | '.join(cells) + ' |')


def round_rows(names):
    print('| Запуск | раунд 1 | раунд 2 | раунд 3 | раунд 4 | раунд 5 | выбран |')
    print('|---|---|---|---|---|---|---|')
    for n in names:
        i = run_info(n)
        cells = {}
        for m in i['rounds']:
            c = m['counts']
            no, out = c['not_out'], c['out']
            ss = no['student_hook'] / no['n'] if no['n'] else float('nan')
            st = no['teacher_hook'] / no['n'] if no['n'] else float('nan')
            rs = 1 - out['student_hook'] / out['n'] if out['n'] else float('nan')
            rt = 1 - out['teacher_hook'] / out['n'] if out['n'] else float('nan')
            cells[m['round']] = f'{pct(ss)}/{pct(st)} · {pct(rs)}/{pct(rt)}'
        print(f"| {n} | " + ' | '.join(cells.get(r, '—') for r in range(1, 6)) + f" | {i['selected']} |")


def ab_rows(names):
    """F9 A/B: the shipped (calibrated at the end of every phase) thresholds against the same weights at 0.5."""
    print('| Запуск | пороги jump/hook/fire | ' + ' | '.join(f'credited {s}: подобранные / 0,5' for s in SHORT) + ' | старт хука в игре: подобранные / 0,5 (учитель) |')
    print('|---|---|' + '---|' * len(SHORT) + '---|')
    for n in names:
        i = run_info(n)
        if not i['arena'] or not i['arena_half']:
            continue
        t = selected_thresholds(n, i['selected'])
        cells = []
        for c in CONDS:
            if c in i['arena'] and c in i['arena_half']:
                cells.append(f"{100*credited(i['arena'][c])[0]:.1f} / {100*credited(i['arena_half'][c])[0]:.1f}")
            else:
                cells.append('—')
        s = pooled_hook(i['hook_play']) if i['hook_play'] else (float('nan'),) * 5
        h = pooled_hook(i['hook_play_half']) if i['hook_play_half'] else (float('nan'),) * 5
        tt = f"{t['jump']:.2f}/{t['hook']:.2f}/{t['fire']:.2f}" if t else '—'
        print(f'| {n} | {tt} | ' + ' | '.join(cells) + f' | {pct(s[0])} / {pct(h[0])} ({pct(s[1])}) |')


def technique_rows():
    """Per-technique successes (of 100 trials) of every evaluated run, for the offensive techniques of the E-008 sum
    (T1-T7, T14-T16): the sum alone is misleading because one technique (T4) carries most of it. `сценарии` marks the runs
    whose teacher data contain technique scenarios (round1-v2, phase 2 d2/cb5 and the headline d2-switch)."""
    cols = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T14', 'T15a', 'T15b', 'T16']
    print('| Запуск | сценарии в данных | ' + ' | '.join(cols) + ' |')
    print('|---|---|' + '---|' * len(cols))
    for d in sorted(glob(os.path.join(ROOT, 'eval', 'e008-*'))):
        n = os.path.basename(d)
        sc = load_json(os.path.join(d, 'scenarios', 'scenarios.json'))
        if not sc:
            continue
        got = {x['id']: x['successes'] for x in sc}
        with_scen = '-d2-' in n or '-cb5-' in n
        print(f"| {n[5:]} | {'да' if with_scen else 'нет'} | " + ' | '.join(str(got.get(c, '—')) for c in cols) + ' |')


def selected_thresholds(name, phase):
    """The thresholds fitted at the end of the phase whose bundle was selected (the shipped `selected.bundle`), not the
    last phase's."""
    th = [m for m in jl(os.path.join(ROOT, name, 'metrics.jsonl')) if m.get('kind') == 'thresholds']
    for m in th:
        if m.get('phase') == phase:
            return m
    return th[-1] if th else None


def abp_rows(names):
    """F9, three threshold variants of the same weights: offline-calibrated (shipped), matched on the in-play start rate
    (play_threshold.py) and everything at 0.5; credited win rate per condition and the in-play start rate."""
    print('| Запуск | порог хука: подобранный офлайн / по игре | ' + ' | '.join(f'credited {s}: офлайн / по игре / 0,5' for s in SHORT) + ' | старт хука в игре (учитель) |')
    print('|---|---|' + '---|' * len(SHORT) + '---|')
    for n in names:
        i = run_info(n)
        pl = load_json(os.path.join(ROOT, 'eval', n, 'arena-play', 'summary.json'))
        pt = load_json(os.path.join(ROOT, 'eval', n, 'play-threshold.json'))
        if not i['arena'] or not pl or not pt:
            continue
        t = selected_thresholds(n, i['selected'])
        cal = t['hook'] if t else float('nan')
        play = {c['name']: c for c in pl['conditions']}
        cells = []
        for c in CONDS:
            a, b, h = i['arena'].get(c), play.get(c), i['arena_half'].get(c)
            cells.append(' / '.join(f'{100*credited(x)[0]:.1f}' if x else '—' for x in (a, b, h)))
        best = min(pt['evaluations'], key=lambda e: abs(e['d']))
        print(f"| {n} | {cal:.3f} / {pt['chosen_hook_threshold']:.2f} | " + ' | '.join(cells) +
              f" | {100*best['start_student']:.0f}% ({100*best['start_teacher']:.0f}%) |")


def scan_rows(name):
    """In-play sensitivity of the hook threshold (threshold_scan.sh): start/release and the clb-left credited win rate."""
    d = os.path.join(ROOT, 'eval', name, 'threshold-scan')
    print('| порог хука | старт ученик / учитель | отпускание ученик / учитель | credited clb-left, % |')
    print('|---|---|---|---|')
    for f in sorted(glob(os.path.join(d, 'hook-play-*.json')), key=lambda x: float(os.path.basename(x)[10:-5])):
        t = os.path.basename(f)[10:-5]
        hp = load_json(f)
        s = pooled_hook(hp)
        a = load_json(os.path.join(d, f'arena-{t}', 'summary.json'))
        cr = f"{100*credited(a['conditions'][0])[0]:.1f}" if a else '—'
        print(f'| {t} | {pct(s[0])} / {pct(s[1])} | {pct(s[2])} / {pct(s[3])} | {cr} |')


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'arms':
        arm_rows(args[0], args[1:])
    elif cmd == 'runs':
        run_rows(args)
    elif cmd == 'rounds':
        round_rows(args)
    elif cmd == 'ab':
        ab_rows(args)
    elif cmd == 'techniques':
        technique_rows()
    elif cmd == 'abp':
        abp_rows(args)
    elif cmd == 'scan':
        scan_rows(args[0])
