#!/usr/bin/env python3
"""Per-round table of the 2026-10-08 duel (19 rounds, 9:10) from the merged clips.

usage: rounds.py <box.txt from map_ascii 74 163 108 190> <jsonl>...   (prints a table and the decisive window of every round with frames)
"""
import sys
import glob
import math
from collections import Counter
sys.path.insert(0, __import__('os').path.dirname(__file__))
from load import *

X0, Y0 = 74, 163


def load_box(path):
    rows = {}
    for line in open(path):
        p = line.split()
        if len(p) == 2 and p[0].isdigit():
            rows[int(p[0])] = p[1]
    return rows


BOX = None


def tl(x, y):
    r = BOX.get(y)
    if r is None or not (X0 <= x < X0 + len(r)):
        return '?'
    return r[x - X0]


def freeze_tile_near(t):
    """The freeze tile(s) the tee touches (tee radius 14 px -> check the 4 corners like the server's freeze check)."""
    x, y = t['ch']['x'], t['ch']['y']
    hits = set()
    for dx in (-14, 14):
        for dy in (-14, 14):
            tx, ty = (x + dx) // 32, (y + dy) // 32
            if tl(tx, ty) in 'Ff':
                hits.add((tx, ty))
    cx, cy = x // 32, y // 32
    if tl(cx, cy) in 'Ff':
        hits.add((cx, cy))
    return sorted(hits)


def where(tiles):
    """A name for a freeze tile of the box."""
    names = set()
    for (x, y) in tiles:
        if y <= 168:
            names.add('ceiling')
        elif x >= 104 and y <= 185:
            names.add('right wall')
        elif x <= 78 and y <= 180:
            names.add('left wall')
        elif y >= 182 and x <= 91:
            names.add('pit left wall / under floor')
        elif y >= 186:
            names.add('pit bottom')
        else:
            names.add(f'({x},{y})')
    return '+'.join(sorted(names)) or 'none'


def onsets(frames, ticks, who):
    out = []
    prev = None
    for t in ticks:
        me = tee(frames[t], who)
        if me is None:
            prev = None
            continue
        if me['frozen'] and prev is not None and not prev['frozen']:
            out.append(t)
        prev = me
    return out


def context(frames, ticks, t0, victim, other, back=40):
    """What happened to `victim` in the `back` ticks before its freeze onset t0."""
    w = [t for t in ticks if t0 - back <= t <= t0]
    hits = [t - t0 for t in w for k, v in evs(frames[t]) if k == 'HammerHit' and v['to'] == victim and v['from'] == other]
    own_hits = [t - t0 for t in w for k, v in evs(frames[t]) if k == 'HammerHit' and v['from'] == victim and v['to'] == other]
    hook_on = [t - t0 for t in w if (tee(frames[t], other) or {}).get('ch', {}).get('hooked_player') == victim]
    my_hook = [t - t0 for t in w if (tee(frames[t], victim) or {}).get('ch', {}).get('hooked_player') == other]
    my_hook_ground = [t - t0 for t in w if (tee(frames[t], victim) or {}).get('ch', {}).get('hook_state') == 5
                      and (tee(frames[t], victim) or {}).get('ch', {}).get('hooked_player') == -1]
    jumps = []
    if victim == OWN:
        last = 0
        for t in w:
            for s in frames[t]['sent']:
                j = s['input']['jump']
                if j and not last:
                    jumps.append(s['tick'] - t0)
                last = j
    else:
        pj = None
        for t in w:
            o = tee(frames[t], victim)
            if o is None:
                continue
            j = o['ch']['jumped'] & 1
            if pj is not None and j and not pj:
                jumps.append(t - t0)
            pj = j
    v = tee(frames[t0], victim)
    return dict(hits=hits, own_hits=own_hits, hook_on=hook_on, my_hook=my_hook, my_hook_ground=my_hook_ground, jumps=jumps,
                vy=vel(v)[1], vx=vel(v)[0], pos=(v['ch']['x'], v['ch']['y']), ftiles=freeze_tile_near(v))


def summ_ticks(xs):
    if not xs:
        return '-'
    # compress runs of consecutive frames (2-tick steps)
    runs = []
    s = p = xs[0]
    for x in xs[1:]:
        if x - p <= 2:
            p = x
            continue
        runs.append((s, p))
        s = p = x
    runs.append((s, p))
    return ','.join(f'{a}' if a == b else f'{a}..{b}' for a, b in runs)


def bot_stats(frames, ticks, a, b):
    lag = Counter()
    cand = []
    tot = []
    fl = Counter()
    for t in ticks:
        if a <= t <= b:
            bt = frames[t]['bot']
            if bt['aimed_tick']:
                lag[bt['aimed_tick'] - t - 1] += 1
            if bt['candidates']:
                cand.append(bt['candidates'])
            if bt['total_us']:
                tot.append(bt['total_us'])
            for k, v in FLAGS.items():
                if bt['flags'] & k:
                    fl[v] += 1
    return lag, cand, tot, fl


def main():
    global BOX
    BOX = load_box(sys.argv[1])
    frames, _ = merged(sys.argv[2:])
    ticks = sorted(frames)
    our_on = onsets(frames, ticks, OWN)
    his_on = onsets(frames, ticks, OPP)
    for k in range(len(LIVES) - 1):
        a, b = LIVES[k], LIVES[k + 1]
        rt = [t for t in ticks if a <= t < b]
        print('=' * 110)
        print(f'round {k + 1} {OUTCOME[k + 1]} ticks {a}..{b} ({b - a} ticks, {(b - a) / 50:.1f} s), frames {len(rt)}, first frame {rt[0] if rt else None}')
        if not rt:
            continue
        # countdown end
        me0 = tee(frames[rt[0]], OWN)
        fe = me0['dd']['freeze_end'] if me0 and me0['dd'] else None
        print(f'  our freeze_end at first frame: {fe} (countdown end = start + {fe - a if fe else None})')
        uo = [t for t in our_on if a + 152 <= t < b]
        ho = [t for t in his_on if a + 152 <= t < b]
        print(f'  our freeze onsets after the countdown: {[t - a for t in uo]} | his: {[t - a for t in ho]} (ticks from round start)')
        for who, lst in ((OWN, uo), (OPP, ho)):
            for t0 in lst:
                c = context(frames, ticks, t0, who, OPP if who == OWN else OWN)
                # how long did the freeze last within the clip
                end = next((t for t in rt if t > t0 and not (tee(frames[t], who) or {'frozen': True})['frozen']), None)
                print(f"   {'US ' if who == OWN else 'HIM'} frozen at +{t0 - a} (end-{b - t0}): pos {c['pos']} tile {(c['pos'][0] // 32, c['pos'][1] // 32)} v({c['vx']:+.1f},{c['vy']:+.1f}) freeze {where(c['ftiles'])} {c['ftiles']}"
                      f" | thawed {'+' + str(end - t0) if end else 'no (in clip)'}")
                print(f"       hits on victim {summ_ticks(c['hits'])} | victim hits {summ_ticks(c['own_hits'])} | other's hook on victim {summ_ticks(c['hook_on'])} | victim hook on other {summ_ticks(c['my_hook'])}"
                      f" | victim hook on ground {summ_ticks(c['my_hook_ground'])} | victim jumps {c['jumps']}")
        last = uo[-1] if (OUTCOME[k + 1] == 'L' and uo) else (ho[-1] if ho else rt[-1])
        lag, cand, tot, fl = bot_stats(frames, ticks, last - 30, last)
        cs = sorted(cand)
        print(f'  last 30 ticks before the decisive freeze ({last}): lag {dict(sorted(lag.items()))} candidates mean {sum(cand) / max(1, len(cand)):.1f} min {cs[0] if cs else None} max {cs[-1] if cs else None}'
              f" total_us p50 {sorted(tot)[len(tot) // 2] if tot else None} max {max(tot) if tot else None} flags {dict(fl)}")
        lag, cand, tot, fl = bot_stats(frames, ticks, a + 150, b)
        print(f'  whole round: lag {dict(sorted(lag.items()))} candidates mean {sum(cand) / max(1, len(cand)):.1f} flags {dict(fl)}')


if __name__ == '__main__':
    main()
