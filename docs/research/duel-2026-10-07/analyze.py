#!/usr/bin/env python3
"""Post-mortem of the 2026-10-07 test duel: per-round stats and the decisive window of each clip.

Input: JSONL from `clip_json` (line 1 header, then frames). Output: text tables on stdout.
"""
import json
import math
import sys
from collections import Counter

OWN = 16
OPP = 0
FLAGS = {1: 'S', 2: 'T', 4: 'H', 8: 'h', 16: 'G', 32: 'vh', 64: 'vf', 128: 'W', 256: 'X', 512: 'P', 1024: 'B', 2048: 'D'}
# S searched, T out of time, H shield stepped in, h shield incomplete, G guarded, vh vetoed hook, vf vetoed fire,
# W wander, X crossing, P planned freeze, B WB holding, D brain decided


def load(path):
    with open(path) as f:
        header = json.loads(f.readline())
        frames = [json.loads(l) for l in f]
    return header, frames


def tee(fr, i):
    for t in fr['tees']:
        if t['id'] == i:
            return t
    return None


def ev_list(fr):
    out = []
    for e in fr['events']:
        if isinstance(e, dict):
            (k, v), = e.items()
            out.append((k, v))
        else:
            out.append((e, {}))
    return out


def flagstr(f):
    return ''.join(v for k, v in FLAGS.items() if f & k)


def vel(t):
    return t['ch']['vel_x'] / 256.0, t['ch']['vel_y'] / 256.0


def rounds(frames):
    """Split by our countdown freezes (freeze_end - tick >= 140 right after a respawn/teleport)."""
    starts = []
    for i, fr in enumerate(frames):
        me = tee(fr, OWN)
        if not me or not me['dd']:
            continue
        fe = me['dd']['freeze_end']
        if me['frozen'] and fe - fr['tick'] >= 140 and me['ch']['vel_x'] == 0 and (
                i == 0 or not (tee(frames[i - 1], OWN) or {}).get('frozen', False)
                or any(k in ('Respawn', 'Kill') for k, _ in ev_list(fr))):
            starts.append((i, fe))
    return starts


def fight_stats(frames, a, b):
    """Stats over frames[a:b] (both free)."""
    c = Counter()
    holds = {OWN: [], OPP: []}
    dirs = {OWN: [], OPP: []}
    lag = Counter()
    tot = []
    above = 0
    n = 0
    dist = []
    hooked_on_us = 0
    our_hook_on_him = 0
    for fr in frames[a:b]:
        me, op = tee(fr, OWN), tee(fr, OPP)
        if not me or not op:
            continue
        n += 1
        for k, v in ev_list(fr):
            if k == 'HammerHit':
                c[f"hit {v['from']}->{v['to']}"] += 1
            elif k == 'HammerFire':
                c[f"fire {v['from']}"] += 1
            elif k == 'HookAttach' and v['target'] in (OWN, OPP):
                c[f"hook {v['id']}->{v['target']}"] += 1
            elif k == 'HookRelease' and v['target'] in (OWN, OPP):
                holds[v['id']].append(v['held'])
        dirs[OPP].append(op['ch']['direction'])
        for s in fr['sent']:
            dirs[OWN].append(s['input']['direction'])
        if fr['bot']['aimed_tick']:
            lag[fr["bot"]["aimed_tick"] - fr["tick"] - 1] += 1
        if fr['bot']['total_us']:
            tot.append(fr['bot']['total_us'])
        if op['ch']['y'] < me['ch']['y'] - 16:
            above += 1
        dist.append(math.hypot(me['ch']['x'] - op['ch']['x'], me['ch']['y'] - op['ch']['y']))
        if op['ch']['hooked_player'] == OWN:
            hooked_on_us += 1
        if me['ch']['hooked_player'] == OPP:
            our_hook_on_him += 1
    flips = {}
    for k, d in dirs.items():
        flips[k] = sum(1 for x, y in zip(d, d[1:]) if x != y)
    return dict(n=n, ev=c, holds=holds, flips=flips, lag=lag, tot=sorted(tot), above=above, dist=sorted(dist),
                hooked_on_us=hooked_on_us, our_hook_on_him=our_hook_on_him, dir_samples={k: len(v) for k, v in dirs.items()})


def pct(xs, p):
    if not xs:
        return float('nan')
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float('nan')


def first_freeze(frames, a, who):
    for i in range(a, len(frames)):
        t = tee(frames[i], who)
        if t and t['frozen']:
            p = tee(frames[i - 1], who) if i > 0 else None
            if p and not p['frozen']:
                return i
    return None


def window(frames, fi, back=44):
    rows = []
    for i in range(max(0, fi - back // 2), min(len(frames), fi + 3)):
        fr = frames[i]
        me, op = tee(fr, OWN), tee(fr, OPP)
        if not me or not op:
            continue
        evs = []
        for k, v in ev_list(fr):
            if k == 'HammerHit':
                evs.append(f"HIT{v['from']}>{v['to']}")
            elif k == 'HookAttach' and v['target'] in (OWN, OPP):
                evs.append(f"hk{v['id']}>{v['target']}")
            elif k == 'HookRelease' and v['target'] in (OWN, OPP):
                evs.append(f"rel{v['id']}({v['held']})")
            elif k == 'FreezeOnset':
                evs.append(f"FRZ{v['id']}")
            elif k in ('Kill', 'Respawn'):
                evs.append(k)
        ins = ' '.join(f"{s['input']['direction']:+d}{'J' if s['input']['jump'] else '.'}{'H' if s['input']['hook'] else '.'}{'F' if (s['input']['fire'] & 1) else '.'}" for s in fr['sent'])
        mvx, mvy = vel(me)
        ovx, ovy = vel(op)
        b = fr['bot']
        lagt = b["aimed_tick"] - fr["tick"] - 1 if b['aimed_tick'] else None
        rows.append(
            f"{fr['tick'] - frames[fi]['tick']:+4d} us({me['ch']['x']:5d},{me['ch']['y']:5d}) v({mvx:+6.1f},{mvy:+6.1f}){' F' if me['frozen'] else '  '} "
            f"hk{me['ch']['hook_state']:+d}/{me['ch']['hooked_player']:+d} | him({op['ch']['x']:5d},{op['ch']['y']:5d}) v({ovx:+6.1f},{ovy:+6.1f}){' F' if op['frozen'] else '  '} "
            f"d{op['ch']['direction']:+d} hk{op['ch']['hook_state']:+d}/{op['ch']['hooked_player']:+d} | in {ins} | lag {lagt} {b['total_us'] / 1000:.1f}ms {flagstr(b['flags'])} | {' '.join(evs)}")
    return rows


def main():
    for path in sys.argv[1:]:
        header, frames = load(path)
        print('=' * 100)
        print(path, header['reason'])
        print('ticks', frames[0]['tick'], frames[-1]['tick'], 'frames', len(frames))
        rs = rounds(frames)
        print('round starts (frame idx, thaw tick):', [(i, frames[i]['tick'], fe) for i, fe in rs])
        # The clip's first segment and each round after a countdown.
        segs = []
        starts = [0] + [i for i, _ in rs]
        for k, s in enumerate(starts):
            e = starts[k + 1] if k + 1 < len(starts) else len(frames)
            # fight begins once both are free
            a = s
            while a < e:
                me, op = tee(frames[a], OWN), tee(frames[a], OPP)
                if me and op and not me['frozen'] and not op['frozen']:
                    break
                a += 1
            if a >= e:
                continue
            fo = first_freeze(frames, a, OWN)
            fh = first_freeze(frames, a, OPP)
            end_candidates = [x for x in (fo, fh) if x is not None and x < e]
            end = min(end_candidates) if end_candidates else e
            segs.append((a, end, fo if fo is not None and fo < e else None, fh if fh is not None and fh < e else None))
        for a, end, fo, fh in segs:
            st = fight_stats(frames, a, end)
            who = 'us' if fo == end else ('him' if fh == end else 'none (clip end/round end)')
            print('-' * 100)
            print(f"segment ticks {frames[a]['tick']}..{frames[end - 1]['tick'] if end > a else '-'} ({(frames[end - 1]['tick'] - frames[a]['tick']) if end > a else 0} ticks), first freeze: {who}")
            ev = st['ev']
            secs = max(1e-9, (frames[end - 1]['tick'] - frames[a]['tick']) / 50.0)
            print(f"  hammer hits him->us {ev['hit 0->16']} us->him {ev['hit 16->0']} | fires him {ev['fire 0']} us {ev['fire 16']} | per 30 s: him {ev['hit 0->16'] * 30 / secs:.1f} us {ev['hit 16->0'] * 30 / secs:.1f}")
            print(f"  hook grabs him->us {ev['hook 0->16']} us->him {ev['hook 16->0']} | hold med him {med(st['holds'][OPP])} us {med(st['holds'][OWN])} | sum held him {sum(st['holds'][OPP])} us {sum(st['holds'][OWN])}")
            print(f"  frames his hook on us {st['hooked_on_us']}/{st['n']}  ours on him {st['our_hook_on_him']}/{st['n']}  he is above us {st['above']}/{st['n']}  dist p50 {pct(st['dist'], .5):.0f}px")
            print(f"  direction changes per 30s: us {st['flips'][OWN] * 30 / secs:.0f} (per-tick sent) him {st['flips'][OPP] * 30 / secs:.0f} (per frame, every 2 ticks: lower bound)")
            print(f"  our arena-lag (aimed-tick-1) {dict(sorted(st['lag'].items()))}  total p50 {pct(st['tot'], .5) / 1000:.1f} p90 {pct(st['tot'], .9) / 1000:.1f} p99 {pct(st['tot'], .99) / 1000:.1f} ms")
        fi = first_freeze(frames, 0, OWN)
        # the incident freeze = the clip's reason tick
        rt = header['reason']['tick']
        fi = next((i for i, fr in enumerate(frames) if fr['tick'] == rt), fi)
        print('-' * 100)
        print('decisive window (t relative to our freeze onset):')
        for r in window(frames, fi, 64):
            print(' ', r)


if __name__ == '__main__':
    main()
