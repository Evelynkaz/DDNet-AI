#!/usr/bin/env python3
"""Us vs the human over the merged duel frames where both are free (the AFK part of round 1 excluded): hammer, hook, jumps, eagerness,
the hammer-safe envelope, positions. Same definitions as docs/research/duel-2026-10-07/stats.py (own id 4 here).

usage: fight_stats.py <jsonl>...
"""
import sys
import math
sys.path.insert(0, __import__('os').path.dirname(__file__))
from load import *

CEIL = 5408
G = 0.5
REACH = 56
AFK_END = 38376804


def main():
    frames, _ = merged(sys.argv[1:])
    ticks = sorted(frames)
    n = 0
    hits = {OWN: 0, OPP: 0}
    swings = {OWN: 0, OPP: 0}
    hookf = {OWN: 0, OPP: 0}
    holds = {OWN: [], OPP: []}
    jumps = {OWN: 0, OPP: 0}
    last_hit = {OWN: -10**9, OPP: -10**9}
    last_sw = {OWN: -10**9, OPP: -10**9}
    opp_frames = {OWN: [], OPP: []}   # (tick) frames: other within REACH and own hammer ready
    swing_ticks = {OWN: [], OPP: []}
    unsafe = {OWN: 0, OPP: 0}
    near = 0
    unsafe_hit = {OWN: 0, OPP: 0}
    above = 0
    xs = {OWN: [], OPP: []}
    prevj = {OWN: None, OPP: None}
    lastsent = 0
    prev_t = None
    for t in ticks:
        f = frames[t]
        me, op = tee(f, OWN), tee(f, OPP)
        for k, v in evs(f):
            if k == 'HammerHit' and v['from'] in hits:
                last_hit[v['from']] = t
            if k == 'HammerFire' and v['from'] in swings:
                last_sw[v['from']] = t
                swing_ticks[v['from']].append(t)
        if not me or not op or me['frozen'] or op['frozen'] or t < AFK_END - 0 and t < 38377564 and t >= 38373276 and t < AFK_END:
            prevj = {OWN: None, OPP: None}
            continue
        rn = round_of(t)
        if rn is None or t < LIVES[rn - 1] + 150:
            continue
        n += 1
        d = dist(me, op)
        for who, a, b in ((OWN, me, op), (OPP, op, me)):
            for k, v in evs(f):
                if k == 'HammerHit' and v['from'] == who and v['to'] in (OWN, OPP):
                    hits[who] += 1
                    # was the victim outside the envelope?
                    vy = vel(b)[1]
                    up = max(0.0, -vy) + 11
                    if b['ch']['y'] - up * up / (2 * G) < CEIL + 16:
                        unsafe_hit[who] += 1
                if k == 'HammerFire' and v['from'] == who:
                    swings[who] += 1
                if k == 'HookRelease' and v['id'] == who and v['target'] in (OWN, OPP):
                    holds[who].append(v['held'])
            if a['ch']['hooked_player'] == (OPP if who == OWN else OWN):
                hookf[who] += 1
            if d <= REACH and t - last_hit[who] >= 16 and t - last_sw[who] >= 7:
                opp_frames[who].append(t)
            xs[who].append(a['ch']['x'] // 32)
            if d <= 100:
                vy = vel(a)[1]
                up = max(0.0, -vy) + 11
                if a['ch']['y'] - up * up / (2 * G) < CEIL + 16:
                    unsafe[who] += 1
        if d <= 100:
            near += 1
        if op['ch']['y'] < me['ch']['y'] - 16:
            above += 1
        # jumps: ours from sent inputs (rising edge), his from jumped&1 rising edge
        for s in f['sent']:
            j = s['input']['jump']
            if j and not lastsent:
                jumps[OWN] += 1
            lastsent = j
        j = op['ch']['jumped'] & 1
        if prevj[OPP] is not None and j and not prevj[OPP]:
            jumps[OPP] += 1
        prevj[OPP] = j
    secs = n * 2 / 50
    print(f'both-free frames {n} ({secs:.0f} s of fight)')
    for who, name in ((OWN, 'us'), (OPP, 'human')):
        eager = sum(1 for t in opp_frames[who] if any(0 < s - t <= 4 for s in swing_ticks[who]))
        hs = sorted(holds[who])
        print(f'{name:5s}: hammer hits {hits[who]} ({hits[who] * 30 / secs:.1f}/30 s), swings {swings[who]} ({swings[who] * 30 / secs:.1f}/30 s); '
              f'eagerness {eager}/{len(opp_frames[who])} = {100 * eager / max(1, len(opp_frames[who])):.0f}%; '
              f'hook on the other {hookf[who]}/{n} = {100 * hookf[who] / n:.0f}% of frames, median hold {hs[len(hs) // 2] if hs else 0} ticks, holds {len(hs)}; '
              f'jumps {jumps[who] if who == OWN else jumps[who]} ({jumps[who] * 30 / secs:.0f}/30 s, ours = presses, his = executed); '
              f'outside the envelope within 100 px {unsafe[who]}/{near} = {100 * unsafe[who] / max(1, near):.0f}%; hits it landed on such a state {unsafe_hit[who]}/{hits[who]}')
    print(f'he is above us (by >16 px) in {above}/{n} = {100 * above / n:.0f}% of frames')
    for who, name in ((OWN, 'us'), (OPP, 'human')):
        c = {}
        for x in xs[who]:
            c[x] = c.get(x, 0) + 1
        print(name, 'x tile histogram:', ' '.join(f'{x}:{100 * c[x] / len(xs[who]):.0f}' for x in sorted(c)))


if __name__ == '__main__':
    main()
