#!/usr/bin/env python3
"""Duel 2026-10-07 post-mortem: cross-clip statistics (JSONL from `ddai-env --example clip_json`).

    python3 stats.py *.jsonl

Prints, per clip, over the frames where both tees are free:
  * position: share of time in the top 3 tiles under the freeze ceiling, on the main floor, in the right pit; jump presses (ours from the sent
    inputs, his from the rising edge of `jumped & 1`);
  * the hammer-safe envelope: share of frames within 100 px where a worst-case hammer hit now (dvy = -11 px/tick, `character.cpp:549-554`) would
    take the tee's ballistic apex above the freeze row; and how many hammer hits landed on such a state;
  * hammer eagerness: frames with the other tee within 56 px and the hammer ready (>= 16 ticks after the last hit, >= 7 after the last swing),
    and the share of them followed by a swing within 4 ticks.
Own id 16, opponent 0 (the five clips of 2026-10-07). The freeze row 168 of "Copy Love Box" ends at y = 5408 px.
"""
import glob
import json
import math
import sys

OWN, OPP = 16, 0
CEIL = 5408
G = 0.5
REACH = 56


def load(p):
    with open(p) as f:
        h = json.loads(f.readline())
        return h, [json.loads(l) for l in f]


def tee(x, i):
    for t in x['tees']:
        if t['id'] == i:
            return t
    return None


def events(x):
    return [list(e.items())[0] for e in x['events'] if isinstance(e, dict)]


def apex_after_hit(y, vy, kick=11.0):
    v = min(vy, 0.0) - kick
    return y - v * v / (2 * G)


def main(paths):
    tot = {'opp': {OWN: 0, OPP: 0}, 'conv': {OWN: 0, OPP: 0}, 'hits': {OWN: 0, OPP: 0}}
    for p in paths:
        h, fr = load(p)
        n = top = {OWN: 0, OPP: 0}
        top = {OWN: 0, OPP: 0}
        floor = {OWN: 0, OPP: 0}
        pit = {OWN: 0, OPP: 0}
        n = 0
        jumps_us = 0
        jumps_him = 0
        prevj = 0
        hj_prev = None
        seen = set()
        near = 0
        unsafe = {OWN: 0, OPP: 0}
        hits = {OWN: 0, OPP: 0}
        hits_unsafe = {OWN: 0, OPP: 0}
        last_fire = {OWN: -999, OPP: -999}
        last_hit = {OWN: -999, OPP: -999}
        opp = {OWN: 0, OPP: 0}
        conv = {OWN: 0, OPP: 0}
        prev = None
        for i, x in enumerate(fr):
            for k, v in events(x):
                if k == 'HammerFire':
                    last_fire[v['from']] = x['tick']
                if k == 'HammerHit':
                    last_hit[v['from']] = x['tick']
            me, op = tee(x, OWN), tee(x, OPP)
            if not me or not op or me['frozen'] or op['frozen']:
                prev = None
                continue
            m, o = me['ch'], op['ch']
            n += 1
            for who, c in ((OWN, m), (OPP, o)):
                top[who] += c['y'] - CEIL < 96
                floor[who] += 5770 <= c['y'] < 5790
                pit[who] += c['y'] > 5800
            for s in x['sent']:
                if s['tick'] in seen:
                    continue
                seen.add(s['tick'])
                j = s['input']['jump']
                jumps_us += bool(j and not prevj)
                prevj = j
            hj = o['jumped'] & 1
            if hj_prev is not None and hj and not hj_prev:
                jumps_him += 1
            hj_prev = hj
            d = math.hypot(m['x'] - o['x'], m['y'] - o['y'])
            if prev is not None:
                pm, po = prev
                for k, v in events(x):
                    if k == 'HammerHit' and v['to'] in (OWN, OPP):
                        victim = v['to']
                        c = pm if victim == OWN else po
                        hits[victim] += 1
                        hits_unsafe[victim] += apex_after_hit(c['y'], c['vel_y'] / 256) < CEIL
            prev = (m, o)
            if d <= 100:
                near += 1
                unsafe[OWN] += apex_after_hit(m['y'], m['vel_y'] / 256) < CEIL
                unsafe[OPP] += apex_after_hit(o['y'], o['vel_y'] / 256) < CEIL
            if d <= REACH:
                for who in (OWN, OPP):
                    if x['tick'] - last_hit[who] >= 16 and x['tick'] - last_fire[who] >= 7:
                        opp[who] += 1
                        conv[who] += any(k == 'HammerFire' and v['from'] == who for y in fr[i + 1:i + 3] for k, v in events(y))
        name = p.split('/')[-1][:26]
        print(f"{name}: free frames {n}")
        print(f"  top-3-tiles us {top[OWN] / n:.0%} him {top[OPP] / n:.0%} | main floor us {floor[OWN] / n:.0%} him {floor[OPP] / n:.0%} | pit us {pit[OWN] / n:.0%} him {pit[OPP] / n:.0%}"
              f" | jump presses us {jumps_us} him {jumps_him}")
        print(f"  within 100 px {near}: outside the hammer-safe envelope us {unsafe[OWN] / max(1, near):.0%} him {unsafe[OPP] / max(1, near):.0%}"
              f" | hits on us {hits[OWN]} ({hits_unsafe[OWN]} on an unsafe state), on him {hits[OPP]} ({hits_unsafe[OPP]})")
        print(f"  hammer ready & in reach: us {opp[OWN]} -> swing within 4 ticks {conv[OWN] / max(1, opp[OWN]):.0%} | him {opp[OPP]} -> {conv[OPP] / max(1, opp[OPP]):.0%}")
        for w in (OWN, OPP):
            tot['opp'][w] += opp[w]
            tot['conv'][w] += conv[w]
            tot['hits'][w] += hits[w]
    print(f"total: hits on us {tot['hits'][OWN]} on him {tot['hits'][OPP]}; eagerness us {tot['conv'][OWN]}/{tot['opp'][OWN]} him {tot['conv'][OPP]}/{tot['opp'][OPP]}")


if __name__ == '__main__':
    main(sys.argv[1:] or sorted(glob.glob('*.jsonl')))
