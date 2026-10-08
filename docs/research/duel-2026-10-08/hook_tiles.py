#!/usr/bin/env python3
"""The "afraid to fight away from hook tiles" hypothesis (2026-10-08 duel): where each tee is, where its rope grabs the map, where the fights
and the blocks happen, by x tile of the box (Copy Love Box f9793b34, box x 79..103; hookable map tiles: the floor corner (91,181), the right wall
(104..105, 176..177) and (85,167), (91,167) above the freeze ceiling; everything else is no-hook `n` or freeze).

usage: hook_tiles.py <jsonl>...
"""
import sys
from collections import Counter
sys.path.insert(0, __import__('os').path.dirname(__file__))
from load import *

HOOKABLE = [(91, 181), (104, 176), (104, 177), (105, 176), (105, 177), (85, 167), (91, 167)]


def near_hookable(x, y, r=380):
    """Distance in px from the tee centre to the nearest hookable tile centre."""
    return min(((x - (tx * 32 + 16)) ** 2 + (y - (ty * 32 + 16)) ** 2) ** 0.5 for tx, ty in HOOKABLE)


def main():
    frames, _ = merged(sys.argv[1:])
    ticks = sorted(frames)
    anchors = {OWN: Counter(), OPP: Counter()}
    ground_frames = {OWN: 0, OPP: 0}
    region = {OWN: Counter(), OPP: Counter()}
    dnear = {OWN: [], OPP: []}
    n = 0
    for t in ticks:
        f = frames[t]
        me, op = tee(f, OWN), tee(f, OPP)
        if not me or not op or me['frozen'] or op['frozen'] or 38373276 <= t < 38376804:
            continue
        rn = round_of(t)
        if rn is None or t < LIVES[rn - 1] + 150:
            continue
        n += 1
        for who, a in ((OWN, me), (OPP, op)):
            ch = a['ch']
            if ch['hook_state'] == 5 and ch['hooked_player'] == -1:
                ground_frames[who] += 1
                anchors[who][(ch['hook_x'] // 32, ch['hook_y'] // 32)] += 1
            x = ch['x'] // 32
            region[who]['left x<=84' if x <= 84 else ('mid 85..90' if x <= 90 else 'right >=91')] += 1
            dnear[who].append(near_hookable(ch['x'], ch['y']))
    print(f'both-free frames {n}')
    for who, name in ((OWN, 'us'), (OPP, 'human')):
        d = sorted(dnear[who])
        print(f"{name:5s}: rope on the map {ground_frames[who]}/{n} = {100 * ground_frames[who] / n:.1f}% of frames, anchors {dict(anchors[who].most_common(6))}; "
              f"region {dict((k, f'{100 * v / n:.0f}%') for k, v in region[who].items())}; distance to the nearest hookable tile p50 {d[len(d) // 2]:.0f} px, "
              f"share within 380 px {100 * sum(x <= 380 for x in d) / len(d):.0f}%")


if __name__ == '__main__':
    main()
