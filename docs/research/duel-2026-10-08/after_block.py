#!/usr/bin/env python3
""""After the block": what the blocker does from the victim's freeze onset until the victim thaws or the round ends (2026-10-08 duel).

usage: after_block.py <box.txt> <jsonl>...
For every freeze onset after the countdown (both directions) prints: how long the victim stayed frozen, where it came to rest (freeze tile or open
floor), whether the round ended with it lying there, and the blocker's actions: share of frames its hook holds the victim, hammer hits on the
frozen victim (a hit unfreezes, `character.cpp:554`), share of frames the blocker is idle (no movement, for us also no input), median distance.
"""
import sys
import statistics
sys.path.insert(0, __import__('os').path.dirname(__file__))
from load import *
import rounds as R


def main():
    R.BOX = R.load_box(sys.argv[1])
    frames, _ = merged(sys.argv[2:])
    ticks = sorted(frames)
    rows = []
    for who in (OWN, OPP):
        other = OPP if who == OWN else OWN
        for t0 in R.onsets(frames, ticks, who):
            rn = round_of(t0)
            if rn is None or t0 < LIVES[rn - 1] + 152:
                continue
            end_round = LIVES[rn]
            w = [t for t in ticks if t0 <= t < end_round and tee(frames[t], who) and tee(frames[t], other)]
            thaw = next((t for t in w if not tee(frames[t], who)['frozen']), None)
            stop = thaw if thaw else end_round
            span = [t for t in w if t < stop]
            n = len(span)
            hook = sum(1 for t in span if tee(frames[t], other)['ch']['hooked_player'] == who)
            hits = sum(1 for t in span for k, v in evs(frames[t]) if k == 'HammerHit' and v['from'] == other and v['to'] == who)
            fires = sum(1 for t in span for k, v in evs(frames[t]) if k == 'HammerFire' and v['from'] == other)
            d = [dist(tee(frames[t], who), tee(frames[t], other)) for t in span]
            # blocker idle: speed < 0.5 px/tick and (for us) no direction/jump/hook input
            idle = 0
            for t in span:
                b = tee(frames[t], other)
                vx, vy = vel(b)
                still = abs(vx) < 0.5 and abs(vy) < 0.5
                if other == OWN:
                    ins = frames[t]['sent']
                    still = still and all(s['input']['direction'] == 0 and not s['input']['jump'] and not s['input']['hook'] for s in ins)
                idle += still
            blocker_frozen = sum(1 for t in span if tee(frames[t], other)['frozen'])
            # where did the victim come to rest: first frame from which it stays still on ground in a freeze tile until stop
            rest_in_freeze = None
            for t in span:
                v = tee(frames[t], who)
                vx, vy = vel(v)
                if abs(vx) < 0.3 and abs(vy) < 0.3 and R.freeze_tile_near(v):
                    rest_in_freeze = t
                    break
            last = tee(frames[span[-1]], who) if span else None
            fall_floor = None
            for t in span:
                v = tee(frames[t], who)
                if v['ch']['y'] >= 5770 and abs(vel(v)[1]) < 0.5:
                    fall_floor = t
                    break
            outcome = 'thawed (escaped)' if thaw else ('round ended (victim lost)' if OUTCOME[rn] == ('L' if who == OWN else 'W') else 'round ended (other)')
            rows.append((who, rn, t0, n, outcome))
            print(f"{'US ' if who == OWN else 'HIM'} frozen r{rn:2d} at {t0} (+{t0 - LIVES[rn - 1]}), frozen frames {n} ({2 * n} ticks), {outcome}; blocker {'him' if other == OPP else 'us'}: "
                  f"hook on victim {hook}/{n} ({100 * hook / max(1, n):.0f}%), hammer hits on frozen victim {hits} (swings {fires}), idle {idle}/{n} ({100 * idle / max(1, n):.0f}%), "
                  f"blocker frozen {blocker_frozen}/{n}, dist p50 {statistics.median(d) if d else 0:.0f} px; victim on the floor from {('+' + str(fall_floor - t0)) if fall_floor else '-'}, "
                  f"resting in freeze from {('+' + str(rest_in_freeze - t0)) if rest_in_freeze else '-'}; victim last pos {(last['ch']['x'], last['ch']['y']) if last else None} {R.where(R.freeze_tile_near(last)) if last else ''}")


if __name__ == '__main__':
    main()
