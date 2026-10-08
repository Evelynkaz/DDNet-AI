#!/usr/bin/env python3
"""Task 3.16 (D-115): the slot model of the input lag, as a Monte Carlo over decisions.

  model.py COSTS.jsonl [--rtt 25.2] [--kappa 3.3] [--jitter 2.5] [--extra 0.5] [--n 200000]

A decision made on a snapshot of tick T goes out in the first input slot it can still make; slots are one tick (20 ms) apart. With
  base = RTT + kappa + margin + jitter        (the time from the snapshot's tick to the first slot a free decision could make)
a decision that costs `cost` ms (plus `extra`: queue hop, driver pick-up, the fly proposer) goes out for the tick E = ceil((base + extra + cost) / 20)
after the snapshot's, i.e. the brain rolls its world lag = E - 1 ticks (the arena's `lag`, the live horizon). The bot *plans* for the lag its rolling p90 of the last 64 costs implies and the driver holds an early decision to that
tick, so E = max(planned, ceil(...)); a decision slower than the estimate is applied late (`later`). This is `ddai_env::sim::LagModel`, line for line.

COSTS.jsonl: the lines of `duel_stats --jsonl` with `lag_cost_hist` (bins of 0.25 ms); conditions are pooled by the budget `B<n>` in their name.
Prints, per budget and margin, the share of decisions by effective lag E, the mean E and the share applied later than planned.
"""
import argparse
import json
import math
import random
import re
from collections import defaultdict

BIN = 0.25
TICK = 20.0
WINDOW = 64
QUANTILE = 0.9


def load_costs(path):
    hists = defaultdict(lambda: None)
    for line in open(path):
        if not line.strip():
            continue
        try:
            v = json.loads(line)
        except json.JSONDecodeError:
            continue  # a run that is still going leaves a half-written last line
        m = re.match(r"B(\d+)", v["condition"])
        h = v.get("lag_cost_hist")
        if not m or not h:
            continue
        b = int(m.group(1))
        if hists[b] is None:
            hists[b] = [0] * len(h)
        for i, n in enumerate(h):
            hists[b][i] += n
    return hists


def sampler(hist, rng):
    total = sum(hist)
    cum = []
    s = 0
    for n in hist:
        s += n
        cum.append(s)
    import bisect

    def draw():
        r = rng.random() * total
        i = bisect.bisect_right(cum, r)
        return (min(i, len(hist) - 1) + rng.random()) * BIN

    return draw


def simulate(draw, base, extra, jitter, n, rng, floor=None):
    ring = []
    nxt = 0
    hist = defaultdict(int)
    later = 0
    cut = [0]
    total_cost = 0.0
    for _ in range(n):
        phase = base + (rng.random() * 2 - 1) * jitter
        est = 6.0
        if ring:
            v = sorted(ring)
            est = v[math.ceil((len(v) - 1) * QUANTILE)]
        cost = draw()
        if floor is not None:
            # A deadline-aware search (hypothetical, not built): the bot knows its slot slack, and when the first slot is within reach with at
            # least `floor` ms of decision, the search is cut to end 0.1 ms before it -- and the bot plans for that cut (the estimate is capped by
            # the room), not for the p90 of the past costs.
            slack = TICK * math.ceil(phase / TICK) - phase
            room = slack - extra - 0.1
            if room >= floor:
                est = min(est, room)
                if cost > room:
                    cost = room
                    cut[0] += 1
        planned = max(1, math.ceil((phase + extra + est) / TICK)) - 1
        wanted = max(1, math.ceil((phase + extra + cost) / TICK)) - 1
        lag = max(planned, wanted)
        later += wanted > planned
        hist[lag] += 1
        total_cost += cost
        if len(ring) < WINDOW:
            ring.append(cost)
        else:
            ring[nxt] = cost
            nxt = (nxt + 1) % WINDOW
    return hist, later, cut[0], total_cost / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("costs")
    ap.add_argument("--rtt", type=float, default=25.2)
    ap.add_argument("--kappa", type=float, default=3.3)
    ap.add_argument("--jitter", type=float, default=2.5)
    ap.add_argument("--extra", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--margins", default="4,6,8,10")
    ap.add_argument("--deadline-floor", type=float, default=None, help="hypothetical deadline-aware search: cut a decision to the slot when at least this many ms remain")
    ap.add_argument("--add-ms", type=float, default=0.0, help="added to every cost (e.g. the fly's proposals when the arena has none)")
    a = ap.parse_args()
    rng = random.Random(316)
    hists = load_costs(a.costs)
    print(f"RTT {a.rtt} + kappa {a.kappa} + margin; extra {a.extra} ms; jitter +-{a.jitter} ms; {a.n} decisions per cell")
    print("| budget | margin | lag 0 | lag 1 | lag 2 | lag 3+ | mean lag | later than planned | mean cost ms | cut |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for b in sorted(hists):
        base_draw = sampler(hists[b], rng)
        draw = lambda: base_draw() + a.add_ms
        for m in [float(x) for x in a.margins.split(",")]:
            h, later, cut, mean_cost = simulate(draw, a.rtt + a.kappa + m, a.extra, a.jitter, a.n, rng, a.deadline_floor)
            tot = sum(h.values())
            sh = lambda lo, hi=None: 100 * sum(n for l, n in h.items() if l >= lo and (hi is None or l <= hi)) / tot
            mean = sum(l * n for l, n in h.items()) / tot  # the arena lag = ticks rolled = E - 1
            print(f"| {b} | {m:g} | {sh(0, 0):.1f}% | {sh(1, 1):.1f}% | {sh(2, 2):.1f}% | {sh(3):.1f}% | {mean:.3f} | {100 * later / tot:.1f}% | {mean_cost:.2f} | {100 * cut / tot:.1f}% |")


if __name__ == "__main__":
    main()
