#!/usr/bin/env python3
"""Task 3.16 (E-032): can the lag drop to 1? The share of decisions that reach lag <= 1 (the input goes out for the 2nd tick after the snapshot).

  reach.py [--n 200000]

Lag <= 1 needs  RTT + kappa + margin + jitter + extra + cost <= 40 ms.  kappa (the part of the snapshot-to-slot phase that is not the round trip or the
margin) is the one number that differs between the stand (3.3 ms, `tools/e2e/lag_shave.sh` with the scripted brain) and the live link (a fit of the
first-slot shares of two live sessions says 6-8 ms); the table is for a range of it. Costs (queue + brain + pick-up, ms): quiet = p50 4.4 / p90 5.2,
loaded (the duel-test box: load 18-30) = p50 4.5 / p90 5.6 with 5% of decisions +6 ms (`pause tail`); `B3` 1 ms less.
"""
import argparse
import random

random.seed(316)
COSTS = {
    "B4 quiet": (4.4, 5.2, 0.0),
    "B4 loaded": (4.54, 5.58, 0.05),
    "B3 quiet": (3.4, 4.2, 0.0),
    "B3 loaded": (3.54, 4.6, 0.05),
    "B2 quiet": (2.6, 3.3, 0.0),
}


def share(margin, kappa, cost, n, rtt=25.2, jitter=2.5):
    med, p90, tail = cost
    sd = (p90 - med) / 1.28
    ok = 0
    for _ in range(n):
        base = rtt + kappa + margin + (random.random() * 2 - 1) * jitter
        r = random.gauss(med, sd) if random.random() > tail else med + random.expovariate(1 / 6.0)
        ok += base + max(r, 0.3) <= 40.0
    return 100.0 * ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    a = ap.parse_args()
    margins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print("| kappa, ms | costs | " + " | ".join(f"m={m}" for m in margins) + " |")
    print("|---|---|" + "---|" * len(margins))
    for kappa in (3.3, 5.0, 6.5, 8.0):
        for name, cost in COSTS.items():
            print(f"| {kappa} | {name} | " + " | ".join(f"{share(m, kappa, cost, a.n):.0f}%" for m in margins) + " |")


if __name__ == "__main__":
    main()
