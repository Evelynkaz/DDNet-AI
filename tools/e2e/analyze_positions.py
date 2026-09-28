#!/usr/bin/env python3
"""tools/e2e/analyze_positions.py — task 2.3 e2e scenario (b): parses `ddnet-ai play`'s log for
"own position" lines (tracing output, `tick=... x=... y=...`) and checks that the tee's own
position genuinely moved right then left (`--brain circle`'s job) rather than staying put.

Real DDNet 20.1 physics on a real block map is used here (`Copy Love Box`), not a synthetic
fixture, so this is deliberately tolerant of things a real map does that a synthetic test would
never need to handle: long flat stretches (the tee touched a freeze tile briefly and could not
move; `--brain circle` does not avoid freeze at all, on purpose — proving inputs move the tee does
not require avoiding every hazard a real block map throws at a careless brain) and jumps/speedups
producing large single-step position deltas. The check is intentionally about the *shape* of the
trace (a genuine rise then a genuine fall by at least `--margin` units), not an exact trajectory.

Review finding F12: also checks that the *y* coordinate genuinely changed and came back (either
direction — see `find_extremum`'s doc comment for why this script does not hard-code which sign
convention "up" uses) by at least `--y-margin` units, evidence of the jump `--brain circle` also
issues twice per cycle, not just the x movement the original check already covered.
"""

import argparse
import re
import sys

LINE_RE = re.compile(r"own position.*?tick=(-?\d+).*?x=(-?\d+).*?y=(-?\d+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse(path):
    samples = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = ANSI_RE.sub("", line)
            m = LINE_RE.search(line)
            if m:
                samples.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return samples


def find_extremum(values, margin):
    """Returns `(index, kind)` of any genuine local extremum ("peak" or "trough") of at least
    `margin` units — a value with some strictly earlier sample at least `margin` away on one side
    and some strictly later sample at least `margin` away on the *same* side. Same "shape of the
    trace, not an exact trajectory" spirit as the x-only rise-then-fall check this generalizes
    (review finding F12): a jump moves the tee up then back down, but whether "up" is a smaller or
    larger y value is an engine-coordinate-convention detail this script deliberately does not
    need to hard-code — a peak (value stands out *above* both neighbourhoods) or a trough (value
    stands out *below* both) are equally good evidence of "genuinely went one way and came back".
    Returns `(None, None)` if no such point exists.
    """
    n = len(values)
    prefix_min = [None] * n
    prefix_max = [None] * n
    running_min = running_max = None
    for i in range(n):
        prefix_min[i] = running_min
        prefix_max[i] = running_max
        running_min = values[i] if running_min is None else min(running_min, values[i])
        running_max = values[i] if running_max is None else max(running_max, values[i])
    suffix_min = [None] * n
    suffix_max = [None] * n
    running_min = running_max = None
    for i in range(n - 1, -1, -1):
        suffix_min[i] = running_min
        suffix_max[i] = running_max
        running_min = values[i] if running_min is None else min(running_min, values[i])
        running_max = values[i] if running_max is None else max(running_max, values[i])

    for j in range(n):
        if prefix_min[j] is None or suffix_min[j] is None:
            continue
        if values[j] - prefix_min[j] >= margin and values[j] - suffix_min[j] >= margin:
            return j, "peak"
        if prefix_max[j] - values[j] >= margin and suffix_max[j] - values[j] >= margin:
            return j, "trough"
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="ddnet-ai play log file to analyze")
    parser.add_argument("--margin", type=int, default=80, help="minimum units of rise/fall to count as real movement")
    parser.add_argument(
        "--y-margin",
        type=int,
        default=20,
        help="minimum units of y change (either direction) to count as real vertical movement (a jump)",
    )
    parser.add_argument("--min-samples", type=int, default=10, help="minimum number of position samples required")
    args = parser.parse_args()

    samples = parse(args.log)
    print(f"parsed {len(samples)} position samples from {args.log}")
    if len(samples) < args.min_samples:
        print(f"FAIL: too few position samples ({len(samples)} < {args.min_samples})")
        return 1

    xs = [x for _, x, _ in samples]
    x_min, x_max = min(xs), max(xs)
    print(f"x range: min={x_min} max={x_max} spread={x_max - x_min}")

    # Find *any* index j that is a genuine local peak: some strictly earlier sample at least
    # `margin` below it (a real rightward rise into j), and some strictly later sample at least
    # `margin` below it (a real leftward fall out of j) — deliberately not tied to the single
    # global maximum, since a multi-cycle `--brain circle` trace has several such peaks and the
    # global max may simply be the last one the log happened to end on (no "after" data left to
    # show the fall) without that meaning no reversal ever happened.
    n = len(xs)
    prefix_min = [None] * n
    running = None
    for i in range(n):
        prefix_min[i] = running
        running = xs[i] if running is None else min(running, xs[i])
    suffix_min = [None] * n
    running = None
    for i in range(n - 1, -1, -1):
        suffix_min[i] = running
        running = xs[i] if running is None else min(running, xs[i])

    peak_idx = None
    for j in range(n):
        if prefix_min[j] is None or suffix_min[j] is None:
            continue
        if xs[j] - prefix_min[j] >= args.margin and xs[j] - suffix_min[j] >= args.margin:
            peak_idx = j
            break

    if peak_idx is None:
        print(f"FAIL: no rise-then-fall of >= {args.margin} units found anywhere in the trace")
        return 1

    ys = [y for _, _, y in samples]
    y_min, y_max = min(ys), max(ys)
    print(f"y range: min={y_min} max={y_max} spread={y_max - y_min}")

    y_idx, y_kind = find_extremum(ys, args.y_margin)
    if y_idx is None:
        print(f"FAIL: no y-axis change of >= {args.y_margin} units found anywhere in the trace (expected a jump)")
        return 1

    print(f"PASS: found a rightward rise then a leftward fall of >= {args.margin} units (peak x={xs[peak_idx]} at sample {peak_idx})")
    print(f"PASS: found a y-axis {y_kind} of >= {args.y_margin} units (y={ys[y_idx]} at sample {y_idx}, evidence of a jump)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
