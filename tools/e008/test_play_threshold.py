#!/usr/bin/env python3
"""Unit test of the threshold search of play_threshold.py on synthetic monotone start-rate curves (python3 test_play_threshold.py)."""
import math
from play_threshold import next_threshold, start_rates


def search(curve, first=0.5, tol=0.03, max_evals=5):
    pts = []
    t = first
    while t is not None and len(pts) < max_evals:
        d = curve(t)
        pts.append((t, d))
        if abs(d) <= tol:
            break
        t = next_threshold(pts)
    return pts


def logistic(t0, width, teacher=0.3):
    return lambda t: 1 / (1 + math.exp((t - t0) / width)) - teacher

# a model that over-starts (like the MLP-w arms: 92 percent at 0.5 against a teacher at 20) and one that under-starts (the fly)
for name, c in (('over', logistic(0.62, 0.05, 0.25)), ('under', logistic(0.3, 0.08, 0.5)), ('flat', logistic(0.5, 0.3, 0.4))):
    pts = search(c)
    t, d = min(pts, key=lambda p: abs(p[1]))
    assert abs(d) <= 0.05, (name, pts)
    assert len(pts) <= 5
# start/teacher never meet inside [0.05, 0.95]: the search stops at the end of the range instead of looping
pts = search(lambda t: 0.5, max_evals=8)
assert all(0.05 <= t <= 0.95 for t, _ in pts) and len(pts) <= 8 and pts[-1][0] in (0.95,), pts
# pooled rates
hp = [{'counts': {'not_out': {'n': 100, 'student_hook': 30, 'teacher_hook': 20}}}, {'counts': {'not_out': {'n': 100, 'student_hook': 10, 'teacher_hook': 40}}}]
assert start_rates(hp) == (0.2, 0.3, 200)
print('ok')
