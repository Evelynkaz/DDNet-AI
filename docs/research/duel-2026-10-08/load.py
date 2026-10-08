#!/usr/bin/env python3
"""Shared loader for the 2026-10-08 duel post-mortem: merges the frames of overlapping clips by tick.

Input: JSONL from `clip_json` (line 1 header, then frames). Frames of one tick that several clips hold are taken once (the first file wins).
"""
import json
import math
import glob
import os

OWN = 4
OPP = 0
DUEL_START = 38373276          # journal: "duel: an F-DDrace 1vs1 is on" (15:22:56 host time)
DUEL_END = 38391610            # journal: "duel: the 1vs1 is over"
# Journal "life started" ticks inside the duel = round starts (each respawn is frozen 150 ticks by the arena).
LIVES = [38373276, 38377564, 38378326, 38379232, 38380930, 38381656, 38382554, 38383018, 38383456, 38384102,
         38384366, 38385526, 38386674, 38387720, 38388106, 38388810, 38389292, 38390354, 38391054, 38391512]
# Outcome by round number (1-based), from the journal ("block held ... died=true" + next life = win; duel-loss clip = loss).
OUTCOME = {1: 'L', 2: 'W', 3: 'W', 4: 'W', 5: 'L', 6: 'L', 7: 'W', 8: 'W', 9: 'W', 10: 'W', 11: 'L', 12: 'L', 13: 'L',
           14: 'L', 15: 'L', 16: 'L', 17: 'W', 18: 'W', 19: 'L'}
FLAGS = {1: 'S', 2: 'T', 4: 'H', 8: 'h', 16: 'G', 32: 'vh', 64: 'vf', 128: 'W', 256: 'X', 512: 'P', 1024: 'B', 2048: 'D'}


def load_clip(path):
    with open(path) as f:
        header = json.loads(f.readline())
        frames = [json.loads(l) for l in f]
    return header, frames


def merged(paths):
    """tick -> frame (first file wins), plus the per-file headers."""
    out = {}
    headers = {}
    for p in sorted(paths):
        h, frames = load_clip(p)
        headers[os.path.basename(p)] = h
        for fr in frames:
            out.setdefault(fr['tick'], fr)
    return out, headers


def tee(fr, i):
    for t in fr['tees']:
        if t['id'] == i:
            return t
    return None


def evs(fr):
    out = []
    for e in fr['events']:
        if isinstance(e, dict):
            (k, v), = e.items()
            out.append((k, v))
        else:
            out.append((e, {}))
    return out


def vel(t):
    return t['ch']['vel_x'] / 256.0, t['ch']['vel_y'] / 256.0


def dist(a, b):
    return math.hypot(a['ch']['x'] - b['ch']['x'], a['ch']['y'] - b['ch']['y'])


def round_of(tick):
    for k in range(len(LIVES) - 1):
        if LIVES[k] <= tick < LIVES[k + 1]:
            return k + 1
    return None


def flagstr(f):
    return ''.join(v for k, v in FLAGS.items() if f & k)


def tile(t):
    return t['ch']['x'] // 32, t['ch']['y'] // 32
