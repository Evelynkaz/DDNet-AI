#!/usr/bin/env python3
"""E-029 (task 8.6): the tuning table of the hysteresis decode on the TRAIN-VAL starts and games only.

Reads `train es eval` files made with `configs/train/e029-tune.toml` (no holdout arena: the holdout is never played while tuning) and prints, per file:
held on V (44 starts), held on V+B (the pilot's "escapable", 145 starts), own freeze (208), first freeze credited (400 games) and the
pre-registered selection score `held(V+B) - own_freeze + credited`. Usage: tune_table.py a.json b.json ...
"""
import json
import sys


def row(path):
    label, p = json.load(open(path))
    s = p["train_starts"]
    g = p["train_games"]
    assert p.get("holdout_starts") is None, "a tuning file must not contain the holdout"
    v = s["held_victim_escapable"]
    score = s["held_escapable"]["p"] - s["self_freeze"]["p"] + g["credited"]["p"]
    return (path.split("/")[-1].removesuffix(".json"), v["p"], v["k"], v["n"], s["held_escapable"]["p"],
            s["self_freeze"]["p"], g["credited"]["p"], score)


if __name__ == "__main__":
    print(f"{'file':28s} {'V held':>8s} {'V+B held':>9s} {'own frz':>8s} {'1st frz':>8s} {'score':>7s}")
    for f in sys.argv[1:]:
        n, vp, vk, vn, esc, sf, cr, sc = row(f)
        print(f"{n:28s} {100*vp:6.1f}% {100*esc:8.1f}% {100*sf:7.1f}% {100*cr:7.1f}% {sc:7.3f}")
