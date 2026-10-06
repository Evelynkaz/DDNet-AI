#!/usr/bin/env python3
"""E-020: are the games of two `ddnet-ai arena run` directories the same games, bit for bit?  Compares, per file and game, the outcome
(`result`, `end_tick`, `credited`, `blocks_by_a`, `a_self_freezes`, spawns) and, per player, the decision hash and the number of decisions
(the hash covers every input the brain sent).  Timing fields are ignored.

  same_games.py A_DIR B_DIR
"""
import glob
import json
import os
import sys


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        for line in open(f):
            if line.strip():
                v = json.loads(line)
                key = (os.path.basename(f), v["game"])
                out[key] = (
                    v["result"], v["end_tick"], v["credited"], v.get("blocks_by_a"), v.get("a_self_freezes"), json.dumps(v["spawns"]),
                    tuple((p["hash"], p["decisions"]) for p in v["players"]),
                )
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(a) & set(b))
diff = [k for k in common if a[k] != b[k]]
print(f"files/games: {len(a)} vs {len(b)}, common {len(common)}, identical {len(common) - len(diff)}, different {len(diff)}")
for k in diff[:10]:
    print("  differs:", k)
sys.exit(1 if diff or len(a) != len(b) or not common else 0)
