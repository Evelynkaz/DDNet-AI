#!/usr/bin/env python3
"""E-025 (task 3.13): writes the arena configs of the duel experiments `configs/arena/e025-*.toml`.

  gen_configs.py            # all files
Arms are `[hybrid]` knob sets; every arm plays the same games (game g of an arena = seed base + g, layout g) against the competitor's
`live-v2` planner (`v2live`) on three arenas: the two halls and the full map from the real spawn ledges.
"""
import argparse
import os

FLY = "~/aiddnet/data/runs/E-005/e005-fly/checkpoints/final.bundle"  # what production `hybrid-fly` plays (launch.toml default)
FLY8 = "~/aiddnet/data/runs/E-008/e008-p2-fly-d2-maskhook-s2/checkpoints/selected.bundle"  # E-008 best, two views (hook head masked)

# name -> (hybrid knobs, rules override)
ARMS = {
    "main": ({}, {}),
    "mirlive": ({"mirror_preset": '"live-v2"'}, {}),
    "mirnv2c": ({"mirror_preset": '"normal-v2"', "rope_ceiling_cost": "1.0"}, {}),
    "liveW": ({"launch_exposure": "1.5", "jumpless_hazard_cost": "0.4"}, {}),
    "liveW-mirlive": ({"launch_exposure": "1.5", "jumpless_hazard_cost": "0.4", "mirror_preset": '"live-v2"'}, {}),
    "hold": ({}, {"hold_target": "true"}),
    "fly": ({"proposer": '"fly"', "fly_model": f'"{FLY}"'}, {}),
    "fly8": ({"proposer": '"fly"', "fly_model": f'"{FLY8}"'}, {}),
}


def player(knobs):
    base = 'brain = "hybrid", mode = "deadline", clock = "work", budget_ms = 4.0'
    if knobs:
        base += ", hybrid = { " + ", ".join(f"{k} = {v}" for k, v in knobs.items()) + " }"
    return "{ " + base + " }"


def condition(arena, arm, knobs, rules, opp, label_opp, ai_first=True):
    out = [f'[[condition]]\nname = "{arena}: {arm} vs {label_opp}"\narena = "{arena}"']
    if rules:
        out.append("rules = { " + ", ".join(f"{k} = {v}" for k, v in rules.items()) + " }")
    out.append(f"players = [{player(knobs)}, {opp}]")
    return "\n".join(out) + "\n"


def write(path, header, name, seed, games, arenas, arms, opp, label_opp, after=250, extra_rules=""):
    text = [header.rstrip("\n"), f'name = "{name}"', f"base_seed = {seed}", f"games = {games}", "",
            f"[rules]\nafter_ticks = {after}\n{extra_rules}"]
    for arm in arms:
        for a in arenas:
            knobs, rules = ARMS[arm]
            text.append(condition(a, arm, knobs, rules, opp, label_opp))
    with open(path, "w") as f:
        f.write("\n".join(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--arms", required=True, help="comma-separated arm names (order = run order)")
    ap.add_argument("--arenas", default="clb-left,clb-right")
    ap.add_argument("--opp", default="live-v2", help="planner preset of the opponent")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "configs", "arena")
    opp = f'{{ brain = "planner", mode = "fixed", preset = "{a.opp}" }}'
    label = {"live-v2": "v2live", "v2-strong-wb": "v2strongwb"}.get(a.opp, a.opp)
    arenas = a.arenas.split(",")
    hdr = (f"# work-clock: default-rate\n# Task 3.13 (E-025): {a.note} 1v1 against the competitor's planner `{a.opp}` on {', '.join(arenas)}.\n"
           f"# {a.games} games per arena and arm from seed {a.seed}, `after_ticks = 250`. Arms: docs/research/e025/gen_configs.py "
           f"(`{a.tag} --seed {a.seed} --games {a.games} --arms {a.arms} --arenas {a.arenas} --opp {a.opp}`).\n"
           f"#   ddnet-ai arena run --config configs/arena/e025-{a.tag}.toml --out ~/aiddnet/data/runs/E-025/{a.tag} --threads 3\n")
    write(os.path.join(root, f"e025-{a.tag}.toml"), hdr, f"E-025: {a.tag}", a.seed, a.games, arenas, a.arms.split(","), opp, label)


main()
