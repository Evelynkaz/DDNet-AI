#!/usr/bin/env python3
"""Builds one crates/ddai-trace/tests/fixtures/*.json golden fixture from a trace v1 file
produced by Oracle A and the per-tick hashes JSON produced by `ddnet-ai trace hashes`.

Not part of the Rust or C++ parity code itself — a maintenance helper invoked by
tools/ddnet-oracle/gen_fixtures.sh. See docs/formats.md for the trace v1 byte layout this
script's `read_trace` mirrors (read-only, independently of both the Rust and C++ implementations,
as a small cross-check that all three agree on the format).
"""

import argparse
import json
import struct

STATE_F32_FIELDS = [
    "pos_x", "pos_y", "vel_x", "vel_y", "hook_pos_x", "hook_pos_y", "hook_dir_x", "hook_dir_y",
    "hook_tele_base_x", "hook_tele_base_y",
]
STATE_I32_FIELDS = [
    "hook_tick", "hook_state", "hooked_player", "active_weapon", "new_hook", "jumped",
    "jumped_total", "jumps", "direction", "angle", "triggered_events", "colliding", "left_wall",
    "move_restrictions", "solo", "collision_disabled", "endless_hook", "hook_hit_disabled",
]
INPUT_FIELDS = [
    "direction", "target_x", "target_y", "jump", "fire", "hook", "player_flags",
    "wanted_weapon", "next_weapon", "prev_weapon",
]


def read_trace(path):
    data = open(path, "rb").read()
    assert data[:4] == b"TRC1", f"bad magic in {path}"
    off = 4
    (version,) = struct.unpack_from("<I", data, off)
    off += 4
    assert version == 1, f"unsupported trace version {version}"
    (meta_len,) = struct.unpack_from("<I", data, off)
    off += 4
    metadata = json.loads(data[off : off + meta_len].decode("utf-8"))
    off += meta_len
    (nchar,) = struct.unpack_from("<I", data, off)
    off += 4
    ids = list(struct.unpack_from(f"<{nchar}I", data, off))
    off += 4 * nchar
    (nticks,) = struct.unpack_from("<I", data, off)
    off += 4

    row_size = 10 * 4 + len(STATE_F32_FIELDS) * 4 + len(STATE_I32_FIELDS) * 4
    rows = []
    for _t in range(nticks):
        tick_rows = []
        for _c in range(nchar):
            inp = struct.unpack_from("<10i", data, off)
            state_off = off + 40
            f32s = struct.unpack_from(f"<{len(STATE_F32_FIELDS)}f", data, state_off)
            i32s = struct.unpack_from(
                f"<{len(STATE_I32_FIELDS)}i", data, state_off + len(STATE_F32_FIELDS) * 4
            )
            tick_rows.append(
                {
                    "input": dict(zip(INPUT_FIELDS, inp)),
                    "state": dict(zip(STATE_F32_FIELDS + STATE_I32_FIELDS, list(f32s) + list(i32s))),
                }
            )
            off += row_size
        rows.append(tick_rows)
    return metadata, ids, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ticks", type=int, required=True)
    p.add_argument("--chars", type=int, required=True)
    p.add_argument("--scenario-sha256", required=True)
    p.add_argument("--trace", required=True)
    p.add_argument("--hashes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--no-weak-hook",
        action="store_true",
        help="record that this fixture's scenario was generated with --no-weak-hook (review round 1, finding F8)",
    )
    p.add_argument(
        "--tune",
        action="append",
        default=[],
        metavar="NAME=VALUE_X100",
        help="record a tuning override this fixture's scenario was generated with (repeatable)",
    )
    args = p.parse_args()

    tuning_overrides = []
    for spec in args.tune:
        name, value = spec.split("=", 1)
        tuning_overrides.append({"name": name.strip(), "value_x100": int(value.strip())})

    metadata, ids, rows = read_trace(args.trace)
    assert len(rows) == args.ticks, f"trace has {len(rows)} ticks, expected {args.ticks}"
    assert len(ids) == args.chars, f"trace has {len(ids)} characters, expected {args.chars}"

    tick_hashes = json.load(open(args.hashes))["tick_hashes"]
    assert len(tick_hashes) == args.ticks

    fixture = {
        "recipe": args.recipe,
        "seed": args.seed,
        "ticks": args.ticks,
        "characters": args.chars,
        "generator": "random-v1",
        # `no_weak_hook`/`tuning_overrides` — always present (default `false`/`[]`) so every
        # fixture has the same shape; review round 1, finding F8 added the ability for these to
        # be non-default, applied on top of the regenerated scenario the same way `ddnet-ai
        # trace gen-scenario --no-weak-hook --tune ...` does (see fixtures_test.rs).
        "no_weak_hook": args.no_weak_hook,
        "tuning_overrides": tuning_overrides,
        "scenario_sha256": args.scenario_sha256,
        "map_sha256": metadata["map_sha256"],
        "character_ids": ids,
        "tick_hashes": tick_hashes,
        "final_tick": {
            "tick": args.ticks - 1,
            "rows": rows[-1],
        },
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
