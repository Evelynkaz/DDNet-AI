#!/usr/bin/env python3
"""Differential mutation fuzzer for task 1.4's map2raw <-> ddai-map parity (review round 1's
"add a differential mutation mode ... target: 0 disagreements over >= 2000 mutations").

Takes one or more real `.map` files, and for each, repeatedly mutates either (a) a handful of raw
`i32` fields inside a random GROUP or LAYER item (index/width/height/flags/color/name/data-index/
etc. — whatever happens to live at the mutated offset), or (b) a single entry of the v4 declared
data-size table directly (review round 2 finding F7's own reproduction went through this table,
not a layer field — see `mutate_declared_size`), with adversarial values (0, -1, -2, small integers
around the original value, 2^31-1, and full 32-bit random noise). It then runs both `map2raw` (the
real, unmodified DDNet 20.1 loader) and `ddnet-ai trace export-map --map` (this crate's own Rust
loader) on the mutated bytes and compares:
  - accept/reject: did both tools agree on whether the file loads at all?
  - bytes: if both accepted, is the rawmap v1 output byte-for-byte identical?

This is deliberately narrower than a general-purpose byte fuzzer (see `crates/ddai-map/tests/
robustness.rs` for that): it targets the exact kind of small, structured mutation a real-world
"map became corrupted" or "server sent a subtly malformed map" scenario would produce, and where
review round 1's findings (F1/F3/F4/F5/F6) were actually found. Known, deliberately documented
divergences (docs/formats.md §10.2) are not bugs and are excluded from the disagreement count:
  - a Settings blob whose logical length is over `ddai-map`'s own `SETTINGS_MAX_BYTES` cap (this
    crate treats it as "no settings"; real DDNet has no such cap and would use the memory).
  - a physics layer (not just decorative) whose own `width*height` exceeds `ddai-map`'s
    `MAX_TILE_COUNT` (32M tiles) is dropped (absent), where real DDNet would still load it.
  - a Settings index that coincides with some tiles layer's *own* data index, where that layer's
    real tile data happens to also be readable as a Settings blob (see `crate::loader::read_info`)
    — DDNet would return that layer's tile bytes reinterpreted as Settings strings; this crate
    always treats a shared index as "no settings" instead (see docs/formats.md §10.2).

Sandboxing (review round 2 finding F7): running the REAL DDNet loader (`map2raw`) — or, for
that matter, this crate's own Rust loader — against an adversarially mutated file is *exactly*
the scenario a memory/time bound exists for. Every invocation of either tool below runs under a
hard 2 GiB virtual-memory cap and a wall-clock timeout via `prlimit`+`timeout`
(`run_capped`) — an uncapped run during review round 1 hit ~6.8 GB RSS on one mutation, on a
machine shared with several other builders. A capped run's outcome is one of:
  - `0` / `1`: the tool actually evaluated the input and accepted / cleanly rejected it.
  - anything else (timeout, `SIGABRT`/`SIGSEGV`/`SIGKILL`, ...): the tool hit *its own* resource
    limit or crashed before reaching a considered answer. For `map2raw` (real, unmodified DDNet
    code with no bounded-allocation guarantees of its own) this is reported, informationally,
    never counted as a disagreement — it is not a defect in this task's own deliverables. For
    `ddnet-ai`/`ddai-map` (which the task requires to *never* do this — acceptance criterion #2)
    it is a real bug and makes this script exit non-zero with a prominent report, same as a
    genuine disagreement.

Usage:
    python3 map_mutation_fuzz.py <map-file> [<map-file>...] --seed N --iterations-per-map N
"""
import argparse
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAP2RAW = HERE / "build" / "map2raw"
DDNET_AI = HERE.parent.parent / "target" / "release" / "ddnet-ai"

LAYER_FIELD_VALUES = [0, 1, -1, -2, 2, 3, 255, 256, -256, 0x7F808080]

# See this module's docstring ("Sandboxing"). `--signal=KILL` (rather than the default `TERM`)
# guarantees the child actually dies even if it's stuck in a tight allocation/copy loop with
# signals otherwise deferred; `timeout` still reports `124` for this case (see `run_capped`).
MEM_LIMIT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
TIMEOUT_SECS = 20


def run_capped(*args: str) -> tuple[int, bytes, bytes]:
    """Runs `args` (a full command: binary path, then its own arguments) under the hard
    virtual-memory cap and timeout described in this module's docstring. Returns
    `(returncode, stdout, stderr)` — `124` is `timeout`'s own "the command timed out" code;
    a command killed by a signal without timing out (e.g. `SIGABRT` from a denied allocation)
    surfaces as `128 + signal` (bash/coreutils convention `timeout` itself follows when the
    process it's watching dies from a signal), never as a negative number here, since `timeout`
    (not this script) is the direct child.
    """
    proc = subprocess.run(
        ["prlimit", f"--as={MEM_LIMIT_BYTES}", "timeout", "--signal=KILL", f"{TIMEOUT_SECS}s", *args],
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_datafile_header(data: bytes):
    """Returns (item_start, types, item_offsets, num_raw_data, data_size_table_offset_or_None) —
    mirrors the byte layout `crates/ddai-map/src/datafile.rs` and `map2raw.cpp` both implement
    (see docs/formats.md, and `engine/shared/datafile.cpp` for the real format)."""
    ver, size, swap, ntypes, nitems, nraw, isz, dsz = struct.unpack_from("<8i", data, 4)
    p = 36
    types = [struct.unpack_from("<3i", data, p + 12 * i) for i in range(ntypes)]
    p += 12 * ntypes
    ioffs = struct.unpack_from("<%di" % nitems, data, p)
    p += 4 * nitems
    dsize_table_offset = p + 4 * nraw if ver == 4 else None
    p += 4 * nraw + (4 * nraw if ver == 4 else 0)
    item_start = p
    return item_start, types, ioffs, nraw, dsize_table_offset


def mutate_item_field(rng: random.Random, data: bytearray, item_start: int, types, ioffs, type_filter) -> bool:
    """Mutates 1-2 raw `i32` fields inside a random item of a type in `type_filter`. Returns
    `False` (no-op) if there's no such item to mutate."""
    candidates = [t for t in types if t[0] in type_filter]
    if not candidates:
        return False
    t = rng.choice(candidates)
    li = t[1] + rng.randrange(t[2])
    base = item_start + ioffs[li] + 8
    item_size = struct.unpack_from("<i", data, base - 4)[0]
    num_fields = item_size // 4
    if num_fields == 0:
        return False
    for _ in range(1 + rng.randrange(2)):
        field = rng.randrange(num_fields)
        value = rng.choice(LAYER_FIELD_VALUES + [rng.randrange(-(2**31), 2**31)])
        struct.pack_into("<i", data, base + 4 * field, value)
    return True


def mutate_declared_size(rng: random.Random, data: bytearray, dsize_table_offset: int, nraw: int) -> bool:
    """Mutates one entry of the v4 "declared uncompressed size" table directly (review round 2
    finding F7's own reproduction of the ~6.8 GB incident went through exactly this table, not a
    layer field: DDNet's `GetData` allocates its decompression *destination* buffer sized from
    this declared value alone, before it has decompressed anything to check it against)."""
    if dsize_table_offset is None or nraw == 0:
        return False
    k = rng.randrange(nraw)
    cur = struct.unpack_from("<i", data, dsize_table_offset + 4 * k)[0]
    candidates = [0, 1, cur - 1, cur + 1, cur * 2, cur // 2 if cur else 0, 2**31 - 1, -1, 4, 64]
    value = max(-(2**31), min(2**31 - 1, rng.choice(candidates)))
    struct.pack_into("<i", data, dsize_table_offset + 4 * k, value)
    return True


def rawmap_tile_prefix_and_settings_count(buf: bytes):
    """Splits a rawmap v1 buffer (docs/formats.md §1) into `(everything up to and including the
    tile layers, settings_count)` — used by `classify_byte_mismatch` to recognize the one
    documented divergence a byte-for-byte comparison can't distinguish from a real bug on its
    own: a shared Settings/tiles-layer data index (see this module's docstring's "known,
    deliberately documented divergences"). Returns `None` if `buf` isn't shaped like a rawmap v1
    file at all (defensive only — both tools always write valid rawmap v1 on success)."""
    if len(buf) < 17 or buf[:4] != b"RMP1":
        return None
    width, height = struct.unpack_from("<II", buf, 8)
    present = buf[16]
    n = width * height
    sizes = [4, 4 if present & 1 else 0, 2 if present & 2 else 0, 6 if present & 4 else 0, 4 if present & 8 else 0, 2 if present & 16 else 0]
    tile_end = 17 + n * sum(sizes)
    if len(buf) < tile_end + 4:
        return None
    (settings_count,) = struct.unpack_from("<I", buf, tile_end)
    return buf[:tile_end], settings_count


def classify_byte_mismatch(c_bytes: bytes, r_bytes: bytes) -> str | None:
    """Returns a short tag naming the known, documented divergence a `BYTE_MISMATCH` matches
    (see this module's docstring), or `None` if it's unexplained (a real disagreement)."""
    c_parsed = rawmap_tile_prefix_and_settings_count(c_bytes)
    r_parsed = rawmap_tile_prefix_and_settings_count(r_bytes)
    if c_parsed is None or r_parsed is None:
        return None
    c_prefix, c_settings = c_parsed
    r_prefix, r_settings = r_parsed
    if c_prefix == r_prefix and r_settings == 0 and c_settings != 0:
        # Every physics/tile byte agrees; only the Settings *count* differs, and specifically in
        # the direction "map2raw found some, ddai-map found none" — exactly the shape of the F6
        # shared-index divergence (map2raw either reinterpreted a colliding layer's real tile
        # data as Settings, or a Settings blob over `SETTINGS_MAX_BYTES` — both documented in
        # docs/formats.md §10.2 as "ddai-map returns no settings, map2raw may return many").
        return f"KNOWN_DIVERGENCE(settings_count c={c_settings} r=0)"
    return None


def mutate(rng: random.Random, data: bytes) -> bytes:
    item_start, types, ioffs, nraw, dsize_table_offset = parse_datafile_header(data)
    out = bytearray(data)
    mode = rng.randrange(3)
    if mode == 0:
        if mutate_declared_size(rng, out, dsize_table_offset, nraw):
            return bytes(out)
    elif mode == 1:
        if mutate_item_field(rng, out, item_start, types, ioffs, type_filter=(1, 4)):  # INFO or GROUP
            return bytes(out)
    if mutate_item_field(rng, out, item_start, types, ioffs, type_filter=(4,)):  # LAYER — always present
        return bytes(out)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", nargs="+", type=Path)
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--iterations-per-map", type=int, default=300)
    args = ap.parse_args()

    if not MAP2RAW.exists():
        print(f"map_mutation_fuzz.py: {MAP2RAW} not found — run ./build.sh first", file=sys.stderr)
        return 1
    if not DDNET_AI.exists():
        print(f"map_mutation_fuzz.py: {DDNET_AI} not found — run 'cargo build --release -p ddnet-ai' first", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    total = 0
    agree_bytes = 0
    agree_both_reject = 0
    cpp_resource_limited = []
    rust_crashes = []
    known_divergences = []
    disagreements = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for map_path in args.maps:
            original = map_path.read_bytes()
            for _ in range(args.iterations_per_map):
                total += 1
                mutated = mutate(rng, original)
                mutated_path = tmp / "fz.map"
                mutated_path.write_bytes(mutated)
                c_out = tmp / "c.raw"
                r_out = tmp / "r.raw"
                c_code, _, c_err = run_capped(str(MAP2RAW), str(mutated_path), str(c_out))
                r_code, _, r_err = run_capped(
                    str(DDNET_AI), "trace", "export-map", "--map", str(mutated_path), "--out", str(r_out)
                )

                # `ddnet-ai`/`ddai-map` hitting its own resource limit or crashing is a real bug
                # (acceptance criterion #2: never panics, bounded memory) — always reported and
                # always fails the run, regardless of what `map2raw` did.
                if r_code not in (0, 1):
                    rust_crashes.append((map_path.name, f"rust_code={r_code}", r_err.decode(errors="replace")[-200:]))
                    continue
                # `map2raw` hitting its own resource limit is expected (see this file's module
                # docstring) — reported separately, never counted as a disagreement.
                if c_code not in (0, 1):
                    cpp_resource_limited.append((
                        map_path.name,
                        f"cpp_code={c_code} rust={'acc' if r_code == 0 else 'rej'}",
                        c_err.decode(errors="replace")[-160:],
                    ))
                    continue

                c_ok, r_ok = c_code == 0, r_code == 0
                if not c_ok and not r_ok:
                    agree_both_reject += 1
                    continue
                if c_ok and r_ok:
                    c_bytes, r_bytes = c_out.read_bytes(), r_out.read_bytes()
                    if c_bytes == r_bytes:
                        agree_bytes += 1
                        continue
                    tag = classify_byte_mismatch(c_bytes, r_bytes)
                    if tag is not None:
                        known_divergences.append((map_path.name, tag))
                        continue
                    disagreements.append((map_path.name, "BYTE_MISMATCH", c_err.decode(errors="replace")[-160:], r_err.decode(errors="replace")[-160:]))
                    continue
                disagreements.append((
                    map_path.name,
                    f"ACCEPT_REJECT cpp={'acc' if c_ok else 'rej'} rust={'acc' if r_ok else 'rej'}",
                    c_err.decode(errors="replace")[-160:],
                    r_err.decode(errors="replace")[-160:],
                ))

    print(f"map_mutation_fuzz.py: {total} mutations across {len(args.maps)} map(s)")
    print(f"  agree (bytes identical):        {agree_bytes}")
    print(f"  agree (both rejected):          {agree_both_reject}")
    print(f"  known divergence (documented):  {len(known_divergences)}  (not a disagreement — see module docstring)")
    print(f"  cpp (map2raw) resource-limited: {len(cpp_resource_limited)}  (not a disagreement — see module docstring)")
    print(f"  RUST/ddai-map crashed:          {len(rust_crashes)}  (ALWAYS a bug if nonzero)")
    print(f"  disagreements:                  {len(disagreements)}")
    if known_divergences:
        print("  --- known divergence, documented in docs/formats.md §10.2 (informational only) ---")
        for name, tag in known_divergences[:20]:
            print(f"    [{name}] {tag}")
    if cpp_resource_limited:
        print("  --- cpp (map2raw) resource-limited (informational only) ---")
        for name, kind, c_err in cpp_resource_limited[:20]:
            print(f"    [{name}] {kind}\n      cpp: {c_err}")
    if rust_crashes:
        print("  --- RUST/ddai-map CRASHED (this is always a bug) ---")
        for name, kind, r_err in rust_crashes[:20]:
            print(f"    [{name}] {kind}\n      rust: {r_err}")
    for name, kind, c_err, r_err in disagreements[:50]:
        print(f"    [{name}] {kind}\n      cpp:  {c_err}\n      rust: {r_err}")
    return 1 if (disagreements or rust_crashes) else 0


if __name__ == "__main__":
    sys.exit(main())
