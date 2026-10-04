#!/usr/bin/env python3
"""Reads trace-b (v2/v3, docs/formats.md section 11) and prints entity-kind statistics.

Usage: trb_stats.py TRACE.trb [...]
Prints, per file: version, tick/character counts, and for every entity kind the number of
ticks in which it was present and the maximum simultaneous count. Used to pick which traces
exercise lasers/beams/plasma/lights (task 1.6 stage B corpus selection).
"""
import struct
import sys


def read_trace(path):
    data = open(path, "rb").read()
    off = 0
    assert data[:4] == b"TRB1", "bad magic"
    off = 4
    (version,) = struct.unpack_from("<I", data, off)
    off += 4
    assert version in (2, 3), f"unsupported trace-b version {version}"
    (mlen,) = struct.unpack_from("<I", data, off)
    off += 4 + mlen
    (nchar,) = struct.unpack_from("<I", data, off)
    off += 4 + 4 * nchar
    (hi, nteam) = struct.unpack_from("<II", data, off)
    off += 8 + 4 * nteam
    (ticks,) = struct.unpack_from("<I", data, off)
    off += 4
    return data, off, version, nchar, hi, nteam, ticks


def stats(path):
    data, off, version, nchar, hi, nteam, ticks = read_trace(path)
    row_chars = nchar * 372
    present = {}
    maxn = {}
    for _ in range(ticks):
        off += 4  # game_tick
        off += 16 * hi * nteam
        (ne,) = struct.unpack_from("<I", data, off)
        off += 4
        cnt = {}
        for _ in range(ne):
            (kind,) = struct.unpack_from("<i", data, off)
            cnt[kind] = cnt.get(kind, 0) + 1
            off += 36
        for k, n in cnt.items():
            present[k] = present.get(k, 0) + 1
            maxn[k] = max(maxn.get(k, 0), n)
        off += row_chars
    assert off == len(data), f"trailing bytes: {len(data) - off}"
    return version, ticks, nchar, present, maxn


if __name__ == "__main__" and not (len(sys.argv) > 2 and sys.argv[1] == "--chars"):
    for p in sys.argv[1:]:
        v, t, n, present, maxn = stats(p)
        kinds = " ".join(f"k{k}:{present[k]}t/max{maxn[k]}" for k in sorted(present))
        print(f"{p}: v{v} ticks={t} chars={n} {kinds}")


def char_summary(path):
    """Per-character summary: ticks alive, distinct active weapons, got-mask values, fire presses."""
    data, off, version, nchar, hi, nteam, ticks = read_trace(path)
    out = [dict(alive=0, weapons=set(), masks=set(), freeze=0) for _ in range(nchar)]
    for _ in range(ticks):
        off += 4 + 16 * hi * nteam
        (ne,) = struct.unpack_from("<I", data, off)
        off += 4 + 36 * ne
        for c in range(nchar):
            base = off + c * 372
            # input 44 bytes, core 112 bytes (active_weapon = core field 13 -> offset 44+4*(10+3)), ddrace 216 bytes
            active_weapon = struct.unpack_from("<i", data, base + 44 + 4 * 13)[0]
            ddr = base + 44 + 112
            alive, = struct.unpack_from("<i", data, ddr)
            freeze_time = struct.unpack_from("<i", data, ddr + 4 * 3)[0]
            got_mask = struct.unpack_from("<i", data, ddr + 4 * 12)[0]
            if alive:
                o = out[c]
                o["alive"] += 1
                o["weapons"].add(active_weapon)
                o["masks"].add(got_mask)
                o["freeze"] += 1 if freeze_time > 0 else 0
        off += nchar * 372
    return out


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--chars":
    for c, s in enumerate(char_summary(sys.argv[2])):
        print(c, s)
