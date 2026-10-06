#!/usr/bin/env python3
"""Tests of live_timing_analyze.py: `python3 -m unittest tools/e2e/test_live_timing_analyze.py` (stdlib only)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import live_timing_analyze as a  # noqa: E402


def pack_int(v):
    """Teeworlds variable-width integer (the inverse of Reader.int)."""
    out = bytearray()
    sign = 0
    if v < 0:
        sign = 1
        v = ~v
    b = (sign << 6) | (v & 0x3F)
    v >>= 6
    while v:
        out.append(b | 0x80)
        b = v & 0x7F
        v >>= 7
    out.append(b)
    return bytes(out)


class VarInt(unittest.TestCase):
    def test_round_trip(self):
        for v in [0, 1, -1, 63, 64, -64, -65, 1000, -1000, 123456789, -123456789]:
            self.assertEqual(a.Reader(pack_int(v)).int(), v, v)


class Align(unittest.TestCase):
    def test_collapse_and_align_skip_a_shadowed_value(self):
        bot = a.collapse([(1, "A"), (2, "A"), (3, "B"), (4, "C"), (5, "C"), (6, "D")])
        self.assertEqual([k for k, _ in bot], [1, 3, 4, 6])
        # The server never applied B (a later input took its tick).
        srv = a.collapse([(10, "A"), (12, "C"), (14, "D")])
        pairs = a.align(bot, srv)
        self.assertEqual([(b, s) for b, s, _ in pairs], [(1, 10), (4, 12), (6, 14)])


class Teehistorian(unittest.TestCase):
    def test_input_records_get_their_tick_and_the_diff_is_added(self):
        header = bytes(16) + b'{"version":"2"}\0'
        body = b""
        body += pack_int(-3) + pack_int(0) + pack_int(5) + pack_int(5)  # PLAYER_NEW cid 0 at tick 0
        body += pack_int(-6) + pack_int(0) + b"".join(pack_int(x) for x in [1, 2, 3, 0, 0, 0, 0, 0, 0, 0])  # INPUT_NEW
        body += pack_int(-2) + pack_int(4)  # TICK_SKIP: 4 empty ticks, the next is tick 5
        body += pack_int(-5) + pack_int(0) + b"".join(pack_int(x) for x in [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])  # INPUT_DIFF jump+1
        body += pack_int(-1)
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(header + body)
        try:
            recs = a.parse_teehistorian(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual(recs[0], (0, 0, (1, 2, 3, 0, 0, 0, 0, 0, 0, 0)))
        self.assertEqual(recs[1][2], (1, 2, 3, 1, 0, 0, 0, 0, 0, 0))
        self.assertEqual(recs[1][0], 5)


if __name__ == "__main__":
    unittest.main()
