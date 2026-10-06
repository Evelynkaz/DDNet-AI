#!/usr/bin/env python3
"""Task 3.11: the server-side truth of input timing.

Joins the bot's input trace (`DDAI_INPUT_TRACE`, see `crates/ddai-bot/src/trace.rs`) with the DDNet server's teehistorian
(`sv_tee_historian 1`) and reports, for every input whose content changed, on which server tick the server really applied it
against the tick the bot meant (`IntendedTick`) and against the tick the decision was aimed at (`exp`).

    live_timing_analyze.py TRACE.jsonl TEEHISTORIAN [TEEHISTORIAN ...] [--cid N] [--json OUT.json]

How the join works: the teehistorian records, in `OnClientPredictedEarlyInput`, every *change* of the input a client has in
force for the tick that is about to be simulated (tick T of the record = the input applies on T + 1, calibrated below: inputs that
arrived in time must come out with an error of exactly 0). The bot's trace has every NETMSG_INPUT it sent (`s`), the server's
INPUTTIMING for it (`t`) and the decisions (`d`). Consecutive equal contents are collapsed on both sides; the two sequences of
distinct contents are aligned greedily (the server may have never applied a value a later input shadowed on the same tick), and
the first server tick of a matched value is the tick it took effect.

Output (per focal bot): n matched changes; the error `applied - intended` (0 = on the tick it was sent for, 1 = a tick late, ...)
and `applied - exp` for the decisions' first inputs; the share of late INPUTTIMING; and the offset calibration.
"""

import argparse
import collections
import json
import struct
import sys


class Reader:
    def __init__(self, data):
        self.d = data
        self.i = 0

    def eof(self):
        return self.i >= len(self.d)

    def int(self):
        d = self.d
        b = d[self.i]
        self.i += 1
        sign = (b >> 6) & 1
        v = b & 0x3F
        shift = 6
        while b & 0x80:
            b = d[self.i]
            self.i += 1
            v |= (b & 0x7F) << shift
            shift += 7
        return v ^ -sign

    def raw(self, n):
        r = self.d[self.i : self.i + n]
        self.i += n
        return r

    def string(self):
        j = self.d.index(b"\0", self.i)
        s = self.d[self.i : j]
        self.i = j + 1
        return s


def parse_teehistorian(path):
    """Yields (tick, cid, input10) for every INPUT_NEW / INPUT_DIFF record (the full input after applying the diff)."""
    with open(path, "rb") as f:
        data = f.read()
    r = Reader(data)
    r.raw(16)
    r.string()  # JSON header
    tick = 0
    implicit_cid = None
    first = True
    prev_input = {}
    out = []
    while not r.eof():
        m = r.int()
        if m >= 0:
            # PLAYER_DIFF
            if implicit_cid is not None and m <= implicit_cid:
                tick += 1
            implicit_cid = m
            r.int()
            r.int()
        elif m == -1:  # FINISH
            break
        elif m == -2:  # TICK_SKIP
            tick += r.int() + 1
            implicit_cid = None
        elif m == -3:  # PLAYER_NEW
            cid = r.int()
            r.int()
            r.int()
            if implicit_cid is not None and cid <= implicit_cid:
                tick += 1
            implicit_cid = cid
        elif m == -4:  # PLAYER_OLD
            cid = r.int()
            if implicit_cid is not None and cid <= implicit_cid:
                tick += 1
            implicit_cid = cid
        elif m == -5:  # INPUT_DIFF
            cid = r.int()
            diff = [r.int() for _ in range(10)]
            cur = [a + b for a, b in zip(prev_input.get(cid, [0] * 10), diff)]
            prev_input[cid] = cur
            out.append((tick, cid, tuple(cur)))
        elif m == -6:  # INPUT_NEW
            cid = r.int()
            cur = [r.int() for _ in range(10)]
            prev_input[cid] = cur
            out.append((tick, cid, tuple(cur)))
        elif m == -7:  # MESSAGE
            r.int()
            n = r.int()
            r.raw(n)
        elif m == -8:  # JOIN
            r.int()
        elif m == -9:  # DROP
            cid = r.int()
            r.string()
            prev_input.pop(cid, None)
        elif m == -10:  # CONSOLE_COMMAND
            r.int()
            r.int()
            r.string()
            n = r.int()
            for _ in range(n):
                r.string()
        elif m == -11:  # EX
            r.raw(16)
            n = r.int()
            r.raw(n)
        else:
            raise ValueError(f"unknown teehistorian message {m} at {r.i}")
    return out


def load_trace(path):
    sent, timing, decisions = [], {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            k = o["k"]
            if k == "s":
                sent.append((o["tick"], tuple(o["in"])))
            elif k == "t":
                timing[o["tick"]] = o["left"]
            elif k == "d":
                decisions[o["tick"]] = o
    return sent, timing, decisions


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, round((len(s) - 1) * p))]


def collapse(seq):
    """[(key, content)] -> only entries whose content differs from the previous one."""
    out = []
    for k, c in seq:
        if not out or out[-1][1] != c:
            out.append((k, c))
    return out


def align(bot, srv):
    """Greedy ordered alignment of two lists of (key, content): returns [(bot_key, srv_key)]."""
    pairs = []
    j = 0
    lookahead = 6
    for k, c in bot:
        for jj in range(j, min(len(srv), j + lookahead)):
            if srv[jj][1] == c:
                pairs.append((k, srv[jj][0], jj))
                j = jj + 1
                break
    return pairs


def analyse(trace, th_files, cid=None):
    sent, timing, decisions = load_trace(trace)
    # The bot may reconnect (ticks restart); keep to the longest run of increasing sent ticks.
    runs, cur = [], []
    for t, c in sent:
        if cur and t < cur[-1][0] - 5:
            runs.append(cur)
            cur = []
        cur.append((t, c))
    runs.append(cur)
    sent = max(runs, key=len)
    t0, t1 = sent[0][0], sent[-1][0]
    inputs = []
    for f in th_files:
        inputs += parse_teehistorian(f)
    by_cid = collections.defaultdict(list)
    for t, c, i in inputs:
        by_cid[c].append((t, i))
    bot = collapse(sent)
    best = None
    for c, seq in by_cid.items():
        srv = collapse(seq)
        pairs = align(bot, srv)
        if cid is not None and c != cid:
            continue
        if best is None or len(pairs) > len(best[1]):
            best = (c, pairs, srv)
    if best is None:
        raise SystemExit("no teehistorian input stream matched the trace")
    c, pairs, srv = best
    # The server's first tick of a content, vs the intended tick it was sent for.
    raw = [(bk, sk) for bk, sk, _ in pairs]
    # Calibrate the constant offset on the inputs that were on time (INPUTTIMING >= 0).
    on_time = [sk - bk for bk, sk in raw if timing.get(bk, -1) >= 0]
    offset = collections.Counter(on_time).most_common(1)[0][0] if on_time else 0
    err = []
    for bk, sk in raw:
        err.append((bk, sk - bk - offset))
    err_hist = collections.Counter(e for _, e in err)
    late_inputs = sum(1 for v in timing.values() if v < 0)
    dec_err, dec_exp_err = [], []
    brain_exp_err = []
    err_by_tick = dict(err)
    for tick, d in decisions.items():
        if tick in err_by_tick and d["exp"]:
            e_int = err_by_tick[tick]
            dec_err.append(e_int)
            dec_exp_err.append(tick + e_int - d["exp"])
            if d.get("brain"):
                brain_exp_err.append(tick + e_int - d["exp"])
    # Decisions whose input changed on the wire, split by whether the first slot was made.
    first_slot = [tick + err_by_tick[tick] - d["exp"] for tick, d in decisions.items() if tick in err_by_tick and d["exp"] and tick == d["first"]]
    res = {
        "cid": c,
        "bot_changes": len(bot),
        "server_changes": len(srv),
        "matched": len(pairs),
        "offset_calibrated": offset,
        "applied_minus_intended": {str(k): v for k, v in sorted(err_hist.items())},
        "late_inputtiming": late_inputs,
        "inputtiming": len(timing),
        "decisions_matched": len(dec_err),
        "applied_minus_expected": {str(k): v for k, v in sorted(collections.Counter(dec_exp_err).items())},
        "brain_decisions_matched": len(brain_exp_err),
        "brain_applied_minus_expected": {str(k): v for k, v in sorted(collections.Counter(brain_exp_err).items())},
        "brain_exact_share_vs_expected": (sum(1 for e in brain_exp_err if e == 0) / len(brain_exp_err)) if brain_exp_err else None,
        "exact_share_vs_expected": (sum(1 for e in dec_exp_err if e == 0) / len(dec_exp_err)) if dec_exp_err else None,
        "exact_share_vs_intended": (sum(1 for e in dec_err if e == 0) / len(dec_err)) if dec_err else None,
        "span_ticks": t1 - t0,
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("teehistorian", nargs="+")
    ap.add_argument("--cid", type=int)
    ap.add_argument("--json")
    a = ap.parse_args()
    res = analyse(a.trace, a.teehistorian, a.cid)
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
