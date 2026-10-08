#!/usr/bin/env python3
"""Task 4.14 (D-124): a stand-in for the bot's duty cycle, to compare scheduling settings under the SAME load at the SAME time.

Every 20 ms it sleeps (the bot's tick), then computes for 4 ms of CPU time (the bot's search). It records
  late_us   how much later than requested the sleep ended (the wake-up latency: the run-queue wait of a woken thread), and
  stall_us  the wall time of the 4 ms of work minus 4 ms (time lost to being preempted in the middle of the search).
Prints one JSON line with percentiles. Several of these, started at once in different cgroups (tools/e2e/cpu_priority_probe.sh),
see the same ambient load, which a sequence of 10-min bot runs cannot give. Stdlib only."""
import json, sys, time


def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p))] if s else None


def main():
    secs = float(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else "probe"
    late, stall = [], []
    end = time.monotonic() + secs
    next_tick = time.monotonic()
    while time.monotonic() < end:
        next_tick += 0.020
        want = next_tick - time.monotonic()
        if want > 0:
            time.sleep(want)
        late.append((time.monotonic() - next_tick) * 1e6)
        w0, c0 = time.monotonic(), time.thread_time()
        while time.thread_time() - c0 < 0.004:
            pass
        stall.append(((time.monotonic() - w0) - 0.004) * 1e6)
        if time.monotonic() - next_tick > 0.1:  # fell far behind: do not try to catch up
            next_tick = time.monotonic()
    out = {"label": label, "n": len(late)}
    for name, v in (("late_us", late), ("stall_us", stall)):
        out[name] = {"p50": round(pct(v, 0.5)), "p90": round(pct(v, 0.9)), "p99": round(pct(v, 0.99)), "p999": round(pct(v, 0.999)), "max": round(max(v))}
    # The share of cycles in which the 4 ms of work took more than 5 ms of wall time (the bot's decision then misses its slot).
    out["work_over_5ms_pct"] = round(100.0 * sum(1 for s in stall if s > 1000) / len(stall), 2)
    out["wake_over_2ms_pct"] = round(100.0 * sum(1 for s in late if s > 2000) / len(late), 2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
