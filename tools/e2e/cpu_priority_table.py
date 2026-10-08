#!/usr/bin/env python3
"""Task 4.14 (D-124): the table of tools/e2e/cpu_priority.sh runs.

    tools/e2e/cpu_priority_table.py <runs-dir> [label ...]        (default: every <label>.json in the directory, sorted by name)
    tools/e2e/cpu_priority_table.py --groups <runs-dir> <label> ...  (mean ± SE over the runs of each unit/load/threads group)
    tools/e2e/cpu_priority_table.py --pool <runs-dir> <label> ...  (the pooled regression on 30-s windows)

Per run: the decision-weighted mean of candidates per decision over the 30-s windows of the status poller (first 35 s skipped: join and warm-up),
the report's candidates p50, the brain-decision share that gave up its first slot (`brain_missed_first_slot / brain_decisions`, D-063/D-101),
brain p50 / p99 (us), total p99 (us), the horizon shares (percent of decisions with lag 0..4 ticks), the unit's CPU pressure ("some" stall per
second of run, ms/s), its CPU use in cores, and the load average before/after (everything on the box, ours and others'). Stdlib only."""
import json, os, sys


def load_meta(p):
    m = {}
    if os.path.exists(p):
        for line in open(p):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                m[k] = v
    return m


def windows(rows):
    """Non-overlapping 30-s windows: rows with a search window, from t >= 35 s, at least 30 s apart."""
    out, last = [], -1e9
    for r in rows:
        sw = r.get("search_window") or {}
        if r["t"] >= 35 and r["t"] - last >= 30 and sw.get("decisions"):
            out.append(sw)
            last = r["t"]
    return out


def summarize(d, label):
    rep = json.load(open(f"{d}/{label}.json"))
    rows = [json.loads(l) for l in open(f"{d}/{label}.status.jsonl")] if os.path.exists(f"{d}/{label}.status.jsonl") else []
    meta = load_meta(f"{d}/{label}.meta")
    det = rep["latency_detail_us"]
    sl = det["slots"]
    ws = windows(rows)
    wn = sum(w["decisions"] for w in ws)
    cand_mean = sum(w["decisions"] * w["candidates_mean"] for w in ws if w.get("candidates_mean") is not None) / wn if wn else None
    cpu = [r for r in rows if "psi_some_us" in r]
    psi = usage = None
    if len(cpu) >= 2 and cpu[-1]["t"] > cpu[0]["t"]:
        dt = cpu[-1]["t"] - cpu[0]["t"]
        psi = (cpu[-1]["psi_some_us"] - cpu[0]["psi_some_us"]) / dt / 1000.0
        usage = (cpu[-1]["usage_usec"] - cpu[0]["usage_usec"]) / dt / 1e6
    loads = [r["load1"] for r in rows]
    busy = [r["machine_busy_pct"] for r in rows if "machine_busy_pct" in r]
    runn = [r["runnable"] for r in rows if "runnable" in r]
    return {
        "label": label, "variant": meta.get("variant"), "load": meta.get("load"), "threads": meta.get("threads"),
        "decisions": sl["brain_decisions"], "cand_mean": cand_mean, "cand_p50": det["candidates"]["p50"],
        "missed_pct": 100.0 * sl["brain_missed_first_slot"] / max(1, sl["brain_decisions"]),
        "brain_p50": det["brain"]["p50"], "brain_p99": det["brain"]["p99"], "total_p99": det["total"]["p99"],
        "lag": [round(x, 1) for x in det["horizon_shares"]], "psi_ms_s": psi, "cores": usage,
        "load1_mean": sum(loads) / len(loads) if loads else None,
        "busy": sum(busy) / len(busy) if busy else None, "runnable": sum(runn) / len(runn) if runn else None,
        "loadavg": f'{meta.get("loadavg_before", "?").split()[0]}/{meta.get("loadavg_after", "?").split()[0]}',
    }


def fmt(x, p=1):
    return "-" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))


def solve(a, b):
    """Gauss elimination for the small normal equations."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[piv] = m[piv], m[i]
        for r in range(n):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [x - f * y for x, y in zip(m[r], m[i])]
    return [m[i][n] / m[i][i] for i in range(n)]


def ols(xs, ys):
    """Least squares with intercept: coefficients and their standard errors (rows are 30-s windows, which are non-overlapping)."""
    k = len(xs[0]) + 1
    rows = [[1.0] + list(x) for x in xs]
    xtx = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    xty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(k)]
    beta = solve(xtx, xty)
    res = [y - sum(b * v for b, v in zip(beta, r)) for r, y in zip(rows, ys)]
    s2 = sum(e * e for e in res) / max(1, len(ys) - k)
    inv = [solve(xtx, [1.0 if i == j else 0.0 for i in range(k)]) for j in range(k)]
    se = [(s2 * inv[i][i]) ** 0.5 for i in range(k)]
    return beta, se


def pooled(d, labels):
    """candidates per decision in each 30-s window ~ intercept + runnable (mean of the 5-s samples of that window) + new-unit + 2-threads."""
    xs, ys = [], []
    for l in labels:
        meta = load_meta(f"{d}/{l}.meta")
        rows = [json.loads(x) for x in open(f"{d}/{l}.status.jsonl")]
        last = -1e9
        for i, r in enumerate(rows):
            sw = r.get("search_window") or {}
            if r["t"] >= 35 and r["t"] - last >= 30 and sw.get("candidates_mean") is not None and sw.get("decisions", 0) >= 25:
                last = r["t"]
                near = [q["runnable"] for q in rows if r["t"] - 30 < q["t"] <= r["t"] and "runnable" in q]
                xs.append([sum(near) / len(near), 1.0 if meta["variant"] != "old" else 0.0, 1.0 if meta["threads"] != "1" else 0.0])
                ys.append(sw["candidates_mean"])
    beta, se = ols(xs, ys)
    names = ["intercept", "per runnable thread", "new unit settings", "search-threads 2"]
    print(f"\nPooled over {len(ys)} non-overlapping 30-s windows of {len(labels)} runs: candidates per decision = intercept + b*runnable + settings")
    for n, b, s in zip(names, beta, se):
        print(f"  {n:22s} {b:+7.2f}  (SE {s:.2f})")


def groups(d, labels):
    """Mean and standard error over the runs of each (unit variant, load, search-threads) group."""
    g = {}
    for l in labels:
        s = summarize(d, l)
        g.setdefault((s["variant"], s["load"], s["threads"]), []).append(s)

    def ms(v):
        n = len(v)
        m = sum(v) / n
        se = (sum((x - m) ** 2 for x in v) / (n - 1) / n) ** 0.5 if n > 1 else None
        return f"{m:.1f}" + (f" ± {se:.1f}" if se is not None else "")

    print("| unit | load | thr | runs | cand/dec | cand p50 | missed first slot % | brain p99 ms | stall ms/s | runnable (mean) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (v, ld, th), S in sorted(g.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        print(f"| {v} | {ld} | {th} | {len(S)} | {ms([s['cand_mean'] for s in S])} | {ms([float(s['cand_p50']) for s in S])} | "
              f"{ms([s['missed_pct'] for s in S])} | {ms([s['brain_p99'] / 1000 for s in S])} | {ms([s['psi_ms_s'] for s in S])} | {ms([s['runnable'] for s in S])} |")


def main():
    if sys.argv[1] == "--groups":
        groups(sys.argv[2], sys.argv[3:])
        return
    if sys.argv[1] == "--pool":
        pooled(sys.argv[2], sys.argv[3:])
        return
    d = sys.argv[1]
    labels = sys.argv[2:] or sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json") and not f.endswith(".status.jsonl"))
    print("| run | unit | load | thr | decisions | cand/dec (30 s windows) | cand p50 | missed first slot % | brain p50 us | brain p99 us | total p99 us | lag 0/1/2/3/4 % | stall ms/s | cores | machine busy % | runnable (mean) | load1 mean | loadavg before/after |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for l in labels:
        try:
            s = summarize(d, l)
        except (OSError, KeyError, ValueError) as e:
            print(f"| {l} | error: {e} |")
            continue
        print(f"| {s['label']} | {s['variant']} | {s['load']} | {s['threads']} | {s['decisions']} | {fmt(s['cand_mean'])} | {s['cand_p50']} | {fmt(s['missed_pct'])} | "
              f"{s['brain_p50']} | {s['brain_p99']} | {s['total_p99']} | {'/'.join(str(x) for x in s['lag'])} | {fmt(s['psi_ms_s'])} | {fmt(s['cores'], 2)} | {fmt(s['busy'])} | {fmt(s['runnable'])} | {fmt(s['load1_mean'])} | {s['loadavg']} |")


if __name__ == "__main__":
    main()
