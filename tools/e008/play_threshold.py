#!/usr/bin/env python3
"""In-play hook-threshold matching (E-005 review F9, E-008): picks the hook threshold at which the model's in-play hook
START rate equals the planner's on the same states, using TRAINING arenas only (clb-left and pit, the `hook-play`
default; never the holdout) and no win-rate information at all.

  tools/e008/play_threshold.py <run-name> [--threads N] [--games 100] [--tol 0.03] [--max-evals 5]

`hook-play` plays the model alone and lets the planner label every state it visits, so the rate of "hook started
from a not-hooking state" is comparable between the two. d(t) = student start rate - teacher start rate decreases in the
hook threshold t (a higher threshold starts less often); the search brackets the zero of d on [0.05, 0.95] by secant
steps with bisection safeguards and stops when |d| <= tol, the budget of evaluations is spent, or the bracket is
narrower than 0.01. The chosen threshold is written to <eval>/selected-play.bundle (jump and fire thresholds keep the
selected bundle's own values) and every evaluation to <eval>/play-threshold.json. Resumable: finished evaluations
(`play-threshold/hook-play-<t>.json`) are reused.
"""
import argparse, json, os, subprocess, sys

ROOT = os.path.expanduser('~/aiddnet/data/runs/E-008')
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def start_rates(hook_play):
    """Pooled start rates (student, teacher) over the arenas of a hook-play JSON: share of not-hooking states in which
    the player started a hook. Returns (student, teacher, states)."""
    n = s = t = 0
    for ev in hook_play:
        v = ev['counts']['not_out']
        n += v['n']
        s += v['student_hook']
        t += v['teacher_hook']
    if n == 0:
        raise ValueError('no not-hooking states in the hook-play output')
    return s / n, t / n, n


def next_threshold(points, lo=0.05, hi=0.95):
    """The next hook threshold to try, given evaluated (t, d) points (d decreasing in t), or None when the zero of d is
    bracketed tighter than 0.01. `points` is non-empty."""
    pts = sorted(points)
    below = [p for p in pts if p[1] > 0]   # student starts too often: t too low
    above = [p for p in pts if p[1] <= 0]  # student starts too rarely or matching: t high enough
    lo_t = max((p[0] for p in below), default=lo)
    hi_t = min((p[0] for p in above), default=hi)
    if hi_t - lo_t < 0.0101:
        return None
    if below and above:
        a = max(below, key=lambda p: p[0])
        b = min(above, key=lambda p: p[0])
        # secant between the bracket ends, kept away from the ends so that the bracket always shrinks
        guess = a[0] + (b[0] - a[0]) * a[1] / (a[1] - b[1])
        margin = 0.1 * (b[0] - a[0])
        guess = min(max(guess, a[0] + margin), b[0] - margin)
    else:
        # one-sided: step by the local slope when two points exist, else by 0.15
        if len(pts) >= 2 and pts[0][1] != pts[-1][1]:
            slope = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
            guess = pts[-1][0] - pts[-1][1] / slope if slope else pts[-1][0] + (0.15 if below else -0.15)
        else:
            guess = pts[-1][0] + (0.15 if below else -0.15)
        guess = min(max(guess, lo), hi)
    return round(guess, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--games', type=int, default=100)
    ap.add_argument('--tol', type=float, default=0.03)
    ap.add_argument('--max-evals', type=int, default=5)
    ap.add_argument('--first', type=float, default=0.5)
    ap.add_argument('--bin', default=os.environ.get('DDNET_AI', os.path.join(REPO, 'target/release/ddnet-ai')))
    a = ap.parse_args()
    kind = 'fly' if 'fly' in a.name else 'mlp' if 'mlpw' in a.name else 'gru' if 'gruw' in a.name else None
    if kind is None:
        sys.exit('unknown model kind in ' + a.name)
    ev = os.path.join(ROOT, 'eval', a.name)
    work = os.path.join(ev, 'play-threshold')
    os.makedirs(work, exist_ok=True)
    src = os.path.join(ROOT, a.name, 'checkpoints', 'selected.bundle')
    cwd = REPO
    env = dict(os.environ, RAYON_NUM_THREADS=str(a.threads))

    def evaluate(t):
        bundle = os.path.join(work, f'hook-{t:.2f}.bundle')
        out = os.path.join(work, f'hook-play-{t:.2f}.json')
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            if not os.path.exists(bundle):
                subprocess.run([a.bin, 'train', 'set-thresholds', '--kind', kind, '--bundle', src, '--out', bundle,
                                '--hook', f'{t:.2f}'], check=True, cwd=cwd, env=env, stderr=subprocess.DEVNULL)
            with open(out + '.part', 'w') as f:
                subprocess.run([a.bin, 'train', 'hook-play', '--actor', f'{kind}:{bundle}', '--arenas', 'clb-left,pit',
                                '--games', str(a.games), '--threads', str(a.threads)], check=True, cwd=cwd, env=env,
                               stdout=f, stderr=open(os.path.join(work, f'hook-play-{t:.2f}.log'), 'w'))
            os.replace(out + '.part', out)
        s, te, n = start_rates(json.load(open(out)))
        return {'hook_threshold': t, 'start_student': s, 'start_teacher': te, 'states': n, 'd': s - te}

    evals = []
    t = a.first
    while t is not None and len(evals) < a.max_evals:
        r = evaluate(t)
        evals.append(r)
        print(f"t={t:.2f}: start student {100*r['start_student']:.1f}% / teacher {100*r['start_teacher']:.1f}% (d={100*r['d']:+.1f})",
              flush=True)
        if abs(r['d']) <= a.tol:
            break
        t = next_threshold([(e['hook_threshold'], e['d']) for e in evals])
        if t is not None and any(abs(e['hook_threshold'] - t) < 0.005 for e in evals):
            break
    best = min(evals, key=lambda e: abs(e['d']))
    chosen = os.path.join(ev, 'selected-play.bundle')
    subprocess.run(['cp', os.path.join(work, f"hook-{best['hook_threshold']:.2f}.bundle"), chosen], check=True)
    json.dump({'name': a.name, 'rule': 'in-play start rate matched to the teacher on clb-left+pit (training arenas)',
              'chosen_hook_threshold': best['hook_threshold'], 'evaluations': evals},
              open(os.path.join(ev, 'play-threshold.json'), 'w'), indent=1)
    print(f"{a.name}: hook threshold {best['hook_threshold']:.2f} (start {100*best['start_student']:.1f}% vs teacher "
          f"{100*best['start_teacher']:.1f}%) -> {chosen}")


if __name__ == '__main__':
    main()
