import re, sys, collections
def load(p):
    rows = []
    for l in open(p):
        m = re.match(r'SAMPLE ctx=(\d) +(\d+) +(\d+) +[\d.]+% (.*)', l)
        if m:
            rows.append((int(m[1]), int(m[2]), int(m[3]), m[4]))
    return [r for r in rows if r[0] != 3 and r[1] != 0]
def group(name):
    if name.startswith('phys ') or name.startswith('collision') or name == 'World::step self':
        return 'World<f32>::step (physics)'
    if name.startswith('adapter'):
        return 'adapter (PhysicsWorld: tee snapshots, inputs, restore)'
    if name.startswith('score') or name in ('eval final terms', 'eval per-tick bookkeeping', 'eval step loop self'):
        return 'scoring (score_tick, flight/ray probes, rollout bookkeeping)'
    if name.startswith('eval') or name in ('hook_allowed', 'thaw_escapable'):
        return 'plan decode (step_to_input, hook gate, opponent model)'
    if name.startswith('decide'):
        return 'search generation and choice'
    return name
for p in sys.argv[1:]:
    rows = load(p)
    tot = sum(r[2] for r in rows)
    g = collections.Counter(); sh = sum(r[2] for r in rows if r[0] == 1)
    for c, ph, n, name in rows:
        g[group(name)] += n
    print(p, 'samples', tot, 'of which inside the shield: %.1f%%' % (100 * sh / tot))
    for k, v in g.most_common():
        print('  %5.1f%%  %s' % (100 * v / tot, k))
    # finest physics rows
    fine = collections.Counter()
    for c, ph, n, name in rows:
        fine[name] += n
    print('  top rows:')
    for k, v in fine.most_common(12):
        print('    %5.1f%%  %s' % (100 * v / tot, k))
