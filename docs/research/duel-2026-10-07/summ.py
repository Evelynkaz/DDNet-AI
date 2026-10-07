# Duel 2026-10-07 post-mortem: pools the per-run JSONL of `ddai-env --example pm_counterfactual` (cf-main.jsonl) by cell, start offset and clip.
# python3 summ.py ~/aiddnet/data/scratch/pm-1007/cf-main.jsonl
import json,sys,collections
rows=[json.loads(l) for l in open(sys.argv[1])]
# starts where the open-loop replay reproduces our real freeze within +-5 ticks (from --sanity)
repro={('chased-into-freeze-76040-s266.clip',30),('chased-into-freeze-76040-s266.clip',20),('chased-into-freeze-76040-s266.clip',10),
 ('self-freeze-69936-s224.clip',40),('self-freeze-69936-s224.clip',30),('self-freeze-69936-s224.clip',10),
 ('self-freeze-71852-s242.clip',30),('self-freeze-71852-s242.clip',10),('self-freeze-80618-s206.clip',10)}
cells=[]
for r in rows:
    if r['cell'] not in cells: cells.append(r['cell'])
def agg(sel):
    a=collections.Counter()
    for r in sel:
        a['n']+=1; a[r['first']]+=1
        if r['first'] in('we','both'):
            a['hooked']+=r['hooked']; a['hammered']+=r['hammered']
            if r['at']<=r['offset']+5: a['by_real']+=1
    return a
def fmt(a):
    n=a['n'] or 1
    return f"{a['we']+a['both']:>3}/{a['n']:<3} {100*(a['we']+a['both'])/n:5.1f}%  he {a['he']:>3} none {a['none']:>3}  hooked {a['hooked']:>3} hammered {a['hammered']:>3} by_real+5 {a['by_real']:>3}"
print('== pooled per cell (all 19 starts)')
for c in cells: print(f"{c:12s}", fmt(agg([r for r in rows if r['cell']==c])))
print('== pooled per cell, only starts the open-loop replay reproduces (9)')
for c in cells: print(f"{c:12s}", fmt(agg([r for r in rows if r['cell']==c and (r['clip'],r['offset']) in repro])))
print('== per offset')
for off in (40,30,20,10):
    print(' offset',off)
    for c in cells: print(f"   {c:12s}", fmt(agg([r for r in rows if r['cell']==c and r['offset']==off])))
print('== per clip (all offsets)')
for clip in sorted(set(r['clip'] for r in rows)):
    print(' ',clip)
    for c in cells: print(f"   {c:12s}", fmt(agg([r for r in rows if r['cell']==c and r['clip']==clip])))
