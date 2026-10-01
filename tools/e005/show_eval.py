import json,sys
path=sys.argv[1]
last={}
for l in open(path):
    d=json.loads(l)
    if d.get('kind')=='eval':
        last[d['set']]=(d['step'],d['report'])
for s,(step,r) in last.items():
    print(f"== {s} (step {step}) scored steps {r['steps']}")
    dm=r['dir']; print(f"  dir  acc {dm['accuracy']:.3f} bal {dm['balanced_accuracy']:.3f} maj {dm['majority_baseline']:.3f} counts {dm['counts']}")
    for h in ['jump','hook','fire']:
        m=r[h]; print(f"  {h:5s} prev {m['prevalence']:.3f} acc {m['accuracy']:.3f} maj {m['majority_baseline']:.3f} bal {m['balanced_accuracy']:.3f} auroc {m['auroc']:.3f} P {m['precision']:.2f} R {m['recall']:.2f}")
    a=r['aim']; print(f"  aim  n {a['n']} med {a['median_error_deg']:.1f}deg mean {a['mean_error_deg']:.1f} <15deg {a['within_15deg']:.2f} <45deg {a['within_45deg']:.2f}")
