import json,sys
sets=['teacher-val','teacher-holdout:chillblock5-ruler','human-val','human-val-tagged']
print('%-14s %-34s %s'%('run','set','dir acc/bal | jump auc | hook auc/bal | fire auc | aim med'))
for path in sys.argv[1:]:
    name=path.rstrip('/').split('/')[-1]
    last={}
    for l in open(path+'/metrics.jsonl'):
        d=json.loads(l)
        if d.get('kind')=='eval': last[d['set']]=d['report']
    for s in sets:
        r=last.get(s)
        if not r: continue
        print('%-14s %-34s %.3f/%.3f | %.3f | %.3f/%.3f | %.3f | %.1f'%(name,s,r['dir']['accuracy'],r['dir']['balanced_accuracy'],r['jump']['auroc'],r['hook']['auroc'],r['hook']['balanced_accuracy'],r['fire']['auroc'],r['aim']['median_error_deg']))
