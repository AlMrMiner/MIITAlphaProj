"""Оценка на фиксированном holdout; разметка не передается обработчику."""
import csv, json, random
from collections import Counter, defaultdict
from pathlib import Path
from .data import read_applications
from .reference import References, read_table
from .engine import Engine, fingerprint, equivalent, wilson
from .split import SEED

CATEGORIES=('correct','incorrect','review','missed')

def partition(rows):
    groups=defaultdict(list)
    for r in rows: groups[fingerprint(r)].append(r['app_id'])
    keys=sorted(groups); random.Random(SEED).shuffle(keys)
    hold=set(); target=round(len(rows)*.2)
    for k in keys:
        if len(hold)>=target: break
        hold.update(groups[k])
    return hold

def write_csv(path, rows, fields):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def run_evaluation(data_dir, output_dir, model_path=None):
    p=Path(data_dir); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=sorted((r for r in read_applications(p/'applications_raw.csv') if r['split']=='train'),key=lambda r:(r['created_at'],r['app_id']))
    hold=partition(rows)
    labels=defaultdict(list)
    for row in read_table(p/'labels_train.csv'):
        labels[row['app_id']].append(row)
    refs=References(p); base=Engine(refs); main=Engine(refs,True)
    stats=defaultdict(lambda:[0,0]); held=[]
    for r in rows:
        a,b=base.analyze(r),main.analyze(r)
        if r['app_id'] in hold:
            held.append((r,a,b)); continue
        truth=defaultdict(list)
        for l in labels[r['app_id']]: truth[l['field']].append(l['correct_value'])
        for field,f in b.items():
            if f.candidate is not None:
                ok=bool(truth[field]) and all(equivalent(field,f.candidate,v) for v in truth[field])
                stats[f.rule][0]+=ok; stats[f.rule][1]+=1
    main.confidence={k:wilson(*v) for k,v in stats.items()}
    calibration={k:{'correct':v[0],'total':v[1],'lower_bound':main.confidence[k]} for k,v in sorted(stats.items())}
    (out/'confidence.json').write_text(json.dumps(main.confidence,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'calibration.json').write_text(json.dumps(calibration,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'seed':SEED,'fit_applications':len(rows)-len(hold),'holdout_applications':len(hold),'methods':{}}
    types=[]; examples=[]
    methods=[('baseline',base,1),('g3',main,2)]
    if model_path:
        from .ml import load_model, ml_decisions, batch_predictions
        model=load_model(model_path)
        ml_scores=batch_predictions(((r,b) for r,a,b in held),model)
        methods.append(('ml',main,2))
    for name,engine,idx in methods:
        totals=Counter(); by_type=defaultdict(Counter); queue=[]; audits=[]; metrics=Counter()
        for item in held:
            r=item[0]; truth=labels[r['app_id']]; tf=defaultdict(list)
            for l in truth: tf[l['field']].append(l)
            decisions=ml_decisions(engine,r,item[idx],model,ml_scores) if name=='ml' else engine.decisions(item[idx]); correct_app=True; introduced=False
            metrics['true_fields']+=len(tf); metrics['predicted_fields']+=len(decisions); metrics['tp']+=len(tf.keys() & decisions.keys())
            metrics['erroneous_input_applications']+=bool(truth)
            metrics['queue_applications']+=any(d['action']=='review' for d in decisions.values())
            for field,d in decisions.items():
                if d['action']=='auto':
                    metrics['auto_fields']+=1
                    ok=bool(tf.get(field)) and all(equivalent(field,d['candidate'],l['correct_value']) for l in tf[field]); metrics['auto_correct_fields']+=ok
                    if not tf.get(field): introduced=True; metrics['false_auto_fields']+=1
                if d['action']=='review': queue.append({'app_id':r['app_id'],'original':r[field],**d})
                audits.append({'app_id':r['app_id'],'original':r[field],**d})
            for l in truth:
                d=decisions.get(l['field']); category='missed' if not d else 'review' if d['action']=='review' else 'correct' if equivalent(l['field'],d['candidate'],l['correct_value']) else 'incorrect'
                totals[category]+=1;by_type[l['error_type']][category]+=1
                if category!='correct': correct_app=False
                if category in ('incorrect','missed') and len([e for e in examples if e['method']==name and e['category']==category])<5:
                    examples.append({'method':name,'category':category,**l,'proposed':d['candidate'] if d else '', 'reason':d['reason'] if d else 'Нет сигнала'})
            metrics['residual_error_applications']+=not correct_app or introduced
            metrics['clean_without_review']+=correct_app and not introduced and not any(d['action']=='review' for d in decisions.values())
        for t,c in sorted(by_type.items()): types.append({'method':name,'error_type':t,**{k:c[k] for k in CATEGORIES}})
        m={**{k:totals[k] for k in CATEGORIES},**metrics,'precision':metrics['tp']/max(1,metrics['predicted_fields']),'recall':metrics['tp']/max(1,metrics['true_fields']),'auto_precision':metrics['auto_correct_fields']/max(1,metrics['auto_fields']),'residual_rate':metrics['residual_error_applications']/len(hold),'clean_without_review_rate':metrics['clean_without_review']/len(hold)}
        summary['methods'][name]=m
        fields=['app_id','field','original','candidate','reason','rule','confidence','action']
        write_csv(out/f'{name}_queue.csv',queue,fields);write_csv(out/f'{name}_audit.csv',audits,fields)
    write_csv(out/'per_type.csv',types,['method','error_type',*CATEGORIES]);write_csv(out/'examples.csv',examples,['method','category','app_id','field','error_type','raw_value','correct_value','proposed','reason'])
    (out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary
