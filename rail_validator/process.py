"""Пакетная обработка и явное подтверждение оператором (без подключения к БД)."""
import json
from pathlib import Path
from html import escape
from collections import defaultdict, Counter
from .data import FIELDS, read_applications
from .reference import References, read_table
from .engine import Engine
from .validation import parse_date
from .evaluate import partition, write_csv

AUDIT_FIELDS=['app_id','field','original','candidate','reason','rule','confidence','action']
REVIEW_FIELDS=[*AUDIT_FIELDS,'operator_action','operator_value']
TICKET_EDIT_FIELDS=['app_id','field','value']

def render_queue(path,queue,summary):
    cols=['app_id','field','original','candidate','reason','confidence']
    def cell(value): return escape(str(value if value is not None else ''))
    def display(row,key):
        if key=='confidence':
            return '—' if not row.get('candidate') else f'{row[key]:.1%}' if row.get(key) is not None else '—'
        value=row.get(key,'')
        if row.get('field') in ('planned_date','created_at') and key in ('original','candidate'):
            parsed=parse_date(value)
            if parsed is not None:return parsed.strftime('%d.%m.%Y')
        return value
    body=''.join('<tr>'+''.join('<td>'+cell(display(r,k))+'</td>' for k in cols)+'</tr>' for r in queue)
    counts=' · '.join(f'{cell(k)}: {cell(v)}' for k,v in summary.items())
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Очередь оператора Г3</title><style>body{font:15px system-ui;margin:32px;color:#172b4d;background:#f5f7fb}table{border-collapse:collapse;background:white;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #d9e0ea}th{position:sticky;top:0;background:#e2e9f4}input{padding:10px;width:min(480px,90%);margin:16px 0}td{overflow-wrap:anywhere}p{max-width:900px}</style><h1>Очередь оператора</h1><p>'''+counts+'''</p><p>Откройте operator_decisions.csv для внесения решений: accept (принять кандидат), edit (ввести значение), clear (подтвердить исходное значение), drop (отклонить заявку). Затем запустите команду resolve из README. До заполнения всех строк заявка остаётся в очереди.</p><input id="search" placeholder="Поиск по номеру, полю, причине" oninput="for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase())"><table><thead><tr>'''+''.join('<th>'+cell(k)+'</th>' for k in cols)+'</tr></thead><tbody>'+body+'</tbody></table></html>'
    path.write_text(page,encoding='utf-8')

def run_processing(data_dir,output_dir,confidence_path,method,split,model_path=None):
    p=Path(data_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    engine=Engine(References(p),method in ('g3','ml'))
    model=None
    if method=='ml':
        from .ml import load_model, ml_decisions
        if not model_path: raise ValueError('Для ML требуется путь к сохранённой модели')
        model=load_model(model_path)
    if method=='g3': engine.confidence=json.loads(Path(confidence_path).read_text(encoding='utf-8'))
    rows=sorted((r for r in read_applications(p/'applications_raw.csv') if split=='all' or r['split']==('train' if split=='holdout' else split)),key=lambda r:(r['created_at'],r['app_id']))
    hold=partition(rows) if split=='holdout' else None
    ready=[];pending=[];audit=[];queue=[];counts=Counter()
    if model is not None:
        from .ml import batch_predictions
        analyzed=[(row,engine.analyze(row)) for row in rows]
        scores=batch_predictions(analyzed,model)
    else:
        analyzed=((row,engine.analyze(row)) for row in rows)
        scores=None
    for row,findings in analyzed:
        decisions=ml_decisions(engine,row,findings,model,scores) if model is not None else engine.decisions(findings)
        if hold is not None and row['app_id'] not in hold: continue
        result=dict(row); counts['processed']+=1
        for field,d in decisions.items():
            entry={'app_id':row['app_id'],'original':row[field],**d};audit.append(entry)
            if d['action']=='auto': result[field]=d['candidate'];counts['auto_fields']+=1
            else: queue.append(entry)
        normalized = parse_date(result['planned_date'])
        if normalized is not None:result['planned_date']=normalized.isoformat()
        if any(d['action']=='review' for d in decisions.values()):pending.append(result)
        else:ready.append(result)
    counts.update({'ready':len(ready),'pending':len(pending),'review_fields':len(queue)})
    write_csv(out/'ready.csv',ready,FIELDS);write_csv(out/'pending.csv',pending,FIELDS)
    write_csv(out/'audit.csv',audit,AUDIT_FIELDS);write_csv(out/'queue.csv',queue,AUDIT_FIELDS)
    write_csv(out/'operator_decisions.csv',[{**q,'operator_action':'','operator_value':''} for q in queue],REVIEW_FIELDS)
    render_queue(out/'queue.html',queue,counts)
    (out/'summary.json').write_text(json.dumps(dict(counts),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(counts),ensure_ascii=False,indent=2))

def resolve(directory, decision_path, output_dir):
    p=Path(directory);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    original={(r['app_id'],r['field']):r for r in read_table(p/'queue.csv')}
    edits={}
    for r in read_table(decision_path):
        key=(r['app_id'],r['field']); action=r['operator_action'].strip().lower()
        if key not in original or key in edits: raise ValueError(f'Неизвестный или повторный ключ {key}')
        if action not in ('','accept','edit','clear','drop'): raise ValueError(f'Неизвестное решение {action}')
        edits[key]=(action,r['operator_value'])
    approved=list(read_applications(p/'ready.csv')); pending=[]; rejected=[]; log=[]
    by_id=defaultdict(list)
    for key,q in original.items():by_id[key[0]].append((key,q))
    extra_edits=defaultdict(dict)
    edit_file=p/'ticket_edits.csv'
    if edit_file.exists():
        for edit in read_table(edit_file):
            if edit['field'] not in FIELDS or edit['field'] in ('app_id','split'):
                raise ValueError('Недопустимое поле в ticket_edits.csv')
            if not edit['value'].strip():raise ValueError('Пустое значение в ticket_edits.csv')
            extra_edits[edit['app_id']][edit['field']]=edit['value']
    for row in read_applications(p/'pending.csv'):
        result=dict(row); unresolved=False; drop=False; local=[]
        for key,q in by_id[row['app_id']]:
            action,value=edits.get(key,('','')); field=key[1]
            if action=='drop':drop=True
            elif not action:unresolved=True
            elif action=='accept':
                if not q['candidate'] or field=='app_id':raise ValueError(f'Кандидат отсутствует или ID неизменяем: {key}')
                result[field]=q['candidate']
            elif action=='edit':
                if not value.strip() or field=='app_id':raise ValueError(f'Недопустимое значение или ID неизменяем: {key}')
                result[field]=value
            elif action=='clear':pass
            if action:local.append({'app_id':key[0],'field':field,'action':action,'value':result[field]})
        if not drop:
            for field,value in extra_edits.get(row['app_id'],{}).items():
                if (row['app_id'],field) in original:raise ValueError('Конфликт редактирования поля')
                result[field]=value
                local.append({'app_id':row['app_id'],'field':field,'action':'edit','value':value})
        if drop:rejected.append(row);log.extend(local)
        elif unresolved:pending.append(row)
        else:
            normalized = parse_date(result['planned_date'])
            if normalized is not None:result['planned_date']=normalized.isoformat()
            approved.append(result);log.extend(local)
    write_csv(out/'approved.csv',approved,FIELDS);write_csv(out/'pending.csv',pending,FIELDS);write_csv(out/'rejected.csv',rejected,FIELDS)
    write_csv(out/'operator_audit.csv',log,['app_id','field','action','value'])
    print(json.dumps({'approved':len(approved),'pending':len(pending),'rejected':len(rejected)},ensure_ascii=False))
