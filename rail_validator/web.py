"""Локальная панель управления пакетной обработкой и очередью оператора."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import uuid
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .validation import parse_date
from .data import FIELDS
from .process import REVIEW_FIELDS, TICKET_EDIT_FIELDS
from .reference import read_table

DATA_FILES = ('applications_raw.csv', 'labels_train.csv', 'stations.csv', 'cargo.csv', 'clients.csv', 'wagon_types.csv')
JOBS = ('inspect', 'evaluate', 'train-ml', 'evaluate-ml', 'process', 'resolve')
MAX_UPLOAD = 100 * 1024 * 1024


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.data = self.root / 'data'
        self.output = self.root / 'output'
        self.data.mkdir(exist_ok=True)
        self.output.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.job = None
        self.generation = 0

    def new_job(self, command, options):
        if command not in JOBS:
            raise ValueError('Неизвестная команда')
        method = options.get('method', 'g3')
        split = options.get('split', 'holdout')
        if method not in ('baseline', 'g3', 'ml') or split not in ('holdout', 'train', 'test', 'all'):
            raise ValueError('Недопустимый метод или выборка')
        argv = [sys.executable, '-m', 'rail_validator', command]
        if command == 'inspect':
            argv += [str(self.data/'applications_raw.csv')]
        elif command in ('evaluate', 'train-ml', 'evaluate-ml'):
            dirs = {'evaluate': 'eval', 'train-ml': 'ml', 'evaluate-ml': 'eval-ml'}
            argv += [str(self.data), '--output', str(self.output/dirs[command])]
            if command == 'evaluate-ml':
                argv += ['--model', str(self.output/'ml'/'model.joblib')]
        elif command == 'process':
            argv += [str(self.data), '--method', method, '--split', split,
                     '--output', str(self.output/f'demo-{method}-{split}')]
            if method == 'g3':
                confidence = self.output/'eval'/'confidence.json'
                if not confidence.exists():
                    confidence = self.root/'metrics'/'confidence.json'
                if not confidence.exists():
                    raise ValueError('Сначала выполните оценку Г3 для получения confidence.json')
                argv += ['--confidence', str(confidence)]
            elif method == 'ml':
                model = self.output/'ml'/'model.joblib'
                if not model.exists():
                    raise ValueError('Сначала обучите ML-модель')
                argv += ['--model', str(model)]
        elif command == 'resolve':
            source = self.resolve_source(options.get('source', ''))
            decisions = source/'operator_decisions.csv'
            if not decisions.is_file():
                raise ValueError('Не найдены решения оператора')
            argv += [str(source), '--decisions', str(decisions),
                     '--output', str(self.output/('resolved-'+source.name))]
        with self.lock:
            if self.job and self.job['status'] == 'running':
                raise ValueError('Другая задача ещё выполняется')
            self.generation += 1
            number = self.generation
            self.job = {'id': number, 'command': command, 'status': 'running', 'output': '',
                        'started': datetime.now().isoformat(timespec='seconds')}
        threading.Thread(target=self._run, args=(number, argv), daemon=True).start()
        return self.job.copy()

    def _run(self, number, argv):
        try:
            process = subprocess.run(argv, cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True,
                                     encoding='utf-8', errors='replace', timeout=1200)
            output = (process.stdout + '\n' + process.stderr).strip()[-50000:]
            status = 'completed' if process.returncode == 0 else 'failed'
        except Exception as exc:
            status, output = 'failed', f'{type(exc).__name__}: {exc}'
        with self.lock:
            if self.job and self.job['id'] == number:
                self.job.update(status=status, output=output,
                                finished=datetime.now().isoformat(timespec='seconds'))

    def status(self):
        with self.lock:
            return dict(self.job) if self.job else None

    def sources(self):
        result = []
        for directory in sorted(self.output.glob('demo-*')):
            if directory.is_dir() and (directory/'queue.csv').exists():
                result.append(directory.name)
        return result

    def resolve_source(self, name):
        if name not in self.sources():
            raise ValueError('Неизвестный набор результатов обработки')
        return self.output/name

    def queue(self, name, query='', page=1):
        p = self.resolve_source(name)
        rows = read_table(p/'queue.csv')
        edits = {(r['app_id'], r['field']): r for r in read_table(p/'operator_decisions.csv')}
        result = []
        for r in rows:
            if query and query.casefold() not in ' '.join(str(v) for v in r.values()).casefold():
                continue
            r['confidence_display'] = ('—' if not r['candidate'] or not r.get('confidence')
                                       else f'{float(r["confidence"])*100:.1f}%')
            ed = edits.get((r['app_id'], r['field']), {})
            r['operator_action'] = ed.get('operator_action', '')
            r['operator_value'] = ed.get('operator_value', '')
            if r['field'] in ('planned_date','created_at'):
                r['original_display'] = format_date_display(r['original'])
                r['candidate_display'] = format_date_display(r['candidate']) if r['candidate'] else ''
                r['operator_value_display'] = format_date_display(r['operator_value']) if r['operator_value'] else ''
            result.append(r)
        per_page = 50
        page = max(1, min(page, max(1, (len(result)+per_page-1)//per_page)))
        return {'total': len(result), 'page': page, 'pages': max(1, (len(result)+per_page-1)//per_page),
                'rows': result[(page-1)*per_page:page*per_page]}

    def ticket(self, name, app_id):
        p = self.resolve_source(name)
        row = next((r for r in read_table(p/'pending.csv') if r['app_id'] == app_id), None)
        if row is None:
            raise ValueError('Заявка не найдена в очереди')
        queue = {r['field']: r for r in read_table(p/'queue.csv') if r['app_id'] == app_id}
        decisions = {r['field']: r for r in read_table(p/'operator_decisions.csv') if r['app_id'] == app_id}
        edits = {r['field']: r['value'] for r in read_table(p/'ticket_edits.csv') if r['app_id'] == app_id} if (p/'ticket_edits.csv').exists() else {}
        return {'app_id':app_id,'fields':[{ 'name':field,'original':row[field],
            'display':format_date_display(row[field]) if field in ('planned_date','created_at') else row[field],
            'value':edits.get(field, decisions.get(field,{}).get('operator_value') if decisions.get(field,{}).get('operator_action') == 'edit' else row[field]),
            'issue': queue.get(field),
            'decision':decisions.get(field,{}).get('operator_action',''),
            'candidate':queue.get(field,{}).get('candidate','')}
            for field in FIELDS]}

    def save_ticket_field(self, name, app_id, field, value):
        if field not in FIELDS or field in ('app_id','split'):
            raise ValueError('Поле не подлежит редактированию')
        if not isinstance(value,str) or not value.strip():
            raise ValueError('Укажите непустое значение')
        p=self.resolve_source(name)
        with self.lock:
            if self.job and self.job['status']=='running':
                raise ValueError('Дождитесь завершения обработки')
            source=next((r for r in read_table(p/'pending.csv') if r['app_id']==app_id),None)
            if source is None:raise ValueError('Заявка не найдена в очереди')
            if field == 'planned_date':
                dt=parse_date(value)
                if dt is None:raise ValueError('Неверная плановая дата')
                value=dt.isoformat()
            if field == 'created_at':
                try: value=datetime.fromisoformat(value).isoformat(timespec='seconds')
                except ValueError:raise ValueError('Используйте дату и время в формате YYYY-MM-DDTHH:MM:SS')
            queue={(r['app_id'],r['field']) for r in read_table(p/'queue.csv')}
            if (app_id,field) in queue:
                return self._save_decision_locked(p,app_id,field,'edit',value)
            file=p/'ticket_edits.csv'
            records=read_table(file) if file.exists() else []
            records=[r for r in records if not (r['app_id']==app_id and r['field']==field)]
            if value != source[field]:records.append({'app_id':app_id,'field':field,'value':value})
            atomic_write_csv(file,TICKET_EDIT_FIELDS,records)
            return {'saved':True,'value':value}

    def _save_decision_locked(self,p,app_id,field,action,value):
        queue={(r['app_id'],r['field']):r for r in read_table(p/'queue.csv')}
        key=(app_id,field)
        if key not in queue:raise ValueError('Поле отсутствует в очереди')
        if field=='app_id' and action in ('accept','edit'):raise ValueError('ID нельзя менять')
        if action=='accept' and not queue[key]['candidate']:raise ValueError('Предложенное значение отсутствует')
        file=p/'operator_decisions.csv'
        records=read_table(file)
        found=False
        for row in records:
            if (row['app_id'],row['field'])==key:
                row['operator_action'],row['operator_value']=action,(value if action=='edit' else '')
                found=True
                break
        if not found:raise ValueError('Запись решения не найдена')
        atomic_write_csv(file,REVIEW_FIELDS,records)
        return {'saved':True}

    def save_decision(self,name,app_id,field,action,value):
        if action not in ('','accept','edit','clear','drop'):
            raise ValueError('Недопустимое решение')
        if action=='edit' and (not isinstance(value,str) or not value.strip()):
            raise ValueError('Для edit требуется значение')
        p=self.resolve_source(name)
        if action=='edit' and field=='planned_date':
            parsed=parse_date(value)
            if parsed is None:raise ValueError('Неверная плановая дата')
            value=parsed.isoformat()
        with self.lock:
            if self.job and self.job['status']=='running':raise ValueError('Дождитесь завершения обработки')
            return self._save_decision_locked(p,app_id,field,action,value)


def format_date_display(value):
    dt=parse_date(value)
    if dt is not None:return dt.strftime('%d.%m.%Y')
    try:return datetime.fromisoformat(value).strftime('%d.%m.%Y %H:%M')
    except (ValueError,TypeError):return value


def atomic_write_csv(file,fields,rows):
    temp=file.with_name(file.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)
        os.replace(temp,file)
    finally:temp.unlink(missing_ok=True)


HTML = r'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rail Validator · Панель</title>
<style>
:root{font:15px system-ui,Arial;color:#243044;background:#f3f6fb}*{box-sizing:border-box}body{margin:0}header{background:#142b48;color:white;padding:18px 28px}header h1{margin:0;font-size:22px}header small{color:#bcd0e5}main{max-width:1240px;margin:auto;padding:22px;display:grid;gap:16px}.card{background:white;border:1px solid #dfe5ef;border-radius:12px;padding:20px;box-shadow:0 1px 4px #172b4d0b}h2{margin:0 0 15px;font-size:18px}h3{margin:15px 0 10px;font-size:15px}button{cursor:pointer;border:0;border-radius:7px;padding:10px 14px;background:#215c97;color:white;font-weight:600}button.secondary{background:#e8eef6;color:#23405d}button:disabled{opacity:.45;cursor:not-allowed}input,select{padding:9px;border:1px solid #b9c9d8;border-radius:6px;max-width:100%;min-width:0}input[type=file]{width:100%}.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.muted{color:#66788c;font-size:13px}.status{font-weight:600}.success{color:#16734b}.error{color:#bb3030}.scroll{overflow:auto;max-height:560px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:left;border-bottom:1px solid #e3eaf1;vertical-align:top}th{position:sticky;top:0;background:#edf2f8;z-index:1}td{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:12px;border-radius:7px;max-height:250px;overflow:auto}.note{background:#f4f7fa;border-radius:7px;padding:10px;margin-top:12px}a{color:#215c97}.smallbtn{padding:6px 9px;font-size:12px}.badge{color:#1e574a;background:#e1f3ed;padding:3px 7px;border-radius:5px}.row-controls{display:flex;gap:5px;align-items:center;flex-wrap:wrap}.row-controls input{width:130px}.row-controls select{max-width:130px}#ticket{display:none;position:fixed;inset:0;background:#0e2331a6;z-index:40;align-items:center;justify-content:center;padding:20px}#ticket.open{display:flex}.ticket-panel{background:white;border-radius:12px;width:min(940px,100%);max-height:90vh;overflow:auto;padding:22px;box-shadow:0 15px 55px #0005}.ticket-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:10px}.ticket-field{padding:10px;border:1px solid #e1e7ed;border-radius:8px;display:grid;gap:5px}.ticket-field.problem{border-color:#d6a751;background:#fffaf0}.ticket-field input{width:100%}.ticket-field .hint{font-size:12px;color:#806426}.ticket-actions{display:flex;gap:7px;flex-wrap:wrap}.ticket-actions button{padding:7px 10px}#messages{position:sticky;top:0;z-index:20}#messages:not(:empty){background:#fff5dd;color:#674d10;padding:12px;border:1px solid #e8bd55;border-radius:7px}
</style></head><body><header><h1>Rail Validator</h1><small>Локальная панель обработки заявок · CSV · Г3 · ML</small></header><main><div id="messages"></div>
<div class="grid"><section class="card"><h2>1. Исходные данные</h2><p class="muted">Загрузите файлы исходного набора. Ранее загруженные файлы с тем же именем будут заменены.</p><select id="file-name"></select><div class="controls" style="margin-top:9px"><input type="file" id="upload" accept=".csv"><button onclick="upload()">Загрузить CSV</button></div><p id="files" class="muted"></p><button class="secondary" onclick="start('inspect')">Проверить структуру (inspect)</button></section>
<section class="card"><h2>2. Обучение и оценка</h2><div class="controls"><button onclick="start('evaluate')">Оценить baseline и Г3</button><button onclick="start('train-ml')">Обучить ML</button><button onclick="start('evaluate-ml')">Сравнить с ML</button></div><p class="muted">Сначала загрузите размеченные данные и справочники. Обучение запускается локально.</p></section></div>
<section class="card"><h2>3. Обработать заявки</h2><div class="controls"><label>Метод <select id="method"><option value="g3">Г3</option><option value="baseline">Baseline</option><option value="ml">Г3 + ML</option></select></label><label>Выборка <select id="split"><option value="holdout">Holdout</option><option value="test">Test</option><option value="train">Train</option><option value="all">Все</option></select></label><button onclick="start('process')">Запустить обработку</button></div><div class="note muted">Обработка создаёт отдельную папку результатов для каждого метода и выборки. Повторный запуск перезапишет результаты этой папки, включая сохранённые решения оператора.</div></section>
<section class="card"><h2>Состояние последней команды</h2><div id="job-status" class="status muted">Команды ещё не запускались</div><pre id="job-output" hidden></pre></section>
<section class="card"><h2>4. Очередь оператора</h2><div class="controls"><label>Результаты <select id="source" onchange="loadQueue()"></select></label><input id="search" placeholder="Поиск: ID, поле, причина" onkeydown="if(event.key==='Enter')loadQueue()"><button class="secondary" onclick="loadQueue()">Найти</button><button onclick="resolveQueue()">Применить решения (resolve)</button></div><p class="muted" id="queue-info">После обработки заявки появятся здесь.</p><div class="scroll"><table><thead><tr><th>ID</th><th>Поле</th><th>Было</th><th>Кандидат</th><th>Причина</th><th>Надёжность</th><th>Решение оператора</th></tr></thead><tbody id="queue"></tbody></table></div><div class="controls" style="margin-top:12px"><button class="secondary" onclick="shiftPage(-1)">Назад</button><span id="pages"></span><button class="secondary" onclick="shiftPage(1)">Вперёд</button></div><div class="note muted">Принять — использовать кандидат; Изменить — ввести значение вручную; Оставить — подтвердить исходное; Отклонить — отклонить заявку. Каждая строка сохраняется отдельно.</div></section>
<section class="card"><h2>5. Выгрузка файлов</h2><div id="downloads" class="controls"></div></section></main>
<div id="ticket" role="dialog" aria-modal="true" aria-labelledby="ticket-title"><div class="ticket-panel"><div class="controls" style="justify-content:space-between"><h2 id="ticket-title">Заявка</h2><button class="secondary" onclick="closeTicket()">Закрыть</button></div><p class="muted">Изменения сохраняются в решения оператора. Для формирования итоговых заявок нажмите «Применить решения».</p><div id="ticket-fields" class="ticket-grid"></div></div></div>
<script>
const $=id=>document.getElementById(id);let page=1,pages=1,busy=false,previousStatus='',ticketId='';const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function displayDate(v){const s=String(v??'');if(/^\d{4}-\d{2}-\d{2}(?:T.*)?$/.test(s)){const [y,m,d]=s.substring(0,10).split('-');return `${d}.${m}.${y}`+(s.includes('T')?' '+s.substring(11,16):'')}return s}
function ticketValue(field,v){return field==='planned_date'||field==='created_at'?displayDate(v):String(v??'')}
function notice(s){$('messages').textContent=s; if(s)setTimeout(()=>{if($('messages').textContent===s)$('messages').textContent=''},6500)}
async function request(url,options){const r=await fetch(url,options);let data;try{data=await r.json()}catch{throw Error('Ошибка ответа сервера')}if(!r.ok)throw Error(data.error||'Ошибка');return data}
async function refresh(){try{let s=await request('/api/status');$('files').textContent='Найдены: '+s.files.join(', ');const selected=$('source').value; $('source').replaceChildren(...s.sources.map(x=>new Option(x,x)));if(s.sources.includes(selected))$('source').value=selected;const j=s.job;busy=!!(j&&j.status==='running');$('job-status').className='status '+(j&&j.status==='failed'?'error':j&&j.status==='completed'?'success':'muted');$('job-status').textContent=j?`${j.command}: ${j.status} · ${j.started}`:'Команды ещё не запускались';$('job-output').hidden=!j||!j.output;$('job-output').textContent=j?.output||'';if(j&&previousStatus!==j.id+':'+j.status&&j.status!=='running'){previousStatus=j.id+':'+j.status; if(j.status==='completed'){loadQueue();showDownloads()}else notice('Команда завершилась с ошибкой — см. журнал')} document.querySelectorAll('button').forEach(b=>{if(b.hasAttribute('data-keep'))return; b.disabled=busy});}catch(e){notice(e.message)}}
async function start(command){try{await request('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,method:$('method').value,split:$('split').value,source:$('source').value})});await refresh()}catch(e){notice(e.message)}}
async function upload(){let file=$('upload').files[0];if(!file){notice('Выберите CSV');return}try{await request('/api/upload?name='+encodeURIComponent($('file-name').value),{method:'POST',headers:{'Content-Type':'text/csv'},body:file});$('upload').value='';notice('Файл загружен');refresh()}catch(e){notice(e.message)}}
function shiftPage(n){if(page+n>=1&&page+n<=pages){page+=n;loadQueue(false)}}
async function loadQueue(reset=true){if(reset)page=1;const name=$('source').value;if(!name){$('queue').replaceChildren();$('queue-info').textContent='Сначала обработайте заявки';return}try{let q=await request('/api/queue?source='+encodeURIComponent(name)+'&page='+page+'&q='+encodeURIComponent($('search').value));page=q.page;pages=q.pages;$('queue-info').textContent='Найдено строк: '+q.total;$('pages').textContent=`${page} / ${pages}`;$('queue').innerHTML=q.rows.map(r=>`<tr><td><button class="secondary smallbtn" data-keep="1">${esc(r.app_id)}</button></td><td>${esc(r.field)}</td><td>${esc(r.original_display||r.original)}</td><td>${esc(r.candidate_display||r.candidate||'—')}</td><td>${esc(r.reason)}</td><td>${esc(r.confidence_display)}</td><td><div class="row-controls"><select class="choice"><option value="">Не решено</option><option value="accept">Принять</option><option value="edit">Изменить</option><option value="clear">Оставить</option><option value="drop">Отклонить</option></select><input class="manual" placeholder="Новое значение" value="${esc(r.operator_value_display||r.operator_value)}"><button class="smallbtn" data-keep="1">Сохранить</button></div></td></tr>`).join('');[...$('queue').children].forEach((tr,i)=>{const r=q.rows[i];tr.querySelector('.choice').value=r.operator_action;tr.querySelector('button').onclick=()=>save(r,tr);tr.querySelector('td button').onclick=()=>openTicket(r.app_id)})}catch(e){notice(e.message)}}
async function save(r,tr){try{const action=tr.querySelector('.choice').value, value=tr.querySelector('.manual').value;await request('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:$('source').value,app_id:r.app_id,field:r.field,action,value})});notice('Решение сохранено')}catch(e){notice(e.message)}}
async function openTicket(appId){ticketId=appId;try{const t=await request('/api/ticket?source='+encodeURIComponent($('source').value)+'&app_id='+encodeURIComponent(appId));$('ticket-title').textContent='Заявка '+appId;$('ticket-fields').innerHTML=t.fields.map(f=>{const iss=f.issue;return `<div class="ticket-field ${iss?'problem':''}" data-field="${esc(f.name)}"><strong>${esc(f.name)}</strong><span class="muted">Исходное: ${esc(f.display)}</span>${iss?`<span class="hint">${esc(iss.reason)}${f.candidate?' · предложение: '+esc(ticketValue(f.name,f.candidate)):''}</span>`:''}<input ${f.name==='app_id'||f.name==='split'?'disabled':''} value="${esc(f.name==='planned_date'&&f.value===f.original?f.display:ticketValue(f.name,f.value))}"><div class="ticket-actions">${f.name==='app_id'||f.name==='split'?'':`<button class="smallbtn" data-keep="1" data-save="1">Сохранить</button>`}${iss?`<button class="secondary smallbtn" data-keep="1" data-action="accept" ${!f.candidate?'disabled':''}>Принять</button><button class="secondary smallbtn" data-keep="1" data-action="clear">Оставить</button><button class="secondary smallbtn" data-keep="1" data-action="drop">Отклонить</button>`:''}</div><small class="muted">${f.decision?'Решение: '+esc(f.decision):iss?'Ожидает решения':''}</small></div>`}).join('');[...$('ticket-fields').children].forEach((node,i)=>{const field=t.fields[i];const save=node.querySelector('[data-save]');if(save)save.onclick=async()=>{try{let val=node.querySelector('input').value;if(field.name==='created_at'&&/^\d{2}\.\d{2}\.\d{4}/.test(val)){const [d,m,y]=val.substring(0,10).split('.');val=y+'-'+m+'-'+d+'T'+(val.slice(11)||'00:00')+':00'}const r=await request('/api/ticket/edit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:$('source').value,app_id:appId,field:field.name,value:val})});node.querySelector('small').textContent='Сохранено';notice('Поле сохранено')}catch(e){notice(e.message)}};node.querySelectorAll('[data-action]').forEach(btn=>btn.onclick=async()=>{try{await request('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:$('source').value,app_id:appId,field:field.name,action:btn.dataset.action,value:''})});node.querySelector('small').textContent='Решение: '+btn.dataset.action;notice('Решение сохранено')}catch(e){notice(e.message)}})});$('ticket').classList.add('open')}catch(e){notice(e.message)}}
function closeTicket(){$('ticket').classList.remove('open');loadQueue(false)}
function resolveQueue(){if(!confirm('Применить сохранённые решения оператора?'))return;start('resolve')}
async function showDownloads(){try{const x=await request('/api/downloads');$('downloads').innerHTML=x.files.map(f=>`<a href="/api/download?file=${encodeURIComponent(f)}" download>${esc(f)}</a>`).join(' · ')}catch(e){notice(e.message)}}
$('file-name').replaceChildren(...'applications_raw.csv labels_train.csv stations.csv cargo.csv clients.csv wagon_types.csv'.split(' ').map(x=>new Option(x,x)));refresh();showDownloads();setInterval(refresh,2500);
</script></body></html>'''


def handler_for(ws):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, data, content_type='application/json; charset=utf-8'):
            body = data if isinstance(data,bytes) else (json.dumps(data,ensure_ascii=False).encode('utf-8') if isinstance(data,(dict,list)) else data.encode('utf-8'))
            self.send_response(code)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'")
            self.end_headers();self.wfile.write(body)

        def _json(self):
            size = int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 16384: raise ValueError('Недопустимый размер запроса')
            return json.loads(self.rfile.read(size))

        def _trusted(self):
            origin=self.headers.get('Origin','')
            host=self.headers.get('Host','')
            valid={'127.0.0.1:8765','localhost:8765'}
            if host not in valid or (origin and origin not in {'http://'+v for v in valid}):
                raise ValueError('Операция разрешена только с локальной панели')

        def do_GET(self):
            u=urlsplit(self.path);params=parse_qs(u.query)
            try:
                if u.path == '/': return self._send(200,HTML,'text/html; charset=utf-8')
                if u.path == '/api/status':
                    return self._send(200,{'files':[f for f in DATA_FILES if (ws.data/f).is_file()],
                                           'sources':ws.sources(),'job':ws.status()})
                if u.path == '/api/ticket':
                    return self._send(200,ws.ticket(params.get('source',[''])[0],params.get('app_id',[''])[0]))
                if u.path == '/api/queue':
                    return self._send(200,ws.queue(params.get('source',[''])[0],params.get('q',[''])[0],int(params.get('page',['1'])[0])))
                if u.path == '/api/downloads':
                    files=[]
                    for p in sorted(ws.output.rglob('*')):
                        if p.is_file() and p.suffix.lower() in ('.csv','.json','.html'):
                            files.append(p.relative_to(ws.output).as_posix())
                    return self._send(200,{'files':files})
                if u.path == '/api/download':
                    name=params.get('file',[''])[0]
                    target=(ws.output/name).resolve()
                    if not target.is_relative_to(ws.output.resolve()) or not target.is_file() or target.suffix.lower() not in ('.csv','.json','.html'):
                        raise ValueError('Недоступный файл')
                    body=target.read_bytes()
                    self.send_response(200);self.send_header('Content-Type','application/octet-stream')
                    self.send_header('Content-Disposition',f'attachment; filename="{target.name}"')
                    self.send_header('Content-Length',str(len(body)));self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
                    return
                return self._send(404,{'error':'Не найдено'})
            except (ValueError,FileNotFoundError) as exc:return self._send(400,{'error':str(exc)})

        def do_POST(self):
            u=urlsplit(self.path)
            try:
                self._trusted()
                if u.path=='/api/upload':
                    name=parse_qs(u.query).get('name',[''])[0]
                    if name not in DATA_FILES:raise ValueError('Неизвестное имя файла')
                    size=int(self.headers.get('Content-Length','0'))
                    if size <= 0 or size > MAX_UPLOAD:raise ValueError('Максимальный размер CSV — 100 МБ')
                    if ws.status() and ws.status()['status']=='running':raise ValueError('Дождитесь завершения команды')
                    temp=ws.data/(name+'.'+uuid.uuid4().hex+'.tmp')
                    try:
                        with temp.open('wb') as f:
                            remaining=size
                            while remaining:
                                block=self.rfile.read(min(1024*1024,remaining))
                                if not block:raise ValueError('Неполный файл')
                                f.write(block);remaining-=len(block)
                        with temp.open('rb') as check:
                            if not check.read(100).strip():raise ValueError('Пустой CSV')
                        os.replace(temp,ws.data/name)
                    finally:temp.unlink(missing_ok=True)
                    return self._send(200,{'uploaded':name,'bytes':size})
                data=self._json()
                if u.path=='/api/jobs':return self._send(200,ws.new_job(data.get('command',''),data))
                if u.path=='/api/ticket/edit':
                    return self._send(200,ws.save_ticket_field(data.get('source',''),data.get('app_id',''),data.get('field',''),data.get('value','')))
                if u.path=='/api/decision':
                    return self._send(200,ws.save_decision(data.get('source',''),data.get('app_id',''),data.get('field',''),data.get('action',''),data.get('value','')))
                return self._send(404,{'error':'Не найдено'})
            except (ValueError,KeyError,TypeError,FileNotFoundError,json.JSONDecodeError) as exc:return self._send(400,{'error':str(exc)})

        def log_message(self, format, *args):
            pass

    return Handler


def serve(root, host='127.0.0.1', port=8765):
    if host not in ('127.0.0.1','localhost') or port != 8765:
        raise ValueError('Панель предназначена только для http://127.0.0.1:8765')
    ws=Workspace(root)
    server=ThreadingHTTPServer(('127.0.0.1',port),handler_for(ws))
    print(f'Rail Validator: http://127.0.0.1:{port}',flush=True)
    try:server.serve_forever()
    finally:server.server_close()
