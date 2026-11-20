"""Общие проверки, baseline и контекстные кандидаты Г3."""
from dataclasses import dataclass
from collections import Counter, defaultdict
from functools import lru_cache
from math import sqrt
from .validation import local_checks, parse_date
from .reference import norm

@dataclass(frozen=True)
class Finding:
    field: str
    reason: str
    candidate: str | None = None
    rule: str = 'manual'

@lru_cache(maxsize=65536)
def date_key(s):
    d=parse_date(s)
    return d.isoformat() if d else s

def fingerprint(r):
    return tuple(date_key(v) if k=='planned_date' else norm(v) for k,v in r.items() if k not in ('app_id','split','created_at','channel'))

class Engine:
    def __init__(self, refs, advanced=False):
        self.refs=refs; self.advanced=advanced
        self.seen={}; self.confidence={}

    def analyze(self,r):
        ref=self.refs; findings={}
        def add(field,reason,candidate=None,rule='manual'):
            if candidate==r[field]: candidate=None
            prev=findings.get(field)
            if prev:
                reason=prev.reason+'; '+reason if reason not in prev.reason else prev.reason
                if prev.candidate is not None and candidate is None:
                    candidate,rule=prev.candidate,prev.rule
                elif prev.candidate is not None and candidate!=prev.candidate:
                    candidate,rule=None,'conflict'
            findings[field]=Finding(field,reason,candidate,rule)
        for issue in local_checks(r): add(issue.field,issue.reason)
        if r['channel'] not in ('ЛК','API','email','Excel-импорт'): add('channel','Неизвестный канал')
        if r['split'] not in ('train','test',''): add('split','Неизвестное значение split')
        if not r['app_id']: add('app_id','Нет номера заявки')
        client={}
        for role, namefield, innfield in [('sender','client_name','client_inn'),('receiver','consignee_name','consignee_inn')]:
            inn=r[innfield]; name=r[namefield]; byname=ref.exact('client',name); byinn=ref.clients.get(inn)
            if byname and (not byinn or self.advanced or (role=='sender' and inn==r['consignee_inn'])):
                if inn!=byname: add(innfield,'ИНН не соответствует имени',byname,'client_inn_by_name')
                client[role]=ref.clients[byname]
            elif byinn: client[role]=byinn
            if not byinn and not byname: add(innfield,'ИНН отсутствует в справочнике')
            if byinn and byname and byname!=inn and not self.advanced: add(innfield,'Конфликт имени и ИНН')
            chosen=client.get(role)
            if chosen:
                canonical=chosen['client_name']
                if canonical!=name: add(namefield,'Имя отличается от справочника',canonical if norm(canonical)==norm(name) else None,'client_name_format')
                if chosen['role'] not in ('оба', 'грузоотправитель' if role=='sender' else 'грузополучатель'): add(innfield,'Роль клиента не соответствует заявке')
            else: add(namefield,'Клиент не идентифицирован')
        sender=client.get('sender'); receiver=client.get('receiver')
        if sender and sender['contract_number']!=r['contract_number']:
            add('contract_number','Договор не принадлежит отправителю',sender['contract_number'],'contract_by_sender')
        elif not sender: add('contract_number','Нельзя проверить договор без отправителя')
        def allowed(c): return set(c['station_codes'].split(';')) if c else set()
        dep, dest=allowed(sender),allowed(receiver)
        dc,tc=r['departure_station_code'],r['destination_station_code']
        swapped=self.advanced and dc!=tc and dc in dest and tc in dep and dc not in dep and tc not in dest and dc in ref.stations and tc in ref.stations
        for prefix,typ,codes in [('departure_station','station',dep),('destination_station','station',dest),('cargo','cargo',set())]:
            cf,nf=prefix+'_code',prefix+'_name'; code,name=r[cf],r[nf]
            table=ref.stations if typ=='station' else ref.cargo; key=typ+'_name'; record=table.get(code); byname=ref.exact(typ,name)
            if swapped and typ=='station':
                other=tc if prefix.startswith('departure') else dc
                add(cf,'Вероятно перепутаны станции',other,'route_swap_code'); add(nf,'Вероятно перепутаны станции',table[other][key],'route_swap_name'); continue
            if record and name==record[key]:
                if self.advanced and typ=='station' and codes and code not in codes: add(cf,'Станция нетипична для клиента')
                continue
            if not record:
                candidate=byname or (ref.fuzzy(typ,name) if self.advanced else None)
                add(cf,'Код отсутствует в справочнике',candidate,'invalid_code_by_name' if byname else 'invalid_code_fuzzy')
                if candidate and table[candidate][key]!=name: add(nf,'Название отличается от справочника',table[candidate][key],'name_from_fuzzy')
                elif not candidate: add(nf,'Не удалось сопоставить название')
            elif byname==code:
                add(nf,'Пробелы или регистр',record[key],'canonical_name')
            elif byname and byname!=code:
                if self.advanced and typ=='station' and code in codes and byname not in codes:
                    add(nf,'Название противоречит коду и станциям клиента',record[key],'station_name_context')
                elif self.advanced and (typ=='cargo' or byname in codes or code not in codes):
                    add(cf,'Код противоречит названию',byname,'cargo_code_by_name' if typ=='cargo' else 'station_code_context')
                else:
                    add(cf,'Конфликт кода и названия'); add(nf,'Конфликт кода и названия')
            else:
                fuzzy=ref.fuzzy(typ,name) if self.advanced else None
                if fuzzy and fuzzy!=code:
                    add(cf,'Нечёткое совпадение противоречит коду',fuzzy,'code_fuzzy'); add(nf,'Нечёткое совпадение названия',table[fuzzy][key],'name_from_fuzzy')
                elif self.advanced:
                    add(nf,'Название не совпадает со справочником',record[key],typ+'_name_by_code')
                else: add(nf,'Название не совпадает со справочником')
        cargo_code=findings.get('cargo_code'); cargo=ref.cargo.get(cargo_code.candidate if cargo_code and cargo_code.candidate else r['cargo_code'])
        wagon=r['wagon_type']; capacity=ref.wagons.get(wagon)
        if capacity is None: add('wagon_type','Неизвестный тип вагона')
        if cargo and cargo['wagon_type']!=wagon:
            add('wagon_type','Тип вагона не соответствует грузу',cargo['wagon_type'] if self.advanced else None,'wagon_by_cargo')
        try:
            weight=float(r['weight_t'].replace(',','.')); count=int(r['wagons_count'])
            if capacity and count>0 and weight>capacity*count:
                fixed=weight/1000
                candidate=str(round(fixed,6)) if self.advanced and weight>capacity*count*100 and 0<fixed<=capacity*count else None
                add('weight_t','Масса превышает грузоподъёмность',candidate,'kg_to_t')
        except (ValueError,OverflowError): pass
        signature=fingerprint(r); prior=self.seen.get(signature)
        if prior and prior!=r['app_id']: add('app_id',f'Вероятный дубликат {prior}',None,'duplicate')
        self.seen.setdefault(signature,r['app_id'])
        return findings

    def decisions(self,findings):
        result={}
        for field,f in findings.items():
            score=(self.confidence.get(f.rule,0.0) if self.advanced else (1.0 if f.rule in ('canonical_name','client_name_format') else 0.0)) if f.candidate is not None else None
            # Baseline exact formatting only; Г3 uses empirical lower confidence bound.
            action='auto' if score is not None and score>=.95 and field!='app_id' else 'review'
            result[field]={'field':field,'reason':f.reason,'candidate':f.candidate,'rule':f.rule,'confidence':score,'action':action}
        return result

def equivalent(field,a,b):
    if a is None or b is None: return a==b
    if field in ('weight_t','wagons_count'):
        try: return float(a)==float(b)
        except ValueError: pass
    if field=='planned_date': return date_key(a)==date_key(b)
    return a==b

def wilson(good,total):
    if not total: return 0.0
    p=good/total; z=1.96
    return (p+z*z/(2*total)-z*sqrt(p*(1-p)/total+z*z/(4*total*total)))/(1+z*z/total)
