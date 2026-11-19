"""Предварительные проверки, не требующие внешних справочников.
Не исправляют исходные данные и не считаются полным baseline.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import re

@dataclass(frozen=True)
class Issue:
    field: str
    original: str
    reason: str
    candidate: str | None = None


def parse_date(value):
    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y'):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            pass
    return None


def inn_valid(value):
    if not re.fullmatch(r'[0-9]{10}', value):
        return False
    return sum(int(n)*w for n,w in zip(value, (2,4,10,3,5,9,4,6,8))) % 11 % 10 == int(value[-1])


def local_checks(row):
    issues = []
    def add(field, reason, candidate=None):
        issues.append(Issue(field,row[field],reason,candidate))
    for field in row:
        if field != 'split' and not row[field].strip():
            add(field,'Пустое значение')
    for field in ('client_inn','consignee_inn'):
        if not inn_valid(row[field].strip()):
            add(field,'Некорректный формат или контрольная сумма ИНН')
    for field in ('departure_station_code','destination_station_code'):
        if not re.fullmatch(r'[0-9]{6}',row[field].strip()):
            add(field,'Код станции должен содержать 6 цифр')
    try:
        created = datetime.fromisoformat(row['created_at']).date()
    except ValueError:
        created = None
        add('created_at','Не распознана дата создания')
    planned = parse_date(row['planned_date'])
    if planned is None:
        add('planned_date','Не распознана плановая дата')
    elif created and planned < created:
        add('planned_date','Плановая дата раньше даты создания')
    for field in ('weight_t','wagons_count'):
        try:
            value = Decimal(row[field].strip().replace(',','.'))
            valid = value.is_finite() and value > 0 and (field != 'wagons_count' or value == value.to_integral_value())
        except InvalidOperation:
            valid = False
        if not valid:
            add(field,'Ожидается положительное число' + (' целое' if field=='wagons_count' else ''))
    return issues
