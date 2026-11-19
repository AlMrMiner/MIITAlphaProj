"""Чтение исходных заявок без преобразования идентификаторов в числа."""
import csv
from pathlib import Path

FIELDS = tuple('app_id split created_at channel client_name client_inn departure_station_code departure_station_name destination_station_code destination_station_name cargo_code cargo_name wagon_type wagons_count weight_t planned_date consignee_name consignee_inn contract_number'.split())

def read_applications(path: str | Path):
    with open(path, encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Отсутствуют поля: {sorted(missing)}')
        for row in reader:
            if None in row or any(row[field] is None for field in FIELDS):
                raise ValueError(f'Некорректная строка {reader.line_num}')
            yield {field: row[field] for field in FIELDS}
