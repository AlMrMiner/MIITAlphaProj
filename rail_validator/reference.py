"""Доверенные справочники из комплекта данных."""
import csv
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from difflib import SequenceMatcher
import re

def norm(s):
    return ' '.join(s.casefold().replace('ё','е').split())

LAYOUT = str.maketrans('qwertyuiop[]asdfghjkl;\'zxcvbnm,.`QWERTYUIOP{}ASDFGHJKL:"ZXCVBNM<>~', 'йцукенгшщзхъфывапролджэячсмитьбюёйцукенгшщзхъфывапролджэячсмитьбюё')

def read_table(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

class References:
    def __init__(self, directory):
        p = Path(directory)
        self.stations = {r['station_code']:r for r in read_table(p/'stations.csv')}
        self.cargo = {r['cargo_code']:r for r in read_table(p/'cargo.csv')}
        self.clients = {r['inn']:r for r in read_table(p/'clients.csv')}
        self.wagons = {r['wagon_type']:float(r['capacity_t']) for r in read_table(p/'wagon_types.csv')}
        self.by_name = {}
        for kind, records, field in [('station',self.stations,'station_name'),('cargo',self.cargo,'cargo_name'),('client',self.clients,'client_name')]:
            idx = defaultdict(list)
            for code,r in records.items():
                idx[norm(r[field])].append(code)
            self.by_name[kind] = idx

    def exact(self, kind, name):
        options = self.by_name[kind].get(norm(name), [])
        return options[0] if len(options)==1 else None

    @lru_cache(maxsize=65536)
    def fuzzy(self, kind, name):
        text = norm(name)
        if not text: return None
        fixed = norm(name.translate(LAYOUT)) if re.search('[a-zA-Z]',name) else text
        exact = self.exact(kind, fixed)
        if exact: return exact
        scores = sorted(((max(SequenceMatcher(None,text,n).ratio(),SequenceMatcher(None,fixed,n).ratio()),n) for n in self.by_name[kind]),reverse=True)
        if scores and scores[0][0]>=.82 and (len(scores)==1 or scores[0][0]-scores[1][0]>=.08):
            return self.exact(kind,scores[0][1])
        return None
