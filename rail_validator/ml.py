"""Обучение модели, оценивающей предложенные Г3 исправления.

Эталонные correct_value используются только при обучении и оценке; в признаки не входят.
"""
import hashlib
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from .data import read_applications
from .engine import Engine, equivalent, fingerprint, wilson
from .evaluate import partition
from .reference import References, norm, read_table
from .split import SEED


def features(row, finding):
    """Признаки, доступные при поступлении заявки, без эталонной разметки."""
    old = row[finding.field]
    candidate = finding.candidate or ''
    return {
        'rule': finding.rule,
        'field': finding.field,
        'channel': row['channel'],
        'reason': finding.reason.split(';')[0],
        'original_length': len(old),
        'candidate_length': len(candidate),
        'length_difference': abs(len(old)-len(candidate)),
        'similarity': SequenceMatcher(None, norm(old), norm(candidate)).ratio(),
        'same_normalized': int(norm(old) == norm(candidate)),
        'is_numeric_old': int(old.isdecimal()),
        'is_numeric_candidate': int(candidate.isdecimal()),
        'has_cyrillic': int(any('а' <= x.lower() <= 'я' for x in old)),
        'route_codes_equal': int(row['departure_station_code'] == row['destination_station_code']),
        'wagon_count': safe_float(row['wagons_count']),
        'weight_t': safe_float(row['weight_t']),
    }


def safe_float(value):
    try:
        v = float(value.replace(',', '.'))
        return min(1000000.0, max(-1000000.0, v)) if abs(v) != float('inf') and v == v else 0.0
    except (ValueError, AttributeError):
        return 0.0


def correct_candidate(field, candidate, label_rows):
    matches = [r['correct_value'] for r in label_rows if r['field'] == field]
    return int(bool(matches) and all(equivalent(field, candidate, value) for value in matches))


def grouped_calibration_key(row):
    raw = repr(fingerprint(row)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % 5 == 0


def train_model(data_dir, output_dir, limit=None):
    """Train/holdout раздельны; внутри fit есть train/calibration для выбора порога."""
    try:
        import joblib
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError('Для ML установите зависимости: pip install -r requirements-ml.txt') from error

    p = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted((r for r in read_applications(p/'applications_raw.csv') if r['split']=='train'),
                  key=lambda r: (r['created_at'],r['app_id']))
    holdout = partition(rows)
    labels = defaultdict(list)
    for item in read_table(p/'labels_train.csv'):
        labels[item['app_id']].append(item)
    engine = Engine(References(p), advanced=True)
    train_x, train_y, calib_x, calib_y = [], [], [], []
    totals = Counter()
    for row in rows:
        findings = engine.analyze(row)  # Контекст дубликатов обновляется хронологически.
        if row['app_id'] in holdout:
            continue
        target_x, target_y = ((calib_x, calib_y) if grouped_calibration_key(row) else (train_x, train_y))
        for finding in findings.values():
            if finding.candidate is None or finding.field == 'app_id':
                continue
            target_x.append(features(row, finding))
            target_y.append(correct_candidate(finding.field, finding.candidate, labels[row['app_id']]))
        totals['fit_rows'] += 1
        if limit and totals['fit_rows'] >= limit:
            break
    if not train_x or len(set(train_y)) != 2:
        raise ValueError('Недостаточно размеченных положительных и отрицательных примеров для обучения')
    if not calib_x:
        raise ValueError('Нет примеров в калибровочной части')
    pipe = Pipeline([
        ('vectorizer', DictVectorizer(sparse=True)),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(train_x, train_y)
    probs = pipe.predict_proba(calib_x)[:, 1]
    # Используем калибровку на отдельной части fit: порог выбирается до holdout.
    # Из всех реально наблюдавшихся уровней выбирается наиболее мягкий, у которого
    # нижняя доверительная граница точности принятых кандидатов >= 0.95.
    thresholds = sorted({round(float(v), 6) for v in probs}, reverse=True)
    chosen = 1.01  # Консервативный вариант: если данных недостаточно, автоисправлений нет.
    for threshold in thresholds:
        accepted = [(float(v), int(y)) for v,y in zip(probs, calib_y) if float(v) >= threshold]
        if accepted and wilson(sum(y for _,y in accepted), len(accepted)) >= 0.95:
            chosen = threshold
    artifact = {'pipeline': pipe, 'threshold': chosen, 'seed': SEED,
                'fit_candidates': len(train_x), 'calibration_candidates': len(calib_x)}
    joblib.dump(artifact, out/'model.joblib')
    accepted = [int(y) for prob,y in zip(probs,calib_y) if float(prob) >= chosen]
    summary = {'seed':SEED, 'algorithm':'LogisticRegression + DictVectorizer',
               'training_candidates':len(train_x), 'positive_training_candidates':sum(train_y),
               'calibration_candidates':len(calib_x), 'positive_calibration_candidates':sum(calib_y),
               'probability_threshold':chosen, 'calibration_auto_count':len(accepted),
               'calibration_auto_correct':sum(accepted),
               'calibration_lower_precision_bound':wilson(sum(accepted),len(accepted)),
               'holdout_used_for_training':False}
    (out/'training.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def load_model(path):
    import joblib
    return joblib.load(path)


def batch_predictions(items, model):
    """Пакетное предсказание для списка (row, findings); ключ (app_id, field)."""
    output = {}
    records, keys = [], []
    def flush():
        if not records:
            return
        probs = model['pipeline'].predict_proba(records)[:,1]
        output.update(zip(keys, (float(p) for p in probs)))
        records.clear()
        keys.clear()
    for row, findings in items:
        for field,f in findings.items():
            if f.candidate is None or field == 'app_id':
                continue
            keys.append((row['app_id'], field))
            records.append(features(row,f))
            if len(records) >= 4096:
                flush()
    flush()
    return output

def ml_decisions(engine, row, findings, model, scores=None):
    """Г3 обнаруживает ошибки, ML решает какие кандидаты применить автоматически."""
    decisions = engine.decisions(findings)
    available = [(key,f) for key,f in findings.items() if f.candidate is not None and key != 'app_id']
    if not available:
        return decisions
    probs = ([scores[(row['app_id'], field)] for field,_ in available] if scores is not None
             else model['pipeline'].predict_proba([features(row, f) for _,f in available])[:,1])
    threshold = model['threshold']
    for (field,_), prob in zip(available, probs):
        score = float(prob)
        decisions[field]['confidence'] = score
        decisions[field]['action'] = 'auto' if score >= threshold else 'review'
    return decisions
