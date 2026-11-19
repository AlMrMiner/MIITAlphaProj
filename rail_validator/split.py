"""Разделение только официальной train-части, группировка по app_id."""
import random

SEED = 42

def holdout_ids(rows, fraction=0.2, seed=SEED):
    if not 0 < fraction < 1:
        raise ValueError('fraction must be between 0 and 1')
    ids = sorted({r['app_id'] for r in rows if r['split'] == 'train'})
    if len(ids) < 2:
        raise ValueError('Need at least two train IDs')
    random.Random(seed).shuffle(ids)
    n = min(len(ids)-1, max(1, round(len(ids)*fraction)))
    return set(ids[n:]), set(ids[:n])
