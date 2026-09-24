"""Per-seed diagnostics on the main real-only test split (Tab 27, Tab 4 source-only row,
and Tab 40 of Appendix M).

Tab 27 / Tab 4: source-majority baseline on each seed's test split. For every source,
the rule predicts the majority label among the non-counterfactual training rows of
that seed (every row with role != "counterfactual" that is not in the seed's test
split); AUC uses the source's share of biased training rows as the score.

Tab 40: fine-tuned BERT, ablation variant C, broken down by demographic axis
(`bias_type`). "n" is the mean number of test instances per seed, "biased" is the
share of biased instances pooled over the five seeds, and accuracy / F1 are
mean +/- SD over seeds.

Inputs (all in this repository):
    ../../datasets/bias_sentences_v9.json
    ../predictions/preds_<variant>_seed_<N>.csv
    test_source_per_seed/test_source_per_seed/test_source_seed_<N>.csv

Usage:
    python per_seed_tables.py           # from 06_edited_data_ablation/tables/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = [1, 2, 3, 4, 5]
SOURCE_CANONICAL = {'biased_corpus_only': 'biased-corpus', 'gemini_only': 'gemini', 'gus_only': 'gus-dataset'}
AXES = {'racial': 'Race', 'religious': 'Religion', 'gender': 'Gender', 'nationality': 'Nationality',
        'disability': 'Disability', 'political': 'Political', 'educational': 'Educational',
        'sexuality': 'Sexuality', 'age': 'Age', 'physical': 'Physical app.', 'socioeconomic': 'Socioeconomic'}

df = pd.DataFrame(json.load(open(ROOT / 'datasets' / 'bias_sentences_v9.json', encoding='utf-8'))['entries'])
df['label'] = df['has_bias'].astype(int)
df['source_eval'] = df['source'].map(SOURCE_CANONICAL)
df = df.set_index('id')


def test_ids(seed):
    return pd.read_csv(HERE / 'test_source_per_seed' / 'test_source_per_seed' / f'test_source_seed_{seed}.csv')['instance_id']


def table27():
    print('Tab 27: source-only majority baseline on each seed\'s test split')
    rows = []
    non_cf = df[df['role'] != 'counterfactual']
    for s in SEEDS:
        te = df.loc[test_ids(s)]
        tr = non_cf.drop(index=te.index)
        share = tr.groupby('source_eval')['label'].mean()
        score = te['source_eval'].map(share).values
        pred = (score >= 0.5).astype(int)
        y = te['label'].values
        rows.append({'seed': s, 'n_test': len(te), 'acc': (pred == y).mean(), 'f1': f1_score(y, pred),
                     'auc': roc_auc_score(y, score), 'label_maj': max(y.mean(), 1 - y.mean())})
    t = pd.DataFrame(rows).set_index('seed')
    print(t.round(3).to_string())
    print('mean ', t[['acc', 'f1', 'auc', 'label_maj']].mean().round(3).to_dict())
    print('sd   ', t[['acc', 'f1', 'auc', 'label_maj']].std(ddof=1).round(3).to_dict())
    return t


def table40():
    print('\nTab 40: fine-tuned BERT (variant C) by demographic axis')
    per = {ax: [] for ax in AXES.values()}
    pooled = {ax: [] for ax in AXES.values()}
    for s in SEEDS:
        p = pd.read_csv(HERE.parent / 'predictions' / f'preds_C_all_edited_seed_{s}.csv')
        p['bias_type'] = df.loc[p['instance_id'], 'bias_type'].values
        p['pred'] = (p['predicted_prob'] >= 0.5).astype(int)
        for bt, ax in AXES.items():
            g = p[p['bias_type'] == bt]
            per[ax].append((len(g), (g['pred'] == g['true_label']).mean(),
                            f1_score(g['true_label'], g['pred'], zero_division=0)))
            pooled[ax].extend(g['true_label'].tolist())
    rows = []
    for ax, v in per.items():
        a = np.array(v)
        rows.append({'axis': ax, 'n': a[:, 0].mean(), 'biased_%': 100 * np.mean(pooled[ax]),
                     'acc': a[:, 1].mean(), 'acc_sd': a[:, 1].std(ddof=1),
                     'f1': a[:, 2].mean(), 'f1_sd': a[:, 2].std(ddof=1)})
    t = pd.DataFrame(rows).set_index('axis')
    print(t.round(3).to_string())
    return t


PAPER27 = {1: (1281, .769, .772, .848, .586), 2: (1277, .797, .802, .865, .569), 3: (1290, .791, .787, .861, .598),
           4: (1280, .774, .782, .850, .570), 5: (1284, .752, .758, .833, .589)}
PAPER40 = {'Race': (45.4, 92.5, .977, .988), 'Religion': (24.0, 89.2, .949, .972), 'Gender': (65.0, 94.5, .929, .962),
           'Nationality': (37.6, 85.1, .925, .957), 'Disability': (23.4, 80.3, .920, .950),
           'Political': (34.4, 75.0, .915, .943), 'Educational': (22.4, 57.1, .915, .926),
           'Sexuality': (27.4, 75.2, .914, .942), 'Age': (39.6, 80.8, .914, .947),
           'Physical app.': (17.2, 65.1, .896, .925), 'Socioeconomic': (41.6, 59.1, .896, .916)}

if __name__ == '__main__':
    t27, t40 = table27(), table40()
    bad = []
    for s, (n, acc, f1, auc, lm) in PAPER27.items():
        r = t27.loc[s]
        if r.n_test != n or any(abs(round(x, 3) - e) > 1e-9 for x, e in [(r.acc, acc), (r.f1, f1), (r.auc, auc), (r.label_maj, lm)]):
            bad.append(f'Tab 27 seed {s}')
    for ax, (n, b, acc, f1) in PAPER40.items():
        r = t40.loc[ax]
        if abs(round(r.n, 1) - n) > 1e-9 or abs(round(r['biased_%'], 1) - b) > 1e-9 or abs(round(r.acc, 3) - acc) > 1e-9 or abs(round(r.f1, 3) - f1) > 1e-9:
            bad.append(f'Tab 40 {ax}')
    print('\nCheck against the paper: rows that differ =', bad or 'none')
