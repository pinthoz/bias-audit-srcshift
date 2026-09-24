"""CrowS-Pairs tables built from the files in this folder (Tab 33, Appendix J; Tabs 38-39,
Appendix L).

Tab 33: predicted-class decomposition of the fine-tuned BERT classifier over the 1 508
        pairs, from the `group` column of outputs/crows_pairs_per_pair.csv (written by
        ../bert_crows_pairs_eval.ipynb).
Tab 38: label audit by bias type. "Flagged" is the share of sent_less members (labelled
        neutral) that Prompt 1 judges to contain a group-level stereotype; "Clear bias" is
        the share of sent_more members (labelled biased) judged CORRECT.
Tab 39: pair-level decomposition of the audit's binary verdicts (actual_has_bias of both
        members), next to the classifier decomposition of Tab 33.

Inputs: audit/crows_audit_input.jsonl, audit/crows_audit_results.jsonl,
        outputs/crows_pairs_per_pair.csv

Usage:
    python crows_audit_tables.py        # from 07_external_evaluations/crows_pairs/
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
inp = pd.DataFrame([json.loads(l) for l in open(HERE / 'audit' / 'crows_audit_input.jsonl', encoding='utf-8')])
res = pd.DataFrame([json.loads(l) for l in open(HERE / 'audit' / 'crows_audit_results.jsonl', encoding='utf-8')])
pp = pd.read_csv(HERE / 'outputs' / 'crows_pairs_per_pair.csv')

d = inp.merge(res, on='id', how='left', validate='one_to_one')
assert d['verdict'].notna().all(), 'every input sentence must have a verdict'
d['pid'] = d['pair_id'].str.replace('cp_', '', regex=False).astype(int)
d = d.merge(pp[['pair_id', 'bias_type']].rename(columns={'pair_id': 'pid'}), on='pid', how='left')

# ---------------- Tab 33 ----------------
groups = {'both_biased': 'Both predicted biased', 'both_neutral': 'Neither predicted biased',
          'correct_pair': 'Only sent_more biased (PairAcc)', 'reversed': 'Inverted (only sent_less biased)'}
t33 = pp['group'].value_counts().reindex(list(groups))
print('Tab 33: CrowS-Pairs predicted-class decomposition')
for g, n in t33.items():
    print(f'  {groups[g]:36s} {n:5d}  {100 * n / len(pp):5.1f} %')

# ---------------- Tab 38 ----------------
neutral = d[d['labeled_has_bias'] == False]      # sent_less members
biased = d[d['labeled_has_bias'] == True]        # sent_more members
t38 = pd.DataFrame({
    'pairs': neutral.groupby('bias_type').size(),
    'flagged_%': 100 * neutral.groupby('bias_type')['actual_has_bias'].mean(),
    'clear_bias_%': 100 * biased.groupby('bias_type')['verdict'].apply(lambda v: (v == 'CORRECT').mean()),
}).sort_values('flagged_%', ascending=False)
t38.loc['All'] = [len(neutral), 100 * neutral['actual_has_bias'].mean(), 100 * (biased['verdict'] == 'CORRECT').mean()]
print('\nTab 38: label audit of CrowS-Pairs by bias type')
print(t38.round(1).to_string())

# ---------------- Tab 39 ----------------
w = d.pivot_table(index='pid', columns=d['id'].str.rsplit('_', n=1).str[1],
                  values='actual_has_bias', aggfunc='first').astype(bool)
audit = {'Both members biased': (w['more'] & w['less']).mean(),
         'Neither member biased': (~w['more'] & ~w['less']).mean(),
         'Only sent_more biased': (w['more'] & ~w['less']).mean(),
         'Only sent_less biased': (~w['more'] & w['less']).mean()}
clf = dict(zip(audit, (t33 / len(pp)).values))
print('\nTab 39: pair-level decomposition (label audit vs classifier)')
for k in audit:
    print(f'  {k:24s} audit {100 * audit[k]:5.1f} %   classifier {100 * clf[k]:5.1f} %')

# ---------------- check against the paper ----------------
PAPER33 = {'both_biased': 797, 'both_neutral': 543, 'correct_pair': 128, 'reversed': 40}
PAPER38 = {'religion': (105, 45.7, 41.9), 'race-color': (516, 41.7, 37.0), 'sexual-orientation': (84, 41.7, 32.1),
           'nationality': (159, 37.1, 35.8), 'disability': (60, 33.3, 30.0), 'socioeconomic': (172, 27.3, 24.4),
           'age': (87, 20.7, 20.7), 'gender': (262, 18.7, 16.4), 'physical-appearance': (63, 14.3, 14.3),
           'All': (1508, 33.2, 29.8)}
PAPER39_AUDIT = [33.0, 20.0, 46.9, 0.2]
bad = [g for g, n in PAPER33.items() if t33[g] != n]
bad += [bt for bt, (n, f, c) in PAPER38.items()
        if t38.loc[bt, 'pairs'] != n or round(t38.loc[bt, 'flagged_%'], 1) != f or round(t38.loc[bt, 'clear_bias_%'], 1) != c]
bad += [k for k, e in zip(audit, PAPER39_AUDIT) if round(100 * audit[k], 1) != e]
print('\nCheck against the paper: values that differ =', bad or 'none')
