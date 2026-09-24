"""Tab 10 (Appendix C): confusion matrix of the human annotation against Prompt 1
(Gemini 3.1 Pro) on the 300 stratified audit instances, plus agreement and Cohen's kappa.

Input: human_audit_sample_300.csv (columns prompt1_verdict, human_verdict), produced with
the tools in ../human_audit_tools/. The Claude re-audit of the same instances
(claude_audit_300_results_4.csv) is analysed in ../claude_label_audit.ipynb (Tabs 11-16).

Usage:
    python human_vs_prompt1_matrix.py   # from this folder
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORDER = ['CORRECT', 'WEAK_BIAS', 'MISLABELED']

h = pd.read_csv(HERE / 'human_audit_sample_300.csv')
m = pd.crosstab(pd.Categorical(h['human_verdict'], ORDER), pd.Categorical(h['prompt1_verdict'], ORDER))
m.index.name, m.columns.name = 'Human verdict', 'Prompt 1 verdict'
print('Tab 10: human annotation vs Prompt 1 (Gemini 3.1 Pro)')
print(m.to_string())

M = m.values.astype(float)
n = M.sum()
po = np.trace(M) / n
pe = (M.sum(0) * M.sum(1)).sum() / n ** 2
kappa = (po - pe) / (1 - pe)
print(f'\nagreement {int(np.trace(M))}/{int(n)} = {100 * po:.1f} %   Cohen kappa = {kappa:.3f}')

PAPER = [[100, 18, 2], [0, 79, 4], [0, 3, 94]]
ok = (m.values == np.array(PAPER)).all() and round(kappa, 3) == 0.865 and int(np.trace(M)) == 273
print('Check against the paper (matrix, 273/300, kappa 0.865):', 'match' if ok else 'DIFFERS')
