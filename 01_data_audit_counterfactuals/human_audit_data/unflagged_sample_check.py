"""Appendix K: human verification of 200 instances that neither audit pass ever flagged.

unflagged_sample_200.csv holds a uniform random sample (fixed seed 2902) of the 4 224
never-flagged instances, annotated with the Prompt 1 criteria. A label error is a
MISLABELED verdict (WEAK_BIAS keeps the label, it only marks a weak bias). The script
prints the counts and the one-sided 95 % Clopper-Pearson upper bound on the residual
label-error rate.

Usage:
    python unflagged_sample_check.py    # from this folder
"""
from pathlib import Path

import pandas as pd
from scipy.stats import beta

d = pd.read_csv(Path(__file__).resolve().parent / 'unflagged_sample_200.csv')
n = len(d)
errors = int((d['verdict'] == 'MISLABELED').sum())
upper = beta.ppf(0.95, errors + 1, n - errors)
print('verdicts:', d['verdict'].value_counts().to_dict())
print(f'label errors: {errors}/{n}  one-sided 95 % upper bound: {100 * upper:.1f} %')
print('Check against the paper (2 errors, 3.1 %):', 'match' if (errors == 2 and round(100 * upper, 1) == 3.1) else 'DIFFERS')
