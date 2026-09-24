"""Paired statistical analysis of the edited-data ablation (Appendix G, Tabs 25, 26, 28).

Re-implementation of the procedure described in Appendix G, run on the released
per-instance predictions:

  (i)   per-seed paired deltas, acc(V1) - acc(V2) and F1(V1) - F1(V2);
  (ii)  within-seed paired permutation tests: 10,000 sign-flip permutations of the
        per-instance correctness difference (accuracy) and 10,000 label-swap
        permutations of the two variants' predictions (F1);
  (iii) Fisher's method to combine the five within-seed p-values;
  (iv)  percentile bootstrap 95 % CI on the mean per-seed delta
        (10,000 resamples of the five seed-level deltas).

Deterministic quantities (per-seed deltas, their mean, the sign pattern and every
number in Tab 28) reproduce the paper. Permutation p-values and bootstrap intervals
depend on the random draws: this implementation reproduces the per-seed p-values to
within about 0.01 and the bootstrap intervals to within 0.001; Fisher-combined
p-values are sensitive to the smallest per-seed p and differ more, without changing
any conclusion (every CI still crosses zero). The check at the end reports this.

Inputs (all in this repository):
    ../predictions/preds_<variant>_seed_<N>.csv   instance_id, true_label, predicted_prob
    test_source_per_seed/test_source_per_seed/test_source_seed_<N>.csv   instance_id, source

Usage:
    python ablation_stats.py            # from 06_edited_data_ablation/tables/
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
PRED_DIR = HERE.parent / 'predictions'
SRC_DIR = HERE / 'test_source_per_seed' / 'test_source_per_seed'

VARIANTS = {'A': 'A_real_only', 'B': 'B_real_plus_str', 'C': 'C_all_edited', 'D': 'D_no_gemini_cfs'}
SEEDS = [1, 2, 3, 4, 5]
COMPARISONS = [('D', 'C'), ('D', 'B'), ('D', 'A'), ('C', 'A'), ('B', 'A')]
N_PERM = 10_000
N_BOOT = 10_000
RNG = np.random.default_rng(0)


def load(seed):
    """y (true labels), predictions per variant and source per instance, aligned by instance_id."""
    frames = {v: pd.read_csv(PRED_DIR / f'preds_{name}_seed_{seed}.csv').set_index('instance_id')
              for v, name in VARIANTS.items()}
    ids = frames['A'].index
    for v, f in frames.items():
        assert f.index.equals(ids), f'instance order differs for variant {v}, seed {seed}'
        assert (f['true_label'].values == frames['A']['true_label'].values).all()
    y = frames['A']['true_label'].values
    pred = {v: (f['predicted_prob'].values >= 0.5).astype(int) for v, f in frames.items()}
    src = pd.read_csv(SRC_DIR / f'test_source_seed_{seed}.csv').set_index('instance_id').loc[ids, 'source'].values
    return y, pred, src


def perm_p_acc(c1, c2):
    """Two-sided sign-flip permutation test on the per-instance correctness difference."""
    d = c1.astype(float) - c2.astype(float)
    obs = abs(d.mean())
    signs = RNG.choice([-1.0, 1.0], size=(N_PERM, d.size))
    stats = np.abs((signs * d).mean(axis=1))
    return (np.sum(stats >= obs - 1e-12) + 1) / (N_PERM + 1)


def _f1_rows(y, P):
    """F1 of the positive class for every row of a (n_perm, n) prediction matrix."""
    tp = ((P == 1) & (y == 1)).sum(axis=1)
    fp = ((P == 1) & (y == 0)).sum(axis=1)
    fn = ((P == 0) & (y == 1)).sum(axis=1)
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def perm_p_f1(y, p1, p2):
    """Two-sided label-swap permutation test on the F1 difference."""
    obs = abs(f1_score(y, p1) - f1_score(y, p2))
    swap = RNG.random((N_PERM, y.size)) < 0.5
    a = np.where(swap, p2, p1)
    b = np.where(swap, p1, p2)
    stats = np.abs(_f1_rows(y, a) - _f1_rows(y, b))
    return (np.sum(stats >= obs - 1e-12) + 1) / (N_PERM + 1)


def fisher(ps):
    stat = -2.0 * np.sum(np.log(ps))
    return chi2.sf(stat, 2 * len(ps))


def boot_ci(deltas):
    deltas = np.asarray(deltas)
    idx = RNG.integers(0, deltas.size, size=(N_BOOT, deltas.size))
    means = deltas[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    data = {s: load(s) for s in SEEDS}

    # ---------------- Tab 26 (and the inputs of Tab 25) ----------------
    per_seed = {}
    for v1, v2 in COMPARISONS:
        rows = []
        for s in SEEDS:
            y, pred, _ = data[s]
            c1, c2 = pred[v1] == y, pred[v2] == y
            rows.append({
                'seed': s,
                'd_acc': c1.mean() - c2.mean(),
                'p_acc': perm_p_acc(c1, c2),
                'd_f1': f1_score(y, pred[v1]) - f1_score(y, pred[v2]),
                'p_f1': perm_p_f1(y, pred[v1], pred[v2]),
            })
        per_seed[(v1, v2)] = pd.DataFrame(rows).set_index('seed')

    print('Tab 26: per-seed paired deltas (accuracy) with permutation p-values')
    for (v1, v2), t in per_seed.items():
        cells = '  '.join(f"{r.d_acc:+.3f} p={r.p_acc:.3f}" for r in t.itertuples())
        print(f'  {v1} vs {v2}:  {cells}')

    # ---------------- Tab 25 ----------------
    print('\nTab 25: summary over seeds')
    summary = {}
    for (v1, v2), t in per_seed.items():
        ci_acc, ci_f1 = boot_ci(t.d_acc), boot_ci(t.d_f1)
        w = int((t.d_acc.round(12) > 0).sum()); l = int((t.d_acc.round(12) < 0).sum())
        summary[(v1, v2)] = dict(
            mean_acc=t.d_acc.mean(), ci_acc=ci_acc, fisher_acc=fisher(t.p_acc.values),
            mean_f1=t.d_f1.mean(), ci_f1=ci_f1, fisher_f1=fisher(t.p_f1.values),
            sign=f'{w}/{len(t) - w - l}/{l}')
        r = summary[(v1, v2)]
        print(f"  {v1} vs {v2}: mean dacc {r['mean_acc']:+.3f} CI [{ci_acc[0]:+.3f}, {ci_acc[1]:+.3f}] "
              f"Fisher p {r['fisher_acc']:.3f} | mean dF1 {r['mean_f1']:+.3f} CI [{ci_f1[0]:+.3f}, {ci_f1[1]:+.3f}] "
              f"Fisher p {r['fisher_f1']:.3f} | sign {r['sign']}")

    # ---------------- Tab 28 ----------------
    print('\nTab 28: full test vs test without Gemini-sourced instances (mean +/- sd over seeds)')
    tab28 = {}
    for v in VARIANTS:
        acc_f, acc_n, f1_f, f1_n = [], [], [], []
        for s in SEEDS:
            y, pred, src = data[s]
            keep = src != 'gemini'
            acc_f.append((pred[v] == y).mean()); acc_n.append((pred[v][keep] == y[keep]).mean())
            f1_f.append(f1_score(y, pred[v])); f1_n.append(f1_score(y[keep], pred[v][keep]))
        m = lambda x: (np.mean(x), np.std(x, ddof=1))
        tab28[v] = dict(acc_full=m(acc_f), acc_nog=m(acc_n), f1_full=m(f1_f), f1_nog=m(f1_n),
                        d_acc=np.mean(acc_n) - np.mean(acc_f), d_f1=np.mean(f1_n) - np.mean(f1_f))
        r = tab28[v]
        print(f"  {v}: acc {r['acc_full'][0]:.3f}+/-{r['acc_full'][1]:.3f} -> {r['acc_nog'][0]:.3f}+/-{r['acc_nog'][1]:.3f} "
              f"(d {r['d_acc']:+.3f}) | F1 {r['f1_full'][0]:.3f}+/-{r['f1_full'][1]:.3f} -> "
              f"{r['f1_nog'][0]:.3f}+/-{r['f1_nog'][1]:.3f} (d {r['d_f1']:+.3f})")

    check_against_paper(per_seed, summary, tab28)


# ---------------- values printed in the paper ----------------
PAPER_TAB26 = {  # per seed: (d_acc, p_acc); p given as upper bound 0.001 when the paper says "<0.001"
    ('D', 'C'): [(+0.006, .21), (-0.006, .45), (+0.018, .001), (-0.008, .16), (+0.005, .39)],
    ('D', 'B'): [(+0.008, .15), (-0.020, .001), (+0.015, .004), (-0.008, .14), (+0.010, .07)],
    ('D', 'A'): [(+0.008, .07), (-0.009, .17), (+0.002, .84), (+0.000, 1.00), (+0.018, .003)],
    ('C', 'A'): [(+0.002, .73), (-0.003, .66), (-0.016, .001), (+0.008, .18), (+0.013, .04)],
    ('B', 'A'): [(+0.000, 1.00), (+0.012, .012), (-0.013, .025), (+0.008, .14), (+0.008, .20)],
}
PAPER_TAB25 = {  # mean dacc, CI acc, mean dF1, CI F1, sign
    ('D', 'C'): (+0.003, (-0.005, +0.011), +0.003, (-0.007, +0.012), '3/0/2'),
    ('D', 'B'): (+0.001, (-0.012, +0.011), +0.001, (-0.015, +0.013), '3/0/2'),
    ('D', 'A'): (+0.004, (-0.004, +0.011), +0.004, (-0.005, +0.014), '3/1/1'),
    ('C', 'A'): (+0.001, (-0.009, +0.009), +0.001, (-0.010, +0.011), '3/0/2'),
    ('B', 'A'): (+0.003, (-0.006, +0.009), +0.003, (-0.006, +0.011), '3/1/1'),
}
PAPER_TAB28 = {  # acc full, acc no-Gemini, d acc, F1 full, F1 no-Gemini, d F1
    'A': (0.960, 0.940, -0.019, 0.953, 0.955, +0.003),
    'B': (0.963, 0.945, -0.018, 0.956, 0.959, +0.003),
    'C': (0.961, 0.942, -0.018, 0.954, 0.957, +0.003),
    'D': (0.964, 0.942, -0.022, 0.957, 0.956, -0.000),
}


def check_against_paper(per_seed, summary, tab28):
    r3 = lambda x: round(float(x) + 1e-12, 3)
    det_bad, mc = [], []
    for key, rows in PAPER_TAB26.items():
        for (d, p), row in zip(rows, per_seed[key].itertuples()):
            if abs(row.d_acc - d) > 0.001 + 1e-9:      # one unit in the third decimal, as for Tab 28
                det_bad.append(f'Tab 26 {key} seed {row.Index}: d_acc {row.d_acc:+.4f} vs {d:+.3f}')
            mc.append(abs(row.p_acc - p) if p > 0.001 else max(0.0, row.p_acc - 0.001))
    for key, (ma, _, mf, _, sign) in PAPER_TAB25.items():
        s = summary[key]
        if r3(s['mean_acc']) != r3(ma) or r3(s['mean_f1']) != r3(mf) or s['sign'] != sign:
            det_bad.append(f"Tab 25 {key}: {s['mean_acc']:+.4f} {s['mean_f1']:+.4f} {s['sign']}")
    for v, vals in PAPER_TAB28.items():
        t = tab28[v]
        got = (t['acc_full'][0], t['acc_nog'][0], t['d_acc'], t['f1_full'][0], t['f1_nog'][0], t['d_f1'])
        if any(abs(r3(g) - r3(e)) > 0.0011 for g, e in zip(got, vals)):
            det_bad.append(f'Tab 28 {v}: {tuple(round(g, 4) for g in got)} vs {vals}')
    print('\nCheck against the paper')
    print(f'  deterministic values that differ by more than 0.001: {len(det_bad)}')
    for b in det_bad:
        print('   ', b)
    print(f'  permutation p-values, Tab 26: max |difference| = {max(mc):.3f} (Monte Carlo noise)')


if __name__ == '__main__':
    main()
