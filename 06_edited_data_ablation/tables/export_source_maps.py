"""Export the v9 source maps used by the App F analyses (Tabs 25-28).

Produces two artifacts:

  1. ``v9_source_map.csv`` — global ``instance_id -> source`` mapping for all
     10 304 v9 instances, with the canonical source label used during
     training (``biased-corpus`` / ``gemini`` / ``gus-dataset``), the raw
     label from the JSON, ``has_bias`` and ``role``.

  2. ``test_source_per_seed/test_source_seed_<N>.csv`` (N = 1..5) — the same
     four columns but restricted to each seed's main test split. The rows
     line up one-for-one with ``preds_<variant>_seed_<N>.csv`` so the
     paired permutation tests / no-Gemini test analyses can be
     joined by ``instance_id`` without losing rows.

The per-seed test indices replicate ``pair_aware_split`` from the Colab
ablation notebook (``bert_edited_data_ablation.ipynb``): a pair-aware
GroupShuffleSplit on the non-counterfactual pool, then keeping only
``edit_type == 'original'`` in the test set.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ── Paths ────────────────────────────────────────────────────────────────
V9_JSON  = Path("dataset/bias_sentences_v9.json")
OUT_DIR  = Path("dataset")
GLOBAL_OUT = OUT_DIR / "v9_source_map.csv"
PER_SEED_DIR = OUT_DIR / "test_source_per_seed"
PER_SEED_DIR.mkdir(parents=True, exist_ok=True)

# Seeds that produced the predictions in preds_<variant>_seed_<N>.csv
SEEDS = [1, 2, 3, 4, 5]
TEST_SIZE = 0.25

# Canonical source labels — same mapping used by the training notebooks
SOURCE_CANONICAL = {
    "biased_corpus_only": "biased-corpus",
    "biased_corpus_v2":   "biased-corpus",
    "gemini_only":        "gemini",
    "gemini_only_v2":     "gemini",
    "gus_only":           "gus-dataset",
    "gus_only_v2":        "gus-dataset",
}


# ── Load v9 ──────────────────────────────────────────────────────────────
with V9_JSON.open(encoding="utf-8") as f:
    df = pd.DataFrame(json.load(f)["entries"]).copy()

df["label"] = df["has_bias"].astype(int)
df["source_canonical"] = df["source"].map(SOURCE_CANONICAL).fillna(df["source"])
y = df["label"].astype(int)


# ── (1) Global v9 source map ─────────────────────────────────────────────
global_df = (
    df[["id", "source_canonical", "source", "has_bias", "role"]]
    .rename(columns={"id": "instance_id",
                     "source_canonical": "source",
                     "source": "source_raw"})
)
global_df.to_csv(GLOBAL_OUT, index=False)
print(f"v9_source_map: {len(global_df)} rows -> {GLOBAL_OUT}")


# ── (2) Per-seed test source maps ────────────────────────────────────────
def pair_aware_test_idx(seed, test_size=TEST_SIZE):
    """Exact replica of pair_aware_split() from bert_edited_data_ablation.ipynb."""
    is_cf  = (df["role"] == "counterfactual").values
    non_cf = np.where(~is_cf)[0]

    nc_pids = df.iloc[non_cf]["pair_id"].copy()
    m_na = nc_pids.isna()
    nc_pids[m_na] = ["unpaired_" + str(i) for i in range(m_na.sum())]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, te_local = next(gss.split(non_cf, y.iloc[non_cf], groups=nc_pids.values))
    te_raw = non_cf[te_local]

    # The notebook keeps only edit_type=='original' in the test set
    te_et = df.iloc[te_raw]["edit_type"].values
    keep  = np.isin(te_et, ["original"])
    return te_raw[keep]


for seed in SEEDS:
    test_idx = pair_aware_test_idx(seed)
    sub = (
        df.iloc[test_idx][["id", "source_canonical", "source", "has_bias", "role"]]
        .rename(columns={"id": "instance_id",
                         "source_canonical": "source",
                         "source": "source_raw"})
    )
    out_path = PER_SEED_DIR / f"test_source_seed_{seed}.csv"
    sub.to_csv(out_path, index=False)
    print(f"seed {seed}: {len(sub)} rows -> {out_path}")
