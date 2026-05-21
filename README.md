# Notebooks and supporting code

> **Auditing Bias Detection Under Source Shift: A Source-Aware Benchmark and Evaluation Protocol**

This folder contains every notebook and Python module needed to reproduce the numbers in the paper. Each artifact is mapped to a specific table or appendix below.

---

## Directory layout

### `01_data_audit_counterfactuals/`

The dataset pipeline that produces **v9**.

| File | Paper reference |
|------|-----------------|
| `gemini_audit.ipynb` | **Algorithm 1 (Section 2.2)** — two-phase Gemini audit of v6 → v9. Uses **Prompts 1 and 2**. |
| `generate_counterfactuals.ipynb` | **Section 2.2 + Appendix B** — counterfactual pair generation with **Prompts 3 and 4**, plus the double-ACCEPT audit filter. The same notebook is re-run with `BIASED_CORPUS_PATH = gus_only.json` to produce the GUS counterfactuals. |
| `claude_label_audit.ipynb` | **Appendix C** — independent **Claude Sonnet 4.6** re-audit of the same 300 stratified instances (κ = 0.842 against Gemini). |
| `human_audit_tools/` | **Appendix C** — scripts used to sample and annotate the 300 stratified human-audit instances. |

The `human_audit_tools/` folder contains:

- `generate_audit_sample.py` — stratified sampling of 300 instances
- `annotate_sample.py` — CLI key-press annotator (Windows / Unix)
- `annotate_gui.py` — Tkinter GUI annotator

---

### `02_attention_pipeline/`

> ⚠️ **`bert_attention_pipeline.ipynb` is the main paper pipeline for the BERT attention-derived family.** It runs the multi-seed bias classifier, calibration, INLP, LOSO error analysis, source-only baselines, and the Appendix E artifact analysis — all on top of the 3 238-dim feature vector it builds.

| File | Paper reference |
|------|-----------------|
| `bert_attention_pipeline.ipynb` | **Appendix A** features + the BERT Attn rows of **Tables 4, 5, 11**; **Tab 8** (LOSO-Gemini errors); **Tab 19** (calibration); **Tabs 20–24** (App E artifacts); **Tabs 29–31** (App G); **Tab 32** (INLP). |
| `gpt2_attention_pipeline.ipynb` | **Appendix A** features + the GPT-2 Attn rows of **Tables 4, 5, 11**; the GPT-2 row of **Tab 32** (INLP). |

Both notebooks depend on the Python modules in [`feature_extraction_code/`](#feature_extraction_code).

---

### `03_source_domain_diagnosis/`

| File | Paper reference |
|------|-----------------|
| `bert_source_diagnosis.ipynb` | Source-domain **diagnostic** (LOSO, source recoverability, residualization) that motivated the protocol choices for the attention-derived family. The breakdown of v9 by source × label printed in cell 6 is what populates **Tab 2**. |

> ℹ️ This notebook does its own model-selection sweep (LogReg / RandomForest / XGBoost / MLP) with `HalvingRandomSearchCV`. **It does not produce the paper's headline numbers** — those come from `02_attention_pipeline/bert_attention_pipeline.ipynb`. Treat the results here as exploratory diagnostics.

---

### `04_text_baselines/`

| File | Paper reference |
|------|-----------------|
| `bert_text_baseline.ipynb` | **Section 3.2 → Tab 4** (FT BERT row), **Tab 11** (FT BERT LOSO). |
| `gpt2_text_baseline.ipynb` | **Section 3.2 → Tab 4** (FT GPT-2 row), **Tab 11** (FT GPT-2 LOSO). |

Both are multi-seed (5 seeds).

---

### `05_bias_classifiers/`

| File | Paper reference |
|------|-----------------|
| `bert_bias_classifier.ipynb` | Single-seed Attn-BERT bias classifier. **Only the TF-IDF + LogReg lexical baseline row of Tab 4 / Tab 11 comes from here** (via `scientific_utils.run_tfidf_baseline`). The Attn-BERT rows themselves are produced by `02_attention_pipeline/bert_attention_pipeline.ipynb` (multi-seed). |
| `gpt2_bias_classifier.ipynb` | Same role for GPT-2 — only the lexical baseline contribution; the Attn-GPT-2 multi-seed numbers are produced by `02_attention_pipeline/gpt2_attention_pipeline.ipynb`. |

---

### `06_edited_data_ablation/`

| File | Paper reference |
|------|-----------------|
| `bert_edited_data_ablation.ipynb` | **Section 5 → Tab 6** (variants A / B / C / D over 5 seeds). Also saves the per-instance predictions `preds_<variant>_seed_<N>.csv` consumed by the App F analysis. |
| `tables/` | **App F → Tabs 25, 26, 28** (paired permutation tests + bootstrap CIs + no-Gemini test). Contains `export_source_maps.py` (data prep — produces `v9_source_map.csv` and the per-seed `test_source_seed_<N>.csv`) plus the professor's analysis script that produces the three tables from the saved predictions. See [`tables/README.md`](06_edited_data_ablation/tables/README.md). |

> 📊 **Tabs 25, 26, 28 are produced by the code in `06_edited_data_ablation/tables/`** — not by the parent notebook. The notebook saves the per-instance predictions; the `tables/` subfolder turns those into the App F statistical analysis.

---

### `07_external_evaluations/`

| File | Paper reference |
|------|-----------------|
| `bert_heldout_pairs_eval.ipynb` | **Section 6.2 + App D → Tab 18** (counterfactual discrimination on 983 held-out pairs: PairAcc 78.2%, DirAcc 97.3%). |
| `bert_crows_pairs_eval.ipynb` | **Appendix H → Tabs 33–37** (CrowS-Pairs zero-shot transfer, anti-stereo robustness, BERT-base PLL independence). |

---

### `feature_extraction_code/`

Python modules imported by the notebooks above. **Required dependency** — without these the notebooks will not import.

```
feature_extraction_code/
└── attention_app/
    ├── models.py                          # ModelManager (tokenizer + BERT/GPT-2 loader)
    ├── metrics.py                         # GAM, flow change
    ├── head_specialization.py             # head metrics, linguistic tags
    ├── isa.py                             # Information Sub-Attention
    └── bias/
        ├── feature_extraction.py          # forward pass → attention weights
        ├── feature_extraction_notebooks.py # entry point: extract_features_for_sentence
        └── scientific_utils.py            # TF-IDF baseline, bootstrap CIs, calibration helpers
```

Functions imported by the notebooks:

- `extract_features_for_sentence` — used by `02_attention_pipeline/*` to build the 3 238-dim vector
- `run_tfidf_baseline` — the lexical TF-IDF + LogReg baseline (Tab 4 row); used by `05_bias_classifiers/*`
- `bootstrap_confidence_intervals`, `compare_with_baseline`, `analyze_error_types`, `analyze_bias_threshold`, `plot_model_calibration`, `analyze_feature_stability` — utilities for `05_bias_classifiers/*`

External dependencies: `numpy`, `torch`, `transformers`, `scikit-learn`, `pandas`.

---

### `datasets/`

Snapshots of the bias-sentences corpus across versions. **`bias_sentences_v9.json` is the dataset used in the paper** (10 304 entries, audited and paired); the earlier versions are kept for reference and for the evolution table (Tab 3).

| File | Description |
|------|-------------|
| `bias_sentences.json`     | v1 — initial 5 500-entry release (label-pure sources, source-only acc = 95.0 %) |
| `bias_sentences_v2.json`  | v2 — 5 700 entries, partial mitigation (source-only acc = 91.7 %) |
| `bias_sentences_v3.json`  | v3 — intermediate (3 sources merged with counterfactuals) |
| `bias_sentences_v4.json`  | v4 — intermediate |
| `bias_sentences_v7.json`  | v7 — after the audit-correct pass |
| `bias_sentences_v8.json`  | v8 — after re-audit |
| **`bias_sentences_v9.json`** | **v9 — final audited release**, 10 304 entries, used by every notebook in this pack |

Each entry carries `id`, `text`, `has_bias`, `source`, `role` (`original` / `counterfactual` / `strengthened` / `relabelled_only`), `pair_id`, `original_id`, `edit_spans`, and `edit_type`, so the pair-aware splits and source-aware diagnostics in the paper can be reproduced.

---

### `prompts/`

Appendix B — exact Gemini prompts used in the audit and counterfactual pipeline.

| File | Role |
|------|------|
| `1-Prompt biased-neutral.txt` | **Prompt 1** — audit (`CORRECT` / `MISLABELED` / `WEAK_BIAS`) |
| `2-Prompt neutral-biased.txt` | **Prompt 2** — repair (`FLIP` / `STRENGTHEN`) |
| `3-Prompt Audit biased-neutral.txt` | **Prompt 3** — biased → neutral counterfactual |
| `4-Prompt Audit neutral-biased.txt` | **Prompt 4** — neutral → biased counterfactual |

---

## Table → notebook map

### Body

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **1** (`tab:curation`)            | Curation summary for dataset v9 | `01_data_audit_counterfactuals/gemini_audit.ipynb` (verdict + repair counts; manual review pass tallied outside the notebook) |
| **2** (`tab:sourcecomp`)          | Final source-label composition for v9 | Computed from `bias_sentences_v9.json`; the per-source × per-label breakdown is printed in `03_source_domain_diagnosis/bert_source_diagnosis.ipynb` (cell 6) |
| **3** (`tab:evolution`)           | Evolution of the dataset across versions (v1 / v2 / v9) | v9 number from `03_source_domain_diagnosis/bert_source_diagnosis.ipynb`; v1 / v2 numbers are historical (from earlier development runs that are no longer in the pack) |
| **4** (`tab:mainresults`)         | Main results across model families (5 seeds) | **Composite**: <br>• Source-only majority on each seed's test split — computed inside each training notebook <br>• TF-IDF + LogReg — `scientific_utils.run_tfidf_baseline` called inside `05_bias_classifiers/bert_bias_classifier.ipynb` <br>• FT BERT / GPT-2 — `04_text_baselines/{bert,gpt2}_text_baseline.ipynb` <br>• Attn BERT / GPT-2 (Orig + Resid) — `02_attention_pipeline/{bert,gpt2}_extract_features.ipynb` |
| **5** (`tab:loso-main`)           | LOSO F1 by source for representative models | Subset of Tab 4 LOSO column (same notebooks) |
| **6** (`tab:ablation`)            | Edited-data ablation A / B / C / D (5 seeds) | `06_edited_data_ablation/bert_edited_data_ablation.ipynb` |
| **7** (`tab:artifacts`)           | Summary of edited-text artifact analysis | `02_attention_pipeline/bert_attention_pipeline.ipynb` (App E section, near the end) |
| **8** (`tab:loso_gemini_errors`)  | Attn-BERT LOSO-Gemini error profile | `02_attention_pipeline/bert_attention_pipeline.ipynb` (LOSO-Gemini error analysis cells) |

### Appendix A — Attention-Derived Features

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **9** (`tab:attn-features`) | Attention-derived feature groups per sentence | Documentation only; counts come from the implementation in `feature_extraction_code/attention_app/bias/feature_extraction_notebooks.py`, exercised by `02_attention_pipeline/{bert,gpt2}_extract_features.ipynb` |

### Appendix B — Detailed LOSO Results

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **11** (`tab:loso_per_source`) | Full per-source LOSO across all evaluated models | **Composite**: <br>• TF-IDF + LogReg — `scientific_utils.run_tfidf_baseline` inside `05_bias_classifiers/bert_bias_classifier.ipynb` <br>• FT BERT / GPT-2 — `04_text_baselines/` <br>• Attn BERT / GPT-2 (Orig + Resid) — `02_attention_pipeline/{bert,gpt2}_extract_features.ipynb` |

### Appendix C — Inter-rater Agreement

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **10** (`tab:confmat_prompt1`)         | Human vs. Prompt 1 (Gemini) confusion matrix | `01_data_audit_counterfactuals/claude_label_audit.ipynb` (human verdicts from `human_audit_tools/`) |
| **12** (`tab:three_way`)               | Pairwise Cohen's κ (human / Gemini / Claude), 3-class + binary | `01_data_audit_counterfactuals/claude_label_audit.ipynb` |
| **13** (`tab:cm_gemini_claude_3way`)   | Gemini vs Claude confusion matrix (3-class) | `01_data_audit_counterfactuals/claude_label_audit.ipynb` |
| **14** (`tab:cm_gemini_claude_binary`) | Gemini vs Claude confusion matrix (binary) | `01_data_audit_counterfactuals/claude_label_audit.ipynb` |
| **15** (`tab:cm_human_claude_3way`)    | Human vs Claude confusion matrix (3-class) | `01_data_audit_counterfactuals/claude_label_audit.ipynb` |
| **16** (`tab:cross-patterns`)          | Three-way agreement pattern distribution | `01_data_audit_counterfactuals/claude_label_audit.ipynb` |
| **17** (`tab:cross-by-source`)         | Auditor-human κ stratified by source | `01_data_audit_counterfactuals/claude_label_audit.ipynb` |

### Appendix D — Calibration Details

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **18** (`tab:pairwise`)     | Held-out counterfactual discrimination (PairAcc / DirAcc / MeanGap) | `07_external_evaluations/bert_heldout_pairs_eval.ipynb` |
| **19** (`tab:calibration`)  | Calibration of the main fine-tuned BERT (Uncalibrated / Platt / Temperature) | `02_attention_pipeline/bert_attention_pipeline.ipynb` (Section "Calibration (Platt, Temperature, Isotonic — Multi-Seed)") |

### Appendix E — Edited-Text Artifact Analysis

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **20** (`tab:art-length`)   | Length and lexical diversity of real vs. edited | `02_attention_pipeline/bert_attention_pipeline.ipynb` |
| **21** (`tab:art-ppl`)      | Median perplexity under GPT-2 / GPT-2 Medium / GPT-Neo | `02_attention_pipeline/bert_attention_pipeline.ipynb` |
| **22** (`tab:art-cls`)      | Real-vs-edited classifier (global + within-source) | `02_attention_pipeline/bert_attention_pipeline.ipynb` |
| **23** (`tab:art-gemini`)   | Gemini real-vs-edited decomposition by edit type | `02_attention_pipeline/bert_attention_pipeline.ipynb` |
| **24** (`tab:art-strat`)    | Gemini real-vs-edited stratified by gold label | `02_attention_pipeline/bert_attention_pipeline.ipynb` |

### Appendix F — Statistical Analysis of the Ablation

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **25** (`tab:ablation-stats-summary`)  | Paired permutation tests / bootstrap CIs summary | `06_edited_data_ablation/tables/` (analysis script on top of `preds_<variant>_seed_<N>.csv` + `test_source_seed_<N>.csv`; see [`tables/README.md`](06_edited_data_ablation/tables/README.md)) |
| **26** (`tab:ablation-stats-per-seed`) | Per-seed paired deltas with within-seed p-values | `06_edited_data_ablation/tables/` (same analysis script as Tab 25) |
| **27** (`tab:sourceonly_test`)         | Per-seed source-only majority baseline on the main test split | Computed inside each training notebook for its seed's test split (`04_text_baselines`, `05_bias_classifiers`, `06_edited_data_ablation`); the **77.6 % ± 1.8 %** mean is the aggregate of those per-seed numbers |
| **28** (`tab:exclude_gemini`)          | Per-variant Accuracy / F1 on full vs. no-Gemini test | `06_edited_data_ablation/tables/` (same analysis as Tabs 25-26; the no-Gemini test mask is built from `test_source_seed_<N>.csv`) |

### Appendix G — LOSO Error Analysis for Gemini

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **29** (`tab:formulation_rules`)       | Hierarchical formulation-type rules | `02_attention_pipeline/bert_attention_pipeline.ipynb` |
| **30** (`tab:loso_gemini_topics_full`) | Errors by topic (top 12) | `02_attention_pipeline/bert_attention_pipeline.ipynb` |
| **31** (`tab:loso_gemini_marker_rates`) | Rate of linguistic markers inside each error class | `02_attention_pipeline/bert_attention_pipeline.ipynb` |

### Appendix H — CrowS-Pairs Transfer

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **33** (`tab:crowspairs-decomp`)       | CrowS-Pairs predicted-class decomposition | `07_external_evaluations/bert_crows_pairs_eval.ipynb` |
| **34** (`tab:crowspairs-main`)         | CrowS-Pairs pair metrics (raw + corrected mapping) | `07_external_evaluations/bert_crows_pairs_eval.ipynb` |
| **35** (`tab:crowspairs-categories`)   | CrowS-Pairs per-category metrics | `07_external_evaluations/bert_crows_pairs_eval.ipynb` |
| **36** (`tab:crowspairs-antistereo`)   | CrowS-Pairs anti-stereotype robustness | `07_external_evaluations/bert_crows_pairs_eval.ipynb` |
| **37** (`tab:crowspairs-independence`) | Per-pair independence vs BERT-base PLL preferences | `07_external_evaluations/bert_crows_pairs_eval.ipynb` |

### Appendix I — Residualization vs INLP

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **32** (`tab:inlp`) | Linear residualization vs INLP on attention-derived features | `02_attention_pipeline/bert_attention_pipeline.ipynb` (BERT row) + `02_attention_pipeline/gpt2_attention_pipeline.ipynb` (GPT-2 row) |
