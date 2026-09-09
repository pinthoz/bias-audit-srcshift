# Auditing Bias Detection Under Source Shift

**A Source-Aware Benchmark and Evaluation Protocol**

> 🎉 **Accepted at AACL-IJCNLP 2026: Main Conference** (Hengqin, China).

This repository is the reproduction pack for the paper: every notebook, prompt, dataset snapshot and Python module needed to regenerate the numbers we report. Each artifact is mapped to a specific table or appendix in the [Table → notebook map](#table--notebook-map).

---

## What the pack contains

| | |
|---|---|
| **Benchmark** | `bias_sentences_v9.json`: 10 304 audited sentences from 3 heterogeneous sources, with counterfactual pairs and per-instance edit provenance |
| **Audit pipeline** | Two-phase Gemini audit + independent Claude re-audit + a 300-instance human annotation study (Cohen's κ = 0.842 auditor-vs-auditor) |
| **Evaluation protocol** | Pair-aware splits, source-only majority baselines, Leave-One-Source-Out (LOSO), linear residualization and INLP |
| **Model families** | TF-IDF + LogReg · fine-tuned BERT / GPT-2 · attention-derived features (3 238 dims) over BERT / GPT-2 |
| **External checks** | Held-out counterfactual discrimination (983 pairs) and zero-shot CrowS-Pairs transfer |

---

## Dataset v9 at a glance

`datasets/bias_sentences_v9.json`: the release used everywhere in the paper.

| Source | Biased | Neutral | Total |
|--------|-------:|--------:|------:|
| `gemini_only` | 2 163 | 2 277 | 4 440 |
| `biased_corpus_only` | 2 066 | 1 169 | 3 235 |
| `gus_only` | 1 268 | 1 361 | 2 629 |
| **Total** | **5 497** | **4 807** | **10 304** |

| Provenance (`role` / `edit_type`) | Count |
|-----------------------------------|------:|
| `original` | 6 457 |
| `counterfactual` | 3 847 |
| of which, complete original ↔ counterfactual pairs | 3 844 pairs |
| `edit_type = strengthened` | 1 349 |
| `edit_type = relabelled_only` | 935 |

Every entry carries `id`, `text`, `has_bias`, `source`, `topic` (32 values), `role`, `pair_id`, `original_id`, `edit_spans` and `edit_type`, so the pair-aware splits and source-aware diagnostics can be reproduced exactly.

> ⚠️ **Never split randomly.** `pair_id` groups an original with its counterfactual; a random split leaks a near-duplicate of every test sentence into training. All notebooks use `GroupShuffleSplit` on `pair_id`.

---

## Quick start

### 1. Environment

Python 3.10+, then:

```bash
pip install numpy pandas scipy scikit-learn xgboost torch transformers \
            matplotlib seaborn joblib tqdm statsmodels ipython
```

Optional, only for specific notebooks: `spacy` + `nltk` (linguistic tagging in the feature extractor), `google-generativeai` (Gemini audit), `anthropic` (Claude re-audit), `python-dotenv` (API keys), `shap` (feature-importance cells).

GPU is not required but makes `04_text_baselines/` and the feature extraction in `02_attention_pipeline/` much faster.

### 2. Expected layout

The notebooks were written inside a parent project and expect to be launched from a root that contains **both** the feature-extraction package and a `dataset/` folder:

```
<project root>/
├── attention_app/            # copy of feature_extraction_code/attention_app/
├── dataset/
│   └── bias_sentences_v9.json   # copy of datasets/bias_sentences_v9.json
└── <this repo>/
```

Concretely, before running anything:

1. Copy `feature_extraction_code/attention_app/` (or `attention/`, see the note below) to your project root, or add it to `sys.path`.
2. Copy `datasets/bias_sentences_v9.json` to `<root>/dataset/bias_sentences_v9.json`.
3. **Edit the hard-coded paths.** A few cells still contain absolute Windows paths (`%cd "C:\Users\...\attention-atlas"`, and one `open(r'C:\...\unseen_bias_test.json')` in `05_bias_classifiers/gpt2_bias_classifier.ipynb`). Point these at your own root.

### 3. Suggested run order

| Step | Notebook | Produces |
|-----:|----------|----------|
| 1 | `01_data_audit_counterfactuals/gemini_audit.ipynb` → `generate_counterfactuals.ipynb` | dataset v9 (only needed if rebuilding the corpus from scratch) |
| 2 | `03_source_domain_diagnosis/bert_source_diagnosis.ipynb` | source-composition diagnostics (Tab 2) |
| 3 | `02_attention_pipeline/bert_attention_pipeline.ipynb` | the headline attention-derived results and most appendices |
| 4 | `04_text_baselines/` + `05_bias_classifiers/` | fine-tuned and lexical baseline rows |
| 5 | `06_edited_data_ablation/` then `06_edited_data_ablation/tables/` | the ablation and its statistical analysis |
| 6 | `07_external_evaluations/` | held-out pairs and CrowS-Pairs transfer |

All multi-seed experiments use `SEEDS = [1, 2, 3, 4, 5]`.

---

## Directory layout

### `01_data_audit_counterfactuals/`

The dataset pipeline that produces **v9**.

| File | Paper reference |
|------|-----------------|
| `gemini_audit.ipynb` | **Algorithm 1 (Section 2.2)**: two-phase Gemini audit of v6 → v9. Uses **Prompts 1 and 2**. |
| `generate_counterfactuals.ipynb` | **Section 2.2 + Appendix B**: counterfactual pair generation with **Prompts 3 and 4**, plus the double-ACCEPT audit filter. The same notebook is re-run with `BIASED_CORPUS_PATH = gus_only.json` to produce the GUS counterfactuals. |
| `claude_label_audit.ipynb` | **Appendix C**: independent **Claude Sonnet 4.6** re-audit of the same 300 stratified instances (κ = 0.842 against Gemini). |
| `human_audit_tools/` | **Appendix C**: scripts used to sample and annotate the 300 stratified human-audit instances. |

The `human_audit_tools/` folder contains:

- `generate_audit_sample.py`: stratified sampling of 300 instances
- `annotate_sample.py`: CLI key-press annotator (Windows / Unix)
- `annotate_gui.py`: Tkinter GUI annotator

---

### `02_attention_pipeline/` (main pipeline)

> ⚠️ **`bert_attention_pipeline.ipynb` is the main paper pipeline for the BERT attention-derived family.** It runs the multi-seed bias classifier, calibration, INLP, LOSO error analysis, source-only baselines, and the Appendix E artifact analysis, all on top of the 3 238-dim feature vector it builds.

| File | Paper reference |
|------|-----------------|
| `bert_attention_pipeline.ipynb` | **Appendix A** features + the BERT Attn rows of **Tables 4, 5, 11**; **Tab 8** (LOSO-Gemini errors); **Tab 19** (calibration); **Tabs 20–24** (App E artifacts); **Tabs 29–31** (App G); **Tab 32** (INLP). |
| `gpt2_attention_pipeline.ipynb` | **Appendix A** features + the GPT-2 Attn rows of **Tables 4, 5, 11**; the GPT-2 row of **Tab 32** (INLP). |

Both notebooks depend on the Python modules in [`feature_extraction_code/`](#feature_extraction_code).

---

### `02_feature_extraction/` (import-name twin of the above)

`bert_extract_features.ipynb` and `gpt2_extract_features.ipynb` are the **same pipeline** as `02_attention_pipeline/`, differing only in which copy of the package they import:

| Folder | Imports |
|--------|---------|
| `02_attention_pipeline/` | `from attention.models import ModelManager` |
| `02_feature_extraction/` | `from attention_app.models import ModelManager` |

Run **one** of the two, whichever matches the package name you placed on your path (`feature_extraction_code/attention/` and `feature_extraction_code/attention_app/` are byte-identical copies). Cite `02_attention_pipeline/` as the canonical version; the table map below refers to it.

---

### `03_source_domain_diagnosis/`

| File | Paper reference |
|------|-----------------|
| `bert_source_diagnosis.ipynb` | Source-domain **diagnostic** (LOSO, source recoverability, residualization) that motivated the protocol choices for the attention-derived family. The breakdown of v9 by source × label printed in cell 6 is what populates **Tab 2**. |

> ℹ️ This notebook does its own model-selection sweep (LogReg / RandomForest / XGBoost / MLP) with `HalvingRandomSearchCV`. **It does not produce the paper's headline numbers**: those come from `02_attention_pipeline/bert_attention_pipeline.ipynb`. Treat the results here as exploratory diagnostics.

---

### `04_text_baselines/`

| File | Paper reference |
|------|-----------------|
| `bert_text_baseline.ipynb` | **Section 3.2 → Tab 4** (FT BERT row), **Tab 11** (FT BERT LOSO). |
| `gpt2_text_baseline.ipynb` | **Section 3.2 → Tab 4** (FT GPT-2 row), **Tab 11** (FT GPT-2 LOSO). |
| `source_only_baseline.ipynb` | Development notebook for the **source-only majority baseline** on the earlier pre-v9 merges: the diagnostic that first quantified how much of the label was recoverable from the source alone, and that showed it dropping as counterfactuals were added. Kept for provenance; the v9 source-only numbers in **Tabs 4 and 27** are computed inside the training notebooks. |

Both fine-tuning notebooks are multi-seed (`BASELINE_SEEDS = [1, 2, 3, 4, 5]`, pair-aware `GroupShuffleSplit`).

---

### `05_bias_classifiers/`

| File | Paper reference |
|------|-----------------|
| `bert_bias_classifier.ipynb` | Single-seed Attn-BERT bias classifier. **Only the TF-IDF + LogReg lexical baseline row of Tab 4 / Tab 11 comes from here** (via `scientific_utils.run_tfidf_baseline`). The Attn-BERT rows themselves are produced by `02_attention_pipeline/bert_attention_pipeline.ipynb` (multi-seed). |
| `gpt2_bias_classifier.ipynb` | Same role for GPT-2: only the lexical baseline contribution; the Attn-GPT-2 multi-seed numbers are produced by `02_attention_pipeline/gpt2_attention_pipeline.ipynb`. |

Both also carry the **regression mini-suites** (16 control sentences + 8 hard negatives) used as sanity checks during development.

---

### `06_edited_data_ablation/`

| File | Paper reference |
|------|-----------------|
| `bert_edited_data_ablation.ipynb` | **Section 5 → Tab 6** (variants A / B / C / D over 5 seeds). Also saves the per-instance predictions `preds_<variant>_seed_<N>.csv` consumed by the App F analysis. |
| `tables/export_source_maps.py` | Data prep for App F: produces `v9_source_map.csv` and the per-seed `test_source_seed_<N>.csv` files (both checked in under `tables/`). |

> 📊 **Tabs 25, 26, 28 are produced from the files in `06_edited_data_ablation/tables/`**, not by the parent notebook: the notebook saves the per-instance predictions, `export_source_maps.py` builds the source maps, and the paired-permutation / bootstrap analysis runs on top of both.
>
> ⚠️ The analysis script itself is **not currently checked in**: only its inputs (`v9_source_map.csv`, `test_source_per_seed/`). Add it before the camera-ready so App F is reproducible end to end.

---

### `07_external_evaluations/`

| File | Paper reference |
|------|-----------------|
| `bert_heldout_pairs_eval.ipynb` | **Section 6.2 + App D → Tab 18** (counterfactual discrimination on 983 held-out pairs: PairAcc 78.2 %, DirAcc 97.3 %). |
| `bert_crows_pairs_eval.ipynb` | **Appendix H → Tabs 33–37** (CrowS-Pairs zero-shot transfer, anti-stereo robustness, BERT-base PLL independence). |

---

### `feature_extraction_code/`

Python modules imported by the notebooks above. **Required dependency**: without these the notebooks will not import.

```
feature_extraction_code/
├── attention/         ─┐  byte-identical copies; pick the one whose
└── attention_app/     ─┘  import name your notebook variant uses
    ├── models.py                          # ModelManager (tokenizer + BERT/GPT-2 loader)
    ├── metrics.py                         # GAM, flow change
    ├── head_specialization.py             # head metrics, linguistic tags
    ├── isa.py                             # Information Sub-Attention
    └── bias/
        ├── feature_extraction.py           # forward pass → attention weights
        ├── feature_extraction_notebooks.py # entry point: extract_features_for_sentence
        └── scientific_utils.py             # TF-IDF baseline, bootstrap CIs, calibration helpers
```

Functions imported by the notebooks:

- `extract_features_for_sentence`: builds the 3 238-dim vector; used by `02_attention_pipeline/*`
- `run_tfidf_baseline`: the lexical TF-IDF + LogReg baseline (Tab 4 row); used by `05_bias_classifiers/*`
- `bootstrap_confidence_intervals`, `compare_with_baseline`, `analyze_error_types`, `analyze_bias_threshold`, `plot_model_calibration`, `analyze_feature_stability`: utilities for `05_bias_classifiers/*`

External dependencies: `numpy`, `torch`, `transformers`, `scikit-learn`, `pandas`.

---

### `datasets/`

Snapshots of the bias-sentences corpus across versions. **`bias_sentences_v9.json` is the dataset used in the paper**; the earlier versions are kept for reference and for the evolution table (Tab 3).

| File | Description |
|------|-------------|
| `bias_sentences.json`     | v1: initial 5 500-entry release (label-pure sources, source-only acc = 95.0 %) |
| `bias_sentences_v2.json`  | v2: 5 700 entries, partial mitigation (source-only acc = 91.7 %) |
| `bias_sentences_v3.json`  | v3: intermediate (3 sources merged with counterfactuals) |
| `bias_sentences_v4.json`  | v4: intermediate |
| `bias_sentences_v7.json`  | v7: after the audit-correct pass |
| `bias_sentences_v8.json`  | v8: after re-audit |
| **`bias_sentences_v9.json`** | **v9: final audited release**, 10 304 entries, used by every notebook in this pack |

---

### `prompts/`

Exact Gemini prompts used in the audit and counterfactual pipeline.

| File | Role |
|------|------|
| `1-Prompt biased-neutral.txt` | **Prompt 1**: audit (`CORRECT` / `MISLABELED` / `WEAK_BIAS`) |
| `2-Prompt neutral-biased.txt` | **Prompt 2**: repair (`FLIP` / `STRENGTHEN`) |
| `3-Prompt Audit biased-neutral.txt` | **Prompt 3**: biased → neutral counterfactual |
| `4-Prompt Audit neutral-biased.txt` | **Prompt 4**: neutral → biased counterfactual |

---

## Table → notebook map

### Body

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **1** (`tab:curation`)            | Curation summary for dataset v9 | `01_data_audit_counterfactuals/gemini_audit.ipynb` (verdict + repair counts; manual review pass tallied outside the notebook) |
| **2** (`tab:sourcecomp`)          | Final source-label composition for v9 | Computed from `bias_sentences_v9.json`; the per-source × per-label breakdown is printed in `03_source_domain_diagnosis/bert_source_diagnosis.ipynb` (cell 6) |
| **3** (`tab:evolution`)           | Evolution of the dataset across versions (v1 / v2 / v9) | v9 number from `03_source_domain_diagnosis/bert_source_diagnosis.ipynb`; v1 / v2 numbers are historical (from earlier development runs that are no longer in the pack) |
| **4** (`tab:mainresults`)         | Main results across model families (5 seeds) | **Composite**: <br>• Source-only majority on each seed's test split: computed inside each training notebook <br>• TF-IDF + LogReg: `scientific_utils.run_tfidf_baseline` called inside `05_bias_classifiers/bert_bias_classifier.ipynb` <br>• FT BERT / GPT-2: `04_text_baselines/{bert,gpt2}_text_baseline.ipynb` <br>• Attn BERT / GPT-2 (Orig + Resid): `02_attention_pipeline/{bert,gpt2}_attention_pipeline.ipynb` |
| **5** (`tab:loso-main`)           | LOSO F1 by source for representative models | Subset of Tab 4 LOSO column (same notebooks) |
| **6** (`tab:ablation`)            | Edited-data ablation A / B / C / D (5 seeds) | `06_edited_data_ablation/bert_edited_data_ablation.ipynb` |
| **7** (`tab:artifacts`)           | Summary of edited-text artifact analysis | `02_attention_pipeline/bert_attention_pipeline.ipynb` (App E section, near the end) |
| **8** (`tab:loso_gemini_errors`)  | Attn-BERT LOSO-Gemini error profile | `02_attention_pipeline/bert_attention_pipeline.ipynb` (LOSO-Gemini error analysis cells) |

### Appendix A: Attention-Derived Features

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **9** (`tab:attn-features`) | Attention-derived feature groups per sentence | Documentation only; counts come from the implementation in `feature_extraction_code/attention_app/bias/feature_extraction_notebooks.py`, exercised by `02_attention_pipeline/{bert,gpt2}_attention_pipeline.ipynb` |

### Appendix B: Detailed LOSO Results

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **11** (`tab:loso_per_source`) | Full per-source LOSO across all evaluated models | **Composite**: <br>• TF-IDF + LogReg: `scientific_utils.run_tfidf_baseline` inside `05_bias_classifiers/bert_bias_classifier.ipynb` <br>• FT BERT / GPT-2: `04_text_baselines/` <br>• Attn BERT / GPT-2 (Orig + Resid): `02_attention_pipeline/{bert,gpt2}_attention_pipeline.ipynb` |

### Appendix C: Inter-rater Agreement

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **10** (`tab:confmat_prompt1`)         | Human vs. Prompt 1 (Gemini) confusion matrix | `01_data_audit_counterfactuals/claude_label_audit.ipynb` (human verdicts from `human_audit_tools/`) |
| **12** (`tab:three_way`)               | Pairwise Cohen's κ (human / Gemini / Claude), 3-class + binary | `01_data_audit_counterfactuals/claude_label_audit.ipynb` |
| **13** (`tab:cm_gemini_claude_3way`)   | Gemini vs Claude confusion matrix (3-class) | `01_data_audit_counterfactuals/claude_label_audit.ipynb` |
| **14** (`tab:cm_gemini_claude_binary`) | Gemini vs Claude confusion matrix (binary) | `01_data_audit_counterfactuals/claude_label_audit.ipynb` |
| **15** (`tab:cm_human_claude_3way`)    | Human vs Claude confusion matrix (3-class) | `01_data_audit_counterfactuals/claude_label_audit.ipynb` |
| **16** (`tab:cross-patterns`)          | Three-way agreement pattern distribution | `01_data_audit_counterfactuals/claude_label_audit.ipynb` |
| **17** (`tab:cross-by-source`)         | Auditor-human κ stratified by source | `01_data_audit_counterfactuals/claude_label_audit.ipynb` |

### Appendix D: Calibration Details

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **18** (`tab:pairwise`)     | Held-out counterfactual discrimination (PairAcc / DirAcc / MeanGap) | `07_external_evaluations/bert_heldout_pairs_eval.ipynb` |
| **19** (`tab:calibration`)  | Calibration of the main fine-tuned BERT (Uncalibrated / Platt / Temperature) | `02_attention_pipeline/bert_attention_pipeline.ipynb` (the multi-seed calibration section) |

### Appendix E: Edited-Text Artifact Analysis

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **20** (`tab:art-length`)   | Length and lexical diversity of real vs. edited | `02_attention_pipeline/bert_attention_pipeline.ipynb` |
| **21** (`tab:art-ppl`)      | Median perplexity under GPT-2 / GPT-2 Medium / GPT-Neo | `02_attention_pipeline/bert_attention_pipeline.ipynb` |
| **22** (`tab:art-cls`)      | Real-vs-edited classifier (global + within-source) | `02_attention_pipeline/bert_attention_pipeline.ipynb` |
| **23** (`tab:art-gemini`)   | Gemini real-vs-edited decomposition by edit type | `02_attention_pipeline/bert_attention_pipeline.ipynb` |
| **24** (`tab:art-strat`)    | Gemini real-vs-edited stratified by gold label | `02_attention_pipeline/bert_attention_pipeline.ipynb` |

### Appendix F: Statistical Analysis of the Ablation

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **25** (`tab:ablation-stats-summary`)  | Paired permutation tests / bootstrap CIs summary | `06_edited_data_ablation/tables/` (analysis on top of `preds_<variant>_seed_<N>.csv` + `test_source_seed_<N>.csv`) |
| **26** (`tab:ablation-stats-per-seed`) | Per-seed paired deltas with within-seed p-values | `06_edited_data_ablation/tables/` (same analysis as Tab 25) |
| **27** (`tab:sourceonly_test`)         | Per-seed source-only majority baseline on the main test split | Computed inside each training notebook for its seed's test split (`04_text_baselines`, `05_bias_classifiers`, `06_edited_data_ablation`); the **77.6 % ± 1.8 %** mean is the aggregate of those per-seed numbers |
| **28** (`tab:exclude_gemini`)          | Per-variant Accuracy / F1 on full vs. no-Gemini test | `06_edited_data_ablation/tables/` (same analysis as Tabs 25–26; the no-Gemini test mask is built from `test_source_seed_<N>.csv`) |

### Appendix G: LOSO Error Analysis for Gemini

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **29** (`tab:formulation_rules`)        | Hierarchical formulation-type rules | `02_attention_pipeline/bert_attention_pipeline.ipynb` |
| **30** (`tab:loso_gemini_topics_full`)  | Errors by topic (top 12) | `02_attention_pipeline/bert_attention_pipeline.ipynb` |
| **31** (`tab:loso_gemini_marker_rates`) | Rate of linguistic markers inside each error class | `02_attention_pipeline/bert_attention_pipeline.ipynb` |

### Appendix H: CrowS-Pairs Transfer

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **33** (`tab:crowspairs-decomp`)       | CrowS-Pairs predicted-class decomposition | `07_external_evaluations/bert_crows_pairs_eval.ipynb` |
| **34** (`tab:crowspairs-main`)         | CrowS-Pairs pair metrics (raw + corrected mapping) | `07_external_evaluations/bert_crows_pairs_eval.ipynb` |
| **35** (`tab:crowspairs-categories`)   | CrowS-Pairs per-category metrics | `07_external_evaluations/bert_crows_pairs_eval.ipynb` |
| **36** (`tab:crowspairs-antistereo`)   | CrowS-Pairs anti-stereotype robustness | `07_external_evaluations/bert_crows_pairs_eval.ipynb` |
| **37** (`tab:crowspairs-independence`) | Per-pair independence vs BERT-base PLL preferences | `07_external_evaluations/bert_crows_pairs_eval.ipynb` |

### Appendix I: Residualization vs INLP

| Table | Caption | Produced by |
|:-----:|---------|-------------|
| **32** (`tab:inlp`) | Linear residualization vs INLP on attention-derived features | `02_attention_pipeline/bert_attention_pipeline.ipynb` (BERT row) + `02_attention_pipeline/gpt2_attention_pipeline.ipynb` (GPT-2 row) |

---

## Citation

The paper has been accepted at AACL-IJCNLP 2026 (Main Conference) but is **not published yet**, so there is no citable reference yet. We are waiting for the proceedings to go online; the official ACL Anthology entry (anthology ID, pages, publisher and full author list) and a ready-to-copy BibTeX block will be added here as soon as it exists.

Until then, please cite the work as *to appear* at AACL-IJCNLP 2026.

---

## Notes and known gaps

- **Absolute paths.** A handful of cells still hard-code `C:\Users\...\attention-atlas`. Rewrite them for your environment before running (see [Quick start](#quick-start)).
- **Duplicated code.** `feature_extraction_code/attention/` and `attention_app/` are identical, as are `02_attention_pipeline/` and `02_feature_extraction/`. They exist only to match two import names; consolidating them is safe.
- **App F analysis script** is not in the repository, only its inputs. See `06_edited_data_ablation/`.
- **`04_text_baselines/source_only_baseline.ipynb`** reads the older `dataset/v2/*.csv` merges, which are not shipped here; it is kept as provenance for the confounding diagnostic, not as a v9 result.
