# Auditing Bias Detection Under Source Shift

**A Source-Aware Benchmark and Evaluation Protocol**

**Ana Pinto** and **Álvaro Figueira**
Department of Computer Science, Faculty of Sciences, University of Porto

> 🎉 **Accepted at AACL-IJCNLP 2026: Main Conference** (Hengqin, China).
> **Citation to be added** once the proceedings are published (see [Citation](#citation)).

This repository is the reproduction pack for the paper: every notebook, prompt, dataset snapshot and Python module needed to regenerate the numbers we report. Each artifact is mapped to a specific table or appendix in the [Table → notebook map](#table--notebook-map).

**Licensing.** The code is released under the MIT License ([`LICENSE`](LICENSE)). The audited dataset and the per-instance predictions are released separately, **for research use only**, under the terms in [`DATA_LICENSE.md`](DATA_LICENSE.md). The one exception is `07_external_evaluations/crows_pairs/`, which is derived from CrowS-Pairs and therefore stays under **CC BY-SA 4.0**.

---

## Repository structure

```
bias-audit-srcshift/
├── 01_data_audit_counterfactuals/   audit + counterfactual generation, human-audit tools
├── 02_attention_pipeline/           main pipeline: attention-derived features, LOSO, calibration, INLP
├── 03_source_domain_diagnosis/      source-recoverability diagnostics
├── 04_text_baselines/               fine-tuned BERT / GPT-2 baselines, source-only baseline
├── 05_bias_classifiers/             single-seed classifiers + TF-IDF lexical baseline
├── 06_edited_data_ablation/         edited-data ablation (variants A-D), 20 per-instance prediction files
├── 07_external_evaluations/         held-out counterfactual pairs, CrowS-Pairs audit and transfer
├── datasets/                        corpus snapshots; bias_sentences_v9.json is the release
├── feature_extraction_code/         the `attention` package imported by the notebooks
├── prompts/                         the four prompts of Appendix B (audit, repair, counterfactual generation) and the counterfactual acceptance rules
├── LICENSE                          MIT, code only
└── DATA_LICENSE.md                  research-use-only terms for the dataset
```

Each folder is documented in [Directory layout](#directory-layout) below.

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

## Artifacts referenced in the paper

| Artifact | Where it lives | Status |
|---|---|---|
| **Audited dataset** with role and audit provenance | [`datasets/bias_sentences_v9.json`](datasets/bias_sentences_v9.json) | present (10 304 instances; `role`, `edit_type`, `pair_id`, `original_id`, `edit_spans`) |
| **Per-instance predictions**, 4 ablation variants × 5 seeds (20 files) | [`06_edited_data_ablation/predictions/`](06_edited_data_ablation/predictions/) | present |
| **CrowS-Pairs audit verdicts** (Prompt 1 on all 3 016 CrowS-Pairs sentences) and transfer outputs | [`07_external_evaluations/crows_pairs/`](07_external_evaluations/crows_pairs/) | present (CC BY-SA 4.0, see below) |
| **Evaluation and statistical-test code** | [`02_attention_pipeline/`](02_attention_pipeline/), [`07_external_evaluations/`](07_external_evaluations/), [`06_edited_data_ablation/tables/`](06_edited_data_ablation/tables/) | present: evaluation notebooks, the CrowS-Pairs audit script, and `06_edited_data_ablation/tables/ablation_stats.py` for the App G paired tests (a re-implementation; see [Notes](#notes-and-known-gaps)) |

The per-instance predictions join one-to-one, by `instance_id`, with the per-seed
test-split source maps in
[`06_edited_data_ablation/tables/`](06_edited_data_ablation/tables/)
(`v9_source_map.csv`, `test_source_seed_<N>.csv`); those are the inputs of the App G
analysis.

> ⚖️ The CrowS-Pairs files are derived from CrowS-Pairs (Nangia et al., 2020), which is
> licensed CC BY-SA 4.0. They are therefore released under **CC BY-SA 4.0**, not under
> the research-only terms of [`DATA_LICENSE.md`](DATA_LICENSE.md). Details in
> [`07_external_evaluations/crows_pairs/README.md`](07_external_evaluations/crows_pairs/README.md).

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
├── attention/                # copy of feature_extraction_code/attention/
├── dataset/
│   └── bias_sentences_v9.json   # copy of datasets/bias_sentences_v9.json
└── <this repo>/
```

Concretely, before running anything:

1. Copy `feature_extraction_code/attention/` to your project root, or add it to `sys.path`. Every notebook imports it as `attention` (`from attention.models import ModelManager`).
2. Copy `datasets/bias_sentences_v9.json` to `<root>/dataset/bias_sentences_v9.json`.
3. **Edit the hard-coded paths.** A few cells still contain absolute Windows paths (`%cd "C:\Users\...\project"`, and one `open(r'C:\...\unseen_bias_test.json')` in `05_bias_classifiers/gpt2_bias_classifier.ipynb`). Point these at your own root.

### 3. Suggested run order

| Step | Notebook | Produces |
|-----:|----------|----------|
| 1 | `01_data_audit_counterfactuals/gemini_audit.ipynb` → `generate_counterfactuals.ipynb` | dataset v9 (only needed if rebuilding the corpus from scratch) |
| 2 | `03_source_domain_diagnosis/bert_source_diagnosis.ipynb` | source-domain diagnostics and the held-out pairs of Tab 18 |
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
| `human_audit_data/` | **Appendices C and K**: the 300-instance human audit with Prompt 1 verdicts (`human_audit_sample_300.csv`), the Claude re-audit of the same instances (`claude_audit_300_results_4.csv`, read by `claude_label_audit.ipynb`; set its `DATA_DIR` to this folder), the 200 never-flagged instances verified in Appendix K (`unflagged_sample_200.csv`), and two scripts: `human_vs_prompt1_matrix.py` (Tab 10) and `unflagged_sample_check.py` (Appendix K: 2 label errors, 3.1 % upper bound). |

The `human_audit_tools/` folder contains:

- `generate_audit_sample.py`: stratified sampling of 300 instances
- `annotate_sample.py`: CLI key-press annotator (Windows / Unix)
- `annotate_gui.py`: Tkinter GUI annotator

---

### `02_attention_pipeline/` (main pipeline)

> ⚠️ **`bert_attention_pipeline.ipynb` is the main paper pipeline for the BERT attention-derived family.** It runs the multi-seed bias classifier, calibration, INLP, LOSO error analysis, source-only baselines, and the Appendix E artifact analysis, all on top of the 3 238-dim feature vector it builds.

| File | Paper reference |
|------|-----------------|
| `bert_attention_pipeline.ipynb` | **Appendix A** features; **Tab 2**; the TF-IDF, FT-BERT and Attn-BERT rows of **Tabs 4, 5, 17**; **Tab 8** and **Tabs 29–31** (App H); **Tab 19** and **Fig 2** (App E); **Tab 7** and **Tabs 20–24** (App F, cell 33); **Tab 32** (App I). |
| `gpt2_attention_pipeline.ipynb` | **Appendix A** features; the GPT-2 Attn rows of **Tabs 4 and 17**; the GPT-2 rows of **Tab 32** (App I). |

Both notebooks depend on the Python modules in [`feature_extraction_code/`](#feature_extraction_code).

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
| `bert_text_baseline.ipynb` | Trains the FT BERT rows of **Tabs 4, 5, 17** (no saved output; the values are printed in `02_attention_pipeline/bert_attention_pipeline.ipynb`, cell 40). |
| `gpt2_text_baseline.ipynb` | FT GPT-2 rows of **Tabs 4 and 17** (cells 7–8). |
| `source_only_baseline.ipynb` | Development notebook for the **source-only majority baseline** on the earlier pre-v9 merges: the diagnostic that first quantified how much of the label was recoverable from the source alone, and that showed it dropping as counterfactuals were added. Kept for provenance; the v9 source-only numbers in **Tabs 4 and 27** are computed inside the training notebooks. |

Both fine-tuning notebooks are multi-seed (`BASELINE_SEEDS = [1, 2, 3, 4, 5]`, pair-aware `GroupShuffleSplit`).

---

### `05_bias_classifiers/`

| File | Paper reference |
|------|-----------------|
| `bert_bias_classifier.ipynb` | Single-seed Attn-BERT bias classifier, used during development. It runs its own split and its own TF-IDF comparison (`scientific_utils.run_tfidf_baseline`), and **does not produce any number in the paper**. The paper's TF-IDF row comes from `02_attention_pipeline/bert_attention_pipeline.ipynb` (cells 38–39) and the Attn-BERT rows from the same notebook. |
| `gpt2_bias_classifier.ipynb` | Same role for GPT-2: only the lexical baseline contribution; the Attn-GPT-2 multi-seed numbers are produced by `02_attention_pipeline/gpt2_attention_pipeline.ipynb`. |

Both also carry the **regression mini-suites** (16 control sentences + 8 hard negatives) used as sanity checks during development.

---

### `06_edited_data_ablation/`

| File | Paper reference |
|------|-----------------|
| `bert_edited_data_ablation.ipynb` | **Section 6 → Tab 6** (variants A / B / C / D over 5 seeds) and **Tab 28**, cell 9. Also saves the per-instance predictions `preds_<variant>_seed_<N>.csv` consumed by the App G analysis and by Tab 40. |
| `predictions/preds_<variant>_seed_<N>.csv` | The 20 per-instance prediction files (variants `A_real_only`, `B_real_plus_str`, `C_all_edited`, `D_no_gemini_cfs` × seeds 1–5). Columns: `instance_id, true_label, predicted_prob`. Each file covers exactly the test split of its seed (1 277 to 1 290 instances). Written by cells 12–14 of `bert_edited_data_ablation.ipynb`, together with `manifest.csv` (per-file test size, accuracy and F1). |
| `tables/export_source_maps.py` | Data prep for App G: produces `v9_source_map.csv` and the per-seed `test_source_seed_<N>.csv` files (both checked in under `tables/`). |
| `tables/ablation_stats.py` | **Tabs 25, 26, 28**: paired permutation tests, Fisher combination and bootstrap CIs on the per-instance predictions, following Appendix G. |
| `tables/per_seed_tables.py` | **Tab 27** (and the source-only row of Tab 4) and **Tab 40**. |

> 📊 **Tabs 25 and 26 are produced from the per-instance predictions plus the source maps**, not by the parent notebook: the notebook saves `predictions/`, `export_source_maps.py` builds the source maps in `tables/`, and the paired-permutation / bootstrap analysis runs on top of both.
>
> ⚠️ The analysis script itself is **not yet checked in**; both of its inputs are. It will be added so App G is reproducible end to end.

---

### `07_external_evaluations/`

| File | Paper reference |
|------|-----------------|
| `bert_heldout_pairs_eval.ipynb` | Per-source breakdown of held-out counterfactual discrimination for the fine-tuned BERT. It uses the main fixed split, whose test side leaves 790 held-out pairs. The **983-pair figures of Tab 18** (PairAcc 78.2 %, DirAcc 97.3 %, MeanGap 0.705) come from `03_source_domain_diagnosis/bert_source_diagnosis.ipynb`, cell 23. |
| `bert_crows_pairs_eval.ipynb` | **Appendix J → Tabs 33–37** (CrowS-Pairs zero-shot transfer, anti-stereo robustness, BERT-base PLL independence). |
| `crows_pairs/audit/` | LLM audit of all 3 016 CrowS-Pairs sentences with Prompt 1: input, verdicts (1 457 `CORRECT`, 800 `WEAK_BIAS`, 759 `MISLABELED`) and the script that produced them. |
| `crows_pairs/outputs/` | Per-pair, per-seed, per-category and PLL outputs of `bert_crows_pairs_eval.ipynb`. |
| `crows_pairs/crows_audit_tables.py` | **Tabs 33, 38, 39**, built from the audit verdicts and the per-pair outputs. |

> ⚖️ Everything under `crows_pairs/` is **CC BY-SA 4.0**, inherited from CrowS-Pairs. See [`crows_pairs/README.md`](07_external_evaluations/crows_pairs/README.md).

---

### `feature_extraction_code/`

Python modules imported by the notebooks above. **Required dependency**: without these the notebooks will not import.

```
feature_extraction_code/
└── attention/                             # imported as `attention` by every notebook
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
- `run_tfidf_baseline`: a TF-IDF + LogReg comparison used by `05_bias_classifiers/*` during development (the paper's TF-IDF row uses a different configuration, in `02_attention_pipeline/bert_attention_pipeline.ipynb`, cell 38)
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
| `Prompt1-Audit.txt` | **Prompt 1**: label audit (`CORRECT` / `MISLABELED` / `WEAK_BIAS`). Used verbatim by `gemini_audit.ipynb` (cell 9), by the Claude re-audit and by `07_external_evaluations/crows_pairs/audit/run_crows_pairs_audit.py` |
| `Prompt2-Repair.txt` | **Prompt 2**: repair (`FLIP` / `STRENGTHEN`). Used by `gemini_audit.ipynb` (cell 18) |
| `1-Prompt biased-neutral.txt` | **Prompt 3**: biased → neutral counterfactual generation (`generate_counterfactuals.ipynb`) |
| `2-Prompt neutral-biased.txt` | **Prompt 4**: neutral → biased counterfactual generation |
| `3-Prompt Audit biased-neutral.txt` | Acceptance rules for Prompt 3 outputs (`ACCEPT` / `REJECT`, the double-ACCEPT filter) |
| `4-Prompt Audit neutral-biased.txt` | Acceptance rules for Prompt 4 outputs |

---

## Table → notebook map

Numbering follows the **camera-ready** version (Tables 1–40, Figures 1–2, Appendices A–M).
"Cell N" is the 0-based index of the cell in the notebook (the first cell is cell 0).

Status legend:
**printed**: the number appears in the saved output of that cell.
**recompute**: the code or data is in the repository and reproduces the number exactly, but the output was not saved.
**script**: a standalone script in this repository regenerates the table and checks it against the values in the paper.

### Evaluation splits

Three splits appear in the paper. They come from the same procedure (`GroupShuffleSplit` by `pair_id`, 25 % test), differing in `random_state` and in whether the test side is restricted to `edit_type == original`:

| Split | Train / test / held-out CFs | Used by |
|---|---|---|
| Diagnostic split (`random_state=42`, no originals-only filter), `03_source_domain_diagnosis/` | 7 703 / 1 615 / 986 | Tab 18 |
| Main fixed split (`random_state=42`, originals-only test), `bert_bias_classifier_v9_split.npz` | 8 229 (6 580 fit + 1 649 val) / 1 282 / 793 | Tabs 4, 5, 17, 19; Fig 2 |
| Per-seed splits (`random_state` = 1…5, originals-only test), `pair_aware_split` in `06_edited_data_ablation/` | ≈ 8 240 / 1 277–1 290 / 773–804 | Tabs 6, 25–28, 40; source-only row of Tab 4 |

### Body

| Table | Content | Source | Status |
|:-----:|---------|--------|--------|
| **1** | Curation summary | `01_data_audit_counterfactuals/gemini_audit.ipynb`, cells 12–15 (pass 1 and repair); final roles from `datasets/bias_sentences_v9.json`. The second audit pass (10 144 / 272) and the manual review (160 corrected, 112 removed) were done outside the notebook. | printed / partly manual |
| **2** | Final source–label composition | `02_attention_pipeline/bert_attention_pipeline.ipynb`, cell 36 | printed |
| **3** | Source-only baseline before / after the audit | v9 value as in Tab 2; the v1 / v2 values are historical and not reproducible from this pack | historical |
| **4** | Main results | **Composite.** Source-only row: mean of Tab 27. TF-IDF: `02_…/bert_attention_pipeline.ipynb` cell 38. FT BERT / GPT-2: trained in `04_text_baselines/{bert,gpt2}_text_baseline.ipynb` (the GPT-2 notebook prints them in cells 7–8; the BERT notebook has no saved output, its numbers are printed in the consolidated table of `bert_attention_pipeline.ipynb` cell 40). Attention rows: `02_…/bert_attention_pipeline.ipynb` cells 30, 40 and `gpt2_attention_pipeline.ipynb` cells 44–45 | printed |
| **5** | LOSO F1 by source | TF-IDF: `bert_attention_pipeline.ipynb` cell 39; FT-BERT and Attn-BERT: `bert_attention_pipeline.ipynb` cell 40 (consolidated table) | printed |
| **6** | Edited-data ablation A–D | `06_edited_data_ablation/bert_edited_data_ablation.ipynb`, cell 9 | printed |
| **7** | Artifact summary | `bert_attention_pipeline.ipynb`, cell 33 (see Tabs 20–24) | recompute |
| **8** | LOSO-Gemini error profile | `bert_attention_pipeline.ipynb`, cell 31 (the ≥ 15-token row sums two length bins) | printed |

### Appendices

| Table | Appendix | Content | Source | Status |
|:-----:|:--------:|---------|--------|--------|
| **9** | A | Attention-derived feature groups | `feature_extraction_code/attention/bias/feature_extraction_notebooks.py` (3 238 features) | documentation |
| **10** | C | Human vs Prompt 1 confusion matrix | `01_data_audit_counterfactuals/human_audit_data/human_vs_prompt1_matrix.py` (κ = 0.865 also printed in `claude_label_audit.ipynb`, cell 22) | script |
| **11** | C | Pairwise Cohen's κ | `claude_label_audit.ipynb`, cell 22 | printed |
| **12** | C | Gemini vs Claude (3-class) | `claude_label_audit.ipynb`, cell 17 (columns in alphabetical order) | printed |
| **13** | C | Gemini vs Claude (binary) | collapsed from Tab 12 | printed |
| **14** | C | Human vs Claude | `claude_label_audit.ipynb`, cells 14–15 (heatmap + classification report) | printed |
| **15** | C | Three-way agreement patterns | `claude_label_audit.ipynb`, cell 20 | printed |
| **16** | C | Auditor–human κ by source | `claude_label_audit.ipynb`, cell 24 | printed |
| **17** | D | Cross-source robustness, all models | Composite, as Tab 4 (TF-IDF: `bert_attention_pipeline.ipynb` cells 38–39) | printed |
| **18** | E | Held-out counterfactual discrimination (983 pairs) | `03_source_domain_diagnosis/bert_source_diagnosis.ipynb`, cell 23 | printed |
| **19**, **Fig 2** | E | Calibration and reliability diagram | `bert_attention_pipeline.ipynb`, cells 58–59 | printed |
| **20** | F | Length and lexical diversity | `bert_attention_pipeline.ipynb`, cell 33 | recompute |
| **21** | F | Perplexity under three LMs | `bert_attention_pipeline.ipynb`, cell 33 (section 4; needs GPT-2, GPT-2 Medium, GPT-Neo) | recompute (GPU recommended) |
| **22–24** | F | Real-vs-edited classifier (global, within source, Gemini decomposition) | `bert_attention_pipeline.ipynb`, cell 33 | recompute |
| **25–26** | G | Paired permutation tests and bootstrap CIs | `06_edited_data_ablation/tables/ablation_stats.py`, on the predictions written by `bert_edited_data_ablation.ipynb` cells 12–14 | script (re-implementation) |
| **27** | G | Per-seed source-only baseline | `06_edited_data_ablation/tables/per_seed_tables.py` | script |
| **28** | G | Full test vs no-Gemini test | `06_edited_data_ablation/bert_edited_data_ablation.ipynb`, cell 9 | printed |
| **29–31** | H | Formulation rules, errors by topic, marker rates | `bert_attention_pipeline.ipynb`, cell 31 | printed |
| **32** | I | Residualization vs INLP | `bert_attention_pipeline.ipynb` cells 28–29 and `gpt2_attention_pipeline.ipynb` cells 27–28 (Original rows, BERT Linear row); the INLP cells (26 and 25) have no saved output and the GPT-2 Linear source accuracy (0.348) is not printed | printed / recompute |
| **33** | J | CrowS-Pairs predicted-class decomposition | `07_external_evaluations/bert_crows_pairs_eval.ipynb`, cells 21–22; `crows_pairs/crows_audit_tables.py` | printed / script |
| **34** | J | CrowS-Pairs pair metrics | `bert_crows_pairs_eval.ipynb`, cell 32 | printed |
| **35** | J | Pair metrics per category | `bert_crows_pairs_eval.ipynb`, cell 28 | printed |
| **36** | J | Anti-stereotype robustness | `bert_crows_pairs_eval.ipynb`, cells 30, 32 | printed |
| **37** | J | Independence from BERT-base PLL (Nangia et al. scoring) | `bert_crows_pairs_eval.ipynb`, cell 36 (stereotype score 60.48 %, writes `crows_pairs_bert_mlm_nangia.csv`) and cell 40 (Pearson +0.040, Spearman −0.002). Cells 34 and 38 are the full-sentence PLL variant, not used in the paper | printed |
| **38–39** | L | Label audit of CrowS-Pairs | `07_external_evaluations/crows_pairs/crows_audit_tables.py` | script |
| **40** | M | Fine-tuned BERT by demographic axis | `06_edited_data_ablation/tables/per_seed_tables.py` | script |

Figure 1 is a diagram and is not generated by code.

---

## Citation

**Citation to be added.** The paper has been accepted at AACL-IJCNLP 2026 (Main Conference) but the proceedings are not out yet. The final bibliographic entry (ACL Anthology ID, pages, publisher) will replace the one below as soon as they are published.

Until then, please cite the OpenReview record:

```bibtex
@inproceedings{pinto2026auditing,
  title     = {Auditing Bias Detection Under Source Shift: A Source-Aware Benchmark and Evaluation Protocol},
  author    = {Pinto, Ana and Figueira, {\'A}lvaro},
  booktitle = {The 5th Asia-Pacific Chapter of the Association for Computational Linguistics {\&} the 15th International Joint Conference on Natural Language Processing},
  year      = {2026},
  url       = {https://openreview.net/forum?id=1L3GSyq3I1}
}
```

---

## Notes and known gaps

- **Absolute paths.** A handful of cells still hard-code `C:\Users\...\project`. Rewrite them for your environment before running (see [Quick start](#quick-start)).
- **`attention_app/` in saved paths.** Some notebooks read and write model artifacts under `<root>/attention_app/bias/models/`. That is an output folder in the parent project, not the Python package: the package to put on your path is `feature_extraction_code/attention/`.
- **App G tests are a re-implementation.** The original analysis script for Tabs 25–26 is not available; `ablation_stats.py` follows the procedure of Appendix G on the released predictions. Deterministic values match the paper; permutation p-values and bootstrap CIs match up to Monte Carlo noise (about 0.01 and 0.001).
- **Outputs not saved.** The artefact analysis (cell 33 of `bert_attention_pipeline.ipynb`, Tabs 7 and 20–24) and the INLP cells (cell 26 of the BERT notebook, cell 25 of the GPT-2 notebook, Tab 32) were saved without outputs. Re-running cell 33 reproduces Tabs 7, 20 and 22–24 exactly; Tab 21 additionally needs three language models.
- **`04_text_baselines/source_only_baseline.ipynb`** reads the older `dataset/v2/*.csv` merges, which are not shipped here; it is kept as provenance for the confounding diagnostic, not as a v9 result.
