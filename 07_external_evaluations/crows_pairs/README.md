# CrowS-Pairs: audit verdicts and evaluation outputs

Artifacts for **Appendix H** of the paper (CrowS-Pairs transfer, Tabs 33–37) and
for the LLM audit of CrowS-Pairs with Prompt 1.

## Licence: CC BY-SA 4.0 (not the research-only terms)

Every file in this folder contains or is derived from **CrowS-Pairs**
(Nangia et al., 2020, *CrowS-Pairs: A Challenge Dataset for Measuring Social
Biases in Masked Language Models*, EMNLP 2020;
<https://github.com/nyu-mll/crows-pairs>), which is released under the
[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

Because that licence is share-alike, **the files in this folder are released under
CC BY-SA 4.0 as well**. The research-use-only terms in the repository's
[`DATA_LICENSE.md`](../../DATA_LICENSE.md) apply to our own corpus (`datasets/`), not
to this folder. The script `audit/run_crows_pairs_audit.py` is code and is covered by
the repository's MIT [`LICENSE`](../../LICENSE).

**Changes made to the original material:** the CrowS-Pairs sentences were reformatted
as one record per sentence (`crows_audit_input.jsonl`); each sentence was then labelled
by an LLM auditor (verdict, confidence, one-line reason and, for `WEAK_BIAS`, a
suggested rewrite); and the pairs were scored by our BERT classifiers and by BERT-base
pseudo-log-likelihood. The original sentences themselves were not modified.

## Contents

### `audit/`: LLM audit of CrowS-Pairs with Prompt 1

| File | Content |
|---|---|
| `crows_audit_input.jsonl` | 3 016 records, one per sentence (`cp_<pair>_more` / `cp_<pair>_less` for the 1 508 pairs): `id, text, labeled_has_bias, source, role, pair_id` |
| `crows_audit_results.jsonl` | one verdict per input record: `id, verdict` (`CORRECT` / `MISLABELED` / `WEAK_BIAS`), `actual_has_bias, confidence, reason, suggested_rewrite` |
| `run_crows_pairs_audit.py` | the script that produced the verdicts: Prompt 1 verbatim, `gemini-3.1-pro-preview`, temperature 0.1, batches of 20; resumable, with a `--mock` offline mode |

Verdict distribution: 1 457 `CORRECT`, 800 `WEAK_BIAS`, 759 `MISLABELED`.

To reproduce (needs a Gemini API key in `GEMINI_API_KEY`):

```bash
pip install google-genai
python audit/run_crows_pairs_audit.py audit/crows_audit_input.jsonl results.jsonl --test   # one batch
python audit/run_crows_pairs_audit.py audit/crows_audit_input.jsonl results.jsonl          # full run
```

### `outputs/`: CrowS-Pairs transfer evaluation

Produced by [`../bert_crows_pairs_eval.ipynb`](../bert_crows_pairs_eval.ipynb).

| File | Content |
|---|---|
| `crows_pairs_predictions.csv` | per pair and per seed: `p_more`, `p_less` from our classifier |
| `crows_pairs_per_pair.csv` | per pair, averaged over seeds: probabilities, `gap`, bias type |
| `crows_pairs_per_seed.csv`, `crows_pairs_overall_corrected.csv` | PairAcc / DirAcc / MeanGap per seed (raw and corrected mapping) |
| `crows_pairs_per_category.csv`, `crows_pairs_per_stereo.csv`, `crows_pairs_per_stereo_corrected.csv` | the same metrics by bias category and by stereo / anti-stereo subset |
| `crows_pairs_bert_mlm.csv` | BERT-base full-sentence pseudo-log-likelihood per pair (notebook cell 34; not used in the paper) |
| `crows_pairs_bert_mlm_nangia.csv` | BERT-base pseudo-log-likelihood with the Nangia et al. (2020) scoring, masking only the tokens shared by both sentences (notebook cell 36; used in Tab 37) |
| `crows_pairs_independence_per_category*.csv` | our DirAcc vs BERT-base stereotype score, by category |
