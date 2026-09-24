# Data licence: research use only

This file governs the **data** in this repository, that is everything under
[`datasets/`](datasets/), including the audited release `bias_sentences_v9.json`
and the earlier snapshots kept for the dataset-evolution table. It also covers the
derived per-instance files under `06_edited_data_ablation/predictions/` and
`06_edited_data_ablation/tables/`, and the human and LLM audit annotations under
`01_data_audit_counterfactuals/human_audit_data/`.

It does **not** cover `07_external_evaluations/crows_pairs/`. Those files are derived
from CrowS-Pairs (Nangia et al., 2020), which is licensed CC BY-SA 4.0, and are
released under that same licence; see
[`07_external_evaluations/crows_pairs/README.md`](07_external_evaluations/crows_pairs/README.md).

The code is licensed separately, under the MIT License; see [`LICENSE`](LICENSE).

## Terms

1. **Research use only.** The dataset is released solely for non-commercial
   academic research: reproducing the results of the accompanying paper,
   benchmarking bias-detection methods, and studying source confounding in bias
   benchmarks. Commercial use, redistribution as a product, and use in deployed
   systems are not permitted under this release.

2. **Upstream terms continue to apply.** The corpus is a derived work. The
   sentences come from three origins:

   | Origin | Provenance |
   |---|---|
   | `biased_corpus_only` | derived from `ethical-spectacle/biased-corpus` (Powers et al., 2024) |
   | `gus_only` | derived from `ethical-spectacle/gus-dataset-v1` (Powers et al., 2024; GUS-Net, arXiv:2410.08388), filtered to the `gen` and `unfair/stereo` annotations |
   | `gemini_only` | synthetic sentences generated with Gemini 3.1 Pro, plus LLM-generated counterfactual and strengthened variants of items from all three origins |

   This release grants no rights beyond those already granted by the providers of
   the upstream corpora and by the terms applicable to model-generated text.
   Before using or redistributing the data, consult the licence terms published
   by those providers and comply with them. Where the upstream terms are more
   restrictive than this file, the upstream terms prevail.

3. **Attribution.** Any use of the dataset must cite the accompanying paper
   (citation to be added once the AACL-IJCNLP 2026 proceedings are published) and
   the upstream corpora listed above.

4. **Content warning.** The corpus contains stereotyping, demeaning and otherwise
   offensive statements about groups of people. This is by design: the material is
   the object of study. Its presence is not an endorsement, and the neutral
   counterparts in each counterfactual pair are labelled as such.

5. **Scope of the labels.** The operational definition of bias used here is
   deliberately strict and does not exhaust the space of social harm. Individual
   level slurs, dog whistles and contested but non-generalizing claims fall outside
   it by design. The benchmark is a research instrument for measurement, not a
   certified detector: it must not be used as the sole basis for moderation,
   screening or any decision affecting people.

6. **No warranty.** The data is provided "as is", without warranty of any kind,
   express or implied. The authors accept no liability for any claim, damages or
   other liability arising from its use.

## Personal data

The corpus consists of sentence-level text about groups, drawn from the public
corpora above and from model-generated text. It was not collected from private
communications, and no annotator identities are released. If you believe an item
contains personal data that should not be distributed, contact the authors and it
will be removed from the release.
