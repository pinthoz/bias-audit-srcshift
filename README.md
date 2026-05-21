Notebooks and supporting code
"Auditing Bias Detection Under Source Shift"

Each notebook maps to a specific section or appendix of the paper.

Structure
---------
01_data_audit_counterfactuals/
    gemini_audit.ipynb              Algorithm 1 (Section 2.2): two-phase
                                    Gemini audit of v6 -> produces v9.
                                    Uses Prompts 1 and 2.
    generate_counterfactuals.ipynb  Section 2.2 + Appendix B: counterfactual
                                    pair generation with Prompts 3 and 4,
                                    plus the double-ACCEPT audit filter.
    claude_label_audit.ipynb        Appendix C: independent Claude Sonnet 4.6
                                    re-audit of the same 300 stratified
                                    instances (kappa = 0.842 against Gemini).
    human_audit_tools/              Appendix C: scripts used to sample and
                                    annotate the 300 stratified human-audit
                                    instances.
        generate_audit_sample.py    Stratified sampling of 300 instances.
        annotate_sample.py          CLI key-press annotator (Win/Unix).
        annotate_gui.py             Tkinter GUI annotator.

02_feature_extraction/
    bert_extract_features.ipynb  Appendix A: builds the 3238-dim attention
    gpt2_extract_features.ipynb  feature vector used by the Attn models.
                                 (uses code in feature_extraction_code/)
                                 bert_extract_features.ipynb also contains
                                 the Appendix E artifact analysis
                                 (length, TTR, perplexity under GPT-2 /
                                 GPT-2 Medium / GPT-Neo, real-vs-edited
                                 TF-IDF classifier, Gemini decomposition).

03_source_domain_diagnosis/
    bert_source_diagnosis.ipynb  Section 3 / Table 2: source-only baseline
                                 and source-label composition diagnosis.

04_text_baselines/
    bert_text_baseline.ipynb     Section 3.2: fine-tuned BERT and GPT-2
    gpt2_text_baseline.ipynb     (Table 5 main results).
    source_only_baseline.ipynb   Section 3 / Appendix F: per-seed source-
                                 only majority baseline on the test split
                                 (Table 21: mean 0.776 +/- 0.018).

05_bias_classifiers/
    bert_bias_classifier.ipynb   Section 3.2 (Attention-derived family):
    gpt2_bias_classifier.ipynb   MLP on the 3238-dim attention features,
                                 with linear residualization variants
                                 (Table 5, Appendix I INLP comparison).

06_edited_data_ablation/
    bert_edited_data_ablation.ipynb
                                 Section 5 / Table 6: variants A/B/C/D
                                 (real-only, +strengthened, all edited,
                                 no Gemini CFs) over 5 seeds, plus the
                                 paired statistical tests reported in
                                 Appendix F.

07_external_evaluations/
    bert_heldout_pairs_eval.ipynb  Section 6.2 / Appendix D: counterfactual
                                   discrimination on 983 held-out pairs
                                   (PairAcc 78.2%, DirAcc 97.3%).
    bert_crows_pairs_eval.ipynb    Appendix H: zero-shot CrowS-Pairs
                                   transfer (DirAcc 0.667 corrected, the
                                   BERT-base PLL independence test).

feature_extraction_code/         Python modules imported by the notebooks
                                 in 02_feature_extraction and
                                 05_bias_classifiers (ModelManager,
                                 attention metrics, ISA, TF-IDF baseline,
                                 calibration helpers).
                                 See feature_extraction_code/README.md.

prompts/                         Appendix B: exact Gemini prompts used
                                 in the audit and counterfactual pipeline.
    1-Prompt biased-neutral.txt        (Prompt 1: audit)
    2-Prompt neutral-biased.txt        (Prompt 2: repair / strengthen)
    3-Prompt Audit biased-neutral.txt  (Prompt 3: biased -> neutral CF)
    4-Prompt Audit neutral-biased.txt  (Prompt 4: neutral -> biased CF)
