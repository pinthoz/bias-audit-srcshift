Feature extraction code used by `bert_source.ipynb` / `gpt2_source.ipynb`
to build the feature matrix pickle (per-sentence attention features).

Layout
------
attention_app/
    models.py                        (ModelManager: tokenizer + BERT/GPT-2 loader)
    metrics.py                       (GAM, flow change)
    head_specialization.py           (head metrics, linguistic tags)
    isa.py                           (Information Sub-Attention)
    bias/
        feature_extraction.py        (low-level: forward pass -> attention weights)
        feature_extraction_notebooks.py
                                     (entry point: extract_features_for_sentence)
        scientific_utils.py          (TF-IDF lexical baseline, bootstrap CIs,
                                      calibration helpers used by the
                                      bias_classifier notebooks).

Usage (from the project root that contains the `attention_app` folder):

    from attention_app.bias.feature_extraction_notebooks import extract_features_for_sentence

This is the function the `bert_source` / `gpt2_source` notebooks call in a
loop over the dataset to produce the `feature_matrix_*.pkl` file.

Dependencies: numpy, torch, transformers.
