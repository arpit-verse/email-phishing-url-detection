# Reproducing results

This snapshot omits datasets and model files. Reproduction depends on obtaining the exact authorized inputs and recording their provenance. Use a separate disposable copy; do not train in a directory containing artifacts that must be preserved.

## Email pipeline

Required input: dataset/email/phishing_email.csv with text_combined and label columns, where 0 is legitimate and 1 is phishing. The training script prepares an 80/20 stratified split with random_state 42, runs 5-fold stratified CV on the training partition, evaluates both pipelines on the holdout, selects by holdout F1, and writes models and result CSVs.

In a fresh disposable checkout with the authorized input placed at that path:

    python training/train_email_model.py

The script writes result CSVs and four model files under models/. It checks for pre-existing model target files before serialization. Keep valuable outputs out of that checkout.

## URL pipeline

The deployed 25-feature workflow expects:
1. dataset/url/urlset_cleaned.csv for training/extract_url_features.py, which writes dataset/url/url_features.csv.
2. dataset/url/url_features.csv for training/prepare_url_dataset.py, which writes url_features_train.csv and url_features_test.csv using an 80/20 stratified split with random_state 42.
3. Those split files for training/train_url_random_forest.py, which fits a 100-tree Random Forest with random_state 42 and writes models/url_random_forest.pkl.

In a fresh disposable checkout with legally authorized files and verified schemas:

    python training/extract_url_features.py
    python training/prepare_url_dataset.py
    python training/train_url_random_forest.py

The URL training script writes to the model path and is not a safe overwrite workflow. Never run it where a model must be preserved. The exact original URL source/version is not identified here, so these commands alone do not reproduce the recorded deployed model.

## Tests and audits

- API tests inject fake model loaders and can run without serialized model files: python -m unittest api.test_api.
- Email predictor tests require the omitted email dataset and email model files.
- URL predictor tests require the omitted deployed URL model and URL test data.
- training/audit_baselines.py requires both datasets and models and writes audit files; use only in a disposable copy after reviewing output paths.
- training/audit_url_unseen_domain.py additionally expects a pinned PSL file and an isolated PSL environment, both omitted here.
- training/run_email_sensitivity_audit.py writes row-level counterfactual output and requires the email dataset/model; review output privacy separately.

The included CSVs are recorded evidence, not proof that results were reproduced from this source-only release.