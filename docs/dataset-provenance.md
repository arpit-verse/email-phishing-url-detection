# Dataset provenance and data requirements

## What the source code records

The email preprocessing module names the consolidated input as dataset/email/phishing_email.csv and expects columns text_combined and label. It maps label 0 to legitimate/benign and label 1 to phishing/malicious/fraud. Its module description states that the consolidated corpus contains 82,486 records from Enron, Ling-Spam, CEAS 2008, SpamAssassin, Nazario, and Nigerian Fraud, with per-source counts recorded in that module. These are source-code claims; this release does not independently validate the raw rows or source counts.

The URL feature-extraction script expects dataset/url/urlset_cleaned.csv and writes dataset/url/url_features.csv. The split script expects a domain identifier and label column, removes selected columns, and creates an 80/20 stratified split with random_state 42. The repository materials do not identify the original URL dataset publisher, version, collection date, or licensing terms.

## What is not established

The source project does not provide complete dataset citations, stable download links, source versions, raw-file hashes, field-level lineage, or redistribution permissions for all corpora. The email dataset also had unresolved credential-shaped material during prior private review. No row-level dataset is included here.

Do not treat dataset names in source comments as a complete citation or license. This release does not provide download links because exact source versions and permitted redistribution routes are not established in the repository evidence. Obtain data only from its original publisher or rights holder after confirming terms and documenting the exact version and hash. Do not use a mirror or redistribute a corpus based only on its name.

## Requirements for reproducing the existing preparation

- Email: provide an authorized consolidated CSV at dataset/email/phishing_email.csv with text_combined and label.
- URL feature extraction: provide an authorized cleaned CSV at dataset/url/urlset_cleaned.csv in the schema expected by training/extract_url_features.py.
- URL split/training: provide the derived dataset/url/url_features.csv expected by training/prepare_url_dataset.py; the deployed model training script then expects url_features_train.csv and url_features_test.csv.

All these paths are excluded by .gitignore in this public snapshot. Verify data permissions, labels, row provenance, and hashes in a separate approved environment before running scripts.