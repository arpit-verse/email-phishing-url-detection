# Public release exclusions

This manifest describes the staging snapshot only. The original project and private backup were not modified. The snapshot source is local commit c860775719f1b3c98ecd9f865ab8a18b3a47182b; the verified private backup snapshot used for cross-checking is dd2fb5c5c3cf7a7e7ee56f5bee19677ed544c307. No .git directory, branch history, or old Git LFS references were copied.

## Entire dataset directory

All raw, derived, and exploratory files under dataset/ are excluded. This avoids raw email or URL records and does not imply that derived files have redistribution permission.
- dataset/email/CEAS_08.csv
- dataset/email/Enron.csv
- dataset/email/Ling.csv
- dataset/email/Nazario.csv
- dataset/email/Nigerian_Fraud.csv
- dataset/email/phishing_email.csv
- dataset/email/SpamAssasin.csv
- dataset/url/combined_features_test.csv
- dataset/url/combined_features_train.csv
- dataset/url/decision_tree_feature_importance.csv
- dataset/url/feature_class_comparison.csv
- dataset/url/feature_correlation.csv
- dataset/url/feature_quality.csv
- dataset/url/random_forest_feature_importance.csv
- dataset/url/test.csv
- dataset/url/train.csv
- dataset/url/url_feature_class_comparison.csv
- dataset/url/url_feature_correlation.csv
- dataset/url/url_feature_feature_correlation.csv
- dataset/url/url_features_test.csv
- dataset/url/url_features_train.csv
- dataset/url/url_features.csv
- dataset/url/urlset_cleaned.csv
- dataset/url/urlset.csv
## Model binaries

All files under models/ are excluded. The API obtains only the two pinned deployment artifacts in docs/model-artifacts.md at runtime. Other experimental model binaries are not part of the deployment manifest.
- models/combined_decision_tree.pkl
- models/combined_random_forest.pkl
- models/combined_voting_ensemble.pkl
- models/decision_tree.pkl
- models/email_logistic_regression.pkl
- models/email_naive_bayes.pkl
- models/email_pipeline.pkl
- models/email_tfidf_vectorizer.pkl
- models/random_forest.pkl
- models/url_random_forest.pkl
- models/voting_ensemble.pkl
## Row-level, private, or unapproved outputs

- results/baseline_audit/email_errors.csv — row-level error references and probabilities.
- results/baseline_audit/url_errors.csv — row-level error references and probabilities.
- results/baseline_audit/run_metadata.json — run metadata excluded from the approved aggregate set.
- results/email_sensitivity_audit/row_level_counterfactuals.csv — row-level counterfactual records.
- results/email_sensitivity_audit/run_metadata.json — run metadata excluded from the approved aggregate set.
- results/email_top_predictive_features.csv — derived lexical terms were not cleared for public release because some resemble sender, organization, or person identifiers.
- The unusual root-level untracked file — excluded without opening or copying it.
- requirements-windows.txt — machine-specific package inventory, not the project runtime manifest.
- .gitattributes — old Git LFS rules refer to excluded dataset/model paths.
- BACKUP_MANIFEST.md and BACKUP_UNUSUAL_ROOT_FILE.txt — private-backup operational files, not application documentation.

## Generated or sensitive paths

- .git/ and historical objects/refs — clean snapshot only.
- .venv/, .venv-api/, .venv-psl-audit/ — local Python environments.
- frontend/node_modules/ and frontend/dist/ — regenerable dependencies/build output.
- .env files, credentials, tokens, private keys, caches, temporary downloads, and temporary files — excluded by policy. No secret-bearing file was found in the candidate file inventory. Credential-shaped strings retained in tests are synthetic placeholders used to test input validation; details are summarized in the audit report.