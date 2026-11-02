# Model cards

## Email text classifier

- Task: assign a binary phishing/legitimate label to supplied email text.
- Inspected deployed pipeline: TfidfVectorizer followed by LogisticRegression; its class order is [0, 1].
- Training configuration: TF-IDF 1- and 2-grams, English stop-word removal, max_features 15,000, min_df 3, and sublinear term frequency. Logistic Regression uses C=1.0, max_iter=500, random_state=42, and class_weight=balanced.
- Training comparison: the code also evaluates Multinomial Naive Bayes with alpha=0.1.
- Preprocessing: HTML entities are decoded, markup stripped, URLs and addresses replaced by canonical tokens, punctuation removed, whitespace collapsed, and text lowercased. See training/preprocess_email.py.
- Output: predicted class and model probability columns for class 0 and class 1.
- Evidence: training/predict_email.py, training/preprocess_email.py, training/train_email_model.py, and aggregate CSV files under results/.
- Limitations: the holdout was used to choose the saved pipeline by F1. Dataset source licenses and full provenance are unresolved. Reported holdout values are not an independent final test.

## URL lexical classifier

- Task: classify a submitted URL string from lexical and structural features.
- Inspected deployed estimator: RandomForestClassifier with 25 input features and class order [0, 1]. The training script specifies 100 trees and random_state=42.
- Input: URL string; no page content or remote reputation is retrieved.
- Output: predicted class and probabilities from the estimator''s class columns.
- The 25 deployed features, in model order, are: url_length, hostname_length, path_length, query_length, digit_count, letter_count, dot_count, hyphen_count, underscore_count, slash_count, question_mark_count, equals_count, ampersand_count, percent_count, at_count, subdomain_count, path_depth, query_parameter_count, long_digit_sequence, suspicious_keyword_count, domain_digit_count, domain_hyphen_count, domain_dot_count, has_php, special_character_count.
- Raw-input features include total string length, character and punctuation counts, long-digit flag, suspicious-keyword count, .php indicator, and special-character count. Parsed URL features include hostname/path/query lengths, subdomain count, path depth, query-parameter count, and hostname digit/hyphen/dot counts. The keyword feature counts presence of fixed substrings, not semantic intent.
- Limitations: lexical counts can be brittle, URL text can be manipulated, and the row-wise holdout shares many domains with training. Recorded metrics do not establish domain-disjoint or future-source performance.
- Evidence: training/predict_url.py, training/extract_url_features.py, training/prepare_url_dataset.py, training/train_url_random_forest.py, and results/baseline_audit/.

Neither model is presented as a substitute for security review. Probability calibration has not been evaluated.