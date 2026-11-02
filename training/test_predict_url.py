"""
Non-destructive test suite for URL prediction stabilization (Phase 1).
Validates:
1. Exact feature extraction symmetry between extract_url_features.py and predict_url.py.
2. Clean importability without stdin blocking.
3. Model loading independence from working directory.
4. Safe console rendering without Unicode encoding crashes.
5. Exact reproduction of test-set baseline metrics on dataset/url/url_features_test.csv.
6. Regression behavior across diverse URL categories.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Ensure the project root and training directory are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# Import the inference functions
import predict_url
from predict_url import extract_url_features, predict, load_model, print_result, FEATURES


# Extract reference logic directly from training/extract_url_features.py
import re
from urllib.parse import urlparse

def reference_training_extract(url):
    """Reference implementation matching lines 25-261 of training/extract_url_features.py."""
    url = str(url).strip()
    parse_url = url
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", parse_url):
        parse_url = "http://" + parse_url

    try:
        parsed = urlparse(parse_url)
    except Exception:
        parsed = urlparse("http://")

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    url_length = len(url)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    digit_count = sum(c.isdigit() for c in url)
    letter_count = sum(c.isalpha() for c in url)
    dot_count = url.count(".")
    hyphen_count = url.count("-")
    underscore_count = url.count("_")
    slash_count = url.count("/")
    question_mark_count = url.count("?")
    equals_count = url.count("=")
    ampersand_count = url.count("&")
    percent_count = url.count("%")
    at_count = url.count("@")

    if hostname:
        subdomain_parts = hostname.split(".")
        subdomain_count = max(len(subdomain_parts) - 2, 0)
    else:
        subdomain_count = 0

    path_depth = len([part for part in path.split("/") if part])
    query_parameter_count = len(query.split("&")) if query else 0
    long_digit_sequence = int(bool(re.search(r"\d{5,}", url)))

    suspicious_words = [
        "login", "signin", "verify", "verification", "secure",
        "account", "update", "confirm", "password", "credential",
        "bank", "paypal", "payment", "wallet", "billing",
        "invoice", "authentication"
    ]
    url_lower = url.lower()
    suspicious_keyword_count = sum(w in url_lower for w in suspicious_words)

    domain_digit_count = sum(c.isdigit() for c in hostname)
    domain_hyphen_count = hostname.count("-")
    domain_dot_count = hostname.count(".")
    has_php = int(".php" in url_lower)
    special_character_count = sum(not c.isalnum() for c in url)

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "query_length": query_length,
        "digit_count": digit_count,
        "letter_count": letter_count,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "underscore_count": underscore_count,
        "slash_count": slash_count,
        "question_mark_count": question_mark_count,
        "equals_count": equals_count,
        "ampersand_count": ampersand_count,
        "percent_count": percent_count,
        "at_count": at_count,
        "subdomain_count": subdomain_count,
        "path_depth": path_depth,
        "query_parameter_count": query_parameter_count,
        "long_digit_sequence": long_digit_sequence,
        "suspicious_keyword_count": suspicious_keyword_count,
        "domain_digit_count": domain_digit_count,
        "domain_hyphen_count": domain_hyphen_count,
        "domain_dot_count": domain_dot_count,
        "has_php": has_php,
        "special_character_count": special_character_count
    }


class TestURLInferenceStabilization(unittest.TestCase):

    def test_01_feature_symmetry_with_training_pipeline(self):
        """Verify that training and inference extractors produce 100% identical values for all 25 features."""
        diverse_test_urls = [
            "google.com",
            "www.google.com",
            "https://www.google.com/search?q=machine+learning&source=hp",
            "http://192.168.1.1/admin/login.php?session=9876543210",
            "https://github.com/torvalds/linux/commit/1234567890abcdef",
            "http://paypal-verification-account-update.xyz/login.php?token=abc&user=test",
            "https://en.wikipedia.org/wiki/Phishing",
            "sub.sub2.domain-name.co.uk/deep/nested/path/to/resource.html",
            "http://user:placeholder@example.com:8080/path?arg1=val1&arg2=val2#anchor",
            "www.sec.gov/edgar.shtml",
            "http://secure-billing-bank-authentication.com/invoice.php",
            "https://legit-site.org/downloads/file_version_1.0.tar.gz"
        ]

        for url in diverse_test_urls:
            ref_features = reference_training_extract(url)
            inf_features = extract_url_features(url)

            self.assertEqual(
                len(inf_features), 25,
                f"Extracted feature count ({len(inf_features)}) must equal 25."
            )

            for feat in FEATURES:
                self.assertIn(feat, inf_features, f"Missing feature {feat} in inference output.")
                self.assertEqual(
                    ref_features[feat], inf_features[feat],
                    f"Feature mismatch on URL '{url}' for feature '{feat}': "
                    f"Training ref={ref_features[feat]} vs Inference={inf_features[feat]}"
                )

    def test_02_feature_names_and_order_match_model(self):
        """Verify that FEATURES list matches feature_names_in_ on models/url_random_forest.pkl."""
        model = load_model()
        model_features = list(model.feature_names_in_)
        self.assertEqual(
            FEATURES, model_features,
            "Inference FEATURES order does not match model.feature_names_in_ exactly."
        )

    def test_03_model_path_resolution(self):
        """Verify model loading works dynamically regardless of working directory."""
        model = load_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.n_features_in_, 25)

    def test_04_import_and_prediction_safety(self):
        """Verify predict() returns valid schema without stdin interaction."""
        res = predict("https://github.com/torvalds/linux")
        self.assertIsInstance(res, dict)
        self.assertIn("prediction", res)
        self.assertIn("phishing_probability", res)
        self.assertIn("legitimate_probability", res)
        self.assertIn("features", res)
        self.assertAlmostEqual(res["phishing_probability"] + res["legitimate_probability"], 1.0, places=5)

    def test_05_console_display_no_unicode_crash(self):
        """Verify print_result() does not raise UnicodeEncodeError on standard streams."""
        res_phish = {"url": "http://test.com", "prediction": 1, "phishing_probability": 0.95, "legitimate_probability": 0.05, "features": extract_url_features("http://test.com")}
        res_legit = {"url": "http://test.com", "prediction": 0, "phishing_probability": 0.05, "legitimate_probability": 0.95, "features": extract_url_features("http://test.com")}

        # Test printing both outcomes
        try:
            print_result(res_phish)
            print_result(res_legit)
        except UnicodeEncodeError as e:
            self.fail(f"print_result raised UnicodeEncodeError: {e}")

    def test_06_test_set_baseline_verification(self):
        """Verify that models/url_random_forest.pkl reproduces exact baseline metrics on url_features_test.csv."""
        test_csv_path = PROJECT_ROOT / "dataset" / "url" / "url_features_test.csv"
        self.assertTrue(test_csv_path.exists(), "url_features_test.csv must exist.")

        df = pd.read_csv(test_csv_path)
        X = df[FEATURES]
        y_true = df["label"]

        model = load_model()
        y_pred = model.predict(X)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print("\n" + "=" * 50)
        print("TEST SET RE-VERIFICATION (url_features_test.csv)")
        print("=" * 50)
        print(f"Total Rows: {len(df)}")
        print(f"True Negatives : {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives : {tp}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print("=" * 50)

        # Baseline asserts
        self.assertEqual(len(df), 19183)
        self.assertEqual(tn, 9128)
        self.assertEqual(fp, 474)
        self.assertEqual(fn, 797)
        self.assertEqual(tp, 8784)
        self.assertAlmostEqual(acc, 0.933743, places=5)
        self.assertAlmostEqual(prec, 0.948801, places=5)
        self.assertAlmostEqual(rec, 0.916815, places=5)
        self.assertAlmostEqual(f1, 0.932533, places=5)


if __name__ == "__main__":
    unittest.main()
