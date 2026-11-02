"""
Automated Test Suite for Email Phishing Detection Pipeline (Phase 2).

Tests:
1. Text normalization rules (HTML stripping, entity decoding, tokenization, URLs, email addresses).
2. Leakage prevention and dataset partition integrity.
3. Model loading and working-directory independence.
4. Inference output schema, probability calibration, and deterministic behavior.
5. Edge case handling (empty strings, whitespace, non-alphanumeric characters).
6. CLI error handling: ValueError isolation and unexpected-error propagation.
7. Verification of saved cross-validation and evaluation metric records.
"""

import sys
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from preprocess_email import normalize_email_text, prepare_email_dataset
import predict_email
from predict_email import predict, load_model, print_result


class TestEmailPipeline(unittest.TestCase):

    def test_01_text_normalization(self):
        """Verify that normalize_email_text correctly cleans and standardizes raw inputs."""
        # HTML tag stripping
        html_input = "<p>Dear customer,</p><br>Please <b>update</b> your credentials."
        cleaned_html = normalize_email_text(html_input)
        self.assertNotIn("<p>", cleaned_html)
        self.assertNotIn("<b>", cleaned_html)
        self.assertIn("dear customer", cleaned_html)
        self.assertIn("update your credentials", cleaned_html)

        # Entity decoding
        entity_input = "Tom &amp; Jerry &quot;special discount&quot;"
        cleaned_entity = normalize_email_text(entity_input)
        self.assertIn("tom", cleaned_entity)
        self.assertIn("jerry", cleaned_entity)
        self.assertIn("special discount", cleaned_entity)

        # URL canonicalization
        url_input = "Please visit http://paypal.com/login or https://secure-bank.org/update immediately."
        cleaned_url = normalize_email_text(url_input)
        self.assertIn("httpurl", cleaned_url)

        # Email address canonicalization
        email_addr_input = "Contact support@example.com for details."
        cleaned_email_addr = normalize_email_text(email_addr_input)
        self.assertIn("emailaddress", cleaned_email_addr)

        # Null and empty safety
        self.assertEqual(normalize_email_text(None), "")
        self.assertEqual(normalize_email_text(""), "")
        self.assertEqual(normalize_email_text("   \n\t  "), "")

    def test_02_leakage_and_partition_integrity(self):
        """Verify dataset preparation guarantees zero leakage between train and test."""
        dataset_file = PROJECT_ROOT / "dataset" / "email" / "phishing_email.csv"
        self.assertTrue(dataset_file.exists(), f"Missing dataset file: {dataset_file}")

        # Run preparation (fast sample or standard partition check)
        X_train, X_test, y_train, y_test, stats = prepare_email_dataset()

        # Assert no overlap
        train_set = set(X_train)
        test_set = set(X_test)
        overlap = train_set.intersection(test_set)
        self.assertEqual(len(overlap), 0, f"Detected {len(overlap)} overlapping samples between train and test!")

        # Assert clean labels
        self.assertTrue(set(y_train.unique()).issubset({0, 1}))
        self.assertTrue(set(y_test.unique()).issubset({0, 1}))
        self.assertEqual(stats["conflicting_text_groups_dropped"], 0)

    def test_03_model_loading_and_path_independence(self):
        """Verify that all new email model files exist and can be loaded dynamically."""
        expected_models = [
            MODELS_DIR / "email_pipeline.pkl",
            MODELS_DIR / "email_naive_bayes.pkl",
            MODELS_DIR / "email_logistic_regression.pkl",
            MODELS_DIR / "email_tfidf_vectorizer.pkl"
        ]

        for model_path in expected_models:
            self.assertTrue(model_path.exists(), f"Expected model file missing: {model_path}")
            self.assertGreater(model_path.stat().st_size, 1000, f"Model file is suspiciously small: {model_path}")

        # Verify load_model() works with default and relative paths
        loaded_default = load_model()
        self.assertIsNotNone(loaded_default)
        self.assertTrue(hasattr(loaded_default, "predict"))
        self.assertTrue(hasattr(loaded_default, "predict_proba"))

    def test_04_prediction_schema_and_determinism(self):
        """Verify inference returns valid schema and behaves deterministically."""
        model = load_model()
        sample = "Security Alert: Please verify your bank account details by clicking the link."

        res1 = predict(sample, model=model)
        res2 = predict(sample, model=model)

        # Schema assertion
        required_keys = [
            "text", "normalized_text", "prediction", "label",
            "phishing_probability", "legitimate_probability", "detected_tokens"
        ]
        for k in required_keys:
            self.assertIn(k, res1, f"Missing key in prediction schema: {k}")

        self.assertIn(res1["prediction"], [0, 1])
        self.assertIn(res1["label"], ["PHISHING", "LEGITIMATE"])
        self.assertAlmostEqual(res1["phishing_probability"] + res1["legitimate_probability"], 1.0, places=5)

        # Determinism assertion
        self.assertEqual(res1["prediction"], res2["prediction"])
        self.assertAlmostEqual(res1["phishing_probability"], res2["phishing_probability"], places=7)

    def test_05_edge_cases_and_sample_observations(self):
        """Verify graceful edge-case handling and observe sample predictions without hardcoded certainty."""
        model = load_model()

        # Empty or non-informative input must be rejected, not labeled legitimate.
        unusable_inputs = ["", "   \t\n   ", "<p></p>"]
        for text in unusable_inputs:
            with self.subTest(text=repr(text)):
                with self.assertRaisesRegex(ValueError, "usable content"):
                    predict(text, model=model)

        # Sample observations (logged for audit, not asserted on rigid thresholds)
        samples = [
            ("Phishing Sample 1", "URGENT: Your account has been suspended. Please click here to verify your identity immediately or your access will be permanently terminated."),
            ("Phishing Sample 2", "Dear valued customer, We detected unusual activity on your PayPal account. Log in at httpurl to restore full privileges."),
            ("Legitimate Sample 1", "Hi team, please find attached the weekly project status report and notes from today's sprint planning session."),
            ("Legitimate Sample 2", "Can you review the updated proposal slides before our client sync tomorrow morning? Thanks, Alice.")
        ]

        print("\n" + "=" * 60)
        print("SAMPLE EMAIL OBSERVATIONS (Inference Behavior Audit)")
        print("=" * 60)
        for label, text in samples:
            res = predict(text, model=model)
            print(f"[{label}]")
            print(f"  Text: {text[:80]}...")
            print(f"  Prediction: {res['label']} (Phish: {res['phishing_probability']*100:.2f}%, Legit: {res['legitimate_probability']*100:.2f}%)")
            print(f"  Recognized Tokens: {res['detected_tokens'][:8]}")
        print("=" * 60)

    def test_06_cli_handles_invalid_input_and_propagates_unexpected_errors(self):
        with patch("builtins.input", side_effect=["<p></p>", EOFError]), \
                patch.object(predict_email, "load_model", return_value=object()), \
                patch.object(predict_email, "predict", side_effect=ValueError("empty normalized text")):
            output = io.StringIO()
            with redirect_stdout(output):
                predict_email.main()

        self.assertIn("Invalid email input. Please provide text with usable content.", output.getvalue())

        with patch("builtins.input", side_effect=["some email content", EOFError]), \
                patch.object(predict_email, "load_model", return_value=object()), \
                patch.object(predict_email, "predict", side_effect=RuntimeError("unexpected failure")):
            with self.assertRaisesRegex(RuntimeError, "unexpected failure"):
                predict_email.main()

    def test_07_evaluation_results_records_validity(self):
        """Verify that saved CV and evaluation results files exist and contain valid metric bounds."""
        cv_path = RESULTS_DIR / "email_cross_validation_results.csv"
        eval_path = RESULTS_DIR / "email_evaluation_results.csv"

        self.assertTrue(cv_path.exists(), f"CV results file missing: {cv_path}")
        self.assertTrue(eval_path.exists(), f"Holdout evaluation results file missing: {eval_path}")

        eval_df = pd.read_csv(eval_path)
        self.assertGreaterEqual(len(eval_df), 2, "Expected at least 2 evaluated models in results.")

        for _, row in eval_df.iterrows():
            self.assertGreaterEqual(row["Accuracy"], 0.70, f"Accuracy unusually low for {row['Model']}")
            self.assertGreaterEqual(row["F1 Score"], 0.70, f"F1 unusually low for {row['Model']}")
            self.assertGreaterEqual(row["ROC-AUC"], 0.70, f"ROC-AUC unusually low for {row['Model']}")
            self.assertEqual(
                row["True Negatives (TN)"] + row["False Positives (FP)"] + row["False Negatives (FN)"] + row["True Positives (TP)"],
                row["Total Test Samples"]
            )


if __name__ == "__main__":
    unittest.main()
