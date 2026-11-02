"""Run independent token-group counterfactual sensitivity checks for email.

This script uses the existing fitted email pipeline and the existing
prepare_email_dataset() holdout. It does not fit or modify any model component.
Raw email text is processed only in memory and is never written to outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "dataset" / "email" / "phishing_email.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "email_pipeline.pkl"
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline_audit"
OUTPUT_DIR = PROJECT_ROOT / "results" / "email_sensitivity_audit"
OUTPUT_NAMES = (
    "run_metadata.json",
    "group_summary.csv",
    "row_level_counterfactuals.csv",
)
TOLERANCE = 1e-12
SENTINEL = "cfmasktokenqzx"
LABEL_NAMES = {0: "LEGITIMATE", 1: "PHISHING"}

# Regex matches are whole normalized tokens. The date pattern deliberately
# masks month words even when they are ordinary prose, and all four-digit
# years from 1900 through 2099. This ambiguity is recorded in run metadata.
MASK_GROUPS = {
    "link_promotional": {
        "pattern": re.compile(
            r"(?<![a-z0-9])(?:click|free|link|view|receive|offer)(?![a-z0-9])",
            flags=re.IGNORECASE,
        ),
        "description": "Exact whole-token masking of click, free, link, view, receive, and offer.",
        "terms": ["click", "free", "link", "view", "receive", "offer"],
        "limitation": "These terms can have ordinary meanings; matching is lexical, not contextual.",
    },
    "date_like": {
        "pattern": re.compile(
            r"(?<![a-z0-9])(?:january|jan|february|feb|march|mar|april|apr|may|june|jun|"
            r"july|jul|august|aug|september|sept|sep|october|oct|november|nov|"
            r"december|dec|(?:19|20)[0-9]{2})(?![a-z0-9])",
            flags=re.IGNORECASE,
        ),
        "description": "Exact whole-token masking of month names/abbreviations and four-digit years 1900-2099.",
        "terms": [
            "January/Jan", "February/Feb", "March/Mar", "April/Apr", "May", "June/Jun",
            "July/Jul", "August/Aug", "September/Sept/Sep", "October/Oct", "November/Nov",
            "December/Dec", "1900-2099",
        ],
        "limitation": "Month names may be ordinary prose (especially 'may'); year-like numbers may not be dates. For example, both tokens in 'aug 2008' are masked.",
    },
    "mime_serialization": {
        "pattern": re.compile(
            r"(?<![a-z0-9])(?:multipart|nextpart|mime|contenttransferencoding|boundary)(?![a-z0-9])",
            flags=re.IGNORECASE,
        ),
        "description": "Exact whole-token masking of MIME/serialization markers.",
        "terms": ["multipart", "nextpart", "mime", "contenttransferencoding", "boundary"],
        "limitation": "The exact normalized markers are masked; other MIME syntax or wording is not. Masking 'mime' also removes the 'message mime' bigram while leaving 'message'.",
    },
}


def _check_output_paths():
    paths = [OUTPUT_DIR / name for name in OUTPUT_NAMES]
    existing = [str(path.relative_to(PROJECT_ROOT)) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing target(s): " + ", ".join(existing))


def _validate_sentinel(vectorizer):
    vocabulary = vectorizer.vocabulary_
    if SENTINEL in vocabulary:
        raise ValueError("Chosen masking sentinel is present in the fitted vocabulary.")
    feature_names = vectorizer.get_feature_names_out()
    if any(SENTINEL in str(feature).split() for feature in feature_names):
        raise ValueError("Chosen masking sentinel appears as a token in a fitted feature.")
    analyzer_tokens = vectorizer.build_analyzer()(SENTINEL)
    if analyzer_tokens != [SENTINEL] or analyzer_tokens[0] in vocabulary:
        raise ValueError("Chosen masking sentinel does not behave as an out-of-vocabulary token.")
    return {
        "token": SENTINEL,
        "absent_as_unigram": True,
        "absent_from_all_fitted_features": True,
        "analyzer_output": analyzer_tokens,
        "replacement_preserves_one_token_position_per_match": True,
    }


def _metrics(y_true, predictions, phishing_probability):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_phishing": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall_phishing": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1_phishing": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "roc_auc_phishing": float(roc_auc_score(y_true, phishing_probability)),
    }


def _probability_summary(prefix, before, after):
    delta = after - before
    return {
        f"{prefix}_baseline_mean_phishing_probability": float(np.mean(before)),
        f"{prefix}_counterfactual_mean_phishing_probability": float(np.mean(after)),
        f"{prefix}_mean_probability_change": float(np.mean(delta)),
        f"{prefix}_median_probability_change": float(np.median(delta)),
        f"{prefix}_p05_probability_change": float(np.quantile(delta, 0.05)),
        f"{prefix}_p95_probability_change": float(np.quantile(delta, 0.95)),
    }


def _summary_row(group_name, y_true, baseline_pred, counterfactual_pred, baseline_prob, counterfactual_prob):
    legitimate = y_true == 0
    phishing = y_true == 1
    baseline_false_positive = legitimate & (baseline_pred == 1)
    counterfactual_false_positive = legitimate & (counterfactual_pred == 1)
    baseline_true_negative = legitimate & (baseline_pred == 0)
    baseline_false_negative = phishing & (baseline_pred == 0)
    baseline_true_positive = phishing & (baseline_pred == 1)

    row = {
        "mask_group": group_name,
        "holdout_rows": int(len(y_true)),
        "legitimate_rows": int(legitimate.sum()),
        "phishing_rows": int(phishing.sum()),
        **{f"baseline_{key}": value for key, value in _metrics(y_true, baseline_pred, baseline_prob).items()},
        **{f"counterfactual_{key}": value for key, value in _metrics(y_true, counterfactual_pred, counterfactual_prob).items()},
        "baseline_false_positives": int(baseline_false_positive.sum()),
        "counterfactual_false_positives": int(counterfactual_false_positive.sum()),
        "baseline_false_negatives": int(baseline_false_negative.sum()),
        "counterfactual_false_negatives": int((phishing & (counterfactual_pred == 0)).sum()),
        "previous_false_positives_now_legitimate": int((baseline_false_positive & (counterfactual_pred == 0)).sum()),
        "previously_correct_legitimate_now_false_positive": int((baseline_true_negative & (counterfactual_pred == 1)).sum()),
        "previously_correct_phishing_now_legitimate": int((baseline_true_positive & (counterfactual_pred == 0)).sum()),
        "previously_missed_phishing_now_detected": int((baseline_false_negative & (counterfactual_pred == 1)).sum()),
    }
    row.update(_probability_summary("all", baseline_prob, counterfactual_prob))
    row.update(_probability_summary("legitimate", baseline_prob[legitimate], counterfactual_prob[legitimate]))
    row.update(_probability_summary("phishing", baseline_prob[phishing], counterfactual_prob[phishing]))
    return row


def _verify_baseline(model, vectorizer, classifier, normalized_texts):
    classes = [int(value) for value in model.classes_]
    classifier_classes = [int(value) for value in classifier.classes_]
    if classes != [0, 1] or classifier_classes != classes:
        raise ValueError("Email model classes_ do not confirm 0=legitimate and 1=phishing.")

    pipeline_predictions = np.asarray(model.predict(normalized_texts), dtype=int)
    pipeline_probabilities = np.asarray(model.predict_proba(normalized_texts), dtype=float)
    transformed = vectorizer.transform(normalized_texts)
    direct_predictions = np.asarray(classifier.predict(transformed), dtype=int)
    direct_probabilities = np.asarray(classifier.predict_proba(transformed), dtype=float)

    if not np.array_equal(pipeline_predictions, direct_predictions):
        raise ValueError("Direct fitted steps predictions differ from the saved pipeline.")
    if not np.allclose(direct_probabilities, pipeline_probabilities, rtol=TOLERANCE, atol=TOLERANCE):
        raise ValueError("Direct fitted steps probabilities differ from the saved pipeline.")

    maximum_difference = float(np.max(np.abs(direct_probabilities - pipeline_probabilities)))
    return {
        "passed": True,
        "holdout_rows_checked": int(len(normalized_texts)),
        "prediction_mismatches": 0,
        "probability_columns_checked": ["legitimate", "phishing"],
        "maximum_absolute_probability_difference": maximum_difference,
        "comparison": "model.predict/model.predict_proba versus clf.predict/clf.predict_proba(tfidf.transform(texts))",
        "numpy_allclose_rtol": TOLERANCE,
        "numpy_allclose_atol": TOLERANCE,
    }, pipeline_predictions, direct_probabilities[:, 1]


def _check_existing_error_rows(errors_path, row_refs, y_true, baseline_pred, baseline_probability):
    saved = pd.read_csv(errors_path)
    false_positives = saved.loc[saved["error_type"] == "false_positive"].copy()
    fp_positions = np.flatnonzero((y_true == 0) & (baseline_pred == 1))
    current_refs = set(int(row_refs[position]) for position in fp_positions)
    saved_refs = set(false_positives["source_row_index"].astype(int))
    if current_refs != saved_refs:
        raise ValueError("Current baseline false-positive references differ from email_errors.csv.")
    by_ref = {int(row_refs[position]): position for position in fp_positions}
    for _, record in false_positives.iterrows():
        position = by_ref[int(record["source_row_index"])]
        if int(record["true_label"]) != 0 or int(record["predicted_label"]) != 1:
            raise ValueError("Saved false-positive row labels are inconsistent.")
        if not np.isclose(
            float(record["phishing_probability"]),
            float(baseline_probability[position]),
            rtol=TOLERANCE,
            atol=TOLERANCE,
        ):
            raise ValueError("Saved false-positive probability differs from the current fitted model.")
    return {"saved_false_positive_rows_verified": int(len(false_positives)), "references_and_probabilities_match": True}


def run():
    # Refuse collisions before loading or evaluating any data.
    _check_output_paths()
    if not DATASET_PATH.is_file() or not MODEL_PATH.is_file():
        raise FileNotFoundError("Required email dataset or model artifact is missing.")
    if not (BASELINE_DIR / "email_errors.csv").is_file():
        raise FileNotFoundError("Existing baseline email error artifact is missing.")

    from training.preprocess_email import prepare_email_dataset

    X_train, X_test, y_train, y_test, split_stats = prepare_email_dataset(DATASET_PATH)
    del X_train, y_train
    model = joblib.load(MODEL_PATH)
    steps = getattr(model, "named_steps", {})
    if "tfidf" not in steps or "clf" not in steps:
        raise ValueError("Fitted model does not contain the verified tfidf and clf pipeline steps.")
    vectorizer = steps["tfidf"]
    classifier = steps["clf"]

    baseline_check, baseline_predictions, baseline_phishing_probability = _verify_baseline(
        model, vectorizer, classifier, X_test.tolist()
    )
    baseline_probabilities = np.asarray(model.predict_proba(X_test.tolist()), dtype=float)
    baseline_legitimate_probability = baseline_probabilities[:, list(model.classes_).index(0)]
    row_refs = X_test.index.to_numpy(dtype=int)
    y_true = y_test.to_numpy(dtype=int)
    error_check = _check_existing_error_rows(
        BASELINE_DIR / "email_errors.csv",
        row_refs,
        y_true,
        baseline_predictions,
        baseline_phishing_probability,
    )
    sentinel_check = _validate_sentinel(vectorizer)

    row_frames = []
    summary_rows = []
    masked_counts = {}
    for group_name, specification in MASK_GROUPS.items():
        pattern = specification["pattern"]
        masked_counts[group_name] = 0
        counterfactual_texts = []
        row_mask_counts = []
        for text in X_test.tolist():
            matches = list(pattern.finditer(text))
            masked_counts[group_name] += len(matches)
            row_mask_counts.append(len(matches))
            counterfactual_texts.append(pattern.sub(SENTINEL, text))

        transformed_matrix = vectorizer.transform(counterfactual_texts)
        counterfactual_predictions = np.asarray(classifier.predict(transformed_matrix), dtype=int)
        transformed_probabilities = np.asarray(classifier.predict_proba(transformed_matrix), dtype=float)
        counterfactual_phishing_probability = transformed_probabilities[:, list(model.classes_).index(1)]

        summary_rows.append(
            _summary_row(
                group_name,
                y_true,
                baseline_predictions,
                counterfactual_predictions,
                baseline_phishing_probability,
                counterfactual_phishing_probability,
            )
        )

        delta = counterfactual_phishing_probability - baseline_phishing_probability
        transitions = np.full(len(y_true), "unchanged", dtype=object)
        transitions[(baseline_predictions == 1) & (counterfactual_predictions == 0)] = "phishing_to_legitimate"
        transitions[(baseline_predictions == 0) & (counterfactual_predictions == 1)] = "legitimate_to_phishing"
        row_frames.append(
            pd.DataFrame(
                {
                    "mask_group": group_name,
                    "source_row_index": row_refs,
                    "true_label": y_true,
                    "true_label_name": [LABEL_NAMES[int(value)] for value in y_true],
                    "baseline_prediction": baseline_predictions,
                    "baseline_label": [LABEL_NAMES[int(value)] for value in baseline_predictions],
                    "counterfactual_prediction": counterfactual_predictions,
                    "counterfactual_label": [LABEL_NAMES[int(value)] for value in counterfactual_predictions],
                    "baseline_phishing_probability": baseline_phishing_probability,
                    "counterfactual_phishing_probability": counterfactual_phishing_probability,
                    "phishing_probability_change": delta,
                    "prediction_transition": transitions,
                    "masked_token_count": np.asarray(row_mask_counts, dtype=int),
                }
            )
        )

    group_summary = pd.DataFrame(summary_rows)
    row_level = pd.concat(row_frames, ignore_index=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "counterfactual sensitivity analysis; not retraining, causal attribution, or evidence of production improvement",
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "holdout": {
            "preparation": "training/preprocess_email.py::prepare_email_dataset(test_size=0.20, random_state=42, stratify=label)",
            "rows": int(len(X_test)),
            "source_row_reference": "zero-based pandas row index in phishing_email.csv",
            "row_order_preserved": True,
            "label_mapping": {"0": "LEGITIMATE", "1": "PHISHING"},
            "split_summary": split_stats,
        },
        "model_pipeline": {
            "steps": [[name, type(step).__name__] for name, step in model.steps],
            "classes_in_probability_order": [int(value) for value in model.classes_],
            "probability_mapping": {"0": "legitimate_probability", "1": "phishing_probability"},
            "components_refit": False,
            "baseline_direct_step_verification": baseline_check,
        },
        "saved_baseline_error_mapping_check": error_check,
        "sentinel_verification": sentinel_check,
        "masking": {
            "normalization_stage": "Applied to already-normalized holdout text returned by prepare_email_dataset().",
            "replacement": f"Each matched token is replaced with the verified out-of-vocabulary token '{SENTINEL}', preserving one token position per match.",
            "groups_evaluated_independently": True,
            "matched_token_counts_by_group": masked_counts,
            "groups": {
                name: {
                    "description": item["description"],
                    "terms_or_pattern": item["terms"],
                    "limitation": item["limitation"],
                    "regex": item["pattern"].pattern,
                }
                for name, item in MASK_GROUPS.items()
            },
        },
        "evaluation": {
            "baseline": "Predictions and probabilities from the existing fitted pipeline on the unmodified normalized holdout.",
            "counterfactual": "Each group is masked separately, then passed through the existing fitted tfidf and clf steps.",
            "interpretation": "Text masking can alter unigram and neighboring n-gram features. Results measure sensitivity to these transformations only; they do not establish causality, explain individual emails, or demonstrate valid production improvement.",
            "date_limitation": "Month names can occur in ordinary prose; all matching month tokens and four-digit years are masked without semantic date parsing.",
        },
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        },
        "outputs": list(OUTPUT_NAMES),
        "raw_email_saved_or_printed": False,
    }

    # Recheck all output paths immediately before writing. No existing artifact
    # is replaced, and only the three declared result files are written.
    existing_outputs = [name for name in OUTPUT_NAMES if (OUTPUT_DIR / name).exists()]
    if existing_outputs:
        raise FileExistsError("Refusing to overwrite existing output(s): " + ", ".join(existing_outputs))

    group_csv = group_summary.to_csv(index=False)
    row_csv = row_level.to_csv(index=False)
    metadata_json = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "group_summary.csv").write_text(group_csv, encoding="utf-8")
    (OUTPUT_DIR / "row_level_counterfactuals.csv").write_text(row_csv, encoding="utf-8")
    (OUTPUT_DIR / "run_metadata.json").write_text(metadata_json, encoding="utf-8")

    print(f"Counterfactual sensitivity outputs written to {OUTPUT_DIR}")
    print(group_summary.to_json(orient="records"))


if __name__ == "__main__":
    run()
