"""Reproduce holdout metrics for the currently deployed email and URL models.

This evaluation-only script does not train models or modify source datasets. It
refuses to overwrite any of the expected outputs in results/baseline_audit/.
Email metrics reproduce the existing holdout, which has previously been used
for model selection and is not an independent final evaluation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline_audit"
OUTPUT_NAMES = (
    "run_metadata.json",
    "email_metrics.csv",
    "email_confusion_matrix.csv",
    "email_errors.csv",
    "url_metrics.csv",
    "url_confusion_matrix.csv",
    "url_errors.csv",
)

EMAIL_DATASET = PROJECT_ROOT / "dataset" / "email" / "phishing_email.csv"
EMAIL_MODEL = PROJECT_ROOT / "models" / "email_pipeline.pkl"
URL_SOURCE = PROJECT_ROOT / "dataset" / "url" / "url_features.csv"
URL_SAVED_TEST = PROJECT_ROOT / "dataset" / "url" / "url_features_test.csv"
URL_MODEL = PROJECT_ROOT / "models" / "url_random_forest.pkl"

EMAIL_LABELS = {0: "LEGITIMATE", 1: "PHISHING"}
URL_REMOVE_FEATURES = [
    "has_https",
    "has_ip_address",
    "has_at_symbol",
    "has_exe",
    "has_html",
]


def _class_positions(model, model_name):
    """Return positions for labels 0/1, refusing ambiguous class metadata."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError(f"{model_name} has no classes_ metadata; probability mapping is unverified.")
    normalized = []
    for value in classes:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{model_name} has unsupported class metadata.") from exc
        if numeric not in (0.0, 1.0):
            raise ValueError(f"{model_name} classes must represent 0=legitimate and 1=phishing.")
        normalized.append(int(numeric))
    if len(normalized) != 2 or set(normalized) != {0, 1}:
        raise ValueError(f"{model_name} classes must contain exactly 0 and 1.")
    return {label: normalized.index(label) for label in (0, 1)}, [int_label for int_label in normalized]


def _metric_row(y_true, y_pred, phishing_probability):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    try:
        auc = float(roc_auc_score(y_true, phishing_probability))
    except ValueError:
        auc = None
    counts = pd.Series(y_true).value_counts().to_dict()
    return {
        "n_samples": int(len(y_true)),
        "legitimate_count": int(counts.get(0, 0)),
        "phishing_count": int(counts.get(1, 0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_phishing": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_phishing": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_phishing": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc_phishing": auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }, matrix


def _confusion_frame(matrix):
    return pd.DataFrame(
        matrix,
        index=["true_legitimate", "true_phishing"],
        columns=["predicted_legitimate", "predicted_phishing"],
    ).rename_axis("true_label")


def _error_frame(row_refs, y_true, y_pred, legitimate_probability, phishing_probability):
    records = []
    for row_ref, actual, predicted, p_legit, p_phish in zip(
        row_refs, y_true, y_pred, legitimate_probability, phishing_probability
    ):
        actual = int(actual)
        predicted = int(predicted)
        if actual == predicted:
            continue
        records.append(
            {
                "source_row_index": int(row_ref),
                "true_label": actual,
                "true_label_name": EMAIL_LABELS[actual],
                "predicted_label": predicted,
                "predicted_label_name": EMAIL_LABELS[predicted],
                "legitimate_probability": float(p_legit),
                "phishing_probability": float(p_phish),
                "error_type": "false_positive" if actual == 0 else "false_negative",
            }
        )
    return pd.DataFrame(
        records,
        columns=[
            "source_row_index",
            "true_label",
            "true_label_name",
            "predicted_label",
            "predicted_label_name",
            "legitimate_probability",
            "phishing_probability",
            "error_type",
        ],
    )


def _verify_url_split():
    """Recreate prepare_url_dataset.py's split in memory and compare all cells."""
    source = pd.read_csv(URL_SOURCE)
    prepared = source.drop(columns=[c for c in URL_REMOVE_FEATURES if c in source.columns])
    if "domain" not in prepared.columns or "label" not in prepared.columns:
        raise ValueError("URL source dataset lacks the expected domain/label columns.")
    X = prepared.drop(columns=["domain", "label"])
    y = prepared["label"]

    from training.predict_url import FEATURES

    if list(X.columns) != FEATURES:
        raise ValueError("URL source feature columns do not match predict_url.FEATURES in order.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    del X_train, y_train
    recreated = X_test.copy()
    recreated["label"] = y_test.values
    saved = pd.read_csv(URL_SAVED_TEST)

    if list(recreated.columns) != list(saved.columns):
        raise ValueError("URL split verification failed: saved test columns/order differ.")
    if recreated.shape != saved.shape:
        raise ValueError(
            f"URL split verification failed: recreated shape {recreated.shape} != saved {saved.shape}."
        )
    if not np.array_equal(recreated.to_numpy(), saved.to_numpy(), equal_nan=True):
        raise ValueError("URL split verification failed: test values or row order differ.")

    return X_test, y_test, {
        "verified": True,
        "comparison": "exact columns, row count, row order, feature values, and labels",
        "source_rows": int(len(source)),
        "test_rows": int(len(recreated)),
        "source_row_reference": "zero-based pandas row index in url_features.csv",
        "feature_columns": list(X.columns),
    }


def _evaluate_email():
    from training.preprocess_email import prepare_email_dataset
    from training.predict_email import predict as predict_email

    X_train, X_test, y_train, y_test, stats = prepare_email_dataset(EMAIL_DATASET)
    del X_train, y_train
    raw = pd.read_csv(EMAIL_DATASET, low_memory=False)
    model = joblib.load(EMAIL_MODEL)
    positions, classes = _class_positions(model, "email_pipeline.pkl")

    y_true = y_test.to_numpy(dtype=int)
    y_pred = np.empty(len(X_test), dtype=int)
    p_legit = np.empty(len(X_test), dtype=float)
    p_phish = np.empty(len(X_test), dtype=float)
    row_refs = X_test.index.to_numpy(dtype=int)

    for position, row_ref in enumerate(row_refs):
        result = predict_email(raw.iloc[int(row_ref)]["text_combined"], model=model)
        if result["normalized_text"] != X_test.iloc[position]:
            raise ValueError("Email normalization differs between preprocessing and inference.")
        y_pred[position] = int(result["prediction"])
        p_legit[position] = float(result["legitimate_probability"])
        p_phish[position] = float(result["phishing_probability"])

    matrix_probabilities = model.predict_proba(X_test.to_numpy())
    if not np.allclose(p_legit, matrix_probabilities[:, positions[0]]) or not np.allclose(
        p_phish, matrix_probabilities[:, positions[1]]
    ):
        raise ValueError("Email predictor probabilities conflict with model classes_ metadata.")

    metrics, matrix = _metric_row(y_true, y_pred, p_phish)
    return {
        "metrics": metrics,
        "confusion": _confusion_frame(matrix),
        "errors": _error_frame(row_refs, y_true, y_pred, p_legit, p_phish),
        "metadata": {
            "dataset": str(EMAIL_DATASET.relative_to(PROJECT_ROOT)),
            "model": str(EMAIL_MODEL.relative_to(PROJECT_ROOT)),
            "split": "prepare_email_dataset(test_size=0.20, random_state=42, stratify=label)",
            "split_summary": stats,
            "source_row_reference": "zero-based pandas row index in phishing_email.csv",
            "model_classes_in_probability_order": classes,
            "probability_mapping": {"0": "legitimate_probability", "1": "phishing_probability"},
            "inference": "training/predict_email.py::predict on original source text; normalization checked against prepared holdout",
            "evaluation_interpretation": "reproduction of the existing holdout; this holdout was previously used for model selection",
        },
    }


def _evaluate_url():
    from training.predict_url import FEATURES

    X_test, y_test, split_metadata = _verify_url_split()
    model = joblib.load(URL_MODEL)
    positions, classes = _class_positions(model, "url_random_forest.pkl")
    y_true = y_test.to_numpy(dtype=int)
    X_model = X_test.loc[:, FEATURES]
    y_pred = np.asarray(model.predict(X_model), dtype=int)
    probabilities = model.predict_proba(X_model)
    p_legit = probabilities[:, positions[0]].astype(float)
    p_phish = probabilities[:, positions[1]].astype(float)
    row_refs = X_test.index.to_numpy(dtype=int)

    metrics, matrix = _metric_row(y_true, y_pred, p_phish)
    return {
        "metrics": metrics,
        "confusion": _confusion_frame(matrix),
        "errors": _error_frame(row_refs, y_true, y_pred, p_legit, p_phish),
        "metadata": {
            "source_dataset": str(URL_SOURCE.relative_to(PROJECT_ROOT)),
            "saved_test_dataset": str(URL_SAVED_TEST.relative_to(PROJECT_ROOT)),
            "model": str(URL_MODEL.relative_to(PROJECT_ROOT)),
            "split": "prepare_url_dataset.py logic: train_test_split(test_size=0.20, random_state=42, stratify=label), after removing the listed weak features",
            "split_verification": split_metadata,
            "model_classes_in_probability_order": classes,
            "probability_mapping": {"0": "legitimate_probability", "1": "phishing_probability"},
            "inference": "predict_url.py feature order and model interface applied to the verified pre-extracted feature matrix; dataset contains no raw URL column",
        },
    }


def run():
    existing = [name for name in OUTPUT_NAMES if (OUTPUT_DIR / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing baseline audit output(s): " + ", ".join(existing)
        )
    for path in (EMAIL_DATASET, EMAIL_MODEL, URL_SOURCE, URL_SAVED_TEST, URL_MODEL):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation input is missing: {path.relative_to(PROJECT_ROOT)}")

    email = _evaluate_email()
    url = _evaluate_url()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root_name": PROJECT_ROOT.name,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "results_are": "freshly computed by this audit runner; existing result files were not read as metric inputs or overwritten",
        "email": email["metadata"],
        "url": url["metadata"],
    }

    # Recheck immediately before creating/writing, so an existing artifact is
    # never replaced if one appeared while evaluation was running.
    existing = [name for name in OUTPUT_NAMES if (OUTPUT_DIR / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing baseline audit output(s): " + ", ".join(existing)
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "email_metrics.csv").write_text(
        pd.DataFrame([email["metrics"]]).to_csv(index=False), encoding="utf-8"
    )
    email["confusion"].to_csv(OUTPUT_DIR / "email_confusion_matrix.csv")
    email["errors"].to_csv(OUTPUT_DIR / "email_errors.csv", index=False)
    (OUTPUT_DIR / "url_metrics.csv").write_text(
        pd.DataFrame([url["metrics"]]).to_csv(index=False), encoding="utf-8"
    )
    url["confusion"].to_csv(OUTPUT_DIR / "url_confusion_matrix.csv")
    url["errors"].to_csv(OUTPUT_DIR / "url_errors.csv", index=False)
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Baseline audit outputs written to {OUTPUT_DIR}")
    print("Email metrics:", json.dumps(email["metrics"], sort_keys=True))
    print("URL metrics:", json.dumps(url["metrics"], sort_keys=True))


if __name__ == "__main__":
    run()
