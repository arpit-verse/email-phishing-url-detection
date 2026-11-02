"""Console-only score audit for candidate unseen-domain URL holdout rows.

This runner never trains a model and never writes audit outputs. It verifies
the split and PSL snapshot, identifies candidate test rows by registrable
domain, and scores the existing deployed model on the precomputed features.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import ipaddress
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The parser and IDNA package are isolated from the project's ML environment.
PSL_SITE_PACKAGES = PROJECT_ROOT / ".venv-psl-audit" / "Lib" / "site-packages"
if not PSL_SITE_PACKAGES.is_dir():
    raise RuntimeError("The approved isolated PSL environment is unavailable.")
sys.path.insert(0, str(PSL_SITE_PACKAGES))

import idna
import joblib
import numpy as np
import pandas as pd
import sklearn
import tldextract
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


EXPECTED_PSL_SHA256 = (
    "0232f4df0831911356e8ee1f4d5c3b6b12c0ed3f6bd66b0f81c8ab5d3d49e9ca"
)
EXPECTED_CANDIDATE_ROWS = 5755
EXPECTED_CANDIDATE_DOMAINS = 5514
EXPECTED_CANDIDATE_LABEL_COUNTS = {0: 2900, 1: 2855}
REMOVE_FEATURES = (
    "has_https",
    "has_ip_address",
    "has_at_symbol",
    "has_exe",
    "has_html",
)
URL_SOURCE = PROJECT_ROOT / "dataset" / "url" / "url_features.csv"
URL_TRAIN = PROJECT_ROOT / "dataset" / "url" / "url_features_train.csv"
URL_TEST = PROJECT_ROOT / "dataset" / "url" / "url_features_test.csv"
URL_MODEL = PROJECT_ROOT / "models" / "url_random_forest.pkl"
URL_RX = re.compile(r"^https?://", re.IGNORECASE)
SCHEME_RX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:(?:/|$)")
CONTROL_OR_WHITESPACE_RX = re.compile(r"[\x00-\x20\x7f]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_identifier(raw_value, extractor):
    """Return (status, normalized host, registrable domain) per PSL audit rules."""
    if pd.isna(raw_value):
        return "missing", None, None

    raw = str(raw_value)
    if not raw or not raw.strip():
        return "missing", None, None
    if raw != raw.strip() or CONTROL_OR_WHITESPACE_RX.search(raw):
        return "malformed", None, None

    if URL_RX.match(raw):
        try:
            parsed = urlsplit(raw)
        except ValueError:
            return "malformed", None, None
        if parsed.scheme.lower() not in {"http", "https"}:
            return "malformed", None, None
    else:
        # Reject URI schemes such as ftp:// and file:/, while accepting host:port.
        if SCHEME_RX.match(raw):
            return "malformed", None, None
        try:
            parsed = urlsplit("//" + raw.lstrip("/"))
        except ValueError:
            return "malformed", None, None

    if not parsed.netloc or "@" in parsed.netloc:
        return "malformed", None, None
    try:
        _ = parsed.port  # Access validates malformed and out-of-range ports.
        hostname = parsed.hostname
    except ValueError:
        return "malformed", None, None
    if not hostname:
        return "malformed", None, None

    hostname = hostname.rstrip(".")
    if not hostname:
        return "malformed", None, None
    try:
        ip_address = ipaddress.ip_address(hostname)
    except ValueError:
        ip_address = None
    if ip_address is not None:
        return "ip", ip_address.compressed.lower(), None

    try:
        hostname = idna.encode(
            hostname,
            uts46=True,
            std3_rules=True,
            transitional=False,
        ).decode("ascii").lower().rstrip(".")
    except (idna.IDNAError, UnicodeError):
        return "malformed", None, None
    if not hostname or any(not label for label in hostname.split(".")):
        return "malformed", None, None

    extracted = extractor(hostname)
    if not extracted.domain or not extracted.suffix:
        return "unassignable", hostname, None
    registrable_domain = f"{extracted.domain}.{extracted.suffix}".lower()
    return "dns", hostname, registrable_domain


def _compare_split(expected: pd.DataFrame, saved_path: Path, name: str) -> None:
    saved = pd.read_csv(saved_path)
    if list(expected.columns) != list(saved.columns):
        raise ValueError(f"{name} saved split columns or order do not match replay.")
    if expected.shape != saved.shape:
        raise ValueError(
            f"{name} saved split shape {saved.shape} differs from replay {expected.shape}."
        )
    if not np.array_equal(expected.to_numpy(), saved.to_numpy(), equal_nan=True):
        raise ValueError(f"{name} saved split values or row order do not match replay.")


def _class_positions(model, features: list[str]) -> dict[int, int]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("Model classes_ metadata is unavailable.")
    normalized = []
    for value in classes:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Model classes_ contains an unsupported value.") from exc
        if numeric not in (0.0, 1.0):
            raise ValueError("Model classes_ must represent 0=legitimate and 1=phishing.")
        normalized.append(int(numeric))
    if len(normalized) != 2 or set(normalized) != {0, 1}:
        raise ValueError("Model classes_ must contain exactly classes 0 and 1.")
    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features != features:
        raise ValueError("Model feature_names_in_ does not match the expected 25-feature order.")
    if getattr(model, "n_features_in_", None) != len(features):
        raise ValueError("Model feature count does not match the expected feature order.")
    return {label: normalized.index(label) for label in (0, 1)}


def _counts_by_label(labels) -> dict[int, int]:
    counts = Counter(int(value) for value in labels)
    return {0: counts.get(0, 0), 1: counts.get(1, 0)}


def _feature_health(frame: pd.DataFrame) -> dict[str, object]:
    missing_by_column = frame.isna().sum()
    try:
        numeric = frame.to_numpy(dtype=float)
        nonfinite_by_column = (~np.isfinite(numeric)).sum(axis=0)
        nonnumeric = 0
    except (TypeError, ValueError):
        numeric = None
        nonfinite_by_column = None
        nonnumeric = 1
    return {
        "missing_cells": int(missing_by_column.sum()),
        "missing_columns": {
            str(column): int(value)
            for column, value in missing_by_column.items()
            if int(value)
        },
        "nonfinite_cells": (
            int(nonfinite_by_column.sum()) if nonfinite_by_column is not None else None
        ),
        "nonfinite_columns": (
            {
                str(column): int(value)
                for column, value in zip(frame.columns, nonfinite_by_column)
                if int(value)
            }
            if nonfinite_by_column is not None
            else {}
        ),
        "nonnumeric_conversion_failed": bool(nonnumeric),
        "usable": numeric is not None
        and not bool(missing_by_column.any())
        and not bool(np.any(~np.isfinite(numeric))),
    }


def run(psl_path: Path, expected_hash: str) -> None:
    if expected_hash.lower() != EXPECTED_PSL_SHA256:
        raise ValueError("Supplied expected PSL hash is not the approved hash.")
    if not psl_path.is_file():
        raise FileNotFoundError("The supplied local PSL file does not exist.")
    psl_hash = _sha256(psl_path)
    if psl_hash != EXPECTED_PSL_SHA256:
        raise ValueError(f"PSL SHA-256 mismatch: got {psl_hash}.")

    for path in (URL_SOURCE, URL_TRAIN, URL_TEST, URL_MODEL):
        if not path.is_file():
            raise FileNotFoundError(f"Required audit input is missing: {path.relative_to(PROJECT_ROOT)}")

    # Explicit local file URL, no cache writes, no network URLs, no bundled fallback.
    extractor = tldextract.TLDExtract(
        cache_dir=None,
        suffix_list_urls=(psl_path.resolve().as_uri(),),
        fallback_to_snapshot=False,
        include_psl_private_domains=True,
    )
    private_probe = extractor("foo.blogspot.com")
    if private_probe.suffix != "blogspot.com" or not private_probe.is_private:
        raise ValueError("Local PSL PRIVATE-section parser verification failed.")

    from training.predict_url import FEATURES

    source = pd.read_csv(URL_SOURCE)
    raw_identifiers = pd.read_csv(
        URL_SOURCE,
        usecols=["domain"],
        dtype={"domain": "string"},
        keep_default_na=False,
    )["domain"]
    if len(raw_identifiers) != len(source):
        raise ValueError("Raw URL identifier row mapping does not match the source dataset.")

    prepared = source.drop(columns=[column for column in REMOVE_FEATURES if column in source.columns])
    if "domain" not in prepared.columns or "label" not in prepared.columns:
        raise ValueError("Source dataset is missing its domain or label column.")
    X = prepared.drop(columns=["domain", "label"])
    y = prepared["label"]
    features = list(FEATURES)
    if list(X.columns) != features or len(features) != 25:
        raise ValueError("Source feature order does not match predict_url.FEATURES.")
    if y.isna().any() or set(pd.unique(y)) != {0, 1}:
        raise ValueError("Source labels are missing or do not map exactly to 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    expected_train = X_train.copy()
    expected_train["label"] = y_train.to_numpy()
    expected_test = X_test.copy()
    expected_test["label"] = y_test.to_numpy()
    _compare_split(expected_train, URL_TRAIN, "Training")
    _compare_split(expected_test, URL_TEST, "Test")

    # Normalize each distinct source identifier once, preserving source indices.
    normalized: dict[str, tuple[str, str | None, str | None]] = {}
    for raw_value in raw_identifiers:
        key = str(raw_value)
        if key not in normalized:
            normalized[key] = _normalize_identifier(raw_value, extractor)

    train_domain_set: set[str] = set()
    for row_index in X_train.index:
        status, _hostname, domain = normalized[str(raw_identifiers.iloc[int(row_index)])]
        if status == "dns" and domain is not None:
            train_domain_set.add(domain)

    candidate_indices: list[int] = []
    status_counts: Counter[str] = Counter()
    excluded_label_counts: dict[str, Counter[int]] = {}
    test_domains: set[str] = set()
    candidate_domains: set[str] = set()
    for row_index in X_test.index:
        row_index = int(row_index)
        status, _hostname, domain = normalized[str(raw_identifiers.iloc[row_index])]
        if status == "dns" and domain is not None:
            test_domains.add(domain)
            if domain not in train_domain_set:
                candidate_indices.append(row_index)
                candidate_domains.add(domain)
                status_counts["candidate_unseen_domain"] += 1
                continue
            exclusion = "shared_registrable_domain"
        else:
            exclusion = status
        status_counts[exclusion] += 1
        excluded_label_counts.setdefault(exclusion, Counter())[int(y.loc[row_index])] += 1

    candidate_indices = sorted(candidate_indices)
    candidate_label_counts = _counts_by_label(y.loc[candidate_indices].to_numpy())
    if (
        len(candidate_indices) != EXPECTED_CANDIDATE_ROWS
        or len(candidate_domains) != EXPECTED_CANDIDATE_DOMAINS
        or candidate_label_counts != EXPECTED_CANDIDATE_LABEL_COUNTS
    ):
        raise ValueError(
            "Candidate subset differs from the approved expected counts; stopping before scoring. "
            f"rows={len(candidate_indices)}, domains={len(candidate_domains)}, "
            f"labels={candidate_label_counts}."
        )

    candidate_domain_values = {
        normalized[str(raw_identifiers.iloc[index])][2]
        for index in candidate_indices
    }
    if None in candidate_domain_values or candidate_domain_values & train_domain_set:
        raise ValueError("Candidate domain-disjointness verification failed.")
    if sum(status_counts.values()) != len(X_test):
        raise ValueError("Test-row candidate/exclusion accounting is incomplete.")

    train_feature_health = _feature_health(X_train.loc[:, features])
    test_feature_health = _feature_health(X_test.loc[:, features])
    candidate_X = X_test.loc[candidate_indices, features]
    candidate_feature_health = _feature_health(candidate_X)
    if not candidate_feature_health["usable"]:
        raise ValueError(
            "Candidate feature matrix contains missing, non-finite, or nonnumeric values; "
            "stopping without imputation or scoring."
        )

    # All prerequisites and expected candidate counts are checked before model inference.
    model = joblib.load(URL_MODEL)
    class_positions = _class_positions(model, features)
    predictions = np.asarray(model.predict(candidate_X), dtype=int)
    probability_matrix = np.asarray(model.predict_proba(candidate_X), dtype=float)
    if probability_matrix.shape != (len(candidate_indices), 2):
        raise ValueError("Model returned an unexpected probability matrix shape.")
    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("Model returned an unexpected prediction class.")
    p_legitimate = probability_matrix[:, class_positions[0]]
    p_phishing = probability_matrix[:, class_positions[1]]
    y_true = y.loc[candidate_indices].to_numpy(dtype=int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    auc = float(roc_auc_score(y_true, p_phishing)) if len(np.unique(y_true)) == 2 else None

    print("AUDIT_LABEL=performance on a candidate domain-disjoint subset of the reconstructed split")
    print("AUDIT_TYPE=existing-model scoring; no training, threshold tuning, or feature changes")
    print("MODEL_PROVENANCE=unverified exact training-data linkage; do not interpret as proven unseen-domain performance")
    print("FILE_SHA256=" + json.dumps({
        "model": _sha256(URL_MODEL),
        "source_dataset": _sha256(URL_SOURCE),
        "train_csv": _sha256(URL_TRAIN),
        "test_csv": _sha256(URL_TEST),
        "psl": psl_hash,
    }, sort_keys=True))
    print("VERSIONS=" + json.dumps({
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tldextract": tldextract.__version__,
        "idna": idna.__version__,
    }, sort_keys=True))
    print("PSL_CONFIG=" + json.dumps({
        "sha256": psl_hash,
        "private_section": True,
        "fallback_to_snapshot": False,
        "suffix_list_source": "local file:// snapshot",
        "parser_cache": False,
    }, sort_keys=True))
    print("SPLIT_VERIFIED=exact train/test CSV columns, shape, row order, features, and labels")
    print("SPLIT_ROWS=" + json.dumps({
        "source": len(source),
        "train": len(X_train),
        "test": len(X_test),
        "train_labels": _counts_by_label(y_train.to_numpy()),
        "test_labels": _counts_by_label(y_test.to_numpy()),
    }, sort_keys=True))
    print("CANDIDATE=" + json.dumps({
        "rows": len(candidate_indices),
        "registrable_domains": len(candidate_domains),
        "labels": candidate_label_counts,
        "domain_absence_verified": True,
    }, sort_keys=True))
    print("TEST_ROW_ACCOUNTING=" + json.dumps({
        "candidate_unseen_domain": status_counts.get("candidate_unseen_domain", 0),
        "excluded_shared_registrable_domain": status_counts.get("shared_registrable_domain", 0),
        "excluded_malformed": status_counts.get("malformed", 0),
        "excluded_unassignable": status_counts.get("unassignable", 0),
        "excluded_ip": status_counts.get("ip", 0),
        "excluded_missing": status_counts.get("missing", 0),
        "excluded_label_counts": {
            name: {"0_legitimate": counts.get(0, 0), "1_phishing": counts.get(1, 0)}
            for name, counts in sorted(excluded_label_counts.items())
        },
    }, sort_keys=True))
    print("FEATURE_HEALTH=" + json.dumps({
        "train": train_feature_health,
        "test": test_feature_health,
        "candidate": candidate_feature_health,
    }, sort_keys=True))
    print("MODEL_CLASSES=" + json.dumps([int(value) for value in model.classes_]))
    print("PROBABILITY_MAPPING=0: legitimate_probability; 1: phishing_probability")
    print("CONFUSION_MATRIX_ROWS=true [LEGITIMATE, PHISHING], COLUMNS=predicted [LEGITIMATE, PHISHING]")
    print(np.array2string(matrix))
    print("METRICS=" + json.dumps({
        "accuracy": float(accuracy_score(y_true, predictions)),
        "phishing_precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "phishing_recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "phishing_f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "phishing_roc_auc": auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }, sort_keys=True))
    print("LIMITATION=Candidate domains are absent from the reconstructed train partition; exact training rows behind the deployed model are not proven by an artifact manifest.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psl", required=True, type=Path, help="Local approved PSL snapshot.")
    parser.add_argument(
        "--expected-psl-sha256",
        required=True,
        help="Expected PSL SHA-256; must equal the approved audit snapshot hash.",
    )
    args = parser.parse_args()
    run(args.psl, args.expected_psl_sha256)


if __name__ == "__main__":
    main()
