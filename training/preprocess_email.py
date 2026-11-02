"""
Email Preprocessing and Leakage-Safe Dataset Preparation Module.

Corpus Composition:
- dataset/email/phishing_email.csv is a consolidated dataset of 82,486 emails compiled from:
  1. Enron (29,767 records: benign corporate emails & spam)
  2. Ling-Spam (2,859 records: academic linguistic corpus & spam)
  3. CEAS 2008 (39,154 records: CEAS challenge benchmark)
  4. SpamAssassin (5,809 records: Apache SpamAssassin corpus)
  5. Nazario (1,565 records: Jose Nazario phishing corpus)
  6. Nigerian Fraud (3,332 records: advance-fee fraud scam corpus)
  Total: 82,486 records

Label Interpretation:
- 0: Legitimate / Benign email
- 1: Phishing / Malicious / Fraud email

Conflicting Label Handling Policy:
- If duplicate text records appear with conflicting labels (e.g. identical text labeled 0 and 1),
  the conflicting records are completely dropped from the training and evaluation corpus
  to prevent ambiguous ground-truth signals and label noise.
"""

import re
import html
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMAIL_DATASET = PROJECT_ROOT / "dataset" / "email" / "phishing_email.csv"


def normalize_email_text(text: str) -> str:
    """
    Normalizes email text for both batch training and live inference.
    - Handles non-string/null inputs safely.
    - Decodes HTML entities (e.g., &amp; -> &).
    - Strips HTML markup tags.
    - Replaces URLs with a canonical token 'httpurl'.
    - Replaces email addresses with a canonical token 'emailaddress'.
    - Strips non-alphanumeric characters while preserving single spaces.
    - Normalizes to lowercase and strips surrounding whitespace.
    """
    if text is None or pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize URLs
    text = re.sub(r"https?://\S+|www\.\S+", " httpurl ", text, flags=re.IGNORECASE)

    # Normalize email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " emailaddress ", text)

    # Strip non-alphanumeric characters (keep alphanumeric and spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Collapse multiple whitespace characters into single space
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()


def prepare_email_dataset(
    dataset_path=None,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Loads, cleans, deduplicates, and splits the email dataset.

    Guarantees:
    - Removes empty/whitespace-only records.
    - Analyzes and handles duplicate text groups:
      * Duplicate texts with conflicting labels are dropped.
      * Duplicate texts with consistent labels are deduplicated to retain a single canonical instance.
    - Performs stratified train/test split.
    - Asserts that no identical email texts exist across train and test partitions (zero leakage).

    Returns:
        X_train (pd.Series): Training text series
        X_test  (pd.Series): Test text series
        y_train (pd.Series): Training labels
        y_test  (pd.Series): Test labels
        stats   (dict): Dataset summary statistics
    """
    if dataset_path is None:
        target_path = DEFAULT_EMAIL_DATASET
    else:
        target_path = Path(dataset_path)
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Email dataset not found at: {target_path}")

    # Load dataset
    df = pd.read_csv(target_path, low_memory=False)
    initial_rows = len(df)

    # Verify expected schema
    if "text_combined" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Dataset at {target_path} missing required columns 'text_combined' or 'label'.")

    # Normalize label values to integer (0 or 1)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Ensure label semantics: only 0 and 1 allowed
    valid_mask = df["label"].isin([0, 1])
    df = df[valid_mask].copy()

    # Apply text normalization
    df["cleaned_text"] = df["text_combined"].apply(normalize_email_text)

    # Remove empty or whitespace-only records
    df = df[df["cleaned_text"].str.len() > 0].copy()
    non_empty_rows = len(df)

    # ----------------------------------------------------
    # Duplicate-group analysis and conflict resolution
    # ----------------------------------------------------
    # Group by cleaned_text and check for conflicting labels
    label_nunique = df.groupby("cleaned_text")["label"].nunique()
    conflicting_texts = label_nunique[label_nunique > 1].index

    num_conflicting = len(conflicting_texts)
    if num_conflicting > 0:
        # Policy: Drop all rows whose text has conflicting labels
        df = df[~df["cleaned_text"].isin(conflicting_texts)].copy()

    # Deduplicate consistent duplicate texts (keep first occurrence)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["cleaned_text"], keep="first").copy()
    dedup_rows = len(df)

    # ----------------------------------------------------
    # Leakage-safe stratified train/test split
    # ----------------------------------------------------
    X = df["cleaned_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Verify zero data leakage
    train_set = set(X_train)
    test_set = set(X_test)
    overlap = train_set.intersection(test_set)
    if len(overlap) > 0:
        raise ValueError(f"Critical data leakage detected: {len(overlap)} emails overlap between train and test sets.")

    stats = {
        "initial_rows": initial_rows,
        "non_empty_rows": non_empty_rows,
        "conflicting_text_groups_dropped": num_conflicting,
        "duplicates_removed": before_dedup - dedup_rows,
        "final_clean_rows": dedup_rows,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_class_distribution": y_train.value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict(),
        "leakage_overlap_count": len(overlap)
    }

    return X_train, X_test, y_train, y_test, stats


if __name__ == "__main__":
    print("Testing email dataset preparation and leakage-safe split...")
    X_tr, X_te, y_tr, y_te, s = prepare_email_dataset()
    print("Preparation complete.")
    for k, v in s.items():
        print(f"  {k}: {v}")
