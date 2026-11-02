import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# 1. Load datasets
# ==========================================

original_path = "dataset/url/urlset_cleaned.csv"
derived_path = "dataset/url/url_features.csv"

original_df = pd.read_csv(original_path)
derived_df = pd.read_csv(derived_path)

print("Original dataset:", original_df.shape)
print("Derived dataset :", derived_df.shape)


# ==========================================
# 2. Verify row alignment
# ==========================================

if len(original_df) != len(derived_df):
    raise ValueError(
        "Original and derived datasets do not have the same number of rows."
    )

if not original_df["domain"].equals(derived_df["domain"]):
    raise ValueError(
        "URL/domain order does not match between the two datasets."
    )

print("URL alignment verified.")


# ==========================================
# 3. Select derived URL features
# ==========================================

weak_features = [
    "has_https",
    "has_ip_address",
    "has_at_symbol",
    "has_exe",
    "has_html"
]

derived_features = [
    col for col in derived_df.columns
    if col not in ["domain", "label"] + weak_features
]

print("\nDerived URL features:", len(derived_features))


# ==========================================
# 4. Build combined dataset
# ==========================================

# Keep original PhishStorm features
original_features = [
    col for col in original_df.columns
    if col not in ["domain", "label"]
]

# Add derived URL features
combined = pd.concat(
    [
        original_df[["domain", "label"] + original_features],
        derived_df[derived_features]
    ],
    axis=1
)

print("\nCombined dataset before cleaning:", combined.shape)


# ==========================================
# 5. Remove duplicate URLs ONLY
# ==========================================

duplicate_url_count = combined["domain"].duplicated().sum()

print("Duplicate URL rows:", duplicate_url_count)

combined = combined.drop_duplicates(
    subset=["domain"],
    keep="first"
)

print(
    "Dataset after duplicate URL removal:",
    combined.shape
)


# ==========================================
# 6. Remove rows with missing values
# ==========================================

before_missing = len(combined)

combined = combined.dropna()

removed_missing = before_missing - len(combined)

print("Rows removed due to missing values:", removed_missing)
print("Final dataset:", combined.shape)


# ==========================================
# 7. Separate features and label
# ==========================================

X = combined.drop(columns=["domain", "label"])
y = combined["label"]


# ==========================================
# 8. Stratified train/test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 9. Save train/test datasets
# ==========================================

train_df = X_train.copy()
train_df["label"] = y_train.values

test_df = X_test.copy()
test_df["label"] = y_test.values

train_path = "dataset/url/combined_features_train.csv"
test_path = "dataset/url/combined_features_test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)


# ==========================================
# 10. Print final information
# ==========================================

print("\n========== FINAL DATASET ==========")

print("Total rows :", len(combined))
print("Features   :", X.shape[1])
print("Train rows :", len(train_df))
print("Test rows  :", len(test_df))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

print("\nSaved:")
print(train_path)
print(test_path)