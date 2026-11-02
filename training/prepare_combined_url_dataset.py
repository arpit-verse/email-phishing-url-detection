import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# LOAD DATA
# ============================================================

ORIGINAL_FILE = "dataset/url/urlset_cleaned.csv"
URL_FEATURE_FILE = "dataset/url/url_features.csv"

print("\n========== LOADING DATA ==========")

original = pd.read_csv(ORIGINAL_FILE)
url_features = pd.read_csv(URL_FEATURE_FILE)

print("Original dataset:", original.shape)
print("URL feature dataset:", url_features.shape)


# ============================================================
# CLEAN ORIGINAL DATA
# ============================================================

print("\n========== CLEANING ORIGINAL DATA ==========")

original = original.dropna(subset=[
    "ranking",
    "mld_res",
    "mld.ps_res",
    "card_rem",
    "ratio_Rrem",
    "ratio_Arem",
    "jaccard_RR",
    "jaccard_RA",
    "jaccard_AR",
    "jaccard_AA",
    "jaccard_ARrd",
    "jaccard_ARrem",
    "label"
]).copy()

print("Rows after cleaning:", len(original))


# ============================================================
# REMOVE DUPLICATE URLS
# ============================================================

print("\n========== DUPLICATE URL CHECK ==========")

duplicate_count = original["domain"].duplicated(keep=False).sum()

print("Rows belonging to duplicated URLs:", duplicate_count)

if duplicate_count > 0:
    original = original.drop_duplicates(
        subset=["domain"],
        keep="first"
    ).copy()

print("Rows after duplicate removal:", len(original))


# ============================================================
# MATCH URL FEATURE DATASET
# ============================================================

print("\n========== MATCHING URL FEATURES ==========")

# Keep only rows whose URL exists in the cleaned original dataset
url_features = url_features[
    url_features["domain"].isin(original["domain"])
].copy()

print("Matched URL-feature rows:", len(url_features))


# Check whether every original URL has URL features
missing_urls = set(original["domain"]) - set(url_features["domain"])

print("Original URLs without URL features:", len(missing_urls))

if len(missing_urls) > 0:
    print("ERROR: Some URLs are missing from URL features.")
    raise SystemExit(1)


# ============================================================
# KEEP EXACT SAME ROWS
# ============================================================

# Reorder URL features to exactly match original dataset order
url_features = (
    url_features
    .set_index("domain")
    .loc[original["domain"]]
    .reset_index()
)

print("Rows after exact alignment:", len(url_features))


# ============================================================
# VERIFY ALIGNMENT
# ============================================================

print("\n========== ALIGNMENT CHECK ==========")

if not original["domain"].equals(url_features["domain"]):
    print("ERROR: URL alignment failed.")
    raise SystemExit(1)

print("SUCCESS: URLs are perfectly aligned.")


# ============================================================
# CREATE SAME TRAIN / TEST INDICES
# ============================================================

print("\n========== CREATING COMMON TRAIN / TEST SPLIT ==========")

indices = list(range(len(original)))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=original["label"]
)

train_idx = sorted(train_idx)
test_idx = sorted(test_idx)

print("Training rows:", len(train_idx))
print("Testing rows :", len(test_idx))


# ============================================================
# ORIGINAL 12 FEATURES
# ============================================================

original_features = [
    "ranking",
    "mld_res",
    "mld.ps_res",
    "card_rem",
    "ratio_Rrem",
    "ratio_Arem",
    "jaccard_RR",
    "jaccard_RA",
    "jaccard_AR",
    "jaccard_AA",
    "jaccard_ARrd",
    "jaccard_ARrem"
]


# ============================================================
# URL FEATURES
# ============================================================

url_feature_columns = [
    column
    for column in url_features.columns
    if column not in ["domain", "label"]
]

print("\n========== FEATURE COUNTS ==========")
print("Original features:", len(original_features))
print("URL features     :", len(url_feature_columns))
print("Combined features:", len(original_features) + len(url_feature_columns))


# ============================================================
# BUILD COMBINED DATASET
# ============================================================

combined = pd.concat(
    [
        original[["domain"] + original_features].reset_index(drop=True),
        url_features[url_feature_columns].reset_index(drop=True)
    ],
    axis=1
)

combined["label"] = original["label"].reset_index(drop=True)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n========== FINAL DATASET ==========")

print("Combined shape:", combined.shape)

print("Expected feature count:", 37)
print(
    "Actual feature count:",
    len(combined.columns) - 2
)
print("(domain + 37 features + label)")


if len(combined.columns) - 2 != 37:
    print("ERROR: Expected 37 features.")
    raise SystemExit(1)


# ============================================================
# CREATE TRAIN / TEST DATASETS
# ============================================================

train_data = combined.iloc[train_idx].copy()
test_data = combined.iloc[test_idx].copy()


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n========== DATA LEAKAGE CHECK ==========")

train_urls = set(train_data["domain"])
test_urls = set(test_data["domain"])

overlap = train_urls.intersection(test_urls)

print("URLs appearing in both train and test:", len(overlap))

if len(overlap) != 0:
    print("ERROR: DATA LEAKAGE DETECTED.")
    raise SystemExit(1)

print("SUCCESS: No URL leakage detected.")


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n========== LABEL DISTRIBUTION ==========")

print("\nTraining:")
print(train_data["label"].value_counts())

print("\nTesting:")
print(test_data["label"].value_counts())


# ============================================================
# SAVE
# ============================================================

train_output = "dataset/url/combined_features_train.csv"
test_output = "dataset/url/combined_features_test.csv"

train_data.to_csv(train_output, index=False)
test_data.to_csv(test_output, index=False)

print("\n========== COMPLETE ==========")

print("Training dataset saved to:")
print(train_output)

print("\nTesting dataset saved to:")
print(test_output)