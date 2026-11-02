import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATASET
# ============================================================

INPUT_FILE = "dataset/url/urlset_cleaned.csv"

df = pd.read_csv(INPUT_FILE)

print("\n========== DATASET LOADED ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. CONVERT NUMERICAL FEATURES
# ============================================================

features = [
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

for feature in features:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


# ============================================================
# 3. REMOVE ROWS WITH MISSING FEATURES
# ============================================================

print("\n========== MISSING VALUES BEFORE ==========")

print(df[features].isna().sum())

missing_feature_rows = df[features].isna().any(axis=1).sum()

print(
    "\nRows with missing feature values:",
    missing_feature_rows
)


# For now, because there is only ONE missing ranking value,
# remove that row from the modeling dataset.
#
# The original cleaned dataset is NOT modified.

df = df.dropna(subset=features).copy()

print(
    "Rows after removing missing feature values:",
    len(df)
)


# ============================================================
# 4. CHECK DUPLICATE URLs
# ============================================================

print("\n========== DUPLICATE URL CHECK ==========")

duplicate_url_count = df["domain"].duplicated(
    keep=False
).sum()

print(
    "Rows belonging to duplicated URLs:",
    duplicate_url_count
)


# ============================================================
# 5. CREATE UNIQUE URL GROUPS
# ============================================================
#
# Every URL receives a group ID.
#
# This allows us to keep the same URL from appearing
# in both training and testing data.
#
# The duplicate URL records therefore stay together.

df["url_group"] = pd.factorize(
    df["domain"]
)[0]


# ============================================================
# 6. CREATE GROUP-LEVEL LABEL
# ============================================================
#
# We already checked that duplicated URLs do not have
# conflicting labels.
#
# Therefore each URL group has one label.

group_labels = (
    df.groupby("url_group")["label"]
    .first()
)


# ============================================================
# 7. SPLIT UNIQUE URL GROUPS
# ============================================================

groups = group_labels.index
labels = group_labels.values


train_groups, test_groups = train_test_split(
    groups,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


# ============================================================
# 8. BUILD TRAINING AND TESTING DATA
# ============================================================

train_df = df[
    df["url_group"].isin(train_groups)
].copy()

test_df = df[
    df["url_group"].isin(test_groups)
].copy()


# ============================================================
# 9. REMOVE INTERNAL GROUP COLUMN
# ============================================================

train_df = train_df.drop(
    columns=["url_group"]
)

test_df = test_df.drop(
    columns=["url_group"]
)


# ============================================================
# 10. DISPLAY SPLIT INFORMATION
# ============================================================

print("\n========== TRAIN / TEST SPLIT ==========")

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print(
    "\nTraining percentage:",
    len(train_df) / len(df) * 100
)

print(
    "Testing percentage:",
    len(test_df) / len(df) * 100
)


# ============================================================
# 11. CHECK LABEL DISTRIBUTION
# ============================================================

print("\n========== TRAINING LABEL DISTRIBUTION ==========")

print(
    train_df["label"]
    .value_counts()
)

print("\nTraining percentages:")

print(
    train_df["label"]
    .value_counts(normalize=True) * 100
)


print("\n========== TESTING LABEL DISTRIBUTION ==========")

print(
    test_df["label"]
    .value_counts()
)

print("\nTesting percentages:")

print(
    test_df["label"]
    .value_counts(normalize=True) * 100
)


# ============================================================
# 12. FINAL DUPLICATE-LEAKAGE CHECK
# ============================================================

train_urls = set(train_df["domain"])
test_urls = set(test_df["domain"])

overlap = train_urls.intersection(test_urls)

print("\n========== DATA LEAKAGE CHECK ==========")

print(
    "URLs appearing in both train and test:",
    len(overlap)
)

if len(overlap) == 0:
    print("SUCCESS: No URL leakage detected.")
else:
    print("WARNING: URL leakage detected!")


# ============================================================
# 13. SAVE DATASETS
# ============================================================

TRAIN_FILE = "dataset/url/train.csv"
TEST_FILE = "dataset/url/test.csv"

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# 14. FINAL INFORMATION
# ============================================================

print("\n========== DATASET PREPARATION COMPLETE ==========")

print("Training dataset saved to:")
print(TRAIN_FILE)

print("\nTesting dataset saved to:")
print(TEST_FILE)

print(
    "\nOriginal dataset remains unchanged:"
)
print(INPUT_FILE)