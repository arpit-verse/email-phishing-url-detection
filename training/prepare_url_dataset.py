import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "dataset/url/url_features.csv"
TRAIN_FILE = "dataset/url/url_features_train.csv"
TEST_FILE = "dataset/url/url_features_test.csv"

REMOVE_FEATURES = [
    "has_https",
    "has_ip_address",
    "has_at_symbol",
    "has_exe",
    "has_html"
]

print("\n========== URL DATASET LOADED ==========")

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))
print("Columns:", len(df.columns))

# --------------------------------------------------
# Remove weak / constant features
# --------------------------------------------------

print("\n========== REMOVING WEAK FEATURES ==========")

for feature in REMOVE_FEATURES:
    if feature in df.columns:
        print("Removing:", feature)
        df = df.drop(columns=[feature])

# --------------------------------------------------
# Separate features and label
# --------------------------------------------------

X = df.drop(columns=["domain", "label"])
y = df["label"]

print("\n========== FINAL FEATURES ==========")

for feature in X.columns:
    print(feature)

print("\nNumber of features:", X.shape[1])

# --------------------------------------------------
# Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_df = X_train.copy()
train_df["label"] = y_train.values

test_df = X_test.copy()
test_df["label"] = y_test.values

# --------------------------------------------------
# Save datasets
# --------------------------------------------------

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("\n========== TRAIN / TEST SPLIT ==========")

print("Training rows:", len(train_df))
print("Testing rows :", len(test_df))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

print("\n========== DATASET PREPARATION COMPLETE ==========")

print("Training dataset saved to:")
print(TRAIN_FILE)

print("\nTesting dataset saved to:")
print(TEST_FILE)