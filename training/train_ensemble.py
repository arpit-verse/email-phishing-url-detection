import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

TRAIN_FILE = "dataset/url/train.csv"
TEST_FILE = "dataset/url/test.csv"

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\n========== DATA LOADED ==========")
print("Training rows:", len(train_df))
print("Testing rows :", len(test_df))


# ============================================================
# 2. FEATURES
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


# ============================================================
# 3. X AND y
# ============================================================

X_train = train_df[features]
y_train = train_df["label"]

X_test = test_df[features]
y_test = test_df["label"]


# ============================================================
# 4. CREATE INDIVIDUAL MODELS
# ============================================================

decision_tree = DecisionTreeClassifier(
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

logistic_regression = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ============================================================
# 5. CREATE VOTING ENSEMBLE
# ============================================================

ensemble = VotingClassifier(
    estimators=[
        ("decision_tree", decision_tree),
        ("random_forest", random_forest),
        ("logistic_regression", logistic_regression)
    ],
    voting="soft"
)


# ============================================================
# 6. TRAIN
# ============================================================

print("\n========== TRAINING VOTING ENSEMBLE ==========")

ensemble.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 7. PREDICTIONS
# ============================================================

y_pred = ensemble.predict(
    X_test
)


# ============================================================
# 8. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


print("\n========== VOTING ENSEMBLE RESULTS ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========== CONFUSION MATRIX ==========")

print(cm)


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Phishing"
        ]
    )
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

MODEL_FILE = "models/voting_ensemble.pkl"

joblib.dump(
    ensemble,
    MODEL_FILE
)

print("\nModel saved to:")
print(MODEL_FILE)


print("\n========================================")
print("VOTING ENSEMBLE EXPERIMENT COMPLETE")
print("========================================")