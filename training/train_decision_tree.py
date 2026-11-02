import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD TRAINING AND TESTING DATA
# ============================================================

TRAIN_FILE = "dataset/url/train.csv"
TEST_FILE = "dataset/url/test.csv"

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\n========== DATA LOADED ==========")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


# ============================================================
# 2. DEFINE FEATURES
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
# 3. CREATE X AND y
# ============================================================

X_train = train_df[features]
y_train = train_df["label"]

X_test = test_df[features]
y_test = test_df["label"]


print("\n========== TRAINING DATA ==========")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\n========== TESTING DATA ==========")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 4. CREATE DECISION TREE
# ============================================================

model = DecisionTreeClassifier(
    random_state=42
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\n========== TRAINING DECISION TREE ==========")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. EVALUATE MODEL
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


print("\n========== DECISION TREE RESULTS ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========== CONFUSION MATRIX ==========")

print(cm)


# ============================================================
# 9. CLASSIFICATION REPORT
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
# 10. FEATURE IMPORTANCE
# ============================================================

print("\n========== FEATURE IMPORTANCE ==========")

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(index=False)
)


# ============================================================
# 11. SAVE FEATURE IMPORTANCE
# ============================================================

importance.to_csv(
    "dataset/url/decision_tree_feature_importance.csv",
    index=False
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

import joblib

MODEL_FILE = "models/decision_tree.pkl"

joblib.dump(
    model,
    MODEL_FILE
)

print("\nModel saved to:")
print(MODEL_FILE)


print("\n========================================")
print("DECISION TREE EXPERIMENT COMPLETE")
print("========================================")