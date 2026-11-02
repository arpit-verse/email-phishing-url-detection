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
# FILE PATHS
# ============================================================

TRAIN_FILE = "dataset/url/combined_features_train.csv"
TEST_FILE = "dataset/url/combined_features_test.csv"
MODEL_FILE = "models/combined_decision_tree.pkl"


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\n========== DATA LOADED ==========")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


# ============================================================
# SEPARATE FEATURES AND LABEL
# ============================================================

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]


# ============================================================
# DATA SHAPES
# ============================================================

print("\n========== TRAINING DATA ==========")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\n========== TESTING DATA ==========")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# FEATURE CHECK
# ============================================================

print("\n========== FEATURES USED ==========")

for i, feature in enumerate(X_train.columns, start=1):
    print(f"{i:2}. {feature}")

print("\nNumber of features:", X_train.shape[1])


# ============================================================
# TRAIN DECISION TREE
# ============================================================

print("\n========== TRAINING COMBINED DECISION TREE ==========")

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== COMBINED DECISION TREE RESULTS ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"]
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance.to_string(index=False))


# ============================================================
# SAVE MODEL
# ============================================================

import joblib

joblib.dump(model, MODEL_FILE)

print("\nModel saved to:")
print(MODEL_FILE)

print("\n========================================")
print("COMBINED DECISION TREE EXPERIMENT COMPLETE")
print("========================================")