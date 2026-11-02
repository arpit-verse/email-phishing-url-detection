import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

TRAIN_FILE = "dataset/url/combined_features_train.csv"
TEST_FILE = "dataset/url/combined_features_test.csv"
MODEL_FILE = "models/combined_random_forest.pkl"


print("\n========== DATA LOADED ==========")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Training rows:", len(train_df))
print("Testing rows :", len(test_df))


# --------------------------------------------------
# Separate features and labels
# --------------------------------------------------

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

print("\n========== DATA SHAPES ==========")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# --------------------------------------------------
# Train Random Forest
# --------------------------------------------------

print("\n========== TRAINING URL RANDOM FOREST ==========")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete.")


# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== URL RANDOM FOREST RESULTS ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"]
    )
)


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

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


# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(model, MODEL_FILE)

print("\nModel saved to:")
print(MODEL_FILE)

print("\n========================================")
print("URL RANDOM FOREST EXPERIMENT COMPLETE")
print("========================================")