import pandas as pd
import joblib

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==============================
# 1. Load combined dataset
# ==============================

train_path = "dataset/url/combined_features_train.csv"
test_path = "dataset/url/combined_features_test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)


# ==============================
# 2. Separate features and label
# ==============================

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

print("Number of features:", X_train.shape[1])


# ==============================
# 3. Define component models
# ==============================

decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=12,
    class_weight="balanced"
)

random_forest = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

logistic_regression = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])


# ==============================
# 4. Create Voting Ensemble
# ==============================

ensemble = VotingClassifier(
    estimators=[
        ("dt", decision_tree),
        ("rf", random_forest),
        ("lr", logistic_regression)
    ],
    voting="soft"
)


# ==============================
# 5. Train ensemble
# ==============================

print("\nTraining Voting Ensemble...")

ensemble.fit(X_train, y_train)

print("Training completed.")


# ==============================
# 6. Make predictions
# ==============================

y_pred = ensemble.predict(X_test)


# ==============================
# 7. Evaluate model
# ==============================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== ENSEMBLE RESULTS ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Phishing"]
))


# ==============================
# 8. Save model
# ==============================

model_path = "models/combined_voting_ensemble.pkl"

joblib.dump(ensemble, model_path)

print("\nModel saved to:", model_path)