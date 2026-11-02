import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. Load training data ONLY
# ==========================================

train_path = "dataset/url/combined_features_train.csv"

train_df = pd.read_csv(train_path)

print("========== DATA LOADED ==========")
print("Training rows:", len(train_df))
print("Training columns:", train_df.shape[1])


# ==========================================
# 2. Separate features and label
# ==========================================

X = train_df.drop(columns=["label"])
y = train_df["label"]

print("\nNumber of features:", X.shape[1])


# ==========================================
# 3. Define models
# ==========================================

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

voting_ensemble = VotingClassifier(
    estimators=[
        ("dt", decision_tree),
        ("rf", random_forest),
        ("lr", logistic_regression)
    ],
    voting="soft"
)


# ==========================================
# 4. Define 5-Fold Stratified CV
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 5. Metrics
# ==========================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


# ==========================================
# 6. Function for cross-validation
# ==========================================

def evaluate_model(name, model):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )

    print("\nFold Results:")

    for i in range(5):
        print(
            f"Fold {i + 1}: "
            f"Accuracy={results['test_accuracy'][i]:.4f}, "
            f"Precision={results['test_precision'][i]:.4f}, "
            f"Recall={results['test_recall'][i]:.4f}, "
            f"F1={results['test_f1'][i]:.4f}"
        )

    print("\nMean ± Standard Deviation:")

    print(
        f"Accuracy : "
        f"{results['test_accuracy'].mean():.4f} "
        f"± {results['test_accuracy'].std():.4f}"
    )

    print(
        f"Precision: "
        f"{results['test_precision'].mean():.4f} "
        f"± {results['test_precision'].std():.4f}"
    )

    print(
        f"Recall   : "
        f"{results['test_recall'].mean():.4f} "
        f"± {results['test_recall'].std():.4f}"
    )

    print(
        f"F1 Score : "
        f"{results['test_f1'].mean():.4f} "
        f"± {results['test_f1'].std():.4f}"
    )

    return results


# ==========================================
# 7. Evaluate all three models
# ==========================================

dt_results = evaluate_model(
    "DECISION TREE",
    decision_tree
)

rf_results = evaluate_model(
    "RANDOM FOREST",
    random_forest
)

ensemble_results = evaluate_model(
    "VOTING ENSEMBLE",
    voting_ensemble
)


# ==========================================
# 8. Create comparison table
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest",
        "Voting Ensemble"
    ],

    "Accuracy Mean": [
        dt_results["test_accuracy"].mean(),
        rf_results["test_accuracy"].mean(),
        ensemble_results["test_accuracy"].mean()
    ],

    "Accuracy Std": [
        dt_results["test_accuracy"].std(),
        rf_results["test_accuracy"].std(),
        ensemble_results["test_accuracy"].std()
    ],

    "Precision Mean": [
        dt_results["test_precision"].mean(),
        rf_results["test_precision"].mean(),
        ensemble_results["test_precision"].mean()
    ],

    "Precision Std": [
        dt_results["test_precision"].std(),
        rf_results["test_precision"].std(),
        ensemble_results["test_precision"].std()
    ],

    "Recall Mean": [
        dt_results["test_recall"].mean(),
        rf_results["test_recall"].mean(),
        ensemble_results["test_recall"].mean()
    ],

    "Recall Std": [
        dt_results["test_recall"].std(),
        rf_results["test_recall"].std(),
        ensemble_results["test_recall"].std()
    ],

    "F1 Mean": [
        dt_results["test_f1"].mean(),
        rf_results["test_f1"].mean(),
        ensemble_results["test_f1"].mean()
    ],

    "F1 Std": [
        dt_results["test_f1"].std(),
        rf_results["test_f1"].std(),
        ensemble_results["test_f1"].std()
    ]
})


# ==========================================
# 9. Display final comparison
# ==========================================

print("\n\n")
print("=" * 80)
print("5-FOLD CROSS-VALIDATION COMPARISON")
print("=" * 80)

print(comparison.to_string(index=False))


# ==========================================
# 10. Save results
# ==========================================

output_path = "results/url_cross_validation_results.csv"

comparison.to_csv(
    output_path,
    index=False
)

print("\nCross-validation results saved to:")
print(output_path)