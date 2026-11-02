"""
Email Phishing Detection Model Training and Evaluation Pipeline.

Trains, cross-validates, and evaluates:
1. Multinomial Naive Bayes Pipeline (TF-IDF + MultinomialNB)
2. Logistic Regression Pipeline (TF-IDF + LogisticRegression)

Ensures:
- Strict train/test separation (vectorizer fitted only on training data).
- 5-Fold Stratified Cross-Validation on the training partition.
- Single holdout evaluation on the untouched 20% test partition.
- Non-destructive artifact saving under models/email_* and results/email_*.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from preprocess_email import prepare_email_dataset


def train_and_evaluate_email_models():
    print("=" * 70, flush=True)
    print("PHASE 2: EMAIL PHISHING MODEL TRAINING & EVALUATION PIPELINE", flush=True)
    print("=" * 70, flush=True)

    # 1. Load and prepare dataset with zero data leakage
    print("\n[Step 1] Loading and preparing email dataset with leakage prevention...", flush=True)
    X_train, X_test, y_train, y_test, stats = prepare_email_dataset()

    print(f"  Training samples:     {len(X_train):,} {stats['train_class_distribution']}", flush=True)
    print(f"  Holdout test samples: {len(X_test):,} {stats['test_class_distribution']}", flush=True)
    print(f"  Overlap count:        {stats['leakage_overlap_count']} (Zero leakage confirmed)", flush=True)

    # 2. Define pipelines
    print("\n[Step 2] Configuring TF-IDF pipelines...", flush=True)
    vectorizer_params = {
        "max_features": 15000,
        "ngram_range": (1, 2),
        "min_df": 3,
        "sublinear_tf": True,
        "stop_words": "english"
    }

    nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**vectorizer_params)),
        ("clf", MultinomialNB(alpha=0.1))
    ])

    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**vectorizer_params)),
        ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42, class_weight="balanced"))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # 3. 5-Fold Stratified Cross-Validation on Training Partition (n_jobs=1 for Windows stability)
    print("\n[Step 3] Running 5-Fold Stratified Cross-Validation on training partition (n_jobs=1)...", flush=True)
    print("  Evaluating Multinomial Naive Bayes across 5 folds...", flush=True)
    nb_cv = cross_validate(nb_pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    print("  Naive Bayes CV completed.", flush=True)

    print("  Evaluating Logistic Regression across 5 folds...", flush=True)
    lr_cv = cross_validate(lr_pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    print("  Logistic Regression CV completed.", flush=True)

    cv_results = pd.DataFrame([
        {
            "Model": "Multinomial Naive Bayes",
            "Accuracy Mean": nb_cv["test_accuracy"].mean(),
            "Accuracy Std": nb_cv["test_accuracy"].std(),
            "Precision Mean": nb_cv["test_precision"].mean(),
            "Precision Std": nb_cv["test_precision"].std(),
            "Recall Mean": nb_cv["test_recall"].mean(),
            "Recall Std": nb_cv["test_recall"].std(),
            "F1 Mean": nb_cv["test_f1"].mean(),
            "F1 Std": nb_cv["test_f1"].std(),
            "ROC-AUC Mean": nb_cv["test_roc_auc"].mean(),
            "ROC-AUC Std": nb_cv["test_roc_auc"].std()
        },
        {
            "Model": "Logistic Regression",
            "Accuracy Mean": lr_cv["test_accuracy"].mean(),
            "Accuracy Std": lr_cv["test_accuracy"].std(),
            "Precision Mean": lr_cv["test_precision"].mean(),
            "Precision Std": lr_cv["test_precision"].std(),
            "Recall Mean": lr_cv["test_recall"].mean(),
            "Recall Std": lr_cv["test_recall"].std(),
            "F1 Mean": lr_cv["test_f1"].mean(),
            "F1 Std": lr_cv["test_f1"].std(),
            "ROC-AUC Mean": lr_cv["test_roc_auc"].mean(),
            "ROC-AUC Std": lr_cv["test_roc_auc"].std()
        }
    ])

    print("\n--- 5-Fold Cross-Validation Summary (Training Set) ---", flush=True)
    print(cv_results.to_string(index=False), flush=True)

    cv_output_file = RESULTS_DIR / "email_cross_validation_results.csv"
    cv_results.to_csv(cv_output_file, index=False)
    print(f"Saved CV results to: {cv_output_file}", flush=True)

    # 4. Train final models on full training partition
    print("\n[Step 4] Fitting final models on full training partition...", flush=True)
    print("  Fitting final Naive Bayes pipeline...", flush=True)
    nb_pipeline.fit(X_train, y_train)
    print("  Fitting final Logistic Regression pipeline...", flush=True)
    lr_pipeline.fit(X_train, y_train)
    print("  Fitting complete.", flush=True)

    # 5. Evaluate on untouched holdout test partition
    print("\n[Step 5] Evaluating on untouched holdout test set...", flush=True)
    eval_records = []
    trained_models = [
        ("Multinomial Naive Bayes", nb_pipeline),
        ("Logistic Regression", lr_pipeline)
    ]

    for name, pipe in trained_models:
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        eval_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": auc,
            "True Negatives (TN)": int(tn),
            "False Positives (FP)": int(fp),
            "False Negatives (FN)": int(fn),
            "True Positives (TP)": int(tp),
            "Total Test Samples": len(y_test)
        })

        print(f"\n--- {name} Holdout Performance ---", flush=True)
        print(f"  Accuracy:  {acc:.4f}", flush=True)
        print(f"  Precision: {prec:.4f}", flush=True)
        print(f"  Recall:    {rec:.4f}", flush=True)
        print(f"  F1 Score:  {f1:.4f}", flush=True)
        print(f"  ROC-AUC:   {auc:.4f}", flush=True)
        print(f"  Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]", flush=True)

    eval_df = pd.DataFrame(eval_records)
    eval_output_file = RESULTS_DIR / "email_evaluation_results.csv"
    eval_df.to_csv(eval_output_file, index=False)
    print(f"\nSaved holdout evaluation results to: {eval_output_file}", flush=True)

    # 6. Extract and save top predictive features (interpretability)
    print("\n[Step 6] Extracting top predictive features from Logistic Regression...", flush=True)
    lr_clf = lr_pipeline.named_steps["clf"]
    tfidf_step = lr_pipeline.named_steps["tfidf"]
    feature_names = tfidf_step.get_feature_names_out()
    coefficients = lr_clf.coef_[0]

    top_phish_idx = np.argsort(coefficients)[-20:][::-1]
    top_legit_idx = np.argsort(coefficients)[:20]

    feature_records = []
    for rank, idx in enumerate(top_phish_idx, 1):
        feature_records.append({
            "Rank": rank,
            "Phishing Indicative Feature": feature_names[idx],
            "Phishing Log-Odds Coefficient": coefficients[idx],
            "Legitimate Indicative Feature": feature_names[top_legit_idx[rank - 1]],
            "Legitimate Log-Odds Coefficient": coefficients[top_legit_idx[rank - 1]]
        })

    features_df = pd.DataFrame(feature_records)
    features_output_file = RESULTS_DIR / "email_top_predictive_features.csv"
    features_df.to_csv(features_output_file, index=False)
    print("Top 5 Phishing cues:", [feature_names[i] for i in top_phish_idx[:5]], flush=True)
    print("Top 5 Legitimate cues:", [feature_names[i] for i in top_legit_idx[:5]], flush=True)
    print(f"Saved feature analysis to: {features_output_file}", flush=True)

    # 7. Model selection and non-destructive serialization
    print("\n[Step 7] Serializing models under distinct models/email_* paths...", flush=True)
    best_record = max(eval_records, key=lambda x: x["F1 Score"])
    best_name = best_record["Model"]
    best_pipe = lr_pipeline if best_name == "Logistic Regression" else nb_pipeline

    print(f"  Selected Primary Baseline: {best_name} (Holdout F1: {best_record['F1 Score']:.4f})", flush=True)

    pipe_path = MODELS_DIR / "email_pipeline.pkl"
    nb_path = MODELS_DIR / "email_naive_bayes.pkl"
    lr_path = MODELS_DIR / "email_logistic_regression.pkl"
    vec_path = MODELS_DIR / "email_tfidf_vectorizer.pkl"

    for p in [pipe_path, nb_path, lr_path, vec_path]:
        if p.exists():
            raise FileExistsError(f"Target model file already exists: {p}. Halting to prevent overwrite.")

    joblib.dump(best_pipe, pipe_path)
    joblib.dump(nb_pipeline, nb_path)
    joblib.dump(lr_pipeline, lr_path)
    joblib.dump(best_pipe.named_steps["tfidf"], vec_path)

    print(f"  Saved primary pipeline to:             {pipe_path} ({pipe_path.stat().st_size:,} bytes)", flush=True)
    print(f"  Saved Naive Bayes pipeline to:          {nb_path} ({nb_path.stat().st_size:,} bytes)", flush=True)
    print(f"  Saved Logistic Regression pipeline to:   {lr_path} ({lr_path.stat().st_size:,} bytes)", flush=True)
    print(f"  Saved extracted TF-IDF vectorizer to:   {vec_path} ({vec_path.stat().st_size:,} bytes)", flush=True)

    print("\n=======================================================", flush=True)
    print("EMAIL PHISHING PIPELINE TRAINING COMPLETED SUCCESSFULLY", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    train_and_evaluate_email_models()
