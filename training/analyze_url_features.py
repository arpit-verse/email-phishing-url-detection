import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD DATASET
# ============================================================

INPUT_FILE = "dataset/url/url_features.csv"

df = pd.read_csv(INPUT_FILE)

print("\n========== DATASET LOADED ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. SELECT URL FEATURES
# ============================================================

excluded_columns = ["domain", "label"]

features = [
    column
    for column in df.columns
    if column not in excluded_columns
]

print("\n========== URL FEATURES ==========")
print("Number of URL features:", len(features))

for feature in features:
    print(feature)


# ============================================================
# 3. BASIC QUALITY CHECK
# ============================================================

print("\n========== MISSING VALUES ==========")

missing = df[features].isna().sum()

print(missing)


print("\n========== DATA TYPES ==========")

print(df[features].dtypes)


# ============================================================
# 4. FEATURE STATISTICS
# ============================================================

print("\n========== FEATURE STATISTICS ==========")

for feature in features:

    series = df[feature]

    print("\n----------------------------------------")
    print("Feature:", feature)
    print("----------------------------------------")

    print("Unique values:", series.nunique())
    print("Minimum:", series.min())
    print("Maximum:", series.max())
    print("Mean:", series.mean())
    print("Median:", series.median())
    print("Standard deviation:", series.std())
    print("Missing:", series.isna().sum())

    if len(series) > 0:
        print(
            "Zero percentage:",
            (series == 0).mean() * 100
        )


# ============================================================
# 5. QUANTILES
# ============================================================

print("\n========== FEATURE QUANTILES ==========")

for feature in features:

    print("\n----------------------------------------")
    print("Feature:", feature)
    print("----------------------------------------")

    print(
        df[feature].quantile(
            [0.00, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
        )
    )


# ============================================================
# 6. COMPARISON BY LABEL
# ============================================================

print("\n========== FEATURE COMPARISON BY LABEL ==========")

comparison = df.groupby("label")[features].agg(
    ["mean", "median"]
)

print(comparison.to_string())


# ============================================================
# 7. CORRELATION WITH LABEL
# ============================================================

print("\n========== CORRELATION WITH LABEL ==========")

correlation = (
    df[features + ["label"]]
    .corr(numeric_only=True)["label"]
    .drop("label")
    .sort_values(
        key=lambda x: abs(x),
        ascending=False
    )
)

print(correlation)


# ============================================================
# 8. STRONG CORRELATIONS BETWEEN URL FEATURES
# ============================================================

print("\n========== FEATURE-TO-FEATURE CORRELATION ==========")

feature_correlation = df[features].corr()

print(
    feature_correlation.to_string()
)


print("\n========== STRONG FEATURE CORRELATIONS ==========")

threshold = 0.90

found = False

for i in range(len(features)):

    for j in range(i + 1, len(features)):

        feature_a = features[i]
        feature_b = features[j]

        correlation_value = feature_correlation.loc[
            feature_a,
            feature_b
        ]

        if abs(correlation_value) >= threshold:

            print(
                f"{feature_a} <-> {feature_b} : "
                f"{correlation_value:.4f}"
            )

            found = True


if not found:
    print("No feature pairs exceeded the threshold.")


# ============================================================
# 9. SAVE ANALYSIS RESULTS
# ============================================================

correlation.to_csv(
    "dataset/url/url_feature_correlation.csv",
    header=["correlation"]
)

comparison.to_csv(
    "dataset/url/url_feature_class_comparison.csv"
)

feature_correlation.to_csv(
    "dataset/url/url_feature_feature_correlation.csv"
)


# ============================================================
# 10. COMPLETION
# ============================================================

print("\n========================================")
print("URL FEATURE ANALYSIS COMPLETE")
print("========================================")

print("\nGenerated files:")

print("1. dataset/url/url_feature_correlation.csv")
print("2. dataset/url/url_feature_class_comparison.csv")
print("3. dataset/url/url_feature_feature_correlation.csv")