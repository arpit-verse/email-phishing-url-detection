import pandas as pd
import numpy as np

# ============================================================
# LOAD DATASET
# ============================================================

file_path = "dataset/url/urlset_cleaned.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED ==========")
print("Shape:", df.shape)


# ============================================================
# SELECT NUMERICAL FEATURES
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

X = df[features].copy()


# ============================================================
# FEATURE QUALITY
# ============================================================

results = []

print("\n========== FEATURE QUALITY ==========")

for feature in features:

    series = X[feature].dropna()

    result = {
        "feature": feature,
        "missing": X[feature].isna().sum(),
        "unique_values": series.nunique(),
        "minimum": series.min(),
        "maximum": series.max(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "skewness": series.skew(),
        "zero_percentage": (series == 0).mean() * 100
    }

    results.append(result)

    print("\n----------------------------------------")
    print("Feature:", feature)
    print("----------------------------------------")
    print("Missing:", result["missing"])
    print("Unique values:", result["unique_values"])
    print("Minimum:", result["minimum"])
    print("Maximum:", result["maximum"])
    print("Mean:", result["mean"])
    print("Median:", result["median"])
    print("Standard deviation:", result["std"])
    print("Skewness:", result["skewness"])
    print("Zero percentage:", result["zero_percentage"])


# ============================================================
# QUANTILES
# ============================================================

print("\n\n========== FEATURE QUANTILES ==========")

for feature in features:

    print("\n----------------------------------------")
    print("Feature:", feature)
    print("----------------------------------------")

    print(
        X[feature].quantile(
            [0.00, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
        )
    )


# ============================================================
# FEATURE CORRELATION
# ============================================================

print("\n\n========== FEATURE-TO-FEATURE CORRELATION ==========")

correlation_matrix = X.corr()

print(correlation_matrix.round(3))


# ============================================================
# FIND STRONG FEATURE CORRELATIONS
# ============================================================

print("\n\n========== STRONG FEATURE CORRELATIONS ==========")

found = False

for i in range(len(features)):
    for j in range(i + 1, len(features)):

        correlation = correlation_matrix.iloc[i, j]

        if abs(correlation) >= 0.70:

            print(
                f"{features[i]} <-> {features[j]} : "
                f"{correlation:.4f}"
            )

            found = True

if not found:
    print("No feature pairs have absolute correlation >= 0.70.")


# ============================================================
# SAVE RESULTS
# ============================================================

quality_df = pd.DataFrame(results)

output_file = "dataset/url/feature_quality.csv"

quality_df.to_csv(output_file, index=False)

print("\n\n========== ANALYSIS COMPLETE ==========")
print("Feature quality report saved to:")
print(output_file)