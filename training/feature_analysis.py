import pandas as pd
import numpy as np

# ==========================================
# LOAD DATASET
# ==========================================

file_path = "dataset/url/urlset_cleaned.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED ==========")
print("Shape:", df.shape)


# ==========================================
# DEFINE FEATURES
# ==========================================

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


# ==========================================
# CONVERT FEATURES TO NUMERIC
# ==========================================

for feature in features:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")


# ==========================================
# BASIC INFORMATION
# ==========================================

print("\n========== FEATURE DATA TYPES ==========")

print(df[features].dtypes)


# ==========================================
# MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")

print(df[features].isnull().sum())


# ==========================================
# ANALYZE EACH FEATURE
# ==========================================

print("\n========== FEATURE STATISTICS ==========")

for feature in features:

    print("\n----------------------------------------")
    print("Feature:", feature)
    print("----------------------------------------")

    print("Minimum:", df[feature].min())
    print("Maximum:", df[feature].max())
    print("Mean:", df[feature].mean())
    print("Median:", df[feature].median())
    print("Standard deviation:", df[feature].std())


# ==========================================
# LEGITIMATE VS PHISHING
# ==========================================

print("\n\n========== FEATURE COMPARISON BY LABEL ==========")

comparison = df.groupby("label")[features].agg(
    ["mean", "median"]
)

print(comparison)


# ==========================================
# CORRELATION WITH LABEL
# ==========================================

print("\n\n========== CORRELATION WITH LABEL ==========")

correlations = df[features + ["label"]].corr()["label"]

correlations = correlations.drop("label")

correlations = correlations.reindex(
    correlations.abs().sort_values(ascending=False).index
)

print(correlations)


# ==========================================
# SAVE CORRELATION RESULTS
# ==========================================

correlation_df = correlations.reset_index()

correlation_df.columns = [
    "feature",
    "correlation_with_label"
]

correlation_df["absolute_correlation"] = (
    correlation_df["correlation_with_label"].abs()
)

correlation_df.to_csv(
    "dataset/url/feature_correlation.csv",
    index=False
)


# ==========================================
# SAVE CLASS-WISE MEANS
# ==========================================

class_means = df.groupby("label")[features].mean().T

class_means.columns = [
    "legitimate_mean",
    "phishing_mean"
]

class_means["difference"] = (
    class_means["phishing_mean"]
    - class_means["legitimate_mean"]
)

class_means.to_csv(
    "dataset/url/feature_class_comparison.csv"
)


# ==========================================
# DOMAIN / URL PRESERVATION CHECK
# ==========================================

print("\n\n========== DOMAIN / URL CHECK ==========")

print("Domain column exists:", "domain" in df.columns)

if "domain" in df.columns:

    print("Number of URLs:", df["domain"].notna().sum())

    print("\nExample URLs:")

    print(df["domain"].head(10).to_string(index=False))


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n\n========================================")
print("FEATURE ANALYSIS COMPLETE")
print("========================================")

print("\nGenerated files:")

print("1. dataset/url/feature_correlation.csv")
print("2. dataset/url/feature_class_comparison.csv")

print("\nThe original domain/URL column has NOT been deleted.")