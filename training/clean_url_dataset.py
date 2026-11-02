import pandas as pd

# ============================================
# 1. LOAD DATASET
# ============================================

file_path = "dataset/url/urlset.csv"

df = pd.read_csv(
    file_path,
    encoding="latin1",
    on_bad_lines="warn"
)

print("Original shape:", df.shape)


# ============================================
# 2. DEFINE FEATURES AND TARGET
# ============================================

numeric_features = [
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

target = "label"


# ============================================
# 3. CONVERT NUMERIC FEATURES
# ============================================

for column in numeric_features:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================
# 4. CONVERT LABEL
# ============================================

df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)


# ============================================
# 5. REMOVE ROWS WITHOUT A LABEL
# ============================================

before = len(df)

df = df.dropna(subset=[target])

after = len(df)

print("\nRows removed because label was missing:", before - after)


# ============================================
# ============================================
# 6. CHECK DUPLICATES
# ============================================

duplicate_count = df.duplicated().sum()

print("\n========== DUPLICATE ANALYSIS ==========")
print("Duplicate rows found:", duplicate_count)

# ============================================
# 7. SHOW REMAINING MISSING VALUES
# ============================================

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df[numeric_features].isnull().sum())


# ============================================
# 8. SHOW FINAL INFORMATION
# ============================================

print("\n========== FINAL DATASET ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== LABEL DISTRIBUTION ==========")
print(df[target].value_counts())


# ============================================
# 9. SAVE CLEANED DATASET
# ============================================

output_path = "dataset/url/urlset_cleaned.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nCleaned dataset saved to:")
print(output_path)