import pandas as pd

file_path = "dataset/url/urlset_cleaned.csv"

df = pd.read_csv(file_path)

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

print("\n========== FEATURE INFORMATION ==========")

for feature in features:

    print("\n----------------------------------------")
    print("Feature:", feature)
    print("----------------------------------------")

    print("Minimum:", df[feature].min())
    print("Maximum:", df[feature].max())
    print("Mean:", df[feature].mean())
    print("Median:", df[feature].median())
    print("Missing:", df[feature].isnull().sum())