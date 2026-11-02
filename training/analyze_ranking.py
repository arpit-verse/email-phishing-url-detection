import pandas as pd

file_path = "dataset/url/urlset_cleaned.csv"

df = pd.read_csv(file_path)

df["ranking"] = pd.to_numeric(
    df["ranking"],
    errors="coerce"
)

print("\n========== RANKING ANALYSIS ==========")

print("\nRanking value counts:")
print(
    df["ranking"]
    .value_counts()
    .head(20)
)


print("\n========== MAXIMUM RANKING ==========")

max_ranking = df["ranking"].max()

max_rows = df[df["ranking"] == max_ranking]

print("Maximum ranking:", max_ranking)
print("Number of rows with maximum ranking:", len(max_rows))

print("\nLabel distribution among maximum-ranking rows:")

print(
    max_rows["label"]
    .value_counts()
)


print("\nLabel percentage among maximum-ranking rows:")

print(
    max_rows["label"]
    .value_counts(normalize=True) * 100
)


print("\n========== LOW RANKING ==========")

low_rows = df[
    df["ranking"] < 10000000
]

print(
    "Rows with ranking below 10,000,000:",
    len(low_rows)
)

print("\nLabel distribution:")

print(
    low_rows["label"]
    .value_counts()
)

print("\nLabel percentage:")

print(
    low_rows["label"]
    .value_counts(normalize=True) * 100
)


print("\n========== SAMPLE MAX-RANKING URLs ==========")

print(
    max_rows[["domain", "ranking", "label"]]
    .head(10)
    .to_string(index=False)
)


print("\n========== ANALYSIS COMPLETE ==========")