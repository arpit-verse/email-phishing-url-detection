import pandas as pd

file_path = "dataset/url/urlset.csv"

df = pd.read_csv(
    file_path,
    encoding="latin1",
    on_bad_lines="warn"
)

columns_to_check = [
    "ranking",
    "mld_res",
    "mld.ps_res",
    "jaccard_ARrd",
    "jaccard_ARrem"
]

for column in columns_to_check:

    print("\n========================================")
    print("COLUMN:", column)
    print("========================================")

    numeric_values = pd.to_numeric(df[column], errors="coerce")

    non_numeric = df.loc[
        numeric_values.isna() & df[column].notna(),
        column
    ]

    print("Number of non-numeric values:", len(non_numeric))

    print("\nExamples:")
    print(non_numeric.head(20).to_string(index=False))