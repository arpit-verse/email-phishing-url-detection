import pandas as pd

# Location of our URL dataset
file_path = "dataset/url/urlset.csv"

# Load the dataset
df = pd.read_csv(
    file_path,
    encoding="latin1",
    on_bad_lines="warn"
)

print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Duplicate rows:", df.duplicated().sum())

print("\n========== LABEL DISTRIBUTION ==========")
print(df["label"].value_counts(dropna=False))

print("\n========== LABEL PERCENTAGE ==========")
print(df["label"].value_counts(normalize=True, dropna=False) * 100)