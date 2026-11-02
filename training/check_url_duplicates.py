import pandas as pd

# ==========================================
# LOAD DATASET
# ==========================================

file_path = "dataset/url/urlset_cleaned.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ==========================================
# EXACT DUPLICATE ROWS
# ==========================================

print("\n========== EXACT DUPLICATE ROWS ==========")

exact_duplicates = df.duplicated().sum()

print("Exact duplicate rows:", exact_duplicates)


# ==========================================
# DUPLICATE URLs
# ==========================================

print("\n========== DUPLICATE URLs ==========")

duplicate_urls = df[df.duplicated(
    subset=["domain"],
    keep=False
)].sort_values("domain")

print("Rows belonging to duplicated URLs:",
      len(duplicate_urls))

print("Number of unique duplicated URLs:",
      duplicate_urls["domain"].nunique())


# ==========================================
# SHOW DUPLICATED URLS
# ==========================================

if len(duplicate_urls) > 0:

    print("\nDuplicated URL records:")

    print(
        duplicate_urls[
            ["domain", "label"]
        ].to_string(index=False)
    )


# ==========================================
# CHECK WHETHER DUPLICATE URLS HAVE
# DIFFERENT LABELS
# ==========================================

print("\n========== LABEL CONSISTENCY ==========")

if len(duplicate_urls) > 0:

    label_counts = (
        duplicate_urls
        .groupby("domain")["label"]
        .nunique()
    )

    conflicting_urls = label_counts[
        label_counts > 1
    ]

    print(
        "URLs with conflicting labels:",
        len(conflicting_urls)
    )

    if len(conflicting_urls) > 0:

        print("\nConflicting URLs:")

        print(
            duplicate_urls[
                duplicate_urls["domain"].isin(
                    conflicting_urls.index
                )
            ][["domain", "label"]]
            .to_string(index=False)
        )

else:

    print("No duplicated URLs found.")


# ==========================================
# FINAL
# ==========================================

print("\n========================================")
print("DUPLICATE ANALYSIS COMPLETE")
print("========================================")