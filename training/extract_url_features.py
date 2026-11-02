import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse


# ============================================================
# 1. LOAD DATASET
# ============================================================

INPUT_FILE = "dataset/url/urlset_cleaned.csv"
OUTPUT_FILE = "dataset/url/url_features.csv"

df = pd.read_csv(INPUT_FILE)

print("\n========== DATASET LOADED ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. URL FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(url):

    # Convert safely to string
    url = str(url).strip()

    # Add a scheme temporarily if one is not present.
    # This allows urlparse() to correctly identify the domain.
    parse_url = url

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", parse_url):
        parse_url = "http://" + parse_url

    try:
        parsed = urlparse(parse_url)
    except Exception:
        parsed = urlparse("http://")

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # ========================================================
    # BASIC URL FEATURES
    # ========================================================

    url_length = len(url)

    hostname_length = len(hostname)

    path_length = len(path)

    query_length = len(query)

    digit_count = sum(
        character.isdigit()
        for character in url
    )

    letter_count = sum(
        character.isalpha()
        for character in url
    )

    dot_count = url.count(".")

    hyphen_count = url.count("-")

    underscore_count = url.count("_")

    slash_count = url.count("/")

    question_mark_count = url.count("?")

    equals_count = url.count("=")

    ampersand_count = url.count("&")

    percent_count = url.count("%")

    at_count = url.count("@")

    # ========================================================
    # STRUCTURAL FEATURES
    # ========================================================

    if hostname:
        subdomain_parts = hostname.split(".")
        subdomain_count = max(
            len(subdomain_parts) - 2,
            0
        )
    else:
        subdomain_count = 0

    path_depth = len([
        part
        for part in path.split("/")
        if part
    ])

    # Number of query parameters
    if query:
        query_parameter_count = len(
            query.split("&")
        )
    else:
        query_parameter_count = 0

    # ========================================================
    # SECURITY / SUSPICIOUS STRUCTURE FEATURES
    # ========================================================

    has_https = int(
        parsed.scheme.lower() == "https"
    )

    has_at_symbol = int(
        "@" in url
    )

    has_ip_address = 0

    # Detect IPv4 address in hostname
    if hostname:
        ip_pattern = (
            r"^(?:\d{1,3}\.){3}\d{1,3}$"
        )

        if re.match(ip_pattern, hostname):
            has_ip_address = 1

    # URL contains an unusually long numerical sequence
    long_digit_sequence = int(
        bool(re.search(r"\d{5,}", url))
    )

    # ========================================================
    # SUSPICIOUS KEYWORD FEATURES
    # ========================================================

    suspicious_words = [
        "login",
        "signin",
        "verify",
        "verification",
        "secure",
        "account",
        "update",
        "confirm",
        "password",
        "credential",
        "bank",
        "paypal",
        "payment",
        "wallet",
        "billing",
        "invoice",
        "authentication"
    ]

    url_lower = url.lower()

    suspicious_keyword_count = sum(
        word in url_lower
        for word in suspicious_words
    )

    # ========================================================
    # DOMAIN CHARACTERISTICS
    # ========================================================

    domain_digit_count = sum(
        character.isdigit()
        for character in hostname
    )

    domain_hyphen_count = hostname.count("-")

    domain_dot_count = hostname.count(".")

    # ========================================================
    # PATH / FILE FEATURES
    # ========================================================

    has_php = int(
        ".php" in url_lower
    )

    has_html = int(
        ".html" in url_lower
    )

    has_exe = int(
        ".exe" in url_lower
    )

    # ========================================================
    # SPECIAL CHARACTER COUNT
    # ========================================================

    special_character_count = sum(
        not character.isalnum()
        for character in url
    )

    # ========================================================
    # RETURN FEATURES
    # ========================================================

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "query_length": query_length,

        "digit_count": digit_count,
        "letter_count": letter_count,

        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "underscore_count": underscore_count,
        "slash_count": slash_count,

        "question_mark_count": question_mark_count,
        "equals_count": equals_count,
        "ampersand_count": ampersand_count,
        "percent_count": percent_count,
        "at_count": at_count,

        "subdomain_count": subdomain_count,
        "path_depth": path_depth,
        "query_parameter_count": query_parameter_count,

        "has_https": has_https,
        "has_ip_address": has_ip_address,
        "has_at_symbol": has_at_symbol,
        "long_digit_sequence": long_digit_sequence,

        "suspicious_keyword_count":
            suspicious_keyword_count,

        "domain_digit_count":
            domain_digit_count,

        "domain_hyphen_count":
            domain_hyphen_count,

        "domain_dot_count":
            domain_dot_count,

        "has_php": has_php,
        "has_html": has_html,
        "has_exe": has_exe,

        "special_character_count":
            special_character_count
    }


# ============================================================
# 3. EXTRACT FEATURES
# ============================================================

print("\n========== EXTRACTING URL FEATURES ==========")

feature_records = []

for url in df["domain"]:

    feature_records.append(
        extract_features(url)
    )


feature_df = pd.DataFrame(
    feature_records
)


# ============================================================
# 4. ADD ORIGINAL URL AND LABEL
# ============================================================

feature_df.insert(
    0,
    "domain",
    df["domain"].values
)

feature_df["label"] = df["label"].values


# ============================================================
# 5. CHECK RESULT
# ============================================================

print("\n========== GENERATED FEATURES ==========")

print(
    "Feature dataset shape:",
    feature_df.shape
)

print("\nFeature names:")

for column in feature_df.columns:
    print(column)


# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

print(
    feature_df.isna().sum()
)


# ============================================================
# 7. SAMPLE
# ============================================================

print("\n========== SAMPLE ==========")

print(
    feature_df.head(5).to_string()
)


# ============================================================
# 8. SAVE
# ============================================================

feature_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("URL FEATURE EXTRACTION COMPLETE")
print("========================================")

print("\nSaved to:")
print(OUTPUT_FILE)