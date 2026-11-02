import re
import sys
from pathlib import Path
from urllib.parse import urlparse
import joblib
import pandas as pd

# ==========================================
# Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "url_random_forest.pkl"


# ==========================================
# Feature order (25 features)
# ==========================================

FEATURES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "digit_count",
    "letter_count",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "slash_count",
    "question_mark_count",
    "equals_count",
    "ampersand_count",
    "percent_count",
    "at_count",
    "subdomain_count",
    "path_depth",
    "query_parameter_count",
    "long_digit_sequence",
    "suspicious_keyword_count",
    "domain_digit_count",
    "domain_hyphen_count",
    "domain_dot_count",
    "has_php",
    "special_character_count"
]


# ==========================================
# Feature extraction (exact match with training pipeline)
# ==========================================

def extract_url_features(url):
    """
    Extracts the exact 25 lexical features used by models/url_random_forest.pkl.
    Matches extract_features() in training/extract_url_features.py.
    """
    url = str(url).strip()

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

    # Basic length features
    url_length = len(url)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    # Character counts
    digit_count = sum(c.isdigit() for c in url)
    letter_count = sum(c.isalpha() for c in url)
    dot_count = url.count(".")
    hyphen_count = url.count("-")
    underscore_count = url.count("_")
    slash_count = url.count("/")
    question_mark_count = url.count("?")
    equals_count = url.count("=")
    ampersand_count = url.count("&")
    percent_count = url.count("%")
    at_count = url.count("@")

    # Structural features
    if hostname:
        subdomain_parts = hostname.split(".")
        subdomain_count = max(len(subdomain_parts) - 2, 0)
    else:
        subdomain_count = 0

    path_depth = len([part for part in path.split("/") if part])

    if query:
        query_parameter_count = len(query.split("&"))
    else:
        query_parameter_count = 0

    # Long digit sequence (>= 5 consecutive digits)
    long_digit_sequence = int(bool(re.search(r"\d{5,}", url)))

    # Suspicious keywords (17 keywords matching extract_url_features.py)
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

    # Domain characteristics
    domain_digit_count = sum(c.isdigit() for c in hostname)
    domain_hyphen_count = hostname.count("-")
    domain_dot_count = hostname.count(".")

    # File indicator
    has_php = int(".php" in url_lower)

    # Special characters: all non-alphanumeric characters in URL
    special_character_count = sum(not c.isalnum() for c in url)

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
        "long_digit_sequence": long_digit_sequence,
        "suspicious_keyword_count": suspicious_keyword_count,
        "domain_digit_count": domain_digit_count,
        "domain_hyphen_count": domain_hyphen_count,
        "domain_dot_count": domain_dot_count,
        "has_php": has_php,
        "special_character_count": special_character_count
    }


# ==========================================
# Load model
# ==========================================

def load_model(model_path=None):
    """
    Safely loads the trained URL model resolving path relative to project root.
    """
    if model_path is None:
        target_path = DEFAULT_MODEL_PATH
    else:
        target_path = Path(model_path)
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Model file not found: {target_path}")

    return joblib.load(target_path)


# ==========================================
# Prediction function (import-safe)
# ==========================================

def predict(url: str, model=None):
    """
    Performs inference for a given URL string.
    Returns a dictionary containing predictions, probabilities, and extracted features.
    """
    if model is None:
        model = load_model()

    feature_dict = extract_url_features(url)

    X = pd.DataFrame(
        [[feature_dict[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    return {
        "url": url,
        "prediction": prediction,
        "label": "PHISHING" if prediction == 1 else "LEGITIMATE",
        "phishing_probability": float(probabilities[1]),
        "legitimate_probability": float(probabilities[0]),
        "features": feature_dict
    }


# ==========================================
# Safe display
# ==========================================

def print_result(result: dict):
    """
    Displays analysis results using console-safe characters (no charmap Unicode crashes).
    """
    url = result["url"]
    prediction = result["prediction"]
    phishing_probability = result["phishing_probability"]
    legitimate_probability = result["legitimate_probability"]

    print("\n========== URL ANALYSIS ==========")
    print("URL:")
    print(url)

    print("\nPhishing probability:  ",
          f"{phishing_probability * 100:.2f}%")
    print("Legitimate probability:",
          f"{legitimate_probability * 100:.2f}%")

    print("\n========== RESULT ==========")
    if prediction == 1:
        print("[!] PHISHING URL")
    else:
        print("[+] LEGITIMATE URL")

    print("\n========== EXTRACTED FEATURES ==========")
    for feature in FEATURES:
        print(f"{feature}: {result['features'][feature]}")


# ==========================================
# Interactive CLI Entry Point
# ==========================================

def main():
    url = input("\nEnter URL to check: ").strip()

    if not url:
        print("No URL entered.")
        return

    model = load_model()
    result = predict(url, model=model)
    print_result(result)


if __name__ == "__main__":
    main()
