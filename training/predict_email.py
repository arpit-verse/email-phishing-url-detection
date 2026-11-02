"""
Live Email Phishing Detection Inference Script.

Features:
- Dynamic model path resolution (works from project root or training/).
- Clean importability without stdin blocking.
- Console-safe formatting under Windows and standard environments.
- Normalization consistent with the training pipeline.
"""

import sys
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "email_pipeline.pkl"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from preprocess_email import normalize_email_text


def load_model(model_path=None):
    """
    Safely loads the trained email model, resolving paths relative to project root.
    """
    if model_path is None:
        target_path = DEFAULT_MODEL_PATH
    else:
        target_path = Path(model_path)
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Email model file not found: {target_path}")

    return joblib.load(target_path)


def predict(email_text: str, model=None) -> dict:
    """
    Performs inference for an input email string.

    Returns:
        dict:
            - text: Original email text
            - normalized_text: Cleaned text processed by model
            - prediction: 1 (Phishing) or 0 (Legitimate)
            - label: "PHISHING" or "LEGITIMATE"
            - phishing_probability: float
            - legitimate_probability: float
            - detected_tokens: list of recognized tokens from model vocabulary
    """
    normalized = normalize_email_text(email_text)

    # Empty or non-informative input must not be presented as legitimate.
    if not normalized:
        raise ValueError("Email text must contain usable content after normalization.")

    if model is None:
        model = load_model()

    probabilities = model.predict_proba([normalized])[0]
    prediction = int(model.predict([normalized])[0])

    phishing_prob = float(probabilities[1])
    legit_prob = float(probabilities[0])

    # Extract recognized vocabulary tokens for interpretability
    try:
        tfidf = model.named_steps.get("tfidf")
        vocab = tfidf.vocabulary_ if tfidf else {}
        words = normalized.split()
        detected_tokens = sorted(list(set(w for w in words if w in vocab)))[:15]
    except Exception:
        detected_tokens = []

    return {
        "text": email_text,
        "normalized_text": normalized,
        "prediction": prediction,
        "label": "PHISHING" if prediction == 1 else "LEGITIMATE",
        "phishing_probability": phishing_prob,
        "legitimate_probability": legit_prob,
        "detected_tokens": detected_tokens
    }


def print_result(result: dict):
    """
    Displays email analysis results cleanly without charmap encoding crashes.
    """
    print("\n========== EMAIL ANALYSIS ==========")
    preview = result["text"].strip().replace("\n", " ")
    if len(preview) > 120:
        preview = preview[:117] + "..."
    print(f"Input Preview: {preview}")

    print(f"\nPhishing probability:   {result['phishing_probability'] * 100:.2f}%")
    print(f"Legitimate probability: {result['legitimate_probability'] * 100:.2f}%")

    print("\n========== RESULT ==========")
    if result["prediction"] == 1:
        print("[!] PHISHING / SUSPICIOUS EMAIL")
    else:
        print("[+] LEGITIMATE EMAIL")

    if result.get("detected_tokens"):
        print("\nRecognized Key Vocabulary Tokens:")
        print(", ".join(result["detected_tokens"]))


def main():
    print("Email Phishing Predictor (CLI)")
    print("Enter or paste email text (press Ctrl+Z or Ctrl+D on empty line to submit, or enter one line):")
    try:
        lines = []
        while True:
            line = input()
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        pass

    email_text = "\n".join(lines).strip()
    if not email_text:
        print("No email text provided.")
        return

    model = load_model()
    try:
        result = predict(email_text, model=model)
    except ValueError:
        print("Invalid email input. Please provide text with usable content.")
        return

    print_result(result)


if __name__ == "__main__":
    main()
