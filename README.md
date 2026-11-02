# Sentinel — Email Phishing and URL Detection

Sentinel is a student machine-learning project with a React security-console frontend and a FastAPI backend for email-text and URL analysis. This evaluator snapshot contains source code and selected aggregate evidence. It intentionally omits raw datasets, serialized models, row-level audit outputs, and Git history.

The backend downloads the two fixed public model artifacts at startup and verifies their SHA-256 hashes before loading them. It does not fetch or visit submitted URLs. See the model manifest and deployment guide for exact revisions and setup.

## What is implemented

- Email text normalization followed by a TF-IDF and Logistic Regression pipeline. The training code also evaluates Multinomial Naive Bayes.
- URL classification with a Random Forest over 25 lexical and structural URL features. The model does not inspect page contents or reputation feeds.
- A FastAPI API with health, email-analysis, and URL-analysis endpoints.
- A React/Vite interface. Overview activity and sample history remain visibly marked demo data; successful analysis requests use API predictions.

## Evaluation at a glance

Recorded artifacts report 98.47% accuracy for the selected email Logistic Regression holdout and 93.37% accuracy for the URL row-wise holdout. These are existing results, not newly run results from this snapshot. The email holdout was used to choose the model by F1, and the URL split contains substantial domain overlap. Neither figure establishes performance on independent future or unseen-source data. Details and confusion matrices are in [Evaluation](docs/evaluation.md).

## Run locally

The API needs Python dependencies and network access to the public Hugging Face model files when the model files are absent locally.

From the repository root:

    python -m pip install -r api/requirements.txt
    python -m uvicorn api.main:app --reload

In a second terminal:

    cd frontend
    npm ci
    npm run dev

The local frontend fallback API URL is http://127.0.0.1:8000. For another API host, set VITE_API_BASE_URL in the frontend build environment. Set CORS_ALLOWED_ORIGINS on the API to an explicit comma-separated list of browser origins. Do not put credentials in Vite variables.

## Read before relying on results

Raw datasets and model binaries are not included. Exact dataset citations, source versions, and redistribution terms are not sufficiently recorded in the source project, so the aggregate results cannot be independently reproduced from this snapshot alone. The project also has no repository license grant. See [dataset provenance](docs/dataset-provenance.md), [privacy and limitations](docs/privacy-and-limitations.md), and [reproducing results](docs/reproducing-results.md).

## Documentation

- [Architecture](docs/architecture.md)
- [Dataset provenance and data requirements](docs/dataset-provenance.md)
- [Model cards](docs/model-cards.md)
- [Model artifact manifest](docs/model-artifacts.md)
- [Evaluation methodology and recorded results](docs/evaluation.md)
- [API contract](docs/api.md)
- [Deployment guide](docs/deployment.md)
- [Privacy, safety, and limitations](docs/privacy-and-limitations.md)
- [Reproduction notes](docs/reproducing-results.md)
- [Public release exclusions](PUBLIC_RELEASE_EXCLUSIONS.md)
- [Complete staging inventory](PUBLIC_RELEASE_INVENTORY.md)

This clean snapshot is based on source commit c860775719f1b3c98ecd9f865ab8a18b3a47182b. It does not carry the original repository history.