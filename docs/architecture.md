# Architecture

## Runtime flow

1. The React/Vite client collects email text or a URL and sends JSON to FastAPI.
2. FastAPI validates the request and uses model objects loaded once during application startup.
3. The email predictor normalizes text and passes it to the fitted TF-IDF/classifier pipeline.
4. The URL predictor extracts the fixed 25-feature vector and passes it to the fitted Random Forest.
5. The API returns a label and class probabilities. It does not return submitted or normalized email text, recognized email tokens, or extracted URL features.

The URL is never opened, fetched, or checked against a remote reputation service by these analysis paths.

## Components

- frontend/src/App.jsx: navigation, analysis forms, loading/error/result UI, and in-memory real scan history. Overview metrics, activity and sample records are demo data.
- api/main.py: request schemas, origin allowlist, startup model preparation, readiness endpoint, analysis endpoints, and sanitized errors.
- training/predict_email.py and training/preprocess_email.py: email inference and shared normalization/preparation logic.
- training/predict_url.py and training/extract_url_features.py: deployed URL inference feature order and training-time extraction.
- render.yaml: Render service definition. Production frontend configuration is described in deployment.md.

## Model lifecycle

At production startup, the API downloads two fixed files from the public Sentinel Hugging Face repository into temporary files under models/. Both downloads must pass SHA-256 checks and LFS-pointer rejection before either destination is replaced. The predictor loaders then load the two artifacts once. If preparation or loading fails, readiness is degraded and analysis returns a sanitized unavailable response.

This code path needs outbound access to the pinned Hugging Face files. Model binaries are not in this snapshot.

## Boundaries

There is no database, login system, or persistent scan history. The server code does not implement durable input storage. Hosting, network, and browser telemetry outside this source tree are not assessed here.