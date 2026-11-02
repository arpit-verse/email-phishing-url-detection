# Deployment guide

This describes repository configuration; it does not claim that a hosted deployment is currently active.

## Render API

The checked-in render.yaml configures the repository root as service root, Python 3.14.3, build command pip install -r api/requirements.txt, start command uvicorn api.main:app --host 0.0.0.0 --port $PORT, and health-check path /health.

Set CORS_ALLOWED_ORIGINS to the exact deployed frontend origin, comma-separated if more than one. Do not use a wildcard. The backend downloads the pinned model files at startup, so the service needs outbound access to the Hugging Face artifact URLs. Startup checks both files before loading either; failed preparation leaves readiness degraded.

## Vercel frontend

Use frontend/ as the project root, npm run build as the build command, and dist as the output directory. Set VITE_API_BASE_URL to the HTTPS base URL of the Render API. This is public build configuration, not a secret. Do not put tokens or credentials in Vite variables.

## Local development

From the repository root, install API dependencies in a dedicated Python environment and run:

    python -m uvicorn api.main:app --reload

For the frontend:

    cd frontend
    npm ci
    npm run dev

The API defaults to the Vite localhost origins if CORS_ALLOWED_ORIGINS is unset. The default browser API URL is http://127.0.0.1:8000.

## Operational limitations

The URL model is about 98.5 MB on disk, in addition to the email model and Python runtime memory. Render startup duration, memory use, cold starts, and Free-tier suitability have not been measured here. Confirm current platform limits and measure the deployed service before relying on availability.