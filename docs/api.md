# API contract

The frontend uses VITE_API_BASE_URL and removes trailing slashes before joining API paths.

## Health

GET /health

A ready service responds 200 with status ok and both model flags true. A degraded service responds 503 with status degraded and false for each unavailable model. The response contains no filesystem paths.

## Email analysis

POST /api/v1/analysis/email

Request field:
- text: string, 1 to 100,000 characters; content must remain non-empty after email normalization.
- Unknown fields and non-string values are rejected.

Successful response fields:
- prediction: integer 0 or 1
- label: LEGITIMATE or PHISHING
- legitimate_probability: number from 0 to 1
- phishing_probability: number from 0 to 1

The response does not echo submitted or normalized email text.

## URL analysis

POST /api/v1/analysis/url

Request field:
- url: string, 1 to 2,048 characters.
- HTTP and HTTPS URLs are accepted. Scheme-less hostname input is accepted and handled as HTTP by the existing extractor.
- Unsupported schemes, malformed hosts/ports, embedded credentials, whitespace, control characters, and empty input are rejected.

The service does not fetch or visit the URL. The successful response has the same named fields, defined by the URL response model.

## Errors

Errors use an object with error.code and error.message:
- 422 INVALID_REQUEST: request validation failed.
- 503 MODEL_UNAVAILABLE: the relevant model is not ready.
- 500 INTERNAL_ERROR: inference failed or an unexpected request error occurred.

Responses do not contain stack traces, exception details, local paths, or request contents.

## CORS

The API accepts explicit comma-separated origins from CORS_ALLOWED_ORIGINS. If unset, only http://localhost:5173 and http://127.0.0.1:5173 are configured. Allowed methods are GET, POST, OPTIONS; the allowed header is Content-Type; credentials are disabled. Wildcard origins are rejected.