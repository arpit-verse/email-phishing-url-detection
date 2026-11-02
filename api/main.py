"""FastAPI service for email and URL phishing analysis and model health checks."""

from contextlib import asynccontextmanager
import hashlib
import logging
import os
import re
import tempfile
import unicodedata
from typing import Literal
from urllib.parse import urlsplit
from urllib.request import urlopen

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from training.predict_email import DEFAULT_MODEL_PATH as EMAIL_MODEL_PATH
from training.predict_email import load_model as load_email_model
from training.predict_email import normalize_email_text
from training.predict_email import predict as predict_email
from training.predict_url import DEFAULT_MODEL_PATH as URL_MODEL_PATH
from training.predict_url import load_model as load_url_model
from training.predict_url import predict as predict_url

logger = logging.getLogger(__name__)

MAX_EMAIL_TEXT_LENGTH = 100_000
MAX_URL_LENGTH = 2_048
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
MODEL_DOWNLOAD_TIMEOUT_SECONDS = 120
GIT_LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"
MODEL_DOWNLOADS = (
    (
        EMAIL_MODEL_PATH.name,
        "https://huggingface.co/arpit-verse/sentinel-phishing-models/resolve/88693cab826ef3c9041908e2f9d14d25a824ec52/email_pipeline.pkl",
        "8f36987f042631ee6831486ef94d16d942f53e70302efc37b3a366e28317884b",
    ),
    (
        URL_MODEL_PATH.name,
        "https://huggingface.co/arpit-verse/sentinel-phishing-models/resolve/de19ff7841af7ce3f96d2324fa25cc863a4cc3e8/url_random_forest.pkl",
        "db3729b9ad3c4b2d069e0f34348b736b00a525b807f838c7fd1c1f9924ee62d3",
    ),
)


def _configured_cors_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOWED_ORIGINS")
    if configured is None:
        return list(DEFAULT_CORS_ORIGINS)

    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    if any("*" in origin for origin in origins):
        raise ValueError("CORS_ALLOWED_ORIGINS must contain explicit origins, not wildcards.")
    return origins


def prepare_model_artifacts(models_dir=None) -> None:
    """Download and verify both fixed model artifacts before replacing either."""
    destination_dir = EMAIL_MODEL_PATH.parent if models_dir is None else models_dir
    destination_dir = os.fspath(destination_dir)
    os.makedirs(destination_dir, exist_ok=True)
    staged_paths: list[tuple[str, str]] = []

    try:
        for filename, download_url, expected_sha256 in MODEL_DOWNLOADS:
            destination = os.path.join(destination_dir, filename)
            temporary_path = None
            try:
                digest = hashlib.sha256()
                with urlopen(
                    download_url, timeout=MODEL_DOWNLOAD_TIMEOUT_SECONDS
                ) as response:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        dir=destination_dir,
                        prefix=f".{filename}.",
                        suffix=".download",
                        delete=False,
                    ) as temporary_file:
                        temporary_path = temporary_file.name
                        first_chunk = response.read(1024 * 1024)
                        if first_chunk.startswith(GIT_LFS_POINTER_HEADER):
                            raise ValueError("Model download returned a Git LFS pointer.")
                        while first_chunk:
                            temporary_file.write(first_chunk)
                            digest.update(first_chunk)
                            first_chunk = response.read(1024 * 1024)

                if digest.hexdigest() != expected_sha256:
                    raise ValueError("Model download SHA-256 verification failed.")
                staged_paths.append((temporary_path, destination))
                temporary_path = None
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.unlink(temporary_path)

        # Both downloads have passed integrity checks before either live path changes.
        for temporary_path, destination in staged_paths:
            os.replace(temporary_path, destination)
    finally:
        for temporary_path, _ in staged_paths:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


class EmailAnalysisRequest(BaseModel):
    """Email request schema; text is limited to 100,000 characters."""

    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(
        min_length=1,
        max_length=MAX_EMAIL_TEXT_LENGTH,
        description=(
            "Email content, up to 100,000 characters. Content that becomes empty "
            "after the project's email normalization is rejected."
        ),
    )

    @field_validator("text", mode="before")
    @classmethod
    def validate_email_text(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("Email text must be a string.")
        if len(value) > MAX_EMAIL_TEXT_LENGTH:
            raise ValueError("Email text exceeds the 100,000 character limit.")
        if not normalize_email_text(value):
            raise ValueError("Email text must contain usable content after normalization.")
        return value


class URLAnalysisRequest(BaseModel):
    """HTTP(S) URL request schema; scheme-less URLs are treated as HTTP."""

    model_config = ConfigDict(extra="forbid", strict=True)

    url: str = Field(
        min_length=1,
        max_length=MAX_URL_LENGTH,
        description=(
            "HTTP or HTTPS URL, up to 2,048 characters. A scheme-less hostname "
            "is accepted and treated as HTTP by the existing feature extractor. "
            "Submitted URLs are never fetched."
        ),
    )

    @field_validator("url", mode="before")
    @classmethod
    def validate_url(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("URL must be a string.")
        if len(value) > MAX_URL_LENGTH:
            raise ValueError("URL exceeds the 2,048 character limit.")

        value = value.strip()
        if not value:
            raise ValueError("URL must not be empty.")
        if any(
            character.isspace() or unicodedata.category(character) == "Cc"
            for character in value
        ):
            raise ValueError("URL must not contain whitespace or control characters.")

        # Recognize explicit URL schemes only at the start of the input. A
        # scheme-less URL may contain another URL in its query string.
        scheme_with_authority = re.match(
            r"^([a-zA-Z][a-zA-Z0-9+.-]*)://", value
        )
        if scheme_with_authority:
            scheme = scheme_with_authority.group(1).lower()
            if scheme not in {"http", "https"}:
                raise ValueError("Only HTTP and HTTPS URLs are accepted.")
            candidate = value
        else:
            # Reject malformed/non-hierarchical schemes such as file:/...;
            # a scheme-less hostname with a port (example.com:8443) is valid.
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:/", value):
                raise ValueError("Only HTTP and HTTPS URLs are accepted.")
            candidate = f"http://{value}"
        try:
            parsed = urlsplit(candidate)
            hostname = parsed.hostname
            # Accessing port performs urllib's built-in port syntax/range check.
            parsed.port
        except ValueError as exc:
            raise ValueError("URL must contain a valid hostname and port.") from exc

        if not hostname:
            raise ValueError("URL must contain a hostname.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("URLs containing embedded credentials are not accepted.")

        # Preserve the submitted scheme-less form. The existing URL predictor
        # adds http:// only for feature parsing and counts the original string.
        return value


class ModelReadiness(BaseModel):
    email_model_loaded: bool
    url_model_loaded: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models: ModelReadiness


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class EmailAnalysisResponse(BaseModel):
    """Email prediction result; submitted email content is not returned."""

    prediction: Literal[0, 1]
    label: Literal["PHISHING", "LEGITIMATE"]
    legitimate_probability: float = Field(ge=0.0, le=1.0)
    phishing_probability: float = Field(ge=0.0, le=1.0)


class URLAnalysisResponse(BaseModel):
    """URL prediction result; no network request is made for the URL."""

    prediction: Literal[0, 1]
    label: Literal["PHISHING", "LEGITIMATE"]
    legitimate_probability: float = Field(ge=0.0, le=1.0)
    phishing_probability: float = Field(ge=0.0, le=1.0)


def create_app(
    email_model_loader=None, url_model_loader=None, model_preparer=None
) -> FastAPI:
    """Create the app; injectable dependencies keep tests artifact-free."""
    email_loader = email_model_loader or load_email_model
    url_loader = url_model_loader or load_url_model
    if model_preparer is None:
        # Supplying test loaders opts out of production downloads unless a test
        # explicitly injects a preparer.
        model_preparer = (
            prepare_model_artifacts
            if email_model_loader is None and url_model_loader is None
            else lambda: None
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loaded_models = {"email": None, "url": None}
        preparation_succeeded = True
        try:
            model_preparer()
        except Exception:
            logger.exception(
                "Model artifact preparation failed; starting with models unavailable."
            )
            # Do not load either destination if artifact preparation failed.
            preparation_succeeded = False

        for name, loader in (
            (("email", email_loader), ("url", url_loader))
            if preparation_succeeded
            else ()
        ):
            try:
                model = loader()
                if model is None:
                    raise RuntimeError("Model loader returned no model.")
                loaded_models[name] = model
            except Exception:
                # Keep the service available so /health can report degraded
                # readiness without exposing loader details or filesystem paths.
                loaded_models[name] = None

        app.state.models = loaded_models
        yield

    app = FastAPI(title="Email Phishing and URL Detection API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_configured_cors_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_REQUEST",
                    message="Request validation failed. Check the documented fields and limits.",
                )
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="INTERNAL_ERROR",
                    message="The request could not be completed.",
                )
            ).model_dump(),
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
        tags=["health"],
    )
    async def health(request: Request) -> JSONResponse:
        models = getattr(request.app.state, "models", {"email": None, "url": None})
        readiness = ModelReadiness(
            email_model_loaded=models.get("email") is not None,
            url_model_loaded=models.get("url") is not None,
        )
        is_ready = readiness.email_model_loaded and readiness.url_model_loaded
        body = HealthResponse(
            status="ok" if is_ready else "degraded",
            models=readiness,
        )
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content=body.model_dump(),
        )

    @app.post(
        "/api/v1/analysis/email",
        response_model=EmailAnalysisResponse,
        responses={
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["analysis"],
    )
    async def analyze_email(
        body: EmailAnalysisRequest, request: Request
    ) -> EmailAnalysisResponse | JSONResponse:
        model = getattr(request.app.state, "models", {}).get("email")
        if model is None:
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code="MODEL_UNAVAILABLE",
                        message="Email analysis is temporarily unavailable.",
                    )
                ).model_dump(),
            )

        try:
            result = predict_email(body.text, model=model)
            return EmailAnalysisResponse(
                prediction=result["prediction"],
                label=result["label"],
                legitimate_probability=result["legitimate_probability"],
                phishing_probability=result["phishing_probability"],
            )
        except Exception:
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code="INTERNAL_ERROR",
                        message="The request could not be completed.",
                    )
                ).model_dump(),
            )

    @app.post(
        "/api/v1/analysis/url",
        response_model=URLAnalysisResponse,
        responses={
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["analysis"],
    )
    async def analyze_url(
        body: URLAnalysisRequest, request: Request
    ) -> URLAnalysisResponse | JSONResponse:
        model = getattr(request.app.state, "models", {}).get("url")
        if model is None:
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code="MODEL_UNAVAILABLE",
                        message="URL analysis is temporarily unavailable.",
                    )
                ).model_dump(),
            )

        try:
            result = predict_url(body.url, model=model)
            return URLAnalysisResponse(
                prediction=result["prediction"],
                label=result["label"],
                legitimate_probability=result["legitimate_probability"],
                phishing_probability=result["phishing_probability"],
            )
        except Exception:
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code="INTERNAL_ERROR",
                        message="The request could not be completed.",
                    )
                ).model_dump(),
            )

    return app


app = create_app()
