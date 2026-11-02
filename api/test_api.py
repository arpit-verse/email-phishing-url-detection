import hashlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import (
    EmailAnalysisRequest,
    URLAnalysisRequest,
    create_app,
    prepare_model_artifacts,
)


class FakeEmailModel:
    def __init__(self):
        self.calls = 0
        self.named_steps = {}

    def predict_proba(self, values):
        self.calls += 1
        return [[0.17, 0.83]]

    def predict(self, values):
        return [1]


class FakeURLModel:
    def __init__(self):
        self.calls = 0

    def predict_proba(self, values):
        self.calls += 1
        return [[0.91, 0.09]]

    def predict(self, values):
        return [0]


class TestRequestSchemas(unittest.TestCase):
    def test_email_accepts_usable_text(self):
        request = EmailAnalysisRequest(text="Please review the attached report.")
        self.assertEqual(request.text, "Please review the attached report.")

    def test_email_rejects_empty_or_unusable_text(self):
        for value in ("", "  \t\n", "<p></p>", "!!!"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValidationError):
                    EmailAnalysisRequest(text=value)

    def test_email_rejects_oversize_and_non_string_values(self):
        with self.assertRaises(ValidationError):
            EmailAnalysisRequest(text="a" * 100_001)
        with self.assertRaises(ValidationError):
            EmailAnalysisRequest(text=123)

    def test_url_accepts_http_https_and_scheme_less_hostnames(self):
        for value in (
            "https://example.com/path?q=1",
            "http://example.com",
            "example.com/path?q=1",
            "example.com/path?next=https://other.example",
            "example.com:8443/path",
        ):
            with self.subTest(value=value):
                self.assertEqual(URLAnalysisRequest(url=value).url, value)

    def test_url_rejects_unsupported_or_malformed_values(self):
        for value in (
            "",
            "   ",
            "ftp://example.com",
            "javascript://example.com",
            "file:/etc/passwd",
            "https:///missing-host",
            "https://example.com:99999/path",
            "https://user:placeholder@example.com",
            "https://example.com/path with space",
            "https://example.com/\x00path",
            "a" * 2_049,
        ):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValidationError):
                    URLAnalysisRequest(url=value)

    def test_requests_reject_extra_fields(self):
        with self.assertRaises(ValidationError):
            EmailAnalysisRequest(text="valid email", model_path="C:/private/model.pkl")
        with self.assertRaises(ValidationError):
            URLAnalysisRequest(url="example.com", model_path="C:/private/model.pkl")


class TestModelPreparation(unittest.TestCase):
    EMAIL_BYTES = b"verified email model test bytes"
    URL_BYTES = b"verified url model test bytes"

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.models_dir = self.temporary_directory.name
        self.manifest = (
            (
                "email_pipeline.pkl",
                "https://models.example.test/email-at-pinned-revision",
                hashlib.sha256(self.EMAIL_BYTES).hexdigest(),
            ),
            (
                "url_random_forest.pkl",
                "https://models.example.test/url-at-pinned-revision",
                hashlib.sha256(self.URL_BYTES).hexdigest(),
            ),
        )

    def _destination(self, filename):
        return os.path.join(self.models_dir, filename)

    def _assert_no_download_temporary_files(self):
        self.assertEqual(
            [name for name in os.listdir(self.models_dir) if name.endswith(".download")],
            [],
        )

    def test_success_verifies_both_files_and_cleans_temporary_files(self):
        with patch("api.main.MODEL_DOWNLOADS", self.manifest), patch(
            "api.main.urlopen",
            side_effect=[io.BytesIO(self.EMAIL_BYTES), io.BytesIO(self.URL_BYTES)],
        ):
            prepare_model_artifacts(self.models_dir)

        with open(self._destination("email_pipeline.pkl"), "rb") as model_file:
            self.assertEqual(model_file.read(), self.EMAIL_BYTES)
        with open(self._destination("url_random_forest.pkl"), "rb") as model_file:
            self.assertEqual(model_file.read(), self.URL_BYTES)
        self._assert_no_download_temporary_files()

    def test_hash_mismatch_does_not_replace_either_destination_and_cleans_up(self):
        original_email = b"existing email artifact"
        original_url = b"existing url artifact"
        for filename, content in (
            ("email_pipeline.pkl", original_email),
            ("url_random_forest.pkl", original_url),
        ):
            with open(self._destination(filename), "wb") as model_file:
                model_file.write(content)
        invalid_manifest = (
            self.manifest[0],
            (self.manifest[1][0], self.manifest[1][1], "0" * 64),
        )

        with patch("api.main.MODEL_DOWNLOADS", invalid_manifest), patch(
            "api.main.urlopen",
            side_effect=[io.BytesIO(self.EMAIL_BYTES), io.BytesIO(self.URL_BYTES)],
        ), self.assertRaisesRegex(ValueError, "SHA-256"):
            prepare_model_artifacts(self.models_dir)

        with open(self._destination("email_pipeline.pkl"), "rb") as model_file:
            self.assertEqual(model_file.read(), original_email)
        with open(self._destination("url_random_forest.pkl"), "rb") as model_file:
            self.assertEqual(model_file.read(), original_url)
        self._assert_no_download_temporary_files()

    def test_git_lfs_pointer_is_rejected_without_replacing_destinations(self):
        original_email = b"existing email artifact"
        original_url = b"existing url artifact"
        for filename, content in (
            ("email_pipeline.pkl", original_email),
            ("url_random_forest.pkl", original_url),
        ):
            with open(self._destination(filename), "wb") as model_file:
                model_file.write(content)
        pointer = (
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:" + b"a" * 64 + b"\nsize 123\n"
        )

        with patch("api.main.MODEL_DOWNLOADS", self.manifest), patch(
            "api.main.urlopen", return_value=io.BytesIO(pointer)
        ), self.assertRaisesRegex(ValueError, "Git LFS pointer"):
            prepare_model_artifacts(self.models_dir)

        with open(self._destination("email_pipeline.pkl"), "rb") as model_file:
            self.assertEqual(model_file.read(), original_email)
        with open(self._destination("url_random_forest.pkl"), "rb") as model_file:
            self.assertEqual(model_file.read(), original_url)
        self._assert_no_download_temporary_files()

    def test_download_exception_cleans_temporary_files_and_preserves_destinations(self):
        original_email = b"existing email artifact"
        original_url = b"existing url artifact"
        for filename, content in (
            ("email_pipeline.pkl", original_email),
            ("url_random_forest.pkl", original_url),
        ):
            with open(self._destination(filename), "wb") as model_file:
                model_file.write(content)

        with patch("api.main.MODEL_DOWNLOADS", self.manifest), patch(
            "api.main.urlopen", side_effect=OSError("private download detail")
        ), self.assertRaisesRegex(OSError, "private download detail"):
            prepare_model_artifacts(self.models_dir)

        with open(self._destination("email_pipeline.pkl"), "rb") as model_file:
            self.assertEqual(model_file.read(), original_email)
        with open(self._destination("url_random_forest.pkl"), "rb") as model_file:
            self.assertEqual(model_file.read(), original_url)
        self._assert_no_download_temporary_files()

    def test_failure_staging_second_file_does_not_replace_first_destination(self):
        original_email = b"existing email artifact"
        original_url = b"existing url artifact"
        for filename, content in (
            ("email_pipeline.pkl", original_email),
            ("url_random_forest.pkl", original_url),
        ):
            with open(self._destination(filename), "wb") as model_file:
                model_file.write(content)

        with patch("api.main.MODEL_DOWNLOADS", self.manifest), patch(
            "api.main.urlopen",
            side_effect=[io.BytesIO(self.EMAIL_BYTES), OSError("second download failed")],
        ), self.assertRaisesRegex(OSError, "second download failed"):
            prepare_model_artifacts(self.models_dir)

        with open(self._destination("email_pipeline.pkl"), "rb") as model_file:
            self.assertEqual(model_file.read(), original_email)
        with open(self._destination("url_random_forest.pkl"), "rb") as model_file:
            self.assertEqual(model_file.read(), original_url)
        self._assert_no_download_temporary_files()


class TestHealthEndpoint(unittest.TestCase):
    def test_model_preparation_failure_degrades_health_without_loading_models(self):
        loader_calls = {"email": 0, "url": 0}

        def load_email():
            loader_calls["email"] += 1
            return FakeEmailModel()

        def load_url():
            loader_calls["url"] += 1
            return FakeURLModel()

        def fail_preparation():
            raise RuntimeError("private download path and details")

        app = create_app(
            email_model_loader=load_email,
            url_model_loader=load_url,
            model_preparer=fail_preparation,
        )
        with TestClient(app) as client:
            health_response = client.get("/health")
            analysis_response = client.post(
                "/api/v1/analysis/email", json={"text": "A valid message"}
            )

        self.assertEqual(health_response.status_code, 503)
        self.assertEqual(
            health_response.json(),
            {
                "status": "degraded",
                "models": {"email_model_loaded": False, "url_model_loaded": False},
            },
        )
        self.assertEqual(analysis_response.status_code, 503)
        self.assertNotIn("private", health_response.text + analysis_response.text)
        self.assertEqual(loader_calls, {"email": 0, "url": 0})

    def test_cors_allows_only_configured_vite_origins(self):
        app = create_app(
            email_model_loader=lambda: object(),
            url_model_loader=lambda: object(),
        )
        with TestClient(app) as client:
            allowed_response = client.options(
                "/api/v1/analysis/email",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type",
                },
            )
            disallowed_response = client.options(
                "/api/v1/analysis/email",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type",
                },
            )

        self.assertEqual(allowed_response.status_code, 200)
        self.assertEqual(
            allowed_response.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )
        self.assertNotIn("access-control-allow-credentials", allowed_response.headers)
        self.assertEqual(disallowed_response.status_code, 400)
        self.assertNotIn("access-control-allow-origin", disallowed_response.headers)

    def test_cors_uses_trimmed_environment_allowlist_without_wildcards(self):
        with patch.dict(
            os.environ,
            {
                "CORS_ALLOWED_ORIGINS": (
                    " https://sentinel.example, , https://preview.sentinel.example "
                )
            },
        ):
            app = create_app(
                email_model_loader=lambda: object(),
                url_model_loader=lambda: object(),
            )
            with TestClient(app) as client:
                allowed = client.options(
                    "/api/v1/analysis/email",
                    headers={
                        "Origin": "https://sentinel.example",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "Content-Type",
                    },
                )
                disallowed = client.options(
                    "/api/v1/analysis/email",
                    headers={
                        "Origin": "http://localhost:5173",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "Content-Type",
                    },
                )

        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(
            allowed.headers.get("access-control-allow-origin"),
            "https://sentinel.example",
        )
        self.assertEqual(disallowed.status_code, 400)
        self.assertNotIn("access-control-allow-origin", disallowed.headers)

        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "https://*.example"}):
            with self.assertRaisesRegex(ValueError, "not wildcards"):
                create_app()

    def test_each_model_loader_is_called_once_per_lifespan(self):
        calls = {"email": 0, "url": 0}

        def load_email():
            calls["email"] += 1
            return object()

        def load_url():
            calls["url"] += 1
            return object()

        app = create_app(email_model_loader=load_email, url_model_loader=load_url)
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/health").status_code, 200)

        self.assertEqual(calls, {"email": 1, "url": 1})

    def test_health_reports_loaded_models_without_paths(self):
        app = create_app(
            email_model_loader=lambda: object(),
            url_model_loader=lambda: object(),
        )
        with TestClient(app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "models": {"email_model_loaded": True, "url_model_loaded": True},
            },
        )
        self.assertNotIn("D:\\", response.text)
        self.assertNotIn("models/", response.text)

    def test_health_reports_degraded_when_a_model_fails_to_load(self):
        app = create_app(
            email_model_loader=lambda: (_ for _ in ()).throw(
                FileNotFoundError("D:/private/project/models/email_pipeline.pkl")
            ),
            url_model_loader=lambda: object(),
        )
        with TestClient(app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "status": "degraded",
                "models": {"email_model_loaded": False, "url_model_loaded": True},
            },
        )
        self.assertNotIn("private", response.text)
        self.assertNotIn("FileNotFoundError", response.text)


class TestAnalysisEndpoints(unittest.TestCase):
    def test_email_and_url_predictions_return_distinct_mapped_results(self):
        app = create_app(
            email_model_loader=FakeEmailModel,
            url_model_loader=FakeURLModel,
        )
        email_text = "Please review the attached report."
        with TestClient(app) as client:
            email_response = client.post(
                "/api/v1/analysis/email", json={"text": email_text}
            )
            url_response = client.post(
                "/api/v1/analysis/url", json={"url": "https://example.com/path"}
            )

        self.assertEqual(email_response.status_code, 200)
        self.assertEqual(
            email_response.json(),
            {
                "prediction": 1,
                "label": "PHISHING",
                "legitimate_probability": 0.17,
                "phishing_probability": 0.83,
            },
        )
        self.assertNotIn(email_text, email_response.text)
        self.assertEqual(url_response.status_code, 200)
        self.assertEqual(
            url_response.json(),
            {
                "prediction": 0,
                "label": "LEGITIMATE",
                "legitimate_probability": 0.91,
                "phishing_probability": 0.09,
            },
        )

    def test_invalid_email_and_url_requests_use_sanitized_validation_error(self):
        app = create_app(
            email_model_loader=FakeEmailModel,
            url_model_loader=FakeURLModel,
        )
        with TestClient(app) as client:
            empty_email_response = client.post(
                "/api/v1/analysis/email", json={"text": ""}
            )
            unusable_email_response = client.post(
                "/api/v1/analysis/email", json={"text": "<p></p>"}
            )
            empty_url_response = client.post(
                "/api/v1/analysis/url", json={"url": ""}
            )
            invalid_url_response = client.post(
                "/api/v1/analysis/url", json={"url": "ftp://example.com"}
            )

        expected = {
            "error": {
                "code": "INVALID_REQUEST",
                "message": "Request validation failed. Check the documented fields and limits.",
            }
        }
        for response in (
            empty_email_response,
            unusable_email_response,
            empty_url_response,
            invalid_url_response,
        ):
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json(), expected)

    def test_unavailable_models_return_sanitized_503(self):
        app = create_app(
            email_model_loader=lambda: None,
            url_model_loader=lambda: (_ for _ in ()).throw(
                FileNotFoundError("D:/private/models/url.pkl")
            ),
        )
        with TestClient(app) as client:
            email_response = client.post(
                "/api/v1/analysis/email", json={"text": "A valid message"}
            )
            url_response = client.post(
                "/api/v1/analysis/url", json={"url": "example.com"}
            )

        for response in (email_response, url_response):
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json(),
                {
                    "error": {
                        "code": "MODEL_UNAVAILABLE",
                        "message": "Email analysis is temporarily unavailable."
                        if response is email_response
                        else "URL analysis is temporarily unavailable.",
                    }
                },
            )
            self.assertNotIn("private", response.text)
            self.assertNotIn("FileNotFoundError", response.text)

    def test_inference_value_errors_and_unexpected_errors_are_sanitized_500(self):
        app = create_app(
            email_model_loader=FakeEmailModel,
            url_model_loader=FakeURLModel,
        )
        with patch(
            "api.main.predict_email",
            side_effect=ValueError("private path and submitted message"),
        ), patch(
            "api.main.predict_url",
            side_effect=RuntimeError("private path and submitted URL"),
        ), TestClient(app) as client:
            email_response = client.post(
                "/api/v1/analysis/email", json={"text": "Valid content"}
            )
            url_response = client.post(
                "/api/v1/analysis/url", json={"url": "https://example.com"}
            )

        for response in (email_response, url_response):
            self.assertEqual(response.status_code, 500)
            self.assertEqual(
                response.json(),
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "The request could not be completed.",
                    }
                },
            )
            self.assertNotIn("private path", response.text)
            self.assertNotIn("submitted", response.text)

    def test_api_calls_reuse_models_loaded_once_at_startup(self):
        calls = {"email_loader": 0, "url_loader": 0}
        email_model = FakeEmailModel()
        url_model = FakeURLModel()

        def load_email():
            calls["email_loader"] += 1
            return email_model

        def load_url():
            calls["url_loader"] += 1
            return url_model

        app = create_app(email_model_loader=load_email, url_model_loader=load_url)
        with TestClient(app) as client:
            for _ in range(2):
                self.assertEqual(
                    client.post(
                        "/api/v1/analysis/email", json={"text": "Valid message"}
                    ).status_code,
                    200,
                )
                self.assertEqual(
                    client.post(
                        "/api/v1/analysis/url", json={"url": "example.com"}
                    ).status_code,
                    200,
                )

        self.assertEqual(calls, {"email_loader": 1, "url_loader": 1})
        self.assertEqual(email_model.calls, 2)
        self.assertEqual(url_model.calls, 2)

    def test_url_scheme_regressions_remain_validated(self):
        app = create_app(
            email_model_loader=FakeEmailModel,
            url_model_loader=FakeURLModel,
        )
        accepted = (
            "https://example.com/path",
            "http://example.com/path",
            "example.com/path?next=https://other.example",
        )
        rejected = ("file:/etc/passwd", "ftp://example.com")
        with TestClient(app) as client:
            for value in accepted:
                with self.subTest(accepted=value):
                    self.assertEqual(
                        client.post("/api/v1/analysis/url", json={"url": value}).status_code,
                        200,
                    )
            for value in rejected:
                with self.subTest(rejected=value):
                    self.assertEqual(
                        client.post("/api/v1/analysis/url", json={"url": value}).status_code,
                        422,
                    )


if __name__ == "__main__":
    unittest.main()
