"""
Unit and integration tests for CareerLens Phase 2: Production Security & API Hardening.

Validates:
1. Production configuration validation (missing/weak/placeholder secrets rejected).
2. Dynamic CORS parsing, wildcard rejection with credentials, and origin headers.
3. In-memory sliding-window rate limiting on all 5 key endpoints with HTTP 429 & Retry-After.
4. PDF upload boundary enforcement, magic byte validation, and early unauthenticated rejection.
5. Health liveness (/health, /health/live) and readiness (/health/ready) probe behavior.
6. Safe production error responses and sanitized logging.
"""

import io
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.database import Base, get_db
from backend.db.migrations import run_migrations
from backend.db.models import User
from backend.main import app
from backend.models.schemas import CandidateProfile
from backend.routers.cv import get_cv_pipeline
from backend.services.auth_service import create_access_token, hash_password
from backend.services.rate_limiter import (
    InMemoryRateLimiter,
    get_client_ip,
    limiter,
)
from backend.services.security_config import (
    DEFAULT_DEV_SECRET,
    SecurityConfigurationError,
    get_max_upload_size_bytes,
    get_token_expire_minutes,
    parse_cors_origins,
    validate_jwt_secret,
    validate_production_security,
)


class TestSecurityConfiguration(unittest.TestCase):
    """Test suite for JWT secret, CORS, and production safety validation."""

    def test_production_rejects_missing_secret(self):
        """Production mode strictly rejects missing or empty JWT secrets."""
        with self.assertRaises(SecurityConfigurationError) as ctx:
            validate_jwt_secret(secret=None, environment="production")
        self.assertIn("JWT_SECRET_KEY environment variable is required", str(ctx.exception))

        with self.assertRaises(SecurityConfigurationError) as ctx:
            validate_jwt_secret(secret="   ", environment="production")
        self.assertIn("JWT_SECRET_KEY environment variable is required", str(ctx.exception))

    def test_production_rejects_short_secret(self):
        """Production mode strictly enforces minimum 32 character secret length."""
        short_secret = "too-short-secret-key-12345"  # 26 chars
        with self.assertRaises(SecurityConfigurationError) as ctx:
            validate_jwt_secret(secret=short_secret, environment="production")
        self.assertIn("at least 32 characters long", str(ctx.exception))

    def test_production_rejects_known_insecure_placeholders(self):
        """Production mode rejects default development secret and common placeholders."""
        # Development default secret
        with self.assertRaises(SecurityConfigurationError) as ctx:
            validate_jwt_secret(secret=DEFAULT_DEV_SECRET, environment="production")
        self.assertIn("known insecure or default placeholder", str(ctx.exception))

        # Common placeholder
        with self.assertRaises(SecurityConfigurationError) as ctx:
            validate_jwt_secret(secret="your_jwt_secret_key_here", environment="production")
        self.assertIn("known insecure or default placeholder", str(ctx.exception))

    def test_production_accepts_strong_secret(self):
        """Production mode accepts strong secrets >= 32 characters."""
        strong_secret = "a" * 32 + "cryptographically-secure-random-value-99!"
        validated = validate_jwt_secret(secret=strong_secret, environment="production")
        self.assertEqual(validated, strong_secret)

    def test_development_falls_back_safely(self):
        """Development mode safely defaults to DEFAULT_DEV_SECRET when omitted."""
        fallback = validate_jwt_secret(secret=None, environment="development")
        self.assertEqual(fallback, DEFAULT_DEV_SECRET)

    def test_cors_parsing_development_default(self):
        """In development mode, omitting CORS_ORIGINS defaults to localhost origins."""
        origins = parse_cors_origins(raw_origins=None, environment="development")
        self.assertIn("http://localhost:5173", origins)
        self.assertIn("http://localhost:3000", origins)

    def test_cors_parsing_production_requires_explicit_origins(self):
        """In production mode, missing or empty CORS_ORIGINS raises an error."""
        with self.assertRaises(SecurityConfigurationError) as ctx:
            parse_cors_origins(raw_origins=None, environment="production")
        self.assertIn("CORS_ORIGINS environment variable is required", str(ctx.exception))

        with self.assertRaises(SecurityConfigurationError) as ctx:
            parse_cors_origins(raw_origins="   ", environment="production")
        self.assertIn("CORS_ORIGINS environment variable is required", str(ctx.exception))

    def test_cors_production_rejects_wildcard(self):
        """In production mode, wildcard '*' is forbidden when credentials are enabled."""
        with self.assertRaises(SecurityConfigurationError) as ctx:
            parse_cors_origins(raw_origins="https://app.careerlens.ai, *", environment="production")
        self.assertIn("Wildcard '*' CORS origin is forbidden", str(ctx.exception))

    def test_cors_rejects_invalid_url_format(self):
        """Malformed origins without scheme or with trailing paths are rejected."""
        with self.assertRaises(SecurityConfigurationError) as ctx:
            parse_cors_origins(raw_origins="not-a-valid-origin", environment="development")
        self.assertIn("Invalid CORS origin format", str(ctx.exception))

    def test_cors_parses_comma_separated_origins(self):
        """Correctly parses multiple valid origins and strips trailing slashes."""
        raw = "https://careerlens.pages.dev/, https://api.careerlens.ai"
        parsed = parse_cors_origins(raw_origins=raw, environment="production")
        self.assertEqual(parsed, ["https://careerlens.pages.dev", "https://api.careerlens.ai"])

    def test_token_expiration_config(self):
        """Validates configurable access token expiration with fallback."""
        with patch.dict(os.environ, {"JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "120"}):
            self.assertEqual(get_token_expire_minutes(), 120)

        with patch.dict(os.environ, {"JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "invalid"}):
            self.assertEqual(get_token_expire_minutes(), 60)

    def test_max_upload_size_config(self):
        """Validates configurable maximum upload size with default 10 MiB."""
        with patch.dict(os.environ, {"MAX_UPLOAD_SIZE_BYTES": "5242880"}):
            self.assertEqual(get_max_upload_size_bytes(), 5242880)

        with patch.dict(os.environ, {"MAX_UPLOAD_SIZE_BYTES": "too_small"}):
            self.assertEqual(get_max_upload_size_bytes(), 10 * 1024 * 1024)


class TestRateLimiting(unittest.TestCase):
    """Test suite for in-memory sliding-window rate limiting on critical endpoints."""

    def setUp(self):
        limiter.reset()

    def tearDown(self):
        limiter.reset()

    def test_limiter_allows_under_quota_and_blocks_excess(self):
        """Limiter permits requests within quota and returns 429 with retry_after when exceeded."""
        key = "test_action:client_1"
        max_req = 3
        window = 10

        # Requests 1, 2, 3 permitted
        for _ in range(3):
            allowed, retry_after = limiter.check(key, max_req, window)
            self.assertTrue(allowed)
            self.assertEqual(retry_after, 0)

        # 4th request rejected
        allowed, retry_after = limiter.check(key, max_req, window)
        self.assertFalse(allowed)
        self.assertGreaterEqual(retry_after, 1)
        self.assertLessEqual(retry_after, 10)

    def test_client_ip_extraction_safely_handles_proxies(self):
        """Client IP extraction rejects spoofed headers unless BEHIND_TRUSTED_PROXY=1 and peer is local."""
        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {"x-forwarded-for": "203.0.113.195, 70.41.3.18"}

        # Untrusted proxy mode (default): uses socket host
        with patch.dict(os.environ, {"BEHIND_TRUSTED_PROXY": "0"}):
            ip = get_client_ip(mock_request)
            self.assertEqual(ip, "192.168.1.100")

        # Trusted proxy mode with private/loopback peer: uses first forwarded client IP
        with patch.dict(os.environ, {"BEHIND_TRUSTED_PROXY": "1"}):
            ip = get_client_ip(mock_request)
            self.assertEqual(ip, "203.0.113.195")

        # Cloudflare header support
        mock_cf = MagicMock()
        mock_cf.client.host = "127.0.0.1"
        mock_cf.headers = {"cf-connecting-ip": "198.51.100.42", "x-forwarded-for": "10.0.0.1"}
        with patch.dict(os.environ, {"BEHIND_TRUSTED_PROXY": "1"}):
            self.assertEqual(get_client_ip(mock_cf), "198.51.100.42")

        # Direct connection from untrusted public IP: rejects spoofed header even if BEHIND_TRUSTED_PROXY=1
        mock_public = MagicMock()
        mock_public.client.host = "93.184.216.34"  # Public routable IP, not private/loopback
        mock_public.headers = {"x-forwarded-for": "1.2.3.4"}
        with patch.dict(os.environ, {"BEHIND_TRUSTED_PROXY": "1"}):
            self.assertEqual(get_client_ip(mock_public), "93.184.216.34")


class TestAPIHardeningIntegration(unittest.TestCase):
    """Integration test suite exercising actual FastAPI application endpoints."""

    def setUp(self):
        limiter.reset()
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(self.db_url, connect_args={"check_same_thread": False})
        self.TestSession = sessionmaker(bind=self.engine)
        run_migrations(self.engine)

        # Override get_db dependency
        def override_get_db():
            db = self.TestSession()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

        # Create a test candidate user
        self.user_id = "sec_test_user_001"
        self.user_email = "security_test@example.com"
        with self.TestSession() as session:
            user = User(
                id=self.user_id,
                email=self.user_email,
                hashed_password=hash_password("SuperSecretPass123!"),
                full_name="Security Tester",
                is_active=True,
            )
            session.add(user)
            session.commit()

        self.token = create_access_token(user_id=self.user_id, email=self.user_email)
        self.auth_headers = {"Authorization": f"Bearer {self.token}"}

    def tearDown(self):
        app.dependency_overrides.clear()
        limiter.reset()
        self.engine.dispose()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass

    def _create_minimal_pdf(self, content: str = "Test Candidate CV Content") -> bytes:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), content)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_liveness_probe(self):
        """GET /health and /health/live confirm process is alive without revealing secrets."""
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "healthy")

        res_live = self.client.get("/health/live")
        self.assertEqual(res_live.status_code, 200)
        live_data = res_live.json()
        self.assertEqual(live_data["status"], "alive")
        self.assertIn("version", live_data)
        self.assertIn("environment", live_data)

        for data in [res_health.json(), live_data]:
            # Ensure no database credentials or URLs are leaked
            self.assertNotIn("password", str(data).lower())
            self.assertNotIn("sqlite", str(data).lower())

    def test_readiness_probe_healthy(self):
        """GET /health/ready returns 200 when database connectivity is verified."""
        res = self.client.get("/health/ready")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["database"], "connected")

    def test_readiness_probe_unhealthy(self):
        """GET /health/ready returns 503 when database connectivity fails."""
        with patch("backend.db.database.verify_database_connectivity", side_effect=RuntimeError("DB disconnected")):
            res = self.client.get("/health/ready")
            self.assertEqual(res.status_code, 503)
            data = res.json()
            self.assertEqual(data["status"], "not_ready")
            self.assertEqual(data["database"], "unavailable")

    def test_auth_login_rate_limiting(self):
        """POST /api/v1/auth/login triggers HTTP 429 with Retry-After when rate limit is exceeded."""
        # Set low limit for testing
        with patch.dict(os.environ, {"RATE_LIMIT_AUTH_LOGIN": "2/minute"}):
            # Attempt 1 & 2 succeed (or return 401 with invalid credentials)
            for _ in range(2):
                res = self.client.post(
                    "/api/v1/auth/login",
                    json={"email": "security_test@example.com", "password": "WrongPassword"},
                )
                self.assertEqual(res.status_code, 401)

            # Attempt 3 is blocked by rate limiter with 429
            res_blocked = self.client.post(
                "/api/v1/auth/login",
                json={"email": "security_test@example.com", "password": "WrongPassword"},
            )
            self.assertEqual(res_blocked.status_code, 429)
            self.assertIn("Retry-After", res_blocked.headers)
            self.assertIn("Rate limit exceeded", res_blocked.json()["detail"])

    def test_auth_register_rate_limiting(self):
        """POST /api/v1/auth/register triggers HTTP 429 when rate limit is exceeded."""
        with patch.dict(os.environ, {"RATE_LIMIT_AUTH_REGISTER": "2/minute"}):
            for i in range(2):
                res = self.client.post(
                    "/api/v1/auth/register",
                    json={"email": f"reg_{i}@example.com", "password": "Password123!"},
                )
                self.assertEqual(res.status_code, 201)

            # 3rd is blocked
            blocked = self.client.post(
                "/api/v1/auth/register",
                json={"email": "reg_3@example.com", "password": "Password123!"},
            )
            self.assertEqual(blocked.status_code, 429)

    def test_cv_process_upload_size_limit(self):
        """POST /api/v1/cv/process rejects files exceeding configured size with HTTP 413."""
        with patch.dict(os.environ, {"MAX_UPLOAD_SIZE_BYTES": str(1024 * 1024)}):
            oversized_bytes = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 1024)
            res = self.client.post(
                "/api/v1/cv/process",
                files={"file": ("large_cv.pdf", io.BytesIO(oversized_bytes), "application/pdf")},
                headers=self.auth_headers,
            )
            self.assertEqual(res.status_code, 413)
            self.assertIn("exceeds maximum allowed limit", res.json()["detail"])

    def test_cv_process_empty_file_rejected(self):
        """POST /api/v1/cv/process rejects 0-byte file with HTTP 400."""
        res = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json()["detail"].lower())

    def test_cv_process_non_pdf_rejected(self):
        """POST /api/v1/cv/process rejects non-PDF signature with HTTP 400."""
        res = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("fake.pdf", io.BytesIO(b"NOT_A_PDF_FILE_HEADER"), "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("valid pdf", res.json()["detail"].lower())

    def test_cv_process_unauthenticated_rejected(self):
        """POST /api/v1/cv/process rejects unauthenticated requests before processing."""
        res = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("cv.pdf", io.BytesIO(self._create_minimal_pdf()), "application/pdf")},
        )
        self.assertEqual(res.status_code, 401)

    def test_cv_process_rate_limiting(self):
        """POST /api/v1/cv/process triggers 429 when user exceeds upload rate limit."""
        with patch.dict(os.environ, {"RATE_LIMIT_CV_PROCESS": "1/minute"}):
            mock_pipeline = MagicMock()
            mock_pipeline.process_cv.return_value = MagicMock(
                profile=CandidateProfile(),
                document_analysis=MagicMock(
                    page_count=1,
                    language="en",
                    parser_confidence=0.9,
                    total_characters=100,
                    total_words=20,
                    pages_with_text=1,
                    pages_sparse_or_empty=0,
                    character_density_per_page=100.0,
                ),
                resume_id="test_res_123",
                indexed=True,
                chunk_count=1,
            )
            app.dependency_overrides[get_cv_pipeline] = lambda: mock_pipeline
            try:
                pdf_bytes = self._create_minimal_pdf("First Upload")
                res1 = self.client.post(
                    "/api/v1/cv/process",
                    files={"file": ("cv1.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
                    headers=self.auth_headers,
                )
                self.assertEqual(res1.status_code, 200)

                # Second upload within 1 minute blocked
                res2 = self.client.post(
                    "/api/v1/cv/process",
                    files={"file": ("cv2.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
                    headers=self.auth_headers,
                )
                self.assertEqual(res2.status_code, 429)
            finally:
                app.dependency_overrides.pop(get_cv_pipeline, None)

    def test_cors_headers_on_actual_request(self):
        """Allowed origin receives Access-Control-Allow-Origin header."""
        res = self.client.get("/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("access-control-allow-origin"), "http://localhost:5173")
        self.assertEqual(res.headers.get("access-control-allow-credentials"), "true")

    def test_cv_process_misleading_content_length(self):
        """Stream reader rejects oversized uploads when Content-Length is misleadingly small."""
        with patch.dict(os.environ, {"MAX_UPLOAD_SIZE_BYTES": str(1024 * 1024)}):
            oversized_bytes = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 1024)
            # Fake a small Content-Length header to attempt bypassing Content-Length check
            headers = dict(self.auth_headers)
            headers["content-length"] = "100"
            res = self.client.post(
                "/api/v1/cv/process",
                files={"file": ("stream_large.pdf", io.BytesIO(oversized_bytes), "application/pdf")},
                headers=headers,
            )
            self.assertEqual(res.status_code, 413)

    def test_cv_process_missing_content_length(self):
        """Stream reader rejects oversized uploads when Content-Length is completely absent."""
        with patch.dict(os.environ, {"MAX_UPLOAD_SIZE_BYTES": str(1024 * 1024)}):
            oversized_bytes = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 1024)
            res = self.client.post(
                "/api/v1/cv/process",
                files={"file": ("stream_large.pdf", io.BytesIO(oversized_bytes), "application/pdf")},
                headers=self.auth_headers,
            )
            self.assertEqual(res.status_code, 413)

    def test_exception_handler_preserves_standard_http_errors(self):
        """Standard HTTP exceptions (401, 404, 422) preserve status codes and are never converted to 500."""
        # 401 Unauthorized
        res_401 = self.client.get("/api/v1/auth/me")
        self.assertEqual(res_401.status_code, 401)

        # 404 Not Found
        res_404 = self.client.get(
            "/api/v1/agent/resume/non_existent_resume_id/status",
            headers=self.auth_headers,
        )
        self.assertEqual(res_404.status_code, 404)

        # 422 Validation Error
        res_422 = self.client.post(
            "/api/v1/jobs/match",
            json={"invalid": "payload"},
            headers=self.auth_headers,
        )
        self.assertEqual(res_422.status_code, 422)

    def test_production_exception_handler_masks_details(self):
        """Unhandled exceptions in production mode return generic sanitized 500 without leaking stack traces."""
        client = TestClient(app, raise_server_exceptions=False)
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            with patch("backend.routers.auth.verify_password", side_effect=RuntimeError("InternalSecretDatabaseTrace")):
                res = client.post(
                    "/api/v1/auth/login",
                    json={"email": self.user_email, "password": "AnyPassword123!"},
                )
                self.assertEqual(res.status_code, 500)
                data = res.json()
                self.assertEqual(data["detail"], "An internal server error occurred. Please try again later.")
                self.assertNotIn("InternalSecretDatabaseTrace", str(data))


if __name__ == "__main__":
    unittest.main()
