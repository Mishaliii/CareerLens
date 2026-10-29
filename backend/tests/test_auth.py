"""
Unit and integration tests for CareerLens local authentication and JWT security.

Validates:
1. Registration (success, normalization, duplicates, invalid email/password, hash safety).
2. Login (success, invalid password, unknown email, inactive account, malformed hash).
3. JWT token issuance and claim validation (signature, expiry, issuer, audience, missing claims).
4. Protected endpoint /api/v1/auth/me (valid auth, missing token, expired token, safe fields).
"""

from datetime import datetime, timedelta, timezone
import os
import tempfile
import unittest
from fastapi.testclient import TestClient
import jwt
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.db.database import Base, get_db
from backend.db.migrations import run_migrations
from backend.db.models import User
from backend.main import app
from backend.services.auth_service import (
    DEFAULT_ALGORITHM,
    DEFAULT_AUDIENCE,
    DEFAULT_DEV_SECRET,
    DEFAULT_ISSUER,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class TestAuthenticationService(unittest.TestCase):
    """Unit tests for low-level cryptographic functions and JWT verification."""

    def test_hash_and_verify_password_success(self):
        """Valid passwords verify correctly against generated Scrypt hashes."""
        plain = "CorrectHorseBatteryStaple123!"
        hashed = hash_password(plain)

        self.assertTrue(hashed.startswith("scrypt$16384$8$1$"))
        self.assertTrue(verify_password(plain, hashed))
        self.assertFalse(verify_password("WrongPassword123!", hashed))

    def test_verify_password_with_malformed_hashes(self):
        """Malformed or manipulated hashes fail safely without raising exceptions."""
        self.assertFalse(verify_password("pass", ""))
        self.assertFalse(verify_password("pass", "not_a_hash"))
        self.assertFalse(verify_password("pass", "scrypt$invalid$format"))
        self.assertFalse(verify_password("pass", "scrypt$16384$8$1$not_hex$not_hex"))
        self.assertFalse(verify_password("pass", "bcrypt$2b$12$somehashhere"))

    def test_empty_password_hashing_rejected(self):
        """Empty passwords must be rejected before hashing."""
        with self.assertRaises(ValueError):
            hash_password("")

    def test_jwt_issuance_and_decoding(self):
        """Tokens are properly signed with claims and decoded accurately."""
        token = create_access_token(
            user_id="user_123",
            email="test@example.com",
            expires_delta=timedelta(minutes=15),
        )
        payload = decode_access_token(token)

        self.assertEqual(payload["sub"], "user_123")
        self.assertEqual(payload["email"], "test@example.com")
        self.assertEqual(payload["iss"], DEFAULT_ISSUER)
        self.assertEqual(payload["aud"], DEFAULT_AUDIENCE)
        self.assertIn("exp", payload)
        self.assertIn("iat", payload)

    def test_jwt_rejects_expired_token(self):
        """Expired tokens raise 401 HTTPException."""
        token = create_access_token(
            user_id="user_123",
            email="test@example.com",
            expires_delta=timedelta(seconds=-10),  # expired 10 seconds ago
        )
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            decode_access_token(token)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("expired", ctx.exception.detail.lower())

    def test_jwt_rejects_wrong_secret(self):
        """Tokens signed with a different secret are rejected with 401."""
        token = jwt.encode(
            {
                "sub": "user_123",
                "email": "test@example.com",
                "iss": DEFAULT_ISSUER,
                "aud": DEFAULT_AUDIENCE,
                "iat": int(datetime.now(timezone.utc).timestamp()),
                "exp": int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()),
            },
            "unauthorized-secret-key-32-bytes-minimum!",
            algorithm=DEFAULT_ALGORITHM,
        )
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            decode_access_token(token)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("invalid", ctx.exception.detail.lower())

    def test_jwt_rejects_wrong_issuer(self):
        """Tokens with mismatched issuer claim are rejected with 401."""
        token = jwt.encode(
            {
                "sub": "user_123",
                "email": "test@example.com",
                "iss": "rogue-untrusted-issuer",
                "aud": DEFAULT_AUDIENCE,
                "iat": int(datetime.now(timezone.utc).timestamp()),
                "exp": int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()),
            },
            DEFAULT_DEV_SECRET,
            algorithm=DEFAULT_ALGORITHM,
        )
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            decode_access_token(token)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("issuer", ctx.exception.detail.lower())

    def test_jwt_rejects_wrong_audience(self):
        """Tokens with mismatched audience claim are rejected with 401."""
        token = jwt.encode(
            {
                "sub": "user_123",
                "email": "test@example.com",
                "iss": DEFAULT_ISSUER,
                "aud": "wrong-audience",
                "iat": int(datetime.now(timezone.utc).timestamp()),
                "exp": int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()),
            },
            DEFAULT_DEV_SECRET,
            algorithm=DEFAULT_ALGORITHM,
        )
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            decode_access_token(token)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("audience", ctx.exception.detail.lower())

    def test_jwt_rejects_missing_subject(self):
        """Tokens without a subject claim are rejected with 401."""
        token = jwt.encode(
            {
                "email": "test@example.com",
                "iss": DEFAULT_ISSUER,
                "aud": DEFAULT_AUDIENCE,
                "iat": int(datetime.now(timezone.utc).timestamp()),
                "exp": int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()),
            },
            DEFAULT_DEV_SECRET,
            algorithm=DEFAULT_ALGORITHM,
        )
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            decode_access_token(token)
        self.assertEqual(ctx.exception.status_code, 401)


class TestAuthenticationEndpoints(unittest.TestCase):
    """Integration tests for registration, login, and protected routes using TestClient."""

    def setUp(self):
        # Create isolated temporary SQLite database for each test
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db_file.name
        self.temp_db_file.close()

        self.db_url = f"sqlite:///{self.temp_db_path}"
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragmas(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA busy_timeout = 5000;")
            cursor.close()

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False,
        )

        run_migrations(self.engine)

        # Override get_db dependency to point to the isolated temporary DB
        def override_get_db():
            db = self.SessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.engine.dispose()
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except PermissionError:
                pass

    def test_register_user_success(self):
        """User can register with valid credentials and safe profile is returned."""
        payload = {
            "email": "candidate.one@example.com",
            "password": "SecurePassword123!",
            "full_name": "Candidate One",
        }
        resp = self.client.post("/api/v1/auth/register", json=payload)
        self.assertEqual(resp.status_code, 201)

        data = resp.json()
        self.assertEqual(data["email"], "candidate.one@example.com")
        self.assertEqual(data["full_name"], "Candidate One")
        self.assertTrue(data["is_active"])
        self.assertIn("id", data)
        # CRITICAL SECURITY: Never return password hashes in responses
        self.assertNotIn("password", data)
        self.assertNotIn("hashed_password", data)

        # Verify password is encrypted in database
        db = self.SessionLocal()
        user_row = db.query(User).filter_by(email="candidate.one@example.com").first()
        db.close()
        self.assertIsNotNone(user_row)
        self.assertTrue(user_row.hashed_password.startswith("scrypt$"))

    def test_register_email_normalization(self):
        """Registration trims whitespace and lowercases email before saving."""
        payload = {
            "email": "   User.MIXED@Domain.COM   ",
            "password": "Password123!",
            "full_name": "Mixed Case",
        }
        resp = self.client.post("/api/v1/auth/register", json=payload)
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["email"], "user.mixed@domain.com")

        # Duplicate check also respects normalization
        resp_dup = self.client.post("/api/v1/auth/register", json=payload)
        self.assertEqual(resp_dup.status_code, 400)
        self.assertIn("already exists", resp_dup.json()["detail"])

    def test_register_rejects_invalid_email_format(self):
        """Invalid email formats fail 422 Unprocessable Entity."""
        invalid_emails = ["notanemail", "missing_domain@", "@missinguser.com", "spaces in@email.com"]
        for email in invalid_emails:
            resp = self.client.post("/api/v1/auth/register", json={"email": email, "password": "Password123!"})
            self.assertEqual(resp.status_code, 422)

    def test_register_rejects_short_or_blank_password(self):
        """Passwords shorter than 8 chars or all whitespace fail validation."""
        # Too short (<8)
        resp1 = self.client.post("/api/v1/auth/register", json={"email": "u1@ex.com", "password": "short"})
        self.assertEqual(resp1.status_code, 422)

        # Blank / Whitespace
        resp2 = self.client.post("/api/v1/auth/register", json={"email": "u2@ex.com", "password": "        "})
        self.assertEqual(resp2.status_code, 422)

    def test_login_success(self):
        """User can login and receive signed JWT access token."""
        # 1. Register
        self.client.post(
            "/api/v1/auth/register",
            json={"email": "login.test@example.com", "password": "ValidPassword123!"},
        )

        # 2. Login
        login_resp = self.client.post(
            "/api/v1/auth/login",
            json={"email": "login.test@example.com", "password": "ValidPassword123!"},
        )
        self.assertEqual(login_resp.status_code, 200)

        data = login_resp.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")
        self.assertGreater(data["expires_in"], 0)
        self.assertEqual(data["user"]["email"], "login.test@example.com")
        self.assertNotIn("password", data)
        self.assertNotIn("hashed_password", data)

    def test_login_incorrect_password_returns_generic_401(self):
        """Incorrect password returns generic 401 to prevent timing/enum attacks."""
        self.client.post(
            "/api/v1/auth/register",
            json={"email": "pw.test@example.com", "password": "CorrectPassword123!"},
        )
        resp = self.client.post(
            "/api/v1/auth/login",
            json={"email": "pw.test@example.com", "password": "WrongPassword123!"},
        )
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Invalid email or password", resp.json()["detail"])

    def test_login_unknown_email_returns_generic_401(self):
        """Nonexistent email returns identical generic 401 error."""
        resp = self.client.post(
            "/api/v1/auth/login",
            json={"email": "nonexistent.user@example.com", "password": "AnyPassword123!"},
        )
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Invalid email or password", resp.json()["detail"])

    def test_login_inactive_user_rejected(self):
        """Inactive user account cannot authenticate and returns 403."""
        # Register user
        reg_resp = self.client.post(
            "/api/v1/auth/register",
            json={"email": "inactive@example.com", "password": "Password123!"},
        )
        user_id = reg_resp.json()["id"]

        # Deactivate user directly in DB
        db = self.SessionLocal()
        user = db.query(User).filter_by(id=user_id).first()
        user.is_active = False
        db.commit()
        db.close()

        # Attempt login
        login_resp = self.client.post(
            "/api/v1/auth/login",
            json={"email": "inactive@example.com", "password": "Password123!"},
        )
        self.assertEqual(login_resp.status_code, 403)
        self.assertIn("inactive", login_resp.json()["detail"].lower())

    def test_get_current_user_profile_me_endpoint(self):
        """Protected /me endpoint returns profile when supplied valid Bearer token."""
        # Register and login
        self.client.post(
            "/api/v1/auth/register",
            json={"email": "me.test@example.com", "password": "Password123!", "full_name": "Me User"},
        )
        login_resp = self.client.post(
            "/api/v1/auth/login",
            json={"email": "me.test@example.com", "password": "Password123!"},
        )
        token = login_resp.json()["access_token"]

        # 1. Success with Bearer token
        headers = {"Authorization": f"Bearer {token}"}
        me_resp = self.client.get("/api/v1/auth/me", headers=headers)
        self.assertEqual(me_resp.status_code, 200)
        data = me_resp.json()
        self.assertEqual(data["email"], "me.test@example.com")
        self.assertEqual(data["full_name"], "Me User")
        self.assertTrue(data["is_active"])
        self.assertNotIn("password", data)

        # 2. Missing Authorization header returns 401
        resp_no_auth = self.client.get("/api/v1/auth/me")
        self.assertEqual(resp_no_auth.status_code, 401)

        # 3. Invalid token returns 401
        resp_bad_auth = self.client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not_a_real_token"})
        self.assertEqual(resp_bad_auth.status_code, 401)

    def test_get_current_user_rejects_deleted_or_inactive_user(self):
        """Valid token for deleted user returns 401; for deactivated user returns 403."""
        reg_resp = self.client.post(
            "/api/v1/auth/register",
            json={"email": "temp.user@example.com", "password": "Password123!"},
        )
        user_id = reg_resp.json()["id"]

        token = create_access_token(user_id=user_id, email="temp.user@example.com")
        headers = {"Authorization": f"Bearer {token}"}

        # Verify works initially
        self.assertEqual(self.client.get("/api/v1/auth/me", headers=headers).status_code, 200)

        # Deactivate user
        db = self.SessionLocal()
        user = db.query(User).filter_by(id=user_id).first()
        user.is_active = False
        db.commit()
        db.close()

        # Inactive returns 403
        self.assertEqual(self.client.get("/api/v1/auth/me", headers=headers).status_code, 403)

        # Delete user
        db = self.SessionLocal()
        user = db.query(User).filter_by(id=user_id).first()
        db.delete(user)
        db.commit()
        db.close()

        # Deleted returns 401
        self.assertEqual(self.client.get("/api/v1/auth/me", headers=headers).status_code, 401)


if __name__ == "__main__":
    unittest.main()
