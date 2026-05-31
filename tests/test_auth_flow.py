from pathlib import Path

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import milk_dashboard.api.app as api_app


def _configure_auth_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APP_AUTH_SECRET", "test-auth-secret")
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode("utf-8"))
    monkeypatch.setenv("MILK_AUTH_DB_PATH", str(tmp_path / "auth.db"))
    api_app._auth_store.cache_clear()


def test_register_login_and_me(monkeypatch, tmp_path: Path):
    _configure_auth_env(monkeypatch, tmp_path)
    client = TestClient(api_app.app)

    register = client.post(
        "/api/auth/register",
        json={"email": "owner@example.com", "password": "strong-pass-123"},
    )
    assert register.status_code == 200
    assert register.json()["user"]["email"] == "owner@example.com"

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    payload = me.json()
    assert payload["authenticated"] is True
    assert payload["user"]["email"] == "owner@example.com"
    assert payload["square_connection"] is None

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200

    me_after_logout = client.get("/api/auth/me")
    assert me_after_logout.status_code == 200
    assert me_after_logout.json()["authenticated"] is False

    login = client.post(
        "/api/auth/login",
        json={"email": "owner@example.com", "password": "strong-pass-123"},
    )
    assert login.status_code == 200
    assert login.json()["status"] == "authenticated"


def test_square_connection_is_encrypted_at_rest(monkeypatch, tmp_path: Path):
    _configure_auth_env(monkeypatch, tmp_path)
    store = api_app._auth_store()
    user = store.create_user(email="owner@example.com", password="strong-pass-123")
    store.upsert_square_connection(
        user_id=user.id,
        token_payload={
            "merchant_id": "merchant-123",
            "access_token": "access-secret",
            "refresh_token": "refresh-secret",
            "token_type": "bearer",
            "scope": "ORDERS_READ",
            "expires_at": "2026-06-01T00:00:00Z",
        },
        square_env="production",
    )

    connection_public = store.get_square_connection_by_user(user.id, include_secrets=False)
    connection_private = store.get_square_connection_by_user(user.id, include_secrets=True)

    assert connection_public is not None
    assert connection_public["merchant_id"] == "merchant-123"
    assert "access_token" not in connection_public
    assert connection_private is not None
    assert connection_private["access_token"] == "access-secret"
    assert connection_private["refresh_token"] == "refresh-secret"
