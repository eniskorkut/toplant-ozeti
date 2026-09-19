"""CORS preflight must allow the browser methods the app actually uses.

Regression: meeting deletion (DELETE) and speaker aliases (PUT) failed in the
browser with "Failed to fetch" because the preflight only allowed GET/POST.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

ORIGIN = "http://localhost:3000"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize("method", ["DELETE", "PUT", "POST", "GET"])
def test_preflight_allows_app_methods(client: TestClient, method: str) -> None:
    response = client.options(
        "/api/v1/meetings/abc123",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200, response.text
    allowed = response.headers["access-control-allow-methods"]
    assert method in allowed
    assert response.headers["access-control-allow-origin"] == ORIGIN


def test_preflight_rejects_unknown_origin(client: TestClient) -> None:
    response = client.options(
        "/api/v1/meetings/abc123",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "DELETE",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_preflight_rejects_unlisted_method(client: TestClient) -> None:
    response = client.options(
        "/api/v1/meetings/abc123",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "TRACE",
        },
    )

    assert response.status_code == 400
