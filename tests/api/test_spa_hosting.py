from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def frontend_build(tmp_path: Path) -> Path:
    dist = tmp_path / "frontend" / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><html><body><div id='root'>TEN Dashboard</div></body></html>", encoding="utf-8")
    (assets / "app.js").write_text("globalThis.TEN = true", encoding="utf-8")
    return dist


def test_dashboard_spa_and_api_share_one_fastapi_origin(tmp_path: Path) -> None:
    app = create_app(frontend_dist=frontend_build(tmp_path))
    with TestClient(app) as client:
        root = client.get("/", headers={"accept": "text/html"})
        spa_route = client.get("/market", headers={"accept": "text/html"})
        colliding_spa_route = client.get("/signals", headers={"accept": "text/html"})
        health = client.get("/health")
        docs = client.get("/docs")
        signals_api = client.get("/signals", headers={"accept": "application/json"})
        unknown_api = client.get("/api/not-real", headers={"accept": "application/json"})
        asset = client.get("/assets/app.js")

    for response in (root, spa_route, colliding_spa_route):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "TEN Dashboard" in response.text
    assert health.status_code == 200 and health.headers["content-type"].startswith("application/json")
    assert docs.status_code == 200 and "swagger-ui" in docs.text
    assert signals_api.status_code == 200 and signals_api.json() == []
    assert unknown_api.status_code == 404 and unknown_api.json() == {"detail": "Not Found"}
    assert asset.status_code == 200 and "globalThis.TEN" in asset.text


def test_production_fails_clearly_without_frontend_build(tmp_path: Path) -> None:
    settings = Settings(
        environment="production",
        public_read_access=False,
        api_keys={"test-admin-key": "admin"},
        cors_origins=["https://ten.example"],
    )
    with pytest.raises(RuntimeError, match=r"frontend[/\\]dist[/\\]index.html"):
        create_app(frontend_dist=tmp_path / "frontend" / "dist", settings_override=settings)
