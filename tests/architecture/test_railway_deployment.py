import json
from pathlib import Path


def test_railway_config_starts_nested_fastapi_app() -> None:
    config = json.loads(Path("railway.json").read_text(encoding="utf-8"))

    assert config["build"]["builder"] == "DOCKERFILE"
    assert config["build"]["dockerfilePath"] == "docker/backend.Dockerfile"
    assert config["deploy"]["startCommand"] == (
        "uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT"
    )
    assert config["deploy"]["healthcheckPath"] == "/health"

    dockerfile = Path("docker/backend.Dockerfile").read_text(encoding="utf-8")
    assert "FROM node:22-slim AS frontend-build" in dockerfile
    assert "cd frontend && npm install" in dockerfile
    assert "cd frontend && npm run build" in dockerfile
    assert "COPY --from=frontend-build /app/frontend/dist ./frontend/dist" in dockerfile
