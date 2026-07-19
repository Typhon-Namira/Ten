import json
from pathlib import Path
import tomllib


def test_railway_config_starts_nested_fastapi_app() -> None:
    config = json.loads(Path("railway.json").read_text(encoding="utf-8"))

    assert config["build"]["builder"] == "DOCKERFILE"
    assert config["build"]["dockerfilePath"] == "docker/backend.Dockerfile"
    assert "startCommand" not in config["deploy"]
    assert config["deploy"]["healthcheckPath"] == "/health"

    toml_config = tomllib.loads(Path("railway.toml").read_text(encoding="utf-8"))
    assert "startCommand" not in toml_config["deploy"]

    dockerfile = Path("docker/backend.Dockerfile").read_text(encoding="utf-8")
    assert "FROM node:22-slim AS frontend-build" in dockerfile
    assert "cd frontend && npm install" in dockerfile
    assert "cd frontend && npm run build" in dockerfile
    assert "COPY --from=frontend-build /app/frontend/dist ./frontend/dist" in dockerfile
    assert 'CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port \\\"${PORT:-8000}\\\""]' in dockerfile
