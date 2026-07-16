import json
from pathlib import Path


def test_railway_config_starts_nested_fastapi_app() -> None:
    config = json.loads(Path("railway.json").read_text(encoding="utf-8"))

    assert config["build"]["builder"] == "RAILPACK"
    assert config["deploy"]["startCommand"] == (
        "uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT"
    )
    assert config["deploy"]["healthcheckPath"] == "/health"
