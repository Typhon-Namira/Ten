from typing import Any, cast

from fastapi.testclient import TestClient

from backend.app.api.dependencies import get_volume_profile_service
from backend.app.engines.volume_profile_engine import InMemoryVolumeProfileRepository, VolumeProfileConfig, VolumeProfileService
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore
from backend.app.main import create_app
from tests.engines.volume_profile_engine.test_volume_profile_production import Liquidity, Market, SMC, series


def test_all_volume_profile_endpoints_are_bounded_read_only_and_secret_free() -> None:
    service = VolumeProfileService(
        cast(Any, Market(series())),
        SMC(),
        Liquidity(),
        InMemoryEventBus(),
        InMemoryFeatureStore(),
        VolumeProfileConfig(default_volume_source="exchange"),
        InMemoryVolumeProfileRepository(),
    )
    app = create_app()
    app.dependency_overrides[get_volume_profile_service] = lambda: service
    with TestClient(app) as client:
        paths = (
            "health",
            "metrics",
            "config",
            "state",
            "snapshot",
            "profiles",
            "developing",
            "completed",
            "sessions",
            "daily",
            "weekly",
            "monthly",
            "composite",
            "anchored",
            "poc",
            "value-area",
            "hvn",
            "lvn",
            "shelves",
            "gaps",
            "shapes",
            "migrations",
            "confluences",
            "mtf",
        )
        assert all(client.get(f"/volume-profile/{path}").status_code == 200 for path in paths)
        candles = series()
        start, end = candles[0].timestamp.isoformat(), candles[-1].timestamp.isoformat()
        assert client.get("/volume-profile/fixed-range", params={"start": start, "end": end}).status_code == 200
        assert client.get("/volume-profile/fixed-range", params={"start": end, "end": start}).status_code == 422
        assert client.get("/volume-profile/fixed-range", params={"start": candles[0].timestamp.replace(year=2024).isoformat(), "end": end}).status_code == 422
        assert client.get("/volume-profile/profiles", params={"limit": 5001}).status_code == 422
        assert client.get("/volume-profile/profiles", params={"min_quality": 101}).status_code == 422
        assert client.get("/volume-profile/profiles", params={"completed_only": True, "lifecycle_state": "completed"}).status_code == 200
        filtered = client.get("/volume-profile/profiles", params={"profile_type": "fixed_range", "tested": False, "start": start, "end": end})
        assert filtered.status_code == 200 and all(x["profile_type"] == "fixed_range" for x in filtered.json())
        assert client.post("/volume-profile/profiles").status_code == 405
        assert "database_url" not in client.get("/volume-profile/config").text.lower()
