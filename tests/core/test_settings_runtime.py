from backend.app.core.config.settings import Settings


def test_railway_runtime_enables_pipeline_workers_by_default(monkeypatch) -> None:
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    settings = Settings(_env_file=None)
    assert settings.market_data_worker_enabled is True
    assert settings.integration_worker_enabled is True


def test_railway_runtime_respects_explicit_worker_disable(monkeypatch) -> None:
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    settings = Settings(_env_file=None, market_data_worker_enabled=False, integration_worker_enabled=False)
    assert settings.market_data_worker_enabled is False
    assert settings.integration_worker_enabled is False
