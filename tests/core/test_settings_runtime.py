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


def test_market_data_sequences_accept_railway_csv_values(monkeypatch) -> None:
    monkeypatch.setenv("TEN_MARKET_DATA_SYMBOLS", "XAUUSD, EURUSD")
    monkeypatch.setenv("TEN_MARKET_DATA_TIMEFRAMES", "M15,H1")

    settings = Settings(_env_file=None, market_data_worker_enabled=False)

    assert settings.market_data_symbols == ("XAUUSD", "EURUSD")
    assert settings.market_data_timeframes == ("M15", "H1")


def test_market_data_sequences_still_accept_json_arrays(monkeypatch) -> None:
    monkeypatch.setenv("TEN_MARKET_DATA_SYMBOLS", '["XAUUSD", "EURUSD"]')
    monkeypatch.setenv("TEN_MARKET_DATA_TIMEFRAMES", '["M15", "H1"]')

    settings = Settings(_env_file=None, market_data_worker_enabled=False)

    assert settings.market_data_symbols == ("XAUUSD", "EURUSD")
    assert settings.market_data_timeframes == ("M15", "H1")


def test_market_data_timeframes_normalize_railway_provider_notation(monkeypatch) -> None:
    monkeypatch.setenv("TEN_MARKET_DATA_TIMEFRAMES", "1m,5m,15m,1h")

    settings = Settings(_env_file=None, market_data_worker_enabled=False)

    assert settings.market_data_timeframes == ("M1", "M5", "M15", "H1")
