import pytest

from backend.app.core.config.settings import Settings


def test_market_data_provider_defaults_to_the_keyless_public_source() -> None:
    assert Settings(_env_file=None, market_data_worker_enabled=False).market_data_provider == "lbma_gold_price"


@pytest.mark.parametrize("provider", ["lbma_gold_price", "kraken", "okx", "yahoo_finance", "stooq", "binance", "twelve_data", "oanda", "alpha_vantage", "financial_modeling_prep"])
def test_market_data_provider_accepts_every_registered_adapter_name(monkeypatch, provider: str) -> None:
    monkeypatch.setenv("TEN_MARKET_DATA_PROVIDER", provider)
    settings = Settings(_env_file=None, market_data_worker_enabled=False)
    assert settings.market_data_provider == provider


def test_market_data_provider_rejects_an_unknown_name(monkeypatch) -> None:
    monkeypatch.setenv("TEN_MARKET_DATA_PROVIDER", "not_a_real_provider")
    with pytest.raises(ValueError, match="TEN_MARKET_DATA_PROVIDER is unsupported"):
        Settings(_env_file=None, market_data_worker_enabled=False)


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


def test_railway_postgresql_url_uses_asyncpg(monkeypatch) -> None:
    monkeypatch.setenv("TEN_DATABASE_URL", "postgresql://postgres:secret@postgres.railway.internal:5432/railway")

    settings = Settings(_env_file=None, market_data_worker_enabled=False)

    assert settings.database_url == (
        "postgresql+asyncpg://postgres:secret@postgres.railway.internal:5432/railway"
    )
