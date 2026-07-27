import pytest

from backend.app.core.config.settings import Settings
from backend.app.core.feature_flags import FeatureFlag
from backend.app.services import build_engine_registry


def test_ai_provider_defaults_are_four_account_groq_pool() -> None:
    settings = Settings(_env_file=None, market_data_worker_enabled=False)

    assert settings.ai_primary_provider == "groq"
    assert settings.groq_pool_enabled is True
    assert settings.groq_pool_size == 4
    assert settings.groq_base_url == "https://api.groq.com/openai/v1"
    assert settings.groq_model == "gpt-oss-120b"
    assert settings.groq_request_timeout_seconds == 60
    assert settings.groq_max_retries_per_account == 1
    assert settings.groq_rate_limit_cooldown_seconds == 3600
    assert settings.groq_quota_cooldown_seconds == 86400
    assert settings.groq_pool_strategy == "ordered_failover"
    assert settings.groq_pool_api_keys == (None, None, None, None)


@pytest.mark.parametrize(
    "base_url",
    (
        "https://api.groq.com/openai/v1/v1",
        "https://api.groq.com/openai/v1/chat/completions",
    ),
)
def test_groq_base_url_rejects_endpoint_path_misconfiguration(
    base_url: str,
) -> None:
    with pytest.raises(ValueError, match="must be a base URL"):
        Settings(
            _env_file=None,
            groq_base_url=base_url,
            market_data_worker_enabled=False,
        )


@pytest.mark.parametrize("api_key", (" key", "key ", '"key"', "'key'"))
def test_groq_pool_keys_reject_shell_quoting_and_whitespace(api_key: str) -> None:
    with pytest.raises(ValueError, match="must not contain surrounding"):
        Settings(
            _env_file=None,
            groq_api_key_1=api_key,
            market_data_worker_enabled=False,
        )


def test_legacy_groq_key_maps_only_to_account_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="legacy-key",
        market_data_worker_enabled=False,
    )

    assert settings.groq_pool_api_keys == ("legacy-key", None, None, None)
    assert "ai_provider.legacy_key_alias.used" in caplog.text


def test_explicit_account_one_key_has_priority_over_legacy_alias() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="legacy-key",
        groq_api_key_1="account-one-key",
        market_data_worker_enabled=False,
    )

    assert settings.groq_pool_api_keys == ("account-one-key", None, None, None)


def test_application_settings_have_no_removed_provider_dependency() -> None:
    removed_provider = "cere" + "bras"
    settings = Settings(_env_file=None, market_data_worker_enabled=False)

    assert all(
        removed_provider not in field_name
        for field_name in type(settings).model_fields
    )


def test_every_groq_pool_environment_variable_uses_ten_prefix() -> None:
    fields = (
        "ai_primary_provider",
        "groq_pool_enabled",
        "groq_pool_size",
        "groq_api_key_1",
        "groq_api_key_2",
        "groq_api_key_3",
        "groq_api_key_4",
        "groq_base_url",
        "groq_model",
        "groq_request_timeout_seconds",
        "groq_max_retries_per_account",
        "groq_rate_limit_cooldown_seconds",
        "groq_quota_cooldown_seconds",
        "groq_pool_strategy",
    )

    environment_variables = tuple(f"TEN_{field.upper()}" for field in fields)
    assert all(name.startswith("TEN_") for name in environment_variables)
    assert len(environment_variables) == len(fields)


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


def test_ai_feature_flag_environment_overrides_are_explicit_and_independent() -> None:
    settings = Settings(
        _env_file=None,
        ai_centric_shadow_mode=True,
        ai_signal_proposals=True,
        ai_signal_monitoring=True,
        ai_signal_publication=False,
        ai_signal_adjustments=False,
        market_data_worker_enabled=False,
    )
    flags = build_engine_registry(settings=settings).context.feature_flags

    assert flags.is_enabled(FeatureFlag.AI_CENTRIC_SHADOW_MODE) is True
    assert flags.is_enabled(FeatureFlag.AI_SIGNAL_PROPOSALS) is True
    assert flags.is_enabled(FeatureFlag.AI_SIGNAL_MONITORING) is True
    assert flags.is_enabled(FeatureFlag.AI_SIGNAL_PUBLICATION) is False
    assert flags.is_enabled(FeatureFlag.AI_SIGNAL_ADJUSTMENTS) is False
