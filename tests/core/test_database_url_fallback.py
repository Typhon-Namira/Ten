from backend.app.core.config.settings import Settings
from backend.app.core.database.url import normalize_async_database_url


def test_settings_falls_back_to_railway_database_url(monkeypatch) -> None:
    monkeypatch.delenv("TEN_DATABASE_URL", raising=False)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://postgres:secret@postgres.railway.internal:5432/railway",
    )

    settings = Settings(_env_file=None, market_data_worker_enabled=False)

    assert settings.database_url == (
        "postgresql+asyncpg://postgres:secret@postgres.railway.internal:5432/railway"
    )


def test_ten_database_url_keeps_precedence_over_railway_fallback(monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://fallback:secret@fallback.internal:5432/railway",
    )
    monkeypatch.setenv(
        "TEN_DATABASE_URL",
        "postgresql://canonical:secret@canonical.internal:5432/ten",
    )

    settings = Settings(_env_file=None, market_data_worker_enabled=False)

    assert settings.database_url == (
        "postgresql+asyncpg://canonical:secret@canonical.internal:5432/ten"
    )


def test_native_railway_postgres_url_normalizes_to_asyncpg() -> None:
    assert normalize_async_database_url("postgres://u:p@host:5432/db") == (
        "postgresql+asyncpg://u:p@host:5432/db"
    )
