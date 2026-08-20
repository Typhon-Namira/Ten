"""Alembic environment for TEN's asynchronous SQLAlchemy database."""

from asyncio import run
from logging.config import fileConfig
import os
import socket
from urllib.parse import urlparse

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from backend.app.core.database.base import Base
from backend.app.core.database.url import normalize_async_database_url
import backend.app.storage.models  # noqa: F401 -- registers every table with Base.metadata


config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def database_url() -> str:
    """Return the managed-runtime async database URL without logging secrets.

    ``TEN_DATABASE_URL`` is canonical. ``DATABASE_URL`` is accepted as a
    Railway/PostgreSQL compatibility fallback so pre-deploy migrations and the
    application runtime resolve the same database even when the platform only
    exposes its generic variable.
    """
    source = "TEN_DATABASE_URL" if os.environ.get("TEN_DATABASE_URL") else "DATABASE_URL"
    raw_url = os.environ.get("TEN_DATABASE_URL") or os.environ.get("DATABASE_URL")
    url = normalize_async_database_url(raw_url) if raw_url else None
    if not url:
        raise RuntimeError(
            "TEN_DATABASE_URL or DATABASE_URL is required to run database migrations"
        )
    if not url.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
        raise RuntimeError(
            "database URL must use postgresql+asyncpg:// or sqlite+aiosqlite://"
        )

    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname and parsed.scheme.startswith("postgresql"):
        try:
            socket.getaddrinfo(hostname, parsed.port or 5432)
        except socket.gaierror as exc:
            raise RuntimeError(
                "database hostname could not be resolved during migration: "
                f"source={source}, host={hostname!r}. "
                "On Railway, configure TEN_DATABASE_URL or DATABASE_URL as a reference "
                "to the PostgreSQL service in the same project environment, for example "
                "${{Postgres.DATABASE_URL}} using the exact service name."
            ) from exc
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = database_url()
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run(run_migrations_online())
