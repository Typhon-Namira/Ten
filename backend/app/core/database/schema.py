"""Database schema lifecycle boundaries."""

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncConnection

from .base import Base


SCHEMA_HEAD_REVISION = "20260804_0024"


async def prepare_database_schema(connection: AsyncConnection, *, managed_runtime: bool) -> None:
    """Create development tables or verify Alembic owns the managed schema."""
    if not managed_runtime:
        await connection.run_sync(Base.metadata.create_all)
        return

    has_version_table = await connection.run_sync(
        lambda sync_connection: inspect(sync_connection).has_table("alembic_version")
    )
    if not has_version_table:
        raise RuntimeError(
            f"database schema is not migrated to {SCHEMA_HEAD_REVISION}; "
            "run `python -m alembic upgrade head` before starting TEN"
        )

    result = await connection.execute(text("SELECT version_num FROM alembic_version"))
    revision = result.scalar_one_or_none()
    if revision != SCHEMA_HEAD_REVISION:
        raise RuntimeError(
            f"database schema is not migrated to {SCHEMA_HEAD_REVISION}; "
            "run `python -m alembic upgrade head` before starting TEN"
        )
