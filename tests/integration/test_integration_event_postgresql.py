"""PostgreSQL-only regression coverage for integration event FK ordering."""

import asyncio
from datetime import UTC, datetime, timedelta
import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.integration import CanonicalEventEnvelope, MissingIntegrationEventError, SqlAlchemyIntegrationRepository


@pytest.mark.asyncio
async def test_postgresql_processed_marker_requires_visible_parent_and_is_concurrent_safe() -> None:
    database_url = os.getenv("TEN_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEN_TEST_DATABASE_URL is required for PostgreSQL FK behavior")
    if not database_url.startswith("postgresql+asyncpg://"):
        pytest.fail("TEN_TEST_DATABASE_URL must use postgresql+asyncpg://")
    if database_url == os.getenv("TEN_DATABASE_URL"):
        pytest.fail("TEN_TEST_DATABASE_URL must not be the configured production database")

    schema = f"ten_integration_fk_{uuid4().hex}"
    engine = create_async_engine(
        database_url,
        connect_args={"server_settings": {"search_path": schema}},
    )
    admin = create_async_engine(database_url)
    try:
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    CREATE TABLE integration_events (
                        event_id varchar(64) PRIMARY KEY,
                        event_type varchar(96) NOT NULL,
                        trace_id uuid NOT NULL,
                        correlation_id uuid NOT NULL,
                        mode varchar(16) NOT NULL,
                        instrument varchar(32),
                        timeframe varchar(16),
                        occurred_at timestamptz NOT NULL,
                        available_at timestamptz NOT NULL,
                        payload_hash varchar(64) NOT NULL,
                        payload jsonb NOT NULL
                    )
                    """
                )
            )
            await connection.execute(
                text(
                    """
                    CREATE TABLE integration_processed_events (
                        event_id varchar(64) PRIMARY KEY
                            REFERENCES integration_events(event_id) ON DELETE CASCADE,
                        processed_at timestamptz NOT NULL
                    )
                    """
                )
            )

        repository = SqlAlchemyIntegrationRepository(async_sessionmaker(engine, expire_on_commit=False))
        missing_event_id = "f" * 64
        with pytest.raises(MissingIntegrationEventError, match=missing_event_id):
            await repository.mark_processed(missing_event_id)

        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM integration_processed_events")) == 0

        now = datetime(2026, 7, 23, 12, tzinfo=UTC)
        candle = Candle(
            timestamp=now - timedelta(minutes=15),
            ingestion_timestamp=now,
            symbol="XAUUSD",
            timeframe=Timeframe.M15,
            open=3300,
            high=3302,
            low=3298,
            close=3301,
            volume=10,
            provider="test",
        )
        envelope = CanonicalEventEnvelope.historical_candle(candle, uuid4(), now)
        await repository.persist_event(envelope)
        await asyncio.gather(
            repository.mark_processed(envelope.event_id),
            repository.mark_processed(envelope.event_id),
        )

        async with engine.connect() as connection:
            assert await connection.scalar(
                text("SELECT count(*) FROM integration_events WHERE event_id = :event_id"),
                {"event_id": envelope.event_id},
            ) == 1
            assert await connection.scalar(
                text("SELECT count(*) FROM integration_processed_events WHERE event_id = :event_id"),
                {"event_id": envelope.event_id},
            ) == 1
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin.dispose()
