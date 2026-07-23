"""PostgreSQL-only regression coverage for market-regime snapshot idempotency."""

import asyncio
from datetime import UTC, datetime, timedelta
import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.market_regime_engine import BaselineMarketRegimeAnalyzer, MarketRegimeContext, MarketRegimeSnapshot, SqlAlchemyMarketRegimeRepository


def _snapshot() -> MarketRegimeSnapshot:
    start = datetime(2026, 7, 23, 8, tzinfo=UTC)
    candles = tuple(
        Candle(
            timestamp=start + timedelta(minutes=15 * index),
            ingestion_timestamp=start + timedelta(minutes=15 * index),
            symbol="XAUUSD",
            timeframe=Timeframe.M15,
            open=3300 + index,
            high=3302 + index,
            low=3298 + index,
            close=3301 + index,
            volume=100 + index,
            provider="postgresql-regression",
        )
        for index in range(40)
    )
    return BaselineMarketRegimeAnalyzer().analyze_snapshot(MarketRegimeContext(candles))


@pytest.mark.asyncio
async def test_postgresql_snapshot_boundary_is_idempotent_and_concurrent_safe() -> None:
    database_url = os.getenv("TEN_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEN_TEST_DATABASE_URL is required for PostgreSQL conflict behavior")
    if not database_url.startswith("postgresql+asyncpg://"):
        pytest.fail("TEN_TEST_DATABASE_URL must use postgresql+asyncpg://")
    if database_url == os.getenv("TEN_DATABASE_URL"):
        pytest.fail("TEN_TEST_DATABASE_URL must not be the configured production database")

    schema = f"ten_market_regime_{uuid4().hex}"
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
                    CREATE TABLE market_regime_snapshots (
                        id uuid PRIMARY KEY,
                        symbol varchar(32) NOT NULL,
                        timeframe varchar(16) NOT NULL,
                        analysis_timestamp timestamptz NOT NULL,
                        dominant_regime varchar(64) NOT NULL,
                        configuration_version varchar(32) NOT NULL,
                        engine_version varchar(32) NOT NULL,
                        payload jsonb NOT NULL,
                        created_at timestamptz NOT NULL
                    )
                    """
                )
            )
            await connection.execute(
                text(
                    """
                    CREATE UNIQUE INDEX ux_market_regime_snapshot_boundary
                    ON market_regime_snapshots (
                        symbol,
                        timeframe,
                        analysis_timestamp,
                        configuration_version
                    )
                    """
                )
            )

        repository = SqlAlchemyMarketRegimeRepository(async_sessionmaker(engine, expire_on_commit=False))
        snapshot = _snapshot()
        same_boundary = snapshot.model_copy(update={"snapshot_id": uuid4()})

        await repository.save_snapshot(snapshot)
        await repository.save_snapshot(snapshot)
        await asyncio.gather(
            repository.save_snapshot(snapshot),
            repository.save_snapshot(same_boundary),
        )

        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM market_regime_snapshots")) == 1
            assert await connection.scalar(text("SELECT id FROM market_regime_snapshots")) == snapshot.snapshot_id
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin.dispose()
