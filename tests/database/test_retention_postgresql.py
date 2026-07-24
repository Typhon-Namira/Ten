"""PostgreSQL-only regression coverage for backend.app.core.database.retention.

Mirrors the schema-per-test pattern used by
tests/engines/market_regime_engine/test_market_regime_postgresql.py: skips unless
TEN_TEST_DATABASE_URL is configured, since the retention deletes rely on real PostgreSQL foreign
key enforcement (ON DELETE CASCADE/RESTRICT) that SQLite does not reproduce faithfully.
"""

from datetime import UTC, datetime, timedelta
import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.core.database.base import Base
from backend.app.core.database.retention import RetentionRepository
from backend.app.storage.models import (
    AIScoreSnapshotRecord,
    IntegrationEventRecord,
    IntegrationOutboxRecord,
    IntegrationSnapshotRecord,
    OperationalSignalRecord,
    SignalDecisionRecord,
    SMCObjectRecord,
)


@pytest.fixture
async def _schema():
    database_url = os.getenv("TEN_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEN_TEST_DATABASE_URL is required for PostgreSQL foreign-key behavior")
    if not database_url.startswith("postgresql+asyncpg://"):
        pytest.fail("TEN_TEST_DATABASE_URL must use postgresql+asyncpg://")
    if database_url == os.getenv("TEN_DATABASE_URL"):
        pytest.fail("TEN_TEST_DATABASE_URL must not be the configured production database")

    schema = f"ten_retention_{uuid4().hex}"
    engine = create_async_engine(database_url, connect_args={"server_settings": {"search_path": schema}})
    admin = create_async_engine(database_url)
    try:
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        async with engine.begin() as connection:
            await connection.execute(text(f'SET search_path TO "{schema}"'))
            await connection.run_sync(Base.metadata.create_all)
        yield engine
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin.dispose()


@pytest.mark.asyncio
async def test_prune_deletes_old_analytical_objects_and_keeps_recent(_schema) -> None:
    engine = _schema
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime(2026, 7, 24, tzinfo=UTC)
    old, recent = uuid4(), uuid4()
    async with session_factory() as session:
        session.add_all(
            [
                SMCObjectRecord(id=old, object_type="zone", symbol="XAUUSD", timeframe="M15", analytical_timestamp=now, availability_timestamp=now, lifecycle_state="invalidated", confidence_score=50.0, quality_score=90.0, algorithm_version="3.0.0", configuration_version="cfg", payload={}, created_at=now - timedelta(days=30)),
                SMCObjectRecord(id=recent, object_type="zone", symbol="XAUUSD", timeframe="M15", analytical_timestamp=now, availability_timestamp=now, lifecycle_state="active", confidence_score=50.0, quality_score=90.0, algorithm_version="3.0.0", configuration_version="cfg", payload={}, created_at=now - timedelta(days=1)),
            ]
        )
        await session.commit()

    repository = RetentionRepository(session_factory)
    deleted = await repository.prune(
        now=now,
        analytical_object_retention_days=14,
        analytical_snapshot_retention_days=14,
        integration_audit_retention_days=14,
        operational_signal_retention_days=180,
        market_data_history_retention_days=7,
        batch_size=500,
    )
    assert deleted["smc_objects"] == 1

    async with engine.connect() as connection:
        remaining = (await connection.execute(text("SELECT id FROM smc_objects"))).scalars().all()
    assert remaining == [recent]


@pytest.mark.asyncio
async def test_prune_integration_events_skips_events_with_unpublished_outbox_items(_schema) -> None:
    engine = _schema
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime(2026, 7, 24, tzinfo=UTC)
    old_available = now - timedelta(days=30)

    published_event_id, pending_event_id = "event-published", "event-pending"
    async with session_factory() as session:
        session.add_all(
            [
                IntegrationEventRecord(event_id=published_event_id, event_type="market.candle.closed", trace_id=uuid4(), correlation_id=uuid4(), mode="live", instrument="XAUUSD", timeframe="M15", occurred_at=old_available, available_at=old_available, payload_hash="h1", payload={}),
                IntegrationEventRecord(event_id=pending_event_id, event_type="market.candle.closed", trace_id=uuid4(), correlation_id=uuid4(), mode="live", instrument="XAUUSD", timeframe="M15", occurred_at=old_available, available_at=old_available, payload_hash="h2", payload={}),
            ]
        )
        await session.commit()
        session.add_all(
            [
                IntegrationOutboxRecord(outbox_id=uuid4(), event_id=published_event_id, available_at=old_available, attempts=1, published_at=old_available),
                IntegrationOutboxRecord(outbox_id=uuid4(), event_id=pending_event_id, available_at=old_available, attempts=0, published_at=None),
            ]
        )
        await session.commit()

    repository = RetentionRepository(session_factory)
    deleted = await repository.prune(
        now=now,
        analytical_object_retention_days=14,
        analytical_snapshot_retention_days=14,
        integration_audit_retention_days=14,
        operational_signal_retention_days=180,
        market_data_history_retention_days=7,
        batch_size=500,
    )
    assert deleted["integration_events"] == 1

    async with engine.connect() as connection:
        remaining = (await connection.execute(text("SELECT event_id FROM integration_events"))).scalars().all()
    assert remaining == [pending_event_id]


@pytest.mark.asyncio
async def test_prune_integration_snapshots_skips_snapshots_referenced_by_operational_signals(_schema) -> None:
    engine = _schema
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime(2026, 7, 24, tzinfo=UTC)
    old_boundary = now - timedelta(days=30)

    referenced_id, orphaned_id = uuid4(), uuid4()
    ai_score_id, decision_id = uuid4(), uuid4()
    async with session_factory() as session:
        # operational_signals.ai_score_id/decision_id are RESTRICT FKs to ai_score_snapshots/
        # signal_decisions — real parent rows are required for the insert below to succeed at all,
        # independent of what this test is actually verifying.
        session.add(
            AIScoreSnapshotRecord(
                id=ai_score_id, instrument="XAUUSD", timeframe="M15", as_of=old_boundary, calculated_at=old_boundary, mode="live", status="complete",
                policy_name="p", policy_version="1", configuration_version="1", configuration_hash="h", input_fingerprint="fp-1", directional_score=0.0, confidence_score=0.0, market_risk_score=0.0, payload={},
            )
        )
        await session.commit()
        session.add(
            SignalDecisionRecord(
                id=decision_id, decision_key="k", input_fingerprint="fp-2", instrument="XAUUSD", timeframe="M15", direction="long", state="eligible", status="active", mode="live",
                as_of=old_boundary, decided_at=old_boundary, valid_from=old_boundary, valid_until=now + timedelta(days=1), ai_score_snapshot_id=ai_score_id, decision_policy_version="1", eligibility_score=0.0, payload={},
            )
        )
        session.add_all(
            [
                IntegrationSnapshotRecord(snapshot_id=referenced_id, semantic_hash="h-ref", trace_id=uuid4(), mode="live", instrument="XAUUSD", timeframe="M15", analytical_boundary=old_boundary, status="ready", payload={}),
                IntegrationSnapshotRecord(snapshot_id=orphaned_id, semantic_hash="h-orphan", trace_id=uuid4(), mode="live", instrument="XAUUSD", timeframe="M15", analytical_boundary=old_boundary, status="ready", payload={}),
            ]
        )
        await session.commit()
        session.add(
            OperationalSignalRecord(
                operational_signal_id=uuid4(),
                semantic_hash="sig-1",
                decision_id=decision_id,
                ai_score_id=ai_score_id,
                snapshot_id=referenced_id,
                trace_id=uuid4(),
                mode="live",
                instrument="XAUUSD",
                timeframe="M15",
                effective_at=old_boundary,
                expires_at=now + timedelta(days=1),
                payload={},
            )
        )
        await session.commit()

    repository = RetentionRepository(session_factory)
    deleted = await repository.prune(
        now=now,
        analytical_object_retention_days=14,
        analytical_snapshot_retention_days=14,
        integration_audit_retention_days=14,
        operational_signal_retention_days=180,
        market_data_history_retention_days=7,
        batch_size=500,
    )
    assert deleted["integration_snapshots"] == 1

    async with engine.connect() as connection:
        remaining = (await connection.execute(text("SELECT snapshot_id FROM integration_snapshots"))).scalars().all()
    assert remaining == [referenced_id]
