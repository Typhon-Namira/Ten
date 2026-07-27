"""PostgreSQL-only regression coverage for AI reasoning persistence.

Item 12 of the required test list: "current-state persistence survives the new database storage
circuit breaker" — this repository's protective layer against a stuck/slow connection is the
configured statement/idle-transaction/pool timeouts (`Settings.db_statement_timeout_ms` /
`db_idle_transaction_timeout_ms` / `db_pool_timeout_seconds`, applied via `connect_args` in
`main.py`'s database engine construction), not a literal circuit-breaker class. This proves a
terminal (failed) AI forecast — the exact row type that used to silently never get persisted
during a provider-backoff window — commits successfully through a real PostgreSQL connection using
those settings, and that `forecast_for_state` reads it back correctly afterward.

Mirrors the schema-per-test pattern used by tests/database/test_retention_postgresql.py and
tests/engines/market_regime_engine/test_market_regime_postgresql.py: skips unless
TEN_TEST_DATABASE_URL is configured.
"""

import asyncio
from datetime import UTC, datetime
import os
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from sqlalchemy import insert, text, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.ai_reasoning.llm_context import build_llm_analysis_context
from backend.app.ai_reasoning.models import (
    AIMarketForecast,
    AIReasoningRequest,
    AIResultStatus,
    MarketMemorySummary,
)
from backend.app.ai_reasoning.repository import SqlAlchemyAIReasoningRepository
from backend.app.ai_reasoning.request_persistence import persisted_request_payload
from backend.app.core.database.base import Base
from backend.app.storage.models import (
    AIReasoningRequestRecord,
    QuantForecastRequestRecord,
    QuantForecastResultRecord,
    UnifiedMarketStateRecord,
)


@pytest.fixture
async def _schema():
    database_url = os.getenv("TEN_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEN_TEST_DATABASE_URL is required for PostgreSQL persistence behavior")
    if not database_url.startswith("postgresql+asyncpg://"):
        pytest.fail("TEN_TEST_DATABASE_URL must use postgresql+asyncpg://")
    if database_url == os.getenv("TEN_DATABASE_URL"):
        pytest.fail("TEN_TEST_DATABASE_URL must not be the configured production database")

    schema = f"ten_ai_reasoning_{uuid4().hex}"
    # The same connect_args main.py applies to the real database engine — statement/idle-txn
    # timeouts are the protective layer this test is proving AI persistence survives.
    engine = create_async_engine(
        database_url,
        connect_args={
            "server_settings": {
                "search_path": schema,
                "statement_timeout": "30000",
                "idle_in_transaction_session_timeout": "30000",
            }
        },
    )
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


def _request(request_id, market_state_id, quant_id, cycle_id, now):
    return AIReasoningRequest(
        request_id=request_id, cycle_id=cycle_id, market_state_id=market_state_id, quantitative_forecast_id=quant_id,
        instrument="XAUUSD", analysis_timestamp=now, knowledge_cutoff=now, trigger_timeframe="M1",
        current_price=2000, supported_timeframe_states=(), data_quality_summary={}, quantitative_probabilities={},
        expected_movement={}, tp_probabilities={}, market_memory=MarketMemorySummary(entry_count=0),
        prompt_version="new_market_analysis_v1", reasoning_policy_version="ai_reasoning_policy_v1", setup_family_registry_version="1.0.0",
        model_identifier="configured-model", quantitative_model_version="1.0.0", feature_schema_version="1.0", market_state_schema_version="1.0",
        created_at=now,
    )


def _failed_forecast(forecast_id, request_id, market_state_id, quant_id, cycle_id, now):
    return AIMarketForecast(
        forecast_id=forecast_id, request_id=request_id, market_state_id=market_state_id, quantitative_forecast_id=quant_id, cycle_id=cycle_id,
        status=AIResultStatus.UNAVAILABLE, model_provider="groq_1", model_identifier="configured-model",
        prompt_version="new_market_analysis_v1", reasoning_policy_version="ai_reasoning_policy_v1", setup_family_registry_version="1.0.0",
        quantitative_model_version="1.0.0", feature_schema_version="1.0", market_state_schema_version="1.0",
        validation_passed=False, retry_count=1, failure_state="authentication_failed", failure_phase="http_request",
        provider_http_status=401, provider_error_code="401", provider_error_message="User not found.",
        fallback_state="no_ai_proposal", reasoning_summary="AI result unavailable; no proposal was created.",
        generated_at=now,
    )


async def _save_parent_rows(
    session_factory,
    *,
    market_state_id,
    quant_id,
    cycle_id,
    now,
) -> None:
    quant_request_id = uuid4()
    async with session_factory() as session:
        await session.execute(
            insert(UnifiedMarketStateRecord).values(
                state_id=market_state_id,
                state_hash=f"state-{market_state_id}",
                instrument="XAUUSD",
                trigger_timeframe="M1",
                market_data_boundary=now,
                knowledge_cutoff=now,
                status="available",
                payload={},
                created_at=now,
            )
        )
        await session.execute(
            insert(QuantForecastRequestRecord).values(
                request_id=quant_request_id,
                market_state_id=market_state_id,
                cycle_id=cycle_id,
                instrument="XAUUSD",
                point_in_time=now,
                model_name="test",
                model_version="1.0",
                mode="shadow",
                payload={},
                created_at=now,
            )
        )
        await session.execute(
            insert(QuantForecastResultRecord).values(
                result_id=quant_id,
                request_id=quant_request_id,
                market_state_id=market_state_id,
                instrument="XAUUSD",
                point_in_time=now,
                status="available",
                model_name="test",
                model_version="1.0",
                payload={},
                generated_at=now,
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_database_reasoning_cycle_claim_is_distributed_and_idempotent(_schema) -> None:
    session_factory = async_sessionmaker(_schema, expire_on_commit=False)
    first_repository = SqlAlchemyAIReasoningRepository(session_factory)
    second_repository = SqlAlchemyAIReasoningRepository(session_factory)
    now = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
    key = "a" * 64

    first, second = await asyncio.gather(
        first_repository.claim_reasoning_cycle(
            key,
            "XAUUSD",
            now,
            "1.0",
            "contract-1",
            now,
        ),
        second_repository.claim_reasoning_cycle(
            key,
            "XAUUSD",
            now,
            "1.0",
            "contract-1",
            now,
        ),
    )

    assert sorted((first, second)) == [False, True]


@pytest.mark.asyncio
async def test_terminal_forecast_persists_and_reads_back_through_real_postgresql(_schema) -> None:
    engine = _schema
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    repository = SqlAlchemyAIReasoningRepository(session_factory)
    now = datetime(2026, 7, 24, 9, 0, tzinfo=UTC)
    market_state_id, quant_id, cycle_id = uuid4(), uuid4(), uuid4()
    request_id = uuid5(NAMESPACE_URL, "ten:test:ai-request")
    forecast_id = uuid5(NAMESPACE_URL, "ten:test:ai-forecast")
    await _save_parent_rows(
        session_factory,
        market_state_id=market_state_id,
        quant_id=quant_id,
        cycle_id=cycle_id,
        now=now,
    )

    request = _request(request_id, market_state_id, quant_id, cycle_id, now)
    await repository.save_request(request)
    persisted = await repository.save_forecast(_failed_forecast(forecast_id, request_id, market_state_id, quant_id, cycle_id, now))

    assert persisted.status == AIResultStatus.UNAVAILABLE
    assert persisted.failure_state == "authentication_failed"

    reread = await repository.forecast_for_state(market_state_id)
    assert reread is not None
    assert reread.forecast_id == forecast_id
    assert reread.status == AIResultStatus.UNAVAILABLE
    assert reread.provider_http_status == 401

    reread_request = await repository.request_for_state(market_state_id)
    assert reread_request is not None
    assert reread_request.request_id == request_id
    assert reread_request.payload_format == "versioned_compact"

    current_payload = persisted_request_payload(
        request,
        build_llm_analysis_context(request),
    )
    legacy_compact_payload = {
        key: value
        for key, value in current_payload.items()
        if key not in {
            "payload_type",
            "payload_schema_version",
            "context_schema_version",
        }
    }
    legacy_compact_payload["schema_version"] = "2.0"
    async with session_factory() as session:
        await session.execute(
            update(AIReasoningRequestRecord)
            .where(AIReasoningRequestRecord.request_id == request_id)
            .values(payload=legacy_compact_payload)
        )
        await session.commit()
    # A fresh repository instance models an application restart against rows written
    # by the already-deployed compact persistence format.
    restarted_repository = SqlAlchemyAIReasoningRepository(session_factory)
    legacy_compact = await restarted_repository.request_for_state(market_state_id)
    assert legacy_compact is not None
    assert legacy_compact.payload_format == "legacy_compact_context"

    async with session_factory() as session:
        await session.execute(
            update(AIReasoningRequestRecord)
            .where(AIReasoningRequestRecord.request_id == request_id)
            .values(payload=request.model_dump(mode="json"))
        )
        await session.commit()
    restarted_repository = SqlAlchemyAIReasoningRepository(session_factory)
    legacy_full = await restarted_repository.request_for_state(market_state_id)
    assert legacy_full is not None
    assert legacy_full.payload_format == "legacy_full_request"
