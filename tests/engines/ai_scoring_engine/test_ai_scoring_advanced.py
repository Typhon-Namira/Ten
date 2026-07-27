from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.ai.provider_client import AIProviderClient, AIProviderCompletion
from backend.app.ai.prompts import PromptLoader
from backend.app.engines.ai_scoring_engine import (
    AIScoringConfig,
    AIScoringService,
    DeterministicAIScoringEngine,
    FixedClock,
    InMemoryAIScoringRepository,
    ProviderScoringEngine,
    ScoreRequest,
    ScoringContext,
    SourceHealth,
    SourceState,
    SystemClock,
)
from backend.app.engines.ai_scoring_engine.engine import AIScoringEngine
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore
from backend.app.features.models import FeatureSnapshot
from tests.engines.ai_scoring_engine.test_ai_scoring import NOW, aligned_input, evidence, scoring_input


class FakeClient(AIProviderClient):
    provider = "groq_1"
    base_url = "https://api.groq.test/openai/v1"
    configured = True

    async def available_models(self) -> tuple[str, ...]:
        return ("gpt-oss-120b",)

    async def complete_json(self, **_: Any) -> AIProviderCompletion:
        return AIProviderCompletion(
            content={"direction": "neutral", "quality_score": 50, "reasoning": ["legacy compatibility"]},
            provider="groq_1",
            model="gpt-oss-120b",
            status_code=200,
            latency_ms=1,
            provider_request_id=None,
            token_usage=None,
            rate_limit_limit=None,
            rate_limit_remaining=None,
            rate_limit_reset=None,
            retry_after=None,
        )


def test_legacy_adapter_and_abstract_contract_are_covered() -> None:
    assert ScoringContext.from_features({}, {}).features == {}
    with pytest.raises(TypeError):
        AIScoringEngine()


@pytest.mark.asyncio
async def test_legacy_provider_adapter_defaults_provider_fields() -> None:
    result = await ProviderScoringEngine(FakeClient(), PromptLoader()).score(ScoringContext())
    assert result.model == "gpt-oss-120b"
    assert result.prompt_version == "signal_analysis_v1"


def test_clock_model_and_numerical_guard_branches() -> None:
    assert SystemClock().now().tzinfo is not None
    with pytest.raises(ValidationError, match="checked_at"):
        SourceHealth(source="smc", state=SourceState.AVAILABLE, checked_at=datetime(2026, 1, 1))
    with pytest.raises(ValidationError, match="timestamps"):
        aligned_input().model_copy(update={"as_of": datetime(2026, 1, 1)}).__class__.model_validate(aligned_input().model_dump() | {"as_of": datetime(2026, 1, 1)})
    engine = DeterministicAIScoringEngine(clock=FixedClock(NOW))
    assert engine._average(()) == 0
    with pytest.raises(ValueError, match="non-finite"):
        engine._bounded(float("nan"), 0, 1)
    future = evidence("smc", "structure", 1).model_copy(update={"publication_timestamp": NOW + timedelta(seconds=10)})
    with pytest.raises(ValueError, match="future source timestamp"):
        engine._freshness(future, NOW, 30, 60)
    expired = engine.score(scoring_input(evidence("smc", "structure", 1, age=timedelta(hours=2))))
    assert expired.components[0].freshness_factor == 0


class MarketDataService:
    async def state(self, *_: Any, **__: Any) -> Any:
        return SimpleNamespace(provider_health="healthy", as_of=NOW)


class EconomicService:
    async def context(self, *_: Any, **__: Any) -> Any:
        return SimpleNamespace(
            context_id="economic-context-1",
            analysis_timestamp=NOW,
            model_dump=lambda **_: {"risk_score": 0.8, "relevance_score": 0.9, "quality_score": 0.8, "risk_window_phase": "active"},
        )


class AnalyticsService:
    def __init__(self, name: str, *, missing: bool = False) -> None:
        self.name = name
        self.missing = missing

    async def state(self, *_: Any, **__: Any) -> Any:
        if self.missing:
            return None
        return SimpleNamespace(id=f"{self.name}-id", snapshot_id=f"{self.name}-snapshot", analysis_timestamp=NOW, engine_version="1.0.0")

    def features(self, _: Any) -> dict[str, object]:
        values: dict[str, dict[str, object]] = {
            "smc": {"current_structure_direction": "bullish", "smc_confidence": 0.8, "smc_input_quality": 0.9},
            "liquidity": {"liquidity_density_above": 2, "liquidity_density_below": 1, "confidence": 0.8, "data_quality": 0.9},
            "volume_profile": {"poc_migration": {"direction": "up"}, "confidence": 0.8, "data_quality": {"overall": 0.8}},
            "institutional_flow": {"directional_pressure": {"net_pressure": 0.6, "confidence": 0.8}, "quality": {"overall": 0.8}},
            "market_regime": {"net_directional_score": 0.7, "confidence": 0.8, "quality": 0.8},
        }
        return values[self.name]


class BrokenService:
    async def state(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("provider secret must not escape")


@pytest.mark.asyncio
async def test_full_point_in_time_collection_duplicate_and_dependency_failure() -> None:
    repository = InMemoryAIScoringRepository()
    service = AIScoringService(
        repository,
        InMemoryEventBus(),
        InMemoryFeatureStore(),
        AIScoringConfig(),
        market_data=MarketDataService(),
        smc=AnalyticsService("smc"),
        liquidity=AnalyticsService("liquidity", missing=True),
        volume_profile=AnalyticsService("volume_profile"),
        institutional_flow=AnalyticsService("institutional_flow"),
        market_regime=AnalyticsService("market_regime"),
        economic_calendar=EconomicService(),
        repository_mode="postgresql",
        clock=FixedClock(NOW),
    )
    await service.start()
    request = ScoreRequest(as_of=NOW)
    first = await service.calculate(request)
    second = await service.calculate(request)
    assert first == second
    assert "liquidity" in first.missing_sources
    assert service.metrics.requests_total == 2
    assert service.health()["persistence"]["status"] == "healthy"
    service.dependencies["smc"] = BrokenService()
    collected = await service.collect_input(request)
    assert next(item for item in collected.source_health if item.source == "smc").state == SourceState.UNAVAILABLE


class FailingFeatureStore(InMemoryFeatureStore):
    async def write(self, feature: Any) -> None:
        raise RuntimeError("write failed")


class FailingEventBus(InMemoryEventBus):
    async def publish(self, event: Any) -> None:
        raise RuntimeError("publish failed")


@pytest.mark.asyncio
async def test_publication_failures_are_observable_and_do_not_corrupt_score() -> None:
    service = AIScoringService(InMemoryAIScoringRepository(), FailingEventBus(), FailingFeatureStore(), AIScoringConfig(), clock=FixedClock(NOW))
    await service.start()
    snapshot = await service.calculate_input(scoring_input(evidence("smc", "structure", 0.9), evidence("institutional_flow", "participation", -0.9)))
    assert snapshot.conflicts
    assert service.metrics.feature_publish_failures_total == 1
    assert service.metrics.event_publish_failures_total == 1
    assert service.health()["feature_store"]["status"] == "degraded"


@pytest.mark.asyncio
async def test_successful_conflict_event_and_naive_request_boundary() -> None:
    events = InMemoryEventBus()
    service = AIScoringService(InMemoryAIScoringRepository(), events, InMemoryFeatureStore(), AIScoringConfig(), clock=FixedClock(NOW))
    await service.start()
    await service.calculate_input(scoring_input(evidence("smc", "structure", 0.9), evidence("institutional_flow", "participation", -0.9)))
    assert len(events.history()) == 2
    with pytest.raises(ValueError, match="timezone-aware"):
        await service.collect_input(ScoreRequest(as_of=datetime(2026, 7, 19, 12)))


@pytest.mark.asyncio
async def test_calculate_failure_metric_and_naive_boundary_guard() -> None:
    service = AIScoringService(InMemoryAIScoringRepository(), InMemoryEventBus(), InMemoryFeatureStore(), AIScoringConfig(), clock=FixedClock(NOW))
    await service.start()
    class BrokenEngine:
        def score(self, _: Any) -> Any:
            raise ValueError("invalid")

    service.engine = BrokenEngine()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="invalid"):
        await service.calculate(ScoreRequest())
    assert service.metrics.failed_total == 1


@pytest.mark.asyncio
async def test_registration_build_execute_and_register() -> None:
    from backend.app.engines.ai_scoring_engine import registration

    engine = registration._build(SimpleNamespace(), {})
    snapshot = FeatureSnapshot(correlation_id=uuid4(), features={"smc": {"current_structure_direction": "bullish"}, "ignored": {}}, engine_versions={"smc": "1.0.0"}, created_at=NOW)
    store = SimpleNamespace(snapshot=AsyncMock(return_value=snapshot))
    context = SimpleNamespace(feature_store=store, correlation_id=snapshot.correlation_id, feature_snapshot=None)
    result = await registration._execute(engine, context)
    assert result.namespace == "ai_score"
    legacy_result = await registration._execute(ProviderScoringEngine(FakeClient(), PromptLoader()), context)
    assert legacy_result.namespace == "ai"
    factory = SimpleNamespace(register=Mock())
    registration.register(factory)
    assert factory.register.call_count == 1
