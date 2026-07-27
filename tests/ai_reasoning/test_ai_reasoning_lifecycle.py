from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import BaseModel

from backend.app.ai_reasoning.config import AIReasoningConfig
from backend.app.ai_reasoning.provider import AIProviderResponse
from backend.app.ai_reasoning.repository import InMemoryAIReasoningRepository
from backend.app.ai_reasoning.request_builder import AIReasoningRequestBuilder
from backend.app.ai_reasoning.service import AIReasoningService
from backend.app.ai_reasoning.validation import StructuredAIOutputValidator
from backend.app.core.config import YamlConfigRepository
from backend.app.core.exceptions import AIProviderFailureDetails, AIProviderRequestError
from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.integration import CanonicalEventEnvelope
from backend.app.market_state import InMemoryUnifiedMarketStateRepository, UnifiedMarketStateService
from backend.app.market_state.service import expected_closed_boundary
from backend.app.quant_forecasting.config import QuantForecastingConfig
from backend.app.quant_forecasting.features import PointInTimeFeatureExtractor
from backend.app.quant_forecasting.provider import DeterministicBaselineProvider
from backend.app.quant_forecasting.repository import InMemoryQuantForecastRepository
from backend.app.quant_forecasting.service import QuantForecastService

BOUNDARY = datetime(2026, 7, 23, 12, 30, tzinfo=UTC)
NOW = BOUNDARY + timedelta(seconds=5)


class EngineOutput(BaseModel):
    snapshot_id: str
    analysis_timestamp: datetime = BOUNDARY
    created_at: datetime = NOW
    engine_version: str = "1.0.0"
    status: str = "ready"
    confidence_score: float = 85
    quality_score: float = 90
    regime: str = "trending"
    structure: dict[str, object] = {"direction": "bullish", "bos": True}


def candle(
    timeframe: Timeframe,
    *,
    boundary: datetime = BOUNDARY,
    now: datetime = NOW,
) -> Candle:
    return Candle(
        timestamp=boundary - timeframe.duration,
        ingestion_timestamp=now,
        symbol="XAUUSD",
        timeframe=timeframe,
        open=3300,
        high=3305,
        low=3298,
        close=3302,
        volume=100,
        spread=0.2,
        provider="existing-provider",
    )


async def state_and_quant(
    boundary: datetime = BOUNDARY,
    *,
    trigger: Timeframe = Timeframe.M5,
):
    now = boundary + timedelta(seconds=5)
    state_service = UnifiedMarketStateService(
        InMemoryUnifiedMarketStateRepository(),
        clock=lambda: now,
    )
    outputs = {
        name: EngineOutput(
            snapshot_id=f"{name}-snapshot-{boundary.isoformat()}",
            analysis_timestamp=boundary,
            created_at=now,
        )
        for name in (
            "smc",
            "liquidity",
            "volume_profile",
            "institutional_flow",
            "market_regime",
            "economic_calendar",
        )
    }
    state = None
    timeframes = [Timeframe.M1, Timeframe.M15]
    timeframes.append(trigger)
    for timeframe in timeframes:
        frame_boundary = expected_closed_boundary(boundary, timeframe.value)
        envelope = CanonicalEventEnvelope.final_candle(
            candle(timeframe, boundary=frame_boundary, now=now),
            uuid4(),
            now,
        )
        state = await state_service.capture_cycle(envelope, outputs)
    assert state is not None
    config = YamlConfigRepository().load_model("quant_forecasting", QuantForecastingConfig)
    quant = await QuantForecastService(
        InMemoryQuantForecastRepository(),
        DeterministicBaselineProvider(config, clock=lambda: now),
        PointInTimeFeatureExtractor(config.feature_schema_version, clock=lambda: now),
        config,
        enabled=True,
        clock=lambda: now,
    ).forecast(state)
    assert quant is not None
    return state, quant


def analysis_payload() -> dict[str, object]:
    evidence = {
        "claim": "Closed M15 structure confirms the interpretation.",
        "kind": "calculated_feature",
        "source_type": "market_structure",
        "source_reference": "feature.market_structure.recent_change",
        "timeframe": "M15",
        "observed_value": "bullish_change_of_character",
    }
    return {
        "market_regime": {
            "classification": "bullish",
            "strength": 75,
            "confidence": 0.8,
            "evidence": [evidence],
        },
        "higher_timeframe_context": {
            "bias": "bullish",
            "description": "Higher-timeframe evidence is aligned.",
            "evidence": [evidence],
        },
        "market_structure": {
            "short_term": "Short-term structure is constructive.",
            "medium_term": "Medium-term structure is constructive.",
            "higher_timeframe": "Higher-timeframe structure remains supported.",
            "recent_change": "A confirmed structural transition occurred.",
            "evidence": [evidence],
        },
        "liquidity_analysis": {
            "summary": "Liquidity remains traceable.",
            "events": ["sell_side_sweep"],
            "unresolved_liquidity": ["buy_side_pool"],
            "evidence": [evidence],
        },
        "supply_demand_analysis": {
            "summary": "Demand remains closer than supply.",
            "nearest_supply": 3350,
            "nearest_demand": 3300,
            "evidence": [evidence],
        },
        "momentum_analysis": {
            "direction": "bullish",
            "strength": 70,
            "trend": "strengthening",
            "evidence": [evidence],
        },
        "volatility_analysis": {
            "state": "normal",
            "trend": "stable",
            "evidence": [evidence],
        },
        "bullish_evidence": [evidence],
        "bearish_evidence": [],
        "contradictions": [],
        "key_risks": [
            {
                **evidence,
                "claim": "A nearby opposing liquidity pool remains unresolved.",
            }
        ],
        "alternative_scenarios": [
            {
                "name": "range_reentry",
                "description": "Price could return to the prior range.",
                "probability": 0.2,
                "confirmation_evidence": ["feature.market_structure.recent_change"],
            }
        ],
        "analysis_confidence": 0.8,
        "executive_summary": "Structure and liquidity support the interpretation.",
    }


class ValidProvider:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload = payload or analysis_payload()
        self.calls = 0

    def metadata(self) -> dict[str, object]:
        return {
            "provider": "cerebras",
            "model_identifier": "configured-model",
            "external_ai_apis": ("cerebras",),
        }

    def metrics(self) -> dict[str, int]:
        return {
            "provider_http_calls": self.calls,
            "cerebras_calls": self.calls,
            "groq_fallback_calls": 0,
            "retry_attempts": 0,
            "schema_corrections": 0,
        }

    async def reason(self, request, *, prompt_version: str) -> AIProviderResponse:
        self.calls += 1
        return AIProviderResponse(
            raw_output=self.payload,
            provider="cerebras",
            model_identifier=request.model_identifier,
            latency_ms=5,
            token_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )


class InvalidProvider(ValidProvider):
    def __init__(self) -> None:
        super().__init__({"proposal": {"direction": "BUY"}})


class TypedFailureProvider(ValidProvider):
    async def reason(self, request, *, prompt_version: str) -> AIProviderResponse:
        self.calls += 1
        raise AIProviderRequestError(
            AIProviderFailureDetails(
                provider="cerebras",
                reason_code="authentication_failed",
                phase="http_request",
                endpoint="https://api.cerebras.ai/v1/chat/completions",
                model=request.model_identifier,
                request_id=str(request.request_id),
                cycle_id=str(request.cycle_id),
                http_status=401,
                exception_class="HTTPStatusError",
            )
        )


class PreflightFailureProvider(ValidProvider):
    async def reason(self, request, *, prompt_version: str) -> AIProviderResponse:
        raise AIProviderRequestError(
            AIProviderFailureDetails(
                provider="cerebras",
                reason_code="request_too_large",
                phase="request_validation",
                endpoint="https://api.cerebras.ai/v1/chat/completions",
                model=request.model_identifier,
                request_id=str(request.request_id),
                cycle_id=str(request.cycle_id),
                exception_class="AIProviderRequestBudgetError",
            )
        )


def build_service(
    repository: InMemoryAIReasoningRepository,
    provider: ValidProvider,
    *,
    shadow: bool = True,
    proposals: bool = False,
    monitoring: bool = False,
    now: datetime = NOW,
) -> AIReasoningService:
    config = YamlConfigRepository().load_model("ai_reasoning", AIReasoningConfig)
    return AIReasoningService(
        repository,
        provider,
        AIReasoningRequestBuilder(
            config,
            model_identifier="configured-model",
            clock=lambda: now,
        ),
        StructuredAIOutputValidator(),
        config,
        shadow_enabled=shadow,
        proposals_enabled=proposals,
        monitoring_enabled=monitoring,
        clock=lambda: now,
    )


@pytest.mark.asyncio
async def test_disabled_analysis_flag_makes_no_provider_call() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    assert await build_service(repository, provider, shadow=False).process(state, quant) is None
    assert provider.calls == 0
    assert repository.analyses == {}


@pytest.mark.asyncio
async def test_valid_analysis_is_persisted_without_forecast_proposal_or_signal() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    result = await build_service(repository, provider).process(state, quant)
    assert result is not None
    assert result.analysis.validation_passed is True
    assert provider.calls == 1
    assert len(repository.analyses) == 1
    assert repository.forecasts == {}
    assert repository.proposals == {}
    assert repository.signals == {}


@pytest.mark.asyncio
async def test_invalid_signal_or_proposal_output_fails_closed() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), InvalidProvider()
    assert await build_service(repository, provider).process(state, quant) is None
    persisted = tuple(repository.analyses.values())
    assert len(persisted) == 1
    assert persisted[0].status.value == "invalid"
    assert repository.forecasts == {}
    assert repository.proposals == {}
    assert repository.signals == {}


@pytest.mark.asyncio
async def test_same_cycle_is_idempotent_and_reuses_one_analysis() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    service = build_service(repository, provider)
    first = await service.process(state, quant)
    second = await service.process(state, quant)
    assert first is not None and second is not None
    assert first.analysis.analysis_id == second.analysis.analysis_id
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_concurrent_workers_share_durable_analysis_claim() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    first, second = await asyncio.gather(
        build_service(repository, provider).process(state, quant),
        build_service(repository, provider).process(state, quant),
    )
    assert provider.calls == 1
    assert sum(item is not None for item in (first, second)) >= 1
    assert len(repository.analyses) == 1


@pytest.mark.asyncio
async def test_provider_failure_persists_analysis_failure_without_trade_artifacts() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), TypedFailureProvider()
    assert await build_service(repository, provider).process(state, quant) is None
    assert provider.calls == 1
    persisted = tuple(repository.analyses.values())
    assert len(persisted) == 1
    assert persisted[0].status.value == "failed"
    assert repository.forecasts == {}
    assert repository.proposals == {}
    assert repository.signals == {}


@pytest.mark.asyncio
async def test_only_m5_trigger_at_utc_five_minute_boundary_is_eligible() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    service = build_service(repository, provider)

    minute_state = state.model_copy(update={"trigger_timeframe": "M1"})
    assert await service.process(minute_state, quant) is None
    assert provider.calls == 0
    assert (
        service.health()["call_control"]["skip_reasons"]["not_five_minute_boundary"]
        == 1
    )

    assert await service.process(state, quant) is not None
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_restart_inside_same_window_reuses_durable_claim() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()

    assert await build_service(repository, provider).process(state, quant) is not None
    restarted = build_service(repository, provider)
    assert await restarted.process(state, quant) is not None

    assert provider.calls == 1
    assert len(repository.analyses) == 1
    assert restarted.health()["call_control"]["provider_http_calls"] == 0


@pytest.mark.asyncio
async def test_controlled_twenty_minute_simulation_has_four_provider_calls() -> None:
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    service = build_service(
        repository,
        provider,
        now=BOUNDARY + timedelta(minutes=20),
    )

    for offset in range(20):
        cycle_boundary = BOUNDARY + timedelta(minutes=offset - offset % 5)
        state, quant = await state_and_quant(cycle_boundary)
        if offset % 5:
            state = state.model_copy(update={"trigger_timeframe": "M1"})
        await service.process(state, quant)

    metrics = service.health()["call_control"]
    assert provider.calls == 4
    assert len(repository.analyses) == 4
    assert metrics["eligible_five_minute_cycles"] == 4
    assert metrics["skipped_before_provider_call"] == 16
    assert metrics["groq_fallback_calls"] == 0


@pytest.mark.asyncio
async def test_stale_market_data_is_rejected_before_provider_call() -> None:
    state, quant = await state_and_quant()
    stale_frames = tuple(
        item.model_copy(update={"stale": True})
        if item.timeframe == "M1"
        else item
        for item in state.timeframes
    )
    state = state.model_copy(update={"timeframes": stale_frames})
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    service = build_service(repository, provider)

    assert await service.process(state, quant) is None
    assert provider.calls == 0
    assert service.health()["call_control"]["skip_reasons"]["market_data_stale"] == 1


@pytest.mark.asyncio
async def test_request_preflight_failure_is_a_skip_not_a_provider_failure() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), PreflightFailureProvider()
    service = build_service(repository, provider)

    assert await service.process(state, quant) is None

    metrics = service.health()["call_control"]
    assert provider.calls == 0
    assert metrics["provider_http_calls"] == 0
    assert metrics["provider_failures"] == 0
    assert metrics["skip_reasons"]["request_preflight_failed"] == 1
    assert repository.analyses == {}
