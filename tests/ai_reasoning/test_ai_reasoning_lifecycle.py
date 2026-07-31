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
    knowledge_delay_seconds: int = 5,
):
    now = boundary + timedelta(seconds=knowledge_delay_seconds)
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
    timeframes = [timeframe for timeframe in (Timeframe.M5, Timeframe.M15) if timeframe != trigger]
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
            "provider": "groq_1",
            "primary_provider": "Groq pool",
            "active_provider": "groq_1",
            "model_identifier": "configured-model",
            "external_ai_apis": ("groq",),
            "configured_account_count": 1,
            "available_account_count": 1,
            "providers": {"groq_1": {"status": "AVAILABLE"}},
        }

    def metrics(self) -> dict[str, int]:
        return {
            "provider_http_calls": self.calls,
            "groq_calls": self.calls,
            "retry_attempts": 0,
            "schema_corrections": 0,
        }

    async def reason(self, request, *, prompt_version: str) -> AIProviderResponse:
        self.calls += 1
        return AIProviderResponse(
            raw_output=self.payload,
            provider="groq_1",
            model_identifier=request.model_identifier,
            latency_ms=5,
            token_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )


class InvalidProvider(ValidProvider):
    def __init__(self) -> None:
        super().__init__({"proposal": {"direction": "BUY"}})


class DegradedValidProvider(ValidProvider):
    def metadata(self) -> dict[str, object]:
        metadata = super().metadata()
        metadata.update(
            {
                "configured_account_count": 4,
                "available_account_count": 3,
                "providers": {
                    "groq_1": {"status": "QUOTA_EXHAUSTED"},
                    "groq_2": {"status": "AVAILABLE"},
                    "groq_3": {"status": "AVAILABLE"},
                    "groq_4": {"status": "AVAILABLE"},
                },
            }
        )
        return metadata


class TypedFailureProvider(ValidProvider):
    async def reason(self, request, *, prompt_version: str) -> AIProviderResponse:
        self.calls += 1
        raise AIProviderRequestError(
            AIProviderFailureDetails(
                provider="groq_1",
                reason_code="authentication_failed",
                phase="http_request",
                endpoint="https://api.groq.test/openai/v1/chat/completions",
                model=request.model_identifier,
                request_id=str(request.request_id),
                cycle_id=str(request.cycle_id),
                http_status=401,
                exception_class="HTTPStatusError",
            )
        )


class ExhaustedPoolProvider(TypedFailureProvider):
    def metadata(self) -> dict[str, object]:
        return {
            "provider": None,
            "primary_provider": "Groq pool",
            "active_provider": None,
            "model_identifier": "gpt-oss-120b",
            "external_ai_apis": ("groq",),
            "configured_account_count": 4,
            "available_account_count": 0,
            "providers": {
                f"groq_{index}": {"status": "QUOTA_EXHAUSTED"}
                for index in range(1, 5)
            },
        }

    def metrics(self) -> dict[str, int]:
        return {
            "provider_http_calls": self.calls * 4,
            "groq_calls": self.calls * 4,
            "retry_attempts": 0,
            "schema_corrections": 0,
        }

    async def reason(self, request, *, prompt_version: str) -> AIProviderResponse:
        self.calls += 1
        raise AIProviderRequestError(
            AIProviderFailureDetails(
                provider="groq_4",
                reason_code="quota_exhausted",
                phase="http_request",
                endpoint="https://api.groq.test/openai/v1/chat/completions",
                model=request.model_identifier,
                request_id=str(request.request_id),
                cycle_id=str(request.cycle_id),
                http_status=429,
                exception_class="HTTPStatusError",
            )
        )


class PreflightFailureProvider(ValidProvider):
    async def reason(self, request, *, prompt_version: str) -> AIProviderResponse:
        raise AIProviderRequestError(
            AIProviderFailureDetails(
                provider="groq_1",
                reason_code="request_too_large",
                phase="request_validation",
                endpoint="https://api.groq.test/openai/v1/chat/completions",
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
async def test_valid_analysis_persists_one_deterministic_analysis_signal() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    result = await build_service(repository, provider).process(state, quant)
    assert result is not None
    assert result.analysis.validation_passed is True
    assert result.signal is not None
    assert result.signal.analysis_id == result.analysis.analysis_id
    assert result.signal.signal.value in {"BUY", "SELL", "HOLD"}
    assert 0 <= result.signal.confidence <= 100
    assert provider.calls == 1
    assert len(repository.analyses) == 1
    assert len(repository.analysis_signals) == 1
    assert repository.forecasts == {}
    assert repository.proposals == {}
    assert repository.signals == {}
    commit = await repository.latest_gate_decision(
        state.instrument, state.market_data_boundary
    )
    assert commit is not None
    assert commit.gate_decision == "COMMITTED"
    assert commit.existing_analysis_id == result.analysis.analysis_id
    assert commit.analysis_market_cutoff == state.market_data_boundary


def test_reasoning_health_is_idle_before_an_eligible_cycle() -> None:
    service = build_service(InMemoryAIReasoningRepository(), ValidProvider())

    health = service.health()

    assert health["operations_status"] == "idle"
    assert health["provider_readiness"] == "idle"


def test_reasoning_health_returns_to_idle_without_a_recent_eligible_cycle() -> None:
    service = build_service(InMemoryAIReasoningRepository(), ValidProvider())
    service.last_eligible_cycle_at = NOW - timedelta(minutes=11)
    service.last_cycle_outcome = "pool_success"

    assert service.health()["operations_status"] == "idle"


@pytest.mark.parametrize(
    ("outcome", "expected"),
    (
        ("pool_success", "healthy"),
        ("failed", "unhealthy"),
    ),
)
def test_reasoning_health_reflects_latest_recent_cycle_outcome(
    outcome: str,
    expected: str,
) -> None:
    service = build_service(InMemoryAIReasoningRepository(), ValidProvider())
    service.last_eligible_cycle_at = NOW
    service.last_cycle_outcome = outcome

    assert service.health()["operations_status"] == expected


@pytest.mark.asyncio
async def test_primary_persisted_analysis_reports_healthy_operations() -> None:
    state, quant = await state_and_quant()
    service = build_service(InMemoryAIReasoningRepository(), ValidProvider())

    assert await service.process(state, quant) is not None

    health = service.health()
    assert health["operations_status"] == "healthy"
    assert health["provider_readiness"] == "healthy"


@pytest.mark.asyncio
async def test_persisted_analysis_with_one_unavailable_account_is_degraded() -> None:
    state, quant = await state_and_quant()
    service = build_service(
        InMemoryAIReasoningRepository(),
        DegradedValidProvider(),
    )

    assert await service.process(state, quant) is not None

    health = service.health()
    assert health["operations_status"] == "degraded"
    assert health["available_account_count"] == 3
    assert health["configured_account_count"] == 4


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
    assert first.signal == second.signal
    assert provider.calls == 1
    assert len(repository.analysis_signals) == 1


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
async def test_provider_failure_persists_failure_without_fabricating_analysis() -> None:
    state, quant = await state_and_quant()
    repository, provider = InMemoryAIReasoningRepository(), TypedFailureProvider()
    assert await build_service(repository, provider).process(state, quant) is None
    assert provider.calls == 1
    assert repository.analyses == {}
    failure = next(iter(repository.failures.values()))
    assert failure.provider_failure is not None
    terminal = failure.provider_failure["terminal"]
    assert terminal["provider"] == "groq_1"
    assert terminal["http_status"] == 401
    assert terminal["reason_code"] == "authentication_failed"
    assert repository.forecasts == {}
    assert repository.proposals == {}
    assert repository.signals == {}


@pytest.mark.asyncio
async def test_all_four_accounts_failing_is_unhealthy_and_persists_no_analysis() -> None:
    state, quant = await state_and_quant()
    repository = InMemoryAIReasoningRepository()
    service = build_service(repository, ExhaustedPoolProvider())

    assert await service.process(state, quant) is None

    health = service.health()
    assert health["operations_status"] == "unhealthy"
    assert health["active_provider"] is None
    assert health["call_control"]["groq_calls"] == 4
    assert repository.analyses == {}
    assert len(repository.failures) == 1


@pytest.mark.asyncio
async def test_only_m5_or_m15_close_trigger_is_eligible() -> None:
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

    m15_state, m15_quant = await state_and_quant(
        BOUNDARY + timedelta(minutes=15),
        trigger=Timeframe.M15,
    )
    service = build_service(
        repository,
        provider,
        now=BOUNDARY + timedelta(minutes=15, seconds=5),
    )
    assert await service.process(m15_state, m15_quant) is not None
    assert provider.calls == 2


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
    assert metrics["groq_calls"] == 4


@pytest.mark.asyncio
async def test_m5_and_m15_at_same_cutoff_have_independent_durable_claims() -> None:
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    service = build_service(repository, provider, now=BOUNDARY + timedelta(seconds=10))
    m5_state, m5_quant = await state_and_quant(
        BOUNDARY,
        trigger=Timeframe.M5,
        knowledge_delay_seconds=5,
    )
    m15_state, m15_quant = await state_and_quant(
        BOUNDARY,
        trigger=Timeframe.M15,
        knowledge_delay_seconds=6,
    )

    m5_result = await service.process(m5_state, m5_quant)
    m15_result = await service.process(m15_state, m15_quant)

    assert m5_result is not None
    assert m15_result is not None
    assert provider.calls == 2
    assert await repository.analysis_for_state(m5_state.state_id) is not None
    assert await repository.analysis_for_state(m15_state.state_id) is not None
    claim_scopes = {
        item["analysis_timeframe"] for item in repository.reasoning_cycles.values()
    }
    assert claim_scopes == {"M5_M15:M5", "M5_M15:M15"}


@pytest.mark.asyncio
async def test_four_consecutive_m15_cutoffs_each_persist_authoritative_analysis() -> None:
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    state_ids = []

    for offset in range(0, 60, 15):
        boundary = BOUNDARY + timedelta(minutes=offset)
        state, quant = await state_and_quant(boundary, trigger=Timeframe.M15)
        state_ids.append(state.state_id)
        service = build_service(
            repository,
            provider,
            now=boundary + timedelta(seconds=5),
        )
        assert await service.process(state, quant) is not None

    assert provider.calls == 4
    assert all(
        [await repository.analysis_for_state(state_id) for state_id in state_ids]
    )


@pytest.mark.asyncio
async def test_stale_skip_reason_is_persisted_for_exact_market_cutoff() -> None:
    state, quant = await state_and_quant(BOUNDARY, trigger=Timeframe.M15)
    state = state.model_copy(
        update={
            "timeframes": tuple(
                frame.model_copy(update={"stale": True})
                if frame.timeframe == "M15"
                else frame
                for frame in state.timeframes
            )
        }
    )
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()

    assert await build_service(repository, provider).process(state, quant) is None

    decision = await repository.latest_gate_decision("XAUUSD", BOUNDARY)
    assert decision is not None
    assert decision.gate_decision == "SKIPPED"
    assert decision.gate_skip_reason == "market_data_stale"
    assert decision.market_state_id == state.state_id
    assert decision.analysis_market_cutoff is None
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_stale_market_data_is_rejected_before_provider_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="backend.app.ai_reasoning.service")
    state, quant = await state_and_quant()
    stale_frames = tuple(
        item.model_copy(update={"stale": True})
        if item.timeframe == "M5"
        else item
        for item in state.timeframes
    )
    state = state.model_copy(update={"timeframes": stale_frames})
    repository, provider = InMemoryAIReasoningRepository(), ValidProvider()
    service = build_service(repository, provider)

    assert await service.process(state, quant) is None
    assert provider.calls == 0
    assert service.health()["call_control"]["skip_reasons"]["market_data_stale"] == 1
    record = next(
        item
        for item in caplog.records
        if item.getMessage() == "ai_reasoning.gate.skipped"
    )
    assert record.reason_code == "stale_data"
    assert record.snapshot_id == str(state.state_id)
    assert record.cycle_id == str(state.cycle_id)
    assert record.details == {"skip_reason": "market_data_stale"}


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
