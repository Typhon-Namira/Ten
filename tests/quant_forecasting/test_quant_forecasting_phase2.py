from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from backend.app.core.config import YamlConfigRepository
from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.integration import CanonicalEventEnvelope
from backend.app.market_state import InMemoryUnifiedMarketStateRepository, UnifiedMarketState, UnifiedMarketStateService
from backend.app.quant_forecasting.calibration import CalibrationReporter
from backend.app.quant_forecasting.config import QuantForecastingConfig
from backend.app.quant_forecasting.features import PointInTimeFeatureExtractor
from backend.app.quant_forecasting.models import CalibrationObservation, CalibrationStatus, FeatureAvailability, ForecastStatus, OutcomeStatus
from backend.app.quant_forecasting.outcomes import ForecastOutcomeEvaluator, ForecastOutcomeWorker
from backend.app.quant_forecasting.provider import DeterministicBaselineProvider
from backend.app.quant_forecasting.repository import InMemoryQuantForecastRepository
from backend.app.quant_forecasting.service import QuantForecastService


BOUNDARY = datetime(2026, 7, 23, 12, 30, tzinfo=UTC)
NOW = BOUNDARY + timedelta(seconds=5)


def config() -> QuantForecastingConfig:
    return YamlConfigRepository().load_model("quant_forecasting", QuantForecastingConfig)


def candle(timeframe: Timeframe, *, timestamp: datetime | None = None, close: float = 3302) -> Candle:
    return Candle(
        timestamp=timestamp or BOUNDARY - timeframe.duration,
        ingestion_timestamp=NOW,
        symbol="XAUUSD",
        timeframe=timeframe,
        open=3300,
        high=max(3305, close),
        low=min(3298, close),
        close=close,
        volume=100,
        spread=0.2,
        provider="test-provider",
    )


async def market_state(*, outputs: dict[str, object] | None = None) -> UnifiedMarketState:
    service = UnifiedMarketStateService(InMemoryUnifiedMarketStateRepository(), clock=lambda: NOW)
    values = outputs or {}
    state = None
    for timeframe in (Timeframe.M5, Timeframe.M15):
        envelope = CanonicalEventEnvelope.final_candle(candle(timeframe), uuid4(), NOW)
        state = await service.capture_cycle(envelope, values)
    assert state is not None
    return state


def build_service(repository: InMemoryQuantForecastRepository, *, enabled: bool = True) -> QuantForecastService:
    settings = config()
    return QuantForecastService(
        repository,
        DeterministicBaselineProvider(settings, clock=lambda: NOW),
        PointInTimeFeatureExtractor(settings.feature_schema_version, clock=lambda: NOW),
        settings,
        enabled=enabled,
        clock=lambda: NOW,
    )


@pytest.mark.asyncio
async def test_exact_configured_horizons_and_probabilities() -> None:
    state = await market_state()
    result = await build_service(InMemoryQuantForecastRepository()).forecast(state)

    assert result is not None
    assert result.status == ForecastStatus.AVAILABLE
    assert [(item.horizon.candle_count, item.horizon.timeframe) for item in result.predictions] == [
        (1, "M5"),
        (3, "M5"),
        (1, "M15"),
        (3, "M15"),
    ]
    assert all(abs(item.buy_probability + item.sell_probability + item.neutral_probability - 1) < 1e-10 for item in result.predictions)
    assert result.shadow_only is True
    assert result.approved_for_publication is False
    assert result.calibration_status == CalibrationStatus.UNCALIBRATED
    assert result.training_dataset_version == "none_infrastructure_baseline"
    assert result.feature_schema_version == config().feature_schema_version
    assert result.calibration_version == "unavailable"


@pytest.mark.asyncio
async def test_feature_lineage_is_point_in_time_and_missing_evidence_is_not_zero_filled() -> None:
    state = await market_state()
    vector = PointInTimeFeatureExtractor(config().feature_schema_version, clock=lambda: NOW).extract(state)

    assert vector.market_state_id == state.state_id
    assert vector.point_in_time == state.market_data_boundary
    assert all(item.source_evidence_ids for item in vector.features if item.availability == FeatureAvailability.AVAILABLE)
    unavailable = [item for item in vector.features if item.availability == FeatureAvailability.UNAVAILABLE]
    assert unavailable
    assert all(item.value is None for item in unavailable)


@pytest.mark.asyncio
async def test_future_evidence_is_rejected_before_feature_extraction() -> None:
    state = await market_state()
    payload = state.model_dump(mode="python")
    payload["evidence"][0]["available_at"] = state.knowledge_cutoff + timedelta(seconds=1)

    with pytest.raises(ValueError, match="unavailable at the knowledge cutoff"):
        UnifiedMarketState.model_validate(payload)


@pytest.mark.asyncio
async def test_shadow_flag_disabled_produces_no_requests_or_forecasts() -> None:
    state = await market_state()
    repository = InMemoryQuantForecastRepository()

    assert await build_service(repository, enabled=False).forecast(state) is None
    assert not repository.requests
    assert not repository.features
    assert not repository.results


@pytest.mark.asyncio
async def test_replay_is_deterministic_and_persists_full_lineage() -> None:
    state = await market_state()
    first_repository = InMemoryQuantForecastRepository()
    second_repository = InMemoryQuantForecastRepository()

    first = await build_service(first_repository).forecast(state)
    second = await build_service(second_repository).forecast(state)

    assert first == second
    assert first is not None
    request = next(iter(first_repository.requests.values()))
    vector = next(iter(first_repository.features.values()))
    assert request.market_state_id == state.state_id == vector.market_state_id == first.market_state_id
    assert request.point_in_time == vector.point_in_time == first.point_in_time


@pytest.mark.asyncio
async def test_provider_failure_is_persisted_without_raising_or_numeric_output() -> None:
    class FailedProvider(DeterministicBaselineProvider):
        async def predict(self, request, features):
            raise RuntimeError("synthetic provider outage")

    state = await market_state()
    settings = config()
    repository = InMemoryQuantForecastRepository()
    service = QuantForecastService(
        repository,
        FailedProvider(settings),
        PointInTimeFeatureExtractor(settings.feature_schema_version, clock=lambda: NOW),
        settings,
        enabled=True,
        clock=lambda: NOW,
    )

    result = await service.forecast(state)
    assert result is not None
    assert result.status == ForecastStatus.FAILED
    assert result.predictions == ()
    assert await repository.latest_result("XAUUSD") == result


@pytest.mark.asyncio
async def test_outcome_waits_for_closed_horizon_and_is_spread_adjusted() -> None:
    state = await market_state()
    result = await build_service(InMemoryQuantForecastRepository()).forecast(state)
    assert result is not None
    prediction = result.predictions[0]
    evaluator = ForecastOutcomeEvaluator()
    future = [
        candle(Timeframe.M5, timestamp=BOUNDARY + timedelta(minutes=5 * index), close=3302 + index)
        for index in range(1)
    ]

    pending = evaluator.evaluate(result, prediction, future, evaluated_at=BOUNDARY + timedelta(minutes=2))
    completed = evaluator.evaluate(result, prediction, future, evaluated_at=BOUNDARY + timedelta(minutes=5))

    assert pending.status == OutcomeStatus.PENDING
    assert completed.status == OutcomeStatus.VALID
    assert completed.candle_count == 1
    assert completed.spread_adjusted_return is not None
    assert completed.realized_return is not None
    assert completed.spread_adjusted_return < completed.realized_return


@pytest.mark.asyncio
async def test_mfe_mae_and_conservative_tp_sl_ordering_are_deterministic() -> None:
    state = await market_state()
    result = await build_service(InMemoryQuantForecastRepository()).forecast(state)
    assert result is not None
    prediction = result.predictions[0].model_copy(
        update={
            "buy_probability": 0.8,
            "sell_probability": 0.1,
            "neutral_probability": 0.1,
            "expected_base_movement": 0.001,
            "expected_minimum_movement": 0.0005,
            "expected_maximum_movement": 0.0015,
            "expected_mae": 0.001,
        }
    )
    entry = prediction.reference_price
    bars = [
        candle(Timeframe.M5, timestamp=BOUNDARY, close=entry),
    ]
    bars[0] = bars[0].model_copy(update={"high": entry * 1.002, "low": entry * 0.998})

    outcome = ForecastOutcomeEvaluator().evaluate(
        result,
        prediction,
        bars,
        evaluated_at=BOUNDARY + timedelta(minutes=5),
    )

    assert outcome.status == OutcomeStatus.VALID
    assert outcome.maximum_favorable_excursion == pytest.approx(0.002)
    assert outcome.maximum_adverse_excursion == pytest.approx(0.002)
    assert outcome.tp1_hit is True
    assert outcome.tp2_hit is True
    assert outcome.stop_loss_hit is True
    assert outcome.stop_before_tp is True


@pytest.mark.asyncio
async def test_calibration_reports_brier_log_loss_ece_and_buckets() -> None:
    state = await market_state()
    result = await build_service(InMemoryQuantForecastRepository()).forecast(state)
    assert result is not None
    prediction = result.predictions[0]
    candles = [candle(Timeframe.M5, timestamp=BOUNDARY, close=3304)]
    outcome = ForecastOutcomeEvaluator().evaluate(
        result,
        prediction,
        candles,
        evaluated_at=BOUNDARY + timedelta(minutes=5),
    )

    report = CalibrationReporter().build(result.model_name, result.model_version, [(prediction, outcome)], generated_at=NOW)

    assert report.sample_count == 1
    assert report.brier_score is not None
    assert report.log_loss is not None
    assert report.expected_calibration_error is not None
    assert report.buckets
    assert report.status == CalibrationStatus.UNCALIBRATED
    segmented = CalibrationReporter().build_segmented(
        result.model_name,
        result.model_version,
        [
            CalibrationObservation(
                prediction=prediction,
                outcome=outcome,
                session="london",
                regime="trending",
                confidence_band="high",
                data_quality_status="available",
            )
        ],
        generated_at=NOW,
    )
    assert {next(iter(item.filters)) for item in segmented} == {
        "all",
        "horizon",
        "session",
        "regime",
        "confidence_band",
        "data_quality_status",
    }


@pytest.mark.asyncio
async def test_outcome_worker_persists_only_after_each_horizon_is_complete() -> None:
    class MarketData:
        async def history(self, symbol, timeframe, *, start, end, limit, refresh):
            return [
                candle(timeframe, timestamp=start + timeframe.duration * index, close=3303 + index)
                for index in range(limit)
            ]

    state = await market_state()
    repository = InMemoryQuantForecastRepository()
    result = await build_service(repository).forecast(state)
    assert result is not None

    outcomes = await ForecastOutcomeWorker(
        repository,
        MarketData(),
        clock=lambda: BOUNDARY + timedelta(minutes=45),
    ).evaluate(result)

    assert len(outcomes) == 4
    assert all(item.status == OutcomeStatus.VALID for item in outcomes)
    assert len(repository.outcomes) == 4


def test_phase_two_configuration_remains_shadow_only_by_default() -> None:
    flags = YamlConfigRepository().load("feature_flags")["flags"]
    assert flags["ai_centric_shadow_mode"] is False
    assert flags["ai_signal_proposals"] is False
    assert flags["ai_signal_monitoring"] is False
    assert flags["ai_signal_publication"] is False
    assert flags["ai_signal_adjustments"] is False
