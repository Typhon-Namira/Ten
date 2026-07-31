"""Application service coordinating creation, history, outcomes and calibration."""

from __future__ import annotations

from datetime import datetime
import logging
from uuid import UUID

from backend.app.engines.market_data_engine.models import Candle
from backend.app.market_state import UnifiedMarketState
from backend.app.quant_forecasting.models import QuantForecastResult
from backend.app.signal_synthesis.models import MultiTimeframeSignalSet

from .calibration import calibration_reliability
from .engine import ScenarioForecastingEngine
from .models import CombinedForwardScenario, ForwardMarketScenario
from .outcome_evaluation import evaluate_expired_scenario
from .repository import ScenarioForecastRepository

logger = logging.getLogger(__name__)


class ScenarioForecastingService:
    def __init__(
        self,
        repository: ScenarioForecastRepository,
        engine: ScenarioForecastingEngine,
    ) -> None:
        self.repository = repository
        self.engine = engine

    async def process(
        self,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        synthesis: MultiTimeframeSignalSet,
        *,
        trigger_timeframe: str,
        candles: tuple[Candle, ...],
        evaluated_at: datetime,
        correlation_id: UUID | None = None,
    ) -> tuple[ForwardMarketScenario, CombinedForwardScenario | None]:
        if trigger_timeframe not in {"M5", "M15"}:
            raise ValueError("scenarios are generated only on M5/M15 candle close")
        await self._evaluate_expired(
            state.instrument,
            trigger_timeframe,
            candles,
            evaluated_at,
        )
        history = await self.repository.completed_history(state.instrument)
        calibration_status, probability = calibration_reliability(
            history,
            timeframe=trigger_timeframe,
        )
        scenario = self.engine.forecast(
            state,
            quant,
            synthesis,
            trigger_timeframe,
            calibrated_probability=probability,
        )
        if calibration_status != scenario.calibration_status:
            scenario = scenario.model_copy(
                update={"calibration_status": calibration_status}
            )
        scenario = await self.repository.save_scenario(scenario)
        logger.info(
            "scenario_forecast.created",
            extra={
                "scenario_id": str(scenario.scenario_id),
                "correlation_id": str(correlation_id) if correlation_id else None,
                "cycle_id": str(scenario.cycle_id),
                "market_state_id": str(scenario.market_state_id),
                "instrument": scenario.instrument,
                "source_timeframe": scenario.timeframe,
                "input_market_cutoff": scenario.market_cutoff_time.isoformat(),
                "evidence_identifiers": scenario.supporting_fact_ids,
                "reference_price": scenario.reference_market_price,
                "expected_range": scenario.expected_range.model_dump(),
                "expected_path": scenario.expected_price_path,
                "geometry_decision": scenario.execution_geometry_validity.value,
                "rejection_reason": scenario.geometry_rejection_reason,
                "scenario_expiry": scenario.expiry.isoformat(),
            },
        )
        combined = None
        if trigger_timeframe == "M15":
            m5 = await self.repository.latest_scenario(
                state.instrument,
                "M5",
                at_or_before=state.market_data_boundary,
            )
            if m5 is None or m5.expiry <= state.market_data_boundary:
                _, m5_probability = calibration_reliability(
                    history,
                    timeframe="M5",
                )
                m5 = await self.repository.save_scenario(
                    self.engine.forecast(
                        state,
                        quant,
                        synthesis,
                        "M5",
                        calibrated_probability=m5_probability,
                    )
                )
            if m5.expiry > state.market_data_boundary:
                combined = await self.repository.save_combined(
                    self.engine.combine(m5, scenario)
                )
                logger.info(
                    "scenario_forecast.combined",
                    extra={
                        "combined_scenario_id": str(combined.combined_scenario_id),
                        "correlation_id": (
                            str(correlation_id) if correlation_id else None
                        ),
                        "m5_scenario_id": str(combined.m5_scenario_id),
                        "m15_scenario_id": str(combined.m15_scenario_id),
                        "agreement": combined.agreement.value,
                        "geometry_decision": combined.execution_geometry_validity.value,
                        "rejection_reason": combined.geometry_rejection_reason,
                    },
                )
        return scenario, combined

    async def _evaluate_expired(
        self,
        instrument: str,
        timeframe: str,
        candles: tuple[Candle, ...],
        evaluated_at: datetime,
    ) -> None:
        pending = await self.repository.pending_before(
            instrument, timeframe, evaluated_at
        )
        for scenario in pending:
            outcome = evaluate_expired_scenario(
                scenario,
                candles,
                evaluated_at=evaluated_at,
            )
            if outcome is None:
                continue
            await self.repository.save_outcome(outcome)
            logger.info(
                "scenario_forecast.outcome",
                extra={
                    "scenario_id": str(scenario.scenario_id),
                    "status": outcome.status.value,
                    "calibration_bucket": outcome.calibration_bucket,
                    "directional_accuracy": outcome.directional_accuracy,
                    "evaluated_at": outcome.evaluated_at.isoformat(),
                },
            )
