"""Authoritative M15 simulation orchestration."""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import NAMESPACE_URL, uuid5

from backend.app.engines.market_data_engine.models import Candle
from backend.app.market_state import UnifiedMarketState
from backend.app.quant_forecasting.models import QuantForecastResult
from backend.app.signal_synthesis.models import MultiTimeframeSignalSet

from .repository import ScenarioForecastRepository
from .simulation_engine import MarketSimulationEngine
from .simulation_models import (
    CandidateDirection,
    CandidateScenarioOutcome,
    PrimaryScenarioSelection,
)
from .simulation_repository import MarketSimulationRepository

logger = logging.getLogger(__name__)


class MarketSimulationService:
    def __init__(
        self,
        repository: MarketSimulationRepository,
        historical_repository: ScenarioForecastRepository,
        engine: MarketSimulationEngine,
    ) -> None:
        self.repository = repository
        self.historical_repository = historical_repository
        self.engine = engine

    async def process(
        self,
        state: UnifiedMarketState,
        quant: QuantForecastResult,
        synthesis: MultiTimeframeSignalSet,
        *,
        trigger_timeframe: str,
        candles: tuple[Candle, ...] = (),
        evaluated_at: datetime | None = None,
    ) -> PrimaryScenarioSelection | None:
        if trigger_timeframe == "M5":
            logger.info(
                "market_simulation.monitoring_only",
                extra={
                    "cycle_id": str(state.cycle_id),
                    "market_state_id": str(state.state_id),
                    "instrument": state.instrument,
                    "trigger_timeframe": trigger_timeframe,
                },
            )
            return None
        if trigger_timeframe != "M15":
            raise ValueError("market simulations run only on completed M5/M15 candles")
        if evaluated_at is not None:
            await self._evaluate_expired(state.instrument, candles, evaluated_at)
        history = tuple(
            item
            for item in await self.historical_repository.completed_history(
                state.instrument
            )
            if item[1].completed_at <= state.market_data_boundary
        )
        simulation, selection = self.engine.simulate(
            state,
            quant,
            synthesis,
            completed_history=history,
        )
        selection = await self.repository.save(simulation, selection)
        logger.info(
            "primary_scenario.persist.completed",
            extra={
                "selection_id": str(selection.selection_id),
                "simulation_cycle_id": str(selection.simulation_cycle_id),
                "cycle_id": str(selection.cycle_id),
                "market_state_id": str(selection.market_state_id),
                "instrument": selection.instrument,
                "market_cutoff": selection.market_cutoff.isoformat(),
                "candidate_count": simulation.candidate_count,
                "status": selection.status.value,
                "authoritative_action": selection.authoritative_action.value,
                "primary_candidate_id": (
                    str(selection.primary_candidate_id)
                    if selection.primary_candidate_id
                    else None
                ),
                "alternative_candidate_id": (
                    str(selection.alternative_candidate_id)
                    if selection.alternative_candidate_id
                    else None
                ),
                "primary_score": (
                    selection.primary.final_scenario_score
                    if selection.primary is not None
                    else None
                ),
                "signal_eligible": selection.signal_eligible,
                "rejection_reason": selection.rejection_reason,
            },
        )
        return selection

    async def _evaluate_expired(
        self,
        instrument: str,
        candles: tuple[Candle, ...],
        evaluated_at: datetime,
    ) -> None:
        for selection, candidate in await self.repository.pending_primary_before(
            instrument, evaluated_at
        ):
            realized = tuple(
                candle
                for candle in candles
                if candidate.market_cutoff < candle.timestamp <= candidate.expiry
            )
            if not realized:
                continue
            actual_high = max(item.high for item in realized)
            actual_low = min(item.low for item in realized)
            actual_close = realized[-1].close
            target = candidate.primary_target
            invalidation = candidate.invalidation_level
            bullish = candidate.direction == CandidateDirection.BULLISH
            target_reached = bool(
                target is not None
                and (actual_high >= target if bullish else actual_low <= target)
            )
            invalidated = bool(
                invalidation is not None
                and (actual_low <= invalidation if bullish else actual_high >= invalidation)
            )
            directional = (
                actual_close > candidate.reference_price
                if bullish
                else actual_close < candidate.reference_price
            )
            status = (
                "TARGET_REACHED"
                if target_reached and not invalidated
                else "INVALIDATED"
                if invalidated
                else "DIRECTION_CORRECT"
                if directional
                else "EXPIRED"
            )
            outcome = CandidateScenarioOutcome(
                outcome_id=uuid5(
                    NAMESPACE_URL,
                    f"ten:candidate-scenario-outcome:{candidate.candidate_id}",
                ),
                candidate_id=candidate.candidate_id,
                selection_id=selection.selection_id,
                instrument=instrument,
                status=status,
                actual_high=actual_high,
                actual_low=actual_low,
                actual_close=actual_close,
                target_reached=target_reached,
                invalidation_occurred=invalidated,
                directional_accuracy=1.0 if directional else 0.0,
                completed_at=evaluated_at,
            )
            await self.repository.save_outcome(outcome)
            logger.info(
                "primary_scenario.outcome.completed",
                extra=outcome.model_dump(mode="json"),
            )
