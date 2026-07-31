"""Authoritative M15 simulation orchestration."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from backend.app.engines.market_data_engine.models import Candle
from backend.app.market_state import UnifiedMarketState
from backend.app.quant_forecasting.models import QuantForecastResult
from backend.app.signal_synthesis.models import MultiTimeframeSignalSet

from .repository import ScenarioForecastRepository
from .simulation_engine import MarketSimulationEngine
from .simulation_models import (
    AuthoritativeSimulationAttempt,
    CandidateDirection,
    CandidateScenarioOutcome,
    PrimaryScenarioSelection,
    SelectionStatus,
    SimulationAttemptStatus,
)
from .simulation_repository import MarketSimulationRepository

logger = logging.getLogger(__name__)


class MarketSimulationService:
    def __init__(
        self,
        repository: MarketSimulationRepository,
        historical_repository: ScenarioForecastRepository,
        engine: MarketSimulationEngine,
        *,
        market_state_repository: Any | None = None,
        quant_repository: Any | None = None,
        synthesis_repository: Any | None = None,
        recovery_max_age_seconds: int = 7200,
    ) -> None:
        self.repository = repository
        self.historical_repository = historical_repository
        self.engine = engine
        self.market_state_repository = market_state_repository
        self.quant_repository = quant_repository
        self.synthesis_repository = synthesis_repository
        self.recovery_max_age_seconds = recovery_max_age_seconds

    async def recover_latest(
        self, instrument: str, *, now: datetime | None = None
    ) -> PrimaryScenarioSelection | None:
        """Recover the latest missed M15 cutoff using only its persisted point-in-time inputs."""

        evaluated_at = now or datetime.now(UTC)
        if (
            self.market_state_repository is None
            or self.quant_repository is None
            or self.synthesis_repository is None
        ):
            return None
        state = await self.market_state_repository.latest_state(
            instrument,
            trigger_timeframe="M15",
        )
        if state is None or state.trigger_timeframe != "M15":
            return None
        if (
            evaluated_at - state.market_data_boundary
        ).total_seconds() > self.recovery_max_age_seconds:
            logger.info(
                "market_simulation.recovery.skipped",
                extra={
                    "instrument": instrument,
                    "market_cutoff": state.market_data_boundary.isoformat(),
                    "reason": "RECOVERY_LOOKBACK_EXCEEDED",
                },
            )
            return None
        existing = await self.repository.latest_attempt(instrument)
        if (
            existing is not None
            and existing.market_cutoff >= state.market_data_boundary
            and existing.status.terminal
        ):
            return await self.repository.at_cutoff(
                instrument, state.market_data_boundary
            )
        quant = await self.quant_repository.result_for_state(state.state_id)
        synthesis = await self.synthesis_repository.for_state(state.state_id)
        if quant is None or synthesis is None:
            reason = (
                "QUANT_FORECAST_MISSING"
                if quant is None
                else "SYNTHESIS_MISSING"
            )
            await self.record_blocked_cutoff(
                instrument=instrument,
                market_cutoff=state.market_data_boundary,
                server_time=evaluated_at,
                reason=reason,
                failure_stage="startup_recovery",
            )
            return None
        logger.info(
            "market_simulation.recovery.started",
            extra={
                "instrument": instrument,
                "market_cutoff": state.market_data_boundary.isoformat(),
                "market_state_id": str(state.state_id),
            },
        )
        return await self.process(
            state,
            quant,
            synthesis,
            trigger_timeframe="M15",
            evaluated_at=evaluated_at,
        )

    def _attempt(
        self,
        *,
        instrument: str,
        market_cutoff: datetime,
        server_time: datetime,
        status: SimulationAttemptStatus,
        eligibility_result: bool,
        eligibility_reason: str,
        provider_timestamp: datetime | None = None,
        candle_open_time: datetime | None = None,
        candle_close_time: datetime | None = None,
        m5_cutoff: datetime | None = None,
        synchronization_status: str = "UNAVAILABLE",
        **updates: object,
    ) -> AuthoritativeSimulationAttempt:
        version = self.engine.config.configuration_version
        scheduled_at = server_time.astimezone(UTC)
        return AuthoritativeSimulationAttempt(
            attempt_id=uuid5(
                NAMESPACE_URL,
                f"ten:authoritative-simulation-attempt:{instrument}:M15:"
                f"{market_cutoff.astimezone(UTC).isoformat()}:{version}",
            ),
            instrument=instrument,
            market_cutoff=market_cutoff,
            simulation_version=version,
            status=status,
            provider_timestamp=provider_timestamp,
            candle_open_time=candle_open_time,
            candle_close_time=candle_close_time,
            resolved_market_cutoff=market_cutoff,
            server_time=server_time,
            eligibility_result=eligibility_result,
            eligibility_reason=eligibility_reason,
            m5_cutoff=m5_cutoff,
            cutoff_difference_seconds=(
                (market_cutoff - m5_cutoff).total_seconds()
                if m5_cutoff is not None
                else None
            ),
            synchronization_status=synchronization_status,
            scheduled_at=scheduled_at,
            **updates,
        )

    async def record_blocked_cutoff(
        self,
        *,
        instrument: str,
        market_cutoff: datetime,
        server_time: datetime,
        reason: str,
        provider_timestamp: datetime | None = None,
        candle_open_time: datetime | None = None,
        candle_close_time: datetime | None = None,
        failure_stage: str | None = None,
    ) -> AuthoritativeSimulationAttempt:
        attempt = self._attempt(
            instrument=instrument,
            market_cutoff=market_cutoff,
            server_time=server_time,
            status=SimulationAttemptStatus.BLOCKED,
            eligibility_result=True,
            eligibility_reason="completed_m15_candle",
            provider_timestamp=provider_timestamp,
            candle_open_time=candle_open_time,
            candle_close_time=candle_close_time,
            synchronization_status="BLOCKED",
            completed_at=server_time,
            failure_stage=failure_stage,
            failure_type=reason,
            failure_message=reason,
        )
        persisted = await self.repository.save_attempt(attempt)
        logger.warning(
            "market_simulation.cycle.blocked",
            extra=persisted.model_dump(mode="json"),
        )
        return persisted

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
        m15 = next(item for item in state.timeframes if item.timeframe == "M15")
        m5 = next(item for item in state.timeframes if item.timeframe == "M5")
        now = evaluated_at or datetime.now(UTC)
        if now < m15.source_candle_close_at:
            skipped = self._attempt(
                instrument=state.instrument,
                market_cutoff=state.market_data_boundary,
                server_time=now,
                status=SimulationAttemptStatus.SKIPPED,
                eligibility_result=False,
                eligibility_reason="M15_CANDLE_NOT_CLOSED",
                provider_timestamp=m15.source_candle_close_at,
                candle_open_time=m15.source_candle_open_at,
                candle_close_time=m15.source_candle_close_at,
                m5_cutoff=m5.source_candle_close_at,
                synchronization_status="NOT_ELIGIBLE",
                completed_at=now,
                skip_reason="M15_CANDLE_NOT_CLOSED",
            )
            await self.repository.save_attempt(skipped)
            logger.info(
                "market_simulation.cycle.skipped",
                extra=skipped.model_dump(mode="json"),
            )
            return None
        scheduled = self._attempt(
            instrument=state.instrument,
            market_cutoff=state.market_data_boundary,
            server_time=now,
            status=SimulationAttemptStatus.SCHEDULED,
            eligibility_result=True,
            eligibility_reason="completed_m15_candle",
            provider_timestamp=m15.source_candle_close_at,
            candle_open_time=m15.source_candle_open_at,
            candle_close_time=m15.source_candle_close_at,
            m5_cutoff=m5.source_candle_close_at,
            synchronization_status="SYNCHRONIZED",
        )
        existing = await self.repository.save_attempt(scheduled)
        if existing.status.terminal:
            logger.info(
                "market_simulation.cycle.duplicate",
                extra=existing.model_dump(mode="json"),
            )
            return await self.repository.at_cutoff(
                state.instrument, state.market_data_boundary
            )
        running = scheduled.model_copy(
            update={
                "status": SimulationAttemptStatus.RUNNING,
                "started_at": now,
                "retry_count": existing.retry_count,
            }
        )
        await self.repository.save_attempt(running)
        logger.info(
            "market_simulation.cycle.running",
            extra=running.model_dump(mode="json"),
        )
        if evaluated_at is not None:
            await self._evaluate_expired(state.instrument, candles, evaluated_at)
        try:
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
        except Exception as exc:
            failed = running.model_copy(
                update={
                    "status": SimulationAttemptStatus.FAILED,
                    "completed_at": datetime.now(UTC),
                    "failure_stage": "simulation_or_persistence",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc)[:1000] or type(exc).__name__,
                }
            )
            await self.repository.save_attempt(failed)
            logger.exception(
                "market_simulation.cycle.failed",
                extra=failed.model_dump(mode="json"),
            )
            raise
        terminal_status = (
            SimulationAttemptStatus.SUCCESS
            if selection.signal_eligible
            else SimulationAttemptStatus.ANALYTICAL_ONLY
            if selection.status == SelectionStatus.SELECTED
            else SimulationAttemptStatus.NO_SIGNAL
            if selection.status
            in {
                SelectionStatus.INSUFFICIENT_CONFIDENCE,
                SelectionStatus.NO_VALID_CANDIDATE,
            }
            else SimulationAttemptStatus.BLOCKED
        )
        terminal = running.model_copy(
            update={
                "status": terminal_status,
                "completed_at": datetime.now(UTC),
                "candidate_count": simulation.candidate_count,
                "simulation_cycle_id": simulation.simulation_cycle_id,
                "primary_scenario_id": selection.primary_candidate_id,
                "alternative_scenario_id": selection.alternative_candidate_id,
                "failure_message": selection.rejection_reason,
            }
        )
        await self.repository.save_attempt(terminal)
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
                "terminal_status": terminal_status.value,
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
