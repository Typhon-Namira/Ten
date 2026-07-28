"""Point-in-time lifecycle and outcome evaluation for analysis signals."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from .analysis import (
    AIAnalysisSignal,
    AIAnalysisSignalOutcome,
    AnalysisSignalAction,
    AnalysisSignalLifecycle,
)


class AnalysisSignalOutcomeEvaluator:
    def initial(
        self,
        signal: AIAnalysisSignal,
    ) -> AIAnalysisSignalOutcome:
        hold = signal.signal == AnalysisSignalAction.HOLD
        return AIAnalysisSignalOutcome(
            outcome_id=uuid5(
                NAMESPACE_URL,
                f"ten:analysis-signal-outcome:{signal.signal_id}",
            ),
            signal_id=signal.signal_id,
            status=(
                AnalysisSignalLifecycle.COMPLETED
                if hold
                else AnalysisSignalLifecycle.ACTIVE
            ),
            entry_reached=False,
            stop_hit=False,
            target_hit=False,
            expired=False,
            evaluated_at=signal.generated_at,
            completed_at=signal.generated_at if hold else None,
            reason_codes=("hold_cycle_completed",) if hold else (),
        )

    def evaluate(
        self,
        signal: AIAnalysisSignal,
        previous: AIAnalysisSignalOutcome,
        *,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        evaluated_at: datetime,
        superseded: bool,
    ) -> AIAnalysisSignalOutcome:
        at = evaluated_at.astimezone(UTC)
        if previous.status not in {
            AnalysisSignalLifecycle.ACTIVE,
            AnalysisSignalLifecycle.STALE,
        }:
            return previous
        assert signal.entry is not None
        assert signal.stop_loss is not None
        assert signal.take_profit is not None
        newly_entered = (
            not previous.entry_reached
            and candle_low <= signal.entry <= candle_high
        )
        entered = previous.entry_reached or newly_entered
        entered_at = previous.entry_reached_at or (at if entered else None)
        buy = signal.signal == AnalysisSignalAction.BUY
        favorable = (
            max(0.0, candle_high - signal.entry)
            if buy
            else max(0.0, signal.entry - candle_low)
        )
        adverse = (
            max(0.0, signal.entry - candle_low)
            if buy
            else max(0.0, candle_high - signal.entry)
        )
        mfe = max(previous.maximum_favorable_excursion, favorable if entered else 0)
        mae = max(previous.maximum_adverse_excursion, adverse if entered else 0)
        target_hit = entered and (
            candle_high >= signal.take_profit
            if buy
            else candle_low <= signal.take_profit
        )
        stop_hit = entered and (
            candle_low <= signal.stop_loss
            if buy
            else candle_high >= signal.stop_loss
        )
        deferred_target = newly_entered and target_hit and not stop_hit
        if deferred_target:
            # OHLC cannot prove that target occurred after the first entry touch.
            target_hit = False
        status = AnalysisSignalLifecycle.ACTIVE
        reason_codes: tuple[str, ...] = ()
        pnl: float | None = None
        completed_at: datetime | None = None
        if target_hit and stop_hit:
            # Intrabar ordering is unknowable from OHLC; resolve conservatively.
            target_hit = False
            status = AnalysisSignalLifecycle.STOP_HIT
            reason_codes = ("same_candle_target_and_stop_conservative_stop",)
            pnl = -abs(signal.entry - signal.stop_loss)
            completed_at = at
        elif target_hit:
            status = AnalysisSignalLifecycle.TARGET_HIT
            pnl = abs(signal.take_profit - signal.entry)
            completed_at = at
        elif stop_hit:
            status = AnalysisSignalLifecycle.STOP_HIT
            pnl = -abs(signal.entry - signal.stop_loss)
            completed_at = at
        elif signal.valid_until is not None and at >= signal.valid_until:
            status = AnalysisSignalLifecycle.EXPIRED
            pnl = (
                candle_close - signal.entry
                if buy and entered
                else signal.entry - candle_close
                if entered
                else 0.0
            )
            completed_at = at
            reason_codes = ("validity_horizon_elapsed",)
        elif superseded:
            status = AnalysisSignalLifecycle.SUPERSEDED
            pnl = (
                candle_close - signal.entry
                if buy and entered
                else signal.entry - candle_close
                if entered
                else 0.0
            )
            completed_at = at
            reason_codes = ("newer_analytical_cycle_completed",)
        elif deferred_target:
            reason_codes = ("entry_target_intrabar_order_unresolved",)
        risk = abs(signal.entry - signal.stop_loss)
        actual_rr = pnl / risk if pnl is not None and risk > 0 else None
        holding = (
            (completed_at - entered_at).total_seconds()
            if completed_at is not None and entered_at is not None
            else None
        )
        return previous.model_copy(
            update={
                "status": status,
                "entry_reached": entered,
                "entry_reached_at": entered_at,
                "stop_hit": stop_hit,
                "target_hit": target_hit,
                "expired": status == AnalysisSignalLifecycle.EXPIRED,
                "maximum_favorable_excursion": mfe,
                "maximum_adverse_excursion": mae,
                "holding_time_seconds": holding,
                "actual_risk_reward": actual_rr,
                "profit_loss": pnl,
                "evaluated_at": at,
                "completed_at": completed_at,
                "reason_codes": reason_codes,
            }
        )
