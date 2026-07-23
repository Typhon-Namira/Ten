"""Point-in-time signal outcome evaluation with unresolved-horizon protection."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.ai_reasoning.models import Direction

from .models import DetailedSignalOutcome, PublishedAnalyticalSignal


class EvaluationCandle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)

    @field_validator("timestamp")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("candle timestamp must be timezone-aware")
        return value.astimezone(UTC)


class SignalOutcomeEvaluator:
    def __init__(self, *, configured_slippage: float = 0.0) -> None:
        self.configured_slippage = configured_slippage

    def evaluate(
        self,
        signal: PublishedAnalyticalSignal,
        candles: tuple[EvaluationCandle, ...],
        *,
        required_horizon_end: datetime,
        spread: float,
        evaluated_at: datetime,
    ) -> DetailedSignalOutcome:
        ordered = tuple(sorted(candles, key=lambda item: item.timestamp))
        if evaluated_at.tzinfo is None or required_horizon_end.tzinfo is None:
            raise ValueError("outcome boundaries must be timezone-aware")
        if not ordered or ordered[-1].timestamp < required_horizon_end:
            return DetailedSignalOutcome(
                outcome_id=uuid5(NAMESPACE_URL, f"ten:detailed-outcome:{signal.signal_id}"),
                signal_id=signal.signal_id,
                status="pending",
                evaluation_horizon_complete=False,
                evaluated_at=evaluated_at,
                reason_codes=("insufficient_future_candles",),
            )
        future = tuple(item for item in ordered if signal.published_at <= item.timestamp <= required_horizon_end)
        entry = (signal.entry_zone.low + signal.entry_zone.high) / 2
        entered = next((item for item in future if item.low <= signal.entry_zone.high and item.high >= signal.entry_zone.low), None)
        if entered is None:
            return DetailedSignalOutcome(
                outcome_id=uuid5(NAMESPACE_URL, f"ten:detailed-outcome:{signal.signal_id}"),
                signal_id=signal.signal_id,
                status="resolved",
                evaluation_horizon_complete=True,
                realized_direction=Direction.NEUTRAL,
                realized_return=0.0,
                maximum_favorable_excursion=0.0,
                maximum_adverse_excursion=0.0,
                expiry_outcome="expired_without_entry",
                spread_adjusted_result=0.0,
                slippage_adjusted_result=0.0,
                signal_lifetime_seconds=(required_horizon_end - signal.published_at).total_seconds(),
                evaluated_at=evaluated_at,
            )
        active = tuple(item for item in future if item.timestamp >= entered.timestamp)
        buy = signal.direction == Direction.BUY
        favorable = max((item.high - entry if buy else entry - item.low) for item in active)
        adverse = max((entry - item.low if buy else item.high - entry) for item in active)
        first_tp = signal.take_profit_levels[0]
        tp_candle = next((item for item in active if item.high >= first_tp), None) if buy else next((item for item in active if item.low <= first_tp), None)
        sl_candle = next((item for item in active if item.low <= signal.stop_loss), None) if buy else next((item for item in active if item.high >= signal.stop_loss), None)
        ordering = "none"
        if tp_candle and sl_candle:
            ordering = "tp_before_sl" if tp_candle.timestamp < sl_candle.timestamp else "sl_before_tp"
        elif tp_candle:
            ordering = "tp_only"
        elif sl_candle:
            ordering = "sl_only"
        exit_price = active[-1].close
        if ordering in {"tp_before_sl", "tp_only"}:
            exit_price = first_tp
        elif ordering in {"sl_before_tp", "sl_only"}:
            exit_price = signal.stop_loss
        gross = exit_price - entry if buy else entry - exit_price
        net = gross - spread
        slippage_adjusted = net - self.configured_slippage
        risk = abs(entry - signal.stop_loss)
        realized_rr = slippage_adjusted / risk if risk else None
        realized_direction = Direction.BUY if gross > 0 else Direction.SELL if gross < 0 else Direction.NEUTRAL
        return DetailedSignalOutcome(
            outcome_id=uuid5(NAMESPACE_URL, f"ten:detailed-outcome:{signal.signal_id}"),
            signal_id=signal.signal_id,
            status="resolved",
            evaluation_horizon_complete=True,
            realized_direction=realized_direction,
            realized_return=gross,
            maximum_favorable_excursion=favorable,
            maximum_adverse_excursion=adverse,
            tp1_result="hit" if tp_candle else "not_hit",
            tp2_result=(
                "hit"
                if len(signal.take_profit_levels) > 1
                and any((item.high >= signal.take_profit_levels[1] if buy else item.low <= signal.take_profit_levels[1]) for item in active)
                else "not_hit"
            ),
            stop_loss_result="hit" if sl_candle else "not_hit",
            tp_sl_ordering=ordering,
            time_to_entry_seconds=(entered.timestamp - signal.published_at).total_seconds(),
            time_to_tp_seconds=(tp_candle.timestamp - entered.timestamp).total_seconds() if tp_candle else None,
            time_to_sl_seconds=(sl_candle.timestamp - entered.timestamp).total_seconds() if sl_candle else None,
            expiry_outcome="resolved_at_horizon",
            spread_adjusted_result=net,
            slippage_adjusted_result=slippage_adjusted,
            realized_risk_to_reward=realized_rr,
            signal_lifetime_seconds=(required_horizon_end - signal.published_at).total_seconds(),
            evaluated_at=evaluated_at,
        )
