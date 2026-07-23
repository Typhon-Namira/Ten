"""Horizon-complete, spread-aware forecast outcome evaluation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from collections.abc import Callable
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from .models import ForecastOutcome, HorizonPrediction, OutcomeStatus, QuantForecastResult
from .repository import QuantForecastRepository


class OutcomeCandle(Protocol):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    spread: float | None


class ForecastOutcomeEvaluator:
    """Evaluates only candles fully closed inside the requested future horizon."""

    def evaluate(
        self,
        forecast: QuantForecastResult,
        prediction: HorizonPrediction,
        candles: list[OutcomeCandle],
        *,
        evaluated_at: datetime,
    ) -> ForecastOutcome:
        horizon_end = forecast.point_in_time + timedelta(seconds=prediction.horizon.duration_seconds)
        outcome_id = uuid5(NAMESPACE_URL, f"ten:outcome:{forecast.result_id}:{prediction.horizon.horizon_id}")
        if evaluated_at.tzinfo is None:
            raise ValueError("outcome evaluation timestamp must be timezone-aware")
        if evaluated_at < horizon_end:
            return ForecastOutcome(
                outcome_id=outcome_id,
                forecast_result_id=forecast.result_id,
                horizon_id=prediction.horizon.horizon_id,
                status=OutcomeStatus.PENDING,
                evaluated_at=evaluated_at,
                candle_count=0,
                reason_codes=("horizon_not_complete",),
            )
        duration = 60 if prediction.horizon.timeframe == "M1" else 300
        eligible = sorted(
            [
                candle
                for candle in candles
                if forecast.point_in_time <= candle.timestamp
                and candle.timestamp + timedelta(seconds=duration) <= horizon_end
            ],
            key=lambda candle: candle.timestamp,
        )
        if len(eligible) < prediction.horizon.candle_count:
            return ForecastOutcome(
                outcome_id=outcome_id,
                forecast_result_id=forecast.result_id,
                horizon_id=prediction.horizon.horizon_id,
                status=OutcomeStatus.MISSING_DATA,
                evaluated_at=evaluated_at,
                candle_count=len(eligible),
                reason_codes=("incomplete_horizon_candles",),
            )
        eligible = eligible[: prediction.horizon.candle_count]
        entry = prediction.reference_price
        direction = "buy" if prediction.buy_probability >= max(prediction.sell_probability, prediction.neutral_probability) else "sell" if prediction.sell_probability >= prediction.neutral_probability else "neutral"
        final_return = eligible[-1].close / entry - 1
        realized_return = -final_return if direction == "sell" else final_return
        highs = [(candle.high / entry - 1) for candle in eligible]
        lows = [(candle.low / entry - 1) for candle in eligible]
        mfe = max(highs) if direction != "sell" else max(-value for value in lows)
        mae = max(-value for value in lows) if direction != "sell" else max(highs)
        sign = -1 if direction == "sell" else 1
        tp1 = entry * (1 + sign * prediction.expected_base_movement * 0.5)
        tp2 = entry * (1 + sign * prediction.expected_base_movement)
        stop = entry * (1 - sign * prediction.expected_mae)
        tp1_hit = tp2_hit = stop_hit = False
        stop_before_tp = False
        target_seen = False
        for candle in eligible:
            this_tp1 = candle.low <= tp1 if direction == "sell" else candle.high >= tp1
            this_tp2 = candle.low <= tp2 if direction == "sell" else candle.high >= tp2
            this_stop = candle.high >= stop if direction == "sell" else candle.low <= stop
            # Intracandle order is unknowable from OHLC; count simultaneous stop/target as stop-first.
            if this_stop and not target_seen:
                stop_before_tp = True
            stop_hit = stop_hit or this_stop
            tp1_hit = tp1_hit or this_tp1
            tp2_hit = tp2_hit or this_tp2
            target_seen = target_seen or this_tp1 or this_tp2
        spread = max((float(candle.spread or 0.0) for candle in eligible), default=0.0)
        return ForecastOutcome(
            outcome_id=outcome_id,
            forecast_result_id=forecast.result_id,
            horizon_id=prediction.horizon.horizon_id,
            status=OutcomeStatus.VALID,
            evaluated_at=evaluated_at.astimezone(UTC),
            realized_return=realized_return,
            realized_direction="buy" if final_return > 0 else "sell" if final_return < 0 else "neutral",
            maximum_favorable_excursion=max(0.0, mfe),
            maximum_adverse_excursion=max(0.0, mae),
            tp1_hit=tp1_hit,
            tp2_hit=tp2_hit,
            stop_loss_hit=stop_hit,
            stop_before_tp=stop_before_tp,
            spread_adjusted_return=realized_return - spread / entry,
            candle_count=len(eligible),
        )


class ForecastOutcomeWorker:
    """Evaluates and persists only horizons complete at the worker's current time."""

    def __init__(
        self,
        repository: QuantForecastRepository,
        market_data: Any,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.market_data = market_data
        self.clock = clock or (lambda: datetime.now(UTC))
        self.evaluator = ForecastOutcomeEvaluator()

    async def evaluate(self, forecast: QuantForecastResult) -> tuple[ForecastOutcome, ...]:
        from backend.app.engines.market_data_engine import Timeframe

        now = self.clock()
        outcomes: list[ForecastOutcome] = []
        for prediction in forecast.predictions:
            horizon_end = forecast.point_in_time + timedelta(seconds=prediction.horizon.duration_seconds)
            candles = []
            if now >= horizon_end:
                candles = await self.market_data.history(
                    forecast.instrument,
                    Timeframe(prediction.horizon.timeframe),
                    start=forecast.point_in_time,
                    end=horizon_end,
                    limit=prediction.horizon.candle_count,
                    refresh=False,
                )
            outcome = self.evaluator.evaluate(forecast, prediction, candles, evaluated_at=now)
            await self.repository.save_outcome(outcome)
            outcomes.append(outcome)
        return tuple(outcomes)
