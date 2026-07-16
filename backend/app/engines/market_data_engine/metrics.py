"""Informational market metrics derived only from normalized raw data."""

from datetime import UTC, datetime
from math import sqrt

from .models import Candle, MarketMetrics, Tick


def calculate_metrics(candles: list[Candle], ticks: list[Tick] | None = None, latency_ms: float = 0, now: datetime | None = None) -> MarketMetrics:
    if not candles:
        raise ValueError("at least one candle is required")
    ordered = sorted(candles, key=lambda item: item.timestamp)
    latest = ordered[-1]
    true_ranges: list[float] = []
    returns: list[float] = []
    for index, candle in enumerate(ordered):
        previous_close = ordered[index - 1].close if index else candle.open
        true_ranges.append(max(candle.high - candle.low, abs(candle.high - previous_close), abs(candle.low - previous_close)))
        if index and previous_close:
            returns.append((candle.close - previous_close) / previous_close)
    mean_return = sum(returns) / len(returns) if returns else 0
    volatility = sqrt(sum((item - mean_return) ** 2 for item in returns) / len(returns)) if returns else 0
    day = latest.timestamp.date()
    daily = [item for item in ordered if item.timestamp.date() == day]
    elapsed = max((latest.timestamp - ordered[0].timestamp).total_seconds(), 1)
    tick_items = ticks or []
    tick_frequency = len(tick_items) / max((tick_items[-1].timestamp - tick_items[0].timestamp).total_seconds(), 1) if len(tick_items) > 1 else 0
    current = now or datetime.now(UTC)
    return MarketMetrics(
        symbol=latest.symbol,
        timeframe=latest.timeframe,
        atr=sum(true_ranges[-14:]) / min(len(true_ranges), 14),
        current_spread=latest.spread,
        average_spread=sum(item.spread for item in ordered) / len(ordered),
        daily_range=max(item.high for item in daily) - min(item.low for item in daily),
        session_range=max(item.high for item in ordered) - min(item.low for item in ordered),
        rolling_volatility=volatility,
        price_velocity=(latest.close - ordered[0].open) / elapsed,
        tick_frequency=tick_frequency,
        data_freshness_seconds=max(0, (current - latest.timestamp).total_seconds()),
        latency_ms=latency_ms,
    )
