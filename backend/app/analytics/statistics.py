from backend.app.engines.market_data_engine import Candle


def average_true_range(candles: list[Candle], period: int = 14) -> float:
    """Calculate a simple true-range average over chronologically ordered candles."""

    ordered = sorted(candles, key=lambda item: item.timestamp)
    if len(ordered) < 2:
        return 0.0
    ranges = [max(current.high - current.low, abs(current.high - previous.close), abs(current.low - previous.close)) for previous, current in zip(ordered, ordered[1:], strict=False)]
    selected = ranges[-period:]
    return sum(selected) / len(selected)

