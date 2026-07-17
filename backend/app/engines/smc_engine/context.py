"""Validated normalized-candle context shared by SMC analyzers."""

from dataclasses import dataclass
from statistics import fmean

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.market_data_engine.models import canonical_symbol

from .config import SMCConfig
from .exceptions import InvalidSMCInput


@dataclass(frozen=True, slots=True)
class CandleContext:
    candles: tuple[Candle, ...]
    symbol: str
    timeframe: Timeframe
    true_ranges: tuple[float, ...]
    average_true_range: float
    minimum_quality: float
    average_quality: float
    degraded: bool
    boundary: str

    @classmethod
    def build(cls, candles: list[Candle], config: SMCConfig) -> "CandleContext":
        if not candles:
            raise InvalidSMCInput("SMC analysis requires normalized candles")
        symbol = canonical_symbol(candles[0].symbol)
        timeframe = candles[0].timeframe
        seen: set[object] = set()
        previous = None
        true_ranges: list[float] = []
        qualities: list[float] = []
        for candle in candles:
            if canonical_symbol(candle.symbol) != symbol or candle.timeframe != timeframe:
                raise InvalidSMCInput("all candles must share symbol and timeframe")
            if candle.timestamp in seen:
                raise InvalidSMCInput(f"duplicate candle at {candle.timestamp.isoformat()}")
            if previous is not None and candle.timestamp <= previous.timestamp:
                raise InvalidSMCInput("candles must be strictly chronological")
            seen.add(candle.timestamp)
            previous_close = previous.close if previous is not None else candle.open
            true_ranges.append(max(candle.high - candle.low, abs(candle.high - previous_close), abs(candle.low - previous_close)))
            qualities.append(candle.quality_score)
            previous = candle
        boundary = f"{symbol}:{timeframe.value}:{candles[-1].timestamp.isoformat()}:{len(candles)}"
        minimum_quality = min(qualities)
        return cls(
            candles=tuple(candles),
            symbol=symbol,
            timeframe=timeframe,
            true_ranges=tuple(true_ranges),
            average_true_range=fmean(true_ranges),
            minimum_quality=minimum_quality,
            average_quality=fmean(qualities),
            degraded=minimum_quality < config.processing.minimum_input_quality,
            boundary=boundary,
        )

    def atr_at(self, index: int, period: int = 14) -> float:
        start = max(0, index - period + 1)
        values = self.true_ranges[start : index + 1]
        return fmean(values) if values else 0.0

    def candle_id(self, index: int) -> str:
        candle = self.candles[index]
        return f"{self.symbol}:{self.timeframe.value}:{candle.timestamp.isoformat()}"
