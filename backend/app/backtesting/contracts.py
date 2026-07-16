from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from backend.app.engines.market_data_engine import Candle
from backend.app.engines.signal_engine import Signal


class BacktestMetrics(BaseModel):
    sample_size: int = Field(ge=0)
    win_rate: float = Field(ge=0, le=1)
    expectancy_r: float
    maximum_drawdown_r: float = Field(ge=0)


class Backtester(ABC):
    @abstractmethod
    def evaluate(self, candles: list[Candle], signals: list[Signal]) -> BacktestMetrics:
        """Evaluate immutable historical scenarios without future-data leakage."""

