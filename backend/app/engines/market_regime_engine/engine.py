from abc import ABC, abstractmethod

from backend.app.engines.market_data_engine import Candle

from .analyzer import BaselineMarketRegimeAnalyzer
from .config import MarketRegimeConfig
from .contracts import MarketRegimeContext
from .models import MarketRegimeSnapshot


class MarketRegimeEngine(ABC):
    name = "market_regime"
    version = "1.0.0"
    compatibility_version = "1.0"

    @abstractmethod
    def analyze(self, candles: list[Candle]) -> MarketRegimeSnapshot: ...


class BaselineMarketRegimeEngine(MarketRegimeEngine):
    def __init__(self, config: MarketRegimeConfig | None = None) -> None:
        self.analyzer = BaselineMarketRegimeAnalyzer(config)

    def analyze(self, candles: list[Candle]) -> MarketRegimeSnapshot:
        if not candles:
            raise ValueError("market regime requires candles")
        return self.analyzer.analyze_snapshot(MarketRegimeContext(tuple(candles)))
