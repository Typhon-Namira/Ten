"""Baseline price-action implementation behind the replaceable SMC port."""

from abc import ABC

from backend.app.engines.common import AnalysisEngine
from backend.app.engines.market_data_engine import Candle

from .config import SMCConfig
from .models import Bias, FairValueGap, SMCResult, StructureEvent


class SMCAnalyzer(AnalysisEngine[list[Candle], SMCResult], ABC):
    """Contract for SMC/ICT analyzers."""


class BaselineSMCAnalyzer(SMCAnalyzer):
    name = "smc"
    version = "1.0.0"

    def __init__(self, config: SMCConfig | None = None) -> None:
        self.config = config or SMCConfig()

    def analyze(self, data: list[Candle]) -> SMCResult:
        if len(data) < 3:
            return SMCResult(observations=["At least three candles are required."])
        candles = sorted(data, key=lambda item: item.timestamp)
        first, last = candles[0], candles[-1]
        bias = Bias.BULLISH if last.close > first.close else Bias.BEARISH if last.close < first.close else Bias.NEUTRAL
        events: list[StructureEvent] = []
        prior_high = max(item.high for item in candles[:-1])
        prior_low = min(item.low for item in candles[:-1])
        if last.close > prior_high:
            events.append(StructureEvent(kind="BOS", direction=Bias.BULLISH, price=prior_high, timestamp=last.timestamp, confidence=0.7))
        elif last.close < prior_low:
            events.append(StructureEvent(kind="BOS", direction=Bias.BEARISH, price=prior_low, timestamp=last.timestamp, confidence=0.7))
        gaps: list[FairValueGap] = []
        for left, _, right in zip(candles, candles[1:], candles[2:], strict=False):
            if right.low > left.high:
                gaps.append(FairValueGap(direction=Bias.BULLISH, low=left.high, high=right.low, timestamp=right.timestamp))
            elif right.high < left.low:
                gaps.append(FairValueGap(direction=Bias.BEARISH, low=right.high, high=left.low, timestamp=right.timestamp))
        dealing_low = min(item.low for item in candles)
        dealing_high = max(item.high for item in candles)
        midpoint = (dealing_high + dealing_low) / 2
        position = "premium" if last.close > midpoint else "discount" if last.close < midpoint else "equilibrium"
        return SMCResult(bias=bias, structure_events=events, fair_value_gaps=gaps[-20:], premium_discount_position=position)

