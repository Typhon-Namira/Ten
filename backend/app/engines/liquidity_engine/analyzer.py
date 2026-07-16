from abc import ABC

from backend.app.engines.common import AnalysisEngine
from backend.app.engines.market_data_engine import Candle

from .config import LiquidityConfig
from .models import LiquidityLevel, LiquidityResult, LiquiditySide


class LiquidityAnalyzer(AnalysisEngine[list[Candle], LiquidityResult], ABC):
    """Contract for liquidity-pool and sweep analyzers."""


class BaselineLiquidityAnalyzer(LiquidityAnalyzer):
    name = "liquidity"
    version = "1.0.0"

    def __init__(self, config: LiquidityConfig | None = None) -> None:
        self.config = config or LiquidityConfig()

    def analyze(self, data: list[Candle]) -> LiquidityResult:
        if not data:
            return LiquidityResult(observations=["No candles supplied."])
        candles = sorted(data, key=lambda item: item.timestamp)
        latest = candles[-1]
        levels: list[LiquidityLevel] = []
        for side, attribute in ((LiquiditySide.BUY_SIDE, "high"), (LiquiditySide.SELL_SIDE, "low")):
            grouped: list[list[Candle]] = []
            for candle in candles[:-1]:
                price = getattr(candle, attribute)
                group = next((g for g in grouped if abs(getattr(g[0], attribute) - price) / price <= self.config.equal_level_tolerance), None)
                if group is None:
                    grouped.append([candle])
                else:
                    group.append(candle)
            for group in grouped:
                if len(group) >= 2:
                    price = sum(getattr(item, attribute) for item in group) / len(group)
                    swept = latest.high > price and latest.close < price if side == LiquiditySide.BUY_SIDE else latest.low < price and latest.close > price
                    levels.append(LiquidityLevel(side=side, price=price, touches=len(group), swept=swept, last_seen=group[-1].timestamp))
        levels = sorted(levels, key=lambda item: item.last_seen, reverse=True)[: self.config.max_levels]
        buys = [item.price for item in levels if item.side == LiquiditySide.BUY_SIDE and item.price >= latest.close]
        sells = [item.price for item in levels if item.side == LiquiditySide.SELL_SIDE and item.price <= latest.close]
        return LiquidityResult(levels=levels, nearest_buy_side=min(buys, default=None), nearest_sell_side=max(sells, default=None), active_session=_session(latest.timestamp.hour))


def _session(hour: int) -> str:
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "new_york"
    return "closed"
