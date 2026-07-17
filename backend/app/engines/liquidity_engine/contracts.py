from dataclasses import dataclass

from backend.app.engines.market_data_engine import Candle
from backend.app.engines.smc_engine.liquidity_contract import SMCLiquidityContext


@dataclass(frozen=True)
class LiquidityContext:
    candles: tuple[Candle, ...]
    smc: SMCLiquidityContext | None = None
