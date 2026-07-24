from dataclasses import dataclass
from typing import Protocol

from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.engines.smc_engine.liquidity_contract import SMCLiquidityContext

from .models import ProfileAnchor, VolumeSourceType


@dataclass(frozen=True)
class VolumeProfileContext:
    candles: tuple[Candle, ...]
    volume_source_type: VolumeSourceType = VolumeSourceType.UNKNOWN
    instrument: str = "unknown"
    tick_size: float | None = None
    smc: SMCLiquidityContext | None = None
    liquidity_source_ids: tuple[str, ...] = ()
    anchors: tuple[ProfileAnchor, ...] = ()
    requested_timeframe: Timeframe | None = None


class VolumeProfileReader(Protocol):
    async def profile_context(self, symbol: str, timeframe: Timeframe, at: object) -> object: ...


class LiquidityProfileEvidenceReader(Protocol):
    async def volume_profile_evidence(self, symbol: str, timeframe: Timeframe, at: object) -> tuple[str, ...]: ...
