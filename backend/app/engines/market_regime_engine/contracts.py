from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from backend.app.engines.market_data_engine import Candle, Timeframe

from .models import CrossSessionRegimeState, MarketRegimeEvidence, MarketRegimeSnapshot, MultiTimeframeRegimeState


@dataclass(frozen=True)
class MarketRegimeContext:
    candles: tuple[Candle, ...]
    evidence: tuple[MarketRegimeEvidence, ...] = ()
    analysis_boundary: datetime | None = None
    missing_dependencies: tuple[str, ...] = ()
    failed_dependencies: tuple[str, ...] = ()
    multi_timeframe: MultiTimeframeRegimeState | None = None
    cross_session: CrossSessionRegimeState | None = None


class RegimeSnapshotReader(Protocol):
    async def state(self, symbol: str, timeframe: Timeframe, at: datetime | None = None) -> object | None: ...


class MarketRegimeReader(Protocol):
    async def state(self, symbol: str, timeframe: Timeframe, at: datetime | None = None) -> MarketRegimeSnapshot | None: ...
