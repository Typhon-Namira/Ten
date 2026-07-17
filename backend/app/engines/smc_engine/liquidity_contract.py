"""Provider-neutral bridge to the dedicated Liquidity Engine."""

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from backend.app.engines.market_data_engine import Timeframe


class ExternalLiquidityEvidence(BaseModel):
    """Optional immutable sweep metadata supplied by the Liquidity Engine."""

    model_config = ConfigDict(frozen=True)
    id: str
    symbol: str
    timeframe: Timeframe
    event_type: str
    price: float = Field(gt=0)
    occurred_at: datetime
    available_at: datetime
    confidence_score: float = Field(ge=0, le=100)
    source_version: str


class LiquidityFeatureReader(Protocol):
    """Read-only contract; implementations remain owned by liquidity_engine."""

    async def evidence(self, symbol: str, timeframe: Timeframe, at: datetime) -> tuple[ExternalLiquidityEvidence, ...]: ...
