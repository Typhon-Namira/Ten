"""Provider-neutral bridge to the dedicated Liquidity Engine."""

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from backend.app.engines.market_data_engine import Timeframe


class SMCLiquidityLevel(BaseModel):
    """Provider-neutral structural level exposed to Liquidity."""

    model_config = ConfigDict(frozen=True)
    id: str
    symbol: str
    timeframe: Timeframe
    kind: str
    scope: str
    price: float = Field(gt=0)
    occurred_at: datetime
    available_at: datetime
    confidence_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)


class SMCLiquidityContext(BaseModel):
    """Replay-valid SMC evidence without implementation imports."""

    model_config = ConfigDict(frozen=True)
    symbol: str
    timeframe: Timeframe
    analyzed_through: datetime
    structure_direction: str
    levels: tuple[SMCLiquidityLevel, ...] = ()
    protected_level_ids: tuple[str, ...] = ()
    structural_event_ids: tuple[str, ...] = ()
    configuration_version: str
    engine_version: str


class SMCLiquidityReader(Protocol):
    async def liquidity_context(self, symbol: str, timeframe: Timeframe, at: datetime) -> SMCLiquidityContext: ...


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
