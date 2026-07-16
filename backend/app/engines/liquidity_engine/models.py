from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class LiquiditySide(StrEnum):
    BUY_SIDE = "buy_side"
    SELL_SIDE = "sell_side"


class LiquidityLevel(BaseModel):
    side: LiquiditySide
    price: float
    touches: int = Field(ge=1)
    swept: bool = False
    last_seen: datetime


class LiquidityResult(BaseModel):
    levels: list[LiquidityLevel] = Field(default_factory=list)
    nearest_buy_side: float | None = None
    nearest_sell_side: float | None = None
    active_session: str = "unknown"
    observations: list[str] = Field(default_factory=list)

