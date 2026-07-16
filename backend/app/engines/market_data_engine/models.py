"""Canonical market-data value objects."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Timeframe(StrEnum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


class MarketSession(StrEnum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    CLOSED = "closed"


class Candle(BaseModel):
    symbol: str = "XAU/USD"
    timeframe: Timeframe
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_ohlc(self) -> "Candle":
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("high/low must contain open and close")
        if self.low > self.high:
            raise ValueError("low cannot exceed high")
        return self


class Tick(BaseModel):
    symbol: str = "XAU/USD"
    timestamp: datetime
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    volume: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_spread(self) -> "Tick":
        if self.ask < self.bid:
            raise ValueError("ask cannot be below bid")
        return self

