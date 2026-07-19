from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    version: str


class MarketStatusResponse(BaseModel):
    symbol: str
    session: str | None
    is_open: bool
    checked_at: datetime
    note: str
    market_status: str
    closure_reason: str | None = None
    next_expected_open_at: datetime | None = None
    server_time_utc: datetime
    latest_candle_at: datetime | None = None
    latest_candle_age_seconds: float | None = None
    provider_status: str

