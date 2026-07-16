from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    version: str


class MarketStatusResponse(BaseModel):
    symbol: str
    session: str
    is_open: bool
    checked_at: datetime
    note: str

